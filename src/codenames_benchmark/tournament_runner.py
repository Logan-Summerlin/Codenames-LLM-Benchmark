"""Shared OpenRouter tournament runner helpers.

This module centralizes the launcher plumbing that used to live in several
script-only entrypoints: output directory creation, JSON artifact writes,
model-preset selection, schedule filtering, transcript writes, and Elo/record
bookkeeping.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import urllib.request

from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.runner import run_game
from codenames_benchmark.tournament import (
    ModelSpec,
    TournamentGame,
    OPENROUTER_CODENAMES_MODELS,
    schedule_double_round_robin,
    schedule_limited_coverage,
    schedule_single_round_robin,
)
from codenames_benchmark.transcript import write_transcript

TOP4_MODEL_SLUGS = [
    "meta-llama/llama-4-maverick",
    "openai/gpt-5.4-nano",
    "microsoft/phi-4",
    "mistralai/mistral-small-3.2-24b-instruct",
]


@dataclass(frozen=True)
class TournamentSelection:
    """Concrete model set and schedule for a tournament run."""

    models: list[ModelSpec]
    schedule: list[TournamentGame]


def default_output_dir(*, root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / "runs" / f"{prefix}-{stamp}"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _live_provider_order_for_model(model_slug: str, *, timeout_seconds: float = 20.0) -> list[str]:
    """Fetch the current active OpenRouter provider order for a model."""
    url = f"https://openrouter.ai/api/v1/models/{model_slug}/endpoints"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        data = json.load(response)
    endpoints = data.get("data", {}).get("endpoints", []) if isinstance(data, dict) else []
    ranked: list[tuple[float, str]] = []
    seen: set[str] = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider_name = endpoint.get("provider_name")
        if not isinstance(provider_name, str) or not provider_name or provider_name in seen:
            continue
        status = endpoint.get("status", 0)
        if status not in (0, None):
            continue
        uptime = endpoint.get("uptime_last_1d", 0) or 0
        try:
            uptime_value = float(uptime)
        except (TypeError, ValueError):
            uptime_value = 0.0
        ranked.append((uptime_value, provider_name))
        seen.add(provider_name)
    ranked.sort(key=lambda item: (-item[0], item[1]))
    order = [provider for _, provider in ranked]
    if not order:
        return []
    return order


def provider_order_payload(models: Sequence[ModelSpec]) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    for model in models:
        try:
            live_order = _live_provider_order_for_model(model.slug)
        except Exception:
            live_order = []
        if model.provider in live_order:
            live_order = [model.provider, *[provider for provider in live_order if provider != model.provider]]
        payload[model.slug] = live_order or [model.provider]
    return payload


def reasoning_effort_payload(models: Sequence[ModelSpec]) -> dict[str, str]:
    return {model.slug: model.reasoning_effort for model in models if model.reasoning_effort}


def model_lookup(models: Sequence[ModelSpec]) -> dict[str, str]:
    return {model.slug: model.label for model in models}


def select_models(models: Sequence[ModelSpec], preset: str) -> list[ModelSpec]:
    if preset == "full-field":
        return list(models)
    if preset == "top4":
        by_slug = {model.slug: model for model in models}
        missing = [slug for slug in TOP4_MODEL_SLUGS if slug not in by_slug]
        if missing:
            raise ValueError(f"top4 preset is missing required models: {missing}")
        return [by_slug[slug] for slug in TOP4_MODEL_SLUGS]
    raise ValueError(f"unknown model preset: {preset}")


def schedule_for_mode(models: Sequence[ModelSpec], *, schedule_mode: str, seed_prefix: str) -> list[TournamentGame]:
    slugs = [model.slug for model in models]
    if schedule_mode == "single-round-robin":
        return schedule_single_round_robin(slugs, seed_prefix=seed_prefix)
    if schedule_mode == "double-round-robin":
        return schedule_double_round_robin(slugs, seed_prefix=seed_prefix)
    if schedule_mode == "limited-coverage":
        return schedule_limited_coverage(slugs, seed_prefix=seed_prefix)
    raise ValueError(f"unknown schedule mode: {schedule_mode}")


def chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def record_tally(models: Sequence[str]) -> dict[str, dict[str, int]]:
    return {model: {"wins": 0, "losses": 0, "ties": 0, "games": 0} for model in models}


def update_record_tally(records: dict[str, dict[str, int]], red_model: str, blue_model: str, winner_model: str | None) -> None:
    records[red_model]["games"] += 1
    records[blue_model]["games"] += 1
    if winner_model == red_model:
        records[red_model]["wins"] += 1
        records[blue_model]["losses"] += 1
    elif winner_model == blue_model:
        records[blue_model]["wins"] += 1
        records[red_model]["losses"] += 1
    else:
        records[red_model]["ties"] += 1
        records[blue_model]["ties"] += 1


def record_rankings(
    records: dict[str, dict[str, int]],
    ratings: EloRatingSystem,
    model_order: Sequence[str],
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in model_order:
        record = records[model]
        games = record["games"]
        win_rate = None if games == 0 else round((record["wins"] + 0.5 * record["ties"]) / games, 4)
        rows.append(
            {
                "model": model,
                "label": (labels or {}).get(model, model),
                "rating": round(ratings.rating_for(model), 2),
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "games": games,
                "win_rate": win_rate,
            }
        )
    return sorted(rows, key=lambda row: (-row["wins"], row["losses"], -row["ties"], -row["rating"], row["model"]))


def elo_rankings(ratings: EloRatingSystem, records: dict[str, dict[str, int]], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for standing in ratings.standings():
        model = standing["model"]
        record = records[model]
        rows.append(
            {
                "rank": standing["rank"],
                "model": model,
                "label": labels.get(model, model),
                "rating": standing["rating"],
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "games": record["games"],
            }
        )
    return rows


def winner_model_from_record(record: Any, red_model: str, blue_model: str) -> str | None:
    if record.winner == Team.RED.value:
        return red_model
    if record.winner == Team.BLUE.value:
        return blue_model
    return None


def play_game(game: TournamentGame, *, output_dir: Path, max_turns: int, labels: dict[str, str], client: OpenRouterClient) -> dict[str, Any]:
    game_dir = output_dir / f"game-{game.game_number:03d}"
    transcript_path = game_dir / "transcript.json"
    summary_path = game_dir / "summary.json"
    red = LLMTeam(name=labels.get(game.red_model, game.red_model), team=Team.RED, model=game.red_model, client=client)
    blue = LLMTeam(name=labels.get(game.blue_model, game.blue_model), team=Team.BLUE, model=game.blue_model, client=client)
    result: dict[str, Any] = {
        "game": game.to_dict(),
        "transcript": str(transcript_path),
        "summary_path": str(summary_path),
    }
    try:
        record = run_game(red, blue, seed=game.seed, max_turns=max_turns)
        write_transcript(record, transcript_path)
        result.update(
            {
                "status": "game_complete",
                "winner_model": winner_model_from_record(record, game.red_model, game.blue_model),
                "winner_color": record.winner,
                "terminal": record.terminal,
                "reason": record.reason,
                "public_events": len(record.public_events),
                "private_events": len(record.private_events),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive artifact write
        result.update({"status": "game_error", "error": str(exc)[:1000]})
    write_json(summary_path, result)
    return result


def write_static_artifacts(*, output_dir: Path, models: Sequence[ModelSpec], schedule: Sequence[TournamentGame], manifest: dict[str, Any]) -> None:
    write_json(output_dir / "models.json", [model.to_dict() for model in models])
    write_json(output_dir / "provider_order.json", provider_order_payload(models))
    write_json(output_dir / "reasoning_effort.json", reasoning_effort_payload(models))
    write_json(output_dir / "schedule.json", [game.to_dict() for game in schedule])
    write_json(output_dir / "manifest.json", manifest)


def tournament_manifest(
    *,
    benchmark: str,
    models: Sequence[ModelSpec],
    game_count: int,
    max_turns: int,
    seed_prefix: str,
    elo_initial: float,
    elo_k: float,
    schedule_mode: str,
    round_size: int,
    max_tokens: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "model_count": len(models),
        "game_count": game_count,
        "max_turns": max_turns,
        "max_tokens": max_tokens,
        "elo_initial": elo_initial,
        "elo_k": elo_k,
        "seed_prefix": seed_prefix,
        "schedule_mode": schedule_mode,
        "round_size": round_size,
        "models": [model.to_dict() for model in models],
        "provider_policy": "provider.order uses the live OpenRouter active-provider allowlist and allow_fallbacks is enabled when multiple routes exist",
        "reasoning_policy": "reasoning.effort is set per model when present in models.json",
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


@dataclass(frozen=True)
class TournamentRunResult:
    status: str
    output_dir: str
    scheduled_games: int
    completed_or_attempted_games: int
    payload: dict[str, Any]


def run_tournament(
    *,
    root: Path,
    model_preset: str,
    schedule_mode: str,
    seed_prefix: str,
    max_turns: int,
    elo_initial: float,
    elo_k: float,
    round_size: int | None,
    workers: int | None,
    start_game: int,
    limit_games: int | None,
    dry_run: bool,
    require_api_key: bool,
    output_dir: Path | None = None,
    benchmark: str = "codenames-openrouter-tournament",
    extra_manifest_fields: dict[str, Any] | None = None,
) -> TournamentRunResult:
    models = select_models(OPENROUTER_CODENAMES_MODELS, model_preset)
    schedule = schedule_for_mode(models, schedule_mode=schedule_mode, seed_prefix=seed_prefix)
    scheduled_games = [game for game in schedule if game.game_number >= start_game]
    if limit_games is not None:
        scheduled_games = scheduled_games[:limit_games]

    resolved_output_dir = output_dir or default_output_dir(root=root, prefix=benchmark)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    labels = model_lookup(models)
    write_static_artifacts(
        output_dir=resolved_output_dir,
        models=models,
        schedule=schedule,
        manifest=tournament_manifest(
            benchmark=benchmark,
            models=models,
            game_count=len(schedule),
            max_turns=max_turns,
            seed_prefix=seed_prefix,
            elo_initial=elo_initial,
            elo_k=elo_k,
            schedule_mode=schedule_mode,
            round_size=round_size or (len(models) // 2 if schedule_mode == "single-round-robin" else 1),
            max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "10000")),
            extra_fields=extra_manifest_fields,
        ),
    )

    api_key_present = bool(os.environ.get("OPENROUTER_API_KEY"))
    if dry_run or not api_key_present:
        status = "dry_run" if dry_run else "configured_no_api_key"
        if require_api_key and not api_key_present:
            write_json(
                resolved_output_dir / "run_state.json",
                {
                    "status": "missing_api_key",
                    "output_dir": str(resolved_output_dir),
                    "scheduled_games": len(schedule),
                    "selected_games": [game.to_dict() for game in scheduled_games],
                    "models": [model.to_dict() for model in models],
                    "schedule_mode": schedule_mode,
                },
            )
            return TournamentRunResult(
                status="missing_api_key",
                output_dir=str(resolved_output_dir),
                scheduled_games=len(schedule),
                completed_or_attempted_games=0,
                payload={"status": "missing_api_key", "output_dir": str(resolved_output_dir)},
            )

        write_json(
            resolved_output_dir / "run_state.json",
            {
                "status": status,
                "output_dir": str(resolved_output_dir),
                "scheduled_games": len(schedule),
                "selected_games": [game.to_dict() for game in scheduled_games],
                "models": [model.to_dict() for model in models],
                "schedule_mode": schedule_mode,
            },
        )
        payload = {
            "status": status,
            "output_dir": str(resolved_output_dir),
            "scheduled_games": len(schedule),
            "selected_games": len(scheduled_games),
        }
        return TournamentRunResult(
            status=status,
            output_dir=str(resolved_output_dir),
            scheduled_games=len(schedule),
            completed_or_attempted_games=len(scheduled_games),
            payload=payload,
        )

    old_provider_order = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
    old_reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
    os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = json.dumps(provider_order_payload(models), sort_keys=True)
    os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = json.dumps(reasoning_effort_payload(models), sort_keys=True)
    try:
        client = OpenRouterClient()
        ratings = EloRatingSystem(models=[model.slug for model in models], initial=elo_initial, k=elo_k)
        records = record_tally([model.slug for model in models])
        round_summaries: list[dict[str, Any]] = []
        write_json(resolved_output_dir / "standings.json", ratings.standings())

        if schedule_mode == "single-round-robin":
            groups = chunked(scheduled_games, round_size or max(1, len(models) // 2))
        else:
            groups = chunked(scheduled_games, round_size or 1)

        for round_index, round_games in enumerate(groups, 1):
            round_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers or len(round_games), len(round_games))) as pool:
                futures = {pool.submit(play_game, game, output_dir=resolved_output_dir, max_turns=max_turns, labels=labels, client=client): game for game in round_games}
                for future in as_completed(futures):
                    game = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive artifact write
                        result = {
                            "game": game.to_dict(),
                            "status": "game_error",
                            "error": str(exc)[:1000],
                            "transcript": str(resolved_output_dir / f"game-{game.game_number:03d}" / "transcript.json"),
                            "summary_path": str(resolved_output_dir / f"game-{game.game_number:03d}" / "summary.json"),
                        }
                        write_json(Path(result["summary_path"]), result)
                    round_results.append(result)

            ordered_results = sorted(round_results, key=lambda item: item["game"]["game_number"])
            completed_round_games = [
                {
                    "red_model": result["game"]["red_model"],
                    "blue_model": result["game"]["blue_model"],
                    "winner_model": result.get("winner_model"),
                    "game_number": result["game"]["game_number"],
                    "round_index": round_index,
                }
                for result in ordered_results
                if result.get("status") == "game_complete"
            ]

            if schedule_mode == "single-round-robin":
                elo_entries = ratings.record_round(completed_round_games)
                round_elo_by_game = {entry["game_number"]: entry for entry in elo_entries}
            else:
                elo_entries = []
                round_elo_by_game = {}
                for result in ordered_results:
                    if result.get("status") != "game_complete":
                        continue
                    elo_entry = ratings.record_game(
                        red_model=result["game"]["red_model"],
                        blue_model=result["game"]["blue_model"],
                        winner_model=result.get("winner_model"),
                        game_number=result["game"]["game_number"],
                    )
                    elo_entries.append(elo_entry)
                    round_elo_by_game[result["game"]["game_number"]] = elo_entry

            for result in ordered_results:
                game_number = result["game"]["game_number"]
                if result.get("status") == "game_complete":
                    result["elo_update"] = round_elo_by_game[game_number]
                    update_record_tally(records, result["game"]["red_model"], result["game"]["blue_model"], result.get("winner_model"))
                write_json(Path(result["summary_path"]), result)
                append_jsonl(resolved_output_dir / "results.jsonl", result)
                print(
                    json.dumps(
                        {
                            "round": round_index,
                            "game_number": game_number,
                            "status": result.get("status"),
                            "winner": result.get("winner_model"),
                            "reason": result.get("reason"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            round_summary = {
                "round_index": round_index,
                "game_numbers": [result["game"]["game_number"] for result in ordered_results],
                "completed_games": len(completed_round_games),
                "error_games": sum(1 for result in ordered_results if result.get("status") != "game_complete"),
                "standings": ratings.standings(),
                "records": record_rankings(records, ratings, [model.slug for model in models], labels),
            }
            round_summaries.append(round_summary)
            write_json(resolved_output_dir / "round_summaries.json", round_summaries)
            write_json(resolved_output_dir / "standings.json", ratings.standings())
            write_json(resolved_output_dir / "elo_history.json", ratings.history)
            write_json(resolved_output_dir / "records.json", records)
            write_json(resolved_output_dir / "elo_rankings.json", elo_rankings(ratings, records, labels))
            write_json(resolved_output_dir / "record_rankings.json", record_rankings(records, ratings, [model.slug for model in models], labels))
            write_json(
                resolved_output_dir / "run_state.json",
                {
                    "status": "running" if round_index < len(groups) else "complete",
                    "output_dir": str(resolved_output_dir),
                    "scheduled_games": len(schedule),
                    "completed_or_attempted_games": sum(len(summary["game_numbers"]) for summary in round_summaries),
                    "round_index": round_index,
                    "round_count": len(groups),
                    "round_size": round_size or (len(models) // 2 if schedule_mode == "single-round-robin" else 1),
                    "standings": ratings.standings(),
                },
            )

        final_elo = elo_rankings(ratings, records, labels)
        final_records = record_rankings(records, ratings, [model.slug for model in models], labels)
        final_payload = {
            "status": "complete",
            "output_dir": str(resolved_output_dir),
            "scheduled_games": len(schedule),
            "completed_games": len(schedule),
            "round_count": len(groups),
            "round_size": round_size or (len(models) // 2 if schedule_mode == "single-round-robin" else 1),
            "elo_rankings": final_elo,
            "record_rankings": final_records,
        }
        write_json(
            resolved_output_dir / "run_state.json",
            {
                "status": "complete",
                "output_dir": str(resolved_output_dir),
                "scheduled_games": len(schedule),
                "completed_or_attempted_games": len(schedule),
                "round_count": len(groups),
                "round_size": round_size or (len(models) // 2 if schedule_mode == "single-round-robin" else 1),
                "elo_rankings": final_elo,
                "record_rankings": final_records,
            },
        )
        return TournamentRunResult(
            status="complete",
            output_dir=str(resolved_output_dir),
            scheduled_games=len(schedule),
            completed_or_attempted_games=len(schedule),
            payload=final_payload,
        )
    finally:
        if old_provider_order is None:
            os.environ.pop("OPENROUTER_PROVIDER_ORDER_JSON", None)
        else:
            os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = old_provider_order
        if old_reasoning_effort is None:
            os.environ.pop("OPENROUTER_REASONING_EFFORT_JSON", None)
        else:
            os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = old_reasoning_effort
