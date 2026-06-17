#!/usr/bin/env python3
"""Run a live OpenRouter single round robin across the full Codenames field.

The 14-model field is scheduled into 13 rounds with 7 concurrent matches per
round. Elo is updated only after each full round completes, so the standings at
the start of round N reflect the results of rounds 1..N-1 only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.runner import run_game
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS, schedule_single_round_robin
from codenames_benchmark.transcript import write_transcript

MODEL_BY_SLUG = {model.slug: model for model in OPENROUTER_CODENAMES_MODELS}
MODEL_SLUGS = [model.slug for model in OPENROUTER_CODENAMES_MODELS]
DEFAULT_ROUND_SIZE = 7


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return ROOT / "runs" / f"openrouter-single-round-robin-{stamp}"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def provider_order_json() -> str:
    return json.dumps({model.slug: [model.provider] for model in OPENROUTER_CODENAMES_MODELS}, sort_keys=True)


def reasoning_effort_json() -> str:
    return json.dumps({model.slug: model.reasoning_effort for model in OPENROUTER_CODENAMES_MODELS if model.reasoning_effort}, sort_keys=True)


def model_name(slug: str) -> str:
    model = MODEL_BY_SLUG.get(slug)
    return model.label if model else slug


def winner_model_from_record(record, red_model: str, blue_model: str) -> str | None:
    if record.winner == Team.RED.value:
        return red_model
    if record.winner == Team.BLUE.value:
        return blue_model
    return None


def round_chunks(schedule: list, round_size: int) -> list[list]:
    if round_size <= 0:
        raise ValueError("round_size must be positive")
    return [schedule[index : index + round_size] for index in range(0, len(schedule), round_size)]


def record_tally(models: list[str]) -> dict[str, dict[str, int]]:
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


def record_rankings(records: dict[str, dict[str, int]], ratings: EloRatingSystem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_SLUGS:
        record = records[model]
        games = record["games"]
        win_rate = None if games == 0 else round((record["wins"] + 0.5 * record["ties"]) / games, 4)
        rows.append(
            {
                "model": model,
                "label": MODEL_BY_SLUG[model].label,
                "rating": round(ratings.rating_for(model), 2),
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "games": games,
                "win_rate": win_rate,
            }
        )
    return sorted(rows, key=lambda row: (-row["wins"], row["losses"], -row["ties"], -row["rating"], row["model"]))


def elo_rankings(ratings: EloRatingSystem, records: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for standing in ratings.standings():
        model = standing["model"]
        record = records[model]
        rows.append(
            {
                "rank": standing["rank"],
                "model": model,
                "label": MODEL_BY_SLUG[model].label,
                "rating": standing["rating"],
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "games": record["games"],
            }
        )
    return rows


def play_game(game, output_dir: Path, max_turns: int) -> dict[str, Any]:
    game_dir = output_dir / f"game-{game.game_number:03d}"
    transcript_path = game_dir / "transcript.json"
    summary_path = game_dir / "summary.json"
    client = OpenRouterClient()
    red = LLMTeam(name=model_name(game.red_model), team=Team.RED, model=game.red_model, client=client)
    blue = LLMTeam(name=model_name(game.blue_model), team=Team.BLUE, model=game.blue_model, client=client)
    result: dict[str, Any] = {
        "game": game.to_dict(),
        "transcript": str(transcript_path),
        "summary_path": str(summary_path),
    }
    try:
        record = run_game(red, blue, seed=game.seed, max_turns=max_turns)
        write_transcript(record, transcript_path)
        winner_model = winner_model_from_record(record, game.red_model, game.blue_model)
        result.update(
            {
                "status": "game_complete",
                "winner_model": winner_model,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Run artifact directory. Defaults to runs/openrouter-single-round-robin-<timestamp>.")
    parser.add_argument("--seed-prefix", default="openrouter-codenames-single", help="Prefix used for deterministic board seeds.")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns per game before a bounded/non-terminal result.")
    parser.add_argument("--round-size", type=int, default=DEFAULT_ROUND_SIZE, help="Concurrent games per round.")
    parser.add_argument("--workers", type=int, default=DEFAULT_ROUND_SIZE, help="Parallel games to run inside each round.")
    parser.add_argument("--elo-initial", type=float, default=1500.0, help="Initial Elo rating for each model.")
    parser.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor.")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(json.dumps({"status": "missing_api_key"}, sort_keys=True))
        return 2

    if len(MODEL_SLUGS) % args.round_size != 0:
        print(json.dumps({"status": "invalid_round_size", "model_count": len(MODEL_SLUGS), "round_size": args.round_size}, sort_keys=True))
        return 2

    models = OPENROUTER_CODENAMES_MODELS
    schedule = schedule_single_round_robin(MODEL_SLUGS, seed_prefix=args.seed_prefix)
    rounds = round_chunks(schedule, args.round_size)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "models.json", [model.to_dict() for model in models])
    write_json(output_dir / "provider_order.json", json.loads(provider_order_json()))
    write_json(output_dir / "reasoning_effort.json", json.loads(reasoning_effort_json()))
    write_json(output_dir / "schedule.json", [game.to_dict() for game in schedule])
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "codenames-openrouter-single-round-robin",
            "model_count": len(models),
            "game_count": len(schedule),
            "round_count": len(rounds),
            "round_size": args.round_size,
            "max_turns": args.max_turns,
            "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "10000")),
            "elo_initial": args.elo_initial,
            "elo_k": args.elo_k,
            "seed_prefix": args.seed_prefix,
            "models": [model.to_dict() for model in models],
            "scoring_policy": "all games in a round are scored against the pre-round ratings, then ratings are updated after the full round",
            "provider_policy": "provider.order is pinned per model and allow_fallbacks is false",
            "reasoning_policy": "reasoning.effort is set per model when reasoning_effort is present in models.json",
        },
    )

    old_provider_order = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
    old_reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
    os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = provider_order_json()
    os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = reasoning_effort_json()

    ratings = EloRatingSystem(models=MODEL_SLUGS, initial=args.elo_initial, k=args.elo_k)
    records = record_tally(MODEL_SLUGS)
    round_summaries: list[dict[str, Any]] = []

    try:
        write_json(output_dir / "standings.json", ratings.standings())
        for round_index, round_games in enumerate(rounds, 1):
            round_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(args.workers, len(round_games))) as pool:
                futures = {pool.submit(play_game, game, output_dir, args.max_turns): game for game in round_games}
                for future in as_completed(futures):
                    game = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive artifact write
                        result = {
                            "game": game.to_dict(),
                            "status": "game_error",
                            "error": str(exc)[:1000],
                            "transcript": str(output_dir / f"game-{game.game_number:03d}" / "transcript.json"),
                            "summary_path": str(output_dir / f"game-{game.game_number:03d}" / "summary.json"),
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
            round_elo_entries = ratings.record_round(completed_round_games)
            round_elo_by_game = {entry["game_number"]: entry for entry in round_elo_entries}

            for result in ordered_results:
                game_number = result["game"]["game_number"]
                if result.get("status") == "game_complete":
                    elo_entry = round_elo_by_game[game_number]
                    result["elo_update"] = elo_entry
                    update_record_tally(records, result["game"]["red_model"], result["game"]["blue_model"], result.get("winner_model"))
                write_json(Path(result["summary_path"]), result)
                append_jsonl(output_dir / "results.jsonl", result)
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
                "records": record_rankings(records, ratings),
            }
            round_summaries.append(round_summary)
            write_json(output_dir / "round_summaries.json", round_summaries)
            write_json(output_dir / "standings.json", ratings.standings())
            write_json(output_dir / "elo_history.json", ratings.history)
            write_json(output_dir / "records.json", records)
            write_json(output_dir / "elo_rankings.json", elo_rankings(ratings, records))
            write_json(output_dir / "record_rankings.json", record_rankings(records, ratings))
            write_json(
                output_dir / "run_state.json",
                {
                    "status": "running" if round_index < len(rounds) else "complete",
                    "output_dir": str(output_dir),
                    "scheduled_games": len(schedule),
                    "completed_or_attempted_games": sum(len(round_summary["game_numbers"]) for round_summary in round_summaries),
                    "round_index": round_index,
                    "round_count": len(rounds),
                    "round_size": args.round_size,
                    "standings": ratings.standings(),
                },
            )

        final_elo = elo_rankings(ratings, records)
        final_records = record_rankings(records, ratings)
        final_payload = {
            "status": "complete",
            "output_dir": str(output_dir),
            "scheduled_games": len(schedule),
            "completed_games": len(schedule),
            "round_count": len(rounds),
            "round_size": args.round_size,
            "elo_rankings": final_elo,
            "record_rankings": final_records,
        }
        write_json(
            output_dir / "run_state.json",
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "scheduled_games": len(schedule),
                "completed_or_attempted_games": len(schedule),
                "round_count": len(rounds),
                "round_size": args.round_size,
                "elo_rankings": final_elo,
                "record_rankings": final_records,
            },
        )
        print(json.dumps(final_payload, sort_keys=True))
        return 0
    finally:
        if old_provider_order is None:
            os.environ.pop("OPENROUTER_PROVIDER_ORDER_JSON", None)
        else:
            os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = old_provider_order
        if old_reasoning_effort is None:
            os.environ.pop("OPENROUTER_REASONING_EFFORT_JSON", None)
        else:
            os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = old_reasoning_effort


if __name__ == "__main__":
    raise SystemExit(main())
