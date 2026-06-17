#!/usr/bin/env python3
"""Serially rerun timeout games from a prior OpenRouter Codenames run."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS, TournamentGame
from codenames_benchmark.tournament_runner import (
    elo_rankings,
    model_lookup,
    play_game,
    provider_order_payload,
    reasoning_effort_payload,
    record_rankings,
    record_tally,
    tournament_manifest,
    update_record_tally,
    write_json,
    write_static_artifacts,
)

SOURCE_RUN = ROOT / "runs" / "failed-plus-issue-random-3x-20260614-070654"
MAX_TURNS = 30
WORKERS = 1
ATTEMPTS = 5
TIMEOUT_SECONDS = 45


def timeout_games_from_source() -> list[TournamentGame]:
    path = SOURCE_RUN / "results.jsonl"
    games: list[TournamentGame] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "game_complete":
                continue
            if "timed out" not in row.get("error", ""):
                continue
            games.append(TournamentGame(**row["game"]))
    if len(games) != 10:
        raise RuntimeError(f"expected 10 timeout games from {path}, found {len(games)}")
    return games


def model_specs_for_schedule(schedule: list[TournamentGame]):
    by_slug = {model.slug: model for model in OPENROUTER_CODENAMES_MODELS}
    needed = []
    seen = set()
    for game in schedule:
        for slug in (game.red_model, game.blue_model):
            if slug in by_slug and slug not in seen:
                needed.append(by_slug[slug])
                seen.add(slug)
            elif slug not in by_slug:
                raise RuntimeError(f"scheduled model {slug!r} is not in current model list")
    return needed


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in the live shell")

    os.environ["OPENROUTER_NODE_ATTEMPTS"] = str(ATTEMPTS)
    os.environ["OPENROUTER_NODE_TIMEOUT_SECONDS"] = str(TIMEOUT_SECONDS)
    os.environ.setdefault("OPENROUTER_MAX_TOKENS", "10000")

    schedule = timeout_games_from_source()
    models = model_specs_for_schedule(schedule)
    labels = model_lookup(OPENROUTER_CODENAMES_MODELS)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = ROOT / "runs" / f"serial-timeout-rerun-45s-5attempts-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = tournament_manifest(
        benchmark="codenames-openrouter-serial-timeout-rerun",
        models=models,
        game_count=len(schedule),
        max_turns=MAX_TURNS,
        seed_prefix="serial-timeout-rerun",
        elo_initial=1500.0,
        elo_k=32.0,
        schedule_mode="custom-timeout-rerun",
        round_size=WORKERS,
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "10000")),
        extra_fields={
            "source_run": str(SOURCE_RUN),
            "source_error_filter": "status != game_complete and error contains 'timed out'",
            "workers": WORKERS,
            "openrouter_node_attempts": ATTEMPTS,
            "openrouter_node_timeout_seconds": TIMEOUT_SECONDS,
            "selected_games": [game.to_dict() for game in schedule],
        },
    )
    write_static_artifacts(output_dir=output_dir, models=models, schedule=schedule, manifest=manifest)

    provider_order = provider_order_payload(models)
    os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = json.dumps(provider_order, sort_keys=True)
    reasoning_effort = reasoning_effort_payload(models)
    os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = json.dumps(reasoning_effort, sort_keys=True)

    all_model_slugs = sorted({slug for game in schedule for slug in (game.red_model, game.blue_model)})
    ratings = EloRatingSystem(models=all_model_slugs, initial=1500.0, k=32.0)
    records = record_tally(all_model_slugs)
    client = OpenRouterClient()
    round_summaries = []

    for round_index, game in enumerate(schedule, 1):
        result = play_game(game, output_dir=output_dir, max_turns=MAX_TURNS, labels=labels, client=client)
        if result.get("status") == "game_complete":
            entry = ratings.record_game(
                red_model=result["game"]["red_model"],
                blue_model=result["game"]["blue_model"],
                winner_model=result.get("winner_model"),
                game_number=result["game"]["game_number"],
            )
            result["elo_update"] = entry
            update_record_tally(records, result["game"]["red_model"], result["game"]["blue_model"], result.get("winner_model"))
        write_json(Path(result["summary_path"]), result)
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps({
            "round": round_index,
            "game_number": result["game"]["game_number"],
            "red_model": result["game"]["red_model"],
            "blue_model": result["game"]["blue_model"],
            "status": result.get("status"),
            "winner": result.get("winner_model"),
            "reason": result.get("reason"),
            "error": result.get("error", "")[:180],
        }, sort_keys=True), flush=True)

        round_summaries.append({
            "round_index": round_index,
            "game_numbers": [result["game"]["game_number"]],
            "completed_games": 1 if result.get("status") == "game_complete" else 0,
            "error_games": 0 if result.get("status") == "game_complete" else 1,
            "standings": ratings.standings(),
            "records": record_rankings(records, ratings, all_model_slugs, labels),
        })
        write_json(output_dir / "round_summaries.json", round_summaries)
        write_json(output_dir / "standings.json", ratings.standings())
        write_json(output_dir / "elo_history.json", ratings.history)
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "elo_rankings.json", elo_rankings(ratings, records, labels))
        write_json(output_dir / "record_rankings.json", record_rankings(records, ratings, all_model_slugs, labels))
        write_json(output_dir / "run_state.json", {
            "status": "running" if round_index < len(schedule) else "complete",
            "output_dir": str(output_dir),
            "source_run": str(SOURCE_RUN),
            "scheduled_games": len(schedule),
            "completed_or_attempted_games": round_index,
            "round_index": round_index,
            "round_count": len(schedule),
            "round_size": WORKERS,
            "workers": WORKERS,
            "openrouter_node_attempts": ATTEMPTS,
            "openrouter_node_timeout_seconds": TIMEOUT_SECONDS,
            "standings": ratings.standings(),
        })

    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "scheduled_games": len(schedule)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
