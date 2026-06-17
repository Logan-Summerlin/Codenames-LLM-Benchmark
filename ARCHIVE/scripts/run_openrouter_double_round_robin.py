#!/usr/bin/env python3
"""Run a live OpenRouter double round-robin Codenames tournament.

The default field uses the verified OpenRouter slugs from the historical model
list, with Mistral 7B Instruct v0.1 intentionally removed because it no longer
has active OpenRouter endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS, tournament_limited_coverage_pairings, tournament_pairings
from codenames_benchmark.transcript import write_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Run artifact directory. Defaults to runs/openrouter-double-round-robin-<timestamp>.")
    parser.add_argument("--seed-prefix", default="openrouter-codenames", help="Prefix used for deterministic board seeds.")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns per game before the game is treated as bounded/non-terminal.")
    parser.add_argument("--elo-initial", type=float, default=1500.0, help="Initial Elo rating for each model.")
    parser.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor.")
    parser.add_argument("--start-game", type=int, default=1, help="First scheduled game number to run, 1-indexed.")
    parser.add_argument("--limit-games", type=int, help="Maximum number of scheduled games to run from --start-game.")
    parser.add_argument(
        "--schedule-mode",
        choices=("double-round-robin", "limited-coverage"),
        default="double-round-robin",
        help="Tournament schedule shape. limited-coverage gives every model at least one appearance.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write schedule/manifest only; do not call OpenRouter.")
    parser.add_argument("--require-api-key", action="store_true", help="Fail if OPENROUTER_API_KEY is absent instead of writing a configured_no_api_key plan.")
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return ROOT / "runs" / f"openrouter-double-round-robin-{stamp}"


def provider_order_json() -> str:
    return json.dumps({model.slug: [model.provider] for model in OPENROUTER_CODENAMES_MODELS}, sort_keys=True)


def reasoning_effort_json() -> str:
    return json.dumps({model.slug: model.reasoning_effort for model in OPENROUTER_CODENAMES_MODELS if model.reasoning_effort}, sort_keys=True)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def selected_schedule(seed_prefix: str, start_game: int, limit_games: int | None, schedule_mode: str):
    if schedule_mode == "limited-coverage":
        schedule = tournament_limited_coverage_pairings(seed_prefix=seed_prefix)
    else:
        schedule = tournament_pairings(seed_prefix=seed_prefix)
    selected = [game for game in schedule if game.game_number >= start_game]
    if limit_games is not None:
        selected = selected[:limit_games]
    return schedule, selected


def model_name(slug: str) -> str:
    for model in OPENROUTER_CODENAMES_MODELS:
        if model.slug == slug:
            return model.label
    return slug


def winner_model_from_record(record, red_model: str, blue_model: str) -> str | None:
    if record.winner == Team.RED.value:
        return red_model
    if record.winner == Team.BLUE.value:
        return blue_model
    return None


def write_static_artifacts(output_dir: Path, args: argparse.Namespace, schedule) -> None:
    write_json(output_dir / "models.json", [model.to_dict() for model in OPENROUTER_CODENAMES_MODELS])
    write_json(output_dir / "provider_order.json", json.loads(provider_order_json()))
    write_json(output_dir / "reasoning_effort.json", json.loads(reasoning_effort_json()))
    write_json(output_dir / "schedule.json", [game.to_dict() for game in schedule])
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": f"codenames-openrouter-{args.schedule_mode}",
            "model_count": len(OPENROUTER_CODENAMES_MODELS),
            "game_count": len(schedule),
            "max_turns": args.max_turns,
            "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "10000")),
            "elo_initial": args.elo_initial,
            "elo_k": args.elo_k,
            "seed_prefix": args.seed_prefix,
            "schedule_mode": args.schedule_mode,
            "models": [model.to_dict() for model in OPENROUTER_CODENAMES_MODELS],
            "scoring_policy": "terminal red/blue wins update as wins; missing winner or bounded non-terminal games update as Elo draws",
            "provider_policy": "provider.order is pinned per model and allow_fallbacks is false",
            "reasoning_policy": "reasoning.effort is set per model when reasoning_effort is present in models.json",
        },
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    schedule, games_to_run = selected_schedule(args.seed_prefix, args.start_game, args.limit_games, args.schedule_mode)
    write_static_artifacts(output_dir, args, schedule)

    api_key_present = bool(os.environ.get("OPENROUTER_API_KEY"))
    if args.dry_run or not api_key_present:
        status = "dry_run" if args.dry_run else "configured_no_api_key"
        if args.require_api_key and not api_key_present:
            print(json.dumps({"status": "missing_api_key", "output_dir": str(output_dir)}, sort_keys=True))
            return 2
        write_json(
            output_dir / "run_state.json",
            {
                "status": status,
                "output_dir": str(output_dir),
                "scheduled_games": len(schedule),
                "selected_games": [game.to_dict() for game in games_to_run],
                "models": [model.to_dict() for model in OPENROUTER_CODENAMES_MODELS],
                "schedule_mode": args.schedule_mode,
            },
        )
        print(json.dumps({"status": status, "output_dir": str(output_dir), "scheduled_games": len(schedule), "selected_games": len(games_to_run)}, sort_keys=True))
        return 0

    old_provider_order = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
    old_reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
    os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = provider_order_json()
    os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = reasoning_effort_json()
    try:
        client = OpenRouterClient()
        ratings = EloRatingSystem(models=[model.slug for model in OPENROUTER_CODENAMES_MODELS], initial=args.elo_initial, k=args.elo_k)
        write_json(output_dir / "standings.json", ratings.standings())

        for game in games_to_run:
            game_dir = output_dir / f"game-{game.game_number:03d}"
            transcript_path = game_dir / "transcript.json"
            summary_path = game_dir / "summary.json"
            try:
                red = LLMTeam(name=model_name(game.red_model), team=Team.RED, model=game.red_model, client=client)
                blue = LLMTeam(name=model_name(game.blue_model), team=Team.BLUE, model=game.blue_model, client=client)
                record = run_game(red, blue, seed=game.seed, max_turns=args.max_turns)
                write_transcript(record, transcript_path)
                winner_model = winner_model_from_record(record, game.red_model, game.blue_model)
                elo_entry = ratings.record_game(
                    red_model=game.red_model,
                    blue_model=game.blue_model,
                    winner_model=winner_model,
                    game_number=game.game_number,
                )
                summary = {
                    "status": "game_complete",
                    "game": game.to_dict(),
                    "winner_model": winner_model,
                    "winner_color": record.winner,
                    "terminal": record.terminal,
                    "reason": record.reason,
                    "public_events": len(record.public_events),
                    "private_events": len(record.private_events),
                    "transcript": str(transcript_path),
                    "elo_update": elo_entry,
                }
            except Exception as exc:
                summary = {
                    "status": "game_error",
                    "game": game.to_dict(),
                    "error": str(exc)[:1000],
                }
            write_json(summary_path, summary)
            append_jsonl(output_dir / "results.jsonl", summary)
            write_json(output_dir / "standings.json", ratings.standings())
            write_json(output_dir / "elo_history.json", ratings.history)
            print(json.dumps({"game_number": game.game_number, "status": summary["status"], "leader": ratings.standings()[0]["model"]}, sort_keys=True), flush=True)

        write_json(
            output_dir / "run_state.json",
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "scheduled_games": len(schedule),
                "completed_or_attempted_games": len(games_to_run),
                "schedule_mode": args.schedule_mode,
                "standings": ratings.standings(),
            },
        )
        print(json.dumps({"status": "complete", "output_dir": str(output_dir), "standings": ratings.standings()}, sort_keys=True))
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
