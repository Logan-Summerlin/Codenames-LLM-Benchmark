#!/usr/bin/env python3
"""Canonical OpenRouter Codenames tournament runner.

This single CLI replaces the old one-off launcher scripts. It supports the full
OpenRouter field or the top-four preset, plus single round robin, double round
robin, and limited-coverage schedules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.tournament_runner import run_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Run artifact directory. Defaults to runs/<benchmark>-<timestamp>.")
    parser.add_argument("--model-preset", default="full-field", choices=("full-field", "top4"), help="Which model subset to run.")
    parser.add_argument(
        "--schedule-mode",
        default="double-round-robin",
        choices=("single-round-robin", "double-round-robin", "limited-coverage"),
        help="Tournament schedule shape.",
    )
    parser.add_argument("--seed-prefix", default="openrouter-codenames", help="Prefix used for deterministic board seeds.")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns per game before a bounded/non-terminal result.")
    parser.add_argument("--round-size", type=int, help="Games scored together before ratings update. Defaults depend on schedule mode.")
    parser.add_argument("--workers", type=int, help="Parallel games to run per round. Defaults to the round size.")
    parser.add_argument("--start-game", type=int, default=1, help="First scheduled game number to run, 1-indexed.")
    parser.add_argument("--limit-games", type=int, help="Maximum number of scheduled games to run from --start-game.")
    parser.add_argument("--elo-initial", type=float, default=1500.0, help="Initial Elo rating for each model.")
    parser.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor.")
    parser.add_argument("--dry-run", action="store_true", help="Write schedule/manifest only; do not call OpenRouter.")
    parser.add_argument("--require-api-key", action="store_true", help="Fail if OPENROUTER_API_KEY is absent instead of writing a configured-no-key plan.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_tournament(
        root=ROOT,
        model_preset=args.model_preset,
        schedule_mode=args.schedule_mode,
        seed_prefix=args.seed_prefix,
        max_turns=args.max_turns,
        elo_initial=args.elo_initial,
        elo_k=args.elo_k,
        round_size=args.round_size,
        workers=args.workers,
        start_game=args.start_game,
        limit_games=args.limit_games,
        dry_run=args.dry_run,
        require_api_key=args.require_api_key,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        benchmark=f"codenames-openrouter-{args.schedule_mode}",
        extra_manifest_fields={"model_preset": args.model_preset},
    )
    print(json.dumps(result.payload, sort_keys=True))
    if result.status == "missing_api_key":
        return 2
    if result.status in {"dry_run", "configured_no_api_key", "complete"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
