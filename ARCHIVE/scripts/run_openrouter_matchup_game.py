#!/usr/bin/env python3
"""Run one live OpenRouter-backed Codenames game between two model teams."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.runner import run_game
from codenames_benchmark.transcript import write_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, help="Board seed to run.")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns before bounding the game.")
    parser.add_argument("--red-model", required=True, help="OpenRouter model slug for the red team.")
    parser.add_argument("--blue-model", required=True, help="OpenRouter model slug for the blue team.")
    parser.add_argument("--red-name", default="red", help="Human-readable red team name.")
    parser.add_argument("--blue-name", default="blue", help="Human-readable blue team name.")
    parser.add_argument("--transcript", help="Path to write transcript JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(json.dumps({
            "status": "configured_no_api_key",
            "seed": args.seed,
            "max_turns": args.max_turns,
            "red_model": args.red_model,
            "blue_model": args.blue_model,
        }, sort_keys=True))
        return 0

    client = OpenRouterClient()
    red = LLMTeam(name=args.red_name, team=Team.RED, model=args.red_model, client=client)
    blue = LLMTeam(name=args.blue_name, team=Team.BLUE, model=args.blue_model, client=client)
    record = run_game(red, blue, seed=args.seed, max_turns=args.max_turns)
    transcript_path = None
    if args.transcript:
        transcript_path = str(write_transcript(record, args.transcript))
    print(json.dumps({
        "status": "game_complete",
        "seed": args.seed,
        "max_turns": args.max_turns,
        "red_model": args.red_model,
        "blue_model": args.blue_model,
        "winner": record.winner,
        "terminal": record.terminal,
        "reason": record.reason,
        "public_events": len(record.public_events),
        "private_events": len(record.private_events),
        "transcript": transcript_path,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
