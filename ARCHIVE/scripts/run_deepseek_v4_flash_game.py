#!/usr/bin/env python3
"""Run a small DeepSeek V4 Flash vs DeepSeek V4 Flash Codenames game.

If OPENROUTER_API_KEY is absent, this script reports configuration readiness and
exits without making network calls. It never prints API keys.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.openrouter import DEEPSEEK_V4_FLASH_MODEL, OpenRouterClient
from codenames_benchmark.runner import run_game
from codenames_benchmark.transcript import write_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek V4 Flash Codenames smoke game.")
    parser.add_argument("--seed", default="1", help="Deterministic board seed.")
    parser.add_argument("--max-turns", type=int, default=6, help="Maximum clue turns before stopping.")
    parser.add_argument("--transcript", help="Optional JSON transcript output path for completed live runs.")
    return parser.parse_args()


def configured_payload(seed: str, max_turns: int) -> dict:
    return {
        "status": "configured_no_api_key",
        "model": DEEPSEEK_V4_FLASH_MODEL,
        "seed": str(seed),
        "max_turns": max_turns,
        "teams": {
            "red": {"name": "deepseek-red", "agents": ["spymaster", "guesser-0", "guesser-1", "guesser-2"]},
            "blue": {"name": "deepseek-blue", "agents": ["spymaster", "guesser-0", "guesser-1", "guesser-2"]},
        },
    }


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(json.dumps(configured_payload(args.seed, args.max_turns), sort_keys=True))
        return 0

    client = OpenRouterClient()
    red = LLMTeam.deepseek_v4_flash("deepseek-red", Team.RED, client)
    blue = LLMTeam.deepseek_v4_flash("deepseek-blue", Team.BLUE, client)
    record = run_game(red, blue, seed=args.seed, max_turns=args.max_turns)
    transcript_path = str(write_transcript(record, args.transcript)) if args.transcript else None
    print(json.dumps({
        "status": "game_complete" if record.terminal else "max_turns_reached",
        "model": DEEPSEEK_V4_FLASH_MODEL,
        "seed": str(args.seed),
        "max_turns": args.max_turns,
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
