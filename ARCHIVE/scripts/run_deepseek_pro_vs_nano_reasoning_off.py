#!/usr/bin/env python3
"""Run a short mirrored benchmark: DeepSeek V4 Pro vs GPT-4.1-nano.

DeepSeek Pro requests are sent with reasoning disabled, while GPT-4.1-nano
uses the default OpenRouter payload.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.runner import run_mirrored_matchup
from codenames_benchmark.transcript import write_transcript

RUN_DIR = ROOT / "runs" / "live-matchup-deepseek-v4-pro-vs-gpt-4.1-nano-20260604"


class RoutedOpenRouterClient(OpenRouterClient):
    """Disable reasoning only for DeepSeek Pro requests."""

    def complete(self, request):
        old = os.environ.get("OPENROUTER_REASONING_ENABLED")
        try:
            if request.model == "deepseek/deepseek-v4-pro":
                os.environ["OPENROUTER_REASONING_ENABLED"] = "false"
            else:
                os.environ.pop("OPENROUTER_REASONING_ENABLED", None)
            return super().complete(request)
        finally:
            if old is None:
                os.environ.pop("OPENROUTER_REASONING_ENABLED", None)
            else:
                os.environ["OPENROUTER_REASONING_ENABLED"] = old


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def main() -> int:
    require_env("OPENROUTER_API_KEY")
    os.environ.setdefault("OPENROUTER_NODE_TIMEOUT_SECONDS", "20")
    os.environ.setdefault("OPENROUTER_NODE_ATTEMPTS", "1")
    os.environ.setdefault("OPENROUTER_MAX_TOKENS", "10000")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    client = RoutedOpenRouterClient()

    print(
        json.dumps(
            {
                "status": "starting",
                "run_dir": str(RUN_DIR),
                "deepseek_model": "deepseek/deepseek-v4-pro",
                "blue_model": "openai/gpt-4.1-nano",
                "reasoning_policy": "deepseek_pro_only_false",
                "mode": "mirrored_matchup_seed_1",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    red = LLMTeam(name="deepseek-v4-pro", team=Team.RED, model="deepseek/deepseek-v4-pro", client=client)
    blue = LLMTeam(name="gpt-4.1-nano", team=Team.BLUE, model="openai/gpt-4.1-nano", client=client)
    records = run_mirrored_matchup(red, blue, seed=1, max_turns=1)

    outputs = []
    for idx, record in enumerate(records, start=1):
        transcript = RUN_DIR / f"mirrored-game-{idx}-transcript.json"
        write_transcript(record, transcript)
        row = {
            "game": idx,
            "winner": record.winner,
            "terminal": record.terminal,
            "reason": record.reason,
            "public_events": len(record.public_events),
            "private_events": len(record.private_events),
            "transcript": str(transcript),
        }
        outputs.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "status": "complete",
        "games": len(outputs),
        "wins_red": sum(1 for row in outputs if row["winner"] == "red"),
        "wins_blue": sum(1 for row in outputs if row["winner"] == "blue"),
        "nonterminal": sum(1 for row in outputs if not row["terminal"]),
        "run_dir": str(RUN_DIR),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
