#!/usr/bin/env python3
"""Summarize a tournament run directory into per-model outcomes and diagnostics.

Reads ``game-*/transcript.json`` artifacts written by the tournament runner and
prints a JSON summary with per-model win/loss/tie records, terminal-vs-bounded
game counts, illegal clue rates, and assassin/neutral/opponent hit counts. Use
it to sanity-check a live run before trusting its standings.

Usage:
    python3 scripts/summarize_run.py runs/<run-directory>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.summary import summarize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing game-*/transcript.json files.")
    parser.add_argument("--output", help="Optional path to also write the JSON summary to.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(json.dumps({"error": f"not a directory: {run_dir}"}), file=sys.stderr)
        return 2
    summary = summarize_run(run_dir)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
