#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is required")

OUT_DIR = ROOT / "runs" / f"limited-random-14-models-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = int(os.environ.get("LIMITED_TOURNAMENT_MAX_TURNS", "10"))
MAX_WORKERS = 3
RETRIES = 2

rng = random.SystemRandom()
models = OPENROUTER_CODENAMES_MODELS.copy()
rng.shuffle(models)

pairs = []
for i in range(0, len(models), 2):
    a = models[i]
    b = models[i + 1]
    red, blue = (a, b) if rng.random() < 0.5 else (b, a)
    pairs.append(
        {
            "match_index": i // 2 + 1,
            "red_model": red.slug,
            "blue_model": blue.slug,
            "red_name": red.label,
            "blue_name": blue.label,
            "seed": f"{OUT_DIR.name}-m{i // 2 + 1:02d}",
            "transcript": str(OUT_DIR / f"match-{i // 2 + 1:02d}.json"),
        }
    )

(OUT_DIR / "pairings.json").write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_match(pair: dict) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_openrouter_matchup_game.py"),
        "--seed",
        pair["seed"],
        "--max-turns",
        str(MAX_TURNS),
        "--red-model",
        pair["red_model"],
        "--blue-model",
        pair["blue_model"],
        "--red-name",
        pair["red_name"],
        "--blue-name",
        pair["blue_name"],
        "--transcript",
        pair["transcript"],
    ]
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = API_KEY
    attempts = []
    for attempt in range(1, RETRIES + 1):
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)
        stdout_lines = (proc.stdout or "").strip().splitlines()
        tail = stdout_lines[-1] if stdout_lines else ""
        parsed = None
        parse_error = None
        try:
            parsed = json.loads(tail) if tail else None
        except Exception as exc:  # pragma: no cover - debug helper
            parse_error = str(exc)
        attempts.append(
            {
                "attempt": attempt,
                "returncode": proc.returncode,
                "parsed": parsed,
                "stdout_tail": stdout_lines[-3:],
                "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
                "parse_error": parse_error,
            }
        )
        if proc.returncode == 0 and isinstance(parsed, dict) and parsed.get("status") == "game_complete":
            return {
                **pair,
                "status": parsed.get("status"),
                "winner": parsed.get("winner"),
                "reason": parsed.get("reason"),
                "terminal": parsed.get("terminal"),
                "public_events": parsed.get("public_events"),
                "private_events": parsed.get("private_events"),
                "transcript_result": parsed.get("transcript"),
                "attempts": attempts,
            }
        time.sleep(2)

    return {
        **pair,
        "status": "failed",
        "attempts": attempts,
    }

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {executor.submit(run_match, pair): pair for pair in pairs}
    for fut in as_completed(future_map):
        result = fut.result()
        results.append(result)
        print(
            json.dumps(
                {
                    "match_index": result["match_index"],
                    "status": result.get("status"),
                    "winner": result.get("winner"),
                    "red_model": result["red_model"],
                    "blue_model": result["blue_model"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

results.sort(key=lambda item: item["match_index"])
summary = {
    "output_dir": str(OUT_DIR),
    "pair_count": len(results),
    "completed": sum(1 for r in results if r.get("status") == "game_complete"),
    "failed": sum(1 for r in results if r.get("status") != "game_complete"),
    "max_turns": MAX_TURNS,
    "concurrency": MAX_WORKERS,
    "results": [
        {
            "match_index": r["match_index"],
            "red_model": r["red_model"],
            "blue_model": r["blue_model"],
            "seed": r["seed"],
            "status": r.get("status"),
            "winner": r.get("winner"),
            "reason": r.get("reason"),
            "terminal": r.get("terminal"),
            "public_events": r.get("public_events"),
            "private_events": r.get("private_events"),
            "transcript": r.get("transcript_result"),
            "attempts": len(r.get("attempts", [])),
        }
        for r in results
    ],
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"done": True, **{k: summary[k] for k in ["output_dir", "pair_count", "completed", "failed", "max_turns", "concurrency"]}}, sort_keys=True))
