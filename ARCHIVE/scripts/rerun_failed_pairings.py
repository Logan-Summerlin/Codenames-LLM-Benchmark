#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_openrouter_matchup_game.py"
BASE_RUN_DIR = ROOT / "runs" / "limited-random-14-models-20260607-214614"
OUT_DIR = BASE_RUN_DIR / f"rerun-failed-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is required")

MAX_TURNS = int(os.environ.get("LIMITED_TOURNAMENT_MAX_TURNS", "30"))
RETRIES = 2
MAX_WORKERS = 3

PAIRS = [
    {
        "match_index": 5,
        "red_model": "meta-llama/llama-3-70b-instruct",
        "blue_model": "deepseek/deepseek-v4-flash",
        "red_name": "Llama 3 70B Instruct",
        "blue_name": "DeepSeek V4 Flash",
        "seed": "limited-random-14-models-20260607-214614-m05",
        "transcript": str(OUT_DIR / "match-05.json"),
    },
    {
        "match_index": 6,
        "red_model": "deepseek/deepseek-chat-v3-0324",
        "blue_model": "mistralai/mistral-medium-3.1",
        "red_name": "DeepSeek V3 0324",
        "blue_name": "Mistral Medium 3.1",
        "seed": "limited-random-14-models-20260607-214614-m06",
        "transcript": str(OUT_DIR / "match-06.json"),
    },
    {
        "match_index": 7,
        "red_model": "xiaomi/mimo-v2.5",
        "blue_model": "qwen/qwen-2.5-72b-instruct",
        "red_name": "MiMo V2.5",
        "blue_name": "Qwen2.5 72B Instruct",
        "seed": "limited-random-14-models-20260607-214614-m07",
        "transcript": str(OUT_DIR / "match-07.json"),
    },
]

(OUT_DIR / "pairings.json").write_text(json.dumps(PAIRS, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_match(pair: dict) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT),
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
    future_map = {executor.submit(run_match, pair): pair for pair in PAIRS}
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
