#!/usr/bin/env python3
"""Run a live OpenRouter single round robin among the four strongest Codenames models.

This helper selects the four top models, pins provider order / reasoning effort
per model, and executes two games at a time so the benchmark can progress in
parallel without overloading the local host.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.tournament import ModelSpec, OPENROUTER_CODENAMES_MODELS, TournamentGame

TOP4_SLUGS = {
    "google/gemini-3.1-flash-lite",
    "openai/gpt-oss-120b",
    "openai/gpt-5-nano",
    "google/gemma-4-31b-it",
}


def selected_models() -> list[ModelSpec]:
    models = [m for m in OPENROUTER_CODENAMES_MODELS if m.slug in TOP4_SLUGS]
    slug_order = [
        "google/gemini-3.1-flash-lite",
        "openai/gpt-oss-120b",
        "openai/gpt-5-nano",
        "google/gemma-4-31b-it",
    ]
    by_slug = {m.slug: m for m in models}
    return [by_slug[slug] for slug in slug_order]


def schedule_single_round_robin(models: list[str], *, seed_prefix: str = "rr") -> list[TournamentGame]:
    """Schedule one color-assigned game for every unordered model pair."""
    games: list[TournamentGame] = []
    game_number = 1
    for pair_index, (a, b) in enumerate(combinations(models, 2)):
        if pair_index % 2 == 0:
            red, blue = a, b
        else:
            red, blue = b, a
        seed = f"{seed_prefix}-{game_number:03d}-{red.replace('/', '_').replace('-', '_')}-vs-{blue.replace('/', '_').replace('-', '_')}"
        games.append(
            TournamentGame(
                game_number=game_number,
                round_index=1,
                red_model=red,
                blue_model=blue,
                seed=seed,
            )
        )
        game_number += 1
    return games


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return ROOT / "runs" / f"top4-single-round-robin-{stamp}"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def provider_order_json(models: list[ModelSpec]) -> str:
    return json.dumps({m.slug: [m.provider] for m in models}, sort_keys=True)


def reasoning_effort_json(models: list[ModelSpec]) -> str:
    return json.dumps({m.slug: m.reasoning_effort for m in models if m.reasoning_effort}, sort_keys=True)


def model_name(slug: str) -> str:
    for model in selected_models():
        if model.slug == slug:
            return model.label
    return slug


def run_game(game, output_dir: Path, max_turns: int) -> dict[str, Any]:
    transcript = output_dir / f"game-{game.game_number:03d}-transcript.json"
    summary_path = output_dir / f"game-{game.game_number:03d}-summary.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_openrouter_matchup_game.py"),
        "--seed",
        game.seed,
        "--max-turns",
        str(max_turns),
        "--red-model",
        game.red_model,
        "--blue-model",
        game.blue_model,
        "--red-name",
        model_name(game.red_model),
        "--blue-name",
        model_name(game.blue_model),
        "--transcript",
        str(transcript),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    result: dict[str, Any] = {
        "game": game.to_dict(),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "transcript": str(transcript),
    }
    if proc.returncode == 0 and transcript.exists():
        try:
            data = json.loads(transcript.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            result.update(
                {
                    "status": "game_complete",
                    "winner": summary.get("winner"),
                    "reason": summary.get("reason"),
                    "terminal": summary.get("terminal"),
                    "turns": len(data.get("turns", [])),
                    "public_events": summary.get("public_event_count"),
                    "private_actions": summary.get("private_action_count"),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive artifact write
            result.update({"status": "transcript_parse_error", "error": str(exc)})
    else:
        result.update({"status": "game_error", "error": (proc.stderr or proc.stdout or "unknown failure")[:1000]})
    write_json(summary_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Output directory for run artifacts.")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2, help="Parallel games to run at once.")
    parser.add_argument("--seed-prefix", default="top4-single-round-robin")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(json.dumps({"status": "missing_api_key"}, sort_keys=True))
        return 2

    models = selected_models()
    schedule = schedule_single_round_robin([m.slug for m in models], seed_prefix=args.seed_prefix)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "models.json", [m.to_dict() for m in models])
    write_json(output_dir / "provider_order.json", json.loads(provider_order_json(models)))
    write_json(output_dir / "reasoning_effort.json", json.loads(reasoning_effort_json(models)))
    write_json(output_dir / "schedule.json", [g.to_dict() for g in schedule])
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "codenames-openrouter-top4-single-round-robin",
            "model_count": len(models),
            "game_count": len(schedule),
            "max_turns": args.max_turns,
            "parallel_workers": args.workers,
            "models": [m.to_dict() for m in models],
            "provider_policy": "provider.order is pinned per model and allow_fallbacks is false",
            "reasoning_policy": "reasoning.effort is set per model when present in models.json",
        },
    )

    old_provider = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
    old_reasoning = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
    os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = provider_order_json(models)
    os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = reasoning_effort_json(models)

    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_game, game, output_dir, args.max_turns): game for game in schedule}
            for future in as_completed(futures):
                game = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive artifact write
                    result = {"game": game.to_dict(), "status": "game_error", "error": str(exc)}
                    write_json(output_dir / f"game-{game.game_number:03d}-summary.json", result)
                results.append(result)
                print(json.dumps({
                    "game_number": game.game_number,
                    "status": result.get("status"),
                    "winner": result.get("winner"),
                    "reason": result.get("reason"),
                }, sort_keys=True), flush=True)
    finally:
        if old_provider is None:
            os.environ.pop("OPENROUTER_PROVIDER_ORDER_JSON", None)
        else:
            os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = old_provider
        if old_reasoning is None:
            os.environ.pop("OPENROUTER_REASONING_EFFORT_JSON", None)
        else:
            os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = old_reasoning

    write_json(output_dir / "results.json", results)
    complete = all(r.get("status") == "game_complete" for r in results)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "complete" if complete else "partial",
            "output_dir": str(output_dir),
            "scheduled_games": len(schedule),
            "completed_or_attempted_games": len(results),
            "parallel_workers": args.workers,
            "results": [{
                "game_number": r["game"]["game_number"],
                "status": r.get("status"),
                "winner": r.get("winner"),
                "reason": r.get("reason"),
            } for r in sorted(results, key=lambda x: x["game"]["game_number"])],
        },
    )
    print(json.dumps({"status": "complete" if complete else "partial", "output_dir": str(output_dir), "games": len(results)}, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
