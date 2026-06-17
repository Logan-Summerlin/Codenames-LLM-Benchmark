#!/usr/bin/env python3
"""Rerun failed Codenames games plus issue-model random matchups.

This is a one-off recovery launcher. It uses the repository's shared tournament
helpers, reads OPENROUTER_API_KEY only from the live process environment, and
never writes credentials to artifacts.
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_TURNS = 30
WORKERS = 3
RANDOM_SEED = "issue-model-random-matchups-20260614"

FAILED_GAMES = [
    TournamentGame(
        game_number=4,
        round_index=1,
        red_model="xiaomi/mimo-v2.5",
        blue_model="mistralai/mistral-medium-3.1",
        seed="openrouter-codenames-004-xiaomi_mimo_v2_5-vs-mistralai_mistral_medium_3_1",
    ),
    TournamentGame(
        game_number=5,
        round_index=1,
        red_model="microsoft/phi-4",
        blue_model="deepseek/deepseek-v4-flash",
        seed="openrouter-codenames-005-microsoft_phi_4-vs-deepseek_deepseek_v4_flash",
    ),
    TournamentGame(
        game_number=9,
        round_index=1,
        red_model="minimax/minimax-m2.7",
        blue_model="stepfun/step-3.7-flash",
        seed="openrouter-codenames-009-minimax_minimax_m2_7-vs-stepfun_step_3_7_flash",
    ),
    TournamentGame(
        game_number=11,
        round_index=1,
        red_model="inclusionai/ling-2.6-flash",
        blue_model="qwen/qwen3-32b",
        seed="openrouter-codenames-011-inclusionai_ling_2_6_flash-vs-qwen_qwen3_32b",
    ),
]

PREVIOUS_ISSUE_MODELS = [
    "xiaomi/mimo-v2.5",
    "mistralai/mistral-medium-3.1",
    "microsoft/phi-4",
    "deepseek/deepseek-v4-flash",
    "minimax/minimax-m2.7",
    "stepfun/step-3.7-flash",
    "inclusionai/ling-2.6-flash",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout",
    "google/gemma-3-27b-it",
    "anthropic/claude-3-haiku",
]


def safe_seed(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_").replace(".", "_").replace("-", "_")


def current_model_slugs() -> list[str]:
    return [model.slug for model in OPENROUTER_CODENAMES_MODELS]


def build_random_issue_games(start_number: int) -> list[TournamentGame]:
    slugs = current_model_slugs()
    slug_set = set(slugs)
    issue_slugs = [slug for slug in PREVIOUS_ISSUE_MODELS if slug in slug_set]
    rng = random.Random(RANDOM_SEED)
    games: list[TournamentGame] = []
    for offset, issue_slug in enumerate(issue_slugs):
        candidates = [slug for slug in slugs if slug != issue_slug]
        opponent = rng.choice(candidates)
        # Alternate colors so issue models are not always red.
        if offset % 2 == 0:
            red, blue = issue_slug, opponent
        else:
            red, blue = opponent, issue_slug
        game_number = start_number + offset
        games.append(
            TournamentGame(
                game_number=game_number,
                round_index=1,
                red_model=red,
                blue_model=blue,
                seed=f"issue-random-{game_number:03d}-{safe_seed(red)}-vs-{safe_seed(blue)}",
            )
        )
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
    # Ling is not in the current model list, but it appears in a failed game.
    # The caller asked to rerun all failed games, so leave it out of model-level
    # static policy artifacts and let the client call the schedule slug directly.
    return needed


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in the live shell")

    os.environ.setdefault("OPENROUTER_NODE_ATTEMPTS", "4")
    os.environ.setdefault("OPENROUTER_NODE_TIMEOUT_SECONDS", "20")
    os.environ.setdefault("OPENROUTER_MAX_TOKENS", "10000")

    random_games = build_random_issue_games(start_number=101)
    schedule = FAILED_GAMES + random_games
    models = model_specs_for_schedule(schedule)
    labels = model_lookup(OPENROUTER_CODENAMES_MODELS)
    labels.setdefault("inclusionai/ling-2.6-flash", "Ling 2.6 Flash")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = ROOT / "runs" / f"failed-plus-issue-random-3x-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = tournament_manifest(
        benchmark="codenames-openrouter-failed-plus-issue-random",
        models=models,
        game_count=len(schedule),
        max_turns=MAX_TURNS,
        seed_prefix="mixed-failed-issue-random",
        elo_initial=1500.0,
        elo_k=32.0,
        schedule_mode="custom-failed-plus-random",
        round_size=WORKERS,
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "10000")),
        extra_fields={
            "workers": WORKERS,
            "failed_games": [game.to_dict() for game in FAILED_GAMES],
            "random_issue_games": [game.to_dict() for game in random_games],
            "previous_issue_models_requested": PREVIOUS_ISSUE_MODELS,
            "previous_issue_models_currently_on_model_list": [slug for slug in PREVIOUS_ISSUE_MODELS if slug in set(current_model_slugs())],
            "omitted_from_random_issue_games_not_on_current_model_list": [slug for slug in PREVIOUS_ISSUE_MODELS if slug not in set(current_model_slugs())],
            "random_seed": RANDOM_SEED,
        },
    )
    write_static_artifacts(output_dir=output_dir, models=models, schedule=schedule, manifest=manifest)

    provider_order = provider_order_payload(models)
    if provider_order:
        os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = json.dumps(provider_order, sort_keys=True)
    reasoning_effort = reasoning_effort_payload(models)
    if reasoning_effort:
        os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = json.dumps(reasoning_effort, sort_keys=True)

    all_model_slugs = sorted({slug for game in schedule for slug in (game.red_model, game.blue_model)})
    ratings = EloRatingSystem(models=all_model_slugs, initial=1500.0, k=32.0)
    records = record_tally(all_model_slugs)
    client = OpenRouterClient()

    round_summaries = []
    completed_or_attempted = 0
    for round_index, start in enumerate(range(0, len(schedule), WORKERS), 1):
        round_games = schedule[start : start + WORKERS]
        round_results = []
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(round_games))) as pool:
            futures = {
                pool.submit(play_game, game, output_dir=output_dir, max_turns=MAX_TURNS, labels=labels, client=client): game
                for game in round_games
            }
            for future in as_completed(futures):
                game = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "game": game.to_dict(),
                        "status": "game_error",
                        "error": str(exc)[:1000],
                        "transcript": str(output_dir / f"game-{game.game_number:03d}" / "transcript.json"),
                        "summary_path": str(output_dir / f"game-{game.game_number:03d}" / "summary.json"),
                    }
                    write_json(Path(result["summary_path"]), result)
                round_results.append(result)

        ordered_results = sorted(round_results, key=lambda item: item["game"]["game_number"])
        completed_round_games = []
        for result in ordered_results:
            if result.get("status") == "game_complete":
                completed_round_games.append(
                    {
                        "red_model": result["game"]["red_model"],
                        "blue_model": result["game"]["blue_model"],
                        "winner_model": result.get("winner_model"),
                        "game_number": result["game"]["game_number"],
                        "round_index": round_index,
                    }
                )
        round_elo_by_game = {}
        for result in ordered_results:
            if result.get("status") != "game_complete":
                continue
            entry = ratings.record_game(
                red_model=result["game"]["red_model"],
                blue_model=result["game"]["blue_model"],
                winner_model=result.get("winner_model"),
                game_number=result["game"]["game_number"],
            )
            round_elo_by_game[result["game"]["game_number"]] = entry

        for result in ordered_results:
            if result.get("status") == "game_complete":
                result["elo_update"] = round_elo_by_game[result["game"]["game_number"]]
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
                "error": result.get("error", "")[:160],
            }, sort_keys=True), flush=True)

        completed_or_attempted += len(ordered_results)
        round_summaries.append({
            "round_index": round_index,
            "game_numbers": [result["game"]["game_number"] for result in ordered_results],
            "completed_games": len(completed_round_games),
            "error_games": sum(1 for result in ordered_results if result.get("status") != "game_complete"),
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
            "status": "running" if completed_or_attempted < len(schedule) else "complete",
            "output_dir": str(output_dir),
            "scheduled_games": len(schedule),
            "completed_or_attempted_games": completed_or_attempted,
            "round_index": round_index,
            "round_count": (len(schedule) + WORKERS - 1) // WORKERS,
            "round_size": WORKERS,
            "workers": WORKERS,
            "standings": ratings.standings(),
        })

    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "scheduled_games": len(schedule)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
