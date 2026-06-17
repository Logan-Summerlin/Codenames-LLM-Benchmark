"""Aggregate a tournament run directory into per-model diagnostics.

This turns a directory of ``game-XXX/transcript.json`` files into a compact,
interpretable summary: per-model win/loss/tie records, terminal-vs-bounded game
counts, and rate-normalized legality and safety diagnostics (illegal clues,
assassin/neutral/opponent hits, clue efficiency). It exists so live benchmark
runs can be trusted before scaling to larger tournaments.

The aggregator only reads already-written transcript artifacts. It performs no
provider calls and imports no provider code.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from codenames_benchmark.metrics import compute_diagnostics


def load_transcripts(run_dir: Path) -> list[dict[str, Any]]:
    """Load every ``game-*/transcript.json`` under a run directory."""
    transcripts: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("game-*/transcript.json")):
        try:
            transcripts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return transcripts


def _blank_model_row() -> dict[str, Any]:
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "terminal_games": 0,
        "bounded_games": 0,
        "clues": 0,
        "total_guesses": 0,
        "illegal_clues": 0,
        "opponent_hits": 0,
        "neutral_hits": 0,
        "assassin_hits": 0,
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize a run directory into per-model outcome and diagnostic rates."""
    transcripts = load_transcripts(run_dir)
    models: dict[str, dict[str, Any]] = defaultdict(_blank_model_row)
    terminal_games = 0
    bounded_games = 0

    for transcript in transcripts:
        summary = transcript.get("summary", {})
        red = summary.get("team_red")
        blue = summary.get("team_blue")
        winner = summary.get("winner")  # "red" | "blue" | None
        terminal = bool(summary.get("terminal"))
        if terminal:
            terminal_games += 1
        else:
            bounded_games += 1

        # Diagnostics are attributed to the side that produced each event, so
        # we split the public event log by team before tallying.
        per_team = _diagnostics_by_team(transcript.get("public_events", []))

        for color, model in (("red", red), ("blue", blue)):
            if not model:
                continue
            row = models[model]
            row["games"] += 1
            if terminal:
                row["terminal_games"] += 1
            else:
                row["bounded_games"] += 1
            if winner == color:
                row["wins"] += 1
            elif winner in ("red", "blue"):
                row["losses"] += 1
            else:
                row["ties"] += 1
            side = per_team.get(color, {})
            for key in ("clues", "total_guesses", "illegal_clues", "opponent_hits", "neutral_hits", "assassin_hits"):
                row[key] += side.get(key, 0)

    leaderboard = [_finalize_model_row(model, row) for model, row in models.items()]
    leaderboard.sort(key=lambda r: (-r["wins"], r["losses"], -r["win_rate"], r["model"]))

    return {
        "run_dir": str(run_dir),
        "games": len(transcripts),
        "terminal_games": terminal_games,
        "bounded_games": bounded_games,
        "models": leaderboard,
    }


def _diagnostics_by_team(events: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        team = event.get("team")
        if team in ("red", "blue"):
            by_team[team].append(event)
    return {team: compute_diagnostics(team_events) for team, team_events in by_team.items()}


def _finalize_model_row(model: str, row: dict[str, Any]) -> dict[str, Any]:
    games = row["games"]
    clues = row["clues"]
    guesses = row["total_guesses"]
    return {
        "model": model,
        "games": games,
        "wins": row["wins"],
        "losses": row["losses"],
        "ties": row["ties"],
        "win_rate": round((row["wins"] + 0.5 * row["ties"]) / games, 4) if games else 0.0,
        "terminal_games": row["terminal_games"],
        "bounded_games": row["bounded_games"],
        "clues": clues,
        "illegal_clues": row["illegal_clues"],
        "illegal_clue_rate": round(row["illegal_clues"] / clues, 4) if clues else 0.0,
        "assassin_hits": row["assassin_hits"],
        "neutral_hits": row["neutral_hits"],
        "opponent_hits": row["opponent_hits"],
        "clue_efficiency": round(guesses / clues, 4) if clues else 0.0,
    }


__all__ = ["load_transcripts", "summarize_run"]
