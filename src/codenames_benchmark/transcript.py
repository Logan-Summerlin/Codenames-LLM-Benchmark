"""Transcript export helpers for Codenames benchmark game records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codenames_benchmark.runner import GameRecord


def build_transcript(record: GameRecord) -> dict[str, Any]:
    """Build a JSON-serializable transcript from a completed or bounded game record.

    The transcript deliberately uses already-recorded game artifacts: public table
    events, private agent actions, and the final hidden board. It does not include
    provider credentials or raw transport metadata.
    """
    return {
        "schema_version": 1,
        "summary": {
            "team_red": record.team_red,
            "team_blue": record.team_blue,
            "seed": record.seed,
            "winner": record.winner,
            "terminal": record.terminal,
            "reason": record.reason,
            "public_event_count": len(record.public_events),
            "private_action_count": len(record.private_events),
        },
        "board": dict(record.board),
        "public_events": [dict(event) for event in record.public_events],
        "private_actions": [dict(event) for event in record.private_events],
        "turns": _turns_from_events(record.public_events, record.private_events),
    }


def write_transcript(record: GameRecord, path: str | Path) -> Path:
    """Write a transcript JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_transcript(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _turns_from_events(public_events: list[dict[str, Any]], private_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    private_index = 0
    for event in public_events:
        if event.get("event") == "clue":
            team = event.get("team")
            turn = {
                "team": team,
                "clue": dict(event.get("clue", {})),
                "spymaster_action": None,
                "guesser_actions": [],
                "reveals": [],
                "stopped": False,
            }
            while private_index < len(private_events) and private_events[private_index].get("team") == team:
                private_event = dict(private_events[private_index])
                private_index += 1
                if private_event.get("event") == "spymaster_action" and turn["spymaster_action"] is None:
                    turn["spymaster_action"] = private_event
                elif private_event.get("event") == "guesser_action":
                    turn["guesser_actions"].append(private_event)
                else:
                    break
            turns.append(turn)
        elif event.get("event") == "guess" and turns:
            turns[-1]["reveals"].append(dict(event))
        elif event.get("event") == "stop" and turns:
            turns[-1]["stopped"] = True
    return turns


__all__ = ["build_transcript", "write_transcript"]
