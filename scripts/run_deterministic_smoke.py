#!/usr/bin/env python3
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.mock import DeterministicMockTeam
from codenames_benchmark.game import Team
from codenames_benchmark.runner import run_game

red = DeterministicMockTeam("deterministic-red", Team.RED)
blue = DeterministicMockTeam("deterministic-blue", Team.BLUE)
record = run_game(red, blue, seed=101, max_turns=20)
print(json.dumps({"terminal": record.terminal, "winner": record.winner, "events": len(record.public_events), "reason": record.reason}, sort_keys=True))
