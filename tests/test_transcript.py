import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.agents.mock import DeterministicMockTeam
from codenames_benchmark.game import Team
from codenames_benchmark.runner import run_game
from codenames_benchmark.transcript import build_transcript, write_transcript


class TranscriptTests(unittest.TestCase):
    def test_build_transcript_contains_summary_public_log_private_actions_and_turns(self):
        red = DeterministicMockTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        record = run_game(red, blue, seed=5, max_turns=2)

        transcript = build_transcript(record)

        self.assertEqual(transcript["schema_version"], 1)
        self.assertEqual(transcript["summary"]["team_red"], "red")
        self.assertEqual(transcript["summary"]["team_blue"], "blue")
        self.assertEqual(transcript["summary"]["seed"], 5)
        self.assertEqual(transcript["board"], record.board)
        self.assertEqual(transcript["public_events"], record.public_events)
        self.assertEqual(transcript["private_actions"], record.private_events)
        self.assertTrue(transcript["turns"])
        self.assertEqual(transcript["turns"][0]["team"], record.public_events[0]["team"])
        self.assertIn("clue", transcript["turns"][0])
        self.assertIn("spymaster_action", transcript["turns"][0])
        self.assertIn("guesser_actions", transcript["turns"][0])

    def test_write_transcript_creates_parent_directories_and_valid_json(self):
        red = DeterministicMockTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        record = run_game(red, blue, seed=5, max_turns=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "game-transcript.json"
            result = write_transcript(record, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(result, path)
        self.assertEqual(payload["summary"]["seed"], 5)
        self.assertEqual(payload["public_events"], record.public_events)


if __name__ == "__main__":
    unittest.main()
