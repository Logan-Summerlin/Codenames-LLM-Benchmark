import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.summary import summarize_run


def _write_transcript(run_dir: Path, game_number: int, transcript: dict) -> None:
    game_dir = run_dir / f"game-{game_number:03d}"
    game_dir.mkdir(parents=True, exist_ok=True)
    (game_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")


class SummarizeRunTests(unittest.TestCase):
    def test_summarizes_outcomes_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # Red beats blue; red gives one clean clue, blue makes an opponent
            # hit and is dealt an illegal clue.
            _write_transcript(
                run_dir,
                1,
                {
                    "summary": {
                        "team_red": "model-a",
                        "team_blue": "model-b",
                        "winner": "red",
                        "terminal": True,
                        "reason": "all_words_found",
                    },
                    "public_events": [
                        {"event": "clue", "team": "red", "clue": {"word": "ocean", "count": 2}},
                        {"event": "guess", "team": "red", "revealed_identity": "red"},
                        {"event": "guess", "team": "red", "revealed_identity": "red"},
                        {"event": "illegal_clue", "team": "blue", "clue": {"word": "x", "count": 1}},
                        {"event": "clue", "team": "blue", "clue": {"word": "fire", "count": 1}},
                        {"event": "guess", "team": "blue", "revealed_identity": "red"},
                    ],
                },
            )
            # A bounded (non-terminal) game with no winner -> a tie for both.
            _write_transcript(
                run_dir,
                2,
                {
                    "summary": {
                        "team_red": "model-a",
                        "team_blue": "model-b",
                        "winner": None,
                        "terminal": False,
                        "reason": None,
                    },
                    "public_events": [
                        {"event": "clue", "team": "red", "clue": {"word": "sky", "count": 1}},
                        {"event": "guess", "team": "red", "revealed_identity": "neutral"},
                    ],
                },
            )

            summary = summarize_run(run_dir)

        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["terminal_games"], 1)
        self.assertEqual(summary["bounded_games"], 1)

        by_model = {row["model"]: row for row in summary["models"]}
        self.assertEqual(by_model["model-a"]["wins"], 1)
        self.assertEqual(by_model["model-a"]["ties"], 1)
        self.assertEqual(by_model["model-a"]["neutral_hits"], 1)
        self.assertEqual(by_model["model-b"]["losses"], 1)
        self.assertEqual(by_model["model-b"]["ties"], 1)
        self.assertEqual(by_model["model-b"]["illegal_clues"], 1)
        self.assertEqual(by_model["model-b"]["opponent_hits"], 1)
        # model-a is sorted first (more wins).
        self.assertEqual(summary["models"][0]["model"], "model-a")

    def test_empty_run_dir_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_run(Path(tmp))
        self.assertEqual(summary["games"], 0)
        self.assertEqual(summary["models"], [])


if __name__ == "__main__":
    unittest.main()
