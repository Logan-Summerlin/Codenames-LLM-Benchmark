import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.metrics import compute_diagnostics

class MetricsTests(unittest.TestCase):
    def test_counts_diagnostic_events(self):
        events = [{"event":"guess","team":"red","revealed_identity":"red"},{"event":"guess","team":"red","revealed_identity":"blue"},{"event":"guess","team":"blue","revealed_identity":"assassin"},{"event":"illegal_clue","team":"red"}]
        m = compute_diagnostics(events)
        self.assertEqual(m["total_guesses"], 3)
        self.assertEqual(m["opponent_hits"], 1)
        self.assertEqual(m["assassin_hits"], 1)
        self.assertEqual(m["illegal_clues"], 1)

if __name__ == "__main__": unittest.main()
