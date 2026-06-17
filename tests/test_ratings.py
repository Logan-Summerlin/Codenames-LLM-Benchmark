import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.ratings import elo_ratings

class RatingsTests(unittest.TestCase):
    def test_winner_rating_increases(self):
        ratings = elo_ratings([("a", "b", "a"), ("a", "b", "a")])
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertEqual(round(ratings["a"] + ratings["b"]), 3000)

if __name__ == "__main__": unittest.main()
