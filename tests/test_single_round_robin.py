import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS, schedule_single_round_robin, tournament_single_round_robin_pairings


class SingleRoundRobinTests(unittest.TestCase):
    def test_single_round_robin_for_three_models_has_three_games(self):
        games = schedule_single_round_robin(["a", "b", "c"], seed_prefix="rr")
        self.assertEqual(len(games), 3)
        self.assertEqual([game.game_number for game in games], [1, 2, 3])
        self.assertEqual([game.round_index for game in games], [1, 2, 3])
        self.assertEqual(
            [(game.red_model, game.blue_model) for game in games],
            [("a", "b"), ("c", "a"), ("b", "c")],
        )
        self.assertEqual(games[0].seed, "rr-001-a-vs-b")

    def test_default_single_round_robin_for_eighteen_models_has_one_hundred_fifty_three_games(self):
        games = tournament_single_round_robin_pairings(seed_prefix="openrouter-codenames-single")
        self.assertEqual(len(games), len(OPENROUTER_CODENAMES_MODELS) * (len(OPENROUTER_CODENAMES_MODELS) - 1) // 2)
        self.assertEqual(len({game.seed for game in games}), len(games))
        self.assertEqual(len(games), 153)
        self.assertTrue(all(1 <= game.round_index <= len(games) for game in games))
        self.assertEqual(len(OPENROUTER_CODENAMES_MODELS), 18)

    def test_round_elo_updates_use_pre_round_ratings_for_all_games(self):
        system = EloRatingSystem(models=["a", "b", "c", "d"], initial=1500, k=32)
        entries = system.record_round(
            [
                {"red_model": "a", "blue_model": "b", "winner_model": "a", "game_number": 1, "round_index": 1},
                {"red_model": "c", "blue_model": "d", "winner_model": "d", "game_number": 2, "round_index": 1},
            ]
        )
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["red_rating_before"], 1500)
        self.assertEqual(entries[1]["red_rating_before"], 1500)
        self.assertNotEqual(system.ratings["a"], 1500)
        self.assertNotEqual(system.ratings["d"], 1500)
        self.assertEqual(len(system.history), 2)


if __name__ == "__main__":
    unittest.main()
