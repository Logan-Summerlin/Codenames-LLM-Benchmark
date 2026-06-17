import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.tournament import (
    OPENROUTER_CODENAMES_MODELS,
    tournament_limited_coverage_pairings,
    tournament_pairings,
    schedule_double_round_robin,
)


class DoubleRoundRobinTests(unittest.TestCase):
    def test_openrouter_codenames_model_list_reflects_the_updated_18_model_field(self):
        slugs = [model.slug for model in OPENROUTER_CODENAMES_MODELS]
        providers = {model.slug: model.provider for model in OPENROUTER_CODENAMES_MODELS}
        reasoning_efforts = {model.slug: model.reasoning_effort for model in OPENROUTER_CODENAMES_MODELS}
        self.assertNotIn("meta-llama/llama-3-70b-instruct", slugs)
        self.assertNotIn("openai/gpt-5-nano", slugs)
        self.assertNotIn("deepseek/deepseek-chat-v3.1", slugs)
        self.assertIn("meta-llama/llama-3.3-70b-instruct", slugs)
        self.assertIn("microsoft/phi-4", slugs)
        self.assertIn("meta-llama/llama-4-scout", slugs)
        self.assertIn("google/gemma-3-27b-it", slugs)
        self.assertIn("amazon/nova-lite-v1", slugs)
        self.assertIn("anthropic/claude-3-haiku", slugs)
        self.assertIn("google/gemini-2.5-flash-lite", slugs)
        self.assertIn("openai/gpt-oss-20b", slugs)
        self.assertIn("qwen/qwen3-32b", slugs)
        self.assertNotIn("qwen/qwen3.5-35b-a3b", slugs)
        self.assertNotIn("xiaomi/mimo-v2-flash", slugs)
        self.assertNotIn("z-ai/glm-4.7-flash", slugs)
        self.assertNotIn("xiaomi/mimo-v2.5", slugs)
        self.assertNotIn("bytedance-seed/seed-2.0-mini", slugs)
        self.assertNotIn("stepfun/step-3.7-flash", slugs)
        self.assertNotIn("minimax/minimax-m2.7", slugs)
        self.assertNotIn("inclusionai/ling-2.6-flash", slugs)
        self.assertNotIn("deepseek/deepseek-v4-flash", slugs)
        self.assertEqual(providers["meta-llama/llama-3.3-70b-instruct"], "Novita")
        self.assertEqual(providers["microsoft/phi-4"], "Microsoft")
        self.assertEqual(providers["meta-llama/llama-4-scout"], "Meta")
        self.assertEqual(providers["google/gemma-3-27b-it"], "Google")
        self.assertEqual(providers["amazon/nova-lite-v1"], "Amazon")
        self.assertEqual(providers["anthropic/claude-3-haiku"], "Anthropic")
        self.assertEqual(providers["google/gemini-2.5-flash-lite"], "Google")
        self.assertEqual(providers["openai/gpt-oss-20b"], "Groq")
        self.assertEqual(providers["qwen/qwen3-32b"], "Qwen")
        self.assertEqual(len(slugs), 18)
        self.assertEqual(slugs[0], "meta-llama/llama-3.3-70b-instruct")
        self.assertTrue(all(effort == "low" for effort in reasoning_efforts.values()))


    def test_tournament_pairings_visit_each_unordered_pair_twice_with_swapped_colors(self):
        games = schedule_double_round_robin(["a", "b", "c"], seed_prefix="rr")
        self.assertEqual(len(games), 6)
        self.assertEqual(
            [(game.red_model, game.blue_model) for game in games],
            [("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("b", "c"), ("c", "b")],
        )
        self.assertEqual([game.game_number for game in games], list(range(1, 7)))
        self.assertEqual(games[0].seed, "rr-001-a-vs-b")
        self.assertEqual(games[1].round_index, 2)

    def test_tournament_pairings_for_default_model_list_has_three_hundred_six_games(self):
        games = tournament_pairings(seed_prefix="openrouter-codenames")
        self.assertEqual(len(games), 18 * 17)
        self.assertEqual(len({game.seed for game in games}), len(games))

    def test_limited_coverage_schedule_covers_every_default_model(self):
        games = tournament_limited_coverage_pairings(seed_prefix="coverage")
        slugs = [model.slug for model in OPENROUTER_CODENAMES_MODELS]
        appearances = {slug: 0 for slug in slugs}
        for game in games:
            appearances[game.red_model] += 1
            appearances[game.blue_model] += 1
        self.assertEqual(len(games), 9)
        self.assertTrue(all(count >= 1 for count in appearances.values()))
        self.assertEqual(appearances[slugs[0]], 1)

    def test_elo_rating_system_updates_incrementally_and_records_history(self):
        system = EloRatingSystem(models=["a", "b"], initial=1500, k=32)
        entry = system.record_game(red_model="a", blue_model="b", winner_model="a", game_number=1)
        self.assertGreater(system.ratings["a"], 1500)
        self.assertLess(system.ratings["b"], 1500)
        self.assertEqual(entry["leader"], "a")
        self.assertEqual(len(system.history), 1)
        draw = system.record_game(red_model="a", blue_model="b", winner_model=None, game_number=2)
        self.assertEqual(draw["score_red"], 0.5)
        self.assertEqual(draw["score_blue"], 0.5)
        self.assertEqual(round(system.ratings["a"] + system.ratings["b"]), 3000)


if __name__ == "__main__":
    unittest.main()
