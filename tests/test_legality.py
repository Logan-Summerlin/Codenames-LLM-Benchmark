import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.game import Board, Identity
from codenames_benchmark.legality import ClueLegalityConfig, check_clue, is_legal_clue


def sample_board() -> Board:
    return Board(
        {
            "apple": Identity.RED,
            "brick": Identity.RED,
            "crown": Identity.RED,
            "dragon": Identity.RED,
            "engine": Identity.RED,
            "forest": Identity.RED,
            "garden": Identity.RED,
            "harbor": Identity.RED,
            "island": Identity.RED,
            "jacket": Identity.BLUE,
            "king": Identity.BLUE,
            "ladder": Identity.BLUE,
            "mountain": Identity.BLUE,
            "needle": Identity.BLUE,
            "ocean": Identity.BLUE,
            "piano": Identity.BLUE,
            "queen": Identity.BLUE,
            "river": Identity.NEUTRAL,
            "saturn": Identity.NEUTRAL,
            "tower": Identity.NEUTRAL,
            "umbrella": Identity.NEUTRAL,
            "valley": Identity.NEUTRAL,
            "window": Identity.NEUTRAL,
            "yacht": Identity.NEUTRAL,
            "zebra": Identity.ASSASSIN,
        }
    )


class ClueLegalityTests(unittest.TestCase):
    def test_rejects_exact_board_word_case_insensitively(self):
        result = check_clue("Apple", sample_board())

        self.assertFalse(result.legal)
        self.assertEqual(result.reason, "board_word")

    def test_rejects_plural_and_simple_suffix_variants(self):
        board = sample_board()

        self.assertEqual(check_clue("apples", board).reason, "morphological_variant")
        self.assertEqual(check_clue("forested", board).reason, "morphological_variant")
        self.assertEqual(check_clue("gardening", board).reason, "morphological_variant")

    def test_rejects_substring_traps_in_either_direction(self):
        board = sample_board()

        self.assertEqual(check_clue("app", board).reason, "substring")
        self.assertEqual(check_clue("applecart", board).reason, "substring")

    def test_rejects_multiword_clues_by_default(self):
        result = check_clue("deep sea", sample_board())

        self.assertFalse(result.legal)
        self.assertEqual(result.reason, "multiword")

    def test_rejects_hyphenated_and_underscored_clues_as_multiword(self):
        board = sample_board()

        self.assertEqual(check_clue("deep-sea", board).reason, "multiword")
        self.assertEqual(check_clue("deep_sea", board).reason, "multiword")

    def test_normalizes_repeated_whitespace_before_dictionary_lookup(self):
        config = ClueLegalityConfig(allow_multiword=True, allowed_words={"deep sea"})

        result = check_clue("  deep   sea  ", sample_board(), config=config)

        self.assertTrue(result.legal)
        self.assertEqual(result.normalized_clue, "deep sea")

    def test_empty_clues_are_rejected(self):
        with self.assertRaises(ValueError):
            check_clue("   ", sample_board())

    def test_dictionary_mode_rejects_unknown_clues(self):
        config = ClueLegalityConfig(allowed_words={"animal", "myth"})

        self.assertTrue(check_clue("animal", sample_board(), config=config).legal)
        self.assertEqual(check_clue("qzxnotaword", sample_board(), config=config).reason, "not_in_dictionary")

    def test_valid_unrelated_clue_is_accepted(self):
        result = check_clue("animal", sample_board())

        self.assertTrue(result.legal)
        self.assertIsNone(result.reason)
        self.assertTrue(is_legal_clue("animal", sample_board()))

    def test_permissive_config_can_allow_multiword_and_substrings(self):
        config = ClueLegalityConfig(allow_multiword=True, reject_substrings=False)

        self.assertTrue(check_clue("deep sea", sample_board(), config=config).legal)
        self.assertTrue(check_clue("applecart", sample_board(), config=config).legal)

    def test_exact_board_words_remain_illegal_even_with_permissive_substrings(self):
        config = ClueLegalityConfig(reject_substrings=False, reject_morphological_variants=False)

        self.assertEqual(check_clue("apple", sample_board(), config=config).reason, "board_word")


if __name__ == "__main__":
    unittest.main()
