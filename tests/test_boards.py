import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.boards import (
    DEFAULT_WORDLIST_PATH,
    generate_board,
    load_word_list,
    mirror_board,
)
from codenames_benchmark.game import Identity, Team


class BoardGenerationTests(unittest.TestCase):
    def test_default_word_list_loads_unique_normalized_words(self):
        words = load_word_list(DEFAULT_WORDLIST_PATH)

        self.assertGreaterEqual(len(words), 50)
        self.assertEqual(len(words), len(set(words)))
        self.assertTrue(all(word == word.lower() for word in words))
        self.assertTrue(all(word.strip() == word for word in words))

    def test_same_seed_produces_same_board(self):
        first = generate_board(seed=12345)
        second = generate_board(seed=12345)

        self.assertEqual(first.to_dict(include_hidden=True), second.to_dict(include_hidden=True))

    def test_different_seeds_change_board_or_assignments(self):
        first = generate_board(seed=111)
        second = generate_board(seed=222)

        self.assertNotEqual(first.to_dict(include_hidden=True), second.to_dict(include_hidden=True))

    def test_generated_board_has_standard_distribution_for_red_start(self):
        board = generate_board(seed=7, starting_team=Team.RED)
        grouped = board.words_by_identity()

        self.assertEqual(len(board.words), 25)
        self.assertEqual(len(grouped[Identity.RED]), 9)
        self.assertEqual(len(grouped[Identity.BLUE]), 8)
        self.assertEqual(len(grouped[Identity.NEUTRAL]), 7)
        self.assertEqual(len(grouped[Identity.ASSASSIN]), 1)

    def test_generated_board_supports_blue_start_distribution(self):
        board = generate_board(seed=7, starting_team=Team.BLUE)
        grouped = board.words_by_identity()

        self.assertEqual(len(grouped[Identity.BLUE]), 9)
        self.assertEqual(len(grouped[Identity.RED]), 8)
        self.assertEqual(len(grouped[Identity.NEUTRAL]), 7)
        self.assertEqual(len(grouped[Identity.ASSASSIN]), 1)

    def test_mirror_board_preserves_words_and_swaps_team_identities(self):
        board = generate_board(seed=42, starting_team=Team.RED)
        mirrored = mirror_board(board)

        self.assertEqual(set(board.words), set(mirrored.words))
        for word, identity in board.words.items():
            if identity is Identity.RED:
                self.assertEqual(mirrored.identity_for(word), Identity.BLUE)
            elif identity is Identity.BLUE:
                self.assertEqual(mirrored.identity_for(word), Identity.RED)
            else:
                self.assertEqual(mirrored.identity_for(word), identity)

    def test_mirror_board_preserves_revealed_state(self):
        board = generate_board(seed=42, starting_team=Team.RED)
        word = next(iter(board.words))
        board.reveal(word)

        mirrored = mirror_board(board)

        self.assertTrue(mirrored.is_revealed(word))
        self.assertEqual(board.revealed, mirrored.revealed)

    def test_custom_word_list_normalizes_duplicates_before_sampling(self):
        words = [f"word{i}" for i in range(25)] + [" Word0 ", "WORD1"]

        board = generate_board(seed=1, words=words)

        self.assertEqual(len(board.words), 25)
        self.assertIn("word0", board.words)
        self.assertIn("word1", board.words)

    def test_word_list_must_have_enough_unique_words(self):
        with self.assertRaises(ValueError):
            generate_board(seed=1, words=["apple", "apple", "brick"])


if __name__ == "__main__":
    unittest.main()
