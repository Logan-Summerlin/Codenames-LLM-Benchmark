import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.game import Board, Clue, GamePhase, GameState, Identity, Team


def red_start_board() -> Board:
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


def blue_start_board() -> Board:
    words = red_start_board().words.copy()
    words["island"] = Identity.BLUE
    return Board(words)


class BoardValidationEdgeCaseTests(unittest.TestCase):
    def test_board_normalizes_revealed_words_and_rejects_unknown_revealed_words(self):
        board = Board(red_start_board().words, revealed={" APPLE "})

        self.assertTrue(board.is_revealed("apple"))
        self.assertEqual(board.remaining_for(Team.RED), 8)

        with self.assertRaises(ValueError):
            Board(red_start_board().words, revealed={"not-on-board"})

    def test_board_rejects_duplicate_words_after_case_and_whitespace_normalization(self):
        words = red_start_board().words.copy()
        del words["apple"]
        words[" Apple "] = Identity.RED
        words["APPLE"] = Identity.RED

        with self.assertRaises(ValueError):
            Board(words)

    def test_board_accepts_string_identity_values_but_rejects_invalid_identity_values(self):
        words = {word: identity.value for word, identity in red_start_board().words.items()}
        board = Board(words)

        self.assertEqual(board.identity_for("apple"), Identity.RED)

        bad_words = words.copy()
        bad_words["apple"] = "green"
        with self.assertRaises(ValueError):
            Board(bad_words)

    def test_board_rejects_invalid_remaining_team_argument(self):
        board = red_start_board()

        with self.assertRaises(ValueError):
            board.remaining_for("green")


class GameRuleEdgeCaseTests(unittest.TestCase):
    def test_blue_start_team_can_win_by_revealing_nine_blue_words(self):
        game = GameState.new(blue_start_board(), starting_team=Team.BLUE)
        game.give_clue(Clue("blue", 8))

        for word in ["jacket", "king", "ladder", "mountain", "needle", "ocean", "piano", "island"]:
            result = game.guess(word)
            self.assertFalse(result.terminal)

        result = game.guess("queen")

        self.assertTrue(result.terminal)
        self.assertEqual(result.winner, Team.BLUE)
        self.assertEqual(result.reason, "all_words_found")

    def test_terminal_game_rejects_further_clues_guesses_and_stops(self):
        game = GameState.new(red_start_board())
        game.give_clue(Clue("danger", 1))
        game.guess("zebra")

        with self.assertRaises(ValueError):
            game.give_clue(Clue("late", 1))
        with self.assertRaises(ValueError):
            game.guess("apple")
        with self.assertRaises(ValueError):
            game.stop_guessing()

    def test_invalid_guess_does_not_consume_guess_or_mutate_revealed_state(self):
        game = GameState.new(red_start_board())
        game.give_clue(Clue("fruit", 1))
        before_remaining = game.guesses_remaining
        before_revealed = set(game.board.revealed)

        with self.assertRaises(ValueError):
            game.guess("not-on-board")

        self.assertEqual(game.guesses_remaining, before_remaining)
        self.assertEqual(game.board.revealed, before_revealed)
        self.assertEqual(game.phase, GamePhase.GUESSING)

    def test_guess_history_records_acting_team_before_turn_switch(self):
        game = GameState.new(red_start_board())
        game.give_clue(Clue("water", 1))

        game.guess("river")

        self.assertEqual(game.current_team, Team.BLUE)
        self.assertEqual(game.history[-1]["event"], "guess")
        self.assertEqual(game.history[-1]["team"], "red")
        self.assertEqual(game.history[-1]["revealed_identity"], "neutral")

    def test_public_serialization_history_is_detached_from_internal_history_list(self):
        game = GameState.new(red_start_board())
        game.give_clue(Clue("fruit", 1))

        state = game.to_dict()
        state["history"].append({"event": "tamper"})

        self.assertEqual(len(game.history), 1)
        self.assertNotEqual(game.history[-1]["event"], "tamper")

    def test_zero_count_clue_allows_one_guess_then_ends_turn(self):
        game = GameState.new(red_start_board())
        game.give_clue(Clue("avoid", 0))

        self.assertEqual(game.guesses_remaining, 1)
        game.guess("apple")

        self.assertEqual(game.current_team, Team.BLUE)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)

    def test_game_new_accepts_string_team_but_rejects_invalid_team(self):
        game = GameState.new(red_start_board(), starting_team="blue")
        self.assertEqual(game.current_team, Team.BLUE)

        with self.assertRaises(ValueError):
            GameState.new(red_start_board(), starting_team="green")


if __name__ == "__main__":
    unittest.main()
