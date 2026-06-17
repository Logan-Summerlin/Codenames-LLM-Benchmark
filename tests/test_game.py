import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark import Board as PublicBoard
from codenames_benchmark import generate_board as public_generate_board
from codenames_benchmark import check_clue as public_check_clue
from codenames_benchmark import SpymasterAction as PublicSpymasterAction
from codenames_benchmark import build_guesser_observation as public_build_guesser_observation
from codenames_benchmark.game import (
    Board,
    Clue,
    GamePhase,
    GameState,
    Identity,
    Team,
)


def sample_board() -> Board:
    words = {
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
    return Board(words)


class PackageSkeletonTests(unittest.TestCase):
    def test_phase_one_api_is_exported_from_package_root(self):
        self.assertIs(PublicBoard, Board)

    def test_phase_two_api_is_exported_from_package_root(self):
        self.assertTrue(callable(public_generate_board))
        self.assertTrue(callable(public_check_clue))

    def test_phase_three_api_is_exported_from_package_root(self):
        self.assertTrue(callable(PublicSpymasterAction))
        self.assertTrue(callable(public_build_guesser_observation))


class BoardModelTests(unittest.TestCase):
    def test_board_constructs_with_valid_distribution_and_serializes(self):
        board = sample_board()

        self.assertEqual(len(board.words), 25)
        self.assertEqual(board.remaining_for(Team.RED), 9)
        self.assertEqual(board.remaining_for(Team.BLUE), 8)
        self.assertEqual(board.identity_for("Apple"), Identity.RED)

        data = board.to_dict()
        self.assertEqual(data["words"]["apple"], "red")
        self.assertEqual(data["revealed"], [])

    def test_board_rejects_duplicate_or_invalid_word_sets(self):
        with self.assertRaises(ValueError):
            Board({"apple": Identity.RED})

        words = sample_board().words_by_identity()
        bad_words = {}
        for identity, identity_words in words.items():
            for word in identity_words:
                bad_words[word] = identity
        bad_words["extra"] = Identity.NEUTRAL

        with self.assertRaises(ValueError):
            Board(bad_words)

    def test_reveal_word_updates_state_and_prevents_double_reveal(self):
        board = sample_board()

        self.assertEqual(board.reveal("apple"), Identity.RED)
        self.assertTrue(board.is_revealed("APPLE"))
        self.assertEqual(board.remaining_for(Team.RED), 8)

        with self.assertRaises(ValueError):
            board.reveal("apple")


class GameStateTests(unittest.TestCase):
    def test_new_game_starts_with_red_clue_phase_and_serializes_public_state(self):
        game = GameState.new(sample_board())

        self.assertEqual(game.current_team, Team.RED)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)
        self.assertIsNone(game.winner)

        public_state = game.to_dict(include_hidden=False)
        self.assertNotIn("identities", public_state["board"])
        self.assertIn("apple", public_state["board"]["words"])

        hidden_state = game.to_dict(include_hidden=True)
        self.assertEqual(hidden_state["board"]["identities"]["apple"], "red")

    def test_give_clue_moves_to_guessing_phase_with_count_based_limit(self):
        game = GameState.new(sample_board())

        game.give_clue(Clue(word="place", count=2))

        self.assertEqual(game.phase, GamePhase.GUESSING)
        self.assertEqual(game.active_clue, Clue(word="place", count=2))
        self.assertEqual(game.guesses_remaining, 3)

    def test_normal_win_when_team_reveals_all_own_words(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("red", 8))

        for word in ["apple", "brick", "crown", "dragon", "engine", "forest", "garden", "harbor"]:
            result = game.guess(word)
            self.assertFalse(result.terminal)

        result = game.guess("island")

        self.assertTrue(result.terminal)
        self.assertEqual(result.winner, Team.RED)
        self.assertEqual(game.phase, GamePhase.TERMINAL)

    def test_assassin_guess_causes_immediate_loss(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("danger", 1))

        result = game.guess("zebra")

        self.assertTrue(result.terminal)
        self.assertEqual(result.winner, Team.BLUE)
        self.assertEqual(result.reason, "assassin")

    def test_opponent_word_reveal_switches_turn_and_can_give_opponent_win(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("royal", 1))

        result = game.guess("king")

        self.assertFalse(result.terminal)
        self.assertEqual(result.revealed_identity, Identity.BLUE)
        self.assertEqual(game.current_team, Team.BLUE)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)

        decisive = GameState.new(sample_board())
        for word in ["jacket", "ladder", "mountain", "needle", "ocean", "piano", "queen"]:
            decisive.board.reveal(word)
        decisive.give_clue(Clue("mistake", 1))

        result = decisive.guess("king")

        self.assertTrue(result.terminal)
        self.assertEqual(result.winner, Team.BLUE)
        self.assertEqual(result.reason, "opponent_completed")

    def test_neutral_word_reveal_ends_turn(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("water", 1))

        result = game.guess("river")

        self.assertFalse(result.terminal)
        self.assertEqual(result.revealed_identity, Identity.NEUTRAL)
        self.assertEqual(game.current_team, Team.BLUE)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)

    def test_voluntary_stop_and_guess_limit_end_turn(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("nature", 1))
        game.guess("apple")

        self.assertEqual(game.phase, GamePhase.GUESSING)
        game.stop_guessing()

        self.assertEqual(game.current_team, Team.BLUE)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)

        game.give_clue(Clue("royal", 1))
        game.guess("jacket")
        game.guess("king")

        self.assertEqual(game.current_team, Team.RED)
        self.assertEqual(game.phase, GamePhase.AWAITING_CLUE)

    def test_illegal_actions_raise_clear_errors(self):
        game = GameState.new(sample_board())

        with self.assertRaises(ValueError):
            game.guess("apple")

        with self.assertRaises(ValueError):
            game.give_clue(Clue("bad", -1))

        game.give_clue(Clue("fruit", 1))

        with self.assertRaises(ValueError):
            game.give_clue(Clue("again", 1))

        with self.assertRaises(ValueError):
            game.guess("not-on-board")


if __name__ == "__main__":
    unittest.main()
