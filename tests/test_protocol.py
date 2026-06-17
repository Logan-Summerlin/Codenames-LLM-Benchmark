import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.game import Board, Clue, GameState, Identity, Team
from codenames_benchmark.protocol import (
    AggregatorAction,
    GuesserAction,
    GuesserObservation,
    SpymasterAction,
    SpymasterObservation,
    build_guesser_observation,
    build_spymaster_observation,
)


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


class ObservationPrivacyTests(unittest.TestCase):
    def test_spymaster_observation_contains_hidden_identities_and_team_context(self):
        game = GameState.new(sample_board())

        observation = build_spymaster_observation(game, team=Team.RED, agent_id="red-spy-1")
        data = observation.to_dict()

        self.assertIsInstance(observation, SpymasterObservation)
        self.assertEqual(data["role"], "spymaster")
        self.assertEqual(data["team"], "red")
        self.assertEqual(data["agent_id"], "red-spy-1")
        self.assertEqual(data["board"]["identities"]["apple"], "red")
        self.assertEqual(data["board"]["identities"]["zebra"], "assassin")
        self.assertIn("hidden_remaining", data)
        self.assertEqual(data["hidden_remaining"]["red"], 9)
        self.assertEqual(data["hidden_remaining"]["blue"], 8)

    def test_guesser_observation_omits_hidden_identities_even_in_serialized_json(self):
        game = GameState.new(sample_board())

        observation = build_guesser_observation(game, team=Team.RED, agent_id="red-guesser-1")
        data = observation.to_dict()
        serialized = json.dumps(data)

        self.assertIsInstance(observation, GuesserObservation)
        self.assertEqual(data["role"], "guesser")
        self.assertEqual(data["team"], "red")
        self.assertNotIn("identities", data["board"])
        self.assertIn("apple", data["board"]["words"])
        self.assertNotIn('"apple": "red"', serialized)
        self.assertNotIn('"zebra": "assassin"', serialized)
        self.assertNotIn("hidden_remaining", data)

    def test_guesser_observation_can_include_revealed_public_identity_only(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("fruit", 1))
        game.guess("apple")

        observation = build_guesser_observation(game, team=Team.BLUE, agent_id="blue-guesser-1")
        data = observation.to_dict()

        self.assertIn("apple", data["board"]["revealed"])
        self.assertEqual(data["revealed_identities"], {"apple": "red"})
        self.assertNotIn("brick", data["revealed_identities"])
        self.assertNotIn("zebra", data["revealed_identities"])

    def test_observation_history_is_public_and_detached_from_game_history(self):
        game = GameState.new(sample_board())
        game.give_clue(Clue("fruit", 1))
        observation = build_guesser_observation(game, team=Team.RED, agent_id="g1")

        data = observation.to_dict()
        data["history"].append({"event": "tamper"})

        self.assertEqual(len(game.history), 1)
        self.assertEqual(game.history[0]["event"], "clue")

    def test_observation_builders_reject_wrong_phase_or_team_inputs(self):
        game = GameState.new(sample_board())

        with self.assertRaises(ValueError):
            build_spymaster_observation(game, team="green", agent_id="bad")
        with self.assertRaises(ValueError):
            build_guesser_observation(game, team="green", agent_id="bad")
        with self.assertRaises(ValueError):
            build_spymaster_observation(game, team=Team.RED, agent_id="")


class ActionSchemaTests(unittest.TestCase):
    def test_spymaster_action_normalizes_and_serializes_clue(self):
        action = SpymasterAction(clue="  Animal ", count=2, rationale="connects mammals")

        self.assertEqual(action.clue, "animal")
        self.assertEqual(action.count, 2)
        self.assertEqual(action.to_clue(), Clue("animal", 2))
        self.assertEqual(
            action.to_dict(),
            {"clue": "animal", "count": 2, "rationale": "connects mammals"},
        )

    def test_spymaster_action_rejects_invalid_clues_and_counts(self):
        with self.assertRaises(ValueError):
            SpymasterAction(clue="", count=1)
        with self.assertRaises(ValueError):
            SpymasterAction(clue="animal", count=-1)
        with self.assertRaises(TypeError):
            SpymasterAction(clue="animal", count="two")

    def test_guesser_action_normalizes_ranked_guesses_and_confidences(self):
        action = GuesserAction(
            ranked_guesses=[" Apple ", "BRICK"],
            confidences={" Apple ": 0.8, "BRICK": 0.6},
            stop=False,
            rationale="apple first",
        )

        self.assertEqual(action.ranked_guesses, ["apple", "brick"])
        self.assertEqual(action.confidences, {"apple": 0.8, "brick": 0.6})
        self.assertFalse(action.stop)
        self.assertEqual(action.to_dict()["ranked_guesses"], ["apple", "brick"])

    def test_guesser_action_rejects_duplicates_empty_guesses_and_bad_confidence(self):
        with self.assertRaises(ValueError):
            GuesserAction(ranked_guesses=["apple", "APPLE"])
        with self.assertRaises(ValueError):
            GuesserAction(ranked_guesses=[""])
        with self.assertRaises(ValueError):
            GuesserAction(ranked_guesses=["apple"], confidences={"apple": 1.2})
        with self.assertRaises(ValueError):
            GuesserAction(ranked_guesses=["apple"], confidences={"brick": 0.5})

    def test_guesser_action_accepts_confidence_list_aligned_with_ranked_guesses(self):
        action = GuesserAction(ranked_guesses=[" Apple ", "BRICK"], confidences=[0.8, 0.6])

        self.assertEqual(action.ranked_guesses, ["apple", "brick"])
        self.assertEqual(action.confidences, {"apple": 0.8, "brick": 0.6})

    def test_guesser_action_rejects_confidence_list_with_wrong_length(self):
        with self.assertRaises(ValueError):
            GuesserAction(ranked_guesses=["apple", "brick"], confidences=[0.8])

    def test_guesser_action_allows_stop_with_no_guesses(self):
        action = GuesserAction(stop=True)

        self.assertTrue(action.stop)
        self.assertEqual(action.ranked_guesses, [])
        self.assertEqual(action.confidences, {})

    def test_aggregator_action_validates_sequence_and_stop_flag(self):
        action = AggregatorAction(guesses=[" Apple ", "BRICK"], stop_after=True)

        self.assertEqual(action.guesses, ["apple", "brick"])
        self.assertTrue(action.stop_after)
        self.assertEqual(action.to_dict(), {"guesses": ["apple", "brick"], "stop_after": True})

        with self.assertRaises(ValueError):
            AggregatorAction(guesses=["apple", "apple"])
        with self.assertRaises(ValueError):
            AggregatorAction(guesses=[""])


if __name__ == "__main__":
    unittest.main()
