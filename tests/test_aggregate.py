import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.aggregate import aggregate_guesser_actions
from codenames_benchmark.protocol import AggregatorAction, GuesserAction

class AggregateTests(unittest.TestCase):
    def test_consensus_orders_by_votes_then_confidence(self):
        actions = [
            GuesserAction(["apple", "brick"], {"apple": .7, "brick": .4}),
            GuesserAction(["apple", "crown"], {"apple": .6, "crown": .9}),
            GuesserAction(["brick", "apple"], {"brick": .8, "apple": .5}),
        ]
        result = aggregate_guesser_actions(
            actions,
            guesses_remaining=2,
            min_consensus_votes=1,
            confidence_threshold=0.0,
        )
        self.assertIsInstance(result, AggregatorAction)
        self.assertEqual(result.guesses, ["apple", "brick"])
    def test_stop_when_majority_recommends_stop_without_consensus(self):
        actions = [GuesserAction(stop=True), GuesserAction(stop=True), GuesserAction(["apple"])]
        result = aggregate_guesser_actions(actions, guesses_remaining=2)
        self.assertEqual(result.guesses, [])
        self.assertTrue(result.stop_after)
    def test_majority_stop_allows_multiple_high_confidence_consensus_guesses(self):
        actions = [
            GuesserAction(["honey", "orange", "tiger"], {"honey": .92, "orange": .88, "tiger": .65}, stop=True),
            GuesserAction(["honey", "orange", "jungle"], {"honey": .90, "orange": .86, "jungle": .80}, stop=True),
            GuesserAction(["honey", "orange", "jungle"], {"honey": .91, "orange": .84, "jungle": .82}, stop=False),
        ]
        result = aggregate_guesser_actions(actions, guesses_remaining=2)
        self.assertEqual(result.guesses, ["honey", "orange"])
        self.assertTrue(result.stop_after)
    def test_default_policy_filters_single_vote_and_low_confidence_candidates(self):
        actions = [
            GuesserAction(["apple", "brick", "crown"], {"apple": .9, "brick": .65, "crown": .95}),
            GuesserAction(["apple", "brick"], {"apple": .8, "brick": .63}),
            GuesserAction(["apple"], {"apple": .85}),
        ]
        result = aggregate_guesser_actions(actions, guesses_remaining=3)
        self.assertEqual(result.guesses, ["apple"])
        self.assertFalse(result.stop_after)
    def test_tie_breaks_by_average_confidence_then_word(self):
        actions = [GuesserAction(["brick"], {"brick": .5}), GuesserAction(["apple"], {"apple": .5})]
        self.assertEqual(
            aggregate_guesser_actions(
                actions,
                guesses_remaining=1,
                min_consensus_votes=1,
                confidence_threshold=0.0,
            ).guesses,
            ["apple"],
        )
    def test_excludes_revealed_or_unavailable_words(self):
        actions = [GuesserAction(["apple", "brick"], {"apple": .9, "brick": .9}), GuesserAction(["apple", "brick"], {"apple": .8, "brick": .8})]
        self.assertEqual(aggregate_guesser_actions(actions, guesses_remaining=2, unavailable_words={"apple"}).guesses, ["brick"])

if __name__ == "__main__": unittest.main()
