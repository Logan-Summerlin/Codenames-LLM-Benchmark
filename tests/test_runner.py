import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.mock import DeterministicMockTeam
from codenames_benchmark.boards import generate_board
from codenames_benchmark.game import Identity, Team
from codenames_benchmark.llm.base import LLMResponse
from codenames_benchmark.protocol import GuesserAction, SpymasterAction
from codenames_benchmark.runner import run_game, run_mirrored_matchup

class RecordingLLMClient:
    def __init__(self, guess_payload=None):
        self.requests = []
        self.guess_payload = guess_payload or {"ranked_guesses":[],"confidences":{},"stop":True}
    def complete(self, request):
        self.requests.append(request)
        if "spymaster" in request.messages[0]["content"]:
            return LLMResponse(raw='{"clue":"animal","count":1}', parsed={"clue":"animal","count":1}, model=request.model)
        return LLMResponse(raw='{}', parsed=self.guess_payload, model=request.model)

class IllegalClueSpymaster:
    agent_id = "illegal-spy"
    def choose_clue(self, observation):
        board_words = list(observation.to_dict()["board"]["words"])
        return SpymasterAction(clue=board_words[0], count=1, rationale="illegal on purpose")

class NeverCalledGuesser:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    def choose_guesses(self, observation):
        raise AssertionError("guessers should not be called after an illegal clue")

class IllegalClueTeam:
    def __init__(self, name: str, team: Team):
        self.name = name
        self.team = team
        self.spymaster = IllegalClueSpymaster()
        self.guessers = [NeverCalledGuesser(f"{name}-g{i}") for i in range(3)]

class InvalidFormatSpymaster:
    agent_id = "invalid-format-spy"
    def choose_clue(self, observation):
        return None

class InvalidFormatTeam:
    def __init__(self, name: str, team: Team):
        self.name = name
        self.team = team
        self.spymaster = InvalidFormatSpymaster()
        self.guessers = [NeverCalledGuesser(f"{name}-g{i}") for i in range(3)]

class RunnerTests(unittest.TestCase):
    def test_illegal_spymaster_clue_ends_turn_immediately(self):
        red = IllegalClueTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        record = run_game(red, blue, seed=5, max_turns=1)

        self.assertFalse(record.terminal)
        self.assertEqual(record.public_events[0]["event"], "illegal_clue")
        self.assertEqual(record.public_events[0]["team"], "red")
        self.assertEqual(record.public_events[0]["reason"], "board_word")
        self.assertEqual(record.public_events[0]["matched_word"], record.public_events[0]["clue"]["clue"])
        self.assertEqual(record.public_events[-1]["event"], "illegal_clue")

    def test_invalid_spymaster_clue_format_skips_the_turn(self):
        red = InvalidFormatTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        record = run_game(red, blue, seed=5, max_turns=1)

        self.assertFalse(record.terminal)
        self.assertEqual(record.public_events[0]["event"], "invalid_clue_format")
        self.assertEqual(record.public_events[0]["team"], "red")
        self.assertEqual(record.public_events[0]["reason"], "invalid_spymaster_output")

    def test_seeded_mock_game_is_deterministic_and_terminal(self):
        red = DeterministicMockTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        first = run_game(red, blue, seed=5, max_turns=20)
        second = run_game(red, blue, seed=5, max_turns=20)
        self.assertTrue(first.terminal)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIn(first.winner, ["red", "blue"])
        self.assertTrue(first.public_events)
    def test_mirrored_matchup_uses_same_words_and_swapped_assignments(self):
        red = DeterministicMockTeam("red", Team.RED)
        blue = DeterministicMockTeam("blue", Team.BLUE)
        pair = run_mirrored_matchup(red, blue, seed=7)
        self.assertEqual(len(pair), 2)
        self.assertEqual(set(pair[0].board["words"]), set(pair[1].board["words"]))
        self.assertNotEqual(pair[0].board["identities"], pair[1].board["identities"])
    def test_llm_runner_does_not_send_oracle_targets_to_guessers(self):
        from codenames_benchmark.agents.llm_agents import LLMTeam
        client = RecordingLLMClient()
        red = LLMTeam.deepseek_v4_flash("red-llm", Team.RED, client)
        blue = LLMTeam.deepseek_v4_flash("blue-llm", Team.BLUE, client)
        run_game(red, blue, seed=11, max_turns=1)
        guesser_payloads = [request.messages[1]["content"] for request in client.requests if "guesser" in request.messages[0]["content"]]
        self.assertTrue(guesser_payloads)
        self.assertFalse(any("oracle_targets" in payload for payload in guesser_payloads))

    def test_llm_runner_sends_compact_guesser_context(self):
        from codenames_benchmark.agents.llm_agents import LLMTeam
        client = RecordingLLMClient({"ranked_guesses": [], "confidences": {}, "stop": True})
        red = LLMTeam.deepseek_v4_flash("red-llm", Team.RED, client)
        blue = LLMTeam.deepseek_v4_flash("blue-llm", Team.BLUE, client)

        run_game(red, blue, seed=11, max_turns=2)

        guesser_requests = []
        for request in client.requests:
            try:
                payload_candidate = json.loads(request.messages[1]["content"])
            except (KeyError, json.JSONDecodeError):
                continue
            if payload_candidate.get("role") == "guesser":
                guesser_requests.append(request)
        self.assertTrue(guesser_requests)
        system_prompt = guesser_requests[0].messages[0]["content"]
        payload = json.loads(guesser_requests[0].messages[1]["content"])

        self.assertIn("You should generally base your guesses on the current clue.", system_prompt)
        self.assertNotIn("number_of_guessers", system_prompt)
        self.assertNotIn("aggregation_method", system_prompt)
        self.assertNotIn("at least 2 of the 3", system_prompt)
        self.assertTrue(payload["goal"].startswith("You are a Codenames guesser."))
        self.assertEqual(payload["guess_rule"], "A word will only be guessed if its average confidence is at least 0.70 from your team.")
        self.assertIn("current_clue", payload)
        self.assertIn("guesses_remaining", payload)
        self.assertIn("board", payload)
        self.assertIn("unrevealed", payload["board"])
        self.assertIn("revealed", payload["board"])
        self.assertIn("clues", payload)
        self.assertIn("your_team", payload["clues"])
        self.assertIn("opponent", payload["clues"])
        self.assertNotIn("identities", json.dumps(payload))
        self.assertNotIn("oracle_targets", json.dumps(payload))
    def test_llm_runner_ignores_off_board_guesses_instead_of_crashing(self):
        from codenames_benchmark.agents.llm_agents import LLMTeam
        client = RecordingLLMClient({"ranked_guesses":["not-on-board"],"confidences":{"not-on-board":0.9},"stop":False})
        red = LLMTeam.deepseek_v4_flash("red-llm", Team.RED, client)
        blue = LLMTeam.deepseek_v4_flash("blue-llm", Team.BLUE, client)
        record = run_game(red, blue, seed=11, max_turns=1)
        self.assertFalse(record.terminal)
        self.assertEqual(record.public_events[-1]["event"], "stop")

    def test_llm_runner_stops_turn_after_exhausting_aggregate_guess_sequence(self):
        from codenames_benchmark.agents.llm_agents import LLMTeam
        board = generate_board(seed=11, starting_team=Team.RED)
        red_target = next(word for word, identity in board.words.items() if identity is Identity.RED)
        client = RecordingLLMClient({"ranked_guesses":[red_target],"confidences":{red_target:0.9},"stop":False})
        red = LLMTeam.deepseek_v4_flash("red-llm", Team.RED, client)
        blue = LLMTeam.deepseek_v4_flash("blue-llm", Team.BLUE, client)
        record = run_game(red, blue, seed=11, max_turns=2)
        self.assertFalse(record.terminal)
        self.assertIn({"event":"stop","team":"red"}, record.public_events)

if __name__ == "__main__": unittest.main()
