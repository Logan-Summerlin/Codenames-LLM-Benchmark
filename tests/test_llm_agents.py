import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.llm_agents import LLMGuesserAgent, LLMSpymasterAgent, LLMTeam
from codenames_benchmark.boards import generate_board
from codenames_benchmark.game import Clue, GameState, Team
from codenames_benchmark.llm.base import FakeLLMClient, LLMResponse
from codenames_benchmark.protocol import build_guesser_observation, build_spymaster_observation

class LLMAgentTests(unittest.TestCase):
    def test_llm_spymaster_and_guesser_parse_structured_client_outputs(self):
        client = FakeLLMClient(['{"clue":"animal","count":1,"rationale":"test"}', '{"ranked_guesses":["apple"],"confidences":{"apple":0.7},"stop":false}'])
        game = GameState.new(generate_board(seed=1))
        spy = LLMSpymasterAgent("spy", Team.RED, "fake-model", client)
        guesser = LLMGuesserAgent("g", Team.RED, "fake-model", client)
        clue = spy.choose_clue(build_spymaster_observation(game, team=Team.RED, agent_id="spy"))
        guess = guesser.choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))
        self.assertEqual(clue.clue, "animal")
        self.assertEqual(guess.ranked_guesses, ["apple"])
        self.assertEqual(client.calls, 2)
    def test_llm_spymaster_prompt_includes_high_level_rules_and_victory_conditions(self):
        class RecordingClient:
            def __init__(self):
                self.request = None
            def complete(self, request):
                self.request = request
                return LLMResponse(raw='{"clue":"animal","count":1}', parsed={"clue":"animal","count":1}, model=request.model)

        client = RecordingClient()
        game = GameState.new(generate_board(seed=1))
        spy = LLMSpymasterAgent("spy", Team.RED, "fake-model", client)
        spy.choose_clue(build_spymaster_observation(game, team=Team.RED, agent_id="spy"))

        system_prompt = client.request.messages[0]["content"]
        self.assertIn("one-word hint/clue and a number", system_prompt)
        self.assertIn("guessers try to reveal that many words associated with the hint", system_prompt)
        self.assertIn("A team wins by revealing all of its own team color words before the opposing team reveals their color words", system_prompt)
        self.assertIn("Revealing the other team's color words grants points to the opposing team", system_prompt)
        self.assertIn("If your team guesses the assassin word, your team loses immediately", system_prompt)
        self.assertIn("Your clue cannot contain a substring that matches a word currently on the board", system_prompt)

    def test_llm_spymaster_prompt_includes_concise_json_template(self):
        class RecordingClient:
            def __init__(self):
                self.request = None
            def complete(self, request):
                self.request = request
                return LLMResponse(raw='{"clue":"animal","count":1}', parsed={"clue":"animal","count":1}, model=request.model)

        client = RecordingClient()
        game = GameState.new(generate_board(seed=1))
        spy = LLMSpymasterAgent("spy", Team.RED, "fake-model", client)
        spy.choose_clue(build_spymaster_observation(game, team=Team.RED, agent_id="spy"))

        system_prompt = client.request.messages[0]["content"]
        self.assertIn('Output exactly one JSON object, no markdown', system_prompt)
        self.assertIn('{"clue":"word","count":1,"rationale":"short reason"}', system_prompt)
        self.assertIn("Your response must follow the format of the clue template", system_prompt)

    def test_llm_guesser_prompt_includes_concise_json_template(self):
        class RecordingClient:
            def __init__(self):
                self.request = None
            def complete(self, request):
                self.request = request
                return LLMResponse(raw='{"ranked_guesses":["apple"],"confidences":{"apple":0.7},"stop":false}', parsed={"ranked_guesses":["apple"],"confidences":{"apple":0.7},"stop":False}, model=request.model)

        client = RecordingClient()
        game = GameState.new(generate_board(seed=1))
        game.give_clue(Clue("fruit", 1))
        guesser = LLMGuesserAgent("g", Team.RED, "fake-model", client)
        guesser.choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))

        system_prompt = client.request.messages[0]["content"]
        self.assertIn('Output exactly one JSON object, no markdown', system_prompt)
        self.assertIn('{"ranked_guesses":["word"],"confidences":{"word":0.8},"stop":false,"rationale":"short reason"}', system_prompt)

    def test_llm_team_has_deepseek_model_and_four_agents(self):
        team = LLMTeam.deepseek_v4_flash("deepseek-red", Team.RED, FakeLLMClient([]))
        self.assertEqual(team.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(len(team.guessers), 3)

    def test_llm_guesser_sanitizes_percentage_and_out_of_range_confidences(self):
        client = FakeLLMClient(['{"ranked_guesses":["apple","bridge","crane"],"confidences":{"apple":90,"bridge":1.2,"crane":"bad","extra":0.8},"stop":false}'])
        game = GameState.new(generate_board(seed=1))
        game.give_clue(Clue("fruit", 2))
        guesser = LLMGuesserAgent("g", Team.RED, "fake-model", client)
        guess = guesser.choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))
        self.assertEqual(guess.confidences["apple"], 0.9)
        self.assertEqual(guess.confidences["bridge"], 1.0)
        self.assertEqual(guess.confidences["crane"], 0.5)

    def test_llm_guesser_sanitizes_structured_and_stringified_guess_items(self):
        client = FakeLLMClient([
            '{"ranked_guesses":[{"word":"circle","confidence":0.91},{"guess":"baker","confidence":"75"},"{\\"word\\": \\"guitar\\", \\"confidence\\": 0.61}",{"bad":"shape"}],"confidences":{},"stop":false}'
        ])
        game = GameState.new(generate_board(seed=1))
        game.give_clue(Clue("object", 3))
        guesser = LLMGuesserAgent("g", Team.RED, "fake-model", client)
        guess = guesser.choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))
        self.assertEqual(guess.ranked_guesses, ["circle", "baker", "guitar"])
        self.assertEqual(guess.confidences["circle"], 0.91)
        self.assertEqual(guess.confidences["baker"], 0.75)
        self.assertEqual(guess.confidences["guitar"], 0.61)

    def test_llm_guesser_deduplicates_repeated_ranked_guesses(self):
        client = FakeLLMClient(['{"ranked_guesses":["bolt","bolt","apple"],"confidences":{"bolt":0.9,"apple":0.8},"stop":false}'])
        game = GameState.new(generate_board(seed=1))
        game.give_clue(Clue("object", 2))
        guesser = LLMGuesserAgent("g", Team.RED, "fake-model", client)
        guess = guesser.choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))
        self.assertEqual(guess.ranked_guesses, ["bolt", "apple"])
        self.assertEqual(guess.confidences["bolt"], 0.9)
        self.assertEqual(guess.confidences["apple"], 0.8)

if __name__ == "__main__": unittest.main()
