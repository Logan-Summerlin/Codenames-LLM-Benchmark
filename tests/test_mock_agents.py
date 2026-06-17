import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.mock import DeterministicMockTeam, OracleGuesserAgent, OracleSpymasterAgent
from codenames_benchmark.boards import generate_board
from codenames_benchmark.game import GameState, Team
from codenames_benchmark.protocol import build_spymaster_observation, build_guesser_observation

class MockAgentTests(unittest.TestCase):
    def test_mock_team_has_one_spymaster_and_three_guessers(self):
        team = DeterministicMockTeam("mock-red", Team.RED)
        self.assertEqual(len(team.guessers), 3)
        self.assertIsInstance(team.spymaster, OracleSpymasterAgent)
    def test_oracle_agents_emit_valid_protocol_actions(self):
        board = generate_board(seed=1)
        game = GameState.new(board)
        team = DeterministicMockTeam("mock-red", Team.RED)
        clue = team.spymaster.choose_clue(build_spymaster_observation(game, team=Team.RED, agent_id="s"))
        self.assertGreaterEqual(clue.count, 1)
        guesses = team.guessers[0].choose_guesses(build_guesser_observation(game, team=Team.RED, agent_id="g"))
        self.assertTrue(guesses.ranked_guesses)
        self.assertIsInstance(team.guessers[0], OracleGuesserAgent)

if __name__ == "__main__": unittest.main()
