import json, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.boards import generate_board
from codenames_benchmark.game import GameState, Team
from codenames_benchmark.protocol import build_guesser_observation

class PrivacyBoundaryTests(unittest.TestCase):
    def test_guesser_prompt_payload_has_no_hidden_identity_map(self):
        game = GameState.new(generate_board(seed=3))
        payload = json.dumps(build_guesser_observation(game, team=Team.RED, agent_id="g").to_dict())
        self.assertNotIn('"identities"', payload)
        self.assertNotIn('"assassin"', payload)
        self.assertNotIn('"hidden_remaining"', payload)

if __name__ == "__main__": unittest.main()
