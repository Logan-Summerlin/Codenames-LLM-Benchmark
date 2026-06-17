import tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.board_suites import load_board_suite
from codenames_benchmark.boards import generate_board

class BoardSuiteTests(unittest.TestCase):
    def test_loads_valid_jsonl_board_suite(self):
        board = generate_board(seed=11).to_dict(include_hidden=True)["identities"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"suite.jsonl"
            import json
            p.write_text(json.dumps({"name":"x","identities":board})+"\n")
            loaded = load_board_suite(p)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(loaded[0].words), 25)

if __name__ == "__main__": unittest.main()
