import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.llm.base import FakeLLMClient, LLMRequest, parse_json_response

class LLMBaseTests(unittest.TestCase):
    def test_fake_client_returns_raw_and_parsed_json(self):
        client = FakeLLMClient(['{"clue":"animal","count":1}'])
        response = client.complete(LLMRequest(model="fake", messages=[{"role":"user","content":"x"}]))
        self.assertEqual(response.parsed["clue"], "animal")
        self.assertEqual(client.calls, 1)
    def test_invalid_json_can_be_repaired_by_extracting_object(self):
        self.assertEqual(parse_json_response('text {"a": 1} trailing'), {"a": 1})
    def test_fake_client_raises_on_exhaustion(self):
        with self.assertRaises(RuntimeError): FakeLLMClient([]).complete(LLMRequest("m", []))

if __name__ == "__main__": unittest.main()
