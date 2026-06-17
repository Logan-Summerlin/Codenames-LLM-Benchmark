import json
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.tournament import schedule_round_robin
from codenames_benchmark.tournament_runner import provider_order_payload, TournamentGame, ModelSpec

class TournamentTests(unittest.TestCase):
    def test_round_robin_schedule_count_and_mirroring(self):
        games = schedule_round_robin(["a", "b", "c"], mirrored_seeds=2, seed_prefix="s")
        self.assertEqual(len(games), 3 * 2 * 2)
        self.assertEqual(games[0].model_a, "a")
        self.assertTrue(games[0].mirror_index in (0,1))

    def test_provider_order_payload_uses_live_openrouter_endpoints(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return json.dumps(self.payload)

        payload = {
            "data": {
                "endpoints": [
                    {"provider_name": "ProviderB", "status": 0, "uptime_last_1d": 98.0},
                    {"provider_name": "ProviderA", "status": 0, "uptime_last_1d": 99.0},
                    {"provider_name": "ProviderC", "status": -2, "uptime_last_1d": 100.0},
                ]
            }
        }
        model = ModelSpec("Test", "fake/model", "Fallback")
        with patch("codenames_benchmark.tournament_runner.urllib.request.urlopen", return_value=FakeResponse(payload)):
            order = provider_order_payload([model])
        self.assertEqual(order["fake/model"], ["ProviderA", "ProviderB"])

    def test_provider_order_payload_prefers_configured_provider_when_live(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return json.dumps(self.payload)

        payload = {
            "data": {
                "endpoints": [
                    {"provider_name": "ProviderA", "status": 0, "uptime_last_1d": 99.0},
                    {"provider_name": "Novita", "status": 0, "uptime_last_1d": 98.0},
                    {"provider_name": "ProviderB", "status": 0, "uptime_last_1d": 97.0},
                ]
            }
        }
        model = ModelSpec("Test", "fake/model", "Novita", "novita/bf16")
        with patch("codenames_benchmark.tournament_runner.urllib.request.urlopen", return_value=FakeResponse(payload)):
            order = provider_order_payload([model])
        self.assertEqual(order["fake/model"], ["Novita", "ProviderA", "ProviderB"])

if __name__ == "__main__": unittest.main()
