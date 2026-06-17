import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codenames_benchmark.llm.base import LLMRequest
from codenames_benchmark.llm.openrouter import OpenRouterClient


class RecordingOpenRouterClient(OpenRouterClient):
    def __init__(self):
        super().__init__(api_key="test-key")
        self.payload = None

    def _post_json(self, payload):
        self.payload = payload
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {}}


class OpenRouterTests(unittest.TestCase):
    def test_default_max_tokens_is_10000(self):
        old_value = os.environ.pop("OPENROUTER_MAX_TOKENS", None)
        try:
            client = RecordingOpenRouterClient()
            client.complete(LLMRequest(model="fake-model", messages=[{"role": "user", "content": "Return json."}], json_schema={"type": "object"}))
        finally:
            if old_value is not None:
                os.environ["OPENROUTER_MAX_TOKENS"] = old_value

        self.assertEqual(client.payload["max_tokens"], 10000)

    def test_env_max_tokens_overrides_default(self):
        old_value = os.environ.get("OPENROUTER_MAX_TOKENS")
        os.environ["OPENROUTER_MAX_TOKENS"] = "512"
        try:
            client = RecordingOpenRouterClient()
            client.complete(LLMRequest(model="fake-model", messages=[{"role": "user", "content": "Return json."}], json_schema={"type": "object"}))
        finally:
            if old_value is None:
                os.environ.pop("OPENROUTER_MAX_TOKENS", None)
            else:
                os.environ["OPENROUTER_MAX_TOKENS"] = old_value

        self.assertEqual(client.payload["max_tokens"], 512)

    def test_provider_order_json_pins_model_provider(self):
        old_value = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
        os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = '{"fake-model": ["DeepInfra", "Novita"]}'
        try:
            client = RecordingOpenRouterClient()
            client.complete(LLMRequest(model="fake-model", messages=[{"role": "user", "content": "Return json."}], json_schema={"type": "object"}))
        finally:
            if old_value is None:
                os.environ.pop("OPENROUTER_PROVIDER_ORDER_JSON", None)
            else:
                os.environ["OPENROUTER_PROVIDER_ORDER_JSON"] = old_value

        self.assertEqual(client.payload["provider"], {"order": ["DeepInfra", "Novita"], "allow_fallbacks": True})

    def test_reasoning_effort_json_sets_per_model_effort(self):
        old_value = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
        os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = '{"fake-model": "low"}'
        try:
            client = RecordingOpenRouterClient()
            client.complete(LLMRequest(model="fake-model", messages=[{"role": "user", "content": "Return json."}], json_schema={"type": "object"}))
        finally:
            if old_value is None:
                os.environ.pop("OPENROUTER_REASONING_EFFORT_JSON", None)
            else:
                os.environ["OPENROUTER_REASONING_EFFORT_JSON"] = old_value

        self.assertEqual(client.payload["reasoning"], {"effort": "low"})

    def test_json_schema_marker_does_not_force_openrouter_response_format(self):
        client = RecordingOpenRouterClient()
        client.complete(
            LLMRequest(
                model="fake-model",
                messages=[{"role": "user", "content": "Return only json."}],
                json_schema={"type": "object"},
            )
        )

        self.assertNotIn("response_format", client.payload)


    def test_default_retry_attempts_is_four_total_calls(self):
        client = OpenRouterClient(api_key="test-key")
        with patch("codenames_benchmark.llm.openrouter.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["node"], timeout=20)), patch("codenames_benchmark.llm.openrouter.time.sleep", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete(LLMRequest(model="fake-model", messages=[{"role": "user", "content": "Return json."}], json_schema={"type": "object"}))

        self.assertIn("after 4 attempts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()