"""OpenRouter adapter for optional real-model smoke tests.

Never prints API keys. The adapter reads OPENROUTER_API_KEY from the environment.
"""
from __future__ import annotations
import json, os, subprocess, tempfile, time
from codenames_benchmark.llm.base import LLMRequest, LLMResponse, parse_json_response

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_V4_FLASH_MODEL = "deepseek/deepseek-v4-flash"


def _provider_order_for_model(model: str) -> list[str] | None:
    """Read optional per-model provider order from the environment.

    Expected format: ``{"model/slug": ["ProviderName"]}``. Invalid JSON is
    ignored so provider pinning cannot break otherwise-valid dry runs.
    """
    raw = os.environ.get("OPENROUTER_PROVIDER_ORDER_JSON")
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        return None
    order = mapping.get(model) if isinstance(mapping, dict) else None
    if isinstance(order, list) and all(isinstance(item, str) and item for item in order):
        return order
    return None


def _reasoning_effort_for_model(model: str) -> str | None:
    """Read optional per-model reasoning effort from the environment."""
    raw = os.environ.get("OPENROUTER_REASONING_EFFORT_JSON")
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        return None
    effort = mapping.get(model) if isinstance(mapping, dict) else None
    if isinstance(effort, str) and effort in {"low", "medium", "high"}:
        return effort
    return None

_NODE_FETCH_SCRIPT = r'''
const fs = require('fs');
const url = process.env.OPENROUTER_URL;
const body = fs.readFileSync(process.env.OPENROUTER_BODY_PATH, 'utf8');
const key = process.env.OPENROUTER_API_KEY;
fetch(url, {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ' + key,
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'codenames-llm-benchmark/0.1',
    'HTTP-Referer': 'https://localhost/codenames-llm-benchmark',
    'X-Title': 'Codenames LLM Benchmark'
  },
  body
}).then(async response => {
  const text = await response.text();
  if (!response.ok) {
    console.error(JSON.stringify({status: response.status, body: text.slice(0, 1000)}));
    process.exit(1);
  }
  console.log(text);
}).catch(error => {
  console.error(String(error));
  process.exit(1);
});
'''

class OpenRouterClient:
    def __init__(self, api_key: str | None = None, *, url: str = OPENROUTER_URL):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.url = url
        if not self.api_key: raise RuntimeError("OPENROUTER_API_KEY is not set")
    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {"model": request.model, "messages": request.messages, "temperature": request.temperature}
        max_tokens = os.environ.get("OPENROUTER_MAX_TOKENS", "10000")
        payload["max_tokens"] = int(max_tokens)
        provider_order = _provider_order_for_model(request.model)
        if provider_order:
            payload["provider"] = {"order": provider_order, "allow_fallbacks": len(provider_order) > 1}
        reasoning_effort = _reasoning_effort_for_model(request.model)
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        reasoning_enabled = os.environ.get("OPENROUTER_REASONING_ENABLED")
        if reasoning_enabled is not None and "reasoning" not in payload:
            payload["reasoning"] = {"enabled": reasoning_enabled.strip().lower() in {"1", "true", "yes", "on"}}
        data = self._post_json(payload)
        raw = data["choices"][0]["message"]["content"]
        try: parsed = parse_json_response(raw)
        except Exception: parsed = None
        return LLMResponse(raw=raw, parsed=parsed, model=request.model, usage=data.get("usage"))

    def _post_json(self, payload: dict) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as body_file:
            json.dump(payload, body_file)
            body_path = body_file.name
        try:
            attempts = int(os.environ.get("OPENROUTER_NODE_ATTEMPTS", os.environ.get("OPENROUTER_CURL_ATTEMPTS", "4")))
            timeout_seconds = float(os.environ.get("OPENROUTER_NODE_TIMEOUT_SECONDS", "20"))
            last_error = ""
            env = dict(os.environ)
            env["OPENROUTER_API_KEY"] = self.api_key
            env["OPENROUTER_URL"] = self.url
            env["OPENROUTER_BODY_PATH"] = body_path
            for attempt in range(1, attempts + 1):
                try:
                    proc = subprocess.run(["node", "-e", _NODE_FETCH_SCRIPT], env=env, text=True, capture_output=True, timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    last_error = f"node fetch timed out after {exc.timeout} seconds"
                    if attempt == attempts:
                        break
                    time.sleep(min(3 * attempt, 20))
                    continue
                if proc.returncode == 0:
                    data = json.loads(proc.stdout)
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        return data
                    err = data.get("error") if isinstance(data, dict) else None
                    if isinstance(err, dict):
                        last_error = f"OpenRouter response missing choices; error code={err.get('code')!r} message={str(err.get('message', ''))[:300]!r}"
                    else:
                        keys = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
                        last_error = f"OpenRouter response missing choices; top-level keys={keys!r}"
                    if attempt == attempts:
                        break
                    time.sleep(min(3 * attempt, 20))
                    continue
                stderr = proc.stderr.replace(self.api_key, "[REDACTED_KEY]")
                stdout = proc.stdout.replace(self.api_key, "[REDACTED_KEY]")
                last_error = stderr or stdout[:500]
                if attempt == attempts:
                    break
                time.sleep(min(3 * attempt, 20))
            raise RuntimeError(f"OpenRouter request failed via node fetch after {attempts} attempts: {last_error}")
        finally:
            try: os.unlink(body_path)
            except OSError: pass
