#!/usr/bin/env python3
"""Minimal OpenRouter model-route smoke matrix. Does not print secrets."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "live-matchup-deepseek-v4-pro-vs-gpt-4.1-nano-20260604" / "diagnostics" / "model-smoke-matrix.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

NODE_SCRIPT = r'''
const fs = require('fs');
const url = 'https://openrouter.ai/api/v1/chat/completions';
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
    console.error(JSON.stringify({status: response.status, body: text.slice(0, 500)}));
    process.exit(1);
  }
  console.log(text);
}).catch(error => {
  console.error(String(error));
  process.exit(1);
});
'''


def call(model: str, *, response_format: bool, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return a tiny answer."},
            {"role": "user", "content": "Say OK as JSON with key result." if response_format else "Say OK."},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f)
        body_path = f.name
    env = dict(os.environ)
    env["OPENROUTER_BODY_PATH"] = body_path
    try:
        try:
            proc = subprocess.run(["node", "-e", NODE_SCRIPT], env=env, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return {"model": model, "response_format": response_format, "status": "timeout", "timeout": exc.timeout}
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).replace(os.environ.get("OPENROUTER_API_KEY", ""), "[REDACTED]")
            return {"model": model, "response_format": response_format, "status": "error", "error": err[:800]}
        try:
            data = json.loads(proc.stdout)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "model": model,
                "response_format": response_format,
                "status": "ok",
                "content_preview": content[:160],
                "usage": data.get("usage"),
            }
        except Exception as exc:
            return {"model": model, "response_format": response_format, "status": "parse_error", "error": str(exc)[:200]}
    finally:
        try:
            os.unlink(body_path)
        except OSError:
            pass


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing", flush=True)
        return 2
    tests = [
        ("openai/gpt-4.1-nano", False),
        ("openai/gpt-4.1-nano", True),
        ("deepseek/deepseek-v4-pro", False),
        ("deepseek/deepseek-v4-pro", True),
        ("deepseek/deepseek-v4-flash", False),
        ("deepseek/deepseek-v4-flash", True),
        ("deepseek/deepseek-chat-v3.1", False),
    ]
    with OUT.open("w", encoding="utf-8") as out:
        for model, fmt in tests:
            result = call(model, response_format=fmt, timeout=30)
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            out.write(line + "\n")
    print(f"WROTE {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
