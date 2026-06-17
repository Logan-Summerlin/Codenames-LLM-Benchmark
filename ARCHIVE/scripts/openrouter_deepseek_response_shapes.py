#!/usr/bin/env python3
"""Inspect abnormal OpenRouter response shapes without printing secrets."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "live-matchup-deepseek-v4-pro-vs-gpt-4.1-nano-20260604" / "diagnostics" / "deepseek-response-shapes.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

NODE_SCRIPT = r'''
const fs = require('fs');
const body = fs.readFileSync(process.env.OPENROUTER_BODY_PATH, 'utf8');
const key = process.env.OPENROUTER_API_KEY;
fetch('https://openrouter.ai/api/v1/chat/completions', {
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
  console.log(JSON.stringify({http_status: response.status, ok: response.ok, text}));
}).catch(error => {
  console.error(String(error));
  process.exit(1);
});
'''


def summarize_raw(raw: str) -> dict:
    redacted = raw.replace(os.environ.get("OPENROUTER_API_KEY", ""), "[REDACTED]")
    try:
        outer = json.loads(redacted)
    except Exception as exc:
        return {"raw_parse": "failed", "error": str(exc), "preview": redacted[:1000]}
    text = outer.get("text", "")
    inner = None
    try:
        inner = json.loads(text)
    except Exception:
        pass
    result = {
        "http_status": outer.get("http_status"),
        "ok": outer.get("ok"),
        "raw_text_preview": text[:1000],
    }
    if isinstance(inner, dict):
        result["inner_keys"] = sorted(inner.keys())
        result["inner_error"] = inner.get("error")
        choices = inner.get("choices")
        result["choices_type"] = type(choices).__name__
        result["choices_len"] = len(choices) if isinstance(choices, list) else None
        if isinstance(choices, list) and choices:
            choice = choices[0]
            result["choice_keys"] = sorted(choice.keys()) if isinstance(choice, dict) else None
            msg = choice.get("message") if isinstance(choice, dict) else None
            result["message_type"] = type(msg).__name__
            result["message_keys"] = sorted(msg.keys()) if isinstance(msg, dict) else None
            result["message_content_preview"] = str((msg or {}).get("content"))[:500] if isinstance(msg, dict) else None
            result["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
        result["usage"] = inner.get("usage")
    return result


def call(model: str, response_format: bool, max_tokens: int = 64, timeout: float = 60) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return a tiny answer."},
            {"role": "user", "content": "Return {\"result\":\"OK\"}." if response_format else "Say OK."},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
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
            return {"model": model, "response_format": response_format, "status": "node_error", "error": err[:1000]}
        result = {"model": model, "response_format": response_format, "status": "returned"}
        result.update(summarize_raw(proc.stdout))
        return result
    finally:
        try:
            os.unlink(body_path)
        except OSError:
            pass


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing")
        return 2
    tests = [
        ("deepseek/deepseek-v4-pro", False),
        ("deepseek/deepseek-v4-pro", True),
        ("deepseek/deepseek-v4-flash", False),
        ("deepseek/deepseek-v4-flash", True),
    ]
    with OUT.open("w", encoding="utf-8") as out:
        for model, fmt in tests:
            result = call(model, fmt)
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            out.write(line + "\n")
    print(f"WROTE {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
