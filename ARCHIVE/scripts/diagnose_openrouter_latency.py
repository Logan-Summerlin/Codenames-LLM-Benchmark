#!/usr/bin/env python3
"""Redacted OpenRouter latency diagnostics for Codenames benchmark payloads.

Reads OPENROUTER_API_KEY from environment, or from stdin when not set.
Does not print the key, prompts, hidden board identities, or raw model output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codenames_benchmark.agents.llm_agents import _messages
from codenames_benchmark.boards import generate_board
from codenames_benchmark.game import Clue, GameState, Team
from codenames_benchmark.protocol import build_guesser_observation, build_spymaster_observation

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SPY_PROMPT = "You are a Codenames spymaster. The spymaster gives a one-word hint/clue and a number. The guessers try to reveal that many words associated with the hint. A team wins by revealing all of its own team color words before the opposing team reveals their color words. Revealing the other team's color words grants points to the opposing team. If your team guesses the assassin word, your team loses immediately. Return only json with clue, count, and optional rationale. Do not reveal board words verbatim in your clue. Directly revealing a board word in your clue will be penalized with the loss of your turn. Your clue cannot contain a substring that matches a word currently on the board."
GUESS_PROMPT = "You are a Codenames guesser. Return only json with ranked_guesses, confidences, stop, and optional rationale. Use only public observation fields."
FAST_SPY_PROMPT = "Codenames spymaster. Output final json only: {\"clue\":\"oneword\",\"count\":1}. No rationale. No markdown. Do not think step by step. Legal clue: one word, not a board word, not a substring of a board word."
FAST_GUESS_PROMPT = "Codenames guesser. Output final json only: {\"ranked_guesses\":[\"word\"],\"confidences\":{\"word\":0.8},\"stop\":false}. No rationale. No markdown. Use only public fields."


def read_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()
    raise SystemExit("OPENROUTER_API_KEY is not set and no key was provided on stdin")


def build_messages(shape: str) -> list[dict[str, str]]:
    if shape == "tiny_plain":
        return [{"role": "user", "content": "Reply with the single word: pong"}]
    if shape == "tiny_json":
        return [{"role": "system", "content": "Return only json."}, {"role": "user", "content": '{"task":"return {\\"ok\\":true}"}'}]
    board = generate_board(seed="diag-60606", starting_team=Team.RED)
    game = GameState.new(board, starting_team=Team.RED)
    if shape == "benchmark_spymaster":
        obs = build_spymaster_observation(game, team=Team.RED, agent_id="diag-spymaster")
        return _messages(SPY_PROMPT, obs.to_dict())
    if shape == "fast_spymaster":
        obs = build_spymaster_observation(game, team=Team.RED, agent_id="diag-spymaster")
        return _messages(FAST_SPY_PROMPT, obs.to_dict())
    if shape == "benchmark_guesser":
        game.give_clue(Clue("animal", 2))
        obs = build_guesser_observation(game, team=Team.RED, agent_id="diag-guesser")
        return _messages(GUESS_PROMPT, obs.to_dict())
    if shape == "fast_guesser":
        game.give_clue(Clue("animal", 2))
        obs = build_guesser_observation(game, team=Team.RED, agent_id="diag-guesser")
        return _messages(FAST_GUESS_PROMPT, obs.to_dict())
    raise ValueError(shape)


def apply_reasoning_mode(payload: dict[str, Any], mode: str) -> None:
    if mode == "default":
        return
    if mode == "include_false":
        payload["include_reasoning"] = False
        return
    if mode == "disabled":
        payload["reasoning"] = {"enabled": False}
        return
    if mode == "exclude":
        payload["reasoning"] = {"exclude": True}
        return
    if mode == "effort_low":
        payload["reasoning"] = {"effort": "low"}
        return
    if mode == "max32":
        payload["reasoning"] = {"max_tokens": 32}
        return
    if mode == "max128":
        payload["reasoning"] = {"max_tokens": 128}
        return
    raise ValueError(mode)


def post_once(key: str, model: str, shape: str, timeout: float, max_tokens: int | None, response_format: bool, reasoning_mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_messages(shape),
        "temperature": 0.2,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    apply_reasoning_mode(payload, reasoning_mode)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "codenames-llm-benchmark/latency-diagnostic",
            "HTTP-Referer": "https://localhost/codenames-llm-benchmark",
            "X-Title": "Codenames LLM Benchmark Diagnostics",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            data = json.loads(text)
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            parsed_ok = False
            if content:
                try:
                    json.loads(content)
                    parsed_ok = True
                except Exception:
                    parsed_ok = False
            return {
                "ok": True,
                "elapsed_s": round(elapsed, 3),
                "http_status": resp.status,
                "finish_reason": choice.get("finish_reason"),
                "content_chars": len(content),
                "parsed_json": parsed_ok,
                "usage": data.get("usage"),
            }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        raw = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "elapsed_s": round(elapsed, 3), "http_status": exc.code, "error_preview": raw.replace(key, "[REDACTED_KEY]")}
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {"ok": False, "elapsed_s": round(elapsed, 3), "error_type": type(exc).__name__, "error_preview": str(exc).replace(key, "[REDACTED_KEY]")[:500]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--shapes", nargs="+", default=["tiny_plain", "tiny_json", "benchmark_spymaster", "benchmark_guesser"])
    parser.add_argument("--timeouts", nargs="+", type=float, default=[20.0])
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[256, 768])
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--reasoning-modes", nargs="+", default=["default"], choices=["default", "include_false", "disabled", "exclude", "effort_low", "max32", "max128"])
    args = parser.parse_args()
    key = read_key()
    if not key:
        raise SystemExit("empty OpenRouter API key")
    rows = []
    for model in args.models:
        for shape in args.shapes:
            for timeout in args.timeouts:
                for cap in args.max_tokens:
                    for reasoning_mode in args.reasoning_modes:
                        row = {"model": model, "shape": shape, "timeout_s": timeout, "max_tokens": cap, "response_format": not args.no_response_format, "reasoning_mode": reasoning_mode}
                        result = post_once(key, model, shape, timeout, cap, not args.no_response_format, reasoning_mode)
                        row.update(result)
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
    ok_count = sum(1 for row in rows if row.get("ok"))
    print(json.dumps({"summary": {"total": len(rows), "ok": ok_count, "failed": len(rows) - ok_count}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
