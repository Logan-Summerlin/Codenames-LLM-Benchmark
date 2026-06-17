#!/usr/bin/env python3
"""Configure two DeepSeek V4 Flash teams and optionally smoke-test OpenRouter.

This script never prints API keys. If OPENROUTER_API_KEY is absent, it prints a
setup-ready status without making network calls.
"""
from pathlib import Path
import os, sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.agents.llm_agents import LLMTeam
from codenames_benchmark.game import Team
from codenames_benchmark.llm.base import LLMRequest
from codenames_benchmark.llm.openrouter import DEEPSEEK_V4_FLASH_MODEL, OpenRouterClient

teams_payload = {
    "red": {"model": DEEPSEEK_V4_FLASH_MODEL, "agents": [f"deepseek-red-agent-{i}" for i in range(4)]},
    "blue": {"model": DEEPSEEK_V4_FLASH_MODEL, "agents": [f"deepseek-blue-agent-{i}" for i in range(4)]},
}
if not os.environ.get("OPENROUTER_API_KEY"):
    print(json.dumps({"status":"configured_no_api_key", "teams": teams_payload}, sort_keys=True))
    raise SystemExit(0)
client = OpenRouterClient()
red_team = LLMTeam.deepseek_v4_flash("deepseek-red", Team.RED, client)
blue_team = LLMTeam.deepseek_v4_flash("deepseek-blue", Team.BLUE, client)
request = LLMRequest(model=DEEPSEEK_V4_FLASH_MODEL, messages=[{"role":"user","content":"Return only JSON: {\"clue\":\"animal\",\"count\":1}"}], temperature=0.0, json_schema={"type":"object"})
response = client.complete(request)
print(json.dumps({"status":"api_smoke_ok", "model": response.model, "parsed_type": type(response.parsed).__name__, "red_agents": 1 + len(red_team.guessers), "blue_agents": 1 + len(blue_team.guessers)}, sort_keys=True))
