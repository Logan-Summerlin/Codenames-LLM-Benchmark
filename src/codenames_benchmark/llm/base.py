"""Provider-neutral LLM client primitives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json, re

@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.2
    json_schema: dict[str, Any] | None = None

@dataclass(frozen=True)
class LLMResponse:
    raw: str
    parsed: Any | None
    model: str
    usage: dict[str, Any] | None = None

def parse_json_response(text: str) -> Any:
    try: return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m: raise
        return json.loads(m.group(0))

class FakeLLMClient:
    def __init__(self, responses: list[str]): self.responses=list(responses); self.calls=0
    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.responses: raise RuntimeError("fake response queue exhausted")
        self.calls += 1
        raw = self.responses.pop(0)
        try: parsed = parse_json_response(raw)
        except Exception: parsed = None
        return LLMResponse(raw=raw, parsed=parsed, model=request.model, usage={"fake": True})
