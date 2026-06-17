"""LLM-backed agent wrappers for Codenames teams."""
from __future__ import annotations
from dataclasses import dataclass, field
import ast
import json
from typing import Any
from codenames_benchmark.game import Team
from codenames_benchmark.llm.base import LLMRequest
from codenames_benchmark.llm.openrouter import DEEPSEEK_V4_FLASH_MODEL
from codenames_benchmark.protocol import GuesserAction, GuesserObservation, SpymasterAction, SpymasterObservation


def _messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role":"system","content":system},{"role":"user","content":json.dumps(payload, sort_keys=True)}]


GUESSER_PROMPT = " ".join(
    [
        "You are a Codenames guesser. Your spymaster gives a one-word clue and a number.",
        "The clue points to that many hidden words belonging to your team.",
        "Your job is to choose unrevealed board words that best match your team's current clue.",
        "Your team wins by revealing all of its own words before the opposing team reveals theirs.",
        "Revealing an opponent word helps the opposing team. Revealing a neutral word ends your turn.",
        "Revealing the assassin word loses the game immediately.",
        "You should generally base your guesses on the current clue.",
        "Guess only unrevealed words. Prefer direct, obvious clue matches. Avoid words tied to opponent clues.",
        "A word will only be guessed if its average confidence is at least 0.70 from your team.",
        "Use confidence >=0.70 only for words you want guessed. Set stop=true if no safe word is >=0.70 confidence.",
        "Output exactly one JSON object, no markdown:",
        '{"ranked_guesses":["word"],"confidences":{"word":0.8},"stop":false,"rationale":"short reason"}',
    ]
)


def _compact_guesser_context(observation: GuesserObservation) -> dict[str, Any]:
    data = observation.to_dict()
    active_clue = data.get("active_clue") or {}
    board = data.get("board", {})
    revealed = set(board.get("revealed", []))
    words = list(board.get("words", []))
    clue_word = active_clue.get("word")
    clue_count = active_clue.get("count")
    current_clue = f"{clue_word} {clue_count}" if clue_word is not None and clue_count is not None else None
    return {
        "role": "guesser",
        "team": data.get("team"),
        "goal": "You are a Codenames guesser. Your spymaster gives a one-word clue and a number. The clue points to that many hidden words belonging to your team. Your job is to choose unrevealed board words that best match your team's current clue. Your team wins by revealing all of its own words before the opposing team reveals theirs. Revealing an opponent word helps the opposing team. Revealing a neutral word ends your turn. Revealing the assassin word loses the game immediately. Guess only when the clue match is strong.",
        "current_clue": current_clue,
        "guesses_remaining": data.get("guesses_remaining"),
        "guess_rule": "A word will only be guessed if its average confidence is at least 0.70 from your team.",
        "stop_rule": "Use stop=true if no safe word is >=0.70 confidence.",
        "board": {
            "unrevealed": [word for word in words if word not in revealed],
            "revealed": [f"{word}={data.get('revealed_identities', {}).get(word, 'revealed')}" for word in sorted(revealed)],
        },
        "clues": _compact_clue_history(data),
        "instructions": "You should generally base your guesses on the current clue. Guess only unrevealed words. Prefer direct matches. Avoid opponent-clue words. No low-confidence guesses.",
        "return_json": {
            "ranked_guesses": ["word"],
            "confidences": {"word": 0.8},
            "stop": False,
            "rationale": "short reason",
        },
    }


def _compact_clue_history(data: dict[str, Any], *, limit_per_side: int = 6) -> dict[str, str]:
    team = data.get("team")
    active_clue = data.get("active_clue") or {}
    history = data.get("history", [])
    buckets: dict[str, list[str]] = {"your_team": [], "opponent": []}
    current: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    for event in history:
        if event.get("event") == "clue":
            if current is not None:
                entries.append(current)
            clue = event.get("clue", {})
            current = {"team": event.get("team"), "clue": clue, "guesses": []}
        elif event.get("event") == "guess" and current is not None:
            word = event.get("word")
            identity = event.get("revealed_identity")
            if word:
                current["guesses"].append(f"{word}={identity}" if identity else str(word))
        elif event.get("event") == "stop" and current is not None:
            entries.append(current)
            current = None
    if current is not None:
        clue = current.get("clue", {})
        if clue != active_clue or current.get("guesses"):
            entries.append(current)

    for entry in entries:
        clue = entry.get("clue", {})
        clue_word = clue.get("word")
        clue_count = clue.get("count")
        if clue_word is None or clue_count is None:
            continue
        side = "your_team" if entry.get("team") == team else "opponent"
        guesses = ", ".join(entry.get("guesses") or ["no guess"])
        buckets[side].append(f"{clue_word} {clue_count} -> {guesses}")
    return {side: "; ".join(values[-limit_per_side:]) if values else "none" for side, values in buckets.items()}

def _safe_count(value: Any, default: int = 1) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, count)


def _safe_spymaster_action(value: Any) -> SpymasterAction | None:
    if isinstance(value, str):
        parsed = _parse_stringified_guess(value)
        if parsed is not value:
            return _safe_spymaster_action(parsed)
        return None
    if not isinstance(value, dict):
        return None
    clue = value.get("clue")
    count = value.get("count")
    if clue is None or count is None:
        return None
    clue_text = str(clue).strip()
    if not clue_text:
        return None
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        return None
    if count_int < 0:
        return None
    return SpymasterAction(clue=clue_text, count=count_int, rationale=value.get("rationale"))

def _clean_word(value: Any) -> str | None:
    word = str(value).strip().lower()
    return word or None

def _parse_stringified_guess(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value

def _guess_item_word_and_confidence(value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        parsed = _parse_stringified_guess(value)
        if parsed is not value:
            return _guess_item_word_and_confidence(parsed)
        return _clean_word(value), None
    if isinstance(value, dict):
        for key in ("word", "guess", "name", "text"):
            if key in value:
                word = _clean_word(value[key])
                if not word:
                    return None, None
                confidence = value.get("confidence", value.get("score", value.get("probability")))
                return word, _safe_confidence(confidence) if confidence is not None else None
        return None, None
    return _clean_word(value), None

def _safe_ranked_guesses(value: Any) -> tuple[list[str], dict[str, float]]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [], {}
    guesses: list[str] = []
    embedded_confidences: dict[str, float] = {}
    seen: set[str] = set()
    for item in value:
        word, confidence = _guess_item_word_and_confidence(item)
        if not word or word in seen:
            continue
        seen.add(word)
        guesses.append(word)
        if confidence is not None:
            embedded_confidences[word] = confidence
    return guesses, embedded_confidences

def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    if confidence > 10.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return min(1.0, max(0.0, confidence))

def _safe_confidences(value: Any, guesses: list[str]) -> dict[str, float]:
    if isinstance(value, list):
        return {guess: _safe_confidence(raw) for guess, raw in zip(guesses, value)}
    if not isinstance(value, dict):
        return {}
    guess_set = {guess.strip().lower() for guess in guesses}
    sanitized: dict[str, float] = {}
    for raw_word, raw_confidence in value.items():
        word = str(raw_word).strip()
        if word.lower() not in guess_set:
            continue
        sanitized[word] = _safe_confidence(raw_confidence)
    return sanitized

@dataclass
class LLMSpymasterAgent:
    agent_id: str
    team: Team
    model: str
    client: Any
    temperature: float = 0.2
    def choose_clue(self, observation: SpymasterObservation) -> SpymasterAction | None:
        prompt = "You are a Codenames spymaster. The spymaster gives a one-word hint/clue and a number. The guessers try to reveal that many words associated with the hint. A team wins by revealing all of its own team color words before the opposing team reveals their color words. Revealing the other team's color words grants points to the opposing team. If your team guesses the assassin word, your team loses immediately. Do not reveal board words verbatim in your clue. Directly revealing a board word in your clue will be penalized with the loss of your turn. Your clue cannot contain a substring that matches a word currently on the board. Output exactly one JSON object, no markdown. Template: {\"clue\":\"word\",\"count\":1,\"rationale\":\"short reason\"}. Your response must follow the format of the clue template."
        response = self.client.complete(LLMRequest(self.model, _messages(prompt, observation.to_dict()), temperature=self.temperature, json_schema={"type":"object"}))
        return _safe_spymaster_action(response.parsed)

@dataclass
class LLMGuesserAgent:
    agent_id: str
    team: Team
    model: str
    client: Any
    temperature: float = 0.2
    def choose_guesses(self, observation: GuesserObservation) -> GuesserAction:
        response = self.client.complete(LLMRequest(self.model, _messages(GUESSER_PROMPT, _compact_guesser_context(observation)), temperature=self.temperature, json_schema={"type":"object"}))
        data = response.parsed or {}
        guesses, embedded_confidences = _safe_ranked_guesses(data.get("ranked_guesses", []))
        confidences = {**embedded_confidences, **_safe_confidences(data.get("confidences", {}), guesses)}
        return GuesserAction(ranked_guesses=guesses, confidences=confidences, stop=bool(data.get("stop", False)), rationale=data.get("rationale"))

@dataclass
class LLMTeam:
    name: str
    team: Team
    model: str
    client: Any
    spymaster: LLMSpymasterAgent = field(init=False)
    guessers: list[LLMGuesserAgent] = field(init=False)
    def __post_init__(self):
        if not isinstance(self.team, Team): self.team = Team(self.team)
        self.spymaster = LLMSpymasterAgent(f"{self.name}-spymaster", self.team, self.model, self.client)
        self.guessers = [LLMGuesserAgent(f"{self.name}-guesser-{i}", self.team, self.model, self.client) for i in range(3)]
    @classmethod
    def deepseek_v4_flash(cls, name: str, team: Team, client: Any) -> "LLMTeam":
        return cls(name=name, team=team, model=DEEPSEEK_V4_FLASH_MODEL, client=client)
