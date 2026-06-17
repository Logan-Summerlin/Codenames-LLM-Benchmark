"""Agent protocol schemas for Codenames benchmark participants.

The protocol layer is intentionally separate from LLM providers. It defines
privacy-safe observations and structured action objects that later adapters,
mock agents, and aggregators can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from codenames_benchmark.game import Board, Clue, GameState, Team


RoleName = str


def _coerce_team(team: Team | str) -> Team:
    if not isinstance(team, Team):
        team = Team(team)
    return team


def _normalize_word(word: str, *, field_name: str = "word") -> str:
    normalized = str(word).strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_agent_id(agent_id: str) -> str:
    normalized = str(agent_id).strip()
    if not normalized:
        raise ValueError("agent_id must be non-empty")
    return normalized


def _copy_history(game: GameState) -> list[dict[str, Any]]:
    # Event values are currently JSON-like scalars and shallow dictionaries.
    # Copy each event dictionary so callers cannot mutate game.history itself.
    return [dict(event) for event in game.history]


def _revealed_identities(game: GameState) -> dict[str, str]:
    return {
        word: game.board.identity_for(word).value
        for word in sorted(game.board.revealed)
    }


@dataclass(frozen=True)
class BaseObservation:
    """Common public fields shared by all agent observations."""

    role: RoleName
    team: Team
    agent_id: str
    board: dict[str, Any]
    current_team: Team
    phase: str
    active_clue: dict[str, Any] | None
    guesses_remaining: int
    winner: Team | None
    terminal_reason: str | None
    history: list[dict[str, Any]] = field(default_factory=list)
    revealed_identities: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "team", _coerce_team(self.team))
        object.__setattr__(self, "current_team", _coerce_team(self.current_team))
        if self.winner is not None:
            object.__setattr__(self, "winner", _coerce_team(self.winner))
        object.__setattr__(self, "agent_id", _validate_agent_id(self.agent_id))
        object.__setattr__(self, "history", [dict(event) for event in self.history])
        object.__setattr__(self, "revealed_identities", dict(self.revealed_identities))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "team": self.team.value,
            "agent_id": self.agent_id,
            "board": _copy_mapping(self.board),
            "current_team": self.current_team.value,
            "phase": self.phase,
            "active_clue": dict(self.active_clue) if self.active_clue else None,
            "guesses_remaining": self.guesses_remaining,
            "winner": self.winner.value if self.winner else None,
            "terminal_reason": self.terminal_reason,
            "history": [dict(event) for event in self.history],
            "revealed_identities": dict(self.revealed_identities),
        }


@dataclass(frozen=True)
class SpymasterObservation(BaseObservation):
    """Private observation for a spymaster; includes hidden board identities."""

    hidden_remaining: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["hidden_remaining"] = dict(self.hidden_remaining)
        return data


@dataclass(frozen=True)
class GuesserObservation(BaseObservation):
    """Public-only observation for guessers; never includes hidden identities."""

    pass


def build_spymaster_observation(
    game: GameState,
    *,
    team: Team | str,
    agent_id: str,
) -> SpymasterObservation:
    """Build a private spymaster observation from current game state."""

    team = _coerce_team(team)
    board = game.board.to_dict(include_hidden=True)
    return SpymasterObservation(
        role="spymaster",
        team=team,
        agent_id=_validate_agent_id(agent_id),
        board=board,
        current_team=game.current_team,
        phase=game.phase.value,
        active_clue=game.active_clue.to_dict() if game.active_clue else None,
        guesses_remaining=game.guesses_remaining,
        winner=game.winner,
        terminal_reason=game.terminal_reason,
        history=_copy_history(game),
        revealed_identities=_revealed_identities(game),
        hidden_remaining={
            Team.RED.value: game.board.remaining_for(Team.RED),
            Team.BLUE.value: game.board.remaining_for(Team.BLUE),
        },
    )


def build_guesser_observation(
    game: GameState,
    *,
    team: Team | str,
    agent_id: str,
) -> GuesserObservation:
    """Build a public guesser observation without hidden board identities."""

    team = _coerce_team(team)
    board = game.board.to_dict(include_hidden=False)
    return GuesserObservation(
        role="guesser",
        team=team,
        agent_id=_validate_agent_id(agent_id),
        board=board,
        current_team=game.current_team,
        phase=game.phase.value,
        active_clue=game.active_clue.to_dict() if game.active_clue else None,
        guesses_remaining=game.guesses_remaining,
        winner=game.winner,
        terminal_reason=game.terminal_reason,
        history=_copy_history(game),
        revealed_identities=_revealed_identities(game),
    )


@dataclass(frozen=True)
class SpymasterAction:
    """Structured spymaster output: clue word plus intended count."""

    clue: str
    count: int
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if self.count < 0:
            raise ValueError("count must be non-negative")
        object.__setattr__(self, "clue", _normalize_word(self.clue, field_name="clue"))
        if self.rationale is not None:
            object.__setattr__(self, "rationale", str(self.rationale))

    def to_clue(self) -> Clue:
        return Clue(self.clue, self.count)

    def to_dict(self) -> dict[str, Any]:
        return {"clue": self.clue, "count": self.count, "rationale": self.rationale}


@dataclass(frozen=True)
class GuesserAction:
    """Structured guesser output: ranked guesses, confidence, and stop advice."""

    ranked_guesses: list[str] = field(default_factory=list)
    confidences: Mapping[str, float] | Sequence[float] | None = None
    stop: bool = False
    rationale: str | None = None

    def __post_init__(self) -> None:
        normalized_guesses = _normalize_unique_words(self.ranked_guesses, field_name="guess")
        normalized_confidences = _normalize_confidences(self.confidences or {}, normalized_guesses)
        object.__setattr__(self, "ranked_guesses", normalized_guesses)
        object.__setattr__(self, "confidences", normalized_confidences)
        object.__setattr__(self, "stop", bool(self.stop))
        if self.rationale is not None:
            object.__setattr__(self, "rationale", str(self.rationale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_guesses": list(self.ranked_guesses),
            "confidences": dict(self.confidences),
            "stop": self.stop,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AggregatorAction:
    """Final deterministic team action selected from guesser proposals."""

    guesses: list[str] = field(default_factory=list)
    stop_after: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "guesses", _normalize_unique_words(self.guesses, field_name="guess"))
        object.__setattr__(self, "stop_after", bool(self.stop_after))

    def to_dict(self) -> dict[str, Any]:
        return {"guesses": list(self.guesses), "stop_after": self.stop_after}


def _normalize_unique_words(words: Iterable[str], *, field_name: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_word in words:
        word = _normalize_word(raw_word, field_name=field_name)
        if word in seen:
            raise ValueError(f"duplicate {field_name}: {word}")
        seen.add(word)
        normalized.append(word)
    return normalized


def _normalize_confidences(confidences: Mapping[str, float] | Sequence[float], guesses: list[str]) -> dict[str, float]:
    guess_set = set(guesses)
    normalized: dict[str, float] = {}
    if not isinstance(confidences, Mapping):
        if isinstance(confidences, (str, bytes)):
            raise TypeError("confidences must be a mapping or a list aligned with ranked_guesses")
        if len(confidences) != len(guesses):
            raise ValueError("confidence list must align one-to-one with ranked_guesses")
        confidences = dict(zip(guesses, confidences))
    for raw_word, raw_value in confidences.items():
        word = _normalize_word(raw_word, field_name="confidence word")
        if word not in guess_set:
            raise ValueError(f"confidence supplied for word not in ranked_guesses: {word}")
        value = float(raw_value)
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence values must be between 0 and 1")
        normalized[word] = value
    return normalized


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            copied[key] = dict(item)
        elif isinstance(item, list):
            copied[key] = list(item)
        else:
            copied[key] = item
    return copied


__all__ = [
    "AggregatorAction",
    "BaseObservation",
    "GuesserAction",
    "GuesserObservation",
    "SpymasterAction",
    "SpymasterObservation",
    "build_guesser_observation",
    "build_spymaster_observation",
]
