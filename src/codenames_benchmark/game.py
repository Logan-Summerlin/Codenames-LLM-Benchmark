"""Deterministic Codenames-style game engine primitives.

This module intentionally contains no LLM logic. It models only the board,
hidden identities, revealed state, turn progression, and terminal conditions
needed by the benchmark simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Team(str, Enum):
    """Playable Codenames teams."""

    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> "Team":
        return Team.BLUE if self is Team.RED else Team.RED


class Identity(str, Enum):
    """Hidden identity assigned to a board word."""

    RED = "red"
    BLUE = "blue"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"

    @classmethod
    def for_team(cls, team: Team) -> "Identity":
        return cls.RED if team is Team.RED else cls.BLUE

    def as_team(self) -> Team | None:
        if self is Identity.RED:
            return Team.RED
        if self is Identity.BLUE:
            return Team.BLUE
        return None


class GamePhase(str, Enum):
    """High-level game phase."""

    AWAITING_CLUE = "awaiting_clue"
    GUESSING = "guessing"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class Clue:
    """Public clue provided by the active spymaster."""

    word: str
    count: int

    def __post_init__(self) -> None:
        normalized = self.word.strip().lower()
        if not normalized:
            raise ValueError("clue word must be non-empty")
        if self.count < 0:
            raise ValueError("clue count must be non-negative")
        object.__setattr__(self, "word", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "count": self.count}


@dataclass(frozen=True)
class GuessResult:
    """Result of revealing a guessed word."""

    word: str
    revealed_identity: Identity
    terminal: bool = False
    winner: Team | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "revealed_identity": self.revealed_identity.value,
            "terminal": self.terminal,
            "winner": self.winner.value if self.winner else None,
            "reason": self.reason,
        }


@dataclass
class Board:
    """A 25-word Codenames board plus reveal state."""

    words: dict[str, Identity]
    revealed: set[str] = field(default_factory=set)

    RED_START_COUNTS = {
        Identity.RED: 9,
        Identity.BLUE: 8,
        Identity.NEUTRAL: 7,
        Identity.ASSASSIN: 1,
    }
    BLUE_START_COUNTS = {
        Identity.RED: 8,
        Identity.BLUE: 9,
        Identity.NEUTRAL: 7,
        Identity.ASSASSIN: 1,
    }
    VALID_COUNTS = (RED_START_COUNTS, BLUE_START_COUNTS)

    def __post_init__(self) -> None:
        normalized: dict[str, Identity] = {}
        for raw_word, identity in self.words.items():
            word = self._normalize_word(raw_word)
            if word in normalized:
                raise ValueError(f"duplicate board word after normalization: {word}")
            if not isinstance(identity, Identity):
                try:
                    identity = Identity(identity)
                except ValueError as exc:
                    raise ValueError(f"invalid identity for {word}: {identity!r}") from exc
            normalized[word] = identity

        if len(normalized) != 25:
            raise ValueError("a Codenames board must contain exactly 25 unique words")

        counts = {identity: 0 for identity in Identity}
        for identity in normalized.values():
            counts[identity] += 1
        if counts not in self.VALID_COUNTS:
            expected = [
                {identity.value: count for identity, count in valid_counts.items()}
                for valid_counts in self.VALID_COUNTS
            ]
            actual = {identity.value: count for identity, count in counts.items()}
            raise ValueError(f"invalid identity distribution: expected one of {expected}, got {actual}")

        normalized_revealed = {self._normalize_word(word) for word in self.revealed}
        unknown_revealed = normalized_revealed.difference(normalized)
        if unknown_revealed:
            raise ValueError(f"revealed words are not on board: {sorted(unknown_revealed)}")

        self.words = normalized
        self.revealed = normalized_revealed

    @staticmethod
    def _normalize_word(word: str) -> str:
        normalized = str(word).strip().lower()
        if not normalized:
            raise ValueError("board words must be non-empty")
        return normalized

    def identity_for(self, word: str) -> Identity:
        normalized = self._normalize_word(word)
        try:
            return self.words[normalized]
        except KeyError as exc:
            raise ValueError(f"word is not on board: {normalized}") from exc

    def is_revealed(self, word: str) -> bool:
        return self._normalize_word(word) in self.revealed

    def reveal(self, word: str) -> Identity:
        normalized = self._normalize_word(word)
        identity = self.identity_for(normalized)
        if normalized in self.revealed:
            raise ValueError(f"word is already revealed: {normalized}")
        self.revealed.add(normalized)
        return identity

    def remaining_for(self, team: Team) -> int:
        if not isinstance(team, Team):
            team = Team(team)
        target_identity = Identity.for_team(team)
        return sum(
            1
            for word, identity in self.words.items()
            if identity is target_identity and word not in self.revealed
        )

    def words_by_identity(self) -> dict[Identity, list[str]]:
        grouped = {identity: [] for identity in Identity}
        for word, identity in self.words.items():
            grouped[identity].append(word)
        for words in grouped.values():
            words.sort()
        return grouped

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        if include_hidden:
            identities = {word: identity.value for word, identity in sorted(self.words.items())}
            return {
                "words": identities,
                "identities": identities,
                "revealed": sorted(self.revealed),
            }
        return {
            "words": sorted(self.words),
            "revealed": sorted(self.revealed),
        }


@dataclass
class GameState:
    """Mutable deterministic state for one Codenames-style game."""

    board: Board
    current_team: Team = Team.RED
    phase: GamePhase = GamePhase.AWAITING_CLUE
    active_clue: Clue | None = None
    guesses_remaining: int = 0
    winner: Team | None = None
    terminal_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls, board: Board, *, starting_team: Team = Team.RED) -> "GameState":
        if not isinstance(starting_team, Team):
            starting_team = Team(starting_team)
        return cls(board=board, current_team=starting_team)

    def give_clue(self, clue: Clue) -> None:
        self._ensure_not_terminal()
        if self.phase is not GamePhase.AWAITING_CLUE:
            raise ValueError("cannot give a clue outside the clue phase")
        if not isinstance(clue, Clue):
            raise TypeError("clue must be a Clue instance")

        self.active_clue = clue
        self.guesses_remaining = self._guess_limit_for_clue(clue)
        self.phase = GamePhase.GUESSING
        self.history.append(
            {
                "event": "clue",
                "team": self.current_team.value,
                "clue": clue.to_dict(),
            }
        )

    @staticmethod
    def _guess_limit_for_clue(clue: Clue) -> int:
        # Baseline Codenames rule: clue count + one optional extra guess.
        # Count-zero clues are rare but legal; in the engine they allow one
        # discretionary guess and can be expanded later by a rule profile.
        return clue.count + 1

    def guess(self, word: str) -> GuessResult:
        self._ensure_not_terminal()
        if self.phase is not GamePhase.GUESSING:
            raise ValueError("cannot guess outside the guessing phase")
        if self.guesses_remaining <= 0:
            raise ValueError("no guesses remaining for the active clue")

        normalized = Board._normalize_word(word)
        acting_team = self.current_team
        identity = self.board.reveal(normalized)
        self.guesses_remaining -= 1

        result = self._resolve_guess(normalized, identity)
        self.history.append(
            {
                "event": "guess",
                "team": acting_team.value,
                "word": normalized,
                "revealed_identity": identity.value,
                "terminal": result.terminal,
                "winner": result.winner.value if result.winner else None,
                "reason": result.reason,
            }
        )
        return result

    def _resolve_guess(self, word: str, identity: Identity) -> GuessResult:
        if identity is Identity.ASSASSIN:
            return self._finish(word, identity, self.current_team.opponent, "assassin")

        identity_team = identity.as_team()
        if identity_team is self.current_team:
            if self.board.remaining_for(self.current_team) == 0:
                return self._finish(word, identity, self.current_team, "all_words_found")
            if self.guesses_remaining == 0:
                self._end_turn()
            return GuessResult(word=word, revealed_identity=identity)

        if identity_team is self.current_team.opponent:
            if self.board.remaining_for(self.current_team.opponent) == 0:
                return self._finish(word, identity, self.current_team.opponent, "opponent_completed")
            self._end_turn()
            return GuessResult(word=word, revealed_identity=identity)

        self._end_turn()
        return GuessResult(word=word, revealed_identity=identity)

    def stop_guessing(self) -> None:
        self._ensure_not_terminal()
        if self.phase is not GamePhase.GUESSING:
            raise ValueError("cannot stop outside the guessing phase")
        self.history.append({"event": "stop", "team": self.current_team.value})
        self._end_turn()

    def _finish(self, word: str, identity: Identity, winner: Team, reason: str) -> GuessResult:
        self.winner = winner
        self.terminal_reason = reason
        self.phase = GamePhase.TERMINAL
        self.guesses_remaining = 0
        return GuessResult(
            word=word,
            revealed_identity=identity,
            terminal=True,
            winner=winner,
            reason=reason,
        )

    def _end_turn(self) -> None:
        self.current_team = self.current_team.opponent
        self.phase = GamePhase.AWAITING_CLUE
        self.active_clue = None
        self.guesses_remaining = 0

    def _ensure_not_terminal(self) -> None:
        if self.phase is GamePhase.TERMINAL:
            raise ValueError("game is already terminal")

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, Any]:
        return {
            "board": self.board.to_dict(include_hidden=include_hidden),
            "current_team": self.current_team.value,
            "phase": self.phase.value,
            "active_clue": self.active_clue.to_dict() if self.active_clue else None,
            "guesses_remaining": self.guesses_remaining,
            "winner": self.winner.value if self.winner else None,
            "terminal_reason": self.terminal_reason,
            "history": list(self.history),
        }


__all__ = [
    "Board",
    "Clue",
    "GamePhase",
    "GameState",
    "GuessResult",
    "Identity",
    "Team",
]
