"""Automated clue-legality checks for strict Codenames benchmark rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from codenames_benchmark.game import Board


@dataclass(frozen=True)
class ClueLegalityConfig:
    """Rule switches for automated clue validation."""

    allow_multiword: bool = False
    reject_substrings: bool = True
    reject_morphological_variants: bool = True
    allowed_words: frozenset[str] | None = None

    def __init__(
        self,
        *,
        allow_multiword: bool = False,
        reject_substrings: bool = True,
        reject_morphological_variants: bool = True,
        allowed_words: Iterable[str] | None = None,
    ) -> None:
        object.__setattr__(self, "allow_multiword", allow_multiword)
        object.__setattr__(self, "reject_substrings", reject_substrings)
        object.__setattr__(self, "reject_morphological_variants", reject_morphological_variants)
        normalized_allowed = None
        if allowed_words is not None:
            normalized_allowed = frozenset(_normalize_clue(word) for word in allowed_words)
        object.__setattr__(self, "allowed_words", normalized_allowed)


@dataclass(frozen=True)
class ClueLegalityResult:
    """Outcome of checking a single clue."""

    legal: bool
    normalized_clue: str
    reason: str | None = None
    matched_word: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "legal": self.legal,
            "normalized_clue": self.normalized_clue,
            "reason": self.reason,
            "matched_word": self.matched_word,
        }


DEFAULT_CONFIG = ClueLegalityConfig()


def check_clue(
    clue: str,
    board: Board,
    *,
    config: ClueLegalityConfig | None = None,
) -> ClueLegalityResult:
    """Return a deterministic legality decision for a proposed clue."""

    config = config or DEFAULT_CONFIG
    normalized = _normalize_clue(clue)

    if not config.allow_multiword and _is_multiword(normalized):
        return ClueLegalityResult(False, normalized, "multiword")

    if config.allowed_words is not None and normalized not in config.allowed_words:
        return ClueLegalityResult(False, normalized, "not_in_dictionary")

    board_words = set(board.words)
    if normalized in board_words:
        return ClueLegalityResult(False, normalized, "board_word", normalized)

    if config.reject_morphological_variants:
        clue_stems = _candidate_stems(normalized)
        for word in board_words:
            if word in clue_stems or normalized in _candidate_stems(word):
                return ClueLegalityResult(False, normalized, "morphological_variant", word)

    if config.reject_substrings:
        for word in board_words:
            if len(normalized) >= 3 and len(word) >= 3 and (normalized in word or word in normalized):
                return ClueLegalityResult(False, normalized, "substring", word)

    return ClueLegalityResult(True, normalized)


def is_legal_clue(
    clue: str,
    board: Board,
    *,
    config: ClueLegalityConfig | None = None,
) -> bool:
    """Convenience boolean wrapper around :func:`check_clue`."""

    return check_clue(clue, board, config=config).legal


def _normalize_clue(clue: str) -> str:
    normalized = str(clue).strip().lower()
    if not normalized:
        raise ValueError("clue must be non-empty")
    return " ".join(normalized.split())


def _is_multiword(clue: str) -> bool:
    return any(separator in clue for separator in (" ", "-", "_"))


def _candidate_stems(word: str) -> set[str]:
    stems = {word}
    suffix_rules = (
        ("ies", "y"),
        ("es", ""),
        ("s", ""),
        ("ing", ""),
        ("ing", "e"),
        ("ed", ""),
        ("ed", "e"),
        ("er", ""),
        ("er", "e"),
        ("ers", ""),
        ("ers", "e"),
    )
    for suffix, replacement in suffix_rules:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stems.add(word[: -len(suffix)] + replacement)
    return stems


__all__ = [
    "ClueLegalityConfig",
    "ClueLegalityResult",
    "check_clue",
    "is_legal_clue",
]
