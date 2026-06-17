"""Deterministic board generation for the Codenames benchmark."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from codenames_benchmark.game import Board, Identity, Team


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORDLIST_PATH = PROJECT_ROOT / "data" / "wordlists" / "base_words.txt"
BOARD_SIZE = 25


def _normalize_word(word: str) -> str:
    normalized = str(word).strip().lower()
    if not normalized:
        raise ValueError("word list entries must be non-empty")
    return normalized


def load_word_list(path: str | Path = DEFAULT_WORDLIST_PATH) -> list[str]:
    """Load a newline-delimited word list with stable normalization."""

    loaded: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            word = _normalize_word(line)
            if word in seen:
                continue
            seen.add(word)
            loaded.append(word)
    return loaded


def generate_board(
    *,
    seed: int | str,
    words: Iterable[str] | None = None,
    starting_team: Team = Team.RED,
) -> Board:
    """Generate a reproducible 25-word board and hidden assignment from a seed."""

    if not isinstance(starting_team, Team):
        starting_team = Team(starting_team)

    word_pool = list(words) if words is not None else load_word_list(DEFAULT_WORDLIST_PATH)
    normalized_pool = _unique_normalized_words(word_pool)
    if len(normalized_pool) < BOARD_SIZE:
        raise ValueError("word list must contain at least 25 unique words")

    rng = random.Random(seed)
    selected_words = rng.sample(normalized_pool, BOARD_SIZE)
    assignment_order = selected_words[:]
    rng.shuffle(assignment_order)

    first_identity = Identity.for_team(starting_team)
    second_identity = Identity.for_team(starting_team.opponent)
    identities: dict[str, Identity] = {}

    for word in assignment_order[:9]:
        identities[word] = first_identity
    for word in assignment_order[9:17]:
        identities[word] = second_identity
    for word in assignment_order[17:24]:
        identities[word] = Identity.NEUTRAL
    identities[assignment_order[24]] = Identity.ASSASSIN

    return Board({word: identities[word] for word in selected_words})


def _unique_normalized_words(words: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_word in words:
        word = _normalize_word(raw_word)
        if word in seen:
            continue
        seen.add(word)
        normalized.append(word)
    return normalized


def mirror_board(board: Board) -> Board:
    """Return a board with identical words but red and blue identities swapped."""

    mirrored: dict[str, Identity] = {}
    for word, identity in board.words.items():
        if identity is Identity.RED:
            mirrored[word] = Identity.BLUE
        elif identity is Identity.BLUE:
            mirrored[word] = Identity.RED
        else:
            mirrored[word] = identity
    return Board(mirrored, revealed=set(board.revealed))


__all__ = [
    "BOARD_SIZE",
    "DEFAULT_WORDLIST_PATH",
    "generate_board",
    "load_word_list",
    "mirror_board",
]
