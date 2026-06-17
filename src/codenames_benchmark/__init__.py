"""Codenames LLM benchmark package."""

from codenames_benchmark.boards import (
    BOARD_SIZE,
    DEFAULT_WORDLIST_PATH,
    generate_board,
    load_word_list,
    mirror_board,
)
from codenames_benchmark.game import (
    Board,
    Clue,
    GamePhase,
    GameState,
    GuessResult,
    Identity,
    Team,
)
from codenames_benchmark.legality import (
    ClueLegalityConfig,
    ClueLegalityResult,
    check_clue,
    is_legal_clue,
)
from codenames_benchmark.protocol import (
    AggregatorAction,
    BaseObservation,
    GuesserAction,
    GuesserObservation,
    SpymasterAction,
    SpymasterObservation,
    build_guesser_observation,
    build_spymaster_observation,
)

__all__ = [
    "BOARD_SIZE",
    "DEFAULT_WORDLIST_PATH",
    "AggregatorAction",
    "BaseObservation",
    "Board",
    "Clue",
    "ClueLegalityConfig",
    "ClueLegalityResult",
    "GamePhase",
    "GameState",
    "GuesserAction",
    "GuesserObservation",
    "GuessResult",
    "Identity",
    "SpymasterAction",
    "SpymasterObservation",
    "Team",
    "build_guesser_observation",
    "build_spymaster_observation",
    "check_clue",
    "generate_board",
    "is_legal_clue",
    "load_word_list",
    "mirror_board",
]
