"""Round-robin tournament scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class ModelSpec:
    """OpenRouter model entry for the Codenames tournament."""

    label: str
    slug: str
    provider: str
    provider_tag: str | None = None
    reasoning_effort: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "label": self.label,
            "slug": self.slug,
            "provider": self.provider,
            "provider_tag": self.provider_tag,
            "reasoning_effort": self.reasoning_effort,
        }


OPENROUTER_CODENAMES_MODELS: list[ModelSpec] = [
    ModelSpec("Llama 3.3 70B Instruct", "meta-llama/llama-3.3-70b-instruct", "Novita", "novita/bf16", "low"),
    ModelSpec("GPT-4o mini", "openai/gpt-4o-mini", "Azure", "azure/swedencentral", "low"),
    ModelSpec("Qwen2.5 72B Instruct", "qwen/qwen-2.5-72b-instruct", "DeepInfra", "deepinfra/fp8", "low"),
    ModelSpec("DeepSeek V3 0324", "deepseek/deepseek-chat-v3-0324", "ModelRun", "modelrun/fp4", "low"),
    ModelSpec("GPT-OSS 120B", "openai/gpt-oss-120b", "Google", "google-vertex/global", "low"),
    ModelSpec("Mistral Small 4", "mistralai/mistral-small-2603", "Mistral", "mistral", "low"),
    ModelSpec("Mistral Medium 3.1", "mistralai/mistral-medium-3.1", "Mistral", "mistral", "low"),
    ModelSpec("Phi-4", "microsoft/phi-4", "Microsoft", None, "low"),
    ModelSpec("Gemini 3.1 Flash Lite", "google/gemini-3.1-flash-lite", "Google", "google-vertex/global", "low"),
    ModelSpec("Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", "Google", "google-vertex/global", "low"),
    ModelSpec("Gemma 4 31B", "google/gemma-4-31b-it", "Parasail", "parasail/fp8", "low"),
    ModelSpec("GPT-5.4-nano", "openai/gpt-5.4-nano", "Azure", "azure", "low"),
    ModelSpec("Llama 4 Scout", "meta-llama/llama-4-scout", "Meta", None, "low"),
    ModelSpec("Gemma 3 27B", "google/gemma-3-27b-it", "Google", None, "low"),
    ModelSpec("Nova Lite 1.0", "amazon/nova-lite-v1", "Amazon", None, "low"),
    ModelSpec("Claude 3 Haiku", "anthropic/claude-3-haiku", "Anthropic", None, "low"),
    ModelSpec("GPT-OSS 20B", "openai/gpt-oss-20b", "Groq", "groq", "low"),
    ModelSpec("Qwen 3 32B", "qwen/qwen3-32b", "Qwen", None, "low"),
]


@dataclass(frozen=True)
class ScheduledGame:
    model_a: str
    model_b: str
    seed: str
    mirror_index: int

    def to_dict(self) -> dict[str, int | str]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TournamentGame:
    """One color-assigned game in a double round robin."""

    game_number: int
    round_index: int
    red_model: str
    blue_model: str
    seed: str

    def to_dict(self) -> dict[str, int | str]:
        return self.__dict__.copy()


def _safe_seed_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def schedule_round_robin(models: list[str], *, mirrored_seeds: int, seed_prefix: str = "seed") -> list[ScheduledGame]:
    """Backward-compatible mirrored round-robin scheduler."""
    games = []
    for a, b in combinations(models, 2):
        for r in range(mirrored_seeds):
            for m in (0, 1):
                games.append(ScheduledGame(a, b, f"{seed_prefix}-{a}-{b}-{r}", m))
    return games


def schedule_double_round_robin(models: list[str], *, seed_prefix: str = "rr") -> list[TournamentGame]:
    """Schedule every unordered model pair twice with swapped colors.

    For ``n`` models this produces ``n * (n - 1)`` games: each model plays one
    red game and one blue game against every other model.
    """
    games: list[TournamentGame] = []
    game_number = 1
    for a, b in combinations(models, 2):
        for round_index, (red, blue) in enumerate(((a, b), (b, a)), 1):
            seed = f"{seed_prefix}-{game_number:03d}-{_safe_seed_name(red)}-vs-{_safe_seed_name(blue)}"
            games.append(
                TournamentGame(
                    game_number=game_number,
                    round_index=round_index,
                    red_model=red,
                    blue_model=blue,
                    seed=seed,
                )
            )
            game_number += 1
    return games


def schedule_single_round_robin(models: list[str], *, seed_prefix: str = "rr") -> list[TournamentGame]:
    """Schedule one color-assigned game for every unordered model pair.

    Colors alternate across the pair list so the field does not get an obvious
    red/blue clustering bias.
    """
    games: list[TournamentGame] = []
    game_number = 1
    for pair_index, (a, b) in enumerate(combinations(models, 2)):
        red, blue = (a, b) if pair_index % 2 == 0 else (b, a)
        seed = f"{seed_prefix}-{game_number:03d}-{_safe_seed_name(red)}-vs-{_safe_seed_name(blue)}"
        games.append(
            TournamentGame(
                game_number=game_number,
                round_index=pair_index + 1,
                red_model=red,
                blue_model=blue,
                seed=seed,
            )
        )
        game_number += 1
    return games


def schedule_limited_coverage(models: list[str], *, seed_prefix: str = "coverage") -> list[TournamentGame]:
    """Schedule a compact coverage tournament where every model appears.

    Adjacent models are paired once. If the field has an odd number of models,
    the final model plays the first model, so every model appears at least once.
    """
    games: list[TournamentGame] = []
    game_number = 1
    for index in range(0, len(models) - 1, 2):
        red = models[index]
        blue = models[index + 1]
        seed = f"{seed_prefix}-{game_number:03d}-{_safe_seed_name(red)}-vs-{_safe_seed_name(blue)}"
        games.append(TournamentGame(game_number, 1, red, blue, seed))
        game_number += 1
    if len(models) % 2 == 1 and len(models) > 1:
        red = models[-1]
        blue = models[0]
        seed = f"{seed_prefix}-{game_number:03d}-{_safe_seed_name(red)}-vs-{_safe_seed_name(blue)}"
        games.append(TournamentGame(game_number, 1, red, blue, seed))
    return games


def tournament_pairings(*, seed_prefix: str = "openrouter-codenames") -> list[TournamentGame]:
    """Return the default OpenRouter Codenames double round-robin schedule."""
    return schedule_double_round_robin([model.slug for model in OPENROUTER_CODENAMES_MODELS], seed_prefix=seed_prefix)


def tournament_single_round_robin_pairings(*, seed_prefix: str = "openrouter-codenames-single") -> list[TournamentGame]:
    """Return the default OpenRouter Codenames single round-robin schedule."""
    return schedule_single_round_robin([model.slug for model in OPENROUTER_CODENAMES_MODELS], seed_prefix=seed_prefix)


def tournament_limited_coverage_pairings(*, seed_prefix: str = "openrouter-codenames-coverage") -> list[TournamentGame]:
    """Return a compact OpenRouter Codenames schedule covering every model."""
    return schedule_limited_coverage([model.slug for model in OPENROUTER_CODENAMES_MODELS], seed_prefix=seed_prefix)


__all__ = [
    "ModelSpec",
    "OPENROUTER_CODENAMES_MODELS",
    "ScheduledGame",
    "TournamentGame",
    "schedule_round_robin",
    "schedule_double_round_robin",
    "schedule_single_round_robin",
    "schedule_limited_coverage",
    "tournament_pairings",
    "tournament_single_round_robin_pairings",
    "tournament_limited_coverage_pairings",
]
