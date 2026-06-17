"""Rating utilities."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return player A's expected Elo score against player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _score_pair(
    red_rating: float,
    blue_rating: float,
    winner_model: str | None,
    red_model: str,
    blue_model: str,
    *,
    k: float,
) -> tuple[float, float, float, float, float, float]:
    expected_red = expected_score(red_rating, blue_rating)
    expected_blue = 1.0 - expected_red
    if winner_model == red_model:
        score_red, score_blue = 1.0, 0.0
    elif winner_model == blue_model:
        score_red, score_blue = 0.0, 1.0
    else:
        score_red, score_blue = 0.5, 0.5
    red_delta = k * (score_red - expected_red)
    blue_delta = k * (score_blue - expected_blue)
    return expected_red, expected_blue, score_red, score_blue, red_delta, blue_delta


def elo_ratings(results: list[tuple[str, str, str | None]], *, initial: float = 1500.0, k: float = 32.0) -> dict[str, float]:
    """Compute final Elo ratings from ordered pairwise results.

    Each result is ``(model_a, model_b, winner)``. ``winner`` may be either model
    name or ``None`` for a draw/bounded non-terminal game.
    """
    system = EloRatingSystem(initial=initial, k=k)
    for game_number, (a, b, winner) in enumerate(results, 1):
        system.record_game(red_model=a, blue_model=b, winner_model=winner, game_number=game_number)
    return dict(system.ratings)


@dataclass
class EloRatingSystem:
    """Incremental Elo tracker for a tournament.

    The benchmark updates this after each completed game so standings can be
    monitored while a live OpenRouter tournament is still running.
    """

    models: list[str] | None = None
    initial: float = 1500.0
    k: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.models:
            for model in self.models:
                self.ratings.setdefault(model, float(self.initial))

    def rating_for(self, model: str) -> float:
        self.ratings.setdefault(model, float(self.initial))
        return self.ratings[model]

    def standings(self) -> list[dict[str, Any]]:
        ordered = sorted(self.ratings.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"rank": rank, "model": model, "rating": round(rating, 2)}
            for rank, (model, rating) in enumerate(ordered, 1)
        ]

    def record_game(
        self,
        *,
        red_model: str,
        blue_model: str,
        winner_model: str | None,
        game_number: int,
    ) -> dict[str, Any]:
        red_rating = self.rating_for(red_model)
        blue_rating = self.rating_for(blue_model)
        expected_red, expected_blue, score_red, score_blue, red_delta, blue_delta = _score_pair(
            red_rating,
            blue_rating,
            winner_model,
            red_model,
            blue_model,
            k=self.k,
        )
        new_red = red_rating + red_delta
        new_blue = blue_rating + blue_delta
        self.ratings[red_model] = new_red
        self.ratings[blue_model] = new_blue

        leader = self.standings()[0]["model"]
        entry = {
            "game_number": game_number,
            "red_model": red_model,
            "blue_model": blue_model,
            "winner_model": winner_model,
            "score_red": score_red,
            "score_blue": score_blue,
            "red_rating_before": round(red_rating, 2),
            "blue_rating_before": round(blue_rating, 2),
            "red_rating_after": round(new_red, 2),
            "blue_rating_after": round(new_blue, 2),
            "leader": leader,
            "standings": self.standings(),
        }
        self.history.append(entry)
        return entry

    def record_round(self, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Update ratings after a batch of concurrent games.

        All expected scores are computed from the ratings at the start of the
        round, and the cumulative deltas are applied only after the full batch
        has been scored. This preserves the semantics of a round-based live run
        where all games in the round are considered simultaneous.
        """
        if not games:
            return []

        starting_ratings = {model: self.rating_for(model) for game in games for model in (game["red_model"], game["blue_model"])}
        round_deltas: dict[str, float] = defaultdict(float)
        entries: list[dict[str, Any]] = []

        for game in games:
            red_model = game["red_model"]
            blue_model = game["blue_model"]
            winner_model = game.get("winner_model")
            game_number = game["game_number"]
            round_index = game.get("round_index")

            red_rating = starting_ratings[red_model]
            blue_rating = starting_ratings[blue_model]
            expected_red, expected_blue, score_red, score_blue, red_delta, blue_delta = _score_pair(
                red_rating,
                blue_rating,
                winner_model,
                red_model,
                blue_model,
                k=self.k,
            )
            round_deltas[red_model] += red_delta
            round_deltas[blue_model] += blue_delta

            entries.append(
                {
                    "game_number": game_number,
                    "round_index": round_index,
                    "red_model": red_model,
                    "blue_model": blue_model,
                    "winner_model": winner_model,
                    "score_red": score_red,
                    "score_blue": score_blue,
                    "expected_red": round(expected_red, 6),
                    "expected_blue": round(expected_blue, 6),
                    "red_rating_before": round(red_rating, 2),
                    "blue_rating_before": round(blue_rating, 2),
                    "red_delta": round(red_delta, 4),
                    "blue_delta": round(blue_delta, 4),
                }
            )

        for model, delta in round_deltas.items():
            self.ratings[model] = starting_ratings[model] + delta

        standings = self.standings()
        leader = standings[0]["model"]
        for entry in entries:
            entry["red_rating_after_round"] = round(self.ratings[entry["red_model"]], 2)
            entry["blue_rating_after_round"] = round(self.ratings[entry["blue_model"]], 2)
            entry["leader"] = leader
            entry["standings"] = standings
            self.history.append(entry)
        return entries


__all__ = ["EloRatingSystem", "elo_ratings", "expected_score"]
