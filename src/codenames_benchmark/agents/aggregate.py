"""Deterministic guess aggregation."""
from __future__ import annotations
from collections import defaultdict
from codenames_benchmark.protocol import AggregatorAction, GuesserAction

def aggregate_guesser_actions(
    actions: list[GuesserAction],
    *,
    guesses_remaining: int,
    unavailable_words: set[str] | None = None,
    stop_threshold: float = 0.5,
    min_consensus_votes: int = 2,
    confidence_threshold: float = 0.7,
) -> AggregatorAction:
    if guesses_remaining <= 0:
        return AggregatorAction([], stop_after=True)
    unavailable_words = {w.strip().lower() for w in (unavailable_words or set())}
    stop_after = bool(actions) and (sum(1 for a in actions if a.stop) / len(actions)) > stop_threshold
    votes = defaultdict(int); conf_sum = defaultdict(float); conf_count = defaultdict(int); best_rank = defaultdict(lambda: 10**6)
    for action in actions:
        for rank, word in enumerate(action.ranked_guesses):
            if word in unavailable_words: continue
            votes[word] += 1
            best_rank[word] = min(best_rank[word], rank)
            conf_sum[word] += action.confidences.get(word, 0.5)
            conf_count[word] += 1
    scored=[]
    for word, vote_count in votes.items():
        avg = conf_sum[word] / max(conf_count[word], 1)
        if vote_count < min_consensus_votes or avg < confidence_threshold:
            continue
        scored.append((-vote_count, -avg, best_rank[word], word))
    scored.sort()
    guesses = [word for *_rest, word in scored[:guesses_remaining]]
    return AggregatorAction(guesses, stop_after=stop_after if guesses else True)
