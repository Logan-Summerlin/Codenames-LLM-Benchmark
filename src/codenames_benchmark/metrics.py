"""Diagnostic metrics over event logs."""
from __future__ import annotations

def compute_diagnostics(events: list[dict]) -> dict[str, int | float]:
    guesses=[e for e in events if e.get("event")=="guess"]
    opponent=0; neutral=0; assassin=0
    for e in guesses:
        team=e.get("team"); ident=e.get("revealed_identity")
        if ident == "assassin": assassin += 1
        elif ident == "neutral": neutral += 1
        elif ident in ("red","blue") and ident != team: opponent += 1
    clues=[e for e in events if e.get("event")=="clue"]
    return {"total_guesses":len(guesses),"clues":len(clues),"opponent_hits":opponent,"neutral_hits":neutral,"assassin_hits":assassin,"illegal_clues":sum(1 for e in events if e.get("event")=="illegal_clue"),"clue_efficiency": (len(guesses)/len(clues) if clues else 0.0)}
