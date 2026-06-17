"""Board-suite loaders."""
from __future__ import annotations
from pathlib import Path
import json
from codenames_benchmark.game import Board, Identity

def load_board_suite(path: str | Path) -> list[Board]:
    boards=[]
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            identities=row.get("identities") or row.get("words")
            boards.append(Board({w: Identity(v) for w,v in identities.items()}))
    return boards
