"""Run manifest persistence."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class RunManifest:
    models: list[str]
    seeds: list[int | str]
    rule_profile: str
    word_list_hash: str
    sampling: dict
    code_version: str = "unversioned"
    def to_dict(self): return self.__dict__.copy()
    def write(self, directory: Path) -> Path:
        directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
        path=directory/"manifest.json"
        path.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2))
        return path
