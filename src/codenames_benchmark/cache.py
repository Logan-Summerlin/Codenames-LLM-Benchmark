"""Simple JSON response cache."""
from __future__ import annotations
from pathlib import Path
import hashlib, json

class ResponseCache:
    def __init__(self, directory: Path): self.directory=Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
    def key_for(self, model: str, prompt: object, settings: object) -> str:
        text=json.dumps({"model":model,"prompt":prompt,"settings":settings}, sort_keys=True, separators=(",",":"))
        return hashlib.sha256(text.encode()).hexdigest()
    def _path(self, key: str) -> Path: return self.directory / f"{key}.json"
    def get(self, key: str):
        p=self._path(key)
        return json.loads(p.read_text()) if p.exists() else None
    def set(self, key: str, value: object):
        self._path(key).write_text(json.dumps(value, sort_keys=True, indent=2))
