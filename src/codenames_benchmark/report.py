"""Markdown reporting."""
from __future__ import annotations

def render_markdown_report(ratings: dict[str, float], metrics: dict[str, object]) -> str:
    lines=["# Codenames Benchmark Report", "", "## Ratings"]
    for model, rating in sorted(ratings.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{model}: {round(rating)}")
    lines += ["", "## Diagnostics"]
    for key, value in sorted(metrics.items()): lines.append(f"{key}: {value}")
    return "\n".join(lines)+"\n"
