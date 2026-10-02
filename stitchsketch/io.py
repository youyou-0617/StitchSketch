from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import PatternSpec, RoundSpec, RoundStitch


def _require(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise ValueError(f"Missing required key: {key}")
    return obj[key]


def load_pattern(path: str | Path) -> PatternSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pattern JSON must be an object at top-level.")

    motif_type = _require(data, "motif_type")
    rounds_data = _require(data, "rounds")
    if not isinstance(rounds_data, list) or not rounds_data:
        raise ValueError("`rounds` must be a non-empty list.")

    rounds: list[RoundSpec] = []
    for idx, r in enumerate(rounds_data):
        if not isinstance(r, dict):
            raise ValueError(f"rounds[{idx}] must be an object.")
        color = _require(r, "color")
        stitch_data = _require(r, "stitch")
        if not isinstance(stitch_data, dict):
            raise ValueError(f"rounds[{idx}].stitch must be an object.")
        stitch = _require(stitch_data, "name")
        repeat = stitch_data.get("repeat")
        notes = stitch_data.get("notes")
        ops = stitch_data.get("ops")
        total = stitch_data.get("total")
        consumed = stitch_data.get("consumed")
        rounds.append(
            RoundSpec(
                color=str(color),
                stitch=RoundStitch(str(stitch), repeat=repeat, notes=notes, ops=ops, total=total, consumed=consumed),
            )
        )

    title = data.get("title")
    background = str(data.get("background", "#ffffff"))
    legend = data.get("legend")
    return PatternSpec(motif_type=motif_type, rounds=rounds, title=title, background=background, legend=legend)
