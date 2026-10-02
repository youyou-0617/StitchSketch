from __future__ import annotations

import json
from typing import Any

from .schema import PatternSpec


def pattern_to_dict(pattern: PatternSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": pattern.title,
        "motif_type": pattern.motif_type,
        "background": pattern.background,
        "rounds": [
            {
                "color": r.color,
                "stitch": {
                    "name": r.stitch.stitch,
                    "repeat": r.stitch.repeat,
                    "notes": r.stitch.notes,
                    "ops": r.stitch.ops,
                    "total": r.stitch.total,
                    "consumed": r.stitch.consumed,
                },
            }
            for r in pattern.rounds
        ],
    }
    if pattern.legend is not None:
        out["legend"] = pattern.legend
    return out


def pattern_to_json(pattern: PatternSpec, *, indent: int = 2) -> str:
    return json.dumps(pattern_to_dict(pattern), ensure_ascii=False, indent=indent)
