from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


MotifType = Literal["circle_coaster", "granny_square"]


@dataclass(frozen=True)
class RoundStitch:
    stitch: str
    repeat: Optional[int] = None
    notes: Optional[str] = None
    ops: Optional[list[dict[str, Any]]] = None
    total: Optional[int] = None
    consumed: Optional[int] = None


@dataclass(frozen=True)
class RoundSpec:
    """
    One "round" / "row" of a motif.

    Rendering in stage-1 uses only `color` and the order of rounds to build
    a preview. The stitch fields are carried through for future stages.
    """

    color: str
    stitch: RoundStitch


@dataclass(frozen=True)
class PatternSpec:
    motif_type: MotifType
    rounds: list[RoundSpec]
    title: Optional[str] = None
    background: str = "#ffffff"
    legend: Optional[dict[str, str]] = None
