from __future__ import annotations

from typing import Literal, TypedDict


GlyphKind = Literal["tick", "post", "chain", "slip", "inc", "dec", "skip"]


class GlyphSpec(TypedDict, total=False):
    """
    Minimal glyph spec for stage-2 rendering.

    The renderer can interpret these fields however it wants; the point is to have
    a stable mapping from stitch symbol -> visual primitive.
    """

    kind: GlyphKind
    length: float  # relative length inside a ring band (0..1)
    bars: int  # crossbars for long stitches (dc=1, tr=2...)


class StitchMetric(TypedDict, total=False):
    """
    Relative metrics used to convert stitch types into "visual density".
    """

    height: float  # relative row height contribution
    width: float  # relative horizontal footprint


DEFAULT_GLYPHS: dict[str, GlyphSpec] = {
    # setup / misc
    # User mental model: sl st does not add visible height; it only joins/sets a landing target.
    "ch": {"kind": "chain", "length": 0.30},
    "sl st": {"kind": "slip", "length": 0.0},
    "sl": {"kind": "slip", "length": 0.0},
    "ss": {"kind": "slip", "length": 0.0},
    # stitches (two notations)
    # Make the visual length gap much larger: ch≈x < t < f < e(tr) < dtr.
    "x": {"kind": "tick", "length": 0.18},  # sc
    "sc": {"kind": "tick", "length": 0.18},
    "t": {"kind": "tick", "length": 0.52},  # hdc (common CN symbol)
    "hdc": {"kind": "tick", "length": 0.52},
    "f": {"kind": "post", "length": 0.86, "bars": 1},  # dc (common CN symbol)
    "dc": {"kind": "post", "length": 0.86, "bars": 1},
    "tr": {"kind": "post", "length": 0.96, "bars": 2},
    "e": {"kind": "post", "length": 0.96, "bars": 2},
    "dtr": {"kind": "post", "length": 0.995, "bars": 3},
    # shaping
    "v": {"kind": "inc", "length": 0.35},
    "a": {"kind": "dec", "length": 0.35},
    "skip": {"kind": "skip", "length": 0.0},
}


DEFAULT_METRICS: dict[str, StitchMetric] = {
    # Make x/sc shorter, and exaggerate tall stitches more aggressively.
    "x": {"height": 0.75, "width": 1.0},
    "sc": {"height": 0.75, "width": 1.0},
    # Goal: t >= 2x x; f/e/dtr clearly taller.
    "t": {"height": 2.0, "width": 1.0},
    "hdc": {"height": 2.0, "width": 1.0},
    "f": {"height": 3.2, "width": 1.0},
    "dc": {"height": 3.2, "width": 1.0},
    "tr": {"height": 4.6, "width": 1.0},
    "e": {"height": 4.6, "width": 1.0},
    "dtr": {"height": 6.0, "width": 1.0},
    # shaping ops behave like baseline stitches in layout; special glyphs show intent.
    "v": {"height": 0.75, "width": 1.0},
    "a": {"height": 0.75, "width": 1.0},
    # setup
    # ch ≈ x; sl st contributes no visible row height.
    "ch": {"height": 0.75, "width": 1.0},
    "sl st": {"height": 0.0, "width": 0.0},
    "sl": {"height": 0.0, "width": 0.0},
    "ss": {"height": 0.0, "width": 0.0},
}
