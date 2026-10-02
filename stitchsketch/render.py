from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from .io import load_pattern
from .schema import PatternSpec
from .glyphs import DEFAULT_GLYPHS, DEFAULT_METRICS

_RING_HEIGHT_BLEND = 0.35
_BUMP_RAW_MIX = 0.20


@dataclass(frozen=True)
class RenderOptions:
    size: int = 512
    padding: int = 24
    rotation_deg: float = 0.0
    stroke: str = "#111111"
    stroke_width: int = 2
    texture: bool = True
    glyph_stroke: str = "#111111"
    glyph_opacity: float = 0.55
    glyph_width: int = 1
    max_glyphs_per_round: int = 160
    # If true, ignore per-stitch `DEFAULT_GLYPHS[*].length` and draw glyph strokes
    # to the top of the ring band (i.e., fill the band radially).
    glyph_full_height: bool = False
    # How to distribute glyph angles around a ring.
    # - by_structure: use the stitch "pos" derived from consumed base-stitch units
    # - uniform: spread glyphs evenly (good for quick previews)
    glyph_angle_mode: Literal["by_structure", "uniform", "by_produced"] = "by_structure"
    glyph_style: Literal["strokes", "metric_strokes"] = "strokes"
    start_mode: Literal["auto", "mr", "ch_join", "foundation_chain"] = "auto"
    start_n: int | None = None
    center_hole_px: float | None = None
    first_prev_total_override: int | None = None
    # Optional per-round outer-boundary modulation for circle motifs.
    # Each entry is for a round_index (0-based): start_offset in [0, prev_total),
    # and segments whose lengths sum to prev_total, each with bump in [0..1].
    circle_round_mod: Optional[dict] = None
    # Optional auto modulation based on local growth (inc/dec/cluster) density.
    circle_auto_bump: bool = False
    # Scope of auto-bump application.
    # - all: apply to every round
    # - outermost: apply only to the outermost round
    circle_auto_bump_scope: Literal["all", "outermost"] = "outermost"
    circle_auto_bump_round: int | None = None  # kept for backward-compat; ignored when scope != "outermost"
    circle_auto_bump_window: int = 3  # odd, in base-stitch units
    circle_auto_bump_smoothing: Literal["weighted", "mean"] = "weighted"
    circle_auto_bump_strength: float = 0.9  # 0..1
    # Outline sampling strategy for modulated circular rings.
    # - uniform: method 1, sample uniformly by prev_total
    # - produced_subdivide: method 2, subdivide each base stitch by produced count
    circle_outline_sampling: Literal["uniform", "produced_subdivide"] = "uniform"
    show_seam_marker: bool = True
    # Visual treatment for ch chains.
    # - stroke: continuous yarn strip for the interactive structure preview
    # - beads: split the same path into rounded chain particles for realization references
    chain_render_style: Literal["stroke", "beads"] = "stroke"
    chain_width_scale: float = 1.0


def _svg_header(size: int, height: int | None = None) -> str:
    height = int(height if height is not None else size)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{height}" viewBox="0 0 {size} {height}">\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"

def _ops_counts(ops: object) -> dict[str, int]:
    """
    Count stitch-like symbols from ops.

    - stitch: counts by `n`
    - inc/dec: count produced base stitches
    - cluster: ignored for ring-band thickness; it affects outline/texture instead
    - repeat: multiplies inner counts
    """
    if not isinstance(ops, list):
        return {}
    counts: dict[str, int] = {}
    for op in ops:
        if not isinstance(op, dict):
            continue
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            n = int(op.get("n", 1))
            if st in {"skip", "-", "ch", "sl st"}:
                continue
            counts[st] = counts.get(st, 0) + n
        elif t == "inc":
            n = int(op.get("n", 1))
            base = str(op.get("base") or "x")
            # inc yields 2 stitches of the base type
            counts[base] = counts.get(base, 0) + (2 * n)
        elif t == "dec":
            n = int(op.get("n", 1))
            base = str(op.get("base") or "x")
            counts[base] = counts.get(base, 0) + n
        elif t == "skip":
            continue
        elif t == "cluster":
            continue
        elif t == "ch_space":
            inner = _ops_counts(op.get("ops"))
            for k, v in inner.items():
                counts[k] = counts.get(k, 0) + v
        elif t == "repeat":
            times = int(op.get("times", 1))
            inner = _ops_counts(op.get("ops"))
            for k, v in inner.items():
                counts[k] = counts.get(k, 0) + (times * v)
    return counts


def _min_height_from_counts(counts: dict[str, int]) -> float:
    if not counts:
        return 1.0
    heights: list[float] = []
    for sym, n in counts.items():
        if n <= 0:
            continue
        metric = DEFAULT_METRICS.get(sym) or DEFAULT_METRICS.get({"sc": "x", "hdc": "t", "dc": "f"}.get(sym, sym)) or {}
        heights.append(float(metric.get("height", 1.0)))
    return min(heights) if heights else 1.0


def _round_weight(pattern: PatternSpec, round_index: int) -> float:
    r = pattern.rounds[round_index]
    if _is_chain_only_ops(r.stitch.ops):
        return 0.0
    raw_height = _min_height_from_counts(_ops_counts(r.stitch.ops))
    return 1.0 + ((raw_height - 1.0) * _RING_HEIGHT_BLEND)


def _is_chain_only_ops(ops: object) -> bool:
    if not isinstance(ops, list) or not ops:
        return False
    saw_chain = False
    for op in ops:
        if not isinstance(op, dict):
            return False
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            if st in {"sl", "sl st", "ss", "skip", "-"}:
                continue
            if st != "ch":
                return False
            if int(op.get("n", 1)) > 0:
                saw_chain = True
        elif t == "skip":
            continue
        elif t == "repeat":
            inner = op.get("ops")
            if not _is_chain_only_ops(inner):
                return False
            saw_chain = True
        else:
            return False
    return saw_chain


def _band_radii(max_radius: float, weights: list[float], *, inner_offset: float = 0.0) -> list[float]:
    total = sum(weights) or 1.0
    inner_offset = max(0.0, min(max_radius, inner_offset))
    radii = [inner_offset]
    acc = 0.0
    for w in weights:
        acc += w
        radii.append(inner_offset + (max_radius - inner_offset) * (acc / total))
    return radii


def _visible_round_band(radii: list[float], round_index: int, *, fallback: float) -> float:
    """Return the stitch-height scale for this round, falling back past zero-width chain rounds."""
    if len(radii) >= 2:
        i = max(0, min(int(round_index), len(radii) - 2))
        band = float(radii[i + 1]) - float(radii[i])
        if band > 1.0:
            return band
        for j in range(i - 1, -1, -1):
            prev_band = float(radii[j + 1]) - float(radii[j])
            if prev_band > 1.0:
                return prev_band
        for j in range(i + 1, len(radii) - 1):
            next_band = float(radii[j + 1]) - float(radii[j])
            if next_band > 1.0:
                return next_band
    return max(12.0, float(fallback))


def _chain_unit_from_band(band: float) -> float:
    """Length of one ch in px. ch uses the same visual height scale as x/sc."""
    ch_h = float((DEFAULT_METRICS.get("ch") or {}).get("height", 1.0))
    x_h = float((DEFAULT_METRICS.get("x") or {}).get("height", ch_h or 1.0))
    ratio = ch_h / x_h if x_h > 0 else 1.0
    return max(12.0, float(band) * ratio)


def _glyph_count(total_stitches: int | None, max_glyphs: int) -> int:
    if total_stitches is None or total_stitches <= 0:
        return 24
    if total_stitches <= max_glyphs:
        return total_stitches
    # Subsample to keep SVG light; still correlates with density.
    return max_glyphs


def _is_skip_op_for_consumption(op: object) -> bool:
    if not isinstance(op, dict):
        return False
    if op.get("type") == "skip":
        return True
    return op.get("type") == "stitch" and str(op.get("st", "")) in {"skip", "-"}


def _skip_op_n_for_consumption(op: object) -> int:
    if not isinstance(op, dict):
        return 0
    return max(0, int(op.get("n", 1)))


def _chain_occupancy_after(ops: list[object], index: int) -> tuple[int, int]:
    skipped = 0
    j = index + 1
    while j < len(ops) and _is_skip_op_for_consumption(ops[j]):
        skipped += _skip_op_n_for_consumption(ops[j])
        j += 1
    lands_with_sl = (
        j < len(ops)
        and isinstance(ops[j], dict)
        and ops[j].get("type") == "stitch"
        and str(ops[j].get("st", "")) in {"sl", "sl st", "ss"}
        and not isinstance(ops[j].get("target"), dict)
    )
    return max(1, skipped + (1 if lands_with_sl else 0)), j


def _previous_op_is_chain(ops: list[object], index: int) -> bool:
    j = index - 1
    while j >= 0 and _is_skip_op_for_consumption(ops[j]):
        j -= 1
    if j < 0:
        return False
    prev = ops[j]
    return isinstance(prev, dict) and prev.get("type") == "stitch" and str(prev.get("st", "")) == "ch"


def _estimate_consumed(ops: object) -> int:
    if not isinstance(ops, list):
        return 0
    total = 0
    i = 0
    while i < len(ops):
        op = ops[i]
        if not isinstance(op, dict):
            i += 1
            continue
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            n = int(op.get("n", 1))
            if st in {"skip", "-"}:
                total += n
            elif st == "sl st":
                if not isinstance(op.get("target"), dict):
                    total += 0 if _previous_op_is_chain(ops, i) else max(0, n)
            elif st == "ch":
                occupancy, next_i = _chain_occupancy_after(ops, i)
                total += occupancy
                i = next_i
                continue
            else:
                total += n
        elif t == "inc":
            total += int(op.get("n", 1))
        elif t == "dec":
            total += 2 * int(op.get("n", 1))
        elif t == "skip":
            total += int(op.get("n", 1))
        elif t == "cluster":
            inner = op.get("ops", [])
            if isinstance(inner, list):
                _ = _estimate_consumed(inner)
            total += 1
        elif t == "ch_space":
            inner = op.get("ops", [])
            if isinstance(inner, list):
                _ = _estimate_consumed(inner)
            total += 1
        elif t == "repeat":
            times = int(op.get("times", 1))
            total += times * _estimate_consumed(op.get("ops"))
        i += 1
    return total


def _has_skip(ops: object) -> bool:
    if not isinstance(ops, list):
        return False
    for op in ops:
        if not isinstance(op, dict):
            continue
        t = op.get("type")
        if t == "skip":
            return True
        if t == "stitch" and str(op.get("st", "")) in {"skip", "-"}:
            return True
        if t == "repeat" and _has_skip(op.get("ops")):
            return True
    return False


def _prev_total(pattern: PatternSpec, round_index: int, consumed_fallback: int) -> int:
    if round_index <= 0:
        return consumed_fallback or 1
    prev = pattern.rounds[round_index - 1].stitch.total
    if isinstance(prev, int) and prev > 0:
        return prev
    return consumed_fallback or 1


def _prev_total_with_override(pattern: PatternSpec, round_index: int, consumed_fallback: int, override: int | None) -> int:
    if round_index == 0 and isinstance(override, int) and override > 0:
        return override
    if round_index > 0:
        prev_text = pattern.rounds[round_index - 1].stitch.stitch.lower()
        if ("ch" in prev_text) and ("sl st" in prev_text) and ("join" in prev_text):
            return 1
    return _prev_total(pattern, round_index, consumed_fallback)


def _infer_ch_n(text: str) -> int | None:
    # Match "ch 4", "ch4"
    m = re.search(r"\bch\s*(\d+)\b", text.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _effective_start(pattern: PatternSpec, opts: RenderOptions) -> tuple[str, int | None]:
    """
    Decide start mode + n.

    - If start_mode is auto: infer from round-1 text (kept as first round by design).
    - If start_mode is explicit but start_n is None: still try to infer n from round-1.
    """
    round0 = pattern.rounds[0].stitch.stitch.lower() if pattern.rounds else ""
    has_join = "sl st" in round0 and "join" in round0
    has_mr = "mr" in round0 or "magic ring" in round0
    n = opts.start_n
    inferred_n = _infer_ch_n(round0)

    if opts.start_mode == "auto":
        if has_mr:
            return "mr", None
        if has_join and inferred_n is not None:
            return "ch_join", inferred_n
        if round0.strip().startswith("ch") and inferred_n is not None:
            return "foundation_chain", inferred_n
        return "auto", None

    # explicit mode
    if n is None:
        n = inferred_n
    return opts.start_mode, n


def _center_hole_px(center_hole_px: float | None, mode: str, n: int | None, max_radius: float) -> float:
    if isinstance(center_hole_px, (int, float)) and center_hole_px >= 0:
        return float(min(max_radius * 0.6, center_hole_px))
    n = n or 0
    if mode == "mr":
        return 0.0
    if mode == "ch_join":
        # Heuristic: small hole that grows with chain count.
        return float(min(max_radius * 0.25, max(4.0, n * 2.0)))
    return 0.0


def _sample_keep_markers(items: list[tuple[float, str]], limit: int) -> list[tuple[float, str]]:
    if len(items) <= limit:
        return items
    markers = [it for it in items if it[1].startswith(("inc:", "dec:"))]
    others = [it for it in items if not it[1].startswith(("inc:", "dec:"))]
    if len(markers) >= limit:
        # keep marker order, subsample if needed
        step = len(markers) / limit
        return [markers[int(i * step)] for i in range(limit)]
    remaining = limit - len(markers)
    step = len(others) / max(remaining, 1)
    sampled_others = [others[int(i * step)] for i in range(remaining)]
    # merge while preserving order by position
    merged = markers + sampled_others
    merged.sort(key=lambda x: x[0])
    return merged


def _mod_from_opts(opts: RenderOptions, round_index: int, prev_total: int) -> tuple[int, list[tuple[int, float]]] | None:
    mod = opts.circle_round_mod
    if not isinstance(mod, dict):
        return None
    if int(mod.get("round_index", -1)) != int(round_index):
        return None
    start_offset = int(mod.get("start_offset", 0) or 0) % max(prev_total, 1)
    segs = mod.get("segments")
    if not isinstance(segs, list) or not segs:
        return None
    out: list[tuple[int, float]] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        length = int(s.get("length", 0) or 0)
        bump = float(s.get("bump", 0.0) or 0.0)
        if length <= 0:
            continue
        out.append((length, max(0.0, min(1.0, bump))))
    if not out:
        return None
    total_len = sum(l for l, _ in out)
    if total_len != prev_total:
        # Adjust last segment to fit (best-effort).
        delta = prev_total - total_len
        l_last, b_last = out[-1]
        out[-1] = (max(1, l_last + delta), b_last)
    return start_offset, out


def _bump_at_pos(pos: float, prev_total: int, start_offset: int, segments: list[tuple[int, float]]) -> float:
    if prev_total <= 0:
        return 0.0
    # shift so user-defined start_offset is at angle start
    p = (pos + start_offset) % prev_total
    acc = 0
    for length, bump in segments:
        if p < acc + length:
            return bump
        acc += length
    return segments[-1][1] if segments else 0.0


def _metric_height(sym: str) -> float:
    normalized = {"sc": "x", "hdc": "t", "dc": "f", "e": "tr"}.get(sym, sym)
    metric = DEFAULT_METRICS.get(normalized) or {}
    return float(metric.get("height", 1.0))


def _estimate_total_ops(ops: object) -> int:
    if not isinstance(ops, list):
        return 0
    total = 0
    for op in ops:
        if not isinstance(op, dict):
            continue
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            n = int(op.get("n", 1))
            if st in {"skip", "-", "sl st"}:
                continue
            total += n
        elif t == "inc":
            total += 2 * int(op.get("n", 1))
        elif t == "dec":
            total += int(op.get("n", 1))
        elif t == "skip":
            total += 0
        elif t == "cluster":
            total += _estimate_total_ops(op.get("ops"))
        elif t == "ch_space":
            total += _estimate_total_ops(op.get("ops"))
        elif t == "repeat":
            times = int(op.get("times", 1))
            total += times * _estimate_total_ops(op.get("ops"))
    return total


def _regular_flat_unit_info(ops: object) -> tuple[int, bool] | None:
    if not isinstance(ops, list):
        return None
    heights: list[float] = []
    has_inc = False
    consumed = 0

    for op in ops:
        if not isinstance(op, dict):
            return None
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            if st in {"skip", "-", "ch", "sl", "sl st", "ss"}:
                return None
            n = int(op.get("n", 1))
            if n <= 0:
                return None
            heights.extend([_metric_height(st)] * n)
            consumed += n
        elif t == "inc":
            n = int(op.get("n", 1))
            if n <= 0:
                return None
            base = str(op.get("base") or "x")
            heights.extend([_metric_height(base)] * n)
            consumed += n
            has_inc = True
        else:
            return None

    if not heights or consumed <= 0:
        return None
    h0 = heights[0]
    if not all(abs(h - h0) < 1e-9 for h in heights):
        return None
    return consumed, has_inc


def _is_standard_flat_increase_unit(ops: object) -> bool:
    """
    Recognize a conventional flat-circle increase unit.

    Odd plain stitches may sit on one side of the increase: [x,v], [3x,v].
    Even plain stitches should be split around the increase: [x,v,x],
    [2x,v,2x].  This avoids treating [2x,v] / [4x,v] as a preset flat
    circle, since those forms intentionally place the increase off-center.
    """
    if not isinstance(ops, list) or not ops:
        return False

    inc_seen = False
    unit_height: float | None = None
    before = 0
    after = 0

    for op in ops:
        if not isinstance(op, dict):
            return False
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            if st in {"skip", "-", "ch", "sl", "sl st", "ss"}:
                return False
            n = int(op.get("n", 1))
            if n <= 0:
                return False
            h = _metric_height(st)
            if unit_height is None:
                unit_height = h
            elif abs(h - unit_height) >= 1e-9:
                return False
            if inc_seen:
                after += n
            else:
                before += n
        elif t == "inc":
            n = int(op.get("n", 1))
            if n != 1 or inc_seen:
                return False
            h = _metric_height(str(op.get("base") or "x"))
            if unit_height is None:
                unit_height = h
            elif abs(h - unit_height) >= 1e-9:
                return False
            inc_seen = True
        else:
            return False

    if not inc_seen:
        return False

    plain = before + after
    if plain == 0:
        return True
    if plain % 2 == 0:
        return before == after == (plain // 2)
    return before == plain or after == plain


def _clear_repeat_period_from_ops(ops: object, prev_total: int) -> int | None:
    if prev_total <= 0 or not isinstance(ops, list) or len(ops) != 1:
        return None
    op = ops[0]
    if not isinstance(op, dict):
        return None
    if op.get("type") == "repeat":
        times = int(op.get("times", 1))
        inner = op.get("ops")
        if times < 2:
            return None
        period = _estimate_consumed(inner)
        if period > 0 and (period * times) == prev_total:
            return period
    if op.get("type") in {"stitch", "inc"}:
        # Uniform shorthand such as "6x" or "6v" has an explicit one-stitch period.
        n = int(op.get("n", 1))
        if n == prev_total and n > 1:
            return 1
    return None


def _smooth_window_for_period(window: int, period: int | None) -> int:
    window = int(window)
    if window <= 1:
        return 1
    if window % 2 == 0:
        window += 1
    if period is None or period <= 0:
        return window
    max_radius = int(math.floor(float(period) / 2.0))
    radius = min(window // 2, max_radius)
    return max(1, (radius * 2) + 1)


def _min_circular_run_length(signal: list[float], *, tol: float = 1e-9) -> int | None:
    n = len(signal)
    if n == 0:
        return None
    if n == 1:
        return 1
    if all(abs(float(v) - float(signal[0])) <= tol for v in signal):
        return n

    start = 0
    for i in range(n):
        prev = signal[(i - 1) % n]
        cur = signal[i]
        if abs(float(prev) - float(cur)) > tol:
            start = i
            break

    runs: list[int] = []
    current = 1
    prev = signal[start]
    for step in range(1, n):
        cur = signal[(start + step) % n]
        if abs(float(cur) - float(prev)) <= tol:
            current += 1
        else:
            runs.append(current)
            current = 1
            prev = cur
    runs.append(current)
    return min(runs) if runs else n


def _smooth_window_for_signal(window: int, signal: list[float]) -> int:
    window = int(window)
    if window <= 1:
        return 1
    if window % 2 == 0:
        window += 1
    run = _min_circular_run_length(signal)
    if run is None:
        return window
    radius = min(window // 2, max(0, run // 2))
    return max(1, (radius * 2) + 1)


def _is_regular_flat_increase_ops(ops: object, prev_total: int) -> bool:
    """
    Detect standard flat circle increase rounds such as 6v, [x,v]*6,
    [x,v,x]*6, [3x,v]*6, or [2x,v,2x]*6.

    These rounds should expand the whole circle evenly, not create local lobes
    at each increase. More complex rounds keep the normal local bump behavior.
    """
    if not isinstance(ops, list) or not ops:
        return False

    if len(ops) == 1 and isinstance(ops[0], dict) and ops[0].get("type") == "inc":
        n = int(ops[0].get("n", 1))
        return n == prev_total and n > 1

    period = _clear_repeat_period_from_ops(ops, prev_total)
    if period is None:
        return False
    op = ops[0]
    if not isinstance(op, dict) or op.get("type") != "repeat":
        return False
    info = _regular_flat_unit_info(op.get("ops"))
    return bool(info and info[1] and _is_standard_flat_increase_unit(op.get("ops")))


def _growth_signal_for_round(pattern: PatternSpec, round_index: int, prev_total: int) -> list[float]:
    """
    Build a per-base-stitch growth signal of length prev_total.

    Each consumed base stitch/space contributes:
    - inc/dec/cluster: shape signal (kept intentionally small)
    - stitch height: stronger positive bias for taller stitches
    """
    out = [0.0 for _ in range(max(prev_total, 1))]
    if prev_total <= 0:
        return out
    ops = pattern.rounds[round_index].stitch.ops
    if not isinstance(ops, list):
        return out

    cursor = 0
    last_base: str | None = None
    # Tunable weights (heuristics):
    # - shape ops should not dominate; they just hint where the outline may expand
    # - height bias should be more visible to reflect stitch type differences
    INC_W = 0.08
    DEC_W = -0.08
    CLUSTER_W = 0.35
    HB_W = 0.60

    def add(idx: int, v: float) -> None:
        out[idx % prev_total] += v

    def walk(op_list: list[object]) -> None:
        nonlocal cursor, last_base
        i = 0
        while i < len(op_list):
            op = op_list[i]
            if not isinstance(op, dict):
                i += 1
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = int(op.get("n", 1))
                if st in {"skip", "-"}:
                    cursor += n
                    i += 1
                    continue
                if st == "sl st":
                    if not isinstance(op.get("target"), dict):
                        cursor += 0 if _previous_op_is_chain(op_list, i) else max(0, n)
                    i += 1
                    continue
                if st == "ch":
                    occupancy, next_i = _chain_occupancy_after(op_list, i)
                    cursor += occupancy
                    i = next_i
                    continue
                # height bias: use absolute metric height so baseline stitches
                # still contribute (ensures long stitches are visibly > short).
                h = _metric_height(st)
                hb = max(0.0, h * HB_W)
                for _ in range(max(n, 0)):
                    add(cursor, hb)
                    cursor += 1
                if st in {"x", "t", "f", "e", "tr", "dtr", "sc", "hdc", "dc"}:
                    last_base = st
            elif t == "inc":
                n = int(op.get("n", 1))
                base = str(op.get("base") or "x")
                hb = max(0.0, _metric_height(base) * HB_W)
                for _ in range(max(n, 0)):
                    add(cursor, INC_W + hb)
                    cursor += 1
            elif t == "dec":
                n = int(op.get("n", 1))
                base = str(op.get("base") or "x")
                hb = max(0.0, _metric_height(base) * (HB_W * 0.8))
                for _ in range(max(n, 0)):
                    add(cursor, DEC_W + hb)
                    cursor += 2
            elif t == "skip":
                cursor += int(op.get("n", 1))
            elif t == "cluster":
                inner = op.get("ops", [])
                produced = _estimate_total_ops(inner)
                counts = _ops_counts(inner)
                inner_height = max((_metric_height(sym) for sym, count in counts.items() if count > 0), default=0.0)
                # base stitch/space consumed=1
                add(cursor, max(0.0, float(produced - 1)) * CLUSTER_W + max(0.0, inner_height * HB_W))
                cursor += 1
            elif t == "ch_space":
                inner = op.get("ops", [])
                produced = _estimate_total_ops(inner)
                counts = _ops_counts(inner)
                inner_height = max((_metric_height(sym) for sym, count in counts.items() if count > 0), default=0.0)
                add(cursor, max(0.0, float(produced - 1)) * CLUSTER_W + max(0.0, inner_height * HB_W))
                cursor += 1
            elif t == "repeat":
                times = int(op.get("times", 1))
                inner = op.get("ops", [])
                if not isinstance(inner, list):
                    i += 1
                    continue
                for _ in range(max(times, 0)):
                    walk(inner)
            i += 1

    walk(ops)
    if _is_regular_flat_increase_ops(ops, prev_total):
        pos = [v for v in out if v > 0.0]
        if pos:
            flat = sum(pos) / len(pos)
            out = [flat if v > 0.0 else 0.0 for v in out]
    return out


def _smooth_circular(signal: list[float], window: int, mode: str = "weighted") -> list[float]:
    n = len(signal)
    if n == 0:
        return []
    window = int(window)
    if window <= 1:
        return list(signal)
    if window % 2 == 0:
        window += 1
    r = window // 2
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        wsum = 0.0
        for k in range(-r, r + 1):
            weight = 1.0 if mode == "mean" else float(r + 1 - abs(k))
            s += float(signal[(i + k) % n]) * weight
            wsum += weight
        out[i] = s / wsum if wsum else float(signal[i])
    return out


def _mix_signals(raw: list[float], smoothed: list[float], raw_mix: float) -> list[float]:
    if len(raw) != len(smoothed):
        return list(smoothed)
    raw_mix = max(0.0, min(1.0, float(raw_mix)))
    smooth_mix = 1.0 - raw_mix
    return [(float(r) * raw_mix) + (float(s) * smooth_mix) for r, s in zip(raw, smoothed)]


def _normalize_bump(signal: list[float], strength: float) -> list[float]:
    if not signal:
        return []
    # Only upward bumps (ears); allow negative to reduce, but clamp at 0 for silhouette.
    strength = max(0.0, min(1.0, float(strength)))
    pos = [float(v) for v in signal if v > 0.0]
    if not pos:
        return [0.0 for _ in signal]
    # If the signal contains true empty/low slots (e.g. repeated long-stitch
    # clusters separated by sl/skip positions), keep those slots as the baseline.
    # Otherwise a motif like [t/f cluster, t/f cluster, sl] has all raised
    # stitches subtracted away and becomes a flat circle.
    baseline = 0.0 if any(float(v) <= 0.0 for v in signal) else min(pos)
    adjusted = [max(0.0, float(v) - baseline) for v in signal]
    pos = [v for v in adjusted if v > 0.0]
    if not pos:
        return [0.0 for _ in signal]

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        q = max(0.0, min(1.0, float(q)))
        xs = sorted(values)
        if len(xs) == 1:
            return xs[0]
        i = q * (len(xs) - 1)
        i0 = int(math.floor(i))
        i1 = min(i0 + 1, len(xs) - 1)
        t = i - i0
        return float(xs[i0] * (1 - t) + xs[i1] * t)

    # Peak-emphasized but *non-thresholded* normalization:
    # - scale by a robust peak (p95) instead of absolute max (less sensitive to outliers)
    # - apply a contrast curve to emphasize peaks without wiping out mid-variation
    peak = percentile(pos, 0.95)
    denom = max(peak, 0.6)

    gamma = 2.0
    out: list[float] = []
    for v in adjusted:
        x = max(0.0, float(v) / denom)
        if x > 1.0:
            x = 1.0
        x = x**gamma
        out.append(x * strength)
    return out


def _bump_field_from_opts(pattern: PatternSpec, opts: RenderOptions, round_index: int, prev_total: int) -> tuple[int, list[float]] | None:
    if not opts.circle_auto_bump or prev_total <= 0:
        return None
    scope = getattr(opts, "circle_auto_bump_scope", "outermost")
    if scope == "outermost":
        target = opts.circle_auto_bump_round
        if target is None:
            target = len(pattern.rounds) - 1
        if round_index != target:
            return None
    elif scope == "all":
        pass
    else:
        return None
    raw = _growth_signal_for_round(pattern, round_index, prev_total)
    period = _clear_repeat_period_from_ops(pattern.rounds[round_index].stitch.ops, prev_total)
    smooth_window = _smooth_window_for_period(opts.circle_auto_bump_window, period)
    smooth_window = _smooth_window_for_signal(smooth_window, raw)
    sm = _smooth_circular(raw, smooth_window, getattr(opts, "circle_auto_bump_smoothing", "weighted"))
    mixed = _mix_signals(raw, sm, _BUMP_RAW_MIX)
    bump = _normalize_bump(mixed, opts.circle_auto_bump_strength)
    return 0, bump


def _bump_value_at(pos: float, prev_total: int, start_offset: int, field: list[float]) -> float:
    if prev_total <= 0 or not field:
        return 0.0
    # `field[i]` describes the i-th base stitch, whose geometric center is at
    # i + 0.5.  Sampling against stitch centers keeps bumps symmetric across
    # the circular seam; otherwise index 0 behaves like a boundary point.
    p = (pos + start_offset - 0.5) % prev_total
    i0 = int(math.floor(p)) % prev_total
    i1 = (i0 + 1) % prev_total
    t = p - math.floor(p)
    return float(field[i0] * (1 - t) + field[i1] * t)

def _combined_bump_at(
    pos: float,
    prev_total: int,
    seg_mod: tuple[int, list[tuple[int, float]]] | None,
    field_mod: tuple[int, list[float]] | None,
) -> float:
    if prev_total <= 0:
        return 0.0
    a = 0.0
    b = 0.0
    if seg_mod is not None:
        so, segs = seg_mod
        if segs:
            a = _bump_at_pos(pos, prev_total, so, segs)
    if field_mod is not None:
        so2, field = field_mod
        if field:
            b = _bump_value_at(pos, prev_total, so2, field)
    return max(float(a), float(b))

def _modulated_ring_path_combined(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start: float,
    prev_total: int,
    seg_mod: tuple[int, list[tuple[int, float]]] | None,
    field_mod: tuple[int, list[float]] | None,
    bump_scale: float,
    inner_bump_scale: float = 0.0,
    r_cap: float | None = None,
) -> str:
    # Sample along the circle; at least 180 points, at most 900.
    steps = max(180, min(900, prev_total * 8 if prev_total > 0 else 360))
    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []
    denom = max(prev_total, 1)
    for k in range(steps + 1):
        pos = (k / steps) * denom
        ang = start + (2 * math.pi * (pos / denom))
        bump = _combined_bump_at(pos, prev_total, seg_mod, field_mod)
        ro = r_outer + bump * bump_scale
        if isinstance(r_cap, (int, float)):
            ro = min(float(r_cap), ro)
        x_out, y_out = _polar(cx, cy, ro, ang)
        ri = r_inner - bump * inner_bump_scale if inner_bump_scale else r_inner
        if ri < 0.0:
            ri = 0.0
        if ri > ro:
            ri = ro
        x_in, y_in = _polar(cx, cy, ri, ang)
        outer_pts.append((x_out, y_out))
        inner_pts.append((x_in, y_in))

    if not outer_pts:
        return ""
    d = [f"M {outer_pts[0][0]:.3f} {outer_pts[0][1]:.3f} "]
    for x, y in outer_pts[1:]:
        d.append(f"L {x:.3f} {y:.3f} ")
    for x, y in reversed(inner_pts):
        d.append(f"L {x:.3f} {y:.3f} ")
    d.append("Z ")
    return "".join(d)


def _ring_path_from_radius_samples(
    cx: float,
    cy: float,
    start: float,
    inner_radii: list[float],
    outer_radii: list[float],
) -> str:
    n = min(len(inner_radii), len(outer_radii))
    if n < 2:
        return ""
    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []
    denom = max(n - 1, 1)
    for k in range(n):
        ang = start + (2 * math.pi * (k / denom))
        x_out, y_out = _polar(cx, cy, max(0.0, outer_radii[k]), ang)
        x_in, y_in = _polar(cx, cy, max(0.0, inner_radii[k]), ang)
        outer_pts.append((x_out, y_out))
        inner_pts.append((x_in, y_in))

    d = [f"M {outer_pts[0][0]:.3f} {outer_pts[0][1]:.3f} "]
    for x, y in outer_pts[1:]:
        d.append(f"L {x:.3f} {y:.3f} ")
    for x, y in reversed(inner_pts):
        d.append(f"L {x:.3f} {y:.3f} ")
    d.append("Z ")
    return "".join(d)


def _modulated_ring_path(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start: float,
    prev_total: int,
    start_offset: int,
    segments: list[tuple[int, float]],
    bump_scale: float,
) -> str:
    # Sample along the circle; at least 180 points, at most 900.
    steps = max(180, min(900, prev_total * 8 if prev_total > 0 else 360))
    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []
    denom = max(prev_total, 1)
    for k in range(steps + 1):
        pos = (k / steps) * denom
        ang = start + (2 * math.pi * (pos / denom))
        bump = _bump_at_pos(pos, prev_total, start_offset, segments)
        ro = r_outer + bump * bump_scale
        x_out, y_out = _polar(cx, cy, ro, ang)
        x_in, y_in = _polar(cx, cy, r_inner, ang)
        outer_pts.append((x_out, y_out))
        inner_pts.append((x_in, y_in))

    if not outer_pts:
        return ""
    d = [f"M {outer_pts[0][0]:.3f} {outer_pts[0][1]:.3f} "]
    for x, y in outer_pts[1:]:
        d.append(f"L {x:.3f} {y:.3f} ")
    for x, y in reversed(inner_pts):
        d.append(f"L {x:.3f} {y:.3f} ")
    d.append("Z ")
    return "".join(d)


def _modulated_ring_path_field(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start: float,
    prev_total: int,
    start_offset: int,
    field: list[float],
    bump_scale: float,
) -> str:
    steps = max(180, min(900, prev_total * 8 if prev_total > 0 else 360))
    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []
    denom = max(prev_total, 1)
    for k in range(steps + 1):
        pos = (k / steps) * denom
        ang = start + (2 * math.pi * (pos / denom))
        bump = _bump_value_at(pos, prev_total, start_offset, field)
        ro = r_outer + bump * bump_scale
        x_out, y_out = _polar(cx, cy, ro, ang)
        x_in, y_in = _polar(cx, cy, r_inner, ang)
        outer_pts.append((x_out, y_out))
        inner_pts.append((x_in, y_in))

    if not outer_pts:
        return ""
    d = [f"M {outer_pts[0][0]:.3f} {outer_pts[0][1]:.3f} "]
    for x, y in outer_pts[1:]:
        d.append(f"L {x:.3f} {y:.3f} ")
    for x, y in reversed(inner_pts):
        d.append(f"L {x:.3f} {y:.3f} ")
    d.append("Z ")
    return "".join(d)

def _ops_to_placements(ops: object) -> tuple[list[tuple[float, str]], int, int]:
    """
    Convert ops into ordered glyph placements along previous-round stitch positions.

    Returns:
    - placements: list of (pos, symbol) where pos is in "consumed stitch units"
    - consumed: how many previous stitches were consumed
    - produced: how many stitches were produced
    """
    if not isinstance(ops, list):
        return [], 0, 0
    placements: list[tuple[float, str]] = []
    cursor = 0
    produced = 0

    def emit(pos: float, sym: str) -> None:
        placements.append((pos, sym))

    def walk(op_list: list[object]) -> None:
        nonlocal cursor, produced
        i = 0
        while i < len(op_list):
            op = op_list[i]
            if not isinstance(op, dict):
                i += 1
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = int(op.get("n", 1))
                if st in {"skip", "-"}:
                    cursor += n
                    i += 1
                    continue
                if st == "sl st":
                    if not isinstance(op.get("target"), dict):
                        cursor += 0 if _previous_op_is_chain(op_list, i) else max(0, n)
                    i += 1
                    continue
                if st == "ch":
                    occupancy, next_i = _chain_occupancy_after(op_list, i)
                    cursor += occupancy
                    produced += n
                    i = next_i
                    continue
                for _ in range(max(n, 0)):
                    emit(cursor + 0.5, st)
                    cursor += 1
                    produced += 1
            elif t == "skip":
                cursor += int(op.get("n", 1))
            elif t == "inc":
                base = str(op.get("base") or "x")
                n = int(op.get("n", 1))
                for _ in range(max(n, 0)):
                    pos = cursor + 0.5
                    emit(pos, f"inc:{base}")
                    emit(pos - 0.15, base)
                    emit(pos + 0.15, base)
                    cursor += 1
                    produced += 2
            elif t == "dec":
                base = str(op.get("base") or "x")
                n = int(op.get("n", 1))
                for _ in range(max(n, 0)):
                    pos = cursor + 1.0
                    emit(pos, f"dec:{base}")
                    emit(pos, base)
                    cursor += 2
                    produced += 1
            elif t == "cluster":
                inner = op.get("ops", [])
                sym = "cluster"
                inner_total = 0
                if isinstance(inner, list):
                    counts = _ops_counts(inner)
                    if counts:
                        sym = max(counts, key=lambda k: _metric_height(k))
                    inner_total = _estimate_total_ops(inner)
                emit(cursor + 0.5, sym)
                cursor += 1
                produced += max(1, inner_total)
            elif t == "ch_space":
                inner = op.get("ops", [])
                inner_syms = _ops_to_produced_placements(inner)
                produced += len(inner_syms)
                cursor += 1
            elif t == "repeat":
                times = int(op.get("times", 1))
                inner = op.get("ops", [])
                if not isinstance(inner, list):
                    i += 1
                    continue
                if times > 2000:
                    raise ValueError("repeat.times too large for rendering")
                for _ in range(max(times, 0)):
                    walk(inner)
            i += 1

    walk(ops)
    return placements, cursor, produced


def _produced_per_base_stitch(ops: object, prev_total: int) -> list[float]:
    """
    Estimate how many stitches are produced into each consumed base stitch slot.

    Returns a list of length `prev_total` (or 1 when prev_total<=0).
    Values may be fractional for dec (2 bases -> 1 stitch).
    """
    n = max(int(prev_total), 1)
    out = [0.0 for _ in range(n)]
    if prev_total <= 0 or not isinstance(ops, list):
        return out
    cursor = 0

    def add(i: int, v: float) -> None:
        out[i % prev_total] += v

    def walk(op_list: list[object]) -> None:
        nonlocal cursor
        i = 0
        while i < len(op_list):
            op = op_list[i]
            if not isinstance(op, dict):
                i += 1
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                nst = int(op.get("n", 1))
                if st in {"skip", "-"}:
                    cursor += nst
                    i += 1
                    continue
                if st == "sl st":
                    if not isinstance(op.get("target"), dict):
                        cursor += 0 if _previous_op_is_chain(op_list, i) else max(0, nst)
                    i += 1
                    continue
                if st == "ch":
                    occupancy, next_i = _chain_occupancy_after(op_list, i)
                    cursor += occupancy
                    i = next_i
                    continue
                for _ in range(max(nst, 0)):
                    add(cursor, 1.0)
                    cursor += 1
            elif t == "skip":
                cursor += int(op.get("n", 1))
            elif t == "inc":
                ninc = int(op.get("n", 1))
                for _ in range(max(ninc, 0)):
                    add(cursor, 2.0)
                    cursor += 1
            elif t == "cluster":
                inner = op.get("ops", [])
                produced = _estimate_total_ops(inner)
                add(cursor, float(max(produced, 0)))
                cursor += 1
            elif t == "ch_space":
                cursor += 1
            elif t == "dec":
                ndec = int(op.get("n", 1))
                for _ in range(max(ndec, 0)):
                    # one produced stitch from two consumed bases
                    add(cursor, 0.5)
                    add(cursor + 1, 0.5)
                    cursor += 2
            elif t == "repeat":
                times = int(op.get("times", 1))
                inner = op.get("ops", [])
                if not isinstance(inner, list):
                    i += 1
                    continue
                for _ in range(max(times, 0)):
                    walk(inner)
            i += 1

    walk(ops)
    return out


def _modulated_ring_path_combined_produced(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start: float,
    prev_total: int,
    seg_mod: tuple[int, list[tuple[int, float]]] | None,
    field_mod: tuple[int, list[float]] | None,
    bump_scale: float,
    *,
    inner_bump_scale: float = 0.0,
    r_cap: float | None = None,
    ops: object,
) -> str:
    """
    Method 2 outline sampling:
    - split by base stitches (prev_total segments)
    - within each base stitch, subdivide by produced count
    """
    prev_total = int(prev_total)
    if prev_total <= 0:
        return ""
    produced = _produced_per_base_stitch(ops, prev_total)
    positions: list[float] = []
    for i in range(prev_total):
        c = produced[i]
        # Ensure at least one sample per base stitch to keep continuity
        m = int(round(c)) if c > 0 else 1
        m = max(1, min(12, m))
        for j in range(m):
            positions.append(i + (j + 0.5) / m)
    # Ensure the seam is properly sampled. Without endpoints at 0 and prev_total,
    # the path would close with a chord across the seam, creating a visible "gap".
    positions.insert(0, 0.0)
    positions.append(float(prev_total))
    if not positions:
        return ""

    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []
    denom = float(prev_total)
    for pos in positions:
        ang = start + (2 * math.pi * (pos / denom))
        bump = _combined_bump_at(pos, prev_total, seg_mod, field_mod)
        ro = r_outer + bump * bump_scale
        if isinstance(r_cap, (int, float)):
            ro = min(float(r_cap), ro)
        ri = r_inner - bump * inner_bump_scale if inner_bump_scale else r_inner
        if ri < 0.0:
            ri = 0.0
        if ri > ro:
            ri = ro
        x_out, y_out = _polar(cx, cy, ro, ang)
        x_in, y_in = _polar(cx, cy, ri, ang)
        outer_pts.append((x_out, y_out))
        inner_pts.append((x_in, y_in))

    d = [f"M {outer_pts[0][0]:.3f} {outer_pts[0][1]:.3f} "]
    for x, y in outer_pts[1:]:
        d.append(f"L {x:.3f} {y:.3f} ")
    for x, y in reversed(inner_pts):
        d.append(f"L {x:.3f} {y:.3f} ")
    d.append("Z ")
    return "".join(d)


def _ops_to_produced_placements(ops: object) -> list[str]:
    """
    Flatten ops into a per-produced-stitch symbol sequence.

    Unlike `_ops_to_placements`, this ignores "consumed base stitch" positions and
    instead returns one symbol per produced stitch, which is useful for a more
    visually "filled" texture around a complete ring.
    """
    if not isinstance(ops, list):
        return []
    out: list[str] = []

    def walk(op_list: list[object]) -> None:
        for op in op_list:
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = int(op.get("n", 1))
                if st in {"skip", "-", "sl st"}:
                    continue
                if st == "ch":
                    continue
                out.extend([st] * max(n, 0))
            elif t == "skip":
                continue
            elif t == "inc":
                base = str(op.get("base") or "x")
                n = int(op.get("n", 1))
                # One base stitch produces 2 stitches of the base type.
                out.extend([base] * (2 * max(n, 0)))
            elif t == "dec":
                base = str(op.get("base") or "x")
                n = int(op.get("n", 1))
                # Two base stitches consume into 1 produced stitch (render as base type).
                out.extend([base] * max(n, 0))
            elif t == "ch_space":
                continue
            elif t == "repeat":
                times = int(op.get("times", 1))
                inner = op.get("ops", [])
                if not isinstance(inner, list):
                    continue
                if times > 2000:
                    raise ValueError("repeat.times too large for rendering")
                for _ in range(max(times, 0)):
                    walk(inner)

    walk(ops)
    return out


def _glyph_for_symbol(sym: str) -> dict:
    if sym.startswith("inc:"):
        base = sym.split(":", 1)[1]
        base_norm = {"sc": "x", "hdc": "t", "dc": "f"}.get(base, base)
        base_g = DEFAULT_GLYPHS.get(base_norm) or DEFAULT_GLYPHS["x"]
        return {"kind": "inc", "length": float(base_g.get("length", 0.35)), "bars": int(base_g.get("bars", 0) or 0)}
    if sym.startswith("dec:"):
        base = sym.split(":", 1)[1]
        base_norm = {"sc": "x", "hdc": "t", "dc": "f"}.get(base, base)
        base_g = DEFAULT_GLYPHS.get(base_norm) or DEFAULT_GLYPHS["x"]
        return {"kind": "dec", "length": float(base_g.get("length", 0.35)), "bars": int(base_g.get("bars", 0) or 0)}
    if sym == "skip":
        return {"kind": "skip", "length": 0.0, "bars": 0}
    normalized = {"sc": "x", "hdc": "t", "dc": "f"}.get(sym, sym)
    return DEFAULT_GLYPHS.get(normalized) or DEFAULT_GLYPHS["x"]


def _chain_events_from_ops(ops: object) -> tuple[list[dict[str, float | int | str]], int]:
    if not isinstance(ops, list):
        return [], 0
    events: list[dict[str, float | int | str]] = []
    chain_refs: list[dict[str, float | int]] = []
    next_chain_id = 0
    cursor = 0

    def anchor(pos: float, sym: str) -> None:
        events.append({"type": "anchor", "pos": float(pos), "sym": sym})

    def chain(n: int, start: float, end: float) -> None:
        nonlocal next_chain_id
        if n > 0:
            chain_id = next_chain_id
            next_chain_id += 1
            events.append({"type": "chain", "count": int(n), "chain_id": chain_id, "start": float(start), "end": float(end)})
            chain_refs.append({"count": int(n), "start": float(start), "end": float(end), "chain_id": chain_id})

    def chain_target(target: object) -> dict[str, float | int] | None:
        if not isinstance(target, dict) or target.get("kind") != "ch":
            return None
        if not chain_refs:
            return None
        try:
            index = int(target.get("index", 1))
        except (TypeError, ValueError):
            index = 1
        # `sl@ch3` conventionally joins into the initial chain of this round.
        # Use the center of the Nth chain segment along that chain's occupied span.
        ref = chain_refs[0]
        count = max(int(ref.get("count", 1)), 1)
        start = float(ref.get("start", 0.0))
        end = float(ref.get("end", start + 1.0))
        idx = min(max(index, 1), count)
        return {
            "pos": start + (end - start) * ((idx - 0.5) / count),
            "target_chain_id": int(ref.get("chain_id", 0)),
            "target_chain_index": idx,
        }

    def walk(op_list: list[object]) -> None:
        nonlocal cursor
        i = 0
        while i < len(op_list):
            op = op_list[i]
            if not isinstance(op, dict):
                i += 1
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = int(op.get("n", 1))
                if st in {"skip", "-"}:
                    cursor += n
                    i += 1
                    continue
                if st == "ch":
                    occupancy, next_i = _chain_occupancy_after(op_list, i)
                    chain(n, cursor, cursor + occupancy)
                    cursor += occupancy
                    i = next_i
                    continue
                if st == "sl st":
                    target = chain_target(op.get("target"))
                    if target is not None:
                        events.append({"type": "anchor", "pos": float(target["pos"]), "sym": st, **target})
                    else:
                        anchor(float(cursor), st)
                        cursor += 0 if _previous_op_is_chain(op_list, i) else max(0, n)
                    i += 1
                    continue
                for _ in range(max(n, 0)):
                    anchor(cursor + 0.5, st)
                    cursor += 1
            elif t == "skip":
                cursor += int(op.get("n", 1))
            elif t == "inc":
                n = int(op.get("n", 1))
                base = str(op.get("base") or "x")
                for _ in range(max(n, 0)):
                    anchor(cursor + 0.5, base)
                    cursor += 1
            elif t == "dec":
                n = int(op.get("n", 1))
                base = str(op.get("base") or "x")
                for _ in range(max(n, 0)):
                    anchor(cursor + 1.0, base)
                    cursor += 2
            elif t == "cluster":
                inner = op.get("ops", [])
                sym = "cluster"
                if isinstance(inner, list):
                    counts = _ops_counts(inner)
                    if counts:
                        sym = max(counts, key=lambda k: _metric_height(k))
                anchor(cursor + 0.5, sym)
                cursor += 1
            elif t == "ch_space":
                inner = op.get("ops", [])
                sym = "ch_space"
                if isinstance(inner, list):
                    counts = _ops_counts(inner)
                    if counts:
                        sym = max(counts, key=lambda k: _metric_height(k))
                anchor(cursor + 0.5, sym)
                cursor += 1
            elif t == "repeat":
                times = int(op.get("times", 1))
                inner = op.get("ops", [])
                if not isinstance(inner, list):
                    i += 1
                    continue
                if times > 2000:
                    raise ValueError("repeat.times too large for rendering")
                for _ in range(max(times, 0)):
                    walk(inner)
            i += 1

    walk(ops)
    return events, cursor


def _chain_spans_from_ops(ops: object, *, default_anchor: float | None = None) -> list[dict[str, float | int | None]]:
    events, _consumed = _chain_events_from_ops(ops)
    spans: list[dict[str, float | int | None]] = []
    last_anchor: float | None = default_anchor
    last_anchor_sym: str | None = None
    i = 0
    while i < len(events):
        event = events[i]
        if event.get("type") == "anchor":
            last_anchor = float(event.get("pos", 0.0))
            last_anchor_sym = str(event.get("sym", "")) or None
            i += 1
            continue
        if event.get("type") != "chain":
            i += 1
            continue
        count = int(event.get("count", 0))
        chain_id = int(event.get("chain_id", -1))
        chain_start = float(event.get("start", last_anchor if last_anchor is not None else 0.0))
        chain_end = float(event.get("end", chain_start + 1.0))
        if count <= 0 or last_anchor is None:
            i += 1
            continue
        j = i + 1
        next_anchor: float | None = None
        next_anchor_sym: str | None = None
        next_target_chain_id: int | None = None
        next_target_chain_index: int | None = None
        while j < len(events):
            nxt = events[j]
            if nxt.get("type") == "anchor":
                next_anchor = float(nxt.get("pos", 0.0))
                next_anchor_sym = str(nxt.get("sym", "")) or None
                if "target_chain_id" in nxt:
                    next_target_chain_id = int(nxt.get("target_chain_id", -1))
                    next_target_chain_index = int(nxt.get("target_chain_index", 1))
                break
            j += 1
        span = {
            "from": chain_start if last_anchor_sym in {"sl st", "sl", "ss"} else last_anchor,
            "to": chain_end if next_anchor_sym in {"sl st", "sl", "ss"} else next_anchor,
            "count": count,
            "chain_id": chain_id,
            "from_sym": last_anchor_sym,
            "to_sym": next_anchor_sym,
        }
        if next_target_chain_id is not None:
            span["to_target_chain_id"] = next_target_chain_id
            span["to_target_chain_index"] = next_target_chain_index or 1
        spans.append(span)
        i += 1
    return spans


def _max_chain_span_count(pattern: PatternSpec) -> int:
    max_count = 0
    for r in pattern.rounds:
        for span in _chain_spans_from_ops(r.stitch.ops, default_anchor=0.0):
            max_count = max(max_count, int(span.get("count") or 0))
    return max_count


def _leading_chain_occupancy(ops: object) -> float | None:
    if not isinstance(ops, list) or not ops:
        return None
    first = ops[0]
    if not isinstance(first, dict) or first.get("type") != "stitch" or str(first.get("st", "")) != "ch":
        return None
    occupancy, _next_i = _chain_occupancy_after(ops, 0)
    return float(occupancy)


def _quad_bezier_length(
    x0: float,
    y0: float,
    xc: float,
    yc: float,
    x1: float,
    y1: float,
    *,
    samples: int = 24,
) -> float:
    total = 0.0
    px, py = x0, y0
    for i in range(1, max(2, samples) + 1):
        t = i / max(2, samples)
        mt = 1.0 - t
        x = (mt * mt * x0) + (2.0 * mt * t * xc) + (t * t * x1)
        y = (mt * mt * y0) + (2.0 * mt * t * yc) + (t * t * y1)
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return total


def _chain_quad_control(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    control_angle: float,
    desired_len: float,
) -> tuple[float, float] | None:
    """Return a quadratic control point only when the chain is long enough to arc."""
    chord = math.hypot(x1 - x0, y1 - y0)
    if float(desired_len) <= chord + 0.001:
        return None
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    nx = math.cos(control_angle)
    ny = math.sin(control_angle)
    lo = 0.0
    hi = max(float(desired_len) - chord, chord * 0.25, 1.0)
    for _ in range(12):
        test_x = mid_x + nx * hi
        test_y = mid_y + ny * hi
        if _quad_bezier_length(x0, y0, test_x, test_y, x1, y1) >= desired_len:
            break
        hi *= 1.5
    for _ in range(18):
        lift = (lo + hi) / 2.0
        test_x = mid_x + nx * lift
        test_y = mid_y + ny * lift
        if _quad_bezier_length(x0, y0, test_x, test_y, x1, y1) < desired_len:
            lo = lift
        else:
            hi = lift
    return mid_x + nx * hi, mid_y + ny * hi


def _render_circle_chain_spans(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    r_inner: float,
    r_outer: float,
    prev_total: int,
    *,
    chain_unit_px: float | None = None,
    r_cap: float | None = None,
) -> str:
    # A round may begin with chains before the next worked stitch, e.g. "5ch x".
    # Treat the beginning of the round as the chain's start anchor; otherwise the
    # leading chain has no previous stitch event and disappears.
    default_anchor = 0.0
    spans = _chain_spans_from_ops(pattern.rounds[round_index].stitch.ops, default_anchor=default_anchor)
    if not spans:
        return ""
    cx = cy = opts.size / 2
    denom = max(prev_total, 1)
    start = (-math.pi / 2) + (math.pi * float(opts.rotation_deg) / 180.0)
    band = max(r_outer - r_inner, opts.size * 0.055)
    chain_unit = _chain_unit_from_band(chain_unit_px if chain_unit_px is not None else band)
    chain_color = pattern.rounds[round_index].color
    block_w = _chain_block_width(opts)
    anchor_inset = _chain_anchor_inset(opts)
    outline_w = block_w + (2.0 * max(1.0, float(opts.stroke_width)))
    base_r = r_inner + (block_w / 2.0)
    parts: list[str] = []
    chain_paths: dict[int, dict[str, object]] = {}

    def anchor_radius(sym: object) -> float:
        if not isinstance(sym, str) or sym in {"", "sl st", "sl", "ss", "skip", "-", "ch"}:
            return base_r
        if opts.glyph_full_height:
            return max(base_r, r_outer - anchor_inset)
        height_frac = max(0.18, min(1.0, float(_metric_height(sym)) / 6.0))
        radius = r_inner + (r_outer - r_inner) * (0.35 + (0.65 * height_frac))
        return max(base_r, radius - anchor_inset)

    def anchor_point(pos: float, sym: object) -> tuple[float, float]:
        ang = start + (2 * math.pi * (pos / denom))
        return _polar(cx, cy, anchor_radius(sym), ang)

    def point_on_record(record: dict[str, object], index: int) -> tuple[float, float] | None:
        count = max(int(record.get("count", 1)), 1)
        t = min(max((int(index) - 0.5) / count, 0.0), 1.0)
        kind = str(record.get("kind", ""))
        if kind == "line":
            x0, y0 = record["p0"]  # type: ignore[misc]
            x1, y1 = record["p1"]  # type: ignore[misc]
            return float(x0) + (float(x1) - float(x0)) * t, float(y0) + (float(y1) - float(y0)) * t
        if kind == "quad":
            x0, y0 = record["p0"]  # type: ignore[misc]
            xc, yc = record["pc"]  # type: ignore[misc]
            x1, y1 = record["p1"]  # type: ignore[misc]
            mt = 1.0 - t
            x = (mt * mt * float(x0)) + (2.0 * mt * t * float(xc)) + (t * t * float(x1))
            y = (mt * mt * float(y0)) + (2.0 * mt * t * float(yc)) + (t * t * float(y1))
            return x, y
        return None

    for span in spans:
        from_pos = float(span["from"])
        to_pos_raw = span.get("to")
        count = int(span.get("count") or 1)
        a0 = start + (2 * math.pi * (from_pos / denom))
        from_sym = span.get("from_sym")
        to_sym = span.get("to_sym")
        particle_attrs = _chain_particle_attrs(opts, count)
        target_xy: tuple[float, float] | None = None
        if "to_target_chain_id" in span:
            try:
                target_record = chain_paths.get(int(span.get("to_target_chain_id", -1)))
                if target_record is not None:
                    target_xy = point_on_record(target_record, int(span.get("to_target_chain_index", 1)))
            except (TypeError, ValueError):
                target_xy = None
        # Chains do not create a filled band of their own. They leave the
        # previous round and land back on a previous-round stitch/space; the
        # following non-chain stitch starts from that landing point.
        if to_pos_raw is None:
            chain_len = max(chain_unit * count, band * 0.75)
            x0, y0 = anchor_point(from_pos, from_sym)
            current_r0 = math.hypot(x0 - cx, y0 - cy)
            r1 = current_r0 + chain_len
            if isinstance(r_cap, (int, float)):
                r1 = min(float(r_cap) - (block_w / 2.0), r1)
                r1 = max(current_r0, r1)
            x1, y1 = _polar(cx, cy, r1, a0)
            chain_id = int(span.get("chain_id", -1))
            if chain_id >= 0:
                chain_paths[chain_id] = {"kind": "line", "count": count, "p0": (x0, y0), "p1": (x1, y1)}
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{opts.stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{chain_color}" stroke-width="{block_w:.3f}" stroke-linecap="round"/>\n'
            )
            continue

        to_pos = float(to_pos_raw)
        if to_pos <= from_pos:
            to_pos += denom
        amid_pos = (from_pos + to_pos) / 2.0
        a1 = start + (2 * math.pi * (to_pos / denom))
        amid = start + (2 * math.pi * (amid_pos / denom))
        x0, y0 = anchor_point(from_pos, from_sym)
        if target_xy is None:
            x1, y1 = anchor_point(to_pos, to_sym)
        else:
            x1, y1 = target_xy
        desired_len = chain_unit * count
        control = _chain_quad_control(x0, y0, x1, y1, amid, desired_len)
        chain_id = int(span.get("chain_id", -1))
        if control is None:
            if chain_id >= 0:
                chain_paths[chain_id] = {"kind": "line", "count": count, "p0": (x0, y0), "p1": (x1, y1)}
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{opts.stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{chain_color}" stroke-width="{block_w:.3f}" stroke-linecap="round"/>\n'
            )
        else:
            xm, ym = control
            d = f"M {x0:.3f} {y0:.3f} Q {xm:.3f} {ym:.3f} {x1:.3f} {y1:.3f}"
            if chain_id >= 0:
                chain_paths[chain_id] = {"kind": "quad", "count": count, "p0": (x0, y0), "pc": (xm, ym), "p1": (x1, y1)}
            parts.append(
                f'<path d="{d}" fill="none"{particle_attrs} stroke="{opts.stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            parts.append(
                f'<path d="{d}" fill="none"{particle_attrs} stroke="{chain_color}" stroke-width="{block_w:.3f}" stroke-linecap="round"/>\n'
            )
    return "".join(parts)


def _ch_space_groups_from_ops(ops: object) -> list[list[object]]:
    if not isinstance(ops, list):
        return []
    groups: list[list[object]] = []

    def walk(op_list: list[object]) -> None:
        for op in op_list:
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "ch_space":
                inner = op.get("ops", [])
                if isinstance(inner, list):
                    groups.append(inner)
            elif t == "repeat":
                times = max(0, int(op.get("times", 1)))
                inner = op.get("ops", [])
                if isinstance(inner, list):
                    for _ in range(times):
                        walk(inner)

    walk(ops)
    return groups


def _ch_space_chain_lift(count: int, chain_unit: float) -> float:
    """
    Target visible length for chains drawn inside ch-spaces.

    Keep ch-space chains on the same unit scale as ordinary round chains:
    one ch uses the same visual height as one baseline stitch.
    """
    count = max(1, int(count))
    return _chain_unit_from_band(chain_unit) * float(count)


def _chain_block_width(opts: RenderOptions) -> float:
    """Visible yarn width for ch chains; keep it stable across rounds."""
    return max(2.0, float(opts.size) * 0.026 * max(0.2, float(opts.chain_width_scale)))


def _chain_particle_attrs(opts: RenderOptions, count: int) -> str:
    """SVG attributes that turn a continuous chain path into rounded ch particles."""
    if opts.chain_render_style != "beads":
        return ""
    # Normalize the path length so each `ch` gets one rounded dash.
    # The dash is intentionally longer than the gap to keep the chain continuous
    # while still revealing individual lock-stitch particles.
    return f' pathLength="{max(1, int(count))}" stroke-dasharray="0.78 0.22"'


def _chain_anchor_inset(opts: RenderOptions) -> float:
    """Small visual inset so chain ends sit slightly inside the receiving stitch."""
    return max(1.0, _chain_block_width(opts) * 0.35)


def _ch_space_slot_angles(start_ang: float, sector_sweep: float, gap: float, item_index: int) -> tuple[float, float, float]:
    """
    Local stitch slots inside one ch-space.

    The ch-space is treated as a small finite list of render slots. Stitches use
    the slot center; base-to-base chains use the slot start/end so repeated
    chains in the same hole do not collapse onto one point.
    """
    a0 = float(start_ang) + (float(item_index) * float(sector_sweep)) + (float(gap) / 2.0)
    sweep = max(0.001, float(sector_sweep) - float(gap))
    a1 = a0 + sweep
    return a0, a1, a0 + (sweep / 2.0)


def _ch_space_slot_span_angles(
    start_ang: float,
    sector_sweep: float,
    gap: float,
    first_slot: int,
    last_slot: int,
) -> tuple[float, float, float]:
    a0 = float(start_ang) + (float(first_slot) * float(sector_sweep)) + (float(gap) / 2.0)
    slot_count = max(1, int(last_slot) - int(first_slot) + 1)
    sweep = max(0.001, (float(slot_count) * float(sector_sweep)) - float(gap))
    a1 = a0 + sweep
    return a0, a1, a0 + (sweep / 2.0)


def _ch_space_slot_boundary_angles(start_ang: float, sector_sweep: float, first_slot: int, last_slot: int) -> tuple[float, float, float]:
    a0 = float(start_ang) + (float(first_slot) * float(sector_sweep))
    a1 = float(start_ang) + (float(int(last_slot) + 1) * float(sector_sweep))
    return a0, a1, (a0 + a1) / 2.0


def _ch_space_item_slot_spans(items: list[dict[str, object]]) -> tuple[list[tuple[int, int]], int]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for item in items:
        demand = 2 if item.get("kind") == "chain" else 1
        spans.append((cursor, cursor + demand - 1))
        cursor += demand
    return spans, cursor


def _ch_space_layout(outward: float, slot_count: int) -> tuple[float, float, float]:
    if slot_count <= 0:
        return float(outward), 0.0, 0.0
    # Work into a ch-space by distributing local slots along the previous ch
    # arch. Use a wide fan so the bases fill the arch instead of clustering
    # around the center of the hole.
    total_sweep = min(math.pi, math.radians(36.0 + (14.0 * float(slot_count))))
    sector_sweep = total_sweep / float(slot_count)
    gap = min(math.radians(1.0), sector_sweep * 0.08)
    return float(outward) - (total_sweep / 2.0), sector_sweep, gap


def _ch_space_stitch_radii(hole_r: float, band: float, height_frac: float) -> tuple[float, float]:
    """
    Radial span for a stitch worked into a chain space.

    Unlike a normal round, the stitch should not start at the center of the
    chain-space hole. It starts from the inner edge of the hole and grows
    outward, closer to how stitches are worked into an existing ch arch.
    """
    inner = max(1.5, min(float(hole_r) * 1.02, float(band) * 0.48))
    body = max(float(band) * 0.28, float(band) * (0.35 + (0.65 * float(height_frac))))
    return inner, inner + body


def _chain_path_point(record: dict[str, object], t: float) -> tuple[float, float] | None:
    path = record.get("path")
    if not isinstance(path, dict):
        return None
    t = max(0.0, min(1.0, float(t)))
    kind = str(path.get("kind", ""))
    if kind == "line":
        x0, y0 = path["p0"]  # type: ignore[misc]
        x1, y1 = path["p1"]  # type: ignore[misc]
        return float(x0) + (float(x1) - float(x0)) * t, float(y0) + (float(y1) - float(y0)) * t
    if kind == "quad":
        x0, y0 = path["p0"]  # type: ignore[misc]
        xc, yc = path["pc"]  # type: ignore[misc]
        x1, y1 = path["p1"]  # type: ignore[misc]
        mt = 1.0 - t
        x = (mt * mt * float(x0)) + (2.0 * mt * t * float(xc)) + (t * t * float(x1))
        y = (mt * mt * float(y0)) + (2.0 * mt * t * float(yc)) + (t * t * float(y1))
        return x, y
    return None


def _ch_space_slot_t(slot_count: int, slot_index: float) -> float:
    if slot_count <= 0:
        return 0.5
    inset = min(0.035, 0.18 / float(slot_count))
    raw = float(slot_index) / float(slot_count)
    return inset + ((1.0 - (2.0 * inset)) * raw)


def _outward_from_center(x: float, y: float, cx: float, cy: float, fallback_angle: float) -> tuple[float, float]:
    dx = float(x) - float(cx)
    dy = float(y) - float(cy)
    length = math.hypot(dx, dy)
    if length > 0.0001:
        return dx / length, dy / length
    return math.cos(fallback_angle), math.sin(fallback_angle)


def _circle_chain_hole_records(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    r_inner: float,
    r_outer: float,
    prev_total: int,
    *,
    r_cap: float | None = None,
) -> list[dict[str, object]]:
    spans = _chain_spans_from_ops(pattern.rounds[round_index].stitch.ops, default_anchor=0.0)
    if not spans:
        return []
    cx = cy = opts.size / 2
    denom = max(prev_total, 1)
    start = (-math.pi / 2) + (math.pi * float(opts.rotation_deg) / 180.0)
    band = max(r_outer - r_inner, opts.size * 0.055)
    chain_unit = max(12.0, band * 0.72)
    block_w = _chain_block_width(opts)
    base_r = r_inner + (block_w / 2.0)
    records: list[dict[str, object]] = []

    for span in spans:
        to_pos_raw = span.get("to")
        if to_pos_raw is None:
            continue
        from_pos = float(span["from"])
        to_pos = float(to_pos_raw)
        count = int(span.get("count") or 1)
        if to_pos <= from_pos:
            to_pos += denom
        amid_pos = (from_pos + to_pos) / 2.0
        a0 = start + (2 * math.pi * (from_pos / denom))
        a1 = start + (2 * math.pi * (to_pos / denom))
        amid = start + (2 * math.pi * (amid_pos / denom))
        x0, y0 = _polar(cx, cy, base_r, a0)
        x1, y1 = _polar(cx, cy, base_r, a1)
        desired_len = chain_unit * count
        control = _chain_quad_control(x0, y0, x1, y1, amid, desired_len)
        if control is None:
            arc_mid_x = (x0 + x1) / 2.0
            arc_mid_y = (y0 + y1) / 2.0
            path: dict[str, object] = {"kind": "line", "p0": (x0, y0), "p1": (x1, y1)}
        else:
            xm, ym = control
            arc_mid_x = (0.25 * x0) + (0.5 * xm) + (0.25 * x1)
            arc_mid_y = (0.25 * y0) + (0.5 * ym) + (0.25 * y1)
            path = {"kind": "quad", "p0": (x0, y0), "pc": (xm, ym), "p1": (x1, y1)}
        base_mid_x, base_mid_y = _polar(cx, cy, base_r, amid)
        hole_x = (arc_mid_x + base_mid_x) / 2.0
        hole_y = (arc_mid_y + base_mid_y) / 2.0
        hole_r = max(block_w * 0.8, min(band * 0.65, math.hypot(arc_mid_x - base_mid_x, arc_mid_y - base_mid_y) * 0.45))
        records.append({"x": hole_x, "y": hole_y, "r": hole_r, "path": path, "count": count})
    return records


def _circle_chain_hole_centers(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    r_inner: float,
    r_outer: float,
    prev_total: int,
    *,
    r_cap: float | None = None,
) -> list[tuple[float, float, float]]:
    return [
        (float(record["x"]), float(record["y"]), float(record["r"]))
        for record in _circle_chain_hole_records(pattern, opts, round_index, r_inner, r_outer, prev_total, r_cap=r_cap)
    ]


def _ch_space_items_from_ops(op_list: list[object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    def neighbor_anchor(start_index: int, step: int) -> str:
        k = start_index
        while 0 <= k < len(op_list):
            candidate = op_list[k]
            if not isinstance(candidate, dict):
                k += step
                continue
            t2 = candidate.get("type")
            if t2 == "stitch":
                st2 = str(candidate.get("st", ""))
                if st2 == "sl st":
                    return "base"
                if st2 in {"skip", "-", "ch"}:
                    k += step
                    continue
                return "stitch"
            if t2 in {"inc", "dec", "cluster", "ch_space"}:
                return "stitch"
            if t2 == "repeat":
                nested = candidate.get("ops", [])
                if isinstance(nested, list) and _ops_to_produced_placements(nested):
                    return "stitch"
            k += step
        return "open"

    for i, op in enumerate(op_list):
        if not isinstance(op, dict):
            continue
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            n = max(0, int(op.get("n", 1)))
            if st == "ch":
                start_anchor = neighbor_anchor(i - 1, -1)
                end_anchor = neighbor_anchor(i + 1, 1)
                items.append(
                    {
                        "kind": "chain",
                        "count": max(1, n),
                        "start_anchor": "base" if start_anchor == "open" else start_anchor,
                        "end_anchor": end_anchor,
                    }
                )
            elif st in {"skip", "-", "sl st"}:
                continue
            else:
                for _ in range(n):
                    items.append({"kind": "stitch", "sym": st})
        elif t == "inc":
            base = str(op.get("base") or "x")
            for _ in range(max(0, int(op.get("n", 1))) * 2):
                items.append({"kind": "stitch", "sym": base})
        elif t == "dec":
            base = str(op.get("base") or "x")
            for _ in range(max(0, int(op.get("n", 1)))):
                items.append({"kind": "stitch", "sym": base})
        elif t in {"cluster", "ch_space"}:
            nested = op.get("ops", [])
            if isinstance(nested, list):
                items.extend(_ch_space_items_from_ops(nested))
        elif t == "repeat":
            times = max(0, int(op.get("times", 1)))
            nested = op.get("ops", [])
            if isinstance(nested, list):
                for _ in range(times):
                    items.extend(_ch_space_items_from_ops(nested))
    return items


def _available_chain_hole_centers(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    radii: list[float],
    *,
    r_cap: float | None = None,
    _cache: dict[int, list[tuple[float, float, float]]] | None = None,
) -> list[tuple[float, float, float]]:
    if round_index < 0 or round_index >= len(pattern.rounds):
        return []
    if _cache is None:
        _cache = {}
    if round_index in _cache:
        return _cache[round_index]

    prev_ops = pattern.rounds[round_index].stitch.ops
    consumed = pattern.rounds[round_index].stitch.consumed
    if not isinstance(consumed, int):
        consumed = _estimate_consumed(prev_ops)
    prev_total = _prev_total_with_override(pattern, round_index, consumed, opts.first_prev_total_override)
    if round_index == 0 and prev_total == 1 and consumed > 1:
        produced0 = pattern.rounds[round_index].stitch.total
        if not isinstance(produced0, int) or produced0 <= 0:
            produced0 = _estimate_total_ops(prev_ops)
        prev_total = int(produced0) if isinstance(produced0, int) and produced0 > 0 else int(consumed)

    holes = list(
        _circle_chain_hole_centers(
            pattern,
            opts,
            round_index,
            radii[round_index],
            radii[min(round_index + 1, len(radii) - 1)],
            prev_total,
            r_cap=r_cap,
        )
    )

    groups = _ch_space_groups_from_ops(pattern.rounds[round_index].stitch.ops)
    if groups and round_index > 0:
        base_holes = _available_chain_hole_centers(pattern, opts, round_index - 1, radii, r_cap=r_cap, _cache=_cache)
        holes.extend(_ch_space_chain_hole_centers(pattern, opts, round_index, radii, base_holes, r_cap=r_cap))

    _cache[round_index] = holes
    return holes


def _available_chain_hole_records(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    radii: list[float],
    *,
    r_cap: float | None = None,
    _cache: dict[int, list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    if round_index < 0 or round_index >= len(pattern.rounds):
        return []
    if _cache is None:
        _cache = {}
    if round_index in _cache:
        return _cache[round_index]

    prev_ops = pattern.rounds[round_index].stitch.ops
    consumed = pattern.rounds[round_index].stitch.consumed
    if not isinstance(consumed, int):
        consumed = _estimate_consumed(prev_ops)
    prev_total = _prev_total_with_override(pattern, round_index, consumed, opts.first_prev_total_override)
    if round_index == 0 and prev_total == 1 and consumed > 1:
        produced0 = pattern.rounds[round_index].stitch.total
        if not isinstance(produced0, int) or produced0 <= 0:
            produced0 = _estimate_total_ops(prev_ops)
        prev_total = int(produced0) if isinstance(produced0, int) and produced0 > 0 else int(consumed)

    records = list(
        _circle_chain_hole_records(
            pattern,
            opts,
            round_index,
            radii[round_index],
            radii[min(round_index + 1, len(radii) - 1)],
            prev_total,
            r_cap=r_cap,
        )
    )

    groups = _ch_space_groups_from_ops(pattern.rounds[round_index].stitch.ops)
    if groups and round_index > 0:
        base_records = _available_chain_hole_records(pattern, opts, round_index - 1, radii, r_cap=r_cap, _cache=_cache)
        records.extend(_ch_space_chain_hole_records(pattern, opts, round_index, radii, base_records, r_cap=r_cap))

    _cache[round_index] = records
    return records


def _ch_space_base_point(
    record: dict[str, object],
    slot_count: int,
    slot_index: float,
    *,
    cx: float,
    cy: float,
    fallback_angle: float,
    fallback_anchor_r: float,
) -> tuple[float, float]:
    point = _chain_path_point(record, _ch_space_slot_t(slot_count, slot_index))
    if point is not None:
        return point
    hx = float(record.get("x", cx))
    hy = float(record.get("y", cy))
    return hx + math.cos(fallback_angle) * fallback_anchor_r, hy + math.sin(fallback_angle) * fallback_anchor_r


def _ch_space_slot_base_segment(
    record: dict[str, object],
    slot_count: int,
    slot0: int,
    slot1: int,
    *,
    cx: float,
    cy: float,
    fallback_angle: float,
    fallback_anchor_r: float,
    trim: float = 0.16,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    start = float(slot0) + trim
    end = float(slot1) + 1.0 - trim
    if end <= start:
        mid = (float(slot0) + float(slot1) + 1.0) / 2.0
        start = mid - 0.22
        end = mid + 0.22
    p0 = _ch_space_base_point(
        record,
        slot_count,
        start,
        cx=cx,
        cy=cy,
        fallback_angle=fallback_angle,
        fallback_anchor_r=fallback_anchor_r,
    )
    p1 = _ch_space_base_point(
        record,
        slot_count,
        end,
        cx=cx,
        cy=cy,
        fallback_angle=fallback_angle,
        fallback_anchor_r=fallback_anchor_r,
    )
    return p0, p1, ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)


def _ch_space_stitch_quad(
    record: dict[str, object],
    slot_count: int,
    slot0: int,
    slot1: int,
    *,
    cx: float,
    cy: float,
    fallback_angle: float,
    fallback_anchor_r: float,
    band: float,
    height_frac: float,
    opts: RenderOptions,
) -> tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]:
    p0, p1, pm = _ch_space_slot_base_segment(
        record,
        slot_count,
        slot0,
        slot1,
        cx=cx,
        cy=cy,
        fallback_angle=fallback_angle,
        fallback_anchor_r=fallback_anchor_r,
        trim=0.13,
    )
    ux, uy = _outward_from_center(pm[0], pm[1], cx, cy, fallback_angle)
    base_inset = max(0.75, _chain_block_width(opts) * 0.12)
    # Ch-space stitches are worked on a small segment of an existing ch chain,
    # not across the whole round band. Keep the local body tied to yarn width,
    # otherwise long stitches in chsp collapse into oversized radial shards.
    yarn_w = _chain_block_width(opts)
    local_body = yarn_w * (1.05 + (1.85 * float(height_frac)))
    band_body = max(float(band) * 0.18, float(band) * (0.20 + (0.38 * float(height_frac))))
    body = max(yarn_w * 1.25, min(band_body, local_body))
    ix0 = p0[0] + ux * base_inset
    iy0 = p0[1] + uy * base_inset
    ix1 = p1[0] + ux * base_inset
    iy1 = p1[1] + uy * base_inset
    ox0 = p0[0] + ux * body
    oy0 = p0[1] + uy * body
    ox1 = p1[0] + ux * body
    oy1 = p1[1] + uy * body
    d = f"M {ix0:.3f} {iy0:.3f} L {ix1:.3f} {iy1:.3f} L {ox1:.3f} {oy1:.3f} L {ox0:.3f} {oy0:.3f} Z"
    base_mid = ((ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0)
    top_mid = ((ox0 + ox1) / 2.0, (oy0 + oy1) / 2.0)
    return d, base_mid, top_mid, (ux, uy)


def _chain_record_from_points(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    control_angle: float,
    desired_len: float,
    count: int,
) -> dict[str, object]:
    x0, y0 = start_xy
    x1, y1 = end_xy
    control = _chain_quad_control(x0, y0, x1, y1, control_angle, desired_len)
    if control is None:
        path: dict[str, object] = {"kind": "line", "p0": (x0, y0), "p1": (x1, y1)}
        hole_x = (x0 + x1) / 2.0
        hole_y = (y0 + y1) / 2.0
    else:
        xc, yc = control
        path = {"kind": "quad", "p0": (x0, y0), "pc": (xc, yc), "p1": (x1, y1)}
        hole_x = (0.25 * x0) + (0.5 * xc) + (0.25 * x1)
        hole_y = (0.25 * y0) + (0.5 * yc) + (0.25 * y1)
    return {"x": hole_x, "y": hole_y, "r": max(2.0, desired_len * 0.18), "path": path, "count": max(1, int(count))}


def _ch_space_chain_hole_records(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    radii: list[float],
    base_records: list[dict[str, object]],
    *,
    r_cap: float | None = None,
) -> list[dict[str, object]]:
    groups = _ch_space_groups_from_ops(pattern.rounds[round_index].stitch.ops)
    if not groups or not base_records:
        return []

    cx = cy = opts.size / 2
    band = max(radii[min(round_index + 1, len(radii) - 1)] - radii[round_index], 1.0)
    chain_unit = _chain_unit_from_band(_visible_round_band(radii, round_index, fallback=opts.size * 0.055))
    geom: list[dict[str, object]] = []

    for idx, group in enumerate(groups):
        if idx >= len(base_records):
            break
        record = base_records[idx]
        items = _ch_space_items_from_ops(group)
        if not items:
            continue
        hx = float(record.get("x", cx))
        hy = float(record.get("y", cy))
        hole_r = float(record.get("r", max(2.0, band * 0.35)))
        outward = math.atan2(hy - cy, hx - cx)
        outer_max = max(hole_r + band * 0.45, band * 0.58)
        fallback_anchor_r = max(1.5, min(hole_r * 1.02, outer_max * 0.60, band * 0.48) - _chain_anchor_inset(opts))
        slot_spans, slot_count = _ch_space_item_slot_spans(items)
        stitch_tops: dict[int, tuple[float, float]] = {}
        chain_jobs: list[int] = []
        for j, item in enumerate(items):
            slot0, slot1 = slot_spans[j]
            if item.get("kind") == "chain":
                chain_jobs.append(j)
                continue
            sym = str(item.get("sym", "x"))
            base_sym = sym.split(":", 1)[1] if sym.startswith(("inc:", "dec:")) else sym
            height_frac = max(0.18, min(1.0, float(_metric_height(base_sym)) / 6.0))
            if opts.glyph_full_height:
                height_frac = 1.0
            _d, _base_mid, top_mid, _dir = _ch_space_stitch_quad(
                record,
                slot_count,
                slot0,
                slot1,
                cx=cx,
                cy=cy,
                fallback_angle=outward,
                fallback_anchor_r=fallback_anchor_r,
                band=band,
                height_frac=height_frac,
                opts=opts,
            )
            stitch_tops[j] = top_mid
        geom.append(
            {
                "record": record,
                "items": items,
                "outward": outward,
                "fallback_anchor_r": fallback_anchor_r,
                "slot_spans": slot_spans,
                "slot_count": slot_count,
                "stitch_tops": stitch_tops,
                "chain_jobs": chain_jobs,
            }
        )

    def group_stitch_top(group_index: int, *, last: bool) -> tuple[float, float] | None:
        if not geom:
            return None
        group_index %= len(geom)
        tops = geom[group_index].get("stitch_tops")
        if not isinstance(tops, dict) or not tops:
            return None
        key = max(tops) if last else min(tops)
        return tops.get(key)

    records: list[dict[str, object]] = []
    for idx, info in enumerate(geom):
        items = info["items"]
        slot_spans = info["slot_spans"]
        stitch_tops = info["stitch_tops"]
        chain_jobs = info["chain_jobs"]
        if not isinstance(items, list) or not isinstance(slot_spans, list) or not isinstance(stitch_tops, dict) or not isinstance(chain_jobs, list):
            continue
        record = info["record"]
        if not isinstance(record, dict):
            continue
        slot_count = int(info["slot_count"])
        outward = float(info["outward"])
        fallback_anchor_r = float(info["fallback_anchor_r"])

        def adjacent_stitch_top(index: int, step: int) -> tuple[float, float] | None:
            k = index + step
            while 0 <= k < len(items):
                if k in stitch_tops:
                    return stitch_tops[k]
                if items[k].get("kind") == "chain":
                    return None
                k += step
            return None

        for j in chain_jobs:
            if j >= len(slot_spans):
                continue
            item = items[j]
            if not isinstance(item, dict):
                continue
            slot0, slot1 = slot_spans[j]
            _p0, _p1, base_mid = _ch_space_slot_base_segment(
                record,
                slot_count,
                int(slot0),
                int(slot1),
                cx=cx,
                cy=cy,
                fallback_angle=outward,
                fallback_anchor_r=fallback_anchor_r,
                trim=0.20,
            )
            ux, uy = _outward_from_center(base_mid[0], base_mid[1], cx, cy, outward)
            segment_out = max(1.0, _chain_block_width(opts) * 0.30)
            start_xy = (base_mid[0] + ux * segment_out, base_mid[1] + uy * segment_out)
            end_xy = start_xy
            start_anchor = item.get("start_anchor")
            end_anchor = item.get("end_anchor")
            if start_anchor == "stitch":
                start_xy = adjacent_stitch_top(j, -1) or start_xy
            elif start_anchor == "open":
                start_xy = group_stitch_top(idx - 1, last=True) or start_xy
            if end_anchor == "stitch":
                end_xy = adjacent_stitch_top(j, 1) or end_xy
            elif end_anchor == "open":
                end_xy = group_stitch_top(idx + 1, last=False) or end_xy
            if start_anchor == "base" and end_anchor == "base":
                start_seg0, start_seg1, start_mid = _ch_space_slot_base_segment(
                    record,
                    slot_count,
                    int(slot0),
                    int(slot0),
                    cx=cx,
                    cy=cy,
                    fallback_angle=outward,
                    fallback_anchor_r=fallback_anchor_r,
                    trim=0.08,
                )
                end_seg0, end_seg1, end_mid = _ch_space_slot_base_segment(
                    record,
                    slot_count,
                    int(slot1),
                    int(slot1),
                    cx=cx,
                    cy=cy,
                    fallback_angle=outward,
                    fallback_anchor_r=fallback_anchor_r,
                    trim=0.08,
                )
                ux0, uy0 = _outward_from_center(start_mid[0], start_mid[1], cx, cy, outward)
                ux1, uy1 = _outward_from_center(end_mid[0], end_mid[1], cx, cy, outward)
                start_xy = (start_mid[0] + ux0 * segment_out, start_mid[1] + uy0 * segment_out)
                end_xy = (end_mid[0] + ux1 * segment_out, end_mid[1] + uy1 * segment_out)

            if start_anchor == "base" and end_anchor == "stitch":
                continue
            count = max(1, int(item.get("count", 1)))
            desired_len = _ch_space_chain_lift(count, chain_unit)
            control_angle = math.atan2(((start_xy[1] + end_xy[1]) / 2.0) - cy, ((start_xy[0] + end_xy[0]) / 2.0) - cx)
            rec = _chain_record_from_points(start_xy, end_xy, control_angle, desired_len, count)
            chord = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
            rec["r"] = max(2.0, min(band * 0.45, max(_chain_block_width(opts), chord * 0.35, desired_len * 0.18)))
            records.append(rec)
    return records


def _ch_space_chain_hole_centers(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    radii: list[float],
    base_holes: list[tuple[float, float, float]],
    *,
    r_cap: float | None = None,
) -> list[tuple[float, float, float]]:
    records = [
        {"x": float(x), "y": float(y), "r": float(r)}
        for x, y, r in base_holes
    ]
    return [
        (float(record["x"]), float(record["y"]), float(record["r"]))
        for record in _ch_space_chain_hole_records(pattern, opts, round_index, radii, records, r_cap=r_cap)
    ]


def _render_circle_ch_spaces(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    radii: list[float],
    *,
    r_cap: float | None = None,
) -> str:
    if round_index <= 0:
        return ""
    groups = _ch_space_groups_from_ops(pattern.rounds[round_index].stitch.ops)
    if not groups:
        return ""

    prev_idx = round_index - 1
    prev_ops = pattern.rounds[prev_idx].stitch.ops
    prev_consumed = pattern.rounds[prev_idx].stitch.consumed
    if not isinstance(prev_consumed, int):
        prev_consumed = _estimate_consumed(prev_ops)
    if " in ring" in pattern.rounds[prev_idx].stitch.stitch.lower() or " into ring" in pattern.rounds[prev_idx].stitch.stitch.lower():
        prev_consumed = 1
    prev_total = _prev_total_with_override(pattern, prev_idx, prev_consumed, opts.first_prev_total_override)
    if prev_idx == 0 and prev_total == 1 and prev_consumed > 1:
        produced0 = pattern.rounds[prev_idx].stitch.total
        if not isinstance(produced0, int) or produced0 <= 0:
            produced0 = _estimate_total_ops(prev_ops)
        prev_total = int(produced0) if isinstance(produced0, int) and produced0 > 0 else int(prev_consumed)

    holes = _available_chain_hole_centers(pattern, opts, prev_idx, radii, r_cap=r_cap)
    if not holes:
        return ""

    color = pattern.rounds[round_index].color
    stroke = opts.stroke
    glyph_stroke = opts.glyph_stroke
    sw = opts.glyph_width
    op = opts.glyph_opacity
    chsp_stroke_width = max(0.75, float(opts.stroke_width) * 0.45)
    chsp_glyph_width = max(0.45, float(sw) * 0.55)
    chsp_glyph_opacity = min(float(op), 0.42)
    parts: list[str] = []
    band = max(radii[min(round_index + 1, len(radii) - 1)] - radii[round_index], 1.0)
    chain_unit = _chain_unit_from_band(_visible_round_band(radii, round_index, fallback=opts.size * 0.055))
    cx = cy = opts.size / 2

    def draw_chain_in_space(hx: float, hy: float, outward: float, count: int, anchor_r: float, *, loop: bool) -> None:
        chain_w = _chain_block_width(opts)
        outline_w = chain_w + (2.0 * max(1.0, float(opts.stroke_width)))
        desired_len = _ch_space_chain_lift(count, chain_unit)
        particle_attrs = _chain_particle_attrs(opts, count)
        if loop:
            spread = min(math.pi * 0.45, max(math.pi * 0.18, count * math.pi * 0.08))
            a0 = outward - (spread / 2.0)
            a1 = outward + (spread / 2.0)
            x0 = hx + math.cos(a0) * anchor_r
            y0 = hy + math.sin(a0) * anchor_r
            x1 = hx + math.cos(a1) * anchor_r
            y1 = hy + math.sin(a1) * anchor_r
            control = _chain_quad_control(x0, y0, x1, y1, outward, desired_len)
            if control is None:
                parts.append(
                    f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
                )
                parts.append(
                    f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{color}" stroke-width="{chain_w:.3f}" stroke-linecap="round"/>\n'
                )
            else:
                xc, yc = control
                d = f"M {x0:.3f} {y0:.3f} Q {xc:.3f} {yc:.3f} {x1:.3f} {y1:.3f}"
                parts.append(
                    f'<path d="{d}" fill="none"{particle_attrs} stroke="{stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
                )
                parts.append(
                    f'<path d="{d}" fill="none"{particle_attrs} stroke="{color}" stroke-width="{chain_w:.3f}" stroke-linecap="round"/>\n'
                )
        else:
            x0 = hx + math.cos(outward) * anchor_r
            y0 = hy + math.sin(outward) * anchor_r
            x1 = hx + math.cos(outward) * (anchor_r + desired_len)
            y1 = hy + math.sin(outward) * (anchor_r + desired_len)
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{color}" stroke-width="{chain_w:.3f}" stroke-linecap="round"/>\n'
            )

    def chain_returns_to_space(op_list: list[object], start_index: int) -> bool:
        for nxt in op_list[start_index:]:
            if not isinstance(nxt, dict):
                continue
            t = nxt.get("type")
            if t == "stitch":
                st = str(nxt.get("st", ""))
                if st == "sl st":
                    return True
                if st in {"skip", "-", "ch"}:
                    continue
                return True
            if t in {"inc", "dec", "cluster", "ch_space"}:
                return True
            if t == "repeat":
                nested = nxt.get("ops", [])
                if isinstance(nested, list) and chain_returns_to_space(nested, 0):
                    return True
        return False

    def draw_inner_chains(inner_ops: list[object], hx: float, hy: float, outward: float, anchor_r: float) -> None:
        for i, op in enumerate(inner_ops):
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch" and str(op.get("st", "")) == "ch":
                draw_chain_in_space(
                    hx,
                    hy,
                    outward,
                    max(1, int(op.get("n", 1))),
                    anchor_r,
                    loop=chain_returns_to_space(inner_ops, i + 1),
                )
            elif t == "repeat":
                times = max(0, int(op.get("times", 1)))
                nested = op.get("ops", [])
                if isinstance(nested, list):
                    for _ in range(times):
                        draw_inner_chains(nested, hx, hy, outward, anchor_r)
            elif t in {"cluster", "ch_space"}:
                nested = op.get("ops", [])
                if isinstance(nested, list):
                    draw_inner_chains(nested, hx, hy, outward, anchor_r)

    def render_items_from_ops(op_list: list[object]) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []

        def neighbor_anchor(start_index: int, step: int) -> str:
            k = start_index
            while 0 <= k < len(op_list):
                candidate = op_list[k]
                if not isinstance(candidate, dict):
                    k += step
                    continue
                t2 = candidate.get("type")
                if t2 == "stitch":
                    st2 = str(candidate.get("st", ""))
                    if st2 == "sl st":
                        return "base"
                    if st2 in {"skip", "-", "ch"}:
                        k += step
                        continue
                    return "stitch"
                if t2 in {"inc", "dec", "cluster", "ch_space"}:
                    return "stitch"
                if t2 == "repeat":
                    nested = candidate.get("ops", [])
                    if isinstance(nested, list) and _ops_to_produced_placements(nested):
                        return "stitch"
                k += step
            return "open"

        for i, op in enumerate(op_list):
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = max(0, int(op.get("n", 1)))
                if st == "ch":
                    start_anchor = neighbor_anchor(i - 1, -1)
                    end_anchor = neighbor_anchor(i + 1, 1)
                    items.append(
                        {
                            "kind": "chain",
                            "count": max(1, n),
                            "start_anchor": "base" if start_anchor == "open" else start_anchor,
                            "end_anchor": end_anchor,
                        }
                    )
                elif st in {"skip", "-", "sl st"}:
                    continue
                else:
                    for _ in range(n):
                        items.append({"kind": "stitch", "sym": st})
            elif t == "inc":
                base = str(op.get("base") or "x")
                for _ in range(max(0, int(op.get("n", 1))) * 2):
                    items.append({"kind": "stitch", "sym": base})
            elif t == "dec":
                base = str(op.get("base") or "x")
                for _ in range(max(0, int(op.get("n", 1)))):
                    items.append({"kind": "stitch", "sym": base})
            elif t in {"cluster", "ch_space"}:
                nested = op.get("ops", [])
                if isinstance(nested, list):
                    items.extend(render_items_from_ops(nested))
            elif t == "repeat":
                times = max(0, int(op.get("times", 1)))
                nested = op.get("ops", [])
                if isinstance(nested, list):
                    for _ in range(times):
                        items.extend(render_items_from_ops(nested))
        return items

    def append_chain_item(
        chain_parts: list[str],
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        control_angle: float,
        count: int,
    ) -> None:
        chain_w = _chain_block_width(opts)
        outline_w = chain_w + (2.0 * max(1.0, float(opts.stroke_width)))
        desired_len = _ch_space_chain_lift(count, chain_unit)
        particle_attrs = _chain_particle_attrs(opts, count)
        x0, y0 = start_xy
        x1, y1 = end_xy
        control = _chain_quad_control(x0, y0, x1, y1, control_angle, desired_len)
        if control is None:
            chain_parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            chain_parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"{particle_attrs} stroke="{color}" stroke-width="{chain_w:.3f}" stroke-linecap="round"/>\n'
            )
        else:
            xc, yc = control
            d = f"M {x0:.3f} {y0:.3f} Q {xc:.3f} {yc:.3f} {x1:.3f} {y1:.3f}"
            chain_parts.append(
                f'<path d="{d}" fill="none"{particle_attrs} stroke="{stroke}" stroke-width="{outline_w:.3f}" stroke-linecap="round"/>\n'
            )
            chain_parts.append(
                f'<path d="{d}" fill="none"{particle_attrs} stroke="{color}" stroke-width="{chain_w:.3f}" stroke-linecap="round"/>\n'
            )

    def first_stitch_top_for_group(group_index: int) -> tuple[float, float] | None:
        if group_index < 0 or group_index >= len(groups) or group_index >= len(holes):
            return None
        next_items = render_items_from_ops(groups[group_index])
        if not next_items:
            return None
        hx2, hy2, hole_r2 = holes[group_index]
        outward2 = math.atan2(hy2 - cy, hx2 - cx)
        outer_max2 = max(hole_r2 + band * 0.45, band * 0.58)
        slot_spans2, slot_count2 = _ch_space_item_slot_spans(next_items)
        if slot_count2 <= 0:
            return None
        start_ang2, sector_sweep2, gap2 = _ch_space_layout(outward2, slot_count2)
        for j2, item2 in enumerate(next_items):
            if item2.get("kind") != "stitch":
                continue
            sym2 = str(item2.get("sym", "x"))
            base_sym2 = sym2.split(":", 1)[1] if sym2.startswith(("inc:", "dec:")) else sym2
            height_frac2 = max(0.18, min(1.0, float(_metric_height(base_sym2)) / 6.0))
            if opts.glyph_full_height:
                height_frac2 = 1.0
            _, outer2 = _ch_space_stitch_radii(hole_r2, band, height_frac2)
            slot0, slot1 = slot_spans2[j2]
            _, _, ang2 = _ch_space_slot_span_angles(start_ang2, sector_sweep2, gap2, slot0, slot1)
            return hx2 + outer2 * math.cos(ang2), hy2 + outer2 * math.sin(ang2)
        return None

    def last_stitch_top_for_group(group_index: int) -> tuple[float, float] | None:
        if group_index < 0 or group_index >= len(groups) or group_index >= len(holes):
            return None
        next_items = render_items_from_ops(groups[group_index])
        if not next_items:
            return None
        hx2, hy2, hole_r2 = holes[group_index]
        outward2 = math.atan2(hy2 - cy, hx2 - cx)
        outer_max2 = max(hole_r2 + band * 0.45, band * 0.58)
        slot_spans2, slot_count2 = _ch_space_item_slot_spans(next_items)
        if slot_count2 <= 0:
            return None
        start_ang2, sector_sweep2, gap2 = _ch_space_layout(outward2, slot_count2)
        for j2 in range(len(next_items) - 1, -1, -1):
            item2 = next_items[j2]
            if item2.get("kind") != "stitch":
                continue
            sym2 = str(item2.get("sym", "x"))
            base_sym2 = sym2.split(":", 1)[1] if sym2.startswith(("inc:", "dec:")) else sym2
            height_frac2 = max(0.18, min(1.0, float(_metric_height(base_sym2)) / 6.0))
            if opts.glyph_full_height:
                height_frac2 = 1.0
            _, outer2 = _ch_space_stitch_radii(hole_r2, band, height_frac2)
            slot0, slot1 = slot_spans2[j2]
            _, _, ang2 = _ch_space_slot_span_angles(start_ang2, sector_sweep2, gap2, slot0, slot1)
            return hx2 + outer2 * math.cos(ang2), hy2 + outer2 * math.sin(ang2)
        return None

    for idx, inner_ops in enumerate(groups):
        if idx >= len(holes):
            break
        items = render_items_from_ops(inner_ops)
        hx, hy, hole_r = holes[idx]
        outward = math.atan2(hy - cy, hx - cx)
        outer_max = max(hole_r + band * 0.45, band * 0.58)
        chain_anchor_r = max(1.5, min(hole_r * 1.02, outer_max * 0.60, band * 0.48) - _chain_anchor_inset(opts))
        slot_spans, slot_count = _ch_space_item_slot_spans(items)
        start_ang, sector_sweep, gap = _ch_space_layout(outward, slot_count)
        chain_parts: list[str] = []
        stitch_parts: list[str] = []
        stitch_tops: dict[int, tuple[float, float]] = {}
        chain_jobs: list[tuple[int, dict[str, object], float]] = []
        for j, item in enumerate(items):
            slot0, slot1 = slot_spans[j]
            sector_start, sector_end, ang = _ch_space_slot_span_angles(start_ang, sector_sweep, gap, slot0, slot1)
            sweep = max(0.001, sector_end - sector_start)
            if item.get("kind") == "chain":
                chain_jobs.append((j, item, ang))
                continue
            sym = str(item.get("sym", "x"))
            glyph = _glyph_for_symbol(sym)
            if glyph.get("kind") != "skip":
                base_sym = sym.split(":", 1)[1] if sym.startswith(("inc:", "dec:")) else sym
                height_frac = max(0.18, min(1.0, float(_metric_height(base_sym)) / 6.0))
                if opts.glyph_full_height:
                    height_frac = 1.0
                inner, outer = _ch_space_stitch_radii(hole_r, band, height_frac)
                d = _ring_sector_path(hx, hy, inner, outer, sector_start, sweep)
                if d:
                    stitch_parts.append(
                        f'<path d="{d}" fill="{color}" stroke="{stroke}" stroke-width="{chsp_stroke_width:.3f}"/>\n'
                    )
                ca = math.cos(ang)
                sa = math.sin(ang)
                x0 = hx + inner * ca
                y0 = hy + inner * sa
                x1 = hx + outer * ca
                y1 = hy + outer * sa
                stitch_tops[j] = (x1, y1)
                stitch_parts.append(
                    f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{glyph_stroke}" stroke-width="{chsp_glyph_width:.3f}" stroke-linecap="round" opacity="{chsp_glyph_opacity:.3f}"/>\n'
                )

        def base_xy(angle: float) -> tuple[float, float]:
            return hx + math.cos(angle) * chain_anchor_r, hy + math.sin(angle) * chain_anchor_r

        def adjacent_stitch_top(index: int, step: int) -> tuple[float, float] | None:
            k = index + step
            while 0 <= k < len(items):
                if k in stitch_tops:
                    return stitch_tops[k]
                if items[k].get("kind") == "chain":
                    return None
                k += step
            return None

        for j, item, ang in chain_jobs:
            start_xy = base_xy(ang)
            end_xy = base_xy(ang)
            if item.get("start_anchor") == "stitch":
                start_xy = adjacent_stitch_top(j, -1) or start_xy
            elif item.get("start_anchor") == "open":
                prev_idx = (idx - 1) % len(groups) if groups else idx
                start_xy = last_stitch_top_for_group(prev_idx) or start_xy
            if item.get("end_anchor") == "stitch":
                end_xy = adjacent_stitch_top(j, 1) or end_xy
            elif item.get("end_anchor") == "open":
                next_idx = (idx + 1) % len(groups) if groups else idx
                end_xy = first_stitch_top_for_group(next_idx) or end_xy
            if item.get("start_anchor") == "base" and item.get("end_anchor") == "base":
                slot0, slot1 = slot_spans[j]
                slot_start, slot_end, slot_mid = _ch_space_slot_boundary_angles(
                    start_ang,
                    max(sector_sweep, math.radians(8.0)),
                    slot0,
                    slot1,
                )
                start_xy = base_xy(slot_start)
                end_xy = base_xy(slot_end)
                ang = slot_mid
            append_chain_item(chain_parts, start_xy, end_xy, ang, int(item.get("count", 1)))
        parts.extend(chain_parts)
        parts.extend(stitch_parts)
    return "".join(parts)


def _render_circle_texture(
    pattern: PatternSpec,
    opts: RenderOptions,
    round_index: int,
    r_inner: float,
    r_outer: float,
    prev_total: int,
    *,
    r_cap: float | None = None,
) -> str:
    ops = pattern.rounds[round_index].stitch.ops
    placements, consumed, produced = _ops_to_placements(ops)
    produced_syms = _ops_to_produced_placements(ops) if opts.glyph_angle_mode == "by_produced" else []
    if not placements and not produced_syms:
        return ""

    size = opts.size
    cx = cy = size / 2
    band = max(r_outer - r_inner, 1.0)
    # "Top-align" texture to the colored band: draw full-length lines and clip
    # to the band area so glyphs visually fill the ring.
    margin = 0.0
    r0 = r_inner + margin

    stroke = opts.glyph_stroke
    sw = opts.glyph_width
    op = opts.glyph_opacity

    parts: list[str] = []
    if produced_syms:
        # One glyph per produced stitch, evenly spread.
        glyph_syms = produced_syms
        npl = len(glyph_syms)
        # For performance, cap rendering; in this mode markers don't matter.
        if npl > opts.max_glyphs_per_round:
            step = npl / opts.max_glyphs_per_round
            glyph_syms = [glyph_syms[int(i * step)] for i in range(opts.max_glyphs_per_round)]
            npl = len(glyph_syms)
        placements = [(0.0, s) for s in glyph_syms]  # pos overwritten below
    else:
        placements = _sample_keep_markers(placements, opts.max_glyphs_per_round)
        npl = len(placements)

    usable = max(band - 2 * margin, 1.0)
    denom = max(prev_total, 1)
    start = (-math.pi / 2) + (math.pi * float(opts.rotation_deg) / 180.0)
    seg_mod = _mod_from_opts(opts, round_index, prev_total)
    field_mod = _bump_field_from_opts(pattern, opts, round_index, prev_total)
    has_bump = (seg_mod is not None) or (field_mod is not None)
    bump_scale = (
        min((r_outer - r_inner) * 1.2, max(6.0, (r_outer - r_inner) * 0.9)) if has_bump else 0.0
    )
    # Height-trace mode: draw a thin polyline around the ring whose radius is
    # proportional to the stitch metric height, instead of drawing per-stitch
    # radial strokes.
    if opts.glyph_style == "metric_strokes":
        # Each stitch still draws a radial stroke, but the stroke length is
        # mapped from the stitch metric height (DEFAULT_METRICS[*].height).
        max_h = 6.0  # dtr
        gamma = 1.6  # emphasize differences: short smaller, long closer to top
        for j, (pos, sym) in enumerate(placements):
            if (opts.glyph_angle_mode in ("uniform", "by_produced")) and npl > 0:
                pos = (j + 0.5) * (prev_total / npl)
            base_sym = sym
            if sym.startswith(("inc:", "dec:")):
                base_sym = sym.split(":", 1)[1]
            frac0 = max(0.0, min(1.0, float(_metric_height(base_sym)) / max_h))
            frac = frac0**gamma
            bump = _combined_bump_at(pos, prev_total, seg_mod, field_mod) if has_bump else 0.0
            r_outer_pos = r_outer + bump * bump_scale if has_bump else r_outer
            if isinstance(r_cap, (int, float)):
                r_outer_pos = min(float(r_cap), r_outer_pos)
            r1 = min(r_outer_pos - margin, r0 + frac * usable)
            ang = start + (2 * math.pi * (pos / denom))
            ca = math.cos(ang)
            sa = math.sin(ang)
            x0 = cx + (r0 * ca)
            y0 = cy + (r0 * sa)
            x1 = cx + (r1 * ca)
            y1 = cy + (r1 * sa)
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
            )
        return "".join(parts)

    for j, (pos, sym) in enumerate(placements):
        glyph = _glyph_for_symbol(sym)
        length = 1.0 if opts.glyph_full_height else float(glyph.get("length", 0.35))
        if (opts.glyph_angle_mode in ("uniform", "by_produced")) and npl > 0:
            # Keep bump evaluation in base-stitch units, but re-spread angles.
            pos = (j + 0.5) * (prev_total / npl)
        bump = _combined_bump_at(pos, prev_total, seg_mod, field_mod) if has_bump else 0.0
        r_outer_pos = r_outer + bump * bump_scale if has_bump else r_outer
        if isinstance(r_cap, (int, float)):
            r_outer_pos = min(float(r_cap), r_outer_pos)
        r1 = min(r_outer_pos - margin, r0 + length * usable)

        ang = start + (2 * math.pi * (pos / denom))
        ca = math.cos(ang)
        sa = math.sin(ang)
        x0 = cx + (r0 * ca)
        y0 = cy + (r0 * sa)
        x1 = cx + (r1 * ca)
        y1 = cy + (r1 * sa)

        kind = glyph.get("kind", "tick")
        if kind == "skip":
            continue
        if kind in ("inc", "dec"):
            delta = 0.10  # radians, small spread
            if kind == "inc":
                # two diverging strokes (V-like)
                for sgn in (-1.0, 1.0):
                    ang2 = ang + (sgn * delta)
                    ca2 = math.cos(ang2)
                    sa2 = math.sin(ang2)
                    x0b = cx + (r0 * ca2)
                    y0b = cy + (r0 * sa2)
                    x1b = cx + (r1 * ca2)
                    y1b = cy + (r1 * sa2)
                    parts.append(
                        f'<line x1="{x0b:.3f}" y1="{y0b:.3f}" x2="{x1b:.3f}" y2="{y1b:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
            else:
                # two converging strokes (Λ-like)
                for sgn in (-1.0, 1.0):
                    ang2 = ang + (sgn * delta)
                    ca2 = math.cos(ang2)
                    sa2 = math.sin(ang2)
                    x0b = cx + (r0 * ca2)
                    y0b = cy + (r0 * sa2)
                    x1b = cx + (r1 * ca)
                    y1b = cy + (r1 * sa)
                    parts.append(
                        f'<line x1="{x0b:.3f}" y1="{y0b:.3f}" x2="{x1b:.3f}" y2="{y1b:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
        elif kind in ("tick", "post", "chain", "slip"):
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
            )
            # Posts get crossbars to differentiate dc/tr.
            bars = int(glyph.get("bars", 0) or 0)
            if bars > 0:
                # Place bars along the upper half of the post.
                for b in range(bars):
                    t = 0.55 + (0.15 * b)
                    xb = x0 + (x1 - x0) * t
                    yb = y0 + (y1 - y0) * t
                    # tangential direction
                    tx = -sa
                    ty = ca
                    bar_len = max(2.0, band * 0.18)
                    xa = xb - (tx * bar_len / 2)
                    ya = yb - (ty * bar_len / 2)
                    xc = xb + (tx * bar_len / 2)
                    yc = yb + (ty * bar_len / 2)
                    parts.append(
                        f'<line x1="{xa:.3f}" y1="{ya:.3f}" x2="{xc:.3f}" y2="{yc:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
        else:
            # Fallback to a simple tick.
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
            )
    return "".join(parts)


def _render_square_texture(
    pattern: PatternSpec, opts: RenderOptions, round_index: int, half_inner: float, half_outer: float, prev_total: int
) -> str:
    ops = pattern.rounds[round_index].stitch.ops
    placements, _consumed, _produced = _ops_to_placements(ops)
    if not placements:
        return ""

    size = opts.size
    cx = cy = size / 2
    band = max(half_outer - half_inner, 1.0)
    # Top-align texture to the colored band; clip keeps it within the band.
    margin = 0.0
    # draw on the middle of the band
    half_mid = (half_inner + half_outer) / 2
    d0 = -band / 2 + margin

    stroke = opts.glyph_stroke
    sw = opts.glyph_width
    op = opts.glyph_opacity

    perim = 8 * half_mid  # 4 sides of length 2*half_mid
    produced_syms = _ops_to_produced_placements(ops) if opts.glyph_angle_mode == "by_produced" else []
    if produced_syms:
        glyph_syms = produced_syms
        npl = len(glyph_syms)
        if npl > opts.max_glyphs_per_round:
            step = npl / opts.max_glyphs_per_round
            glyph_syms = [glyph_syms[int(i * step)] for i in range(opts.max_glyphs_per_round)]
            npl = len(glyph_syms)
        placements = [(0.0, s) for s in glyph_syms]  # pos overwritten below
    else:
        placements = _sample_keep_markers(placements, opts.max_glyphs_per_round)
        npl = len(placements)
    denom = max(prev_total, 1)
    step = perim / denom if denom else perim

    def point_at(s: float) -> tuple[float, float, float, float]:
        # returns (x, y, nx, ny) where n is outward normal
        side = 2 * half_mid
        s = s % perim
        if s < side:
            # top: left->right, normal up
            x = cx - half_mid + s
            y = cy - half_mid
            return x, y, 0.0, -1.0
        s -= side
        if s < side:
            # right: top->bottom, normal right
            x = cx + half_mid
            y = cy - half_mid + s
            return x, y, 1.0, 0.0
        s -= side
        if s < side:
            # bottom: right->left, normal down
            x = cx + half_mid - s
            y = cy + half_mid
            return x, y, 0.0, 1.0
        s -= side
        # left: bottom->top, normal left
        x = cx - half_mid
        y = cy + half_mid - s
        return x, y, -1.0, 0.0

    parts: list[str] = []
    usable = max(band - 2 * margin, 1.0)
    npl = len(placements)
    if opts.glyph_style == "metric_strokes":
        max_h = 6.0
        gamma = 1.6
        for j, (pos, sym) in enumerate(placements):
            if (opts.glyph_angle_mode in ("uniform", "by_produced")) and npl > 0:
                pos = (j + 0.5) * (prev_total / npl)
            base_sym = sym
            if sym.startswith(("inc:", "dec:")):
                base_sym = sym.split(":", 1)[1]
            frac0 = max(0.0, min(1.0, float(_metric_height(base_sym)) / max_h))
            frac = frac0**gamma
            d1 = min(band / 2 - margin, d0 + frac * usable)
            x, y, nx, ny = point_at(pos * step)
            x0 = x + nx * d0
            y0 = y + ny * d0
            x1 = x + nx * d1
            y1 = y + ny * d1
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
            )
        return "".join(parts)

    for j, (pos, sym) in enumerate(placements):
        glyph = _glyph_for_symbol(sym)
        length = 1.0 if opts.glyph_full_height else float(glyph.get("length", 0.35))
        if (opts.glyph_angle_mode in ("uniform", "by_produced")) and npl > 0:
            pos = (j + 0.5) * (prev_total / npl)
        d1 = min(band / 2 - margin, d0 + length * usable)

        x, y, nx, ny = point_at(pos * step)
        x0 = x + nx * d0
        y0 = y + ny * d0
        x1 = x + nx * d1
        y1 = y + ny * d1
        if glyph.get("kind") == "skip":
            continue
        kind = glyph.get("kind", "tick")
        if kind in ("inc", "dec"):
            # Use a small tangential split to make a V/Λ shape
            tx, ty = -ny, nx
            split = max(1.5, band * 0.08)
            if kind == "inc":
                # diverging
                for sgn in (-1.0, 1.0):
                    x0b = x0 + (tx * split * sgn)
                    y0b = y0 + (ty * split * sgn)
                    x1b = x1 + (tx * split * sgn)
                    y1b = y1 + (ty * split * sgn)
                    parts.append(
                        f'<line x1="{x0b:.3f}" y1="{y0b:.3f}" x2="{x1b:.3f}" y2="{y1b:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
            else:
                # converging to the tip
                for sgn in (-1.0, 1.0):
                    x0b = x0 + (tx * split * sgn)
                    y0b = y0 + (ty * split * sgn)
                    parts.append(
                        f'<line x1="{x0b:.3f}" y1="{y0b:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
        else:
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
            )
            bars = int(glyph.get("bars", 0) or 0)
            if bars > 0:
                # tangential direction is perpendicular to normal
                tx, ty = -ny, nx
                for b in range(bars):
                    t = 0.55 + (0.15 * b)
                    xb = x0 + (x1 - x0) * t
                    yb = y0 + (y1 - y0) * t
                    bar_len = max(2.0, band * 0.18)
                    xa = xb - (tx * bar_len / 2)
                    ya = yb - (ty * bar_len / 2)
                    xc = xb + (tx * bar_len / 2)
                    yc = yb + (ty * bar_len / 2)
                    parts.append(
                        f'<line x1="{xa:.3f}" y1="{ya:.3f}" x2="{xc:.3f}" y2="{yc:.3f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>\n'
                    )
    return "".join(parts)


def _circle_path(cx: float, cy: float, r: float) -> str:
    # Full circle as two arcs (SVG arc can't draw full circle in one command).
    return (
        f"M {cx:.3f} {cy - r:.3f} "
        f"A {r:.3f} {r:.3f} 0 1 1 {cx:.3f} {cy + r:.3f} "
        f"A {r:.3f} {r:.3f} 0 1 1 {cx:.3f} {cy - r:.3f} Z "
    )


def _ring_clip_defs_circle(clip_id: str, cx: float, cy: float, r_inner: float, r_outer: float) -> str:
    inner = max(r_inner, 0.0)
    outer = max(r_outer, inner)
    if inner <= 0.0:
        d = _circle_path(cx, cy, outer)
    else:
        d = _circle_path(cx, cy, outer) + _circle_path(cx, cy, inner)
    return f'<defs><clipPath id="{clip_id}"><path d="{d}" fill-rule="evenodd"/></clipPath></defs>\n'


def _ring_band_path_circle(cx: float, cy: float, r_inner: float, r_outer: float) -> str:
    """
    Closed path for a full ring band (donut). Use with `fill-rule="evenodd"`.
    """
    inner = max(r_inner, 0.0)
    outer = max(r_outer, inner)
    if inner <= 0.0:
        return _circle_path(cx, cy, outer)
    return _circle_path(cx, cy, outer) + _circle_path(cx, cy, inner)


def _polar(cx: float, cy: float, r: float, ang: float) -> tuple[float, float]:
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def _ring_sector_path(cx: float, cy: float, r_inner: float, r_outer: float, start: float, sweep: float) -> str:
    sweep = max(0.0, min(2 * math.pi, sweep))
    if sweep <= 0.0 or r_outer <= 0.0:
        return ""
    end = start + sweep
    large = 1 if sweep > math.pi else 0

    x0, y0 = _polar(cx, cy, r_outer, start)
    x1, y1 = _polar(cx, cy, r_outer, end)

    if r_inner <= 0.0:
        # pie slice
        return (
            f"M {cx:.3f} {cy:.3f} "
            f"L {x0:.3f} {y0:.3f} "
            f"A {r_outer:.3f} {r_outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} "
            f"Z "
        )

    xi1, yi1 = _polar(cx, cy, r_inner, end)
    xi0, yi0 = _polar(cx, cy, r_inner, start)
    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {r_outer:.3f} {r_outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} "
        f"L {xi1:.3f} {yi1:.3f} "
        f"A {r_inner:.3f} {r_inner:.3f} 0 {large} 0 {xi0:.3f} {yi0:.3f} "
        f"Z "
    )


def _worked_sector_start_and_sweep(ops: object, start: float, consumed: int, prev_total: int) -> tuple[float, float]:
    denom = max(int(prev_total), 1)
    consumed_f = max(float(consumed), 0.0)
    offset = 0.0
    leading_chain = _leading_chain_occupancy(ops)
    if leading_chain is not None and 0.0 < leading_chain < consumed_f:
        offset = float(leading_chain)
    sector_start = start + (2 * math.pi * (offset / denom))
    sector_sweep = (2 * math.pi) * (max(0.0, consumed_f - offset) / denom)
    return sector_start, sector_sweep


def _worked_intervals_from_ops(ops: object) -> list[tuple[float, float]]:
    if not isinstance(ops, list):
        return []
    intervals: list[tuple[float, float]] = []
    cursor = 0.0

    def add_interval(start_pos: float, end_pos: float) -> None:
        if end_pos > start_pos:
            intervals.append((start_pos, end_pos))

    def walk(op_list: list[object]) -> None:
        nonlocal cursor
        i = 0
        while i < len(op_list):
            op = op_list[i]
            if not isinstance(op, dict):
                i += 1
                continue
            t = op.get("type")
            if t == "stitch":
                st = str(op.get("st", ""))
                n = max(0, int(op.get("n", 1)))
                if st in {"skip", "-"}:
                    cursor += n
                    i += 1
                    continue
                if st == "ch":
                    occupancy, next_i = _chain_occupancy_after(op_list, i)
                    cursor += occupancy
                    i = next_i
                    continue
                if st == "sl st":
                    if not isinstance(op.get("target"), dict):
                        cursor += 0 if _previous_op_is_chain(op_list, i) else n
                    i += 1
                    continue
                add_interval(cursor, cursor + n)
                cursor += n
            elif t == "skip":
                cursor += max(0, int(op.get("n", 1)))
            elif t == "inc":
                n = max(0, int(op.get("n", 1)))
                add_interval(cursor, cursor + n)
                cursor += n
            elif t == "dec":
                n = max(0, int(op.get("n", 1)))
                add_interval(cursor, cursor + (2 * n))
                cursor += 2 * n
            elif t == "cluster":
                add_interval(cursor, cursor + 1)
                cursor += 1
            elif t == "ch_space":
                cursor += 1
            elif t == "repeat":
                times = max(0, int(op.get("times", 1)))
                inner = op.get("ops", [])
                if isinstance(inner, list):
                    for _ in range(times):
                        walk(inner)
            i += 1

    walk(ops)
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if merged and abs(a - merged[-1][1]) < 1e-9:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _worked_sector_path_from_ops(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start: float,
    prev_total: int,
    ops: object,
) -> str:
    denom = max(int(prev_total), 1)
    parts: list[str] = []
    for a, b in _worked_intervals_from_ops(ops):
        sweep = (2 * math.pi) * (max(0.0, b - a) / denom)
        if sweep <= 0.0:
            continue
        sector_start = start + (2 * math.pi * (a / denom))
        d = _ring_sector_path(cx, cy, r_inner, r_outer, sector_start, sweep)
        if d:
            parts.append(d)
    return "".join(parts)


def _clip_defs_path(clip_id: str, d: str) -> str:
    return f'<defs><clipPath id="{clip_id}"><path d="{d}"/></clipPath></defs>\n'


def _clip_defs_path_with_transform(clip_id: str, d: str, transform: str | None) -> str:
    if transform:
        return f'<defs><clipPath id="{clip_id}"><path d="{d}" transform="{transform}"/></clipPath></defs>\n'
    return _clip_defs_path(clip_id, d)


def _rect_path(x: float, y: float, w: float, h: float) -> str:
    return f"M {x:.3f} {y:.3f} H {x + w:.3f} V {y + h:.3f} H {x:.3f} Z "


def _ring_clip_defs_square(
    clip_id: str, cx: float, cy: float, half_inner: float, half_outer: float, *, transform: str | None = None
) -> str:
    outer = max(half_outer, half_inner)
    inner = max(half_inner, 0.0)
    d = _rect_path(cx - outer, cy - outer, outer * 2, outer * 2)
    if inner > 0.0:
        d += _rect_path(cx - inner, cy - inner, inner * 2, inner * 2)
    if transform:
        return f'<defs><clipPath id="{clip_id}"><path d="{d}" fill-rule="evenodd" transform="{transform}"/></clipPath></defs>\n'
    return f'<defs><clipPath id="{clip_id}"><path d="{d}" fill-rule="evenodd"/></clipPath></defs>\n'


def _render_circle(pattern: PatternSpec, opts: RenderOptions) -> str:
    size = opts.size
    cx = cy = size / 2
    max_chain_count = _max_chain_span_count(pattern)
    canvas_extra = 0
    canvas_height = size
    canvas_width = size
    max_radius = (size / 2) - opts.padding
    n = len(pattern.rounds)
    weights = [_round_weight(pattern, i) for i in range(n)]
    mode, n0 = _effective_start(pattern, opts)
    hole = _center_hole_px(opts.center_hole_px, mode, n0, max_radius)
    # If we are going to modulate the *outermost* ring, reserve some headroom so
    # the silhouette can expand beyond the base radius. Without this, the outer
    # ring's r_outer == max_radius and bump has no visible effect.
    outer_idx = n - 1
    outer_has_seg_mod = False
    if isinstance(opts.circle_round_mod, dict) and int(opts.circle_round_mod.get("round_index", -2)) == outer_idx:
        outer_has_seg_mod = True
    outer_has_auto = False
    if opts.circle_auto_bump:
        scope = getattr(opts, "circle_auto_bump_scope", "outermost")
        if scope == "all":
            outer_has_auto = True
        else:
            target = opts.circle_auto_bump_round
            if target is None:
                target = outer_idx
            outer_has_auto = (target == outer_idx)
    reserve = 0.0
    if (outer_has_seg_mod or outer_has_auto) and max_radius > 0:
        reserve = min(72.0, max(18.0, max_radius * 0.22))
        reserve = max(0.0, min(reserve, max_radius - hole))
    base_max_radius = max_radius - reserve
    radii = _band_radii(base_max_radius, weights, inner_offset=hole)
    if max_chain_count > 0:
        chain_units = [
            _chain_unit_from_band(_visible_round_band(radii, i, fallback=opts.size * 0.055))
            for i in range(n)
        ]
        chain_unit_max = max(chain_units) if chain_units else (opts.size * 0.055)
        canvas_extra = int(
            round(min(size * 1.50, max(size * 0.20, max_chain_count * chain_unit_max * 0.90)))
        )
        canvas_width = size + (2 * canvas_extra)
        canvas_height = size + (2 * canvas_extra)

    parts: list[str] = [_svg_header(canvas_width, canvas_height)]
    parts.append(f'<rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" fill="{pattern.background}"/>\n')
    if canvas_extra:
        parts.append(f'<g transform="translate({canvas_extra} {canvas_extra})">\n')
    if hole > 0:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{hole:.3f}" fill="{pattern.background}" stroke="none"/>\n')

    # Keep a consistent rotation reference point for circular motifs: for circles we
    # rotate by shifting the angular "start" (instead of SVG transform).
    seam_marker: str | None = None
    scope_for_render = getattr(opts, "circle_auto_bump_scope", "outermost")

    if opts.circle_auto_bump and scope_for_render == "all":
        sample_steps = 720
        prev_boundary = [hole for _ in range(sample_steps + 1)]
        start = (-math.pi / 2) + (math.pi * float(opts.rotation_deg) / 180.0)
        ring_records: list[dict[str, object]] = []
        natural_max_radius = hole

        for i in range(n):
            r_outer = radii[i + 1]
            r_inner = radii[i]
            band_width = max(r_outer - r_inner, 0.0)
            color = pattern.rounds[i].color
            ops = pattern.rounds[i].stitch.ops
            chain_only = _is_chain_only_ops(ops)
            consumed = pattern.rounds[i].stitch.consumed
            if not isinstance(consumed, int):
                consumed = _estimate_consumed(ops)
            if " in ring" in pattern.rounds[i].stitch.stitch.lower() or " into ring" in pattern.rounds[i].stitch.stitch.lower():
                consumed = 1
            prev_total = _prev_total_with_override(pattern, i, consumed, opts.first_prev_total_override)

            prev_total_geom = prev_total
            if i == 0 and prev_total == 1 and consumed > 1:
                produced0 = pattern.rounds[i].stitch.total
                if not isinstance(produced0, int) or produced0 <= 0:
                    produced0 = _estimate_total_ops(ops)
                prev_total_geom = int(produced0) if isinstance(produced0, int) and produced0 > 0 else int(consumed)

            if consumed > prev_total and not (i == 0 and prev_total == 1):
                raise ValueError(
                    f"Round {i+1} consumes {consumed} stitches, but previous round has {prev_total}. "
                    "Refuse to render ambiguous structure. Consider fixing repeat counts or splitting into multiple rounds."
                )
            incomplete = (consumed < prev_total) or _has_skip(ops) or (_leading_chain_occupancy(ops) is not None)
            seg_mod = _mod_from_opts(opts, i, prev_total)
            field_mod = _bump_field_from_opts(pattern, opts, i, prev_total_geom) if not incomplete else None
            has_bump = (seg_mod is not None) or (field_mod is not None)

            bump_scale = 0.0
            if has_bump and not incomplete:
                bump_scale = min(band_width * 3.2, max(14.0, band_width * 2.6))
                bump_scale = min(bump_scale, max(0.0, max_radius - r_outer))

            sampled_bumps: list[float] = []
            if bump_scale > 0.0:
                for k in range(sample_steps + 1):
                    pos = (k / sample_steps) * max(prev_total_geom, 1)
                    sampled_bumps.append(_combined_bump_at(pos, prev_total_geom, seg_mod, field_mod))
                # Base band width already represents the round's normal advance.
                # Only carry the local variation forward, otherwise a uniform
                # round such as "10f" expands the whole boundary and hides later
                # rounds.
                base_bump = min(sampled_bumps) if sampled_bumps else 0.0
                sampled_bumps = [max(0.0, b - base_bump) for b in sampled_bumps]

            outer_boundary: list[float] = []
            for k, inner_r in enumerate(prev_boundary):
                bump = sampled_bumps[k] if sampled_bumps else 0.0
                outer_boundary.append(inner_r + band_width + (bump * bump_scale))

            natural_max_radius = max(natural_max_radius, max(outer_boundary, default=hole))
            ring_records.append(
                {
                    "i": i,
                    "inner": list(prev_boundary),
                    "outer": outer_boundary,
                    "r_inner": r_inner,
                    "r_outer": r_outer,
                    "prev_total_geom": prev_total_geom,
                    "incomplete": incomplete,
                    "consumed": consumed,
                    "prev_total": prev_total,
                    "color": color,
                    "chain_only": chain_only,
                }
            )
            if not incomplete:
                prev_boundary = outer_boundary

        radius_scale = 1.0
        if natural_max_radius > max_radius and natural_max_radius > hole:
            radius_scale = max(0.0, (max_radius - hole) / (natural_max_radius - hole))

        def scale_radius(r: float) -> float:
            return hole + ((float(r) - hole) * radius_scale)

        scaled_radii = [scale_radius(r) for r in radii]

        for record in ring_records:
            i = int(record["i"])
            color = str(record["color"])
            prev_total_geom = int(record["prev_total_geom"])
            incomplete = bool(record["incomplete"])
            chain_only = bool(record.get("chain_only", False))
            if incomplete:
                consumed = int(record["consumed"])
                prev_total = int(record["prev_total"])
                ops = pattern.rounds[i].stitch.ops
                d = _worked_sector_path_from_ops(
                    cx,
                    cy,
                    scale_radius(float(record["r_inner"])),
                    scale_radius(float(record["r_outer"])),
                    start,
                    prev_total,
                    ops,
                )
                r_inner_tex = scale_radius(float(record["r_inner"]))
                r_outer_tex = scale_radius(float(record["r_outer"]))
            else:
                inner_boundary = [scale_radius(r) for r in record["inner"]]  # type: ignore[index]
                outer_boundary = [scale_radius(r) for r in record["outer"]]  # type: ignore[index]
                d = _ring_path_from_radius_samples(cx, cy, start, inner_boundary, outer_boundary)
                r_inner_tex = min(inner_boundary) if inner_boundary else scale_radius(float(record["r_inner"]))
                r_outer_tex = max(outer_boundary) if outer_boundary else scale_radius(float(record["r_outer"]))
            if opts.texture:
                chain_unit = _chain_unit_from_band(_visible_round_band(scaled_radii, i, fallback=opts.size * 0.055))
                parts.append(
                    _render_circle_chain_spans(
                        pattern,
                        opts,
                        i,
                        r_inner_tex,
                        r_outer_tex,
                        prev_total_geom,
                        chain_unit_px=chain_unit,
                        r_cap=None,
                    )
                )
            if d:
                if not chain_only:
                    parts.append(f'<path d="{d}" fill="{color}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}"/>\n')

            if opts.texture:
                if not chain_only and d:
                    clip_id = f"ring_c_{i}"
                    parts.append(_clip_defs_path(clip_id, d))
                    parts.append(f'<g clip-path="url(#{clip_id})">\n')
                    parts.append(_render_circle_texture(pattern, opts, i, r_inner_tex, r_outer_tex, prev_total_geom, r_cap=max_radius))
                    parts.append("</g>\n")

        for i in range(n):
            overlay = _render_circle_ch_spaces(pattern, opts, i, scaled_radii, r_cap=None)
            if overlay:
                parts.append(overlay)

        if canvas_extra:
            parts.append("</g>\n")
        return "".join(parts) + _svg_footer()

    ch_space_overlays: list[str] = []

    # Draw from outside -> inside so inner rounds cover the previous fill.
    for i in range(n - 1, -1, -1):
        r_outer = radii[i + 1]
        r_inner = radii[i]
        color = pattern.rounds[i].color
        ops = pattern.rounds[i].stitch.ops
        chain_only = _is_chain_only_ops(ops)
        consumed = pattern.rounds[i].stitch.consumed
        if not isinstance(consumed, int):
            consumed = _estimate_consumed(ops)
        if " in ring" in pattern.rounds[i].stitch.stitch.lower() or " into ring" in pattern.rounds[i].stitch.stitch.lower():
            consumed = 1
        prev_total = _prev_total_with_override(pattern, i, consumed, opts.first_prev_total_override)
        ch_space_overlay = _render_circle_ch_spaces(pattern, opts, i, radii, r_cap=None)
        if ch_space_overlay:
            ch_space_overlays.append(ch_space_overlay)

        # Special case: Round-1 into a ring/magic ring.
        # Users may set `first_prev_total_override=1` (meaning a single ring space),
        # but the round can still *produce* many stitches (e.g. "12 f").
        # For geometry (angular placement, auto bump), we need an effective "around-the-circle"
        # count; otherwise all glyphs collapse onto the same angle.
        prev_total_geom = prev_total
        if i == 0 and prev_total == 1 and consumed > 1:
            produced0 = pattern.rounds[i].stitch.total
            if not isinstance(produced0, int) or produced0 <= 0:
                produced0 = _estimate_total_ops(ops)
            prev_total_geom = int(produced0) if isinstance(produced0, int) and produced0 > 0 else int(consumed)

        if consumed > prev_total and not (i == 0 and prev_total == 1):
            raise ValueError(
                f"Round {i+1} consumes {consumed} stitches, but previous round has {prev_total}. "
                "Refuse to render ambiguous structure. Consider fixing repeat counts or splitting into multiple rounds."
            )
        incomplete = (consumed < prev_total) or _has_skip(ops) or (_leading_chain_occupancy(ops) is not None)
        start = (-math.pi / 2) + (math.pi * float(opts.rotation_deg) / 180.0)
        sweep = (2 * math.pi) * (consumed / max(prev_total, 1))
        if i == 0 and prev_total == 1:
            sweep = 2 * math.pi
        sector_start, sector_sweep = _worked_sector_start_and_sweep(ops, start, consumed, prev_total)
        if i == 0 and prev_total == 1:
            sector_start, sector_sweep = start, 2 * math.pi
        seg_mod = _mod_from_opts(opts, i, prev_total)
        field_mod = _bump_field_from_opts(pattern, opts, i, prev_total_geom) if not incomplete else None
        has_bump = (seg_mod is not None) or (field_mod is not None)
        if opts.texture:
            chain_unit = _chain_unit_from_band(_visible_round_band(radii, i, fallback=opts.size * 0.055))
            parts.append(
                _render_circle_chain_spans(
                    pattern,
                    opts,
                    i,
                    r_inner,
                    r_outer,
                    prev_total_geom,
                    chain_unit_px=chain_unit,
                    r_cap=None,
                )
            )
        if chain_only:
            pass
        elif has_bump and not incomplete:
            # Increase modulation amplitude so bumps are visibly pronounced.
            bump_scale_base = min((r_outer - r_inner) * 3.2, max(14.0, (r_outer - r_inner) * 2.6))
            bump_scale_base = min(bump_scale_base, max(0.0, max_radius - r_outer))
            # When auto-bump applies to every ring:
            # - outermost ring expands outward (outer boundary modulation)
            # - inner rings modulate their *inner* boundary only (avoid covering the next ring)
            # - innermost ring keeps its inner boundary fixed so center holes stay visible
            scope = getattr(opts, "circle_auto_bump_scope", "outermost")
            outer_bump_scale = bump_scale_base if (scope != "all" or i == outer_idx) else 0.0
            inner_bump_scale = bump_scale_base if (scope == "all" and i not in (0, outer_idx)) else 0.0
            if getattr(opts, "circle_outline_sampling", "uniform") == "produced_subdivide":
                d = _modulated_ring_path_combined_produced(
                    cx,
                    cy,
                    r_inner,
                    r_outer,
                    start,
                    prev_total_geom,
                    seg_mod,
                    field_mod,
                    outer_bump_scale,
                    inner_bump_scale=inner_bump_scale,
                    r_cap=max_radius,
                    ops=ops,
                )
            else:
                d = _modulated_ring_path_combined(
                    cx,
                    cy,
                    r_inner,
                    r_outer,
                    start,
                    prev_total_geom,
                    seg_mod,
                    field_mod,
                    outer_bump_scale,
                    inner_bump_scale=inner_bump_scale,
                    r_cap=max_radius,
                )
            parts.append(
                f'<path d="{d}" fill="{color}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}"/>\n'
            )
            # Visual seam marker: defer until after all rings/textures are drawn
            # so inner rounds do not cover part of it. Start at the center-hole
            # boundary when a hole exists.
            scope = getattr(opts, "circle_auto_bump_scope", "outermost")
            if bool(getattr(opts, "show_seam_marker", True)) and scope == "outermost" and i == outer_idx:
                try:
                    pos0 = 0.0
                    bump0 = _combined_bump_at(pos0, prev_total_geom, seg_mod, field_mod) if has_bump else 0.0
                    ro0 = r_outer + bump0 * outer_bump_scale
                    if isinstance(max_radius, (int, float)):
                        ro0 = min(float(max_radius), ro0)
                    xse, yse = _polar(cx, cy, ro0, start)
                    x0, y0 = _polar(cx, cy, hole, start) if hole > 0 else (cx, cy)
                    seam_marker = f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{xse:.3f}" y2="{yse:.3f}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}" stroke-linecap="round"/>\n'
                except Exception:
                    pass
        else:
            if incomplete:
                d = _worked_sector_path_from_ops(cx, cy, r_inner, r_outer, start, prev_total, ops)
                if d:
                    parts.append(
                        f'<path d="{d}" fill="{color}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}"/>\n'
                    )
            else:
                d = _ring_band_path_circle(cx, cy, r_inner, r_outer)
                parts.append(
                    f'<path d="{d}" fill="{color}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}" fill-rule="evenodd"/>\n'
                )
        if opts.texture:
            if chain_only:
                continue
            clip_id = f"ring_c_{i}"
            if has_bump and not incomplete:
                bump_scale_base = min((r_outer - r_inner) * 3.2, max(14.0, (r_outer - r_inner) * 2.6))
                bump_scale_base = min(bump_scale_base, max(0.0, max_radius - r_outer))
                scope = getattr(opts, "circle_auto_bump_scope", "outermost")
                outer_bump_scale = bump_scale_base if (scope != "all" or i == outer_idx) else 0.0
                inner_bump_scale = bump_scale_base if (scope == "all" and i not in (0, outer_idx)) else 0.0
                if getattr(opts, "circle_outline_sampling", "uniform") == "produced_subdivide":
                    d = _modulated_ring_path_combined_produced(
                        cx,
                        cy,
                        r_inner,
                        r_outer,
                        start,
                        prev_total_geom,
                        seg_mod,
                        field_mod,
                        outer_bump_scale,
                        inner_bump_scale=inner_bump_scale,
                        r_cap=max_radius,
                        ops=ops,
                    )
                else:
                    d = _modulated_ring_path_combined(
                        cx,
                        cy,
                        r_inner,
                        r_outer,
                        start,
                        prev_total_geom,
                        seg_mod,
                        field_mod,
                        outer_bump_scale,
                        inner_bump_scale=inner_bump_scale,
                        r_cap=max_radius,
                    )
                parts.append(_clip_defs_path(clip_id, d))
            else:
                if incomplete:
                    d = _worked_sector_path_from_ops(cx, cy, r_inner, r_outer, start, prev_total, ops)
                    parts.append(_clip_defs_path(clip_id, d))
                else:
                    parts.append(_ring_clip_defs_circle(clip_id, cx, cy, r_inner, r_outer))
            parts.append(f'<g clip-path="url(#{clip_id})">\n')
            parts.append(_render_circle_texture(pattern, opts, i, r_inner, r_outer, prev_total_geom, r_cap=max_radius))
            parts.append("</g>\n")

    if ch_space_overlays:
        parts.extend(ch_space_overlays)
    if bool(getattr(opts, "show_seam_marker", True)) and seam_marker:
        parts.append(seam_marker)

    if canvas_extra:
        parts.append("</g>\n")
    return "".join(parts) + _svg_footer()


def _render_square(pattern: PatternSpec, opts: RenderOptions) -> str:
    size = opts.size
    n = len(pattern.rounds)
    max_half = (size / 2) - opts.padding
    weights = [_round_weight(pattern, i) for i in range(n)]
    mode, n0 = _effective_start(pattern, opts)
    hole = _center_hole_px(opts.center_hole_px, mode, n0, max_half)
    halves = _band_radii(max_half, weights, inner_offset=hole)

    parts: list[str] = [_svg_header(size)]
    parts.append(f'<rect x="0" y="0" width="{size}" height="{size}" fill="{pattern.background}"/>\n')
    cx = cy = size / 2
    rot = float(opts.rotation_deg or 0.0)
    transform = f"rotate({rot:.3f} {cx:.3f} {cy:.3f})" if abs(rot) > 1e-6 else None
    if transform:
        parts.append(f'<g transform="{transform}">\n')
    if hole > 0:
        xh = (size / 2) - hole
        yh = (size / 2) - hole
        parts.append(f'<rect x="{xh:.3f}" y="{yh:.3f}" width="{(2*hole):.3f}" height="{(2*hole):.3f}" fill="{pattern.background}" stroke="none"/>\n')

    # Outer -> inner nested squares
    for i in range(n - 1, -1, -1):
        half = halves[i + 1]
        x = (size / 2) - half
        y = (size / 2) - half
        w = h = half * 2
        color = pattern.rounds[i].color
        parts.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{color}" stroke="{opts.stroke}" stroke-width="{opts.stroke_width}"/>\n'
        )
        if opts.texture:
            half_outer = halves[i + 1]
            half_inner = halves[i]
            clip_id = f"ring_s_{i}"
            parts.append(_ring_clip_defs_square(clip_id, cx, cy, half_inner, half_outer, transform=transform))
            parts.append(f'<g clip-path="url(#{clip_id})">\n')
            ops = pattern.rounds[i].stitch.ops
            consumed = pattern.rounds[i].stitch.consumed
            if not isinstance(consumed, int):
                consumed = _estimate_consumed(ops)
            if " in ring" in pattern.rounds[i].stitch.stitch.lower() or " into ring" in pattern.rounds[i].stitch.stitch.lower():
                consumed = 1
            prev_total = _prev_total_with_override(pattern, i, consumed, opts.first_prev_total_override)
            if consumed > prev_total:
                raise ValueError(
                    f"Round {i+1} consumes {consumed} stitches, but previous round has {prev_total}. "
                    "Refuse to render ambiguous structure. Consider fixing repeat counts or splitting into multiple rounds."
                )
            parts.append(_render_square_texture(pattern, opts, i, half_inner, half_outer, prev_total))
            parts.append("</g>\n")

    if transform:
        parts.append("</g>\n")
    return "".join(parts) + _svg_footer()


def render_svg(pattern: PatternSpec, opts: RenderOptions | None = None) -> str:
    opts = opts or RenderOptions()
    if pattern.motif_type == "circle_coaster":
        return _render_circle(pattern, opts)
    if pattern.motif_type == "granny_square":
        return _render_square(pattern, opts)
    raise ValueError(f"Unsupported motif_type: {pattern.motif_type!r}")


def _write_out(out_path: Path, svg: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a stage-1 StitchSketch motif preview as SVG.")
    parser.add_argument("json", help="Path to pattern JSON")
    parser.add_argument("--out", "-o", default=None, help="Output SVG path (default: alongside JSON).")
    parser.add_argument("--size", type=int, default=512, help="SVG canvas size in px")
    parser.add_argument("--padding", type=int, default=24, help="Padding in px")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate the whole motif by degrees (clockwise).")
    parser.add_argument("--start-mode", default="auto", choices=["auto", "mr", "ch_join", "foundation_chain"])
    parser.add_argument("--start-n", type=int, default=None, help="Chain count for some start modes (e.g. ch_join).")
    parser.add_argument("--hole-px", type=float, default=None, help="Override center hole radius (px).")
    parser.add_argument("--round1-prev-total", type=int, default=None, help="Override round-1 prev_total for validation/layout.")
    parser.add_argument("--mod-round", type=int, default=None, help="Enable modulation on this round (1-based).")
    parser.add_argument("--mod-start-offset", type=int, default=0, help="Start offset in stitches for modulation.")
    parser.add_argument(
        "--mod-segments",
        default=None,
        help="Comma list of length:bump pairs whose lengths sum to prev_total, e.g. '3:0,3:1,6:0,3:1,9:0'.",
    )
    parser.add_argument("--auto-bump", action="store_true", help="Enable auto smooth bump (circle only).")
    parser.add_argument("--auto-bump-round", type=int, default=None, help="(Deprecated) Apply auto bump to this round (1-based).")
    parser.add_argument("--auto-bump-window", type=int, default=3, help="Smoothing window (odd, in base-stitch units).")
    parser.add_argument("--auto-bump-strength", type=float, default=0.9, help="Bump strength in [0..1].")
    args = parser.parse_args(list(argv) if argv is not None else None)

    pattern = load_pattern(args.json)
    circle_round_mod = None
    if args.mod_round is not None and args.mod_segments:
        segs = []
        for part in str(args.mod_segments).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            a, b = part.split(":", 1)
            segs.append({"length": int(a), "bump": float(b)})
        circle_round_mod = {"round_index": int(args.mod_round) - 1, "start_offset": int(args.mod_start_offset), "segments": segs}

    svg = render_svg(
        pattern,
        RenderOptions(
            size=args.size,
            padding=args.padding,
            rotation_deg=float(args.rotate_deg),
            start_mode=args.start_mode,
            start_n=args.start_n,
            center_hole_px=args.hole_px,
            first_prev_total_override=args.round1_prev_total,
            circle_round_mod=circle_round_mod,
            circle_auto_bump=bool(args.auto_bump),
            circle_auto_bump_scope="outermost",
            circle_auto_bump_round=(int(args.auto_bump_round) - 1) if args.auto_bump_round else None,
            circle_auto_bump_window=int(args.auto_bump_window),
            circle_auto_bump_strength=float(args.auto_bump_strength),
        ),
    )

    json_path = Path(args.json)
    out_path = Path(args.out) if args.out else json_path.with_suffix(".svg")
    _write_out(out_path, svg)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
