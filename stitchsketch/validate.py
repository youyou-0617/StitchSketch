from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .schema import PatternSpec


RoundStatus = Literal["ok", "incomplete", "overconsume"]


@dataclass(frozen=True)
class RoundValidation:
    round_index: int  # 0-based
    prev_total: int
    consumed: int
    produced: int
    has_skip: bool
    status: RoundStatus
    message: Optional[str] = None


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


def _is_ring_setup(text: str) -> bool:
    s = (text or "").lower()
    return ("ch" in s) and ("sl st" in s) and ("join" in s)


def _is_working_into_ring(text: str) -> bool:
    s = (text or "").lower()
    return (" in ring" in s) or (" into ring" in s)


def _estimate_total(ops: object) -> int:
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
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                continue
            total += _estimate_total(inner)
        elif t == "ch_space":
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                continue
            total += _estimate_total(inner)
        elif t == "repeat":
            times = int(op.get("times", 1))
            total += times * _estimate_total(op.get("ops"))
    return total


def _is_skip_op(op: object) -> bool:
    if not isinstance(op, dict):
        return False
    if op.get("type") == "skip":
        return True
    return op.get("type") == "stitch" and str(op.get("st", "")) in {"skip", "-"}


def _skip_op_n(op: object) -> int:
    if not isinstance(op, dict):
        return 0
    return max(0, int(op.get("n", 1)))


def _chain_occupancy_after(ops: list[object], index: int) -> tuple[int, int]:
    skipped = 0
    j = index + 1
    while j < len(ops) and _is_skip_op(ops[j]):
        skipped += _skip_op_n(ops[j])
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
    while j >= 0 and _is_skip_op(ops[j]):
        j -= 1
    if j < 0:
        return False
    prev = ops[j]
    return isinstance(prev, dict) and prev.get("type") == "stitch" and str(prev.get("st", "")) == "ch"


def _estimate_consumed(ops: object) -> int:
    if not isinstance(ops, list):
        return 0
    consumed = 0
    i = 0
    while i < len(ops):
        op = ops[i]
        if not isinstance(op, dict):
            i += 1
            continue
        t = op.get("type")
        if t == "stitch":
            st = str(op.get("st", ""))
            if st == "sl st":
                if not isinstance(op.get("target"), dict):
                    consumed += 0 if _previous_op_is_chain(ops, i) else max(0, int(op.get("n", 1)))
                i += 1
                continue
            if st == "ch":
                occupancy, next_i = _chain_occupancy_after(ops, i)
                consumed += occupancy
                i = next_i
                continue
            consumed += int(op.get("n", 1))
        elif t == "inc":
            consumed += int(op.get("n", 1))
        elif t == "dec":
            consumed += 2 * int(op.get("n", 1))
        elif t == "skip":
            consumed += int(op.get("n", 1))
        elif t == "cluster":
            # consume exactly one base stitch/space
            inner = op.get("ops", [])
            _ = _estimate_consumed(inner)
            consumed += 1
        elif t == "ch_space":
            inner = op.get("ops", [])
            _ = _estimate_consumed(inner)
            consumed += 1
        elif t == "repeat":
            times = int(op.get("times", 1))
            consumed += times * _estimate_consumed(op.get("ops"))
        i += 1
    return consumed


def validate_pattern(pattern: PatternSpec, *, first_prev_total_override: int | None = None) -> list[RoundValidation]:
    out: list[RoundValidation] = []
    for i, r in enumerate(pattern.rounds):
        ops = r.stitch.ops
        produced = r.stitch.total if isinstance(r.stitch.total, int) else _estimate_total(ops)
        consumed = r.stitch.consumed if isinstance(r.stitch.consumed, int) else _estimate_consumed(ops)
        if _is_working_into_ring(r.stitch.stitch):
            consumed = 1
        has_skip = _has_skip(ops)

        if i == 0:
            if isinstance(first_prev_total_override, int) and first_prev_total_override > 0:
                prev_total = first_prev_total_override
            else:
                prev_total = consumed if consumed > 0 else max(produced, 1)
        else:
            prev = pattern.rounds[i - 1].stitch.total
            if _is_ring_setup(pattern.rounds[i - 1].stitch.stitch):
                prev_total = 1
            else:
                prev_total = prev if isinstance(prev, int) and prev > 0 else max(consumed, 1)

        if consumed > prev_total:
            out.append(
                RoundValidation(
                    round_index=i,
                    prev_total=prev_total,
                    consumed=consumed,
                    produced=produced,
                    has_skip=has_skip,
                    status="overconsume",
                    message="consumed > prev_total：结构不自洽（重复次数可能写多，或需要拆分成多圈）",
                )
            )
            continue

        if (not has_skip) and consumed < prev_total:
            out.append(
                RoundValidation(
                    round_index=i,
                    prev_total=prev_total,
                    consumed=consumed,
                    produced=produced,
                    has_skip=has_skip,
                    status="incomplete",
                    message="未织完一圈：consumed < prev_total 且无 skip",
                )
            )
            continue

        out.append(
            RoundValidation(
                round_index=i,
                prev_total=prev_total,
                consumed=consumed,
                produced=produced,
                has_skip=has_skip,
                status="ok",
                message=None,
            )
        )
    return out
