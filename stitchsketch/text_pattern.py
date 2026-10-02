from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from .schema import MotifType, PatternSpec, RoundSpec, RoundStitch
from .serialize import pattern_to_json


TokenType = Literal["LPAREN", "RPAREN", "LBRACK", "RBRACK", "COMMA", "MUL", "NUMBER", "STITCH", "SL_TARGET", "WORD"]


@dataclass(frozen=True)
class Token:
    t: TokenType
    v: str


_STITCH_ALIASES: dict[str, str] = {
    "ch": "ch",
    "chsp": "chsp",
    "x": "x",
    "t": "t",
    "f": "f",
    "e": "e",
    "v": "v",
    "tv": "tv",
    "fv": "fv",
    "a": "a",
    "inc": "v",
    "dec": "a",
    "sc": "sc",
    "hdc": "hdc",
    "dc": "dc",
    "tr": "tr",
    "dtr": "dtr",
    "sl st": "sl st",
    "sl": "sl st",
    "slst": "sl st",
    "ss": "sl st",
    "sk": "skip",
    "skip": "skip",
    "tog": "tog",
}

_IGNORE_WORDS = {
    "to",
    "join",
    "in",
    "ring",
    "into",
    "sp",
    "space",
    "st",
    "stitch",
    "rnd",
    "round",
    # Chinese descriptors / common words that may appear in mixed patterns
    "短针",
    "中长针",
    "长针",
    "长长针",
    "锁针",
    "加针",
    "减针",
    "引拔",
    "引拔针",
    "圈",
    "环",
}


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("slst", "sl st")
    s = s.replace("×", "x")
    # Common 2tog forms (normalize before generic spacing rules)
    s = re.sub(r"\b(sc|hdc|dc|tr|dtr)\s*2\s*tog\b", r"\1 2tog", s)
    s = re.sub(r"\b(sc|hdc|dc|tr|dtr)2tog\b", r"\1 2tog", s)
    s = re.sub(r"([a-z]+)(\d+)", r"\1 \2", s)
    s = re.sub(r"(\d+)([a-z]+)", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(line: str) -> list[Token]:
    s = _normalize_text(line)
    # Historical compatibility: treat {} as repeat-group brackets.
    s = s.replace("{", "[").replace("}", "]")
    tokens: list[Token] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(Token("LPAREN", c))
            i += 1
            continue
        if c == ")":
            tokens.append(Token("RPAREN", c))
            i += 1
            continue
        if c == "[":
            tokens.append(Token("LBRACK", c))
            i += 1
            continue
        if c == "]":
            tokens.append(Token("RBRACK", c))
            i += 1
            continue
        if c == ",":
            tokens.append(Token("COMMA", c))
            i += 1
            continue
        if c == "*":
            tokens.append(Token("MUL", c))
            i += 1
            continue
        if c == "-":
            tokens.append(Token("STITCH", "skip"))
            i += 1
            continue

        m = re.match(r"@ch\s*(\d+)", s[i:])
        if m:
            tokens.append(Token("SL_TARGET", f"ch{m.group(1)}"))
            i += len(m.group(0))
            continue
        m = re.match(r"@(\d+)\s*ch", s[i:])
        if m:
            tokens.append(Token("SL_TARGET", f"ch{m.group(1)}"))
            i += len(m.group(0))
            continue

        m = re.match(r"\d+", s[i:])
        if m:
            tokens.append(Token("NUMBER", m.group(0)))
            i += len(m.group(0))
            continue

        # Repeat marker "x" only when it multiplies a group: "x4" / "x 4"
        # Otherwise, "x" is a stitch symbol (short stitch / 短针) in some notations.
        if c == "x":
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j].isdigit():
                tokens.append(Token("MUL", "x"))
            else:
                tokens.append(Token("STITCH", _STITCH_ALIASES["x"]))
            i += 1
            continue

        # "sl st" must be checked before generic words
        if s.startswith("sl st", i):
            tokens.append(Token("STITCH", "sl st"))
            i += len("sl st")
            continue

        # CJK words (e.g. "短针") appear in some mixed patterns; treat as WORD.
        m = re.match(r"[\u4e00-\u9fff]+", s[i:])
        if m:
            tokens.append(Token("WORD", m.group(0)))
            i += len(m.group(0))
            continue

        m = re.match(r"[a-z]+", s[i:])
        if m:
            word = m.group(0)
            if word in _STITCH_ALIASES:
                tokens.append(Token("STITCH", _STITCH_ALIASES[word]))
            else:
                tokens.append(Token("WORD", word))
            i += len(word)
            continue

        raise ValueError(f"Unexpected character at {i}: {s[i:i+10]!r}")
    return tokens


# -------- AST --------


@dataclass(frozen=True)
class AstNode: ...


@dataclass(frozen=True)
class Stitch(AstNode):
    name: str
    count: Optional[int] = None
    count_position: Literal["before", "after"] = "before"
    target: Optional[dict] = None


@dataclass(frozen=True)
class Seq(AstNode):
    items: list[AstNode]


@dataclass(frozen=True)
class Repeat(AstNode):
    node: AstNode
    times: int


@dataclass(frozen=True)
class Cluster(AstNode):
    """
    Work all inner items into the same base stitch / space (consume=1).

    Stage-2 text syntax: parentheses `(...)` represent a cluster.
    """

    node: AstNode


@dataclass(frozen=True)
class ChainSpace(AstNode):
    """
    Work all inner items into a chain space/hole from the previous round.

    Syntax: `chsp(...)`.
    """

    node: AstNode


class _Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Optional[Token]:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _eat(self, t: TokenType | None = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of input")
        if t is not None and tok.t != t:
            raise ValueError(f"Expected {t}, got {tok.t} ({tok.v!r})")
        self._pos += 1
        return tok

    def parse(self) -> AstNode:
        node = self._parse_seq(until=None)
        if self._peek() is not None:
            tok = self._peek()
            raise ValueError(f"Unexpected token: {tok.t} {tok.v!r}")
        return node

    def _parse_seq(self, until: TokenType | None) -> AstNode:
        items: list[AstNode] = []
        while True:
            tok = self._peek()
            if tok is None or (until is not None and tok.t == until):
                break
            if tok.t == "COMMA":
                self._eat("COMMA")
                continue
            if tok.t == "WORD":
                raise ValueError(f"Unknown word in pattern: {tok.v!r}")
            items.append(self._parse_item())
        if not items:
            return Seq([])
        if len(items) == 1:
            return items[0]
        return Seq(items)

    def _parse_item(self) -> AstNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of input")
        # Support "12 [x, v]" as shorthand for "[x, v] * 12"
        if tok.t == "NUMBER":
            next_tok = self._tokens[self._pos + 1] if self._pos + 1 < len(self._tokens) else None
            if next_tok is not None and next_tok.t == "LBRACK":
                times = int(self._eat("NUMBER").v)
                self._eat("LBRACK")
                inner = self._parse_seq(until="RBRACK")
                self._eat("RBRACK")
                node: AstNode = Repeat(node=inner, times=times)
                rep = self._try_parse_repeat()
                if rep is not None:
                    node = Repeat(node=node, times=rep)
                return node
        if tok.t == "LPAREN":
            self._eat("LPAREN")
            inner = self._parse_seq(until="RPAREN")
            self._eat("RPAREN")
            # Parentheses always mean "cluster into one stitch".
            return Cluster(inner)

        if tok.t == "STITCH" and tok.v == "chsp":
            self._eat("STITCH")
            self._eat("LPAREN")
            inner = self._parse_seq(until="RPAREN")
            self._eat("RPAREN")
            return ChainSpace(inner)

        if tok.t == "LBRACK":
            self._eat("LBRACK")
            inner = self._parse_seq(until="RBRACK")
            self._eat("RBRACK")
            rep = self._try_parse_repeat()
            if rep is None:
                raise ValueError("Expected repeat count after [...] (use '* N' or 'xN').")
            return Repeat(node=inner, times=rep)

        node = self._parse_stitch()
        rep = self._try_parse_repeat()
        if rep is not None:
            node = Repeat(node=node, times=rep)
        return node

    def _try_parse_repeat(self) -> Optional[int]:
        tok = self._peek()
        if tok is None:
            return None
        if tok.t == "MUL":
            self._eat("MUL")
            n = int(self._eat("NUMBER").v)
            return n
        return None

    def _parse_stitch(self) -> Stitch:
        # Support: "ch 4" (stitch then number), "12 dc" (number then stitch),
        # and "3 dc" from "3dc" normalization.
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of input")

        if tok.t == "NUMBER":
            count = int(self._eat("NUMBER").v)
            name_tok = self._eat("STITCH")
            target = self._try_parse_sl_target(name_tok.v)
            return Stitch(name=name_tok.v, count=count, count_position="before", target=target)

        if tok.t == "STITCH":
            name = self._eat("STITCH").v
            if self._peek() is not None and self._peek().t == "NUMBER":
                count = int(self._eat("NUMBER").v)
                target = self._try_parse_sl_target(name)
                return Stitch(name=name, count=count, count_position="after", target=target)
            target = self._try_parse_sl_target(name)
            return Stitch(name=name, target=target)

        raise ValueError(f"Expected stitch, got {tok.t} {tok.v!r}")

    def _try_parse_sl_target(self, name: str) -> Optional[dict]:
        tok = self._peek()
        if tok is None or tok.t != "SL_TARGET":
            return None
        if name != "sl st":
            raise ValueError("@Nch/@chN target is only supported after sl/sl st.")
        target_tok = self._eat("SL_TARGET").v
        m = re.fullmatch(r"ch(\d+)", target_tok)
        if not m:
            raise ValueError(f"Invalid slip-stitch target: {target_tok!r}")
        return {"kind": "ch", "index": int(m.group(1))}


def parse_line(line: str) -> AstNode:
    tokens = tokenize(line)
    # remove ignorable words early to make parsing less brittle
    tokens = [t for t in tokens if not (t.t == "WORD" and t.v in _IGNORE_WORDS)]
    return _Parser(tokens).parse()


def _ast_to_canonical(node: AstNode) -> str:
    if isinstance(node, Stitch):
        target = ""
        if isinstance(node.target, dict):
            if node.target.get("kind") == "ch" and isinstance(node.target.get("index"), int):
                target = f"@ch{node.target['index']}"
        if node.name == "tog":
            # Prefer the more common "2tog" style in canonical form when count exists.
            if node.count is not None:
                return f"{node.count}tog{target}"
            return f"tog{target}"
        if node.count is None:
            return f"{node.name}{target}"
        if node.count_position == "after":
            return f"{node.name} {node.count}{target}"
        return f"{node.count} {node.name}{target}"
    if isinstance(node, Seq):
        parts: list[str] = []
        for item in node.items:
            rendered = _ast_to_canonical(item)
            if rendered:
                parts.append(rendered)
        return ", ".join(parts)
    if isinstance(node, Cluster):
        inner = _ast_to_canonical(node.node)
        return f"({inner})"
    if isinstance(node, ChainSpace):
        inner = _ast_to_canonical(node.node)
        return f"chsp({inner})"
    if isinstance(node, Repeat):
        inner = _ast_to_canonical(node.node)
        if isinstance(node.node, Stitch):
            return f"{inner} * {node.times}"
        return f"[{inner}] * {node.times}"
    raise TypeError(f"Unknown AST node: {type(node)}")


def _ast_to_ops(node: AstNode) -> list[dict]:
    # Backward-compatible wrapper: parse ops while tracking base stitch for v/a.
    ops, _ = _ast_to_ops_with_context(node, last_base=None)
    return ops


def _normalize_base(sym: str | None) -> str:
    if not sym:
        return "x"
    sym = {"sc": "x", "hdc": "t", "dc": "f"}.get(sym, sym)
    sym = {"e": "tr"}.get(sym, sym)
    if sym in {"x", "t", "f", "tr", "dtr"}:
        return sym
    return "x"


def _ast_to_ops_with_context(node: AstNode, *, last_base: str | None) -> tuple[list[dict], str | None]:
    """
    Convert AST to ops while tracking "base stitch" so that:
    - v (inc) and a (dec) default to x/sc to avoid local-context ambiguity.
    """
    if isinstance(node, Stitch):
        n = node.count if node.count is not None else 1
        if node.name == "tog":
            base = _normalize_base(last_base)
            return [{"type": "dec", "base": base, "n": n}], last_base
        if node.name == "v":
            base = "x"
            return [{"type": "inc", "base": base, "n": n}], last_base
        if node.name == "tv":
            return [{"type": "inc", "base": "t", "n": n}], last_base
        if node.name == "fv":
            return [{"type": "inc", "base": "f", "n": n}], last_base
        if node.name == "a":
            base = "x"
            return [{"type": "dec", "base": base, "n": n}], last_base
        if node.name == "skip":
            return [{"type": "skip", "n": n}], last_base
        if node.name == "sl st":
            op = {"type": "stitch", "st": node.name, "n": n}
            if isinstance(node.target, dict):
                op["target"] = dict(node.target)
            return [op], last_base
        # update base on real stitch symbols; keep last_base for ch/sl st
        if node.name in {"x", "t", "f", "e", "tr", "dtr", "sc", "hdc", "dc"}:
            return [{"type": "stitch", "st": node.name, "n": n}], _normalize_base(node.name)
        return [{"type": "stitch", "st": node.name, "n": n}], last_base

    if isinstance(node, Seq):
        out: list[dict] = []
        cur = last_base
        for item in node.items:
            ops_i, cur = _ast_to_ops_with_context(item, last_base=cur)
            out.extend(ops_i)
        return out, cur

    if isinstance(node, Repeat):
        # Resolve v/a bases inside the repeated group using the base context at the group's start.
        inner_ops, inner_last = _ast_to_ops_with_context(node.node, last_base=last_base)
        return [{"type": "repeat", "times": node.times, "ops": inner_ops}], inner_last
    if isinstance(node, Cluster):
        inner_ops, inner_last = _ast_to_ops_with_context(node.node, last_base=last_base)
        return [{"type": "cluster", "ops": inner_ops}], inner_last
    if isinstance(node, ChainSpace):
        inner_ops, inner_last = _ast_to_ops_with_context(node.node, last_base=last_base)
        return [{"type": "ch_space", "ops": inner_ops}], inner_last

    raise TypeError(f"Unknown AST node: {type(node)}")

def _estimate_total_from_ops(ops: list[dict]) -> int:
    total = 0
    for op in ops:
        t = op.get("type")
        if t == "stitch":
            if str(op.get("st", "")) == "sl st":
                continue
            total += int(op.get("n", 1))
        elif t == "inc":
            # one increase operation yields 2 stitches (e.g. 2sc in one stitch)
            total += 2 * int(op.get("n", 1))
        elif t == "dec":
            # one decrease operation yields 1 stitch (e.g. sc2tog)
            total += int(op.get("n", 1))
        elif t == "skip":
            total += 0
        elif t == "cluster":
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("cluster.ops must be a list")
            total += _estimate_total_from_ops(inner)
        elif t == "ch_space":
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("ch_space.ops must be a list")
            total += _estimate_total_from_ops(inner)
        elif t == "repeat":
            times = int(op.get("times", 1))
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("repeat.ops must be a list")
            total += times * _estimate_total_from_ops(inner)
        else:
            raise ValueError(f"Unknown op type: {t!r}")
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


def _chain_occupancy_after(ops: list[dict], index: int) -> tuple[int, int]:
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


def _previous_op_is_chain(ops: list[dict], index: int) -> bool:
    j = index - 1
    while j >= 0 and _is_skip_op(ops[j]):
        j -= 1
    if j < 0:
        return False
    prev = ops[j]
    return isinstance(prev, dict) and prev.get("type") == "stitch" and str(prev.get("st", "")) == "ch"


def _estimate_consumed_from_ops(ops: list[dict]) -> int:
    consumed = 0
    i = 0
    while i < len(ops):
        op = ops[i]
        t = op.get("type")
        if t == "stitch":
            if str(op.get("st", "")) == "sl st":
                if not isinstance(op.get("target"), dict):
                    consumed += 0 if _previous_op_is_chain(ops, i) else max(0, int(op.get("n", 1)))
                i += 1
                continue
            if str(op.get("st", "")) == "ch":
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
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("cluster.ops must be a list")
            # cluster works into one base stitch/space
            _ = _estimate_consumed_from_ops(inner)  # validate op structure
            consumed += 1
        elif t == "ch_space":
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("ch_space.ops must be a list")
            _ = _estimate_consumed_from_ops(inner)  # validate op structure
            consumed += 1
        elif t == "repeat":
            times = int(op.get("times", 1))
            inner = op.get("ops", [])
            if not isinstance(inner, list):
                raise ValueError("repeat.ops must be a list")
            consumed += times * _estimate_consumed_from_ops(inner)
        else:
            raise ValueError(f"Unknown op type: {t!r}")
        i += 1
    return consumed


def _is_working_into_ring(line: str) -> bool:
    s = line.lower()
    return bool(re.search(r"\b(into|in)\s+ring\b", s))


def _is_ring_setup(line: str) -> bool:
    s = line.lower()
    # common: "ch 4, sl st to join" (forming a ring)
    return ("ch" in s) and ("sl st" in s) and ("join" in s)


_STITCH_LEGEND: dict[str, str] = {
    "ch": "chain / 锁针",
    "chsp": "chain space / 锁针洞内钩织",
    "sl st": "slip stitch / 引拔针",
    "sc": "single crochet / 短针",
    "x": "single crochet / 短针 (sc)",
    "t": "half double crochet / 中长针 (hdc)",
    "f": "double crochet / 长针 (dc)",
    "e": "treble crochet / 长长针 (tr)",
    "v": "increase / 加针 (inc)",
    "tv": "t increase / 中长针加针",
    "fv": "f increase / 长针加针",
    "a": "decrease / 减针 (dec)",
    "hdc": "half double crochet / 半长针",
    "dc": "double crochet / 长针",
    "tr": "treble crochet / 长长针",
    "dtr": "double treble crochet / 2次长长针",
    "skip": "skip / 跳针",
    "tog": "together / 并针（默认按 dec 解析）",
}


def _collect_stitches(node: AstNode) -> set[str]:
    if isinstance(node, Stitch):
        return set() if node.name == "chsp" else {node.name}
    if isinstance(node, Seq):
        out: set[str] = set()
        for item in node.items:
            out |= _collect_stitches(item)
        return out
    if isinstance(node, Cluster):
        return _collect_stitches(node.node)
    if isinstance(node, ChainSpace):
        return _collect_stitches(node.node)
    if isinstance(node, Repeat):
        return _collect_stitches(node.node)
    return set()


_DEFAULT_PALETTE = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#1C2541",
    "#3A506B",
    "#5BC0BE",
]


def text_to_pattern(
    text: str,
    *,
    motif_type: MotifType = "circle_coaster",
    title: str | None = None,
    background: str = "#ffffff",
    palette: list[str] | None = None,
) -> PatternSpec:
    palette = palette or _DEFAULT_PALETTE
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Text pattern is empty.")

    rounds: list[RoundSpec] = []
    used_stitches: set[str] = set()
    for idx, ln in enumerate(lines):
        ast = parse_line(ln)
        used_stitches |= _collect_stitches(ast)
        canonical = _ast_to_canonical(ast)
        ops = _ast_to_ops(ast)
        total = _estimate_total_from_ops(ops)
        consumed = _estimate_consumed_from_ops(ops)
        # If the line explicitly says we're working "in/into ring", treat it as a
        # single base space for consumed-stitch accounting.
        if _is_working_into_ring(ln):
            consumed = 1
        if idx > 0:
            # Special-case: if previous line is a ring setup ("ch n, sl st to join"),
            # the next round typically works into a single ring space.
            prev_line = lines[idx - 1]
            prev_total = 1 if _is_ring_setup(prev_line) else (rounds[idx - 1].stitch.total or 0)
            if prev_total > 0 and consumed > prev_total:
                raise ValueError(
                    "结构不自洽：第 {r} 圈消耗了 {consumed} 个基针位，但上一圈只有 {prev} 针。"
                    " 这通常表示重复次数写多了，或需要把这一圈拆成多圈。"
                    "（例：上一圈 `12 x` 后直接写 `12 (x,v)` 会出现该问题）".format(
                        r=idx + 1, consumed=consumed, prev=prev_total
                    )
                )
        color = palette[idx % len(palette)]
        rounds.append(RoundSpec(color=color, stitch=RoundStitch(canonical, ops=ops, total=total, consumed=consumed)))

    legend = {k: _STITCH_LEGEND.get(k, k) for k in sorted(used_stitches)} if used_stitches else None
    return PatternSpec(motif_type=motif_type, rounds=rounds, title=title, background=background, legend=legend)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse crochet text pattern into StitchSketch stage-1 JSON.")
    parser.add_argument("--motif-type", default="circle_coaster", choices=["circle_coaster", "granny_square"])
    parser.add_argument("--title", default=None)
    parser.add_argument("--background", default="#ffffff")
    parser.add_argument("--text", default=None, help="Pattern text (use \\n for multiple lines).")
    parser.add_argument("--file", default=None, help="Read pattern text from a file.")
    parser.add_argument("--out", "-o", default=None, help="Write JSON to a file; otherwise prints.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if (args.text is None) == (args.file is None):
        raise SystemExit("Provide exactly one of --text or --file.")

    text = args.text.replace("\\n", "\n") if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    pattern = text_to_pattern(text, motif_type=args.motif_type, title=args.title, background=args.background)
    out = pattern_to_json(pattern)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(str(out_path))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
