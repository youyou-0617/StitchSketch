from __future__ import annotations

import re
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class TranslationResult:
    code: str
    notes: list[str]


def _normalize_punctuation(text: str) -> str:
    s = text or ""
    table = str.maketrans(
        {
            "，": ",",
            "。": ".",
            "：": ":",
            "；": ";",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "［": "[",
            "］": "]",
            "＊": "*",
            "×": "*",
            "、": ",",
        }
    )
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = s[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = s[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_rounds(text: str) -> list[tuple[int, str]]:
    s = _normalize_punctuation(text)
    # Normalize common round labels to Rn: so OCR line breaks do not matter.
    s = re.sub(r"(第\s*(\d+)\s*[圈行])\s*[:：]?", lambda m: f" R{m.group(2)}: ", s)
    matches = list(re.finditer(r"\bR\s*(\d+)\s*[:：]", s, flags=re.IGNORECASE))
    if not matches:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [(i + 1, ln) for i, ln in enumerate(lines)]

    rounds: list[tuple[int, str]] = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(s)
        body = s[start:end].strip(" ,;")
        if body:
            rounds.append((int(m.group(1)), body))
    return rounds


def _remove_legend_lines(text: str) -> str:
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Lines like "F:长针 ch:锁针 sl:引拔" are legends, not rounds.
        if re.search(r"\b(ch|sl|x|t|f|e)\s*[:：]", line, flags=re.IGNORECASE) and not re.search(
            r"\bR\s*\d+\s*[:：]", line, flags=re.IGNORECASE
        ):
            continue
        if not re.search(r"\bR\s*\d+\s*[:：]|第\s*\d+\s*[圈行]", line, flags=re.IGNORECASE):
            # Ignore title-only lines with no numbers or stitch names.
            if not re.search(r"\d|锁针|短针|中长针|长针|引拔|ch|sl|sc|hdc|dc|tr", line, flags=re.IGNORECASE):
                continue
        kept.append(line)
    return "\n".join(kept)


def _translate_stitches(s: str) -> str:
    # Longer names first.
    replacements = [
        ("长长针", "e"),
        ("中长针", "t"),
        ("短针", "x"),
        ("长针", "f"),
        ("引拔针", "sl"),
        ("引拔", "sl"),
        ("锁针", "ch"),
        ("辫子针", "ch"),
        ("辫子", "ch"),
        ("空针", "skip"),
        ("跳针", "skip"),
        ("加针", "v"),
        ("减针", "a"),
    ]
    for cn, sym in replacements:
        s = s.replace(cn, sym)

    # Normalize English symbols, especially uppercase chart abbreviations.
    s = re.sub(r"\bSC\b", "x", s, flags=re.IGNORECASE)
    s = re.sub(r"\bHDC\b", "t", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDC\b", "f", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTR\b", "e", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSL\s*ST\b", "sl", s, flags=re.IGNORECASE)
    s = re.sub(r"\bstl\b", "sl", s, flags=re.IGNORECASE)
    s = re.sub(r"\bZch\b", "7ch", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d)Zch\b", r"\1ch", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d+)\s*F\b", r"\1f", s)
    s = re.sub(r"(\d+)\s*T\b", r"\1t", s)
    s = re.sub(r"(\d+)\s*X\b", r"\1x", s)
    s = re.sub(r"\bF\b", "f", s)
    s = re.sub(r"\bT\b", "t", s)
    s = re.sub(r"\bX\b", "x", s)
    return s


def _extract_sl_target(s: str) -> tuple[str, str | None]:
    target: str | None = None

    def repl_cn(m: re.Match[str]) -> str:
        nonlocal target
        target = f"sl@ch{m.group(1)}"
        return ""

    s = re.sub(r"引拔[^,;，；。]*第\s*(\d+)\s*个?\s*(?:锁针|ch)[^,;，；。]*", repl_cn, s)
    s = re.sub(r"sl\s*(?:在|到|至)?[^,;]*第\s*(\d+)\s*个?\s*ch[^,;]*", repl_cn, s, flags=re.IGNORECASE)
    return s, target


def _strip_hints(s: str) -> tuple[str, bool]:
    chsp = bool(re.search(r"上一圈[^,;)]*(?:ch|锁针)[^,;)]*处钩织|(?:ch|锁针)[^,;)]*洞|锁针洞", s, flags=re.IGNORECASE))

    def remove_hint(m: re.Match[str]) -> str:
        nonlocal chsp
        inner = m.group(1)
        if re.search(r"上一圈|处钩织|锁针洞|ch\s*处|长针处|引拔", inner, flags=re.IGNORECASE):
            return ""
        # OCR often damages Chinese hints like "上一圈2ch处钩织" into a noisy
        # parenthetical that still contains "2ch". Treat a non-pattern-looking
        # parenthetical with 2ch as a chain-space hint instead of literal code.
        if re.search(r"2\s*ch", inner, flags=re.IGNORECASE) and re.search(r"[^0-9a-zA-Z,()[\]\s*]", inner):
            chsp = True
            return ""
        return m.group(0)

    s = re.sub(r"\(([^()]*)\)", remove_hint, s)
    s = re.sub(r"环形起针|魔术环起针|起立的?|处钩织|钩织|组", "", s)
    return s, chsp


def _convert_prefix_repeats(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(
            r"(?<![\w\]])(\d+)\s*\*\s*\(([^()]*)\)",
            lambda m: f"[{m.group(2).strip()}]*{m.group(1)}",
            s,
        )
        s = re.sub(
            r"(?<![\w\]])(\d+)\s*\*\s*\[([^\[\]]*)\]",
            lambda m: f"[{m.group(2).strip()}]*{m.group(1)}",
            s,
        )
    return s


def _convert_trailing_repeat(s: str) -> str:
    m = re.search(r"(?:重复|repeat)\s*(\d+)\s*(?:组|次)?\s*$", s, flags=re.IGNORECASE)
    if not m:
        # OCR commonly turns "重复6组" into fragments such as "BH628",
        # "#628", or "B64". Only apply this to a line that already looks like
        # a repeated sl/ch motif to avoid rewriting ordinary code.
        noisy = re.search(r"[, ](?:b|bh|#|h|8|6|2){1,6}6(?:2?8|4)?\s*$", s, flags=re.IGNORECASE)
        if noisy and len(re.findall(r"\bsl\b", s, flags=re.IGNORECASE)) >= 3 and len(re.findall(r"\b\d+\s*ch\b", s, flags=re.IGNORECASE)) >= 3:
            body = s[: noisy.start()].strip(" ,;")
            return f"[{body}]*6" if body else s
    if not m:
        return s
    n = m.group(1)
    body = s[: m.start()].strip(" ,;")
    if not body:
        return s
    if body.startswith("[") and re.search(r"\]\s*\*\s*\d+\s*$", body):
        return body
    return f"[{body}]*{n}"


def _normalize_code_spacing(s: str) -> str:
    s = _normalize_punctuation(s)
    s = _translate_stitches(s)
    s = _convert_prefix_repeats(s)
    s = _convert_trailing_repeat(s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\[\s*", "[", s)
    s = re.sub(r"\s*\]", "]", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    s = re.sub(r"\s*\*\s*", "*", s)
    return s.strip(" ,;.")


def _wrap_chsp(expr: str) -> str:
    expr = expr.strip()
    if not expr or "chsp(" in expr:
        return expr
    m = re.fullmatch(r"\[(.*)\]\s*\*\s*(\d+)", expr)
    if m:
        return f"[chsp({m.group(1).strip()})]*{m.group(2)}"

    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(f"chsp({', '.join(buf)})")
            buf.clear()

    for part in _split_top_level_commas(expr):
        repeat = re.fullmatch(r"\[(.*)\]\s*\*\s*(\d+)", part)
        if repeat:
            flush()
            out.append(f"[chsp({repeat.group(1).strip()})]*{repeat.group(2)}")
        else:
            buf.append(part)
    flush()
    return ", ".join(out)


def _convert_round(body: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    s, sl_target = _extract_sl_target(body)
    s, use_chsp = _strip_hints(s)
    s = _normalize_code_spacing(s)
    if sl_target:
        s = f"{s}, {sl_target}" if s else sl_target
    if use_chsp:
        s = _wrap_chsp(s)
    if re.search(r"上一圈|起针|钩织|引拔在|第\s*\d+\s*个", s):
        notes.append("有未完全识别的中文描述，请检查转换结果。")
    return s, notes


def natural_text_to_pattern(text: str) -> TranslationResult:
    cleaned = _remove_legend_lines(text)
    rounds = _split_rounds(cleaned)
    notes: list[str] = []
    lines: list[str] = []
    if not rounds:
        code, round_notes = _convert_round(cleaned)
        if code:
            lines.append(code)
        notes.extend(round_notes)
    else:
        for _idx, body in rounds:
            code, round_notes = _convert_round(body)
            if code:
                lines.append(code)
            notes.extend(round_notes)

    code = "\n".join(lines)
    if not code.strip():
        notes.append("没有识别到可转换的圈/行。")
    return TranslationResult(code=code, notes=notes)


def _available_tesseract_langs() -> set[str]:
    try:
        out = subprocess.run(["tesseract", "--list-langs"], check=False, capture_output=True, text=True).stdout
    except Exception:
        return set()
    langs: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if line and not line.lower().startswith("list of available") and re.fullmatch(r"[a-zA-Z0-9_+.-]+", line):
            langs.add(line)
    return langs


def ocr_image_bytes(data: bytes) -> str:
    img = Image.open(BytesIO(data)).convert("RGB")
    g = ImageOps.grayscale(img)
    max_side = max(g.size)
    scale = 2 if max_side > 1400 else 3
    g = g.resize((g.size[0] * scale, g.size[1] * scale), resample=Image.Resampling.BILINEAR)
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = ImageEnhance.Sharpness(g).enhance(1.6)

    langs = _available_tesseract_langs()
    lang = "chi_sim+eng" if {"chi_sim", "eng"}.issubset(langs) else ("chi_sim" if "chi_sim" in langs else "eng")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "input.png"
        out_base = td_path / "out"
        g.save(in_path)
        cmd = ["tesseract", str(in_path), str(out_base), "-l", lang, "--psm", "6"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        txt_path = out_base.with_suffix(".txt")
        if not txt_path.exists():
            detail = proc.stderr.strip() or "tesseract did not produce text"
            raise RuntimeError(detail)
        return txt_path.read_text(encoding="utf-8", errors="ignore").strip()
