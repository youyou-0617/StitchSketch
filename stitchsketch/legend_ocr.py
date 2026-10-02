from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageOps


def _run_tesseract(img: Image.Image, *, lang: str) -> str:
    """
    Run system `tesseract` CLI on a small text crop.
    Returns raw text (may be empty).
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "in.png"
        out_base = td_path / "out"
        img.save(in_path)
        cmd = ["tesseract", str(in_path), str(out_base), "-l", lang, "--psm", "7"]
        subprocess.run(cmd, check=False, capture_output=True, text=True)
        txt_path = out_base.with_suffix(".txt")
        if not txt_path.exists():
            return ""
        return txt_path.read_text(encoding="utf-8", errors="ignore")

def _available_langs() -> set[str]:
    try:
        out = subprocess.run(["tesseract", "--list-langs"], check=False, capture_output=True, text=True).stdout
    except Exception:
        return set()
    langs: set[str] = set()
    for line in (out or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of available"):
            continue
        if line.endswith(":"):
            continue
        # typical lines are just "eng"
        if re.fullmatch(r"[a-zA-Z0-9_+.-]+", line):
            langs.add(line)
    return langs


def _prep_for_ocr(img: Image.Image) -> Image.Image:
    # grayscale + upscale + contrast
    g = ImageOps.grayscale(img)
    g = g.resize((g.size[0] * 3, g.size[1] * 3), resample=Image.Resampling.BILINEAR)
    g = ImageEnhance.Contrast(g).enhance(2.2)
    g = ImageEnhance.Sharpness(g).enhance(1.8)
    # simple threshold based on percentile
    import numpy as np

    arr = np.array(g, dtype="uint8")
    thr = int(np.percentile(arr, 62))
    bw = Image.fromarray((arr < thr).astype("uint8") * 255, mode="L")
    return bw


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("|", "1")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", " ", s)
    # Keep CJK / alnum / a few punctuation
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff()·+*/\- ]+", "", s)
    s = s.strip()
    return s


def _suggest_stitch(name: str) -> str | None:
    """
    Best-effort mapping from Chinese names to stage2 symbols.
    """
    n = name
    if not n:
        return None
    # ordering matters (longer first)
    rules: list[tuple[str, str]] = [
        ("长长针", "e"),
        ("长针", "f"),
        ("中长针", "t"),
        ("短针", "x"),
        ("引拔", "sl st"),
        ("锁针", "ch"),
        ("辫子", "ch"),
        ("加针", "v"),
        ("减针", "a"),
    ]
    for k, v in rules:
        if k in n:
            return v
    return None


def ocr_library(dir_path: Path, *, lang_primary: str = "chi_sim", lang_fallback: str = "eng") -> dict[str, Any]:
    lib_path = dir_path / "library.json"
    data = json.loads(lib_path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("library.json missing `entries` list")

    updated = 0
    langs = _available_langs()
    primary_ok = (not lang_primary) or (lang_primary in langs)
    fallback_ok = (not lang_fallback) or (lang_fallback in langs)
    for e in entries:
        if not isinstance(e, dict):
            continue
        text_path = e.get("text_path")
        if not isinstance(text_path, str) or not text_path:
            continue
        img_path = dir_path / text_path
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        bw = _prep_for_ocr(img)
        raw = ""
        cleaned = ""
        raw2 = ""
        if primary_ok and lang_primary:
            raw = _run_tesseract(bw, lang=lang_primary)
            cleaned = _clean_text(raw)
        if (not cleaned) and fallback_ok and lang_fallback:
            raw2 = _run_tesseract(bw, lang=lang_fallback)
            cleaned = _clean_text(raw2)

        e["name_ocr_raw"] = raw.strip() if isinstance(raw, str) else ""
        e["name_ocr_raw_fallback"] = raw2.strip() if isinstance(raw2, str) else ""
        e["name_ocr"] = cleaned or None
        e["stitch_suggest"] = _suggest_stitch(cleaned) if cleaned else None
        updated += 1

    data["entries"] = entries
    data["ocr"] = {
        "lang_primary": lang_primary,
        "lang_fallback": lang_fallback,
        "available_langs": sorted(langs),
        "primary_lang_available": primary_ok,
        "fallback_lang_available": fallback_ok,
        "notes": "name_ocr is best-effort; please review/correct manually.",
    }
    data["updated_entries"] = updated
    return data


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR `text_*.png` in a StitchSketch symbol library directory.")
    parser.add_argument("--dir", "-d", required=True, help="Symbol library directory containing library.json.")
    parser.add_argument("--in-place", action="store_true", help="Write results back to library.json.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <dir>/library_ocr.json).")
    parser.add_argument("--lang", default="chi_sim", help="Primary tesseract lang (default chi_sim).")
    parser.add_argument("--fallback-lang", default="eng", help="Fallback tesseract lang (default eng).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    d = Path(args.dir)
    out_data = ocr_library(d, lang_primary=str(args.lang), lang_fallback=str(args.fallback_lang))

    if args.in_place:
        (d / "library.json").write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(d / "library.json"))
        return 0

    out_path = Path(args.out) if args.out else (d / "library_ocr.json")
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
