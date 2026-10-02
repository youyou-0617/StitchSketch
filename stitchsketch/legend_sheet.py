from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def clamp(self, w: int, h: int) -> "BBox":
        x0 = max(0, min(w, int(self.x0)))
        x1 = max(0, min(w, int(self.x1)))
        y0 = max(0, min(h, int(self.y0)))
        y1 = max(0, min(h, int(self.y1)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBox(x0, y0, x1, y1)

    def as_list(self) -> list[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV not installed.")


def _to_bin(img: Image.Image) -> np.ndarray:
    gray = np.array(ImageOps.grayscale(img), dtype=np.uint8)
    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 7
        )
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        return thr

    # Fallback: simple Otsu + light open using numpy
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    total = gray.size
    if total <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr_val = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thr_val = t
    bin_img = (gray <= thr_val).astype(np.uint8) * 255

    def erode3(a: np.ndarray) -> np.ndarray:
        p = np.pad(a > 0, 1, mode="constant", constant_values=False)
        out = np.ones_like(a, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out &= p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]
        return (out.astype(np.uint8) * 255)

    def dilate3(a: np.ndarray) -> np.ndarray:
        p = np.pad(a > 0, 1, mode="constant", constant_values=False)
        out = np.zeros_like(a, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]
        return (out.astype(np.uint8) * 255)

    return dilate3(erode3(bin_img))


def _connected_components_bboxes(bin_img: np.ndarray) -> list[BBox]:
    h, w = bin_img.shape[:2]
    fg = (bin_img > 0).astype(np.uint8)
    lab = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    next_label = 0
    parent = [0]
    for y in range(h):
        row = fg[y]
        for x in range(w):
            if row[x] == 0:
                continue
            left = lab[y, x - 1] if x > 0 else 0
            up = lab[y - 1, x] if y > 0 else 0
            if left == 0 and up == 0:
                next_label += 1
                if next_label >= len(parent):
                    parent.append(next_label)
                else:
                    parent[next_label] = next_label
                lab[y, x] = next_label
            elif left != 0 and up != 0:
                lab[y, x] = min(left, up)
                union(left, up)
            else:
                lab[y, x] = left or up

    bbs: dict[int, list[int]] = {}
    ys, xs = np.nonzero(lab)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r = find(int(lab[y, x]))
        if r not in bbs:
            bbs[r] = [x, y, x, y]
        bb = bbs[r]
        bb[0] = min(bb[0], x)
        bb[1] = min(bb[1], y)
        bb[2] = max(bb[2], x)
        bb[3] = max(bb[3], y)
    out: list[BBox] = []
    for _r, (x0, y0, x1, y1) in bbs.items():
        out.append(BBox(int(x0), int(y0), int(x1 + 1), int(y1 + 1)).clamp(w, h))
    return out


def detect_grid_boxes(img: Image.Image) -> list[BBox]:
    """
    Detect square-ish symbol boxes from a "legend sheet" image.
    """
    bin_img = _to_bin(img)
    h, w = bin_img.shape[:2]
    out: list[BBox] = []
    area_img = h * w
    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _hier = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < area_img * 0.001 or area > area_img * 0.12:
                continue
            if bw < 20 or bh < 20:
                continue
            ar = max(bw / max(bh, 1), bh / max(bw, 1))
            if ar > 1.35:
                continue
            out.append(BBox(x, y, x + bw, y + bh).clamp(w, h))
    else:
        # Pure numpy: edge-based boxes (more robust than raw bin CCs on this sheet style)
        gray = ImageOps.grayscale(img)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges, dtype=np.uint8)
        # keep strong edges
        thr = int(np.percentile(arr, 92))
        edge_bin = (arr >= thr).astype(np.uint8) * 255
        comps = _connected_components_bboxes(edge_bin)
        for b in comps:
            bw = b.x1 - b.x0
            bh = b.y1 - b.y0
            area = bw * bh
            if area < area_img * 0.001 or area > area_img * 0.12:
                continue
            if bw < 20 or bh < 20:
                continue
            ar = max(bw / max(bh, 1), bh / max(bw, 1))
            if ar > 1.35:
                continue
            # box borders should have relatively low edge fill inside bbox
            roi = edge_bin[b.y0 : b.y1, b.x0 : b.x1]
            dens = float(np.mean(roi > 0)) if roi.size else 1.0
            if dens > 0.25:
                continue
            out.append(b)
    out.sort(key=lambda b: (b.y0, b.x0))
    # Deduplicate near-identical boxes (simple NMS)
    kept: list[BBox] = []
    for b in out:
        ok = True
        for k in kept:
            if abs(b.x0 - k.x0) < 6 and abs(b.y0 - k.y0) < 6 and abs((b.x1 - b.x0) - (k.x1 - k.x0)) < 6:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


def ocr_text(img: Image.Image, *, lang: str = "chi_sim") -> Optional[str]:
    # Use the system tesseract CLI (no pytesseract dependency).
    g = ImageOps.grayscale(img)
    g = g.resize((g.size[0] * 2, g.size[1] * 2), resample=Image.Resampling.BILINEAR)
    # binarize
    arr = np.array(g, dtype=np.uint8)
    thr = int(np.percentile(arr, 60))
    bw = Image.fromarray((arr < thr).astype(np.uint8) * 255, mode="L")
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.png"
        out_base = Path(td) / "out"
        bw.save(in_path)
        def run_with(lang_opt: str | None) -> None:
            if lang_opt:
                cmd = ["tesseract", str(in_path), str(out_base), "-l", str(lang_opt), "--psm", "7"]
            else:
                cmd = ["tesseract", str(in_path), str(out_base), "--psm", "7"]
            subprocess.run(cmd, check=False, capture_output=True, text=True)

        # try chinese first, then english fallback
        run_with(str(lang) if lang else None)
        txt_path = out_base.with_suffix(".txt")
        txt = txt_path.read_text(encoding="utf-8", errors="ignore").strip() if txt_path.exists() else ""
        txt = " ".join(txt.split())
        if not txt:
            run_with("eng")
            txt = txt_path.read_text(encoding="utf-8", errors="ignore").strip() if txt_path.exists() else ""
            txt = " ".join(txt.split())

        if not txt:
            return None
        # Heuristic quality gate: require at least one CJK char or >=2 alphabetic chars,
        # and reject if mostly punctuation.
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in txt)
        alpha = sum(ch.isalpha() for ch in txt)
        digits = sum(ch.isdigit() for ch in txt)
        other = len(txt) - alpha - digits
        if not has_cjk and alpha < 2:
            return None
        if other > max(4, len(txt) * 0.6):
            return None
        return txt


def build_symbol_library(
    img: Image.Image,
    *,
    out_dir: Path,
    lang: str = "chi_sim",
    pad: int = 3,
    text_right_pad: int = 6,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = detect_grid_boxes(img)
    w, h = img.size
    entries: list[dict[str, Any]] = []

    for i, b in enumerate(boxes):
        # Symbol crop = inside the square with padding
        bb = BBox(b.x0 + pad, b.y0 + pad, b.x1 - pad, b.y1 - pad).clamp(w, h)
        symbol = img.crop((bb.x0, bb.y0, bb.x1, bb.y1)).convert("RGB")

        # Text crop = to the right of box, same row height, until next box or page end
        # Find next box in same row to bound text region
        row_mid = (b.y0 + b.y1) / 2
        next_x0 = w
        for nb in boxes:
            if nb.x0 <= b.x0:
                continue
            if abs(((nb.y0 + nb.y1) / 2) - row_mid) < max(12, (b.y1 - b.y0) * 0.6):
                next_x0 = min(next_x0, nb.x0)
        tx0 = b.x1 + 4
        tx1 = max(tx0 + 10, next_x0 - text_right_pad)
        text_bb = BBox(tx0, b.y0, tx1, b.y1).clamp(w, h)
        text_img = img.crop((text_bb.x0, text_bb.y0, text_bb.x1, text_bb.y1)).convert("RGB")

        txt = ocr_text(text_img, lang=lang)

        sym_path = out_dir / f"sym_{i:03d}.png"
        txt_path = out_dir / f"text_{i:03d}.png"
        symbol.save(sym_path)
        text_img.save(txt_path)

        entries.append(
            {
                "id": i,
                "name_ocr": txt,
                "symbol_bbox": b.as_list(),
                "text_bbox": text_bb.as_list(),
                "symbol_path": sym_path.name,
                "text_path": txt_path.name,
            }
        )

    meta = {
        "version": "legend_sheet.v0",
        "image_size": {"w": w, "h": h},
        "entries": entries,
        "notes": "name_ocr is best-effort; please review/correct manually.",
    }
    (out_dir / "library.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a stitch symbol library from a legend sheet image.")
    parser.add_argument("--file", "-f", required=True, help="Input legend sheet image path.")
    parser.add_argument("--out-dir", "-o", required=True, help="Output directory.")
    parser.add_argument("--lang", default="chi_sim", help="Tesseract language (default chi_sim).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    img = Image.open(args.file).convert("RGB")
    meta = build_symbol_library(img, out_dir=Path(args.out_dir), lang=str(args.lang))
    print(str(Path(args.out_dir) / "library.json"))
    print(f"entries={len(meta.get('entries', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
