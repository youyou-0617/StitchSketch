from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime
from dataclasses import replace
import html as html_lib
import re
import inspect
import time
import uuid
import zipfile
from io import BytesIO
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components

from stitchsketch.render import RenderOptions, render_svg
from stitchsketch.schema import PatternSpec, RoundSpec, RoundStitch
from stitchsketch.serialize import pattern_to_json
from stitchsketch.text_pattern import text_to_pattern
from stitchsketch.text_translate import natural_text_to_pattern, ocr_image_bytes
from stitchsketch.validate import validate_pattern


st.set_page_config(page_title="StitchSketch", layout="wide")
st.title("StitchSketch")

_APP_DIR = Path(__file__).resolve().parent
_SAVED_DIR = _APP_DIR / "saved_patterns"
_SAVED_DIR.mkdir(parents=True, exist_ok=True)

if "raw_json_next" in st.session_state:
    st.session_state["raw_json"] = st.session_state.pop("raw_json_next")

_START_MODE_LABELS = {
    "auto": "自动",
    "mr": "魔术环（MR）",
    "ch_join": "锁针成环",
    "foundation_chain": "片钩起针（锁针起针）",
}

_DEFAULT_AUTO_BUMP_WINDOW = 3
_LEGACY_AUTO_BUMP_WINDOW = 5

if not st.session_state.get("_auto_bump_window_default_migrated", False):
    for _key in ("json_auto_bump_window", "text_auto_bump_window"):
        if int(st.session_state.get(_key, _LEGACY_AUTO_BUMP_WINDOW) or _LEGACY_AUTO_BUMP_WINDOW) == _LEGACY_AUTO_BUMP_WINDOW:
            st.session_state[_key] = _DEFAULT_AUTO_BUMP_WINDOW
    st.session_state["_auto_bump_window_default_migrated"] = True

def _safe_render_options(cls: type, kwargs: dict) -> object:
    """
    Streamlit hot-reload can keep an older `RenderOptions` class object alive while
    `app.py` is updated. Passing newly-added kwargs would then crash the app.
    Filter kwargs by signature and retry after dropping unexpected keys.
    """
    try:
        params = set(inspect.signature(cls).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        filtered = dict(kwargs)

    try:
        return cls(**filtered)
    except TypeError as e:
        msg = str(e)
        m = re.search(r"unexpected keyword argument '([^']+)'", msg)
        while m and filtered:
            filtered.pop(m.group(1), None)
            try:
                return cls(**filtered)
            except TypeError as e2:
                msg = str(e2)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
        raise


def _auto_padding(size: int) -> int:
    return max(12, int(round(float(size) * 24.0 / 512.0)))


def _render_interactive_svg_preview(svg: str, *, size: int, component_key: str, height: int = 640) -> None:
    svg_data_uri = f"data:image/svg+xml;utf8,{quote(svg)}"
    dom_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", component_key)
    width_match = re.search(r'<svg[^>]*\bwidth="([0-9.]+)"', svg)
    height_match = re.search(r'<svg[^>]*\bheight="([0-9.]+)"', svg)
    svg_width = int(float(width_match.group(1))) if width_match else int(size)
    svg_height = int(float(height_match.group(1))) if height_match else int(size)
    components.html(
        f"""
<div id="{dom_id}_viewport" class="ss-preview-viewport">
  <img id="{dom_id}_image" class="ss-preview-image" src="{svg_data_uri}" draggable="false" />
  <button id="{dom_id}_reset" class="ss-preview-reset" title="重置视图" type="button">↺</button>
</div>
<style>
  .ss-preview-viewport {{
    position: relative;
    width: 100%;
    height: {height}px;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    cursor: grab;
    touch-action: none;
  }}
  .ss-preview-viewport:active {{
    cursor: grabbing;
  }}
  .ss-preview-image {{
    position: absolute;
    top: 0;
    left: 0;
    width: {svg_width}px;
    height: {svg_height}px;
    max-width: none;
    transform-origin: 0 0;
    user-select: none;
    will-change: transform;
    z-index: 1;
  }}
  .ss-preview-reset {{
    position: absolute;
    right: 12px;
    top: 12px;
    width: 34px;
    height: 34px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.92);
    color: #374151;
    font: 20px/1 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    cursor: pointer;
    z-index: 5;
  }}
  .ss-preview-reset:hover {{
    background: #f3f4f6;
  }}
</style>
<script>
(() => {{
  const viewport = document.getElementById("{dom_id}_viewport");
  const image = document.getElementById("{dom_id}_image");
  const reset = document.getElementById("{dom_id}_reset");
  const imageWidth = {svg_width};
  const imageHeight = {svg_height};
  let scale = 1;
  let x = 0;
  let y = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  function clamp(value, min, max) {{
    return Math.max(min, Math.min(max, value));
  }}

  function apply() {{
    image.style.transform = `translate(${{x}}px, ${{y}}px) scale(${{scale}})`;
  }}

  function fit() {{
    const pad = 32;
    const w = Math.max(1, viewport.clientWidth - pad * 2);
    const h = Math.max(1, viewport.clientHeight - pad * 2);
    scale = clamp(Math.min(w / imageWidth, h / imageHeight), 0.05, 2.4);
    x = (viewport.clientWidth - imageWidth * scale) / 2;
    y = (viewport.clientHeight - imageHeight * scale) / 2;
    apply();
  }}

  viewport.addEventListener("wheel", (event) => {{
    event.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const beforeX = (mx - x) / scale;
    const beforeY = (my - y) / scale;
    const factor = event.deltaY < 0 ? 1.12 : 0.88;
    scale = clamp(scale * factor, 0.08, 12);
    x = mx - beforeX * scale;
    y = my - beforeY * scale;
    apply();
  }}, {{ passive: false }});

  viewport.addEventListener("pointerdown", (event) => {{
    if (event.target === reset || reset.contains(event.target)) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    viewport.setPointerCapture(event.pointerId);
  }});

  viewport.addEventListener("pointermove", (event) => {{
    if (!dragging) return;
    x += event.clientX - lastX;
    y += event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    apply();
  }});

  viewport.addEventListener("pointerup", (event) => {{
    dragging = false;
    try {{ viewport.releasePointerCapture(event.pointerId); }} catch (_) {{}}
  }});

  viewport.addEventListener("pointercancel", () => {{
    dragging = false;
  }});

  reset.addEventListener("pointerdown", (event) => {{
    event.preventDefault();
    event.stopPropagation();
  }});
  reset.addEventListener("click", (event) => {{
    event.preventDefault();
    event.stopPropagation();
    fit();
  }});
  window.addEventListener("resize", fit);
  if (image.complete) {{
    requestAnimationFrame(fit);
  }} else {{
    image.addEventListener("load", () => requestAnimationFrame(fit), {{ once: true }});
  }}
}})();
</script>
        """,
        height=height,
        scrolling=False,
    )


def _render_pattern_text_header(copy_text: str = "") -> None:
    text_json = json.dumps(copy_text or "", ensure_ascii=False)
    st.markdown(
        """
<style>
  .ss-pattern-text-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0 0 0.35rem;
  }
  .ss-pattern-text-title {
    color: #111827;
    font-size: 1rem;
    font-weight: 600;
  }
  .ss-pattern-text-actions {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }
  .ss-copy-pattern-button {
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    background: #ffffff;
    color: #374151;
    font: 700 17px/1 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    cursor: pointer;
    padding: 0;
    user-select: none;
  }
  .ss-copy-pattern-button:hover {
    background: #f3f4f6;
  }
  .ss-rule-help {
    position: relative;
  }
  .ss-rule-help summary {
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    background: #ffffff;
    color: #374151;
    font-weight: 700;
    cursor: pointer;
    list-style: none;
    user-select: none;
  }
  .ss-rule-help summary::-webkit-details-marker {
    display: none;
  }
  .ss-rule-help summary:hover {
    background: #f3f4f6;
  }
  .ss-rule-help-panel {
    position: absolute;
    right: 0;
    top: 36px;
    z-index: 10;
    width: min(420px, calc(100vw - 80px));
    max-height: 420px;
    overflow: auto;
    padding: 0.9rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
    color: #4b5563;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .ss-rule-help-panel h4 {
    margin: 0 0 0.4rem;
    color: #111827;
    font-size: 0.95rem;
    font-weight: 700;
  }
  .ss-rule-help-panel h4 + p {
    margin-top: 0;
  }
  .ss-rule-help-panel p {
    margin: 0 0 0.8rem;
  }
  .ss-rule-help-panel p:last-child {
    margin-bottom: 0;
  }
  .ss-rule-help-panel code {
    color: #15803d;
    background: #f9fafb;
    border-radius: 4px;
    padding: 0.05rem 0.25rem;
  }
</style>
<div class="ss-pattern-text-header">
  <div class="ss-pattern-text-title">Pattern text</div>
  <div class="ss-pattern-text-actions">
    <button id="copyPatternText" class="ss-copy-pattern-button" type="button" title="复制图纸">⧉</button>
    <details class="ss-rule-help">
      <summary title="输入规则">?</summary>
      <div class="ss-rule-help-panel">
        <h4>输入规则</h4>
        <p>每行文字表示一圈；<code>(...)</code> 表示同一针目内钩织；<code>[...]</code> 表示重复组（后接 <code>*N</code> / <code>xN</code>）；<code>chsp(...)</code> 表示在上一圈锁针洞里钩织；<code>sl@3ch</code>/<code>sl@ch3</code> 表示引拔到第 3 个锁针。</p>
        <h4>符号对应</h4>
        <p><code>x/sc</code>=短针；<code>t/hdc</code>=中长针；<code>f/dc</code>=长针；<code>e/tr</code>=长长针；<code>ch</code>=锁针；<code>sl</code>/<code>sl st</code>=引拔（不增加高度，只设置落点）；<code>@Nch</code>/<code>@chN</code>=引拔到第 N 个锁针；<code>skip</code>/<code>-</code>=空针；<code>v/inc</code>=x 加针；<code>tv</code>=t 加针；<code>fv</code>=f 加针；<code>a/dec</code>=减针。</p>
      </div>
    </details>
  </div>
</div>
<script>
(() => {
  const button = document.getElementById("copyPatternText");
  const text = __TEXT_JSON__;
  if (!button) return;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "✓";
      button.title = "已复制";
      setTimeout(() => {
        button.textContent = "⧉";
        button.title = "复制图纸";
      }, 1800);
    } catch (error) {
      button.textContent = "!";
      button.title = "复制失败";
      setTimeout(() => {
        button.textContent = "⧉";
        button.title = "复制图纸";
      }, 1800);
    }
  });
})();
</script>
            """.replace("__TEXT_JSON__", text_json),
            unsafe_allow_html=True,
        )


def _round_color_key(prefix: str, round_index: int) -> str:
    return f"{prefix}_round_color_{round_index + 1}"


def _pattern_line_totals(text: str, motif_type: str) -> list[str]:
    lines = str(text or "").splitlines() or [""]
    totals = ["" for _ in lines]
    try:
        pattern = text_to_pattern(text, motif_type=motif_type, title=None)
    except Exception:
        return ["?" if line.strip() else "" for line in lines]

    round_idx = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if round_idx < len(pattern.rounds):
            total = pattern.rounds[round_idx].stitch.total
            totals[i] = str(total) if total is not None else "?"
        else:
            totals[i] = "?"
        round_idx += 1
    return totals


def _render_pattern_side_rows(values: list[str], *, title: str, height: int, align: str = "center") -> None:
    rows = "\n".join(f'<div class="ss-pattern-side-row">{html_lib.escape(str(v))}</div>' for v in values)
    st.markdown(
        f"""
<style>
  .ss-pattern-side-title {{
    color: #9ca3af;
    font-size: 0.75rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 0.35rem;
    text-align: {align};
  }}
  .ss-pattern-side-box {{
    height: {int(height)}px;
    overflow: hidden;
    color: #6b7280;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.88rem;
    line-height: 1.5rem;
    padding-top: 0.75rem;
    text-align: {align};
  }}
  .ss-pattern-side-row {{
    min-height: 1.5rem;
    border-bottom: 1px solid #eef2f7;
    white-space: nowrap;
  }}
</style>
<div class="ss-pattern-side-title">{html_lib.escape(title)}</div>
<div class="ss-pattern-side-box">{rows}</div>
        """,
        unsafe_allow_html=True,
    )


def _normalize_round_text(text: str) -> str:
    return " ".join(part.strip() for part in str(text or "").splitlines() if part.strip())


def _single_round_total(round_text: str, motif_type: str) -> str:
    line = _normalize_round_text(round_text)
    if not line:
        return ""
    try:
        pattern = text_to_pattern(line, motif_type=motif_type, title=None)
        if pattern.rounds:
            total = pattern.rounds[0].stitch.total
            return str(total) if total is not None else "?"
    except Exception:
        return "?"
    return "?"


def _render_pattern_text_editor(*, key: str, height: int, motif_type: str, value: str | None = None) -> None:
    if value is not None and key not in st.session_state:
        st.session_state[key] = value
    current = str(st.session_state.get(key, "") or "")
    _render_pattern_text_header(current)

    source_key = f"{key}_round_editor_source"
    count_key = f"{key}_round_editor_count"
    if st.session_state.get(source_key) != current:
        rounds = current.splitlines() or [""]
        st.session_state[count_key] = max(1, len(rounds))
        for i, line in enumerate(rounds):
            st.session_state[f"{key}_round_{i}"] = line
        st.session_state[source_key] = current

    round_count = max(1, int(st.session_state.get(count_key, 1) or 1))
    for i in range(round_count):
        st.session_state.setdefault(f"{key}_round_{i}", "")

    st.markdown(
        """
<style>
  .ss-round-meta {
    color: #6b7280;
    font-size: 0.88rem;
    font-weight: 700;
    padding-top: 0.68rem;
    white-space: nowrap;
  }
  .ss-round-total {
    color: #374151;
    font-size: 0.9rem;
    font-weight: 700;
    padding-top: 0.68rem;
    text-align: right;
    white-space: nowrap;
  }
  .ss-round-total span {
    color: #ef4444;
    font-size: 1rem;
  }
</style>
        """,
        unsafe_allow_html=True,
    )

    next_values: list[str] = []
    for i in range(round_count):
        row = st.columns([0.13, 0.69, 0.18], gap="small")
        with row[0]:
            st.markdown(f'<div class="ss-round-meta">第 {i + 1} 圈</div>', unsafe_allow_html=True)
        with row[1]:
            value_i = st.text_area(
                f"第 {i + 1} 圈 Pattern text",
                key=f"{key}_round_{i}",
                height=68,
                label_visibility="collapsed",
                placeholder="输入这一圈的针法",
            )
        with row[2]:
            total = _single_round_total(value_i, motif_type)
            display_total = html_lib.escape(total or "-")
            st.markdown(
                f'<div class="ss-round-total">本圈针目 <span>{display_total}</span></div>',
                unsafe_allow_html=True,
            )
        next_values.append(value_i)

    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        if st.button("新增一圈", key=f"{key}_add_round", use_container_width=True):
            st.session_state[count_key] = round_count + 1
            st.session_state[f"{key}_round_{round_count}"] = ""
            st.rerun()
    with btn_cols[1]:
        if st.button("删除末尾空圈", key=f"{key}_remove_empty_round", use_container_width=True):
            if round_count > 1 and not str(st.session_state.get(f"{key}_round_{round_count - 1}", "")).strip():
                st.session_state.pop(f"{key}_round_{round_count - 1}", None)
                st.session_state[count_key] = round_count - 1
                st.rerun()

    combined = "\n".join(_normalize_round_text(v) for v in next_values if _normalize_round_text(v))
    if combined != current:
        st.session_state[key] = combined
        st.session_state[source_key] = combined

    st.caption(
        "每个输入框对应一圈；同圈内可换行书写，系统会自动合并为同一圈。新增一圈后继续输入下一圈。"
    )


def _round_color_overrides(pattern: PatternSpec, prefix: str) -> list[str]:
    colors: list[str] = []
    for i, round_spec in enumerate(pattern.rounds):
        color = st.session_state.get(_round_color_key(prefix, i), round_spec.color)
        colors.append(str(color))
    return colors


def _parse_hex_color(color: object) -> tuple[int, int, int] | None:
    if not isinstance(color, str):
        return None
    value = color.strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return None
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _color_luma(color: object) -> float | None:
    rgb = _parse_hex_color(color)
    if rgb is None:
        return None
    r, g, b = [v / 255.0 for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_distance(a: object, b: object) -> float | None:
    rgb_a = _parse_hex_color(a)
    rgb_b = _parse_hex_color(b)
    if rgb_a is None or rgb_b is None:
        return None
    return sum(abs(x - y) for x, y in zip(rgb_a, rgb_b)) / (255.0 * 3.0)


def _reset_round_color_state(pattern: PatternSpec, prefix: str) -> None:
    st.session_state[f"{prefix}_background_color"] = pattern.background
    for i, round_spec in enumerate(pattern.rounds):
        st.session_state[_round_color_key(prefix, i)] = round_spec.color

    stale_keys = [
        str(k)
        for k in st.session_state.keys()
        if re.fullmatch(rf"{re.escape(prefix)}_round_color_\d+", str(k))
        and int(str(k).rsplit("_", 1)[1]) > len(pattern.rounds)
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)


def _repair_unreadable_color_state(pattern: PatternSpec, prefix: str) -> None:
    background = st.session_state.get(f"{prefix}_background_color", pattern.background)
    colors = _round_color_overrides(pattern, prefix)
    if not colors:
        return

    # Streamlit color pickers can keep stale values when the parsed pattern gains
    # new rounds. If a stale near-black ring color would hide the preview, restore
    # that ring to the pattern palette default.
    repaired = False
    for i, round_spec in enumerate(pattern.rounds):
        key = _round_color_key(prefix, i)
        current = st.session_state.get(key)
        if current is None:
            continue
        luma = _color_luma(current)
        default_luma = _color_luma(round_spec.color)
        if luma is not None and default_luma is not None and luma < 0.08 and default_luma >= 0.12:
            st.session_state[key] = round_spec.color
            repaired = True
    if repaired:
        colors = _round_color_overrides(pattern, prefix)

    distances = [_color_distance(background, color) for color in colors]
    lumas = [_color_luma(color) for color in colors]
    background_luma = _color_luma(background)
    if any(v is None for v in distances) or any(v is None for v in lumas) or background_luma is None:
        return

    max_distance = max(float(v) for v in distances if v is not None)
    max_luma = max(float(v) for v in lumas if v is not None)
    min_luma = min(float(v) for v in lumas if v is not None)
    all_dark = background_luma < 0.08 and max_luma < 0.12
    all_light = background_luma > 0.92 and min_luma > 0.88
    if max_distance < 0.08 or all_dark or all_light:
        _reset_round_color_state(pattern, prefix)


def _apply_round_color_overrides(pattern: PatternSpec, prefix: str) -> PatternSpec:
    rounds = [
        replace(round_spec, color=color)
        for round_spec, color in zip(pattern.rounds, _round_color_overrides(pattern, prefix))
    ]
    background = str(st.session_state.get(f"{prefix}_background_color", pattern.background))
    return replace(pattern, rounds=rounds, background=background)


def _svg_to_png_bytes(svg: str, *, output_width: int = 1024) -> bytes:
    try:
        import cairosvg  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少 cairosvg。请在运行 StitchSketch 的环境中执行：pip install cairosvg") from exc
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=output_width)


def _realization_control_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    control_opts = dict(opts_kwargs)
    control_opts.update(
        {
            "stroke": "#111827",
            "stroke_width": 2,
            "glyph_stroke": "#111827",
            "glyph_opacity": 0.0,
            "glyph_width": 1,
            "max_glyphs_per_round": 120,
            "show_seam_marker": False,
            "chain_render_style": "beads",
            "chain_width_scale": 1.35,
        }
    )
    control_pattern = replace(pattern, background="#ffffff")
    return render_svg(control_pattern, _safe_render_options(RenderOptions, control_opts))


def _realization_strict_shape_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    strict_opts = dict(opts_kwargs)
    strict_opts.update(
        {
            "stroke": "#111827",
            "stroke_width": 2,
            "texture": False,
            "glyph_stroke": "#111827",
            "glyph_opacity": 0.0,
            "glyph_width": 1,
            "max_glyphs_per_round": 0,
            "show_seam_marker": False,
            "chain_render_style": "stroke",
            "chain_width_scale": 1.35,
        }
    )
    return render_svg(pattern, _safe_render_options(RenderOptions, strict_opts))


def _realization_stitch_units_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    unit_opts = dict(opts_kwargs)
    unit_opts.update(
        {
            "stroke": "#111827",
            "stroke_width": 1,
            "texture": True,
            "glyph_stroke": "#0f172a",
            "glyph_opacity": 0.95,
            "glyph_width": 2,
            "max_glyphs_per_round": 2000,
            "glyph_full_height": True,
            "glyph_angle_mode": "by_produced",
            "glyph_style": "strokes",
            "show_seam_marker": False,
            "chain_render_style": "beads",
            "chain_width_scale": 1.55,
        }
    )
    rounds = [replace(round_spec, color="#f8fafc") for round_spec in pattern.rounds]
    unit_pattern = replace(pattern, rounds=rounds, background="#ffffff")
    return render_svg(unit_pattern, _safe_render_options(RenderOptions, unit_opts))


def _realization_stitch_primitives_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    primitive_opts = dict(opts_kwargs)
    primitive_opts.update(
        {
            "stroke": "#94a3b8",
            "stroke_width": 1,
            "texture": True,
            "glyph_stroke": "#2563eb",
            "glyph_opacity": 0.72,
            "glyph_width": 5,
            "max_glyphs_per_round": 2000,
            "glyph_full_height": False,
            "glyph_angle_mode": "by_produced",
            "glyph_style": "metric_strokes",
            "show_seam_marker": False,
            "chain_render_style": "beads",
            "chain_width_scale": 1.65,
        }
    )
    rounds = [replace(round_spec, color="#ffffff") for round_spec in pattern.rounds]
    primitive_pattern = replace(pattern, rounds=rounds, background="#ffffff")
    return render_svg(primitive_pattern, _safe_render_options(RenderOptions, primitive_opts))


def _annotation_symbols_from_ops(ops: object) -> list[str]:
    out: list[str] = []

    def add(sym: str, n: int = 1) -> None:
        for _ in range(max(0, n)):
            out.append(sym)

    def walk(op_list: object) -> None:
        if not isinstance(op_list, list):
            return
        for op in op_list:
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch":
                add(str(op.get("st", "")), max(0, int(op.get("n", 1))))
            elif t == "inc":
                base = str(op.get("base") or "x")
                add(base, max(0, int(op.get("n", 1))) * 2)
            elif t == "dec":
                base = str(op.get("base") or "x")
                add(base, max(0, int(op.get("n", 1))))
            elif t in {"cluster", "ch_space"}:
                walk(op.get("ops", []))
            elif t == "repeat":
                inner = op.get("ops", [])
                for _ in range(max(0, int(op.get("times", 1)))):
                    walk(inner)

    walk(ops)
    return out


def _annotation_polar(cx: float, cy: float, r: float, ang: float) -> tuple[float, float]:
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def _annotation_sector_path(cx: float, cy: float, r_inner: float, r_outer: float, start: float, sweep: float) -> str:
    if sweep <= 0 or r_outer <= 0:
        return ""
    end = start + sweep
    large = 1 if sweep > math.pi else 0
    x0, y0 = _annotation_polar(cx, cy, r_outer, start)
    x1, y1 = _annotation_polar(cx, cy, r_outer, end)
    if r_inner <= 0:
        return (
            f"M {cx:.3f} {cy:.3f} L {x0:.3f} {y0:.3f} "
            f"A {r_outer:.3f} {r_outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} Z"
        )
    xi1, yi1 = _annotation_polar(cx, cy, r_inner, end)
    xi0, yi0 = _annotation_polar(cx, cy, r_inner, start)
    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {r_outer:.3f} {r_outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} "
        f"L {xi1:.3f} {yi1:.3f} "
        f"A {r_inner:.3f} {r_inner:.3f} 0 {large} 0 {xi0:.3f} {yi0:.3f} Z"
    )


def _annotation_chart_symbol_svg(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    ang: float,
    sym: str,
    sweep: float,
) -> str:
    normalized = {"sc": "x", "hdc": "t", "dc": "f", "tr": "e"}.get(sym, sym)
    if normalized not in {"x", "t", "f", "e"}:
        return ""

    band = max(1.0, r_outer - r_inner)
    r_mid = (r_inner + r_outer) / 2
    arc_w = max(1.0, r_mid * max(0.02, sweep))
    ca = math.cos(ang)
    sa = math.sin(ang)
    # Radial and tangential unit vectors.
    ux, uy = ca, sa
    tx, ty = -sa, ca
    stroke = "#111827"
    sw = max(1.8, min(3.4, band * 0.08))
    op = 0.92

    def line(x0: float, y0: float, x1: float, y1: float) -> str:
        return (
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
            f'stroke="{stroke}" stroke-width="{sw:.3f}" stroke-linecap="round" opacity="{op:.3f}"/>\n'
        )

    if normalized == "x":
        # Common chart symbol for single crochet: an X that occupies most of
        # the stitch cell, so the AI reads one marked cell as one stitch.
        half_t = min(22.0, max(5.0, min(arc_w, band) * 0.36))
        half_r = min(band * 0.42, max(5.0, band * 0.34))
        mx, my = _annotation_polar(cx, cy, r_mid, ang)
        p1 = (mx - tx * half_t - ux * half_r, my - ty * half_t - uy * half_r)
        p2 = (mx + tx * half_t + ux * half_r, my + ty * half_t + uy * half_r)
        p3 = (mx + tx * half_t - ux * half_r, my + ty * half_t - uy * half_r)
        p4 = (mx - tx * half_t + ux * half_r, my - ty * half_t + uy * half_r)
        return line(*p1, *p2) + line(*p3, *p4)

    # hdc/dc/tr: a post with a top bar; dc/tr add one/two yarn-over slashes.
    edge_gap = max(1.5, min(4.0, band * 0.045))
    r0 = r_inner + edge_gap
    r1 = r_outer - edge_gap
    if r1 <= r0:
        r0 = r_inner + band * 0.04
        r1 = r_outer - band * 0.04
    x0, y0 = _annotation_polar(cx, cy, r0, ang)
    x1, y1 = _annotation_polar(cx, cy, r1, ang)
    out = [line(x0, y0, x1, y1)]

    top_len = min(24.0, max(7.0, min(arc_w, band) * 0.42))
    out.append(line(x1 - tx * top_len, y1 - ty * top_len, x1 + tx * top_len, y1 + ty * top_len))

    bars = {"t": 0, "f": 1, "e": 2}[normalized]
    slash_len = min(18.0, max(5.0, min(arc_w, band) * 0.30))
    for b in range(bars):
        frac = 0.58 + (0.16 * b)
        xb = x0 + (x1 - x0) * frac
        yb = y0 + (y1 - y0) * frac
        # Diagonal yarn-over slash across the post.
        sx = (tx * 0.78) - (ux * 0.34)
        sy = (ty * 0.78) - (uy * 0.34)
        out.append(line(xb - sx * slash_len, yb - sy * slash_len, xb + sx * slash_len, yb + sy * slash_len))
    return "".join(out)


def _realization_annotated_stitch_map_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    size = int(opts_kwargs.get("size", 512) or 512)
    padding = float(opts_kwargs.get("padding", 24) or 24)
    rounds = max(1, len(pattern.rounds))
    raw_max_r = max(16.0, (size / 2) - padding)
    reserve = 0.0
    if bool(opts_kwargs.get("circle_auto_bump", False)) and raw_max_r > 0:
        reserve = min(72.0, max(18.0, raw_max_r * 0.22))
        reserve = max(0.0, min(reserve, raw_max_r - 8.0))
    max_r = max(16.0, raw_max_r - reserve)
    inner_r = max(10.0, max_r * 0.16)
    band = max(18.0, (max_r - inner_r) / rounds)
    start = (-math.pi / 2) + (math.pi * float(opts_kwargs.get("rotation_deg", 0.0) or 0.0) / 180.0)
    label_map = {"sl st": "sl", "ss": "sl", "skip": "-", "-": "-", "inc": "v"}

    # Use the actual StitchSketch preview shape as the base. The annotation layer
    # should explain stitch allocation without replacing the true contour.
    base_opts = dict(opts_kwargs)
    base_opts.update(
        {
            "stroke": "#111827",
            "stroke_width": 2,
            "texture": False,
            "glyph_stroke": "#111827",
            "glyph_opacity": 0.0,
            "glyph_width": 1,
            "max_glyphs_per_round": 0,
            "show_seam_marker": False,
            "chain_render_style": "stroke",
            "chain_width_scale": 1.35,
        }
    )
    base_svg = render_svg(pattern, _safe_render_options(RenderOptions, base_opts))
    canvas_w = size
    canvas_h = size
    viewbox_match = re.search(r'viewBox="[^"]*\s+[^"]*\s+([0-9.]+)\s+([0-9.]+)"', base_svg)
    if viewbox_match:
        canvas_w = int(float(viewbox_match.group(1)))
        canvas_h = int(float(viewbox_match.group(2)))
    else:
        width_match = re.search(r'width="([0-9.]+)"', base_svg)
        height_match = re.search(r'height="([0-9.]+)"', base_svg)
        if width_match:
            canvas_w = int(float(width_match.group(1)))
        if height_match:
            canvas_h = int(float(height_match.group(1)))
    cx = canvas_w / 2
    cy = canvas_h / 2
    if "</svg>" in base_svg:
        parts = [base_svg.rsplit("</svg>", 1)[0], "\n"]
    else:
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>\n',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n',
            '<rect width="100%" height="100%" fill="#ffffff"/>\n',
        ]
    parts.append(
        '<style>'
        '.ann-label{font-family:Arial,Helvetica,sans-serif;font-weight:800;dominant-baseline:middle;text-anchor:middle}'
        '.ann-note{font-family:Arial,Helvetica,sans-serif;font-weight:800}'
        '</style>\n'
    )

    for i, round_spec in enumerate(pattern.rounds):
        symbols = _annotation_symbols_from_ops(round_spec.stitch.ops)
        cells: list[str] = []
        for raw_sym in symbols:
            sym = str(raw_sym)
            if sym in {"sl", "sl st", "ss", "ch", "skip", "-"}:
                continue
            cells.append(sym)
        if not cells:
            continue
        count = len(cells)
        r0 = inner_r + (i * band)
        r1 = inner_r + ((i + 1) * band)
        sweep = (2 * math.pi) / count
        text_size = max(10.0, min(22.0, band * 0.42))

        for v_idx, raw_sym in enumerate(cells):
            sym = str(raw_sym)
            short = label_map.get(sym, sym)
            a0 = start + (v_idx * sweep)
            line_inset = max(4.0, (r1 - r0) * 0.14)
            x0, y0 = _annotation_polar(cx, cy, r0 + line_inset, a0)
            x1, y1 = _annotation_polar(cx, cy, r1 - line_inset, a0)
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
                'stroke="#111827" stroke-width="1.15" stroke-opacity="0.72"/>\n'
            )
            if sym in {"skip", "-"}:
                continue
            a_mid = a0 + (sweep / 2)
            chart_symbol = _annotation_chart_symbol_svg(cx, cy, r0, r1, a_mid, sym, sweep)
            if chart_symbol:
                parts.append(chart_symbol)
            else:
                tx, ty = _annotation_polar(cx, cy, (r0 + r1) / 2, a_mid)
                parts.append(
                    f'<text class="ann-label" x="{tx:.3f}" y="{ty:.3f}" fill="#111827" font-size="{text_size:.1f}" '
                    f'transform="rotate({math.degrees(a_mid) + 90:.2f} {tx:.3f} {ty:.3f})">{short}</text>\n'
                )
        # Close the final divider; this makes the stitch count explicit without
        # redrawing the contour as a filled circle.
        a_end = start + (count * sweep)
        line_inset = max(4.0, (r1 - r0) * 0.14)
        x0, y0 = _annotation_polar(cx, cy, r0 + line_inset, a_end)
        x1, y1 = _annotation_polar(cx, cy, r1 - line_inset, a_end)
        parts.append(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
            'stroke="#111827" stroke-width="1.15" stroke-opacity="0.72"/>\n'
        )

    parts.append("</svg>\n")
    return "".join(parts)


def _realization_color_palette_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    size = int(opts_kwargs.get("size", 512) or 512)
    swatch_w = max(80, int(size * 0.34))
    swatch_h = 54
    gap = 16
    rows = 1 + len(pattern.rounds)
    height = max(size // 2, 40 + rows * (swatch_h + gap))
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{height}" viewBox="0 0 {size} {height}">\n',
        '<rect width="100%" height="100%" fill="#ffffff"/>\n',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-weight:700;dominant-baseline:middle}</style>\n',
        '<text x="32" y="30" fill="#111827" font-size="20">Color palette</text>\n',
    ]
    entries = [("background", str(pattern.background))] + [
        (f"R{i}", str(round_spec.color)) for i, round_spec in enumerate(pattern.rounds, start=1)
    ]
    y = 54
    for label, color in entries:
        parts.append(
            f'<rect x="32" y="{y}" width="{swatch_w}" height="{swatch_h}" rx="8" fill="{color}" stroke="#111827" stroke-width="1.5"/>\n'
        )
        parts.append(f'<text x="{48 + swatch_w}" y="{y + swatch_h / 2:.1f}" fill="#111827" font-size="17">{label}: {color}</text>\n')
        y += swatch_h + gap
    parts.append("</svg>\n")
    return "".join(parts)


def _realization_color_reference_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    ref_opts = dict(opts_kwargs)
    ref_opts.update(
        {
            "stroke": "#111827",
            "stroke_width": 2,
            "glyph_stroke": "#111827",
            "glyph_opacity": 0.18,
            "glyph_width": 1,
            "max_glyphs_per_round": 120,
            "show_seam_marker": False,
            "chain_render_style": "beads",
            "chain_width_scale": 1.35,
        }
    )
    return render_svg(pattern, _safe_render_options(RenderOptions, ref_opts))


def _realization_height_source_svg(pattern: PatternSpec, opts_kwargs: dict) -> str:
    height_opts = dict(opts_kwargs)
    height_opts.update(
        {
            "stroke": "#404040",
            "stroke_width": 2,
            "glyph_stroke": "#404040",
            "glyph_opacity": 0.0,
            "glyph_width": 1,
            "max_glyphs_per_round": 120,
            "show_seam_marker": False,
            "chain_render_style": "beads",
            "chain_width_scale": 1.35,
        }
    )
    rounds = [replace(round_spec, color="#ffffff") for round_spec in pattern.rounds]
    height_pattern = replace(pattern, rounds=rounds, background="#000000")
    return render_svg(height_pattern, _safe_render_options(RenderOptions, height_opts))


def _height_map_from_source_png(source_png: bytes) -> bytes:
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageStat
    except Exception as exc:
        raise RuntimeError("缺少 Pillow，无法生成高度图。请安装 pillow。") from exc

    with Image.open(BytesIO(source_png)) as im:
        gray = im.convert("L")

    sample = gray.crop((0, 0, gray.size[0], max(1, min(16, gray.size[1]))))
    background_is_dark = ImageStat.Stat(sample).mean[0] < 128
    if background_is_dark:
        stitch_mask = gray.point(lambda v: 255 if v > 24 else 0)
        drawn_edge_mask = gray.point(lambda v: 255 if 24 < v < 180 else 0)
    else:
        white = Image.new("L", gray.size, 255)
        diff = ImageChops.difference(gray, white)
        stitch_mask = diff.point(lambda v: 255 if v > 8 else 0)
        drawn_edge_mask = Image.new("L", gray.size, 0)

    eroded = stitch_mask.filter(ImageFilter.MinFilter(5))
    dilated = stitch_mask.filter(ImageFilter.MaxFilter(5))
    boundary_mask = ImageChops.difference(dilated, eroded).point(lambda v: 255 if v > 0 else 0)
    edge_mask = ImageChops.lighter(drawn_edge_mask, boundary_mask)

    height = Image.new("L", gray.size, 0)
    height.paste(220, mask=stitch_mask)
    height.paste(70, mask=edge_mask)

    out = BytesIO()
    height.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _mask_from_height_source_png(source_png: bytes) -> bytes:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("缺少 Pillow，无法生成 mask。请安装 pillow。") from exc

    with Image.open(BytesIO(source_png)) as im:
        gray = im.convert("L")
    mask = gray.point(lambda v: 255 if v > 24 else 0)
    out = BytesIO()
    mask.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _symbol_counts_from_ops(ops: object) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(sym: str, n: int) -> None:
        if n <= 0:
            return
        counts[sym] = counts.get(sym, 0) + n

    def walk(op_list: object, multiplier: int = 1) -> None:
        if not isinstance(op_list, list):
            return
        for op in op_list:
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch":
                add(str(op.get("st", "")), multiplier * max(0, int(op.get("n", 1))))
            elif t == "inc":
                add(str(op.get("base") or "x"), multiplier * max(0, int(op.get("n", 1))) * 2)
            elif t == "dec":
                add(str(op.get("base") or "x"), multiplier * max(0, int(op.get("n", 1))))
            elif t in {"cluster", "ch_space"}:
                walk(op.get("ops", []), multiplier)
            elif t == "repeat":
                walk(op.get("ops", []), multiplier * max(0, int(op.get("times", 1))))

    walk(ops)
    return counts


def _flatten_ops_for_manifest(ops: object, *, limit: int = 240) -> list[str]:
    tokens: list[str] = []

    def add(token: str, n: int = 1) -> None:
        for _ in range(max(0, n)):
            if len(tokens) >= limit:
                return
            tokens.append(token)

    def walk(op_list: object, multiplier: int = 1) -> None:
        if not isinstance(op_list, list):
            return
        for op in op_list:
            if len(tokens) >= limit:
                return
            if not isinstance(op, dict):
                continue
            t = op.get("type")
            if t == "stitch":
                add(str(op.get("st", "")), multiplier * max(0, int(op.get("n", 1))))
            elif t == "inc":
                base = str(op.get("base") or "x")
                for _ in range(multiplier * max(0, int(op.get("n", 1)))):
                    add(f"{base}+{base}@same_slot")
            elif t == "dec":
                base = str(op.get("base") or "x")
                add(f"{base}_dec", multiplier * max(0, int(op.get("n", 1))))
            elif t == "cluster":
                inner = _flatten_ops_for_manifest(op.get("ops", []), limit=limit)
                add("(" + ",".join(inner) + ")@same_slot", multiplier)
            elif t == "ch_space":
                inner = _flatten_ops_for_manifest(op.get("ops", []), limit=limit)
                add("chsp(" + ",".join(inner) + ")", multiplier)
            elif t == "repeat":
                walk(op.get("ops", []), multiplier * max(0, int(op.get("times", 1))))

    walk(ops)
    if len(tokens) >= limit:
        tokens.append("...")
    return tokens


def _realization_stitch_drawing_guide_text() -> str:
    return (
        "StitchSketch stitch drawing guide for realism rendering.\n"
        "\n"
        "Use this as a visual dictionary. Keep the structure from the reference images, then convert each symbol into yarn texture.\n"
        "\n"
        "Symbol meanings and visual rules:\n"
        "- x / sc: single crochet. Short, dense crochet stitch. Low height. Looks compact and rounded.\n"
        "- t / hdc: half double crochet. Medium height, taller than x and shorter than f.\n"
        "- f / dc: double crochet. Tall radial stitch. It should look like a longer raised yarn column or rib.\n"
        "- e / tr: treble crochet. Extra tall stitch, visibly longer than f.\n"
        "- ch: chain stitch. A chain/loop/bridge made of rounded yarn segments. It often creates an open hole; do not fill the hole.\n"
        "- sl / sl st: slip stitch. It is a join/landing instruction, not a visible new stitch. Do not draw it as an extra raised stitch.\n"
        "- @Nch / @chN: slip-stitch target. Move the landing point to the N-th chain stitch; do not draw this as yarn.\n"
        "- skip / -: empty previous stitch slot. Leave that slot open; do not render a stitch there.\n"
        "- v / inc: increase. Two x stitches worked into the same previous slot.\n"
        "- tv: two t stitches worked into the same previous slot.\n"
        "- fv: two f stitches worked into the same previous slot.\n"
        "- a / dec: decrease. Two previous slots are merged into one visible stitch.\n"
        "- (...): multiple stitches worked into the same previous slot. Draw them as a local fan/cluster from one base.\n"
        "- chsp(...): stitches worked into a previous chain-space hole. Distribute these stitches along that chain-space; do not start every stitch from the exact center.\n"
        "\n"
        "Important rendering constraints:\n"
        "- The number of visible stitch groups must match structure_manifest.txt and round_stitch_counts.txt.\n"
        "- Do not invent extra loops, petals, rings, or filled areas.\n"
        "- Do not render black outlines, radial guide lines, start markers, labels, or seam guides as yarn.\n"
        "- Use annotated_stitch_map.png as the human-readable stitch-position map. It divides each round into exact stitch cells and labels the stitch type in each cell.\n"
        "- In annotated_stitch_map.png, red sl/no height labels are join instructions only. They must not become raised yarn stitches.\n"
        "- Use stitch_units_reference.png only as a count/slot guide. Its dark dividers mark individual stitch units; they must not appear as black yarn in the final image.\n"
        "- Use optional_count_only_stitch_primitives_reference.png only as a count guide. It is NOT a silhouette guide, NOT a hole guide, and NOT a final-shape guide. One rounded capsule/post equals one stitch, but strict_shape_reference.png always controls the final shape.\n"
    )


def _round_stitch_code(round_spec: RoundSpec) -> str:
    stitch = getattr(round_spec, "stitch", None)
    code = getattr(stitch, "stitch", None)
    return str(code or "")


def _realization_structure_manifest_text(pattern: PatternSpec, pattern_text: str) -> str:
    lines = [
        "StitchSketch structure manifest for realism rendering.",
        "",
        "Hard constraints:",
        "- Preserve the outer silhouette, holes, ring layout, and color regions from reference_color.png / strict_shape_reference.png.",
        "- Preserve the round count and visible group count. Do not invent extra petals, loops, or rings.",
        "- Do not render construction aids: radial start/seam markers, center guide lines, black outlines, labels, or diagram strokes must disappear in the realistic photo.",
        "- Convert flat regions into crochet texture, but keep the geometry from the reference images.",
        "",
        "Rounds:",
    ]
    for i, round_spec in enumerate(pattern.rounds, start=1):
        counts = _symbol_counts_from_ops(round_spec.stitch.ops)
        count_text = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) if counts else "none"
        total = round_spec.stitch.total if round_spec.stitch.total is not None else "unknown"
        consumed = round_spec.stitch.consumed if round_spec.stitch.consumed is not None else "unknown"
        lines.append(
            f"R{i}: produced={total}, consumed_previous_slots={consumed}, color={round_spec.color}, stitch_counts={count_text}, code={_round_stitch_code(round_spec)}"
        )
    lines.extend(["", "Pattern code:", pattern_text.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _realization_round_stitch_counts_text(pattern: PatternSpec) -> str:
    lines = [
        "Round stitch counts and expanded stitch order.",
        "Use this file as exact count guidance. Keep the visible stitch count and grouping for each round.",
        "",
    ]
    for i, round_spec in enumerate(pattern.rounds, start=1):
        counts = _symbol_counts_from_ops(round_spec.stitch.ops)
        count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "none"
        sequence = " ".join(_flatten_ops_for_manifest(round_spec.stitch.ops))
        produced = round_spec.stitch.total if round_spec.stitch.total is not None else "unknown"
        consumed = round_spec.stitch.consumed if round_spec.stitch.consumed is not None else "unknown"
        lines.extend(
            [
                f"R{i}",
                f"- code: {_round_stitch_code(round_spec)}",
                f"- produced visible units: {produced}",
                f"- consumed previous slots: {consumed}",
                f"- symbol counts: {count_text}",
                f"- expanded order: {sequence or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _realization_prompt_text(background_color: str) -> str:
    background_color = background_color.strip() or "the reference background color"
    return (
        "Task: turn the supplied annotated crochet structure diagram into a realistic top-down crochet photo.\n"
        "\n"
        "Use only these references:\n"
        "1. annotated_stitch_map.png is the main structure reference. It defines the final outline, round layout, stitch cells, and stitch chart symbols.\n"
        "2. color_palette.png defines the background color and the yarn color for each round.\n"
        "3. prompt.txt / README_FIRST.txt defines how to interpret the labels.\n"
        "\n"
        "Treat annotated_stitch_map.png as the exact geometric template for the final result. The final crochet photo must keep the same silhouette, holes, ring boundaries, cell positions, chain-loop positions, and overall proportions. Do not reinterpret, redraw, simplify, round off, or rearrange the structure.\n"
        "Only change the visual material: replace the flat diagram fills, labels, and guide lines with realistic yarn texture while preserving the same geometry underneath.\n"
        "Each black cell boundary separates ordinary stitch positions. Each chart symbol or fallback text label inside a cell names the stitch that should be rendered in that cell. Unlabeled chain loops or chain bridges visible in the structure map are ch stitches; preserve their shape but do not subdivide them into labeled cells.\n"
        "Chart symbols are drawn large inside their cells on purpose: each full-height symbol corresponds to exactly one crochet stitch occupying that cell from its inner edge to its outer edge.\n"
        "Render exactly one visible crochet stitch for each marked cell, in the same cell. Do not create multiple small stitches inside one marked cell, and do not merge adjacent cells into one stitch.\n"
        "Interpret chart symbols as follows: X-shaped symbol means x/sc short single crochet; T-shaped post means t/hdc medium height; T-shaped post with one diagonal slash means f/dc tall; T-shaped post with two diagonal slashes means e/tr extra tall. Ch stitches are shown as unlabeled rounded chain loops or bridges. Slip-stitch joins (sl) are omitted from the diagram because they are landing instructions only; do not add raised stitches for them.\n"
        "Use color_palette.png exactly: keep the plain background color and use the listed round colors for the corresponding yarn regions.\n"
        "\n"
        "Prompt: realistic top-down photograph of a handmade crochet piece, 4-ply milk cotton yarn, soft cotton yarn, visible yarn fibers, raised crochet stitches, plush handcrafted texture, soft natural shadow, plain background matching the background swatch in color_palette.png. Use annotated_stitch_map.png as a strict shape-and-position template. Preserve the exact outline, holes, chain loops, round layout, stitch count, and one-stitch-per-cell placement. Remove all diagram chart symbols, text labels, and black cell lines in the final photo; replace only their visual style with realistic yarn stitches in the same positions.\n"
        "\n"
        "Negative prompt: changed structure, changed outline, extra stitches, missing stitches, multiple stitches inside one marked cell, wrong stitch count, filled holes, moved holes, text labels, chart symbols, visible diagram lines, black outlines, seam guide, center guide line, radial start marker, flat vector, cartoon, watermark, hands.\n"
    )


def _realization_reference_package_bytes(
    pattern: PatternSpec,
    opts_kwargs: dict,
    *,
    pattern_text: str,
    pattern_json: str,
    output_width: int,
) -> bytes:
    annotated_stitch_map_svg = _realization_annotated_stitch_map_svg(pattern, opts_kwargs)
    color_palette_svg = _realization_color_palette_svg(pattern, opts_kwargs)
    annotated_stitch_map_png = _svg_to_png_bytes(annotated_stitch_map_svg, output_width=output_width)
    color_palette_png = _svg_to_png_bytes(color_palette_svg, output_width=output_width)

    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_FIRST.txt", _realization_prompt_text(str(pattern.background)))
        zf.writestr("annotated_stitch_map.png", annotated_stitch_map_png)
        zf.writestr("color_palette.png", color_palette_png)
        zf.writestr("prompt.txt", _realization_prompt_text(str(pattern.background)))
    return out.getvalue()


def _comfyui_upload_image(server_url: str, png_bytes: bytes, *, prefix: str = "stitchsketch_control") -> str:
    url = server_url.rstrip("/")
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    resp = requests.post(
        f"{url}/upload/image",
        files={"image": (filename, png_bytes, "image/png")},
        data={"overwrite": "true", "type": "input"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("name") or filename)


def _comfyui_controlnet_workflow(
    *,
    image_name: str,
    checkpoint: str,
    controlnet: str,
    lora_name: str,
    lora_strength: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    control_strength: float,
    width: int,
    height: int,
) -> dict:
    width = max(64, int(width) // 8 * 8)
    height = max(64, int(height) // 8 * 8)
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora_name,
                "strength_model": float(lora_strength),
                "strength_clip": 0.0,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 1], "text": prompt},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 1], "text": negative_prompt},
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "6": {
            "class_type": "Canny",
            "inputs": {"image": ["5", 0], "low_threshold": 0.12, "high_threshold": 0.48},
        },
        "7": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet},
        },
        "8": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "control_net": ["7", 0],
                "image": ["6", 0],
                "strength": float(control_strength),
                "start_percent": 0.0,
                "end_percent": 0.85,
            },
        },
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["9", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "StitchSketch_realized"},
        },
    }


def _comfyui_queue_prompt(server_url: str, workflow: dict) -> str:
    url = server_url.rstrip("/")
    client_id = uuid.uuid4().hex
    resp = requests.post(f"{url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    resp.raise_for_status()
    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI 没有返回 prompt_id：{resp.text[:500]}")
    return str(prompt_id)


def _comfyui_wait_for_image(server_url: str, prompt_id: str, *, timeout_s: int = 900) -> bytes:
    url = server_url.rstrip("/")
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        resp = requests.get(f"{url}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        history = resp.json().get(prompt_id)
        if history:
            status = history.get("status") or {}
            messages = status.get("messages") or []
            last_error = json.dumps(messages[-2:], ensure_ascii=False) if messages else last_error
            outputs = history.get("outputs") or {}
            for node_out in outputs.values():
                for img in node_out.get("images", []) if isinstance(node_out, dict) else []:
                    params = {
                        "filename": img.get("filename"),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                    view = requests.get(f"{url}/view", params=params, timeout=60)
                    view.raise_for_status()
                    return view.content
        time.sleep(1.5)
    raise TimeoutError(f"ComfyUI 生成超时。最后状态：{last_error}")


def _realize_with_comfyui(
    svg: str,
    *,
    server_url: str,
    checkpoint: str,
    controlnet: str,
    lora_name: str,
    lora_strength: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    control_strength: float,
    control_width: int,
) -> tuple[bytes, bytes]:
    png = _svg_to_png_bytes(svg, output_width=control_width)
    image_name = _comfyui_upload_image(server_url, png)
    workflow = _comfyui_controlnet_workflow(
        image_name=image_name,
        checkpoint=checkpoint,
        controlnet=controlnet,
        lora_name=lora_name,
        lora_strength=lora_strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        control_strength=control_strength,
        width=control_width,
        height=control_width,
    )
    prompt_id = _comfyui_queue_prompt(server_url, workflow)
    return png, _comfyui_wait_for_image(server_url, prompt_id)


def _default_segments(prev_total: int, k: int = 8) -> list[dict]:
    k = max(2, min(16, k))
    if prev_total <= 0:
        return [{"length": 1, "bump": 0.0, "label": ""} for _ in range(k)]
    base = max(1, prev_total // k)
    segs = [{"length": base, "bump": 0.0, "label": ""} for _ in range(k)]
    total = sum(int(s["length"]) for s in segs)
    segs[-1]["length"] = max(1, int(segs[-1]["length"]) + (prev_total - total))
    return segs


def _slugify(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff _-]+", "", name)
    return name.strip().replace(" ", "_")[:60]


def _list_saved() -> list[Path]:
    return sorted(_SAVED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _read_saved(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_saved(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_save_payload(*, name: str, now_iso: str) -> dict:
    return {
        "name": name.strip(),
        "saved_at": now_iso,
        "raw_json": st.session_state.get("raw_json", ""),
        "text_pattern": st.session_state.get("text_pattern", ""),
        "settings": {
            "json_start_mode": st.session_state.get("json_start_mode", "auto"),
            "json_hole_px": st.session_state.get("json_hole_px", 0.0),
            "json_round1_prev_total": st.session_state.get("json_round1_prev_total", 0),
            "json_rotation_deg": st.session_state.get("json_rotation_deg", 0.0),
            "json_glyph_angle_mode": st.session_state.get("json_glyph_angle_mode", "by_structure"),
            "json_glyph_style": st.session_state.get("json_glyph_style", "strokes"),
            # outline is always auto-bump now (no segmented outline UI)
            "json_auto_bump_scope": st.session_state.get("json_auto_bump_scope", "all"),
            "json_auto_bump_window": st.session_state.get("json_auto_bump_window", _DEFAULT_AUTO_BUMP_WINDOW),
            "json_auto_bump_strength": 1.0,
            "text_start_mode": st.session_state.get("text_start_mode", "auto"),
            "text_hole_px": st.session_state.get("text_hole_px", 0.0),
            "text_round1_prev_total": st.session_state.get("text_round1_prev_total", 0),
            "text_rotation_deg": st.session_state.get("text_rotation_deg", 0.0),
            "text_glyph_angle_mode": st.session_state.get("text_glyph_angle_mode", "by_structure"),
            "text_glyph_style": st.session_state.get("text_glyph_style", "metric_strokes"),
            # outline is always auto-bump now (no segmented outline UI)
            "text_auto_bump_scope": st.session_state.get("text_auto_bump_scope", "all"),
            "text_auto_bump_window": st.session_state.get("text_auto_bump_window", _DEFAULT_AUTO_BUMP_WINDOW),
            "text_auto_bump_smoothing": st.session_state.get("text_auto_bump_smoothing", "weighted"),
            "text_auto_bump_strength": 1.0,
            "text_motif_type": st.session_state.get("text_motif_type", "circle_coaster"),
            "text_background_color": st.session_state.get("text_background_color", "#ffffff"),
            "text_round_colors": [
                str(st.session_state[k])
                for k in sorted(
                    [str(k) for k in st.session_state.keys() if re.fullmatch(r"text_round_color_\d+", str(k))],
                    key=lambda key: int(key.rsplit("_", 1)[1]),
                )
            ],
        },
        # Backward-compat: older saves may include these, but we no longer write them.
        "json_circle_round_mod": None,
        "text_circle_round_mod": None,
    }


def _apply_saved_to_session(saved: dict) -> None:
    raw_json = saved.get("raw_json")
    if isinstance(raw_json, str) and raw_json.strip():
        st.session_state["raw_json_next"] = raw_json

    text_pattern = saved.get("text_pattern")
    if isinstance(text_pattern, str):
        st.session_state["text_pattern"] = text_pattern

    settings = saved.get("settings", {})
    if isinstance(settings, dict):
        for k, v in settings.items():
            st.session_state[k] = v
        colors = settings.get("text_round_colors")
        if isinstance(colors, list):
            for i, color in enumerate(colors):
                if isinstance(color, str) and color.strip():
                    st.session_state[_round_color_key("text", i)] = color
        background = settings.get("text_background_color")
        if isinstance(background, str) and background.strip():
            st.session_state["text_background_color"] = background

    json_mod = saved.get("json_circle_round_mod")
    if isinstance(json_mod, dict):
        st.session_state["json_circle_round_mod"] = json_mod
        try:
            r = int(json_mod.get("round_index", 0)) + 1
            st.session_state[f"json_mod_segments_{r}"] = [
                {"length": int(s.get("length", 1)), "bump": float(s.get("bump", 0.0)), "label": ""} for s in json_mod.get("segments", [])
            ]
        except Exception:
            pass

    text_mod = saved.get("text_circle_round_mod")
    if isinstance(text_mod, dict):
        st.session_state["text_circle_round_mod"] = text_mod
        try:
            r = int(text_mod.get("round_index", 0)) + 1
            st.session_state[f"text_mod_segments_{r}"] = [
                {"length": int(s.get("length", 1)), "bump": float(s.get("bump", 0.0)), "label": ""} for s in text_mod.get("segments", [])
            ]
        except Exception:
            pass


with st.sidebar:
    st.header("图纸库")
    with st.expander("已保存图纸", expanded=True):
        saved_files = _list_saved()
        labels = [p.stem for p in saved_files]
        if labels:
            selected = st.selectbox("选择", labels, index=0, key="lib_selected")
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_a:
                if st.button("打开", key="lib_open"):
                    path = _SAVED_DIR / f"{selected}.json"
                    _apply_saved_to_session(_read_saved(path))
                    st.session_state["lib_current_slug"] = selected
                    st.rerun()
            with col_b:
                if st.button("删除", key="lib_delete"):
                    path = _SAVED_DIR / f"{selected}.json"
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    st.rerun()
            with col_c:
                if st.button("保存", key="lib_overwrite_selected"):
                    path = _SAVED_DIR / f"{selected}.json"
                    now = datetime.now().isoformat(timespec="seconds")
                    payload = _build_save_payload(name=str(selected), now_iso=now)
                    _write_saved(path, payload)
                    st.session_state["lib_current_slug"] = selected
                    st.success(f"已覆盖保存：{selected}")
        else:
            st.caption("暂无保存的图纸。")

    with st.expander("保存当前", expanded=False):
        name = st.text_input("名称", key="lib_save_name", placeholder="例如：小猫挂件_v1")
        overwrite = st.checkbox("覆盖同名", value=False, key="lib_save_overwrite")
        if st.button("保存", key="lib_save"):
            slug = _slugify(name)
            if not slug:
                st.error("请输入名称。")
            else:
                path = _SAVED_DIR / f"{slug}.json"
                if path.exists() and not overwrite:
                    st.error("同名已存在，勾选“覆盖同名”后再保存。")
                else:
                    now = datetime.now().isoformat(timespec="seconds")
                    payload = _build_save_payload(name=name, now_iso=now)
                    _write_saved(path, payload)
                    st.success(f"已保存：{slug}")
        if st.button("新图纸", key="lib_new"):
            st.session_state["lib_current_slug"] = None
            tmpl = st.session_state.get("json_template") or "examples/circle_coaster_min.json"
            raw_template = (_APP_DIR / str(tmpl)).read_text(encoding="utf-8")
            st.session_state["raw_json_next"] = raw_template
            st.session_state["text_pattern"] = ""
            st.rerun()


def _render_user_page() -> None:
    st.session_state.setdefault("text_pattern", "6x\n6v\n6[x,v]\n6[2x,v]")
    st.session_state.setdefault("text_motif_type", "circle_coaster")
    st.session_state.setdefault("text_start_mode", "auto")
    st.session_state.setdefault("text_auto_bump_scope", "all")
    st.session_state.setdefault("text_auto_bump_window", _DEFAULT_AUTO_BUMP_WINDOW)
    st.session_state.setdefault("text_auto_bump_smoothing", "weighted")
    st.session_state.setdefault("text_outline_sampling", "uniform")
    st.session_state.setdefault("text_glyph_angle_mode", "by_structure")
    st.session_state.setdefault("text_glyph_style", "metric_strokes")

    input_col, output_col = st.columns([0.92, 1.08], gap="large")

    with input_col:
        st.subheader("输入图纸")
        with st.expander("自动识别", expanded=False):
            st.caption("可粘贴中文图解文字，也可上传包含文字说明的图片；转换后请检查再应用。")
            uploaded_text_image = st.file_uploader(
                "上传图解图片（可选）",
                type=["png", "jpg", "jpeg", "webp"],
                key="natural_text_image",
            )
            if st.button("识别图片文字", key="natural_ocr_button"):
                if uploaded_text_image is None:
                    st.warning("请先上传图片。")
                else:
                    try:
                        st.session_state["natural_source_text"] = ocr_image_bytes(uploaded_text_image.getvalue())
                        st.success("已识别图片文字，请检查后再转换。")
                    except Exception as e:
                        st.error(f"OCR 失败：{e}")
            st.text_area(
                "白话图解 / OCR 文字",
                key="natural_source_text",
                height=150,
                placeholder="例如：R1：环形起针，5ch，5*(F，2ch)，引拔在起立的第3个锁针处",
            )
            if st.button("转换成代码并应用", key="natural_convert_apply_button", use_container_width=True):
                result = natural_text_to_pattern(st.session_state.get("natural_source_text", ""))
                st.session_state["natural_pattern_code"] = result.code
                st.session_state["natural_pattern_notes"] = "\n".join(result.notes)
                if result.code.strip():
                    st.session_state["text_pattern"] = result.code
                    st.rerun()
                else:
                    st.warning("没有识别到可应用的 Pattern code。")
            if st.session_state.get("natural_pattern_notes"):
                st.caption(st.session_state["natural_pattern_notes"])
            st.text_area("转换结果", key="natural_pattern_code", height=120)

        with st.expander("手动输入", expanded=True):
            _render_pattern_text_editor(
                key="text_pattern",
                height=300,
                motif_type=str(st.session_state.get("text_motif_type", "circle_coaster")),
            )

        try:
            _color_pattern = text_to_pattern(
                st.session_state.get("text_pattern", ""),
                motif_type=st.session_state.get("text_motif_type", "circle_coaster"),
                title=None,
            )
            _repair_unreadable_color_state(_color_pattern, "text")
            with st.expander("颜色", expanded=False):
                if "text_background_color" not in st.session_state:
                    st.session_state["text_background_color"] = _color_pattern.background
                if st.button("恢复默认配色", key="text_reset_round_colors"):
                    _reset_round_color_state(_color_pattern, "text")
                    st.rerun()
                st.color_picker("背景颜色", key="text_background_color")
                color_cols = st.columns(3)
                for i, round_spec in enumerate(_color_pattern.rounds):
                    key = _round_color_key("text", i)
                    if key not in st.session_state:
                        st.session_state[key] = round_spec.color
                    with color_cols[i % len(color_cols)]:
                        st.color_picker(f"第 {i + 1} 圈", key=key)
        except Exception as e:
            st.caption(f"Pattern text 暂不可解析，修正后可调色：{e}")

        size = st.slider("画布尺寸 (px)", min_value=256, max_value=512, value=512, step=64, key="text_size")
        st.slider("画布旋转（度）", min_value=-180.0, max_value=180.0, value=0.0, step=1.0, key="text_rotation_deg")

    with output_col:
        st.subheader("预览")
        try:
            pattern = text_to_pattern(
                st.session_state.get("text_pattern", ""),
                motif_type=st.session_state.get("text_motif_type", "circle_coaster"),
                title=None,
            )
            pattern = _apply_round_color_overrides(pattern, "text")
            out_json = pattern_to_json(pattern)
        except Exception as e:
            st.error(str(e))
            st.stop()

        validations = validate_pattern(pattern)
        problems = [v for v in validations if v.status != "ok"]
        if problems:
            st.warning("部分圈数可能未完全对应上一圈针目，预览可能不稳定。")

        size = int(st.session_state.get("text_size", 512) or 512)
        opts_kwargs = {
            "size": size,
            "padding": _auto_padding(size),
            "rotation_deg": float(st.session_state.get("text_rotation_deg") or 0.0),
            "start_mode": "auto",
            "center_hole_px": None,
            "first_prev_total_override": None,
            "circle_round_mod": None,
            "circle_auto_bump": True,
            "circle_auto_bump_scope": "all",
            "circle_auto_bump_round": None,
            "circle_auto_bump_window": int(st.session_state.get("text_auto_bump_window", _DEFAULT_AUTO_BUMP_WINDOW) or _DEFAULT_AUTO_BUMP_WINDOW),
            "circle_auto_bump_smoothing": str(st.session_state.get("text_auto_bump_smoothing") or "weighted"),
            "circle_auto_bump_strength": 1.0,
            "circle_outline_sampling": "uniform",
            "glyph_full_height": True,
            "glyph_angle_mode": "by_structure",
            "glyph_style": "metric_strokes",
            "chain_render_style": "stroke",
            "chain_width_scale": 1.18,
        }
        svg = render_svg(pattern, _safe_render_options(RenderOptions, opts_kwargs))
        _render_interactive_svg_preview(svg, size=size, component_key="user_text_preview", height=640)
        with st.expander("生图参考包", expanded=False):
            control_width = st.slider("结构图导出宽度", min_value=512, max_value=1536, value=1024, step=128, key="realize_control_width")
            try:
                reference_zip = _realization_reference_package_bytes(
                    pattern,
                    opts_kwargs,
                    pattern_text=str(st.session_state.get("text_pattern", "")),
                    pattern_json=out_json,
                    output_width=int(control_width),
                )
                st.download_button(
                    "下载生图参考包",
                    data=reference_zip,
                    file_name="stitchsketch_realization_reference.zip",
                    mime="application/zip",
                    key="realize_reference_package_download",
                )
            except Exception as e:
                st.warning(f"参考包暂时无法生成：{e}")
        st.download_button("下载 JSON", data=out_json.encode("utf-8"), file_name="stitchsketch_pattern.json", mime="application/json")

    st.stop()


_render_user_page()
