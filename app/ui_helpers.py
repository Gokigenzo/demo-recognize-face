"""Shared Streamlit UI helpers.

Centralizes the Google-Teachable-Machine-inspired look & feel: rounded cards,
soft shadows, pastel accent colors, big friendly headings, and the
"educational message" callout used at the bottom of every lifecycle stage.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
TM_CSS = """
<style>
    :root {
        --tm-blue: #4285F4;
        --tm-teal: #00BFA5;
        --tm-orange: #F4B400;
        --tm-pink: #EA4C89;
        --tm-bg: #F7F9FC;
    }
    .stApp { background: var(--tm-bg); }
    .tm-hero {
        background: linear-gradient(135deg, #4285F4 0%, #00BFA5 100%);
        border-radius: 22px; padding: 26px 30px; color: white;
        box-shadow: 0 8px 24px rgba(66,133,244,0.25); margin-bottom: 18px;
    }
    .tm-hero h1 { color:#fff; font-size: 30px; margin:0 0 6px 0; font-weight:800; }
    .tm-hero p { color: rgba(255,255,255,0.92); font-size:16px; margin:0; }
    .tm-card {
        background:#fff; border-radius:18px; padding:20px 24px;
        box-shadow:0 4px 16px rgba(0,0,0,0.06); margin-bottom:16px;
        border:1px solid #eef1f6;
    }
    .tm-step {
        display:inline-block; background:#fff; border-radius:14px;
        padding:10px 16px; margin:4px; font-weight:600; color:#3c4043;
        box-shadow:0 2px 8px rgba(0,0,0,0.06); border:1px solid #eef1f6;
    }
    .tm-arrow { font-size:22px; color:var(--tm-blue); padding:0 6px; }
    .tm-lesson {
        background:linear-gradient(135deg,#FFF8E1 0%,#FFFDE7 100%);
        border-left:6px solid var(--tm-orange); border-radius:12px;
        padding:14px 18px; font-size:16px; color:#5f4b00; font-weight:600;
        margin-top:14px;
    }
    .tm-pill {
        display:inline-block; padding:4px 12px; border-radius:999px;
        font-size:12px; font-weight:700; letter-spacing:.3px;
    }
    .tm-pill.ok { background:#E6F4EA; color:#137333; }
    .tm-pill.warn { background:#FEEFC3; color:#A56300; }
    .tm-pill.bad { background:#FCE8E6; color:#C5221F; }
    .tm-metric { font-size:34px; font-weight:800; color:var(--tm-blue); }
</style>
"""


def inject_theme() -> None:
    st.markdown(TM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='tm-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def lesson(text: str) -> None:
    """The yellow 'educational message' callout."""
    st.markdown(f"<div class='tm-lesson'>💡 {text}</div>", unsafe_allow_html=True)


def pipeline(steps: List[str]) -> None:
    """Render a left-to-right 'A → B → C' pipeline of steps."""
    html = "<div style='margin:8px 0;'>"
    for i, step in enumerate(steps):
        html += f"<span class='tm-step'>{step}</span>"
        if i < len(steps) - 1:
            html += "<span class='tm-arrow'>→</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def pill(text: str, kind: str = "ok") -> str:
    """Return HTML for a colored status pill ('ok' | 'warn' | 'bad')."""
    return f"<span class='tm-pill {kind}'>{text}</span>"


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for Streamlit display."""
    if image is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def file_to_bgr(uploaded) -> np.ndarray:
    """Convert a Streamlit uploaded/camera file to an OpenCV BGR image."""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
