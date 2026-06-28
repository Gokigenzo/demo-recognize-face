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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --tm-blue: #4285F4;
        --tm-teal: #00BFA5;
        --tm-orange: #F4B400;
        --tm-pink: #EA4C89;
        --tm-bg: #F8FAFC;
        --tm-text-main: #1E293B;
        --tm-text-muted: #64748B;
    }
    
    /* Font overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: var(--tm-text-main);
    }
    
    /* Global large font sizes */
    p, li, span, label, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    
    /* Headers sizing */
    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Hero container styling */
    .tm-hero {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        border-radius: 24px; padding: 32px 40px; color: white;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2); margin-bottom: 24px;
        transition: transform 0.3s ease;
    }
    .tm-hero:hover {
        transform: translateY(-2px);
    }
    .tm-hero h1 { color:#fff !important; font-size: 34px !important; margin:0 0 8px 0; font-weight:800; }
    .tm-hero p { color: rgba(255,255,255,0.95) !important; font-size:18px !important; margin:0; }
    
    /* Card design with hover effects */
    .tm-card {
        background:#fff; border-radius:20px; padding:24px 28px;
        box-shadow:0 4px 20px rgba(0,0,0,0.04); margin-bottom:20px;
        border:1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    .tm-card:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border-color: #CBD5E1;
    }
    
    /* Pipeline steps styling */
    .tm-step {
        display:inline-block; background:#fff; border-radius:16px;
        padding:12px 20px; margin:6px; font-weight:600; color:#334155;
        box-shadow:0 4px 12px rgba(0,0,0,0.05); border:1px solid #E2E8F0;
        font-size: 16px !important;
        transition: all 0.2s ease;
    }
    .tm-step:hover {
        transform: translateY(-1px);
        border-color: var(--tm-blue);
    }
    .tm-arrow { font-size:26px; color:var(--tm-blue); padding:0 8px; font-weight: 700; }
    
    /* Educational callouts */
    .tm-lesson {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 6px solid var(--tm-orange); border-radius: 16px;
        padding: 18px 24px; font-size: 18px !important; color: #78350F; font-weight: 600;
        margin-top: 18px; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);
    }
    
    /* Badges/Pills */
    .tm-pill {
        display:inline-block; padding:6px 14px; border-radius:999px;
        font-size:14px !important; font-weight:700; letter-spacing:.3px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .tm-pill.ok { background:#D1FAE5; color:#065F46; }
    .tm-pill.warn { background:#FEF3C7; color:#92400E; }
    .tm-pill.bad { background:#FEE2E2; color:#991B1B; }
    
    /* Metrics font overrides */
    .tm-metric { font-size:38px !important; font-weight:800; color:var(--tm-blue); }
    div[data-testid="stMetricValue"] {
        font-size: 40px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: var(--tm-text-muted) !important;
    }
    
    /* Form inputs and buttons styling */
    div[data-testid="stWidgetLabel"] p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #334155;
    }
    .stButton>button {
        font-size: 18px !important;
        padding: 12px 28px !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stSelectbox div[data-baseweb="select"] {
        font-size: 18px !important;
        border-radius: 12px !important;
    }
    .stTextInput input, .stTextArea textarea {
        font-size: 18px !important;
        border-radius: 12px !important;
    }
    
    /* Sidebar specific enhancements */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] li {
        font-size: 16px !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #0F172A;
    }
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
