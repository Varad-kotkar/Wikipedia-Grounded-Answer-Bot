"""
app.py — WikiTruth AI
A six-page Streamlit application for the Wikipedia-Grounded Answer Bot.

Pages:
  1. Home               — Project intro, RAG architecture, tech stack
  2. Ask WikiTruth      — Question answering with grounding + images
  3. Grounded vs Ungrounded — Side-by-side comparison
  4. Evaluation Dashboard   — 20-question evaluation results with charts
  5. Prompt Engineering Lab — Zero-shot vs few-shot comparison
  6. About Project          — Architecture, units, responsible AI

P53 College Mini-Project | MIT ADT University | Generative AI Course
"""

import io
import json
import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WikiTruth AI — Wikipedia-Grounded Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d1b2a 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Cards */
    .wiki-card {
        background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #0d2137 0%, #132a44 100%);
        border: 1px solid #4fc3f7;
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Grounded answer */
    .grounded-answer {
        background: linear-gradient(135deg, #0d2137 0%, #132a44 100%);
        border: 1px solid #4caf50;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Ungrounded answer */
    .ungrounded-answer {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d2020 100%);
        border: 1px solid #ef5350;
        border-left: 4px solid #ef5350;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #90caf9;
        margin-top: 0.25rem;
    }

    /* Status badges */
    .badge-grounded {
        background: #1b5e20;
