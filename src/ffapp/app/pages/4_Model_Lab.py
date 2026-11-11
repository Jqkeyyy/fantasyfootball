"""Model Lab: one home for model health, prediction accuracy, audits, and trends.

Each section lives in `ffapp/app/model_lab/` (outside `pages/`, so it stays out of the
sidebar) and runs with this page's `league` and `settings`. Only the selected section
renders, so a section that stops early (no data yet) never hides the others.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.ui import apply_app_shell
from ffapp.config import load_settings

SECTIONS_DIR = Path(__file__).resolve().parent.parent / "model_lab"
SECTIONS = {
    "health": (
        ":material/monitor_heart: Health",
        "Model health",
        "Which projection source is live, how each source is doing this season, the blend "
        "weights in use, and the full offline evaluation reports.",
        "health.py",
    ),
    "predictions": (
        ":material/compare_arrows: Predictions vs Actuals",
        "Predictions vs actuals",
        "The last pregame prediction next to what each player actually scored, week by "
        "week or one player at a time.",
        "predictions.py",
    ),
    "audit": (
        ":material/fact_check: Accuracy Audit",
        "Accuracy audit",
        "Head-to-head source comparisons on the same players and snapshot, plus whether "
        "role, injury, and kickoff adjustments helped.",
        "audit.py",
    ),
    "trends": (
        ":material/show_chart: Over Time",
        "Performance over time",
        "Season-long charts of projected versus real points for the top 100, each "
        "position, and every player.",
        "trends.py",
    ),
}

st.set_page_config(page_title="Model Lab", layout="wide")
apply_app_shell()
league = select_league()
settings = load_settings()

st.markdown(
    """
    <style>
    .ff-lab-hero {
        padding: 1.1rem 1.3rem;
        margin-bottom: .9rem;
        border-radius: 16px;
        color: #ffffff;
        background: linear-gradient(135deg, #132238 0%, #1b3453 60%, #24507a 100%);
        box-shadow: 0 10px 26px rgba(19,34,56,.18);
    }
    .ff-lab-hero__eyebrow {
        font-size: .7rem; font-weight: 800; letter-spacing: .14em; opacity: .75;
    }
    .ff-lab-hero__title { margin: .2rem 0 .25rem; font-size: 1.7rem; font-weight: 780; }
    .ff-lab-hero__sub { margin: 0; font-size: .95rem; opacity: .85; }
    .ff-lab-section {
        margin: .4rem 0 1rem;
        padding: .75rem 1rem;
        border-left: 4px solid #2f6fb0;
        border-radius: 10px;
        background: rgba(47,111,176,.08);
    }
    .ff-lab-section b { display: block; margin-bottom: .15rem; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="ff-lab-hero">
      <div class="ff-lab-hero__eyebrow">MODEL LAB · {league.display_name.upper()}</div>
      <div class="ff-lab-hero__title">How good are the projections?</div>
      <p class="ff-lab-hero__sub">
        Health, accuracy, source audits, and season trends in one place.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

requested = st.query_params.get("section", "health")
choice = st.segmented_control(
    "Section",
    list(SECTIONS),
    default=requested if requested in SECTIONS else "health",
    format_func=lambda key: SECTIONS[key][0],
    label_visibility="collapsed",
    key="model_lab_section",
)
section = choice or "health"
st.query_params["section"] = section
_, heading, blurb, filename = SECTIONS[section]
st.markdown(f'<div class="ff-lab-section"><b>{heading}</b>{blurb}</div>', unsafe_allow_html=True)

runpy.run_path(
    str(SECTIONS_DIR / filename),
    init_globals={"league": league, "settings": settings},
    run_name=f"ffapp_model_lab_{section}",
)
