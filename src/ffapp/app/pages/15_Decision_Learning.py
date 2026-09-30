"""Personal recommendation, choice, and outcome learning dashboard."""

from __future__ import annotations

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.config import load_settings
from ffapp.evaluation.decision_learning import (
    decision_calibration,
    learning_message,
    personal_profile,
)
from ffapp.tools.decision_ledger import decision_summary, ledger_path, record_choice

st.set_page_config(page_title="Decision Learning", layout="wide")
settings = load_settings()
league = select_league()
st.title("Decision Learning")
st.caption(f"{league.display_name} · what you chose, what happened, and which advice helped")

path = ledger_path(settings.data_root, league.slug)
if not path.exists():
    st.info("Open Weekly Actions to save the first recommendation snapshot.")
    st.link_button("Open Weekly Actions", "/Weekly_Actions")
    st.stop()

ledger = pl.read_parquet(path)
profile = personal_profile(ledger)
a, b, c, d = st.columns(4)
a.metric("Recommendations", str(profile["recommendations"]))
b.metric("Choices recorded", str(profile["choices"]))
c.metric("Settled choices", str(profile["settled"]))
follow_rate = profile["follow_rate"]
d.metric(
    "Follow rate",
    "—" if follow_rate is None else f"{follow_rate:.0%}",
)
st.info(learning_message(ledger))

pending = ledger.filter(pl.col("status") == "recommended").sort(["season", "week"], descending=True)
st.subheader("Record your choices")
if pending.is_empty():
    st.success("Every saved recommendation has a recorded choice.")
else:
    labels = {
        str(row["decision_id"]): f"W{row['week']} · {row['recommended_action']}"
        for row in pending.iter_rows(named=True)
    }
    decision_id = st.selectbox(
        "Recommendation", list(labels), format_func=lambda value: labels[value]
    )
    yes, no = st.columns(2)
    if yes.button("I followed it", type="primary", width="stretch"):
        record_choice(path, decision_id, accepted=True)
        st.rerun()
    if no.button("I chose something else", width="stretch"):
        record_choice(path, decision_id, accepted=False)
        st.rerun()

st.subheader("Results by decision type")
st.dataframe(decision_summary(ledger), width="stretch", hide_index=True)
calibration = decision_calibration(ledger)
if calibration.is_empty():
    st.caption(
        "Result calibration appears after games settle and the next full refresh imports actual "
        "points."
    )
else:
    st.subheader("Which confidence levels helped")
    st.dataframe(
        calibration,
        width="stretch",
        hide_index=True,
        column_config={
            "help_rate": st.column_config.ProgressColumn(
                "Help rate", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )

settled = ledger.filter(pl.col("settled_at_utc").is_not_null()).sort(
    ["season", "week"], descending=True
)
if not settled.is_empty():
    st.subheader("Decision history")
    st.dataframe(
        settled.select(
            "season",
            "week",
            "decision_type",
            "recommended_action",
            "status",
            "expected_delta",
            "realized_delta",
            "decision_regret",
        ),
        width="stretch",
        hide_index=True,
    )
