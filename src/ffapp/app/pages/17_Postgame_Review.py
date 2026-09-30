"""Automatic completed-week decision and model autopsy."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.ui import apply_app_shell
from ffapp.config import load_settings
from ffapp.evaluation.inseason import load_prediction_history
from ffapp.evaluation.postgame import decision_autopsy, model_miss_autopsy
from ffapp.evaluation.weekly_review import build_weekly_review, score_summary
from ffapp.tools.decision_ledger import empty_ledger, ledger_path

st.set_page_config(page_title="Postgame Review", layout="wide")
apply_app_shell()
league = select_league()
settings = load_settings()
st.title("Weekly Postgame Autopsy")
st.caption(f"{league.display_name} · decisions, model misses, and evidence after games finish")


@st.cache_data(ttl=300, show_spinner="Building the postgame review…")
def _load_review(root: str, slug: str, revision: float) -> pl.DataFrame:
    data = Path(root)
    override_path = data / "outputs" / slug / "injury_review.json"
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}

    def read(name: str) -> pl.DataFrame:
        path = data / "interim" / f"{name}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    return build_weekly_review(
        load_prediction_history(data / "outputs" / slug / "prediction_log", include_kickoff=True),
        read("schedule"),
        read("player_week_stats"),
        read("player_week_usage"),
        read("injuries"),
        overrides,
    )


override_path = settings.data_root / "outputs" / league.slug / "injury_review.json"
review = _load_review(
    str(settings.data_root),
    league.slug,
    override_path.stat().st_mtime if override_path.exists() else 0,
)
ready = review.filter(pl.col("status").is_in(["Scored", "Injury excluded"]))
if ready.is_empty():
    st.info("No completed week with real results and a verified pregame forecast is ready yet.")
    st.stop()

periods = sorted(ready.select("season", "week").unique().iter_rows(), reverse=True)
season, week = st.selectbox(
    "Completed week", periods, format_func=lambda period: f"{period[0]} · Week {period[1]}"
)
week_review = review.filter((pl.col("season") == season) & (pl.col("week") == week))
summary = score_summary(week_review, ["season", "week"])
ledger_file = ledger_path(settings.data_root, league.slug)
ledger = pl.read_parquet(ledger_file) if ledger_file.exists() else empty_ledger()
decisions = decision_autopsy(ledger, week_review, season=season, week=week)
misses = model_miss_autopsy(week_review)

metrics = summary.row(0, named=True) if not summary.is_empty() else {}
counted = decisions.filter(pl.col("counted")) if not decisions.is_empty() else pl.DataFrame()
helped = counted.filter(pl.col("your_choice_value") > 0).height if not counted.is_empty() else 0
a, b, c, d = st.columns(4)
a.metric("Average model miss", f"{float(metrics.get('mae') or 0):.1f}")
b.metric("Within 5 points", f"{float(metrics.get('within_five') or 0):.0%}")
c.metric("Choices reviewed", str(counted.height))
d.metric("Choices that helped", str(helped))
injuries = week_review.filter(pl.col("status") == "Injury excluded").height
st.caption(
    f"{injuries} confirmed injury outcome(s) excluded. Evidence describes association, "
    "not proof of cause."
)

st.subheader("Your decisions")
if decisions.is_empty():
    st.info("No recommendation choice was recorded for this week.")
else:
    st.dataframe(
        decisions,
        width="stretch",
        hide_index=True,
        column_config={
            "projected_edge": st.column_config.NumberColumn(format="%+.1f"),
            "realized_edge": st.column_config.NumberColumn(format="%+.1f"),
            "your_choice_value": st.column_config.NumberColumn(format="%+.1f"),
        },
    )

st.subheader("Biggest model misses")
if misses.is_empty():
    st.info("No non-injury scored players are available for this week.")
else:
    st.dataframe(
        misses,
        width="stretch",
        hide_index=True,
        column_config={
            "projected": st.column_config.NumberColumn(format="%.1f"),
            "actual": st.column_config.NumberColumn(format="%.1f"),
            "miss": st.column_config.NumberColumn(format="%.1f"),
        },
    )
st.link_button("Open the full model report", "/Weekly_Report")
