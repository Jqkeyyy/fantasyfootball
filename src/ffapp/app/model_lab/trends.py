"""Automatically updating projection and accuracy charts for each player cohort."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.app.performance_trends import POSITIONS, performance_trends
from ffapp.evaluation.inseason import load_prediction_history
from ffapp.evaluation.weekly_review import build_weekly_review

league = league  # noqa: F821 -- provided by the Model Lab page
settings = settings  # noqa: F821
st.caption(f"{league.display_name} · Saved pregame projections versus real fantasy points")
st.caption(
    "Charts check for new saved results every five minutes while this page is open. "
    "Completed weeks appear after results are refreshed; these are not live game scores."
)


@st.cache_data(ttl=300, show_spinner="Matching saved forecasts to results…")
def load_review(data_root: str, slug: str) -> pl.DataFrame:
    root = Path(data_root)
    output = root / "outputs" / slug
    history = load_prediction_history(output / "prediction_log", include_kickoff=True)
    seasons = history["season"].unique().to_list() if "season" in history.columns else []

    def read(name: str) -> pl.DataFrame:
        path = root / "interim" / f"{name}.parquet"
        if not path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(path)
        return frame.filter(pl.col("season").is_in(seasons)) if "season" in frame.columns else frame

    path = output / "injury_review.json"
    overrides = json.loads(path.read_text()) if path.exists() else {}
    return build_weekly_review(
        history,
        read("schedule"),
        read("player_week_stats"),
        read("player_week_usage"),
        read("injuries"),
        overrides,
    )


@st.fragment(run_every="5m")
def render_trends() -> None:
    if st.button("Refresh charts now"):
        load_review.clear()
    review = load_review(str(settings.data_root), league.slug)
    st.caption(f"Last checked: {datetime.now().astimezone():%b %d, %Y %H:%M %Z}")
    if review.is_empty():
        st.info("No saved forecasts yet. Charts fill in as pregame predictions and results arrive.")
        return
    seasons = sorted(review["season"].unique().to_list(), reverse=True)
    season = st.selectbox("Season", seasons)
    review = review.filter(pl.col("season") == season)
    include_injured = st.toggle("Include injury-affected games", value=True)
    measure = st.radio("Points shown", ["Average per player", "Total points"], horizontal=True)
    st.caption(
        "Top 100 = each week's 100 highest saved pregame projections, ranked before checking "
        "results. Missing results are never treated as zero. Both lines use the same players. "
        "Everything includes all logged positions with verified pregame projections."
    )

    def chart(cohort: str) -> None:
        st.subheader(cohort)
        trends = performance_trends(review, cohort=cohort, include_injured=include_injured)
        if trends.is_empty() or trends["matched"].sum() == 0:
            st.info("No completed, matched results for this group yet.")
            return
        latest = trends.filter(pl.col("matched") > 0).row(-1, named=True)
        a, b, c = st.columns(3)
        a.metric("Week", str(latest["week"]))
        b.metric("Players", str(latest["matched"]))
        c.metric("Avg. miss", f"{latest['average_miss']:.1f}")
        st.caption(
            f"Latest scored week: {latest['matched']} of {latest['selected']} selected players. "
            "Average miss is in fantasy points."
        )
        suffix = "average" if measure == "Average per player" else "total"
        points = trends.select(
            "Week",
            pl.col(f"predicted_{suffix}").alias("Projected"),
            pl.col(f"actual_{suffix}").alias("Actual"),
        )
        st.line_chart(
            points,
            x="Week",
            y=["Projected", "Actual"],
            color=["#2563eb", "#059669"],
            y_label="Fantasy points",
            height=270,
        )
        st.caption("Prediction error over time · lower average miss is better")
        st.line_chart(
            trends.select(
                "Week", pl.col("average_miss").alias("Average miss"), pl.col("bias").alias("Bias")
            ),
            x="Week",
            y=["Average miss", "Bias"],
            color=["#d97706", "#7c3aed"],
            y_label="Points per player",
            height=200,
        )
        st.caption("Positive bias means projections were too high; negative means too low.")
        if (trends["unscored_or_excluded"] > 0).any():
            st.warning(
                "Some selected players lack usable results or are excluded. See coverage below."
            )
        with st.expander("Weekly numbers and coverage"):
            st.dataframe(trends, hide_index=True, width="stretch")
            st.download_button(
                "Download weekly results",
                trends.write_csv(),
                file_name=f"{league.slug}-{season}-{cohort.lower().replace(' ', '-')}.csv",
                mime="text/csv",
                key=f"download-{cohort}",
            )

    top, positions, everything = st.tabs(["Top 100", "By position", "Everything"])
    with top:
        chart("Top 100")
    with positions:
        for start in range(0, len(POSITIONS), 2):
            for column, position in zip(st.columns(2), POSITIONS[start : start + 2], strict=True):
                with column:
                    chart(position)
    with everything:
        chart("Everything")
    with st.expander("Data availability"):
        st.dataframe(review.group_by("status").len().sort("status"), hide_index=True)
        st.caption("Unverified snapshots and unfinished weeks are excluded from every chart.")


render_trends()
