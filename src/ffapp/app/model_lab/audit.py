"""Fair comparisons and attribution from saved pregame forecasts."""

import json

import polars as pl
import streamlit as st

from ffapp.evaluation.accuracy_audit import (
    SOURCES,
    STAGES,
    adjustment_effects,
    audit_metrics,
    matched_audit,
)
from ffapp.evaluation.inseason import load_prediction_history

league = league  # noqa: F821 -- provided by the Model Lab page
settings = settings  # noqa: F821
st.warning(
    "A season-selection bug was confirmed in the old ESPN importer: it could read last year's "
    "projection for the same week. The importer now requires an explicit season match. "
    "Historical forecasts are preserved, but unverified ESPN estimates are excluded from "
    "automatic source weighting. Their errors are not evidence of ESPN's current accuracy."
)
st.caption(f"{league.display_name} · Compare the same players using the same saved snapshot")
st.info(
    "These are saved pipeline estimates, not an independently verified comparison of websites. "
    "Each comparison uses one pre-kickoff snapshot containing all selected estimates. "
    "Provider publication times were not recorded, so equal provider freshness cannot be proved. "
    "Older weekly live estimates were recomputed for logging and may differ from the dashboard."
)

root = settings.data_root
output = root / "outputs" / league.slug
history = load_prediction_history(output / "prediction_log", include_kickoff=True)


def read_table(name: str) -> pl.DataFrame:
    path = root / "interim" / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


override_path = output / "injury_review.json"
overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
mode = st.radio("Audit", ["Compare sources", "Check our adjustments"], horizontal=True)
if mode == "Compare sources":
    selected = st.multiselect(
        "Sources", list(SOURCES), default=["ESPN pipeline estimate", "Custom model"]
    )
    sources = {label: SOURCES[label] for label in selected}
    st.caption(
        "ESPN's weekly stats are scored using league rules; custom and recent-performance "
        "estimates use league-scored targets. ESPN pipeline estimates can include fallback "
        "values. Consensus is excluded by default because its league-scoring compatibility "
        "is unverified. Do not use it to declare a platform winner."
    )
else:
    sources = STAGES
    st.caption(
        "Uses recorded stages from production projections. Positive error reduction means "
        "an adjustment helped. Historical stages are never reconstructed. The base already "
        "includes source selection and model availability; this isolates subsequent role, "
        "injury, and final kickoff changes only."
    )
if len(sources) < 2:
    st.info("Select at least two sources.")
    st.stop()
include_injured = st.toggle("Include confirmed injury exclusions", value=False)
rows, review = matched_audit(
    history,
    read_table("schedule"),
    read_table("player_week_stats"),
    read_table("injuries"),
    overrides,
    sources,
    include_injured=include_injured,
)
if history.is_empty():
    st.info("No saved predictions yet.")
    st.stop()
total = history.select("season", "week", "player_id").unique().height
st.caption(
    f"{total:,} logged player-weeks; {review.height:,} have all selected estimates in one "
    f"snapshot; {rows.height:,} pass pregame, completed-week, actual-score and injury checks."
)
if not review.is_empty():
    with st.expander("Eligibility and exclusions"):
        st.dataframe(review.group_by("status").len(), hide_index=True)
if rows.is_empty():
    st.info(
        "No eligible matched results yet. Adjustment stages start with projections generated "
        "after this audit was installed and become scoreable after games finish. "
        "For source comparisons, try a different pair of sources."
    )
    st.stop()
periods = sorted(rows.select("season", "week").unique().iter_rows(), reverse=True)
period = st.selectbox(
    "Period",
    [None, *periods],
    format_func=lambda p: "All completed weeks" if p is None else f"{p[0]} · Week {p[1]}",
)
positions = st.multiselect(
    "Positions",
    sorted(rows["position"].unique().to_list()),
    default=sorted(rows["position"].unique().to_list()),
)
scope = st.radio(
    "Players", ["All players", "My roster at snapshot", "My starters at snapshot"], horizontal=True
)
minimum = st.slider("Minimum points predicted by any selected source", 0.0, 25.0, 5.0, 1.0)
if period is not None:
    rows = rows.filter((pl.col("season") == period[0]) & (pl.col("week") == period[1]))
rows = rows.filter(
    pl.col("position").is_in(positions) & (pl.max_horizontal(list(sources.values())) >= minimum)
)
if scope != "All players":
    column = "is_my_roster" if scope == "My roster at snapshot" else "was_starting"
    rows = rows.filter(pl.col(column) == True) if column in rows.columns else rows.head(0)  # noqa: E712
if rows.is_empty():
    st.warning("No matched players under these filters; roster flags may be unavailable.")
    st.stop()
if "espn_mean" in sources.values():
    verified_count = (
        rows["espn_season_verified"].fill_null(False).sum()
        if "espn_season_verified" in rows.columns
        else 0
    )
    st.caption(f"ESPN season verified: {verified_count:,} of {rows.height:,} matched rows.")
st.caption(
    "Lower average miss and RMSE are better. Positive bias means predictions were too high. "
    "The points filter uses predictions only, never actual results. Small samples are descriptive, "
    "not proof that a source will win next week. This audit does not change production weights."
)
st.dataframe(audit_metrics(rows, sources), hide_index=True, width="stretch")
if mode == "Check our adjustments":
    effects = adjustment_effects(rows)
    if effects.is_empty():
        st.info("No adjustments changed predictions in this sample.")
    else:
        st.dataframe(effects, hide_index=True, width="stretch")
with st.expander("Results by position"):
    tables = [
        audit_metrics(group, sources).with_columns(pl.lit(group["position"][0]).alias("Position"))
        for group in rows.partition_by("position")
    ]
    st.dataframe(pl.concat(tables), hide_index=True)
with st.expander("Results by week"):
    tables = [
        audit_metrics(group, sources).with_columns(
            pl.lit(group["season"][0]).alias("Season"), pl.lit(group["week"][0]).alias("Week")
        )
        for group in rows.partition_by(["season", "week"])
    ]
    st.dataframe(pl.concat(tables), hide_index=True)
detail = rows.select(
    "season",
    "week",
    "player",
    "player_id",
    "position",
    "snapshot",
    "actual",
    *[pl.col(column).alias(label) for label, column in sources.items()],
)
st.subheader("Matched predictions")
st.dataframe(detail, hide_index=True, width="stretch")
st.download_button(
    "Download matched comparison",
    detail.write_csv(),
    file_name=f"{league.slug}-accuracy-audit.csv",
    mime="text/csv",
)
