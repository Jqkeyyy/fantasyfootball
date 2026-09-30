"""Weekly accuracy, injury review and evidence behind projection misses."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.config import load_settings
from ffapp.evaluation.inseason import load_prediction_history
from ffapp.evaluation.learning import recurring_patterns, replay_bias_correction
from ffapp.evaluation.weekly_review import build_weekly_review, review_key, score_summary
from ffapp.tools.artifacts import atomic_write_json

st.set_page_config(page_title="Weekly Report", layout="wide")
league = select_league()
settings = load_settings()
st.title("Weekly Model Report")
st.caption(f"{league.display_name} · Saved predictions, real results, visible exclusions")


@st.cache_data(ttl=300, show_spinner="Comparing saved projections with results…")
def load_review(root: str, slug: str, revision: float) -> pl.DataFrame:
    data = Path(root)
    overrides_path = data / "outputs" / slug / "injury_review.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}

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


review_path = settings.data_root / "outputs" / league.slug / "injury_review.json"
review = load_review(
    str(settings.data_root), league.slug, review_path.stat().st_mtime if review_path.exists() else 0
)
if review.is_empty():
    st.info(
        "No saved weekly predictions yet. This report fills in as refreshes save forecasts "
        "and completed-game results arrive."
    )
    st.stop()

periods = sorted(review.select("season", "week").unique().iter_rows(), reverse=True)
ready = sorted(
    review.filter(pl.col("status") == "Scored").select("season", "week").unique().iter_rows(),
    reverse=True,
)
period = st.selectbox(
    "Week to review",
    periods,
    index=periods.index(ready[0]) if ready else 0,
    format_func=lambda p: f"{p[0]} · Week {p[1]}",
)
minimum = st.slider(
    "Minimum projected points",
    0,
    25,
    5,
    help="Focus on relevant players. Set to 0 to include the full saved player pool.",
)
st.caption(
    "Main score excludes Out/IR designations and injuries you confirm below. "
    "Questionable tags and low snap counts alone do not remove a miss. "
    "Unreported in-game injuries may remain until reviewed."
)
filtered = review.filter(pl.col("projected") >= minimum)
week_rows = review.filter((pl.col("season") == period[0]) & (pl.col("week") == period[1]))
pool = week_rows.filter(pl.col("projected") >= minimum)
scored = pool.filter(pl.col("status") == "Scored")
excluded = pool.filter(pl.col("status") == "Injury excluded")
summary = score_summary(pool, ["season", "week"])
if summary.is_empty():
    st.info(
        "This week does not yet have scoreable pregame projections and real results. "
        "The availability table below shows what is missing."
    )
else:
    metrics = summary.row(0, named=True)
    a, b = st.columns(2)
    a.metric("Average miss", f"{metrics['mae']:.1f} pts")
    b.metric("Within 5 points", f"{metrics['within_five']:.0%}")
    c, d = st.columns(2)
    c.metric("Bias: predicted − actual", f"{metrics['bias']:+.1f} pts")
    d.metric("Players scored", metrics["players"])
    st.caption(
        f"{excluded.height} injury exclusions at this points threshold. "
        "Lower average miss is better. Positive bias means overprediction."
    )
    if metrics["coverage"] is not None:
        st.write(
            f"**Uncertainty check:** {metrics['coverage']:.0%} of "
            f"{metrics['interval_players']} available floor-to-ceiling ranges contained "
            "the result (target: 80%)."
        )
    if metrics["baseline_players"]:
        st.write(
            f"**Baseline comparison:** model {metrics['paired_model_mae']:.1f} vs "
            f"simple baseline {metrics['baseline_mae']:.1f} points average miss, "
            f"on the same {metrics['baseline_players']} players."
        )

    st.subheader("Where the model struggled")
    positions = score_summary(pool, ["position"])
    supported = positions.filter(pl.col("players") >= 10).sort("mae", descending=True)
    if not supported.is_empty():
        worst = supported.row(0, named=True)
        direction = "too high" if worst["bias"] > 0 else "too low"
        st.write(
            f"**First area to review: {worst['position']}.** Average miss was "
            f"{worst['mae']:.1f} points across {worst['players']} players; projections "
            f"averaged {abs(worst['bias']):.1f} points {direction}. Look for the same "
            "pattern over several weeks before changing the model."
        )
    if metrics["coverage"] is not None and metrics["interval_players"] >= 20:
        if metrics["coverage"] < 0.7:
            st.write(
                "**Uncertainty needs review:** too many results fell outside the expected "
                "range this week. Check whether future ranges need widening if this repeats."
            )
    st.dataframe(
        positions.select("position", "players", "mae", "bias", "within_five"),
        hide_index=True,
        width="stretch",
        column_config={
            "position": "Position",
            "players": "Players",
            "mae": st.column_config.NumberColumn("Average miss", format="%.1f"),
            "bias": st.column_config.NumberColumn("Bias", format="%+.1f"),
            "within_five": st.column_config.NumberColumn("Within 5", format="percent"),
        },
    )
    st.subheader("Biggest misses & evidence")
    st.caption(
        "These are diagnostic clues, not proven causes. Usage is compared with the "
        "prior four appearances in the same season. Touchdowns are observed results; "
        "we cannot attribute their point error without saved component forecasts."
    )
    st.dataframe(
        scored.sort("absolute_error", descending=True)
        .select("player", "position", "projected", "actual", "error", "signal", "evidence")
        .head(30),
        hide_index=True,
        width="stretch",
        column_config={
            "player": "Player",
            "position": "Pos",
            "signal": "Clue",
            "evidence": "Evidence",
            "projected": st.column_config.NumberColumn("Projected", format="%.1f"),
            "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
            "error": st.column_config.NumberColumn("Predicted − actual", format="%+.1f"),
        },
    )

st.subheader("Injury review")
st.caption(
    "Mark a confirmed injury that prevented a normal game, including an in-game exit. "
    "A poor result by itself is not evidence of an injury. Changes are saved for this league."
)
flags = week_rows.filter(pl.col("needs_review"))
if not flags.is_empty():
    st.warning(
        f"{flags.height} players had a large snap-share drop. Review them for injury "
        "or a role change; they remain scored unless an injury is confirmed."
    )
    st.dataframe(flags.select("player", "evidence"), hide_index=True, width="stretch")
names = {r["player_id"]: r["player"] for r in week_rows.sort("player").to_dicts()}
player_id = st.selectbox("Player", list(names), format_func=lambda key: names[key])
selected = week_rows.filter(pl.col("player_id") == player_id).row(0, named=True)
key = review_key(selected)
saved = json.loads(review_path.read_text()) if review_path.exists() else {}
existing = saved.get(key, {})
with st.form(f"injury_{key}"):
    exclude = st.checkbox(
        "Exclude this game for a confirmed injury", value=existing.get("exclude", False)
    )
    note = st.text_input(
        "Evidence / note",
        value=existing.get("note", ""),
        placeholder="For example: left in Q1 with an ankle injury",
    )
    if st.form_submit_button("Save injury review"):
        if exclude and not note.strip():
            st.error("Add a brief note explaining the injury exclusion.")
        else:
            latest = json.loads(review_path.read_text()) if review_path.exists() else {}
            latest[key] = {"exclude": exclude, "note": note.strip()}
            atomic_write_json(latest, review_path)
            load_review.clear()
            st.rerun()

with st.expander("Exclusions & data availability"):
    st.dataframe(
        week_rows.group_by("status").len().rename({"status": "Status", "len": "Players"}),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        week_rows.filter(pl.col("status") != "Scored").select(
            "player", "status", "injury_reason", "snapshot"
        ),
        hide_index=True,
        width="stretch",
    )
    tagged = week_rows.filter(pl.col("injury_data_available")).height
    st.caption(
        f"{tagged} players match an injury-report record. No record does not prove health. "
        "Results use the latest saved snapshot before each player's kickoff. "
        "Weeks open for scoring eight hours after the last scheduled kickoff, when "
        "non-placeholder actuals are available; pending actuals remain unscored. "
        "Updates appear within five minutes of a data refresh."
    )

st.subheader("Week-by-week trend")
st.caption(
    "Trends and learning experiments use your current points threshold and injury exclusions."
)
trend = score_summary(filtered, ["season", "week"])
if not trend.is_empty():
    chart = trend.with_columns(
        (pl.col("season").cast(pl.String) + " W" + pl.col("week").cast(pl.String)).alias("Week")
    ).select("Week", pl.col("mae").alias("Average miss"))
    st.line_chart(chart.to_pandas().set_index("Week"))
    st.dataframe(
        trend.select("season", "week", "players", "mae", "bias"), hide_index=True, width="stretch"
    )
st.subheader("What keeps going wrong?")
patterns = recurring_patterns(filtered)
if patterns.is_empty():
    st.info(
        "No recurring pattern has enough evidence yet. A pattern needs at least three "
        "qualifying weeks among the last six, with ten or more players per position."
    )
else:
    st.dataframe(patterns, width="stretch", hide_index=True)
with st.expander("Learning lab: test a correction before using it"):
    st.write(
        "This experiment learns a position's average bias from two to four earlier weeks, "
        "caps the adjustment at three points, and tests it on the next unseen week. "
        "Positive improvement means fewer points of error. It does not change live forecasts."
    )
    trials = replay_bias_correction(filtered)
    if trials.is_empty():
        st.info(
            "Waiting for at least two training weeks and one later test week for the same "
            "source and position, with at least ten players each week."
        )
    else:
        st.dataframe(trials, width="stretch", hide_index=True)
        st.caption(
            "Treat this as an experiment. Look for improvements across multiple test weeks "
            "before promoting a change; one good week is not enough."
        )
st.download_button(
    "Download this week's review",
    week_rows.write_csv(),
    file_name=f"weekly-review-{league.slug}-{period[0]}-{period[1]}.csv",
    mime="text/csv",
)
