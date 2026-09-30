"""Week-by-week model predictions next to what players actually scored."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.predictions_page import player_history, prediction_comparison, weekly_accuracy
from ffapp.app.ui import apply_app_shell
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.evaluation.inseason import load_prediction_history
from ffapp.evaluation.weekly_review import build_weekly_review
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper

st.set_page_config(page_title="Predictions vs Actuals", layout="wide")
apply_app_shell()
league = select_league()
settings = load_settings()
st.title("Predictions vs Actuals")
st.caption(
    f"{league.display_name} · the model's last pregame prediction next to what each player scored"
)


@st.cache_data(ttl=300, show_spinner="Matching predictions to results…")
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


@st.cache_data(ttl=300, show_spinner=False)
def _my_roster_ids(slug: str, league_id: str | None) -> set[str]:
    """Canonical ids on my roster; empty when the Sleeper cache is unavailable."""
    if league_id is None or settings.sleeper_username is None:
        return set()
    try:
        players = mapping.build_players_dim(
            nflverse.fetch_player_ids(offline=True, settings=settings),
            sleeper.fetch_players(offline=True, settings=settings),
            mapping.ID_OVERRIDES_PATH,
        )
        ids = dict(players.select("sleeper_id", "player_id").drop_nulls().iter_rows())
        rosters = json.loads(
            sleeper.fetch_rosters(league_id, offline=True, settings=settings).read_text()
        )
        user = json.loads(
            sleeper.fetch_user(
                settings.sleeper_username, offline=True, settings=settings
            ).read_text()
        )
        roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
        mine = next(row for row in rosters if row["roster_id"] == roster_id)
    except (OSError, ValueError, StopIteration):
        return set()
    return {ids[player] for player in (mine.get("players") or []) if player in ids}


override_path = settings.data_root / "outputs" / league.slug / "injury_review.json"
review = _load_review(
    str(settings.data_root),
    league.slug,
    override_path.stat().st_mtime if override_path.exists() else 0,
)

with st.sidebar:
    include_injured = st.toggle(
        "Include players hurt mid-game",
        value=False,
        help="Off by default: no pregame prediction could have known about an in-game injury.",
    )

comparison = prediction_comparison(review, include_injured=include_injured)
if comparison.is_empty():
    st.info(
        "No finished week has both a saved pregame prediction and real scores yet. "
        "This fills in after the first Tuesday refresh following a completed week."
    )
    st.stop()

my_ids = _my_roster_ids(league.slug, league.league_id)
positions = sorted(comparison["position"].drop_nulls().unique().to_list())

filter_a, filter_b, filter_c = st.columns([1.4, 1, 1])
chosen_positions = filter_a.multiselect("Positions", positions, default=positions)
scope_options = ["All players", "My roster"] if my_ids else ["All players"]
scope = filter_b.radio("Players", scope_options, horizontal=True)
minimum = filter_c.slider(
    "Minimum predicted points",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=1.0,
    help="Hides deep bench players the model expected almost nothing from.",
)

filtered = comparison.filter(pl.col("position").is_in(chosen_positions))
if scope == "My roster":
    filtered = filtered.filter(pl.col("player_id").is_in(list(my_ids)))
else:
    filtered = filtered.filter(pl.col("predicted") >= minimum)
if filtered.is_empty():
    st.warning("No players match these filters.")
    st.stop()

number = st.column_config.NumberColumn
by_week, by_player, accuracy = st.tabs(["Week by week", "One player", "Accuracy by week"])

with by_week:
    periods = sorted(filtered.select("season", "week").unique().iter_rows(), reverse=True)
    season, week = st.selectbox(
        "Week", periods, format_func=lambda period: f"{period[0]} · Week {period[1]}"
    )
    week_rows = filtered.filter((pl.col("season") == season) & (pl.col("week") == week))
    summary = weekly_accuracy(week_rows).row(0, named=True)
    a, b, c, d = st.columns(4)
    a.metric("Players", str(summary["players"]))
    b.metric("Average miss", f"{summary['average_miss']:.1f} pts")
    c.metric("Within 5 points", f"{summary['within_five']:.0%}")
    d.metric(
        "Model leaned",
        "Too high" if summary["average_difference"] < 0 else "Too low",
        f"{summary['average_difference']:+.1f} pts per player",
        delta_color="off",
    )
    order = st.radio(
        "Sort by",
        ["Predicted points", "Biggest misses", "Beat the prediction", "Fell short"],
        horizontal=True,
    )
    sort_column, descending = {
        "Predicted points": ("predicted", True),
        "Biggest misses": ("miss", True),
        "Beat the prediction": ("difference", True),
        "Fell short": ("difference", False),
    }[order]
    st.dataframe(
        week_rows.sort(sort_column, descending=descending).select(
            "player", "position", "predicted", "actual", "difference"
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "player": "Player",
            "position": "Pos",
            "predicted": number("Predicted", format="%.1f"),
            "actual": number("Actual", format="%.1f"),
            "difference": number(
                "Actual − predicted",
                format="%+.1f",
                help="Positive means the player scored more than the model predicted.",
            ),
        },
    )
    st.caption("Each dot is a player. Dots on the diagonal were predicted exactly.")
    st.scatter_chart(
        week_rows.select("predicted", "actual", "position").to_pandas(),
        x="predicted",
        y="actual",
        color="position",
        x_label="Predicted points",
        y_label="Actual points",
    )

with by_player:
    names = (
        filtered.group_by("player_id")
        .agg(pl.col("player").last(), pl.col("predicted").mean().alias("average"))
        .sort("average", descending=True)
    )
    labels = dict(zip(names["player_id"].to_list(), names["player"].to_list(), strict=True))
    chosen = st.selectbox("Player", list(labels), format_func=lambda player_id: labels[player_id])
    history = player_history(comparison, chosen)
    a, b, c = st.columns(3)
    a.metric("Weeks scored", str(history.height))
    b.metric("Average predicted", f"{history['predicted'].mean():.1f}")
    c.metric(
        "Average actual",
        f"{history['actual'].mean():.1f}",
        f"{history['difference'].mean():+.1f} vs predicted",
    )
    chart = history.select(
        pl.format("{} W{}", "season", "week").alias("Week"),
        pl.col("predicted").alias("Predicted"),
        pl.col("actual").alias("Actual"),
    )
    st.line_chart(chart.to_pandas().set_index("Week"))
    st.dataframe(
        history.select("season", "week", "predicted", "actual", "difference"),
        width="stretch",
        hide_index=True,
        column_config={
            "season": number("Season", format="%d"),
            "week": number("Week", format="%d"),
            "predicted": number("Predicted", format="%.1f"),
            "actual": number("Actual", format="%.1f"),
            "difference": number("Actual − predicted", format="%+.1f"),
        },
    )

with accuracy:
    st.caption("How close the model was each week, for the players matching the filters above.")
    st.dataframe(
        weekly_accuracy(filtered),
        width="stretch",
        hide_index=True,
        column_config={
            "season": number("Season", format="%d"),
            "week": number("Week", format="%d"),
            "players": "Players",
            "average_miss": number("Average miss", format="%.1f"),
            "average_difference": number(
                "Actual − predicted",
                format="%+.1f",
                help="Average per player. Negative means the model predicted too high.",
            ),
            "within_five": number("Within 5 points", format="percent"),
            "predicted_total": number("Predicted total", format="%.0f"),
            "actual_total": number("Actual total", format="%.0f"),
        },
    )
