"""Interactive rest-of-season rankings and player detail page."""

from __future__ import annotations

from typing import Any, cast

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.ros_rankings_page import (
    SORT_OPTIONS,
    RosBoardSchemaError,
    explain_ros_player,
    filter_board,
    player_week_schedule,
    prepare_board,
    style_rank_change,
    team_specific_recommendations,
    validate_board_schema,
)
from ffapp.config import load_settings
from ffapp.league_format import parse_league_format

st.set_page_config(page_title="ROS Rankings", layout="wide")

settings = load_settings()
league = select_league()

st.title("Rest-of-Season Player Rankings")
st.caption(league.display_name)
st.info(
    "This board includes every projected player. VOR uses the current free-agent "
    "replacement player at each position and this league's size, rosters, and scoring."
)

latest_path = settings.data_root / "outputs" / league.slug / "rankings_ros" / "latest.parquet"
if not latest_path.exists():
    st.error(
        f"Missing {latest_path}. Run `ffapp project --from-week --through-week --league "
        f"{league.slug}` then `ffapp rankings ros --league {league.slug}` first."
    )
    st.stop()

board = pl.read_parquet(latest_path)
try:
    validate_board_schema(board)
except RosBoardSchemaError as exc:
    st.error(str(exc))
    st.stop()
displayed = style_rank_change(board)

with st.sidebar:
    st.header("ROS filters")
    search = st.text_input("Player search", placeholder="Start typing a player name")
    positions = ["All", *sorted(board["position"].unique().to_list())]
    position_choice = st.selectbox("Position", options=positions)
    availability_choice = st.selectbox("Availability", options=["All", "Available", "Rostered"])
    nfl_teams = ["All", *sorted(board["nfl_team"].drop_nulls().unique().to_list())]
    nfl_team_choice = st.selectbox("NFL team", options=nfl_teams)
    fantasy_teams = ["All", *sorted(board["fantasy_team"].drop_nulls().unique().to_list())]
    fantasy_team_choice = st.selectbox("Fantasy team", options=fantasy_teams)
    st.header("Ranking")
    sort_label = st.selectbox("Rank by", options=list(SORT_OPTIONS))

filtered = filter_board(
    displayed,
    position=None if position_choice == "All" else position_choice,
    availability=None if availability_choice == "All" else availability_choice,
    nfl_team=None if nfl_team_choice == "All" else nfl_team_choice,
    fantasy_team=None if fantasy_team_choice == "All" else fantasy_team_choice,
    search=search,
)
ranked = prepare_board(filtered, sort_label=sort_label)

st.caption(f"{ranked.height} of {board.height} players shown — ranked by {sort_label}.")
table = ranked.select(
    "view_rank",
    "player_name",
    "position",
    "position_tier",
    "nfl_team",
    "availability",
    "fantasy_team",
    "vor_ros",
    "ros_points",
    "ros_ppg",
    "ros_p10",
    "ros_p50",
    "ros_p90",
    "expected_games",
    "playoff_weeks_value",
    "rank_change_display",
    "player_id",
)
event = st.dataframe(
    table,
    width="stretch",
    height=700,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "view_rank": "Rank",
        "position_tier": "Pos Tier",
        "nfl_team": "NFL Team",
        "fantasy_team": "Fantasy Team",
        "vor_ros": st.column_config.NumberColumn("VOR", format="%.1f"),
        "ros_points": st.column_config.NumberColumn("ROS Points", format="%.1f"),
        "ros_ppg": st.column_config.NumberColumn("PPG", format="%.1f"),
        "ros_p10": st.column_config.NumberColumn("Floor", format="%.1f"),
        "ros_p50": st.column_config.NumberColumn("Median", format="%.1f"),
        "ros_p90": st.column_config.NumberColumn("Upside", format="%.1f"),
        "player_id": None,
    },
)
st.caption(
    "Rank change always compares VOR rank with the prior run. Click a column header for "
    "an additional table sort, or select a player row for the full breakdown."
)

with st.expander("Best fits for my roster", expanded=True):
    recommendations = team_specific_recommendations(board, parse_league_format(league))
    if recommendations.is_empty():
        st.info("My roster could not be resolved, or no projected free agents are available.")
    else:
        st.caption(
            "Lineup gain compares each free agent with your projection-optimal current lineup. "
            "Playoff impact uses the same comparison over playoff-week value."
        )
        st.dataframe(
            recommendations.drop("player_id"),
            width="stretch",
            hide_index=True,
            column_config={
                "lineup_gain_ppg": st.column_config.NumberColumn(
                    "Lineup Gain / Game", format="+%.2f"
                ),
                "playoff_lineup_gain": st.column_config.NumberColumn(
                    "Playoff Gain", format="+%.1f"
                ),
                "vor_ros": st.column_config.NumberColumn("VOR", format="%.1f"),
            },
        )

selected_rows = cast(Any, event).selection.rows
if selected_rows:
    selected = ranked.row(selected_rows[0], named=True)
    st.divider()
    st.subheader(f"{selected['player_name']} — player detail")
    owner = selected["fantasy_team"] or "Free agent"
    st.caption(
        f"{selected['position']} · {selected['nfl_team']} · {owner} · "
        f"Position tier {selected['position_tier']}"
    )
    metrics = st.columns(5)
    metrics[0].metric("VOR rank", int(selected["rank"]))
    metrics[1].metric("ROS points", f"{float(selected['ros_points']):.1f}")
    metrics[2].metric("Expected games", f"{float(selected['expected_games']):.1f}")
    metrics[3].metric("Floor", f"{float(selected['ros_p10']):.1f}")
    metrics[4].metric("Upside", f"{float(selected['ros_p90']):.1f}")

    ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
    weekly = (
        player_week_schedule(pl.read_parquet(ros_path), str(selected["player_id"]))
        if ros_path.exists()
        else pl.DataFrame()
    )
    for explanation in explain_ros_player(selected, remaining_weeks=weekly.height):
        st.write(f"• {explanation}")

    if not weekly.is_empty():
        st.caption(
            f"Current-week source: {settings.model.projection_source}. Future weeks use the "
            "season consensus shaped by opponent and schedule. Schedule signal is relative "
            "to this player's own average weekly projection."
        )
        chart = weekly.select("week", "q10", "mean", "q90").to_pandas().set_index("week")
        st.line_chart(chart, width="stretch")
        st.dataframe(weekly, width="stretch", hide_index=True)
else:
    st.info("Select a row to open that player's weekly projection and ranking explanation.")
