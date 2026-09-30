"""Multiweek roster strategy planner."""

from __future__ import annotations

import json
from typing import cast

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.roster_strategy_page import bye_pressure, player_values, roster_strategy
from ffapp.app.weekly_rankings_page import UNAVAILABLE_STATUSES
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format

st.set_page_config(page_title="Roster Strategy", layout="wide")
settings = load_settings()
league = select_league()
fmt = parse_league_format(league)
st.title("Roster Strategy Planner")
st.caption(f"{league.display_name} · four-week strength, bye pressure, playoff value, and targets")

ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
if league.league_id is None or not ros_path.exists() or settings.sleeper_username is None:
    st.error("This page needs a configured Sleeper league and rest-of-season projections.")
    st.stop()

try:
    ros = pl.read_parquet(ros_path)
    current_week = cast(int, ros["week"].min())
    players = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    names = dict(players.select("player_id", "full_name").drop_nulls().iter_rows())
    sleeper_to_player = dict(players.select("sleeper_id", "player_id").drop_nulls().iter_rows())
    rosters = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    users = json.loads(
        sleeper.fetch_users(league.league_id, offline=True, settings=settings).read_text()
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    my_id = str(resolve_my_roster_id(str(user["user_id"]), rosters))
except (OSError, ValueError, StopIteration) as exc:
    st.error(f"Roster strategy data is unavailable: {exc}")
    st.stop()


def canonical(values: object) -> list[str]:
    return (
        [sleeper_to_player[str(value)] for value in values if str(value) in sleeper_to_player]
        if isinstance(values, list)
        else []
    )


roster_players = {str(row["roster_id"]): canonical(row.get("players")) for row in rosters}
owner_names = {}
user_names = {
    str(row.get("user_id")): str(
        (row.get("metadata") or {}).get("team_name")
        or row.get("display_name")
        or row.get("user_id")
    )
    for row in users
}
for row in rosters:
    owner_names[str(row["roster_id"])] = user_names.get(
        str(row.get("owner_id")), f"Team {row['roster_id']}"
    )

starter_counts = dict(fmt.starters)
for positions in fmt.flex_eligible.values():
    for position in positions:
        starter_counts[position] = starter_counts.get(position, 0) + 1
values = player_values(
    ros, names, current_week=current_week, playoff_week_start=fmt.playoff_week_start
)
unavailable_ids = set(
    players.filter(pl.col("injury_status").fill_null("").is_in(list(UNAVAILABLE_STATUSES)))[
        "player_id"
    ].to_list()
)
health, targets, chips = roster_strategy(
    values,
    roster_players,
    my_id,
    starter_counts,
    unavailable_player_ids=unavailable_ids,
)

if health.is_empty():
    st.warning("No remaining projection data is available.")
    st.stop()

priority = health.filter(pl.col("status") == "Priority need").height
strengths = health.filter(pl.col("status") == "Strength").height
playoff_total = values.filter(pl.col("player_id").is_in(roster_players[my_id]))[
    "playoff_points"
].sum()
a, b, c = st.columns(3)
a.metric("Priority position needs", str(priority))
b.metric("Position strengths", str(strengths))
c.metric("Projected playoff points", f"{float(playoff_total or 0):.1f}")

st.subheader("Position plan")
st.dataframe(
    health.select(
        "position",
        "status",
        "starter_value",
        "league_median",
        "edge",
        "roster_count",
        "starters_needed",
        "surplus",
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "starter_value": st.column_config.NumberColumn("Your next 4", format="%.1f"),
        "league_median": st.column_config.NumberColumn("League median", format="%.1f"),
        "edge": st.column_config.NumberColumn("Edge", format="%+.1f"),
    },
)

st.subheader("Bye and schedule pressure")
pressure = bye_pressure(
    ros,
    roster_players[my_id],
    current_week=current_week,
    final_week=cast(int, ros["week"].max()),
)
st.bar_chart(pressure.to_pandas().set_index("week")[["players_without_game"]])
worst = pressure.sort("players_without_game", descending=True).head(3)
st.caption(
    "Highest pressure: "
    + ", ".join(
        f"Week {row['week']} ({row['players_without_game']} without a game)"
        for row in worst.iter_rows(named=True)
    )
)

st.subheader("Best ways to improve")
if targets.is_empty():
    st.success("No obvious external upgrade was found for your weakest positions.")
else:
    target_view = targets.with_columns(
        pl.col("owner").replace_strict(owner_names, default=pl.col("owner"))
    )
    st.dataframe(
        target_view.select(
            "path",
            "player_name",
            "position",
            "team",
            "owner",
            "next_4_points",
            "playoff_points",
            "ros_points",
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("Trade chips from roster surplus")
if chips.is_empty():
    st.info("Your roster has no clear surplus-position player to shop right now.")
else:
    st.dataframe(
        chips.select(
            "player_name", "position", "team", "next_4_points", "playoff_points", "ros_points"
        ),
        width="stretch",
        hide_index=True,
    )
    st.link_button("Build a trade", "/Trade_Analyzer", icon=":material/handshake:")
