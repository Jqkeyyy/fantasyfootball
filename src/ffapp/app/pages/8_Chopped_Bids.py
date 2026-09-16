"""FAAB calculator for players released by Sleeper chopped transactions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import polars as pl
import streamlit as st

from ffapp.app.chopped_calculator_page import build_player_values, is_chopped_league
from ffapp.app.data_status import render_league_data_controls
from ffapp.app.league_selector import ordered_leagues
from ffapp.config import load_all_leagues, load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.tools.chopped_bids import (
    build_chopped_bid_board,
    chopped_candidates,
    remaining_faab,
)
from ffapp.tools.waivers import rostered_sleeper_ids

st.set_page_config(page_title="Chopped Bids", layout="wide")

settings = load_settings()
leagues = [league for league in ordered_leagues(load_all_leagues()) if is_chopped_league(league)]
if not leagues:
    st.error("No chopped/elimination league is configured.")
    st.stop()
by_slug = {league.slug: league for league in leagues}
selected_slug = st.sidebar.selectbox(
    "Chopped league",
    options=list(by_slug),
    format_func=lambda slug: by_slug[slug].display_name,
    key="selected_chopped_league_slug",
)
league = by_slug[selected_slug]
render_league_data_controls(league)
fmt = parse_league_format(league)

st.title("Chopped Player Bid Calculator")
st.caption(
    f"{league.display_name} — roster-relative value, opponent demand, remaining FAAB, "
    "and future-chop reserve"
)

if league.league_id is None or settings.sleeper_username is None:
    st.error("Sleeper league ID and username must be configured.")
    st.stop()

projections_path = settings.data_root / "outputs" / league.slug / "projections.parquet"
ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
if not projections_path.exists():
    st.error(
        f"Missing `{projections_path}`. Run `ffapp refresh weekly --league {league.slug} "
        "--no-offline` first."
    )
    st.stop()

weekly_projections = pl.read_parquet(projections_path)
season = cast(int, weekly_projections["season"].max())
week = int(cast(int, weekly_projections.filter(pl.col("season") == season)["week"].max()))
ros_projections = pl.read_parquet(ros_path) if ros_path.exists() else None

with st.sidebar:
    st.subheader("Bid strategy")
    reserve_chops = st.slider(
        "Future chops to reserve for",
        min_value=0,
        max_value=8,
        value=3,
        help="Higher values preserve more FAAB for later eliminated rosters.",
    )
    aggressiveness = st.slider(
        "Aggressiveness",
        min_value=0.5,
        max_value=2.0,
        value=float(settings.waivers.aggressiveness),
        step=0.1,
    )
    refresh_live = st.button("Refresh Sleeper transactions", type="primary")

try:
    rosters: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(
            league.league_id, offline=not refresh_live, settings=settings
        ).read_text()
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
except Exception as exc:
    st.error(f"Could not load Sleeper rosters: {exc}")
    st.stop()

transactions: list[dict[str, Any]] = []
missing_weeks: list[int] = []
for transaction_week in range(1, week + 1):
    should_refresh = refresh_live and transaction_week >= max(1, week - 1)
    try:
        path = sleeper.fetch_transactions(
            league.league_id,
            transaction_week,
            offline=not should_refresh,
            settings=settings,
        )
        rows: list[dict[str, Any]] = json.loads(path.read_text())
        for row in rows:
            row["_week"] = transaction_week
        transactions.extend(rows)
    except Exception:
        missing_weeks.append(transaction_week)

if missing_weeks:
    st.warning(
        "No cached transaction data for week(s) "
        + ", ".join(str(value) for value in missing_weeks)
        + ". Use “Refresh Sleeper transactions”."
    )

candidates = chopped_candidates(transactions, rostered_sleeper_ids(rosters))
if candidates.is_empty():
    st.info("No currently available players from a completed chopped transaction were found.")
    st.stop()


@st.cache_data(show_spinner="Loading player identities...")
def _players_dim() -> pl.DataFrame:
    crosswalk = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=True, settings=settings)
    return mapping.build_players_dim(crosswalk, sleeper_players, mapping.ID_OVERRIDES_PATH)


player_values = build_player_values(
    weekly_projections,
    ros_projections,
    _players_dim(),
    season=season,
    week=week,
)
total_budget = fmt.waiver_budget or 0
my_roster = next(row for row in rosters if int(row["roster_id"]) == my_roster_id)
my_remaining = remaining_faab(my_roster, total_budget)

try:
    board = build_chopped_bid_board(
        candidates,
        player_values,
        rosters,
        my_roster_id,
        fmt,
        current_week=week,
        total_budget=total_budget,
        reserve_chops=reserve_chops,
        aggressiveness=aggressiveness,
    )
except Exception as exc:
    st.error(f"Could not calculate chopped bids: {exc}")
    st.stop()

unresolved = candidates.height - board.height
active_teams = sum(bool(roster.get("players")) for roster in rosters)
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Your remaining FAAB", f"${my_remaining}")
metric_b.metric("Available chopped players", candidates.height)
metric_c.metric("Teams remaining", active_teams)
metric_d.metric("Projection week", week)

if unresolved:
    st.warning(
        f"{unresolved} chopped player(s) lack a current projection and are excluded from bids."
    )
if board.is_empty():
    st.warning("None of the available chopped players has a usable projection.")
    st.stop()

st.dataframe(
    board.select(
        "player_name",
        "position",
        "team",
        "recommendation",
        "recommended_bid",
        "value_bid",
        "market_bid",
        "max_bid",
        "value_basis",
        "lineup_gain_ppg",
        "depth_gain_ppg",
        "projection_ppg",
        "current_week_projection",
        "competing_teams",
        "highest_opponent_estimate",
        "drop_player",
        "chopped_week",
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "recommended_bid": st.column_config.NumberColumn("Recommended", format="$%d"),
        "value_bid": st.column_config.NumberColumn("Value bid", format="$%d"),
        "market_bid": st.column_config.NumberColumn("Market estimate", format="$%d"),
        "max_bid": st.column_config.NumberColumn("Do not exceed", format="$%d"),
        "highest_opponent_estimate": st.column_config.NumberColumn(
            "Highest opponent estimate", format="$%d"
        ),
        "lineup_gain_ppg": st.column_config.NumberColumn("Lineup gain/wk", format="%+.1f"),
        "depth_gain_ppg": st.column_config.NumberColumn("Depth value/wk", format="%+.1f"),
        "projection_ppg": st.column_config.NumberColumn("ROS PPG", format="%.1f"),
        "current_week_projection": st.column_config.NumberColumn("This week", format="%.1f"),
    },
)

st.caption(f"Calculated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} from cached projections.")
with st.expander("How the bid is calculated"):
    st.markdown(
        """
- **Lineup gain** reruns the league's actual lineup optimizer with the player added.
- **Depth value** gives a smaller insurance credit when a player upgrades your weakest bench
  option without entering the optimal lineup immediately; this matters in a no-trade format.
- **Value bid** allocates your remaining FAAB across this chopped cohort according to each
  player's remaining-season lineup value, discounted by your future-chop reserve.
- **Market estimate** is the 75th percentile of the same calculation across surviving opponents,
  using each opponent's roster needs and remaining FAAB.
- **Recommended** is enough to clear that market estimate when it remains below your
  value-based **do not exceed** ceiling. “Pass above max” means the estimated market is richer
  than the player is worth to your roster.

These are decision estimates, not knowledge of hidden waiver bids. Adjust the reserve and
aggressiveness controls to compare strategies before submitting claims in Sleeper.
"""
    )
