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
from ffapp.sim.chopped_survival import simulate_chopped_survival
from ffapp.tools.chopped_bids import (
    build_chopped_bid_board,
    chopped_candidates,
    remaining_faab,
)
from ffapp.tools.waiver_history import (
    build_manager_bid_profiles,
    extract_waiver_outcomes,
    profile_multipliers,
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

try:
    league_users: list[dict[str, Any]] = json.loads(
        sleeper.fetch_users(
            league.league_id, offline=not refresh_live, settings=settings
        ).read_text()
    )
except Exception:
    league_users = []
display_name_by_user = {
    str(row.get("user_id")): str(row.get("display_name") or row.get("user_id"))
    for row in league_users
}
manager_by_roster = {
    int(roster["roster_id"]): display_name_by_user.get(
        str(roster.get("owner_id")), str(roster.get("owner_id") or f"Roster {roster['roster_id']}")
    )
    for roster in rosters
}

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
waiver_outcomes = extract_waiver_outcomes(transactions)
bid_profiles = build_manager_bid_profiles(
    waiver_outcomes, [int(roster["roster_id"]) for roster in rosters]
)
survival_table, survival_impacts = simulate_chopped_survival(
    rosters,
    player_values,
    my_roster_id,
    fmt,
    candidates["sleeper_id"].to_list(),
    n_sims=settings.simulation.week_sims,
)

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
        opponent_aggression=profile_multipliers(bid_profiles),
    )
except Exception as exc:
    st.error(f"Could not calculate chopped bids: {exc}")
    st.stop()
board = board.join(survival_impacts, on="sleeper_id", how="left").with_columns(
    (pl.col("survival_probability_gain") * 100.0).alias("survival_gain_pct")
)

unresolved = candidates.height - board.height
active_teams = sum(bool(roster.get("players")) for roster in rosters)
my_survival = survival_table.filter(pl.col("roster_id") == my_roster_id)[
    "survival_probability"
].item()
metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
metric_a.metric("Your remaining FAAB", f"${my_remaining}")
metric_b.metric("Available chopped players", candidates.height)
metric_c.metric("Teams remaining", active_teams)
metric_d.metric("Projection week", week)
metric_e.metric("Survive this chop", f"{my_survival:.0%}")

with st.expander("This week's elimination risk"):
    survival_display = survival_table.with_columns(
        pl.col("roster_id")
        .replace_strict(manager_by_roster, default=None, return_dtype=pl.String)
        .alias("manager")
    ).select(
        "manager",
        "projected_score",
        "elimination_probability",
        "survival_probability",
    )
    st.dataframe(
        survival_display,
        width="stretch",
        hide_index=True,
        column_config={
            "projected_score": st.column_config.NumberColumn("Projected lineup", format="%.1f"),
            "elimination_probability": st.column_config.ProgressColumn(
                "Chop risk", min_value=0.0, max_value=1.0, format="percent"
            ),
            "survival_probability": st.column_config.ProgressColumn(
                "Survival", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
    st.caption(
        "Monte Carlo estimate from current optimized lineups and projection uncertainty; "
        "it is decision support, not a guarantee."
    )

with st.expander("League bidding tendencies"):
    if waiver_outcomes.is_empty():
        st.caption("No completed FAAB waivers are available yet; neutral manager behavior is used.")
    else:
        profile_display = bid_profiles.with_columns(
            pl.col("roster_id")
            .replace_strict(manager_by_roster, default=None, return_dtype=pl.String)
            .alias("manager")
        ).select(
            "manager",
            "n_winning_bids",
            "mean_bid",
            "median_bid",
            "p75_bid",
            "max_bid",
            "aggression_multiplier",
        )
        st.dataframe(
            profile_display,
            width="stretch",
            hide_index=True,
            column_config={
                "mean_bid": st.column_config.NumberColumn("Average win", format="$%.1f"),
                "median_bid": st.column_config.NumberColumn("Median win", format="$%.1f"),
                "p75_bid": st.column_config.NumberColumn("75th percentile", format="$%.1f"),
                "max_bid": st.column_config.NumberColumn("Largest win", format="$%d"),
                "aggression_multiplier": st.column_config.NumberColumn(
                    "Learned multiplier", format="%.2fx"
                ),
            },
        )

evaluation_path = (
    settings.data_root / "outputs" / league.slug / "chopped_notifications" / "evaluation.parquet"
)
with st.expander("Recommendation results"):
    if not evaluation_path.exists():
        st.caption(
            "Results will appear after a Discord recommendation is followed by a completed "
            "Sleeper waiver award."
        )
    else:
        recommendation_results = pl.read_parquet(evaluation_path)
        if recommendation_results.is_empty():
            st.caption("No recommended player has reached a completed waiver result yet.")
        else:
            result_a, result_b = st.columns(2)
            mean_error = float(
                recommendation_results.select(pl.col("recommendation_error").mean()).item()
                or 0.0
            )
            winning_rate = float(
                recommendation_results.select(
                    pl.col("recommended_met_winning_bid").mean()
                ).item()
                or 0.0
            )
            result_a.metric(
                "Mean bid error",
                f"${mean_error:+.1f}",
                help=(
                    "Recommended minus actual winning bid; positive means the recommendation "
                    "was higher."
                ),
            )
            result_b.metric(
                "Met winning bid",
                f"{winning_rate:.0%}",
            )
            st.dataframe(
                recommendation_results,
                width="stretch",
                hide_index=True,
                column_config={
                    "recommended_bid": st.column_config.NumberColumn("Recommended", format="$%d"),
                    "max_bid": st.column_config.NumberColumn("Max", format="$%d"),
                    "actual_winning_bid": st.column_config.NumberColumn(
                        "Winning bid", format="$%d"
                    ),
                    "recommendation_error": st.column_config.NumberColumn(
                        "Bid error", format="$%d"
                    ),
                    "recommended_met_winning_bid": st.column_config.CheckboxColumn(
                        "Would meet winning bid"
                    ),
                },
            )

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
        "conservative_bid",
        "recommended_bid",
        "aggressive_bid",
        "recommended_win_probability",
        "survival_urgency",
        "survival_probability_after",
        "survival_gain_pct",
        "faab_after_recommended",
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
        "conservative_bid": st.column_config.NumberColumn("Conservative", format="$%d"),
        "recommended_bid": st.column_config.NumberColumn("Recommended", format="$%d"),
        "aggressive_bid": st.column_config.NumberColumn("Aggressive", format="$%d"),
        "recommended_win_probability": st.column_config.ProgressColumn(
            "Est. win chance", min_value=0.0, max_value=1.0, format="percent"
        ),
        "survival_urgency": st.column_config.ProgressColumn(
            "Survival urgency", min_value=0.0, max_value=1.0, format="percent"
        ),
        "survival_probability_after": st.column_config.ProgressColumn(
            "Survival after add", min_value=0.0, max_value=1.0, format="percent"
        ),
        "survival_gain_pct": st.column_config.NumberColumn("Survival gain", format="%+.1f%%"),
        "faab_after_recommended": st.column_config.NumberColumn("FAAB after bid", format="$%d"),
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
- **Conservative** and **aggressive** bracket the recommended bid while staying under the same
  value ceiling; aggressive aims to clear the strongest estimated competitor.
- **Estimated win chance** compares the bid with each opponent's roster-specific estimate.
- **Survival urgency** rises as your optimized lineup falls behind the active rosters and adds a
  small premium to the recommended bid.
- **Recommended** is enough to clear that market estimate when it remains below your
  value-based **do not exceed** ceiling. “Pass above max” means the estimated market is richer
  than the player is worth to your roster.

These are decision estimates, not knowledge of hidden waiver bids. Adjust the reserve and
aggressiveness controls to compare strategies before submitting claims in Sleeper.
"""
    )
