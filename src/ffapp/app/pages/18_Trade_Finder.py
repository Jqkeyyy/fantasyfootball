"""Search every roster for fair packages that improve the user's weekly lineup."""

from __future__ import annotations

import json

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.trade_page import (
    build_trade_rosters,
    find_trade_candidates,
    matchup_schedule,
    standings_from_rosters,
    trade_analysis_blocker,
)
from ffapp.app.ui import apply_app_shell
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.sim.trade import TradeProposal, analyze_trade

st.set_page_config(page_title="Trade Finder", layout="wide")
apply_app_shell()
settings = load_settings()
league = select_league()
fmt = parse_league_format(league)
st.title("Trade Finder")
st.caption(f"{league.display_name} · search every roster for trades that improve your lineup")

blocker = trade_analysis_blocker(league, fmt.playoff_week_start)
ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
if blocker or league.league_id is None or not ros_path.exists() or not settings.sleeper_username:
    st.info(blocker or "This league needs Sleeper rosters and ROS projections first.")
    st.stop()

try:
    ros = pl.read_parquet(ros_path)
    players = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    sleeper_to_player = dict(players.select("sleeper_id", "player_id").drop_nulls().iter_rows())
    names = dict(players.select("player_id", "full_name").drop_nulls().iter_rows())
    rosters = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    users = json.loads(
        sleeper.fetch_users(league.league_id, offline=True, settings=settings).read_text()
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    my_team_id = str(resolve_my_roster_id(str(user["user_id"]), rosters))
except (OSError, ValueError, StopIteration) as exc:
    st.error(f"Trade finder data is unavailable: {exc}")
    st.stop()

user_names = {
    str(row.get("user_id")): str(
        (row.get("metadata") or {}).get("team_name")
        or row.get("display_name")
        or row.get("user_id")
    )
    for row in users
}
team_names = {
    str(row["roster_id"]): user_names.get(str(row.get("owner_id")), f"Team {row['roster_id']}")
    for row in rosters
}
roster_players = {
    str(row["roster_id"]): [
        sleeper_to_player[str(player)]
        for player in (row.get("players") or [])
        if str(player) in sleeper_to_player
    ]
    for row in rosters
}
weeks = sorted(int(value) for value in ros["week"].unique().to_list())
teams, vor = build_trade_rosters(ros, roster_players, from_week=min(weeks))
supported = {player.position for team in teams for player in team.players}
supported_fmt = fmt.__class__(
    n_teams=fmt.n_teams,
    starters={key: value for key, value in fmt.starters.items() if key in supported},
    flex_slots=fmt.flex_slots,
    flex_eligible={
        key: [position for position in positions if position in supported]
        for key, positions in fmt.flex_eligible.items()
    },
    bench=fmt.bench,
    ir=fmt.ir,
    playoff_week_start=fmt.playoff_week_start,
    waiver_budget=fmt.waiver_budget,
)

with st.spinner("Searching fair packages across every roster…"):
    candidates = find_trade_candidates(
        teams,
        vor,
        names,
        team_names,
        my_team_id,
        supported_fmt,
        remaining_weeks=weeks,
    )
if candidates.is_empty():
    st.info(
        "No fair package currently improves your projected lineup without badly hurting "
        "the partner."
    )
    st.stop()

minimum_gain = st.slider("Minimum projected lineup gain", 0.0, 15.0, 1.0, 0.5)
mutual_only = st.checkbox("Only show trades that improve both teams", value=True)
shown = candidates.filter(pl.col("your_lineup_gain") >= minimum_gain)
if mutual_only:
    shown = shown.filter(pl.col("mutual_benefit"))
if shown.is_empty():
    st.warning(
        "No candidate clears those filters. Lower the minimum or allow a small partner loss."
    )
    st.stop()

st.dataframe(
    shown.drop("send_ids", "receive_ids", "partner_id", "trade_score", "why"),
    width="stretch",
    hide_index=True,
    column_config={
        "your_lineup_gain": st.column_config.NumberColumn("Your gain", format="%+.1f"),
        "partner_lineup_gain": st.column_config.NumberColumn("Partner gain", format="%+.1f"),
        "value_match": st.column_config.ProgressColumn(
            "Value match", min_value=0.0, max_value=1.0, format="percent"
        ),
    },
)

options = {index: row for index, row in enumerate(shown.iter_rows(named=True))}
selected_index = st.selectbox(
    "Trade to inspect",
    list(options),
    format_func=lambda index: (
        f"Send {options[index]['you_send']} · receive {options[index]['you_receive']} "
        f"from {options[index]['partner']}"
    ),
)
selected = options[selected_index]
st.info(str(selected["why"]))
if st.button("Run full playoff simulation", type="primary"):
    regular = [week for week in weeks if week < fmt.playoff_week_start]
    try:
        with st.spinner("Running before-and-after season simulations…"):
            payloads = {
                week: json.loads(
                    sleeper.fetch_matchups(
                        league.league_id, week, offline=False, settings=settings
                    ).read_text()
                )
                for week in regular
            }
            schedule = matchup_schedule(payloads)
            league_payload = json.loads(
                sleeper.fetch_league(league.league_id, offline=True, settings=settings).read_text()
            )
            initial_wins, initial_points = standings_from_rosters(rosters)
            analysis = analyze_trade(
                teams,
                schedule,
                supported_fmt,
                settings.simulation.correlation,
                TradeProposal(
                    my_team_id,
                    str(selected["partner_id"]),
                    list(selected["send_ids"]),
                    list(selected["receive_ids"]),
                ),
                vor,
                remaining_weeks=weeks,
                playoff_week_start=fmt.playoff_week_start,
                n_playoff_teams=int(
                    (league_payload.get("settings") or {}).get(
                        "playoff_teams", max(2, fmt.n_teams // 2)
                    )
                ),
                season_sims=settings.simulation.season_sims,
                initial_wins=initial_wins,
                initial_points=initial_points,
                rng_seed=20260922,
            )
        result = pl.DataFrame(
            [
                {
                    "team": team_names[side.team_id],
                    "expected_wins_delta": side.delta_expected_wins,
                    "playoff_probability_delta": side.delta_p_playoffs,
                    "title_probability_delta": side.delta_p_title,
                    "ros_value_delta": side.naive_vor_delta,
                }
                for side in (analysis.team_a, analysis.team_b)
            ]
        )
        st.dataframe(result, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Full simulation unavailable: {exc}")
