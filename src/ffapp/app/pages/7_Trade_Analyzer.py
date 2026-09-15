"""Interactive, lineup-aware rest-of-season trade analyzer."""

from __future__ import annotations

import json

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.trade_page import build_trade_rosters, matchup_schedule
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.sim.trade import TradeProposal, analyze_trade

st.set_page_config(page_title="Trade Analyzer", layout="wide")
settings = load_settings()
league = select_league()
fmt = parse_league_format(league)
st.title("Trade Analyzer")
st.caption(f"{league.display_name} — lineup-aware rest-of-season impact for both teams")

ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
if league.league_id is None or not ros_path.exists():
    st.error("This league needs cached rosters and rest-of-season projections first.")
    st.stop()

try:
    user = json.loads(
        sleeper.fetch_user(
            settings.sleeper_username or "", offline=True, settings=settings
        ).read_text()
    )
    rosters: list[dict[str, object]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    users: list[dict[str, object]] = json.loads(
        sleeper.fetch_users(league.league_id, offline=True, settings=settings).read_text()
    )
    my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
    players_dim = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
except Exception as exc:
    st.error(f"Could not load cached league data: {exc}")
    st.stop()

sleeper_to_player = {
    str(row["sleeper_id"]): str(row["player_id"])
    for row in players_dim.select("sleeper_id", "player_id").drop_nulls().iter_rows(named=True)
}
name_by_player = {
    str(row["player_id"]): str(row["full_name"])
    for row in players_dim.select("player_id", "full_name").drop_nulls().iter_rows(named=True)
}


def _team_name(row: dict[str, object]) -> str:
    metadata = row.get("metadata")
    custom = metadata.get("team_name") if isinstance(metadata, dict) else None
    return str(custom or row.get("display_name") or row.get("user_id"))


user_names = {str(row.get("user_id")): _team_name(row) for row in users}
roster_players: dict[str, list[str]] = {}
team_names: dict[str, str] = {}
for roster in rosters:
    team_id = str(roster["roster_id"])
    raw_players = roster.get("players")
    sleeper_ids = raw_players if isinstance(raw_players, list) else []
    roster_players[team_id] = [
        sleeper_to_player[str(player)] for player in sleeper_ids if str(player) in sleeper_to_player
    ]
    team_names[team_id] = user_names.get(str(roster.get("owner_id")), f"Team {team_id}")

ros = pl.read_parquet(ros_path)
available_weeks = sorted(int(week) for week in ros["week"].unique().to_list())
if not available_weeks:
    st.error("The rest-of-season projection file has no weeks.")
    st.stop()
from_week = min(available_weeks)
teams, vor = build_trade_rosters(ros, roster_players, from_week=from_week)
supported_positions = {player.position for team in teams for player in team.players}
supported_fmt = fmt.__class__(
    n_teams=fmt.n_teams,
    starters={key: value for key, value in fmt.starters.items() if key in supported_positions},
    flex_slots=fmt.flex_slots,
    flex_eligible={
        key: [position for position in values if position in supported_positions]
        for key, values in fmt.flex_eligible.items()
    },
    bench=fmt.bench,
    ir=fmt.ir,
    playoff_week_start=fmt.playoff_week_start,
    waiver_budget=fmt.waiver_budget,
)

my_team_id = str(my_roster_id)
opponent_ids = [team_id for team_id in roster_players if team_id != my_team_id]
opponent_id = st.selectbox(
    "Trade partner", opponent_ids, format_func=lambda value: team_names[value]
)


def _player_options(team_id: str) -> list[str]:
    projected = {
        player.player_id for team in teams if team.team_id == team_id for player in team.players
    }
    return sorted(projected, key=lambda player_id: name_by_player.get(player_id, player_id))


give = st.multiselect(
    "You send",
    _player_options(my_team_id),
    format_func=lambda value: name_by_player.get(value, value),
)
receive = st.multiselect(
    "You receive",
    _player_options(opponent_id),
    format_func=lambda value: name_by_player.get(value, value),
)

if st.button("Simulate trade", type="primary", disabled=not give or not receive):
    regular_weeks = [week for week in available_weeks if week < fmt.playoff_week_start]
    try:
        with st.spinner("Fetching league matchups and simulating both rosters..."):
            matchup_payloads = {
                week: json.loads(
                    sleeper.fetch_matchups(
                        league.league_id, week, offline=False, settings=settings
                    ).read_text()
                )
                for week in regular_weeks
            }
            schedule = matchup_schedule(matchup_payloads)
            league_payload = json.loads(
                sleeper.fetch_league(league.league_id, offline=True, settings=settings).read_text()
            )
            league_settings = league_payload.get("settings", {})
            analysis = analyze_trade(
                teams,
                schedule,
                supported_fmt,
                settings.simulation.correlation,
                TradeProposal(my_team_id, opponent_id, give, receive),
                vor,
                remaining_weeks=available_weeks,
                playoff_week_start=fmt.playoff_week_start,
                n_playoff_teams=int(league_settings.get("playoff_teams", max(2, fmt.n_teams // 2))),
                season_sims=settings.simulation.season_sims,
                rng_seed=20260915,
            )
        result = pl.DataFrame(
            [
                {
                    "team": team_names[side.team_id],
                    "expected_wins_delta": side.delta_expected_wins,
                    "playoff_probability_delta": side.delta_p_playoffs,
                    "title_probability_delta": side.delta_p_title,
                    "approx_ros_vor_delta": side.naive_vor_delta,
                    "roster_change": side.summary(),
                }
                for side in (analysis.team_a, analysis.team_b)
            ]
        )
        st.dataframe(result, width="stretch", hide_index=True)
        if analysis.team_a.delta_p_playoffs > 0 and analysis.team_b.delta_p_playoffs > 0:
            st.success("The simulation projects a playoff-probability gain for both teams.")
        else:
            st.info(
                "At least one side loses playoff probability; adjust the package before "
                "proposing it."
            )
    except Exception as exc:
        st.error(f"Trade simulation unavailable: {exc}")
