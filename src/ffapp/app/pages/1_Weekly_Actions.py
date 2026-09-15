"""Weekly action cockpit: lineup, start/sit simulation, and waiver upgrades."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.weekly_actions_page import (
    explain_player,
    projection_supported_format,
    recommended_lineup,
    sim_players,
    streaming_recommendations,
    waiver_recommendations,
)
from ffapp.app.weekly_rankings_page import build_weekly_rankings, load_projections
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.ingest import rankings as rankings_ingest
from ffapp.league_format import parse_league_format
from ffapp.projections.aggregate import apply_league_scoring
from ffapp.sim.startsit import evaluate_start_sit
from ffapp.tools.pipeline_health import health_table, inspect_weekly_pipeline

st.set_page_config(page_title="Weekly Actions", layout="wide")

settings = load_settings()
league = select_league()
fmt = parse_league_format(league)

st.title("Weekly Actions")
st.caption(f"{league.display_name} — lineup decisions and roster-relative upgrades")

projections_path = settings.data_root / "outputs" / league.slug / "projections.parquet"
features_path = settings.data_root / "features" / "player_week_features.parquet"
schedule_path = settings.data_root / "interim" / "schedule.parquet"
for required in (projections_path, features_path, schedule_path):
    if not required.exists():
        st.error(f"Missing `{required}`. Run the weekly projection pipeline first.")
        st.stop()


@st.cache_data(show_spinner="Loading player identities...")
def _players_dim() -> pl.DataFrame:
    crosswalk = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=True, settings=settings)
    return mapping.build_players_dim(crosswalk, sleeper_players, mapping.ID_OVERRIDES_PATH)


all_projections = load_projections(projections_path)
features = pl.read_parquet(features_path)
schedule = pl.read_parquet(schedule_path)
players_dim = _players_dim()

weeks = all_projections.select("season", "week").unique().sort(
    ["season", "week"], descending=True
)
week_options = [(row["season"], row["week"]) for row in weeks.to_dicts()]
with st.sidebar:
    season, week = st.selectbox(
        "Season / Week", week_options, format_func=lambda value: f"{value[0]} week {value[1]}"
    )

pipeline_health = inspect_weekly_pipeline(settings, league.slug, season, week)
if pipeline_health.status == "failed":
    st.error("Weekly pipeline health: failed. Recommendations may be unavailable.")
elif pipeline_health.status == "degraded":
    st.warning("Weekly pipeline health: degraded. Review the checks before acting.")
else:
    st.success("Weekly pipeline health: healthy.")
with st.expander("Pipeline health details"):
    st.dataframe(health_table(pipeline_health), width="stretch", hide_index=True)

if league.league_id is None or settings.sleeper_username is None:
    st.error("The primary league ID and Sleeper username must be configured.")
    st.stop()

try:
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    rosters: list[dict[str, object]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
    my_roster = next(row for row in rosters if row.get("roster_id") == my_roster_id)
except Exception as exc:
    st.error(f"Could not load your cached Sleeper roster: {exc}")
    st.stop()

sleeper_to_player = {
    row["sleeper_id"]: row["player_id"]
    for row in players_dim.select("sleeper_id", "player_id").drop_nulls().iter_rows(named=True)
}


def _string_list(values: object) -> list[str]:
    return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []


def _canonical_ids(values: object) -> set[str]:
    ids = _string_list(values)
    return {sleeper_to_player[value] for value in ids if value in sleeper_to_player}


my_roster_ids = _canonical_ids(my_roster.get("players"))
current_starter_ids = _canonical_ids(my_roster.get("starters"))
all_rostered_sleeper_ids = {
    player_id
    for roster in rosters
    for player_id in _string_list(roster.get("players"))
}
all_rostered_ids = {
    sleeper_to_player[player_id]
    for player_id in all_rostered_sleeper_ids
    if player_id in sleeper_to_player
}
opponent_roster_ids = [
    _canonical_ids(roster.get("players"))
    for roster in rosters
    if roster.get("roster_id") != my_roster_id
]

rankings = build_weekly_rankings(
    all_projections,
    features,
    schedule,
    players_dim,
    season=season,
    week=week,
    my_roster_ids=my_roster_ids,
    rostered_ids=all_rostered_ids,
)
if rankings.is_empty():
    st.warning("No ranked players are available for this week.")
    st.stop()

unprojected_roster = rankings.filter(
    pl.col("player_id").is_in(list(my_roster_ids)) & pl.col("proj_mean").is_null()
)
if not unprojected_roster.is_empty():
    st.warning(
        "Excluded rostered players with no weekly projection: "
        + ", ".join(unprojected_roster["player_name"].drop_nulls().cast(pl.String).to_list())
    )

lineup = recommended_lineup(rankings, my_roster_ids, current_starter_ids, fmt)
supported_positions = set(rankings["position"].unique().to_list())
unsupported_starters = sorted(set(fmt.starters) - supported_positions)

st.subheader("Recommended lineup")
recommended_total = float(lineup["projected_points"].sum()) if not lineup.is_empty() else 0.0
current_rows = rankings.filter(pl.col("player_id").is_in(list(current_starter_ids)))
current_total = float(current_rows["proj_mean"].sum()) if not current_rows.is_empty() else 0.0
changed = lineup.filter(~pl.col("currently_starting"))
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Recommended points", f"{recommended_total:.1f}")
metric_b.metric("Current supported starters", f"{current_total:.1f}")
metric_c.metric("Projected improvement", f"{recommended_total - current_total:+.1f}")
if unsupported_starters:
    st.info(
        "The lineup optimizer handles skill players here; use the streaming section below for "
        + ", ".join(unsupported_starters)
        + "."
    )
if changed.is_empty():
    st.success("Your supported starters already match the projection-optimal lineup.")
else:
    st.warning(
        "Recommended changes: "
        + ", ".join(changed["player_name"].cast(pl.String).to_list())
    )
st.dataframe(
    lineup.drop("player_id"),
    width="stretch",
    hide_index=True,
    column_config={
        "projected_points": st.column_config.NumberColumn(format="%.1f"),
        "floor": st.column_config.NumberColumn(format="%.1f"),
        "ceiling": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.subheader("Why this player?")
explainable = rankings.filter(pl.col("player_id").is_in(list(my_roster_ids)))
if explainable.is_empty():
    explainable = rankings.head(25)
explain_rows = {str(row["player_id"]): row for row in explainable.iter_rows(named=True)}
selected_player = st.selectbox(
    "Player",
    options=list(explain_rows),
    format_func=lambda player_id: str(explain_rows[player_id]["player_name"]),
)
for explanation in explain_player(explain_rows[selected_player]):
    st.write(f"- {explanation}")

st.subheader("Matchup-aware start/sit")
st.caption("Fetches the current Sleeper matchup when you run it; no automatic network polling.")
if st.button("Run win-probability simulation"):
    try:
        matchup_rows: list[dict[str, object]] = json.loads(
            sleeper.fetch_matchups(
                league.league_id, week, offline=False, settings=settings
            ).read_text()
        )
        mine = next(row for row in matchup_rows if row.get("roster_id") == my_roster_id)
        matchup_id = mine.get("matchup_id")
        opponent_matchup = next(
            row
            for row in matchup_rows
            if row.get("matchup_id") == matchup_id and row.get("roster_id") != my_roster_id
        )
        opponent_roster_id = opponent_matchup.get("roster_id")
        opponent_roster = next(row for row in rosters if row.get("roster_id") == opponent_roster_id)
        opponent_ids = _canonical_ids(opponent_roster.get("players"))
        my_sim = sim_players(rankings, my_roster_ids)
        opponent_sim = sim_players(rankings, opponent_ids)
        supported_fmt = projection_supported_format(fmt, supported_positions)
        result = evaluate_start_sit(
            my_sim,
            opponent_sim,
            supported_fmt,
            settings.simulation.correlation,
            week_sims=settings.simulation.week_sims,
            rng=np.random.default_rng(2026_00 + week),
        )
        names = dict(
            zip(rankings["player_id"].to_list(), rankings["player_name"].to_list(), strict=True)
        )
        candidates = pl.DataFrame(
            [
                {
                    "decision": (
                        "Projection-optimal lineup"
                        if candidate.swapped_in is None
                        else f"Start {names.get(candidate.swapped_in, candidate.swapped_in)} over "
                        f"{names.get(candidate.swapped_out, candidate.swapped_out)}"
                    ),
                    "projected_points_delta": candidate.delta_projected_points,
                    "win_probability": candidate.p_win,
                    "win_probability_delta": candidate.delta_p_win,
                }
                for candidate in result.candidates
            ]
        )
        st.dataframe(
            candidates,
            width="stretch",
            hide_index=True,
            column_config={
                "win_probability": st.column_config.NumberColumn(format="%.1%%"),
                "win_probability_delta": st.column_config.NumberColumn(format="%+.1%%"),
                "projected_points_delta": st.column_config.NumberColumn(format="%+.1f"),
            },
        )
    except Exception as exc:
        st.error(f"Matchup simulation unavailable: {exc}")

st.subheader("Waiver upgrades")
waiver_budget = fmt.waiver_budget or 0
roster_settings = my_roster.get("settings")
used_budget = (
    int(roster_settings.get("waiver_budget_used", 0)) if isinstance(roster_settings, dict) else 0
)
remaining_budget = max(0, waiver_budget - used_budget)
waivers = waiver_recommendations(
    rankings,
    my_roster_ids,
    fmt,
    current_week=week,
    remaining_budget=remaining_budget,
    playoff_weight=settings.waivers.playoff_weight,
    aggressiveness=settings.waivers.aggressiveness,
    ros_projections=(
        pl.read_parquet(
            settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
        )
        if (settings.data_root / "outputs" / league.slug / "projections_ros.parquet").exists()
        else None
    ),
    opponent_roster_ids=opponent_roster_ids,
)
if waivers.is_empty():
    st.success("No available skill player projects as a starting-lineup upgrade this week.")
else:
    st.caption(f"Remaining waiver budget: {remaining_budget}")
    st.dataframe(
        waivers.select(
            "player_name",
            "position",
            "team",
            "opponent",
            "proj_mean",
            "value_added_per_week",
            "suggested_bid",
            "competing_teams",
            "max_opponent_need",
            "projection_basis",
            "drop_player",
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "proj_mean": st.column_config.NumberColumn(format="%.1f"),
            "value_added_per_week": st.column_config.NumberColumn(format="%+.1f"),
        },
    )

st.subheader("Kicker and defense streamers")
try:
    espn_payload = json.loads(
        rankings_ingest.fetch_espn(season, offline=True, settings=settings).read_text()
    )
    weekly_stats = rankings_ingest.normalize_espn_weekly(
        espn_payload, season=season, week=week
    )
    weekly_points = apply_league_scoring(
        weekly_stats, league.league_cache["scoring_settings"]
    )
    streamers = streaming_recommendations(
        weekly_points,
        players_dim,
        schedule,
        all_rostered_sleeper_ids,
        season=season,
        week=week,
    )
    if streamers.is_empty():
        st.info("No available kicker or defense projection was found in the cached weekly feed.")
    else:
        st.dataframe(
            streamers,
            width="stretch",
            hide_index=True,
            column_config={"points": st.column_config.NumberColumn(format="%.1f")},
        )
except Exception as exc:
    st.warning(f"Kicker/defense streamers unavailable: {exc}")
