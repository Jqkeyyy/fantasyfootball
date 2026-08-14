"""Schedule grid Streamlit page (SPEC.md §14.5; task 2.8).

Third page in SPEC §15's own build order, under `app/pages/` per
Streamlit's own multipage convention (matching `2_Weekly_Rankings.py`'s
own precedent, task 1.19). Reads the already-built interim tables
(`schedule.parquet`, `defense_position_allowed.parquet`,
`player_week_features.parquet`) rather than recomputing anything
model-level on page load -- SPEC §15's own "fast to load" constraint,
same as every other page. The real math (SOS sums, the grid pivot,
matchup-detail breakdown) lives in `tools.sos`; roster-highlight
resolution and heatmap styling live in `app.schedule_grid_page`; this
script is thin `st.*` glue, verified by actually running it (CLAUDE.md's
UI rule), not a pytest test.

Three tabs, matching SPEC §14.5's own three deliverables exactly:
positional SOS (full season / rest of season / fantasy playoffs),
the schedule-grid heatmap (with a bye-aware, confidence-greyed colour
scale and an own-roster highlight toggle), and a matchup detail view for
one real player-week -- SPEC's own "required honesty" clause applied
literally: matchup grade is shown as a plain breakdown table, never a
colour badge, always next to that player's own real usage trend.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`, then
open "Schedule Grid" from the sidebar.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import streamlit as st

from ffapp.app.schedule_grid_page import resolve_my_teams, style_schedule_grid
from ffapp.config import load_primary_league, load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.features.opponent import ALL_POSITION_GROUPS
from ffapp.ids import mapping as ids_mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.tools import sos

st.set_page_config(page_title="Schedule Grid", layout="wide")


@st.cache_data
def _players_dim_cached() -> pl.DataFrame:
    settings = load_settings()
    crosswalk = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=True, settings=settings)
    return ids_mapping.build_players_dim(crosswalk, sleeper_players, ids_mapping.ID_OVERRIDES_PATH)


@st.cache_data(show_spinner="Resolving your roster from Sleeper...")
def _load_my_roster_ids(league_id: str, sleeper_username: str | None) -> set[str]:
    """Real canonical `player_id`s on my own roster -- same offline-cache
    resolution `2_Weekly_Rankings.py` already established, trimmed to
    just the one set this page's own roster-highlight toggle needs."""
    if sleeper_username is None:
        return set()
    settings = load_settings()
    players_dim = _players_dim_cached()
    sleeper_to_player = dict(
        zip(players_dim["sleeper_id"].to_list(), players_dim["player_id"].to_list(), strict=True)
    )
    try:
        rosters: list[dict[str, Any]] = json.loads(
            sleeper.fetch_rosters(league_id, offline=True, settings=settings).read_text()
        )
        user = json.loads(
            sleeper.fetch_user(sleeper_username, offline=True, settings=settings).read_text()
        )
        my_roster_id = resolve_my_roster_id(user["user_id"], rosters)
        my_roster = next(r for r in rosters if r["roster_id"] == my_roster_id)
        my_sleeper_ids = my_roster.get("players") or []
        return {sleeper_to_player[sid] for sid in my_sleeper_ids if sid in sleeper_to_player}
    except Exception:  # a roster we can't resolve just means no highlight overlay
        return set()


settings = load_settings()
league = load_primary_league()
league_format = parse_league_format(league)

st.title("Schedule Grid")
st.caption(league.display_name)

schedule_path = settings.data_root / "interim" / "schedule.parquet"
dpa_path = settings.data_root / "interim" / "defense_position_allowed.parquet"
features_path = settings.data_root / "features" / "player_week_features.parquet"
missing = [p for p in (schedule_path, dpa_path, features_path) if not p.exists()]
if missing:
    st.error(
        "Missing required table(s): "
        + ", ".join(str(p) for p in missing)
        + ". See HANDOFF.md for the build steps."
    )
    st.stop()

schedule = pl.read_parquet(schedule_path)
defense_position_allowed = pl.read_parquet(dpa_path)
features = pl.read_parquet(features_path)

available_seasons = sorted(
    schedule.filter(pl.col("season_type") == "REG")["season"].unique().to_list(), reverse=True
)
if not available_seasons:
    st.warning("schedule.parquet has no real regular-season rows.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", options=available_seasons)
    position_group = st.selectbox("Position group", options=ALL_POSITION_GROUPS)
    real_weeks = sos.full_season_weeks(schedule, season=season)
    as_of_week = st.number_input(
        "Current week (for rest-of-season SOS)",
        min_value=0,
        max_value=max(real_weeks),
        value=0,
        step=1,
    )
    highlight_my_roster = st.checkbox("Highlight my roster's teams", value=False)

my_teams: set[str] = set()
if highlight_my_roster and league.league_id is not None:
    my_roster_ids = _load_my_roster_ids(league.league_id, settings.sleeper_username)
    my_teams = resolve_my_teams(my_roster_ids, _players_dim_cached())
    if not my_teams:
        st.sidebar.caption("Could not resolve any real teams for your roster.")

team_schedule = sos.team_position_group_schedule(
    defense_position_allowed, schedule, season=season, position_group=position_group
)
confidence_thresholds = sos.position_group_confidence_thresholds(
    defense_position_allowed, season=season
)
position_confidence_threshold = confidence_thresholds.get(position_group, 0.0)

sos_tab, grid_tab, detail_tab = st.tabs(["Positional SOS", "Schedule Grid", "Matchup Detail"])

with sos_tab:
    st.caption(
        f"{position_group} -- higher = an easier real schedule "
        "(the opponent allows more EPA at this position group)."
    )

    ranges = {
        "full_season": sos.full_season_weeks(schedule, season=season),
        "rest_of_season": sos.rest_of_season_weeks(schedule, season=season, as_of_week=as_of_week),
        "fantasy_playoffs": sos.playoff_weeks(
            schedule, season=season, playoff_week_start=league_format.playoff_week_start
        ),
    }

    sos_tables = []
    for label, weeks in ranges.items():
        if not weeks:
            continue
        table = sos.positional_sos(team_schedule, weeks=weeks).select(
            "team",
            pl.col("sos_value").alias(f"sos_{label}"),
            pl.col("confident").alias(f"confident_{label}"),
        )
        sos_tables.append(table)

    if not sos_tables:
        st.info("No real weeks available for any SOS range with the current filters.")
    else:
        combined = sos_tables[0]
        for table in sos_tables[1:]:
            combined = combined.join(table, on="team", how="full", coalesce=True)
        if my_teams:
            combined = combined.with_columns(
                pl.when(pl.col("team").is_in(list(my_teams)))
                .then(pl.lit("* "))
                .otherwise(pl.lit(""))
                .add(pl.col("team"))
                .alias("team")
            )
        st.dataframe(
            combined.sort(
                "sos_full_season" if "sos_full_season" in combined.columns else "team",
                descending=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "`confident_*` is false when most of the range's own real weeks fall "
            f"below {position_group}'s own real bottom-quartile weekly sample size "
            f"({position_confidence_threshold:.0f} plays) -- treat those rows as "
            "noisier than they look."
        )

with grid_tab:
    st.caption(
        "Blue = easier real matchup, red = harder, gray/blocked = a real bye "
        f"or fewer than {position_group}'s own real bottom-quartile weekly sample "
        f"size ({position_confidence_threshold:.0f} plays) behind the estimate."
    )
    grid = sos.schedule_grid(team_schedule)
    confidence = sos.schedule_grid_confidence(team_schedule)
    if my_teams:
        marker = (
            pl.when(pl.col("team").is_in(list(my_teams)))
            .then(pl.lit("* "))
            .otherwise(pl.lit(""))
            .add(pl.col("team"))
            .alias("team")
        )
        grid = grid.with_columns(marker)
        confidence = confidence.with_columns(marker)
    if grid.is_empty():
        st.info("No real schedule data for this season/position group.")
    else:
        st.dataframe(style_schedule_grid(grid, confidence), use_container_width=True, height=600)

with detail_tab:
    st.caption(
        "Matchup grade shown alongside usage trend, never alone -- "
        "SPEC §14.5's own required honesty."
    )
    detail_weeks = sorted(features.filter(pl.col("season") == season)["week"].unique().to_list())
    if not detail_weeks:
        st.info("No real player-week features for this season.")
    else:
        detail_week = st.selectbox("Week", options=detail_weeks, key="detail_week")
        detail_position_options = ["QB", "RB", "WR", "TE"]
        detail_position = st.selectbox("Position", options=detail_position_options)

        week_rows = features.filter(
            (pl.col("season") == season)
            & (pl.col("week") == detail_week)
            & (pl.col("position") == detail_position)
        )
        players_dim = _players_dim_cached()
        named = week_rows.join(
            players_dim.select("player_id", pl.col("full_name").alias("player_name")),
            on="player_id",
            how="left",
        ).sort("player_name")

        if named.is_empty():
            st.info("No real players at this position for this season/week.")
        else:
            names = named["player_name"].to_list()
            ids = named["player_id"].to_list()
            options = dict(zip(names, ids, strict=True))
            player_name = st.selectbox("Player", options=list(options.keys()))
            row = named.filter(pl.col("player_id") == options[player_name]).to_dicts()[0]

            usage_col, matchup_col = st.columns(2)
            with usage_col:
                st.subheader("Usage trend")
                snap_trend = row.get("snap_pct_trend")
                st.metric(
                    "Snap % trend (recent vs longer trailing window)",
                    f"{snap_trend:+.1%}" if snap_trend is not None else "n/a",
                )
                st.caption("Positive = rising role; negative = declining role.")
            with matchup_col:
                st.subheader("Matchup")
                for component in sos.matchup_detail(
                    row, detail_position, confidence_thresholds=confidence_thresholds
                ):
                    confident_label = "confident" if component["confident"] else "low sample"
                    st.write(
                        f"**{component['position_group']}** -- "
                        f"adj EPA allowed: {component['adj_epa_allowed']:.3f} "
                        f"({component['n_plays']} plays, {confident_label})"
                        if component["adj_epa_allowed"] is not None
                        else f"**{component['position_group']}** -- no real data yet this season"
                    )

            st.divider()
            game_col, total_col = st.columns(2)
            with game_col:
                spread = row.get("spread")
                st.metric(
                    "Team spread (own perspective, + = favoured)",
                    f"{spread:+.1f}" if spread is not None else "n/a",
                )
            with total_col:
                implied = row.get("implied_team_total")
                st.metric("Implied team total", f"{implied:.1f}" if implied is not None else "n/a")
