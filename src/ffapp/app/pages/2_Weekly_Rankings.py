"""Weekly rankings Streamlit page (SPEC.md §15 page 2, §14.1; task 1.19).

Second page in SPEC §15's own build order, under `app/pages/` per
Streamlit's own multipage convention (`streamlit_app.py`'s own docstring
already named this as the reason `pages/` stayed empty until now).
Reads the pre-built `outputs/projections.parquet` (`ffapp project`, task
1.18) rather than recomputing anything on page load, per SPEC §15's own
"fast to load... nothing trained on page load" constraint. Enrichment
(opponent/matchup grade/owner status) and filtering live in
`weekly_rankings_page.py`, unit-tested there; this script is thin `st.*`
glue, verified by actually running it (CLAUDE.md's UI rule), not a
pytest test.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`, then
open "Weekly Rankings" from the sidebar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from ffapp.app.weekly_rankings_page import (
    ProjectionsNotBuiltError,
    build_weekly_rankings,
    filter_rankings,
    load_projections,
)
from ffapp.config import load_primary_league, load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping as ids_mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.tools.waivers import rostered_sleeper_ids

st.set_page_config(page_title="Weekly Rankings", layout="wide")


@st.cache_data(show_spinner="Loading projections...")
def _load_projections_cached(path_str: str, mtime: float) -> pl.DataFrame:
    """`mtime` is only ever used as part of the cache key. SPEC §15 says
    cache "keyed on model_version and as_of" -- a single file-level key
    doesn't fit cleanly here, since one real `projections.parquet` holds
    multiple weeks, each with its own `model_version`/`as_of_utc` (task
    1.18's own upsert-by-week design). mtime busts the cache exactly when
    the file's real content changes -- the same substitution
    `draft_board_page`'s own cached loader already made for an analogous
    reason (task 0.13)."""
    return load_projections(Path(path_str))


@st.cache_data
def _players_dim_cached() -> pl.DataFrame:
    settings = load_settings()
    crosswalk = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=True, settings=settings)
    return ids_mapping.build_players_dim(crosswalk, sleeper_players, ids_mapping.ID_OVERRIDES_PATH)


@st.cache_data(show_spinner="Resolving your roster from Sleeper...")
def _load_roster_context(league_id: str, sleeper_username: str | None) -> tuple[set[str], set[str]]:
    """Real canonical `player_id` sets: `(my_roster_ids, rostered_ids)`,
    resolved offline from the already-warmed cache (this page's own "fast
    to load, no live network call on every page load" precedent, matching
    everything else on it) -- a live re-fetch happens whenever the cache
    is warmed again (`ffapp cache warm`), the same staleness policy every
    other offline-served page in this project already relies on."""
    settings = load_settings()
    players_dim = _players_dim_cached()
    sleeper_to_player = dict(
        zip(players_dim["sleeper_id"].to_list(), players_dim["player_id"].to_list(), strict=True)
    )

    rosters: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(league_id, offline=True, settings=settings).read_text()
    )
    rostered_sleeper = rostered_sleeper_ids(rosters)
    rostered_ids = {sleeper_to_player[sid] for sid in rostered_sleeper if sid in sleeper_to_player}

    my_roster_ids: set[str] = set()
    if sleeper_username is not None:
        try:
            user = json.loads(
                sleeper.fetch_user(sleeper_username, offline=True, settings=settings).read_text()
            )
            my_roster_id = resolve_my_roster_id(user["user_id"], rosters)
            my_roster = next(r for r in rosters if r["roster_id"] == my_roster_id)
            my_sleeper_ids = my_roster.get("players") or []
            my_roster_ids = {
                sleeper_to_player[sid] for sid in my_sleeper_ids if sid in sleeper_to_player
            }
        except Exception:  # a roster we can't resolve just means no "my roster" filter
            pass
    return my_roster_ids, rostered_ids


settings = load_settings()
league = load_primary_league()

st.title("Weekly Rankings")
st.caption(league.display_name)

projections_path = settings.data_root / "outputs" / "projections.parquet"
if not projections_path.exists():
    st.error(f"No projections found at `{projections_path}`. Run `ffapp project --week N` first.")
    st.stop()

try:
    all_projections = _load_projections_cached(
        str(projections_path), projections_path.stat().st_mtime
    )
except ProjectionsNotBuiltError as exc:
    st.error(str(exc))
    st.stop()

available_weeks = (
    all_projections.select("season", "week").unique().sort(["season", "week"], descending=True)
)
if available_weeks.is_empty():
    st.warning("projections.parquet exists but has no rows.")
    st.stop()

with st.sidebar:
    st.header("Week")
    week_options = [(row["season"], row["week"]) for row in available_weeks.to_dicts()]
    season, week = st.selectbox(
        "Season / Week", options=week_options, format_func=lambda sw: f"{sw[0]} week {sw[1]}"
    )

features_path = settings.data_root / "features" / "player_week_features.parquet"
schedule_path = settings.data_root / "interim" / "schedule.parquet"
if not features_path.exists() or not schedule_path.exists():
    st.error("Missing feature/schedule tables. See HANDOFF.md for the build steps.")
    st.stop()
features = pl.read_parquet(features_path)
schedule = pl.read_parquet(schedule_path)
players_dim = _players_dim_cached()

my_roster_ids: set[str] = set()
rostered_ids: set[str] = set()
if league.league_id is not None:
    try:
        my_roster_ids, rostered_ids = _load_roster_context(
            league.league_id, settings.sleeper_username
        )
    except Exception as exc:
        st.warning(f"Could not resolve roster ownership from Sleeper: {exc}")

rankings = build_weekly_rankings(
    all_projections,
    features,
    schedule,
    players_dim,
    season=season,
    week=week,
    my_roster_ids=my_roster_ids,
    rostered_ids=rostered_ids,
)

with st.sidebar:
    st.header("Filters")
    position_options = rankings["position"].unique(maintain_order=False).sort().to_list()
    selected_positions = st.multiselect("Position", options=position_options, default=[])
    availability = st.selectbox(
        "Availability", options=["all", "my_roster", "free_agent", "rostered_elsewhere"]
    )

filtered = filter_rankings(rankings, positions=selected_positions, availability=availability)

st.caption(f"{filtered.height} of {rankings.height} players shown -- season {season} week {week}.")

distinct_positions = sorted(filtered["position"].unique().to_list())
if not distinct_positions:
    st.info("No players match the current filters.")
else:
    position_tabs = st.tabs(distinct_positions)
    for tab, position in zip(position_tabs, distinct_positions, strict=True):
        with tab:
            # SPEC §14.1: "show floor and ceiling as a visible range, not a
            # hidden column" -- rendered as one combined text range rather
            # than two separate numeric columns, so it reads as the range
            # it is rather than two easy-to-skim-past numbers.
            position_df = (
                filtered.filter(pl.col("position") == position)
                .sort("proj_mean", descending=True)
                .with_columns(
                    (
                        pl.col("floor").round(1).cast(pl.String)
                        + " - "
                        + pl.col("ceiling").round(1).cast(pl.String)
                    ).alias("floor_to_ceiling")
                )
                .select(
                    "player_name",
                    "team",
                    "opponent",
                    "p_active",
                    "proj_mean",
                    "floor_to_ceiling",
                    "median",
                    "matchup_grade",
                    "n_plays_behind_matchup_grade",
                    "owner_status",
                )
            )
            st.dataframe(position_df, use_container_width=True, height=600)
