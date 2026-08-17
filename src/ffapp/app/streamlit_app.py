"""Draft board Streamlit page (SPEC.md §15 page 1, §9.7, §9.8; tasks 0.13, 0.14).

Reads the pre-built draft board CSV (`ffapp draft board`, task 0.12) --
per SPEC §15's own design constraint ("fast to load, everything
precomputed, nothing trained on page load"), nothing is recomputed here.
Filtering, tier-break styling, and live-draft pool logic live in
`draft_board_page.py`/`draft.live`, unit-tested there; this script is thin
`st.*` glue, verified by actually running it (CLAUDE.md's UI rule: "start
the dev server and use the feature in a browser"), not by a pytest test --
a live Streamlit script has no return value to assert on, it *is* the side
effect.

The live draft tab (SPEC §9.8) is a sub-tab of this same page per SPEC
§15's own layout, not a separate page. It fetches live from Sleeper
(`offline=False`) on demand -- a button, not automatic 5-10s polling
(Streamlit has no built-in auto-refresh primitive; adding one is a
reasonable follow-up, not required by TASKS.md 0.14's own acceptance bar,
which is about the available pool staying correct, not the UI's refresh
cadence).

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from ffapp.app.draft_board_page import (
    DEFAULT_ROW_CAP,
    DEFAULT_SORT_LABEL,
    ROW_CAP_THRESHOLD,
    SORT_OPTIONS,
    DraftBoardNotBuiltError,
    cap_rows,
    consensus_rankings,
    filter_board,
    load_board,
    single_source_rankings,
    sort_board,
    source_rank_columns,
    style_tier_breaks,
)
from ffapp.config import load_primary_league, load_settings
from ffapp.draft import live
from ffapp.draft.board import draft_board_csv_path, source_rankings_csv_path
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ingest import sleeper
from ffapp.league_format import parse_league_format

st.set_page_config(page_title="Draft Board", layout="wide")


@st.cache_data(show_spinner="Loading draft board...")
def _load_board_cached(csv_path_str: str, mtime: float) -> pl.DataFrame:
    """`mtime` is only ever used as part of the cache key, never read inside
    -- re-running `ffapp draft board` mid-session (a fresher ADP pull, say)
    changes the file's mtime and busts the cache automatically, without
    restarting Streamlit. Substitutes for SPEC §15's "keyed on model_version
    and as_of": there is no model_version in Phase 0, and the file's own
    mtime is a more direct freshness signal than re-parsing the `as_of_utc`
    column just to build a cache key.
    """
    return load_board(Path(csv_path_str))


settings = load_settings()
league = load_primary_league()
csv_path = draft_board_csv_path(settings, season=league.season)
source_rankings_path = source_rankings_csv_path(settings, season=league.season)

st.title("Draft Board")
st.caption(f"{league.display_name} -- {league.season}")

if not csv_path.exists():
    st.error(f"No draft board found at `{csv_path}`. Run `ffapp draft board` to build one first.")
    st.stop()

try:
    board = _load_board_cached(str(csv_path), csv_path.stat().st_mtime)
except DraftBoardNotBuiltError as exc:
    st.error(str(exc))
    st.stop()

source_rankings = None
if source_rankings_path.exists():
    source_rankings = _load_board_cached(
        str(source_rankings_path), source_rankings_path.stat().st_mtime
    )

board_tab, pure_rankings_tab, live_tab = st.tabs(["Draft Board", "Pure Rankings", "Live Draft"])

with board_tab:
    with st.sidebar:
        st.header("Filters")
        position_options = board["position"].unique(maintain_order=False).sort().to_list()
        selected_positions = st.multiselect("Position", options=position_options, default=[])
        tier_options = board["tier"].unique(maintain_order=False).sort().to_list()
        selected_tiers = st.multiselect("Tier", options=tier_options, default=[])
        st.header("Sort")
        sort_label = st.selectbox(
            "Sort by",
            options=list(SORT_OPTIONS),
            index=list(SORT_OPTIONS).index(DEFAULT_SORT_LABEL),
        )

    filtered = sort_board(
        filter_board(board, positions=selected_positions, tiers=selected_tiers),
        sort_label=sort_label,
    )

    show_all = False
    if filtered.height > ROW_CAP_THRESHOLD:
        with st.sidebar:
            show_all = st.checkbox(
                f"Show all {filtered.height} players (slow to render)", value=False
            )
    displayed = cap_rows(filtered, show_all=show_all)

    cap_note = (
        f' (capped at {DEFAULT_ROW_CAP} -- check "Show all" in the sidebar for the rest)'
        if displayed.height < filtered.height
        else ""
    )
    st.caption(
        f"{displayed.height} of {filtered.height} players shown{cap_note} -- "
        f"sorted by {sort_label}."
    )
    st.dataframe(style_tier_breaks(displayed), use_container_width=True, height=700)

    if board.height > 0:
        as_of = board["as_of_utc"][0]
        commit = board["git_commit"][0]
        st.caption(f"Board generated {as_of}" + (f" at commit `{commit}`" if commit else ""))

with pure_rankings_tab:
    if source_rankings is None:
        st.error(
            f"No source rankings found at `{source_rankings_path}`. "
            "Run `ffapp draft board` to build one (it writes both files)."
        )
    else:
        st.caption(
            "No VOR, no tiers, no ADP -- each source's own real published overall rank "
            "(from manually-exported cheat sheets, not this app's own live scrapers), and "
            "a plain average/median across sources. Uses the same Position filter as the "
            "Draft Board tab. Refresh by re-downloading and re-uploading the same filenames "
            "into rankings/, then re-running `ffapp draft board`."
        )
        position_filtered = filter_board(source_rankings, positions=selected_positions)
        sources = source_rank_columns(source_rankings)
        consensus_subtab, *source_subtabs = st.tabs(["Consensus", *sources])

        with consensus_subtab:
            consensus_display = cap_rows(consensus_rankings(position_filtered))
            st.caption(f"{consensus_display.height} players -- sorted by average rank.")
            st.dataframe(consensus_display, use_container_width=True, height=700)

        for source, subtab in zip(sources, source_subtabs, strict=True):
            with subtab:
                source_display = cap_rows(single_source_rankings(position_filtered, source))
                st.caption(
                    f"{source_display.height} players -- sorted by {source}'s own published rank."
                )
                st.dataframe(source_display, use_container_width=True, height=700)

with live_tab:
    st.caption(
        "Fetches live from Sleeper on demand -- click Refresh once the real draft is underway."
    )

    if "draft_picks" not in st.session_state:
        st.session_state.draft_picks = []

    if st.button("Refresh picks from Sleeper"):
        try:
            assert league.league_id is not None
            drafts: list[dict[str, Any]] = json.loads(
                sleeper.fetch_drafts(league.league_id, offline=False, settings=settings).read_text()
            )
            current_draft = next((d for d in drafts if d.get("season") == str(league.season)), None)
            if current_draft is None:
                st.warning(f"No {league.season} draft found for {league.display_name} on Sleeper.")
            else:
                picks: list[dict[str, Any]] = json.loads(
                    sleeper.fetch_draft_picks(
                        current_draft["draft_id"], offline=False, settings=settings
                    ).read_text()
                )
                st.session_state.draft_picks = picks
        except Exception as exc:  # a transient fetch failure shouldn't crash the page
            st.error(f"Could not fetch live picks: {exc}")

    picks = st.session_state.draft_picks
    st.caption(f"{len(picks)} pick(s) made so far.")

    pool = live.available_pool(board, picks)
    st.caption(f"{pool.height} of {board.height} players still available.")

    st.subheader("Best available")
    st.dataframe(
        live.best_available(pool, n=15).select(
            ["overall_rank", "player", "position", "tier", "vor", "opportunity_cost"]
        ),
        use_container_width=True,
    )

    st.subheader("Tier depth remaining")
    st.caption("Current (best) tier per position and how many are left in it.")
    st.dataframe(live.current_tier_summary(pool), use_container_width=True)

    st.subheader("Positional runs")
    active_runs = {pos: info for pos, info in live.positional_run(picks).items() if info["is_run"]}
    if active_runs:
        for pos, info in active_runs.items():
            st.warning(
                f"{pos} is going at {info['recent_rate']:.0%} of the last "
                f"{live.RUN_WINDOW} picks vs a {info['baseline_rate']:.0%} baseline -- "
                "looks like a run."
            )
    else:
        st.caption("No positional run detected.")

    st.subheader("Your starting lineup gaps")
    if settings.sleeper_username is None:
        st.caption("No Sleeper username configured (settings.yml sleeper.username).")
    else:
        try:
            assert league.league_id is not None
            user = json.loads(
                sleeper.fetch_user(
                    settings.sleeper_username, offline=False, settings=settings
                ).read_text()
            )
            rosters: list[dict[str, Any]] = json.loads(
                sleeper.fetch_rosters(
                    league.league_id, offline=False, settings=settings
                ).read_text()
            )
            my_roster_id = resolve_my_roster_id(user["user_id"], rosters)
            my_picks = [p for p in picks if p.get("roster_id") == my_roster_id]
            gaps = live.starting_lineup_gaps(my_picks, parse_league_format(league))
            if gaps:
                st.dataframe(
                    pl.DataFrame({"slot": list(gaps.keys()), "still_needed": list(gaps.values())}),
                    use_container_width=True,
                )
            else:
                st.caption("Starting lineup is full.")
        except Exception as exc:
            st.error(f"Could not resolve your roster: {exc}")
