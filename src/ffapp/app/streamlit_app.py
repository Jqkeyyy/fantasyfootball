"""Draft board Streamlit page (SPEC.md §15 page 1, §9.7; task 0.13).

Reads the pre-built draft board CSV (`ffapp draft board`, task 0.12) --
per SPEC §15's own design constraint ("fast to load, everything
precomputed, nothing trained on page load"), nothing is recomputed here.
Filtering and tier-break styling logic lives in `draft_board_page.py`,
unit-tested there; this script is thin `st.*` glue, verified by actually
running it (CLAUDE.md's UI rule: "start the dev server and use the feature
in a browser"), not by a pytest test -- a live Streamlit script has no
return value to assert on, it *is* the side effect.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.app.draft_board_page import (
    DraftBoardNotBuiltError,
    filter_board,
    load_board,
    style_tier_breaks,
)
from ffapp.config import load_primary_league, load_settings
from ffapp.draft.board import draft_board_csv_path

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

with st.sidebar:
    st.header("Filters")
    position_options = board["position"].unique(maintain_order=False).sort().to_list()
    selected_positions = st.multiselect("Position", options=position_options, default=[])
    tier_options = board["tier"].unique(maintain_order=False).sort().to_list()
    selected_tiers = st.multiselect("Tier", options=tier_options, default=[])

filtered = filter_board(board, positions=selected_positions, tiers=selected_tiers)

st.caption(f"{filtered.height} of {board.height} players shown -- sorted by VOR descending.")
st.dataframe(style_tier_breaks(filtered), use_container_width=True, height=700)

if board.height > 0:
    as_of = board["as_of_utc"][0]
    commit = board["git_commit"][0]
    st.caption(f"Board generated {as_of}" + (f" at commit `{commit}`" if commit else ""))
