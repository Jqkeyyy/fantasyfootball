"""Rest-of-season rankings Streamlit page (SPEC-ADDENDUM-04.md §D.5;
task 1.21). Sixth page in SPEC §15's own build order. Reads
`rankings_ros/latest.parquet` (Task 11's own output, `ffapp rankings ros`)
directly rather than recomputing anything model-level on page load --
same "fast to load" precedent every other page here already follows.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`, then
open "ROS Rankings" from the sidebar.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from ffapp.app.ros_rankings_page import filter_board, style_rank_change
from ffapp.config import load_primary_league, load_settings

st.set_page_config(page_title="ROS Rankings", layout="wide")

settings = load_settings()
league = load_primary_league()

st.title("Rest-of-Season Rankings")
st.caption(league.display_name)

latest_path = settings.data_root / "outputs" / league.slug / "rankings_ros" / "latest.parquet"
if not latest_path.exists():
    st.error(
        f"Missing {latest_path}. Run `ffapp project --from-week --through-week --league "
        f"{league.slug}` then `ffapp rankings ros --league {league.slug}` first."
    )
    st.stop()

board = pl.read_parquet(latest_path)
displayed = style_rank_change(board)

with st.sidebar:
    st.header("Filters")
    positions = ["All", *sorted(board["position"].unique().to_list())]
    position_choice = st.selectbox("Position", options=positions)
    st.caption(
        "Every row here is already a real current free agent -- rostered players never "
        "reach this board (Task 10's own `current_free_agent_projections` scoping)."
    )

position_filter = None if position_choice == "All" else position_choice
filtered = filter_board(displayed, position=position_filter, available_ids=None)

st.dataframe(
    filtered.select(
        "rank",
        "player_name",
        "rank_change_display",
        "position",
        "vor_ros",
        "ros_points",
        "ros_p10",
        "ros_p50",
        "ros_p90",
        "expected_games",
        "playoff_weeks_value",
    ).sort("rank"),
    use_container_width=True,
    height=700,
)
st.caption(
    "`rank_change_display` compares this run to the prior real run's own latest board -- "
    "an em dash means this is either the first real run or a genuinely new free agent."
)
