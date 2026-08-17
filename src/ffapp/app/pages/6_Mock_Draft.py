"""Mock draft page (not a numbered SPEC/TASKS.md task -- see `draft.mock`'s
own module docstring for why this exists: manually re-creating every draft
in the real Sleeper app to practice strategy was too slow to "mass
practice").

Streamlit glue only -- every real decision (bot logic, pick bookkeeping,
the grid's own cell data) lives in `draft.mock`, rendering-only helpers
live in `app.mock_draft_page`; this script owns `st.session_state` and the
`st.*` calls, verified by actually running it (CLAUDE.md's UI rule), not
by a unit test.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`
"""

from __future__ import annotations

import random

import streamlit as st

from ffapp.app.mock_draft_page import (
    available_pool_display,
    render_draft_grid_html,
    roster_table,
)
from ffapp.config import load_primary_league, load_settings
from ffapp.draft import mock

st.set_page_config(page_title="Mock Draft", layout="wide")

STATE_KEY = "mock_draft_state"
RNG_KEY = "mock_draft_rng"

# Real Sleeper draft board layout (teams as fixed columns, rounds as rows) --
# the reference the project owner asked this to look like. Colors are
# rgba overlays, not solid fills, so both light and dark Streamlit themes
# still show real contrast underneath (same reasoning `5_Draft_Mobile.py`'s
# own CSS block already uses for its card borders).
st.markdown(
    """
<style>
.mdg-scroll { overflow-x: auto; margin-bottom: 1rem; }
.mdg-table { border-collapse: collapse; width: 100%; }
.mdg-table th, .mdg-table td.mdg-cell {
    border: 1px solid rgba(128, 128, 128, 0.35);
    padding: 4px 6px;
    text-align: center;
    font-size: 0.8rem;
    min-width: 90px;
    white-space: nowrap;
}
.mdg-table th { font-size: 0.75rem; }
.mdg-mine { background: rgba(255, 193, 7, 0.28); }
.mdg-current { background: rgba(76, 175, 80, 0.4); font-weight: 700; }
.mdg-keeper { background: rgba(0, 172, 193, 0.35); }
.mdg-pos { opacity: 0.7; font-size: 0.7rem; }
.mdg-pending { opacity: 0.45; }
.mdg-trade { font-size: 0.65rem; opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)

settings = load_settings()
league = load_primary_league()

st.title("Mock Draft")
st.caption(
    f"{league.display_name} — {league.season} — practice against bots drafting off "
    "Sleeper's own real ADP, your real locked keepers, and this league's own real pick order."
)


def _start_new_draft() -> None:
    try:
        state = mock.init_mock_draft(league, settings, season=league.season)
    except mock.MockDraftBoardNotBuiltError as exc:
        st.error(str(exc))
        st.stop()
    rng = random.Random()
    mock.run_bot_picks_until_user_turn(state, rng=rng)
    st.session_state[STATE_KEY] = state
    st.session_state[RNG_KEY] = rng


button_label = "Restart" if STATE_KEY in st.session_state else "Start New Mock Draft"
if st.button(button_label, type="primary"):
    _start_new_draft()
    st.rerun()

if STATE_KEY not in st.session_state:
    st.info('Click "Start New Mock Draft" to begin.')
    st.stop()

state: mock.MockDraftState = st.session_state[STATE_KEY]
rng: random.Random = st.session_state[RNG_KEY]

if state.is_complete():
    st.success(f"Draft complete — {state.total_picks} picks made.")
else:
    round_label = f"Round {mock.current_round(state)} — pick {state.pick_no} of {state.total_picks}"
    st.subheader(f"{round_label} — your turn")

grid_rows = mock.draft_grid(state)
upcoming = mock.my_upcoming_picks(grid_rows)
legend = "🟡 Your cells below · 🟢 on the clock now · 🔒 keeper (locked)"
if upcoming:
    rest = ", ".join(str(p) for p in upcoming[1:6])
    now_label = f"now (#{upcoming[0]})" if not state.is_complete() else f"#{upcoming[0]}"
    tail = f", then {rest}{', ...' if len(upcoming) > 6 else ''}" if rest else ""
    legend += f" · Your next pick: {now_label}{tail}"
st.caption(legend)

st.markdown(render_draft_grid_html(grid_rows, state.team_names), unsafe_allow_html=True)

st.markdown("**Your roster**")
st.dataframe(
    roster_table(state.team_rosters.get(state.my_roster_id, [])),
    hide_index=True,
    width="stretch",
)

if not state.is_complete():
    st.markdown("**Available players**")
    positions = ["ALL", *sorted(state.pool["position"].unique().to_list())]
    position = st.segmented_control(
        "Position",
        options=positions,
        default="ALL",
        key="mock_draft_position_filter",
        label_visibility="collapsed",
    )
    position_filter = None if position in (None, "ALL") else position
    pool_view = available_pool_display(state.pool, position=position_filter)

    st.dataframe(
        pool_view.select("player_name", "position", "team", "vor", "tier", "adp"),
        hide_index=True,
        width="stretch",
    )

    options = {
        f"{row['player_name']} ({row['position']} - {row['team']}) — VOR {row['vor']:.1f}": row[
            "join_key"
        ]
        for row in pool_view.iter_rows(named=True)
    }
    if options:
        choice_label = st.selectbox(
            "Draft a player", options=list(options.keys()), key="mock_draft_choice"
        )
        if st.button("Draft this player", type="primary"):
            mock.record_pick(state, state.my_roster_id, options[choice_label])
            mock.run_bot_picks_until_user_turn(state, rng=rng)
            st.rerun()
    else:
        st.info("No players match this filter.")
