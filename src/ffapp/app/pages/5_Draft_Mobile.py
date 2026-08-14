"""Mobile draft page (SPEC-ADDENDUM-03.md §C; task 0.15).

Draft-day support for a phone screen -- the desktop Draft Board /
`streamlit_app.py` Live Draft tab is unusable on a ~380px screen with
roughly nineteen columns (ADDENDUM-03 §C's own reasoning), so this is a
separate page, not a responsive version of that table.

Reads the same pre-built board CSV every other page reads (task 0.12) and
reuses `draft.live`'s own already-tested pool/tier-depth/best-available
functions (task 0.14) unmodified -- a new view over the same live-draft
state, not a new pipeline. Card-building, filtering, and the
above-the-fold summary are pure logic in `app/draft_mobile_page.py`,
unit-tested there; this script is thin `st.*` glue, verified by actually
running it (CLAUDE.md's UI rule), not by a pytest test.

`ffapp draft live --replay` (SPEC-ADDENDUM-03.md §E) writes a recorded
-draft replay session this page reads instead of live Sleeper picks when
one is active -- for verifying auto-refresh and card behaviour without an
actual draft in progress.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import streamlit as st

from ffapp.app.draft_board_page import DraftBoardNotBuiltError, load_board
from ffapp.app.draft_mobile_page import (
    DEFAULT_CARD_COUNT,
    POSITION_FILTER_ORDER,
    build_cards,
    filter_pool_by_position,
    top_line_summary,
)
from ffapp.config import LeagueConfig, Settings, load_primary_league, load_settings
from ffapp.draft import live
from ffapp.draft import replay as draft_replay
from ffapp.draft.board import draft_board_csv_path
from ffapp.ingest import sleeper

st.set_page_config(page_title="Draft Mobile", layout="centered")

# Minimum 16px body text (larger for the player name), big tap targets for
# the position filter, cards not a dataframe -- ADDENDUM-03 §C's own
# literal rendering requirements, none of which a bare st.dataframe/
# st.multiselect can satisfy.
st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 16px; }
.card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
}
.card-name { font-size: 1.2rem; font-weight: 700; }
.card-meta { font-size: 0.95rem; opacity: 0.75; }
.card-line { font-size: 1rem; margin: 0.15rem 0; }
.card-why { font-size: 0.95rem; opacity: 0.8; }
div[data-testid="stSegmentedControl"] button { min-height: 2.75rem; font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading draft board...")
def _load_board_cached(csv_path_str: str, mtime: float) -> pl.DataFrame:
    """Same mtime-as-cache-key idiom every other page uses (task 0.13's
    own precedent) -- the board itself only changes when `ffapp draft
    board`/`ffapp draft export` re-runs, not every 10-second refresh."""
    return load_board(Path(csv_path_str))


def _resolve_current_draft_id(league: LeagueConfig, settings: Settings) -> str | None:
    assert league.league_id is not None
    drafts: list[dict[str, Any]] = json.loads(
        sleeper.fetch_drafts(league.league_id, offline=False, settings=settings).read_text()
    )
    current = next((d for d in drafts if d.get("season") == str(league.season)), None)
    return current["draft_id"] if current is not None else None


settings = load_settings()
league = load_primary_league()
csv_path = draft_board_csv_path(settings, season=league.season)

st.title("Draft Mobile")
st.caption(f"{league.display_name} — {league.season}")

if not csv_path.exists():
    st.error(f"No draft board found at `{csv_path}`. Run `ffapp draft board` to build one first.")
    st.stop()

try:
    board = _load_board_cached(str(csv_path), csv_path.stat().st_mtime)
except DraftBoardNotBuiltError as exc:
    st.error(str(exc))
    st.stop()


@st.fragment(run_every=10)
def _live_board() -> None:
    position = st.segmented_control(
        "Position",
        options=POSITION_FILTER_ORDER,
        default="ALL",
        required=True,
        key="mobile_position_filter",
        label_visibility="collapsed",
    )

    replay_session = draft_replay.load_replay_session(settings, league_slug=league.slug)
    if replay_session is not None:
        picks = draft_replay.picks_revealed_so_far(replay_session)
        st.info(
            f"REPLAY draft {replay_session.draft_id}: {len(picks)} of "
            f"{len(replay_session.picks)} real picks revealed. "
            "(`ffapp draft live --stop` to clear.)"
        )
    else:
        try:
            draft_id = _resolve_current_draft_id(league, settings)
            if draft_id is None:
                picks = []
                st.warning(f"No {league.season} draft found for {league.display_name} on Sleeper.")
            else:
                picks_path = sleeper.fetch_draft_picks(draft_id, offline=False, settings=settings)
                picks = json.loads(picks_path.read_text())
        except Exception as exc:
            picks = st.session_state.get("mobile_last_picks", [])
            st.error(f"Could not fetch live picks ({exc}) — showing last known state.")
    st.session_state.mobile_last_picks = picks

    pool = live.available_pool(board, picks)
    filtered_pool = filter_pool_by_position(pool, position)
    tier_depth = live.tier_depth_remaining(pool)
    tier_summary = live.current_tier_summary(pool)
    summary_pool = filtered_pool if filtered_pool.height > 0 else pool
    summary = top_line_summary(summary_pool, tier_summary)

    st.caption(f"Last updated {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    if summary["best_player"] is not None:
        best_vor = summary["best_vor"]
        delta = f"VOR {best_vor:.1f}" if best_vor is not None else None
        st.metric("Best available", str(summary["best_player"]), delta)
        p_avail = summary["best_p_avail_next"]
        if p_avail is not None:
            st.caption(f"Falls to you at your next pick: {cast(float, p_avail):.0%}")

    st.caption("Tier depth remaining (current tier per position):")
    depth_rows = cast("list[dict[str, object]]", summary["tier_depth_by_position"])
    if depth_rows:
        st.write(
            "&nbsp;&nbsp;·&nbsp;&nbsp;".join(
                f"**{html.escape(str(row['position']))}** T{row['tier']} ({row['remaining']} left)"
                for row in depth_rows
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No players remaining.")

    st.caption(f"{filtered_pool.height} of {pool.height} available players shown.")

    cards = build_cards(filtered_pool, tier_depth, n=DEFAULT_CARD_COUNT)
    if not cards:
        st.info("No available players match this filter.")
    for card in cards:
        player = html.escape(str(card["player"]))
        position_label = html.escape(str(card["position"]))
        team = html.escape(str(card["team"]))
        why_line = html.escape(str(card["why_line"]))
        st.markdown(
            f"""<div class="card">
  <div class="card-name">{player}
    <span class="card-meta">{position_label} · {team} · bye {card["bye_week"]}</span>
  </div>
  <div class="card-line">Tier {card["tier"]} · VOR {card["vor"]:.1f}</div>
  <div class="card-why">{why_line}</div>
</div>""",
            unsafe_allow_html=True,
        )


_live_board()
