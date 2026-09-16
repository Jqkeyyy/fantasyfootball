"""Shared league selection for Streamlit pages."""

from __future__ import annotations

import streamlit as st

from ffapp.app.data_status import render_league_data_controls
from ffapp.config import LeagueConfig, load_all_leagues


def ordered_leagues(leagues: list[LeagueConfig]) -> list[LeagueConfig]:
    """Primary league first, then stable display-name order."""
    return sorted(leagues, key=lambda league: (not league.is_primary, league.display_name.lower()))


def select_league() -> LeagueConfig:
    """Render one session-persistent league selector and return its config."""
    leagues = ordered_leagues(load_all_leagues())
    if not leagues:
        raise RuntimeError("No league configs found under config/leagues")
    by_slug = {league.slug: league for league in leagues}
    selected = st.sidebar.selectbox(
        "League",
        options=list(by_slug),
        format_func=lambda slug: by_slug[slug].display_name,
        key="selected_league_slug",
    )
    league = by_slug[selected]
    render_league_data_controls(league)
    return league


__all__ = ["ordered_leagues", "select_league"]
