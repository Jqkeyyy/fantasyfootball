from __future__ import annotations

from ffapp.app.league_selector import ordered_leagues
from ffapp.config import LeagueConfig


def _league(slug: str, display: str, *, primary: bool = False) -> LeagueConfig:
    return LeagueConfig(slug, display, primary, "1", 2026, {}, {})


def test_ordered_leagues_puts_primary_first_then_sorts_names() -> None:
    leagues = [
        _league("z", "Zulu"),
        _league("primary", "Main", primary=True),
        _league("a", "Alpha"),
    ]

    assert [league.slug for league in ordered_leagues(leagues)] == ["primary", "a", "z"]
