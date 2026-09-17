"""Pure data preparation for the chopped-player bid calculator."""

from __future__ import annotations

from ffapp.config import LeagueConfig
from ffapp.tools.chopped_bids import build_player_values


def is_chopped_league(league: LeagueConfig) -> bool:
    return bool(
        league.league_cache.get("league_type") == 3 or league.league_cache.get("disable_trades")
    )


__all__ = ["build_player_values", "is_chopped_league"]
