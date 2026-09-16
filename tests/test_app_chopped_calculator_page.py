from __future__ import annotations

import polars as pl

from ffapp.app.chopped_calculator_page import build_player_values, is_chopped_league
from ffapp.config import LeagueConfig


def test_is_chopped_league_uses_sleeper_league_metadata() -> None:
    chopped = LeagueConfig("c", "C", False, "1", 2026, {"league_type": 3}, {})
    standard = LeagueConfig("s", "S", True, "2", 2026, {"league_type": 1}, {})

    assert is_chopped_league(chopped)
    assert not is_chopped_league(standard)


def test_build_player_values_prefers_remaining_season_average() -> None:
    weekly = pl.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2026],
            "week": [2],
            "mean": [12.0],
        }
    )
    ros = pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2026, 2026],
            "week": [2, 3],
            "mean": [10.0, 14.0],
        }
    )
    players = pl.DataFrame(
        {
            "player_id": ["p1"],
            "sleeper_id": ["s1"],
            "full_name": ["Player One"],
            "position": ["RB"],
            "team": ["A"],
            "search_rank": [1],
        }
    )

    result = build_player_values(weekly, ros, players, season=2026, week=2)

    assert result["current_week_projection"].item() == 12.0
    assert result["projection_ppg"].item() == 12.0


def test_build_player_values_falls_back_to_current_week_without_ros() -> None:
    weekly = pl.DataFrame(
        {"player_id": ["p1"], "season": [2026], "week": [2], "mean": [11.0]}
    )
    players = pl.DataFrame(
        {
            "player_id": ["p1"],
            "sleeper_id": ["s1"],
            "full_name": ["Player One"],
            "position": ["WR"],
            "team": ["B"],
            "search_rank": [1],
        }
    )

    result = build_player_values(weekly, None, players, season=2026, week=2)

    assert result["projection_ppg"].item() == 11.0
