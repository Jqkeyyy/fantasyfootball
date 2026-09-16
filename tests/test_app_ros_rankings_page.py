# tests/test_app_ros_rankings_page.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.app.ros_rankings_page import (
    RosBoardSchemaError,
    explain_ros_player,
    filter_board,
    player_week_schedule,
    prepare_board,
    style_rank_change,
    validate_board_schema,
)


def test_style_rank_change_formats_signed_movement() -> None:
    board = pl.DataFrame({"player_id": ["p1", "p2", "p3"], "rank_change": [3, -1, None]})
    result = style_rank_change(board)
    assert result["rank_change_display"].to_list() == ["+3", "-1", "—"]


def test_filter_board_by_position() -> None:
    board = pl.DataFrame(
        {"player_id": ["p1", "p2"], "position": ["RB", "WR"], "vor_ros": [10.0, 5.0]}
    )
    result = filter_board(board, position="RB", availability=None)
    assert result["player_id"].to_list() == ["p1"]


def test_filter_board_by_availability() -> None:
    board = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "position": ["RB", "RB"],
            "availability": ["Rostered", "Available"],
            "vor_ros": [10.0, 5.0],
        }
    )
    result = filter_board(board, position=None, availability="Available")
    assert result["player_id"].to_list() == ["p2"]


def test_filter_board_supports_search_and_team_filters() -> None:
    board = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "player_name": ["Josh Allen", "Jahmyr Gibbs"],
            "position": ["QB", "RB"],
            "availability": ["Rostered", "Rostered"],
            "nfl_team": ["BUF", "DET"],
            "fantasy_team": ["Alpha", "Beta"],
        }
    )
    result = filter_board(
        board,
        position=None,
        availability=None,
        nfl_team="BUF",
        fantasy_team="Alpha",
        search="allen",
    )
    assert result["player_id"].to_list() == ["p1"]


def test_prepare_board_sorts_selected_metric_and_adds_tiers() -> None:
    board = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "position": ["RB", "RB", "WR", "WR"],
            "vor_ros": [20.0, 10.0, 15.0, 5.0],
            "ros_points": [100.0, 90.0, 110.0, 80.0],
            "expected_games": [10.0, 10.0, 10.0, 10.0],
            "ros_p90": [130.0, 100.0, 150.0, 90.0],
            "ros_p10": [70.0, 60.0, 75.0, 50.0],
            "playoff_weeks_value": [20.0, 15.0, 25.0, 10.0],
        }
    )
    result = prepare_board(board, sort_label="Upside (P90)")
    assert result["player_id"].to_list() == ["p3", "p1", "p2", "p4"]
    assert result["view_rank"].to_list() == [1, 2, 3, 4]
    assert result["position_tier"].min() == 1
    assert result.filter(pl.col("player_id") == "p1")["ros_ppg"].item() == 10.0


def test_player_week_schedule_adds_relative_signal() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "week": [2, 3, 4],
            "opponent_team": ["A", "B", "C"],
            "mean": [8.0, 10.0, 12.0],
            "q10": [4.0, 5.0, 6.0],
            "q50": [8.0, 10.0, 12.0],
            "q90": [12.0, 15.0, 18.0],
            "is_current_week": [True, False, False],
        }
    )
    result = player_week_schedule(projections, "p1")
    assert result["schedule_signal"].to_list() == ["Tough", "Neutral", "Favorable"]


def test_explain_ros_player_reports_replacement_and_rank_change() -> None:
    explanations = explain_ros_player(
        {
            "position": "RB",
            "ros_points": 100.0,
            "vor_ros": 30.0,
            "expected_games": 8.0,
            "ros_p10": 60.0,
            "ros_p90": 140.0,
            "rank_change": 2,
        },
        remaining_weeks=10,
    )
    assert any("80%" in line for line in explanations)
    assert any("70.0 ROS points" in line for line in explanations)
    assert any("moved up 2" in line for line in explanations)


def test_validate_board_schema_rejects_old_artifact() -> None:
    with pytest.raises(RosBoardSchemaError, match="older artifact schema"):
        validate_board_schema(pl.DataFrame({"player_id": ["p1"]}))
