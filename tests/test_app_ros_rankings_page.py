# tests/test_app_ros_rankings_page.py (new file)

from __future__ import annotations

import polars as pl

from ffapp.app.ros_rankings_page import filter_board, style_rank_change


def test_style_rank_change_formats_signed_movement() -> None:
    board = pl.DataFrame({"player_id": ["p1", "p2", "p3"], "rank_change": [3, -1, None]})
    result = style_rank_change(board)
    assert result["rank_change_display"].to_list() == ["+3", "-1", "—"]


def test_filter_board_by_position() -> None:
    board = pl.DataFrame(
        {"player_id": ["p1", "p2"], "position": ["RB", "WR"], "vor_ros": [10.0, 5.0]}
    )
    result = filter_board(board, position="RB", available_ids=None)
    assert result["player_id"].to_list() == ["p1"]


def test_filter_board_by_availability() -> None:
    board = pl.DataFrame(
        {"player_id": ["p1", "p2"], "position": ["RB", "RB"], "vor_ros": [10.0, 5.0]}
    )
    result = filter_board(board, position=None, available_ids={"p2"})
    assert result["player_id"].to_list() == ["p2"]
