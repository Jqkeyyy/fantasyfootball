from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.tools import ros_rankings


def _league_format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=10,
        starters={"RB": 2, "WR": 2},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=6,
        ir=1,
        playoff_week_start=15,
        waiver_budget=100,
    )


def test_current_free_agent_projections_excludes_rostered_players() -> None:
    ros_points = pl.DataFrame({"player_id": ["p1", "p2"], "ros_points": [80.0, 60.0]})
    players_dim = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "sleeper_id": ["s1", "s2"],
            "position": ["RB", "WR"],
            "active": [True, True],
            "team": ["KC", "BUF"],
        }
    )
    result = ros_rankings.current_free_agent_projections(
        ros_points, players_dim, rostered_ids={"s1"}, eligible_positions={"RB", "WR"}
    )
    assert result["player_id"].to_list() == ["p2"]


def test_build_ros_board_adds_vor_ros_and_differs_by_league_format() -> None:
    # 50 RBs + 50 WRs -- large enough that neither league format's dedicated
    # starter count (10-team RB2 = 20, 18-team RB2 = 36) exhausts the real
    # pool and falls back to `replacement_level`'s own clamp (which would
    # otherwise silently collapse both formats onto the same worst-available
    # player and make this test's own assertion false regardless of whether
    # LeagueFormat is wired correctly).
    n_per_position = 50
    ros_points = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(1, 2 * n_per_position + 1)],
            "ros_points": [float(200 - i) for i in range(1, 2 * n_per_position + 1)],
        }
    )
    players_dim = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(1, 2 * n_per_position + 1)],
            "sleeper_id": [f"s{i}" for i in range(1, 2 * n_per_position + 1)],
            "position": ["RB"] * n_per_position + ["WR"] * n_per_position,
            "active": [True] * (2 * n_per_position),
            "team": ["KC"] * (2 * n_per_position),
        }
    )
    fmt_10team = _league_format()
    fmt_18team = LeagueFormat(
        n_teams=18,
        starters={"RB": 2, "WR": 2},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=6,
        ir=1,
        playoff_week_start=15,
        waiver_budget=100,
    )
    board_10 = ros_rankings.build_ros_board(
        ros_points, players_dim, set(), {"RB", "WR"}, fmt_10team
    )
    board_18 = ros_rankings.build_ros_board(
        ros_points, players_dim, set(), {"RB", "WR"}, fmt_18team
    )
    vor_10 = dict(zip(board_10["player_id"].to_list(), board_10["vor_ros"].to_list(), strict=True))
    vor_18 = dict(zip(board_18["player_id"].to_list(), board_18["vor_ros"].to_list(), strict=True))
    assert vor_10 != vor_18  # replacement level must move materially between formats


def test_rank_change_reports_null_with_no_previous_board() -> None:
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [20.0, 10.0]})
    result = ros_rankings.rank_change(current, None)
    assert result["rank_change"].null_count() == result.height


def test_rank_change_reports_real_movement() -> None:
    previous = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [10.0, 20.0]})  # p2 was rank 1
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [20.0, 10.0]})  # p1 now rank 1
    result = ros_rankings.rank_change(current, previous)
    by_player = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert by_player["p1"]["rank_change"] == 1  # moved up one spot
    assert by_player["p2"]["rank_change"] == -1


def test_rank_change_null_for_a_new_player() -> None:
    previous = pl.DataFrame({"player_id": ["p1"], "vor_ros": [10.0]})
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [10.0, 30.0]})
    result = ros_rankings.rank_change(current, previous)
    new_row = result.filter(pl.col("player_id") == "p2").row(0, named=True)
    assert new_row["rank_change"] is None


def test_current_free_agent_projections_preserves_all_ros_points_table_columns() -> None:
    """Real production `ros_points_table` (tools.ros_aggregate.aggregate_ros's
    own output) carries far more than just `ros_points` -- p10/p50/p90,
    expected_games, playoff_weeks_value are all real columns a caller
    downstream (the ROS Rankings Streamlit page) needs. The join+select
    here must never silently drop them -- a real bug found this way once
    already (SPEC-ADDENDUM-04.md §D.5's own literal UI requirement)."""
    ros_points = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "ros_points": [80.0, 60.0],
            "ros_p10": [50.0, 40.0],
            "ros_p50": [80.0, 60.0],
            "ros_p90": [110.0, 80.0],
            "expected_games": [7.5, 6.2],
            "playoff_weeks_value": [20.0, 15.0],
        }
    )
    players_dim = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "sleeper_id": ["s1", "s2"],
            "position": ["RB", "WR"],
            "active": [True, True],
            "team": ["KC", "BUF"],
        }
    )
    result = ros_rankings.current_free_agent_projections(
        ros_points, players_dim, rostered_ids=set(), eligible_positions={"RB", "WR"}
    )
    for column in ("ros_p10", "ros_p50", "ros_p90", "expected_games", "playoff_weeks_value"):
        assert column in result.columns, f"{column} was dropped"
    p1_row = result.filter(pl.col("player_id") == "p1").row(0, named=True)
    assert p1_row["ros_p10"] == 50.0
    assert p1_row["expected_games"] == 7.5
    assert p1_row["playoff_weeks_value"] == 20.0


def test_build_ros_board_preserves_all_ros_points_table_columns() -> None:
    """Same real gap, one layer up -- build_ros_board's own final output
    (what actually gets written to rankings_ros/latest.parquet) must also
    carry these columns through compute_vor unchanged."""
    ros_points = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "ros_points": [80.0, 60.0, 40.0, 20.0],
            "ros_p10": [50.0, 40.0, 25.0, 10.0],
            "ros_p50": [80.0, 60.0, 40.0, 20.0],
            "ros_p90": [110.0, 80.0, 55.0, 30.0],
            "expected_games": [7.5, 6.2, 5.0, 3.0],
            "playoff_weeks_value": [20.0, 15.0, 10.0, 5.0],
        }
    )
    players_dim = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "sleeper_id": ["s1", "s2", "s3", "s4"],
            "position": ["RB", "RB", "WR", "WR"],
            "active": [True, True, True, True],
            "team": ["KC", "KC", "BUF", "BUF"],
        }
    )
    fmt = LeagueFormat(
        n_teams=1,
        starters={"RB": 1, "WR": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=0,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    board = ros_rankings.build_ros_board(ros_points, players_dim, set(), {"RB", "WR"}, fmt)
    columns = ("ros_p10", "ros_p50", "ros_p90", "expected_games", "playoff_weeks_value", "vor_ros")
    for column in columns:
        assert column in board.columns, f"{column} was dropped"
