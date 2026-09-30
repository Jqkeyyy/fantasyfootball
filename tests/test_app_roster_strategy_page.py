import polars as pl

from ffapp.app.roster_strategy_page import bye_pressure, roster_strategy


def test_roster_strategy_finds_weak_position_and_external_targets() -> None:
    values = pl.DataFrame(
        {
            "player_id": ["a_qb", "a_wr", "b_qb", "b_wr", "free_wr"],
            "player_name": ["AQ", "AW", "BQ", "BW", "FW"],
            "position": ["QB", "WR", "QB", "WR", "WR"],
            "team": ["A", "A", "B", "B", "C"],
            "next_4_points": [80.0, 20.0, 70.0, 60.0, 55.0],
            "playoff_points": [40.0, 10.0, 35.0, 30.0, 28.0],
            "ros_points": [160.0, 40.0, 140.0, 120.0, 110.0],
            "games_remaining": [10, 10, 10, 10, 10],
        }
    )
    health, targets, _ = roster_strategy(
        values,
        {"mine": ["a_qb", "a_wr"], "other": ["b_qb", "b_wr"]},
        "mine",
        {"QB": 1, "WR": 1},
    )

    wr = health.filter(pl.col("position") == "WR").row(0, named=True)
    assert wr["edge"] < 0
    assert "free_wr" in targets["player_id"].to_list()
    assert targets.filter(pl.col("player_id") == "free_wr")["path"].item() == "Waiver"


def test_bye_pressure_counts_missing_player_weeks() -> None:
    ros = pl.DataFrame({"player_id": ["a", "b", "a"], "week": [3, 3, 4], "mean": [10.0, 9.0, 11.0]})
    result = bye_pressure(ros, ["a", "b"], current_week=3, final_week=4)
    assert result["players_without_game"].to_list() == [0, 1]


def test_roster_strategy_excludes_unavailable_acquisition_target() -> None:
    values = pl.DataFrame(
        {
            "player_id": ["my_wr", "their_wr", "injured_free_wr"],
            "player_name": ["Mine", "Theirs", "Injured"],
            "position": ["WR", "WR", "WR"],
            "team": ["A", "B", "C"],
            "next_4_points": [10.0, 40.0, 60.0],
            "playoff_points": [10.0, 30.0, 40.0],
            "ros_points": [20.0, 80.0, 120.0],
            "games_remaining": [8, 8, 8],
        }
    )

    _, targets, _ = roster_strategy(
        values,
        {"mine": ["my_wr"], "other": ["their_wr"]},
        "mine",
        {"WR": 1},
        unavailable_player_ids={"injured_free_wr"},
    )

    assert "injured_free_wr" not in targets["player_id"].to_list()
