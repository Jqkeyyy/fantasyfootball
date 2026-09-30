import polars as pl

from ffapp.app.predictions_page import player_history, prediction_comparison, weekly_accuracy


def _review() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026, 2026],
            "week": [1, 1, 2, 2, 2],
            "player_id": ["a", "b", "a", "b", "c"],
            "player": ["Alpha", "Bravo", "Alpha", "Bravo", "Charlie"],
            "position": ["RB", "WR", "RB", "WR", "QB"],
            "projected": [10.0, 8.0, 12.0, 9.0, 20.0],
            "actual": [16.0, 5.0, 2.0, None, 22.0],
            "status": [
                "Scored",
                "Scored",
                "Injury excluded",
                "Awaiting actuals",
                "Scored",
            ],
        }
    )


def test_comparison_keeps_only_scored_rows_and_signs_difference_as_actual_minus_predicted() -> None:
    result = prediction_comparison(_review())

    assert result.height == 3
    alpha = result.filter((pl.col("player_id") == "a") & (pl.col("week") == 1)).row(0, named=True)
    assert alpha["difference"] == 6.0
    assert alpha["miss"] == 6.0
    bravo = result.filter(pl.col("player_id") == "b").row(0, named=True)
    assert bravo["difference"] == -3.0
    assert bravo["miss"] == 3.0
    assert not result["injured"].any()


def test_comparison_can_include_injured_players_but_never_unscored_ones() -> None:
    result = prediction_comparison(_review(), include_injured=True)

    assert result.height == 4
    injured = result.filter(pl.col("injured")).row(0, named=True)
    assert (injured["player_id"], injured["week"], injured["difference"]) == ("a", 2, -10.0)
    assert result.filter((pl.col("player_id") == "b") & (pl.col("week") == 2)).is_empty()


def test_weekly_accuracy_summarises_each_week_newest_first() -> None:
    result = weekly_accuracy(prediction_comparison(_review()))

    assert result["week"].to_list() == [2, 1]
    week_one = result.filter(pl.col("week") == 1).row(0, named=True)
    assert week_one["players"] == 2
    assert week_one["average_miss"] == 4.5
    assert week_one["average_difference"] == 1.5
    assert week_one["within_five"] == 0.5
    assert week_one["predicted_total"] == 18.0
    assert week_one["actual_total"] == 21.0


def test_player_history_is_in_week_order() -> None:
    comparison = prediction_comparison(_review(), include_injured=True)

    history = player_history(comparison, "a")

    assert history["week"].to_list() == [1, 2]
    assert history["actual"].to_list() == [16.0, 2.0]


def test_empty_review_yields_empty_frames() -> None:
    comparison = prediction_comparison(pl.DataFrame())

    assert comparison.is_empty()
    assert weekly_accuracy(comparison).is_empty()
