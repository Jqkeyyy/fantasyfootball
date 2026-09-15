from __future__ import annotations

import polars as pl

from ffapp.evaluation.inseason import (
    recommend_projection_source,
    summarize_inseason_performance,
    summarize_interval_calibration,
    summarize_lineup_regret,
    valid_scored_history,
)
from ffapp.league_format import LeagueFormat


def _history(actuals: list[float], *, week: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026] * len(actuals),
            "week": [week] * len(actuals),
            "run_label": ["sunday"] * len(actuals),
            "player_id": [f"p{i}" for i in range(len(actuals))],
            "position": ["RB"] * len(actuals),
            "actual_points": actuals,
            "model_mean": [value + 2.0 for value in actuals],
            "b2_mean": actuals,
            "b3_mean": [value + 1.0 for value in actuals],
            "live_mean": [value + 0.5 for value in actuals],
            "projection_source": ["espn_weekly"] * len(actuals),
            "live_q10": [value - 2.0 for value in actuals],
            "live_q25": [value - 1.0 for value in actuals],
            "live_q50": actuals,
            "live_q75": [value + 1.0 for value in actuals],
            "live_q90": [value + 2.0 for value in actuals],
            "is_my_roster": [True] * len(actuals),
        }
    )


def test_all_zero_placeholder_week_is_not_treated_as_scored() -> None:
    assert valid_scored_history(_history([0.0, 0.0])).is_empty()
    assert summarize_inseason_performance(_history([0.0, 0.0])).is_empty()


def test_summary_reports_accuracy_and_rank_metrics() -> None:
    result = summarize_inseason_performance(_history([5.0, 10.0, 20.0]))

    direct = result.filter((pl.col("source") == "direct") & (pl.col("position") == "ALL"))
    b2 = result.filter((pl.col("source") == "baseline_b2") & (pl.col("position") == "ALL"))
    assert direct["mae"].item() == 2.0
    assert direct["weekly_spearman"].item() == 1.0
    assert direct["n_obs"].item() == 3
    assert b2["mae"].item() == 0.0
    espn = result.filter(
        (pl.col("source") == "espn_weekly") & (pl.col("position") == "ALL")
    )
    assert espn["mae"].item() == 0.5


def test_latest_run_label_is_used_once_per_player_week() -> None:
    tuesday = _history([5.0, 10.0]).with_columns(
        pl.lit("tuesday").alias("run_label"), pl.lit(100.0).alias("model_mean")
    )
    sunday = _history([5.0, 10.0])

    result = summarize_inseason_performance(pl.concat([tuesday, sunday]))

    direct = result.filter((pl.col("source") == "direct") & (pl.col("position") == "ALL"))
    assert direct["mae"].item() == 2.0
    assert direct["n_obs"].item() == 2


def test_interval_calibration_scores_the_logged_live_distribution() -> None:
    result = summarize_interval_calibration(_history([5.0, 10.0, 20.0]))

    live_80 = result.filter(
        (pl.col("source") == "espn_weekly") & (pl.col("interval") == "80%")
    )
    assert live_80["observed_coverage"].item() == 1.0
    assert live_80["mean_width"].item() == 4.0


def test_lineup_regret_uses_only_the_logged_roster() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"RB": 1},
        flex_slots={},
        flex_eligible={},
        bench=2,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    result = summarize_lineup_regret(_history([5.0, 10.0, 20.0]), fmt)

    live = result.filter(pl.col("source") == "espn_weekly")
    assert live["mean_lineup_regret"].item() == 0.0


def test_source_recommendation_waits_for_enough_evidence() -> None:
    result = summarize_inseason_performance(_history([5.0, 10.0, 20.0]))

    assert recommend_projection_source(result, "espn_weekly").startswith("Insufficient")
