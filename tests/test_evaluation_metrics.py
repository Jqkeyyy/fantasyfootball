"""Task 1.13's own literal acceptance bar (SPEC §12.4-§12.5): every
metric computed per position with observation counts and confidence
intervals -- exercised with small synthetic fixtures, no live `data/`
needed, same fixture-vs-live-run convention as the rest of `evaluation/`.
The real end-to-end run against `data/outputs/eval/.../predictions.parquet`
(task 1.12's real output) is documented in HANDOFF.md instead.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffapp.evaluation import metrics
from ffapp.league_format import LeagueFormat

# --- fixtures ---------------------------------------------------------------------


def _league_format(**overrides: object) -> LeagueFormat:
    base = dict(
        n_teams=2,
        starters={"RB": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=2,
        ir=0,
        playoff_week_start=15,
        waiver_budget=None,
    )
    base.update(overrides)
    return LeagueFormat(**base)  # type: ignore[arg-type]


def _predictions() -> pl.DataFrame:
    """One predictor ("model"), 2 RBs, 4 real weeks -- `prediction` tracks
    `target` closely for p1 (a "good" predictor row) and is inverted for
    p2 (a "bad" predictor row), so Spearman/MAE/top-k behave predictably
    and are easy to hand-verify."""
    rows = []
    for week in (1, 2, 3, 4):
        rows.append(
            {
                "player_id": "p1",
                "season": 2021,
                "week": week,
                "position": "RB",
                "team": "KC",
                "target": 10.0 + week,
                "predictor": "model",
                "prediction": 10.0 + week,  # exact
            }
        )
        rows.append(
            {
                "player_id": "p2",
                "season": 2021,
                "week": week,
                "position": "RB",
                "team": "KC",
                "target": 5.0 + week,
                "predictor": "model",
                "prediction": 5.0 + week - 0.5,  # close but not exact
            }
        )
    return pl.DataFrame(rows)


# --- bootstrap_ci_over_rows --------------------------------------------------------


def test_bootstrap_ci_over_rows_brackets_the_point_estimate() -> None:
    df = _predictions()
    rng = np.random.default_rng(0)

    low, high = metrics.bootstrap_ci_over_rows(df, metrics._mae, n_bootstrap=200, rng=rng)

    point = metrics._mae(df)
    assert low <= point <= high


def test_bootstrap_ci_over_rows_returns_nan_for_empty_input() -> None:
    empty = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "prediction": pl.Float64,
            "target": pl.Float64,
        }
    )

    low, high = metrics.bootstrap_ci_over_rows(empty, metrics._mae, n_bootstrap=10)

    assert np.isnan(low)
    assert np.isnan(high)


def test_bootstrap_ci_over_rows_duplicates_a_resampled_weeks_rows() -> None:
    """A degenerate rng that always draws the same week index must widen
    to include that single week's own MAE at the extreme, since every
    resample becomes "that week, n_weeks times over" -- proving rows
    really are duplicated per resampled week, not deduplicated."""
    df = _predictions()

    class _AlwaysZero:
        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            return np.zeros(size, dtype=int)

    week_1_mae = metrics._mae(df.filter(pl.col("week") == 1))

    low, high = metrics.bootstrap_ci_over_rows(
        df,
        metrics._mae,
        n_bootstrap=5,
        rng=_AlwaysZero(),  # type: ignore[arg-type]
    )

    assert low == pytest.approx(week_1_mae)
    assert high == pytest.approx(week_1_mae)


# --- bootstrap_ci_over_weekly_values ------------------------------------------------


def test_bootstrap_ci_over_weekly_values_brackets_the_mean() -> None:
    values = [0.1, 0.5, 0.9, 0.3]
    rng = np.random.default_rng(0)

    low, high = metrics.bootstrap_ci_over_weekly_values(values, n_bootstrap=500, rng=rng)

    assert low <= sum(values) / len(values) <= high


def test_bootstrap_ci_over_weekly_values_drops_nans() -> None:
    low, high = metrics.bootstrap_ci_over_weekly_values(
        [0.5, float("nan"), 0.5], n_bootstrap=50, rng=np.random.default_rng(0)
    )

    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_bootstrap_ci_over_weekly_values_returns_nan_for_all_nan_input() -> None:
    low, high = metrics.bootstrap_ci_over_weekly_values([float("nan")], n_bootstrap=10)

    assert np.isnan(low)
    assert np.isnan(high)


# --- accuracy_metrics ---------------------------------------------------------------


def test_accuracy_metrics_computes_overall_and_per_position_mae_rmse() -> None:
    results = metrics.accuracy_metrics(_predictions(), n_bootstrap=20, rng=np.random.default_rng(0))

    by_key = {(r.metric, r.position, r.scope): r for r in results}
    assert set(by_key) == {
        ("mae", None, "all"),
        ("rmse", None, "all"),
        ("mae", "RB", "all"),
        ("rmse", "RB", "all"),
    }
    # p1 is exact (0 error), p2 is off by 0.5 every week -- overall MAE = 0.25
    assert by_key[("mae", None, "all")].value == pytest.approx(0.25)
    assert by_key[("mae", None, "all")].n_obs == 8
    assert by_key[("mae", "RB", "all")].value == pytest.approx(0.25)


def test_accuracy_metrics_reports_a_confidence_interval_around_every_value() -> None:
    results = metrics.accuracy_metrics(_predictions(), n_bootstrap=50, rng=np.random.default_rng(0))

    for r in results:
        assert r.ci_low <= r.value <= r.ci_high


def test_accuracy_metrics_excludes_null_predictions() -> None:
    df = _predictions().with_columns(
        pl.when(pl.col("week") == 1).then(None).otherwise(pl.col("prediction")).alias("prediction")
    )

    results = metrics.accuracy_metrics(df, n_bootstrap=10)

    overall = next(r for r in results if r.metric == "mae" and r.position is None)
    assert overall.n_obs == 6  # 8 rows minus 2 null-prediction rows (week 1, both players)


def test_accuracy_metrics_computes_startable_scope_when_startable_counts_given() -> None:
    """startable_counts={"RB": 1} keeps only the real top-1 RB (by actual
    target) each week -- p1 always outscores p2 in this fixture, so only
    p1's rows survive."""
    results = metrics.accuracy_metrics(
        _predictions(), startable_counts={"RB": 1}, n_bootstrap=10, rng=np.random.default_rng(0)
    )

    startable_overall = next(
        r for r in results if r.metric == "mae" and r.scope == "startable" and r.position is None
    )
    assert startable_overall.n_obs == 4  # p1's 4 weeks only
    assert startable_overall.value == pytest.approx(0.0)  # p1's prediction is exact


def test_accuracy_metrics_per_predictor() -> None:
    df = pl.concat(
        [_predictions(), _predictions().with_columns(pl.lit("other").alias("predictor"))],
        how="vertical_relaxed",
    )

    results = metrics.accuracy_metrics(df, n_bootstrap=10)

    assert {r.predictor for r in results} == {"model", "other"}


# --- ranking_metrics -----------------------------------------------------------------


def test_ranking_metrics_spearman_is_perfect_when_prediction_never_reorders_players() -> None:
    """Every week, p1 > p2 in both target and prediction (see fixture) --
    a perfect same-week rank agreement every time."""
    results = metrics.ranking_metrics(_predictions(), n_bootstrap=20, rng=np.random.default_rng(0))

    spearman = next(r for r in results if r.metric == "spearman")
    assert spearman.value == pytest.approx(1.0)
    assert spearman.n_obs == 4  # 4 real weeks


def test_ranking_metrics_spearman_excludes_weeks_where_prediction_has_zero_variance() -> None:
    """A real bug found only by running against real data (B0's pooled
    positional mean gives every player at a position the *same* predicted
    value each week -- undefined correlation): polars' `corr` returns
    float `NaN` for zero-variance input, not `null`, so a plain
    `drop_nulls` silently leaves those weeks in, corrupting the mean with
    `NaN`. Every week here has p1 and p2 given the identical prediction."""
    df = _predictions().with_columns(pl.lit(50.0).alias("prediction"))

    results = metrics.ranking_metrics(df, n_bootstrap=10)

    spearman = next(r for r in results if r.metric == "spearman")
    assert spearman.n_obs == 0
    assert np.isnan(spearman.value)
    assert np.isnan(spearman.ci_low)
    assert np.isnan(spearman.ci_high)


def test_ranking_metrics_top_k_precision_included_only_when_k_positive() -> None:
    no_k = metrics.ranking_metrics(_predictions(), n_bootstrap=10)
    assert not any(r.metric.startswith("top_") for r in no_k)

    with_k = metrics.ranking_metrics(_predictions(), startable_counts={"RB": 1}, n_bootstrap=10)
    top_k = [r for r in with_k if r.metric.startswith("top_")]
    assert len(top_k) == 1
    assert top_k[0].metric == "top_1_precision"
    assert top_k[0].value == pytest.approx(1.0)  # p1 is both the real and predicted #1 every week


def test_ranking_metrics_per_position() -> None:
    df = _predictions().with_columns(
        pl.when(pl.col("player_id") == "p2")
        .then(pl.lit("WR"))
        .otherwise(pl.col("position"))
        .alias("position")
    )

    results = metrics.ranking_metrics(df, n_bootstrap=10)

    assert {r.position for r in results if r.metric == "spearman"} == {"RB", "WR"}


# --- distribution: pinball_loss / interval_coverage -----------------------------------


def test_pinball_loss_is_zero_for_a_perfect_prediction() -> None:
    assert metrics.pinball_loss([10.0, 20.0], [10.0, 20.0], quantile=0.5) == pytest.approx(0.0)


def test_pinball_loss_penalizes_underprediction_more_at_a_high_quantile() -> None:
    """At quantile=0.9, underpredicting (predicted < target) should cost
    more than overpredicting by the same margin -- the asymmetric pinball
    definition's whole point."""
    under = metrics.pinball_loss([10.0], [5.0], quantile=0.9)  # target 10, predicted 5
    over = metrics.pinball_loss([10.0], [15.0], quantile=0.9)  # target 10, predicted 15

    assert under > over


def test_interval_coverage_counts_the_fraction_actually_inside_the_interval() -> None:
    target = [5.0, 15.0, 25.0]
    lower = [0.0, 0.0, 0.0]
    upper = [10.0, 10.0, 10.0]

    coverage = metrics.interval_coverage(target, lower, upper)

    assert coverage == pytest.approx(1 / 3)


# --- start_sit_accuracy ---------------------------------------------------------------


def test_start_sit_accuracy_scores_same_team_flex_eligible_pairs() -> None:
    """p1 and p2 are both on KC, both RB (flex-eligible) -- the model's
    pick (p1, always higher prediction) also always has the higher real
    target in this fixture, so accuracy should be 1.0."""
    results = metrics.start_sit_accuracy(
        _predictions(), {"RB"}, n_bootstrap=20, rng=np.random.default_rng(0)
    )

    assert len(results) == 1
    result = results[0]
    assert result.value == pytest.approx(1.0)
    assert result.n_obs == 4  # 1 pair/week x 4 weeks


def test_start_sit_accuracy_ignores_players_on_different_teams() -> None:
    df = _predictions().with_columns(
        pl.when(pl.col("player_id") == "p2")
        .then(pl.lit("BAL"))
        .otherwise(pl.col("team"))
        .alias("team")
    )

    results = metrics.start_sit_accuracy(df, {"RB"}, n_bootstrap=10)

    assert results[0].n_obs == 0
    assert np.isnan(results[0].value)


def test_start_sit_accuracy_ignores_non_flex_eligible_positions() -> None:
    results = metrics.start_sit_accuracy(_predictions(), {"WR"}, n_bootstrap=10)

    assert results[0].n_obs == 0


def test_start_sit_accuracy_scores_a_worse_predictor_below_a_perfect_one() -> None:
    """A predictor that always picks the *wrong* player (inverted
    prediction) must score below the fixture's naturally-correct one."""
    inverted = _predictions().with_columns(
        pl.when(pl.col("player_id") == "p1")
        .then(pl.lit(1.0))
        .otherwise(pl.lit(100.0))
        .alias("prediction"),
        pl.lit("inverted").alias("predictor"),
    )
    df = pl.concat([_predictions(), inverted], how="vertical_relaxed")

    results = metrics.start_sit_accuracy(df, {"RB"}, n_bootstrap=10)

    by_predictor = {r.predictor: r for r in results}
    assert by_predictor["inverted"].value == pytest.approx(0.0)
    assert by_predictor["model"].value == pytest.approx(1.0)


# --- startable_counts_from_predictions ------------------------------------------------


def test_startable_counts_from_predictions_uses_real_target_values() -> None:
    league_format = _league_format(n_teams=1, starters={"RB": 2})

    counts = metrics.startable_counts_from_predictions(_predictions(), league_format)

    assert counts == {"RB": 2}


def test_startable_counts_from_predictions_ignores_which_predictor_is_present() -> None:
    """`target` is predictor-independent -- adding a second predictor's
    identical-target rows must not change the pool size."""
    league_format = _league_format(n_teams=1, starters={"RB": 2})
    two_predictors = pl.concat(
        [_predictions(), _predictions().with_columns(pl.lit("other").alias("predictor"))],
        how="vertical_relaxed",
    )

    counts = metrics.startable_counts_from_predictions(two_predictors, league_format)

    assert counts == {"RB": 2}
