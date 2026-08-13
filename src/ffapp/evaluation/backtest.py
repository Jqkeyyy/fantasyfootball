"""Walk-forward backtest harness (SPEC.md §12.2; task 1.12).

SPEC's own pseudocode, applied literally:

    FOR season S in [validation seasons]:
      FOR week W in 1..18:
          train_rows = all rows with (season, week) strictly before (S, W)
                       and season >= settings.seasons.train_start
          IF len(train_rows) < min_train_rows: skip
          fit model on train_rows
          predict week (S, W) using snapshot(as_of = kickoff(S, W))
          store predictions

`run_walk_forward_backtest` operates directly on
`features/player_week_features.parquet` (task 1.9), which is already
walk-forward-safe by construction -- every feature column is lag-shifted
onto the row it's used for, and `target` is that row's own actual
outcome -- and independently audited by task 1.11's
`evaluation.snapshot`/`assert_no_leakage`. This harness's own
`train_rows`/target-rows split, filtering strictly on `(season, week)`,
*is* the as_of contract applied at the row-selection level; re-deriving
every feature from raw interim tables via `snapshot()` on each of the
~90 (season, week) iterations a five-season backtest walks over would
re-run the entire feature-computation pipeline that many times for no
additional safety -- `evaluation.snapshot`'s own module docstring
already frames itself as an independent trust-but-verify layer on top
of the pipeline, not a required step of the production predict path.

No code path here performs a random split (CLAUDE.md rule 1) -- the
only filter applied anywhere is time-ordered: `(season, week)` strictly
before the target week.

No real trainable model exists yet (tasks 1.14-1.16 land those) -- this
task's own literal acceptance bar is that the harness runs end to end,
so `Predictor` is a small protocol any future model can implement, and
`BaselinePredictor` (wrapping SPEC §12.3's B0/B1/B2, `models.baselines`)
is what this task exercises it with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import polars as pl


class Predictor(Protocol):
    """Anything `run_walk_forward_backtest` can fit-and-predict with --
    a real LightGBM model (tasks 1.14-1.16) and `BaselinePredictor`
    (this module) both satisfy this the same way."""

    name: str

    def fit(self, train_rows: pl.DataFrame) -> Any: ...

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series: ...


class BaselinePredictor:
    """Wraps an already-computed baseline column (SPEC §12.3's B0/B1/B2,
    `models.baselines`) as a `Predictor`. `fit` is a genuine no-op, not
    a stub standing in for future work: a baseline column's correctness
    comes entirely from `models.baselines`' own `.shift(1)` construction
    upstream, not from anything `train_rows` could tell it here --
    exercising the harness with something real that needs no fitting at
    all, ahead of task 1.14+ landing a model that does.
    """

    def __init__(self, name: str, column: str) -> None:
        self.name = name
        self.column = column

    def fit(self, train_rows: pl.DataFrame) -> None:
        return None

    def predict(self, fitted: None, target_rows: pl.DataFrame) -> pl.Series:
        return target_rows[self.column]


def _validation_weeks(schedule: pl.DataFrame, season: int) -> list[int]:
    """Real scheduled weeks for `season`, straight from `schedule` --
    not a hardcoded `range(1, 19)` (CLAUDE.md rule 5): the 2015-2020
    16-game season and the 2021+ 17-game season both read correctly."""
    return sorted(schedule.filter(pl.col("season") == season)["week"].unique().to_list())


_PREDICTIONS_SCHEMA = {
    "player_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "position": pl.Utf8,
    "target": pl.Float64,
    "predictor": pl.Utf8,
    "prediction": pl.Float64,
}


def run_walk_forward_backtest(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    predictors: Sequence[Predictor],
    *,
    validation_seasons: Sequence[int],
    train_start: int,
    min_train_rows: int,
) -> pl.DataFrame:
    """Walk forward over every (season, week) in `validation_seasons`,
    fitting and predicting with each of `predictors`. Returns one row
    per (player_id, season, week, predictor) with the real `target`
    alongside the `prediction`, ready for task 1.13's metrics module.
    """
    predictions: list[pl.DataFrame] = []
    for season in validation_seasons:
        for week in _validation_weeks(schedule, season):
            before = (pl.col("season") < season) | (
                (pl.col("season") == season) & (pl.col("week") < week)
            )
            train_rows = features.filter(before & (pl.col("season") >= train_start))
            if train_rows.height < min_train_rows:
                continue

            target_rows = features.filter((pl.col("season") == season) & (pl.col("week") == week))
            if target_rows.is_empty():
                continue

            base = target_rows.select("player_id", "season", "week", "position", "target")
            for predictor in predictors:
                fitted = predictor.fit(train_rows)
                preds = predictor.predict(fitted, target_rows)
                predictions.append(
                    base.with_columns(
                        pl.lit(predictor.name).alias("predictor"),
                        pl.Series("prediction", preds, dtype=pl.Float64),
                    )
                )

    if not predictions:
        return pl.DataFrame(schema=_PREDICTIONS_SCHEMA)
    return pl.concat(predictions, how="vertical_relaxed")


__all__ = ["BaselinePredictor", "Predictor", "run_walk_forward_backtest"]
