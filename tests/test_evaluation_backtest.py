"""Task 1.12's own literal acceptance bar (SPEC §12.2): the walk-forward
loop itself, exercised with small synthetic fixtures -- no live `data/`
needed, same fixture-vs-live-run convention as `test_evaluation_snapshot.py`
and `test_leakage.py`. The real end-to-end run (`ffapp evaluate --seasons
...` against real cached data) is documented in HANDOFF.md instead.
"""

from __future__ import annotations

import polars as pl

from ffapp.evaluation import backtest

# --- fixtures ---------------------------------------------------------------------


def _features() -> pl.DataFrame:
    """Two players, two seasons, three weeks each -- `b1_col` stands in
    for an already-walk-forward-safe baseline column (task 1.10's own
    `.shift(1)` construction is upstream of this module's concern)."""
    rows = []
    for season in (2020, 2021):
        for week in (1, 2, 3):
            for player, position, team, target, b1 in (
                ("p1", "RB", "KC", 10.0 + week, float(week)),
                ("p2", "WR", "BAL", 5.0 + week, float(week) * 2),
            ):
                rows.append(
                    {
                        "player_id": player,
                        "season": season,
                        "week": week,
                        "position": position,
                        "team": team,
                        "availability_flag": True,
                        "target": target,
                        "b1_col": b1,
                    }
                )
    return pl.DataFrame(rows)


def _schedule(weeks: list[int], season: int = 2021) -> pl.DataFrame:
    return pl.DataFrame({"season": [season] * len(weeks), "week": weeks})


class _SpyPredictor:
    name = "spy"

    def __init__(self) -> None:
        self.seen_train_rows: list[pl.DataFrame] = []

    def fit(self, train_rows: pl.DataFrame) -> None:
        self.seen_train_rows.append(train_rows)
        return None

    def predict(self, fitted: None, target_rows: pl.DataFrame) -> pl.Series:
        return pl.Series([0.0] * target_rows.height)


# --- run_walk_forward_backtest -------------------------------------------------------


def test_never_trains_on_the_target_week_or_any_later_week() -> None:
    """The as_of contract (CLAUDE.md rule 2), applied to `train_rows`
    itself: for every (season, week) walked, `train_rows` must contain
    nothing from that week or later."""
    spy = _SpyPredictor()
    backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [spy],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
    )

    assert len(spy.seen_train_rows) == 3  # weeks 1, 2, 3
    for week, train_rows in zip([1, 2, 3], spy.seen_train_rows, strict=True):
        assert train_rows.filter((pl.col("season") == 2021) & (pl.col("week") >= week)).is_empty()
        assert train_rows.filter(pl.col("season") > 2021).is_empty()


def test_never_performs_a_random_split_train_rows_are_exactly_the_strictly_prior_ones() -> None:
    spy = _SpyPredictor()
    backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [spy],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
    )

    # week 3's train_rows: everything from season 2020 (6 rows) plus
    # season 2021 weeks 1-2 (4 rows) -- exactly 10, not a random sample.
    week_3_train = spy.seen_train_rows[2]
    assert week_3_train.height == 10


def test_respects_train_start_excluding_seasons_before_it() -> None:
    spy = _SpyPredictor()
    backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [spy],
        validation_seasons=[2021],
        train_start=2021,
        min_train_rows=0,
    )

    # week 3's train_rows should now exclude all of season 2020.
    week_3_train = spy.seen_train_rows[2]
    assert week_3_train.filter(pl.col("season") < 2021).is_empty()
    assert week_3_train.height == 4  # season 2021 weeks 1-2 only


def test_skips_weeks_below_min_train_rows() -> None:
    result = backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [backtest.BaselinePredictor("b1", "b1_col")],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1000,  # far more than this fixture ever has
    )

    assert result.is_empty()


def test_uses_real_schedule_weeks_not_a_hardcoded_range() -> None:
    """`schedule` lists only weeks 1 and 3 for the season -- week 2 (present
    in `features`) must never appear as a *validated* week, proving the
    loop walks `schedule`'s own weeks, not `range(1, 19)`."""
    result = backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 3]),
        [backtest.BaselinePredictor("b1", "b1_col")],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
    )

    assert set(result["week"].unique().to_list()) == {1, 3}


def test_output_has_one_row_per_player_week_predictor_with_target_and_prediction() -> None:
    result = backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [backtest.BaselinePredictor("b1", "b1_col")],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
    )

    # 2 players x 3 weeks x 1 predictor = 6 rows
    assert result.height == 6
    assert set(result.columns) == {
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "played",
        "target",
        "predictor",
        "prediction",
    }
    row = result.filter((pl.col("player_id") == "p1") & (pl.col("week") == 2)).row(0, named=True)
    assert row["target"] == 12.0
    assert row["prediction"] == 2.0  # b1_col's real value for p1/week 2
    assert row["predictor"] == "b1"


def test_returns_empty_dataframe_with_the_right_schema_when_nothing_qualifies() -> None:
    result = backtest.run_walk_forward_backtest(
        _features(),
        _schedule([]),
        [backtest.BaselinePredictor("b1", "b1_col")],
        validation_seasons=[2099],
        train_start=2015,
        min_train_rows=1,
    )

    assert result.is_empty()
    assert set(result.columns) == {
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "played",
        "target",
        "predictor",
        "prediction",
    }


def test_multiple_predictors_each_produce_their_own_rows() -> None:
    result = backtest.run_walk_forward_backtest(
        _features(),
        _schedule([1, 2, 3]),
        [
            backtest.BaselinePredictor("b1", "b1_col"),
            backtest.BaselinePredictor("target_copy", "target"),
        ],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
    )

    assert set(result["predictor"].unique().to_list()) == {"b1", "target_copy"}
    assert result.height == 12  # 2 players x 3 weeks x 2 predictors


# --- target_column ---------------------------------------------------------------------


def test_target_column_lets_the_harness_predict_a_different_outcome_column() -> None:
    """Task 1.14's own need: predicting `availability_flag` (a bool),
    not `target` (fantasy points), through the same harness."""
    features = _features().with_columns((pl.col("player_id") == "p1").alias("availability_flag"))

    result = backtest.run_walk_forward_backtest(
        features,
        _schedule([1, 2, 3]),
        [backtest.BaselinePredictor("b1", "b1_col")],
        validation_seasons=[2021],
        train_start=2015,
        min_train_rows=1,
        target_column="availability_flag",
    )

    # p1's real availability_flag is True (1.0) every week; p2's is False.
    p1_targets = result.filter(pl.col("player_id") == "p1")["target"].unique().to_list()
    p2_targets = result.filter(pl.col("player_id") == "p2")["target"].unique().to_list()
    assert p1_targets == [1.0]
    assert p2_targets == [0.0]


# --- BaselinePredictor ----------------------------------------------------------------


def test_baseline_predictor_fit_is_a_real_no_op() -> None:
    predictor = backtest.BaselinePredictor("b1", "b1_col")

    assert predictor.fit(_features()) is None


def test_baseline_predictor_predicts_the_wrapped_column_verbatim() -> None:
    predictor = backtest.BaselinePredictor("b1", "b1_col")
    target_rows = _features().filter((pl.col("season") == 2021) & (pl.col("week") == 2))

    preds = predictor.predict(None, target_rows)

    assert preds.to_list() == target_rows["b1_col"].to_list()


def test_baseline_predictor_name_is_stored_verbatim() -> None:
    predictor = backtest.BaselinePredictor("my_baseline", "some_col")

    assert predictor.name == "my_baseline"
