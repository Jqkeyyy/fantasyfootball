"""Task 1.18's own literal acceptance bar (SPEC §6.2, §11.8): "every row
carries model_version, as_of_utc, feature_hash, and git commit." Small
fast-fitting LightGBM fixtures, no live `data/` needed -- the real
end-to-end run against an already-played 2015-2025 week is documented in
docs/JOURNAL.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.models import availability as availability_module
from ffapp.models import points as points_module
from ffapp.models import predict

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

_DEFAULT_FEATURES = dict.fromkeys(points_module.COMMON_FEATURE_COLUMNS, 0.0)
_DEFAULT_FEATURES.update(
    {
        "report_status": "None",
        "practice_participation": "Full",
        "depth_chart_rank": 1.0,
        "age": 25.0,
    }
)


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "position": "RB",
        "availability_flag": True,
        "target": 10.0,
        **_DEFAULT_FEATURES,
    }
    row.update(kwargs)
    for group in opponent.POSITION_TO_GROUPS.get(row["position"], []):
        for metric in points_module._OPPONENT_ADJ_METRICS:
            row.setdefault(f"{metric}_{group.lower()}", 0.0)
    return row


def _training_frame(n_weeks: int = 8, rows_per_week: int = 6, position: str = "RB") -> pl.DataFrame:
    """`target_share_ewm_3` deterministically drives both `target` and
    `availability_flag` -- a real, learnable signal for all three model
    types this task composes, matching task 1.15/1.16's own fixture
    convention."""
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(rows_per_week):
            share = i / rows_per_week
            played = share > 0.1
            rows.append(
                _row(
                    player_id=f"p{i}",
                    week=week,
                    position=position,
                    target_share_ewm_3=share,
                    target=(10.0 + 50.0 * share) if played else 0.0,
                    availability_flag=played,
                    weeks_since_return=0.0,
                    snap_pct_trend=0.0,
                )
            )
    return pl.DataFrame(rows)


def _target_week_frame(week: int, position: str = "RB", rows_per_week: int = 6) -> pl.DataFrame:
    rows = []
    for i in range(rows_per_week):
        share = i / rows_per_week
        rows.append(
            _row(
                player_id=f"p{i}",
                week=week,
                position=position,
                target_share_ewm_3=share,
                weeks_since_return=0.0,
                snap_pct_trend=0.0,
            )
        )
    return pl.DataFrame(rows)


class TestComputeModelVersionAndFeatureHash:
    def test_model_version_is_deterministic_and_twelve_hex_chars(self) -> None:
        v1 = predict.compute_model_version(["a", "b"], _FAST_PARAMS, (2025, 10), "abc123")
        v2 = predict.compute_model_version(["b", "a"], _FAST_PARAMS, (2025, 10), "abc123")

        assert v1 == v2  # feature name order shouldn't matter
        assert len(v1) == 12
        int(v1, 16)  # valid hex

    def test_model_version_changes_when_train_cutoff_changes(self) -> None:
        v1 = predict.compute_model_version(["a"], _FAST_PARAMS, (2025, 10), "abc123")
        v2 = predict.compute_model_version(["a"], _FAST_PARAMS, (2025, 11), "abc123")

        assert v1 != v2

    def test_model_version_changes_when_code_version_changes(self) -> None:
        v1 = predict.compute_model_version(["a"], _FAST_PARAMS, (2025, 10), "abc123")
        v2 = predict.compute_model_version(["a"], _FAST_PARAMS, (2025, 10), "def456")

        assert v1 != v2

    def test_feature_hash_is_order_independent(self) -> None:
        assert predict.compute_feature_hash(["a", "b"]) == predict.compute_feature_hash(["b", "a"])

    def test_feature_hash_differs_for_different_feature_sets(self) -> None:
        assert predict.compute_feature_hash(["a"]) != predict.compute_feature_hash(["a", "b"])


class TestProjectWeek:
    def test_output_has_every_spec_column(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        assert set(predict.OUTPUT_COLUMNS).issubset(set(result.columns))
        assert result.height == 6

    def test_mean_equals_p_active_times_conditional_points(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        # hurdle formula: E[points] = P(plays) x E[points|plays] -- recompute
        # independently from the two underlying models and compare, rather
        # than trusting project_week's own internal arithmetic.
        avail_model = availability_module.fit_availability_model(
            train, lightgbm_params=_FAST_PARAMS
        )
        p_active = availability_module.predict_p_active(avail_model, target)
        points_model = points_module.fit_points_model(train, lightgbm_params=_FAST_PARAMS)
        conditional = points_module.predict_points(points_model, target)

        expected_mean = (p_active * conditional).to_list()
        assert result["mean"].to_list() == pytest.approx(expected_mean)

    def test_quantiles_are_monotonically_non_decreasing_per_row(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        for row in result.to_dicts():
            values = [row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]]
            assert values == sorted(values)

    def test_every_row_carries_model_version_as_of_utc_feature_hash_and_git_commit(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        for row in result.to_dicts():
            assert row["model_version"] is not None
            assert len(row["model_version"]) == 12
            assert row["as_of_utc"] == "2025-11-01T00:00:00+00:00"
            assert row["feature_hash"] is not None
            assert row["git_commit"] == "abc123"

    def test_scoped_to_skill_positions_only(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        dst_row = _row(player_id="dst1", week=9, position="DST", team="AAA")
        features = pl.concat([train, target, pl.DataFrame([dst_row])], how="diagonal_relaxed")

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        assert "dst1" not in result["player_id"].to_list()

    def test_not_enough_training_data_returns_empty_not_a_crash(self) -> None:
        train = _training_frame(n_weeks=1)  # far below min_train_rows
        target = _target_week_frame(week=2)
        features = pl.concat([train, target], how="vertical_relaxed")

        result = predict.project_week(
            features,
            2025,
            2,
            train_start=2015,
            min_train_rows=1000,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        assert result.is_empty()

    def test_no_row_universe_for_the_target_week_returns_empty_not_a_crash(self) -> None:
        train = _training_frame()

        result = predict.project_week(
            train,
            2025,
            99,  # a week with no rows at all
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
        )

        assert result.is_empty()


class TestWriteProjections:
    def test_writing_to_a_fresh_path_creates_the_file(self, tmp_path) -> None:
        output_path = tmp_path / "projections.parquet"
        rows = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [9]})

        combined = predict.write_projections(rows, output_path)

        assert output_path.exists()
        assert combined.height == 1

    def test_a_different_week_is_appended_not_overwritten(self, tmp_path) -> None:
        output_path = tmp_path / "projections.parquet"
        week9 = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [9]})
        week10 = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [10]})

        predict.write_projections(week9, output_path)
        combined = predict.write_projections(week10, output_path)

        assert combined.height == 2
        assert set(combined["week"].to_list()) == {9, 10}

    def test_the_same_week_is_overwritten_not_duplicated(self, tmp_path) -> None:
        output_path = tmp_path / "projections.parquet"
        first = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [9]})
        second = pl.DataFrame({"player_id": ["p2"], "season": [2025], "week": [9]})

        predict.write_projections(first, output_path)
        combined = predict.write_projections(second, output_path)

        assert combined.height == 1
        assert combined["player_id"].to_list() == ["p2"]
