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

from ffapp.config import InvalidProjectionSourceError, LightGBMSettings
from ffapp.features import opponent
from ffapp.models import availability as availability_module
from ffapp.models import baselines as baselines_module
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
        "as_of_utc": "2025-11-01T00:00:00Z",
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

    def test_output_carries_the_projection_source(self) -> None:
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
            projection_source="direct",
        )

        assert set(result["projection_source"].to_list()) == {"direct"}

    def test_an_unknown_projection_source_raises(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        with pytest.raises(InvalidProjectionSourceError):
            predict.project_week(
                features,
                2025,
                9,
                train_start=2015,
                min_train_rows=10,
                lightgbm_params=_FAST_PARAMS,
                code_version="abc123",
                now=datetime(2025, 11, 1, tzinfo=UTC),
                projection_source="made_up",
            )


class TestProjectWeekAnchoredSource:
    """task 1.20 -- kept as a real, working option (SPEC's own enum names
    it) even though it isn't the shipped default; see
    docs/JOURNAL.md's 2026-08-16 closing entry."""

    def test_mean_composes_p_active_with_the_residual_model(self) -> None:
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
            projection_source="anchored",
        )

        assert result.height == 6
        assert set(result["projection_source"].to_list()) == {"anchored"}
        assert result["mean"].null_count() == 0

    def test_quantiles_stay_monotonic(self) -> None:
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
            projection_source="anchored",
        )

        for row in result.to_dicts():
            values = [row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]]
            assert values == sorted(values)


class TestProjectWeekBaselineB2Source:
    def test_mean_is_the_trailing_b2_value_not_multiplied_by_p_active(self) -> None:
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
            projection_source="baseline_b2",
        )

        from ffapp.models import baselines as real_baselines

        expected_b2 = (
            real_baselines.add_b2_ewm_4(features)
            .filter((pl.col("season") == 2025) & (pl.col("week") == 9))
            .sort("player_id")["b2_ewm_4"]
            .to_list()
        )
        assert result.sort("player_id")["mean"].to_list() == pytest.approx(expected_b2)

    def test_quantile_grid_uses_the_real_empirical_error_distribution(self) -> None:
        """`q10`..`q90` come from `baselines.empirical_error_quantiles` on
        `train_rows` (real, strictly-prior data), not task 1.16's own
        quantile model -- see docs/JOURNAL.md's 2026-08-16 entry for why
        (v1's quantile spread recentered around a different-source mean
        badly failed a real coverage check)."""
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
            projection_source="baseline_b2",
        )

        from ffapp.config import DEFAULT_QUANTILES
        from ffapp.models import baselines as real_baselines

        train_with_b2 = real_baselines.add_b2_ewm_4(features).filter(
            (pl.col("season") == 2025) & (pl.col("week") < 9)
        )
        error_quantiles = real_baselines.empirical_error_quantiles(
            train_with_b2, "b2_ewm_4", DEFAULT_QUANTILES
        )

        for row in result.to_dicts():
            # every row in this fixture is position "RB" (_target_week_frame's default)
            expected_q50 = max(row["mean"] + error_quantiles["RB"][0.5], 0.0)
            assert row["q50"] == pytest.approx(expected_q50)
            values = [row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]]
            assert values == sorted(values)
            assert row["q10"] >= 0.0


_EMPTY_B3_HISTORICAL = pl.DataFrame(
    schema={"player_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}
)


class TestProjectWeekConsensusB3Source:
    def test_requires_players_dim(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        with pytest.raises(ValueError, match="players_dim"):
            predict.project_week(
                features,
                2025,
                9,
                train_start=2015,
                min_train_rows=10,
                lightgbm_params=_FAST_PARAMS,
                code_version="abc123",
                now=datetime(2025, 11, 1, tzinfo=UTC),
                projection_source="consensus_b3",
                b3_historical=_EMPTY_B3_HISTORICAL,
            )

    def test_requires_b3_historical(self) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        with pytest.raises(ValueError, match="b3_historical"):
            predict.project_week(
                features,
                2025,
                9,
                train_start=2015,
                min_train_rows=10,
                lightgbm_params=_FAST_PARAMS,
                code_version="abc123",
                now=datetime(2025, 11, 1, tzinfo=UTC),
                projection_source="consensus_b3",
                players_dim=pl.DataFrame({"player_id": []}),
            )

    def test_mean_is_the_real_b3_value_for_a_resolved_player(self, monkeypatch) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        fake_b3 = pl.DataFrame(
            {
                "player_id": ["p0", "p1"],
                "season": [2025, 2025],
                "week": [9, 9],
                "b3_points": [12.5, 30.0],
            }
        )
        monkeypatch.setattr(
            baselines_module,
            "fetch_b3_for_week",
            lambda *args, **kwargs: fake_b3,
        )

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
            projection_source="consensus_b3",
            players_dim=pl.DataFrame({"player_id": []}),
            b3_historical=_EMPTY_B3_HISTORICAL,
        )

        assert result.height == 6  # every real row universe player, even unresolved ones
        by_player = {row["player_id"]: row["mean"] for row in result.to_dicts()}
        assert by_player["p0"] == pytest.approx(12.5)
        assert by_player["p1"] == pytest.approx(30.0)

    def test_quantile_grid_uses_the_real_empirical_b3_error_distribution(self, monkeypatch) -> None:
        """Real historical B3 rows, joined onto strictly-prior `train_rows`,
        drive the quantile spread -- not task 1.16's own quantile model
        (see docs/JOURNAL.md's 2026-08-16 entry)."""
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        fake_b3 = pl.DataFrame(
            {"player_id": ["p0"], "season": [2025], "week": [9], "b3_points": [12.5]}
        )
        monkeypatch.setattr(baselines_module, "fetch_b3_for_week", lambda *args, **kwargs: fake_b3)

        # Real historical B3 rows for weeks 1-8 (strictly prior to the
        # target week) -- b3_points always undershoots the real target by
        # exactly 3.0, so the empirical median error is a known, exact
        # value to check the recentering against.
        historical_rows = []
        for week in range(1, 9):
            for i in range(6):
                real_target = 10.0 + 50.0 * (i / 6)
                historical_rows.append(
                    {
                        "player_id": f"p{i}",
                        "season": 2025,
                        "week": week,
                        "b3_points": real_target - 3.0,
                    }
                )
        b3_historical = pl.DataFrame(historical_rows)

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
            projection_source="consensus_b3",
            players_dim=pl.DataFrame({"player_id": []}),
            b3_historical=b3_historical,
        )

        p0_row = result.filter(pl.col("player_id") == "p0").row(0, named=True)
        # every real historical error for p0's own position (RB) is
        # exactly +3.0 (real_target - (real_target - 3.0)) -- the
        # empirical median is exactly 3.0, so q50 = mean + 3.0.
        assert p0_row["q50"] == pytest.approx(p0_row["mean"] + 3.0)

    def test_a_real_player_with_no_b3_row_gets_an_honest_null_mean(self, monkeypatch) -> None:
        train = _training_frame()
        target = _target_week_frame(week=9)
        features = pl.concat([train, target], how="vertical_relaxed")

        empty_b3 = pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "season": pl.Int64,
                "week": pl.Int64,
                "b3_points": pl.Float64,
            }
        )
        monkeypatch.setattr(
            baselines_module,
            "fetch_b3_for_week",
            lambda *args, **kwargs: empty_b3,
        )

        result = predict.project_week(
            features,
            2025,
            9,
            train_start=2015,
            min_train_rows=10,
            lightgbm_params=_FAST_PARAMS,
            code_version="abc123",
            now=datetime(2025, 11, 1, tzinfo=UTC),
            projection_source="consensus_b3",
            players_dim=pl.DataFrame({"player_id": []}),
            b3_historical=_EMPTY_B3_HISTORICAL,
        )

        assert result["mean"].null_count() == 6
        # honest null mean -> the recentered quantile grid is null too,
        # not a guessed value.
        assert result["q50"].null_count() == 6


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
