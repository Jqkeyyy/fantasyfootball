# tests/test_models_predict_ros.py (new file)

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.models import predict_ros


def test_project_week_range_current_week_matches_existing_project_week(monkeypatch) -> None:
    """The anchor week's own row(s) must come from the real, unchanged
    `models.predict.project_week` (mocked here to isolate this test from
    needing real fitted models/network) -- this test's real job is
    proving predict_ros doesn't reimplement or alter current-week logic,
    only calls it."""
    from ffapp.models import predict as predict_module

    called_with = {}

    def fake_project_week(features, season, week, **kwargs):
        called_with["week"] = week
        return pl.DataFrame(
            {
                "player_id": ["p1"],
                "season": [season],
                "week": [week],
                "p_active": [0.95],
                "mean": [15.0],
                "q10": [8.0],
                "q25": [11.0],
                "q50": [15.0],
                "q75": [19.0],
                "q90": [23.0],
                "model_version": ["v1"],
                "projection_source": ["consensus_b3"],
                "as_of_utc": ["2026-09-01T00:00:00+00:00"],
                "feature_hash": ["h1"],
                "git_commit": ["abc"],
            }
        )

    monkeypatch.setattr(predict_module, "project_week", fake_project_week)

    result = predict_ros.project_week_range(
        features=pl.DataFrame(
            schema={
                "player_id": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "position": pl.String,
                "team": pl.String,
            }
        ),
        schedule=pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "home_team": pl.String,
                "away_team": pl.String,
                "season_type": pl.String,
            }
        ),
        defense_position_allowed=pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "defteam": pl.String,
                "position_group": pl.String,
                "adj_epa_allowed": pl.Float64,
                "n_plays": pl.Int64,
            }
        ),
        season=2026,
        from_week=5,
        through_week=5,
        league_slug="test-league",
        scoring_settings={},
        players_dim=pl.DataFrame(
            schema={
                "player_id": pl.String,
                "normalized_name": pl.String,
                "full_name": pl.String,
                "position": pl.String,
                "sleeper_id": pl.String,
            }
        ),
        b3_historical=pl.DataFrame(
            schema={
                "player_id": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "b3_points": pl.Float64,
            }
        ),
        actuals_to_date=pl.DataFrame(
            schema={"player_id": pl.String, "actual_points_to_date": pl.Float64}
        ),
        season_points_by_source={},
        trend_by_source={},
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        now=datetime(2026, 9, 1, tzinfo=UTC),
        train_start=2015,
        min_train_rows=1,
        lightgbm_params=None,
        code_version="abc",
        offline=True,
        settings=None,
    )
    assert called_with["week"] == 5
    assert result.filter(pl.col("is_current_week"))["mean"].to_list() == [15.0]


def test_project_week_range_future_week_uses_shape_not_project_week(monkeypatch) -> None:
    """A future week's row must NOT come from `models.predict.project_week`
    at all (it would need a weekly consensus that doesn't exist for that
    week) -- proven by making the mock raise if called for any week other
    than the anchor."""
    from ffapp.models import predict as predict_module

    def fake_project_week(features, season, week, **kwargs):
        assert week == 5, f"project_week must only be called for the anchor week, got {week}"
        return pl.DataFrame(
            {
                "player_id": ["p1"],
                "season": [season],
                "week": [week],
                "p_active": [0.95],
                "mean": [15.0],
                "q10": [8.0],
                "q25": [11.0],
                "q50": [15.0],
                "q75": [19.0],
                "q90": [23.0],
                "model_version": ["v1"],
                "projection_source": ["consensus_b3"],
                "as_of_utc": ["x"],
                "feature_hash": ["h1"],
                "git_commit": ["abc"],
            }
        )

    monkeypatch.setattr(predict_module, "project_week", fake_project_week)

    features = pl.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2026],
            "week": [5],
            "position": ["WR"],
            "team": ["KC"],
            "as_of_utc": ["2026-10-01T00:00:00+00:00"],
            "target": [10.0],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [6, 7],
            "home_team": ["KC", "DEN"],
            "away_team": ["DEN", "KC"],
            "season_type": ["REG", "REG"],
        }
    )
    dpa = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "defteam": pl.String,
            "position_group": pl.String,
            "adj_epa_allowed": pl.Float64,
            "n_plays": pl.Int64,
        }
    )
    players_dim = pl.DataFrame(
        {
            "player_id": ["p1"],
            "normalized_name": ["p one"],
            "full_name": ["P One"],
            "position": ["WR"],
            "sleeper_id": ["s1"],
        }
    )
    season_points = {
        "espn": pl.DataFrame(
            {"player_id": ["p1"], "position": ["WR"], "team": ["KC"], "points": [150.0]}
        )
    }

    result = predict_ros.project_week_range(
        features=features,
        schedule=schedule,
        defense_position_allowed=dpa,
        season=2026,
        from_week=5,
        through_week=7,
        league_slug="test-league",
        scoring_settings={},
        players_dim=players_dim,
        b3_historical=pl.DataFrame(
            schema={
                "player_id": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "b3_points": pl.Float64,
            }
        ),
        actuals_to_date=pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [50.0]}),
        season_points_by_source=season_points,
        trend_by_source={"espn": "flat"},
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        now=datetime(2026, 10, 1, tzinfo=UTC),
        train_start=2015,
        min_train_rows=1,
        lightgbm_params=None,
        code_version="abc",
        offline=True,
        settings=None,
    )
    future_rows = result.filter(~pl.col("is_current_week"))
    assert set(future_rows["week"].to_list()) == {6, 7}
    # season_consensus_ros_points = 150 - 50 = 100, split across weeks 6/7
    assert future_rows["mean"].sum() == pytest.approx(100.0, rel=0.01)
