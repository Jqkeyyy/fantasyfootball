# tests/test_models_predict_ros.py (new file)

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.config import CacheSettings, ModelSettings, RosSettings, SeasonsSettings, Settings
from ffapp.models import predict_ros


def _minimal_settings(tmp_path, *, season_end_week: int) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
        model=ModelSettings(min_train_rows=1, retrain_cadence_weeks=1),
        seasons=SeasonsSettings(train_start=2015, current=2026),
        ros=RosSettings(season_end_week=season_end_week),
    )


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
    # Real full remaining season (weeks 6..18, default season_end_week=18,
    # no bye) -- matches production `schedule.parquet`'s own shape, unlike
    # a truncated 2-week fixture, which would silently mask the
    # full-remaining-season-allocation fix under test here.
    full_season_weeks = list(range(6, 19))
    schedule = pl.DataFrame(
        {
            "season": [2026] * len(full_season_weeks),
            "week": full_season_weeks,
            "home_team": ["KC"] * len(full_season_weeks),
            "away_team": ["DEN"] * len(full_season_weeks),
            "season_type": ["REG"] * len(full_season_weeks),
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
    # season_consensus_ros_points = 150 - 50 (actuals_to_date) = 100, then
    # the anchor week's own already-known mean (15.0, from the mocked
    # project_week) is subtracted before allocation (85.0) -- otherwise
    # week 5's value would be counted once via `current` and again via
    # the future-week spread. That 85.0 is split evenly (this fixture's
    # `dpa` is empty, so every week gets the same weight) across the real
    # full remaining season (13 weeks, 6..18, default season_end_week),
    # and only the caller's requested weeks 6/7 (2 of the 13) are kept.
    expected_total = (100.0 - 15.0) * (2 / 13)
    assert future_rows["mean"].sum() == pytest.approx(expected_total, rel=0.01)
    # This fixture's `features`/`b3_historical` carry zero real historical
    # rows, so `empirical_error_quantiles` has no recorded error for WR at
    # any tau. `apply_empirical_error_quantiles` must produce an honest
    # null in that case, not a fabricated zero-width interval collapsed
    # onto `mean` -- the same "never guess, leave it null" convention
    # `predict.project_week`'s own current-week path already relies on by
    # calling the identical function directly.
    for column in ("q10", "q25", "q50", "q75", "q90"):
        assert future_rows[column].is_null().all(), f"{column} should be null, not a guessed offset"


def _ros_fixtures(anchor_mean: float, season_points_total: float, actuals_to_date: float):
    """Shared fixture builder for the two anchor-week/horizon regression
    tests below -- both isolate a single number (the anchor's own mean,
    or `settings.ros.season_end_week`) against an otherwise identical
    setup, so a shared builder keeps the two tests' real difference
    visible instead of buried in fixture noise."""
    from ffapp.models import predict as predict_module

    def fake_project_week(features, season, week, **kwargs):
        return pl.DataFrame(
            {
                "player_id": ["p1"],
                "season": [season],
                "week": [week],
                "p_active": [0.95],
                "mean": [anchor_mean],
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

    predict_module_ref = predict_module

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
    full_season_weeks = list(range(6, 19))
    schedule = pl.DataFrame(
        {
            "season": [2026] * len(full_season_weeks),
            "week": full_season_weeks,
            "home_team": ["KC"] * len(full_season_weeks),
            "away_team": ["DEN"] * len(full_season_weeks),
            "season_type": ["REG"] * len(full_season_weeks),
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
            {
                "player_id": ["p1"],
                "position": ["WR"],
                "team": ["KC"],
                "points": [season_points_total],
            }
        )
    }
    kwargs = dict(
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
        actuals_to_date=pl.DataFrame(
            {"player_id": ["p1"], "actual_points_to_date": [actuals_to_date]}
        ),
        season_points_by_source=season_points,
        trend_by_source={"espn": "flat"},
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        now=datetime(2026, 10, 1, tzinfo=UTC),
        train_start=2015,
        min_train_rows=1,
        lightgbm_params=None,
        code_version="abc",
        offline=True,
    )
    return predict_module_ref, fake_project_week, kwargs


def test_project_week_range_anchor_week_not_double_counted(monkeypatch) -> None:
    """The season-long consensus LEVEL still includes the anchor week's
    own value (it's "full season minus actuals strictly before
    from_week", and from_week hasn't been played yet) -- if that level
    were allocated to future weeks without first subtracting the anchor
    week's own already-known value (from `current`), the anchor week's
    points would be counted twice: once directly via `current`, once
    more spread across future weeks. Proven by making the anchor's own
    mean exactly equal to the post-actuals level (115.0), so a correct
    subtraction leaves ~0 to spread across future weeks -- the bug would
    instead spread the full, un-reduced 115.0 across them."""
    predict_module, fake_project_week, kwargs = _ros_fixtures(
        anchor_mean=115.0, season_points_total=165.0, actuals_to_date=50.0
    )
    monkeypatch.setattr(predict_module, "project_week", fake_project_week)

    result = predict_ros.project_week_range(settings=None, **kwargs)

    future_rows = result.filter(~pl.col("is_current_week"))
    assert future_rows["mean"].sum() == pytest.approx(0.0, abs=1e-6)


def test_project_week_range_full_level_returned_when_horizon_matches_season_end(
    monkeypatch, tmp_path
) -> None:
    """Complement to the "squeezed into a short horizon" behaviour proven
    by `test_project_week_range_future_week_uses_shape_not_project_week`
    (only 2 of 13 real remaining weeks requested, so only ~2/13 of the
    level comes back): when `settings.ros.season_end_week` is set to
    match the caller's own `through_week` exactly, `full_future_weeks ==
    requested_future_weeks`, so the ENTIRE post-anchor-subtraction level
    must come back, not a fraction of it -- a direct regression test that
    `season_end_week` genuinely drives the allocation window rather than
    being dead configuration (as it was before this fix)."""
    predict_module, fake_project_week, kwargs = _ros_fixtures(
        anchor_mean=15.0, season_points_total=150.0, actuals_to_date=50.0
    )
    monkeypatch.setattr(predict_module, "project_week", fake_project_week)
    settings = _minimal_settings(tmp_path, season_end_week=7)  # matches through_week=7

    result = predict_ros.project_week_range(settings=settings, **kwargs)

    future_rows = result.filter(~pl.col("is_current_week"))
    # level = 150 - 50 (actuals) - 15 (anchor's own mean) = 85.0, entirely
    # returned since full_future_weeks == requested_future_weeks == [6, 7].
    assert future_rows["mean"].sum() == pytest.approx(85.0, rel=0.01)
