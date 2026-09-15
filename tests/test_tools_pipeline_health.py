from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.config import CacheSettings, Settings
from ffapp.tools.pipeline_health import inspect_weekly_pipeline


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
    )


def test_missing_projection_artifact_is_failed(tmp_path: Path) -> None:
    health = inspect_weekly_pipeline(_settings(tmp_path), "league", 2026, 2)

    assert health.status == "failed"
    assert health.checks[0].status == "failed"


def test_partial_projections_and_failed_source_are_degraded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = tmp_path / "outputs" / "league"
    output.mkdir(parents=True)
    pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [2, 2],
            "mean": [10.0, None],
            "projection_source": ["espn_weekly", "espn_weekly"],
        }
    ).write_parquet(output / "projections.parquet")
    log_dir = output / "prediction_log"
    log_dir.mkdir()
    pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [2, 2],
            "source": ["espn", "draftsharks"],
            "fetched_at_utc": ["2026-09-15T12:00:00+00:00"] * 2,
            "fetch_error": [None, "0 rows parsed"],
        }
    ).write_parquet(log_dir / "source_fetches.parquet")

    health = inspect_weekly_pipeline(
        settings,
        "league",
        2026,
        2,
        now=datetime.now(UTC),
        stale_after_hours=100_000,
    )

    assert health.status == "degraded"
    details = {check.name: check.detail for check in health.checks}
    assert "1/2 usable" in details["Weekly projections"]
    assert "draftsharks" in details["Source fetches"]
    source_check = next(c for c in health.checks if c.name == "Source fetches")
    assert source_check.status == "healthy"


def test_deep_projection_omissions_do_not_degrade_relevant_coverage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = tmp_path / "outputs" / "league"
    output.mkdir(parents=True)
    pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "mean": [10.0, None],
            "projection_source": ["espn_weekly", "espn_weekly"],
        }
    ).write_parquet(output / "projections.parquet")
    pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "projected": [True, False],
            "fantasy_relevant": [True, False],
        }
    ).write_parquet(output / "projection_coverage.parquet")

    health = inspect_weekly_pipeline(settings, "league", 2026, 1)

    projection_check = next(c for c in health.checks if c.name == "Weekly projections")
    assert projection_check.status == "healthy"
    assert "0 fantasy-relevant missing" in projection_check.detail
