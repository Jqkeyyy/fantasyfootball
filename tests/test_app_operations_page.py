from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.app.operations_page import league_operation_row, recent_run_rows
from ffapp.config import CacheSettings, LeagueConfig, Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="manager",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
    )


def _league() -> LeagueConfig:
    return LeagueConfig(
        slug="league",
        display_name="League",
        is_primary=True,
        league_id="123",
        season=2026,
        league_cache={},
        overrides={},
    )


def test_league_operation_row_combines_manifest_coverage_and_alerts(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "league"
    (output / "refresh_runs").mkdir(parents=True)
    (output / "alerts").mkdir()
    (output / "refresh_runs" / "latest.json").write_text(
        json.dumps(
            {
                "status": "healthy",
                "season": 2026,
                "week": 3,
                "generated_at_utc": "2026-09-22T12:00:00+00:00",
                "steps": [],
            }
        )
    )
    (output / "alerts" / "latest.json").write_text(json.dumps({"alerts": [{"kind": "swap"}]}))
    pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [3, 3],
            "projected": [True, False],
        }
    ).write_parquet(output / "projection_coverage.parquet")

    row = league_operation_row(
        _settings(tmp_path),
        _league(),
        now=datetime(2026, 9, 22, 14, tzinfo=UTC),
    )

    assert row["status"] == "healthy"
    assert row["coverage"] == "1/2"
    assert row["decision_alerts"] == 1
    assert row["age_hours"] == 2.0


def test_recent_run_rows_reports_problem_steps(tmp_path: Path) -> None:
    directory = tmp_path / "outputs" / "league" / "refresh_runs"
    directory.mkdir(parents=True)
    (directory / "2026-w03-20260922T120000Z.json").write_text(
        json.dumps(
            {
                "status": "degraded",
                "generated_at_utc": "2026-09-22T12:00:00+00:00",
                "season": 2026,
                "week": 3,
                "steps": [{"name": "source", "status": "degraded"}],
            }
        )
    )

    rows = recent_run_rows(_settings(tmp_path), _league())

    assert rows[0]["problems"] == "source"
