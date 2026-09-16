from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ffapp.app.data_status import (
    artifact_freshness,
    build_refresh_command,
    load_latest_alerts,
    load_refresh_manifest,
    run_label_for_date,
)


def test_artifact_freshness_reports_missing(tmp_path: Path) -> None:
    result = artifact_freshness("ROS", tmp_path / "missing.parquet")
    assert result.exists is False
    assert result.stale is True


def test_artifact_freshness_marks_old_file_stale(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"x")
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    result = artifact_freshness(
        "ROS", path, now=modified + timedelta(hours=73), stale_after_hours=72
    )
    assert result.exists is True
    assert result.age_hours == 73.0
    assert result.stale is True


def test_build_refresh_command_is_argument_safe_and_live() -> None:
    command = build_refresh_command("bdff-chopped", executable="uv", run_label="thursday")
    assert command == [
        "uv",
        "run",
        "ffapp",
        "refresh",
        "weekly",
        "--league",
        "bdff-chopped",
        "--run-label",
        "thursday",
        "--no-offline",
    ]


def test_run_label_for_date_uses_standard_weekly_windows() -> None:
    assert run_label_for_date(datetime(2026, 9, 15, tzinfo=UTC)) == "tuesday"
    assert run_label_for_date(datetime(2026, 9, 16, tzinfo=UTC)) == "thursday"
    assert run_label_for_date(datetime(2026, 9, 19, tzinfo=UTC)) == "sunday"


def test_load_latest_alerts_is_resilient(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text('{"alerts": [{"message": "Replace injured starter"}]}')
    assert load_latest_alerts(path)[0]["message"] == "Replace injured starter"
    path.write_text("not-json")
    assert load_latest_alerts(path) == []


def test_load_refresh_manifest_returns_structured_status(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text('{"status": "degraded", "steps": []}')
    assert load_refresh_manifest(path) == {"status": "degraded", "steps": []}
