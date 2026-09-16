from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ffapp.app.data_status import artifact_freshness, build_refresh_command


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
    command = build_refresh_command("bdff-chopped", executable="uv")
    assert command == [
        "uv",
        "run",
        "ffapp",
        "refresh",
        "weekly",
        "--league",
        "bdff-chopped",
        "--no-offline",
    ]
