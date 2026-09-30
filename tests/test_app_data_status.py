from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from ffapp.app import data_status
from ffapp.app.data_status import (
    artifact_freshness,
    build_refresh_command,
    load_latest_alerts,
    load_refresh_manifest,
    next_scheduled_refresh,
    run_label_for_date,
    sync_live_league_state,
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


def test_build_refresh_command_supports_all_leagues() -> None:
    command = build_refresh_command(None, executable="uv", run_label="sunday")
    assert "--all-leagues" in command
    assert "--league" not in command


def test_next_scheduled_refresh_uses_central_windows() -> None:
    next_run, label = next_scheduled_refresh(datetime(2026, 9, 22, 13, tzinfo=UTC))
    assert label == "Thursday"
    assert next_run.weekday() == 3
    assert next_run.hour == 7


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


def test_live_sync_refreshes_fast_sleeper_inputs_without_model_rebuild(monkeypatch) -> None:
    calls: list[str] = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append(name)
            return Path("cached.json")

        return fake

    for name in (
        "fetch_players",
        "fetch_user",
        "fetch_rosters",
        "fetch_users",
        "fetch_matchups",
        "fetch_transactions",
    ):
        monkeypatch.setattr(data_status.sleeper, name, record(name))
    monkeypatch.setattr(
        data_status,
        "refresh_weekly_alerts",
        lambda settings, league, season, week: (Path("alerts.json"), 2),
    )

    result = sync_live_league_state(
        SimpleNamespace(sleeper_username="jacob"),
        SimpleNamespace(league_id="123", season=2026),
        3,
    )

    assert result == 2
    assert calls == [
        "fetch_players",
        "fetch_user",
        "fetch_rosters",
        "fetch_users",
        "fetch_matchups",
        "fetch_transactions",
    ]
