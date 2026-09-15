from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.tools.pipeline_health import HealthCheck, PipelineHealth

runner = CliRunner()

_LEAGUE = LeagueConfig(
    slug="weekly-league",
    display_name="Weekly League",
    is_primary=True,
    league_id="123",
    season=2026,
    league_cache={"scoring_settings": {}},
    overrides={},
)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
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


def _patch_refresh_dependencies(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, *, health_status: str = "healthy"
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(cli, "load_league", lambda slug: _LEAGUE)
    monkeypatch.setattr(
        cli,
        "refresh_features",
        lambda *args, **kwargs: {
            "player_week_stats": 100,
            "current_actual_rows": 10,
            "features": 200,
        },
    )
    monkeypatch.setattr(cli.sleeper, "fetch_user", lambda *args, **kwargs: Path("user.json"))
    monkeypatch.setattr(cli.sleeper, "fetch_rosters", lambda *args, **kwargs: Path("rosters.json"))
    monkeypatch.setattr(
        cli.sleeper, "fetch_matchups", lambda *args, **kwargs: Path("matchups.json")
    )
    monkeypatch.setattr(cli, "project_command", lambda **kwargs: None)
    monkeypatch.setattr(
        cli,
        "refresh_weekly_alerts",
        lambda *args, **kwargs: (settings.data_root / "alerts.json", 0),
    )
    monkeypatch.setattr(cli, "log_backfill_command", lambda **kwargs: None)
    monkeypatch.setattr(cli, "log_week_command", lambda **kwargs: None)
    monkeypatch.setattr(
        cli,
        "inspect_weekly_pipeline",
        lambda *args, **kwargs: PipelineHealth(
            status=health_status,  # type: ignore[arg-type]
            checks=(HealthCheck("Weekly projections", health_status, "fixture"),),  # type: ignore[arg-type]
        ),
    )


def test_weekly_refresh_runs_stack_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    _patch_refresh_dependencies(monkeypatch, fixture_settings)

    result = runner.invoke(
        cli.app,
        ["refresh", "weekly", "--week", "2", "--run-label", "tuesday", "--offline"],
    )

    assert result.exit_code == 0, result.output
    latest = fixture_settings.data_root / "outputs" / _LEAGUE.slug / "refresh_runs" / "latest.json"
    payload = json.loads(latest.read_text())
    assert payload["status"] == "healthy"
    assert {step["name"] for step in payload["steps"]} >= {
        "sleeper",
        "features",
        "projections",
        "decision_alerts",
        "prior_week_actuals",
        "prediction_log",
    }


def test_weekly_refresh_records_degraded_prior_actuals_without_hiding_projection(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    _patch_refresh_dependencies(monkeypatch, fixture_settings)
    monkeypatch.setattr(
        cli,
        "log_backfill_command",
        lambda **kwargs: (_ for _ in ()).throw(cli.prediction_log.InvalidActualsError("all zero")),
    )

    result = runner.invoke(cli.app, ["refresh", "weekly", "--week", "2", "--offline"])

    assert result.exit_code == 0, result.output
    latest = fixture_settings.data_root / "outputs" / _LEAGUE.slug / "refresh_runs" / "latest.json"
    payload = json.loads(latest.read_text())
    assert payload["status"] == "degraded"
    actuals = next(step for step in payload["steps"] if step["name"] == "prior_week_actuals")
    assert actuals["status"] == "degraded"
