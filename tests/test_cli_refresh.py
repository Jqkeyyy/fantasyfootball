from __future__ import annotations

import json
from pathlib import Path

import polars as pl
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

_SECOND_LEAGUE = LeagueConfig(
    slug="second-weekly-league",
    display_name="Second Weekly League",
    is_primary=False,
    league_id="456",
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
    schedule_path = settings.data_root / "interim" / "schedule.parquet"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"week": [1, 2, 3]}).write_parquet(schedule_path)
    user_path = settings.data_root / "user.json"
    user_path.write_text('{"user_id": "manager"}')
    rosters_path = settings.data_root / "rosters.json"
    rosters_path.write_text('[{"roster_id": 1, "owner_id": "manager", "players": ["p1"]}]')
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
    monkeypatch.setattr(cli.sleeper, "fetch_user", lambda *args, **kwargs: user_path)
    monkeypatch.setattr(cli.sleeper, "fetch_rosters", lambda *args, **kwargs: rosters_path)
    monkeypatch.setattr(
        cli.sleeper, "fetch_matchups", lambda *args, **kwargs: Path("matchups.json")
    )
    monkeypatch.setattr(
        cli.sleeper, "fetch_transactions", lambda *args, **kwargs: Path("transactions.json")
    )
    monkeypatch.setattr(cli, "project_command", lambda **kwargs: None)
    monkeypatch.setattr(cli, "rankings_ros_command", lambda **kwargs: None)
    monkeypatch.setattr(cli.sos, "full_season_weeks", lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(
        cli,
        "refresh_weekly_alerts",
        lambda *args, **kwargs: (settings.data_root / "alerts.json", 0),
    )
    monkeypatch.setattr(
        cli.waiver_history,
        "refresh_waiver_history",
        lambda *args, **kwargs: (settings.data_root / "waivers.parquet", 0),
    )
    monkeypatch.setattr(
        cli,
        "refresh_news_events",
        lambda *args, **kwargs: {"status": "skipped", "reason": "fixture"},
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
        "news",
        "projections",
        "decision_alerts",
        "ros_decisions",
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


def test_weekly_refresh_all_leagues_rebuilds_shared_features_once(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    _patch_refresh_dependencies(monkeypatch, fixture_settings)
    leagues = {_LEAGUE.slug: _LEAGUE, _SECOND_LEAGUE.slug: _SECOND_LEAGUE}
    monkeypatch.setattr(cli, "load_all_leagues", lambda: list(leagues.values()))
    monkeypatch.setattr(cli, "load_league", lambda slug: leagues[slug])
    feature_calls: list[str] = []
    news_calls: list[str] = []
    projected_leagues: list[str] = []
    monkeypatch.setattr(
        cli,
        "refresh_features",
        lambda settings, league, **kwargs: (
            feature_calls.append(league.slug)
            or {"player_week_stats": 100, "current_actual_rows": 10, "features": 200}
        ),
    )
    monkeypatch.setattr(
        cli,
        "project_command",
        lambda **kwargs: projected_leagues.append(str(kwargs["league"])),
    )
    monkeypatch.setattr(
        cli,
        "refresh_news_events",
        lambda *args, **kwargs: (
            news_calls.append("refresh") or {"status": "healthy", "new_items": 0}
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "refresh",
            "weekly",
            "--all-leagues",
            "--week",
            "2",
            "--skip-backfill",
            "--skip-ros",
            "--offline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert feature_calls == [_LEAGUE.slug]
    assert news_calls == ["refresh"]
    assert projected_leagues == [_LEAGUE.slug, _SECOND_LEAGUE.slug]
    for league in leagues.values():
        assert (
            fixture_settings.data_root / "outputs" / league.slug / "refresh_runs" / "latest.json"
        ).exists()


def test_weekly_refresh_rejects_league_with_all_leagues(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    _patch_refresh_dependencies(monkeypatch, fixture_settings)

    result = runner.invoke(
        cli.app,
        ["refresh", "weekly", "--all-leagues", "--league", _LEAGUE.slug, "--week", "2"],
    )

    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_weekly_refresh_caches_recent_chopped_transactions(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    _patch_refresh_dependencies(monkeypatch, fixture_settings)
    chopped = LeagueConfig(
        slug=_LEAGUE.slug,
        display_name=_LEAGUE.display_name,
        is_primary=True,
        league_id=_LEAGUE.league_id,
        season=2026,
        league_cache={"scoring_settings": {}, "league_type": 3},
        overrides={},
    )
    monkeypatch.setattr(cli, "load_primary_league", lambda: chopped)
    calls: list[int] = []
    monkeypatch.setattr(
        cli.sleeper,
        "fetch_transactions",
        lambda league_id, week, **kwargs: calls.append(week) or Path("transactions.json"),
    )

    result = runner.invoke(
        cli.app,
        ["refresh", "weekly", "--week", "2", "--skip-backfill", "--skip-ros", "--offline"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [1, 2]
