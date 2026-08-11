from pathlib import Path

import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.cache import registry as cache_registry
from ffapp.cache.registry import CacheRequirement, DiscoveredLeague
from ffapp.config import CacheSettings, Settings

runner = CliRunner()


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"sleeper_league": 168},
            warn_on_stale=True,
        ),
    )


def test_ingest_sleeper_without_discover_flag_exits_nonzero() -> None:
    result = runner.invoke(cli.app, ["ingest", "sleeper", "--season", "2026"])

    assert result.exit_code == 1
    assert "--discover" in result.output


def test_ingest_sleeper_discover_refuses_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFAPP_OFFLINE", "1")

    result = runner.invoke(cli.app, ["ingest", "sleeper", "--season", "2026", "--discover"])

    assert result.exit_code == 1
    assert "network" in result.output.lower()


def test_ingest_sleeper_discover_writes_leagues_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(
        cache_registry,
        "discover_leagues",
        lambda season, *, settings, leagues_dir=None: [
            DiscoveredLeague(
                slug="main-ppr", league_id="111", path=Path("config/leagues/main-ppr.yml")
            )
        ],
    )

    result = runner.invoke(
        cli.app, ["ingest", "sleeper", "--season", "2026", "--discover", "--no-offline"]
    )

    assert result.exit_code == 0
    assert "main-ppr" in result.output
    assert "is_primary" in result.output


def test_cache_warm_without_all_leagues_flag_exits_nonzero() -> None:
    result = runner.invoke(cli.app, ["cache", "warm", "--season", "2026"])

    assert result.exit_code == 1
    assert "--all-leagues" in result.output


def test_cache_warm_refuses_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFAPP_OFFLINE", "1")

    result = runner.invoke(cli.app, ["cache", "warm", "--season", "2026", "--all-leagues"])

    assert result.exit_code == 1
    assert "network" in result.output.lower()


def test_cache_warm_success_when_online(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    called = {}
    monkeypatch.setattr(
        cache_registry,
        "warm_sleeper",
        lambda season, *, settings, leagues_dir=None: called.update(season=season),
    )

    result = runner.invoke(
        cli.app, ["cache", "warm", "--season", "2026", "--all-leagues", "--no-offline"]
    )

    assert result.exit_code == 0
    assert called == {"season": 2026}


def test_cache_status_reports_nothing_cached(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cache_registry, "cache_status", lambda settings: [])

    result = runner.invoke(cli.app, ["cache", "status"])

    assert result.exit_code == 0
    assert "nothing cached" in result.output.lower()


def test_cache_status_prints_each_row(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(
        cache_registry,
        "cache_status",
        lambda settings: [
            {
                "artifact": "league_111.json",
                "source": "sleeper",
                "fetched_at_utc": "2026-08-11T00:00:00+00:00",
                "age_hours": 1.0,
                "verdict": "fresh",
            }
        ],
    )

    result = runner.invoke(cli.app, ["cache", "status"])

    assert result.exit_code == 0
    assert "league_111.json" in result.output
    assert "fresh" in result.output


def test_cache_verify_unknown_task_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    result = runner.invoke(cli.app, ["cache", "verify", "--for-task", "0.99"])

    assert result.exit_code == 1
    assert "0.99" in result.output


def test_cache_verify_reports_missing_requirement_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    req = CacheRequirement(
        description="league cache", check=lambda s: False, warm_hint="ffapp cache warm"
    )
    monkeypatch.setattr(cache_registry, "cache_verify", lambda task_id, *, settings: [(req, False)])

    result = runner.invoke(cli.app, ["cache", "verify", "--for-task", "0.5"])

    assert result.exit_code == 1
    assert "MISSING" in result.output
    assert "ffapp cache warm" in result.output


def test_cache_verify_all_satisfied_exits_zero(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    req = CacheRequirement(
        description="league cache", check=lambda s: True, warm_hint="ffapp cache warm"
    )
    monkeypatch.setattr(cache_registry, "cache_verify", lambda task_id, *, settings: [(req, True)])

    result = runner.invoke(cli.app, ["cache", "verify", "--for-task", "0.5"])

    assert result.exit_code == 0
    assert "OK" in result.output
