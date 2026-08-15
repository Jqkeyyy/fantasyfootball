from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.cache import registry as cache_registry
from ffapp.cache.registry import CacheRequirement, DiscoveredLeague
from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.draft import board as draft_board
from ffapp.ingest import rankings

runner = CliRunner()

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={"total_rosters": 10},
    overrides={},
)


def _fake_source(height: int) -> pl.DataFrame:
    return pl.DataFrame({"player_name": [f"P{i}" for i in range(height)]})


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


# --- ingest rankings (SPEC-ADDENDUM-03.md §E's "morning of" runbook step) ---------


def test_ingest_rankings_refreshes_every_source_and_reports_counts(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(
        draft_board,
        "fetch_point_sources",
        lambda season, *, offline, settings: [_fake_source(5), _fake_source(3)],
    )
    monkeypatch.setattr(
        draft_board,
        "fetch_rank_sources",
        lambda season, *, offline, settings: [_fake_source(5)],
    )
    monkeypatch.setattr(
        rankings, "fetch_adp", lambda season, *, teams, offline, settings: Path("adp.json")
    )

    result = runner.invoke(cli.app, ["ingest", "rankings", "--no-offline"])

    assert result.exit_code == 0
    assert f"Point sources: 2/{len(draft_board.POINT_SOURCE_NAMES)}" in result.output
    assert f"Rank sources: 1/{len(draft_board.RANK_SOURCE_NAMES)}" in result.output
    assert "ADP: refreshed" in result.output


def test_ingest_rankings_defaults_season_to_the_leagues_own_season(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    seasons_seen = []
    monkeypatch.setattr(
        draft_board,
        "fetch_point_sources",
        lambda season, *, offline, settings: seasons_seen.append(season) or [_fake_source(1)],
    )
    monkeypatch.setattr(
        draft_board, "fetch_rank_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    monkeypatch.setattr(
        rankings, "fetch_adp", lambda season, *, teams, offline, settings: Path("adp.json")
    )

    result = runner.invoke(cli.app, ["ingest", "rankings", "--no-offline"])

    assert result.exit_code == 0
    assert seasons_seen == [2026]


def test_ingest_rankings_season_override_is_respected(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    seasons_seen = []
    monkeypatch.setattr(
        draft_board,
        "fetch_point_sources",
        lambda season, *, offline, settings: seasons_seen.append(season) or [_fake_source(1)],
    )
    monkeypatch.setattr(
        draft_board, "fetch_rank_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    monkeypatch.setattr(
        rankings, "fetch_adp", lambda season, *, teams, offline, settings: Path("adp.json")
    )

    result = runner.invoke(cli.app, ["ingest", "rankings", "--season", "2025", "--no-offline"])

    assert result.exit_code == 0
    assert seasons_seen == [2025]


def test_ingest_rankings_passes_the_leagues_own_team_count_to_adp(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(
        draft_board, "fetch_point_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    monkeypatch.setattr(
        draft_board, "fetch_rank_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    teams_seen = []
    monkeypatch.setattr(
        rankings,
        "fetch_adp",
        lambda season, *, teams, offline, settings: teams_seen.append(teams) or Path("adp.json"),
    )

    result = runner.invoke(cli.app, ["ingest", "rankings", "--no-offline"])

    assert result.exit_code == 0
    assert teams_seen == [10]  # _LEAGUE's own total_rosters


def test_ingest_rankings_reports_when_every_point_source_fails(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(draft_board, "fetch_point_sources", lambda season, *, offline, settings: [])
    monkeypatch.setattr(
        draft_board, "fetch_rank_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    monkeypatch.setattr(
        rankings, "fetch_adp", lambda season, *, teams, offline, settings: Path("adp.json")
    )

    result = runner.invoke(cli.app, ["ingest", "rankings", "--no-offline"])

    assert result.exit_code == 1
    assert "Every per-stat rankings source failed" in result.output


def test_ingest_rankings_reports_when_adp_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(
        draft_board, "fetch_point_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )
    monkeypatch.setattr(
        draft_board, "fetch_rank_sources", lambda season, *, offline, settings: [_fake_source(1)]
    )

    def raise_stale(season: object, *, teams: object, offline: object, settings: object) -> Path:
        raise RuntimeError("ADP is 36h stale under FFAPP_CACHE_STRICT=1")

    monkeypatch.setattr(rankings, "fetch_adp", raise_stale)

    result = runner.invoke(cli.app, ["ingest", "rankings", "--no-offline"])

    assert result.exit_code == 1
    assert "ADP" in result.output
    assert "36h stale" in result.output
