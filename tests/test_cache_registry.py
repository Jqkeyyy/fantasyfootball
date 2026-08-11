import json
from pathlib import Path

import pytest

from ffapp.cache import registry
from ffapp.cache.offline import write_sidecar
from ffapp.config import CacheSettings, Settings, load_league
from ffapp.ingest import sleeper


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"sleeper_league": 168, "sleeper_rosters": 24},
            warn_on_stale=True,
        ),
    )


def _mock_sleeper(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/user/fixture_user": {"user_id": "u1"},
        "/user/u1/leagues/nfl/2026": [
            {"league_id": "111"},
            {"league_id": "222"},
        ],
        "/league/111": {
            "league_id": "111",
            "name": "Main League!",
            "total_rosters": 10,
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
            "scoring_settings": {"rec": 1},
            "settings": {"waiver_type": 2, "waiver_budget": 100, "playoff_week_start": 15},
        },
        "/league/222": {
            "league_id": "222",
            "name": "Main League!",  # duplicate name on purpose, to test slug de-duplication
            "total_rosters": 12,
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
            "scoring_settings": {"rec": 0.5},
            "settings": {"waiver_type": 1, "waiver_budget": None, "playoff_week_start": 15},
        },
    }
    monkeypatch.setattr(sleeper, "_get", lambda path: responses[path])


def test_discover_leagues_writes_a_stub_per_league_with_deduplicated_slugs(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    _mock_sleeper(monkeypatch)
    leagues_dir = tmp_path / "leagues"

    discovered = registry.discover_leagues(2026, settings=settings, leagues_dir=leagues_dir)

    slugs = {d.slug for d in discovered}
    assert slugs == {"main-league", "main-league-2"}
    main = load_league("main-league", leagues_dir=leagues_dir)
    assert main.league_cache["scoring_settings"]["rec"] == 1
    assert main.league_cache["waiver_type"] == 2


def test_discover_leagues_reuses_existing_slug_for_a_known_league_id(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    _mock_sleeper(monkeypatch)
    leagues_dir = tmp_path / "leagues"

    def _slug_for(discovered: list[registry.DiscoveredLeague], league_id: str) -> str:
        return next(d.slug for d in discovered if d.league_id == league_id)

    first = registry.discover_leagues(2026, settings=settings, leagues_dir=leagues_dir)
    slug_for_111 = _slug_for(first, "111")

    second = registry.discover_leagues(2026, settings=settings, leagues_dir=leagues_dir)
    slug_for_111_again = _slug_for(second, "111")

    assert slug_for_111 == slug_for_111_again


def test_cache_status_reports_age_and_freshness_verdict(settings: Settings) -> None:
    path = settings.cache.root / "sleeper" / "league_111.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"league_id": "111"}))
    write_sidecar(path, source="sleeper", call="/league/111", cache_key="sleeper_league")

    rows = registry.cache_status(settings)

    assert len(rows) == 1
    assert rows[0]["verdict"] == "fresh"
    assert rows[0]["source"] == "sleeper"


def test_cache_status_returns_empty_list_when_nothing_cached(settings: Settings) -> None:
    assert registry.cache_status(settings) == []


def test_cache_verify_raises_for_unregistered_task(settings: Settings) -> None:
    with pytest.raises(ValueError, match="0.99"):
        registry.cache_verify("0.99", settings=settings)
