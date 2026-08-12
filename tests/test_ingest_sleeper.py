import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ffapp.cache.offline import OfflineCacheMiss, StaleCacheError, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
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


def _age_stamp(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _stale_league_meta() -> dict[str, str]:
    return {
        "source": "sleeper",
        "fetched_at_utc": _age_stamp(200),
        "cache_key": "sleeper_league",
    }


def test_fetch_league_online_writes_raw_json_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: {"league_id": "111", "scoring_settings": {}})

    path = sleeper.fetch_league("111", offline=False, settings=settings)

    assert path.exists()
    assert json.loads(path.read_text()) == {"league_id": "111", "scoring_settings": {}}
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "sleeper"
    assert meta["cache_key"] == "sleeper_league"


def test_fetch_league_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom(path: str) -> None:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(sleeper, "_get", _boom)
    path = settings.cache.root / "sleeper" / "league_111.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"league_id": "111"}))
    write_sidecar(path, source="sleeper", call="/league/111", cache_key="sleeper_league")

    result = sleeper.fetch_league("111", offline=True, settings=settings)

    assert result == path


def test_fetch_league_offline_without_cache_raises_offline_cache_miss(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: pytest.fail("should not fetch"))

    with pytest.raises(OfflineCacheMiss) as exc_info:
        sleeper.fetch_league("111", offline=True, settings=settings)

    message = str(exc_info.value)
    assert "sleeper" in message
    assert "league" in message
    assert "111" in message
    assert "cache warm" in message


def test_fetch_league_offline_with_stale_cache_logs_warning(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FFAPP_CACHE_STRICT", raising=False)
    path = settings.cache.root / "sleeper" / "league_111.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"league_id": "111"}))
    sidecar_path(path).write_text(json.dumps(_stale_league_meta()))

    with caplog.at_level(logging.WARNING):
        result = sleeper.fetch_league("111", offline=True, settings=settings)

    assert result == path
    assert any("stale" in record.message.lower() for record in caplog.records)


def test_fetch_league_offline_with_stale_cache_and_strict_env_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    path = settings.cache.root / "sleeper" / "league_111.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"league_id": "111"}))
    sidecar_path(path).write_text(json.dumps(_stale_league_meta()))

    with pytest.raises(StaleCacheError):
        sleeper.fetch_league("111", offline=True, settings=settings)


def test_fetch_user_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or {"user_id": "abc"})

    path = sleeper.fetch_user("Maybe17", offline=False, settings=settings)

    assert calls == ["/user/Maybe17"]
    assert path.name == "user_Maybe17.json"


def test_fetch_leagues_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_leagues("abc", 2026, offline=False, settings=settings)

    assert calls == ["/user/abc/leagues/nfl/2026"]
    assert path.name == "leagues_abc_2026.json"


def test_fetch_rosters_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: [])

    path = sleeper.fetch_rosters("111", offline=False, settings=settings)

    assert path.name == "rosters_111.json"


def test_fetch_users_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: [])

    path = sleeper.fetch_users("111", offline=False, settings=settings)

    assert path.name == "users_111.json"


def test_fetch_matchups_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_matchups("111", 14, offline=False, settings=settings)

    assert calls == ["/league/111/matchups/14"]
    assert path.name == "matchups_111_w14.json"


def test_fetch_transactions_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_transactions("111", 14, offline=False, settings=settings)

    assert calls == ["/league/111/transactions/14"]
    assert path.name == "transactions_111_w14.json"


def test_fetch_drafts_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: [])

    path = sleeper.fetch_drafts("111", offline=False, settings=settings)

    assert path.name == "drafts_111.json"


def test_fetch_draft_picks_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_draft_picks("d1", offline=False, settings=settings)

    assert calls == ["/draft/d1/picks"]
    assert path.name == "draft_picks_d1.json"


def test_fetch_traded_picks_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_traded_picks("111", offline=False, settings=settings)

    assert calls == ["/league/111/traded_picks"]
    assert path.name == "traded_picks_111.json"


def test_fetch_trending_writes_expected_path(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(sleeper, "_get", lambda path: calls.append(path) or [])

    path = sleeper.fetch_trending("add", offline=False, settings=settings)

    assert calls == ["/players/nfl/trending/add?lookback_hours=24&limit=25"]
    assert path.name == "trending_add.json"


def test_fetch_players_second_call_within_24h_does_not_refetch(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    call_count = 0

    def _get(path: str) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"1": {"full_name": "Test Player"}}

    monkeypatch.setattr(sleeper, "_get", _get)

    first = sleeper.fetch_players(offline=False, settings=settings)
    second = sleeper.fetch_players(offline=False, settings=settings)

    assert first == second
    assert call_count == 1


def test_fetch_players_force_refetches_even_within_24h(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    call_count = 0

    def _get(path: str) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"1": {"full_name": "Test Player"}}

    monkeypatch.setattr(sleeper, "_get", _get)

    sleeper.fetch_players(offline=False, settings=settings)
    sleeper.fetch_players(offline=False, settings=settings, force=True)

    assert call_count == 2


def test_fetch_players_offline_without_cache_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(sleeper, "_get", lambda path: pytest.fail("should not fetch"))

    with pytest.raises(OfflineCacheMiss):
        sleeper.fetch_players(offline=True, settings=settings)
