import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ffapp.cache.offline import OfflineCacheMiss, StaleCacheError, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import nflverse

FIXTURE_CSV = (
    "mfl_id,gsis_id,sleeper_id,pfr_id,espn_id,name,merge_name,position,team,birthdate\n"
    "1,00-0031234,421,MahoPa00,3139477,Patrick Mahomes,patrick mahomes,QB,KC,1995-09-17\n"
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"nflverse_player_ids": 168},
            warn_on_stale=True,
        ),
    )


def _age_stamp(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _stale_meta() -> dict[str, str]:
    return {
        "source": "nflverse",
        "fetched_at_utc": _age_stamp(200),
        "cache_key": "nflverse_player_ids",
    }


def test_fetch_player_ids_online_writes_raw_csv_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(nflverse, "_get_csv", lambda: FIXTURE_CSV)

    path = nflverse.fetch_player_ids(offline=False, settings=settings)

    assert path.exists()
    assert path.read_text() == FIXTURE_CSV
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "nflverse"
    assert meta["cache_key"] == "nflverse_player_ids"


def test_fetch_player_ids_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom() -> str:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(nflverse, "_get_csv", _boom)
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    write_sidecar(
        path, source="nflverse", call=nflverse.CROSSWALK_URL, cache_key="nflverse_player_ids"
    )

    result = nflverse.fetch_player_ids(offline=True, settings=settings)

    assert result == path


def test_fetch_player_ids_offline_without_cache_raises_offline_cache_miss(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(nflverse, "_get_csv", lambda: pytest.fail("should not fetch"))

    with pytest.raises(OfflineCacheMiss) as exc_info:
        nflverse.fetch_player_ids(offline=True, settings=settings)

    message = str(exc_info.value)
    assert "nflverse" in message
    assert "player_ids" in message
    assert "cache warm" in message or "ingest" in message


def test_fetch_player_ids_offline_with_stale_cache_logs_warning(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FFAPP_CACHE_STRICT", raising=False)
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    sidecar_path(path).write_text(json.dumps(_stale_meta()))

    with caplog.at_level(logging.WARNING):
        result = nflverse.fetch_player_ids(offline=True, settings=settings)

    assert result == path
    assert any("stale" in record.message.lower() for record in caplog.records)


def test_fetch_player_ids_offline_with_stale_cache_and_strict_env_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    sidecar_path(path).write_text(json.dumps(_stale_meta()))

    with pytest.raises(StaleCacheError):
        nflverse.fetch_player_ids(offline=True, settings=settings)


# --- fetch_player_stats / fetch_team_stats / fetch_schedules -------------------
# All three share `_fetch_nflreadpy_parquet`, so one function's tests cover the
# mechanism; the others just confirm they're wired to the right nflreadpy loader.


@pytest.fixture
def stats_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"nflverse_player_stats": 168},
            warn_on_stale=True,
        ),
    )


def test_fetch_player_stats_online_writes_parquet_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"player_id": ["00-0031234"], "week": [1], "passing_yards": [250]})
    monkeypatch.setattr(nflverse.nfl, "load_player_stats", lambda seasons: fixture_df)

    path = nflverse.fetch_player_stats(2025, offline=False, settings=stats_settings)

    assert path.exists()
    assert pl.read_parquet(path).equals(fixture_df)
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "nflverse"
    assert meta["cache_key"] == "nflverse_player_stats"
    assert meta["rows"] == 1


def test_fetch_player_stats_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    def _boom(seasons: list[int]) -> pl.DataFrame:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(nflverse.nfl, "load_player_stats", _boom)
    path = stats_settings.cache.root / "nflverse" / "player_stats_2025.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"player_id": ["00-0031234"]}).write_parquet(path)
    write_sidecar(path, source="nflverse", call="x", cache_key="nflverse_player_stats")

    result = nflverse.fetch_player_stats(2025, offline=True, settings=stats_settings)

    assert result == path


def test_fetch_player_stats_offline_without_cache_raises_offline_cache_miss(
    stats_settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        nflverse.fetch_player_stats(2025, offline=True, settings=stats_settings)

    message = str(exc_info.value)
    assert "nflverse" in message
    assert "season=2025" in message


def test_fetch_team_stats_online_calls_load_team_stats(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"team": ["KC"], "week": [1], "def_sacks": [3]})
    monkeypatch.setattr(nflverse.nfl, "load_team_stats", lambda seasons: fixture_df)

    path = nflverse.fetch_team_stats(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_team_stats"


def test_fetch_pbp_online_calls_load_pbp(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"game_id": ["2025_01_KC_BAL"], "week": [1], "defteam": ["BAL"]})
    monkeypatch.setattr(nflverse.nfl, "load_pbp", lambda seasons: fixture_df)

    path = nflverse.fetch_pbp(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_pbp"


def test_fetch_schedules_online_calls_load_schedules(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"game_id": ["2025_01_KC_BAL"], "home_score": [27]})
    monkeypatch.setattr(nflverse.nfl, "load_schedules", lambda seasons: fixture_df)

    path = nflverse.fetch_schedules(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_schedules"
