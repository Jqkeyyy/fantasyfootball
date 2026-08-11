from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ffapp.cache.offline import (
    OfflineCacheMiss,
    StaleCacheError,
    age_hours,
    cache_miss,
    check_staleness,
    is_offline,
    read_sidecar,
    write_sidecar,
)


def test_is_offline_defaults_true_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FFAPP_OFFLINE", raising=False)

    assert is_offline() is True


def test_is_offline_false_when_env_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFAPP_OFFLINE", "0")

    assert is_offline() is False


def test_is_offline_explicit_override_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFAPP_OFFLINE", "1")

    assert is_offline(override=False) is False


def test_write_sidecar_then_read_sidecar_round_trips(tmp_path: Path) -> None:
    raw_path = tmp_path / "league_111.json"
    raw_path.write_text("{}")

    write_sidecar(
        raw_path, source="sleeper", call="/league/111", cache_key="sleeper_league", rows=1
    )
    meta = read_sidecar(raw_path)

    assert meta is not None
    assert meta["source"] == "sleeper"
    assert meta["call"] == "/league/111"
    assert meta["cache_key"] == "sleeper_league"
    assert meta["rows"] == 1
    assert "fetched_at_utc" in meta


def test_read_sidecar_returns_none_when_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "missing.json"

    assert read_sidecar(raw_path) is None


def test_age_hours_computes_elapsed_time() -> None:
    fetched_at = (datetime.now(UTC) - timedelta(hours=5)).isoformat()

    assert age_hours(fetched_at) == pytest.approx(5.0, abs=0.01)


def test_check_staleness_returns_no_policy_when_cache_key_unset() -> None:
    meta = {"fetched_at_utc": (datetime.now(UTC) - timedelta(hours=999)).isoformat()}

    assert check_staleness(meta, None, {"sleeper_league": 168}) == "no_policy"


def test_check_staleness_returns_fresh_within_threshold() -> None:
    meta = {"fetched_at_utc": (datetime.now(UTC) - timedelta(hours=1)).isoformat()}

    assert check_staleness(meta, "sleeper_league", {"sleeper_league": 168}) == "fresh"


def test_check_staleness_returns_stale_past_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FFAPP_CACHE_STRICT", raising=False)
    meta = {"fetched_at_utc": (datetime.now(UTC) - timedelta(hours=200)).isoformat()}

    assert check_staleness(meta, "sleeper_league", {"sleeper_league": 168}) == "stale"


def test_check_staleness_raises_when_strict_and_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    meta = {"fetched_at_utc": (datetime.now(UTC) - timedelta(hours=200)).isoformat()}

    with pytest.raises(StaleCacheError):
        check_staleness(meta, "sleeper_league", {"sleeper_league": 168})


def test_cache_miss_message_names_source_artifact_and_fetch_command() -> None:
    err = cache_miss(
        source="sleeper",
        artifact="matchups",
        params="season=2025 week=14",
        fetch_hint="ffapp cache warm --season 2025 --weeks 14 --league main-ppr",
    )

    assert isinstance(err, OfflineCacheMiss)
    message = str(err)
    assert "sleeper" in message
    assert "matchups" in message
    assert "season=2025 week=14" in message
    assert "ffapp cache warm --season 2025 --weeks 14 --league main-ppr" in message
