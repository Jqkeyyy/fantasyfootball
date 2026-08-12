import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ffapp.cache.offline import OfflineCacheMiss, StaleCacheError, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import weather

FORECAST_FIXTURE = {
    "hourly": {
        "time": ["2026-09-13T17:00", "2026-09-13T18:00", "2026-09-13T19:00"],
        "wind_speed_10m": [5.0, 12.5, 8.0],
        "temperature_2m": [70.0, 68.0, 65.0],
        "precipitation_probability": [10, 40, 20],
    }
}

HISTORICAL_FIXTURE_DRY = {
    "hourly": {
        "time": ["2025-09-07T17:00", "2025-09-07T18:00"],
        "wind_speed_10m": [3.0, 6.5],
        "temperature_2m": [80.0, 78.0],
        "precipitation": [0.0, 0.0],
    }
}

HISTORICAL_FIXTURE_WET = {
    "hourly": {
        "time": ["2025-09-07T17:00", "2025-09-07T18:00"],
        "wind_speed_10m": [3.0, 6.5],
        "temperature_2m": [80.0, 78.0],
        "precipitation": [0.0, 1.2],
    }
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(self._payload)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"weather_forecast": 6},
            warn_on_stale=True,
        ),
    )


def _age_stamp(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


# --- normalize_weather -----------------------------------------------------------


def test_normalize_weather_picks_the_matching_hour_for_forecast() -> None:
    result = weather.normalize_weather(
        FORECAST_FIXTURE, kickoff_hour_utc="2026-09-13T18:00", source="forecast"
    )

    assert result == {"wind_mph": 12.5, "precip_prob": 40, "temp_f": 68.0}


def test_normalize_weather_historical_precip_prob_is_zero_when_dry() -> None:
    result = weather.normalize_weather(
        HISTORICAL_FIXTURE_DRY, kickoff_hour_utc="2025-09-07T18:00", source="historical"
    )

    assert result["precip_prob"] == 0.0
    assert result["wind_mph"] == 6.5


def test_normalize_weather_historical_precip_prob_is_100_when_any_rain_fell() -> None:
    result = weather.normalize_weather(
        HISTORICAL_FIXTURE_WET, kickoff_hour_utc="2025-09-07T18:00", source="historical"
    )

    assert result["precip_prob"] == 100.0


# --- fetch_weather -----------------------------------------------------------------


def test_fetch_weather_online_writes_raw_json_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fake_session = _FakeSession(FORECAST_FIXTURE)
    monkeypatch.setattr(weather, "_get_session", lambda: fake_session)

    path = weather.fetch_weather(
        "2026_02_KC_BAL",
        39.05,
        -94.48,
        "2026-09-13",
        source="forecast",
        offline=False,
        settings=settings,
    )

    assert path.exists()
    assert json.loads(path.read_text()) == FORECAST_FIXTURE
    assert fake_session.calls[0]["url"] == weather.FORECAST_URL
    assert fake_session.calls[0]["params"]["latitude"] == 39.05
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "open-meteo"
    assert meta["cache_key"] == "weather_forecast"


def test_fetch_weather_uses_archive_url_for_historical_source(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fake_session = _FakeSession(HISTORICAL_FIXTURE_DRY)
    monkeypatch.setattr(weather, "_get_session", lambda: fake_session)

    weather.fetch_weather(
        "2025_01_KC_BAL",
        39.05,
        -94.48,
        "2025-09-07",
        source="historical",
        offline=False,
        settings=settings,
    )

    assert fake_session.calls[0]["url"] == weather.ARCHIVE_URL
    assert "precipitation_probability" not in fake_session.calls[0]["params"]["hourly"]


def test_fetch_weather_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom() -> None:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(weather, "_get_session", _boom)
    path = settings.cache.root / "weather" / "forecast_2026_02_KC_BAL.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FORECAST_FIXTURE))
    write_sidecar(path, source="open-meteo", call="forecast", cache_key="weather_forecast")

    result = weather.fetch_weather(
        "2026_02_KC_BAL",
        39.05,
        -94.48,
        "2026-09-13",
        source="forecast",
        offline=True,
        settings=settings,
    )

    assert result == path


def test_fetch_weather_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        weather.fetch_weather(
            "2026_02_KC_BAL",
            39.05,
            -94.48,
            "2026-09-13",
            source="forecast",
            offline=True,
            settings=settings,
        )

    message = str(exc_info.value)
    assert "weather" in message
    assert "2026_02_KC_BAL" in message


def test_fetch_weather_offline_with_stale_cache_and_strict_env_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    path = settings.cache.root / "weather" / "forecast_2026_02_KC_BAL.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FORECAST_FIXTURE))
    sidecar_path(path).write_text(
        json.dumps(
            {
                "source": "open-meteo",
                "fetched_at_utc": _age_stamp(200),
                "cache_key": "weather_forecast",
            }
        )
    )

    with pytest.raises(StaleCacheError):
        weather.fetch_weather(
            "2026_02_KC_BAL",
            39.05,
            -94.48,
            "2026-09-13",
            source="forecast",
            offline=True,
            settings=settings,
        )


# --- fetch_weather_for_schedule ------------------------------------------------------


def _schedule_row(**kwargs: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": "2025_01_KC_BAL",
        "season": 2025,
        "week": 1,
        "roof": "outdoors",
        "stadium_id": "KAN00",
        "kickoff_utc": "2025-09-07T18:00:00Z",
    }
    row.update(kwargs)
    return row


def _schedule(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _stadiums() -> pl.DataFrame:
    return pl.DataFrame(
        {"stadium_id": ["KAN00", "DET00"], "lat": [39.0489, 42.3400], "lon": [-94.4839, -83.0456]}
    )


def test_fetch_weather_for_schedule_applies_dome_override_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dome games must not call fetch_weather")

    monkeypatch.setattr(weather, "fetch_weather", _boom)
    schedule = _schedule([_schedule_row(roof="dome", stadium_id="DET00")])

    result = weather.fetch_weather_for_schedule(
        schedule,
        _stadiums(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        offline=False,
        settings=settings,
    )

    row = result.row(0, named=True)
    assert row["wind_mph"] == 0.0
    assert row["precip_prob"] == 0.0
    assert row["temp_f"] == 70.0
    assert row["is_dome"] is True
    assert row["source"] == "dome_override"


def test_fetch_weather_for_schedule_uses_historical_for_a_past_game(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls: list[str] = []

    def _fake_fetch(
        game_id: str, lat: float, lon: float, date: str, *, source: str, **kwargs: Any
    ) -> Path:
        calls.append(source)
        path = tmp_raw_path(settings, f"{source}_{game_id}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(HISTORICAL_FIXTURE_DRY))
        return path

    def tmp_raw_path(settings: Settings, name: str) -> Path:
        return settings.cache.root / "weather" / name

    monkeypatch.setattr(weather, "fetch_weather", _fake_fetch)
    schedule = _schedule([_schedule_row(kickoff_utc="2025-09-07T18:00:00Z")])

    result = weather.fetch_weather_for_schedule(
        schedule,
        _stadiums(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        offline=False,
        settings=settings,
    )

    assert calls == ["historical"]
    row = result.row(0, named=True)
    assert row["source"] == "historical"
    assert row["is_dome"] is False
    assert row["wind_mph"] == 6.5


def test_fetch_weather_for_schedule_uses_forecast_for_a_future_game(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls: list[str] = []

    def _fake_fetch(
        game_id: str, lat: float, lon: float, date: str, *, source: str, **kwargs: Any
    ) -> Path:
        calls.append(source)
        path = settings.cache.root / "weather" / f"{source}_{game_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(FORECAST_FIXTURE))
        return path

    monkeypatch.setattr(weather, "fetch_weather", _fake_fetch)
    schedule = _schedule([_schedule_row(kickoff_utc="2026-09-13T18:00:00Z")])

    result = weather.fetch_weather_for_schedule(
        schedule,
        _stadiums(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        offline=False,
        settings=settings,
    )

    assert calls == ["forecast"]
    assert result.row(0, named=True)["source"] == "forecast"


def test_fetch_weather_for_schedule_skips_games_with_no_kickoff_utc(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not fetch a game with no known kickoff time")

    monkeypatch.setattr(weather, "fetch_weather", _boom)
    schedule = _schedule([_schedule_row(kickoff_utc=None)])

    result = weather.fetch_weather_for_schedule(
        schedule,
        _stadiums(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        offline=False,
        settings=settings,
    )

    assert result.height == 0
