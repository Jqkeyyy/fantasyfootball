"""Open-Meteo weather ingestion (SPEC.md §6.1, §10.3; task 1.3).

Only wind has a large, reliable effect on fantasy outcomes (SPEC §10.3);
this module deliberately does the minimum to get `wind_mph`/`precip_prob`/
`temp_f` right per game, not a general-purpose weather client.

Two network paths, matching SPEC §10.3's own "which one to use" rule:
    - the forecast endpoint (`api.open-meteo.com`), for a future kickoff --
      used for live projections.
    - the historical archive endpoint (`archive-api.open-meteo.com`), for a
      past kickoff -- used for backtesting, since it returns actual recorded
      conditions rather than a forecast.

`fetch_weather_for_schedule` is the orchestrating entry point: for each game
it decides forecast vs. historical from the game's own date versus `now`,
skips the network call entirely for `dome`/`closed` games (SPEC's override --
wind/precip/temp are fixed constants, not fetched), and returns the fully
normalized table directly rather than a raw path plus a separate normalise()
step. This is a deliberate, documented departure from this project's usual
ingest/-stays-pure fetch()/normalise() split (§6.3): the forecast-vs-
historical-vs-dome choice is inseparable from the fetch itself (you cannot
decide which endpoint to call, or whether to call one at all, without
evaluating the game first), so splitting it into a separate "business logic"
step in interim/build.py would just relocate the same per-game loop across a
module boundary without changing what it does. CLAUDE.md's "no network calls
outside ingest/" rule is the one actually being protected here, and it is:
the network call itself, and the one-line date comparison that selects it,
both stay inside this module.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ffapp.cache.offline import (
    cache_miss,
    check_staleness,
    is_offline,
    read_sidecar,
    write_sidecar,
)
from ffapp.config import Settings
from ffapp.config import load_settings as _load_settings

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

# SPEC §6.2: roof values that trigger the dome override.
DOME_ROOFS = {"dome", "closed"}

DOME_WIND_MPH = 0.0
DOME_PRECIP_PROB = 0.0
DOME_TEMP_F = 70.0

Source = Literal["forecast", "historical"]

logger = logging.getLogger("ffapp.ingest.weather")

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session = session
    return _session


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "weather"


def fetch_weather(
    game_id: str,
    latitude: float,
    longitude: float,
    date: str,
    *,
    source: Source,
    offline: bool | None = None,
    settings: Settings | None = None,
) -> Path:
    """One Open-Meteo call for one game's full kickoff day at the venue's
    coordinates (SPEC §6.1's "batch stadium requests rather than one per
    player" -- one call per game already satisfies that; there is no
    coarser granularity that still identifies the right kickoff hour).
    Archives the raw hourly response to
    data/raw/weather/<source>_<game_id>.json; the specific kickoff hour is
    picked out later by `normalize_weather`.
    """
    settings = _resolve_settings(settings)
    path = _raw_dir(settings) / f"{source}_{game_id}.json"
    cache_key = f"weather_{source}"

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "weather",
                source,
                f"game_id={game_id}",
                f"ffapp ingest weather --game-id {game_id} --source {source} --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, cache_key, settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "weather %s %s is stale (fetched_at_utc=%s); run to refresh.",
                    source,
                    game_id,
                    meta["fetched_at_utc"],
                )
        return path

    url = FORECAST_URL if source == "forecast" else ARCHIVE_URL
    hourly_vars = (
        "temperature_2m,precipitation_probability,wind_speed_10m"
        if source == "forecast"
        else "temperature_2m,precipitation,wind_speed_10m"
    )
    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": hourly_vars,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    response = _get_session().get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    write_sidecar(
        path,
        source="open-meteo",
        call=f"{source} lat={latitude} lon={longitude} date={date}",
        cache_key=cache_key,
        rows=len(payload.get("hourly", {}).get("time", [])),
    )
    return path


def normalize_weather(
    raw: dict[str, Any], *, kickoff_hour_utc: str, source: Source
) -> dict[str, float]:
    """Pick the hourly row matching `kickoff_hour_utc` ("YYYY-MM-DDTHH:00",
    Open-Meteo's own hourly timestamp format -- already UTC since
    `fetch_weather` requests `timezone=UTC`) out of one game's raw response.

    The forecast endpoint returns a real `precipitation_probability` (SPEC's
    own field). The historical archive endpoint has no probability for
    actuals -- only measured `precipitation` (mm) -- so for
    `source="historical"`, `precip_prob` is a documented simplification:
    100.0 if any precipitation fell that hour, else 0.0. SPEC §10.3 itself
    says not to over-engineer this; wind is the feature that matters.
    """
    hourly = raw["hourly"]
    idx = hourly["time"].index(kickoff_hour_utc)
    wind_mph = hourly["wind_speed_10m"][idx]
    temp_f = hourly["temperature_2m"][idx]
    if source == "forecast":
        precip_prob = hourly["precipitation_probability"][idx]
    else:
        precip_prob = 100.0 if (hourly["precipitation"][idx] or 0) > 0 else 0.0
    return {"wind_mph": wind_mph, "precip_prob": precip_prob, "temp_f": temp_f}


def _kickoff_hour_utc(kickoff_utc: str) -> str:
    """ "2025-09-05T20:20:00Z" -> "2025-09-05T20:00" (Open-Meteo's hourly
    timestamp format, floored to the hour)."""
    dt = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%dT%H:00")


def fetch_weather_for_schedule(
    schedule: pl.DataFrame,
    stadiums: pl.DataFrame,
    *,
    now: datetime,
    offline: bool | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`interim/weather.parquet`: one row per game_id with `wind_mph`,
    `precip_prob`, `temp_f`, `is_dome`, `source`
    ("dome_override" / "forecast" / "historical"). Requires
    `schedule.kickoff_utc` already populated
    (`interim.build.add_kickoff_utc`) -- a game with no kickoff_utc is
    skipped (a genuinely unknowable game time, not a value to guess at --
    CLAUDE.md rule 2).

    Games with `roof` in {"dome", "closed"} never hit the network -- SPEC
    §10.3's override applies directly from `schedule.roof`, the real
    per-game value, not `stadiums.dome` (a static per-venue reference field
    that can't capture a retractable roof's actual state on a given day).
    """
    settings = _resolve_settings(settings)
    with_coords = schedule.join(
        stadiums.select("stadium_id", "lat", "lon"), on="stadium_id", how="left"
    )

    rows: list[dict[str, Any]] = []
    for game in with_coords.iter_rows(named=True):
        if game["kickoff_utc"] is None:
            continue

        if game["roof"] in DOME_ROOFS:
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "week": game["week"],
                    "wind_mph": DOME_WIND_MPH,
                    "precip_prob": DOME_PRECIP_PROB,
                    "temp_f": DOME_TEMP_F,
                    "is_dome": True,
                    "source": "dome_override",
                }
            )
            continue

        kickoff_dt = datetime.fromisoformat(game["kickoff_utc"].replace("Z", "+00:00"))
        source: Source = "historical" if kickoff_dt.date() < now.date() else "forecast"
        raw_path = fetch_weather(
            game["game_id"],
            game["lat"],
            game["lon"],
            kickoff_dt.date().isoformat(),
            source=source,
            offline=offline,
            settings=settings,
        )
        raw = json.loads(raw_path.read_text())
        weather = normalize_weather(
            raw, kickoff_hour_utc=_kickoff_hour_utc(game["kickoff_utc"]), source=source
        )
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
                "wind_mph": weather["wind_mph"],
                "precip_prob": weather["precip_prob"],
                "temp_f": weather["temp_f"],
                "is_dome": False,
                "source": source,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "wind_mph": pl.Float64,
            "precip_prob": pl.Float64,
            "temp_f": pl.Float64,
            "is_dome": pl.Boolean,
            "source": pl.Utf8,
        },
    )


__all__ = [
    "ARCHIVE_URL",
    "DOME_PRECIP_PROB",
    "DOME_ROOFS",
    "DOME_TEMP_F",
    "DOME_WIND_MPH",
    "FORECAST_URL",
    "fetch_weather",
    "fetch_weather_for_schedule",
    "normalize_weather",
]
