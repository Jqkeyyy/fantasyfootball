"""ffverse ingestion (SPEC.md §6.1, §6.3, §7; ADDENDUM-01 §C.2 kicker stats).

Two network paths:
    - `_get_csv()`: a raw HTTP GET for the dynastyprocess/ffverse player-id
      crosswalk (the base table for `ids/mapping.py`).
    - `nflreadpy.load_*()`: the `player_stats`/`team_stats`/`schedules` fetchers
      call nflreadpy directly rather than hitting an HTTP endpoint by hand.
      nflreadpy hands back an already-parsed polars DataFrame (no raw bytes to
      archive verbatim), so the DataFrame itself, written to parquet, is this
      project's raw archive -- the moral equivalent of `sleeper.py` re-serialising
      a decoded JSON payload back to disk as its "raw" copy.

Both paths follow the same offline-cache shape: offline reads the cache or raises
`OfflineCacheMiss`; online fetches and archives the raw payload plus a sidecar.

Scope note (see HANDOFF.md §4): only enough of nflreadpy is wired up here to
unblock task 0.5's golden test -- one season of weekly player/team stats and
schedules. Full multi-season ingestion (play-by-play, snap counts, depth charts,
rosters, injuries) is task 1.1, Phase 1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
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

CROSSWALK_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
)
USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

logger = logging.getLogger("ffapp.ingest.nflverse")

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


def _get_csv() -> str:
    """Fetch the crosswalk CSV as text. The only network call in this module."""
    response = _get_session().get(CROSSWALK_URL, timeout=30)
    response.raise_for_status()
    return response.text


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "nflverse"


def fetch_player_ids(*, offline: bool | None = None, settings: Settings | None = None) -> Path:
    """Fetch the ffverse player-id crosswalk (SPEC §7 step 1) to
    data/raw/nflverse/player_ids.csv.
    """
    settings = _resolve_settings(settings)
    path = _raw_dir(settings) / "player_ids.csv"

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "nflverse",
                "player_ids",
                "",
                "ffapp ingest nflverse --player-ids --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, "nflverse_player_ids", settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "player_ids is stale (fetched_at_utc=%s); run ingest nflverse to refresh.",
                    meta["fetched_at_utc"],
                )
        return path

    text = _get_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    rows = max(len(text.splitlines()) - 1, 0)
    write_sidecar(
        path, source="nflverse", call=CROSSWALK_URL, cache_key="nflverse_player_ids", rows=rows
    )
    return path


def _fetch_nflreadpy_parquet(
    *,
    filename: str,
    call_desc: str,
    cache_key: str,
    load: Callable[[], pl.DataFrame],
    artifact: str,
    params: str,
    offline: bool | None,
    settings: Settings,
) -> Path:
    path = _raw_dir(settings) / filename

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "nflverse",
                artifact,
                params,
                f"ffapp ingest nflverse --{artifact} {params.replace(' ', ' --')} --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, cache_key, settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "%s %s is stale (fetched_at_utc=%s); run to refresh.",
                    artifact,
                    params,
                    meta["fetched_at_utc"],
                )
        return path

    df = load()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    write_sidecar(path, source="nflverse", call=call_desc, cache_key=cache_key, rows=df.height)
    return path


def fetch_player_stats(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's per-player-week stat table for one season to
    data/raw/nflverse/player_stats_<season>.parquet.
    """
    settings = _resolve_settings(settings)
    return _fetch_nflreadpy_parquet(
        filename=f"player_stats_{season}.parquet",
        call_desc=f"load_player_stats(seasons=[{season}])",
        cache_key="nflverse_player_stats",
        load=lambda: nfl.load_player_stats(seasons=[season]),
        artifact="player-stats",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


def fetch_team_stats(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's per-team-week stat table for one season (DST scoring
    inputs) to data/raw/nflverse/team_stats_<season>.parquet.
    """
    settings = _resolve_settings(settings)
    return _fetch_nflreadpy_parquet(
        filename=f"team_stats_{season}.parquet",
        call_desc=f"load_team_stats(seasons=[{season}])",
        cache_key="nflverse_team_stats",
        load=lambda: nfl.load_team_stats(seasons=[season]),
        artifact="team-stats",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


def fetch_pbp(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's play-by-play for one season (needed to derive genuine
    defensive/return touchdowns -- see scoring/stats.py's `_defensive_return_tds`)
    to data/raw/nflverse/pbp_<season>.parquet.
    """
    settings = _resolve_settings(settings)
    return _fetch_nflreadpy_parquet(
        filename=f"pbp_{season}.parquet",
        call_desc=f"load_pbp(seasons=[{season}])",
        cache_key="nflverse_pbp",
        load=lambda: nfl.load_pbp(seasons=[season]),
        artifact="pbp",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


def fetch_schedules(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's schedule (game scores, for points_allowed) for one season
    to data/raw/nflverse/schedules_<season>.parquet.
    """
    settings = _resolve_settings(settings)
    return _fetch_nflreadpy_parquet(
        filename=f"schedules_{season}.parquet",
        call_desc=f"load_schedules(seasons=[{season}])",
        cache_key="nflverse_schedules",
        load=lambda: nfl.load_schedules(seasons=[season]),
        artifact="schedules",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


__all__ = [
    "CROSSWALK_URL",
    "fetch_pbp",
    "fetch_player_ids",
    "fetch_player_stats",
    "fetch_schedules",
    "fetch_team_stats",
]
