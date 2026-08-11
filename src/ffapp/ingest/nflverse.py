"""ffverse ingestion (SPEC.md §6.1, §6.3, §7).

Currently limited to the dynastyprocess/ffverse player-id crosswalk (the base table
for `ids/mapping.py`). Network access lives only in `_get_csv()`, matching the
`ingest/sleeper.py` shape: offline reads the cache or raises `OfflineCacheMiss`;
online hits the URL and archives the raw payload plus a sidecar.
"""

from __future__ import annotations

import logging
from pathlib import Path

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


__all__ = ["CROSSWALK_URL", "fetch_player_ids"]
