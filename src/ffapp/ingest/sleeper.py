"""Sleeper API ingestion (SPEC.md §6.1, §6.3, §18; SPEC-ADDENDUM-02.md §A-C).

Network access lives only in `_get()`. Every public `fetch_*` function follows the
same shape: offline, read the cache or raise `OfflineCacheMiss`; online, hit the API
and archive the raw JSON payload plus a sidecar. No transformation logic here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ffapp.cache.offline import (
    age_hours,
    cache_miss,
    check_staleness,
    is_offline,
    read_sidecar,
    write_sidecar,
)
from ffapp.config import Settings
from ffapp.config import load_settings as _load_settings

BASE_URL = "https://api.sleeper.app/v1"
USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

logger = logging.getLogger("ffapp.ingest.sleeper")

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


def _get(path: str) -> Any:
    """Perform one GET against the Sleeper API. The only network call in this module."""
    response = _get_session().get(f"{BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "sleeper"


def _raw_count(data: Any) -> int | None:
    if isinstance(data, (list, dict)):
        return len(data)
    return None


def _fetch_or_read(
    *,
    filename: str,
    call: str,
    artifact: str,
    params: str,
    cache_key: str | None,
    offline: bool | None,
    settings: Settings,
) -> Path:
    path = _raw_dir(settings) / filename

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "sleeper",
                artifact,
                params,
                "ffapp cache warm --season <season> --all-leagues"
                f"  # or: ffapp ingest sleeper {call}",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, cache_key, settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "%s %s is stale (fetched_at_utc=%s); run cache warm to refresh.",
                    artifact,
                    params,
                    meta["fetched_at_utc"],
                )
        return path

    data = _get(call)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    write_sidecar(path, source="sleeper", call=call, cache_key=cache_key, rows=_raw_count(data))
    return path


def fetch_user(
    username: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"user_{username}.json",
        call=f"/user/{username}",
        artifact="user",
        params=f"username={username}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_leagues(
    user_id: str, season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"leagues_{user_id}_{season}.json",
        call=f"/user/{user_id}/leagues/nfl/{season}",
        artifact="leagues",
        params=f"user_id={user_id} season={season}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_league(
    league_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"league_{league_id}.json",
        call=f"/league/{league_id}",
        artifact="league",
        params=f"league_id={league_id}",
        cache_key="sleeper_league",
        offline=offline,
        settings=settings,
    )


def fetch_rosters(
    league_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"rosters_{league_id}.json",
        call=f"/league/{league_id}/rosters",
        artifact="rosters",
        params=f"league_id={league_id}",
        cache_key="sleeper_rosters",
        offline=offline,
        settings=settings,
    )


def fetch_users(
    league_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"users_{league_id}.json",
        call=f"/league/{league_id}/users",
        artifact="users",
        params=f"league_id={league_id}",
        cache_key="sleeper_rosters",
        offline=offline,
        settings=settings,
    )


def fetch_matchups(
    league_id: str, week: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"matchups_{league_id}_w{week}.json",
        call=f"/league/{league_id}/matchups/{week}",
        artifact="matchups",
        params=f"league_id={league_id} week={week}",
        cache_key=None,  # immutable once the week is final; no staleness policy applies
        offline=offline,
        settings=settings,
    )


def fetch_transactions(
    league_id: str, week: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"transactions_{league_id}_w{week}.json",
        call=f"/league/{league_id}/transactions/{week}",
        artifact="transactions",
        params=f"league_id={league_id} week={week}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_drafts(
    league_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"drafts_{league_id}.json",
        call=f"/league/{league_id}/drafts",
        artifact="drafts",
        params=f"league_id={league_id}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_draft_picks(
    draft_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"draft_picks_{draft_id}.json",
        call=f"/draft/{draft_id}/picks",
        artifact="draft_picks",
        params=f"draft_id={draft_id}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_traded_picks(
    league_id: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Not in SPEC.md's §18 endpoint table -- added for task 0.11 once a real
    draft slot turned out to trade picks. Each record is
    `{round, season, roster_id, owner_id, previous_owner_id}`: `roster_id`
    identifies the pick by its *original* owner, `owner_id` is who currently
    holds it. Confirmed live against the primary league: no record appears
    twice for the same (season, round, roster_id), so `owner_id` is always
    the final owner, not one hop in a longer chain.
    """
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"traded_picks_{league_id}.json",
        call=f"/league/{league_id}/traded_picks",
        artifact="traded_picks",
        params=f"league_id={league_id}",
        cache_key="sleeper_rosters",
        offline=offline,
        settings=settings,
    )


def fetch_trending(
    kind: Literal["add", "drop"] = "add",
    lookback_hours: int = 24,
    limit: int = 25,
    *,
    offline: bool | None = None,
    settings: Settings | None = None,
) -> Path:
    settings = _resolve_settings(settings)
    return _fetch_or_read(
        filename=f"trending_{kind}.json",
        call=f"/players/nfl/trending/{kind}?lookback_hours={lookback_hours}&limit={limit}",
        artifact="trending",
        params=f"kind={kind}",
        cache_key=None,
        offline=offline,
        settings=settings,
    )


def fetch_players(
    *, offline: bool | None = None, settings: Settings | None = None, force: bool = False
) -> Path:
    """Fetch the full player dictionary. Cached at most once per 24h (SPEC §6.1) —
    never call this in a loop, and a second invocation within 24h must not re-fetch it.
    """
    settings = _resolve_settings(settings)
    path = _raw_dir(settings) / "players_nfl.json"

    if not force and path.exists():
        meta = read_sidecar(path)
        if meta is not None and age_hours(meta["fetched_at_utc"]) < 24:
            return path

    return _fetch_or_read(
        filename="players_nfl.json",
        call="/players/nfl",
        artifact="players",
        params="",
        cache_key=None,
        offline=offline,
        settings=settings,
    )
