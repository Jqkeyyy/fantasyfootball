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
from urllib.parse import urlencode

import polars as pl
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
    """Perform one GET against the Sleeper API. The only network call in this module.

    `path` is normally relative to `BASE_URL` (the documented `/v1` API).
    `fetch_sleeper_adp` passes a full URL instead -- its real data lives on
    a different, undocumented host (`api.sleeper.com`, no `/v1`) -- so an
    already-absolute `path` is used as-is rather than prefixed.
    """
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    response = _get_session().get(url, timeout=30)
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


# --- Sleeper's own ADP (draft.mock, mock-draft bot behaviour) --------------------
#
# Not in SPEC's §18 endpoint table and not the documented `/v1/players/nfl`
# endpoint (that payload's only relevance signal is `search_rank`, Sleeper's
# own internal fantasy-relevance ranking -- confirmed live to be a
# materially different number from real ADP, e.g. a real bench RB with
# real ADP ~465 had `search_rank` 9999999, the deep-bench sentinel).
#
# Sleeper doesn't document or expose ADP anywhere in api.sleeper.app/v1 --
# confirmed by reading docs.sleeper.com directly, nothing there. The real
# source, confirmed live via a real browser session (Chrome, logged in):
# a league's own "Players" tab, switched from a weekly view to "Season",
# calls `api.sleeper.com/projections/nfl/<season>` (a different,
# undocumented host from this module's own `BASE_URL` -- no `/v1`) with
# `position[]` repeated per position and `order_by=pts_ppr`. Each returned
# player's `stats.adp_ppr` is Sleeper's own real ADP for PPR scoring
# (`adp_std`/`adp_half_ppr`/`adp_2qb`/`adp_dynasty*` also ride along in the
# same payload but aren't used here) -- confirmed live: `curl` with no
# session cookie or auth header at all still returns 200 with real data
# (Jahmyr Gibbs adp_ppr=1.1, Bijan Robinson adp_ppr=2.1 on 2026-08-16,
# matching that day's real top-of-market consensus), so this is genuinely
# public, not an artefact of being logged in. One call returns every
# fantasy-relevant position (3300 rows checked live, DEF included) -- no
# pagination needed.
SLEEPER_ADP_URL_TEMPLATE = "https://api.sleeper.com/projections/nfl/{season}"
SLEEPER_ADP_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
# Sleeper's own vocabulary already matches this project's canonical codes
# except defense -- same DEF->DST alias `draft.board`/`league_format.py`
# already apply to Sleeper's other payloads.
SLEEPER_ADP_POSITION_MAP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST"}


def fetch_sleeper_adp(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch Sleeper's own real per-player ADP for `season` to
    data/raw/sleeper/adp_<season>.json -- see the module comment above for
    what this is and how it was confirmed live.
    """
    settings = _resolve_settings(settings)
    query = urlencode(
        [("season_type", "regular")]
        + [("position[]", pos) for pos in SLEEPER_ADP_POSITIONS]
        + [("order_by", "pts_ppr")]
    )
    call = f"{SLEEPER_ADP_URL_TEMPLATE.format(season=season)}?{query}"
    return _fetch_or_read(
        filename=f"adp_{season}.json",
        call=call,
        artifact="adp",
        params=f"season={season}",
        cache_key="sleeper_adp",
        offline=offline,
        settings=settings,
    )


def normalize_sleeper_adp(payload: list[dict[str, Any]], *, season: int) -> pl.DataFrame:
    """Extract each player's real Sleeper ADP into the canonical per-player
    ADP schema (same shape as `ingest.rankings.normalize_adp`, minus the
    spread/times_drafted/bye_week columns Sleeper's own payload doesn't
    carry). A row missing `player`, `position`, or `stats.adp_ppr` entirely
    (seen live for a small number of non-fantasy entries even with the
    `position[]` filter applied) is skipped rather than guessed at.
    """
    rows: list[dict[str, Any]] = []
    for row in payload:
        player = row.get("player") or {}
        position = SLEEPER_ADP_POSITION_MAP.get(player.get("position") or "")
        adp = (row.get("stats") or {}).get("adp_ppr")
        if not position or adp is None:
            continue
        first_name = player.get("first_name") or ""
        last_name = player.get("last_name") or ""
        name = f"{first_name} {last_name}".strip()
        if not name:
            continue
        rows.append(
            {
                "source": "sleeper",
                "season": season,
                "player_name": name,
                "position": position,
                "team": row.get("team") or player.get("team"),
                "adp": float(adp),
                "adp_sd": None,
                "adp_high": None,
                "adp_low": None,
                "times_drafted": None,
                "bye_week": None,
            }
        )
    schema = {
        "source": pl.Utf8,
        "season": pl.Int64,
        "player_name": pl.Utf8,
        "position": pl.Utf8,
        "team": pl.Utf8,
        "adp": pl.Float64,
        "adp_sd": pl.Float64,
        "adp_high": pl.Float64,
        "adp_low": pl.Float64,
        "times_drafted": pl.Int64,
        "bye_week": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)
