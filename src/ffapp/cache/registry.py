"""Cache orchestration: `ffapp cache warm/status/verify` (SPEC-ADDENDUM-02.md §B).

Registered here so cli.py stays a thin argument-parsing layer. Only the Sleeper
source is wired up for now — nflverse, rankings, odds, and weather join this
registry as their own ingest modules land in later tasks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ffapp.cache.offline import age_hours, check_staleness
from ffapp.config import LEAGUES_DIR, Settings, load_all_leagues, write_league_stub
from ffapp.ingest import sleeper


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "league"


def _unique_slug(base_slug: str, taken: set[str]) -> str:
    if base_slug not in taken:
        return base_slug
    n = 2
    while f"{base_slug}-{n}" in taken:
        n += 1
    return f"{base_slug}-{n}"


def _existing_slug_for_league_id(league_id: str, leagues_dir: Path) -> str | None:
    for existing in load_all_leagues(leagues_dir):
        if existing.league_id == league_id:
            return existing.slug
    return None


@dataclass(frozen=True)
class DiscoveredLeague:
    slug: str
    league_id: str
    path: Path


def discover_leagues(
    season: int, *, settings: Settings, leagues_dir: Path = LEAGUES_DIR
) -> list[DiscoveredLeague]:
    """Enumerate every Sleeper league on the account and write a config stub per league.

    Re-running is idempotent: a league_id that already has a stub keeps its slug (and
    its hand-set is_primary/overrides), even if the league's display name changed.
    """
    username = settings.sleeper_username
    if not username:
        raise ValueError("sleeper.username is not set in config/settings.yml")

    user = json.loads(sleeper.fetch_user(username, offline=False, settings=settings).read_text())
    user_id = user["user_id"]

    leagues = json.loads(
        sleeper.fetch_leagues(user_id, season, offline=False, settings=settings).read_text()
    )

    discovered: list[DiscoveredLeague] = []
    taken_slugs = {league.slug for league in load_all_leagues(leagues_dir)}
    for summary in leagues:
        league_id = summary["league_id"]
        league = json.loads(
            sleeper.fetch_league(league_id, offline=False, settings=settings).read_text()
        )

        slug = _existing_slug_for_league_id(league_id, leagues_dir)
        if slug is None:
            slug = _unique_slug(slugify(league.get("name", league_id)), taken_slugs)
            taken_slugs.add(slug)

        league_settings = league.get("settings", {})
        league_cache = {
            "total_rosters": league.get("total_rosters"),
            "roster_positions": league.get("roster_positions", []),
            "scoring_settings": league.get("scoring_settings", {}),
            "waiver_type": league_settings.get("waiver_type"),
            "waiver_budget": league_settings.get("waiver_budget"),
            "playoff_week_start": league_settings.get("playoff_week_start"),
        }
        path = write_league_stub(
            slug=slug,
            display_name=league.get("name", slug),
            league_id=league_id,
            season=season,
            league_cache=league_cache,
            leagues_dir=leagues_dir,
        )
        discovered.append(DiscoveredLeague(slug=slug, league_id=league_id, path=path))

    return discovered


def warm_sleeper(season: int, *, settings: Settings, leagues_dir: Path = LEAGUES_DIR) -> None:
    """Warm the Sleeper portion of the cache: league discovery, then per-league
    rosters/users/drafts, plus the account-level players dictionary and trending lists.
    """
    discovered = discover_leagues(season, settings=settings, leagues_dir=leagues_dir)

    for league in discovered:
        sleeper.fetch_rosters(league.league_id, offline=False, settings=settings)
        sleeper.fetch_users(league.league_id, offline=False, settings=settings)
        drafts_path = sleeper.fetch_drafts(league.league_id, offline=False, settings=settings)
        for draft in json.loads(drafts_path.read_text()):
            sleeper.fetch_draft_picks(draft["draft_id"], offline=False, settings=settings)

    sleeper.fetch_players(offline=False, settings=settings)
    sleeper.fetch_trending("add", offline=False, settings=settings)
    sleeper.fetch_trending("drop", offline=False, settings=settings)


def cache_status(settings: Settings) -> list[dict[str, Any]]:
    sleeper_dir = settings.cache.root / "sleeper"
    if not sleeper_dir.exists():
        return []

    rows = []
    for meta_path in sorted(sleeper_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        verdict = check_staleness(
            meta, meta.get("cache_key"), settings.cache.staleness_hours, raise_if_strict=False
        )
        rows.append(
            {
                "artifact": meta_path.name.removesuffix(".meta.json"),
                "source": meta.get("source"),
                "fetched_at_utc": meta.get("fetched_at_utc"),
                "age_hours": round(age_hours(meta["fetched_at_utc"]), 1),
                "verdict": verdict,
            }
        )
    return rows


@dataclass(frozen=True)
class CacheRequirement:
    description: str
    check: Callable[[Settings], bool]
    warm_hint: str


# Extend as later tasks' ingest modules land. Tasks 0.1/0.2 need nothing from cache —
# 0.2 *is* the live-network discovery step that populates it.
TASK_CACHE_REQUIREMENTS: dict[str, list[CacheRequirement]] = {}


def cache_verify(task_id: str, *, settings: Settings) -> list[tuple[CacheRequirement, bool]]:
    if task_id not in TASK_CACHE_REQUIREMENTS:
        raise ValueError(
            f"No cache requirements are registered for task {task_id} yet. "
            "Extend TASK_CACHE_REQUIREMENTS in ffapp/cache/registry.py as its ingest "
            "module lands."
        )
    return [(req, req.check(settings)) for req in TASK_CACHE_REQUIREMENTS[task_id]]
