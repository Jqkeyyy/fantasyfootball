"""Config loading and path resolution.

Layout (SPEC.md §5, superseded by SPEC-ADDENDUM-01.md §A.2 for multi-league):

    config/
    ├── leagues/
    │   ├── main-ppr.yml       # one file per Sleeper league, `is_primary: true` on exactly one
    │   └── league-b.yml
    └── settings.yml           # league-agnostic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yml"
LEAGUES_DIR = CONFIG_DIR / "leagues"


class NoPrimaryLeagueError(Exception):
    """No league in leagues_dir has `is_primary: true`."""


class MultiplePrimaryLeaguesError(Exception):
    """More than one league in leagues_dir has `is_primary: true`."""


class LeagueNotFoundError(Exception):
    """No league file exists for the given slug."""


@dataclass(frozen=True)
class CacheSettings:
    root: Path
    offline_default: bool
    staleness_hours: dict[str, float]
    warn_on_stale: bool


@dataclass(frozen=True)
class Settings:
    data_root: Path
    sleeper_username: str | None
    cache: CacheSettings


@dataclass(frozen=True)
class LeagueConfig:
    slug: str
    display_name: str
    is_primary: bool
    league_id: str | None
    season: int
    league_cache: dict[str, Any]
    overrides: dict[str, Any]


def load_settings(path: Path = SETTINGS_PATH, *, root: Path | None = None) -> Settings:
    """`root` is the base relative paths (paths.data_root, cache.root) resolve against.

    Deliberately independent of `path`'s own location, not `path.parent` — settings.yml
    lives inside config/, one level below the repo root where data/ actually is, so
    resolving "./data" against the file's own directory would land it at config/data/.
    """
    base = root if root is not None else REPO_ROOT
    raw = yaml.safe_load(path.read_text()) or {}

    paths = raw.get("paths", {})
    data_root = (base / paths.get("data_root", "./data")).resolve()

    sleeper = raw.get("sleeper", {})
    sleeper_username = sleeper.get("username")

    cache_raw = raw.get("cache", {})
    cache = CacheSettings(
        root=(base / cache_raw.get("root", "./data/raw")).resolve(),
        offline_default=bool(cache_raw.get("offline_default", True)),
        staleness_hours=dict(cache_raw.get("staleness_hours", {})),
        warn_on_stale=bool(cache_raw.get("warn_on_stale", True)),
    )

    return Settings(data_root=data_root, sleeper_username=sleeper_username, cache=cache)


def _league_from_dict(raw: dict[str, Any]) -> LeagueConfig:
    sleeper = raw.get("sleeper", {})
    return LeagueConfig(
        slug=raw["slug"],
        display_name=raw.get("display_name", raw["slug"]),
        is_primary=bool(raw.get("is_primary", False)),
        league_id=sleeper.get("league_id"),
        season=sleeper.get("season"),
        league_cache=dict(raw.get("league_cache", {})),
        overrides=dict(raw.get("overrides", {})),
    )


def load_league(slug: str, leagues_dir: Path = LEAGUES_DIR) -> LeagueConfig:
    path = leagues_dir / f"{slug}.yml"
    if not path.exists():
        raise LeagueNotFoundError(f"No league config at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return _league_from_dict(raw)


def load_all_leagues(leagues_dir: Path = LEAGUES_DIR) -> list[LeagueConfig]:
    if not leagues_dir.exists():
        return []
    return [
        _league_from_dict(yaml.safe_load(path.read_text()) or {})
        for path in sorted(leagues_dir.glob("*.yml"))
    ]


def load_primary_league(leagues_dir: Path = LEAGUES_DIR) -> LeagueConfig:
    primaries = [league for league in load_all_leagues(leagues_dir) if league.is_primary]
    if not primaries:
        raise NoPrimaryLeagueError(
            f"No league in {leagues_dir} has is_primary: true. "
            "Set it by hand on the league you want commands to default to."
        )
    if len(primaries) > 1:
        slugs = ", ".join(league.slug for league in primaries)
        raise MultiplePrimaryLeaguesError(
            f"More than one league is marked is_primary: true ({slugs}). "
            "Exactly one league must be primary."
        )
    return primaries[0]


def write_league_stub(
    slug: str,
    display_name: str,
    league_id: str,
    season: int,
    league_cache: dict[str, Any],
    leagues_dir: Path = LEAGUES_DIR,
) -> Path:
    """Write (or refresh) a league config stub from freshly fetched Sleeper data.

    Refreshes generated fields (league_cache, display_name, league_id, season) on every
    call so re-running discovery picks up settings changes. Preserves `is_primary` and
    `overrides`, which are hand-edited after the stub is first written.
    """
    leagues_dir.mkdir(parents=True, exist_ok=True)
    path = leagues_dir / f"{slug}.yml"

    is_primary = False
    overrides: dict[str, Any] = {
        "flex_eligible": ["RB", "WR", "TE"],
        "superflex_eligible": ["QB", "RB", "WR", "TE"],
    }
    if path.exists():
        existing = _league_from_dict(yaml.safe_load(path.read_text()) or {})
        is_primary = existing.is_primary
        overrides = existing.overrides

    document = {
        "slug": slug,
        "display_name": display_name,
        "is_primary": is_primary,
        "sleeper": {"league_id": league_id, "season": season},
        "league_cache": league_cache,
        "overrides": overrides,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path
