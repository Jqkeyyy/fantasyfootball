"""Offline-first primitives (SPEC-ADDENDUM-02.md §A, §C).

Network access exists at exactly one layer: `ingest/*.fetch()`. In offline mode,
fetch() reads only from data/raw/. On a miss it raises OfflineCacheMiss with a
message naming the source, the exact artefact, and the command that would fetch it.
Never return empty data on a miss — that produces a model trained on partial data,
a draft board missing players, and a golden test that passes by comparing zero rows.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OfflineCacheMiss(Exception):
    """Required data is not cached and the network is disabled."""


class StaleCacheError(Exception):
    """Cached data exceeds the staleness policy and FFAPP_CACHE_STRICT is set."""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in ("0", "false", "False")


def is_offline(override: bool | None = None) -> bool:
    if override is not None:
        return override
    return _env_flag("FFAPP_OFFLINE", default=True)


def is_cache_strict(override: bool | None = None) -> bool:
    if override is not None:
        return override
    return _env_flag("FFAPP_CACHE_STRICT", default=False)


def sidecar_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.name + ".meta.json")


def write_sidecar(
    raw_path: Path,
    *,
    source: str,
    call: str,
    cache_key: str | None = None,
    rows: int | None = None,
) -> None:
    meta = {
        "source": source,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "rows": rows,
        "call": call,
        "cache_key": cache_key,
    }
    sidecar_path(raw_path).write_text(json.dumps(meta, indent=2))


def read_sidecar(raw_path: Path) -> dict[str, Any] | None:
    path = sidecar_path(raw_path)
    if not path.exists():
        return None
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def age_hours(fetched_at_utc: str) -> float:
    fetched = datetime.fromisoformat(fetched_at_utc)
    return (datetime.now(UTC) - fetched).total_seconds() / 3600


def check_staleness(
    meta: dict[str, Any],
    cache_key: str | None,
    staleness_hours: dict[str, float],
    *,
    raise_if_strict: bool = True,
) -> str:
    """Classify a cached artefact as "fresh", "stale", or "no_policy".

    Raises StaleCacheError instead of returning "stale" when FFAPP_CACHE_STRICT is set
    (SPEC-ADDENDUM-02.md §G) — the deliberate escape hatch for treating stale data as a
    hard failure rather than a warning. Reporting commands (`cache status`, `cache verify`)
    pass raise_if_strict=False since their entire job is to surface staleness, not fail on it.
    """
    if cache_key is None or cache_key not in staleness_hours:
        return "no_policy"

    hours = age_hours(meta["fetched_at_utc"])
    threshold = staleness_hours[cache_key]
    if hours <= threshold:
        return "fresh"

    if raise_if_strict and is_cache_strict():
        raise StaleCacheError(
            f"{cache_key} is {hours:.1f}h old (staleness policy: {threshold}h) "
            "and FFAPP_CACHE_STRICT=1."
        )
    return "stale"


def cache_miss(source: str, artifact: str, params: str, fetch_hint: str) -> OfflineCacheMiss:
    return OfflineCacheMiss(
        f"OfflineCacheMiss: {source}/{artifact} {params} not cached.\n"
        "  Run on an unrestricted network:\n"
        f"    {fetch_hint}"
    )
