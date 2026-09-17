"""Artifact inventory and last-known-good promotion for weekly refreshes."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from ffapp.tools.artifacts import atomic_copy_file, atomic_write_json

WEEKLY_ARTIFACTS = (
    "projections.parquet",
    "projections_ros.parquet",
    "rankings_ros.parquet",
    "projection_coverage.parquet",
    "alerts/latest.json",
)


def file_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "sha256": digest.hexdigest(),
    }


def inventory_weekly_artifacts(output_dir: Path) -> list[dict[str, object]]:
    return [
        {"name": name, **file_metadata(output_dir / name)}
        for name in WEEKLY_ARTIFACTS
        if (output_dir / name).is_file()
    ]


def promote_last_known_good(
    output_dir: Path,
    *,
    league_slug: str,
    season: int,
    week: int,
    now: datetime | None = None,
) -> Path:
    """Atomically copy the current decision artifacts into a recovery snapshot."""
    promoted_at = now or datetime.now(UTC)
    destination = output_dir / "last_known_good"
    artifacts: list[dict[str, object]] = []
    for name in WEEKLY_ARTIFACTS:
        source = output_dir / name
        if not source.is_file():
            continue
        target = destination / name
        atomic_copy_file(source, target)
        artifacts.append({"name": name, **file_metadata(target)})
    if not artifacts:
        raise FileNotFoundError(f"No weekly artifacts were available under {output_dir}")
    manifest: dict[str, object] = {
        "league_slug": league_slug,
        "season": season,
        "week": week,
        "promoted_at_utc": promoted_at.isoformat(),
        "artifacts": artifacts,
    }
    path = destination / "manifest.json"
    atomic_write_json(manifest, path)
    return path


__all__ = [
    "WEEKLY_ARTIFACTS",
    "file_metadata",
    "inventory_weekly_artifacts",
    "promote_last_known_good",
]
