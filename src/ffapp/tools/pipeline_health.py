"""Structured health checks for weekly projection artifacts and source logs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import polars as pl

from ffapp.config import Settings

HealthStatus = Literal["healthy", "degraded", "failed"]
_CRITICAL_LOG_SOURCES = {"fantasypros"}
_MIN_ACTUAL_COVERAGE = 0.98


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: HealthStatus
    detail: str


@dataclass(frozen=True)
class PipelineHealth:
    status: HealthStatus
    checks: tuple[HealthCheck, ...]


def _worst_status(checks: list[HealthCheck]) -> HealthStatus:
    if any(check.status == "failed" for check in checks):
        return "failed"
    if any(check.status == "degraded" for check in checks):
        return "degraded"
    return "healthy"


def inspect_weekly_pipeline(
    settings: Settings,
    league_slug: str,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
    stale_after_hours: float = 72.0,
) -> PipelineHealth:
    """Inspect the materialized weekly output and per-source prediction log."""
    current_time = now or datetime.now(UTC)
    checks: list[HealthCheck] = []
    projections_path = settings.data_root / "outputs" / league_slug / "projections.parquet"
    if not projections_path.exists():
        checks.append(HealthCheck("Weekly projections", "failed", f"Missing {projections_path}"))
    else:
        projections = pl.read_parquet(projections_path).filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        if projections.is_empty():
            checks.append(
                HealthCheck("Weekly projections", "failed", f"No rows for {season} week {week}")
            )
        else:
            usable = projections.filter(pl.col("mean").is_not_null()).height
            missing = projections.height - usable
            relevant_missing = missing
            if usable == 0:
                status: HealthStatus = "failed"
            else:
                coverage_path = (
                    settings.data_root
                    / "outputs"
                    / league_slug
                    / "projection_coverage.parquet"
                )
                if coverage_path.exists():
                    coverage = pl.read_parquet(coverage_path).filter(
                        (pl.col("season") == season) & (pl.col("week") == week)
                    )
                    if not coverage.is_empty():
                        relevant_missing = coverage.filter(
                            ~pl.col("projected") & pl.col("fantasy_relevant")
                        ).height
                status = "degraded" if relevant_missing else "healthy"
            if "projection_source" in projections.columns:
                source_counts = projections.group_by("projection_source").len().sort(
                    "len", descending=True
                )
                source = ", ".join(
                    f"{row['projection_source']}={row['len']}"
                    for row in source_counts.iter_rows(named=True)
                )
            else:
                source = "unknown"
            checks.append(
                HealthCheck(
                    "Weekly projections",
                    status,
                    (
                        f"{usable}/{projections.height} usable rows; "
                        f"{relevant_missing} fantasy-relevant missing; sources: {source}"
                    ),
                )
            )
        modified = datetime.fromtimestamp(projections_path.stat().st_mtime, tz=UTC)
        age_hours = (current_time - modified).total_seconds() / 3600
        checks.append(
            HealthCheck(
                "Artifact freshness",
                "degraded" if age_hours > stale_after_hours else "healthy",
                f"Updated {age_hours:.1f} hours ago",
            )
        )

    fetches_path = (
        settings.data_root / "outputs" / league_slug / "prediction_log" / "source_fetches.parquet"
    )
    if not fetches_path.exists():
        checks.append(HealthCheck("Source fetches", "degraded", "No source-fetch log found"))
    else:
        fetches = pl.read_parquet(fetches_path).filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        if fetches.is_empty():
            checks.append(
                HealthCheck("Source fetches", "degraded", f"No fetch log for {season} week {week}")
            )
        else:
            latest_time = fetches["fetched_at_utc"].max()
            latest = fetches.filter(pl.col("fetched_at_utc") == latest_time)
            failures = latest.filter(pl.col("fetch_error").is_not_null())
            failed_names = failures["source"].cast(pl.String).to_list()
            critical_failures = sorted(_CRITICAL_LOG_SOURCES.intersection(failed_names))
            checks.append(
                HealthCheck(
                    "Source fetches",
                    "degraded" if critical_failures else "healthy",
                    (
                        f"Critical failures: {', '.join(critical_failures)}"
                        if critical_failures
                        else f"Optional failures: {', '.join(failed_names)}"
                        if failed_names
                        else f"All {latest.height} logged sources succeeded"
                    ),
                )
            )

    if week > 1:
        prior_path = (
            settings.data_root
            / "outputs"
            / league_slug
            / "prediction_log"
            / f"season={season}"
            / f"week={week - 1:02d}.parquet"
        )
        if not prior_path.exists():
            checks.append(HealthCheck("Prior-week actuals", "degraded", "No prior-week log found"))
        else:
            prior = pl.read_parquet(prior_path)
            filled = prior.filter(pl.col("actual_points").is_not_null()).height
            actual_sum = float(prior["actual_points"].drop_nulls().sum() or 0.0)
            actual_coverage = filled / prior.height if prior.height else 0.0
            status = (
                "healthy"
                if actual_coverage >= _MIN_ACTUAL_COVERAGE and actual_sum > 0
                else "degraded"
            )
            checks.append(
                HealthCheck(
                    "Prior-week actuals",
                    status,
                    f"{filled}/{prior.height} filled; total actual points={actual_sum:.1f}",
                )
            )

    return PipelineHealth(status=_worst_status(checks), checks=tuple(checks))


def health_table(health: PipelineHealth) -> pl.DataFrame:
    """Render-friendly table for Streamlit and CLI callers."""
    return pl.DataFrame(
        [
            {"check": check.name, "status": check.status, "detail": check.detail}
            for check in health.checks
        ],
        schema={"check": pl.String, "status": pl.String, "detail": pl.String},
    )


__all__ = [
    "HealthCheck",
    "HealthStatus",
    "PipelineHealth",
    "health_table",
    "inspect_weekly_pipeline",
]
