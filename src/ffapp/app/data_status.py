"""Shared league freshness status and one-click weekly refresh controls."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import streamlit as st

from ffapp.config import LeagueConfig, Settings, load_settings

DEFAULT_STALE_HOURS = 72.0


@dataclass(frozen=True)
class ArtifactFreshness:
    label: str
    path: Path
    exists: bool
    updated_at: datetime | None
    age_hours: float | None
    stale: bool


@dataclass(frozen=True)
class LeagueDataStatus:
    artifacts: tuple[ArtifactFreshness, ...]
    season: int | None
    week: int | None
    projection_source: str | None
    ros_schema_version: int | None


def artifact_freshness(
    label: str,
    path: Path,
    *,
    now: datetime | None = None,
    stale_after_hours: float = DEFAULT_STALE_HOURS,
) -> ArtifactFreshness:
    current = now or datetime.now(UTC)
    if not path.exists():
        return ArtifactFreshness(label, path, False, None, None, True)
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age = max(0.0, (current - updated).total_seconds() / 3600)
    return ArtifactFreshness(label, path, True, updated, age, age > stale_after_hours)


def league_data_status(settings: Settings, league: LeagueConfig) -> LeagueDataStatus:
    output = settings.data_root / "outputs" / league.slug
    weekly_path = output / "projections.parquet"
    ros_path = output / "projections_ros.parquet"
    rankings_path = output / "rankings_ros" / "latest.parquet"
    artifacts = tuple(
        artifact_freshness(label, path)
        for label, path in (
            ("Weekly", weekly_path),
            ("ROS projections", ros_path),
            ("ROS rankings", rankings_path),
        )
    )

    season: int | None = None
    week: int | None = None
    source: str | None = None
    if weekly_path.exists():
        weekly = pl.read_parquet(weekly_path)
        if not weekly.is_empty():
            latest = weekly.sort(["season", "week"], descending=True).row(0, named=True)
            season = int(latest["season"])
            week = int(latest["week"])
            raw_source = latest.get("projection_source")
            source = str(raw_source) if raw_source is not None else None

    schema_version: int | None = None
    if rankings_path.exists():
        schema = pl.read_parquet_schema(rankings_path)
        if "artifact_schema_version" in schema:
            versions = (
                pl.read_parquet(rankings_path, columns=["artifact_schema_version"])[
                    "artifact_schema_version"
                ]
                .drop_nulls()
                .unique()
                .to_list()
            )
            if len(versions) == 1:
                schema_version = int(versions[0])

    return LeagueDataStatus(artifacts, season, week, source, schema_version)


def run_label_for_date(value: datetime) -> str:
    """Choose the nearest standard snapshot for a manual refresh."""
    if value.weekday() <= 1:
        return "tuesday"
    if value.weekday() <= 3:
        return "thursday"
    return "sunday"


def load_latest_alerts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    alerts = payload.get("alerts", [])
    return alerts if isinstance(alerts, list) else []


def load_refresh_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_refresh_command(
    league_slug: str, *, executable: str = "uv", run_label: str | None = None
) -> list[str]:
    if Path(executable).name.lower().startswith("uv"):
        prefix = [executable, "run", "ffapp"]
    else:
        prefix = [executable]
    return [
        *prefix,
        "refresh",
        "weekly",
        "--league",
        league_slug,
        "--run-label",
        run_label or run_label_for_date(datetime.now().astimezone()),
        "--no-offline",
    ]


def run_weekly_refresh(
    league_slug: str, *, timeout_seconds: int = 900
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("uv") or shutil.which("ffapp")
    if executable is None:
        raise RuntimeError("Could not find `uv` or `ffapp` to run the weekly refresh.")
    return subprocess.run(
        build_refresh_command(league_slug, executable=executable),
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def render_league_data_controls(league: LeagueConfig) -> None:
    """Render the same freshness panel and refresh action on every league page."""
    settings = load_settings()
    status = league_data_status(settings, league)
    with st.sidebar.expander("Data status", expanded=False):
        if status.season is not None and status.week is not None:
            st.caption(f"Season {status.season} · Week {status.week}")
        st.caption(f"Projection source: {status.projection_source or 'unknown'}")
        if status.ros_schema_version is not None:
            st.caption(f"ROS artifact schema: v{status.ros_schema_version}")
        for artifact in status.artifacts:
            if not artifact.exists:
                st.error(f"{artifact.label}: missing")
            elif artifact.stale:
                st.warning(f"{artifact.label}: {artifact.age_hours:.1f}h old")
            else:
                st.success(f"{artifact.label}: {artifact.age_hours:.1f}h old")

        alerts = load_latest_alerts(
            settings.data_root / "outputs" / league.slug / "alerts" / "latest.json"
        )
        if alerts:
            st.caption(f"Latest decision alerts ({len(alerts)})")
            for alert in alerts[:5]:
                st.warning(str(alert.get("message", "Actionable lineup change detected.")))

        manifest = load_refresh_manifest(
            settings.data_root / "outputs" / league.slug / "refresh_runs" / "latest.json"
        )
        if manifest is not None:
            refresh_status = str(manifest.get("status", "unknown"))
            generated = str(manifest.get("generated_at_utc", "unknown time"))
            st.caption(f"Last refresh: {refresh_status} · {generated}")
            steps = manifest.get("steps", [])
            if isinstance(steps, list):
                problems = [
                    step
                    for step in steps
                    if isinstance(step, dict) and step.get("status") in {"degraded", "failed"}
                ]
                for step in problems[:3]:
                    st.warning(f"{step.get('name')}: {step.get('detail')}")

        result_key = f"refresh_result_{league.slug}"
        if result_key in st.session_state:
            st.caption(str(st.session_state.pop(result_key)))
        if st.button("Refresh league data", key=f"refresh_league_{league.slug}"):
            with st.spinner("Refreshing projections, rosters, ROS rankings, and alerts..."):
                try:
                    completed = run_weekly_refresh(league.slug)
                except Exception as exc:
                    st.error(f"Refresh failed: {exc}")
                else:
                    if completed.returncode == 0:
                        st.cache_data.clear()
                        st.session_state[result_key] = "Refresh completed successfully."
                        st.rerun()
                    else:
                        detail = completed.stderr.strip() or completed.stdout.strip()
                        st.error(f"Refresh failed: {detail[-1200:]}")


__all__ = [
    "ArtifactFreshness",
    "LeagueDataStatus",
    "artifact_freshness",
    "build_refresh_command",
    "league_data_status",
    "load_latest_alerts",
    "load_refresh_manifest",
    "render_league_data_controls",
    "run_label_for_date",
    "run_weekly_refresh",
]
