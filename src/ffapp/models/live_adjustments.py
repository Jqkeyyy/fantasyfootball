"""Point-in-time injury and role adjustments for weekly and ROS projections.

The projection sources remain the estimate of a healthy player's scoring level.
This module changes that level only when current availability or a corroborated
usage shift supplies information that the source can be slow to reflect.  Every
change is returned as an audit row so the dashboard can explain it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import polars as pl

POINT_COLUMNS = ("mean", "q10", "q25", "q50", "q75", "q90")
UNAVAILABLE_STATUSES = {
    "out",
    "ir",
    "injured reserve",
    "pup",
    "nfi",
    "suspended",
    "na",
    "inactive",
}

_RANGE_WEEKS = re.compile(r"(?<!\d)(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*weeks?", re.I)
_SINGLE_WEEKS = re.compile(r"(?<!\d)(\d{1,2})\s*weeks?", re.I)
_RANGE_DAYS = re.compile(r"(?<!\d)(\d{1,3})\s*(?:-|–|—|to)\s*(\d{1,3})\s*days?", re.I)


def parse_injury_duration(text: str | None) -> tuple[int, int] | None:
    """Return a conservative `(minimum, maximum)` missed-week range from text."""
    if not text:
        return None
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("season-ending", "season ending", "out for season")):
        return (99, 99)
    match = _RANGE_WEEKS.search(text)
    if match:
        low, high = sorted((int(match.group(1)), int(match.group(2))))
        return (low, high)
    match = _SINGLE_WEEKS.search(text)
    if match:
        weeks = int(match.group(1))
        return (weeks, weeks)
    match = _RANGE_DAYS.search(text)
    if match:
        low, high = sorted((int(match.group(1)), int(match.group(2))))
        return (max(1, math.ceil(low / 7)), max(1, math.ceil(high / 7)))
    if "week-to-week" in lowered or "week to week" in lowered:
        return (1, 3)
    if "day-to-day" in lowered or "day to day" in lowered:
        return (0, 1)
    return None


def _availability_for_week(status: str, duration: tuple[int, int] | None, offset: int) -> float:
    normalized = status.strip().lower()
    if normalized == "questionable":
        return 0.85 if offset == 0 else 1.0
    if normalized == "doubtful":
        return (0.20, 0.75, 1.0)[min(offset, 2)]
    if normalized not in UNAVAILABLE_STATUSES:
        return 1.0
    if duration == (99, 99):
        return 0.0
    if duration is not None:
        low, high = duration
        if offset < low:
            return 0.0
        if high <= low:
            return 0.75 if offset == low else 1.0
        if offset >= high:
            return 1.0
        return (offset - low + 1) / (high - low + 1)
    if normalized in {"ir", "injured reserve", "pup", "nfi"}:
        return (0.0, 0.0, 0.0, 0.0, 0.55, 0.85, 1.0)[min(offset, 6)]
    return (0.0, 0.45, 0.80, 0.95, 1.0)[min(offset, 4)]


def build_injury_adjustments(
    projections: pl.DataFrame, players_dim: pl.DataFrame, *, anchor_week: int
) -> pl.DataFrame:
    """Build one explained availability multiplier per affected player-week."""
    required = {"player_id", "injury_status"}
    if projections.is_empty() or not required.issubset(players_dim.columns):
        return pl.DataFrame()
    identity_columns = [
        column
        for column in (
            "player_id",
            "full_name",
            "injury_status",
            "injury_body_part",
            "injury_notes",
        )
        if column in players_dim.columns
    ]
    identities = players_dim.select(identity_columns).unique(subset=["player_id"], keep="first")
    projection_columns = ["player_id", "season", "week"]
    if "mean" in projections.columns:
        projection_columns.append("mean")
    projected = projections.select(projection_columns).unique()
    if "mean" in projected.columns:
        projected = projected.filter(pl.col("mean").is_not_null() & (pl.col("mean") > 0.25))
    joined = projected.join(
        identities, on="player_id", how="left"
    )
    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        status = str(row.get("injury_status") or "").strip()
        if not status:
            continue
        notes = str(row.get("injury_notes") or "").strip()
        duration = parse_injury_duration(notes)
        offset = max(0, int(row["week"]) - anchor_week)
        multiplier = _availability_for_week(status, duration, offset)
        if multiplier >= 0.999:
            continue
        body = str(row.get("injury_body_part") or "").strip()
        if duration == (99, 99):
            timing = "season-ending designation"
        elif duration is not None:
            timing = f"reported {duration[0]}-{duration[1]} week recovery"
        elif status.lower() in {"ir", "injured reserve", "pup", "nfi"}:
            timing = "no return date; conservative reserve-list curve"
        else:
            timing = "no return date; conservative status-based curve"
        detail = ", ".join(value for value in (status, body) if value)
        rows.append(
            {
                "player_id": row["player_id"],
                "player_name": row.get("full_name"),
                "season": row["season"],
                "week": row["week"],
                "adjustment_type": "injury",
                "multiplier": multiplier,
                "direction": "down",
                "confidence": 1.0 if multiplier == 0.0 else 0.75,
                "reason": f"{detail}: {timing}",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _mean(values: Iterable[object]) -> float | None:
    usable = [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    return sum(usable) / len(usable) if usable else None


def build_role_adjustments(
    usage: pl.DataFrame,
    players_dim: pl.DataFrame,
    *,
    season: int,
    target_week: int,
) -> pl.DataFrame:
    """Detect sustained role movement using only games before the target week.

    Two recent games are compared with three to six earlier games.  A change
    must appear in at least two position-relevant shares, which prevents one
    touchdown, target, or unusual game script from moving a projection.
    """
    required = {"player_id", "season", "week", "offense_snap_pct"}
    if usage.is_empty() or not required.issubset(usage.columns):
        return pl.DataFrame()
    position_column = "position" if "position" in players_dim.columns else None
    if position_column is None:
        return pl.DataFrame()
    names = [
        column for column in ("player_id", "full_name", "position") if column in players_dim.columns
    ]
    identity = players_dim.select(names).unique(subset=["player_id"], keep="first")
    historical = usage.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < target_week))
    ).join(identity, on="player_id", how="inner")
    rows: list[dict[str, object]] = []
    thresholds = {
        "offense_snap_pct": 0.12,
        "target_share": 0.06,
        "carry_share": 0.10,
        "rz_touch_share": 0.10,
    }
    for player in historical.partition_by("player_id", as_dict=False):
        ordered = player.sort(["season", "week"])
        recent = ordered.tail(2)
        baseline_length = min(6, max(0, ordered.height - 2))
        baseline = ordered.slice(ordered.height - 2 - baseline_length, baseline_length)
        if recent.height < 2 or baseline.height < 3:
            continue
        recent_periods = recent.select("season", "week").to_dicts()
        if any(int(period["season"]) != season for period in recent_periods):
            continue
        recent_weeks = [int(period["week"]) for period in recent_periods]
        if max(recent_weeks) != target_week - 1 or min(recent_weeks) < target_week - 3:
            continue
        position = str(ordered["position"][-1] or "")
        metrics = ["offense_snap_pct"]
        if position in {"RB", "WR", "TE"}:
            metrics.append("target_share")
        if position in {"RB", "QB"}:
            metrics.append("carry_share")
        if position in {"RB", "WR", "TE"}:
            metrics.append("rz_touch_share")
        signals: list[tuple[str, float, float, float]] = []
        for metric in metrics:
            if metric not in ordered.columns:
                continue
            recent_mean = _mean(recent[metric].to_list())
            baseline_mean = _mean(baseline[metric].to_list())
            if recent_mean is None or baseline_mean is None:
                continue
            delta = recent_mean - baseline_mean
            if abs(delta) >= thresholds[metric]:
                signals.append((metric, delta, recent_mean, baseline_mean))
        positive = [signal for signal in signals if signal[1] > 0]
        negative = [signal for signal in signals if signal[1] < 0]
        coherent = positive if len(positive) >= len(negative) else negative
        if len(coherent) < 2:
            continue
        direction = 1.0 if coherent[0][1] > 0 else -1.0
        if any(signal[1] * direction <= 0 for signal in coherent):
            continue
        strength = sum(
            min(2.0, abs(delta) / thresholds[metric]) for metric, delta, _, _ in coherent
        ) / len(coherent)
        raw_multiplier = max(0.85, min(1.15, 1.0 + direction * 0.075 * strength))
        current_games = ordered.filter(pl.col("season") == season).height
        early_season_weight = min(1.0, current_games / 4.0)
        multiplier = 1.0 + (raw_multiplier - 1.0) * early_season_weight
        confidence = min(
            0.95,
            (0.60 + 0.05 * min(baseline.height, 6) + 0.05 * len(coherent))
            * (0.70 + 0.30 * early_season_weight),
        )
        labels = {
            "offense_snap_pct": "snap share",
            "target_share": "target share",
            "carry_share": "carry share",
            "rz_touch_share": "red-zone share",
        }
        evidence = "; ".join(
            f"{labels[metric]} {baseline_mean:.0%}→{recent_mean:.0%}"
            for metric, _, recent_mean, baseline_mean in coherent
        )
        guard = (
            f"; early-season strength {early_season_weight:.0%} "
            f"from {current_games}/4 games"
            if early_season_weight < 1.0
            else ""
        )
        rows.append(
            {
                "player_id": ordered["player_id"][-1],
                "player_name": ordered["full_name"][-1] if "full_name" in ordered else None,
                "season": season,
                "week": target_week,
                "adjustment_type": "role",
                "multiplier": multiplier,
                "direction": "up" if direction > 0 else "down",
                "confidence": confidence,
                "evidence_games": current_games,
                "early_season_weight": early_season_weight,
                "reason": f"two-game role shift: {evidence}{guard}",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def expand_role_adjustments(
    role_changes: pl.DataFrame, projection_weeks: pl.DataFrame, *, anchor_week: int
) -> pl.DataFrame:
    """Expand current role changes across a ROS horizon, fading them over four weeks."""
    if role_changes.is_empty() or projection_weeks.is_empty():
        return pl.DataFrame()
    projection_columns = ["player_id", "season", "week"]
    if "mean" in projection_weeks.columns:
        projection_columns.append("mean")
    projected = projection_weeks.select(projection_columns).unique()
    if "mean" in projected.columns:
        projected = projected.filter(pl.col("mean").is_not_null() & (pl.col("mean") > 0.25))
    base = role_changes.drop("week").join(
        projected.drop("mean", strict=False),
        on=["player_id", "season"],
        how="inner",
    )
    return base.with_columns(
        (
            1.0
            + (pl.col("multiplier") - 1.0)
            * (1.0 - 0.25 * (pl.col("week") - anchor_week)).clip(0.0, 1.0)
        ).alias("multiplier"),
        (
            pl.col("reason")
            + pl.when(pl.col("week") > anchor_week)
            .then(pl.lit("; effect fades as new usage arrives"))
            .otherwise(pl.lit(""))
        ).alias("reason"),
    ).filter((pl.col("multiplier") - 1.0).abs() > 0.001)


def apply_adjustments(
    projections: pl.DataFrame, adjustments: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply role first and injury second, returning adjusted rows and the audit."""
    if projections.is_empty() or adjustments.is_empty():
        return projections, adjustments
    keys = ["player_id", "season", "week"]
    combined = adjustments.group_by(keys).agg(
        pl.col("multiplier").product().alias("_total_multiplier")
    )
    adjusted = projections.join(combined, on=keys, how="left").with_columns(
        pl.col("_total_multiplier").fill_null(1.0)
    )
    adjusted = adjusted.with_columns(
        [
            (pl.col(column) * pl.col("_total_multiplier")).alias(column)
            for column in POINT_COLUMNS
            if column in adjusted.columns
        ]
    )
    if "p_active" in adjusted.columns:
        injury_caps = (
            adjustments.filter(pl.col("adjustment_type") == "injury")
            .group_by(keys)
            .agg(pl.col("multiplier").min().alias("_injury_cap"))
        )
        adjusted = adjusted.join(injury_caps, on=keys, how="left").with_columns(
            pl.min_horizontal(pl.col("p_active"), pl.col("_injury_cap").fill_null(1.0)).alias(
                "p_active"
            )
        ).drop("_injury_cap")
    return adjusted.drop("_total_multiplier"), adjustments


__all__ = [
    "POINT_COLUMNS",
    "UNAVAILABLE_STATUSES",
    "apply_adjustments",
    "build_injury_adjustments",
    "build_role_adjustments",
    "expand_role_adjustments",
    "parse_injury_duration",
]
