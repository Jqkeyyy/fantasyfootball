"""Explain meaningful within-week projection movement across saved sources."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

SOURCE_COLUMNS = {
    "espn_mean": "ESPN",
    "b3_mean": "FantasyPros",
    "b2_mean": "recent scoring",
    "model_mean": "custom model",
}

MOVEMENT_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "player_id": pl.String,
    "position": pl.String,
    "team": pl.String,
    "from_run": pl.String,
    "to_run": pl.String,
    "as_of_utc": pl.String,
    "signal": pl.String,
    "direction": pl.String,
    "live_delta": pl.Float64,
    "espn_delta": pl.Float64,
    "fantasypros_delta": pl.Float64,
    "recent_scoring_delta": pl.Float64,
    "model_delta": pl.Float64,
    "availability_delta": pl.Float64,
    "sources_agreeing": pl.Int64,
    "first_mover": pl.String,
    "confidence": pl.Float64,
    "is_my_roster": pl.Boolean,
    "was_starting": pl.Boolean,
    "reason": pl.String,
}


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _threshold(value: float) -> float:
    """Ignore normal refresh noise: require 1.5 points or 10%, whichever is larger."""
    return max(1.5, abs(value) * 0.10)


def _delta(first: dict[str, Any], last: dict[str, Any], column: str) -> float | None:
    start, end = _number(first.get(column)), _number(last.get(column))
    return end - start if start is not None and end is not None else None


def _same_direction(value: float, direction: float) -> bool:
    return value * direction > 0


def _first_movers(rows: list[dict[str, Any]], meaningful: dict[str, float]) -> str:
    if not meaningful:
        return "model pipeline"
    first = rows[0]
    for candidate in rows[1:]:
        crossed: list[str] = []
        for column in meaningful:
            start = _number(first.get(column))
            current = _number(candidate.get(column))
            if (
                start is not None
                and current is not None
                and abs(current - start) >= _threshold(start)
            ):
                crossed.append(SOURCE_COLUMNS[column])
        if crossed:
            return " + ".join(crossed) if len(crossed) <= 2 else "multiple sources"
    return "unclear"


def detect_projection_movements(
    history: pl.DataFrame,
    *,
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Compare the earliest and latest snapshots for one player-week.

    Every comparison stays inside one `(season, week)` and uses saved snapshots,
    so a later game's result can never enter the movement explanation.
    """
    required = {"season", "week", "player_id", "run_label", "as_of_utc", "live_mean"}
    if history.is_empty() or not required.issubset(history.columns):
        return pl.DataFrame(schema=MOVEMENT_SCHEMA)
    periods = history.select("season", "week").unique().sort(["season", "week"])
    if periods.is_empty():
        return pl.DataFrame(schema=MOVEMENT_SCHEMA)
    selected_season = int(season if season is not None else periods["season"][-1])
    if week is None:
        selected_week = int(
            periods.filter(pl.col("season") == selected_season).sort("week")["week"][-1]
        )
    else:
        selected_week = week
    period = history.filter(
        (pl.col("season") == selected_season) & (pl.col("week") == selected_week)
    )
    rows_out: list[dict[str, object]] = []
    for player in period.partition_by("player_id", as_dict=False):
        rows = player.sort("as_of_utc").to_dicts()
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        live_delta = _delta(first, last, "live_mean")
        source_deltas = {
            column: value
            for column in SOURCE_COLUMNS
            if column in period.columns and (value := _delta(first, last, column)) is not None
        }
        meaningful = {
            column: value
            for column, value in source_deltas.items()
            if (start := _number(first.get(column))) is not None
            and abs(value) >= _threshold(start)
        }
        availability_delta = _delta(first, last, "p_active") or 0.0
        max_source_change = max((abs(value) for value in meaningful.values()), default=0.0)
        if (
            (live_delta is None or abs(live_delta) < 2.0)
            and max_source_change < 2.5
            and abs(availability_delta) < 0.25
        ):
            continue
        direction_value = live_delta if live_delta is not None and abs(live_delta) >= 0.5 else 0.0
        if direction_value == 0.0:
            direction_value = sum(meaningful.values()) or availability_delta
        agreeing = sum(
            1 for value in meaningful.values() if _same_direction(value, direction_value)
        )
        if abs(availability_delta) >= 0.25:
            signal, confidence = "availability", 0.92
        elif agreeing >= 2:
            signal, confidence = "confirmed", min(0.92, 0.72 + 0.08 * agreeing)
        elif not meaningful:
            signal, confidence = "pipeline", 0.70
        else:
            signal, confidence = "source-led", 0.66
        first_mover = (
            "availability"
            if signal == "availability"
            else _first_movers(rows, meaningful)
        )
        delta_text = ", ".join(
            f"{SOURCE_COLUMNS[column]} {value:+.1f}"
            for column, value in meaningful.items()
            if _same_direction(value, direction_value)
        )
        if signal == "availability":
            reason = (
                f"Availability moved {availability_delta:+.0%}; live projection "
                f"moved {(live_delta or 0.0):+.1f}."
            )
        elif agreeing >= 2:
            reason = (
                f"{first_mover} moved first; {agreeing} sources agree ({delta_text}). "
                f"Live projection moved {(live_delta or 0.0):+.1f}."
            )
        elif signal == "source-led":
            reason = (
                f"{first_mover} supplied the only meaningful source move ({delta_text}); "
                f"live projection moved {(live_delta or 0.0):+.1f}."
            )
        else:
            reason = (
                f"Live projection moved {(live_delta or 0.0):+.1f} without a matching external "
                "source move; check role, injury, and model-pipeline adjustments."
            )
        rows_out.append(
            {
                "season": selected_season,
                "week": selected_week,
                "player_id": str(last["player_id"]),
                "position": last.get("position"),
                "team": last.get("team"),
                "from_run": str(first["run_label"]),
                "to_run": str(last["run_label"]),
                "as_of_utc": str(last["as_of_utc"]),
                "signal": signal,
                "direction": "up" if direction_value > 0 else "down",
                "live_delta": live_delta,
                "espn_delta": source_deltas.get("espn_mean"),
                "fantasypros_delta": source_deltas.get("b3_mean"),
                "recent_scoring_delta": source_deltas.get("b2_mean"),
                "model_delta": source_deltas.get("model_mean"),
                "availability_delta": availability_delta,
                "sources_agreeing": agreeing,
                "first_mover": first_mover,
                "confidence": confidence,
                "is_my_roster": bool(last.get("is_my_roster") or False),
                "was_starting": bool(last.get("was_starting") or False),
                "reason": reason,
            }
        )
    if not rows_out:
        return pl.DataFrame(schema=MOVEMENT_SCHEMA)
    return pl.DataFrame(rows_out, schema=MOVEMENT_SCHEMA).sort(
        ["confidence", "live_delta"], descending=[True, True]
    )


__all__ = ["MOVEMENT_SCHEMA", "SOURCE_COLUMNS", "detect_projection_movements"]
