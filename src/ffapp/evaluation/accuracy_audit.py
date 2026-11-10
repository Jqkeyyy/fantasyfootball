"""Matched-sample audits of saved forecasts; never reconstruct past predictions."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from ffapp.evaluation.weekly_review import build_weekly_review

SOURCES = {
    "Saved live estimate": "live_mean",
    "ESPN pipeline estimate": "espn_mean",
    "Custom model": "model_mean",
    "Recent-performance baseline": "b2_mean",
    "Consensus (scoring unverified)": "b3_mean",
}
STAGES = {
    "Before role / injury adjustments": "audit_base_mean",
    "After role adjustment": "audit_role_mean",
    "After injury adjustment": "audit_injury_mean",
    "Final saved projection": "live_mean",
}
KEYS = ["season", "week", "player_id"]


def matched_audit(
    history: pl.DataFrame,
    schedule: pl.DataFrame,
    stats: pl.DataFrame,
    injuries: pl.DataFrame,
    overrides: dict[str, Any],
    sources: dict[str, str],
    *,
    include_injured: bool = False,
    now: Any = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Use the latest pregame snapshot containing every selected estimate.

    A later kickoff snapshot with no competitors cannot be compared with earlier
    competitors. Actuals may be backfilled from another snapshot, forecasts may not.
    The returned review retains rejection reasons for coverage reporting.
    """
    if history.is_empty() or not sources:
        return pl.DataFrame(), pl.DataFrame()
    columns = list(dict.fromkeys(sources.values()))
    if any(c not in history.columns for c in columns):
        return pl.DataFrame(), pl.DataFrame()
    complete = history.filter(pl.all_horizontal([pl.col(c).is_finite() for c in columns]))
    if complete.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    # Copy outcomes only. Keep immutable forecasts and their original timestamps.
    outcomes = (
        history.filter(pl.col("actual_points").is_finite())
        .group_by(KEYS)
        .agg(pl.col("actual_points").last().alias("_actual"))
    )
    complete = (
        complete.join(outcomes, on=KEYS, how="left")
        .with_columns(pl.coalesce("actual_points", "_actual").alias("actual_points"))
        .drop("_actual")
    )
    review = build_weekly_review(
        complete, schedule, stats, pl.DataFrame(), injuries, overrides, now=now
    )
    statuses = ["Scored", "Injury excluded"] if include_injured else ["Scored"]
    eligible = review.filter(pl.col("status").is_in(statuses))
    payload = complete.unique(subset=[*KEYS, "as_of_utc"], keep="last").select(
        *KEYS,
        pl.col("as_of_utc").alias("snapshot"),
        *columns,
        *[
            c
            for c in ["is_my_roster", "was_starting", "espn_season_verified"]
            if c in complete.columns
        ],
    )
    return eligible.join(payload, on=[*KEYS, "snapshot"], how="inner"), review


def audit_metrics(rows: pl.DataFrame, sources: dict[str, str]) -> pl.DataFrame:
    """Every source is measured on exactly the same finite rows."""
    if rows.is_empty():
        return pl.DataFrame()
    rows = rows.filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in ["actual", *sources.values()]])
    )
    if rows.is_empty():
        return pl.DataFrame()
    result = []
    for label, column in sources.items():
        errors = rows[column] - rows["actual"]
        result.append(
            {
                "Source": label,
                "Players / weeks": rows.height,
                "Weeks": rows.select("season", "week").unique().height,
                "Average miss": errors.abs().mean(),
                "RMSE": math.sqrt(float((errors**2).mean())),
                "Bias (predicted − actual)": errors.mean(),
                "Within 5 points": (errors.abs() <= 5).mean(),
            }
        )
    return pl.DataFrame(result)


def adjustment_effects(rows: pl.DataFrame) -> pl.DataFrame:
    """Paired error changes for saved stages, including only changed predictions."""
    result = []
    stages = list(STAGES.items())
    for (before_name, before), (after_name, after) in zip(stages, stages[1:], strict=False):
        if rows.is_empty() or before not in rows.columns or after not in rows.columns:
            continue
        changed = rows.filter(
            pl.col(before).is_finite()
            & pl.col(after).is_finite()
            & ((pl.col(before) - pl.col(after)).abs() > 1e-8)
        )
        if changed.is_empty():
            continue
        gain = (changed[before] - changed["actual"]).abs() - (
            changed[after] - changed["actual"]
        ).abs()
        result.append(
            {
                "Step": f"{before_name} → {after_name}",
                "Changed player / weeks": changed.height,
                "Error reduction (positive helps)": gain.mean(),
                "Helped": (gain > 0).sum(),
                "Hurt": (gain < 0).sum(),
            }
        )
    return pl.DataFrame(result)
