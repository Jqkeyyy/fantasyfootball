"""Personal feedback summaries from saved recommendations, choices, and outcomes."""

from __future__ import annotations

from typing import TypedDict

import polars as pl


class PersonalProfile(TypedDict):
    recommendations: int
    choices: int
    settled: int
    follow_rate: float | None
    helped: float | None
    mean_regret: float | None
    preferred_edge: float | None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def decision_calibration(ledger: pl.DataFrame) -> pl.DataFrame:
    """Measure whether recommendations with stated confidence actually helped."""
    if ledger.is_empty():
        return pl.DataFrame()
    settled = ledger.filter(pl.col("realized_delta").is_not_null())
    if settled.is_empty():
        return pl.DataFrame()
    return (
        settled.with_columns(
            (pl.col("realized_delta") > 0).alias("recommendation_won"),
            pl.when(pl.col("confidence").is_null())
            .then(pl.lit("Not scored"))
            .when(pl.col("confidence") >= 0.75)
            .then(pl.lit("High"))
            .when(pl.col("confidence") >= 0.55)
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Low"))
            .alias("confidence_band"),
        )
        .group_by("decision_type", "confidence_band")
        .agg(
            pl.len().alias("decisions"),
            pl.col("recommendation_won").mean().alias("help_rate"),
            pl.col("expected_delta").mean().alias("expected_gain"),
            pl.col("realized_delta").mean().alias("realized_gain"),
        )
        .sort("decisions", descending=True)
    )


def personal_profile(ledger: pl.DataFrame) -> PersonalProfile:
    """Summarize the user's recorded behavior and the value of followed advice."""
    if ledger.is_empty():
        return {
            "recommendations": 0,
            "choices": 0,
            "settled": 0,
            "follow_rate": None,
            "helped": None,
            "mean_regret": None,
            "preferred_edge": None,
        }
    choices = ledger.filter(pl.col("status").is_in(["accepted", "rejected"]))
    settled = choices.filter(pl.col("decision_regret").is_not_null())
    accepted = choices.filter(pl.col("status") == "accepted")
    helped = (
        _optional_float(
            (settled.filter(pl.col("status") == "accepted")["realized_delta"] > 0).mean()
        )
        if settled.filter(pl.col("status") == "accepted").height
        else None
    )
    return {
        "recommendations": ledger.height,
        "choices": choices.height,
        "settled": settled.height,
        "follow_rate": choices.filter(pl.col("status") == "accepted").height / choices.height
        if choices.height
        else None,
        "helped": helped,
        "mean_regret": _optional_float(settled["decision_regret"].mean())
        if settled.height
        else None,
        "preferred_edge": (
            _optional_float(accepted["expected_delta"].median()) if accepted.height else None
        ),
    }


def learning_message(ledger: pl.DataFrame) -> str:
    profile = personal_profile(ledger)
    if profile["choices"] == 0:
        return (
            "Record whether you followed each recommendation. Completed games will be "
            "scored automatically after refreshes."
        )
    settled_count = profile["settled"] if isinstance(profile["settled"], int) else 0
    if settled_count < 5:
        return (
            f"{profile['choices']} choice(s) recorded. Keep logging decisions; at least five "
            "settled choices are needed before treating a pattern as useful."
        )
    helped = profile["helped"]
    edge = profile["preferred_edge"]
    return (
        f"Followed recommendations have helped {helped:.0%} of the time when settled. "
        f"Your typical accepted projected edge is {edge:.1f} points."
        if isinstance(helped, int | float) and isinstance(edge, int | float)
        else "Your settled decision history is ready for review below."
    )


__all__ = ["decision_calibration", "learning_message", "personal_profile"]
