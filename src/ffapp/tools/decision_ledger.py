"""Durable recommendation, choice, and outcome tracking.

The ledger is deliberately generic: product code records one row per real
decision, then later annotates that same row with the user's choice and the
observed player outcomes.  This keeps recommendation quality measurable
without coupling the storage layer to Streamlit.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.tools.artifacts import atomic_write_parquet

LEDGER_SCHEMA = {
    "decision_id": pl.String,
    "league_slug": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "decision_type": pl.String,
    "recommended_action": pl.String,
    "subject_player_id": pl.String,
    "subject_player_name": pl.String,
    "alternative_player_id": pl.String,
    "alternative_player_name": pl.String,
    "expected_delta": pl.Float64,
    "confidence": pl.Float64,
    "status": pl.String,
    "created_at_utc": pl.String,
    "decided_at_utc": pl.String,
    "settled_at_utc": pl.String,
    "subject_actual_points": pl.Float64,
    "alternative_actual_points": pl.Float64,
    "realized_delta": pl.Float64,
    "decision_regret": pl.Float64,
}


def ledger_path(data_root: Path, league_slug: str) -> Path:
    return data_root / "outputs" / league_slug / "decisions" / "ledger.parquet"


def empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(schema=LEDGER_SCHEMA)


def _decision_id(row: dict[str, object]) -> str:
    identity = "|".join(
        str(row.get(field) or "")
        for field in (
            "league_slug",
            "season",
            "week",
            "decision_type",
            "subject_player_id",
            "alternative_player_id",
            "recommended_action",
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def recommendation_rows(
    league_slug: str,
    season: int,
    week: int,
    lineup: pl.DataFrame,
    rankings: pl.DataFrame,
    current_starter_ids: set[str],
    waivers: pl.DataFrame,
    *,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Create stable lineup-swap and waiver recommendation rows."""
    created_at = (now or datetime.now(UTC)).isoformat()
    recommended_ids = set(lineup["player_id"].to_list()) if not lineup.is_empty() else set()
    incoming = lineup.filter(~pl.col("player_id").is_in(list(current_starter_ids))).sort(
        "projected_points", descending=True
    )
    outgoing = rankings.filter(
        pl.col("player_id").is_in(list(current_starter_ids - recommended_ids))
    ).sort("proj_mean")
    rows: list[dict[str, object]] = []
    for add, drop in zip(
        incoming.iter_rows(named=True), outgoing.iter_rows(named=True), strict=False
    ):
        add_points = float(add["projected_points"])
        drop_points = float(drop["proj_mean"])
        spread = max(0.0, float(add["ceiling"]) - float(add["floor"]))
        row: dict[str, object] = {
            "league_slug": league_slug,
            "season": season,
            "week": week,
            "decision_type": "lineup",
            "recommended_action": f"Start {add['player_name']} over {drop['player_name']}",
            "subject_player_id": str(add["player_id"]),
            "subject_player_name": str(add["player_name"]),
            "alternative_player_id": str(drop["player_id"]),
            "alternative_player_name": str(drop["player_name"]),
            "expected_delta": add_points - drop_points,
            "confidence": max(0.0, min(1.0, 1.0 - spread / max(1.0, add_points * 4.0))),
            "status": "recommended",
            "created_at_utc": created_at,
            "decided_at_utc": None,
            "settled_at_utc": None,
            "subject_actual_points": None,
            "alternative_actual_points": None,
            "realized_delta": None,
            "decision_regret": None,
        }
        row["decision_id"] = _decision_id(row)
        rows.append(row)

    for waiver in waivers.iter_rows(named=True):
        row = {
            "league_slug": league_slug,
            "season": season,
            "week": week,
            "decision_type": "waiver",
            "recommended_action": (
                f"Add {waiver['player_name']}"
                + (f"; drop {waiver['drop_player']}" if waiver.get("drop_player") else "")
            ),
            "subject_player_id": str(waiver["player_id"]),
            "subject_player_name": str(waiver["player_name"]),
            "alternative_player_id": (
                str(waiver["drop_candidate"]) if waiver.get("drop_candidate") else None
            ),
            "alternative_player_name": (
                str(waiver["drop_player"]) if waiver.get("drop_player") else None
            ),
            "expected_delta": float(waiver["value_added_per_week"]),
            "confidence": None,
            "status": "recommended",
            "created_at_utc": created_at,
            "decided_at_utc": None,
            "settled_at_utc": None,
            "subject_actual_points": None,
            "alternative_actual_points": None,
            "realized_delta": None,
            "decision_regret": None,
        }
        row["decision_id"] = _decision_id(row)
        rows.append(row)
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA) if rows else empty_ledger()


def append_recommendations(path: Path, recommendations: pl.DataFrame) -> pl.DataFrame:
    """Append new stable IDs while preserving choices/outcomes already recorded."""
    existing = pl.read_parquet(path) if path.exists() else empty_ledger()
    if recommendations.is_empty():
        return existing
    new_rows = recommendations.filter(
        ~pl.col("decision_id").is_in(existing["decision_id"].to_list())
    )
    combined = pl.concat([existing, new_rows], how="vertical")
    atomic_write_parquet(combined, path)
    return combined


def record_choice(
    path: Path, decision_id: str, *, accepted: bool, now: datetime | None = None
) -> pl.DataFrame:
    """Record whether the user followed a recommendation."""
    ledger = pl.read_parquet(path)
    if decision_id not in ledger["decision_id"].to_list():
        raise KeyError(f"Unknown decision_id: {decision_id}")
    updated = ledger.with_columns(
        pl.when(pl.col("decision_id") == decision_id)
        .then(pl.lit("accepted" if accepted else "rejected"))
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.col("decision_id") == decision_id)
        .then(pl.lit((now or datetime.now(UTC)).isoformat()))
        .otherwise(pl.col("decided_at_utc"))
        .alias("decided_at_utc"),
    )
    atomic_write_parquet(updated, path)
    return updated


def settle_outcomes(
    path: Path, actuals: pl.DataFrame, *, now: datetime | None = None
) -> pl.DataFrame:
    """Settle recorded player-v-player decisions from actual fantasy points."""
    ledger = pl.read_parquet(path)
    required = {"season", "week", "player_id", "actual_points"}
    if not required.issubset(actuals.columns):
        raise ValueError(f"actuals must contain {sorted(required)}")
    points = {
        (int(row["season"]), int(row["week"]), str(row["player_id"])): float(
            row["actual_points"]
        )
        for row in actuals.drop_nulls("actual_points").iter_rows(named=True)
    }
    settled_at = (now or datetime.now(UTC)).isoformat()
    rows: list[dict[str, object]] = []
    for row in ledger.iter_rows(named=True):
        key = (int(row["season"]), int(row["week"]))
        subject = points.get((*key, str(row["subject_player_id"])))
        alternative_id = row["alternative_player_id"]
        alternative = (
            points.get((*key, str(alternative_id))) if alternative_id is not None else None
        )
        if subject is not None and alternative is not None:
            realized = subject - alternative
            status = str(row["status"])
            chosen_delta = realized if status == "accepted" else -realized
            row.update(
                {
                    "subject_actual_points": subject,
                    "alternative_actual_points": alternative,
                    "realized_delta": realized,
                    "decision_regret": (
                        max(0.0, -chosen_delta)
                        if status in {"accepted", "rejected"}
                        else None
                    ),
                    "settled_at_utc": settled_at,
                }
            )
        rows.append(row)
    updated = pl.DataFrame(rows, schema=LEDGER_SCHEMA)
    atomic_write_parquet(updated, path)
    return updated


def decision_summary(ledger: pl.DataFrame) -> pl.DataFrame:
    """Summarize recommendation adoption and settled value by decision type."""
    if ledger.is_empty():
        return pl.DataFrame(
            schema={
                "decision_type": pl.String,
                "recommendations": pl.UInt32,
                "choices_recorded": pl.UInt32,
                "follow_rate": pl.Float64,
                "settled": pl.UInt32,
                "mean_expected_delta": pl.Float64,
                "mean_realized_delta": pl.Float64,
                "mean_regret": pl.Float64,
            }
        )
    return (
        ledger.with_columns(
            pl.col("status").is_in(["accepted", "rejected"]).alias("choice_recorded"),
            (pl.col("status") == "accepted").alias("followed"),
            pl.col("settled_at_utc").is_not_null().alias("is_settled"),
        )
        .group_by("decision_type")
        .agg(
            pl.len().alias("recommendations"),
            pl.col("choice_recorded").sum().alias("choices_recorded"),
            pl.when(pl.col("choice_recorded"))
            .then(pl.col("followed").cast(pl.Float64))
            .otherwise(None)
            .mean()
            .alias("follow_rate"),
            pl.col("is_settled").sum().alias("settled"),
            pl.col("expected_delta").mean().alias("mean_expected_delta"),
            pl.col("realized_delta").mean().alias("mean_realized_delta"),
            pl.col("decision_regret").mean().alias("mean_regret"),
        )
        .sort("decision_type")
    )


__all__ = [
    "LEDGER_SCHEMA",
    "append_recommendations",
    "decision_summary",
    "empty_ledger",
    "ledger_path",
    "recommendation_rows",
    "record_choice",
    "settle_outcomes",
]
