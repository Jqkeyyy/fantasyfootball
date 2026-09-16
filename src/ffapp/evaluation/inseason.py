"""Rolling in-season evaluation from immutable prediction snapshots."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from ffapp.config import LeagueConfig, Settings
from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup
from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet

_REFERENCE_SOURCE_COLUMNS = {
    "direct": "model_mean",
    "baseline_b2": "b2_mean",
    "consensus_b3": "b3_mean",
}
_SCHEMA = {
    "source": pl.String,
    "position": pl.String,
    "mae": pl.Float64,
    "rmse": pl.Float64,
    "weekly_spearman": pl.Float64,
    "n_obs": pl.Int64,
    "n_weeks": pl.Int64,
}
_RUN_ORDER = {"tuesday": 1, "thursday": 2, "sunday": 3}


def load_prediction_history(log_dir: Path) -> pl.DataFrame:
    """Load one league's partitioned week logs, excluding the latest pointer."""
    paths = sorted(log_dir.glob("season=*/week=*.parquet"))
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")


def _one_snapshot_per_player_week(history: pl.DataFrame) -> pl.DataFrame:
    return (
        history.with_columns(
            pl.col("run_label")
            .replace_strict(_RUN_ORDER, default=0, return_dtype=pl.Int64)
            .alias("_run_order")
        )
        .sort(["season", "week", "player_id", "_run_order"])
        .unique(subset=["season", "week", "player_id"], keep="last", maintain_order=True)
        .drop("_run_order")
    )


def valid_scored_history(history: pl.DataFrame) -> pl.DataFrame:
    """Keep completed weeks and reject placeholder all-zero backfills."""
    if history.is_empty() or "actual_points" not in history.columns:
        return pl.DataFrame()
    one_snapshot = _one_snapshot_per_player_week(history)
    valid_weeks = (
        one_snapshot.group_by("season", "week")
        .agg(
            pl.col("actual_points").is_not_null().sum().alias("n_actual"),
            pl.col("actual_points").abs().sum().alias("actual_magnitude"),
        )
        .filter((pl.col("n_actual") > 0) & (pl.col("actual_magnitude") > 0))
        .select("season", "week")
    )
    return one_snapshot.join(valid_weeks, on=["season", "week"], how="inner")


def summarize_inseason_performance(history: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling accuracy and within-week rank quality by source/position."""
    scored = valid_scored_history(history)
    if scored.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    summaries: list[dict[str, object]] = []
    positions = [*sorted(scored["position"].drop_nulls().unique().to_list()), "ALL"]
    source_columns = list(_REFERENCE_SOURCE_COLUMNS.items())
    if "live_mean" in scored.columns and "projection_source" in scored.columns:
        live_sources = scored["projection_source"].drop_nulls().unique().to_list()
        source_columns.extend(
            (str(source), "live_mean")
            for source in live_sources
            if str(source) not in _REFERENCE_SOURCE_COLUMNS
        )

    for source, column in source_columns:
        if column not in scored.columns:
            continue
        available = scored.filter(
            pl.col(column).is_not_null() & pl.col("actual_points").is_not_null()
        )
        if column == "live_mean":
            available = available.filter(pl.col("projection_source") == source)
        for position in positions:
            rows = (
                available if position == "ALL" else available.filter(pl.col("position") == position)
            )
            if rows.is_empty():
                continue
            errors = rows.select((pl.col(column) - pl.col("actual_points")).alias("error"))
            weekly_ranks = (
                rows.group_by("season", "week")
                .agg(pl.corr(column, "actual_points", method="spearman").alias("rho"))
                .drop_nulls("rho")
            )
            error_values = [float(value) for value in errors["error"].to_list()]
            rho_values = [float(value) for value in weekly_ranks["rho"].to_list()]
            summaries.append(
                {
                    "source": source,
                    "position": position,
                    "mae": sum(abs(value) for value in error_values) / len(error_values),
                    "rmse": math.sqrt(sum(value**2 for value in error_values) / len(error_values)),
                    "weekly_spearman": sum(rho_values) / len(rho_values) if rho_values else None,
                    "n_obs": rows.height,
                    "n_weeks": rows.select("season", "week").unique().height,
                }
            )
    return pl.DataFrame(summaries, schema=_SCHEMA) if summaries else pl.DataFrame(schema=_SCHEMA)


def summarize_interval_calibration(history: pl.DataFrame) -> pl.DataFrame:
    """Observed coverage and width for logged 50% and 80% prediction intervals."""
    scored = valid_scored_history(history)
    schema = {
        "source": pl.String,
        "interval": pl.String,
        "nominal_coverage": pl.Float64,
        "observed_coverage": pl.Float64,
        "mean_width": pl.Float64,
        "n_obs": pl.Int64,
    }
    if scored.is_empty():
        return pl.DataFrame(schema=schema)
    specs = [("consensus_b3", "b3")]
    if {"live_mean", "projection_source"}.issubset(scored.columns):
        specs.extend(
            (str(source), "live")
            for source in scored["projection_source"].drop_nulls().unique().to_list()
            if str(source) != "consensus_b3"
        )
    rows: list[dict[str, object]] = []
    for source, prefix in specs:
        source_rows = scored
        if prefix == "live":
            source_rows = source_rows.filter(pl.col("projection_source") == source)
        for label, lower, upper, nominal in (
            ("50%", f"{prefix}_q25", f"{prefix}_q75", 0.5),
            ("80%", f"{prefix}_q10", f"{prefix}_q90", 0.8),
        ):
            if lower not in source_rows.columns or upper not in source_rows.columns:
                continue
            available = source_rows.filter(
                pl.col(lower).is_not_null()
                & pl.col(upper).is_not_null()
                & pl.col("actual_points").is_not_null()
            )
            if available.is_empty():
                continue
            coverage = available.select(
                (
                    (pl.col("actual_points") >= pl.col(lower))
                    & (pl.col("actual_points") <= pl.col(upper))
                ).mean()
            ).item()
            width = available.select((pl.col(upper) - pl.col(lower)).mean()).item()
            rows.append(
                {
                    "source": source,
                    "interval": label,
                    "nominal_coverage": nominal,
                    "observed_coverage": float(coverage),
                    "mean_width": float(width),
                    "n_obs": available.height,
                }
            )
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def summarize_lineup_regret(history: pl.DataFrame, fmt: LeagueFormat) -> pl.DataFrame:
    """Compare each logged source's chosen roster lineup with the ex-post optimum."""
    scored = valid_scored_history(history)
    schema = {
        "source": pl.String,
        "mean_lineup_regret": pl.Float64,
        "n_weeks": pl.Int64,
    }
    if scored.is_empty() or "is_my_roster" not in scored.columns:
        return pl.DataFrame(schema=schema)
    scored = scored.filter(pl.col("is_my_roster") == True)  # noqa: E712
    source_columns = list(_REFERENCE_SOURCE_COLUMNS.items())
    if {"live_mean", "projection_source"}.issubset(scored.columns):
        source_columns.extend(
            (str(source), "live_mean")
            for source in scored["projection_source"].drop_nulls().unique().to_list()
            if str(source) not in _REFERENCE_SOURCE_COLUMNS
        )
    regrets: dict[str, list[float]] = {}
    for source, column in source_columns:
        if column not in scored.columns:
            continue
        for week_rows in scored.partition_by(["season", "week"], maintain_order=True):
            if column == "live_mean":
                week_rows = week_rows.filter(pl.col("projection_source") == source)
            available = week_rows.filter(
                pl.col(column).is_not_null() & pl.col("actual_points").is_not_null()
            )
            if available.is_empty():
                continue
            predicted = [
                PlayerProjection(str(row["player_id"]), str(row["position"]), value, value, value)
                for row in available.iter_rows(named=True)
                if (value := float(row[column])) >= 0
            ]
            actual = [
                PlayerProjection(str(row["player_id"]), str(row["position"]), value, value, value)
                for row in available.iter_rows(named=True)
                if (value := float(row["actual_points"])) >= 0
            ]
            chosen = optimal_lineup(predicted, fmt)
            optimal = optimal_lineup(actual, fmt)
            actual_by_id = {
                str(row["player_id"]): float(row["actual_points"])
                for row in available.iter_rows(named=True)
            }
            chosen_actual = sum(actual_by_id[player_id] for player_id in chosen.slots.values())
            regrets.setdefault(source, []).append(max(0.0, optimal.total_points - chosen_actual))
    rows = [
        {
            "source": source,
            "mean_lineup_regret": sum(values) / len(values),
            "n_weeks": len(values),
        }
        for source, values in regrets.items()
    ]
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def recommend_projection_source(
    performance: pl.DataFrame,
    current_source: str,
    *,
    min_observations: int = 200,
    min_weeks: int = 4,
    min_mae_improvement: float = 0.5,
) -> str:
    """Return a conservative source recommendation from completed in-season evidence."""
    eligible = performance.filter(
        (pl.col("position") == "ALL")
        & (pl.col("n_obs") >= min_observations)
        & (pl.col("n_weeks") >= min_weeks)
    ).sort(["mae", "weekly_spearman"], descending=[False, True])
    if eligible.is_empty():
        return "Insufficient evidence: keep the configured source."
    current = eligible.filter(pl.col("source") == current_source)
    if current.is_empty():
        return "The configured source lacks enough scored observations; review manually."
    best = eligible.row(0, named=True)
    current_mae = float(current["mae"].item())
    improvement = current_mae - float(best["mae"])
    if best["source"] != current_source and improvement >= min_mae_improvement:
        return f"Consider promoting {best['source']}: MAE improves by {improvement:.2f} points."
    return f"Keep {current_source}: no alternative clears the promotion threshold."


def source_reliability_weights(
    performance: pl.DataFrame, *, min_observations: int = 50, min_weeks: int = 2
) -> pl.DataFrame:
    """Turn scored MAE into transparent ensemble weights after enough evidence exists."""
    schema = {
        "source": pl.String,
        "position": pl.String,
        "weight": pl.Float64,
        "mae": pl.Float64,
        "n_obs": pl.Int64,
        "n_weeks": pl.Int64,
    }
    eligible = performance.filter(
        (pl.col("n_obs") >= min_observations)
        & (pl.col("n_weeks") >= min_weeks)
        & pl.col("mae").is_not_null()
    )
    if eligible.is_empty():
        return pl.DataFrame(schema=schema)
    return (
        eligible.with_columns((1.0 / pl.col("mae").clip(lower_bound=0.25)).alias("_score"))
        .with_columns((pl.col("_score") / pl.col("_score").sum().over("position")).alias("weight"))
        .select("source", "position", "weight", "mae", "n_obs", "n_weeks")
        .sort(["position", "weight"], descending=[False, True])
    )


def weekly_accuracy(history: pl.DataFrame) -> pl.DataFrame:
    """Return one rolling trend point per completed week and source."""
    scored = valid_scored_history(history)
    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "source": pl.String,
        "mae": pl.Float64,
        "n_obs": pl.Int64,
    }
    if scored.is_empty():
        return pl.DataFrame(schema=schema)
    rows: list[pl.DataFrame] = []
    source_columns = list(_REFERENCE_SOURCE_COLUMNS.items())
    if {"live_mean", "projection_source"}.issubset(scored.columns):
        source_columns.extend(
            (str(source), "live_mean")
            for source in scored["projection_source"].drop_nulls().unique().to_list()
            if str(source) not in _REFERENCE_SOURCE_COLUMNS
        )
    for source, column in source_columns:
        if column not in scored.columns:
            continue
        available = scored.filter(pl.col(column).is_not_null())
        if column == "live_mean":
            available = available.filter(pl.col("projection_source") == source)
        if not available.is_empty():
            rows.append(
                available.group_by("season", "week")
                .agg(
                    (pl.col(column) - pl.col("actual_points")).abs().mean().alias("mae"),
                    pl.len().alias("n_obs"),
                )
                .with_columns(pl.lit(source).alias("source"))
                .select("season", "week", "source", "mae", "n_obs")
            )
    return (
        pl.concat(rows).sort(["season", "week", "source"]) if rows else pl.DataFrame(schema=schema)
    )


def materialize_inseason_report(
    settings: Settings, league: LeagueConfig, fmt: LeagueFormat
) -> dict[str, object]:
    """Persist accuracy artifacts after refresh so the UI stays fast and auditable."""
    history = load_prediction_history(
        settings.data_root / "outputs" / league.slug / "prediction_log"
    )
    performance = summarize_inseason_performance(history)
    weights = source_reliability_weights(performance)
    calibration = summarize_interval_calibration(history)
    regret = summarize_lineup_regret(history, fmt)
    trends = weekly_accuracy(history)
    output = settings.data_root / "outputs" / league.slug / "model_health"
    for name, frame in (
        ("performance", performance),
        ("source_weights", weights),
        ("calibration", calibration),
        ("lineup_regret", regret),
        ("weekly_accuracy", trends),
    ):
        atomic_write_parquet(frame, output / f"{name}.parquet")
    recommendation = recommend_projection_source(performance, settings.model.projection_source)
    summary = {
        "league_slug": league.slug,
        "scored_weeks": int(trends.select("season", "week").unique().height),
        "recommendation": recommendation,
    }
    atomic_write_json(summary, output / "latest.json")
    return summary


__all__ = [
    "load_prediction_history",
    "materialize_inseason_report",
    "recommend_projection_source",
    "source_reliability_weights",
    "summarize_interval_calibration",
    "summarize_inseason_performance",
    "summarize_lineup_regret",
    "valid_scored_history",
    "weekly_accuracy",
]
