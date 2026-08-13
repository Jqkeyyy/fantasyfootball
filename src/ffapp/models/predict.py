"""Projection output pipeline (SPEC.md §6.2, §11.8; task 1.18).

`ffapp project --week N` composes every fitted model this project has --
Part A `models.availability`, Part B `models.points`, and
`models.quantiles` -- into SPEC §6.2's own `outputs/projections.parquet`
schema: real per-row `p_active`/`mean`/`q10`..`q90`. `mean` is SPEC
§11.2's hurdle formula applied literally, `E[points] = P(plays) x
E[points | plays]`. `q10`..`q90` are `models.quantiles.mixture_with_p_active`'s
own unconditional output -- SPEC §11.5's mixture math, already proven
correct by task 1.16's own acceptance bar, not re-derived here.

Mirrors `evaluation.backtest.run_walk_forward_backtest`'s own train/target
`(season, week)` split (task 1.12) rather than importing it directly:
that harness's `Predictor` protocol returns exactly one prediction column
per row, and this task genuinely needs three different model types'
outputs combined for a single target week -- the same boundary task
1.17's own CLI extension already names for the quantile models
specifically, one level further.

**Scoped to `SKILL_POSITIONS`** (QB/RB/WR/TE), matching every model this
composes. DST has its own separate model and its own separate weekly
streamer list (task 2.7), not folded into this schema; K has no model in
this project at all (this project's now-recurring "K/DST value is
marginal" precedent, e.g. task 0.9).

**`model_version`/`feature_hash`: SPEC §11.8's own formula, but two
distinct hashes, since SPEC lists both as separate output columns and
doesn't otherwise explain the difference.** `model_version` is one value
shared by every row a single invocation produces -- literally SPEC's
formula (the union of every feature name any of the three model types
used, this run's hyperparameters, the real training cutoff, and the
current git commit): the identity of *this generation run* as a whole.
`feature_hash` is narrower and genuinely per-row: just that row's own
position's feature-name set (`points.feature_columns` differs per
position via opponent-group columns), letting a caller detect "did this
row's own feature set change" independent of a hyperparameter or
training-cutoff change `model_version` would also pick up.
`data/models/<position>/<model_version>/` (SPEC §11.8's own directory --
serialised booster, hyperparams, training cutoff, evaluation report) is
not built -- not required by this task's own literal acceptance bar
("every row carries model_version, as_of_utc, feature_hash, and git
commit"), and every model in this pipeline is refit fresh on every
invocation (matching `evaluate`'s own ephemeral-fit precedent, task
1.17) rather than served from a persisted artefact.

`write_projections` upserts by `(season, week)` -- CLAUDE.md's own "all
ingest is idempotent, re-running for the same season/week overwrites
cleanly" rule, applied to this output for the first time: a prior run's
rows for a *different* week are preserved, this run's own week's rows
replace whatever was there before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import polars as pl

from ffapp.config import DEFAULT_QUANTILES, LightGBMSettings
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import availability, points, quantiles

OUTPUT_COLUMNS = [
    "player_id",
    "season",
    "week",
    "p_active",
    "mean",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "model_version",
    "as_of_utc",
    "feature_hash",
    "git_commit",
]

_OUTPUT_SCHEMA = {
    "player_id": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "p_active": pl.Float64,
    "mean": pl.Float64,
    "q10": pl.Float64,
    "q25": pl.Float64,
    "q50": pl.Float64,
    "q75": pl.Float64,
    "q90": pl.Float64,
    "model_version": pl.String,
    "as_of_utc": pl.String,
    "feature_hash": pl.String,
    "git_commit": pl.String,
}

_Q_COLUMN_NAMES = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}


def compute_model_version(
    feature_names: Sequence[str],
    hyperparams: LightGBMSettings,
    train_cutoff: tuple[int, int],
    code_version: str | None,
) -> str:
    """SPEC §11.8: `sha256(feature_names + hyperparams + train_cutoff +
    code_version)[:12]` -- one identity for an entire generation run (see
    module docstring for why this differs from `compute_feature_hash`)."""
    payload = json.dumps(
        {
            "feature_names": sorted(set(feature_names)),
            "hyperparams": asdict(hyperparams),
            "train_cutoff": list(train_cutoff),
            "code_version": code_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def compute_feature_hash(feature_names: Sequence[str]) -> str:
    """Narrower than `compute_model_version` -- just one position's real
    feature-name set, order-independent."""
    payload = ",".join(sorted(set(feature_names)))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _train_target_split(
    features: pl.DataFrame, season: int, week: int, train_start: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mirrors `evaluation.backtest.run_walk_forward_backtest`'s own
    train/target `(season, week)` split for a single target week (see
    module docstring for why this isn't imported directly)."""
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    train_rows = features.filter(before & (pl.col("season") >= train_start))
    target_rows = features.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("position").is_in(SKILL_POSITIONS)
    )
    return train_rows, target_rows


def project_week(
    features: pl.DataFrame,
    season: int,
    week: int,
    *,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings,
    code_version: str | None,
    now: datetime,
    quantile_alphas: Sequence[float] = DEFAULT_QUANTILES,
) -> pl.DataFrame:
    """The real pipeline: fit availability + points + quantiles on every
    real row strictly before `(season, week)`, predict onto that week's
    own real row universe, combine per SPEC's hurdle formula and mixture
    math. Empty (not a crash) when there's too little training data or no
    real row universe for the target week yet -- e.g. a season that
    hasn't been played/published (CLAUDE.md rule 4 applied to "not enough
    data" rather than a join)."""
    train_rows, target_rows = _train_target_split(features, season, week, train_start)
    if train_rows.height < min_train_rows or target_rows.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    availability_model = availability.fit_availability_model(
        train_rows, lightgbm_params=lightgbm_params
    )
    p_active = availability.predict_p_active(availability_model, target_rows)

    points_model = points.fit_points_model(train_rows, lightgbm_params=lightgbm_params)
    conditional_mean = points.predict_points(points_model, target_rows)

    quantile_model = quantiles.fit_quantile_models(
        train_rows, lightgbm_params=lightgbm_params, quantile_alphas=quantile_alphas
    )
    conditional_quantiles = quantiles.predict_quantiles(quantile_model, target_rows)
    unconditional = quantiles.mixture_with_p_active(
        conditional_quantiles, p_active, quantile_alphas
    )

    all_feature_names: set[str] = set(availability.FEATURE_COLUMNS)
    for position in points_model.boosters:
        all_feature_names.update(points.feature_columns(position))
    model_version = compute_model_version(
        sorted(all_feature_names), lightgbm_params, (season, week), code_version
    )
    feature_hash_df = pl.DataFrame(
        {
            "position": list(SKILL_POSITIONS),
            "feature_hash": [
                compute_feature_hash(points.feature_columns(position))
                for position in SKILL_POSITIONS
            ],
        }
    )

    mean = p_active * conditional_mean

    result = target_rows.select("player_id", "season", "week", "position").with_columns(
        p_active.alias("p_active"), mean.alias("mean")
    )
    for tau, column_name in _Q_COLUMN_NAMES.items():
        result = result.with_columns(unconditional[f"unconditional_q_{tau}"].alias(column_name))

    result = (
        result.join(feature_hash_df, on="position", how="left")
        .with_columns(
            pl.lit(model_version).alias("model_version"),
            pl.lit(now.isoformat()).alias("as_of_utc"),
            pl.lit(code_version).alias("git_commit"),
        )
        .drop("position")
    )

    return result.select(OUTPUT_COLUMNS)


def write_projections(projections: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    """Upsert by `(season, week)` (see module docstring). Returns the
    full combined table actually written, so a caller can report a real
    total row count without a second read."""
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        keys = projections.select("season", "week").unique()
        existing = existing.join(keys, on=["season", "week"], how="anti")
        combined = pl.concat([existing, projections], how="vertical_relaxed")
    else:
        combined = projections
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    return combined


__all__ = [
    "OUTPUT_COLUMNS",
    "compute_feature_hash",
    "compute_model_version",
    "project_week",
    "write_projections",
]
