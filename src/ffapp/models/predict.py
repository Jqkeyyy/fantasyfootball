"""Projection output pipeline (SPEC.md §6.2, §11.8; task 1.18).

`ffapp project --week N` composes real fitted models -- Part A
`models.availability` always, plus whichever conditional-mean source
`projection_source` names (`SPEC-ADDENDUM-04.md` §C: `"direct"` task
1.15's `models.points`, `"anchored"` task 1.20's `models.residual`,
`"baseline_b2"`/`"consensus_b3"` real external/trailing point
estimates) -- into SPEC §6.2's own `outputs/projections.parquet` schema:
real per-row `p_active`/`mean`/`q10`..`q90`/`projection_source`.

For `"direct"`/`"anchored"`, `mean` is SPEC §11.2's hurdle formula
applied literally, `E[points] = P(plays) x E[points | plays]`, and
`q10`..`q90` are `models.quantiles.mixture_with_p_active`'s own
unconditional output -- SPEC §11.5's mixture math, already proven
correct by task 1.16's own acceptance bar, not re-derived here.

For `"baseline_b2"`/`"consensus_b3"`, `mean` is already an unconditional
quantity by construction (no `p_active` re-multiplication -- see
`project_week`'s own docstring), and `q10`..`q90` come from the real
empirical distribution of that source's own historical error
(`models.baselines.empirical_error_quantiles`) -- task 1.16's own
quantile model, recentered around a different-source mean, was tried
first and failed a real coverage check badly (`docs/JOURNAL.md`'s
2026-08-16 entry has the full account).

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

from ffapp.config import (
    DEFAULT_QUANTILES,
    PROJECTION_SOURCES,
    InvalidProjectionSourceError,
    LightGBMSettings,
    Settings,
)
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import availability, baselines, points, quantiles, residual

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
    "projection_source",
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
    "projection_source": pl.String,
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


def _mean_source_feature_columns(projection_source: str, position: str) -> list[str]:
    """The real per-position feature set the configured `projection_source`
    actually used to produce `mean` -- `feature_hash`'s own job is
    detecting "did this row's own feature set change," so it must track
    whichever model actually ran, not always `points.py`'s own set.
    `baseline_b2`/`consensus_b3` fit no LightGBM model at all any more (a
    trailing average / a real external estimate for `mean`, an empirical
    error distribution -- not a learned model -- for the quantile spread),
    so their own real feature footprint is honestly empty."""
    if projection_source == "anchored":
        return residual.residual_feature_columns(position)
    if projection_source == "direct":
        return points.feature_columns(position)
    return []


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
    projection_source: str = "direct",
    players_dim: pl.DataFrame | None = None,
    b3_historical: pl.DataFrame | None = None,
    offline: bool | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """The real pipeline: fit availability + quantiles (task 1.14/1.16,
    always) plus whichever conditional-mean source `projection_source`
    names (`SPEC-ADDENDUM-04.md` §C) on every real row strictly before
    `(season, week)`, predict onto that week's own real row universe,
    combine per SPEC's hurdle formula and mixture math. Empty (not a
    crash) when there's too little training data or no real row universe
    for the target week yet -- e.g. a season that hasn't been
    played/published (CLAUDE.md rule 4 applied to "not enough data"
    rather than a join).

    **`projection_source` (default `"direct"`, task 1.15's own original
    behaviour -- the production default lives in `config.ModelSettings`,
    `"consensus_b3"`, and is the caller's job to pass explicitly; this
    function's own default only preserves prior callers' behaviour):**

    - `"direct"` (task 1.15): `E[points] = P(plays) x E[points|plays]`,
      the points model predicting from scratch.
    - `"anchored"` (task 1.20): same hurdle formula, but
      `E[points|plays] = B2 + residual_model(features)` -- neither
      cleared its own real evaluation bar (`docs/JOURNAL.md`'s
      2026-08-16 closing entry), kept as a real, working option (SPEC's
      own enum names it) but not the shipped default.
    - `"baseline_b2"`: `mean` is the player's own trailing `b2_ewm_4`
      directly -- already an unconditional quantity by construction (a
      trailing average of real scored points, including real zero-score
      weeks), so the hurdle formula's `p_active` multiplier is not
      reapplied on top of it (that would double-count the same
      playing-time signal B2's own history already reflects).
    - `"consensus_b3"` (the real shipped default): `mean` is a real,
      live-fetched FantasyPros weekly consensus point estimate
      (`baselines.fetch_b3_for_week`) -- same reasoning, no `p_active`
      re-multiplication. Requires `players_dim` (to resolve FantasyPros'
      own names to this project's canonical `player_id`) and
      `b3_historical` (real historical B3 rows, `player_id`/`season`/
      `week`/`b3_points` -- `data/interim/b3_predictions.parquet`, this
      project's own real materialized archive); raises `ValueError` if
      either is missing.

    For `"baseline_b2"`/`"consensus_b3"`, `q10`..`q90` come from the real,
    per-position **empirical distribution of `target - mean`** on real
    historical rows strictly before this week (`baselines
    .empirical_error_quantiles`/`apply_empirical_error_quantiles`) --
    `SPEC-ADDENDUM-04.md` §D's own blocking work item, resolved by a real
    coverage check (`docs/JOURNAL.md`'s 2026-08-16 entry): task 1.16's
    own already-validated quantile SHAPE, recentered around a
    different-source mean, was tried first and badly under-covered (up
    to 24 percentage points off nominal on the 80% interval, every
    position) -- v1's own width_scale was calibrated against v1's own
    conditional mean's residuals, not B2/B3's real, different error
    structure. The empirical approach cleared task 1.16's own 5-point
    bar at every position on both the 50% and 80% intervals on real
    held-out 2023-2024 data. For `"baseline_b2"`, the historical
    residuals come directly from `train_rows` (already walk-forward
    correct -- `b2_ewm_4` and `target` are both already present for
    every strictly-prior row). For `"consensus_b3"`, they come from
    `b3_historical` joined onto `train_rows` (its own real archive
    predates live fetching every historical week's B3 on every call,
    which would be prohibitively expensive against GitHub's real rate
    limit).
    """
    if projection_source not in PROJECTION_SOURCES:
        raise InvalidProjectionSourceError(
            f"projection_source={projection_source!r} is not one of {PROJECTION_SOURCES}"
        )
    if projection_source == "consensus_b3":
        if players_dim is None:
            raise ValueError("players_dim is required when projection_source='consensus_b3'")
        if b3_historical is None:
            raise ValueError("b3_historical is required when projection_source='consensus_b3'")

    if projection_source in ("baseline_b2", "anchored"):
        features = baselines.add_b2_ewm_4(features)
    if projection_source == "anchored":
        features = residual.add_points_history_features(features)

    train_rows, target_rows = _train_target_split(features, season, week, train_start)
    if train_rows.height < min_train_rows or target_rows.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    availability_model = availability.fit_availability_model(
        train_rows, lightgbm_params=lightgbm_params
    )
    p_active = availability.predict_p_active(availability_model, target_rows)

    # One cohesive working frame from here on -- `target_rows`' own row
    # order, with p_active attached by direct Series assignment
    # (`predict_p_active` writes back via boolean-mask numpy indexing
    # internally, so its own output order is guaranteed aligned with
    # `target_rows`). `mean` for `direct`/`anchored`/`baseline_b2` is
    # attached the same safe way. Only `consensus_b3` needs a real JOIN
    # (external data, keyed by `player_id`) -- done as a single
    # whole-frame join so every already-attached column (`p_active`)
    # travels with it, rather than extracting a bare Series from a
    # separately-ordered joined frame and re-attaching it positionally,
    # which a polars join's row order is not guaranteed to preserve
    # (this project's own established caution, see `interim.build`'s
    # own `cum_sum().shift(1)` re-sort precedent).
    work = target_rows.select("player_id", "season", "week", "position").with_columns(
        p_active.alias("p_active")
    )

    all_feature_names: set[str] = set(availability.FEATURE_COLUMNS)
    fitted_positions: set[str] = set()
    error_quantiles: dict[str, dict[float, float]] | None = None

    if projection_source == "direct":
        points_model = points.fit_points_model(train_rows, lightgbm_params=lightgbm_params)
        conditional_mean = points.predict_points(points_model, target_rows)
        work = work.with_columns((p_active * conditional_mean).alias("mean"))
        fitted_positions |= set(points_model.boosters)
    elif projection_source == "anchored":
        residual_model = residual.fit_residual_model(train_rows, lightgbm_params=lightgbm_params)
        conditional_mean = residual.predict_residual_points(residual_model, target_rows)
        work = work.with_columns((p_active * conditional_mean).alias("mean"))
        fitted_positions |= set(residual_model.boosters)
    elif projection_source == "baseline_b2":
        work = work.with_columns(target_rows["b2_ewm_4"].alias("mean"))
        # Real, per-position empirical distribution of `target - b2_ewm_4`
        # on strictly-prior rows -- already walk-forward correct, `train_rows`
        # already carries both columns. See this function's own docstring
        # for why this replaced task 1.16's quantile spread here.
        error_quantiles = baselines.empirical_error_quantiles(
            train_rows, "b2_ewm_4", quantile_alphas
        )
    else:  # consensus_b3
        assert players_dim is not None  # validated above; narrows the type for mypy
        assert b3_historical is not None
        cutoff_utc = target_rows["as_of_utc"][0]
        b3 = baselines.fetch_b3_for_week(
            season, week, cutoff_utc, players_dim, offline=offline, settings=settings
        )
        # `.unique(keep="first")` guards the join against a real (if
        # unlikely) duplicate `player_id` in the external source --
        # without it, a duplicate on the join's right side would fan out
        # `work`'s own rows and silently change `result.height`.
        b3_deduped = b3.unique(subset=["player_id"], keep="first")
        work = work.join(
            b3_deduped.select("player_id", pl.col("b3_points").alias("mean")),
            on="player_id",
            how="left",
        )
        # Real historical B3 archive, joined onto strictly-prior rows --
        # this project's real materialized archive predates live-fetching
        # every historical week's B3 on every call (prohibitively
        # expensive against GitHub's real rate limit).
        train_rows_with_b3 = train_rows.join(
            b3_historical.select("player_id", "season", "week", "b3_points"),
            on=["player_id", "season", "week"],
            how="inner",
        )
        error_quantiles = baselines.empirical_error_quantiles(
            train_rows_with_b3, "b3_points", quantile_alphas
        )

    for position in fitted_positions:
        all_feature_names.update(_mean_source_feature_columns(projection_source, position))
    model_version = compute_model_version(
        sorted(all_feature_names), lightgbm_params, (season, week), code_version
    )
    feature_hash_df = pl.DataFrame(
        {
            "position": list(SKILL_POSITIONS),
            "feature_hash": [
                compute_feature_hash(_mean_source_feature_columns(projection_source, position))
                for position in SKILL_POSITIONS
            ],
        }
    )

    if error_quantiles is not None:
        for tau, column_name in _Q_COLUMN_NAMES.items():
            recentered = baselines.apply_empirical_error_quantiles(
                work["mean"], work["position"], error_quantiles, tau
            )
            work = work.with_columns(recentered.alias(column_name))
    else:
        quantile_model = quantiles.fit_quantile_models(
            train_rows, lightgbm_params=lightgbm_params, quantile_alphas=quantile_alphas
        )
        conditional_quantiles = quantiles.predict_quantiles(quantile_model, target_rows)
        unconditional = quantiles.mixture_with_p_active(
            conditional_quantiles, p_active, quantile_alphas
        )
        for tau, column_name in _Q_COLUMN_NAMES.items():
            work = work.with_columns(unconditional[f"unconditional_q_{tau}"].alias(column_name))

    result = (
        work.join(feature_hash_df, on="position", how="left")
        .with_columns(
            pl.lit(model_version).alias("model_version"),
            pl.lit(projection_source).alias("projection_source"),
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
