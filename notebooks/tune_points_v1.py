"""Hyperparameter search for the conditional points model (task 1.15
revisit). Scratch, not imported by src/ (CLAUDE.md's notebooks/ rule).

Searches ONE shared LightGBM config across all four positions (QB/RB/WR/TE)
-- the same architecture already shipped in `models.points`, just with
tuned hyperparameters instead of the untuned shared defaults in
`config/settings.yml`. Per-position tuning was considered and rejected:
TE's real per-season row count (~1100-1300) is small enough that a
position-specific search would likely just fit search noise.

Dev seasons for the search are 2018-2020 -- strictly before the four
seasons already reported as 1.15's validation range (2021-2024) and
strictly before the fully-held-out season (2025, per SPEC §12.5's "hold
out the most recent season entirely"). This keeps the search itself from
being the "iterate on the validation seasons repeatedly" overfitting SPEC
warns against: the search sees 2018-2020 only, the real confirmation run
afterwards uses `ffapp evaluate` on 2021-2024 exactly once.

Refit cadence during the search is every 4 weeks, not every week --
SPEC §12.2's own sanctioned "faster experimentation" cadence -- to keep
~20-30 trials tractable. The final confirmation run uses the real
`ffapp evaluate` CLI, which still refits weekly at full fidelity; this
script's own cadence shortcut never touches that path.

**Real bug found while building this, fixed here but not (yet) upstream**:
LightGBM's sklearn `subsample` parameter does nothing unless
`subsample_freq` (bagging_freq) is also set > 0 -- `models.points`
(and `models.availability`/`models.dst`/`models.quantiles`, which share
the same `LightGBMSettings.subsample=0.8` default) never sets it, so
`subsample=0.8` has silently been a no-op everywhere. This script sets
`subsample_freq=1` for every trial so the `subsample` search dimension is
real. Not fixed in the other three modules here -- they're already-shipped,
already-passing tasks (1.14/1.16/2.7) and changing their effective
regularization needs its own confirmation, out of scope for 1.15.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ffapp.config import load_settings
from ffapp.models import baselines, points

DEV_SEASONS = [2018, 2019, 2020]
CADENCE_WEEKS = 4
N_TRIALS = 25
SEED = 20260813

SEARCH_SPACE: dict[str, list[float | int]] = {
    "n_estimators": [300, 500, 800, 1200, 1600],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [20, 40, 80, 150],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
}


@dataclass(frozen=True)
class Trial:
    params: dict[str, float | int]
    mae: float
    mae_b2: float
    spearman: float
    spearman_b2: float
    n_obs: int
    seconds: float


def sample_params(rng: random.Random) -> dict[str, float | int]:
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def fit_one_config(
    train_rows: pl.DataFrame, params: dict[str, float | int]
) -> dict[str, lgb.LGBMRegressor]:
    played = train_rows.filter(pl.col(points.AVAILABILITY_COLUMN))
    boosters: dict[str, lgb.LGBMRegressor] = {}
    for position in ["QB", "RB", "WR", "TE"]:
        position_rows = played.filter(pl.col("position") == position)
        if position_rows.is_empty():
            continue
        columns = points.feature_columns(position)
        booster = lgb.LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            subsample_freq=1,
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            monotone_constraints=points.monotone_constraints(position),
            verbosity=-1,
        )
        booster.fit(
            points.to_feature_frame(position_rows, columns),
            position_rows[points.TARGET_COLUMN].to_numpy(),
            categorical_feature=[c for c in points.CATEGORICAL_COLUMNS if c in columns],
        )
        boosters[position] = booster
    return boosters


def predict_one_config(boosters: dict[str, lgb.LGBMRegressor], rows: pl.DataFrame) -> pl.Series:
    predictions = np.full(rows.height, np.nan, dtype=float)
    row_positions = rows["position"].to_numpy()
    for position, booster in boosters.items():
        mask = row_positions == position
        if not mask.any():
            continue
        columns = points.feature_columns(position)
        subset = rows.filter(pl.Series(mask))
        predictions[mask] = np.asarray(booster.predict(points.to_feature_frame(subset, columns)))
    return pl.Series("prediction", predictions)


def run_dev_backtest(features: pl.DataFrame, params: dict[str, float | int]) -> pl.DataFrame:
    rows_out: list[pl.DataFrame] = []
    for season in DEV_SEASONS:
        weeks = sorted(features.filter(pl.col("season") == season)["week"].unique().to_list())
        for week in weeks[::CADENCE_WEEKS]:
            before = (pl.col("season") < season) | (
                (pl.col("season") == season) & (pl.col("week") < week)
            )
            train_rows = features.filter(before & (pl.col("season") >= 2015))
            if train_rows.height < 2000:
                continue
            target_rows = features.filter(
                (pl.col("season") == season)
                & (pl.col("week") == week)
                & pl.col(points.AVAILABILITY_COLUMN)
                & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
            )
            if target_rows.is_empty():
                continue
            boosters = fit_one_config(train_rows, params)
            preds = predict_one_config(boosters, target_rows)
            rows_out.append(
                target_rows.select(
                    "season",
                    "week",
                    "position",
                    pl.col(points.TARGET_COLUMN).alias("target"),
                    "b2_ewm_4",
                ).with_columns(preds)
            )
    return pl.concat(rows_out, how="vertical_relaxed")


def score(dev_predictions: pl.DataFrame) -> tuple[float, float, float, float, int]:
    df = dev_predictions.drop_nulls(["prediction", "b2_ewm_4", "target"])
    mae = float((df["prediction"] - df["target"]).abs().mean())
    mae_b2 = float((df["b2_ewm_4"] - df["target"]).abs().mean())

    def _weekly_spearman(col: str) -> list[float]:
        values = []
        for (_s, _w, _p), group in df.group_by(["season", "week", "position"]):
            if group.height < 3:
                continue
            rho, _ = spearmanr(group[col], group["target"])
            if not np.isnan(rho):
                values.append(rho)
        return values

    spear = _weekly_spearman("prediction")
    spear_b2 = _weekly_spearman("b2_ewm_4")
    return mae, mae_b2, float(np.mean(spear)), float(np.mean(spear_b2)), df.height


def main() -> None:
    settings = load_settings()
    features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")
    features = baselines.add_b2_ewm_4(features)

    rng = random.Random(SEED)
    trials: list[Trial] = []
    for i in range(N_TRIALS):
        params = sample_params(rng)
        t0 = time.time()
        dev_preds = run_dev_backtest(features, params)
        mae, mae_b2, spear, spear_b2, n_obs = score(dev_preds)
        elapsed = time.time() - t0
        trial = Trial(params, mae, mae_b2, spear, spear_b2, n_obs, elapsed)
        trials.append(trial)
        print(
            f"[{i + 1}/{N_TRIALS}] mae={mae:.4f} (b2={mae_b2:.4f}) "
            f"spearman={spear:.4f} (b2={spear_b2:.4f}) n={n_obs} "
            f"{elapsed:.1f}s params={params}"
        )

    trials.sort(key=lambda t: t.mae)
    print("\n--- best by MAE ---")
    for t in trials[:5]:
        print(t)

    trials_by_spearman = sorted(trials, key=lambda t: -t.spearman)
    print("\n--- best by Spearman ---")
    for t in trials_by_spearman[:5]:
        print(t)


if __name__ == "__main__":
    main()
