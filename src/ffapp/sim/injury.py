"""Injury hazard model (SPEC.md §13.3; task 2.3).

Discrete-time hazard: `P(misses game w | played through w-1, covariates)`.
"Keep this simple. It exists to stop the system pretending everyone
plays 17 games; precision beyond that is not where the value is" -- a
plain logistic regression (SPEC's own "logistic model or small GBM"),
not a second LightGBM classifier.

Deliberately built directly from raw nflverse tables
(`rosters`/`schedule`/`injuries`/`snap_counts`, real per-week status
history), not from `features/player_week_features.parquet` (task 1.9) --
unlike `models.availability` (task 1.14, Part A of SPEC §11.2, a
different question: "will this player record >=1 offensive snap this
week," fit on the full engineered feature table), this task's own inputs
are all nflverse-native and need no Sleeper league data at all:

- **Grid + target.** `nflreadpy`'s real per-week roster `status` field
  already distinguishes "on the 53-man, dressed" (`ACT`) from "on the
  53-man, inactive for this game" (`INA`) -- confirmed live and matching
  task 1.9's own gotcha. Restricting to those two statuses gives a clean,
  real gameday-roster population with an unambiguous binary outcome,
  `missed = (status == "INA")`, with no need to infer "played" from a
  box-score join at all.
- **Age.** `rosters`' own `birth_date` column against the real schedule's
  `gameday` for that (season, week) -- no crosswalk needed.
- **Snap load trend.** `snap_counts` only carries `pfr_player_id`, not
  `gsis_id` -- bridged via the dynastyprocess crosswalk's own
  `pfr_id <-> gsis_id` columns (the same crosswalk task 0.3 already
  trusts), still no Sleeper call.

`p_miss[player, week]` is "consumed by the season simulator and by the
games-played adjustment in §9.3" per SPEC -- neither integration is
built here (see the module's own `__all__`/tests for what's exported;
task 0.8's crude `POSITION_BASE_AVAILABILITY` prior stays in place until
a future task wires this in, same deferral task 0.8's own HANDOFF entry
already named).
"""

from __future__ import annotations

import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ffapp.interim.build import SKILL_POSITIONS

_GAMEDAY_STATUSES = ("ACT", "INA")
_NO_PRIOR_MISS_SENTINEL = 999

NUMERIC_COLUMNS = ["age", "missed_prior_two_seasons", "weeks_since_return", "snap_pct_trend"]
CATEGORICAL_COLUMNS = ["position", "report_status"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "missed"


def build_hazard_grid(rosters: pl.DataFrame) -> pl.DataFrame:
    """One row per (player, season, week) the player was truly on a real
    NFL gameday roster -- `status` in `ACT`/`INA` only (task 1.9's own
    gotcha: `DEV`/`RES`/`CUT`/... aren't gameday-roster membership, and
    SPEC's own hazard question doesn't apply to them).

    Scoped to `SKILL_POSITIONS` (QB/RB/WR/TE, the same set
    `models.points`/`models.quantiles` already train on) -- confirmed
    against a real live pull: `rosters`' raw `position` column spans
    every individual NFL position, including IDP ones (DB/OL/DL/LB/...),
    which outnumber the fantasy-relevant rows several times over (in a
    real 2015-2025 pull, DB/OL/DL/LB alone were the four *largest*
    position groups) and would otherwise dilute both the fit and the
    positional-base-rate comparison with a population this app has no
    use for. K is deliberately excluded too, matching this project's
    existing precedent of not GBM-modelling K (HANDOFF: "K/DST value is
    marginal"); DST doesn't participate at all -- it's a team, not an
    individual player, so this per-player hazard question doesn't apply
    to it."""
    return (
        rosters.filter(
            pl.col("status").is_in(list(_GAMEDAY_STATUSES))
            & pl.col("position").is_in(list(SKILL_POSITIONS))
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            "week",
            "position",
            "birth_date",
            (pl.col("status") == "INA").alias("missed"),
        )
        .unique(subset=["player_id", "season", "week"], keep="first")
        .sort(["player_id", "season", "week"])
    )


def add_age(grid: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Real age in years at that week's own earliest game date."""
    week_dates = (
        schedule.with_columns(pl.col("gameday").str.to_date())
        .group_by(["season", "week"])
        .agg(pl.col("gameday").min().alias("_week_date"))
    )
    return (
        grid.join(week_dates, on=["season", "week"], how="left")
        .with_columns(
            ((pl.col("_week_date") - pl.col("birth_date")).dt.total_days() / 365.25).alias("age")
        )
        .drop("_week_date")
    )


def add_missed_prior_two_seasons(grid: pl.DataFrame) -> pl.DataFrame:
    """SPEC's own "games missed in the prior two seasons" -- a real,
    strictly-prior sum (never the current season), a season-level
    covariate applied to every week of that season (CLAUDE.md rule 1/2:
    walk-forward by construction, no lookahead)."""
    season_misses = grid.group_by(["player_id", "season"]).agg(
        pl.col("missed").sum().alias("_season_misses")
    )
    prior1 = season_misses.select(
        "player_id", (pl.col("season") + 1).alias("season"), pl.col("_season_misses").alias("_m1")
    )
    prior2 = season_misses.select(
        "player_id", (pl.col("season") + 2).alias("season"), pl.col("_season_misses").alias("_m2")
    )
    return (
        grid.join(prior1, on=["player_id", "season"], how="left")
        .join(prior2, on=["player_id", "season"], how="left")
        .with_columns(
            (pl.col("_m1").fill_null(0) + pl.col("_m2").fill_null(0)).alias(
                "missed_prior_two_seasons"
            )
        )
        .drop("_m1", "_m2")
    )


def add_weeks_since_return(grid: pl.DataFrame) -> pl.DataFrame:
    """Real weeks elapsed since this player's most recent PRIOR missed
    week, strictly before the current row -- resets at every season
    boundary (an injury designation doesn't meaningfully carry across an
    offseason). `_NO_PRIOR_MISS_SENTINEL` when there's no real prior miss
    yet this season."""
    ordered = grid.sort(["player_id", "season", "week"])
    missed_week = pl.when(pl.col("missed")).then(pl.col("week")).otherwise(None)
    ordered = ordered.with_columns(missed_week.alias("_missed_week"))
    ordered = ordered.with_columns(
        pl.col("_missed_week").shift(1).over(["player_id", "season"]).alias("_prev_missed_week")
    )
    ordered = ordered.with_columns(
        pl.col("_prev_missed_week")
        .forward_fill()
        .over(["player_id", "season"])
        .alias("_last_missed_week")
    )
    return ordered.with_columns(
        pl.when(pl.col("_last_missed_week").is_not_null())
        .then(pl.col("week") - pl.col("_last_missed_week"))
        .otherwise(_NO_PRIOR_MISS_SENTINEL)
        .alias("weeks_since_return")
    ).drop("_missed_week", "_prev_missed_week", "_last_missed_week")


def add_injury_report(grid: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    """Real weekly `report_status` (Questionable/Doubtful/Out/...),
    `"None"` for a player with no injury-report entry that week (not on
    the report at all -- a real, meaningful category, not a missing
    value).

    `injuries`' own `season`/`week` columns come back `Float64` from
    nflreadpy (confirmed live -- `rosters`/`schedule`/`snap_counts` are
    all `Int32` for the same two columns; only `injuries` isn't, no
    nulls or fractional values involved, just an nflverse-side
    inconsistency) -- cast explicitly rather than let a silent-or-loud
    dtype mismatch decide the join (this project's own recurring
    gotcha, see HANDOFF.md)."""
    report = injuries.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("report_status").fill_null("None").alias("report_status"),
    ).unique(subset=["player_id", "season", "week"], keep="first")
    return grid.join(report, on=["player_id", "season", "week"], how="left").with_columns(
        pl.col("report_status").fill_null("None")
    )


def add_snap_pct_trend(
    grid: pl.DataFrame, snap_counts: pl.DataFrame, crosswalk: pl.DataFrame, *, window: int = 4
) -> pl.DataFrame:
    """Trailing mean offense snap share over the last `window` real
    weeks, strictly excluding the current week (`.shift(1)`, the same
    trailing-through-W-1 convention `models.baselines`' B2 uses).
    `snap_counts` only carries `pfr_player_id`, not `gsis_id` -- bridged
    through the crosswalk's own `pfr_id`/`gsis_id` columns."""
    bridge = crosswalk.select(pl.col("pfr_id"), pl.col("gsis_id").alias("player_id")).drop_nulls()
    snaps = (
        snap_counts.join(bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select("player_id", "season", "week", "offense_pct")
        .unique(subset=["player_id", "season", "week"], keep="first")
    )
    joined = grid.join(snaps, on=["player_id", "season", "week"], how="left").sort(
        ["player_id", "season", "week"]
    )
    return joined.with_columns(
        pl.col("offense_pct")
        .fill_null(0.0)
        .rolling_mean(window_size=window, min_samples=1)
        .shift(1)
        .over(["player_id", "season"])
        .fill_null(0.0)
        .alias("snap_pct_trend")
    ).drop("offense_pct")


def build_hazard_features(
    rosters: pl.DataFrame,
    schedule: pl.DataFrame,
    injuries: pl.DataFrame,
    snap_counts: pl.DataFrame,
    crosswalk: pl.DataFrame,
) -> pl.DataFrame:
    """The full assembly: `build_hazard_grid` plus every covariate
    function above, in the only order that respects each one's own
    dependencies."""
    grid = build_hazard_grid(rosters)
    grid = add_age(grid, schedule)
    grid = add_missed_prior_two_seasons(grid)
    grid = add_weeks_since_return(grid)
    grid = add_injury_report(grid, injuries)
    grid = add_snap_pct_trend(grid, snap_counts, crosswalk)
    return grid


def _to_feature_frame(rows: pl.DataFrame) -> pd.DataFrame:
    return rows.select(FEATURE_COLUMNS).to_pandas()


def fit_hazard_model(train_rows: pl.DataFrame) -> Pipeline:
    """A plain logistic regression (SPEC's own "logistic model or small
    GBM," the simpler of the two) -- `ColumnTransformer` scales the
    numeric covariates and one-hot encodes `position`/`report_status`
    (`handle_unknown="ignore"` so a category unseen in training, e.g. a
    report status that never appeared in the fit window, doesn't crash
    prediction). No `class_weight="balanced"` -- that would bias
    `predict_proba` away from a genuinely calibrated `P(miss)`, which is
    the whole point of this model (SPEC: consumed downstream by the
    season simulator and the games-played adjustment, both of which need
    real probabilities, not a discrimination-optimised score)."""
    # Real gap found live during task 13's own e2e verification, not by any unit
    # test: `age` is null for 157/96081 real 2015-2025 rows (a real nflverse
    # `birth_date` gap, or a roster week with no matching `schedule` row) --
    # `missed_prior_two_seasons`/`weeks_since_return`/`snap_pct_trend` are each
    # already null-safe by construction (explicit `.fill_null`/sentinel above),
    # but `age` wasn't, and scikit-learn's `LogisticRegression` refuses NaN
    # input outright. Median imputation (fit on train, applied identically at
    # predict time via the pipeline) keeps `predict_p_miss`'s one-row-in/
    # one-row-out contract intact rather than silently shrinking the output --
    # this project's own CLAUDE.md rule 4 concern (don't silently drop rows).
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_COLUMNS,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline([("preprocess", preprocessor), ("logit", LogisticRegression(max_iter=1000))])
    model.fit(_to_feature_frame(train_rows), train_rows[TARGET_COLUMN].to_numpy())
    return model


def predict_p_miss(model: Pipeline, rows: pl.DataFrame) -> pl.Series:
    proba = model.predict_proba(_to_feature_frame(rows))[:, 1]
    return pl.Series("p_miss", proba)


def positional_base_rate(train_rows: pl.DataFrame) -> dict[str, float]:
    """The task's own comparison baseline: each position's real
    marginal miss rate in `train_rows`, no covariates at all."""
    return dict(
        train_rows.group_by("position")
        .agg(pl.col(TARGET_COLUMN).cast(pl.Float64).mean().alias("rate"))
        .iter_rows()
    )


def predict_positional_base_rate(rates: dict[str, float], rows: pl.DataFrame) -> pl.Series:
    default = sum(rates.values()) / len(rates) if rates else 0.0
    return pl.Series(
        "p_miss_base_rate",
        [rates.get(position, default) for position in rows["position"].to_list()],
    )


def estimate_recovery_prob(hazard_grid: pl.DataFrame) -> dict[str, float]:
    """Real per-position injury-duration estimate for
    `sim.season.simulate_availability`'s own geometric-duration
    persistence mechanic (SPEC §13.4: "sample duration, not independent
    per-week draws"). A real "run" is a maximal sequence of consecutive
    real gameday-roster ROWS (`build_hazard_grid`'s own ACT/INA-scoped
    rows, sorted by week) with `missed=True` for the same player within
    the same season -- deliberately measured in consecutive real rows,
    not consecutive calendar week numbers, since a real bye week has no
    row in this grid at all (see `build_hazard_grid`'s own docstring) and
    `simulate_availability`'s own `remaining_weeks` concept already
    counts decision points the same way, not raw week numbers. A
    documented simplification, not a guess: this project has no per-team
    bye-aware duration model anywhere yet.

    `recovery_prob[position] = 1 / mean(real_run_length)` -- the method-
    of-moments estimator for a geometric distribution's own parameter,
    matching exactly what `simulate_availability`'s
    `rng.geometric(recovery_prob)` draws (mean = 1/p). A position with
    zero real recorded miss-runs is omitted, not defaulted -- there is
    nothing real to estimate from; callers fall back to
    `config.RosSettings.default_recovery_prob` explicitly.
    """
    ordered = hazard_grid.sort(["player_id", "season", "week"])
    prev_missed = pl.col("missed").shift(1).over(["player_id", "season"]).fill_null(False)
    run_start = pl.col("missed") & ~prev_missed
    run_id = run_start.cum_sum().over(["player_id", "season"]).alias("_run_id")
    with_run_id = ordered.with_columns(run_id)
    runs = (
        with_run_id.filter(pl.col("missed"))
        .group_by(["player_id", "season", "_run_id"])
        .agg(pl.len().alias("run_length"), pl.col("position").first().alias("position"))
    )
    if runs.is_empty():
        return {}
    by_position = runs.group_by("position").agg(pl.col("run_length").mean().alias("mean_duration"))
    return {
        row["position"]: float(1.0 / row["mean_duration"])
        for row in by_position.iter_rows(named=True)
    }


__all__ = [
    "CATEGORICAL_COLUMNS",
    "FEATURE_COLUMNS",
    "NUMERIC_COLUMNS",
    "TARGET_COLUMN",
    "add_age",
    "add_injury_report",
    "add_missed_prior_two_seasons",
    "add_snap_pct_trend",
    "add_weeks_since_return",
    "build_hazard_features",
    "build_hazard_grid",
    "estimate_recovery_prob",
    "fit_hazard_model",
    "positional_base_rate",
    "predict_p_miss",
    "predict_positional_base_rate",
]
