"""Decomposed model v2, Stage 2: opportunity (SPEC.md §11.4; not a numbered
TASKS.md task -- see docs/design-model-v2-stage2-opportunity.md for the full
design). Predicts a player's own expected targets, carries, and red-zone
touches for a week, as a plain arithmetic composition of their own trailing
share of team volume and Stage 1's own predicted team volume -- no trained
model here, per that design's own reasoning: Stage 1's own trained model
added real complexity (and a real constraint-sign bug) without beating a
naive baseline for team-level volume, so there's no reason to expect a
second trained model would do better at this stage either.

Position eligibility is NOT automatic from `features.usage`'s own share
columns -- `target_share_ewm_3`/`carry_share_ewm_3`/`rz_touch_share_ewm_6`
are computed for every row regardless of position (the `_WindowedFeature
.positions` field there is metadata for the feature registry only, not a
row filter that nulls out ineligible rows), so a QB row can carry a real
but meaningless non-null `target_share_ewm_3` value. This module gates each
formula explicitly using `features.usage.PASS_CATCHERS_AND_RB`/`RB_QB`, the
same real position lists `features.usage`'s own share features are already
documented against.

`stage1_predictions` must be Stage 1's own real out-of-sample walk-forward
predictions (`evaluation.backtest.run_walk_forward_backtest`'s own output
for `team_environment.TeamEnvironmentPredictor`), never Stage 1's ground
truth -- using ground truth would hide Stage 1's real prediction error and
make this stage look better than it would actually perform live. Building
that predictions table is the evaluation script's job (not this module's),
matching the same separation Stage 1 itself keeps between its own table-
building functions and its evaluation script.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB

TARGET_COLUMNS = ["targets", "carries", "rz_touches"]


def build_opportunity_table(
    player_week_features: pl.DataFrame,
    player_week_usage: pl.DataFrame,
    stage1_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """One row per real `(player_id, season, week)` from
    `player_week_features` (task 1.9's own assembled table -- already has
    real `position`/`team` and the already-lag-shifted trailing shares),
    joined to `player_week_usage`'s own real, same-week target/carry/
    red-zone-touch counts (the real outcomes this stage is trying to
    predict -- not shifted, matching how Stage 1's own `team_plays`/
    `pass_rate` targets came from `team_context`'s same-week real values)
    and `stage1_predictions` (see module docstring).

    `expected_targets`/`expected_carries`/`expected_rz_touches` are null
    for a position that share doesn't apply to (e.g. `expected_targets`
    for a QB row) -- an honest "not applicable," not a guessed zero.
    """
    features = player_week_features.select(
        "player_id",
        "season",
        "week",
        "team",
        "position",
        "target_share_ewm_3",
        "carry_share_ewm_3",
        "rz_touch_share_ewm_6",
    )
    with_predictions = features.join(stage1_predictions, on=["team", "season", "week"], how="left")
    with_real_outcomes = with_predictions.join(
        player_week_usage.select(
            "player_id", "season", "week", "targets", "carries", "rz_targets", "rz_carries"
        ),
        on=["player_id", "season", "week"],
        how="left",
    )
    return with_real_outcomes.with_columns(
        (pl.col("rz_targets") + pl.col("rz_carries")).alias("rz_touches"),
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("target_share_ewm_3") * pl.col("predicted_pass_attempts"))
        .otherwise(None)
        .alias("expected_targets"),
        pl.when(pl.col("position").is_in(RB_QB))
        .then(pl.col("carry_share_ewm_3") * pl.col("predicted_rush_attempts"))
        .otherwise(None)
        .alias("expected_carries"),
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("rz_touch_share_ewm_6") * pl.col("predicted_team_plays"))
        .otherwise(None)
        .alias("expected_rz_touches"),
    )


__all__ = ["TARGET_COLUMNS", "build_opportunity_table"]
