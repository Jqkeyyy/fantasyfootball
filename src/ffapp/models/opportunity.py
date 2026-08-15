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
from ffapp.models.baselines import pooled_rolling_mean

TARGET_COLUMNS = ["targets", "carries", "rz_touches"]

_COMPOSITION_COLUMNS = {
    "targets": "expected_targets",
    "carries": "expected_carries",
    "rz_touches": "expected_rz_touches",
}


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
    # SPEC §11.1: a real active-roster player-week with no recorded stat
    # line (DNP/inactive -- no match in player_week_usage) is 0 real usage,
    # not a dropped/null row -- the same survivorship-bias rule
    # `features/build.py::_add_target_and_availability` already applies to
    # `target`/`availability_flag` (fill to 0/False, never drop), applied
    # here to the real usage counts this stage's outcomes are built from.
    with_real_outcomes = with_real_outcomes.with_columns(
        pl.col("targets").fill_null(0),
        pl.col("carries").fill_null(0),
        pl.col("rz_targets").fill_null(0),
        pl.col("rz_carries").fill_null(0),
    )
    with_real_outcomes = with_real_outcomes.with_columns(
        (pl.col("rz_targets") + pl.col("rz_carries")).alias("rz_touches")
    )
    with_real_outcomes = with_real_outcomes.join(
        _team_rz_touches_trailing(with_real_outcomes), on=["team", "season", "week"], how="left"
    )
    return with_real_outcomes.with_columns(
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("target_share_ewm_3") * pl.col("predicted_pass_attempts"))
        .otherwise(None)
        .alias("expected_targets"),
        pl.when(pl.col("position").is_in(RB_QB))
        .then(pl.col("carry_share_ewm_3") * pl.col("predicted_rush_attempts"))
        .otherwise(None)
        .alias("expected_carries"),
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("rz_touch_share_ewm_6") * pl.col("team_rz_touches_trailing_ewm_6"))
        .otherwise(None)
        .alias("expected_rz_touches"),
    )


def _team_rz_touches_trailing(table: pl.DataFrame) -> pl.DataFrame:
    """A team's own trailing red-zone-touch volume -- `rz_touch_share`'s
    real denominator (SPEC/`features/usage.py`: "(rz targets + rz carries)
    / team rz touches"), which no upstream table persists on its own
    (`interim/build.py` computes it only long enough to derive the *share*,
    then drops it). Stage 1 doesn't predict this either -- it only models
    `team_plays`/`pass_rate` -- so unlike the pass/rush formulas, there is
    no Stage-1 prediction to multiply the share by. This rebuilds the real
    team-week total directly from this table's own already-real, already-
    filled `rz_touches` column (summed across every player row for that
    team-week) and trails it the same way every other B2-equivalent
    baseline in this project does (`ewm_mean(span=6)`, matching
    `rz_touch_share_ewm_6`'s own window, `.shift(1)`'d so the target week's
    own outcome never leaks in, within-season only per this project's
    windowing convention). A team's first tracked week of a season has no
    prior week to trail, so `team_rz_touches_trailing_ewm_6` is null there
    -- the same cold-start gap every other trailing feature has.

    Real judgment call, not a certainty: `player_week_features`/
    `player_week_usage` are scoped to skill positions only (SPEC's
    `SKILL_POSITIONS`), so a real red-zone touch by a non-skill-position
    player (a fullback dive, a trick-play carry) is invisible here and
    slightly undercounts the true team total `interim/build.py` itself
    used when it computed the real share -- rare enough not to block this
    fix, but worth knowing if the numbers still look off after this."""
    return (
        table.group_by(["team", "season", "week"])
        .agg(pl.col("rz_touches").sum().alias("_team_rz_touches"))
        .sort(["team", "season", "week"])
        .with_columns(
            pl.col("_team_rz_touches")
            .ewm_mean(span=6)
            .shift(1)
            .over(["team", "season"])
            .alias("team_rz_touches_trailing_ewm_6")
        )
        .select("team", "season", "week", "team_rz_touches_trailing_ewm_6")
    )


def add_opportunity_baselines(table: pl.DataFrame) -> pl.DataFrame:
    """Two baselines per target, following this project's established B0/B2
    pattern (SPEC §12.3) at player grain. Precondition: `table` must already
    have `TARGET_COLUMNS` (`targets`, `carries`, `rz_touches`) as real,
    non-null columns -- i.e. it must be `build_opportunity_table`'s own
    output (which derives `rz_touches` itself), not a raw usage table.

    - `*_league_mean` (B0-equivalent, sanity floor): every player pooled by
      `position`, via `models.baselines.pooled_rolling_mean` -- a position-
      blind pool (RB and WR carries averaged together) would be meaningless,
      unlike Stage 1's single "TEAM_ENV" pool.
    - `*_b2_ewm_4` (the real bar): this player's own trailing `ewm_4` of the
      real raw count, `.shift(1)`'d so the target week's own outcome never
      leaks in -- same shape as every other B2 in this project (see
      `models.dst.add_dst_b2_ewm_4`, `models.team_environment
      .add_team_environment_baselines`).
    """
    with_league_means = table
    for target_column in TARGET_COLUMNS:
        with_league_means = pooled_rolling_mean(
            with_league_means, "position", target_column, f"{target_column}_league_mean"
        )

    sorted_table = with_league_means.sort(["player_id", "season", "week"])
    with_b2 = sorted_table
    for target_column in TARGET_COLUMNS:
        with_b2 = with_b2.with_columns(
            pl.col(target_column)
            .ewm_mean(span=4)
            .shift(1)
            .over(["player_id", "season"])
            .alias(f"{target_column}_b2_ewm_4")
        )
    return with_b2


def add_opportunity_blend(table: pl.DataFrame) -> pl.DataFrame:
    """A third predictor per target: a straight, equal-weight average of
    the arithmetic composition (`expected_*`) and the trailing-raw B2
    baseline (`*_b2_ewm_4`) -- the real bar the composition alone lost to
    on all three outputs (see docs/JOURNAL.md's Stage 2 entries). Not a
    trained blend (no fit weight, no second model) -- a cheap first probe
    of a specific hypothesis: the composition isn't losing because Stage
    1's team-volume signal is worthless, it's losing because multiplying
    two independently-noisy estimates together (a trailing share and
    Stage 1's own team-volume prediction, which itself doesn't beat
    `league_mean`) stacks more variance than a single already-integrated
    trailing average removes. If that's right, averaging the two -- not
    replacing one with the other -- should do better than either alone.

    Precondition: `table` must already have both `build_opportunity_table`'s
    `expected_*` columns and `add_opportunity_baselines`'s `*_b2_ewm_4`
    columns -- i.e. this runs after both, chained.

    Null, not a silent fallback to whichever input exists, when either
    side is missing (composition null for an ineligible position or a
    cold-start week; b2 null for a player's own first tracked week) -- a
    fair like-for-like blend needs both real inputs, not a value that's
    secretly just one of them relabelled."""
    return table.with_columns(
        [
            ((pl.col(composition_column) + pl.col(f"{target_column}_b2_ewm_4")) / 2).alias(
                f"{target_column}_blend"
            )
            for target_column, composition_column in _COMPOSITION_COLUMNS.items()
        ]
    )


__all__ = [
    "TARGET_COLUMNS",
    "add_opportunity_baselines",
    "add_opportunity_blend",
    "build_opportunity_table",
]
