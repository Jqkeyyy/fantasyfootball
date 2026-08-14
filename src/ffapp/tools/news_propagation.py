"""Second-order news propagation (SPEC.md §14.8; task 2.10).

**SPEC's own "the valuable part":** "The headline 'RB1 is out' is on
every fantasy site within minutes. The automatically recomputed
projection for RB2, the team's revised run rate, and the FAAB bid for
the third-string back are not."

This module implements SPEC's own three-step cascade for a real
ruled-out player, reusing existing, already-validated machinery at
every step rather than re-deriving any of it:

1. **Recompute `teammate_vacated_target_share`/`.carry_share`**
   (`recompute_vacated_shares`) -- calls the exact same
   `features.team_context.add_vacated_shares` the official nflverse
   injury report already drives (task 1.7), fed one synthetic `Out` row
   (`synthetic_out_row`) instead of waiting for that official report to
   catch up. This *is* the mechanism SPEC names as the feature: an
   early, LLM-structured "ruled out" signal reacts through the identical
   real pipeline a Wednesday injury report reacts through days later.
2. **Re-run projections for affected teammates**
   (`propagate_ruled_out_player`) -- patches the recomputed vacated
   shares into a copy of `player_week_features.parquet`'s own real rows
   for just that one (team, week), then calls `models.predict
   .project_week` (task 1.18) unmodified, scoped to the team's real
   remaining skill-position players.
3. **Surface the handcuff** -- the same-position teammate with the
   highest recomputed projection is the real next-man-up; its recomputed
   `mean` is exactly the shape `tools.waivers.build_waiver_board`'s own
   `projection_by_player: dict[str, float]` parameter already expects
   (task 2.6), so feeding it in is the caller's own one-line integration,
   not a second waiver-board implementation here.

**Not built: "adjust the team's projected pass rate if the change is at
QB."** SPEC names this as its own fourth cascade step, but no existing
mechanism in this project computes a live pass-rate adjustment from a
single ruled-out QB, and this task's own literal TASKS.md acceptance bar
("a ruled-out RB1 automatically propagates to the backup's projection
and the waiver board") never mentions it either. Inventing a new
adjustment formula with no real data behind it would be exactly the
"guessing produces code that looks right and is silently wrong" failure
CLAUDE.md warns against -- a genuine, documented scope boundary, not an
oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.features.team_context import RULED_OUT_STATUS, add_vacated_shares
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models.predict import project_week

VACATED_COLUMNS = ["teammate_vacated_target_share", "teammate_vacated_carry_share"]


def synthetic_out_row(player_id: str, *, season: int, week: int) -> pl.DataFrame:
    """One injuries-shaped row marking `player_id` `Out` for `(season,
    week)` -- the exact real shape `features.team_context
    .add_vacated_shares` already reads from the official nflverse
    report (`player_id`, `season`, `week`, `report_status ==
    RULED_OUT_STATUS`), fed a synthetic early-signal event instead of a
    real official one.

    `season`/`week` are explicitly `Int32` -- every real nflverse-derived
    table in this project (`interim/injuries.parquet`,
    `features/player_week_features.parquet`, `interim/schedule.parquet`)
    uses `Int32` for both, but a bare Python `int` in a polars literal
    infers `Int64`. Caught live against real 2025 data, not assumed: an
    `Int64` synthetic row's own `season`/`week` upcasts the `diagonal_relaxed`
    concat's *entire* column to `Int64`, while `add_vacated_shares`'s own
    `join_asof(by=["player_id", "season"])` requires both sides to share
    exactly one dtype -- an unpinned literal breaks that real join with a
    dtype-mismatch error a same-Int64 fixture never surfaces.
    """
    return pl.DataFrame(
        {
            "player_id": [player_id],
            "season": [season],
            "week": [week],
            "report_status": [RULED_OUT_STATUS],
        },
        schema={
            "player_id": pl.String,
            "season": pl.Int32,
            "week": pl.Int32,
            "report_status": pl.String,
        },
    )


def recompute_vacated_shares(
    team_week_context: pl.DataFrame,
    injuries: pl.DataFrame,
    features: pl.DataFrame,
    *,
    ruled_out_player_id: str,
    season: int,
    week: int,
) -> pl.DataFrame:
    """The real `add_vacated_shares` (task 1.7), given the real
    official `injuries` table plus one synthetic `Out` row appended for
    `ruled_out_player_id`. `features` supplies the "usage_features"
    `add_vacated_shares` needs (`target_share_ewm_3`/`carry_share_ewm_3`)
    directly from `player_week_features.parquet`'s own already-built
    wide table (task 1.9) -- these are already real columns there, so
    this doesn't call `features.usage.build_usage_features` a second
    time over raw interim tables.

    `injuries` (real `interim/injuries.parquet`, task 1.4) carries real
    columns `synthetic_out_row` doesn't set (`team`, `practice_status`,
    `report_primary_injury`, `date_modified`, ...) --
    `how="diagonal_relaxed"` unions the two schemas and null-fills what
    the synthetic row doesn't have, rather than requiring an identical
    column set the way `vertical_relaxed` does. `add_vacated_shares`
    itself only ever reads `player_id`/`season`/`week`/`report_status`
    (task 1.7's own real implementation), so the nulls are never touched
    downstream. Caught live against real 2025 data, not assumed: a
    same-shaped fixture in this module's own tests never exercised the
    real injuries table's real wider schema.
    """
    synthetic = synthetic_out_row(ruled_out_player_id, season=season, week=week)
    patched_injuries = pl.concat([injuries, synthetic], how="diagonal_relaxed")
    usage_slice = features.select(
        "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
    )
    return add_vacated_shares(team_week_context, patched_injuries, usage_slice)


def patch_vacated_shares(
    features: pl.DataFrame,
    recomputed_team_week_context: pl.DataFrame,
    *,
    team: str,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Overwrites `teammate_vacated_target_share`/`.carry_share` for
    `team`'s own real rows at `(season, week)` with the recomputed
    values from `recompute_vacated_shares` -- every other team, week, or
    column in `features` is untouched. A `(team, season, week)` missing
    from `recomputed_team_week_context` (should not happen for a real
    scheduled team-week) returns `features` unchanged rather than
    guessing a value.
    """
    recomputed_row = recomputed_team_week_context.filter(
        (pl.col("team") == team) & (pl.col("season") == season) & (pl.col("week") == week)
    )
    if recomputed_row.is_empty():
        return features

    values = recomputed_row.row(0, named=True)
    mask = (pl.col("team") == team) & (pl.col("season") == season) & (pl.col("week") == week)
    return features.with_columns(
        pl.when(mask)
        .then(pl.lit(values["teammate_vacated_target_share"]))
        .otherwise(pl.col("teammate_vacated_target_share"))
        .alias("teammate_vacated_target_share"),
        pl.when(mask)
        .then(pl.lit(values["teammate_vacated_carry_share"]))
        .otherwise(pl.col("teammate_vacated_carry_share"))
        .alias("teammate_vacated_carry_share"),
    )


def affected_teammates(
    features: pl.DataFrame,
    *,
    ruled_out_player_id: str,
    team: str,
    season: int,
    week: int,
) -> pl.DataFrame:
    """The real remaining skill-position teammates on `team` at
    `(season, week)` -- every real row for that team-week except the
    ruled-out player's own, scoped to `SKILL_POSITIONS` (the only
    positions `models.predict.project_week` covers)."""
    return features.filter(
        (pl.col("team") == team)
        & (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("player_id") != ruled_out_player_id)
        & pl.col("position").is_in(SKILL_POSITIONS)
    )


@dataclass(frozen=True)
class PropagationResult:
    """`patched_features` is the full table with the recomputed vacated
    shares applied (a caller who wants to re-run anything else off the
    same patch can reuse it directly, not just this module's own
    re-projection). `reprojections` is `models.predict.project_week`'s
    real output, scoped to `affected_teammates`, with `position` joined
    back on (project_week's own output schema carries no position
    column). `handcuff_player_id`/`.handcuff_projection_ppg` are the
    real same-position teammate with the highest recomputed `mean` --
    `None` when no same-position teammate exists on this real roster
    that week, not a guessed value."""

    patched_features: pl.DataFrame
    reprojections: pl.DataFrame
    handcuff_player_id: str | None
    handcuff_projection_ppg: float | None


def propagate_ruled_out_player(
    features: pl.DataFrame,
    team_week_context: pl.DataFrame,
    injuries: pl.DataFrame,
    *,
    ruled_out_player_id: str,
    ruled_out_position: str,
    team: str,
    season: int,
    week: int,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings,
    code_version: str | None,
    now: datetime,
) -> PropagationResult:
    """The real end-to-end cascade: recompute vacated shares for `team`
    with `ruled_out_player_id` treated as `Out`, patch them into
    `features`, and re-run `models.predict.project_week` (unmodified,
    the same function `ffapp project` itself calls) on the patched
    table, scoped to the team's own real remaining skill-position
    players. The handcuff is the same-position teammate whose
    recomputed projection is now highest -- SPEC's own literal
    "next-man-up," ready to feed into `tools.waivers.build_waiver_board`
    `projection_by_player` (task 2.6) unmodified.
    """
    recomputed_context = recompute_vacated_shares(
        team_week_context,
        injuries,
        features,
        ruled_out_player_id=ruled_out_player_id,
        season=season,
        week=week,
    )
    patched = patch_vacated_shares(
        features, recomputed_context, team=team, season=season, week=week
    )
    teammates = affected_teammates(
        patched, ruled_out_player_id=ruled_out_player_id, team=team, season=season, week=week
    )

    all_projections = project_week(
        patched,
        season,
        week,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params,
        code_version=code_version,
        now=now,
    )
    reprojections = all_projections.join(
        teammates.select("player_id", "position"), on="player_id", how="inner"
    )

    handcuff_id: str | None = None
    handcuff_ppg: float | None = None
    same_position = reprojections.filter(pl.col("position") == ruled_out_position).sort(
        "mean", descending=True
    )
    if not same_position.is_empty():
        top = same_position.row(0, named=True)
        handcuff_id = top["player_id"]
        handcuff_ppg = top["mean"]

    return PropagationResult(
        patched_features=patched,
        reprojections=reprojections,
        handcuff_player_id=handcuff_id,
        handcuff_projection_ppg=handcuff_ppg,
    )


__all__ = [
    "VACATED_COLUMNS",
    "PropagationResult",
    "affected_teammates",
    "patch_vacated_shares",
    "propagate_ruled_out_player",
    "recompute_vacated_shares",
    "synthetic_out_row",
]
