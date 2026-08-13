"""Weekly rankings page logic (SPEC.md §14.1, §15; task 1.19).

Pure, pytest-testable functions only -- enrichment, grading, filtering.
The real Streamlit page (`app/pages/2_Weekly_Rankings.py`) is thin glue
on top, matching `draft_board_page.py`'s own precedent (task 0.13): reads
the pre-built `outputs/projections.parquet` (task 1.18) rather than
recomputing anything on page load (SPEC §15's own "fast to load...
nothing trained on page load" constraint).

**Floor/ceiling is "the differentiating output of this system"** (SPEC's
own words) -- `q10`/`q90` are exposed here as `floor`/`ceiling` and the
page renders them as a visible range, never a hidden column.
**`matchup_grade` is deliberately de-emphasized**, per CLAUDE.md's own
standing rule (never displayed more prominently than the numbers that
actually differentiate this system) -- a plain letter grade plus its own
`n_plays_behind_matchup_grade` sample size, not a colour-coded badge.

**A player's own real matchup spans one or two `POSITION_TO_GROUPS`
columns** (task 1.9/`features.opponent`): WR/TE have one, RB/QB have two
(rushing + receiving/passing), both already sitting on
`player_week_features.parquet` (task 1.9) -- no second join to
`defense_position_allowed.parquet` needed. Combined into one
`matchup_grade` via an `n_plays`-weighted average, not a plain mean: a
group with more real sampled plays behind it should count for more, the
same "trust the estimate with more support behind it" principle
`n_plays` itself exists to communicate.

`matchup_grade` letters are quintile buckets of that combined value
*within the real (season, week, position) cohort* -- SPEC names no exact
formula for "grade," so this is a documented judgment call: `A` = the
easiest 20% of real matchups that week for that position, `F` = the
hardest. A row with no relevant opponent-adjustment data at all (a
genuine early-season gap, task 1.8's own null pattern) gets an honestly
null grade, not a guessed one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from ffapp.features.opponent import POSITION_TO_GROUPS, team_opponent

_GRADE_LABELS = ["F", "D", "C", "B", "A"]  # ascending value -> ascending grade


class ProjectionsNotBuiltError(Exception):
    """No projections.parquet exists yet for this (season, week)."""


def load_projections(path: Path) -> pl.DataFrame:
    """Read the pre-built projections parquet (task 1.18's own
    `outputs/projections.parquet`). Raises `ProjectionsNotBuiltError`
    naming the fix (`ffapp project`), matching `draft_board_page
    .DraftBoardNotBuiltError`'s own convention."""
    if not path.exists():
        raise ProjectionsNotBuiltError(
            f"No projections found at {path}. Run `ffapp project --week N` to build one first."
        )
    return pl.read_parquet(path)


def _combined_matchup(row: dict[str, object], position: str) -> tuple[float | None, int | None]:
    """This row's own real matchup value/sample size, combining every
    `POSITION_TO_GROUPS[position]` column present -- see module docstring
    for the n_plays-weighted-average rationale."""
    total_n = 0
    weighted_sum = 0.0
    for group in POSITION_TO_GROUPS.get(position, []):
        epa_raw = row.get(f"def_adj_epa_allowed_{group.lower()}")
        n_plays_raw = row.get(f"def_n_plays_{group.lower()}")
        if epa_raw is None or n_plays_raw is None:
            continue
        epa, n_plays = float(epa_raw), int(n_plays_raw)  # type: ignore[arg-type,call-overload]
        if n_plays <= 0:
            continue
        weighted_sum += epa * n_plays
        total_n += n_plays
    if total_n == 0:
        return None, None
    return weighted_sum / total_n, total_n


def add_matchup_grade(df: pl.DataFrame) -> pl.DataFrame:
    """Adds `_matchup_epa` (the combined real value, kept for testability
    and for the grading step below), `matchup_grade`, and
    `n_plays_behind_matchup_grade`. Grading is scoped per real `position`
    present in `df` -- a WR's matchup is never ranked against a QB's."""
    rows = df.to_dicts()
    matchup_epa: list[float | None] = []
    n_plays_behind: list[int | None] = []
    for row in rows:
        epa, n_plays = _combined_matchup(row, str(row["position"]))
        matchup_epa.append(epa)
        n_plays_behind.append(n_plays)

    result = df.with_columns(
        pl.Series("_matchup_epa", matchup_epa, dtype=pl.Float64),
        pl.Series("n_plays_behind_matchup_grade", n_plays_behind, dtype=pl.Int64),
    )

    grades: list[str | None] = [None] * result.height
    for position in result["position"].unique().to_list():
        position_mask = (result["position"] == position) & result["_matchup_epa"].is_not_null()
        indices = np.where(position_mask.to_numpy())[0]
        if len(indices) == 0:
            continue
        values = result["_matchup_epa"].to_numpy()[indices]
        # Rank ascending (lowest real value -> hardest matchup -> "F"),
        # bucketed into 5 equal-width percentile groups. Average-rank ties
        # (scipy's own "average" method) so two real defenses tied on the
        # same value always land in the same bucket -- plain argsort would
        # break the tie arbitrarily by array position instead.
        ranks = rankdata(values, method="average") - 1  # 0-based
        percentile = ranks / max(len(values) - 1, 1)
        bucket = np.clip((percentile * 5).astype(int), 0, 4)
        for i, idx in enumerate(indices):
            grades[idx] = _GRADE_LABELS[bucket[i]]

    return result.with_columns(pl.Series("matchup_grade", grades, dtype=pl.String))


def build_weekly_rankings(
    projections: pl.DataFrame,
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    players_dim: pl.DataFrame,
    *,
    season: int,
    week: int,
    my_roster_ids: set[str] | None = None,
    rostered_ids: set[str] | None = None,
) -> pl.DataFrame:
    """SPEC §14.1's own output table, assembled from `projections`
    (task 1.18), `features` (task 1.9's wide table, for `position`/`team`
    and the opponent-adjustment columns `add_matchup_grade` needs),
    `schedule` (`opponent`, via `features.opponent.team_opponent`), and
    `players_dim` (`player_name`). Sorted by `proj_mean` descending --
    SPEC's own stated default. Inner-joins `projections` onto `features`:
    every real projection row comes from that exact `(season, week)`
    slice of `features` by construction (task 1.18's own `project_week`),
    so this can never silently drop a real projection the way an
    unrelated join could (CLAUDE.md rule 4).

    `my_roster_ids`/`rostered_ids` are canonical `player_id` sets, already
    resolved by the caller (the real Streamlit page, via Sleeper +
    `players_dim`'s own `sleeper_id` column) -- this module stays
    decoupled from any live Sleeper fetch, matching every other pure
    logic module in `app/`.
    """
    proj = projections.filter((pl.col("season") == season) & (pl.col("week") == week))
    feature_columns = [f"def_adj_epa_allowed_{g.lower()}" for g in POSITION_TO_GROUPS_FLAT()] + [
        f"def_n_plays_{g.lower()}" for g in POSITION_TO_GROUPS_FLAT()
    ]
    feat = features.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "player_id", "team", "position", *feature_columns
    )
    merged = proj.join(feat, on="player_id", how="inner")

    opponents = (
        team_opponent(schedule)
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("team", "opponent")
    )
    merged = merged.join(opponents, on="team", how="left")

    graded = add_matchup_grade(merged)

    named = graded.join(
        players_dim.select("player_id", pl.col("full_name").alias("player_name")),
        on="player_id",
        how="left",
    )

    my_roster_ids = my_roster_ids or set()
    rostered_ids = rostered_ids or set()
    owner_status = (
        pl.when(pl.col("player_id").is_in(list(my_roster_ids)))
        .then(pl.lit("my_roster"))
        .when(pl.col("player_id").is_in(list(rostered_ids)))
        .then(pl.lit("rostered_elsewhere"))
        .otherwise(pl.lit("free_agent"))
        .alias("owner_status")
    )

    result = named.with_columns(
        owner_status,
        pl.col("mean").alias("proj_mean"),
        pl.col("q10").alias("floor"),
        pl.col("q50").alias("median"),
        pl.col("q90").alias("ceiling"),
    )

    return result.select(
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "p_active",
        "proj_mean",
        "floor",
        "median",
        "ceiling",
        "matchup_grade",
        "n_plays_behind_matchup_grade",
        "owner_status",
    ).sort("proj_mean", descending=True)


def POSITION_TO_GROUPS_FLAT() -> list[str]:  # noqa: N802 -- reads as a constant at call sites
    """Every real `position_group` string across all positions -- avoids
    hardcoding the flattened list a second time (`features.opponent
    .ALL_POSITION_GROUPS` already exists but the local column-name build
    here reads clearer with the plain function form)."""
    return sorted({g for groups in POSITION_TO_GROUPS.values() for g in groups})


def filter_rankings(
    df: pl.DataFrame,
    *,
    positions: list[str] | None = None,
    availability: str | None = None,
) -> pl.DataFrame:
    """SPEC §14.1: "filterable by position and by availability (all / my
    roster / free agents / rostered elsewhere)". `None`/empty/`"all"`
    means no filter for that dimension -- an empty multiselect in the UI
    should show everything, matching `draft_board_page.filter_board`'s
    own precedent."""
    filtered = df
    if positions:
        filtered = filtered.filter(pl.col("position").is_in(positions))
    if availability and availability != "all":
        filtered = filtered.filter(pl.col("owner_status") == availability)
    return filtered


__all__ = [
    "ProjectionsNotBuiltError",
    "add_matchup_grade",
    "build_weekly_rankings",
    "filter_rankings",
    "load_projections",
]
