"""The opponent feature block (SPEC.md §10.2 "Opponent"; task 1.9).

Maps `interim/defense_position_allowed.parquet`'s per-(defteam, season,
week, position_group) ridge-adjusted values (task 1.8) onto individual
player-week rows, based on each player's own position. Deliberately *not*
lag-shifted the way `features.usage`/`features.team_context` are: a
`defense_position_allowed` row for (defteam, season, week) is already
computed using only that defense's *prior* weeks (task 1.8's own
walk-forward design) -- it already represents "the opponent-adjusted
estimate as of right before this week," so a player's row for that same
week joins directly onto it, with no further shift.

A player's position determines which group(s) apply -- not a 1:1 mapping,
since a WR/TE only ever contribute one kind of play (receiving) while an
RB and QB each straddle two (SPEC's own position_group list splits
RB/QB into `_receiving`/`_rushing` and `_passing`/`_rushing` respectively):

    WR -> WR
    TE -> TE
    RB -> RB_receiving, RB_rushing
    QB -> QB_passing, QB_rushing

Column names follow SPEC §10.2's own `def_adj_epa_allowed_<group>`
template literally, lowercased (`def_adj_epa_allowed_wr`,
`def_adj_epa_allowed_rb_rushing`, ...). `def_n_plays_<group>` is carried
alongside each (SPEC §10.4: "Report n_plays alongside every estimate so
the UI can grey out low-confidence matchup grades") -- the *player's*
own copy of the same n_plays that grades their specific matchup.

A player's group(s) not relevant to their own position are always null,
not zero -- "opponent-adjusted rushing EPA allowed" simply doesn't apply
to a WR's own row.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.registry import FeatureSpec, register

SOURCE_TABLE = "defense_position_allowed"

_POSITION_TO_GROUPS = {
    "WR": ["WR"],
    "TE": ["TE"],
    "RB": ["RB_receiving", "RB_rushing"],
    "QB": ["QB_passing", "QB_rushing"],
}
ALL_POSITION_GROUPS = sorted({g for groups in _POSITION_TO_GROUPS.values() for g in groups})

_RATE_OUTCOME_ADJ_COLUMNS = {
    "adj_epa_allowed": "def_adj_epa_allowed",
    "adj_success_allowed": "def_adj_success_allowed",
    "adj_ypt_allowed": "def_adj_ypt_allowed",
    "adj_td_rate_allowed": "def_adj_td_rate_allowed",
}
_N_PLAYS_PREFIX = "def_n_plays"


def _position_group_pairs() -> pl.DataFrame:
    rows = [
        {"position": position, "position_group": group}
        for position, groups in _POSITION_TO_GROUPS.items()
        for group in groups
    ]
    return pl.DataFrame(rows)


def _team_opponent(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, season, week): that team's real opponent for a
    scheduled game. A bye week has no row -- callers join on this and
    byes naturally drop out."""
    home_side = schedule.select(
        "season", "week", pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent")
    )
    away_side = schedule.select(
        "season", "week", pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent")
    )
    return pl.concat([home_side, away_side], how="vertical_relaxed")


def _feature_column_name(raw_column: str, group: str) -> str:
    prefix = _RATE_OUTCOME_ADJ_COLUMNS.get(raw_column, _N_PLAYS_PREFIX)
    return f"{prefix}_{group.lower()}"


def add_opponent_features(
    grid: pl.DataFrame, schedule: pl.DataFrame, defense_position_allowed: pl.DataFrame
) -> pl.DataFrame:
    """`grid` needs `player_id`, `season`, `week`, `team`, `position`."""
    with_opponent = grid.join(_team_opponent(schedule), on=["team", "season", "week"], how="left")
    relevant = with_opponent.join(_position_group_pairs(), on="position", how="inner")

    value_columns = [*_RATE_OUTCOME_ADJ_COLUMNS, "n_plays"]
    long = relevant.join(
        defense_position_allowed.rename({"defteam": "opponent"}),
        on=["opponent", "season", "week", "position_group"],
        how="left",
    )

    pivoted = long.pivot(
        on="position_group",
        index=["player_id", "season", "week"],
        values=value_columns,
    )

    # A group absent from `defense_position_allowed` entirely (shouldn't
    # happen at real scale, but a narrow fixture can hit this) still gets
    # its column, all-null -- the schema must not depend on what happens
    # to be present in this particular input.
    expected_raw_columns = [
        f"{value_col}_{group}" for value_col in value_columns for group in ALL_POSITION_GROUPS
    ]
    missing = [c for c in expected_raw_columns if c not in pivoted.columns]
    if missing:
        pivoted = pivoted.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])

    renamed = pivoted.rename(
        {
            f"{value_col}_{group}": _feature_column_name(value_col, group)
            for value_col in value_columns
            for group in ALL_POSITION_GROUPS
        }
    )

    return grid.join(renamed, on=["player_id", "season", "week"], how="left")


def register_opponent_features(*, registry: dict[str, FeatureSpec] | None = None) -> None:
    """Registers every `def_adj_*_<group>`/`def_n_plays_<group>` column
    `add_opponent_features` can produce. Separated from that function
    (unlike `usage.build_usage_features`/`team_context
    .build_team_context_features`, which register while they compute)
    because these columns aren't windowed per-call the same way -- the
    full set is static and known up front from `_POSITION_TO_GROUPS`.
    """
    descriptions = {
        "adj_epa_allowed": "ridge-adjusted EPA allowed to the player's position group",
        "adj_success_allowed": "ridge-adjusted success rate allowed to the player's position group",
        "adj_ypt_allowed": "adjusted yards per touch allowed",
        "adj_td_rate_allowed": "adjusted TD rate allowed",
        "n_plays": "sample size behind the opponent-adjusted estimate",
    }
    for position, groups in _POSITION_TO_GROUPS.items():
        for group in groups:
            for raw_column, description in descriptions.items():
                register(
                    FeatureSpec(
                        name=_feature_column_name(raw_column, group),
                        description=description,
                        positions=[position],
                        window=None,
                        source_table=SOURCE_TABLE,
                        available_at_inference=True,
                        lag_weeks=1,
                    ),
                    registry=registry,
                )


def build_opponent_features(
    grid: pl.DataFrame,
    schedule: pl.DataFrame,
    defense_position_allowed: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    result = add_opponent_features(grid, schedule, defense_position_allowed)
    register_opponent_features(registry=registry)
    return result


__all__ = [
    "ALL_POSITION_GROUPS",
    "SOURCE_TABLE",
    "add_opponent_features",
    "build_opponent_features",
    "register_opponent_features",
]
