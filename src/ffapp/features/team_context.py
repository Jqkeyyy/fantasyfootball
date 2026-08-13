"""The team context feature block (SPEC.md §10.2 "Team context"; task 1.7).

Windowing here mirrors `features/usage.py`'s own generic primitives
(within-season only, trailing through and including each row's own week,
`lag_weeks=1` on every registered feature -- see that module's docstring
for the full reasoning) but grouped by `team` instead of `player_id`, so
it isn't reused directly from there.

`proe`/`neutral_pace_sec` are real raw columns of `interim/team_week_context
.parquet` (task 1.7's own `interim.build.add_proe`/`.add_neutral_pace`).
`ol_continuity` and the two `teammate_vacated_*` features are *not* part of
that table's declared schema (SPEC §6.2 doesn't list them for
`team_week_context.parquet` -- they only appear in §10.2's feature
catalogue), and are computed entirely here instead of in `interim/build.py`:

- `ol_continuity` needs `snap_counts` (a source `team_week_context` never
  takes), so it has no natural home in that table.
- `teammate_vacated_target_share`/`.carry_share` need each *ruled-out*
  player's own recent target_share/carry_share -- that is, an
  *already-windowed* feature from `features.usage`, not a raw interim
  value. Computing it in `interim/build.py` would mean that module
  reaching forward into the features layer's own output, inverting the
  project's established interim -> features dependency direction. This
  module takes `usage_features` (the *output* of
  `usage.build_usage_features`) as an input specifically so the
  dependency stays one-directional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ffapp.features.registry import FeatureSpec, register

SOURCE_TABLE = "team_week_context"

RULED_OUT_STATUS = "Out"

# task 1.9: which of this module's registered features are "current week"
# facts (known before that week's own kickoff, joined onto a target week
# directly) versus trailing/windowed values (through-and-including their
# own week, needing task 1.9's own one-week shift onto a target week).
# Every OTHER feature this module registers is windowed (proe_ewm_5,
# neutral_pace_ewm_8, ...) and needs the shift; these four don't, despite
# sharing the same `source_table` -- `features.build` must not lump them
# in with the windowed ones, or it silently staples last week's Vegas
# line and injury news onto this week's row.
CURRENT_WEEK_COLUMNS = (
    "implied_team_total",
    "spread",
    "teammate_vacated_target_share",
    "teammate_vacated_carry_share",
)

# task 1.7 ol_continuity: the "starting 5" per team-game is the top
# offense_snaps player at each of these OL slots, from real snap_counts
# position codes. Rare combined codes ("G/OT", "C/G", ...) are excluded --
# a handful of versatile backups aren't worth chasing for this signal.
_OL_SLOT_COUNTS = {"C": 1, "G": 2, "T": 2}
_OL_STARTING_FIVE_SIZE = 5


def _sort(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["team", "season", "week"])


def ewm(df: pl.DataFrame, value_col: str, span: int, out_col: str) -> pl.DataFrame:
    """Exponentially weighted mean, span `span`, within-season, trailing
    through (and including) each row's own week -- team-grouped version of
    `features.usage.ewm`; see that module's docstring for why windows
    reset at a season boundary."""
    return _sort(df).with_columns(
        pl.col(value_col).ewm_mean(span=span).over(["team", "season"]).alias(out_col)
    )


def _starting_ol_by_game(snap_counts: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, week, team, pfr_player_id) for that game's
    "starting 5" OL: the top `offense_snaps` player(s) at each of
    C(1)/G(2)/T(2) -- confirmed as the definition to use."""
    ol = snap_counts.filter(pl.col("position").is_in(list(_OL_SLOT_COUNTS)))
    with_rank_and_slot = ol.with_columns(
        pl.col("offense_snaps")
        .rank(method="ordinal", descending=True)
        .over(["season", "week", "team", "position"])
        .alias("_rank"),
        pl.when(pl.col("position") == "C")
        .then(_OL_SLOT_COUNTS["C"])
        .when(pl.col("position") == "G")
        .then(_OL_SLOT_COUNTS["G"])
        .when(pl.col("position") == "T")
        .then(_OL_SLOT_COUNTS["T"])
        .otherwise(None)
        .alias("_slot_n"),
    )
    return with_rank_and_slot.filter(pl.col("_rank") <= pl.col("_slot_n")).select(
        "season", "week", "team", "pfr_player_id"
    )


def ol_continuity_raw(snap_counts: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, season, week): `ol_continuity_raw`, the fraction
    (0.0-1.0) of that game's "starting 5" OL who were also in the
    immediately preceding *in-season* game's starting 5. Null for a
    team's first tracked week of a season -- no prior week to compare to,
    same convention as `features.usage.weeks_in_current_role`.

    Public (not `_`-prefixed): task 2.7's DST model (`features.dst`) is a
    second real caller needing this same team-level signal for the
    *opponent* side, without pulling in this module's own
    `build_team_context_features` (which also needs `usage_features`/
    `injuries` for `teammate_vacated_*`, irrelevant to DST) -- promoted
    rather than duplicated, matching `sim.lineup.slot_instances`' and
    `sim.season.to_projection`/`to_marginal`'s own promotion this same
    session."""
    starters = _starting_ol_by_game(snap_counts)
    sets_by_game = _sort(
        starters.group_by(["season", "week", "team"]).agg(
            pl.col("pfr_player_id").alias("_starters")
        )
    )
    with_prev = sets_by_game.with_columns(
        pl.col("_starters").shift(1).over(["team", "season"]).alias("_prev_starters")
    )

    def _overlap_ratio(s: dict[str, object]) -> float | None:
        prev = s["_prev_starters"]
        if not isinstance(prev, list):
            return None
        cur = s["_starters"]
        cur_list = cur if isinstance(cur, list) else []
        return len(set(cur_list) & set(prev)) / _OL_STARTING_FIVE_SIZE

    return with_prev.with_columns(
        pl.struct(["_starters", "_prev_starters"])
        .map_elements(_overlap_ratio, return_dtype=pl.Float64)
        .alias("ol_continuity_raw")
    ).select("season", "week", "team", "ol_continuity_raw")


@dataclass(frozen=True)
class _WindowedFeature:
    raw_column: str
    name_base: str
    description: str
    windows: list[str] = field(default_factory=list)


_WINDOWED_FEATURES = [
    _WindowedFeature("plays", "plays_per_game", "offensive plays", ["ewm_5"]),
    _WindowedFeature(
        "neutral_pace_sec",
        "neutral_pace",
        "sec/play, score within 7, Q1-Q3",
        ["ewm_8"],
    ),
    _WindowedFeature("proe", "proe", "pass rate over expectation", ["ewm_5"]),
    _WindowedFeature("epa_per_play_off", "team_epa_off", "offensive quality", ["ewm_8"]),
    _WindowedFeature("success_rate_off", "team_success_off", "offensive quality", ["ewm_8"]),
    _WindowedFeature(
        "ol_continuity_raw", "ol_continuity", "starting OL snaps with same 5", ["ewm_5"]
    ),
]


def build_team_context_features(
    team_week_context: pl.DataFrame,
    snap_counts: pl.DataFrame,
    injuries: pl.DataFrame,
    usage_features: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Assemble every SPEC §10.2 "Team context" feature and register each
    one's `FeatureSpec` (task 1.5's registry). Every registered feature
    declares `lag_weeks=1` and `available_at_inference=True` -- none of
    these have an in-season availability gap the way route participation
    does (SPEC §10.5).
    """
    result = team_week_context.join(
        ol_continuity_raw(snap_counts), on=["team", "season", "week"], how="left"
    )

    for feature in _WINDOWED_FEATURES:
        for window in feature.windows:
            out_col = f"{feature.name_base}_{window}"
            result = ewm(result, feature.raw_column, int(window.removeprefix("ewm_")), out_col)
            register(
                FeatureSpec(
                    name=out_col,
                    description=feature.description,
                    positions=[],
                    window=window,
                    source_table=SOURCE_TABLE,
                    available_at_inference=True,
                    lag_weeks=1,
                ),
                registry=registry,
            )

    # implied_team_total/spread: "current week", no smoothing -- straight
    # passthroughs of team_week_context's own already-current-week values.
    result = result.rename({"implied_total": "implied_team_total"})
    for name, description in [
        ("implied_team_total", "from Vegas total and spread"),
        ("spread", "signed, team perspective"),
    ]:
        register(
            FeatureSpec(
                name=name,
                description=description,
                positions=[],
                window=None,
                source_table=SOURCE_TABLE,
                available_at_inference=True,
                lag_weeks=1,
            ),
            registry=registry,
        )

    result = add_vacated_shares(result, injuries, usage_features)
    for name, description in [
        ("teammate_vacated_target_share", "sum of target_share of teammates ruled Out"),
        ("teammate_vacated_carry_share", "as above for carries"),
    ]:
        register(
            FeatureSpec(
                name=name,
                description=description,
                positions=[],
                window=None,
                source_table=SOURCE_TABLE,
                available_at_inference=True,
                lag_weeks=1,
            ),
            registry=registry,
        )

    return result


def add_vacated_shares(
    team_week_context: pl.DataFrame,
    injuries: pl.DataFrame,
    usage_features: pl.DataFrame,
) -> pl.DataFrame:
    """`teammate_vacated_target_share`/`.carry_share`: per (team, season,
    week), the sum of `target_share_ewm_3`/`carry_share_ewm_3` (each
    ruled-out player's own recent trailing role -- their *current* raw
    share is meaningless, since they didn't play) over players on that
    team with `report_status == "Out"` that week.

    A player who is genuinely `Out` has no `player_week_usage` row for
    that week at all (nflverse's `player_stats` only has rows for players
    who recorded a stat line -- confirmed live, an exact
    `(player_id, season, week)` join against `usage_features` drops every
    real `Out` player, since their own week never has a usage row).
    `join_asof(..., strategy="backward")` instead looks up their *most
    recent prior* in-season week's already-windowed share -- the role
    established before they got hurt, which is what "vacated" means.
    `available_shares` excludes any week a player was ruled `Out` before
    the asof lookup, so a stray same-week row can never be matched as if
    it were their established role.

    `usage_features` is `features.usage.build_usage_features`'s own
    output (already carries `team`, `target_share_ewm_3`,
    `carry_share_ewm_3` per player-week) -- this function doesn't
    recompute any windowing itself, only looks up and sums an
    already-windowed value over the ruled-out subset.
    """
    out_players = injuries.filter(pl.col("report_status") == RULED_OUT_STATUS).select(
        "player_id", "season", "week"
    )
    shares = usage_features.select(
        "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
    )
    available_shares = shares.join(out_players, on=["player_id", "season", "week"], how="anti")

    matched = out_players.sort(["player_id", "season", "week"]).join_asof(
        available_shares.sort(["player_id", "season", "week"]),
        on="week",
        by=["player_id", "season"],
        strategy="backward",
        check_sortedness=False,  # both sides are already sorted by (by..., on) above
    )
    vacated = (
        matched.drop_nulls("team")
        .group_by(["season", "week", "team"])
        .agg(
            pl.col("target_share_ewm_3").fill_null(0).sum().alias("teammate_vacated_target_share"),
            pl.col("carry_share_ewm_3").fill_null(0).sum().alias("teammate_vacated_carry_share"),
        )
    )
    return team_week_context.join(vacated, on=["team", "season", "week"], how="left").with_columns(
        pl.col("teammate_vacated_target_share").fill_null(0.0),
        pl.col("teammate_vacated_carry_share").fill_null(0.0),
    )


__all__ = [
    "RULED_OUT_STATUS",
    "SOURCE_TABLE",
    "add_vacated_shares",
    "build_team_context_features",
    "ewm",
    "ol_continuity_raw",
]
