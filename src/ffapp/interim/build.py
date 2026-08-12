"""Canonical interim tables built from real business logic across multiple
nflverse sources (SPEC.md §6.2; task 1.1).

Kept out of `ingest/nflverse.py` deliberately -- CLAUDE.md: no business
logic in ingest/ beyond schema normalisation. Real joins/aggregations
across sources (snap counts, play-by-play, player stats) live here,
matching this project's established precedent (`projections/aggregate.py`
vs. `ingest/rankings.py`; `scoring/stats.py` vs. `ingest/nflverse.py`).

Several SPEC §6.2 columns are deliberately left null here rather than
guessed at -- each is explicitly a *later* task's own deliverable, and
guessing wrong now would be a silently-wrong number, not an honestly
missing one:

- `player_week_usage.route_participation` -- external data gap (NGS
  participation ended mid-2023; FTN's replacement only publishes
  post-season). See SPEC §10.5.
- `player_week_usage.xfp` -- task 1.2 (ffopportunity ingestion).
- `team_week_context.proe`, `.neutral_pace_sec`, `.implied_total`,
  `.spread` -- task 1.7 ("Team context features") for the first two (both
  need real modelling -- an expected-pass-rate baseline, careful
  play-sequencing to measure real elapsed time -- not mechanical
  aggregation); `.implied_total`/`.spread` need `spread_line`'s sign
  convention verified first, which is explicitly task 1.3's job (SPEC:
  "positive spread = home favoured (verify sign at ingest and
  document)") -- not yet verified, so not guessed at here.
- `schedule.kickoff_utc`, `.home_implied_total`, `.away_implied_total` --
  task 1.3, same sign-convention dependency plus a per-stadium timezone
  lookup (`config/stadiums.csv`, also 1.3's own deliverable).
- `defense_position_allowed.adj_*` -- task 1.8 ("Opponent adjustment");
  needs the real ridge-regression/shrinkage model, not raw allowed rates.
- `defense_position_allowed`'s position groups collapse `WR_perimeter`/
  `WR_slot` into one undifferentiated `WR` -- splitting by alignment needs
  the same missing NGS/FTN charting data as the route_participation gap
  above.
"""

from __future__ import annotations

import polars as pl

from ffapp.scoring.stats import build_stat_frame

_SCRIMMAGE_PLAY_TYPES = ("pass", "run")
RED_ZONE_YARDLINE = 20
GOAL_ZONE_YARDLINE = 5

# (position, play_type) -> SPEC §6.2's defense_position_allowed group.
# WR_perimeter/WR_slot collapsed to "WR" -- see module docstring.
_POSITION_GROUP_MAP = {
    ("QB", "pass"): "QB_passing",
    ("QB", "run"): "QB_rushing",
    ("RB", "run"): "RB_rushing",
    ("RB", "pass"): "RB_receiving",
    ("TE", "pass"): "TE",
    ("WR", "pass"): "WR",
}


def build_player_week_stats(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    """`interim/player_week_stats.parquet` (SPEC §6.2) -- reuses
    `scoring.stats.build_stat_frame` (task 0.5's golden-test assembly)
    directly rather than re-deriving the same DST play-by-play logic a
    second time; that module's own docstring already named this task as
    its intended successor.
    """
    return build_stat_frame(player_stats, team_stats, schedules, pbp)


def _scrimmage_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    return pbp.filter(pl.col("play_type").is_in(_SCRIMMAGE_PLAY_TYPES))


def build_team_week_context(pbp: pl.DataFrame) -> pl.DataFrame:
    """`interim/team_week_context.parquet` (SPEC §6.2), basic version:
    `plays`, raw `pass_rate`, `epa_per_play_off`, `success_rate_off` --
    all direct per-play aggregation from real nflverse `epa`/`success`
    columns, no modelling needed. `proe`/`neutral_pace_sec`/
    `implied_total`/`spread` stay null -- see module docstring.
    """
    plays = _scrimmage_plays(pbp)
    return (
        plays.group_by(["season", "week", "posteam"])
        .agg(
            pl.len().alias("plays"),
            (pl.col("play_type") == "pass").mean().alias("pass_rate"),
            pl.col("epa").mean().alias("epa_per_play_off"),
            pl.col("success").mean().alias("success_rate_off"),
        )
        .rename({"posteam": "team"})
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("neutral_pace_sec"),
            pl.lit(None, dtype=pl.Float64).alias("proe"),
            pl.lit(None, dtype=pl.Float64).alias("implied_total"),
            pl.lit(None, dtype=pl.Float64).alias("spread"),
        )
        .select(
            "team",
            "season",
            "week",
            "plays",
            "neutral_pace_sec",
            "pass_rate",
            "proe",
            "epa_per_play_off",
            "success_rate_off",
            "implied_total",
            "spread",
        )
    )


def _player_position_by_season(player_stats: pl.DataFrame) -> pl.DataFrame:
    """(player_id, season) -> position, from player_stats' own weekly rows
    -- whichever position appears first for that player-season (a player's
    position essentially never changes mid-season)."""
    return (
        player_stats.sort(["player_id", "season", "week"])
        .group_by(["player_id", "season"], maintain_order=True)
        .agg(pl.col("position").first())
    )


def build_defense_position_allowed(pbp: pl.DataFrame, player_stats: pl.DataFrame) -> pl.DataFrame:
    """`interim/defense_position_allowed.parquet` (SPEC §6.2), basic
    version: `n_plays` per (defteam, season, week, position_group), from
    real play-by-play. `adj_*` columns stay null -- see module docstring.
    """
    positions = _player_position_by_season(player_stats)
    plays = _scrimmage_plays(pbp)

    pass_plays = (
        plays.filter((pl.col("play_type") == "pass") & pl.col("receiver_player_id").is_not_null())
        .join(
            positions.rename({"player_id": "receiver_player_id"}),
            on=["receiver_player_id", "season"],
            how="left",
        )
        .with_columns(pl.lit("pass").alias("_play_kind"))
    )
    rush_plays = (
        plays.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null())
        .join(
            positions.rename({"player_id": "rusher_player_id"}),
            on=["rusher_player_id", "season"],
            how="left",
        )
        .with_columns(pl.lit("run").alias("_play_kind"))
    )

    combined = pl.concat(
        [
            pass_plays.select("season", "week", "defteam", "position", "_play_kind"),
            rush_plays.select("season", "week", "defteam", "position", "_play_kind"),
        ],
        how="vertical_relaxed",
    )

    with_group = combined.with_columns(
        pl.struct(["position", "_play_kind"])
        .map_elements(
            lambda s: _POSITION_GROUP_MAP.get((s["position"], s["_play_kind"])),
            return_dtype=pl.Utf8,
        )
        .alias("position_group")
    ).filter(pl.col("position_group").is_not_null())

    return (
        with_group.group_by(["defteam", "season", "week", "position_group"])
        .agg(pl.len().alias("n_plays"))
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("adj_epa_allowed"),
            pl.lit(None, dtype=pl.Float64).alias("adj_success_allowed"),
            pl.lit(None, dtype=pl.Float64).alias("adj_ypt_allowed"),
            pl.lit(None, dtype=pl.Float64).alias("adj_td_rate_allowed"),
        )
        .select(
            "defteam",
            "season",
            "week",
            "position_group",
            "adj_epa_allowed",
            "adj_success_allowed",
            "adj_ypt_allowed",
            "adj_td_rate_allowed",
            "n_plays",
        )
    )


def _snap_counts_by_player_id(snap_counts: pl.DataFrame, players_dim: pl.DataFrame) -> pl.DataFrame:
    """Snap counts are PFR-sourced (`pfr_player_id`, e.g. `"BrowSp00"`),
    not keyed by gsis_id like everything else here -- resolved via
    `players_dim`'s own `pfr_id` column (task 0.3's crosswalk). A
    `pfr_id` matched to more than one crosswalk row keeps the first
    (rare; not something this table's accuracy depends on).
    """
    pfr_to_player_id = (
        players_dim.filter(pl.col("pfr_id").is_not_null())
        .select("pfr_id", "player_id")
        .unique(subset=["pfr_id"], keep="first")
    )
    return snap_counts.join(
        pfr_to_player_id.rename({"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="left"
    ).select(
        "player_id",
        "season",
        "week",
        "offense_snaps",
        pl.col("offense_pct").alias("offense_snap_pct"),
    )


def _red_zone_touch_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (player_id, season, week): `rz_targets`, `rz_carries`,
    `gz_carries` -- real play-by-play counts, not estimated."""
    plays = _scrimmage_plays(pbp)

    rz_targets = (
        plays.filter(
            (pl.col("play_type") == "pass")
            & (pl.col("yardline_100") <= RED_ZONE_YARDLINE)
            & pl.col("receiver_player_id").is_not_null()
        )
        .group_by(["season", "week", pl.col("receiver_player_id").alias("player_id")])
        .agg(pl.len().alias("rz_targets"))
    )
    rz_carries = (
        plays.filter(
            (pl.col("play_type") == "run")
            & (pl.col("yardline_100") <= RED_ZONE_YARDLINE)
            & pl.col("rusher_player_id").is_not_null()
        )
        .group_by(["season", "week", pl.col("rusher_player_id").alias("player_id")])
        .agg(pl.len().alias("rz_carries"))
    )
    gz_carries = (
        plays.filter(
            (pl.col("play_type") == "run")
            & (pl.col("yardline_100") <= GOAL_ZONE_YARDLINE)
            & pl.col("rusher_player_id").is_not_null()
        )
        .group_by(["season", "week", pl.col("rusher_player_id").alias("player_id")])
        .agg(pl.len().alias("gz_carries"))
    )

    return (
        rz_targets.join(rz_carries, on=["season", "week", "player_id"], how="full", coalesce=True)
        .join(gz_carries, on=["season", "week", "player_id"], how="full", coalesce=True)
        .with_columns(
            pl.col("rz_targets").fill_null(0),
            pl.col("rz_carries").fill_null(0),
            pl.col("gz_carries").fill_null(0),
        )
    )


def build_player_week_usage(
    player_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    pbp: pl.DataFrame,
    players_dim: pl.DataFrame,
) -> pl.DataFrame:
    """`interim/player_week_usage.parquet` (SPEC §6.2). Most columns come
    straight from nflreadpy's own `player_stats` -- `target_share`,
    `air_yards_share`, and `wopr` are already precomputed there, not
    re-derived by hand. `offense_snaps`/`offense_snap_pct` come from
    `snap_counts` via the pfr_id crosswalk (see `_snap_counts_by_player_id`).
    `rz_targets`/`rz_carries`/`gz_carries`/`rz_touch_share` come from
    play-by-play. `route_participation`/`xfp` stay null -- see module
    docstring.
    """
    team_carries = player_stats.group_by(["season", "week", "team"]).agg(
        pl.col("carries").sum().alias("_team_carries")
    )
    team_rz_touches = (
        _red_zone_touch_counts(pbp)
        .join(
            player_stats.select("player_id", "season", "week", "team"),
            on=["player_id", "season", "week"],
            how="left",
        )
        .group_by(["season", "week", "team"])
        .agg((pl.col("rz_targets").sum() + pl.col("rz_carries").sum()).alias("_team_rz_touches"))
    )

    base = (
        player_stats.select(
            "player_id",
            "season",
            "week",
            "team",
            "targets",
            "target_share",
            pl.col("receiving_air_yards").alias("air_yards"),
            "air_yards_share",
            "wopr",
            "carries",
        )
        .join(team_carries, on=["season", "week", "team"], how="left")
        .with_columns(
            pl.when(pl.col("targets") > 0)
            .then(pl.col("air_yards") / pl.col("targets"))
            .otherwise(None)
            .alias("adot"),
            pl.when(pl.col("_team_carries") > 0)
            .then(pl.col("carries") / pl.col("_team_carries"))
            .otherwise(None)
            .alias("carry_share"),
        )
    )

    with_snaps = base.join(
        _snap_counts_by_player_id(snap_counts, players_dim),
        on=["player_id", "season", "week"],
        how="left",
    )

    with_rz = (
        with_snaps.join(_red_zone_touch_counts(pbp), on=["player_id", "season", "week"], how="left")
        .join(team_rz_touches, on=["season", "week", "team"], how="left")
        .with_columns(
            pl.col("rz_targets").fill_null(0),
            pl.col("rz_carries").fill_null(0),
            pl.col("gz_carries").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("_team_rz_touches").fill_null(0) > 0)
            .then((pl.col("rz_targets") + pl.col("rz_carries")) / pl.col("_team_rz_touches"))
            .otherwise(None)
            .alias("rz_touch_share")
        )
    )

    return with_rz.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("route_participation"),
        pl.lit(None, dtype=pl.Float64).alias("xfp"),
    ).select(
        "player_id",
        "season",
        "week",
        "offense_snaps",
        "offense_snap_pct",
        "targets",
        "target_share",
        "air_yards",
        "air_yards_share",
        "wopr",
        "adot",
        "carries",
        "carry_share",
        "rz_targets",
        "rz_carries",
        "rz_touch_share",
        "gz_carries",
        "route_participation",
        "xfp",
    )


__all__ = [
    "GOAL_ZONE_YARDLINE",
    "RED_ZONE_YARDLINE",
    "build_defense_position_allowed",
    "build_player_week_stats",
    "build_player_week_usage",
    "build_team_week_context",
]
