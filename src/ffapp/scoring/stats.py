"""Assemble the per-player-week stat frame scoring.engine.score_stat_line consumes,
from nflreadpy's player_stats/team_stats/schedules/pbp (SPEC.md §8.4's golden test).

Scoped minimally for task 0.5 -- one season, no partitioning, no persistence. Task
1.1 (Phase 1) will supersede this with the full interim/player_week_stats.parquet
pipeline covering 2015-2026; keep this module's logic in sync with keymap.py's
column contract until then, not the other way around.

Sleeper drafts a team defense as its own roster entity (player_id = team
abbreviation, e.g. "KC"), so DST gets its own row here too, built from
`load_team_stats` rather than summed from individual defenders by hand.
"""

from __future__ import annotations

import polars as pl

# `load_team_stats` is a team's own full offensive box score (passing_yards,
# rushing_yards, receptions, ...) *alongside* its defensive/DST columns -- it is
# not a DST-only table. Selecting only these keeps the DST row from also picking
# up the team's own offense and getting scored as if the defense threw for 300
# yards; every other column (offense DirectStat targets like passing_yards) must
# stay absent from this row so score_stat_line's fill_null(0) makes them inert.
#
# Deliberately excludes team_stats' own `def_tds`/`fumble_recovery_tds` columns --
# both are unreliable for DST credit. `fumble_recovery_tds` conflates a genuine
# defensive score (recovering the *opponent's* fumble) with an offensive player
# recovering their *own* team's fumble and scoring, which is not a defensive
# event (confirmed live: HOU week 15 2025, RB W.Marks recovering QB C.Stroud's
# own fumbled snap). `def_return_tds`, derived below from play-by-play, is the
# reliable replacement for both.
#
# Also excludes team_stats' own `def_fumbles_forced`/`fumble_recovery_opp` --
# both aggregate scrimmage-play AND special-teams-play events together, but
# Sleeper scores those as two separate key pairs (ff/fum_rec vs.
# def_st_ff/def_st_fum_rec). Confirmed live: PIT's real week-1-2025 kickoff-return
# fumble was already being counted under team_stats' general columns, so naively
# adding separate special-teams credit on top would have double-counted it.
# `_forced_fumbles`/`_opponent_fumble_recoveries` below replace both, split by
# `special_teams_play`.
_DST_STAT_COLUMNS = (
    "player_id",
    "season",
    "week",
    "position",
    "points_allowed",
    "opponent_blocked_kicks",
    "special_teams_forced_fumbles",
    "special_teams_fumble_recoveries",
    "def_sacks",
    "def_interceptions",
    "def_fumbles_forced",
    "fumble_recovery_opp",
    "def_return_tds",
    "def_safeties",
    "special_teams_tds",
)


def _team_scores(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team) -> that team's own final score."""
    home = schedules.select(
        pl.col("game_id"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("points_scored"),
    )
    away = schedules.select(
        pl.col("game_id"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("points_scored"),
    )
    return pl.concat([home, away], how="vertical_relaxed")


# nflreadpy's `return_touchdown` flags *any* return score -- interception/fumble
# returns during a scrimmage play (`play_type` "pass"/"run") AND punt/kickoff
# returns (`play_type` "punt"/"kickoff"). Sleeper scores those two categories with
# different keys (def_td vs. def_st_td/st_td), so counting all of them here would
# double-count a punt/kickoff return TD against both -- confirmed live, NE's real
# week-4-2025 87-yard punt-return TD was being credited as both. Restricting to
# scrimmage-play types isolates genuine defensive/turnover scores only.
_SCRIMMAGE_PLAY_TYPES = ("pass", "run")


def _defensive_return_tds(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, week, team) -> count of genuine defensive/turnover
    touchdowns: scrimmage plays where the scoring team (`td_team`) was on defense
    (`defteam`) -- an interception return, or a recovery of the *opponent's*
    fumble, either way returned for a score. Excludes an offense recovering its
    own fumble and scoring (that play's `defteam` is the other team, so
    `td_team != defteam` and it's correctly not counted here), and excludes
    punt/kickoff return touchdowns (special-teams credit, not `def_td`'s job).
    """
    return (
        pbp.filter(
            (pl.col("return_touchdown") == 1)
            & (pl.col("td_team") == pl.col("defteam"))
            & pl.col("play_type").is_in(_SCRIMMAGE_PLAY_TYPES)
        )
        .group_by(["season", "week", "defteam"])
        .agg(pl.len().alias("def_return_tds"))
    )


def _fumble_events(pbp: pl.DataFrame) -> pl.DataFrame:
    """Every play with a fumble, carrying which team forced it and which team (if
    any) recovered it -- the raw material both `_forced_fumbles` and
    `_opponent_fumble_recoveries` filter and aggregate from."""
    return pbp.filter(pl.col("fumble") == 1).select(
        "season",
        "week",
        "special_teams_play",
        "forced_fumble_player_1_team",
        "fumbled_1_team",
        "fumble_recovery_1_team",
    )


def _forced_fumbles(fumbles: pl.DataFrame, *, special_teams: bool) -> pl.DataFrame:
    """One row per (season, week, team) -> count of fumbles that team forced,
    restricted to scrimmage plays (special_teams=False, for `ff`) or
    special-teams plays (special_teams=True, for `def_st_ff`)."""
    return (
        fumbles.filter(pl.col("special_teams_play") == (1 if special_teams else 0))
        .filter(pl.col("forced_fumble_player_1_team").is_not_null())
        .group_by(["season", "week", pl.col("forced_fumble_player_1_team").alias("team")])
        .agg(pl.len().alias("count"))
    )


def _opponent_fumble_recoveries(fumbles: pl.DataFrame, *, special_teams: bool) -> pl.DataFrame:
    """One row per (season, week, team) -> count of fumbles that team recovered
    that belonged to its *opponent* -- a genuine turnover, not a team recovering
    its own loose ball. Same scrimmage/special-teams split as `_forced_fumbles`."""
    return (
        fumbles.filter(pl.col("special_teams_play") == (1 if special_teams else 0))
        .filter(
            pl.col("fumble_recovery_1_team").is_not_null()
            & (pl.col("fumble_recovery_1_team") != pl.col("fumbled_1_team"))
        )
        .group_by(["season", "week", pl.col("fumble_recovery_1_team").alias("team")])
        .agg(pl.len().alias("count"))
    )


def _join_team_week_count(
    frame: pl.DataFrame, counts: pl.DataFrame, *, as_column: str
) -> pl.DataFrame:
    """Left-join a (season, week, team) count table onto `frame` (keyed by the
    same three columns), naming the result `as_column` and filling unmatched
    team-weeks with 0. Drops any existing column of that name first -- team_stats
    already carries `def_fumbles_forced`/`fumble_recovery_opp` under these exact
    names, and this derivation replaces them rather than colliding with them."""
    base = frame.drop(as_column) if as_column in frame.columns else frame
    return base.join(
        counts.rename({"count": as_column}), on=["season", "week", "team"], how="left"
    ).with_columns(pl.col(as_column).fill_null(0))


def build_dst_stat_frame(
    team_stats: pl.DataFrame, schedules: pl.DataFrame, pbp: pl.DataFrame
) -> pl.DataFrame:
    """One row per team-week, keyed by team abbreviation as `player_id`, matching
    how Sleeper rosters a DST. Adds `points_allowed` (opponent's score that game),
    `opponent_blocked_kicks` (credit for forcing a block, not suffering one),
    `def_return_tds` (genuine defensive/return touchdowns), and the scrimmage vs.
    special-teams forced-fumble/opponent-recovery splits -- all derived from
    play-by-play (see module docstring for why team_stats' own columns aren't
    trustworthy for any of these).
    """
    scores = _team_scores(schedules)
    blocks = team_stats.select(
        "game_id",
        "team",
        (pl.col("fg_blocked").fill_null(0) + pl.col("pat_blocked").fill_null(0)).alias(
            "kicks_blocked"
        ),
    )
    return_tds = _defensive_return_tds(pbp)
    fumbles = _fumble_events(pbp)

    with_points_allowed = team_stats.join(
        scores.rename({"team": "opponent_team", "points_scored": "points_allowed"}),
        on=["game_id", "opponent_team"],
        how="left",
    )
    with_blocks = with_points_allowed.join(
        blocks.rename({"team": "opponent_team", "kicks_blocked": "opponent_blocked_kicks"}),
        on=["game_id", "opponent_team"],
        how="left",
    )
    with_return_tds = with_blocks.join(
        return_tds.rename({"defteam": "team"}),
        on=["season", "week", "team"],
        how="left",
    ).with_columns(pl.col("def_return_tds").fill_null(0))

    with_fumble_credit = with_return_tds
    for column, counts in (
        ("def_fumbles_forced", _forced_fumbles(fumbles, special_teams=False)),
        ("fumble_recovery_opp", _opponent_fumble_recoveries(fumbles, special_teams=False)),
        ("special_teams_forced_fumbles", _forced_fumbles(fumbles, special_teams=True)),
        (
            "special_teams_fumble_recoveries",
            _opponent_fumble_recoveries(fumbles, special_teams=True),
        ),
    ):
        with_fumble_credit = _join_team_week_count(with_fumble_credit, counts, as_column=column)

    return with_fumble_credit.with_columns(
        pl.col("team").alias("player_id"),
        pl.lit("DST").alias("position"),
    ).select(_DST_STAT_COLUMNS)


def build_player_stat_frame(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Individual player-week rows (offense + kickers). nflreadpy's `player_id`
    already matches the crosswalk's `gsis_id`, and its stat column names already
    match keymap.py's contract -- nothing to rename."""
    return player_stats


def build_stat_frame(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    """Individual player rows and DST team rows, concatenated into the single frame
    `score_stat_line` expects. Columns only one side has (e.g. `passing_yards` for
    DST, `points_allowed` for players) are filled null on the other -- every
    `StatSpec` in keymap.py already handles null via `fill_null`.
    """
    players = build_player_stat_frame(player_stats)
    dst = build_dst_stat_frame(team_stats, schedules, pbp)
    return pl.concat([players, dst], how="diagonal_relaxed")


__all__ = ["build_dst_stat_frame", "build_player_stat_frame", "build_stat_frame"]
