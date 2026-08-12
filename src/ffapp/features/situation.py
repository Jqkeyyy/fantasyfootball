"""The situation feature block (SPEC.md §10.2 "Situation"; task 1.9).

No dedicated earlier task owns this block (unlike usage/team_context/
opponent, each with their own task 1.6/1.7/1.8) -- it's mechanical
passthrough/derivation from sources already ingested (`schedule`,
`injuries`, `weather`), so it lands here as part of task 1.9's own wide-
table assembly, matching `features/build.py`'s own docstring precedent.

Every function here takes an already-built player-week grid (`player_id`,
`season`, `week`, `team` at minimum -- the row universe `features.build`
constructs) and adds columns to it, the same shape as
`usage.build_usage_features`/`team_context.build_team_context_features`.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.registry import FeatureSpec, register

SOURCE_TABLE = "schedule/injuries/weather"

_ALL_OFFENSE = ["QB", "RB", "WR", "TE"]

RULED_OUT_STATUS = "Out"
DEFAULT_REPORT_STATUS = "None"

# is_primetime: Thursday/Monday games are always primetime; a Sunday game
# is primetime only if it kicks off late local time (Sunday Night
# Football). nflverse has no direct broadcast-network column, so this is
# a documented heuristic on real, already-ingested fields (weekday +
# local gametime), not a guess at data that doesn't exist.
_PRIMETIME_WEEKDAYS = ("Thursday", "Monday")
SUNDAY_NIGHT_LOCAL_HOUR = 20


def _is_primetime_expr() -> pl.Expr:
    hour = pl.col("gametime").str.slice(0, 2).cast(pl.Int64)
    return pl.col("weekday").is_in(_PRIMETIME_WEEKDAYS) | (
        (pl.col("weekday") == "Sunday") & (hour >= SUNDAY_NIGHT_LOCAL_HOUR)
    )


def _team_game(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, season, week): that team's own `game_id`,
    `is_home`, `rest_days`, `is_primetime` for a real scheduled game.
    A team with no game that week (a bye) has no row here -- callers join
    on this and byes naturally drop out, rather than needing an explicit
    bye-week filter elsewhere."""
    home_side = schedule.select(
        "game_id",
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.lit(True).alias("is_home"),
        pl.col("home_rest").alias("rest_days"),
        "weekday",
        "gametime",
    )
    away_side = schedule.select(
        "game_id",
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.lit(False).alias("is_home"),
        pl.col("away_rest").alias("rest_days"),
        "weekday",
        "gametime",
    )
    return (
        pl.concat([home_side, away_side], how="vertical_relaxed")
        .with_columns(_is_primetime_expr().alias("is_primetime"))
        .drop("weekday", "gametime")
    )


def add_schedule_situation(grid: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """`is_home`, `rest_days`, `is_primetime`, `week_number` (SPEC §10.2).
    `week_number` is a plain passthrough of the grid's own `week` column,
    not a lookup -- included as its own named feature since SPEC lists it
    separately ("captures late-season rest/tanking effects")."""
    return grid.join(_team_game(schedule), on=["team", "season", "week"], how="left").with_columns(
        pl.col("week").alias("week_number")
    )


def add_weather(grid: pl.DataFrame, schedule: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """`wind_mph`, `precip_prob`, `temp_f`, `is_dome` -- joined via the
    team's own `game_id` for that week (from `_team_game`), then onto
    `weather.parquet` (task 1.3's `ingest.weather.fetch_weather_for_schedule`
    output, one row per `game_id`)."""
    team_games = _team_game(schedule).select("team", "season", "week", "game_id")
    with_game = grid.join(team_games, on=["team", "season", "week"], how="left")
    return with_game.join(
        weather.select("game_id", "wind_mph", "precip_prob", "temp_f", "is_dome"),
        on="game_id",
        how="left",
    ).drop("game_id")


def _latest_injury_report(injuries: pl.DataFrame) -> pl.DataFrame:
    """A player can have more than one real row for the same
    (player_id, season, week) -- confirmed live: 4 real instances across
    2015-2025, some from a genuine mid-week trade (two different teams,
    same week, both real), some from multiple same-week report updates
    to the same team. Keeps the row with the latest `date_modified` --
    "as of the Friday report" (SPEC §10.2) means the most recent real
    designation as of the cutoff, the same authority this project already
    gives `date_modified` elsewhere (task 1.4's own backfill logic)."""
    return (
        injuries.sort("date_modified")
        .group_by(["player_id", "season", "week"], maintain_order=True)
        .agg(pl.all().exclude(["player_id", "season", "week"]).last())
        .select(injuries.columns)
    )


def add_injury_report(grid: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    """`report_status`, `practice_participation` (SPEC §10.2). A player
    with no real injury-report row that week is healthy, not unknown --
    `report_status` fills to the literal `"None"` (one of SPEC's own
    stated categories: "Out / Doubtful / Questionable / None"), not left
    null. `practice_participation` has no such "healthy" category in
    SPEC's own definition, so it stays null (honestly no practice report
    to describe, not an unknown one).

    `interim/injuries.parquet`'s own column is named `practice_status`
    (task 1.4's naming) -- aliased to `practice_participation` here to
    match SPEC §10.2's own feature name; confirmed live, the real column
    does not carry the name this module's own feature does."""
    latest = _latest_injury_report(injuries)
    return grid.join(
        latest.select(
            "player_id",
            "season",
            "week",
            "report_status",
            pl.col("practice_status").alias("practice_participation"),
        ),
        on=["player_id", "season", "week"],
        how="left",
    ).with_columns(pl.col("report_status").fill_null(DEFAULT_REPORT_STATUS))


def add_weeks_since_return(grid: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    """`weeks_since_return`: real games elapsed since the player's last
    week with `report_status == "Out"` (SPEC: "games since last missed
    game due to injury") -- a *count of real games*, not a raw
    `season*100+week` arithmetic difference, which would badly
    misrepresent the gap across a season boundary (e.g. week 17 of one
    season to week 1 of the next is 1 real elapsed week, not the ~83 a
    naive numeric encoding would produce). Ranked per player over the
    grid's own chronological row order instead, which is robust to that
    boundary by construction.

    Deliberately spans seasons rather than resetting each one (unlike
    `features.usage`'s within-season-only windows -- see that module's
    own docstring for why *those* reset): a "role" genuinely resets each
    year, but a torn ACL from last December doesn't. Null when the player
    has no real `Out` week anywhere in their tracked history -- nothing
    to measure a return from, an honest gap rather than a guessed 0.
    """
    ranked = grid.sort(["player_id", "season", "week"]).with_columns(
        pl.int_range(pl.len()).over("player_id").cast(pl.Float64).alias("_rank")
    )
    latest = _latest_injury_report(injuries)
    out_ranks = (
        latest.filter(pl.col("report_status") == RULED_OUT_STATUS)
        .select("player_id", "season", "week")
        .join(
            ranked.select("player_id", "season", "week", "_rank"),
            on=["player_id", "season", "week"],
            how="inner",
        )
        .rename({"_rank": "_out_rank"})
        .sort(["player_id", "_out_rank"])
    )

    lookup = ranked.with_columns((pl.col("_rank") - 0.5).alias("_asof_rank")).sort(
        ["player_id", "_asof_rank"]
    )
    matched = lookup.join_asof(
        out_ranks.select("player_id", pl.col("_out_rank").alias("_asof_rank"), "_out_rank"),
        on="_asof_rank",
        by="player_id",
        strategy="backward",
        check_sortedness=False,  # both sides are already sorted by (by..., on) above
    )
    return matched.with_columns(
        (pl.col("_rank") - pl.col("_out_rank")).alias("weeks_since_return")
    ).drop(["_rank", "_asof_rank", "_out_rank"])


_SITUATION_FEATURES = [
    ("is_home", None),
    ("rest_days", "days since previous game"),
    ("is_primetime", None),
    ("week_number", "captures late-season rest/tanking effects"),
    ("wind_mph", None),
    ("precip_prob", None),
    ("temp_f", None),
    ("is_dome", "dome forces wind to 0"),
    ("report_status", "Out / Doubtful / Questionable / None, as of the Friday report"),
    ("practice_participation", "DNP / Limited / Full"),
    ("weeks_since_return", "games since last missed game due to injury"),
]


def build_situation_features(
    grid: pl.DataFrame,
    schedule: pl.DataFrame,
    injuries: pl.DataFrame,
    weather: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Assemble every SPEC §10.2 "Situation" feature and register each
    one's `FeatureSpec`. Every feature here is `lag_weeks=1`,
    `available_at_inference=True` -- `report_status`/`practice_participation`
    are "as of the Friday report," strictly pre-kickoff, same as every
    other situation feature; none of these have an in-season gap the way
    route participation does (SPEC §10.5).
    """
    result = add_schedule_situation(grid, schedule)
    result = add_weather(result, schedule, weather)
    result = add_injury_report(result, injuries)
    result = add_weeks_since_return(result, injuries)

    for name, description in _SITUATION_FEATURES:
        register(
            FeatureSpec(
                name=name,
                description=description or name,
                positions=_ALL_OFFENSE,
                window=None,
                source_table=SOURCE_TABLE,
                available_at_inference=True,
                lag_weeks=1,
            ),
            registry=registry,
        )

    return result


__all__ = [
    "DEFAULT_REPORT_STATUS",
    "RULED_OUT_STATUS",
    "SOURCE_TABLE",
    "SUNDAY_NIGHT_LOCAL_HOUR",
    "add_injury_report",
    "add_schedule_situation",
    "add_weather",
    "add_weeks_since_return",
    "build_situation_features",
]
