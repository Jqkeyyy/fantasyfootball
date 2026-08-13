"""DST feature table (SPEC.md §11.6; task 2.7).

SPEC's own literal feature list: "opponent implied team total, opponent
sack rate allowed, opponent pressure rate allowed, opponent
turnover-worthy play rate, opponent interception rate, opponent
offensive line continuity, own defensive pressure rate, own takeaway
rate, home/away, weather." Ten features, all derivable from
`interim.build.build_team_week_defense` (task 2.7's own new interim
table -- sack/pressure/turnover rates), `interim.build.team_week_context`
(`implied_total`, task 1.3/1.7), `features.team_context.ol_continuity_raw`
(task 1.7), `interim/schedule.parquet` (home/away), and
`interim/weather.parquet` (task 1.3/1.9) -- no dependency on the
skill-position pipeline (`player_week_usage`, `xfp`, `injuries`), which
SPEC's own framing ("separate model, different feature set") doesn't
call for.

Every "opponent" and "own defence" column is a *trailing* signal
(exponentially weighted, span `DST_RATE_SPAN`/`DST_OL_CONTINUITY_SPAN`)
shifted one week before being joined onto its target week -- the same
as_of contract every other feature module in this project enforces
(CLAUDE.md rule 2), applied here directly rather than through
`features/build.py`'s shared shift logic, since DST has no other
consumer of `team_week_defense`'s raw per-week rates to share that
machinery with. `opp_implied_team_total`/`is_home`/weather are *not*
shifted -- they're already known before kickoff of the target week
itself (the same `CURRENT_WEEK_COLUMNS` treatment
`features.team_context` gives `implied_team_total`/`spread`).
"""

from __future__ import annotations

import polars as pl

from ffapp.features.registry import FeatureSpec, register
from ffapp.features.team_context import ewm, ol_continuity_raw
from ffapp.interim.build import RELOCATED_TEAM_ALIASES

SOURCE_TABLE = "team_week_defense"

# Sack/pressure/turnover counts come from ~35-75 dropbacks/plays a single
# game -- noisier week to week than a skill player's own usage shares, so
# a wider span than `features.usage`'s typical ewm_3/ewm_4 is used
# throughout, matching `features.team_context`'s own "team quality"
# windows (`team_epa_off_ewm_8`, `team_success_off_ewm_8`).
DST_RATE_SPAN = 8
DST_OL_CONTINUITY_SPAN = 5

# (raw column on team_week_defense, output column, description) -- each
# becomes both a trailing `ewm` column and a registered `FeatureSpec`.
_WINDOWED_DEFENSE_COLUMNS = (
    ("sack_rate_allowed", "opp_sack_rate_allowed_ewm_8", "opponent's own sacks allowed rate"),
    (
        "pressure_rate_allowed",
        "opp_pressure_rate_allowed_ewm_8",
        "opponent's own pressure allowed rate",
    ),
    (
        "turnover_rate",
        "opp_turnover_rate_ewm_8",
        "opponent's own turnover rate (SPEC's turnover-worthy-play proxy)",
    ),
    (
        "interception_rate_thrown",
        "opp_interception_rate_ewm_8",
        "opponent's own interception-thrown rate",
    ),
    (
        "pressure_rate_forced",
        "own_pressure_rate_forced_ewm_8",
        "this defence's own pressure rate",
    ),
    ("takeaway_rate", "own_takeaway_rate_ewm_8", "this defence's own takeaway rate"),
)

_OPPONENT_OFFENSE_COLUMNS = [
    "opp_sack_rate_allowed_ewm_8",
    "opp_pressure_rate_allowed_ewm_8",
    "opp_turnover_rate_ewm_8",
    "opp_interception_rate_ewm_8",
]
_OWN_DEFENSE_COLUMNS = ["own_pressure_rate_forced_ewm_8", "own_takeaway_rate_ewm_8"]

FEATURE_COLUMNS = [
    "is_home",
    "opp_implied_team_total",
    *_OPPONENT_OFFENSE_COLUMNS,
    "opp_ol_continuity_ewm_5",
    *_OWN_DEFENSE_COLUMNS,
    "wind_mph",
    "precip_prob",
    "temp_f",
    "is_dome",
]


def _team_week_rows(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, season, week) a team actually played -- unpacks
    `schedule`'s home/away pair structure, the same pattern
    `scoring.stats._team_scores` already uses for the identical
    home/away -> per-team-row reshape.

    `home_team`/`away_team` are remapped through
    `interim.build.RELOCATED_TEAM_ALIASES` first -- `schedule` carries a
    relocated franchise's real period-accurate code (e.g. "STL" for the
    Rams in 2015), but `team_week_defense` (pbp-derived, like
    `team_week_context`) only ever carries the modern code ("LA").
    Confirmed live: without this remap, `build_dst_features`/
    `models.dst.build_dst_table`'s later join against the real target
    table silently dropped exactly 129 real team-weeks -- the same
    relocated-franchise gap `interim.build.add_schedule_context`'s own
    docstring already names, hit here independently."""
    home = schedule.select(
        "game_id",
        "season",
        "week",
        pl.col("home_team").replace(RELOCATED_TEAM_ALIASES).alias("team"),
        pl.col("away_team").replace(RELOCATED_TEAM_ALIASES).alias("opponent_team"),
        pl.lit(True).alias("is_home"),
    )
    away = schedule.select(
        "game_id",
        "season",
        "week",
        pl.col("away_team").replace(RELOCATED_TEAM_ALIASES).alias("team"),
        pl.col("home_team").replace(RELOCATED_TEAM_ALIASES).alias("opponent_team"),
        pl.lit(False).alias("is_home"),
    )
    return pl.concat([home, away], how="vertical_relaxed")


def _register_dst_feature(
    name: str,
    description: str,
    *,
    window: str | None,
    source_table: str,
    registry: dict[str, FeatureSpec] | None,
) -> None:
    register(
        FeatureSpec(
            name=name,
            description=description,
            positions=["DST"],
            window=window,
            source_table=source_table,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )


def build_dst_features(
    team_week_defense: pl.DataFrame,
    team_week_context: pl.DataFrame,
    snap_counts: pl.DataFrame,
    schedule: pl.DataFrame,
    weather: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """SPEC §11.6's full feature table, one row per (team, season, week)
    a team actually played. `target` is left for the caller to join on
    separately -- this module owns features, not the scoring engine."""
    with_ol = team_week_context.join(
        ol_continuity_raw(snap_counts), on=["team", "season", "week"], how="left"
    )
    with_ol = ewm(with_ol, "ol_continuity_raw", DST_OL_CONTINUITY_SPAN, "ol_continuity_ewm_5")
    _register_dst_feature(
        "opp_ol_continuity_ewm_5",
        "opponent's own starting-OL continuity",
        window="ewm_5",
        source_table=SOURCE_TABLE,
        registry=registry,
    )

    windowed_defense = team_week_defense
    for raw_column, out_column, description in _WINDOWED_DEFENSE_COLUMNS:
        windowed_defense = ewm(windowed_defense, raw_column, DST_RATE_SPAN, out_column)
        _register_dst_feature(
            out_column, description, window="ewm_8", source_table=SOURCE_TABLE, registry=registry
        )

    lagged_ol = (
        with_ol.sort(["team", "season", "week"])
        .with_columns(pl.col("ol_continuity_ewm_5").shift(1).over(["team", "season"]))
        .select("team", "season", "week", "ol_continuity_ewm_5")
    )
    windowed_defense_columns = [c for _, c, _ in _WINDOWED_DEFENSE_COLUMNS]
    lagged_defense = windowed_defense.sort(["team", "season", "week"]).with_columns(
        [pl.col(c).shift(1).over(["team", "season"]) for c in windowed_defense_columns]
    )

    rows = _team_week_rows(schedule)

    own_defense = rows.join(
        lagged_defense.select("team", "season", "week", *_OWN_DEFENSE_COLUMNS),
        on=["team", "season", "week"],
        how="left",
    )

    opponent_offense = lagged_defense.select(
        pl.col("team").alias("opponent_team"), "season", "week", *_OPPONENT_OFFENSE_COLUMNS
    )
    with_opponent = own_defense.join(
        opponent_offense, on=["opponent_team", "season", "week"], how="left"
    )

    opponent_ol = lagged_ol.select(
        pl.col("team").alias("opponent_team"),
        "season",
        "week",
        pl.col("ol_continuity_ewm_5").alias("opp_ol_continuity_ewm_5"),
    )
    with_ol_joined = with_opponent.join(
        opponent_ol, on=["opponent_team", "season", "week"], how="left"
    )

    opponent_implied_total = team_week_context.select(
        pl.col("team").alias("opponent_team"),
        "season",
        "week",
        pl.col("implied_total").alias("opp_implied_team_total"),
    )
    _register_dst_feature(
        "opp_implied_team_total",
        "opponent's own Vegas-implied points total this week",
        window=None,
        source_table="team_week_context",
        registry=registry,
    )
    with_implied = with_ol_joined.join(
        opponent_implied_total, on=["opponent_team", "season", "week"], how="left"
    )
    _register_dst_feature(
        "is_home",
        "whether this DST's own team is the home team",
        window=None,
        source_table="schedule",
        registry=registry,
    )
    for name, description in [
        ("wind_mph", "at kickoff, real or dome-override"),
        ("precip_prob", "at kickoff, real or dome-override"),
        ("temp_f", "at kickoff, real or dome-override"),
        ("is_dome", "closed/dome roof override applied"),
    ]:
        _register_dst_feature(
            name, description, window=None, source_table="weather", registry=registry
        )

    weather_columns = ["game_id", "wind_mph", "precip_prob", "temp_f", "is_dome"]
    return with_implied.join(weather.select(weather_columns), on="game_id", how="left").select(
        "team", "opponent_team", "season", "week", *FEATURE_COLUMNS
    )


__all__ = ["DST_OL_CONTINUITY_SPAN", "DST_RATE_SPAN", "FEATURE_COLUMNS", "build_dst_features"]
