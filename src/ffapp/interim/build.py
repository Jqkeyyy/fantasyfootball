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
- `player_week_usage.xfp` -- populated by `add_xfp` (task 1.2, ffopportunity
  ingestion), a separate step from `build_player_week_usage` since it needs
  a second source (`ingest.nflverse.fetch_ff_opportunity`) task 1.1 didn't
  fetch.
- `team_week_context.proe`, `.neutral_pace_sec` -- task 1.7 ("Team context
  features"); both need real modelling (an expected-pass-rate baseline,
  careful play-sequencing to measure real elapsed time), not mechanical
  aggregation. `.implied_total`/`.spread` *are* populated (task 1.3,
  `add_schedule_context`) now that `spread_line`'s sign convention has
  been verified against real data.
- `schedule.kickoff_utc` -- *is* populated now too (`add_kickoff_utc`,
  task 1.3), via `config/stadiums.csv`'s per-venue timezone, keyed by
  `stadium_id` (the actual game venue, correct for a relocated team or an
  international game, not just the home team's usual city).
  `.home_implied_total`/`.away_implied_total` *are* populated
  (`ingest.nflverse.normalize_schedule`, task 1.3) now that
  `spread_line`'s sign convention has been verified: across all 3,028
  real completed games 2015-2025, `spread_line` correlates +0.44 with the
  actual home-away score margin, and the extreme cases are unambiguous --
  positive `spread_line` means the home team is favoured by that many
  points, confirming SPEC's own stated assumption.
- `defense_position_allowed.adj_*` *are* populated now (task 1.8,
  `add_opponent_adjustment`) -- see that function's own docstring for the
  walk-forward/shrinkage design.
- `defense_position_allowed`'s position groups collapse `WR_perimeter`/
  `WR_slot` into one undifferentiated `WR` -- splitting by alignment needs
  the same missing NGS/FTN charting data as the route_participation gap
  above.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge

from ffapp.scoring.stats import build_stat_frame

_SCRIMMAGE_PLAY_TYPES = ("pass", "run")
RED_ZONE_YARDLINE = 20
GOAL_ZONE_YARDLINE = 5

# task 1.7 proe: the down/distance/score baseline model's feature set.
# ydstogo/score_differential are capped -- an uncapped garbage-time
# 40-point differential or a 3rd-and-38 has no real bearing on ordinary
# play-calling and would only add noise/leverage to the fit.
YDSTOGO_CAP = 20.0
SCORE_DIFFERENTIAL_CAP = 28.0
_PROE_CONTINUOUS_FEATURES = (
    "ydstogo",
    "score_differential",
    "half_seconds_remaining",
    "yardline_100",
)
_PROE_DOWNS = (1.0, 2.0, 3.0, 4.0)

# player_week_usage is about offensive skill-position usage (targets, air
# yards, carries) -- nflreadpy's own player_stats carries a row for every
# position that recorded *any* stat that week, 26 distinct position codes
# including LB/CB/DE/DL/OL/LS (the same "IDP-style columns on every row"
# quirk task 0.4/0.5 already found -- a defender's stray def_sacks/def_int
# value still produces a player_stats row). Confirmed live: without this
# filter, task 1.2's xfp coverage check came out at 30% purely because
# ~140k of ~200k player_week_usage rows were non-skill-position players
# ffopportunity correctly has no data for.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# (receiver position, play_type) -> SPEC §6.2's defense_position_allowed
# group. WR_perimeter/WR_slot collapsed to "WR" -- see module docstring.
# QB_passing is deliberately *not* here -- see _passer_scoped_plays.
_RECEIVER_POSITION_GROUP_MAP = {
    ("RB", "pass"): "RB_receiving",
    ("TE", "pass"): "TE",
    ("WR", "pass"): "WR",
}
# (rusher position, play_type) -> group.
_RUSHER_POSITION_GROUP_MAP = {
    ("QB", "run"): "QB_rushing",
    ("RB", "run"): "RB_rushing",
}
QB_PASSING_GROUP = "QB_passing"


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
    columns, no modelling needed. `proe`/`neutral_pace_sec`/`implied_total`/
    `spread` all start null; `implied_total`/`spread` are filled in
    separately by `add_schedule_context` (task 1.3) since they need a
    second source (`schedule`) this function doesn't take. `proe`/
    `neutral_pace_sec` stay null -- see module docstring.
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


def _proe_training_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Real play-calling decisions only, for fitting the proe baseline:
    scrimmage plays with a real `down` (excludes two-point-conversion
    tries, where `down` is null) and excluding kneels/spikes -- neither is
    a real pass-vs-run decision, and including them would bias the
    baseline toward "run" in the exact clock-killing/clock-stopping
    situations they occur in. `ydstogo`/`score_differential` are capped
    (see their module-level constants) before fitting.

    Target is nflverse's own `pass` indicator, not `play_type` -- `pass`
    already reflects passer *intent* (a sack or a scramble both count as
    "pass" there), which is what "pass rate over expectation" means;
    `play_type` would misclassify a scramble as a designed run.
    """
    plays = _scrimmage_plays(pbp)
    return plays.filter(
        pl.col("down").is_not_null() & (pl.col("qb_kneel") == 0) & (pl.col("qb_spike") == 0)
    ).with_columns(
        pl.col("ydstogo").clip(upper_bound=YDSTOGO_CAP),
        pl.col("score_differential").clip(-SCORE_DIFFERENTIAL_CAP, SCORE_DIFFERENTIAL_CAP),
    )


def _proe_design_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot `down` (its effect is categorical, not linear -- 4th down
    behaves nothing like an extrapolation from 1st-3rd) plus the
    continuous situational features."""
    return df.select(
        *[(pl.col("down") == d).cast(pl.Float64).alias(f"_down_{int(d)}") for d in _PROE_DOWNS],
        *_PROE_CONTINUOUS_FEATURES,
    )


def _fit_proe_models(training: pl.DataFrame) -> dict[int, LogisticRegression]:
    """One logistic model per season, fit only on *prior* seasons' plays
    (walk-forward, confirmed) -- the earliest season in the input has no
    prior data and gets no model, so its `proe` stays honestly null
    rather than using a same-season or future-season fit. Refitting once
    per season (not per week) is enough: within-season situational
    tendencies barely shift, and this avoids ~18 refits/season for no
    real behavioural gain.
    """
    models: dict[int, LogisticRegression] = {}
    seasons = sorted(training["season"].unique().to_list())
    for season in seasons:
        prior = training.filter(pl.col("season") < season)
        if prior.height == 0:
            continue
        design = _proe_design_matrix(prior)
        model = LogisticRegression(max_iter=1000)
        model.fit(design.to_numpy(), prior["pass"].to_numpy())
        models[season] = model
    return models


def add_proe(team_week_context: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Task 1.7: `proe` (SPEC §10.2) -- each play's actual `pass` indicator
    minus the walk-forward baseline model's predicted pass probability for
    that play's situation, averaged per (team, season, week). Confirmed
    with you: walk-forward per season (see `_fit_proe_models`), not
    nflverse's own pre-built `xpass`/`pass_oe` (a single static fit across
    the whole historical release, which would leak later seasons' league-
    wide passing trends into earlier seasons' backtest folds) and not a
    single static fit of our own either.
    """
    training = _proe_training_frame(pbp)
    models = _fit_proe_models(training)

    if not models:
        return team_week_context

    parts = []
    for season, model in models.items():
        season_plays = training.filter(pl.col("season") == season)
        design = _proe_design_matrix(season_plays)
        predicted = model.predict_proba(design.to_numpy())[:, 1]
        parts.append(season_plays.with_columns(pl.Series("_predicted_pass_prob", predicted)))
    with_predictions = pl.concat(parts, how="vertical_relaxed")

    proe_by_team_week = (
        with_predictions.with_columns(
            (pl.col("pass") - pl.col("_predicted_pass_prob")).alias("_proe_play")
        )
        .group_by(["season", "week", pl.col("posteam").alias("team")])
        .agg(pl.col("_proe_play").mean().alias("proe"))
    )
    return team_week_context.drop("proe").join(
        proe_by_team_week, on=["team", "season", "week"], how="left"
    )


def _parse_minutes_seconds(col: str) -> pl.Expr:
    parts = pl.col(col).str.split(":")
    return parts.list.get(0).cast(pl.Int64) * 60 + parts.list.get(1).cast(pl.Int64)


NEUTRAL_SCRIPT_SCORE_MARGIN = 7
NEUTRAL_SCRIPT_QUARTERS = (1.0, 2.0, 3.0)


def add_neutral_pace(team_week_context: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Task 1.7: `neutral_pace_sec` (SPEC §10.2) -- seconds per play in
    neutral game script (score within 7, Q1-Q3), per (team, season, week).

    Uses real per-drive `drive_time_of_possession`/`drive_play_count`
    (already-computed nflverse drive aggregates -- confirmed live,
    `drive_play_count` counts only real scrimmage snaps, not the
    kickoff/PAT rows that share the same `drive` value) rather than
    differencing `game_seconds_remaining` between consecutive plays --
    the latter needs careful handling of timeouts/quarter breaks/
    injury stoppages to avoid inflating the gap; the former sidesteps all
    of that by using a value nflverse has already computed correctly.

    A drive's neutral-script classification is decided by its *starting*
    context (first scrimmage play's `score_differential`/
    `drive_quarter_start`), not evaluated play-by-play -- score changes
    mid-drive, and pace is a property of the drive's game-script
    conditions when it began, not a play-by-play filter within it.
    """
    plays = _scrimmage_plays(pbp)
    per_drive = (
        plays.sort(["game_id", "play_id"])
        .group_by(["season", "week", "posteam", "drive"], maintain_order=True)
        .agg(
            pl.col("score_differential").first().alias("_start_score_differential"),
            pl.col("drive_quarter_start").first().alias("_start_quarter"),
            pl.col("drive_time_of_possession").first().alias("_top_str"),
            pl.col("drive_play_count").first().alias("_play_count"),
        )
    )
    neutral_drives = per_drive.filter(
        (pl.col("_start_score_differential").abs() <= NEUTRAL_SCRIPT_SCORE_MARGIN)
        & pl.col("_start_quarter").is_in(NEUTRAL_SCRIPT_QUARTERS)
    ).with_columns(_parse_minutes_seconds("_top_str").alias("_top_seconds"))

    per_team_week = (
        neutral_drives.group_by(["season", "week", pl.col("posteam").alias("team")])
        .agg(
            pl.col("_top_seconds").sum().alias("_total_seconds"),
            pl.col("_play_count").sum().alias("_total_plays"),
        )
        .with_columns(
            pl.when(pl.col("_total_plays") > 0)
            .then(pl.col("_total_seconds") / pl.col("_total_plays"))
            .otherwise(None)
            .alias("neutral_pace_sec")
        )
        .select("team", "season", "week", "neutral_pace_sec")
    )
    return team_week_context.drop("neutral_pace_sec").join(
        per_team_week, on=["team", "season", "week"], how="left"
    )


# nflverse's `schedule` table keeps each game's period-accurate team code
# (the Rams as "STL" in 2015; the Chargers as "SD" through 2016; the
# Raiders as "OAK" through 2019), but `pbp.posteam`/`.defteam` -- which
# `team_week_context` is built from -- backfills every historical row to
# the team's current/final franchise code. Confirmed live: joining this
# function's schedule-derived rows straight onto real `team_week_context`
# (2015-2025) left exactly 129 real rows with a null `spread`/
# `implied_total` -- every one a pre-move season for one of these three
# franchises, not an honest data gap -- because `team_week_context`
# contains zero "STL"/"SD"/"OAK" rows at all, only the modern codes.
_RELOCATED_TEAM_ALIASES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def add_schedule_context(team_week_context: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Task 1.3: join `spread`/`implied_total` (team's own perspective)
    from `schedule`'s real `spread_line`/`home_implied_total`/
    `away_implied_total` onto `team_week_context` (null from
    `build_team_week_context` until this runs).

    A home team's row gets `spread_line`/`home_implied_total` directly
    (positive `spread_line` = home favoured, verified -- see module
    docstring); an away team's row gets the mirrored `spread`
    (`-spread_line`, since a team's own spread is *its* margin of
    expected victory, not necessarily the home team's) and
    `away_implied_total`. `home_team`/`away_team` are remapped through
    `_RELOCATED_TEAM_ALIASES` first so a relocated franchise's pre-move
    seasons join correctly against `team_week_context`'s modern-code-only
    rows (see that mapping's own comment).
    """
    home_side = schedule.select(
        "season",
        "week",
        pl.col("home_team").replace(_RELOCATED_TEAM_ALIASES).alias("team"),
        pl.col("spread_line").alias("spread"),
        pl.col("home_implied_total").alias("implied_total"),
    )
    away_side = schedule.select(
        "season",
        "week",
        pl.col("away_team").replace(_RELOCATED_TEAM_ALIASES).alias("team"),
        (-pl.col("spread_line")).alias("spread"),
        pl.col("away_implied_total").alias("implied_total"),
    )
    per_team = pl.concat([home_side, away_side], how="vertical_relaxed")

    return team_week_context.drop(["spread", "implied_total"]).join(
        per_team, on=["season", "week", "team"], how="left"
    )


def add_kickoff_utc(schedule: pl.DataFrame, stadiums: pl.DataFrame) -> pl.DataFrame:
    """Task 1.3: derive `kickoff_utc` (SPEC §6.2's `as_of` boundary) from
    `gameday`+`gametime` (local kickoff wall-clock time, both already
    confirmed non-null across the full real 2015-2025 range) and each
    game's real venue timezone (`config/stadiums.csv`, joined on
    `stadium_id` -- the actual game venue, which already disambiguates a
    relocated team's old stadium or a neutral-site/international game
    from the current row's `home_team`).

    polars' `dt.replace_time_zone` takes one fixed timezone string per
    call, not a per-row value, so this loops over the small number of
    distinct real timezones in `stadiums` (currently 8: five US zones
    plus Berlin/London/Mexico_City/Sao_Paulo for international games)
    rather than one replace_time_zone call per row. A game whose
    `stadium_id` has no match in `stadiums` keeps `kickoff_utc` null
    rather than guessed -- CLAUDE.md rule 2: this is the single most
    leakage-sensitive column in the project, so an honest gap beats a
    wrong timestamp.
    """
    with_tz = (
        schedule.drop("kickoff_utc")
        .join(stadiums.select("stadium_id", "tz"), on="stadium_id", how="left")
        .with_columns(
            pl.concat_str([pl.col("gameday"), pl.lit(" "), pl.col("gametime")])
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
            .alias("_local_dt")
        )
    )

    tzs = with_tz.select("tz").unique().drop_nulls().to_series().to_list()
    parts = [
        with_tz.filter(pl.col("tz") == tz).with_columns(
            pl.col("_local_dt")
            .dt.replace_time_zone(tz)
            .dt.convert_time_zone("UTC")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .alias("kickoff_utc")
        )
        for tz in tzs
    ]
    unmatched = with_tz.filter(pl.col("tz").is_null()).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("kickoff_utc")
    )

    combined = pl.concat(parts + [unmatched], how="vertical_relaxed").drop(["tz", "_local_dt"])
    return combined.select(schedule.columns)


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
    with_group = _position_group_plays(pbp, player_stats)

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


_POSITION_GROUP_PLAY_COLUMNS = (
    "season",
    "week",
    "posteam",
    "defteam",
    "position_group",
    "epa",
    "success",
    "yards_gained",
    "touchdown",
    "home",
)


def _receiver_scoped_plays(plays: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """WR/TE/RB_receiving: one row per real target, grouped by the
    *receiver's* own position -- a defense's differential effect on plays
    where the ball actually went to a given position group."""
    return (
        plays.filter((pl.col("play_type") == "pass") & pl.col("receiver_player_id").is_not_null())
        .join(
            positions.rename({"player_id": "receiver_player_id"}),
            on=["receiver_player_id", "season"],
            how="left",
        )
        .with_columns(
            pl.struct(["position"])
            .map_elements(
                lambda s: _RECEIVER_POSITION_GROUP_MAP.get((s["position"], "pass")),
                return_dtype=pl.Utf8,
            )
            .alias("position_group")
        )
    )


def _rusher_scoped_plays(plays: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """QB_rushing/RB_rushing: one row per real carry, grouped by the
    *rusher's* own position (a QB scramble/designed run is a real QB_rushing
    play; a QB kneel is excluded upstream by `_scrimmage_plays`'s own
    pass/run scoping only in the sense that kneels are still `play_type ==
    "run"` -- not filtered here, since defense_position_allowed has no
    reason to exclude a real clock-killing snap the way proe's baseline
    model does)."""
    return (
        plays.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null())
        .join(
            positions.rename({"player_id": "rusher_player_id"}),
            on=["rusher_player_id", "season"],
            how="left",
        )
        .with_columns(
            pl.struct(["position"])
            .map_elements(
                lambda s: _RUSHER_POSITION_GROUP_MAP.get((s["position"], "run")),
                return_dtype=pl.Utf8,
            )
            .alias("position_group")
        )
    )


def _passer_scoped_plays(plays: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """QB_passing (task 1.8 fix): one row per real pass *attempt* -- not
    joined on the intended receiver's position, which almost never
    resolves to "QB" (confirmed live: only 123 real plays across the full
    2015-2025 range under the old receiver-based join, versus 118k-132k
    for every other group). Joined on the *passer's* own position instead,
    scoped to real pass attempts with a resolvable `passer_player_id` --
    a broader population than the receiver-scoped groups, since a sack has
    no `receiver_player_id` at all but is still a real pass-defense
    outcome (heavily negative EPA) that QB_passing should capture.
    Deliberately overlaps the receiver-scoped groups: a single completed
    pass to a WR contributes to *both* "WR" (the receiver-specific signal)
    and "QB_passing" (the defense's overall effect on the passer) -- two
    different signals over intersecting populations, not a partition.
    """
    return (
        plays.filter((pl.col("play_type") == "pass") & pl.col("passer_player_id").is_not_null())
        .join(
            positions.rename({"player_id": "passer_player_id"}),
            on=["passer_player_id", "season"],
            how="left",
        )
        .filter(pl.col("position") == "QB")
        .with_columns(pl.lit(QB_PASSING_GROUP).alias("position_group"))
    )


def _position_group_plays(pbp: pl.DataFrame, player_stats: pl.DataFrame) -> pl.DataFrame:
    """The real per-play population behind both `n_plays`
    (`build_defense_position_allowed`) and the ridge opponent-adjustment
    fit (task 1.8) -- computed once here so the two can never drift apart
    on which plays actually count for a given position_group.
    """
    positions = _player_position_by_season(player_stats)
    plays = _scrimmage_plays(pbp).with_columns(
        (pl.col("posteam_type") == "home").cast(pl.Float64).alias("home")
    )

    combined = pl.concat(
        [
            _receiver_scoped_plays(plays, positions),
            _rusher_scoped_plays(plays, positions),
            _passer_scoped_plays(plays, positions),
        ],
        how="vertical_relaxed",
    )
    return (
        combined.filter(pl.col("position_group").is_not_null())
        .select(*_POSITION_GROUP_PLAY_COLUMNS)
        # task 1.8: the ridge fit needs every outcome column populated --
        # a stray null (rare; hasn't been confirmed in real data) would
        # otherwise reach numpy as NaN and silently poison an entire fit.
        .drop_nulls(["epa", "success", "yards_gained", "touchdown", "home"])
    )


OPPONENT_ADJUSTMENT_SHRINKAGE_K = 250.0
OPPONENT_ADJUSTMENT_RECENCY_SPAN = 8
RIDGE_ALPHA = 1.0
_RATE_OUTCOMES = ("epa", "success", "yards_gained", "touchdown")
_RATE_OUTCOME_ADJ_COLUMNS = {
    "epa": "adj_epa_allowed",
    "success": "adj_success_allowed",
    "yards_gained": "adj_ypt_allowed",
    "touchdown": "adj_td_rate_allowed",
}


def _recency_weights(as_of_week: int, play_weeks: pl.Series, span: int) -> np.ndarray:
    """Exponential recency weight, span `span` -- same span semantics as
    `features.usage.ewm`'s `.ewm_mean(span=k)` (`alpha = 2/(k+1)`), applied
    manually here since sklearn's `Ridge.fit` takes a plain
    `sample_weight` array, not a polars ewm expression. A play `L` weeks
    before `as_of_week` gets weight `(1-alpha)**L` -- defences change
    through a season (injuries, coordinator adjustments), so a play from
    8 weeks ago should count for much less than one from last week."""
    alpha = 2.0 / (span + 1)
    weeks_ago = (as_of_week - play_weeks).to_numpy()
    return (1.0 - alpha) ** weeks_ago


def _ridge_defense_coefficients(
    plays: pl.DataFrame, sample_weight: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """SPEC §10.4's formula, literally: `y = mu + offense_team +
    defense_team + home + eps`, ridge-regularised, one-hot team factors,
    fit once across all 4 rate outcomes simultaneously (`Ridge` supports a
    multi-output `y`, since every outcome shares the same design matrix --
    a single `.fit()` call rather than 4). Ridge's own L2 penalty handles
    the intercept/one-hot collinearity without needing to drop a reference
    team the way plain OLS would.

    Returns `{defteam: {outcome: coefficient}}` -- the fitted
    `defense_team` block of the coefficient matrix, which SPEC calls "the
    opponent-adjusted values" directly. A team that never appears in
    `plays` has no entry (not a 0.0 -- see `add_opponent_adjustment`'s own
    per-team shrinkage handling for why that distinction matters).
    """
    offense_teams = sorted(plays["posteam"].unique().to_list())
    defense_teams = sorted(plays["defteam"].unique().to_list())

    offense_cols = [
        (pl.col("posteam") == team).cast(pl.Float64).alias(f"_off_{i}")
        for i, team in enumerate(offense_teams)
    ]
    defense_cols = [
        (pl.col("defteam") == team).cast(pl.Float64).alias(f"_def_{i}")
        for i, team in enumerate(defense_teams)
    ]
    design = plays.select(*offense_cols, *defense_cols, "home")
    x = design.to_numpy()
    y = plays.select(list(_RATE_OUTCOMES)).to_numpy()

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x, y, sample_weight=sample_weight)

    n_offense = len(offense_teams)
    defense_block = model.coef_[:, n_offense : n_offense + len(defense_teams)]

    return {
        team: {outcome: float(defense_block[i, j]) for i, outcome in enumerate(_RATE_OUTCOMES)}
        for j, team in enumerate(defense_teams)
    }


def _team_n_plays(plays: pl.DataFrame) -> dict[str, int]:
    return dict(plays.group_by("defteam").agg(pl.len().alias("n")).iter_rows())


def _shrinkage_weight(n_plays: int, k: float = OPPONENT_ADJUSTMENT_SHRINKAGE_K) -> float:
    """SPEC §10.4's empirical-Bayes blend weight: `w = n_plays / (n_plays
    + k)`. `w=0` with no trailing data (pure prior-season estimate);
    approaches 1 as trailing volume grows large relative to `k`."""
    return n_plays / (n_plays + k)


def add_opponent_adjustment(
    defense_position_allowed: pl.DataFrame, pbp: pl.DataFrame, player_stats: pl.DataFrame
) -> pl.DataFrame:
    """Task 1.8, SPEC §10.4: fills in `defense_position_allowed.parquet`'s
    `adj_epa_allowed`/`adj_success_allowed`/`adj_ypt_allowed`/
    `adj_td_rate_allowed` (null since task 1.1).

    Walk-forward per (position_group, season, week): the "current-season"
    ridge estimate is fit only on that season's *strictly prior* weeks
    (never the target week itself or later -- the as_of contract), with
    exponential recency weighting (`_recency_weights`, span 8). This is
    blended with a *separately* fit prior-season estimate (one ridge fit
    per (position_group, season) using that whole season's plays) via
    SPEC's own shrinkage formula: `w = n_plays / (n_plays + k)`, `k=250`,
    where `n_plays` is that specific *team's* own trailing play count in
    the current season so far -- not the group-wide total, so a defense
    that's individually thin on data (bye weeks, a short trailing window)
    shrinks harder even mid-season while its opponents don't.

    A team with zero current-season trailing plays (every season's own
    week 1, or a team missing from the ridge fit entirely) gets `w=0` --
    the estimate is the prior-season value alone. A team with no
    prior-season estimate either (the first tracked season, 2015, or a
    team that didn't exist last season) falls back to the current-season
    estimate alone once one exists, and stays honestly null before that
    (season 2015's own week 1: no prior season, no current-season trailing
    data -- nothing to estimate from, same precedent as task 1.7's `proe`
    for the same reason).
    """
    plays = _position_group_plays(pbp, player_stats)

    prior_season_fits: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for position_group in plays["position_group"].unique().sort().to_list():
        group_plays = plays.filter(pl.col("position_group") == position_group)
        for season in group_plays["season"].unique().sort().to_list():
            season_plays = group_plays.filter(pl.col("season") == season)
            prior_season_fits[(position_group, season)] = _ridge_defense_coefficients(season_plays)

    rows: list[dict[str, object]] = []
    for position_group in plays["position_group"].unique().sort().to_list():
        group_plays = plays.filter(pl.col("position_group") == position_group)
        for season in group_plays["season"].unique().sort().to_list():
            season_plays = group_plays.filter(pl.col("season") == season)
            prior_estimate = prior_season_fits.get((position_group, season - 1))

            for week in season_plays["week"].unique().sort().to_list():
                trailing = season_plays.filter(pl.col("week") < week)
                current_estimate: dict[str, dict[str, float]] | None = None
                n_plays_by_team: dict[str, int] = {}
                if trailing.height > 0:
                    n_plays_by_team = _team_n_plays(trailing)
                    weights = _recency_weights(
                        week, trailing["week"], OPPONENT_ADJUSTMENT_RECENCY_SPAN
                    )
                    current_estimate = _ridge_defense_coefficients(trailing, sample_weight=weights)

                teams = set(n_plays_by_team) | (set(prior_estimate) if prior_estimate else set())
                for team in teams:
                    n = n_plays_by_team.get(team, 0)
                    w = _shrinkage_weight(n)
                    cur_team = current_estimate.get(team) if current_estimate else None
                    pri_team = prior_estimate.get(team) if prior_estimate else None

                    row: dict[str, object] = {
                        "defteam": team,
                        "season": season,
                        "week": week,
                        "position_group": position_group,
                    }
                    for outcome, col in _RATE_OUTCOME_ADJ_COLUMNS.items():
                        cur = cur_team.get(outcome) if cur_team else None
                        pri = pri_team.get(outcome) if pri_team else None
                        if cur is None and pri is None:
                            adj = None
                        elif pri is None:
                            adj = cur
                        elif cur is None:
                            adj = pri
                        else:
                            adj = w * cur + (1 - w) * pri
                        row[col] = adj
                    rows.append(row)

    # season/week must match `defense_position_allowed`'s own real dtypes
    # exactly (Int32 in real nflverse-derived data) -- a mismatched-width
    # join key doesn't error in polars, it just silently matches zero rows
    # (the same footgun task 1.2's ff_opportunity join already hit once).
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "defteam": pl.Utf8,
        "season": defense_position_allowed.schema["season"],
        "week": defense_position_allowed.schema["week"],
        "position_group": pl.Utf8,
        **{col: pl.Float64 for col in _RATE_OUTCOME_ADJ_COLUMNS.values()},
    }
    adjustments = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    return (
        defense_position_allowed.drop(list(_RATE_OUTCOME_ADJ_COLUMNS.values()))
        .join(adjustments, on=["defteam", "season", "week", "position_group"], how="left")
        .select(defense_position_allowed.columns)
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


def _designed_rush_attempts(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (player_id, season, week): rush attempts excluding QB
    scrambles (task 1.6, SPEC §10.2's `designed_rush_share` -- a called
    run, not the passer improvising out of the pocket). Real for any
    rusher, not just QBs: pbp's own `qb_scramble` flag is only ever 1 when
    the play's rusher is the quarterback (confirmed live: 10,039 of
    156,789 real 2015-2025 runs), so this equals a non-QB's plain carry
    count -- the QB/non-QB distinction the feature actually needs falls
    out naturally rather than needing a position filter here.
    """
    plays = _scrimmage_plays(pbp)
    return (
        plays.filter(
            (pl.col("play_type") == "run")
            & (pl.col("qb_scramble") == 0)
            & pl.col("rusher_player_id").is_not_null()
        )
        .group_by(["season", "week", pl.col("rusher_player_id").alias("player_id")])
        .agg(pl.len().alias("designed_rush_attempts"))
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

    `team` is kept in the output -- task 1.6's window features need it for
    per-team-week joins (and it costs nothing extra; `base` already joins
    it in). `gz_carry_share` (task 1.6, SPEC §10.2) mirrors
    `rz_touch_share`'s own team-denominator pattern.
    `designed_rush_attempts`/`designed_rush_share` (task 1.6, QB rushing
    floor) come from `_designed_rush_attempts` above.

    Output rows are scoped to `SKILL_POSITIONS` (see its own comment) --
    but `_team_carries`/`_team_rz_touches`/`_team_gz_carries` denominators
    are computed from the *unfiltered* `player_stats`/play-by-play first,
    since a team's real rushing total can include a non-skill-position
    trick-play carry (a fullback, a wildcat snap) that a skill-position-only
    sum would undercount.
    """
    team_carries = player_stats.group_by(["season", "week", "team"]).agg(
        pl.col("carries").sum().alias("_team_carries")
    )
    rz_touches = _red_zone_touch_counts(pbp)
    team_rz_touches = (
        rz_touches.join(
            player_stats.select("player_id", "season", "week", "team"),
            on=["player_id", "season", "week"],
            how="left",
        )
        .group_by(["season", "week", "team"])
        .agg(
            (pl.col("rz_targets").sum() + pl.col("rz_carries").sum()).alias("_team_rz_touches"),
            pl.col("gz_carries").sum().alias("_team_gz_carries"),
        )
    )
    designed_rush = _designed_rush_attempts(pbp)

    base = (
        player_stats.filter(pl.col("position").is_in(SKILL_POSITIONS))
        .select(
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
        .join(designed_rush, on=["player_id", "season", "week"], how="left")
        .with_columns(
            pl.when(pl.col("targets") > 0)
            .then(pl.col("air_yards") / pl.col("targets"))
            .otherwise(None)
            .alias("adot"),
            pl.when(pl.col("_team_carries") > 0)
            .then(pl.col("carries") / pl.col("_team_carries"))
            .otherwise(None)
            .alias("carry_share"),
            pl.col("designed_rush_attempts").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("_team_carries") > 0)
            .then(pl.col("designed_rush_attempts") / pl.col("_team_carries"))
            .otherwise(None)
            .alias("designed_rush_share")
        )
    )

    with_snaps = base.join(
        _snap_counts_by_player_id(snap_counts, players_dim),
        on=["player_id", "season", "week"],
        how="left",
    )

    with_rz = (
        with_snaps.join(rz_touches, on=["player_id", "season", "week"], how="left")
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
            .alias("rz_touch_share"),
            pl.when(pl.col("_team_gz_carries").fill_null(0) > 0)
            .then(pl.col("gz_carries") / pl.col("_team_gz_carries"))
            .otherwise(None)
            .alias("gz_carry_share"),
        )
    )

    return with_rz.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("route_participation"),
        pl.lit(None, dtype=pl.Float64).alias("xfp"),
    ).select(
        "player_id",
        "season",
        "week",
        "team",
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
        "gz_carry_share",
        "designed_rush_attempts",
        "designed_rush_share",
        "route_participation",
        "xfp",
    )


def add_xfp(player_week_usage: pl.DataFrame, ff_opportunity: pl.DataFrame) -> pl.DataFrame:
    """Task 1.2: join ffopportunity's real `total_fantasy_points_exp` onto
    `player_week_usage`'s `xfp` column (null from `build_player_week_usage`
    until this runs).

    `ff_opportunity`'s own `season`/`week` come back from nflreadpy as
    `String`/`Float64` -- confirmed live, yet another source-specific dtype
    quirk (same lesson as `normalize_injuries`'s `Float64` season/week) --
    cast to `Int32` to match `player_week_usage` before joining, or the
    join silently returns zero matches instead of raising (an inner/left
    join on mismatched dtypes doesn't error in polars, it just never
    matches -- exactly the kind of silently-empty result CLAUDE.md warns
    against).
    """
    xfp = ff_opportunity.select(
        "player_id",
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("total_fantasy_points_exp").alias("xfp"),
    )
    return player_week_usage.drop("xfp").join(xfp, on=["player_id", "season", "week"], how="left")


# Real gap found live (task 1.4): nflreadpy's injuries source has real
# report_status/practice_status designations for the entire 2025 season
# but `date_modified` is null for every one of its 6,068 rows -- confirmed
# against a same-day fresh (not stale) fetch, and confirmed the gap is
# upstream in nflreadpy itself, not introduced by `normalize_injuries`.
# Every other season (2015-2024) has `date_modified` fully populated.
# SPEC §6.2 calls this column "essential" for the as_of contract -- without
# it, 2025 can't be walk-forward validated honestly. Confirmed with you:
# fall back to a documented heuristic rather than leaving 2025 unusable or
# guessing silently.
#
# The heuristic is grounded in the real 2015-2024 data, not assumed: of
# the 24,209 real rows with both a `report_status` and a real
# `date_modified`, 83% land on a Friday (matching the NFL's real "final
# injury report" cadence), and the single most common hour across all of
# them is 12:00 UTC. `INJURY_REPORT_LAG_DAYS = 2` (published 2 days before
# kickoff) generalises correctly across Thursday/Sunday/Monday games
# without hardcoding "Friday" specifically, since it's always relative to
# that team's own game that week, not the calendar.
INJURY_REPORT_LAG_DAYS = 2
INJURY_REPORT_FALLBACK_HOUR_UTC = 12


def backfill_injury_date_modified(injuries: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Task 1.4: fill `date_modified` for rows where nflreadpy's source has
    none (2025, confirmed -- see the module-level comment above
    `INJURY_REPORT_LAG_DAYS`) from that team's own real game date that
    week, minus `INJURY_REPORT_LAG_DAYS` at `INJURY_REPORT_FALLBACK_HOUR_UTC`.

    Adds `date_modified_is_estimated` so this approximation is never
    silently indistinguishable from a real, sourced timestamp -- task
    1.5's as_of logic (or anything else consuming this column) can choose
    to exclude or down-weight estimated rows rather than trust them
    blindly.

    Rows whose `team` has no matching game in `schedule` that week (should
    not happen for real data, but not assumed) keep their original
    `date_modified` -- null stays null rather than guessed with no game
    date to anchor to.
    """
    home_side = schedule.select("season", "week", pl.col("home_team").alias("team"), "gameday")
    away_side = schedule.select("season", "week", pl.col("away_team").alias("team"), "gameday")
    team_gameday = pl.concat([home_side, away_side], how="vertical_relaxed")

    with_gameday = injuries.join(team_gameday, on=["season", "week", "team"], how="left")

    fallback = (
        pl.col("gameday")
        .str.strptime(pl.Date, "%Y-%m-%d")
        .cast(pl.Datetime(time_unit="us"))
        .dt.offset_by(f"-{INJURY_REPORT_LAG_DAYS}d")
        .dt.offset_by(f"{INJURY_REPORT_FALLBACK_HOUR_UTC}h")
        .dt.replace_time_zone("UTC")
    )

    return (
        with_gameday.with_columns(
            pl.col("date_modified").is_null().alias("date_modified_is_estimated"),
            pl.coalesce([pl.col("date_modified"), fallback]).alias("date_modified"),
        )
        .drop("gameday")
        .select(injuries.columns + ["date_modified_is_estimated"])
    )


__all__ = [
    "GOAL_ZONE_YARDLINE",
    "INJURY_REPORT_FALLBACK_HOUR_UTC",
    "INJURY_REPORT_LAG_DAYS",
    "NEUTRAL_SCRIPT_QUARTERS",
    "NEUTRAL_SCRIPT_SCORE_MARGIN",
    "OPPONENT_ADJUSTMENT_RECENCY_SPAN",
    "OPPONENT_ADJUSTMENT_SHRINKAGE_K",
    "QB_PASSING_GROUP",
    "RED_ZONE_YARDLINE",
    "RIDGE_ALPHA",
    "SCORE_DIFFERENTIAL_CAP",
    "SKILL_POSITIONS",
    "YDSTOGO_CAP",
    "add_neutral_pace",
    "add_opponent_adjustment",
    "add_proe",
    "add_schedule_context",
    "add_xfp",
    "backfill_injury_date_modified",
    "build_defense_position_allowed",
    "build_player_week_stats",
    "build_player_week_usage",
    "build_team_week_context",
]
