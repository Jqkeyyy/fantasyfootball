"""In-season prediction logging (`SPEC-ADDENDUM-05.md` §B; TASKS.md task
3.8 -- renumbered from the addendum's own literal "3.7", already taken by
the DST/K streamer tab).

**Why this exists, and why it cannot wait:** weekly consensus projections
are not re-fetchable. A source's own live page shows whatever is current
*right now*; last week's real numbers are gone once this week's replace
them. `ADDENDUM-02 §C.3` already flags this for ADP; it is worse here,
since ADP at least has archives. If this isn't captured as the season
runs, every offseason task in `ADDENDUM-05` §C-§E is impossible in
January -- there would be nothing to train a residual model against.

**Per-source, not just the aggregate -- and split into two real
namespaces by unit, not one.** B3 (`models.baselines.fetch_b3_for_week`)
is one real, already-confirmed-weekly PER-WEEK source (FantasyPros' own
R2P consensus, mined via git-history for the historical archive, fetched
live here for the current week) -- logged under `weekly_source_points`.
The other six sources this project already knows how to fetch (the same
seven `draft.board` uses for the preseason draft board) were re-fetched
here on the same weekly cadence and found, live, against real 2025
week-10 data, to return real **season-long point totals**, not weekly
ones -- confirmed by value, not just inferred from the URL (a real QB
row showed `fantasypros`/`model_mean`/`b2_mean` all agreeing around
15-18 points while `espn`/`cbs`/`fantasysharks`/`footballguys` all
returned 210-300, the right order of magnitude for a full season, not
one week). `apply_league_scoring` scores these correctly; the raw stat
columns underneath are simply season projections, exactly what
`draft.board`'s own preseason use needs and exactly wrong to average
into a weekly consensus number. They are logged separately, under
`season_source_points` -- **not dropped**: a season projection that
updates as the real season plays out is a genuine, real rest-of-season
signal (`SPEC-ADDENDUM-04.md` §D), and it is exactly as perishable as
the weekly numbers. `dispersion`/`n_sources` are computed from
`weekly_source_points` only (today, in practice, that means `fantasypros`
alone -- an honest reflection of how few genuinely-weekly sources this
project has, not a bug); a parallel `season_dispersion`/`n_season_sources`
covers the season set as its own, legitimate, separately-scaled
signal, never mixed into the weekly ones.

**Whether a `season_source_points` value is a genuine rest-of-season
forward signal, or a stale full-season number that still bakes in
points a player already scored, is a real, open question `check_sources`
answers from real weekly evidence (§B.2's own resolution mechanism,
extended): does that source's own real value for the same real players
decline materially week over week (a real ROS signal) or stay flat (a
frozen preseason snapshot, still worth logging, just not a forward
signal) -- resolves itself by about Week 4, the same real-evidence
principle already used for `refresh_status`.**

**`weekly_source_points["fantasypros"]` is `models.baselines
.fetch_b3_for_week`'s own value, not `draft.board`'s separate preseason
FantasyPros ECR source.** Two different real FantasyPros mechanisms
exist in this codebase already -- the weekly R2P consensus (`b3_mean`'s
own source, already proven weekly, and now doubly confirmed: it never
calls `apply_league_scoring`/`fetch_espn`/etc. at all, an entirely
separate pipeline from the six season-scoped sources -- verified before
this fix, since a units bug in the weekly log would say nothing about
whether `b3_mean` itself, or the 2021-2025 historical archive that
calibrated the distribution wrapper, SPEC-ADDENDUM-04.md §D, were
contaminated the same way; they are not) and a preseason-only Expert
Consensus Ranking mirror (`ingest.rankings.fetch_fantasypros`, rank-only,
mapped onto points via a reference curve for the draft board). Logging
the already-confirmed-weekly one under the `fantasypros` key, rather
than inventing an eighth source or duplicating `b3_mean` under two
different meanings, is the only choice that doesn't conflate the two.

**Nothing about `draft.board`'s own preseason ingestion changes here.**
The six sources return season totals because the draft board needs
season totals -- that path is correct for its own real purpose (the
real Aug 22 draft) and stays untouched. The units mismatch is a property
of using those same fetchers inside a *weekly* log, not a bug in the
fetchers themselves.

Deliberately calls `ingest.rankings`'s own public `fetch_*`/`normalize_*`
functions directly, not `draft.board`'s private `_POINT_SOURCE_FETCHERS`/
`_RANK_SOURCE_FETCHERS` dicts -- this module needs the real raw `Path`
each fetch returns (to hash the actual payload bytes for
`payload_sha256`), which those wrapper lambdas discard internally after
one inline read.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ffapp.config import CONFIG_DIR, LightGBMSettings, Settings
from ffapp.ids import mapping
from ffapp.ingest import rankings
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import baselines, predict
from ffapp.projections.aggregate import (
    add_join_key,
    apply_league_scoring,
    build_reference_curve,
    map_ranks_to_points,
    rank_within_position,
)

SOURCE_NAMES: tuple[str, ...] = (
    "espn",
    "cbs",
    "fantasysharks",
    "fftoday",
    "footballguys",
    "draftsharks",
    "fantasypros",
)
POINT_SOURCES: tuple[str, ...] = ("espn", "cbs", "fantasysharks", "fftoday")
RANK_SOURCES: tuple[str, ...] = ("footballguys", "draftsharks")

# Confirmed live against real 2025 week-10 data, not inferred -- see
# module docstring. fantasypros is the one source whose own value is a
# genuine single-week number; the other six return real season-long
# totals through the exact same code path draft.board's own preseason
# board correctly relies on.
WEEKLY_SOURCES: tuple[str, ...] = ("fantasypros",)
SEASON_SOURCES: tuple[str, ...] = (
    "espn",
    "cbs",
    "fantasysharks",
    "fftoday",
    "footballguys",
    "draftsharks",
)

REFRESH_STATUSES: tuple[str, ...] = ("weekly_confirmed", "frozen", "manual", "unknown")
RUN_LABELS: tuple[str, ...] = ("tuesday", "thursday", "sunday")
SEASON_TRENDS: tuple[str, ...] = ("declining", "flat", "insufficient_data")

_SOURCE_STATUS_PATH = CONFIG_DIR / "source_refresh_status.yml"

PREDICTION_LOG_SCHEMA = {
    "league_slug": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "run_label": pl.String,
    "as_of_utc": pl.String,
    "player_id": pl.String,
    "position": pl.String,
    "team": pl.String,
    "projection_source": pl.String,
    "b3_mean": pl.Float64,
    "b3_q10": pl.Float64,
    "b3_q25": pl.Float64,
    "b3_q50": pl.Float64,
    "b3_q75": pl.Float64,
    "b3_q90": pl.Float64,
    **{f"weekly_source_points_{name}": pl.Float64 for name in WEEKLY_SOURCES},
    **{f"season_source_points_{name}": pl.Float64 for name in SEASON_SOURCES},
    "n_sources": pl.Int64,
    "dispersion": pl.Float64,
    "n_season_sources": pl.Int64,
    "season_dispersion": pl.Float64,
    "model_mean": pl.Float64,
    "b2_mean": pl.Float64,
    "p_active": pl.Float64,
    "actual_points": pl.Float64,
    "model_version": pl.String,
    "feature_hash": pl.String,
    "git_sha": pl.String,
}

SOURCE_FETCH_SCHEMA = {
    "league_slug": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "run_label": pl.String,
    "source": pl.String,
    "payload_sha256": pl.String,
    "fetched_at_utc": pl.String,
    "refresh_status": pl.String,
    "n_rows_parsed": pl.Int64,
    "fetch_error": pl.String,
}

_EMPTY_POINTS = pl.DataFrame(
    schema={"player_id": pl.String, "position": pl.String, "team": pl.String, "points": pl.Float64}
)


def load_source_refresh_status(path: Path = _SOURCE_STATUS_PATH) -> dict[str, str]:
    """The current best-known `refresh_status` per source
    (`config/source_refresh_status.yml`) -- a real, mutable reference file
    `check_sources` updates from real weekly evidence, not a hardcoded
    guess baked into this module. A source absent from the file (or the
    file itself missing) defaults to `"unknown"`, never a guessed
    positive status."""
    if not path.exists():
        return dict.fromkeys(SOURCE_NAMES, "unknown")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    statuses = dict(raw.get("sources", {}))
    return {name: statuses.get(name, "unknown") for name in SOURCE_NAMES}


def write_source_refresh_status(statuses: dict[str, str], path: Path = _SOURCE_STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sources": statuses}, sort_keys=True), encoding="utf-8")


@dataclass
class SourceFetchResult:
    points: pl.DataFrame  # player_id, position, team, points -- empty on failure
    ranked: (
        pl.DataFrame | None
    )  # position, rank, points -- point sources only, for the reference curve
    payload_sha256: str | None
    fetched_at_utc: str
    n_rows_parsed: int
    fetch_error: str | None


def _resolve_to_player_id(df: pl.DataFrame, players_dim: pl.DataFrame) -> pl.DataFrame:
    """Real canonical `player_id` resolution -- the same `add_join_key` +
    `mapping.dedupe_to_one_row_per_name_position` pattern
    `models.baselines.add_b3_fp_weekly_consensus` already established,
    applied here to each of the other six sources too, so every source in
    this log shares one real player identity, not seven different
    name-matching schemes."""
    with_key = add_join_key(df)
    resolved = mapping.dedupe_to_one_row_per_name_position(players_dim).select(
        "join_key", "player_id"
    )
    return with_key.join(resolved, on="join_key", how="left").drop_nulls("player_id")


def fetch_point_source(
    name: str,
    season: int,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    *,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> SourceFetchResult:
    fetched_at = now.isoformat()
    try:
        if name == "espn":
            path = rankings.fetch_espn(season, offline=offline, settings=settings)
            raw_df = rankings.normalize_espn(json.loads(path.read_text()), season=season)
        elif name == "cbs":
            path = rankings.fetch_cbs(season, offline=offline, settings=settings)
            raw_df = rankings.normalize_cbs(json.loads(path.read_text()), season=season)
        elif name == "fantasysharks":
            path = rankings.fetch_fantasysharks(offline=offline, settings=settings)
            raw_df = rankings.normalize_fantasysharks(json.loads(path.read_text()), season=season)
        elif name == "fftoday":
            path = rankings.fetch_fftoday(season, offline=offline, settings=settings)
            raw_df = rankings.normalize_fftoday(json.loads(path.read_text()), season=season)
        else:  # pragma: no cover -- defensive, POINT_SOURCES is a closed set
            raise ValueError(f"unknown point source {name!r}")

        payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_df.height == 0:
            return SourceFetchResult(
                _EMPTY_POINTS, None, payload_sha256, fetched_at, 0, "0 rows parsed"
            )
        scored = apply_league_scoring(raw_df, scoring_settings)
        resolved = _resolve_to_player_id(scored, players_dim)
        ranked = rank_within_position(scored).select("position", "rank", "points")
        return SourceFetchResult(
            resolved.select("player_id", "position", "team", "points"),
            ranked,
            payload_sha256,
            fetched_at,
            raw_df.height,
            None,
        )
    except Exception as exc:
        return SourceFetchResult(_EMPTY_POINTS, None, None, fetched_at, 0, str(exc))


def fetch_rank_source(
    name: str,
    season: int,
    reference_curve: pl.DataFrame,
    players_dim: pl.DataFrame,
    *,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> SourceFetchResult:
    fetched_at = now.isoformat()
    try:
        if name == "footballguys":
            path = rankings.fetch_footballguys(offline=offline, settings=settings)
            raw_df = rankings.normalize_footballguys(
                path.read_text(encoding="utf-8"), season=season
            )
        elif name == "draftsharks":
            path = rankings.fetch_draftsharks(offline=offline, settings=settings)
            raw_df = rankings.normalize_draftsharks(json.loads(path.read_text()), season=season)
        else:  # pragma: no cover -- defensive, RANK_SOURCES is a closed set
            raise ValueError(f"unknown rank source {name!r}")

        payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_df.height == 0:
            return SourceFetchResult(
                _EMPTY_POINTS, None, payload_sha256, fetched_at, 0, "0 rows parsed"
            )
        mapped = map_ranks_to_points(raw_df, reference_curve)
        resolved = _resolve_to_player_id(mapped, players_dim)
        return SourceFetchResult(
            resolved.select("player_id", "position", "team", "points"),
            None,
            payload_sha256,
            fetched_at,
            raw_df.height,
            None,
        )
    except Exception as exc:
        return SourceFetchResult(_EMPTY_POINTS, None, None, fetched_at, 0, str(exc))


def _fetch_fantasypros(
    season: int,
    week: int,
    cutoff_utc: str,
    players_dim: pl.DataFrame,
    *,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> SourceFetchResult:
    fetched_at = now.isoformat()
    try:
        # `models.baselines.fetch_b3_for_week`'s own commit-selection
        # sequence, replicated here rather than called -- calling it AND
        # fetching the commit list again here for the hash means two real
        # paginated GitHub API round-trips per source per run
        # (`fetch_fp_weekly_commits` is NOT a same-call idempotent no-op;
        # a live 403 rate-limit on `api.github.com` was hit this session
        # from exactly that doubling). One real fetch feeds both the hash
        # and `add_b3_fp_weekly_consensus` -- the same function
        # `fetch_b3_for_week` itself calls last.
        commits_path = rankings.fetch_fp_weekly_commits(offline=offline, settings=settings)
        commits = json.loads(commits_path.read_text())["commits"]
        sha = rankings.select_commit_before(commits, cutoff_utc)
        if sha is None:
            return SourceFetchResult(_EMPTY_POINTS, None, None, fetched_at, 0, "0 rows resolved")

        snapshot_path = rankings.fetch_fp_weekly_snapshot(sha, offline=offline, settings=settings)
        payload_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        fp_weekly = rankings.normalize_fp_weekly(
            snapshot_path.read_text(), season=season, week=week
        )
        b3 = baselines.add_b3_fp_weekly_consensus(fp_weekly, players_dim)
        if b3.height == 0:
            return SourceFetchResult(
                _EMPTY_POINTS, None, payload_sha256, fetched_at, 0, "0 rows resolved"
            )
        with_position = b3.join(
            players_dim.select("player_id", "position").unique(subset=["player_id"]),
            on="player_id",
            how="left",
        ).with_columns(pl.lit(None, dtype=pl.String).alias("team"))
        points = with_position.rename({"b3_points": "points"}).select(
            "player_id", "position", "team", "points"
        )
        return SourceFetchResult(points, None, payload_sha256, fetched_at, b3.height, None)
    except Exception as exc:
        return SourceFetchResult(_EMPTY_POINTS, None, None, fetched_at, 0, str(exc))


def fetch_all_sources(
    season: int,
    week: int,
    cutoff_utc: str,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    *,
    league_slug: str,
    run_label: str,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Real live fetch of all seven sources -- a genuine network call
    each, regardless of that source's current believed `refresh_status`
    (see module docstring). Returns `{source_name: points_df}` (each
    `player_id`/`position`/`team`/`points`) plus the real `source_fetches`
    metadata rows (`check_sources`'s own real input). A single source
    failing never sinks the whole week's log -- caught and recorded as
    `fetch_error`, matching `draft.board.fetch_point_sources`'s own
    graceful-degradation precedent for the same seven sources.
    """
    statuses = load_source_refresh_status()
    points: dict[str, pl.DataFrame] = {}
    fetch_rows: list[dict[str, Any]] = []

    point_results: dict[str, SourceFetchResult] = {}
    for name in POINT_SOURCES:
        result = fetch_point_source(
            name, season, scoring_settings, players_dim, offline=offline, settings=settings, now=now
        )
        point_results[name] = result
        points[name] = result.points

    real_ranked = [
        r.ranked for r in point_results.values() if r.ranked is not None and r.ranked.height > 0
    ]
    reference_curve = (
        build_reference_curve(real_ranked)
        if real_ranked
        else pl.DataFrame(
            schema={"position": pl.String, "rank": pl.Int64, "ref_points": pl.Float64}
        )
    )

    rank_results: dict[str, SourceFetchResult] = {}
    for name in RANK_SOURCES:
        result = fetch_rank_source(
            name, season, reference_curve, players_dim, offline=offline, settings=settings, now=now
        )
        rank_results[name] = result
        points[name] = result.points

    fp_result = _fetch_fantasypros(
        season, week, cutoff_utc, players_dim, offline=offline, settings=settings, now=now
    )
    points["fantasypros"] = fp_result.points

    for name, result in {**point_results, **rank_results, "fantasypros": fp_result}.items():
        fetch_rows.append(
            {
                "league_slug": league_slug,
                "season": season,
                "week": week,
                "run_label": run_label,
                "source": name,
                "payload_sha256": result.payload_sha256,
                "fetched_at_utc": result.fetched_at_utc,
                "refresh_status": statuses.get(name, "unknown"),
                "n_rows_parsed": result.n_rows_parsed,
                "fetch_error": result.fetch_error,
            }
        )

    fetch_df = pl.DataFrame(fetch_rows, schema=SOURCE_FETCH_SCHEMA)
    return points, fetch_df


def _dispersion(values: list[float]) -> float:
    """Population stdev -- same real definition
    `projections.aggregate._dispersion` already uses for the draft
    board's own consensus confidence, applied here to whichever of the
    seven sources actually returned a value for this player-week."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def build_prediction_log(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    season: int,
    week: int,
    run_label: str,
    *,
    league_slug: str,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings,
    quantile_alphas: Sequence[float],
    b3_historical: pl.DataFrame,
    code_version: str | None,
    now: datetime,
    offline: bool | None,
    settings: Settings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The real per-week log build: three real `models.predict.project_week`
    calls (`"direct"` for `model_mean` -- logged even though it isn't
    shipping, per `SPEC-ADDENDUM-05.md` §B.2's own explicit instruction --
    `"baseline_b2"` for `b2_mean`, `"consensus_b3"` for the real shipped
    `b3_mean`/`b3_q10`..`q90` and this row's own real `model_version`/
    `feature_hash`/`git_sha` provenance), plus a real live fetch of all
    seven ranking sources (`fetch_all_sources`). Empty (not a crash) if
    there's no real row universe for `(season, week)` yet or too little
    training data -- `project_week`'s own honest-empty convention,
    propagated here rather than partially built.

    Every join here is keyed on `player_id` and built from real
    DataFrames throughout (never a bare Series re-attached positionally
    across a separately-ordered join) -- the same row-alignment
    discipline `models.predict.project_week`'s own real bug fix
    established earlier this session.
    """
    cutoff_row = schedule.filter((pl.col("season") == season) & (pl.col("week") == week))
    if cutoff_row.is_empty():
        return pl.DataFrame(schema=PREDICTION_LOG_SCHEMA), pl.DataFrame(schema=SOURCE_FETCH_SCHEMA)
    cutoff_utc = str(cutoff_row["kickoff_utc"].min())

    direct = predict.project_week(
        features,
        season,
        week,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params,
        code_version=code_version,
        now=now,
        quantile_alphas=quantile_alphas,
        projection_source="direct",
    )
    b2 = predict.project_week(
        features,
        season,
        week,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params,
        code_version=code_version,
        now=now,
        quantile_alphas=quantile_alphas,
        projection_source="baseline_b2",
    )
    b3 = predict.project_week(
        features,
        season,
        week,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params,
        code_version=code_version,
        now=now,
        quantile_alphas=quantile_alphas,
        projection_source="consensus_b3",
        players_dim=players_dim,
        b3_historical=b3_historical,
        offline=offline,
        settings=settings,
    )
    if direct.is_empty() or b2.is_empty() or b3.is_empty():
        return pl.DataFrame(schema=PREDICTION_LOG_SCHEMA), pl.DataFrame(schema=SOURCE_FETCH_SCHEMA)

    per_source_points, fetch_df = fetch_all_sources(
        season,
        week,
        cutoff_utc,
        scoring_settings,
        players_dim,
        league_slug=league_slug,
        run_label=run_label,
        offline=offline,
        settings=settings,
        now=now,
    )

    row_universe = features.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("position").is_in(SKILL_POSITIONS)
    ).select("player_id", "position", "team")

    work = (
        row_universe.join(
            b3.select(
                "player_id",
                "p_active",
                pl.col("mean").alias("b3_mean"),
                pl.col("q10").alias("b3_q10"),
                pl.col("q25").alias("b3_q25"),
                pl.col("q50").alias("b3_q50"),
                pl.col("q75").alias("b3_q75"),
                pl.col("q90").alias("b3_q90"),
                "model_version",
                "feature_hash",
                pl.col("git_commit").alias("git_sha"),
                "as_of_utc",
                "projection_source",
            ),
            on="player_id",
            how="left",
        )
        .join(
            direct.select("player_id", pl.col("mean").alias("model_mean")),
            on="player_id",
            how="left",
        )
        .join(b2.select("player_id", pl.col("mean").alias("b2_mean")), on="player_id", how="left")
    )
    for name in WEEKLY_SOURCES:
        work = work.join(
            per_source_points[name].select(
                "player_id", pl.col("points").alias(f"weekly_source_points_{name}")
            ),
            on="player_id",
            how="left",
        )
    for name in SEASON_SOURCES:
        work = work.join(
            per_source_points[name].select(
                "player_id", pl.col("points").alias(f"season_source_points_{name}")
            ),
            on="player_id",
            how="left",
        )

    # dispersion/n_sources: WEEKLY sources only -- a real, comparable
    # single-week signal. season_dispersion/n_season_sources: the season
    # set, its own separate scale, never mixed with the weekly one (see
    # module docstring -- this split exists because averaging a real
    # season total with a real weekly number produced ~90pp of garbage
    # "dispersion" on live 2025 week-10 data before this fix).
    weekly_cols = [f"weekly_source_points_{name}" for name in WEEKLY_SOURCES]
    season_cols = [f"season_source_points_{name}" for name in SEASON_SOURCES]
    work = (
        work.with_columns(
            pl.concat_list(weekly_cols).list.drop_nulls().alias("_weekly_values"),
            pl.concat_list(season_cols).list.drop_nulls().alias("_season_values"),
        )
        .with_columns(
            pl.col("_weekly_values").list.len().alias("n_sources"),
            pl.col("_weekly_values")
            .map_elements(lambda values: _dispersion(list(values)), return_dtype=pl.Float64)
            .alias("dispersion"),
            pl.col("_season_values").list.len().alias("n_season_sources"),
            pl.col("_season_values")
            .map_elements(lambda values: _dispersion(list(values)), return_dtype=pl.Float64)
            .alias("season_dispersion"),
        )
        .drop("_weekly_values", "_season_values")
    )

    work = work.with_columns(
        pl.lit(league_slug).alias("league_slug"),
        pl.lit(season).alias("season"),
        pl.lit(week).alias("week"),
        pl.lit(run_label).alias("run_label"),
        pl.lit(None, dtype=pl.Float64).alias("actual_points"),
    )

    return work.select(list(PREDICTION_LOG_SCHEMA)), fetch_df


def _log_dir(settings: Settings, league_slug: str) -> Path:
    return settings.data_root / "outputs" / league_slug / "prediction_log"


def _week_path(settings: Settings, league_slug: str, season: int, week: int) -> Path:
    return _log_dir(settings, league_slug) / f"season={season}" / f"week={week:02d}.parquet"


def write_prediction_log(
    rows: pl.DataFrame,
    fetch_rows: pl.DataFrame,
    season: int,
    week: int,
    run_label: str,
    *,
    league_slug: str,
    settings: Settings,
) -> Path:
    """Upsert by `(season, week, run_label)` -- CLAUDE.md's own "re-running
    for the same scope overwrites cleanly" idempotent-ingest rule, applied
    here for the first time to a REAL, committed-to-git artefact (§B.3:
    the same non-reproducibility reasoning as the rankings gitignore
    exception -- weekly consensus is not re-fetchable, losing this log
    would be unrecoverable in a way a trained model is not). Writes the
    prediction rows to `season=NNNN/week=NN.parquet` and the source-fetch
    metadata to a separate, small `source_fetches.parquet` (one growing
    file across the whole season -- `check_sources`'s own real input),
    plus refreshes the `latest` pointer.
    """
    week_path = _week_path(settings, league_slug, season, week)
    week_path.parent.mkdir(parents=True, exist_ok=True)
    if week_path.exists():
        existing = pl.read_parquet(week_path)
        existing = existing.filter(pl.col("run_label") != run_label)
        combined = pl.concat([existing, rows], how="vertical_relaxed")
    else:
        combined = rows
    combined.write_parquet(week_path)

    latest_path = _log_dir(settings, league_slug) / "latest.parquet"
    latest_path.write_bytes(week_path.read_bytes())

    fetches_path = _log_dir(settings, league_slug) / "source_fetches.parquet"
    if fetches_path.exists():
        existing_fetches = pl.read_parquet(fetches_path)
        key = ["season", "week", "run_label", "source"]
        existing_fetches = existing_fetches.join(
            fetch_rows.select(key).unique(), on=key, how="anti"
        )
        combined_fetches = pl.concat([existing_fetches, fetch_rows], how="vertical_relaxed")
    else:
        combined_fetches = fetch_rows
    combined_fetches.write_parquet(fetches_path)

    return week_path


class MissingBackfillError(Exception):
    """A prior week's `actual_points` was never backfilled -- warned
    loudly (SPEC-ADDENDUM-05.md §B.4: "should warn loudly at the next
    run rather than leaving silent nulls that look like zeros"), not
    silently skipped."""


def backfill_actual_points(
    features: pl.DataFrame, season: int, week: int, *, league_slug: str, settings: Settings
) -> pl.DataFrame:
    """Fills `actual_points` for every real row logged for `(season,
    week)`, from `features`' own already-scored `target` column (task
    1.9 -- the same real actual-outcome column every other model in this
    project is evaluated against). Updates every real `run_label` row
    logged for that week, in place (upsert by `(season, week,
    run_label)`, matching `write_prediction_log`'s own convention).
    """
    week_path = _week_path(settings, league_slug, season, week)
    if not week_path.exists():
        raise MissingBackfillError(
            f"No prediction log found for season={season} week={week} at {week_path} -- "
            "nothing to backfill. Was this week ever logged?"
        )
    logged = pl.read_parquet(week_path)
    actuals = features.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "player_id", pl.col("target").alias("_actual")
    )
    filled = (
        logged.drop("actual_points")
        .join(actuals, on="player_id", how="left")
        .rename({"_actual": "actual_points"})
        .select(logged.columns)
    )
    filled.write_parquet(week_path)

    latest_path = _log_dir(settings, league_slug) / "latest.parquet"
    if latest_path.exists():
        latest = pl.read_parquet(latest_path)
        if not latest.is_empty() and latest["week"][0] == week and latest["season"][0] == season:
            latest_path.write_bytes(week_path.read_bytes())

    return filled


_TREND_MIN_WEEKS = 3
_TREND_DECLINE_THRESHOLD = -0.05  # relative slope per week; more negative = declining


def season_source_trend(log_dir: Path) -> pl.DataFrame:
    """Real per-season-source trend detection: does this source's own
    real mean value (across every real player logged that week) decline
    materially week over week, or stay flat? A declining season-total
    number is real evidence it updates to reflect games already played
    (a genuine rest-of-season signal, `SPEC-ADDENDUM-04.md` §D); a flat
    one is real evidence it's a static preseason snapshot never revised
    (still worth logging -- a real, if less useful, reference point).

    Reads every real logged week file for this league directly (not
    `source_fetches.parquet`, which has no per-player point values),
    computes each season source's own real mean across all logged
    players per week, and fits a simple linear trend (relative slope per
    week) across at least `_TREND_MIN_WEEKS` real weeks. Fewer real
    weeks than that: `insufficient_data`, an honest gap, not a guess.
    """
    week_files = sorted(log_dir.glob("season=*/week=*.parquet"))
    if not week_files:
        return pl.DataFrame(schema={"source": pl.String, "n_weeks": pl.Int64, "trend": pl.String})

    season_cols = [f"season_source_points_{name}" for name in SEASON_SOURCES]
    weekly_means: list[dict[str, Any]] = []
    for path in week_files:
        df = pl.read_parquet(path)
        if df.is_empty() or "season" not in df.columns:
            continue
        season = df["season"][0]
        week = df["week"][0]
        for name, col in zip(SEASON_SOURCES, season_cols, strict=True):
            if col not in df.columns:
                continue
            values = df[col].drop_nulls()
            if values.is_empty():
                continue
            weekly_means.append(
                {
                    "source": name,
                    "season": season,
                    "week": week,
                    "mean_points": float(values.to_numpy().mean()),
                }
            )

    if not weekly_means:
        return pl.DataFrame(schema={"source": pl.String, "n_weeks": pl.Int64, "trend": pl.String})

    means_df = pl.DataFrame(weekly_means).unique(subset=["source", "season", "week"], keep="last")

    rows: list[dict[str, Any]] = []
    for name in SEASON_SOURCES:
        pos = means_df.filter(pl.col("source") == name).sort(["season", "week"])
        n_weeks = pos.height
        if n_weeks < _TREND_MIN_WEEKS:
            rows.append({"source": name, "n_weeks": n_weeks, "trend": "insufficient_data"})
            continue
        x = np.arange(n_weeks, dtype=float)
        y = pos["mean_points"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
        relative_slope = slope / y[0] if y[0] != 0 else 0.0
        trend = "declining" if relative_slope <= _TREND_DECLINE_THRESHOLD else "flat"
        rows.append({"source": name, "n_weeks": n_weeks, "trend": trend})

    return pl.DataFrame(rows, schema={"source": pl.String, "n_weeks": pl.Int64, "trend": pl.String})


def check_sources(
    *, league_slug: str, settings: Settings, path: Path = _SOURCE_STATUS_PATH
) -> pl.DataFrame:
    """§B.2's own real resolution mechanism, in two parts.

    **Weekly-source refresh confirmation:** reads every real logged
    `source_fetches` row, counts how many DISTINCT `payload_sha256`
    values appeared per source across every real logged week, and
    promotes `unknown`/re-confirms non-`frozen`/`manual` statuses from
    that real evidence -- a source with exactly one distinct hash across
    at least 3 real logged weeks is `frozen` (it never actually
    changed); a source whose hash changed at least once is
    `weekly_confirmed`. Sources already marked `frozen`/`manual` from a
    real, resolved static check (this session's own URL inspection) are
    left alone -- real evidence can only add confidence, not override an
    already-confirmed real fact. Writes the updated statuses back to
    `config/source_refresh_status.yml`.

    **Season-source trend detection** (`season_source_trend`): a
    separate, real question for `SEASON_SOURCES` specifically -- does
    the real value decline (a genuine rest-of-season signal) or stay
    flat (a static full-season snapshot)? Reported alongside, in the
    same returned frame, under `season_trend`/`season_n_weeks` -- `null`
    for `WEEKLY_SOURCES`, which this question doesn't apply to.
    """
    fetches_path = _log_dir(settings, league_slug) / "source_fetches.parquet"
    if not fetches_path.exists():
        return pl.DataFrame(
            schema={
                "source": pl.String,
                "n_weeks_logged": pl.Int64,
                "n_distinct_hashes": pl.Int64,
                "status": pl.String,
                "season_trend": pl.String,
                "season_n_weeks": pl.Int64,
            }
        )
    fetches = pl.read_parquet(fetches_path).drop_nulls("payload_sha256")
    summary = fetches.group_by("source").agg(
        pl.col("week").n_unique().alias("n_weeks_logged"),
        pl.col("payload_sha256").n_unique().alias("n_distinct_hashes"),
    )

    trend_df = season_source_trend(_log_dir(settings, league_slug))
    trend_by_source = {row["source"]: row for row in trend_df.iter_rows(named=True)}

    current = load_source_refresh_status(path)
    updated: dict[str, str] = dict(current)
    rows: list[dict[str, Any]] = []
    for row in summary.iter_rows(named=True):
        name = row["source"]
        prior_status = current.get(name, "unknown")
        if prior_status in ("frozen", "manual"):
            new_status = prior_status  # a resolved static fact isn't overridden by more evidence
        elif row["n_weeks_logged"] >= 3:
            new_status = "frozen" if row["n_distinct_hashes"] == 1 else "weekly_confirmed"
        else:
            new_status = prior_status
        updated[name] = new_status
        trend_row = trend_by_source.get(name)
        rows.append(
            {
                "source": name,
                "n_weeks_logged": row["n_weeks_logged"],
                "n_distinct_hashes": row["n_distinct_hashes"],
                "status": new_status,
                "season_trend": trend_row["trend"] if trend_row else None,
                "season_n_weeks": trend_row["n_weeks"] if trend_row else None,
            }
        )

    write_source_refresh_status(updated, path)
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "n_weeks_logged": pl.Int64,
            "n_distinct_hashes": pl.Int64,
            "status": pl.String,
            "season_trend": pl.String,
            "season_n_weeks": pl.Int64,
        },
    )


__all__ = [
    "MissingBackfillError",
    "POINT_SOURCES",
    "PREDICTION_LOG_SCHEMA",
    "RANK_SOURCES",
    "REFRESH_STATUSES",
    "RUN_LABELS",
    "SOURCE_FETCH_SCHEMA",
    "SOURCE_NAMES",
    "SourceFetchResult",
    "backfill_actual_points",
    "build_prediction_log",
    "check_sources",
    "fetch_all_sources",
    "fetch_point_source",
    "fetch_rank_source",
    "load_source_refresh_status",
    "season_source_trend",
    "write_prediction_log",
    "write_source_refresh_status",
]
