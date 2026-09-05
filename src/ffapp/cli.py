import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import typer

from ffapp import __version__
from ffapp.cache import registry as cache_registry
from ffapp.cache.offline import is_offline
from ffapp.config import (
    Settings,
    load_all_leagues,
    load_league,
    load_primary_league,
    load_ros_calibration,
    load_settings,
)
from ffapp.draft import board as draft_board
from ffapp.draft import export as draft_export
from ffapp.draft import replay as draft_replay
from ffapp.env import load_env
from ffapp.evaluation import backtest
from ffapp.evaluation import metrics as evaluation_metrics
from ffapp.evaluation import report as evaluation_report
from ffapp.ids import mapping
from ffapp.ingest import nflverse, rankings, sleeper
from ffapp.league_format import LeagueFormat, parse_league_format
from ffapp.models import availability, baselines, points, predict, predict_ros, ros_consensus
from ffapp.scoring import golden
from ffapp.sim import injury
from ffapp.tools import prediction_log, ros_aggregate, ros_rankings, sos, waivers

load_env()

app = typer.Typer(name="ffapp", help="Fantasy football decision-support CLI.")
ingest_app = typer.Typer(name="ingest", help="Ingest raw data from external sources.")
cache_app = typer.Typer(name="cache", help="Manage the offline data cache (SPEC-ADDENDUM-02.md).")
ids_app = typer.Typer(name="ids", help="Cross-source player id resolution (SPEC.md §7).")
scoring_app = typer.Typer(name="scoring", help="League scoring engine (SPEC.md §8).")
draft_app = typer.Typer(name="draft", help="Draft board and draft-day support (SPEC.md §9).")
log_app = typer.Typer(name="log", help="In-season prediction logging (SPEC-ADDENDUM-05.md §B).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(cache_app, name="cache")
app.add_typer(ids_app, name="ids")
app.add_typer(scoring_app, name="scoring")
app.add_typer(draft_app, name="draft")
app.add_typer(log_app, name="log")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ffapp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Fantasy football decision-support CLI."""


@ingest_app.command("sleeper")
def ingest_sleeper(
    season: int = typer.Option(..., "--season", help="NFL season to discover leagues for."),
    discover: bool = typer.Option(
        False,
        "--discover",
        help="Enumerate every league on the account and write a config stub per league.",
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Resolve username -> user_id -> leagues and write config/leagues/<slug>.yml stubs."""
    if not discover:
        typer.echo(
            "ffapp ingest sleeper currently only supports --discover "
            "(enumerate every league on the account).",
            err=True,
        )
        raise typer.Exit(code=1)

    if is_offline(offline):
        typer.echo(
            "League discovery needs live network (SPEC-ADDENDUM-02.md §E, Group 3). "
            "Re-run with --no-offline on an unrestricted network.",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = load_settings()
    discovered = cache_registry.discover_leagues(season, settings=settings)
    for league in discovered:
        typer.echo(f"  {league.slug} -> {league.path}")
    typer.echo(
        f"Discovered {len(discovered)} league(s). Set is_primary: true by hand on the "
        "one you want commands to default to."
    )


@ingest_app.command("rankings")
def ingest_rankings(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to the league's own configured season."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Refresh every rankings/ADP source's raw cache (SPEC-ADDENDUM-03.md
    §E's "morning of" runbook step: `ffapp ingest rankings --no-offline`).
    Fetches every per-stat source, every ranks-only source, and ADP, with
    the same per-source graceful degradation `ffapp draft board` already
    uses -- one source failing doesn't block the others. Doesn't assemble
    a board itself; run `ffapp draft board` after this to build one from
    the freshly-refreshed cache.
    """
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else league_config.season
    league_format = parse_league_format(league_config)

    point_sources = draft_board.fetch_point_sources(
        resolved_season, offline=offline, settings=settings
    )
    n_point_sources = len(draft_board.POINT_SOURCE_NAMES)
    typer.echo(f"Point sources: {len(point_sources)}/{n_point_sources} refreshed")

    rank_sources = draft_board.fetch_rank_sources(
        resolved_season, offline=offline, settings=settings
    )
    n_rank_sources = len(draft_board.RANK_SOURCE_NAMES)
    typer.echo(f"Rank sources: {len(rank_sources)}/{n_rank_sources} refreshed")

    try:
        rankings.fetch_adp(
            resolved_season, teams=league_format.n_teams, offline=offline, settings=settings
        )
        typer.echo("ADP: refreshed")
    except Exception as exc:
        typer.echo(f"ADP: failed to refresh ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    if not point_sources:
        typer.echo(
            "Every per-stat rankings source failed -- ffapp draft board will not be able "
            "to build a board from this cache.",
            err=True,
        )
        raise typer.Exit(code=1)


@cache_app.command("warm")
def cache_warm(
    season: int = typer.Option(..., "--season"),
    all_leagues: bool = typer.Option(
        False, "--all-leagues", help="Warm every league on the account."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Pull and archive raw Sleeper data (SPEC-ADDENDUM-02.md §B)."""
    if not all_leagues:
        typer.echo("ffapp cache warm currently only supports --all-leagues.", err=True)
        raise typer.Exit(code=1)

    if is_offline(offline):
        typer.echo("cache warm needs live network. Re-run with --no-offline.", err=True)
        raise typer.Exit(code=1)

    settings = load_settings()
    cache_registry.warm_sleeper(season, settings=settings)
    typer.echo("Cache warmed.")


@cache_app.command("status")
def cache_status_command() -> None:
    """Print every cached artefact with its age and staleness verdict."""
    settings = load_settings()
    rows = cache_registry.cache_status(settings)
    if not rows:
        typer.echo("Nothing cached yet. Run `ffapp cache warm`.")
        return
    for row in rows:
        typer.echo(f"{row['artifact']:40s} {row['verdict']:9s} age={row['age_hours']}h")


@cache_app.command("verify")
def cache_verify_command(
    for_task: str = typer.Option(..., "--for-task", help="TASKS.md task id, e.g. 0.7"),
) -> None:
    """Check whether the cache can satisfy a task's data needs without network."""
    settings = load_settings()
    try:
        results = cache_registry.cache_verify(for_task, settings=settings)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    missing = [req for req, ok in results if not ok]
    for req, ok in results:
        status = "OK" if ok else "MISSING"
        typer.echo(f"[{status}] {req.description}")
    if missing:
        typer.echo("Run to fetch missing artefacts:")
        for req in missing:
            typer.echo(f"  {req.warm_hint}")
        raise typer.Exit(code=1)


@ids_app.command("check")
def ids_check(
    season: int = typer.Option(..., "--season"),
    top_n: int = typer.Option(
        300, "--top-n", help="Fail if any unmatched player ranks within this many by search_rank."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Report players not resolved to a real cross-source id (SPEC.md §7).

    The blocking gate is scoped to the primary league's own roster positions and
    active players only: Sleeper's search_rank spans every player it tracks,
    including retirees and IDP positions this league may not start.
    """
    settings = load_settings()
    league = load_primary_league()
    eligible_positions = mapping.league_relevant_positions(league)
    unmatched = mapping.unmatched_report(season, settings=settings, offline=offline)

    if unmatched.is_empty():
        typer.echo("ffapp ids check: 0 unmatched players.")
        return

    for row in unmatched.iter_rows(named=True):
        name = row["full_name"] or row["sleeper_id"]
        typer.echo(f"  {name:30s} sleeper_id={row['sleeper_id']} search_rank={row['search_rank']}")

    blocking = mapping.within_top_n(mapping.league_relevant(unmatched, eligible_positions), top_n)
    if not blocking.is_empty():
        typer.echo(
            f"{blocking.height} unmatched player(s) within top {top_n} by search_rank "
            "-- build failure.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"{unmatched.height} unmatched player(s), none within top {top_n}.")


def _validate_one(slug: str, *, offline: bool | None) -> bool:
    try:
        result = golden.run_golden_test(slug, offline=offline)
    except golden.NoPlayedSeasonError as exc:
        typer.echo(f"[{slug}] {exc}", err=True)
        return False

    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(
        f"[{slug}] {verdict}: {result.agreement_rate:.2%} agreement "
        f"({len(result.disagreements)} disagreement(s) / {result.total_player_weeks} player-weeks)"
    )
    for d in result.disagreements:
        note = " (no computed row)" if d.missing_computed_row else ""
        typer.echo(
            f"    week {d.week} {d.player_id}: sleeper={d.sleeper_points:.2f} "
            f"computed={d.computed_points:.2f}{note}"
        )
    return result.passed


@scoring_app.command("validate")
def scoring_validate(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    all_leagues: bool = typer.Option(False, "--all-leagues", help="Validate every league."),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Validate score_stat_line against Sleeper's own players_points (SPEC §8.4).

    Runs against each league's most recently PLAYED season (its `previous_league_id`
    at time of writing, since current-season config is still pre-draft), not the
    league's current-season config -- scoring can change year to year.
    """
    if all_leagues:
        slugs = [lg.slug for lg in load_all_leagues()]
    elif league is not None:
        slugs = [league]
    else:
        primary = load_primary_league()
        typer.echo(f"No --league given; defaulting to primary league '{primary.slug}'.")
        slugs = [primary.slug]

    results = [_validate_one(slug, offline=offline) for slug in slugs]
    if not all(results):
        raise typer.Exit(code=1)


@draft_app.command("board")
def draft_board_command(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to the league's own configured season."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Assemble the draft board CSV (SPEC.md §9.7): every projected player,
    ranked by VOR, with tiers, ADP, survival probability, and opportunity
    cost, written to data/outputs/draft_board_<season>.csv. Also writes the
    "no model" source-rankings CSV alongside it (each source's own
    positional rank, no VOR) to data/outputs/source_rankings_<season>.csv.
    """
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else league_config.season

    try:
        result = draft_board.build_draft_board(
            league_config, settings, season=resolved_season, offline=offline
        )
        source_ranks = draft_board.build_source_rankings(
            league_config, settings, season=resolved_season, offline=offline
        )
    except (
        draft_board.NoRankingsSourcesAvailableError,
        draft_board.NotEnoughPicksError,
        draft_board.DuplicatePlayerRowsError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    output_path = draft_board.draft_board_csv_path(settings, season=resolved_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    typer.echo(f"Wrote {result.height} players to {output_path}")

    source_rankings_path = draft_board.source_rankings_csv_path(settings, season=resolved_season)
    source_ranks.write_csv(source_rankings_path)
    typer.echo(f"Wrote {source_ranks.height} players to {source_rankings_path}")


@draft_app.command("export")
def draft_export_command(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to the league's own configured season."
    ),
    out: Path | None = typer.Option(  # noqa: B008 -- same typer.Option default pattern as above
        None,
        "--out",
        help="Output HTML path. Defaults to data/outputs/draft_board_<season>_export.html; "
        "a CSV fallback is written alongside it with a .csv extension.",
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Self-contained static HTML export of the draft board (SPEC-ADDENDUM-03.md §D):
    board data and every script inline, no CDN, no external fonts, no network calls --
    opens on a phone with WiFi and cellular both off. Also writes a CSV fallback.
    """
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else league_config.season

    try:
        bundle = draft_export.build_export_bundle(
            league_config, settings, season=resolved_season, offline=offline
        )
    except (
        draft_board.NoRankingsSourcesAvailableError,
        draft_board.NotEnoughPicksError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    html_path = (
        out if out is not None else draft_export.export_html_path(settings, season=resolved_season)
    )
    csv_path = html_path.with_suffix(".csv")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(draft_export.render_html(bundle, league=league_config), encoding="utf-8")
    bundle.board.write_csv(csv_path)
    typer.echo(f"Wrote {bundle.board.height} players to {html_path} and {csv_path}")


@draft_app.command("live")
def draft_live_command(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Start a replay session against the league's most recent completed real draft, "
        "for local UI verification without a real draft in progress (SPEC-ADDENDUM-03.md §E).",
    ),
    pace_seconds: float = typer.Option(
        8.0, "--pace-seconds", help="Seconds between each replayed pick becoming visible."
    ),
    stop: bool = typer.Option(False, "--stop", help="Clear an active replay session."),
    offline: bool | None = typer.Option(  # noqa: B008 -- same typer.Option default pattern as above
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Live draft assistant support (SPEC §9.8; task 0.14). Real-time polling
    itself lives in the Streamlit app's own Live Draft tab and Draft Mobile
    page (a refresh button / auto-refresh, both hitting Sleeper directly);
    this command only manages recorded-draft replay sessions those pages
    read instead when one is active, for verification before a real draft.
    """
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()

    if stop:
        draft_replay.stop_replay(settings, league_slug=league_config.slug)
        typer.echo(f"Stopped replay for {league_config.slug}.")
        return

    if not replay:
        typer.echo(
            "ffapp draft live currently only supports --replay (start a recorded-draft replay "
            "session) or --stop. Real live polling happens in the Streamlit app itself.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        session = draft_replay.start_replay(
            league_config, settings, pace_seconds=pace_seconds, offline=offline
        )
    except draft_replay.NoCompletedDraftError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Replaying draft {session.draft_id} ({len(session.picks)} real picks) for "
        f"{league_config.slug} -- one more pick every {pace_seconds:.0f}s. Open the Draft "
        "Mobile page in Streamlit to watch it (`ffapp draft live --stop` to clear)."
    )


def _flex_eligible_positions(fmt: LeagueFormat) -> set[str]:
    """Positions eligible for any *active* flex slot (count > 0) -- the
    real roster shape `evaluation.metrics.start_sit_accuracy` needs to
    scope its pairwise flex choices to, not every position the league
    schema could theoretically flex (CLAUDE.md rule 5: driven by
    `LeagueFormat`, never hardcoded)."""
    positions: set[str] = set()
    for slot_name, count in fmt.flex_slots.items():
        if count > 0:
            positions.update(fmt.flex_eligible.get(slot_name, []))
    return positions


def _final_training_window(
    features: pl.DataFrame, settings: Settings, seasons: list[int]
) -> pl.DataFrame:
    """Every row strictly before the earliest validation season, for a
    single "current, deployable model" fit -- feature importances describe
    *this* fit, not any one walk-forward week's own fold."""
    train_cutoff = min(seasons)
    return features.filter(
        (pl.col("season") >= settings.seasons.train_start) & (pl.col("season") < train_cutoff)
    )


@app.command("evaluate")
def evaluate_command(
    seasons: list[int] = typer.Option(  # noqa: B008 -- typer's own multi-value-option pattern
        ..., "--seasons", help="Validation seasons, e.g. --seasons 2021 2022 2023 2024 2025."
    ),
) -> None:
    """Walk-forward backtest (SPEC §12.2; task 1.12): fit-and-predict week by
    week over `--seasons`, using only strictly-prior data (no random split,
    anywhere), and write raw predictions to
    data/outputs/eval/<timestamp>/predictions.parquet (points-style
    predictors) and `availability_predictions.parquet` (task 1.14's own
    binary target, a separate walk-forward run since it predicts a
    different `target_column`), plus a markdown evaluation report
    (SPEC §12.6; task 1.17) alongside them in the same directory.

    Exercises the harness with baselines B0-B2 (SPEC §12.3) plus the real
    fitted `PointsPredictor`/`AvailabilityPredictor` (tasks 1.15/1.14).
    B3 needs a separately materialised multi-season consensus-projections
    table (`ingest.rankings`' FantasyPros weekly-archive git-history
    mining) and stays out of scope for this command, a deliberate
    boundary from task 1.12's own original design, not revisited here.
    Quantile-model distribution metrics (task 1.16, SPEC §11.5) also stay
    out -- `predict_quantiles` returns a full per-row quantile grid, not
    the single-`Series`-per-row shape `evaluation.backtest.Predictor`
    expects, so it doesn't fit this harness without a real redesign
    outside this task's own scope.
    """
    settings = load_settings()

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    schedule_path = settings.data_root / "interim" / "schedule.parquet"
    for path in (features_path, schedule_path):
        if not path.exists():
            typer.echo(
                f"Missing {path}. Materialise the interim/feature tables first "
                "(see HANDOFF.md for the real end-to-end build steps).",
                err=True,
            )
            raise typer.Exit(code=1)

    features = pl.read_parquet(features_path)
    schedule = pl.read_parquet(schedule_path)

    features = baselines.add_b0_positional_mean(features)
    features = baselines.add_b1_season_to_date_mean(features)
    features = baselines.add_b2_ewm_4(features)
    features = baselines.add_availability_base_rate(features)

    points_predictors: list[backtest.Predictor] = [
        backtest.BaselinePredictor("b0_positional_mean", "b0_positional_mean"),
        backtest.BaselinePredictor("b1_season_to_date_mean", "b1_season_to_date_mean"),
        backtest.BaselinePredictor("b2_ewm_4", "b2_ewm_4"),
        points.PointsPredictor(settings.model.lightgbm),
    ]
    predictions = backtest.run_walk_forward_backtest(
        features,
        schedule,
        points_predictors,
        validation_seasons=seasons,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
    )

    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = settings.data_root / "outputs" / "eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"
    predictions.write_parquet(output_path)
    typer.echo(f"Wrote {predictions.height} predictions to {output_path}")

    availability_predictors: list[backtest.Predictor] = [
        backtest.BaselinePredictor("availability_base_rate", "availability_base_rate"),
        availability.AvailabilityPredictor(settings.model.lightgbm),
    ]
    availability_predictions = backtest.run_walk_forward_backtest(
        features,
        schedule,
        availability_predictors,
        validation_seasons=seasons,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        target_column="availability_flag",
    )
    availability_output_path = output_dir / "availability_predictions.parquet"
    availability_predictions.write_parquet(availability_output_path)
    typer.echo(
        f"Wrote {availability_predictions.height} availability predictions "
        f"to {availability_output_path}"
    )

    computed_metrics: list[evaluation_metrics.MetricResult] = []
    feature_importances: dict[str, list[tuple[str, float]]] = {}
    calibration_curves: dict[str, list[tuple[float, float, int]]] = {}

    if not predictions.is_empty():
        league = load_primary_league()
        fmt = parse_league_format(league)
        startable_counts = evaluation_metrics.startable_counts_from_predictions(predictions, fmt)
        computed_metrics.extend(
            evaluation_metrics.accuracy_metrics(predictions, startable_counts=startable_counts)
        )
        computed_metrics.extend(
            evaluation_metrics.ranking_metrics(predictions, startable_counts=startable_counts)
        )
        computed_metrics.extend(
            evaluation_metrics.start_sit_accuracy(predictions, _flex_eligible_positions(fmt))
        )
        computed_metrics.extend(evaluation_metrics.lineup_regret(predictions, fmt))

        final_train = _final_training_window(features, settings, seasons)
        if not final_train.is_empty():
            fitted_points = points.fit_points_model(
                final_train, lightgbm_params=settings.model.lightgbm
            )
            for position, booster in fitted_points.boosters.items():
                feature_importances[f"points_{position}"] = (
                    evaluation_report.extract_feature_importance(booster)
                )

    if not availability_predictions.is_empty():
        for predictor in ("availability_lightgbm", "availability_base_rate"):
            pred_df = availability_predictions.filter(
                (pl.col("predictor") == predictor) & pl.col("prediction").is_not_null()
            )
            if pred_df.is_empty():
                continue
            brier = evaluation_metrics.brier_score(
                pred_df["prediction"].to_numpy(), pred_df["target"].to_numpy()
            )
            ci_low, ci_high = evaluation_metrics.bootstrap_ci_over_rows(
                pred_df,
                lambda df: evaluation_metrics.brier_score(
                    df["prediction"].to_numpy(), df["target"].to_numpy()
                ),
            )
            computed_metrics.append(
                evaluation_metrics.MetricResult(
                    metric="brier_score",
                    predictor=predictor,
                    position=None,
                    scope="all",
                    value=brier,
                    n_obs=pred_df.height,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )
            calibration_curves[predictor] = evaluation_metrics.calibration_curve(
                pred_df["prediction"].to_numpy(), pred_df["target"].to_numpy()
            )

        final_train = _final_training_window(features, settings, seasons)
        if not final_train.is_empty():
            fitted_availability = availability.fit_availability_model(
                final_train, lightgbm_params=settings.model.lightgbm
            )
            feature_importances["availability"] = evaluation_report.extract_feature_importance(
                fitted_availability.booster
            )

    report_markdown = evaluation_report.render_report_markdown(
        seasons=seasons,
        generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=evaluation_report.current_git_commit(),
        metrics=computed_metrics,
        feature_importances=feature_importances or None,
        calibration_curves=calibration_curves or None,
    )
    report_path = evaluation_report.write_report(output_dir, report_markdown)
    typer.echo(f"Wrote evaluation report to {report_path}")


@app.command("project")
def project_command(
    week: int = typer.Option(..., "--week", help="Target week to generate projections for."),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to settings.seasons.current."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
    from_week: int | None = typer.Option(
        None,
        "--from-week",
        help=(
            "Start of the ROS horizon. Must be given together with "
            "--through-week (range mode) -- it does NOT default to --week; "
            "--week is still required by this command but is ignored in "
            "range mode."
        ),
    ),
    through_week: int | None = typer.Option(
        None, "--through-week", help="End of the ROS horizon (inclusive)."
    ),
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
) -> None:
    """Real weekly projections (SPEC §6.2, §11.8; task 1.18): fit
    availability + quantiles, plus whichever conditional-mean source
    `settings.model.projection_source` names (`SPEC-ADDENDUM-04.md` §C
    -- real default `consensus_b3`, see `config.ModelSettings`'s own
    docstring and `docs/JOURNAL.md`'s 2026-08-16 closing entry), on
    every row strictly before `(season, week)`, predict onto that week's
    own real row universe, and upsert into
    `data/outputs/projections.parquet` -- every row carrying
    `model_version`, `projection_source`, `as_of_utc`, `feature_hash`,
    and `git_commit`.

    Requires that week's own row universe to already exist in
    `features/player_week_features.parquet` -- for the *current*, not-yet-
    played 2026 season this means whatever nflverse has published so far
    (see HANDOFF.md; 2026 has no release at all as of this session).

    `consensus_b3` needs a real, live-fetched crosswalk plus a real
    FantasyPros weekly-archive commit/snapshot -- pass `--no-offline` the
    first time for a given week (subsequent calls reuse the same cached
    commit/snapshot, `baselines.fetch_b3_for_week`'s own idempotent
    fetch). It also needs `data/interim/b3_predictions.parquet` -- this
    project's own real historical B3 archive, used to build the real
    empirical quantile spread around this week's own B3 mean (SPEC-
    ADDENDUM-04.md §D, `docs/JOURNAL.md`'s 2026-08-16 entry) -- exits
    with a clear error naming the missing file if it isn't built yet.
    """
    settings = load_settings()
    resolved_season = season if season is not None else settings.seasons.current

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing {features_path}. Materialise the feature table first "
            "(see HANDOFF.md for the real end-to-end build steps).",
            err=True,
        )
        raise typer.Exit(code=1)
    features = pl.read_parquet(features_path)

    if from_week is not None or through_week is not None:
        if from_week is None or through_week is None:
            typer.echo("--from-week and --through-week must be given together.", err=True)
            raise typer.Exit(code=1)
        league_config = load_league(league) if league is not None else load_primary_league()
        scoring_settings = league_config.league_cache["scoring_settings"]
        schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
        dpa = pl.read_parquet(settings.data_root / "interim" / "defense_position_allowed.parquet")

        crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
        sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
        players_dim_ros = mapping.build_players_dim(
            crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
        )
        # No join_key pre-processing needed here -- project_week_range (Task 8)
        # already calls mapping.dedupe_to_one_row_per_name_position internally
        # on whatever players_dim it receives, and fetch_season_consensus's own
        # fetchers (via prediction_log._resolve_to_player_id) do the same. The
        # raw build_players_dim output is exactly what every downstream
        # consumer expects -- confirmed by reading both call chains directly.

        b3_historical_path_ros = settings.data_root / "interim" / "b3_predictions.parquet"
        if not b3_historical_path_ros.exists():
            typer.echo(f"Missing {b3_historical_path_ros}. See HANDOFF.md.", err=True)
            raise typer.Exit(code=1)
        b3_historical_ros = pl.read_parquet(b3_historical_path_ros)

        now_ros = datetime.now(UTC)
        season_points = ros_consensus.fetch_season_consensus(
            resolved_season,
            scoring_settings,
            players_dim_ros,
            offline=offline,
            settings=settings,
            now=now_ros,
        )
        log_dir = settings.data_root / "outputs" / league_config.slug / "prediction_log"
        trend_by_source: dict[str, str] = {}
        fetches_path = log_dir / "source_fetches.parquet"
        if fetches_path.exists():
            # Fix 5/M1 (final review fix wave): `check_sources` unconditionally
            # rewrites the real, git-tracked `config/source_refresh_status.yml`
            # as a side effect -- unwanted here, since this is a read-mostly
            # ranking command, not the dedicated `ffapp log check-sources`
            # command. `season_source_trend` (promoted public in Task 6
            # specifically so a caller could read trend information WITHOUT
            # that side effect) returns the exact same real per-source trend
            # rows `check_sources` itself reads internally -- reused directly.
            trend_df = prediction_log.season_source_trend(log_dir)
            trend_by_source = {
                row["source"]: row["trend"]
                for row in trend_df.iter_rows(named=True)
                if row["trend"] is not None
            }
        actuals_to_date = (
            features.filter((pl.col("season") == resolved_season) & (pl.col("week") < from_week))
            .group_by("player_id")
            .agg(pl.col("target").sum().alias("actual_points_to_date"))
        )

        result = predict_ros.project_week_range(
            features,
            schedule,
            dpa,
            resolved_season,
            from_week,
            through_week,
            league_config.slug,
            scoring_settings,
            players_dim_ros,
            b3_historical_ros,
            actuals_to_date,
            season_points,
            trend_by_source,
            settings.model.quantiles,
            now_ros,
            train_start=settings.seasons.train_start,
            min_train_rows=settings.model.min_train_rows,
            lightgbm_params=settings.model.lightgbm,
            code_version=evaluation_report.current_git_commit(),
            offline=offline,
            settings=settings,
        )
        if result.is_empty():
            typer.echo("No ROS projections generated -- see HANDOFF.md.", err=True)
            raise typer.Exit(code=1)
        output_path_ros = (
            settings.data_root / "outputs" / league_config.slug / "projections_ros.parquet"
        )
        combined_ros = predict.write_projections(result, output_path_ros)
        typer.echo(
            f"Wrote {result.height} ROS projections to {output_path_ros} "
            f"({combined_ros.height} total rows)."
        )
        return

    players_dim = None
    b3_historical = None
    if settings.model.projection_source == "consensus_b3":
        crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
        sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
        players_dim = mapping.build_players_dim(
            crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
        )
        b3_historical_path = settings.data_root / "interim" / "b3_predictions.parquet"
        if not b3_historical_path.exists():
            typer.echo(
                f"Missing {b3_historical_path}. consensus_b3's quantile spread needs this "
                "project's real historical B3 archive -- see HANDOFF.md for how to build it.",
                err=True,
            )
            raise typer.Exit(code=1)
        b3_historical = pl.read_parquet(b3_historical_path)

    now = datetime.now(UTC)
    result = predict.project_week(
        features,
        resolved_season,
        week,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        lightgbm_params=settings.model.lightgbm,
        quantile_alphas=settings.model.quantiles,
        code_version=evaluation_report.current_git_commit(),
        now=now,
        projection_source=settings.model.projection_source,
        players_dim=players_dim,
        b3_historical=b3_historical,
        offline=offline,
        settings=settings,
    )
    if result.is_empty():
        typer.echo(
            f"No projections generated for season {resolved_season} week {week} -- either not "
            "enough training data, or that week's row universe doesn't exist yet in "
            "player_week_features.parquet. See HANDOFF.md.",
            err=True,
        )
        raise typer.Exit(code=1)

    output_path = settings.data_root / "outputs" / "projections.parquet"
    combined = predict.write_projections(result, output_path)
    typer.echo(
        f"Wrote {result.height} projections for season {resolved_season} week {week} "
        f"to {output_path} ({combined.height} total rows)."
    )


rankings_app = typer.Typer(name="rankings", help="Rest-of-season and other rankings views.")
app.add_typer(rankings_app, name="rankings")


@rankings_app.command("ros")
def rankings_ros_command(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to settings.seasons.current."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Rest-of-season VOR board (`SPEC-ADDENDUM-04.md` §D.3-§D.5; task
    1.21) over the CURRENT free-agent pool, with rank-change since the
    prior real run. Requires `projections_ros.parquet` to already exist
    for this league (`ffapp project --from-week --through-week --league`).

    Fix 4 (final review fix wave): unlike `project_command`'s own real
    `--offline/--no-offline` flag, this command used to hardcode
    `offline=True` for every real fetch below, silently weakening
    §D.3's "replacement level over the CURRENT free-agent pool"
    requirement to "whatever roster snapshot happens to be cached," with
    no way to force a fresh fetch from the CLI. Threaded through every
    real fetch call this command makes."""
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else settings.seasons.current
    league_format = parse_league_format(league_config)

    ros_path = settings.data_root / "outputs" / league_config.slug / "projections_ros.parquet"
    if not ros_path.exists():
        typer.echo(
            f"Missing {ros_path}. Run `ffapp project --from-week --through-week "
            f"--league {league_config.slug}` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    projections_ros = pl.read_parquet(ros_path).filter(pl.col("season") == resolved_season)

    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    players_dim = mapping.build_players_dim(
        crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
    )

    if league_config.league_id is None:
        typer.echo(
            f"League {league_config.slug} has no real sleeper.league_id configured.", err=True
        )
        raise typer.Exit(code=1)
    rosters = json.loads(
        sleeper.fetch_rosters(
            league_config.league_id, offline=offline, settings=settings
        ).read_text()
    )
    rostered_ids = waivers.rostered_sleeper_ids(rosters)
    # league_relevant_positions takes the real LeagueConfig (needs .league_cache/
    # .overrides), not LeagueFormat -- confirmed against its real signature. The
    # existing `mapping` import (ffapp.ids.mapping) is this file's own established
    # alias for that module (see ids_check's own use of
    # mapping.league_relevant_positions) -- no separate "ids_mapping" alias exists
    # here, so this reuses the one already in scope rather than adding a duplicate.
    eligible_positions = mapping.league_relevant_positions(league_config)

    if projections_ros.filter(pl.col("is_current_week")).is_empty():
        typer.echo(
            f"{ros_path} has no real current-week row -- was it built for this season?", err=True
        )
        raise typer.Exit(code=1)
    anchor_week = int(projections_ros.filter(pl.col("is_current_week"))["week"].min())  # type: ignore[arg-type]

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    features = pl.read_parquet(features_path)
    before_anchor = (pl.col("season") < resolved_season) | (
        (pl.col("season") == resolved_season) & (pl.col("week") < anchor_week)
    )
    anchor_row = (pl.col("season") == resolved_season) & (pl.col("week") == anchor_week)
    train_rows = features.filter(before_anchor)
    target_rows = features.filter(anchor_row)

    availability_model = availability.fit_availability_model(
        train_rows, lightgbm_params=settings.model.lightgbm
    )
    p_active_series = availability.predict_p_active(availability_model, target_rows)
    p_active_now = dict(
        zip(target_rows["player_id"].to_list(), p_active_series.to_list(), strict=True)
    )

    # rosters/snap_counts/injuries have no usable data/interim/ counterpart here --
    # sim.injury's own build_hazard_features reads the RAW nflverse tables directly
    # (confirmed against sim/injury.py's own module docstring and docs/JOURNAL.md's
    # task 2.3 entry). schedule genuinely does have a real interim table this needs.
    # Real bug found live during task 13's own e2e verification, not by any unit
    # test (the CLI test mocks injury.build_hazard_features out entirely): this used
    # to read interim/injuries.parquet, but ingest.nflverse.normalize_injuries
    # (task 1.4) already renames that table's own gsis_id -> player_id, while
    # sim.injury.add_injury_report expects the RAW nflverse schema, still named
    # gsis_id -- exactly like rosters_table/snap_counts two lines below, which
    # already correctly read the raw table via nflverse.fetch_*, not interim/.
    # Reading interim/injuries.parquet raised a real ColumnNotFoundError on
    # "gsis_id" the first time this command was run against real 2025 data.
    # Real bug found live running this command for the first time against the
    # genuinely current season (every prior real run used a past season as a
    # backtest stand-in while settings.seasons.current already pointed past
    # it, e.g. resolved_season=2025 with current=2026, which coincidentally
    # kept the exclusive range below wide enough): an exclusive
    # `range(train_start, current)` always excludes `current` itself, so
    # `hazard_grid` (built from `rosters_table` below) has zero rows for any
    # week of the real live season -- `hazard_target = hazard_grid.filter(
    # anchor_row)` then has 0 rows and `predict_p_miss` crashes inside
    # sklearn's `SimpleImputer` ("Found array with 0 sample(s)"). Range must
    # include `current` so the anchor week's own season has real rows to
    # filter down to.
    train_season_range = list(range(settings.seasons.train_start, settings.seasons.current + 1))
    rosters_table = pl.read_parquet(
        nflverse.fetch_rosters(train_season_range, offline=offline, settings=settings)
    )
    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    # injuries/snap_counts, unlike rosters, genuinely have no real data for the
    # live current season yet (confirmed live 2026-09-05: PFR/the official
    # injury report only start publishing once real games are played, days
    # into the season at the earliest) -- requesting `current` here would
    # raise or return nothing useful either way. build_hazard_features joins
    # both onto `rosters_table`'s own grid (the base table) and is already
    # null-safe for a row with no covariate match, so scoping this fetch to
    # real history only costs nothing and avoids that failure mode entirely.
    covariate_season_range = list(range(settings.seasons.train_start, settings.seasons.current))
    injuries = pl.read_parquet(
        nflverse.fetch_injuries(covariate_season_range, offline=offline, settings=settings)
    )
    snap_counts = pl.read_parquet(
        nflverse.fetch_snap_counts(covariate_season_range, offline=offline, settings=settings)
    )
    # `fetch_player_ids` returns a Path to the raw CSV (`data/raw/nflverse/player_ids.csv`),
    # not parquet -- `mapping.load_crosswalk_base` is the same already-tested loader
    # `mapping.build_players_dim` itself calls internally, reused here rather than
    # re-parsing the CSV a second way.
    crosswalk = mapping.load_crosswalk_base(crosswalk_path)
    hazard_grid = injury.build_hazard_features(
        rosters_table, schedule, injuries, snap_counts, crosswalk
    )
    hazard_train = hazard_grid.filter(before_anchor)
    hazard_target = hazard_grid.filter(anchor_row)
    hazard_model = injury.fit_hazard_model(hazard_train)
    p_miss_series = injury.predict_p_miss(hazard_model, hazard_target)
    p_miss_now = dict(
        zip(hazard_target["player_id"].to_list(), p_miss_series.to_list(), strict=True)
    )

    # Fix 1 (final review fix wave, the most severe finding): real positional
    # fallback rates for any real ranked player with no anchor-week
    # `p_active_now`/`p_miss_now` entry at all -- `aggregate_ros` no longer
    # silently defaults such a player to "definitely healthy" (see that
    # function's own docstring/comments). Computed from the exact same
    # `train_rows`/`hazard_train` frames already built above for the model
    # fits, not a separate query.
    default_p_active_by_position = baselines.positional_availability_base_rate(train_rows)
    default_p_miss_by_position = injury.positional_base_rate(hazard_train)

    board_player_ids = set(projections_ros["player_id"].unique().to_list())
    n_missing_availability = sum(
        1 for pid in board_player_ids if pid not in p_active_now or pid not in p_miss_now
    )
    typer.echo(
        f"{n_missing_availability} of {len(board_player_ids)} ranked players had no "
        "anchor-week availability data; using positional base rates."
    )

    position_by_player = dict(
        zip(
            projections_ros["player_id"].to_list(),
            projections_ros["position"].to_list(),
            strict=False,
        )
    )

    calibration = load_ros_calibration()
    playoff_week_list = sos.playoff_weeks(
        schedule, season=resolved_season, playoff_week_start=league_format.playoff_week_start
    )
    aggregated = ros_aggregate.aggregate_ros(
        projections_ros,
        p_active_now,
        p_miss_now,
        position_by_player,
        calibration,
        playoff_weeks=playoff_week_list,
        ros_sims=settings.ros.ros_sims,
        default_recovery_prob=settings.ros.default_recovery_prob,
        correlation=settings.simulation.correlation,
        rng=np.random.default_rng(),
        default_p_active_by_position=default_p_active_by_position,
        default_p_miss_by_position=default_p_miss_by_position,
    )
    board = ros_rankings.build_ros_board(
        aggregated, players_dim, rostered_ids, eligible_positions, league_format
    )

    out_dir = settings.data_root / "outputs" / league_config.slug / "rankings_ros"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.parquet"
    previous_board = pl.read_parquet(latest_path) if latest_path.exists() else None
    with_rank_change = board.join(
        ros_rankings.rank_change(board, previous_board), on="player_id", how="left"
    )

    board_path = run_dir / "board.parquet"
    with_rank_change.write_parquet(board_path)
    with_rank_change.write_parquet(latest_path)
    typer.echo(
        f"Wrote {with_rank_change.height} ROS ranked players to {board_path} (and latest.parquet)."
    )
    typer.echo(with_rank_change.head(20).to_pandas().to_string(index=False))


def _resolve_league_slug(league: str | None) -> tuple[str, dict[str, float]]:
    league_config = load_league(league) if league is not None else load_primary_league()
    return league_config.slug, league_config.league_cache["scoring_settings"]


@log_app.command("week")
def log_week_command(
    week: int = typer.Option(..., "--week", help="Target week to log."),
    run_label: str = typer.Option(..., "--run-label", help="tuesday | thursday | sunday."),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to settings.seasons.current."
    ),
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
    offline: bool | None = typer.Option(
        None, "--offline/--no-offline", help="Override FFAPP_OFFLINE for this run."
    ),
) -> None:
    """Real in-season prediction logging (`SPEC-ADDENDUM-05.md` §B; task
    3.8) -- one row per real player, capturing `b3_mean`/`model_mean`/
    `b2_mean`/`p_active` plus every real per-source point value
    (`espn`/`cbs`/`fantasysharks`/`fftoday`/`footballguys`/`draftsharks`/
    `fantasypros`), for this exact moment -- weekly consensus projections
    are not re-fetchable, so this cannot be re-run for a past week and
    get the same real numbers back. Committed to git (§B.3): run
    `git add data/outputs/<league_slug>/prediction_log/` after this and
    commit, same non-reproducibility reasoning as the rankings gitignore
    exception.
    """
    if run_label not in prediction_log.RUN_LABELS:
        typer.echo(f"run_label={run_label!r} is not one of {prediction_log.RUN_LABELS}.", err=True)
        raise typer.Exit(code=1)

    settings = load_settings()
    resolved_season = season if season is not None else settings.seasons.current
    league_slug, scoring_settings = _resolve_league_slug(league)

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    schedule_path = settings.data_root / "interim" / "schedule.parquet"
    b3_historical_path = settings.data_root / "interim" / "b3_predictions.parquet"
    for path in (features_path, schedule_path, b3_historical_path):
        if not path.exists():
            typer.echo(
                f"Missing {path}. Materialise the interim/feature tables and the real B3 "
                "archive first (see HANDOFF.md).",
                err=True,
            )
            raise typer.Exit(code=1)
    features = pl.read_parquet(features_path)
    schedule = pl.read_parquet(schedule_path)
    b3_historical = pl.read_parquet(b3_historical_path)

    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    players_dim = mapping.build_players_dim(
        crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
    )

    now = datetime.now(UTC)
    rows, fetch_rows = prediction_log.build_prediction_log(
        features,
        schedule,
        resolved_season,
        week,
        run_label,
        league_slug=league_slug,
        scoring_settings=scoring_settings,
        players_dim=players_dim,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        lightgbm_params=settings.model.lightgbm,
        quantile_alphas=settings.model.quantiles,
        b3_historical=b3_historical,
        code_version=evaluation_report.current_git_commit(),
        now=now,
        offline=offline,
        settings=settings,
    )
    if rows.is_empty():
        typer.echo(
            f"No prediction log rows generated for season {resolved_season} week {week} -- "
            "either not enough training data, or that week's row universe doesn't exist yet.",
            err=True,
        )
        raise typer.Exit(code=1)

    path = prediction_log.write_prediction_log(
        rows,
        fetch_rows,
        resolved_season,
        week,
        run_label,
        league_slug=league_slug,
        settings=settings,
    )
    n_errors = fetch_rows.filter(pl.col("fetch_error").is_not_null()).height
    typer.echo(
        f"Wrote {rows.height} prediction log rows for {league_slug} season {resolved_season} "
        f"week {week} run={run_label} to {path} ({n_errors} of {fetch_rows.height} sources "
        "failed this run -- see fetch_error). Remember: `git add` and commit this file (§B.3)."
    )


@log_app.command("backfill")
def log_backfill_command(
    week: int = typer.Option(..., "--week", help="Week to backfill actual_points for."),
    season: int | None = typer.Option(
        None, "--season", help="Defaults to settings.seasons.current."
    ),
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
) -> None:
    """Fills `actual_points` for every real row already logged for
    `(season, week)` (`SPEC-ADDENDUM-05.md` §B.4). Run in the Tuesday job
    for the prior week, once games are complete. Raises a clear, named
    error (not a silent no-op) if that week was never logged.
    """
    settings = load_settings()
    resolved_season = season if season is not None else settings.seasons.current
    league_slug, _ = _resolve_league_slug(league)

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    if not features_path.exists():
        typer.echo(f"Missing {features_path}.", err=True)
        raise typer.Exit(code=1)
    features = pl.read_parquet(features_path)

    try:
        filled = prediction_log.backfill_actual_points(
            features, resolved_season, week, league_slug=league_slug, settings=settings
        )
    except prediction_log.MissingBackfillError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    n_filled = filled.filter(pl.col("actual_points").is_not_null()).height
    typer.echo(
        f"Backfilled actual_points for {n_filled} of {filled.height} logged rows, "
        f"season {resolved_season} week {week}. Remember: `git add` and commit (§B.3)."
    )


@log_app.command("check-sources")
def log_check_sources_command(
    league: str | None = typer.Option(
        None, "--league", help="League slug. Defaults to the primary league."
    ),
) -> None:
    """`SPEC-ADDENDUM-05.md` §B.2's own real resolution mechanism, in two
    parts. Weekly sources: counts distinct `payload_sha256` values per
    source across every real logged week, promotes a source with exactly
    one distinct hash across at least 3 real logged weeks to `frozen`,
    confirms one whose hash changed as `weekly_confirmed`, and writes the
    result to `config/source_refresh_status.yml`. Season sources
    (`espn`/`cbs`/`fantasysharks`/`fftoday`/`footballguys`/`draftsharks`
    -- real season-long totals, not weekly numbers, confirmed live
    2026-08-16): reports whether each one's own real mean value declines
    week over week (a genuine rest-of-season signal) or stays flat (a
    static full-season snapshot). Run after Week 3-4.
    """
    settings = load_settings()
    league_slug, _ = _resolve_league_slug(league)

    summary = prediction_log.check_sources(league_slug=league_slug, settings=settings)
    if summary.is_empty():
        typer.echo("No source_fetches.parquet logged yet -- run `ffapp log week` first.", err=True)
        raise typer.Exit(code=1)

    for row in summary.sort("source").iter_rows(named=True):
        line = (
            f"{row['source']}: {row['n_distinct_hashes']} distinct hash(es) across "
            f"{row['n_weeks_logged']} real logged week(s) -> {row['status']}"
        )
        if row["season_trend"] is not None:
            line += f"  |  season_trend={row['season_trend']} (n_weeks={row['season_n_weeks']})"
        typer.echo(line)
    typer.echo("Wrote config/source_refresh_status.yml.")
