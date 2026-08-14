from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from ffapp import __version__
from ffapp.cache import registry as cache_registry
from ffapp.cache.offline import is_offline
from ffapp.config import Settings, load_all_leagues, load_league, load_primary_league, load_settings
from ffapp.draft import board as draft_board
from ffapp.draft import export as draft_export
from ffapp.env import load_env
from ffapp.evaluation import backtest
from ffapp.evaluation import metrics as evaluation_metrics
from ffapp.evaluation import report as evaluation_report
from ffapp.ids import mapping
from ffapp.league_format import LeagueFormat, parse_league_format
from ffapp.models import availability, baselines, points, predict
from ffapp.scoring import golden

load_env()

app = typer.Typer(name="ffapp", help="Fantasy football decision-support CLI.")
ingest_app = typer.Typer(name="ingest", help="Ingest raw data from external sources.")
cache_app = typer.Typer(name="cache", help="Manage the offline data cache (SPEC-ADDENDUM-02.md).")
ids_app = typer.Typer(name="ids", help="Cross-source player id resolution (SPEC.md §7).")
scoring_app = typer.Typer(name="scoring", help="League scoring engine (SPEC.md §8).")
draft_app = typer.Typer(name="draft", help="Draft board and draft-day support (SPEC.md §9).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(cache_app, name="cache")
app.add_typer(ids_app, name="ids")
app.add_typer(scoring_app, name="scoring")
app.add_typer(draft_app, name="draft")


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
    cost, written to data/outputs/draft_board_<season>.csv.
    """
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else league_config.season

    try:
        result = draft_board.build_draft_board(
            league_config, settings, season=resolved_season, offline=offline
        )
    except (
        draft_board.NoRankingsSourcesAvailableError,
        draft_board.NotEnoughPicksError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    output_path = draft_board.draft_board_csv_path(settings, season=resolved_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    typer.echo(f"Wrote {result.height} players to {output_path}")


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
) -> None:
    """Real weekly projections (SPEC §6.2, §11.8; task 1.18): fit
    availability + points + quantiles on every row strictly before
    `(season, week)`, predict onto that week's own real row universe, and
    upsert into `data/outputs/projections.parquet` -- every row carrying
    `model_version`, `as_of_utc`, `feature_hash`, and `git_commit`.

    Requires that week's own row universe to already exist in
    `features/player_week_features.parquet` -- for the *current*, not-yet-
    played 2026 season this means whatever nflverse has published so far
    (see HANDOFF.md; 2026 has no release at all as of this session).
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
