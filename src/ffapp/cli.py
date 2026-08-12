import typer

from ffapp import __version__
from ffapp.cache import registry as cache_registry
from ffapp.cache.offline import is_offline
from ffapp.config import load_all_leagues, load_league, load_primary_league, load_settings
from ffapp.draft import board as draft_board
from ffapp.env import load_env
from ffapp.ids import mapping
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
