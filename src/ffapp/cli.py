import typer

from ffapp import __version__
from ffapp.cache import registry as cache_registry
from ffapp.cache.offline import is_offline
from ffapp.config import load_settings
from ffapp.env import load_env

load_env()

app = typer.Typer(name="ffapp", help="Fantasy football decision-support CLI.")
ingest_app = typer.Typer(name="ingest", help="Ingest raw data from external sources.")
cache_app = typer.Typer(name="cache", help="Manage the offline data cache (SPEC-ADDENDUM-02.md).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(cache_app, name="cache")


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
