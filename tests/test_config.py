from pathlib import Path

import pytest

from ffapp.config import (
    MultiplePrimaryLeaguesError,
    NoPrimaryLeagueError,
    load_all_leagues,
    load_league,
    load_primary_league,
    load_settings,
    write_league_stub,
)

FIXTURES = Path(__file__).parent / "fixtures" / "config"


def test_load_settings_resolves_data_root_relative_to_root_not_settings_file(
    tmp_path: Path,
) -> None:
    settings = load_settings(FIXTURES / "settings.yml", root=tmp_path)

    assert settings.data_root == (tmp_path / "data").resolve()


def test_load_settings_defaults_root_to_the_repo_root() -> None:
    from ffapp.config import REPO_ROOT

    settings = load_settings(FIXTURES / "settings.yml")

    assert settings.data_root == (REPO_ROOT / "data").resolve()


def test_load_settings_reads_sleeper_username() -> None:
    settings = load_settings(FIXTURES / "settings.yml")

    assert settings.sleeper_username == "fixture_user"


def test_load_settings_reads_cache_staleness_hours() -> None:
    settings = load_settings(FIXTURES / "settings.yml")

    assert settings.cache.staleness_hours["sleeper_rosters"] == 24
    assert settings.cache.offline_default is True


def test_load_league_reads_slug_and_scoring_settings() -> None:
    league = load_league("main-ppr", leagues_dir=FIXTURES / "leagues")

    assert league.slug == "main-ppr"
    assert league.is_primary is True
    assert league.league_cache["scoring_settings"]["rec"] == 1


def test_load_all_leagues_returns_every_league_file() -> None:
    leagues = load_all_leagues(leagues_dir=FIXTURES / "leagues")

    assert {league.slug for league in leagues} == {"main-ppr", "league-b"}


def test_load_primary_league_returns_the_one_marked_primary() -> None:
    primary = load_primary_league(leagues_dir=FIXTURES / "leagues")

    assert primary.slug == "main-ppr"


def test_load_primary_league_raises_when_none_marked_primary(tmp_path: Path) -> None:
    leagues_dir = tmp_path / "leagues"
    leagues_dir.mkdir()
    (leagues_dir / "a.yml").write_text(
        "slug: a\ndisplay_name: A\nis_primary: false\n"
        "sleeper:\n  league_id: '1'\n  season: 2026\n"
        "league_cache: {}\noverrides: {}\n"
    )

    with pytest.raises(NoPrimaryLeagueError):
        load_primary_league(leagues_dir=leagues_dir)


def test_load_primary_league_raises_when_more_than_one_marked_primary(tmp_path: Path) -> None:
    leagues_dir = tmp_path / "leagues"
    leagues_dir.mkdir()
    for slug in ("a", "b"):
        (leagues_dir / f"{slug}.yml").write_text(
            f"slug: {slug}\ndisplay_name: {slug}\nis_primary: true\n"
            "sleeper:\n  league_id: '1'\n  season: 2026\n"
            "league_cache: {}\noverrides: {}\n"
        )

    with pytest.raises(MultiplePrimaryLeaguesError):
        load_primary_league(leagues_dir=leagues_dir)


def test_write_league_stub_creates_a_new_league_file(tmp_path: Path) -> None:
    leagues_dir = tmp_path / "leagues"
    leagues_dir.mkdir()

    path = write_league_stub(
        slug="new-league",
        display_name="New League",
        league_id="333",
        season=2026,
        league_cache={"total_rosters": 10},
        leagues_dir=leagues_dir,
    )

    assert path == leagues_dir / "new-league.yml"
    league = load_league("new-league", leagues_dir=leagues_dir)
    assert league.is_primary is False
    assert league.league_cache["total_rosters"] == 10


def test_write_league_stub_preserves_hand_set_is_primary_but_refreshes_league_cache(
    tmp_path: Path,
) -> None:
    leagues_dir = tmp_path / "leagues"
    leagues_dir.mkdir()
    write_league_stub(
        slug="existing",
        display_name="Existing",
        league_id="1",
        season=2026,
        league_cache={"total_rosters": 10},
        leagues_dir=leagues_dir,
    )
    # Hand-set is_primary, as the workflow instructs.
    league_path = leagues_dir / "existing.yml"
    league_path.write_text(league_path.read_text().replace("is_primary: false", "is_primary: true"))

    write_league_stub(
        slug="existing",
        display_name="Existing",
        league_id="1",
        season=2026,
        league_cache={"total_rosters": 999},
        leagues_dir=leagues_dir,
    )

    league = load_league("existing", leagues_dir=leagues_dir)
    assert league.is_primary is True
    assert league.league_cache["total_rosters"] == 999
