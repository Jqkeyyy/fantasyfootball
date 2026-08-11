import polars as pl
import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.config import LeagueConfig
from ffapp.ids import mapping

runner = CliRunner()

_SCHEMA = {
    "sleeper_id": pl.Utf8,
    "full_name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "search_rank": pl.Int64,
    "active": pl.Boolean,
}

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={"roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN"]},
    overrides={"flex_eligible": ["RB", "WR", "TE"]},
)


def _df(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_SCHEMA) if rows else pl.DataFrame(schema=_SCHEMA)


@pytest.fixture(autouse=True)
def _primary_league(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)


def test_ids_check_reports_zero_unmatched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping, "unmatched_report", lambda season, **kwargs: _df([]))

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 0
    assert "0 unmatched" in result.output.lower()


def test_ids_check_passes_when_no_unmatched_player_is_within_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "8888",
                    "full_name": "Deep Bench Guy",
                    "position": "RB",
                    "team": "KC",
                    "search_rank": 9000,
                    "active": True,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 0
    assert "Deep Bench Guy" in result.output


def test_ids_check_fails_when_unmatched_player_is_within_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "9999",
                    "full_name": "Zzyzx Unmatched Guy",
                    "position": "WR",
                    "team": "KC",
                    "search_rank": 50,
                    "active": True,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 1
    assert "build failure" in result.output.lower()


def test_ids_check_respects_custom_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "8888",
                    "full_name": "Deep Bench Guy",
                    "position": "RB",
                    "team": "KC",
                    "search_rank": 150,
                    "active": True,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026", "--top-n", "100"])

    assert result.exit_code == 0


def test_ids_check_ignores_positions_this_league_does_not_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "6001",
                    "full_name": "Rookie Linebacker Guy",
                    "position": "LB",
                    "team": "KC",
                    "search_rank": 70,
                    "active": True,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 0


def test_ids_check_ignores_inactive_players(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "6002",
                    "full_name": "Retired High Rank Guy",
                    "position": "WR",
                    "team": "KC",
                    "search_rank": 60,
                    "active": False,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 0


def test_ids_check_ignores_players_with_no_current_nfl_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mapping,
        "unmatched_report",
        lambda season, **kwargs: _df(
            [
                {
                    "sleeper_id": "6003",
                    "full_name": "Unsigned Camp Cut Guy",
                    "position": "WR",
                    "team": None,
                    "search_rank": 60,
                    "active": True,
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["ids", "check", "--season", "2026"])

    assert result.exit_code == 0
