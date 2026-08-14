from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.draft import board as draft_board
from ffapp.draft import export as draft_export

runner = CliRunner()

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={},
    overrides={},
)

_BOARD = pl.DataFrame(
    {
        "overall_rank": [1],
        "pos_rank": [1],
        "tier": [1],
        "player": ["Elite RB"],
        "position": ["RB"],
        "team": ["DET"],
        "bye_week": [6],
        "proj_points_adj": [280.0],
        "proj_ppg": [16.5],
        "expected_games": [17.0],
        "vor": [150.0],
        "dispersion": [5.0],
        "n_sources": [4],
        "adp": [2.0],
        "adp_sd": [1.0],
        "value_vs_adp": [0],
        "p_avail_next": [0.1],
        "opportunity_cost": [40.0],
        "playoff_sos": [None],
        "as_of_utc": ["2026-08-12T00:00:00+00:00"],
        "git_commit": ["abc123"],
    }
)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
    )


@pytest.fixture(autouse=True)
def _primary_league(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)


def test_draft_board_writes_csv_and_reports_row_count(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(draft_board, "build_draft_board", lambda league, settings, **kwargs: _BOARD)

    result = runner.invoke(cli.app, ["draft", "board"])

    assert result.exit_code == 0
    assert "1 players" in result.output
    output_path = fixture_settings.data_root / "outputs" / "draft_board_2026.csv"
    assert output_path.exists()
    written = pl.read_csv(output_path)
    assert written["player"].to_list() == ["Elite RB"]


def test_draft_board_defaults_season_to_the_leagues_own_season(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    seasons_seen = []
    monkeypatch.setattr(
        draft_board,
        "build_draft_board",
        lambda league, settings, **kwargs: seasons_seen.append(kwargs["season"]) or _BOARD,
    )

    runner.invoke(cli.app, ["draft", "board"])

    assert seasons_seen == [2026]


def test_draft_board_season_override_is_respected(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(draft_board, "build_draft_board", lambda league, settings, **kwargs: _BOARD)

    result = runner.invoke(cli.app, ["draft", "board", "--season", "2025"])

    assert result.exit_code == 0
    assert (fixture_settings.data_root / "outputs" / "draft_board_2025.csv").exists()


def test_draft_board_reports_no_rankings_sources_error(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    def raise_no_sources(league: object, settings: object, **kwargs: object) -> pl.DataFrame:
        raise draft_board.NoRankingsSourcesAvailableError("nothing to aggregate")

    monkeypatch.setattr(draft_board, "build_draft_board", raise_no_sources)

    result = runner.invoke(cli.app, ["draft", "board"])

    assert result.exit_code == 1
    assert "nothing to aggregate" in result.output


def test_draft_board_reports_not_enough_picks_error(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    def raise_not_enough_picks(league: object, settings: object, **kwargs: object) -> pl.DataFrame:
        raise draft_board.NotEnoughPicksError("only 1 pick")

    monkeypatch.setattr(draft_board, "build_draft_board", raise_not_enough_picks)

    result = runner.invoke(cli.app, ["draft", "board"])

    assert result.exit_code == 1
    assert "only 1 pick" in result.output


# --- draft export --------------------------------------------------------------

_PICK_CONTEXT = draft_board.PickContext(my_roster_id=7, my_slot=3, my_picks=[3, 18, 27])


def _stub_export_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(draft_board, "build_draft_board", lambda league, settings, **kwargs: _BOARD)
    monkeypatch.setattr(
        draft_board, "resolve_pick_context", lambda league, settings, **kwargs: _PICK_CONTEXT
    )
    monkeypatch.setattr(
        draft_export.rankings, "fetch_adp", lambda *args, **kwargs: Path("adp.json")
    )
    monkeypatch.setattr(draft_export, "_RANKINGS_SOURCE_FETCHERS", {})


def test_draft_export_writes_html_and_csv(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    _stub_export_bundle(monkeypatch)

    result = runner.invoke(cli.app, ["draft", "export"])

    assert result.exit_code == 0
    html_path = fixture_settings.data_root / "outputs" / "draft_board_2026_export.html"
    csv_path = fixture_settings.data_root / "outputs" / "draft_board_2026_export.csv"
    assert html_path.exists()
    assert csv_path.exists()
    assert "Elite RB" in html_path.read_text(encoding="utf-8")
    assert pl.read_csv(csv_path)["player"].to_list() == ["Elite RB"]


def test_draft_export_respects_custom_out_path(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    _stub_export_bundle(monkeypatch)
    custom_out = tmp_path / "phone" / "board.html"

    result = runner.invoke(cli.app, ["draft", "export", "--out", str(custom_out)])

    assert result.exit_code == 0
    assert custom_out.exists()
    assert (tmp_path / "phone" / "board.csv").exists()


def test_draft_export_reports_no_rankings_sources_error(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    def raise_no_sources(league: object, settings: object, **kwargs: object) -> pl.DataFrame:
        raise draft_board.NoRankingsSourcesAvailableError("nothing to aggregate")

    monkeypatch.setattr(draft_board, "build_draft_board", raise_no_sources)

    result = runner.invoke(cli.app, ["draft", "export"])

    assert result.exit_code == 1
    assert "nothing to aggregate" in result.output
