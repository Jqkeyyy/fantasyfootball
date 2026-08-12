import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.config import LeagueConfig
from ffapp.scoring import golden

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


def _result(
    *, passed: bool, disagreements: list[golden.Disagreement] | None = None
) -> golden.GoldenTestResult:
    disagreements = disagreements or []
    return golden.GoldenTestResult(
        total_player_weeks=100,
        disagreements=disagreements,
        agreement_rate=1 - len(disagreements) / 100,
        passed=passed,
    )


def test_scoring_validate_defaults_to_primary_league(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)
    monkeypatch.setattr(golden, "run_golden_test", lambda slug, **kwargs: _result(passed=True))

    result = runner.invoke(cli.app, ["scoring", "validate"])

    assert result.exit_code == 0
    assert "test-league" in result.output


def test_scoring_validate_exits_zero_on_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(golden, "run_golden_test", lambda slug, **kwargs: _result(passed=True))

    result = runner.invoke(cli.app, ["scoring", "validate", "--league", "test-league"])

    assert result.exit_code == 0
    assert "PASS" in result.output


def test_scoring_validate_exits_nonzero_on_fail_and_logs_disagreements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disagreement = golden.Disagreement(
        week=3,
        player_id="00-0031234",
        sleeper_points=8.84,
        computed_points=6.0,
        missing_computed_row=False,
    )
    monkeypatch.setattr(
        golden,
        "run_golden_test",
        lambda slug, **kwargs: _result(passed=False, disagreements=[disagreement]),
    )

    result = runner.invoke(cli.app, ["scoring", "validate", "--league", "test-league"])

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "00-0031234" in result.output


def test_scoring_validate_all_leagues_runs_each_and_fails_if_any_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_league = LeagueConfig(
        slug="other-league",
        display_name="Other League",
        is_primary=False,
        league_id="2",
        season=2026,
        league_cache={},
        overrides={},
    )
    monkeypatch.setattr(cli, "load_all_leagues", lambda: [_LEAGUE, other_league])

    def fake_run(slug: str, **kwargs: object) -> golden.GoldenTestResult:
        return _result(passed=(slug == "test-league"))

    monkeypatch.setattr(golden, "run_golden_test", fake_run)

    result = runner.invoke(cli.app, ["scoring", "validate", "--all-leagues"])

    assert result.exit_code == 1
    assert "test-league" in result.output
    assert "other-league" in result.output


def test_scoring_validate_reports_no_played_season_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_no_season(slug: str, **kwargs: object) -> golden.GoldenTestResult:
        raise golden.NoPlayedSeasonError(f"League {slug} has no previous_league_id.")

    monkeypatch.setattr(golden, "run_golden_test", raise_no_season)

    result = runner.invoke(cli.app, ["scoring", "validate", "--league", "test-league"])

    assert result.exit_code == 1
    assert "no previous_league_id" in result.output.lower()
