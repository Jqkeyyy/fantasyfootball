"""`ffapp evaluate` CLI wiring: task 1.12's own predictions.parquet write,
plus task 1.17's report.md written alongside it in the same timestamped
directory. Small synthetic features/schedule fixtures, same
fixture-vs-live-run convention as `test_cli_draft.py` -- the real
end-to-end run needs real `features/player_week_features.parquet` (task
1.9), documented in HANDOFF.md instead.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import ffapp.cli as cli
from ffapp.config import (
    CacheSettings,
    LeagueConfig,
    LightGBMSettings,
    ModelSettings,
    SeasonsSettings,
    Settings,
)
from ffapp.models import points

runner = CliRunner()

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={"roster_positions": ["RB", "BN"], "total_rosters": 1},
    overrides={},
)

# Fast, tiny-data-friendly params -- `settings.model.lightgbm`'s own real
# defaults (n_estimators=800, min_child_samples=40) would still run on this
# fixture's handful of rows, just needlessly slowly for a unit test.
_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

# `PointsPredictor`/`AvailabilityPredictor` (task 1.15/1.14) now run for
# real in `evaluate`'s own real code path -- every feature column either
# model reads (`points.feature_columns("RB")`, `availability.FEATURE_COLUMNS`)
# needs *some* value present, even a placeholder one; this fixture only
# exercises CLI wiring, not real model quality (that's each model's own
# test module's job).
_DEFAULT_FEATURES: dict[str, object] = dict.fromkeys(points.feature_columns("RB"), 0.0)
_DEFAULT_FEATURES.update(
    {
        "report_status": "None",
        "practice_participation": "Full",
        "depth_chart_rank": 1.0,
        "age": 25.0,
    }
)


def _features() -> pl.DataFrame:
    rows = []
    for season, weeks in ((2020, (1, 2, 3)), (2021, (1,))):
        for week in weeks:
            for player, team in (("p1", "AAA"), ("p2", "BBB")):
                rows.append(
                    {
                        "player_id": player,
                        "season": season,
                        "week": week,
                        "position": "RB",
                        "team": team,
                        "availability_flag": True,
                        "target": 10.0 + week,
                        **_DEFAULT_FEATURES,
                    }
                )
    return pl.DataFrame(rows)


def _schedule() -> pl.DataFrame:
    rows = []
    for season, weeks in ((2020, (1, 2, 3)), (2021, (1,))):
        for week in weeks:
            rows.append({"season": season, "week": week})
    return pl.DataFrame(rows)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    features_dir = tmp_path / "features"
    interim_dir = tmp_path / "interim"
    features_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    _features().write_parquet(features_dir / "player_week_features.parquet")
    _schedule().write_parquet(interim_dir / "schedule.parquet")
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
        seasons=SeasonsSettings(train_start=2020, current=2021),
        model=ModelSettings(min_train_rows=1, retrain_cadence_weeks=1, lightgbm=_FAST_PARAMS),
    )


@pytest.fixture(autouse=True)
def _primary_league(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)


def test_evaluate_writes_predictions_and_report(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    result = runner.invoke(cli.app, ["evaluate", "--seasons", "2021"])

    assert result.exit_code == 0, result.output

    eval_dirs = list((fixture_settings.data_root / "outputs" / "eval").iterdir())
    assert len(eval_dirs) == 1
    output_dir = eval_dirs[0]

    predictions_path = output_dir / "predictions.parquet"
    assert predictions_path.exists()
    predictions = pl.read_parquet(predictions_path)
    assert predictions.height > 0

    report_path = output_dir / "report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Evaluation report" in report_text
    assert "2021" in report_text
    assert "## mae" in report_text
    assert "b0_positional_mean" in report_text


def test_evaluate_report_has_no_metrics_section_when_predictions_are_empty(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    # 2024 has no rows anywhere in the fixture schedule/features -- the
    # backtest loop's own weeks come from `schedule`, so an unscheduled
    # season simply produces zero predictions, not a crash.
    result = runner.invoke(cli.app, ["evaluate", "--seasons", "2024"])

    assert result.exit_code == 0, result.output
    eval_dirs = list((fixture_settings.data_root / "outputs" / "eval").iterdir())
    report_text = (eval_dirs[0] / "report.md").read_text(encoding="utf-8")
    assert "No metrics" in report_text
