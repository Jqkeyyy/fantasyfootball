"""`ffapp project --week N` CLI wiring (SPEC §6.2, §11.8; task 1.18): small
synthetic features fixture, same fixture-vs-live-run convention as
`test_cli_evaluate.py` -- the real end-to-end run against an already-
played 2015-2025 week is documented in docs/JOURNAL.md.
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

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

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
    for season, weeks in ((2020, (1, 2, 3, 4, 5, 6, 7, 8)), (2021, (1,))):
        for week in weeks:
            for i, player in enumerate(("p1", "p2", "p3")):
                share = i / 3
                rows.append(
                    {
                        "player_id": player,
                        "season": season,
                        "week": week,
                        "position": "RB",
                        "team": "AAA",
                        "availability_flag": True,
                        "target": 10.0 + week + share,
                        "target_share_ewm_3": share,
                        "as_of_utc": f"{season}-09-{week:02d}T00:00:00Z",
                        **{k: v for k, v in _DEFAULT_FEATURES.items() if k != "target_share_ewm_3"},
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)
    _features().write_parquet(features_dir / "player_week_features.parquet")
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
        seasons=SeasonsSettings(train_start=2020, current=2021),
        model=ModelSettings(
            min_train_rows=1,
            retrain_cadence_weeks=1,
            lightgbm=_FAST_PARAMS,
            # This CLI test fixture predates task 1.20's projection_source
            # config (SPEC-ADDENDUM-04.md §C); "direct" matches its own
            # real intent -- testing the CLI's upsert/error-handling
            # plumbing with the fast points model, not a real B3 network
            # fetch (CLAUDE.md: no live network calls in tests).
            projection_source="direct",
        ),
    )


def test_project_writes_projections_parquet_with_every_spec_column(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    result = runner.invoke(cli.app, ["project", "--season", "2020", "--week", "8"])

    assert result.exit_code == 0, result.output
    output_path = fixture_settings.data_root / "outputs" / "projections.parquet"
    assert output_path.exists()
    written = pl.read_parquet(output_path)
    assert written.height == 3
    expected_columns = {
        "player_id",
        "season",
        "week",
        "p_active",
        "mean",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "model_version",
        "as_of_utc",
        "feature_hash",
        "git_commit",
    }
    assert expected_columns.issubset(set(written.columns))


def test_project_upserts_a_second_week_alongside_the_first(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    runner.invoke(cli.app, ["project", "--season", "2020", "--week", "7"])
    result = runner.invoke(cli.app, ["project", "--season", "2020", "--week", "8"])

    assert result.exit_code == 0, result.output
    written = pl.read_parquet(fixture_settings.data_root / "outputs" / "projections.parquet")
    assert set(written["week"].to_list()) == {7, 8}


def test_project_exits_nonzero_when_the_target_week_has_no_rows(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    result = runner.invoke(cli.app, ["project", "--season", "2020", "--week", "99"])

    assert result.exit_code == 1
    assert "No projections generated" in result.output


def test_project_defaults_season_to_settings_seasons_current(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    # settings.seasons.current == 2021, which has real week-1 rows in the fixture
    result = runner.invoke(cli.app, ["project", "--week", "1"])

    assert result.exit_code == 0, result.output


def test_project_wires_consensus_b3_source_end_to_end(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    """SPEC-ADDENDUM-04.md §C: `settings.model.projection_source` fully
    drives the CLI, including building `players_dim` for the real
    resolution step -- network calls mocked (CLAUDE.md: no live network
    calls in tests)."""
    from ffapp.config import ModelSettings
    from ffapp.models import baselines as baselines_module

    b3_source_settings = Settings(
        data_root=fixture_settings.data_root,
        sleeper_username=fixture_settings.sleeper_username,
        cache=fixture_settings.cache,
        seasons=fixture_settings.seasons,
        model=ModelSettings(
            min_train_rows=1,
            retrain_cadence_weeks=1,
            lightgbm=_FAST_PARAMS,
            projection_source="consensus_b3",
        ),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: b3_source_settings)
    monkeypatch.setattr(
        cli.nflverse, "fetch_player_ids", lambda **kwargs: tmp_path / "crosswalk.csv"
    )
    monkeypatch.setattr(cli.sleeper, "fetch_players", lambda **kwargs: tmp_path / "sleeper.json")
    monkeypatch.setattr(
        cli.mapping, "build_players_dim", lambda *args, **kwargs: pl.DataFrame({"player_id": []})
    )

    interim_dir = b3_source_settings.data_root / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        schema={"player_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}
    ).write_parquet(interim_dir / "b3_predictions.parquet")

    fake_b3 = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "season": [2020, 2020, 2020],
            "week": [8, 8, 8],
            "b3_points": [11.0, 22.0, 33.0],
        }
    )
    monkeypatch.setattr(baselines_module, "fetch_b3_for_week", lambda *args, **kwargs: fake_b3)

    result = runner.invoke(cli.app, ["project", "--season", "2020", "--week", "8"])

    assert result.exit_code == 0, result.output
    written = pl.read_parquet(b3_source_settings.data_root / "outputs" / "projections.parquet")
    assert set(written["projection_source"].to_list()) == {"consensus_b3"}
    by_player = {row["player_id"]: row["mean"] for row in written.to_dicts()}
    assert by_player["p1"] == pytest.approx(11.0)
    assert by_player["p2"] == pytest.approx(22.0)
    assert by_player["p3"] == pytest.approx(33.0)


def test_project_exits_with_a_clear_error_when_the_b3_archive_is_missing(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    from ffapp.config import ModelSettings

    b3_source_settings = Settings(
        data_root=fixture_settings.data_root,
        sleeper_username=fixture_settings.sleeper_username,
        cache=fixture_settings.cache,
        seasons=fixture_settings.seasons,
        model=ModelSettings(
            min_train_rows=1,
            retrain_cadence_weeks=1,
            lightgbm=_FAST_PARAMS,
            projection_source="consensus_b3",
        ),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: b3_source_settings)
    monkeypatch.setattr(
        cli.nflverse, "fetch_player_ids", lambda **kwargs: tmp_path / "crosswalk.csv"
    )
    monkeypatch.setattr(cli.sleeper, "fetch_players", lambda **kwargs: tmp_path / "sleeper.json")
    monkeypatch.setattr(
        cli.mapping, "build_players_dim", lambda *args, **kwargs: pl.DataFrame({"player_id": []})
    )
    # No data/interim/b3_predictions.parquet written -- real, missing-archive case.

    result = runner.invoke(cli.app, ["project", "--season", "2020", "--week", "8"])

    assert result.exit_code == 1
    assert "b3_predictions.parquet" in result.output


# --- project --from-week/--through-week/--league (task 1.21 CLI wiring) -----------


_ROS_LEAGUE = LeagueConfig(
    slug="ros-test-league",
    display_name="ROS Test League",
    is_primary=True,
    league_id="1",
    season=2020,
    league_cache={"scoring_settings": {"pass_yd": 0.04}},
    overrides={},
)


def _write_ros_interim_tables(settings: Settings) -> None:
    interim_dir = settings.data_root / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        schema={"season": pl.Int64, "week": pl.Int64, "gameday": pl.Utf8, "home_team": pl.Utf8}
    ).write_parquet(interim_dir / "schedule.parquet")
    pl.DataFrame(
        schema={"position_group": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "team": pl.Utf8}
    ).write_parquet(interim_dir / "defense_position_allowed.parquet")
    pl.DataFrame(
        schema={"player_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}
    ).write_parquet(interim_dir / "b3_predictions.parquet")


def test_project_command_accepts_from_week_through_week_and_league(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    """A smoke test at the CLI-wiring level (mirrors this file's existing
    `project_command` tests' own style) -- proves the new flags parse and
    route to `models.predict_ros.project_week_range`, not that the real
    math is correct (already proven in tests/test_models_predict_ros.py).
    """
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_league", lambda slug: _ROS_LEAGUE)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _ROS_LEAGUE)
    monkeypatch.setattr(
        cli.nflverse, "fetch_player_ids", lambda **kwargs: tmp_path / "crosswalk.csv"
    )
    monkeypatch.setattr(cli.sleeper, "fetch_players", lambda **kwargs: tmp_path / "sleeper.json")
    monkeypatch.setattr(
        cli.mapping, "build_players_dim", lambda *args, **kwargs: pl.DataFrame({"player_id": []})
    )
    monkeypatch.setattr(
        cli.ros_consensus, "fetch_season_consensus", lambda *args, **kwargs: {}
    )
    _write_ros_interim_tables(fixture_settings)

    calls: dict[str, object] = {}

    def fake_project_week_range(*args: object, **kwargs: object) -> pl.DataFrame:
        calls["args"] = args
        return pl.DataFrame(
            {
                "player_id": ["p1"],
                "season": [2020],
                "week": [8],
                "position": ["RB"],
                "team": ["AAA"],
                "opponent_team": [None],
                "mean": [12.0],
                "q10": [8.0],
                "q25": [10.0],
                "q50": [12.0],
                "q75": [14.0],
                "q90": [16.0],
                "is_current_week": [True],
                "as_of_utc": ["2020-10-01T00:00:00+00:00"],
            }
        )

    monkeypatch.setattr(cli.predict_ros, "project_week_range", fake_project_week_range)

    result = runner.invoke(
        cli.app,
        [
            "project",
            "--season",
            "2020",
            "--week",
            "8",
            "--from-week",
            "8",
            "--through-week",
            "10",
            "--league",
            "ros-test-league",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "args" in calls
    output_path = (
        fixture_settings.data_root / "outputs" / "ros-test-league" / "projections_ros.parquet"
    )
    assert output_path.exists()
    written = pl.read_parquet(output_path)
    assert written["player_id"].to_list() == ["p1"]


def test_project_command_requires_both_from_week_and_through_week(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)

    result = runner.invoke(
        cli.app, ["project", "--season", "2020", "--week", "8", "--from-week", "8"]
    )

    assert result.exit_code == 1
    assert "--from-week and --through-week must be given together" in result.output
