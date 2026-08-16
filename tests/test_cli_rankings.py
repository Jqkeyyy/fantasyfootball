"""Smoke tests for `ffapp rankings ros` -- CLI wiring only, matching
tests/test_cli_project.py's own existing style for mocking settings and
asserting exit codes/output-path behavior. Real math is covered by
tests/test_tools_ros_rankings.py and tests/test_tools_ros_aggregate.py.

`rankings_ros_command` fits the availability model and the injury hazard
model fresh and calls the Monte Carlo aggregator -- all real, expensive
work covered by its own dedicated test modules. Here those pieces are
mocked out at the same granularity `test_cli_project.py`'s own
`test_project_wires_consensus_b3_source_end_to_end` already uses for deep
internal calls (`baselines.fetch_b3_for_week`), not attempting to mock
every single internal call individually.
"""

from __future__ import annotations

import json
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

_LEAGUE = LeagueConfig(
    slug="ros-rank-league",
    display_name="ROS Rank League",
    is_primary=True,
    league_id="99",
    season=2020,
    league_cache={
        "scoring_settings": {"pass_yd": 0.04},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "BN"],
        "total_rosters": 10,
        "playoff_week_start": 15,
    },
    overrides={},
)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
        model=ModelSettings(min_train_rows=1, retrain_cadence_weeks=1, lightgbm=_FAST_PARAMS),
        seasons=SeasonsSettings(train_start=2015, current=2020),
    )


def _write_projections_ros(settings: Settings) -> None:
    out_dir = settings.data_root / "outputs" / _LEAGUE.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "player_id": ["p2", "p3"],
            "season": [2020, 2020],
            "week": [8, 8],
            "position": ["RB", "WR"],
            "team": ["KC", "KC"],
            "is_current_week": [True, True],
        }
    ).write_parquet(out_dir / "projections_ros.parquet")


def _write_features(settings: Settings) -> None:
    features_dir = settings.data_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "player_id": ["p2", "p3", "p2", "p3"],
            "season": [2020, 2020, 2020, 2020],
            "week": [8, 8, 7, 7],
        }
    ).write_parquet(features_dir / "player_week_features.parquet")


def _write_interim_tables(settings: Settings) -> None:
    interim_dir = settings.data_root / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "season": [2020] * 17,
            "week": list(range(1, 18)),
            "season_type": ["REG"] * 17,
        }
    ).write_parquet(interim_dir / "schedule.parquet")
    pl.DataFrame(schema={"player_id": pl.Utf8}).write_parquet(interim_dir / "injuries.parquet")


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "sleeper_id": ["s1", "s2", "s3"],
            "position": ["RB", "RB", "WR"],
            "active": [True, True, True],
            "team": ["KC", "KC", "KC"],
        }
    )


def _apply_common_mocks(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_league", lambda slug: _LEAGUE)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)

    monkeypatch.setattr(
        cli.nflverse, "fetch_player_ids", lambda **kwargs: tmp_path / "crosswalk.csv"
    )
    (tmp_path / "crosswalk.csv").write_text(
        "gsis_id,sleeper_id,pfr_id,espn_id,name,position,team,birthdate\n"
        "p1,s1,,,Player One,RB,KC,1998-01-01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli.sleeper, "fetch_players", lambda **kwargs: tmp_path / "sleeper.json")
    monkeypatch.setattr(cli.mapping, "build_players_dim", lambda *args, **kwargs: _players_dim())

    rosters_json_path = tmp_path / "rosters.json"
    rosters_json_path.write_text(json.dumps([{"players": ["s1"]}]), encoding="utf-8")
    monkeypatch.setattr(cli.sleeper, "fetch_rosters", lambda league_id, **kwargs: rosters_json_path)

    nflverse_rosters_path = tmp_path / "nflverse_rosters.parquet"
    pl.DataFrame(schema={"player_id": pl.Utf8}).write_parquet(nflverse_rosters_path)
    monkeypatch.setattr(
        cli.nflverse, "fetch_rosters", lambda *args, **kwargs: nflverse_rosters_path
    )

    snap_counts_path = tmp_path / "snap_counts.parquet"
    pl.DataFrame(schema={"player_id": pl.Utf8}).write_parquet(snap_counts_path)
    monkeypatch.setattr(cli.nflverse, "fetch_snap_counts", lambda *args, **kwargs: snap_counts_path)

    # Real bug found live during task 13's own e2e verification: cli.py used to
    # read interim/injuries.parquet (already normalized gsis_id -> player_id by
    # ingest.nflverse.normalize_injuries), but sim.injury.add_injury_report needs
    # the RAW nflverse schema (still gsis_id) -- same raw-table pattern as
    # fetch_rosters/fetch_snap_counts just above. This mock matches the corrected
    # production code; build_hazard_features itself is still mocked out below, so
    # the fixture's own schema doesn't need to carry a real gsis_id column.
    nflverse_injuries_path = tmp_path / "nflverse_injuries.parquet"
    pl.DataFrame(schema={"gsis_id": pl.Utf8}).write_parquet(nflverse_injuries_path)
    monkeypatch.setattr(
        cli.nflverse, "fetch_injuries", lambda *args, **kwargs: nflverse_injuries_path
    )

    monkeypatch.setattr(
        cli.availability,
        "fit_availability_model",
        lambda train_rows, **kwargs: object(),
    )
    monkeypatch.setattr(
        cli.availability,
        "predict_p_active",
        lambda model, rows: pl.Series("p_active", [0.9] * rows.height),
    )

    hazard_grid = pl.DataFrame(
        {
            "player_id": ["p2", "p3", "p2", "p3"],
            "season": [2020, 2020, 2020, 2020],
            "week": [8, 8, 7, 7],
        }
    )
    monkeypatch.setattr(cli.injury, "build_hazard_features", lambda *args, **kwargs: hazard_grid)
    monkeypatch.setattr(cli.injury, "fit_hazard_model", lambda train_rows: object())
    monkeypatch.setattr(
        cli.injury,
        "predict_p_miss",
        lambda model, rows: pl.Series("p_miss", [0.1] * rows.height),
    )

    monkeypatch.setattr(
        cli.ros_aggregate,
        "aggregate_ros",
        lambda *args, **kwargs: pl.DataFrame(
            {"player_id": ["p2", "p3"], "ros_points": [120.0, 90.0]}
        ),
    )

    _write_projections_ros(settings)
    _write_features(settings)
    _write_interim_tables(settings)


def test_rankings_ros_writes_board_and_latest_parquet(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    _apply_common_mocks(monkeypatch, fixture_settings, tmp_path)

    result = runner.invoke(cli.app, ["rankings", "ros", "--league", "ros-rank-league"])

    assert result.exit_code == 0, result.output
    latest_path = (
        fixture_settings.data_root
        / "outputs"
        / "ros-rank-league"
        / "rankings_ros"
        / "latest.parquet"
    )
    assert latest_path.exists()
    board = pl.read_parquet(latest_path)
    assert set(board["player_id"].to_list()) == {"p2", "p3"}
    assert "vor_ros" in board.columns
    assert "rank" in board.columns
    assert "rank_change" in board.columns
    # No prior real board exists yet on a first-ever run -- rank_change is
    # honestly null (SPEC-ADDENDUM-04.md §D.5), never a guessed value.
    assert board["rank_change"].is_null().all()
    # p1 is rostered (see rosters.json) -- must never appear on the free-agent board.
    assert "p1" not in board["player_id"].to_list()


def test_rankings_ros_exits_nonzero_when_projections_ros_is_missing(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: fixture_settings)
    monkeypatch.setattr(cli, "load_league", lambda slug: _LEAGUE)
    monkeypatch.setattr(cli, "load_primary_league", lambda: _LEAGUE)

    result = runner.invoke(cli.app, ["rankings", "ros", "--league", "ros-rank-league"])

    assert result.exit_code == 1
    assert "projections_ros.parquet" in result.output


def test_rankings_ros_records_rank_change_against_a_prior_run(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings, tmp_path: Path
) -> None:
    """Real wiring proof for §D.5's own rank-change requirement: a second
    real run reads the first run's `latest.parquet` as `previous_board`
    and joins a real (non-null) `rank_change` onto every player who
    appeared in both -- the CLI-level plumbing this task adds, not
    `rank_change`'s own arithmetic (covered by
    tests/test_tools_ros_rankings.py)."""
    _apply_common_mocks(monkeypatch, fixture_settings, tmp_path)

    first = runner.invoke(cli.app, ["rankings", "ros", "--league", "ros-rank-league"])
    assert first.exit_code == 0, first.output
    out_dir = fixture_settings.data_root / "outputs" / "ros-rank-league" / "rankings_ros"
    first_latest = pl.read_parquet(out_dir / "latest.parquet")
    assert first_latest["rank_change"].is_null().all()

    second = runner.invoke(cli.app, ["rankings", "ros", "--league", "ros-rank-league"])
    assert second.exit_code == 0, second.output

    second_latest = pl.read_parquet(out_dir / "latest.parquet")
    # A second real run against the same real prior board resolves a real,
    # non-null rank_change for every player who appeared in both.
    assert second_latest["rank_change"].is_null().sum() == 0
    assert second_latest["rank_change"].to_list() == [0, 0]
