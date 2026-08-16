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
from ffapp.config import CacheSettings, LightGBMSettings, ModelSettings, SeasonsSettings, Settings
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
