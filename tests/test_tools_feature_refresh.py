from __future__ import annotations

from pathlib import Path

import polars as pl

from ffapp.config import CacheSettings, Settings
from ffapp.tools.feature_refresh import (
    _extend_live_rosters,
    _merge_existing_history,
    _read_current_partition,
    _read_history_and_current,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
    )


def test_history_and_current_fetch_includes_live_actual_partition(tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []

    def fetcher(seasons: object, *, offline: object, settings: Settings) -> Path:
        calls.append((seasons, offline))
        label = "history" if isinstance(seasons, list) else "current"
        path = tmp_path / f"{label}.parquet"
        season = 2025 if label == "history" else 2026
        pl.DataFrame({"season": [season], "week": [1]}).write_parquet(path)
        return path

    result = _read_history_and_current(
        fetcher,
        [2025],
        2026,
        offline=False,
        settings=_settings(tmp_path),
        current_required=True,
    )

    assert result["season"].to_list() == [2025, 2026]
    assert calls == [([2025], True), (2026, False)]


def test_optional_current_partition_can_lag_without_losing_history(tmp_path: Path) -> None:
    def fetcher(seasons: object, *, offline: object, settings: Settings) -> Path:
        if not isinstance(seasons, list):
            raise RuntimeError("current source not published")
        path = tmp_path / "history.parquet"
        pl.DataFrame({"season": [2025], "week": [18]}).write_parquet(path)
        return path

    result = _read_history_and_current(
        fetcher,
        [2025],
        2026,
        offline=False,
        settings=_settings(tmp_path),
        current_required=False,
    )

    assert result.to_dicts() == [{"season": 2025, "week": 18}]


def test_incremental_reader_does_not_load_historical_rows(tmp_path: Path) -> None:
    calls: list[object] = []

    def fetcher(seasons: object, *, offline: object, settings: Settings) -> Path:
        calls.append(seasons)
        path = tmp_path / "current.parquet"
        pl.DataFrame({"season": [2026], "week": [3]}).write_parquet(path)
        return path

    result = _read_current_partition(
        fetcher,
        [2020, 2021, 2022, 2023, 2024, 2025],
        2026,
        offline=False,
        settings=_settings(tmp_path),
        current_required=True,
    )

    assert result.to_dicts() == [{"season": 2026, "week": 3}]
    assert calls == [2026, [2020, 2021, 2022, 2023, 2024, 2025]]


def test_incremental_merge_replaces_only_live_season(tmp_path: Path) -> None:
    path = tmp_path / "features.parquet"
    pl.DataFrame({"season": [2025, 2026], "week": [18, 2], "value": [10, 20]}).write_parquet(path)
    current = pl.DataFrame({"season": [2026], "week": [3], "value": [30]})

    result = _merge_existing_history(path, current, 2026)

    assert result.sort(["season", "week"]).to_dicts() == [
        {"season": 2025, "week": 18, "value": 10},
        {"season": 2026, "week": 3, "value": 30},
    ]


def test_live_rosters_are_carried_into_future_schedule_weeks() -> None:
    rosters = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "gsis_id": ["p1", "p1"],
            "team": ["KC", "KC"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 2, 3],
            "home_team": ["KC", "KC", "KC"],
        }
    )

    result = _extend_live_rosters(rosters, schedule, 2026)

    assert result.sort("week").select("week", "gsis_id").to_dicts() == [
        {"week": 1, "gsis_id": "p1"},
        {"week": 2, "gsis_id": "p1"},
        {"week": 3, "gsis_id": "p1"},
    ]
