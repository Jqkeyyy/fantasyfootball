from __future__ import annotations

from pathlib import Path

import polars as pl

from ffapp.config import CacheSettings, Settings
from ffapp.tools.feature_refresh import _read_history_and_current


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
