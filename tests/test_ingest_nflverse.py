import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ffapp.cache.offline import OfflineCacheMiss, StaleCacheError, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import nflverse

FIXTURE_CSV = (
    "mfl_id,gsis_id,sleeper_id,pfr_id,espn_id,name,merge_name,position,team,birthdate\n"
    "1,00-0031234,421,MahoPa00,3139477,Patrick Mahomes,patrick mahomes,QB,KC,1995-09-17\n"
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"nflverse_player_ids": 168},
            warn_on_stale=True,
        ),
    )


def _age_stamp(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _stale_meta() -> dict[str, str]:
    return {
        "source": "nflverse",
        "fetched_at_utc": _age_stamp(200),
        "cache_key": "nflverse_player_ids",
    }


def test_fetch_player_ids_online_writes_raw_csv_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(nflverse, "_get_csv", lambda: FIXTURE_CSV)

    path = nflverse.fetch_player_ids(offline=False, settings=settings)

    assert path.exists()
    assert path.read_text() == FIXTURE_CSV
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "nflverse"
    assert meta["cache_key"] == "nflverse_player_ids"


def test_fetch_player_ids_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom() -> str:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(nflverse, "_get_csv", _boom)
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    write_sidecar(
        path, source="nflverse", call=nflverse.CROSSWALK_URL, cache_key="nflverse_player_ids"
    )

    result = nflverse.fetch_player_ids(offline=True, settings=settings)

    assert result == path


def test_fetch_player_ids_offline_without_cache_raises_offline_cache_miss(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(nflverse, "_get_csv", lambda: pytest.fail("should not fetch"))

    with pytest.raises(OfflineCacheMiss) as exc_info:
        nflverse.fetch_player_ids(offline=True, settings=settings)

    message = str(exc_info.value)
    assert "nflverse" in message
    assert "player_ids" in message
    assert "cache warm" in message or "ingest" in message


def test_fetch_player_ids_offline_with_stale_cache_logs_warning(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FFAPP_CACHE_STRICT", raising=False)
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    sidecar_path(path).write_text(json.dumps(_stale_meta()))

    with caplog.at_level(logging.WARNING):
        result = nflverse.fetch_player_ids(offline=True, settings=settings)

    assert result == path
    assert any("stale" in record.message.lower() for record in caplog.records)


def test_fetch_player_ids_offline_with_stale_cache_and_strict_env_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    path = settings.cache.root / "nflverse" / "player_ids.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FIXTURE_CSV)
    sidecar_path(path).write_text(json.dumps(_stale_meta()))

    with pytest.raises(StaleCacheError):
        nflverse.fetch_player_ids(offline=True, settings=settings)


# --- fetch_player_stats / fetch_team_stats / fetch_schedules -------------------
# All three share `_fetch_nflreadpy_parquet`, so one function's tests cover the
# mechanism; the others just confirm they're wired to the right nflreadpy loader.


@pytest.fixture
def stats_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"nflverse_player_stats": 168},
            warn_on_stale=True,
        ),
    )


def test_fetch_player_stats_online_writes_parquet_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"player_id": ["00-0031234"], "week": [1], "passing_yards": [250]})
    monkeypatch.setattr(nflverse.nfl, "load_player_stats", lambda seasons: fixture_df)

    path = nflverse.fetch_player_stats(2025, offline=False, settings=stats_settings)

    assert path.exists()
    assert pl.read_parquet(path).equals(fixture_df)
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "nflverse"
    assert meta["cache_key"] == "nflverse_player_stats"
    assert meta["rows"] == 1


def test_fetch_player_stats_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    def _boom(seasons: list[int]) -> pl.DataFrame:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(nflverse.nfl, "load_player_stats", _boom)
    path = stats_settings.cache.root / "nflverse" / "player_stats_2025.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"player_id": ["00-0031234"]}).write_parquet(path)
    write_sidecar(path, source="nflverse", call="x", cache_key="nflverse_player_stats")

    result = nflverse.fetch_player_stats(2025, offline=True, settings=stats_settings)

    assert result == path


def test_fetch_player_stats_offline_without_cache_raises_offline_cache_miss(
    stats_settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        nflverse.fetch_player_stats(2025, offline=True, settings=stats_settings)

    message = str(exc_info.value)
    assert "nflverse" in message
    assert "seasons=2025" in message


def test_fetch_team_stats_online_calls_load_team_stats(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"team": ["KC"], "week": [1], "def_sacks": [3]})
    monkeypatch.setattr(nflverse.nfl, "load_team_stats", lambda seasons: fixture_df)

    path = nflverse.fetch_team_stats(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_team_stats"


def test_fetch_pbp_online_calls_load_pbp(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"game_id": ["2025_01_KC_BAL"], "week": [1], "defteam": ["BAL"]})
    monkeypatch.setattr(nflverse.nfl, "load_pbp", lambda seasons: fixture_df)

    path = nflverse.fetch_pbp(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_pbp"


def test_fetch_schedules_online_calls_load_schedules(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"game_id": ["2025_01_KC_BAL"], "home_score": [27]})
    monkeypatch.setattr(nflverse.nfl, "load_schedules", lambda seasons: fixture_df)

    path = nflverse.fetch_schedules(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_schedules"


# --- multi-season support (task 1.1) --------------------------------------------


def test_fetch_player_stats_accepts_a_season_range(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    calls = []
    fixture_df = pl.DataFrame({"player_id": ["1"], "season": [2015]})
    monkeypatch.setattr(
        nflverse.nfl, "load_player_stats", lambda seasons: calls.append(seasons) or fixture_df
    )

    path = nflverse.fetch_player_stats([2015, 2016, 2017], offline=False, settings=stats_settings)

    assert calls == [[2015, 2016, 2017]]
    assert path.name == "player_stats_2015-2017.parquet"


def test_fetch_player_stats_single_season_filename_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    """Backward compatible with task 0.5's golden test, which calls with a
    bare int and expects the same filename as before this task."""
    monkeypatch.setattr(nflverse.nfl, "load_player_stats", lambda seasons: pl.DataFrame({"a": [1]}))

    path = nflverse.fetch_player_stats(2025, offline=False, settings=stats_settings)

    assert path.name == "player_stats_2025.parquet"


# --- fetch_snap_counts / fetch_depth_charts / fetch_rosters / fetch_injuries ------


def test_fetch_snap_counts_online_calls_load_snap_counts(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"player": ["A"], "week": [1], "offense_snaps": [50]})
    monkeypatch.setattr(nflverse.nfl, "load_snap_counts", lambda seasons: fixture_df)

    path = nflverse.fetch_snap_counts(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_snap_counts"


def test_fetch_depth_charts_online_calls_load_depth_charts(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"gsis_id": ["1"], "week": [1], "depth_position": ["WR1"]})
    monkeypatch.setattr(nflverse.nfl, "load_depth_charts", lambda seasons: fixture_df)

    path = nflverse.fetch_depth_charts(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_depth_charts"


def test_fetch_rosters_online_calls_load_rosters_weekly(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    """Not `load_rosters` (season-level -- see fetch_rosters' own
    docstring for the real row-count gap this task 1.9 fix closed)."""
    fixture_df = pl.DataFrame({"gsis_id": ["1"], "team": ["KC"], "position": ["QB"]})
    monkeypatch.setattr(nflverse.nfl, "load_rosters_weekly", lambda seasons: fixture_df)

    path = nflverse.fetch_rosters(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_rosters"


def test_fetch_injuries_online_calls_load_injuries(
    monkeypatch: pytest.MonkeyPatch, stats_settings: Settings
) -> None:
    fixture_df = pl.DataFrame({"gsis_id": ["1"], "week": [1], "report_status": ["Questionable"]})
    monkeypatch.setattr(nflverse.nfl, "load_injuries", lambda seasons: fixture_df)

    path = nflverse.fetch_injuries(2025, offline=False, settings=stats_settings)

    assert pl.read_parquet(path).equals(fixture_df)
    assert json.loads(sidecar_path(path).read_text())["cache_key"] == "nflverse_injuries"


# --- normalize_schedule -----------------------------------------------------------


def test_normalize_schedule_maps_to_canonical_columns() -> None:
    raw = pl.DataFrame(
        {
            "game_id": ["2025_01_KC_BAL"],
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "gameday": ["2025-09-05"],
            "gametime": ["20:20"],
            "weekday": ["Friday"],
            "spread_line": [-2.5],
            "total_line": [48.5],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "stadium_id": ["KAN00"],
            "home_rest": [7],
            "away_rest": [7],
        }
    )

    result = nflverse.normalize_schedule(raw)

    row = result.row(0, named=True)
    assert row["season_type"] == "REG"
    assert row["home_rest"] == 7
    assert row["away_rest"] == 7
    assert row["spread_line"] == -2.5
    # kickoff_utc needs config/stadiums.csv's timezones (this same task,
    # not yet built when this function runs) -- not guessed here.
    assert row["kickoff_utc"] is None
    # home_implied_total/away_implied_total ARE computed now -- see below.


def test_normalize_schedule_computes_implied_totals_with_verified_sign() -> None:
    """Sign convention verified against real data (task 1.3): positive
    spread_line means the home team is favoured. total=48.5, spread=-2.5
    (home is a 2.5-point underdog) -> home gets the smaller share."""
    raw = pl.DataFrame(
        {
            "game_id": ["2025_01_KC_BAL"],
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "gameday": ["2025-09-05"],
            "gametime": ["20:20"],
            "weekday": ["Friday"],
            "spread_line": [-2.5],
            "total_line": [48.5],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "stadium_id": ["KAN00"],
            "home_rest": [7],
            "away_rest": [7],
        }
    )

    result = nflverse.normalize_schedule(raw)

    row = result.row(0, named=True)
    assert row["home_implied_total"] == pytest.approx(23.0)  # 48.5/2 - 2.5/2
    assert row["away_implied_total"] == pytest.approx(25.5)  # 48.5/2 + 2.5/2
    # internal consistency: they sum to the total and diff to the spread
    assert row["home_implied_total"] + row["away_implied_total"] == pytest.approx(48.5)
    assert row["home_implied_total"] - row["away_implied_total"] == pytest.approx(-2.5)


def test_normalize_schedule_output_column_order_matches_spec() -> None:
    raw = pl.DataFrame(
        {
            "game_id": ["1"],
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "gameday": ["2025-09-05"],
            "gametime": ["20:20"],
            "weekday": ["Friday"],
            "spread_line": [-2.5],
            "total_line": [48.5],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "stadium_id": ["KAN00"],
            "home_rest": [7],
            "away_rest": [7],
        }
    )

    result = nflverse.normalize_schedule(raw)

    assert result.columns == [
        "game_id",
        "season",
        "week",
        "season_type",
        "home_team",
        "away_team",
        "gameday",
        "gametime",
        "weekday",
        "kickoff_utc",
        "spread_line",
        "total_line",
        "home_implied_total",
        "away_implied_total",
        "roof",
        "surface",
        "stadium_id",
        "home_rest",
        "away_rest",
    ]


# --- normalize_injuries -----------------------------------------------------------


def test_normalize_injuries_maps_gsis_id_to_player_id() -> None:
    raw = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "season": [2025],
            "week": [3],
            "team": ["KC"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation in Practice"],
            "report_primary_injury": ["Ankle"],
            "date_modified": ["2025-09-20T18:00:00Z"],
        }
    )

    result = nflverse.normalize_injuries(raw)

    row = result.row(0, named=True)
    assert row["player_id"] == "00-0031234"
    assert row["team"] == "KC"
    assert row["report_status"] == "Questionable"
    assert row["date_modified"] == "2025-09-20T18:00:00Z"


def test_normalize_injuries_casts_season_and_week_to_int32() -> None:
    """Real gotcha, found by running the actual 2015-2025 build: nflreadpy's
    load_injuries() hands back season/week as Float64 (no nulls, just an
    upstream quirk unique to this source), which would break any join
    against the other five interim tables' Int32 season/week columns."""
    raw = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "season": [2025.0],
            "week": [3.0],
            "team": ["KC"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation in Practice"],
            "report_primary_injury": ["Ankle"],
            "date_modified": ["2025-09-20T18:00:00Z"],
        }
    )

    result = nflverse.normalize_injuries(raw)

    assert result.schema["season"] == pl.Int32
    assert result.schema["week"] == pl.Int32
    row = result.row(0, named=True)
    assert row["season"] == 2025
    assert row["week"] == 3


def test_normalize_injuries_keeps_rows_with_no_gsis_id() -> None:
    """CLAUDE.md rule 4: never silently drop rows -- a practice-squad player
    nflverse hasn't linked a gsis_id for yet still gets a row, just with a
    null player_id."""
    raw = pl.DataFrame(
        {
            "gsis_id": [None],
            "season": [2025],
            "week": [3],
            "team": ["KC"],
            "report_status": ["Out"],
            "practice_status": [None],
            "report_primary_injury": ["Hamstring"],
            "date_modified": ["2025-09-20T18:00:00Z"],
        },
        schema_overrides={"practice_status": pl.Utf8},
    )

    result = nflverse.normalize_injuries(raw)

    assert result.height == 1
    assert result.row(0, named=True)["player_id"] is None


def test_normalize_injuries_blanks_a_literal_newline_practice_status_to_null() -> None:
    """Real gotcha, found live: 213 real rows across 2015-2025 carry a
    literal "\n" (sometimes with trailing spaces) instead of an empty
    practice_status -- a missing value with extra steps, not a real
    fourth category alongside Full/Limited/Did Not Participate."""
    raw = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "season": [2025],
            "week": [3],
            "team": ["KC"],
            "report_status": ["Out"],
            "practice_status": ["\n    "],
            "report_primary_injury": ["Ankle"],
            "date_modified": ["2025-09-20T18:00:00Z"],
        }
    )

    result = nflverse.normalize_injuries(raw)

    assert result.row(0, named=True)["practice_status"] is None


# --- fetch_ff_opportunity (task 1.2) -----------------------------------------------


@pytest.fixture
def ffopp_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"ffopportunity_weekly": 24},
            warn_on_stale=True,
        ),
    )


def test_fetch_ff_opportunity_online_writes_parquet_sidecar_and_license(
    monkeypatch: pytest.MonkeyPatch, ffopp_settings: Settings
) -> None:
    fixture_df = pl.DataFrame(
        {
            "player_id": ["00-0033873"],
            "season": ["2025"],
            "week": [1.0],
            "total_fantasy_points_exp": [18.4],
        }
    )
    monkeypatch.setattr(nflverse.nfl, "load_ff_opportunity", lambda seasons, stat_type: fixture_df)

    path = nflverse.fetch_ff_opportunity(2025, offline=False, settings=ffopp_settings)

    assert path.name == "weekly_2025.parquet"
    assert pl.read_parquet(path).equals(fixture_df)
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "ffopportunity"
    assert meta["cache_key"] == "ffopportunity_weekly"
    license_path = path.parent / "LICENSE.txt"
    assert license_path.exists()
    assert "CC BY-SA" in license_path.read_text()


def test_fetch_ff_opportunity_writes_to_its_own_directory_not_nflverse(
    monkeypatch: pytest.MonkeyPatch, ffopp_settings: Settings
) -> None:
    """SPEC §6.1: ffopportunity's CC-BY-SA licence is distinct from the rest
    of nflverse's plain CC-BY data -- must not land in the shared
    data/raw/nflverse/ directory, which carries no such obligation."""
    monkeypatch.setattr(
        nflverse.nfl,
        "load_ff_opportunity",
        lambda seasons, stat_type: pl.DataFrame({"a": [1]}),
    )

    path = nflverse.fetch_ff_opportunity(2025, offline=False, settings=ffopp_settings)

    assert path.parent.name == "ffopportunity"


def test_fetch_ff_opportunity_accepts_a_season_range(
    monkeypatch: pytest.MonkeyPatch, ffopp_settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(
        nflverse.nfl,
        "load_ff_opportunity",
        lambda seasons, stat_type: calls.append((seasons, stat_type)) or pl.DataFrame({"a": [1]}),
    )

    path = nflverse.fetch_ff_opportunity([2015, 2016, 2017], offline=False, settings=ffopp_settings)

    assert calls == [([2015, 2016, 2017], "weekly")]
    assert path.name == "weekly_2015-2017.parquet"


def test_fetch_ff_opportunity_offline_without_cache_raises_offline_cache_miss(
    ffopp_settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        nflverse.fetch_ff_opportunity(2025, offline=True, settings=ffopp_settings)

    assert "ffopportunity" in str(exc_info.value)


def test_fetch_ff_opportunity_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, ffopp_settings: Settings
) -> None:
    def _boom(seasons: list[int], stat_type: str) -> pl.DataFrame:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(nflverse.nfl, "load_ff_opportunity", _boom)
    path = ffopp_settings.cache.root / "ffopportunity" / "weekly_2025.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(path)
    write_sidecar(path, source="ffopportunity", call="x", cache_key="ffopportunity_weekly")

    result = nflverse.fetch_ff_opportunity(2025, offline=True, settings=ffopp_settings)

    assert result == path
