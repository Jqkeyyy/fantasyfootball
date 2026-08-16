"""Task 3.8's in-season prediction logging (SPEC-ADDENDUM-05.md §B):
small fast-fitting LightGBM fixtures for `build_prediction_log`, real
DataFrame logic tested directly for the writer/backfill/check-sources
functions, and mocked network calls for `fetch_all_sources` (CLAUDE.md:
no live network calls in tests, ever).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.ingest import rankings as rankings_module
from ffapp.models import baselines as baselines_module
from ffapp.models import points as points_module
from ffapp.tools import prediction_log

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

_DEFAULT_FEATURES = dict.fromkeys(points_module.COMMON_FEATURE_COLUMNS, 0.0)
_DEFAULT_FEATURES.update(
    {
        "report_status": "None",
        "practice_participation": "Full",
        "depth_chart_rank": 1.0,
        "age": 25.0,
    }
)


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "position": "RB",
        "team": "AAA",
        "availability_flag": True,
        "target": 10.0,
        "as_of_utc": "2025-11-01T00:00:00Z",
        **_DEFAULT_FEATURES,
    }
    row.update(kwargs)
    for group in opponent.POSITION_TO_GROUPS.get(row["position"], []):
        for metric in points_module._OPPONENT_ADJ_METRICS:
            row.setdefault(f"{metric}_{group.lower()}", 0.0)
    return row


def _features(n_weeks: int = 8, rows_per_week: int = 3, target_week: int = 9) -> pl.DataFrame:
    rows = []
    for week in list(range(1, n_weeks + 1)) + [target_week]:
        for i in range(rows_per_week):
            share = i / rows_per_week
            played = share > 0.1
            rows.append(
                _row(
                    player_id=f"p{i}",
                    week=week,
                    target_share_ewm_3=share,
                    target=(10.0 + 50.0 * share) if played else 0.0,
                    availability_flag=played,
                    weeks_since_return=0.0,
                    snap_pct_trend=0.0,
                )
            )
    return pl.DataFrame(rows)


def _schedule(season: int = 2025, week: int = 9) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "week": [week],
            "season_type": ["REG"],
            "kickoff_utc": ["2025-11-01T00:00:00Z"],
        }
    )


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p0", "p1", "p2"],
            "full_name": ["Player Zero", "Player One", "Player Two"],
            "normalized_name": ["player zero", "player one", "player two"],
            "position": ["RB", "RB", "RB"],
            "sleeper_id": ["100", "101", "102"],
        }
    )


def _empty_b3_historical() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"player_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}
    )


# --- load_source_refresh_status / write_source_refresh_status ----------------------------


def test_load_source_refresh_status_defaults_to_unknown_when_missing(tmp_path) -> None:
    statuses = prediction_log.load_source_refresh_status(tmp_path / "missing.yml")

    assert statuses == dict.fromkeys(prediction_log.SOURCE_NAMES, "unknown")


def test_write_then_load_source_refresh_status_round_trips(tmp_path) -> None:
    path = tmp_path / "status.yml"
    written = {"espn": "weekly_confirmed", "cbs": "frozen"}

    prediction_log.write_source_refresh_status(written, path)
    loaded = prediction_log.load_source_refresh_status(path)

    assert loaded["espn"] == "weekly_confirmed"
    assert loaded["cbs"] == "frozen"
    assert loaded["draftsharks"] == "unknown"  # not written -- honest default


# --- fetch_all_sources ---------------------------------------------------------------------


def test_fetch_all_sources_hashes_the_real_raw_payload_and_resolves_player_id(
    tmp_path, monkeypatch
) -> None:
    espn_path = tmp_path / "espn.json"
    espn_path.write_text('{"players": []}', encoding="utf-8")
    monkeypatch.setattr(rankings_module, "fetch_espn", lambda season, **kwargs: espn_path)
    monkeypatch.setattr(
        rankings_module,
        "normalize_espn",
        lambda payload, *, season: pl.DataFrame(
            {
                "player_name": ["Player One"],
                "position": ["RB"],
                "team": ["AAA"],
                "receiving_yards": [50.0],
                "receptions": [5.0],
                "receiving_tds": [0.0],
                "rushing_yards": [0.0],
                "rushing_tds": [0.0],
            }
        ),
    )
    # every other source: real, empty-but-valid responses
    for name in ["cbs", "fantasysharks", "fftoday"]:
        monkeypatch.setattr(
            rankings_module, f"fetch_{name}", lambda *a, **k: tmp_path / "empty.json"
        )
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rankings_module, "normalize_cbs", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(rankings_module, "normalize_fantasysharks", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(rankings_module, "normalize_fftoday", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(rankings_module, "fetch_footballguys", lambda **k: tmp_path / "empty.json")
    monkeypatch.setattr(rankings_module, "normalize_footballguys", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(rankings_module, "fetch_draftsharks", lambda **k: tmp_path / "empty.json")
    monkeypatch.setattr(rankings_module, "normalize_draftsharks", lambda *a, **k: pl.DataFrame())
    # fantasypros: `_fetch_fantasypros` fetches the commit list + snapshot
    # exactly once (not via `fetch_b3_for_week`, which would fetch the
    # same commit list a second time -- a real GitHub API 403 rate-limit
    # was hit live this session from that doubling) and builds b3 via
    # `baselines.add_b3_fp_weekly_consensus` directly.
    commits_path = tmp_path / "commits.json"
    commits_path.write_text(
        '{"commits": [{"sha": "abc123", "date": "2025-10-01T00:00:00Z"}]}', encoding="utf-8"
    )
    snapshot_path = tmp_path / "fp_snapshot.csv"
    snapshot_path.write_text("player_name,pos\n", encoding="utf-8")
    monkeypatch.setattr(rankings_module, "fetch_fp_weekly_commits", lambda **k: commits_path)
    monkeypatch.setattr(rankings_module, "fetch_fp_weekly_snapshot", lambda sha, **k: snapshot_path)
    monkeypatch.setattr(
        rankings_module,
        "normalize_fp_weekly",
        lambda *a, **k: pl.DataFrame(
            schema={
                "player_name": pl.Utf8,
                "pos": pl.Utf8,
                "team": pl.Utf8,
                "b3_points": pl.Float64,
                "season": pl.Int64,
                "week": pl.Int64,
            }
        ),
    )

    points, fetch_df = prediction_log.fetch_all_sources(
        2025,
        9,
        "2025-11-01T00:00:00Z",
        {"rec_yd": 0.1, "rec": 1.0, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0},
        _players_dim(),
        league_slug="test-league",
        run_label="tuesday",
        offline=True,
        settings=None,
        now=datetime(2025, 11, 1, tzinfo=UTC),
    )

    espn_row = points["espn"].row(0, named=True)
    assert espn_row["player_id"] == "p1"
    assert espn_row["points"] == pytest.approx(5.0 + 5.0)  # 0.1*50 (rec_yd) + 1.0*5 (rec)

    espn_fetch = fetch_df.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_fetch["payload_sha256"] is not None
    assert espn_fetch["fetch_error"] is None
    assert set(fetch_df["source"].to_list()) == set(prediction_log.SOURCE_NAMES)

    # the normalize_fp_weekly mock above returns an empty-schema
    # DataFrame (0 rows), so this source honestly reports "0 rows
    # resolved" -- not a crash -- while still hashing its own real raw
    # snapshot payload, same as every other source.
    fp_fetch = fetch_df.filter(pl.col("source") == "fantasypros").row(0, named=True)
    assert fp_fetch["payload_sha256"] is not None


def test_fetch_all_sources_records_a_failing_source_without_crashing(tmp_path, monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("real network failure")

    for name in ["espn", "cbs", "fantasysharks", "fftoday"]:
        monkeypatch.setattr(rankings_module, f"fetch_{name}", boom)
    monkeypatch.setattr(rankings_module, "fetch_footballguys", boom)
    monkeypatch.setattr(rankings_module, "fetch_draftsharks", boom)
    # first real call inside `_fetch_fantasypros` now that it fetches the
    # commit list itself rather than going through `fetch_b3_for_week`
    monkeypatch.setattr(rankings_module, "fetch_fp_weekly_commits", boom)

    points, fetch_df = prediction_log.fetch_all_sources(
        2025,
        9,
        "2025-11-01T00:00:00Z",
        {},
        _players_dim(),
        league_slug="test-league",
        run_label="tuesday",
        offline=True,
        settings=None,
        now=datetime(2025, 11, 1, tzinfo=UTC),
    )

    assert points["espn"].is_empty()
    espn_fetch = fetch_df.filter(pl.col("source") == "espn").row(0, named=True)
    assert "real network failure" in espn_fetch["fetch_error"]


# --- build_prediction_log -------------------------------------------------------------------


def _mock_fetch_all_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*args: object, **kwargs: object) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
        empty = pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "position": pl.Utf8,
                "team": pl.Utf8,
                "points": pl.Float64,
            }
        )
        points = dict.fromkeys(prediction_log.SOURCE_NAMES, empty)
        points = {
            **points,
            # season sources (real season-long totals -- see module
            # docstring for why these are namespaced separately)
            "espn": pl.DataFrame(
                {"player_id": ["p1"], "position": ["RB"], "team": ["AAA"], "points": [220.0]}
            ),
            "cbs": pl.DataFrame(
                {"player_id": ["p1"], "position": ["RB"], "team": ["AAA"], "points": [240.0]}
            ),
            # the one real weekly source
            "fantasypros": pl.DataFrame(
                {
                    "player_id": ["p1", "p2"],
                    "position": ["RB", "RB"],
                    "team": ["AAA", "AAA"],
                    "points": [22.0, 5.0],
                }
            ),
        }
        fetch_df = pl.DataFrame(
            [
                {
                    "league_slug": "test-league",
                    "season": 2025,
                    "week": 9,
                    "run_label": "tuesday",
                    "source": name,
                    "payload_sha256": "abc",
                    "fetched_at_utc": "2025-11-01T00:00:00Z",
                    "refresh_status": "unknown",
                    "n_rows_parsed": 1,
                    "fetch_error": None,
                }
                for name in prediction_log.SOURCE_NAMES
            ],
            schema=prediction_log.SOURCE_FETCH_SCHEMA,
        )
        return points, fetch_df

    monkeypatch.setattr(prediction_log, "fetch_all_sources", fake)
    # models.predict.project_week's own consensus_b3 branch calls this
    # directly (not via prediction_log.fetch_all_sources) -- must be
    # mocked too, or it would try to resolve real project settings and
    # touch real cached files on disk (CLAUDE.md: no live network calls
    # in tests, and no accidental dependency on this machine's own data/).
    monkeypatch.setattr(
        baselines_module,
        "fetch_b3_for_week",
        lambda *a, **k: pl.DataFrame(
            {"player_id": ["p1"], "season": [2025], "week": [9], "b3_points": [18.0]}
        ),
    )


def test_build_prediction_log_assembles_a_real_row_per_player(monkeypatch) -> None:
    _mock_fetch_all_sources(monkeypatch)
    features = _features()
    schedule = _schedule()

    rows, fetch_df = prediction_log.build_prediction_log(
        features,
        schedule,
        2025,
        9,
        "tuesday",
        league_slug="test-league",
        scoring_settings={},
        players_dim=_players_dim(),
        train_start=2015,
        min_train_rows=10,
        lightgbm_params=_FAST_PARAMS,
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        b3_historical=_empty_b3_historical(),
        code_version="abc123",
        now=datetime(2025, 11, 1, tzinfo=UTC),
        offline=True,
        settings=None,
    )

    assert rows.height == 3  # p0, p1, p2 -- the real row universe
    assert set(rows.columns) == set(prediction_log.PREDICTION_LOG_SCHEMA)
    p1_row = rows.filter(pl.col("player_id") == "p1").row(0, named=True)
    # weekly namespace: fantasypros only -- the one real confirmed-weekly source
    assert p1_row["weekly_source_points_fantasypros"] == pytest.approx(22.0)
    assert p1_row["n_sources"] == 1
    assert p1_row["dispersion"] == pytest.approx(0.0)  # a single real weekly value
    # season namespace: espn/cbs -- real season-long totals, never mixed
    # into the weekly dispersion (see module docstring for why)
    assert p1_row["season_source_points_espn"] == pytest.approx(220.0)
    assert p1_row["season_source_points_cbs"] == pytest.approx(240.0)
    assert p1_row["n_season_sources"] == 2
    assert p1_row["season_dispersion"] == pytest.approx(10.0)  # pstdev([220.0, 240.0])
    assert p1_row["model_mean"] is not None
    assert p1_row["b2_mean"] is not None
    assert p1_row["b3_mean"] is not None
    assert p1_row["league_slug"] == "test-league"
    assert p1_row["run_label"] == "tuesday"
    assert p1_row["actual_points"] is None

    p0_row = rows.filter(pl.col("player_id") == "p0").row(0, named=True)
    assert p0_row["n_sources"] == 0
    assert p0_row["dispersion"] == pytest.approx(0.0)
    assert p0_row["n_season_sources"] == 0
    assert p0_row["season_dispersion"] == pytest.approx(0.0)


def test_build_prediction_log_is_honestly_empty_with_no_row_universe(monkeypatch) -> None:
    _mock_fetch_all_sources(monkeypatch)
    features = _features()
    schedule = _schedule(week=9)

    rows, fetch_df = prediction_log.build_prediction_log(
        features,
        schedule,
        2025,
        99,  # no real rows for this week
        "tuesday",
        league_slug="test-league",
        scoring_settings={},
        players_dim=_players_dim(),
        train_start=2015,
        min_train_rows=10,
        lightgbm_params=_FAST_PARAMS,
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        b3_historical=_empty_b3_historical(),
        code_version="abc123",
        now=datetime(2025, 11, 1, tzinfo=UTC),
        offline=True,
        settings=None,
    )

    assert rows.is_empty()
    assert fetch_df.is_empty()


# --- write_prediction_log / backfill_actual_points / check_sources -------------------------


class _FakeSettings:
    def __init__(self, data_root):
        self.data_root = data_root


def _sample_rows(season=2025, week=9, run_label="tuesday", player_id="p1") -> pl.DataFrame:
    row = {c: None for c in prediction_log.PREDICTION_LOG_SCHEMA}
    row.update(
        {
            "league_slug": "test-league",
            "season": season,
            "week": week,
            "run_label": run_label,
            "player_id": player_id,
            "position": "RB",
            "n_sources": 0,
            "dispersion": 0.0,
        }
    )
    return pl.DataFrame([row], schema=prediction_log.PREDICTION_LOG_SCHEMA)


def test_write_prediction_log_creates_the_file_and_latest_pointer(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    rows = _sample_rows()
    fetch_rows = pl.DataFrame(schema=prediction_log.SOURCE_FETCH_SCHEMA)

    path = prediction_log.write_prediction_log(
        rows, fetch_rows, 2025, 9, "tuesday", league_slug="test-league", settings=settings
    )

    assert path.exists()
    latest = tmp_path / "outputs" / "test-league" / "prediction_log" / "latest.parquet"
    assert latest.exists()


def test_write_prediction_log_upserts_by_run_label_not_duplicating(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    fetch_rows = pl.DataFrame(schema=prediction_log.SOURCE_FETCH_SCHEMA)

    prediction_log.write_prediction_log(
        _sample_rows(run_label="tuesday"),
        fetch_rows,
        2025,
        9,
        "tuesday",
        league_slug="test-league",
        settings=settings,
    )
    path = prediction_log.write_prediction_log(
        _sample_rows(run_label="sunday"),
        fetch_rows,
        2025,
        9,
        "sunday",
        league_slug="test-league",
        settings=settings,
    )

    written = pl.read_parquet(path)
    assert set(written["run_label"].to_list()) == {"tuesday", "sunday"}
    assert written.height == 2

    # re-running the SAME run_label overwrites, not duplicates
    path = prediction_log.write_prediction_log(
        _sample_rows(run_label="tuesday"),
        fetch_rows,
        2025,
        9,
        "tuesday",
        league_slug="test-league",
        settings=settings,
    )
    written = pl.read_parquet(path)
    assert written.height == 2


def test_backfill_actual_points_fills_from_real_target(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    fetch_rows = pl.DataFrame(schema=prediction_log.SOURCE_FETCH_SCHEMA)
    prediction_log.write_prediction_log(
        _sample_rows(player_id="p1"),
        fetch_rows,
        2025,
        9,
        "tuesday",
        league_slug="test-league",
        settings=settings,
    )
    features = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [9], "target": [17.5]})

    filled = prediction_log.backfill_actual_points(
        features, 2025, 9, league_slug="test-league", settings=settings
    )

    assert filled.filter(pl.col("player_id") == "p1")["actual_points"][0] == pytest.approx(17.5)


def test_backfill_actual_points_raises_a_named_error_when_week_was_never_logged(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    features = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [9], "target": [17.5]})

    with pytest.raises(prediction_log.MissingBackfillError):
        prediction_log.backfill_actual_points(
            features, 2025, 9, league_slug="test-league", settings=settings
        )


def test_check_sources_promotes_frozen_after_three_identical_weeks(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    status_path = tmp_path / "source_refresh_status.yml"
    prediction_log.write_source_refresh_status({"espn": "unknown"}, status_path)

    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "espn",
                "payload_sha256": "same-hash-every-week",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "unknown",
                "n_rows_parsed": 10,
                "fetch_error": None,
            }
            for week in [1, 2, 3]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    log_dir.mkdir(parents=True)
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=status_path
    )

    espn_summary = summary.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_summary["n_distinct_hashes"] == 1
    assert espn_summary["status"] == "frozen"
    assert prediction_log.load_source_refresh_status(status_path)["espn"] == "frozen"


def test_check_sources_confirms_weekly_when_hash_changes(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    status_path = tmp_path / "source_refresh_status.yml"

    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "espn",
                "payload_sha256": f"hash-{week}",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "unknown",
                "n_rows_parsed": 10,
                "fetch_error": None,
            }
            for week in [1, 2, 3]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    log_dir.mkdir(parents=True)
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=status_path
    )

    espn_summary = summary.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_summary["n_distinct_hashes"] == 3
    assert espn_summary["status"] == "weekly_confirmed"


def test_check_sources_never_overrides_an_already_resolved_frozen_status(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    status_path = tmp_path / "source_refresh_status.yml"
    prediction_log.write_source_refresh_status({"footballguys": "frozen"}, status_path)

    # real evidence that LOOKS weekly (hash changed) -- should not matter,
    # a resolved static fact (the hardcoded preseason URL) isn't overridden.
    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "footballguys",
                "payload_sha256": f"hash-{week}",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "frozen",
                "n_rows_parsed": 10,
                "fetch_error": None,
            }
            for week in [1, 2, 3]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    log_dir.mkdir(parents=True)
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    prediction_log.check_sources(league_slug="test-league", settings=settings, path=status_path)

    assert prediction_log.load_source_refresh_status(status_path)["footballguys"] == "frozen"


# --- season-source trend detection (SPEC-ADDENDUM-05.md §D real question) -----------------


def _week_log_row(season: int, week: int, espn_value: float) -> dict:
    row = {c: None for c in prediction_log.PREDICTION_LOG_SCHEMA}
    row.update(
        {
            "league_slug": "test-league",
            "season": season,
            "week": week,
            "run_label": "tuesday",
            "player_id": "p1",
            "position": "RB",
            "season_source_points_espn": espn_value,
            "n_sources": 0,
            "dispersion": 0.0,
            "n_season_sources": 1,
            "season_dispersion": 0.0,
        }
    )
    return row


def _write_week_logs(log_dir, season: int, values: dict[int, float]) -> None:
    for week, value in values.items():
        week_dir = log_dir / f"season={season}"
        week_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            [_week_log_row(season, week, value)], schema=prediction_log.PREDICTION_LOG_SCHEMA
        )
        df.write_parquet(week_dir / f"week={week:02d}.parquet")


def test_check_sources_reports_declining_season_trend(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    # real season total shrinking week over week as games get played --
    # a genuine rest-of-season signal.
    _write_week_logs(log_dir, 2025, {1: 200.0, 2: 180.0, 3: 160.0, 4: 140.0})
    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "espn",
                "payload_sha256": f"hash-{week}",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "unknown",
                "n_rows_parsed": 1,
                "fetch_error": None,
            }
            for week in [1, 2, 3, 4]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=tmp_path / "status.yml"
    )

    espn_row = summary.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_row["season_trend"] == "declining"
    assert espn_row["season_n_weeks"] == 4


def test_check_sources_reports_flat_season_trend(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    # a real static preseason snapshot, never revised
    _write_week_logs(log_dir, 2025, {1: 200.0, 2: 200.0, 3: 200.0, 4: 200.0})
    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "espn",
                "payload_sha256": "same-hash",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "unknown",
                "n_rows_parsed": 1,
                "fetch_error": None,
            }
            for week in [1, 2, 3, 4]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=tmp_path / "status.yml"
    )

    espn_row = summary.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_row["season_trend"] == "flat"


def test_check_sources_reports_insufficient_data_before_three_weeks(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    _write_week_logs(log_dir, 2025, {1: 200.0, 2: 180.0})
    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "espn",
                "payload_sha256": f"hash-{week}",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "unknown",
                "n_rows_parsed": 1,
                "fetch_error": None,
            }
            for week in [1, 2]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=tmp_path / "status.yml"
    )

    espn_row = summary.filter(pl.col("source") == "espn").row(0, named=True)
    assert espn_row["season_trend"] == "insufficient_data"


def test_check_sources_season_trend_is_null_for_weekly_sources(tmp_path) -> None:
    settings = _FakeSettings(tmp_path)
    log_dir = tmp_path / "outputs" / "test-league" / "prediction_log"
    log_dir.mkdir(parents=True)
    fetch_rows = pl.DataFrame(
        [
            {
                "league_slug": "test-league",
                "season": 2025,
                "week": week,
                "run_label": "tuesday",
                "source": "fantasypros",
                "payload_sha256": f"hash-{week}",
                "fetched_at_utc": "2025-11-01T00:00:00Z",
                "refresh_status": "weekly_confirmed",
                "n_rows_parsed": 1,
                "fetch_error": None,
            }
            for week in [1, 2, 3]
        ],
        schema=prediction_log.SOURCE_FETCH_SCHEMA,
    )
    fetch_rows.write_parquet(log_dir / "source_fetches.parquet")

    summary = prediction_log.check_sources(
        league_slug="test-league", settings=settings, path=tmp_path / "status.yml"
    )

    fp_row = summary.filter(pl.col("source") == "fantasypros").row(0, named=True)
    assert fp_row["season_trend"] is None
