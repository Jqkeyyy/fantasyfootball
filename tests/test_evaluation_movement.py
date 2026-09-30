from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ffapp.config import CacheSettings, Settings
from ffapp.evaluation.movement import detect_projection_movements
from ffapp.tools.weekly_alerts import enrich_alerts_with_movements


def _history() -> pl.DataFrame:
    rows = []
    for run, stamp, live, espn, fantasypros in (
        ("tuesday", "2026-09-22T12:00:00+00:00", 10.0, 10.0, 10.0),
        ("thursday", "2026-09-24T12:00:00+00:00", 13.0, 13.0, 10.5),
        ("sunday", "2026-09-27T12:00:00+00:00", 14.0, 14.0, 13.0),
    ):
        rows.append(
            {
                "season": 2026,
                "week": 3,
                "player_id": "p1",
                "position": "WR",
                "team": "AAA",
                "run_label": run,
                "as_of_utc": stamp,
                "live_mean": live,
                "espn_mean": espn,
                "b3_mean": fantasypros,
                "b2_mean": 9.0,
                "model_mean": 11.0,
                "p_active": 1.0,
                "is_my_roster": True,
                "was_starting": True,
            }
        )
    return pl.DataFrame(rows)


def test_movement_identifies_first_source_and_later_confirmation() -> None:
    result = detect_projection_movements(_history())

    row = result.row(0, named=True)
    assert row["signal"] == "confirmed"
    assert row["first_mover"] == "ESPN"
    assert row["sources_agreeing"] == 2
    assert row["live_delta"] == 4.0
    assert "2 sources agree" in row["reason"]


def test_movement_ignores_small_refresh_noise() -> None:
    history = _history().with_columns(
        pl.when(pl.col("run_label") == "tuesday").then(10.0).otherwise(10.4).alias("live_mean"),
        pl.when(pl.col("run_label") == "tuesday").then(10.0).otherwise(10.4).alias("espn_mean"),
        pl.lit(10.0).alias("b3_mean"),
    )

    assert detect_projection_movements(history).is_empty()


def test_movement_never_compares_different_weeks() -> None:
    next_week = _history().with_columns(
        pl.lit(4, dtype=pl.Int64).alias("week"),
        pl.lit("tuesday").alias("run_label"),
        pl.lit("2026-09-29T12:00:00+00:00").alias("as_of_utc"),
        pl.lit(50.0).alias("live_mean"),
    ).head(1)

    result = detect_projection_movements(pl.concat([_history(), next_week]))

    assert result.is_empty()


def test_availability_change_is_a_high_confidence_signal() -> None:
    history = _history().filter(pl.col("run_label") != "thursday").with_columns(
        pl.when(pl.col("run_label") == "sunday").then(0.0).otherwise(1.0).alias("p_active")
    )

    row = detect_projection_movements(history).row(0, named=True)

    assert row["signal"] == "availability"
    assert row["availability_delta"] == -1.0
    assert row["confidence"] == 0.92


def test_movement_enriches_existing_starter_alert(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )
    alerts_dir = tmp_path / "outputs" / "league" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "latest.json").write_text(
        json.dumps(
            {
                "alerts": [
                    {
                        "kind": "starter_projection_changed",
                        "player_id": "p1",
                        "confidence": 0.7,
                        "magnitude": 4.0,
                    }
                ]
            }
        )
    )
    pl.DataFrame(
        {"player_id": ["p1"], "player_name": ["Player One"]}
    ).write_parquet(alerts_dir / "latest_snapshot.parquet")
    movement = detect_projection_movements(_history())

    changed = enrich_alerts_with_movements(settings, "league", movement)

    assert changed == 1
    alert = json.loads((alerts_dir / "latest.json").read_text())["alerts"][0]
    assert alert["confidence"] > 0.7
    assert "sources agree" in alert["why"]


def test_movement_adds_consequential_starter_alert(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )
    alerts_dir = tmp_path / "outputs" / "league" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "latest.json").write_text(json.dumps({"alerts": []}))
    pl.DataFrame(
        {"player_id": ["p1"], "player_name": ["Player One"]}
    ).write_parquet(alerts_dir / "latest_snapshot.parquet")

    changed = enrich_alerts_with_movements(
        settings, "league", detect_projection_movements(_history())
    )

    assert changed == 1
    alert = json.loads((alerts_dir / "latest.json").read_text())["alerts"][0]
    assert alert["player_name"] == "Player One"
    assert alert["message"].endswith("from tuesday to sunday.")
