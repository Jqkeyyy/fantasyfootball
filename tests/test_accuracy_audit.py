from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.evaluation.accuracy_audit import adjustment_effects, audit_metrics, matched_audit
from ffapp.models.live_adjustments import apply_adjustments


def sample():
    rows = []
    for player, actual in [("a", 12.0), ("b", 4.0)]:
        for stamp, live, espn in [
            ("2026-09-10T10:00:00+00:00", 10.0, 8.0),
            ("2026-09-13T15:00:00+00:00", 11.0, None),
            ("2026-09-14T10:00:00+00:00", actual, actual),
        ]:
            rows.append(
                dict(
                    season=2026,
                    week=1,
                    player_id=player,
                    position="WR",
                    team="BUF",
                    as_of_utc=stamp,
                    live_mean=live,
                    espn_mean=espn,
                    projection_source="espn_weekly",
                    actual_points=actual,
                )
            )
    schedule = pl.DataFrame(
        dict(
            season=[2026],
            week=[1],
            home_team=["BUF"],
            away_team=["NYJ"],
            kickoff_utc=["2026-09-13T17:00:00+00:00"],
        )
    )
    return pl.DataFrame(rows), schedule


def run_audit(history=None, overrides=None, **kwargs):
    data, schedule = sample()
    return matched_audit(
        data if history is None else history,
        schedule,
        pl.DataFrame(),
        pl.DataFrame(),
        overrides or {},
        {"Live": "live_mean", "ESPN": "espn_mean"},
        now=datetime(2026, 9, 16, tzinfo=UTC),
        **kwargs,
    )


def test_same_snapshot_rejects_postgame_and_does_not_mix_latest_live_with_older_source():
    rows, _ = run_audit()
    assert rows.height == 2
    assert rows["live_mean"].to_list() == [10.0, 10.0]
    assert rows["snapshot"].unique().to_list() == ["2026-09-10T10:00:00+00:00"]
    metrics = audit_metrics(rows, {"Live": "live_mean", "ESPN": "espn_mean"})
    assert metrics["Players / weeks"].to_list() == [2, 2]
    assert metrics["Average miss"].to_list() == [4.0, 4.0]


def test_missing_nonfinite_and_unverified_times_are_excluded():
    data, _ = sample()
    data = data.with_columns(
        pl.when(pl.col("player_id") == "b")
        .then(float("nan"))
        .otherwise(pl.col("espn_mean"))
        .alias("espn_mean")
    )
    rows, _ = run_audit(data)
    assert rows["player_id"].to_list() == ["a"]
    rows, review = run_audit(data.with_columns(pl.lit("invalid").alias("as_of_utc")))
    assert rows.is_empty()
    assert review["status"][0] == "No verified pregame snapshot"


def test_backfilled_actuals_allowed_but_confirmed_injuries_and_zero_placeholders_excluded():
    data, _ = sample()
    data = data.with_columns(
        pl.when(pl.col("as_of_utc").str.starts_with("2026-09-14"))
        .then(pl.col("actual_points"))
        .otherwise(None)
        .alias("actual_points")
    )
    overrides = {"2026:1:a": {"exclude": True, "note": "confirmed"}}
    rows, _ = run_audit(data, overrides)
    assert rows["player_id"].to_list() == ["b"]
    rows, _ = run_audit(data, overrides, include_injured=True)
    assert rows.height == 2
    rows, _ = run_audit(data.with_columns(pl.lit(0.0).alias("actual_points")))
    assert rows.is_empty()


def test_stage_recording_and_paired_error_attribution_handles_zero_injury_multiplier():
    projections = pl.DataFrame(
        dict(
            player_id=["a", "b"],
            season=[2026] * 2,
            week=[1] * 2,
            mean=[10.0, 5.0],
            p_active=[1.0, 1.0],
        )
    )
    adjustments = pl.DataFrame(
        dict(
            player_id=["a", "a"],
            season=[2026] * 2,
            week=[1] * 2,
            adjustment_type=["role", "injury"],
            multiplier=[1.1, 0.0],
        )
    )
    result, _ = apply_adjustments(projections, adjustments)
    a = result.filter(pl.col("player_id") == "a").row(0, named=True)
    assert a["audit_base_mean"] == 10
    assert a["audit_role_mean"] == 11
    assert a["audit_injury_mean"] == 0
    b = result.filter(pl.col("player_id") == "b").row(0, named=True)
    assert b["audit_role_mean"] == 5
    effects = adjustment_effects(
        result.with_columns(pl.col("mean").alias("live_mean"), pl.lit(0.0).alias("actual"))
    )
    assert effects["Error reduction (positive helps)"].to_list() == pytest.approx([-1, 11])
    unchanged, _ = apply_adjustments(projections, pl.DataFrame())
    assert unchanged["audit_base_mean"].to_list() == [10, 5]


def test_legacy_logs_do_not_invent_adjustment_stages():
    from ffapp.evaluation.accuracy_audit import STAGES

    data, schedule = sample()
    rows, _ = matched_audit(data, schedule, pl.DataFrame(), pl.DataFrame(), {}, STAGES)
    assert rows.is_empty()


def test_projection_upsert_preserves_legacy_rows_with_null_audit_fields(tmp_path):
    from ffapp.models.predict import write_projections

    path = tmp_path / "projections.parquet"
    old = pl.DataFrame(dict(season=[2026], week=[1], player_id=["a"], mean=[10.0]))
    old.write_parquet(path)
    new = old.with_columns(pl.lit(2).cast(pl.Int64).alias("week"))
    new, _ = apply_adjustments(new, pl.DataFrame())
    combined = write_projections(new, path)
    assert combined.height == 2
    assert combined.filter(pl.col("week") == 1)["audit_base_mean"][0] is None
    assert combined.filter(pl.col("week") == 2)["audit_base_mean"][0] == 10.0
