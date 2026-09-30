from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.evaluation.weekly_review import build_weekly_review, score_summary

NOW = datetime(2026, 9, 22, 12, tzinfo=UTC)


def snapshot(player="a", **changes):
    return (
        dict(
            season=2026,
            week=2,
            player_id=player,
            team="BUF",
            position="WR",
            as_of_utc="2026-09-19T12:00:00Z",
            projection_source="consensus_b3",
            live_mean=10.0,
            actual_points=8.0,
            b2_mean=9.0,
            live_q10=2.0,
            live_q90=20.0,
        )
        | changes
    )


def review(rows, *, injuries=None, stats=None, usage=None, overrides=None, now=NOW):
    schedule = pl.DataFrame(
        [
            dict(
                season=2026,
                week=2,
                home_team="BUF",
                away_team="MIA",
                kickoff_utc="2026-09-20T17:00:00Z",
            )
        ]
    )
    return build_weekly_review(
        pl.DataFrame(rows),
        schedule,
        pl.DataFrame(stats or []),
        pl.DataFrame(usage or []),
        pl.DataFrame(injuries or []),
        overrides,
        now=now,
    )


def test_latest_pregame_snapshot_wins_and_postgame_only_is_excluded():
    result = review(
        [
            snapshot(live_mean=6.0),
            snapshot(live_mean=9.0, as_of_utc="2026-09-20T16:00:00Z"),
            snapshot(live_mean=100.0, as_of_utc="2026-09-21T12:00:00Z"),
            snapshot("b", as_of_utc="2026-09-21T12:00:00Z"),
        ]
    )
    assert result.filter(pl.col("player_id") == "a")["projected"][0] == 9
    assert result.filter(pl.col("player_id") == "b")["status"][0] == "No verified pregame snapshot"
    assert score_summary(result, ["week"])["players"][0] == 1


def test_injury_exclusions_keep_zero_noninjury_misses_and_allow_reversal():
    rows = [snapshot(), snapshot("b", actual_points=0.0), snapshot("c", actual_points=0.0)]
    injuries = [
        dict(season=2026, week=2, player_id="a", report_status="Out"),
        dict(season=2026, week=2, player_id="c", report_status="Questionable"),
    ]
    result = review(
        rows, injuries=injuries, overrides={"2026:2:b": {"exclude": True, "note": "Left Q1"}}
    )
    assert result["status"].to_list() == ["Injury excluded", "Injury excluded", "Scored"]
    assert score_summary(result, ["week"])["mae"][0] == 10
    restored = review(rows, injuries=injuries, overrides={"2026:2:b": {"exclude": False}})
    assert restored.filter(pl.col("player_id") == "b")["status"][0] == "Scored"


def test_missing_and_all_zero_actuals_are_not_a_good_score():
    assert review([snapshot(actual_points=0.0)])["status"][0] == "Awaiting real actuals"
    assert review([snapshot(actual_points=None)])["status"][0] == "Awaiting actuals"
    assert (
        review([snapshot()], now=datetime(2026, 9, 20, 18, tzinfo=UTC))["status"][0]
        == "Week not ready"
    )


def test_legacy_source_fallback_does_not_substitute_a_different_model():
    result = review(
        [
            snapshot(live_mean=None, b3_mean=12.0),
            snapshot("b", live_mean=None, b3_mean=12.0, projection_source="anchored"),
        ]
    )
    assert result["projected"][0] == 12
    assert result["status"][1] == "Missing saved live projection"


def test_metrics_use_same_baseline_sample_and_missing_intervals_are_not_misses():
    result = review(
        [
            snapshot(),
            snapshot(
                "b", live_mean=20.0, actual_points=10.0, b2_mean=None, live_q10=None, live_q90=None
            ),
        ]
    )
    metrics = score_summary(result, ["week"]).row(0, named=True)
    assert metrics["mae"] == 6
    assert metrics["bias"] == 6
    assert metrics["rmse"] == pytest.approx(52**0.5)
    assert metrics["coverage"] == 1
    assert metrics["interval_players"] == 1
    assert metrics["baseline_players"] == 1
    assert metrics["paired_model_mae"] == 2
    assert metrics["baseline_mae"] == 1


def test_usage_clues_only_use_prior_games_and_low_snaps_do_not_prove_injury():
    stats = [
        dict(season=2026, week=w, player_id="a", targets=t)
        for w, t in [(0, 10), (1, 10), (2, 1), (3, 100)]
    ]
    usage = [
        dict(season=2026, week=w, player_id="a", offense_snap_pct=s)
        for w, s in [(0, 0.9), (1, 0.9), (2, 0.1)]
    ]
    result = review([snapshot()], stats=stats, usage=usage).row(0, named=True)
    assert result["needs_review"] is True
    assert "avg 10.0" in result["evidence"]
    assert "Usage changed" in result["signal"]
    assert result["status"] == "Scored"
