from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import polars as pl

from ffapp.evaluation.learning import recurring_patterns, replay_bias_correction
from ffapp.evaluation.weekly_review import build_weekly_review
from ffapp.tools import discord_notifications as discord
from ffapp.tools.gameday import apply_availability, due_windows, save_snapshot


def schedule(kickoff):
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "season_type": "REG",
                "kickoff_utc": kickoff.isoformat(),
                "home_team": "KC",
                "away_team": "BAL",
            }
        ]
    )


def test_checkpoints_cover_international_monday_and_never_after_kickoff():
    for kickoff in [
        datetime(2026, 9, 27, 13, 30, tzinfo=UTC),
        datetime(2026, 9, 29, 0, 15, tzinfo=UTC),
    ]:
        games = schedule(kickoff)
        assert due_windows(games, kickoff - timedelta(minutes=91)) == []
        assert due_windows(games, kickoff - timedelta(minutes=89))[0]["checkpoint"] == 90
        assert due_windows(games, kickoff - timedelta(minutes=29))[0]["checkpoint"] == 30
        assert due_windows(games, kickoff - timedelta(minutes=4)) == []
        assert due_windows(games, kickoff + timedelta(minutes=1)) == []


def test_only_confirmed_out_upcoming_players_are_zeroed():
    projections = pl.DataFrame(
        {"player_id": ["a", "b", "c"], "mean": [15.0, 12.0, 10.0], "p_active": [0.8, 0.7, 0.9]}
    )
    ids = pl.DataFrame({"player_id": ["a", "b", "c"], "sleeper_id": ["1", "2", "3"]})
    players = {
        "1": {"injury_status": "Out", "team": "KC"},
        "2": {"injury_status": "Questionable", "team": "KC"},
        "3": {"injury_status": "Out", "team": "SEA"},
    }
    adjusted, statuses = apply_availability(projections, ids, players, {"KC"})
    assert adjusted["mean"].to_list() == [0.0, 12.0, 10.0]
    assert statuses == {"a": "Out"}


def test_snapshots_exclude_started_games_and_are_not_overwritten(tmp_path):
    now = datetime(2026, 9, 27, 17, tzinfo=UTC)
    projections = pl.DataFrame(
        {
            "season": [2026],
            "week": [3],
            "player_id": ["a"],
            "mean": [12.0],
            **{f"q{q}": [float(q)] for q in [10, 25, 50, 75, 90]},
        }
    )
    features = pl.DataFrame(
        {"season": [2026], "week": [3], "player_id": ["a"], "team": ["KC"], "position": ["WR"]}
    )
    games = schedule(now + timedelta(minutes=30))
    save_snapshot(tmp_path, projections, features, games, {}, now)
    save_snapshot(tmp_path, projections, features, games, {}, now + timedelta(minutes=5))
    paths = list((tmp_path / "prediction_log/kickoff").glob("*.parquet"))
    assert len(paths) == 2
    save_snapshot(tmp_path, projections, features, games, {}, now + timedelta(minutes=31))
    assert len(list((tmp_path / "prediction_log/kickoff").glob("*.parquet"))) == 2


def test_kickoff_snapshot_uses_weekly_backfilled_actual_without_changing_forecast():
    old = {
        "season": 2026,
        "week": 3,
        "player_id": "a",
        "team": "KC",
        "position": "WR",
        "projection_source": "direct",
        "as_of_utc": "2026-09-25T12:00:00Z",
        "live_mean": 10.0,
        "actual_points": 8.0,
    }
    latest = old | {"as_of_utc": "2026-09-27T16:30:00Z", "actual_points": None, "live_mean": 12.0}
    result = build_weekly_review(
        pl.DataFrame([old, latest]),
        schedule(datetime(2026, 9, 27, 17, tzinfo=UTC)),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    assert result["actual"][0] == 8
    assert result["projected"][0] == 12


def sample_review():
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": week,
                "source": "model",
                "position": "WR",
                "status": "Scored",
                "projected": 12.0,
                "actual": actual,
                "error": 12.0 - actual,
                "absolute_error": abs(12.0 - actual),
                "in_range": True,
                "baseline": 10.0,
            }
            for week, actual in [(1, 9.0), (2, 9.0), (3, 20.0), (4, 9.0)]
            for _ in range(12)
        ]
    )


def test_bias_trial_uses_only_earlier_weeks_and_can_report_a_worse_result():
    trials = replay_bias_correction(sample_review())
    first = trials.filter(pl.col("Week tested") == 3).row(0, named=True)
    assert first["Correction"] == 3.0
    assert first["Trial miss"] == 11.0
    assert first["Improvement"] == -3.0
    assert trials["Week tested"].min() == 3
    changed_future = sample_review().filter(pl.col("week") <= 3)
    assert replay_bias_correction(changed_future).row(0, named=True) == first


def test_recurring_patterns_require_three_supported_weeks_and_ignore_injuries():
    data = sample_review()
    assert recurring_patterns(data.filter(pl.col("week") <= 2)).is_empty()
    assert "Consistently too high" in recurring_patterns(data)["Pattern"].to_list()
    excluded = data.with_columns(
        pl.when(pl.col("week") == 4)
        .then(pl.lit("Injury excluded"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    assert recurring_patterns(excluded).is_empty()


def test_discord_deduplicates_success_but_retries_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(discord, "load_settings", lambda: SimpleNamespace(data_root=tmp_path))
    calls = []

    def send(content):
        calls.append(content)
        return discord.NotificationResult("failed" if len(calls) == 1 else "sent", "test")

    monkeypatch.setattr(discord, "send_discord_message", send)
    assert discord.send_action_notification("hello", key="league").status == "failed"
    assert discord.send_action_notification("hello", key="league").status == "sent"
    assert discord.send_action_notification("hello", key="league").status == "skipped"
    assert len(calls) == 2


def test_webhook_rejects_non_discord_destination(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(discord, "webhook_path", lambda: tmp_path / "discord.json")
    with pytest.raises(ValueError):
        discord.save_webhook("https://example.com/api/webhooks/123/secret")
    assert not (tmp_path / "discord.json").exists()
