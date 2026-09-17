from __future__ import annotations

import polars as pl

from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.tools import chopped_alerts
from ffapp.tools.chopped_alerts import (
    completed_chop_transactions,
    evaluate_recommendations,
    format_chopped_recommendation,
)


def test_completed_chops_ignore_ordinary_and_incomplete_transactions() -> None:
    rows = [
        {
            "transaction_id": "new",
            "type": "chopped",
            "status": "complete",
            "drops": {"p": 1},
            "created": 2,
        },
        {
            "transaction_id": "old",
            "type": "chopped",
            "status": "complete",
            "drops": {"q": 2},
            "created": 1,
        },
        {"transaction_id": "pending", "type": "chopped", "status": "pending", "drops": {"r": 3}},
        {"transaction_id": "waiver", "type": "waiver", "status": "complete", "drops": {"s": 4}},
    ]
    assert [row["transaction_id"] for row in completed_chop_transactions(rows)] == ["old", "new"]


def test_chopped_message_contains_bid_bands_and_survival_gain() -> None:
    board = pl.DataFrame(
        {
            "player_name": ["Star Player"],
            "position": ["RB"],
            "conservative_bid": [40],
            "recommended_bid": [55],
            "aggressive_bid": [70],
            "max_bid": [80],
            "recommended_win_probability": [0.75],
            "survival_probability_after": [0.88],
            "survival_probability_gain": [0.12],
            "drop_player": ["Bench Player"],
        }
    )
    message = format_chopped_recommendation("Chopped League", 4, board)
    assert "Star Player" in message
    assert "$40 / **$55** / $70" in message
    assert "survival 88% (+12.0%)" in message
    assert "drop Bench Player" in message


def test_evaluation_matches_first_award_after_chop() -> None:
    recommendations = pl.DataFrame(
        {
            "chopped_transaction_id": ["chop"],
            "chopped_at_ms": [100],
            "sleeper_id": ["p1"],
            "player_name": ["Player One"],
            "recommended_bid": [31],
            "max_bid": [40],
        }
    )
    outcomes = pl.DataFrame(
        {
            "transaction_id": ["before", "after"],
            "week": [1, 2],
            "created_at_ms": [90, 110],
            "status": ["complete", "complete"],
            "roster_id": [2, 3],
            "sleeper_id": ["p1", "p1"],
            "bid": [5, 28],
        }
    )
    result = evaluate_recommendations(recommendations, outcomes).row(0, named=True)
    assert result["actual_winning_bid"] == 28
    assert result["recommendation_error"] == 3
    assert result["recommended_met_winning_bid"] is True


def test_first_refresh_baselines_existing_chops_without_duplicate_alerts(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        data_root=tmp_path,
        sleeper_username="manager",
        cache=CacheSettings(tmp_path / "raw", True, {}, True),
    )
    league = LeagueConfig(
        slug="chopped",
        display_name="Chopped",
        is_primary=False,
        league_id="123",
        season=2026,
        league_cache={"league_type": 3},
        overrides={},
    )
    existing = [
        {
            "transaction_id": "already-seen",
            "type": "chopped",
            "status": "complete",
            "drops": {"p1": 2},
            "created": 100,
            "_week": 1,
        }
    ]
    monkeypatch.setattr(chopped_alerts, "_load_transactions", lambda *args, **kwargs: existing)

    first = chopped_alerts.refresh_chopped_alerts(settings, league, 2026, 2, [], offline=True)
    second = chopped_alerts.refresh_chopped_alerts(settings, league, 2026, 2, [], offline=True)

    assert first.status == "initialized"
    assert second.status == "skipped"
    assert "No new" in second.detail
