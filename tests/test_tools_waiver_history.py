from __future__ import annotations

import polars as pl

from ffapp.tools.waiver_history import (
    build_manager_bid_profiles,
    extract_waiver_outcomes,
    profile_multipliers,
)


def test_extract_waiver_outcomes_keeps_real_awards_only() -> None:
    transactions = [
        {
            "transaction_id": "won",
            "type": "waiver",
            "status": "complete",
            "created": 100,
            "adds": {"p1": 2},
            "settings": {"waiver_bid": 37},
            "_week": 3,
        },
        {
            "transaction_id": "free",
            "type": "free_agent",
            "status": "complete",
            "adds": {"p2": 1},
            "_week": 3,
        },
    ]
    result = extract_waiver_outcomes(transactions)
    assert result.row(0, named=True) == {
        "transaction_id": "won",
        "week": 3,
        "created_at_ms": 100,
        "status": "complete",
        "roster_id": 2,
        "sleeper_id": "p1",
        "bid": 37,
    }


def test_profiles_learn_aggression_but_shrink_small_samples() -> None:
    outcomes = pl.DataFrame(
        {
            "transaction_id": ["a", "b", "c", "d"],
            "week": [1, 2, 1, 2],
            "created_at_ms": [1, 2, 3, 4],
            "status": ["complete"] * 4,
            "roster_id": [1, 1, 2, 2],
            "sleeper_id": ["p1", "p2", "p3", "p4"],
            "bid": [40, 50, 5, 10],
        }
    )
    profiles = build_manager_bid_profiles(outcomes, [1, 2, 3])
    multipliers = profile_multipliers(profiles)
    assert multipliers[1] > 1.0
    assert multipliers[2] < 1.0
    assert multipliers[3] == 1.0
