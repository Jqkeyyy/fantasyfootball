from __future__ import annotations

import polars as pl
import pytest

from ffapp.scoring.targets import DuplicateStatRowsError, apply_league_scoring_target


def test_apply_league_scoring_target_replaces_a_different_leagues_target() -> None:
    features = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "season": [2026, 2026],
            "week": [1, 1],
            "target": [999.0, 999.0],
            "feature": [1.0, 2.0],
        }
    )
    stats = pl.DataFrame(
        {"player_id": ["p1"], "season": [2026], "week": [1], "receptions": [6]}
    )

    half_ppr = apply_league_scoring_target(features, stats, {"rec": 0.5})
    full_ppr = apply_league_scoring_target(features, stats, {"rec": 1.0})

    assert half_ppr["target"].to_list() == [3.0, 0.0]
    assert full_ppr["target"].to_list() == [6.0, 0.0]
    assert half_ppr["feature"].to_list() == [1.0, 2.0]


def test_apply_league_scoring_target_rejects_duplicate_stat_rows() -> None:
    features = pl.DataFrame(
        {"player_id": ["p1"], "season": [2026], "week": [1], "target": [0.0]}
    )
    stats = pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2026, 2026],
            "week": [1, 1],
            "receptions": [1, 2],
        }
    )

    with pytest.raises(DuplicateStatRowsError):
        apply_league_scoring_target(features, stats, {"rec": 1.0})
