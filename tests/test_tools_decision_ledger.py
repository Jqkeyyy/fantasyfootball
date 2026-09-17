from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.tools.decision_ledger import (
    append_recommendations,
    decision_summary,
    recommendation_rows,
    record_choice,
    settle_outcomes,
)


def _recommendations() -> pl.DataFrame:
    lineup = pl.DataFrame(
        {
            "player_id": ["add"],
            "player_name": ["Add Me"],
            "projected_points": [14.0],
            "floor": [8.0],
            "ceiling": [20.0],
        }
    )
    rankings = pl.DataFrame(
        {
            "player_id": ["drop"],
            "player_name": ["Bench Me"],
            "proj_mean": [10.0],
        }
    )
    return recommendation_rows(
        "league",
        2026,
        3,
        lineup,
        rankings,
        {"drop"},
        pl.DataFrame(),
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )


def test_recommendations_are_idempotent_and_choices_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    recommendations = _recommendations()

    first = append_recommendations(path, recommendations)
    chosen = record_choice(path, first["decision_id"].item(), accepted=True)
    second = append_recommendations(path, recommendations)

    assert first.height == 1
    assert chosen["status"].item() == "accepted"
    assert second.height == 1
    assert second["status"].item() == "accepted"


def test_settle_outcomes_measures_realized_delta_and_regret(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    saved = append_recommendations(path, _recommendations())
    record_choice(path, saved["decision_id"].item(), accepted=True)

    settled = settle_outcomes(
        path,
        pl.DataFrame(
            {
                "season": [2026, 2026],
                "week": [3, 3],
                "player_id": ["add", "drop"],
                "actual_points": [7.0, 12.0],
            }
        ),
        now=datetime(2026, 9, 22, tzinfo=UTC),
    )

    assert settled["realized_delta"].item() == -5.0
    assert settled["decision_regret"].item() == 5.0
    assert settled["settled_at_utc"].item() is not None
    summary = decision_summary(settled)
    assert summary["follow_rate"].item() == 1.0
    assert summary["mean_regret"].item() == 5.0
