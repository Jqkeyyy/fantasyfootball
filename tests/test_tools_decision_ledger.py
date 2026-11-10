from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.league_format import LeagueFormat
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
            "player_id": ["add", "drop"],
            "player_name": ["Add Me", "Bench Me"],
            "position": ["RB", "RB"],
            "proj_mean": [14.0, 10.0],
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
        fmt=_FORMAT,
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )


_FORMAT = LeagueFormat(
    n_teams=2,
    starters={"RB": 1},
    flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
    flex_eligible={},
    bench=3,
    ir=0,
    playoff_week_start=15,
    waiver_budget=100,
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


def test_empty_lineup_spot_settles_against_zero(tmp_path: Path) -> None:
    for accepted, points, regret in [(True, 8.0, 0.0), (False, 8.0, 8.0), (True, -2.0, 2.0)]:
        path = tmp_path / "empty.parquet"
        rows = _recommendations().with_columns(
            pl.lit(None, dtype=pl.String).alias("alternative_player_id"),
            pl.lit(None, dtype=pl.String).alias("alternative_player_name"),
        )
        rows.write_parquet(path)
        record_choice(path, rows["decision_id"].item(), accepted=accepted)
        settled = settle_outcomes(
            path,
            pl.DataFrame(
                {
                    "season": [2026],
                    "week": [3],
                    "player_id": ["add"],
                    "actual_points": [points],
                }
            ),
        )
        assert settled["alternative_actual_points"].item() == 0.0
        assert settled["realized_delta"].item() == points
        assert settled["decision_regret"].item() == regret
        assert decision_summary(settled)["settled"].item() == 1


def test_missing_actuals_and_no_drop_waivers_remain_unsettled(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    for kind, alternative, actual_id in [
        ("lineup", None, "other"),
        ("lineup", "drop", "add"),
        ("waiver", None, "add"),
    ]:
        _recommendations().with_columns(
            pl.lit(kind).alias("decision_type"),
            pl.lit(alternative, dtype=pl.String).alias("alternative_player_id"),
        ).write_parquet(path)
        settled = settle_outcomes(
            path,
            pl.DataFrame(
                {
                    "season": [2026],
                    "week": [3],
                    "player_id": [actual_id],
                    "actual_points": [8.0],
                }
            ),
        )
        assert settled["settled_at_utc"].item() is None
