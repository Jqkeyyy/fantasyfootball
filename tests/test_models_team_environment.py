from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import team_environment


def _team_context_features() -> pl.DataFrame:
    """Two teams, two consecutive weeks -- week 2's row is what exercises
    the lag shift (week 2's feature values must come from week 1's real
    numbers, not week 2's own)."""
    return pl.DataFrame(
        {
            "team": ["KC", "KC", "BAL", "BAL"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 1, 2],
            "plays": [65, 70, 60, 58],
            "pass_rate": [0.60, 0.65, 0.55, 0.50],
            "implied_team_total": [24.5, 27.0, 20.0, 21.5],
            "spread": [-3.0, -2.5, 3.0, 2.5],
            "proe_ewm_5": [0.02, 0.03, -0.01, -0.02],
            "neutral_pace_ewm_8": [28.0, 27.5, 31.5, 31.0],
            "opponent_neutral_pace_ewm_8": [31.5, 31.0, 28.0, 27.5],
        }
    )


def test_build_team_environment_table_reshapes_to_harness_contract() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["player_id"] == "KC"
    assert row["position"] == "TEAM_ENV"
    assert row["availability_flag"] is True


def test_build_team_environment_table_carries_real_targets_unshifted() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["team_plays"] == 70  # week 2's own real outcome, not week 1's
    assert row["pass_rate"] == pytest.approx(0.65)


def test_build_team_environment_table_lag_shifts_trailing_features_by_one_week() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["proe_ewm_5"] == pytest.approx(0.02)  # week 1's value, not week 2's 0.03
    assert row["neutral_pace_ewm_8"] == pytest.approx(28.0)  # week 1's value
    assert row["opponent_neutral_pace_ewm_8"] == pytest.approx(31.5)  # week 1's value


def test_build_team_environment_table_first_week_has_null_trailing_features() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 1)).row(0, named=True)
    assert row["proe_ewm_5"] is None  # no week-0 data to shift from
    assert row["team_plays"] == 65  # the real target is still present even though features are null


def test_build_team_environment_table_current_week_features_are_not_shifted() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["implied_team_total"] == pytest.approx(27.0)  # week 2's own real Vegas line
    assert row["spread"] == pytest.approx(-2.5)
