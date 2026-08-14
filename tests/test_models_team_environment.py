from __future__ import annotations

import polars as pl
import pytest

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS
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
    # week 2's own real opponent pace value, not week 1's 31.5 -- already
    # internally lagged upstream (see add_opponent_pace), so it must not be
    # shifted a second time here.
    assert row["opponent_neutral_pace_ewm_8"] == pytest.approx(31.0)


# --- add_team_environment_baselines (task 3) ----------------------------------------


def _reshaped_table() -> pl.DataFrame:
    base = team_environment.build_team_environment_table(_team_context_features())
    return base


def test_add_team_environment_baselines_adds_all_four_columns() -> None:
    result = team_environment.add_team_environment_baselines(_reshaped_table())

    assert "team_plays_league_mean" in result.columns
    assert "team_plays_b2_ewm_4" in result.columns
    assert "pass_rate_league_mean" in result.columns
    assert "pass_rate_b2_ewm_4" in result.columns


def test_add_team_environment_baselines_b2_never_leaks_the_target_week() -> None:
    result = team_environment.add_team_environment_baselines(_reshaped_table())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    # week 2's b2 baseline must be built only from week 1's real outcome (65),
    # never week 2's own (70) -- with a single prior week, ewm_4 of one point
    # equals that point exactly.
    assert row["team_plays_b2_ewm_4"] == pytest.approx(65.0)


# --- fit_team_environment_model / predict_team_environment / TeamEnvironmentPredictor (task 4) ---


def _training_rows() -> pl.DataFrame:
    """20 rows, enough real variation in the feature columns for LightGBM
    to fit without every leaf collapsing to a single value."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "team": "KC",
                "season": 2024,
                "week": (i % 17) + 1,
                "team_plays": 60.0 + i,
                "pass_rate": 0.5 + (i % 5) * 0.02,
                "implied_team_total": 20.0 + i * 0.3,
                "spread": -3.0 + i * 0.1,
                "proe_ewm_5": 0.01 * i,
                "neutral_pace_ewm_8": 28.0 - i * 0.1,
                "opponent_neutral_pace_ewm_8": 29.0 + i * 0.1,
            }
        )
    return pl.DataFrame(rows)


def test_fit_and_predict_team_plays_model() -> None:
    train_rows = _training_rows()

    model = team_environment.fit_team_environment_model(
        train_rows, target_column="team_plays", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )
    predictions = team_environment.predict_team_environment(model, train_rows)

    assert predictions.len() == train_rows.height
    assert predictions.null_count() == 0


def test_fit_and_predict_pass_rate_model() -> None:
    train_rows = _training_rows()

    model = team_environment.fit_team_environment_model(
        train_rows, target_column="pass_rate", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )
    predictions = team_environment.predict_team_environment(model, train_rows)

    assert predictions.len() == train_rows.height
    assert predictions.null_count() == 0


def test_team_environment_predictor_satisfies_the_harness_protocol() -> None:
    train_rows = _training_rows()
    predictor = team_environment.TeamEnvironmentPredictor(
        name="team_env_plays", target_column="team_plays", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )

    fitted = predictor.fit(train_rows)
    predictions = predictor.predict(fitted, train_rows)

    assert predictor.name == "team_env_plays"
    assert predictions.len() == train_rows.height


# --- derive_attempts (task 5) ---


def test_derive_attempts_sums_to_team_plays_exactly() -> None:
    team_plays = pl.Series([70.0, 60.0])
    pass_rate = pl.Series([0.6, 0.55])

    pass_attempts, rush_attempts = team_environment.derive_attempts(team_plays, pass_rate)

    assert pass_attempts.to_list() == pytest.approx([42.0, 33.0])
    assert rush_attempts.to_list() == pytest.approx([28.0, 27.0])
    assert (pass_attempts + rush_attempts).to_list() == pytest.approx(team_plays.to_list())
