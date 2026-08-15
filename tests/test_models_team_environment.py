from __future__ import annotations

import polars as pl
import pytest

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS
from ffapp.features import team_context
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
            "plays_per_game_ewm_5": [64.0, 65.0, 59.0, 59.5],
            "pass_rate_ewm_5": [0.58, 0.60, 0.53, 0.54],
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
    assert row["plays_per_game_ewm_5"] == pytest.approx(64.0)  # week 1's value, not week 2's 65.0
    assert row["pass_rate_ewm_5"] == pytest.approx(0.58)  # week 1's value, not week 2's 0.60


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


def test_build_team_environment_table_bye_week_return_has_null_trailing_features() -> None:
    """A team coming off a bye has no row at `week - 1` -- the positional
    shift (`.shift(1).over(["team", "season"])`) must still land on the
    real *previous played week*'s value, not silently null just because a
    week number was skipped. Week 3 here is KC's bye (no week-3 row at
    all); week 4 is their next game and must pick up week 2's real trailing
    value, not go null the way a week-arithmetic (`week - 1`) shift would."""
    features = pl.DataFrame(
        {
            "team": ["KC", "KC", "KC"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 4],  # week 3 is a bye -- no row
            "plays": [65, 70, 68],
            "pass_rate": [0.60, 0.65, 0.62],
            "implied_team_total": [24.5, 27.0, 23.0],
            "spread": [-3.0, -2.5, -1.0],
            "proe_ewm_5": [0.02, 0.03, 0.025],
            "neutral_pace_ewm_8": [28.0, 27.5, 27.0],
            "opponent_neutral_pace_ewm_8": [31.5, 31.0, 30.5],
            "plays_per_game_ewm_5": [64.0, 65.0, 66.0],
            "pass_rate_ewm_5": [0.58, 0.60, 0.61],
        }
    )

    result = team_environment.build_team_environment_table(features)

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 4)).row(0, named=True)
    # week 2's real trailing values (the previous *played* week), not null.
    assert row["proe_ewm_5"] == pytest.approx(0.03)
    assert row["neutral_pace_ewm_8"] == pytest.approx(27.5)
    assert row["plays_per_game_ewm_5"] == pytest.approx(65.0)
    assert row["pass_rate_ewm_5"] == pytest.approx(0.60)


def test_current_feature_columns_stays_a_subset_of_team_contexts_current_week_columns() -> None:
    """Regression test: this classification cost two fix rounds during
    implementation (task review caught opponent_neutral_pace_ewm_8
    misclassified as trailing, twice, before landing correctly). If
    team_context.CURRENT_WEEK_COLUMNS is ever reclassified, this module's
    own CURRENT_FEATURE_COLUMNS must not silently drift out of sync."""
    assert set(team_environment.CURRENT_FEATURE_COLUMNS) <= set(team_context.CURRENT_WEEK_COLUMNS)
    assert set(team_environment.TRAILING_FEATURE_COLUMNS).isdisjoint(
        team_context.CURRENT_WEEK_COLUMNS
    )


# --- monotone_constraints -----------------------------------------------------------
# Closes a real coverage gap flagged by the final Stage 1 review: `points.py`'s own
# monotone_constraints has a direct value-level test (test_models_points.py); this
# module's didn't, and its own sign was wrong once already (see module docstring).


def test_monotone_constraints_marks_pace_features_as_decreasing_for_team_plays() -> None:
    constraints = team_environment.monotone_constraints("team_plays")
    by_column = dict(zip(team_environment.FEATURE_COLUMNS, constraints, strict=True))

    assert by_column["neutral_pace_ewm_8"] == -1
    assert by_column["opponent_neutral_pace_ewm_8"] == -1


def test_monotone_constraints_marks_own_trailing_features_as_increasing() -> None:
    team_plays_constraints = dict(
        zip(
            team_environment.FEATURE_COLUMNS,
            team_environment.monotone_constraints("team_plays"),
            strict=True,
        )
    )
    pass_rate_constraints = dict(
        zip(
            team_environment.FEATURE_COLUMNS,
            team_environment.monotone_constraints("pass_rate"),
            strict=True,
        )
    )

    assert team_plays_constraints["plays_per_game_ewm_5"] == 1
    assert pass_rate_constraints["pass_rate_ewm_5"] == 1
    assert pass_rate_constraints["proe_ewm_5"] == 1


def test_monotone_constraints_leaves_current_week_features_unconstrained() -> None:
    constraints = team_environment.monotone_constraints("team_plays")
    by_column = dict(zip(team_environment.FEATURE_COLUMNS, constraints, strict=True))

    assert by_column["implied_team_total"] == 0
    assert by_column["spread"] == 0


def test_monotone_constraints_length_matches_feature_columns() -> None:
    for target_column in team_environment.TARGET_COLUMNS:
        assert len(team_environment.monotone_constraints(target_column)) == len(
            team_environment.FEATURE_COLUMNS
        )


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
                "plays_per_game_ewm_5": 60.0 + i * 0.5,
                "pass_rate_ewm_5": 0.5 + (i % 5) * 0.015,
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
