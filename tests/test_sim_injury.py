"""Task 2.3's own literal acceptance bar (SPEC §13.3): "p_miss is
produced per player-week and beats a positional base rate." The
feature-engineering functions here are exercised with small hand-built
fixtures -- no live data needed for correctness. "Beats a positional
base rate" is instead verified against real historical nflverse data
(rosters/schedule/injuries/snap_counts -- none of it needs Sleeper) and
documented in HANDOFF.md, the same fixture-vs-live-run convention as
`test_evaluation_metrics.py`/`test_models_availability.py`.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ffapp.sim.injury import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_age,
    add_injury_report,
    add_missed_prior_two_seasons,
    add_snap_pct_trend,
    add_weeks_since_return,
    build_hazard_grid,
    estimate_recovery_prob,
    fit_hazard_model,
    positional_base_rate,
    predict_p_miss,
    predict_positional_base_rate,
)

# --- build_hazard_grid ------------------------------------------------------------


def test_build_hazard_grid_keeps_only_act_and_ina_status_rows() -> None:
    rosters = pl.DataFrame(
        {
            "gsis_id": ["p1", "p1", "p1", "p1"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 3, 4],
            "position": ["RB", "RB", "RB", "RB"],
            "birth_date": [date(1998, 1, 1)] * 4,
            "status": ["ACT", "INA", "DEV", "RES"],
        }
    )

    grid = build_hazard_grid(rosters)

    assert sorted(grid["week"].to_list()) == [1, 2]


def test_build_hazard_grid_excludes_non_skill_positions() -> None:
    """IDP positions (and K) are real rows in nflverse's own `rosters`
    table but out of scope for this fantasy app entirely -- confirmed
    live, they'd otherwise dwarf the fantasy-relevant population."""
    rosters = pl.DataFrame(
        {
            "gsis_id": ["p1", "p2", "p3"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "position": ["RB", "LB", "K"],
            "birth_date": [date(1998, 1, 1)] * 3,
            "status": ["ACT", "ACT", "ACT"],
        }
    )

    grid = build_hazard_grid(rosters)

    assert grid["player_id"].to_list() == ["p1"]


def test_build_hazard_grid_marks_ina_as_missed_and_act_as_played() -> None:
    rosters = pl.DataFrame(
        {
            "gsis_id": ["p1", "p1"],
            "season": [2023, 2023],
            "week": [1, 2],
            "position": ["RB", "RB"],
            "birth_date": [date(1998, 1, 1)] * 2,
            "status": ["ACT", "INA"],
        }
    )

    grid = build_hazard_grid(rosters)

    row1 = grid.filter(pl.col("week") == 1).row(0, named=True)
    row2 = grid.filter(pl.col("week") == 2).row(0, named=True)
    assert row1["missed"] is False
    assert row2["missed"] is True


# --- add_age -----------------------------------------------------------------------


def test_add_age_computes_years_from_birth_date_to_the_weeks_gameday() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2023],
            "week": [1],
            "position": ["RB"],
            "birth_date": [date(1998, 9, 1)],
            "missed": [False],
        }
    )
    schedule = pl.DataFrame({"season": [2023], "week": [1], "gameday": ["2023-09-01"]})

    result = add_age(grid, schedule)

    assert result["age"][0] == pytest.approx(25.0, abs=0.01)


# --- add_missed_prior_two_seasons ---------------------------------------------------


def test_add_missed_prior_two_seasons_sums_the_two_real_prior_seasons() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "season": [2021, 2022, 2023],
            "week": [1, 1, 1],
            "missed": [True, True, False],
        }
    )

    result = add_missed_prior_two_seasons(grid)

    # 2023's own value sums 2021 (1 miss) + 2022 (1 miss) = 2.
    row_2023 = result.filter(pl.col("season") == 2023).row(0, named=True)
    assert row_2023["missed_prior_two_seasons"] == 2
    # 2021 has no real prior seasons in this fixture -- 0, not null/crash.
    row_2021 = result.filter(pl.col("season") == 2021).row(0, named=True)
    assert row_2021["missed_prior_two_seasons"] == 0


def test_add_missed_prior_two_seasons_never_counts_the_current_season() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2023, 2023],
            "week": [1, 5],
            "missed": [False, True],
        }
    )

    result = add_missed_prior_two_seasons(grid)

    assert result["missed_prior_two_seasons"].to_list() == [0, 0]


# --- add_weeks_since_return ----------------------------------------------------------


def test_add_weeks_since_return_counts_from_the_most_recent_prior_miss() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "season": [2023] * 4,
            "week": [1, 2, 3, 4],
            "missed": [False, True, False, False],
        }
    )

    result = add_weeks_since_return(grid)

    values = result.sort("week")["weeks_since_return"].to_list()
    # week 1: no prior miss -> sentinel; week 2 is itself the miss (still no
    # *prior* miss yet); week 3 is 1 week after the week-2 miss; week 4 is 2.
    assert values[0] > 100
    assert values[1] > 100
    assert values[2] == 1
    assert values[3] == 2


def test_add_weeks_since_return_resets_at_a_season_boundary() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2022, 2023],
            "week": [17, 1],
            "missed": [True, False],
        }
    )

    result = add_weeks_since_return(grid)

    # The 2023 week-1 row has no *prior* miss within its own season.
    row_2023 = result.filter(pl.col("season") == 2023).row(0, named=True)
    assert row_2023["weeks_since_return"] > 100


# --- add_injury_report ---------------------------------------------------------------


def test_add_injury_report_joins_report_status_and_defaults_to_none() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "season": [2023, 2023],
            "week": [1, 1],
            "missed": [False, False],
        }
    )
    injuries = pl.DataFrame(
        {
            "gsis_id": ["p1"],
            "season": [2023],
            "week": [1],
            "report_status": ["Questionable"],
        }
    )

    result = add_injury_report(grid, injuries).sort("player_id")

    assert result["report_status"].to_list() == ["Questionable", "None"]


def test_add_injury_report_joins_even_when_injuries_has_float_season_and_week() -> None:
    """Real nflreadpy `injuries` data comes back with `season`/`week` as
    `Float64` (confirmed live) while `rosters`-derived grids are
    `Int32`/`Int64` -- a dtype mismatch that must not silently drop every
    real match."""
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2023], "week": [1], "missed": [False]})
    injuries = pl.DataFrame(
        {
            "gsis_id": ["p1"],
            "season": [2023.0],
            "week": [1.0],
            "report_status": ["Questionable"],
        }
    )

    result = add_injury_report(grid, injuries)

    assert result["report_status"].to_list() == ["Questionable"]


# --- add_snap_pct_trend ---------------------------------------------------------------


def test_add_snap_pct_trend_is_a_trailing_mean_excluding_the_current_week() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "missed": [False, False, False],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "pfr_player_id": ["X1", "X1", "X1"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "offense_pct": [0.5, 1.0, 0.0],
        }
    )
    crosswalk = pl.DataFrame({"pfr_id": ["X1"], "gsis_id": ["p1"]})

    result = add_snap_pct_trend(grid, snap_counts, crosswalk, window=4).sort("week")

    values = result["snap_pct_trend"].to_list()
    assert values[0] == pytest.approx(0.0)  # nothing prior yet
    assert values[1] == pytest.approx(0.5)  # trailing mean of just week 1
    assert values[2] == pytest.approx(0.75)  # trailing mean of weeks 1-2


# --- fit_hazard_model / predict_p_miss / positional_base_rate ------------------------


def _training_rows() -> pl.DataFrame:
    # RB position is deliberately miss-prone (label correlates with
    # position + report_status), WR is not -- enough real signal for a
    # tiny logistic fit to separate on, without needing real data.
    rows = []
    for i in range(60):
        is_rb = i % 2 == 0
        questionable = i % 3 == 0
        missed = is_rb and questionable
        rows.append(
            {
                "position": "RB" if is_rb else "WR",
                "age": 26.0,
                "missed_prior_two_seasons": 1 if is_rb else 0,
                "report_status": "Questionable" if questionable else "None",
                "weeks_since_return": 3,
                "snap_pct_trend": 0.6,
                TARGET_COLUMN: missed,
            }
        )
    return pl.DataFrame(rows)


def test_fit_hazard_model_and_predict_p_miss_returns_one_probability_per_row() -> None:
    train = _training_rows()

    model = fit_hazard_model(train)
    predicted = predict_p_miss(model, train)

    assert predicted.len() == train.height
    assert predicted.min() >= 0.0
    assert predicted.max() <= 1.0


def test_fit_hazard_model_learns_a_real_signal() -> None:
    """A questionable RB should predict meaningfully higher p_miss than a
    healthy WR -- proves the fit isn't just predicting the constant
    marginal rate."""
    train = _training_rows()
    model = fit_hazard_model(train)

    risky = pl.DataFrame(
        {
            "position": ["RB"],
            "age": [26.0],
            "missed_prior_two_seasons": [1],
            "report_status": ["Questionable"],
            "weeks_since_return": [3],
            "snap_pct_trend": [0.6],
        }
    )
    safe = pl.DataFrame(
        {
            "position": ["WR"],
            "age": [26.0],
            "missed_prior_two_seasons": [0],
            "report_status": ["None"],
            "weeks_since_return": [3],
            "snap_pct_trend": [0.6],
        }
    )

    p_risky = predict_p_miss(model, risky)[0]
    p_safe = predict_p_miss(model, safe)[0]
    assert p_risky > p_safe


def test_positional_base_rate_and_predict_positional_base_rate() -> None:
    train = _training_rows()

    rates = positional_base_rate(train)
    predicted = predict_positional_base_rate(rates, train.select("position"))

    assert set(rates) == {"RB", "WR"}
    assert predicted.len() == train.height


def test_feature_columns_matches_what_fit_hazard_model_actually_consumes() -> None:
    train = _training_rows()
    assert set(FEATURE_COLUMNS) <= set(train.columns)


# --- estimate_recovery_prob --------------------------------------------------


def test_estimate_recovery_prob_from_real_run_lengths() -> None:
    # player p1: misses weeks 3-5 (a real 3-week run), plays otherwise.
    # player p2: misses week 2 alone (a real 1-week run).
    # Both RB. mean run length = (3 + 1) / 2 = 2.0 -> recovery_prob = 0.5.
    grid = pl.DataFrame(
        {
            "player_id": ["p1"] * 6 + ["p2"] * 4,
            "season": [2023] * 10,
            "week": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            "position": ["RB"] * 10,
            "missed": [False, False, True, True, True, False, False, True, False, False],
        }
    )
    result = estimate_recovery_prob(grid)
    assert result["RB"] == pytest.approx(0.5)


def test_estimate_recovery_prob_omits_position_with_no_real_misses() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["q1", "q1"],
            "season": [2023, 2023],
            "week": [1, 2],
            "position": ["QB", "QB"],
            "missed": [False, False],
        }
    )
    result = estimate_recovery_prob(grid)
    assert "QB" not in result
