"""Task 1.15's conditional points model (SPEC §11.3 / §11.2 Part B):
exercised with synthetic fixtures carrying a real learnable signal, no
live `data/` needed. The real end-to-end run (beats B2 on MAE and
Spearman-within-position-week across four validation seasons) is
documented in HANDOFF.md.
"""

from __future__ import annotations

import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.models import points

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

_DEFAULT_FEATURES = dict.fromkeys(points.COMMON_FEATURE_COLUMNS, 0.0)
_DEFAULT_FEATURES.update({"report_status": "None", "practice_participation": "Full"})


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "position": "RB",
        "availability_flag": True,
        "target": 10.0,
        **_DEFAULT_FEATURES,
    }
    row.update(kwargs)
    for group in opponent.POSITION_TO_GROUPS.get(row["position"], []):
        for metric in points._OPPONENT_ADJ_METRICS:
            row.setdefault(f"{metric}_{group.lower()}", 0.0)
    return row


def _training_frame(
    n_weeks: int = 12, rows_per_week: int = 6, position: str = "RB"
) -> pl.DataFrame:
    """`target_share_ewm_3` deterministically drives `target` (points =
    10 + 50 * target_share_ewm_3) -- a real, easily-learnable relationship
    so a tiny/fast model fits it reliably without needing hundreds of
    rows."""
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(rows_per_week):
            share = i / rows_per_week
            rows.append(
                _row(
                    player_id=f"p{i}",
                    week=week,
                    position=position,
                    target_share_ewm_3=share,
                    target=10.0 + 50.0 * share,
                )
            )
    return pl.DataFrame(rows)


# --- feature_columns / opponent_feature_columns ------------------------------------------


def test_opponent_feature_columns_covers_both_groups_for_rb() -> None:
    columns = points.opponent_feature_columns("RB")

    assert "def_adj_epa_allowed_rb_receiving" in columns
    assert "def_adj_epa_allowed_rb_rushing" in columns


def test_opponent_feature_columns_covers_a_single_group_for_wr() -> None:
    columns = points.opponent_feature_columns("WR")

    assert columns == [
        "def_adj_epa_allowed_wr",
        "def_adj_success_allowed_wr",
        "def_adj_ypt_allowed_wr",
        "def_adj_td_rate_allowed_wr",
    ]


def test_feature_columns_excludes_position_itself() -> None:
    """`position` is constant within one position's own model -- it
    would carry zero information, unlike `models.availability`'s single
    cross-position classifier."""
    assert "position" not in points.feature_columns("WR")


# --- monotone_constraints ---------------------------------------------------------------


def test_monotone_constraints_marks_target_share_and_carry_share_as_increasing() -> None:
    columns = points.feature_columns("RB")
    constraints = points.monotone_constraints("RB")

    by_column = dict(zip(columns, constraints, strict=True))
    assert by_column["target_share_ewm_3"] == 1
    assert by_column["carry_share_ewm_3"] == 1
    assert by_column["implied_team_total"] == 1


def test_monotone_constraints_marks_epa_allowed_as_increasing_not_decreasing() -> None:
    """The corrected sign (see module docstring): SPEC's own literal
    "decreasing" bullet is wrong given its own stated semantic and real
    empirical data."""
    columns = points.feature_columns("WR")
    constraints = points.monotone_constraints("WR")

    by_column = dict(zip(columns, constraints, strict=True))
    assert by_column["def_adj_epa_allowed_wr"] == 1


def test_monotone_constraints_leaves_unlisted_features_unconstrained() -> None:
    columns = points.feature_columns("RB")
    constraints = points.monotone_constraints("RB")

    by_column = dict(zip(columns, constraints, strict=True))
    assert by_column["is_home"] == 0
    assert by_column["temp_f"] == 0


def test_monotone_constraints_length_matches_feature_columns() -> None:
    for position in ("QB", "RB", "WR", "TE"):
        assert len(points.monotone_constraints(position)) == len(points.feature_columns(position))


# --- fit_points_model / predict_points ---------------------------------------------------


def test_fit_points_model_returns_a_booster_per_position_present() -> None:
    train = pl.concat(
        [
            _training_frame(position="RB"),
            _training_frame(position="WR"),
        ],
        how="diagonal_relaxed",
    )

    model = points.fit_points_model(train, lightgbm_params=_FAST_PARAMS)

    assert set(model.boosters) == {"RB", "WR"}


def test_fit_points_model_excludes_non_played_rows() -> None:
    """A pile of `availability_flag=False` rows with wildly wrong-looking
    target values must not influence the fit -- if they did, predictions
    would drift from the real played-only relationship."""
    played = _training_frame()
    bogus_unplayed = pl.DataFrame(
        [
            _row(
                player_id="ghost",
                week=w,
                availability_flag=False,
                target=999.0,
                target_share_ewm_3=0.0,
            )
            for w in range(1, 13)
        ]
    )
    train_with_noise = pl.concat([played, bogus_unplayed], how="vertical_relaxed")

    model_clean = points.fit_points_model(played, lightgbm_params=_FAST_PARAMS)
    model_with_noise = points.fit_points_model(train_with_noise, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(week=13, target_share_ewm_3=0.5)])
    preds_clean = points.predict_points(model_clean, target_rows)
    preds_with_noise = points.predict_points(model_with_noise, target_rows)

    assert preds_clean[0] == pytest.approx(preds_with_noise[0], abs=0.5)


def test_predict_points_learns_the_real_target_share_relationship() -> None:
    train = _training_frame()
    model = points.fit_points_model(train, lightgbm_params=_FAST_PARAMS)

    low_share = pl.DataFrame([_row(week=13, target_share_ewm_3=0.0)])
    high_share = pl.DataFrame([_row(week=13, target_share_ewm_3=0.8)])

    low_pred = points.predict_points(model, low_share)[0]
    high_pred = points.predict_points(model, high_share)[0]

    assert high_pred > low_pred


def test_predict_points_is_null_for_a_position_with_no_fitted_booster() -> None:
    train = _training_frame(position="RB")
    model = points.fit_points_model(train, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(position="TE", week=13)])
    preds = points.predict_points(model, target_rows)

    assert preds[0] is None


def test_predict_points_scores_every_row_regardless_of_its_own_played_flag() -> None:
    """The conditional expectation is defined for "if this player plays" --
    a target row with `availability_flag=False` still gets a real
    prediction, not a null."""
    train = _training_frame()
    model = points.fit_points_model(train, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(week=13, availability_flag=False, target_share_ewm_3=0.5)])
    preds = points.predict_points(model, target_rows)

    assert preds[0] is not None


# --- PointsPredictor (evaluation.backtest.Predictor conformance) -----------------------


def test_points_predictor_fit_predict_round_trip() -> None:
    predictor = points.PointsPredictor(_FAST_PARAMS)
    train = _training_frame()

    fitted = predictor.fit(train)
    preds = predictor.predict(fitted, train)

    assert preds.len() == train.height
    assert predictor.name == "conditional_points_lightgbm"
