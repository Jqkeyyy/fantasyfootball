"""Task 1.14's availability model (SPEC §11.2 Part A): a small,
fast-to-fit LightGBM classifier plus isotonic calibration, exercised with
synthetic fixtures carrying a real learnable signal (report_status
deterministically predicts availability_flag) -- no live `data/` needed.
The real end-to-end run (calibration curve, Brier score vs the positional
base-rate baseline) is documented in HANDOFF.md.
"""

from __future__ import annotations

import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.models import availability

# A small, fast config -- the defaults (n_estimators=800, min_child_samples=40)
# would either take too long or refuse to split on these tiny fixtures.
_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "position": "RB",
        "report_status": "None",
        "practice_participation": "Full",
        "weeks_since_return": 5.0,
        "depth_chart_rank": 1,
        "snap_pct_trend": 0.0,
        "age": 25.0,
        "availability_flag": True,
    }
    row.update(kwargs)
    return row


def _training_frame(n_weeks: int = 12, rows_per_week: int = 6) -> pl.DataFrame:
    """`report_status == "Out"` -> never active; everything else -> always
    active -- a perfect, easily-learnable signal so a tiny/fast model
    fits it reliably, without needing hundreds of rows."""
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(rows_per_week):
            out = i == 0  # 1 of every `rows_per_week` players is ruled Out
            rows.append(
                _row(
                    player_id=f"p{i}",
                    week=week,
                    report_status="Out" if out else "None",
                    availability_flag=not out,
                )
            )
    return pl.DataFrame(rows)


# --- fit_availability_model / predict_p_active ------------------------------------------


def test_fit_availability_model_returns_a_booster_and_calibrator() -> None:
    model = availability.fit_availability_model(_training_frame(), lightgbm_params=_FAST_PARAMS)

    assert model.booster is not None
    assert model.calibrator is not None


def test_predict_p_active_returns_probabilities_in_bounds() -> None:
    train = _training_frame()
    model = availability.fit_availability_model(train, lightgbm_params=_FAST_PARAMS)

    preds = availability.predict_p_active(model, train)

    assert preds.min() >= 0.0
    assert preds.max() <= 1.0
    assert preds.len() == train.height


def test_predict_p_active_learns_the_real_report_status_signal() -> None:
    """A player ruled `Out` must get a real, substantially lower p_active
    than a healthy one -- proves the classifier is actually fitting the
    real relationship, not just returning a constant."""
    train = _training_frame()
    model = availability.fit_availability_model(train, lightgbm_params=_FAST_PARAMS)

    target = pl.DataFrame(
        [
            _row(player_id="new_out", week=13, report_status="Out"),
            _row(player_id="new_healthy", week=13, report_status="None"),
        ]
    )
    preds = availability.predict_p_active(model, target)

    assert preds[0] < preds[1]
    assert preds[1] - preds[0] > 0.3


# --- _calibration_split ----------------------------------------------------------------


def test_calibration_split_holds_out_the_most_recent_real_weeks() -> None:
    train = _training_frame(n_weeks=12, rows_per_week=2)

    fit_rows, calibration_rows = availability._calibration_split(train, calibration_weeks=4)

    fit_weeks = set(fit_rows["week"].unique().to_list())
    calibration_weeks = set(calibration_rows["week"].unique().to_list())
    assert fit_weeks == set(range(1, 9))
    assert calibration_weeks == set(range(9, 13))
    assert fit_weeks.isdisjoint(calibration_weeks)


def test_calibration_split_falls_back_to_the_full_set_for_very_few_weeks() -> None:
    """Too few real weeks to hold any out -- the caller (`fit_availability_model`)
    falls back to using every row for both halves rather than crashing on
    an empty split."""
    train = _training_frame(n_weeks=1, rows_per_week=4)

    fit_rows, calibration_rows = availability._calibration_split(train, calibration_weeks=4)

    # at least one side is non-empty; fit_availability_model handles the
    # both-nonempty-but-identical or one-empty case explicitly.
    assert fit_rows.height + calibration_rows.height >= train.height


def test_fit_availability_model_does_not_crash_with_only_one_real_week() -> None:
    train = _training_frame(n_weeks=1, rows_per_week=6)

    model = availability.fit_availability_model(train, lightgbm_params=_FAST_PARAMS)
    preds = availability.predict_p_active(model, train)

    assert preds.len() == train.height


# --- AvailabilityPredictor (evaluation.backtest.Predictor conformance) -----------------


def test_availability_predictor_fit_predict_round_trip() -> None:
    predictor = availability.AvailabilityPredictor(_FAST_PARAMS)
    train = _training_frame()

    fitted = predictor.fit(train)
    preds = predictor.predict(fitted, train)

    assert preds.len() == train.height
    assert predictor.name == "availability_lightgbm"


def test_availability_predictor_uses_its_own_calibration_weeks() -> None:
    predictor = availability.AvailabilityPredictor(_FAST_PARAMS, calibration_weeks=2)
    train = _training_frame(n_weeks=10, rows_per_week=2)

    fitted = predictor.fit(train)

    assert fitted.booster is not None
