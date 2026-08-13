"""Task 1.16's quantile models (SPEC §11.5): exercised with synthetic
fixtures and small numeric scenarios, no live `data/` needed. The real
end-to-end run (80% interval coverage within 5 points of nominal, per
position) is documented in HANDOFF.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.models import points, quantiles

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)
_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)

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
    for metric in points._OPPONENT_ADJ_METRICS:
        row.setdefault(f"{metric}_rb_receiving", 0.0)
        row.setdefault(f"{metric}_rb_rushing", 0.0)
    return row


def _training_frame(n_weeks: int = 16, rows_per_week: int = 8) -> pl.DataFrame:
    """`target_share_ewm_3` drives a real, noisy-but-learnable spread of
    outcomes (mean 10 + 50*share, real per-row noise via a fixed seeded
    rng) -- enough real variance for quantile regression to have
    something genuine to fit, unlike task 1.15's fully deterministic
    fixture."""
    rng = np.random.default_rng(0)
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(rows_per_week):
            share = i / rows_per_week
            noise = rng.normal(0, 3.0)
            rows.append(
                _row(
                    player_id=f"p{i}",
                    week=week,
                    target_share_ewm_3=share,
                    target=max(0.0, 10.0 + 50.0 * share + noise),
                )
            )
    return pl.DataFrame(rows)


# --- find_width_scale -----------------------------------------------------------------


def test_find_width_scale_widens_an_undercovered_interval() -> None:
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 10, size=2000)
    median = np.zeros_like(actual)
    lower = np.full_like(actual, -2.0)  # deliberately too narrow for 80% coverage
    upper = np.full_like(actual, 2.0)

    scale = quantiles.find_width_scale(
        median=median, lower=lower, upper=upper, actual=actual, nominal_coverage=0.80
    )

    assert scale > 1.0


def test_find_width_scale_narrows_an_overcovered_interval() -> None:
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, size=2000)
    median = np.zeros_like(actual)
    lower = np.full_like(actual, -20.0)  # deliberately far too wide
    upper = np.full_like(actual, 20.0)

    scale = quantiles.find_width_scale(
        median=median, lower=lower, upper=upper, actual=actual, nominal_coverage=0.80
    )

    assert scale < 1.0


def test_find_width_scale_achieves_close_to_nominal_coverage() -> None:
    rng = np.random.default_rng(2)
    actual = rng.normal(0, 5, size=5000)
    median = np.zeros_like(actual)
    lower = np.full_like(actual, -1.0)
    upper = np.full_like(actual, 1.0)

    scale = quantiles.find_width_scale(
        median=median, lower=lower, upper=upper, actual=actual, nominal_coverage=0.80
    )
    scaled_lower = median + scale * (lower - median)
    scaled_upper = median + scale * (upper - median)
    achieved = ((actual >= scaled_lower) & (actual <= scaled_upper)).mean()

    assert achieved == pytest.approx(0.80, abs=0.02)


# --- fit_quantile_models / predict_quantiles --------------------------------------------


def test_fit_quantile_models_returns_a_booster_per_alpha_per_position() -> None:
    train = _training_frame()

    model = quantiles.fit_quantile_models(
        train, lightgbm_params=_FAST_PARAMS, quantile_alphas=_ALPHAS
    )

    assert set(model.boosters) == {"RB"}
    assert set(model.boosters["RB"]) == set(_ALPHAS)


def test_default_calibration_weeks_is_twelve_not_availability_modules_four() -> None:
    """Confirmed live against real data: 4 weeks (models.availability's
    own default) is too small a sample to stabilize a width-scale
    estimate on the thinner positions -- see module docstring."""
    assert quantiles.DEFAULT_CALIBRATION_WEEKS == 12


def test_predict_quantiles_is_sorted_ascending_per_row() -> None:
    train = _training_frame()
    model = quantiles.fit_quantile_models(
        train, lightgbm_params=_FAST_PARAMS, quantile_alphas=_ALPHAS
    )

    target_rows = pl.DataFrame(
        [_row(week=17, target_share_ewm_3=s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    )
    preds = quantiles.predict_quantiles(model, target_rows)

    values = preds.select([f"q_{a}" for a in sorted(_ALPHAS)]).to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()


def test_predict_quantiles_median_increases_with_target_share() -> None:
    train = _training_frame()
    model = quantiles.fit_quantile_models(
        train, lightgbm_params=_FAST_PARAMS, quantile_alphas=_ALPHAS
    )

    low = quantiles.predict_quantiles(model, pl.DataFrame([_row(week=17, target_share_ewm_3=0.0)]))
    high = quantiles.predict_quantiles(model, pl.DataFrame([_row(week=17, target_share_ewm_3=0.9)]))

    assert high["q_0.5"][0] > low["q_0.5"][0]


def test_predict_quantiles_is_null_for_a_position_with_no_fitted_booster() -> None:
    train = _training_frame()
    model = quantiles.fit_quantile_models(
        train, lightgbm_params=_FAST_PARAMS, quantile_alphas=_ALPHAS
    )

    target_rows = pl.DataFrame([_row(position="TE", week=17)])
    preds = quantiles.predict_quantiles(model, target_rows)

    assert preds["q_0.5"][0] is None


def test_fit_quantile_models_records_a_real_crossing_rate_and_width_scale() -> None:
    train = _training_frame()

    model = quantiles.fit_quantile_models(
        train, lightgbm_params=_FAST_PARAMS, quantile_alphas=_ALPHAS
    )

    assert 0.0 <= model.crossing_rate["RB"] <= 1.0
    assert model.width_scale["RB"] > 0.0


# --- mixture_with_p_active --------------------------------------------------------------


def test_mixture_with_p_active_floors_at_zero_when_probability_is_low() -> None:
    """A player with p_active=0.2 -- the bottom 80% of outcomes are
    exactly 0 (the point mass), so even the 0.75 quantile should floor
    at 0."""
    conditional = pl.DataFrame({f"q_{a}": [10.0] for a in _ALPHAS})
    p_active = pl.Series([0.2])

    result = quantiles.mixture_with_p_active(conditional, p_active, _ALPHAS)

    assert result["unconditional_q_0.75"][0] == pytest.approx(0.0)


def test_mixture_with_p_active_matches_conditional_quantiles_when_p_is_one() -> None:
    """p_active=1.0 -- no point mass at all, the mixture collapses to the
    conditional distribution exactly."""
    conditional = pl.DataFrame({f"q_{a}": [10.0 + 20.0 * a] for a in _ALPHAS})
    p_active = pl.Series([1.0])

    result = quantiles.mixture_with_p_active(conditional, p_active, _ALPHAS)

    for a in _ALPHAS:
        assert result[f"unconditional_q_{a}"][0] == pytest.approx(
            conditional[f"q_{a}"][0], abs=1e-6
        )


def test_mixture_with_p_active_never_produces_negative_values() -> None:
    conditional = pl.DataFrame({f"q_{a}": [-5.0] for a in _ALPHAS})
    p_active = pl.Series([0.9])

    result = quantiles.mixture_with_p_active(conditional, p_active, _ALPHAS)

    for a in _ALPHAS:
        assert result[f"unconditional_q_{a}"][0] >= 0.0


def test_mixture_with_p_active_is_monotonic_in_p_active_for_a_high_quantile() -> None:
    """Higher p_active -> a higher-probability player -> the unconditional
    0.9 quantile should not decrease as p_active rises (more mass shifts
    off the zero floor and onto the real conditional distribution)."""
    conditional = pl.DataFrame({f"q_{a}": [10.0 + 20.0 * a] for a in _ALPHAS})

    low_p = quantiles.mixture_with_p_active(conditional, pl.Series([0.3]), _ALPHAS)
    high_p = quantiles.mixture_with_p_active(conditional, pl.Series([0.95]), _ALPHAS)

    assert high_p["unconditional_q_0.9"][0] >= low_p["unconditional_q_0.9"][0]
