"""Task 1.20's anchored residual model (SPEC-ADDENDUM-04.md §B): exercised
with synthetic fixtures carrying a real learnable signal, no live `data/`
needed. The real end-to-end evaluation (Spearman/lineup regret vs B2 vs
consensus, real validation seasons) is documented in HANDOFF.md.
"""

from __future__ import annotations

import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.models import points, residual

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
        "b2_ewm_4": 10.0,
        "ewm_points_8": 10.0,
        "points_last_week": 10.0,
        **_DEFAULT_FEATURES,
    }
    row.update(kwargs)
    for group in opponent.POSITION_TO_GROUPS.get(row["position"], []):
        for metric in points._OPPONENT_ADJ_METRICS:
            row.setdefault(f"{metric}_{group.lower()}", 0.0)
    return row


def _training_frame(
    n_weeks: int = 12, rows_per_week: int = 6, position: str = "RB", anchor: float = 8.0
) -> pl.DataFrame:
    """`target_share_ewm_3` deterministically drives the RESIDUAL against
    a fixed anchor (`target = anchor + 20 * target_share_ewm_3`, so
    `target - b2_ewm_4 == 20 * target_share_ewm_3` exactly when
    `b2_ewm_4 == anchor`) -- a real, easily-learnable relationship for a
    tiny/fast model."""
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
                    b2_ewm_4=anchor,
                    target=anchor + 20.0 * share,
                )
            )
    return pl.DataFrame(rows)


# --- residual_feature_columns / residual_monotone_constraints ----------------------------


def test_residual_feature_columns_extends_points_columns_with_anchor_and_history() -> None:
    columns = residual.residual_feature_columns("RB")
    base = points.feature_columns("RB")

    assert columns[: len(base)] == base
    assert columns[len(base) :] == ["b2_ewm_4", "ewm_points_8", "points_last_week"]


def test_residual_monotone_constraints_leaves_new_columns_unconstrained() -> None:
    columns = residual.residual_feature_columns("RB")
    constraints = residual.residual_monotone_constraints("RB")

    by_column = dict(zip(columns, constraints, strict=True))
    assert by_column["b2_ewm_4"] == 0
    assert by_column["ewm_points_8"] == 0
    assert by_column["points_last_week"] == 0
    # base constraints carried through unchanged
    assert by_column["target_share_ewm_3"] == 1


def test_residual_monotone_constraints_length_matches_feature_columns() -> None:
    for position in ("QB", "RB", "WR", "TE"):
        assert len(residual.residual_monotone_constraints(position)) == len(
            residual.residual_feature_columns(position)
        )


# --- add_points_history_features ----------------------------------------------------------


def test_add_points_history_features_shifts_strictly_through_prior_week() -> None:
    df = pl.DataFrame(
        [
            {"player_id": "p1", "season": 2025, "week": 1, "target": 10.0},
            {"player_id": "p1", "season": 2025, "week": 2, "target": 20.0},
            {"player_id": "p1", "season": 2025, "week": 3, "target": 30.0},
        ]
    )
    out = residual.add_points_history_features(df)

    # week 1 has no prior history -- both columns null
    week1 = out.filter(pl.col("week") == 1)
    assert week1["ewm_points_8"][0] is None
    assert week1["points_last_week"][0] is None

    # week 2's points_last_week is week 1's own target, not week 2's own
    week2 = out.filter(pl.col("week") == 2)
    assert week2["points_last_week"][0] == pytest.approx(10.0)

    # week 3's points_last_week is week 2's own target
    week3 = out.filter(pl.col("week") == 3)
    assert week3["points_last_week"][0] == pytest.approx(20.0)


def test_add_points_history_features_resets_across_seasons() -> None:
    df = pl.DataFrame(
        [
            {"player_id": "p1", "season": 2024, "week": 18, "target": 99.0},
            {"player_id": "p1", "season": 2025, "week": 1, "target": 10.0},
        ]
    )
    out = residual.add_points_history_features(df)

    season2025_week1 = out.filter((pl.col("season") == 2025) & (pl.col("week") == 1))
    assert season2025_week1["points_last_week"][0] is None


# --- fit_residual_model / predict_residual_points -----------------------------------------


def test_fit_residual_model_returns_a_booster_per_position_present() -> None:
    train = pl.concat(
        [_training_frame(position="RB"), _training_frame(position="WR")],
        how="diagonal_relaxed",
    )

    model = residual.fit_residual_model(train, lightgbm_params=_FAST_PARAMS)

    assert set(model.boosters) == {"RB", "WR"}


def test_fit_residual_model_excludes_non_played_rows() -> None:
    played = _training_frame()
    bogus_unplayed = pl.DataFrame(
        [
            _row(
                player_id="ghost",
                week=w,
                availability_flag=False,
                target=999.0,
                b2_ewm_4=8.0,
                target_share_ewm_3=0.0,
            )
            for w in range(1, 13)
        ]
    )
    train_with_noise = pl.concat([played, bogus_unplayed], how="vertical_relaxed")

    model_clean = residual.fit_residual_model(played, lightgbm_params=_FAST_PARAMS)
    model_with_noise = residual.fit_residual_model(train_with_noise, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(week=13, b2_ewm_4=8.0, target_share_ewm_3=0.5)])
    preds_clean = residual.predict_residual_points(model_clean, target_rows)
    preds_with_noise = residual.predict_residual_points(model_with_noise, target_rows)

    assert preds_clean[0] == pytest.approx(preds_with_noise[0], abs=1.0)


def test_fit_residual_model_excludes_rows_with_null_anchor() -> None:
    """A row with no `b2_ewm_4` (a player's own first tracked week) must
    not corrupt the fit -- its residual is undefined."""
    played = _training_frame()
    bogus_null_anchor = played.with_columns(
        pl.when(pl.col("player_id") == "p0")
        .then(None)
        .otherwise(pl.col("b2_ewm_4"))
        .alias("b2_ewm_4"),
        pl.when(pl.col("player_id") == "p0")
        .then(9999.0)
        .otherwise(pl.col("target"))
        .alias("target"),
    )

    model_clean = residual.fit_residual_model(played, lightgbm_params=_FAST_PARAMS)
    model_with_null = residual.fit_residual_model(bogus_null_anchor, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(week=13, b2_ewm_4=8.0, target_share_ewm_3=0.5)])
    preds_clean = residual.predict_residual_points(model_clean, target_rows)
    preds_with_null = residual.predict_residual_points(model_with_null, target_rows)

    assert preds_clean[0] == pytest.approx(preds_with_null[0], abs=1.0)


def test_predict_residual_points_composes_anchor_and_learned_residual() -> None:
    train = _training_frame(anchor=8.0)
    model = residual.fit_residual_model(train, lightgbm_params=_FAST_PARAMS)

    low_share = pl.DataFrame([_row(week=13, b2_ewm_4=8.0, target_share_ewm_3=0.0)])
    high_share = pl.DataFrame([_row(week=13, b2_ewm_4=8.0, target_share_ewm_3=0.8)])

    low_pred = residual.predict_residual_points(model, low_share)[0]
    high_pred = residual.predict_residual_points(model, high_share)[0]

    assert high_pred > low_pred
    # A model that has learned the zero-residual case returns close to
    # the anchor itself.
    assert low_pred == pytest.approx(8.0, abs=2.0)


def test_predict_residual_points_tracks_a_different_anchor_at_predict_time() -> None:
    """The SAME fitted residual relationship, applied on top of a
    DIFFERENT real anchor value at predict time -- proves the model
    genuinely composes with B2 rather than memorising an absolute
    points level."""
    train = _training_frame(anchor=8.0)
    model = residual.fit_residual_model(train, lightgbm_params=_FAST_PARAMS)

    low_anchor = pl.DataFrame([_row(week=13, b2_ewm_4=2.0, target_share_ewm_3=0.0)])
    high_anchor = pl.DataFrame([_row(week=13, b2_ewm_4=20.0, target_share_ewm_3=0.0)])

    low_pred = residual.predict_residual_points(model, low_anchor)[0]
    high_pred = residual.predict_residual_points(model, high_anchor)[0]

    assert high_pred - low_pred == pytest.approx(18.0, abs=2.0)


def test_predict_residual_points_is_null_when_anchor_is_missing() -> None:
    train = _training_frame()
    model = residual.fit_residual_model(train, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(week=13, b2_ewm_4=None, target_share_ewm_3=0.5)])
    preds = residual.predict_residual_points(model, target_rows)

    assert preds[0] is None


def test_predict_residual_points_is_null_for_a_position_with_no_fitted_booster() -> None:
    train = _training_frame(position="RB")
    model = residual.fit_residual_model(train, lightgbm_params=_FAST_PARAMS)

    target_rows = pl.DataFrame([_row(position="TE", week=13, b2_ewm_4=8.0)])
    preds = residual.predict_residual_points(model, target_rows)

    assert preds[0] is None


# --- ResidualPredictor (evaluation.backtest.Predictor conformance) ------------------------


def test_residual_predictor_fit_predict_round_trip() -> None:
    predictor = residual.ResidualPredictor(_FAST_PARAMS)
    train = _training_frame()

    fitted = predictor.fit(train)
    preds = predictor.predict(fitted, train)

    assert preds.len() == train.height
    assert predictor.name == "anchored_residual"


# --- fit_blend_weight / apply_blend_weight -------------------------------------------------


def _predictions_row(
    player_id: str,
    season: int,
    week: int,
    position: str,
    predictor: str,
    prediction: float,
    target: float,
) -> dict:
    return {
        "player_id": player_id,
        "season": season,
        "week": week,
        "position": position,
        "team": "AAA",
        "played": True,
        "target": target,
        "predictor": predictor,
        "prediction": prediction,
    }


def _blend_fixture() -> pl.DataFrame:
    """Two positions with opposite real ranking outcomes: at RB the
    anchored-residual prediction ranks players in the exact real order
    (perfect Spearman), b2 does not; at QB it's the reverse. `w` should
    land near 1.0 for RB and 0.0 for QB."""
    rows: list[dict] = []
    for week in range(1, 6):
        rb_targets = [10.0, 20.0, 30.0]
        for i, t in enumerate(rb_targets):
            rows.append(_predictions_row(f"rb{i}", 2025, week, "RB", "anchored_residual", t, t))
            rows.append(
                _predictions_row(f"rb{i}", 2025, week, "RB", "b2_ewm_4", rb_targets[-1 - i], t)
            )
        qb_targets = [5.0, 15.0, 25.0]
        for i, t in enumerate(qb_targets):
            rows.append(
                _predictions_row(
                    f"qb{i}", 2025, week, "QB", "anchored_residual", qb_targets[-1 - i], t
                )
            )
            rows.append(_predictions_row(f"qb{i}", 2025, week, "QB", "b2_ewm_4", t, t))
    return pl.DataFrame(rows)


def test_fit_blend_weight_favors_residual_where_it_ranks_better() -> None:
    """Both extremes achieve perfect (tied) Spearman across a range of
    `w` with only 3 players per position -- the grid search's own
    smallest-w tie-break means RB doesn't necessarily land exactly at
    1.0, but it must land meaningfully higher than QB, whose ranking
    only improves as w shrinks toward 0."""
    weights = residual.fit_blend_weight(_blend_fixture())

    assert weights["RB"] >= 0.5
    assert weights["QB"] == pytest.approx(0.0)


def test_fit_blend_weight_defaults_to_zero_for_a_position_with_no_rows() -> None:
    weights = residual.fit_blend_weight(_blend_fixture())

    assert weights["TE"] == 0.0
    assert weights["WR"] == 0.0


def test_apply_blend_weight_computes_the_weighted_average() -> None:
    fixture = _blend_fixture()
    blended = residual.apply_blend_weight(fixture, {"RB": 0.25, "QB": 0.0})

    rb_row = blended.filter((pl.col("position") == "RB") & (pl.col("player_id") == "rb0")).row(
        0, named=True
    )
    # residual_pred=10.0, b2_pred=30.0, w=0.25 -> 0.25*10 + 0.75*30 = 25.0
    assert rb_row["prediction"] == pytest.approx(25.0)
    assert rb_row["predictor"] == "anchored_residual_blend"

    qb_row = blended.filter((pl.col("position") == "QB") & (pl.col("player_id") == "qb0")).row(
        0, named=True
    )
    # w=0.0 -> falls back to b2_pred exactly
    assert qb_row["prediction"] == pytest.approx(qb_row["target"])


def test_apply_blend_weight_output_schema_matches_input() -> None:
    fixture = _blend_fixture()
    blended = residual.apply_blend_weight(fixture, {"RB": 0.5, "QB": 0.5})

    assert blended.columns == fixture.columns
