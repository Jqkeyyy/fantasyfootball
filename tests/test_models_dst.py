"""Task 2.7's DST model: exercised with synthetic fixtures carrying a
real learnable signal, no live `data/` needed. The real end-to-end run
(beats B2 on a live 2015-2025 backtest) is documented in HANDOFF.md,
the same fixture-vs-live-run split `models.points`'s own test module
uses.
"""

from __future__ import annotations

import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.features.dst import FEATURE_COLUMNS
from ffapp.models import dst

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)


def _feature_row(**kwargs: object) -> dict:
    row: dict[str, object] = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    row.update(
        {
            "team": "A",
            "opponent_team": "B",
            "season": 2025,
            "week": 1,
            "is_home": True,
            "is_dome": False,
        }
    )
    row.update(kwargs)
    return row


def _training_frame(n_weeks: int = 12, teams: int = 6) -> pl.DataFrame:
    """`opp_implied_team_total` deterministically drives `target` --
    points = 2 + 0.4 * opp_implied_team_total -- a real, easily-learnable
    relationship (a defence facing a higher-scoring offense should score
    fewer real fantasy points, but the *sign* doesn't matter for this
    roundtrip check, only that the model can learn *some* real relationship)."""
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(teams):
            total = 15.0 + i * 2.0
            rows.append(
                _feature_row(
                    team=f"t{i}",
                    week=week,
                    opp_implied_team_total=total,
                    target=2.0 + 0.4 * total,
                )
            )
    return pl.DataFrame(rows)


# --- build_dst_table ---------------------------------------------------------------------


def test_build_dst_table_adds_player_id_position_availability_flag() -> None:
    features = pl.DataFrame(
        [{"team": "KC", "opponent_team": "BAL", "season": 2025, "week": 1, "wind_mph": 5.0}]
    )
    target = pl.DataFrame([{"team": "KC", "season": 2025, "week": 1, "target": 8.0}])

    result = dst.build_dst_table(features, target)

    row = result.row(0, named=True)
    assert row["player_id"] == "KC"
    assert row["position"] == "DST"
    assert row["availability_flag"] is True
    assert row["target"] == 8.0


def test_build_dst_table_only_keeps_rows_with_a_real_target() -> None:
    """An inner join, not a left join -- a team-week with features but no
    matching target (shouldn't happen for real data, but not assumed)
    is dropped, not silently scored against a null target."""
    features = pl.DataFrame(
        [
            {"team": "KC", "opponent_team": "BAL", "season": 2025, "week": 1},
            {"team": "KC", "opponent_team": "BUF", "season": 2025, "week": 2},
        ]
    )
    target = pl.DataFrame([{"team": "KC", "season": 2025, "week": 1, "target": 8.0}])

    result = dst.build_dst_table(features, target)

    assert result.height == 1


# --- add_dst_b2_ewm_4 --------------------------------------------------------------------


def test_add_dst_b2_ewm_4_excludes_the_target_weeks_own_outcome() -> None:
    table = pl.DataFrame(
        [
            {"player_id": "KC", "season": 2025, "week": 1, "target": 10.0},
            {"player_id": "KC", "season": 2025, "week": 2, "target": 20.0},
        ]
    )

    result = dst.add_dst_b2_ewm_4(table)

    row1 = result.filter(pl.col("week") == 1).row(0, named=True)
    row2 = result.filter(pl.col("week") == 2).row(0, named=True)
    assert row1["dst_b2_ewm_4"] is None  # no prior week
    assert row2["dst_b2_ewm_4"] == 10.0  # week 1's own value, not week 2's


# --- fit_dst_model / predict_dst ----------------------------------------------------------


def test_predict_dst_learns_a_real_relationship() -> None:
    train = _training_frame()
    model = dst.fit_dst_model(train, lightgbm_params=_FAST_PARAMS)

    low = pl.DataFrame([_feature_row(week=13, opp_implied_team_total=15.0)])
    high = pl.DataFrame([_feature_row(week=13, opp_implied_team_total=25.0)])

    low_pred = dst.predict_dst(model, low)[0]
    high_pred = dst.predict_dst(model, high)[0]

    assert high_pred != low_pred


# --- DstPredictor (evaluation.backtest.Predictor conformance) -----------------------------


def test_dst_predictor_fit_predict_round_trip() -> None:
    predictor = dst.DstPredictor(_FAST_PARAMS)
    train = _training_frame()

    fitted = predictor.fit(train)
    preds = predictor.predict(fitted, train)

    assert preds.len() == train.height
    assert predictor.name == "dst_lightgbm"


# --- weekly_streamer_list -----------------------------------------------------------------


def test_weekly_streamer_list_ranks_by_prediction_descending() -> None:
    predictions = pl.DataFrame(
        [
            {"player_id": "KC", "season": 2025, "week": 10, "prediction": 7.0},
            {"player_id": "BAL", "season": 2025, "week": 10, "prediction": 12.0},
            {"player_id": "NYJ", "season": 2025, "week": 10, "prediction": 4.0},
            {"player_id": "KC", "season": 2025, "week": 9, "prediction": 99.0},  # other week
        ]
    )

    result = dst.weekly_streamer_list(predictions, season=2025, week=10)

    assert result["team"].to_list() == ["BAL", "KC", "NYJ"]
    assert result["prediction"].to_list() == [12.0, 7.0, 4.0]
