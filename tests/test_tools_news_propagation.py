"""Second-order news propagation (SPEC.md §14.8; task 2.10). Small,
fast-fitting LightGBM fixtures, same convention as
`tests/test_models_predict.py` -- no live `data/` needed here; the real
historical-scenario verification is documented in docs/JOURNAL.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.features.team_context import RULED_OUT_STATUS
from ffapp.models import points as points_module
from ffapp.tools import news_propagation as prop

_FAST_PARAMS = LightGBMSettings(
    n_estimators=15,
    learning_rate=0.3,
    num_leaves=7,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

_DEFAULT_FEATURES = dict.fromkeys(points_module.COMMON_FEATURE_COLUMNS, 0.0)
_DEFAULT_FEATURES.update(
    {
        "report_status": "None",
        "practice_participation": "Full",
        "depth_chart_rank": 1.0,
        "age": 25.0,
    }
)

TEAM = "SF"
RB1 = "p_rb1"  # gets ruled out
RB2 = "p_rb2"  # the real handcuff


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "team": TEAM,
        "position": "RB",
        "availability_flag": True,
        "target": 10.0,
        **_DEFAULT_FEATURES,
    }
    row.update(kwargs)
    for group in opponent.POSITION_TO_GROUPS.get(row["position"], []):
        for metric in points_module._OPPONENT_ADJ_METRICS:
            row.setdefault(f"{metric}_{group.lower()}", 0.0)
    return row


def _features_fixture() -> pl.DataFrame:
    """Weeks 1-4: real training history where `teammate_vacated_target_share`
    genuinely, deterministically drives RB2's own real `target` (weeks 2
    and 4 simulate "RB1 was out that week") -- a learnable signal, same
    convention as `test_models_predict.py`'s own `target_share_ewm_3`.
    Week 5 is the real target week: RB1 still shows `availability_flag
    =True` (not yet officially marked out -- that's what this session's
    own synthetic event does), RB2's own `teammate_vacated_target_share`
    is still 0 (no one currently out).
    """
    rows = []
    for week in range(1, 5):
        rb1_out_this_week = week in (2, 4)
        rows.append(
            _row(
                player_id=RB1,
                week=week,
                target_share_ewm_3=0.5,
                carry_share_ewm_3=0.5,
                availability_flag=not rb1_out_this_week,
                target=0.0 if rb1_out_this_week else 15.0,
            )
        )
        rows.append(
            _row(
                player_id=RB2,
                week=week,
                target_share_ewm_3=0.2,
                carry_share_ewm_3=0.2,
                teammate_vacated_target_share=0.5 if rb1_out_this_week else 0.0,
                teammate_vacated_carry_share=0.5 if rb1_out_this_week else 0.0,
                availability_flag=True,
                target=28.0 if rb1_out_this_week else 8.0,
            )
        )
    # Week 5: the real target week, nobody currently marked Out.
    rows.append(
        _row(
            player_id=RB1,
            week=5,
            target_share_ewm_3=0.5,
            carry_share_ewm_3=0.5,
            availability_flag=True,
            target=15.0,
        )
    )
    rows.append(
        _row(
            player_id=RB2,
            week=5,
            target_share_ewm_3=0.2,
            carry_share_ewm_3=0.2,
            teammate_vacated_target_share=0.0,
            teammate_vacated_carry_share=0.0,
            availability_flag=True,
            target=8.0,
        )
    )
    return pl.DataFrame(rows)


def _team_week_context_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": [TEAM] * 5,
            "season": [2025] * 5,
            "week": [1, 2, 3, 4, 5],
        }
    )


def _empty_injuries() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.String,
            "season": pl.Int64,
            "week": pl.Int64,
            "report_status": pl.String,
        }
    )


class TestSyntheticOutRow:
    def test_has_the_real_shape_add_vacated_shares_expects(self) -> None:
        row = prop.synthetic_out_row("p1", season=2025, week=5)

        d = row.to_dicts()[0]
        assert d["player_id"] == "p1"
        assert d["season"] == 2025
        assert d["week"] == 5
        assert d["report_status"] == RULED_OUT_STATUS


class TestRecomputeVacatedShares:
    def test_a_wider_real_shaped_injuries_table_still_works(self) -> None:
        # The real interim/injuries.parquet carries columns (team,
        # practice_status, report_primary_injury, date_modified, ...)
        # the synthetic row doesn't set -- a same-shaped fixture would
        # never catch a real vertical-concat schema mismatch.
        wide_injuries = pl.DataFrame(
            {
                "player_id": ["someone_else"],
                "season": [2025],
                "week": [3],
                "team": [TEAM],
                "report_status": ["Questionable"],
                "practice_status": ["Limited"],
                "report_primary_injury": ["Ankle"],
            }
        )
        features = _features_fixture()

        recomputed = prop.recompute_vacated_shares(
            _team_week_context_fixture(),
            wide_injuries,
            features,
            ruled_out_player_id=RB1,
            season=2025,
            week=5,
        )

        row = recomputed.filter(
            (pl.col("team") == TEAM) & (pl.col("season") == 2025) & (pl.col("week") == 5)
        ).row(0, named=True)
        assert row["teammate_vacated_target_share"] == pytest.approx(0.5)

    def test_recomputes_a_real_nonzero_vacated_share_for_the_team_week(self) -> None:
        features = _features_fixture()
        recomputed = prop.recompute_vacated_shares(
            _team_week_context_fixture(),
            _empty_injuries(),
            features,
            ruled_out_player_id=RB1,
            season=2025,
            week=5,
        )

        row = recomputed.filter(
            (pl.col("team") == TEAM) & (pl.col("season") == 2025) & (pl.col("week") == 5)
        ).row(0, named=True)
        assert row["teammate_vacated_target_share"] == pytest.approx(0.5)
        assert row["teammate_vacated_carry_share"] == pytest.approx(0.5)

    def test_a_week_with_no_synthetic_event_is_unaffected(self) -> None:
        features = _features_fixture()
        recomputed = prop.recompute_vacated_shares(
            _team_week_context_fixture(),
            _empty_injuries(),
            features,
            ruled_out_player_id=RB1,
            season=2025,
            week=5,
        )

        row = recomputed.filter(
            (pl.col("team") == TEAM) & (pl.col("season") == 2025) & (pl.col("week") == 1)
        ).row(0, named=True)
        assert row["teammate_vacated_target_share"] == pytest.approx(0.0)


class TestPatchVacatedShares:
    def test_patches_only_the_named_team_week(self) -> None:
        features = _features_fixture()
        recomputed = prop.recompute_vacated_shares(
            _team_week_context_fixture(),
            _empty_injuries(),
            features,
            ruled_out_player_id=RB1,
            season=2025,
            week=5,
        )

        patched = prop.patch_vacated_shares(features, recomputed, team=TEAM, season=2025, week=5)

        rb2_week5 = patched.filter((pl.col("player_id") == RB2) & (pl.col("week") == 5)).row(
            0, named=True
        )
        assert rb2_week5["teammate_vacated_target_share"] == pytest.approx(0.5)

        rb2_week1 = patched.filter((pl.col("player_id") == RB2) & (pl.col("week") == 1)).row(
            0, named=True
        )
        assert rb2_week1["teammate_vacated_target_share"] == pytest.approx(0.0)

    def test_a_missing_team_week_returns_features_unchanged(self) -> None:
        features = _features_fixture()
        empty_context = pl.DataFrame(
            schema={
                "team": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "teammate_vacated_target_share": pl.Float64,
                "teammate_vacated_carry_share": pl.Float64,
            }
        )

        result = prop.patch_vacated_shares(features, empty_context, team=TEAM, season=2025, week=5)

        assert result.equals(features)


class TestAffectedTeammates:
    def test_excludes_the_ruled_out_player_themself(self) -> None:
        features = _features_fixture()

        teammates = prop.affected_teammates(
            features, ruled_out_player_id=RB1, team=TEAM, season=2025, week=5
        )

        assert RB1 not in teammates["player_id"].to_list()
        assert RB2 in teammates["player_id"].to_list()


class TestPropagateRuledOutPlayer:
    def test_the_handcuffs_real_recomputed_projection_beats_its_own_original(self) -> None:
        features = _features_fixture()
        now = datetime(2025, 10, 1, tzinfo=UTC)

        # Baseline: RB2's own real projection with nobody marked out.
        from ffapp.models.predict import project_week

        baseline = project_week(
            features,
            2025,
            5,
            train_start=2025,
            min_train_rows=1,
            lightgbm_params=_FAST_PARAMS,
            code_version="test",
            now=now,
        )
        baseline_rb2_mean = baseline.filter(pl.col("player_id") == RB2).row(0, named=True)["mean"]

        result = prop.propagate_ruled_out_player(
            features,
            _team_week_context_fixture(),
            _empty_injuries(),
            ruled_out_player_id=RB1,
            ruled_out_position="RB",
            team=TEAM,
            season=2025,
            week=5,
            train_start=2025,
            min_train_rows=1,
            lightgbm_params=_FAST_PARAMS,
            code_version="test",
            now=now,
        )

        rb2_recomputed_mean = result.reprojections.filter(pl.col("player_id") == RB2).row(
            0, named=True
        )["mean"]
        assert rb2_recomputed_mean > baseline_rb2_mean

    def test_identifies_the_real_same_position_handcuff(self) -> None:
        features = _features_fixture()
        now = datetime(2025, 10, 1, tzinfo=UTC)

        result = prop.propagate_ruled_out_player(
            features,
            _team_week_context_fixture(),
            _empty_injuries(),
            ruled_out_player_id=RB1,
            ruled_out_position="RB",
            team=TEAM,
            season=2025,
            week=5,
            train_start=2025,
            min_train_rows=1,
            lightgbm_params=_FAST_PARAMS,
            code_version="test",
            now=now,
        )

        assert result.handcuff_player_id == RB2
        assert result.handcuff_projection_ppg is not None
        assert result.handcuff_projection_ppg > 0

    def test_no_same_position_teammate_returns_no_handcuff(self) -> None:
        features = _features_fixture().filter(pl.col("player_id") != RB2)
        now = datetime(2025, 10, 1, tzinfo=UTC)

        result = prop.propagate_ruled_out_player(
            features,
            _team_week_context_fixture(),
            _empty_injuries(),
            ruled_out_player_id=RB1,
            ruled_out_position="RB",
            team=TEAM,
            season=2025,
            week=5,
            train_start=2025,
            min_train_rows=1,
            lightgbm_params=_FAST_PARAMS,
            code_version="test",
            now=now,
        )

        assert result.handcuff_player_id is None
        assert result.handcuff_projection_ppg is None

    def test_patched_features_carries_the_recomputed_share_forward(self) -> None:
        features = _features_fixture()
        now = datetime(2025, 10, 1, tzinfo=UTC)

        result = prop.propagate_ruled_out_player(
            features,
            _team_week_context_fixture(),
            _empty_injuries(),
            ruled_out_player_id=RB1,
            ruled_out_position="RB",
            team=TEAM,
            season=2025,
            week=5,
            train_start=2025,
            min_train_rows=1,
            lightgbm_params=_FAST_PARAMS,
            code_version="test",
            now=now,
        )

        rb2_row = result.patched_features.filter(
            (pl.col("player_id") == RB2) & (pl.col("week") == 5)
        ).row(0, named=True)
        assert rb2_row["teammate_vacated_target_share"] == pytest.approx(0.5)
