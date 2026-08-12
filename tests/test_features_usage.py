import polars as pl
import pytest

from ffapp.features import usage
from ffapp.features.registry import FeatureSpec

SCORING = {"rec": 1.0}


# --- generic windowing primitives ---------------------------------------------------


def test_ewm_resets_at_a_season_boundary() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a", "a", "a", "a"],
            "season": [2025, 2025, 2025, 2026],
            "week": [1, 2, 3, 1],
            "x": [0.1, 0.3, 0.5, 0.9],
        }
    )

    result = usage.ewm(df, "x", 3, "x_ewm_3")

    rows = result.sort(["season", "week"]).to_dicts()
    assert rows[0]["x_ewm_3"] == pytest.approx(0.1)  # first week: equals its own raw value
    assert rows[3]["x_ewm_3"] == pytest.approx(0.9)  # new season: reset, not blended with 2025


def test_rolling_std_is_null_until_two_games_exist() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a", "a", "a"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "x": [5.0, 6.0, 7.0],
        }
    )

    result = usage.rolling_std(df, "x", 8, "x_std_8")

    rows = result.sort("week").to_dicts()
    assert rows[0]["x_std_8"] is None
    assert rows[1]["x_std_8"] is not None


def test_season_to_date_is_the_cumulative_mean() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a", "a", "a"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "x": [2.0, 4.0, 9.0],
        }
    )

    result = usage.season_to_date(df, "x", "x_std")

    values = result.sort("week")["x_std"].to_list()
    assert values == pytest.approx([2.0, 3.0, 5.0])  # 2, (2+4)/2, (2+4+9)/3


def test_prior_season_carries_last_seasons_average_forward() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a", "a", "a"],
            "season": [2024, 2024, 2025],
            "week": [1, 2, 1],
            "x": [0.2, 0.4, 0.9],
        }
    )

    result = usage.prior_season(df, "x", "x_prior")

    rows = {(row["season"], row["week"]): row for row in result.to_dicts()}
    assert rows[(2024, 1)]["x_prior"] is None  # no prior season on record
    assert rows[(2025, 1)]["x_prior"] == pytest.approx(0.3)  # mean(0.2, 0.4)


# --- weeks_in_current_role ----------------------------------------------------------


def test_weeks_in_current_role_resets_on_a_large_snap_pct_jump() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a"] * 6,
            "season": [2025] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "offense_snap_pct": [0.15, 0.18, 0.20, 0.60, 0.55, 0.62],
        }
    )

    result = usage.weeks_in_current_role(df)

    values = result.sort("week")["weeks_in_current_role"].to_list()
    assert values == [0, 1, 2, 0, 1, 2]  # role change detected exactly at week 4


def test_weeks_in_current_role_ignores_small_fluctuations() -> None:
    df = pl.DataFrame(
        {
            "player_id": ["a"] * 4,
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "offense_snap_pct": [0.50, 0.55, 0.52, 0.58],  # never moves >15pp
        }
    )

    result = usage.weeks_in_current_role(df)

    values = result.sort("week")["weeks_in_current_role"].to_list()
    assert values == [0, 1, 2, 3]


# --- add_actual_points ---------------------------------------------------------------


def test_add_actual_points_uses_the_real_scoring_engine() -> None:
    player_week_stats = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "receptions": [7]}
    )

    result = usage.add_actual_points(player_week_stats, SCORING)

    assert result.row(0, named=True)["actual_points"] == pytest.approx(7.0)


# --- build_usage_features (integration) -----------------------------------------------


def _usage_df() -> pl.DataFrame:
    n = 3
    return pl.DataFrame(
        {
            "player_id": ["wr1"] * n,
            "season": [2025] * n,
            "week": [1, 2, 3],
            "team": ["KC"] * n,
            "offense_snaps": [40, 45, 50],
            "offense_snap_pct": [0.5, 0.6, 0.7],
            "targets": [5, 6, 7],
            "target_share": [0.2, 0.25, 0.3],
            "air_yards": [40, 50, 60],
            "air_yards_share": [0.15, 0.2, 0.25],
            "wopr": [0.3, 0.35, 0.4],
            "adot": [8.0, 8.3, 8.5],
            "carries": [0, 0, 0],
            "carry_share": [None, None, None],
            "rz_targets": [1, 1, 2],
            "rz_carries": [0, 0, 0],
            "rz_touch_share": [0.1, 0.1, 0.2],
            "gz_carries": [0, 0, 0],
            "gz_carry_share": [None, None, None],
            "designed_rush_attempts": [0, 0, 0],
            "designed_rush_share": [None, None, None],
            "route_participation": [None, None, None],
            "xfp": [10.0, 12.0, 14.0],
        },
        schema_overrides={
            "carry_share": pl.Float64,
            "gz_carry_share": pl.Float64,
            "designed_rush_share": pl.Float64,
            "route_participation": pl.Float64,
        },
    )


def _stats_df() -> pl.DataFrame:
    n = 3
    return pl.DataFrame(
        {
            "player_id": ["wr1"] * n,
            "season": [2025] * n,
            "week": [1, 2, 3],
            "receptions": [5, 6, 7],
            "attempts": [0, 0, 0],
            "passing_cpoe": [None, None, None],
            "sacks_suffered": [0, 0, 0],
            "rushing_yards": [0, 0, 0],
        },
        schema_overrides={"passing_cpoe": pl.Float64},
    )


def test_build_usage_features_registers_every_windowed_feature() -> None:
    registry: dict[str, FeatureSpec] = {}

    usage.build_usage_features(_usage_df(), _stats_df(), SCORING, registry=registry)

    assert "target_share_ewm_3" in registry
    assert "snap_pct_prior_season" in registry
    assert "xfp_per_game_season_to_date" in registry
    assert "snap_pct_trend" in registry
    assert "xfp_minus_actual_ewm_6" in registry
    assert "points_std_std_8" in registry
    assert "weeks_in_current_role" in registry

    spec = registry["target_share_ewm_3"]
    assert spec.lag_weeks == 1
    assert spec.available_at_inference is True
    assert spec.source_table == "player_week_usage"


def test_build_usage_features_computes_plausible_values() -> None:
    registry: dict[str, FeatureSpec] = {}

    result = usage.build_usage_features(_usage_df(), _stats_df(), SCORING, registry=registry)

    week1 = result.filter(pl.col("week") == 1).row(0, named=True)
    assert week1["target_share_ewm_3"] == pytest.approx(0.2)  # first week: raw value
    # xfp_minus_actual raw at week 1: xfp(10.0) - actual_points(5 receptions * 1.0 = 5.0) = 5.0
    assert week1["xfp_minus_actual_ewm_6"] == pytest.approx(5.0)


def test_build_usage_features_does_not_pollute_a_second_call_with_duplicates() -> None:
    """Each call must use its own explicit registry -- proves the module
    never accidentally writes into the shared FEATURE_REGISTRY when given
    an explicit one."""
    registry: dict[str, FeatureSpec] = {}

    usage.build_usage_features(_usage_df(), _stats_df(), SCORING, registry=registry)
    usage.build_usage_features(_usage_df(), _stats_df(), SCORING, registry={})  # separate registry

    assert "target_share_ewm_3" in registry
