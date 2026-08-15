from datetime import date

import polars as pl
import pytest

from ffapp.features import build
from ffapp.features.build import LeakageError, assert_inference_availability, assert_training_lag
from ffapp.features.registry import FeatureSpec

SCORING = {"rec": 1.0}


def _spec(**kwargs: object) -> FeatureSpec:
    fields: dict[str, object] = {
        "name": "target_share_ewm_4",
        "description": "targets / team pass attempts, ewm span 4",
        "positions": ["WR", "TE", "RB"],
        "window": "ewm_4",
        "source_table": "player_week_usage",
        "available_at_inference": True,
        "lag_weeks": 1,
    }
    fields.update(kwargs)
    return FeatureSpec(**fields)


# --- assert_training_lag ------------------------------------------------------------


def test_assert_training_lag_passes_for_a_correctly_lagged_feature() -> None:
    assert_training_lag([_spec(lag_weeks=1)])  # should not raise


def test_assert_training_lag_raises_for_a_zero_lag_feature() -> None:
    """The deliberately mis-specified feature TASKS.md 1.5 requires: a
    feature with lag_weeks=0 would see the target week's own data."""
    with pytest.raises(LeakageError, match="lag_weeks"):
        assert_training_lag([_spec(name="leaky_feature", lag_weeks=0)])


def test_assert_training_lag_raises_for_negative_lag() -> None:
    with pytest.raises(LeakageError):
        assert_training_lag([_spec(name="leaky_feature", lag_weeks=-1)])


def test_assert_training_lag_checks_every_spec_not_just_the_first() -> None:
    with pytest.raises(LeakageError, match="second_feature"):
        assert_training_lag(
            [_spec(name="first_feature", lag_weeks=1), _spec(name="second_feature", lag_weeks=0)]
        )


# --- assert_inference_availability ---------------------------------------------------


def test_assert_inference_availability_passes_for_an_available_feature() -> None:
    assert_inference_availability([_spec(available_at_inference=True)])  # should not raise


def test_assert_inference_availability_raises_for_a_training_only_feature() -> None:
    """The deliberately mis-specified feature TASKS.md 1.5 requires: a
    training-only feature (e.g. route participation, SPEC §10.5) must
    never be handed to an inference model."""
    with pytest.raises(LeakageError, match="available_at_inference"):
        assert_inference_availability(
            [_spec(name="route_participation", available_at_inference=False)]
        )


def test_assert_inference_availability_checks_every_spec_not_just_the_first() -> None:
    with pytest.raises(LeakageError, match="second_feature"):
        assert_inference_availability(
            [
                _spec(name="first_feature", available_at_inference=True),
                _spec(name="second_feature", available_at_inference=False),
            ]
        )


# --- _row_grid (task 1.9) ---------------------------------------------------------------


def _roster_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "gsis_id": "p1",
        "season": 2025,
        "week": 1,
        "team": "KC",
        "position": "WR",
        "status": "ACT",
        "birth_date": date(1995, 1, 1),
    }
    row.update(kwargs)
    return row


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "game_id": "2025_01_KC_BAL",
        "season": 2025,
        "week": 1,
        "home_team": "KC",
        "away_team": "BAL",
        "kickoff_utc": "2025-09-07T17:00:00Z",
        "weekday": "Sunday",
        "gametime": "13:00",
        "home_rest": 7,
        "away_rest": 7,
    }
    row.update(kwargs)
    return row


def test_row_grid_keeps_only_active_skill_position_players_with_a_real_id() -> None:
    rosters = pl.DataFrame(
        [
            _roster_row(gsis_id="p1", status="ACT", position="WR"),
            _roster_row(gsis_id="p2", status="CUT", position="WR"),  # not active
            _roster_row(gsis_id="p3", status="ACT", position="LB"),  # not a skill position
            _roster_row(gsis_id=None, status="ACT", position="WR"),  # no real id
        ]
    )
    schedule = pl.DataFrame([_schedule_row()])

    result = build._row_grid(rosters, schedule)

    assert result["player_id"].to_list() == ["p1"]


def test_row_grid_keeps_a_game_day_inactive_player_not_just_act() -> None:
    """Real bug found by end-to-end verification: a player ruled `Out`
    for a game (real roster status `"INA"`) is still on the active
    53-man roster -- excluding them would silently reproduce exactly the
    survivorship bias SPEC §11.1 says this row universe exists to avoid.
    `"DEV"` (practice squad) and `"RES"` (injured reserve) are genuinely
    off the active roster and must stay excluded."""
    rosters = pl.DataFrame(
        [
            _roster_row(gsis_id="p1", status="INA", position="WR"),  # active, game-day inactive
            _roster_row(gsis_id="p2", status="DEV", position="WR"),  # practice squad
            _roster_row(gsis_id="p3", status="RES", position="WR"),  # injured reserve
        ]
    )
    schedule = pl.DataFrame([_schedule_row()])

    result = build._row_grid(rosters, schedule)

    assert result["player_id"].to_list() == ["p1"]


def test_row_grid_drops_a_bye_week() -> None:
    rosters = pl.DataFrame([_roster_row(gsis_id="p1", team="DAL")])  # DAL has no game
    schedule = pl.DataFrame([_schedule_row(home_team="KC", away_team="BAL")])

    result = build._row_grid(rosters, schedule)

    assert result.height == 0


# --- _add_target_and_availability (task 1.9) --------------------------------------------


def test_add_target_and_availability_scores_a_real_stat_line() -> None:
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1]})
    stats = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1], "receptions": [7]})
    usage_df = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "offense_snaps": [40]}
    )

    result = build._add_target_and_availability(grid, stats, usage_df, SCORING)

    row = result.row(0, named=True)
    assert row["target"] == pytest.approx(7.0)
    assert row["availability_flag"] is True


def test_add_target_and_availability_is_zero_and_false_for_a_player_with_no_stats() -> None:
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1]})
    stats = pl.DataFrame(
        {"player_id": [], "season": [], "week": [], "receptions": []},
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "receptions": pl.Int64,
        },
    )
    usage_df = pl.DataFrame(
        {"player_id": [], "season": [], "week": [], "offense_snaps": []},
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Int64,
        },
    )

    result = build._add_target_and_availability(grid, stats, usage_df, SCORING)

    row = result.row(0, named=True)
    assert row["target"] == pytest.approx(0.0)
    assert row["availability_flag"] is False


def test_add_target_and_availability_is_false_when_zero_snaps_recorded() -> None:
    """Distinct from "no row at all" -- a player who dressed but recorded
    zero offensive snaps (e.g. a healthy scratch who still has a usage
    row via some other source) is still unavailable."""
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1]})
    stats = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1], "receptions": [0]})
    usage_df = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "offense_snaps": [0]}
    )

    result = build._add_target_and_availability(grid, stats, usage_df, SCORING)

    assert result.row(0, named=True)["availability_flag"] is False


# --- _add_as_of_utc (task 1.9) -----------------------------------------------------------


def test_add_as_of_utc_uses_the_earliest_kickoff_that_week() -> None:
    grid = pl.DataFrame({"season": [2025], "week": [1]})
    schedule = pl.DataFrame(
        [
            _schedule_row(kickoff_utc="2025-09-07T20:00:00Z"),
            _schedule_row(home_team="DAL", away_team="NYG", kickoff_utc="2025-09-07T17:00:00Z"),
        ]
    )

    result = build._add_as_of_utc(grid, schedule)

    assert result.row(0, named=True)["as_of_utc"] == "2025-09-07T17:00:00Z"


# --- lag_shift_join (task 2: promotion to public) ------------------------------------------


def test_lag_shift_join_is_public() -> None:
    """Task 2: lag_shift_join must be promoted from private _lag_shift_join to
    public so models.team_environment can use it."""
    assert hasattr(build, "lag_shift_join")


# --- _lag_shift_join (task 1.9) -----------------------------------------------------------


def test_lag_shift_join_pulls_the_prior_weeks_row_onto_the_target_week() -> None:
    grid = pl.DataFrame({"player_id": ["p1", "p1"], "season": [2025, 2025], "week": [2, 3]})
    feature_table = pl.DataFrame(
        {"player_id": ["p1", "p1"], "season": [2025, 2025], "week": [1, 2], "x": [10.0, 20.0]}
    )

    result = build.lag_shift_join(grid, feature_table, "player_id", ["x"])

    rows = {row["week"]: row["x"] for row in result.iter_rows(named=True)}
    assert rows[2] == pytest.approx(10.0)  # week 2's row got week 1's value
    assert rows[3] == pytest.approx(20.0)  # week 3's row got week 2's value


# --- build_player_week_features (integration, task 1.9) ---------------------------------


def _usage_df(player_id: str, team: str, weeks: list[int]) -> pl.DataFrame:
    n = len(weeks)
    return pl.DataFrame(
        {
            "player_id": [player_id] * n,
            "season": [2025] * n,
            "week": weeks,
            "team": [team] * n,
            "offense_snaps": [40] * n,
            "offense_snap_pct": [0.5] * n,
            "targets": [5] * n,
            "target_share": [0.2 + 0.1 * i for i in range(n)],
            "air_yards": [40] * n,
            "air_yards_share": [0.15] * n,
            "wopr": [0.3] * n,
            "adot": [8.0] * n,
            "carries": [0] * n,
            "carry_share": [None] * n,
            "rz_targets": [1] * n,
            "rz_carries": [0] * n,
            "rz_touch_share": [0.1] * n,
            "gz_carries": [0] * n,
            "gz_carry_share": [None] * n,
            "designed_rush_attempts": [0] * n,
            "designed_rush_share": [None] * n,
            "route_participation": [None] * n,
            "xfp": [10.0] * n,
        },
        schema_overrides={
            "carry_share": pl.Float64,
            "gz_carry_share": pl.Float64,
            "designed_rush_share": pl.Float64,
            "route_participation": pl.Float64,
        },
    )


def _stats_df(player_id: str, weeks: list[int]) -> pl.DataFrame:
    n = len(weeks)
    return pl.DataFrame(
        {
            "player_id": [player_id] * n,
            "season": [2025] * n,
            "week": weeks,
            "receptions": [5] * n,
            "attempts": [0] * n,
            "passing_cpoe": [None] * n,
            "sacks_suffered": [0] * n,
            "rushing_yards": [0] * n,
        },
        schema_overrides={"passing_cpoe": pl.Float64},
    )


def test_build_player_week_features_lag_shifts_usage_but_not_opponent_or_situation() -> None:
    """The central correctness property of task 1.9: a usage feature at
    the target week must come from the *prior* week's trailing value
    (never the target week's own outcome), while opponent/situation
    features describe the target week's own game directly."""
    weeks = [1, 2]
    rosters = pl.DataFrame(
        [_roster_row(gsis_id="p1", team="KC", position="WR", week=w) for w in weeks]
    )
    schedule = pl.DataFrame([_schedule_row(week=w, home_team="KC", away_team="BAL") for w in weeks])
    player_week_stats = _stats_df("p1", weeks)
    player_week_usage = _usage_df("p1", "KC", weeks)
    snap_counts = pl.DataFrame(
        {
            "season": [],
            "week": [],
            "team": [],
            "pfr_player_id": [],
            "position": [],
            "offense_snaps": [],
        },
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "pfr_player_id": pl.Utf8,
            "position": pl.Utf8,
            "offense_snaps": pl.Int64,
        },
    )
    team_week_context = pl.DataFrame(
        {
            "team": ["KC"] * len(weeks),
            "season": [2025] * len(weeks),
            "week": weeks,
            "plays": [60.0, 64.0],
            "pass_rate": [0.55, 0.60],
            "neutral_pace_sec": [28.0] * len(weeks),
            "proe": [0.0] * len(weeks),
            "epa_per_play_off": [0.1] * len(weeks),
            "success_rate_off": [0.48] * len(weeks),
            "implied_total": [24.0, 27.5],  # varies by week -- tests the no-shift fix
            "spread": [-3.0] * len(weeks),
        }
    )
    defense_position_allowed = pl.DataFrame(
        {
            "defteam": ["BAL"] * len(weeks),
            "season": [2025] * len(weeks),
            "week": weeks,
            "position_group": ["WR"] * len(weeks),
            "adj_epa_allowed": [-0.05, -0.09],
            "adj_success_allowed": [-0.01] * len(weeks),
            "adj_ypt_allowed": [-0.3] * len(weeks),
            "adj_td_rate_allowed": [-0.01] * len(weeks),
            "n_plays": [20, 22],
        }
    )
    injuries = pl.DataFrame(
        {
            "player_id": [],
            "season": [],
            "week": [],
            "report_status": [],
            "practice_status": [],
            "date_modified": [],
        },
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "report_status": pl.Utf8,
            "practice_status": pl.Utf8,
            "date_modified": pl.Datetime(time_zone="UTC"),
        },
    )
    weather = pl.DataFrame(
        {"game_id": [], "wind_mph": [], "precip_prob": [], "temp_f": [], "is_dome": []},
        schema={
            "game_id": pl.Utf8,
            "wind_mph": pl.Float64,
            "precip_prob": pl.Float64,
            "temp_f": pl.Float64,
            "is_dome": pl.Boolean,
        },
    )
    depth_charts = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "gsis_id": ["p1", "p1"],
            "formation": ["Offense", "Offense"],
            "depth_team": ["2", "1"],  # varies by week -- proves the no-shift join
        }
    )

    result = build.build_player_week_features(
        rosters,
        schedule,
        player_week_stats,
        player_week_usage,
        snap_counts,
        team_week_context,
        defense_position_allowed,
        injuries,
        weather,
        depth_charts,
        SCORING,
        registry={},
    )

    week2 = result.filter(pl.col("week") == 2).row(0, named=True)
    # target_share_ewm_3 at week 2 must equal week 1's own raw value
    # (0.2, from _usage_df's week-1 row) -- never week 2's own (0.3).
    assert week2["target_share_ewm_3"] == pytest.approx(0.2)
    # opponent/situation: describe week 2's own game directly, no shift.
    assert week2["def_adj_epa_allowed_wr"] == pytest.approx(-0.09)
    assert week2["is_home"] is True
    # team_context.CURRENT_WEEK_COLUMNS (real bug found by end-to-end
    # verification): implied_team_total must be week 2's own value
    # (27.5), never week 1's shifted-forward value (24.0).
    assert week2["implied_team_total"] == pytest.approx(27.5)
    # depth_chart_rank (task 1.14): week 2's own real depth slot (1), not
    # week 1's shifted-forward value (2).
    assert week2["depth_chart_rank"] == 1
    # age (task 1.14): a real, non-null fractional age from rosters' own
    # birth_date.
    assert week2["age"] is not None
    assert week2["age"] > 0
