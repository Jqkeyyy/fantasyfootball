"""Strength of schedule / schedule grid logic (SPEC.md §14.5; task 2.8)."""

from __future__ import annotations

import polars as pl
import pytest

from ffapp.tools import sos


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "season_type": "REG",
        "home_team": "KC",
        "away_team": "BAL",
    }
    row.update(kwargs)
    return row


def _dpa_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "defteam": "BAL",
        "season": 2025,
        "week": 1,
        "position_group": "WR",
        "adj_epa_allowed": -0.1,
        "adj_success_allowed": -0.02,
        "adj_ypt_allowed": -0.5,
        "adj_td_rate_allowed": -0.01,
        "n_plays": 120,
    }
    row.update(kwargs)
    return row


class TestRealWeeks:
    def test_full_season_weeks_excludes_real_postseason(self) -> None:
        schedule = pl.DataFrame(
            [
                _schedule_row(week=1, season_type="REG"),
                _schedule_row(week=17, season_type="REG"),
                _schedule_row(week=19, season_type="WC"),
                _schedule_row(week=22, season_type="SB"),
            ]
        )

        weeks = sos.full_season_weeks(schedule, season=2025)

        assert weeks == [1, 17]

    def test_rest_of_season_weeks_is_strictly_after_as_of(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=w) for w in range(1, 6)])

        weeks = sos.rest_of_season_weeks(schedule, season=2025, as_of_week=3)

        assert weeks == [4, 5]

    def test_playoff_weeks_is_playoff_start_through_17_literally(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=w) for w in range(1, 18)])

        weeks = sos.playoff_weeks(schedule, season=2025, playoff_week_start=15)

        assert weeks == [15, 16, 17]

    def test_playoff_weeks_stops_at_17_even_on_an_18_week_schedule(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=w) for w in range(1, 19)])

        weeks = sos.playoff_weeks(schedule, season=2025, playoff_week_start=15)

        assert weeks == [15, 16, 17]

    def test_playoff_weeks_empty_when_start_is_unset(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=w) for w in range(1, 18)])

        weeks = sos.playoff_weeks(schedule, season=2025, playoff_week_start=0)

        assert weeks == []


class TestTeamPositionGroupSchedule:
    def test_maps_each_team_to_its_real_opponents_adjusted_rate(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=1, home_team="KC", away_team="BAL")])
        dpa = pl.DataFrame([_dpa_row(defteam="BAL", week=1, position_group="WR")])

        result = sos.team_position_group_schedule(dpa, schedule, season=2025, position_group="WR")

        row = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert row["opponent"] == "BAL"
        assert row["adj_epa_allowed"] == pytest.approx(-0.1)
        assert row["n_plays"] == 120

    def test_relocated_franchise_still_joins_via_the_alias(self) -> None:
        # 2015 Rams: schedule says "STL", defense_position_allowed (built
        # from pbp, backfilled to current codes) says "LA" -- unaliased,
        # this join would silently drop the row (CLAUDE.md rule 4).
        schedule = pl.DataFrame(
            [_schedule_row(season=2015, week=1, home_team="ARI", away_team="STL")]
        )
        dpa = pl.DataFrame(
            [_dpa_row(defteam="LA", season=2015, week=1, position_group="WR", adj_epa_allowed=0.2)]
        )

        result = sos.team_position_group_schedule(dpa, schedule, season=2015, position_group="WR")

        row = result.filter(pl.col("team") == "ARI").row(0, named=True)
        assert row["opponent"] == "STL"
        assert row["adj_epa_allowed"] == pytest.approx(0.2)

    def test_a_bye_week_has_no_row(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=1, home_team="KC", away_team="BAL")])
        dpa = pl.DataFrame([_dpa_row(defteam="BAL", week=1, position_group="WR")])

        result = sos.team_position_group_schedule(dpa, schedule, season=2025, position_group="WR")

        # a third team with no game at all this week never appears
        assert set(result["team"].to_list()) == {"KC", "BAL"}

    def _n_plays_population(self) -> pl.DataFrame:
        # 25th percentile of [10, 20, 30, 40, 50] is 20 (linear interp.) --
        # a real, non-trivial cutoff computed from this group's own
        # distribution, not a fixed cross-position constant.
        return pl.DataFrame(
            [
                _dpa_row(defteam=f"D{i}", week=i, position_group="WR", n_plays=n)
                for i, n in enumerate([10, 20, 30, 40, 50], start=1)
            ]
        )

    def test_below_the_groups_own_quartile_is_flagged_not_confident(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=1, home_team="KC", away_team="D1")])
        dpa = self._n_plays_population()

        result = sos.team_position_group_schedule(dpa, schedule, season=2025, position_group="WR")

        row = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert row["n_plays"] == 10
        assert row["confident"] is False

    def test_above_the_groups_own_quartile_is_flagged_confident(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=5, home_team="KC", away_team="D5")])
        dpa = self._n_plays_population()

        result = sos.team_position_group_schedule(dpa, schedule, season=2025, position_group="WR")

        row = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert row["n_plays"] == 50
        assert row["confident"] is True

    def test_the_same_n_plays_can_be_confident_in_one_group_and_not_another(self) -> None:
        # A thin-volume group (e.g. real QB_rushing weekly counts run in
        # the single digits) must not share a low-volume group's own
        # threshold with a high-volume one (e.g. real WR weekly counts) --
        # the whole point of a per-group threshold is that it self-scales.
        schedule = pl.DataFrame([_schedule_row(week=1, home_team="KC", away_team="BAL")])
        thin_group = pl.DataFrame(
            [
                _dpa_row(defteam=f"D{i}", week=i, position_group="QB_rushing", n_plays=n)
                for i, n in enumerate([1, 2, 3], start=1)
            ]
            + [_dpa_row(defteam="BAL", week=1, position_group="QB_rushing", n_plays=5)]
        )
        thick_group = pl.DataFrame(
            [
                _dpa_row(defteam=f"D{i}", week=i, position_group="WR", n_plays=n)
                for i, n in enumerate([30, 40, 50], start=1)
            ]
            + [_dpa_row(defteam="BAL", week=1, position_group="WR", n_plays=5)]
        )

        thin_result = sos.team_position_group_schedule(
            thin_group, schedule, season=2025, position_group="QB_rushing"
        )
        thick_result = sos.team_position_group_schedule(
            thick_group, schedule, season=2025, position_group="WR"
        )

        assert thin_result.filter(pl.col("team") == "KC").row(0, named=True)["confident"] is True
        assert thick_result.filter(pl.col("team") == "KC").row(0, named=True)["confident"] is False

    def test_missing_defense_data_is_honestly_null_not_zero(self) -> None:
        schedule = pl.DataFrame([_schedule_row(week=1, home_team="KC", away_team="BAL")])
        dpa = pl.DataFrame(
            [_dpa_row(defteam="BAL", week=1, position_group="TE")]
        )  # no WR row for BAL

        result = sos.team_position_group_schedule(dpa, schedule, season=2025, position_group="WR")

        row = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert row["adj_epa_allowed"] is None
        assert row["confident"] is False


class TestPositionalSos:
    def _team_schedule(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "team": ["KC", "KC", "KC", "BUF", "BUF"],
                "week": [1, 2, 3, 1, 2],
                "opponent": ["BAL", "LAC", "DEN", "NYJ", "MIA"],
                "adj_epa_allowed": [0.1, 0.2, -0.1, -0.3, -0.2],
                "n_plays": [300, 300, 300, 300, 300],
                "confident": [True, True, True, True, True],
            }
        )

    def test_sums_adj_epa_allowed_across_the_given_weeks(self) -> None:
        result = sos.positional_sos(self._team_schedule(), weeks=[1, 2, 3])

        kc = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert kc["sos_value"] == pytest.approx(0.2)
        assert kc["n_weeks_included"] == 3

    def test_a_narrower_week_range_only_sums_those_weeks(self) -> None:
        result = sos.positional_sos(self._team_schedule(), weeks=[1])

        kc = result.filter(pl.col("team") == "KC").row(0, named=True)
        assert kc["sos_value"] == pytest.approx(0.1)
        assert kc["n_weeks_included"] == 1

    def test_higher_sos_value_means_easier_schedule_sign_convention(self) -> None:
        # BUF's opponents allow *less* EPA (tougher matchups) than KC's --
        # BUF's sos_value must be lower, not inverted into a "difficulty" score.
        result = sos.positional_sos(self._team_schedule(), weeks=[1, 2])

        by_team = {r["team"]: r["sos_value"] for r in result.to_dicts()}
        assert by_team["KC"] > by_team["BUF"]

    def test_a_bye_contributes_nothing_not_a_guessed_average(self) -> None:
        # BUF has no week-3 row at all (a real bye) -- summing weeks [1,2,3]
        # for BUF must equal summing just [1,2].
        result_with_bye = sos.positional_sos(self._team_schedule(), weeks=[1, 2, 3])
        result_without = sos.positional_sos(self._team_schedule(), weeks=[1, 2])

        buf_with = result_with_bye.filter(pl.col("team") == "BUF").row(0, named=True)
        buf_without = result_without.filter(pl.col("team") == "BUF").row(0, named=True)
        assert buf_with["sos_value"] == pytest.approx(buf_without["sos_value"])
        assert buf_with["n_weeks_included"] == 2


class TestScheduleGrid:
    def test_pivots_to_teams_by_weeks(self) -> None:
        team_schedule = pl.DataFrame(
            {
                "team": ["KC", "KC", "BUF"],
                "week": [1, 2, 1],
                "opponent": ["BAL", "LAC", "NYJ"],
                "adj_epa_allowed": [0.1, 0.2, -0.3],
                "n_plays": [300, 300, 300],
                "confident": [True, True, True],
            }
        )

        grid = sos.schedule_grid(team_schedule)

        assert set(grid.columns) == {"team", "1", "2"}
        kc = grid.filter(pl.col("team") == "KC").row(0, named=True)
        assert kc["1"] == pytest.approx(0.1)
        assert kc["2"] == pytest.approx(0.2)

    def test_a_bye_pivots_to_a_null_cell_not_zero(self) -> None:
        # KC has a real week-2 game; BUF's week 2 is a real bye -- no row
        # for BUF at week 2 at all, so its cell must pivot to null.
        team_schedule = pl.DataFrame(
            {
                "team": ["KC", "KC", "BUF"],
                "week": [1, 2, 1],
                "opponent": ["BAL", "LAC", "NYJ"],
                "adj_epa_allowed": [0.1, 0.2, -0.3],
                "n_plays": [300, 300, 300],
                "confident": [True, True, True],
            }
        )

        grid = sos.schedule_grid(team_schedule)

        buf = grid.filter(pl.col("team") == "BUF").row(0, named=True)
        assert buf["2"] is None  # BUF has no real week-2 game in this fixture


class TestPositionGroupConfidenceThresholds:
    def test_computes_a_separate_threshold_per_real_group(self) -> None:
        dpa = pl.DataFrame(
            [
                _dpa_row(defteam=f"D{i}", week=i, position_group="WR", n_plays=n)
                for i, n in enumerate([10, 20, 30, 40, 50], start=1)
            ]
            + [
                _dpa_row(defteam=f"D{i}", week=i, position_group="QB_rushing", n_plays=n)
                for i, n in enumerate([1, 2, 3, 4, 5], start=1)
            ]
        )

        thresholds = sos.position_group_confidence_thresholds(dpa, season=2025)

        assert thresholds["WR"] == pytest.approx(20.0)
        assert thresholds["QB_rushing"] == pytest.approx(2.0)

    def test_only_scopes_to_the_given_season(self) -> None:
        dpa = pl.DataFrame(
            [_dpa_row(defteam="A", season=2024, week=1, position_group="WR", n_plays=5)]
            + [
                _dpa_row(defteam=f"D{i}", season=2025, week=i, position_group="WR", n_plays=n)
                for i, n in enumerate([10, 20, 30, 40, 50], start=1)
            ]
        )

        thresholds = sos.position_group_confidence_thresholds(dpa, season=2025)

        assert thresholds["WR"] == pytest.approx(20.0)  # the 2024 row must not shift this


class TestMatchupDetail:
    def test_wr_gets_one_group(self) -> None:
        row = {
            "def_adj_epa_allowed_wr": -0.1,
            "def_adj_success_allowed_wr": -0.02,
            "def_adj_ypt_allowed_wr": -0.5,
            "def_adj_td_rate_allowed_wr": -0.01,
            "def_n_plays_wr": 30,
        }

        detail = sos.matchup_detail(row, "WR", confidence_thresholds={"WR": 20.0})

        assert len(detail) == 1
        assert detail[0]["position_group"] == "WR"
        assert detail[0]["adj_epa_allowed"] == pytest.approx(-0.1)
        assert detail[0]["confident"] is True

    def test_no_thresholds_given_defaults_to_permissive(self) -> None:
        # No `confidence_thresholds` passed at all -- any real recorded
        # sample is trusted rather than crashing or guessing a cutoff.
        row = {"def_adj_epa_allowed_wr": -0.1, "def_n_plays_wr": 1}

        detail = sos.matchup_detail(row, "WR")

        assert detail[0]["confident"] is True

    def test_rb_gets_both_receiving_and_rushing_groups(self) -> None:
        row = {
            "def_adj_epa_allowed_rb_receiving": -0.05,
            "def_n_plays_rb_receiving": 40,
            "def_adj_epa_allowed_rb_rushing": 0.08,
            "def_n_plays_rb_rushing": 160,
        }

        detail = sos.matchup_detail(row, "RB")

        groups = {d["position_group"] for d in detail}
        assert groups == {"RB_receiving", "RB_rushing"}

    def test_low_n_plays_group_is_not_confident(self) -> None:
        row = {"def_adj_epa_allowed_te": 0.02, "def_n_plays_te": 5}

        detail = sos.matchup_detail(row, "TE", confidence_thresholds={"TE": 20.0})

        assert detail[0]["confident"] is False

    def test_a_group_missing_from_thresholds_falls_back_to_permissive(self) -> None:
        row = {"def_adj_epa_allowed_te": 0.02, "def_n_plays_te": 5}

        detail = sos.matchup_detail(row, "TE", confidence_thresholds={"WR": 20.0})

        assert detail[0]["confident"] is True

    def test_missing_data_is_null_not_a_crash(self) -> None:
        row: dict[str, object] = {}

        detail = sos.matchup_detail(row, "TE")

        assert detail[0]["adj_epa_allowed"] is None
        assert detail[0]["n_plays"] is None
        assert detail[0]["confident"] is False
