"""Weekly rankings page logic (SPEC.md §14.1, §15; task 1.19). Pure,
pytest-testable functions only -- the real Streamlit page
(`app/pages/2_Weekly_Rankings.py`) is verified by actually running it
(CLAUDE.md's UI rule), documented in docs/JOURNAL.md, not here.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ffapp.app.weekly_rankings_page import (
    ProjectionsNotBuiltError,
    add_matchup_grade,
    build_weekly_rankings,
    filter_rankings,
    load_projections,
)


def _projections() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "season": [2025, 2025, 2025, 2025],
            "week": [10, 10, 10, 10],
            "p_active": [0.9, 0.95, 0.8, 0.99],
            "mean": [15.0, 20.0, 5.0, 10.0],
            "q10": [2.0, 5.0, 0.0, 1.0],
            "q25": [8.0, 12.0, 1.0, 4.0],
            "q50": [15.0, 20.0, 5.0, 10.0],
            "q75": [22.0, 28.0, 9.0, 16.0],
            "q90": [30.0, 35.0, 14.0, 22.0],
            "model_version": ["v1"] * 4,
            "as_of_utc": ["2025-11-01T00:00:00+00:00"] * 4,
            "feature_hash": ["h1"] * 4,
            "git_commit": ["abc123"] * 4,
        }
    )


def _features() -> pl.DataFrame:
    """p1/p2 are WR (one opponent group); p3 is RB (two groups, weighted
    combination); p4 is a different team/opponent entirely."""
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "season": [2025, 2025, 2025, 2025],
            "week": [10, 10, 10, 10],
            "team": ["AAA", "AAA", "AAA", "CCC"],
            "position": ["WR", "WR", "RB", "WR"],
            "def_adj_epa_allowed_wr": [0.10, 0.10, None, -0.20],
            "def_n_plays_wr": [120, 120, None, 150],
            "def_adj_epa_allowed_rb_receiving": [None, None, 0.05, None],
            "def_n_plays_rb_receiving": [None, None, 40, None],
            "def_adj_epa_allowed_rb_rushing": [None, None, -0.15, None],
            "def_n_plays_rb_rushing": [None, None, 160, None],
            "def_adj_epa_allowed_te": [None, None, None, None],
            "def_n_plays_te": [None, None, None, None],
            "def_adj_epa_allowed_qb_passing": [None, None, None, None],
            "def_n_plays_qb_passing": [None, None, None, None],
            "def_adj_epa_allowed_qb_rushing": [None, None, None, None],
            "def_n_plays_qb_rushing": [None, None, None, None],
        }
    )


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [10, 10],
            "home_team": ["AAA", "CCC"],
            "away_team": ["BBB", "DDD"],
        }
    )


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "full_name": ["Player One", "Player Two", "Player Three", "Player Four"],
        }
    )


class TestLoadProjections:
    def test_raises_a_named_error_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectionsNotBuiltError, match="ffapp project"):
            load_projections(tmp_path / "projections.parquet")

    def test_loads_a_real_parquet_file(self, tmp_path: Path) -> None:
        path = tmp_path / "projections.parquet"
        _projections().write_parquet(path)

        loaded = load_projections(path)

        assert loaded.height == 4


class TestAddMatchupGrade:
    def test_a_single_group_position_uses_its_own_value_and_n_plays(self) -> None:
        df = _features().filter(pl.col("player_id") == "p1")

        graded = add_matchup_grade(df)

        row = graded.to_dicts()[0]
        assert row["n_plays_behind_matchup_grade"] == 120

    def test_a_two_group_position_combines_via_an_n_plays_weighted_average(self) -> None:
        df = _features().filter(pl.col("player_id") == "p3")

        graded = add_matchup_grade(df)

        row = graded.to_dicts()[0]
        # weighted mean: (0.05*40 + -0.15*160) / (40+160) = (2 - 24) / 200 = -0.11
        assert row["n_plays_behind_matchup_grade"] == 200
        assert row["_matchup_epa"] == pytest.approx(-0.11)

    def test_grades_rank_within_the_position_cohort_best_to_worst(self) -> None:
        # p1 and p2 share the exact same real matchup value in this fixture,
        # so both should land in the same grade bucket -- not split arbitrarily.
        df = _features().filter(pl.col("position") == "WR")

        graded = add_matchup_grade(df)

        grades = dict(
            zip(graded["player_id"].to_list(), graded["matchup_grade"].to_list(), strict=True)
        )
        assert grades["p1"] == grades["p2"]

    def test_a_row_with_no_relevant_opponent_data_gets_a_null_grade_not_a_crash(self) -> None:
        df = pl.DataFrame(
            {
                "player_id": ["p9"],
                "position": ["TE"],
                "def_adj_epa_allowed_te": [None],
                "def_n_plays_te": [None],
            }
        )

        graded = add_matchup_grade(df)

        row = graded.to_dicts()[0]
        assert row["matchup_grade"] is None
        assert row["n_plays_behind_matchup_grade"] is None


class TestBuildWeeklyRankings:
    def test_output_has_every_spec_column(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        expected = {
            "player_name",
            "position",
            "team",
            "opponent",
            "p_active",
            "proj_mean",
            "floor",
            "median",
            "ceiling",
            "matchup_grade",
            "n_plays_behind_matchup_grade",
        }
        assert expected.issubset(set(result.columns))
        assert result.height == 4

    def test_opponent_is_resolved_from_the_real_schedule(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        by_player = {r["player_id"]: r["opponent"] for r in result.to_dicts()}
        assert by_player["p1"] == "BBB"  # AAA's real opponent that week
        assert by_player["p4"] == "DDD"  # CCC's real opponent that week

    def test_floor_and_ceiling_are_q10_and_q90(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        row = result.filter(pl.col("player_id") == "p1").to_dicts()[0]
        assert row["floor"] == pytest.approx(2.0)
        assert row["median"] == pytest.approx(15.0)
        assert row["ceiling"] == pytest.approx(30.0)

    def test_default_row_order_is_proj_mean_descending(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        means = result["proj_mean"].to_list()
        assert means == sorted(means, reverse=True)

    def test_owner_status_classifies_my_roster_rostered_elsewhere_and_free_agents(self) -> None:
        result = build_weekly_rankings(
            _projections(),
            _features(),
            _schedule(),
            _players_dim(),
            season=2025,
            week=10,
            my_roster_ids={"p1"},
            rostered_ids={"p1", "p2"},
        )

        by_player = {r["player_id"]: r["owner_status"] for r in result.to_dicts()}
        assert by_player["p1"] == "my_roster"
        assert by_player["p2"] == "rostered_elsewhere"
        assert by_player["p3"] == "free_agent"


class TestFilterRankings:
    def test_filters_by_position(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        filtered = filter_rankings(result, positions=["RB"])

        assert filtered["position"].to_list() == ["RB"]

    def test_filters_by_availability(self) -> None:
        result = build_weekly_rankings(
            _projections(),
            _features(),
            _schedule(),
            _players_dim(),
            season=2025,
            week=10,
            my_roster_ids={"p1"},
            rostered_ids={"p1", "p2"},
        )

        filtered = filter_rankings(result, availability="free_agent")

        assert set(filtered["player_id"].to_list()) == {"p3", "p4"}

    def test_no_filters_returns_everything(self) -> None:
        result = build_weekly_rankings(
            _projections(), _features(), _schedule(), _players_dim(), season=2025, week=10
        )

        filtered = filter_rankings(result)

        assert filtered.height == result.height
