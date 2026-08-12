"""Tests for scoring/golden.py: the SPEC §8.4 golden test comparison logic.

Only the pure functions (extraction, id resolution, comparison, summarisation) are
unit-tested with fixtures here -- the orchestrating `run_golden_test` fetches real
Sleeper matchups and nflverse stats and is verified by an actual live run (see
HANDOFF.md §2), matching how `ids.mapping.unmatched_report` is tested vs. verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ffapp.config import LeagueConfig
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.scoring import golden

FIXTURES = Path(__file__).parent / "fixtures" / "ids"
CROSSWALK_CSV = FIXTURES / "crosswalk.csv"
SLEEPER_PLAYERS_JSON = FIXTURES / "sleeper_players.json"
OVERRIDES_CSV = FIXTURES / "overrides.csv"


# --- extract_players_points -----------------------------------------------------


def test_extract_players_points_merges_across_every_roster_in_the_week() -> None:
    matchups = [
        {"roster_id": 1, "players_points": {"100": 8.84, "ARI": 5.0}},
        {"roster_id": 2, "players_points": {"200": 12.3}},
    ]

    merged = golden.extract_players_points(matchups)

    assert merged == {"100": 8.84, "ARI": 5.0, "200": 12.3}


def test_extract_players_points_empty_week_returns_empty_dict() -> None:
    assert golden.extract_players_points([]) == {}


# --- resolve_player_ids ----------------------------------------------------------


def test_resolve_player_ids_uses_team_abbreviation_directly_for_dst() -> None:
    players_dim = pl.DataFrame({"sleeper_id": ["100"], "player_id": ["00-0031234"]})

    resolved = golden.resolve_player_ids({"100", "ARI"}, players_dim, team_abbreviations={"ARI"})

    assert resolved == {"100": "00-0031234", "ARI": "ARI"}


def test_resolve_player_ids_applies_the_known_rams_team_code_alias() -> None:
    """Sleeper's DST sleeper_id for the Rams is "LAR"; nflverse's team code is
    "LA" -- the only mismatch across all 32 teams, confirmed live against both
    real datasets (see HANDOFF.md §5)."""
    players_dim = pl.DataFrame(
        {"sleeper_id": [], "player_id": []}, schema={"sleeper_id": pl.Utf8, "player_id": pl.Utf8}
    )

    resolved = golden.resolve_player_ids({"LAR"}, players_dim, team_abbreviations={"LA"})

    assert resolved == {"LAR": "LA"}


def test_resolve_player_ids_falls_back_to_sleeper_id_when_unmapped() -> None:
    """An unresolved sleeper_id (no crosswalk row at all) still gets a key -- CLAUDE.md
    rule 4: never silently drop, let the comparison step surface it as a miss."""
    players_dim = pl.DataFrame(
        {"sleeper_id": [], "player_id": []}, schema={"sleeper_id": pl.Utf8, "player_id": pl.Utf8}
    )

    resolved = golden.resolve_player_ids({"999"}, players_dim, team_abbreviations=set())

    assert resolved == {"999": "999"}


# --- compare_points ---------------------------------------------------------------


def test_compare_points_agrees_within_tolerance() -> None:
    sleeper_points = {"00-0031234": 8.84}
    computed_points = {"00-0031234": 8.839}

    disagreements = golden.compare_points(sleeper_points, computed_points, tolerance=0.01)

    assert disagreements == []


def test_compare_points_flags_disagreement_beyond_tolerance() -> None:
    sleeper_points = {"00-0031234": 8.84}
    computed_points = {"00-0031234": 6.0}

    disagreements = golden.compare_points(sleeper_points, computed_points, tolerance=0.01)

    assert len(disagreements) == 1
    assert disagreements[0].player_id == "00-0031234"
    assert disagreements[0].sleeper_points == pytest.approx(8.84)
    assert disagreements[0].computed_points == pytest.approx(6.0)


def test_compare_points_treats_missing_computed_row_as_zero_and_flags_it() -> None:
    sleeper_points = {"00-0099999": 8.84}
    computed_points: dict[str, float] = {}

    disagreements = golden.compare_points(sleeper_points, computed_points, tolerance=0.01)

    assert len(disagreements) == 1
    assert disagreements[0].computed_points == 0.0
    assert disagreements[0].missing_computed_row is True


def test_compare_points_agrees_when_sleeper_and_missing_row_are_both_zero() -> None:
    """A bye-week player with no nflverse row and Sleeper's own 0.0 isn't a real
    disagreement -- it's the expected shape of "didn't play"."""
    sleeper_points = {"00-0031234": 0.0}
    computed_points: dict[str, float] = {}

    disagreements = golden.compare_points(sleeper_points, computed_points, tolerance=0.01)

    assert disagreements == []


# --- summarize ---------------------------------------------------------------------


def test_summarize_computes_agreement_rate_and_pass_threshold() -> None:
    disagreements = [
        golden.Disagreement(
            week=1,
            player_id="a",
            sleeper_points=8.0,
            computed_points=6.0,
            missing_computed_row=False,
        )
    ]

    result = golden.summarize(total_player_weeks=100, disagreements=disagreements)

    assert result.agreement_rate == pytest.approx(0.99)
    assert result.passed is True  # exactly the 99% threshold


# --- run_golden_test (orchestration) ---------------------------------------------
# Bijan Robinson (sleeper_id 9226 -> gsis_id 00-0039163) from the ids/mapping
# fixtures stands in for the one player-week this end-to-end test scores.


def test_run_golden_test_orchestrates_fetch_score_and_compare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        golden,
        "_load_league",
        lambda slug: LeagueConfig(
            slug=slug,
            display_name=slug,
            is_primary=True,
            league_id="CURRENT1",
            season=2026,
            league_cache={},
            overrides={},
        ),
    )

    league_raw = {
        "CURRENT1": {"previous_league_id": "PREV1"},
        "PREV1": {"season": 2025, "scoring_settings": {"pass_yd": 0.04}},
    }

    def fake_fetch_league(
        league_id: str, *, offline: object = None, settings: object = None
    ) -> Path:
        path = tmp_path / f"league_{league_id}.json"
        path.write_text(json.dumps(league_raw[league_id]))
        return path

    monkeypatch.setattr(sleeper, "fetch_league", fake_fetch_league)
    monkeypatch.setattr(nflverse, "fetch_player_ids", lambda **kwargs: CROSSWALK_CSV)
    monkeypatch.setattr(sleeper, "fetch_players", lambda **kwargs: SLEEPER_PLAYERS_JSON)
    monkeypatch.setattr(mapping, "ID_OVERRIDES_PATH", OVERRIDES_CSV)

    player_stats_path = tmp_path / "player_stats.parquet"
    pl.DataFrame(
        {
            "player_id": ["00-0039163"],
            "season": [2025],
            "week": [1],
            "position": ["RB"],
            "passing_yards": [250],
        }
    ).write_parquet(player_stats_path)
    monkeypatch.setattr(nflverse, "fetch_player_stats", lambda season, **kwargs: player_stats_path)

    team_stats_path = tmp_path / "team_stats.parquet"
    pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "opponent_team": ["BAL"],
            "game_id": ["2025_01_BAL_KC"],
            "def_sacks": [0],
            "def_interceptions": [0],
            "def_fumbles_forced": [0],
            "fumble_recovery_opp": [0],
            "fumble_recovery_tds": [0],
            "def_safeties": [0],
            "def_tds": [0],
            "special_teams_tds": [0],
            "fg_blocked": [0],
            "pat_blocked": [0],
        }
    ).write_parquet(team_stats_path)
    monkeypatch.setattr(nflverse, "fetch_team_stats", lambda season, **kwargs: team_stats_path)

    schedules_path = tmp_path / "schedules.parquet"
    pl.DataFrame(
        {
            "game_id": ["2025_01_BAL_KC"],
            "season": [2025],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "home_score": [27],
            "away_score": [17],
        }
    ).write_parquet(schedules_path)
    monkeypatch.setattr(nflverse, "fetch_schedules", lambda season, **kwargs: schedules_path)

    pbp_path = tmp_path / "pbp.parquet"
    pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "defteam": pl.Utf8,
            "td_team": pl.Utf8,
            "return_touchdown": pl.Int64,
            "play_type": pl.Utf8,
            "fumble": pl.Int64,
            "special_teams_play": pl.Int64,
            "forced_fumble_player_1_team": pl.Utf8,
            "fumbled_1_team": pl.Utf8,
            "fumble_recovery_1_team": pl.Utf8,
        }
    ).write_parquet(pbp_path)
    monkeypatch.setattr(nflverse, "fetch_pbp", lambda season, **kwargs: pbp_path)

    week1_matchups = tmp_path / "matchups_w1.json"
    week1_matchups.write_text(json.dumps([{"players_points": {"9226": 10.0}}]))
    empty_matchups = tmp_path / "matchups_empty.json"
    empty_matchups.write_text(json.dumps([]))

    def fake_fetch_matchups(
        league_id: str, week: int, *, offline: object = None, settings: object = None
    ) -> Path:
        return week1_matchups if week == 1 else empty_matchups

    monkeypatch.setattr(sleeper, "fetch_matchups", fake_fetch_matchups)

    result = golden.run_golden_test("rogan-radinator-league")

    assert result.total_player_weeks == 1
    assert result.disagreements == []
    assert result.passed is True


def test_run_golden_test_raises_when_league_has_no_played_season(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        golden,
        "_load_league",
        lambda slug: LeagueConfig(
            slug=slug,
            display_name=slug,
            is_primary=True,
            league_id="CURRENT1",
            season=2026,
            league_cache={},
            overrides={},
        ),
    )

    def fake_fetch_league(
        league_id: str, *, offline: object = None, settings: object = None
    ) -> Path:
        path = tmp_path / "league.json"
        path.write_text(json.dumps({"previous_league_id": None}))
        return path

    monkeypatch.setattr(sleeper, "fetch_league", fake_fetch_league)

    with pytest.raises(golden.NoPlayedSeasonError):
        golden.run_golden_test("rogan-radinator-league")


def test_summarize_fails_below_threshold() -> None:
    disagreements = [
        golden.Disagreement(
            week=1,
            player_id=f"p{i}",
            sleeper_points=8.0,
            computed_points=6.0,
            missing_computed_row=False,
        )
        for i in range(5)
    ]

    result = golden.summarize(total_player_weeks=100, disagreements=disagreements)

    assert result.agreement_rate == pytest.approx(0.95)
    assert result.passed is False
