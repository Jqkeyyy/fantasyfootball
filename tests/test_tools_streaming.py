"""Tests for tools/streaming.py: the empirical, streaming-aware replacement
level for DST/K (not a numbered TASKS.md task -- see module docstring)."""

from __future__ import annotations

import polars as pl

from ffapp.scoring import stats as scoring_stats
from ffapp.tools import streaming


def _points_row(*, position: str, player_id: str, season: int, week: int, points: float) -> dict:
    return {
        "position": position,
        "player_id": player_id,
        "season": season,
        "week": week,
        "points": points,
    }


# --- _season_streaming_total / streaming_replacement_points -----------------


def test_streaming_replacement_excludes_drafted_teams_from_the_weekly_pool() -> None:
    """The single best-scoring team all season (DST_A) is treated as
    "drafted" (n_drafted=1) -- its own huge week-1 score must never count
    toward the streaming total, even though it's the best score that week."""
    rows = pl.DataFrame(
        [
            _points_row(position="DST", player_id="DST_A", season=2024, week=1, points=100.0),
            _points_row(position="DST", player_id="DST_B", season=2024, week=1, points=10.0),
            _points_row(position="DST", player_id="DST_A", season=2024, week=2, points=100.0),
            _points_row(position="DST", player_id="DST_B", season=2024, week=2, points=12.0),
        ]
    )

    result = streaming.streaming_replacement_points(
        rows, position="DST", n_drafted=1, seasons=[2024], availability_rank=1
    )

    # DST_A (season total 200, the single best) is "drafted" and excluded;
    # the streaming total is just DST_B's own two weeks.
    assert result == 10.0 + 12.0


def test_streaming_replacement_uses_the_nth_best_available_score_each_week() -> None:
    """`availability_rank=2` takes the *second*-best undrafted score each
    week, not the single best -- the documented conservative haircut."""
    rows = pl.DataFrame(
        [
            _points_row(position="K", player_id="K_A", season=2024, week=1, points=9.0),
            _points_row(position="K", player_id="K_B", season=2024, week=1, points=7.0),
            _points_row(position="K", player_id="K_C", season=2024, week=1, points=5.0),
        ]
    )

    best = streaming.streaming_replacement_points(
        rows, position="K", n_drafted=0, seasons=[2024], availability_rank=1
    )
    second_best = streaming.streaming_replacement_points(
        rows, position="K", n_drafted=0, seasons=[2024], availability_rank=2
    )

    assert best == 9.0
    assert second_best == 7.0


def test_streaming_replacement_skips_a_week_with_too_few_available_teams() -> None:
    """A week where fewer than `availability_rank` teams are undrafted
    contributes 0 for that week rather than raising or faking a value."""
    rows = pl.DataFrame(
        [_points_row(position="DST", player_id="DST_A", season=2024, week=1, points=15.0)]
    )

    result = streaming.streaming_replacement_points(
        rows, position="DST", n_drafted=0, seasons=[2024], availability_rank=2
    )

    assert result == 0.0


def test_streaming_replacement_averages_across_seasons() -> None:
    rows = pl.DataFrame(
        [
            _points_row(position="DST", player_id="DST_A", season=2023, week=1, points=20.0),
            _points_row(position="DST", player_id="DST_A", season=2024, week=1, points=40.0),
        ]
    )

    result = streaming.streaming_replacement_points(
        rows, position="DST", n_drafted=0, seasons=[2023, 2024], availability_rank=1
    )

    assert result == (20.0 + 40.0) / 2


def test_streaming_replacement_overrides_maps_each_position_independently() -> None:
    rows = pl.DataFrame(
        [
            _points_row(position="DST", player_id="DST_A", season=2024, week=1, points=20.0),
            _points_row(position="K", player_id="K_A", season=2024, week=1, points=8.0),
        ]
    )

    overrides = streaming.streaming_replacement_overrides(
        rows,
        n_drafted_by_position={"DST": 0, "K": 0},
        seasons=[2024],
        availability_rank=1,
    )

    assert overrides == {"DST": 20.0, "K": 8.0}


# --- score_historical_stats: real regular-season scoping ---------------------


def _team_stats(**overrides: object) -> pl.DataFrame:
    base = {
        "season": [2025],
        "week": [1],
        "season_type": ["REG"],
        "team": ["KC"],
        "opponent_team": ["BAL"],
        "game_id": ["2025_01_BAL_KC"],
        "def_sacks": [3],
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
    base.update(overrides)
    return pl.DataFrame(base)


def _schedules(**overrides: object) -> pl.DataFrame:
    base = {
        "game_id": ["2025_01_BAL_KC", "2025_20_BAL_KC"],
        "season": [2025, 2025],
        "week": [1, 20],
        "home_team": ["KC", "KC"],
        "away_team": ["BAL", "BAL"],
        "home_score": [27, 27],
        "away_score": [17, 17],
    }
    base.update(overrides)
    return pl.DataFrame(base)


_EMPTY_PBP = pl.DataFrame(
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
)

_EMPTY_PLAYER_STATS = pl.DataFrame(
    schema={
        "player_id": pl.Utf8,
        "position": pl.Utf8,
        "season": pl.Int64,
        "week": pl.Int64,
        "season_type": pl.Utf8,
    }
)


def test_score_historical_stats_excludes_postseason_team_rows() -> None:
    """A POST-week DST row must never reach the scored output -- nflverse's
    raw team_stats includes real playoff games no fantasy league plays
    through (same reasoning as tools.sos's own REG scoping)."""
    team_stats = pl.concat(
        [
            _team_stats(week=[1], season_type=["REG"]),
            _team_stats(week=[20], season_type=["POST"], def_sacks=[9]),
        ]
    )

    result = streaming.score_historical_stats(
        _EMPTY_PLAYER_STATS, team_stats, _schedules(), _EMPTY_PBP, {"sack": 1.0}
    )

    assert result.filter(pl.col("position") == "DST")["week"].to_list() == [1]


def test_score_historical_stats_reuses_build_stat_frame_scoring_unmodified() -> None:
    """Sanity check that this module adds no new scoring math of its own --
    the scored points for a REG row must match calling
    scoring.stats.build_stat_frame/scoring.engine.score_stat_line directly."""
    from ffapp.scoring.engine import score_stat_line

    team_stats = _team_stats()
    schedules = _schedules()
    scoring_settings = {"sack": 1.0}

    direct = scoring_stats.build_stat_frame(
        _EMPTY_PLAYER_STATS, team_stats, schedules, _EMPTY_PBP
    )
    expected = direct.with_columns(score_stat_line(direct, scoring_settings).alias("points"))

    result = streaming.score_historical_stats(
        _EMPTY_PLAYER_STATS, team_stats, schedules, _EMPTY_PBP, scoring_settings
    )

    assert result["points"].to_list() == expected["points"].to_list()
