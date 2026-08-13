"""Task 2.7's DST feature table (SPEC §11.6): the as_of contract applied
to a team-level (not player-level) row -- "opponent" columns must reflect
the *opponent's own* trailing history through the week *before* the
target week, "own defence" columns the same for this team, and
`opp_implied_team_total`/`is_home`/weather must stay unshifted (already
known before the target week's own kickoff). Small hand-verifiable
fixtures, no live `data/` needed.
"""

from __future__ import annotations

import polars as pl

from ffapp.features import dst

# --- fixtures --------------------------------------------------------------------------

_DEFENSE_COLUMNS = (
    "sack_rate_allowed",
    "pressure_rate_allowed",
    "turnover_rate",
    "interception_rate_thrown",
    "pressure_rate_forced",
    "takeaway_rate",
)


def _defense_row(team: str, week: int, **rates: float) -> dict:
    row: dict[str, object] = {"team": team, "season": 2025, "week": week}
    row.update(dict.fromkeys(_DEFENSE_COLUMNS, 0.0))
    row.update(rates)
    return row


def _context_row(team: str, week: int, implied_total: float) -> dict:
    return {"team": team, "season": 2025, "week": week, "implied_total": implied_total}


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "home_team": "A", "away_team": "B"},
            {"game_id": "g2", "season": 2025, "week": 2, "home_team": "C", "away_team": "A"},
        ]
    )


def _weather() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": "g1",
                "wind_mph": 5.0,
                "precip_prob": 0.0,
                "temp_f": 70.0,
                "is_dome": False,
            },
            {
                "game_id": "g2",
                "wind_mph": 12.0,
                "precip_prob": 30.0,
                "temp_f": 55.0,
                "is_dome": False,
            },
        ]
    )


def _empty_snap_counts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "pfr_player_id": pl.Utf8,
            "position": pl.Utf8,
            "offense_snaps": pl.Int64,
        }
    )


def _build() -> pl.DataFrame:
    team_week_defense = pl.DataFrame(
        [
            _defense_row(
                "A",
                1,
                sack_rate_allowed=0.10,
                pressure_rate_allowed=0.20,
                turnover_rate=0.05,
                interception_rate_thrown=0.02,
                pressure_rate_forced=0.15,
                takeaway_rate=0.03,
            ),
            _defense_row(
                "B",
                1,
                sack_rate_allowed=0.05,
                pressure_rate_allowed=0.10,
                turnover_rate=0.01,
                interception_rate_thrown=0.01,
                pressure_rate_forced=0.25,
                takeaway_rate=0.06,
            ),
            _defense_row(
                "C",
                1,
                sack_rate_allowed=0.08,
                pressure_rate_allowed=0.12,
                turnover_rate=0.02,
                interception_rate_thrown=0.01,
                pressure_rate_forced=0.18,
                takeaway_rate=0.04,
            ),
            _defense_row("A", 2, sack_rate_allowed=0.20, pressure_rate_forced=0.05),
            _defense_row("C", 2, sack_rate_allowed=0.09, pressure_rate_forced=0.19),
        ]
    )
    team_week_context = pl.DataFrame(
        [
            _context_row("A", 1, 22.0),
            _context_row("B", 1, 19.5),
            _context_row("C", 1, 24.0),
            _context_row("A", 2, 21.0),
            _context_row("C", 2, 27.5),
        ]
    )
    return dst.build_dst_features(
        team_week_defense,
        team_week_context,
        _empty_snap_counts(),
        _schedule(),
        _weather(),
        registry={},
    )


# --- tests -------------------------------------------------------------------------------


def test_week_one_has_no_trailing_history() -> None:
    """A team's very first tracked week has no prior week to trail from
    -- every windowed opponent/own-defence column must be honestly
    null, not a guessed 0.0."""
    result = _build()

    a_week1 = result.filter((pl.col("team") == "A") & (pl.col("week") == 1)).row(0, named=True)
    for column in dst._OPPONENT_OFFENSE_COLUMNS + dst._OWN_DEFENSE_COLUMNS:
        assert a_week1[column] is None


def test_opponent_columns_reflect_the_opponents_own_prior_week_not_the_current_one() -> None:
    """Team A's week-2 opponent is C. C's own week-2 `opp_*` values must
    equal C's own real week-1 raw rate (a single trailing data point, so
    `ewm_mean` == that point itself), not C's own week-2 raw rate -- the
    as_of contract applied to an opponent's own history."""
    result = _build()

    a_week2 = result.filter((pl.col("team") == "A") & (pl.col("week") == 2)).row(0, named=True)
    assert a_week2["opponent_team"] == "C"
    assert a_week2["opp_sack_rate_allowed_ewm_8"] == 0.08  # C's real week-1 value
    assert a_week2["opp_sack_rate_allowed_ewm_8"] != 0.09  # not C's week-2 value


def test_own_defence_columns_reflect_this_teams_own_prior_week() -> None:
    """Same as_of contract, applied to this team's own defensive
    production rather than the opponent's offense."""
    result = _build()

    a_week2 = result.filter((pl.col("team") == "A") & (pl.col("week") == 2)).row(0, named=True)
    assert a_week2["own_pressure_rate_forced_ewm_8"] == 0.15  # A's real week-1 value
    assert a_week2["own_pressure_rate_forced_ewm_8"] != 0.05  # not A's week-2 value


def test_implied_team_total_and_home_away_are_current_week_not_shifted() -> None:
    """`opp_implied_team_total`/`is_home` are Vegas/schedule facts already
    known before the target week's own kickoff -- unlike the trailing
    rate columns, these must reflect the *current* week's own real
    values, matching `features.team_context.CURRENT_WEEK_COLUMNS`'
    established precedent for `implied_team_total`/`spread`."""
    result = _build()

    a_week1 = result.filter((pl.col("team") == "A") & (pl.col("week") == 1)).row(0, named=True)
    assert a_week1["is_home"] is True
    assert a_week1["opp_implied_team_total"] == 19.5  # B's own week-1 implied total

    a_week2 = result.filter((pl.col("team") == "A") & (pl.col("week") == 2)).row(0, named=True)
    assert a_week2["is_home"] is False
    assert a_week2["opp_implied_team_total"] == 27.5  # C's own week-2 implied total


def test_weather_is_joined_by_game_id() -> None:
    result = _build()

    a_week1 = result.filter((pl.col("team") == "A") & (pl.col("week") == 1)).row(0, named=True)
    assert a_week1["wind_mph"] == 5.0
    a_week2 = result.filter((pl.col("team") == "A") & (pl.col("week") == 2)).row(0, named=True)
    assert a_week2["wind_mph"] == 12.0


def test_returns_one_row_per_team_per_game_played() -> None:
    result = _build()

    # 2 games x 2 teams each = 4 team-week rows.
    assert result.height == 4
