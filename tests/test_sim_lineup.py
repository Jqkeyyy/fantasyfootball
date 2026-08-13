"""Task 2.1's own literal acceptance bar (SPEC §13.1): "the ILP produces
known-correct lineups on FLEX and superflex fixtures." Every fixture here
is small enough to hand-verify the optimal assignment by inspection --
the point of these tests is that the ILP's real output matches that
hand-verified optimum, not just that it runs without crashing.
"""

from __future__ import annotations

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import Lineup, PlayerProjection, optimal_lineup, optimal_lineup_points

# --- fixtures ---------------------------------------------------------------------


def _player(
    player_id: str,
    position: str,
    *,
    mean: float,
    median: float | None = None,
    ceiling: float | None = None,
) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id,
        position=position,
        mean=mean,
        median=median if median is not None else mean,
        ceiling=ceiling if ceiling is not None else mean,
    )


def _single_flex_format() -> LeagueFormat:
    """1 RB starter, 1 FLEX (RB/WR/TE eligible) -- SPEC's own literal
    "FLEX fixture" case."""
    return LeagueFormat(
        n_teams=10,
        starters={"RB": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=4,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )


def _superflex_format() -> LeagueFormat:
    """1 QB starter, 1 SUPER_FLEX (QB/RB/WR/TE eligible) -- SPEC's own
    literal "superflex fixture" case."""
    return LeagueFormat(
        n_teams=10,
        starters={"QB": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 1, "REC_FLEX": 0},
        flex_eligible={"SUPER_FLEX": ["QB", "RB", "WR", "TE"]},
        bench=4,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )


# --- optimal_lineup: FLEX -----------------------------------------------------------


def test_flex_fixture_fills_rb_and_flex_with_the_two_best_flex_eligible_players() -> None:
    """3 flex-eligible players (RB1=20, WR1=15, TE1=10), 1 RB slot + 1
    FLEX slot -- the optimum is unambiguous: RB1 in RB (nothing else is
    RB-eligible for that slot), WR1 in FLEX (the next-best of what's
    left), TE1 benched."""
    players = [
        _player("rb1", "RB", mean=20.0),
        _player("wr1", "WR", mean=15.0),
        _player("te1", "TE", mean=10.0),
    ]

    lineup = optimal_lineup(players, _single_flex_format())

    assert lineup.slots["RB_1"] == "rb1"
    assert lineup.slots["FLEX_1"] == "wr1"
    assert "te1" not in lineup.slots.values()
    assert lineup.total_points == 35.0


def test_flex_fixture_never_uses_the_same_player_in_two_slots() -> None:
    """Only one flex-eligible player exists at all -- both the RB slot
    and the FLEX slot could legally point at him, but "each player used
    at most once" (SPEC §13.1) means only one of the two slots can
    actually be filled by him, and there's no other candidate to fill
    the other -- the ILP must leave one slot legitimately empty rather
    than double-book the same player into both."""
    players = [_player("rb1", "RB", mean=20.0)]

    lineup = optimal_lineup(players, _single_flex_format())

    assert list(lineup.slots.values()).count("rb1") == 1
    assert lineup.total_points == 20.0


def test_flex_fixture_respects_position_eligibility() -> None:
    """A QB is not FLEX-eligible in this format (FLEX = RB/WR/TE only) --
    even though the QB has the highest raw value, he can fill neither the
    RB slot nor the FLEX slot, so he must be left out entirely."""
    players = [
        _player("qb1", "QB", mean=30.0),
        _player("rb1", "RB", mean=10.0),
        _player("wr1", "WR", mean=8.0),
    ]

    lineup = optimal_lineup(players, _single_flex_format())

    assert "qb1" not in lineup.slots.values()
    assert lineup.slots["RB_1"] == "rb1"
    assert lineup.slots["FLEX_1"] == "wr1"


# --- optimal_lineup: SUPER_FLEX -----------------------------------------------------


def test_superflex_fixture_can_start_two_quarterbacks() -> None:
    """2 QBs (25, 22) and 1 lower-value RB (10) -- SUPER_FLEX is QB-
    eligible, and QB2 (22) beats the RB (10), so the optimal lineup
    starts both QBs, not QB1 + RB. Both QB_1 and SUPER_FLEX_1 are
    QB-eligible with no tiebreaker between them, so which specific QB
    lands in which slot is a legitimately tied choice -- only the
    *set* of players used and the total are pinned down."""
    players = [
        _player("qb1", "QB", mean=25.0),
        _player("qb2", "QB", mean=22.0),
        _player("rb1", "RB", mean=10.0),
    ]

    lineup = optimal_lineup(players, _superflex_format())

    assert set(lineup.slots.values()) == {"qb1", "qb2"}
    assert lineup.total_points == 47.0


def test_superflex_fixture_prefers_a_better_flex_eligible_player_over_a_worse_qb() -> None:
    """1 QB (12) and 1 RB (18) -- the RB outscores the backup-quality QB,
    so the optimal SUPER_FLEX pick is the RB, not a second, weaker QB
    that doesn't exist here anyway; this exercises SUPER_FLEX's own
    broader eligibility list actually being used, not just defaulting to
    "always fill with a QB if any QB is left"."""
    players = [
        _player("qb1", "QB", mean=12.0),
        _player("rb1", "RB", mean=18.0),
    ]

    lineup = optimal_lineup(players, _superflex_format())

    assert lineup.slots["QB_1"] == "qb1"
    assert lineup.slots["SUPER_FLEX_1"] == "rb1"


# --- optimal_lineup: objective ------------------------------------------------------


def test_objective_ceiling_can_pick_a_lower_mean_higher_ceiling_player() -> None:
    """WR1 has the higher mean (15 vs 12) but a lower ceiling (18) than
    WR2 (25) -- optimizing on "ceiling" should flip the FLEX pick versus
    optimizing on "mean"."""
    players = [
        _player("rb1", "RB", mean=20.0, median=20.0, ceiling=20.0),
        _player("wr1", "WR", mean=15.0, median=15.0, ceiling=18.0),
        _player("wr2", "WR", mean=12.0, median=12.0, ceiling=25.0),
    ]

    mean_lineup = optimal_lineup(players, _single_flex_format(), objective="mean")
    ceiling_lineup = optimal_lineup(players, _single_flex_format(), objective="ceiling")

    assert mean_lineup.slots["FLEX_1"] == "wr1"
    assert ceiling_lineup.slots["FLEX_1"] == "wr2"


# --- optimal_lineup_points ------------------------------------------------------------


def test_optimal_lineup_points_returns_the_same_total_as_optimal_lineup() -> None:
    """SPEC §13.1: `optimal_lineup_points(actual_points, fmt)`, "for
    computing lineup regret in evaluation" -- reuses the exact same
    `list[PlayerProjection]` shape, populated with realised points
    instead of a forecast, and returns just the resulting total."""
    actual_points = [
        _player("rb1", "RB", mean=20.0),
        _player("wr1", "WR", mean=15.0),
        _player("te1", "TE", mean=10.0),
    ]

    total = optimal_lineup_points(actual_points, _single_flex_format())

    assert total == 35.0


# --- Lineup dataclass ----------------------------------------------------------------


def test_lineup_is_a_plain_dataclass_with_slots_and_total_points() -> None:
    lineup = Lineup(slots={"RB_1": "rb1"}, total_points=20.0)

    assert lineup.slots == {"RB_1": "rb1"}
    assert lineup.total_points == 20.0
