import polars as pl
import pytest

from ffapp.league_format import LeagueFormat
from ffapp.tools import vor


def _players(position: str, n: int, top_points: float, step: float = 1.0) -> list[dict]:
    return [
        {
            "position": position,
            "player_name": f"{position}{i}",
            "proj_points_adj": top_points - (i - 1) * step,
        }
        for i in range(1, n + 1)
    ]


def _format(**overrides: object) -> LeagueFormat:
    base = dict(
        n_teams=10,
        starters={},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=6,
        ir=0,
        playoff_week_start=15,
        waiver_budget=None,
    )
    base.update(overrides)
    return LeagueFormat(**base)  # type: ignore[arg-type]


# --- replacement_level: dedicated slots only --------------------------------


def test_replacement_level_uses_dedicated_starters_only_when_no_flex_slots() -> None:
    projections = pl.DataFrame(_players("QB", 15, 90.0) + _players("RB", 30, 70.0))
    league_format = _format(n_teams=10, starters={"QB": 1, "RB": 2})

    replacement = vor.replacement_level(projections, league_format)

    assert replacement["QB"] == pytest.approx(90.0 - 9)  # rank 10 (10*1)
    assert replacement["RB"] == pytest.approx(70.0 - 19)  # rank 20 (10*2)


# --- replacement_level: FLEX fixed-point iteration --------------------------


def test_replacement_level_standard_12_team_league_with_flex() -> None:
    """SPEC §9.4's own algorithm, hand-verified: RB's points curve is
    constructed to always dominate WR/TE for the FLEX-quality tier (its
    floor stays far above WR/TE's ceiling even after two rounds of
    baseline growth), so every FLEX slot goes to RB and the iteration
    settles at RB baseline = 24 (dedicated) + 12 (flex) = 36 by pass 2."""
    projections = pl.DataFrame(
        _players("QB", 15, 90.0)
        + _players("RB", 60, 200.0)
        + _players("WR", 30, 60.0)
        + _players("TE", 20, 55.0)
        + _players("K", 15, 30.0)
        + _players("DST", 20, 25.0)
    )
    league_format = _format(
        n_teams=12,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
    )

    replacement = vor.replacement_level(projections, league_format)

    assert replacement == pytest.approx(
        {
            "QB": 79.0,  # rank 12, no flex eligibility
            "RB": 165.0,  # rank 36 = 24 dedicated + 12 flex fillers
            "WR": 37.0,  # rank 24, dedicated only -- RB won every flex slot
            "TE": 44.0,  # rank 12, dedicated only -- RB won every flex slot
            "K": 19.0,  # rank 12, not flex eligible
            "DST": 14.0,  # rank 12, not flex eligible
        }
    )


def test_startable_counts_matches_replacement_levels_own_baseline() -> None:
    """`startable_counts` is a public accessor for the same fixed-point
    baseline `replacement_level` already computes internally (task 1.13's
    `evaluation.metrics` reuses it directly) -- the rank counts here must
    be exactly the ranks `replacement_level`'s own docstring already
    hand-verifies for this fixture (RB 36, others dedicated-only)."""
    points_by_position = {
        "QB": [90.0 - (i - 1) for i in range(1, 16)],
        "RB": [200.0 - (i - 1) for i in range(1, 61)],
        "WR": [60.0 - (i - 1) for i in range(1, 31)],
        "TE": [55.0 - (i - 1) for i in range(1, 21)],
        "K": [30.0 - (i - 1) for i in range(1, 16)],
        "DST": [25.0 - (i - 1) for i in range(1, 21)],
    }
    league_format = _format(
        n_teams=12,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
    )

    counts = vor.startable_counts(points_by_position, league_format)

    assert counts == {"QB": 12, "RB": 36, "WR": 24, "TE": 12, "K": 12, "DST": 12}


def test_replacement_level_converges_within_two_passes_for_the_standard_league() -> None:
    """TASKS.md 0.9's literal acceptance bar: convergence in under 10 passes.
    Capping max_iterations at 2 must already match the fully-converged
    (max_iterations=10) result for this fixture."""
    projections = pl.DataFrame(
        _players("QB", 15, 90.0)
        + _players("RB", 60, 200.0)
        + _players("WR", 30, 60.0)
        + _players("TE", 20, 55.0)
        + _players("K", 15, 30.0)
        + _players("DST", 20, 25.0)
    )
    league_format = _format(
        n_teams=12,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
    )

    fully_converged = vor.replacement_level(projections, league_format, max_iterations=10)
    capped_at_two = vor.replacement_level(projections, league_format, max_iterations=2)

    assert capped_at_two == pytest.approx(fully_converged)


def test_replacement_level_includes_a_flex_only_position_with_no_dedicated_slot() -> None:
    """A position with zero dedicated starters but flex eligibility (e.g. a
    league with no standalone TE slot) must still get a real replacement
    level once it wins flex slots -- CLAUDE.md rule 5, driven by
    LeagueFormat, not a hardcoded position list."""
    projections = pl.DataFrame(
        _players("RB", 30, 60.0) + _players("WR", 30, 55.0) + _players("TE", 20, 200.0)
    )
    league_format = _format(
        n_teams=10,
        starters={"RB": 2, "WR": 2},  # no dedicated TE slot at all
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
    )

    replacement = vor.replacement_level(projections, league_format)

    assert "TE" in replacement
    assert replacement["TE"] == pytest.approx(200.0 - 9)  # rank 10 (0 dedicated + 10 flex)


# --- superflex: QB baseline shifts dramatically ------------------------------


def test_qb_baseline_shifts_dramatically_under_superflex() -> None:
    """SPEC §9.4's own note: 'In a superflex league the QB baseline moves
    dramatically... do not special-case superflex by hand.' QB's points
    curve dominates RB/WR/TE here, so every SUPER_FLEX slot goes to QB."""
    projections = pl.DataFrame(
        _players("QB", 40, 300.0)
        + _players("RB", 40, 100.0)
        + _players("WR", 40, 90.0)
        + _players("TE", 20, 80.0)
    )
    shared_starters = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    single_qb_format = _format(n_teams=12, starters=shared_starters)
    superflex_format = _format(
        n_teams=12,
        starters=shared_starters,
        flex_slots={"FLEX": 0, "SUPER_FLEX": 1, "REC_FLEX": 0},
        flex_eligible={"SUPER_FLEX": ["QB", "RB", "WR", "TE"]},
    )

    single_qb_replacement = vor.replacement_level(projections, single_qb_format)
    superflex_replacement = vor.replacement_level(projections, superflex_format)

    assert single_qb_replacement["QB"] == pytest.approx(300.0 - 11)  # rank 12
    assert superflex_replacement["QB"] == pytest.approx(300.0 - 23)  # rank 24 -- baseline doubled
    assert superflex_replacement["QB"] < single_qb_replacement["QB"]
    # Non-QB baselines are untouched -- QB dominance means it wins every
    # SUPER_FLEX slot, not RB/WR/TE.
    assert superflex_replacement["RB"] == pytest.approx(single_qb_replacement["RB"])


# --- replacement_level: overrides for streamable positions (DST/K) ----------


def test_replacement_level_override_replaces_the_computed_value() -> None:
    projections = pl.DataFrame(_players("DST", 12, 25.0))
    league_format = _format(n_teams=10, starters={"DST": 1})

    replacement = vor.replacement_level(
        projections, league_format, replacement_overrides={"DST": 232.2}
    )

    assert replacement["DST"] == pytest.approx(232.2)


def test_replacement_level_override_only_touches_named_positions() -> None:
    projections = pl.DataFrame(_players("QB", 15, 90.0) + _players("DST", 12, 25.0))
    league_format = _format(n_teams=10, starters={"QB": 1, "DST": 1})

    without_override = vor.replacement_level(projections, league_format)
    with_override = vor.replacement_level(
        projections, league_format, replacement_overrides={"DST": 232.2}
    )

    assert with_override["QB"] == pytest.approx(without_override["QB"])
    assert with_override["DST"] == pytest.approx(232.2)


def test_replacement_level_override_ignores_a_position_this_league_never_starts() -> None:
    projections = pl.DataFrame(_players("QB", 15, 90.0))
    league_format = _format(n_teams=10, starters={"QB": 1})

    replacement = vor.replacement_level(
        projections, league_format, replacement_overrides={"DST": 232.2}
    )

    assert "DST" not in replacement


def test_compute_vor_applies_replacement_overrides() -> None:
    """A streaming-derived replacement level far above the top preseason
    DST total (the real, confirmed case -- see tools.streaming) must drive
    every DST's VOR negative, not just shrink it."""
    projections = pl.DataFrame(_players("DST", 12, 140.0))
    league_format = _format(n_teams=10, starters={"DST": 1})

    result = vor.compute_vor(projections, league_format, replacement_overrides={"DST": 232.2})

    assert (result["vor"] < 0).all()


# --- compute_vor --------------------------------------------------------------


def test_compute_vor_adds_vor_as_points_minus_replacement() -> None:
    projections = pl.DataFrame(_players("QB", 15, 90.0) + _players("RB", 30, 70.0))
    league_format = _format(n_teams=10, starters={"QB": 1, "RB": 2})

    result = vor.compute_vor(projections, league_format)

    qb1 = result.filter(pl.col("player_name") == "QB1").row(0, named=True)
    assert qb1["vor"] == pytest.approx(90.0 - 81.0)  # top QB vs. replacement (rank 10 = 81)


def test_compute_vor_raises_for_a_position_this_league_never_rosters() -> None:
    projections = pl.DataFrame(_players("QB", 15, 90.0) + _players("LB", 5, 20.0))
    league_format = _format(n_teams=10, starters={"QB": 1})

    with pytest.raises(ValueError, match="LB"):
        vor.compute_vor(projections, league_format)
