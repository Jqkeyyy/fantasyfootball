"""Games-played adjustment (SPEC.md §9.3; task 0.8).

Season projections from public sources are usually implicitly "if healthy" --
this converts each source's consensus season total to points-per-game, then
rescales by a crude positional/age prior for expected games played, so that a
pretend-everyone-plays-17-games total doesn't systematically overvalue
injury-prone players and RBs generally.

`POSITION_BASE_AVAILABILITY`/`AGE_CLIFF` are documented, informed placeholder
rates -- not fit from data. SPEC §9.3 is explicit that Phase 0 only needs "a
crude but honest prior"; the full hazard model driven by real
`injury_history` is task 2.3/§13.3, a Phase 2 deliverable. `p_available_baseline`
accepts `injury_history` to match SPEC's documented signature, but it is
unused here -- revisit when §13.3 lands.

Player matching reuses `projections/aggregate.py`'s normalized (name,
position) `join_key`, the same deliberate simplification task 0.7 already
made (not task 0.3's canonical player_id crosswalk) -- see that module's
docstring.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ffapp.ids.mapping import normalize_name

GAMES_IN_SEASON = 17
MIN_AVAILABILITY = 0.5

# Informed placeholder rates for "fraction of a 17-game season this position
# typically plays," not fit from real injury data (see module docstring).
POSITION_BASE_AVAILABILITY: dict[str, float] = {
    "QB": 0.94,
    "RB": 0.83,
    "WR": 0.89,
    "TE": 0.87,
    "K": 0.97,
    "DST": 1.0,
}

# position -> (threshold_age, decay_per_year_over_threshold). Availability
# erodes linearly once a player crosses their position's "aging cliff."
# DST has no individual age (a team entity); K's performance/availability
# curve is famously flat with age -- neither gets an age adjustment.
AGE_CLIFF: dict[str, tuple[float, float]] = {
    "QB": (35.0, 0.015),
    "RB": (27.0, 0.035),
    "WR": (30.0, 0.02),
    "TE": (30.0, 0.02),
}


def p_available_baseline(
    position: str, age: float | None, injury_history: object | None = None
) -> float:
    """SPEC §9.3's crude positional/age prior for fraction of a season played."""
    if position not in POSITION_BASE_AVAILABILITY:
        raise ValueError(f"Unknown position for games-played prior: {position!r}")

    base = POSITION_BASE_AVAILABILITY[position]
    if age is None or position not in AGE_CLIFF:
        return base

    threshold, decay = AGE_CLIFF[position]
    if age <= threshold:
        return base

    return max(MIN_AVAILABILITY, base - decay * (age - threshold))


def expected_games(position: str, age: float | None, injury_history: object | None = None) -> float:
    """SPEC §9.3: `17 x p_available_baseline(position, age, injury_history)`."""
    return GAMES_IN_SEASON * p_available_baseline(position, age, injury_history)


def player_ages_from_players_dim(players_dim: pl.DataFrame, *, as_of: date) -> pl.DataFrame:
    """Fractional age in years as of `as_of`, keyed by the same normalized
    (name, position) `join_key` `projections/aggregate.py.add_join_key` uses,
    from a `players_dim`-shaped table (`full_name`, `position`, `birth_date`).
    """
    as_of_days = as_of.toordinal()
    return players_dim.select(
        (
            pl.col("full_name").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key"),
        (
            (
                as_of_days
                - pl.col("birth_date").cast(pl.Utf8).str.to_date(strict=False).dt.epoch("d")
                - 719163
            )
            / 365.25
        ).alias("age"),
    )


def add_games_played_adjustment(projections: pl.DataFrame, ages: pl.DataFrame) -> pl.DataFrame:
    """Add `proj_ppg`, `expected_games`, `proj_points_adj` (SPEC §9.3) to an
    aggregated projections table (`projections/aggregate.py.aggregate_projections`
    output: `join_key`, `position`, `proj_points`).

    A player with no `ages` match (e.g. unresolved to the crosswalk) still
    gets `expected_games` populated via the position-only baseline
    (`age=None`), never a null/dropped row (CLAUDE.md rule 4).
    """
    joined = projections.join(ages, on="join_key", how="left")

    return (
        joined.with_columns((pl.col("proj_points") / GAMES_IN_SEASON).alias("proj_ppg"))
        .pipe(
            lambda df: df.with_columns(
                pl.struct(["position", "age"])
                .map_elements(
                    lambda s: expected_games(s["position"], s["age"]), return_dtype=pl.Float64
                )
                .alias("expected_games")
            )
        )
        .with_columns((pl.col("proj_ppg") * pl.col("expected_games")).alias("proj_points_adj"))
    )


__all__ = [
    "AGE_CLIFF",
    "GAMES_IN_SEASON",
    "MIN_AVAILABILITY",
    "POSITION_BASE_AVAILABILITY",
    "add_games_played_adjustment",
    "expected_games",
    "p_available_baseline",
    "player_ages_from_players_dim",
]
