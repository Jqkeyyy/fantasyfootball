"""Value over replacement (SPEC.md §9.4; task 0.9).

Replacement level is circular with the FLEX slot -- which positions fill
FLEX depends on player values, which depend on replacement level -- so SPEC
resolves it with a short fixed-point iteration, followed literally here
rather than solved in closed form.

Driven entirely by `LeagueFormat` (CLAUDE.md rule 5: never hardcode a
league's starting lineup shape). K and DST get a replacement level like
everyone else; SPEC notes their VOR is almost always tiny, which is the
correct result, not a bug. In a superflex league the QB baseline moves
dramatically -- this algorithm handles it automatically, no special-casing.

Known gap, inherited directly from SPEC's own pseudocode: when a league has
more than one *simultaneously active* flex type (e.g. both FLEX and
SUPER_FLEX with nonzero slots), each flex type's candidate pool is computed
independently within a single pass, using the same prior-pass positional
baseline -- a player eligible for two active flex types could in principle
be selected as a "filler" under both in the same pass, inflating that pass's
flex_count. SPEC's algorithm doesn't address this, and neither real league
configured for this project uses more than one simultaneously active flex
type, so it's left as-is rather than guessed at.
"""

from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat

DEFAULT_POINTS_COLUMN = "proj_points_adj"
DEFAULT_MAX_ITERATIONS = 10


def _points_by_position(projections: pl.DataFrame, points_column: str) -> dict[str, list[float]]:
    """position -> that position's projected points, sorted descending."""
    return {
        position: sorted(
            projections.filter(pl.col("position") == position)[points_column]
            .drop_nulls()
            .to_list(),
            reverse=True,
        )
        for position in projections["position"].unique().to_list()
    }


def _all_positions(league_format: LeagueFormat) -> set[str]:
    """Every position this league starts, dedicated or flex-only (e.g. a
    league with no standalone TE slot but TE is FLEX-eligible)."""
    positions = set(league_format.starters)
    for eligible in league_format.flex_eligible.values():
        positions.update(eligible)
    return positions


def _converge_baseline(
    points_by_position: dict[str, list[float]],
    league_format: LeagueFormat,
    max_iterations: int,
) -> dict[str, int]:
    """SPEC §9.4 steps 1-2: the fixed-point iteration over flex fillers."""
    positions = _all_positions(league_format)
    n_teams = league_format.n_teams
    dedicated = {pos: n_teams * league_format.starters.get(pos, 0) for pos in positions}
    baseline = dict(dedicated)

    for _pass in range(max_iterations):
        flex_count = dict.fromkeys(positions, 0)

        for flex_type, eligible_positions in league_format.flex_eligible.items():
            n_slots = league_format.flex_slots.get(flex_type, 0)
            if n_slots <= 0:
                continue

            candidates: list[tuple[float, str]] = []
            for pos in eligible_positions:
                pool = points_by_position.get(pos, [])
                candidates.extend((points, pos) for points in pool[baseline[pos] :])
            candidates.sort(key=lambda c: c[0], reverse=True)

            for _points, pos in candidates[: n_teams * n_slots]:
                flex_count[pos] += 1

        new_baseline = {pos: dedicated[pos] + flex_count[pos] for pos in positions}
        if new_baseline == baseline:
            return new_baseline
        baseline = new_baseline

    return baseline


def startable_counts(
    points_by_position: dict[str, list[float]],
    league_format: LeagueFormat,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, int]:
    """Public accessor for SPEC §9.4's fixed-point baseline counts --
    reused directly by `evaluation.metrics` (task 1.13) for "startable"
    row scoping and top-k precision (SPEC §12.4), so both places share
    exactly one definition of "how many players at this position are
    startable in this league format," never a second one that could
    silently drift from this module's own VOR baseline.
    """
    return _converge_baseline(points_by_position, league_format, max_iterations)


def replacement_level(
    projections: pl.DataFrame,
    league_format: LeagueFormat,
    *,
    points_column: str = DEFAULT_POINTS_COLUMN,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    replacement_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """SPEC §9.4's fixed-point replacement-level algorithm. Returns
    `{position: replacement_points}`, one entry per position this league
    actually starts (dedicated or flex-eligible) with a nonzero baseline --
    a position with baseline 0 (never started, not flex eligible either)
    has no meaningful replacement level and is omitted.

    `replacement_overrides`, if given, replaces the computed value for any
    position it names (silently ignoring an override for a position this
    league doesn't actually start -- nothing to override). Built for DST/K:
    "the Nth-best preseason total" is the wrong replacement level for a
    position that's realistically streamed off waivers every week rather
    than drafted for season-long value -- see `tools.streaming`'s module
    docstring for the real historical data behind this, and why the
    fixed-point baseline below is the right default for every other
    position but not those two.
    """
    points_by_position = _points_by_position(projections, points_column)
    baseline = _converge_baseline(points_by_position, league_format, max_iterations)

    replacement: dict[str, float] = {}
    for pos, rank in baseline.items():
        if rank <= 0:
            continue
        pool = points_by_position.get(pos, [])
        if not pool:
            continue
        # Defensive clamp: a position with fewer real projected players than
        # its roster baseline requires (thin real-world coverage, or a
        # deliberately small test fixture) still gets a value, from whoever
        # its worst projected player is, rather than an IndexError.
        replacement[pos] = pool[min(rank, len(pool)) - 1]

    if replacement_overrides:
        for pos, value in replacement_overrides.items():
            if pos in replacement:
                replacement[pos] = value
    return replacement


def compute_vor(
    projections: pl.DataFrame,
    league_format: LeagueFormat,
    *,
    points_column: str = DEFAULT_POINTS_COLUMN,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    replacement_overrides: dict[str, float] | None = None,
) -> pl.DataFrame:
    """SPEC §9.4 step 4: adds a `vor` column, `proj_points_adj[p] -
    replacement_points[position(p)]`.

    Raises ValueError for any position present in `projections` that this
    league's `LeagueFormat` never starts and never flex-starts -- scope the
    input first (e.g. with `ids.mapping.league_relevant_positions`) if that
    position (an IDP slot this league doesn't roster, say) is expected to be
    there but irrelevant.
    """
    replacement = replacement_level(
        projections,
        league_format,
        points_column=points_column,
        max_iterations=max_iterations,
        replacement_overrides=replacement_overrides,
    )

    present_positions = set(projections["position"].unique().to_list())
    unrostered = present_positions - set(replacement)
    if unrostered:
        raise ValueError(
            f"No replacement level for position(s) {sorted(unrostered)} -- this league's "
            "LeagueFormat never starts or flex-starts them. Scope the projections table to "
            "this league's rosterable positions first."
        )

    replacement_expr = pl.col("position").replace_strict(replacement, return_dtype=pl.Float64)
    return projections.with_columns((pl.col(points_column) - replacement_expr).alias("vor"))


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_POINTS_COLUMN",
    "compute_vor",
    "replacement_level",
    "startable_counts",
]
