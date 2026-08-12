"""Live draft assistant (SPEC.md §9.8; task 0.14).

Pure, pytest-testable logic only. Sleeper's `/draft/{id}/picks` (already
wrapped by `ingest.sleeper.fetch_draft_picks`, task 0.2 -- polling it every
5-10s during a live draft, per SPEC, is just calling it repeatedly with
`offline=False`; no new ingest function needed) returns each pick's own
`metadata` (`first_name`, `last_name`, `position`, ...) directly, so
resolving a pick to this project's normalized `(name, position)` join key
needs no separate player-dictionary lookup -- confirmed live against the
primary league's real, completed 2025 draft.

`draft.board`'s output (the CSV `ffapp draft board` writes) doesn't carry a
`join_key` column -- SPEC §9.7's column list doesn't include it -- so it's
rebuilt here from `player`/`position` the same way
`projections/aggregate.add_join_key` builds it from `player_name`/`position`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl

from ffapp.ids.mapping import normalize_name
from ffapp.league_format import LeagueFormat

RUN_WINDOW = 8
RUN_MULTIPLIER = 2.0


def pick_join_key(pick: dict[str, Any]) -> str | None:
    """A drafted pick's normalized `(name, position)` join key, straight
    from the pick's own `metadata`. `None` for a pick with no usable
    metadata (defensive -- not seen in any real pick this session).
    """
    metadata = pick.get("metadata") or {}
    first_name = metadata.get("first_name")
    last_name = metadata.get("last_name")
    position = metadata.get("position")
    if not first_name or not last_name or not position:
        return None
    return f"{normalize_name(f'{first_name} {last_name}')}|{position}"


def drafted_join_keys(picks: list[dict[str, Any]]) -> set[str]:
    return {key for pick in picks if (key := pick_join_key(pick)) is not None}


def available_pool(board: pl.DataFrame, picks: list[dict[str, Any]]) -> pl.DataFrame:
    """`board` (SPEC §9.7's `player`/`position` columns) minus everyone
    already drafted. Never mutates the board's own row order (already VOR
    descending); the available pool at any moment is just a filtered view.
    """
    taken = drafted_join_keys(picks)
    with_key = board.with_columns(
        (
            pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key")
    )
    if taken:
        with_key = with_key.filter(~pl.col("join_key").is_in(list(taken)))
    return with_key.drop("join_key")


def best_available(pool: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """SPEC §9.8 priority #1: best available by VOR, with tier and
    opportunity cost -- `pool` is already VOR-sorted (inherited from the
    board), so this is just the top slice.
    """
    return pool.head(n)


def tier_depth_remaining(pool: pl.DataFrame) -> pl.DataFrame:
    """SPEC §9.8 priority #2 -- "the single most actionable number on the
    screen": how many players are left in each (position, tier).
    """
    return (
        pool.group_by(["position", "tier"])
        .agg(pl.len().alias("remaining"))
        .sort(["position", "tier"])
    )


def current_tier_summary(pool: pl.DataFrame) -> pl.DataFrame:
    """SPEC §9.8's own worked example -- "3 left in RB tier 4" -- one row
    per position: the *lowest* (best) tier that still has players in it,
    and how many are left there. This, not the full multi-tier breakdown
    `tier_depth_remaining` returns, is what should drive an on-the-clock
    positional decision.
    """
    depth = tier_depth_remaining(pool)
    return depth.group_by("position", maintain_order=True).first()


def positional_run(
    picks: list[dict[str, Any]], *, window: int = RUN_WINDOW
) -> dict[str, dict[str, float | bool]]:
    """SPEC §9.8 priority #3: flag a position going at >= 2x its baseline
    rate over the last `window` picks. Baseline = that position's share of
    every pick made so far in this draft (confirmed with the project owner
    -- SPEC doesn't define "expected mix" and there's no historical-draft
    dataset to derive one from; a recent-window-vs-running-average is the
    standard technique and needs no external data). A position with zero
    picks so far never appears here at all (nothing to compare against --
    "2x zero" isn't a meaningful signal).
    """
    positions = [
        pick["metadata"]["position"] for pick in picks if pick.get("metadata", {}).get("position")
    ]
    if not positions:
        return {}

    recent = positions[-window:]
    baseline_counts = Counter(positions)
    recent_counts = Counter(recent)
    total = len(positions)
    recent_total = len(recent)

    result: dict[str, dict[str, float | bool]] = {}
    for position, baseline_count in baseline_counts.items():
        baseline_rate = baseline_count / total
        recent_rate = recent_counts.get(position, 0) / recent_total
        result[position] = {
            "baseline_rate": baseline_rate,
            "recent_rate": recent_rate,
            "is_run": recent_rate >= RUN_MULTIPLIER * baseline_rate,
        }
    return result


def starting_lineup_gaps(
    my_picks: list[dict[str, Any]], league_format: LeagueFormat
) -> dict[str, int]:
    """SPEC §9.8 priority #4: starting-lineup slots not yet filled by
    `my_picks`, given `LeagueFormat`. Dedicated positions are filled first;
    whatever's left over from `my_picks` after that competes for flex slots
    in `league_format.flex_eligible`'s own order. Same known limitation as
    `tools/vor.py`'s replacement-level iteration: a league with more than
    one *simultaneously active* flex type could, in principle, want a
    player counted against either -- not exercised by either real league
    here (neither has more than one active flex type), so not solved.
    Returns `{slot: count_still_needed}`, positions/flex-types already
    fully filled are omitted.
    """
    drafted_positions = [
        pick["metadata"]["position"]
        for pick in my_picks
        if pick.get("metadata", {}).get("position")
    ]

    remaining_starters = dict(league_format.starters)
    unassigned: list[str] = []
    for position in drafted_positions:
        if remaining_starters.get(position, 0) > 0:
            remaining_starters[position] -= 1
        else:
            unassigned.append(position)

    remaining_flex = dict(league_format.flex_slots)
    for flex_type, eligible_positions in league_format.flex_eligible.items():
        slots_left = remaining_flex.get(flex_type, 0)
        still_unassigned: list[str] = []
        for position in unassigned:
            if slots_left > 0 and position in eligible_positions:
                slots_left -= 1
            else:
                still_unassigned.append(position)
        remaining_flex[flex_type] = slots_left
        unassigned = still_unassigned

    gaps = {position: count for position, count in remaining_starters.items() if count > 0}
    gaps.update({flex_type: count for flex_type, count in remaining_flex.items() if count > 0})
    return gaps


__all__ = [
    "RUN_MULTIPLIER",
    "RUN_WINDOW",
    "available_pool",
    "best_available",
    "current_tier_summary",
    "drafted_join_keys",
    "pick_join_key",
    "positional_run",
    "starting_lineup_gaps",
    "tier_depth_remaining",
]
