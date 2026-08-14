"""Mobile draft page logic (SPEC-ADDENDUM-03.md §C; task 0.15).

Pure, pytest-testable functions only -- card-building, position filtering,
and the above-the-fold summary. `pages/5_Draft_Mobile.py` is thin glue on
top (matching every other page's own precedent): fetch or replay the
current picks, call `draft.live.available_pool`/`tier_depth_remaining`/
`current_tier_summary` for the raw numbers (task 0.14, unmodified -- this
is a new view over the same live-draft state, not a new pipeline), then
hand the result here to shape into what actually renders on a phone
screen.

Every design choice below is ADDENDUM-03 §C's own literal wording, not a
judgment call: cards not a dataframe, 20-30 players not 300, tier depth
(not matchup grade or anything else) is the number that drives an
on-the-clock decision.
"""

from __future__ import annotations

import polars as pl

DEFAULT_CARD_COUNT = 25
POSITION_FILTER_ORDER = ["ALL", "QB", "RB", "WR", "TE", "K", "DST"]


def filter_pool_by_position(pool: pl.DataFrame, position: str | None) -> pl.DataFrame:
    """`None` or `"ALL"` means no filter -- matches every other page's own
    "empty filter shows everything" convention (`draft_board_page
    .filter_board`, `weekly_rankings_page.filter_rankings`)."""
    if position is None or position == "ALL":
        return pool
    return pool.filter(pl.col("position") == position)


def _tier_remaining_lookup(tier_depth: pl.DataFrame) -> dict[tuple[str, int], int]:
    return {
        (row["position"], row["tier"]): row["remaining"] for row in tier_depth.iter_rows(named=True)
    }


def format_why_line(row: dict[str, object], tier_remaining: dict[tuple[str, int], int]) -> str:
    """ADDENDUM-03 §C's own worked example: `Tier 4 · 3 left · falls to
    you 71%`. Tier depth is this player's own (position, tier) remaining
    count, not the position's current-best-tier count -- a player sitting
    in a worse tier than the position's best should show *their own*
    tier's depth, not someone else's."""
    parts = [f"Tier {row['tier']}"]
    key = (row["position"], row["tier"])
    remaining = tier_remaining.get(key)  # type: ignore[arg-type]
    if remaining is not None:
        parts.append(f"{remaining} left")
    p_avail_next = row.get("p_avail_next")
    if p_avail_next is not None:
        parts.append(f"falls to you {float(p_avail_next):.0%}")  # type: ignore[arg-type]
    return " · ".join(parts)


def build_cards(
    pool: pl.DataFrame, tier_depth: pl.DataFrame, *, n: int = DEFAULT_CARD_COUNT
) -> list[dict[str, object]]:
    """The top `n` available players (already VOR-sorted, `draft.live
    .best_available`'s own ordering) as plain dicts ready for card
    rendering -- `player`/`position`/`team`/`bye_week` for line one,
    `tier`/`vor` for line two, `why_line` for the third."""
    tier_remaining = _tier_remaining_lookup(tier_depth)
    cards = []
    for row in pool.head(n).iter_rows(named=True):
        cards.append(
            {
                "player": row["player"],
                "position": row["position"],
                "team": row["team"],
                "bye_week": row["bye_week"],
                "tier": row["tier"],
                "vor": row["vor"],
                "why_line": format_why_line(row, tier_remaining),
            }
        )
    return cards


def top_line_summary(pool: pl.DataFrame, tier_summary: pl.DataFrame) -> dict[str, object]:
    """The three above-the-fold numbers ADDENDUM-03 §C specifies, in its
    own order: best available by VOR, tier depth remaining per position,
    and survival probability to the next pick -- the best-available
    player's own `p_avail_next`, since that's the one number "survival to
    your next pick" means anything for at the top of the screen (every
    card below already carries its own)."""
    if pool.height == 0:
        return {
            "best_player": None,
            "best_vor": None,
            "best_p_avail_next": None,
            "tier_depth_by_position": [],
        }
    best = pool.row(0, named=True)
    return {
        "best_player": best["player"],
        "best_vor": best["vor"],
        "best_p_avail_next": best.get("p_avail_next"),
        "tier_depth_by_position": tier_summary.sort("position").to_dicts(),
    }


__all__ = [
    "DEFAULT_CARD_COUNT",
    "POSITION_FILTER_ORDER",
    "build_cards",
    "filter_pool_by_position",
    "format_why_line",
    "top_line_summary",
]
