"""Real draft pick ownership and snake pick numbers (SPEC.md §9.6; task 0.11).

SPEC §9.6 says "given your draft slot and league size, compute your pick
numbers" -- written for a plain redraft where every team keeps its own slot
every round. The primary league trades picks: confirmed live, its 2026 draft
has 45 traded-pick records, and this project's own draft slot (roster 7) has
lost 7 of its own picks and gained 11 traded in from other rosters. A
fixed-slot formula would be silently wrong for most of the draft, so pick
ownership is resolved from Sleeper's real `traded_picks` data instead of
assumed from `draft_order` alone.

Sleeper's `/league/{id}/traded_picks` returns one record per
(season, round, roster_id) -- `roster_id` identifies the pick by its
*original* owner, `owner_id` is who currently holds it. Confirmed live
against the primary league: no (season, round, roster_id) key appears
twice, so `owner_id` is always the final owner, never one hop in a longer
trade chain. A pick traded more than once would need chain resolution this
module doesn't implement -- not exercised by any real data seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TradedPick:
    season: str
    round: int
    roster_id: int
    owner_id: int
    previous_owner_id: int


def parse_traded_picks(raw: list[dict[str, Any]]) -> list[TradedPick]:
    """Parse Sleeper's raw `/league/{id}/traded_picks` payload."""
    return [
        TradedPick(
            season=str(record["season"]),
            round=int(record["round"]),
            roster_id=int(record["roster_id"]),
            owner_id=int(record["owner_id"]),
            previous_owner_id=int(record["previous_owner_id"]),
        )
        for record in raw
    ]


def roster_id_by_slot(draft_order: dict[str, int], rosters: list[dict[str, Any]]) -> dict[int, int]:
    """slot (1..n_teams) -> roster_id, joining Sleeper's `draft_order` (keyed
    by user_id) through each roster's own `owner_id`. A roster with no
    owner_id, or an owner_id absent from draft_order (e.g. an orphaned/co-
    owned roster), is silently skipped -- SPEC §9.6 has no defined behaviour
    for a roster with no draft slot at all.
    """
    result: dict[int, int] = {}
    for roster in rosters:
        owner_id = roster.get("owner_id")
        roster_id = roster.get("roster_id")
        if owner_id is None or roster_id is None:
            continue
        slot = draft_order.get(str(owner_id))
        if slot is not None:
            result[int(slot)] = int(roster_id)
    return result


def snake_pick_number(round_num: int, slot: int, n_teams: int) -> int:
    """Overall pick number (1-indexed) for (round, slot) in a standard snake
    draft -- odd rounds run 1..n_teams, even rounds reverse. The pick's
    *identity* (which slot it is) never moves when it's traded; only who
    makes it does.
    """
    if round_num % 2 == 1:
        return (round_num - 1) * n_teams + slot
    return (round_num - 1) * n_teams + (n_teams - slot + 1)


def pick_owner(
    round_num: int,
    slot: int,
    roster_by_slot: dict[int, int],
    traded_picks: list[TradedPick],
    *,
    season: str,
) -> int:
    """The roster_id who will actually make this (round, slot) pick, after
    trades. Falls back to the slot's original owner if it was never traded.
    """
    base_owner = roster_by_slot[slot]
    for pick in traded_picks:
        if pick.season == season and pick.round == round_num and pick.roster_id == base_owner:
            return pick.owner_id
    return base_owner


def resolve_my_roster_id(user_id: str, rosters: list[dict[str, Any]]) -> int:
    """The roster_id owned by `user_id` (SPEC §9.6's "given your draft
    slot" -- resolved here from Sleeper's own data rather than typed in by
    hand, since `config/settings.yml`'s `sleeper.username` already resolves
    to a `user_id` via `ingest/sleeper.fetch_user`).

    Raises ValueError if no roster in `rosters` is owned by this user_id --
    a config/account mismatch a human needs to see, not a silent None.
    """
    for roster in rosters:
        if str(roster.get("owner_id")) == str(user_id):
            return int(roster["roster_id"])
    raise ValueError(f"No roster in this league is owned by user_id={user_id!r}")


def my_pick_numbers(
    my_roster_id: int,
    *,
    draft_order: dict[str, int],
    rosters: list[dict[str, Any]],
    traded_picks: list[TradedPick],
    n_teams: int,
    num_rounds: int,
    season: str,
) -> list[int]:
    """Every overall pick number `my_roster_id` will actually make this
    draft, sorted ascending, accounting for trades in both directions (picks
    given away are excluded; picks acquired from other rosters are
    included).
    """
    roster_by_slot = roster_id_by_slot(draft_order, rosters)
    picks = [
        snake_pick_number(round_num, slot, n_teams)
        for round_num in range(1, num_rounds + 1)
        for slot in range(1, n_teams + 1)
        if pick_owner(round_num, slot, roster_by_slot, traded_picks, season=season) == my_roster_id
    ]
    return sorted(picks)


__all__ = [
    "TradedPick",
    "my_pick_numbers",
    "parse_traded_picks",
    "pick_owner",
    "resolve_my_roster_id",
    "roster_id_by_slot",
    "snake_pick_number",
]
