"""Lineup optimiser (SPEC.md §13.1; task 2.1).

Given a roster's own player projections and a `LeagueFormat`, find the
starting lineup that maximises total value -- a small binary ILP over
(player, slot) pairs, solved exactly with `pulp`'s bundled CBC solver
(no external solver install, no network at solve time). SPEC's own
words: "This handles FLEX, SUPER_FLEX, and multi-flex formats exactly.
Problems of this size solve in milliseconds" -- confirmed live, every
fixture in `tests/test_sim_lineup.py` solves well under that.

`PlayerProjection` isn't defined elsewhere in SPEC.md as a concrete
dataclass (only referenced by name) -- defined here, the one module that
actually consumes it, rather than guessed into a separate shared types
module ahead of a second real caller needing it (CLAUDE.md's
no-premature-abstraction rule). `optimal_lineup_points` (SPEC's own
"expose ... for computing lineup regret in evaluation") reuses the exact
same shape rather than inventing a second one: pass real, realised
points in as `mean` (and `median`/`ceiling`, since they're required
fields) and read `.total_points` off the result -- lineup regret is
`optimal_lineup_points(actual) - <model's own recommended lineup's real
total>`, task 1.13's own deferred metric, revisited now that this
exists.

ILP variable names deliberately don't embed a real `player_id`/slot
label -- a real `player_id` can itself contain a `-` (e.g. an nflverse
`gsis_id` like `00-0034796`), which `pulp`'s own variable-name character
set forbids. Enumerated `x{i}` names sidestep the entire class of bug
rather than sanitising ids per call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pulp

from ffapp.league_format import LeagueFormat

Objective = Literal["mean", "median", "ceiling"]


@dataclass(frozen=True)
class PlayerProjection:
    player_id: str
    position: str
    mean: float
    median: float
    ceiling: float


@dataclass(frozen=True)
class Lineup:
    slots: dict[str, str]  # slot id (e.g. "RB_1", "FLEX_2") -> player_id
    total_points: float


def slot_instances(fmt: LeagueFormat) -> list[tuple[str, list[str]]]:
    """One entry per real starting slot -- a dedicated position slot is
    eligible for exactly that position; a flex slot (only emitted when
    its own count is > 0, i.e. actually active in this league) is
    eligible for `fmt.flex_eligible[slot_type]`. Handles any number of
    simultaneously-active flex types generically -- never hardcodes
    "just FLEX" or "just SUPER_FLEX" (CLAUDE.md rule 5)."""
    slots: list[tuple[str, list[str]]] = []
    for position, count in fmt.starters.items():
        for i in range(count):
            slots.append((f"{position}_{i + 1}", [position]))
    for slot_type, count in fmt.flex_slots.items():
        if count == 0:
            continue
        eligible = fmt.flex_eligible.get(slot_type, [])
        for i in range(count):
            slots.append((f"{slot_type}_{i + 1}", eligible))
    return slots


def _objective_value(player: PlayerProjection, objective: Objective) -> float:
    if objective == "mean":
        return player.mean
    if objective == "median":
        return player.median
    return player.ceiling


def _cbc_solver() -> pulp.COIN_CMD:
    """Prefer PuLP 4's supported CBC entry point, with a PuLP 3 bundle fallback."""
    solver = pulp.COIN_CMD(msg=False)
    if solver.available():
        return solver
    legacy_class = getattr(pulp, "PULP_CBC_CMD", None)
    bundled_path = getattr(legacy_class, "pulp_cbc_path", None)
    return pulp.COIN_CMD(path=bundled_path, msg=False)


def optimal_lineup(
    players: list[PlayerProjection],
    fmt: LeagueFormat,
    objective: Objective = "mean",
) -> Lineup:
    """SPEC §13.1: binary variable per (player, slot), constraints that
    each slot is filled at most once, each player is used at most once,
    and slot eligibility is respected. A short roster remains solvable;
    positive projected values still fill every slot that can be filled."""
    slots = slot_instances(fmt)

    combos = [
        (player, slot_id)
        for slot_id, eligible in slots
        for player in players
        if player.position in eligible
    ]
    problem = pulp.LpProblem("optimal_lineup", pulp.LpMaximize)
    variables = {
        (player.player_id, slot_id): problem.add_variable(f"x{i}", cat="Binary")
        for i, (player, slot_id) in enumerate(combos)
    }

    problem += pulp.lpSum(
        _objective_value(player, objective) * variables[(player.player_id, slot_id)]
        for player, slot_id in combos
    )

    for slot_id, eligible in slots:
        eligible_vars = [
            variables[(player.player_id, slot_id)]
            for player in players
            if player.position in eligible
        ]
        problem += pulp.lpSum(eligible_vars) <= 1

    for player in players:
        player_vars = [
            variables[(player.player_id, slot_id)]
            for slot_id, eligible in slots
            if player.position in eligible
        ]
        if player_vars:
            problem += pulp.lpSum(player_vars) <= 1

    problem.solve(_cbc_solver())

    assigned: dict[str, str] = {}
    total = 0.0
    for player, slot_id in combos:
        if pulp.value(variables[(player.player_id, slot_id)]) == 1:
            assigned[slot_id] = player.player_id
            total += _objective_value(player, objective)

    return Lineup(slots=assigned, total_points=total)


def optimal_lineup_points(actual_points: list[PlayerProjection], fmt: LeagueFormat) -> float:
    """SPEC §13.1: "Also expose optimal_lineup_points(actual_points, fmt)
    for computing lineup regret in evaluation" -- the real, ex-post-best-
    possible lineup total, for `optimal_lineup_points(actual) -
    model_recommended_lineup_points` (task 1.13's deferred lineup-regret
    metric)."""
    return optimal_lineup(actual_points, fmt).total_points


def fills_slots(positions: Sequence[str], fmt: LeagueFormat) -> bool:
    """Whether these starters can each occupy a distinct slot that accepts their
    position -- a bipartite matching by augmenting paths."""
    slots = [eligible for _, eligible in slot_instances(fmt)]
    owner: list[int | None] = [None] * len(slots)

    def place(player: int, seen: set[int]) -> bool:
        for slot, eligible in enumerate(slots):
            if positions[player] not in eligible or slot in seen:
                continue
            seen.add(slot)
            holder = owner[slot]
            if holder is None or place(holder, seen):
                owner[slot] = player
                return True
        return False

    return all(place(player, set()) for player in range(len(positions)))


def legal_swap_pairs(
    incoming: Sequence[str],
    outgoing: Sequence[str],
    starters: Mapping[str, str],
    fmt: LeagueFormat,
) -> list[tuple[str, str | None]]:
    """Pair optimizer changes into start/sit swaps that each leave a legal lineup.

    `incoming` is in priority order (best first), `outgoing` in sit-first order
    (weakest first), and `starters` maps every current starter -- plus each
    incoming player -- to its position. A naive zip of the two lists can tell
    a manager to start a RB over their only TE; this only pairs a swap when
    the resulting starters still fill every slot (a cross-position swap is
    legal exactly when a flex spot can absorb it). Starters at positions this
    format doesn't model (e.g. K when only skill slots are optimised) are
    ignored rather than making every swap look illegal.

    An incoming player who can fill a currently empty slot is paired with
    `None` (nobody needs to sit) before any real starter is benched for them.
    """
    slot_count = len(slot_instances(fmt))
    modelled = {position for _, eligible in slot_instances(fmt) for position in eligible}
    lineup = {
        player_id
        for player_id, position in starters.items()
        if position in modelled and player_id not in incoming
    }
    remaining = list(outgoing)
    pairs: list[tuple[str, str | None]] = []
    for start in incoming:
        filled = lineup | {start}
        if len(lineup) < slot_count and fills_slots(
            [starters[player_id] for player_id in filled], fmt
        ):
            pairs.append((start, None))
            lineup = filled
            continue
        for sit in remaining:
            after = (lineup - {sit}) | {start}
            if fills_slots([starters[player_id] for player_id in after], fmt):
                pairs.append((start, sit))
                remaining.remove(sit)
                lineup = after
                break
    return pairs


__all__ = [
    "Lineup",
    "Objective",
    "PlayerProjection",
    "fills_slots",
    "legal_swap_pairs",
    "optimal_lineup",
    "optimal_lineup_points",
    "slot_instances",
]
