"""Start/sit assistant (SPEC.md §14.3; task 2.5).

SPEC's own words: "The feature most tools get wrong. The correct
objective is probability of winning your matchup, not expected
points." Composes task 2.1's lineup optimiser and task 2.4's
`SimPlayer`/`to_projection`/`to_marginal` shapes with task 2.2's
correlated weekly simulation, exactly per SPEC's own five steps:

    1. Determine opponent's likely lineup (optimal by projection).
    2. Simulate opponent's total score distribution (§13.2).
    3. Enumerate candidate lineups for your roster: the
       projection-optimal lineup, plus all single-swap variants at
       each flex-eligible slot.
    4. For each candidate, jointly simulate your total with the
       opponent's (a shared correlation matrix -- same-game players on
       both rosters matter).
    5. Rank candidates by P(win).

"Flex-eligible slot" is read literally as SPEC's own established term
(`LeagueFormat.flex_eligible`/`flex_slots` -- FLEX/SUPER_FLEX/REC_FLEX),
not any slot with bench depth: a dedicated position slot (e.g. RB1) has
no swap variant here, since the module composes existing rosterable
slots rather than inventing a second start/sit mode SPEC doesn't
describe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import Lineup, optimal_lineup, slot_instances
from ffapp.sim.season import SimPlayer, to_marginal, to_projection
from ffapp.sim.week import simulate_week


@dataclass(frozen=True)
class CandidateLineup:
    """One lineup under consideration: the projection-optimal lineup
    itself (`swapped_slot is None`), or a single-swap variant at one
    flex-eligible slot (SPEC §14.3 step 3)."""

    lineup: Lineup
    swapped_slot: str | None
    swapped_out: str | None
    swapped_in: str | None


@dataclass(frozen=True)
class StartSitCandidate:
    """A `CandidateLineup` plus its simulated outcome against the
    opponent, and its deltas against the projection-optimal baseline --
    SPEC's own required output: "a table of each considered swap
    showing Δ projected points and Δ P(win)."""

    lineup: Lineup
    swapped_slot: str | None
    swapped_out: str | None
    swapped_in: str | None
    delta_projected_points: float
    p_win: float
    delta_p_win: float


@dataclass(frozen=True)
class StartSitResult:
    recommended: StartSitCandidate
    candidates: list[StartSitCandidate]  # ranked by P(win) descending (SPEC step 5)


def enumerate_candidate_lineups(
    roster: Sequence[SimPlayer], fmt: LeagueFormat
) -> list[CandidateLineup]:
    """SPEC §14.3 step 3. The projection-optimal lineup, plus one
    variant per (flex-eligible slot, eligible bench player) pair -- the
    incumbent at that slot swapped out for a player not already
    starting elsewhere in the lineup."""
    projections = {p.player_id: to_projection(p) for p in roster}
    baseline = optimal_lineup(list(projections.values()), fmt)
    candidates = [
        CandidateLineup(lineup=baseline, swapped_slot=None, swapped_out=None, swapped_in=None)
    ]

    used = set(baseline.slots.values())
    for slot_id, eligible_positions in slot_instances(fmt):
        slot_type = slot_id.rsplit("_", 1)[0]
        if slot_type not in fmt.flex_eligible:
            continue  # dedicated position slot -- not a flex decision
        incumbent = baseline.slots.get(slot_id)
        if incumbent is None:
            continue
        for player_id, proj in projections.items():
            if player_id in used or proj.position not in eligible_positions:
                continue
            new_slots = dict(baseline.slots)
            new_slots[slot_id] = player_id
            new_total = baseline.total_points - projections[incumbent].mean + proj.mean
            candidates.append(
                CandidateLineup(
                    lineup=Lineup(slots=new_slots, total_points=new_total),
                    swapped_slot=slot_id,
                    swapped_out=incumbent,
                    swapped_in=player_id,
                )
            )
    return candidates


def evaluate_start_sit(
    my_roster: Sequence[SimPlayer],
    opponent_roster: Sequence[SimPlayer],
    fmt: LeagueFormat,
    correlation: CorrelationSettings,
    *,
    week_sims: int,
    rng: np.random.Generator,
) -> StartSitResult:
    """SPEC §14.3's full algorithm. The opponent's own lineup is fixed
    (their projection-optimal one, step 1) and simulated jointly with
    each of my candidates in turn (step 4) -- `simulate_week` is handed
    the combined player list each time, so its own pairwise-correlation
    rules (`sim.week.build_correlation_matrix`) naturally cover
    same-game players regardless of which fantasy roster they're on,
    which is exactly SPEC's "shared correlation matrix" requirement."""
    opponent_lineup = optimal_lineup([to_projection(p) for p in opponent_roster], fmt)
    opponent_marginals = {p.player_id: to_marginal(p) for p in opponent_roster}
    opponent_starters = [opponent_marginals[pid] for pid in opponent_lineup.slots.values()]

    my_marginals = {p.player_id: to_marginal(p) for p in my_roster}
    candidates = enumerate_candidate_lineups(my_roster, fmt)
    baseline_points = candidates[0].lineup.total_points

    evaluated: list[StartSitCandidate] = []
    for candidate in candidates:
        my_starters = [my_marginals[pid] for pid in candidate.lineup.slots.values()]
        scores = simulate_week(
            [*my_starters, *opponent_starters], correlation, week_sims=week_sims, rng=rng
        )
        my_total = scores[:, : len(my_starters)].sum(axis=1)
        opp_total = scores[:, len(my_starters) :].sum(axis=1)
        p_win = float((my_total > opp_total).mean())
        evaluated.append(
            StartSitCandidate(
                lineup=candidate.lineup,
                swapped_slot=candidate.swapped_slot,
                swapped_out=candidate.swapped_out,
                swapped_in=candidate.swapped_in,
                delta_projected_points=candidate.lineup.total_points - baseline_points,
                p_win=p_win,
                delta_p_win=0.0,  # filled in below, once the baseline's own p_win is known
            )
        )

    baseline_p_win = evaluated[0].p_win
    ranked = sorted(
        (
            StartSitCandidate(
                lineup=c.lineup,
                swapped_slot=c.swapped_slot,
                swapped_out=c.swapped_out,
                swapped_in=c.swapped_in,
                delta_projected_points=c.delta_projected_points,
                p_win=c.p_win,
                delta_p_win=c.p_win - baseline_p_win,
            )
            for c in evaluated
        ),
        key=lambda c: c.p_win,
        reverse=True,
    )

    return StartSitResult(recommended=ranked[0], candidates=ranked)


__all__ = [
    "CandidateLineup",
    "StartSitCandidate",
    "StartSitResult",
    "enumerate_candidate_lineups",
    "evaluate_start_sit",
]
