"""Trade analyzer (SPEC.md §14.6; task 2.9).

SPEC's own words: "trade value is not additive because lineups have
fixed slots... Only a lineup-aware simulation captures this." Composes
task 2.4's `simulate_season` directly, exactly per SPEC's own four
steps:

    1. Run the season simulator for the current league state.
    2. Apply the trade. Re-run with identical random seeds (common
       random numbers).
    3. Report the delta for both teams.
    4. Also report the naive value delta (sum of ROS VOR) for reference.

**Common random numbers, and an honest limitation.** "Identical random
seed" here means a *freshly constructed* `np.random.default_rng(seed)`
for each of the two `simulate_season` calls -- reusing one mutated
generator across both would give the second run a completely different
draw sequence, defeating the whole point. This does reduce the variance
of the difference, as SPEC says, but not perfectly: `simulate_week`'s
correlated draw is built from a covariance matrix derived from the
*current* player list (`sim.week.build_correlation_matrix`), so a trade
that changes roster composition changes that matrix, which changes the
linear transform applied to the same underlying standard-normal draws --
the untraded majority of the league still shares highly-correlated
noise with the "before" run, but individual post-trade values are not
byte-identical replays with the traded players' rows swapped out.
Redesigning the simulator's RNG to key draws per-player-identity (true
per-player substreams) would fix this precisely, but that is a real
change to already-shipped, tested code (`sim.week`/`sim.season`, tasks
2.2/2.4) that SPEC's own text for *this* task doesn't ask for --
flagged here rather than silently assumed away, not attempted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.season import Matchup, Roster, SimPlayer, simulate_season


@dataclass(frozen=True)
class TradeProposal:
    team_a: str
    team_b: str
    players_from_a: Sequence[str]  # player_ids team A gives up (to team B)
    players_from_b: Sequence[str]  # player_ids team B gives up (to team A)


@dataclass(frozen=True)
class TeamTradeDelta:
    """SPEC's own literal output shape for one side: "Δ E[wins], Δ
    P(playoffs), Δ P(title), plus the naive VOR delta and a
    plain-language summary of which positions each side gained and
    lost." `position_deltas` is that summary as data (positive = net
    gained, negative = net lost, per position) -- `summary()` renders
    it as the plain-language string SPEC asks for."""

    team_id: str
    delta_expected_wins: float
    delta_p_playoffs: float
    delta_p_title: float
    naive_vor_delta: float
    position_deltas: dict[str, int]

    def summary(self) -> str:
        parts = [
            f"{'+' if delta > 0 else ''}{delta} {position}"
            for position, delta in sorted(self.position_deltas.items())
            if delta != 0
        ]
        return ", ".join(parts) if parts else "no position change"


@dataclass(frozen=True)
class TradeAnalysis:
    """SPEC's own closing note, taken literally: "Show the other team's
    deltas too. A trade that helps you and visibly helps them is a
    trade that gets accepted." Both sides are always present."""

    team_a: TeamTradeDelta
    team_b: TeamTradeDelta


def apply_trade(teams: Sequence[Roster], proposal: TradeProposal) -> list[Roster]:
    """Swap the named players between `proposal.team_a`/`.team_b`'s
    rosters; every other team is returned unchanged. Raises `ValueError`
    if either team_id or any named player_id isn't found on the roster
    it's supposed to be traded from -- CLAUDE.md rule 4's "never
    silently..." principle applied to a malformed trade proposal rather
    than a join."""
    by_id = {team.team_id: team for team in teams}
    for team_id in (proposal.team_a, proposal.team_b):
        if team_id not in by_id:
            raise ValueError(f"Unknown team_id in trade proposal: {team_id!r}")

    a_players = {p.player_id: p for p in by_id[proposal.team_a].players}
    b_players = {p.player_id: p for p in by_id[proposal.team_b].players}
    missing_a = [pid for pid in proposal.players_from_a if pid not in a_players]
    missing_b = [pid for pid in proposal.players_from_b if pid not in b_players]
    if missing_a or missing_b:
        raise ValueError(
            f"Players not found on their proposed side: "
            f"team_a missing {missing_a}, team_b missing {missing_b}"
        )

    new_a = [p for pid, p in a_players.items() if pid not in proposal.players_from_a] + [
        b_players[pid] for pid in proposal.players_from_b
    ]
    new_b = [p for pid, p in b_players.items() if pid not in proposal.players_from_b] + [
        a_players[pid] for pid in proposal.players_from_a
    ]

    result = []
    for team in teams:
        if team.team_id == proposal.team_a:
            result.append(Roster(team_id=team.team_id, players=new_a))
        elif team.team_id == proposal.team_b:
            result.append(Roster(team_id=team.team_id, players=new_b))
        else:
            result.append(team)
    return result


def _position_deltas(gained: Sequence[SimPlayer], lost: Sequence[SimPlayer]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in gained:
        counts[player.position] = counts.get(player.position, 0) + 1
    for player in lost:
        counts[player.position] = counts.get(player.position, 0) - 1
    return {position: delta for position, delta in counts.items() if delta != 0}


def _naive_vor_delta(
    gained: Sequence[SimPlayer], lost: Sequence[SimPlayer], vor_by_player: Mapping[str, float]
) -> float:
    return sum(vor_by_player.get(p.player_id, 0.0) for p in gained) - sum(
        vor_by_player.get(p.player_id, 0.0) for p in lost
    )


def analyze_trade(
    teams: Sequence[Roster],
    schedule: Sequence[Matchup],
    fmt: LeagueFormat,
    correlation: CorrelationSettings,
    proposal: TradeProposal,
    vor_by_player: Mapping[str, float],
    *,
    remaining_weeks: Sequence[int],
    playoff_week_start: int,
    n_playoff_teams: int,
    season_sims: int,
    recovery_prob: float = 0.5,
    rng_seed: int,
) -> TradeAnalysis:
    """SPEC §14.6's full algorithm. `vor_by_player` is SPEC step 4's
    "sum of ROS VOR" input -- an already-computed `tools.vor` table,
    passed in rather than recomputed here, the same "composition, not
    re-derivation" pattern `sim.season`/`sim.startsit` already use for
    their own player-shape inputs."""
    before = simulate_season(
        teams,
        schedule,
        fmt,
        correlation,
        remaining_weeks=remaining_weeks,
        playoff_week_start=playoff_week_start,
        n_playoff_teams=n_playoff_teams,
        season_sims=season_sims,
        recovery_prob=recovery_prob,
        rng=np.random.default_rng(rng_seed),
    )
    traded_teams = apply_trade(teams, proposal)
    after = simulate_season(
        traded_teams,
        schedule,
        fmt,
        correlation,
        remaining_weeks=remaining_weeks,
        playoff_week_start=playoff_week_start,
        n_playoff_teams=n_playoff_teams,
        season_sims=season_sims,
        recovery_prob=recovery_prob,
        rng=np.random.default_rng(rng_seed),  # same seed -- common random numbers
    )

    by_id = {team.team_id: team for team in teams}
    a_roster = {p.player_id: p for p in by_id[proposal.team_a].players}
    b_roster = {p.player_id: p for p in by_id[proposal.team_b].players}
    a_lost = [a_roster[pid] for pid in proposal.players_from_a]
    a_gained = [b_roster[pid] for pid in proposal.players_from_b]
    b_lost = [b_roster[pid] for pid in proposal.players_from_b]
    b_gained = [a_roster[pid] for pid in proposal.players_from_a]

    def _side(
        team_id: str, gained: Sequence[SimPlayer], lost: Sequence[SimPlayer]
    ) -> TeamTradeDelta:
        return TeamTradeDelta(
            team_id=team_id,
            delta_expected_wins=after.expected_wins[team_id] - before.expected_wins[team_id],
            delta_p_playoffs=after.p_playoffs[team_id] - before.p_playoffs[team_id],
            delta_p_title=after.p_title[team_id] - before.p_title[team_id],
            naive_vor_delta=_naive_vor_delta(gained, lost, vor_by_player),
            position_deltas=_position_deltas(gained, lost),
        )

    return TradeAnalysis(
        team_a=_side(proposal.team_a, a_gained, a_lost),
        team_b=_side(proposal.team_b, b_gained, b_lost),
    )


__all__ = [
    "TeamTradeDelta",
    "TradeAnalysis",
    "TradeProposal",
    "analyze_trade",
    "apply_trade",
]
