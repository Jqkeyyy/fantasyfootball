"""Season simulator (SPEC.md §13.4; task 2.4).

Monte Carlo over the remainder of a season, composing task 2.1 (lineup
optimiser), 2.2 (correlated weekly simulation), and 2.3 (injury hazard,
via `simulate_availability`'s own persistence mechanic) exactly as
SPEC's own pseudocode lays out:

    FOR sim in 1..season_sims:
        FOR week in remaining_weeks:
            sample availability (with persistence)
            sample correlated weekly scores
            set each team's optimal lineup from its *projected* points
                (NOT the sampled actuals)
            record head-to-head result against the league schedule
        determine playoff seeding, simulate the bracket
        record: wins, playoff berth, title

SPEC's own warning, taken literally: "That parenthetical is the most
commonly botched detail in fantasy season simulators. Lineups are set on
expectation; results are drawn from the sample." Each team's lineup
(`simulate_team_week_totals`) is computed exactly once, from
`SimPlayer.mean`/the fitted quantile grid -- never from a simulated
draw -- and the *same* lineup is then scored against that week's own
sampled, possibly-zeroed-by-injury actuals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import Lineup, PlayerProjection, optimal_lineup
from ffapp.sim.week import PlayerMarginal, marginal_ppf, simulate_week


@dataclass(frozen=True)
class SimPlayer:
    """One player's real inputs to the season simulator -- SPEC's own
    hurdle architecture (§11.2), composed rather than re-derived:
    `mean` is task 1.15's own Part-A/B mean estimate; `alphas`/
    `quantile_values` are task 1.16's own `mixture_with_p_active` output
    (already zero-inflated); `p_miss` is task 2.3's own weekly hazard
    probability. `median`/`ceiling` (task 2.1's own `PlayerProjection`
    fields) are derived from the quantile grid, not asked for twice."""

    player_id: str
    position: str
    team: str
    opponent_team: str | None
    mean: float
    alphas: Sequence[float]
    quantile_values: Sequence[float]
    p_miss: float


# `to_projection`/`to_marginal` are public (not `_`-prefixed) because
# task 2.5's start/sit assistant is a second real caller needing the
# exact same SimPlayer -> PlayerProjection/PlayerMarginal shape this
# module already derives -- promoted rather than duplicated, matching
# `sim.lineup.slot_instances`' own promotion for the same reason.


@dataclass(frozen=True)
class Roster:
    team_id: str
    players: Sequence[SimPlayer]


@dataclass(frozen=True)
class Matchup:
    week: int
    home: str
    away: str


@dataclass(frozen=True)
class SeasonSimResult:
    """SPEC's own literal `OUTPUT per team` shape."""

    expected_wins: dict[str, float]
    p_playoffs: dict[str, float]
    p_title: dict[str, float]


def to_projection(player: SimPlayer) -> PlayerProjection:
    median = float(marginal_ppf(np.array([0.5]), player.alphas, player.quantile_values)[0])
    ceiling = float(player.quantile_values[-1])
    return PlayerProjection(
        player_id=player.player_id,
        position=player.position,
        mean=player.mean,
        median=median,
        ceiling=ceiling,
    )


def to_marginal(player: SimPlayer) -> PlayerMarginal:
    return PlayerMarginal(
        player_id=player.player_id,
        position=player.position,
        team=player.team,
        opponent_team=player.opponent_team,
        alphas=player.alphas,
        quantile_values=player.quantile_values,
    )


def simulate_availability(
    p_miss: np.ndarray,
    *,
    season_sims: int,
    recovery_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """SPEC §13.4: "sample availability per player from injury hazard
    (with persistence: a multi-week injury keeps the player out for a
    sampled duration, not resampled independently each week)."

    `p_miss`: `(n_weeks, n_players)` -- this week's real hazard
    probability per player (task 2.3's own `p_miss[player, week]`; a
    caller with one constant rate per player broadcasts it across weeks
    itself). Returns `(season_sims, n_weeks, n_players)`, `True` =
    available. A player only re-rolls `p_miss` once they're not already
    serving a sampled multi-week absence -- `remaining_out` carries that
    countdown forward week to week within each simulated path.
    """
    n_weeks, n_players = p_miss.shape
    available = np.ones((season_sims, n_weeks, n_players), dtype=bool)
    remaining_out = np.zeros((season_sims, n_players), dtype=int)

    for week in range(n_weeks):
        still_out = remaining_out > 0
        available[:, week, :] = ~still_out
        remaining_out = np.where(still_out, remaining_out - 1, remaining_out)

        eligible = ~still_out
        draws = rng.random((season_sims, n_players))
        new_miss = eligible & (draws < p_miss[week, :][None, :])
        available[:, week, :] = np.where(new_miss, False, available[:, week, :])

        durations = rng.geometric(recovery_prob, size=(season_sims, n_players))
        remaining_out = np.where(new_miss, np.maximum(durations - 1, 0), remaining_out)

    return available


def simulate_team_week_totals(
    teams: Sequence[Roster],
    fmt: LeagueFormat,
    correlation: CorrelationSettings,
    *,
    remaining_weeks: Sequence[int],
    season_sims: int,
    recovery_prob: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Lineup]]:
    """The per-week engine SPEC's pseudocode describes, without the
    schedule/standings/playoff bookkeeping `simulate_season` adds on
    top -- exposed separately so "lineups are set from projections, not
    samples" is directly testable in isolation.

    Each team's lineup is computed exactly once, from `SimPlayer.mean`/
    the quantile grid (never from a sampled draw) -- SPEC's own warning,
    taken literally. Returns `{team_id: (season_sims, n_weeks) totals}`
    plus the lineups actually used, for inspection.
    """
    all_players = [player for team in teams for player in team.players]
    player_index = {player.player_id: i for i, player in enumerate(all_players)}
    marginals = [to_marginal(player) for player in all_players]

    n_weeks = len(remaining_weeks)
    p_miss = np.tile(np.array([player.p_miss for player in all_players]), (n_weeks, 1))
    availability = simulate_availability(
        p_miss, season_sims=season_sims, recovery_prob=recovery_prob, rng=rng
    )

    lineups = {
        team.team_id: optimal_lineup([to_projection(p) for p in team.players], fmt)
        for team in teams
    }

    totals = {team.team_id: np.zeros((season_sims, n_weeks)) for team in teams}
    for week_idx in range(n_weeks):
        week_scores = simulate_week(marginals, correlation, week_sims=season_sims, rng=rng)
        week_available = availability[:, week_idx, :]
        actual_scores = np.where(week_available, week_scores, 0.0)
        for team in teams:
            starter_idx = [player_index[pid] for pid in lineups[team.team_id].slots.values()]
            if starter_idx:
                totals[team.team_id][:, week_idx] = actual_scores[:, starter_idx].sum(axis=1)

    return totals, lineups


def _seed_order(wins: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Per-sim descending rank by (wins, points) -- wins dominate via a
    scaling factor far larger than any realistic points total, so this
    is equivalent to a real lexicographic sort without needing a
    per-row `lexsort`."""
    combined = wins.astype(float) * 1e6 + points
    return np.argsort(-combined, axis=1)


def simulate_season(
    teams: Sequence[Roster],
    schedule: Sequence[Matchup],
    fmt: LeagueFormat,
    correlation: CorrelationSettings,
    *,
    remaining_weeks: Sequence[int],
    playoff_week_start: int,
    n_playoff_teams: int,
    season_sims: int,
    recovery_prob: float = 0.5,
    rng: np.random.Generator,
) -> SeasonSimResult:
    """SPEC §13.4's full pseudocode. `n_playoff_teams` must be a power of
    two -- a real single-elimination bracket, not a placeholder; SPEC
    doesn't specify bye-handling for a non-power-of-two field, so that
    case fails loudly rather than guessing a seeding rule."""
    if n_playoff_teams < 2 or (n_playoff_teams & (n_playoff_teams - 1)) != 0:
        raise ValueError("n_playoff_teams must be a power of 2 >= 2")
    if n_playoff_teams > len(teams):
        raise ValueError("n_playoff_teams cannot exceed the number of teams")

    totals, _ = simulate_team_week_totals(
        teams,
        fmt,
        correlation,
        remaining_weeks=remaining_weeks,
        season_sims=season_sims,
        recovery_prob=recovery_prob,
        rng=rng,
    )
    week_to_idx = {week: i for i, week in enumerate(remaining_weeks)}
    team_ids = [team.team_id for team in teams]

    wins = {team_id: np.zeros(season_sims, dtype=int) for team_id in team_ids}
    for matchup in schedule:
        if matchup.week >= playoff_week_start or matchup.week not in week_to_idx:
            continue
        idx = week_to_idx[matchup.week]
        home_scores = totals[matchup.home][:, idx]
        away_scores = totals[matchup.away][:, idx]
        home_won = home_scores > away_scores
        wins[matchup.home] += home_won.astype(int)
        wins[matchup.away] += (~home_won).astype(int)

    regular_idx = [i for week, i in week_to_idx.items() if week < playoff_week_start]
    points = {
        team_id: totals[team_id][:, regular_idx].sum(axis=1)
        if regular_idx
        else np.zeros(season_sims)
        for team_id in team_ids
    }

    win_matrix = np.stack([wins[t] for t in team_ids], axis=1)
    points_matrix = np.stack([points[t] for t in team_ids], axis=1)
    seed_order = _seed_order(win_matrix, points_matrix)

    made_playoffs = np.zeros((season_sims, len(team_ids)), dtype=bool)
    current = seed_order[:, :n_playoff_teams].copy()
    np.put_along_axis(made_playoffs, current, True, axis=1)

    playoff_weeks = sorted(week for week in remaining_weeks if week >= playoff_week_start)
    rounds = n_playoff_teams.bit_length() - 1
    if len(playoff_weeks) < rounds:
        raise ValueError(
            f"need {rounds} playoff week(s) in remaining_weeks for a {n_playoff_teams}-team "
            f"bracket, found {len(playoff_weeks)}"
        )

    totals_tensor = np.stack([totals[t] for t in team_ids], axis=1)  # (sims, n_teams, n_weeks)
    for round_i in range(rounds):
        week_idx = week_to_idx[playoff_weeks[round_i]]
        week_scores = totals_tensor[:, :, week_idx]
        n_active = current.shape[1]
        next_round = np.zeros((season_sims, n_active // 2), dtype=int)
        for pair_i in range(n_active // 2):
            a_idx = current[:, pair_i]
            b_idx = current[:, n_active - 1 - pair_i]
            a_scores = np.take_along_axis(week_scores, a_idx[:, None], axis=1)[:, 0]
            b_scores = np.take_along_axis(week_scores, b_idx[:, None], axis=1)[:, 0]
            winner_is_a = a_scores >= b_scores
            next_round[:, pair_i] = np.where(winner_is_a, a_idx, b_idx)
        current = next_round

    won_title = np.zeros((season_sims, len(team_ids)), dtype=bool)
    np.put_along_axis(won_title, current, True, axis=1)

    return SeasonSimResult(
        expected_wins={t: float(wins[t].mean()) for t in team_ids},
        p_playoffs={t: float(made_playoffs[:, i].mean()) for i, t in enumerate(team_ids)},
        p_title={t: float(won_title[:, i].mean()) for i, t in enumerate(team_ids)},
    )


__all__ = [
    "Matchup",
    "Roster",
    "SeasonSimResult",
    "SimPlayer",
    "simulate_availability",
    "simulate_season",
    "simulate_team_week_totals",
    "to_marginal",
    "to_projection",
]
