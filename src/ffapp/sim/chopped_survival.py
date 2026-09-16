"""Monte Carlo elimination-risk estimates for chopped leagues."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup

TEAM_SCHEMA = {
    "roster_id": pl.Int64,
    "projected_score": pl.Float64,
    "score_sd": pl.Float64,
    "elimination_probability": pl.Float64,
    "survival_probability": pl.Float64,
}

IMPACT_SCHEMA = {
    "sleeper_id": pl.String,
    "baseline_survival_probability": pl.Float64,
    "survival_probability_after": pl.Float64,
    "survival_probability_gain": pl.Float64,
    "elimination_probability_after": pl.Float64,
    "projected_score_gain": pl.Float64,
}


def _lineup_distribution(
    sleeper_ids: list[str],
    by_sleeper_id: dict[str, dict[str, Any]],
    fmt: LeagueFormat,
) -> tuple[float, float]:
    players: list[PlayerProjection] = []
    sd_by_player: dict[str, float] = {}
    for sleeper_id in sleeper_ids:
        row = by_sleeper_id.get(str(sleeper_id))
        if row is None or row.get("current_week_projection") is None:
            continue
        mean = max(0.0, float(row["current_week_projection"]))
        player_id = str(row["player_id"])
        players.append(PlayerProjection(player_id, str(row["position"]), mean, mean, mean))
        sd_by_player[player_id] = max(2.5, mean * 0.30)
    lineup = optimal_lineup(players, fmt)
    chosen_ids = list(lineup.slots.values())
    variance = sum(sd_by_player[player_id] ** 2 for player_id in chosen_ids)
    return lineup.total_points, variance**0.5


def simulate_chopped_survival(
    rosters: list[dict[str, Any]],
    player_values: pl.DataFrame,
    my_roster_id: int,
    fmt: LeagueFormat,
    candidate_ids: list[str],
    *,
    n_sims: int = 20_000,
    seed: int = 2026,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate next-chop risk and each candidate's effect using common random draws."""
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    active = [roster for roster in rosters if roster.get("players")]
    roster_ids = [int(roster["roster_id"]) for roster in active]
    if my_roster_id not in roster_ids:
        raise ValueError("Your roster is not active in this chopped league")
    by_sleeper_id = {
        str(row["sleeper_id"]): row
        for row in player_values.filter(pl.col("sleeper_id").is_not_null()).iter_rows(named=True)
    }
    distributions = [
        _lineup_distribution(
            [str(value) for value in (roster.get("players") or [])], by_sleeper_id, fmt
        )
        for roster in active
    ]
    means = np.asarray([value[0] for value in distributions], dtype=float)
    sds = np.asarray([value[1] for value in distributions], dtype=float)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_sims, len(active)))
    scores = np.maximum(0.0, means + noise * sds)
    eliminated = np.argmin(scores, axis=1)
    elimination = np.bincount(eliminated, minlength=len(active)) / n_sims
    team_rows = [
        {
            "roster_id": roster_id,
            "projected_score": float(means[index]),
            "score_sd": float(sds[index]),
            "elimination_probability": float(elimination[index]),
            "survival_probability": float(1.0 - elimination[index]),
        }
        for index, roster_id in enumerate(roster_ids)
    ]
    my_index = roster_ids.index(my_roster_id)
    my_players = [str(value) for value in (active[my_index].get("players") or [])]
    baseline_survival = 1.0 - float(elimination[my_index])
    impact_rows: list[dict[str, object]] = []
    for sleeper_id in candidate_ids:
        new_mean, new_sd = _lineup_distribution([*my_players, str(sleeper_id)], by_sleeper_id, fmt)
        candidate_scores = scores.copy()
        candidate_scores[:, my_index] = np.maximum(0.0, new_mean + noise[:, my_index] * new_sd)
        risk_after = float(np.mean(np.argmin(candidate_scores, axis=1) == my_index))
        survival_after = 1.0 - risk_after
        impact_rows.append(
            {
                "sleeper_id": str(sleeper_id),
                "baseline_survival_probability": baseline_survival,
                "survival_probability_after": survival_after,
                "survival_probability_gain": survival_after - baseline_survival,
                "elimination_probability_after": risk_after,
                "projected_score_gain": new_mean - float(means[my_index]),
            }
        )
    teams = pl.DataFrame(team_rows, schema=TEAM_SCHEMA)
    impacts = (
        pl.DataFrame(impact_rows, schema=IMPACT_SCHEMA)
        if impact_rows
        else pl.DataFrame(schema=IMPACT_SCHEMA)
    )
    return teams.sort("elimination_probability", descending=True), impacts.sort(
        "survival_probability_gain", descending=True
    )


__all__ = ["simulate_chopped_survival"]
