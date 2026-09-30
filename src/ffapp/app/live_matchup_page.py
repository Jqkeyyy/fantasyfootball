"""Pure calculations for the live matchup command center."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TypedDict

import polars as pl


class PlayerState(TypedDict):
    player: str
    position: str
    team: str
    projection: float
    floor: float
    ceiling: float
    points: float
    kickoff: str | None


class SideProjection(TypedDict):
    live: float
    remaining_mean: float
    remaining_variance: float
    projected_final: float
    remaining: list[PlayerState]
    active_or_final: list[PlayerState]
    upside: list[PlayerState]
    downside: list[PlayerState]


class MatchupProjection(TypedDict):
    mine: SideProjection
    opponent: SideProjection
    projected_margin: float
    win_probability: float
    updated_at: str


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def matchup_projection(
    rankings: pl.DataFrame,
    my_starters: Sequence[str],
    opponent_starters: Sequence[str],
    current_points: Mapping[str, float],
    kickoff_by_team: Mapping[str, datetime],
    *,
    now: datetime | None = None,
) -> MatchupProjection:
    """Combine live Sleeper points with forecasts for games that have not kicked off."""
    current = now or datetime.now(UTC)
    by_id = {str(row["player_id"]): row for row in rankings.iter_rows(named=True)}

    def side(player_ids: Sequence[str]) -> SideProjection:
        live = sum(float(current_points.get(player_id, 0.0)) for player_id in player_ids)
        remaining_mean = 0.0
        remaining_variance = 0.0
        remaining: list[PlayerState] = []
        active: list[PlayerState] = []
        upside: list[PlayerState] = []
        downside: list[PlayerState] = []
        for player_id in player_ids:
            row = by_id.get(str(player_id))
            if row is None or row.get("proj_mean") is None:
                continue
            kickoff = kickoff_by_team.get(str(row.get("team") or ""))
            item: PlayerState = {
                "player": str(row.get("player_name") or player_id),
                "position": str(row.get("position") or ""),
                "team": str(row.get("team") or ""),
                "projection": float(row["proj_mean"]),
                "floor": float(row.get("floor") or row["proj_mean"]),
                "ceiling": float(row.get("ceiling") or row["proj_mean"]),
                "points": float(current_points.get(player_id, 0.0)),
                "kickoff": kickoff.isoformat() if kickoff else None,
            }
            if kickoff is not None and kickoff > current:
                remaining.append(item)
                remaining_mean += _number(item["projection"])
                spread = max(0.5, _number(item["ceiling"]) - _number(item["floor"]))
                remaining_variance += (spread / (2 * 1.28155)) ** 2
                upside.append(item)
                downside.append(item)
            else:
                active.append(item)
        return {
            "live": live,
            "remaining_mean": remaining_mean,
            "remaining_variance": remaining_variance,
            "projected_final": live + remaining_mean,
            "remaining": remaining,
            "active_or_final": active,
            "upside": sorted(
                upside,
                key=lambda item: _number(item["ceiling"]) - _number(item["projection"]),
                reverse=True,
            )[:3],
            "downside": sorted(
                downside,
                key=lambda item: _number(item["projection"]) - _number(item["floor"]),
                reverse=True,
            )[:3],
        }

    mine = side(my_starters)
    opponent = side(opponent_starters)
    difference = _number(mine["projected_final"]) - _number(opponent["projected_final"])
    variance = _number(mine["remaining_variance"]) + _number(opponent["remaining_variance"])
    if variance <= 0:
        win_probability = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
    else:
        win_probability = _normal_cdf(difference / math.sqrt(variance))
    return {
        "mine": mine,
        "opponent": opponent,
        "projected_margin": difference,
        "win_probability": min(1.0, max(0.0, win_probability)),
        "updated_at": current.isoformat(),
    }


def late_decisions(
    lineup_moves: pl.DataFrame,
    kickoff_by_team: Mapping[str, datetime],
    *,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Return actionable swaps ordered by the nearest lineup lock."""
    if lineup_moves.is_empty():
        return lineup_moves
    current = now or datetime.now(UTC)
    rows = []
    for row in lineup_moves.iter_rows(named=True):
        lock = row.get("lock_time")
        lock_dt: datetime | None = None
        if isinstance(lock, str) and lock not in {"Unknown", "Locked"}:
            try:
                lock_dt = datetime.fromisoformat(lock.replace("Z", "+00:00"))
            except ValueError:
                lock_dt = None
        minutes = (lock_dt - current).total_seconds() / 60 if lock_dt else None
        rows.append({**row, "minutes_to_lock": minutes})
    return pl.DataFrame(rows).sort("minutes_to_lock", nulls_last=True)


__all__ = [
    "MatchupProjection",
    "PlayerState",
    "SideProjection",
    "late_decisions",
    "matchup_projection",
]
