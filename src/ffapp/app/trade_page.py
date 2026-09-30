"""Pure data adapters for the lineup-aware trade analyzer page."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, slot_instances
from ffapp.sim.season import Matchup, Roster, SimPlayer, WeekProjection


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _expected_lineup_value(roster: Roster, weeks: Sequence[int], fmt: LeagueFormat) -> float:
    total = 0.0
    for week in weeks:
        players = []
        for player in roster.players:
            weekly = player.weekly or {}
            projection = weekly.get(week)
            if projection is None:
                continue
            values = list(projection.quantile_values)
            players.append(
                PlayerProjection(
                    player.player_id,
                    player.position,
                    projection.mean,
                    values[1] if len(values) > 1 else projection.mean,
                    values[-2] if len(values) > 1 else projection.mean,
                )
            )
        if players:
            total += _fast_lineup_points(players, fmt)
    return total


def _fast_lineup_points(players: Sequence[PlayerProjection], fmt: LeagueFormat) -> float:
    """Solve the lineup as a bipartite assignment without spawning CBC."""
    slots = slot_instances(fmt)
    if not slots or not players:
        return 0.0
    values = np.zeros((len(players) + len(slots), len(slots)), dtype=float)
    values[: len(players), :] = -1e9
    for player_index, player in enumerate(players):
        for slot_index, (_, eligible) in enumerate(slots):
            if player.position in eligible:
                values[player_index, slot_index] = player.mean
    rows, columns = linear_sum_assignment(values, maximize=True)
    return float(
        sum(
            values[row, column]
            for row, column in zip(rows, columns, strict=True)
            if row < len(players) and values[row, column] > -1e8
        )
    )


def _changed_roster(
    roster: Roster, outgoing: Sequence[str], incoming: Sequence[SimPlayer]
) -> Roster:
    remove = set(outgoing)
    return Roster(
        roster.team_id,
        [player for player in roster.players if player.player_id not in remove] + list(incoming),
    )


def find_trade_candidates(
    teams: Sequence[Roster],
    vor_by_player: Mapping[str, float],
    names: Mapping[str, str],
    team_names: Mapping[str, str],
    my_team_id: str,
    fmt: LeagueFormat,
    *,
    remaining_weeks: Sequence[int],
    limit: int = 30,
) -> pl.DataFrame:
    """Search fair 1-for-1 and uneven packages, then score lineup impact for both teams."""
    by_team = {team.team_id: team for team in teams}
    mine = by_team.get(my_team_id)
    if mine is None:
        return pl.DataFrame()
    base_mine = _expected_lineup_value(mine, remaining_weeks, fmt)
    my_pool = sorted(
        [p for p in mine.players if vor_by_player.get(p.player_id, 0) > 0],
        key=lambda p: vor_by_player.get(p.player_id, 0),
        reverse=True,
    )[:10]
    my_packages = [*(itertools.combinations(my_pool, 1)), *(itertools.combinations(my_pool, 2))]
    prelim: list[tuple[float, Roster, tuple[SimPlayer, ...], tuple[SimPlayer, ...], float]] = []
    for partner in teams:
        if partner.team_id == my_team_id:
            continue
        partner_pool = sorted(
            [p for p in partner.players if vor_by_player.get(p.player_id, 0) > 0],
            key=lambda p: vor_by_player.get(p.player_id, 0),
            reverse=True,
        )[:10]
        partner_packages = [
            *itertools.combinations(partner_pool, 1),
            *itertools.combinations(partner_pool, 2),
        ]
        partner_rows = []
        for outgoing in my_packages:
            for incoming in partner_packages:
                if len(outgoing) == len(incoming) == 2:
                    continue
                sent = sum(vor_by_player.get(player.player_id, 0.0) for player in outgoing)
                received = sum(vor_by_player.get(player.player_id, 0.0) for player in incoming)
                fairness = 1.0 - abs(sent - received) / max(1.0, sent, received)
                if fairness < 0.72:
                    continue
                package_size_penalty = 0.03 * (len(outgoing) + len(incoming) - 2)
                pre_score = fairness - package_size_penalty + min(sent, received) / 1000
                partner_rows.append((pre_score, partner, outgoing, incoming, fairness))
        prelim.extend(sorted(partner_rows, key=lambda item: item[0], reverse=True)[:60])

    results = []
    value_cache: dict[tuple[str, ...], float] = {}

    def value(roster: Roster) -> float:
        key = tuple(sorted(player.player_id for player in roster.players))
        if key not in value_cache:
            value_cache[key] = _expected_lineup_value(roster, remaining_weeks, fmt)
        return value_cache[key]

    for _, partner, outgoing, incoming, fairness in sorted(
        prelim, key=lambda item: item[0], reverse=True
    )[:240]:
        base_partner = value(partner)
        new_mine = _changed_roster(mine, [p.player_id for p in outgoing], incoming)
        new_partner = _changed_roster(partner, [p.player_id for p in incoming], outgoing)
        my_gain = value(new_mine) - base_mine
        partner_gain = value(new_partner) - base_partner
        if my_gain < 0.5 or partner_gain < -1.0:
            continue
        score = my_gain + max(0.0, partner_gain) * 0.45 + fairness * 2.0
        results.append(
            {
                "partner": team_names.get(partner.team_id, partner.team_id),
                "partner_id": partner.team_id,
                "you_send": " + ".join(names.get(p.player_id, p.player_id) for p in outgoing),
                "send_ids": [p.player_id for p in outgoing],
                "you_receive": " + ".join(names.get(p.player_id, p.player_id) for p in incoming),
                "receive_ids": [p.player_id for p in incoming],
                "your_lineup_gain": my_gain,
                "partner_lineup_gain": partner_gain,
                "value_match": fairness,
                "mutual_benefit": partner_gain > 0,
                "trade_score": score,
                "why": (
                    f"Adds {my_gain:.1f} expected lineup points for you over the remaining "
                    f"schedule; {team_names.get(partner.team_id, partner.team_id)} changes "
                    f"by {partner_gain:+.1f}."
                ),
            }
        )
    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results).sort("trade_score", descending=True).head(limit)


def suggested_trade_packages(
    teams: Sequence[Roster],
    vor_by_player: Mapping[str, float],
    names: Mapping[str, str],
    team_names: Mapping[str, str],
    my_team_id: str,
    *,
    limit: int = 12,
) -> pl.DataFrame:
    """Find fair one-for-one ideas that address each roster's relative weak spots."""
    rows = [
        {
            "team_id": team.team_id,
            "player_id": player.player_id,
            "position": player.position,
            "value": float(vor_by_player.get(player.player_id, 0.0)),
            "weekly_mean": float(player.mean),
        }
        for team in teams
        for player in team.players
    ]
    if not rows:
        return pl.DataFrame()
    players = pl.DataFrame(rows)
    strength = players.group_by("team_id", "position").agg(
        pl.col("weekly_mean").sort(descending=True).head(2).sum().alias("strength")
    )
    baseline = strength.group_by("position").agg(pl.col("strength").median().alias("median"))
    needs = strength.join(baseline, on="position").with_columns(
        (pl.col("strength") - pl.col("median")).alias("edge")
    )
    need_by_team = {
        (str(row["team_id"]), str(row["position"])): _number(row["edge"])
        for row in needs.iter_rows(named=True)
    }
    mine = [row for row in rows if row["team_id"] == my_team_id and _number(row["value"]) > 0]
    theirs = [row for row in rows if row["team_id"] != my_team_id and _number(row["value"]) > 0]
    ideas = []
    for target in theirs:
        my_fit = -need_by_team.get((my_team_id, str(target["position"])), 0.0)
        if my_fit <= 0:
            continue
        for offer in mine:
            if offer["position"] == target["position"]:
                continue
            partner_fit = -need_by_team.get((str(target["team_id"]), str(offer["position"])), 0.0)
            if partner_fit <= 0:
                continue
            target_value, offer_value = _number(target["value"]), _number(offer["value"])
            fairness = 1.0 - abs(target_value - offer_value) / max(1.0, target_value, offer_value)
            if fairness < 0.68:
                continue
            ideas.append(
                {
                    "target": names.get(str(target["player_id"]), str(target["player_id"])),
                    "target_id": target["player_id"],
                    "target_position": target["position"],
                    "partner": team_names.get(str(target["team_id"]), str(target["team_id"])),
                    "partner_id": target["team_id"],
                    "offer": names.get(str(offer["player_id"]), str(offer["player_id"])),
                    "offer_id": offer["player_id"],
                    "offer_position": offer["position"],
                    "target_value": target_value,
                    "offer_value": offer_value,
                    "fairness": fairness,
                    "fit_score": my_fit + partner_fit,
                }
            )
    if not ideas:
        return pl.DataFrame()
    return pl.DataFrame(ideas).sort(["fit_score", "fairness"], descending=True).head(limit)


def trade_analysis_blocker(league: object, playoff_week_start: int) -> str | None:
    """Explain why the standard head-to-head trade simulation does not apply."""
    league_cache = getattr(league, "league_cache", {})
    if league_cache.get("disable_trades"):
        return "Trades are disabled in this league."
    if league_cache.get("league_type") == 3 or playoff_week_start <= 0:
        return "Elimination leagues need a survival model, not head-to-head playoff odds."
    return None


def build_trade_rosters(
    projections_ros: pl.DataFrame,
    roster_players: Mapping[str, Sequence[str]],
    *,
    from_week: int,
) -> tuple[list[Roster], dict[str, float]]:
    """Convert weekly ROS projections into one lineup-aware player per roster."""
    required_values = ["mean", "q10", "q25", "q50", "q75", "q90"]
    future = projections_ros.filter(pl.col("week") >= from_week).drop_nulls(required_values)
    if future.is_empty():
        return [], {}
    players = future.group_by("player_id").agg(
        pl.col("position").drop_nulls().first(),
        pl.col("team").drop_nulls().first(),
        pl.col("opponent_team").drop_nulls().first(),
        pl.col("mean").mean(),
        pl.col("q10").mean(),
        pl.col("q25").mean(),
        pl.col("q50").mean(),
        pl.col("q75").mean(),
        pl.col("q90").mean(),
        pl.col("mean").sum().alias("ros_points"),
    )
    by_id = {str(row["player_id"]): row for row in players.iter_rows(named=True)}
    weekly_by_id: dict[str, dict[int, WeekProjection]] = {}
    for row in future.iter_rows(named=True):
        weekly_by_id.setdefault(str(row["player_id"]), {})[int(row["week"])] = WeekProjection(
            mean=float(row["mean"]),
            quantile_values=tuple(float(row[name]) for name in ("q10", "q25", "q50", "q75", "q90")),
            opponent_team=(str(row["opponent_team"]) if row["opponent_team"] is not None else None),
        )
    rosters: list[Roster] = []
    for team_id, player_ids in roster_players.items():
        sim_players: list[SimPlayer] = []
        for player_id in player_ids:
            player_row = by_id.get(player_id)
            if player_row is None:
                continue
            sim_players.append(
                SimPlayer(
                    player_id=player_id,
                    position=str(player_row["position"]),
                    team=str(player_row["team"]),
                    opponent_team=(
                        str(player_row["opponent_team"])
                        if player_row["opponent_team"] is not None
                        else None
                    ),
                    mean=float(player_row["mean"]),
                    alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
                    quantile_values=tuple(
                        float(player_row[name]) for name in ("q10", "q25", "q50", "q75", "q90")
                    ),
                    p_miss=0.0,
                    weekly=weekly_by_id[player_id],
                )
            )
        rosters.append(Roster(team_id=str(team_id), players=sim_players))

    replacement = players.group_by("position").agg(pl.col("ros_points").median().alias("base"))
    baseline = dict(zip(replacement["position"], replacement["base"], strict=True))
    vor = {
        player_id: max(0.0, float(row["ros_points"]) - float(baseline[str(row["position"])]))
        for player_id, row in by_id.items()
    }
    return rosters, vor


def matchup_schedule(
    payload_by_week: Mapping[int, Sequence[Mapping[str, object]]],
) -> list[Matchup]:
    """Convert Sleeper matchup rows into simulator pairings."""
    schedule: list[Matchup] = []
    for week, rows in sorted(payload_by_week.items()):
        grouped: dict[object, list[str]] = {}
        for row in rows:
            matchup_id = row.get("matchup_id")
            roster_id = row.get("roster_id")
            if matchup_id is not None and roster_id is not None:
                grouped.setdefault(matchup_id, []).append(str(roster_id))
        for team_ids in grouped.values():
            if len(team_ids) == 2:
                schedule.append(Matchup(week=week, home=team_ids[0], away=team_ids[1]))
    return schedule


def standings_from_rosters(
    rosters: Sequence[Mapping[str, object]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Read current Sleeper wins and points-for, including decimal components."""
    wins: dict[str, float] = {}
    points: dict[str, float] = {}
    for roster in rosters:
        team_id = str(roster["roster_id"])
        settings = roster.get("settings")
        values = settings if isinstance(settings, dict) else {}
        wins[team_id] = float(values.get("wins", 0)) + 0.5 * float(values.get("ties", 0))
        points[team_id] = (
            float(values.get("fpts", 0)) + float(values.get("fpts_decimal", 0)) / 100.0
        )
    return wins, points


__all__ = [
    "build_trade_rosters",
    "find_trade_candidates",
    "matchup_schedule",
    "suggested_trade_packages",
    "standings_from_rosters",
    "trade_analysis_blocker",
]
