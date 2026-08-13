"""Waiver wire (SPEC.md §14.4; task 2.6).

**"Value is relative to your roster, not absolute"** is the whole point
(SPEC's own words) -- `value_added` runs `sim.lineup.optimal_lineup`
(task 2.1) twice, once on my roster alone and once with the candidate
added, and reports the difference. A free agent projected for 11 points a
week is worth nothing if my three worst starters already project for 13;
this is exactly what the two-lineup-optimisation approach captures for
free, without a second, hand-rolled "am I deep here" heuristic.

**No real rest-of-season projections pipeline exists yet** (task 1.18,
`ffapp project --week N`, is still open) -- `current_form` uses the same
validated `ewm_mean(span=4)` window `models.baselines.add_b2_ewm_4` (task
1.10) already computes, just not shifted: B2's own `.shift(1)` exists to
keep a *backtest* honest about not seeing the target week's own outcome,
but here there is no future target to leak -- a player's trailing value is
being read forward from their last real game, which is fully known at
serving time. Task 1.15's own real conditional-points model doesn't beat
this baseline (documented in its own entry), so reusing it here would add
model risk for no real gain; revisit once 1.18 exists.

**SPEC's own `suggested_bid` formula names an "aggressiveness parameter"
to calibrate against league history, without defining a default.**
`config.WaiverSettings.aggressiveness` (default 1.0) and `.playoff_weight`
(default 1.5) are documented judgment calls, not a precise fit --
`docs/JOURNAL.md`'s task 2.6 entry has the real historical FAAB data that
informed them.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ffapp.ids.mapping import league_relevant
from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup, optimal_lineup_points

DEFAULT_EWM_SPAN = 4


def rostered_sleeper_ids(rosters: list[dict[str, Any]]) -> set[str]:
    """Every player id currently on *any* team's roster in this league --
    SPEC's own "rosters from Sleeper, minus all rostered players" needs
    every roster, not just mine."""
    ids: set[str] = set()
    for roster in rosters:
        ids.update(roster.get("players") or [])
    return ids


def free_agent_pool(
    players_dim: pl.DataFrame,
    rostered_ids: set[str],
    eligible_positions: set[str],
) -> pl.DataFrame:
    """`players_dim` (task 0.3's `ids.mapping.build_players_dim`) filtered
    to this league's real relevant positions and active/rostered-on-an-NFL-team
    players (`ids.mapping.league_relevant`, the same scoping the draft
    board and ID-check gate already use), minus every player already on a
    Sleeper roster in this league."""
    relevant = league_relevant(players_dim, eligible_positions)
    return relevant.filter(~pl.col("sleeper_id").is_in(list(rostered_ids)))


def current_form(player_week_features: pl.DataFrame, span: int = DEFAULT_EWM_SPAN) -> pl.DataFrame:
    """One row per player: their own trailing `ewm_mean(span=4)` of real
    `target` through their most recently played real week, unshifted (see
    module docstring for why the shift isn't needed here). Matches
    `models.baselines.add_b2_ewm_4`'s exact windowing convention
    (within-season only, `target` including non-played zero weeks) so
    "current form" and the validated B2 baseline never silently drift
    apart on what "trailing" means."""
    with_form = player_week_features.sort(["player_id", "season", "week"]).with_columns(
        pl.col("target").ewm_mean(span=span).over(["player_id", "season"]).alias("_form")
    )
    return (
        with_form.group_by("player_id", maintain_order=True)
        .agg(pl.all().last())
        .select(
            "player_id",
            "position",
            "season",
            "week",
            pl.col("_form").alias("projection_ppg"),
        )
    )


def value_added(
    my_roster: list[PlayerProjection],
    candidate: PlayerProjection,
    fmt: LeagueFormat,
) -> tuple[float, str | None]:
    """SPEC §14.4's own algorithm: baseline = my roster's own optimal
    starting-lineup points; with_p = the same with the candidate added.
    `optimal_lineup` only ever scores *started* players, so SPEC's "worst
    bench player dropped" doesn't change the points total by itself --
    what it actually names is which existing roster player becomes the
    real drop candidate once the candidate is added, returned as the
    second element. `None` when the candidate adds no value -- there is
    no real add/drop decision to make for a player who can't crack the
    lineup."""
    baseline = optimal_lineup_points(my_roster, fmt)
    lineup_with_candidate = optimal_lineup([*my_roster, candidate], fmt, objective="mean")
    added = lineup_with_candidate.total_points - baseline

    if added <= 0:
        return added, None

    started_ids = set(lineup_with_candidate.slots.values())
    benched_existing = [p for p in my_roster if p.player_id not in started_ids]
    drop_candidate = (
        min(benched_existing, key=lambda p: p.mean).player_id if benched_existing else None
    )
    return added, drop_candidate


def weeks_remaining(current_week: int, season_end_week: int = 17) -> list[int]:
    return list(range(current_week, season_end_week + 1))


def ros_value(
    value_added_per_week: float,
    weeks: list[int],
    playoff_week_start: int,
    playoff_weight: float = 1.5,
) -> float:
    """SPEC §14.4: "weight later weeks slightly higher if they fall in the
    fantasy playoffs" -- `playoff_weight` is a documented default (see
    module docstring), not a number SPEC itself gives."""
    return sum(
        value_added_per_week * (playoff_weight if week >= playoff_week_start else 1.0)
        for week in weeks
    )


def suggested_bid(
    candidate_ros_value: float,
    total_ros_value: float,
    remaining_budget: int,
    *,
    aggressiveness: float = 1.0,
    floor: int = 1,
) -> int:
    """SPEC §14.4's own formula, clamped `[floor, remaining_budget]` --
    except a candidate with no real value (`candidate_ros_value <= 0`,
    e.g. one that can't crack the lineup at all) bids 0, not SPEC's
    literal floor: the floor is meant for a genuinely useful but
    marginal player, not a free agent worth nothing to this roster. No
    remaining budget also bids 0 rather than raising on the resulting
    division."""
    if candidate_ros_value <= 0 or remaining_budget <= 0:
        return 0
    if total_ros_value <= 0:
        return floor
    raw = aggressiveness * (candidate_ros_value / total_ros_value) * remaining_budget
    return int(round(min(max(raw, floor), remaining_budget)))


def join_trending(board: pl.DataFrame, trending_ids: list[str]) -> pl.DataFrame:
    """`trending_ids` is Sleeper's own already-ranked trending-adds list
    (task 0.2's `ingest.sleeper.fetch_trending`) -- rank = position in
    that list (1 = most-added); null for a candidate not currently
    trending."""
    trend_df = pl.DataFrame(
        {"sleeper_id": trending_ids, "trend_rank": list(range(1, len(trending_ids) + 1))},
        schema={"sleeper_id": pl.String, "trend_rank": pl.Int64},
    )
    return board.join(trend_df, on="sleeper_id", how="left")


def build_waiver_board(
    free_agents: pl.DataFrame,
    projection_by_player: dict[str, float],
    my_roster: list[PlayerProjection],
    fmt: LeagueFormat,
    *,
    current_week: int,
    remaining_budget: int,
    season_end_week: int = 17,
    trending_ids: list[str] | None = None,
    playoff_weight: float = 1.5,
    aggressiveness: float = 1.0,
) -> pl.DataFrame:
    """SPEC §14.4's output table: `player, position, value_added_per_week,
    ros_value, suggested_bid, trend_rank, drop_candidate`, sorted by
    `ros_value` descending. `free_agents` needs `sleeper_id`/`player_id`/
    `position` columns (`free_agent_pool`'s own output shape). A free
    agent with no `current_form` estimate (never recorded a real
    player-week) is silently excluded, not crashed on or faked to 0 --
    there is genuinely nothing to rank them by."""
    weeks = weeks_remaining(current_week, season_end_week)
    trend_rank_by_id = {pid: i + 1 for i, pid in enumerate(trending_ids or [])}

    rows: list[dict[str, Any]] = []
    for row in free_agents.iter_rows(named=True):
        player_id = row["player_id"]
        projection_ppg = projection_by_player.get(player_id)
        if projection_ppg is None:
            continue
        candidate = PlayerProjection(
            player_id=player_id,
            position=row["position"],
            mean=projection_ppg,
            median=projection_ppg,
            ceiling=projection_ppg,
        )
        added, drop = value_added(my_roster, candidate, fmt)
        rv = ros_value(added, weeks, fmt.playoff_week_start, playoff_weight)
        sleeper_id = row.get("sleeper_id")
        rows.append(
            {
                "sleeper_id": sleeper_id,
                "player_id": player_id,
                "position": row["position"],
                "value_added_per_week": added,
                "ros_value": rv,
                "drop_candidate": drop,
                "trend_rank": trend_rank_by_id.get(sleeper_id) if sleeper_id is not None else None,
            }
        )

    total_positive_value = sum(r["ros_value"] for r in rows if r["ros_value"] > 0)
    for r in rows:
        r["suggested_bid"] = suggested_bid(
            r["ros_value"], total_positive_value, remaining_budget, aggressiveness=aggressiveness
        )

    schema = {
        "sleeper_id": pl.String,
        "player_id": pl.String,
        "position": pl.String,
        "value_added_per_week": pl.Float64,
        "ros_value": pl.Float64,
        "drop_candidate": pl.String,
        "trend_rank": pl.Int64,
        "suggested_bid": pl.Int64,
    }
    board = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return board.sort("ros_value", descending=True)


__all__ = [
    "build_waiver_board",
    "current_form",
    "free_agent_pool",
    "join_trending",
    "ros_value",
    "rostered_sleeper_ids",
    "suggested_bid",
    "value_added",
    "weeks_remaining",
]
