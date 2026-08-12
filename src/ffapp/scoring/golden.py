"""The scoring golden test (SPEC.md §8.4): compare `score_stat_line`'s output
against Sleeper's own computed `players_points` for every player-week of the
league's most recent PLAYED season, and refuse to trust the scoring engine until
they agree.

Both real leagues are pre-draft for the season in `config/leagues/`, so `run_golden_test`
resolves to the linked `previous_league_id`'s season rather than the current one --
see HANDOFF.md §4.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import polars as pl

from ffapp.config import Settings
from ffapp.config import load_league as _load_league
from ffapp.config import load_settings as _load_settings
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.scoring import engine, stats

logger = logging.getLogger("ffapp.scoring.golden")

# Regular season (17 weeks since 2021) plus a few playoff weeks. Sleeper returns an
# empty matchups list for weeks past the season's actual end, so this only needs to
# be a safe upper bound, not exact -- run_golden_test skips empty weeks.
MAX_WEEK = 18


def extract_players_points(matchups: list[dict[str, object]]) -> dict[str, float]:
    """Merge `players_points` across every roster in one week's matchups response
    into a single sleeper_id -> points mapping."""
    merged: dict[str, float] = {}
    for matchup in matchups:
        points = matchup.get("players_points")
        if isinstance(points, dict):
            merged.update(points)
    return merged


# Sleeper's own DST sleeper_id differs from nflverse's team code in exactly one
# case, confirmed live against both real datasets' full 32-team lists: Sleeper
# uses "LAR" for the Rams; nflverse uses "LA". Every other code matches exactly.
SLEEPER_TEAM_ALIASES: dict[str, str] = {"LAR": "LA"}


def resolve_player_ids(
    sleeper_ids: set[str], players_dim: pl.DataFrame, *, team_abbreviations: set[str]
) -> dict[str, str]:
    """sleeper_id -> canonical player_id used in the computed stat frame.

    DST entries are keyed by team abbreviation on both sides (Sleeper's own
    sleeper_id for a team defense *is* the team abbreviation, aliased through
    `SLEEPER_TEAM_ALIASES` where the two sources disagree) and never go through
    the crosswalk, which only carries individual-player ids. An id with no
    crosswalk row falls back to itself rather than being dropped (CLAUDE.md rule
    4) -- `compare_points` is what turns that into a visible miss.
    """
    lookup = {
        row["sleeper_id"]: row["player_id"]
        for row in players_dim.iter_rows(named=True)
        if row["sleeper_id"] is not None
    }
    result = {}
    for sid in sleeper_ids:
        team_code = SLEEPER_TEAM_ALIASES.get(sid, sid)
        result[sid] = team_code if team_code in team_abbreviations else lookup.get(sid, sid)
    return result


@dataclass(frozen=True)
class Disagreement:
    week: int
    player_id: str
    sleeper_points: float
    computed_points: float
    missing_computed_row: bool


def compare_points(
    sleeper_points: dict[str, float],
    computed_points: dict[str, float],
    *,
    tolerance: float = 0.01,
    week: int = 0,
) -> list[Disagreement]:
    """One week's disagreements: `sleeper_points` (already id-resolved) vs.
    `computed_points`, both keyed by canonical player_id. A sleeper_id with no
    computed row is treated as computed=0.0 -- agreement if Sleeper also says 0.0
    (the expected shape of a bye week), a flagged miss otherwise.
    """
    disagreements = []
    for player_id, sleeper_value in sleeper_points.items():
        missing = player_id not in computed_points
        computed_value = computed_points.get(player_id, 0.0)
        if abs(sleeper_value - computed_value) > tolerance:
            disagreements.append(
                Disagreement(
                    week=week,
                    player_id=player_id,
                    sleeper_points=sleeper_value,
                    computed_points=computed_value,
                    missing_computed_row=missing,
                )
            )
    return disagreements


@dataclass(frozen=True)
class GoldenTestResult:
    total_player_weeks: int
    disagreements: list[Disagreement]
    agreement_rate: float
    passed: bool


PASS_THRESHOLD = 0.99


def summarize(total_player_weeks: int, disagreements: list[Disagreement]) -> GoldenTestResult:
    if total_player_weeks == 0:
        return GoldenTestResult(
            total_player_weeks=0, disagreements=disagreements, agreement_rate=0.0, passed=False
        )
    agreement_rate = 1 - (len(disagreements) / total_player_weeks)
    return GoldenTestResult(
        total_player_weeks=total_player_weeks,
        disagreements=disagreements,
        agreement_rate=agreement_rate,
        passed=agreement_rate >= PASS_THRESHOLD,
    )


class NoPlayedSeasonError(Exception):
    """The league has no `previous_league_id` -- nothing has been played yet to
    validate the scoring engine against."""


def run_golden_test(
    slug: str,
    *,
    settings: Settings | None = None,
    offline: bool | None = None,
    max_week: int = MAX_WEEK,
) -> GoldenTestResult:
    """SPEC §8.4: validate `score_stat_line` against Sleeper's own `players_points`
    for every completed week of `slug`'s most recent PLAYED season.

    Both real leagues are pre-draft for their *current* config season, so this
    resolves the linked `previous_league_id` and uses *that* league object's own
    scoring_settings/season -- a league's scoring can change year to year, and the
    golden test must validate against what actually produced the historical
    numbers, not this season's config (HANDOFF.md §4).
    """
    settings = settings or _load_settings()
    league = _load_league(slug)
    if league.league_id is None:
        raise ValueError(f"League {slug} has no sleeper.league_id configured.")

    current_raw = json.loads(
        sleeper.fetch_league(league.league_id, offline=offline, settings=settings).read_text()
    )
    previous_league_id = current_raw.get("previous_league_id")
    if not previous_league_id:
        raise NoPlayedSeasonError(
            f"League {slug} has no previous_league_id -- nothing played yet to validate against."
        )

    previous_raw = json.loads(
        sleeper.fetch_league(previous_league_id, offline=offline, settings=settings).read_text()
    )
    season = int(previous_raw["season"])
    scoring = previous_raw.get("scoring_settings", {})

    player_stats = pl.read_parquet(
        nflverse.fetch_player_stats(season, offline=offline, settings=settings)
    )
    team_stats = pl.read_parquet(
        nflverse.fetch_team_stats(season, offline=offline, settings=settings)
    )
    schedules = pl.read_parquet(
        nflverse.fetch_schedules(season, offline=offline, settings=settings)
    )
    pbp = pl.read_parquet(nflverse.fetch_pbp(season, offline=offline, settings=settings))
    frame = stats.build_stat_frame(player_stats, team_stats, schedules, pbp)
    computed = frame.with_columns(engine.score_stat_line(frame, scoring).alias("_computed_points"))

    team_abbreviations = set(
        computed.filter(pl.col("position") == "DST")["player_id"].unique().to_list()
    )

    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    overrides_path = mapping.ID_OVERRIDES_PATH if mapping.ID_OVERRIDES_PATH.exists() else None
    players_dim = mapping.build_players_dim(crosswalk_path, sleeper_players_path, overrides_path)

    all_disagreements: list[Disagreement] = []
    total_player_weeks = 0

    for week in range(1, max_week + 1):
        matchups_path = sleeper.fetch_matchups(
            previous_league_id, week, offline=offline, settings=settings
        )
        matchups = json.loads(matchups_path.read_text())
        raw_points = extract_players_points(matchups)
        if not raw_points:
            continue

        resolved = resolve_player_ids(
            set(raw_points), players_dim, team_abbreviations=team_abbreviations
        )
        sleeper_points = {resolved[sid]: pts for sid, pts in raw_points.items()}

        week_frame = computed.filter(pl.col("week") == week)
        computed_points = dict(
            zip(week_frame["player_id"], week_frame["_computed_points"], strict=True)
        )

        week_disagreements = compare_points(sleeper_points, computed_points, week=week)
        all_disagreements.extend(week_disagreements)
        total_player_weeks += len(sleeper_points)

        for d in week_disagreements:
            logger.warning(
                "week %d player %s: sleeper=%.2f computed=%.2f%s",
                d.week,
                d.player_id,
                d.sleeper_points,
                d.computed_points,
                " (no computed row)" if d.missing_computed_row else "",
            )

    return summarize(total_player_weeks, all_disagreements)


__all__ = [
    "MAX_WEEK",
    "PASS_THRESHOLD",
    "Disagreement",
    "GoldenTestResult",
    "NoPlayedSeasonError",
    "compare_points",
    "extract_players_points",
    "resolve_player_ids",
    "run_golden_test",
    "summarize",
]
