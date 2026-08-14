"""Draft board assembly (SPEC.md §9.7; task 0.12).

Wires together every piece built by tasks 0.6-0.11 into the single
per-player table SPEC §9.7 specifies, sorted by VOR descending. No new
projection/valuation logic lives here -- this module is orchestration glue,
not a new source of truth.

`playoff_sos` is left null throughout (SPEC §14.5 isn't built yet -- TASKS.md
0.12 says explicitly to leave it null rather than fake it). `as_of_utc` and
`git_commit` are appended beyond SPEC §9.7's own column table, per
CLAUDE.md's standing rule that every output artefact records them; there is
no `model_version` yet since Phase 0 has no trained model.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ffapp.config import REPO_ROOT, LeagueConfig, Settings
from ffapp.draft import pick_order
from ffapp.ids import mapping
from ffapp.ingest import nflverse, rankings, sleeper
from ffapp.league_format import parse_league_format
from ffapp.projections import aggregate, games_played
from ffapp.tools import adp as adp_tool
from ffapp.tools import tiers as tiers_tool
from ffapp.tools import vor as vor_tool

logger = logging.getLogger("ffapp.draft.board")

# Sleeper's roster-position vocabulary uses "DEF"; the rest of the codebase
# (scoring/keymap.py, league_format.py, scoring/stats.py) standardises on
# "DST" -- same alias league_format.py already applies to LeagueFormat.starters,
# needed again here since ids.mapping.league_relevant_positions returns the
# raw Sleeper tokens.
_POSITION_ALIASES = {"DEF": "DST"}

BOARD_COLUMNS = [
    "overall_rank",
    "pos_rank",
    "tier",
    "player",
    "position",
    "team",
    "bye_week",
    "proj_points_adj",
    "proj_ppg",
    "expected_games",
    "vor",
    "dispersion",
    "n_sources",
    "adp",
    "adp_sd",
    "value_vs_adp",
    "p_avail_next",
    "opportunity_cost",
    "playoff_sos",
    "as_of_utc",
    "git_commit",
]


def draft_board_csv_path(settings: Settings, *, season: int) -> Path:
    """`data/outputs/draft_board_<season>.csv` (SPEC §9.7) -- the one place
    that knows this path, shared by `ffapp draft board` (which writes it)
    and the Streamlit page (which reads it, task 0.13).
    """
    return settings.data_root / "outputs" / f"draft_board_{season}.csv"


# Sources with real per-stat projections (rescaled via league scoring before
# aggregating, SPEC §9.2). Ranks-only sources (FantasyPros, FootballGuys,
# DraftSharks) are handled separately below via _RANK_SOURCE_FETCHERS --
# mapped onto the point scale via the reference curve these sources build.
_POINT_SOURCE_FETCHERS: dict[str, Callable[[int, bool | None, Settings], pl.DataFrame]] = {
    "espn": lambda season, offline, settings: rankings.normalize_espn(
        json.loads(rankings.fetch_espn(season, offline=offline, settings=settings).read_text()),
        season=season,
    ),
    "fantasysharks": lambda season, offline, settings: rankings.normalize_fantasysharks(
        json.loads(rankings.fetch_fantasysharks(offline=offline, settings=settings).read_text()),
        season=season,
    ),
    "cbs": lambda season, offline, settings: rankings.normalize_cbs(
        json.loads(rankings.fetch_cbs(season, offline=offline, settings=settings).read_text()),
        season=season,
    ),
    "fftoday": lambda season, offline, settings: rankings.normalize_fftoday(
        json.loads(rankings.fetch_fftoday(season, offline=offline, settings=settings).read_text()),
        season=season,
    ),
}

# Ranks-only sources: no raw per-stat data, so no league-scoring rescale --
# each is instead mapped onto the point scale via the reference curve the
# point sources above build (SPEC §9.2). Each returns the same schema as
# `normalize_fantasypros` (source/season/player_name/position/team/rank/
# rank_sd/rank_best/rank_worst).
_RANK_SOURCE_FETCHERS: dict[str, Callable[[int, bool | None, Settings], pl.DataFrame]] = {
    "fantasypros": lambda season, offline, settings: rankings.normalize_fantasypros(
        rankings.fetch_fantasypros(offline=offline, settings=settings).read_text(encoding="utf-8"),
        season=season,
    ),
    "footballguys": lambda season, offline, settings: rankings.normalize_footballguys(
        rankings.fetch_footballguys(offline=offline, settings=settings).read_text(encoding="utf-8"),
        season=season,
    ),
    "draftsharks": lambda season, offline, settings: rankings.normalize_draftsharks(
        json.loads(rankings.fetch_draftsharks(offline=offline, settings=settings).read_text()),
        season=season,
    ),
}


class NoRankingsSourcesAvailableError(Exception):
    """Every per-stat rankings source failed -- there's nothing to aggregate."""


class NotEnoughPicksError(Exception):
    """This drafter owns fewer than 2 picks this draft -- p_avail_after_next
    and opportunity_cost have no second pick to reference."""


@dataclass(frozen=True)
class PickContext:
    my_roster_id: int
    my_slot: int | None
    my_picks: list[int]


def resolve_pick_context(
    league: LeagueConfig, settings: Settings, *, season: int, offline: bool | None = None
) -> PickContext:
    """Draft slot and every overall pick number the drafter owns this draft
    (SPEC §9.6), accounting for traded picks. Shared by `build_draft_board`
    (needs `next_pick`/`after_next_pick` for survival probabilities) and
    `draft.export` (needs the full list plus slot for the static export's
    header, SPEC-ADDENDUM-03.md §D) -- extracted rather than duplicated so
    the two never drift on how a pick's owner is resolved.
    """
    assert league.league_id is not None
    rosters_raw: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=offline, settings=settings).read_text()
    )
    draft_raw = json.loads(
        sleeper.fetch_drafts(league.league_id, offline=offline, settings=settings).read_text()
    )[0]
    traded_picks_path = sleeper.fetch_traded_picks(
        league.league_id, offline=offline, settings=settings
    )
    traded_picks = pick_order.parse_traded_picks(json.loads(traded_picks_path.read_text()))
    assert settings.sleeper_username is not None
    user_path = sleeper.fetch_user(settings.sleeper_username, offline=offline, settings=settings)
    user = json.loads(user_path.read_text())
    my_roster_id = pick_order.resolve_my_roster_id(user["user_id"], rosters_raw)
    roster_by_slot = pick_order.roster_id_by_slot(draft_raw["draft_order"], rosters_raw)
    my_slot = next((slot for slot, rid in roster_by_slot.items() if rid == my_roster_id), None)
    my_picks = pick_order.my_pick_numbers(
        my_roster_id,
        draft_order=draft_raw["draft_order"],
        rosters=rosters_raw,
        traded_picks=traded_picks,
        n_teams=draft_raw["settings"]["teams"],
        num_rounds=draft_raw["settings"]["rounds"],
        season=str(season),
    )
    return PickContext(my_roster_id=my_roster_id, my_slot=my_slot, my_picks=my_picks)


def _fetch_point_sources(
    season: int, *, offline: bool | None, settings: Settings
) -> list[pl.DataFrame]:
    """Fetch every per-stat rankings source, skipping (with a logged
    warning) any that fails live or comes back empty -- a single external
    source going down (confirmed to happen more than once: FFToday 403s,
    ESPN's bulk endpoint silently returning 0 rows, both fixed live in a
    later session -- see ingest.rankings' own module comments) must not
    sink the whole draft board. CLAUDE.md rule 4 ("never silently drop
    rows") is about players within a join, not about which of several
    redundant consensus sources contributed -- `aggregate_projections`'s own
    `n_sources`/`coverage` columns exist precisely to represent partial
    agreement honestly, which is what happens here.
    """
    sources = []
    for name, fetch in _POINT_SOURCE_FETCHERS.items():
        try:
            df = fetch(season, offline, settings)
        except Exception as exc:
            logger.warning("skipping rankings source %r: %s", name, exc)
            continue
        if df.height == 0:
            logger.warning("skipping rankings source %r: returned 0 rows", name)
            continue
        sources.append(df)
    return sources


def _fetch_rank_sources(
    season: int, *, offline: bool | None, settings: Settings
) -> list[pl.DataFrame]:
    """Same graceful degradation as `_fetch_point_sources`, for the
    ranks-only sources (FantasyPros, FootballGuys, DraftSharks) instead."""
    sources = []
    for name, fetch in _RANK_SOURCE_FETCHERS.items():
        try:
            df = fetch(season, offline, settings)
        except Exception as exc:
            logger.warning("skipping rankings source %r: %s", name, exc)
            continue
        if df.height == 0:
            logger.warning("skipping rankings source %r: returned 0 rows", name)
            continue
        sources.append(df)
    return sources


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _team_by_join_key(players_dim: pl.DataFrame) -> pl.DataFrame:
    """`aggregate_projections` deliberately keeps only `join_key`,
    `player_name`, `position`, `points` (see `projections/aggregate.py`'s
    own selection) -- `team` needs pulling back in separately, from
    `players_dim` via the same normalized `(name, position)` join key
    `games_played.player_ages_from_players_dim` already uses.

    Deduped via `ids.mapping.dedupe_to_one_row_per_name_position` first --
    a raw `players_dim` can have more than one row per (name, position)
    (confirmed live: an active player and a same-base-name retired relative
    both normalizing to the same key), which would otherwise fan a single
    real player's row out via this left join too, the same bug that hit
    `games_played.player_ages_from_players_dim`.
    """
    deduped = mapping.dedupe_to_one_row_per_name_position(players_dim)
    return deduped.filter(pl.col("team").is_not_null()).select("join_key", "team")


def _add_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """`overall_rank`/`pos_rank` (VOR descending) and `value_vs_adp`
    (`adp_rank - overall_rank`, positive = falls to you, per SPEC §9.7).
    `adp_rank` (and so `value_vs_adp`) stays null for a player with no ADP
    coverage -- `pl.Expr.rank` already keeps nulls null, so this falls out
    for free rather than needing a special case.
    """
    return df.with_columns(
        pl.col("vor").rank(method="ordinal", descending=True).cast(pl.Int64).alias("overall_rank"),
        pl.col("vor")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int64)
        .alias("pos_rank"),
        pl.col("adp").rank(method="ordinal").cast(pl.Int64).alias("adp_rank"),
    ).with_columns((pl.col("adp_rank") - pl.col("overall_rank")).alias("value_vs_adp"))


def finalize_draft_board(
    df: pl.DataFrame, *, as_of_utc: str, git_commit: str | None
) -> pl.DataFrame:
    """Add ranks (`overall_rank`, `pos_rank`, `value_vs_adp`) and select the
    final SPEC §9.7 column set, sorted by VOR descending. `df` must already
    carry every other board column (`vor`, `position`, `player_name`,
    `team`, `bye_week`, `proj_points_adj`, `proj_ppg`, `expected_games`,
    `dispersion`, `n_sources`, `adp`, `adp_sd`, `p_avail_next`,
    `opportunity_cost`, `tier`) -- this function only ranks and reshapes,
    it doesn't compute any of those.
    """
    ranked = _add_ranks(df)
    return ranked.sort("vor", descending=True).select(
        "overall_rank",
        "pos_rank",
        "tier",
        pl.col("player_name").alias("player"),
        "position",
        "team",
        "bye_week",
        "proj_points_adj",
        "proj_ppg",
        "expected_games",
        "vor",
        "dispersion",
        "n_sources",
        "adp",
        "adp_sd",
        "value_vs_adp",
        "p_avail_next",
        "opportunity_cost",
        pl.lit(None, dtype=pl.Float64).alias("playoff_sos"),
        pl.lit(as_of_utc).alias("as_of_utc"),
        # "g" prefix, not the bare hash: a short hash like "05236e5" is a
        # valid float literal (5236e5 in scientific notation) -- confirmed
        # live, `pl.read_csv`'s default type inference silently mangled it
        # on read-back, and Excel would do the same. The prefix makes it
        # unambiguously non-numeric for every downstream CSV reader.
        pl.lit(f"g{git_commit}" if git_commit is not None else None, dtype=pl.Utf8).alias(
            "git_commit"
        ),
    )


def build_draft_board(
    league: LeagueConfig,
    settings: Settings,
    *,
    season: int,
    offline: bool | None = None,
) -> pl.DataFrame:
    """Assemble the full draft board (SPEC §9.7) for `league`."""
    league_format = parse_league_format(league)
    scoring_settings = league.league_cache["scoring_settings"]
    assert league.league_id is not None

    point_sources = _fetch_point_sources(season, offline=offline, settings=settings)
    if not point_sources:
        raise NoRankingsSourcesAvailableError(
            "Every per-stat rankings source failed or returned no data -- nothing to build "
            "a draft board from. Check network reachability and each source's own status."
        )
    scored_point_sources = [
        aggregate.apply_league_scoring(df, scoring_settings) for df in point_sources
    ]
    reference_curve = aggregate.build_reference_curve(scored_point_sources)

    rank_sources = _fetch_rank_sources(season, offline=offline, settings=settings)
    rank_points = [aggregate.map_ranks_to_points(df, reference_curve) for df in rank_sources]

    n_sources = len(scored_point_sources) + len(rank_points)
    scored = [aggregate.add_join_key(df) for df in (*scored_point_sources, *rank_points)]
    projections = aggregate.aggregate_projections(scored, n_sources=n_sources)

    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    players_dim = mapping.build_players_dim(
        crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
    )
    ages = games_played.player_ages_from_players_dim(players_dim, as_of=date.today())
    with_games = games_played.add_games_played_adjustment(projections, ages)
    with_team = with_games.join(_team_by_join_key(players_dim), on="join_key", how="left")

    eligible_positions = {
        _POSITION_ALIASES.get(position, position)
        for position in mapping.league_relevant_positions(league)
    }
    scoped = with_team.filter(pl.col("position").is_in(list(eligible_positions)))
    with_vor = vor_tool.compute_vor(scoped, league_format)

    rosters_raw: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=offline, settings=settings).read_text()
    )
    keeper_ids = adp_tool.keeper_sleeper_ids(rosters_raw)
    keeper_keys = adp_tool.keeper_join_keys(keeper_ids, players_dim)
    excluded_keepers = adp_tool.exclude_keepers(with_vor, keeper_keys)

    adp_raw = json.loads(
        rankings.fetch_adp(
            season, teams=league_format.n_teams, offline=offline, settings=settings
        ).read_text()
    )
    adp_df = aggregate.add_join_key(rankings.normalize_adp(adp_raw, season=season))
    with_adp = adp_tool.join_adp(excluded_keepers, adp_df)

    pick_context = resolve_pick_context(league, settings, season=season, offline=offline)
    if len(pick_context.my_picks) < 2:
        raise NotEnoughPicksError(
            f"Only {len(pick_context.my_picks)} pick(s) resolved for roster "
            f"{pick_context.my_roster_id} this draft -- need at least 2 "
            "(p_avail_after_next/opportunity_cost need a second pick)."
        )
    next_pick, after_next_pick = pick_context.my_picks[0], pick_context.my_picks[1]

    with_survival = adp_tool.add_survival_probabilities(
        with_adp,
        next_pick=next_pick,
        after_next_pick=after_next_pick,
        adp_sd_fallback=settings.draft.adp_sd_fallback,
    )
    with_opportunity_cost = adp_tool.add_opportunity_cost(
        with_survival, fallback_pick=after_next_pick, adp_sd_fallback=settings.draft.adp_sd_fallback
    )

    with_tiers = tiers_tool.assign_tiers(with_opportunity_cost, method=settings.draft.tier_method)

    return finalize_draft_board(
        with_tiers,
        as_of_utc=datetime.now(UTC).isoformat(),
        git_commit=_current_git_commit(),
    )


__all__ = [
    "BOARD_COLUMNS",
    "NoRankingsSourcesAvailableError",
    "NotEnoughPicksError",
    "PickContext",
    "build_draft_board",
    "draft_board_csv_path",
    "finalize_draft_board",
    "resolve_pick_context",
]
