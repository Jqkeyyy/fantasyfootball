"""Draft board assembly (SPEC.md §9.7; task 0.12).

Wires together every piece built by tasks 0.6-0.11 into the single
per-player table SPEC §9.7 specifies, sorted by VOR descending. No new
projection/valuation logic lives here -- this module is orchestration glue,
not a new source of truth.

`playoff_sos` is left null throughout (SPEC §14.5 isn't built yet -- TASKS.md
0.12 says explicitly to leave it null rather than fake it). `as_of_utc`,
`git_commit`, and `is_keeper` are appended beyond SPEC §9.7's own column
table, per CLAUDE.md's standing rule that every output artefact records
its own provenance; there is no `model_version` yet since Phase 0 has no
trained model.

Kept players are no longer excluded from the board (a real design change,
made live this session at the project owner's own request after they
noticed the board was silently missing this year's real keepers) -- every
other computed column (VOR, tier, ADP survival probability, opportunity
cost) still treats a keeper exactly like any other player, so `is_keeper`
is purely informational, not a filter: a keeper's own `p_avail_next`/
`opportunity_cost` describe a hypothetical "if this pick were live," not
a real upcoming draft event, since Sleeper auto-assigns keepers their own
real pick before the live draft assistant (`draft.live`) ever sees them.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from rapidfuzz import fuzz, process

from ffapp.cache.offline import OfflineCacheMiss
from ffapp.config import REPO_ROOT, LeagueConfig, Settings
from ffapp.draft import pick_order
from ffapp.ids import mapping
from ffapp.ingest import manual_rankings, nflverse, rankings, sleeper
from ffapp.league_format import LeagueFormat, parse_league_format
from ffapp.projections import aggregate, games_played
from ffapp.tools import adp as adp_tool
from ffapp.tools import streaming as streaming_tool
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
    "is_keeper",
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


def source_rankings_csv_path(settings: Settings, *, season: int) -> Path:
    """`data/outputs/source_rankings_<season>.csv` -- the "no model" board:
    each source's own real published overall rank side by side, no VOR/
    league-scoring valuation. Written by the same `ffapp draft board`
    command as the main board, read by the Streamlit page's per-source tabs.
    """
    return settings.data_root / "outputs" / f"source_rankings_{season}.csv"


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

# `build_draft_board`'s own point/rank split, unrelated to `build_source_
# rankings` -- that now sources every tab from `ingest.manual_rankings`'s
# real published overall ranks instead (see that module's docstring for
# why: CBS/ESPN/FantasySharks/FFToday's live-scraped data here is stat
# projections, not a rank of any kind, confirmed live to disagree with
# CBS's own real rankings page in exactly the way you'd expect).
POINT_SOURCE_NAMES = frozenset(_POINT_SOURCE_FETCHERS)
RANK_SOURCE_NAMES = frozenset(_RANK_SOURCE_FETCHERS)


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
    final SPEC §9.7 column set (plus `is_keeper`), sorted by VOR descending.
    `df` must already carry every other board column (`vor`, `position`,
    `player_name`, `team`, `bye_week`, `proj_points_adj`, `proj_ppg`,
    `expected_games`, `dispersion`, `n_sources`, `adp`, `adp_sd`,
    `p_avail_next`, `opportunity_cost`, `tier`, `is_keeper`) -- this
    function only ranks and reshapes, it doesn't compute any of those.
    """
    ranked = _add_ranks(df)
    return ranked.sort("vor", descending=True).select(
        "overall_rank",
        "pos_rank",
        "tier",
        pl.col("player_name").alias("player"),
        "is_keeper",
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


def _streaming_replacement_overrides(
    league_format: LeagueFormat,
    scoring_settings: dict[str, float],
    *,
    board_positions: set[str],
    season: int,
    offline: bool | None,
    settings: Settings,
) -> dict[str, float]:
    """Empirical DST/K replacement level from this league's own real
    regular-season scoring history (`tools.streaming`) -- not a numbered
    TASKS.md task, direct request 2026-08-14 following up on task 0.9's
    VOR. See that module's own docstring for why "the Nth-best preseason
    total" (the standard fixed-point baseline every other position uses)
    is the wrong replacement level for a position that's realistically
    streamed off waivers every week, confirmed against this league's own
    real 2021-2025 scoring (`docs/JOURNAL.md`'s 2026-08-14 entry).

    Scoped to whichever of `tools.streaming.STREAMING_POSITIONS` this
    league actually starts (CLAUDE.md rule 5 -- never hardcode league
    format) AND are still present in `board_positions` -- if
    `settings.draft.excluded_positions` has already dropped a position
    from the board entirely (see `build_draft_board`), there's nothing to
    compute a replacement level for. Falls back to no override at all
    (the standard fixed-point baseline) if the historical raw nflverse
    tables aren't cached locally yet -- HANDOFF.md §6's own rebuild
    sequence, not this function's job to enforce; a draft board should
    still build without them.
    """
    n_drafted_by_position = {
        position: league_format.n_teams * league_format.starters.get(position, 0)
        for position in streaming_tool.STREAMING_POSITIONS
        if position in board_positions and league_format.starters.get(position, 0) > 0
    }
    if not n_drafted_by_position:
        return {}

    historical_seasons = list(range(2015, season))
    try:
        player_stats = pl.read_parquet(
            nflverse.fetch_player_stats(historical_seasons, offline=offline, settings=settings)
        )
        team_stats = pl.read_parquet(
            nflverse.fetch_team_stats(historical_seasons, offline=offline, settings=settings)
        )
        schedules = pl.read_parquet(
            nflverse.fetch_schedules(historical_seasons, offline=offline, settings=settings)
        )
        pbp = pl.read_parquet(
            nflverse.fetch_pbp(historical_seasons, offline=offline, settings=settings)
        )
    except OfflineCacheMiss:
        logger.warning(
            "Historical nflverse tables not cached -- DST/K VOR will use the standard "
            "fixed-point replacement level (the Nth-best preseason total), not the "
            "streaming-aware one. Run HANDOFF.md's data rebuild sequence to enable it."
        )
        return {}

    scored_stats = streaming_tool.score_historical_stats(
        player_stats, team_stats, schedules, pbp, scoring_settings
    )
    return streaming_tool.streaming_replacement_overrides(
        scored_stats, n_drafted_by_position=n_drafted_by_position
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
    } - set(settings.draft.excluded_positions)
    scoped = with_team.filter(pl.col("position").is_in(list(eligible_positions)))
    streaming_overrides = _streaming_replacement_overrides(
        league_format,
        scoring_settings,
        board_positions=eligible_positions,
        season=season,
        offline=offline,
        settings=settings,
    )
    with_vor = vor_tool.compute_vor(
        scoped, league_format, replacement_overrides=streaming_overrides
    )

    rosters_raw: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=offline, settings=settings).read_text()
    )
    keeper_ids = adp_tool.keeper_sleeper_ids(rosters_raw)
    keeper_keys = adp_tool.keeper_join_keys(keeper_ids, players_dim)
    with_keeper_flag = with_vor.with_columns(
        pl.col("join_key").is_in(list(keeper_keys)).alias("is_keeper")
    )

    adp_raw = json.loads(
        rankings.fetch_adp(
            season, teams=league_format.n_teams, offline=offline, settings=settings
        ).read_text()
    )
    adp_df = aggregate.add_join_key(rankings.normalize_adp(adp_raw, season=season))
    with_adp = adp_tool.join_adp(with_keeper_flag, adp_df)

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


def _per_source_rank_columns(df: pl.DataFrame) -> list[str]:
    """Every `rank_<source>` column in `df`, sorted. `rank_sd` is the
    cross-source dispersion column `aggregate.aggregate_source_ranks`
    itself adds, not a source -- excluded despite the prefix match (a real
    bug caught live: selecting it both explicitly and via this list raised
    a polars DuplicateError).
    """
    return sorted(col for col in df.columns if col.startswith("rank_") and col != "rank_sd")


def _cbs_manual_name_candidates(
    other_sources: list[pl.DataFrame],
) -> dict[str, dict[str, dict[str, tuple[str, list[float]]]]]:
    """Real full names (and each identity's own ranks, for the ambiguity
    tie-break below) from `other_sources` (the other six manual-ranking
    DataFrames), grouped by position and then by the same "<first-initial>.
    <rest>" abbreviated form CBS's manual export uses ("Jahmyr Gibbs" ->
    "J. Gibbs"), normalised via `mapping.normalize_name` -- the exact-match
    half of `_resolve_cbs_manual_names`. Each abbreviated-form bucket is
    itself keyed by real player identity (`aggregate.add_join_key`'s own
    join_key, which already folds in the DST/nickname alias table and
    suffix-stripping) rather than the raw name string, so "Kenneth Walker
    III"/"Kenneth Walker"/"Ken Walker III" -- three real spellings of the
    same real player across three different sources -- collapse into one
    identity instead of looking like three candidates. Only a bucket with
    2+ *distinct* identities is a genuine ambiguity.

    Deliberately NOT `players_dim`: a real collision confirmed live,
    `players_dim`'s much larger universe (every player nflverse/Sleeper has
    ever tracked, including deep bench and long-retired players no real
    cheat sheet would list) has two real RBs who both abbreviate to
    "J. Taylor" -- "Jonathan Taylor" and an obscure "J'Mari Taylor" -- and
    CBS's own "J. Taylor" row silently resolved to whichever one happened
    to sort first, not the real Jonathan Taylor CBS actually ranked. The
    other six sources' own player pools are already scoped to exactly the
    same tier of real, currently-relevant players a cheat sheet covers.
    """
    grouped: dict[str, dict[str, dict[str, tuple[str, list[float]]]]] = {}
    for df in other_sources:
        keyed = aggregate.add_join_key(df.select("player_name", "position", "rank"))
        for row in keyed.iter_rows(named=True):
            full_name, position, join_key, rank = (
                row["player_name"],
                row["position"],
                row["join_key"],
                row["rank"],
            )
            if full_name is None or position is None:
                continue
            tokens = full_name.split()
            if not tokens:
                continue
            abbreviated = (
                f"{tokens[0][0]}. {' '.join(tokens[1:])}" if len(tokens) > 1 else tokens[0]
            )
            abbreviated_key = mapping.normalize_name(abbreviated)
            by_identity = grouped.setdefault(position, {}).setdefault(abbreviated_key, {})
            name, ranks = by_identity.get(join_key, (full_name, []))
            if len(full_name) > len(name):
                name = full_name  # prefer the longest spelling seen (keeps suffixes)
            if rank is not None:
                ranks.append(rank)
            by_identity[join_key] = (name, ranks)
    return grouped


def _resolve_cbs_manual_names(
    cbs_df: pl.DataFrame, other_sources: list[pl.DataFrame], *, fuzzy_floor: int = 90
) -> pl.DataFrame:
    """CBS's manual export abbreviates every name ("J. Gibbs") with no team
    column to disambiguate -- resolved here against the other six manual
    sources' own real full names (see `_cbs_manual_name_candidates` for why
    not `players_dim`, and for how same-player spelling variants across
    sources are collapsed to one identity rather than looking ambiguous),
    position-scoped (never cross-position, matching every other fuzzy-match
    precedent in this codebase, `ids.mapping.fuzzy_match_remainder`).
    Matching the same normalised abbreviated form exactly handles the
    overwhelming majority; rapidfuzz is the fallback for anything that
    doesn't (typos, an unusual abbreviation).

    Two *different* real players at the same position genuinely sharing one
    abbreviated form (e.g. Bijan Robinson vs. Brian Robinson Jr., confirmed
    live) is resolved by proximity: whichever candidate's own average rank
    (across the other sources) is closest to CBS's own rank for this row
    wins -- CBS ranking "B. Robinson" #2 overall is obviously about the
    player every other source *also* has near #2, not the one they have
    near #90, and the same logic correctly goes the other way for a
    "B. Robinson" row CBS ranks much lower. Only when neither candidate has
    any rank data to compare (never observed live, but not assumed away)
    does this fall back to leaving the row unresolved -- silently picking
    one with no signal at all would be worse than leaving it abbreviated.
    A row with no confident match keeps its original abbreviated name and
    logs a warning -- never silently dropped (CLAUDE.md rule 4).
    """
    candidates_by_position = _cbs_manual_name_candidates(other_sources)

    def resolve_one(position: str, abbreviated_name: str, cbs_rank: float | None) -> str | None:
        by_abbreviation = candidates_by_position.get(position, {})
        normalized = mapping.normalize_name(abbreviated_name)
        by_identity = by_abbreviation.get(normalized)
        if by_identity is not None:
            if len(by_identity) == 1:
                return next(iter(by_identity.values()))[0]
            if cbs_rank is not None:
                scored = [
                    (abs(statistics.mean(ranks) - cbs_rank), name)
                    for name, ranks in by_identity.values()
                    if ranks
                ]
                if scored:
                    return min(scored, key=lambda item: item[0])[1]
            return None
        flat_pool = {
            name: abbreviated_key
            for abbreviated_key, identities in by_abbreviation.items()
            for name, _ranks in identities.values()
        }
        if not flat_pool:
            return None
        choice = process.extractOne(
            normalized, flat_pool, scorer=fuzz.ratio, score_cutoff=fuzzy_floor
        )
        return choice[2] if choice is not None else None

    rows = cbs_df.select("position", "player_name", "rank").iter_rows(named=True)
    resolved = [resolve_one(row["position"], row["player_name"], row["rank"]) for row in rows]

    unresolved = [
        original
        for original, full_name in zip(cbs_df["player_name"].to_list(), resolved, strict=True)
        if full_name is None
    ]
    if unresolved:
        logger.warning(
            "%d CBS manual-ranking row(s) could not be resolved to a real player name, "
            "left abbreviated: %s",
            len(unresolved),
            unresolved,
        )

    final_names = [
        full_name if full_name is not None else original
        for original, full_name in zip(cbs_df["player_name"].to_list(), resolved, strict=True)
    ]
    return cbs_df.with_columns(pl.Series("player_name", final_names))


def _fetch_manual_rankings_sources() -> list[pl.DataFrame]:
    """Same graceful per-source degradation as `_fetch_point_sources` --
    a missing or malformed manual-ranking file (not yet uploaded, or
    uploaded with an unexpected layout) must not sink the whole "no model"
    board.
    """
    sources = []
    for source_name, (fetch, normalize) in manual_rankings.MANUAL_RANKING_FETCHERS.items():
        try:
            df = normalize(fetch())
        except Exception as exc:
            logger.warning("skipping manual ranking source %r: %s", source_name, exc)
            continue
        if df.height == 0:
            logger.warning("skipping manual ranking source %r: returned 0 rows", source_name)
            continue
        sources.append(df)
    return sources


def build_source_rankings(
    league: LeagueConfig,
    settings: Settings,
    *,
    season: int,
    offline: bool | None = None,
) -> pl.DataFrame:
    """Assemble the "no model" board: each source's own real, published
    overall rank side by side, plus avg/median/sd across sources (see
    `aggregate.aggregate_source_ranks`'s own docstring). Sourced entirely
    from `ingest.manual_rankings` -- manually-exported cheat sheets the
    project owner downloaded and dropped in `rankings/` at the repo root,
    each carrying a genuine overall rank, unlike this app's own live
    scrapers (`ingest.rankings`, used by `build_draft_board` instead,
    fetch per-stat projections for four of these seven sources, not a
    published rank of any kind). No VOR, no tiers, no ADP. `season`/
    `offline` are unused today (the manual files aren't season- or
    network-parameterised) -- kept for call-site symmetry with
    `build_draft_board` and in case a future upload becomes season-specific.
    """
    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    players_dim = mapping.build_players_dim(
        crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
    )

    manual_sources = _fetch_manual_rankings_sources()
    if not manual_sources:
        raise NoRankingsSourcesAvailableError(
            f"No manual ranking files found in {manual_rankings.MANUAL_RANKINGS_DIR} -- nothing "
            "to build source rankings from. Upload the cheat-sheet exports there (see "
            "ingest/manual_rankings.py for the expected filenames)."
        )
    non_cbs_sources = [df for df in manual_sources if df["source"][0] != "cbs"]
    manual_sources = [
        _resolve_cbs_manual_names(df, non_cbs_sources) if df["source"][0] == "cbs" else df
        for df in manual_sources
    ]

    all_sources = [aggregate.add_join_key(df) for df in manual_sources]
    ranks = aggregate.aggregate_source_ranks(all_sources)

    with_team = ranks.join(_team_by_join_key(players_dim), on="join_key", how="left")

    eligible_positions = {
        _POSITION_ALIASES.get(position, position)
        for position in mapping.league_relevant_positions(league)
    }
    scoped = with_team.filter(pl.col("position").is_in(list(eligible_positions)))

    rank_columns = _per_source_rank_columns(scoped)
    return scoped.select(
        pl.col("player_name").alias("player"),
        "position",
        "team",
        "avg_rank",
        "median_rank",
        "rank_sd",
        "n_sources",
        *rank_columns,
    ).sort("avg_rank")


__all__ = [
    "BOARD_COLUMNS",
    "POINT_SOURCE_NAMES",
    "RANK_SOURCE_NAMES",
    "NoRankingsSourcesAvailableError",
    "NotEnoughPicksError",
    "PickContext",
    "build_draft_board",
    "build_source_rankings",
    "draft_board_csv_path",
    "finalize_draft_board",
    "resolve_pick_context",
    "source_rankings_csv_path",
]
