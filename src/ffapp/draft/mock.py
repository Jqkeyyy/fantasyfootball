"""Local mock draft simulator with bot opponents (not a numbered SPEC/TASKS.md
task -- built 2026-08-16 on direct request: manually re-creating every draft
in the real Sleeper app to practice strategy is too slow to "mass practice."

Not a new source of truth. Reuses this project's own already-built, already-
verified pieces rather than re-deriving any of them: the real draft board
(VOR/tier, `draft.board`) as the player pool, real locked keepers
(`tools.adp.keeper_sleeper_ids`/`keeper_join_keys`, same functions
`draft.board.build_draft_board` itself uses), the real trade-aware pick
order (`draft.pick_order`, since the primary league trades picks), and
`draft.live`'s own already-tested `starting_lineup_gaps`/`positional_run`
for the bot's need/run signals -- a bot deciding "do I still need a
starter here" or "is a position running" is exactly the same question the
live draft assistant already answers for a human, just asked on a team's
behalf instead of displayed to one.

Bot logic (confirmed design, not SPEC): sample (not argmax) from a weighted
distribution over the available pool -- `exp(-|adp - pick_no| / temperature)`
(a soft "stay near ADP, sometimes reach or fall") times a need multiplier
(strong boost for an unfilled starter/flex slot, soft suppression once a
position's bench is already stacked) times a run multiplier (mild boost
when `draft.live.positional_run` flags the position). `adp` here prefers,
in order: the project owner's own hand-exported Sleeper ADP file
(`ingest.manual_rankings.fetch_manual_sleeper_adp`, direct request
2026-08-16 -- a real snapshot the project owner controls, over this
project's own live `ingest.sleeper.fetch_sleeper_adp` API pull, which the
project owner found didn't match what real bot behaviour should look like);
then that live API pull, if the manual file has no coverage for a player;
then the board's own FFC-sourced `adp`; finally `overall_rank`, for a
player none of the three ADP sources cover at all.

Keepers occupy their real overall pick number (`draft.keepers`, resolved
from a hand-curated config, not Sleeper's own buggy `roster.keepers`
field -- see that module's docstring for the real mechanic and why it's
not trusted from the API). A keeper's pick number is intrinsic to the
player (last season's own draft cost), independent of this year's trades
or slot assignments -- reserving it for the real keeper-owner effectively
displaces whichever team structurally would have made that exact pick.
This means team roster sizes can legitimately differ across the draft
(a team whose own natural slot got claimed by someone ELSE's keeper, with
no keeper of their own to claim back an equivalent slot, ends up with
fewer total picks than a team with no such collision) -- a real, expected
outcome of how keeper leagues work, not a bug to balance away.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ffapp.config import CONFIG_DIR, LeagueConfig, Settings
from ffapp.draft import keepers as keepers_module
from ffapp.draft import live, pick_order
from ffapp.draft.board import draft_board_csv_path
from ffapp.ids.mapping import normalize_name
from ffapp.ingest import manual_rankings, sleeper
from ffapp.league_format import LeagueFormat, parse_league_format

logger = logging.getLogger("ffapp.draft.mock")

REACH_TEMPERATURE = 6.0
NEED_STARTER_BOOST = 2.5
NEED_SUPPRESS_FACTOR = 0.15
BENCH_SOFT_CAP_EXTRA = 2
RUN_BOOST = 1.5


class PlayerNotAvailableError(Exception):
    """Attempted to draft a `join_key` no longer (or never) in the pool."""


class DraftCompleteError(Exception):
    """Attempted to draft past the last real pick of the mock draft."""


class MockDraftBoardNotBuiltError(Exception):
    """No draft board CSV exists yet for this season -- `init_mock_draft`
    reads the real board `ffapp draft board` already writes rather than
    rebuilding VOR/tiers itself (see module docstring)."""


@dataclass
class MockDraftState:
    pool: pl.DataFrame
    team_rosters: dict[int, list[dict[str, Any]]]
    history: list[dict[str, Any]]
    pick_no: int
    n_teams: int
    num_rounds: int
    roster_by_slot: dict[int, int]
    traded_picks: list[pick_order.TradedPick]
    season: str
    my_roster_id: int
    my_slot: int | None
    league_format: LeagueFormat
    team_names: dict[int, str] = field(default_factory=dict)
    keeper_picks: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.num_rounds

    def is_complete(self) -> bool:
        return self.pick_no > self.total_picks


def _advance_past_keeper_picks(state: MockDraftState) -> None:
    """A keeper pick isn't a live decision -- it was already resolved
    (real overall pick number reserved for the real keeper-owner, see
    `draft.keepers`). Skip `state.pick_no` forward past every consecutive
    keeper pick so the turn order never asks anyone to "decide" one.
    """
    while not state.is_complete() and state.pick_no in state.keeper_picks:
        state.pick_no += 1


def _pick_no_to_round_slot(pick_no: int, n_teams: int) -> tuple[int, int]:
    round_num = (pick_no - 1) // n_teams + 1
    pos_in_round = (pick_no - 1) % n_teams
    slot = pos_in_round + 1 if round_num % 2 == 1 else n_teams - pos_in_round
    return round_num, slot


def current_pick_owner(state: MockDraftState) -> int:
    """The `roster_id` on the clock for `state.pick_no`, trade-aware (same
    `pick_order.pick_owner` the real board uses for "my next pick")."""
    round_num, slot = _pick_no_to_round_slot(state.pick_no, state.n_teams)
    return pick_order.pick_owner(
        round_num, slot, state.roster_by_slot, state.traded_picks, season=state.season
    )


def current_round(state: MockDraftState) -> int:
    """1-indexed round for `state.pick_no` -- for display only."""
    round_num, _slot = _pick_no_to_round_slot(state.pick_no, state.n_teams)
    return round_num


@dataclass(frozen=True)
class GridCell:
    round: int
    slot: int
    pick_no: int
    original_roster_id: int
    owner_roster_id: int
    player_name: str | None
    position: str | None
    is_mine: bool
    is_traded: bool
    is_current: bool
    is_keeper: bool


def draft_grid(state: MockDraftState) -> list[list[GridCell]]:
    """One row per round, one cell per real draft slot -- a slot's column
    never moves even when that pick is later traded, same layout the real
    Sleeper draft board itself uses (the image this was modeled on).
    `pick_order.pick_owner` resolves real trade ownership independently of
    whether a pick has actually happened yet, so this shows every one of
    the user's own upcoming picks across the *whole* draft, not just the
    immediate next one -- cheap, no simulation needed.

    A keeper pick (`state.keeper_picks`) overrides ownership entirely --
    its real owner is whoever holds that player as a keeper, independent
    of trades, same real mechanic `draft.keepers`' own docstring explains.
    `state.history` deliberately never contains keeper picks (see
    `build_mock_draft_state`), so a keeper pick_no is only ever found here.
    """
    picks_by_no = {pick["pick_no"]: pick for pick in state.history}
    rows: list[list[GridCell]] = []
    for round_num in range(1, state.num_rounds + 1):
        row: list[GridCell] = []
        for slot in range(1, state.n_teams + 1):
            pick_no = pick_order.snake_pick_number(round_num, slot, state.n_teams)
            original_roster_id = state.roster_by_slot[slot]
            keeper = state.keeper_picks.get(pick_no)
            made: dict[str, Any] | None
            if keeper is not None:
                owner_roster_id = keeper["roster_id"]
                made = keeper
                is_keeper = True
            else:
                owner_roster_id = pick_order.pick_owner(
                    round_num, slot, state.roster_by_slot, state.traded_picks, season=state.season
                )
                made = picks_by_no.get(pick_no)
                is_keeper = False
            row.append(
                GridCell(
                    round=round_num,
                    slot=slot,
                    pick_no=pick_no,
                    original_roster_id=original_roster_id,
                    owner_roster_id=owner_roster_id,
                    player_name=made["player_name"] if made else None,
                    position=made["metadata"]["position"] if made else None,
                    is_mine=owner_roster_id == state.my_roster_id,
                    is_traded=owner_roster_id != original_roster_id,
                    is_current=pick_no == state.pick_no,
                    is_keeper=is_keeper,
                )
            )
        rows.append(row)
    return rows


def my_upcoming_picks(rows: list[list[GridCell]]) -> list[int]:
    """Every pick number that belongs to the user and hasn't happened yet,
    ascending -- answers "when am I next" across the whole draft, not just
    the very next turn (which is always `state.pick_no` itself, since bots
    always resolve up to it).
    """
    return sorted(
        cell.pick_no for row in rows for cell in row if cell.is_mine and cell.player_name is None
    )


def _need_multiplier(
    position: str, roster_picks: list[dict[str, Any]], league_format: LeagueFormat
) -> float:
    gaps = live.starting_lineup_gaps(roster_picks, league_format)
    flex_needed_positions = {
        eligible
        for flex_type, remaining in gaps.items()
        if remaining > 0 and flex_type in league_format.flex_eligible
        for eligible in league_format.flex_eligible[flex_type]
    }
    if gaps.get(position, 0) > 0 or position in flex_needed_positions:
        return NEED_STARTER_BOOST

    count_at_position = sum(
        1 for pick in roster_picks if pick.get("metadata", {}).get("position") == position
    )
    soft_cap = league_format.starters.get(position, 0) + BENCH_SOFT_CAP_EXTRA
    if count_at_position >= soft_cap:
        return NEED_SUPPRESS_FACTOR
    return 1.0


def _run_multiplier(position: str, history: list[dict[str, Any]]) -> float:
    runs = live.positional_run(history, window=live.RUN_WINDOW)
    if runs.get(position, {}).get("is_run"):
        return RUN_BOOST
    return 1.0


def pick_weights(state: MockDraftState, roster_id: int) -> list[float]:
    """One weight per row of `state.pool`, in the same row order -- the
    real bot-decision math, kept separate from `bot_pick`'s own RNG draw so
    it can be inspected/tested deterministically without sampling.
    """
    roster_picks = state.team_rosters.get(roster_id, [])
    weights: list[float] = []
    for row in state.pool.iter_rows(named=True):
        adp = row["adp"] if row["adp"] is not None else float(state.total_picks + 1)
        base = math.exp(-abs(adp - state.pick_no) / REACH_TEMPERATURE)
        need = _need_multiplier(row["position"], roster_picks, state.league_format)
        run = _run_multiplier(row["position"], state.history)
        weights.append(base * need * run)
    return weights


def bot_pick(state: MockDraftState, roster_id: int, *, rng: random.Random) -> dict[str, Any]:
    """Choose (but do not record) the player `roster_id`'s bot drafts next."""
    if state.pool.height == 0:
        raise PlayerNotAvailableError("pool is empty")
    rows = state.pool.to_dicts()
    weights = pick_weights(state, roster_id)
    if sum(weights) <= 0:
        return min(rows, key=lambda r: r["adp"] if r["adp"] is not None else float("inf"))
    return rng.choices(rows, weights=weights, k=1)[0]


def record_pick(state: MockDraftState, roster_id: int, join_key: str) -> dict[str, Any]:
    """Apply a decision (bot's or the user's own) to `state`: move the
    player from pool to `roster_id`'s roster, append to history, advance
    `pick_no`. Mutates `state` in place -- held in Streamlit session state
    across many picks, not rebuilt fresh each time.
    """
    if state.is_complete():
        raise DraftCompleteError(f"draft already has {state.total_picks} picks")
    matches = state.pool.filter(pl.col("join_key") == join_key)
    if matches.height == 0:
        raise PlayerNotAvailableError(join_key)
    row = matches.row(0, named=True)
    first, _, last = row["player_name"].partition(" ")
    pick = {
        "pick_no": state.pick_no,
        "roster_id": roster_id,
        "metadata": {"first_name": first, "last_name": last or first, "position": row["position"]},
        "join_key": join_key,
        "player_name": row["player_name"],
        "team": row["team"],
        "vor": row["vor"],
        "tier": row["tier"],
        "adp": row["adp"],
    }
    state.team_rosters.setdefault(roster_id, []).append(pick)
    state.history.append(pick)
    state.pool = state.pool.filter(pl.col("join_key") != join_key)
    state.pick_no += 1
    return pick


def run_bot_picks_until_user_turn(
    state: MockDraftState, *, rng: random.Random
) -> list[dict[str, Any]]:
    """Resolve consecutive bot picks until it's `state.my_roster_id`'s turn,
    the draft ends, or the pool runs dry. Returns every pick made, in order,
    for the caller to show as a running log. Skips any keeper pick_no
    encountered along the way (or sitting at the very start) -- those were
    never a live decision to begin with.
    """
    made: list[dict[str, Any]] = []
    _advance_past_keeper_picks(state)
    while (
        not state.is_complete()
        and state.pool.height > 0
        and current_pick_owner(state) != state.my_roster_id
    ):
        roster_id = current_pick_owner(state)
        choice = bot_pick(state, roster_id, rng=rng)
        made.append(record_pick(state, roster_id, choice["join_key"]))
        _advance_past_keeper_picks(state)
    return made


def build_mock_draft_state(
    board: pl.DataFrame,
    sleeper_adp: pl.DataFrame,
    *,
    rosters_raw: list[dict[str, Any]],
    draft_order: dict[str, int],
    num_rounds: int,
    traded_picks: list[pick_order.TradedPick],
    keeper_assignments: list[keepers_module.KeeperAssignment],
    league_format: LeagueFormat,
    my_roster_id: int,
    team_names: dict[int, str],
    season: str,
    manual_sleeper_adp: pl.DataFrame | None = None,
) -> MockDraftState:
    """Pure assembly, no I/O -- `board` is the already-materialized draft
    board (`draft.board.draft_board_csv_path`'s own columns), `sleeper_adp`
    is `ingest.sleeper.normalize_sleeper_adp`'s own output, `keeper_assignments`
    is `draft.keepers.resolve_keeper_assignments`'s own output (real overall
    pick numbers, not Sleeper's own buggy `roster.keepers` field).
    `manual_sleeper_adp` (`ingest.manual_rankings.normalize_manual_sleeper_adp`'s
    own output, project owner's own direct request 2026-08-16, over the live
    API `sleeper_adp` -- a real hand-exported snapshot the project owner
    controls) wins when present: it has no `position` column at all (a real
    gap in that export), so it's joined by player name alone, not the usual
    `(name, position)` join_key every other source here uses.
    """
    with_key = board.with_columns(
        (
            pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key"),
        pl.col("player")
        .map_elements(normalize_name, return_dtype=pl.Utf8)
        .alias("normalized_name"),
    )
    adp_by_key = sleeper_adp.with_columns(
        (
            pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key")
    ).select("join_key", pl.col("adp").alias("sleeper_adp"))
    joined = with_key.join(adp_by_key, on="join_key", how="left")

    if manual_sleeper_adp is not None and manual_sleeper_adp.height > 0:
        manual_by_name = (
            manual_sleeper_adp.with_columns(
                pl.col("player_name")
                .map_elements(normalize_name, return_dtype=pl.Utf8)
                .alias("normalized_name")
            )
            .sort("adp")
            .unique(subset=["normalized_name"], keep="first")
            .select("normalized_name", pl.col("adp").alias("manual_sleeper_adp"))
        )
        joined = joined.join(manual_by_name, on="normalized_name", how="left")
    else:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("manual_sleeper_adp"))

    pool = joined.with_columns(
        pl.coalesce(["manual_sleeper_adp", "sleeper_adp", "adp", "overall_rank"])
        .cast(pl.Float64)
        .alias("effective_adp")
    ).select(
        "join_key",
        pl.col("player").alias("player_name"),
        "position",
        "team",
        "vor",
        "tier",
        "overall_rank",
        pl.col("effective_adp").alias("adp"),
    )

    roster_by_slot = pick_order.roster_id_by_slot(draft_order, rosters_raw)
    my_slot = next((slot for slot, rid in roster_by_slot.items() if rid == my_roster_id), None)

    team_rosters: dict[int, list[dict[str, Any]]] = {}
    keeper_picks: dict[int, dict[str, Any]] = {}
    keeper_join_keys: set[str] = set()
    for assignment in keeper_assignments:
        row_df = pool.filter(pl.col("join_key") == assignment.join_key)
        if row_df.height == 0:
            continue  # keeper's position isn't on this board (e.g. excluded DST/K)
        row = row_df.row(0, named=True)
        first, _, last = assignment.player_name.partition(" ")
        pick = {
            "pick_no": assignment.pick_no,
            "roster_id": assignment.roster_id,
            "metadata": {
                "first_name": first,
                "last_name": last or first,
                "position": assignment.position,
            },
            "join_key": assignment.join_key,
            "player_name": assignment.player_name,
            "team": assignment.team,
            "vor": row["vor"],
            "tier": row["tier"],
            "adp": row["adp"],
            "is_keeper": True,
        }
        keeper_join_keys.add(assignment.join_key)
        keeper_picks[assignment.pick_no] = pick
        team_rosters.setdefault(assignment.roster_id, []).append(pick)

    available_pool = pool.filter(~pl.col("join_key").is_in(list(keeper_join_keys)))

    state = MockDraftState(
        pool=available_pool,
        team_rosters=team_rosters,
        history=[],
        pick_no=1,
        n_teams=league_format.n_teams,
        num_rounds=num_rounds,
        roster_by_slot=roster_by_slot,
        traded_picks=traded_picks,
        season=season,
        my_roster_id=my_roster_id,
        my_slot=my_slot,
        league_format=league_format,
        team_names=team_names,
        keeper_picks=keeper_picks,
    )
    _advance_past_keeper_picks(state)
    return state


def init_mock_draft(
    league: LeagueConfig, settings: Settings, *, season: int, offline: bool | None = None
) -> MockDraftState:
    """I/O-performing wrapper around `build_mock_draft_state`: fetches
    everything real it needs (real board CSV already built by
    `ffapp draft board`, Sleeper's own real ADP, real rosters/keepers/draft
    order/trades/owner display names) and assembles the state.
    """
    assert league.league_id is not None
    board_path = draft_board_csv_path(settings, season=season)
    if not board_path.exists():
        raise MockDraftBoardNotBuiltError(
            f"No draft board found at {board_path}. Run `ffapp draft board` to build one first."
        )
    board = pl.read_csv(board_path)

    sleeper_adp_raw = json.loads(
        sleeper.fetch_sleeper_adp(season, offline=offline, settings=settings).read_text()
    )
    sleeper_adp = sleeper.normalize_sleeper_adp(sleeper_adp_raw, season=season)

    try:
        manual_sleeper_adp = manual_rankings.normalize_manual_sleeper_adp(
            manual_rankings.fetch_manual_sleeper_adp()
        )
    except Exception as exc:
        logger.warning(
            "Could not load the manual Sleeper ADP export (%s) -- falling back to the live "
            "Sleeper ADP API for bot pick decisions.",
            exc,
        )
        manual_sleeper_adp = None

    rosters_raw: list[dict[str, Any]] = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=offline, settings=settings).read_text()
    )
    users_raw: list[dict[str, Any]] = json.loads(
        sleeper.fetch_users(league.league_id, offline=offline, settings=settings).read_text()
    )
    draft_raw = json.loads(
        sleeper.fetch_drafts(league.league_id, offline=offline, settings=settings).read_text()
    )[0]
    traded_picks = pick_order.parse_traded_picks(
        json.loads(
            sleeper.fetch_traded_picks(
                league.league_id, offline=offline, settings=settings
            ).read_text()
        )
    )
    assert settings.sleeper_username is not None
    user = json.loads(
        sleeper.fetch_user(
            settings.sleeper_username, offline=offline, settings=settings
        ).read_text()
    )
    my_roster_id = pick_order.resolve_my_roster_id(user["user_id"], rosters_raw)

    display_name_by_owner = {
        str(u["user_id"]): u.get("display_name") or str(u["user_id"]) for u in users_raw
    }
    team_names = {
        int(roster["roster_id"]): display_name_by_owner.get(
            str(roster.get("owner_id")), f"Team {roster['roster_id']}"
        )
        for roster in rosters_raw
        if roster.get("roster_id") is not None
    }

    league_format = parse_league_format(league)

    keeper_config_path = keepers_module.keeper_config_path(
        CONFIG_DIR, league_slug=league.slug, season=season
    )
    if keeper_config_path.exists():
        keeper_config = keepers_module.load_keeper_config(keeper_config_path)
        keeper_assignments = keepers_module.resolve_keeper_assignments(
            keeper_config.entries,
            board,
            users_raw=users_raw,
            rosters_raw=rosters_raw,
            n_teams=league_format.n_teams,
        )
    else:
        keeper_assignments = []

    return build_mock_draft_state(
        board,
        sleeper_adp,
        rosters_raw=rosters_raw,
        draft_order=draft_raw["draft_order"],
        num_rounds=int(draft_raw["settings"]["rounds"]),
        traded_picks=traded_picks,
        keeper_assignments=keeper_assignments,
        league_format=league_format,
        my_roster_id=my_roster_id,
        team_names=team_names,
        season=str(season),
        manual_sleeper_adp=manual_sleeper_adp,
    )


__all__ = [
    "BENCH_SOFT_CAP_EXTRA",
    "NEED_STARTER_BOOST",
    "NEED_SUPPRESS_FACTOR",
    "REACH_TEMPERATURE",
    "RUN_BOOST",
    "DraftCompleteError",
    "GridCell",
    "MockDraftBoardNotBuiltError",
    "MockDraftState",
    "PlayerNotAvailableError",
    "bot_pick",
    "build_mock_draft_state",
    "current_pick_owner",
    "current_round",
    "draft_grid",
    "init_mock_draft",
    "my_upcoming_picks",
    "pick_weights",
    "record_pick",
    "run_bot_picks_until_user_turn",
]
