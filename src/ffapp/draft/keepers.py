"""Real keeper overrides for the mock draft (not a numbered SPEC/TASKS.md
task -- see `draft.mock`'s own module docstring for the broader context).

Sleeper's own `roster.keepers` field is known-buggy for this league --
confirmed live by the project owner, who spotted it incorrectly flagging
Jahmyr Gibbs (never actually kept) as a keeper. Rather than trust it,
keepers are hand-entered in `config/keepers_<league_slug>_<season>.yml`
(a real, committed config file, same precedent as `config/id_overrides
.csv` -- small, hand-curated, corrects a real upstream data bug) and
resolved here against the real board and real Sleeper roster/user data.

Real mechanic, confirmed by the project owner against their own live
draft board (2026-08-16): a keeper occupies the exact SAME overall pick
number the player was drafted at last season -- not "pulled from the
pool with no round," and not necessarily the round they'd go in THIS
year's ADP. Sleeper's own "<round>.<pick_within_round>" notation --
exactly what's hand-entered in the YAML, read straight off the real
draft board -- uses `pick_within_round` as the CHRONOLOGICAL Nth pick
made in that round, not a column/slot index: confirmed live, Jonathan
Taylor's real keeper cost "2.6" sat under a DIFFERENT team's column than
"slot 6" would occupy structurally, because round 2 is an even
(reversed) round -- the 6th pick made chronologically in round 2 lands
in column 5, not column 6. So converting to an overall pick number is
always `(round - 1) * n_teams + pick_within_round`, regardless of
whether the round is odd or even -- no snake-direction logic needed,
unlike `pick_order.snake_pick_number` (which goes the other way: column
-> chronological pick).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from rapidfuzz import fuzz, process

from ffapp.ids.mapping import normalize_name

PLAYER_MATCH_FLOOR = 85


class KeeperOwnerNotFoundError(Exception):
    """A keeper config entry's `owner` display name doesn't match any
    real user in this league -- fails loudly rather than silently
    dropping a real keeper (CLAUDE.md rule 4's spirit)."""


class KeeperPlayerNotFoundError(Exception):
    """A keeper config entry's `player` name didn't fuzzy-match any
    player on the real board closely enough to trust."""


@dataclass(frozen=True)
class RawKeeperEntry:
    owner: str
    player: str
    pick: str


@dataclass(frozen=True)
class KeeperConfig:
    season: int
    league_slug: str
    entries: list[RawKeeperEntry]


@dataclass(frozen=True)
class KeeperAssignment:
    pick_no: int
    roster_id: int
    join_key: str
    player_name: str
    position: str
    team: str | None


def keeper_config_path(config_dir: Path, *, league_slug: str, season: int) -> Path:
    return config_dir / f"keepers_{league_slug}_{season}.yml"


def load_keeper_config(path: Path) -> KeeperConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = [
        RawKeeperEntry(owner=e["owner"], player=e["player"], pick=str(e["pick"]))
        for e in raw.get("keepers", [])
    ]
    return KeeperConfig(season=raw["season"], league_slug=raw["league_slug"], entries=entries)


def parse_pick_notation(pick: str, *, n_teams: int) -> int:
    """ "<round>.<pick_within_round>" (Sleeper's own real draft-board
    notation, `pick_within_round` chronological, not a column index) ->
    overall pick number. See the module docstring for why this is plain
    arithmetic, not `pick_order.snake_pick_number`.
    """
    round_str, _, chron_str = pick.partition(".")
    round_num = int(round_str)
    chron_pick = int(chron_str)
    return (round_num - 1) * n_teams + chron_pick


def _roster_id_by_display_name(
    users_raw: list[dict[str, Any]], rosters_raw: list[dict[str, Any]]
) -> dict[str, int]:
    roster_id_by_owner_id = {
        str(r["owner_id"]): int(r["roster_id"])
        for r in rosters_raw
        if r.get("owner_id") is not None
    }
    return {
        (u.get("display_name") or "").strip().lower(): roster_id_by_owner_id[str(u["user_id"])]
        for u in users_raw
        if str(u["user_id"]) in roster_id_by_owner_id
    }


def resolve_keeper_assignments(
    entries: list[RawKeeperEntry],
    board: pl.DataFrame,
    *,
    users_raw: list[dict[str, Any]],
    rosters_raw: list[dict[str, Any]],
    n_teams: int,
) -> list[KeeperAssignment]:
    """Resolve every hand-entered keeper against real data: `owner` to a
    real `roster_id` (via Sleeper's own users/rosters), `player` to a
    real board row (fuzzy match -- the config is free-text, not
    guaranteed to match the board's own exact spelling), `pick` to a real
    overall pick number. Raises loudly, naming the offending entry,
    rather than silently skipping a real keeper the project owner
    explicitly entered.
    """
    roster_by_owner = _roster_id_by_display_name(users_raw, rosters_raw)
    name_pool = {
        row["player"]: row
        for row in board.select("player", "position", "team").unique().iter_rows(named=True)
    }

    assignments: list[KeeperAssignment] = []
    for entry in entries:
        roster_id = roster_by_owner.get(entry.owner.strip().lower())
        if roster_id is None:
            raise KeeperOwnerNotFoundError(
                f"Keeper config names owner {entry.owner!r} for player {entry.player!r} -- "
                "no real user in this league matches that display name."
            )

        match = process.extractOne(
            entry.player, list(name_pool), scorer=fuzz.ratio, score_cutoff=PLAYER_MATCH_FLOOR
        )
        if match is None:
            raise KeeperPlayerNotFoundError(
                f"Keeper config names player {entry.player!r} (owner {entry.owner!r}) -- no "
                "player on the real draft board matched closely enough "
                f"(floor {PLAYER_MATCH_FLOOR})."
            )
        matched_name = match[0]
        row = name_pool[matched_name]
        assignments.append(
            KeeperAssignment(
                pick_no=parse_pick_notation(entry.pick, n_teams=n_teams),
                roster_id=roster_id,
                join_key=f"{normalize_name(matched_name)}|{row['position']}",
                player_name=matched_name,
                position=row["position"],
                team=row["team"],
            )
        )
    return assignments


__all__ = [
    "PLAYER_MATCH_FLOOR",
    "KeeperAssignment",
    "KeeperConfig",
    "KeeperOwnerNotFoundError",
    "KeeperPlayerNotFoundError",
    "RawKeeperEntry",
    "keeper_config_path",
    "load_keeper_config",
    "parse_pick_notation",
    "resolve_keeper_assignments",
]
