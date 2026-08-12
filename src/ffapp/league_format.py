"""Parse a league's `roster_positions` into `LeagueFormat` (SPEC.md §8.5).

Nothing downstream (VOR, lineup optimiser, season simulator) may hardcode a
starting lineup shape -- CLAUDE.md rule 5. This module is the one place that
translates Sleeper's roster_positions list into structured counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from ffapp.config import LeagueConfig

_FLEX_SLOTS = ("FLEX", "SUPER_FLEX", "REC_FLEX")
_BENCH_SLOT = "BN"
_IR_SLOT = "IR"
_TAXI_SLOT = "TAXI"

# Sleeper serialises team defense as "DEF"; the rest of this codebase
# (scoring/keymap.py, scoring/stats.py) standardises on "DST".
_POSITION_ALIASES = {"DEF": "DST"}

_DEFAULT_FLEX_ELIGIBLE: dict[str, list[str]] = {
    "FLEX": ["RB", "WR", "TE"],
    "SUPER_FLEX": ["QB", "RB", "WR", "TE"],
    "REC_FLEX": ["WR", "TE"],
}

_FLEX_OVERRIDE_KEYS = {
    "FLEX": "flex_eligible",
    "SUPER_FLEX": "superflex_eligible",
    "REC_FLEX": "rec_flex_eligible",
}


@dataclass(frozen=True)
class LeagueFormat:
    n_teams: int
    starters: dict[str, int]
    flex_slots: dict[str, int]
    flex_eligible: dict[str, list[str]]
    bench: int
    ir: int
    playoff_week_start: int
    waiver_budget: int | None


def parse_league_format(league: LeagueConfig) -> LeagueFormat:
    roster_positions: list[str] = league.league_cache.get("roster_positions", [])

    starters: dict[str, int] = {}
    flex_slots: dict[str, int] = dict.fromkeys(_FLEX_SLOTS, 0)
    bench = 0
    ir = 0

    for slot in roster_positions:
        if slot in flex_slots:
            flex_slots[slot] += 1
        elif slot == _BENCH_SLOT:
            bench += 1
        elif slot == _IR_SLOT:
            ir += 1
        elif slot == _TAXI_SLOT:
            continue  # taxi squad isn't start-eligible; not modelled by LeagueFormat
        else:
            position = _POSITION_ALIASES.get(slot, slot)
            starters[position] = starters.get(position, 0) + 1

    flex_eligible = {
        slot: list(league.overrides.get(_FLEX_OVERRIDE_KEYS[slot], _DEFAULT_FLEX_ELIGIBLE[slot]))
        for slot in _FLEX_SLOTS
        if flex_slots[slot] > 0
    }

    return LeagueFormat(
        n_teams=league.league_cache.get("total_rosters") or 0,
        starters=starters,
        flex_slots=flex_slots,
        flex_eligible=flex_eligible,
        bench=bench,
        ir=ir,
        playoff_week_start=league.league_cache.get("playoff_week_start") or 0,
        waiver_budget=league.league_cache.get("waiver_budget"),
    )
