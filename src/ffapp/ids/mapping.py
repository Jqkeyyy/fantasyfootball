"""Cross-source player id resolution (SPEC.md §7).

Pipeline, in order:
    1. load_crosswalk_base   -- the published ffverse/dynastyprocess id crosswalk
    2. layer_sleeper_ids     -- fill crosswalk gaps with Sleeper's own cross-reference
                                 fields; add rows for Sleeper players absent entirely
    3. fuzzy_match_remainder -- name-match whatever step 1+2 couldn't link
    4. assign_canonical_id   -- player_id = gsis_id, else synthetic_<hash>
    5. apply_overrides       -- config/id_overrides.csv, applied last, wins over everything

Every row survives every step — CLAUDE.md rule 4: never silently drop rows in a join.
A player who resolves to nothing but a synthetic id is still present in the table;
`unmatched_players()` is how you find them, not a dropped row.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import polars as pl
from rapidfuzz import fuzz, process

from ffapp.config import CONFIG_DIR, LeagueConfig, Settings
from ffapp.config import load_settings as _load_settings
from ffapp.ingest import nflverse, sleeper

ID_OVERRIDES_PATH = CONFIG_DIR / "id_overrides.csv"

# Sleeper roster slots that are draft-position placeholders, not player positions.
_NON_POSITION_SLOTS = {"BN", "IR", "TAXI", "FLEX", "SUPER_FLEX", "REC_FLEX"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT_RE = re.compile(r"[^\w\s]")

_BASE_COLUMNS = [
    "gsis_id",
    "sleeper_id",
    "pfr_id",
    "espn_id",
    "full_name",
    "position",
    "team",
    "birth_date",
    "normalized_name",
]

SOURCE_ID_COLUMNS = {
    "sleeper": "sleeper_id",
    "gsis": "gsis_id",
    "pfr": "pfr_id",
    "espn": "espn_id",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes, collapse whitespace."""
    lowered = name.lower()
    stripped = _PUNCT_RE.sub("", lowered)
    tokens = stripped.split()
    if tokens and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def load_crosswalk_base(csv_path: Path) -> pl.DataFrame:
    """Load the ffverse/dynastyprocess player-id crosswalk (SPEC §7 step 1)."""
    raw = pl.read_csv(csv_path, null_values=["NA"], infer_schema_length=None)
    base = raw.select(
        pl.col("gsis_id").cast(pl.Utf8),
        pl.col("sleeper_id").cast(pl.Utf8),
        pl.col("pfr_id").cast(pl.Utf8),
        pl.col("espn_id").cast(pl.Utf8),
        pl.col("name").alias("full_name"),
        pl.col("position"),
        pl.col("team"),
        pl.col("birthdate").cast(pl.Utf8).alias("birth_date"),
    )
    return base.with_columns(
        pl.col("full_name")
        .map_elements(normalize_name, return_dtype=pl.Utf8)
        .alias("normalized_name")
    )


def _load_sleeper_players(path: Path) -> pl.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [
        {
            "sleeper_id": str(sleeper_id),
            "full_name": p.get("full_name"),
            "position": p.get("position"),
            "team": p.get("team"),
            "gsis_id": p.get("gsis_id"),
            "espn_id": (str(p["espn_id"]) if p.get("espn_id") is not None else None),
            "birth_date": p.get("birth_date"),
            "search_rank": p.get("search_rank"),
            "active": p.get("active"),
        }
        for sleeper_id, p in raw.items()
    ]
    df = pl.DataFrame(records)
    return df.with_columns(
        pl.when(pl.col("full_name").is_not_null())
        .then(pl.col("full_name").map_elements(normalize_name, return_dtype=pl.Utf8))
        .otherwise(None)
        .alias("normalized_name")
    )


def layer_sleeper_ids(base: pl.DataFrame, sleeper_players_path: Path) -> pl.DataFrame:
    """Fill base's missing sleeper_id via Sleeper's own gsis_id field (SPEC §7 step 2),
    then append rows for Sleeper players linked to no crosswalk row at all.
    """
    sleeper_df = _load_sleeper_players(sleeper_players_path)

    gsis_to_sleeper = {
        row["gsis_id"]: row["sleeper_id"]
        for row in sleeper_df.filter(pl.col("gsis_id").is_not_null()).iter_rows(named=True)
    }

    filled = base.with_columns(
        pl.when(pl.col("sleeper_id").is_null())
        .then(pl.col("gsis_id").replace_strict(gsis_to_sleeper, default=None, return_dtype=pl.Utf8))
        .otherwise(pl.col("sleeper_id"))
        .alias("sleeper_id")
    )

    linked_sleeper_ids = set(filled.filter(pl.col("sleeper_id").is_not_null())["sleeper_id"])

    new_rows = sleeper_df.filter(~pl.col("sleeper_id").is_in(list(linked_sleeper_ids))).select(
        "gsis_id",
        "sleeper_id",
        pl.lit(None, dtype=pl.Utf8).alias("pfr_id"),
        "espn_id",
        "full_name",
        "position",
        "team",
        "birth_date",
        "normalized_name",
        "search_rank",
        "active",
    )

    filled_with_rank = filled.join(
        sleeper_df.select("sleeper_id", "search_rank", "active"), on="sleeper_id", how="left"
    )

    return pl.concat([filled_with_rank, new_rows], how="vertical_relaxed")


def fuzzy_match_remainder(df: pl.DataFrame, floor: int = 92) -> pl.DataFrame:
    """Match whatever step 2 left unlinked (SPEC §7 step 3): exact normalised-name
    match tiered by (name, position, team) then (name, position), falling back to
    rapidfuzz name-only matching with a similarity floor for the last tier.
    """
    records = df.to_dicts()

    crosswalk_candidates = [
        r for r in records if r["gsis_id"] is not None and r["sleeper_id"] is None
    ]
    sleeper_candidates = [
        r
        for r in records
        if r["gsis_id"] is None and r["sleeper_id"] is not None and r["full_name"] is not None
    ]

    matched_sleeper_ids: set[str] = set()
    merges: dict[str, dict[str, Any]] = {}  # crosswalk gsis_id -> the sleeper record it absorbs

    for sc in sleeper_candidates:
        pool = [c for c in crosswalk_candidates if c["gsis_id"] not in merges]

        tier1 = [
            c
            for c in pool
            if c["normalized_name"] == sc["normalized_name"]
            and c["position"] == sc["position"]
            and c["team"] == sc["team"]
        ]
        tier2 = [
            c
            for c in pool
            if c["normalized_name"] == sc["normalized_name"] and c["position"] == sc["position"]
        ]

        best = tier1[0] if tier1 else (tier2[0] if tier2 else None)
        if best is None and pool:
            choice = process.extractOne(
                sc["normalized_name"],
                {c["gsis_id"]: c["normalized_name"] for c in pool},
                scorer=fuzz.ratio,
                score_cutoff=floor,
            )
            if choice is not None:
                matched_gsis = choice[2]
                best = next(c for c in pool if c["gsis_id"] == matched_gsis)

        if best is not None:
            matched_sleeper_ids.add(sc["sleeper_id"])
            merges[best["gsis_id"]] = sc

    result_records = []
    for r in records:
        if r["gsis_id"] is not None and r["gsis_id"] in merges:
            sc = merges[r["gsis_id"]]
            merged = dict(r)
            merged["sleeper_id"] = sc["sleeper_id"]
            merged["espn_id"] = merged["espn_id"] or sc["espn_id"]
            merged["search_rank"] = sc["search_rank"]
            merged["active"] = sc["active"]
            result_records.append(merged)
        elif r["sleeper_id"] in matched_sleeper_ids and r["gsis_id"] is None:
            continue  # absorbed into the crosswalk row above
        else:
            result_records.append(r)

    return pl.DataFrame(result_records, schema=df.schema)


def _synthetic_id(
    sleeper_id: str | None, normalized_name: str | None, position: str | None, team: str | None
) -> str:
    key = sleeper_id if sleeper_id is not None else f"{normalized_name}|{position}|{team}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    return f"synthetic_{digest}"


def assign_canonical_id(df: pl.DataFrame) -> pl.DataFrame:
    """player_id = gsis_id when present, else a stable synthetic id (SPEC.md §6.2)."""
    columns = set(df.columns)
    normalized_name_col = (
        pl.col("normalized_name") if "normalized_name" in columns else pl.lit(None, dtype=pl.Utf8)
    )
    position_col = pl.col("position") if "position" in columns else pl.lit(None, dtype=pl.Utf8)
    team_col = pl.col("team") if "team" in columns else pl.lit(None, dtype=pl.Utf8)

    fallback_struct = pl.struct(
        [
            pl.col("sleeper_id").alias("sleeper_id"),
            normalized_name_col.alias("normalized_name"),
            position_col.alias("position"),
            team_col.alias("team"),
        ]
    ).map_elements(
        lambda s: _synthetic_id(s["sleeper_id"], s["normalized_name"], s["position"], s["team"]),
        return_dtype=pl.Utf8,
    )

    return df.with_columns(
        pl.when(pl.col("gsis_id").is_not_null())
        .then(pl.col("gsis_id"))
        .otherwise(fallback_struct)
        .alias("player_id")
    )


def apply_overrides(df: pl.DataFrame, overrides_path: Path) -> pl.DataFrame:
    """Apply config/id_overrides.csv (SPEC §7 step 4) last, overriding everything."""
    overrides = pl.read_csv(overrides_path)
    if overrides.height == 0:
        return df

    result = df
    for source, column in SOURCE_ID_COLUMNS.items():
        source_overrides = overrides.filter(pl.col("source") == source)
        if source_overrides.height == 0:
            continue
        mapping = {
            str(row["source_id"]): row["player_id"]
            for row in source_overrides.iter_rows(named=True)
        }
        result = result.with_columns(
            pl.when(pl.col(column).is_in(list(mapping)))
            .then(pl.col(column).replace_strict(mapping, default=None, return_dtype=pl.Utf8))
            .otherwise(pl.col("player_id"))
            .alias("player_id")
        )
    return result


def build_players_dim(
    crosswalk_path: Path,
    sleeper_players_path: Path,
    overrides_path: Path | None = None,
    *,
    fuzzy_floor: int = 92,
) -> pl.DataFrame:
    """Run the full pipeline: SPEC §7 steps 1-4."""
    base = load_crosswalk_base(crosswalk_path)
    layered = layer_sleeper_ids(base, sleeper_players_path)
    matched = fuzzy_match_remainder(layered, floor=fuzzy_floor)
    assigned = assign_canonical_id(matched)
    if overrides_path is not None and Path(overrides_path).exists():
        assigned = apply_overrides(assigned, overrides_path)
    return assigned


def unmatched_players(players_dim: pl.DataFrame) -> pl.DataFrame:
    """Rows not resolved to a real cross-source id, ranked by Sleeper search_rank."""
    return players_dim.filter(pl.col("player_id").str.starts_with("synthetic_")).sort(
        "search_rank", nulls_last=True
    )


def within_top_n(unmatched: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """The subset of `unmatched_players()` relevant enough to be a build failure."""
    return unmatched.filter(pl.col("search_rank").is_not_null() & (pl.col("search_rank") <= top_n))


def league_relevant_positions(league: LeagueConfig) -> set[str]:
    """The player positions this league actually starts (CLAUDE.md rule 5: derived
    from the league's own config, never hardcoded).

    Sleeper's search_rank spans every player it tracks, including positions a given
    league may not roster (IDP) at all, so the top-N relevance gate needs scoping to
    what this league's `roster_positions` actually resolve to, FLEX slots included.
    """
    roster_positions = league.league_cache.get("roster_positions", [])
    positions = {p for p in roster_positions if p not in _NON_POSITION_SLOTS}
    if "FLEX" in roster_positions:
        positions |= set(league.overrides.get("flex_eligible", []))
    if "SUPER_FLEX" in roster_positions:
        positions |= set(league.overrides.get("superflex_eligible", []))
    return positions


def league_relevant(df: pl.DataFrame, eligible_positions: set[str]) -> pl.DataFrame:
    """Rows for currently-rosterable players at a position this league starts.

    Sleeper's search_rank includes retired players (active=False) alongside current
    ones, and a null `active` (no Sleeper data at all, e.g. a crosswalk-only row) is
    treated the same as inactive rather than assumed relevant. Sleeper's active=True
    only means "not purged from Sleeper's system" -- a cut or unsigned player can
    still be active=True with team=null, so a real current NFL team is required too.
    """
    return df.filter(
        pl.col("position").is_in(list(eligible_positions))
        & pl.col("active").fill_null(False)
        & pl.col("team").is_not_null()
    )


def unmatched_report(
    season: int,
    *,
    settings: Settings | None = None,
    offline: bool | None = None,
) -> pl.DataFrame:
    """Players present in Sleeper's dictionary but not resolved to a real cross-source
    id (SPEC §7's mandatory guardrail), ranked by fantasy relevance.

    Ranked by Sleeper's own `search_rank` rather than ADP: consensus-rankings
    ingestion (task 0.7) doesn't exist yet, and search_rank is already cached and a
    verified stand-in (rank 1-10 today are Bijan Robinson, Josh Allen, etc.). Revisit
    once 0.7 lands real ADP.

    `season` is accepted per SPEC §7's mandatory signature; the crosswalk and Sleeper's
    player dictionary are both always-current snapshots today, not season-partitioned.
    """
    settings = settings or _load_settings()
    crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
    overrides_path = ID_OVERRIDES_PATH if ID_OVERRIDES_PATH.exists() else None

    dim = build_players_dim(crosswalk_path, sleeper_players_path, overrides_path)
    return unmatched_players(dim)


__all__ = [
    "ID_OVERRIDES_PATH",
    "SOURCE_ID_COLUMNS",
    "apply_overrides",
    "assign_canonical_id",
    "build_players_dim",
    "fuzzy_match_remainder",
    "layer_sleeper_ids",
    "league_relevant",
    "league_relevant_positions",
    "load_crosswalk_base",
    "normalize_name",
    "unmatched_players",
    "unmatched_report",
    "within_top_n",
]
