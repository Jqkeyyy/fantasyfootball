"""ADP, pick-survival probability, and opportunity cost (SPEC.md §9.6; task 0.11).

Also handles two real-world mechanics SPEC §9.6 doesn't anticipate (it's
written for a plain redraft with a fixed slot every round) -- confirmed
against the primary league's real 2026 data and scoped with the project
owner before building:

- **Keepers.** This is a 1-keeper league (`league.settings.max_keepers`).
  Kept players never enter the draft pool at all -- SPEC-ADDENDUM-01.md's
  own scope note ("Dynasty or keeper league valuation... Redraft only")
  covers *valuation methodology*, not this: a kept player left in the
  draftable pool would get a real, wrong survival probability and
  opportunity cost for a pick that will never happen. Sleeper's roster
  objects carry a `keepers` field directly (list of sleeper player_ids);
  `keeper_join_keys` resolves those to the same normalized `(name,
  position)` join key `projections/aggregate.py` already uses.
- **Traded picks.** Handled upstream by `draft.pick_order` -- this module
  just consumes whatever pick numbers it's given, real or not.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from scipy.stats import norm

ADP_COLUMNS = ["adp", "adp_sd", "adp_high", "adp_low", "times_drafted"]


def keeper_sleeper_ids(rosters_raw: list[dict[str, Any]]) -> set[str]:
    """Every sleeper player_id currently marked as a keeper, across every
    roster in the league (Sleeper's raw `/league/{id}/rosters` payload). A
    roster with no keeper set yet (`keepers` null or absent) contributes
    nothing -- not an error, just "no keeper locked in yet."
    """
    ids: set[str] = set()
    for roster in rosters_raw:
        for player_id in roster.get("keepers") or []:
            ids.add(str(player_id))
    return ids


def keeper_join_keys(sleeper_ids: set[str], players_dim: pl.DataFrame) -> set[str]:
    """Resolve keeper sleeper_ids to the normalized `(name, position)` join
    key `projections/aggregate.py.add_join_key` produces, via `players_dim`
    (`ids/mapping.py.build_players_dim`'s own `sleeper_id`/`normalized_name`/
    `position` columns -- `normalized_name` is already `normalize_name`
    applied, so no need to recompute it here).
    """
    matches = players_dim.filter(pl.col("sleeper_id").is_in(list(sleeper_ids)))
    return {
        f"{row['normalized_name']}|{row['position']}"
        for row in matches.select(["normalized_name", "position"]).iter_rows(named=True)
    }


def exclude_keepers(df: pl.DataFrame, join_keys: set[str]) -> pl.DataFrame:
    """Drop rows whose `join_key` is a locked-in keeper -- a deliberate,
    documented business-rule filter (these players will never be drafted),
    not the kind of accidental join-drop CLAUDE.md rule 4 warns against.
    """
    if not join_keys:
        return df
    return df.filter(~pl.col("join_key").is_in(list(join_keys)))


def join_adp(df: pl.DataFrame, adp_df: pl.DataFrame) -> pl.DataFrame:
    """Left-join ADP (+ spread) onto a projections table by `join_key` --
    both sides must already carry one (`projections/aggregate.add_join_key`).
    A player projected but not covered by the ADP source keeps every ADP
    column null rather than being dropped (CLAUDE.md rule 4): thin ADP
    coverage is real for deep bench players, not a bug.
    """
    adp_cols = adp_df.select(["join_key", *ADP_COLUMNS])
    return df.join(adp_cols, on="join_key", how="left")


def p_available(
    pick: int, adp_mean: float | None, adp_sd: float | None, *, adp_sd_fallback: float
) -> float:
    """SPEC §9.6: `P(available at pick k) = 1 - Phi((k - adp_mean) / adp_sd)`.

    A player with no `adp_mean` at all (not covered by the ADP source --
    typically a deep/undrafted-in-practice player) is treated as certain to
    be available at any pick in this draft, rather than propagating a null
    through `opportunity_cost`.
    """
    if adp_mean is None:
        return 1.0
    sd = adp_sd if adp_sd is not None and adp_sd > 0 else adp_sd_fallback
    return float(1.0 - norm.cdf((pick - adp_mean) / sd))


def add_survival_probabilities(
    df: pl.DataFrame, *, next_pick: int, after_next_pick: int, adp_sd_fallback: float
) -> pl.DataFrame:
    """Add `p_avail_next`/`p_avail_after_next` (SPEC §9.6) for every row,
    given the two upcoming overall pick numbers this drafter actually owns
    (from `draft.pick_order.my_pick_numbers`, trade-aware).
    """
    return df.with_columns(
        pl.struct(["adp", "adp_sd"])
        .map_elements(
            lambda s: p_available(
                next_pick, s["adp"], s["adp_sd"], adp_sd_fallback=adp_sd_fallback
            ),
            return_dtype=pl.Float64,
        )
        .alias("p_avail_next"),
        pl.struct(["adp", "adp_sd"])
        .map_elements(
            lambda s: p_available(
                after_next_pick, s["adp"], s["adp_sd"], adp_sd_fallback=adp_sd_fallback
            ),
            return_dtype=pl.Float64,
        )
        .alias("p_avail_after_next"),
    )


def expected_best_available_vor(
    position_df: pl.DataFrame,
    pick: int,
    *,
    adp_sd_fallback: float,
    vor_column: str = "vor",
) -> float:
    """E[best available VOR at this position at `pick`] -- SPEC §9.6's
    `opportunity_cost` needs this per position.

    Ranks the position's players by VOR descending, then for each player i
    (in that order) treats "player i is the best one still available" as
    "player i is available AND every higher-VOR player at this position is
    not" (independent survival probabilities -- the same simplifying
    assumption SPEC's own single-player p_avail formula already makes).
    `E = sum(vor_i * P(i is the best available))`; the remaining mass (every
    player already gone) contributes 0, not a phantom below-replacement
    value.
    """
    ranked = position_df.sort(vor_column, descending=True)
    vors = ranked[vor_column].to_list()
    adp_means = ranked["adp"].to_list()
    adp_sds = ranked["adp_sd"].to_list()

    expected = 0.0
    survival_of_all_higher = 1.0
    for vor, adp_mean, adp_sd in zip(vors, adp_means, adp_sds, strict=True):
        avail = p_available(pick, adp_mean, adp_sd, adp_sd_fallback=adp_sd_fallback)
        expected += vor * avail * survival_of_all_higher
        survival_of_all_higher *= 1.0 - avail
    return expected


def add_opportunity_cost(
    df: pl.DataFrame,
    *,
    fallback_pick: int,
    adp_sd_fallback: float,
    vor_column: str = "vor",
) -> pl.DataFrame:
    """Add `opportunity_cost` (SPEC §9.6): `vor[p] - E[best available VOR at
    position(p) at my next pick]`.

    `fallback_pick` is deliberately NOT the same pick number as
    `p_avail_next`'s -- confirmed with the project owner after live testing
    surfaced the ambiguity: SPEC §9.6 reuses "my next pick" for both, but
    read literally that makes opportunity_cost near-self-referential (taking
    player P now vs. the pool available at the very pick you're using to
    take him -- almost every elite player is still "available" at their own
    selection). The decision this number actually answers is "if I pass on P
    now, what's the best I'll likely get at his position when my turn comes
    around again" -- so `fallback_pick` must be the picker's *next* pick
    AFTER the one currently being decided (`my_pick_numbers[1]` when
    evaluating the choice at `my_pick_numbers[0]`, i.e. what
    `add_survival_probabilities` calls `after_next_pick`), not
    `p_avail_next`'s own pick number. Passing the same pick used for
    `p_avail_next` here is the trap to avoid.

    Computed once per position (every player at a position shares the same
    expected-best-available-elsewhere value) and broadcast back onto every
    row.
    """
    positions = df["position"].unique().to_list()
    expected_by_position = {
        position: expected_best_available_vor(
            df.filter(pl.col("position") == position),
            fallback_pick,
            adp_sd_fallback=adp_sd_fallback,
            vor_column=vor_column,
        )
        for position in positions
    }
    expected_expr = pl.col("position").replace_strict(expected_by_position, return_dtype=pl.Float64)
    return df.with_columns((pl.col(vor_column) - expected_expr).alias("opportunity_cost"))


__all__ = [
    "ADP_COLUMNS",
    "add_opportunity_cost",
    "add_survival_probabilities",
    "exclude_keepers",
    "expected_best_available_vor",
    "join_adp",
    "keeper_join_keys",
    "keeper_sleeper_ids",
    "p_available",
]
