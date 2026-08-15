"""Decomposed model v2, Stage 3: efficiency priors (SPEC.md §11.4; not a
numbered TASKS.md task -- see docs/design-model-v2-stage3-efficiency-priors.md
for the full design). Predicts a player's own expected yards and
touchdown rate per touch, split by touch type (target vs. carry), as an
Empirical Bayes shrinkage estimate (SPEC's own literal RULE) -- no
trained model, continuing Stage 1/2's own precedent.

**No opponent-adjustment offset.** SPEC §11.4's own Stage 3 input list
names "opponent adjusted rates" alongside "player efficiency history,"
and an earlier version of this module applied one -- a same-week
additive offset built from `player_week_features`'s `def_adj_*_<group>`
columns (task 1.9's own already-lag-safe opponent-adjustment mapping).
It was removed 2026-08-15 on real evidence, not a design preference: a
paired-bootstrap ablation against real 2021-2025 data (see
`docs/JOURNAL.md`'s Stage 3 entries for the full numbers) showed the
offset measurably HURT `yards_per_carry` (RB and QB) and
`yards_per_target` (RB and WR) -- CIs excluding zero, i.e. a real,
not-noise effect -- and was statistically indistinguishable from zero
for every other position/output combination, including TE (whose own
point estimate sat between RB's and WR's, both confirmed harmful, with
no evidence its wider CI reflected a different underlying effect rather
than a smaller sample). Keeping the offset for TE alone would have been
selecting on noise. This is a finding about the specific additive
functional form this stage used -- treating a QB's own rushing matchup
identically to an RB's, on top of an already-shrunk task 1.8 estimate --
not a finding that opponent adjustment itself is worthless; a future
rework (a multiplicative form, a QB-specific term, or further shrinking
the adjustment itself rather than applying task 1.8's own estimate at
full weight) is a real option, not something this removal forecloses.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB
from ffapp.models.baselines import pooled_rolling_mean

TARGET_COLUMNS = ["yards_per_target", "td_rate_per_target", "yards_per_carry", "td_rate_per_carry"]

TARGET_PRIOR_WEIGHT = 50.0
CARRY_PRIOR_WEIGHT = 80.0


@dataclass(frozen=True)
class _OutputSpec:
    """Per-output config, one of `TARGET_COLUMNS` -- matches this
    project's own established convention for structured per-item config
    (CLAUDE.md's own "dataclasses for structured config objects", the
    same pattern `features/usage.py`'s `_WindowedFeature` already uses),
    not a raw heterogeneous dict (which `mypy` cannot type-check field
    access on)."""

    numerator: str
    denominator: str
    positions: list[str]
    prior_weight: float


_OUTPUT_SPECS: dict[str, _OutputSpec] = {
    "yards_per_target": _OutputSpec(
        numerator="receiving_yards",
        denominator="targets",
        positions=PASS_CATCHERS_AND_RB,
        prior_weight=TARGET_PRIOR_WEIGHT,
    ),
    "td_rate_per_target": _OutputSpec(
        numerator="receiving_tds",
        denominator="targets",
        positions=PASS_CATCHERS_AND_RB,
        prior_weight=TARGET_PRIOR_WEIGHT,
    ),
    "yards_per_carry": _OutputSpec(
        numerator="rushing_yards",
        denominator="carries",
        positions=RB_QB,
        prior_weight=CARRY_PRIOR_WEIGHT,
    ),
    "td_rate_per_carry": _OutputSpec(
        numerator="rushing_tds",
        denominator="carries",
        positions=RB_QB,
        prior_weight=CARRY_PRIOR_WEIGHT,
    ),
}


def build_efficiency_table(
    player_week_features: pl.DataFrame,
    player_week_usage: pl.DataFrame,
    player_week_stats: pl.DataFrame,
) -> pl.DataFrame:
    """One row per real `(player_id, season, week)` from
    `player_week_features`, joined to `player_week_usage`'s real same-week
    `targets`/`carries` and `player_week_stats`'s real same-week
    `receiving_yards`/`receiving_tds`/`rushing_yards`/`rushing_tds`. A
    real active-roster player-week with no recorded stat line resolves to
    0 real usage (SPEC §11.1, same treatment `features/build.py` and
    Stage 2's own `build_opportunity_table` already give this exact
    case) -- correct for a trailing SUM (a real 0 contributes 0 either
    way) and required so a missing join row can't poison a later
    `cum_sum()` with an unintended null.
    """
    features = player_week_features.select("player_id", "season", "week", "team", "position")
    with_usage = features.join(
        player_week_usage.select("player_id", "season", "week", "targets", "carries"),
        on=["player_id", "season", "week"],
        how="left",
    )
    with_stats = with_usage.join(
        player_week_stats.select(
            "player_id",
            "season",
            "week",
            "receiving_yards",
            "receiving_tds",
            "rushing_yards",
            "rushing_tds",
        ),
        on=["player_id", "season", "week"],
        how="left",
    )
    table = with_stats.with_columns(
        pl.col("targets").fill_null(0),
        pl.col("carries").fill_null(0),
        pl.col("receiving_yards").fill_null(0),
        pl.col("receiving_tds").fill_null(0),
        pl.col("rushing_yards").fill_null(0),
        pl.col("rushing_tds").fill_null(0),
    ).sort(["player_id", "season", "week"])

    for output_name, spec in _OUTPUT_SPECS.items():
        table = _add_raw_ingredients(table, output_name, spec)
        table = _shrink(
            table,
            f"_n_touches_{output_name}",
            f"trailing_raw_{output_name}",
            f"league_mean_{output_name}",
            prior_weight=spec.prior_weight,
            out_col=f"_shrunk_{output_name}",
        )
        # expected_<output> is exactly the shrunk estimate, gated by
        # position eligibility -- no further combination step. No
        # clamping either: a touchdown is scored on a touch, so a real
        # per-week td_rate is always in [0, 1], and _shrunk_<output> is a
        # weighted average of two quantities (trailing_raw, league_mean)
        # already bounded in [0, 1] for the two td_rate outputs -- a
        # convex combination of two in-range values can't leave that
        # range, so there is nothing left to clamp.
        table = table.with_columns(
            pl.when(pl.col("position").is_in(spec.positions))
            .then(pl.col(f"_shrunk_{output_name}"))
            .otherwise(None)
            .alias(f"expected_{output_name}")
        )

    return table


def _add_raw_ingredients(table: pl.DataFrame, output_name: str, spec: _OutputSpec) -> pl.DataFrame:
    numerator = spec.numerator
    denominator = spec.denominator

    # Defensive: `cum_sum().shift(1).over(["player_id", "season"])` below
    # depends on row order within each player/season group. This function
    # is called four times per table (once per output), each call ending
    # in two `pooled_rolling_mean` joins (`how="left"`), and polars does
    # not guarantee a left join preserves left-side row order. The caller
    # (`build_efficiency_table`) already sorts once before the first
    # call, but this function's own correctness should not depend on an
    # unguaranteed ordering surviving three prior calls' worth of joins --
    # sort again here so this function is correct standalone.
    table = table.sort(["player_id", "season", "week"])

    # Real outcome (this stage's own evaluation target): that week's own
    # real per-touch rate. Undefined -- not zero -- for a player-week
    # with no real touches of that type, same "never fabricate a value
    # where none exists" discipline as everywhere else in this project.
    table = table.with_columns(
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(f"real_{output_name}")
    )

    # Player-level trailing rate: a real cumulative ratio, season-to-date
    # through week W-1, never week W's own outcome -- the identical
    # cum_sum().shift(1).over([player_id, season]) shape
    # models.baselines.add_b1_season_to_date_mean already uses, kept as a
    # real sum (not divided by week count) so it stays directly
    # comparable to SPEC's own "~50 targets/~80 carries" prior weight.
    # Resets each season, no prior-season carryover -- see the design
    # doc's own rationale (a player's own history resetting is the
    # shrinkage formula's intended degenerate case, not a gap to patch).
    cum_num = pl.col(numerator).cum_sum().shift(1).over(["player_id", "season"])
    cum_den = pl.col(denominator).cum_sum().shift(1).over(["player_id", "season"])
    n_touches_col = f"_n_touches_{output_name}"
    table = table.with_columns(cum_den.fill_null(0).alias(n_touches_col))
    table = table.with_columns(
        pl.when(pl.col(n_touches_col) > 0)
        .then(cum_num / pl.col(n_touches_col))
        .otherwise(None)
        .alias(f"trailing_raw_{output_name}")
    )

    # Positional pooled rate: two separate pooled_rolling_mean calls (raw
    # numerator, raw denominator) -- NOT one call on a precomputed
    # per-row ratio, which would suffer the identical mean-of-ratios flaw
    # the trailing rate above avoids, just pooled across players instead
    # of weeks. Both pooled means share the same real pooled n (players x
    # weeks) by construction, so dividing them is a correct ratio-of-sums.
    table = pooled_rolling_mean(table, "position", numerator, f"_pos_{output_name}_num_mean")
    table = pooled_rolling_mean(table, "position", denominator, f"_pos_{output_name}_den_mean")
    table = table.with_columns(
        pl.when(pl.col(f"_pos_{output_name}_den_mean") > 0)
        .then(pl.col(f"_pos_{output_name}_num_mean") / pl.col(f"_pos_{output_name}_den_mean"))
        .otherwise(None)
        .alias(f"league_mean_{output_name}")
    )

    return table


def _shrink(
    table: pl.DataFrame,
    n_touches_col: str,
    trailing_col: str,
    league_mean_col: str,
    *,
    prior_weight: float,
    out_col: str,
) -> pl.DataFrame:
    """SPEC's own literal RULE: `shrunk = (n*trailing + prior_weight*mean)
    / (n + prior_weight)`. `trailing_col.fill_null(0.0)` is safe here even
    though a null trailing rate is a real "no data yet" case, not a real
    0 -- it is always multiplied by `n_touches`, which is genuinely 0 in
    exactly the same rows where `trailing_col` is null, so the term
    contributes 0 to the sum regardless of what placeholder value fills
    the null. `league_mean_col` is deliberately NOT filled -- if the
    positional mean itself is null (the very first tracked season, no
    prior-season fallback available either), the whole shrunk estimate
    must stay null too, not silently substitute 0 for a real population
    mean.
    """
    return table.with_columns(
        (
            (
                pl.col(n_touches_col) * pl.col(trailing_col).fill_null(0.0)
                + prior_weight * pl.col(league_mean_col)
            )
            / (pl.col(n_touches_col) + prior_weight)
        ).alias(out_col)
    )


__all__ = ["CARRY_PRIOR_WEIGHT", "TARGET_COLUMNS", "TARGET_PRIOR_WEIGHT", "build_efficiency_table"]
