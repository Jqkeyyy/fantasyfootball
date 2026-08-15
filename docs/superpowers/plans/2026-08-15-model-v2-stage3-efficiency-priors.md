# Decomposed Model v2 — Stage 3 (Efficiency Priors) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate Stage 3 of SPEC.md §11.4's decomposed model v2 — an Empirical Bayes shrinkage estimate (SPEC's own literal RULE) of a player's expected yards and touchdown rate per touch, split by touch type, plus an additive opponent-adjustment offset — and prove (against real 2021-2025 data) whether it beats a player's own trailing raw per-touch rate.

**Architecture:** No trained model. `models/efficiency.py::build_efficiency_table` computes, per player-week and per output (`yards_per_target`, `td_rate_per_target`, `yards_per_carry`, `td_rate_per_carry`): a real cumulative per-player trailing rate (season-to-date sums, never a mean of weekly ratios), a real pooled positional rate (ratio of two pooled sums, never a mean of per-player ratios), blends them via SPEC's Empirical Bayes formula, then adds an opponent-adjustment offset already sitting in `player_week_features` (task 1.9's `features.opponent.add_opponent_features` — no new join, no new table). Unlike Stage 2, this stage does not consume Stage 1's or Stage 2's own output at all.

**Tech Stack:** Python 3.11+, polars, pytest, `uv run`. No LightGBM in this stage.

**Spec:** `docs/design-model-v2-stage3-efficiency-priors.md`

## Global Constraints

- No random train/test splits — this stage adds no new time-ordering risk itself (no model fitting at all), but every trailing/pooled quantity must be `.shift(1)`'d so a target week's own outcome never leaks into its own prediction.
- Every real evaluation must scope to `season_type == "REG"` from the start, same convention Stage 1/2 both already established.
- Every real evaluation must report `n_obs` and CI alongside MAE (SPEC §12.5).
- Every real evaluation must filter to rows where the real per-touch rate is actually defined (≥1 real touch of that type that week) *before* calling `run_walk_forward_backtest` — the harness itself does not drop a null `target_column`, and `accuracy_metrics` only filters on a null `prediction`, not a null real outcome. This is a real correctness requirement, not an optional nicety — see Task 4.
- Type hints on every public function; `ruff check`/`ruff format --check`/`mypy src/` must stay clean.
- Tests before implementation (TDD) — write the failing test, watch it fail for the right reason, then implement.
- No live network calls in tests — small hand-built polars fixtures only, matching this repo's existing test files.
- No modification to `evaluation/backtest.py`, `evaluation/metrics.py`, `features/opponent.py`, or `models/baselines.py` — this stage consumes all four as-is, per the design doc's own explicit out-of-scope section.
- This stage produces no trained model and adds no wiring into any player-facing consumer (the draft board, weekly rankings, etc.) — stays standalone, same precedent as Stage 1/2.

---

### Task 1: Raw ingredients — real outcome, player-trailing rate, positional-pooled rate

**Files:**
- Create: `src/ffapp/models/efficiency.py`
- Test: `tests/test_models_efficiency.py`

**Interfaces:**
- Consumes: `features.usage.PASS_CATCHERS_AND_RB`/`RB_QB` (already public). `models.baselines.pooled_rolling_mean(df, group_column, target_column, output_column) -> pl.DataFrame` (already exists). `player_week_features` (shaped like `data/features/player_week_features.parquet` — real columns `player_id`, `season`, `week`, `team`, `position`, plus the ten `def_adj_ypt_allowed_<group>`/`def_adj_td_rate_allowed_<group>` columns for `<group>` in `wr`/`te`/`rb_receiving`/`rb_rushing`/`qb_rushing`, task 1.9). `player_week_usage` (shaped like `data/interim/player_week_usage.parquet` — real columns `player_id`, `season`, `week`, `targets`, `carries`). `player_week_stats` (shaped like `data/interim/player_week_stats.parquet` — real columns `player_id`, `season`, `week`, `receiving_yards`, `receiving_tds`, `rushing_yards`, `rushing_tds`).
- Produces: `efficiency.build_efficiency_table(player_week_features, player_week_usage, player_week_stats) -> pl.DataFrame` (incomplete after this task — no shrinkage or opponent-adjustment yet, Tasks 2-3 add those to the same function) and `efficiency.TARGET_COLUMNS = ["yards_per_target", "td_rate_per_target", "yards_per_carry", "td_rate_per_carry"]`. After this task, the output has, per output name in `TARGET_COLUMNS`: `real_<output>` (this week's own real per-touch rate, null if no real touches that week), `trailing_raw_<output>` (player's own cumulative rate through last week, null with 0 trailing touches), `_n_touches_<output>` (real cumulative touch count, never null — 0 not null), `league_mean_<output>` (pooled positional rate). Tasks 2-3 read all of these directly by name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_efficiency.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import efficiency


def _player_week_features() -> pl.DataFrame:
    """Two WRs on different teams (wr1, wr2 -- two distinct real teams so
    the league-average opponent adjustment Task 3 needs has more than one
    data point to average) and one RB (rb1, wr1's own teammate), two real
    weeks each. def_adj_* values are deliberately non-trivial (not all
    equal, not all zero) so Task 3's tests have real signal to check --
    Tasks 1-2 don't examine these columns at all."""
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "team": ["KC", "KC", "BUF", "BUF", "KC", "KC"],
            "position": ["WR", "WR", "WR", "WR", "RB", "RB"],
            "def_adj_ypt_allowed_wr": [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
            "def_adj_ypt_allowed_te": [0.0] * 6,
            "def_adj_ypt_allowed_rb_receiving": [0.0] * 6,
            "def_adj_ypt_allowed_rb_rushing": [1.0] * 6,
            "def_adj_ypt_allowed_qb_rushing": [0.0] * 6,
            "def_adj_td_rate_allowed_wr": [0.04, 0.04, 0.0, 0.0, 0.04, 0.04],
            "def_adj_td_rate_allowed_te": [0.0] * 6,
            "def_adj_td_rate_allowed_rb_receiving": [0.0] * 6,
            "def_adj_td_rate_allowed_rb_rushing": [0.02] * 6,
            "def_adj_td_rate_allowed_qb_rushing": [0.0] * 6,
        }
    )


def _player_week_usage() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "targets": [8, 10, 6, 6, 1, 1],
            "carries": [0, 0, 0, 0, 15, 18],
        }
    )


def _player_week_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "receiving_yards": [100, 120, 60, 60, 5, 8],
            "receiving_tds": [1, 1, 0, 0, 0, 0],
            "rushing_yards": [0, 0, 0, 0, 75, 90],
            "rushing_tds": [0, 0, 0, 0, 1, 1],
        }
    )


def _build_table() -> pl.DataFrame:
    return efficiency.build_efficiency_table(
        _player_week_features(), _player_week_usage(), _player_week_stats()
    )


def test_build_efficiency_table_computes_real_outcome_for_a_week_with_touches() -> None:
    result = _build_table()

    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["real_yards_per_target"] == pytest.approx(100 / 8)

    rb1_week1 = result.filter((pl.col("player_id") == "rb1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert rb1_week1["real_yards_per_carry"] == pytest.approx(75 / 15)


def test_build_efficiency_table_real_outcome_is_null_with_no_touches_that_week() -> None:
    result = _build_table()

    # wr1 has 0 real carries in both weeks -- real_yards_per_carry must be
    # null, not a fabricated 0.
    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["real_yards_per_carry"] is None


def test_build_efficiency_table_trailing_raw_is_null_in_a_players_first_tracked_week() -> None:
    result = _build_table()

    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["trailing_raw_yards_per_target"] is None
    assert wr1_week1["_n_touches_yards_per_target"] == 0


def test_build_efficiency_table_trailing_raw_is_a_ratio_of_cumulative_sums_not_a_mean_of_weekly_ratios() -> (
    None
):
    result = _build_table()

    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # Week 1's real cumulative sum (through week 1 only, week 2's own
    # outcome must never leak in): 100 yards / 8 targets = 12.5.
    assert wr1_week2["trailing_raw_yards_per_target"] == pytest.approx(100 / 8)
    assert wr1_week2["_n_touches_yards_per_target"] == 8


def test_build_efficiency_table_league_mean_is_a_ratio_of_pooled_sums_not_a_mean_of_player_ratios() -> (
    None
):
    result = _build_table()

    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # Pooled across BOTH real WRs' own week-1 real values: (100+60) yards
    # / (8+6) targets = 160/14 -- NOT the naive mean of each player's own
    # week-1 ratio ((100/8 + 60/6)/2 = 11.25), which would wrongly
    # equal-weight wr1's 8-target week and wr2's 6-target week.
    expected = (100 + 60) / (8 + 6)
    assert wr1_week2["league_mean_yards_per_target"] == pytest.approx(expected)
    naive_wrong_value = (100 / 8 + 60 / 6) / 2
    assert wr1_week2["league_mean_yards_per_target"] != pytest.approx(naive_wrong_value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_efficiency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffapp.models.efficiency'`

- [ ] **Step 3: Implement `models/efficiency.py`**

```python
"""Decomposed model v2, Stage 3: efficiency priors (SPEC.md §11.4; not a
numbered TASKS.md task -- see docs/design-model-v2-stage3-efficiency-priors.md
for the full design). Predicts a player's own expected yards and
touchdown rate per touch, split by touch type (target vs. carry), as an
Empirical Bayes shrinkage estimate (SPEC's own literal RULE) plus an
additive opponent-adjustment offset -- no trained model, per that same
RULE, continuing Stage 1/2's own precedent.

Unlike Stage 2, this stage does not depend on Stage 1's or Stage 2's own
output at all -- SPEC's own input list for Stage 3 names only "player
efficiency history" and "opponent adjusted rates," both already real,
already-available columns: `player_week_usage`/`player_week_stats` for
the first, and `player_week_features`'s own `def_adj_*_<group>` columns
(already mapped onto each player's own position by task 1.9's
`features.opponent.add_opponent_features`, already lag-safe) for the
second.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB
from ffapp.models.baselines import pooled_rolling_mean

TARGET_COLUMNS = ["yards_per_target", "td_rate_per_target", "yards_per_carry", "td_rate_per_carry"]

TARGET_PRIOR_WEIGHT = 50.0
CARRY_PRIOR_WEIGHT = 80.0

_ALL_GROUPS = ["wr", "te", "rb_receiving", "rb_rushing", "qb_rushing"]


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
    group_column_by_position: dict[str, str]
    prior_weight: float
    adj_prefix: str
    clamp: bool


_OUTPUT_SPECS: dict[str, _OutputSpec] = {
    "yards_per_target": _OutputSpec(
        numerator="receiving_yards",
        denominator="targets",
        positions=PASS_CATCHERS_AND_RB,
        group_column_by_position={"WR": "wr", "TE": "te", "RB": "rb_receiving"},
        prior_weight=TARGET_PRIOR_WEIGHT,
        adj_prefix="def_adj_ypt_allowed",
        clamp=False,
    ),
    "td_rate_per_target": _OutputSpec(
        numerator="receiving_tds",
        denominator="targets",
        positions=PASS_CATCHERS_AND_RB,
        group_column_by_position={"WR": "wr", "TE": "te", "RB": "rb_receiving"},
        prior_weight=TARGET_PRIOR_WEIGHT,
        adj_prefix="def_adj_td_rate_allowed",
        clamp=True,
    ),
    "yards_per_carry": _OutputSpec(
        numerator="rushing_yards",
        denominator="carries",
        positions=RB_QB,
        group_column_by_position={"RB": "rb_rushing", "QB": "qb_rushing"},
        prior_weight=CARRY_PRIOR_WEIGHT,
        adj_prefix="def_adj_ypt_allowed",
        clamp=False,
    ),
    "td_rate_per_carry": _OutputSpec(
        numerator="rushing_tds",
        denominator="carries",
        positions=RB_QB,
        group_column_by_position={"RB": "rb_rushing", "QB": "qb_rushing"},
        prior_weight=CARRY_PRIOR_WEIGHT,
        adj_prefix="def_adj_td_rate_allowed",
        clamp=True,
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
    adj_columns = [f"def_adj_ypt_allowed_{g}" for g in _ALL_GROUPS] + [
        f"def_adj_td_rate_allowed_{g}" for g in _ALL_GROUPS
    ]
    features = player_week_features.select(
        "player_id", "season", "week", "team", "position", *adj_columns
    )
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

    return table


def _add_raw_ingredients(table: pl.DataFrame, output_name: str, spec: _OutputSpec) -> pl.DataFrame:
    numerator = spec.numerator
    denominator = spec.denominator

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


__all__ = ["CARRY_PRIOR_WEIGHT", "TARGET_COLUMNS", "TARGET_PRIOR_WEIGHT", "build_efficiency_table"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_efficiency.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/efficiency.py tests/test_models_efficiency.py && uv run ruff format --check src/ffapp/models/efficiency.py && uv run mypy src/ffapp/models/efficiency.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/efficiency.py tests/test_models_efficiency.py
git commit -m "feat: add raw efficiency ingredients for model v2 Stage 3"
```

---

### Task 2: Empirical Bayes shrinkage

**Files:**
- Modify: `src/ffapp/models/efficiency.py`
- Test: `tests/test_models_efficiency.py`

**Interfaces:**
- Consumes: `_n_touches_<output>`, `trailing_raw_<output>`, `league_mean_<output>` (Task 1), `_OUTPUT_SPECS` (Task 1, for `prior_weight` per output).
- Produces: a new private `_shrink` function, and `build_efficiency_table`'s output gains `_shrunk_<output>` for every output in `TARGET_COLUMNS`. Task 3 reads `_shrunk_<output>` directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_efficiency.py — add to the existing file

def test_shrink_with_zero_touches_returns_exactly_the_positional_mean() -> None:
    table = pl.DataFrame({"n_touches": [0], "trailing_raw": [None], "league_mean": [8.0]})

    result = efficiency._shrink(
        table, "n_touches", "trailing_raw", "league_mean", prior_weight=50.0, out_col="shrunk"
    )

    assert result["shrunk"][0] == pytest.approx(8.0)


def test_shrink_with_touches_equal_to_prior_weight_returns_the_exact_midpoint() -> None:
    table = pl.DataFrame({"n_touches": [50], "trailing_raw": [12.0], "league_mean": [8.0]})

    result = efficiency._shrink(
        table, "n_touches", "trailing_raw", "league_mean", prior_weight=50.0, out_col="shrunk"
    )

    # (50*12 + 50*8) / (50+50) = 10.0 -- the exact midpoint between the
    # two inputs when n_touches equals the prior weight.
    assert result["shrunk"][0] == pytest.approx(10.0)


def test_shrink_with_a_very_large_touch_count_lands_close_to_the_players_own_rate() -> None:
    table = pl.DataFrame({"n_touches": [10_000], "trailing_raw": [12.0], "league_mean": [8.0]})

    result = efficiency._shrink(
        table, "n_touches", "trailing_raw", "league_mean", prior_weight=50.0, out_col="shrunk"
    )

    assert result["shrunk"][0] == pytest.approx(12.0, abs=0.05)


def test_build_efficiency_table_wires_shrinkage_into_all_four_outputs() -> None:
    result = _build_table()

    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    trailing = wr1_week2["trailing_raw_yards_per_target"]
    league_mean = wr1_week2["league_mean_yards_per_target"]
    n_touches = wr1_week2["_n_touches_yards_per_target"]
    expected = (n_touches * trailing + efficiency.TARGET_PRIOR_WEIGHT * league_mean) / (
        n_touches + efficiency.TARGET_PRIOR_WEIGHT
    )
    assert wr1_week2["_shrunk_yards_per_target"] == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_efficiency.py -k shrink -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.efficiency' has no attribute '_shrink'`

- [ ] **Step 3: Implement `_shrink` and wire it into `build_efficiency_table`**

Add this function to `src/ffapp/models/efficiency.py`, after `_add_raw_ingredients`:

```python
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
            (pl.col(n_touches_col) * pl.col(trailing_col).fill_null(0.0)
             + prior_weight * pl.col(league_mean_col))
            / (pl.col(n_touches_col) + prior_weight)
        ).alias(out_col)
    )
```

In `build_efficiency_table`, change the loop body to also call `_shrink`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_efficiency.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/efficiency.py tests/test_models_efficiency.py && uv run mypy src/ffapp/models/efficiency.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/efficiency.py tests/test_models_efficiency.py
git commit -m "feat: add Empirical Bayes shrinkage for model v2 Stage 3"
```

---

### Task 3: Opponent-adjustment offset, clamping, and position eligibility

**Files:**
- Modify: `src/ffapp/models/efficiency.py`
- Test: `tests/test_models_efficiency.py`

**Interfaces:**
- Consumes: `_shrunk_<output>` (Task 2), the ten `def_adj_*_<group>` columns (Task 1's own `SELECT`), `_OUTPUT_SPECS` (for `group_column_by_position`, `adj_prefix`, `clamp`, `positions`).
- Produces: `build_efficiency_table`'s final output gains `expected_<output>` for every output in `TARGET_COLUMNS` — this completes the function's real, public contract. Task 4 (the evaluation script) reads `expected_<output>` directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_efficiency.py — add to the existing file

def test_build_efficiency_table_adjusts_upward_against_a_friendlier_than_average_matchup() -> None:
    result = _build_table()

    # WR league average of def_adj_ypt_allowed_wr at any week in this
    # fixture is (2.0 + 0.0) / 2 = 1.0. wr1's own opponent value is 2.0
    # (above average, a friendlier matchup for the offense) -- the offset
    # is +1.0, so expected_yards_per_target must be exactly 1.0 higher
    # than the shrunk estimate alone.
    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    assert wr1_week2["expected_yards_per_target"] == pytest.approx(
        wr1_week2["_shrunk_yards_per_target"] + 1.0
    )


def test_build_efficiency_table_adjusts_downward_against_a_tougher_than_average_matchup() -> None:
    result = _build_table()

    # wr2's own opponent value is 0.0 -- 1.0 below the same 1.0 league
    # average -- the offset is -1.0.
    wr2_week2 = result.filter((pl.col("player_id") == "wr2") & (pl.col("week") == 2)).row(
        0, named=True
    )
    assert wr2_week2["expected_yards_per_target"] == pytest.approx(
        wr2_week2["_shrunk_yards_per_target"] - 1.0
    )


def test_build_efficiency_table_leaves_the_shrunk_rate_unchanged_against_an_average_matchup() -> None:
    result = _build_table()

    # rb1 is the only real RB in this fixture, so its own
    # def_adj_ypt_allowed_rb_rushing value (1.0) IS the league average --
    # the offset must be exactly 0.
    rb1_week2 = result.filter((pl.col("player_id") == "rb1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    assert rb1_week2["expected_yards_per_carry"] == pytest.approx(
        rb1_week2["_shrunk_yards_per_carry"]
    )


def test_build_efficiency_table_nulls_expected_output_for_an_ineligible_position() -> None:
    result = _build_table()

    # rb1 is not in PASS_CATCHERS_AND_RB's complement -- wait, RB IS in
    # PASS_CATCHERS_AND_RB, so use wr1 for the carry-side null check
    # instead: WR is not in RB_QB.
    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    assert wr1_week2["expected_yards_per_carry"] is None
    assert wr1_week2["expected_td_rate_per_carry"] is None


def test_build_efficiency_table_computes_expected_output_for_an_eligible_position() -> None:
    result = _build_table()

    rb1_week2 = result.filter((pl.col("player_id") == "rb1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # RB is in PASS_CATCHERS_AND_RB (receives) AND RB_QB (carries) --
    # both output families must be populated for a real RB row.
    assert rb1_week2["expected_yards_per_target"] is not None
    assert rb1_week2["expected_yards_per_carry"] is not None


def test_td_rate_clamps_to_one_instead_of_exceeding_it() -> None:
    table = pl.DataFrame(
        {
            "_shrunk_td_rate_per_target": [0.95],
            "_adj_td_rate_per_target": [10.0],
            "_league_avg_adj_td_rate_per_target": [0.0],
        }
    )

    result = efficiency._combine_and_clamp(
        table,
        shrunk_col="_shrunk_td_rate_per_target",
        adj_col="_adj_td_rate_per_target",
        league_avg_col="_league_avg_adj_td_rate_per_target",
        clamp=True,
        out_col="expected_td_rate_per_target",
    )

    assert result["expected_td_rate_per_target"][0] == pytest.approx(1.0)


def test_td_rate_clamps_to_zero_instead_of_going_negative() -> None:
    table = pl.DataFrame(
        {
            "_shrunk_td_rate_per_target": [0.05],
            "_adj_td_rate_per_target": [-10.0],
            "_league_avg_adj_td_rate_per_target": [0.0],
        }
    )

    result = efficiency._combine_and_clamp(
        table,
        shrunk_col="_shrunk_td_rate_per_target",
        adj_col="_adj_td_rate_per_target",
        league_avg_col="_league_avg_adj_td_rate_per_target",
        clamp=True,
        out_col="expected_td_rate_per_target",
    )

    assert result["expected_td_rate_per_target"][0] == pytest.approx(0.0)


def test_yards_per_touch_is_never_clamped() -> None:
    table = pl.DataFrame(
        {
            "_shrunk_yards_per_target": [5.0],
            "_adj_yards_per_target": [10.0],
            "_league_avg_adj_yards_per_target": [0.0],
        }
    )

    result = efficiency._combine_and_clamp(
        table,
        shrunk_col="_shrunk_yards_per_target",
        adj_col="_adj_yards_per_target",
        league_avg_col="_league_avg_adj_yards_per_target",
        clamp=False,
        out_col="expected_yards_per_target",
    )

    assert result["expected_yards_per_target"][0] == pytest.approx(15.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_efficiency.py -k "adjust or clamp or ineligible or eligible_position" -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.efficiency' has no attribute '_combine_and_clamp'` (or a `KeyError`/`ColumnNotFoundError` for `expected_yards_per_target` not existing yet)

- [ ] **Step 3: Implement the opponent-adjustment step**

Add these two functions to `src/ffapp/models/efficiency.py`, after `_shrink`:

```python
def _select_by_position(group_column_by_position: dict[str, str], adj_prefix: str) -> pl.Expr:
    """Picks the real `<adj_prefix>_<group>` column for a row's own
    position -- e.g. a WR row reads `def_adj_ypt_allowed_wr`, an RB row
    reads `def_adj_ypt_allowed_rb_receiving`, both already real,
    already-mapped columns sitting in `player_week_features` (task 1.9).
    A position not present in `group_column_by_position` (e.g. a QB row
    for a target-side output) stays null -- correct, since that position
    has no real matchup value for this output at all."""
    expr = pl.lit(None, dtype=pl.Float64)
    for position, group in group_column_by_position.items():
        expr = pl.when(pl.col("position") == position).then(pl.col(f"{adj_prefix}_{group}")).otherwise(expr)
    return expr


def _combine_and_clamp(
    table: pl.DataFrame,
    *,
    shrunk_col: str,
    adj_col: str,
    league_avg_col: str,
    clamp: bool,
    out_col: str,
) -> pl.DataFrame:
    """Step 2 of the design doc's formula: `final = shrunk + (this_week's
    real opponent adjustment - the league-average adjustment for that
    same group and week)`. `adj_col`/`league_avg_col` are filled to 0.0
    before combining -- deliberately: a missing opponent-adjustment value
    (task 1.8's own real, rare, early-season null pattern) should degrade
    gracefully to "no matchup adjustment applied," not null out an
    otherwise-valid shrunk estimate entirely. `td_rate_per_*` outputs are
    clamped to `[0, 1]` -- a real, if rare, edge case where an additive
    offset against an outlier matchup could otherwise push a probability
    outside its valid range. `yards_per_*` outputs are never clamped --
    real yardage has no natural upper bound and is never negative to
    begin with.
    """
    combined = pl.col(shrunk_col) + (
        pl.col(adj_col).fill_null(0.0) - pl.col(league_avg_col).fill_null(0.0)
    )
    if clamp:
        combined = combined.clip(0.0, 1.0)
    return table.with_columns(combined.alias(out_col))
```

In `build_efficiency_table`, change the loop body one more time to compute the position-selected adjustment column, the same-week league-average, combine, clamp, and gate by position eligibility:

```python
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
        table = table.with_columns(
            _select_by_position(spec.group_column_by_position, spec.adj_prefix).alias(
                f"_adj_{output_name}"
            )
        )
        # League-average adjustment for this output's own group, same
        # week -- a same-week pooled mean across every eligible player's
        # own already-real adjustment value. No shift needed: the value
        # already reflects "as of before this week" (task 1.9's own
        # module docstring). Null rows (an ineligible position) are
        # skipped by .mean()'s own default null handling.
        table = table.with_columns(
            pl.col(f"_adj_{output_name}").mean().over(["season", "week"]).alias(
                f"_league_avg_adj_{output_name}"
            )
        )
        table = _combine_and_clamp(
            table,
            shrunk_col=f"_shrunk_{output_name}",
            adj_col=f"_adj_{output_name}",
            league_avg_col=f"_league_avg_adj_{output_name}",
            clamp=spec.clamp,
            out_col=f"_combined_{output_name}",
        )
        table = table.with_columns(
            pl.when(pl.col("position").is_in(spec.positions))
            .then(pl.col(f"_combined_{output_name}"))
            .otherwise(None)
            .alias(f"expected_{output_name}")
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_efficiency.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/efficiency.py tests/test_models_efficiency.py && uv run ruff format --check src/ffapp/models/efficiency.py && uv run mypy src/ffapp/models/efficiency.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/efficiency.py tests/test_models_efficiency.py
git commit -m "feat: add opponent-adjustment offset and position gating for model v2 Stage 3"
```

---

### Task 4: Real evaluation against 2021-2025 data

**Files:**
- Create: `notebooks/evaluate_efficiency_v2_stage3.py` (scratch, per CLAUDE.md's `notebooks/` convention)
- Modify: `docs/JOURNAL.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes everything built in Tasks 1-3 (`efficiency.build_efficiency_table`, `efficiency.TARGET_COLUMNS`), plus already-existing real pipeline pieces (`evaluation.backtest.run_walk_forward_backtest`, `evaluation.backtest.BaselinePredictor`, `evaluation.metrics.accuracy_metrics`, `features.usage.PASS_CATCHERS_AND_RB`/`RB_QB`).
- Produces: no new library code — a real, documented result.

**Unlike Stage 2, this script does not need to re-run any other stage's model fitting** — Stage 3 doesn't consume Stage 1's or Stage 2's own predictions, and there is no trained model anywhere in this stage. It should run noticeably faster than Stage 1/2's own evaluation scripts (no LightGBM fitting at all), but is still a real walk-forward loop over five real seasons across four separate outputs — let it finish rather than assuming a long run means something is broken.

- [ ] **Step 1: Write the verification script**

```python
# notebooks/evaluate_efficiency_v2_stage3.py
"""One-off script: real evaluation of model v2 Stage 3 (efficiency
priors) against 2021-2025 data. Scratch, per CLAUDE.md's notebooks/
convention -- not imported by anything under src/.

Unlike Stage 2, this stage doesn't depend on Stage 1's or Stage 2's own
predictions -- SPEC's own Stage 3 input list names only "player
efficiency history" and "opponent adjusted rates," both already real,
already-available columns (models.efficiency.build_efficiency_table's
own job) -- so this script does not re-run any other stage's model
first.

Routes every predictor (the shrunk model AND both baselines) through
evaluation.backtest.run_walk_forward_backtest +
evaluation.backtest.BaselinePredictor from the start -- the exact
discipline Stage 2's own final review had to retrofit after a real
row-set-mismatch bug, applied proactively here. Filters to rows with a
real, defined per-touch outcome (>=1 real touch of that type that week)
BEFORE calling the harness -- the harness itself does not drop a null
target_column, and accuracy_metrics only filters on a null prediction,
not a null real outcome."""

import polars as pl

from ffapp.config import load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB
from ffapp.models import efficiency

settings = load_settings()

# --- Load real data. ---
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
player_week_features = pl.read_parquet(
    settings.data_root / "features" / "player_week_features.parquet"
)
player_week_usage = pl.read_parquet(settings.data_root / "interim" / "player_week_usage.parquet")
player_week_stats = pl.read_parquet(settings.data_root / "interim" / "player_week_stats.parquet")

# Real regular season only, upstream of everything else -- same
# convention Stage 1/2's own (corrected) evaluation scripts already
# established.
schedule = schedule.filter(pl.col("season_type") == "REG")
_reg_weeks = schedule.select("season", "week").unique()
player_week_features = player_week_features.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_usage = player_week_usage.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_stats = player_week_stats.join(_reg_weeks, on=["season", "week"], how="inner")

# --- Build Stage 3's own table. ---
table = efficiency.build_efficiency_table(player_week_features, player_week_usage, player_week_stats)

# run_walk_forward_backtest needs availability_flag (task 1.9's own
# column, not produced by build_efficiency_table) -- joined in here,
# matching the same separation Stage 1/2 both keep between their own
# table-building functions and their evaluation scripts.
table = table.join(
    player_week_features.select("player_id", "season", "week", "availability_flag"),
    on=["player_id", "season", "week"],
    how="left",
)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

# --- Score each output through the real walk-forward harness, once per
# target, with all three predictors (shrunk model, trailing_raw,
# league_mean) reading from the SAME position-eligible, real-outcome-
# defined row set -- so every predictor for a given target is scored on
# the identical row set from the start. ---
for target_column, eligible_positions in [
    ("yards_per_target", PASS_CATCHERS_AND_RB),
    ("td_rate_per_target", PASS_CATCHERS_AND_RB),
    ("yards_per_carry", RB_QB),
    ("td_rate_per_carry", RB_QB),
]:
    scoped = (
        table.filter(pl.col("position").is_in(eligible_positions))
        .filter(pl.col(f"real_{target_column}").is_not_null())
        .select(
            "player_id",
            "season",
            "week",
            "position",
            "team",
            "availability_flag",
            pl.col(f"real_{target_column}").alias(target_column),
            pl.col(f"expected_{target_column}").alias("_shrunk_model"),
            pl.col(f"trailing_raw_{target_column}").alias("_trailing_raw"),
            pl.col(f"league_mean_{target_column}").alias("_league_mean"),
        )
    )
    predictors = [
        BaselinePredictor(name="shrunk_model", column="_shrunk_model"),
        BaselinePredictor(name="trailing_raw", column="_trailing_raw"),
        BaselinePredictor(name="league_mean", column="_league_mean"),
    ]
    predictions = run_walk_forward_backtest(
        scoped,
        schedule,
        predictors,
        validation_seasons=validation_seasons,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        target_column=target_column,
    )
    print(f"\n=== {target_column} ===")
    for result in accuracy_metrics(predictions):
        if result.scope == "all" and result.position is None:
            print(
                f"{result.predictor}: {result.metric}={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )
```

- [ ] **Step 2: Run it against real cached data**

Run: `uv run python notebooks/evaluate_efficiency_v2_stage3.py`

If it fails because a specific input path doesn't exist locally yet, rebuild it first following `HANDOFF.md` §6's documented sequence — do not guess at a substitute path. This script needs the same real inputs Stage 2's own evaluation already used successfully (`player_week_features.parquet`, `player_week_usage.parquet`, `schedule.parquet`), plus `data/interim/player_week_stats.parquet` (already required by task 1.1/0.5's golden test, should already exist).

- [ ] **Step 3: Record the real result**

Read the printed MAE/n_obs/CI values for all four outputs (`yards_per_target`, `td_rate_per_target`, `yards_per_carry`, `td_rate_per_carry`). Add an entry to `docs/JOURNAL.md` §2 (Completed tasks), following this project's existing entry style (see the Stage 1/2 entries for the exact voice/format — specific real numbers, not vague, and the honest-reporting discipline those entries already establish), stating:
- The real MAE/n_obs/CI for `shrunk_model` vs `trailing_raw` (the real bar) vs `league_mean` (sanity floor, not the bar), for all four outputs.
- Whether the shrunk model beat `trailing_raw` on each of the four outputs, stated plainly.
- Reference the design doc's own noise caveat (a single week's real per-touch rate is extremely noisy at low touch counts) when interpreting the absolute MAE values — the relative comparison between predictors is what matters most here.
- If the shrunk model did not beat the baseline on one or more outputs, state that plainly as a real, documented result — same treatment every prior real result in this project has gotten. Do not tune anything to force a passing number; there is nothing to tune here (no hyperparameters, no model) — if the result is unfavorable, the honest next step is a conversation with the project owner about Stage 3's own approach or about Stage 4, not a code change made unilaterally in this task.

Update `HANDOFF.md`'s "Next task" line and current-session narrative to reflect this real outcome, matching the existing style of the Stage 1/2 entries directly above it. Update `TASKS.md`'s 3.1 entry with Stage 3's own real result, following the same pattern already used for Stage 1/2.

- [ ] **Step 4: Commit**

```bash
git add notebooks/evaluate_efficiency_v2_stage3.py docs/JOURNAL.md HANDOFF.md TASKS.md
git commit -m "test: real evaluation of model v2 Stage 3 against 2021-2025 data"
```

---

## After this plan

Do not proceed to Stage 4 (Monte Carlo recombination) automatically. Report the real result from Task 4 to the project owner and let them decide next steps — same pattern this project has used at every prior real decision point (Stage 1's own "after this plan" precedent, Stage 2's own, and the project owner's own explicit choice this session to continue the pipeline despite Stage 1/2's mixed real results).
