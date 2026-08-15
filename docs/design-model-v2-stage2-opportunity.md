# Design: decomposed model v2, Stage 2 — opportunity

**Status:** proposed, approved in brainstorming 2026-08-14, not yet implemented.
**Relates to:** SPEC.md §11.4 ("v2: decomposed pipeline"), TASKS.md 3.1,
`docs/design-model-v2-stage1-team-environment.md` (Stage 1, complete, merged
to `main`).

## Why, and what this is not

Stage 1 predicts a team's own weekly play volume and pass rate. On its own
that says nothing about any individual player — Stage 2 is the first stage
that produces a per-player number. **It still isn't player value or a draft
ranking.** Its output is *expected targets/carries/red-zone touches*, a real,
checkable opportunity number (does McCaffrey's own model show a high carry
share?) — not fantasy points. Stage 3 (efficiency: yards per touch, TD rate)
and Stage 4 (Monte Carlo recombination into a points distribution) are still
required after this before anything resembling "who's worth drafting" exists.
Each gets its own design/plan/build cycle, not designed together with this
one — Stage 1 already showed real bugs hide in the details of one stage at a
time, and trying to design three stages at once is exactly how those slip
through unreviewed.

## Approach, confirmed in brainstorming

**No trained model.** Arithmetic composition only: a player's own trailing
share of team volume, multiplied by Stage 1's own predicted team volume.
Directly informed by Stage 1's own result — a trained LightGBM model added
real complexity (and a real constraint-sign bug) without beating a naive
baseline for team-level volume; there's no reason to expect a second trained
model here would do better, and the arithmetic version is simpler to reason
about and impossible to get a monotonic-constraint sign wrong on, because
there are no constraints.

**No separate vacated-share mechanism**, despite SPEC §11.4 naming "vacated
shares" as a Stage 2 input. A player's own trailing share
(`target_share_ewm_3`, etc., `features/usage.py`, already built, already
`lag_weeks=1`-safe) already organically absorbs a teammate's vacated share
over the following real weeks of games. The *immediate* "a starter was just
ruled out this week" case — where trailing history hasn't caught up yet — is
already handled by the existing news pipeline (task 2.10,
`tools/news_propagation.py`), which re-runs projections with a synthetic
vacated-share signal. Building a second redistribution mechanism inside
Stage 2 would duplicate that, not extend it.

## The formula

One per output, each a plain multiplication:

```
expected_targets    = target_share_ewm_3    × predicted_pass_attempts
expected_carries     = carry_share_ewm_3     × predicted_rush_attempts
expected_rz_touches  = rz_touch_share_ewm_6  × predicted_team_plays
```

`rz_touch_share_ewm_6` isn't split by pass/rush in `features/usage.py`'s own
definition ("(rz targets + rz carries) / team rz touches"), so its
denominator here is total predicted team plays, not a pass- or rush-specific
subset. Flagged as a real judgment call, not a certainty — revisit if the
real evaluation numbers look wrong for this one specifically.

Position eligibility is inherited directly from `features/usage.py`'s own
existing `_WindowedFeature.positions` scoping for each share
(`target_share`→pass-catchers+RB, `carry_share`→RB+QB,
`rz_touch_share`→pass-catchers+RB) — not re-derived here.

## The one real correctness requirement: predictions, not ground truth

`predicted_pass_attempts`/`predicted_rush_attempts`/`predicted_team_plays`
must come from **Stage 1's own out-of-sample walk-forward predictions**, not
the real actual team_plays/pass_rate for that week. Using ground truth would
hide Stage 1's real prediction error and make Stage 2 look better than it
would actually perform live — the entire point of walk-forward evaluation
(CLAUDE.md rule 1) is that a component never sees information it wouldn't
have at serving time, and that applies to *consuming* an upstream stage's
output just as much as to consuming raw features.

Concretely: re-run `evaluation.backtest.run_walk_forward_backtest` with
`TeamEnvironmentPredictor` (Stage 1, already built) exactly as
`notebooks/evaluate_team_environment_v2_stage1.py` already does, but this
time keep the **full returned predictions DataFrame** (one row per
`(team, season, week)` with `prediction`), not just the printed accuracy
summary. Two separate predictor runs (one for `team_plays`, one for
`pass_rate`), joined back together on `(team, season, week)`, then
`team_environment.derive_attempts(predicted_team_plays, predicted_pass_rate)`
— the exact same pure function Stage 1 already built, applied to Stage 1's
own predictions instead of real actuals, which is a legitimate reuse of the
same "never predict pass_attempts/rush_attempts directly" rule.

## Evaluation

**Real target:** `player_week_usage.parquet`'s own real `targets`/`carries`
columns, and `rz_targets + rz_carries` for red-zone touches (already
real play-by-play counts, `interim/build.py`'s existing derivation, no new
data needed).

**Real baseline (the bar to beat):** a player's own trailing *raw* count —
`targets`/`carries`/`rz_targets+rz_carries` each windowed with the same
`ewm_4` shape every other B2-equivalent baseline in this project uses
(`.ewm_mean(span=4).shift(1)`), ignoring team volume changes entirely. Same
question as always: does composing with team-level volume actually add
anything over "assume similar volume to recently"?

**Real sanity floor:** a positional pooled mean (B0-equivalent,
`models.baselines.pooled_rolling_mean`, already promoted and reusable
directly), for the same reason Stage 1 kept one — not the pass/fail bar, but
worth seeing.

**Scope:** same validation window as Stage 1 (`validation_seasons=[2021..2025]`,
`train_start=2015`, real REG-season data only) for direct comparability.

**Honest reporting, learned from Stage 1's own final review:** print
`n_obs`/CI alongside every MAE from the start (not bolted on after a review
catches its absence), and scope to `season_type == "REG"` from the start
(not bolted on after a review catches postseason leakage) — both real
findings from Stage 1's final review, applied proactively here rather than
waiting to be caught again.

## File structure

New module `src/ffapp/models/opportunity.py`, mirroring
`models/team_environment.py`'s shape minus the fit/predict machinery (no
model to fit):

- `build_opportunity_table(usage_features, stage1_predictions) -> pl.DataFrame`
  — joins a player-week usage table to Stage 1's per-(team,week) predictions
  by team, applies the three formulas, carries the real target/carry/rz-touch
  counts through for evaluation.
- `add_opportunity_baselines(table) -> pl.DataFrame` — the trailing-raw-count
  B2-equivalent and pooled-mean B0-equivalent baselines, mirroring
  `add_team_environment_baselines`'s exact shape.

New scratch script `notebooks/evaluate_opportunity_v2_stage2.py`, mirroring
`evaluate_team_environment_v2_stage1.py`'s shape: loads real data, re-runs
Stage 1's backtest to capture predictions (not just accuracy), builds the
opportunity table, evaluates all three outputs against both baselines, prints
real MAE/n_obs/CI, and the result gets recorded honestly in
`docs/JOURNAL.md`/`HANDOFF.md` regardless of what it shows — same standing
rule as every other model task in this project.

## Testing plan

TDD as usual:

1. `build_opportunity_table`'s three formulas — fixture tests confirming
   each output is genuinely `share × predicted_volume`, using small
   hand-built player-week + team-week-prediction fixtures.
2. Position scoping — a fixture confirming a QB row's `expected_targets` is
   null (QBs aren't in `target_share`'s eligible position group) while their
   `expected_carries` is populated (QBs are in `carry_share`'s), and the
   mirror case for a WR (targets populated, carries null).
3. `add_opportunity_baselines` — fixture tests for the trailing-raw-count B2
   (never leaks the target week, same `.shift(1)` discipline caught missing
   once already in Stage 1) and the pooled B0-equivalent.
4. A real walk-forward-adjacent verification run against real 2021-2025
   REG-season data (Task 6-equivalent), reported honestly.

## Explicitly out of scope for this design

- Stage 3 (efficiency priors) and Stage 4 (Monte Carlo recombination) — each
  their own design/plan/build cycle, started only after this one is real,
  reviewed, and verified.
- Any new vacated-share redistribution mechanism (see above — the existing
  news-pipeline mechanism already covers the real use case).
- Wiring Stage 2's output into anything player-facing (the draft board, the
  weekly rankings page, or any other shipped consumer) — stays a standalone,
  independently-evaluated piece, same precedent as Stage 1.
