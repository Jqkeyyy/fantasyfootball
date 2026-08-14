# Design: decomposed model v2, Stage 1 — team environment

**Status:** proposed, approved in brainstorming 2026-08-14, not yet implemented.
**Relates to:** SPEC.md §11.4 ("v2: decomposed pipeline"), TASKS.md 3.1.

## Why

Task 1.15's v1 (a single LightGBM regressor per position, SPEC §11.3) is built,
evaluated, and confirmed to lose to baseline B2 (a player's own trailing
`ewm_4`) — a real, already-investigated architectural ceiling (a 25-trial
hyperparameter search moved MAE past B2 but never Spearman-within-position-week;
see `docs/JOURNAL.md`'s 1.15 entry), not an undertuned model. SPEC §11.4's own
diagnosis: "opportunity is stable and efficiency is close to noise; modelling
their sum lets the noise swamp the signal." The decomposed pipeline exists to
separate those two signals instead of asking one model to learn both at once.

## Scope

**This design covers Stage 1 only** — "team environment," the first of
SPEC §11.4's four stages. Stages 2-4 (opportunity, efficiency priors, Monte
Carlo recombination) are each their own follow-up brainstorm/design/plan cycle,
started only once Stage 1 is built and has actually beaten its own baseline.
Nothing in this design wires into `models/points.py`, the draft board, or any
other already-shipped consumer — it is a standalone, independently-evaluated
piece until proven.

## Inputs

All but one already exist in `features/team_context.py`, real and
`as_of`-safe:

- `implied_team_total`, `spread` — current-week passthroughs (Vegas closing
  lines are known pre-kickoff, no lag needed; sourced from nflverse's real
  schedule `spread_line`/`home_implied_total`/`away_implied_total`, sign
  convention empirically verified against 3,028 real games in
  `interim/build.py::add_schedule_context`).
- `proe_ewm_5`, `neutral_pace_ewm_8` — trailing EWM features, `lag_weeks=1`.

**New:** `opponent_neutral_pace_ewm_8` — the *opponent's* own trailing pace,
joined onto a team's row. Doesn't exist yet. Built by mirroring
`features/opponent.py::add_opponent_features`'s existing join pattern (that
module already joins an opponent's defensive rates onto the offense's row the
same way) — same trick, applied to pace instead of defensive rates.

## Targets

No new target engineering needed — `interim/team_week_context.parquet`
already has real `plays` and `pass_rate` columns for every historical
team-week.

Two models, not three:

- `team_plays` (count) — one LightGBM regressor.
- `pass_rate` (0-1) — one LightGBM regressor.
- `pass_attempts = team_plays × pass_rate`, `rush_attempts = team_plays × (1 −
  pass_rate)` — **derived, never predicted directly**, so the two always sum
  to the predicted total exactly. (Rejected alternative: three independent
  regressors — simpler per-model, but nothing would guarantee
  `pass_attempts + rush_attempts == team_plays`, an inconsistency Stage 2
  would otherwise inherit.)

Monotonic constraints (same philosophy as v1, SPEC §11.3): `neutral_pace_ewm_8`
and `opponent_neutral_pace_ewm_8` increasing on `team_plays`; `proe_ewm_5`
increasing on `pass_rate`.

New module: `models/team_environment.py`, following the existing
`models/points.py` / `models/dst.py` / `models/availability.py` pattern
(dataclass-wrapped fitted model, `fit_*`/`predict_*` functions, a `Predictor`
implementation for the harness).

## Evaluation

**Reuses `evaluation/backtest.py::run_walk_forward_backtest` completely
unmodified** — no changes to that file, so nothing any of the four existing
predictors (points, dst, availability, quantiles) already depends on can
regress. Instead, `build_team_environment_table` reshapes team-week rows to
fit the harness's existing player-week-shaped contract, exactly the trick
`models/dst.py::build_dst_table` already uses for the same reason (its own
docstring: avoiding "a second, parallel walk-forward loop"):

- `player_id = team` (the team abbreviation)
- `position = "TEAM_ENV"` (new sentinel, doesn't collide with a real position)
- `availability_flag = True` (always — a team always "plays" its own game)

Called twice: once with `target_column="team_plays"`, once with
`target_column="pass_rate"`.

**Baselines**, following the existing `models/baselines.py` B0/B2 pattern but
at team grain instead of player grain:

- League-wide mean that week (sanity floor, B0-equivalent) — reported, but
  not the real bar.
- Team's own trailing `ewm_4` (B2-equivalent) — **the real bar.** Stage 1
  must beat this on MAE for `team_plays` and `pass_rate` (and therefore the
  derived `pass_attempts`/`rush_attempts`) before it's considered to work.
  Matches this project's standing rule (CLAUDE.md #6): beat the baseline
  before believing the model.

## Testing plan

TDD as usual (CLAUDE.md: tests alongside implementation, not after):

1. `opponent_neutral_pace_ewm_8` — fixture tests mirroring
   `features/opponent.py`'s own existing test style (a real join, correct
   opponent resolution, no leakage of the target week's own pace).
2. `build_team_environment_table` — fixture tests confirming the DST-style
   reshape (correct `player_id`/`position`/`availability_flag` values, real
   `team_plays`/`pass_rate` carried through unmodified).
3. `fit_team_plays_model` / `fit_pass_rate_model` (or one parameterized
   pair) — fixture tests for fit/predict shape, monotonic constraints set
   correctly.
4. A real walk-forward run against 2021-2025 data (same window the existing
   evaluation report uses) confirming Stage 1 beats the trailing-average
   baseline on real MAE — the same "verified against real data" bar every
   other model in this project has met before being called done. If it
   *doesn't* beat baseline, that's a real, documented result (same treatment
   as 1.15's own outcome) — not a bug to force a passing number out of.

## Explicitly out of scope for this design

- Stages 2-4 of the decomposed pipeline (separate design cycles).
- Wiring Stage 1's outputs into `models/points.py`, the draft board, or any
  other shipped consumer.
- Any change to `evaluation/backtest.py` itself.
- Kicker/DST-specific modelling (unrelated, already has its own model /
  is deliberately deprioritized per SPEC §11.7).
