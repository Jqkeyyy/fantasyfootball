# Design: decomposed model v2, Stage 3 — efficiency priors

**Status:** implemented, evaluated, and merged to `main` (2026-08-15). **Step 2
(opponent-adjustment offset, below) was removed the same day** on real
evidence from a paired-bootstrap ablation against real 2021-2025 data — see
the note directly after Step 2's own description, and `docs/JOURNAL.md`'s
Stage 3 entries for the full numbers. The description of Step 2 is kept
below, not deleted, since it documents what was tried and why, and remains
the starting point for any future rework.
**Relates to:** SPEC.md §11.4 ("v2: decomposed pipeline"), TASKS.md 3.1,
`docs/design-model-v2-stage1-team-environment.md` (Stage 1, complete, merged
to `main`), `docs/design-model-v2-stage2-opportunity.md` (Stage 2, complete,
merged to `main`).

## Why, and what this is not

Stage 2 predicts *how often* a player touches the ball (expected targets,
carries, red-zone touches). It says nothing about how well they do with
those touches once they get them. Stage 3 predicts that: expected yards and
touchdown rate *per touch*, conditional on a touch happening. It still
isn't fantasy points — Stage 4 (Monte Carlo recombination) is what actually
multiplies opportunity by efficiency and applies league scoring to produce
a points distribution. Each stage gets its own design/build/evaluate cycle,
same discipline Stage 1/2 already established.

**Real context this design doesn't hide:** Stage 1 beat its own trailing
bar but lost to a plain league-wide mean; Stage 2 lost to a player's own
trailing raw count on all three outputs, even after fixing a real bug and
trying a blend. The project owner weighed that evidence directly and chose
to continue building the full 4-stage pipeline anyway (see `HANDOFF.md`'s
2026-08-15 entries) — this design proceeds on that instruction, not on an
assumption that Stage 3 will necessarily fare better.

**SPEC §11.4 is unusually prescriptive about Stage 3**, more so than it was
for Stage 1/2:

```
Stage 3  Efficiency priors
         inputs : player efficiency history, opponent adjusted rates
         outputs: yards per target, yards per carry, TD probability per touch
         RULE   : shrink hard toward positional means. Empirical Bayes with a
                  prior weight equivalent to ~50 targets or ~80 carries.
```

Two things this settles before any brainstorming was needed: (1) no trained
model — an explicit Empirical Bayes shrinkage estimator, continuing Stage
2's own precedent that a trained model has no particular reason to beat a
well-constructed arithmetic/statistical baseline at this stage either; (2)
task 1.8's opponent-adjustment work already produces exactly the second
named input, and — checked directly in `features/opponent.py` while
writing this design, not assumed — it's not just sitting in
`interim/defense_position_allowed.parquet` waiting to be joined: task 1.9's
`features.opponent.add_opponent_features` already maps it onto individual
player rows by the player's own position (`def_adj_ypt_allowed_wr`,
`def_adj_ypt_allowed_rb_rushing`, `def_adj_td_rate_allowed_qb_rushing`,
...), already lag-safe (that module's own docstring: a
`defense_position_allowed` row already reflects "as of right before this
week," so no further shift is needed), and this is already sitting in
`features/player_week_features.parquet` — the same table Stage 2 already
reads. No new join, no new table — Stage 3 reads columns that already
exist.

**Stage 3 does not depend on Stage 2's output.** SPEC's own input list for
Stage 3 names only "player efficiency history" and "opponent adjusted
rates" — Stage 2's expected touches never appear. The two stages only
combine at Stage 4 (opportunity × efficiency). This means Stage 3 can be
designed, built, and evaluated entirely independently of Stage 2's own
known weakness.

## Approach, confirmed in brainstorming

**No trained model**, per SPEC's own RULE line. Four outputs, computed
independently per player-week, **split by touch type** (confirmed in
brainstorming — real rushing vs. receiving TD rates differ meaningfully by
role, e.g. a goal-line back vs. a slot receiver, so a single combined
touchdown-per-touch rate would blur a real distinction the split
naturally preserves):

- `yards_per_target`, `td_rate_per_target` — eligible positions
  `features.usage.PASS_CATCHERS_AND_RB` (same list Stage 2's
  `expected_targets` already uses).
- `yards_per_carry`, `td_rate_per_carry` — eligible positions
  `features.usage.RB_QB` (same list Stage 2's `expected_carries` already
  uses).

## The formula

Two steps per output, applied in this order (confirmed in brainstorming —
see the ordering rationale below):

**Step 1 — Empirical Bayes shrinkage** (SPEC's literal RULE). A player's
own trailing per-touch rate, weighted by their own real trailing touch
count, blended with the positional league mean, weighted by SPEC's own
prior weight (~50 for target-based rates, ~80 for carry-based rates):

```
shrunk_rate = (n_touches × player_trailing_rate + prior_weight × positional_mean)
              / (n_touches + prior_weight)
```

- `player_trailing_rate` is a ratio of two real **cumulative sums**,
  season-to-date through week W-1, never week W's own outcome:
  `cum_sum(yards).shift(1).over([player_id, season]) /
  cum_sum(touches).shift(1).over([player_id, season])` — the identical
  `cum_sum().shift(1).over([player_id, season])` shape
  `models.baselines.add_b1_season_to_date_mean` already uses, just kept
  as a real sum instead of divided by week-count. **Not** a mean of
  weekly ratios (a 1-target 40-yard week must not carry equal weight to
  a 10-target 40-yard week) and **not** an EWM — refined during plan-
  writing from this design's own first draft (which specified an EWM
  ratio): an EWM of touches gives a *per-game rate* (e.g. "8 targets/game"),
  not a real cumulative count on the same scale as SPEC's own "~50 targets"
  prior weight, so it can't be compared against `prior_weight` meaningfully
  without an invented rescaling factor. A real cumulative sum needs none.
  Resets each season, no prior-season carryover — see below for why that's
  the right call here, unlike `positional_mean`.
- `n_touches` is that same cumulative sum's own denominator —
  `cum_sum(touches).shift(1).over([player_id, season])` — a real touch
  count directly comparable to SPEC's `~50`/`~80` prior weight, no
  rescaling needed.
- A player with 0 trailing touches of that type this season — every
  player's own week 1, and any player who simply hasn't touched the ball
  yet — gets exactly `positional_mean`, the shrinkage formula's own
  degenerate case, not a special-cased fallback. **Deliberately no
  prior-season carryover for `player_trailing_rate`/`n_touches`**, unlike
  `positional_mean` below: a whole position going null in week 1 would be
  a real problem (breaks the shrinkage formula for every player at that
  position), but one player's own history resetting to "shrink hard
  toward the positional mean until this season's own sample rebuilds" is
  exactly the intended behavior, not a gap to patch — and it's exactly
  the degenerate case already in this design's own testing plan.
- `positional_mean` is **not** a single call to `pooled_rolling_mean` on
  a precomputed per-row rate column — that would suffer the identical
  mean-of-ratios problem `player_trailing_rate` above avoids, just pooled
  across players instead of across weeks. Instead:
  `pooled_rolling_mean(table, "position", numerator_col, ...)` and
  `pooled_rolling_mean(table, "position", denominator_col, ...)` — two
  separate calls to the existing function (already built, already tested,
  reused directly, not modified), one for the raw numerator (e.g. real
  weekly `receiving_yards`) and one for the raw denominator (real weekly
  `targets`) — then `positional_mean = pooled_numerator_mean /
  pooled_denominator_mean`. Both pooled calls share the same real
  `n` (players × weeks pooled) by construction, so dividing the two
  means is mathematically identical to dividing the two underlying pooled
  sums directly — a correct ratio-of-sums, not a mean-of-ratios, achieved
  with zero new pooling logic. `pooled_rolling_mean`'s own prior-season
  fallback (already built in) is the right behavior here, unlike at
  player grain — an entire position having no real current-season data
  yet (true only in each dataset's very first season) should fall back to
  last season's real position-wide rate, not go null for every player at
  that position simultaneously.

**Step 2 — opponent-adjustment offset**, applied *after* shrinkage, not
blended into the prior. This week's specific opponent's real
`def_adj_ypt_allowed_<group>`/`def_adj_td_rate_allowed_<group>` — already
present in `player_week_features.parquet` (task 1.9's
`features.opponent.add_opponent_features`), already selected for the
right group by position (`_wr`/`_te`/`_rb_receiving` feed the
target-side outputs; `_rb_rushing`/`_qb_rushing` feed the carry-side
outputs — the same `pl.when(position...)` position-gating shape Stage 2
already uses for its own three formulas, just picking a column name
instead of a share column) — added onto the shrunk rate from Step 1:

```
final_rate = shrunk_rate + (this_opponent_adj_rate - league_avg_adj_rate)
```

**Additive, not multiplicative — a real correctness fix caught in this
design's own self-review, not assumed from SPEC's prose.** Checked
`interim/build.py::_ridge_defense_coefficients` directly: `adj_ypt_allowed`/
`adj_td_rate_allowed` are Ridge regression *coefficients* (SPEC §10.4's
own `y = mu + offense_team + defense_team + home + eps`) — a defense's
real yards-per-touch/TD-rate deviation from the league-average defense
(the fitted intercept absorbs the population baseline separately), already
roughly zero-centered by construction, not an absolute rate on a
ratio-safe scale. Dividing by a value that's already close to zero would
be meaningless; adding the deviation directly is the correct operation
for a coefficient in the same real units as the shrunk rate itself.
`league_avg_adj_rate` (a same-week mean of `adj_*_allowed` across every
real eligible player row for that position group — not one vote per
defense; a defense facing more rostered players that week is weighted
proportionally more heavily in the average, by construction of a
row-level `.mean()`) is still subtracted explicitly
rather than assumed to be exactly 0 — task 1.8's own per-team shrinkage
blend (current-season vs. prior-season ridge estimates) means the real
stored values aren't guaranteed to sum to exactly zero every week, so
this is a real, cheap guard, not a no-op.

**`td_rate_per_target`/`td_rate_per_carry` are clamped to `[0, 1]` after
Step 2** — a real, if rare, edge case: an additive offset applied to an
already-low shrunk rate against a real outlier matchup could otherwise
push a probability output outside its valid range. `yards_per_target`/
`yards_per_carry` are not clamped (real yardage has no natural upper
bound and a real per-touch average is never negative to begin with, so
there's no equivalent invalid range to guard against).

**Why applied after shrinkage, not baked into the prior (confirmed in
brainstorming):** the opponent-adjustment signal changes every week (this
specific defense, this specific week); the shrinkage prior is a stable,
season-long population value. Mixing a weekly-changing signal into a
long-run prior is a category mismatch — keeping them as two separate
steps, each doing one clear job, avoids that.

**Step 2 removed, 2026-08-15 — real evidence, not a design preference.**
The final whole-branch review ran a paired-bootstrap ablation comparing
`expected_*` (shrinkage + this offset) against the shrinkage step alone,
on real 2021-2025 data, and found the offset measurably HURT
`yards_per_carry` (RB and QB, both CIs excluding zero) and
`yards_per_target` (RB and WR, both CIs excluding zero), and was
statistically indistinguishable from zero everywhere else, including TE
(target-side). A first pass of this ablation ran against a commit that
still had a real null-handling bug in this same combine step (a missing
`adj_col` value degraded to a spurious non-zero offset instead of exactly
0.0) — the project owner correctly asked whether the harm finding and the
bug could be the same thing before trusting it, and they were not: the
bug was fixed, the ablation was re-run on the fixed code with a
per-position breakdown, and the harm on RB/WR yards and QB carries held.
TE's own point estimate (+0.017 mean absolute-error increase) sat between
RB's (+0.019) and WR's (+0.014), both confirmed harmful, with a wider CI
that reflects TE's smaller sample, not a smaller real effect — keeping
the offset for TE alone would have been selecting on noise, so it was
removed for every position, not carved into a position-specific
exception. `models/efficiency.py`'s `expected_<output>` is now exactly
the shrunk estimate from Step 1, with no further combination step and no
clamping (a convex combination of two already-bounded values, provably
never needs clamping — see the module's own current docstring).

**This is a finding about the specific additive functional form used
here, not a finding that opponent adjustment is worthless.** QB carries
showed the single largest harm (+0.075, the biggest effect measured
anywhere in this ablation) at exactly the position SPEC §10.4 itself
predicts matchup should matter most for rushing — evidence against this
formula, not against the underlying effect. The prime suspect for a
future rework: this design applied the offset identically to QB and RB
rushing via the same additive term, when a QB's own rushing matchup
(spy usage, containment schemes) is plausibly a different signal than an
RB's, and task 1.8's own already-shrunk defensive estimate was applied
at full weight in Stage 3's own combine step, with no further
discounting specific to how much real signal a given position's
opponent-adjustment differential actually carries. A multiplicative
form, a QB-specific term, or an explicit further shrinkage of the offset
itself are all real options for later — not attempted here, since this
task's job was evaluating the design as specified, not redesigning it
unilaterally.

Position eligibility for each output is inherited directly from
`features.usage`'s own established lists (`PASS_CATCHERS_AND_RB`,
`RB_QB`), same precedent Stage 2 already set — not re-derived here.

## Evaluation

**Real target:** for each real player-week with ≥1 real touch of that
type, that week's own real per-touch rate (`actual_yards / actual_touches`,
`actual_tds / actual_touches`, from `player_week_stats.parquet`'s real
`receiving_yards`/`receiving_tds`/`rushing_yards`/`rushing_tds` and
`player_week_usage.parquet`'s real `targets`/`carries`). Undefined and
excluded — not zero — for a player-week with no touches of that type, same
"never fabricate a value where none exists" discipline as everywhere else
in this project.

**Real bar (`trailing_raw`):** a player's own simple trailing per-touch
rate with no shrinkage and no opponent adjustment — literally
`player_trailing_rate` from Step 1 above, used directly as its own
predictor. A clean ablation: does the shrinkage + opponent-adjustment
machinery actually add anything over "just use the player's own recent
rate"? Same question Stage 1/2 both already asked of their own real bars.

**Real sanity floor (`league_mean`):** the pooled positional mean,
`positional_mean` from Step 1 above, used directly. Not the pass/fail bar,
same role it plays in Stage 1/2.

**Scope:** same validation window as Stage 1/2
(`validation_seasons=[2021..2025]`, `train_start=2015`, real REG-season
data only, `season_type == "REG"` scoped from the start), for direct
comparability.

**A real, known limitation, stated here rather than found after the
fact:** a single week's real per-touch rate is extremely noisy at low
touch counts (one long completion on one target inflates that week's
`yards_per_target` to 40+; a red-zone target that falls incomplete
produces a real `td_rate_per_target` of 0 that says nothing about the
player's real ability). Every predictor here — the shrunk model and both
baselines alike — faces a large, mostly irreducible MAE floor from this
alone, unrelated to whether the underlying approach is sound. The
*relative* comparison between predictors stays meaningful; the *absolute*
MAE values should not be over-interpreted the way Stage 1/2's own (also
real, but comparatively less noisy) MAE values could be. Going in, this
stage is plausibly *more* likely to "lose" to `trailing_raw` on raw MAE
than Stage 1/2 were, for reasons that have nothing to do with whether
shrinkage + opponent-adjustment is the right idea — worth remembering
when reading the real result, not a reason to avoid running the real
evaluation.

## File structure

New module `src/ffapp/models/efficiency.py`, mirroring
`models/opportunity.py`'s shape (no fit/predict machinery, no model to
fit):

- `build_efficiency_table(player_week_features, player_week_usage,
  player_week_stats) -> pl.DataFrame` — no `defense_position_allowed`
  parameter needed. Joins the real per-touch outcome counts (from
  `player_week_usage`/`player_week_stats`), computes the trailing
  yards/touch cumulative sums per player and per position, and applies
  Step 1 (shrinkage only, Step 2 removed) to produce
  `expected_yards_per_target`, `expected_td_rate_per_target`,
  `expected_yards_per_carry`, `expected_td_rate_per_carry` plus the real
  outcome columns and both baseline columns (`*_trailing_raw`,
  `*_league_mean`) needed for evaluation. `player_week_features` no
  longer needs its `def_adj_*_<group>` columns selected at all, now that
  Step 2 is gone.

New scratch script `notebooks/evaluate_efficiency_v2_stage3.py`, mirroring
`evaluate_opportunity_v2_stage2.py`'s shape: loads real data, builds the
efficiency table, scores all three predictors (shrunk model, trailing_raw,
league_mean) per output through `evaluation.backtest
.run_walk_forward_backtest` + `evaluation.backtest.BaselinePredictor` (same
harness, same guarantee every predictor is scored on an identical row set —
the exact discipline Stage 2's own final review had to retrofit after
catching a real row-set-mismatch bug; applied proactively here from the
start), prints real MAE/n_obs/CI per predictor, and the result gets
recorded honestly in `docs/JOURNAL.md`/`HANDOFF.md` regardless of what it
shows — same standing rule as every other model task in this project.

## Testing plan

TDD as usual:

1. The shrinkage formula — fixture tests confirming the blend math at a
   few hand-computed touch counts (0 touches → exactly `positional_mean`;
   a touch count equal to the prior weight → exactly the midpoint between
   `player_trailing_rate` and `positional_mean`; a very large touch count
   → close to `player_trailing_rate`).
2. ~~The opponent-adjustment offset~~ — removed 2026-08-15 along with
   Step 2 itself; see the note above. `expected_<output>` is tested
   directly against `_shrunk_<output>` instead (exact equality, no
   further combination).
3. Position eligibility — a fixture confirming a QB row's
   `expected_yards_per_target`/`expected_td_rate_per_target` are null
   (QBs aren't in `PASS_CATCHERS_AND_RB`) while their carry-side outputs
   are populated, and the mirror case for a WR.
4. `player_trailing_rate`'s and `positional_mean`'s own ratio-of-sums
   construction — a fixture proving neither is a mean of weekly/per-player
   ratios (a 1-target 40-yard week, or a 1-target-volume player, must not
   carry equal weight to a 10-target 40-yard week or a 10-target-volume
   player).
5. ~~TD-rate clamping~~ — removed along with Step 2; no longer needed
   (see the module's own current docstring for why a convex combination
   of two already-bounded values can't leave `[0, 1]`).
6. A real walk-forward-adjacent verification run against real 2021-2025
   REG-season data, reported honestly per the Evaluation section above.

## Explicitly out of scope for this design

- Stage 4 (Monte Carlo recombination) — its own design/plan/build cycle,
  started only after this one is real, reviewed, and verified.
- Any change to Stage 1 or Stage 2 — both consumed as-is (Stage 3 doesn't
  even depend on Stage 2's output, see above).
- Any trained model — SPEC's own RULE line already settles this.
- Wiring Stage 3's output into anything player-facing (the draft board,
  the weekly rankings page, or any other shipped consumer) — stays a
  standalone, independently-evaluated piece, same precedent as Stage 1/2.
