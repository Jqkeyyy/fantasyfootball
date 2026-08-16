# SPEC Addendum 04 — rest-of-season weekly rankings

**Date:** 2026-08-13
**Status:** supersedes `SPEC.md` §11.3 and §14.2. Read after Addendum 03.

Goal: weekly-refreshing rest-of-season rankings, live for the 2026 season. Season opens in roughly four weeks, so this is scoped to *working by Week 1*, not finished.

The blocker is task 1.15: the conditional points model does not beat baseline B2 on Spearman-within-position-week. This addendum argues that result is probably not yet a true architectural verdict, gives cheap diagnostics to find out, and specifies the ranking pipeline either way.

---

## A. Re-open the 1.15 diagnosis before rebuilding anything

The hyperparameter search concluded the gap is architectural. That conclusion is premature — a 25-trial search over the same features, same target, and same loss can only tell you that *those* choices are not undertuned. It cannot distinguish an architecture problem from a feature problem, a target problem, or a measurement problem.

Run these five diagnostics before writing a line of v2. They are hours of work, not days, and any one of them could resolve 1.15 outright.

### A.1 Is Spearman being measured on the wrong population?

`SPEC.md` §12.4 requires accuracy metrics on **startable rows only** — players rostered and above a projection threshold. Confirm the Spearman figure honours that filter.

If it is computed across all player-weeks, it is dominated by third-stringers and inactive players scoring 0–3 points, where ordering is close to pure noise and B2's "he scored 2 last week too" is unbeatable. Ranking quality among the top 24 RBs is the entire product; ranking quality among RB60–RB120 is measurement noise that can bury a real win.

**Test:** recompute Spearman restricted to, per position-week, the top *N* by projection where N is roughly `n_teams × starters[pos] × 2`. Compare model vs B2 on that subset only.

This is the cheapest of the five and the most likely to change the answer.

### A.2 Does the model have its own strongest predictor as an input?

`SPEC.md` §10.2's feature catalogue lists `points_std` (volatility of own league-scored points) but **no rolling mean of own league-scored points**. If that is genuinely absent, the model is being asked to beat B2 without being given B2.

B2 is a *player-specific* estimator: a player's own trailing average silently encodes talent, role, offensive quality, and coaching tendency in one number. The GBM is a *population-level* estimator that must reconstruct all of that from features, and loses information doing so. That asymmetry alone can explain the whole result.

**The complication:** rolling own points is league-scored, and `ADDENDUM-01 §A.6` forbids a points column in the feature table. Resolve it the same way the target is resolved — compute it in the training pipeline, per league, at fit time:

```python
y          = score_stat_line(stats, league.scoring_settings)
ewm_points = ewm_by_player(y, span=4, shift=1)   # shift=1 preserves the as_of contract
```

It never touches `features/player_week_features.parquet`, so the seam holds.

**Test:** add `ewm_points_4`, `ewm_points_8`, and `points_last_week` as fit-time features. Re-run the harness. This is the single highest-leverage experiment in this addendum.

### A.3 Are the predictions compressed?

MAE improved while Spearman did not. That is the signature of mean reversion: an L1/L2 objective is minimised by shrinking toward the conditional mean, which lowers absolute error and simultaneously flattens the spread that ordering depends on.

**Test:** within each position-week, compare `std(predictions)` to `std(actuals)`. A ratio well below 1 (say under 0.6) confirms compression. If so, A.4 is the fix.

### A.4 Train the ranking objective directly

You are optimising L1 and measuring rank correlation. LightGBM can optimise the thing you actually care about.

**Test:** `objective="lambdarank"` (or `rank_xendcg`), with the group defined as `(position, week)` and the label a bucketed relevance grade of actual points. Evaluate with the same harness.

This is the textbook remedy for "MAE moved, rank did not," and it costs one training run.

### A.5 Is the failure uniform across positions?

An aggregate Spearman number can hide a model that wins clearly at QB and TE while losing at RB and WR.

**Test:** report Spearman per position, model vs B2, with the per-week bootstrap CIs `SPEC.md` §12.5 requires. If the model wins anywhere, ship it *there* and use B2 elsewhere. A per-position champion is a legitimate production system, not a compromise.

---

## B. Anchored residual architecture (new task 1.20)

If the diagnostics in §A do not clear the bar, do this **before** attempting the decomposed v2 in `SPEC.md` §11.4. It is a fraction of the work and it makes the floor safe.

Rather than predicting points, predict the **residual against B2**:

```
target  = actual_points − B2(player, week)
predict = B2(player, week) + model(features)
```

Properties that matter here:

- **The floor is the baseline.** A model that learns nothing outputs ~0 and you get B2 back. You cannot ship something materially worse than the baseline, which is exactly the risk you are currently carrying.
- **The model only learns what B2 misses** — matchup, injury-driven role change, game script, vacated share, weather. That is a far smaller and better-specified learning problem than "predict fantasy points from scratch."
- **It composes with everything already built.** The availability model (1.14), the quantile models (1.16), and the projection pipeline (1.18) are unchanged; only the conditional points stage swaps.

Add a blend weight fitted on validation:

```
final = w × (B2 + residual_model) + (1 − w) × B2
```

with `w ∈ [0, 1]` chosen per position on held-out seasons. If the model is useless for a position, `w` goes to 0 automatically and that position falls back to baseline. Report the fitted `w` per position in the evaluation report — it is a direct, honest readout of how much the model is actually contributing.

**Acceptance:** beats B2 on Spearman-within-position-week over startable rows, across at least four validation seasons, with bootstrap CIs excluding zero. Same bar as 1.15, no softening.

---

## C. The shipping rule

`SPEC.md` §12.3 already settles what to do if none of this clears the bar: **ship B3 (consensus), keep working in the background.**

Make that explicit in the pipeline rather than an informal fallback. `models/predict.py` gains a configured projection source:

```yaml
model:
  projection_source: "anchored"   # anchored | direct | baseline_b2 | consensus_b3
```

Every projection artefact already records `model_version`; extend it to record `projection_source` too. The Model Health page shows which source is live and what its current margin over B2 is.

There is no shame in shipping consensus. B3 is a blend of full-time analysts with paid charting data, and beating it is genuinely hard. What matters is that the system knows which source it is using and reports it honestly.

---

## D. Rest-of-season rankings pipeline (new task 1.21)

This is the actual deliverable and it is mostly independent of which projection source wins.

### D.1 Multi-week projection

Extend `ffapp project` to a horizon:

```
ffapp project --from-week N --through-week 18 --league <slug>
```

For each future week, features must be built from data available *now*, not from the future. Concretely: usage and team-context features are frozen at the current week's values and carried forward; opponent and schedule features vary by week because the schedule is known; injury status decays toward the availability model's baseline as the horizon extends.

**Do not** roll rolling-window features forward as if future weeks had already happened. That is leakage-by-construction and it will look fine.

Write to `outputs/<league_slug>/projections_ros.parquet`, one row per (player, week), carrying the same distributional columns as the weekly file.

### D.2 Aggregation to rest-of-season value

```
FOR each player:
    FOR each remaining week w:
        p_play[w]  = availability model × injury hazard survival (task 2.3)
        points[w]  = projection distribution for week w
    ros_points   = Σ_w  p_play[w] × E[points[w]]
    ros_dist     = Monte Carlo over weekly distributions with the
                   correlation structure from task 2.2
    expected_games = Σ_w p_play[w]
```

Bye weeks contribute zero. Weight fantasy-playoff weeks (`playoff_week_start`..17) as a separate reported column rather than folding them into the main total — the two answer different questions and merging them hides both.

### D.3 Rest-of-season VOR

Reuse the §9.4 fixed point, but with replacement level computed over **remaining** value and the **current** free-agent pool, not preseason. Replacement level drifts through a season as the waiver pool thins; a rest-of-season ranking that uses August's replacement level will systematically overvalue depth.

Rank by `vor_ros`. Never by raw projected points.

### D.4 Weekly refresh

Fold into the Tuesday run in `SPEC.md` §16.1:

```
Tue 07:30  ffapp project --from-week N+1 --through-week 18 --all-leagues
Tue 07:45  ffapp rankings ros --all-leagues
```

Each run writes to a timestamped directory and updates a `latest` pointer. Keep the history — comparing this week's ROS ranking to last week's is how you see the model actually reacting to news, and it is the cheapest sanity check you will ever run.

### D.5 UI

New page, `6_ROS_Rankings.py`: sortable by `vor_ros`, filterable by position and availability, showing `ros_points`, `expected_games`, `p10/p50/p90` season total, playoff-weeks value, and **rank change since last week**. That last column is the one you will actually look at.

---

## E. Weeks 1–3 cold start

A real problem worth planning for now: 2026 has no nflverse release yet, so at Week 1 every rolling usage feature has zero current-season data.

Rules:

- **Weeks 1–3:** default `projection_source` to `consensus_b3`. Public projections incorporate beat reporting, preseason usage, and camp news that your feature set has no access to. Do not pretend otherwise.
- **Weeks 4–6:** blend, with weight shifting toward the model as sample accumulates. `w = min(1, (week − 3) / 4)` is a reasonable start; tune it on historical early-season data.
- **Week 7+:** the model source proper.
- Prior-season features (`prior_season` windows in §10.2) carry more weight early. Verify they are actually populated for rookies and role-changers rather than silently null.

Encode this as a configured schedule, not a manual switch you have to remember on a Tuesday morning in September.

---

## F. Sequencing

Season opens in ~4 weeks. Draft is Aug 22 and outranks all of this.

*Revised 2026-08-15. Draft is Aug 22; season opens ~Sep 10.*

| When | Work |
|---|---|
| Now → Aug 22 | **Draft only.** Tasks 0.15/0.16 and the two dead ranking sources. Nothing in this addendum, and no further v2 stage work. |
| Aug 23–27 | §A diagnostics, all five. Cheap, and may resolve 1.15 outright. |
| Aug 28–Sep 3 | Task 1.20 (anchored residual) if §A did not clear the bar. |
| Sep 4–10 | Task 1.21 (ROS pipeline) against whichever source won. §E cold-start config. Task 1.22. |
| Week 1 | Ship on consensus per §E. Watch the Model Health page. |
| Weeks 2–6 | Let the blend shift. Compare weekly ROS deltas for sanity. |
| Offseason | Decomposed v2 (SPEC §11.4) as Phase 3, now with a season of your own logged predictions to evaluate against. |

**On the v2 work currently in flight.** Stages 1–3 of the decomposed pipeline are built and Stage 4 is scoped. That work is real and worth keeping, but it is Phase 3 by SPEC's own scoping, and the QB-passing gap discovered during Stage 4 planning confirms why: closing v2 properly needs two more full stages (per-QB attempt attribution, passing efficiency) plus recombination, each with its own evaluation gate. That is offseason-sized.

The decision point: finish Stage 4 non-QB and run the three-arm comparison — which answers cheaply whether the decomposition beats v1 at all — then stop v2 and move to §A. Do not start the QB passing sub-pipeline before the gating comparison says the decomposition is worth it.

The decomposed v2 stays a Phase 3 project. It is the right long-term architecture and the wrong thing to start four weeks before a season.

---

## G. Task list additions

- **1.20 (new, ⏱ 6h)** — anchored residual model with per-position blend weight (§B). Acceptance in §B.
- **1.21 (new, ⏱ 8h)** — ROS projection, aggregation, VOR, refresh, and UI (§D).
- **1.22 (new, ⏱ 2h)** — cold-start source schedule (§E).
- **1.15** — remains open until §A's diagnostics are run. Do not close it as "architectural" until A.1 and A.2 have been tested; the current evidence does not support that conclusion.
