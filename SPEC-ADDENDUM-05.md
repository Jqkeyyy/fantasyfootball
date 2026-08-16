# SPEC Addendum 05 — consensus as anchor, and the offseason model plan

**Date:** 2026-08-16
**Status:** extends `ADDENDUM-04`. Supersedes `SPEC.md` §11.3's framing of the model's job.

Tasks 1.15 and 1.20 established that a from-scratch points model does not beat consensus (B3), and that B3 beats B2 robustly. `projection_source: consensus_b3` now ships.

This addendum reframes what the model is for, and specifies the work that makes a better one possible. **Exactly one item here is urgent** (§B, in-season logging). Everything else is offseason.

---

## A. The reframe

The model has been treated as a competitor to consensus. That was the wrong framing, and the diagnostics in `ADDENDUM-04 §A` showed why: consensus encodes camp reports, beat reporting, coaching signals, and role information that no feature set built from nflverse play-by-play can reconstruct. Trying to out-predict it from scratch means re-learning, badly, what it already knows.

The standard move in forecasting is to treat the best available forecast as an **input**, and model only its residual error. That is what §C specifies.

The reframe also clarifies what this project's value actually is, and it was never "beat FantasyPros at predicting points." Consensus publishes generic-PPR point estimates. It does not know that this league scores 4-point passing touchdowns, starts two W/R/T flex slots, pays per field-goal yard, or that your bench is already three deep at WR. Every layer downstream of the projection — the scoring engine, the VOR fixed point, win-probability start/sit, roster-relative waiver value, lineup-aware trade simulation — is the product. B3 is one input to it.

---

## B. In-season prediction logging (task 3.7) — **urgent, before Week 1**

### B.1 Why this cannot wait

Everything in §C and §D requires a training set of *consensus errors*. That training set does not exist and cannot be created retroactively, because **weekly consensus projections are not re-fetchable**. Week 3's published projections are gone by Week 4. This is the same non-reproducibility that `ADDENDUM-02 §C.3` flags for ADP, but worse — ADP at least has archives; weekly projections generally do not.

nflverse data is immutable and re-downloadable forever. Consensus projections are perishable. If they are not captured as the season runs, the offseason work in this addendum is impossible and January starts from exactly where July did.

The existing 2021–2025 B3 historical archive (built for the distribution wrapper) covers season-level and whatever weekly data was recoverable. Live logging is what makes 2026 a first-class training season.

### B.2 What to log

Append-only, written every week by the Tuesday and Sunday pipeline runs. One row per (league, season, week, player, source).

| Column | Notes |
|---|---|
| `league_slug`, `season`, `week`, `player_id` | keys |
| `as_of_utc` | when the projection was made — the snapshot boundary |
| `run_label` | `tuesday` \| `thursday` \| `sunday` — the same player gets several rows per week and the differences between them are themselves signal |
| `projection_source` | which source produced the shipped number |
| `b3_mean` | consensus centre, league-rescored |
| `b3_q10 … b3_q90` | the shipped calibrated spread |
| `per_source_points` | one column per ranking source, league-rescored — **not just the aggregate** |
| `n_sources`, `dispersion` | consensus confidence |
| `model_mean` | the model's projection, logged even though it is not shipping |
| `b2_mean` | trailing-4 baseline, for a permanent live reference |
| `p_active` | availability model output |
| `actual_points` | backfilled after the week completes |
| `model_version`, `feature_hash`, `git_sha` | provenance |

Log `model_mean` even though the model is not live. A season of paired (model, B3, actual) is what makes the §C comparison possible, and it costs one extra column.

Log **per-source** projections, not just the trimmed mean. §E depends on it, and the individual sources are as perishable as the aggregate.

### B.3 Where it lives

`data/outputs/<league_slug>/prediction_log/season=2026/week=NN.parquet`, plus a `latest` pointer.

**This is committed to git**, as the rankings exception in `CLAUDE.md` already establishes for non-reproducible artefacts. It is small — a few thousand rows per week — and losing it would be unrecoverable in a way that losing a trained model is not.

### B.4 Backfill

`ffapp log backfill --week N` fills `actual_points` after games complete. Run it in the Tuesday job for the prior week. A missing backfill should warn loudly at the next run rather than leaving silent nulls that look like zeros.

---

## C. B3-anchored residual model (task 3.8) — offseason

Same structure as task 1.20, better anchor.

```
target  = actual_points − b3_mean
predict = b3_mean + residual_model(features)
final   = w × (b3_mean + residual) + (1 − w) × b3_mean
```

Properties: the floor is what already ships. A model that learns nothing outputs ~0 and returns B3. `ADDENDUM-04 §A.3`'s compression stops being a failure mode.

### C.1 Feature restriction — the important design choice

**Do not hand the residual model the full feature set.** Give it only features plausibly *absent* from consensus:

| Feature block | Rationale |
|---|---|
| `def_adj_*` opponent rates (all seven position groups) | Consensus adjusts for matchup weakly and without sample shrinkage |
| `wind_mph`, `is_dome`, `precip_prob`, `temp_f` | Rarely incorporated; wind above ~15mph is a real effect |
| `teammate_vacated_target_share`, `teammate_vacated_carry_share` | Consensus updates on a publishing cycle; this recomputes on news |
| `implied_team_total`, `spread` | Strongest team-level signal, inconsistently incorporated |
| `rest_days`, `is_primetime`, `week_number` | Situational, generally ignored |
| `dispersion`, `n_sources` | How uncertain consensus is about *this* player |

Explicitly **excluded**: season-long usage, snap share, target share, talent proxies, prior-season production. Consensus already encodes all of it. Including them invites the model to re-derive what B3 knows and add variance doing so — the same failure mode that sank Stage 2 and the opponent-adjustment offset in v2.

If the restricted model wins, the finding is clean: these specific signals are what consensus misses. That is a far more useful result than an unrestricted model winning by an unattributable margin.

### C.2 Acceptance

Must beat B3 — not B2 — on **both** startable-rows Spearman-within-position-week **and** lineup regret, under the gates in §F. If it beats one and not the other, it does not ship.

---

## D. Gated residual (task 3.9) — offseason, higher upside

A global residual has to be right on average. A gate only has to be right where it fires.

Consensus is systematically weakest in identifiable situations: the week after a teammate's injury, on role changes, on rookies, on players returning from injury, and in extreme weather. Beat reporting has not propagated and the projection is stale.

```
Stage 1 — uncertainty model
    target : |actual − b3_mean|
    output : expected magnitude of consensus error for this player-week

Stage 2 — gated correction
    where predicted error is in the top decile → apply the §C residual
    elsewhere                                  → ship B3 unchanged
```

`dispersion` across your five sources is the most promising single input to Stage 1 and it is free — when the sources disagree sharply, consensus is more likely wrong.

Report what fraction of player-weeks the gate fires on, and regret **restricted to gated rows**. A model that improves 8% of decisions meaningfully is more useful than one that improves 100% of them imperceptibly, and the aggregate metric will hide exactly that.

---

## E. Learned source weighting (task 3.10) — cheap, can run before the offseason

The aggregator uses a 20% trimmed mean across sources. Reasonable, and untested.

Source skill plausibly varies by position — a site with strong RB analysis may be mediocre at TE. With per-source logging from §B, fit non-negative weights per position on held-out seasons and compare against the trimmed mean on the same gates in §F.

Low risk, small scope, and it improves the thing that is actually shipping rather than a hypothetical replacement for it. This is the highest expected-value item in the addendum after §B.

Constrain weights to be non-negative and sum to one. An unconstrained fit will produce negative weights that look great in-sample and generalise terribly.

---

## F. Evaluation gates — encoding what task 1.20 taught

Task 1.20's regret-fit weight search found a dev-season result of 28.02 against a 39.02 baseline, and it reversed on held-out data. That is the most reusable lesson this project has produced, and these gates exist to make it unrepeatable.

Every model in this addendum must clear all six:

1. **Tuning data and evaluation data are disjoint.** Any hyperparameter, blend weight, or gate threshold is fit on dev seasons and evaluated on report seasons. Never both.
2. **Lineup regret is multi-seed.** At least 20 roster draws. Report mean, standard deviation, and win rate against the comparator. Regret has 2–3 point seed-to-seed noise; a single-seed result is not a result.
3. **Spearman is on startable rows only.** `ADDENDUM-04 §A.1`. This has been computed wrong twice — assert the filter in code, do not rely on a label.
4. **Bootstrap CIs resampled by week**, not by row. Rows within a week are correlated.
5. **Common-support comparison.** Compare only on player-weeks where every comparator produces a projection.
6. **The comparator is B3**, not B2. B2 is no longer the bar.

A result that clears five of six is a negative result.

---

## G. The charting data decision — January

Route participation is unavailable in-season (`SPEC.md` §10.5) and is among the best WR/TE predictors. PFF and Fantasy Points Data sell it.

The July version of this decision was a guess. The January version has evidence: the free-data model demonstrably loses to consensus. The question becomes concrete — would route data plausibly close a gap that opponent adjustment, weather, and vacated share could not?

Decide it after §C and §D report. If a restricted residual model wins on the free signals, buying data is a plausible next increment. If it does not, paid data is unlikely to rescue an approach that is losing for other reasons.

---

## H. Sequencing

| When | Work |
|---|---|
| **Before Week 1** | §B logging only. Nothing else. |
| Weeks 1–18 | Log every week. Verify backfill runs. Do not model. |
| Nov–Dec | §E source weighting, if the itch is unbearable — it is low risk and improves what ships. |
| January | §C restricted residual, then §D gate. §F gates on both. |
| February | §G data decision. v2 Stage 4 rewrite (`ADDENDUM-04`) if §C/§D justify continued investment. |

**Do not model during the season.** The system works, it is calibrated, and it is shipping consensus with validated spreads. In-season effort belongs on `ADDENDUM-04 §D`'s ROS pipeline and on actually using the tools.

---

## I. Task additions

- **3.7 (new, ⏱ 3h) — urgent, before Week 1.** Prediction logging per §B. Done when the Tuesday and Sunday pipeline runs write per-player, per-source rows with `as_of_utc`, backfill populates `actual_points` for a completed week, the log is committed to git, and a missing backfill warns loudly.
- **3.8 (new, ⏱ 6h)** — B3-anchored restricted residual per §C. Acceptance in §C.2 under §F's gates.
- **3.9 (new, ⏱ 8h)** — gated residual per §D. Report gate fire rate and gated-rows regret separately.
- **3.10 (new, ⏱ 4h)** — learned per-position source weighting per §E, non-negative and normalised.
