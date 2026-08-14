# HANDOFF.md — current state

**Last updated:** 2026-08-13
**Last machine:** `Maybe` (Windows). Sleeper reachable here — confirmed live, `api.sleeper.app` returns 200.
**Last commit:** `ee8dbaf` — **NOT pushed** (`origin/main` is at `ae7d5d7`; verified via `git status`, not assumed — the prior header's "`1beeb5e` (pushed)" claim turned out to be wrong when actually checked, see task 1.15/2.8/2.10/2.11's own push earlier this session). Working tree is otherwise clean, nothing uncommitted. Real commits since `a75554a` (the last one confirmed actually on `origin`): `5fb86b9` (1.15 hyperparameter-search follow-up), `cd16095` (2.8), `a87c39a` (2.10), `e461586` (2.11), `ae7d5d7` (tracking-doc updates), `ee8dbaf` (`docs/summary.md`, new this close-out). Push whenever you're ready — nothing here needs it done automatically.

This file is **state and pointers**, not design or history. `SPEC.md` and the addenda hold the design. Per-task evidence, implementation decisions, and gotchas live in `docs/JOURNAL.md` — read that when investigating a specific past task or bug, not every session. `docs/summary.md` is a third, newer reference: a standing narrative snapshot of current state and how things are going, meant to be read cover to cover rather than searched. Maintenance rules are at the bottom.

---

## 1. Where things stand

Current phase: **Phase 2 is fully complete.** Phase 0 (draft board, 0.1–0.14) has been done for a long time. Phase 1 has exactly one open item, 1.15 (see below); every other Phase 1 task is done. Nothing in Phase 3 has been started. The narrative below is kept in roughly chronological order across sessions, oldest first — read `docs/summary.md` instead if you want the current picture without the history.

Tasks 1.13 (Metrics module) and 1.17 (Evaluation report) are complete. Verified this session with a real 5-season walk-forward run: `uv run ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025` wrote `data/outputs/eval/20260813T183533Z/report.md` with populated metrics, real LightGBM feature importances, and real calibration curves. Both tasks' evidence is in `docs/JOURNAL.md`.

Phase 2 tasks 2.1–2.5, 2.7, and 2.9 were already complete — the simulation layer, start/sit assistant, DST model, and trade analyzer — built out of TASKS.md order on a prior work machine where Sleeper was network-blocked, confirmed with you first (see 2.1's entry in `docs/JOURNAL.md`).

Task 2.6 (Waiver wire) is complete — built this session, the first task on this machine (Sleeper reachable here) to actually need Sleeper's free-agent pool and transaction history. New `src/ffapp/tools/waivers.py` + `config.WaiverSettings`. Verified against the completed 2025 season (2026's real rosters are pre-draft leftovers, not usable — draft is Aug 22) and against real historical FAAB bidding data pulled from `bdff-chopped`'s actual transactions. Committed as `ea5cd70`, pushed.

Task 1.18 (Projection output pipeline) is complete — new `src/ffapp/models/predict.py` + `ffapp project --week N` CLI command, composing the availability/points/quantile models (tasks 1.14/1.15/1.16) into SPEC §6.2's `outputs/projections.parquet` schema. Verified for real against already-played weeks (2025 weeks 10/11 — 2026 has no nflverse release yet): 445/477 real projections, 0 nulls, full provenance, real upsert-by-`(season, week)` behavior confirmed with a second real run.

Task 1.19 (Streamlit weekly rankings page) is also complete — new `src/ffapp/app/weekly_rankings_page.py` + `src/ffapp/app/pages/2_Weekly_Rankings.py`, SPEC §15's second page. Verified live in a real Chrome session via `claude-in-chrome` (see §7 for the connection hiccup and how it resolved) against the real weeks 1.18 generated: position tabs, the floor/ceiling visible range, and both filters (position + real Sleeper-resolved availability) all confirmed working against real data, 0 console errors. Both 1.18 and 1.19 committed as `a75554a`, pushed. Full evidence for both tasks is in `docs/JOURNAL.md`.

Task 1.15's hyperparameter-search follow-up was attempted this session (a real 25-trial random search, `notebooks/tune_points_v1.py`/`.log`) and stopped partway through on your instruction ("stop here, deal with this later") — full result in `docs/JOURNAL.md`'s 1.15 entry and TASKS.md's own updated status line. Short version: tuning moves MAE past B2 (best config 5.005 vs B2's 5.107 on dev seasons 2018-2020) but **not one of 25 trials beat B2 on Spearman-within-position-week** — confirms the gap is architectural (SPEC §11.4's own diagnosis), not an undertuned model. The real confirmation run (`ffapp evaluate --seasons 2021 2022 2023 2024` with the tuned config wired into `points.py`) was deliberately not run — next step if resumed.

Task 2.8 (SOS and schedule grid) is complete — built later the same session, after 1.15's follow-up was paused. New `src/ffapp/tools/sos.py` (positional SOS, the schedule-grid pivot, matchup detail) + `src/ffapp/app/schedule_grid_page.py` (roster-highlight resolution, heatmap styling) + `src/ffapp/app/pages/3_Schedule_Grid.py`, SPEC §15's third page. A real design bug was caught live in a running Streamlit session (not a fixture) and fixed before shipping: the first confidence-threshold design reused SPEC §10.4's `k=250` shrinkage constant directly, which greyed out every real cell at every position — fixed with a per-position-group threshold computed from that group's own real data. Full account, including a second smaller Streamlit-rendering gotcha, in `docs/JOURNAL.md`'s 2.8 entry. Verified live end to end in a real Chrome session against `rogan-radinator-league`/2025 data, 0 console errors.

Task 2.10 (News pipeline) is complete — built later the same session, the biggest single Phase 2 task by SPEC's own hour estimate. New `src/ffapp/ingest/news.py` (real RSS ingestion — ESPN/CBS Sports/Yahoo NFL feeds, all three confirmed live before writing any code — plus the real Anthropic API structuring call, `claude-opus-5`, structured outputs, confidence/name-resolution routing to a real manual review queue) and `src/ffapp/tools/news_propagation.py` (the real second-order cascade SPEC calls "the valuable part": reuses task 1.7's own already-validated `add_vacated_shares` fed a synthetic early `Out` event, re-runs `models.predict.project_week` unmodified, surfaces the real handcuff in exactly the shape `tools.waivers.build_waiver_board` already expects). Two real bugs were caught only by running the cascade against real 2015-2025 data (a schema-mismatch concat and an `Int32`/`Int64` dtype mismatch, both invisible to same-shaped fixtures) and fixed with dedicated regression tests. Verified against a real historical case (Bucky Irving ruled out, Tampa Bay, real 2025 week 5): the recomputed vacated shares closely matched the real official values the existing pipeline separately produced, the real handcuff (Rachaad White) was correctly identified, and his recomputed projection moved substantially in the right direction, then flowed through the real waiver-board function end to end. **Not verified live against the real Anthropic API** — `ANTHROPIC_API_KEY` is empty in this machine's `.env` (confirmed before starting); the structuring call is tested against a mocked client only. Full account in `docs/JOURNAL.md`'s 2.10 entry.

Task 2.11 (Model health page) is complete — built later the same session, closing out **all of Phase 2**. New `src/ffapp/app/model_health_page.py` + `src/ffapp/app/pages/4_Model_Health.py`, SPEC §15's fourth page. Reads task 1.17's own already-built `report.md` directly and renders it via `st.markdown()` — deliberately no new chart, since 1.17 already made and documented the real decision to render "calibration plots" as a markdown table. Verified live in a real Chrome session against the real 2021-2025 evaluation report: every metric table, real feature importances, and real calibration tables rendered, 0 console errors; the report selector correctly showed only the one real directory (of three) that actually has a `report.md`. Full account in `docs/JOURNAL.md`'s 2.11 entry.

**Next task: undecided.** 1.18/1.19/2.8/2.10/2.11 close out everything else. **Phase 2 is now fully complete** — every task in TASKS.md's own Phase 2 list is checked. The only remaining open item anywhere is 1.15 (paused mid-follow-up, see above); everything past that is Phase 3 (SPEC §11.4's decomposed model v2, trade finder, empirical correlation estimation, season-long simulated rankings, route-data cost evaluation, public-readiness audit) — not started, not scoped in detail yet.

All of the above (2.8/2.10/2.11 plus the 1.15 follow-up) is real, committed work — six commits, `5fb86b9` through `ee8dbaf`, see the header. `docs/summary.md` is new this close-out: a standing narrative overview of the whole project (what it is, current state, what it does today, how it's performing, what's left), written for anyone — including a future cold-start session — who wants the full picture without reading `docs/JOURNAL.md` end to end.

**Blocking on me (the human), not the agent:**

- [x] GitHub remote created and pushed
- [x] Sleeper username supplied (`Maybe17`)
- [x] Primary league chosen — `rogan-radinator-league`
- [x] Ranking sources chosen — FantasyPros, ESPN, FantasySharks, CBS Sports, FFToday. **FFToday now 403s live; ESPN's bulk endpoint now returns 0 rows.** Board runs on 3 of 5 sources today; re-check both before the real Aug 22 draft.
- [x] Draft date and slot — slot 3 of 10, Aug 22 2026. League trades draft picks and is a 1-keeper league — see task 0.11's entry.
- [ ] Waiver type/budget — captured automatically via `--discover` (`rogan-radinator-league`: type 1, $100 FAAB). Resolved unless you want to override it.

---

## 3. In progress

Nothing is mid-implementation. The working tree is clean; everything described in §1 is committed (see the header for the real commit list). The only two genuinely open threads, both already named above, not repeated in full here:

- **1.15's hyperparameter-search follow-up** is paused, not finished — the search itself completed (25/25 trials, real results in `docs/JOURNAL.md`), but the planned confirmation step (wire the best-MAE config into `models/points.py`, re-run `ffapp evaluate --seasons 2021 2022 2023 2024` once) was never started. `models/points.py` itself is untouched — still the original shared `LightGBMSettings` defaults, no tuned override wired in anywhere. Resuming cold: either finish that confirmation run (low expectation it clears 1.15's bar, per the dev-season search result), or drop 1.15 and scope v2's decomposed pipeline (SPEC §11.4) as its own task instead — your call, not decided yet.
- **`structure_news_item`** (task 2.10) has never been exercised against a real Anthropic API call — `ANTHROPIC_API_KEY` is empty in this machine's `.env`. The natural first step once a key is supplied is one real live smoke-test call, not a code change.
- **`ee8dbaf` is committed but not pushed** — see the header. A `git push` is all that's needed; nothing else is pending on it.

This session's own work, in order: (1) reconciled `TASKS.md` against an orphaned commit (message `"g"`) from a prior session that had landed real, tested work for 1.13/1.17 without updating either tracking doc; (2) restructured this file — moved §2/§4/§5 to `docs/JOURNAL.md` verbatim, this file keeps only state and the rebuild/environment reference below; (3) built, verified, committed, and pushed task 2.6 (Waiver wire); (4) built, verified, committed, and pushed tasks 1.18 (Projection output pipeline) and 1.19 (Streamlit weekly rankings page) together, the latter verified live in a real browser session after resolving a real Chrome/Vivaldi extension-connection issue (see §7); (5) later session, same day: committed the doc reconciliation from (1)-(4) that had been left uncommitted (`1beeb5e`); (6) ran and stopped 1.15's hyperparameter-search follow-up (see above); (7) built, verified live, and documented task 2.8 (SOS and schedule grid), including finding and fixing a real confidence-threshold design bug live in the browser before calling it done; (8) built, verified against real 2015-2025 data, and documented task 2.10 (News pipeline), including finding and fixing two real schema/dtype bugs the propagation cascade's own fixture tests hadn't caught, before calling it done; (9) built, verified live, and documented task 2.11 (Model health page), closing out Phase 2 in full; (10) committed everything from (6)-(9) as five separate, logically-scoped commits, then discovered via `git status` (not assumed) that the prior session's `1beeb5e` had never actually reached `origin` despite its own header claiming otherwise, and pushed all six real commits together; (11) wrote and committed `docs/summary.md`, a new standing narrative overview of the whole project.

---

## 6. Rebuilding `data/` from empty

Full rationale for each step is in `docs/JOURNAL.md`'s per-task entries (§2). Run in this order on an unrestricted network — see ADDENDUM-02 §B.

```bash
uv run ffapp ingest sleeper --season 2026 --discover --no-offline
uv run ffapp cache warm --season 2026 --all-leagues --no-offline
uv run ffapp cache status
```

Player-id crosswalk (task 0.3 — no CLI command yet, only the function):

```bash
uv run python -c "from ffapp.ingest import nflverse; from ffapp.config import load_settings; nflverse.fetch_player_ids(offline=False, settings=load_settings())"
uv run ffapp ids check --season 2026
```

The six canonical interim tables (task 1.1, 2015–2025 — 2026 has no nflverse release until games are played) through the full feature pipeline (tasks 1.2–1.9) are built by a Python script, not a CLI command. `docs/JOURNAL.md` task 1.9's entry has the exact real call sequence ending in `features_build.build_player_week_features(...)` → `data/features/player_week_features.parquet`; task 1.1's entry lists the six interim tables and task 1.3/1.4's entries cover schedule/weather/injuries. **On a network where Sleeper is blocked**, `ids.mapping.build_players_dim` can skip `layer_sleeper_ids` — see JOURNAL task 1.1's entry for the exact workaround and its real coverage numbers.

Baselines B0–B3 (task 1.10): B0–B2 are pure functions over `player_week_features.parquet`. B3 needs a live FantasyPros weekly-archive git-history mining pass (`fetch_fp_weekly_commits`/`select_commit_before`/`fetch_fp_weekly_snapshot`, ~90+ GitHub requests for a 5-season range) — see JOURNAL task 1.10's entry for the exact sequence.

Rankings/ADP/traded-picks (tasks 0.7, 0.11) — no CLI command; see JOURNAL for the exact fetch calls, then:

```bash
uv run ffapp draft board --season 2026
```

Evaluation report (tasks 1.12–1.17):

```bash
uv run ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025
```

`--seasons` must repeat per value — typer's `Option` doesn't support `nargs=-1` (space-separated values after one flag), unlike SPEC's own literal example invocation. See JOURNAL's gotchas for the full explanation.

DST model (task 2.7): a narrower real materialisation, no Sleeper needed — only `pbp`/`team_stats`/`player_stats`/`schedules`/`snap_counts` (raw) and `schedule`/`team_week_context`/`weather` (interim). See JOURNAL task 2.7's entry for the exact script.

Weekly projections (task 1.18) — writes `data/outputs/projections.parquet`, upserted by `(season, week)`:

```bash
uv run ffapp project --season 2025 --week 10
```

Needs that `(season, week)`'s own row universe to already exist in `player_week_features.parquet` — for a genuinely live/current week this means whatever nflverse has published so far (2026 has none yet).

Schedule Grid page (task 2.8): no CLI command, same no-CLI precedent as every other UI-only page. Needs `interim/schedule.parquet`, `interim/defense_position_allowed.parquet`, and `features/player_week_features.parquet` to already exist — all three already built by the steps above, nothing task-2.8-specific to run first.

News pipeline (task 2.10): no CLI command. `ingest.news.fetch_rss_feed` needs `--no-offline` for a real fetch (writes `data/raw/news/<source>.xml`); `structure_news_item` needs a real `ANTHROPIC_API_KEY` in `.env` (empty on this machine — see §1). `tools.news_propagation.propagate_ruled_out_player` needs `interim/injuries.parquet`, `interim/team_week_context.parquet`, and `features/player_week_features.parquet` — all three already built by the steps above.

Model Health page (task 2.11): no CLI command. Needs at least one real `data/outputs/eval/<timestamp>/report.md` to exist — built by the evaluation report command above (task 1.12-1.17); nothing task-2.11-specific to run first.

**Not reproducible by re-running the above:** `data/raw/rankings/` (time-sensitive projections/ADP — committed to git as CLAUDE.md's stated exception; do not delete), `data/raw/sleeper/traded_picks_*.json`/`rosters_*.json` (keeper locks, time-sensitive), odds snapshots if the paid API is enabled.

**2026 nflverse data doesn't exist yet.** Every `fetch_*` call for season 2026 404s until real games are played.

**Must be supplied by hand on a new machine:** `.env` (copy from `.env.example`; `FFAPP_OFFLINE=1`/`FFAPP_CACHE_STRICT=1` are sane defaults).

**Windows console gotcha, not a code bug:** `print()`-ing a polars DataFrame crashes with `UnicodeEncodeError` under `cp1252` stdout encoding. Use `.to_dicts()`/plain prints in one-off scripts run here.

---

## 7. Environment notes

**Python:** 3.13.5 (3.11+ satisfied), managed with `uv`. `uv.lock` is committed.
**Offline default:** `FFAPP_OFFLINE=1` in `.env`. Cache warming and league discovery need `--no-offline` explicitly.
**Dependencies:** `lxml`(+stubs), `scikit-learn` (`numpy` pinned `<2.5` — see JOURNAL task 0.10's gotcha), `scipy`, `streamlit`(+`pandas-stubs`), `lightgbm`, `pandas`, `pulp`, `anthropic`, `defusedxml`(+dev-only `types-defusedxml`). Full per-task rationale for each addition is in `docs/JOURNAL.md`.

**Network reachability, most recent confirmation per location:**

| Location | Sleeper | nflverse / GitHub | PyPI | Rankings sites |
|---|---|---|---|---|
| `Maybe` (this session, 2026-08-13) | ✓ (`api.sleeper.app` 200) | not re-checked this session, previously ✓ | not re-checked, previously ✓ | not re-checked this session |
| Work, `ITHQ-172-26-LT1` (2026-08-13) | ✗ — Cisco Umbrella content filter blocks Sleeper specifically (confirmed: real `block.sse.cisco.com` redirect, not a TLS/cert issue) | ✓ | ✓ | ✓ (FantasyPros, FantasyFootballCalculator, CBS Sports confirmed) |
| Home | ? | ? | ? | ? |

If a new machine shows up, add a row rather than overwriting existing ones — don't assume `Maybe` again without checking the actual hostname.

**`claude-in-chrome` browser verification on this machine (`Maybe`) needs the actual Chrome browser, not Vivaldi (the machine's apparent default) or any other Chromium-based browser, and the Claude extension must be installed *and signed in* there specifically.** `list_connected_browsers`/`tabs_context_mcp` both return empty until that's true — closing Vivaldi and opening Chrome alone isn't sufficient if the extension isn't logged into the same claude.ai account this session uses. Once signed in, connection works immediately (confirmed live, task 1.19's own real browser verification).

---

## Maintenance rules

At the end of every session, before committing:

1. Update the header (date, machine, commit).
2. Move newly completed tasks' evidence into `docs/JOURNAL.md` §2, and tick the matching box in `TASKS.md`.
3. Update §1 with the next task and its first concrete step.
4. Record anything half-finished in §3, precisely enough to resume cold.
5. Add any implementation decision to `docs/JOURNAL.md` §4 and any gotcha to §5.
6. Keep §6 and §7 accurate as commands and reachability change.

Delete sections that have gone stale rather than letting them rot. A handoff file nobody trusts is worse than none.
