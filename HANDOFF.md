# HANDOFF.md — current state

**Last updated:** 2026-08-14
**Last machine:** `Maybe` (Windows).
**Last commit:** see `git log` — this session's own work (DST/K streaming-aware VOR + board exclusion, see below) is committed by the end of this session; verify with `git status`/`git log origin/main` rather than trusting this line, same lesson the prior header's own wrong "pushed" claim already taught. Everything through the prior session's own work was confirmed actually on `origin` before this session started.

This file is **state and pointers**, not design or history. `SPEC.md` and the addenda hold the design. Per-task evidence, implementation decisions, and gotchas live in `docs/JOURNAL.md` — read that when investigating a specific past task or bug, not every session. `docs/summary.md` is a third, newer reference: a standing narrative snapshot of current state and how things are going, meant to be read cover to cover rather than searched. Maintenance rules are at the bottom.

---

## 1. Where things stand

Current phase: **Phase 2 is fully complete; both of ADDENDUM-03's new Phase 0 tasks (0.15, 0.16) are done.** `SPEC-ADDENDUM-03.md` (phone access and draft-day operations, real Aug 22 draft deadline) landed this session. **0.15 (mobile Streamlit page) and 0.16 (static HTML export) are both done this session** — see `docs/JOURNAL.md`. Phase 1 still has exactly one open item, 1.15 (see below); every other Phase 1 task is done. Nothing in Phase 3 has been started. The narrative below is kept in roughly chronological order across sessions, oldest first — read `docs/summary.md` instead if you want the current picture without the history (note: `docs/summary.md` predates this session's ADDENDUM-03 work and hasn't been refreshed to mention it yet).

Tasks 1.13 (Metrics module) and 1.17 (Evaluation report) are complete. Verified this session with a real 5-season walk-forward run: `uv run ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025` wrote `data/outputs/eval/20260813T183533Z/report.md` with populated metrics, real LightGBM feature importances, and real calibration curves. Both tasks' evidence is in `docs/JOURNAL.md`.

Phase 2 tasks 2.1–2.5, 2.7, and 2.9 were already complete — the simulation layer, start/sit assistant, DST model, and trade analyzer — built out of TASKS.md order on a prior work machine where Sleeper was network-blocked, confirmed with you first (see 2.1's entry in `docs/JOURNAL.md`).

Task 2.6 (Waiver wire) is complete — built this session, the first task on this machine (Sleeper reachable here) to actually need Sleeper's free-agent pool and transaction history. New `src/ffapp/tools/waivers.py` + `config.WaiverSettings`. Verified against the completed 2025 season (2026's real rosters are pre-draft leftovers, not usable — draft is Aug 22) and against real historical FAAB bidding data pulled from `bdff-chopped`'s actual transactions. Committed as `ea5cd70`, pushed.

Task 1.18 (Projection output pipeline) is complete — new `src/ffapp/models/predict.py` + `ffapp project --week N` CLI command, composing the availability/points/quantile models (tasks 1.14/1.15/1.16) into SPEC §6.2's `outputs/projections.parquet` schema. Verified for real against already-played weeks (2025 weeks 10/11 — 2026 has no nflverse release yet): 445/477 real projections, 0 nulls, full provenance, real upsert-by-`(season, week)` behavior confirmed with a second real run.

Task 1.19 (Streamlit weekly rankings page) is also complete — new `src/ffapp/app/weekly_rankings_page.py` + `src/ffapp/app/pages/2_Weekly_Rankings.py`, SPEC §15's second page. Verified live in a real Chrome session via `claude-in-chrome` (see §7 for the connection hiccup and how it resolved) against the real weeks 1.18 generated: position tabs, the floor/ceiling visible range, and both filters (position + real Sleeper-resolved availability) all confirmed working against real data, 0 console errors. Both 1.18 and 1.19 committed as `a75554a`, pushed. Full evidence for both tasks is in `docs/JOURNAL.md`.

Task 1.15's hyperparameter-search follow-up was attempted this session (a real 25-trial random search, `notebooks/tune_points_v1.py`/`.log`) and stopped partway through on your instruction ("stop here, deal with this later") — full result in `docs/JOURNAL.md`'s 1.15 entry and TASKS.md's own updated status line. Short version: tuning moves MAE past B2 (best config 5.005 vs B2's 5.107 on dev seasons 2018-2020) but **not one of 25 trials beat B2 on Spearman-within-position-week** — confirms the gap is architectural (SPEC §11.4's own diagnosis), not an undertuned model. The real confirmation run (`ffapp evaluate --seasons 2021 2022 2023 2024` with the tuned config wired into `points.py`) was deliberately not run — next step if resumed.

Task 2.8 (SOS and schedule grid) is complete — built later the same session, after 1.15's follow-up was paused. New `src/ffapp/tools/sos.py` (positional SOS, the schedule-grid pivot, matchup detail) + `src/ffapp/app/schedule_grid_page.py` (roster-highlight resolution, heatmap styling) + `src/ffapp/app/pages/3_Schedule_Grid.py`, SPEC §15's third page. A real design bug was caught live in a running Streamlit session (not a fixture) and fixed before shipping: the first confidence-threshold design reused SPEC §10.4's `k=250` shrinkage constant directly, which greyed out every real cell at every position — fixed with a per-position-group threshold computed from that group's own real data. Full account, including a second smaller Streamlit-rendering gotcha, in `docs/JOURNAL.md`'s 2.8 entry. Verified live end to end in a real Chrome session against `rogan-radinator-league`/2025 data, 0 console errors.

Task 2.10 (News pipeline) is complete — built later the same session, the biggest single Phase 2 task by SPEC's own hour estimate. New `src/ffapp/ingest/news.py` (real RSS ingestion — ESPN/CBS Sports/Yahoo NFL feeds, all three confirmed live before writing any code — plus the real Anthropic API structuring call, `claude-opus-5`, structured outputs, confidence/name-resolution routing to a real manual review queue) and `src/ffapp/tools/news_propagation.py` (the real second-order cascade SPEC calls "the valuable part": reuses task 1.7's own already-validated `add_vacated_shares` fed a synthetic early `Out` event, re-runs `models.predict.project_week` unmodified, surfaces the real handcuff in exactly the shape `tools.waivers.build_waiver_board` already expects). Two real bugs were caught only by running the cascade against real 2015-2025 data (a schema-mismatch concat and an `Int32`/`Int64` dtype mismatch, both invisible to same-shaped fixtures) and fixed with dedicated regression tests. Verified against a real historical case (Bucky Irving ruled out, Tampa Bay, real 2025 week 5): the recomputed vacated shares closely matched the real official values the existing pipeline separately produced, the real handcuff (Rachaad White) was correctly identified, and his recomputed projection moved substantially in the right direction, then flowed through the real waiver-board function end to end. **Not verified live against the real Anthropic API** — `ANTHROPIC_API_KEY` is empty in this machine's `.env` (confirmed before starting); the structuring call is tested against a mocked client only. Full account in `docs/JOURNAL.md`'s 2.10 entry.

Task 2.11 (Model health page) is complete — built later the same session, closing out **all of Phase 2**. New `src/ffapp/app/model_health_page.py` + `src/ffapp/app/pages/4_Model_Health.py`, SPEC §15's fourth page. Reads task 1.17's own already-built `report.md` directly and renders it via `st.markdown()` — deliberately no new chart, since 1.17 already made and documented the real decision to render "calibration plots" as a markdown table. Verified live in a real Chrome session against the real 2021-2025 evaluation report: every metric table, real feature importances, and real calibration tables rendered, 0 console errors; the report selector correctly showed only the one real directory (of three) that actually has a `report.md`. Full account in `docs/JOURNAL.md`'s 2.11 entry.

**Next task: undecided — the model v2 Stage 1 verification you asked for is done; the result is a real, honest mixed outcome, your call on next steps.** Model v2 Stage 1 (team environment, TASKS.md 3.1, SPEC §11.4) is built and verified against real 2021-2025 data: it beats the design's own stated bar (`trailing_ewm_4`) on MAE for both `team_plays` (7.053 vs 7.463) and `pass_rate` (0.0861 vs 0.0886) — a real pass, the first model in the decomposed-v2 line to clear its bar — but it does *not* beat the plain `league_mean` sanity floor on either target (6.748 and 0.0852 respectively), which wasn't the pass/fail bar but is still a real, notable wrinkle worth your attention before treating Stage 1 as "working" in a stronger sense than "technically passed." Not tuned to force a cleaner number. Full account: `docs/JOURNAL.md`'s new entry, `.superpowers/sdd/2026-08-14-model-v2-stage1-team-environment/task-6-report.md`. Per the design's own "after this plan" instruction, Stage 2 was not started automatically — decide whether to proceed to Stage 2 (opportunity) as-is, investigate the league_mean gap first, or stop here, same as 1.15's own outcome. Before that: DST/K's replacement-level VOR fixed (streaming-aware, real historical data) and then DST/K excluded from the draft board entirely, per your own explicit workflow (see below) — both live-verified against the real 2026 board. See `docs/JOURNAL.md` for the full account. 1.15 remains a separate open item outside Phase 0/3 (paused mid-follow-up, see below); everything past Phase 0/1 is Phase 3 (SPEC §11.4's decomposed model v2 — Stage 1 above, Stages 2-4 not started — trade finder, empirical correlation estimation, season-long simulated rankings, route-data cost evaluation, public-readiness audit) — not started, not scoped in detail yet.

This session (2026-08-14): you kept feeling a DST/K appearing anywhere near draft-relevant was wrong even after the TE/K explanation from the prior session (below) — investigated rather than just capping the display. Built this league's real 2021-2025 historical DST/K scoring and simulated a simple streaming strategy (best available matchup each week): it beat even the single best *drafted* DST/K's own season total by ~1.5-2x, every one of the last 5 real seasons — confirms `tools.vor`'s "Nth-best preseason total" replacement level is the wrong proxy for a streamed position, exactly as SPEC §9.4 itself implies but the standard fixed-point baseline doesn't achieve on its own. New `tools/streaming.py` (empirical replacement level from real history, not a guessed constant) + a new `replacement_overrides` param on `tools.vor`. Verified live: best DST moved from board rank 44 (VOR +27.7) to 251 (VOR -97.4); best K from rank 67 (VOR +6.9) to 918 of 1076 (VOR -192.0); TE (McBride/Bowers, #8/#9) and everything else unchanged. Then, on a direct follow-up ("remove all of the kickers and defenses off the draft board"), added `draft.excluded_positions` (`config/settings.yml`, `["DST", "K"]`) — DST/K are now fully absent from the board (981 rows, positions exactly `QB/RB/TE/WR`), not just buried; the streaming-aware replacement level from the first fix is kept, not deleted, so it's still correct if that exclusion is ever turned off for a different league/scenario. Also added TASKS.md 3.7 (Phase 3, explicitly deferred): a weekly DST/K streamer tab for *that week's* projection, since you stream both positions rather than draft them — `models.dst.weekly_streamer_list` (task 2.7) already produces this for DST. Full evidence, including the real per-season streaming numbers and a separately-noticed-but-deliberately-unfixed real issue (kicker preseason point projections look implausibly low — SPEC §11.7 already says not to spend effort there), in `docs/JOURNAL.md`.

This session (2026-08-14, later): model v2 Stage 1 (team environment, SPEC §11.4's decomposed pipeline, TASKS.md 3.1) built end to end via a 6-task plan (design/plan docs under `.superpowers/sdd/2026-08-14-model-v2-stage1-team-environment/`) and verified for real against 2021-2025 data (Task 6). Tasks 1-5: new `opponent_neutral_pace_ewm_8` feature (with a real leakage bug caught and fixed in review before Task 2 even started — the brief's own same-week join design leaked the shared game's outcome; fixed to use the opponent's already-lagged pace), `models/team_environment.py`'s DST-style table reshape, the two baseline columns, the two monotonic-constrained LightGBM regressors, and derived `pass_attempts`/`rush_attempts` (always sum to `team_plays` exactly, by construction). Task 6's real result: **Stage 1 beats the design's own stated bar (`trailing_ewm_4`) on MAE for both `team_plays` (7.053 vs 7.463) and `pass_rate` (0.0861 vs 0.0886)** — a real pass, the first model in the decomposed-v2 line to clear its bar — **but does not beat the plain `league_mean` sanity floor on either target** (6.748 and 0.0852), which wasn't the pass/fail bar but is a real, honest wrinkle reported as-is, not tuned away. Full result and the note that the verification script had to be re-run directly (no process was actually executing it when this task resumed) in `docs/JOURNAL.md`'s new entry and `.superpowers/sdd/2026-08-14-model-v2-stage1-team-environment/task-6-report.md`. Per the design's own "after this plan" instruction, Stage 2 (opportunity) was not started — this is a real decision point for you, same pattern as 1.15's own outcome.

This session (2026-08-13): SPEC-ADDENDUM-03.md housekeeping; task 0.16 (static HTML export) and task 0.15 (mobile draft page, with a new `ffapp draft live --replay/--stop` command) both built and verified live; **ESPN and FFToday rankings sources fixed**; **two new ranks-only sources added, FootballGuys (526 real players) and DraftSharks (125 real players, `Crawl-delay: 10` respected)** — `draft/board.py` generalised to a `_RANK_SOURCE_FETCHERS` dict; **keepers restored to the board, visibly marked** — prototyped safely against the dummy CSV first, then built into `draft/board.py` for real once you confirmed it, with a distinct amber highlight + 🔒 marker in both the Streamlit board page and the static HTML export (a real debugging detour here: the Streamlit page briefly *appeared* unstyled after this change, turned out to be a stale long-running dev-server process with a cached pre-edit import, not a real bug — restarting fixed it, and that lesson paid off again below); **a real DST cross-source matching bug found and fixed** — you flagged TE (McBride/Bowers) and DST/K ranking unusually high, which turned out to be two very different things: the TE/K part is real, intended VOR behaviour (all 7 sources agree, dispersion lower than comparable WRs, real ADP confirms the model is finding genuine positional-scarcity value ADP-driven consensus doesn't price in — not a bug), but DST was a real bug: every one of the 32 real teams was spelled a different way by each of 5 sources (ESPN "Texans D/ST", FantasySharks/FFToday "Houston Texans", CBS "Houston", ADP "Houston Defense"), so cross-source matching silently never aggregated any DST at all — every DST board row was really one source's single, unchecked opinion (`n_sources=1`, `dispersion=0.0`, `adp=None` on every one). Fixed with a new canonical team-name table in `projections/aggregate.py` (32 teams × the 4 confirmed real spellings each); DST now properly aggregates (`n_sources=6` per team, real dispersion, real ADP). Board grew 854 → 1141 → 1077 players over the session (rankings expansion added players, then the DST fix collapsed ~64 duplicate unaggregated DST rows into 32 real ones); **Draft Board page row cap added** — you asked for the board to stop rendering all 1000+ rows by default (genuinely slow to paint) and cap at 400 unless a button/checkbox is used to see more. New `cap_rows` in `app/draft_board_page.py`, wired into the sidebar as a checkbox that only appears once the filtered board actually exceeds 1000 rows; verified live (dev server restarted first, same lesson as above) showing "400 of 1077" capped, then "1077 of 1077" once checked. Full evidence for all of the above in `docs/JOURNAL.md`. `docs/summary.md` was not refreshed this session — still describes the state as of the prior close-out; it doesn't mention any of this session's work yet.

**Blocking on me (the human), not the agent:**

- [x] GitHub remote created and pushed
- [x] Sleeper username supplied (`Maybe17`)
- [x] Primary league chosen — `rogan-radinator-league`
- [x] Ranking sources: **seven now live and contributing** — FantasyPros, ESPN, FantasySharks, CBS Sports, FFToday (ESPN/FFToday fixed this session), plus FootballGuys and DraftSharks (added this session). See `docs/JOURNAL.md` for per-source depth/coverage detail.
- [x] Draft date and slot — slot 3 of 10, Aug 22 2026. League trades draft picks and is a 1-keeper league — see task 0.11's entry.
- [ ] Waiver type/budget — captured automatically via `--discover` (`rogan-radinator-league`: type 1, $100 FAAB). Resolved unless you want to override it.

---

## 3. In progress

Nothing is mid-implementation. Everything asked for this session (2026-08-14: DST/K streaming-aware VOR + board exclusion) and the prior one (2026-08-13: rankings repair/expansion, keepers restored to the board) is fully done, verified live, and committed. Other genuinely open threads, already named above, not repeated in full here:

- **Keepers don't yet show a marker on the mobile draft page (task 0.15) or in the live draft assistant (`draft.live`)** — `available_pool` will naturally stop showing a keeper once Sleeper's own real pick event fires during the live draft, but before that (right now, pre-draft) a keeper still appears as an ordinary "best available" card with no visual distinction. Worth a small follow-up if that turns out confusing on draft day.
- **`data/outputs/draft_board_2026_dummy.csv` still exists** — a safe test copy from this session's keeper exploration, gitignored, harmless to leave or delete.

- **1.15's hyperparameter-search follow-up** is paused, not finished — the search itself completed (25/25 trials, real results in `docs/JOURNAL.md`), but the planned confirmation step (wire the best-MAE config into `models/points.py`, re-run `ffapp evaluate --seasons 2021 2022 2023 2024` once) was never started. `models/points.py` itself is untouched — still the original shared `LightGBMSettings` defaults, no tuned override wired in anywhere. Resuming cold: either finish that confirmation run (low expectation it clears 1.15's bar, per the dev-season search result), or drop 1.15 and scope v2's decomposed pipeline (SPEC §11.4) as its own task instead — your call, not decided yet.
- **`structure_news_item`** (task 2.10) has never been exercised against a real Anthropic API call — `ANTHROPIC_API_KEY` is empty in this machine's `.env`. The natural first step once a key is supplied is one real live smoke-test call, not a code change.
- **No `ffapp ingest rankings` CLI command exists**, despite ADDENDUM-03 §E's own draft-day runbook assuming one (`uv run ffapp ingest rankings --no-offline` as the "morning of" step). Today, refreshing rankings/ADP live only happens as a side effect of `draft board`/`draft export`'s own `--no-offline` flag — which works (confirmed live this session) but doesn't match the runbook's literal wording. Worth a small dedicated task before Aug 22 if the runbook is meant to be followed literally, not revisited this session.
- **`docs/summary.md` is stale** — written before ADDENDUM-03 landed, doesn't mention the two new Phase 0 tasks. Refresh when next touched, per its own "rewrite in place when stale" convention.

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

**`claude-in-chrome` browser verification on this machine (`Maybe`) needs the actual Chrome browser, not Vivaldi (the machine's apparent default) or any other Chromium-based browser, and the Claude extension must be installed *and signed in* there specifically.** `list_connected_browsers`/`tabs_context_mcp` both return empty until that's true — closing Vivaldi and opening Chrome alone isn't sufficient if the extension isn't logged into the same claude.ai account this session uses. Once signed in, connection works immediately (confirmed live again this session, task 0.16's own real browser verification, same as task 1.19's).

**`claude-in-chrome`'s `navigate` tool refuses `file://` URLs by default** ("Can't interact with browser-internal or unparseable URLs") — confirmed live this session verifying task 0.16's static export. Workaround: serve the directory over `127.0.0.1` with `python -m http.server <port>` and navigate to that instead; doesn't compromise a "no network calls" test since `read_network_requests` still shows exactly what the page itself requests.

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
