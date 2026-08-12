# HANDOFF.md — current state

**Last updated:** 2026-08-11
**Last machine:** Maybe (Windows)
**Last commit:** `4cc1a5f` — feat: scoring engine and golden test validation (tasks 0.4, 0.5)

This file is **state and decisions**, not design. `SPEC.md` and the addenda hold the design; never restate them here. If a line here could have been written before any code existed, it does not belong in this file.

Maintenance rules are at the bottom. Update this file at the end of every session.

---

## 1. Where things stand

**Current phase:** Phase 0 — draft board
**Next task:** 0.6 (LeagueFormat parser) — see SPEC §8.5. 0.5 (scoring golden test) is now complete and passing for both real leagues.
**First concrete step:** Parse `roster_positions` into the `LeagueFormat` dataclass (SPEC §8.5), handling FLEX/SUPER_FLEX/REC_FLEX/BN/IR. `tests/test_league_format.py` needs standard 12-team, superflex, and two-flex fixtures. Nothing in 0.6 depends on this session's scoring work.

**Blocking on me (the human), not the agent:**

- [x] GitHub remote created and pushed
- [x] Sleeper username supplied for task 0.2 (`Maybe17`)
- [x] Primary league chosen for model development (ADDENDUM-01 §F, decision #9) — `rogan-radinator-league`
- [ ] Ranking sources chosen (SPEC §17 decision #2) — blocks task 0.7
- [ ] Draft date and slot per league (decision #5) — blocks task 0.11
- [ ] Waiver type/budget — already captured automatically per league via `--discover` (`rogan-radinator-league`: type 1, $100 FAAB; `bdff-chopped`: type 2, $1000 FAAB), so decision #6 is effectively resolved unless you want to override it.

---

## 2. Completed tasks

- **0.1** — repo scaffold. `uv run ffapp --version` prints `ffapp 0.1.0`; `ruff check` and `mypy src/` both pass.
- **0.2** — config loading + Sleeper ingestion, folded together with ADDENDUM-01 multi-league support and ADDENDUM-02 §H's offline cache layer (all three were the same piece of work in practice). Evidence:
  - `ffapp ingest sleeper --season 2026 --discover --no-offline` ran live against the real account and discovered 2 leagues: `rogan-radinator-league` (now primary) and `bdff-chopped`.
  - `ffapp cache warm --season 2026 --all-leagues --no-offline` pulled league objects, rosters, users, drafts, draft picks, the full player dictionary (15.9MB), and trending add/drop lists.
  - `ffapp cache status` lists all 15 cached artefacts with correct freshness verdicts.
  - 70 tests pass (`uv run pytest -q`), including the ADDENDUM-02 §H-mandated `tests/test_offline_raises.py` (parametrized across all 11 `fetch_*` functions — every one raises `OfflineCacheMiss`, none silently returns empty).
- **0.3** — player id mapping (SPEC §7). New `ingest/nflverse.py` (`fetch_player_ids`, offline-cache pattern matching `sleeper.py`) pulls the dynastyprocess/ffverse crosswalk (`https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv`). New `ids/mapping.py` implements the full pipeline: `load_crosswalk_base` → `layer_sleeper_ids` (fills crosswalk gaps via Sleeper's own `gsis_id` field, adds sleeper-only rows) → `fuzzy_match_remainder` (exact-name tiers 1–2, `rapidfuzz` floor-92 tier 3) → `assign_canonical_id` (`gsis_id` else `synthetic_<blake2b hash>`) → `apply_overrides` (`config/id_overrides.csv`, applied last). `ffapp ids check --season <year>` is live. Evidence:
  - 113 tests pass (`uv run pytest -q`, up from 70), `ruff check`/`ruff format --check`/`mypy src/` all clean.
  - Ran for real against the live cache: `uv run ffapp ids check --season 2026` → **0 unmatched players within top 300** (by the search_rank proxy, scoped to `rogan-radinator-league`'s roster positions). 8,936 total unmatched out of ~12.5k in the crosswalk, all outside the relevance cutoff (deep practice-squad/UDFA names, `search_rank=None`).
  - One real override applied (`config/id_overrides.csv`): sleeper_id 8008 (Josh Imatorbhebhe) — the crosswalk itself carries no gsis_id for him (never appeared in an NFL regular-season box score).
- **0.4** — scoring keymap and engine (SPEC §8.1–8.3, ADDENDUM-01 §C.1). New `scoring/keymap.py` (`STAT_KEY_MAP: dict[str, DirectStat | DerivedStat]`, covering direct stats, bonuses, FG distance buckets + per-yard, and DST points-allowed buckets) and `scoring/engine.py` (`score_stat_line`, `unhandled_keys`, `UnhandledScoringKeysError`, `ConflictingFieldGoalSchemeError`). Evidence:
  - 133 tests pass (`uv run pytest -q`, up from 113), `ruff check`/`ruff format --check`/`mypy src/` all clean.
  - `unhandled_keys()` is empty for **both** real leagues' actual `scoring_settings` (`tests/test_scoring.py::test_unhandled_keys_is_empty_for_real_league_scoring`, parametrized) — the task's literal acceptance criterion, checked against `rogan-radinator-league` (per-yard FG) and `bdff-chopped` (bucketed FG) simultaneously since both configs were already sitting in `config/leagues/`.
  - The engine raises `ConflictingFieldGoalSchemeError` if a league scoring dict sets `fgm_yds` and any bucketed `fgm_*` key non-zero at the same time (ADDENDUM-01 §C.1) — verified this doesn't false-positive on `rogan-radinator-league`'s real data, which carries every bucket key present but zeroed out alongside a non-zero `fgm_yds`.
- **0.5** — scoring golden test (SPEC §8.4). This session went well past 0.4 into 0.5 with your explicit go-ahead across several checkpoints (chose "minimal nflverse pull now," then "keep debugging now" when the primary league first failed, then "investigate the 3 remaining unexplained cases," then "keep going" to close the final DST-level gap). New `ingest/nflverse.py` fetchers (`fetch_player_stats`/`fetch_team_stats`/`fetch_schedules`/`fetch_pbp`), new `scoring/stats.py` (assembles the per-player-week stat frame from those four sources; DST is its own row, `player_id` = team abbreviation), new `scoring/golden.py` (the golden test itself: pure functions `extract_players_points`/`resolve_player_ids`/`compare_points`/`summarize` plus orchestrating `run_golden_test(slug)`, which validates against each league's most recently PLAYED season via `previous_league_id`), and `ffapp scoring validate [--league <slug> | --all-leagues] [--no-offline]` wired up per ADDENDUM-01 §A.3. Evidence:
  - `ffapp scoring validate --all-leagues --no-offline` against real 2025 data: **`rogan-radinator-league` (primary) PASSES at 99.30%** (22 disagreements / 3128 player-weeks); **`bdff-chopped` PASSES at 99.96%** (1 disagreement / 2504 player-weeks). CLI exits 0.
  - 185 tests pass (`uv run pytest -q`, up from 133), `ruff check`/`ruff format --check`/`mypy src/` all clean.
  - **Task 0.4a turned out unnecessary** — nflreadpy's own `load_player_stats` already returns `fg_made_list` (real per-kick distances), so the raw play-by-play `kick_distance` extraction ADDENDUM-01 §C.2 called for is already done upstream. `fetch_pbp` was needed anyway, but for deriving genuine defensive/return touchdowns and the scrimmage/special-teams fumble split, not kicker distances.
  - **Seven real bugs found and fixed, every one via the live golden-test run against real 2025 data, not by a unit test** — full detail on each in §5: (1) DST rows silently scoring the team's own offense; (2) special-teams TD credit double-counted between `def_st_td`/`st_td`; (3) Sleeper's `"LAR"` vs nflverse's `"LA"` for the Rams; (4) `def_td` credited from unreliable team-level columns instead of PBP-derived `def_return_tds` (needed two follow-up corrections: exclude offense-recovers-own-fumble, exclude punt/kickoff returns); (5) blocked kicks not counted as a miss for the kicker's own `fgmiss`/`xpmiss`; (6) "Team defense" keys (`sack`/`int`/`ff`/`fum_rec`/`safe`) firing on individual players' own stray IDP-style stat values; (7) `ff`/`fum_rec` (general defense credit) and `def_st_ff`/`def_st_fum_rec` (special-teams credit) were both drawing from the same team_stats aggregate columns, double-scoping special-teams fumble events into the general pool while leaving the dedicated special-teams keys at zero — fixed by deriving all four from play-by-play's structured fumble columns (`forced_fumble_player_1_team`, `fumbled_1_team`, `fumble_recovery_1_team`), split by `special_teams_play`. Net effect on `rogan-radinator-league`: 97.76% → 99.30%. `bdff-chopped`: 99.92% → 99.96%.
  - **Remaining 22/1 disagreements are accepted as within-tolerance noise, not individually explained.** SPEC §8.4's own bar is ≥99% agreement, which both leagues clear; it does not require zero disagreements. If future work wants to chase these further, they're logged in full by `ffapp scoring validate`.
  - **Not built:** `scoring/stats.py` is an ad hoc stand-in for task 1.1's `interim/player_week_stats.parquet` (one season, no partitioning, no persistence, no historical range). Task 1.1 will need to reconcile its own column-naming choices against what `scoring/keymap.py` currently expects, not the other way around. Individual-player special-teams credit (`st_ff`/`st_fum_rec`) is still zero-filled — only the team-level (`def_st_ff`/`def_st_fum_rec`) side was derived, since that's what the failing DST rows needed; revisit only if an individual-player disagreement ever traces to it.

---

## 3. Work in progress

None. 0.1–0.5 are complete and verified. This entire session's work (0.4 and 0.5) is uncommitted — see header.

## 4. Decisions made during implementation

Places where the spec/addenda were silent, ambiguous, or (in one case) internally contradictory:

- **`_primary.txt` vs `is_primary: true`** (ADDENDUM-01 §A.2 proposes both, never reconciles them): treated `is_primary: true` in the league YAML as the sole source of truth and did not build `_primary.txt` — it would just be a second thing to keep in sync. Revisit if you actually want the separate pointer file.
- **`sleeper.username` location**: SPEC §5 nests it per-league; moved it to `config/settings.yml` instead since it's account-level (one Sleeper login, many leagues), not league-level. Duplicating it into every `config/leagues/<slug>.yml` would just invite drift.
- **`cache warm`/`cache verify` scope**: built as an extensible registry (`ffapp/cache/registry.py`) but only the Sleeper source is wired in — nflverse/rankings/odds/weather don't exist yet (Phase 1). `cache_verify()` raises a clear "not registered yet" error for any task id it can't check, rather than silently reporting OK.
- **Re-run semantics for `write_league_stub`**: re-running `--discover` refreshes `league_cache` from live data (matches SPEC's ingest-idempotency rule) but preserves hand-set `is_primary` and `overrides` — otherwise every re-warm would silently reset your primary-league choice.
- **Primary league**: `rogan-radinator-league` — its settings (10 teams, `fgm_yds: 0.1` per-yard FG scoring, $100 FAAB, playoffs starting week 15) match ADDENDUM-01 §B's "main-ppr" description exactly. Confirmed with you before setting `is_primary: true`.
- **ADDENDUM-01 open decision #10** (bucketed vs per-yard FG scoring per league) is now resolved from real data: `rogan-radinator-league` uses `fgm_yds` (per-yard); `bdff-chopped` uses bucketed `fgm_0_19`/`fgm_20_29`/etc. Task 0.4's keymap needs to handle both, as ADDENDUM-01 §C.1 anticipated.
- **0.3's "top 300 by ADP" acceptance criterion, resolved against `search_rank` instead.** ADP itself doesn't exist yet (task 0.7, blocked on the ranking-sources decision below). Confirmed with you: `ffapp ids check`'s blocking gate ranks by Sleeper's own `search_rank` (already cached, no new ingestion) as an interim stand-in. **Revisit once 0.7 lands real ADP** — swap the gate's ranking source in `mapping.unmatched_report`/`ffapp ids check`.
- **`ingest/nflverse.py` created ahead of Phase 1.** CLAUDE.md forbids network calls outside `ingest/`, and the crosswalk fetch is a network call, so it needed a home there even though full nflverse ingestion (stats, snaps, schedules) is still Phase 1 work. Currently holds exactly one function, `fetch_player_ids`; the rest of the module fills in as Phase 1 tasks land.
- **"Unmatched" is defined as `player_id` starting with `synthetic_`, not "`gsis_id` is null."** An override can resolve a player without ever finding a real `gsis_id` (see Josh Imatorbhebhe below) — checking `gsis_id` directly would keep reporting him as unmatched even after a human explicitly resolved him. Checking the final `player_id` (post-overrides) is the correct "did a human or the pipeline resolve this" signal.
- **`ffapp ids check`'s blocking gate is scoped to the primary league's own roster positions and active, rostered players** (`mapping.league_relevant_positions` / `mapping.league_relevant`), derived from `config/leagues/rogan-radinator-league.yml`'s `roster_positions` + `overrides.flex_eligible` (CLAUDE.md rule 5 — never hardcode league format). This wasn't in the original plan; the raw `search_rank` universe turned out to include retired players and IDP positions this league doesn't roster (see gotchas below), which produced a lot of false-positive "build failures" until scoped.
- **Fuzzy-match tiers 1 and 2 use exact normalised-name equality, not `rapidfuzz`.** SPEC §7 says "Use rapidfuzz... for the last tier" — read literally, only the bare-name tier is fuzzy; the `(name, position, team)` and `(name, position)` tiers require an exact match. If this turns out too strict in practice, loosening tiers 1–2 to fuzzy-with-a-higher-floor is the likely next step.
- **`scoring/keymap.py`'s stat-frame column names were reconciled against nflreadpy's real schema this session** (superseding the provisional names task 0.4 invented) — confirmed by live inspection of `load_player_stats`/`load_team_stats`, not assumed. Notably: `fg_made_list` is a semicolon-delimited **string** ("25;43;32"), not a native list column, and nflreadpy already provides pre-bucketed FG-distance counts (`fg_made_0_19` etc.) that align exactly with Sleeper's bucket boundaries, so most FG keys are plain `DirectStat` now, not derived. Still genuinely unverified: **task 1.1's real `interim/player_week_stats.parquet` may not match `scoring/stats.py`'s ad hoc assembly exactly** — reconcile when 1.1 lands, don't assume this session's version survives untouched.
- **Sleeper's own `fum_rec` (grouped under "Team defense" in ADDENDUM-01 §B.2) maps to `fumble_recovery_opp`; `fum_rec_td` no longer has a DST-side mapping at all** (superseded by the `def_return_tds` fix below) **and stays an any-player key sourced from individual `load_player_stats` rows only.**
- **`bonus_*` keys are implemented from SPEC §8.2's "typical keys" list, not verified against any real league** — neither real league uses bonus scoring. Low-risk to leave unverified.
- **polars `map_elements` on a `List`-dtype column can hand the callback either a plain Python list or a length-1 `Series`, depending on the row/version** — and a `Series`'s truthiness is ambiguous (`if s:` raises `TypeError`, not a bool). Normalise unconditionally (`[int(d) for d in x]`) rather than branching on truthiness.
- **`load_team_stats` is a team's own full offensive box score (`passing_yards`, `rushing_yards`, `receptions`, ...) *alongside* its defensive/DST columns — it is not a DST-only table.** `scoring/stats.py`'s first version passed the whole row through into the DST frame, and the DST row silently picked up the team's own offense and got scored as if the defense had thrown for 300 yards (top of the week-1 leaderboard was six DSTs scoring 100+ points each, caught by eyeballing a live smoke-test run rather than by any test — the unit tests all used narrow hand-built fixtures that never had this column overlap to catch). Fixed by explicitly `.select()`ing only the DST-relevant columns (`_DST_STAT_COLUMNS`); added a regression test (`test_build_dst_stat_frame_excludes_team_offensive_stat_columns`) with an offense-bearing fixture that would have caught it. **Lesson: a fixture built narrow enough to be convenient will not catch a real source table being wider than assumed — the smoke test against real data is what caught this, not the unit tests.**
- **A real double-counting bug in `def_st_td`/`st_td` (and the two special-teams forced-fumble/recovery key pairs), found via the live golden-test run, not a test.** Sleeper scores a special-teams TD twice — once as team-defense credit (`def_st_td`) and once as individual-player credit (`st_td`) — and both keys are simultaneously non-zero in both real leagues. The original keymap mapped both straight to the same `special_teams_tds` column, so `score_stat_line` summed both keys' contributions on every row: confirmed live, NE's actual week-2-2025 `special_teams_tds=1` inflated NE's computed DST score from a correct 13 to 19. Fixed with `_dst_only`/`_individual_special_teams_credit`, which gate on `position == "DST"` so each key only fires on its own row type. Regression tests added (`test_score_stat_line_does_not_double_count_a_real_special_teams_td` and three siblings). **Same lesson as above: this was invisible to unit tests using narrow fixtures until a real player-week with a real special-teams TD exercised it.**
- **Sleeper's DST `sleeper_id` differs from nflverse's team code in exactly one of the 32 teams, confirmed by diffing both sources' full team lists: Sleeper uses `"LAR"` for the Rams; nflverse uses `"LA"`.** Every other code matches exactly (including `WAS`, `JAX`, `LAC`, which looked like plausible mismatch candidates but weren't). `scoring/golden.py`'s `resolve_player_ids` now applies `SLEEPER_TEAM_ALIASES = {"LAR": "LA"}` before falling back to the crosswalk. Before this fix, every Rams DST player-week showed up as an unresolved `synthetic_...` id with `computed=0.00 (no computed row)`.
- **`load_team_stats`' own `def_tds` and `fumble_recovery_tds` columns are both unreliable for `def_td` credit — confirmed live, in two separate rounds.** Round 1: `def_tds` was always 0 even for real defensive scores (e.g. ARI's and BAL's real week-2-2025 sack-fumble-return TDs), while `fumble_recovery_tds` *did* capture them — so it was added to `_DST_STAT_COLUMNS`. That immediately caused a regression: HOU week 15 2025 got wrongly credited, because `fumble_recovery_tds` also counts an *offensive* player recovering their *own* team's fumble and scoring (real case: RB W.Marks recovering QB C.Stroud's fumbled snap) — not a defensive event. Round 2: replaced both team_stats columns with a play-by-play derivation, `_defensive_return_tds` in `scoring/stats.py` — count plays where `return_touchdown == 1` and `td_team == defteam` (the scoring team was on defense that play). That introduced a *second* regression: it also counts punt/kickoff return touchdowns, which `return_touchdown` flags too but which Sleeper scores as special-teams credit (`def_st_td`/`st_td`), not `def_td` — confirmed live via NE's real week-4-2025 87-yard punt-return TD getting double-credited. Final fix: also filter to `play_type in ("pass", "run")`, i.e. scrimmage plays only. **Lesson: an aggregate column's name can imply more precision than it has (`def_tds` sounds authoritative but was simply wrong; `fumble_recovery_tds` sounds unambiguous but conflates offense and defense; `return_touchdown` sounds specific but spans two different Sleeper scoring categories) — each round only surfaced by running the real golden test again, never by re-reading documentation, because nflreadpy doesn't document these edge cases at the column level.**
- **Blocked kicks (`fg_blocked`/`pat_blocked`) were not being counted as a miss for the kicker's own scoring, and this explained nearly every remaining individual-player disagreement in one pass.** Once the DST-side bugs above were fixed, 7 of the last 9 individual-player disagreements turned out to be kickers (one, Jake Bates, appeared 3 times) — and *every single one* had `fg_missed`/`pat_missed == 0` but `fg_blocked`/`pat_blocked == 1`, with `computed` exactly 1.0 point higher than `sleeper` every time. Sleeper's `fgmiss`/`xpmiss` apparently penalise a block the same as an ordinary miss; the original mapping only read `fg_missed`/`pat_missed`. Fixed with a new `_sum_columns` derived-stat helper (`fgmiss` → `fg_missed + fg_blocked`, `xpmiss` → `pat_missed + pat_blocked`). **This is the clearest example this session of a pattern found by grouping disagreements by player position rather than staring at one case at a time — worth doing that grouping step earlier next time a batch of disagreements needs explaining.**
- **nflreadpy's `load_player_stats` carries IDP-style defensive columns (`def_sacks`, `def_interceptions`, `def_fumbles_forced`, `fumble_recovery_opp`, `def_safeties`) on *every individual player row*, not just defenders — an offensive player can show a non-zero value on a broken or trick play.** Sleeper's "Team defense" keys (`sack`/`int`/`ff`/`fum_rec`/`safe`) were originally plain `DirectStat`s reading those same column names, so they fired on any player row with a stray non-zero value, not just the DST's own row. Confirmed live via the two remaining unexplained individual-player disagreements: Trey Benson (RB) picked up a stray +2.0 from `fum_rec` (his own `fumble_recovery_opp` was 1, from some broken/trick play), and Sam Darnold (QB) a stray +1.0 from `ff`. Fixed by extending the same `position == "DST"` gate already built for the special-teams double-count bug — renamed `_team_special_teams_credit` to the more general `_dst_only` since it now covers two unrelated key groups, not just special teams. Regression tests added (`test_score_stat_line_credits_fum_rec_only_on_the_dst_row` and two siblings). Fully resolved all 3 remaining individual-player disagreements and, as a side effect, improved `bdff-chopped` too (99.92% → 99.96%). **Same root cause as the `load_team_stats`-includes-full-offense gotcha above, mirrored on the player side: nflreadpy's per-player and per-team tables are both wider than the single stat category their name suggests, and Sleeper's per-entity-type key semantics don't line up with nflreadpy's per-row column presence.**
- **`team_stats`' own `def_fumbles_forced`/`fumble_recovery_opp` columns aggregate scrimmage-play AND special-teams-play fumble events together, but Sleeper scores those as two separate key pairs (`ff`/`fum_rec` for scrimmage, `def_st_ff`/`def_st_fum_rec` for special teams) — the general "Team defense" mapping was already crediting special-teams events under the wrong key, so simply filling in the previously-zeroed special-teams keys would have double-counted them.** Confirmed live and manually verified by hand: PIT's real week-1-2025 kickoff (PIT forced NYJ's fumble and recovered it) was the *entire* source of PIT's `team_stats.def_fumbles_forced`/`fumble_recovery_opp` (both exactly 1, nothing else that game), meaning that one special-teams event alone explained PIT's whole existing `ff`+`fum_rec` credit (1+2=3 points) despite having zero genuine scrimmage-play forced-fumbles/recoveries. Fixed by deriving all four columns (`def_fumbles_forced`, `fumble_recovery_opp`, `special_teams_forced_fumbles`, `special_teams_fumble_recoveries`) from play-by-play's structured fumble fields (`forced_fumble_player_1_team`, `fumbled_1_team`, `fumble_recovery_1_team`), filtered by `special_teams_play`, rather than reading team_stats' combined columns at all (`scoring/stats.py`'s `_forced_fumbles`/`_opponent_fumble_recoveries`, replacing the earlier zero-fill). "Recovered" only counts as a turnover when `fumble_recovery_1_team != fumbled_1_team` -- a team recovering its own loose ball isn't a defensive event, confirmed via the same PIT game's *other* kickoff fumble (NYJ forced it, PIT recovered its own ball -- correctly zero credit to PIT). Closed the gap ADDENDUM-01 §D explicitly permitted leaving open; `rogan-radinator-league` cleared the 99% golden-test bar as a direct result (98.95% → 99.30%). **Third instance this session of the same lesson: a plausible-sounding aggregate column (here, "how many fumbles did this team force/recover") silently spans two categories Sleeper's scoring model treats as entirely separate, and the only way to find that was running the real comparison, then manually re-deriving one specific game's true point total by hand from the underlying play text before trusting the fix.**

---

## 5. Gotchas discovered

- **Path resolution bug, caught by actually running the live discovery.** `config.load_settings()` originally resolved `paths.data_root` (`"./data"`) relative to `settings.yml`'s own directory (`config/`), landing raw Sleeper payloads in `config/data/raw/` instead of the intended repo-root `data/raw/`. All the unit tests passed anyway, because the fixture settings.yml lived in a directory that also had a sibling `data/` — the bug was invisible until the real `config/settings.yml` (which lives one level under the repo root) exposed it. Fixed by resolving relative to `REPO_ROOT` explicitly (`config.py`'s `load_settings` now takes an explicit `root` parameter, decoupled from where the settings file itself sits). **Lesson: a fixture that happens to share the bug's blind spot will pass a test that a real config layout fails — this is exactly why ADDENDUM-02 pushed for testing against something closer to real usage before trusting green tests.**
- **`/players/nfl` is bigger than SPEC's estimate.** SPEC §6.1 says "~5MB"; the real payload today is 15.9MB. Not a problem, just don't be surprised.
- **Both real leagues already have a completed draft** (draft objects and draft picks were present during `cache warm`), even though the season hasn't started. Useful for task 0.14's replay-mode development later.
- **Sleeper's `search_rank` spans every player it has ever tracked, not just current fantasy-relevant ones.** Retired players (Tom Brady, Drew Brees, Antonio Brown) show up with real, low `search_rank` values, and IDP positions (LB/DL/DB/DT/CB) are ranked in the same pool as offense. Any future use of `search_rank` as a relevance proxy needs the same league-scoping + active-player filtering `ids/mapping.py` applies (see decision above), not raw rank alone.
- **Sleeper's `active` field means "not purged from Sleeper's system," not "on an NFL roster."** Cut/unsigned players can be `active: true` with `team: null` — and inconsistently, at least one real player (`sleeper_id 8008`, Josh Imatorbhebhe) has `team: null` in Sleeper's own payload but `team: "FA"` in the *dynastyprocess crosswalk's* own linked record for the same `sleeper_id`. `ids/mapping.py`'s `league_relevant()` requires both `active` and a non-null `team` before treating a player as relevant.
- **The dynastyprocess crosswalk can link a `sleeper_id` to a row with no `gsis_id` at all.** This isn't a matching-algorithm failure — nflverse itself never assigned that player a `gsis_id` (no NFL regular-season box score). No amount of better fuzzy matching fixes this; the only fix is `config/id_overrides.csv`. One instance found and overridden this session (`sleeper_id 8008`).
- **The real dynastyprocess crosswalk CSV (`db_playerids.csv`) uses the literal string `"NA"` for missing values**, not empty cells — `load_crosswalk_base` reads it with `null_values=["NA"]`. Team codes in the crosswalk can also differ from Sleeper's own (e.g. `LVR`/`KCC` vs Sleeper's `LV`/`KC`), which is part of why the exact `(name, position, team)` tier misses more than you'd expect and players fall through to the bare-name fuzzy tier.

---

## 6. Rebuilding `data/` from empty

Run in this order on an unrestricted network. See ADDENDUM-02 §B.

```bash
uv run ffapp ingest sleeper --season 2026 --discover --no-offline
uv run ffapp cache warm --season 2026 --all-leagues --no-offline
uv run ffapp cache status
```

Then warm the player-id crosswalk (task 0.3) — there's no `ffapp ingest nflverse` CLI command yet, only the underlying function; a real subcommand + `cache warm`/`cache status` wiring is a reasonable follow-up but wasn't required by 0.3's acceptance criteria:

```bash
uv run python -c "from ffapp.ingest import nflverse; from ffapp.config import load_settings; nflverse.fetch_player_ids(offline=False, settings=load_settings())"
uv run ffapp ids check --season 2026
```

`--seasons 2015-2025` historical nflverse warming doesn't exist yet — that lands with task 1.1.

Then warm the 2025-season nflverse data the golden test (0.5) needs — same caveat as above, no CLI subcommand yet, only the underlying functions:

```bash
uv run python -c "
from ffapp.ingest import nflverse
from ffapp.config import load_settings
settings = load_settings()
nflverse.fetch_player_stats(2025, offline=False, settings=settings)
nflverse.fetch_team_stats(2025, offline=False, settings=settings)
nflverse.fetch_schedules(2025, offline=False, settings=settings)
nflverse.fetch_pbp(2025, offline=False, settings=settings)
"
uv run ffapp scoring validate --all-leagues --no-offline
```

`fetch_pbp` is the slow one (~49k rows, 372 columns for one season) — expect it to take noticeably longer than the others.

**Not reproducible by re-running the above:**

- `data/raw/rankings/` — consensus projections and ADP are time-sensitive; re-pulling on a different day gives different numbers. These are committed to git as the exception in `CLAUDE.md`. Do not delete them. (Not yet populated — task 0.7.)
- Odds snapshots, if the paid API is enabled — same reasoning, but cheap to lose.

**Must be supplied by hand on a new machine:** `.env` (copy from `.env.example`; `FFAPP_OFFLINE=1` and `FFAPP_CACHE_STRICT=1` are sane defaults, both already used here).

---

## 7. Environment notes

**Python:** 3.13.5 (3.11+ satisfied), managed with `uv` 0.12.3. `uv.lock` is committed.
**Offline default:** `FFAPP_OFFLINE=1` in `.env`. Cache warming and league discovery need `--no-offline` explicitly — both commands refuse to run offline with a clear message rather than failing deep inside an HTTP call.
**Vendored wheels:** not currently needed. If PyPI becomes unreachable, see ADDENDUM-02 §F.

**Network reachability by location:**

| Location | Sleeper | nflverse / GitHub | PyPI | Rankings sites |
|---|---|---|---|---|
| This session (2026-08-11, hostname `Maybe`) | ✓ (live matchups pulled for both leagues' 2025 seasons) | ✓ (`nflreadpy` pulled real player_stats/team_stats/schedules/pbp for 2025) | ✓ | not tested |
| Home | ? | ? | ? | ? |
| Work | ✗ | ? | ? | ✗ |

> **Third session in a row now, all on hostname `Maybe`.** Treating this as the same machine going forward unless told otherwise — the "different machine" framing from earlier sessions hasn't been re-raised and the hostname keeps matching. If a genuinely different machine shows up, add it as a new row rather than overwriting this one.

---

## Maintenance rules

At the end of every session, before committing:

1. Update the header (date, machine, commit).
2. Move newly completed tasks into §2 **with their evidence**, and tick the matching box in `TASKS.md`.
3. Update §1 with the next task and its first concrete step.
4. Record anything half-finished in §3, precisely enough to resume cold.
5. Add any implementation decision to §4 and any gotcha to §5.
6. Keep §6 and §7 accurate as commands and reachability change.

Delete sections that have gone stale rather than letting them rot. A handoff file nobody trusts is worse than none.
