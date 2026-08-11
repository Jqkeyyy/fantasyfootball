# SPEC Addendum 02 — offline-first development

**Date:** 2026-08-11
**Status:** extends `SPEC.md` §6.3 and §16.2. Read after Addendum 01.

Development happens partly on a network where Sleeper and several data sources are unreachable. This addendum makes the system work offline by default and network-dependent only at explicit, batched moments.

This is not a workaround. Every rule below is something the project should do anyway — reproducibility, deterministic tests, and archived raw payloads are all served by it.

---

## A. Principle

Network access exists at exactly one layer: `ingest/*.fetch()`. Everything else — normalisation, features, models, evaluation, tools, UI — reads from `data/raw/` and never opens a socket.

`SPEC.md` §6.3 already mandates this split ("No transformation logic in `fetch`. No network calls in `normalise`"). Addendum 02 adds the enforcement and the cache layer that makes the split useful.

**Enforcement:** add `tests/test_no_network.py` which imports every module outside `ingest/` with sockets monkeypatched to raise, and executes a representative call path from each. Any accidental network dependency outside the ingest layer fails the suite.

---

## B. Cache warming

Run these on an unrestricted network (home, or a phone hotspot). They pull everything the offline workflow needs and archive raw payloads per `SPEC.md` §6.3.

```
ffapp cache warm --season 2026 --all-leagues
ffapp cache warm --seasons 2015-2025          # historical, one time, large
ffapp cache status
ffapp cache verify --for-task 0.7
```

### `cache warm`

Pulls and archives, in order:

| Source | What | Size | Re-warm cadence |
|---|---|---|---|
| Sleeper league objects | all leagues on the account | KB | Weekly — settings rarely change |
| Sleeper rosters, users | per league | KB | Daily in season |
| Sleeper matchups | every completed week, all seasons the leagues existed | MB | Once per completed week — needed for the §8.4 golden test |
| Sleeper transactions | per league per week | MB | Weekly |
| Sleeper `/players/nfl` | full dictionary | ~5MB | Daily max, per SPEC §6.1 |
| Sleeper drafts + picks | per league | KB | After each draft |
| ffverse player-id crosswalk | ID mapping base | MB | Weekly |
| nflverse | PBP, stats, snaps, depth charts, rosters, injuries, schedules | GB for 2015–2026 | Historical: once. Current season: after game days |
| ffopportunity | expected fantasy points releases | MB | With nflverse |
| Consensus rankings + ADP | per source | MB | **Daily in preseason** — see §C.3 |
| Odds | if enabled | KB | Weekly |
| Weather | historical archive for training seasons; forecast for current week | MB | Forecast: before each week |

### `cache status`

Prints every cached artefact with its `fetched_at_utc`, age, and a staleness verdict against the policy in §C.3. Run this first thing every session — a stale ADP file in August is the failure most likely to matter and least likely to announce itself.

### `cache verify --for-task <id>`

Checks whether the cache can satisfy a given task's data needs without network. Prints exactly which artefacts are missing and the `cache warm` invocation that would fetch them. Use before leaving an unrestricted network so you discover gaps at home rather than at work.

---

## C. Offline mode

### C.1 Activation

Set `FFAPP_OFFLINE=1` in `.env`, or pass `--offline` to any command. Default it to `1` in `.env.example` so the safe mode is the default and network access is the deliberate act.

### C.2 Semantics — fail loud, never silent

In offline mode, `fetch()` reads only from `data/raw/`. On a miss it raises:

```python
class OfflineCacheMiss(Exception):
    """Required data is not cached and the network is disabled."""
```

The message must name the source, the exact artefact, and the command that would fetch it:

```
OfflineCacheMiss: sleeper/matchups season=2025 week=14 not cached.
  Run on an unrestricted network:
    ffapp cache warm --season 2025 --weeks 14 --league main-ppr
```

**This is the single most important rule in this addendum.** An offline mode that returns an empty dataframe on a miss produces a model trained on partial data, a draft board missing forty players, and a golden test that passes because it compared zero rows. All three look fine. Never return empty; always raise.

Add `tests/test_offline_raises.py` asserting that a cache miss under `FFAPP_OFFLINE=1` raises rather than returning an empty frame.

### C.3 Staleness policy

`cache status` classifies each artefact:

| Source | Fresh | Stale | Never stale |
|---|---|---|---|
| nflverse, seasons ≤ 2025 | — | — | ✓ completed seasons are immutable |
| nflverse, current season | < 24h since last game day | otherwise | |
| Sleeper league settings | < 7d | otherwise | |
| Sleeper rosters | < 24h in season | otherwise | |
| Sleeper matchups, completed weeks | — | — | ✓ immutable once final |
| **Rankings / ADP** | **< 24h in preseason** | otherwise | |
| Odds | < 12h | otherwise | |
| Weather forecast | < 6h | otherwise | |
| Weather historical archive | — | — | ✓ |

Stale data is usable — offline mode does not refuse to run on it — but every command that consumes a stale artefact logs a warning naming the artefact and its age, and any generated draft board or projection file records the age of its inputs in its provenance metadata.

Rankings and ADP are the ones to watch. They move daily in August, and a board built from a five-day-old ADP snapshot will give you survival probabilities that are quietly wrong.

---

## D. Fixtures

`SPEC.md` §16.3 already requires tests to use committed fixtures and never touch the network. Generate them from real cached payloads:

```
ffapp cache fixtures --out tests/fixtures/
```

Extracts a small, representative slice — one league, three weeks, ~50 players, covering every scoring key in use including the special-teams and per-yard-FG cases from Addendum 01 §B.2 and §C — and writes it to `tests/fixtures/`. These are committed, so the full test suite runs on any machine with no network and no `data/` directory at all.

Regenerate whenever a source schema changes. Keep them small; they are read constantly and committed forever.

---

## E. Task classification by network dependency

The practical payoff. Tasks fall into three groups.

### Group 1 — pure logic, fully offline, no cache required

These need only a scoring settings dict and a `roster_positions` list, both of which you already have from Addendum 01 §B. Save them as `tests/fixtures/league_main_ppr.json` and this group runs anywhere.

| Task | What |
|---|---|
| **0.4** | Scoring keymap and engine |
| **0.4a** | Kicker stat build from PBP (logic; needs a small PBP fixture) |
| **0.6** | LeagueFormat parser |
| **0.9** | Value over replacement, fixed-point replacement level |
| **0.10** | Tiers |
| **1.5** | Feature registry and as_of contract |
| **1.10** | Baselines |
| **1.11** | Snapshot and leakage test |
| **1.13** | Metrics module |
| **2.1** | Lineup optimiser (ILP) |
| **2.2** | Correlated weekly simulation |

That list is most of the algorithmic core of the project. The scoring engine, the VOR fixed point, the lineup ILP, and the copula simulation are the four hardest pieces of logic in the build, and none of them need a network.

### Group 2 — cache-then-offline

Warm once at home, then develop offline indefinitely. Historical nflverse data is immutable, so a single large pull covers all model work for the season.

| Task | Cache needed |
|---|---|
| 0.3 | ffverse crosswalk, Sleeper players dict |
| 0.5 | Sleeper matchups for completed weeks |
| 0.7, 0.8, 0.11, 0.12 | Rankings, ADP, historical stats |
| 1.1–1.4, 1.6–1.9 | nflverse 2015–2026, ffopportunity, weather archive |
| 1.12, 1.14–1.19 | Everything above |
| 2.3–2.9 | Everything above |

### Group 3 — genuinely needs live network

| Task | Why | Mitigation |
|---|---|---|
| 0.2 | First-time league discovery | One-time, ~2 minutes at home |
| 0.14 | Live draft polling | Develop against a recorded pick sequence; see below |
| 2.10 | News ingestion | Cache RSS snapshots; the LLM call also needs network |
| In-season weekly runs | Fresh injuries, odds, inactives | Warm the cache Tue/Thu/Sun per SPEC §16.1 |

**Live draft assistant (0.14) offline development:** `ffapp cache warm` archives a completed draft's full pick list. Add a replay mode — `ffapp draft live --replay <draft_id> --speed 10` — that feeds those picks to the assistant on a timer as if they were arriving live. This is a better development loop than the real thing regardless of network, because it is deterministic and repeatable. Build it as the primary path and treat live polling as a thin adapter over the same interface.

### Suggested sequencing

1. At home: `ffapp cache warm --season 2026 --all-leagues` plus `--seasons 2015-2025`. Budget a few GB and some patience for the historical pull.
2. At home: task 0.2 (league discovery) and 0.1 (`uv sync`, which needs PyPI).
3. Anywhere: 0.4, 0.4a, 0.6, then 0.9 and 0.10. The algorithmic core.
4. Offline against cache: 0.3, 0.5, 0.7, 0.8, 0.11, 0.12.
5. At home, before the draft: re-warm rankings and ADP, regenerate the board.

---

## F. Python environment portability

`uv sync` needs PyPI, which may also be unreachable.

- Commit `uv.lock`. Non-negotiable, and already required by `CLAUDE.md`.
- If PyPI is reachable, `uv` caches wheels locally; a warmed cache survives losing network.
- If PyPI is blocked, vendor wheels at home: `uv pip download -r requirements.txt -d vendor/wheels`, then install offline with `uv pip install --no-index --find-links vendor/wheels -r requirements.txt`. Add `vendor/` to `.gitignore` and carry it on the same drive as `data/`.
- Do not add dependencies while offline. Note the need, add it at home, re-warm.

---

## G. Config additions

`.env.example`:

```
FFAPP_OFFLINE=1
FFAPP_CACHE_STRICT=1     # raise on stale cache instead of warning
ODDS_API_KEY=
ANTHROPIC_API_KEY=
```

`config/settings.yml`:

```yaml
cache:
  root: "./data/raw"
  offline_default: true
  staleness_hours:
    sleeper_league: 168
    sleeper_rosters: 24
    rankings_adp: 24
    odds: 12
    weather_forecast: 6
  warn_on_stale: true
```

---

## H. Task list changes

- **0.1** — add `uv.lock` commit and the vendored-wheels fallback (§F).
- **0.2** — add `ffapp cache warm`, `cache status`, `cache verify`, and `OfflineCacheMiss` with actionable messages. Add `tests/test_offline_raises.py`. This grows 0.2 by roughly 3 hours and is worth every minute.
- **0.2a (new, ⏱ 1h)** — `ffapp cache fixtures` and the committed fixture set (§D).
- **0.14** — implement `--replay` against a recorded draft as the primary development path; live polling is an adapter.
- **New, anywhere in Phase 1** — `tests/test_no_network.py` (§A).

---

## I. Open decisions — updated

| # | Decision | Status |
|---|---|---|
| 11 (new) | Are PyPI and GitHub also blocked, or only Sleeper and ranking sites? | Determines whether §F vendoring is needed. Test with `pip download --dry-run requests` and a raw.githubusercontent.com fetch |
| 12 (new) | Is there enough local disk for the full 2015–2026 nflverse cache (several GB)? | If not, narrow `seasons.train_start` to 2018 and re-evaluate later |
