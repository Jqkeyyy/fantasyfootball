# SPEC Addendum 01 — multi-league support and verified league settings

**Date:** 2026-08-11
**Status:** supersedes the named sections of `SPEC.md`. Read after `SPEC.md`.

Two changes: the system must support several Sleeper leagues with different settings, and one league's scoring includes a setting the original keymap did not anticipate.

---

## A. Multi-league architecture

### A.1 Why this costs almost nothing

The original spec already forbids hardcoding league settings — everything routes through `LeagueFormat` and the league's `scoring_settings`. Supporting N leagues is therefore a config and namespacing change, not a redesign. The parts that genuinely differ per league are:

| Layer | Differs per league? | Why |
|---|---|---|
| Raw ingestion (nflverse, weather, odds) | No | League-agnostic NFL data |
| Feature engineering | No | Features are stats and context, not points |
| Scoring engine | Yes — parameterised | Already takes `scoring` as an argument |
| Model target | Yes | Target is league-scored points |
| Replacement level / VOR | Yes | Depends on roster slots and team count |
| Draft board | Yes | Downstream of the two above |
| All decision tools | Yes | Read that league's projections and rosters |

Ingestion and features — the expensive parts — are shared. Only the cheap tail is per-league.

### A.2 Config change (supersedes SPEC §5)

Replace the single `config/league.yml` with a directory:

```
config/
├── leagues/
│   ├── _primary.txt          # slug of the league used for model development
│   ├── main-ppr.yml
│   ├── league-b.yml
│   └── league-c.yml
└── settings.yml              # unchanged, league-agnostic
```

Each league file keeps the structure from SPEC §5 and adds:

```yaml
slug: "main-ppr"              # filename stem, used for namespacing
display_name: "Main league"
is_primary: true              # exactly one league must be primary
```

`ffapp ingest sleeper --season 2026 --discover` enumerates every league on the account (`GET /user/{user_id}/leagues/nfl/2026`) and writes a stub file per league with `league_cache` populated. You then set one `is_primary: true` by hand.

### A.3 CLI change (supersedes SPEC §16.2)

Every command that touches league-specific logic takes `--league <slug>`, defaulting to the primary. Add `--all-leagues` where it makes sense:

```
ffapp scoring validate    --league main-ppr | --all-leagues
ffapp train               --league main-ppr
ffapp project --week N    --league main-ppr | --all-leagues
ffapp draft board         --league league-b
ffapp startsit --week N   --league league-c
```

Commands that are league-agnostic (`ingest`, `features build`, `ids check`) take no `--league` flag. If a command needs the flag and none is given, it uses the primary and **logs which league it used** — silently defaulting is how you end up reading league B's start/sit advice for league A.

### A.4 Namespacing

```
data/models/<league_slug>/<position>/<model_version>/
data/outputs/<league_slug>/projections.parquet
data/outputs/<league_slug>/draft_board_2026.csv
data/outputs/<league_slug>/eval/<timestamp>/report.md
```

`data/features/` and `data/interim/` stay shared — they are league-agnostic by construction. If you ever find yourself wanting to namespace a feature table by league, something has gone wrong: it means league scoring has leaked into feature engineering.

### A.5 Model training strategy

**v1 (this season): train per league.** The target is league-scored points, so each league needs its own fit. This is cheap — LightGBM refits in minutes at this data size — and requires only that `train.py` takes the league as a parameter, which it already effectively does.

Practical approach: do all model *development* against the primary league only. Once the evaluation harness says the model beats baseline B2 there, run the identical pipeline for the other leagues. Do not develop against three leagues simultaneously; you will triple the evaluation surface and the noise, and you will not learn three times as much.

**v2 (offseason): one model, N leagues.** The decomposed pipeline in SPEC §11.4 predicts *stats*, then applies league scoring at Stage 4 recombination. That structure serves any number of leagues from a single training run. Multi-league is an additional argument for building v2 — but not for skipping v1.

### A.6 The seam that must exist now

One small requirement that prevents a rewrite later. The target column must never be precomputed and stored as a bare `fantasy_points`. It is always derived at training time:

```python
y = score_stat_line(stats, league.scoring_settings)
```

If a `fantasy_points` column ever appears in `features/player_week_features.parquet`, the seam is broken and the feature table is no longer league-agnostic. Add an assertion in `features/build.py` that rejects any column named `fantasy_points`, `points`, or `fpts`.

### A.7 Note on how different these leagues actually are

Full PPR with two W/R/T flex slots produces a materially different board from half PPR with one flex — replacement level at WR and TE moves several rounds' worth. Do not assume "slightly different settings" means "roughly the same rankings." Generate a board per league and compare them; the differences will be larger than expected, and seeing that is itself a good check that the VOR fixed point is working.

---

## B. Verified settings — league `main-ppr`

Transcribed from the Sleeper app. **These must be confirmed against the actual `GET /league/{league_id}` response before the keymap is finalised** — the app's display labels are not identical to the API's scoring keys.

### B.1 Roster positions

```
QB, RB, RB, WR, WR, TE, WRT, WRT, K, DEF
```

Ten starters. `WRT` is Sleeper's W/R/T flex (WR, RB, or TE eligible) and is most likely serialised as `FLEX` in `roster_positions` — verify. Bench and IR slot counts were not visible in the screenshot; read them from the API.

Resulting `LeagueFormat`:

```python
starters      = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
flex_slots    = {"FLEX": 2}
flex_eligible = {"FLEX": ["RB", "WR", "TE"]}
```

### B.2 Scoring (reconstructed key names — verify)

```yaml
scoring_settings:
  # Passing
  pass_yd: 0.04
  pass_td: 4
  pass_2pt: 2
  pass_int: -1

  # Rushing
  rush_yd: 0.1
  rush_td: 6
  rush_2pt: 2

  # Receiving — full PPR, no TE premium
  rec: 1
  rec_yd: 0.1
  rec_td: 6
  rec_2pt: 2

  # Kicking — PER-YARD, not distance buckets. See §C.
  fgm_yds: 0.1
  xpm: 1
  fgmiss: -1
  xpmiss: -1

  # Team defense
  def_td: 6
  pts_allow_0: 10
  pts_allow_1_6: 7
  pts_allow_7_13: 4
  pts_allow_14_20: 1
  # pts_allow_21_27 absent => 0
  pts_allow_28_34: -1
  pts_allow_35p: -4
  sack: 1
  int: 2
  fum_rec: 2
  safe: 2
  ff: 1
  blk_kick: 2

  # Special teams — team defense credit
  def_st_td: 6
  def_st_ff: 1
  def_st_fum_rec: 1

  # Special teams — individual player credit
  st_td: 6
  st_ff: 1
  st_fum_rec: 1

  # Misc
  fum_lost: -2
  fum_rec_td: 6
```

### B.3 What these settings imply

- **Full PPR with two flex slots** pushes replacement level deep at WR and makes pass-catching backs and receiving tight ends more valuable than a standard-format board would suggest. The §9.4 fixed point handles this automatically; do not adjust by hand.
- **No `bonus_rec_te`** — tight ends get no premium beyond flex eligibility.
- **QB is not premium**: 4-point passing TDs, 0.04/yard, and only −1 per interception. One QB slot, no superflex.
- **DST scoring has a wide range**: +10 for a shutout down to −4 for allowing 35+, with sacks, takeaways, and blocked kicks on top. That spread is large enough that streaming defences by matchup is worth real points. It raises the priority of the DST model (SPEC §11.6) above where the original spec placed it.
- **`fum` (any fumble) is absent** — only lost fumbles are penalised, at −2.

---

## C. Keymap correction — per-yard field goals

**This supersedes the kicking portion of SPEC §8.3.**

The original keymap assumed distance-bucketed field goal keys (`fgm_0_19`, `fgm_20_29`, …). This league instead uses `fgm_yds: 0.1` — **points per yard of each made field goal**. A 50-yarder is worth 5 points; a 25-yarder is worth 2.5.

Three consequences:

### C.1 The keymap must support both schemes

Some of your leagues may use buckets and some per-yard. `STAT_KEY_MAP` must handle both, and the engine must raise if a league somehow specifies both simultaneously (they would double-count).

### C.2 Bucketed aggregate stats cannot compute this score

This is the part that will silently break if it is not handled deliberately. Per-yard FG scoring requires **the actual distance of every made field goal**. If your kicker stats come from an aggregated weekly table that only carries counts per distance bucket, the exact score is not recoverable — you can only approximate it with bucket midpoints, and that approximation will disagree with Sleeper.

Therefore: **derive kicker scoring from play-by-play**, using the per-kick `kick_distance` field on made field goal plays, aggregated to player-week. Do not use bucketed FG counts for any league with `fgm_yds`.

Add this to the ingestion layer as an explicit kicker stat build step, and cover it in the §8.4 golden test — kickers are the most likely position to fail that test, and now you know why in advance.

### C.3 Kicker projection is slightly less pointless than the spec claimed

SPEC §11.7 says to spend no time on kickers. That advice stands, but soften it marginally: per-yard scoring rewards leg strength and long-range attempt volume, which are somewhat more persistent than make/miss luck. A kicker on a team that stalls in the 30s is worth measurably more here than in a bucket-scoring league. Still a last-round pick; still not worth model effort beyond implied team total, dome status, and attempted-distance history.

---

## D. Special teams scoring — data availability

The league scores special teams events at both the team-defense level (`def_st_*`) and the individual player level (`st_*`). Return touchdowns are available in nflverse player stats. Special-teams forced fumbles and fumble recoveries attributed to individual players are lower-frequency and must be derived from play-by-play on kicking plays.

These are small point contributions and rare events. Do not over-invest, but do implement them — the §8.4 golden test compares against Sleeper's own totals, and an unimplemented scoring key will show up as a systematic disagreement on exactly the player-weeks where a return touchdown happened.

---

## E. Task list changes

Amendments to `TASKS.md`:

- **0.2** — add `--discover` to enumerate all leagues on the account and write a stub config per league. Set `is_primary` by hand afterwards.
- **0.4** — keymap must cover both `fgm_yds` and bucketed FG keys, and must raise if both are present. Add the special-teams keys from §B.2.
- **0.4a (new, ⏱ 2h)** — kicker stat build from play-by-play `kick_distance`. Blocks 0.5 for any league using `fgm_yds`.
- **0.5** — run the golden test per league (`--all-leagues`). Expect kickers to be the first thing that fails; §C.2 is the reason.
- **0.12 / 0.13** — draft board and UI take `--league`; the Streamlit page gets a league selector. Generate a board per league and compare them (§A.7).
- **1.9** — add the assertion from §A.6 rejecting any precomputed points column in the feature table.
- **New before 1.15** — decide the primary league; all model development happens there only.

---

## F. Open decisions — updated

| # | Decision | Status |
|---|---|---|
| 1 | League format | **Resolved for `main-ppr`** (§B.1, §B.2), pending API verification. Other leagues still to capture via `--discover` |
| 2 | Ranking sources | Still open — blocks task 0.7 |
| 5 | Draft slot and date | Still open, per league — blocks task 0.11 |
| 6 | Waiver type and budget | Still open — read from API per league |
| 9 (new) | Which league is primary for model development? | Must be set before task 1.15 |
| 10 (new) | Do the other leagues use bucketed or per-yard FG scoring? | Affects §C.1; read from API |
