# SPEC Addendum 03 — phone access and draft-day operations

**Date:** 2026-08-13
**Status:** extends `SPEC.md` §15 (UI) and §16 (operations). Scoped to the 2026 draft, Aug 22.

Goal: read the draft board and the live draft assistant from a phone during the draft. Not a hosting project — the app stays local; only access changes.

---

## A. The constraint that shapes everything

Draft day has a clock. A tool that works 95% of the time is worse than a worse tool that works 100% of the time, because the 5% arrives at pick 3 with 60 seconds left.

Therefore build **two layers**:

1. **Live layer** — the real Streamlit app, reachable from the phone, with the draft assistant polling picks. This is the valuable one.
2. **Static fallback** — a single self-contained HTML file of the board, on the phone already, needing no laptop, no network, no server. This one cannot fail.

The static fallback is not optional. It is the thing that guarantees you can draft.

---

## B. Serving the live app

### B.1 Recommended: Tailscale

A private mesh network between laptop and phone. Free for personal use, works over cellular, no public URL, no port forwarding, no firewall rules.

Setup (once, well before draft day):

1. Install Tailscale on the Windows machine and on the phone; sign in to the same account on both.
2. Note the laptop's Tailscale name or `100.x.y.z` address (`tailscale ip -4`).
3. Start the app bound to all interfaces:

```powershell
uv run streamlit run src/ffapp/app/streamlit_app.py `
  --server.address 0.0.0.0 `
  --server.port 8501 `
  --server.headless true
```

4. On the phone, open `http://<tailscale-name>:8501` or `http://100.x.y.z:8501`.

**Security note:** `0.0.0.0` binds to every interface, including whatever LAN the laptop is on. On a home network that is fine. Do not run it bound this way on the work network. To bind only to Tailscale, pass the `100.x.y.z` address as `--server.address` instead.

### B.2 Simpler alternative: plain LAN

If drafting from home on the same WiFi, skip Tailscale entirely. Same `streamlit run` command, then browse to `http://<laptop-lan-ip>:8501` from the phone. Windows Firewall will prompt on first run — allow on **Private** networks only. If no prompt appears:

```powershell
# Administrator
New-NetFirewallRule -DisplayName "Streamlit 8501" -Direction Inbound `
  -LocalPort 8501 -Protocol TCP -Action Allow -Profile Private
```

Downside: breaks the moment you switch to cellular or draft from anywhere else.

### B.3 Not recommended for this deadline

- **Streamlit Community Cloud** — requires the app's data in the repo. `data/` is gigabytes and gitignored, and the app reads cached parquet at runtime. Making this work is a re-architecture, not a deploy. Not in nine days.
- **Public tunnels (ngrok, cloudflared quick tunnels)** — workable, but they put a public URL in front of your league data and the free tiers rotate URLs and rate-limit. Tailscale is strictly better here.

### B.4 Keep the laptop awake

Windows sleeping mid-draft kills the server. Plug in and disable sleep on AC:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

Restore afterwards with sensible values.

---

## C. Mobile UI (new task 0.15)

Streamlit's default dataframe is unusable on a ~380px screen, and the draft board has roughly nineteen columns. Do not try to make the desktop table responsive. Build a separate page.

**New page:** `src/ffapp/app/pages/5_Draft_Mobile.py`

**Design rules:**

- **Cards, not tables.** One card per player. Name, position, team, bye on line one; tier and VOR on line two; one short "why" line (e.g. `Tier 4 · 3 left · falls to you 71%`).
- **Show 20–30, not 300.** Best available only, filtered to the current pool. Scrolling past a hundred players on a phone during a 60-second clock does not happen.
- **Three numbers above the fold**, in this order: best available by VOR, **tier depth remaining per position**, and survival probability to your next pick. Tier depth is the number that actually drives on-the-clock decisions.
- **Position filter as big tap targets**, not a dropdown. `ALL / QB / RB / WR / TE / K / DST` as a horizontal button row.
- **Auto-refresh** every 10 seconds during a live draft so picks land without a manual reload. Show a visible "last updated" timestamp — a silently stale board is worse than an obviously stale one.
- **No hover, no tooltips, no side-by-side comparison.** None of it works on touch.
- Minimum 16px body text; larger for the player name.

**Explicitly out of scope:** editing, trade tools, charts, the SOS heatmap. Draft day needs one screen that answers "who do I take."

---

## D. Static fallback export (new task 0.16)

**New command:** `ffapp draft export --league <slug> [--out <path>]`

Writes a single self-contained `.html` file — board data embedded as inline JSON, all CSS and JS inline, no CDN, no network calls, no server. Opens from the phone's Files app offline.

Contents:

- The full board, sorted by VOR, with tier breaks visible.
- Client-side position filter and text search (a few dozen lines of vanilla JS).
- Header stating the generation timestamp and **the age of the ADP and rankings inputs** — per `ADDENDUM-02 §C.3`, a board built on stale ADP is quietly wrong, and the fallback must say so on its face.
- Your draft slot and computed pick numbers.

Generate the morning of the draft, immediately after refreshing rankings, and get it onto the phone (email, Drive, AirDrop — anything that leaves a local copy).

Also export the same board as CSV to a phone-accessible spreadsheet as a second fallback. Redundancy here is cheap.

---

## E. Draft-day runbook

Add to `docs/` and follow it literally.

**T-7 days**
- Fix or replace the two dead ranking sources (ESPN 0 rows, FFToday 403). Board is on 3 of 5.
- Install and test Tailscale end to end: phone loads the app over **cellular**, with WiFi off, to prove it is not silently working via LAN.
- Build tasks 0.15 and 0.16.

**T-1 day**
- `uv run ffapp cache warm --season 2026 --all-leagues --no-offline`
- Full dry run of the live assistant in `--replay` mode against a recorded draft, viewed on the phone.
- Confirm laptop power settings.

**Morning of**
- `uv run ffapp ingest rankings --no-offline` (ADP staleness policy is 24h and **will** hard-fail under `FFAPP_CACHE_STRICT=1` — this is by design)
- `uv run ffapp draft board --league <slug>`
- `uv run ffapp draft export --league <slug>` → move the HTML to the phone
- Start Streamlit; confirm phone access before the draft opens
- Leave the terminal window open and untouched

**Do not** set `FFAPP_CACHE_STRICT=0` to silence a staleness error. That guardrail exists to stop you drafting off day-old ADP without noticing. Refresh instead.

---

## F. Task list additions

- **0.15 (new, ⏱ 3h)** — mobile draft page (§C). Done when the page is usable one-handed on a real phone, shows tier depth remaining, and auto-refreshes during a replayed draft.
- **0.16 (new, ⏱ 2h)** — static HTML export (§D). Done when the file opens on the phone with WiFi and cellular disabled, filters work, and the input-age header is present and accurate.

Both are draft-blocking and outrank everything in Phase 3.
