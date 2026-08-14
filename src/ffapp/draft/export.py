"""Static HTML draft-board export (SPEC-ADDENDUM-03.md §D; task 0.16).

Draft day has a clock, and the live Streamlit app needs a laptop and a
network. This is the fallback that cannot fail: a single self-contained
HTML file -- board data and every script inline, no CDN, no external
fonts, no network call of any kind -- that opens from a phone's Files app
with WiFi and cellular both off.

No new valuation logic lives here. `draft.board.build_draft_board` remains
the one source of truth for the board itself; this module only bundles it
with draft-day context (pick numbers, input staleness) and renders it.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.cache import offline as offline_cache
from ffapp.config import LeagueConfig, Settings
from ffapp.draft import board as draft_board
from ffapp.ingest import rankings
from ffapp.league_format import parse_league_format

# Raw-fetch-only wrappers for the sources `draft.board` already aggregates --
# used here only to recover each source's own cached Path, so its sidecar
# (fetched_at_utc) can be read for the export header. Mirrors
# `board._fetch_point_sources`'s graceful degradation: a source that can't
# be fetched just doesn't get an age row, it doesn't block the export.
_RANKINGS_SOURCE_FETCHERS: dict[str, Callable[[int, int, bool | None, Settings], Path]] = {
    "espn": lambda season, teams, offline, settings: rankings.fetch_espn(
        season, offline=offline, settings=settings
    ),
    "fantasysharks": lambda season, teams, offline, settings: rankings.fetch_fantasysharks(
        offline=offline, settings=settings
    ),
    "cbs": lambda season, teams, offline, settings: rankings.fetch_cbs(
        season, offline=offline, settings=settings
    ),
    "fftoday": lambda season, teams, offline, settings: rankings.fetch_fftoday(
        season, offline=offline, settings=settings
    ),
    "fantasypros": lambda season, teams, offline, settings: rankings.fetch_fantasypros(
        offline=offline, settings=settings
    ),
}

# SPEC-ADDENDUM-02.md §C.3: rankings/ADP are "fresh" under 24h in preseason.
# Not every source above has its own entry in config/settings.yml's
# staleness_hours (only rankings_adp does today) -- the export header states
# this one documented threshold rather than silently using an undocumented
# per-source default.
RANKINGS_STALENESS_HOURS = 24.0


@dataclass(frozen=True)
class InputAge:
    name: str
    fetched_at_utc: str | None
    age_hours: float | None


@dataclass(frozen=True)
class ExportBundle:
    board: pl.DataFrame
    generated_at_utc: str
    my_slot: int | None
    my_picks: list[int]
    adp_age: InputAge
    rankings_ages: list[InputAge]


def _input_age(name: str, path: Path) -> InputAge:
    meta = offline_cache.read_sidecar(path)
    if meta is None:
        return InputAge(name=name, fetched_at_utc=None, age_hours=None)
    fetched_at = meta["fetched_at_utc"]
    return InputAge(
        name=name, fetched_at_utc=fetched_at, age_hours=offline_cache.age_hours(fetched_at)
    )


def _collect_rankings_ages(
    season: int, teams: int, *, offline: bool | None, settings: Settings
) -> list[InputAge]:
    ages = []
    for name, fetch in _RANKINGS_SOURCE_FETCHERS.items():
        try:
            path = fetch(season, teams, offline, settings)
        except Exception:
            continue
        ages.append(_input_age(name, path))
    return ages


def build_export_bundle(
    league: LeagueConfig, settings: Settings, *, season: int, offline: bool | None = None
) -> ExportBundle:
    """Assemble everything SPEC-ADDENDUM-03.md §D's export needs: the board
    itself, the drafter's slot and pick numbers, and the staleness of the
    ADP and rankings inputs that produced it.
    """
    board_df = draft_board.build_draft_board(league, settings, season=season, offline=offline)
    league_format = parse_league_format(league)
    pick_context = draft_board.resolve_pick_context(
        league, settings, season=season, offline=offline
    )

    adp_path = rankings.fetch_adp(
        season, teams=league_format.n_teams, offline=offline, settings=settings
    )
    adp_age = _input_age("adp", adp_path)
    rankings_ages = _collect_rankings_ages(
        season, league_format.n_teams, offline=offline, settings=settings
    )

    return ExportBundle(
        board=board_df,
        generated_at_utc=datetime.now(UTC).isoformat(),
        my_slot=pick_context.my_slot,
        my_picks=pick_context.my_picks,
        adp_age=adp_age,
        rankings_ages=rankings_ages,
    )


def export_html_path(settings: Settings, *, season: int) -> Path:
    return settings.data_root / "outputs" / f"draft_board_{season}_export.html"


def export_csv_path(settings: Settings, *, season: int) -> Path:
    return settings.data_root / "outputs" / f"draft_board_{season}_export.csv"


def _format_age(age: InputAge) -> str:
    if age.age_hours is None:
        return f"{age.name}: unknown (not cached)"
    flag = " [STALE]" if age.age_hours > RANKINGS_STALENESS_HOURS else ""
    return f"{age.name}: {age.age_hours:.1f}h old{flag}"


_TABLE_COLUMNS = [
    ("overall_rank", "Rank"),
    ("tier", "Tier"),
    ("player", "Player"),
    ("position", "Pos"),
    ("team", "Team"),
    ("bye_week", "Bye"),
    ("vor", "VOR"),
    ("proj_points_adj", "Proj"),
    ("proj_ppg", "PPG"),
    ("adp", "ADP"),
    ("p_avail_next", "Avail@Next"),
]


def render_html(bundle: ExportBundle, *, league: LeagueConfig) -> str:
    """A single self-contained HTML string: board data as inline JSON, all
    CSS/JS inline, no CDN links, no external fonts, no network calls of any
    kind (SPEC-ADDENDUM-03.md §D).
    """
    rows = bundle.board.to_dicts()
    # `</` inside embedded JSON could otherwise be read by the HTML parser
    # as closing the <script> tag early -- escape it the standard way.
    board_json = json.dumps(rows, default=str).replace("</", "<\\/")

    rankings_summary = (
        " &middot; ".join(html.escape(_format_age(age)) for age in bundle.rankings_ages)
        if bundle.rankings_ages
        else "no rankings sources cached"
    )
    slot_text = f"Slot {bundle.my_slot}" if bundle.my_slot is not None else "slot unknown"
    picks_text = (
        ", ".join(str(p) for p in bundle.my_picks[:6]) if bundle.my_picks else "none resolved"
    )
    header_cells = "".join(f"<th>{label}</th>" for _, label in _TABLE_COLUMNS)
    positions = sorted({row["position"] for row in rows if row.get("position")})
    position_buttons = "".join(
        f'<button class="pos-btn" data-pos="{html.escape(pos)}">{html.escape(pos)}</button>'
        for pos in positions
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(league.display_name)} draft board (offline export)</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #111827; --muted: #6b7280; --border: #e5e7eb;
    --accent: #2563eb; --accent-fg: #ffffff; --tier-line: #f59e0b; --stale: #b91c1c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1rem; background: var(--bg); color: var(--fg);
    font: 16px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  }}
  h1 {{ font-size: 1.25rem; margin: 0 0 0.25rem; }}
  .meta, .staleness {{ color: var(--muted); font-size: 0.85rem; margin: 0.15rem 0; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0; }}
  #search {{
    flex: 1 1 100%; padding: 0.6rem; font-size: 1rem; border: 1px solid var(--border);
    border-radius: 6px;
  }}
  .pos-filter {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
  .pos-btn {{
    padding: 0.6rem 0.9rem; font-size: 1rem; border: 1px solid var(--border);
    border-radius: 999px; background: var(--bg); color: var(--fg); cursor: pointer;
  }}
  .pos-btn.active {{
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent);
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{
    text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  thead {{ position: sticky; top: 0; background: var(--bg); }}
  tr.tier-break td {{ border-top: 3px solid var(--tier-line); }}
  .table-wrap {{ overflow-x: auto; }}
  .count {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(league.display_name)} &mdash; draft board (offline export)</h1>
  <p class="meta">
    Generated {html.escape(bundle.generated_at_utc)} &middot; {html.escape(slot_text)}
    &middot; next picks: {html.escape(picks_text)}
  </p>
  <p class="staleness">
    ADP: {html.escape(_format_age(bundle.adp_age))} &middot; Rankings &mdash; {rankings_summary}
  </p>
</header>
<div class="controls">
  <input id="search" type="text" placeholder="Search player...">
  <div class="pos-filter">
    <button class="pos-btn active" data-pos="ALL">ALL</button>
    {position_buttons}
  </div>
</div>
<div class="table-wrap">
  <table id="board">
    <thead><tr>{header_cells}</tr></thead>
    <tbody id="board-body"></tbody>
  </table>
</div>
<p class="count" id="count"></p>
<script id="board-data" type="application/json">{board_json}</script>
<script>
(function () {{
  var data = JSON.parse(document.getElementById('board-data').textContent);
  var columns = {json.dumps([key for key, _ in _TABLE_COLUMNS])};
  var tbody = document.getElementById('board-body');
  var search = document.getElementById('search');
  var countEl = document.getElementById('count');
  var buttons = document.querySelectorAll('.pos-btn');
  var activePos = 'ALL';

  function esc(value) {{
    var d = document.createElement('div');
    d.textContent = value === null || value === undefined ? '' : String(value);
    return d.innerHTML;
  }}

  function fmt(key, value) {{
    if (value === null || value === undefined) return '--';
    if (key === 'p_avail_next') return Math.round(value * 100) + '%';
    if (typeof value === 'number') {{
      return Number.isInteger(value) ? String(value) : value.toFixed(1);
    }}
    return esc(value);
  }}

  function render() {{
    var q = search.value.trim().toLowerCase();
    var lastTier = null;
    var shown = 0;
    tbody.innerHTML = '';
    data.forEach(function (row) {{
      if (activePos !== 'ALL' && row.position !== activePos) return;
      if (q && String(row.player || '').toLowerCase().indexOf(q) === -1) return;
      shown += 1;
      var tr = document.createElement('tr');
      if (lastTier !== null && row.tier !== lastTier) tr.className = 'tier-break';
      lastTier = row.tier;
      tr.innerHTML = columns.map(function (key) {{
        return '<td>' + fmt(key, row[key]) + '</td>';
      }}).join('');
      tbody.appendChild(tr);
    }});
    countEl.textContent = shown + ' of ' + data.length + ' players shown';
  }}

  search.addEventListener('input', render);
  buttons.forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      buttons.forEach(function (b) {{ b.classList.remove('active'); }});
      btn.classList.add('active');
      activePos = btn.getAttribute('data-pos');
      render();
    }});
  }});

  render();
}})();
</script>
</body>
</html>
"""


__all__ = [
    "ExportBundle",
    "InputAge",
    "RANKINGS_STALENESS_HOURS",
    "build_export_bundle",
    "export_csv_path",
    "export_html_path",
    "render_html",
]
