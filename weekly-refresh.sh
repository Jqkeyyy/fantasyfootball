#!/usr/bin/env bash
# Weekly in-season refresh, driven by host cron via `docker compose exec`.
# Usage: weekly-refresh.sh [tuesday|thursday|sunday]
#
# Tuesday   : rebuild feature pipeline, backfill last week's actuals,
#             warm caches, refresh rankings, project week N, build
#             rest-of-season projections + ROS rankings, log snapshot.
# Thursday  : refresh rankings, project week N, log snapshot.
# Sunday    : refresh rankings, project week N, log snapshot.
#
# Current NFL week is resolved live from Sleeper's public state endpoint
# (https://api.sleeper.app/v1/state/nfl) via `requests`, already a project
# dependency -- ffapp's own CLI does not resolve this itself (--week is a
# required option everywhere), so this replaces get-current-week.ps1's role
# without a PowerShell dependency. Every step logs to data/outputs/logs/.
set -uo pipefail

cd "$(dirname "$0")"

RUN_LABEL="${1:-}"
if [ -z "$RUN_LABEL" ]; then
    echo "Usage: weekly-refresh.sh [tuesday|thursday|sunday]"
    exit 1
fi

SEASON=2026
LEAGUES="rogan-radinator-league bdff-chopped"

LOG_DIR="data/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/weekly_${RUN_LABEL}_$(date +%Y%m%d_%H%M).log"

{
    echo "============================================"
    echo " Weekly refresh: $RUN_LABEL ($(date))"
    echo "============================================"
} > "$LOG_FILE"

WEEK=$(uv run python -c "import requests; print(requests.get('https://api.sleeper.app/v1/state/nfl', timeout=10).json()['week'])" 2>>"$LOG_FILE")
if [ -z "$WEEK" ]; then
    echo "FAILED: could not resolve current NFL week from Sleeper's state endpoint" >> "$LOG_FILE"
    exit 1
fi
echo "Current NFL week: $WEEK" >> "$LOG_FILE"
PREV_WEEK=$((WEEK - 1))

if [ "$RUN_LABEL" = "tuesday" ]; then
    echo "" >> "$LOG_FILE"
    echo "[Tuesday] rebuilding feature pipeline..." >> "$LOG_FILE"
    if ! uv run python notebooks/build_features_pipeline.py >> "$LOG_FILE" 2>&1; then
        echo "  WARNING: feature pipeline rebuild failed, continuing anyway" >> "$LOG_FILE"
    fi

    echo "" >> "$LOG_FILE"
    echo "[Tuesday] warming Sleeper cache for all leagues..." >> "$LOG_FILE"
    uv run ffapp cache warm --season "$SEASON" --all-leagues --no-offline >> "$LOG_FILE" 2>&1
fi

for LEAGUE in $LEAGUES; do
    echo "" >> "$LOG_FILE"
    echo "[$RUN_LABEL] $LEAGUE: refreshing rankings/ADP..." >> "$LOG_FILE"
    uv run ffapp ingest rankings --league "$LEAGUE" --season "$SEASON" --no-offline >> "$LOG_FILE" 2>&1

    echo "[$RUN_LABEL] $LEAGUE: projecting week $WEEK..." >> "$LOG_FILE"
    uv run ffapp project --week "$WEEK" --season "$SEASON" --league "$LEAGUE" --no-offline >> "$LOG_FILE" 2>&1

    if [ "$RUN_LABEL" = "tuesday" ]; then
        if [ "$PREV_WEEK" -ge 1 ]; then
            echo "[Tuesday] $LEAGUE: backfilling week $PREV_WEEK actuals..." >> "$LOG_FILE"
            uv run ffapp log backfill --week "$PREV_WEEK" --season "$SEASON" --league "$LEAGUE" >> "$LOG_FILE" 2>&1
        fi

        echo "[Tuesday] $LEAGUE: rest-of-season projections..." >> "$LOG_FILE"
        uv run ffapp project --week "$WEEK" --from-week "$WEEK" --through-week 18 --season "$SEASON" --league "$LEAGUE" --no-offline >> "$LOG_FILE" 2>&1

        echo "[Tuesday] $LEAGUE: rest-of-season rankings..." >> "$LOG_FILE"
        uv run ffapp rankings ros --season "$SEASON" --league "$LEAGUE" --no-offline >> "$LOG_FILE" 2>&1
    fi

    echo "[$RUN_LABEL] $LEAGUE: logging week $WEEK snapshot, label=$RUN_LABEL..." >> "$LOG_FILE"
    uv run ffapp log week --week "$WEEK" --run-label "$RUN_LABEL" --season "$SEASON" --league "$LEAGUE" --no-offline >> "$LOG_FILE" 2>&1
done

echo "" >> "$LOG_FILE"
echo "Weekly refresh ($RUN_LABEL) finished $(date)" >> "$LOG_FILE"
