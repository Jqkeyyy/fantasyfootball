#!/usr/bin/env bash
# Run hourly from a UTC host. Resolve the intended production schedule in
# America/Chicago so daylight-saving changes do not move refreshes.
set -euo pipefail

CENTRAL_DAY="$(TZ=America/Chicago date +%u)"
CENTRAL_HOUR="$(TZ=America/Chicago date +%H)"

case "${CENTRAL_DAY}:${CENTRAL_HOUR}" in
    2:07) LABEL=tuesday ;;
    4:07) LABEL=thursday ;;
    7:08) LABEL=sunday ;;
    *) exit 0 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$PROJECT_ROOT/weekly-refresh.sh" "$LABEL"
