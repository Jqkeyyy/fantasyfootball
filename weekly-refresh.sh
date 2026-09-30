#!/usr/bin/env bash
# Weekly in-season refresh, driven by host cron via `docker compose exec`.
# Usage: weekly-refresh.sh [tuesday|thursday|sunday]
#
# The CLI owns week selection and the complete account-wide workflow. Keeping
# this wrapper thin ensures Linux cron and Windows Task Scheduler run the same
# production path.
set -uo pipefail
export UV_NO_SYNC=1
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")"
mkdir -p data/outputs
exec 8>data/outputs/.refresh.lock
flock -w 900 8 || exit 1

RUN_LABEL="${1:-}"
case "$RUN_LABEL" in
    tuesday|thursday|sunday) ;;
    *)
        echo "Usage: weekly-refresh.sh [tuesday|thursday|sunday]"
        exit 1
        ;;
esac

LOG_DIR="data/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/scheduled-refresh-${RUN_LABEL}.log"

{
    echo ""
    echo "============================================"
    echo " Weekly refresh: $RUN_LABEL ($(date --iso-8601=seconds))"
    echo "============================================"
} >> "$LOG_FILE"

uv run ffapp refresh weekly --all-leagues --run-label "$RUN_LABEL" --no-offline \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "CLI exit code: $EXIT_CODE" >> "$LOG_FILE"
if [ "$EXIT_CODE" -ne 0 ]; then
    uv run python - "$RUN_LABEL" "$EXIT_CODE" >> "$LOG_FILE" 2>&1 <<'PY'
import sys

from ffapp.tools.discord_notifications import send_action_notification, with_dashboard_link

label, exit_code = sys.argv[1:]
send_action_notification(
    with_dashboard_link(
        f"Scheduled {label} fantasy-football refresh failed with exit code {exit_code}."
    ),
    key=f"scheduled-refresh-failed:{label}:{exit_code}",
    cooldown=3600,
)
PY
fi
exit "$EXIT_CODE"
