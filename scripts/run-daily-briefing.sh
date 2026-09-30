#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p data/outputs/logs
[[ "$(TZ=America/Chicago date +%H)" == "07" ]] || exit 0
exec 8>data/outputs/.briefing.lock
flock -n 8 || exit 0
exec uv run python -m ffapp.tools.daily_briefing
