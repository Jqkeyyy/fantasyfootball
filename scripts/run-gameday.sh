#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p data/outputs/logs
exec 8>data/outputs/.refresh.lock
flock -n 8 || exit 0
uv run python -m ffapp.tools.transaction_watch || echo "Transaction watcher failed; continuing."
exec uv run python -m ffapp.tools.gameday
