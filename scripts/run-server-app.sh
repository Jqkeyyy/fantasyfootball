#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p data/outputs/logs
export UV_NO_SYNC=1
export PATH="$HOME/.local/bin:$PATH"

# Cron checks this launcher every minute. Hold the lock for the lifetime of
# Streamlit so every later check exits without starting a duplicate process.
exec 9> data/outputs/.streamlit.lock
flock -n 9 || exit 0

exec uv run streamlit run src/ffapp/app/streamlit_app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
