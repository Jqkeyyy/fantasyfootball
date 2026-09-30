#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p data/outputs/logs
exec 8>data/outputs/.discord-bot.lock
flock -n 8 || exit 0
exec uv run python -m ffapp.tools.discord_bot
