#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run this installer with sudo." >&2
    exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# The first run prints a one-time sign-in URL. Complete it in your browser;
# tailscale up resumes automatically after the server joins the tailnet.
tailscale up

# Keep Streamlit private on loopback and let Tailscale provide authenticated
# tailnet access and HTTPS. The background Serve configuration survives reboot.
tailscale serve --bg http://127.0.0.1:8501

echo
tailscale status
echo
tailscale serve status
