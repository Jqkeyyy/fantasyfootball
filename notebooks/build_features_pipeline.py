"""Compatibility wrapper for the supported feature-refresh workflow.

Prefer `uv run ffapp refresh features --no-offline`. This script remains
for existing scheduled jobs and delegates to the same tested implementation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ffapp.config import load_primary_league, load_settings
from ffapp.tools.feature_refresh import refresh_features


def main() -> None:
    summary = refresh_features(
        load_settings(), load_primary_league(), offline=False, now=datetime.now(UTC)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
