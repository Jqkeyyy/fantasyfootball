from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ffapp.tools.run_manifest import inventory_weekly_artifacts, promote_last_known_good


def test_promote_last_known_good_copies_artifacts_and_records_checksums(tmp_path: Path) -> None:
    (tmp_path / "alerts").mkdir()
    (tmp_path / "projections.parquet").write_bytes(b"projection-data")
    (tmp_path / "alerts" / "latest.json").write_text("{}", encoding="utf-8")

    manifest_path = promote_last_known_good(
        tmp_path,
        league_slug="main",
        season=2026,
        week=3,
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )
    manifest = json.loads(manifest_path.read_text())

    assert (tmp_path / "last_known_good" / "projections.parquet").read_bytes() == b"projection-data"
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert {item["name"] for item in inventory_weekly_artifacts(tmp_path)} == {
        "projections.parquet",
        "alerts/latest.json",
    }
