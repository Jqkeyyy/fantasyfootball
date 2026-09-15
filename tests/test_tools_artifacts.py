from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet


def test_atomic_writers_replace_existing_artifacts(tmp_path: Path) -> None:
    parquet = tmp_path / "nested" / "artifact.parquet"
    manifest = tmp_path / "nested" / "latest.json"
    atomic_write_parquet(pl.DataFrame({"value": [1]}), parquet)
    atomic_write_parquet(pl.DataFrame({"value": [2]}), parquet)
    atomic_write_json({"status": "healthy"}, manifest)

    assert pl.read_parquet(parquet)["value"].item() == 2
    assert json.loads(manifest.read_text())["status"] == "healthy"
    assert not list(parquet.parent.glob("*.tmp"))
