"""Crash-safe writers for generated application artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import polars as pl


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    return Path(raw_path)


def atomic_write_parquet(frame: pl.DataFrame, destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(content: str, destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Mapping[str, object], destination: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2) + "\n", destination)


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a file without ever exposing a partially-written destination."""
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["atomic_copy_file", "atomic_write_json", "atomic_write_parquet", "atomic_write_text"]
