"""Shared artifact IO helpers for benchmark studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records, returning an empty list when absent."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_ndjson(path: Path, record: dict[str, Any]) -> None:
    """Append one sorted-key JSON record to an NDJSON file."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def write_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Overwrite an NDJSON file with sorted-key records."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Overwrite a CSV file with a stable field order."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def reset_artifacts(output_dir: str | Path, filenames: Iterable[str], *, resume: bool) -> Path:
    """Create an output directory and optionally remove stale study artifacts."""

    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    if resume:
        return resolved
    for filename in filenames:
        path = resolved / filename
        if path.exists():
            path.unlink()
    return resolved
