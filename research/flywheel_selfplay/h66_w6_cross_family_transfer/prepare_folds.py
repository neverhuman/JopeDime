#!/usr/bin/env python3
"""Prepare W6 family/physical-table grouped folds from an admitted train roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKET_FILE_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
PACKET_IDENTITY_SHA256 = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
ROSTER_SCHEMA = "jope.w6.train-roster/v1"
OUTPUT_SCHEMA = "jope.w6.family-folds/v1"
ROW_FIELDS = {
    "context_sha256",
    "family_id",
    "physical_table_sha256",
    "task",
    "dl_split",
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class RosterError(ValueError):
    """The supplied roster is outside W6's admitted train-only metadata scope."""


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise RosterError(f"{field} must be a lowercase SHA-256")
    return value


def _components(rows: list[dict[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(family: str) -> str:
        parent.setdefault(family, family)
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            low, high = sorted((a, b))
            parent[high] = low

    table_family: dict[str, str] = {}
    for row in rows:
        family = row["family_id"]
        find(family)
        table = row["physical_table_sha256"]
        previous = table_family.setdefault(table, family)
        union(previous, family)

    grouped: dict[str, set[str]] = defaultdict(set)
    for family in parent:
        grouped[find(family)].add(family)
    return sorted(grouped.values(), key=lambda group: tuple(sorted(group)))


def build_manifest(roster: dict[str, Any], roster_sha256: str) -> dict[str, Any]:
    if set(roster) != {
        "schema",
        "packet_manifest_file_sha256",
        "packet_identity_sha256",
        "rows",
    }:
        raise RosterError("roster has missing or unrecognized top-level fields")
    if roster["schema"] != ROSTER_SCHEMA:
        raise RosterError("unsupported W6 train roster schema")
    if roster["packet_manifest_file_sha256"] != PACKET_FILE_SHA256:
        raise RosterError("packet manifest file identity mismatch")
    if roster["packet_identity_sha256"] != PACKET_IDENTITY_SHA256:
        raise RosterError("packet identity mismatch")
    rows = roster["rows"]
    if not isinstance(rows, list) or len(rows) != 1219:
        raise RosterError("W6 requires exactly 1,219 admitted DL-Train rows")

    clean_rows: list[dict[str, str]] = []
    contexts: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise RosterError(f"row {index} has missing or unrecognized fields")
        context = _require_hash(row["context_sha256"], f"row {index} context_sha256")
        table = _require_hash(
            row["physical_table_sha256"], f"row {index} physical_table_sha256"
        )
        family = row["family_id"]
        if not isinstance(family, str) or not family:
            raise RosterError(f"row {index} family_id must be nonempty")
        if row["task"] != "regression" or row["dl_split"] != "DL-Train":
            raise RosterError(f"row {index} is outside the DL-Train regression closure")
        if context in contexts:
            raise RosterError(f"duplicate context_sha256 {context}")
        contexts.add(context)
        clean_rows.append(
            {
                "context_sha256": context,
                "family_id": family,
                "physical_table_sha256": table,
            }
        )

    families = sorted({row["family_id"] for row in clean_rows})
    if len(families) != 205:
        raise RosterError("W6 requires exactly 205 DL-Train families")

    components = _components(clean_rows)
    if len(components) < 5:
        raise RosterError("fewer than five independent family/table components")
    shuffled = list(components)
    random.Random(4900).shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(5)]
    family_fold: dict[str, int] = {}
    for index, component in enumerate(shuffled):
        fold = index % 5
        folds[fold].extend(sorted(component))
        for family in component:
            family_fold[family] = fold

    assignments = [
        {
            **row,
            "fold": family_fold[row["family_id"]],
        }
        for row in sorted(clean_rows, key=lambda item: item["context_sha256"])
    ]
    return {
        "schema": OUTPUT_SCHEMA,
        "scope": "DL-Train-only family and physical-table grouped metadata",
        "source": {
            "roster_schema": ROSTER_SCHEMA,
            "roster_sha256": _require_hash(roster_sha256, "roster_sha256"),
            "packet_manifest_file_sha256": PACKET_FILE_SHA256,
            "packet_identity_sha256": PACKET_IDENTITY_SHA256,
        },
        "fold_seed": 4900,
        "fold_count": 5,
        "context_count": len(assignments),
        "family_count": len(families),
        "family_table_component_count": len(components),
        "families_by_fold": [sorted(fold) for fold in folds],
        "assignments": assignments,
        "outcome_values_read": 0,
        "dl_val_outer_test_or_sealed_rows_read": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, help="admitted external train-only metadata JSON")
    parser.add_argument("--out", type=Path, help="durable output JSON path")
    args = parser.parse_args()
    if args.roster is None or args.out is None:
        parser.print_help()
        return 0
    roster_path = args.roster.resolve(strict=True)
    output_path = args.out.resolve()
    if output_path == roster_path:
        raise RosterError("refusing to overwrite the source roster")
    roster_bytes = roster_path.read_bytes()
    roster = json.loads(roster_bytes)
    manifest = build_manifest(roster, hashlib.sha256(roster_bytes).hexdigest())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary_fields = (
        "schema",
        "context_count",
        "family_count",
        "family_table_component_count",
        "fold_seed",
        "fold_count",
    )
    print(json.dumps({key: manifest[key] for key in summary_fields}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
