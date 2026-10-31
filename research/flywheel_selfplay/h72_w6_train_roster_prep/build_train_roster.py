#!/usr/bin/env python3
"""Project the admitted H50 packet into W6's strict DL-Train metadata roster.

The selected packet places its train array first. This reader stops at the end
of that array and never parses the following DL-Val section. The metadata map
is scanned record by record; only dataset IDs selected by DL-Train are decoded.
No pool, table, checkpoint, or outcome payload is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, TextIO

PACKET_FILE_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
PACKET_IDENTITY_SHA256 = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
SELECTED_FILE_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
TASK_METADATA_FILE_SHA256 = "510fb1a0536766246564002b6d8186c260426f88a0f1ae9d16661b53dcda13f0"
EXPECTED_CONTEXTS = 1219
EXPECTED_FAMILIES = 205
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RosterError(ValueError):
    """The packet is not the pinned training-only source expected by W6."""


class JsonStream:
    """Small character reader that can skip unselected JSON values."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.pending: str | None = None

    def peek(self) -> str:
        if self.pending is None:
            self.pending = self.stream.read(1)
        return self.pending

    def get(self) -> str:
        if self.pending is not None:
            value, self.pending = self.pending, None
            return value
        return self.stream.read(1)

    def skip_ws(self) -> None:
        while self.peek() and self.peek().isspace():
            self.get()

    def raw_string(self) -> str:
        self.skip_ws()
        if self.get() != '"':
            raise RosterError("expected JSON string")
        chars = ['"']
        escaped = False
        while True:
            char = self.get()
            if not char:
                raise RosterError("truncated JSON string")
            chars.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return "".join(chars)

    def value(self, *, decode: bool) -> Any:
        """Read one JSON value; unselected subtrees are scanned, not decoded."""
        self.skip_ws()
        first = self.peek()
        if not first:
            raise RosterError("missing JSON value")
        chars: list[str] | None = [] if decode else None

        def take() -> str:
            char = self.get()
            if not char:
                raise RosterError("truncated JSON value")
            if chars is not None:
                chars.append(char)
            return char

        first = take()
        if first in "{[":
            stack = ["}" if first == "{" else "]"]
            in_string = False
            escaped = False
            while stack:
                char = take()
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char in "}]":
                    if char != stack[-1]:
                        raise RosterError("mismatched JSON delimiters")
                    stack.pop()
        elif first == '"':
            escaped = False
            while True:
                char = take()
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    break
        else:
            while self.peek() and self.peek() not in ",]}" and not self.peek().isspace():
                take()

        if chars is None:
            return None
        return json.loads("".join(chars))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_rows(path: Path) -> list[dict[str, Any]]:
    """Decode only the first top-level train array; stop before DL-Val."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        reader = JsonStream(stream)
        reader.skip_ws()
        if reader.get() != "{":
            raise RosterError("selected packet must be a JSON object")
        key = json.loads(reader.raw_string())
        if key != "train":
            raise RosterError("train partition must be the first selected packet field")
        reader.skip_ws()
        if reader.get() != ":":
            raise RosterError("missing train partition separator")
        reader.skip_ws()
        if reader.get() != "[":
            raise RosterError("train partition must be an array")
        while True:
            reader.skip_ws()
            if reader.peek() == ",":
                reader.get()
                continue
            if reader.peek() == "]":
                reader.get()
                break
            row = reader.value(decode=True)
            if not isinstance(row, dict):
                raise RosterError("train partition entries must be objects")
            rows.append(row)
    return rows


def read_selected_metadata(path: Path, dataset_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Decode metadata values only for the dataset IDs admitted by DL-Train."""
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        reader = JsonStream(stream)
        reader.skip_ws()
        if reader.get() != "{":
            raise RosterError("task metadata must be a JSON object")
        while True:
            reader.skip_ws()
            if reader.peek() == ",":
                reader.get()
                continue
            if reader.peek() == "}":
                reader.get()
                break
            key = json.loads(reader.raw_string())
            reader.skip_ws()
            if reader.get() != ":":
                raise RosterError("missing task metadata separator")
            value = reader.value(decode=key in dataset_ids)
            if key in dataset_ids:
                if not isinstance(value, dict):
                    raise RosterError("selected task metadata must be an object")
                found[key] = value
    missing = dataset_ids - found.keys()
    if missing:
        raise RosterError(f"missing selected training metadata for {len(missing)} dataset IDs")
    return found


def make_roster(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = input_dir / "source_manifest.json"
    selected_path = input_dir / "selected.json"
    metadata_path = input_dir / "task_metadata.json"

    if sha256_file(source_path) != PACKET_FILE_SHA256:
        raise RosterError("H50 source manifest file SHA256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("source_manifest_sha256") != PACKET_IDENTITY_SHA256:
        raise RosterError("H50 packet identity mismatch")
    if (source.get("dl_split_counts", {}).get("DL-Train"), source.get("dl_split_counts", {}).get("DL-Val")) != (1219, 91):
        raise RosterError("H50 split counts differ from the admitted packet")
    if (source.get("excluded_dl_holdout_count"), source.get("excluded_payloads_opened"), source.get("outer_rows_delivered")) != (93, 0, 0):
        raise RosterError("H50 excluded-row closure differs from the admitted packet")
    if sha256_file(selected_path) != SELECTED_FILE_SHA256:
        raise RosterError("selected packet SHA256 mismatch")
    if sha256_file(metadata_path) != TASK_METADATA_FILE_SHA256:
        raise RosterError("task metadata SHA256 mismatch")

    rows = read_train_rows(selected_path)
    if len(rows) != EXPECTED_CONTEXTS:
        raise RosterError("W6 requires exactly 1,219 DL-Train rows")
    contexts: set[str] = set()
    families: set[str] = set()
    dataset_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("dl_split") != "DL-Train" or row.get("task") != "regression":
            raise RosterError(f"selected row {index} is outside DL-Train regression")
        if row.get("source_split") not in {"train", "inner"}:
            raise RosterError(f"selected row {index} has an unadmitted source split")
        for field in ("context_sha256", "family_id", "dataset_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise RosterError(f"selected row {index} has no {field}")
        if not SHA256_RE.fullmatch(row["context_sha256"]):
            raise RosterError(f"selected row {index} context identity is not SHA-256")
        if row["context_sha256"] in contexts:
            raise RosterError("duplicate training context identity")
        contexts.add(row["context_sha256"])
        families.add(row["family_id"])
        dataset_ids.add(row["dataset_id"])
    if len(families) != EXPECTED_FAMILIES:
        raise RosterError("W6 requires exactly 205 DL-Train families")

    metadata = read_selected_metadata(metadata_path, dataset_ids)
    clean_rows: list[dict[str, str]] = []
    source_objects: dict[str, dict[str, str]] = {}
    for row in rows:
        item = metadata[row["dataset_id"]]
        key = item.get("key")
        version_id = item.get("version_id")
        source_sha = item.get("sha256")
        physical_sha = item.get("canonical_sha256")
        if not isinstance(key, str) or not key.startswith("datasets/"):
            raise RosterError("training source key is outside the public datasets prefix")
        if not isinstance(version_id, str) or not version_id:
            raise RosterError("training source object has no pinned S3 version")
        if not isinstance(source_sha, str) or not SHA256_RE.fullmatch(source_sha):
            raise RosterError("training source object has no SHA-256")
        if not isinstance(physical_sha, str) or not SHA256_RE.fullmatch(physical_sha):
            raise RosterError("training source object has no canonical physical-table SHA-256")
        source_objects[row["dataset_id"]] = {
            "dataset_id": row["dataset_id"],
            "s3_bucket": "veox-jopedime-datasets-use2",
            "s3_key": key,
            "version_id": version_id,
            "object_sha256": source_sha,
            "physical_table_sha256": physical_sha,
        }
        clean_rows.append(
            {
                "context_sha256": row["context_sha256"],
                "family_id": row["family_id"],
                "physical_table_sha256": physical_sha,
                "task": "regression",
                "dl_split": "DL-Train",
            }
        )
    clean_rows.sort(key=lambda row: row["context_sha256"])
    roster = {
        "schema": "jope.w6.train-roster/v1",
        "packet_manifest_file_sha256": PACKET_FILE_SHA256,
        "packet_identity_sha256": PACKET_IDENTITY_SHA256,
        "rows": clean_rows,
    }
    inventory = {
        "schema": "jope.w6.training-source-inventory/v1",
        "source_manifest_file_sha256": PACKET_FILE_SHA256,
        "packet_identity_sha256": PACKET_IDENTITY_SHA256,
        "selected_file_sha256": SELECTED_FILE_SHA256,
        "task_metadata_file_sha256": TASK_METADATA_FILE_SHA256,
        "roster_context_count": len(clean_rows),
        "roster_family_count": len(families),
        "outcome_values_read": 0,
        "pool_table_checkpoint_or_score_bodies_opened": 0,
        "dl_val_rows_parsed": 0,
        "excluded_holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "training_source_objects": [source_objects[key] for key in sorted(source_objects)],
    }
    return roster, inventory


def write_new(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    args = parser.parse_args()
    roster, inventory = make_roster(args.input)
    roster_sha = write_new(args.roster, roster)
    inventory_sha = write_new(args.inventory, inventory)
    print(json.dumps({"context_count": len(roster["rows"]), "family_count": inventory["roster_family_count"], "roster_sha256": roster_sha, "source_inventory_sha256": inventory_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
