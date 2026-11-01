#!/usr/bin/env python3
"""Export the pinned 91-row DL-Val metadata roster without reading payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "rnk10_r882_acceptance/evaluation_manifest.json"
)
EXPECTED_SHA256 = "1736ac3a7f8b547b83e0a5792cdacfa117200c9c223ab4964bb13967ead24063"
DEV_ROSTER_SHA256 = "eedd7e98eb02e625f75ea71245a32d8765b75610f0b7f56689088a10b0a08193"


def main() -> int:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("pinned DL-Val evaluation manifest hash mismatch")
    manifest = json.loads(raw)
    source_rows = manifest.get("contexts")
    fields = (
        "context_sha256",
        "dataset_id",
        "dl_split",
        "family_id",
        "task",
        "pool_sha256",
        "train_only_jtt1_sha256",
    )
    contexts = [{field: row[field] for field in fields} for row in source_rows]
    if len(contexts) != 91 or len({row["family_id"] for row in contexts}) != 19:
        raise SystemExit("pinned roster must contain 91 contexts from 19 families")
    if any(
        row.get("dl_split") != "DL-Val" or row.get("task") != "regression"
        for row in contexts
    ):
        raise SystemExit("pinned roster contains a non-DL-Val regression context")
    result = {
        "schema": "jope.rnk10-m5-dev-roster/v1",
        "source_manifest_sha256": manifest["source_manifest_file_sha256"],
        "selected_roster_sha256": manifest["selected_roster_sha256"],
        "evaluation_manifest_sha256": EXPECTED_SHA256,
        "context_count": 91,
        "family_count": 19,
        "payloads_read": 0,
        "contexts": contexts,
    }
    output = Path(__file__).with_name("dev_roster.json")
    output_bytes = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    if hashlib.sha256(output_bytes).hexdigest() != DEV_ROSTER_SHA256:
        raise SystemExit(
            "generated metadata roster differs from its committed hash pin"
        )
    output.write_bytes(output_bytes)
    print(f"wrote metadata-only roster: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
