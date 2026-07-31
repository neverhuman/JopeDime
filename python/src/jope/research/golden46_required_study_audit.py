#!/usr/bin/env python3
"""Audit the three-dataset Golden46 required-study publication gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from golden46_execution import (
    CAMPAIGN_ID,
    COHORT_PATH,
    DEFAULT_ARTIFACT_ROOT,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from golden46_matrix_audit import PANEL46, cell_status


DATASETS = ("ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296")
DEFAULT_INDEX_ROOT = COHORT_PATH.parent.parent / "indexes"


def required_study_audit(artifact_root: Path, index_root: Path) -> dict[str, Any]:
    panel = read_json(PANEL46)
    if tuple(panel["dataset_ids"][-4:-1]) != DATASETS:
        raise ValueError("frozen panel46 required-study dataset order changed")
    cohort = read_json(COHORT_PATH)
    methods = cohort["methods"]
    cells = [
        cell_status(dataset_id, method, artifact_root, None)
        for dataset_id in DATASETS
        for method in methods
    ]
    resolved = sum(cell["resolved"] for cell in cells)
    scored = sum(cell["status"] == "validated-score" for cell in cells)
    terminal = sum(
        cell["resolved"] and cell["status"] != "validated-score"
        for cell in cells
    )
    unresolved_statuses: dict[str, int] = {}
    for cell in cells:
        if not cell["resolved"]:
            status = str(cell["status"])
            unresolved_statuses[status] = unresolved_statuses.get(status, 0) + 1
    payload = {
        "schema": "golden46.required-study-matrix-audit.v1",
        "campaign_id": CAMPAIGN_ID,
        "run_id": "golden46-new3-20260723-v4",
        "created_at": utc_now(),
        "scope": {
            "required_dataset_ids": list(DATASETS),
            "excluded_dataset_ids": ["ZTS"],
            "full_zts_artifacts_preserved": True,
        },
        "panel46_sha256": sha256_file(PANEL46),
        "cohort_content_sha256": cohort["content_sha256"],
        "dataset_order": list(DATASETS),
        "method_count": len(methods),
        "expected_cells": len(DATASETS) * len(methods),
        "counts": {
            "resolved": resolved,
            "validated_scores": scored,
            "explicit_terminal_receipts": terminal,
            "unresolved": len(cells) - resolved,
            "unresolved_by_status": unresolved_statuses,
        },
        "cells": cells,
        "resolution_contract": {
            "validated_scores": 372,
            "explicit_terminal_receipts": 3,
            "terminal_breakdown": {
                "historical_aspfm_not_rerun": 3,
            },
        },
        "publication_gate_open": (
            resolved == len(cells) and scored == 372 and terminal == 3
        ),
        "raw_new_dataset_bytes_publishable": False,
    }
    if payload["expected_cells"] != 375:
        raise ValueError("required-study matrix is not exactly 3 datasets by 125 methods")
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    atomic_write_json(index_root / "golden46-new3.matrix-audit.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    args = parser.parse_args()
    result = required_study_audit(args.artifact_root, args.index_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["publication_gate_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
