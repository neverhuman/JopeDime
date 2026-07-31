#!/usr/bin/env python3
"""Account for every prospective Golden46 cell and enforce publication gating."""

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


PANEL46 = Path("/home/ubuntu/golden42/golden42_foundry/data/panel46.json")
DEFAULT_INDEX_ROOT = COHORT_PATH.parent.parent / "indexes"
NEW_DATASET_COUNT = 4


def cell_status(
    dataset_id: str,
    method: dict[str, Any],
    artifact_root: Path,
    terminal_reason: str | None,
) -> dict[str, Any]:
    method_id = method["method_id"]
    output = artifact_root / dataset_id / method_id
    report = output / "report.json"
    terminal = output / "terminal-receipt.json"
    compatibility = output / "compatibility-receipt.json"
    if report.is_file():
        payload = read_json(report)
        common_valid = (
            payload.get("status") == "completed"
            and payload.get("campaign_id") == CAMPAIGN_ID
            and payload.get("algorithm") == method_id
            and payload.get("dataset_hash") == dataset_id
            and payload.get("metrics", {}).get("rmse") is not None
        )
        if method["execution"] == "adaptive-discovery":
            valid = (
                common_valid
                and payload.get("schema") == "golden46.aspfm-validated-report.v1"
                and payload.get("benchmark_valid") is False
                and payload.get("provenance", {}).get("three_node_confirmed") is True
            )
        else:
            valid = (
                common_valid
                and payload.get("schema") == "golden46.validated-report.v1"
                and payload.get("benchmark_valid") is True
            )
        return {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "execution": method["execution"],
            "status": "validated-score" if valid else "invalid-report",
            "resolved": valid,
            "report_sha256": sha256_file(report),
            "rmse": payload.get("metrics", {}).get("rmse"),
        }
    if terminal.is_file():
        payload = read_json(terminal)
        if method["execution"] == "historical-aspfm-reference-not-rerun-as-baseline":
            resolved = (
                payload.get("schema")
                == "golden46.historical-aspfm-terminal-receipt.v1"
                and payload.get("campaign_id") == CAMPAIGN_ID
                and payload.get("dataset_hash") == dataset_id
                and payload.get("algorithm") == method_id
                and payload.get("status")
                == "terminal-historical-reference-not-rerun"
                and payload.get("benchmark_cell_resolved") is True
                and payload.get("score_present") is False
                and payload.get("method_identity_changed") is False
            )
        else:
            resolved = (
                dataset_id == "ZTS"
                and terminal_reason is not None
                and payload.get("schema") == "golden46.terminal-receipt.v1"
                and payload.get("campaign_id") == CAMPAIGN_ID
                and payload.get("dataset_hash") == dataset_id
                and payload.get("algorithm") == method_id
                and payload.get("status") == "skipped-ineligible"
                and payload.get("reason") == terminal_reason
                and payload.get("internal_downsampling") is False
                and payload.get("method_identity_changed") is False
            )
        return {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "execution": method["execution"],
            "status": payload.get("status", "unknown-terminal"),
            "resolved": resolved,
            "terminal_receipt_sha256": sha256_file(terminal),
        }
    if compatibility.is_file():
        payload = read_json(compatibility)
        return {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "execution": method["execution"],
            "status": payload.get("status", "unknown-compatibility"),
            "resolved": payload.get("benchmark_cell_resolved") is True,
            "compatibility_receipt_sha256": sha256_file(compatibility),
        }
    return {
        "dataset_id": dataset_id,
        "method_id": method_id,
        "execution": method["execution"],
        "status": "missing",
        "resolved": False,
    }


def matrix_audit(artifact_root: Path, index_root: Path) -> dict[str, Any]:
    panel = read_json(PANEL46)
    datasets = panel["dataset_ids"][-NEW_DATASET_COUNT:]
    cohort = read_json(COHORT_PATH)
    methods = cohort["methods"]
    zts_terminal_reasons = {
        row["method_id"]: row["reason"]
        for row in cohort["full_zts_terminal_ineligibilities"]
    }
    cells = [
        cell_status(
            dataset_id,
            method,
            artifact_root,
            zts_terminal_reasons.get(method["method_id"]),
        )
        for dataset_id in datasets
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
        "schema": "golden46.new-cell-matrix-audit.v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": utc_now(),
        "panel46_sha256": sha256_file(PANEL46),
        "cohort_content_sha256": cohort["content_sha256"],
        "dataset_order": datasets,
        "method_count": len(methods),
        "expected_cells": len(datasets) * len(methods),
        "counts": {
            "resolved": resolved,
            "validated_scores": scored,
            "explicit_terminal_receipts": terminal,
            "unresolved": len(cells) - resolved,
            "unresolved_by_status": unresolved_statuses,
        },
        "cells": cells,
        "resolution_contract": {
            "validated_scores": 476,
            "explicit_terminal_receipts": 24,
            "terminal_breakdown": {
                "full_zts_size_ineligibilities": 20,
                "historical_aspfm_not_rerun": 4,
            },
            "note": (
                "Four historical-ASPFM cells plus twenty full-ZTS ineligibilities "
                "make the only cohort-consistent 500-cell split."
            ),
        },
        "publication_gate_open": (
            resolved == len(cells) and scored == 476 and terminal == 24
        ),
        "raw_new_dataset_bytes_publishable": False,
    }
    if payload["expected_cells"] != 500:
        raise ValueError("Golden46 matrix is not exactly 4 datasets by 125 methods")
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    atomic_write_json(index_root / "golden46-new4.matrix-audit.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    args = parser.parse_args()
    result = matrix_audit(args.artifact_root, args.index_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
