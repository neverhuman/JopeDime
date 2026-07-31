#!/usr/bin/env python3
"""Audit the frozen special-runner gate for one Golden46 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from golden46_execution import (
    CAMPAIGN_ID,
    CANONICAL_ROOT,
    DEFAULT_ARTIFACT_ROOT,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from golden46_special_execution import (
    CHIMERA_EXPECTED,
    COHORT_PATH,
    SPECIAL_SOURCE_PATH,
    frozen_special_ids,
    special_identity,
    verify_special,
)


DEFAULT_INDEX_ROOT = COHORT_PATH.parent.parent / "indexes"


def audit_specials(
    dataset_id: str,
    *,
    canonical_root: Path,
    artifact_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    cells = []
    validated = 0
    blocked = 0
    errors = []
    for method_id in sorted(frozen_special_ids()):
        output = artifact_root / dataset_id / method_id
        try:
            if (output / "report.json").is_file():
                verification = verify_special(
                    dataset_id,
                    method_id,
                    canonical_root=canonical_root,
                    artifact_root=artifact_root,
                )
                report = read_json(output / "report.json")
                cells.append(
                    {
                        "method_id": method_id,
                        "status": "validated-score",
                        "rmse": verification["rmse"],
                        "report_sha256": sha256_file(output / "report.json"),
                        "worker_receipt_sha256": sha256_file(
                            output / "worker-receipt.json"
                        ),
                        "predictions_sha256": sha256_file(output / "predictions.npy"),
                    }
                )
                if report.get("benchmark_valid") is not True:
                    raise ValueError("special baseline report is not benchmark_valid")
                validated += 1
            else:
                receipt_path = output / "compatibility-receipt.json"
                receipt = read_json(receipt_path)
                if receipt.get("status") != "blocked-missing-frozen-artifact":
                    raise ValueError("unrecognized unresolved special-runner status")
                if receipt.get("benchmark_cell_resolved") is not False:
                    raise ValueError("missing checkpoint was incorrectly resolved")
                if receipt.get("missing_artifact") != CHIMERA_EXPECTED:
                    raise ValueError("missing checkpoint identity mismatch")
                cells.append(
                    {
                        "method_id": method_id,
                        "status": receipt["status"],
                        "benchmark_cell_resolved": False,
                        "compatibility_receipt_sha256": sha256_file(receipt_path),
                        "missing_artifact": CHIMERA_EXPECTED,
                    }
                )
                blocked += 1
        except Exception as error:  # noqa: BLE001
            errors.append({"method_id": method_id, "error": repr(error)})
    payload = {
        "schema": "golden46.special-runner-audit.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "created_at": utc_now(),
        "special_execution_source_sha256": sha256_file(SPECIAL_SOURCE_PATH),
        "expected_method_ids": sorted(frozen_special_ids()),
        "method_identities": {
            method_id: special_identity(method_id)
            for method_id in sorted(frozen_special_ids())
        },
        "counts": {
            "expected": len(frozen_special_ids()),
            "validated_scores": validated,
            "blocked_missing_frozen_artifact": blocked,
            "errors": len(errors),
        },
        "cells": cells,
        "errors": errors,
        "complete": validated == len(frozen_special_ids()) and not errors,
        "publication_gate_open": validated == len(frozen_special_ids()) and not errors,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    atomic_write_json(index_root / f"{dataset_id}.special-audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_specials(
        args.dataset,
        canonical_root=args.canonical_root,
        artifact_root=args.artifact_root,
        index_root=args.index_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
