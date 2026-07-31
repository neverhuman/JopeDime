#!/usr/bin/env python3
"""Write the frozen historical-ASPFM terminal receipt for Golden46 new4."""

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
    read_json,
    utc_now,
)


DATASETS = {"ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296", "ZTS"}


def write_receipt(dataset_id: str, artifact_root: Path) -> dict[str, Any]:
    if dataset_id not in DATASETS:
        raise ValueError("historical ASPFM receipts are restricted to Golden46 new4")
    cohort = read_json(COHORT_PATH)
    method_id = str(cohort["historical_aspfm_report_id"])
    receipt = {
        "schema": "golden46.historical-aspfm-terminal-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": method_id,
        "status": "terminal-historical-reference-not-rerun",
        "benchmark_cell_resolved": True,
        "score_present": False,
        "method_identity_changed": False,
        "reason": (
            "The report-backed historical ASPFM identity is retained for Golden42 "
            "provenance but is not rerun as a pre-ASPFM baseline. The separately "
            "declared jope-dime discovery entry is the only new ASPFM evaluation."
        ),
        "created_at": utc_now(),
    }
    atomic_write_json(artifact_root / dataset_id / method_id / "terminal-receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    print(json.dumps(write_receipt(args.dataset, args.artifact_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
