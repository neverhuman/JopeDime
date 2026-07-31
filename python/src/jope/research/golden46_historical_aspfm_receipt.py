#!/usr/bin/env python3
"""Write the explicit non-rerun receipt for the historical ASPFM reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from golden46_execution import (
    CAMPAIGN_ID,
    COHORT_PATH,
    DEFAULT_ARTIFACT_ROOT,
    atomic_write_json,
    read_json,
    utc_now,
)


METHOD_ID = "jope-dime-regressor"


def write_receipt(dataset_id: str, artifact_root: Path) -> dict[str, object]:
    cohort = read_json(COHORT_PATH)
    method = next(
        (row for row in cohort["methods"] if row["method_id"] == METHOD_ID),
        None,
    )
    if method is None or method["execution"] != "historical-aspfm-reference-not-rerun-as-baseline":
        raise ValueError("historical ASPFM identity changed in the frozen cohort")
    receipt = {
        "schema": "golden46.historical-aspfm-terminal-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": METHOD_ID,
        "status": "terminal-historical-reference-not-rerun",
        "benchmark_cell_resolved": True,
        "score_present": False,
        "reason": (
            "The report-backed historical ASPFM identity is retained for Golden42 "
            "provenance but is not rerun as a pre-ASPFM baseline. The separately "
            "declared jope-dime discovery entry is the only new ASPFM evaluation."
        ),
        "method_identity_changed": False,
        "created_at": utc_now(),
    }
    atomic_write_json(
        artifact_root / dataset_id / METHOD_ID / "terminal-receipt.json",
        receipt,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    print(json.dumps(write_receipt(args.dataset, args.artifact_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
