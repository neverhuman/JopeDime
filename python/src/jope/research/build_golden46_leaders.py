#!/usr/bin/env python3
"""Freeze Golden46 external-baseline leaders from controller-validated artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import golden46_execution as execution
import golden46_special_execution as special


DEFAULT_OUTPUT = execution.COHORT_PATH.parent / "leaders-new3.json"
PANEL_DATASET_IDS = ("ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296", "ZTS")
REQUIRED_DATASET_IDS = ("ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296")


def immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to change frozen Golden46 leader manifest: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def validate_terminal(
    dataset_id: str,
    method_id: str,
    reason: str,
    artifact_root: Path,
) -> dict[str, str]:
    path = artifact_root / dataset_id / method_id / "terminal-receipt.json"
    receipt = execution.read_json(path)
    if (
        receipt.get("schema") != "golden46.terminal-receipt.v1"
        or receipt.get("campaign_id") != execution.CAMPAIGN_ID
        or receipt.get("dataset_hash") != dataset_id
        or receipt.get("algorithm") != method_id
        or receipt.get("status") != "skipped-ineligible"
        or receipt.get("reason") != reason
        or receipt.get("internal_downsampling") is not False
        or receipt.get("method_identity_changed") is not False
    ):
        raise ValueError(f"{dataset_id}/{method_id}: invalid terminal receipt")
    return {
        "method_id": method_id,
        "reason": reason,
        "receipt_sha256": execution.sha256_file(path),
    }


def build_leaders(
    *,
    canonical_root: Path,
    artifact_root: Path,
    cohort_path: Path,
    panel_path: Path,
) -> dict[str, Any]:
    cohort = execution.read_json(cohort_path)
    panel = execution.read_json(panel_path)
    if panel.get("dataset_ids", [])[-4:] != list(PANEL_DATASET_IDS):
        raise ValueError("frozen panel46 does not end with the four Golden46 datasets")
    registered = list(cohort["registered_method_ids"])
    specials = list(cohort["special_runner_ids"])
    if len(registered) != 116 or len(specials) != 7:
        raise ValueError("Golden46 external cohort is not 116 registered plus seven special")
    zts_ineligible = {
        row["method_id"]: row["reason"]
        for row in cohort["full_zts_terminal_ineligibilities"]
    }

    datasets = []
    for dataset_id in REQUIRED_DATASET_IDS:
        scores: dict[str, float] = {}
        reports: dict[str, str] = {}
        terminals = []
        for method_id in registered:
            if dataset_id == "ZTS" and method_id in zts_ineligible:
                terminals.append(
                    validate_terminal(
                        dataset_id,
                        method_id,
                        zts_ineligible[method_id],
                        artifact_root,
                    )
                )
                continue
            verified = execution.verify_report(
                dataset_id,
                method_id,
                canonical_root=canonical_root,
                artifact_root=artifact_root,
            )
            report_path = artifact_root / dataset_id / method_id / "report.json"
            report = execution.read_json(report_path)
            if report.get("benchmark_valid") is not True:
                raise ValueError(f"{dataset_id}/{method_id}: report is not benchmark-valid")
            scores[method_id] = float(verified["rmse"])
            reports[method_id] = execution.sha256_file(report_path)
        for method_id in specials:
            verified = special.verify_special(
                dataset_id,
                method_id,
                canonical_root=canonical_root,
                artifact_root=artifact_root,
            )
            report_path = artifact_root / dataset_id / method_id / "report.json"
            report = execution.read_json(report_path)
            if report.get("benchmark_valid") is not True:
                raise ValueError(f"{dataset_id}/{method_id}: report is not benchmark-valid")
            scores[method_id] = float(verified["rmse"])
            reports[method_id] = execution.sha256_file(report_path)

        expected_scores = 123
        if (
            len(scores) != expected_scores
            or terminals
            or len(scores) + len(terminals) != 123
            or not all(math.isfinite(value) and value >= 0.0 for value in scores.values())
        ):
            raise ValueError(f"{dataset_id}: external baseline resolution is incomplete")
        best = min(scores.values())
        leader_models = sorted(
            method_id for method_id, value in scores.items() if value == best
        )
        source, metadata = execution.canonical_dataset(dataset_id, canonical_root)
        train = execution.read_canonical_frame(source, metadata, "train")
        test = execution.read_canonical_frame(source, metadata, "test")
        datasets.append(
            {
                "dataset_hash": dataset_id,
                "train_rows": int(train.shape[0]),
                "test_rows": int(test.shape[0]),
                "features": int(train.shape[1] - 1),
                "model_order": sorted(scores),
                "rmse": dict(sorted(scores.items())),
                "leader_models": leader_models,
                "leader_rmse": best,
                "validated_report_sha256": dict(sorted(reports.items())),
                "terminal_receipts": sorted(terminals, key=lambda row: row["method_id"]),
                "external_baseline_cells": 123,
            }
        )
    payload = {
        "schema": "golden46.external-leaders.v1",
        "campaign_id": execution.CAMPAIGN_ID,
        "benchmark_valid": True,
        "derivation": (
            "minimum finite controller-recomputed RMSE among the 123 frozen external "
            "baseline scores for the three required-study datasets"
        ),
        "panel46_sha256": execution.sha256_file(panel_path),
        "cohort_sha256": execution.sha256_file(cohort_path),
        "required_dataset_ids": list(REQUIRED_DATASET_IDS),
        "excluded_dataset_ids": ["ZTS"],
        "datasets": datasets,
    }
    payload["content_sha256"] = execution.sha256_bytes(execution.canonical_json(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=execution.CANONICAL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=execution.DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cohort", type=Path, default=execution.COHORT_PATH)
    parser.add_argument(
        "--panel",
        type=Path,
        default=execution.COHORT_PATH.parent / "panel46.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = build_leaders(
        canonical_root=args.canonical_root,
        artifact_root=args.artifact_root,
        cohort_path=args.cohort,
        panel_path=args.panel,
    )
    content = execution.canonical_json(payload)
    if args.verify:
        if not args.output.is_file() or args.output.read_bytes() != content:
            raise ValueError("Golden46 leader manifest verification mismatch")
    else:
        immutable_write(args.output, content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "datasets": len(payload["datasets"]),
                "content_sha256": payload["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
