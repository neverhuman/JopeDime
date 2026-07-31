#!/usr/bin/env python3
"""Finalize and independently audit registered Golden46 prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import golden46_execution as execution


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def content_address(payload: dict[str, Any]) -> dict[str, Any]:
    addressed = dict(payload)
    addressed["content_sha256"] = execution.sha256_bytes(execution.canonical_json(payload))
    return addressed


def checkpoint_manifest() -> dict[str, Any]:
    root = Path("/home/ubuntu/guant/tabpfn")
    names = (
        "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "tabpfn-v2.5-regressor-v2.5_real.ckpt",
        "tabpfn-v3-regressor-v3_default.ckpt",
    )
    artifacts = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {
            "path": str(path),
            "sha256": execution.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "source": {
            "repository": "Prior-Labs/tabpfn_3",
            "v3_filename": "tabpfn-v3-regressor-v3_default.ckpt",
            "acquisition": "huggingface_hub hf download",
            "cache_verification": "one requested remote file checked",
        },
        "artifacts": artifacts,
    }


def audit_registered_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cohort = read_json(args.cohort)
    method_ids = list(cohort["registered_method_ids"])
    if len(method_ids) != 116 or len(set(method_ids)) != 116:
        raise ValueError("frozen cohort does not contain exactly 116 unique registered methods")

    _, model_zoo = execution.import_queue_modules()
    specs = model_zoo.specs_by_name("regression")
    worker_manifest = read_json(args.worker_root / args.dataset / "worker-manifest.json")
    info = model_zoo.DatasetInfo(
        args.dataset,
        int(worker_manifest["train_rows"]),
        int(worker_manifest["test_rows"]),
        int(worker_manifest["n_features"]),
    )
    expected_implementation = {
        "execution_source_sha256": execution.sha256_file(execution.EXECUTION_SOURCE_PATH),
        "model_queue_sha256": execution.sha256_file(execution.MODEL_QUEUE_PATH),
        "model_zoo_sha256": execution.sha256_file(execution.MODEL_ZOO_PATH),
    }

    completed: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    runtime_fingerprints: dict[str, list[str]] = {}
    frozen_zts_ineligible = {
        row["method_id"]: row["reason"]
        for row in cohort["full_zts_terminal_ineligibilities"]
    }
    for method_id in method_ids:
        try:
            spec = specs.get(method_id)
            if spec is None:
                raise ValueError("registered identity disappeared from model_zoo")
            eligible, reason = spec.check_eligibility(info)
            if not eligible:
                if args.dataset != "ZTS" or frozen_zts_ineligible.get(method_id) != reason:
                    raise ValueError(f"unexpected registered ineligibility: {reason}")
                terminal_path = (
                    args.artifact_root / args.dataset / method_id / "terminal-receipt.json"
                )
                terminal = read_json(terminal_path)
                if (
                    terminal.get("schema") != "golden46.terminal-receipt.v1"
                    or terminal.get("campaign_id") != execution.CAMPAIGN_ID
                    or terminal.get("dataset_hash") != args.dataset
                    or terminal.get("algorithm") != method_id
                    or terminal.get("status") != "skipped-ineligible"
                    or terminal.get("reason") != reason
                    or terminal.get("internal_downsampling") is not False
                    or terminal.get("method_identity_changed") is not False
                ):
                    raise ValueError("terminal ineligibility receipt differs from frozen cohort")
                terminals.append(
                    {
                        "method_id": method_id,
                        "reason": reason,
                        "terminal_receipt_sha256": execution.sha256_file(terminal_path),
                    }
                )
                continue
            if args.dataset == "ZTS" and method_id in frozen_zts_ineligible:
                raise ValueError("frozen full-ZTS ineligibility unexpectedly became eligible")
            receipt_path = args.artifact_root / args.dataset / method_id / "worker-receipt.json"
            receipt = read_json(receipt_path)
            if receipt.get("worker_implementation") != expected_implementation:
                raise ValueError("stale or altered worker implementation")
            distributions = receipt.get("environment", {}).get("distributions")
            if not isinstance(distributions, dict):
                raise ValueError("runtime distribution versions are absent")
            for dependency in spec.deps:
                distribution = {
                    "sklearn": "scikit-learn",
                    "pytorch_tabnet": "pytorch-tabnet",
                    "pytorch_tabr": "pytorch-tabr",
                    "torch_frame": "pytorch-frame",
                }.get(dependency, dependency)
                if distributions.get(distribution) is None:
                    raise ValueError(f"runtime distribution is unresolved: {distribution}")
            runtime_sha = execution.sha256_bytes(execution.canonical_json(distributions))
            runtime_fingerprints.setdefault(runtime_sha, []).append(method_id)
            report = execution.finalize_report(
                args.dataset,
                method_id,
                canonical_root=args.canonical_root,
                worker_root=args.worker_root,
                artifact_root=args.artifact_root,
                controller_root=args.controller_root,
            )
            verification = execution.verify_report(
                args.dataset,
                method_id,
                canonical_root=args.canonical_root,
                artifact_root=args.artifact_root,
            )
            completed.append(
                {
                    "method_id": method_id,
                    "rmse": verification["rmse"],
                    "report_sha256": execution.sha256_file(
                        args.artifact_root / args.dataset / method_id / "report.json"
                    ),
                    "prediction_sha256": report["provenance"]["predictions_sha256"],
                    "runtime_fingerprint_sha256": runtime_sha,
                }
            )
        except Exception as error:  # noqa: BLE001 - every failed cell belongs in the index
            errors.append({"method_id": method_id, "error": f"{type(error).__name__}: {error}"})

    expected_scores = len(method_ids) - (
        len(frozen_zts_ineligible) if args.dataset == "ZTS" else 0
    )
    expected_terminals = len(frozen_zts_ineligible) if args.dataset == "ZTS" else 0
    complete = (
        len(completed) == expected_scores
        and len(terminals) == expected_terminals
        and not errors
    )
    payload = {
        "schema": "golden46.registered-dataset-audit.v1",
        "campaign_id": execution.CAMPAIGN_ID,
        "dataset_hash": args.dataset,
        "cohort_sha256": execution.sha256_file(args.cohort),
        "worker_manifest_sha256": execution.sha256_file(
            args.worker_root / args.dataset / "worker-manifest.json"
        ),
        "registered_cohort_size": len(method_ids),
        "registered_expected_scores": expected_scores,
        "registered_expected_terminal_receipts": expected_terminals,
        "registered_completed": len(completed),
        "registered_terminal_receipts": len(terminals),
        "registered_errors": len(errors),
        "complete": complete,
        "worker_implementation": expected_implementation,
        "runtime_fingerprints": {
            digest: sorted(methods) for digest, methods in sorted(runtime_fingerprints.items())
        },
        "checkpoint_manifest": checkpoint_manifest(),
        "scores": sorted(completed, key=lambda row: row["method_id"]),
        "terminal_receipts": sorted(terminals, key=lambda row: row["method_id"]),
        "errors": errors,
    }
    index = content_address(payload)
    output = args.output or (
        execution.CAMPAIGN_ROOT / "indexes" / f"{args.dataset}.registered-audit.json"
    )
    execution.atomic_write_json(output, index)
    if not index["complete"] and not args.allow_incomplete:
        raise RuntimeError(
            "registered audit incomplete: "
            f"{len(completed)}/{expected_scores} scores, "
            f"{len(terminals)}/{expected_terminals} terminals; {len(errors)} errors"
        )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cohort", type=Path, default=execution.COHORT_PATH)
    parser.add_argument("--canonical-root", type=Path, default=execution.CANONICAL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=execution.DEFAULT_WORKER_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=execution.DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--controller-root", type=Path, default=execution.DEFAULT_CONTROLLER_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = audit_registered_dataset(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "registered_completed": result["registered_completed"],
                "registered_terminal_receipts": result["registered_terminal_receipts"],
                "registered_errors": result["registered_errors"],
                "content_sha256": result["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
