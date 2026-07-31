#!/usr/bin/env python3
"""Run and controller-finalize the frozen Golden46 special cohort sequentially."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import golden46_special_execution as special
from golden46_execution import CAMPAIGN_ROOT, atomic_write_json


PHASE_RESOURCES = {
    "cpu": {"cpu", "cpu_light"},
    "gpu": {"gpu"},
}


def run_phase(dataset_id: str, phase: str, device: str) -> dict[str, Any]:
    methods = [
        method_id
        for method_id, spec in special.SPECIAL_SPECS.items()
        if spec["resource"] in PHASE_RESOURCES[phase]
    ]
    started = time.monotonic()
    completed = []
    failures = []
    for method_id in methods:
        output = special.DEFAULT_ARTIFACT_ROOT / dataset_id / method_id
        try:
            if (output / "report.json").is_file():
                verification = special.verify_special(
                    dataset_id,
                    method_id,
                    canonical_root=special.CANONICAL_ROOT,
                    artifact_root=special.DEFAULT_ARTIFACT_ROOT,
                )
                completed.append({**verification, "resume": "validated-report"})
                continue
            if not (output / "worker-receipt.json").is_file():
                special.run_special(
                    dataset_id,
                    method_id,
                    worker_root=special.DEFAULT_WORKER_ROOT,
                    artifact_root=special.DEFAULT_ARTIFACT_ROOT,
                    device=device,
                )
            report = special.finalize_special(
                dataset_id,
                method_id,
                canonical_root=special.CANONICAL_ROOT,
                worker_root=special.DEFAULT_WORKER_ROOT,
                artifact_root=special.DEFAULT_ARTIFACT_ROOT,
                controller_root=special.DEFAULT_CONTROLLER_ROOT,
            )
            completed.append(
                {
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "rmse": report["metrics"]["rmse"],
                    "resume": "fresh-or-unscored-receipt",
                }
            )
        except Exception as error:  # noqa: BLE001 - campaign summary must retain every failure
            failures.append(
                {
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summary = {
        "schema": "golden46.special-phase-summary.v1",
        "dataset_id": dataset_id,
        "phase": phase,
        "device": device if phase == "gpu" else None,
        "one_gpu_model_at_a_time": phase == "gpu",
        "methods": methods,
        "completed": completed,
        "failures": failures,
        "wall_time_seconds": time.monotonic() - started,
    }
    atomic_write_json(
        CAMPAIGN_ROOT / "indexes" / f"{dataset_id}.{phase}.special-run.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_RESOURCES))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_phase(args.dataset, args.phase, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
