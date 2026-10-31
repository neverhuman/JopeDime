#!/usr/bin/env python3
"""H99 CPU preparation, resource preflight, and explicitly authorized FIFO executor."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"
)

from runner import cpu_dry_run, execution_plan
from coordinator import launch_preflight, run_fifo_cells, apply_worker_limits, acquire_h99_fifo
from executor import INPUT, CHECKPOINT, run_cell, score_fold_b, validate_execution_paths
from build_metrics import build_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-dry-run", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--launch-preflight", action="store_true")
    modes.add_argument("--execute-wave", action="store_true")
    parser.add_argument("--authorized-execution", action="store_true", help="required explicit owner authorization for the GPU wave")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "jope-data/h99/h99-wave")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--eta-utc", help="owner ETA recorded in qualification.queue while waiting")
    args = parser.parse_args()
    if args.cpu_dry_run:
        result = cpu_dry_run()
    elif args.launch_preflight:
        result = {"schema": "jope.h99.launch-preflight/v1", "status": "pass", **launch_preflight()}
    elif args.execute_wave:
        if not args.authorized_execution:
            raise RuntimeError("H99 GPU execution requires --authorized-execution after owner launch approval")
        if not args.eta_utc:
            raise RuntimeError("H99 FIFO enqueue requires an owner ETA via --eta-utc")
        try:
            eta = datetime.fromisoformat(args.eta_utc.replace("Z", "+00:00"))
            if eta.tzinfo is None:
                raise ValueError("timezone is required")
        except ValueError as exc:
            raise RuntimeError("--eta-utc must be an ISO-8601 timestamp with timezone") from exc
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("H99 GPU execution requires CUDA")
        run_root = args.run_root
        input_dir = args.input
        checkpoint = args.checkpoint
        validate_execution_paths(run_root, input_dir, checkpoint)
        cells = [*execution_plan()["candidate_cells"], execution_plan()["null_cell"]]
        def worker(cell, staged, scratch, _remaining):
            return run_cell(cell, staged, scratch, run_root, input_dir=input_dir)
        def finalize(staged, scratch, _remaining):
            return score_fold_b(staged, scratch, run_root, input_dir=input_dir)
        def acquire(status_path, *, wait_total):
            return acquire_h99_fifo(
                status_path, wait_total=wait_total, owner="luna_v1_res03", eta_utc=args.eta_utc
            )
        budget = run_fifo_cells(
            cells, run_root=run_root, checkpoint=checkpoint, worker=worker,
            finalize_worker=finalize, setup_fn=apply_worker_limits, acquire_fn=acquire,
        )
        fold_b = json.loads((run_root / "outcomes/fold_b.json").read_text())
        metrics = build_metrics(fold_b)
        out = run_root / "outcomes/admission.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".new")
        temporary.write_text(json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.replace(out)
        result = {"schema": "jope.h99.wave/v1", "status": "complete", "budget_attempts": len(budget["attempts"]), "admission": str(out)}
    else:
        result = execution_plan()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
