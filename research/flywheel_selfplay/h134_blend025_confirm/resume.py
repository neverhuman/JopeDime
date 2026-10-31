#!/usr/bin/env python3
"""H134 Amendment A-resume launcher: one sha-verified resume of run-20261031T2036Z (owner decision t1715).

The preregistered runner.py is imported byte-pinned and not edited. This launcher repeats its run() steps with two
differences only: the 39 completed fits, the split and the r882 reference costs of the aborted attempt are reused
after each file matches the pinned partial manifest (sha256 f0205dba...), and only the 6 missing fits are run.
Fold, seeds, lambda, arm, gates and readout are the runner's own. Disk stays a hard abort at 93.8%.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "runner.py"
RUNNER_SHA = "5f38be275788f542eacdcb55a72aff6ba6b50e7ff1a6ad046a91506c8c2dbf62"
RESUME_ROOT_NAME = "run-20261031T2036Z"
RESUME_MANIFEST_SHA = "f0205dba5795e91b09d9d5fa49625f21fdfd66b0bb90eb01a895d47e641a842e"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runner():
    if sha(RUNNER_PATH) != RUNNER_SHA:
        raise RuntimeError("H134 runner differs from the preregistered bytes")
    spec = importlib.util.spec_from_file_location("h134_resume_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_resume_root(root):
    """Every file of the aborted attempt must match its sha in the pinned partial manifest; resume only once."""
    manifest_path = root / "manifest.json"
    if root.name != RESUME_ROOT_NAME or sha(manifest_path) != RESUME_MANIFEST_SHA:
        raise RuntimeError("resume root or its pinned manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["status"] != "ABORTED_DISK_GUARD_INCOMPLETE"
        or manifest["outcomes_computed"] is not False
    ):
        raise RuntimeError("only the outcome-free disk-guard abort may be resumed")
    for name, entry in manifest["files"].items():
        if sha(root / name) != entry["sha256"]:
            raise RuntimeError(f"pinned file changed: {name}")
    if (root / "resume-identity.json").exists() or (root / "metrics.json").exists():
        raise RuntimeError("the attempt was already resumed once (no further retries)")
    return manifest


def task_name(task):
    seed, excluded = task
    return f"seed{seed}-exclude{''.join(map(str, excluded))}"


def resume(root):
    root = Path(root)
    r = load_runner()
    parent = Path.home() / "jope-data/h134"
    if root.parent != parent:
        raise RuntimeError("H134 resume root must be under ~/jope-data/h134")
    manifest = verify_resume_root(root)
    h129 = r.load_h129()
    h124 = h129.load_h124()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r.REPO, text=True
    ).strip()
    prereg_sha = r.verify_prereg_committed(source_commit)
    launch = dict(h124.preflight(parent), **r.guards(parent))
    t0, cpu0, launcher_sha = time.monotonic(), time.process_time(), sha(Path(__file__))
    h124.write(
        root / "resume-identity.json",
        {
            "source_commit": source_commit,
            "prereg_commit": r.H134_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": RUNNER_SHA,
            "launcher_sha256": launcher_sha,
            "resume_of_manifest_sha256": RESUME_MANIFEST_SHA,
            "protocol_sha256": r.protocol_sha256(),
            "command": sys.argv,
            "launch": launch,
            "environment": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        },
    )
    with h124.RssWatchdog(root) as watchdog:
        h124.headroom(root)
        selected, context, pools, samples, mapping, context_reads = h124.load_data(root)
        rows = selected["train"]
        if len(rows) != 1219 or len({x["family_id"] for x in rows}) != 205:
            raise RuntimeError("H134 full pinned DL-Train roster required")
        if sha(root / "split.json") != manifest["files"]["split.json"]["sha256"]:
            raise RuntimeError("resumed split differs from the aborted attempt")
        with np.load(root / "r882-train-costs.npz", allow_pickle=False) as z:
            baseline = {x["context_sha256"]: z[x["context_sha256"]] for x in rows}
        tasks = [(seed, (fold,)) for seed in r.SEEDS for fold in range(5)] + [
            (seed, pair)
            for seed in r.SEEDS
            for pair in itertools.combinations(range(5), 2)
        ]
        fits, reused = [], 0
        for task in tasks:
            name = task_name(task)
            if name + ".json" in manifest["files"]:
                receipt = json.loads((root / (name + ".json")).read_text())
                if (
                    receipt["prediction_sha256"]
                    != manifest["files"][name + ".npz"]["sha256"]
                ):
                    raise RuntimeError(
                        f"{name}: prediction does not match its pinned receipt"
                    )
                fits.append(receipt)
                reused += 1
                continue
            r.guards(root)
            fits.append(
                h124.fit_one(task, root, rows, context, pools, samples, mapping)
            )
            h124.headroom(root)
        if len(fits) != 45 or reused != manifest["fits_completed"]:
            raise RuntimeError(
                "H134 resume requires 45 fits with every pinned completed fit reused"
            )
        packets, gate_receipts = h129.tree_packets(h124, root, rows, pools, baseline)
        result, arrays = r.summary(h124, rows, packets)
        dope = json.loads((root / "dope-train.json").read_text())
        if (
            dope["before"] != h124.DOPE
            or dope["after"] != h124.DOPE
            or sha(Path(__file__)) != launcher_sha
        ):
            raise RuntimeError("H134 frozen DOPE/source guard failed")
        arrays_raw = json.dumps(arrays, sort_keys=True).encode()
        import sklearn
        import torch

        metrics = {
            "schema": "jope.h134.blend025-confirm/v1",
            "status": "COMPLETE",
            "verdict": result["verdict"],
            "amendment_a_resume": {
                "resumed_from_manifest_sha256": RESUME_MANIFEST_SHA,
                "fits_reused": reused,
                "fits_run_on_resume": 45 - reused,
                "baseline": "pinned r882-train-costs.npz of the aborted attempt",
                "launcher_sha256": launcher_sha,
            },
            "protocol": r.protocol(),
            "protocol_sha256": r.protocol_sha256(),
            "source_commit": source_commit,
            "prereg_commit": r.H134_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": RUNNER_SHA,
            "arms": result["arms"],
            "calibration_by_seed_fold": {
                k: v["calibration"] for k, v in gate_receipts.items()
            },
            "family_arrays_sha256": hashlib.sha256(arrays_raw).hexdigest(),
            "fit_receipts": [
                {
                    k: f[k]
                    for k in (
                        "seed",
                        "excluded_folds",
                        "fit_rows",
                        "iterations",
                        "prediction_sha256",
                        "wall_seconds",
                    )
                }
                for f in fits
            ],
            "read_boundary": {
                "rows_opened": len(rows),
                "dl_val_cache_rows_decoded": sum(
                    v["overlapping_reads"] for v in context_reads.values()
                ),
                "dl_val_payloads_opened": 0,
                "holdout_payloads_opened": 0,
                "sealed_payloads_opened": 0,
                "meta_test_payloads_opened": 0,
                "live_loop_paths": 0,
                "dl_val_identity_metadata": "selected.json dev rows, cache cids and proj16 offsets are read as layout/identity metadata by the pinned H124 loader (disclosed in PREREG)",
            },
            "DOPE_sha256_before": dope["before"],
            "DOPE_sha256_after": dope["after"],
            "runtime": {
                "wall_seconds": time.monotonic() - t0,
                "process_cpu_seconds": time.process_time() - cpu0,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "rss_watchdog": watchdog.receipt(),
                "scratch_bytes": h124.scratch_bytes(root),
            },
            "versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "GPU_used": False,
            "CUDA_initialized": torch.cuda.is_initialized(),
        }
        raw = h129.compact_metrics(
            metrics
        )  # size checked before any result file is written
        (root / "family_arrays.json").write_bytes(arrays_raw)
        (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"verdict": resume(args.root)["verdict"]}))


if __name__ == "__main__":
    main()
