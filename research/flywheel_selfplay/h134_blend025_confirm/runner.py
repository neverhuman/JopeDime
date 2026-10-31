#!/usr/bin/env python3
"""H134 prospective single-primary confirmation of H129 BLEND_025 vs r882 (CPU only).

BLEND_025 = r882 + 0.25 * (GBM - r882), lambda fixed. The merged H129 runner (sha-pinned) is imported and only its
SEEDS, FOLD_SEED and SAMPLE_SEED are rebound (new family fold 134141, sampling seed 134142, model seeds 141-143);
loading, fitting, the r882 reference pass, calibration and arm scoring are its unchanged code (and, through it, the
pinned H124 module). H134 adds only its own preregistered single-primary readout and run guards.
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
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H129_PATH = HERE.parent / "h129_gbm_blend_harm_gate" / "runner.py"
H129_SHA = "e34d4a3f7ecb82971fa303f05cca665bb1108509abb90021d1a54717fe02cb6c"
H134_PREREG_COMMIT = "7e4537dbfb008728bed255092912c01782b15fb0"
SEEDS = (141, 142, 143)
FOLD_SEED = 134141
SAMPLE_SEED = 134142
PRIMARY = "BLEND_025"
REPORT_ONLY = ("PURE_GBM", "BLEND_050")
CONTROL = "r882"
GAIN_MARGIN = 0.001
HARM_DELTA = -0.01
HARM_CAP = 0.10
TAIL_K = 10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 134000
ELO_BOOTSTRAP_SEED = 134001
MEM_AVAILABLE_MIN_KIB = 30 * 1024**2
DISK_USED_MAX = 0.938
MIN_NICE = 19
METRICS_LIMIT_BYTES = 100_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_h129():
    """The merged H129 runner, byte-pinned, with only SEEDS / FOLD_SEED / SAMPLE_SEED rebound."""
    if sha(H129_PATH) != H129_SHA:
        raise RuntimeError("H129 runner differs from the pinned bytes")
    spec = importlib.util.spec_from_file_location("h134_h129", H129_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEEDS, module.FOLD_SEED, module.SAMPLE_SEED = SEEDS, FOLD_SEED, SAMPLE_SEED
    return module


def tail_checks(family_mean_gain):
    """Pre-declared tail checks on the three-seed mean family gains."""
    g = np.asarray(family_mean_gain, np.float64)
    top = np.sort(g)[::-1][:TAIL_K]
    total = float(g.sum())
    keep = np.argsort(-np.abs(g), kind="stable")[TAIL_K:]
    return {
        "family_median_gain": float(np.median(g)),
        "top10_gain_share": float(top.sum() / total) if total != 0 else None,
        "trimmed_mean_drop10_largest_abs": float(g[keep].mean()),
    }


def readout(gain, samples):
    """gain: (seeds, families) arm - r882 family values. Returns statistics and the H134 admission."""
    per_seed = gain.mean(axis=1)
    boot = np.median(gain[:, samples].mean(axis=2), axis=0)
    family_mean = gain.mean(axis=0)
    harms = family_mean < HARM_DELTA
    out = {
        "per_seed_mean_gain": per_seed.tolist(),
        "median_seed_gain": float(np.median(per_seed)),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "lcb95": float(np.quantile(boot, 0.05)),
        "harm_count": int(harms.sum()),
        "harm_fraction": float(harms.mean()),
        "family_count": int(gain.shape[1]),
        **tail_checks(family_mean),
    }
    out["gates"] = {
        "median_seed_gain_gt_0.001": out["median_seed_gain"] > GAIN_MARGIN,
        "lcb95_gt_0": out["lcb95"] > 0,
        "all_seeds_gt_0": bool((per_seed > 0).all()),
        "harm_le_10pct": out["harm_fraction"] <= HARM_CAP,
        "trimmed_mean_gt_0": out["trimmed_mean_drop10_largest_abs"] > 0,
    }
    out["admitted"] = all(out["gates"].values())
    return out


def summary(h124, rows, packets):
    families, reference = h124.family_values(rows, packets, CONTROL)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(families), size=(BOOTSTRAP_DRAWS, len(families))
    )
    arms, arrays = {}, {}
    for arm in (PRIMARY,) + REPORT_ONLY:
        _, values = h124.family_values(rows, packets, arm)
        gain = values - reference
        arms[arm] = readout(gain, samples)
        if arm != PRIMARY:
            arms[arm]["admitted"] = None  # report-only
        e = h124.elo_h124(rows, packets, arm, rng_seed=ELO_BOOTSTRAP_SEED)
        arms[arm]["elo"] = {
            "registered_estimator": e["registered_estimator"],
            "median_seed_elo": e["median_seed_elo"],
            "pooled_score_elo_additional": e["pooled_score_elo_additional"],
            "reference_elo": e["reference_elo"],
        }
        arrays[arm] = gain.round(8).tolist()
    verdict = "CONFIRMED: BLEND_025" if arms[PRIMARY]["admitted"] else "NO_ADVANCE"
    return {"arms": arms, "verdict": verdict}, {
        "families": families,
        "family_gain_by_seed": arrays,
    }


def protocol():
    return {
        "wave": "H134",
        "primary": PRIMARY,
        "lambda": 0.25,
        "report_only": list(REPORT_ONLY),
        "fold_seed": FOLD_SEED,
        "sampling_seed": SAMPLE_SEED,
        "seeds": list(SEEDS),
        "gain_margin": GAIN_MARGIN,
        "harm_delta": HARM_DELTA,
        "harm_cap": HARM_CAP,
        "tail_k": TAIL_K,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "elo_bootstrap_seed": ELO_BOOTSTRAP_SEED,
        "h129_runner_sha256": H129_SHA,
        "fit_count": 45,
    }


def protocol_sha256():
    return hashlib.sha256(
        json.dumps(protocol(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def guards(path):
    """H134 limits: nice >= 19, MemAvailable >= 30 GiB, filesystem use < 93.8%."""
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H134 requires nice >= 19")
    available = next(
        int(p[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if (p := line.split())[0] == "MemAvailable:"
    )
    usage = shutil.disk_usage(path)
    used = 1 - usage.free / usage.total
    if available < MEM_AVAILABLE_MIN_KIB:
        raise RuntimeError(f"MemAvailable {available} KiB below 30 GiB")
    if used >= DISK_USED_MAX:
        raise RuntimeError(f"disk use {used:.4f} >= 93.8%")
    return {"mem_available_kib": available, "disk_used_fraction": round(used, 4)}


def verify_prereg_committed(source_commit):
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", H134_PREREG_COMMIT, source_commit],
        cwd=REPO,
        check=True,
    )
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{H134_PREREG_COMMIT}:research/flywheel_selfplay/h134_blend025_confirm/PREREG.md",
        ],
        cwd=REPO,
    )
    if expected != (HERE / "PREREG.md").read_bytes():
        raise RuntimeError("H134 PREREG bytes changed after the preregistration commit")
    return hashlib.sha256(expected).hexdigest()


def run(root):
    root = Path(root)
    parent = Path.home() / "jope-data/h134"
    if root.exists() or root.parent != parent or not root.name.startswith("run-"):
        raise RuntimeError("H134 requires a fresh run-* root under ~/jope-data/h134")
    h129 = load_h129()
    h124 = h129.load_h124()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    prereg_sha = verify_prereg_committed(source_commit)
    parent.mkdir(parents=True, exist_ok=True)
    launch = dict(h124.preflight(parent), **guards(parent))
    root.mkdir()
    t0, cpu0, run_sha = time.monotonic(), time.process_time(), sha(Path(__file__))
    h124.write(
        root / "execution-identity.json",
        {
            "source_commit": source_commit,
            "prereg_commit": H134_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "h129_runner_sha256": H129_SHA,
            "h124_module_sha256": h129.H124_SHA,
            "protocol_sha256": protocol_sha256(),
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
        if len(rows) != 1219 or len({r["family_id"] for r in rows}) != 205:
            raise RuntimeError("H134 full pinned DL-Train roster required")
        projection = h124.train_projection(rows, selected["dev"])
        baseline, net = h124.baseline(root, rows, context, pools, projection)
        del net
        h124.release_memory()
        tasks = [(seed, (fold,)) for seed in SEEDS for fold in range(5)] + [
            (seed, pair)
            for seed in SEEDS
            for pair in itertools.combinations(range(5), 2)
        ]
        fits = []
        for task in tasks:
            guards(root)
            fits.append(
                h124.fit_one(task, root, rows, context, pools, samples, mapping)
            )
            h124.headroom(root)
        if len(fits) != 45:
            raise RuntimeError("H134 requires 45 fit receipts")
        packets, gate_receipts = h129.tree_packets(h124, root, rows, pools, baseline)
        result, arrays = summary(h124, rows, packets)
        dope = json.loads((root / "dope-train.json").read_text())
        if (
            dope["before"] != h124.DOPE
            or dope["after"] != h124.DOPE
            or sha(Path(__file__)) != run_sha
        ):
            raise RuntimeError("H134 frozen DOPE/source guard failed")
        arrays_raw = json.dumps(arrays, sort_keys=True).encode()
        import sklearn
        import torch

        metrics = {
            "schema": "jope.h134.blend025-confirm/v1",
            "status": "COMPLETE",
            "verdict": result["verdict"],
            "protocol": protocol(),
            "protocol_sha256": protocol_sha256(),
            "source_commit": source_commit,
            "prereg_commit": H134_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
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


def cpu_dry_run():
    """Synthetic readout control only: no data reads, no fits."""
    rng = np.random.default_rng(0)
    gain = 0.01 + 0.002 * rng.standard_normal((3, 205))
    samples = rng.integers(0, 205, size=(200, 205))
    if not readout(gain, samples)["admitted"] or readout(-gain, samples)["admitted"]:
        raise RuntimeError("H134 readout control failed")
    return {
        "schema": "jope.h134.cpu-dry-run/v1",
        "status": "CPU_DRY_RUN_PASS",
        "protocol_sha256": protocol_sha256(),
        "data_reads": 0,
        "estimator_fits": 0,
        "GPU_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.cpu_dry_run:
        print(json.dumps(cpu_dry_run(), sort_keys=True))
        return
    if args.root is None:
        parser.error("--root is required with --run")
    print(json.dumps({"verdict": run(args.root)["verdict"]}))


if __name__ == "__main__":
    main()
