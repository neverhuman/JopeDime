#!/usr/bin/env python3
"""H129 GBM-prior blend replication, smaller lambda and a fit-fold harm gate vs r882 (CPU only).

H124 found a CI-clean U4 gain for the lambda 0.5 blend of the GBM context prior toward r882 (+0.032, LCB95 +0.017)
that failed only its harm cap (22/205 families = 10.7% > 10%). H129 is a fresh confirmation, preregistered before any
scoring: new family fold (RNG 129029), new model seeds 91-93, the same inputs and H119/H85 recipe, and three primary
arms against the matching fresh r882 reference:

* BLEND_050       r882 + 0.5  * (GBM - r882), the H124 primary replicated;
* BLEND_025       r882 + 0.25 * (GBM - r882);
* HARM_GATED_050  BLEND_050 on a context only when its GBM-vs-r882 disagreement is at or below a threshold frozen
                  from fit-fold (inner out-of-fold) predictions; r882 otherwise.

Fitting, data loading, the r882 reference pass and the RSS watchdog are the merged H124 module (sha-pinned) run
unchanged with H129's fold, sampling seed and model seeds. No H124 outcome enters H129.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
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
H124_PATH = HERE.parent / "h124_gbm_shrink_blend" / "runner.py"
H124_SHA = "9ebce9652a7caadc7e50896a9c1b667de5aea53f1e3cf4f8797f815d8a16285a"
H129_PREREG_COMMIT = "__PREREG_COMMIT__"
SEEDS = (91, 92, 93)
FOLD_SEED = 129029
SAMPLE_SEED = 129029
LAMBDAS = {"BLEND_050": 0.5, "BLEND_025": 0.25}
GATED = "HARM_GATED_050"
PRIMARY = ("BLEND_050", "BLEND_025", GATED)
REPORT_ONLY = ("PURE_GBM",)
CONTROL = "r882"
GATE_QUANTILES = (0.25, 0.5, 0.75, 1.0)
FIT_HARM_TARGET = 0.05
HARM_DELTA = -0.01
HARM_CAP = 0.10
LCB_MARGIN = 0.001
HOLM_ALPHA = 0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 129000
ELO_BOOTSTRAP_SEED = 129001
MEM_AVAILABLE_MIN_KIB = 30 * 1024**2
DISK_USED_MAX = 0.94
METRICS_LIMIT_BYTES = 100_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_h124():
    """The merged H124 runner, byte-pinned, rebound to H129's fold, sampling RNG and model seeds."""
    if sha(H124_PATH) != H124_SHA:
        raise RuntimeError("H124 module differs from the pinned bytes")
    spec = importlib.util.spec_from_file_location("h129_h124", H124_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEEDS, module.FOLD_SEED, module.SAMPLE_SEED = SEEDS, FOLD_SEED, SAMPLE_SEED
    return module


def blend(r882_cost, gbm_cost, lam):
    """Utility-space interpolation; negating both costs makes cost-space blending identical (as in H124)."""
    reference, prior = np.asarray(r882_cost), np.asarray(gbm_cost)
    if lam not in LAMBDAS.values() or reference.shape != prior.shape:
        raise ValueError("H129 lambdas are fixed to 0.5 and 0.25")
    if not (np.isfinite(reference).all() and np.isfinite(prior).all()):
        raise ValueError("finite score vectors required")
    return (reference + lam * (prior - reference)).astype(np.float32)


def disagreement(gbm_cost, r882_cost):
    """1 - Spearman rank correlation of the two costs over unrevealed slots 4..P-1 (0 = same order, 2 = reversed).

    Outcome-free; invariant to the positive-slope calibration. A constant vector gives +inf (gate closed)."""
    a, b = np.asarray(gbm_cost, np.float64)[4:], np.asarray(r882_cost, np.float64)[4:]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.inf

    def ranks(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=r)
        return (sums / counts)[inverse]  # average ranks for ties

    return float(1.0 - np.corrcoef(ranks(a), ranks(b))[0, 1])


def family_gain_harm(deltas, families):
    """Equal-family mean of per-family mean deltas, and the fraction of families with mean < HARM_DELTA."""
    by = {}
    for d, f in zip(deltas, families):
        by.setdefault(f, []).append(d)
    means = np.asarray([np.mean(v) for _, v in sorted(by.items())])
    return float(means.mean()), float((means < HARM_DELTA).mean())


def choose_threshold(d_values, blend_deltas, families):
    """Fit-fold-only gate threshold (preregistered rule).

    Candidates are the fit-fold disagreement quantiles in GATE_QUANTILES (1.0 = gate always open). A candidate is
    feasible when the gated fit-fold family harm fraction is <= FIT_HARM_TARGET; the feasible candidate with the
    largest fit-fold equal-family gain wins, ties to the smaller quantile. With none feasible the gate stays closed
    (-inf: r882 everywhere)."""
    d = np.asarray(d_values, np.float64)
    finite = d[np.isfinite(d)]
    best = (-math.inf, None, None)
    trace = []
    for q in GATE_QUANTILES:
        tau = float(np.quantile(finite, q)) if len(finite) else -math.inf
        gated = np.where(d <= tau, blend_deltas, 0.0)
        gain, harm = family_gain_harm(gated, families)
        trace.append({"quantile": q, "tau": tau, "fit_gain": gain, "fit_harm": harm})
        if harm <= FIT_HARM_TARGET and gain > best[0]:
            best = (gain, q, tau)
    return {
        "tau": best[2] if best[1] is not None else -math.inf,
        "quantile": best[1],
        "candidates": trace,
    }


def tree_packets(h124, root, rows, pools, baseline):
    """Calibrate as H124, choose each outer fold's gate from its fit folds, then score every arm on held-out folds."""
    _, mapping = h124.folds(rows)
    family = {r["context_sha256"]: r["family_id"] for r in rows}
    packets, receipts = {}, {}
    for seed in SEEDS:
        packets[str(seed)] = {}
        for k in range(5):
            inner = {}
            for j in range(5):
                if j == k:
                    continue
                pair = tuple(sorted((k, j)))
                with np.load(
                    root / f"seed{seed}-exclude{pair[0]}{pair[1]}.npz",
                    allow_pickle=False,
                ) as z:
                    for row in rows:
                        cid = row["context_sha256"]
                        if mapping[cid] == j:
                            inner[cid] = z[cid].astype(np.float64)
            pred = np.concatenate([inner[c][4:] for c in sorted(inner)])
            fit = np.concatenate([pools[c]["fitness"][4:] for c in sorted(inner)])
            gate = np.concatenate([pools[c]["gate"][4:] for c in sorted(inner)])
            a, b = h124.calibration(pred, fit, gate)
            d_fit, delta_fit, fam_fit = [], [], []
            for cid in sorted(inner):
                cost = (a - b * inner[cid]).astype(np.float32)
                u = pools[cid]["u"]
                d_fit.append(disagreement(cost, baseline[cid]))
                delta_fit.append(
                    h124.endpoints(blend(baseline[cid], cost, 0.5), u)["u4"]
                    - h124.endpoints(baseline[cid], u)["u4"]
                )
                fam_fit.append(family[cid])
            gate_rule = choose_threshold(d_fit, delta_fit, fam_fit)
            receipts[f"{seed}/{k}"] = {
                "calibration": [a, b],
                "gate": gate_rule,
                "fit_contexts": len(inner),
            }
            with np.load(root / f"seed{seed}-exclude{k}.npz", allow_pickle=False) as z:
                for row in rows:
                    cid = row["context_sha256"]
                    if mapping[cid] != k:
                        continue
                    gbm = (a - b * z[cid]).astype(np.float32)
                    u, ref = pools[cid]["u"], baseline[cid]
                    d = disagreement(gbm, ref)
                    opened = bool(d <= gate_rule["tau"])
                    b50 = blend(ref, gbm, 0.5)
                    packets[str(seed)][cid] = {
                        CONTROL: h124.endpoints(ref, u),
                        "BLEND_050": h124.endpoints(b50, u),
                        "BLEND_025": h124.endpoints(blend(ref, gbm, 0.25), u),
                        GATED: h124.endpoints(b50 if opened else ref, u),
                        "PURE_GBM": h124.endpoints(gbm, u),
                        "disagreement": d,
                        "gate_open": opened,
                    }
    return packets, receipts


def holm(p_values, alpha=HOLM_ALPHA):
    """Holm step-down: reject in ascending-p order while p_(i) <= alpha / (m - i); stop at the first failure."""
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m, reject = len(order), {k: False for k in p_values}
    for i, key in enumerate(order):
        if p_values[key] > alpha / (m - i):
            break
        reject[key] = True
    return reject


def summary(h124, rows, packets):
    families, reference = h124.family_values(rows, packets, CONTROL)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(families), size=(BOOTSTRAP_DRAWS, len(families))
    )
    arms, arrays = {}, {}
    for arm in PRIMARY + REPORT_ONLY:
        _, values = h124.family_values(rows, packets, arm)
        gain = values - reference
        per_seed = gain.mean(axis=1)
        boot = np.median(gain[:, indices].mean(axis=2), axis=0)
        harms = gain.mean(axis=0) < HARM_DELTA
        arms[arm] = {
            "per_seed_mean_gain": per_seed.tolist(),
            "median_seed_gain": float(np.median(per_seed)),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": float(np.quantile(boot, 0.05)),
            "p_one_sided_vs_margin": float(
                (1 + np.sum(boot <= LCB_MARGIN)) / (BOOTSTRAP_DRAWS + 1)
            ),
            "harm_count": int(harms.sum()),
            "harm_fraction": float(harms.mean()),
            "family_count": len(families),
        }
        arrays[arm] = gain.round(8).tolist()
    reject = holm({arm: arms[arm]["p_one_sided_vs_margin"] for arm in PRIMARY})
    for arm in PRIMARY:
        s = arms[arm]
        s["holm_reject"] = reject[arm]
        s["admitted"] = bool(
            s["median_seed_gain"] > 0 and reject[arm] and s["harm_fraction"] <= HARM_CAP
        )
        s["elo"] = h124.elo_h124(rows, packets, arm, rng_seed=ELO_BOOTSTRAP_SEED)
    arms["PURE_GBM"]["admitted"] = None
    arms["PURE_GBM"]["elo"] = h124.elo_h124(
        rows, packets, "PURE_GBM", rng_seed=ELO_BOOTSTRAP_SEED
    )
    for arm in (
        PRIMARY + REPORT_ONLY
    ):  # keep metrics compact: the registered headline and its interval only
        e = arms[arm]["elo"]
        arms[arm]["elo"] = {
            "registered_estimator": e["registered_estimator"],
            "median_seed_elo": e["median_seed_elo"],
            "pooled_score_elo_additional": e["pooled_score_elo_additional"],
            "reference_elo": e["reference_elo"],
        }
    gate_open = {
        str(seed): float(
            np.mean(
                [packets[str(seed)][r["context_sha256"]]["gate_open"] for r in rows]
            )
        )
        for seed in SEEDS
    }
    passed = [arm for arm in PRIMARY if arms[arm]["admitted"]]
    return {
        "arms": arms,
        "holm": {
            "alpha": HOLM_ALPHA,
            "m": len(PRIMARY),
            "margin": LCB_MARGIN,
            "reject": reject,
        },
        "gate_open_fraction_by_seed": gate_open,
        "verdict": ("CANDIDATE: " + ", ".join(passed)) if passed else "NO_ADVANCE",
        "candidate_breakthrough": bool(passed),
    }, {"families": families, "family_gain_by_seed": arrays}


def protocol():
    return {
        "wave": "H129",
        "fold_seed": FOLD_SEED,
        "sampling_seed": SAMPLE_SEED,
        "seeds": list(SEEDS),
        "lambdas": LAMBDAS,
        "gated_arm": GATED,
        "gate_quantiles": list(GATE_QUANTILES),
        "fit_harm_target": FIT_HARM_TARGET,
        "primary": list(PRIMARY),
        "report_only": list(REPORT_ONLY),
        "harm_delta": HARM_DELTA,
        "harm_cap": HARM_CAP,
        "lcb_margin": LCB_MARGIN,
        "holm_alpha": HOLM_ALPHA,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "elo_bootstrap_seed": ELO_BOOTSTRAP_SEED,
        "h124_module_sha256": H124_SHA,
        "fit_count": 45,
    }


def protocol_sha256():
    return hashlib.sha256(
        json.dumps(protocol(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def host_limits(path):
    """H129 limits on top of H124's: MemAvailable >= 30 GiB and filesystem use < 94%."""
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
        raise RuntimeError(f"disk use {used:.3f} >= 94%")
    return {"mem_available_kib": available, "disk_used_fraction": round(used, 4)}


def compact_metrics(metrics):
    """Serialize and size-check before anything is written (the H124 defect)."""
    raw = json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if len(raw.encode()) >= METRICS_LIMIT_BYTES:
        raise RuntimeError(f"H129 metrics would be {len(raw.encode())} bytes >= 100 KB")
    return raw


def verify_prereg_committed(source_commit):
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", H129_PREREG_COMMIT, source_commit],
        cwd=REPO,
        check=True,
    )
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{H129_PREREG_COMMIT}:research/flywheel_selfplay/h129_gbm_blend_harm_gate/PREREG.md",
        ],
        cwd=REPO,
    )
    if expected != (HERE / "PREREG.md").read_bytes():
        raise RuntimeError("H129 PREREG bytes changed after the preregistration commit")
    return hashlib.sha256(expected).hexdigest()


def run(root):
    root = Path(root)
    parent = Path.home() / "jope-data/h129"
    if root.exists() or root.parent != parent or not root.name.startswith("run-"):
        raise RuntimeError("H129 requires a fresh run-* root under ~/jope-data/h129")
    h124 = load_h124()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    prereg_sha = verify_prereg_committed(source_commit)
    parent.mkdir(parents=True, exist_ok=True)
    launch = dict(h124.preflight(parent), **host_limits(parent))
    root.mkdir()
    t0, cpu0, run_sha = time.monotonic(), time.process_time(), sha(Path(__file__))
    h124.write(
        root / "execution-identity.json",
        {
            "source_commit": source_commit,
            "prereg_commit": H129_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "h124_module_sha256": H124_SHA,
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
            raise RuntimeError("H129 full pinned DL-Train roster required")
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
            host_limits(root)
            fits.append(
                h124.fit_one(task, root, rows, context, pools, samples, mapping)
            )
            h124.headroom(root)
        if len(fits) != 45:
            raise RuntimeError("H129 requires 45 fit receipts")
        packets, gate_receipts = tree_packets(h124, root, rows, pools, baseline)
        result, arrays = summary(h124, rows, packets)
        dope = json.loads((root / "dope-train.json").read_text())
        if (
            dope["before"] != h124.DOPE
            or dope["after"] != h124.DOPE
            or sha(Path(__file__)) != run_sha
        ):
            raise RuntimeError("H129 frozen DOPE/source guard failed")
        arrays_raw = json.dumps(arrays, sort_keys=True).encode()
        import sklearn
        import torch

        metrics = {
            "schema": "jope.h129.gbm-blend-harm-gate/v1",
            "status": "COMPLETE",
            "verdict": result["verdict"],
            "candidate_breakthrough": result["candidate_breakthrough"],
            "protocol": protocol(),
            "protocol_sha256": protocol_sha256(),
            "source_commit": source_commit,
            "prereg_commit": H129_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "arms": result["arms"],
            "holm": result["holm"],
            "gate_open_fraction_by_seed": result["gate_open_fraction_by_seed"],
            "gates_by_seed_fold": {
                k: {
                    "calibration": v["calibration"],
                    "tau": v["gate"]["tau"],
                    "quantile": v["gate"]["quantile"],
                }
                for k, v in gate_receipts.items()
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
        raw = compact_metrics(metrics)  # size checked before any result file is written
        (root / "family_arrays.json").write_bytes(arrays_raw)
        (root / "gate_receipts.json").write_text(
            json.dumps(gate_receipts, indent=1, sort_keys=True, default=str)
        )
        (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def cpu_dry_run():
    """Synthetic controls only: no H50/H58/H51 reads, no fits."""
    rng = np.random.default_rng(0)
    r882 = rng.standard_normal(40)
    gbm = r882 + 0.1 * rng.standard_normal(40)
    if not 0 <= disagreement(gbm, r882) < disagreement(-gbm, r882):
        raise RuntimeError("disagreement ordering failed")
    if holm({"a": 0.01, "b": 0.02, "c": 0.04}) != {"a": True, "b": True, "c": True}:
        raise RuntimeError("Holm control failed")
    return {
        "schema": "jope.h129.cpu-dry-run/v1",
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
