#!/usr/bin/env python3
"""H139 winner's-curse / selection-rule audit of r882 vs retained flywheel adapters (zero optimizer updates, CPU).

Each arm is a sha-pinned copy of a flywheel checkpoint taken at claim time. All arms share the frozen DOPE and
differ only in the 29,211 trainable adapter parameters. The merged H124 module (byte-pinned) supplies the frozen
r882-style CPU forward pass (`baseline`), the train-row-only feature reads and the RSS watchdog; H139 supplies the
roster, the native flywheel endpoint, the readout and the decision. No parameter is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
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
H139_PREREG_COMMIT = "__PREREG_COMMIT__"
INPUT = Path("/home/ubuntu/jope-data/h50/input")
ARM_DIR = Path("/home/ubuntu/jope-data/h139/inputs")
ARMS = {  # name: (file, sha256) — copied at claim time 2026-10-31T23:40:37Z, chosen by round number
    "R882": (
        "R882.pt",
        "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    ),
    "R782": (
        "R782.pt",
        "427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c",
    ),
    "R859": (
        "R859.pt",
        "5802245f447b1852610576cde08b8a299a9e1e99eb82991aac63c958284fa715",
    ),
    "R1703": (
        "R1703.pt",
        "8614dff5276fc6c9377d8bae5d6ebfc686409ed9ae8aa04b3fb4cd917720c957",
    ),
    "R2481": (
        "R2481.pt",
        "d922f331a458ca5a59bc98da252d3edf1864ce6a1b8bf5fc36713bb8907deac9",
    ),
    "REJ": (
        "REJ-round-2482.pt",
        "8ff38df8c845200e6fdfabe206324ab8194a7424b20b3b60e7f100ed3c3faba5",
    ),
}
REFERENCE = (
    "best_dev_checkpoint.json",
    "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a",
)
EVALUATION_IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
ANCHOR = "R882"
DEGENERATE = (
    "R1703",
    "R2481",
)  # adapters bitwise identical to R882 (structural, pre-outcome)
DISTINCT = ("R782", "R859", "REJ")
RECORDED_VAL = {  # adapter_admissions.jsonl candidate_utility for the same parameters (metadata)
    "R882": 1.1377941898164525,
    "R782": 1.1073772627425547,
    "R859": 1.1377941898164525,
    "REJ": 1.1354387884248622,
}
FOLD_SEEDS = (139191, 139192, 139193)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 139000
ELO_BOOTSTRAP_DRAWS = 1_000
ELO_BOOTSTRAP_SEED = 139001
SPEARMAN_SEED = 139002
HOLM_ALPHA = 0.05
HARM_DELTA = -0.01
DISK_FREE_MIN_BYTES = 40 * 10**9
MIN_NICE = 19
METRICS_LIMIT_BYTES = 100_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_h124():
    if sha(H124_PATH) != H124_SHA:
        raise RuntimeError("H124 module differs from the pinned bytes")
    spec = importlib.util.spec_from_file_location("h139_h124", H124_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- endpoint and statistics (pure, unit-tested)


def native(cost, fitness, gate):
    """Flywheel protocol fixed-dev-cost-argmin-negative-fitness/v1: first argmin over the full pool; -2 if invalid."""
    cost, fitness, gate = np.asarray(cost), np.asarray(fitness), np.asarray(gate)
    if (
        cost.ndim != 1
        or cost.size == 0
        or cost.shape != fitness.shape
        or not np.isfinite(cost).all()
    ):
        raise ValueError("invalid complete pool")
    i = int(np.argmin(cost))
    return float(-fitness[i] if gate[i] else -2.0)


def family_matrix(values, families):
    """values: {arm: per-context array}; returns (family list, {arm: per-family mean array})."""
    order = sorted(set(families))
    index = {f: k for k, f in enumerate(order)}
    out = {}
    for arm, v in values.items():
        sums, counts = np.zeros(len(order)), np.zeros(len(order))
        for x, f in zip(v, families):
            sums[index[f]] += x
            counts[index[f]] += 1
        out[arm] = sums / counts
    return order, out


def folds(families, seed):
    fam = sorted(set(families))
    return [
        sorted(g.tolist())
        for g in np.array_split(np.random.default_rng(seed).permutation(fam), 5)
    ]


def gain_stats(gain, samples):
    """gain: per-family arm - R882. Equal-family mean, percentile 95% CI, two-sided bootstrap p, harm."""
    boot = gain[samples].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = min(
        1.0,
        2
        * min(
            (1 + np.sum(boot <= 0)) / (len(boot) + 1),
            (1 + np.sum(boot >= 0)) / (len(boot) + 1),
        ),
    )
    if np.all(gain == 0):
        p = 1.0
    return {
        "mean_gain": float(gain.mean()),
        "median_family_gain": float(np.median(gain)),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(p),
        "harm_fraction": float((gain < HARM_DELTA).mean()),
        "families_win_tie_loss": [
            int((gain > 1e-12).sum()),
            int((np.abs(gain) <= 1e-12).sum()),
            int((gain < -1e-12).sum()),
        ],
    }


def holm(p_values, alpha=HOLM_ALPHA):
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m, reject = len(order), {k: False for k in p_values}
    for i, key in enumerate(order):
        if p_values[key] > alpha / (m - i):
            break
        reject[key] = True
    return reject


def spearman(a, b):
    def ranks(x):
        x = np.asarray(x, np.float64)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=r) / counts)[inv]

    ra, rb = ranks(a), ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def selection_optimism(family_means, families_order, seed, arms_in_round_order):
    """Cross-fit argmax audit: pick the best arm on four folds, score it on the fifth (ties -> earliest round)."""
    index = {f: k for k, f in enumerate(families_order)}
    out = []
    for k, held in enumerate(folds(families_order, seed)):
        held_idx = np.array([index[f] for f in held])
        train_idx = np.setdiff1d(np.arange(len(families_order)), held_idx)
        insample = {
            a: float(family_means[a][train_idx].mean()) for a in arms_in_round_order
        }
        best = max(insample.values())
        chosen = next(a for a in arms_in_round_order if insample[a] == best)
        ref_in = insample[ANCHOR]
        out.append(
            {
                "fold": k,
                "chosen": chosen,
                "in_sample_gain": insample[chosen] - ref_in,
                "held_out_gain": float(
                    family_means[chosen][held_idx].mean()
                    - family_means[ANCHOR][held_idx].mean()
                ),
            }
        )
    return out


def regression_check(values, reference):
    """R882 must reproduce the live loop's best_dev reference: every one of 91 contexts within 1e-6 and the mean
    1.1377942 within 1e-6 (the H135 baseline gate)."""
    values = np.asarray(values, np.float64)
    expected = np.asarray(reference["utility"]["per_context"], np.float64)
    if (
        values.shape != (91,)
        or expected.shape != (91,)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("R882 regression needs 91 finite contexts")
    err = float(np.max(np.abs(values - expected)))
    mean_err = abs(float(values.mean()) - reference["best_utility"])
    matched = int(np.sum(np.abs(values - expected) <= 1e-6))
    if matched != 91 or mean_err > 1e-6 or abs(float(values.mean()) - 1.1377942) > 1e-6:
        raise RuntimeError(
            f"R882 regression FAIL (matched {matched}/91, max error {err}); stop before readout"
        )
    return {
        "status": "PASS",
        "contexts_matched": matched,
        "max_abs_error": err,
        "mean": float(values.mean()),
        "mean_error": mean_err,
    }


def decide(arm_stats, spearman_ci):
    """SELECTION_ARTEFACT if >=1 distinct arm's 95% CI covers or exceeds 0 AND the val-vs-cross-fit Spearman CI
    includes 0 (or is undefined); else PLATEAU_REAL."""
    covers = [a for a in DISTINCT if arm_stats[a]["ci95"][1] >= 0]
    lo, hi = spearman_ci
    includes_zero = (not (math.isfinite(lo) and math.isfinite(hi))) or (lo <= 0 <= hi)
    return (
        ("SELECTION_ARTEFACT" if covers and includes_zero else "PLATEAU_REAL"),
        covers,
        includes_zero,
    )


# ---------------------------------------------------------------- run


def guards(path):
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H139 requires nice >= 19")
    free = shutil.disk_usage(path).free
    if free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(f"host free {free} B < 40 GB")
    return {"disk_free_bytes": free}


def verify_prereg_committed(source_commit):
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", H139_PREREG_COMMIT, source_commit],
        cwd=REPO,
        check=True,
    )
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{H139_PREREG_COMMIT}:research/flywheel_selfplay/h139_selection_audit/PREREG.md",
        ],
        cwd=REPO,
    )
    if expected != (HERE / "PREREG.md").read_bytes():
        raise RuntimeError("H139 PREREG bytes changed after the preregistration commit")
    return hashlib.sha256(expected).hexdigest()


def roster(h124):
    # Pinned dependencies, without opening source_manifest or cache_manifest (as H135).
    for path, digest in h124.FILES.items():
        if path.name in ("source_manifest.json", "cache_manifest.json", "round-882.pt"):
            continue
        if sha(path) != digest:
            raise RuntimeError(f"dependency hash mismatch: {path}")
    selected = INPUT / "selected.json"
    if sha(selected) != h124.FILES[selected]:
        raise RuntimeError("selected packet changed")
    packet = json.loads(selected.read_text())
    train, val = packet["train"], packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("cohort count mismatch")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            for r in rows
        ):
            raise RuntimeError("unauthorized row")
    reference_path = ARM_DIR / REFERENCE[0]
    if sha(reference_path) != REFERENCE[1]:
        raise RuntimeError("reference contract changed")
    reference = json.loads(reference_path.read_text())
    contract = reference["evaluation_contract"]
    if (
        reference["evaluation_identity"] != EVALUATION_IDENTITY
        or hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        != EVALUATION_IDENTITY
        or [c["cid"] for c in contract["contexts"]]
        != [r["context_sha256"] for r in val]
    ):
        raise RuntimeError("validation contract mismatch")
    return train, val, reference


def pools_for(rows):
    pools = {}
    for r in rows:
        path = INPUT / "pools" / f"{r['context_sha256']}.npz"
        if sha(path) != r["pool_sha256"]:
            raise RuntimeError("pool identity mismatch")
        with np.load(path, allow_pickle=False) as z:
            fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        pools[r["context_sha256"]] = dict(
            path=path,
            pool_sha256=r["pool_sha256"],
            fitness=fit,
            gate=gate,
            u=np.where(gate, -fit, -2.0),
        )
    return pools


def structural_check(h124):
    """Every arm: pinned bytes, DOPE identity field, non-adapter tensors bitwise equal to R882."""
    import torch

    states, receipt = {}, {}
    for arm, (name, digest) in ARMS.items():
        path = ARM_DIR / name
        if sha(path) != digest:
            raise RuntimeError(f"{arm} checkpoint changed")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("dope_sha256") != h124.DOPE:
            raise RuntimeError(f"{arm} DOPE identity mismatch")
        states[arm] = ck["state_dict"]
    ref = states[ANCHOR]
    for arm, sd in states.items():
        if set(sd) != set(ref):
            raise RuntimeError(f"{arm} parameter set differs")
        nonadapter = [
            k
            for k in sd
            if not k.startswith("adapter_10tok.") and not torch.equal(sd[k], ref[k])
        ]
        if nonadapter:
            raise RuntimeError(f"{arm}: non-adapter tensors differ from R882")
        diff = sum(
            int((sd[k] != ref[k]).sum()) for k in sd if k.startswith("adapter_10tok.")
        )
        receipt[arm] = {"adapter_elements_differing_from_R882": diff}
        if (arm in DEGENERATE) != (diff == 0 and arm != ANCHOR):
            raise RuntimeError(
                f"{arm}: degenerate/distinct classification differs from the preregistration"
            )
    return receipt


def run(root):
    root = Path(root)
    parent = Path.home() / "jope-data/h139"
    if root.exists() or root.parent != parent or not root.name.startswith("run-"):
        raise RuntimeError("H139 requires a fresh run-* root under ~/jope-data/h139")
    h124 = load_h124()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    prereg_sha = verify_prereg_committed(source_commit)
    launch = dict(h124.preflight(parent), **guards(parent))
    root.mkdir()
    t0, cpu0, run_sha = time.monotonic(), time.process_time(), sha(Path(__file__))
    h124.write(
        root / "execution-identity.json",
        {
            "source_commit": source_commit,
            "prereg_commit": H139_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "h124_module_sha256": H124_SHA,
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
        structure = structural_check(h124)
        train, val, reference = roster(h124)
        rows = (
            val + train
        )  # one forward pass per arm; val first so the regression runs on R882 before anything else
        context, reads = h124.train_context(rows, [])
        projection = h124.train_projection(rows, [])
        pools = pools_for(rows)
        native_val, native_train, u4_train, dope = {}, {}, {}, {}
        for arm, (name, digest) in ARMS.items():
            guards(root)
            h124.headroom(root)
            path = ARM_DIR / name
            h124.R882 = path
            h124.FILES[path] = digest
            arm_root = root / arm
            arm_root.mkdir()
            costs, net = h124.baseline(arm_root, rows, context, pools, projection)
            del net
            h124.release_memory()
            dope[arm] = json.loads((arm_root / "dope-secondary.json").read_text())
            (
                arm_root / "r882-copy.pt"
            ).unlink()  # byte copy of the pinned arm checkpoint (scratch only)
            native_val[arm] = [
                native(
                    costs[r["context_sha256"]],
                    pools[r["context_sha256"]]["fitness"],
                    pools[r["context_sha256"]]["gate"],
                )
                for r in val
            ]
            native_train[arm] = [
                native(
                    costs[r["context_sha256"]],
                    pools[r["context_sha256"]]["fitness"],
                    pools[r["context_sha256"]]["gate"],
                )
                for r in train
            ]
            u4_train[arm] = [
                h124.endpoints(
                    costs[r["context_sha256"]], pools[r["context_sha256"]]["u"]
                )["u4"]
                for r in train
            ]
            if (
                arm == ANCHOR
            ):  # regression first: exact 91/91 parity with the live loop's best_dev reference
                regression = regression_check(native_val[arm], reference)
        for arm in DEGENERATE:
            if (
                native_train[arm] != native_train[ANCHOR]
                or native_val[arm] != native_val[ANCHOR]
            ):
                raise RuntimeError(
                    f"{arm} outputs differ from R882 despite identical adapters"
                )
        result = readout(h124, train, val, native_train, native_val, u4_train)
        if any(
            d["before"] != h124.DOPE or d["after"] != h124.DOPE for d in dope.values()
        ):
            raise RuntimeError("DOPE identity changed")
        per_context = {
            "train_cids": [r["context_sha256"] for r in train],
            "val_cids": [r["context_sha256"] for r in val],
            "native_train": native_train,
            "native_val": native_val,
            "u4_train": u4_train,
        }
        per_raw = json.dumps(per_context, sort_keys=True).encode()
        import sklearn
        import torch

        metrics = {
            "schema": "jope.h139.selection-audit/v1",
            "status": "COMPLETE",
            "verdict": result.pop("verdict"),
            "source_commit": source_commit,
            "prereg_commit": H139_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "h124_module_sha256": H124_SHA,
            "arms": {a: {"file": f, "sha256": s} for a, (f, s) in ARMS.items()},
            "structure": structure,
            "regression": dict(regression, reference_sha256=REFERENCE[1]),
            **result,
            "per_context_sha256": hashlib.sha256(per_raw).hexdigest(),
            "dope": {
                a: {
                    "before": d["before"],
                    "after": d["after"],
                    "optimizer_updates": d["optimizer_updates"],
                }
                for a, d in dope.items()
            },
            "read_boundary": {
                "dl_train_contexts": len(train),
                "dl_val_contexts_flywheel_contract": len(val),
                "dl_val_cache_rows_decoded_beyond_contract": 0,
                "sealed_payloads_opened": 0,
                "holdout_payloads_opened": 0,
                "meta_test_payloads_opened": 0,
                "live_dir_writes": 0,
            },
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
            "optimizer_updates": 0,
            "GPU_used": False,
            "CUDA_initialized": torch.cuda.is_initialized(),
        }
        raw = json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if len(raw.encode()) >= METRICS_LIMIT_BYTES:
            raise RuntimeError("H139 metrics would exceed 100 KB")
        (root / "per_context.json").write_bytes(per_raw)
        (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def readout(h124, train, val, native_train, native_val, u4_train):
    families = [r["family_id"] for r in train]
    order, fm = family_matrix(native_train, families)
    _, fm_u4 = family_matrix(u4_train, families)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(order), size=(BOOTSTRAP_DRAWS, len(order))
    )
    arms = {}
    for arm in ARMS:
        if arm == ANCHOR:
            continue
        s = gain_stats(fm[arm] - fm[ANCHOR], samples)
        s["fold_seed_mean_gain"] = {
            str(fs): s["mean_gain"] for fs in FOLD_SEEDS
        }  # deterministic arms: identical
        s["median_over_fold_seeds"] = s["mean_gain"]
        s["u4_mean_gain_descriptive"] = float((fm_u4[arm] - fm_u4[ANCHOR]).mean())
        s["degenerate_identical_to_R882"] = arm in DEGENERATE
        arms[arm] = s
    reject = holm({a: arms[a]["p_two_sided"] for a in DISTINCT})
    for a in DISTINCT:
        arms[a]["holm_reject"] = reject[a]
    # Secondary: flywheel 91-context validation values and their rank agreement with the DL-Train cross-fit.
    val_families = [r["family_id"] for r in val]
    val_mean = {a: float(np.mean(native_val[a])) for a in ARMS}
    ranked = (ANCHOR,) + DISTINCT
    rho = spearman([val_mean[a] for a in ranked], [float(fm[a].mean()) for a in ranked])
    vorder = sorted(set(val_families))
    vidx = {f: [i for i, g in enumerate(val_families) if g == f] for f in vorder}
    rng = np.random.default_rng(SPEARMAN_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        ts = rng.integers(0, len(order), len(order))
        vs = rng.integers(0, len(vorder), len(vorder))
        vi = [i for f in vs for i in vidx[vorder[f]]]
        draws.append(
            spearman(
                [float(np.mean(np.asarray(native_val[a])[vi])) for a in ranked],
                [float(fm[a][ts].mean()) for a in ranked],
            )
        )
    finite = np.asarray([d for d in draws if math.isfinite(d)])
    rho_ci = (
        [float(x) for x in np.quantile(finite, [0.025, 0.975])]
        if len(finite)
        else [math.nan, math.nan]
    )
    optimism = {
        str(fs): selection_optimism(fm, order, fs, tuple(ARMS)) for fs in FOLD_SEEDS
    }
    flat = [x for v in optimism.values() for x in v]
    verdict, covers, includes_zero = decide(arms, rho_ci)
    return {
        "verdict": verdict,
        "decision": {
            "arms_ci_covering_or_exceeding_zero": covers,
            "spearman_ci_includes_zero": includes_zero,
        },
        "primary_dl_train": arms,
        "holm": {"alpha": HOLM_ALPHA, "arms": list(DISTINCT), "reject": reject},
        "elo": elo(h124, train, native_train),
        "secondary_val": {
            "val_mean_native": val_mean,
            "recorded_val_candidate_utility": RECORDED_VAL,
            "recomputed_minus_recorded": {
                a: val_mean[a] - RECORDED_VAL[a] for a in RECORDED_VAL
            },
            "spearman_val_vs_dl_train": rho,
            "spearman_ci95": rho_ci,
            "spearman_finite_draws": int(len(finite)),
            "ranked_arms": list(ranked),
        },
        "cross_fit_selection_optimism": {
            "per_fold_seed": optimism,
            "mean_in_sample_gain": float(np.mean([x["in_sample_gain"] for x in flat])),
            "mean_held_out_gain": float(np.mean([x["held_out_gain"] for x in flat])),
        },
    }


def elo(h124, train, native_train):
    helper = h124.source_module("python/src/jope/north_star/elo_tracker.py", "h139_elo")
    contenders = (ANCHOR,) + DISTINCT

    rows_by_family = defaultdict(list)
    for j, r in enumerate(train):
        rows_by_family[r["family_id"]].append((j, r))

    def ratings(families_subset):
        recs = []
        for i, f in enumerate(families_subset):
            for j, r in rows_by_family[f]:
                recs.append(
                    helper.DatasetEvalRecord(
                        f"{i}:{r['context_sha256']}",
                        f"{i}:{f}",
                        "regression",
                        {a: native_train[a][j] for a in contenders},
                    )
                )
        return helper.compute_family_grouped_bradley_terry_elo(
            recs, ANCHOR, anchor_elo=1500.0, tie_margin=1e-4, prior=0.01, n_iters=400
        )["ratings"]

    order = sorted({r["family_id"] for r in train})
    point = ratings(order)
    rng = np.random.default_rng(ELO_BOOTSTRAP_SEED)
    draws = defaultdict(list)
    for _ in range(ELO_BOOTSTRAP_DRAWS):
        for a, v in ratings(
            [order[k] for k in rng.integers(0, len(order), len(order))]
        ).items():
            draws[a].append(v)
    return {
        "ratings": point,
        "ci95": {
            a: [float(x) for x in np.quantile(v, [0.025, 0.975])]
            for a, v in draws.items()
        },
        "anchor": ANCHOR,
        "bootstrap_draws": ELO_BOOTSTRAP_DRAWS,
        "degenerate_arms_equal_anchor": list(DEGENERATE),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"verdict": run(args.root)["verdict"]}))


if __name__ == "__main__":
    main()
