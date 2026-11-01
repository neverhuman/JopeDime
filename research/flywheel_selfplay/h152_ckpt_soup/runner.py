#!/usr/bin/env python3
"""H152 zero-update checkpoint soup of saved flywheel adapters (CPU only, 0 optimizer updates).

Weight-space blends of already-saved flywheel rounds, scored with the merged H139 evaluator (byte-pinned; it in turn
uses the pinned H124 frozen CPU forward pass and the native flywheel endpoint). Only the 47 trainable adapter tensors
(29,211 parameters) are averaged; every other tensor and buffer is taken from R882 unchanged (and verified identical
across the four inputs).

Arms: R882 (regression gate), MID = 0.5 R782 + 0.5 R882 (single primary), SOUP4 = uniform mean of R782/R785/R859/R882
(secondary), A025 = 0.25 R782 + 0.75 R882 and A075 = 0.75 R782 + 0.25 R882 (exploratory, never admitted).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
H139_PATH = HERE.parent / "h139_selection_audit" / "runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
PREREG_SHA = "c3570fa133a285f0134db94d271e000b0ca790f532e00dd3815af7707963d3b6"  # PREREG.md bytes as reviewed (a content hash survives a central squash merge)
BOUND_FILES = (  # executed bytes must equal both the reviewed head and the merged source commit
    "research/flywheel_selfplay/h152_ckpt_soup/PREREG.md",
    "research/flywheel_selfplay/h152_ckpt_soup/runner.py",
    "research/flywheel_selfplay/h152_ckpt_soup/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
INPUTS = Path("/home/ubuntu/jope-data/h152/inputs")
SOURCES = {  # round: (file, sha256) — one-time read-only copies from xbabe1 run-20261030 at 2026-11-01T03:24:28Z
    782: (
        "R782.pt",
        "427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c",
    ),
    785: (
        "R785.pt",
        "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a",
    ),
    859: (
        "R859.pt",
        "5802245f447b1852610576cde08b8a299a9e1e99eb82991aac63c958284fa715",
    ),
    882: (
        "R882.pt",
        "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    ),
}
ARMS = {  # arm: {round: weight}
    "R882": {882: 1.0},
    "MID": {782: 0.5, 882: 0.5},
    "SOUP4": {782: 0.25, 785: 0.25, 859: 0.25, 882: 0.25},
    "A025": {782: 0.25, 882: 0.75},
    "A075": {782: 0.75, 882: 0.25},
}
PRIMARY = "MID"
SECONDARY = "SOUP4"
ADMISSIBLE = (PRIMARY, SECONDARY)
EXPLORATORY = ("A025", "A075")
ANCHOR = "R882"
SEEDS = (152101, 152102, 152103)
DRAWS = 10_000
HOLM_ALPHA = 0.05
TRAINABLE_TENSORS = 47
TRAINABLE_PARAMETERS = 29_211
RSS_LIMIT_BYTES = 6_000_000_000
DISK_USED_MAX = 0.95  # absolute guard (xbabe1 sits at 94%): stop at >= 95% used
DISK_FREE_MIN_BYTES = 40_000_000_000  # ... or < 40 GB free
SCRATCH_LIMIT_BYTES = 1_000_000_000  # the whole ~/jope-data/h152 tree, inputs included
TORCH_THREADS = 1  # + main thread + 1-thread supervisor stays within the 4-thread process-tree budget
LOAD_MAX = 85.0
MIN_NICE = 19
METRICS_LIMIT_BYTES = 100_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_h139():
    if sha(H139_PATH) != H139_SHA:
        raise RuntimeError("H139 runner differs from the pinned bytes")
    spec = importlib.util.spec_from_file_location("h152_h139", H139_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ARM_DIR = INPUTS  # the reference contract copy lives with the H152 inputs (same pinned bytes)
    return module


# ------------------------------------------------------------------ soups (pure tensor math, unit-tested)


def blend(states, weights, trainable):
    """New state = R882's state with only `trainable` tensors replaced by the weighted average."""
    import torch

    if abs(sum(weights.values()) - 1.0) > 1e-12 or any(
        w <= 0 for w in weights.values()
    ):
        raise ValueError("soup weights must be positive and sum to 1")
    out = {k: v.clone() for k, v in states[882].items()}
    for name in trainable:
        acc = torch.zeros_like(states[882][name], dtype=torch.float64)
        for rnd, w in sorted(weights.items()):
            acc += w * states[rnd][name].to(torch.float64)
        out[name] = acc.to(states[882][name].dtype)
    return out


def frozen_identity(states, trainable):
    """Every non-trainable tensor/buffer must be bitwise identical across the inputs (else listed, R882 used)."""
    ref = states[882]
    differing = sorted(
        k
        for k in ref
        if k not in trainable
        and any(
            states[r][k].dtype != ref[k].dtype
            or states[r][k].shape != ref[k].shape
            or states[r][k].numpy().tobytes() != ref[k].numpy().tobytes()
            for r in states
        )
    )
    return differing


def gain_readout(gain, seeds=SEEDS, draws=DRAWS):
    """gain: per-family arm - R882 (equal-family). Point estimate is deterministic; each seed re-draws the 19-family
    bootstrap. Returns per-seed CI/p and the medians across seeds."""
    gain = np.asarray(gain, np.float64)
    per_seed = {}
    for s in seeds:
        idx = np.random.default_rng(s).integers(0, len(gain), size=(draws, len(gain)))
        boot = gain[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        per_seed[str(s)] = {
            "ci95": [float(lo), float(hi)],
            "p_one_sided": float((1 + np.sum(boot <= 0)) / (draws + 1)),
        }
    los = [v["ci95"][0] for v in per_seed.values()]
    his = [v["ci95"][1] for v in per_seed.values()]
    ps = [v["p_one_sided"] for v in per_seed.values()]
    score = np.where(gain > 1e-12, 1.0, np.where(gain < -1e-12, 0.0, 0.5))
    wins = float(score.sum())
    return {
        "gain": float(gain.mean()),
        "median_family_gain": float(np.median(gain)),
        "families_win_tie_loss": [
            int((gain > 1e-12).sum()),
            int((np.abs(gain) <= 1e-12).sum()),
            int((gain < -1e-12).sum()),
        ],
        "per_seed": per_seed,
        "median_ci95": [float(np.median(los)), float(np.median(his))],
        "median_p_one_sided": float(np.median(ps)),
        "elo": float(1500 + 400 * np.log10((wins + 0.01) / (len(gain) - wins + 0.01))),
    }


def elo_ci(gain, seed=SEEDS[0], draws=1000):
    gain = np.asarray(gain, np.float64)
    score = np.where(gain > 1e-12, 1.0, np.where(gain < -1e-12, 0.0, 0.5))
    idx = np.random.default_rng(seed).integers(0, len(gain), size=(draws, len(gain)))
    w = score[idx].sum(axis=1)
    e = 1500 + 400 * np.log10((w + 0.01) / (len(gain) - w + 0.01))
    return [float(x) for x in np.quantile(e, [0.025, 0.975])]


def decide(primary, secondary):
    """Holm over MID and SOUP4 on the median one-sided p; ADVANCE also needs a strictly positive point gain and a
    strictly positive median CI lower bound. Ties (zero gain) reject. Exploratory arms are never admitted."""
    p = {
        PRIMARY: primary["median_p_one_sided"],
        SECONDARY: secondary["median_p_one_sided"],
    }
    out, adjusted = {}, 0.0
    for rank, arm in enumerate(sorted(p, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[arm] * (len(p) - rank)))
        r = primary if arm == PRIMARY else secondary
        gates = {
            "gain_gt_0": r["gain"] > 0,
            "median_ci_lb_gt_0": r["median_ci95"][0] > 0,
            "holm": adjusted < HOLM_ALPHA,
        }
        out[arm] = {
            "holm_adjusted_p": adjusted,
            "gates": gates,
            "advance": all(gates.values()),
        }
    return out


# ------------------------------------------------------------------ guards and identity


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def guards(path):
    """Stop (never pause) on nice, disk, load or scratch; called at launch and before every arm. The external
    supervisor (monitor.py) applies the same limits continuously, plus process-tree threads and RSS."""
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H152 requires nice >= 19")
    usage = shutil.disk_usage(path)
    used = 1 - usage.free / usage.total
    if used >= DISK_USED_MAX or usage.free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(f"disk stop: used {used:.4f}, free {usage.free} B")
    load = os.getloadavg()[0]
    if load > LOAD_MAX:
        raise RuntimeError(f"load stop: {load:.1f} > 85")
    scratch = tree_bytes(path)
    if scratch >= SCRATCH_LIMIT_BYTES:
        raise RuntimeError(f"scratch stop: {scratch} B >= 1 GB")
    return {
        "disk_used_fraction": round(used, 4),
        "disk_free_bytes": usage.free,
        "load1": load,
        "scratch_bytes": scratch,
    }


def bind_source(launch, repo=REPO):
    """Squash-compatible binding. Every bound file's executed bytes must equal its blob at the reviewed head AND at the
    merged source commit (content, not ancestry: a central squash merge keeps blobs but drops the author commits).
    PREREG.md must also hash to the preregistered PREREG_SHA."""
    reviewed, source = launch["reviewed_head"], launch["source_commit"]
    if sha(repo / BOUND_FILES[0]) != PREREG_SHA:
        raise RuntimeError("PREREG bytes differ from the preregistration")
    bound = {}
    for rel in BOUND_FILES:
        local = (repo / rel).read_bytes()
        for commit in (reviewed, source):
            if (
                subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=repo)
                != local
            ):
                raise RuntimeError(f"{rel}: executed bytes differ from {commit}")
        bound[rel] = hashlib.sha256(local).hexdigest()
    return {"reviewed_head": reviewed, "source_commit": source, "bound_sha256": bound}


def bounded_threads(fn, *args, **kwargs):
    """The byte-pinned H124 baseline hardcodes torch.set_num_threads(4); hold the pool at TORCH_THREADS instead."""
    from unittest.mock import patch

    import torch

    setter = torch.set_num_threads

    def capped(requested):
        setter(TORCH_THREADS)

    setter(TORCH_THREADS)
    with patch.object(torch, "set_num_threads", capped):
        return fn(*args, **kwargs)


def trainable_names(h124):
    """The 47 non-anchor adapter parameters (29,211 elements), read from the frozen fused network's module tree."""
    import torch

    spec = importlib.util.spec_from_file_location(
        "h152_fused", h124.INPUT / "cpu_exact_fused.py"
    )
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    names = tuple(
        n
        for n, _ in net.named_parameters()
        if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
    )
    sizes = dict((n, p.numel()) for n, p in net.named_parameters())
    if (
        len(names) != TRAINABLE_TENSORS
        or sum(sizes[n] for n in names) != TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("adapter parameter inventory")
    del net
    return names


def load_sources(h124):
    import torch

    checkpoints = {}
    for rnd, (name, digest) in SOURCES.items():
        path = INPUTS / name
        if sha(path) != digest:
            raise RuntimeError(f"R{rnd} copy changed")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("dope_sha256") != h124.DOPE:
            raise RuntimeError(f"R{rnd} DOPE identity mismatch")
        checkpoints[rnd] = ck
    return checkpoints


def build_soups(h124, root):
    """Write one checkpoint per arm (R882 is the pinned copy itself); return paths, hashes and the structure receipt."""
    import torch

    names = trainable_names(h124)
    cks = load_sources(h124)
    states = {r: ck["state_dict"] for r, ck in cks.items()}
    if any(set(s) != set(states[882]) for s in states.values()):
        raise RuntimeError("parameter sets differ")
    differing = frozen_identity(states, set(names))
    paths = {ANCHOR: (INPUTS / SOURCES[882][0], SOURCES[882][1])}
    root.mkdir(parents=True, exist_ok=True)
    drift = {}
    for arm, weights in ARMS.items():
        if arm == ANCHOR:
            continue
        sd = blend(states, weights, names)
        ck = dict(
            cks[882],
            state_dict=sd,
            round=f"h152-{arm}",
            h152_weights={str(k): v for k, v in weights.items()},
        )
        p = root / f"{arm}.pt"
        torch.save(ck, p)
        paths[arm] = (p, sha(p))
        drift[arm] = float(
            torch.linalg.vector_norm(
                torch.cat(
                    [
                        (sd[n].double() - states[882][n].double()).reshape(-1)
                        for n in names
                    ]
                )
            )
        )
    receipt = {
        "trainable_tensors": len(names),
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "non_trainable_tensors_differing_across_inputs": differing,
        "non_trainable_source": "R882 (unchanged)",
        "adapter_l2_distance_from_R882": drift,
        "soup_sha256": {a: v[1] for a, v in paths.items()},
        "weights": {a: {str(k): v for k, v in w.items()} for a, w in ARMS.items()},
    }
    return paths, receipt


# ------------------------------------------------------------------ run


def regression(values, reference):
    """R882 must reproduce the live best_dev utilities exactly: 91/91 contexts with zero error and the same mean."""
    exp = np.asarray(reference["utility"]["per_context"], np.float64)
    got = np.asarray(values, np.float64)
    if got.shape != (91,) or exp.shape != (91,) or not np.array_equal(got, exp):
        raise RuntimeError("R882 regression failed: 91/91 exact required")
    if float(got.mean()) != reference["best_utility"]:
        raise RuntimeError("R882 regression failed: mean differs from best_utility")
    return {
        "status": "PASS",
        "contexts_matched": 91,
        "max_abs_error": 0.0,
        "mean": float(got.mean()),
    }


def evaluate_arm(h124, root, arm, path, digest, rows, context, pools, projection):
    guards(root.parent)
    h124.headroom(root)
    h124.R882 = path
    h124.FILES[path] = digest
    arm_root = root / f"eval-{arm}"
    arm_root.mkdir()
    costs, net = bounded_threads(
        h124.baseline, arm_root, rows, context, pools, projection
    )
    del net
    h124.release_memory()
    (arm_root / "r882-copy.pt").unlink()
    return costs, json.loads((arm_root / "dope-secondary.json").read_text())


def native_values(h139, costs, pools, rows):
    out = []
    for r in rows:
        pool = pools[r["context_sha256"]]
        out.append(
            h139.native(costs[r["context_sha256"]], pool["fitness"], pool["gate"])
        )
    return out


def run(root, smoke=False, launch=None):
    """--smoke: build the soups, score only R882 (same rows and path as the run) and require the exact regression.
    --run: bind the merged source, then score every arm once."""
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    root = Path(root)
    parent = Path.home() / "jope-data/h152"
    if (
        root.exists()
        or root.parent != parent
        or not root.name.startswith(("run-", "smoke-"))
    ):
        raise RuntimeError(
            "H152 requires a fresh run-*/smoke-* root under ~/jope-data/h152"
        )
    if smoke == (launch is not None):
        raise RuntimeError("--run needs --launch; --smoke takes none")
    launch_guard = guards(parent)
    binding = None if smoke else bind_source(json.loads(Path(launch).read_text()))
    h139 = load_h139()
    h124 = h139.load_h124()
    h124.RSS_LIMIT_KIB = (
        RSS_LIMIT_BYTES // 1024
    )  # decimal 6 GB ceiling for the inherited synchronous RSS check
    t0, cpu0, run_sha = time.monotonic(), time.process_time(), sha(Path(__file__))
    paths, structure = build_soups(h124, root)
    train, val, reference = h139.roster(h124)
    rows = val + train
    context, _ = h124.train_context(rows, [])
    projection = h124.train_projection(rows, [])
    pools = h139.pools_for(rows)
    identity = {
        "binding": binding,
        "prereg_sha256": PREREG_SHA,
        "runner_sha256": run_sha,
        "h139_runner_sha256": H139_SHA,
        "launch_guard": launch_guard,
        "structure": structure,
        "torch_threads": TORCH_THREADS,
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    h124.write(root / "execution-identity.json", identity)
    native_val, native_train, u4, dope = {}, {}, {}, {}
    for arm in (ANCHOR,) if smoke else ARMS:
        costs, dope[arm] = evaluate_arm(
            h124, root, arm, *paths[arm], rows, context, pools, projection
        )
        if dope[arm]["before"] != h124.DOPE or dope[arm]["after"] != h124.DOPE:
            raise RuntimeError("DOPE identity changed")
        native_val[arm] = native_values(h139, costs, pools, val)
        native_train[arm] = native_values(h139, costs, pools, train)
        u4[arm] = [
            h124.endpoints(costs[r["context_sha256"]], pools[r["context_sha256"]]["u"])[
                "u4"
            ]
            for r in train
        ]
        if arm == ANCHOR:  # regression first, before any soup is scored
            reg = regression(native_val[arm], reference)
        print("ARM DONE", arm, flush=True)
    end_guard = guards(parent)
    if smoke:
        out = {
            "status": "PASS",
            "optimizer_updates": 0,
            "soup_arms_scored": 0,
            "regression": reg,
            "dope": dope[ANCHOR],
            "structure": structure,
            "launch_guard": launch_guard,
            "end_guard": end_guard,
            "torch_threads": torch.get_num_threads(),
            "wall_seconds": time.monotonic() - t0,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        h124.write(root / "smoke.json", out)
        print("SMOKE PASS", json.dumps(reg), flush=True)
        return out
    result = readout(val, train, native_val, native_train, u4)
    per_raw = json.dumps(
        {"val": native_val, "train": native_train, "u4": u4}, sort_keys=True
    ).encode()
    import sklearn

    metrics = {
        "schema": "jope.h152.ckpt-soup/v2",
        "status": "COMPLETE",
        "verdict": result.pop("verdict"),
        "binding": binding,
        "prereg_sha256": PREREG_SHA,
        "runner_sha256": run_sha,
        "h139_runner_sha256": H139_SHA,
        "structure": structure,
        "regression": reg,
        **result,
        "per_context_sha256": hashlib.sha256(per_raw).hexdigest(),
        "dope": {
            a: {"before": d["before"], "after": d["after"]} for a, d in dope.items()
        },
        "read_boundary": {
            "dl_train_contexts": len(train),
            "dl_val_contexts": len(val),
            "sealed_payloads_opened": 0,
            "holdout_payloads_opened": 0,
            "meta_test_payloads_opened": 0,
            "live_dir_writes": 0,
        },
        "runtime": {
            "wall_seconds": time.monotonic() - t0,
            "process_cpu_seconds": time.process_time() - cpu0,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_threads": TORCH_THREADS,
            "launch_guard": launch_guard,
            "end_guard": end_guard,
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
        raise RuntimeError("H152 metrics would exceed 100 KB")
    (root / "per_context.json").write_bytes(per_raw)
    (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def family_means(values, families):
    order = sorted(set(families))
    idx = {f: i for i, f in enumerate(order)}
    s, c = np.zeros(len(order)), np.zeros(len(order))
    for v, f in zip(values, families):
        s[idx[f]] += v
        c[idx[f]] += 1
    return s / c


def train_folds(families, seed):
    """Deterministic family-disjoint 5-fold partition of the TRAIN families for one seed (descriptive only)."""
    fams = np.array(sorted(set(families)))
    return [
        sorted(g.tolist())
        for g in np.array_split(np.random.default_rng(seed).permutation(fams), 5)
    ]


def fold_readout(train, native_train):
    """For each seed 152101-152103: family-disjoint folds and, per arm, the equal-family gain vs R882 within each fold.
    No fitting or arm selection uses these folds; they show how the TRAIN-exposed gain spreads across family subsets."""
    tf = [r["family_id"] for r in train]
    order = sorted(set(tf))
    ref = family_means(native_train[ANCHOR], tf)
    out = {}
    for seed in SEEDS:
        groups = train_folds(tf, seed)
        fold_of = {f: k for k, g in enumerate(groups) for f in g}
        gains = {}
        for arm in ARMS:
            if arm == ANCHOR:
                continue
            g = family_means(native_train[arm], tf) - ref
            gains[arm] = [
                float(np.mean([g[i] for i, f in enumerate(order) if fold_of[f] == k]))
                for k in range(5)
            ]
        out[str(seed)] = {
            "fold_families": [len(g) for g in groups],
            "fold_contexts": [sum(fold_of[f] == k for f in tf) for k in range(5)],
            "assignment_sha256": hashlib.sha256(
                json.dumps(groups).encode()
            ).hexdigest(),
            "fold_gain_by_arm": gains,
        }
    return out


def readout(val, train, native_val, native_train, u4):
    vf, tf = [r["family_id"] for r in val], [r["family_id"] for r in train]
    ref_v, ref_t, ref_u = (
        family_means(native_val[ANCHOR], vf),
        family_means(native_train[ANCHOR], tf),
        family_means(u4[ANCHOR], tf),
    )
    arms = {}
    for arm in ARMS:
        if arm == ANCHOR:
            continue
        g = family_means(native_val[arm], vf) - ref_v
        r = gain_readout(g)
        r["elo_ci95"] = elo_ci(g)
        r["role"] = (
            "primary"
            if arm == PRIMARY
            else "secondary"
            if arm == SECONDARY
            else "exploratory (never admitted)"
        )
        r["val_context_mean"] = float(np.mean(native_val[arm]))
        r["val_invalid_picks"] = int(sum(v == -2.0 for v in native_val[arm]))
        gt = family_means(native_train[arm], tf) - ref_t
        rt = gain_readout(gt)
        r["train_descriptive"] = {
            "gain": rt["gain"],
            "median_ci95": rt["median_ci95"],
            "families_win_tie_loss": rt["families_win_tie_loss"],
            "elo": rt["elo"],
            "invalid_picks": int(sum(v == -2.0 for v in native_train[arm])),
            "u4_gain": float((family_means(u4[arm], tf) - ref_u).mean()),
            "note": "TRAIN-exposed: every source adapter was trained on these contexts",
        }
        arms[arm] = r
    decision = decide(arms[PRIMARY], arms[SECONDARY])
    for a in ADMISSIBLE:
        arms[a]["decision"] = decision[a]
    verdict = (
        "ADVANCE: MID"
        if decision[PRIMARY]["advance"]
        else ("ADVANCE: SOUP4" if decision[SECONDARY]["advance"] else "NO_ADVANCE")
    )
    return {
        "verdict": verdict,
        "arms": arms,
        "train_family_folds_descriptive": fold_readout(train, native_train),
        "baseline": {
            "R882_val_context_mean": float(np.mean(native_val[ANCHOR])),
            "R882_val_invalid_picks": int(sum(v == -2.0 for v in native_val[ANCHOR])),
            "R882_train_invalid_picks": int(
                sum(v == -2.0 for v in native_train[ANCHOR])
            ),
        },
        "seed_note": "point estimates are deterministic; the three seeds re-draw only the 19-family bootstrap",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument(
        "--smoke",
        action="store_true",
        help="build soups; score R882 only for the exact regression",
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--launch",
        type=Path,
        help="launch.json naming reviewed_head and source_commit (--run only)",
    )
    a = ap.parse_args()
    run(a.root, smoke=a.smoke, launch=a.launch)


if __name__ == "__main__":
    main()
