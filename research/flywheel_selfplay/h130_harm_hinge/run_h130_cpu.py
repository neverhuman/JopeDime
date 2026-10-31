#!/usr/bin/env python3
"""H130 CPU-only executor: fit-fold harm-constrained HINGE and CVAR objectives vs CONTROL.

Guardrails enforced:
- CPU only (CUDA disabled, 0 GPU seconds, no qualification.lock)
- <= 6 threads (4 threads configured)
- RSS <= 6 GiB, MemAvailable >= 30 GiB, load1 <= 86.7, disk used <= 94%
- Fresh hash-verified r882 copy per cell; DOPE frozen (c51a56d0...) verified pre & post
- Fresh 5-fold family-grouped DL-Train split (fold_seed = 130094; fold 0 = 41 families; folds 1..4 = 164 fit families; context counts are checked against the pinned DL-Train input at execution)
- Unchanged-r882 regression gate (0.0 prediction/utility/loss/grad diff) before harm baseline or optimization
- Frozen r882 per-fit-family loss baselines; assessment families never enter either penalty
- Local custody receipts only; no network/S3 access
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

_script = Path(__file__).resolve()
WORKTREE = next(
    (p for p in _script.parents if (p / "research/flywheel_selfplay/h48/run.py").is_file()),
    Path("/home/ubuntu/JopeDime/.agent/worktrees/agy_h61"),
)
H130_GIT_DIR = WORKTREE / "research/flywheel_selfplay/h130_harm_hinge"
CUSTODY_ROOT = Path("/home/ubuntu/jope-rescue-20260921/h130-harm-hinge")
INPUT_DIR = Path("/home/ubuntu/jope-data/h50/input")
R882_PATH = Path("/home/ubuntu/jope-data/h51/ckpt/round-882.pt")
W1_CTX_NPZ = Path("/home/ubuntu/jope-data/w1/cache/train_only_context_features.npz")

R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
FUSED_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
SOURCE_MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
FOLD_SEED = 130094
FOLD_SHA = "98196cbc017c10d12db1f8c4390c163a11d3faddf1c457b79108a6386ed785ea"

SEEDS = (94, 95, 96)
ARMS = ("CONTROL", "HINGE", "CVAR")
ROUNDS = 20
STEPS_PER_ROUND = 2
UPDATES_PER_CELL = ROUNDS * STEPS_PER_ROUND  # 40
BATCH_SIZE = 16
SLATE_SIZE = 192
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BASE_GRAD_CLIP = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 130094
TRAINABLE_PARAMS = 29_211


MAX_RSS_BYTES = 6 * 1024 * 1024 * 1024  # 6 GiB
MIN_MEM_AVAILABLE_BYTES = 30 * 1024 * 1024 * 1024  # 30 GiB
MAX_LOAD1 = 86.7
MAX_DISK_USED_PCT = 94.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def mem_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def check_guards(stage: str) -> dict:
    load1, load5, _ = os.getloadavg()
    mem_avail = mem_available_bytes()
    rss = current_rss_bytes()
    fs = os.statvfs(str(CUSTODY_ROOT.parent))
    fs_available = fs.f_bavail * fs.f_frsize
    disk_pct = 100.0 * (1.0 - fs.f_bavail / fs.f_blocks)
    snap = {
        "stage": stage,
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "mem_available_gib": round(mem_avail / (1024**3), 2),
        "disk_used_pct": round(disk_pct, 2),
        "disk_free_gib": round(fs_available / (1024**3), 2),
        "rss_mib": round(rss / (1024**2), 2),
        "rss_bytes": rss,
        "torch_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        raise RuntimeError(f"[{stage}] CUDA must be disabled in CPU-only H130")
    if torch.get_num_threads() > 6:
        raise RuntimeError(f"[{stage}] torch_threads={torch.get_num_threads()} > 6")
    if rss > MAX_RSS_BYTES:
        raise RuntimeError(f"[{stage}] RSS {rss} exceeds 6 GiB cap")
    if mem_avail < MIN_MEM_AVAILABLE_BYTES:
        raise RuntimeError(f"[{stage}] MemAvailable {mem_avail} below 30 GiB floor")
    if load1 > MAX_LOAD1:
        raise RuntimeError(f"[{stage}] load1={load1:.2f} exceeds {MAX_LOAD1}")
    if disk_pct >= MAX_DISK_USED_PCT:
        raise RuntimeError(f"[{stage}] disk_used_pct={disk_pct:.2f}% is at or above {MAX_DISK_USED_PCT}%")
    return snap


def local_record_and_verify(local_path: Path, rel_key: str) -> dict:
    """Record local custody output without network or cloud credential access."""
    actual = sha256(local_path)
    return {"status": "LOCAL_ONLY", "path": str(local_path), "relative_key": rel_key,
            "sha256": actual, "bytes": local_path.stat().st_size}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(WORKTREE / "research/flywheel_selfplay"))
sys.path.insert(0, str(WORKTREE / "python/src"))
h48 = load_module("h130_h48", WORKTREE / "research/flywheel_selfplay/h48/run.py")
h55 = load_module("h130_h55", WORKTREE / "research/flywheel_selfplay/h55/runner.py")
h53b = load_module("h130_h53b", WORKTREE / "research/flywheel_selfplay/h53b/runner.py")
h59 = load_module("h130_h59", WORKTREE / "research/flywheel_selfplay/h59/runner.py")
h61 = load_module("h130_h61", WORKTREE / "research/flywheel_selfplay/h61/runner.py")
elo_mod = load_module("h130_elo", WORKTREE / "python/src/jope/north_star/elo_tracker.py")


def stage_r882_copy(tag: str) -> Path:
    scratch_dir = CUSTODY_ROOT / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dst = scratch_dir / f"r882_{tag}.pt"
    shutil.copy2(R882_PATH, dst)
    if sha256(dst) != R882_SHA:
        raise RuntimeError(f"Staged r882 SHA mismatch at {dst}")
    return dst


def paired_stream_seed(seed: int, round_index: int, step_index: int) -> int:
    sampler_key = seed * 1_000_003 + 10_007
    payload = f"flywheel-batch/v1:{sampler_key}:{round_index}:{step_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def build_arm_seed_batch_plan(
    arm: str,
    fit_cids: list[str],
    sorted_fit_by_loss: list[str],
    pools: dict[str, dict],
    orders: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    full_ids = tuple(sorted(fit_cids))
    sampler_key = seed * 1_000_003 + 10_007
    rng_state = np.random.get_state()
    plan = []
    try:
        for rnd in range(1, ROUNDS + 1):
            round_rng_seed = int((sampler_key + rnd * 104729) % (2**32))
            np.random.seed(round_rng_seed)
            for step in range(8):
                eff_step = step if step < STEPS_PER_ROUND else (STEPS_PER_ROUND - 1)
                update_idx = (rnd - 1) * STEPS_PER_ROUND + eff_step
                if arm not in ARMS:
                    raise ValueError(f"Unknown arm {arm}")
                q_u = 1.0
                eligible_ids = full_ids

                rot = rnd % len(eligible_ids)
                rotated = eligible_ids[rot:] + eligible_ids[:rot]
                stream = paired_stream_seed(seed, rnd, step)
                cids = tuple(random.Random(stream).choices(rotated, k=BATCH_SIZE))
                ix = tuple(
                    np.asarray(
                        h53b.cached_training_indices(pools[c], orders[c], SLATE_SIZE),
                        dtype=np.int64,
                    ).copy()
                    for c in cids
                )
                if step < STEPS_PER_ROUND:
                    pool_hash = h61.compute_candidate_pool_hash(cids, ix, pools)
                    plan.append(
                        {
                            "update_index": update_idx,
                            "round_index": rnd,
                            "step_index": step,
                            "q_fraction": q_u,
                            "eligible_count": len(eligible_ids),
                            "stream_seed": stream,
                            "round_rng_seed": round_rng_seed,
                            "context_ids": cids,
                            "candidate_indices": ix,
                            "candidate_pool_sha256": pool_hash,
                        }
                    )
    finally:
        np.random.set_state(rng_state)
    if len(plan) != UPDATES_PER_CELL:
        raise RuntimeError(f"Expected {UPDATES_PER_CELL} steps, got {len(plan)}")
    return plan


def per_context_losses(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    smooth = F.smooth_l1_loss(prediction, targets, reduction="none", beta=1.0).mean(dim=1)
    kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-targets / 0.2, dim=-1),
        reduction="none",
    ).sum(dim=1)
    return smooth + 0.1 * kl


def h130_objective(
    arm: str,
    context_losses: torch.Tensor,
    family_ids: list[str],
    fit_family_baseline_losses: dict[str, float],
    *,
    mu: float = 1.0,
    margin: float = 0.0,
    cvar_fraction: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return objective, mean-context term, and fit-family harm penalty."""
    if context_losses.ndim != 1 or len(family_ids) != context_losses.numel():
        raise ValueError("context losses and family IDs must have the same one-dimensional length")
    if arm not in ARMS:
        raise ValueError(f"Unknown H130 arm {arm}")
    mean_term = context_losses.mean()
    if arm == "CONTROL":
        zero = mean_term.new_zeros(())
        return mean_term, mean_term, zero

    unique_families = sorted(set(family_ids))
    deltas = []
    for family in unique_families:
        if family not in fit_family_baseline_losses:
            raise ValueError(f"No frozen fit-fold r882 baseline for family {family}")
        positions = [i for i, value in enumerate(family_ids) if value == family]
        family_loss = context_losses[positions].mean()
        deltas.append(family_loss - float(fit_family_baseline_losses[family]) + margin)
    deltas_tensor = torch.stack(deltas)
    positive_deltas = deltas_tensor.relu()
    if arm == "HINGE":
        penalty = positive_deltas.mean()
    else:
        top_count = max(1, math.ceil(cvar_fraction * len(unique_families)))
        top_indices = torch.argsort(deltas_tensor, descending=True)[:top_count]
        # CVaR weights the worst 20% by 1/0.2, with zero weight elsewhere.
        penalty = positive_deltas[top_indices].sum() * (1.0 / cvar_fraction) / len(unique_families)
    return mean_term + mu * penalty, mean_term, penalty


def predict_pool_scores_fast(
    cids: list[str],
    pools: dict[str, dict],
    predict,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    preds_by_cid: dict[str, np.ndarray] = {}
    by_size: dict[int, list[str]] = defaultdict(list)
    for cid in sorted(cids):
        by_size[len(pools[cid]["pids"])].append(cid)
    with torch.no_grad():
        for pool_size, group in sorted(by_size.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                indexes = [np.arange(pool_size, dtype=np.int64) for _ in batch]
                values = predict(batch, indexes).detach().cpu().numpy()
                for row, cid in zip(values, batch, strict=True):
                    preds_by_cid[cid] = np.asarray(row, dtype=np.float64)
    return preds_by_cid


def paired_family_elo_from_deltas(fam_deltas: np.ndarray, tol: float = 1e-12) -> dict:
    wins = int(np.sum(fam_deltas > tol))
    losses = int(np.sum(fam_deltas < -tol))
    ties = int(len(fam_deltas) - wins - losses)
    n = max(1, len(fam_deltas))
    score = (wins + 0.5 * ties) / n
    clipped = min(max(score, 1e-6), 1.0 - 1e-6)
    elo = 1500.0 + 400.0 * math.log10(clipped / (1.0 - clipped))
    return {
        "elo": round(float(elo), 2),
        "win_rate": float(score),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def evaluate_arm_vs_comparator(
    arm_name: str,
    comp_name: str,
    arm_fam_by_seed: dict[int, dict[str, float]],
    comp_fam_by_seed: dict[int, dict[str, float]],
    arm_ctx_by_seed: dict[int, dict[str, float]],
    comp_ctx_by_seed: dict[int, dict[str, float]],
    assess_fams: list[str],
    assess_cids: list[str],
) -> dict:
    n_fams = len(assess_fams)
    n_ctxs = len(assess_cids)
    per_seed = []
    seed_gains = []
    seed_harm_fam_fracs = []
    seed_harm_ctx_fracs = []
    seed_elos = []

    fam_delta_matrix = np.zeros((len(SEEDS), n_fams), dtype=np.float64)
    ctx_delta_matrix = np.zeros((len(SEEDS), n_ctxs), dtype=np.float64)

    for s_idx, seed in enumerate(SEEDS):
        a_fam = np.array([arm_fam_by_seed[seed][f] for f in assess_fams], dtype=np.float64)
        c_fam = np.array([comp_fam_by_seed[seed][f] for f in assess_fams], dtype=np.float64)
        d_fam = a_fam - c_fam
        fam_delta_matrix[s_idx] = d_fam

        a_ctx = np.array([arm_ctx_by_seed[seed][c] for c in assess_cids], dtype=np.float64)
        c_ctx = np.array([comp_ctx_by_seed[seed][c] for c in assess_cids], dtype=np.float64)
        d_ctx = a_ctx - c_ctx
        ctx_delta_matrix[s_idx] = d_ctx

        fam_gain = float(np.mean(d_fam))
        ctx_gain = float(np.mean(d_ctx))
        harm_fams = int(np.sum(d_fam < -0.01))
        harm_ctxs = int(np.sum(d_ctx < -0.01))
        harm_fam_frac = float(harm_fams / n_fams)
        harm_ctx_frac = float(harm_ctxs / n_ctxs)
        elo_info = paired_family_elo_from_deltas(d_fam)

        seed_gains.append(fam_gain)
        seed_harm_fam_fracs.append(harm_fam_frac)
        seed_harm_ctx_fracs.append(harm_ctx_frac)
        seed_elos.append(elo_info["elo"])

        per_seed.append(
            {
                "seed": seed,
                "arm_family_mean": float(np.mean(a_fam)),
                "comparator_family_mean": float(np.mean(c_fam)),
                "family_mean_delta": fam_gain,
                "family_median_delta": float(np.median(d_fam)),
                "context_mean_delta": ctx_gain,
                "harmed_families_count": harm_fams,
                "harmed_families_fraction": harm_fam_frac,
                "harmed_contexts_count": harm_ctxs,
                "harmed_contexts_fraction": harm_ctx_frac,
                "paired_elo": elo_info["elo"],
                "family_win_rate": elo_info["win_rate"],
                "family_wtl": [elo_info["wins"], elo_info["ties"], elo_info["losses"]],
            }
        )

    fam_seed_avg_deltas = np.mean(fam_delta_matrix, axis=0)
    ctx_seed_avg_deltas = np.mean(ctx_delta_matrix, axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_fams, size=(BOOTSTRAP_DRAWS, n_fams))
    boot_means = np.mean(fam_seed_avg_deltas[idx], axis=1)

    lcb95 = float(np.quantile(boot_means, 0.05))
    ci95_low, ci95_high = (
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )
    p_one_sided = float((1 + int(np.sum(boot_means <= 0.0))) / (BOOTSTRAP_DRAWS + 1))

    pooled_flat = fam_delta_matrix.reshape(-1)
    pooled_elo = paired_family_elo_from_deltas(pooled_flat)
    wins_mat = (fam_delta_matrix > 1e-12).astype(np.float64)
    ties_mat = (np.abs(fam_delta_matrix) <= 1e-12).astype(np.float64)
    fam_scores = np.mean(wins_mat + 0.5 * ties_mat, axis=0)
    boot_scores = np.clip(np.mean(fam_scores[idx], axis=1), 1e-6, 1.0 - 1e-6)
    boot_elos = 1500.0 + 400.0 * np.log10(boot_scores / (1.0 - boot_scores))
    elo_ci95 = [
        round(float(np.quantile(boot_elos, 0.025)), 2),
        round(float(np.quantile(boot_elos, 0.975)), 2),
    ]

    return {
        "comparator": comp_name,
        "median_seed_gain": float(np.median(seed_gains)),
        "pooled_family_median_gain": float(np.median(fam_seed_avg_deltas)),
        "pooled_family_harm_fraction": float(np.mean(fam_seed_avg_deltas < -0.01)),
        "mean_seed_gain": float(np.mean(seed_gains)),
        "per_seed_family_mean_deltas": seed_gains,
        "lcb95_one_sided": lcb95,
        "ci95_two_sided": [ci95_low, ci95_high],
        "p_value_one_sided": p_one_sided,
        "max_seed_harmed_family_fraction": float(np.max(seed_harm_fam_fracs)),
        "max_seed_harmed_context_fraction": float(np.max(seed_harm_ctx_fracs)),
        "seed_mean_harmed_family_fraction": float(np.mean(fam_seed_avg_deltas < -0.01)),
        "seed_mean_harmed_context_fraction": float(np.mean(ctx_seed_avg_deltas < -0.01)),
        "median_seed_paired_elo": float(np.median(seed_elos)),
        "pooled_paired_elo": pooled_elo["elo"],
        "pooled_paired_elo_ci95": elo_ci95,
        "pooled_family_win_rate": pooled_elo["win_rate"],
        "per_seed": per_seed,
    }


def synthetic_smoke() -> int:
    """Exercise H130 objective gradients on synthetic tensors only; never steps an optimizer."""
    losses = torch.tensor([0.40, 0.80, 0.70, 0.90], dtype=torch.float64, requires_grad=True)
    family_ids = ["fit-a", "fit-a", "fit-b", "fit-c"]
    baseline = {"fit-a": 0.30, "fit-b": 0.30, "fit-c": 0.30}
    rows = {}
    for arm in ARMS:
        objective, mean_term, penalty = h130_objective(arm, losses, family_ids, baseline)
        if not torch.isfinite(objective):
            raise RuntimeError(f"{arm} objective is non-finite")
        grad = torch.autograd.grad(objective, losses, retain_graph=True)[0]
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"{arm} gradient is non-finite")
        if arm != "CONTROL" and float(penalty) <= 0.0:
            raise RuntimeError(f"{arm} smoke did not activate harm penalty")
        rows[arm] = {"objective": float(objective), "mean_term": float(mean_term), "penalty": float(penalty),
                     "gradient_l1": float(grad.abs().sum())}
    control, _, _ = h130_objective("CONTROL", losses, family_ids, baseline)
    if float(control) != float(losses.mean()):
        raise RuntimeError("CONTROL does not equal the uniform context mean")
    if not math.isclose(rows["HINGE"]["penalty"], 1.3 / 3.0, abs_tol=1e-12):
        raise RuntimeError("HINGE did not average the family-level positive deltas")
    if not math.isclose(rows["CVAR"]["penalty"], 1.0, abs_tol=1e-12):
        raise RuntimeError("CVAR did not apply top-20% family weight 1/0.2")
    print(json.dumps({"status": "PASS", "mode": "synthetic_only", "optimizer_updates": 0,
                      "assessment_read": False, "rows": rows}, sort_keys=True))
    return 0


def main() -> int:
    merged = subprocess.run(
        ["git", "-C", str(WORKTREE), "merge-base", "--is-ancestor", "HEAD", "origin/main"],
        check=False,
        capture_output=True,
        text=True,
    )
    if merged.returncode != 0:
        raise RuntimeError("H130 execution requires the preregistration source to be merged into origin/main")
    t0 = time.monotonic()
    CUSTODY_ROOT.mkdir(parents=True, exist_ok=True)
    guard_init = check_guards("wave_start")
    print(f"[H130] Wave start guards OK: {json.dumps(guard_init)}", flush=True)

    # 1. Verify input and model hashes
    for path, exp in [
        (R882_PATH, R882_SHA),
        (INPUT_DIR / "cpu_exact_fused.py", FUSED_SHA),
        (INPUT_DIR / "selected.json", SELECTED_SHA),
        (INPUT_DIR / "source_manifest.json", SOURCE_MANIFEST_SHA),
    ]:
        actual = sha256(path)
        if actual != exp:
            raise RuntimeError(f"Hash mismatch for {path}: {actual} != {exp}")

    prereg_sha = sha256(H130_GIT_DIR / "PREREG.md")

    selected, train_rows, _ = h55.input_metadata(INPUT_DIR, h48)
    if len(train_rows) != 1219:
        raise RuntimeError("Expected 1219 DL-Train rows")

    # 2. Build fresh family split (fold_seed = 130094)
    families = sorted({r["family_id"] for r in train_rows})
    rng_fold = np.random.default_rng(FOLD_SEED)
    order = list(np.asarray(families, dtype=object)[rng_fold.permutation(len(families))])
    folds = [sorted(map(str, chunk)) for chunk in np.array_split(order, 5)]
    fold_raw = json.dumps(folds, sort_keys=True, separators=(",", ":")).encode()
    actual_fold_sha = hashlib.sha256(fold_raw).hexdigest()
    if actual_fold_sha != FOLD_SHA:
        raise RuntimeError(f"Fold SHA mismatch: {actual_fold_sha} != {FOLD_SHA}")

    fold_path = CUSTODY_ROOT / "fold_manifest_130094.json"
    fold_path.write_bytes(fold_raw)
    local_fold = local_record_and_verify(fold_path, "fold_manifest_130094.json")

    assess_fams = set(folds[0])
    fit_fams = set(f for k in range(1, 5) for f in folds[k])
    if len(assess_fams) != 41 or len(fit_fams) != 164 or (assess_fams & fit_fams):
        raise RuntimeError("Fold family partition invalid")

    assess_rows = [r for r in train_rows if r["family_id"] in assess_fams]
    fit_rows = [r for r in train_rows if r["family_id"] in fit_fams]
    assess_cids = sorted(r["context_sha256"] for r in assess_rows)
    fit_cids = sorted(r["context_sha256"] for r in fit_rows)
    all_cids = sorted(r["context_sha256"] for r in train_rows)
    family_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}

    if len(assess_cids) + len(fit_cids) != len(train_rows):
        raise RuntimeError("Family split did not account for every DL-Train context exactly once")

    # 3. Verify all 1,219 DL-Train pools and .jtt1 headers
    print("[H130] Loading & verifying 1,219 DL-Train pools and .jtt1 headers...", flush=True)
    w1_mat = h48.base.evaluator.get_w1_sign_matrix() / np.sqrt(412.0)
    pools: dict[str, dict] = {}
    orders: dict[str, np.ndarray] = {}
    probe_tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, row in enumerate(train_rows):
        if row.get("dl_split") != "DL-Train" or row.get("source_split") not in ("train", "inner"):
            raise RuntimeError("Row outside DL-Train closure")
        cid = row["context_sha256"]
        p_path = INPUT_DIR / "pools" / f"{cid}.npz"
        t_path = INPUT_DIR / "tables" / f"{cid}.jtt1"
        if sha256(p_path) != row["pool_sha256"]:
            raise RuntimeError(f"Pool SHA mismatch for {cid}")
        raw = t_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"Table SHA mismatch for {cid}")
        nt, nq, width = h48.base.struct.unpack_from("<III", raw, 4)
        if nq != 0 or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError(f"Invalid train-only JTT1 header for {cid}")
        if idx < 4:
            vals = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
            probe_tables[cid] = (vals[:, :-1], vals[:, -1])
        del raw

        with np.load(p_path, allow_pickle=False) as z:
            progs = z["program"].astype(np.float32)
            fit = z["fitness"].astype(np.float64)
            gate = z["gate"].astype(bool)
            pids = [v.decode() if isinstance(v, bytes) else str(v) for v in z["program_sha256"].tolist()]
        pools[cid] = {
            "rec": row,
            "pids": pids,
            "program": progs,
            "proj16": (progs.astype(np.float64) @ w1_mat).astype(np.float32),
            "fitness": fit,
            "gate": gate,
        }
        orders[cid] = np.argsort(fit)

    device = torch.device("cpu")
    ctx_npz = np.load(W1_CTX_NPZ, allow_pickle=False)
    tokens: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for i, cid in enumerate(ctx_npz["cids"]):
        c_str = str(cid)
        tokens[c_str] = torch.as_tensor(ctx_npz["tokens_10x32"][i], dtype=torch.float32, device=device)
        masks[c_str] = torch.as_tensor(ctx_npz["mask_10"][i], dtype=torch.bool, device=device)
    if set(tokens.keys()) != set(all_cids):
        raise RuntimeError("Context feature cache CID set mismatch")

    # 4. Step 1: Unchanged-r882 Regression-First Gate
    print("[H130] Step 1: Running unchanged-r882 Regression-First Gate...", flush=True)
    ckpt_a = stage_r882_copy("reg_a")
    ckpt_b = stage_r882_copy("reg_b")
    try:
        net_a, params_a, guard_a, state_a = h55.build_model(INPUT_DIR, ckpt_a, device, h48)
        net_b, params_b, guard_b, state_b = h55.build_model(INPUT_DIR, ckpt_b, device, h48)
    finally:
        ckpt_a.unlink(missing_ok=True)
        ckpt_b.unlink(missing_ok=True)

    dope_a = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
    dope_b = h48.h46.checkpoint_guard(guard_b, net_b, expected=DOPE_SHA)
    adapter_a = h53b.adapter_sha(net_a)
    adapter_b = h53b.adapter_sha(net_b)
    if adapter_a != adapter_b:
        raise RuntimeError("Initial r882 adapter SHA mismatch across copies")

    tok_probe, msk_probe = h48.h46.make_tokens(net_a, probe_tables, device)
    for cid in probe_tables:
        t_diff = float(torch.max(torch.abs(tok_probe[cid] - tokens[cid])).item())
        m_diff = bool(torch.any(msk_probe[cid] != masks[cid]).item())
        if t_diff != 0.0 or m_diff:
            raise RuntimeError(f"Precomputed DOPE token mismatch on {cid}: t_diff={t_diff}, m_diff={m_diff}")

    pred_a = h48.make_predictor(net_a, pools, tokens, masks, device)
    pred_b = h48.make_predictor(net_b, pools, tokens, masks, device)

    scores_a_fold0 = predict_pool_scores_fast(assess_cids, pools, pred_a, batch_size=64)
    scores_b_fold0 = predict_pool_scores_fast(assess_cids, pools, pred_b, batch_size=64)
    max_pred_diff_fold0 = max(
        float(np.max(np.abs(scores_a_fold0[c] - scores_b_fold0[c]))) for c in assess_cids
    )
    if max_pred_diff_fold0 != 0.0:
        raise RuntimeError(f"Two-copy prediction diff on fold0 is non-zero: {max_pred_diff_fold0}")

    eval_a_unrev = h61.evaluate_pool_predictions(assess_cids, pools, scores_a_fold0, eval_mode="unrev")
    eval_b_unrev = h61.evaluate_pool_predictions(assess_cids, pools, scores_b_fold0, eval_mode="unrev")
    eval_a_native = h61.evaluate_pool_predictions(assess_cids, pools, scores_a_fold0, eval_mode="native")
    utility_max_abs_diff = abs(float(eval_a_unrev["mean"]) - float(eval_b_unrev["mean"]))
    if utility_max_abs_diff != 0.0:
        raise RuntimeError(f"Two-copy utility diff on fold0 is non-zero: {utility_max_abs_diff}")

    # Verify exact-zero CONTROL loss and gradient diff across two independent copies (net_a vs net_b)
    reg_plan = build_arm_seed_batch_plan("CONTROL", fit_cids, fit_cids, pools, orders, SEEDS[0])
    b0 = reg_plan[0]
    cids0 = b0["context_ids"]
    ixs0 = list(b0["candidate_indices"])
    targets0 = h59.preregistered_targets(
        torch.as_tensor(np.stack([pools[c]["fitness"][ix] for c, ix in zip(cids0, ixs0)]), dtype=torch.float32),
        torch.as_tensor(np.stack([pools[c]["gate"][ix] for c, ix in zip(cids0, ixs0)]), dtype=torch.bool),
    )

    trainable_a = [p for _, p in net_a.named_parameters() if p.requires_grad]
    trainable_b = [p for _, p in net_b.named_parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable_a) != TRAINABLE_PARAMS:
        raise RuntimeError(f"Expected {TRAINABLE_PARAMS} trainable params")

    for p in trainable_a:
        p.grad = None
    for p in trainable_b:
        p.grad = None

    loss_a = per_context_losses(pred_a(list(cids0), ixs0), targets0).mean()
    loss_a.backward()
    grad_a_vec = torch.cat(
        [
            p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel())
            for p in trainable_a
        ],
        dim=0,
    )

    loss_b = per_context_losses(pred_b(list(cids0), ixs0), targets0).mean()
    loss_b.backward()
    grad_b_vec = torch.cat(
        [
            p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel())
            for p in trainable_b
        ],
        dim=0,
    )

    control_loss_abs_diff = float( abs(loss_a.detach().item() - loss_b.detach().item()) )
    control_grad_max_abs_diff = float(torch.max(torch.abs(grad_a_vec - grad_b_vec)).item())
    if control_loss_abs_diff != 0.0 or control_grad_max_abs_diff != 0.0:
        raise RuntimeError(
            f"Two-copy CONTROL loss/grad diff non-zero: loss={control_loss_abs_diff}, grad={control_grad_max_abs_diff}"
        )

    for p in trainable_a:
        p.grad = None
    del net_b, guard_b, pred_b, scores_b_fold0, trainable_b, grad_a_vec, grad_b_vec

    reg_path = CUSTODY_ROOT / "regression_receipt.json"
    if not reg_path.exists():
        reg_receipt = {
            "schema": "jope.h130.regression_receipt/v1",
            "wave": "H130",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "r882_sha256": R882_SHA,
            "dope_sha256_copy_a": dope_a,
            "dope_sha256_copy_b": dope_b,
            "initial_adapter_sha256": adapter_a,
            "trainable_adapter_params": TRAINABLE_PARAMS,
            "fold_seed": FOLD_SEED,
            "fold_manifest_sha256": FOLD_SHA,
            "assessment_families": len(assess_fams),
            "assessment_contexts": len(assess_cids),
            "fit_families": len(fit_fams),
            "fit_contexts": len(fit_cids),
            "two_copy_fold0_pred_max_abs_diff": max_pred_diff_fold0,
            "two_copy_fold0_utility_max_abs_diff": utility_max_abs_diff,
            "two_copy_control_loss_abs_diff": control_loss_abs_diff,
            "two_copy_control_grad_max_abs_diff": control_grad_max_abs_diff,
            "r882_fold0_unrev_family_utility": eval_a_unrev["mean"],
            "r882_fold0_unrev_context_utility": eval_a_unrev["context_mean"],
            "r882_fold0_native_family_utility": eval_a_native["mean"],
            "optimizer_updates": 0,
            "status": "PASS",
        }
        reg_path.write_text(json.dumps(reg_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    local_reg = local_record_and_verify(reg_path, "regression_receipt.json")
    print(
        f"[H130] Step 1 PASS: r882_fold0_unrev={eval_a_unrev['mean']:.6f}, local={local_reg['sha256']}",
        flush=True,
    )

    # 5. Step 2: Frozen Fit-Fold Baseline Loss Ranking (0 updates)
    print(
        f"[H130] Step 2: Computing frozen r882 per-context baseline loss across {len(fit_cids)} fit-fold contexts...",
        flush=True,
    )
    fit_baseline_losses: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(fit_cids), 32):
            batch_cids = fit_cids[start : start + 32]
            batch_ixs = [
                np.asarray(h53b.cached_training_indices(pools[c], orders[c], SLATE_SIZE), dtype=np.int64).copy()
                for c in batch_cids
            ]
            batch_targets = h59.preregistered_targets(
                torch.as_tensor(
                    np.stack([pools[c]["fitness"][ix] for c, ix in zip(batch_cids, batch_ixs)]),
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    np.stack([pools[c]["gate"][ix] for c, ix in zip(batch_cids, batch_ixs)]),
                    dtype=torch.bool,
                ),
            )
            batch_preds = pred_a(batch_cids, batch_ixs)
            batch_losses = per_context_losses(batch_preds, batch_targets).detach().cpu().numpy()
            for cid, lval in zip(batch_cids, batch_losses, strict=True):
                fit_baseline_losses[cid] = float(lval)

    if set(fit_baseline_losses.keys()) & set(assess_cids):
        raise RuntimeError("Assessment contexts leaked into fit-family baseline!")
    if len(fit_baseline_losses) != len(fit_cids):
        raise RuntimeError("Incomplete fit-fold loss ranking")

    post_cal_dope = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
    post_cal_adapter = h53b.adapter_sha(net_a)
    if post_cal_adapter != adapter_a or post_cal_dope != DOPE_SHA:
        raise RuntimeError("Calibration mutated model weights!")

    fit_family_baseline_losses = {
        family: float(np.mean([fit_baseline_losses[cid] for cid in fit_cids if family_by_cid[cid] == family]))
        for family in sorted(fit_fams)
    }
    family_baseline_sha = hashlib.sha256(
        json.dumps(fit_family_baseline_losses, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sorted_fit_pairs = sorted((lval, cid) for cid, lval in fit_baseline_losses.items())
    sorted_fit_by_loss = [cid for _, cid in sorted_fit_pairs]
    sorted_loss_vals = np.array([lval for lval, _ in sorted_fit_pairs], dtype=np.float64)

    ranking_sha = hashlib.sha256(
        json.dumps(sorted_fit_by_loss, separators=(",", ":")).encode()
    ).hexdigest()

    cal_path = CUSTODY_ROOT / "fit_family_baseline.json"
    if cal_path.exists():
        cal_manifest = json.loads(cal_path.read_text(encoding="utf-8"))
    else:
        cal_manifest = {
            "schema": "jope.h130.fit_family_baseline/v1",
            "wave": "H130",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fit_contexts_count": len(fit_cids),
            "assessment_contexts_touched": 0,
            "sampling_fraction_start": 1.0,
            "sampling_fraction_end": 1.0,
            "initial_eligible_count_u0": len(fit_cids),
            "final_eligible_count_u39": len(fit_cids),
            "fit_family_baseline_losses": fit_family_baseline_losses,
            "fit_family_baseline_sha256": family_baseline_sha,
            "loss_quantiles": {
                "min": float(np.min(sorted_loss_vals)),
                "q10": float(np.quantile(sorted_loss_vals, 0.10)),
                "q25": float(np.quantile(sorted_loss_vals, 0.25)),
                "median_q50": float(np.quantile(sorted_loss_vals, 0.50)),
                "mean": float(np.mean(sorted_loss_vals)),
                "q75": float(np.quantile(sorted_loss_vals, 0.75)),
                "q90": float(np.quantile(sorted_loss_vals, 0.90)),
                "max": float(np.max(sorted_loss_vals)),
            },
            "easiest_5_contexts": [
                {"cid": cid, "family_id": family_by_cid[cid], "loss": lval}
                for lval, cid in sorted_fit_pairs[:5]
            ],
            "hardest_5_contexts": [
                {"cid": cid, "family_id": family_by_cid[cid], "loss": lval}
                for lval, cid in sorted_fit_pairs[-5:]
            ],
            "sorted_fit_by_loss_sha256": ranking_sha,
            "sorted_fit_by_loss": [
                {"rank": idx + 1, "cid": cid, "family_id": family_by_cid[cid], "loss": lval}
                for idx, (lval, cid) in enumerate(sorted_fit_pairs)
            ],
            "initial_adapter_sha256": adapter_a,
            "post_calibration_adapter_sha256": post_cal_adapter,
            "dope_sha256": post_cal_dope,
            "adapter_mutated": False,
            "optimizer_updates": 0,
        }
        cal_path.write_text(json.dumps(cal_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    local_cal = local_record_and_verify(cal_path, "fit_family_baseline.json")
    print(
        f"[H130] Step 2 Frozen Fit-Family Baseline OK: median_loss={cal_manifest['loss_quantiles']['median_q50']:.6f}, "
        f"min={cal_manifest['loss_quantiles']['min']:.6f}, max={cal_manifest['loss_quantiles']['max']:.6f}, "
        f"local={local_cal['sha256']}",
        flush=True,
    )

    # 6. Step 3: Paired 3x3 Cell Execution (CONTROL, HINGE, CVAR x seeds 94, 95, 96)
    cells_dir = CUSTODY_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cell_results: dict[tuple[str, int], dict] = {}
    local_cell_receipts: dict[str, dict] = {}

    for seed in SEEDS:
        for arm in ARMS:
            c_rel = f"cells/{arm}_seed{seed}.json"
            c_path = CUSTODY_ROOT / c_rel
            if c_path.exists():
                cell_doc = json.loads(c_path.read_text(encoding="utf-8"))
                local_c = local_record_and_verify(c_path, c_rel)
                local_cell_receipts[f"{arm}_seed{seed}"] = local_c
                cell_results[(arm, seed)] = cell_doc
                print(
                    f"[H130] Cell ({arm}, seed={seed}) verified from local custody ({local_c['sha256']}): "
                    f"fold0_fam_u={cell_doc['fold0_family_mean_utility']:.6f}",
                    flush=True,
                )
                continue
            cell_t0 = time.monotonic()
            g_pre = check_guards(f"cell_start:{arm}_seed{seed}")
            plan = build_arm_seed_batch_plan(arm, fit_cids, sorted_fit_by_loss, pools, orders, seed)

            cell_ckpt = stage_r882_copy(f"{arm}_seed{seed}")
            try:
                st_cell = torch.load(cell_ckpt, map_location=device, weights_only=True)
                net_a.load_state_dict(st_cell["state_dict"], strict=True)
            finally:
                cell_ckpt.unlink(missing_ok=True)

            for p in net_a.parameters():
                p.requires_grad_(False)
            for name, p in net_a.adapter_10tok.named_parameters():
                if not name.startswith("anchor."):
                    p.requires_grad_(True)
            guard_a.lock_dope_weights_and_eval_mode(net_a)
            trainable = [p for _, p in net_a.named_parameters() if p.requires_grad]
            if sum(p.numel() for p in trainable) != TRAINABLE_PARAMS:
                raise RuntimeError("Trainable parameter count mismatch")

            dope_pre = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
            init_sha = h53b.adapter_sha(net_a)
            init_vec = torch.cat([p.detach().reshape(-1).clone() for p in trainable], dim=0)

            optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            guard_a.assert_optimizer_excludes_dope(optimizer, net_a)

            step_losses: list[float] = []
            step_mean_losses: list[float] = []
            step_harm_penalties: list[float] = []
            step_frozen_losses: list[float] = []
            pre_clip_norms: list[float] = []
            post_clip_norms: list[float] = []
            eligible_counts: list[int] = []
            q_fractions: list[float] = []

            for b in plan:
                torch.manual_seed(b["stream_seed"])
                cids = b["context_ids"]
                ixs = list(b["candidate_indices"])
                targets = h59.preregistered_targets(
                    torch.as_tensor(
                        np.stack([pools[c]["fitness"][ix] for c, ix in zip(cids, ixs)]),
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        np.stack([pools[c]["gate"][ix] for c, ix in zip(cids, ixs)]),
                        dtype=torch.bool,
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                preds = pred_a(list(cids), ixs)
                ctx_l = per_context_losses(preds, targets)
                loss, mean_term, harm_penalty = h130_objective(
                    arm,
                    ctx_l,
                    [family_by_cid[cid] for cid in cids],
                    fit_family_baseline_losses,
                    mu=1.0,
                    margin=0.0,
                )
                loss.backward()

                pre_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, BASE_GRAD_CLIP, error_if_nonfinite=True).item()
                )
                post_norm = float(
                    torch.linalg.vector_norm(
                        torch.stack(
                            [torch.linalg.vector_norm(p.grad.detach()) for p in trainable if p.grad is not None]
                        )
                    ).item()
                )
                optimizer.step()

                step_losses.append(float(loss.detach().item()))
                step_mean_losses.append(float(mean_term.detach().item()))
                step_harm_penalties.append(float(harm_penalty.detach().item()))
                step_frozen_losses.append(float(np.mean([fit_baseline_losses[c] for c in cids])))
                pre_clip_norms.append(pre_norm)
                post_clip_norms.append(post_norm)
                eligible_counts.append(int(b["eligible_count"]))
                q_fractions.append(float(b["q_fraction"]))

            dope_post = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
            final_sha = h53b.adapter_sha(net_a)
            final_vec = torch.cat([p.detach().reshape(-1).clone() for p in trainable], dim=0)
            param_l2_drift = float(torch.linalg.vector_norm(final_vec - init_vec).item())
            if final_sha == init_sha or param_l2_drift <= 0.0:
                raise RuntimeError(f"Cell ({arm}, {seed}) did not update adapter parameters!")

            net_a.eval()
            cell_scores = predict_pool_scores_fast(assess_cids, pools, pred_a, batch_size=64)
            cell_eval = h61.evaluate_pool_predictions(assess_cids, pools, cell_scores, eval_mode="unrev")
            g_post = check_guards(f"cell_end:{arm}_seed{seed}")
            cell_wall = time.monotonic() - cell_t0

            cell_doc = {
                "schema": "jope.h130.cell_receipt/v1",
                "wave": "H130",
                "arm": arm,
                "seed": seed,
                "updates": len(step_losses),
                "grad_clip_cap": BASE_GRAD_CLIP,
                "dope_sha256_pre": dope_pre,
                "dope_sha256_post": dope_post,
                "initial_adapter_sha256": init_sha,
                "final_adapter_sha256": final_sha,
                "adapter_l2_drift_from_r882": param_l2_drift,
                "mean_step_loss": float(np.mean(step_losses)),
                "mean_context_loss": float(np.mean(step_mean_losses)),
                "mean_harm_penalty": float(np.mean(step_harm_penalties)),
                "final_step_loss": float(step_losses[-1]),
                "mean_sampled_frozen_loss": float(np.mean(step_frozen_losses)),
                "first5_sampled_frozen_loss": float(np.mean(step_frozen_losses[:5])),
                "last5_sampled_frozen_loss": float(np.mean(step_frozen_losses[-5:])),
                "median_pre_clip_grad_norm": float(np.median(pre_clip_norms)),
                "median_post_clip_grad_norm": float(np.median(post_clip_norms)),
                "eligible_count_u0": eligible_counts[0],
                "eligible_count_u39": eligible_counts[-1],
                "fold0_family_mean_utility": float(cell_eval["mean"]),
                "fold0_context_mean_utility": float(cell_eval["context_mean"]),
                "fold0_per_family_utilities": cell_eval["per_family"],
                "fold0_per_context_utilities": cell_eval["per_context"],
                "elapsed_seconds": round(cell_wall, 3),
                "guards_pre": g_pre,
                "guards_post": g_post,
            }
            c_rel = f"cells/{arm}_seed{seed}.json"
            c_path = CUSTODY_ROOT / c_rel
            c_path.write_text(json.dumps(cell_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            local_c = local_record_and_verify(c_path, c_rel)
            local_cell_receipts[f"{arm}_seed{seed}"] = local_c
            cell_results[(arm, seed)] = cell_doc
            print(
                f"[H130] Cell ({arm}, seed={seed}) OK in {cell_wall:.2f}s: "
                f"fold0_fam_u={cell_eval['mean']:.6f} (r882={eval_a_unrev['mean']:.6f}), "
                f"drift={param_l2_drift:.5f}, sampled_frozen_loss={np.mean(step_frozen_losses):.4f}",
                flush=True,
            )

    # 7. Step 4: Statistical Readout, Holm Step-Down & Elo
    print("[H130] Step 4: Computing 10,000-draw paired family-cluster bootstrap & Elo readout...", flush=True)
    assess_fams_sorted = sorted(assess_fams)

    fam_by_arm_seed: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    ctx_by_arm_seed: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            fam_by_arm_seed[arm][seed] = cell_results[(arm, seed)]["fold0_per_family_utilities"]
            ctx_by_arm_seed[arm][seed] = cell_results[(arm, seed)]["fold0_per_context_utilities"]

    r882_fam_by_seed = {s: eval_a_unrev["per_family"] for s in SEEDS}
    r882_ctx_by_seed = {s: eval_a_unrev["per_context"] for s in SEEDS}

    vs_control = {
        arm: evaluate_arm_vs_comparator(
            arm,
            "CONTROL",
            fam_by_arm_seed[arm],
            fam_by_arm_seed["CONTROL"],
            ctx_by_arm_seed[arm],
            ctx_by_arm_seed["CONTROL"],
            assess_fams_sorted,
            assess_cids,
        )
        for arm in ("HINGE", "CVAR")
    }

    vs_r882 = {
        arm: evaluate_arm_vs_comparator(
            arm,
            "R882",
            fam_by_arm_seed[arm],
            r882_fam_by_seed,
            ctx_by_arm_seed[arm],
            r882_ctx_by_seed,
            assess_fams_sorted,
            assess_cids,
        )
        for arm in ARMS
    }

    hinge_vs_cvar = evaluate_arm_vs_comparator(
        "HINGE",
        "CVAR",
        fam_by_arm_seed["HINGE"],
        fam_by_arm_seed["CVAR"],
        ctx_by_arm_seed["HINGE"],
        ctx_by_arm_seed["CVAR"],
        assess_fams_sorted,
        assess_cids,
    )

    # Holm family-wise correction across the two primary H130 arms vs r882.
    ranked = sorted(("HINGE", "CVAR"), key=lambda arm: vs_r882[arm]["p_value_one_sided"])
    holm_vs_r882: dict[str, dict] = {}
    prior_reject = True
    for rank_idx, arm in enumerate(ranked):
        alpha_k = 0.05 / (2 - rank_idx)
        stat = vs_r882[arm]
        p_pass = prior_reject and stat["p_value_one_sided"] <= alpha_k
        prior_reject = p_pass
        gates = {
            "holm_pass": bool(p_pass),
            "lcb95_gt_0_001": bool(stat["lcb95_one_sided"] > 0.001),
            "family_median_nonnegative": bool(stat["pooled_family_median_gain"] >= 0.0),
            "family_harm_le_10pct": bool(stat["pooled_family_harm_fraction"] <= 0.10),
        }
        admitted = all(gates.values())
        holm_vs_r882[arm] = {"rank": rank_idx + 1, "alpha_threshold": alpha_k,
                             "p_value_one_sided": stat["p_value_one_sided"], "gates": gates,
                             "admitted": admitted}
        stat["holm_alpha"] = alpha_k
        stat["holm_pass"] = bool(p_pass)
        stat["admitted"] = admitted

    # CONTROL and direct HINGE/CVAR comparisons are secondary only.
    hinge_vs_cvar = evaluate_arm_vs_comparator(
        "HINGE", "CVAR", fam_by_arm_seed["HINGE"], fam_by_arm_seed["CVAR"],
        ctx_by_arm_seed["HINGE"], ctx_by_arm_seed["CVAR"], assess_fams_sorted, assess_cids,
    )
    candidate_breakthrough = any(holm_vs_r882[arm]["admitted"] for arm in ("HINGE", "CVAR"))

    # Joint Bradley-Terry Elo across {R882, CONTROL, HINGE, CVAR}
    bt_records = []
    for seed in SEEDS:
        for cid in assess_cids:
            bt_records.append(
                elo_mod.DatasetEvalRecord(
                    dataset_id=f"{cid}__seed{seed}",
                    family_id=family_by_cid[cid],
                    task_type="regression",
                    scores_by_contender={
                        "R882": float(eval_a_unrev["per_context"][cid]),
                        "CONTROL": float(ctx_by_arm_seed["CONTROL"][seed][cid]),
                        "HINGE": float(ctx_by_arm_seed["HINGE"][seed][cid]),
                        "CVAR": float(ctx_by_arm_seed["CVAR"][seed][cid]),
                    },
                )
            )
    bt_vs_control = elo_mod.compute_family_grouped_bradley_terry_elo(
        records=bt_records,
        anchor_contender="CONTROL",
        anchor_elo=1500.0,
        prior=0.01,
    )
    bt_vs_r882 = elo_mod.compute_family_grouped_bradley_terry_elo(
        records=bt_records,
        anchor_contender="R882",
        anchor_elo=1500.0,
        prior=0.01,
    )

    any_admitted = any(holm_vs_r882[a]["admitted"] for a in ("HINGE", "CVAR"))
    verdict = "ADMITTED" if any_admitted else "NO_ADMISSION"
    total_wall = time.monotonic() - t0
    g_final = check_guards("wave_complete")

    summary = {
        "schema": "jope.h130.metrics/v1",
        "wave": "H130",
        "status": "COMPLETE",
        "complete": True,
        "verdict": verdict,
        "candidate_breakthrough": candidate_breakthrough,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(total_wall, 2),
        "gpu_seconds": 0.0,
        "cells_completed": 9,
        "optimizer_updates": 9 * UPDATES_PER_CELL,
        "dl_val_reads": 0,
        "outer_test_reads": 0,
        "sealed_metatest_reads": 0,
        "live_loop_reads": 0,
        "provenance": {
            "r882_sha256": R882_SHA,
            "dope_sha256": DOPE_SHA,
            "fused_sha256": FUSED_SHA,
            "selected_sha256": SELECTED_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "prereg_sha256": prereg_sha,
            "fold_seed": FOLD_SEED,
            "fold_manifest_sha256": FOLD_SHA,
            "fold_manifest_local_receipt": local_fold,
            "seeds": list(SEEDS),
            "assessment_fold": 0,
            "assessment_families": 41,
            "assessment_contexts": len(assess_cids),
            "fit_folds": [1, 2, 3, 4],
            "fit_families": 164,
            "fit_contexts": len(fit_cids),
        },
        "regression_gate": {
            "status": "PASS",
            "r882_fold0_unrev_family_utility": eval_a_unrev["mean"],
            "r882_fold0_unrev_context_utility": eval_a_unrev["context_mean"],
            "r882_fold0_native_family_utility": eval_a_native["mean"],
            "two_copy_fold0_pred_max_abs_diff": max_pred_diff_fold0,
            "two_copy_fold0_utility_max_abs_diff": utility_max_abs_diff,
            "two_copy_control_loss_abs_diff": control_loss_abs_diff,
            "two_copy_control_grad_max_abs_diff": control_grad_max_abs_diff,
            "local_receipt": local_reg,
        },
        "frozen_fit_family_r882_losses": {
            "fit_family_count": len(fit_family_baseline_losses),
            "assessment_families_touched": 0,
            "family_baseline_sha256": hashlib.sha256(
                json.dumps(fit_family_baseline_losses, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "adapter_mutated": False,
            "local_receipt": local_cal,
        },
        "secondary_vs_control": vs_control,
        "primary_vs_r882": vs_r882,
        "holm_step_down_vs_r882": holm_vs_r882,
        "secondary_hinge_vs_cvar": hinge_vs_cvar,
        "bradley_terry_elo_vs_control_1500": bt_vs_control["ratings"],
        "bradley_terry_elo_vs_r882_1500": bt_vs_r882["ratings"],
        "cells": {
            f"{arm}_seed{s}": {
                "arm": arm,
                "seed": s,
                "updates": cell_results[(arm, s)]["updates"],
                "grad_clip_cap": cell_results[(arm, s)]["grad_clip_cap"],
                "initial_adapter_sha256": cell_results[(arm, s)]["initial_adapter_sha256"],
                "final_adapter_sha256": cell_results[(arm, s)]["final_adapter_sha256"],
                "adapter_l2_drift_from_r882": cell_results[(arm, s)]["adapter_l2_drift_from_r882"],
                "mean_step_loss": cell_results[(arm, s)]["mean_step_loss"],
                "final_step_loss": cell_results[(arm, s)]["final_step_loss"],
                "mean_sampled_frozen_loss": cell_results[(arm, s)]["mean_sampled_frozen_loss"],
                "first5_sampled_frozen_loss": cell_results[(arm, s)]["first5_sampled_frozen_loss"],
                "last5_sampled_frozen_loss": cell_results[(arm, s)]["last5_sampled_frozen_loss"],
                "median_pre_clip_grad_norm": cell_results[(arm, s)]["median_pre_clip_grad_norm"],
                "median_post_clip_grad_norm": cell_results[(arm, s)]["median_post_clip_grad_norm"],
                "eligible_count_u0": cell_results[(arm, s)]["eligible_count_u0"],
                "eligible_count_u39": cell_results[(arm, s)]["eligible_count_u39"],
                "fold0_family_mean_utility": cell_results[(arm, s)]["fold0_family_mean_utility"],
                "fold0_context_mean_utility": cell_results[(arm, s)]["fold0_context_mean_utility"],
                "elapsed_seconds": cell_results[(arm, s)]["elapsed_seconds"],
                "local_receipt": local_cell_receipts[f"{arm}_seed{s}"],
            }
            for s in SEEDS
            for arm in ARMS
        },
        "final_guards": g_final,
    }

    sum_path = CUSTODY_ROOT / "h130_summary.json"
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(summary_bytes) > 100_000:
        raise RuntimeError(f"H130 summary exceeds 100 KB: {len(summary_bytes)} bytes")
    sum_path.write_bytes(summary_bytes)
    local_sum = local_record_and_verify(sum_path, "h130_summary.json")
    receipt_path = CUSTODY_ROOT / "h130_summary.receipt.json"
    receipt_path.write_text(json.dumps(local_sum, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"[H130] Wave COMPLETE! Verdict={verdict}, local receipt={local_sum['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    if "--synthetic-smoke" in sys.argv[1:]:
        raise SystemExit(synthetic_smoke())
    if "--execute-wave" not in sys.argv[1:]:
        raise SystemExit("H130 execution is gated; use --synthetic-smoke or explicitly pass --execute-wave after preregistration merge")
    raise SystemExit(main())
