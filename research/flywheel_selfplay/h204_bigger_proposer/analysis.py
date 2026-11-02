"""H204 Bigger Proposer: Preregistered Statistical Analysis & Gate Verification.

Evaluates wider proposer architectures (widths 128 -> 256 -> 512) trained on
labeled context-candidate CV utility data with research-only DOPE unfreeze
(`JOPE_RESEARCH_DOPE_UNFREEZE=1` + `JOPE_RESEARCH_RUN_ID`), against:
  - Native `r882` baseline (`1.1377941898164525` on 91 DL-Val contexts)
  - `R882_CONT`: compute-matched `r882` continuation control (width 64, frozen DOPE)
  - `H180_RAND`: random-initialized wider proposer + linear readout control
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

WAVE = "H204"
OWNER = "gpu_tokbench_agy"
HOST = "xbabe2"
GPU_DEVICE = "cuda:0"
GPU_NAME_EXPECTED = "NVIDIA GeForce RTX 3090"

AUTHORITY_STRING = (
    '"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): '
    '"Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."'
)

EVALUATOR_CONTRACT_ID = "fixed-dev-cost-argmin-negative-fitness/v1"
R882_EXPECTED_DL_VAL_MEAN_COST = 1.1377941898164525
R882_PARITY_ATOL = 1e-12
DL_VAL_EXPECTED_CONTEXTS = 91
DL_TRAIN_EXPECTED_FAMILIES = 205
DL_TRAIN_EXPECTED_CONTEXTS = 1219

SEEDS: tuple[int, ...] = (204101, 204102, 204103)
FOLD_SEED = 204140
BOOT_SEED = 204141
N_FOLDS = 5
N_BOOTSTRAP = 10_000

WIDTHS: tuple[int, ...] = (128, 256, 512)
STAGE1_WIDTH = 128

CONTROL_ARMS: tuple[str, ...] = ("R882_CONT", "H180_RAND")
PROPOSER_ARMS: tuple[str, ...] = ("PROPOSER_W128", "PROPOSER_W256", "PROPOSER_W512")
STAGE1_ARMS: tuple[str, ...] = ("R882_CONT", "H180_RAND", "PROPOSER_W128")
ALL_ARMS: tuple[str, ...] = ("R882_CONT", "H180_RAND", "PROPOSER_W128", "PROPOSER_W256", "PROPOSER_W512")

S3_EVIDENCE_PREFIX = "s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/"
S3_DATASETS_BUCKET = "s3://veox-jopedime-datasets-use2"

# Common promotion / stage gates
GATE_MIN_GAIN = 0.005
GATE_MIN_ELO_LIFT = 10.0
GATE_MIN_POSITIVE_SEEDS = 2
GATE_MAX_CATASTROPHIC_REGRESSION = 0.05
GATE_MAX_FALLBACK_RATE = 0.20

# H178 205-family pooled selector parameters
SELECTOR_MIN_DEV_GAIN = 0.001
SELECTOR_MIN_DEV_LCB95 = 0.0
SELECTOR_MAX_DEV_HARM_FRAC = 0.10

# Resource / thread rule (b) constants
MAX_NUMERIC_THREADS = 4
MAX_CPU_UTIL_PCT = 85.0
MAX_RSS_BYTES = 6 * (1024**3)
MAX_LOCAL_CACHE_BYTES = 20 * (1024**3)
MAX_DISK_USED_RATIO = 0.938
MIN_DISK_FREE_BYTES = 40 * (1024**3)
MAX_LABELER_WORKERS = 2
REQUIRED_NICE = 19


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def assign_family_folds(
    family_ids: list[str] | np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    """Deterministically partition unique family IDs into `n_folds` folds."""
    unique_fams = sorted({str(f) for f in family_ids})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_fams))
    mapping: dict[str, int] = {}
    for rank, idx in enumerate(perm):
        mapping[unique_fams[int(idx)]] = int(rank % n_folds)
    return mapping


def verify_r882_native_parity(
    dl_val_costs: np.ndarray,
    expected_mean: float = R882_EXPECTED_DL_VAL_MEAN_COST,
    atol: float = R882_PARITY_ATOL,
    expected_n: int = DL_VAL_EXPECTED_CONTEXTS,
) -> dict[str, Any]:
    """Verify 91/91 exact native `r882` DL-Val parity before any candidate forward."""
    arr = np.asarray(dl_val_costs, dtype=np.float64)
    n_ctx = int(arr.shape[0])
    finite_all = bool(np.all(np.isfinite(arr)))
    mean_cost = float(np.mean(arr)) if n_ctx > 0 else float("nan")
    abs_diff = abs(mean_cost - expected_mean) if np.isfinite(mean_cost) else float("inf")
    passed = bool(
        n_ctx == expected_n
        and finite_all
        and abs_diff <= atol
    )
    return {
        "passed": passed,
        "n_contexts": n_ctx,
        "expected_contexts": expected_n,
        "finite_all": finite_all,
        "mean_cost": mean_cost,
        "expected_mean_cost": expected_mean,
        "abs_diff": abs_diff,
        "atol": atol,
    }


def first_argmin_cost(scores: np.ndarray, costs: np.ndarray) -> tuple[int, float]:
    """Fixed-dev-cost-argmin-negative-fitness/v1: pick `int(np.argmin(scores))`."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    costs_arr = np.asarray(costs, dtype=np.float64)
    if scores_arr.ndim != 1 or costs_arr.ndim != 1 or scores_arr.shape[0] != costs_arr.shape[0]:
        raise ValueError(f"Invalid shapes for first_argmin_cost: {scores_arr.shape} vs {costs_arr.shape}")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("Non-finite candidate scores encountered")
    pick_idx = int(np.argmin(scores_arr))
    return pick_idx, float(costs_arr[pick_idx])


def aggregate_family_means(
    values: np.ndarray,
    family_ids: list[str] | np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Aggregate context-level values into sorted unique family means."""
    vals = np.asarray(values, dtype=np.float64)
    fams = [str(f) for f in family_ids]
    if vals.shape[0] != len(fams):
        raise ValueError("Length mismatch between values and family_ids")
    buckets: dict[str, list[float]] = {}
    for fam, v in zip(fams, vals, strict=True):
        buckets.setdefault(fam, []).append(float(v))
    sorted_fams = sorted(buckets.keys())
    fam_means = np.array([np.mean(buckets[f]) for f in sorted_fams], dtype=np.float64)
    return sorted_fams, fam_means


def paired_family_bootstrap(
    r882_costs: np.ndarray,
    candidate_costs: np.ndarray,
    family_ids: list[str] | np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Compute paired family-cluster bootstrap CI and one-sided p-value on `cost(r882) - cost(cand)`."""
    base = np.asarray(r882_costs, dtype=np.float64)
    cand = np.asarray(candidate_costs, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {cand.shape}")

    ctx_gains = base - cand
    mean_gain = float(np.mean(ctx_gains))
    median_gain = float(np.median(ctx_gains))
    max_regression = float(np.max(cand - base)) if ctx_gains.size > 0 else 0.0
    harm_frac_ctx = float(np.mean((cand - base) > 0.01)) if ctx_gains.size > 0 else 0.0

    sorted_fams, fam_gains = aggregate_family_means(ctx_gains, family_ids)
    n_fams = len(sorted_fams)
    fam_mean_gain = float(np.mean(fam_gains)) if n_fams > 0 else 0.0
    harm_frac_fam = float(np.mean(fam_gains < -0.01)) if n_fams > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_fams, size=(n_boot, n_fams))
    boot_means = np.mean(fam_gains[idx], axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    lcb95 = float(np.quantile(boot_means, 0.05))
    p_one_sided = float((1 + np.sum(boot_means <= 0.0)) / (n_boot + 1))

    # Elo lift from win rate across contexts (with 0.5 for ties)
    wins = float(np.sum(cand < base - 1e-12))
    losses = float(np.sum(cand > base + 1e-12))
    ties = float(base.size - wins - losses)
    win_score = (wins + 0.5 * ties) / max(1.0, float(base.size))
    win_clipped = min(max(win_score, 1e-4), 1.0 - 1e-4)
    elo_lift = float(400.0 * math.log10(win_clipped / (1.0 - win_clipped)))

    return {
        "n_contexts": int(base.size),
        "n_families": int(n_fams),
        "mean_r882_cost": float(np.mean(base)),
        "mean_candidate_cost": float(np.mean(cand)),
        "mean_gain": mean_gain,
        "family_mean_gain": fam_mean_gain,
        "median_gain": median_gain,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "lcb95": lcb95,
        "p_one_sided": p_one_sided,
        "elo_lift": elo_lift,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "max_regression": max_regression,
        "harm_frac_context": harm_frac_ctx,
        "harm_frac_family": harm_frac_fam,
    }


def evaluate_h178_selector(
    r882_oof_costs: np.ndarray,
    cand_oof_costs: np.ndarray,
    family_ids: list[str] | np.ndarray,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Evaluate the H178 205-family pooled out-of-fold selector before DL-Val."""
    stats = paired_family_bootstrap(
        r882_costs=r882_oof_costs,
        candidate_costs=cand_oof_costs,
        family_ids=family_ids,
        n_boot=2000,
        seed=seed,
    )
    admitted = bool(
        stats["mean_gain"] >= SELECTOR_MIN_DEV_GAIN
        and stats["lcb95"] > SELECTOR_MIN_DEV_LCB95
        and stats["harm_frac_family"] <= SELECTOR_MAX_DEV_HARM_FRAC
        and stats["max_regression"] <= GATE_MAX_CATASTROPHIC_REGRESSION
    )
    return {
        "admitted": admitted,
        "fallback_to_r882": not admitted,
        "oof_stats": stats,
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Compute Holm-Bonferroni adjusted p-values for a family of hypotheses."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_raw) in enumerate(items):
        adj = min(1.0, float(p_raw) * float(m - rank))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def evaluate_arm_gate(
    seed_results: list[dict[str, Any]],
    control_best_gain: float,
) -> dict[str, Any]:
    """Evaluate common promotion/stage gate across the 3 preregistered seeds."""
    if not seed_results:
        return {"passed": False, "reason": "no_seed_results"}

    mean_gains = [float(s["dl_val"]["mean_gain"]) for s in seed_results]
    ci_lows = [float(s["dl_val"]["ci95_low"]) for s in seed_results]
    elo_lifts = [float(s["dl_val"]["elo_lift"]) for s in seed_results]
    max_regs = [float(s["dl_val"]["max_regression"]) for s in seed_results]
    fallback_rates = [float(s.get("fallback_rate", 0.0)) for s in seed_results]

    positive_seeds = sum(1 for g in mean_gains if g > 0.0)
    pooled_mean_gain = float(np.mean(mean_gains))
    pooled_ci_low = float(np.mean(ci_lows))
    pooled_elo_lift = float(np.mean(elo_lifts))
    worst_regression = float(max(max_regs))
    mean_fallback_rate = float(np.mean(fallback_rates))
    beats_controls = bool(pooled_mean_gain > control_best_gain)

    passed = bool(
        pooled_mean_gain >= GATE_MIN_GAIN
        and pooled_ci_low > 0.0
        and pooled_elo_lift >= GATE_MIN_ELO_LIFT
        and positive_seeds >= GATE_MIN_POSITIVE_SEEDS
        and worst_regression <= GATE_MAX_CATASTROPHIC_REGRESSION
        and mean_fallback_rate <= GATE_MAX_FALLBACK_RATE
        and beats_controls
    )
    return {
        "passed": passed,
        "pooled_mean_gain": pooled_mean_gain,
        "pooled_ci95_low": pooled_ci_low,
        "pooled_elo_lift": pooled_elo_lift,
        "positive_seeds": int(positive_seeds),
        "total_seeds": len(seed_results),
        "worst_regression": worst_regression,
        "mean_fallback_rate": mean_fallback_rate,
        "control_best_gain": float(control_best_gain),
        "beats_controls": beats_controls,
    }
