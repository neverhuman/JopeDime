"""H204 Bigger Proposer: Preregistered Statistical Analysis & Gate Verification.

Evaluates wider proposer architectures (widths 128 -> 256 -> 512) trained on
labeled context-candidate CV utility data with research-only DOPE unfreeze
(`JOPE_RESEARCH_DOPE_UNFREEZE=1` + `JOPE_RESEARCH_RUN_ID`), against:
  - Native `r882` baseline (`1.1377941898164525` mean first-argmin utility on 91 DL-Val contexts)
  - `R882_CONT`: continuation of actual `r882` loaded width-64 `adapter_10tok` (frozen DOPE)
  - `H180_RAND`: random-initialized width-128 proposer trunk (frozen) + trained linear readout
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
R882_EXPECTED_DL_VAL_MEAN_UTILITY = 1.1377941898164525
R882_PARITY_ATOL = 1e-12
DL_VAL_EXPECTED_CONTEXTS = 91
DL_TRAIN_EXPECTED_FAMILIES = 205
DL_TRAIN_EXPECTED_CONTEXTS = 1219

R882_EXPECTED_DL_VAL_PICKS: tuple[int, ...] = (
    678,
    701,
    486,
    214,
    884,
    380,
    1,
    3,
    481,
    393,
    744,
    1,
    3,
    1142,
    0,
    19,
    445,
    980,
    521,
    0,
    421,
    503,
    245,
    238,
    2,
    310,
    1,
    803,
    274,
    391,
    100,
    795,
    642,
    1,
    3,
    0,
    0,
    1065,
    1078,
    0,
    573,
    432,
    175,
    40,
    461,
    263,
    0,
    2,
    0,
    9,
    1013,
    883,
    264,
    132,
    381,
    305,
    201,
    831,
    3,
    821,
    2,
    0,
    948,
    777,
    69,
    2,
    390,
    561,
    1099,
    61,
    1107,
    3,
    1047,
    1,
    656,
    928,
    269,
    352,
    544,
    241,
    0,
    281,
    2,
    1,
    506,
    471,
    354,
    694,
    712,
    1,
    172,
)
R882_EXPECTED_DL_VAL_PICKS_SHA256 = (
    "460560e44b8d77e445f18db063601c8d02f748718376fa355d6997e0c3624d96"
)
R882_EXPECTED_DL_VAL_UTILITIES_SHA256 = (
    "77fd7f5963905a1aa0fd321f4ae04fe24328961357aa745efb18bbe0ebfec8e9"
)

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
ALL_ARMS: tuple[str, ...] = (
    "R882_CONT",
    "H180_RAND",
    "PROPOSER_W128",
    "PROPOSER_W256",
    "PROPOSER_W512",
)

S3_EVIDENCE_PREFIX = (
    "s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/"
)
S3_DATASETS_BUCKET = "s3://veox-jopedime-datasets-use2"

# Common promotion / stage gates
GATE_MIN_GAIN = 0.005
GATE_MIN_ELO_LIFT = 10.0
GATE_MIN_POSITIVE_SEEDS = 2
GATE_MAX_CATASTROPHIC_REGRESSION = 0.05
GATE_MAX_FALLBACK_RATE = 0.20
GATE_MAX_HOLM_P = 0.05

# H178 205-family pooled selector parameters
SELECTOR_MIN_DEV_GAIN = 0.001
SELECTOR_MIN_DEV_LCB95 = 0.0
SELECTOR_MAX_DEV_HARM_FRAC = 0.10

# Resource / thread rule (b) & campaign budget constants
MAX_NUMERIC_THREADS = 4
MAX_CPU_UTIL_PCT = 85.0
MAX_RSS_BYTES = 6 * (1024**3)
MAX_LOCAL_CACHE_BYTES = 20 * (1024**3)
MAX_DISK_USED_RATIO = 0.938
MIN_DISK_FREE_BYTES = 40 * (1024**3)
MAX_LABELER_WORKERS = 2
REQUIRED_NICE = 19
MAX_GPU_HOURS = 30.0
MAX_CPU_CORE_HOURS = 200.0
MAX_WALL_HOURS = 12.0


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
    dl_val_utilities: np.ndarray,
    dl_val_picks: np.ndarray | list[int] | tuple[int, ...] = R882_EXPECTED_DL_VAL_PICKS,
    expected_mean: float = R882_EXPECTED_DL_VAL_MEAN_UTILITY,
    expected_picks: tuple[int, ...] = R882_EXPECTED_DL_VAL_PICKS,
    atol: float = R882_PARITY_ATOL,
    expected_n: int = DL_VAL_EXPECTED_CONTEXTS,
) -> dict[str, Any]:
    """Verify 91/91 exact native `r882` DL-Val picks and mean utility (`1.1377941898164525`)."""
    arr = np.asarray(dl_val_utilities, dtype="<f8")
    picks_arr = np.asarray(dl_val_picks, dtype="<i8")
    expected_picks_arr = np.asarray(expected_picks, dtype="<i8")

    n_ctx = int(arr.shape[0])
    finite_all = bool(np.all(np.isfinite(arr)))
    mean_u = float(np.mean(arr)) if n_ctx > 0 else float("nan")
    abs_diff = abs(mean_u - expected_mean) if np.isfinite(mean_u) else float("inf")

    exact_pick_matches = (
        int(np.sum(picks_arr == expected_picks_arr))
        if picks_arr.shape == expected_picks_arr.shape
        else 0
    )
    picks_exact_91_of_91 = bool(
        picks_arr.shape == expected_picks_arr.shape and exact_pick_matches == expected_n
    )
    picks_sha256 = sha256_bytes(picks_arr.tobytes())
    utilities_sha256 = sha256_bytes(arr.tobytes())

    passed = bool(
        n_ctx == expected_n and finite_all and abs_diff <= atol and picks_exact_91_of_91
    )
    return {
        "passed": passed,
        "n_contexts": n_ctx,
        "expected_contexts": expected_n,
        "finite_all": finite_all,
        "exact_pick_matches": exact_pick_matches,
        "picks_exact_91_of_91": picks_exact_91_of_91,
        "picks_sha256": picks_sha256,
        "expected_picks_sha256": R882_EXPECTED_DL_VAL_PICKS_SHA256,
        "utilities_sha256": utilities_sha256,
        "mean_utility": mean_u,
        "expected_mean_utility": expected_mean,
        "abs_diff": abs_diff,
        "atol": atol,
    }


def first_argmin_cost(scores: np.ndarray, outcomes: np.ndarray) -> tuple[int, float]:
    """Fixed-dev-cost-argmin-negative-fitness/v1: pick `int(np.argmin(scores))` and return its outcome."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    outcomes_arr = np.asarray(outcomes, dtype=np.float64)
    if (
        scores_arr.ndim != 1
        or outcomes_arr.ndim != 1
        or scores_arr.shape[0] != outcomes_arr.shape[0]
    ):
        raise ValueError(
            f"Invalid shapes for first_argmin_cost: {scores_arr.shape} vs {outcomes_arr.shape}"
        )
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("Non-finite candidate scores encountered")
    pick_idx = int(np.argmin(scores_arr))
    return pick_idx, float(outcomes_arr[pick_idx])


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
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    """Compute paired family-cluster bootstrap CI and one-sided p-value."""
    base = np.asarray(r882_costs, dtype=np.float64)
    cand = np.asarray(candidate_costs, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {cand.shape}")

    ctx_gains = (cand - base) if higher_is_better else (base - cand)
    mean_gain = float(np.mean(ctx_gains))
    median_gain = float(np.median(ctx_gains))
    max_regression = float(np.max(-ctx_gains)) if ctx_gains.size > 0 else 0.0
    harm_frac_ctx = float(np.mean(ctx_gains < -0.01)) if ctx_gains.size > 0 else 0.0

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

    wins = float(np.sum(ctx_gains > 1e-12))
    losses = float(np.sum(ctx_gains < -1e-12))
    ties = float(base.size - wins - losses)
    win_score = (wins + 0.5 * ties) / max(1.0, float(base.size))
    win_clipped = min(max(win_score, 1e-4), 1.0 - 1e-4)
    elo_lift = float(400.0 * math.log10(win_clipped / (1.0 - win_clipped)))

    return {
        "n_contexts": int(base.size),
        "n_families": int(n_fams),
        "n_boot": int(n_boot),
        "mean_r882": float(np.mean(base)),
        "mean_candidate": float(np.mean(cand)),
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
        "family_gains": fam_gains.tolist(),
    }


def evaluate_h178_selector(
    r882_oof_costs: np.ndarray,
    cand_oof_costs: np.ndarray,
    family_ids: list[str] | np.ndarray,
    seed: int = BOOT_SEED,
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    """Evaluate the H178 205-family pooled out-of-fold selector (`10,000` bootstrap draws) before DL-Val."""
    stats = paired_family_bootstrap(
        r882_costs=r882_oof_costs,
        candidate_costs=cand_oof_costs,
        family_ids=family_ids,
        n_boot=N_BOOTSTRAP,
        seed=seed,
        higher_is_better=higher_is_better,
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
        "n_bootstrap": int(N_BOOTSTRAP),
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
    holm_adjusted_p: float = 1.0,
    n_boot: int = N_BOOTSTRAP,
    boot_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Evaluate common promotion/stage gate across the 3 preregistered seeds.

    Computes a true paired pooled-family bootstrap CI over the seed-averaged family gains
    (`10,000` resamples) and enforces Holm-Bonferroni significance (`holm_adjusted_p <= 0.05`).
    """
    if not seed_results:
        return {"passed": False, "reason": "no_seed_results"}

    mean_gains = [float(s["dl_val"]["mean_gain"]) for s in seed_results]
    elo_lifts = [float(s["dl_val"]["elo_lift"]) for s in seed_results]
    max_regs = [float(s["dl_val"]["max_regression"]) for s in seed_results]
    fallback_rates = [float(s.get("fallback_rate", 0.0)) for s in seed_results]

    # Paired pooled-family bootstrap across the seed-averaged family gain vector
    fam_gain_rows = [
        np.asarray(
            s["dl_val"].get("family_gains", [s["dl_val"]["mean_gain"]]),
            dtype=np.float64,
        )
        for s in seed_results
    ]
    pooled_fam_gains = np.mean(np.stack(fam_gain_rows, axis=0), axis=0)
    n_fams = int(pooled_fam_gains.shape[0])
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n_fams, size=(n_boot, n_fams))
    boot_means = np.mean(pooled_fam_gains[idx], axis=1)
    pooled_ci_low, pooled_ci_high = np.quantile(boot_means, [0.025, 0.975])
    pooled_p_one_sided = float((1 + np.sum(boot_means <= 0.0)) / (n_boot + 1))

    positive_seeds = sum(1 for g in mean_gains if g > 0.0)
    pooled_mean_gain = float(np.mean(mean_gains))
    pooled_elo_lift = float(np.mean(elo_lifts))
    worst_regression = float(max(max_regs))
    mean_fallback_rate = float(np.mean(fallback_rates))
    beats_controls = bool(pooled_mean_gain > control_best_gain)
    holm_significant = bool(float(holm_adjusted_p) <= GATE_MAX_HOLM_P)

    passed = bool(
        pooled_mean_gain >= GATE_MIN_GAIN
        and float(pooled_ci_low) > 0.0
        and pooled_elo_lift >= GATE_MIN_ELO_LIFT
        and positive_seeds >= GATE_MIN_POSITIVE_SEEDS
        and worst_regression <= GATE_MAX_CATASTROPHIC_REGRESSION
        and mean_fallback_rate <= GATE_MAX_FALLBACK_RATE
        and beats_controls
        and holm_significant
    )
    return {
        "passed": passed,
        "pooled_mean_gain": pooled_mean_gain,
        "pooled_ci95_low": float(pooled_ci_low),
        "pooled_ci95_high": float(pooled_ci_high),
        "pooled_p_one_sided": pooled_p_one_sided,
        "holm_adjusted_p": float(holm_adjusted_p),
        "holm_significant": holm_significant,
        "pooled_elo_lift": pooled_elo_lift,
        "positive_seeds": int(positive_seeds),
        "total_seeds": len(seed_results),
        "worst_regression": worst_regression,
        "mean_fallback_rate": mean_fallback_rate,
        "control_best_gain": float(control_best_gain),
        "beats_controls": beats_controls,
    }
