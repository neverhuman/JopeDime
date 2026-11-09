"""H298 (h298_family_balanced_gradaccum): family-balanced sampling + 4x grad-accum analysis & gates."""

from __future__ import annotations

from typing import Any
import numpy as np

R3175_DEV91_MEAN = 1.1389572403748425
STRICT_MARGIN = 0.001
STRICT_THRESHOLD = R3175_DEV91_MEAN + STRICT_MARGIN
TOP4_CONCENTRATION_CAP = 0.80
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 29800
REPLICATION_SEEDS = (29801, 29802, 29803)


def sample_fambal_cids(
    fam_ids: list[str],
    fam_cids: list[list[str]],
    order_seed: int,
    n_presentations: int,
) -> list[str]:
    """Sample families uniformly, then contexts uniformly within family (temperature 1.0).

    Because draws are sequential per presentation k = 0..n_presentations-1,
    n_presentations=8192 (FAMBAL_ACC1) is an exact prefix of n_presentations=32768 (FAMBAL_ACC4).
    """
    if len(fam_ids) != len(fam_cids) or not fam_ids:
        raise ValueError("nonempty aligned family lists required")
    if n_presentations < 1:
        raise ValueError("n_presentations must be positive")
    rng = np.random.default_rng(order_seed)
    f_idx = rng.integers(0, len(fam_ids), size=n_presentations)
    u = rng.random(n_presentations)
    out: list[str] = []
    for fi, ui in zip(f_idx, u):
        fc = fam_cids[int(fi)]
        out.append(fc[int(ui * len(fc))])
    return out


def family_groups(dev_families: list[str]) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Partition dev91 context indices by sorted unique family_id (19 dev families)."""
    fids = sorted(set(dev_families))
    arr = np.asarray(dev_families)
    groups = [np.flatnonzero(arr == f) for f in fids]
    sizes = np.array([len(g) for g in groups], dtype=np.float64)
    return fids, groups, sizes


def paired_family_bootstrap(
    cand_per_context: list[float] | np.ndarray,
    base_per_context: list[float] | np.ndarray,
    dev_families: list[str],
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """19-family cluster bootstrap (10k draws) of paired per-context delta vs baseline."""
    cand = np.asarray(cand_per_context, dtype=np.float64)
    base = np.asarray(base_per_context, dtype=np.float64)
    if cand.shape != base.shape or cand.ndim != 1 or len(dev_families) != len(cand):
        raise ValueError("mismatched per-context vectors or family labels")
    delta = cand - base
    fids, groups, sizes = family_groups(dev_families)
    n_fam = len(fids)
    fam_sums = np.array([float(np.sum(delta[g])) for g in groups], dtype=np.float64)
    fam_means = np.array([float(np.mean(delta[g])) for g in groups], dtype=np.float64)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_fam, size=(n_boot, n_fam))
    counts = np.zeros((n_boot, n_fam), dtype=np.float64)
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1.0)

    boot_deltas = (counts @ fam_sums) / (counts @ sizes)
    lcb95, ucb95 = np.quantile(boot_deltas, [0.025, 0.975]).tolist()
    return {
        "n_contexts": int(len(delta)),
        "n_families": int(n_fam),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "mean_delta": float(np.mean(delta)),
        "equal_family_mean_delta": float(np.mean(fam_means)),
        "median_family_delta": float(np.median(fam_means)),
        "boot_median": float(np.median(boot_deltas)),
        "lcb95": float(lcb95),
        "ucb95": float(ucb95),
        "ci95": [float(lcb95), float(ucb95)],
        "_boot_samples": boot_deltas,
    }


def top4_positive_gain_concentration(
    cand_per_context: list[float] | np.ndarray,
    base_per_context: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute share of total positive per-context gain carried by the top-4 contexts."""
    cand = np.asarray(cand_per_context, dtype=np.float64)
    base = np.asarray(base_per_context, dtype=np.float64)
    delta = cand - base
    pos = np.sort(delta[delta > 0.0])[::-1]
    total_pos = float(np.sum(pos)) if len(pos) else 0.0
    top4_pos = float(np.sum(pos[:4])) if len(pos) else 0.0
    share = (top4_pos / total_pos) if total_pos > 0.0 else 1.0
    top4_idx = np.argsort(delta)[::-1][:4].tolist()
    return {
        "n_positive_contexts": int(len(pos)),
        "total_positive_gain": total_pos,
        "top4_positive_gain": top4_pos,
        "top4_positive_share": float(share),
        "passes_lt_80pct": bool(total_pos > 0.0 and share < TOP4_CONCENTRATION_CAP),
        "top4_context_indices": [int(i) for i in top4_idx],
        "top4_context_deltas": [float(delta[i]) for i in top4_idx],
    }


def compute_wtl_and_changed(
    cand_per_context: list[float] | np.ndarray,
    base_per_context: list[float] | np.ndarray,
    cand_programs: list[str],
    base_programs: list[str],
    tie_tol: float = 1e-12,
) -> dict[str, int]:
    """Compute W/T/L context counts and changed program selection counts."""
    cand = np.asarray(cand_per_context, dtype=np.float64)
    base = np.asarray(base_per_context, dtype=np.float64)
    delta = cand - base
    wins = int(np.sum(delta > tie_tol))
    losses = int(np.sum(delta < -tie_tol))
    ties = int(len(delta) - wins - losses)
    changed = int(sum(a != b for a, b in zip(cand_programs, base_programs)))
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "changed_contexts": changed,
    }


def compute_mm_bt_elo_with_family_bootstrap(
    cand_per_context: list[float] | np.ndarray,
    base_per_context: list[float] | np.ndarray,
    dev_families: list[str],
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """MM Bradley-Terry Elo (baseline r3175 = 1500) with 19-family bootstrap 95% CI."""
    cand = np.asarray(cand_per_context, dtype=np.float64)
    base = np.asarray(base_per_context, dtype=np.float64)
    delta = cand - base
    scores = np.where(delta > tie_tol, 1.0, np.where(delta < -tie_tol, 0.0, 0.5))

    fids, groups, _ = family_groups(dev_families)
    n_fam = len(fids)
    fam_win = np.array([float(np.sum(scores[g])) for g in groups], dtype=np.float64)
    fam_loss = np.array([float(np.sum(1.0 - scores[g])) for g in groups], dtype=np.float64)

    w_tot = float(np.sum(fam_win))
    l_tot = float(np.sum(fam_loss))
    point_elo = float(1500.0 + 400.0 * np.log10((w_tot + 0.01) / (l_tot + 0.01)))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_fam, size=(n_boot, n_fam))
    counts = np.zeros((n_boot, n_fam), dtype=np.float64)
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1.0)

    boot_w = counts @ fam_win
    boot_l = counts @ fam_loss
    boot_elo = 1500.0 + 400.0 * np.log10((boot_w + 0.01) / (boot_l + 0.01))
    lo, hi = np.quantile(boot_elo, [0.025, 0.975]).tolist()
    return {
        "baseline_elo": 1500.0,
        "point_elo": point_elo,
        "delta_elo": float(point_elo - 1500.0),
        "ci95": [float(lo), float(hi)],
    }


def evaluate_seed_comparison(
    cand_result: dict[str, Any],
    ctrl_result: dict[str, Any],
    r3175_result: dict[str, Any],
    dev_families: list[str],
    n_boot: int = BOOTSTRAP_DRAWS,
    boot_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate a candidate arm seed against r3175 and its paired control."""
    cand_u = cand_result["per_context"]
    ctrl_u = ctrl_result["per_context"]
    base_u = r3175_result["per_context"]
    cand_mean = float(cand_result["mean"])
    ctrl_mean = float(ctrl_result["mean"])
    base_mean = float(r3175_result["mean"])

    boot_vs_r3175 = paired_family_bootstrap(cand_u, base_u, dev_families, n_boot=n_boot, seed=boot_seed)
    boot_vs_ctrl = paired_family_bootstrap(cand_u, ctrl_u, dev_families, n_boot=n_boot, seed=boot_seed + 100)
    conc = top4_positive_gain_concentration(cand_u, base_u)
    wtl_r3175 = compute_wtl_and_changed(
        cand_u, base_u, cand_result["selected_programs"], r3175_result["selected_programs"]
    )
    wtl_ctrl = compute_wtl_and_changed(
        cand_u, ctrl_u, cand_result["selected_programs"], ctrl_result["selected_programs"]
    )
    elo_r3175 = compute_mm_bt_elo_with_family_bootstrap(
        cand_u, base_u, dev_families, n_boot=n_boot, seed=boot_seed
    )

    passes_mean_threshold = bool(cand_mean > base_mean + STRICT_MARGIN)
    passes_paired_control = bool(cand_mean > ctrl_mean)
    passes_lcb95 = bool(boot_vs_r3175["lcb95"] > 0.0)
    passes_concentration = bool(conc["passes_lt_80pct"])
    seed_pass = bool(
        passes_mean_threshold and passes_paired_control and passes_lcb95 and passes_concentration
    )

    boot_r3175_clean = {k: v for k, v in boot_vs_r3175.items() if not k.startswith("_")}
    boot_ctrl_clean = {k: v for k, v in boot_vs_ctrl.items() if not k.startswith("_")}

    return {
        "seed": int(cand_result["seed"]),
        "arm": str(cand_result["arm"]),
        "dev91_mean": cand_mean,
        "paired_control_mean": ctrl_mean,
        "r3175_mean": base_mean,
        "lift_vs_r3175": float(cand_mean - base_mean),
        "lift_vs_paired_control": float(cand_mean - ctrl_mean),
        "bootstrap_vs_r3175": boot_r3175_clean,
        "bootstrap_vs_paired_control": boot_ctrl_clean,
        "top4_concentration_vs_r3175": conc,
        "wtl_vs_r3175": wtl_r3175,
        "wtl_vs_paired_control": wtl_ctrl,
        "elo_vs_r3175": elo_r3175,
        "gates": {
            "mean_gt_r3175_plus_0_001": passes_mean_threshold,
            "mean_gt_paired_control": passes_paired_control,
            "family_bootstrap_lcb95_gt_0": passes_lcb95,
            "top4_positive_share_lt_80pct": passes_concentration,
            "seed_pass": seed_pass,
        },
        "_boot_samples_vs_r3175": boot_vs_r3175["_boot_samples"],
    }


def summarize_arm_across_seeds(seed_evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize an arm across the 3 predeclared replication seeds."""
    lifts_r3175 = [float(s["lift_vs_r3175"]) for s in seed_evals]
    lifts_ctrl = [float(s["lift_vs_paired_control"]) for s in seed_evals]
    means = [float(s["dev91_mean"]) for s in seed_evals]
    # Median-of-seeds lift and 95% CI from the median seed's family bootstrap distribution
    order = np.argsort(lifts_r3175)
    med_idx = int(order[len(order) // 2])
    med_seed_eval = seed_evals[med_idx]
    med_boot = med_seed_eval["_boot_samples_vs_r3175"]
    med_ci95 = np.quantile(med_boot, [0.025, 0.975]).tolist()

    passes_count = sum(1 for s in seed_evals if s["gates"]["seed_pass"])
    clean_seeds = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in seed_evals
    ]
    return {
        "n_seeds": len(seed_evals),
        "seeds_passing_all_gates": int(passes_count),
        "all_3_of_3_pass": bool(passes_count == 3 and len(seed_evals) == 3),
        "dev91_mean_by_seed": {str(s["seed"]): float(s["dev91_mean"]) for s in seed_evals},
        "median_dev91_mean": float(np.median(means)),
        "mean_dev91_mean": float(np.mean(means)),
        "median_lift_vs_r3175": float(np.median(lifts_r3175)),
        "median_seed_family_bootstrap_ci95_vs_r3175": [float(med_ci95[0]), float(med_ci95[1])],
        "seed_range_lift_vs_r3175": [float(min(lifts_r3175)), float(max(lifts_r3175))],
        "median_lift_vs_paired_control": float(np.median(lifts_ctrl)),
        "seed_evaluations": clean_seeds,
    }
