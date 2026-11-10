"""H301 (h301_paired_control_drift_meta): paired-control drift meta-analysis & flywheel winner's-curse estimator."""

from __future__ import annotations

from typing import Any
import math
import numpy as np

R3175_DEV91_MEAN = 1.1389572403748425
EVAL_IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 30100
TIE_TOL = 1e-12


def family_groups(dev_families: list[str]) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Partition the 91 dev contexts into the 19 sorted unique family_id clusters."""
    fids = sorted(set(dev_families))
    arr = np.asarray(dev_families)
    groups = [np.flatnonzero(arr == f) for f in fids]
    sizes = np.array([len(g) for g in groups], dtype=np.float64)
    return fids, groups, sizes


def make_family_bootstrap_counts(
    n_families: int = 19,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Generate (n_boot, n_families) cluster resample multiplicity matrix."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_families, size=(n_boot, n_families))
    counts = np.zeros((n_boot, n_families), dtype=np.float64)
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1.0)
    return counts


def evaluate_trajectory_vs_r3175(
    wave: str,
    label: str,
    seed: int | str,
    n_updates: int,
    lr: float | None,
    cand_per_context: list[float] | np.ndarray,
    cand_programs: list[str] | None,
    r3175_per_context: list[float] | np.ndarray,
    r3175_programs: list[str],
    dev_families: list[str],
    counts: np.ndarray,
) -> dict[str, Any]:
    """Compute dev91 mean, lift vs r3175, 19-family 10k bootstrap CI95, W/T/L, changed contexts, and MM BT Elo."""
    cand = np.asarray(cand_per_context, dtype=np.float64)
    base = np.asarray(r3175_per_context, dtype=np.float64)
    assert cand.shape == (91,) and base.shape == (91,)
    delta = cand - base

    _, groups, sizes = family_groups(dev_families)
    den = counts @ sizes
    fam_sums = np.array([float(np.sum(delta[g])) for g in groups], dtype=np.float64)
    boot_lift = (counts @ fam_sums) / den
    lcb95, ucb95 = np.quantile(boot_lift, [0.025, 0.975]).tolist()

    wins = int(np.sum(delta > TIE_TOL))
    losses = int(np.sum(delta < -TIE_TOL))
    ties = int(91 - wins - losses)
    if cand_programs is not None:
        changed = int(sum(a != b for a, b in zip(cand_programs, r3175_programs)))
    else:
        changed = int(wins + losses)

    # MM Bradley-Terry 2-player exact rate (r3175 anchored at 1500)
    scores = np.where(delta > 1e-4, 1.0, np.where(delta < -1e-4, 0.0, 0.5))
    fam_w = np.array([float(np.sum(scores[g])) for g in groups], dtype=np.float64)
    fam_l = np.array([float(np.sum(1.0 - scores[g])) for g in groups], dtype=np.float64)
    w_tot = float(np.sum(fam_w))
    l_tot = float(np.sum(fam_l))
    elo_point = float(1500.0 + 400.0 * np.log10((w_tot + 0.01) / (l_tot + 0.01)))
    boot_w = counts @ fam_w
    boot_l = counts @ fam_l
    boot_elo = 1500.0 + 400.0 * np.log10((boot_w + 0.01) / (boot_l + 0.01))
    elo_lo, elo_hi = np.quantile(boot_elo, [0.025, 0.975]).tolist()

    return {
        "wave": wave,
        "label": label,
        "seed": seed,
        "n_updates": int(n_updates),
        "lr": float(lr) if lr is not None else None,
        "dev91_mean": float(np.mean(cand)),
        "lift_vs_r3175": float(np.mean(delta)),
        "ci95_vs_r3175": [float(lcb95), float(ucb95)],
        "wtl_vs_r3175": {"wins": wins, "ties": ties, "losses": losses},
        "changed_contexts": changed,
        "elo_vs_r3175": {
            "point": elo_point,
            "ci95": [float(elo_lo), float(elo_hi)],
        },
        "_boot_lift": boot_lift,
        "_boot_elo": boot_elo,
        "_delta": delta,
    }


def summarize_wave_group(
    wave_id: str,
    n_updates: int,
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a wave/horizon group across its seeds/LRs with 19-family 10k bootstrap CI95."""
    lifts = np.array([t["lift_vs_r3175"] for t in trajectories], dtype=np.float64)
    means = np.array([t["dev91_mean"] for t in trajectories], dtype=np.float64)
    chgs = [int(t["changed_contexts"]) for t in trajectories]
    elos = np.array([t["elo_vs_r3175"]["point"] for t in trajectories], dtype=np.float64)

    boot_lifts = np.stack([t["_boot_lift"] for t in trajectories], axis=0)
    boot_med_lift = np.median(boot_lifts, axis=0)
    ci_lo, ci_hi = np.quantile(boot_med_lift, [0.025, 0.975]).tolist()

    boot_elos = np.stack([t["_boot_elo"] for t in trajectories], axis=0)
    boot_med_elo = np.median(boot_elos, axis=0)
    elo_lo, elo_hi = np.quantile(boot_med_elo, [0.025, 0.975]).tolist()

    return {
        "wave": wave_id,
        "n_updates": int(n_updates),
        "n_trajectories": int(len(trajectories)),
        "median_dev91_mean": float(np.median(means)),
        "mean_dev91_mean": float(np.mean(means)),
        "median_lift_vs_r3175": float(np.median(lifts)),
        "median_lift_ci95": [float(ci_lo), float(ci_hi)],
        "median_changed_contexts": float(np.median(chgs)),
        "median_elo_vs_r3175": float(np.median(elos)),
        "median_elo_ci95": [float(elo_lo), float(elo_hi)],
        "_boot_median_lift": boot_med_lift,
        "_boot_median_elo": boot_med_elo,
    }


def fit_drift_slope_bootstrap(
    x_values: np.ndarray,
    y_point: np.ndarray,
    y_boot: np.ndarray,
) -> dict[str, Any]:
    """OLS slope and intercept of y vs x with 19-family 10k bootstrap CI95.

    x_values: shape (K,)
    y_point: shape (K,)
    y_boot: shape (K, n_boot)
    """
    x = np.asarray(x_values, dtype=np.float64)
    xm = float(np.mean(x))
    xd = x - xm
    sxx = float(np.sum(xd * xd))
    if sxx <= 0.0:
        raise ValueError("x_values must have positive variance")

    slope_pt = float(np.sum(xd * (y_point - np.mean(y_point))) / sxx)
    intercept_pt = float(np.mean(y_point) - slope_pt * xm)

    y_boot_mean = np.mean(y_boot, axis=0)
    slope_boot = (xd @ (y_boot - y_boot_mean[None, :])) / sxx
    intercept_boot = y_boot_mean - slope_boot * xm

    s_lo, s_hi = np.quantile(slope_boot, [0.025, 0.975]).tolist()
    i_lo, i_hi = np.quantile(intercept_boot, [0.025, 0.975]).tolist()
    return {
        "slope_per_unit": slope_pt,
        "slope_ci95": [float(s_lo), float(s_hi)],
        "slope_strictly_negative_95": bool(s_hi < 0.0),
        "intercept": intercept_pt,
        "intercept_ci95": [float(i_lo), float(i_hi)],
    }


def normal_ppf(p: float) -> float:
    """Acklam / Beasley-Springer rational approximation to standard normal inverse CDF."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    # Python 3.8+ statistics.NormalDist().inv_cdf is exact
    import statistics
    return float(statistics.NormalDist().inv_cdf(p))


def analyze_winners_curse(
    r3175_per_context: np.ndarray,
    post_r3175_utilities: np.ndarray,
    post_r3175_rounds: np.ndarray,
    all_admission_utilities: np.ndarray,
    dev_families: list[str],
    counts: np.ndarray,
) -> dict[str, Any]:
    """Compute flywheel val-history distribution, expected max under per-round spread, and r3175 vs q95 in SE units."""
    base = np.asarray(r3175_per_context, dtype=np.float64)
    r3175_mean = float(np.mean(base))
    _, groups, sizes = family_groups(dev_families)
    den = counts @ sizes

    # 19-family bootstrap SE of r3175 dev91 utility across the 19 dev families
    fam_sums_r3175 = np.array([float(np.sum(base[g])) for g in groups], dtype=np.float64)
    boot_r3175 = (counts @ fam_sums_r3175) / den
    se_fam_r3175 = float(np.std(boot_r3175, ddof=1))

    # Post-r3175 candidate means from H289 history.npz (rounds 3176..4675, all 2-update proposals from r3175)
    post_means = np.mean(post_r3175_utilities, axis=1)
    n_post = int(len(post_means))
    mu_post = float(np.mean(post_means))
    sd_post = float(np.std(post_means, ddof=1))
    med_post = float(np.median(post_means))
    q25_post, q75_post, q90_post, q95_post, q99_post = np.quantile(
        post_means, [0.25, 0.75, 0.90, 0.95, 0.99]
    ).tolist()
    max_post = float(np.max(post_means))
    min_post = float(np.min(post_means))

    # Find the recorded post-r3175 candidate closest to q95_post to also compute paired SE_fam(r3175 - q95_candidate)
    q95_idx = int(np.argmin(np.abs(post_means - q95_post)))
    q95_cand_vec = post_r3175_utilities[q95_idx]
    diff_q95_vec = base - q95_cand_vec
    fam_sums_diff_q95 = np.array([float(np.sum(diff_q95_vec[g])) for g in groups], dtype=np.float64)
    boot_diff_q95 = (counts @ fam_sums_diff_q95) / den
    se_fam_paired_q95 = float(np.std(boot_diff_q95, ddof=1))

    # Expected max of N draws under Gaussian order statistics: mu + sd * Phi^{-1}(N / (N + 1))
    z_post = normal_ppf(n_post / (n_post + 1.0))
    expected_max_gaussian_post = float(mu_post + sd_post * z_post)

    # Full admission log (all rounds with evaluation_identity == EVAL_IDENTITY)
    n_all = int(len(all_admission_utilities))
    mu_all = float(np.mean(all_admission_utilities))
    sd_all = float(np.std(all_admission_utilities, ddof=1))
    q95_all = float(np.quantile(all_admission_utilities, 0.95))
    q99_all = float(np.quantile(all_admission_utilities, 0.99))
    z_all = normal_ppf(n_all / (n_all + 1.0))
    expected_max_gaussian_all = float(mu_all + sd_all * z_all)

    gap_r3175_minus_q95_post = float(r3175_mean - q95_post)
    gap_r3175_minus_q95_all = float(r3175_mean - q95_all)

    ratio_se_marginal_post = float(gap_r3175_minus_q95_post / se_fam_r3175)
    ratio_se_paired_post = float(gap_r3175_minus_q95_post / se_fam_paired_q95) if se_fam_paired_q95 > 0 else float("inf")
    ratio_se_marginal_all = float(gap_r3175_minus_q95_all / se_fam_r3175)

    return {
        "r3175_observed_mean": r3175_mean,
        "r3175_family_bootstrap_se": se_fam_r3175,
        "r3175_paired_vs_q95_family_bootstrap_se": se_fam_paired_q95,
        "post_r3175_2update_distribution": {
            "n_rounds": n_post,
            "first_round": int(post_r3175_rounds[0]),
            "last_round": int(post_r3175_rounds[-1]),
            "mean": mu_post,
            "std": sd_post,
            "median": med_post,
            "iqr": float(q75_post - q25_post),
            "min": min_post,
            "q90": float(q90_post),
            "q95": float(q95_post),
            "q99": float(q99_post),
            "max_observed_rejected": max_post,
            "expected_max_gaussian_order_stat": expected_max_gaussian_post,
            "r3175_minus_mean": float(r3175_mean - mu_post),
            "r3175_z_score_in_round_std": float((r3175_mean - mu_post) / sd_post),
            "r3175_minus_q95": gap_r3175_minus_q95_post,
            "r3175_minus_q95_in_marginal_family_se": ratio_se_marginal_post,
            "r3175_minus_q95_in_paired_family_se": ratio_se_paired_post,
            "exceeds_q95_by_lt_2_marginal_se": bool(gap_r3175_minus_q95_post < 2.0 * se_fam_r3175),
            "exceeds_q95_by_lt_2_paired_se": bool(gap_r3175_minus_q95_post < 2.0 * se_fam_paired_q95),
        },
        "all_identity_matched_admissions_distribution": {
            "n_rounds": n_all,
            "mean": mu_all,
            "std": sd_all,
            "q95": q95_all,
            "q99": q99_all,
            "expected_max_gaussian_order_stat": expected_max_gaussian_all,
            "r3175_minus_q95": gap_r3175_minus_q95_all,
            "r3175_minus_q95_in_marginal_family_se": ratio_se_marginal_all,
            "exceeds_q95_by_lt_2_marginal_se": bool(gap_r3175_minus_q95_all < 2.0 * se_fam_r3175),
        },
    }
