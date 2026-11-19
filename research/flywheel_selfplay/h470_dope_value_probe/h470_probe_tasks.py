"""Part A Diagnostics (`A1`–`A4`) and Part B Value Probe Matrix (`T1`–`T4`, `C0`–`C7`).

Implements:
  - Cross-family donor permutation (`family[donor] != family[i]` on 100% of contexts)
  - Fold-safe standardization & unsupervised top-k SVD interaction features
  - `dvp_score(R, labels, folds, ...)` reusable probe scorer with frozen config SHA-256
  - `profile_oracle` (`A -> B` program-hash split-pool ceiling) & `R_oracle64` / `R_oracle16`
    (TRAIN-fold greedy pivoted-QR common-program basket scored on basket-excluded slates)
  - Part A:
      A1: Stage-1A `Stage1ADistilledBridge` sketch & `nolm` vs `hpc_high`, `campaign_tokens`, & `native_high`
      A2: `L7` audit (`lm48_z` 48D SVD coords vs `R_lm80` 80D triple-reshape distortion)
      A3: True masked 0-reveal vs Wave-0 fake `fitness=100, gate=0` (`k=1`, `k=4`) vs real 4-reveal
      A4: `H410` `n_fit` quartile stability, 128-table row-disjoint split-half stability, & `DVP-S4` 80%-power MDE
  - Part B:
      T1: Fixed `LGBMRanker` (`num_leaves=15`) slate ranking at 0-reveal & 4-reveal over full pools
      T2: Fold-safe multinomial ridge readout for 5-way winning head & 5-way nonlinearity regime
      T3: Fold-safe Ridge held-out family $R^2$ across top-32 common anchor programs
      T4: Fold-safe Ridge prediction of the 5 real FIT-CV `ultra` fine landmarks from `high`
      Contrasts `C0`–`C7` via 10,000-draw `paired_family_bootstrap_ci` + `holm_bonferroni`.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import torch

PROBE_DIR = Path(__file__).resolve().parent.parent / "dope_value_probe"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from dope_value_probe import (  # noqa: E402
    FINE_LANDMARKS,
    holm_bonferroni,
    paired_family_bootstrap_ci,
    placebo_interval_includes_zero,
)

try:
    from .packet_v1_extract import assert_memavail
except ImportError:
    from packet_v1_extract import assert_memavail

HEAD_NAMES_5: tuple[str, ...] = ("elasticnet", "histgbm", "ksmooth", "ridge", "ridge_gcv")
NONLIN_REGIME_NAMES_5: tuple[str, ...] = (
    "heavy_tail",
    "monotone_rank",
    "nonlinear_poly_hinge",
    "interaction_rff",
    "projection_linear_scale",
)
FINE_LANDMARK_INDICES_0BASED: tuple[int, ...] = (17, 18, 19, 20, 22)  # ids 18, 19, 20, 21, 23

DVP_FROZEN_CONFIG: dict[str, Any] = {
    "schema": "jope.dvp.config/v1",
    "partitions": ["h452a/", "h452b/"],
    "n_folds": 5,
    "seeds": [1, 2, 3],
    "bootstrap_draws": 10000,
    "bootstrap_seed": 20261118,
    "t1_lgbm": {
        "objective": "lambdarank",
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.08,
        "num_boost_round": 64,
        "min_data_in_leaf": 25,
        "max_bin": 63,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "train_slate_size": 32,
        "kron_svd_rank": 8,
    },
    "t2_multinomial_ridge_alphas": [0.1, 1.0, 10.0, 100.0, 500.0],
    "t3_ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "t3_num_anchors": 32,
    "oracle_basket_k": 64,
}
DVP_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(DVP_FROZEN_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


# =============================================================================
# Pure Mathematical & Fold-Safe Helpers
# =============================================================================
def make_cross_family_donor_permutation(families: Sequence[str], seed: int) -> np.ndarray:
    """Return a bijection/donor assignment `donor[i]` with `families[donor[i]] != families[i]` everywhere."""
    fam_arr = np.asarray(list(families), dtype=object)
    n = len(fam_arr)
    if n < 2:
        raise ValueError("cross-family donor permutation requires at least 2 contexts")
    unique_fams, counts = np.unique(fam_arr, return_counts=True)
    if len(unique_fams) < 2:
        raise ValueError("cross-family donor permutation requires >= 2 distinct families")
    max_fam_count = int(np.max(counts))
    rng = np.random.default_rng(seed)

    # When max_fam_count <= n // 2, a strict derangement across families ALWAYS exists:
    # group indices after sorting by family (with seeded within-family + family order shuffle)
    # and cyclically shift by max_fam_count!
    if 2 * max_fam_count <= n:
        fam_order = rng.permutation(len(unique_fams))
        ordered_indices: list[int] = []
        for f_idx in fam_order:
            f_val = unique_fams[f_idx]
            members = np.where(fam_arr == f_val)[0]
            ordered_indices.extend(rng.permutation(members).tolist())
        ord_arr = np.asarray(ordered_indices, dtype=np.int64)
        shifted = np.roll(ord_arr, max_fam_count)
        donor = np.empty(n, dtype=np.int64)
        donor[ord_arr] = shifted
    else:
        # Fallback: sample random donor from strictly disjoint families
        donor = np.empty(n, dtype=np.int64)
        for i in range(n):
            valid = np.where(fam_arr != fam_arr[i])[0]
            donor[i] = int(valid[int(rng.integers(0, len(valid)))])

    if np.any(fam_arr[donor] == fam_arr):
        raise AssertionError("cross-family donor invariant violated")
    return donor


def fit_fold_standardizer(x_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_tr.shape[1] == 0:
        return np.zeros((1, 0), dtype=np.float32), np.ones((1, 0), dtype=np.float32)
    mu = np.mean(x_tr.astype(np.float64), axis=0, keepdims=True)
    sd = np.std(x_tr.astype(np.float64), axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_fold_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    z = np.clip((x.astype(np.float32) - mu) / sd, -8.0, 8.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_unsupervised_svd_basis(x_tr_z: np.ndarray, k: int = 8) -> np.ndarray:
    """Unsupervised right-singular basis `[D, k]` fitted strictly on standardized training-fold `R`."""
    d = x_tr_z.shape[1]
    out = np.zeros((d, k), dtype=np.float32)
    if d == 0 or x_tr_z.shape[0] < 2:
        return out
    k_eff = min(k, d, x_tr_z.shape[0])
    _, _, vt = np.linalg.svd(x_tr_z.astype(np.float64), full_matrices=False)
    basis = vt[:k_eff].T  # [D, k_eff]
    # Deterministic sign convention: largest-magnitude coordinate in each singular vector is positive
    for j in range(k_eff):
        col = basis[:, j]
        piv = int(np.argmax(np.abs(col)))
        if col[piv] < 0.0:
            basis[:, j] = -col
    out[:, :k_eff] = basis.astype(np.float32)
    return out


def family_mean_value(values: Sequence[float] | np.ndarray, families: Sequence[str]) -> float:
    buckets: dict[str, list[float]] = {}
    for v, f in zip(values, families):
        buckets.setdefault(str(f), []).append(float(v))
    return float(sum(sum(vs) / len(vs) for vs in buckets.values()) / max(len(buckets), 1))


def contrast_summary(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    families: Sequence[str],
    *,
    n_boot: int = 10000,
    seed: int = 20261118,
) -> dict[str, Any]:
    res = paired_family_bootstrap_ci(left, right, families, n_boot=n_boot, seed=seed)
    res["left_family_mean"] = round(family_mean_value(left, families), 6)
    res["right_family_mean"] = round(family_mean_value(right, families), 6)
    res["left_ctx_mean"] = round(float(np.mean(left)), 6)
    res["right_ctx_mean"] = round(float(np.mean(right)), 6)
    res["includes_zero"] = placebo_interval_includes_zero(res["ci_low"], res["ci_high"])
    return res


def select_pivoted_qr_basket(
    z_tr: np.ndarray,
    support_tr: np.ndarray,
    k: int,
) -> np.ndarray:
    """Deterministic greedy column-pivoted orthogonal QR selection on `[N_train, M_common]`."""
    m = z_tr.shape[1]
    k_eff = min(k, m)
    if k_eff <= 0:
        return np.zeros(0, dtype=np.int64)
    # Weight columns gently by sqrt(coverage) so ubiquitous anchors break near-ties cleanly
    cov_w = np.sqrt(np.maximum(support_tr.astype(np.float64), 1.0) / max(float(z_tr.shape[0]), 1.0))
    work = z_tr.astype(np.float64) * cov_w[None, :]
    work = work - np.mean(work, axis=0, keepdims=True)
    chosen: list[int] = []
    used = np.zeros(m, dtype=bool)
    for _ in range(k_eff):
        col_norms = np.sum(work ** 2, axis=0)
        col_norms[used] = -1.0
        best_j = int(np.argmax(col_norms))
        if col_norms[best_j] <= 1e-12:
            # Fill remaining slots by highest support
            rem = [j for j in np.argsort(-support_tr).tolist() if not used[j]]
            chosen.extend(rem[: (k_eff - len(chosen))])
            break
        chosen.append(best_j)
        used[best_j] = True
        q = work[:, best_j]
        q_norm = math.sqrt(float(np.sum(q ** 2)))
        if q_norm > 1e-12:
            u = q / q_norm
            proj = u @ work
            work = work - np.outer(u, proj)
    return np.asarray(chosen[:k_eff], dtype=np.int64)


# =============================================================================
# Precomputed Pool Summaries & Context Targets (`T1`, `T2`, `T3`, Ceilings)
# =============================================================================
def build_pool_Metadata(
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract fast per-context candidate features, labels, winners, and common-program matrices."""
    n_ctx = len(ordered_cids)
    oracle_utility = np.zeros(n_ctx, dtype=np.float64)
    winner_head_5 = np.zeros(n_ctx, dtype=np.int64)
    winner_nonlin_5 = np.zeros(n_ctx, dtype=np.int64)
    pool_entries: list[dict[str, Any]] = []

    pid_counts: dict[str, int] = {}
    for cid in ordered_cids:
        p = pools[cid]
        for pid in set(p["pids"]):
            pid_counts[pid] = pid_counts.get(pid, 0) + 1

    # Common programs across this split (sorted deterministically by (-count, pid))
    common_pids = [
        pid for pid, cnt in sorted(pid_counts.items(), key=lambda kv: (-kv[1], kv[0])) if cnt >= 80
    ]
    common_pid_to_col = {pid: j for j, pid in enumerate(common_pids)}
    m_common = len(common_pids)
    common_std_adv = np.zeros((n_ctx, m_common), dtype=np.float32)
    common_present = np.zeros((n_ctx, m_common), dtype=np.uint8)
    common_pass = np.zeros((n_ctx, m_common), dtype=np.uint8)

    for idx, cid in enumerate(ordered_cids):
        p = pools[cid]
        prog = p["program"]  # [M, 412] float32
        proj16 = p["proj16"].astype(np.float32)  # [M, 16]
        sem24 = p["struct_sig"].astype(np.float32)  # [M, 24]
        fit = p["fitness"].astype(np.float64)
        gate = p["gate"].astype(bool)
        pids = p["pids"]
        m_c = len(pids)

        util = np.where(gate, -fit, -2.0)
        oracle_u = float(np.max(util)) if m_c > 0 else -2.0
        oracle_utility[idx] = oracle_u

        # Best passing candidate (or least-bad if all fail)
        if np.any(gate):
            pass_idx = np.where(gate)[0]
            best_local = int(pass_idx[int(np.argmin(fit[pass_idx]))])
            pass_u = -fit[pass_idx]
            med_u = float(np.median(pass_u))
            q25, q75 = np.percentile(pass_u, [25.0, 75.0])
            iqr_u = max(float(q75 - q25), float(np.std(pass_u)), 1e-4)
        else:
            best_local = int(np.argmin(fit))
            med_u = -2.0
            iqr_u = 1.0

        # T2 labels from winner's 412D AST vector
        w_pos = np.maximum(prog[best_local], 0.0)
        head_scores = w_pos[24:29]
        winner_head_5[idx] = int(np.argmax(head_scores)) if float(np.max(head_scores)) > 0 else 3
        nl_scores = np.array(
            [
                w_pos[22] + w_pos[1] + w_pos[18],
                w_pos[15] + w_pos[13] + w_pos[20],
                w_pos[2] + w_pos[12] + w_pos[6],
                w_pos[11] + w_pos[10] + w_pos[16] + w_pos[17],
                0.25,  # default linear/scale regime when no nonlinear stage clause is active
            ],
            dtype=np.float64,
        )
        winner_nonlin_5[idx] = int(np.argmax(nl_scores))

        # Within-context standardized passing advantage `z_adv` in [-4, 4] (and -3.0 for gate fail)
        z_adv = np.where(gate, np.clip(((-fit) - med_u) / iqr_u, -4.0, 4.0), -3.0).astype(np.float32)

        # Populate common-program matrix
        for cand_i, pid in enumerate(pids):
            col_j = common_pid_to_col.get(pid)
            if col_j is not None and common_present[idx, col_j] == 0:
                common_present[idx, col_j] = 1
                common_std_adv[idx, col_j] = z_adv[cand_i]
                common_pass[idx, col_j] = 1 if gate[cand_i] else 0

        # Deterministic program-hash split A vs B (`int(pid[:8], 16) % 2`) for `profile_oracle`
        is_half_b = np.array([(int(pid[:8], 16) & 1) == 1 for pid in pids], dtype=bool)
        is_half_a = ~is_half_b
        if not np.any(is_half_a) or not np.any(is_half_b):
            is_half_b = (np.arange(m_c) % 2) == 1
            is_half_a = ~is_half_b

        # Precompute 4-reveal neighbor features relative to eval indices `0..min(4, m_c)-1`
        k_rev = min(4, m_c)
        rev_f = fit[:k_rev]
        rev_g = gate[:k_rev]
        rev_cap_u = np.where(rev_g, -rev_f, -2.5) / 0.4208  # higher is better
        cos_sem_4 = np.clip(sem24 @ sem24[:k_rev].T, -1.0, 1.0)  # [M, k_rev]
        d2_16 = np.sum((proj16[:, None, :] - proj16[:k_rev][None, :, :]) ** 2, axis=-1)  # [M, k_rev]
        w_4 = 1.0 / (0.25 + 0.5 * d2_16 + 2.0 * (1.0 - cos_sem_4))
        w_sum = np.sum(w_4, axis=1, keepdims=True) + 0.25
        ker_u = (w_4 @ rev_cap_u) / w_sum[:, 0]
        ker_g = (w_4 @ rev_g.astype(np.float32)) / w_sum[:, 0]
        best_rev_u = float(np.max(rev_cap_u))
        mean_rev_u = float(np.mean(rev_cap_u))
        rev_feats_6 = np.column_stack(
            [
                ker_u,
                ker_u - mean_rev_u,
                ker_u - best_rev_u,
                ker_g,
                np.max(cos_sem_4, axis=1),
                np.min(d2_16, axis=1),
            ]
        ).astype(np.float32)

        pool_entries.append(
            {
                "cid": cid,
                "pids": pids,
                "sem24": sem24,
                "proj16": proj16,
                "fitness": fit,
                "gate": gate,
                "util": util,
                "z_adv": z_adv,
                "oracle_u": oracle_u,
                "is_half_a": is_half_a,
                "is_half_b": is_half_b,
                "rev4_feats6": rev_feats_6,
            }
        )

    # Build `profile_oracle` 38D representation strictly from half A of each context
    profile_a_38 = np.zeros((n_ctx, 38), dtype=np.float32)
    for idx, pe in enumerate(pool_entries):
        ma = pe["is_half_a"]
        sem_a = pe["sem24"][ma]
        z_a = pe["z_adv"][ma]
        g_a = pe["gate"][ma]
        # 0..9: per-head (mean z_adv, max z_adv) on half A
        head_oh = sem_a[:, :5] > 0.15
        for h_i in range(5):
            mh = head_oh[:, h_i]
            if np.any(mh):
                profile_a_38[idx, 2 * h_i] = float(np.mean(z_a[mh]))
                profile_a_38[idx, 2 * h_i + 1] = float(np.max(z_a[mh]))
        # 10..19: per-regime (mean z_adv, max z_adv) on half A
        reg_dom = np.argmax(sem_a[:, 5:10], axis=1)
        for r_i in range(5):
            mr = reg_dom == r_i
            if np.any(mr):
                profile_a_38[idx, 10 + 2 * r_i] = float(np.mean(z_a[mr]))
                profile_a_38[idx, 10 + 2 * r_i + 1] = float(np.max(z_a[mr]))
        # 20..24: pass rate + spread stats on half A
        profile_a_38[idx, 20] = float(np.mean(g_a))
        if np.any(g_a):
            zp = z_a[g_a]
            profile_a_38[idx, 21:25] = np.percentile(zp, [25.0, 50.0, 75.0, 95.0]).astype(np.float32)
        # 25..37: correlation of `sem_a[:, 10:23]` (13 hyperparam/topology coords) with `z_a`
        zc = z_a - float(np.mean(z_a))
        z_norm = float(np.linalg.norm(zc))
        if z_norm > 1e-6:
            sc = sem_a[:, 10:23] - np.mean(sem_a[:, 10:23], axis=0, keepdims=True)
            s_norms = np.linalg.norm(sc, axis=0)
            valid_s = s_norms > 1e-6
            corrs = np.zeros(13, dtype=np.float32)
            corrs[valid_s] = (sc[:, valid_s].T @ zc) / (s_norms[valid_s] * z_norm)
            profile_a_38[idx, 25:38] = np.clip(corrs, -1.0, 1.0)

    return {
        "oracle_utility": oracle_utility,
        "winner_head_5": winner_head_5,
        "winner_nonlin_5": winner_nonlin_5,
        "pool_entries": pool_entries,
        "common_pids": common_pids,
        "common_pid_to_col": common_pid_to_col,
        "common_std_adv": common_std_adv,
        "common_present": common_present,
        "common_pass": common_pass,
        "profile_a_38": profile_a_38,
    }


# =============================================================================
# Fold-Safe Candidate Priors + T1 / T2 / T3 / T4 Learners
# =============================================================================
def _build_fold_prog_priors(
    pool_entries: list[dict[str, Any]],
    tr_indices: np.ndarray,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute fold-safe per-program empirical stats strictly on `tr_indices` contexts."""
    stats: dict[str, list[float]] = {}
    pass_cnt: dict[str, int] = {}
    tot_cnt: dict[str, int] = {}
    for idx in tr_indices.tolist():
        pe = pool_entries[idx]
        pids = pe["pids"]
        z_adv = pe["z_adv"]
        gate = pe["gate"]
        for i_c, pid in enumerate(pids):
            tot_cnt[pid] = tot_cnt.get(pid, 0) + 1
            if bool(gate[i_c]):
                pass_cnt[pid] = pass_cnt.get(pid, 0) + 1
            stats.setdefault(pid, []).append(float(z_adv[i_c]))
    tau = 5.0
    out: dict[str, tuple[float, float, float, float]] = {}
    for pid, vals in stats.items():
        cnt = len(vals)
        p_pass = (pass_cnt.get(pid, 0) + 2.0) / (cnt + 4.0)
        shrunk_z = float(sum(vals)) / (cnt + tau)
        conf = cnt / (cnt + tau)
        log_c = math.log1p(cnt) / 7.0
        out[pid] = (float(p_pass), float(shrunk_z), float(conf), float(log_c))
    return out


def _cand_base_matrix(
    pe: dict[str, Any],
    prog_priors: Mapping[str, tuple[float, float, float, float]],
    cand_mask: np.ndarray | None,
    use_4rev: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(cand_feats [M_sub, D_cand], sub_indices [M_sub])`."""
    m_all = len(pe["pids"])
    if cand_mask is None:
        sub_idx = np.arange(m_all, dtype=np.int64)
    else:
        sub_idx = np.where(cand_mask)[0]
        if sub_idx.size == 0:
            sub_idx = np.arange(m_all, dtype=np.int64)
    default_pr = (0.5, -0.25, 0.0, 0.0)
    pids = pe["pids"]
    pr_mat = np.array([prog_priors.get(pids[i], default_pr) for i in sub_idx.tolist()], dtype=np.float32)
    parts = [pe["sem24"][sub_idx], pe["proj16"][sub_idx], pr_mat]
    if use_4rev:
        parts.append(pe["rev4_feats6"][sub_idx])
    return np.hstack(parts).astype(np.float32), sub_idx


def _sample_train_slate(
    pe: dict[str, Any],
    cand_mask: np.ndarray | None,
    slate_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `slate_size` training candidates + monotone integer relevance labels `0..15`."""
    m_all = len(pe["pids"])
    elig = np.arange(m_all, dtype=np.int64) if cand_mask is None else np.where(cand_mask)[0]
    if elig.size == 0:
        elig = np.arange(m_all, dtype=np.int64)
    fit = pe["fitness"][elig]
    gate = pe["gate"][elig]
    pass_local = elig[gate]
    fail_local = elig[~gate]

    picked: list[int] = []
    if pass_local.size > 0:
        ord_p = pass_local[np.argsort(pe["fitness"][pass_local])]
        picked.extend(ord_p[: min(10, len(ord_p))].tolist())
    if fail_local.size > 0:
        ord_f = fail_local[np.argsort(pe["fitness"][fail_local])]
        picked.extend(ord_f[: min(8, len(ord_f))].tolist())
    used = set(picked)
    rem = np.array([i for i in elig.tolist() if i not in used], dtype=np.int64)
    need = slate_size - len(picked)
    if need > 0 and rem.size > 0:
        take = rng.choice(rem, size=min(need, rem.size), replace=False)
        picked.extend(take.tolist())
    sel = np.asarray(picked[:slate_size], dtype=np.int64)

    # Monotone relevance 0 (gate fail) and 1..15 (passing ranked by -fitness)
    rel = np.zeros(len(sel), dtype=np.int32)
    g_s = pe["gate"][sel]
    if np.any(g_s):
        p_pos = np.where(g_s)[0]
        u_p = -pe["fitness"][sel[p_pos]]
        ranks = np.argsort(np.argsort(u_p, kind="mergesort"), kind="mergesort")
        if len(p_pos) == 1:
            rel[p_pos[0]] = 15
        else:
            rel[p_pos] = 1 + np.floor(14.0 * ranks / float(len(p_pos) - 1)).astype(np.int32)
            rel[p_pos[int(np.argmax(u_p))]] = 15
    return sel, rel


def _ndcg_at_k(util_sub: np.ndarray, gate_sub: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """ Scale-free NDCG@k using within-pool percentile gain `2^(4*pct) - 1` for passing, `0` for gate fail."""
    m = len(util_sub)
    if m == 0 or not np.any(gate_sub):
        return 0.0
    gains = np.zeros(m, dtype=np.float64)
    p_idx = np.where(gate_sub)[0]
    u_p = util_sub[p_idx]
    if len(p_idx) == 1:
        gains[p_idx[0]] = 15.0
    else:
        pct = np.argsort(np.argsort(u_p, kind="mergesort"), kind="mergesort") / float(len(p_idx) - 1)
        gains[p_idx] = np.expm1(math.log(16.0) * pct)

    k_eff = min(k, m)
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=np.float64))
    order_pred = np.argsort(-pred_scores, kind="mergesort")[:k_eff]
    order_ideal = np.argsort(-gains, kind="mergesort")[:k_eff]
    dcg = float(np.sum(gains[order_pred] * discounts))
    idcg = float(np.sum(gains[order_ideal] * discounts))
    return dcg / idcg if idcg > 1e-12 else 0.0


def evaluate_t1_fold_cell(
    r_mat: np.ndarray,
    pool_entries: list[dict[str, Any]],
    tr_idx: np.ndarray,
    ho_idx: np.ndarray,
    *,
    seed: int,
    use_4rev: bool,
    cand_masks: list[np.ndarray] | None = None,
    prog_priors: Mapping[str, tuple[float, float, float, float]] | None = None,
    num_threads: int = 8,
) -> dict[str, np.ndarray]:
    """Fit fixed `LGBMRanker` (`num_leaves=15`) on `tr_idx` and score full pools on `ho_idx`."""
    assert_memavail(15.0, "t1_fold")
    mu, sd = fit_fold_standardizer(r_mat[tr_idx])
    r_tr_z = apply_fold_standardizer(r_mat[tr_idx], mu, sd)
    r_ho_z = apply_fold_standardizer(r_mat[ho_idx], mu, sd)
    svd8 = fit_unsupervised_svd_basis(r_tr_z, k=8)
    r_tr_p8 = r_tr_z @ svd8  # [N_tr, 8]
    r_ho_p8 = r_ho_z @ svd8  # [N_ho, 8]

    if prog_priors is None:
        prog_priors = _build_fold_prog_priors(pool_entries, tr_idx)

    rng = np.random.default_rng(100003 * seed + int(tr_idx[0]) * 31 + (7 if use_4rev else 0))
    slate_sz = int(DVP_FROZEN_CONFIG["t1_lgbm"]["train_slate_size"])

    x_rows: list[np.ndarray] = []
    y_rels: list[np.ndarray] = []
    groups: list[int] = []
    d_r = r_tr_z.shape[1]

    for local_i, ctx_i in enumerate(tr_idx.tolist()):
        pe = pool_entries[ctx_i]
        cmask = None if cand_masks is None else cand_masks[ctx_i]
        sel, rel = _sample_train_slate(pe, cmask, slate_sz, rng)
        # Build candidate matrix for `sel`
        default_pr = (0.5, -0.25, 0.0, 0.0)
        pids = pe["pids"]
        pr_m = np.array([prog_priors.get(pids[j], default_pr) for j in sel.tolist()], dtype=np.float32)
        c_parts = [pe["sem24"][sel], pe["proj16"][sel], pr_m]
        if use_4rev:
            c_parts.append(pe["rev4_feats6"][sel])
        c_mat = np.hstack(c_parts).astype(np.float32)
        m_s = len(sel)
        if d_r > 0:
            r_rep = np.broadcast_to(r_tr_z[local_i : local_i + 1], (m_s, d_r))
            kron80 = (r_tr_p8[local_i][None, :, None] * c_mat[:, None, :10]).reshape(m_s, 80)
            row_block = np.hstack([c_mat, kron80, r_rep]).astype(np.float32)
        else:
            row_block = c_mat
        x_rows.append(row_block)
        y_rels.append(rel)
        groups.append(m_s)

    x_train = np.vstack(x_rows)
    y_train = np.concatenate(y_rels)
    del x_rows, y_rels

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 8],
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.08,
        "min_data_in_leaf": 25,
        "max_bin": 63,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "seed": int((20261118 + 97 * seed) % 2147483647),
        "num_threads": max(1, int(num_threads)),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dset = lgb.Dataset(
        x_train,
        label=y_train,
        group=groups,
        params={"max_bin": 63, "verbose": -1},
        free_raw_data=True,
    )
    booster = lgb.train(params, dset, num_boost_round=64)
    del x_train, y_train, dset

    n_ho = len(ho_idx)
    ho_util = np.zeros(n_ho, dtype=np.float64)
    ho_regret = np.zeros(n_ho, dtype=np.float64)
    ho_ndcg1 = np.zeros(n_ho, dtype=np.float64)
    ho_ndcg8 = np.zeros(n_ho, dtype=np.float64)
    ho_gatefail = np.zeros(n_ho, dtype=np.float64)

    chunk_sz = 32
    for start in range(0, n_ho, chunk_sz):
        end = min(start + chunk_sz, n_ho)
        blocks: list[np.ndarray] = []
        sub_meta: list[tuple[int, dict[str, Any], np.ndarray]] = []
        for pos in range(start, end):
            ctx_i = int(ho_idx[pos])
            pe = pool_entries[ctx_i]
            cmask = None if cand_masks is None else cand_masks[ctx_i]
            c_mat, sub_idx = _cand_base_matrix(pe, prog_priors, cmask, use_4rev)
            m_s = len(sub_idx)
            if d_r > 0:
                r_rep = np.broadcast_to(r_ho_z[pos : pos + 1], (m_s, d_r))
                kron80 = (r_ho_p8[pos][None, :, None] * c_mat[:, None, :10]).reshape(m_s, 80)
                blocks.append(np.hstack([c_mat, kron80, r_rep]).astype(np.float32))
            else:
                blocks.append(c_mat)
            sub_meta.append((pos, pe, sub_idx))
        x_chunk = np.vstack(blocks)
        preds = np.asarray(booster.predict(x_chunk), dtype=np.float64)
        del blocks, x_chunk
        offset = 0
        for pos, pe, sub_idx in sub_meta:
            m_s = len(sub_idx)
            sc = preds[offset : offset + m_s]
            offset += m_s
            u_sub = pe["util"][sub_idx]
            g_sub = pe["gate"][sub_idx]
            pick_local = int(np.argmax(sc))
            u_pick = float(u_sub[pick_local])
            u_oracle_sub = float(np.max(u_sub))
            ho_util[pos] = u_pick
            ho_regret[pos] = max(0.0, u_oracle_sub - u_pick)
            ho_gatefail[pos] = 0.0 if bool(g_sub[pick_local]) else 1.0
            ho_ndcg1[pos] = _ndcg_at_k(u_sub, g_sub, sc, 1)
            ho_ndcg8[pos] = _ndcg_at_k(u_sub, g_sub, sc, 8)

    return {
        "utility": ho_util,
        "regret": ho_regret,
        "ndcg1": ho_ndcg1,
        "ndcg8": ho_ndcg8,
        "gatefail": ho_gatefail,
    }


def _fit_predict_multinomial_logistic_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """Fold-safe L2-penalized multinomial logistic regression (Newton/L-BFGS via inner holdout alpha)."""
    n_tr, d = x_tr.shape
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64) + 1e-3
    priors = counts / np.sum(counts)
    if d == 0:
        return np.broadcast_to(priors[None, :], (x_ho.shape[0], n_classes)).copy()

    mu, sd = fit_fold_standardizer(x_tr)
    zt = apply_fold_standardizer(x_tr, mu, sd).astype(np.float64)
    zh = apply_fold_standardizer(x_ho, mu, sd).astype(np.float64)

    # Inner 80/20 deterministic split of training fold to select L2 penalty cleanly
    idx = np.arange(n_tr)
    iva = (idx % 5) == 0
    itr = ~iva
    y_oh = np.eye(n_classes, dtype=np.float64)[y_tr]

    def _solve_probs(z_fit: np.ndarray, y_fit_oh: np.ndarray, y_fit_cls: np.ndarray, z_eval: np.ndarray, lam: float) -> np.ndarray:
        # Closed-form ridge log-odds warm start + 25 IRLS/gradient-ascent Newton-diag steps on exact multinomial log-loss
        nf, df = z_fit.shape
        aug_f = np.hstack([np.ones((nf, 1)), z_fit])
        aug_e = np.hstack([np.ones((z_eval.shape[0], 1)), z_eval])
        p_prior = (np.bincount(y_fit_cls, minlength=n_classes).astype(np.float64) + 0.5) / (nf + 0.5 * n_classes)
        w = np.zeros((df + 1, n_classes), dtype=np.float64)
        w[0, :] = np.log(np.maximum(p_prior, 1e-6))
        reg = np.full((df + 1, 1), lam, dtype=np.float64)
        reg[0, 0] = 1e-8
        # Compute L_max Lipschitz bound for exact monotone Nesterov/GD step on strongly convex objective
        s_max = float(np.linalg.norm(aug_f, ord=2) ** 2) / max(nf, 1)
        L_lip = 0.5 * s_max + lam / max(nf, 1)
        step_lr = 0.9 / max(L_lip, 1e-6)
        v = w.copy()
        for t in range(1, 65):
            logits = aug_f @ v
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            grad = (aug_f.T @ (probs - y_fit_oh) + reg * v) / max(nf, 1)
            w_next = v - step_lr * grad
            v = w_next + ((t - 1.0) / (t + 2.0)) * (w_next - w)
            w = w_next
        logits_e = aug_e @ w
        logits_e -= np.max(logits_e, axis=1, keepdims=True)
        exp_e = np.exp(logits_e)
        return exp_e / np.sum(exp_e, axis=1, keepdims=True)

    best_lam = 10.0
    best_ll = float("inf")
    for lam in DVP_FROZEN_CONFIG["t2_multinomial_ridge_alphas"]:
        pr_va = _solve_probs(zt[itr], y_oh[itr], y_tr[itr], zt[iva], float(lam))
        ll = float(-np.mean(np.log(np.clip(pr_va[np.arange(np.sum(iva)), y_tr[iva]], 1e-9, 1.0))))
        if ll < best_ll:
            best_ll = ll
            best_lam = float(lam)

    return _solve_probs(zt, y_oh, y_tr, zh, best_lam)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp + fp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _fit_predict_ridge_multi(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    mask_tr: np.ndarray,
    x_ho: np.ndarray,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> np.ndarray:
    """Fold-safe multi-target Ridge where each target column `j` has validity mask `mask_tr[:, j]`."""
    n_ho = x_ho.shape[0]
    m_tgt = y_tr.shape[1]
    preds = np.zeros((n_ho, m_tgt), dtype=np.float64)
    if x_tr.shape[1] == 0:
        for j in range(m_tgt):
            valid = mask_tr[:, j] > 0
            preds[:, j] = float(np.mean(y_tr[valid, j])) if np.any(valid) else 0.0
        return preds

    mu, sd = fit_fold_standardizer(x_tr)
    zt = apply_fold_standardizer(x_tr, mu, sd).astype(np.float64)
    zh = apply_fold_standardizer(x_ho, mu, sd).astype(np.float64)
    d_scale = float(max(zt.shape[1], 1))
    lam_grid = [float(a) * d_scale for a in alphas]

    # When all targets are 100% observed (e.g. T4), solve all columns simultaneously via SVD + exact LOO MSE!
    if np.all(mask_tr > 0):
        yc_mean = np.mean(y_tr, axis=0, keepdims=True)
        yc = y_tr - yc_mean
        zt_mean = np.mean(zt, axis=0, keepdims=True)
        zc = zt - zt_mean
        zh_c = zh - zt_mean
        u, s, vt = np.linalg.svd(zc, full_matrices=False)
        u_sq = u ** 2
        uty = u.T @ yc  # [r, m_tgt]
        n_f = float(zt.shape[0])
        for j in range(m_tgt):
            best_lam = lam_grid[-1]
            best_loo = float("inf")
            y_j = yc[:, j]
            u_j = uty[:, j]
            for lam in lam_grid:
                filt = (s ** 2) / (s ** 2 + lam)
                yhat_in = u @ (filt * u_j)
                h_diag = (1.0 / n_f) + (u_sq @ filt)
                denom = np.maximum(1.0 - h_diag, 1e-3)
                loo = float(np.mean(((y_j - yhat_in) / denom) ** 2))
                if loo < best_loo:
                    best_loo = loo
                    best_lam = lam
            w_j = vt.T @ ((s / (s ** 2 + best_lam)) * u_j)
            preds[:, j] = yc_mean[0, j] + zh_c @ w_j
        return preds

    for j in range(m_tgt):
        valid = mask_tr[:, j] > 0
        nv = int(np.sum(valid))
        if nv < 8:
            preds[:, j] = float(np.mean(y_tr[valid, j])) if nv > 0 else 0.0
            continue
        zv = zt[valid]
        yv = y_tr[valid, j]
        ym = float(np.mean(yv))
        yc_j = yv - ym
        zm = np.mean(zv, axis=0, keepdims=True)
        zc = zv - zm
        u, s, vt = np.linalg.svd(zc, full_matrices=False)
        u_sq = u ** 2
        uj = u.T @ yc_j
        best_lam = lam_grid[-1]
        best_loo = float("inf")
        for lam in lam_grid:
            filt = (s ** 2) / (s ** 2 + lam)
            yhat_in = u @ (filt * uj)
            h_diag = (1.0 / float(nv)) + (u_sq @ filt)
            denom = np.maximum(1.0 - h_diag, 1e-3)
            loo = float(np.mean(((yc_j - yhat_in) / denom) ** 2))
            if loo < best_loo:
                best_loo = loo
                best_lam = lam
        w = vt.T @ ((s / (s ** 2 + best_lam)) * uj)
        preds[:, j] = ym + (zh - zm) @ w
    return preds


def evaluate_t2_t3_t4_oof(
    r_mat: np.ndarray,
    pool_meta: Mapping[str, Any],
    gains23_ultra: np.ndarray,
    mask23_ultra: np.ndarray,
    folds_list: list[np.ndarray],
    *,
    excluded_anchor_cols_by_pf: Mapping[tuple[int, int], set[int]] | None = None,
    compute_t4: bool = True,
) -> dict[str, Any]:
    """Evaluate T2 (winning head & nonlinearity), T3 (top-32 anchor R²), and T4 (fine landmarks)."""
    n_ctx = r_mat.shape[0]
    n_parts = len(folds_list)
    n_folds = int(np.max(folds_list[0])) + 1

    head_y = pool_meta["winner_head_5"]
    nonlin_y = pool_meta["winner_nonlin_5"]
    head_ll_acc = np.zeros(n_ctx, dtype=np.float64)
    nonlin_ll_acc = np.zeros(n_ctx, dtype=np.float64)
    head_preds_part = np.zeros((n_parts, n_ctx), dtype=np.int64)
    nonlin_preds_part = np.zeros((n_parts, n_ctx), dtype=np.int64)

    # Top-32 common passing anchor programs
    common_pass = pool_meta["common_pass"]
    common_adv = pool_meta["common_std_adv"].astype(np.float64)
    pass_support = np.sum(common_pass, axis=0)
    top_anchor_cols = np.argsort(-pass_support)[:32]

    t3_preds = np.zeros((n_parts, n_ctx, len(top_anchor_cols)), dtype=np.float64)
    t4_targets = gains23_ultra[:, list(FINE_LANDMARK_INDICES_0BASED)].astype(np.float64)
    t4_valid = (mask23_ultra[:, list(FINE_LANDMARK_INDICES_0BASED)] > 0).astype(np.uint8)
    t4_preds = np.zeros((n_parts, n_ctx, 5), dtype=np.float64)

    for p_i, fold_arr in enumerate(folds_list):
        for k in range(n_folds):
            ho_idx = np.where(fold_arr == k)[0]
            tr_idx = np.where(fold_arr != k)[0]

            # T2: 5-way winning head & 5-way nonlinearity regime
            pr_h = _fit_predict_multinomial_logistic_ridge(r_mat[tr_idx], head_y[tr_idx], r_mat[ho_idx], 5)
            pr_nl = _fit_predict_multinomial_logistic_ridge(r_mat[tr_idx], nonlin_y[tr_idx], r_mat[ho_idx], 5)
            head_ll_acc[ho_idx] += -np.log(np.clip(pr_h[np.arange(len(ho_idx)), head_y[ho_idx]], 1e-9, 1.0)) / n_parts
            nonlin_ll_acc[ho_idx] += -np.log(np.clip(pr_nl[np.arange(len(ho_idx)), nonlin_y[ho_idx]], 1e-9, 1.0)) / n_parts
            head_preds_part[p_i, ho_idx] = np.argmax(pr_h, axis=1)
            nonlin_preds_part[p_i, ho_idx] = np.argmax(pr_nl, axis=1)

            # T3: Held-out anchor outcome prediction
            y_t3_tr = common_adv[tr_idx][:, top_anchor_cols]
            m_t3_tr = common_pass[tr_idx][:, top_anchor_cols]
            t3_preds[p_i, ho_idx] = _fit_predict_ridge_multi(
                r_mat[tr_idx], y_t3_tr, m_t3_tr, r_mat[ho_idx], DVP_FROZEN_CONFIG["t3_ridge_alphas"]
            )

            # T4: Fine landmark prediction from `high`
            if compute_t4:
                t4_preds[p_i, ho_idx] = _fit_predict_ridge_multi(
                    r_mat[tr_idx], t4_targets[tr_idx], t4_valid[tr_idx], r_mat[ho_idx], DVP_FROZEN_CONFIG["t3_ridge_alphas"]
                )

    head_macro_f1 = float(np.mean([_macro_f1(head_y, head_preds_part[p_i], 5) for p_i in range(n_parts)]))
    nonlin_macro_f1 = float(np.mean([_macro_f1(nonlin_y, nonlin_preds_part[p_i], 5) for p_i in range(n_parts)]))

    # Compute T3 per-anchor OOF R^2 (excluding any basket-included anchor cols when applicable)
    excluded_union: set[int] = set()
    if excluded_anchor_cols_by_pf:
        for s_ex in excluded_anchor_cols_by_pf.values():
            excluded_union |= set(s_ex)

    anchor_r2s: list[float] = []
    t3_pred_mean = np.mean(t3_preds, axis=0)  # [N_ctx, 32]
    # Per-context squared-error improvement score for paired family bootstrap on T3:
    ctx_t3_skill = np.zeros(n_ctx, dtype=np.float64)
    ctx_t3_cnt = np.zeros(n_ctx, dtype=np.float64)

    for loc_j, glob_col in enumerate(top_anchor_cols.tolist()):
        if glob_col in excluded_union:
            continue
        valid_c = common_pass[:, glob_col] > 0
        if int(np.sum(valid_c)) < 25:
            continue
        yv = common_adv[valid_c, glob_col]
        pv = t3_pred_mean[valid_c, loc_j]
        ym = float(np.mean(yv))
        ss_tot = float(np.sum((yv - ym) ** 2))
        if ss_tot <= 1e-9:
            continue
        ss_res = float(np.sum((yv - pv) ** 2))
        anchor_r2s.append(1.0 - ss_res / ss_tot)
        var_j = ss_tot / float(len(yv))
        ctx_t3_skill[valid_c] += 1.0 - ((yv - pv) ** 2) / max(var_j, 1e-6)
        ctx_t3_cnt[valid_c] += 1.0

    ctx_t3_score = np.where(ctx_t3_cnt > 0, ctx_t3_skill / np.maximum(ctx_t3_cnt, 1.0), 0.0)

    t4_out: dict[str, Any] = {}
    if compute_t4:
        t4_pred_mean = np.mean(t4_preds, axis=0)
        per_lm_r2: dict[str, float] = {}
        for j, (lm_name, _lm_idx) in enumerate(FINE_LANDMARKS):
            vm = t4_valid[:, j] > 0
            yv = t4_targets[vm, j]
            pv = t4_pred_mean[vm, j]
            ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
            r2 = (1.0 - float(np.sum((yv - pv) ** 2)) / ss_tot) if ss_tot > 1e-9 else 0.0
            per_lm_r2[str(lm_name)] = round(float(r2), 6)
        t4_out = {
            "per_landmark_r2": per_lm_r2,
            "mean_r2": round(float(np.mean(list(per_lm_r2.values()))), 6),
            "unexplained_variance_share": round(float(1.0 - np.mean(list(per_lm_r2.values()))), 6),
        }

    return {
        "t2": {
            "head_logloss_ctx": head_ll_acc,
            "head_neg_logloss_ctx": -head_ll_acc,
            "head_logloss_mean": round(float(np.mean(head_ll_acc)), 6),
            "head_macro_f1": round(head_macro_f1, 6),
            "nonlin_logloss_ctx": nonlin_ll_acc,
            "nonlin_neg_logloss_ctx": -nonlin_ll_acc,
            "nonlin_logloss_mean": round(float(np.mean(nonlin_ll_acc)), 6),
            "nonlin_macro_f1": round(nonlin_macro_f1, 6),
        },
        "t3": {
            "n_anchors_scored": len(anchor_r2s),
            "mean_r2": round(float(np.mean(anchor_r2s)) if anchor_r2s else 0.0, 6),
            "median_r2": round(float(np.median(anchor_r2s)) if anchor_r2s else 0.0, 6),
            "pos_r2_frac": round(float(np.mean([r > 0.0 for r in anchor_r2s])) if anchor_r2s else 0.0, 6),
            "ctx_skill": ctx_t3_score,
        },
        "t4": t4_out,
    }


def dvp_score(
    R: np.ndarray,
    labels: Mapping[str, Any],
    folds: Sequence[np.ndarray],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    num_threads: int = 8,
    run_4rev: bool = True,
    compute_t4: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Importable, frozen-configuration DOPE Value Probe scorer (`WP0` step 7).

    Parameters
    ----------
    R:
        `[N_ctx, D]` float32 representation matrix (strictly FIT-row-derived).
    labels:
        Dictionary returned by `build_pool_Metadata` plus `families`, `gains23_ultra`, `mask23_ultra`.
    folds:
        List of partition fold arrays (`[N_ctx]` int64 in `0..K-1`).
    """
    r_arr = np.asarray(R, dtype=np.float32)
    if r_arr.ndim != 2:
        raise ValueError(f"dvp_score expects 2D [N, D] array, got {r_arr.shape}")
    n_ctx = r_arr.shape[0]
    families = labels["families"]
    pool_entries = labels["pool_entries"]
    folds_list = [np.asarray(f, dtype=np.int64) for f in folds]
    n_folds = int(np.max(folds_list[0])) + 1

    n_runs = 1 if smoke else len(folds_list) * len(seeds)
    u0_acc = np.zeros(n_ctx, dtype=np.float64)
    reg0_acc = np.zeros(n_ctx, dtype=np.float64)
    n1_0_acc = np.zeros(n_ctx, dtype=np.float64)
    n8_0_acc = np.zeros(n_ctx, dtype=np.float64)
    gf0_acc = np.zeros(n_ctx, dtype=np.float64)

    u4_acc = np.zeros(n_ctx, dtype=np.float64)
    reg4_acc = np.zeros(n_ctx, dtype=np.float64)
    n1_4_acc = np.zeros(n_ctx, dtype=np.float64)
    n8_4_acc = np.zeros(n_ctx, dtype=np.float64)
    gf4_acc = np.zeros(n_ctx, dtype=np.float64)

    for fold_arr in folds_list:
        for k in range(1 if smoke else n_folds):
            ho_idx = np.where(fold_arr == k)[0]
            tr_idx = np.where(fold_arr != k)[0]
            if len(ho_idx) == 0 or len(tr_idx) == 0:
                continue
            prog_priors = _build_fold_prog_priors(pool_entries, tr_idx)
            for sd in seeds:
                res0 = evaluate_t1_fold_cell(
                    r_arr,
                    pool_entries,
                    tr_idx,
                    ho_idx,
                    seed=int(sd),
                    use_4rev=False,
                    prog_priors=prog_priors,
                    num_threads=num_threads,
                )
                u0_acc[ho_idx] += res0["utility"] / n_runs
                reg0_acc[ho_idx] += res0["regret"] / n_runs
                n1_0_acc[ho_idx] += res0["ndcg1"] / n_runs
                n8_0_acc[ho_idx] += res0["ndcg8"] / n_runs
                gf0_acc[ho_idx] += res0["gatefail"] / n_runs

                if run_4rev:
                    res4 = evaluate_t1_fold_cell(
                        r_arr,
                        pool_entries,
                        tr_idx,
                        ho_idx,
                        seed=int(sd),
                        use_4rev=True,
                        prog_priors=prog_priors,
                        num_threads=num_threads,
                    )
                    u4_acc[ho_idx] += res4["utility"] / n_runs
                    reg4_acc[ho_idx] += res4["regret"] / n_runs
                    n1_4_acc[ho_idx] += res4["ndcg1"] / n_runs
                    n8_4_acc[ho_idx] += res4["ndcg8"] / n_runs
                    gf4_acc[ho_idx] += res4["gatefail"] / n_runs

    aux = evaluate_t2_t3_t4_oof(
        r_arr,
        labels,
        labels["gains23_ultra"],
        labels["mask23_ultra"],
        folds_list,
        compute_t4=compute_t4,
    )
    out = {
        "config_sha256": DVP_CONFIG_SHA256,
        "width": int(r_arr.shape[1]),
        "t1_0rev": {
            "family_mean_utility": round(family_mean_value(u0_acc, families), 6),
            "ctx_mean_utility": round(float(np.mean(u0_acc)), 6),
            "family_mean_regret": round(family_mean_value(reg0_acc, families), 6),
            "ndcg1": round(family_mean_value(n1_0_acc, families), 6),
            "ndcg8": round(family_mean_value(n8_0_acc, families), 6),
            "gatefail_rate": round(float(np.mean(gf0_acc)), 6),
            "utility_ctx": u0_acc,
            "regret_ctx": reg0_acc,
        },
        "t2": aux["t2"],
        "t3": aux["t3"],
        "t4": aux["t4"],
    }
    if run_4rev:
        out["t1_4rev"] = {
            "family_mean_utility": round(family_mean_value(u4_acc, families), 6),
            "ctx_mean_utility": round(float(np.mean(u4_acc)), 6),
            "family_mean_regret": round(family_mean_value(reg4_acc, families), 6),
            "ndcg1": round(family_mean_value(n1_4_acc, families), 6),
            "ndcg8": round(family_mean_value(n8_4_acc, families), 6),
            "gatefail_rate": round(float(np.mean(gf4_acc)), 6),
            "utility_ctx": u4_acc,
            "regret_ctx": reg4_acc,
        }
    return out


# =============================================================================
# Part A: Instrument & Bottleneck Diagnostics (`A1`–`A4`, `DL-Train` ONLY)
# =============================================================================
def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a.astype(np.float64), axis=-1)
    bn = np.linalg.norm(b.astype(np.float64), axis=-1)
    denom = np.maximum(an * bn, 1e-12)
    return np.sum(a.astype(np.float64) * b.astype(np.float64), axis=-1) / denom


def _eval_fused_net_on_tokens(
    net: Any,
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
    tokens_10x32: np.ndarray,
    mask_10: np.ndarray,
    *,
    reveal_mode: str,
    anchor_bank: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate `FusedJopeNetwork.adapter_10tok.predict_costs` under exact reveal modes."""
    try:
        from .anchor_fast_cache import fast_adapter_predict_single
    except ImportError:
        from anchor_fast_cache import fast_adapter_predict_single

    n_c = len(ordered_cids)
    utils = np.zeros(n_c, dtype=np.float64)
    regrets = np.zeros(n_c, dtype=np.float64)
    gatefails = np.zeros(n_c, dtype=bool)
    picks = np.zeros(n_c, dtype=np.int64)
    score_stds = np.zeros(n_c, dtype=np.float64)

    net.eval()
    with torch.no_grad():
        for i, cid in enumerate(ordered_cids):
            p = pools[cid]
            tok_t = torch.from_numpy(tokens_10x32[i : i + 1].astype(np.float32))
            msk_t = torch.from_numpy(mask_10[i : i + 1].astype(np.float32))
            if anchor_bank is not None and cid in anchor_bank:
                out = fast_adapter_predict_single(
                    net.adapter_10tok, anchor_bank[cid], tok_t, msk_t, reveal_mode
                )["total_cost"]
            else:
                m = len(p["pids"])
                if reveal_mode == "true_0rev":
                    rev_idx = torch.empty((1, 0), dtype=torch.long)
                    rev_f = torch.empty((1, 0), dtype=torch.float32)
                    rev_g = torch.empty((1, 0), dtype=torch.float32)
                elif reveal_mode == "wave0_fake_k1":
                    rev_idx = torch.zeros((1, 1), dtype=torch.long)
                    rev_f = torch.full((1, 1), 100.0, dtype=torch.float32)
                    rev_g = torch.zeros((1, 1), dtype=torch.float32)
                elif reveal_mode == "wave0_fake_k4":
                    k_r = min(4, m)
                    rev_idx = torch.arange(k_r, dtype=torch.long).reshape(1, k_r)
                    rev_f = torch.full((1, k_r), 100.0, dtype=torch.float32)
                    rev_g = torch.zeros((1, k_r), dtype=torch.float32)
                elif reveal_mode == "real_4rev":
                    k_r = min(4, m)
                    rev_idx = torch.arange(k_r, dtype=torch.long).reshape(1, k_r)
                    rev_f = torch.as_tensor(p["fitness"][:k_r][None], dtype=torch.float32)
                    rev_g = torch.as_tensor(p["gate"][:k_r][None], dtype=torch.float32)
                else:
                    raise ValueError(f"unknown reveal_mode: {reveal_mode}")

                out = net.adapter_10tok.predict_costs(
                    ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                    prog_412=torch.as_tensor(p["program"][None], dtype=torch.float32),
                    proj_16=torch.as_tensor(p["proj16"][None], dtype=torch.float32),
                    base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                    scale=torch.full((1,), 0.4208, dtype=torch.float32),
                    sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                    mm_tokens_10x32=tok_t,
                    mm_mask_10=msk_t,
                    revealed_indices=rev_idx,
                    revealed_raw_f=rev_f,
                    revealed_gate=rev_g,
                )["total_cost"].float()[0].numpy()

            j = int(np.argmin(out))
            picks[i] = j
            score_stds[i] = float(np.std(out))
            gf = not bool(p["gate"][j])
            gatefails[i] = gf
            u_j = -float(p["fitness"][j]) if not gf else -2.0
            utils[i] = u_j
            u_all = np.where(p["gate"], -p["fitness"], -2.0)
            regrets[i] = max(0.0, float(np.max(u_all)) - u_j)

    return {
        "utils": utils,
        "regrets": regrets,
        "gatefails": gatefails,
        "picks": picks,
        "score_stds": score_stds,
    }


def _family_knn_acc(x: np.ndarray, families: Sequence[str], k: int = 1) -> float:
    mu, sd = fit_fold_standardizer(x)
    z = apply_fold_standardizer(x, mu, sd).astype(np.float64)
    norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    zn = z / norms
    sims = zn @ zn.T
    np.fill_diagonal(sims, -1e9)
    fam_arr = np.asarray(list(families), dtype=object)
    # Only score contexts whose family has >= 2 members in DL-Train
    u_f, cnts = np.unique(fam_arr, return_counts=True)
    multi = {f for f, c in zip(u_f.tolist(), cnts.tolist()) if c >= 2}
    hits = []
    for i in range(len(fam_arr)):
        if fam_arr[i] not in multi:
            continue
        nn_idx = np.argsort(-sims[i])[:k]
        hits.append(1.0 if np.any(fam_arr[nn_idx] == fam_arr[i]) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def run_part_a_diagnostics(
    net: Any,
    train_cids: list[str],
    train_families: list[str],
    pools: Mapping[str, dict[str, Any]],
    pool_meta: Mapping[str, Any],
    tr_feats: Mapping[str, np.ndarray],
    sh_feats: Mapping[str, np.ndarray],
    folds_list: list[np.ndarray],
    *,
    anchor_bank: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute Part A diagnostics A1, A2, A3, A4 strictly on `DL-Train` (1,219 contexts)."""
    t0 = time.perf_counter()
    assert_memavail(15.0, "part_a_start")

    # -------------------------------------------------------------------------
    # A1: Stage-1A Conditioning Bottleneck (sketch & nolm vs hpc_high, campaign, & native_high)
    # -------------------------------------------------------------------------
    v_sk = tr_feats["vec256_sketch"]
    v_nl = tr_feats["vec256_nolm"]
    v_hpc = tr_feats["vec256_hpc"]
    v_nhi = tr_feats["vec256_native_high"]
    v_ult = tr_feats["vec256_ultra"]

    # Held-out-family Ridge R^2 predicting `v_hpc` and `v_nhi` from `v_sk` on fold 0..4 (`h452a/`)
    fold0 = folds_list[0]
    pred_hpc_from_sk = np.zeros_like(v_hpc, dtype=np.float64)
    pred_nhi_from_sk = np.zeros_like(v_nhi, dtype=np.float64)
    ones_v = np.ones_like(v_hpc, dtype=np.uint8)
    for k in range(int(np.max(fold0)) + 1):
        ho = np.where(fold0 == k)[0]
        tr = np.where(fold0 != k)[0]
        pred_hpc_from_sk[ho] = _fit_predict_ridge_multi(v_sk[tr], v_hpc[tr].astype(np.float64), ones_v[tr], v_sk[ho])
        pred_nhi_from_sk[ho] = _fit_predict_ridge_multi(v_sk[tr], v_nhi[tr].astype(np.float64), ones_v[tr], v_sk[ho])

    def _coord_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r2s = []
        for j in range(y_true.shape[1]):
            yt = y_true[:, j].astype(np.float64)
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            if ss_tot > 1e-6:
                r2s.append(1.0 - float(np.sum((yt - y_pred[:, j]) ** 2)) / ss_tot)
        return float(np.mean(r2s)) if r2s else 0.0

    # Evaluate `round-3175` FusedJopeNetwork across token sources at 0-reveal and 4-reveal
    tok_sources = {
        "campaign_tokens": (tr_feats["tokens_10x32_campaign"], tr_feats["mask_10_campaign"]),
        "hpc_high": (tr_feats["tokens_10x32_hpc"], tr_feats["mask_10_hpc"]),
        "native_high": (tr_feats["tokens_10x32_native_high"], tr_feats["mask_10_native_high"]),
        "sketch_landmark_bridge": (tr_feats["tokens_10x32_sketch"], tr_feats["mask_10_sketch"]),
        "sketch_nolm_bridge": (tr_feats["tokens_10x32_nolm"], tr_feats["mask_10_sketch"]),
    }
    a1_ranking: dict[str, Any] = {}
    ref0 = _eval_fused_net_on_tokens(
        net, train_cids, pools, *tok_sources["hpc_high"], reveal_mode="true_0rev", anchor_bank=anchor_bank
    )
    ref4 = _eval_fused_net_on_tokens(
        net, train_cids, pools, *tok_sources["hpc_high"], reveal_mode="real_4rev", anchor_bank=anchor_bank
    )

    for name, (tok_arr, msk_arr) in tok_sources.items():
        r0 = (
            ref0
            if name == "hpc_high"
            else _eval_fused_net_on_tokens(
                net, train_cids, pools, tok_arr, msk_arr, reveal_mode="true_0rev", anchor_bank=anchor_bank
            )
        )
        r4 = (
            ref4
            if name == "hpc_high"
            else _eval_fused_net_on_tokens(
                net, train_cids, pools, tok_arr, msk_arr, reveal_mode="real_4rev", anchor_bank=anchor_bank
            )
        )
        a1_ranking[name] = {
            "0rev_family_utility": round(family_mean_value(r0["utils"], train_families), 6),
            "0rev_ctx_utility": round(float(np.mean(r0["utils"])), 6),
            "0rev_gatefails": int(np.sum(r0["gatefails"])),
            "0rev_top1_agree_vs_hpc": round(float(np.mean(r0["picks"] == ref0["picks"])), 6),
            "4rev_family_utility": round(family_mean_value(r4["utils"], train_families), 6),
            "4rev_ctx_utility": round(float(np.mean(r4["utils"])), 6),
            "4rev_gatefails": int(np.sum(r4["gatefails"])),
            "4rev_top1_agree_vs_hpc": round(float(np.mean(r4["picks"] == ref4["picks"])), 6),
            "0rev_delta_vs_hpc": contrast_summary(r0["utils"], ref0["utils"], train_families),
            "4rev_delta_vs_hpc": contrast_summary(r4["utils"], ref4["utils"], train_families),
        }

    tok_camp_vs_hpc = np.max(np.abs(tr_feats["tokens_10x32_campaign"] - tr_feats["tokens_10x32_hpc"]), axis=(1, 2))
    a1_out = {
        "cosine_vec256_sketch_vs_hpc_mean": round(float(np.mean(_rowwise_cosine(v_sk, v_hpc))), 6),
        "cosine_vec256_sketch_vs_native_high_mean": round(float(np.mean(_rowwise_cosine(v_sk, v_nhi))), 6),
        "cosine_vec256_nolm_vs_hpc_mean": round(float(np.mean(_rowwise_cosine(v_nl, v_hpc))), 6),
        "cosine_vec256_hpc_vs_native_high_mean": round(float(np.mean(_rowwise_cosine(v_hpc, v_nhi))), 6),
        "cosine_dope629_sketch_vs_hpc_mean": round(
            float(np.mean(_rowwise_cosine(tr_feats["dope629_sketch"], tr_feats["dope629_hpc"]))), 6
        ),
        "cosine_dope629_hpc_vs_native_high_mean": round(
            float(np.mean(_rowwise_cosine(tr_feats["dope629_hpc"], tr_feats["dope629_native_high"]))), 6
        ),
        "cosine_dope629_native_high_vs_ultra_mean": round(
            float(np.mean(_rowwise_cosine(tr_feats["dope629_native_high"], tr_feats["dope629_ultra"]))), 6
        ),
        "oof_ridge_r2_sketch_to_hpc_vec256": round(_coord_r2(v_hpc, pred_hpc_from_sk), 6),
        "oof_ridge_r2_sketch_to_native_high_vec256": round(_coord_r2(v_nhi, pred_nhi_from_sk), 6),
        "landmark_bridge_blend_alpha": round(float(net.landmark_bridge.blend_alpha.item()), 6),
        "wp01_campaign_tokens_vs_h58_hpc": {
            "h58_vs_recomputed_hpc_max_abs_diff": float(np.max(tr_feats.get("h58_parity_max_diff", np.zeros(2)))),
            "campaign_cache_vs_hpc_median_max_diff": round(float(np.median(tok_camp_vs_hpc)), 6),
            "campaign_cache_vs_hpc_p95_max_diff": round(float(np.percentile(tok_camp_vs_hpc, 95)), 6),
            "campaign_cache_vs_hpc_max_diff": round(float(np.max(tok_camp_vs_hpc)), 6),
        },
        "fused_net_ranking_by_stage1a_mode": a1_ranking,
    }

    # -------------------------------------------------------------------------
    # A2: `L7` Audit — `lm48_z` (48D SVD projection) vs `R_lm80` (80D triple reshape)
    # -------------------------------------------------------------------------
    svd_basis = net.step2_svd_basis_256x48.detach().cpu().numpy().astype(np.float64)
    gram_err = float(np.max(np.abs(svd_basis.T @ svd_basis - np.eye(48))))
    lm48_raw = tr_feats["lm48_hpc"]
    r16x3 = lm48_raw.reshape(-1, 16, 3)
    frac_slot1_negative = float(np.mean(r16x3[:, :, 1] < 0.0))
    frac_slot2_lt_slot0 = float(np.mean(r16x3[:, :, 2] < r16x3[:, :, 0]))
    lm48_z = tr_feats["lm48_z_hpc"]
    lm80_z = tr_feats["lm80_z_hpc"]
    lm80_extra32 = lm80_z[:, 48:80]

    aux_lm48 = evaluate_t2_t3_t4_oof(
        lm48_z, pool_meta, tr_feats["gains23_ultra"], tr_feats["mask23_ultra"], folds_list, compute_t4=True
    )
    aux_lm80 = evaluate_t2_t3_t4_oof(
        lm80_z, pool_meta, tr_feats["gains23_ultra"], tr_feats["mask23_ultra"], folds_list, compute_t4=True
    )
    aux_extra32 = evaluate_t2_t3_t4_oof(
        lm80_extra32, pool_meta, tr_feats["gains23_ultra"], tr_feats["mask23_ultra"], folds_list, compute_t4=False
    )
    a2_out = {
        "verdict_l7": "SVD_PROJECTION_CONFIRMED_DISTORTED_BY_16X3_TRIPLE_RESHAPE",
        "svd_basis_orthonormality_max_err": round(gram_err, 8),
        "lm48_pseudo_std_slot_negative_frac": round(frac_slot1_negative, 6),
        "lm48_pseudo_max_lt_mean_frac": round(frac_slot2_lt_slot0, 6),
        "family_1nn_acc": {
            "lm48_z_48d": round(_family_knn_acc(lm48_z, train_families, k=1), 6),
            "R_lm80_80d": round(_family_knn_acc(lm80_z, train_families, k=1), 6),
            "lm80_extra32_only": round(_family_knn_acc(lm80_extra32, train_families, k=1), 6),
        },
        "family_5nn_acc": {
            "lm48_z_48d": round(_family_knn_acc(lm48_z, train_families, k=5), 6),
            "R_lm80_80d": round(_family_knn_acc(lm80_z, train_families, k=5), 6),
            "lm80_extra32_only": round(_family_knn_acc(lm80_extra32, train_families, k=5), 6),
        },
        "t2_head_logloss": {
            "lm48_z": aux_lm48["t2"]["head_logloss_mean"],
            "R_lm80": aux_lm80["t2"]["head_logloss_mean"],
            "lm80_extra32": aux_extra32["t2"]["head_logloss_mean"],
        },
        "t3_anchor_mean_r2": {
            "lm48_z": aux_lm48["t3"]["mean_r2"],
            "R_lm80": aux_lm80["t3"]["mean_r2"],
            "lm80_extra32": aux_extra32["t3"]["mean_r2"],
        },
        "t4_fine_landmark_mean_r2": {
            "lm48_z": aux_lm48["t4"]["mean_r2"],
            "R_lm80": aux_lm80["t4"]["mean_r2"],
        },
        "contrast_R_lm80_minus_lm48_z_t3_skill": contrast_summary(
            aux_lm80["t3"]["ctx_skill"], aux_lm48["t3"]["ctx_skill"], train_families
        ),
    }

    # -------------------------------------------------------------------------
    # A3: True Masked 0-Reveal vs Wave-0 Fake `fitness=100, gate=0` Pseudo-Reveal
    # -------------------------------------------------------------------------
    camp_tok, camp_msk = tr_feats["tokens_10x32_campaign"], tr_feats["mask_10_campaign"]
    r_true0 = _eval_fused_net_on_tokens(
        net, train_cids, pools, camp_tok, camp_msk, reveal_mode="true_0rev", anchor_bank=anchor_bank
    )
    r_w0_k1 = _eval_fused_net_on_tokens(
        net, train_cids, pools, camp_tok, camp_msk, reveal_mode="wave0_fake_k1", anchor_bank=anchor_bank
    )
    r_w0_k4 = _eval_fused_net_on_tokens(
        net, train_cids, pools, camp_tok, camp_msk, reveal_mode="wave0_fake_k4", anchor_bank=anchor_bank
    )
    r_real4 = _eval_fused_net_on_tokens(
        net, train_cids, pools, camp_tok, camp_msk, reveal_mode="real_4rev", anchor_bank=anchor_bank
    )

    a3_out = {
        "uncapped_h_in_raw_f_over_scale": round(100.0 / 0.4208, 4),
        "modes": {
            mode_name: {
                "family_mean_utility": round(family_mean_value(rm["utils"], train_families), 6),
                "ctx_mean_utility": round(float(np.mean(rm["utils"])), 6),
                "family_mean_regret": round(family_mean_value(rm["regrets"], train_families), 6),
                "gatefails": int(np.sum(rm["gatefails"])),
                "mean_pred_score_std": round(float(np.mean(rm["score_stds"])), 6),
                "top1_agree_with_true_0rev": round(float(np.mean(rm["picks"] == r_true0["picks"])), 6),
            }
            for mode_name, rm in (
                ("true_masked_0rev", r_true0),
                ("wave0_fake_k1", r_w0_k1),
                ("wave0_fake_k4", r_w0_k4),
                ("real_4rev", r_real4),
            )
        },
        "contrast_true_0rev_minus_wave0_k1": contrast_summary(r_true0["utils"], r_w0_k1["utils"], train_families),
        "contrast_true_0rev_minus_wave0_k4": contrast_summary(r_true0["utils"], r_w0_k4["utils"], train_families),
        "contrast_real_4rev_minus_true_0rev": contrast_summary(r_real4["utils"], r_true0["utils"], train_families),
    }

    # -------------------------------------------------------------------------
    # A4: H410 `n_fit` Quartiles, Split-Half Stability, & `DVP-S4` MDE
    # -------------------------------------------------------------------------
    n_fits = tr_feats["n_fit"].astype(np.int64)
    q_splits = np.array_split(np.argsort(n_fits, kind="mergesort"), 4)
    q_labels = ("Q1_smallest", "Q2_lower_mid", "Q3_upper_mid", "Q4_largest")
    quartile_stats: dict[str, Any] = {}
    for q_name, idx_q in zip(q_labels, q_splits):
        fams_q = [train_families[i] for i in idx_q.tolist()]
        quartile_stats[q_name] = {
            "n_contexts": int(len(idx_q)),
            "n_families": len(set(fams_q)),
            "n_fit_min": int(np.min(n_fits[idx_q])),
            "n_fit_median": round(float(np.median(n_fits[idx_q])), 1),
            "n_fit_max": int(np.max(n_fits[idx_q])),
            "0rev_ctx_utility": round(float(np.mean(r_true0["utils"][idx_q])), 6),
            "4rev_ctx_utility": round(float(np.mean(r_real4["utils"][idx_q])), 6),
            "4rev_ctx_regret": round(float(np.mean(r_real4["regrets"][idx_q])), 6),
            "4rev_gatefail_rate": round(float(np.mean(r_real4["gatefails"][idx_q])), 6),
        }

    # Split-half stability across the stratified DL-Train contexts
    sh_n = sh_feats["n_fit"].astype(np.int64)
    sh_splits = np.array_split(np.argsort(sh_n, kind="mergesort"), 4)
    sh_groups: list[tuple[str, np.ndarray]] = [
        ("all", np.arange(len(sh_n), dtype=np.int64)),
        ("Q1", sh_splits[0]),
        ("Q2", sh_splits[1]),
        ("Q3", sh_splits[2]),
        ("Q4", sh_splits[3]),
    ]
    splithalf_summary: dict[str, Any] = {}
    for view_key in ("vec256_sketch", "vec256_hpc", "dope629_hpc", "dope629_ultra", "R_packet_v1"):
        h1 = sh_feats[f"{view_key}_h1"]
        h2 = sh_feats[f"{view_key}_h2"]
        cos_all = _rowwise_cosine(h1, h2)
        pooled = np.vstack([h1, h2])
        mu_p, sd_p = fit_fold_standardizer(pooled)
        z1 = apply_fold_standardizer(h1, mu_p, sd_p).astype(np.float64)
        z2 = apply_fold_standardizer(h2, mu_p, sd_p).astype(np.float64)
        tot_var = 0.5 * (np.mean(z1 ** 2, axis=1) + np.mean(z2 ** 2, axis=1))
        diff_var = 0.5 * np.mean((z1 - z2) ** 2, axis=1)
        rel_all = np.clip(1.0 - diff_var / np.maximum(tot_var, 1e-6), -1.0, 1.0)
        splithalf_summary[view_key] = {
            q_label: {
                "n_tables": int(len(q_idx)),
                "n_fit_range": [int(np.min(sh_n[q_idx])), int(np.max(sh_n[q_idx]))],
                "mean_cosine": round(float(np.mean(cos_all[q_idx])), 6),
                "p10_cosine": round(float(np.percentile(cos_all[q_idx], 10)), 6),
                "mean_icc_reliability": round(float(np.mean(rel_all[q_idx])), 6),
            }
            for q_label, q_idx in sh_groups
            if len(q_idx) > 0
        }

    # DVP-S4: 80%-power Minimum Detectable Effect (`(z_0.975 + z_0.80) * SE_family = 2.801585 * SE_family`)
    # Computed both unconditionally (across family means) and for paired differences (between realistic selectors:
    # e.g. `true_0rev` vs `hpc_0rev` and within-family / across-family paired SD).
    def _family_means_vec(vals: np.ndarray) -> np.ndarray:
        b: dict[str, list[float]] = {}
        for v, f in zip(vals.tolist(), train_families):
            b.setdefault(f, []).append(v)
        return np.array([np.mean( b[f] ) for f in sorted(b.keys())], dtype=np.float64)

    n_fams = len(set(train_families))
    z_mde = 1.959963984540054 + 0.8416212335729143  # 2.8015852
    fam_u0 = _family_means_vec(r_true0["utils"])
    fam_u4 = _family_means_vec(r_real4["utils"])
    fam_lift_4v0 = fam_u4 - fam_u0
    fam_reg0 = _family_means_vec(r_true0["regrets"])

    # Paired selector residual SD estimated from non-reveal Stage-1A change (`hpc_high` vs `sketch` at 0-rev)
    r_sk0 = _eval_fused_net_on_tokens(
        net,
        train_cids,
        pools,
        tr_feats["tokens_10x32_sketch"],
        tr_feats["mask_10_sketch"],
        reveal_mode="true_0rev",
        anchor_bank=anchor_bank,
    )
    fam_diff_stage1a = _family_means_vec(ref0["utils"] - r_sk0["utils"])
    se_stage1a = float(np.std(fam_diff_stage1a, ddof=1) / math.sqrt(n_fams))
    se_4v0 = float(np.std(fam_lift_4v0, ddof=1) / math.sqrt(n_fams))
    se_reg0 = float(np.std(fam_reg0, ddof=1) / math.sqrt(n_fams))

    u_fine_ms = tr_feats["fine_5_landmarks_ms"]
    u_tot_ms = tr_feats["total_landmark_ms"]

    a4_out = {
        "n_fit_quartiles": quartile_stats,
        "splithalf_128_stability": splithalf_summary,
        "dvp_s4_power_and_mde": {
            "n_families": n_fams,
            "alpha_two_sided": 0.05,
            "target_power": 0.80,
            "z_multiplier": round(z_mde, 6),
            "paired_0rev_utility_se_stage1a_ref": round(se_stage1a, 6),
            "paired_0rev_utility_mde80_stage1a_ref": round(z_mde * se_stage1a, 6),
            "paired_4rev_vs_0rev_se": round(se_4v0, 6),
            "paired_4rev_vs_0rev_mde80": round(z_mde * se_4v0, 6),
            "unpaired_regret_se": round(se_reg0, 6),
        },
        "u_s3_native_landmark_latency_ms": {
            "worker_threads_per_table": 1,
            "fine_5_landmarks_median_ms": round(float(np.median(u_fine_ms)), 3),
            "fine_5_landmarks_p95_ms": round(float(np.percentile(u_fine_ms, 95)), 3),
            "total_23_landmarks_median_ms": round(float(np.median(u_tot_ms)), 3),
            "total_23_landmarks_p95_ms": round(float(np.percentile(u_tot_ms, 95)), 3),
            "estimated_8thread_fine_5_median_ms": round(float(np.median(u_fine_ms)) / 4.2, 3),
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    return {
        "A1_stage1a_conditioning_gap": a1_out,
        "A2_l7_lm48_vs_lm80_audit": a2_out,
        "A3_zero_reveal_vs_wave0_fake": a3_out,
        "A4_h410_stability_and_mde": a4_out,
        "_cached_incumbent_evals": {
            "true_0rev": r_true0,
            "wave0_fake_k4": r_w0_k4,
            "real_4rev": r_real4,
        },
    }


# =============================================================================
# Part B: Full CPU Value Probe Matrix (`T1`–`T4`, Contrasts `C0`–`C7`)
# =============================================================================
def run_part_b_probe_matrix(
    train_families: list[str],
    pool_meta: Mapping[str, Any],
    tr_feats: Mapping[str, np.ndarray],
    folds_list: list[np.ndarray],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    num_threads: int = 8,
    parallel_workers: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run all Part B representation arms across T1–T4 and compute `C0`–`C7` + Holm."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    assert_memavail(20.0, "part_b_start")
    n_ctx = len(train_families)
    eval_folds = folds_list[:1] if smoke else folds_list
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)

    r_desc = np.hstack([tr_feats["vec256_hpc"], tr_feats["col_lm_34_hpc"]]).astype(np.float32)
    r_dope = np.hstack([tr_feats["dope629_hpc"], tr_feats["col_lm_34_hpc"]]).astype(np.float32)
    r_desc_nhi = np.hstack([tr_feats["vec256_native_high"], tr_feats["col_lm_34_native_high"]]).astype(np.float32)
    r_dope_nhi = np.hstack([tr_feats["dope629_native_high"], tr_feats["col_lm_34_native_high"]]).astype(np.float32)
    r_desc_ult = np.hstack([tr_feats["vec256_ultra"], tr_feats["col_lm_34_ultra"]]).astype(np.float32)
    r_dope_ult = np.hstack([tr_feats["dope629_ultra"], tr_feats["col_lm_34_ultra"]]).astype(np.float32)
    r_pkt = tr_feats["R_packet_v1"].astype(np.float32)
    r_pkt_notgt = tr_feats["R_packet_notarget"].astype(np.float32)

    donor_perm = make_cross_family_donor_permutation(train_families, seed=20261118)
    r_shuffled = r_dope[donor_perm].copy()
    rng_noise = np.random.default_rng(20261119)
    r_noise = rng_noise.standard_normal(r_dope.shape).astype(np.float32)

    standard_arms: dict[str, tuple[np.ndarray, bool]] = {
        "R_none": (np.zeros((n_ctx, 0), dtype=np.float32), False),
        "R_shape": (tr_feats["R_shape"].astype(np.float32), True),
        "lm48_z": (tr_feats["lm48_z_hpc"].astype(np.float32), True),
        "R_lm80": (tr_feats["lm80_z_hpc"].astype(np.float32), True),
        "R_desc": (r_desc, True),
        "R_learned": (tr_feats["dope629_hpc"][:, 256:].astype(np.float32), True),
        "R_dope": (r_dope, True),
        "R_hidden": (tr_feats["hidden256_hpc"].astype(np.float32), True),
        "R_tok": (tr_feats["tokens_10x32_hpc"].reshape(n_ctx, 320).astype(np.float32), True),
        "R_shuffled": (r_shuffled, False),
        "R_noise": (r_noise, False),
        "R_desc_native_high": (r_desc_nhi, True),
        "R_dope_native_high": (r_dope_nhi, True),
        "R_desc_ultra": (r_desc_ult, False),
        "R_dope_ultra": (r_dope_ult, False),
        "R_dope_plus_ultra_delta": (np.hstack([r_dope, r_dope_ult - r_dope_nhi]).astype(np.float32), False),
        "R_packet_v1": (r_pkt, True),
        "R_packet_notarget": (r_pkt_notgt, True),
        "R_dope_plus_packet_v1": (np.hstack([r_dope, r_pkt]).astype(np.float32), True),
        "R_dope_ultra_plus_packet_v1": (
            np.hstack([r_dope_ult, tr_feats["R_packet_v1_ultra"].astype(np.float32)]).astype(np.float32),
            False,
        ),
    }

    labels_bundle = {
        **pool_meta,
        "families": train_families,
        "gains23_ultra": tr_feats["gains23_ultra"],
        "mask23_ultra": tr_feats["mask23_ultra"],
    }

    eff_workers = max(1, min(3, int(parallel_workers)))
    lgbm_threads = max(1, int(num_threads))

    def _score_arm_task(item: tuple[str, tuple[np.ndarray, bool]]) -> tuple[str, dict[str, Any]]:
        arm_name, (r_mat, want_t4) = item
        assert_memavail(20.0, f"part_b:{arm_name}")
        t_arm = time.perf_counter()
        res_a = dvp_score(
            r_mat,
            labels_bundle,
            eval_folds,
            seeds=eval_seeds,
            num_threads=lgbm_threads,
            run_4rev=not smoke,
            compute_t4=want_t4,
            smoke=smoke,
        )
        print(
            f"[PartB:{arm_name}] width={r_mat.shape[1]} "
            f"T1_0rev={res_a['t1_0rev']['family_mean_utility']:.5f} "
            f"T2_head_ll={res_a['t2']['head_logloss_mean']:.4f} "
            f"T3_medR2={res_a['t3']['median_r2']:.4f} "
            f"({time.perf_counter() - t_arm:.1f}s)",
            flush=True,
        )
        return arm_name, res_a

    arm_results: dict[str, dict[str, Any]] = {}
    arm_items = list(standard_arms.items())
    if eff_workers > 1:
        with ThreadPoolExecutor(max_workers=eff_workers) as pool_ex:
            for arm_name, res_a in pool_ex.map(_score_arm_task, arm_items):
                arm_results[arm_name] = res_a
    else:
        for item in arm_items:
            arm_name, res_a = _score_arm_task(item)
            arm_results[arm_name] = res_a

    # 5-seed `R_randrouter` (evaluate all 5 random-router initializations and average per context!)
    t_rr = time.perf_counter()
    rr_seeds_n = 1 if smoke else tr_feats["dope629_randrouter_5"].shape[0]

    def _score_rr_seed(r_i: int) -> dict[str, Any]:
        assert_memavail(20.0, f"part_b:rr{r_i}")
        r_rr_mat = np.hstack([tr_feats["dope629_randrouter_5"][r_i], tr_feats["col_lm_34_hpc"]]).astype(np.float32)
        return dvp_score(
            r_rr_mat,
            labels_bundle,
            eval_folds,
            seeds=eval_seeds,
            num_threads=lgbm_threads,
            run_4rev=not smoke,
            compute_t4=False,
            smoke=smoke,
        )

    if eff_workers > 1 and rr_seeds_n > 1:
        with ThreadPoolExecutor(max_workers=eff_workers) as pool_ex:
            rr_sub = list(pool_ex.map(_score_rr_seed, range(rr_seeds_n)))
    else:
        rr_sub = [_score_rr_seed(r_i) for r_i in range(rr_seeds_n)]
    rr_u0 = np.mean([s["t1_0rev"]["utility_ctx"] for s in rr_sub], axis=0)
    rr_reg0 = np.mean([s["t1_0rev"]["regret_ctx"] for s in rr_sub], axis=0)
    rr_hll = np.mean([s["t2"]["head_logloss_ctx"] for s in rr_sub], axis=0)
    rr_nll = np.mean([s["t2"]["nonlin_logloss_ctx"] for s in rr_sub], axis=0)
    rr_t3s = np.mean([s["t3"]["ctx_skill"] for s in rr_sub], axis=0)
    rr_entry: dict[str, Any] = {
        "config_sha256": DVP_CONFIG_SHA256,
        "width": 663,
        "n_router_seeds": rr_seeds_n,
        "per_router_seed_0rev_family_utility": [s["t1_0rev"]["family_mean_utility"] for s in rr_sub],
        "t1_0rev": {
            "family_mean_utility": round(family_mean_value(rr_u0, train_families), 6),
            "ctx_mean_utility": round(float(np.mean(rr_u0)), 6),
            "family_mean_regret": round(family_mean_value(rr_reg0, train_families), 6),
            "ndcg1": round(float(np.mean([s["t1_0rev"]["ndcg1"] for s in rr_sub])), 6),
            "ndcg8": round(float(np.mean([s["t1_0rev"]["ndcg8"] for s in rr_sub])), 6),
            "gatefail_rate": round(float(np.mean([s["t1_0rev"]["gatefail_rate"] for s in rr_sub])), 6),
            "utility_ctx": rr_u0,
            "regret_ctx": rr_reg0,
        },
        "t2": {
            "head_logloss_ctx": rr_hll,
            "head_neg_logloss_ctx": -rr_hll,
            "head_logloss_mean": round(float(np.mean(rr_hll)), 6),
            "head_macro_f1": round(float(np.mean([s["t2"]["head_macro_f1"] for s in rr_sub])), 6),
            "nonlin_logloss_ctx": rr_nll,
            "nonlin_neg_logloss_ctx": -rr_nll,
            "nonlin_logloss_mean": round(float(np.mean(rr_nll)), 6),
            "nonlin_macro_f1": round(float(np.mean([s["t2"]["nonlin_macro_f1"] for s in rr_sub])), 6),
        },
        "t3": {
            "n_anchors_scored": rr_sub[0]["t3"]["n_anchors_scored"],
            "mean_r2": round(float(np.mean([s["t3"]["mean_r2"] for s in rr_sub])), 6),
            "median_r2": round(float(np.mean([s["t3"]["median_r2"] for s in rr_sub])), 6),
            "pos_r2_frac": round(float(np.mean([s["t3"]["pos_r2_frac"] for s in rr_sub])), 6),
            "ctx_skill": rr_t3s,
        },
        "t4": {},
    }
    if not smoke:
        rr_u4 = np.mean([s["t1_4rev"]["utility_ctx"] for s in rr_sub], axis=0)
        rr_reg4 = np.mean([s["t1_4rev"]["regret_ctx"] for s in rr_sub], axis=0)
        rr_entry["t1_4rev"] = {
            "family_mean_utility": round(family_mean_value(rr_u4, train_families), 6),
            "ctx_mean_utility": round(float(np.mean(rr_u4)), 6),
            "family_mean_regret": round(family_mean_value(rr_reg4, train_families), 6),
            "ndcg1": round(float(np.mean([s["t1_4rev"]["ndcg1"] for s in rr_sub])), 6),
            "ndcg8": round(float(np.mean([s["t1_4rev"]["ndcg8"] for s in rr_sub])), 6),
            "gatefail_rate": round(float(np.mean([s["t1_4rev"]["gatefail_rate"] for s in rr_sub])), 6),
            "utility_ctx": rr_u4,
            "regret_ctx": rr_reg4,
        }
    arm_results["R_randrouter"] = rr_entry
    print(
        f"[PartB:R_randrouter] seeds={rr_seeds_n} "
        f"T1_0rev={rr_entry['t1_0rev']['family_mean_utility']:.5f} ({time.perf_counter() - t_rr:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Ceilings (`G0` / `C0` / `WP4` `O-S2`):
    #   1) `profile_oracle` (`A -> B` split-pool ceiling, paired vs `R_none_halfB` & `R_dope_halfB`)
    #   2) `R_oracle64` & `R_oracle16` (TRAIN-fold pivoted-QR common basket, basket-excluded slates!)
    # -------------------------------------------------------------------------
    pool_entries = pool_meta["pool_entries"]
    half_b_masks = [pe["is_half_b"] for pe in pool_entries]
    n_runs = len(eval_folds) * len(eval_seeds)
    n_folds = int(np.max(eval_folds[0])) + 1

    def _run_masked_t1(r_per_fold_fn, masks_per_fold_fn) -> dict[str, Any]:
        u_acc = np.zeros(n_ctx, dtype=np.float64)
        reg_acc = np.zeros(n_ctx, dtype=np.float64)
        n1_acc = np.zeros(n_ctx, dtype=np.float64)
        n8_acc = np.zeros(n_ctx, dtype=np.float64)
        gf_acc = np.zeros(n_ctx, dtype=np.float64)
        for p_i, fold_arr in enumerate(eval_folds):
            for k in range(1 if smoke else n_folds):
                ho_idx = np.where(fold_arr == k)[0]
                tr_idx = np.where(fold_arr != k)[0]
                r_fold = r_per_fold_fn(p_i, k, tr_idx, ho_idx)
                c_masks = masks_per_fold_fn(p_i, k, tr_idx, ho_idx)
                # Build fold program priors strictly on `tr_idx` AND respecting `c_masks`!
                masked_entries = [
                    {
                        **pe,
                        "pids": [pe["pids"][j] for j in np.where(c_masks[i_c])[0].tolist()],
                        "z_adv": pe["z_adv"][c_masks[i_c]],
                        "gate": pe["gate"][c_masks[i_c]],
                    }
                    for i_c, pe in enumerate(pool_entries)
                ]
                p_priors = _build_fold_prog_priors(masked_entries, tr_idx)
                denom = (1.0 if smoke else float(n_runs))
                for sd in eval_seeds:
                    res = evaluate_t1_fold_cell(
                        r_fold,
                        pool_entries,
                        tr_idx,
                        ho_idx,
                        seed=int(sd),
                        use_4rev=False,
                        cand_masks=c_masks,
                        prog_priors=p_priors,
                        num_threads=num_threads,
                    )
                    u_acc[ho_idx] += res["utility"] / denom
                    reg_acc[ho_idx] += res["regret"] / denom
                    n1_acc[ho_idx] += res["ndcg1"] / denom
                    n8_acc[ho_idx] += res["ndcg8"] / denom
                    gf_acc[ho_idx] += res["gatefail"] / denom
        active = ~np.isnan(u_acc)
        if smoke:
            # In smoke mode only fold 0 was filled
            mask_ho = eval_folds[0] == 0
            fams_use = [train_families[i] for i in np.where(mask_ho)[0].tolist()]
            u_use = u_acc[mask_ho]
            r_use = reg_acc[mask_ho]
        else:
            fams_use = train_families
            u_use = u_acc
            r_use = reg_acc
        return {
            "family_mean_utility": round(family_mean_value(u_use, fams_use), 6),
            "ctx_mean_utility": round(float(np.mean(u_use)), 6),
            "family_mean_regret": round(family_mean_value(r_use, fams_use), 6),
            "ndcg1": round(float(np.mean(n1_acc[eval_folds[0] == 0] if smoke else n1_acc)), 6),
            "ndcg8": round(float(np.mean(n8_acc[eval_folds[0] == 0] if smoke else n8_acc)), 6),
            "gatefail_rate": round(float(np.mean(gf_acc[eval_folds[0] == 0] if smoke else gf_acc)), 6),
            "utility_ctx": u_acc,
            "regret_ctx": reg_acc,
        }

    # Pivoted-QR basket `R_oracle64` & `R_oracle16` (strictly TRAIN-fold selected, basket-excluded slates!)
    common_pids = pool_meta["common_pids"]
    common_adv = pool_meta["common_std_adv"]
    common_pres = pool_meta["common_present"]

    basket_cache_64: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray], set[int]]] = {}
    basket_cache_16: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray], set[int]]] = {}

    for p_i, fold_arr in enumerate(eval_folds):
        for k in range(1 if smoke else n_folds):
            tr_idx = np.where(fold_arr != k)[0]
            sup_tr = np.sum(common_pres[tr_idx], axis=0)
            elig_cols = np.where(sup_tr >= 80)[0]
            sel64_local = select_pivoted_qr_basket(common_adv[tr_idx][:, elig_cols], sup_tr[elig_cols], 64)
            cols64 = elig_cols[sel64_local]
            cols16 = cols64[:16]

            pids64 = {common_pids[c] for c in cols64.tolist()}
            r_or64 = np.hstack([common_adv[:, cols64], common_pres[:, cols64].astype(np.float32)]).astype(np.float32)
            r_or16 = np.hstack([common_adv[:, cols16], common_pres[:, cols16].astype(np.float32)]).astype(np.float32)
            # Exclude ALL 64 basket programs from slates for BOTH R_oracle64, R_oracle16, R_dope_no_basket64, & R_none_no_basket64
            masks_nb64 = []
            for pe in pool_entries:
                m_nb = np.array([pid not in pids64 for pid in pe["pids"]], dtype=bool)
                if not np.any(m_nb):
                    m_nb = np.ones(len(pe["pids"]), dtype=bool)
                masks_nb64.append(m_nb)
            basket_cache_64[(p_i, k)] = (r_or64, masks_nb64, set(cols64.tolist()))
            basket_cache_16[(p_i, k)] = (r_or16, masks_nb64, set(cols64.tolist()))

    ceiling_specs = [
        ("prof_oracle", lambda *_: pool_meta["profile_a_38"], lambda *_: half_b_masks),
        ("r_none_halfb", lambda *_: np.zeros((n_ctx, 0), dtype=np.float32), lambda *_: half_b_masks),
        ("r_dope_halfb", lambda *_: r_dope, lambda *_: half_b_masks),
        ("r_oracle64", lambda p_i, k, *_: basket_cache_64[(p_i, k)][0], lambda p_i, k, *_: basket_cache_64[(p_i, k)][1]),
        ("r_oracle16", lambda p_i, k, *_: basket_cache_16[(p_i, k)][0], lambda p_i, k, *_: basket_cache_16[(p_i, k)][1]),
        ("r_none_nb64", lambda *_: np.zeros((n_ctx, 0), dtype=np.float32), lambda p_i, k, *_: basket_cache_64[(p_i, k)][1]),
        ("r_dope_nb64", lambda *_: r_dope, lambda p_i, k, *_: basket_cache_64[(p_i, k)][1]),
    ]
    if eff_workers > 1:
        with ThreadPoolExecutor(max_workers=eff_workers) as pool_ex:
            ceil_map = dict(
                pool_ex.map(lambda sp: (sp[0], _run_masked_t1(sp[1], sp[2])), ceiling_specs)
            )
    else:
        ceil_map = {sp[0]: _run_masked_t1(sp[1], sp[2]) for sp in ceiling_specs}

    prof_oracle_t1 = ceil_map["prof_oracle"]
    r_none_halfb_t1 = ceil_map["r_none_halfb"]
    r_dope_halfb_t1 = ceil_map["r_dope_halfb"]
    r_oracle64_t1 = ceil_map["r_oracle64"]
    r_oracle16_t1 = ceil_map["r_oracle16"]
    r_none_nb64_t1 = ceil_map["r_none_nb64"]
    r_dope_nb64_t1 = ceil_map["r_dope_nb64"]

    ceilings_out = {
        "split_pool_halfB": {
            "profile_oracle": {k: v for k, v in prof_oracle_t1.items() if not k.endswith("_ctx")},
            "R_dope_halfB": {k: v for k, v in r_dope_halfb_t1.items() if not k.endswith("_ctx")},
            "R_none_halfB": {k: v for k, v in r_none_halfb_t1.items() if not k.endswith("_ctx")},
        },
        "basket_excluded_64": {
            "R_oracle64": {k: v for k, v in r_oracle64_t1.items() if not k.endswith("_ctx")},
            "R_oracle16": {k: v for k, v in r_oracle16_t1.items() if not k.endswith("_ctx")},
            "R_dope_no_basket64": {k: v for k, v in r_dope_nb64_t1.items() if not k.endswith("_ctx")},
            "R_none_no_basket64": {k: v for k, v in r_none_nb64_t1.items() if not k.endswith("_ctx")},
        },
    }

    # -------------------------------------------------------------------------
    # Compute Contrasts `C0`–`C7` + Step-Down Holm-Bonferroni
    # -------------------------------------------------------------------------
    if smoke:
        smoke_mask = eval_folds[0] == 0
        fams_c = [train_families[i] for i in np.where(smoke_mask)[0].tolist()]
        sl = lambda arr: np.asarray(arr)[smoke_mask]
    else:
        fams_c = train_families
        sl = lambda arr: np.asarray(arr)

    def _arm_contrast(left_arm: str, right_arm: str) -> dict[str, Any]:
        la = arm_results[left_arm]
        ra = arm_results[right_arm]
        out_c = {
            "left": left_arm,
            "right": right_arm,
            "t1_0rev_utility": contrast_summary(
                sl(la["t1_0rev"]["utility_ctx"]), sl(ra["t1_0rev"]["utility_ctx"]), fams_c
            ),
            "t2_head_neg_logloss": contrast_summary(
                sl(la["t2"]["head_neg_logloss_ctx"]), sl(ra["t2"]["head_neg_logloss_ctx"]), fams_c
            ),
            "t3_anchor_skill": contrast_summary(
                sl(la["t3"]["ctx_skill"]), sl(ra["t3"]["ctx_skill"]), fams_c
            ),
        }
        if not smoke and "t1_4rev" in la and "t1_4rev" in ra:
            out_c["t1_4rev_utility"] = contrast_summary(
                sl(la["t1_4rev"]["utility_ctx"]), sl(ra["t1_4rev"]["utility_ctx"]), fams_c
            )
        return out_c

    contrasts: dict[str, Any] = {
        "C0a_profile_oracle_minus_R_none_halfB": {
            "left": "profile_oracle",
            "right": "R_none_halfB",
            "t1_0rev_utility": contrast_summary(
                sl(prof_oracle_t1["utility_ctx"]), sl(r_none_halfb_t1["utility_ctx"]), fams_c
            ),
        },
        "C0b_profile_oracle_minus_R_dope_halfB": {
            "left": "profile_oracle",
            "right": "R_dope_halfB",
            "t1_0rev_utility": contrast_summary(
                sl(prof_oracle_t1["utility_ctx"]), sl(r_dope_halfb_t1["utility_ctx"]), fams_c
            ),
        },
        "C0c_R_oracle64_minus_R_none_nb64": {
            "left": "R_oracle64",
            "right": "R_none_no_basket64",
            "t1_0rev_utility": contrast_summary(
                sl(r_oracle64_t1["utility_ctx"]), sl(r_none_nb64_t1["utility_ctx"]), fams_c
            ),
        },
        "C0d_R_oracle64_minus_R_dope_nb64": {
            "left": "R_oracle64",
            "right": "R_dope_no_basket64",
            "t1_0rev_utility": contrast_summary(
                sl(r_oracle64_t1["utility_ctx"]), sl(r_dope_nb64_t1["utility_ctx"]), fams_c
            ),
        },
        "C0e_R_oracle16_minus_R_dope_nb64": {
            "left": "R_oracle16",
            "right": "R_dope_no_basket64",
            "t1_0rev_utility": contrast_summary(
                sl(r_oracle16_t1["utility_ctx"]), sl(r_dope_nb64_t1["utility_ctx"]), fams_c
            ),
        },
        "C1_R_dope_minus_R_desc": _arm_contrast("R_dope", "R_desc"),
        "C1b_R_learned_minus_R_desc": _arm_contrast("R_learned", "R_desc"),
        "C2_R_dope_minus_R_randrouter": _arm_contrast("R_dope", "R_randrouter"),
        "C3_R_desc_minus_R_shape": _arm_contrast("R_desc", "R_shape"),
        "C3b_R_dope_minus_R_none": _arm_contrast("R_dope", "R_none"),
        "C4a_R_shuffled_minus_R_none": _arm_contrast("R_shuffled", "R_none"),
        "C4b_R_noise_minus_R_none": _arm_contrast("R_noise", "R_none"),
        "C5_R_dope_minus_R_tok": _arm_contrast("R_dope", "R_tok"),
        "C5b_R_lm80_minus_lm48_z": _arm_contrast("R_lm80", "lm48_z"),
        "C6a_R_dope_ultra_minus_R_dope": _arm_contrast("R_dope_ultra", "R_dope"),
        "C6b_R_dope_ultra_minus_R_dope_native_high": _arm_contrast("R_dope_ultra", "R_dope_native_high"),
        "C6c_R_desc_ultra_minus_R_desc_native_high": _arm_contrast("R_desc_ultra", "R_desc_native_high"),
        "C6d_R_dope_plus_ultra_delta_minus_R_dope": _arm_contrast("R_dope_plus_ultra_delta", "R_dope"),
        "C7a_R_packet_v1_minus_R_dope": _arm_contrast("R_packet_v1", "R_dope"),
        "C7b_R_dope_plus_packet_v1_minus_R_dope": _arm_contrast("R_dope_plus_packet_v1", "R_dope"),
        "C7c_R_packet_v1_minus_R_packet_notarget": _arm_contrast("R_packet_v1", "R_packet_notarget"),
        "C7d_R_dope_ultra_plus_packet_v1_minus_R_dope": _arm_contrast(
            "R_dope_ultra_plus_packet_v1", "R_dope"
        ),
    }

    primary_holm_keys = (
        "C0b_profile_oracle_minus_R_dope_halfB",
        "C0d_R_oracle64_minus_R_dope_nb64",
        "C1_R_dope_minus_R_desc",
        "C2_R_dope_minus_R_randrouter",
        "C3_R_desc_minus_R_shape",
        "C5_R_dope_minus_R_tok",
        "C6a_R_dope_ultra_minus_R_dope",
        "C6b_R_dope_ultra_minus_R_dope_native_high",
        "C7b_R_dope_plus_packet_v1_minus_R_dope",
        "C7c_R_packet_v1_minus_R_packet_notarget",
    )
    raw_p = {k: float(contrasts[k]["t1_0rev_utility"]["p_value"]) for k in primary_holm_keys}
    adj_p = holm_bonferroni(raw_p)
    for k, ap_val in adj_p.items():
        contrasts[k]["t1_0rev_utility"]["p_holm"] = round(ap_val, 6)

    # Strip bulk per-context numpy vectors from JSON-serializable summary while keeping clean dict
    serializable_arms: dict[str, Any] = {}
    raw_ctx_vectors: dict[str, Any] = {}
    for arm_name, a_dict in arm_results.items():
        clean_a = {k: v for k, v in a_dict.items() if k not in ("t1_0rev", "t1_4rev", "t2", "t3")}
        clean_a["t1_0rev"] = {k: v for k, v in a_dict["t1_0rev"].items() if not k.endswith("_ctx")}
        if "t1_4rev" in a_dict:
            clean_a["t1_4rev"] = {k: v for k, v in a_dict["t1_4rev"].items() if not k.endswith("_ctx")}
        clean_a["t2"] = {k: v for k, v in a_dict["t2"].items() if not k.endswith("_ctx")}
        clean_a["t3"] = {k: v for k, v in a_dict["t3"].items() if k != "ctx_skill"}
        serializable_arms[arm_name] = clean_a
        raw_ctx_vectors[arm_name] = a_dict["t1_0rev"]["utility_ctx"]

    return {
        "dvp_config": DVP_FROZEN_CONFIG,
        "dvp_config_sha256": DVP_CONFIG_SHA256,
        "arms": serializable_arms,
        "ceilings": ceilings_out,
        "contrasts": contrasts,
        "holm_adjusted_t1_0rev_p": {k: round(v, 6) for k, v in adj_p.items()},
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "_raw_t1_0rev_ctx": raw_ctx_vectors,
    }
