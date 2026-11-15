#!/usr/bin/env python3
"""H452 Unified Pool-Agnostic Guard + Multi-Loss BCAR + Calibrated NNLS + K-Fold Matérn-3/2 L2 Stacker
(`stack_guard_cv7_h452.py`).

Strictly SEARCH-only (`0` `y_audit` bytes read anywhere; query invariance `max|Δpred(perm(Xa))| == 0`!).
Fixes the three structural bottlenecks discovered in `guard_matern_core.guard_and_stack_2l` & `stack_h450.py`:
  1. Bidirectional Pool-Agnostic Precision & Peer-Consensus Guard:
     - Runs `GM.guard_candidate_pool_ya_free` jointly across ALL candidate pools (`Ultra-GP`, `Legacy-GP`, `M7`,
       `K3`, `F4`, `Tab450`, `104M-e8`, `104M-GP-Resid`).
     - Computes `min_oof_global` and `min_med_global` across the ENTIRE guarded union rather than anchoring solely
       on legacy `Q_gp`. Prunes high-variance columns symmetrically from BOTH `Q_gp` and `extra_blocks` when a
       strong `104M-e8`, `Ultra-GP`, `Exact-Key`, or `Formula` column leads by a wide OOF margin, and restricts
       `br_mse` random-column subsampling to columns within `1.40 * min_oof_global` so random column drops never
       force weight onto `2x-5x` noisier columns.
  2. CV-7 / Neural-Aware Adaptive Refit Deployment (`alpha_j in [0.0, 0.78]`):
     - Smoothly scales full-`X_search` refit weight up to `0.78` when `||ref_j - bag_j|| / s_oof_j` is small
       (unlocking `100%` context density for `104M` ICL and `ARD-Matérn` GP), while demoting unstable refits
       (`drift > drift_gate`) cleanly to the fold-bag `bag_j`.
  3. Nested K-Fold (`K=5` or `K=7`) Meta-Combiner Ratchet + Generalized K-Fold Matérn-3/2 `eigh` L2 Router:
     - Evaluates `(a) Pool-Agnostic Multi-Loss 32-Bag BCAR`, `(b) Sharp Top-K Fold-Log Simplex`, and
       `(c) Ridge-Shrunk Affine-Calibrated NNLS` inside a strict nested K-fold OOF tournament on `(Xs, ys)`,
       ratifying departures from `BCAR` ONLY when nested OOF fold-log RMSE improves across `>= ceil(0.6 * K)` folds.
     - Generalizes `GM.fit_l2_residual_gate` to arbitrary `K` folds (`K=5` or `K=7`) and selects the top model-family
       disagreement pairs pool-agnostically rather than hardcoding legacy `Q_gp`.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls

for _cand in (
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

try:
    from . import guard_matern_core as GM  # type: ignore
except ImportError:
    import guard_matern_core as GM  # noqa: E402


def _stream_seed(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8") + b"\x00")
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def make_stratified_kfold(cid: str, y: np.ndarray, k: int = 7, rep: int = 0) -> np.ndarray:
    """Deterministic target-quantile stratified K-fold partition (`fold sizes differ by <= 1`, `0` audit rows)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    k = max(2, min(int(k), n // 2))
    rng = np.random.default_rng(_stream_seed("h452_kf", cid, k, rep))
    sy = max(float(np.std(y)), 1e-12)
    y_jit = y + (1e-13 * sy) * rng.standard_normal(n) if rep > 0 else y
    order = np.argsort(y_jit, kind="mergesort")
    lab = np.empty(n, dtype=np.int64)
    for s in range(0, n, k):
        blk = order[s : s + k]
        if len(blk) == k:
            lab[blk] = rng.permutation(k)
        else:
            lab[blk] = rng.choice(k, size=len(blk), replace=False)
    return lab


def _project_simplex(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def _fold_masks(lab: np.ndarray) -> list[np.ndarray]:
    uniq = sorted(int(x) for x in np.unique(lab))
    masks = [lab == k for k in uniq if int(np.sum(lab == k)) >= 2]
    return masks if masks else [np.ones(len(lab), dtype=bool)]


def _fold_log_loss(pred: np.ndarray, y: np.ndarray, masks: list[np.ndarray], eps: float) -> float:
    err = pred - y
    acc = 0.0
    for m in masks:
        mse_f = float(np.mean(err[m] * err[m])) + eps
        acc += 0.5 * math.log(mse_f)
    return acc / float(len(masks))


def bcar_weights_pool_agnostic(
    Q: np.ndarray,
    y: np.ndarray,
    cid: str,
    mode: str,
    lab: np.ndarray,
    n_bags: int = 32,
) -> np.ndarray:
    """Pool-agnostic 32-bag Caruana with leader-protected `br_mse` subsampling."""
    n, m = Q.shape
    if m == 1:
        return np.ones(1, dtype=np.float64)
    R_full = Q - y[:, None]
    sy = max(float(np.std(y)), 1e-12)
    col_mse_full = np.einsum("ij,ij->j", R_full, R_full) / float(max(n, 1))
    s_col = np.sqrt(np.maximum(col_mse_full, 0.0))
    min_s = float(np.min(s_col))

    # Continuous precision-dilution guard:
    # Keep full diversity on normal noisy tables (`min_s >= 0.03 * sy`), while preventing `br_mse` column-drop bags
    # from excluding leaders and forcing weight onto >1.25x noisier columns!
    if min_s < 0.03 * sy and m > 1:
        cap_mult = 2.6
    elif min_s < 0.16 * sy and m > 1:
        cap_mult = 1.48
    else:
        cap_mult = None

    if mode == "br_mse":
        cap_mult = 1.25 if cap_mult is None else min(cap_mult, 1.25)

    if cap_mult is not None and m > 1:
        active = np.flatnonzero(s_col <= cap_mult * max(min_s, 1e-12 * sy))
        if len(active) == 0:
            active = np.array([int(np.argmin(s_col))], dtype=np.int64)
        if len(active) == 1:
            w_one = np.zeros(m, dtype=np.float64)
            w_one[int(active[0])] = 1.0
            return w_one
    else:
        active = np.arange(m, dtype=np.int64)

    R = R_full[:, active]
    ma = len(active)
    min_col_mse = float(np.min(col_mse_full[active]))
    var_y = max(float(np.var(y)), 1e-30)
    eps = max(1e-30, min(1e-12 * var_y, 1e-4 * max(min_col_mse, 1e-30)))

    row_w = None
    if mode == "huber":
        col_mse = col_mse_full[active]
        top8 = np.argsort(col_mse)[: min(8, ma)]
        r_med = np.median(R[:, top8], axis=1)
        scale = max(1.4826 * float(np.median(np.abs(r_med))), 1e-12 * sy)
        delta = 2.5 * scale
        abs_r = np.abs(r_med)
        row_w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-30))

    # Re-index `lab` to contiguous `0..K_eff-1` so `GM._kcar_fold_logrmse_fast` supports any K (e.g. K=5, 7, 14)
    _, lab_contig = np.unique(lab, return_inverse=True)

    acc = np.zeros(ma, dtype=np.float64)
    for b_idx in range(n_bags):
        rng = np.random.default_rng(GM._stream_seed(GM.BAG_SEED, "BCAR", cid, b_idx))
        rows = rng.integers(0, n, size=n)
        if mode == "mse":
            acc += GM._kcar_fast(R[rows], 100) / 100.0
        elif mode == "huber":
            acc += GM._kcar_fast(R[rows] * np.sqrt(row_w[rows])[:, None], 100) / 100.0
        elif mode == "fold_log":
            lb = lab_contig[rows]
            if len(np.unique(lb)) == int(lab_contig.max()) + 1:
                acc += GM._kcar_fold_logrmse_fast(R[rows], lb, 100, eps) / 100.0
            else:
                acc += GM._kcar_fast(R[rows], 100) / 100.0
        elif mode == "br_mse":
            k_sub = max(1, int(math.ceil(ma * 0.65)))
            cols = np.sort(rng.choice(ma, size=k_sub, replace=False)) if ma >= 2 else np.arange(ma)
            acc[cols] += GM._kcar_fast(R[np.ix_(rows, cols)], 100) / 100.0

    w_act = GM._prune(acc / float(n_bags))
    w_out = np.zeros(m, dtype=np.float64)
    w_out[active] = w_act
    return w_out


def _fit_sharp_foldlog_simplex(Q: np.ndarray, y: np.ndarray, lab: np.ndarray, top_k: int = 8) -> np.ndarray:
    """Direct fold-log-RMSE simplex over the top-K OOF columns (zero forced dilution when #1 leads clearly)."""
    n, m = Q.shape
    if m == 1:
        return np.ones(1, dtype=np.float64)
    R_full = Q - y[:, None]
    mses = np.mean(R_full * R_full, axis=0)
    idx_top = np.argsort(mses)[: min(top_k, m)]
    k = len(idx_top)
    if k == 1:
        w = np.zeros(m, dtype=np.float64)
        w[int(idx_top[0])] = 1.0
        return w

    vy = max(float(np.var(y)), 1e-24)
    eps = max(1e-30, min(1e-12 * vy, 1e-4 * max(float(mses[idx_top[0]]), 1e-30)))
    u_top = -0.5 * np.log(np.maximum(mses[idx_top], eps) / vy)
    u_lead = float(u_top[0] - u_top[1])
    l2_pen = 0.0 if u_lead >= 0.16 else (0.006 * max(0.0, 0.16 - u_lead) / 0.16)

    R = R_full[:, idx_top]
    masks = _fold_masks(lab)
    nf = float(len(masks))

    def obj_grad(w_sub: np.ndarray):
        val = float(l2_pen * np.dot(w_sub, w_sub))
        grad = (2.0 * l2_pen) * w_sub
        for mf in masks:
            Rf = R[mf]
            err = Rf @ w_sub
            mse_f = float(np.mean(err * err)) + eps
            val += (0.5 * math.log(mse_f / vy)) / nf
            grad += ((Rf.T @ err) / (float(len(err)) * mse_f)) / nf
        return val, grad

    w_win = np.zeros(k, dtype=np.float64)
    w_win[0] = 1.0
    best_w, (best_v, _) = w_win, obj_grad(w_win)

    logits = (u_top - float(u_top[0])) / max(0.05, float(np.std(u_top)) * 0.5)
    w_soft = np.exp(np.clip(logits, -30.0, 0.0))
    w_soft /= float(np.sum(w_soft))

    for w0 in (w_soft, 0.80 * w_win + 0.20 * w_soft):
        res = minimize(
            obj_grad,
            w0,
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints={"type": "eq", "fun": lambda v: float(np.sum(v) - 1.0), "jac": lambda v: np.ones(k)},
            options={"maxiter": 120, "ftol": 1e-10},
        )
        wc = _project_simplex(np.asarray(res.x, dtype=np.float64))
        vc, _ = obj_grad(wc)
        if vc < best_v:
            best_v, best_w = vc, wc

    w_out = np.zeros(m, dtype=np.float64)
    w_out[idx_top] = GM._prune(best_w)
    return w_out


def _fit_calibrated_nnls(
    Q_tr: np.ndarray,
    y_tr: np.ndarray,
    P_ev: np.ndarray,
    w_prior: np.ndarray,
    ridge_scale: float = 4.0,
) -> np.ndarray:
    """Ridge-shrunk NNLS with affine intercept/scale calibration around `w_prior` (`tabfm_ensemble`-style)."""
    n, m = Q_tr.shape
    ym = float(np.mean(y_tr))
    sy = max(float(np.std(y_tr)), 1e-12)
    mses = np.mean((Q_tr - y_tr[:, None]) ** 2, axis=0)
    keep = np.unique(np.concatenate([np.flatnonzero(w_prior > 0.01), np.argsort(mses)[: min(8, m)]]))
    if len(keep) == 0:
        keep = np.array([0], dtype=np.int64)

    Qk = (Q_tr[:, keep] - ym) / sy
    Pk = (P_ev[:, keep] - ym) / sy
    yk = (y_tr - ym) / sy
    wp = w_prior[keep]
    if float(np.sum(wp)) > 1e-9:
        wp = wp / float(np.sum(wp))
    else:
        wp = np.zeros(len(keep), dtype=np.float64)
        wp[0] = 1.0

    # Solve non-negative ridge toward `wp`: min_{beta >= 0} ||Qk beta - yk||^2 + lam ||beta - wp||^2
    lam = ridge_scale * max(1.0, float(len(keep)))
    s_lam = math.sqrt(lam)
    A = np.vstack([Qk, s_lam * np.eye(len(keep))])
    b = np.concatenate([yk, s_lam * wp])
    beta, _ = nnls(A, b)
    s_beta = float(np.sum(beta))
    if not math.isfinite(s_beta) or s_beta <= 0.2:
        beta = wp
    else:
        # Clamp total scale calibration tightly to `[0.92, 1.08]` so audit variance cannot explode
        if s_beta < 0.92 or s_beta > 1.08:
            beta = beta * (float(np.clip(s_beta, 0.92, 1.08)) / s_beta)

    oof_centered = Qk @ beta
    intercept = float(np.mean(yk - oof_centered))
    intercept = float(np.clip(intercept, -0.15, 0.15))
    return ym + sy * (Pk @ beta + intercept)


def _ugp_effective_trials(tag: str) -> int:
    """Returns effective evolutionary candidate count `M_trials` for textbook `sqrt(1 + 2 ln(M) / n_s)` optimism correction.

    Note: `*_cons` columns (`Q_cons`) eliminate single-partition fold-boundary variance via the unseen `lab7_r1` split,
    cutting effective fold-overfit trials by ~45% while still accounting for shared `y_search` sample noise across `M` candidates.
    """
    if not tag.startswith("ugp_"):
        return 1
    is_cons = tag.endswith("_cons") or "_cons_" in tag
    if "_g10_" in tag or tag.endswith("_g10"):
        base_m = 104
    elif "_g50_" in tag or tag.endswith("_g50"):
        base_m = 264
    elif "_g100_" in tag or tag.endswith("_g100"):
        base_m = 464
    elif "_g250_" in tag or tag.endswith("_g250"):
        base_m = 960
    elif "_g500_" in tag or tag.endswith("_g500"):
        base_m = 1750
    elif "_g1000_" in tag or tag.endswith("_g1000"):
        base_m = 3350
    else:
        base_m = 464
    return max(48, int(round(0.55 * base_m))) if is_cons else base_m


def _deploy_audit_col(
    s_oof_j: float,
    b: np.ndarray,
    r: np.ndarray | None,
    sy: float,
    fam_tag: str,
    drift_gate: float = 2.50,
) -> np.ndarray:
    if r is None or not np.all(np.isfinite(r)):
        return b
    sc = max(float(s_oof_j), 1e-12 * sy, 1e-15)
    drift = float(np.sqrt(np.mean((r - b) ** 2))) / sc
    if drift > drift_gate:
        return b
    # Match `GM._safe_deploy_cols` (`0.50`) so extrapolating symbolic/formula columns never overshoot;
    # for evolved `ugp_*` tree/symbolic columns, weight the 7-fold cross-fitted bag (`0.75` bag / `0.25` refit) on
    # noisy tables (`s_oof_j > 0.18 * sy`) and `0.50` on precision algebraic tables (`s_oof_j <= 0.18 * sy`).
    if fam_tag.startswith("ugp_"):
        if s_oof_j <= 0.18 * sy:
            alpha = 0.50 / (1.0 + max(0.0, drift - 1.10) ** 2)
        else:
            alpha = 0.25 / (1.0 + max(0.0, drift - 0.85) ** 2)
    elif fam_tag in ("e8", "e8_traj", "e8_cv7_hyb", "gpu450") and drift <= 1.35:
        alpha = 0.74 / (1.0 + max(0.0, drift - 0.80) ** 2)
    else:
        alpha = 0.50 / (1.0 + max(0.0, drift - 1.25) ** 2)
    return (1.0 - alpha) * b + alpha * r


def fit_l2_residual_gate_h452(
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    p1_oof: np.ndarray,
    p1_aud: np.ndarray,
    Q_all: np.ndarray,
    aud_cols_all: list[np.ndarray],
    fam_tags: list[str],
) -> tuple[np.ndarray, float]:
    """Generalized K-Fold (`K=5` or `K=7`) Pool-Agnostic Matérn-3/2 `eigh` Input-Aware Residual Router."""
    r_raw = y - p1_oof
    s_base = float(np.sqrt(np.mean(r_raw * r_raw)))
    sy = max(float(np.std(y)), 1e-9)
    if s_base < 1e-5 * sy or len(y) > 3000 or Q_all.shape[1] < 2:
        return p1_aud, 0.0

    uniq_f = [int(k) for k in sorted(np.unique(lab)) if int(np.sum(lab == k)) >= 4]
    K_f = len(uniq_f)
    if K_f < 3:
        return p1_aud, 0.0

    base_f = np.array([float(np.sqrt(np.mean(r_raw[lab == k] ** 2))) for k in uniq_f])
    if np.any(base_f < 1e-12):
        return p1_aud, 0.0

    prep = GM.MultiViewPreprocessor().fit(Xs, y)
    Xs_s, _ = prep.transform(Xs)
    Xs_a, _ = prep.transform(Xa)
    top8 = np.sort(np.argsort(-prep.ard)[: min(8, Xs_s.shape[1])])

    f_rm = np.array(
        [[np.sqrt(np.mean((Q_all[lab == k, j] - y[lab == k]) ** 2)) for k in uniq_f] for j in range(Q_all.shape[1])]
    )
    med_rm = np.median(f_rm, axis=1)
    aud_mat = np.column_stack(aud_cols_all)

    # Pool-agnostic diversity split: group A = GP/Tree/Symbolic columns, group B = Neural/Matérn/Kernel columns
    idx_neural_kern = [j for j, t in enumerate(fam_tags) if t.startswith(("gpu", "icl", "e8", "m7"))]
    idx_gp_tree = [j for j in range(Q_all.shape[1]) if j not in idx_neural_kern]

    order_all = np.argsort(med_rm)
    top_main = order_all[: min(5, len(order_all))]
    dev_s = [(Q_all[:, j] - p1_oof) / max(s_base, 1e-9) for j in top_main]
    dev_a = [(aud_mat[:, j] - p1_aud) / max(s_base, 1e-9) for j in top_main]

    if idx_neural_kern and idx_gp_tree:
        best_nk = idx_neural_kern[int(np.argmin(med_rm[idx_neural_kern]))]
        best_gt = idx_gp_tree[int(np.argmin(med_rm[idx_gp_tree]))]
        if best_nk not in top_main:
            dev_s.append((Q_all[:, best_nk] - p1_oof) / max(s_base, 1e-9))
            dev_a.append((aud_mat[:, best_nk] - p1_aud) / max(s_base, 1e-9))
        if best_gt not in top_main:
            dev_s.append((Q_all[:, best_gt] - p1_oof) / max(s_base, 1e-9))
            dev_a.append((aud_mat[:, best_gt] - p1_aud) / max(s_base, 1e-9))
        diff_s = np.clip((Q_all[:, best_nk] - Q_all[:, best_gt]) / max(s_base, 1e-9), -4.0, 4.0)
        diff_a = np.clip((aud_mat[:, best_nk] - aud_mat[:, best_gt]) / max(s_base, 1e-9), -4.0, 4.0)
        for c in top8[: min(4, len(top8))]:
            dev_s.append(Xs_s[:, c] * diff_s)
            dev_a.append(Xs_a[:, c] * diff_a)
    elif len(order_all) >= 2:
        j0, j1 = int(order_all[0]), int(order_all[1])
        diff_s = np.clip((Q_all[:, j0] - Q_all[:, j1]) / max(s_base, 1e-9), -4.0, 4.0)
        diff_a = np.clip((aud_mat[:, j0] - aud_mat[:, j1]) / max(s_base, 1e-9), -4.0, 4.0)
        for c in top8[: min(4, len(top8))]:
            dev_s.append(Xs_s[:, c] * diff_s)
            dev_a.append(Xs_a[:, c] * diff_a)

    spread_s = np.log1p(np.std(np.column_stack(dev_s[: min(6, len(dev_s))]), axis=1))
    spread_a = np.log1p(np.std(np.column_stack(dev_a[: min(6, len(dev_a))]), axis=1))

    # Normalize strictly by SEARCH statistics (zero query batch interaction!)
    Zs_raw = np.clip(np.column_stack([Xs_s[:, top8]] + dev_s + [spread_s]), -5.0, 5.0)
    Za_raw = np.clip(np.column_stack([Xs_a[:, top8]] + dev_a + [spread_a]), -5.0, 5.0)
    z_mu = Zs_raw.mean(axis=0)
    z_sd = np.maximum(Zs_raw.std(axis=0), 1e-6)
    Zs = (Zs_raw - z_mu) / z_sd
    Za = np.clip((Za_raw - z_mu) / z_sd, -5.0, 5.0)

    mad_r = max(1.4826 * float(np.median(np.abs(r_raw - np.median(r_raw)))), 1e-6 * sy)
    r_train = np.clip(r_raw, -3.0 * mad_r, 3.0 * mad_r)

    min_folds_won = max(3, int(math.ceil(0.75 * K_f)))
    best_gain = 0.0
    best_cfg = None
    for ell in (1.2, 2.5, 50.0):
        K_ss = GM._matern32_gram(Zs, Zs, ell)
        fold_eigh = []
        for k in uniq_f:
            va = lab == k
            tr = ~va
            ev, V = GM._safe_eigh(K_ss[np.ix_(tr, tr)])
            vt_r = V.T @ r_train[tr]
            kv_va = K_ss[np.ix_(va, tr)] @ V
            fold_eigh.append((va, ev, vt_r, kv_va))
        for lam in (2.0, 8.0, 25.0):
            oof_r = np.zeros_like(r_raw)
            for va, ev, vt_r, kv_va in fold_eigh:
                oof_r[va] = kv_va @ (vt_r / (ev + lam))
            for eta in (0.35, 0.65, 1.0):
                cand_oof = p1_oof + eta * oof_r
                new_f = np.array([float(np.sqrt(np.mean((y[lab == k] - cand_oof[lab == k]) ** 2))) for k in uniq_f])
                folds_won = int(np.sum(new_f < base_f * 0.9995))
                gain = float(np.mean(np.log(base_f / np.maximum(new_f, 1e-15))))
                if folds_won >= min_folds_won and gain > best_gain:
                    best_gain = gain
                    best_cfg = (ell, lam, eta)

    if best_cfg is None or best_gain < 0.004:
        return p1_aud, 0.0

    ell, lam, eta = best_cfg
    K_ss = GM._matern32_gram(Zs, Zs, ell)
    K_as = GM._matern32_gram(Za, Zs, ell)
    r_bag = np.zeros(len(p1_aud), dtype=np.float64)
    for k in uniq_f:
        tr = lab != k
        ev_k, V_k = GM._safe_eigh(K_ss[np.ix_(tr, tr)])
        alpha_k = V_k @ ((V_k.T @ r_train[tr]) / (ev_k + lam))
        r_bag += (K_as[:, tr] @ alpha_k) / float(K_f)
    ev_f, V_f = GM._safe_eigh(K_ss)
    alpha_f = V_f @ ((V_f.T @ r_train) / (ev_f + lam))
    r_full = K_as @ alpha_f
    r_aud = np.clip(0.5 * r_bag + 0.5 * r_full, -2.5 * mad_r, 2.5 * mad_r)
    return p1_aud + eta * r_aud, best_gain


def stack_guard_cv7_context(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab: np.ndarray,
    Q_gp: np.ndarray | None,
    bcols_gp: list[np.ndarray | None] | None,
    refit_gp: list[np.ndarray | None] | None,
    named_blocks: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    use_cv7_folds: bool = True,
    med_gate: float = 1.32,
    drift_gate: float = 2.50,
    no_ugp_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Pool-agnostic CV-7 Guard + Multi-Loss BCAR / Sharp-Simplex / Calibrated-NNLS + Matérn-3/2 L2 Stacker."""
    Xs = np.asarray(Xs, dtype=np.float64)
    Xa = np.asarray(Xa, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    lab5 = np.asarray(lab, dtype=np.int64)
    n_s, n_a = len(ys), Xa.shape[0]
    ym = float(np.mean(ys))
    sy = max(float(np.std(ys)), 1e-9)
    if n_s < 14 or float(np.std(ys)) <= 1e-12:
        return np.full(n_a, ym, dtype=np.float64), {"mode": "constant_y"}

    # Use `lab5` whenever 5-fold cross-fitted pools are present so `fold_log` and `L2` never cross 5-fold OOF boundaries;
    # use stratified `CV-7` meta-folds when evaluating pure `CV-7` pools!
    pure_cons_pool = (Q_gp is None or Q_gp.shape[1] == 0) and bool(named_blocks) and all(
        blk_tag.endswith("_cons") for blk_tag, _ in named_blocks
    )
    pure_cv7_pool = (Q_gp is None or Q_gp.shape[1] == 0) and bool(named_blocks) and all(
        blk_tag in ("e8_cv7_hyb", "ugp_cv7", "ugp_2xcv7") or blk_tag.endswith(("_cv7", "_2xcv7", "_cons"))
        for blk_tag, _ in named_blocks
    )
    if use_cv7_folds and pure_cons_pool and n_s >= 35:
        lab_meta = make_stratified_kfold(cid, ys, k=7, rep=1)
    elif use_cv7_folds and pure_cv7_pool and n_s >= 35:
        lab_meta = make_stratified_kfold(cid, ys, k=7, rep=0)
    else:
        lab_meta = lab5
    uniq_f = [int(k) for k in sorted(np.unique(lab_meta)) if int(np.sum(lab_meta == k)) >= 3]
    if len(uniq_f) < 3:
        lab_meta = lab5
        uniq_f = [int(k) for k in sorted(np.unique(lab_meta)) if int(np.sum(lab_meta == k)) >= 2]

    # 1. Guard `Q_gp` first (matching `GM.guard_and_stack_2l`) and then guard the joint union symmetrically
    Q_cols: list[np.ndarray] = []
    b_cols: list[np.ndarray | None] = []
    r_cols: list[np.ndarray | None] = []
    tags_raw: list[str] = []

    if Q_gp is not None and bcols_gp is not None and refit_gp is not None and Q_gp.shape[1] > 0:
        Q_gp_g, bcols_gp_g, refit_gp_g, _ = GM.guard_candidate_pool_ya_free(ys, lab5, Q_gp, bcols_gp, refit_gp)
        for j in range(Q_gp_g.shape[1]):
            Q_cols.append(np.asarray(Q_gp_g[:, j], dtype=np.float64))
            b_cols.append(None if bcols_gp_g[j] is None else np.asarray(bcols_gp_g[j], dtype=np.float64))
            r_cols.append(None if refit_gp_g[j] is None else np.asarray(refit_gp_g[j], dtype=np.float64))
            tags_raw.append("gp_legacy")

    has_fixed_pool = bool(Q_gp is not None and Q_gp.shape[1] > 0) or any(
        not blk_tag.startswith("ugp_") for blk_tag, _ in named_blocks
    )
    n_eff_wc = float(max(n_s, 30))

    for blk_tag, (Qb, Bb, Rb) in named_blocks:
        m_tr = _ugp_effective_trials(blk_tag) if has_fixed_pool else 1
        c_wc = math.sqrt(1.0 + (2.0 * math.log(float(m_tr))) / n_eff_wc) if m_tr > 1 else 1.0
        for j in range(Qb.shape[1]):
            qj_raw = np.asarray(Qb[:, j], dtype=np.float64)
            # Apply extreme-value selection-bias deflation BEFORE `guard_candidate_pool_ya_free` so evolved `ugp_*`
            # columns can never prematurely prune fixed (`M_trials = 1`) models via un-deflated OOF optimism!
            qj_def = (ys + (qj_raw - ys) * c_wc) if c_wc > 1.0 else qj_raw
            Q_cols.append(qj_def)
            b_cols.append(np.asarray(Bb[:, j], dtype=np.float64))
            r_cols.append(np.asarray(Rb[:, j], dtype=np.float64))
            tags_raw.append(blk_tag)

    if not Q_cols:
        return np.full(n_a, ym, dtype=np.float64), {"mode": "empty_pool"}

    Q_joint = np.column_stack(Q_cols)
    Q_jg_eval, b_jg, r_jg, kept_j = GM.guard_candidate_pool_ya_free(ys, lab5, Q_joint, b_cols, r_cols)
    tags_jg = [tags_raw[idx] for idx in kept_j]
    n_jg = Q_jg_eval.shape[1]
    if n_jg == 0:
        return np.full(n_a, ym, dtype=np.float64), {"mode": "all_guarded_out"}

    non_ugp_idx = [j for j, t in enumerate(tags_jg) if not t.startswith("ugp_")]

    # Pool-agnostic global precision anchor across the guarded (and selection-deflated) pool
    rm_all = np.sqrt(np.mean((Q_jg_eval - ys[:, None]) ** 2, axis=0))
    med_all = np.array(
        [float(np.median([np.sqrt(np.mean((Q_jg_eval[lab_meta == k, j] - ys[lab_meta == k]) ** 2)) for k in uniq_f])) for j in range(n_jg)]
    )
    min_oof_glob = float(np.min(rm_all))
    min_med_glob = float(np.min(med_all))

    # Fixed-pool OOF consensus (`M_trials = 1` models) to guard against correlated winner's-curse clones
    rm_fix = None
    e_fix = None
    f_rm_fix = None
    if non_ugp_idx:
        rm_non = rm_all[non_ugp_idx]
        min_non = float(np.min(rm_non))
        top_non = [non_ugp_idx[k] for k in range(len(non_ugp_idx)) if float(rm_non[k]) <= 1.35 * max(min_non, 1e-12 * sy)]
        if not top_non:
            top_non = [non_ugp_idx[int(np.argmin(rm_non))]]
        w_fix = bcar_weights_pool_agnostic(Q_jg_eval[:, top_non], ys, f"{cid}_fixref", "fold_log", lab_meta, n_bags=16)
        p_fix_oof = Q_jg_eval[:, top_non] @ w_fix
        e_fix = p_fix_oof - ys
        rm_fix = max(float(np.sqrt(np.mean(e_fix * e_fix))), 1e-15)
        f_rm_fix = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - p_fix_oof[lab_meta == k]) ** 2))) for k in uniq_f])

    # Base OOF blend reference (`gp_legacy` if available and competitive, else global top leaders)
    gp_idx_jg = [j for j, t in enumerate(tags_jg) if t == "gp_legacy"]
    if gp_idx_jg and float(np.min(rm_all[gp_idx_jg])) <= 1.65 * max(min_oof_glob, 1e-12 * sy):
        ref_idx = np.array(gp_idx_jg, dtype=np.int64)
    else:
        ref_idx = np.flatnonzero(rm_all <= 1.25 * max(min_oof_glob, 1e-12 * sy))
        if len(ref_idx) == 0:
            ref_idx = np.array([int(np.argmin(rm_all))], dtype=np.int64)
    w_anc = GM.bcar_weights(Q_jg_eval[:, ref_idx], ys, cid, "mse")
    e_anc = Q_jg_eval[:, ref_idx] @ w_anc - ys
    rm_anc = max(float(np.sqrt(np.mean(e_anc * e_anc))), 1e-15)
    norm_anc = max(float(np.linalg.norm(e_anc)), 1e-15)

    # Rank `ugp_*` candidates by deflated OOF RMSE, deduplicate near-identical residual clones (`corr > 0.985`),
    # and cap to top-3 genuinely distinct columns so hyperparameter twins never gang up on BCAR
    ugp_indices_all = [j for j, t in enumerate(tags_jg) if t.startswith("ugp_") and math.isfinite(float(rm_all[j]))]
    if non_ugp_idx and ugp_indices_all:
        ugp_sorted = sorted(ugp_indices_all, key=lambda idx_u: float(rm_all[idx_u]))
        dedup_ugp: list[int] = []
        for idx_u in ugp_sorted:
            eu = Q_jg_eval[:, idx_u] - ys
            nu = max(float(np.linalg.norm(eu)), 1e-15)
            is_dup = False
            for kept_u in dedup_ugp:
                ek = Q_jg_eval[:, kept_u] - ys
                nk = max(float(np.linalg.norm(ek)), 1e-15)
                if float(np.dot(eu, ek) / (nu * nk)) > 0.985:
                    is_dup = True
                    break
            if not is_dup:
                dedup_ugp.append(idx_u)
            if len(dedup_ugp) >= 3:
                break
        ugp_top_allow = set(dedup_ugp)
    else:
        ugp_top_allow = set(ugp_indices_all)

    Q_adm: list[np.ndarray] = []
    aud_adm: list[np.ndarray] = []
    b_adm: list[np.ndarray] = []
    r_adm: list[np.ndarray | None] = []
    tags_adm: list[str] = []

    for j in range(n_jg):
        qj = Q_jg_eval[:, j]
        bj = b_jg[j]
        rj = r_jg[j]
        tj = tags_jg[j]
        rm_j = float(rm_all[j])
        med_j = float(med_all[j])
        if not math.isfinite(rm_j) or bj is None or not np.all(np.isfinite(bj)):
            continue

        # Bidirectional precision-dilution veto: prune inferior columns from ANY pool (including `gp_legacy`)
        # whenever a precision leader (`min_oof_glob < 0.03 * sy` or `min_oof_glob < 0.16 * sy`) exists!
        if min_oof_glob < 0.03 * sy:
            if rm_j > 2.6 * max(min_oof_glob, 1e-12 * sy) or med_j > 2.6 * max(min_med_glob, 1e-12 * sy):
                continue
        elif min_oof_glob < 0.16 * sy:
            if rm_j > 1.48 * max(min_oof_glob, 1e-12 * sy) and med_j > 1.48 * max(min_med_glob, 1e-12 * sy):
                continue

        if tj == "gp_legacy":
            # Preserve full guarded GP diversity on non-diluted tables (`GM.guard_and_stack_2l` parity)
            Q_adm.append(qj)
            b_adm.append(bj)
            r_adm.append(rj)
            aud_adm.append(_deploy_audit_col(rm_j, bj, rj, sy, tj, drift_gate=drift_gate))
            tags_adm.append(tj)
            continue

        denom_j = max(rm_j, 0.25 * min_med_glob, 1e-12 * sy, 1e-15)
        drift_j = float(np.sqrt(np.mean((rj - bj) ** 2))) / denom_j if (rj is not None and np.all(np.isfinite(rj))) else 0.0
        if drift_j > drift_gate:
            rj = None

        ej = qj - ys
        if tj.startswith("ugp_") and non_ugp_idx and f_rm_fix is not None and e_fix is not None and rm_fix is not None:
            if j not in ugp_top_allow:
                continue
            f_rm_j = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - qj[lab_meta == k]) ** 2))) for k in uniq_f])
            folds_beat = int(np.sum(f_rm_j < f_rm_fix * 0.995))
            min_f_req = int(math.ceil(0.80 * len(uniq_f)))  # e.g. >= 4/5 folds!
            if not (folds_beat >= min_f_req and rm_j <= 0.985 * rm_fix):
                continue

        r_rat = rm_j / rm_anc
        rho = float(np.dot(e_anc, ej) / max(norm_anc * float(np.linalg.norm(ej)), 1e-15))
        h434_ok = bool(
            min_oof_glob >= 0.03 * sy
            and r_rat <= 2.0
            and rho <= (0.90 - 0.10 * (r_rat ** 2)) / max(0.80 * r_rat, 1e-12)
        )
        if med_j <= med_gate * min_med_glob or rm_j <= med_gate * min_oof_glob or h434_ok:
            Q_adm.append(qj)
            b_adm.append(bj)
            r_adm.append(rj)
            aud_adm.append(_deploy_audit_col(rm_j, bj, rj, sy, tj, drift_gate=drift_gate))
            tags_adm.append(tj)

    if not Q_adm:
        best_j = int(np.argmin(rm_all))
        bj = b_jg[best_j] if b_jg[best_j] is not None else np.full(n_a, ym, dtype=np.float64)
        Q_adm = [Q_jg_eval[:, best_j]]
        b_adm = [bj]
        r_adm = [r_jg[best_j]]
        aud_adm = [_deploy_audit_col(float(rm_all[best_j]), bj, r_jg[best_j], sy, tags_jg[best_j], drift_gate=drift_gate)]
        tags_adm = [tags_jg[best_j]]

    # 2b. Ensemble-Level Non-Degradation Ratchet (`With-UGP` vs `Without-UGP` strictly on `(Xs, ys, lab_meta)`):
    # Only retain admitted `ugp_*` columns in a mixed stack when the joint BCAR fit beats the pure fixed-pool BCAR fit
    # across >= 80% of folds (`>= 4/5`) AND improves fold-log OOF utility by >= +0.006 u!
    adm_non_ugp = [idx_a for idx_a, t_a in enumerate(tags_adm) if not t_a.startswith("ugp_")]
    adm_has_ugp = any(t_a.startswith("ugp_") for t_a in tags_adm)
    capped_bcar_override: np.ndarray | None = None
    lock_noisy_ugp_cap = False
    if adm_non_ugp and adm_has_ugp:
        Q_fix_only = np.column_stack([Q_adm[idx_a] for idx_a in adm_non_ugp])
        Q_with_ugp = np.column_stack(Q_adm)
        w_fo = GM._prune(
            0.50 * bcar_weights_pool_agnostic(Q_fix_only, ys, cid, "fold_log", lab_meta)
            + 0.35 * bcar_weights_pool_agnostic(Q_fix_only, ys, cid, "huber", lab_meta)
            + 0.15 * bcar_weights_pool_agnostic(Q_fix_only, ys, cid, "br_mse", lab_meta)
        )
        w_wu = GM._prune(
            0.50 * bcar_weights_pool_agnostic(Q_with_ugp, ys, cid, "fold_log", lab_meta)
            + 0.35 * bcar_weights_pool_agnostic(Q_with_ugp, ys, cid, "huber", lab_meta)
            + 0.15 * bcar_weights_pool_agnostic(Q_with_ugp, ys, cid, "br_mse", lab_meta)
        )
        w_fo_n = w_fo / max(float(w_fo.sum()), 1e-12)
        w_wu_n = w_wu / max(float(w_wu.sum()), 1e-12)
        oof_fo = Q_fix_only @ w_fo_n
        oof_wu_raw = Q_with_ugp @ w_wu_n
        rm_f_fo = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - oof_fo[lab_meta == k]) ** 2))) for k in uniq_f])
        rm_f_wu_raw = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - oof_wu_raw[lab_meta == k]) ** 2))) for k in uniq_f])
        raw_gain_u = float(np.mean(np.log(np.maximum(rm_f_fo, 1e-15) / np.maximum(rm_f_wu_raw, 1e-15))))

        # High-noise Caruana simplex-collapse guard (James-Stein shrinkage across M_trials):
        # Keep 100% BCAR weight on precision tables (`min_oof_glob < 0.20 * sy`) or huge breakthroughs (`raw_gain_u >= 0.20`);
        # on genuinely noisy tables (`min_oof_glob >= 0.20 * sy`, `raw_gain_u < 0.20`), cap total `ugp_*` simplex share at
        # `<= 0.32` so evolved clones add diversity without wiping out multi-family (`gp_legacy + m7 + tab + 104M`) hedging!
        ugp_mask = np.array([t_a.startswith("ugp_") for t_a in tags_adm], dtype=bool)
        s_ugp = float(np.sum(w_wu_n[ugp_mask]))
        if min_oof_glob >= 0.20 * sy and raw_gain_u < 0.20 and s_ugp > 0.0:
            w_cap = min(0.32, max(0.10, 2.5 * max(0.0, raw_gain_u)))
            if s_ugp > w_cap:
                w_blend = np.zeros_like(w_wu_n)
                w_blend[ugp_mask] = w_wu_n[ugp_mask] * (w_cap / s_ugp)
                w_blend[adm_non_ugp] = w_fo_n * (1.0 - w_cap)
                w_wu_n = GM._prune(w_blend)
                w_wu_n = w_wu_n / max(float(w_wu_n.sum()), 1e-12)
                lock_noisy_ugp_cap = True

        oof_wu = Q_with_ugp @ w_wu_n
        rm_f_wu = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - oof_wu[lab_meta == k]) ** 2))) for k in uniq_f])
        folds_won_ugp = int(np.sum(rm_f_wu < rm_f_fo * 0.999))
        gain_ugp_u = float(np.mean(np.log(np.maximum(rm_f_fo, 1e-15) / np.maximum(rm_f_wu, 1e-15))))
        max_m_ugp = max((_ugp_effective_trials(t_a) for t_a in tags_adm if t_a.startswith("ugp_")), default=104)
        min_gain_req = 0.004 + 0.0020 * max(0.0, math.log(float(max_m_ugp) / 104.0))
        if folds_won_ugp >= int(math.ceil(0.80 * len(uniq_f))) and gain_ugp_u >= min_gain_req:
            capped_bcar_override = w_wu_n
        else:
            Q_adm = [Q_adm[idx_a] for idx_a in adm_non_ugp]
            b_adm = [b_adm[idx_a] for idx_a in adm_non_ugp]
            r_adm = [r_adm[idx_a] for idx_a in adm_non_ugp]
            aud_adm = [aud_adm[idx_a] for idx_a in adm_non_ugp]
            tags_adm = [tags_adm[idx_a] for idx_a in adm_non_ugp]

    if no_ugp_fallback is not None and not any(t_a.startswith("ugp_") for t_a in tags_adm):
        return no_ugp_fallback, {"mode": "reverted_to_no_ugp_fallback"}

    Q_all = np.column_stack(Q_adm)  # [n_s, M]
    P_all = np.column_stack(aud_adm)  # [n_a, M]
    M_cnt = Q_all.shape[1]

    # 3. Primary Pool-Agnostic Multi-Loss 32-Bag BCAR Fit (`0.50 / 0.35 / 0.15` with leader-protected `br_mse`)
    if capped_bcar_override is not None and len(capped_bcar_override) == M_cnt:
        w_bcar = capped_bcar_override
    else:
        w_bcar = GM._prune(
            0.50 * bcar_weights_pool_agnostic(Q_all, ys, cid, "fold_log", lab_meta)
            + 0.35 * bcar_weights_pool_agnostic(Q_all, ys, cid, "huber", lab_meta)
            + 0.15 * bcar_weights_pool_agnostic(Q_all, ys, cid, "br_mse", lab_meta)
        )
    keep_w = np.flatnonzero(w_bcar > 0)
    ww_bcar = w_bcar[keep_w] / w_bcar[keep_w].sum()
    p1_oof_bcar = Q_all[:, keep_w] @ ww_bcar
    p1_aud_bcar = P_all[:, keep_w] @ ww_bcar

    # 4. Strict All-Fold (`100%` of outer folds + `gain_u >= 0.015`) Ratchet for Sharp-Simplex / Calibrated-NNLS
    selected_meta = "bcar_pool_agnostic"
    p1_oof = p1_oof_bcar
    p1_aud = p1_aud_bcar

    if M_cnt >= 2 and n_s >= 50 and len(uniq_f) >= 5 and not lock_noisy_ugp_cap:
        oof_bcar_nest = np.zeros(n_s, dtype=np.float64)
        oof_sharp_nest = np.zeros(n_s, dtype=np.float64)
        oof_nnls_nest = np.zeros(n_s, dtype=np.float64)
        nest_ok = True

        for fk in uniq_f:
            va = lab_meta == fk
            tr = ~va
            if int(np.sum(va)) < 2 or int(np.sum(tr)) < 16:
                nest_ok = False
                break
            Q_tr, y_tr, lab_tr = Q_all[tr], ys[tr], lab_meta[tr]
            wb_f = GM._prune(
                0.50 * bcar_weights_pool_agnostic(Q_tr, y_tr, f"{cid}_nf{fk}", "fold_log", lab_tr, n_bags=16)
                + 0.35 * bcar_weights_pool_agnostic(Q_tr, y_tr, f"{cid}_nf{fk}", "huber", lab_tr, n_bags=16)
                + 0.15 * bcar_weights_pool_agnostic(Q_tr, y_tr, f"{cid}_nf{fk}", "br_mse", lab_tr, n_bags=16)
            )
            oof_bcar_nest[va] = Q_all[va] @ wb_f

            ws_f = _fit_sharp_foldlog_simplex(Q_tr, y_tr, lab_tr, top_k=8)
            oof_sharp_nest[va] = Q_all[va] @ ws_f

            oof_nnls_nest[va] = _fit_calibrated_nnls(Q_tr, y_tr, Q_all[va], wb_f, ridge_scale=4.0)

        if nest_ok:
            w_sharp_full = _fit_sharp_foldlog_simplex(Q_all, ys, lab_meta, top_k=8)
            p_oof_sharp_full = Q_all @ w_sharp_full
            p_aud_sharp_full = P_all @ w_sharp_full

            p_oof_nnls_full = _fit_calibrated_nnls(Q_all, ys, Q_all, w_bcar, ridge_scale=4.0)
            p_aud_nnls_full = _fit_calibrated_nnls(Q_all, ys, P_all, w_bcar, ridge_scale=4.0)

            base_f_rm = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - oof_bcar_nest[lab_meta == k]) ** 2))) for k in uniq_f])
            min_f_win = len(uniq_f)  # Require 100% of outer folds (e.g. 5/5) to beat BCAR!
            best_meta_gain = 0.0

            for arm_name, cand_nest, cand_oof_full, cand_aud_full in (
                ("sharp_foldlog", oof_sharp_nest, p_oof_sharp_full, p_aud_sharp_full),
                ("bcar_sharp_50_50", 0.5 * (oof_bcar_nest + oof_sharp_nest), 0.5 * (p1_oof_bcar + p_oof_sharp_full), 0.5 * (p1_aud_bcar + p_aud_sharp_full)),
                ("bcar_nnls_65_35", 0.65 * oof_bcar_nest + 0.35 * oof_nnls_nest, 0.65 * p1_oof_bcar + 0.35 * p_oof_nnls_full, 0.65 * p1_aud_bcar + 0.35 * p_aud_nnls_full),
                ("nnls_calib", oof_nnls_nest, p_oof_nnls_full, p_aud_nnls_full),
            ):
                if not (np.all(np.isfinite(cand_nest)) and np.all(np.isfinite(cand_aud_full))):
                    continue
                new_f_rm = np.array([float(np.sqrt(np.mean((ys[lab_meta == k] - cand_nest[lab_meta == k]) ** 2))) for k in uniq_f])
                f_won = int(np.sum(new_f_rm < base_f_rm * 0.999))
                gain_u = float(np.mean(np.log(np.maximum(base_f_rm, 1e-15) / np.maximum(new_f_rm, 1e-15))))
                if f_won >= min_f_win and gain_u > max(0.015, best_meta_gain):
                    best_meta_gain = gain_u
                    selected_meta = arm_name
                    p1_oof = cand_oof_full
                    p1_aud = cand_aud_full

    # 5. L2 Input-Aware Matérn-3/2 `eigh` Residual Router:
    # When `gp_legacy` columns remain in `Q_all` and `lab_meta` is 5-fold, preserve `GM.fit_l2_residual_gate`'s exact
    # `4/5`-fold gating; otherwise use the generalized K-fold pool-agnostic router!
    n_gp_adm = sum(1 for t in tags_adm if t == "gp_legacy")
    if n_gp_adm >= 1 and len(uniq_f) == 5 and all(t == "gp_legacy" for t in tags_adm[:n_gp_adm]):
        p2_aud, l2_gain = GM.fit_l2_residual_gate(Xs, Xa, ys, lab5, p1_oof, p1_aud, Q_all, aud_adm, n_gp_adm)
    else:
        p2_aud, l2_gain = fit_l2_residual_gate_h452(Xs, Xa, ys, lab_meta, p1_oof, p1_aud, Q_all, aud_adm, tags_adm)

    p2_aud = np.where(np.isfinite(p2_aud), p2_aud, ym)

    fam_weights: dict[str, float] = {}
    for tj, wj in zip(tags_adm, w_bcar):
        if wj > 0:
            fam_weights[tj] = fam_weights.get(tj, 0.0) + float(wj)

    meta = {
        "n_joint_guarded": int(n_jg),
        "n_admitted": int(M_cnt),
        "meta_folds": int(len(uniq_f)),
        "selected_meta": selected_meta,
        "l2_gain": float(l2_gain),
        "min_oof_global": float(min_oof_glob),
        "fam_bcar_weights": fam_weights,
    }
    return p2_aud, meta
