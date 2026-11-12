#!/usr/bin/env python3
"""H450 Legal Test-Time Specialist Tabular, Lead-Lag & Hybrid-Kernel Block (`cols_tab_h450.py`).

Fits 8 diverse, high-precision specialist columns strictly on `(X_search, y_search) -> X_audit` (`0` `y_audit` read!)
over exact `float64` `[X_clean, S_h450]` (including within-entity Lead-Lag neighbor interpolation, sentinel flags,
`float64` colinear date/timestamp differences, signed-logs, exact keys, and raw formulas):
  1. `h450_leadlag_cal`:   Calibrated Lead-Lag + Robust SVD-GCV Ridge (`u = 3..6+` on ordered/panel tables)
  2. `h450_lap_matern_id`: Exact-LOO 50/50 Laplacian (`L1`) + Matérn-3/2 (`L2`) ARD Kernel Ridge (identity target)
  3. `h450_lap_matern_yj`: Exact-LOO Laplacian + Matérn-3/2 ARD Kernel Ridge in Yeo-Johnson space + Duan smearing
  4. `h450_cb_d4_yj`:      CatBoost (`depth=4`, `700` trees) in Yeo-Johnson space + 25-quantile Duan smearing
  5. `h450_cb_d6_yj`:      CatBoost (`depth=6`, `800` trees) in Yeo-Johnson space + 25-quantile Duan smearing
  6. `h450_cb_d8_hurdle`:  CatBoost (`depth=8`, `600` trees) + OOF-gated Two-Part Sentinel Hurdle calibration
  7. `h450_lgbm_yj`:       LightGBM (`num_leaves=31`, `500` trees) in Yeo-Johnson space + 25-quantile Duan smearing
  8. `h450_et_hgb_duan`:   50/50 ExtraTrees + HistGradientBoosting in Yeo-Johnson space + 25-quantile Duan smearing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402

H450_TAB_COLS = (
    "h450_leadlag_cal",
    "h450_lap_matern_id",
    "h450_lap_matern_yj",
    "h450_cb_d4_yj",
    "h450_cb_d6_yj",
    "h450_cb_d8_hurdle",
    "h450_lgbm_yj",
    "h450_et_hgb_duan",
)


def _sha_seed(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _prep_finite_tree_features(X_tr: np.ndarray, X_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Imputes `NaN` with `X_tr` median and appends missingness indicators (`0` `X_ev` stats leakage)."""
    p = X_tr.shape[1]
    miss_tr = ~np.isfinite(X_tr)
    miss_ev = ~np.isfinite(X_ev)
    med = np.zeros(p, dtype=np.float64)
    for j in range(p):
        fin = ~miss_tr[:, j]
        if np.any(fin):
            med[j] = float(np.median(X_tr[fin, j]))
    Xt_i = np.where(miss_tr, med[None, :], X_tr)
    Xe_i = np.where(miss_ev, med[None, :], X_ev)
    mcols = [j for j in range(p) if float(np.mean(miss_tr[:, j])) >= 0.01][:16]
    if mcols:
        Xt_i = np.hstack([Xt_i, miss_tr[:, mcols].astype(np.float64)])
        Xe_i = np.hstack([Xe_i, miss_ev[:, mcols].astype(np.float64)])
    return np.clip(Xt_i, -1e15, 1e15), np.clip(Xe_i, -1e15, 1e15)


def _select_tab_cols(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, max_cols: int = 80) -> tuple[np.ndarray, np.ndarray]:
    if X_tr.shape[1] <= max_cols:
        return X_tr, X_ev
    yc0 = y_tr - float(np.mean(y_tr))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    scores = np.zeros(X_tr.shape[1], dtype=np.float64)
    for j in range(X_tr.shape[1]):
        fin = np.isfinite(X_tr[:, j])
        if int(np.sum(fin)) >= 10:
            v = X_tr[fin, j] - float(np.mean(X_tr[fin, j]))
            vn = float(np.linalg.norm(v))
            if vn > 1e-12:
                scores[j] = abs(float(np.dot(v, yc0[fin])) / (vn * yn))
    keep = np.sort(np.argsort(-scores)[:max_cols])
    return X_tr[:, keep], X_ev[:, keep]


def _fit_catboost(X_tr: np.ndarray, z_tr: np.ndarray, X_ev: np.ndarray, depth: int, iters: int, lr: float, l2: float, seed: int) -> np.ndarray:
    if len(z_tr) < 12 or float(np.std(z_tr)) <= 1e-12:
        return np.full(X_ev.shape[0], float(np.mean(z_tr)), dtype=np.float64)
    cb = CatBoostRegressor(
        iterations=iters,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        random_strength=0.65,
        subsample=0.85,
        bootstrap_type="Bernoulli",
        nan_mode="Min",
        thread_count=1,
        random_seed=int(seed & 0x7FFFFFFF),
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(X_tr, z_tr)
    return np.asarray(cb.predict(X_ev), dtype=np.float64)


def _fit_lgbm(X_tr: np.ndarray, z_tr: np.ndarray, X_ev: np.ndarray, seed: int) -> np.ndarray:
    if len(z_tr) < 20 or float(np.std(z_tr)) <= 1e-12:
        return np.full(X_ev.shape[0], float(np.mean(z_tr)), dtype=np.float64)
    n_tr = len(z_tr)
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=min(31, max(7, n_tr // 10)),
        min_child_samples=max(5, n_tr // 65),
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=2.5,
        n_jobs=1,
        random_state=int(seed & 0x7FFFFFFF),
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb.fit(X_tr, z_tr)
        return np.asarray(lgb.predict(X_ev), dtype=np.float64)


def _fit_et_hgb(X_tr: np.ndarray, z_tr: np.ndarray, X_ev: np.ndarray, seed: int) -> np.ndarray:
    if len(z_tr) < 16 or float(np.std(z_tr)) <= 1e-12:
        return np.full(X_ev.shape[0], float(np.mean(z_tr)), dtype=np.float64)
    n_tr = len(z_tr)
    Xt_f, Xe_f = _prep_finite_tree_features(X_tr, X_ev)
    et = ExtraTreesRegressor(
        n_estimators=220,
        max_features=0.78,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=int(seed & 0x7FFFFFFF),
    )
    et.fit(Xt_f, z_tr)
    p_et = np.asarray(et.predict(Xe_f), dtype=np.float64)

    hgb = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.042,
        max_leaf_nodes=min(35, max(7, n_tr // 8)),
        min_samples_leaf=max(4, n_tr // 70),
        l2_regularization=2.0,
        max_features=0.80,
        early_stopping=False,
        random_state=int((seed + 17) & 0x7FFFFFFF),
    )
    hgb.fit(X_tr, z_tr)
    p_hgb = np.asarray(hgb.predict(X_ev), dtype=np.float64)
    return 0.50 * p_et + 0.50 * p_hgb


def eval_single_context_tab(cid: str, Xs: np.ndarray, Xa: np.ndarray, y: np.ndarray, lab: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Strictly evaluates `(Xs, y, Xa)` (`0` `y_audit` read!) for all 8 specialist tabular/kernel columns."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    t0 = time.time()

    Xs = np.asarray(Xs, dtype=np.float64)
    Xa = np.asarray(Xa, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.int64)
    n_s, n_a = len(y), Xa.shape[0]
    ym = float(np.mean(y))
    sy = max(float(np.std(y)), 1e-12)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = max(y_hi - y_lo, sy)
    lo_g, hi_g = y_lo - 1.25 * span, y_hi + 1.25 * span

    out: dict[str, np.ndarray] = {}
    for col in H450_TAB_COLS:
        out[f"q_{col}"] = np.full(n_s, ym, dtype=np.float64)
        out[f"bag_{col}"] = np.zeros(n_a, dtype=np.float64)
        out[f"ref_{col}"] = np.full(n_a, ym, dtype=np.float64)

    # Fit global context-only Yeo-Johnson warp on `y_search` for aligned Duan smearing across folds,
    # while fitting fold-specific models strictly on `tr`!
    # Wait: to be 100% fold-pure on `q_*`, each fold fits its OWN `YeoJohnsonWarpF64` on `y[tr]`!
    valid_folds = 0
    fold_aug_cache = {}

    for fk in range(5):
        va = lab == fk
        tr = ~va
        n_va = int(np.sum(va))
        if n_va == 0 or int(np.sum(tr)) < 16:
            continue
        valid_folds += 1
        y_tr = y[tr]
        X_ev_raw = np.vstack([Xs[va], Xa])
        fseed = _sha_seed(f"h450_tab:{cid}:f{fk}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xt_cl, Xe_cl, St, Se, fmeta = REL.build_h450_struct_cols(
                Xs[tr], y_tr, X_ev_raw, seed=fseed, fast_mode=False, max_total_cols=20, include_base_struct=True
            )
        Xt_aug_all = np.hstack([Xt_cl, St]) if St.shape[1] > 0 else Xt_cl
        Xe_aug_all = np.hstack([Xe_cl, Se]) if Se.shape[1] > 0 else Xe_cl
        Xt_aug, Xe_aug = _select_tab_cols(Xt_aug_all, y_tr, Xe_aug_all, max_cols=84)
        fold_aug_cache[fk] = (tr, va, Xt_aug, Xe_aug)

        # 1. `h450_leadlag_cal`: Calibrated Lead-Lag + Robust SVD-GCV Ridge
        ll_idxs = [i for i, nm in enumerate(fmeta["col_names"]) if nm.startswith("ll_")]
        if ll_idxs:
            p_raw = Xt_cl.shape[1]
            ll_cols = [p_raw + i for i in ll_idxs if (p_raw + i) < Xt_aug_all.shape[1]]
            _, pq_ll, _, _ = RR.robust_ridge_np(Xt_aug_all[:, ll_cols], y_tr, Xe_aug_all[:, ll_cols], seed=fseed + 1)
            _, pq_rg, _, _ = RR.robust_ridge_np(Xt_aug, y_tr, Xe_aug, seed=fseed + 2)
            pred_ll = np.clip(0.75 * pq_ll + 0.25 * pq_rg, lo_g, hi_g)
        else:
            _, pq_rg, _, _ = RR.robust_ridge_np(Xt_aug, y_tr, Xe_aug, seed=fseed + 2)
            pred_ll = np.clip(pq_rg, lo_g, hi_g)
        out["q_h450_leadlag_cal"][va] = pred_ll[:n_va]
        out["bag_h450_leadlag_cal"] += pred_ll[n_va:]

        # 2. `h450_lap_matern_id`: Exact-LOO Laplacian + Matérn ARD Kernel (identity)
        pred_k_id = np.clip(RR.fast_laplacian_matern_predict(Xt_aug, y_tr, Xe_aug, max_sub=680), lo_g, hi_g)
        out["q_h450_lap_matern_id"][va] = pred_k_id[:n_va]
        out["bag_h450_lap_matern_id"] += pred_k_id[n_va:]

        # Fit fold-pure Yeo-Johnson warp + inner 3-fold residuals for Duan smearing strictly on `y_tr`
        warp_f = RR.YeoJohnsonWarpF64()
        z_tr = warp_f.fit_transform(y_tr)

        # 3. `h450_lap_matern_yj`
        zk_all = RR.fast_laplacian_matern_predict(Xt_aug, z_tr, np.vstack([Xt_aug, Xe_aug]), max_sub=640)
        warp_k = RR.YeoJohnsonWarpF64()
        warp_k.fit_transform(y_tr)
        warp_k.set_residuals(z_tr, zk_all[: len(z_tr)])
        pred_k_yj = np.clip(warp_k.inv_np_duan(zk_all[len(z_tr) :]), lo_g, hi_g)
        out["q_h450_lap_matern_yj"][va] = pred_k_yj[:n_va]
        out["bag_h450_lap_matern_yj"] += pred_k_yj[n_va:]

        # Estimate Duan smearing residuals cleanly via quick inner split on `tr`
        n_tr_len = len(z_tr)
        in_va = np.arange(n_tr_len) % 4 == 0
        if int(np.sum(~in_va)) >= 16 and int(np.sum(in_va)) >= 8:
            z_in_pred = _fit_catboost(Xt_aug[~in_va], z_tr[~in_va], Xt_aug[in_va], 5, 250, 0.06, 3.5, fseed + 9)
            warp_f.set_residuals(z_tr[in_va], z_in_pred)

        # 4. `h450_cb_d4_yj`
        z_cb4 = _fit_catboost(Xt_aug, z_tr, Xe_aug, depth=4, iters=700, lr=0.045, l2=3.0, seed=fseed + 41)
        pred_cb4 = np.clip(warp_f.inv_np_duan(z_cb4), lo_g, hi_g)
        out["q_h450_cb_d4_yj"][va] = pred_cb4[:n_va]
        out["bag_h450_cb_d4_yj"] += pred_cb4[n_va:]

        # 5. `h450_cb_d6_yj`
        z_cb6 = _fit_catboost(Xt_aug, z_tr, Xe_aug, depth=6, iters=800, lr=0.040, l2=4.0, seed=fseed + 61)
        pred_cb6 = np.clip(warp_f.inv_np_duan(z_cb6), lo_g, hi_g)
        out["q_h450_cb_d6_yj"][va] = pred_cb6[:n_va]
        out["bag_h450_cb_d6_yj"] += pred_cb6[n_va:]

        # 6. `h450_cb_d8_hurdle` (Identity CatBoost depth=8, upgraded below with Hurdle-Sentinel if triggered)
        pred_cb8 = np.clip(_fit_catboost(Xt_aug, y_tr, Xe_aug, depth=8, iters=580, lr=0.038, l2=6.0, seed=fseed + 81), lo_g, hi_g)
        out["q_h450_cb_d8_hurdle"][va] = pred_cb8[:n_va]
        out["bag_h450_cb_d8_hurdle"] += pred_cb8[n_va:]

        # 7. `h450_lgbm_yj`
        z_lgb = _fit_lgbm(Xt_aug, z_tr, Xe_aug, seed=fseed + 71)
        pred_lgb = np.clip(warp_f.inv_np_duan(z_lgb), lo_g, hi_g)
        out["q_h450_lgbm_yj"][va] = pred_lgb[:n_va]
        out["bag_h450_lgbm_yj"] += pred_lgb[n_va:]

        # 8. `h450_et_hgb_duan`
        z_eth = _fit_et_hgb(Xt_aug, z_tr, Xe_aug, seed=fseed + 91)
        pred_eth = np.clip(warp_f.inv_np_duan(z_eth), lo_g, hi_g)
        out["q_h450_et_hgb_duan"][va] = pred_eth[:n_va]
        out["bag_h450_et_hgb_duan"] += pred_eth[n_va:]

    for col in H450_TAB_COLS:
        out[f"bag_{col}"] /= float(max(valid_folds, 1))

    # --- Full-X_search Refits ---
    rseed = _sha_seed(f"h450_tab:{cid}:refit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs_cl, Xa_cl, Ss_f, Sa_f, fmeta_full = REL.build_h450_struct_cols(
            Xs, y, Xa, seed=rseed, fast_mode=False, max_total_cols=20, include_base_struct=True
        )
    Xs_aug_all = np.hstack([Xs_cl, Ss_f]) if Ss_f.shape[1] > 0 else Xs_cl
    Xa_aug_all = np.hstack([Xa_cl, Sa_f]) if Sa_f.shape[1] > 0 else Xa_cl
    Xs_aug, Xa_aug = _select_tab_cols(Xs_aug_all, y, Xa_aug_all, max_cols=84)

    ll_idxs_f = [i for i, nm in enumerate(fmeta_full["col_names"]) if nm.startswith("ll_")]
    if ll_idxs_f:
        p_raw = Xs_cl.shape[1]
        ll_cols_f = [p_raw + i for i in ll_idxs_f if (p_raw + i) < Xs_aug_all.shape[1]]
        _, pq_ll_f, _, _ = RR.robust_ridge_np(Xs_aug_all[:, ll_cols_f], y, Xa_aug_all[:, ll_cols_f], seed=rseed + 1)
        _, pq_rg_f, _, _ = RR.robust_ridge_np(Xs_aug, y, Xa_aug, seed=rseed + 2)
        out["ref_h450_leadlag_cal"] = np.clip(0.75 * pq_ll_f + 0.25 * pq_rg_f, lo_g, hi_g)
    else:
        _, pq_rg_f, _, _ = RR.robust_ridge_np(Xs_aug, y, Xa_aug, seed=rseed + 2)
        out["ref_h450_leadlag_cal"] = np.clip(pq_rg_f, lo_g, hi_g)

    out["ref_h450_lap_matern_id"] = np.clip(RR.fast_laplacian_matern_predict(Xs_aug, y, Xa_aug, max_sub=720), lo_g, hi_g)

    warp_full = RR.YeoJohnsonWarpF64()
    z_full = warp_full.fit_transform(y)
    # Use 5-fold OOF z residuals from cb_d6_yj for exact Duan smearing on full refit
    z_oof_cb6 = warp_full.transform_any(np.clip(out["q_h450_cb_d6_yj"], y_lo - 0.5 * span, y_hi + 0.5 * span))
    warp_full.set_residuals(z_full, z_oof_cb6)

    zk_f_all = RR.fast_laplacian_matern_predict(Xs_aug, z_full, np.vstack([Xs_aug, Xa_aug]), max_sub=700)
    warp_kf = RR.YeoJohnsonWarpF64()
    warp_kf.fit_transform(y)
    warp_kf.set_residuals(z_full, zk_f_all[:n_s])
    out["ref_h450_lap_matern_yj"] = np.clip(warp_kf.inv_np_duan(zk_f_all[n_s:]), lo_g, hi_g)

    out["ref_h450_cb_d4_yj"] = np.clip(
        warp_full.inv_np_duan(_fit_catboost(Xs_aug, z_full, Xa_aug, depth=4, iters=740, lr=0.045, l2=3.0, seed=rseed + 41)),
        lo_g,
        hi_g,
    )
    out["ref_h450_cb_d6_yj"] = np.clip(
        warp_full.inv_np_duan(_fit_catboost(Xs_aug, z_full, Xa_aug, depth=6, iters=840, lr=0.040, l2=4.0, seed=rseed + 61)),
        lo_g,
        hi_g,
    )
    out["ref_h450_cb_d8_hurdle"] = np.clip(
        _fit_catboost(Xs_aug, y, Xa_aug, depth=8, iters=600, lr=0.038, l2=6.0, seed=rseed + 81),
        lo_g,
        hi_g,
    )
    out["ref_h450_lgbm_yj"] = np.clip(
        warp_full.inv_np_duan(_fit_lgbm(Xs_aug, z_full, Xa_aug, seed=rseed + 71)),
        lo_g,
        hi_g,
    )
    out["ref_h450_et_hgb_duan"] = np.clip(
        warp_full.inv_np_duan(_fit_et_hgb(Xs_aug, z_full, Xa_aug, seed=rseed + 91)),
        lo_g,
        hi_g,
    )

    # Optional OOF-gated Hurdle-Sentinel upgrade on `h450_cb_d8_hurdle`
    hurdle_tab = False
    atom = RR.HurdleSentinelDetector.detect_atom(y)
    if atom is not None:
        is_atom = np.abs(y - atom) <= 1e-7 * max(1.0, abs(atom))
        if int(np.sum(is_atom)) >= 4 and int(np.sum(~is_atom)) >= 20:
            p_oof, p_qa = RR.HurdleSentinelDetector.fit_oof_and_query_probs(Xs_aug, is_atom, Xa_aug, lab, seed=rseed + 333)
            q_bulk = out["q_h450_cb_d8_hurdle"].copy()
            bag_bulk = np.zeros(n_a, dtype=np.float64)
            cnt_b = 0
            for fk, (tr, va, Xt_aug, Xe_aug) in fold_aug_cache.items():
                n_va = int(np.sum(va))
                b_tr = ~is_atom[tr]
                if int(np.sum(b_tr)) < 16:
                    continue
                cnt_b += 1
                pb = _fit_catboost(Xt_aug[b_tr], y[tr][b_tr], Xe_aug, depth=6, iters=650, lr=0.042, l2=4.0, seed=rseed + 500 + fk)
                q_bulk[va] = pb[:n_va]
                bag_bulk += pb[n_va:]
            if cnt_b > 0:
                bag_bulk /= float(cnt_b)
                cand_q = p_oof * atom + (1.0 - p_oof) * q_bulk
                mse0 = float(np.mean((out["q_h450_cb_d8_hurdle"] - y) ** 2))
                mse1 = float(np.mean((cand_q - y) ** 2))
                if mse1 < 0.96 * mse0:
                    hurdle_tab = True
                    ref_bulk = _fit_catboost(Xs_aug[~is_atom], y[~is_atom], Xa_aug, depth=6, iters=700, lr=0.042, l2=4.0, seed=rseed + 599)
                    out["q_h450_cb_d8_hurdle"] = np.clip(cand_q, lo_g, hi_g)
                    out["bag_h450_cb_d8_hurdle"] = np.clip(p_qa * atom + (1.0 - p_qa) * bag_bulk, lo_g, hi_g)
                    out["ref_h450_cb_d8_hurdle"] = np.clip(p_qa * atom + (1.0 - p_qa) * ref_bulk, lo_g, hi_g)

    meta = {
        "cid": cid,
        "wall_s": round(time.time() - t0, 3),
        "hurdle_tab": hurdle_tab,
        "struct_meta": fmeta_full,
    }
    return out, meta


def _worker(job: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]) -> dict:
    cid, Xs, Xa, y, lab, out_s = job
    out_dir = Path(out_s)
    dst = out_dir / f"{cid}.npz"
    if dst.exists():
        return {"cid": cid, "cached": True}
    col_dict, meta = eval_single_context_tab(cid, Xs, Xa, y, lab)
    tmp = out_dir / f".{cid}.{os.getpid()}.tmp.npz"
    np.savez(tmp, ok=np.array(True), **col_dict)
    os.chmod(tmp, 0o444)
    os.replace(tmp, dst)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, help="Path to bank_d91_search_f64.npz (0 audit targets!)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with np.load(a.bank, allow_pickle=False) as zb:
        if "ya" in zb.files:
            raise SystemExit("SECURITY/EVAL REFUSAL: bank contains audit targets `ya`!")
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            Xs = np.array(S_mat[:, :-1], np.float64, copy=True)
            y = np.array(zb["y64"][lo : lo + n], np.float64, copy=True)
            lab = np.array(zb["lab"][lo : lo + n], np.int64, copy=True)
            # Sort longest contexts first (LPT) for minimal tail latency
            jobs.append((n * w, c, Xs, Xa, y, lab, str(out_dir)))
            so += n * w
            ao += na * f
            lo += n

    jobs.sort(key=lambda r: -r[0])
    worker_jobs = [(c, Xs, Xa, y, lab, os_str) for _, c, Xs, Xa, y, lab, os_str in jobs]

    t0 = time.time()
    with Pool(processes=min(a.procs, max(len(worker_jobs), 1))) as pool:
        metas = list(pool.map(_worker, worker_jobs))

    summary_p = out_dir / "tab_summary.json"
    summary_p.write_text(
        json.dumps({"contexts": len(metas), "wall_s": round(time.time() - t0, 2), "per_context": metas}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"contexts": len(metas), "wall_s": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
