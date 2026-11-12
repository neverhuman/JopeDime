#!/usr/bin/env python3
"""H450 Legal Test-Time Multi-Mode GPU Column Generator (`cols_h450_gpu.py`).

Evaluates `(X_search, y_search, X_audit) -> (q_*, bag_*, ref_*)` in exact `float64` normalization (`ICLH450`)
with `WIDE_GPU = 64` features and ZERO `y_audit` access (`assert "ya" not in zb.files`):
  1. `icl_robridge_a`:   SeedA (`L_A + M_A`) x 8 views on Robust-Ridge residuals (`[X_clean, S_h450]`)
  2. `icl_robridge_b`:   SeedB (`L_B + M_B`) x 8 views on Robust-Ridge residuals (`[X_clean, S_h450]`)
  3. `icl_robridge_e32`: 4-checkpoint 32-pass soup + OOF-gated Two-Part Hurdle Sentinel calibration
  4. `icl_yj_exact_e16`: 4 checkpoints x 4 views in Yeo-Johnson space via exact 1,024-bin `mean_warped` integration
  5. `icl_nlres_e16`:    4 checkpoints x 4 views on 5-fold `fast_nonlinear_base_oof` (`HGB + Laplacian-Matérn`) residuals
  6. `icl_boost2s_e16`:  2-Stage Self-Boosted `ICLH450` refining Stage-1 5-fold OOF residuals (`y - q_robridge_e32`)
                         with 5-fold OOF-selected shrinkage step `eta in {0.0, 0.25, 0.50, 0.75, 1.0}`!
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_h450 import ICLH450  # noqa: E402

WIDE_GPU = 64
H450_GPU_COLS = (
    "icl_robridge_a",
    "icl_robridge_b",
    "icl_robridge_e32",
    "icl_yj_exact_e16",
    "icl_nlres_e16",
    "icl_boost2s_e16",
)


def _sha_seed(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _load_icl_h450(ckpt_p: str, size: str, dev: torch.device) -> ICLH450:
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck["model"]
    m = ICLH450(size).to(dev).eval()
    m.load_state_dict(sd, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _build_views(
    X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, n_views: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    p = X_tr.shape[1]
    rng = np.random.default_rng(seed)
    views = []
    if p <= WIDE_GPU:
        for v in range(n_views):
            perm = np.arange(p) if v == 0 else rng.permutation(p)
            views.append((X_tr[:, perm], X_ev[:, perm]))
        return views

    # Rank columns strictly on `(X_tr, y_tr)` for wide tables (`F > 64`)
    yc0 = y_tr - float(np.mean(y_tr))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        fin = np.isfinite(X_tr[:, j])
        if int(np.sum(fin)) >= 10:
            col0 = X_tr[fin, j] - float(np.mean(X_tr[fin, j]))
            cn = float(np.linalg.norm(col0))
            if cn > 1e-12:
                scores[j] = abs(float(np.dot(col0, yc0[fin])) / (cn * yn))

    order = np.argsort(-scores)
    for v in range(n_views):
        if v == 0:
            keep = order[:WIDE_GPU]
        else:
            n_top = int(round(0.65 * WIDE_GPU))
            top = order[:n_top]
            rem = order[n_top:]
            pick = rng.choice(rem, size=WIDE_GPU - len(top), replace=False)
            keep = rng.permutation(np.concatenate([top, pick]))
        views.append((X_tr[:, keep], X_ev[:, keep]))
    return views


@torch.no_grad()
def _batched_icl_predict(
    models_and_weights: list[tuple[ICLH450, float]],
    views: list[tuple[np.ndarray, np.ndarray]],
    y_tr: np.ndarray,
    dev: torch.device,
    seed: int,
    inv_fn=None,
) -> tuple[np.ndarray, int]:
    """Runs batched `[V, N_tr + N_ev, Fn]` forward pass per model in `float64` norm + `bf16` attention."""
    V = len(views)
    n_tr = views[0][0].shape[0]
    n_ev = views[0][1].shape[0]
    Fn = views[0][0].shape[1]
    if n_ev == 0:
        return np.zeros(0, dtype=np.float64), 0

    xb_np = np.stack([np.vstack([vt, ve]) for vt, ve in views], axis=0)  # [V, N, Fn]
    yb_np = np.broadcast_to(np.asarray(y_tr, dtype=np.float64)[None, :], (V, n_tr)).copy()
    xt = torch.from_numpy(xb_np).to(device=dev, dtype=torch.float64)
    yt = torch.from_numpy(yb_np).to(device=dev, dtype=torch.float64)

    g = torch.Generator(device=dev).manual_seed(seed)
    slots = torch.randperm(FID_SLOTS, generator=g, device=dev)[:Fn]

    pred_acc = np.zeros(n_ev, dtype=np.float64)
    w_sum = 0.0
    cells_eval = 0
    for mdl, w in models_and_weights:
        logits, ym, ys = mdl(xt, yt, n_tr, slots=slots, amp=(dev.type == "cuda"))
        if inv_fn is None:
            out_v = mdl.mean_f64(logits, ym, ys)  # [V, n_ev]
        else:
            out_v = mdl.mean_warped(logits, ym, ys, inv_fn)  # [V, n_ev]
        pred_acc += float(w) * out_v.mean(dim=0).detach().cpu().numpy()
        w_sum += float(w)
        cells_eval += int(V * (n_tr + n_ev) * (Fn + 1))

    return pred_acc / max(w_sum, 1e-12), cells_eval


def eval_single_context_gpu(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    pair_a: list[tuple[ICLH450, float]],
    pair_b: list[tuple[ICLH450, float]],
    dev: torch.device,
    n_views_override: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Strictly evaluates `(Xs, y, Xa)` (`0` `y_audit` read!) and returns all 6 `H450` neural column triples."""
    t0 = time.time()
    nv8 = int(n_views_override) if n_views_override is not None else 8
    nv4 = int(n_views_override) if n_views_override is not None else 4
    Xs = np.asarray(Xs, dtype=np.float64)
    Xa = np.asarray(Xa, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.int64)
    n_s, n_a = len(y), Xa.shape[0]
    ym_glob = float(np.mean(y))
    sy_glob = max(float(np.std(y)), 1e-12)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = max(y_hi - y_lo, sy_glob)
    lo_guard, hi_guard = y_lo - 1.5 * span, y_hi + 1.5 * span

    all_four = [(m, 0.5 * w) for m, w in pair_a] + [(m, 0.5 * w) for m, w in pair_b]

    res: dict[str, np.ndarray] = {}
    for col in H450_GPU_COLS:
        res[f"q_{col}"] = np.full(n_s, ym_glob, dtype=np.float64)
        res[f"bag_{col}"] = np.zeros(n_a, dtype=np.float64)
        res[f"ref_{col}"] = np.full(n_a, ym_glob, dtype=np.float64)

    total_cells = 0
    valid_folds = 0

    # Store per-fold augmented matrices so Stage-2 Self-Boosting reuses them cleanly with zero leakage!
    fold_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for fk in range(5):
        va = lab == fk
        tr = ~va
        n_va = int(np.sum(va))
        if n_va == 0 or int(np.sum(tr)) < 16:
            continue
        valid_folds += 1
        X_tr_raw, y_tr = Xs[tr], y[tr]
        X_ev_raw = np.vstack([Xs[va], Xa])

        fseed = _sha_seed(f"h450_gpu:{cid}:f{fk}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr_cl, X_ev_cl, S_tr, S_ev, _ = REL.build_h450_struct_cols(
                X_tr_raw, y_tr, X_ev_raw, seed=fseed, fast_mode=False, max_total_cols=18, include_base_struct=True
            )
        X_tr_aug = np.hstack([X_tr_cl, S_tr]) if S_tr.shape[1] > 0 else X_tr_cl
        X_ev_aug = np.hstack([X_ev_cl, S_ev]) if S_ev.shape[1] > 0 else X_ev_cl
        fold_cache[fk] = (tr, va, X_tr_aug, X_ev_aug)

        # --- Mode 1: Robust-Ridge Residual (`SeedA`, `SeedB`, `e32`) ---
        oof_rr, pq_rr, used_rr, _ = RR.robust_ridge_np(X_tr_aug, y_tr, X_ev_aug, seed=fseed + 11)
        y_tr_rr = (y_tr - oof_rr) if used_rr else y_tr
        base_rr = pq_rr if used_rr else np.zeros(n_va + n_a, dtype=np.float64)

        views8_a = _build_views(X_tr_aug, y_tr_rr, X_ev_aug, n_views=nv8, seed=fseed + 101)
        views8_b = _build_views(X_tr_aug, y_tr_rr, X_ev_aug, n_views=nv8, seed=fseed + 202)
        pa_ev, c1 = _batched_icl_predict(pair_a, views8_a, y_tr_rr, dev, seed=fseed + 101)
        pb_ev, c2 = _batched_icl_predict(pair_b, views8_b, y_tr_rr, dev, seed=fseed + 202)
        total_cells += c1 + c2

        pred_a = np.clip(base_rr + pa_ev, lo_guard, hi_guard)
        pred_b = np.clip(base_rr + pb_ev, lo_guard, hi_guard)
        pred_e32 = 0.5 * (pred_a + pred_b)

        res["q_icl_robridge_a"][va] = pred_a[:n_va]
        res["bag_icl_robridge_a"] += pred_a[n_va:]
        res["q_icl_robridge_b"][va] = pred_b[:n_va]
        res["bag_icl_robridge_b"] += pred_b[n_va:]
        res["q_icl_robridge_e32"][va] = pred_e32[:n_va]
        res["bag_icl_robridge_e32"] += pred_e32[n_va:]

        # --- Mode 2: Yeo-Johnson Exact 1,024-Bin Softmax Integration (`icl_yj_exact_e16`) ---
        warp = RR.YeoJohnsonWarpF64()
        z_tr = warp.fit_transform(y_tr)
        views4_yj = _build_views(X_tr_aug, z_tr, X_ev_aug, n_views=nv4, seed=fseed + 303)
        pyj_ev, c3 = _batched_icl_predict(all_four, views4_yj, z_tr, dev, seed=fseed + 303, inv_fn=warp.inv_torch)
        total_cells += c3
        pred_yj = np.clip(pyj_ev, lo_guard, hi_guard)
        res["q_icl_yj_exact_e16"][va] = pred_yj[:n_va]
        res["bag_icl_yj_exact_e16"] += pred_yj[n_va:]

        # --- Mode 3: Nonlinear-Base (`HGB + Laplacian-Matérn`) Residual Zooming (`icl_nlres_e16`) ---
        oof_nl, pq_nl, used_nl, _ = RR.fast_nonlinear_base_oof(X_tr_aug, y_tr, X_ev_aug, seed=fseed + 404)
        if used_nl:
            X_tr_nl = np.column_stack([oof_nl, X_tr_aug])
            X_ev_nl = np.column_stack([pq_nl, X_ev_aug])
            y_tr_nl = y_tr - oof_nl
            views4_nl = _build_views(X_tr_nl, y_tr_nl, X_ev_nl, n_views=nv4, seed=fseed + 404)
            pnl_ev, c4 = _batched_icl_predict(all_four, views4_nl, y_tr_nl, dev, seed=fseed + 404)
            total_cells += c4
            pred_nl = np.clip(pq_nl + pnl_ev, lo_guard, hi_guard)
        else:
            pred_nl = pred_e32.copy()
        res["q_icl_nlres_e16"][va] = pred_nl[:n_va]
        res["bag_icl_nlres_e16"] += pred_nl[n_va:]

    for col in ("icl_robridge_a", "icl_robridge_b", "icl_robridge_e32", "icl_yj_exact_e16", "icl_nlres_e16"):
        res[f"bag_{col}"] /= float(max(valid_folds, 1))

    # --- Full-X_search Refits for Modes 1, 2, 3 ---
    rseed = _sha_seed(f"h450_gpu:{cid}:refit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs_cl, Xa_cl, Ss_full, Sa_full, s_meta = REL.build_h450_struct_cols(
            Xs, y, Xa, seed=rseed, fast_mode=False, max_total_cols=18, include_base_struct=True
        )
    Xs_aug = np.hstack([Xs_cl, Ss_full]) if Ss_full.shape[1] > 0 else Xs_cl
    Xa_aug = np.hstack([Xa_cl, Sa_full]) if Sa_full.shape[1] > 0 else Xa_cl

    oof_rr_f, pq_rr_f, used_rr_f, _ = RR.robust_ridge_np(Xs_aug, y, Xa_aug, seed=rseed + 11)
    y_rr_f = (y - oof_rr_f) if used_rr_f else y
    base_rr_f = pq_rr_f if used_rr_f else np.zeros(n_a, dtype=np.float64)

    v8a_f = _build_views(Xs_aug, y_rr_f, Xa_aug, n_views=nv8, seed=rseed + 101)
    v8b_f = _build_views(Xs_aug, y_rr_f, Xa_aug, n_views=nv8, seed=rseed + 202)
    pa_ref, c5 = _batched_icl_predict(pair_a, v8a_f, y_rr_f, dev, seed=rseed + 101)
    pb_ref, c6 = _batched_icl_predict(pair_b, v8b_f, y_rr_f, dev, seed=rseed + 202)
    total_cells += c5 + c6
    res["ref_icl_robridge_a"] = np.clip(base_rr_f + pa_ref, lo_guard, hi_guard)
    res["ref_icl_robridge_b"] = np.clip(base_rr_f + pb_ref, lo_guard, hi_guard)
    res["ref_icl_robridge_e32"] = 0.5 * (res["ref_icl_robridge_a"] + res["ref_icl_robridge_b"])

    warp_f = RR.YeoJohnsonWarpF64()
    z_f = warp_f.fit_transform(y)
    v4yj_f = _build_views(Xs_aug, z_f, Xa_aug, n_views=nv4, seed=rseed + 303)
    pyj_ref, c7 = _batched_icl_predict(all_four, v4yj_f, z_f, dev, seed=rseed + 303, inv_fn=warp_f.inv_torch)
    total_cells += c7
    res["ref_icl_yj_exact_e16"] = np.clip(pyj_ref, lo_guard, hi_guard)

    oof_nl_f, pq_nl_f, used_nl_f, _ = RR.fast_nonlinear_base_oof(Xs_aug, y, Xa_aug, seed=rseed + 404)
    if used_nl_f:
        Xs_nl_f = np.column_stack([oof_nl_f, Xs_aug])
        Xa_nl_f = np.column_stack([pq_nl_f, Xa_aug])
        v4nl_f = _build_views(Xs_nl_f, y - oof_nl_f, Xa_nl_f, n_views=nv4, seed=rseed + 404)
        pnl_ref, c8 = _batched_icl_predict(all_four, v4nl_f, y - oof_nl_f, dev, seed=rseed + 404)
        total_cells += c8
        res["ref_icl_nlres_e16"] = np.clip(pq_nl_f + pnl_ref, lo_guard, hi_guard)
    else:
        res["ref_icl_nlres_e16"] = res["ref_icl_robridge_e32"].copy()

    # --- Optional Hurdle-Sentinel Calibration on `icl_robridge_e32` (Strictly 5-fold OOF gated!) ---
    hurdle_used = False
    atom = RR.HurdleSentinelDetector.detect_atom(y)
    if atom is not None:
        is_atom = np.abs(y - atom) <= 1e-7 * max(1.0, abs(atom))
        if int(np.sum(is_atom)) >= 4 and int(np.sum(~is_atom)) >= 20:
            p_oof, p_qa = RR.HurdleSentinelDetector.fit_oof_and_query_probs(Xs_aug, is_atom, Xa_aug, lab, seed=rseed + 555)
            # Bulk-only 5-fold + refit `ICLH450` prediction
            bulk_q = res["q_icl_robridge_e32"].copy()
            bulk_bag = np.zeros(n_a, dtype=np.float64)
            bf_cnt = 0
            for fk, (tr, va, X_tr_aug, X_ev_aug) in fold_cache.items():
                n_va = int(np.sum(va))
                b_mask = ~is_atom[tr]
                if int(np.sum(b_mask)) < 16:
                    continue
                bf_cnt += 1
                Xb_tr, yb_tr = X_tr_aug[b_mask], y[tr][b_mask]
                oof_b, pq_b, u_b, _ = RR.robust_ridge_np(Xb_tr, yb_tr, X_ev_aug, seed=rseed + 600 + fk)
                yb_res = (yb_tr - oof_b) if u_b else yb_tr
                vb = _build_views(Xb_tr, yb_res, X_ev_aug, n_views=nv4, seed=rseed + 600 + fk)
                pb_ev2, cb2 = _batched_icl_predict(all_four, vb, yb_res, dev, seed=rseed + 600 + fk)
                total_cells += cb2
                pred_b_all = (pq_b if u_b else 0.0) + pb_ev2
                bulk_q[va] = pred_b_all[:n_va]
                bulk_bag += pred_b_all[n_va:]
            if bf_cnt > 0:
                bulk_bag /= float(bf_cnt)
                cand_q = p_oof * atom + (1.0 - p_oof) * bulk_q
                mse_base = float(np.mean((res["q_icl_robridge_e32"] - y) ** 2))
                mse_hurd = float(np.mean((cand_q - y) ** 2))
                if mse_hurd < 0.96 * mse_base:
                    hurdle_used = True
                    b_all = ~is_atom
                    Xb_f, yb_f = Xs_aug[b_all], y[b_all]
                    oof_bf, pq_bf, u_bf, _ = RR.robust_ridge_np(Xb_f, yb_f, Xa_aug, seed=rseed + 699)
                    yb_fres = (yb_f - oof_bf) if u_bf else yb_f
                    vbf = _build_views(Xb_f, yb_fres, Xa_aug, n_views=nv4, seed=rseed + 699)
                    pb_ref2, cb3 = _batched_icl_predict(all_four, vbf, yb_fres, dev, seed=rseed + 699)
                    total_cells += cb3
                    bulk_ref = (pq_bf if u_bf else 0.0) + pb_ref2
                    res["q_icl_robridge_e32"] = np.clip(cand_q, lo_guard, hi_guard)
                    res["bag_icl_robridge_e32"] = np.clip(p_qa * atom + (1.0 - p_qa) * bulk_bag, lo_guard, hi_guard)
                    res["ref_icl_robridge_e32"] = np.clip(p_qa * atom + (1.0 - p_qa) * bulk_ref, lo_guard, hi_guard)

    # --- Mode 4: 2-Stage Self-Boosted `ICLH450` (`icl_boost2s_e16`) ---
    q_s1 = res["q_icl_robridge_e32"]
    bag_s1 = res["bag_icl_robridge_e32"]
    ref_s1 = res["ref_icl_robridge_e32"]
    r1_s = y - q_s1
    rmse_s1 = float(np.sqrt(np.mean(r1_s * r1_s)))
    eta_best = 0.0
    if rmse_s1 < 0.55 * sy_glob and float(np.std(r1_s)) > 1e-6 * sy_glob and len(fold_cache) >= 4:
        r2_oof = np.zeros(n_s, dtype=np.float64)
        r2_bag = np.zeros(n_a, dtype=np.float64)
        cnt_s2 = 0
        for fk, (tr, va, X_tr_aug, X_ev_aug) in fold_cache.items():
            n_va = int(np.sum(va))
            r1_tr = r1_s[tr]
            if float(np.std(r1_tr)) <= 1e-12:
                continue
            cnt_s2 += 1
            s1_col_tr = q_s1[tr]
            s1_col_ev = np.concatenate([q_s1[va], bag_s1])
            X2_tr = np.column_stack([s1_col_tr, X_tr_aug])
            X2_ev = np.column_stack([s1_col_ev, X_ev_aug])
            v2 = _build_views(X2_tr, r1_tr, X2_ev, n_views=nv4, seed=rseed + 800 + fk)
            pr2_ev, cs2 = _batched_icl_predict(all_four, v2, r1_tr, dev, seed=rseed + 800 + fk)
            total_cells += cs2
            r2_oof[va] = pr2_ev[:n_va]
            r2_bag += pr2_ev[n_va:]
        if cnt_s2 > 0:
            r2_bag /= float(cnt_s2)
            best_mse2 = rmse_s1 * rmse_s1
            for eta in (0.20, 0.40, 0.65, 0.85, 1.00):
                mse_cand = float(np.mean((q_s1 + eta * r2_oof - y) ** 2))
                if mse_cand < 0.992 * best_mse2:
                    best_mse2 = mse_cand
                    eta_best = eta
            if eta_best > 0.0:
                X2_f_tr = np.column_stack([q_s1, Xs_aug])
                X2_f_ev = np.column_stack([ref_s1, Xa_aug])
                v2_f = _build_views(X2_f_tr, r1_s, X2_f_ev, n_views=nv4, seed=rseed + 899)
                r2_ref, cs2f = _batched_icl_predict(all_four, v2_f, r1_s, dev, seed=rseed + 899)
                total_cells += cs2f
                res["q_icl_boost2s_e16"] = np.clip(q_s1 + eta_best * r2_oof, lo_guard, hi_guard)
                res["bag_icl_boost2s_e16"] = np.clip(bag_s1 + eta_best * r2_bag, lo_guard, hi_guard)
                res["ref_icl_boost2s_e16"] = np.clip(ref_s1 + eta_best * r2_ref, lo_guard, hi_guard)
            else:
                res["q_icl_boost2s_e16"] = q_s1.copy()
                res["bag_icl_boost2s_e16"] = bag_s1.copy()
                res["ref_icl_boost2s_e16"] = ref_s1.copy()
    else:
        res["q_icl_boost2s_e16"] = q_s1.copy()
        res["bag_icl_boost2s_e16"] = bag_s1.copy()
        res["ref_icl_boost2s_e16"] = ref_s1.copy()

    meta = {
        "cid": cid,
        "wall_s": round(time.time() - t0, 3),
        "eval_cell_tokens": total_cells,
        "hurdle_used": hurdle_used,
        "boost2s_eta": eta_best,
        "struct_meta": s_meta,
    }
    return res, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, help="Path to bank_d91_search_f64.npz (0 audit targets!)")
    ap.add_argument("--ckpt-la", required=True)
    ap.add_argument("--ckpt-ma", required=True)
    ap.add_argument("--ckpt-lb", default="")
    ap.add_argument("--ckpt-mb", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    m_la = _load_icl_h450(a.ckpt_la, "L", dev)
    m_ma = _load_icl_h450(a.ckpt_ma, "M", dev)
    pair_a = [(m_la, 0.65), (m_ma, 0.35)]
    if a.ckpt_lb and a.ckpt_mb:
        m_lb = _load_icl_h450(a.ckpt_lb, "L", dev)
        m_mb = _load_icl_h450(a.ckpt_mb, "M", dev)
        pair_b = [(m_lb, 0.65), (m_mb, 0.35)]
    else:
        pair_b = pair_a

    rows = []
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
            rows.append((i, c, Xs, Xa, y, lab))
            so += n * w
            ao += na * f
            lo += n

    shard_rows = [r for r in rows if (r[0] % a.num_shards) == a.shard_idx]
    tot_cells = 0
    t0 = time.time()
    metas = []
    for _, c, Xs, Xa, y, lab in shard_rows:
        dst = out_dir / f"{c}.npz"
        if dst.exists():
            continue
        col_dict, meta = eval_single_context_gpu(c, Xs, Xa, y, lab, pair_a, pair_b, dev)
        tot_cells += int(meta["eval_cell_tokens"])
        metas.append(meta)
        tmp = out_dir / f".{c}.{os.getpid()}.tmp.npz"
        np.savez(tmp, ok=np.array(True), **col_dict)
        os.chmod(tmp, 0o444)
        os.replace(tmp, dst)

    summary_p = out_dir / f"shard_{a.shard_idx}_of_{a.num_shards}.json"
    summary_p.write_text(
        json.dumps(
            {
                "shard_idx": a.shard_idx,
                "num_shards": a.num_shards,
                "contexts_done": len(metas),
                "eval_cell_tokens": tot_cells,
                "wall_s": round(time.time() - t0, 2),
                "per_context": metas,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"shard": a.shard_idx, "done": len(metas), "eval_cell_tokens": tot_cells, "wall_s": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
