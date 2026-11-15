#!/usr/bin/env python3
"""H462 P1 (`MEMBERS TO SATURATION`) `K = 1 -> 2 -> 6 -> 12 -> 18` GP Seed Scaling on `FITSEL`.

Implements `cl_mtc` (`15:08:14Z` P1) + `cl_fwriter` (`15:09Z` & `Correction 1` `213b5f84…`):
  1. Fixed comparator: `tabcv7_cvrep8_prec48_all` (`clean-88 = 2.253677617837636`, `all-106 = 2.4627244340885923`).
  2. Standalone GP seed-bag scaling (`K = 1, 2, 6, 12, 18, ...`), equal & `M6-PRECISION-AVG` at GP-seed level.
  3. Stacked inside `tabcv7` (`[m7, k3, f4, tab450, e8, gpbag]`):
       - `rep0` (`cvrep1`) across ALL `K in {1, 2, 6, 12, 18, ...}` for both `2col` `(eq, prec)` and
         `allcol` `(gp_s1..gp_sK, eq, prec)`.
       - Exact `cvrep8` (`48` precision-weighted members) for `K = 0` (comparator reproduction),
         `K = 6` (`2col`, `allcol`), `K = 12` (`allcol`), and `K_max` (`2col`, `allcol`, plus disjoint
         half-splits `halfA` and `halfB`).
  4. Exact `cl_fwriter` Correction 1 noise unit (`sd_f(u_A,f - u_B,f) / sqrt(2 * n_fam)` on `clean_88`),
     `H416` paired family sign-flip test + `lofo_min` vs `tabcv7_cvrep8_prec48_all`, and `clean15`/`strict13`
     secondary per-family readouts (`0` `dev91` opens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
for _p in (
    str(HERE.parent),
    str(HERE.parent / "deps"),
    "/home/ubuntu/h452-stage/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

from h462_apex_prism import guard_matern_core as GM  # noqa: E402
from h462_apex_prism.edge_certify import bag_predictions_precision_weighted_oof  # noqa: E402
from h462_apex_prism.eval_h462_k5_fair0 import (  # noqa: E402
    H430_K3,
    H433_F4,
    H450_TAB_COLS,
    _check_safe_path,
    _clip_f4,
    _clip_k3,
    _load_bank,
    _load_col_block_flex,
    _nested_oof_cv7_rep,
)
from h462_apex_prism.stack_guard_cv7_h452 import make_stratified_kfold, stack_guard_cv7_context  # noqa: E402

try:
    import meta_h416 as M  # type: ignore[import-not-found]
    import train_h416 as T16  # type: ignore[import-not-found]
except ImportError:
    M = None  # type: ignore[assignment]
    T16 = None  # type: ignore[assignment]


def _uncentered_error_corr(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> float:
    e1 = np.asarray(p1, np.float64) - np.asarray(y, np.float64)
    e2 = np.asarray(p2, np.float64) - np.asarray(y, np.float64)
    d = float(np.linalg.norm(e1) * np.linalg.norm(e2))
    return float(np.dot(e1, e2) / d) if d > 1e-12 else 1.0


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _load_gp_seeds_from_dirs(
    gpbag_paths: Sequence[Path], n_s: int, n_a: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    qs: List[np.ndarray] = []
    bags: List[np.ndarray] = []
    refs: List[np.ndarray] = []
    for p in gpbag_paths:
        with np.load(p, allow_pickle=False) as z:
            for k in range(1, 7):
                q = np.asarray(z[f"q_gp_s{k}"], np.float64)
                b = np.asarray(z[f"bag_gp_s{k}"], np.float64)
                r = np.asarray(z[f"ref_gp_s{k}"], np.float64)
                assert len(q) == n_s and len(b) == n_a and len(r) == n_a
                qs.append(q)
                bags.append(b)
                refs.append(r)
    return qs, bags, refs


def _build_gpbag_blocks_subset(
    y: np.ndarray,
    lab5: np.ndarray,
    q_sub: Sequence[np.ndarray],
    b_sub: Sequence[np.ndarray],
    r_sub: Sequence[np.ndarray],
    tau_shrink: float = 0.25,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns `(blk_2col, blk_allcol)` with strictly nested outer-fold `q_pr` (`H5`-compliant, matching `emit_gp_runbag6_h462.py`)."""
    if len(q_sub) == 1:
        q1 = q_sub[0][:, None]
        b1 = b_sub[0][:, None]
        r1 = r_sub[0][:, None]
        return (q1, b1, r1), (q1, b1, r1)
    Q_mat = np.column_stack(q_sub)
    B_mat = np.column_stack(b_sub)
    R_mat = np.column_stack(r_sub)
    q_eq = np.mean(Q_mat, axis=1)
    b_eq = np.mean(B_mat, axis=1)
    r_eq = np.mean(R_mat, axis=1)

    n_s, k_s = Q_mat.shape
    u_w = np.full(k_s, 1.0 / float(k_s), dtype=np.float64)
    eps = 1e-4 * max(float(np.var(y)), 1e-12)

    def _w_fit(Q_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
        mse = np.mean((Q_tr - y_tr[:, None]) ** 2, axis=0)
        prec = 1.0 / (mse + eps)
        sp = float(np.sum(prec))
        w_raw = (prec / sp) if (sp > 0.0 and np.isfinite(sp)) else u_w
        w = (1.0 - tau_shrink) * w_raw + tau_shrink * u_w
        return w / float(np.sum(w))

    q_pr = np.empty(n_s, dtype=np.float64)
    for fk in sorted(int(x) for x in np.unique(lab5)):
        va = lab5 == fk
        tr = ~va
        w_tr = _w_fit(Q_mat[tr], y[tr]) if int(np.sum(tr)) >= 10 else u_w
        q_pr[va] = Q_mat[va] @ w_tr

    w_full = _w_fit(Q_mat, y)
    b_pr = B_mat @ w_full
    r_pr = R_mat @ w_full
    blk_2 = (
        np.column_stack([q_eq, q_pr]),
        np.column_stack([b_eq, b_pr]),
        np.column_stack([r_eq, r_pr]),
    )
    blk_all = (
        np.column_stack([Q_mat, q_eq, q_pr]),
        np.column_stack([B_mat, b_eq, b_pr]),
        np.column_stack([R_mat, r_eq, r_pr]),
    )
    return blk_2, blk_all


def _run_tabcv7_prec(
    c: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    ya_ref: np.ndarray,
    ba_ref: float,
    labs_rep: Sequence[np.ndarray],
    cids_rep: Sequence[str],
    gp_packs: Sequence[Tuple],
    base_common_cv7: Sequence[Tuple],
    r4_tri: Sequence[Tuple],
    gblk: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_reps_use: int,
) -> float:
    preds, oofs = [], []
    for r4_blk in r4_tri:
        tb_cv7 = list(base_common_cv7) + ([("e8", r4_blk)] if r4_blk is not None else [])
        if gblk is not None:
            tb_cv7.append(("gpbag", gblk))
        for _gi, Q_gp, bcols, rcols in gp_packs:
            for r in range(n_reps_use):
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_tb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_cv7, use_cv7_folds=True)
                q_tb = _nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7)
                preds.append(np.asarray(p_tb, np.float64))
                oofs.append(q_tb)
    p_pr, _ = bag_predictions_precision_weighted_oof(y, oofs, preds)
    return float(M.utility(p_pr, ya_ref, ba_ref)[0])


def _eval_ctx_worker(job: Tuple) -> Dict:
    c, Xs, Xa, y, lab0, gp0_p, gp1_p, blk_m, blk_k, blk_f, blk_tab, r4_wc, r4_s1, r4_s2, gpbag_paths, k_rungs = job
    n_s, n_a = len(y), len(Xa)
    qs, bags, refs = _load_gp_seeds_from_dirs(gpbag_paths, n_s, n_a)

    gp_packs = []
    ya_ref: Optional[np.ndarray] = None
    ba_ref: float = 0.0
    for gi, gp_p in enumerate((gp0_p, gp1_p)):
        with np.load(gp_p, allow_pickle=False) as zg:
            Q_gp = np.asarray(zg["Q"], np.float64)
            ya = np.asarray(zg["ya"], np.float64)
            ba = float(zg["ba"])
            ym = float(zg["ym"])
            m_gp, na_z = Q_gp.shape[1], len(ya)
            mean_a = np.full(na_z, ym, dtype=np.float64)
            BAG_gp = np.asarray(zg["BAG"], np.float64)
            REF_gp = np.asarray(zg["REF"], np.float64)
            bcols = [BAG_gp[:, j] if np.all(np.isfinite(BAG_gp[:, j])) else mean_a for j in range(m_gp)]
            rcols = [REF_gp[:, j] if np.all(np.isfinite(REF_gp[:, j])) else None for j in range(m_gp)]
        if ya_ref is None:
            ya_ref = ya
            ba_ref = ba
        gp_packs.append((gi, Q_gp, bcols, rcols))
    assert ya_ref is not None

    # 1. Standalone GP seed-bag scaling (K = 1, 2, 6, 12, 18, ...)
    u_each = [float(M.utility(b, ya_ref, ba_ref)[0]) for b in bags]
    n_corr = min(len(bags), 18)
    rho_gp = float(
        np.mean([_uncentered_error_corr(bags[i], bags[j], ya_ref) for i in range(n_corr) for j in range(i + 1, n_corr)])
    )
    st_res: Dict[str, float] = {
        "u_gp_k1_mean": float(np.mean(u_each)),
        "u_gp_k1_s1": float(u_each[0]),
        "rho_err_gp_seeds": rho_gp,
    }

    all_k_steps = [1, 2] + [k for k in k_rungs if k > 2]
    blocks_2 = {}
    blocks_all = {}
    for k in all_k_steps:
        b2, ball = _build_gpbag_blocks_subset(y, lab0, qs[:k], bags[:k], refs[:k])
        blocks_2[k] = b2
        blocks_all[k] = ball
        if k >= 2:
            b_eq = np.mean(bags[:k], axis=0)
            b_pr, _ = bag_predictions_precision_weighted_oof(y, qs[:k], bags[:k])
            st_res[f"u_gp_k{k}_eq"] = float(M.utility(b_eq, ya_ref, ba_ref)[0])
            st_res[f"u_gp_k{k}_prec"] = float(M.utility(b_pr, ya_ref, ba_ref)[0])

    # 2. TabCV7 stack evaluation (`rep0` across all K; exact `cvrep8` on comparator + key rungs + K_max halves)
    labs_rep = [lab0 if r == 0 else make_stratified_kfold(c, y, k=5, rep=r) for r in range(8)]
    cids_rep = [c if r == 0 else f"{c}#cvrep{r}" for r in range(8)]
    base_common_cv7 = [(t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None]
    r4_tri = (r4_wc, r4_s1, r4_s2)

    sk_res: Dict[str, float] = {}
    # Baseline K=0 (rep0 and exact cvrep8 comparator `tabcv7_cvrep8_prec48_all`)
    sk_res["tabcv7_rep0_prec6_base"] = _run_tabcv7_prec(
        c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, None, 1
    )
    sk_res["tabcv7_cvrep8_prec48_all"] = _run_tabcv7_prec(
        c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, None, 8
    )

    for k in all_k_steps:
        if k == 1:
            sk_res["tabcv7_rep0_gpbag_k1"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, blocks_all[1], 1
            )
        else:
            sk_res[f"tabcv7_rep0_gpbag_k{k}_2col"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, blocks_2[k], 1
            )
            sk_res[f"tabcv7_rep0_gpbag_k{k}_allcol"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, blocks_all[k], 1
            )

    # Disjoint half-splits (Correction 1 §2) on rep0 for every K >= 6
    for k in k_rungs:
        if k >= 4:
            kh = k // 2
            b2_A, ball_A = _build_gpbag_blocks_subset(y, lab0, qs[:kh], bags[:kh], refs[:kh])
            b2_B, ball_B = _build_gpbag_blocks_subset(y, lab0, qs[kh:k], bags[kh:k], refs[kh:k])
            sk_res[f"tabcv7_rep0_gpbag_k{k}_2col_halfA"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, b2_A, 1
            )
            sk_res[f"tabcv7_rep0_gpbag_k{k}_2col_halfB"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, b2_B, 1
            )
            sk_res[f"tabcv7_rep0_gpbag_k{k}_allcol_halfA"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, ball_A, 1
            )
            sk_res[f"tabcv7_rep0_gpbag_k{k}_allcol_halfB"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, ball_B, 1
            )

    # Exact `cvrep8` (`48` members) on K=2 (`Correction 2/3` comparator!) and every 6-seed rung + K_max disjoint halves
    k_max = k_rungs[-1]
    for k_cv8 in sorted({k for k in ([2] + list(k_rungs)) if k in blocks_all}):
        sk_res[f"tabcv7_cvrep8_gpbag_k{k_cv8}_allcol"] = _run_tabcv7_prec(
            c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, blocks_all[k_cv8], 8
        )
        if k_cv8 in (2, 6, k_max):
            sk_res[f"tabcv7_cvrep8_gpbag_k{k_cv8}_2col"] = _run_tabcv7_prec(
                c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, blocks_2[k_cv8], 8
            )

    kh_max = k_max // 2
    b2_mA, ball_mA = _build_gpbag_blocks_subset(y, lab0, qs[:kh_max], bags[:kh_max], refs[:kh_max])
    b2_mB, ball_mB = _build_gpbag_blocks_subset(y, lab0, qs[kh_max:k_max], bags[kh_max:k_max], refs[kh_max:k_max])
    sk_res[f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_halfA"] = _run_tabcv7_prec(
        c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, ball_mA, 8
    )
    sk_res[f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_halfB"] = _run_tabcv7_prec(
        c, Xs, Xa, y, ya_ref, ba_ref, labs_rep, cids_rep, gp_packs, base_common_cv7, r4_tri, ball_mB, 8
    )

    return {"cid": c, "standalone": st_res, "stack": sk_res}


def _compute_correction1_noise_unit(
    H, umap_A: Dict[str, float], umap_B: Dict[str, float], cids: List[str], fam_map: Dict[str, str]
) -> dict:
    fk = {c: fam_map[c] for c in cids}
    fams = sorted(set(fk.values()))
    n_fam = len(fams)
    d_fam = np.asarray(
        [np.mean([umap_A[c] - umap_B[c] for c in cids if fk[c] == fm]) for fm in fams],
        dtype=np.float64,
    )
    sd_d = float(np.std(d_fam, ddof=1))
    unit = sd_d / float(np.sqrt(2.0 * n_fam))
    return {
        "n_fam": n_fam,
        "family_mean_u_half_A": float(H.family_mean(umap_A, fk)),
        "family_mean_u_half_B": float(H.family_mean(umap_B, fk)),
        "sd_f_d": sd_d,
        "noise_unit": unit,
    }


def _summarize(H, done: List[Dict], cids: List[str], fam_map: Dict[str, str], k_rungs: Sequence[int], include_per_fam_kmax: bool = False) -> dict:
    fk = {c: fam_map[c] for c in cids}
    st_by = {r["cid"]: r["standalone"] for r in done}
    sk_by = {r["cid"]: r["stack"] for r in done}

    st_keys = [k for k in sorted(next(iter(st_by.values())).keys()) if k != "rho_err_gp_seeds"]
    st_umaps = {k: {c: float(st_by[c][k]) for c in cids} for k in st_keys}
    sk_keys = sorted(next(iter(sk_by.values())).keys())
    sk_umaps = {k: {c: float(sk_by[c][k]) for c in cids} for k in sk_keys}

    hwm_col = "tabcv7_cvrep8_prec48_all"
    k2_all_col = "tabcv7_cvrep8_gpbag_k2_allcol"
    k2_2c_col = "tabcv7_cvrep8_gpbag_k2_2col"
    k_max = k_rungs[-1]

    # Paired tests vs fixed HWM comparator `tabcv7_cvrep8_prec48_all` AND vs `Correction 2` K=2 distinct-seed comparator
    vs_hwm = {}
    vs_k2 = {}
    for k_cv8 in sorted({k for k in ([2] + list(k_rungs)) if f"tabcv7_cvrep8_gpbag_k{k}_allcol" in sk_umaps}):
        ak = f"tabcv7_cvrep8_gpbag_k{k_cv8}_allcol"
        vs_hwm[f"{ak}_vs_hwm"] = T16.pair_stats(np, H, sk_umaps[ak], sk_umaps[hwm_col], fk)
        if k_cv8 > 2:
            vs_k2[f"{ak}_vs_k2_allcol"] = T16.pair_stats(np, H, sk_umaps[ak], sk_umaps[k2_all_col], fk)
        tk = f"tabcv7_cvrep8_gpbag_k{k_cv8}_2col"
        if tk in sk_umaps:
            vs_hwm[f"{tk}_vs_hwm"] = T16.pair_stats(np, H, sk_umaps[tk], sk_umaps[hwm_col], fk)
            if k_cv8 > 2:
                vs_k2[f"{tk}_vs_k2_2col"] = T16.pair_stats(np, H, sk_umaps[tk], sk_umaps[k2_2c_col], fk)

    # Incremental paired tests along rep0 ladder K = 0 -> 1 -> 2 -> 6 -> 12 -> 18
    rep0_steps = {}
    all_k = [1, 2] + [k for k in k_rungs if k > 2]
    prev_all = "tabcv7_rep0_prec6_base"
    for k in all_k:
        cur_all = "tabcv7_rep0_gpbag_k1" if k == 1 else f"tabcv7_rep0_gpbag_k{k}_allcol"
        rep0_steps[f"allcol_k{k}_vs_prev"] = T16.pair_stats(np, H, sk_umaps[cur_all], sk_umaps[prev_all], fk)
        prev_all = cur_all

    # Correction 1 noise units
    noise_units = {}
    for k in k_rungs:
        if k >= 4:
            noise_units[f"rep0_k{k}_allcol"] = _compute_correction1_noise_unit(
                H,
                sk_umaps[f"tabcv7_rep0_gpbag_k{k}_allcol_halfA"],
                sk_umaps[f"tabcv7_rep0_gpbag_k{k}_allcol_halfB"],
                cids,
                fam_map,
            )
            noise_units[f"rep0_k{k}_2col"] = _compute_correction1_noise_unit(
                H,
                sk_umaps[f"tabcv7_rep0_gpbag_k{k}_2col_halfA"],
                sk_umaps[f"tabcv7_rep0_gpbag_k{k}_2col_halfB"],
                cids,
                fam_map,
            )
    noise_units[f"cvrep8_k{k_max}_allcol"] = _compute_correction1_noise_unit(
        H,
        sk_umaps[f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_halfA"],
        sk_umaps[f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_halfB"],
        cids,
        fam_map,
    )

    out = {
        "n_contexts": len(cids),
        "n_families": len(set(fk.values())),
        "standalone_gp_scaling": {
            "family_mean_u": {k: float(H.family_mean(st_umaps[k], fk)) for k in st_keys},
            "mean_error_rho_gp_seeds": float(np.mean([st_by[c]["rho_err_gp_seeds"] for c in cids])),
        },
        "tabcv7_stack_scaling": {
            "family_mean_u": {k: float(H.family_mean(sk_umaps[k], fk)) for k in sk_keys},
            "paired_vs_k2_distinct_comparator_correction2": vs_k2,
            "paired_vs_locked_hwm_tabcv7_cvrep8_prec48_all": vs_hwm,
            "rep0_incremental_steps": rep0_steps,
            "correction1_noise_units": noise_units,
        },
    }
    if include_per_fam_kmax:
        cand_key = f"tabcv7_cvrep8_gpbag_k{k_max}_allcol"
        out["per_family_kmax_allcol_vs_hwm"] = [
            {
                "cid": c,
                "family_key": fam_map[c],
                f"u_{cand_key}": round(sk_umaps[cand_key][c], 6),
                "u_tabcv7_cvrep8_prec48_all": round(sk_umaps[hwm_key][c], 6),
                "delta_vs_hwm": round(sk_umaps[cand_key][c] - sk_umaps[hwm_key][c], 6),
            }
            for c in cids
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--gpbag-dirs", required=True, help="Comma-separated 6-seed GPBAG FITSEL dirs")
    ap.add_argument("--wc-overlap-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ctx-cache-dir", default="", help="Optional per-context JSON cache dir for resume")
    ap.add_argument("--valley-mode", default="v12_avg")
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass

    t0 = time.monotonic()
    root = Path(a.bundle_root)
    out_p = Path(a.out_json)
    gpbag_dirs = [Path(x.strip()) for x in a.gpbag_dirs.split(",") if x.strip()]
    for p in [root, out_p] + gpbag_dirs:
        _check_safe_path(p)
    if out_p.exists():
        raise SystemExit(f"REFUSAL: create-once receipt already exists: {out_p}")

    k_rungs = [6 * (i + 1) for i in range(len(gpbag_dirs))]
    _, H = T16._stats_mod()
    fs_in = root / "fitsel_inputs"
    cols_fs = root / "cols_fitsel"
    ex = json.loads((fs_in / "EXCLUSION_FITSEL.json").read_text())
    fam_map = {c: str(ex["contexts"][c]["family_key"]) for c in ex["contexts"]}
    all_106 = sorted(fam_map)
    clean_88 = sorted(c for c, info in ex["contexts"].items() if info.get("tier") != "copy")
    bank_rows = _load_bank(fs_in / "bank_FITSEL.npz")
    vmode = str(a.valley_mode)

    jobs = []
    for c in all_106:
        Xs, Xa, y, lab0 = bank_rows[c]
        n_s, n_a = len(y), len(Xa)
        blk_m = _load_col_block_flex(fs_in / "cols_matern_FITSEL" / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
        blk_k = _clip_k3(_load_col_block_flex(fs_in / "cols_k3_FITSEL" / f"{c}.npz", H430_K3, n_s, n_a), y)
        blk_f = _clip_f4(_load_col_block_flex(fs_in / "cols_f4_FITSEL" / f"{c}.npz", H433_F4, n_s, n_a), y)
        blk_tab = _load_col_block_flex(cols_fs / "cols_tab_h450_FITSEL" / f"{c}.npz", H450_TAB_COLS, n_s, n_a)
        r4_wc = _load_col_block_flex(fs_in / "cols_r4e8_wc_FITSEL" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s1 = _load_col_block_flex(cols_fs / f"r4e8_s46201_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s2 = _load_col_block_flex(cols_fs / f"r4e8_s46202_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        gp_paths = [gd / f"{c}.npz" for gd in gpbag_dirs]
        jobs.append(
            (
                c,
                Xs,
                Xa,
                y,
                lab0,
                fs_in / "dev1_gp" / f"{c}.npz",
                fs_in / "dev2_gp" / f"{c}.npz",
                blk_m,
                blk_k,
                blk_f,
                blk_tab,
                r4_wc,
                r4_s1,
                r4_s2,
                gp_paths,
                k_rungs,
            )
        )

    jobs.sort(key=lambda j: -len(j[3]))
    cache_dir_s = getattr(a, "ctx_cache_dir", "").strip()
    cache_dir = Path(cache_dir_s) if cache_dir_s else None
    if cache_dir is not None:
        _check_safe_path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    cached_done: List[Dict] = []
    pending_jobs = []
    for jb in jobs:
        cid_j = str(jb[0])
        cp_j = (cache_dir / f"{cid_j}.json") if cache_dir is not None else None
        if cp_j is not None and cp_j.is_file():
            cached_done.append(json.loads(cp_j.read_text()))
        else:
            pending_jobs.append(jb)
    done = list(cached_done)
    if pending_jobs:
        with Pool(processes=min(max(1, a.procs), len(pending_jobs))) as pool:
            for item in pool.imap_unordered(_eval_ctx_worker, pending_jobs, chunksize=1):
                if cache_dir is not None:
                    tmp_c = cache_dir / f".{item['cid']}.{os.getpid()}.tmp.json"
                    tmp_c.write_text(json.dumps(item, sort_keys=True) + "\n")
                    os.replace(tmp_c, cache_dir / f"{item['cid']}.json")
                done.append(item)

    s88 = _summarize(H, done, clean_88, fam_map, k_rungs, include_per_fam_kmax=False)
    s106 = _summarize(H, done, all_106, fam_map, k_rungs, include_per_fam_kmax=False)
    ov_doc = json.loads(Path(a.wc_overlap_json).read_text())
    c15 = [c for c in ov_doc["excl_wc80_clean15_cids"] if c in fam_map]
    c13 = [c for c in ov_doc["excl_wc80_strict13_cids"] if c in fam_map]
    s15 = _summarize(H, done, c15, fam_map, k_rungs, include_per_fam_kmax=True)
    s13 = _summarize(H, done, c13, fam_map, k_rungs, include_per_fam_kmax=True)

    k_max = k_rungs[-1]
    u_hwm = s88["tabcv7_stack_scaling"]["family_mean_u"]["tabcv7_cvrep8_prec48_all"]
    u_k2_all = s88["tabcv7_stack_scaling"]["family_mean_u"]["tabcv7_cvrep8_gpbag_k2_allcol"]
    u_cand = s88["tabcv7_stack_scaling"]["family_mean_u"][f"tabcv7_cvrep8_gpbag_k{k_max}_allcol"]
    ps_cand_hwm = s88["tabcv7_stack_scaling"]["paired_vs_locked_hwm_tabcv7_cvrep8_prec48_all"][
        f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_vs_hwm"
    ]
    ps_cand_k2 = s88["tabcv7_stack_scaling"]["paired_vs_k2_distinct_comparator_correction2"][
        f"tabcv7_cvrep8_gpbag_k{k_max}_allcol_vs_k2_allcol"
    ]
    nu_cand = s88["tabcv7_stack_scaling"]["correction1_noise_units"][f"cvrep8_k{k_max}_allcol"]
    gain_vs_k2 = float(u_cand - u_k2_all)
    gain_vs_hwm = float(u_cand - u_hwm)
    stop_pass_vs_k2 = bool(gain_vs_k2 >= nu_cand["noise_unit"] and float(ps_cand_k2["lofo_min"]) > 0.0)
    stop_pass_vs_hwm = bool(gain_vs_hwm >= nu_cand["noise_unit"] and float(ps_cand_hwm["lofo_min"]) > 0.0)

    doc = {
        "schema": "jope.h462.p1_gpbag_scaling_fitsel/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "dev91_opens": 0,
        "comparator": {
            "hwm_descriptive_rule": "tabcv7_cvrep8_prec48_all",
            "clean88_u_hwm_reproduced": u_hwm,
            "clean88_u_hwm_reference": 2.253677617837636,
            "correction2_k2_comparator_rule": "tabcv7_cvrep8_gpbag_k2_allcol",
            "clean88_u_k2_allcol": u_k2_all,
        },
        "k_rungs": k_rungs,
        "k_max_summary_clean88": {
            "candidate_rule": f"tabcv7_cvrep8_gpbag_k{k_max}_allcol",
            "clean88_u": u_cand,
            "clean88_u_k2_comparator": u_k2_all,
            "gain_vs_k2_comparator": gain_vs_k2,
            "gain_vs_k2_in_noise_units": gain_vs_k2 / max(nu_cand["noise_unit"], 1e-12),
            "lofo_min_vs_k2": float(ps_cand_k2["lofo_min"]),
            "signflip_p_vs_k2": float(ps_cand_k2["signflip_p"]),
            "wtl_vs_k2": ps_cand_k2["wtl"],
            "stop_rule_continue_vs_k2_correction2": stop_pass_vs_k2,
            "gain_vs_hwm_descriptive": gain_vs_hwm,
            "lofo_min_vs_hwm": float(ps_cand_hwm["lofo_min"]),
            "signflip_p_vs_hwm": float(ps_cand_hwm["signflip_p"]),
            "wtl_vs_hwm": ps_cand_hwm["wtl"],
            "stop_rule_continue_vs_hwm_descriptive": stop_pass_vs_hwm,
            "noise_unit_info": nu_cand,
        },
        "elapsed_s": round(time.monotonic() - t0, 2),
        "fitsel_clean_non_copy_88": s88,
        "fitsel_excl_wc80_clean15": s15,
        "fitsel_excl_wc80_strict13": s13,
        "fitsel_all_106": s106,
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out_p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "OK",
                "k_rungs": k_rungs,
                "k_max_summary_clean88": doc["k_max_summary_clean88"],
                "clean88_tabcv7_family_mean_u": s88["tabcv7_stack_scaling"]["family_mean_u"],
                "out_sha256": _sha256_file(out_p),
                "elapsed_s": doc["elapsed_s"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
