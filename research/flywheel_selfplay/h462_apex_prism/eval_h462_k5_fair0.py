#!/usr/bin/env python3
"""H462 `0.000000% dev91` `K=5` Neural Valley Seeds (`wc_M + s46201..s46204`) + Clean `cv7_hyb_fair0` + `CVREP4` (`eval_h462_k5_fair0.py`).

Tests `ENG:watcher`'s (`12:36Z` & `13:10Z`) `K=2 -> K=4 -> K=5` `0.000000% dev91` neural valley seed scaling AND
re-prices `cv7_hyb` with `gp_anchor_dir=""` (`cv7_hyb_fair0_*`, zero `gp_anchors_p0` anywhere in the transitive input
closure!) + `CVREP4` (`rep in 0..3`):
  - Step A (`--mode select-fitsel`, `0` `dev91` opens): Evaluates `K in {1, 2, 4, 5}` across `gm2l_r4e8`, `tabcv7_hyb0`,
    and cross-stacker `dual_hyb0 = {gm2l_r4e8, tabcv7_hyb0}` (`R in {1, 4}`, `eq` vs `prec_avg`) on `FITSEL`
    (`clean_non_copy_88` primary, `all_106` descriptive), writing create-once `FITSEL_LOCK_K5_FAIR0.json` (`0444`).
  - Step B (`--mode price-d91`, single frozen pass): Verifies `FITSEL_LOCK_K5_FAIR0.json` and prices once on
    quarantined `board_post_h448` (`91` contexts, `19` families, `0.000000% dev91`).
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

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
from h462_apex_prism.eval_h462_tabcv7_cvrep import H430_K3, H433_F4, H450_TAB_COLS  # noqa: E402
from h462_apex_prism.stack_guard_cv7_h452 import (  # noqa: E402
    _fit_sharp_foldlog_simplex,
    bcar_weights_pool_agnostic,
    make_stratified_kfold,
    stack_guard_cv7_context,
)

try:
    import h335_stats as H  # type: ignore[import-not-found]
    import meta_h416 as M  # type: ignore[import-not-found]
    import train_h416 as T16  # type: ignore[import-not-found]
except ImportError:
    H = None  # type: ignore[assignment]
    M = None  # type: ignore[assignment]
    T16 = None  # type: ignore[assignment]

CV7_HYBRID_COLS = (
    "icl_e8_cw12_robridge_cv7",
    "icl_e8_cw12_yj_cv7",
    "icl_e8_rankgauss_duan_cv7",
    "icl_e8_gp_resid_cv7",
)

ANCHOR = T16.ANCHOR if T16 is not None else "h335.H257_REPLICA"
LADDER_SEED = T16.LADDER_SEED if T16 is not None else 335160
FAIR_CANONICAL = "h448.APEX_FAIR_CHAMPION"
KUMO_LARGE = "kumo_tabular_large_e8.v1"
TABFM = "tabfm_ensemble.v1"

FORBIDDEN_TOKENS = ("CONFIRM", "FMGAP_B", "FMGAP_C", "HOLDOUT", "SEALED", "PROBE_YA", "/tmp/", "gp_anchors_p0")


def _check_safe_path(p: Path) -> None:
    s = str(p)
    up = s.upper()
    for tok in FORBIDDEN_TOKENS:
        if tok.upper() in up:
            raise PermissionError(f"Forbidden path token {tok!r} in {s}")
    if "reference-board/boards/dev91-audit-v1" in s:
        raise PermissionError(f"Canonical reference-board dev91 path forbidden: {s}")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _load_col_block_flex(p: Path, names: Tuple[str, ...], n_s: int, n_a: int):
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        if "ok" in z.files and not bool(z["ok"]):
            return None
        if not all(f"q_{v}" in z.files and len(z[f"q_{v}"]) == n_s for v in names):
            return None
        Q = np.column_stack([np.asarray(z[f"q_{v}"], np.float64) for v in names])
        BAG = np.column_stack([np.asarray(z[f"bag_{v}"], np.float64) for v in names])
        REF = np.column_stack(
            [
                np.asarray(z[f"ref_{v}"], np.float64) if f"ref_{v}" in z.files else np.asarray(z[f"bag_{v}"], np.float64)
                for v in names
            ]
        )
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(BAG)) and np.all(np.isfinite(REF))):
        return None
    return Q, BAG, REF


def _clip_k3(blk_k_raw, y: np.ndarray):
    if blk_k_raw is None:
        return None
    Qk, Bk, Rk = [arr.copy() for arr in blk_k_raw]
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
    for A in (Qk, Bk, Rk):
        A[:, 0] = np.clip(A[:, 0], y_lo - y_pad, y_hi + y_pad)
    return Qk, Bk, Rk


def _clip_f4(blk_f_raw, y: np.ndarray):
    if blk_f_raw is None:
        return None
    Qf, Bf, Rf = [arr.copy() for arr in blk_f_raw]
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_pad = 0.50 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
    for A in (Qf, Bf, Rf):
        for idx_f in (2, 3):
            A[:, idx_f] = np.clip(A[:, idx_f], y_lo - y_pad, y_hi + y_pad)
    return Qf, Bf, Rf


def _nested_oof_gm2l_rep(
    cid_r: str,
    y: np.ndarray,
    lab_r: np.ndarray,
    Q_gp: np.ndarray,
    act_blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    n_s = Q_gp.shape[0]
    ym = float(np.mean(y))
    q_oof = np.full(n_s, ym, dtype=np.float64)
    Q_ext = np.hstack([b[0] for b in act_blocks]) if act_blocks else None
    Q_joint = np.hstack([Q_gp, Q_ext]) if Q_ext is not None else Q_gp
    m_tot = Q_joint.shape[1]
    for fk in sorted(int(k) for k in np.unique(lab_r)):
        va = lab_r == fk
        tr = ~va
        if int(np.sum(va)) == 0 or int(np.sum(tr)) < 15:
            continue
        lab_in = make_stratified_kfold(f"{cid_r}_gm2l_nf{fk}", y[tr], k=5, rep=0)
        b_va = [np.asarray(Q_joint[va, j], np.float64) for j in range(m_tot)]
        _, _, _, kept = GM.guard_candidate_pool_ya_free(y[tr], lab_in, Q_joint[tr], b_va, [None] * m_tot)
        if len(kept) == 0:
            kept = np.array([int(np.argmin(np.mean((Q_joint[tr] - y[tr, None]) ** 2, axis=0)))], dtype=np.int64)
        Q_tr_k = Q_joint[np.ix_(tr, kept)]
        Q_va_k = Q_joint[np.ix_(va, kept)]
        w = GM._prune(
            0.50 * GM.bcar_weights(Q_tr_k, y[tr], f"{cid_r}_nf{fk}", "fold_log", lab_in)
            + 0.35 * GM.bcar_weights(Q_tr_k, y[tr], f"{cid_r}_nf{fk}", "huber", lab_in)
            + 0.15 * GM.bcar_weights(Q_tr_k, y[tr], f"{cid_r}_nf{fk}", "br_mse", lab_in)
        )
        sw = float(np.sum(w))
        q_oof[va] = (Q_va_k @ (w / sw)) if sw > 1e-12 else Q_va_k[:, 0]
    return q_oof


def _nested_oof_cv7_rep(
    cid_r: str,
    y: np.ndarray,
    lab_r: np.ndarray,
    Q_gp: np.ndarray,
    tagged_blocks: Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> np.ndarray:
    n_s = Q_gp.shape[0]
    ym = float(np.mean(y))
    q_oof = np.full(n_s, ym, dtype=np.float64)
    Q_ext = [b[0] for _, b in tagged_blocks if b is not None]
    Q_joint = np.hstack([Q_gp] + Q_ext) if Q_ext else Q_gp
    m_tot = Q_joint.shape[1]
    for fk in sorted(int(k) for k in np.unique(lab_r)):
        va = lab_r == fk
        tr = ~va
        if int(np.sum(va)) == 0 or int(np.sum(tr)) < 15:
            continue
        lab_in = (
            make_stratified_kfold(f"{cid_r}_cv7_nf{fk}", y[tr], k=7, rep=0)
            if int(np.sum(tr)) >= 35
            else make_stratified_kfold(f"{cid_r}_cv7_nf{fk}", y[tr], k=5, rep=0)
        )
        b_va = [np.asarray(Q_joint[va, j], np.float64) for j in range(m_tot)]
        _, _, _, kept = GM.guard_candidate_pool_ya_free(y[tr], lab_in, Q_joint[tr], b_va, [None] * m_tot)
        if len(kept) == 0:
            kept = np.array([int(np.argmin(np.mean((Q_joint[tr] - y[tr, None]) ** 2, axis=0)))], dtype=np.int64)
        Q_tr_k = Q_joint[np.ix_(tr, kept)]
        Q_va_k = Q_joint[np.ix_(va, kept)]
        mses = np.mean((Q_tr_k - y[tr, None]) ** 2, axis=0)
        vy = max(float(np.var(y[tr])), 1e-24)
        idx2 = np.argsort(mses)[: min(2, len(mses))]
        u_lead = (
            float(-0.5 * np.log(max(float(mses[idx2[0]]), 1e-30) / vy) - (-0.5 * np.log(max(float(mses[idx2[-1]]), 1e-30) / vy)))
            if len(idx2) == 2
            else 0.0
        )
        if u_lead >= 0.25 and len(kept) >= 2:
            w = _fit_sharp_foldlog_simplex(Q_tr_k, y[tr], lab_in, top_k=8)
        else:
            w = GM._prune(
                0.50 * bcar_weights_pool_agnostic(Q_tr_k, y[tr], f"{cid_r}_cv7_nf{fk}", "fold_log", lab_in, n_bags=16)
                + 0.35 * bcar_weights_pool_agnostic(Q_tr_k, y[tr], f"{cid_r}_cv7_nf{fk}", "huber", lab_in, n_bags=16)
                + 0.15 * bcar_weights_pool_agnostic(Q_tr_k, y[tr], f"{cid_r}_cv7_nf{fk}", "br_mse", lab_in, n_bags=16)
            )
        sw = float(np.sum(w))
        q_oof[va] = (Q_va_k @ (w / sw)) if sw > 1e-12 else Q_va_k[:, 0]
    return q_oof


RULE_KEYS = (
    "n0_mean_conv",
    "pred_avg_2_n0",
    # R4E8 gm2l K=2 -> K=4 -> K=5 + CVREP4
    "gm2l_k2_rep0_eq4",
    "gm2l_k4_rep0_eq8",
    "gm2l_k4_rep0_prec8",
    "gm2l_k5_rep0_prec10",
    "gm2l_k2_cvrep4_eq16",
    "gm2l_k4_cvrep4_eq32",
    "gm2l_k4_cvrep4_prec32",
    "gm2l_k5_cvrep4_prec40",
    # Clean 0% dev91 cv7_hyb_fair0 + tab450 K=1 -> K=2 -> K=4 -> K=5 + CVREP4
    "hyb0_wcm_rep0_eq2",
    "hyb0_k2_rep0_eq4",
    "hyb0_k2_rep0_prec4",
    "hyb0_k4_rep0_eq8",
    "hyb0_k4_rep0_prec8",
    "hyb0_k5_rep0_eq10",
    "hyb0_k5_rep0_prec10",
    "hyb0_k4_cvrep4_eq32",
    "hyb0_k4_cvrep4_prec32",
    "hyb0_k5_cvrep4_eq40",
    "hyb0_k5_cvrep4_prec40",
    # Cross-stacker {gm2l_r4e8, hyb0} + CVREP4
    "dual_k4_rep0_prec16",
    "dual_k4_cvrep4_eq64",
    "dual_k4_cvrep4_prec64",
    "dual_k5_cvrep4_prec80",
)

CANDIDATE_RULES = (
    "gm2l_k4_rep0_eq8",
    "gm2l_k4_rep0_prec8",
    "gm2l_k5_rep0_prec10",
    "gm2l_k4_cvrep4_eq32",
    "gm2l_k4_cvrep4_prec32",
    "gm2l_k5_cvrep4_prec40",
    "hyb0_wcm_rep0_eq2",
    "hyb0_k2_rep0_eq4",
    "hyb0_k2_rep0_prec4",
    "hyb0_k4_rep0_eq8",
    "hyb0_k4_rep0_prec8",
    "hyb0_k5_rep0_eq10",
    "hyb0_k5_rep0_prec10",
    "hyb0_k4_cvrep4_eq32",
    "hyb0_k4_cvrep4_prec32",
    "hyb0_k5_cvrep4_eq40",
    "hyb0_k5_cvrep4_prec40",
    "dual_k4_rep0_prec16",
    "dual_k4_cvrep4_eq64",
    "dual_k4_cvrep4_prec64",
    "dual_k5_cvrep4_prec80",
)


def _score_k5_grid(
    y: np.ndarray,
    ya: np.ndarray,
    ba: float,
    pg: Dict[Tuple[int, int, int, int], np.ndarray],
    qg: Dict[Tuple[int, int, int, int], np.ndarray],
) -> Dict[str, float]:
    """`st in {0=gm2l_r4e8, 1=tabcv7_hyb0}, ni in {0=wc_M, 1=s46201, 2=s46202, 3=s46203, 4=s46204}, gi in {0,1}, r in {0..3}`."""
    u_n0_mean = 0.5 * (float(M.utility(pg[(0, 0, 0, 0)], ya, ba)[0]) + float(M.utility(pg[(0, 0, 1, 0)], ya, ba)[0]))

    def _eq(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        return float(M.utility(np.mean([pg[k] for k in keys], axis=0), ya, ba)[0])

    def _prec(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        p_pr, _ = bag_predictions_precision_weighted_oof(y, [qg[k] for k in keys], [pg[k] for k in keys])
        return float(M.utility(p_pr, ya, ba)[0])

    return {
        "u_n0_mean_conv": u_n0_mean,
        "u_pred_avg_2_n0": _eq([(0, 0, 0, 0), (0, 0, 1, 0)]),
        "u_gm2l_k2_rep0_eq4": _eq([(0, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]),
        "u_gm2l_k4_rep0_eq8": _eq([(0, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "u_gm2l_k4_rep0_prec8": _prec([(0, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "u_gm2l_k5_rep0_prec10": _prec([(0, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "u_gm2l_k2_cvrep4_eq16": _eq([(0, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(4)]),
        "u_gm2l_k4_cvrep4_eq32": _eq([(0, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_gm2l_k4_cvrep4_prec32": _prec([(0, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_gm2l_k5_cvrep4_prec40": _prec([(0, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "u_hyb0_wcm_rep0_eq2": _eq([(1, 0, 0, 0), (1, 0, 1, 0)]),
        "u_hyb0_k2_rep0_eq4": _eq([(1, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]),
        "u_hyb0_k2_rep0_prec4": _prec([(1, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]),
        "u_hyb0_k4_rep0_eq8": _eq([(1, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "u_hyb0_k4_rep0_prec8": _prec([(1, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "u_hyb0_k5_rep0_eq10": _eq([(1, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "u_hyb0_k5_rep0_prec10": _prec([(1, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "u_hyb0_k4_cvrep4_eq32": _eq([(1, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_hyb0_k4_cvrep4_prec32": _prec([(1, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_hyb0_k5_cvrep4_eq40": _eq([(1, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "u_hyb0_k5_cvrep4_prec40": _prec([(1, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "u_dual_k4_rep0_prec16": _prec([(st, ni, gi, 0) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "u_dual_k4_cvrep4_eq64": _eq([(st, ni, gi, r) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_dual_k4_cvrep4_prec64": _prec([(st, ni, gi, r) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "u_dual_k5_cvrep4_prec80": _prec([(st, ni, gi, r) for st in (0, 1) for ni in range(5) for gi in (0, 1) for r in range(4)]),
    }


def _eval_ctx_worker(job: Tuple) -> Dict:
    c, Xs, Xa, y, lab0, gp0_p, gp1_p, is_d91_gp, blk_m, blk_k, blk_f, blk_tab, r4_list, hyb_list = job
    n_reps = 4
    gp_packs = []
    ya_ref: Optional[np.ndarray] = None
    ba_ref: float = 0.0
    for gi, gp_p in enumerate((gp0_p, gp1_p)):
        with np.load(gp_p, allow_pickle=False) as zg:
            Q_gp = np.asarray(zg["Q"], np.float64)
            ya = np.asarray(zg["ya"], np.float64)
            ba = float(zg["ba"])
            ym = float(zg["ym"])
            m_gp, n_a = Q_gp.shape[1], len(ya)
            mean_a = np.full(n_a, ym, dtype=np.float64)
            if is_d91_gp:
                bag_gp = [None if not np.all(np.isfinite(zg["AKF"][:, i])) else zg["AKF"][:, i].astype(np.float64) for i in range(m_gp)]
                ref_gp = [zg["PA"][:, i].astype(np.float64) if bool(zg["pa_ok"][i]) else None for i in range(m_gp)]
                bcols = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]
                rcols = ref_gp
            else:
                BAG_gp = np.asarray(zg["BAG"], np.float64)
                REF_gp = np.asarray(zg["REF"], np.float64)
                bcols = [BAG_gp[:, j] if np.all(np.isfinite(BAG_gp[:, j])) else mean_a for j in range(m_gp)]
                rcols = [REF_gp[:, j] if np.all(np.isfinite(REF_gp[:, j])) else None for j in range(m_gp)]
        if ya_ref is None:
            ya_ref = ya
            ba_ref = ba
        gp_packs.append((gi, Q_gp, bcols, rcols))

    assert ya_ref is not None
    labs_rep = [lab0 if r == 0 else make_stratified_kfold(c, y, k=5, rep=r) for r in range(n_reps)]
    cids_rep = [c if r == 0 else f"{c}#cvrep{r}" for r in range(n_reps)]
    base_cv7 = [(t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None]

    pg, qg = {}, {}
    for ni in range(5):
        r4_blk = r4_list[ni]
        hyb_blk = hyb_list[ni]
        act_gm = [b for b in (blk_m, blk_k, blk_f, r4_blk) if b is not None]
        tb_hyb = list(base_cv7) + ([("e8_cv7_hyb", hyb_blk)] if hyb_blk is not None else [])
        for gi, Q_gp, bcols, rcols in gp_packs:
            for r in range(n_reps):
                cr, lr = cids_rep[r], labs_rep[r]
                p_gm, _ = GM.guard_and_stack_2l(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, act_gm)
                pg[(0, ni, gi, r)] = np.asarray(p_gm, np.float64)
                qg[(0, ni, gi, r)] = _nested_oof_gm2l_rep(cr, y, lr, Q_gp, act_gm)

                p_hyb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_hyb, use_cv7_folds=True)
                pg[(1, ni, gi, r)] = np.asarray(p_hyb, np.float64)
                qg[(1, ni, gi, r)] = _nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_hyb)

    return {"cid": c, "scores": _score_k5_grid(y, ya_ref, ba_ref, pg, qg)}


def _summarize_slice(by_cid: Dict[str, Dict[str, float]], cids: List[str], fam_map: Dict[str, str]) -> dict:
    fk = {c: fam_map[c] for c in cids}
    umaps = {rk: {c: float(by_cid[c][f"u_{rk}"]) for c in cids} for rk in RULE_KEYS}
    return {
        "n_contexts": len(cids),
        "n_families": len(set(fk.values())),
        "family_mean_u": {rk: float(H.family_mean(umaps[rk], fk)) for rk in RULE_KEYS},
        "ctx_mean_u": {rk: float(np.mean(list(umaps[rk].values()))) for rk in RULE_KEYS},
        "pair_stats_hyb0_k5_cvrep4_prec_vs_gm2l_k2_cvrep4_eq": T16.pair_stats(
            np, H, umaps["hyb0_k5_cvrep4_prec40"], umaps["gm2l_k2_cvrep4_eq16"], fk
        ),
        "pair_stats_dual_k5_cvrep4_prec_vs_gm2l_k2_cvrep4_eq": T16.pair_stats(
            np, H, umaps["dual_k5_cvrep4_prec80"], umaps["gm2l_k2_cvrep4_eq16"], fk
        ),
        "pair_stats_gm2l_k4_cvrep4_eq_vs_gm2l_k2_cvrep4_eq": T16.pair_stats(
            np, H, umaps["gm2l_k4_cvrep4_eq32"], umaps["gm2l_k2_cvrep4_eq16"], fk
        ),
    }


def _load_bank(bank_p: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rows = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        assert "ya" not in zb.files
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            rows[c] = (
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n
    return rows


def run_select_fitsel(a) -> int:
    t0 = time.monotonic()
    root = Path(a.bundle_root)
    out_lock = Path(a.lock_json)
    for p in (root, out_lock):
        _check_safe_path(p)
    out_lock.parent.mkdir(parents=True, exist_ok=True)
    if out_lock.exists():
        raise SystemExit(f"REFUSAL: create-once lock already exists: {out_lock}")

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

        r4_list = [_load_col_block_flex(fs_in / "cols_r4e8_wc_FITSEL" / f"{c}.npz", ("r4_e8",), n_s, n_a)]
        hyb_list = [_load_col_block_flex(cols_fs / "cv7_hyb_wc_m_base" / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a)]
        for sn in (46201, 46202, 46203, 46204):
            r4_list.append(_load_col_block_flex(cols_fs / f"r4e8_s{sn}_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a))
            hyb_list.append(_load_col_block_flex(cols_fs / f"cv7_hyb_s{sn}_{vmode}" / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a))

        jobs.append((c, Xs, Xa, y, lab0, fs_in / "dev1_gp" / f"{c}.npz", fs_in / "dev2_gp" / f"{c}.npz", False, blk_m, blk_k, blk_f, blk_tab, r4_list, hyb_list))

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(max(1, a.procs), len(jobs))) as pool:
        done = list(pool.imap_unordered(_eval_ctx_worker, jobs, chunksize=1))

    by_cid = {r["cid"]: r["scores"] for r in done}
    s88 = _summarize_slice(by_cid, clean_88, fam_map)
    s106 = _summarize_slice(by_cid, all_106, fam_map)

    fmu = s88["family_mean_u"]
    best_rule = max(CANDIDATE_RULES, key=lambda rk: fmu[rk])
    base_rule = "gm2l_k2_cvrep4_eq16"
    delta_u = float(fmu[best_rule] - fmu[base_rule])
    gate_pass = bool(delta_u > 0.0)
    locked_rule = best_rule if gate_pass else base_rule

    doc = {
        "schema": "jope.h462.m6_k5_fair0_fitsel_lock/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": os.uname().nodename,
        "selected_valley_mode": vmode,
        "dev91_opens": 0,
        "gp_anchor_dir": "",
        "quarantine": {"confirm_opens": 0, "fmgap_bc_opens": 0, "holdout_opens": 0, "sealed_opens": 0},
        "elapsed_s": round(time.monotonic() - t0, 2),
        "best_candidate": best_rule,
        "clean88_best_u": float(fmu[best_rule]),
        "clean88_gm2l_k2_cvrep4_u": float(fmu[base_rule]),
        "clean88_delta_u": delta_u,
        "gate_pass": gate_pass,
        "locked_rule": locked_rule,
        "fitsel_clean_non_copy_88": s88,
        "fitsel_all_106": s106,
    }
    fd = os.open(str(out_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "OK", "mode": "select-fitsel", "locked_rule": locked_rule, "gate_pass": gate_pass, "clean88_delta_u": round(delta_u, 6), "lock_sha256": _sha256_file(out_lock), "elapsed_s": doc["elapsed_s"]}, sort_keys=True), flush=True)
    return 0


def _eval_ctx_worker_d91_locked(job: Tuple) -> Dict:
    """R1 single-entrant D91 worker: computes ONLY `locked_rule` + 4 fixed references (0 exploratory variants)."""
    c, Xs, Xa, y, lab0, gp0_p, gp1_p, blk_m, blk_k, blk_f, blk_tab, r4_list, hyb_list, locked_rule = job
    gp_packs = []
    ya_ref: Optional[np.ndarray] = None
    ba_ref: float = 0.0
    for gi, gp_p in enumerate((gp0_p, gp1_p)):
        with np.load(gp_p, allow_pickle=False) as zg:
            Q_gp = np.asarray(zg["Q"], np.float64)
            ya = np.asarray(zg["ya"], np.float64)
            ba = float(zg["ba"])
            ym = float(zg["ym"])
            m_gp, n_a = Q_gp.shape[1], len(ya)
            mean_a = np.full(n_a, ym, dtype=np.float64)
            bag_gp = [None if not np.all(np.isfinite(zg["AKF"][:, i])) else zg["AKF"][:, i].astype(np.float64) for i in range(m_gp)]
            ref_gp = [zg["PA"][:, i].astype(np.float64) if bool(zg["pa_ok"][i]) else None for i in range(m_gp)]
            bcols = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]
            rcols = ref_gp
        if ya_ref is None:
            ya_ref = ya
            ba_ref = ba
        gp_packs.append((gi, Q_gp, bcols, rcols))

    assert ya_ref is not None
    labs_rep = [lab0 if r == 0 else make_stratified_kfold(c, y, k=5, rep=r) for r in range(8)]
    cids_rep = [c if r == 0 else f"{c}#cvrep{r}" for r in range(8)]
    base_cv7 = [(t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None]

    pg, qg = {}, {}
    for ni in range(5):
        r4_blk = r4_list[ni]
        hyb_blk = hyb_list[ni]
        act_gm = [b for b in (blk_m, blk_k, blk_f, r4_blk) if b is not None]
        tb_hyb = list(base_cv7) + ([("e8_cv7_hyb", hyb_blk)] if hyb_blk is not None else [])
        max_r_hyb = 8 if ni in (0, 1, 2) else 4
        for gi, Q_gp, bcols, rcols in gp_packs:
            for r in range(4):
                cr, lr = cids_rep[r], labs_rep[r]
                p_gm, _ = GM.guard_and_stack_2l(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, act_gm)
                pg[(0, ni, gi, r)] = np.asarray(p_gm, np.float64)
                qg[(0, ni, gi, r)] = _nested_oof_gm2l_rep(cr, y, lr, Q_gp, act_gm)
            for r in range(max_r_hyb):
                cr, lr = cids_rep[r], labs_rep[r]
                p_hyb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_hyb, use_cv7_folds=True)
                pg[(1, ni, gi, r)] = np.asarray(p_hyb, np.float64)
                qg[(1, ni, gi, r)] = _nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_hyb)

    def _eq(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        return float(M.utility(np.mean([pg[k] for k in keys], axis=0), ya_ref, ba_ref)[0])

    def _prec(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        p_pr, _ = bag_predictions_precision_weighted_oof(y, [qg[k] for k in keys], [pg[k] for k in keys])
        return float(M.utility(p_pr, ya_ref, ba_ref)[0])

    # Evaluate ONLY the single locked entrant from the full rule table + 4 fixed references
    all_rule_fn = {
        "gm2l_k2_cvrep4_eq16": lambda: _eq([(0, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(4)]),
        "gm2l_k4_rep0_eq8": lambda: _eq([(0, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "gm2l_k4_rep0_prec8": lambda: _prec([(0, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "gm2l_k5_rep0_prec10": lambda: _prec([(0, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "gm2l_k4_cvrep4_eq32": lambda: _eq([(0, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "gm2l_k4_cvrep4_prec32": lambda: _prec([(0, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "gm2l_k5_cvrep4_prec40": lambda: _prec([(0, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "hyb0_wcm_rep0_eq2": lambda: _eq([(1, 0, 0, 0), (1, 0, 1, 0)]),
        "hyb0_k2_rep0_eq4": lambda: _eq([(1, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]),
        "hyb0_k2_rep0_prec4": lambda: _prec([(1, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]),
        "hyb0_k4_rep0_eq8": lambda: _eq([(1, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "hyb0_k4_rep0_prec8": lambda: _prec([(1, ni, gi, 0) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "hyb0_k5_rep0_eq10": lambda: _eq([(1, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "hyb0_k5_rep0_prec10": lambda: _prec([(1, ni, gi, 0) for ni in range(5) for gi in (0, 1)]),
        "hyb0_k4_cvrep4_eq32": lambda: _eq([(1, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "hyb0_k4_cvrep4_prec32": lambda: _prec([(1, ni, gi, r) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "hyb0_k5_cvrep4_eq40": lambda: _eq([(1, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "hyb0_k5_cvrep4_prec40": lambda: _prec([(1, ni, gi, r) for ni in range(5) for gi in (0, 1) for r in range(4)]),
        "dual_k4_rep0_prec16": lambda: _prec([(st, ni, gi, 0) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1)]),
        "dual_k4_cvrep4_eq64": lambda: _eq([(st, ni, gi, r) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "dual_k4_cvrep4_prec64": lambda: _prec([(st, ni, gi, r) for st in (0, 1) for ni in (1, 2, 3, 4) for gi in (0, 1) for r in range(4)]),
        "dual_k5_cvrep4_prec80": lambda: _prec([(st, ni, gi, r) for st in (0, 1) for ni in range(5) for gi in (0, 1) for r in range(4)]),
    }
    if locked_rule not in all_rule_fn:
        raise ValueError(f"Unknown locked_rule: {locked_rule!r}")

    u_n0_mean = 0.5 * (float(M.utility(pg[(0, 0, 0, 0)], ya_ref, ba_ref)[0]) + float(M.utility(pg[(0, 0, 1, 0)], ya_ref, ba_ref)[0]))
    return {
        "cid": c,
        "scores": {
            "locked": all_rule_fn[locked_rule](),
            "ref_n0_mean_conv": u_n0_mean,
            "ref_pred_avg_2_n0": _eq([(0, 0, 0, 0), (0, 0, 1, 0)]),
            "ref_gm2l_k2_cvrep4_eq16": _eq([(0, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(4)]),
            "ref_tabcv7_fair0_cvrep8_prec48": _prec([(1, ni, gi, r) for ni in (0, 1, 2) for gi in (0, 1) for r in range(8)]),
        },
    }


def _verify_pre_ledger_and_next_pass(ledger_p: Path, lock_sha: str) -> Tuple[int, dict]:
    """R2 Look-Ledger gate: hard-requires a prior `type: 'pre'` record matching `fitsel_lock_sha256`."""
    if not ledger_p.is_file():
        raise SystemExit(f"REFUSAL (R2): missing LOOK_LEDGER.jsonl at {ledger_p}")
    recs = [json.loads(line) for line in ledger_p.read_text().splitlines() if line.strip()]
    pre_matches = [r for r in recs if r.get("type") == "pre" and r.get("fitsel_lock_sha256") == lock_sha]
    if not pre_matches:
        raise SystemExit(
            f"REFUSAL (R2): no `type: \"pre\"` record in {ledger_p} for fitsel_lock_sha256={lock_sha}"
        )
    prior_looks = [r for r in recs if r.get("type") == "look" and r.get("fitsel_lock_sha256") == lock_sha]
    if prior_looks:
        raise SystemExit(
            f"REFUSAL (R1/R2): `type: \"look\"` already recorded in {ledger_p} for fitsel_lock_sha256={lock_sha}"
        )
    max_pass = max([int(r.get("pass_number", 0)) for r in recs if "pass_number" in r] + [7])
    pre_rec = pre_matches[-1]
    pass_num = int(pre_rec.get("pass_number", max_pass + 1))
    return pass_num, pre_rec


def run_price_d91(a) -> int:
    t0 = time.monotonic()
    root = Path(a.bundle_root)
    lock_p = Path(a.lock_json)
    out_p = Path(a.out_json)
    ledger_p = Path(a.look_ledger_jsonl)
    for p in (root, lock_p, out_p, ledger_p):
        _check_safe_path(p)
    if out_p.exists():
        raise SystemExit(f"REFUSAL: create-once D91 receipt already exists: {out_p}")
    lock_sha = _sha256_file(lock_p)
    lock_doc = json.loads(lock_p.read_text())
    assert lock_doc.get("schema") == "jope.h462.m6_k5_fair0_fitsel_lock/v1"
    # R2 Gate BEFORE opening any dev91 board or bank file!
    pass_num, pre_rec = _verify_pre_ledger_and_next_pass(ledger_p, lock_sha)

    locked_rule = str(lock_doc["locked_rule"])
    vmode = str(lock_doc["selected_valley_mode"])
    series_label = f"dev91 development read, pass {pass_num}, adaptive series, locked entrant"

    d91_in = root / "d91_inputs"
    cols_d91 = root / "cols_d91"
    board_dir = d91_in / "board_post_h448"
    b_meta = json.loads((board_dir / "BOARD.json").read_text())
    cids = [x["cid"] for x in b_meta["contexts"]]
    fam = {x["cid"]: x["family_id"] for x in b_meta["contexts"]}
    fk = {c: fam[c] for c in cids}

    base_field = {}
    for jf in sorted((board_dir / "scores").glob("*.jsonl")):
        recs = [json.loads(line) for line in jf.read_text().splitlines() if line.strip()]
        if len(recs) == len(cids):
            base_field[jf.stem] = {r["cid"]: float(r["utility"]) for r in recs}

    bank_rows = _load_bank(d91_in / "bank_d91_search_f64.npz")
    jobs = []
    for c in cids:
        Xs, Xa, y, lab0 = bank_rows[c]
        n_s, n_a = len(y), len(Xa)
        blk_m = _load_col_block_flex(d91_in / "cols_matern" / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
        blk_k = _clip_k3(_load_col_block_flex(d91_in / "cols_k3" / f"{c}.npz", H430_K3, n_s, n_a), y)
        blk_f = _clip_f4(_load_col_block_flex(d91_in / "cols_f4" / f"{c}.npz", H433_F4, n_s, n_a), y)
        blk_tab = _load_col_block_flex(cols_d91 / "cols_tab_h450_D91" / f"{c}.npz", H450_TAB_COLS, n_s, n_a)

        r4_list = [_load_col_block_flex(d91_in / "cols_r4_e8" / f"{c}.npz", ("r4_e8",), n_s, n_a)]
        # MUST load strictly from `cv7_hyb_fair0_*` (`gp_anchor_dir=""`, 0% dev91)!
        hyb_list = [_load_col_block_flex(cols_d91 / "cv7_hyb_fair0_wc_m" / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a)]
        for sn in (46201, 46202, 46203, 46204):
            r4_list.append(_load_col_block_flex(cols_d91 / f"r4e8_s{sn}_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a))
            hyb_list.append(_load_col_block_flex(cols_d91 / f"cv7_hyb_fair0_s{sn}_{vmode}" / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a))

        jobs.append((c, Xs, Xa, y, lab0, d91_in / "d1_gp" / f"{c}.npz", d91_in / "d2_gp" / f"{c}.npz", blk_m, blk_k, blk_f, blk_tab, r4_list, hyb_list, locked_rule))

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(max(1, a.procs), len(jobs))) as pool:
        done = list(pool.imap_unordered(_eval_ctx_worker_d91_locked, jobs, chunksize=1))

    by_cid = {r["cid"]: r["scores"] for r in done}
    fld_base = {e: {c: v[c] for c in cids} for e, v in base_field.items() if e != FAIR_CANONICAL}

    # R1: Score ONLY the single locked entrant + 4 fixed references (0 exploratory variants!)
    scored_keys = (
        ("locked", f"m6k5fair0.locked.{locked_rule}"),
        ("ref_n0_mean_conv", "m6k5fair0.ref.n0_mean_conv"),
        ("ref_pred_avg_2_n0", "m6k5fair0.ref.pred_avg_2_n0"),
        ("ref_gm2l_k2_cvrep4_eq16", "m6k5fair0.ref.gm2l_k2_cvrep4_eq16"),
        ("ref_tabcv7_fair0_cvrep8_prec48", "m6k5fair0.ref.tabcv7_fair0_cvrep8_prec48"),
    )
    single_placements = {}
    joint_maps = {}
    umaps = {}
    for sk, eid in scored_keys:
        umap = {c: float(by_cid[c][sk]) for c in cids}
        umaps[sk] = umap
        joint_maps[eid] = umap
        panel = dict(sorted({**fld_base, eid: umap}.items(), key=lambda kv: kv[0] + ".jsonl"))
        _, _, _, elo = T16.elo_boot(np, H, panel, fk, ANCHOR, LADDER_SEED, a.n_boot)
        order = sorted(elo, key=lambda e: -elo[e]["elo"])
        single_placements[sk] = {
            "entrant_id": eid,
            "elo": float(elo[eid]["elo"]),
            "ci95": [float(x) for x in elo[eid]["ci95"]],
            "rank": int(order.index(eid) + 1),
            "n_entrants": len(order),
            "family_mean_u": float(H.family_mean(umap, fk)),
            "ctx_mean_u": float(np.mean(list(umap.values()))),
            "vs_kumo_large_elo": float(elo[eid]["elo"] - elo[KUMO_LARGE]["elo"]) if KUMO_LARGE in elo else None,
            "vs_tabfm_ens_elo": float(elo[eid]["elo"] - elo[TABFM]["elo"]) if TABFM in elo else None,
        }

    panel_joint = dict(sorted({**fld_base, **joint_maps}.items(), key=lambda kv: kv[0] + ".jsonl"))
    arms_j, point_j, boot_j, _ = T16.elo_boot(np, H, panel_joint, fk, ANCHOR, LADDER_SEED, a.n_boot)
    eid_lk = f"m6k5fair0.locked.{locked_rule}"

    doc = {
        "schema": "jope.h462.m6_k5_fair0_d91_price/v2_r1r2r3",
        "series_label": series_label,
        "pass_number": pass_num,
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": os.uname().nodename,
        "provenance_class": "0.000000% dev91",
        "gp_anchor_dir": "",
        "selected_valley_mode": vmode,
        "fitsel_lock_sha256": lock_sha,
        "fitsel_gate_pass": bool(lock_doc["gate_pass"]),
        "fitsel_locked_rule": locked_rule,
        "exploratory_variants_scored_on_d91": 0,
        "n_new_entrants_scored": 1,
        "n_fixed_references_scored": 4,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "locked_single_placement": single_placements["locked"],
        "fixed_reference_placements": {k: v for k, v in single_placements.items() if k != "locked"},
        "h416_paired_family_tests_on_locked_entrant": {
            "locked_vs_tabcv7_fair0_cvrep8_prec48": T16.pair_stats(np, H, umaps["locked"], umaps["ref_tabcv7_fair0_cvrep8_prec48"], fk),
            "locked_vs_gm2l_k2_cvrep4_eq16": T16.pair_stats(np, H, umaps["locked"], umaps["ref_gm2l_k2_cvrep4_eq16"], fk),
            "locked_vs_pred_avg_2_n0": T16.pair_stats(np, H, umaps["locked"], umaps["ref_pred_avg_2_n0"], fk),
            "locked_vs_n0_mean_conv": T16.pair_stats(np, H, umaps["locked"], umaps["ref_n0_mean_conv"], fk),
        },
        "joint_paired_deltas_on_locked_entrant": {
            "locked_vs_tabcv7_fair0_cvrep8_prec48": T16.delta_elo(np, arms_j, point_j, boot_j, eid_lk, "m6k5fair0.ref.tabcv7_fair0_cvrep8_prec48"),
            "locked_vs_gm2l_k2_cvrep4_eq16": T16.delta_elo(np, arms_j, point_j, boot_j, eid_lk, "m6k5fair0.ref.gm2l_k2_cvrep4_eq16"),
            "locked_vs_pred_avg_2_n0": T16.delta_elo(np, arms_j, point_j, boot_j, eid_lk, "m6k5fair0.ref.pred_avg_2_n0"),
            "locked_vs_n0_mean_conv": T16.delta_elo(np, arms_j, point_j, boot_j, eid_lk, "m6k5fair0.ref.n0_mean_conv"),
        },
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out_p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    out_sha = _sha256_file(out_p)

    # R2: Append `type: "look"` to LOOK_LEDGER.jsonl
    look_entry = {
        "type": "look",
        "timestamp_utc": doc["timestamp_utc"],
        "pass_number": pass_num,
        "series_label": series_label,
        "wave": "H462-M6-K5-FAIR0",
        "arm": locked_rule,
        "fitsel_lock_sha256": lock_sha,
        "receipt_path": str(out_p),
        "receipt_sha256": out_sha,
        "locked_elo": round(single_placements["locked"]["elo"], 4),
        "n_new_entrants_scored": 1,
        "n_fixed_references_scored": 4,
        "n_elo_objects_scored": 5,
        "exploratory_variants_scored": 0,
    }
    with ledger_p.open("a", encoding="utf-8") as lf:
        lf.write(json.dumps(look_entry, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "OK",
                "mode": "price-d91",
                "pass_number": pass_num,
                "series_label": series_label,
                "locked_rule": locked_rule,
                "locked_elo": round(single_placements["locked"]["elo"], 2),
                "ref_tabcv7_fair0_cvrep8_prec48_elo": round(single_placements["ref_tabcv7_fair0_cvrep8_prec48"]["elo"], 2),
                "out_sha256": out_sha,
                "elapsed_s": doc["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("select-fitsel", "price-d91"), required=True)
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--lock-json", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--look-ledger-jsonl", default="/home/ubuntu/jope_plan/evidence/H458/LOOK_LEDGER.jsonl")
    ap.add_argument("--valley-mode", choices=("v12_avg", "v2"), default="v12_avg")
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    if a.mode == "select-fitsel":
        return run_select_fitsel(a)
    if not a.out_json:
        raise SystemExit("--out-json required for price-d91")
    return run_price_d91(a)


if __name__ == "__main__":
    raise SystemExit(main())
