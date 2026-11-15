#!/usr/bin/env python3
"""H462 `0.000000% dev91` `TAB450 + Stack-Guard-CV7 + R4E8 + CVREP8` (`eval_h462_tabcv7_cvrep.py`).

Implements pre-registered `H462-M6-TABCV7-CVREP` (`2026-11-15T12:59Z` in `shared_CHAT.md`):
  - Strictly `0.000000% dev91` across EVERY input byte:
      * `P418 D1, D2` symbolic GP (`0% dev91`)
      * `M7` Matérn kernel, `K3` exact-key, `F4` formula (`0% dev91`)
      * `cols_tab_h450` (`8` strictly per-table classical learners: Lead-Lag SVD-GCV Ridge, Exact-LOO Lap+Matérn ARD
        in identity & Yeo-Johnson+Duan space, CatBoost `d4/d6/d8-hurdle` YJ+Duan, LightGBM YJ+Duan, ExtraTrees+HGB
        YJ+Duan; `0` pretrained neural weights, `0` `gp_anchors_p0`, `0` `ya` read)
      * `r4_e8` (`wc_M`, `s46201`, `s46202`; `d91_share=0.0`, `566` masked tables, `0` `gp_anchors_p0`)
  - Evaluates BOTH stacker families (`S0 = GM.guard_and_stack_2l` baseline vs `S1 = stack_guard_cv7_context + tab450`)
    and cross-stacker `dual = {S0, S1}` across `R in {1, 4, 8}` stratified 5-fold CV partitions (`rep in 0..7`) with
    both equal-weight (`eq`) and audit-blind OOF inverse-MSE precision weighting (`M6-PRECISION-AVG`).
  - Step A (`--mode select-fitsel`, `0` `dev91` opens): Selects optimal rule strictly on `FITSEL` (`clean_non_copy_88`
    primary, `all_106` descriptive) and writes create-once `FITSEL_LOCK_TABCV7_CVREP.json` (`0444`).
  - Step B (`--mode price-d91`, single frozen pass): Verifies `FITSEL_LOCK_TABCV7_CVREP.json` and prices once on
    quarantined `board_post_h448` (`91` contexts, `19` families).
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

ANCHOR = T16.ANCHOR if T16 is not None else "dope.v1"
LADDER_SEED = T16.LADDER_SEED if T16 is not None else 416001
FAIR_CANONICAL = "h448.APEX_FAIR_CHAMPION"
KUMO_LARGE = "kumo_tabular_large_e8.v1"
TABFM = "tabfm_ensemble.v1"

H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
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

FORBIDDEN_TOKENS = ("CONFIRM", "FMGAP_B", "FMGAP_C", "HOLDOUT", "SEALED", "PROBE_YA", "/tmp/")


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
    """Strictly nested outer-fold CV7 Guard+BCAR/Sharp-Simplex OOF (`0` `ya`/`Xa` read)."""
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


def _uncentered_error_corr(e1: np.ndarray, e2: np.ndarray) -> float:
    d = float(np.linalg.norm(e1) * np.linalg.norm(e2))
    return float(np.dot(e1, e2) / d) if d > 1e-18 else 1.0


RULE_KEYS = (
    "n0_mean_conv",
    "pred_avg_2_n0",
    "gm2l_rep0_eq4_s12",
    "gm2l_cvrep4_eq16_s12",
    "gm2l_cvrep8_prec32_s12",
    "tabcv7_n0_rep0_eq2",
    "tabcv7_rep0_eq4_s12",
    "tabcv7_rep0_prec4_s12",
    "tabcv7_cvrep4_eq16_s12",
    "tabcv7_cvrep4_prec16_s12",
    "tabcv7_cvrep8_eq32_s12",
    "tabcv7_cvrep8_prec32_s12",
    "tabcv7_cvrep8_eq48_all",
    "tabcv7_cvrep8_prec48_all",
    "dual_rep0_eq8_s12",
    "dual_rep0_prec8_s12",
    "dual_cvrep4_eq32_s12",
    "dual_cvrep4_prec32_s12",
    "dual_cvrep8_eq64_s12",
    "dual_cvrep8_prec64_s12",
)

CANDIDATE_RULES = (
    "tabcv7_rep0_eq4_s12",
    "tabcv7_rep0_prec4_s12",
    "tabcv7_cvrep4_eq16_s12",
    "tabcv7_cvrep4_prec16_s12",
    "tabcv7_cvrep8_eq32_s12",
    "tabcv7_cvrep8_prec32_s12",
    "tabcv7_cvrep8_eq48_all",
    "tabcv7_cvrep8_prec48_all",
    "dual_rep0_eq8_s12",
    "dual_rep0_prec8_s12",
    "dual_cvrep4_eq32_s12",
    "dual_cvrep4_prec32_s12",
    "dual_cvrep8_eq64_s12",
    "dual_cvrep8_prec64_s12",
)


def _score_tabcv7_grid(
    y: np.ndarray,
    ya: np.ndarray,
    ba: float,
    pred_grid: Dict[Tuple[int, int, int, int], np.ndarray],
    oof_grid: Dict[Tuple[int, int, int, int], np.ndarray],
) -> Dict[str, float]:
    """Prices `(st in {0=gm2l, 1=tabcv7}, ni in {0,1,2}, gi in {0,1}, rep in {0..7})` (`0` `ya` read inside weights!)."""
    u_n0_d1 = float(M.utility(pred_grid[(0, 0, 0, 0)], ya, ba)[0])
    u_n0_d2 = float(M.utility(pred_grid[(0, 0, 1, 0)], ya, ba)[0])
    u_n0_mean = 0.5 * (u_n0_d1 + u_n0_d2)

    def _eq(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        p = np.mean([pred_grid[k] for k in keys], axis=0)
        return float(M.utility(p, ya, ba)[0])

    def _prec(keys: Sequence[Tuple[int, int, int, int]]) -> float:
        p_pr, _ = bag_predictions_precision_weighted_oof(
            y,
            [oof_grid[k] for k in keys],
            [pred_grid[k] for k in keys],
        )
        return float(M.utility(p_pr, ya, ba)[0])

    gm_n0_rep0 = [(0, 0, 0, 0), (0, 0, 1, 0)]
    gm_s12_rep0 = [(0, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]
    gm_s12_cv4 = [(0, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(4)]
    gm_s12_cv8 = [(0, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(8)]

    tb_n0_rep0 = [(1, 0, 0, 0), (1, 0, 1, 0)]
    tb_s12_rep0 = [(1, ni, gi, 0) for ni in (1, 2) for gi in (0, 1)]
    tb_s12_cv4 = [(1, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(4)]
    tb_s12_cv8 = [(1, ni, gi, r) for ni in (1, 2) for gi in (0, 1) for r in range(8)]
    tb_all_cv8 = [(1, ni, gi, r) for ni in (0, 1, 2) for gi in (0, 1) for r in range(8)]

    du_s12_rep0 = [(st, ni, gi, 0) for st in (0, 1) for ni in (1, 2) for gi in (0, 1)]
    du_s12_cv4 = [(st, ni, gi, r) for st in (0, 1) for ni in (1, 2) for gi in (0, 1) for r in range(4)]
    du_s12_cv8 = [(st, ni, gi, r) for st in (0, 1) for ni in (1, 2) for gi in (0, 1) for r in range(8)]

    rho_stacker = float(
        np.mean(
            [
                _uncentered_error_corr(pred_grid[(0, ni, gi, 0)] - ya, pred_grid[(1, ni, gi, 0)] - ya)
                for ni in (1, 2)
                for gi in (0, 1)
            ]
        )
    )
    rho_tabcv7_cvrep = float(
        np.mean(
            [
                _uncentered_error_corr(pred_grid[(1, ni, gi, 0)] - ya, pred_grid[(1, ni, gi, r)] - ya)
                for ni in (1, 2)
                for gi in (0, 1)
                for r in range(1, 8)
            ]
        )
    )

    return {
        "u_n0_mean_conv": u_n0_mean,
        "u_pred_avg_2_n0": _eq(gm_n0_rep0),
        "u_gm2l_rep0_eq4_s12": _eq(gm_s12_rep0),
        "u_gm2l_cvrep4_eq16_s12": _eq(gm_s12_cv4),
        "u_gm2l_cvrep8_prec32_s12": _prec(gm_s12_cv8),
        "u_tabcv7_n0_rep0_eq2": _eq(tb_n0_rep0),
        "u_tabcv7_rep0_eq4_s12": _eq(tb_s12_rep0),
        "u_tabcv7_rep0_prec4_s12": _prec(tb_s12_rep0),
        "u_tabcv7_cvrep4_eq16_s12": _eq(tb_s12_cv4),
        "u_tabcv7_cvrep4_prec16_s12": _prec(tb_s12_cv4),
        "u_tabcv7_cvrep8_eq32_s12": _eq(tb_s12_cv8),
        "u_tabcv7_cvrep8_prec32_s12": _prec(tb_s12_cv8),
        "u_tabcv7_cvrep8_eq48_all": _eq(tb_all_cv8),
        "u_tabcv7_cvrep8_prec48_all": _prec(tb_all_cv8),
        "u_dual_rep0_eq8_s12": _eq(du_s12_rep0),
        "u_dual_rep0_prec8_s12": _prec(du_s12_rep0),
        "u_dual_cvrep4_eq32_s12": _eq(du_s12_cv4),
        "u_dual_cvrep4_prec32_s12": _prec(du_s12_cv4),
        "u_dual_cvrep8_eq64_s12": _eq(du_s12_cv8),
        "u_dual_cvrep8_prec64_s12": _prec(du_s12_cv8),
        "rho_err_stacker_axis": rho_stacker,
        "rho_err_tabcv7_cvrep_axis": rho_tabcv7_cvrep,
    }


class _NoAuditLabelNpz:
    """Wrapper over `np.load` forbidding any access to audit labels (`ya`, `ba`, `y_audit`, `audit_y`)."""

    _FORBIDDEN = frozenset({"ya", "ba", "y_audit", "audit_y"})

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        _check_safe_path(self._path)
        self._npz = np.load(self._path, allow_pickle=False)

    def __enter__(self) -> "_NoAuditLabelNpz":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._npz.close()

    @property
    def files(self) -> List[str]:
        return [k for k in self._npz.files if k not in self._FORBIDDEN]

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._FORBIDDEN:
            raise PermissionError(f"Audit label key {key!r} forbidden in _NoAuditLabelNpz ({self._path})")
        return self._npz[key]


def predict_tabcv7_cvrep8_prec48_context_no_ya(
    c: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab0: np.ndarray,
    gp0_p: Path,
    gp1_p: Path,
    is_d91_gp: bool,
    m7_p: Path,
    k3_p: Path,
    f4_p: Path,
    tab450_p: Path,
    wc_r4_p: Path,
    s1_r4_p: Path,
    s2_r4_p: Path,
    n_reps: int = 8,
) -> Dict[str, object]:
    """`TC`'s canonical `0`-`ya` per-context `tabcv7_cvrep8_prec48_all` predictor (`F2` / `G4` / `Addition 1`).

    Uses `TC`'s own `GM.PRIMARY_NAMES`, `H430_K3`, `H433_F4`, `H450_TAB_COLS`, `_load_col_block_flex`,
    `_clip_k3`, `_clip_f4`, `stack_guard_cv7_context`, `_nested_oof_cv7_rep`, and
    `bag_predictions_precision_weighted_oof` while opening `gp0_p`/`gp1_p` through `_NoAuditLabelNpz`
    (`0` `ya`/`ba` reads, `0` `M.utility` calls).
    """
    n_s, n_a = len(y), len(Xa)
    blk_m = _load_col_block_flex(Path(m7_p), GM.PRIMARY_NAMES, n_s, n_a)
    blk_k = _clip_k3(_load_col_block_flex(Path(k3_p), H430_K3, n_s, n_a), y)
    blk_f = _clip_f4(_load_col_block_flex(Path(f4_p), H433_F4, n_s, n_a), y)
    blk_tab = _load_col_block_flex(Path(tab450_p), H450_TAB_COLS, n_s, n_a)
    r4_wc = _load_col_block_flex(Path(wc_r4_p), ("r4_e8",), n_s, n_a)
    r4_s1 = _load_col_block_flex(Path(s1_r4_p), ("r4_e8",), n_s, n_a)
    r4_s2 = _load_col_block_flex(Path(s2_r4_p), ("r4_e8",), n_s, n_a)

    gp_packs = []
    for gi, gp_p in enumerate((gp0_p, gp1_p)):
        with _NoAuditLabelNpz(Path(gp_p)) as zg:
            Q_gp = np.asarray(zg["Q"], np.float64)
            ym = float(zg["ym"])
            m_gp = Q_gp.shape[1]
            mean_a = np.full(n_a, ym, dtype=np.float64)
            if is_d91_gp:
                bag_gp = [
                    None if not np.all(np.isfinite(zg["AKF"][:, i])) else zg["AKF"][:, i].astype(np.float64)
                    for i in range(m_gp)
                ]
                ref_gp = [zg["PA"][:, i].astype(np.float64) if bool(zg["pa_ok"][i]) else None for i in range(m_gp)]
                bcols = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]
                rcols = ref_gp
            else:
                BAG_gp = np.asarray(zg["BAG"], np.float64)
                REF_gp = np.asarray(zg["REF"], np.float64)
                bcols = [BAG_gp[:, j] if np.all(np.isfinite(BAG_gp[:, j])) else mean_a for j in range(m_gp)]
                rcols = [REF_gp[:, j] if np.all(np.isfinite(REF_gp[:, j])) else None for j in range(m_gp)]
        gp_packs.append((gi, Q_gp, bcols, rcols))

    labs_rep = [lab0 if r == 0 else make_stratified_kfold(c, y, k=5, rep=r) for r in range(n_reps)]
    cids_rep = [c if r == 0 else f"{c}#cvrep{r}" for r in range(n_reps)]
    base_common_cv7 = [
        (t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None
    ]

    preds_all: List[np.ndarray] = []
    oofs_all: List[np.ndarray] = []
    for _ni, r4_blk in ((0, r4_wc), (1, r4_s1), (2, r4_s2)):
        tb_cv7 = list(base_common_cv7) + ([("e8", r4_blk)] if r4_blk is not None else [])
        for _gi, Q_gp, bcols, rcols in gp_packs:
            for r in range(n_reps):
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_tb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_cv7, use_cv7_folds=True)
                q_tb = _nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7)
                preds_all.append(np.asarray(p_tb, np.float64))
                oofs_all.append(np.asarray(q_tb, np.float64))

    p_ens, meta_w = bag_predictions_precision_weighted_oof(y, oofs_all, preds_all)
    return {
        "p_audit_ens": np.asarray(p_ens, np.float64),
        "member_audit": np.column_stack(preds_all),
        "member_oof_hwm_fast": np.column_stack(oofs_all),
        "member_weights": np.asarray(meta_w["weights"], np.float64),
        "blocks_loaded": {
            "m7": blk_m is not None,
            "k3": blk_k is not None,
            "f4": blk_f is not None,
            "tab450": blk_tab is not None,
            "wc_r4e8": r4_wc is not None,
            "s1_r4e8": r4_s1 is not None,
            "s2_r4e8": r4_s2 is not None,
        },
    }


def _eval_ctx_worker(job: Tuple) -> Dict:
    c, Xs, Xa, y, lab0, gp0_p, gp1_p, is_d91_gp, blk_m, blk_k, blk_f, blk_tab, r4_wc, r4_s1, r4_s2, n_reps = job

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
    base_common_cv7 = [(t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None]

    pg, qg = {}, {}
    for ni, r4_blk in ((0, r4_wc), (1, r4_s1), (2, r4_s2)):
        act_gm = [b for b in (blk_m, blk_k, blk_f, r4_blk) if b is not None]
        tb_cv7 = list(base_common_cv7) + ([("e8", r4_blk)] if r4_blk is not None else [])
        # Note: for ni == 0, we only need r == 0 on gm2l (`gm_n0_rep0`), plus r in 0..n_reps-1 on tabcv7 (`tb_all_cv8`)
        gm_reps = [0] if ni == 0 else list(range(n_reps))
        for gi, Q_gp, bcols, rcols in gp_packs:
            for r in gm_reps:
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_gm, _ = GM.guard_and_stack_2l(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, act_gm)
                pg[(0, ni, gi, r)] = np.asarray(p_gm, np.float64)
                qg[(0, ni, gi, r)] = _nested_oof_gm2l_rep(cr, y, lr, Q_gp, act_gm)
            for r in range(n_reps):
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_tb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_cv7, use_cv7_folds=True)
                pg[(1, ni, gi, r)] = np.asarray(p_tb, np.float64)
                qg[(1, ni, gi, r)] = _nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7)

    return {"cid": c, "scores": _score_tabcv7_grid(y, ya_ref, ba_ref, pg, qg)}


def _summarize_slice(by_cid: Dict[str, Dict[str, float]], cids: List[str], fam_map: Dict[str, str]) -> dict:
    fk = {c: fam_map[c] for c in cids}
    umaps = {rk: {c: float(by_cid[c][f"u_{rk}"]) for c in cids} for rk in RULE_KEYS}
    return {
        "n_contexts": len(cids),
        "n_families": len(set(fk.values())),
        "family_mean_u": {rk: float(H.family_mean(umaps[rk], fk)) for rk in RULE_KEYS},
        "ctx_mean_u": {rk: float(np.mean(list(umaps[rk].values()))) for rk in RULE_KEYS},
        "pair_stats_tabcv7_cvrep4_prec_vs_gm2l_cvrep4_eq": T16.pair_stats(
            np, H, umaps["tabcv7_cvrep4_prec16_s12"], umaps["gm2l_cvrep4_eq16_s12"], fk
        ),
        "pair_stats_tabcv7_cvrep8_prec_vs_gm2l_cvrep4_eq": T16.pair_stats(
            np, H, umaps["tabcv7_cvrep8_prec32_s12"], umaps["gm2l_cvrep4_eq16_s12"], fk
        ),
        "pair_stats_dual_cvrep4_prec_vs_gm2l_cvrep4_eq": T16.pair_stats(
            np, H, umaps["dual_cvrep4_prec32_s12"], umaps["gm2l_cvrep4_eq16_s12"], fk
        ),
        "pair_stats_dual_cvrep8_prec_vs_gm2l_cvrep4_eq": T16.pair_stats(
            np, H, umaps["dual_cvrep8_prec64_s12"], umaps["gm2l_cvrep4_eq16_s12"], fk
        ),
        "pair_stats_dual_cvrep8_prec_vs_pred_avg_2_n0": T16.pair_stats(
            np, H, umaps["dual_cvrep8_prec64_s12"], umaps["pred_avg_2_n0"], fk
        ),
        "mean_error_rho_stacker_axis": float(np.mean([by_cid[c]["rho_err_stacker_axis"] for c in cids])),
        "mean_error_rho_tabcv7_cvrep_axis": float(np.mean([by_cid[c]["rho_err_tabcv7_cvrep_axis"] for c in cids])),
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
    _check_safe_path(root)
    out_lock = Path(a.lock_json)
    _check_safe_path(out_lock)
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
        r4_wc = _load_col_block_flex(fs_in / "cols_r4e8_wc_FITSEL" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s1 = _load_col_block_flex(cols_fs / f"r4e8_s46201_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s2 = _load_col_block_flex(cols_fs / f"r4e8_s46202_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        jobs.append(
            (
                c,
                Xs,
                Xa,
                y,
                lab0,
                fs_in / "dev1_gp" / f"{c}.npz",
                fs_in / "dev2_gp" / f"{c}.npz",
                False,
                blk_m,
                blk_k,
                blk_f,
                blk_tab,
                r4_wc,
                r4_s1,
                r4_s2,
                8,
            )
        )

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(max(1, a.procs), len(jobs))) as pool:
        done = list(pool.imap_unordered(_eval_ctx_worker, jobs, chunksize=1))

    by_cid = {r["cid"]: r["scores"] for r in done}
    s88 = _summarize_slice(by_cid, clean_88, fam_map)
    s106 = _summarize_slice(by_cid, all_106, fam_map)

    fmu = s88["family_mean_u"]
    best_rule = max(CANDIDATE_RULES, key=lambda rk: fmu[rk])
    base_cvrep = "gm2l_cvrep4_eq16_s12"
    delta_vs_cvrep = float(fmu[best_rule] - fmu[base_cvrep])
    delta_vs_rep0 = float(fmu[best_rule] - fmu["gm2l_rep0_eq4_s12"])
    gate_pass = bool(delta_vs_cvrep > 0.0)
    locked_rule = best_rule if gate_pass else base_cvrep

    doc = {
        "schema": "jope.h462.m6_tabcv7_cvrep_fitsel_lock/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": os.uname().nodename,
        "valley_mode": vmode,
        "dev91_opens": 0,
        "quarantine": {"confirm_opens": 0, "fmgap_bc_opens": 0, "holdout_opens": 0, "sealed_opens": 0},
        "elapsed_s": round(time.monotonic() - t0, 2),
        "best_tabcv7_candidate": best_rule,
        "clean88_best_candidate_u": float(fmu[best_rule]),
        "clean88_gm2l_cvrep4_eq16_u": float(fmu[base_cvrep]),
        "clean88_gm2l_rep0_eq4_u": float(fmu["gm2l_rep0_eq4_s12"]),
        "clean88_delta_vs_gm2l_cvrep4": delta_vs_cvrep,
        "clean88_delta_vs_gm2l_rep0": delta_vs_rep0,
        "gate_pass": gate_pass,
        "locked_rule": locked_rule,
        "fitsel_clean_non_copy_88": s88,
        "fitsel_all_106": s106,
    }
    fd = os.open(str(out_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "OK",
                "mode": "select-fitsel",
                "host": doc["host"],
                "valley_mode": vmode,
                "best_tabcv7_candidate": best_rule,
                "clean88_best_candidate_u": round(float(fmu[best_rule]), 6),
                "clean88_gm2l_cvrep4_eq16_u": round(float(fmu[base_cvrep]), 6),
                "clean88_delta_vs_gm2l_cvrep4": round(delta_vs_cvrep, 6),
                "clean88_delta_vs_gm2l_rep0": round(delta_vs_rep0, 6),
                "gate_pass": gate_pass,
                "locked_rule": locked_rule,
                "lock_sha256": _sha256_file(out_lock),
                "elapsed_s": doc["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def run_price_d91(a) -> int:
    t0 = time.monotonic()
    root = Path(a.bundle_root)
    lock_p = Path(a.lock_json)
    out_p = Path(a.out_json)
    for p in (root, lock_p, out_p):
        _check_safe_path(p)
    if not lock_p.is_file():
        raise SystemExit(f"REFUSAL: missing FITSEL lock {lock_p}")

    lock_doc = json.loads(lock_p.read_text())
    assert lock_doc.get("schema") == "jope.h462.m6_tabcv7_cvrep_fitsel_lock/v1"
    locked_rule = str(lock_doc["locked_rule"])
    best_cand = str(lock_doc["best_tabcv7_candidate"])
    vmode = str(lock_doc["valley_mode"])

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
        r4_wc = _load_col_block_flex(d91_in / "cols_r4_e8" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s1 = _load_col_block_flex(cols_d91 / f"r4e8_s46201_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s2 = _load_col_block_flex(cols_d91 / f"r4e8_s46202_{vmode}" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        jobs.append(
            (
                c,
                Xs,
                Xa,
                y,
                lab0,
                d91_in / "d1_gp" / f"{c}.npz",
                d91_in / "d2_gp" / f"{c}.npz",
                True,
                blk_m,
                blk_k,
                blk_f,
                blk_tab,
                r4_wc,
                r4_s1,
                r4_s2,
                8,
            )
        )

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(max(1, a.procs), len(jobs))) as pool:
        done = list(pool.imap_unordered(_eval_ctx_worker, jobs, chunksize=1))

    by_cid = {r["cid"]: r["scores"] for r in done}
    fld_base = {e: {c: v[c] for c in cids} for e, v in base_field.items() if e != FAIR_CANONICAL}

    single_placements = {}
    joint_maps = {}
    for rk in RULE_KEYS:
        eid = f"m6tabcv7.{rk}"
        umap = {c: float(by_cid[c][f"u_{rk}"]) for c in cids}
        joint_maps[eid] = umap
        panel = dict(sorted({**fld_base, eid: umap}.items(), key=lambda kv: kv[0] + ".jsonl"))
        _, _, _, elo = T16.elo_boot(np, H, panel, fk, ANCHOR, LADDER_SEED, a.n_boot)
        order = sorted(elo, key=lambda e: -elo[e]["elo"])
        single_placements[rk] = {
            "elo": float(elo[eid]["elo"]),
            "ci95": [float(x) for x in elo[eid]["ci95"]],
            "rank": int(order.index(eid) + 1),
            "n_entrants": len(order),
            "family_mean_u": float(H.family_mean(umap, fk)),
            "vs_kumo_large_elo": float(elo[eid]["elo"] - elo[KUMO_LARGE]["elo"]) if KUMO_LARGE in elo else None,
            "vs_tabfm_ens_elo": float(elo[eid]["elo"] - elo[TABFM]["elo"]) if TABFM in elo else None,
        }

    panel_joint = dict(sorted({**fld_base, **joint_maps}.items(), key=lambda kv: kv[0] + ".jsonl"))
    arms_j, point_j, boot_j, _ = T16.elo_boot(np, H, panel_joint, fk, ANCHOR, LADDER_SEED, a.n_boot)

    doc = {
        "schema": "jope.h462.m6_tabcv7_cvrep_d91_price/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": os.uname().nodename,
        "provenance_class": "0.000000% dev91",
        "valley_mode": vmode,
        "fitsel_lock_sha256": _sha256_file(lock_p),
        "fitsel_gate_pass": bool(lock_doc["gate_pass"]),
        "fitsel_locked_rule": locked_rule,
        "fitsel_best_candidate": best_cand,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "locked_single_placement": single_placements[locked_rule],
        "best_candidate_single_placement": single_placements[best_cand],
        "single_placements": single_placements,
        "joint_paired_deltas": {
            "locked_vs_gm2l_cvrep4_eq16": T16.delta_elo(
                np, arms_j, point_j, boot_j, f"m6tabcv7.{locked_rule}", "m6tabcv7.gm2l_cvrep4_eq16_s12"
            ),
            "locked_vs_pred_avg_2_n0": T16.delta_elo(
                np, arms_j, point_j, boot_j, f"m6tabcv7.{locked_rule}", "m6tabcv7.pred_avg_2_n0"
            ),
            "best_cand_vs_gm2l_cvrep4_eq16": T16.delta_elo(
                np, arms_j, point_j, boot_j, f"m6tabcv7.{best_cand}", "m6tabcv7.gm2l_cvrep4_eq16_s12"
            ),
            "dual_cvrep8_prec64_vs_pred_avg_2_n0": T16.delta_elo(
                np, arms_j, point_j, boot_j, "m6tabcv7.dual_cvrep8_prec64_s12", "m6tabcv7.pred_avg_2_n0"
            ),
        },
        "slice_summary": _summarize_slice(by_cid, cids, fam),
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "OK",
                "mode": "price-d91",
                "host": doc["host"],
                "valley_mode": vmode,
                "locked_rule": locked_rule,
                "locked_elo": round(single_placements[locked_rule]["elo"], 2),
                "best_candidate": best_cand,
                "best_candidate_elo": round(single_placements[best_cand]["elo"], 2),
                "gm2l_cvrep4_elo": round(single_placements["gm2l_cvrep4_eq16_s12"]["elo"], 2),
                "out_sha256": _sha256_file(out_p),
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
    ap.add_argument("--valley-mode", choices=("v12_avg", "v2"), default="v12_avg")
    ap.add_argument("--procs", type=int, default=24)
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
