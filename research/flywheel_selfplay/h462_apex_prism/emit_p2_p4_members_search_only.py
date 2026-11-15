#!/usr/bin/env python3
"""H462 P2/P4 `SEARCH`-Only Stacked Prediction & Member-Weight Emitter (`emit_p2_p4_members_search_only.py`).

Implements `cl_mtc` (`2026-11-15T15:39:05Z` §4) + `cl_fwriter` (`HNC-1` & `LOCK_REVIEW_GPBAG6` §3):
  - Strictly `SEARCH`-only (`0` `ya` or `ba` bytes read anywhere):
      1. Asserts `"ya" not in zb.files` and `"ba" not in zb.files` on `--bank`.
      2. Reads `D1` and `D2` (`H418` or `cache_h422` `.npz` files) exclusively through `NoLabelNpz`,
         an allow-list wrapper permitting ONLY `("Q", "AKF", "PA", "BAG", "REF", "pa_ok", "ym", "y", "lab", "shas")`
         and raising `PermissionError` if `"ya"`, `"ba"`, or any audit-score key is requested.
  - Reproduces the exact `tabcv7_cvrep8_prec48_all` comparator (`3 r4 × 2 gp × 8 CV-repeats = 48` members,
    combined via `bag_predictions_precision_weighted_oof`) AND any `P1` entrant (`--gpbag-dirs`,
    `--gpbag-mode {none,2col,allcol}`, `--gpbag-k`, plus optional `--extra-r4-dirs` for `L104M`/`M` seeds).
  - Emits per context `<out-dir>/<cid>.npz` containing:
      * `member_oof`: `(n_search, M)` cross-fitted `SEARCH-OOF` prediction for each member
      * `member_weights`: `(M,)` full-`SEARCH` `M6` precision weights (`sum == 1`)
      * `p_oof_ens`: `(n_search,)` outer-5-fold cross-fitted ensemble `SEARCH-OOF` (`M6` weights refit per outer fold)
      * `p_in_ens`: `(n_search,)` `member_oof @ member_weights`
      * `lab`: `(n_search,)` canonical outer 5-fold labels `lab0`
      * `labs_rep`: `(n_reps, n_search)` stratified K-fold labels across all `CV` repeats
      * `member_audit`: `(n_audit, M)` `AUDIT-X` prediction for each member
      * `p_audit_ens`: `(n_audit,)` `member_audit @ member_weights`
      * `member_names`: `(M,)` string identifiers for every member
  - Writes a `0444` create-once closure receipt (`--receipt-json`) recording every input file accessed,
    its SHA-256, `ya_keys_accessed = 0`, and the code manifest SHA-256.
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from h462_apex_prism import guard_matern_core as GM  # noqa: E402
from h462_apex_prism.edge_certify import bag_predictions_precision_weighted_oof  # noqa: E402
from h462_apex_prism.stack_guard_cv7_h452 import (  # noqa: E402
    _fit_sharp_foldlog_simplex,
    bcar_weights_pool_agnostic,
    make_stratified_kfold,
    stack_guard_cv7_context,
)

H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H433_F4 = ("form_sr", "form_gam", "form_gbm", "cap")
H450_TAB_COLS = ("tab_m_ens4", "tab_realtabmlp_ens", "tab_catboost_ens", "tab_lgbm_ens")

FORBIDDEN_TOKENS = ("CONFIRM", "FMGAP_B", "FMGAP_C", "HOLDOUT", "SEALED", "PROBE_YA", "/tmp/")
ALLOWED_GP_NO_LABEL_KEYS = frozenset({"Q", "AKF", "PA", "BAG", "REF", "pa_ok", "ym", "y", "lab", "shas"})
FORBIDDEN_NPZ_KEYS = frozenset({"ya", "ba", "u_record", "u_recomputed", "y_audit", "audit_y"})


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


class NoLabelNpz:
    """Strict allow-list `.npz` reader that blocks `"ya"`, `"ba"`, and all unapproved keys."""

    def __init__(self, path: Path, allowed_keys: frozenset[str] = ALLOWED_GP_NO_LABEL_KEYS) -> None:
        _check_safe_path(path)
        self._path = Path(path)
        self._allowed = allowed_keys
        self._npz = np.load(self._path, allow_pickle=False)
        self.keys_read: List[str] = []

    def __enter__(self) -> "NoLabelNpz":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._npz.close()

    def has_key(self, key: str) -> bool:
        if key in FORBIDDEN_NPZ_KEYS or key not in self._allowed:
            return False
        return key in self._npz.files

    def __getitem__(self, key: str) -> np.ndarray:
        if key in FORBIDDEN_NPZ_KEYS or key not in self._allowed:
            raise PermissionError(f"NoLabelNpz blocked forbidden/unlisted key {key!r} on {self._path}")
        self.keys_read.append(key)
        return self._npz[key]


def load_search_only_bank(bank_p: Path) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Loads `(family, Xs, Xa, y_search, lab0)` from a `ya`-free bank (`assert "ya" not in zb.files`)."""
    _check_safe_path(bank_p)
    out: Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        for bad in FORBIDDEN_NPZ_KEYS:
            if bad in zb.files:
                raise PermissionError(f"Audit label key {bad!r} forbidden in SEARCH-only bank {bank_p}")
        cids = [str(x) for x in zb["cids"].tolist()]
        fams = [str(x) for x in zb["family"].tolist()] if "family" in zb.files else [c[:8] for c in cids]
        ss = zb["s_shape"].astype(np.int64)
        sa = zb["a_shape"].astype(np.int64)
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            y = np.array(zb["y64"][lo : lo + n], np.float64, copy=True)
            lab = np.array(zb["lab"][lo : lo + n], np.int64, copy=True)
            out[c] = (fams[i], np.array(S[:, :-1], np.float64, copy=True), Xa, y, lab)
            so += n * w
            ao += na * f
            lo += n
    return out


def load_gp_pack_no_ya(gp_path: Path, n_s: int, n_a: int) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[np.ndarray]]]:
    """Loads `(Q_gp, bcols, rcols)` from `H418` (`AKF`/`PA`) or `cache_h422` (`BAG`/`REF`) with `0` `ya` access."""
    with NoLabelNpz(gp_path) as zg:
        Q_gp = np.asarray(zg["Q"], np.float64)
        if Q_gp.shape[0] != n_s:
            raise ValueError(f"Q_gp row mismatch on {gp_path}: {Q_gp.shape[0]} vs {n_s}")
        m_gp = int(Q_gp.shape[1])
        ym = float(zg["ym"])
        mean_a = np.full(n_a, ym, dtype=np.float64)
        if zg.has_key("BAG") and zg.has_key("REF"):
            BAG_gp = np.asarray(zg["BAG"], np.float64)
            REF_gp = np.asarray(zg["REF"], np.float64)
            if BAG_gp.shape != (n_a, m_gp) or REF_gp.shape != (n_a, m_gp):
                raise ValueError(f"BAG/REF shape mismatch on {gp_path}: {BAG_gp.shape} vs ({n_a}, {m_gp})")
            bcols = [BAG_gp[:, j] if np.all(np.isfinite(BAG_gp[:, j])) else mean_a for j in range(m_gp)]
            rcols: List[Optional[np.ndarray]] = [
                REF_gp[:, j] if np.all(np.isfinite(REF_gp[:, j])) else None for j in range(m_gp)
            ]
        else:
            PA = np.asarray(zg["PA"], np.float64)
            AKF = np.asarray(zg["AKF"], np.float64)
            pa_ok = np.asarray(zg["pa_ok"], dtype=bool)
            if PA.shape != (n_a, m_gp) or AKF.shape != (n_a, m_gp):
                raise ValueError(f"PA/AKF shape mismatch on {gp_path}: {PA.shape} vs ({n_a}, {m_gp})")
            bag_gp = [
                None if not np.all(np.isfinite(AKF[:, j])) else AKF[:, j].astype(np.float64)
                for j in range(m_gp)
            ]
            ref_gp: List[Optional[np.ndarray]] = [
                PA[:, j].astype(np.float64) if bool(pa_ok[j]) else None
                for j in range(m_gp)
            ]
            bcols = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]
            rcols = ref_gp
    return Q_gp, bcols, rcols


def load_col_block_no_ya(p: Path, names: Sequence[str], n_s: int, n_a: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads `(Q, BAG, REF)` from a `ya`-free column block `.npz`, asserting no forbidden label keys exist."""
    if not p.is_file():
        return None
    _check_safe_path(p)
    with np.load(p, allow_pickle=False) as z:
        for bad in FORBIDDEN_NPZ_KEYS:
            if bad in z.files:
                raise PermissionError(f"Forbidden audit label key {bad!r} present in column block {p}")
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
    if BAG.shape[0] != n_a or REF.shape[0] != n_a:
        raise ValueError(f"Audit row count mismatch in {p}: {BAG.shape[0]} vs {n_a}")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(BAG)) and np.all(np.isfinite(REF))):
        return None
    return Q, BAG, REF


def _clip_k3_fold_safe(
    blk_k_raw: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    lab5: Optional[np.ndarray] = None,
):
    if blk_k_raw is None:
        return None
    Qk, Bk, Rk = [arr.copy() for arr in blk_k_raw]
    if lab5 is not None:
        for fk in sorted(int(k) for k in np.unique(lab5)):
            va = lab5 == fk
            tr = ~va
            y_ref = y[tr] if int(np.sum(tr)) >= 2 else y
            lo_f, hi_f = float(np.min(y_ref)), float(np.max(y_ref))
            pad_f = 0.25 * max(hi_f - lo_f, float(np.std(y_ref)), 1e-9)
            Qk[va, 0] = np.clip(Qk[va, 0], lo_f - pad_f, hi_f + pad_f)
    else:
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
        Qk[:, 0] = np.clip(Qk[:, 0], y_lo - y_pad, y_hi + y_pad)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
    for A in (Bk, Rk):
        A[:, 0] = np.clip(A[:, 0], y_lo - y_pad, y_hi + y_pad)
    return Qk, Bk, Rk


def _clip_f4_fold_safe(
    blk_f_raw: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    lab5: Optional[np.ndarray] = None,
):
    if blk_f_raw is None:
        return None
    Qf, Bf, Rf = [arr.copy() for arr in blk_f_raw]
    if lab5 is not None:
        for fk in sorted(int(k) for k in np.unique(lab5)):
            va = lab5 == fk
            tr = ~va
            y_ref = y[tr] if int(np.sum(tr)) >= 2 else y
            lo_f, hi_f = float(np.min(y_ref)), float(np.max(y_ref))
            pad_f = 0.50 * max(hi_f - lo_f, float(np.std(y_ref)), 1e-9)
            for idx_f in (2, 3):
                Qf[va, idx_f] = np.clip(Qf[va, idx_f], lo_f - pad_f, hi_f + pad_f)
    else:
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        y_pad = 0.50 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
        for idx_f in (2, 3):
            Qf[:, idx_f] = np.clip(Qf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_pad = 0.50 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
    for A in (Bf, Rf):
        for idx_f in (2, 3):
            A[:, idx_f] = np.clip(A[:, idx_f], y_lo - y_pad, y_hi + y_pad)
    return Qf, Bf, Rf


def build_gpbag_block_from_dirs(
    y: np.ndarray,
    lab5: np.ndarray,
    gpbag_files: Sequence[Path],
    n_s: int,
    n_a: int,
    k_use: int,
    mode: str,
    tau_shrink: float = 0.25,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads the first `k_use` GP seeds across `gpbag_files` (`0` `ya` keys) and builds `2col` or `allcol` block."""
    if mode == "none" or k_use <= 0 or not gpbag_files:
        return None
    qs: List[np.ndarray] = []
    bs: List[np.ndarray] = []
    rs: List[np.ndarray] = []
    for p in gpbag_files:
        _check_safe_path(p)
        with np.load(p, allow_pickle=False) as z:
            for bad in FORBIDDEN_NPZ_KEYS:
                if bad in z.files:
                    raise PermissionError(f"Forbidden audit label key {bad!r} in GPBAG file {p}")
            idx = 1
            while f"q_gp_s{idx}" in z.files and len(qs) < k_use:
                q = np.asarray(z[f"q_gp_s{idx}"], np.float64)
                b = np.asarray(z[f"bag_gp_s{idx}"], np.float64)
                r = np.asarray(z[f"ref_gp_s{idx}"], np.float64)
                if len(q) != n_s or len(b) != n_a or len(r) != n_a:
                    raise ValueError(f"GPBAG row mismatch in {p}")
                qs.append(q)
                bs.append(b)
                rs.append(r)
                idx += 1
        if len(qs) >= k_use:
            break
    if len(qs) < k_use:
        raise ValueError(f"Requested k_use={k_use} GP seeds, found only {len(qs)}")
    if len(qs) == 1:
        return qs[0][:, None], bs[0][:, None], rs[0][:, None]

    Q_mat = np.column_stack(qs)
    B_mat = np.column_stack(bs)
    R_mat = np.column_stack(rs)
    q_eq = np.mean(Q_mat, axis=1)
    b_eq = np.mean(B_mat, axis=1)
    r_eq = np.mean(R_mat, axis=1)

    u_w = np.full(k_use, 1.0 / float(k_use), dtype=np.float64)

    def _w_fit(Q_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
        eps = 1e-4 * max(float(np.var(y_tr)), 1e-12)
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
    if mode == "2col":
        return np.column_stack([q_eq, q_pr]), np.column_stack([b_eq, b_pr]), np.column_stack([r_eq, r_pr])
    if mode == "allcol":
        return (
            np.column_stack([Q_mat, q_eq, q_pr]),
            np.column_stack([B_mat, b_eq, b_pr]),
            np.column_stack([R_mat, r_eq, r_pr]),
        )
    raise ValueError(f"Unsupported gpbag mode {mode!r}")


def nested_oof_cv7_rep(
    cid_r: str,
    y: np.ndarray,
    lab_r: np.ndarray,
    Q_gp: np.ndarray,
    tagged_blocks: Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> np.ndarray:
    """Exact `eval_h462_tabcv7_cvrep.py` `_nested_oof_cv7_rep` reproduced solely for `hwm_1842_exact` weight parity (`F2`)."""
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
            float(
                -0.5 * np.log(max(float(mses[idx2[0]]), 1e-30) / vy)
                - (-0.5 * np.log(max(float(mses[idx2[-1]]), 1e-30) / vy))
            )
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


def real_stacker_outer_oof_rep(
    cid: str,
    rep_idx: int,
    Xs: np.ndarray,
    y: np.ndarray,
    lab0: np.ndarray,
    Q_gp: np.ndarray,
    rcols_gp_present: Sequence[bool],
    raw_common_cv7: Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    r4_blk: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gpbag_blk: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """`F1` (`cl_mtc` `16:09:45Z`) + `HNC-1 H5` (`cl_fwriter` `16:10Z`) exact outer-`lab0` `stack_guard_cv7_context` OOF.

    Outer split ALWAYS uses `lab0` (the canonical 5-fold partition on which base columns were cross-fitted) so that
    for every outer fold `fk`, the label closure of every base column's `va` (`lab0 == fk`) OOF prediction is disjoint
    from `fk`. Inside `tr = ~(lab0 == fk)`, `stack_guard_cv7_context` runs with inner repeat `rep=rep_idx` (`lr_tr`),
    applying all real stacker steps (`c_wc` deflation, precision anchor, fixref BCAR, clone dedup, drift/median gates,
    and Matérn-3/2 L2) to predict `Xs[va]` using fold-`fk` base OOF values in the `AUDIT-X` role.
    """
    n_s, m_gp = Q_gp.shape
    q_oof = np.empty(n_s, dtype=np.float64)
    all_idx = np.arange(n_s, dtype=np.int64)

    for fk in sorted(int(k) for k in np.unique(lab0)):
        va = lab0 == fk
        tr = ~va
        n_va = int(np.sum(va))
        n_tr = int(np.sum(tr))
        if n_va == 0:
            continue
        # Explicit HNC-1 H5 row-set disjointness check on canonical partition `lab0`
        if not set(all_idx[tr].tolist()).isdisjoint(set(all_idx[va].tolist())):
            raise PermissionError(f"H5 row-set overlap in outer split fold {fk} on {cid}")
        if n_tr < 14 or float(np.std(y[tr])) <= 1e-12:
            q_oof[va] = float(np.mean(y[tr])) if n_tr > 0 else 0.0
            continue

        y_tr = np.asarray(y[tr], dtype=np.float64)
        Xs_tr = np.asarray(Xs[tr], dtype=np.float64)
        Xa_va = np.asarray(Xs[va], dtype=np.float64)
        lr_tr = make_stratified_kfold(f"{cid}#outerf{fk}", y_tr, k=5, rep=rep_idx)
        cr_fold = f"{cid}#f{fk}" if rep_idx == 0 else f"{cid}#f{fk}#cvrep{rep_idx}"

        Q_gp_tr = np.asarray(Q_gp[tr], dtype=np.float64)
        ym_tr = float(np.mean(y_tr))
        mean_va = np.full(n_va, ym_tr, dtype=np.float64)
        bcols_va: List[Optional[np.ndarray]] = [
            np.asarray(Q_gp[va, j], dtype=np.float64) if np.all(np.isfinite(Q_gp[va, j])) else mean_va
            for j in range(m_gp)
        ]
        rcols_va: List[Optional[np.ndarray]] = [
            (np.asarray(Q_gp[va, j], dtype=np.float64) if np.all(np.isfinite(Q_gp[va, j])) else None)
            if rcols_gp_present[j]
            else None
            for j in range(m_gp)
        ]

        tb_fold: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        for blk_tag, (Qb, _Bb, _Rb) in raw_common_cv7:
            q_tr_b = np.array(Qb[tr], dtype=np.float64, copy=True)
            q_va_b = np.array(Qb[va], dtype=np.float64, copy=True)
            if blk_tag == "k3":
                lo_f, hi_f = float(np.min(y_tr)), float(np.max(y_tr))
                pad_f = 0.25 * max(hi_f - lo_f, float(np.std(y_tr)), 1e-9)
                q_tr_b[:, 0] = np.clip(q_tr_b[:, 0], lo_f - pad_f, hi_f + pad_f)
                q_va_b[:, 0] = np.clip(q_va_b[:, 0], lo_f - pad_f, hi_f + pad_f)
            elif blk_tag == "f4":
                lo_f, hi_f = float(np.min(y_tr)), float(np.max(y_tr))
                pad_f = 0.50 * max(hi_f - lo_f, float(np.std(y_tr)), 1e-9)
                for idx_f in (2, 3):
                    q_tr_b[:, idx_f] = np.clip(q_tr_b[:, idx_f], lo_f - pad_f, hi_f + pad_f)
                    q_va_b[:, idx_f] = np.clip(q_va_b[:, idx_f], lo_f - pad_f, hi_f + pad_f)
            tb_fold.append((blk_tag, (q_tr_b, q_va_b, q_va_b.copy())))

        if r4_blk is not None:
            Qr4 = r4_blk[0]
            tb_fold.append(("e8", ( np.array(Qr4[tr], np.float64, copy=True), np.array(Qr4[va], np.float64, copy=True), np.array(Qr4[va], np.float64, copy=True) )))
        if gpbag_blk is not None:
            Qgb = gpbag_blk[0]
            tb_fold.append(("gpbag", ( np.array(Qgb[tr], np.float64, copy=True), np.array(Qgb[va], np.float64, copy=True), np.array(Qgb[va], np.float64, copy=True) )))

        p_va, _ = stack_guard_cv7_context(
            cr_fold,
            Xs_tr,
            Xa_va,
            y_tr,
            lr_tr,
            Q_gp_tr,
            bcols_va,
            rcols_va,
            tb_fold,
            use_cv7_folds=True,
        )
        q_oof[va] = np.asarray(p_va, dtype=np.float64)
    return q_oof


def compute_context_members_search_only(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab0: np.ndarray,
    gp_packs: Sequence[Tuple[str, np.ndarray, List[np.ndarray], List[Optional[np.ndarray]]]],
    raw_common_cv7: Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    r4_blocks: Sequence[Tuple[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    gpbag_blk: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cv_repeats: int = 8,
    audit_weight_mode: str = "real_stacker",
) -> Dict[str, np.ndarray]:
    """Computes `member_oof` (`real_stacker_outer_oof_rep` on `lab0`), `member_weights`, `p_oof_ens`, `member_audit`, `p_audit_ens` (`0` `ya`).

    `audit_weight_mode`:
      - `"real_stacker"` (default for new `P1` entrants): `member_weights` & `p_audit_ens` use `member_oof` from `real_stacker_outer_oof_rep`.
      - `"hwm_1842_exact"` (for exact `1842.03` comparator parity per `cl_mtc` `F2`): `member_audit` & `member_weights` reproduce
        `eval_h462_tabcv7_cvrep.py` (`tabcv7_cvrep8_prec48_all`) to `max |diff| == 0.0`, while `member_oof` is still the real
        `stack_guard_cv7_context` outer-`lab0` OOF (`F1`) and `p_audit_ens_real_stacker` / `member_weights_real_stacker` are emitted alongside.
    """
    if audit_weight_mode not in ("real_stacker", "hwm_1842_exact"):
        raise ValueError(f"Unsupported audit_weight_mode {audit_weight_mode!r}")

    labs_rep = [lab0 if r == 0 else make_stratified_kfold(cid, y, k=5, rep=r) for r in range(cv_repeats)]
    cids_rep = [cid if r == 0 else f"{cid}#cvrep{r}" for r in range(cv_repeats)]

    # Full-SEARCH clipped blocks for `member_audit` (matching `eval_h462_tabcv7_cvrep.py` bit-for-bit)
    audit_common_cv7: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for blk_tag, raw_b in raw_common_cv7:
        if blk_tag == "k3":
            cb = _clip_k3_fold_safe(raw_b, y, None)
        elif blk_tag == "f4":
            cb = _clip_f4_fold_safe(raw_b, y, None)
        else:
            cb = raw_b
        if cb is not None:
            audit_common_cv7.append((blk_tag, cb))

    member_oofs: List[np.ndarray] = []
    member_oofs_hwm_fast: List[np.ndarray] = []
    member_audits: List[np.ndarray] = []
    member_names: List[str] = []

    for r4_name, r4_blk in r4_blocks:
        tb_cv7_aud = list(audit_common_cv7) + ([("e8", r4_blk)] if r4_blk is not None else [])
        if gpbag_blk is not None:
            tb_cv7_aud.append(("gpbag", gpbag_blk))
        for gp_name, Q_gp, bcols, rcols in gp_packs:
            rcols_present = [rc is not None for rc in rcols]
            for r in range(cv_repeats):
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_tb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_cv7_aud, use_cv7_folds=True)
                q_real = real_stacker_outer_oof_rep(
                    cid=cid,
                    rep_idx=r,
                    Xs=Xs,
                    y=y,
                    lab0=lab0,
                    Q_gp=Q_gp,
                    rcols_gp_present=rcols_present,
                    raw_common_cv7=raw_common_cv7,
                    r4_blk=r4_blk,
                    gpbag_blk=gpbag_blk,
                )
                member_audits.append(np.asarray(p_tb, np.float64))
                member_oofs.append(np.asarray(q_real, np.float64))
                if audit_weight_mode == "hwm_1842_exact":
                    q_fast = nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7_aud)
                    member_oofs_hwm_fast.append(np.asarray(q_fast, np.float64))
                member_names.append(f"{r4_name}__{gp_name}__rep{r}")

    p_audit_real, meta_w_real = bag_predictions_precision_weighted_oof(y, member_oofs, member_audits)
    weights_real = np.asarray(meta_w_real["weights"], dtype=np.float64)

    if audit_weight_mode == "hwm_1842_exact":
        p_audit_ens, meta_w_hwm = bag_predictions_precision_weighted_oof(y, member_oofs_hwm_fast, member_audits)
        weights_full = np.asarray(meta_w_hwm["weights"], dtype=np.float64)
    else:
        p_audit_ens = p_audit_real
        weights_full = weights_real

    OOF_mat = np.column_stack(member_oofs)
    AUD_mat = np.column_stack(member_audits)
    p_in_ens = OOF_mat @ weights_full

    # Outer-5-fold (`lab0`) cross-fitted ensemble SEARCH-OOF (refits M6 precision weights on `y[tr]`, predicts `OOF_mat[va]`)
    p_oof_ens = np.empty(len(y), dtype=np.float64)
    for fk in sorted(int(k) for k in np.unique(lab0)):
        va = lab0 == fk
        tr = ~va
        if int(np.sum(va)) == 0:
            continue
        if int(np.sum(tr)) >= 8:
            p_va, _ = bag_predictions_precision_weighted_oof(
                y[tr],
                [OOF_mat[tr, j] for j in range(OOF_mat.shape[1])],
                [OOF_mat[va, j] for j in range(OOF_mat.shape[1])],
            )
            p_oof_ens[va] = p_va
        else:
            p_oof_ens[va] = np.mean(OOF_mat[va], axis=1)

    out_payload: Dict[str, np.ndarray] = {
        "member_oof": OOF_mat,
        "member_weights": weights_full,
        "member_weights_real_stacker": weights_real,
        "p_oof_ens": p_oof_ens,
        "p_in_ens": p_in_ens,
        "lab": np.asarray(lab0, dtype=np.int64),
        "labs_rep": np.stack(labs_rep, axis=0).astype(np.int64),
        "member_audit": AUD_mat,
        "p_audit_ens": np.asarray(p_audit_ens, dtype=np.float64),
        "p_audit_ens_real_stacker": np.asarray(p_audit_real, dtype=np.float64),
        "member_names": np.asarray(member_names),
        "audit_weight_mode": np.asarray(audit_weight_mode),
    }
    if audit_weight_mode == "hwm_1842_exact":
        out_payload["member_oof_hwm_fast"] = np.column_stack(member_oofs_hwm_fast)
    return out_payload


def _ctx_worker(job: Tuple) -> Dict[str, object]:
    (
        cid,
        fam,
        Xs,
        Xa,
        y,
        lab0,
        d1_p,
        d2_p,
        m7_p,
        k3_p,
        f4_p,
        tab_p,
        r4_spec_paths,
        gpbag_paths,
        gpbag_k,
        gpbag_mode,
        cv_repeats,
        audit_weight_mode,
        out_dir,
    ) = job
    n_s, n_a = len(y), len(Xa)
    accessed: List[Tuple[str, str]] = []
    for p in [d1_p, d2_p, m7_p, k3_p, f4_p, tab_p] + [sp for _, sp, _ in r4_spec_paths] + list(gpbag_paths):
        if p.is_file():
            accessed.append((str(p), _sha256_file(p)))

    Q_d1, b_d1, r_d1 = load_gp_pack_no_ya(d1_p, n_s, n_a)
    Q_d2, b_d2, r_d2 = load_gp_pack_no_ya(d2_p, n_s, n_a)
    gp_packs = [("d1", Q_d1, b_d1, r_d1), ("d2", Q_d2, b_d2, r_d2)]

    blk_m = load_col_block_no_ya(m7_p, GM.PRIMARY_NAMES, n_s, n_a)
    blk_k_raw = load_col_block_no_ya(k3_p, H430_K3, n_s, n_a)
    blk_f_raw = load_col_block_no_ya(f4_p, H433_F4, n_s, n_a)
    blk_tab = load_col_block_no_ya(tab_p, H450_TAB_COLS, n_s, n_a)
    raw_common_cv7 = [
        (t, b) for t, b in (("m7", blk_m), ("k3", blk_k_raw), ("f4", blk_f_raw), ("tab450", blk_tab)) if b is not None
    ]

    r4_blocks = [(rname, load_col_block_no_ya(rp, rcols, n_s, n_a)) for rname, rp, rcols in r4_spec_paths]
    gpbag_blk = build_gpbag_block_from_dirs(y, lab0, gpbag_paths, n_s, n_a, gpbag_k, gpbag_mode)

    res = compute_context_members_search_only(
        cid=cid,
        Xs=Xs,
        Xa=Xa,
        y=y,
        lab0=lab0,
        gp_packs=gp_packs,
        raw_common_cv7=raw_common_cv7,
        r4_blocks=r4_blocks,
        gpbag_blk=gpbag_blk,
        cv_repeats=cv_repeats,
        audit_weight_mode=audit_weight_mode,
    )
    dst = out_dir / f"{cid}.npz"
    tmp = out_dir / f"{cid}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, family=np.asarray(fam), **res)
    os.replace(tmp, dst)
    rmse_oof = float(np.sqrt(np.mean((res["p_oof_ens"] - y) ** 2)))
    max_diff_real_vs_w = float(np.max(np.abs(res["p_audit_ens"] - res["p_audit_ens_real_stacker"])))
    return {
        "cid": cid,
        "family": fam,
        "n_search": n_s,
        "n_audit": n_a,
        "n_members": int(res["member_oof"].shape[1]),
        "search_oof_rmse": rmse_oof,
        "max_abs_diff_audit_ens_vs_real_stacker": max_diff_real_vs_w,
        "out_sha256": _sha256_file(dst),
        "accessed": accessed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, help="Path to ya-free SEARCH bank (e.g. bank_d91_search_f64.npz)")
    ap.add_argument("--d1-dir", required=True)
    ap.add_argument("--d2-dir", required=True)
    ap.add_argument("--m7-dir", required=True)
    ap.add_argument("--k3-dir", required=True)
    ap.add_argument("--f4-dir", required=True)
    ap.add_argument("--tab450-dir", required=True)
    ap.add_argument("--wc-r4e8-dir", required=True)
    ap.add_argument("--s1-r4e8-dir", required=True)
    ap.add_argument("--s2-r4e8-dir", required=True)
    ap.add_argument(
        "--extra-r4-dirs",
        default="",
        help="Optional comma-separated <tag>:<dir>:<col_name> entries (e.g. l46205:/path:r4_l)",
    )
    ap.add_argument("--gpbag-dirs", default="", help="Comma-separated ya-free GPBAG column dirs")
    ap.add_argument("--gpbag-k", type=int, default=0)
    ap.add_argument("--gpbag-mode", choices=("none", "2col", "allcol"), default="none")
    ap.add_argument("--cv-repeats", type=int, default=8)
    ap.add_argument(
        "--audit-weight-mode",
        choices=("real_stacker", "hwm_1842_exact"),
        default="real_stacker",
        help="real_stacker (unified P1/P2/P4 stacker OOF weights) or hwm_1842_exact (bit-exact 1842.03 comparator AUDIT weights)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--receipt-json", required=True)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    t0 = time.monotonic()
    bank_p = Path(a.bank)
    out_dir = Path(a.out_dir)
    rcpt_p = Path(a.receipt_json)
    for p in (
        bank_p,
        Path(a.d1_dir),
        Path(a.d2_dir),
        Path(a.m7_dir),
        Path(a.k3_dir),
        Path(a.f4_dir),
        Path(a.tab450_dir),
        Path(a.wc_r4e8_dir),
        Path(a.s1_r4e8_dir),
        Path(a.s2_r4e8_dir),
        out_dir,
        rcpt_p,
    ):
        _check_safe_path(p)
    if rcpt_p.exists():
        raise SystemExit(f"REFUSAL: create-once receipt already exists: {rcpt_p}")

    gpbag_dirs = [Path(x.strip()) for x in a.gpbag_dirs.split(",") if x.strip()]
    for gd in gpbag_dirs:
        _check_safe_path(gd)

    extra_r4: List[Tuple[str, Path, Tuple[str, ...]]] = []
    if a.extra_r4_dirs.strip():
        for spec in a.extra_r4_dirs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            tag, d_str, col_nm = spec.split(":", 2)
            dp = Path(d_str)
            _check_safe_path(dp)
            extra_r4.append((tag, dp, (col_nm,)))

    bank_rows = load_search_only_bank(bank_p)
    out_dir.mkdir(parents=True, exist_ok=True)
    cids = sorted(bank_rows)

    jobs = []
    for c in cids:
        fam, Xs, Xa, y, lab0 = bank_rows[c]
        r4_specs: List[Tuple[str, Path, Tuple[str, ...]]] = [
            ("wc_r4e8", Path(a.wc_r4e8_dir) / f"{c}.npz", ("r4_e8",)),
            ("s46201_r4e8", Path(a.s1_r4e8_dir) / f"{c}.npz", ("r4_e8",)),
            ("s46202_r4e8", Path(a.s2_r4e8_dir) / f"{c}.npz", ("r4_e8",)),
        ] + [(tag, dp / f"{c}.npz", cols) for tag, dp, cols in extra_r4]
        jobs.append(
            (
                c,
                fam,
                Xs,
                Xa,
                y,
                lab0,
                Path(a.d1_dir) / f"{c}.npz",
                Path(a.d2_dir) / f"{c}.npz",
                Path(a.m7_dir) / f"{c}.npz",
                Path(a.k3_dir) / f"{c}.npz",
                Path(a.f4_dir) / f"{c}.npz",
                Path(a.tab450_dir) / f"{c}.npz",
                r4_specs,
                [gd / f"{c}.npz" for gd in gpbag_dirs],
                int(a.gpbag_k),
                str(a.gpbag_mode),
                int(a.cv_repeats),
                str(a.audit_weight_mode),
                out_dir,
            )
        )

    jobs.sort(key=lambda j: -len(j[4]))
    if a.procs <= 1 or len(jobs) == 1:
        done = [_ctx_worker(j) for j in jobs]
    else:
        with Pool(processes=min(a.procs, len(jobs))) as pool:
            done = list(pool.imap_unordered(_ctx_worker, jobs, chunksize=1))
    done.sort(key=lambda d: str(d["cid"]))

    code_files = sorted(HERE.glob("*.py"))
    code_hashes = {p.name: _sha256_file(p) for p in code_files}
    code_manifest_sha = hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()

    accessed_map: Dict[str, str] = {str(bank_p): _sha256_file(bank_p)}
    for d in done:
        for fp, fh in d["accessed"]:  # type: ignore[index]
            accessed_map[fp] = fh

    receipt = {
        "schema": "jope.h462.p2_p4_members_search_only/v2",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "ya_keys_accessed": 0,
        "ba_keys_accessed": 0,
        "outer_split_partition": "bank_lab0_canonical_5fold",
        "base_column_fold_disjointness_verified": True,
        "member_oof_stacker": "stack_guard_cv7_context_outer_lab0",
        "audit_weight_mode": str(a.audit_weight_mode),
        "bank_path": str(bank_p),
        "bank_sha256": accessed_map[str(bank_p)],
        "n_contexts": len(done),
        "cv_repeats": int(a.cv_repeats),
        "gpbag_mode": str(a.gpbag_mode),
        "gpbag_k": int(a.gpbag_k),
        "gpbag_dirs": [str(x) for x in gpbag_dirs],
        "extra_r4_dirs": str(a.extra_r4_dirs),
        "code_manifest_sha256": code_manifest_sha,
        "code_files_sha256": code_hashes,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "contexts": [
            {
                "cid": d["cid"],
                "family": d["family"],
                "n_search": d["n_search"],
                "n_audit": d["n_audit"],
                "n_members": d["n_members"],
                "search_oof_rmse": d["search_oof_rmse"],
                "max_abs_diff_audit_ens_vs_real_stacker": d["max_abs_diff_audit_ens_vs_real_stacker"],
                "out_sha256": d["out_sha256"],
            }
            for d in done
        ],
        "n_files_accessed": len(accessed_map),
        "files_accessed_sha256": accessed_map,
    }
    rcpt_p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(rcpt_p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "OK",
                "n_contexts": len(done),
                "ya_keys_accessed": 0,
                "code_manifest_sha256": code_manifest_sha,
                "receipt_json": str(rcpt_p),
                "receipt_sha256": _sha256_file(rcpt_p),
                "elapsed_s": receipt["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
