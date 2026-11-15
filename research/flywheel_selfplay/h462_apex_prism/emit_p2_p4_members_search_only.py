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
from h462_apex_prism.contracts_v3 import verify_nested_oof_fn  # noqa: E402
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


def assert_same_outer_folds(stored_lab: np.ndarray, lab0: np.ndarray, source: Path | str) -> Dict[int, int]:
    """`G2` (`cl_mtc` `16:27:16Z`) + `gp_chain_h458.py:258-260` exact outer-fold row-set equality check.

    Compares partitions as sets of row indices rather than fold label integers and returns the mapping
    `{fk_in_lab0: fk_in_stored_lab}` so `nest_q` rows can be indexed deterministically by `lab0` fold id.
    """
    sl = np.asarray(stored_lab, dtype=np.int64).reshape(-1)
    l0 = np.asarray(lab0, dtype=np.int64).reshape(-1)
    if sl.shape != l0.shape:
        raise PermissionError(f"H5 partition length mismatch on {source}: {sl.shape} vs {l0.shape}")
    s_folds = sorted(int(k) for k in np.unique(sl))
    l_folds = sorted(int(k) for k in np.unique(l0))
    s_map = {frozenset(np.flatnonzero(sl == o).tolist()): o for o in s_folds}
    l_map = {frozenset(np.flatnonzero(l0 == o).tolist()): o for o in l_folds}
    if set(s_map.keys()) != set(l_map.keys()):
        raise PermissionError(
            f"H5 outer-fold row-set mismatch on {source}: stored partition differs from canonical bank lab0"
        )
    return {fk_l0: s_map[rset] for rset, fk_l0 in l_map.items()}


def load_gp_pack_no_ya(
    gp_path: Path,
    n_s: int,
    n_a: int,
    lab0: Optional[np.ndarray] = None,
    require_lab: bool = False,
    keys_sink: Optional[set[str]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[np.ndarray]]]:
    """Loads `(Q_gp, bcols, rcols)` from `H418` (`AKF`/`PA`) or `cache_h422` (`BAG`/`REF`) with `0` `ya` access."""
    with NoLabelNpz(gp_path) as zg:
        if lab0 is not None:
            if zg.has_key("lab"):
                assert_same_outer_folds(np.asarray(zg["lab"], np.int64), lab0, gp_path)
            elif require_lab:
                raise PermissionError(f"H5 GP pack {gp_path} missing required 'lab' partition key")
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
        if keys_sink is not None:
            keys_sink.update(zg.keys_read)
    return Q_gp, bcols, rcols


def load_col_block_no_ya(
    p: Path,
    names: Sequence[str],
    n_s: int,
    n_a: int,
    lab0: Optional[np.ndarray] = None,
    keys_sink: Optional[set[str]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads `(Q, BAG, REF)` from a `ya`-free column block `.npz`, asserting no forbidden label keys exist."""
    res = load_col_block_no_ya_with_nest(p, names, n_s, n_a, lab0=lab0, keys_sink=keys_sink)
    return None if res is None else res[0]


def load_col_block_no_ya_with_nest(
    p: Path,
    names: Sequence[str],
    n_s: int,
    n_a: int,
    lab0: Optional[np.ndarray] = None,
    keys_sink: Optional[set[str]] = None,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[np.ndarray]]]:
    """Loads `((Q, BAG, REF), nest_Q)` from a `ya`-free column block `.npz`.

    When `lab0` is supplied and `nest_lab_<v>`, `nest_q_<v>`, `nest_a_<v>` exist for every `v in names`
    (`gp_chain_h458.py:251-263` layout), verifies row-set equality against `lab0` and returns `nest_Q`
    of shape `(n_folds_lab0, n_s, len(names))` aligned to `sorted(unique(lab0))`.
    """
    if not p.is_file():
        return None
    _check_safe_path(p)
    nest_cols: List[np.ndarray] = []
    with np.load(p, allow_pickle=False) as z:
        for bad in FORBIDDEN_NPZ_KEYS:
            if bad in z.files:
                raise PermissionError(f"Forbidden audit label key {bad!r} present in column block {p}")
        if "ok" in z.files and not bool(z["ok"]):
            return None
        if lab0 is not None and "lab" in z.files:
            assert_same_outer_folds(np.asarray(z["lab"], np.int64), lab0, p)
            if keys_sink is not None:
                keys_sink.add("lab")
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
        if keys_sink is not None:
            for v in names:
                keys_sink.update((f"q_{v}", f"bag_{v}"))
                if f"ref_{v}" in z.files:
                    keys_sink.add(f"ref_{v}")
        if lab0 is not None and all(
            f"nest_lab_{v}" in z.files and f"nest_q_{v}" in z.files and f"nest_a_{v}" in z.files for v in names
        ):
            folds_l0 = sorted(int(k) for k in np.unique(lab0))
            for v in names:
                nl = np.asarray(z[f"nest_lab_{v}"], dtype=np.int64)
                nq = np.asarray(z[f"nest_q_{v}"], dtype=np.float64)
                na = np.asarray(z[f"nest_a_{v}"], dtype=np.float64)
                if keys_sink is not None:
                    keys_sink.update((f"nest_lab_{v}", f"nest_q_{v}", f"nest_a_{v}"))
                fmap = assert_same_outer_folds(nl, lab0, f"{p}:{v}")
                k_o = int(np.max(nl)) + 1
                if nq.shape != (k_o, n_s) or na.shape != (k_o, n_a) or not (np.all(np.isfinite(nq)) and np.all(np.isfinite(na))):
                    raise ValueError(f"Invalid nest_q/nest_a shape or non-finite values in {p}:{v}")
                nq_aligned = np.stack([nq[fmap[fk]] for fk in folds_l0], axis=0)
                nest_cols.append(nq_aligned)
    if BAG.shape[0] != n_a or REF.shape[0] != n_a:
        raise ValueError(f"Audit row count mismatch in {p}: {BAG.shape[0]} vs {n_a}")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(BAG)) and np.all(np.isfinite(REF))):
        return None
    nest_Q = np.stack(nest_cols, axis=-1) if nest_cols else None
    return (Q, BAG, REF), nest_Q


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
    cid: str = "ctx",
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[np.ndarray]]:
    """Loads the first `k_use` GP seeds across `gpbag_files` (`0` `ya` keys) and builds `(block, nest_Q_gpbag)`.

    When every seed carries `nest_lab_gp_s*`, `nest_q_gp_s*`, `nest_a_gp_s*` (`P2 v1.4 §3`), verifies row-set
    equality against `lab5` and builds `nest_Q_gpbag` of shape `(n_folds_lab5, n_s, m_cols)` where precision
    weights on outer fold `fk` are cross-fitted strictly inside `SEARCH \\ fk` (`max_abs_leak == 0.0`).
    """
    if mode == "none" or k_use <= 0 or not gpbag_files:
        return None, None
    qs: List[np.ndarray] = []
    bs: List[np.ndarray] = []
    rs: List[np.ndarray] = []
    nqs: List[np.ndarray] = []
    all_have_nest = True
    folds_l0 = sorted(int(x) for x in np.unique(lab5))

    for p in gpbag_files:
        _check_safe_path(p)
        with np.load(p, allow_pickle=False) as z:
            for bad in FORBIDDEN_NPZ_KEYS:
                if bad in z.files:
                    raise PermissionError(f"Forbidden audit label key {bad!r} in GPBAG file {p}")
            if "lab" in z.files:
                assert_same_outer_folds(np.asarray(z["lab"], np.int64), lab5, p)
            idx = 1
            while f"q_gp_s{idx}" in z.files and len(qs) < k_use:
                nm = f"gp_s{idx}"
                q = np.asarray(z[f"q_{nm}"], np.float64)
                b = np.asarray(z[f"bag_{nm}"], np.float64)
                r = np.asarray(z[f"ref_{nm}"], np.float64)
                if len(q) != n_s or len(b) != n_a or len(r) != n_a:
                    raise ValueError(f"GPBAG row mismatch in {p}")
                qs.append(q)
                bs.append(b)
                rs.append(r)
                if all_have_nest and all(k in z.files for k in (f"nest_lab_{nm}", f"nest_q_{nm}", f"nest_a_{nm}")):
                    nl = np.asarray(z[f"nest_lab_{nm}"], np.int64)
                    nq = np.asarray(z[f"nest_q_{nm}"], np.float64)
                    na = np.asarray(z[f"nest_a_{nm}"], np.float64)
                    fmap = assert_same_outer_folds(nl, lab5, f"{p}:{nm}")
                    k_o = int(np.max(nl)) + 1
                    if nq.shape != (k_o, n_s) or na.shape != (k_o, n_a) or not (np.all(np.isfinite(nq)) and np.all(np.isfinite(na))):
                        raise ValueError(f"Invalid nest_q/nest_a in {p}:{nm}")
                    nqs.append(np.stack([nq[fmap[fk]] for fk in folds_l0], axis=0))
                else:
                    all_have_nest = False
                idx += 1
        if len(qs) >= k_use:
            break
    if len(qs) < k_use:
        raise ValueError(f"Requested k_use={k_use} GP seeds, found only {len(qs)}")

    u_w = np.full(k_use, 1.0 / float(k_use), dtype=np.float64)

    def _w_fit(Q_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
        eps = 1e-4 * max(float(np.var(y_tr)), 1e-12)
        mse = np.mean((Q_tr - y_tr[:, None]) ** 2, axis=0)
        prec = 1.0 / (mse + eps)
        sp = float(np.sum(prec))
        w_raw = (prec / sp) if (sp > 0.0 and np.isfinite(sp)) else u_w
        w = (1.0 - tau_shrink) * w_raw + tau_shrink * u_w
        return w / float(np.sum(w))

    if len(qs) == 1:
        nest_1 = nqs[0][:, :, None] if (all_have_nest and len(nqs) == 1) else None
        return (qs[0][:, None], bs[0][:, None], rs[0][:, None]), nest_1

    Q_mat = np.column_stack(qs)
    B_mat = np.column_stack(bs)
    R_mat = np.column_stack(rs)
    q_eq = np.mean(Q_mat, axis=1)
    b_eq = np.mean(B_mat, axis=1)
    r_eq = np.mean(R_mat, axis=1)

    q_pr = np.empty(n_s, dtype=np.float64)
    for fk in folds_l0:
        va = lab5 == fk
        tr = ~va
        w_tr = _w_fit(Q_mat[tr], y[tr]) if int(np.sum(tr)) >= 10 else u_w
        q_pr[va] = Q_mat[va] @ w_tr

    w_full = _w_fit(Q_mat, y)
    b_pr = B_mat @ w_full
    r_pr = R_mat @ w_full

    nest_out: Optional[np.ndarray] = None
    if all_have_nest and len(nqs) == k_use:
        # Shape `(n_folds_l0, n_s, k_use)`
        NQ_seeds = np.stack(nqs, axis=-1)
        nq_eq = np.mean(NQ_seeds, axis=-1)
        nq_pr = np.empty((len(folds_l0), n_s), dtype=np.float64)
        for fpos, fk in enumerate(folds_l0):
            va = lab5 == fk
            tr = ~va
            n_tr = int(np.sum(tr))
            Q_in_f = NQ_seeds[fpos, tr, :]
            y_tr_f = y[tr]
            if n_tr < 10:
                nq_pr[fpos, :] = nq_eq[fpos, :]
                continue
            lab_in = make_stratified_kfold(f"{cid}_gpbag_nest_f{fk}", y_tr_f, k=5, rep=0)
            q_pr_tr = np.empty(n_tr, dtype=np.float64)
            for ifk in sorted(int(x) for x in np.unique(lab_in)):
                iva = lab_in == ifk
                itr = ~iva
                if int(np.sum(iva)) == 0:
                    continue
                w_itr = _w_fit(Q_in_f[itr], y_tr_f[itr]) if int(np.sum(itr)) >= 8 else u_w
                q_pr_tr[iva] = Q_in_f[iva] @ w_itr
            w_f = _w_fit(Q_in_f, y_tr_f)
            nq_pr[fpos, tr] = q_pr_tr
            if int(np.sum(va)) > 0:
                nq_pr[fpos, va] = NQ_seeds[fpos, va, :] @ w_f
        if mode == "2col":
            nest_out = np.stack([nq_eq, nq_pr], axis=-1)
        elif mode == "allcol":
            nest_out = np.concatenate([NQ_seeds, nq_eq[:, :, None], nq_pr[:, :, None]], axis=-1)

    if mode == "2col":
        return (np.column_stack([q_eq, q_pr]), np.column_stack([b_eq, b_pr]), np.column_stack([r_eq, r_pr])), nest_out
    if mode == "allcol":
        return (
            (
                np.column_stack([Q_mat, q_eq, q_pr]),
                np.column_stack([B_mat, b_eq, b_pr]),
                np.column_stack([R_mat, r_eq, r_pr]),
            ),
            nest_out,
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
    r4_nest_q: Optional[np.ndarray] = None,
    gpbag_nest_q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """`F1` (`cl_mtc` `16:09:45Z`) + `P2 v1.4 §3 / v1.5` exact outer-`lab0` `stack_guard_cv7_context` OOF.

    Outer split ALWAYS uses `lab0` (verified by `assert_same_outer_folds` at load time against stored `lab` / `nest_lab_*`).
    When `r4_nest_q` or `gpbag_nest_q` `(n_folds_lab0, n_s, m_cols)` is supplied (`nest_lab_ / nest_q_ / nest_a_` layout),
    outer fold `fpos` (`fk = folds_l0[fpos]`) receives `nest_q[fpos, tr]` on `SEARCH \\ fk` and `nest_q[fpos, va]` on `fk`,
    eliminating both first-order and second-order `SEARCH \\ g` stacking coupling (`max_abs_leak == 0.0`).
    """
    n_s, m_gp = Q_gp.shape
    q_oof = np.empty(n_s, dtype=np.float64)
    folds_l0 = sorted(int(k) for k in np.unique(lab0))

    for fpos, fk in enumerate(folds_l0):
        va = lab0 == fk
        tr = ~va
        n_va = int(np.sum(va))
        n_tr = int(np.sum(tr))
        if n_va == 0:
            continue
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
            if r4_nest_q is not None:
                q_r4_tr = np.array(r4_nest_q[fpos, tr], dtype=np.float64, copy=True)
                q_r4_va = np.array(r4_nest_q[fpos, va], dtype=np.float64, copy=True)
            else:
                Qr4 = r4_blk[0]
                q_r4_tr = np.array(Qr4[tr], dtype=np.float64, copy=True)
                q_r4_va = np.array(Qr4[va], dtype=np.float64, copy=True)
            tb_fold.append(("e8", (q_r4_tr, q_r4_va, q_r4_va.copy())))
        if gpbag_blk is not None:
            if gpbag_nest_q is not None:
                q_gb_tr = np.array(gpbag_nest_q[fpos, tr], dtype=np.float64, copy=True)
                q_gb_va = np.array(gpbag_nest_q[fpos, va], dtype=np.float64, copy=True)
            else:
                Qgb = gpbag_blk[0]
                q_gb_tr = np.array(Qgb[tr], dtype=np.float64, copy=True)
                q_gb_va = np.array(Qgb[va], dtype=np.float64, copy=True)
            tb_fold.append(("gpbag", (q_gb_tr, q_gb_va, q_gb_va.copy())))

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
    audit_weight_mode: str = "hwm_1842_exact",
    r4_nest_map: Optional[Dict[str, Optional[np.ndarray]]] = None,
    gpbag_nest_q: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Computes `member_oof` (`real_stacker_outer_oof_rep` on `lab0`), `member_oof_hwm_fast`, `member_audit`, `p_audit_ens` (`0` `ya`).

    `audit_weight_mode` (`G1` closed: defaults to `"hwm_1842_exact"` matching `eval_h462_p1_gpbag_scaling_fitsel.py`):
      - `"hwm_1842_exact"` (default): `member_weights` & `p_audit_ens` weight `member_audit` by precision on `member_oof_hwm_fast`
        (matching both `1842.03` HWM comparator and `P1` `FITSEL` selection to `max |diff| == 0.0`).
      - `"real_stacker"`: `member_weights` & `p_audit_ens` weight `member_audit` by precision on `member_oof` (`real_stacker_outer_oof_rep`).
      Both `member_oof` and `member_oof_hwm_fast` are ALWAYS written to `members/<cid>.npz` for `P2 v1.5`.
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
        r4_nq = None if r4_nest_map is None else r4_nest_map.get(r4_name)
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
                    r4_nest_q=r4_nq,
                    gpbag_nest_q=gpbag_nest_q,
                )
                q_fast = nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7_aud)
                member_audits.append(np.asarray(p_tb, np.float64))
                member_oofs.append(np.asarray(q_real, np.float64))
                member_oofs_hwm_fast.append(np.asarray(q_fast, np.float64))
                member_names.append(f"{r4_name}__{gp_name}__rep{r}")

    p_audit_real, meta_w_real = bag_predictions_precision_weighted_oof(y, member_oofs, member_audits)
    weights_real = np.asarray(meta_w_real["weights"], dtype=np.float64)
    p_audit_hwm, meta_w_hwm = bag_predictions_precision_weighted_oof(y, member_oofs_hwm_fast, member_audits)
    weights_hwm = np.asarray(meta_w_hwm["weights"], dtype=np.float64)

    if audit_weight_mode == "hwm_1842_exact":
        p_audit_ens = p_audit_hwm
        weights_full = weights_hwm
    else:
        p_audit_ens = p_audit_real
        weights_full = weights_real

    OOF_mat = np.column_stack(member_oofs)
    OOF_hwm_mat = np.column_stack(member_oofs_hwm_fast)
    AUD_mat = np.column_stack(member_audits)
    p_in_ens = OOF_mat @ weights_full

    # Outer-5-fold (`lab0`) cross-fitted ensemble SEARCH-OOF (`P2 v1.5`: weights refit on `SEARCH \ fk`, applied to `OOF_mat[va]`)
    p_oof_ens = np.empty(len(y), dtype=np.float64)
    W_src_mat = OOF_hwm_mat if audit_weight_mode == "hwm_1842_exact" else OOF_mat
    for fk in sorted(int(k) for k in np.unique(lab0)):
        va = lab0 == fk
        tr = ~va
        if int(np.sum(va)) == 0:
            continue
        if int(np.sum(tr)) >= 8:
            p_va, _ = bag_predictions_precision_weighted_oof(
                y[tr],
                [W_src_mat[tr, j] for j in range(W_src_mat.shape[1])],
                [OOF_mat[va, j] for j in range(OOF_mat.shape[1])],
            )
            p_oof_ens[va] = p_va
        else:
            p_oof_ens[va] = np.mean(OOF_mat[va], axis=1)

    return {
        "member_oof": OOF_mat,
        "member_oof_hwm_fast": OOF_hwm_mat,
        "member_weights": weights_full,
        "member_weights_hwm_fast": weights_hwm,
        "member_weights_real_stacker": weights_real,
        "p_oof_ens": p_oof_ens,
        "p_in_ens": p_in_ens,
        "lab": np.asarray(lab0, dtype=np.int64),
        "labs_rep": np.stack(labs_rep, axis=0).astype(np.int64),
        "member_audit": AUD_mat,
        "p_audit_ens": np.asarray(p_audit_ens, dtype=np.float64),
        "p_audit_ens_hwm_fast": np.asarray(p_audit_hwm, dtype=np.float64),
        "p_audit_ens_real_stacker": np.asarray(p_audit_real, dtype=np.float64),
        "member_names": np.asarray(member_names),
        "audit_weight_mode": np.asarray(audit_weight_mode),
    }


HWM_BASE_R4_NAMES = frozenset(("wc_r4e8", "s46201_r4e8", "s46202_r4e8"))


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
        require_nested_added,
        run_search_canary,
        out_dir,
    ) = job
    n_s, n_a = len(y), len(Xa)
    accessed: List[Tuple[str, str]] = []
    keys_read: set[str] = set()
    for p in [d1_p, d2_p, m7_p, k3_p, f4_p, tab_p] + [sp for _, sp, _ in r4_spec_paths] + list(gpbag_paths):
        if p.is_file():
            accessed.append((str(p), _sha256_file(p)))

    Q_d1, b_d1, r_d1 = load_gp_pack_no_ya(d1_p, n_s, n_a, lab0=lab0, require_lab=True, keys_sink=keys_read)
    Q_d2, b_d2, r_d2 = load_gp_pack_no_ya(d2_p, n_s, n_a, lab0=lab0, require_lab=True, keys_sink=keys_read)
    gp_packs = [("d1", Q_d1, b_d1, r_d1), ("d2", Q_d2, b_d2, r_d2)]

    blk_m = load_col_block_no_ya(m7_p, GM.PRIMARY_NAMES, n_s, n_a, lab0=lab0, keys_sink=keys_read)
    blk_k_raw = load_col_block_no_ya(k3_p, H430_K3, n_s, n_a, lab0=lab0, keys_sink=keys_read)
    blk_f_raw = load_col_block_no_ya(f4_p, H433_F4, n_s, n_a, lab0=lab0, keys_sink=keys_read)
    blk_tab = load_col_block_no_ya(tab_p, H450_TAB_COLS, n_s, n_a, lab0=lab0, keys_sink=keys_read)
    raw_common_cv7 = [
        (t, b) for t, b in (("m7", blk_m), ("k3", blk_k_raw), ("f4", blk_f_raw), ("tab450", blk_tab)) if b is not None
    ]

    r4_blocks: List[Tuple[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    r4_nest_map: Dict[str, Optional[np.ndarray]] = {}
    extra_r4_nested: Dict[str, bool] = {}
    for rname, rp, rcols in r4_spec_paths:
        loaded_r4 = load_col_block_no_ya_with_nest(rp, rcols, n_s, n_a, lab0=lab0, keys_sink=keys_read)
        if loaded_r4 is None:
            r4_blocks.append((rname, None))
            r4_nest_map[rname] = None
            if rname not in HWM_BASE_R4_NAMES:
                extra_r4_nested[rname] = False
        else:
            r4_blk_i, r4_nq_i = loaded_r4
            r4_blocks.append((rname, r4_blk_i))
            r4_nest_map[rname] = r4_nq_i
            if rname not in HWM_BASE_R4_NAMES:
                has_n = r4_nq_i is not None
                extra_r4_nested[rname] = has_n
                if require_nested_added and not has_n:
                    raise PermissionError(
                        f"P2/P4 REFUSAL on {cid}: added R4 block {rname!r} ({rp}) lacks nest_lab_/nest_q_/nest_a_ layout"
                    )

    gpbag_blk, gpbag_nest_q = build_gpbag_block_from_dirs(
        y, lab0, gpbag_paths, n_s, n_a, gpbag_k, gpbag_mode, cid=cid
    )
    gpbag_lab_verified = False
    if gpbag_blk is not None:
        gpbag_lab_verified = all(
            ("lab" in np.load(gp_p, allow_pickle=False).files) for gp_p in gpbag_paths if gp_p.is_file()
        )
        keys_read.update(("q_gp_s*", "bag_gp_s*", "ref_gp_s*"))
    gpbag_is_nested = bool(gpbag_blk is not None and gpbag_nest_q is not None)
    if require_nested_added and gpbag_blk is not None and gpbag_nest_q is None:
        raise PermissionError(
            f"P2/P4 REFUSAL on {cid}: added GPBAG block lacks outer-nested nest_lab_/nest_q_/nest_a_ layout"
        )

    canary_report: Optional[Dict[str, object]] = None
    if run_search_canary:
        r0_name = r4_blocks[0][0] if r4_blocks else "none"
        r0_blk = r4_blocks[0][1] if r4_blocks else None
        r0_nq = r4_nest_map.get(r0_name)
        rcols_d1_present = [rc is not None for rc in r_d1]

        def _canary_oof_fn(Xs_in: np.ndarray, y_in: np.ndarray, lab_in: np.ndarray) -> np.ndarray:
            if gpbag_k > 0 and gpbag_mode != "none" and gpbag_is_nested:
                gb_b, gb_n = build_gpbag_block_from_dirs(
                    y_in, lab_in, gpbag_paths, n_s, n_a, gpbag_k, gpbag_mode, cid=cid
                )
            else:
                gb_b, gb_n = None, None
            return real_stacker_outer_oof_rep(
                cid=cid,
                rep_idx=0,
                Xs=Xs_in,
                y=y_in,
                lab0=lab_in,
                Q_gp=Q_d1,
                rcols_gp_present=rcols_d1_present,
                raw_common_cv7=raw_common_cv7,
                r4_blk=r0_blk,
                gpbag_blk=gb_b,
                r4_nest_q=r0_nq,
                gpbag_nest_q=gb_n,
            )

        c_out = verify_nested_oof_fn(_canary_oof_fn, Xs, y, lab0)
        canary_report = {
            "cid": cid,
            "member": f"{r0_name}__d1__rep0",
            "gpbag_included_in_canary": bool(gpbag_k > 0 and gpbag_mode != "none" and gpbag_is_nested),
            "fold_unchanged": bool(float(c_out["max_abs_leak"]) <= 1e-12),
            "other_fold_changed": bool(float(c_out["min_other_fold_response"]) > 1e-9),
            **c_out,
        }

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
        r4_nest_map=r4_nest_map,
        gpbag_nest_q=gpbag_nest_q,
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
        "search_oof_rmse_diagnostic_coupled": rmse_oof,
        "max_abs_diff_audit_ens_vs_real_stacker": max_diff_real_vs_w,
        "d1_d2_lab0_verified": True,
        "gpbag_lab0_verified": gpbag_lab_verified,
        "gpbag_nested": gpbag_is_nested,
        "extra_r4_nested": extra_r4_nested,
        "npz_keys_read": sorted(keys_read),
        "search_canary": canary_report,
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
        choices=("hwm_1842_exact", "real_stacker"),
        default="hwm_1842_exact",
        help="hwm_1842_exact (default, matches FITSEL P1 & 1842.03 HWM rule per G1) or real_stacker",
    )
    ap.add_argument(
        "--require-nested-added",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 = refuse any added extra-r4 or gpbag block lacking nest_lab_/nest_q_/nest_a_ (P2 v1.5 gate)",
    )
    ap.add_argument(
        "--canary-sample-json",
        default="",
        help="Optional path to S10F_DECLARED_SAMPLE_19.json to run real-SEARCH F3(b) both-halves stacker canary",
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

    canary_cids: set[str] = set()
    canary_sample_sha: Optional[str] = None
    if a.canary_sample_json.strip():
        cs_p = Path(a.canary_sample_json.strip())
        _check_safe_path(cs_p)
        canary_sample_sha = _sha256_file(cs_p)
        cs_doc = json.loads(cs_p.read_text())
        canary_cids = {str(x["cid"]) for x in cs_doc.get("contexts", [])}

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
                bool(a.require_nested_added),
                bool(c in canary_cids),
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
    all_keys_read: set[str] = set()
    for d in done:
        for fp, fh in d["accessed"]:  # type: ignore[index]
            accessed_map[fp] = fh
        all_keys_read.update(d["npz_keys_read"])  # type: ignore[arg-type]

    ya_keys_cnt = sum(1 for k in all_keys_read if k in ("ya", "y_audit", "audit_y"))
    ba_keys_cnt = sum(1 for k in all_keys_read if k == "ba")
    assert ya_keys_cnt == 0 and ba_keys_cnt == 0, f"Forbidden keys read: {all_keys_read}"

    all_gpbag_nested = all(bool(d["gpbag_nested"]) for d in done) if int(a.gpbag_k) > 0 and a.gpbag_mode != "none" else None
    canary_rows = [d["search_canary"] for d in done if d["search_canary"] is not None]
    receipt = {
        "schema": "jope.h462.p2_p4_members_search_only/v3",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "ya_keys_accessed": ya_keys_cnt,
        "ba_keys_accessed": ba_keys_cnt,
        "npz_keys_read_union": sorted(all_keys_read),
        "outer_split_partition": "bank_lab0_canonical_5fold",
        "d1_d2_lab0_partition_match_count": sum(1 for d in done if d["d1_d2_lab0_verified"]),
        "gpbag_lab0_partition_match_count": sum(1 for d in done if d["gpbag_lab0_verified"]),
        "static_base_blocks_provenance": (
            "m7/k3/f4/tab450/r4e8 static column files do not store lab; outer-fold row-set identity to bank lab0 "
            "is established by generator provenance (cols_matern, cols_k3, cols_f4, cols_tab_h450, cols_r4_e8, "
            "emit_r4m_d91_cpu_xbabe3.py:79-85, emit_l104m_gpu_h462.py:116,143-146)"
        ),
        "member_oof_stacker": "stack_guard_cv7_context_outer_lab0",
        "p_oof_ens_diagnostic_label": (
            "DIAGNOSTIC ONLY (second-order SEARCH\\g coupled; never read by s10f_forecast_d91 Pass 1 or Pass 2)"
        ),
        "audit_weight_mode": str(a.audit_weight_mode),
        "require_nested_added": bool(a.require_nested_added),
        "all_gpbag_nested": all_gpbag_nested,
        "comparator_member_label": "cross-fitted; base inputs not nested",
        "added_gpbag_member_label": (
            "NESTED (nest_lab_/nest_q_/nest_a_)"
            if all_gpbag_nested
            else ("UPPER BOUND (not nested)" if all_gpbag_nested is False else "none")
        ),
        "search_canary_f3b": (
            {
                "sample_json": str(a.canary_sample_json),
                "sample_sha256": canary_sample_sha,
                "n_requested": len(canary_cids),
                "n_passed_both_halves": sum(
                    1 for cr in canary_rows if bool(cr["fold_unchanged"]) and bool(cr["other_fold_changed"])  # type: ignore[index]
                ),
                "contexts": canary_rows,
            }
            if canary_sample_sha is not None
            else None
        ),
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
                "search_oof_rmse_diagnostic_coupled": d["search_oof_rmse_diagnostic_coupled"],
                "max_abs_diff_audit_ens_vs_real_stacker": d["max_abs_diff_audit_ens_vs_real_stacker"],
                "d1_d2_lab0_verified": d["d1_d2_lab0_verified"],
                "gpbag_nested": d["gpbag_nested"],
                "extra_r4_nested": d["extra_r4_nested"],
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
                "ya_keys_accessed": ya_keys_cnt,
                "d1_d2_lab0_partition_match_count": receipt["d1_d2_lab0_partition_match_count"],
                "audit_weight_mode": str(a.audit_weight_mode),
                "all_gpbag_nested": all_gpbag_nested,
                "n_canary_passed": None if canary_sample_sha is None else receipt["search_canary_f3b"]["n_passed_both_halves"],  # type: ignore[index]
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
