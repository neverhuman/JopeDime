#!/usr/bin/env python3
"""H462 `0.000000% dev91` `K=6` Independent GP-Seed Run-Bag Emitter (`emit_gp_runbag6_h462.py`).

Implements pre-registered Rung A (`H462-M6-GPBAG6`, `2026-11-15T11:49Z` in `shared_CHAT.md`):
  - Runs `K=6` independent `0.000000% dev91` GP evolution seeds (`46211..46216`) of `evolve_single_context`
    (`ultra_gp_evolve_h452.py`: frozen TRAIN-only `DOPE c51a56d0` + `bootstrap.s1main.superset-2a848c8f8085`
    + `LEARNERS-V1`, `anchor_104m=None`, `assert "ya" not in zb.files`).
  - Reads `cids` and `family` directly from `--bank` (`bank_FITSEL.npz` in Step A; `bank_d91_search_f64.npz` in Step B)
    so Step A (`FITSEL`) opens ZERO `dev91` files or Stage-V metadata.
  - Distills per-context `.npz` column blocks (`--cols-out-dir`) containing:
      * `q_gp_s1..q_gp_s6`, `bag_gp_s1..bag_gp_s6`, `ref_gp_s1..ref_gp_s6` (strictly nested 5-fold OOF `BCAR` over
        each seed's `2xCV-7` island-diverse GP elites)
      * `q_gp_runbag6_eq`, `bag_gp_runbag6_eq`, `ref_gp_runbag6_eq` (equal 6-seed GP run-bag, `a6b`-style)
      * `q_gp_runbag6_prec`, `bag_gp_runbag6_prec`, `ref_gp_runbag6_prec` (`M6-PRECISION-AVG` 6-seed GP run-bag
        weighted strictly by nested-OOF inverse MSE on `(Xs, ys)`, `0` `ya` read!)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (
    str(HERE.parent),
    "/home/ubuntu/h452-stage/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .stack_guard_cv7_h452 import bcar_weights_pool_agnostic, make_stratified_kfold  # type: ignore[import-not-found]
except ImportError:
    from h462_apex_prism.stack_guard_cv7_h452 import bcar_weights_pool_agnostic, make_stratified_kfold  # noqa: E402

FORBIDDEN_TOKENS = ("CONFIRM", "FMGAP_B", "FMGAP_C", "HOLDOUT", "SEALED", "PROBE_YA", "/tmp/")
DEFAULT_SEEDS = (46211, 46212, 46213, 46214, 46215, 46216)

GPBAG6_ALL_COLS = (
    "gp_s1",
    "gp_s2",
    "gp_s3",
    "gp_s4",
    "gp_s5",
    "gp_s6",
    "gp_runbag6_eq",
    "gp_runbag6_prec",
)


def _load_h452_evolution_runtime():
    from bagged_neural_seed_h452 import BaggedNeuralEncoderH452
    from ultra_gp_evolve_h452 import (
        C,
        SUPERSET_BIN,
        SUPERSET_SHA,
        _parse_indices,
        _sha256_file,
        evolve_single_context,
    )

    return C, BaggedNeuralEncoderH452, SUPERSET_BIN, SUPERSET_SHA, _parse_indices, _sha256_file, evolve_single_context


def _check_safe_path(p: Path) -> None:
    s = str(p)
    up = s.upper()
    for tok in FORBIDDEN_TOKENS:
        if tok.upper() in up:
            raise PermissionError(f"Forbidden path token {tok!r} in {s}")


def _fold_clip_bounds(y_ref: np.ndarray) -> Tuple[float, float]:
    y_lo, y_hi = float(np.min(y_ref)), float(np.max(y_ref))
    pad = 1.5 * max(y_hi - y_lo, float(np.std(y_ref)), 1e-9)
    return y_lo - pad, y_hi + pad


def _nested_bcar_gp_seed(
    cid: str,
    seed: int,
    ys: np.ndarray,
    lab5: np.ndarray,
    Q_elites: np.ndarray,
    BAG_elites: np.ndarray,
    REF_elites: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes strictly fold-local outer-5-fold OOF `q_s` + full-search `bag_s`, `ref_s` (`0` `ya` read!)."""
    n_s, m = Q_elites.shape
    lo_full, hi_full = _fold_clip_bounds(ys)

    if m == 1:
        q_out = np.empty(n_s, dtype=np.float64)
        for fk in sorted(int(x) for x in np.unique(lab5)):
            va = lab5 == fk
            tr = ~va
            lo_f, hi_f = _fold_clip_bounds(ys[tr]) if int(np.sum(tr)) >= 2 else (lo_full, hi_full)
            q_out[va] = np.clip(Q_elites[va, 0], lo_f, hi_f)
        return (
            q_out,
            np.clip(BAG_elites[:, 0], lo_full, hi_full),
            np.clip(REF_elites[:, 0], lo_full, hi_full),
        )

    def _bcar3(Q_sub: np.ndarray, y_sub: np.ndarray, tag: str, lab_sub: np.ndarray) -> np.ndarray:
        return (
            0.50 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "fold_log", lab_sub, n_bags=16)
            + 0.35 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "huber", lab_sub, n_bags=16)
            + 0.15 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "br_mse", lab_sub, n_bags=16)
        )

    q_oof = np.empty(n_s, dtype=np.float64)
    for fk in sorted(int(x) for x in np.unique(lab5)):
        va = lab5 == fk
        tr = ~va
        if int(np.sum(va)) == 0:
            continue
        lo_f, hi_f = _fold_clip_bounds(ys[tr]) if int(np.sum(tr)) >= 2 else (lo_full, hi_full)
        if int(np.sum(tr)) < 14:
            q_oof[va] = np.clip(Q_elites[va, 0], lo_f, hi_f)
            continue
        lab_in = make_stratified_kfold(f"{cid}_gpbag_s{seed}_f{fk}", ys[tr], k=5, rep=0)
        w_tr = _bcar3(Q_elites[tr], ys[tr], f"{cid}_s{seed}_f{fk}", lab_in)
        sw = float(np.sum(w_tr))
        raw_va = (Q_elites[va] @ (w_tr / sw)) if sw > 1e-12 else Q_elites[va, 0]
        q_oof[va] = np.clip(raw_va, lo_f, hi_f)

    w_full = _bcar3(Q_elites, ys, f"{cid}_s{seed}_full", lab5)
    sw_f = float(np.sum(w_full))
    w_f = (w_full / sw_f) if sw_f > 1e-12 else np.eye(1, m, 0, dtype=np.float64)[0]
    bag_pred = BAG_elites @ w_f
    ref_pred = REF_elites @ w_f
    return (
        q_oof,
        np.clip(bag_pred, lo_full, hi_full),
        np.clip(ref_pred, lo_full, hi_full),
    )


def run_outer_nested_oof_seed(
    cid: str,
    seed: int,
    Xs: np.ndarray,
    ys: np.ndarray,
    lab5: np.ndarray,
    fold_elite_fn,
) -> np.ndarray:
    """`HNC-1` `H5` strictly outer-nested per-seed `q_oof` emitter (`0` validation-label access!).

    `fold_elite_fn(fk, Xs_tr, ys_tr, lab_in, Xs_va) -> (Q_elites_tr, pred_elites_va)` evolves/fits
    candidate programs strictly on `(Xs_tr, ys_tr)` inside each outer fold `fk`, then fits `BCAR` weights
    and clipping bounds strictly on `ys_tr` before predicting `Xs_va`. Guaranteed to pass `verify_nested_oof_fn`.
    """
    n_s = int(Xs.shape[0])
    q_oof = np.empty(n_s, dtype=np.float64)

    def _bcar3(Q_sub: np.ndarray, y_sub: np.ndarray, tag: str, lab_sub: np.ndarray) -> np.ndarray:
        return (
            0.50 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "fold_log", lab_sub, n_bags=16)
            + 0.35 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "huber", lab_sub, n_bags=16)
            + 0.15 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "br_mse", lab_sub, n_bags=16)
        )

    for fk in sorted(int(x) for x in np.unique(lab5)):
        va = lab5 == fk
        tr = ~va
        if int(np.sum(va)) == 0:
            continue
        Xs_tr = np.array(Xs[tr], np.float64, copy=True)
        ys_tr = np.array(ys[tr], np.float64, copy=True)
        Xs_va = np.array(Xs[va], np.float64, copy=True)
        if len(ys_tr) < 10:
            q_oof[va] = float(np.mean(ys_tr)) if len(ys_tr) > 0 else 0.0
            continue
        lo_f, hi_f = _fold_clip_bounds(ys_tr)
        lab_in = make_stratified_kfold(f"{cid}_onest_s{seed}_f{fk}", ys_tr, k=5, rep=0)
        Q_tr_el, P_va_el = fold_elite_fn(fk, Xs_tr, ys_tr, lab_in, Xs_va)
        Q_tr_el = np.asarray(Q_tr_el, dtype=np.float64)
        P_va_el = np.asarray(P_va_el, dtype=np.float64)
        if Q_tr_el.ndim == 1:
            Q_tr_el = Q_tr_el[:, None]
            P_va_el = P_va_el[:, None]
        if Q_tr_el.shape[1] == 1 or len(ys_tr) < 14:
            raw_va = P_va_el[:, 0]
        else:
            w_tr = _bcar3(Q_tr_el, ys_tr, f"{cid}_onest_s{seed}_f{fk}", lab_in)
            sw = float(np.sum(w_tr))
            raw_va = (P_va_el @ (w_tr / sw)) if sw > 1e-12 else P_va_el[:, 0]
        q_oof[va] = np.clip(raw_va, lo_f, hi_f)
    return q_oof


def run_outer_nested_oof_committee(
    cid: str,
    seeds: Sequence[int],
    Xs: np.ndarray,
    ys: np.ndarray,
    lab5: np.ndarray,
    seed_fold_elite_fn,
    tau_shrink: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`HNC-1` `H5` strictly outer-nested multi-seed GP committee (`max_abs_leak == 0.0` on all columns!).

    Avoids both:
      (1) full-SEARCH elite selection optimism, AND
      (2) two-stage outer-OOF transitive `y[va]` coupling (`Q_seeds[tr]` having trained on `va` in other folds),
    by computing inner-fold per-seed OOF predictions strictly inside `(Xs[tr], ys[tr])` for each outer fold `fk`
    to fit fold-local `M6` precision weights `w_tr` with zero `y[va]` dependence.

    `seed_fold_elite_fn(sd, fk, Xs_tr, ys_tr, lab_in, Xs_va) -> (Q_tr_el, P_va_el)`.
    Returns `(Q_seeds_oof, q_eq_oof, q_prec_oof)`.
    """
    n_s = int(Xs.shape[0])
    k_s = len(seeds)
    Q_seeds_oof = np.empty((n_s, k_s), dtype=np.float64)
    q_prec_oof = np.empty(n_s, dtype=np.float64)
    u_w = np.full(k_s, 1.0 / float(k_s), dtype=np.float64)

    def _bcar3(Q_sub: np.ndarray, y_sub: np.ndarray, tag: str, lab_sub: np.ndarray) -> np.ndarray:
        return (
            0.50 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "fold_log", lab_sub, n_bags=16)
            + 0.35 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "huber", lab_sub, n_bags=16)
            + 0.15 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "br_mse", lab_sub, n_bags=16)
        )

    def _weights_local(Q_in_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
        eps_local = 1e-4 * max(float(np.var(y_tr)), 1e-12)
        mse = np.mean((Q_in_tr - y_tr[:, None]) ** 2, axis=0)
        prec = 1.0 / (mse + eps_local)
        sp = float(np.sum(prec))
        w_raw = (prec / sp) if (sp > 0.0 and np.isfinite(sp)) else u_w
        w = (1.0 - tau_shrink) * w_raw + tau_shrink * u_w
        return w / float(np.sum(w))

    for fk in sorted(int(x) for x in np.unique(lab5)):
        va = lab5 == fk
        tr = ~va
        if int(np.sum(va)) == 0:
            continue
        Xs_tr = np.array(Xs[tr], np.float64, copy=True)
        ys_tr = np.array(ys[tr], np.float64, copy=True)
        Xs_va = np.array(Xs[va], np.float64, copy=True)
        if len(ys_tr) < 10:
            m_tr = float(np.mean(ys_tr)) if len(ys_tr) > 0 else 0.0
            Q_seeds_oof[va, :] = m_tr
            q_prec_oof[va] = m_tr
            continue

        lo_f, hi_f = _fold_clip_bounds(ys_tr)
        lab_in = make_stratified_kfold(f"{cid}_onest_comm_f{fk}", ys_tr, k=5, rep=0)
        Q_in_seeds = np.empty((len(ys_tr), k_s), dtype=np.float64)
        P_va_seeds = np.empty((int(np.sum(va)), k_s), dtype=np.float64)

        for si, sd in enumerate(seeds):
            Q_tr_el, P_va_el = seed_fold_elite_fn(sd, fk, Xs_tr, ys_tr, lab_in, Xs_va)
            Q_tr_el = np.asarray(Q_tr_el, dtype=np.float64)
            P_va_el = np.asarray(P_va_el, dtype=np.float64)
            if Q_tr_el.ndim == 1:
                Q_tr_el = Q_tr_el[:, None]
                P_va_el = P_va_el[:, None]

            if Q_tr_el.shape[1] == 1 or len(ys_tr) < 14:
                q_in_s = np.clip(Q_tr_el[:, 0], lo_f, hi_f)
                p_va_s = np.clip(P_va_el[:, 0], lo_f, hi_f)
            else:
                # Inner-5-fold OOF of BCAR strictly inside `ys_tr` for unbiased fold-local precision weighting
                q_in_s = np.empty(len(ys_tr), dtype=np.float64)
                for ifk in sorted(int(x) for x in np.unique(lab_in)):
                    iva = lab_in == ifk
                    itr = ~iva
                    if int(np.sum(iva)) == 0:
                        continue
                    if int(np.sum(itr)) < 10:
                        q_in_s[iva] = np.clip(Q_tr_el[iva, 0], lo_f, hi_f)
                    else:
                        lab_in2 = make_stratified_kfold(f"{cid}_onest_s{sd}_f{fk}_i{ifk}", ys_tr[itr], k=4, rep=0)
                        w_itr = _bcar3(Q_tr_el[itr], ys_tr[itr], f"{cid}_onest_s{sd}_f{fk}_i{ifk}", lab_in2)
                        sw_i = float(np.sum(w_itr))
                        raw_iva = (Q_tr_el[iva] @ (w_itr / sw_i)) if sw_i > 1e-12 else Q_tr_el[iva, 0]
                        q_in_s[iva] = np.clip(raw_iva, lo_f, hi_f)
                w_tr_s = _bcar3(Q_tr_el, ys_tr, f"{cid}_onest_s{sd}_f{fk}", lab_in)
                sw_s = float(np.sum(w_tr_s))
                raw_va = (P_va_el @ (w_tr_s / sw_s)) if sw_s > 1e-12 else P_va_el[:, 0]
                p_va_s = np.clip(raw_va, lo_f, hi_f)

            Q_in_seeds[:, si] = q_in_s
            P_va_seeds[:, si] = p_va_s

        Q_seeds_oof[va, :] = P_va_seeds
        w_comm = _weights_local(Q_in_seeds, ys_tr)
        q_prec_oof[va] = P_va_seeds @ w_comm

    q_eq_oof = np.mean(Q_seeds_oof, axis=1)
    return Q_seeds_oof, q_eq_oof, q_prec_oof


def build_nested_gpbag_from_fold_elites(
    cid: str,
    seeds: Sequence[int],
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab5: np.ndarray,
    seed_fold_elites_with_audit_fn,
    tau_shrink: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Builds `cl_h458` / `P2 v1.4 §3` `nest_lab_<nm>`, `nest_q_<nm>` `(k_o, n_s)`, `nest_a_<nm>` `(k_o, n_a)`.

    For every outer fold `fk in range(k_o)` (`hold = (lab5 == fk)`):
      - `nest_q_<nm>[fk, ~hold]` is cross-fitted strictly inside `(Xs[~hold], ys[~hold])` (zero rows of `hold` seen),
      - `nest_q_<nm>[fk, hold]` is the out-of-sample prediction on `Xs[hold]` from the fit on `~hold`,
      - `nest_a_<nm>[fk, :]` is the audit-slot prediction on `Xa` from the fit on `~hold`.
    `seed_fold_elites_with_audit_fn(sd, fk, Xs_tr, ys_tr, lab_in, Xs_va, Xa) -> (Q_tr_el, P_va_el, P_aud_el)`.
    """
    lab_o = np.asarray(lab5, dtype=np.int64)
    n_s = int(Xs.shape[0])
    n_a = int(Xa.shape[0])
    k_o = int(np.max(lab_o)) + 1
    k_s = len(seeds)
    u_w = np.full(k_s, 1.0 / float(k_s), dtype=np.float64)

    nest_q_seeds = np.empty((k_o, n_s, k_s), dtype=np.float64)
    nest_a_seeds = np.empty((k_o, n_a, k_s), dtype=np.float64)
    nest_q_eq = np.empty((k_o, n_s), dtype=np.float64)
    nest_a_eq = np.empty((k_o, n_a), dtype=np.float64)
    nest_q_pr = np.empty((k_o, n_s), dtype=np.float64)
    nest_a_pr = np.empty((k_o, n_a), dtype=np.float64)

    def _bcar3(Q_sub: np.ndarray, y_sub: np.ndarray, tag: str, lab_sub: np.ndarray) -> np.ndarray:
        return (
            0.50 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "fold_log", lab_sub, n_bags=16)
            + 0.35 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "huber", lab_sub, n_bags=16)
            + 0.15 * bcar_weights_pool_agnostic(Q_sub, y_sub, tag, "br_mse", lab_sub, n_bags=16)
        )

    def _weights_local(Q_in_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
        eps_local = 1e-4 * max(float(np.var(y_sub)), 1e-12)
        mse = np.mean((Q_in_sub - y_sub[:, None]) ** 2, axis=0)
        prec = 1.0 / (mse + eps_local)
        sp = float(np.sum(prec))
        w_raw = (prec / sp) if (sp > 0.0 and np.isfinite(sp)) else u_w
        w = (1.0 - tau_shrink) * w_raw + tau_shrink * u_w
        return w / float(np.sum(w))

    for fk in range(k_o):
        va = lab_o == fk
        tr = ~va
        n_va = int(np.sum(va))
        n_tr = int(np.sum(tr))
        Xs_tr = np.array(Xs[tr], np.float64, copy=True)
        ys_tr = np.array(ys[tr], np.float64, copy=True)
        Xs_va = np.array(Xs[va], np.float64, copy=True)
        if n_tr < 10:
            m_tr = float(np.mean(ys_tr)) if n_tr > 0 else 0.0
            nest_q_seeds[fk, :, :] = m_tr
            nest_a_seeds[fk, :, :] = m_tr
            nest_q_eq[fk, :] = m_tr
            nest_a_eq[fk, :] = m_tr
            nest_q_pr[fk, :] = m_tr
            nest_a_pr[fk, :] = m_tr
            continue

        lo_f, hi_f = _fold_clip_bounds(ys_tr)
        lab_in = make_stratified_kfold(f"{cid}_onest_comm_f{fk}", ys_tr, k=5, rep=0)
        Q_in_seeds = np.empty((n_tr, k_s), dtype=np.float64)
        P_va_seeds = np.empty((n_va, k_s), dtype=np.float64)
        P_aud_seeds = np.empty((n_a, k_s), dtype=np.float64)

        for si, sd in enumerate(seeds):
            Q_tr_el, P_va_el, P_aud_el = seed_fold_elites_with_audit_fn(sd, fk, Xs_tr, ys_tr, lab_in, Xs_va, Xa)
            Q_tr_el = np.asarray(Q_tr_el, dtype=np.float64)
            P_va_el = np.asarray(P_va_el, dtype=np.float64)
            P_aud_el = np.asarray(P_aud_el, dtype=np.float64)
            if Q_tr_el.ndim == 1:
                Q_tr_el = Q_tr_el[:, None]
                P_va_el = P_va_el[:, None]
                P_aud_el = P_aud_el[:, None]

            if Q_tr_el.shape[1] == 1 or n_tr < 14:
                q_in_s = np.clip(Q_tr_el[:, 0], lo_f, hi_f)
                p_va_s = np.clip(P_va_el[:, 0], lo_f, hi_f) if n_va > 0 else np.empty(0, dtype=np.float64)
                p_aud_s = np.clip(P_aud_el[:, 0], lo_f, hi_f)
            else:
                q_in_s = np.empty(n_tr, dtype=np.float64)
                for ifk in sorted(int(x) for x in np.unique(lab_in)):
                    iva = lab_in == ifk
                    itr = ~iva
                    if int(np.sum(iva)) == 0:
                        continue
                    if int(np.sum(itr)) < 10:
                        q_in_s[iva] = np.clip(Q_tr_el[iva, 0], lo_f, hi_f)
                    else:
                        lab_in2 = make_stratified_kfold(f"{cid}_onest_s{sd}_f{fk}_i{ifk}", ys_tr[itr], k=4, rep=0)
                        w_itr = _bcar3(Q_tr_el[itr], ys_tr[itr], f"{cid}_onest_s{sd}_f{fk}_i{ifk}", lab_in2)
                        sw_i = float(np.sum(w_itr))
                        raw_iva = (Q_tr_el[iva] @ (w_itr / sw_i)) if sw_i > 1e-12 else Q_tr_el[iva, 0]
                        q_in_s[iva] = np.clip(raw_iva, lo_f, hi_f)
                w_tr_s = _bcar3(Q_tr_el, ys_tr, f"{cid}_onest_s{sd}_f{fk}", lab_in)
                sw_s = float(np.sum(w_tr_s))
                raw_va = (P_va_el @ (w_tr_s / sw_s)) if (sw_s > 1e-12 and n_va > 0) else (P_va_el[:, 0] if n_va > 0 else np.empty(0, dtype=np.float64))
                raw_aud = (P_aud_el @ (w_tr_s / sw_s)) if sw_s > 1e-12 else P_aud_el[:, 0]
                p_va_s = np.clip(raw_va, lo_f, hi_f) if n_va > 0 else np.empty(0, dtype=np.float64)
                p_aud_s = np.clip(raw_aud, lo_f, hi_f)

            Q_in_seeds[:, si] = q_in_s
            if n_va > 0:
                P_va_seeds[:, si] = p_va_s
            P_aud_seeds[:, si] = p_aud_s

        nest_q_seeds[fk, tr, :] = Q_in_seeds
        if n_va > 0:
            nest_q_seeds[fk, va, :] = P_va_seeds
        nest_a_seeds[fk, :, :] = P_aud_seeds

        nest_q_eq[fk, :] = np.mean(nest_q_seeds[fk], axis=1)
        nest_a_eq[fk, :] = np.mean(P_aud_seeds, axis=1)

        # Inner cross-fitted precision committee on `tr` + fold-local precision committee on `va` & `audit`
        q_pr_tr = np.empty(n_tr, dtype=np.float64)
        for ifk in sorted(int(x) for x in np.unique(lab_in)):
            iva = lab_in == ifk
            itr = ~iva
            if int(np.sum(iva)) == 0:
                continue
            w_itr_c = _weights_local(Q_in_seeds[itr], ys_tr[itr]) if int(np.sum(itr)) >= 8 else u_w
            q_pr_tr[iva] = Q_in_seeds[iva] @ w_itr_c
        w_comm = _weights_local(Q_in_seeds, ys_tr)
        nest_q_pr[fk, tr] = q_pr_tr
        if n_va > 0:
            nest_q_pr[fk, va] = P_va_seeds @ w_comm
        nest_a_pr[fk, :] = P_aud_seeds @ w_comm

    out: Dict[str, np.ndarray] = {}
    for k_i in range(k_s):
        nm = f"gp_s{k_i + 1}"
        out[f"nest_lab_{nm}"] = lab_o.copy()
        out[f"nest_q_{nm}"] = nest_q_seeds[:, :, k_i]
        out[f"nest_a_{nm}"] = nest_a_seeds[:, :, k_i]
    eq_nm = "gp_runbag6_eq" if k_s == 6 else f"gp_runbag{k_s}_eq"
    pr_nm = "gp_runbag6_prec" if k_s == 6 else f"gp_runbag{k_s}_prec"
    out[f"nest_lab_{eq_nm}"] = lab_o.copy()
    out[f"nest_q_{eq_nm}"] = nest_q_eq
    out[f"nest_a_{eq_nm}"] = nest_a_eq
    out[f"nest_lab_{pr_nm}"] = lab_o.copy()
    out[f"nest_q_{pr_nm}"] = nest_q_pr
    out[f"nest_a_{pr_nm}"] = nest_a_pr
    return out


def _precision_combine_oof(
    ys: np.ndarray,
    lab5: np.ndarray,
    Q_seeds: np.ndarray,
    BAG_seeds: np.ndarray,
    REF_seeds: np.ndarray,
    tau_shrink: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold precision-weighted (`1 / MSE_oof`) GP-seed committee with strictly fold-local `eps`."""
    n_s, k_s = Q_seeds.shape
    u_w = np.full(k_s, 1.0 / float(k_s), dtype=np.float64)

    def _weights(Q_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
        eps_local = 1e-4 * max(float(np.var(y_sub)), 1e-12)
        mse = np.mean((Q_sub - y_sub[:, None]) ** 2, axis=0)
        prec = 1.0 / (mse + eps_local)
        sp = float(np.sum(prec))
        w_raw = (prec / sp) if (sp > 0.0 and np.isfinite(sp)) else u_w
        w = (1.0 - tau_shrink) * w_raw + tau_shrink * u_w
        return w / float(np.sum(w))

    q_prec = np.empty(n_s, dtype=np.float64)
    for fk in sorted(int(x) for x in np.unique(lab5)):
        va = lab5 == fk
        tr = ~va
        w_tr = _weights(Q_seeds[tr], ys[tr]) if int(np.sum(tr)) >= 10 else u_w
        q_prec[va] = Q_seeds[va] @ w_tr

    w_full = _weights(Q_seeds, ys)
    return q_prec, BAG_seeds @ w_full, REF_seeds @ w_full, w_full


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--cols-out-dir", required=True)
    ap.add_argument("--seeds", default="46211,46212,46213,46214,46215,46216")
    ap.add_argument("--indices", default="")
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--inner-slots", type=int, default=2)
    ap.add_argument("--rayon-threads", type=int, default=1)
    ap.add_argument("--n-pop", type=int, default=28)
    ap.add_argument("--n-gens", type=int, default=36)
    ap.add_argument("--top-export", type=int, default=8)
    ap.add_argument(
        "--outer-nested-q",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 = HNC-1 H5 strict outer-nested GP evolution per outer SEARCH fold (emits nest_lab_*, nest_q_*, nest_a_*)",
    )
    ap.add_argument(
        "--resume-stage-v",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 = call sanctioned common_h317.resume_stage_v() when emitting ya-free SEARCH columns on Stage V bank",
    )
    a = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    (
        C_h317,
        BaggedNeuralEncoderH452,
        SUPERSET_BIN,
        SUPERSET_SHA,
        _parse_indices,
        _sha256_file,
        evolve_single_context,
    ) = _load_h452_evolution_runtime()
    if bool(a.resume_stage_v):
        C_h317.resume_stage_v()

    bank_p = Path(a.bank)
    work_root = Path(a.work_root)
    cols_out = Path(a.cols_out_dir)
    for p in (bank_p, work_root, cols_out):
        _check_safe_path(p)

    assert _sha256_file(SUPERSET_BIN) == SUPERSET_SHA, f"Worker sha mismatch: {SUPERSET_BIN}"
    seeds = tuple(int(x.strip()) for x in a.seeds.split(",") if x.strip())
    assert len(seeds) == 6, f"Expected 6 seeds, got {seeds}"
    gen_snap = int(a.n_gens)
    outer_nested_q = bool(a.outer_nested_q)

    work_root.mkdir(parents=True, exist_ok=True)
    cols_out.mkdir(parents=True, exist_ok=True)
    slots_root = work_root / "slots"
    slots_root.mkdir(parents=True, exist_ok=True)

    bank_rows: Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden in search bank!"
        b_cids = [str(x) for x in zb["cids"].tolist()]
        b_fams = [str(x) for x in zb["family"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            bank_rows[c] = (
                b_fams[i],
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n

    cids_sorted = sorted(bank_rows)
    if a.indices.strip():
        target_indices = [idx for idx in _parse_indices(a.indices) if 0 <= idx < len(cids_sorted)]
    else:
        target_indices = list(range(len(cids_sorted)))

    encoder = BaggedNeuralEncoderH452(dope_threads=1)
    t0 = time.monotonic()

    # Phase 1: Evolve full-SEARCH (and, if --outer-nested-q 1, each outer fold's train-only split) across `a.workers`
    tasks: List[Tuple[int, str, int, int, int]] = []
    for idx in target_indices:
        c = cids_sorted[idx]
        if (cols_out / f"{c}.npz").is_file():
            continue
        _fam, _Xs, _Xa, ys, lab5 = bank_rows[c]
        folds_c = sorted(int(x) for x in np.unique(lab5))
        for si, sd in enumerate(seeds):
            snap_f = work_root / f"seed_{sd}" / f"g{gen_snap}" / f"{c}.npz"
            if not snap_f.is_file():
                tasks.append((idx, c, si, sd, -1))
            if outer_nested_q:
                for fk in folds_c:
                    if int(np.sum(lab5 != fk)) < 10 or int(np.sum(lab5 == fk)) == 0:
                        continue
                    snap_fk = work_root / f"seed_{sd}_f{fk}" / f"g{gen_snap}" / f"{c}.npz"
                    if not snap_fk.is_file():
                        tasks.append((idx, c, si, sd, fk))

    print(
        json.dumps(
            {
                "event": "start_gpbag6",
                "bank": str(bank_p),
                "n_contexts": len(target_indices),
                "outer_nested_q": outer_nested_q,
                "pending_seed_runs": len(tasks),
                "workers": a.workers,
                "inner_slots": a.inner_slots,
                "n_pop": a.n_pop,
                "n_gens": a.n_gens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def _run_seed_unit(unit: Tuple[int, str, int, int, int]) -> dict:
        idx, c, si, sd, fk = unit
        fam, Xs, Xa, ys, lab5 = bank_rows[c]
        if fk < 0:
            out_seed_dir = work_root / f"seed_{sd}"
            slot_d = slots_root / f"ctx_{idx:03d}_s{si}"
            return evolve_single_context(
                cid=c,
                family_id=fam,
                Xs=Xs,
                Xa=Xa,
                ys=ys,
                lab5=lab5,
                anchor_104m=None,
                out_root=out_seed_dir,
                slot_dir=slot_d,
                encoder=encoder,
                n_pop=a.n_pop,
                n_gens=a.n_gens,
                snapshots=(gen_snap,),
                top_export=a.top_export,
                seed=sd,
                anchor_only_ablation=False,
                inner_slots=a.inner_slots,
                rayon_threads=a.rayon_threads,
            )
        va = lab5 == fk
        tr = ~va
        Xs_tr = np.array(Xs[tr], np.float64, copy=True)
        ys_tr = np.array(ys[tr], np.float64, copy=True)
        Xs_va = np.array(Xs[va], np.float64, copy=True)
        Xa_joint = np.vstack([Xs_va, Xa])
        lab_in = make_stratified_kfold(f"{c}_onest_s{sd}_f{fk}", ys_tr, k=5, rep=0)
        out_seed_dir = work_root / f"seed_{sd}_f{fk}"
        slot_d = slots_root / f"ctx_{idx:03d}_s{si}_f{fk}"
        return evolve_single_context(
            cid=c,
            family_id=fam,
            Xs=Xs_tr,
            Xa=Xa_joint,
            ys=ys_tr,
            lab5=lab_in,
            anchor_104m=None,
            out_root=out_seed_dir,
            slot_dir=slot_d,
            encoder=encoder,
            n_pop=a.n_pop,
            n_gens=a.n_gens,
            snapshots=(gen_snap,),
            top_export=a.top_export,
            seed=sd * 31 + fk + 1,
            anchor_only_ablation=False,
            inner_slots=a.inner_slots,
            rayon_threads=a.rayon_threads,
        )

    done_u = 0
    if tasks:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(_run_seed_unit, u) for u in tasks]
            for f in cf.as_completed(futs):
                _ = f.result()
                done_u += 1
                if done_u % 30 == 0 or done_u == len(tasks):
                    print(
                        json.dumps(
                            {
                                "event": "progress_seed_runs",
                                "done": done_u,
                                "total": len(tasks),
                                "elapsed_s": round(time.monotonic() - t0, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    # Phase 2: Distill nested-OOF per-seed + 6-seed equal & precision-weighted GP run-bag `.npz` blocks
    def _distill_ctx(idx: int) -> str:
        c = cids_sorted[idx]
        dst = cols_out / f"{c}.npz"
        if dst.is_file():
            return c
        _fam, Xs, Xa, ys, lab5 = bank_rows[c]
        q_list, bag_list, ref_list = [], [], []
        for sd in seeds:
            snap_p = work_root / f"seed_{sd}" / f"g{gen_snap}" / f"{c}.npz"
            with np.load(snap_p, allow_pickle=False) as zs:
                Q_el = np.asarray(zs["Q_2xcv7"], np.float64)
                BAG_el = np.asarray(zs["AKF_2xcv7"], np.float64)
                REF_el = np.asarray(zs["PA_2xcv7"] if "PA_2xcv7" in zs.files else zs["PA"], np.float64)
            qs, bs, rs = _nested_bcar_gp_seed(c, sd, ys, lab5, Q_el, BAG_el, REF_el)
            q_list.append(qs)
            bag_list.append(bs)
            ref_list.append(rs)

        Q_seeds = np.column_stack(q_list)
        BAG_seeds = np.column_stack(bag_list)
        REF_seeds = np.column_stack(ref_list)
        assert BAG_seeds.shape[0] == Xa.shape[0]

        nest_payload: Dict[str, np.ndarray] = {}
        if outer_nested_q:
            def _load_seed_fold_elites_with_aud(
                sd: int,
                fk: int,
                _xtr: np.ndarray,
                _ytr: np.ndarray,
                _lin: np.ndarray,
                xva: np.ndarray,
                xa_aud: np.ndarray,
            ):
                sf_p = work_root / f"seed_{sd}_f{fk}" / f"g{gen_snap}" / f"{c}.npz"
                n_va_f = int(xva.shape[0])
                n_a_f = int(xa_aud.shape[0])
                with np.load(sf_p, allow_pickle=False) as zf:
                    q_tr = np.asarray(zf["Q_2xcv7"], np.float64)
                    akf = np.asarray(zf["AKF_2xcv7"], np.float64)
                assert akf.shape[0] == n_va_f + n_a_f, f"Nested AKF row mismatch: {akf.shape[0]} vs {n_va_f + n_a_f}"
                return q_tr, akf[:n_va_f], akf[n_va_f:]

            nest_payload = build_nested_gpbag_from_fold_elites(
                c, seeds, Xs, Xa, ys, lab5, _load_seed_fold_elites_with_aud
            )

            def _load_seed_fold_elites(sd: int, fk: int, xtr: np.ndarray, ytr: np.ndarray, lin: np.ndarray, xva: np.ndarray):
                q_tr, p_va, _p_aud = _load_seed_fold_elites_with_aud(sd, fk, xtr, ytr, lin, xva, Xa)
                return q_tr, p_va

            Q_seeds, q_eq, q_pr = run_outer_nested_oof_committee(c, seeds, Xs, ys, lab5, _load_seed_fold_elites)
            _, bag_pr, ref_pr, w_pr = _precision_combine_oof(ys, lab5, Q_seeds, BAG_seeds, REF_seeds)
            bag_eq = np.mean(BAG_seeds, axis=1)
            ref_eq = np.mean(REF_seeds, axis=1)
        else:
            q_eq = np.mean(Q_seeds, axis=1)
            bag_eq = np.mean(BAG_seeds, axis=1)
            ref_eq = np.mean(REF_seeds, axis=1)
            q_pr, bag_pr, ref_pr, w_pr = _precision_combine_oof(ys, lab5, Q_seeds, BAG_seeds, REF_seeds)

        payload = {
            "ok": np.array(True),
            "lab": np.asarray(lab5, dtype=np.int64),
            "outer_nested_q": np.array(outer_nested_q),
            "w_prec": w_pr,
            "seeds": np.array(seeds, dtype=np.int64),
            **nest_payload,
        }
        for k_i in range(len(seeds)):
            nm = f"gp_s{k_i + 1}"
            payload[f"q_{nm}"] = Q_seeds[:, k_i]
            payload[f"bag_{nm}"] = BAG_seeds[:, k_i]
            payload[f"ref_{nm}"] = REF_seeds[:, k_i]
        payload["q_gp_runbag6_eq"] = q_eq if len(seeds) == 6 else np.mean(Q_seeds[:, :6], axis=1)
        payload["bag_gp_runbag6_eq"] = bag_eq if len(seeds) == 6 else np.mean(BAG_seeds[:, :6], axis=1)
        payload["ref_gp_runbag6_eq"] = ref_eq if len(seeds) == 6 else np.mean(REF_seeds[:, :6], axis=1)
        if len(seeds) == 6:
            payload["q_gp_runbag6_prec"] = q_pr
            payload["bag_gp_runbag6_prec"] = bag_pr
            payload["ref_gp_runbag6_prec"] = ref_pr
        else:
            q6_pr, b6_pr, r6_pr, _ = _precision_combine_oof(ys, lab5, Q_seeds[:, :6], BAG_seeds[:, :6], REF_seeds[:, :6])
            if outer_nested_q:
                _, _, q6_pr = run_outer_nested_oof_committee(c, seeds[:6], Xs, ys, lab5, _load_seed_fold_elites)
            payload["q_gp_runbag6_prec"] = q6_pr
            payload["bag_gp_runbag6_prec"] = b6_pr
            payload["ref_gp_runbag6_prec"] = r6_pr
            payload[f"q_gp_runbag{len(seeds)}_eq"] = q_eq
            payload[f"bag_gp_runbag{len(seeds)}_eq"] = bag_eq
            payload[f"ref_gp_runbag{len(seeds)}_eq"] = ref_eq
            payload[f"q_gp_runbag{len(seeds)}_prec"] = q_pr
            payload[f"bag_gp_runbag{len(seeds)}_prec"] = bag_pr
            payload[f"ref_gp_runbag{len(seeds)}_prec"] = ref_pr

        tmp = cols_out / f"{c}.tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, dst)
        return c

    with cf.ThreadPoolExecutor(max_workers=min(32, max(1, a.workers))) as ex:
        list(ex.map(_distill_ctx, target_indices))

    dope_ok, dope_sha = encoder.verify_dope_unchanged()
    assert dope_ok, "DOPE weights mutated!"
    n_files = len(list(cols_out.glob("*.npz")))
    print(
        json.dumps(
            {
                "status": "OK",
                "cols_out_dir": str(cols_out),
                "n_files": n_files,
                "dope_sha256": dope_sha,
                "wall_s": round(time.monotonic() - t0, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
