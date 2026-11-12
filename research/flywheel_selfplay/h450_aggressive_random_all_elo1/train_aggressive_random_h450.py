#!/usr/bin/env python3
"""H450 pub-r4 maximum-entropy random-split fine-tune (cl_fwriter; owner direction 2026-11-12 ~20:30Z).

Owner (Jepson, cl_fwriter session): "train ALL regression data train/test ALL of it, and randomly sample every single
sample with a different random seed in the data, never the same, and inject 0-10% columns with synthetic columns ...
take our best model and begin training it ... So maximum data entropy please, and make sure you sample our dev91 set
for validation so we can track it's elo for each major epoch, the goal is to beat kumo."

Data: the combined float64 bank `everything_combined_f64_h450_bank.npz` (train U test merged and row-shuffled before
serialisation; 3,525 tables incl. the 91 dev91 tables and 42 dev91 cousins). A gate refuses any sealed cohort
(APEX-CONFIRM, DL-Holdout, ...). dev91 enters at its natural smart-stratified share (no oversampling).

Every real episode draws a fresh blake2b seed, then independently and at random:
  - the table (smart stratified weights: rows^0.3 / sqrt(family size) x structural-stratum balance^0.25);
  - the target: own target 85 %, another real column 10 %, a synthetic teacher on the real features 5 %;
    Huberised 1.5-3 % target jitter on 30 % of real-target episodes;
  - the rows (75 % target-quantile stratified subsample, else uniform), never duplicated;
  - the context share p_ctx in [0.20, 0.95] and the context/query split (75 % target-stratified, else uniform);
  - the real columns (random crop 40 % of the time; widest tables use a 70 % top-|corr| + 30 % random selection)
    and their order and slot ids;
  - 0-10 % injected synthetic columns (k <= floor(0.10 * F) of the F columns the model sees; 7 kinds);
  - the target space: raw 40 %, Ridge-GCV residual 40 %, Yeo-Johnson warp 20 %.
Fully synthetic batches (v3s prior, panel/lead-lag and sentinel tables) take `--synth-prob` of steps.

Writes, create-once and read-only: `ckpt_epoch_NN.pt` (EMA weights, every `--epoch-hours` of training time) with
`epoch_NN.json` (exposure ledger per cohort), `ckpt_h450_final.pt` and `manifest.json`. `ckpt_last.pt` (model, EMA,
optimiser, ledger) is rewritten every `--last-every-s` seconds so `--resume` continues after a GPU-window pause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random as std_random
import secrets
import signal
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.special import erfinv

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_h450 import ICLH450, n_params  # noqa: E402
from icl_prior_v3s import PROBS_V3S, Prior, draw_features  # noqa: E402
import icl_inject as INJ  # noqa: E402

WIDE_GPU = 64
MAX_TRAIN_ROWS = 2048
INJ_MAX_FRAC = 0.10
SEALED_TOKENS = ("CONFIRM", "HOLDOUT", "SEALED", "DL-VAL", "DLVAL")
EXPECTED_COHORTS = {
    "APEX-DEV-FULL", "APEX-FIT-ICL-FULL", "APEX-FIT-SEL-FULL", "D91-COMBINED-SHUFFLED", "D91-S3-COUSIN",
    "DL-TRAIN-FULL", "S3-DEV-FULL", "S3-FIT-ICL-FULL", "S3-FIT-SEL-FULL",
}
BATCH_N = (192, 320, 512, 768, 1024)
BATCH_N_P = (0.12, 0.18, 0.25, 0.25, 0.20)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CombinedF64Bank:
    """In-memory float64 combined-table bank, shared copy-on-write with forked episode workers."""

    def __init__(self, path: Path, expect_sha256: str | None = None):
        self.path = Path(path)
        self.sha256 = _sha256_file(self.path)
        if expect_sha256 and self.sha256 != expect_sha256:
            raise SystemExit(f"bank sha256 {self.sha256} != expected {expect_sha256}")
        with np.load(self.path, allow_pickle=False) as z:
            data = z["data"].astype(np.float64, copy=False)
            shapes = z["shape"].astype(np.int64)
            offs = z["offset"].astype(np.int64)
            self.cids = [str(x) for x in z["cids"].tolist()]
            self.cohorts = [str(x) for x in z["cohort"].tolist()]
            self.families = [str(x) for x in z["family"].tolist()]
            self.tabs_f64: list[np.ndarray] = []
            for i in range(len(shapes)):
                r, w = int(shapes[i, 0]), int(shapes[i, 1])
                o = int(offs[i])
                self.tabs_f64.append(np.array(data[o : o + r * w], dtype=np.float64, copy=True).reshape(r, w))
        # Sealed-cohort gate: refuse to train on any sealed or unexpected cohort.
        names = sorted(set(self.cohorts))
        bad = [c for c in names if any(t in c.upper() for t in SEALED_TOKENS)]
        if bad or set(names) != EXPECTED_COHORTS:
            raise SystemExit(f"REFUSING bank: sealed {bad}, unexpected {sorted(set(names) ^ EXPECTED_COHORTS)}")
        self.cohort_names = names
        self.cohort_idx = np.array([names.index(c) for c in self.cohorts], dtype=np.int64)
        fam_names = sorted(set(self.families))
        self.family_idx = np.array([fam_names.index(f) for f in self.families], dtype=np.int64)
        self.n_tables = len(self.tabs_f64)
        self.rows_arr = np.array([t.shape[0] for t in self.tabs_f64], dtype=np.int64)
        self.cols_arr = np.array([t.shape[1] - 1 for t in self.tabs_f64], dtype=np.int64)
        self.strata = self._compute_structural_strata()
        self.is_d91 = np.array([c.startswith("D91") for c in self.cohorts], dtype=bool)
        self.all_indices = np.arange(self.n_tables, dtype=np.int64)
        self.all_weights = self._smart_stratified_weights(self.all_indices)

    def sample_table_idx(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.all_indices, p=self.all_weights))

    def sample_eligible(self, rng: np.random.Generator, elig: np.ndarray) -> int | None:
        w = self.all_weights * elig
        s = float(w.sum())
        if s <= 0.0:
            return None
        return int(rng.choice(self.all_indices, p=w / s))

    def _compute_structural_strata(self) -> list[str]:
        strata: list[str] = []
        for i, tab in enumerate(self.tabs_f64):
            r, p = int(self.rows_arr[i]), int(self.cols_arr[i])
            r_b = "S" if r < 250 else ("M" if r < 1200 else "L")
            p_b = "N" if p <= 8 else ("W" if p <= 30 else "XW")
            yf = tab[:, -1][np.isfinite(tab[:, -1])]
            tail_b = "norm"
            if yf.size >= 16:
                q10, q50, q90 = np.percentile(yf, [10.0, 50.0, 90.0])
                if float(np.max(np.abs(yf - q50))) / max(q90 - q10, 1e-12) > 8.0:
                    tail_b = "heavy"
            strata.append(f"{r_b}:{p_b}:{tail_b}")
        return strata

    def _smart_stratified_weights(self, idxs: np.ndarray) -> np.ndarray:
        fam_counts: dict[str, int] = {}
        strat_counts: dict[str, int] = {}
        for i in idxs.tolist():
            fam_counts[self.families[i]] = fam_counts.get(self.families[i], 0) + 1
            strat_counts[self.strata[i]] = strat_counts.get(self.strata[i], 0) + 1
        mean_sc = float(len(idxs)) / max(float(len(strat_counts)), 1.0)
        w = np.array(
            [
                ((min(self.rows_arr[i], 2048) ** 0.30) / math.sqrt(float(fam_counts[self.families[i]])))
                * ((mean_sc / float(strat_counts[self.strata[i]])) ** 0.25)
                for i in idxs.tolist()
            ],
            dtype=np.float64,
        )
        return w / float(np.sum(w))


def _std_ctx_f64(Xc: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Context-only robust standardisation in float64 (no query leakage)."""
    Zc = np.zeros_like(Xc, dtype=np.float64)
    Zq = np.zeros_like(Xq, dtype=np.float64)
    for j in range(Xc.shape[1]):
        fin_c = np.isfinite(Xc[:, j])
        if int(np.sum(fin_c)) < 4:
            continue
        vals = Xc[fin_c, j]
        q25, med, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        iqr = (q75 - q25) / 1.349
        sd = float(np.std(vals))
        scale = iqr if (iqr > 1e-8 * max(sd, 1.0) and iqr > 0.0) else (sd if sd > 1e-12 else 1.0)
        Zc[:, j] = np.where(np.isfinite(Xc[:, j]), np.clip((Xc[:, j] - med) / scale, -12.0, 12.0), 0.0)
        Zq[:, j] = np.where(np.isfinite(Xq[:, j]), np.clip((Xq[:, j] - med) / scale, -12.0, 12.0), 0.0)
    return Zc, Zq


def _inject_synth_cols_np(Xc: np.ndarray, Xq: np.ndarray, k_inj: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """k_inj synthetic columns of 7 kinds (distractors, heavy-tailed distractors, collinear, bilinear/quadratic,
    warps, context-quantile steps, Fourier), fitted on the context rows only."""
    p = Xc.shape[1]
    if p == 0 or k_inj <= 0:
        return np.zeros((Xc.shape[0], 0)), np.zeros((Xq.shape[0], 0))
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z_all = np.vstack([Zc, Zq])
    n_tot, n_c = Z_all.shape[0], Xc.shape[0]
    inj_cols = np.zeros((n_tot, k_inj), dtype=np.float64)
    for m in range(k_inj):
        kind = int(rng.integers(0, 7))
        za = Z_all[:, int(rng.integers(0, p))]
        zb = Z_all[:, int(rng.integers(0, p))]
        alt = bool(rng.random() < 0.5)
        if kind == 0:
            col = rng.standard_normal(n_tot) if alt else (rng.random(n_tot) * 2.0 - 1.0) * 1.732
        elif kind == 1:
            if alt:
                chi = np.sqrt(np.maximum(np.mean(rng.standard_normal((n_tot, 3)) ** 2, axis=1), 1e-3))
                col = np.clip(rng.standard_normal(n_tot) / chi, -25.0, 25.0)
            else:
                u = np.clip(rng.random(n_tot), 1e-3, 1.0 - 1e-3)
                col = np.clip(np.tan(math.pi * (u - 0.5)), -25.0, 25.0)
        elif kind == 2:
            a = float(rng.random())
            col = a * za + (1.0 - a) * zb + 0.15 * rng.standard_normal(n_tot)
        elif kind == 3:
            col = np.clip(za * zb if alt else (za * za - 1.0), -30.0, 30.0)
        elif kind == 4:
            if alt:
                col = np.sign(za) * np.log1p(np.abs(za))
            else:
                c_sorted = np.sort(za[:n_c])
                ranks = (np.searchsorted(c_sorted, za, side="left") + 0.5) / float(n_c + 1)
                col = math.sqrt(2.0) * erfinv(np.clip(2.0 * ranks - 1.0, -0.9999, 0.9999))
        elif kind == 5:
            col = (za > float(np.percentile(za[:n_c], 25.0 if alt else 75.0))).astype(np.float64)
        else:
            w = float(rng.standard_normal() * math.exp(rng.uniform(math.log(0.5), math.log(3.0))))
            ph = float(rng.random() * 2.0 * math.pi)
            col = np.sin(w * za + ph) if alt else np.cos(w * za + ph)
        inj_cols[:, m] = col
    return inj_cols[:n_c], inj_cols[n_c:]


def _synth_teacher_target_np(Xc: np.ndarray, Xq: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """Nonlinear RFF-GP / oblique-tree / monotone-warp teacher target on the real features."""
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z = np.vstack([Zc, Zq])
    n_tot, p = Z.shape
    if p == 0 or n_tot < 24:
        return None
    sub_k = min(p, int(rng.integers(2, min(14, p) + 1))) if p >= 2 else 1
    Zs = Z[:, rng.choice(p, size=sub_k, replace=False)]
    ls = np.exp(rng.uniform(-0.7, 1.2, size=(1, sub_k)))
    W = rng.standard_normal((sub_k, 48)) / ls.T
    b = rng.uniform(0.0, 2.0 * math.pi, size=(1, 48))
    f_gp = (np.cos(Zs @ W + b) * math.sqrt(2.0 / 48)) @ rng.standard_normal(48)
    f_tree = np.zeros(n_tot, dtype=np.float64)
    for _ in range(4):
        depth = int(rng.integers(2, 5))
        leaf_idx = np.zeros(n_tot, dtype=np.int64)
        for _d in range(depth):
            proj = Zs @ rng.standard_normal(sub_k)
            thr = float(np.percentile(proj[: Xc.shape[0]], float(rng.uniform(20.0, 80.0))))
            leaf_idx = (leaf_idx << 1) | (proj > thr).astype(np.int64)
        f_tree += rng.standard_normal(1 << depth)[leaf_idx] * 0.5
    u = float(rng.random())
    f = f_gp if u < 0.35 else (f_tree if u < 0.65 else (0.6 * f_gp + 0.4 * f_tree))
    sd = float(np.std(f))
    if not math.isfinite(sd) or sd <= 1e-9:
        return None
    f = (f - float(np.mean(f))) / sd
    w_kind = int(rng.integers(0, 4))
    if w_kind == 1:
        f = np.sign(f) * (np.abs(f) ** float(rng.uniform(0.6, 1.8)))
    elif w_kind == 2:
        f = np.sinh(0.65 * f)
    elif w_kind == 3:
        f = f + 0.25 * (f ** 2)
    y = f + float(math.exp(rng.uniform(math.log(0.01), math.log(0.22)))) * float(np.std(f)) * rng.standard_normal(n_tot)
    n_c = Xc.shape[0]
    if float(np.std(y[:n_c])) <= 1e-9 or not np.all(np.isfinite(y)):
        return None
    return y[:n_c], y[n_c:]


def _select_cols(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """k of the p real columns: a random subset, except that tables wider than WIDE_GPU keep 70 % top-|corr|."""
    p = Xc.shape[1]
    if k >= p:
        return Xc, Xq
    if p <= WIDE_GPU:
        idx = rng.choice(p, size=k, replace=False)
        return Xc[:, idx], Xq[:, idx]
    yc0 = yc - float(np.mean(yc))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        fin = np.isfinite(Xc[:, j])
        if int(np.sum(fin)) >= 12:
            v = Xc[fin, j] - float(np.mean(Xc[fin, j]))
            vn = float(np.linalg.norm(v))
            if vn > 1e-12:
                scores[j] = abs(float(np.dot(v, yc0[fin])) / (vn * yn))
    top = np.argsort(-scores)[: int(round(0.70 * k))]
    rest = rng.choice(np.setdiff1d(np.arange(p), top), size=k - len(top), replace=False)
    idx = np.concatenate([top, rest])
    return Xc[:, idx], Xq[:, idx]


def _sample_full_synth_special(B: int, N: int, Fn: int, n_ctx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fully synthetic float64 panel/lead-lag or sentinel/hurdle point-mass tables (pub-r3 Regime 3, unchanged)."""
    is_panel = bool(rng.random() < 0.55)
    xs = np.zeros((B, N, Fn), dtype=np.float64)
    ys = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        if is_panel:
            n_ent = int(rng.integers(6, max(8, min(48, N // 8))))
            ent = rng.integers(0, n_ent, size=N)
            t_step = rng.uniform(0.0, 2000.0, size=N)
            base_epoch_ms = float(rng.choice([1.7040672e12, 20725.0, 15000.0]))
            ts_col = base_epoch_ms + t_step * (60_000.0 if base_epoch_ms > 1e9 else 0.01)
            covs = rng.standard_normal((N, max(1, Fn - 2)))
            ent_base = rng.standard_normal(n_ent) * 2.5
            ent_freq = rng.uniform(0.003, 0.02, size=n_ent)
            ent_ph = rng.uniform(0.0, 2.0 * math.pi, size=n_ent)
            ent_slope = rng.standard_normal(n_ent) * 0.002
            w_cov = rng.standard_normal(covs.shape[1]) * 0.35
            y_clean = ent_base[ent] + 1.8 * np.sin(ent_freq[ent] * t_step + ent_ph[ent]) + ent_slope[ent] * t_step + covs @ w_cov
            y_b = y_clean + float(rng.uniform(0.015, 0.12)) * max(float(np.std(y_clean)), 1e-3) * rng.standard_normal(N)
            X_full = np.column_stack([ts_col, ent.astype(np.float64), covs])[:, :Fn]
            perm = rng.permutation(N)
            X_full, y_b = X_full[perm], y_b[perm]
            if Fn >= 8 and rng.random() < 0.55:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Xc_cl, Xq_cl, Sc, Sq, _ = REL.build_h450_struct_cols(
                        X_full[:n_ctx, : max(2, Fn - 6)], y_b[:n_ctx], X_full[n_ctx:, : max(2, Fn - 6)],
                        seed=int(rng.integers(1, 1 << 30)), fast_mode=True, max_total_cols=6, include_base_struct=False,
                    )
                X_aug = np.vstack([np.hstack([Xc_cl, Sc]) if Sc.shape[1] else Xc_cl, np.hstack([Xq_cl, Sq]) if Sq.shape[1] else Xq_cl])
                if X_aug.shape[1] < Fn:
                    X_aug = np.hstack([X_aug, rng.standard_normal((N, Fn - X_aug.shape[1]))])
                X_full = X_aug[:, :Fn]
            xs[b] = X_full[:, rng.permutation(Fn)]
            ys[b] = y_b
        else:
            X_b = rng.standard_normal((N, Fn))
            w1 = rng.standard_normal(Fn) / math.sqrt(max(Fn, 1))
            w2 = rng.standard_normal(Fn) / math.sqrt(max(Fn, 1))
            y_bulk = 15.0 + 8.0 * np.sin(X_b @ w1) + 5.0 * ((X_b @ w2) ** 2 - 1.0) + 0.6 * rng.standard_normal(N)
            atom_val = float(rng.choice([-99.99, -999.0, 0.0, 9999.0]))
            gate_score = 0.75 * X_b[:, 0] - 0.55 * X_b[:, min(1, Fn - 1)] + 0.25 * rng.standard_normal(N)
            is_atom = gate_score < float(np.percentile(gate_score, float(rng.uniform(8.0, 22.0))))
            if rng.random() < 0.65:
                X_b[is_atom, 0] = -99.99
            xs[b] = X_b[:, rng.permutation(Fn)]
            ys[b] = np.where(is_atom, atom_val, y_bulk)
    return xs, ys


def _sample_full_synth_special_gpu(dev: torch.device, B: int, N: int, Fn: int, n_ctx: int, rng: np.random.Generator):
    """pub-r3 name kept for selftest_h450.py test 6: the same tables as float64 tensors on dev."""
    xs, ys = _sample_full_synth_special(B, N, Fn, n_ctx, rng)
    return torch.from_numpy(xs).to(device=dev, dtype=torch.float64), torch.from_numpy(ys).to(device=dev, dtype=torch.float64)


def real_episode(bank: CombinedF64Bank, t_idx: int, rng: np.random.Generator, n_rows_fixed: int | None = None,
                 f_fixed: int | None = None, p_ctx_fixed: float | None = None):
    """One random episode from table t_idx. Returns (X [n,F], y [n], n_ctx, info) or None.
    info = (table, target mode 0 own / 1 other column / 2 teacher, k injected, F, n rows, target space)."""
    tab = bank.tabs_f64[t_idx]
    c_tot = tab.shape[1]
    if tab.shape[0] < 28 or c_tot < 2:
        return None
    u = float(rng.random())
    tmode = 0 if u < 0.85 else (1 if u < 0.95 else 2)
    tcol = c_tot - 1
    if tmode == 1:
        cand = [j for j in range(c_tot - 1) if int(np.sum(np.isfinite(tab[:, j]))) >= 32]
        rng.shuffle(cand)
        tcol = next((j for j in cand[:8] if np.unique(tab[np.isfinite(tab[:, j]), j]).size >= 12), c_tot - 1)
        if tcol == c_tot - 1:
            tmode = 0
    if tmode in (0, 1):
        tab = tab[np.isfinite(tab[:, tcol])]
        if tmode == 0 and rng.random() < 0.45 and tab.shape[0] >= 32:
            atom = RR.HurdleSentinelDetector.detect_atom(tab[:, tcol])
            if atom is not None:
                keep = np.abs(tab[:, tcol] - atom) > 1e-7 * max(1.0, abs(atom))
                if int(np.sum(keep)) >= 32 and (n_rows_fixed is None or int(np.sum(keep)) >= n_rows_fixed):
                    tab = tab[keep]
    r_tot = tab.shape[0]
    if r_tot < 28:
        return None
    if n_rows_fixed is None:
        if rng.random() < 0.75:
            n_rows = min(r_tot, MAX_TRAIN_ROWS)
        else:
            hi = max(48, min(r_tot, MAX_TRAIN_ROWS))
            n_rows = int(min(r_tot, max(36, round(math.exp(rng.uniform(math.log(36.0), math.log(float(hi))))))))
    else:
        if r_tot < n_rows_fixed:
            return None                                     # never duplicate rows across context and query
        n_rows = n_rows_fixed
    ycol = tab[:, tcol]
    if r_tot > n_rows and rng.random() < 0.75:             # target-quantile stratified subsample
        y_ord = np.argsort(ycol + 1e-12 * max(float(np.nanstd(ycol)), 1.0) * rng.standard_normal(r_tot), kind="mergesort")
        picked, need = [], n_rows
        bins = np.array_split(y_ord, int(rng.integers(6, 17)))
        for bi, b_arr in enumerate(bins):
            if len(b_arr) == 0 or need <= 0:
                continue
            take = need if bi == len(bins) - 1 else min(len(b_arr), max(1, int(round(n_rows * len(b_arr) / float(r_tot)))))
            take = min(take, len(b_arr), need)
            picked.append(rng.permutation(b_arr)[:take])
            need -= take
        sel = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
        if len(sel) < n_rows:
            rest = np.setdiff1d(np.arange(r_tot, dtype=np.int64), sel)
            sel = np.concatenate([sel, rng.permutation(rest)[: n_rows - len(sel)]])
        sub = tab[rng.permutation(sel[:n_rows])]
    else:
        sub = tab[rng.permutation(r_tot)[:n_rows]]
    if p_ctx_fixed is None:
        u = float(rng.random())
        p_ctx = float(rng.uniform(0.20, 0.56) if u < 0.25 else (rng.uniform(0.56, 0.84) if u < 0.75 else rng.uniform(0.84, 0.95)))
    else:
        p_ctx = p_ctx_fixed
    n_ctx = max(16, min(n_rows - 4, int(round(p_ctx * n_rows))))
    feats = [j for j in range(c_tot) if j != tcol]
    if tmode == 2 and len(feats) > 1 and rng.random() < 0.5:
        feats = [j for j in feats if j != c_tot - 1] or feats    # the real target may or may not be a feature
    X_all = sub[:, feats]
    y_all = sub[:, tcol].copy()
    if rng.random() < 0.75 and n_rows >= 24:               # target-stratified context/query split
        ord_s = np.argsort(y_all + 1e-12 * max(float(np.nanstd(y_all)), 1.0) * rng.standard_normal(n_rows), kind="mergesort")
        cp, qp = [], []
        for b_arr in np.array_split(ord_s, int(rng.integers(5, min(17, max(6, n_rows // 4))))):
            bp = rng.permutation(b_arr)
            cp.append(bp[: int(round(p_ctx * len(bp)))])
            qp.append(bp[int(round(p_ctx * len(bp))):])
        c_idx, q_idx = rng.permutation(np.concatenate(cp)), rng.permutation(np.concatenate(qp))
        if len(c_idx) >= n_ctx:                              # move the surplus to the query side (exact n_ctx)
            order = np.concatenate([c_idx[:n_ctx], c_idx[n_ctx:], q_idx])
        else:
            order = np.concatenate([c_idx, q_idx[: n_ctx - len(c_idx)], q_idx[n_ctx - len(c_idx):]])
        X_all, y_all = X_all[order], y_all[order]
    Xc, Xq = X_all[:n_ctx], X_all[n_ctx:]
    yc, yq = y_all[:n_ctx].copy(), y_all[n_ctx:].copy()
    if tmode < 2 and float(np.std(yc)) <= 1e-12:
        return None

    # columns: F seen by the model, k = injected synthetic columns with k <= floor(0.10 * F)
    p = Xc.shape[1]
    if f_fixed is None:
        p_keep = p
        if p >= 6 and rng.random() < 0.40:
            p_keep = int(rng.integers(max(3, int(round(0.65 * p))), p + 1))
        k = int(rng.integers(0, min(p_keep, WIDE_GPU) // 9 + 1))
        F = min(WIDE_GPU, p_keep + k)
        k = min(k, int(math.floor(INJ_MAX_FRAC * F)))
        k_real = F - k
    else:
        F = f_fixed
        k_lo = max(0, F - p)
        k_hi = int(math.floor(INJ_MAX_FRAC * F))
        if k_lo > k_hi:
            return None
        k = int(rng.integers(k_lo, k_hi + 1))
        k_real = F - k
    Xc, Xq = _select_cols(Xc, yc if tmode < 2 else np.zeros(n_ctx), Xq, k_real, rng)
    if tmode == 2:
        t_pair = _synth_teacher_target_np(Xc, Xq, rng)
        if t_pair is None:
            return None
        yc, yq = t_pair
    elif rng.random() < 0.30:
        sy = float(np.std(yc))
        e = rng.standard_normal(n_rows) / np.sqrt(np.maximum(np.mean(rng.standard_normal((n_rows, 3)) ** 2, axis=1), 1e-3)) / 1.732
        jit = sy * float(rng.uniform(0.015, 0.03)) * np.clip(e, -1.345, 1.345)
        yc, yq = yc + jit[:n_ctx], yq + jit[n_ctx:]
    if k > 0:
        Ic, Iq = _inject_synth_cols_np(Xc, Xq, k, rng)
        Xc, Xq = np.hstack([Xc, Ic]), np.hstack([Xq, Iq])
    perm_f = rng.permutation(Xc.shape[1])
    Xc, Xq = Xc[:, perm_f], Xq[:, perm_f]

    u = float(rng.random())
    space = 0
    y_ep = np.concatenate([yc, yq])
    if u < 0.40:
        oof, pq, used, _ = RR.robust_ridge_np(Xc, yc, Xq, seed=int(rng.integers(1, 1 << 30)))
        if used:
            y_ep = np.concatenate([yc - oof, yq - pq])
            space = 1
    elif u < 0.60:
        warp = RR.YeoJohnsonWarpF64()
        zc = warp.fit_transform(yc)
        zq = warp.transform_any(yq)
        if np.all(np.isfinite(zc)) and np.all(np.isfinite(zq)) and float(np.std(zc)) > 1e-8:
            y_ep = np.concatenate([zc, zq])
            space = 2
    X_ep = np.vstack([Xc, Xq])
    if X_ep.shape[1] != F or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx])) <= 1e-12:
        return None
    return X_ep, y_ep, n_ctx, (int(t_idx), tmode, k, F, n_rows, space)


def _episode_worker_loop(worker_id, run_nonce, bank, tokens_cap, row_tokens_cap, bmax, out_q, stop_ev) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    seq = 0
    while not stop_ev.is_set():
        seq += 1
        b_seed = hashlib.blake2b(run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint32))
        if float(rng.random()) < 0.62:
            got = real_episode(bank, bank.sample_table_idx(rng), rng)
            if got is None:
                continue
            X_ep, y_ep, n_ctx, info = got
            item = (X_ep[None], y_ep[None], int(n_ctx), [info])
        else:
            t0 = bank.sample_table_idx(rng)
            ok_n = [n for n in BATCH_N if n + 8 <= int(bank.rows_arr[t0])]
            if not ok_n:
                got = real_episode(bank, t0, rng)
                if got is None:
                    continue
                item = (got[0][None], got[1][None], int(got[2]), [got[3]])
            else:
                pn = np.array([BATCH_N_P[BATCH_N.index(n)] for n in ok_n])
                N_t = int(rng.choice(ok_n, p=pn / pn.sum()))
                p0 = min(int(bank.cols_arr[t0]), WIDE_GPU)
                F_t = max(1, min(WIDE_GPU, p0 + int(rng.integers(0, p0 // 9 + 1))))
                u = float(rng.random())
                p_ctx = float(rng.uniform(0.25, 0.56) if u < 0.25 else (rng.uniform(0.56, 0.84) if u < 0.75 else rng.uniform(0.84, 0.94)))
                B_t = max(1, min(bmax, tokens_cap // max(N_t * (F_t + 4), 1), row_tokens_cap // max(N_t, 1)))
                elig = ((bank.rows_arr >= N_t + 8) & (bank.cols_arr >= F_t - int(math.floor(INJ_MAX_FRAC * F_t)))).astype(np.float64)
                xs, ys, infos, n_ctx_b = [], [], [], None
                for b in range(B_t):
                    t_idx = t0 if b == 0 else bank.sample_eligible(rng, elig)
                    if t_idx is None:
                        break
                    got = real_episode(bank, t_idx, rng, n_rows_fixed=N_t, f_fixed=F_t, p_ctx_fixed=p_ctx)
                    if got is None or got[0].shape != (N_t, F_t) or (n_ctx_b is not None and got[2] != n_ctx_b):
                        continue
                    n_ctx_b = got[2]
                    xs.append(got[0]); ys.append(got[1]); infos.append(got[3])
                if not xs:
                    continue
                item = (np.stack(xs), np.stack(ys), int(n_ctx_b), infos)
        while not stop_ev.is_set():
            try:
                out_q.put(item, timeout=1.0)
                break
            except queue.Full:
                continue


class Ledger:
    """Exposure ledger per cohort: episodes, distinct tables and families, rows, cell tokens, injected columns."""

    def __init__(self, bank: CombinedF64Bank):
        self.bank = bank
        self.table_eps = np.zeros(bank.n_tables, dtype=np.int64)
        self.table_cells = np.zeros(bank.n_tables, dtype=np.int64)
        self.table_rows = np.zeros(bank.n_tables, dtype=np.int64)
        self.tmode = np.zeros(3, dtype=np.int64)
        self.space = np.zeros(3, dtype=np.int64)
        self.inj_cols = 0
        self.real_cols = 0
        self.synth_datasets = 0
        self.synth_cells = 0

    def add_real(self, infos) -> int:
        cells = 0
        for t, tm, k, F, n, sp in infos:
            c = int(n * (F + 1))
            self.table_eps[t] += 1
            self.table_cells[t] += c
            self.table_rows[t] += int(n)
            self.tmode[tm] += 1
            self.space[sp] += 1
            self.inj_cols += int(k)
            self.real_cols += int(F - k)
            cells += c
        return cells

    def add_synth(self, B: int, N: int, F: int) -> int:
        self.synth_datasets += int(B)
        self.synth_cells += int(B * N * (F + 1))
        return int(B * N * (F + 1))

    def state(self) -> dict:
        return {"table_eps": self.table_eps.tolist(), "table_cells": self.table_cells.tolist(),
                "table_rows": self.table_rows.tolist(), "tmode": self.tmode.tolist(), "space": self.space.tolist(),
                "inj_cols": self.inj_cols, "real_cols": self.real_cols, "synth_datasets": self.synth_datasets,
                "synth_cells": self.synth_cells}

    def load(self, s: dict) -> None:
        self.table_eps = np.array(s["table_eps"], dtype=np.int64)
        self.table_cells = np.array(s["table_cells"], dtype=np.int64)
        self.table_rows = np.array(s["table_rows"], dtype=np.int64)
        self.tmode = np.array(s["tmode"], dtype=np.int64)
        self.space = np.array(s["space"], dtype=np.int64)
        for k in ("inj_cols", "real_cols", "synth_datasets", "synth_cells"):
            setattr(self, k, int(s[k]))

    def summary(self) -> dict:
        b = self.bank
        per = {}
        for ci, name in enumerate(b.cohort_names):
            m = b.cohort_idx == ci
            seen = m & (self.table_eps > 0)
            per[name] = {
                "tables": int(m.sum()), "tables_seen": int(seen.sum()),
                "families": int(np.unique(b.family_idx[m]).size), "families_seen": int(np.unique(b.family_idx[seen]).size),
                "episodes": int(self.table_eps[m].sum()), "rows_seen": int(self.table_rows[m].sum()),
                "cell_tokens": int(self.table_cells[m].sum()), "test_rows_included": True,
            }
        real_eps = int(self.table_eps.sum())
        d91 = int(self.table_eps[b.cohort_idx == b.cohort_names.index("D91-COMBINED-SHUFFLED")].sum())
        return {"per_cohort": per, "real_episodes": real_eps, "synthetic_datasets": self.synth_datasets,
                "real_cell_tokens": int(self.table_cells.sum()), "synthetic_cell_tokens": self.synth_cells,
                "dev91_share_of_real_episodes": (d91 / real_eps) if real_eps else None,
                "target_modes": dict(zip(("own", "other_column", "teacher"), self.tmode.tolist())),
                "target_spaces": dict(zip(("raw", "ridge_residual", "yeo_johnson"), self.space.tolist())),
                "injected_column_share": self.inj_cols / max(1, self.inj_cols + self.real_cols),
                "sealed_cohorts_in_bank": 0}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--bank-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, required=True, help="total training hours (cosine schedule length)")
    ap.add_argument("--epoch-hours", type=float, default=1.0)
    ap.add_argument("--last-every-s", type=float, default=600.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--synth-prob", type=float, default=0.30)
    ap.add_argument("--tokens", type=int, default=140_000)
    ap.add_argument("--row-tokens", type=int, default=4_096)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=450_401)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_p, last_p = out_dir / "ckpt_h450_final.pt", out_dir / "ckpt_last.pt"
    if final_p.exists():
        raise SystemExit(f"{final_p} exists (create-once)")
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    if _sha256_file(Path(a.init)) != a.init_sha256:
        raise SystemExit("init checkpoint sha256 mismatch")

    t_load0 = time.time()
    bank = CombinedF64Bank(Path(a.bank), expect_sha256=a.bank_sha256)
    bank_load_s = round(time.time() - t_load0, 2)
    ledger = Ledger(bank)
    run_nonce = hashlib.blake2b(secrets.token_bytes(32) + int(a.seed).to_bytes(8, "little"), digest_size=32).digest()

    ctx = mp.get_context("fork")
    ep_q = ctx.Queue(maxsize=max(16, 4 * a.workers))
    stop_ev = ctx.Event()
    procs = [ctx.Process(target=_episode_worker_loop, args=(w, run_nonce, bank, a.tokens, a.row_tokens, a.bmax, ep_q, stop_ev),
                         daemon=True) for w in range(a.workers)]
    for p in procs:
        p.start()

    stop_flag = {"stop": False}

    def _on_term(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = ICLH450(a.size).to(dev)
    ema = ICLH450(a.size).to(dev).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)
    state = {"step": 0, "micro_batches": 0, "real_batches": 0, "synth_batches": 0, "datasets": 0, "cell_tokens": 0,
             "oom_skips": 0, "nonfinite_skips": 0, "train_s": 0.0, "epochs_saved": 0, "resumes": 0}
    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        state["resumes"] += 1
        ledger.load(ck["ledger"])
        if ck["bank_sha256"] != bank.sha256 or ck["init_sha256"] != a.init_sha256:
            raise SystemExit("resume: bank or init differs from ckpt_last.pt")
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        w_init = ck.get("ema") or ck["model"]
        model.load_state_dict(w_init, strict=True)
        ema.load_state_dict(w_init, strict=True)
    params_cnt = int(n_params(model))
    prior = Prior(dev, a.seed + 17 + 1000 * state["resumes"])
    py_rng = np.random.default_rng(np.frombuffer(hashlib.blake2b(run_nonce + b"main", digest_size=32).digest(), dtype=np.uint32))

    def save_last() -> None:
        _atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "state": state,
                      "ledger": ledger.state(), "size": a.size, "bank_sha256": bank.sha256, "init_sha256": a.init_sha256,
                      "argv": sys.argv}, last_p)

    def save_epoch(k: int) -> None:
        ep_p = out_dir / f"ckpt_epoch_{k:02d}.pt"
        if ep_p.exists():
            return
        _atomic_save({"ema": ema.state_dict(), "model": ema.state_dict(), "state": dict(state), "size": a.size}, ep_p)
        os.chmod(ep_p, 0o444)
        doc = {"schema": "jope.h450.epoch/v1", "epoch": k, "train_hours": state["train_s"] / 3600.0, "step": state["step"],
               "ckpt": str(ep_p), "ckpt_sha256": _sha256_file(ep_p), "init_sha256": a.init_sha256,
               "bank_sha256": bank.sha256, "quarantined_in_corpus": True, "canonical_promotion_forbidden": True,
               "exposure": ledger.summary()}
        jp = out_dir / f"epoch_{k:02d}.json"
        jp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.chmod(jp, 0o444)
        state["epochs_saved"] = k

    budget_s = max(15.0, a.hours * 3600.0)
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)
    log_f.write(json.dumps({"start": time.strftime("%FT%TZ", time.gmtime()), "resume": bool(a.resume), "argv": sys.argv,
                            "params": params_cnt, "bank_tables": bank.n_tables, "bank_load_s": bank_load_s,
                            "train_s": state["train_s"], "step": state["step"]}) + "\n")
    grad_accum = max(1, int(a.grad_accum))
    opt.zero_grad(set_to_none=True)
    t_seg = time.time()
    s_seg0 = state["train_s"]
    t_log = t_last = time.time()
    run_loss, run_n = 0.0, 0
    try:
        while not stop_flag["stop"]:
            state["train_s"] = s_seg0 + (time.time() - t_seg)
            if state["train_s"] >= budget_s or (a.max_steps > 0 and state["step"] >= a.max_steps):
                break
            k_ep = int(state["train_s"] // (a.epoch_hours * 3600.0))
            if k_ep > state["epochs_saved"]:
                save_epoch(k_ep)
            frac = min(1.0, state["train_s"] / budget_s)
            lr_mult = (frac / 0.04) if frac < 0.04 else (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * (frac - 0.04) / 0.96)))
            cur_lr = a.lr * max(0.05, lr_mult)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            if float(py_rng.random()) < a.synth_prob:
                r_py = std_random.Random(int(py_rng.integers(1, 1 << 62)))
                Fn = max(6, min(WIDE_GPU, draw_features(r_py)))
                N = int(py_rng.choice([256, 512, 768, 1024], p=[0.20, 0.35, 0.25, 0.20]))
                B = max(1, min(a.bmax, a.tokens // max(N * (Fn + 4), 1), a.row_tokens // max(N, 1)))
                n_ctx = max(16, min(N - 4, int(round(float(py_rng.uniform(0.25, 0.93)) * N))))
                if float(py_rng.random()) < 0.60:
                    xt, yt, _, _ = INJ.sample_synth(prior, r_py, B, N, Fn, 0.55, PROBS_V3S)
                    xt, yt = xt.double(), yt.double()
                else:
                    xs, ys = _sample_full_synth_special(B, N, Fn, n_ctx, py_rng)
                    xt = torch.from_numpy(xs).to(dev)
                    yt = torch.from_numpy(ys).to(dev)
                if float(py_rng.random()) < 0.40:
                    yt_f, _, _ = RR.robust_residual_targets_torch(xt.float(), yt.float(), n_ctx, seed=int(py_rng.integers(1, 1 << 30)))
                    yt = yt_f.double()
                infos = None
            else:
                xb, yb, n_ctx, infos = ep_q.get(timeout=120.0)
                xt = torch.from_numpy(xb).to(device=dev, dtype=torch.float64, non_blocking=True)
                yt = torch.from_numpy(yb).to(device=dev, dtype=torch.float64, non_blocking=True)
            Fn_cur = xt.shape[2]
            slots = torch.randperm(FID_SLOTS, device=dev)[:Fn_cur]
            try:
                logits, ym, ys_ = model(xt, yt[:, :n_ctx], n_ctx, slots=slots, amp=(dev.type == "cuda"))
                loss = model.loss(logits, yt[:, n_ctx:], ym, ys_).mean()
                if not torch.isfinite(loss):
                    state["nonfinite_skips"] += 1
                    continue
                (loss / float(grad_accum)).backward()
            except torch.cuda.OutOfMemoryError:
                state["oom_skips"] += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            if infos is None:
                state["cell_tokens"] += ledger.add_synth(int(xt.shape[0]), int(xt.shape[1]), Fn_cur)
                state["synth_batches"] += 1
            else:
                state["cell_tokens"] += ledger.add_real(infos)
                state["real_batches"] += 1
            state["micro_batches"] += 1
            state["datasets"] += int(xt.shape[0])
            run_loss += float(loss.detach())
            run_n += 1
            if state["micro_batches"] % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                with torch.no_grad():
                    decay = min(0.995, (1.0 + state["step"]) / (10.0 + state["step"]))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - decay)
            now = time.time()
            if now - t_log >= 30.0 and run_n > 0:
                s = ledger.summary()
                log_f.write(json.dumps({
                    "step": state["step"], "train_h": round(state["train_s"] / 3600.0, 4), "loss": round(run_loss / run_n, 5),
                    "lr": round(cur_lr, 9), "datasets": state["datasets"], "ds_per_s": round(run_n / (now - t_log), 3),
                    "real_batches": state["real_batches"], "synth_batches": state["synth_batches"],
                    "real_eps": s["real_episodes"], "d91_share": s["dev91_share_of_real_episodes"],
                    "inj_share": round(s["injected_column_share"], 4), "oom": state["oom_skips"],
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0.0}) + "\n")
                run_loss, run_n, t_log = 0.0, 0, now
            if now - t_last >= a.last_every_s:
                save_last()
                t_last = now
    finally:
        stop_ev.set()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=2.0)
    state["train_s"] = s_seg0 + (time.time() - t_seg)
    save_last()
    if stop_flag["stop"]:
        log_f.write(json.dumps({"paused": time.strftime("%FT%TZ", time.gmtime()), "step": state["step"],
                                "train_h": round(state["train_s"] / 3600.0, 4)}) + "\n")
        log_f.close()
        return
    k_ep = int(state["train_s"] // (a.epoch_hours * 3600.0))
    if k_ep > state["epochs_saved"]:
        save_epoch(k_ep)
    _atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "state": state, "size": a.size}, final_p)
    os.chmod(final_p, 0o444)
    manifest = {
        "schema": "jope.h450.ckpt_manifest/v3", "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True, "size": a.size, "params": params_cnt, "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "seed": a.seed,
        "run_nonce_sha256": hashlib.sha256(run_nonce).hexdigest(), "init_ckpt": str(a.init), "init_sha256": a.init_sha256,
        "bank": str(a.bank), "bank_sha256": bank.sha256, "bank_tables": bank.n_tables, "synth_prob": a.synth_prob,
        "inj_max_frac": INJ_MAX_FRAC, "ckpt_final_path": str(final_p), "ckpt_final_sha256": _sha256_file(final_p),
        "state": state, "exposure": ledger.summary(),
    }
    man_p = out_dir / "manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(man_p, 0o444)
    log_f.close()
    print(json.dumps({k: manifest[k] for k in ("ckpt_final_sha256", "state")}, sort_keys=True))


if __name__ == "__main__":
    main()
