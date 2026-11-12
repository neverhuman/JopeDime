#!/usr/bin/env python3
"""H450 100% Online 3-Regime Random-Split Multi-Seed GPU Trainer (`train_aggressive_random_h450.py`).

Implements the owner-specified (`2026-11-12T18:48:20Z`) 3-Regime Random-Split & Augmentation Curriculum:
  1. Combined-First `float64` Bank: Loads `everything_combined_f64_h450_bank.npz` (`3,525` non-sealed tables where
     `train U test` was merged in exact `float64` and CSPRNG row-shuffled BEFORE serialization; `0` sealed opens).
  2. Step-Unique Cryptographic Random Splits (`0` Static Folds, `0` Exact `dev91` Split Matches):
     - Forked CPU ring-buffer (`mp.get_context("fork")` BEFORE CUDA init -> shared copy-on-write RAM!) draws a fresh
       seed `blake2b(run_nonce, worker_id, batch_seq, item_idx)` on EVERY episode, permuting rows anew and sampling
       random split ratios `p_ctx in [0.20, 0.95]` at untruncated row counts up to `N = 2,048` (`WIDE_GPU = 64`).
  3. Randomized 3-Regime Dataset Curriculum Across Every Episode / Micro-Batch:
     - Regime 1 — Pure Original Random-Split Data (`~32%` of real episodes):
       Pure original features (random feature crop & slot permutation, `0` added columns), `85%` real target + `15%`
       other continuous real column as self-supervised target.
     - Regime 2A — Reviewed & Validated Signal Column Addition (`~36%` of real episodes):
       Screens candidate Lead-Lag entity-time interpolations, exact `float64` date/timestamp differences, signed-log
       warps, sentinel flags, key encodings, formulas, and 3-fold OOF `HGB + Laplacian-Matérn` base predictions on
       `(X_ctx, y_ctx)` ONLY, appending only validated signal columns (`|corr_ctx| >= 0.025` or sentinel indicator).
     - Regime 2B — Synthetic Column & Teacher Injection (`~32%` of real episodes):
       Appends `2..18` synthetic columns (`Gaussian/Uniform` & `Student-t3/Cauchy` distractors, noisy collinear
       duplicates, bilinear/quadratic interactions, monotone/rank-Gauss warps, context-quantile steps, random Fourier
       features) shuffled across real feature slots, with `28%` synthetic nonlinear `GP + ObliqueTree + Warp` teacher
       targets `y_synth(X_real)` on real feature geometry and `30%` Huberized target noise.
     - Regime 3 — Full Synthetic Datasets (`--synth-prob 0.22` of GPU micro-batches):
       `60%` `Prior` + `PROBS_V3S` SCMs (`INJ.sample_synth` with `20–60%` injected distractors), `22%` full synthetic
       `float64` Panel/Temporal Entity Lead-Lag time-series (`1.70e12 + dt` ms epoch timestamps), and `18%` full
       synthetic Sentinel/Hurdle point-mass tables (`y = -99.99` / `-999` tail atom + smooth nonlinear bulk).
  4. Saves EMA (`0.995`) + Tail-4 SWA Checkpoint Soup (`ckpt_swa_soup.pt`) + Exact Integer Regime/Token/FLOP Manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import secrets
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
from sklearn.ensemble import HistGradientBoostingRegressor

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_h450 import ICLH450, n_params  # noqa: E402
from icl_prior import Prior  # noqa: E402
from icl_prior_v3s import PROBS_V3S, draw_features  # noqa: E402
import icl_inject as INJ  # noqa: E402

WIDE_GPU = 64
MAX_TRAIN_ROWS = 2048


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CombinedF64Bank:
    """In-memory `float64` combined-table bank shared copy-on-write across forked episode workers."""

    def __init__(self, path: Path, d91_mix_prob: float = 0.32):
        self.path = Path(path)
        self.sha256 = _sha256_file(self.path)
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

        self.n_tables = len(self.tabs_f64)
        self.rows_arr = np.array([t.shape[0] for t in self.tabs_f64], dtype=np.int64)
        self.is_d91 = np.array([c.startswith("D91") for c in self.cohorts], dtype=bool)
        self.d91_indices = np.flatnonzero(self.is_d91)
        self.non_d91_indices = np.flatnonzero(~self.is_d91)
        self.d91_mix_prob = float(d91_mix_prob)

        self.non_d91_weights = self._family_balanced_weights(self.non_d91_indices)
        self.d91_weights = self._family_balanced_weights(self.d91_indices)

    def _family_balanced_weights(self, idxs: np.ndarray) -> np.ndarray:
        if len(idxs) == 0:
            return np.zeros(0, dtype=np.float64)
        fam_counts: dict[str, int] = {}
        for i in idxs.tolist():
            fk = self.families[i]
            fam_counts[fk] = fam_counts.get(fk, 0) + 1
        w = np.array(
            [(min(self.rows_arr[i], 2048) ** 0.30) / math.sqrt(float(fam_counts[self.families[i]])) for i in idxs.tolist()],
            dtype=np.float64,
        )
        return w / float(np.sum(w))


def _std_ctx_f64(Xc: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Context-only robust standardization in `float64` (zero query leakage)."""
    Zc = np.zeros_like(Xc, dtype=np.float64)
    Zq = np.zeros_like(Xq, dtype=np.float64)
    for j in range(Xc.shape[1]):
        col_c = Xc[:, j]
        fin_c = np.isfinite(col_c)
        if int(np.sum(fin_c)) < 4:
            continue
        vals = col_c[fin_c]
        q25, med, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        iqr = (q75 - q25) / 1.349
        sd = float(np.std(vals))
        scale = iqr if (iqr > 1e-8 * max(sd, 1.0) and iqr > 0.0) else (sd if sd > 1e-12 else 1.0)
        Zc[:, j] = np.where(np.isfinite(Xc[:, j]), np.clip((Xc[:, j] - med) / scale, -12.0, 12.0), 0.0)
        Zq[:, j] = np.where(np.isfinite(Xq[:, j]), np.clip((Xq[:, j] - med) / scale, -12.0, 12.0), 0.0)
    return Zc, Zq


def _inject_synth_cols_np(Xc: np.ndarray, Xq: np.ndarray, k_inj: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Inject `k_inj` synthetic columns (7 H419 kinds: noise/Cauchy distractors, collinear, bilinear, warps, quantile steps, Fourier) fitted strictly on `Xc`."""
    p = Xc.shape[1]
    if p == 0 or k_inj <= 0:
        return np.zeros((Xc.shape[0], 0)), np.zeros((Xq.shape[0], 0))
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z_all = np.vstack([Zc, Zq])
    n_tot = Z_all.shape[0]
    n_c = Xc.shape[0]
    inj_cols = np.zeros((n_tot, k_inj), dtype=np.float64)

    for m in range(k_inj):
        kind = int(rng.integers(0, 7))
        ia = int(rng.integers(0, p))
        ib = int(rng.integers(0, p))
        za = Z_all[:, ia]
        zb = Z_all[:, ib]
        alt = bool(rng.random() < 0.5)
        if kind == 0:
            # Gaussian or Uniform pure distractor
            col = rng.standard_normal(n_tot) if alt else (rng.random(n_tot) * 2.0 - 1.0) * 1.732
        elif kind == 1:
            # Heavy-tailed Student-t3 or clipped Cauchy distractor
            if alt:
                e = rng.standard_normal(n_tot)
                chi = np.sqrt(np.maximum(np.mean(rng.standard_normal((n_tot, 3)) ** 2, axis=1), 1e-3))
                col = np.clip(e / chi, -25.0, 25.0)
            else:
                u = np.clip(rng.random(n_tot), 1e-3, 1.0 - 1e-3)
                col = np.clip(np.tan(math.pi * (u - 0.5)), -25.0, 25.0)
        elif kind == 2:
            # Noisy collinear combination of two real features
            a = float(rng.random())
            col = a * za + (1.0 - a) * zb + 0.15 * rng.standard_normal(n_tot)
        elif kind == 3:
            # Bilinear interaction or quadratic
            col = np.clip(za * zb if alt else (za * za - 1.0), -30.0, 30.0)
        elif kind == 4:
            # Signed-log warp or context-fitted rank-Gauss normal score
            if alt:
                col = np.sign(za) * np.log1p(np.abs(za))
            else:
                c_sorted = np.sort(za[:n_c])
                ranks = (np.searchsorted(c_sorted, za, side="left") + 0.5) / float(n_c + 1)
                col = math.sqrt(2.0) * erfinv(np.clip(2.0 * ranks - 1.0, -0.9999, 0.9999))
        elif kind == 5:
            # Context-quantile step indicator
            qv = float(np.percentile(za[:n_c], 25.0 if alt else 75.0))
            col = (za > qv).astype(np.float64)
        else:
            # Random Fourier feature
            w = float(rng.standard_normal() * math.exp(rng.uniform(math.log(0.5), math.log(3.0))))
            ph = float(rng.random() * 2.0 * math.pi)
            col = np.sin(w * za + ph) if alt else np.cos(w * za + ph)
        inj_cols[:, m] = col
    return inj_cols[:n_c], inj_cols[n_c:]


def _synth_teacher_target_np(Xc: np.ndarray, Xq: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """Synthesize a rich nonlinear GP/RFF + Oblique-Tree + Monotone-Warp teacher target on real table features."""
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z = np.vstack([Zc, Zq])
    n_tot, p = Z.shape
    if p == 0 or n_tot < 24:
        return None
    sub_k = min(p, int(rng.integers(2, min(14, p) + 1)))
    cols = rng.choice(p, size=sub_k, replace=False)
    Zs = Z[:, cols]

    # Smooth RFF / ARD kernel component
    ls = np.exp(rng.uniform(-0.7, 1.2, size=(1, sub_k)))
    m_rff = 48
    W = rng.standard_normal((sub_k, m_rff)) / ls.T
    b = rng.uniform(0.0, 2.0 * math.pi, size=(1, m_rff))
    phi = np.cos(Zs @ W + b) * math.sqrt(2.0 / m_rff)
    f_gp = phi @ rng.standard_normal(m_rff)

    # Oblique tree ensemble component
    f_tree = np.zeros(n_tot, dtype=np.float64)
    for _ in range(4):
        depth = int(rng.integers(2, 5))
        leaf_idx = np.zeros(n_tot, dtype=np.int64)
        for d in range(depth):
            proj = Zs @ rng.standard_normal(sub_k)
            thr = float(np.percentile(proj[: Xc.shape[0]], float(rng.uniform(20.0, 80.0))))
            leaf_idx = (leaf_idx << 1) | (proj > thr).astype(np.int64)
        leaf_vals = rng.standard_normal(1 << depth)
        f_tree += leaf_vals[leaf_idx] * 0.5

    u = float(rng.random())
    f = f_gp if u < 0.35 else (f_tree if u < 0.65 else (0.6 * f_gp + 0.4 * f_tree))
    sd = float(np.std(f))
    if not math.isfinite(sd) or sd <= 1e-9:
        return None
    f = (f - float(np.mean(f))) / sd

    # Optional monotone warp + realistic heteroskedastic noise
    w_kind = int(rng.integers(0, 4))
    if w_kind == 1:
        f = np.sign(f) * (np.abs(f) ** float(rng.uniform(0.6, 1.8)))
    elif w_kind == 2:
        f = np.sinh(0.65 * f)
    elif w_kind == 3:
        f = f + 0.25 * (f ** 2)

    snr_noise = float(math.exp(rng.uniform(math.log(0.01), math.log(0.22))))
    y = f + snr_noise * float(np.std(f)) * rng.standard_normal(n_tot)
    n_c = Xc.shape[0]
    if float(np.std(y[:n_c])) <= 1e-9 or not np.all(np.isfinite(y)):
        return None
    return y[:n_c], y[n_c:]


def _screen_reviewed_extra_cols(Xc_base: np.ndarray, yc: np.ndarray, Sc: np.ndarray, Sq: np.ndarray, max_keep: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Review candidate extra columns strictly on `(Xc_base, yc)` and retain ONLY signal-bearing / sentinel columns."""
    if Sc.shape[1] == 0:
        return Sc, Sq
    yc0 = yc - float(np.mean(yc))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    keep_scores: list[tuple[float, int]] = []
    for j in range(Sc.shape[1]):
        col = Sc[:, j]
        fin = np.isfinite(col)
        if int(np.sum(fin)) < 8:
            continue
        v = col[fin] - float(np.mean(col[fin]))
        vn = float(np.linalg.norm(v))
        if vn <= 1e-12:
            continue
        r = abs(float(np.dot(v, yc0[fin])) / (vn * yn))
        is_binary_flag = np.unique(col[fin]).size <= 3
        if r >= 0.025 or (is_binary_flag and r >= 0.015):
            keep_scores.append((r, j))
    if not keep_scores:
        return np.zeros((Xc_base.shape[0], 0)), np.zeros((Sq.shape[0], 0))
    keep_scores.sort(key=lambda t: -t[0])
    chosen = [j for _, j in keep_scores[:max_keep]]
    return Sc[:, chosen], Sq[:, chosen]


def _fast_train_nl_base(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Ultra-fast 3-fold `HGB + Laplacian-Matérn` OOF base (`~10ms` on CPU worker) for Stage-2 residual zooming."""
    n_c, p = Xc.shape
    n_q = Xq.shape[0]
    ym = float(np.mean(yc))
    sy = float(np.std(yc))
    if n_c < 36 or p == 0 or sy <= 1e-12:
        return np.full(n_c, ym), np.full(n_q, ym), False
    rng = np.random.default_rng(seed)
    folds = 3
    fold = rng.permutation(n_c) % folds
    oof = np.full(n_c, ym, dtype=np.float64)
    pred_q = np.zeros(n_q, dtype=np.float64)
    p_sub = min(p, 36)
    Xc_w = Xc[:, :p_sub]
    Xq_w = Xq[:, :p_sub]
    for k in range(folds):
        va = fold == k
        tr = ~va
        if not np.any(va) or int(np.sum(tr)) < 16:
            continue
        hgb = HistGradientBoostingRegressor(
            max_iter=45,
            learning_rate=0.08,
            max_leaf_nodes=min(23, max(7, int(np.sum(tr)) // 9)),
            min_samples_leaf=max(5, int(np.sum(tr)) // 50),
            l2_regularization=2.0,
            early_stopping=False,
            random_state=int((seed + 29 * k) & 0x7FFFFFFF),
        )
        hgb.fit(Xc_w[tr], yc[tr])
        p_hgb_va = hgb.predict(Xc_w[va])
        p_hgb_q = hgb.predict(Xq_w)
        p_ker_all = RR.fast_laplacian_matern_predict(Xc_w[tr], yc[tr], np.vstack([Xc_w[va], Xq_w]), max_sub=180)
        n_va = int(np.sum(va))
        oof[va] = 0.60 * p_hgb_va + 0.40 * p_ker_all[:n_va]
        pred_q += (0.60 * p_hgb_q + 0.40 * p_ker_all[n_va:]) / float(folds)

    y_lo, y_hi = float(np.min(yc)), float(np.max(yc))
    span = max(y_hi - y_lo, sy)
    oof = np.clip(oof, y_lo - span, y_hi + span)
    pred_q = np.clip(pred_q, y_lo - span, y_hi + span)
    rc = yc - oof
    rmse_oof = float(np.sqrt(np.mean(rc * rc)))
    ok = bool(np.all(np.isfinite(oof)) and np.all(np.isfinite(pred_q)) and rmse_oof < 0.94 * sy and float(np.std(rc)) >= 5e-4 * sy)
    return oof, pred_q, ok


def _select_wide_cols(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, max_f: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = Xc.shape[1]
    if p <= max_f:
        perm_f = rng.permutation(p)
        return Xc[:, perm_f], Xq[:, perm_f]
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
    n_top = int(round(0.70 * max_f))
    top_idx = np.argsort(-scores)[:n_top]
    rem_pool = np.setdiff1d(np.arange(p), top_idx, assume_unique=False)
    rand_idx = rng.choice(rem_pool, size=max_f - len(top_idx), replace=False)
    chosen = np.concatenate([top_idx, rand_idx])
    rng.shuffle(chosen)
    return Xc[:, chosen], Xq[:, chosen]


def _sample_full_synth_special_gpu(dev: torch.device, B: int, N: int, Fn: int, n_ctx: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize full `float64` Panel/Temporal Lead-Lag OR Sentinel/Hurdle Point-Mass datasets directly for Regime 3."""
    is_panel = bool(rng.random() < 0.55)
    xs = np.zeros((B, N, Fn), dtype=np.float64)
    ys = np.zeros((B, N), dtype=np.float64)

    for b in range(B):
        if is_panel:
            # High-offset `float64` timestamp (`1.70e12 + dt_ms`) + discrete entity ID + AR(1)/smooth latent trajectory
            n_ent = int(rng.integers(6, max(8, min(48, N // 8))))
            ent = rng.integers(0, n_ent, size=N)
            t_step = rng.uniform(0.0, 2000.0, size=N)
            base_epoch_ms = float(rng.choice([1.7040672e12, 20725.0, 15000.0]))
            scale_dt = 60_000.0 if base_epoch_ms > 1e9 else 0.01
            ts_col = base_epoch_ms + t_step * scale_dt
            covs = rng.standard_normal((N, max(1, Fn - 2)))
            ent_base = rng.standard_normal(n_ent) * 2.5
            ent_freq = rng.uniform(0.003, 0.02, size=n_ent)
            ent_ph = rng.uniform(0.0, 2.0 * math.pi, size=n_ent)
            ent_slope = rng.standard_normal(n_ent) * 0.002
            w_cov = rng.standard_normal(covs.shape[1]) * 0.35
            y_clean = (
                ent_base[ent]
                + 1.8 * np.sin(ent_freq[ent] * t_step + ent_ph[ent])
                + ent_slope[ent] * t_step
                + covs @ w_cov
            )
            noise = float(rng.uniform(0.015, 0.12)) * max(float(np.std(y_clean)), 1e-3) * rng.standard_normal(N)
            y_b = y_clean + noise
            X_full = np.column_stack([ts_col, ent.astype(np.float64), covs])[:, :Fn]
            # Shuffle row order so context/query split is a genuine random panel split!
            perm = rng.permutation(N)
            X_full = X_full[perm]
            y_b = y_b[perm]
            if Fn >= 8 and rng.random() < 0.55:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Xc_cl, Xq_cl, Sc, Sq, _ = REL.build_h450_struct_cols(
                        X_full[:n_ctx, : max(2, Fn - 6)],
                        y_b[:n_ctx],
                        X_full[n_ctx:, : max(2, Fn - 6)],
                        seed=int(rng.integers(1, 1 << 30)),
                        fast_mode=True,
                        max_total_cols=6,
                        include_base_struct=False,
                    )
                X_aug_c = np.hstack([Xc_cl, Sc]) if Sc.shape[1] > 0 else Xc_cl
                X_aug_q = np.hstack([Xq_cl, Sq]) if Sq.shape[1] > 0 else Xq_cl
                X_aug = np.vstack([X_aug_c, X_aug_q])
                if X_aug.shape[1] < Fn:
                    X_aug = np.hstack([X_aug, rng.standard_normal((N, Fn - X_aug.shape[1]))])
                X_full = X_aug[:, :Fn]
            col_perm = rng.permutation(Fn)
            xs[b] = X_full[:, col_perm]
            ys[b] = y_b
        else:
            # Synthetic Sentinel / Hurdle Point-Mass dataset (`y = -99.99` or `-999.0` on latent condition + nonlinear bulk)
            X_b = rng.standard_normal((N, Fn))
            w1 = rng.standard_normal(Fn) / math.sqrt(max(Fn, 1))
            w2 = rng.standard_normal(Fn) / math.sqrt(max(Fn, 1))
            y_bulk = 15.0 + 8.0 * np.sin(X_b @ w1) + 5.0 * ((X_b @ w2) ** 2 - 1.0) + 0.6 * rng.standard_normal(N)
            atom_val = float(rng.choice([-99.99, -999.0, 0.0, 9999.0]))
            gate_score = 0.75 * X_b[:, 0] - 0.55 * X_b[:, min(1, Fn - 1)] + 0.25 * rng.standard_normal(N)
            q_thr = float(np.percentile(gate_score, float(rng.uniform(8.0, 22.0))))
            is_atom = gate_score < q_thr
            if rng.random() < 0.65:
                # Mirror sentinel marker in 1-2 feature columns as seen on real industrial tables
                X_b[is_atom, 0] = -99.99
            y_b = np.where(is_atom, atom_val, y_bulk)
            col_perm = rng.permutation(Fn)
            xs[b] = X_b[:, col_perm]
            ys[b] = y_b

    return (
        torch.from_numpy(xs).to(device=dev, dtype=torch.float64),
        torch.from_numpy(ys).to(device=dev, dtype=torch.float64),
    )


def _sample_single_episode(
    bank: CombinedF64Bank,
    tbl_idx: int,
    target_n: int | None,
    target_f: int,
    p_ctx_fixed: float | None,
    seed_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, int, bool, int] | None:
    """Returns `(X_ep, y_ep, n_ctx, is_d91, regime_code)` where `regime_code` is:
      0 = Regime 1 (Pure Original Random-Split Data)
      1 = Regime 2A (Reviewed & Validated Signal Column Addition)
      2 = Regime 2B (Synthetic Column & Teacher Injection onto Real Table)
    """
    rng = np.random.default_rng(np.frombuffer(seed_bytes, dtype=np.uint32))
    tab = bank.tabs_f64[tbl_idx]
    r_tot, c_tot = tab.shape
    if r_tot < 28 or c_tot < 2:
        return None

    u_reg = float(rng.random())
    regime_code = 0 if u_reg < 0.32 else (1 if u_reg < 0.68 else 2)

    # Choose target column (`15%` self-supervised other continuous real column in Regime 1)
    tcol = c_tot - 1
    if regime_code == 0 and c_tot >= 4 and rng.random() < 0.15:
        cand_cols = [j for j in range(c_tot - 1) if int(np.sum(np.isfinite(tab[:, j]))) >= 32]
        if cand_cols:
            pick_c = int(rng.choice(cand_cols))
            vals_c = tab[np.isfinite(tab[:, pick_c]), pick_c]
            if np.unique(vals_c).size >= 12 and float(np.std(vals_c)) > 1e-8:
                tcol = pick_c

    fin_y = np.isfinite(tab[:, tcol])
    if int(np.sum(fin_y)) < 28:
        return None
    tab = tab[fin_y]
    r_tot = tab.shape[0]

    # Optional clean-bulk sentinel filtering 45% of the time in Regimes 1 & 2A
    if regime_code in (0, 1) and rng.random() < 0.45:
        atom = RR.HurdleSentinelDetector.detect_atom(tab[:, tcol])
        if atom is not None:
            keep_bulk = np.abs(tab[:, tcol] - atom) > 1e-7 * max(1.0, abs(atom))
            if int(np.sum(keep_bulk)) >= 32:
                tab = tab[keep_bulk]
                r_tot = tab.shape[0]

    if target_n is None:
        if rng.random() < 0.75:
            n_rows = min(r_tot, MAX_TRAIN_ROWS)
        else:
            hi = max(48, min(r_tot, MAX_TRAIN_ROWS))
            n_rows = int(min(r_tot, max(36, round(math.exp(rng.uniform(math.log(36.0), math.log(float(hi))))))))
    else:
        n_rows = min(r_tot, target_n)

    # Brand-new CSPRNG row permutation on EVERY episode (`0` static splits!)
    perm_r = rng.permutation(r_tot)[:n_rows]
    sub = tab[perm_r]
    if target_n is not None and n_rows < target_n:
        extra = rng.integers(0, n_rows, size=target_n - n_rows)
        sub = np.vstack([sub, sub[extra]])
        n_rows = target_n

    if p_ctx_fixed is None:
        u = float(rng.random())
        p_ctx = float(rng.uniform(0.20, 0.56) if u < 0.25 else (rng.uniform(0.56, 0.84) if u < 0.75 else rng.uniform(0.84, 0.95)))
    else:
        p_ctx = p_ctx_fixed
    n_ctx = max(16, min(n_rows - 4, int(round(p_ctx * n_rows))))

    feat_idxs = [j for j in range(c_tot) if j != tcol]
    X_all_raw = sub[:, feat_idxs]
    y_all_raw = sub[:, tcol].copy()

    Xc_raw = X_all_raw[:n_ctx]
    yc_raw = y_all_raw[:n_ctx].copy()
    Xq_raw = X_all_raw[n_ctx:]
    yq_raw = y_all_raw[n_ctx:].copy()
    if float(np.std(yc_raw)) <= 1e-12:
        return None

    ep_seed = int(rng.integers(1, 1 << 30))

    if regime_code == 0:
        # REGIME 1: Pure Original Data (optional random feature crop `[65%..100%]`, `0` added columns)
        p_orig = Xc_raw.shape[1]
        if p_orig >= 6 and rng.random() < 0.40:
            k_keep = int(rng.integers(max(3, int(round(0.65 * p_orig))), p_orig + 1))
            sub_f = rng.choice(p_orig, size=k_keep, replace=False)
            Xc_work, Xq_work = Xc_raw[:, sub_f], Xq_raw[:, sub_f]
        else:
            Xc_work, Xq_work = Xc_raw, Xq_raw
        yc_work, yq_work = yc_raw, yq_raw

    elif regime_code == 1:
        # REGIME 2A: Reviewed & Validated Signal Column Addition (strictly fitted & screened on `Xc_raw, yc_raw`)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xc_cl, Xq_cl, Sc, Sq, _ = REL.build_h450_struct_cols(
                Xc_raw,
                yc_raw,
                Xq_raw,
                seed=ep_seed,
                fast_mode=True,
                max_total_cols=16,
                include_base_struct=(rng.random() < 0.45),
            )
        Sc_rev, Sq_rev = _screen_reviewed_extra_cols(Xc_cl, yc_raw, Sc, Sq, max_keep=16)
        Xc_work = np.hstack([Xc_cl, Sc_rev]) if Sc_rev.shape[1] > 0 else Xc_cl
        Xq_work = np.hstack([Xq_cl, Sq_rev]) if Sq_rev.shape[1] > 0 else Xq_cl
        yc_work, yq_work = yc_raw, yq_raw

    else:
        # REGIME 2B: Synthetic Column & Teacher Injection onto Real Table
        p_orig = Xc_raw.shape[1]
        if p_orig >= 6 and rng.random() < 0.45:
            k_keep = int(rng.integers(max(2, int(round(0.55 * p_orig))), p_orig + 1))
            sub_f = rng.choice(p_orig, size=k_keep, replace=False)
            Xc_base, Xq_base = Xc_raw[:, sub_f], Xq_raw[:, sub_f]
        else:
            Xc_base, Xq_base = Xc_raw, Xq_raw

        max_inj = max(2, min(18, max(4, target_f - min(Xc_base.shape[1], target_f - 4))))
        k_inj = int(rng.integers(2, max_inj + 1))
        Ic, Iq = _inject_synth_cols_np(Xc_base, Xq_base, k_inj=k_inj, rng=rng)
        Xc_work = np.hstack([Xc_base, Ic]) if Ic.shape[1] > 0 else Xc_base
        Xq_work = np.hstack([Xq_base, Iq]) if Iq.shape[1] > 0 else Xq_base

        if rng.random() < 0.28:
            t_pair = _synth_teacher_target_np(Xc_base, Xq_base, rng=rng)
            if t_pair is not None:
                yc_work, yq_work = t_pair
            else:
                yc_work, yq_work = yc_raw, yq_raw
        else:
            yc_work, yq_work = yc_raw.copy(), yq_raw.copy()
            if rng.random() < 0.30:
                # H317/H419 Regime-4 Huberized Student-t3 target jitter (1.5-3.0% of context std)
                sy_c = float(np.std(yc_work))
                e_all = rng.standard_normal(n_rows) / np.sqrt(np.maximum(np.mean(rng.standard_normal((n_rows, 3)) ** 2, axis=1), 1e-3)) / 1.732
                jit = sy_c * float(rng.uniform(0.015, 0.03)) * np.clip(e_all, -1.345, 1.345)
                yc_work = yc_work + jit[:n_ctx]
                yq_work = yq_work + jit[n_ctx:]

    # 4-mode target & residual curriculum across all real-table regimes
    umode = float(rng.random())
    y_ep = np.concatenate([yc_work, yq_work])

    if umode < 0.40:
        oof_r, pq_r, used_r, _ = RR.robust_ridge_np(Xc_work, yc_work, Xq_work, seed=ep_seed)
        if used_r:
            y_ep[:n_ctx] = yc_work - oof_r
            y_ep[n_ctx:] = yq_work - pq_r
    elif umode < 0.66 and regime_code in (1, 2):
        oof_nl, pq_nl, used_nl = _fast_train_nl_base(Xc_work, yc_work, Xq_work, seed=ep_seed)
        if used_nl:
            Xc_work = np.column_stack([oof_nl, Xc_work])
            Xq_work = np.column_stack([pq_nl, Xq_work])
            y_ep[:n_ctx] = yc_work - oof_nl
            y_ep[n_ctx:] = yq_work - pq_nl
    elif umode < 0.88:
        warp = RR.YeoJohnsonWarpF64()
        zc = warp.fit_transform(yc_work)
        zq = warp.transform_any(yq_work)
        if np.all(np.isfinite(zc)) and np.all(np.isfinite(zq)) and float(np.std(zc)) > 1e-8:
            y_ep[:n_ctx] = zc
            y_ep[n_ctx:] = zq

    Xc_sel, Xq_sel = _select_wide_cols(Xc_work, y_ep[:n_ctx], Xq_work, max_f=target_f, rng=rng)
    X_ep = np.vstack([Xc_sel, Xq_sel])
    if X_ep.shape[1] < target_f and target_n is not None:
        pad_w = target_f - X_ep.shape[1]
        # Fill remaining multi-table batch slots with realistic injected columns rather than dead zeros!
        Ic_pad, Iq_pad = _inject_synth_cols_np(Xc_sel, Xq_sel, k_inj=pad_w, rng=rng)
        X_pad = np.vstack([Ic_pad, Iq_pad])
        perm_f = rng.permutation(target_f)
        X_ep = np.hstack([X_ep, X_pad])[:, perm_f]

    if not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx])) <= 1e-12:
        return None
    return X_ep, y_ep, n_ctx, bool(bank.is_d91[tbl_idx]), regime_code


def _episode_worker_loop(
    worker_id: int,
    run_nonce: bytes,
    bank: CombinedF64Bank,
    tokens_cap: int,
    row_tokens_cap: int,
    bmax: int,
    out_q: mp.Queue,
    stop_ev: mp.Event,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    seq = 0
    while not stop_ev.is_set():
        seq += 1
        b_seed = hashlib.blake2b(run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint32))

        pick_d91 = len(bank.d91_indices) > 0 and (float(rng.random()) < bank.d91_mix_prob)
        if float(rng.random()) < 0.62:
            idx = int(
                rng.choice(bank.d91_indices, p=bank.d91_weights)
                if pick_d91
                else rng.choice(bank.non_d91_indices, p=bank.non_d91_weights)
            )
            ep_seed = hashlib.blake2b(b_seed + b"\x01", digest_size=32).digest()
            got = _sample_single_episode(bank, idx, target_n=None, target_f=WIDE_GPU, p_ctx_fixed=None, seed_bytes=ep_seed)
            if got is None:
                continue
            X_ep, y_ep, n_ctx, is_d91, reg_code = got
            xb = X_ep[None, :, :]
            yb = y_ep[None, :]
            cells = int(xb.shape[1] * (xb.shape[2] + 1))
            reg_counts = (1 if reg_code == 0 else 0, 1 if reg_code == 1 else 0, 1 if reg_code == 2 else 0)
            item = (xb, yb, int(n_ctx), cells if is_d91 else 0, cells, 1, reg_counts)
        else:
            N_tgt = int(rng.choice([192, 320, 512, 768, 1024], p=[0.12, 0.18, 0.25, 0.25, 0.20]))
            F_tgt = int(rng.choice([24, 36, 48, 64], p=[0.15, 0.25, 0.30, 0.30]))
            u = float(rng.random())
            p_ctx = float(rng.uniform(0.25, 0.56) if u < 0.25 else (rng.uniform(0.56, 0.84) if u < 0.75 else rng.uniform(0.84, 0.94)))
            B_tgt = max(1, min(bmax, tokens_cap // max(N_tgt * (F_tgt + 4), 1), row_tokens_cap // max(N_tgt, 1)))
            xs, ys = [], []
            d91_c = 0
            c_r0 = c_r1 = c_r2 = 0
            n_ctx_shared = max(16, min(N_tgt - 4, int(round(p_ctx * N_tgt))))
            for b in range(B_tgt):
                use_d91 = len(bank.d91_indices) > 0 and (float(rng.random()) < bank.d91_mix_prob)
                t_idx = int(
                    rng.choice(bank.d91_indices, p=bank.d91_weights)
                    if use_d91
                    else rng.choice(bank.non_d91_indices, p=bank.non_d91_weights)
                )
                ep_seed = hashlib.blake2b(b_seed + (b + 2).to_bytes(4, "little"), digest_size=32).digest()
                got = _sample_single_episode(bank, t_idx, target_n=N_tgt, target_f=F_tgt, p_ctx_fixed=p_ctx, seed_bytes=ep_seed)
                if got is not None and got[2] == n_ctx_shared and got[0].shape == (N_tgt, F_tgt):
                    xs.append(got[0])
                    ys.append(got[1])
                    if got[3]:
                        d91_c += int(N_tgt * (F_tgt + 1))
                    rc = got[4]
                    if rc == 0:
                        c_r0 += 1
                    elif rc == 1:
                        c_r1 += 1
                    else:
                        c_r2 += 1
            if not xs:
                continue
            xb = np.stack(xs, axis=0)
            yb = np.stack(ys, axis=0)
            tot_c = int(xb.shape[0] * xb.shape[1] * (xb.shape[2] + 1))
            item = (xb, yb, n_ctx_shared, d91_c, tot_c, len(xs), (c_r0, c_r1, c_r2))

        while not stop_ev.is_set():
            try:
                out_q.put(item, timeout=1.0)
                break
            except queue.Full:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=0.50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--d91-mix", type=float, default=0.35)
    ap.add_argument("--synth-prob", type=float, default=0.22)
    ap.add_argument("--tokens", type=int, default=140_000)
    ap.add_argument("--row-tokens", type=int, default=4_096)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=450_101)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_p = out_dir / "ckpt_h450_final.pt"
    swa_p = out_dir / "ckpt_swa_soup.pt"
    if final_p.exists() or swa_p.exists():
        raise SystemExit(f"{final_p} or {swa_p} already exists (create-once)")

    t_load0 = time.time()
    bank = CombinedF64Bank(Path(a.bank), d91_mix_prob=a.d91_mix)
    bank_load_s = round(time.time() - t_load0, 2)

    run_nonce = hashlib.blake2b(secrets.token_bytes(32) + int(a.seed).to_bytes(8, "little"), digest_size=32).digest()

    ctx = mp.get_context("fork")
    ep_q: mp.Queue = ctx.Queue(maxsize=max(16, 4 * a.workers))
    stop_ev = ctx.Event()
    procs = []
    for wid in range(a.workers):
        p = ctx.Process(
            target=_episode_worker_loop,
            args=(wid, run_nonce, bank, a.tokens, a.row_tokens, a.bmax, ep_q, stop_ev),
            daemon=True,
        )
        p.start()
        procs.append(p)

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    w_init = ck.get("ema") or ck["model"]
    model = ICLH450(a.size).to(dev)
    model.load_state_dict(w_init, strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    params_cnt = int(n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)
    prior = Prior(dev, a.seed + 17)
    py_rng = np.random.default_rng(a.seed + 33)

    budget_s = max(15.0, a.hours * 3600.0)
    swa_marks = [0.40, 0.60, 0.80]
    swa_captured = [False, False, False]
    swa_sum_sd: dict[str, torch.Tensor] | None = None
    swa_count = 0

    def _accumulate_swa(ema_mod: torch.nn.Module) -> None:
        nonlocal swa_sum_sd, swa_count
        sd_cpu = {k: v.detach().cpu().double() for k, v in ema_mod.state_dict().items()}
        if swa_sum_sd is None:
            swa_sum_sd = sd_cpu
        else:
            for k in swa_sum_sd:
                swa_sum_sd[k] += sd_cpu[k]
        swa_count += 1

    state = {
        "step": 0,
        "micro_batches": 0,
        "datasets": 0,
        "cum_unique_random_splits": 0,
        "regime_orig_episodes": 0,
        "regime_reviewed_aug_episodes": 0,
        "regime_synth_inject_episodes": 0,
        "regime_full_synth_episodes": 0,
        "cum_row_tokens": 0,
        "cum_cell_tokens": 0,
        "cum_real_cell_tokens": 0,
        "cum_d91_cell_tokens": 0,
        "approx_train_flops": 0,
        "oom_skips": 0,
        "elapsed_s": 0.0,
    }

    log_f = open(out_dir / "log.jsonl", "a", buffering=1)
    t0 = time.time()
    t_log = t0
    run_loss = 0.0
    run_n = 0
    grad_accum = max(1, int(a.grad_accum))
    opt.zero_grad(set_to_none=True)

    try:
        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed >= budget_s:
                break
            if a.max_steps > 0 and state["step"] >= a.max_steps:
                break

            frac = min(1.0, elapsed / budget_s)
            for idx_m, mk in enumerate(swa_marks):
                if not swa_captured[idx_m] and frac >= mk:
                    _accumulate_swa(ema)
                    swa_captured[idx_m] = True

            lr_mult = (frac / 0.06) if frac < 0.06 else (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * (frac - 0.06) / 0.94)))
            cur_lr = a.lr * max(0.08, lr_mult)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            # REGIME 3: Full Synthetic Datasets (`--synth-prob 0.22`)
            if float(py_rng.random()) < a.synth_prob:
                import random as std_random
                r_py = std_random.Random(int(py_rng.integers(1, 1 << 30)))
                Fn_s = max(6, min(WIDE_GPU, draw_features(r_py)))
                N_s = int(py_rng.choice([256, 512, 768, 1024], p=[0.20, 0.35, 0.25, 0.20]))
                B_s = max(1, min(a.bmax, a.tokens // max(N_s * (Fn_s + 4), 1), a.row_tokens // max(N_s, 1)))
                n_ctx_s = max(16, min(N_s - 4, int(round(float(py_rng.uniform(0.25, 0.93)) * N_s))))
                if float(py_rng.random()) < 0.60:
                    xt, yt, _, _ = INJ.sample_synth(prior, r_py, B_s, N_s, Fn_s, 0.55, PROBS_V3S)
                    xt = xt.double()
                    yt = yt.double()
                else:
                    xt, yt = _sample_full_synth_special_gpu(dev, B_s, N_s, Fn_s, n_ctx_s, py_rng)
                if float(py_rng.random()) < 0.45:
                    yt_f, _, _ = RR.robust_residual_targets_torch(xt.float(), yt.float(), n_ctx_s, seed=int(py_rng.integers(1, 1 << 30)))
                    yt = yt_f.double()
                d91_c, real_c, tot_c, n_sp = 0, 0, int(B_s * N_s * (Fn_s + 1)), int(B_s)
                reg_counts = (0, 0, 0)
                full_synth_eps = int(B_s)
            else:
                xb, yb, n_ctx_s, d91_c, real_c, n_sp, reg_counts = ep_q.get(timeout=60.0)
                tot_c = real_c
                full_synth_eps = 0
                xt = torch.from_numpy(xb).to(device=dev, dtype=torch.float64, non_blocking=True)
                yt = torch.from_numpy(yb).to(device=dev, dtype=torch.float64, non_blocking=True)

            Fn_cur = xt.shape[2]
            slots = torch.randperm(FID_SLOTS, device=dev)[:Fn_cur]

            try:
                logits, ym, ys = model(xt, yt[:, :n_ctx_s], n_ctx_s, slots=slots, amp=(dev.type == "cuda"))
                loss = model.loss(logits, yt[:, n_ctx_s:], ym, ys).mean()
                if not torch.isfinite(loss):
                    continue
                (loss / float(grad_accum)).backward()
            except torch.cuda.OutOfMemoryError:
                state["oom_skips"] += 1
                opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            B_cur, N_cur = int(xt.shape[0]), int(xt.shape[1])
            state["micro_batches"] += 1
            state["datasets"] += B_cur
            state["cum_unique_random_splits"] += int(n_sp)
            state["regime_orig_episodes"] += int(reg_counts[0])
            state["regime_reviewed_aug_episodes"] += int(reg_counts[1])
            state["regime_synth_inject_episodes"] += int(reg_counts[2])
            state["regime_full_synth_episodes"] += int(full_synth_eps)
            state["cum_row_tokens"] += int(B_cur * N_cur)
            state["cum_cell_tokens"] += int(tot_c)
            state["cum_real_cell_tokens"] += int(real_c)
            state["cum_d91_cell_tokens"] += int(d91_c)
            state["approx_train_flops"] += int(6 * params_cnt * tot_c)
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

            now2 = time.time()
            state["elapsed_s"] = round(now2 - t0, 3)
            if now2 - t_log >= 20.0 and run_n > 0:
                mem_gb = round(torch.cuda.max_memory_allocated() / float(1 << 30), 2) if dev.type == "cuda" else 0.0
                rec = {
                    "step": state["step"],
                    "micro_batches": state["micro_batches"],
                    "elapsed_s": state["elapsed_s"],
                    "loss": round(run_loss / run_n, 5),
                    "lr": round(cur_lr, 8),
                    "unique_random_splits": state["cum_unique_random_splits"],
                    "regimes": {
                        "orig": state["regime_orig_episodes"],
                        "reviewed_aug": state["regime_reviewed_aug_episodes"],
                        "synth_inject": state["regime_synth_inject_episodes"],
                        "full_synth": state["regime_full_synth_episodes"],
                    },
                    "cell_tokens_m": round(state["cum_cell_tokens"] / 1e6, 2),
                    "real_cell_tokens_m": round(state["cum_real_cell_tokens"] / 1e6, 2),
                    "d91_cell_tokens_m": round(state["cum_d91_cell_tokens"] / 1e6, 2),
                    "oom_skips": state["oom_skips"],
                    "mem_gb": mem_gb,
                }
                log_f.write(json.dumps(rec) + "\n")
                run_loss, run_n, t_log = 0.0, 0, now2
    finally:
        stop_ev.set()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=2.0)

    state["elapsed_s"] = round(time.time() - t0, 3)
    _accumulate_swa(ema)

    assert swa_sum_sd is not None and swa_count >= 1
    swa_sd = {k: (v / float(swa_count)).to(dtype=ema.state_dict()[k].dtype) for k, v in swa_sum_sd.items()}

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "state": state, "size": a.size}, final_p)
    torch.save({"model": swa_sd, "ema": swa_sd, "state": state, "size": a.size, "swa_snapshots": swa_count}, swa_p)
    os.chmod(final_p, 0o444)
    os.chmod(swa_p, 0o444)

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    manifest = {
        "schema": "jope.h450.random_split_ckpt_manifest/v2",
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "size": a.size,
        "params": params_cnt,
        "device": str(dev),
        "gpu_name": gpu_name,
        "seed": a.seed,
        "run_nonce_sha256": hashlib.sha256(run_nonce).hexdigest(),
        "init_ckpt": str(a.init),
        "init_sha256": _sha256_file(Path(a.init)),
        "bank": str(a.bank),
        "bank_sha256": bank.sha256,
        "bank_tables": bank.n_tables,
        "bank_d91_tables": int(len(bank.d91_indices)),
        "bank_load_s": bank_load_s,
        "d91_mix_prob": a.d91_mix,
        "synth_prob": a.synth_prob,
        "swa_snapshots": swa_count,
        "ckpt_final_path": str(final_p),
        "ckpt_final_sha256": _sha256_file(final_p),
        "ckpt_swa_soup_path": str(swa_p),
        "ckpt_sha256": _sha256_file(swa_p),
        "state": state,
    }
    man_p = out_dir / "manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(man_p, 0o444)
    log_f.close()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
