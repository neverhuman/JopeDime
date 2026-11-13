#!/usr/bin/env python3
"""104M (`ICL_L`) MPS/CUDA Trainer with Bidirectional Feature Synthesis, Pre-Forward 128-bit Dedup,
Zero-Copy `MmapF64Bank`, and Host CPU Governor (`<= 85%` average) (`train_mps_104m_h450.py`).

Designed specifically for `xbabe4` (Mac Studio M3 Ultra, 80-core Metal `mps` GPU, 512 GiB Unified Memory) and portable
to `cuda`/`cpu`, using `mp.get_context("spawn")` + `MmapF64Bank` (`< 60 MiB` worker RSS, zero macOS fork issues).
Writes `ckpt_epoch_NN.pt` + `epoch_NN.json` in exact `cols_h450_gpu.py` / `jope.h450.epoch/v1` format so checkpoints
plug straight into the fleet `dev91` frozen-split Elo evaluation pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import random as std_random
import secrets
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
for _cand in (
    str(HERE),
    os.environ.get("JOPE_REF_CODE", ""),
    "/home/ubuntu/h450-stage/pub-r5/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    str(Path.home() / "jope-data/h450-fleet-agy/ref_code"),
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

import icl_inject as INJ  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_mps_h450 import ICLH450MPS, n_params  # noqa: E402
from icl_prior_v3s import PROBS_V3S, Prior, draw_features  # noqa: E402
from mmap_bank_h450 import MmapF64Bank, sha256_file  # noqa: E402


def _erfinv_np(x: np.ndarray) -> np.ndarray:
    return torch.erfinv(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()


def _detect_atom_np(y: np.ndarray) -> float | None:
    yf = np.asarray(y, dtype=np.float64)
    yf = yf[np.isfinite(yf)]
    if yf.size < 32:
        return None
    vals, cnts = np.unique(yf, return_counts=True)
    top_i = int(np.argmax(cnts))
    cnt = int(cnts[top_i])
    val = float(vals[top_i])
    if cnt < 6 or cnt / float(yf.size) < 0.05 or cnt >= yf.size - 16:
        return None
    rest = yf[yf != val]
    if rest.size < 16:
        return None
    med = float(np.median(rest))
    q25, q75 = np.percentile(rest, [25.0, 75.0])
    scale = max((q75 - q25) / 1.349, float(np.std(rest)), 1e-9)
    if abs(val - med) >= 2.5 * scale or val in (-99.99, -999.0, -9999.0, 9999.0):
        return val
    return None


def _robust_ridge_np(X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, seed: int = 450, folds: int = 5):
    """Pure NumPy SVD-GCV 5-fold OOF robust ridge (identical math to `robust_resid_h450.robust_ridge_np`, zero scipy)."""
    X_ctx = np.asarray(X_ctx, dtype=np.float64)
    y_ctx = np.asarray(y_ctx, dtype=np.float64)
    X_q = np.asarray(X_q, dtype=np.float64)
    n_c, p = X_ctx.shape
    n_q = X_q.shape[0]
    ym = float(np.mean(y_ctx))
    sy = float(np.std(y_ctx))
    if n_c < 16 or p == 0 or sy <= 1e-12:
        return np.full(n_c, ym), np.full(n_q, ym), False, 0.0
    miss_c = ~np.isfinite(X_ctx)
    miss_q = ~np.isfinite(X_q)
    med = np.zeros(p, dtype=np.float64)
    sc = np.ones(p, dtype=np.float64)
    for j in range(p):
        fin = ~miss_c[:, j]
        if np.any(fin):
            v = X_ctx[fin, j]
            med[j] = float(np.median(v))
            q25, q75 = np.quantile(v, [0.25, 0.75]) if len(v) >= 4 else (med[j] - 1.0, med[j] + 1.0)
            iqr = float(q75 - q25) / 1.349
            sd = float(np.std(v))
            sc[j] = iqr if iqr > 1e-8 * max(sd, 1.0) and iqr > 0.0 else (sd if sd > 1e-12 else 1.0)
    zc = 6.0 * np.tanh(np.clip((np.where(miss_c, med[None, :], X_ctx) - med[None, :]) / sc[None, :], -30.0, 30.0) / 6.0)
    zq = 6.0 * np.tanh(np.clip((np.where(miss_q, med[None, :], X_q) - med[None, :]) / sc[None, :], -30.0, 30.0) / 6.0)
    miss_cols = [j for j in range(p) if float(np.mean(miss_c[:, j])) >= 0.01][:16]
    if miss_cols:
        zc = np.hstack([zc, miss_c[:, miss_cols].astype(np.float64)])
        zq = np.hstack([zq, miss_q[:, miss_cols].astype(np.float64)])
    mu_z = zc.mean(axis=0, keepdims=True)
    zc0, zq0, yc0 = zc - mu_z, zq - mu_z, y_ctx - ym
    U, s, Vt = np.linalg.svd(np.where(np.isfinite(zc0), zc0, 0.0), full_matrices=False)
    s2 = s * s
    uty = U.T @ yc0
    U2 = U * U
    best_lam, best_loo_mse = 1.0, float("inf")
    for lam in (0.15, 1.0, 8.0, 60.0):
        filt = s2 / (s2 + lam)
        yhat = U @ (filt * uty)
        h = np.clip(U2 @ filt + (1.0 / n_c), 0.0, 0.98)
        loo_err = (yc0 - yhat) / (1.0 - h)
        mse = float(np.mean(loo_err * loo_err))
        if mse < best_loo_mse:
            best_loo_mse, best_lam = mse, lam
    w_full = Vt.T @ ((s / (s2 + best_lam)) * uty)
    pred_q = zq0 @ w_full + ym
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n_c) % folds
    oof = np.full(n_c, ym, dtype=np.float64)
    d = zc0.shape[1]
    G_full, rhs_full, s_full, sy_full = zc0.T @ zc0, zc0.T @ yc0, zc0.sum(axis=0), float(yc0.sum())
    eye = best_lam * np.eye(d, dtype=np.float64)
    for k in range(folds):
        mk = fold == k
        nk = int(np.sum(mk))
        if nk == 0 or nk >= n_c:
            continue
        zk, yk = zc0[mk], yc0[mk]
        ntr = float(n_c - nk)
        ym_tr = (sy_full - float(yk.sum())) / ntr
        s_tr = s_full - zk.sum(axis=0)
        G_tr = G_full - zk.T @ zk - np.outer(s_tr, s_tr) / ntr + eye
        b_tr = (rhs_full - zk.T @ yk) - ym_tr * s_tr
        try:
            wk = np.linalg.solve(G_tr, b_tr)
        except Exception:
            wk = np.linalg.lstsq(G_tr, b_tr, rcond=1e-8)[0]
        oof[mk] = (zk - (s_tr / ntr)[None, :]) @ wk + ym + ym_tr
    y_lo, y_hi = float(np.min(y_ctx)), float(np.max(y_ctx))
    span = max(y_hi - y_lo, sy)
    oof = np.clip(oof, y_lo - 1.5 * span, y_hi + 1.5 * span)
    pred_q = np.clip(pred_q, y_lo - 1.5 * span, y_hi + 1.5 * span)
    rc = y_ctx - oof
    rmse_oof = float(np.sqrt(np.mean(rc * rc)))
    u_oof = -math.log(max(rmse_oof / sy, 1e-12))
    used = bool(np.all(np.isfinite(oof)) and np.all(np.isfinite(pred_q)) and rmse_oof < 0.985 * sy and float(np.std(rc)) >= 5e-4 * sy)
    return oof, pred_q, used, u_oof


def _yj_eval_np(yn: np.ndarray, lam: float) -> np.ndarray:
    pos = yn >= 0.0
    out = np.empty_like(yn, dtype=np.float64)
    if abs(lam) < 1e-6:
        out[pos] = np.log1p(yn[pos])
    else:
        out[pos] = (np.power(yn[pos] + 1.0, lam) - 1.0) / lam
    lam2 = 2.0 - lam
    if abs(lam2) < 1e-6:
        out[~pos] = -np.log1p(-yn[~pos])
    else:
        out[~pos] = -(np.power(1.0 - yn[~pos], lam2) - 1.0) / lam2
    return out


def _yeojohnson_pair_np(yc: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact pure-NumPy Yeo-Johnson MLE fit on `yc` applied to `(yc, yq)` (zero scipy)."""
    mu = float(np.mean(yc))
    sd = max(float(np.std(yc)), 1e-12)
    ync = (yc - mu) / sd
    ynq = (yq - mu) / sd
    if ync.size < 16:
        return None
    skew = abs(float(np.mean(ync ** 3)))
    if skew < 0.25:
        return ync, ynq
    sgn_log = np.sign(ync) * np.log1p(np.abs(ync))
    sum_sl = float(np.sum(sgn_log))
    n = float(ync.size)
    best_lam, best_ll = 1.0, -float("inf")
    for lam in np.linspace(-1.5, 2.5, 41):
        zt = _yj_eval_np(ync, float(lam))
        v = float(np.var(zt))
        if v <= 1e-18 or not math.isfinite(v):
            continue
        ll = -0.5 * n * math.log(v) + (float(lam) - 1.0) * sum_sl
        if ll > best_ll:
            best_ll, best_lam = ll, float(lam)
    zt_c = _yj_eval_np(ync, best_lam)
    zt_q = _yj_eval_np(ynq, best_lam)
    z_mu = float(np.mean(zt_c))
    z_sd = float(np.std(zt_c))
    if z_sd <= 1e-9 or not (np.all(np.isfinite(zt_c)) and np.all(np.isfinite(zt_q))):
        return None
    return (zt_c - z_mu) / z_sd, (zt_q - z_mu) / z_sd

WIDE_GPU = 64
MAX_TRAIN_ROWS = 2048
BATCH_N = (192, 320, 512, 768, 1024)
BATCH_N_P = (0.15, 0.22, 0.28, 0.20, 0.15)


class HostCpuGovernor:
    """Keeps total host CPU utilization <= `hard_pct` (default 82.0%, strictly under the 85% fleet ceiling)."""

    def __init__(self, hard_pct: float = 82.0, check_every_s: float = 2.5):
        self.hard_pct = float(hard_pct)
        self.check_every_s = float(check_every_s)
        self.is_darwin = platform.system() == "Darwin"
        self.n_cpus = max(1, os.cpu_count() or 1)
        self._last_t = 0.0
        self._last_pct = 0.0
        self._prev_stat: tuple[int, int] | None = None

    def sample_pct(self) -> float:
        now = time.time()
        if now - self._last_t < self.check_every_s:
            return self._last_pct
        self._last_t = now
        try:
            if not self.is_darwin and Path("/proc/stat").is_file():
                parts = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                tot = sum(parts)
                if self._prev_stat is not None:
                    d_tot = tot - self._prev_stat[0]
                    d_idl = idle - self._prev_stat[1]
                    if d_tot > 0:
                        self._last_pct = 100.0 * (1.0 - d_idl / float(d_tot))
                self._prev_stat = (tot, idle)
            else:
                r = subprocess.run(["ps", "-A", "-o", "%cpu"], capture_output=True, text=True, timeout=1.5)
                if r.returncode == 0:
                    s = sum(float(x) for x in r.stdout.splitlines()[1:] if x.strip())
                    self._last_pct = min(100.0, s / float(self.n_cpus))
        except Exception:
            pass
        return self._last_pct

    def pace(self) -> float:
        pct = self.sample_pct()
        if pct > self.hard_pct:
            over = min(15.0, pct - self.hard_pct)
            time.sleep(min(0.35, 0.025 * over))
        return pct


def dataset_fingerprint_128(X: np.ndarray, y: np.ndarray, n_ctx: int) -> bytes:
    """Row-order-invariant 128-bit `blake2b` fingerprint canonicalizing `-0.0/+0.0` and all non-finite sentinels."""
    Xf = np.asarray(X, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    miss_x = (~np.isfinite(Xf)).astype(np.float64)
    Xc = np.where(np.isfinite(Xf), Xf, 1.7976931348623157e308)
    yc = np.where(np.isfinite(yf), yf, 1.7976931348623157e308)
    Z = np.column_stack([Xc, miss_x, yc])
    Z = np.where(Z == 0.0, 0.0, Z)  # canonicalize -0.0 -> +0.0 before lexsort & tobytes
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([Xf.shape[0], Xf.shape[1], int(n_ctx)], dtype=np.int64).tobytes())
    for part in (Z[:n_ctx], Z[n_ctx:]):
        if part.shape[0] > 1:
            part = part[np.lexsort(part.T[::-1])]
        h.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
    return h.digest()


def _std_ctx_f64(Xc: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                col = math.sqrt(2.0) * _erfinv_np(np.clip(2.0 * ranks - 1.0, -0.9999, 0.9999))
        elif kind == 5:
            col = (za > float(np.percentile(za[:n_c], 25.0 if alt else 75.0))).astype(np.float64)
        else:
            w = float(rng.standard_normal() * math.exp(rng.uniform(math.log(0.5), math.log(3.0))))
            ph = float(rng.random() * 2.0 * math.pi)
            col = np.sin(w * za + ph) if alt else np.cos(w * za + ph)
        inj_cols[:, m] = col
    return inj_cols[:n_c], inj_cols[n_c:]


def _synth_teacher_target_np(Xc: np.ndarray, Xq: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
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


def bidirectional_real_episode(
    bank: MmapF64Bank,
    t_idx: int,
    rng: np.random.Generator,
    n_rows_fixed: int | None = None,
    f_fixed: int | None = None,
    p_ctx_fixed: float | None = None,
    random_split_every_batch: bool = True,
    feature_drop_every_batch: bool = True,
    drop_max_frac: float = 0.35,
    synth_inj_every_batch: bool = True,
    inj_max_frac: float = 0.10,
):
    """Samples one episode with fresh random train/test split, mandatory random feature removal (`p >= 2`),
    and strict `k <= floor(inj_max_frac * F)` (`<= 10%`) synthetic feature injection + FROM-feature targets."""
    inj_max_frac = max(0.0, min(0.10, float(inj_max_frac)))
    drop_max_frac = max(0.05, min(0.50, float(drop_max_frac)))
    tab = np.asarray(bank.tabs_f64[t_idx], dtype=np.float64)
    c_tot = tab.shape[1]
    if tab.shape[0] < 28 or c_tot < 2:
        return None
    # Target-from-features modes: 0 = own target (65%), 1 = masked real feature column X_{-j}->X_j (17%),
    # 2 = nonlinear synthetic teacher g(X_sub) (13%) or convex real+teacher blend (5%)
    u = float(rng.random())
    tmode = 0 if u < 0.65 else (1 if u < 0.82 else 2)
    blend_real_teacher = (u >= 0.95)
    tcol = c_tot - 1
    if tmode == 1:
        cand = [j for j in range(c_tot - 1) if int(np.sum(np.isfinite(tab[:, j]))) >= 32]
        rng.shuffle(cand)
        tcol = next((j for j in cand[:8] if np.unique(tab[np.isfinite(tab[:, j]), j]).size >= 12), c_tot - 1)
        if tcol == c_tot - 1:
            tmode = 0
    if tmode in (0, 1) or blend_real_teacher:
        tab = tab[np.isfinite(tab[:, tcol])]
        if tmode == 0 and rng.random() < 0.45 and tab.shape[0] >= 32:
            atom = _detect_atom_np(tab[:, tcol])
            if atom is not None:
                keep = np.abs(tab[:, tcol] - atom) > 1e-7 * max(1.0, abs(atom))
                if int(np.sum(keep)) >= 32 and (n_rows_fixed is None or int(np.sum(keep)) >= n_rows_fixed):
                    tab = tab[keep]
    r_tot = tab.shape[0]
    if r_tot < 28:
        return None
    if n_rows_fixed is None:
        if rng.random() < 0.70:
            n_rows = min(r_tot, MAX_TRAIN_ROWS)
        else:
            hi = max(48, min(r_tot, MAX_TRAIN_ROWS))
            n_rows = int(min(r_tot, max(36, round(math.exp(rng.uniform(math.log(36.0), math.log(float(hi))))))))
    else:
        if r_tot < n_rows_fixed:
            return None
        n_rows = n_rows_fixed

    ycol = tab[:, tcol]
    if r_tot > n_rows and rng.random() < 0.75:
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
        u_p = float(rng.random())
        p_ctx = float(rng.uniform(0.20, 0.56) if u_p < 0.25 else (rng.uniform(0.56, 0.84) if u_p < 0.75 else rng.uniform(0.84, 0.95)))
    else:
        p_ctx = p_ctx_fixed
    n_ctx = max(16, min(n_rows - 4, int(round(p_ctx * n_rows))))
    feats = [j for j in range(c_tot) if j != tcol]
    if tmode == 2 and not blend_real_teacher and len(feats) > 1 and rng.random() < 0.5:
        feats = [j for j in feats if j != c_tot - 1] or feats
    X_all = sub[:, feats]
    y_all = sub[:, tcol].copy()

    if rng.random() < 0.75 and n_rows >= 24:
        ord_s = np.argsort(y_all + 1e-12 * max(float(np.nanstd(y_all)), 1.0) * rng.standard_normal(n_rows), kind="mergesort")
        cp, qp = [], []
        for b_arr in np.array_split(ord_s, int(rng.integers(5, min(17, max(6, n_rows // 4))))):
            bp = rng.permutation(b_arr)
            cp.append(bp[: int(round(p_ctx * len(bp)))])
            qp.append(bp[int(round(p_ctx * len(bp))):])
        c_idx, q_idx = rng.permutation(np.concatenate(cp)), rng.permutation(np.concatenate(qp))
        if len(c_idx) >= n_ctx:
            order = np.concatenate([c_idx[:n_ctx], c_idx[n_ctx:], q_idx])
        else:
            order = np.concatenate([c_idx, q_idx[: n_ctx - len(c_idx)], q_idx[n_ctx - len(c_idx):]])
        X_all, y_all = X_all[order], y_all[order]

    Xc, Xq = X_all[:n_ctx], X_all[n_ctx:]
    yc, yq = y_all[:n_ctx].copy(), y_all[n_ctx:].copy()
    if (tmode < 2 or blend_real_teacher) and float(np.std(yc)) <= 1e-12:
        return None

    # Strict 0-10% TO-features synthetic column injection (k <= floor(inj_max_frac * F), inj_max_frac <= 0.10)
    # plus per-episode random feature removal (k_real <= p - 1 whenever p >= 2 and feature_drop_every_batch=True).
    p = Xc.shape[1]
    if f_fixed is None:
        if feature_drop_every_batch and p >= 2:
            max_drop = max(1, min(p - 1, int(round(drop_max_frac * p))))
            p_keep = p - int(rng.integers(1, max_drop + 1))
        else:
            p_keep = p
            if p >= 6 and rng.random() < 0.45:
                p_keep = int(rng.integers(max(3, int(round(0.60 * p))), p + 1))
        p_base = min(WIDE_GPU, p_keep)
        max_inj = int(math.floor((inj_max_frac / max(1e-9, 1.0 - inj_max_frac)) * p_base)) if inj_max_frac > 0.0 else 0
        if max_inj >= 1:
            k_try = int(rng.integers(1 if synth_inj_every_batch else 0, max_inj + 1))
        else:
            k_try = 0
        F = min(WIDE_GPU, p_keep + k_try)
        k_hi = int(math.floor(inj_max_frac * F))
        k = min(k_try, k_hi)
        if synth_inj_every_batch and k_hi >= 1 and k < 1:
            k = 1
        if F - k < 1:
            k = 0
            F = max(1, p_base)
        k_real = F - k
    else:
        F = f_fixed
        k_hi = int(math.floor(inj_max_frac * F))
        p_cap = (p - 1) if (feature_drop_every_batch and p >= 2) else p
        k_min_inj = 1 if (synth_inj_every_batch and k_hi >= 1) else 0
        k_lo = max(k_min_inj, F - p_cap)
        if k_lo > k_hi:
            return None
        k = int(rng.integers(k_lo, k_hi + 1))
        k_real = F - k
        if k_real < 1:
            return None

    Xc, Xq = _select_cols(Xc, yc if tmode < 2 else np.zeros(n_ctx), Xq, k_real, rng)
    if tmode == 2:
        t_pair = _synth_teacher_target_np(Xc, Xq, rng)
        if t_pair is None:
            return None
        if blend_real_teacher:
            yc_s = (yc - float(np.mean(yc))) / max(float(np.std(yc)), 1e-9)
            yq_s = (yq - float(np.mean(yc))) / max(float(np.std(yc)), 1e-9)
            alpha = float(rng.uniform(0.25, 0.65))
            yc = (1.0 - alpha) * yc_s + alpha * t_pair[0]
            yq = (1.0 - alpha) * yq_s + alpha * t_pair[1]
        else:
            yc, yq = t_pair
    elif rng.random() < 0.35:
        sy = float(np.std(yc))
        e = rng.standard_normal(n_rows) / np.sqrt(np.maximum(np.mean(rng.standard_normal((n_rows, 3)) ** 2, axis=1), 1e-3)) / 1.732
        jit = sy * float(rng.uniform(0.015, 0.03)) * np.clip(e, -1.345, 1.345)
        yc, yq = yc + jit[:n_ctx], yq + jit[n_ctx:]

    if k > 0:
        Ic, Iq = _inject_synth_cols_np(Xc, Xq, k, rng)
        Xc, Xq = np.hstack([Xc, Ic]), np.hstack([Xq, Iq])
    perm_f = rng.permutation(Xc.shape[1])
    Xc, Xq = Xc[:, perm_f], Xq[:, perm_f]

    u_sp = float(rng.random())
    space = 0
    y_ep = np.concatenate([yc, yq])
    if u_sp < 0.40:
        oof, pq, used, _ = _robust_ridge_np(Xc, yc, Xq, seed=int(rng.integers(1, 1 << 30)))
        if used:
            y_ep = np.concatenate([yc - oof, yq - pq])
            space = 1
    elif u_sp < 0.60:
        yj_pair = _yeojohnson_pair_np(yc, yq)
        if yj_pair is not None and float(np.std(yj_pair[0])) > 1e-8:
            y_ep = np.concatenate([yj_pair[0], yj_pair[1]])
            space = 2
    X_ep = np.vstack([Xc, Xq])
    if X_ep.shape[1] != F or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx])) <= 1e-12:
        return None
    return X_ep, y_ep, n_ctx, (int(t_idx), tmode, k, F, n_rows, space, int(max(0, p - k_real)), int(p))


def _sample_full_synth_special(
    B: int,
    N: int,
    Fn: int,
    n_ctx: int,
    rng: np.random.Generator,
    feature_drop_every_batch: bool = True,
    drop_max_frac: float = 0.35,
    synth_inj_every_batch: bool = True,
    inj_max_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    inj_max_frac = max(0.0, min(0.10, float(inj_max_frac)))
    drop_max_frac = max(0.05, min(0.50, float(drop_max_frac)))
    is_panel = bool(rng.random() < 0.55)
    xs = np.zeros((B, N, Fn), dtype=np.float64)
    ys = np.zeros((B, N), dtype=np.float64)
    Fn_gen = Fn + (max(1, int(round(drop_max_frac * Fn))) if feature_drop_every_batch else 0)
    for b in range(B):
        if is_panel:
            n_ent = int(rng.integers(6, max(8, min(48, N // 8))))
            ent = rng.integers(0, n_ent, size=N)
            t_step = rng.uniform(0.0, 2000.0, size=N)
            base_epoch_ms = float(rng.choice([1.7040672e12, 20725.0, 15000.0]))
            ts_col = base_epoch_ms + t_step * (60_000.0 if base_epoch_ms > 1e9 else 0.01)
            covs = rng.standard_normal((N, max(1, Fn_gen - 2)))
            ent_base = rng.standard_normal(n_ent) * 2.5
            ent_freq = rng.uniform(0.003, 0.02, size=n_ent)
            ent_ph = rng.uniform(0.0, 2.0 * math.pi, size=n_ent)
            ent_slope = rng.standard_normal(n_ent) * 0.002
            w_cov = rng.standard_normal(covs.shape[1]) * 0.35
            y_clean = ent_base[ent] + 1.8 * np.sin(ent_freq[ent] * t_step + ent_ph[ent]) + ent_slope[ent] * t_step + covs @ w_cov
            y_b = y_clean + float(rng.uniform(0.015, 0.12)) * max(float(np.std(y_clean)), 1e-3) * rng.standard_normal(N)
            X_raw = np.column_stack([ts_col, ent.astype(np.float64), covs])[:, :Fn_gen]
            perm = rng.permutation(N)
            X_raw, y_b = X_raw[perm], y_b[perm]
        else:
            X_raw = rng.standard_normal((N, Fn_gen))
            w1 = rng.standard_normal(Fn_gen) / math.sqrt(max(Fn_gen, 1))
            w2 = rng.standard_normal(Fn_gen) / math.sqrt(max(Fn_gen, 1))
            y_bulk = 15.0 + 8.0 * np.sin(X_raw @ w1) + 5.0 * ((X_raw @ w2) ** 2 - 1.0) + 0.6 * rng.standard_normal(N)
            atom_val = float(rng.choice([-99.99, -999.0, 0.0, 9999.0]))
            gate_score = 0.75 * X_raw[:, 0] - 0.55 * X_raw[:, min(1, Fn_gen - 1)] + 0.25 * rng.standard_normal(N)
            is_atom = gate_score < float(np.percentile(gate_score, float(rng.uniform(8.0, 22.0))))
            if rng.random() < 0.65:
                X_raw[is_atom, 0] = -99.99
            perm = rng.permutation(N)
            X_raw, y_b = X_raw[perm], np.where(is_atom, atom_val, y_bulk)[perm]

        k_hi_sp = int(math.floor(inj_max_frac * Fn))
        if k_hi_sp >= 1 and (synth_inj_every_batch or rng.random() < 0.50):
            k_sp = int(rng.integers(1 if synth_inj_every_batch else 0, k_hi_sp + 1))
        else:
            k_sp = 0
        k_keep = max(1, Fn - k_sp)
        keep_idx = rng.choice(X_raw.shape[1], size=k_keep, replace=False) if X_raw.shape[1] > k_keep else np.arange(X_raw.shape[1])
        X_sub = X_raw[:, keep_idx]
        if k_sp >= 1:
            Ic, Iq = _inject_synth_cols_np(X_sub[:n_ctx], X_sub[n_ctx:], k_sp, rng)
            X_full = np.vstack([np.hstack([X_sub[:n_ctx], Ic]), np.hstack([X_sub[n_ctx:], Iq])])
        else:
            X_full = X_sub
        if X_full.shape[1] < Fn:
            X_full = np.hstack([X_full, rng.standard_normal((N, Fn - X_full.shape[1]))])
        X_full = X_full[:, :Fn]
        xs[b] = X_full[:, rng.permutation(Fn)]
        ys[b] = y_b
    return xs, ys


def _spawned_worker_loop(worker_id: int, run_nonce: bytes, mmap_dir_s: str, expect_sha: str,
                         tokens_cap: int, row_tokens_cap: int, bmax: int, cpu_hard: float,
                         random_split_every_batch: bool, feature_drop_every_batch: bool, drop_max_frac: float,
                         synth_inj_every_batch: bool, inj_max_frac: float,
                         out_q, stop_ev) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        os.nice(12)
    except Exception:
        pass
    inj_max_frac = max(0.0, min(0.10, float(inj_max_frac)))
    drop_max_frac = max(0.05, min(0.50, float(drop_max_frac)))
    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha)
    gov = HostCpuGovernor(hard_pct=cpu_hard, check_every_s=2.5)
    seq = 0
    while not stop_ev.is_set():
        gov.pace()
        seq += 1
        b_seed = hashlib.blake2b(run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint32))
        if float(rng.random()) < 0.58:
            got = bidirectional_real_episode(
                bank, bank.sample_table_idx(rng), rng,
                random_split_every_batch=random_split_every_batch,
                feature_drop_every_batch=feature_drop_every_batch,
                drop_max_frac=drop_max_frac,
                synth_inj_every_batch=synth_inj_every_batch,
                inj_max_frac=inj_max_frac,
            )
            if got is None:
                continue
            X_ep, y_ep, n_ctx, info = got
            item = (X_ep[None], y_ep[None], int(n_ctx), [info], [dataset_fingerprint_128(X_ep, y_ep, n_ctx)])
        else:
            t0 = bank.sample_table_idx(rng)
            p0_raw = int(bank.cols_arr[t0])
            ok_n = [n for n in BATCH_N if n + 8 <= int(bank.rows_arr[t0])]
            if not ok_n or (feature_drop_every_batch and p0_raw < 3):
                got = bidirectional_real_episode(
                    bank, t0, rng,
                    random_split_every_batch=random_split_every_batch,
                    feature_drop_every_batch=feature_drop_every_batch,
                    drop_max_frac=drop_max_frac,
                    synth_inj_every_batch=synth_inj_every_batch,
                    inj_max_frac=inj_max_frac,
                )
                if got is None:
                    continue
                item = (got[0][None], got[1][None], int(got[2]), [got[3]], [dataset_fingerprint_128(got[0], got[1], got[2])])
            else:
                pn = np.array([BATCH_N_P[BATCH_N.index(n)] for n in ok_n])
                N_t = int(rng.choice(ok_n, p=pn / pn.sum()))
                if feature_drop_every_batch and p0_raw >= 3:
                    max_drop0 = max(1, min(p0_raw - 2, int(round(drop_max_frac * p0_raw))))
                    p0_keep = p0_raw - int(rng.integers(1, max_drop0 + 1))
                else:
                    p0_keep = p0_raw
                p0 = min(p0_keep, WIDE_GPU)
                max_inj0 = int(math.floor((inj_max_frac / max(1e-9, 1.0 - inj_max_frac)) * p0)) if inj_max_frac > 0.0 else 0
                k0 = int(rng.integers(1 if synth_inj_every_batch else 0, max_inj0 + 1)) if max_inj0 >= 1 else 0
                F_t = max(2, min(WIDE_GPU, p0 + k0))
                u = float(rng.random())
                p_ctx = float(rng.uniform(0.25, 0.56) if u < 0.25 else (rng.uniform(0.56, 0.84) if u < 0.75 else rng.uniform(0.84, 0.94)))
                B_t = max(1, min(bmax, tokens_cap // max(N_t * (F_t + 4), 1), row_tokens_cap // max(N_t, 1)))
                k_hi_t = int(math.floor(inj_max_frac * F_t))
                min_p_req = (F_t - k_hi_t + 1) if feature_drop_every_batch else max(1, F_t - k_hi_t)
                elig = ((bank.rows_arr >= N_t + 8) & (bank.cols_arr >= min_p_req)).astype(np.float64)
                xs, ys, infos, n_ctx_b = [], [], [], None
                for b in range(B_t):
                    t_idx = t0 if b == 0 else bank.sample_eligible(rng, elig)
                    if t_idx is None:
                        break
                    got = bidirectional_real_episode(
                        bank, t_idx, rng, n_rows_fixed=N_t, f_fixed=F_t, p_ctx_fixed=p_ctx,
                        random_split_every_batch=random_split_every_batch,
                        feature_drop_every_batch=feature_drop_every_batch,
                        drop_max_frac=drop_max_frac,
                        synth_inj_every_batch=synth_inj_every_batch,
                        inj_max_frac=inj_max_frac,
                    )
                    if got is None or got[0].shape != (N_t, F_t) or (n_ctx_b is not None and got[2] != n_ctx_b):
                        continue
                    n_ctx_b = got[2]
                    xs.append(got[0]); ys.append(got[1]); infos.append(got[3])
                if not xs:
                    continue
                item = (np.stack(xs), np.stack(ys), int(n_ctx_b), infos,
                        [dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs, ys)])
        while not stop_ev.is_set():
            try:
                out_q.put(item, timeout=1.0)
                break
            except queue.Full:
                gov.pace()
                continue


class Ledger128:
    """128-bit pre-forward deduplicating ledger compatible with `jope.h450.epoch/v1`."""

    def __init__(self, bank: MmapF64Bank):
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
        self.seen_fp_128: set[bytes] = set()
        self.dup_filtered_pre_forward = 0
        self.real_eps_p_ge_2 = 0
        self.real_eps_dropped = 0
        self.dropped_real_cols = 0

    def filter_unseen_indices(self, fps: list[bytes]) -> list[int]:
        keep = []
        for idx, fp in enumerate(fps):
            if fp in self.seen_fp_128:
                self.dup_filtered_pre_forward += 1
            else:
                self.seen_fp_128.add(fp)
                keep.append(idx)
        return keep

    def add_real(self, infos) -> int:
        cells = 0
        for row in infos:
            t, tm, k, F, n, sp = row[:6]
            n_drop = int(row[6]) if len(row) > 6 else 0
            p_src = int(row[7]) if len(row) > 7 else int(F - k)
            c = int(n * (F + 1))
            self.table_eps[t] += 1
            self.table_cells[t] += c
            self.table_rows[t] += int(n)
            self.tmode[tm] += 1
            self.space[sp] += 1
            self.inj_cols += int(k)
            self.real_cols += int(F - k)
            if p_src >= 2:
                self.real_eps_p_ge_2 += 1
                if n_drop >= 1:
                    self.real_eps_dropped += 1
            self.dropped_real_cols += n_drop
            cells += c
        return cells

    def add_synth(self, B: int, N: int, F: int) -> int:
        self.synth_datasets += int(B)
        self.synth_cells += int(B * N * (F + 1))
        return int(B * N * (F + 1))

    def state(self) -> dict:
        raw_fp = b"".join(self.seen_fp_128)
        return {
            "table_eps": self.table_eps.tolist(), "table_cells": self.table_cells.tolist(),
            "table_rows": self.table_rows.tolist(), "tmode": self.tmode.tolist(), "space": self.space.tolist(),
            "inj_cols": self.inj_cols, "real_cols": self.real_cols, "synth_datasets": self.synth_datasets,
            "synth_cells": self.synth_cells, "dup_filtered_pre_forward": self.dup_filtered_pre_forward,
            "real_eps_p_ge_2": self.real_eps_p_ge_2, "real_eps_dropped": self.real_eps_dropped,
            "dropped_real_cols": self.dropped_real_cols,
            "seen_fp_128_bytes": np.frombuffer(raw_fp, dtype=np.uint8).copy(),
        }

    def load(self, s: dict) -> None:
        self.table_eps = np.array(s["table_eps"], dtype=np.int64)
        self.table_cells = np.array(s["table_cells"], dtype=np.int64)
        self.table_rows = np.array(s["table_rows"], dtype=np.int64)
        self.tmode = np.array(s["tmode"], dtype=np.int64)
        self.space = np.array(s["space"], dtype=np.int64)
        for k in ("inj_cols", "real_cols", "synth_datasets", "synth_cells", "dup_filtered_pre_forward",
                  "real_eps_p_ge_2", "real_eps_dropped", "dropped_real_cols"):
            setattr(self, k, int(s.get(k, 0)))
        raw = np.asarray(s.get("seen_fp_128_bytes", np.empty(0, np.uint8)), dtype=np.uint8).tobytes()
        self.seen_fp_128 = {raw[i : i + 16] for i in range(0, len(raw), 16)}

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
        return {
            "per_cohort": per, "real_episodes": real_eps, "synthetic_datasets": self.synth_datasets,
            "real_cell_tokens": int(self.table_cells.sum()), "synthetic_cell_tokens": self.synth_cells,
            "dev91_share_of_real_episodes": (d91 / real_eps) if real_eps else None,
            "target_modes": dict(zip(("own", "other_column", "teacher"), self.tmode.tolist())),
            "target_spaces": dict(zip(("raw", "ridge_residual", "yeo_johnson"), self.space.tolist())),
            "injected_column_share": self.inj_cols / max(1, self.inj_cols + self.real_cols),
            "feature_drop_share_p_ge_2": (self.real_eps_dropped / self.real_eps_p_ge_2) if self.real_eps_p_ge_2 else None,
            "dropped_real_columns_total": self.dropped_real_cols,
            "sealed_cohorts_in_bank": 0,
            "unique_datasets_seen": len(self.seen_fp_128),
            "datasets_seen": real_eps + self.synth_datasets,
            "duplicate_datasets_fed_to_model": 0,
            "duplicates_filtered_pre_forward": self.dup_filtered_pre_forward,
            "duplicate_real_datasets": 0,
            "duplicate_synthetic_datasets": 0,
            "distinct_source_tables_seen": int((self.table_eps > 0).sum()),
            "source_tables_in_bank": int(b.n_tables),
        }


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), default="L")
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--mmap-bank", required=True)
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--save-every-datasets", type=int, default=25_000)
    ap.add_argument("--last-every-s", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=4.0e-5)
    ap.add_argument("--synth-prob", type=float, default=0.25)
    ap.add_argument("--tokens", type=int, default=180_000)
    ap.add_argument("--row-tokens", type=int, default=4_096)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpu-hard-pct", type=float, default=82.0)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.80)
    ap.add_argument("--random-split-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.35)
    ap.add_argument("--synth-inj-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--synth-inj-max-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=450_441)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if not (0.0 <= a.synth_inj_max_frac <= 0.10):
        raise SystemExit(f"--synth-inj-max-frac {a.synth_inj_max_frac} exceeds strict owner cap 0.10")
    if not (0.05 <= a.feature_drop_max_frac <= 0.50):
        raise SystemExit(f"--feature-drop-max-frac {a.feature_drop_max_frac} outside [0.05, 0.50]")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_p, last_p = out_dir / "ckpt_h450_final.pt", out_dir / "ckpt_last.pt"
    if final_p.exists():
        raise SystemExit(f"{final_p} exists (create-once)")
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    if sha256_file(Path(a.init)) != a.init_sha256:
        raise SystemExit("init checkpoint sha256 mismatch")

    t_load0 = time.time()
    bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, verify_data_sha=True)
    bank_load_s = round(time.time() - t_load0, 3)
    ledger = Ledger128(bank)
    run_nonce = hashlib.blake2b(secrets.token_bytes(32) + int(a.seed).to_bytes(8, "little"), digest_size=32).digest()

    ctx = mp.get_context("spawn")
    ep_q = ctx.Queue(maxsize=max(12, 6 * a.workers))
    stop_ev = ctx.Event()
    procs = [
        ctx.Process(
            target=_spawned_worker_loop,
            args=(
                w, run_nonce, str(a.mmap_bank), a.bank_sha256, a.tokens, a.row_tokens, a.bmax, a.cpu_hard_pct,
                bool(a.random_split_every_batch), bool(a.feature_drop_every_batch), float(a.feature_drop_max_frac),
                bool(a.synth_inj_every_batch), float(a.synth_inj_max_frac),
                ep_q, stop_ev,
            ),
            daemon=True,
        )
        for w in range(a.workers)
    ]
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
        if a.cuda_vram_frac > 0.0:
            torch.cuda.set_per_process_memory_fraction(min(0.84, a.cuda_vram_frac + 0.03), dev.index or 0)

    model = ICLH450MPS(a.size).to(dev)
    ema = ICLH450MPS(a.size).to(dev).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)
    state = {"step": 0, "micro_batches": 0, "accum_in_group": 0, "real_batches": 0, "synth_batches": 0,
             "datasets": 0, "cell_tokens": 0, "oom_skips": 0, "nonfinite_skips": 0, "train_s": 0.0,
             "epochs_saved": 0, "resumes": 0}
    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        if ck.get("bank_sha256") and ck["bank_sha256"] != bank.sha256:
            raise SystemExit(f"resume bank_sha256 mismatch: {ck.get('bank_sha256')} != {bank.sha256}")
        if ck.get("init_sha256") and ck["init_sha256"] != a.init_sha256:
            raise SystemExit(f"resume init_sha256 mismatch: {ck.get('init_sha256')} != {a.init_sha256}")
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        state["resumes"] += 1
        ledger.load(ck["ledger"])
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        w_init = ck.get("ema") or ck["model"]
        model.load_state_dict(w_init, strict=True)
        ema.load_state_dict(w_init, strict=True)

    params_cnt = int(n_params(model))
    prior_dev = torch.device("cpu") if dev.type == "mps" else dev
    prior = Prior(prior_dev, a.seed + 17 + 1000 * state["resumes"])
    py_rng = np.random.default_rng(np.frombuffer(hashlib.blake2b(run_nonce + b"main", digest_size=32).digest(), dtype=np.uint32))
    gov = HostCpuGovernor(hard_pct=a.cpu_hard_pct, check_every_s=2.5)

    def save_last() -> None:
        _atomic_save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                      "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                      "opt": opt.state_dict(), "state": state, "ledger": ledger.state(),
                      "size": a.size, "bank_sha256": bank.sha256, "init_sha256": a.init_sha256, "argv": sys.argv}, last_p)

    def save_epoch(k: int) -> None:
        ep_p = out_dir / f"ckpt_epoch_{k:02d}.pt"
        if ep_p.exists():
            return
        ema_cpu = { kk: vv.cpu() for kk, vv in ema.state_dict().items() }
        _atomic_save({"ema": ema_cpu, "model": ema_cpu, "state": dict(state), "size": a.size}, ep_p)
        os.chmod(ep_p, 0o444)
        doc = {
            "schema": "jope.h450.epoch/v1", "epoch": k, "train_hours": state["train_s"] / 3600.0, "step": state["step"],
            "ckpt": str(ep_p), "ckpt_sha256": sha256_file(ep_p), "init_sha256": a.init_sha256,
            "bank_sha256": bank.sha256, "quarantined_in_corpus": True, "canonical_promotion_forbidden": True,
            "exposure": ledger.summary(),
        }
        jp = out_dir / f"epoch_{k:02d}.json"
        jp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.chmod(jp, 0o444)
        state["epochs_saved"] = k

    budget_s = max(15.0, a.hours * 3600.0)
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)
    log_f.write(json.dumps({"start": time.strftime("%FT%TZ", time.gmtime()), "host": platform.node(), "device": str(dev),
                            "resume": bool(a.resume), "params": params_cnt, "bank_tables": bank.n_tables,
                            "bank_load_s": bank_load_s, "train_s": state["train_s"], "step": state["step"]}) + "\n")

    grad_accum = max(1, int(a.grad_accum))
    opt.zero_grad(set_to_none=True)
    t_seg = time.time()
    s_seg0 = state["train_s"]
    t_log = t_last = time.time()
    run_loss, run_n, run_ds = 0.0, 0, 0

    try:
        while not stop_flag["stop"]:
            host_cpu_pct = gov.pace()
            state["train_s"] = s_seg0 + (time.time() - t_seg)
            if state["train_s"] >= budget_s or (a.max_steps > 0 and state["step"] >= a.max_steps):
                break
            k_ep = int(state["datasets"] // max(1, a.save_every_datasets))
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
                N = int(py_rng.choice([256, 512, 768, 1024], p=[0.25, 0.35, 0.25, 0.15]))
                B = max(1, min(a.bmax, a.tokens // max(N * (Fn + 4), 1), a.row_tokens // max(N, 1)))
                n_ctx = max(16, min(N - 4, int(round(float(py_rng.uniform(0.25, 0.93)) * N))))
                if float(py_rng.random()) < 0.60:
                    Fn_gen = min(WIDE_GPU, Fn + (max(1, int(round(a.feature_drop_max_frac * Fn))) if a.feature_drop_every_batch else 0))
                    xt_c, yt_c, _, _ = INJ.sample_synth(prior, r_py, B, N, Fn_gen, 0.55, PROBS_V3S)
                    xa_g, ya = xt_c.detach().cpu().double().numpy(), yt_c.detach().cpu().double().numpy()
                    if a.feature_drop_every_batch and Fn_gen >= 3:
                        max_ds = max(1, min(Fn_gen - 2, int(round(a.feature_drop_max_frac * Fn_gen))))
                        n_ds = int(py_rng.integers(1, max_ds + 1))
                        keep_s = py_rng.choice(Fn_gen, size=Fn_gen - n_ds, replace=False)
                        xa = xa_g[:, :, keep_s]
                    else:
                        xa = xa_g
                else:
                    xa, ya = _sample_full_synth_special(
                        B, N, Fn, n_ctx, py_rng,
                        feature_drop_every_batch=bool(a.feature_drop_every_batch),
                        drop_max_frac=float(a.feature_drop_max_frac),
                        synth_inj_every_batch=bool(a.synth_inj_every_batch),
                        inj_max_frac=float(a.synth_inj_max_frac),
                    )
                if float(py_rng.random()) < 0.40:
                    seed_r = int(py_rng.integers(1, 1 << 30))
                    for b_i in range(xa.shape[0]):
                        oof_b, pq_b, used_b, _ = _robust_ridge_np(xa[b_i, :n_ctx], ya[b_i, :n_ctx], xa[b_i, n_ctx:], seed=seed_r + 31 * b_i)
                        if used_b:
                            ya[b_i, :n_ctx] -= oof_b
                            ya[b_i, n_ctx:] -= pq_b
                infos = None
                fps = [dataset_fingerprint_128(xa[b_], ya[b_], n_ctx) for b_ in range(xa.shape[0])]
            else:
                xa, ya, n_ctx, infos, fps = ep_q.get(timeout=180.0)

            # Pre-forward 128-bit deduplication: NEVER feed a duplicate dataset to the model!
            keep_idx = ledger.filter_unseen_indices(fps)
            if not keep_idx:
                continue
            if len(keep_idx) < xa.shape[0]:
                xa = xa[keep_idx]
                ya = ya[keep_idx]
                if infos is not None:
                    infos = [infos[i] for i in keep_idx]

            # On MPS, keep raw float64 on CPU so normalise_f64 & loss avoid float64 MPS conversion errors!
            in_dev = torch.device("cpu") if dev.type == "mps" else dev
            xt = torch.from_numpy(xa).to(device=in_dev, dtype=torch.float64)
            yt = torch.from_numpy(ya).to(device=in_dev, dtype=torch.float64)
            Fn_cur = xt.shape[2]
            slots = torch.randperm(FID_SLOTS, device=dev)[:Fn_cur]
            try:
                logits, ym, ys_ = model(xt, yt[:, :n_ctx], n_ctx, slots=slots, amp=(dev.type in ("cuda", "mps")))
                loss = model.loss(logits, yt[:, n_ctx:], ym, ys_).mean()
                if not torch.isfinite(loss):
                    state["nonfinite_skips"] += 1
                    continue
                (loss / float(grad_accum)).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    state["oom_skips"] += 1
                    opt.zero_grad(set_to_none=True)
                    state["accum_in_group"] = 0
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                    continue
                raise

            if infos is None:
                state["cell_tokens"] += ledger.add_synth(int(xt.shape[0]), int(xt.shape[1]), Fn_cur)
                state["synth_batches"] += 1
            else:
                state["cell_tokens"] += ledger.add_real(infos)
                state["real_batches"] += 1
            state["micro_batches"] += 1
            state["accum_in_group"] += 1
            b_cnt = int(xt.shape[0])
            state["datasets"] += b_cnt
            run_loss += float(loss.detach().cpu())
            run_n += 1
            run_ds += b_cnt

            if state["accum_in_group"] >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                state["accum_in_group"] = 0
                with torch.no_grad():
                    decay = min(0.995, (1.0 + state["step"]) / (10.0 + state["step"]))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - decay)
                if dev.type == "cuda" and a.cuda_vram_frac > 0.0 and state["step"] == 3:
                    try:
                        free_b, tot_b = torch.cuda.mem_get_info(dev)
                        used_b = tot_b - free_b
                        tgt_b = int(a.cuda_vram_frac * tot_b)
                        if used_b < tgt_b - (64 << 20):
                            _pool_warm = torch.empty(tgt_b - used_b, dtype=torch.uint8, device=dev)
                            del _pool_warm
                    except Exception:
                        pass
                if state["step"] == 10:
                    save_last()
                    t_last = time.time()

            now = time.time()
            if now - t_log >= 30.0 and run_n > 0:
                s = ledger.summary()
                dt = max(1e-3, now - t_log)
                mps_gb = round(torch.mps.current_allocated_memory() / 2**30, 2) if (dev.type == "mps" and hasattr(torch.mps, "current_allocated_memory")) else 0.0
                cuda_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else mps_gb
                log_f.write(json.dumps({
                    "step": state["step"], "train_h": round(state["train_s"] / 3600.0, 4), "loss": round(run_loss / run_n, 5),
                    "lr": round(cur_lr, 9), "datasets": state["datasets"],
                    "ds_per_s": round(run_ds / dt, 3), "mb_per_s": round(run_n / dt, 3),
                    "real_batches": state["real_batches"], "synth_batches": state["synth_batches"],
                    "real_eps": s["real_episodes"], "d91_share": s["dev91_share_of_real_episodes"],
                    "unique_datasets": s["unique_datasets_seen"], "datasets_seen": s["datasets_seen"],
                    "dup_fed": s["duplicate_datasets_fed_to_model"], "dup_filtered": s["duplicates_filtered_pre_forward"],
                    "src_tables_seen": s["distinct_source_tables_seen"], "inj_share": round(s["injected_column_share"], 4),
                    "drop_share_p2": round(s["feature_drop_share_p_ge_2"], 4) if s["feature_drop_share_p_ge_2"] is not None else None,
                    "host_cpu_pct": round(host_cpu_pct, 1), "oom": state["oom_skips"], "mem_gb": cuda_gb,
                }) + "\n")
                run_loss, run_n, run_ds, t_log = 0.0, 0, 0, now
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
                                "datasets": state["datasets"], "train_h": round(state["train_s"] / 3600.0, 4)}) + "\n")
        log_f.close()
        return
    k_ep = max(1, int(state["datasets"] // max(1, a.save_every_datasets)))
    if k_ep > state["epochs_saved"]:
        save_epoch(k_ep)
    ema_cpu = {kk: vv.cpu() for kk, vv in ema.state_dict().items()}
    mod_cpu = {kk: vv.cpu() for kk, vv in model.state_dict().items()}
    _atomic_save({"model": mod_cpu, "ema": ema_cpu, "state": state, "size": a.size}, final_p)
    os.chmod(final_p, 0o444)
    manifest = {
        "schema": "jope.h450.ckpt_manifest/v3", "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True, "size": a.size, "params": params_cnt, "device": str(dev),
        "host": platform.node(), "seed": a.seed, "init_ckpt": str(a.init), "init_sha256": a.init_sha256,
        "bank": str(a.mmap_bank), "bank_sha256": bank.sha256, "bank_tables": bank.n_tables,
        "ckpt_final_path": str(final_p), "ckpt_final_sha256": sha256_file(final_p),
        "state": state, "exposure": ledger.summary(),
    }
    man_p = out_dir / "manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(man_p, 0o444)
    log_f.close()
    print(json.dumps({k: manifest[k] for k in ("ckpt_final_sha256", "state")}, sort_keys=True))


if __name__ == "__main__":
    main()
