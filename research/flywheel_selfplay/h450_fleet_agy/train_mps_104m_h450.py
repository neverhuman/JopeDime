#!/usr/bin/env python3
"""104M (`ICL_L`) Maximum-Entropy Uniform-Dataset MPS/CUDA Trainer (`train_mps_104m_h450.py` — `fleet-r9`).

Owner Directive (`Jepson 2026-11-13 05:39 MST` / `12:41Z`):
  0. Keep-Latest-Only Checkpoint Policy (`--keep-latest-only 1`):
     - Never lose the latest checkpoint: atomic `.tmp -> os.replace` updates to `ckpt_last.pt` (full resume state)
       and `ckpt_latest.pt` (latest `ema_swa` inference weights, single rolling file per run, zero per-epoch `.pt`
       accumulation, zero `unlink`/`rm` calls).
     - Preserves all numbers and exposure metrics (`log.jsonl` + per-milestone `epoch_NN.json`).
  1. Wider Zero-Leakage Column Drop & Synthetic Injection Ranges:
     - Random real-column drop: `[5%, 65%]` (`--feature-drop-min-frac 0.05 --feature-drop-max-frac 0.65`) on `100%` of `p >= 2`.
     - Zero-leakage synthetic column injection: `[5%, 30%]` of episode width `F` (`--synth-inj-min-frac 0.05 --synth-inj-max-frac 0.30`,
       14 structural + distractor families fitted strictly on context features `X[:n_ctx]` only).
  2. Varied Episode Row Subsampling Across Full Corpus (`[32 .. 2048]` rows, NO permanent downsample to 300 rows):
     - Full table rows (`r_raw`) remain in play on every episode; fresh uniform / quantile-stratified / projection-tilted
       row subsampling draws varied episode sizes across `ROW_BINS = (32 .. 2048)` with `p_ctx in [0.18, 0.95]`.
  3. Random Valid Target Column Choice:
     - Strided-probe validator caches all continuous non-ID columns per table; episodes sample `45%` random valid column
       (`X_{-j} -> X_j`), `40%` canonical target, and `15%` nonlinear GP/oblivious-tree teacher.
  4. Per-Episode Feature & Slot Permutations, Random Per-Feature Scaling/Affine, and Small Context-Scaled Noise:
     - Full column permutation + independent per-episode `FID_SLOTS` permutation `[B, 1, F]`;
     - Zero-leakage raw sign flip `{-1, +1}` + log-uniform scaling `exp(U(-3, 3))` + shift + `[0.2%, 5%]` context-scaled
       feature noise, plus post-`normalise_f64` mild per-feature affine jitter (`[0.80, 1.25]` scale, `[-0.18, 0.18]` shift).
  5. Uniform-by-Dataset Mixing Across All 3,525 S3 Sources (`--dataset-mix uniform`):
     - Samples `t_idx ~ Uniform({0, ..., 3524})` (`1 / 3,525` per dataset, independent of row count) and batches episodes
       via a per-worker `(N_b, F_b)` bin accumulator so small-row datasets are never rejected against batch row thresholds.
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
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/pub-r5/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    str(Path.home() / "jope-data/h450-fleet-agy/ref_code"),
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

import icl_inject as INJ  # noqa: E402
from icl_model import CLIP_X, FID_SLOTS  # noqa: E402
from icl_model_mps_h450 import ICLH450MPS, n_params  # noqa: E402
from icl_prior_v3s import PROBS_V3S, Prior, draw_features  # noqa: E402
from mmap_bank_h450 import MmapF64Bank, sha256_file  # noqa: E402


WIDE_GPU = 64
DEFAULT_MAX_EPISODE_ROWS = 2048

# Full-spectrum episode row & width ladders (full corpus stays in play; NO permanent 300-row downsample)
ROW_BINS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1408, 1728, 2048)
FALLBACK_ROW_BINS = (32, 64, 128, 256, 512, 1024)
WIDTH_BINS = (1, 2, 4, 8, 12, 20, 32, 48, 64)

SYNTH_N = (48, 96, 192, 320, 512, 768, 1152, 1600, 2048)
SYNTH_N_P = (0.06, 0.08, 0.14, 0.18, 0.18, 0.14, 0.10, 0.07, 0.05)


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


def _discover_valid_target_cols(tab_mmap: np.ndarray, min_rows: int = 32) -> tuple[int, ...]:
    """Identifies all valid continuous non-ID columns in `tab_mmap` via strided finite probes."""
    r_raw, c_tot = tab_mmap.shape
    valid: list[int] = []
    for j in range(c_tot):
        col_j = tab_mmap[:, j]
        fin_j = np.flatnonzero(np.isfinite(col_j))
        if fin_j.size < min_rows:
            continue
        if j == c_tot - 1:
            probe_y = np.asarray(col_j[fin_j[:: max(1, fin_j.size // 512)][:512]], dtype=np.float64)
            if float(np.std(probe_y)) > 1e-12:
                valid.append(j)
            continue
        stride = max(1, fin_j.size // 512)
        probe = np.asarray(col_j[fin_j[::stride][:512]], dtype=np.float64)
        if probe.size < 24:
            continue
        sd = float(np.std(probe))
        if not math.isfinite(sd) or sd <= 1e-9:
            continue
        u_vals, u_cnts = np.unique(probe, return_counts=True)
        if u_vals.size < 10 or (int(u_cnts.max()) / float(probe.size)) > 0.72:
            continue
        diffs = np.diff(probe)
        if diffs.size >= 16 and abs(float(diffs[0])) > 1e-12 and bool(np.all(np.abs(diffs - diffs[0]) <= 1e-9 * max(1.0, abs(float(diffs[0]))))):
            continue
        valid.append(j)
    if (c_tot - 1) not in valid:
        fin_last = np.flatnonzero(np.isfinite(tab_mmap[:, c_tot - 1]))
        if fin_last.size >= 28:
            valid.append(c_tot - 1)
    return tuple(valid)


def _robust_ridge_np(X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, seed: int = 450, folds: int = 5):
    """Pure NumPy SVD-GCV 5-fold OOF robust ridge (context-fitted only, zero query leakage)."""
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
    """Exact pure-NumPy Yeo-Johnson MLE fit on `yc` only, applied to `(yc, yq)` (zero query leakage)."""
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
                load_pct = min(100.0, 100.0 * (os.getloadavg()[0] / float(self.n_cpus)))
                if load_pct > self.hard_pct - 8.0:
                    r = subprocess.run(["ps", "-A", "-o", "%cpu"], capture_output=True, text=True, timeout=1.5)
                    if r.returncode == 0:
                        ps_pct = min(100.0, sum(float(x) for x in r.stdout.splitlines()[1:] if x.strip()) / float(self.n_cpus))
                        self._last_pct = 0.5 * (load_pct + ps_pct)
                    else:
                        self._last_pct = load_pct
                else:
                    self._last_pct = load_pct
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
    """Fast vectorized row-order-invariant 128-bit `blake2b` fingerprint over canonicalized `float64` row bytes."""
    Xf = np.asarray(X, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64).reshape(-1)
    n_tot, f_dim = Xf.shape
    Z = np.empty((n_tot, f_dim + 1), dtype=np.float64)
    Z[:, :f_dim] = Xf
    Z[:, f_dim] = yf
    Z[Z == 0.0] = 0.0
    Zu = Z.view(np.uint64)
    nan_m = np.isnan(Z)
    if np.any(nan_m):
        Zu[nan_m] = np.uint64(0x7FF8000000000001)
    with np.errstate(over="ignore"):
        mults = (np.arange(1, f_dim + 2, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) | np.uint64(1)
        mixed = Zu ^ (Zu >> np.uint64(30))
        mixed = mixed * np.uint64(0xBF58476D1CE4E5B9)
        mixed = mixed ^ (mixed >> np.uint64(27))
        rh_all = np.bitwise_xor.reduce(mixed * mults[None, :], axis=1)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([n_tot, f_dim, int(n_ctx)], dtype=np.int64).tobytes())
    for part, rh in ((Zu[:n_ctx], rh_all[:n_ctx]), (Zu[n_ctx:], rh_all[n_ctx:])):
        if part.shape[0] > 1:
            ord_i = np.argsort(rh, kind="mergesort")
            part = part[ord_i]
            rh_s = rh[ord_i]
            ties = rh_s[1:] == rh_s[:-1]
            if np.any(ties) and np.any(part[1:][ties] != part[:-1][ties]):
                part = part[np.lexsort(part.T[::-1])]
        h.update(np.ascontiguousarray(part).tobytes())
    return h.digest()


def _std_ctx_f64(Xc: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Context-only robust standardisation in float64 (strictly zero query/target leakage)."""
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
    """14-family synthetic column bank fitted strictly on context features `Xc` (zero target or query leakage)."""
    p = Xc.shape[1]
    if p == 0 or k_inj <= 0:
        return np.zeros((Xc.shape[0], 0)), np.zeros((Xq.shape[0], 0))
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z_all = np.vstack([Zc, Zq])
    n_tot, n_c = Z_all.shape[0], Xc.shape[0]
    miss_all = np.vstack([~np.isfinite(Xc), ~np.isfinite(Xq)])
    inj_cols = np.zeros((n_tot, k_inj), dtype=np.float64)
    for m in range(k_inj):
        kind = int(rng.integers(0, 14))
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
            col = a * za + (1.0 - a) * zb + 0.12 * rng.standard_normal(n_tot)
        elif kind == 3:
            if alt:
                col = np.clip(za * zb, -30.0, 30.0)
            else:
                col = np.clip(za / (1.0 + np.abs(zb)), -25.0, 25.0)
        elif kind == 4:
            if alt:
                col = np.sign(za) * np.log1p(np.abs(za))
            else:
                c_sorted = np.sort(za[:n_c])
                ranks = (np.searchsorted(c_sorted, za, side="left") + 0.5) / float(n_c + 1)
                col = math.sqrt(2.0) * _erfinv_np(np.clip(2.0 * ranks - 1.0, -0.9999, 0.9999))
        elif kind == 5:
            q_pct = float(rng.choice([20.0, 35.0, 50.0, 65.0, 80.0]))
            col = (za > float(np.percentile(za[:n_c], q_pct))).astype(np.float64)
        elif kind == 6:
            w = float(rng.standard_normal() * math.exp(rng.uniform(math.log(0.4), math.log(3.5))))
            ph = float(rng.random() * 2.0 * math.pi)
            col = np.sin(w * za + ph) if alt else np.cos(w * za + ph)
        elif kind == 7:
            sub_m = min(p, int(rng.integers(2, min(12, p) + 1))) if p >= 2 else 1
            Z_sub = Z_all[:, rng.choice(p, size=sub_m, replace=False)]
            col = np.mean(Z_sub, axis=1) if alt else np.sqrt(np.mean(Z_sub * Z_sub, axis=1))
        elif kind == 8:
            miss_rate = np.mean(miss_all, axis=1)
            if alt and float(np.std(miss_rate[:n_c])) > 1e-6:
                col = miss_rate
            else:
                col = np.mean((np.abs(Z_all) > 2.0).astype(np.float64), axis=1)
        elif kind == 9:
            sub_m = min(p, int(rng.integers(2, min(10, p) + 1))) if p >= 2 else 1
            Z_sub = Z_all[:, rng.choice(p, size=sub_m, replace=False)]
            proto = Z_sub[int(rng.integers(0, n_c))]
            d2 = np.mean((Z_sub - proto[None, :]) ** 2, axis=1)
            col = np.exp(-0.5 * d2) if alt else np.sqrt(d2)
        elif kind == 10:
            # Context-hinge / ReLU spline around a context quantile knot
            knot = float(np.percentile(za[:n_c], float(rng.uniform(20.0, 80.0))))
            col = np.maximum(0.0, za - knot) if alt else np.abs(za - knot)
        elif kind == 11:
            # Signed geometric product or standardized difference
            prod = za * zb
            col = (np.sign(prod) * np.sqrt(np.abs(prod))) if alt else (za - zb)
        elif kind == 12:
            # Random Gaussian projection over 2..12 context-standardized columns
            sub_m = min(p, int(rng.integers(2, min(12, p) + 1))) if p >= 2 else 1
            Z_sub = Z_all[:, rng.choice(p, size=sub_m, replace=False)]
            u_vec = rng.standard_normal(sub_m) / math.sqrt(float(sub_m))
            col = Z_sub @ u_vec
        else:
            # Context-decile bucket index (piecewise-constant ordinal feature, fitted strictly on Xc)
            n_q = int(rng.integers(4, 10))
            edges = np.quantile(za[:n_c], np.linspace(1.0 / n_q, (n_q - 1.0) / n_q, n_q - 1))
            col = np.digitize(za, edges).astype(np.float64) / float(n_q - 1)
        inj_cols[:, m] = col
    return inj_cols[:n_c], inj_cols[n_c:]


def _augment_features_zero_leakage(
    Xc: np.ndarray,
    Xq: np.ndarray,
    rng: np.random.Generator,
    noise_max_frac: float = 0.05,
    noise_prob: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies per-feature random sign flip, log-uniform scaling, shift,small context-scaled noise,
    and full random column permutation using strictly context-only statistics (`Xc`, zero query leakage)."""
    p = Xc.shape[1]
    if p == 0:
        return Xc, Xq
    X_full = np.vstack([Xc, Xq]).astype(np.float64, copy=True)
    n_tot, n_c = X_full.shape[0], Xc.shape[0]
    fin_full = np.isfinite(X_full)
    fin_c = fin_full[:n_c]

    scales_c = np.ones(p, dtype=np.float64)
    has_spread = np.zeros(p, dtype=bool)
    for j in range(p):
        fc = fin_c[:, j]
        if int(np.sum(fc)) >= 4:
            vals = X_full[:n_c, j][fc]
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            iqr = float(q75 - q25) / 1.349
            sd = float(np.std(vals))
            sc_j = iqr if (iqr > 1e-8 * max(sd, 1.0) and iqr > 0.0) else sd
            if math.isfinite(sc_j) and sc_j > 1e-12:
                scales_c[j] = sc_j
                has_spread[j] = True

    signs = np.where(rng.random(p) < 0.5, -1.0, 1.0)
    log_muls = np.exp(rng.uniform(-3.0, 3.0, size=p))
    a_mul = signs * log_muls
    b_shift = rng.standard_normal(p) * (2.0 * scales_c * log_muls)
    X_full = np.where(fin_full, X_full * a_mul[None, :] + b_shift[None, :], X_full)

    if noise_max_frac > 0.0 and float(rng.random()) < noise_prob and bool(np.any(has_spread)):
        lo_n = min(0.002, float(noise_max_frac))
        sig_rel = np.exp(rng.uniform(math.log(max(1e-5, lo_n)), math.log(max(1e-5, float(noise_max_frac))), size=p))
        sig_abs = np.where(has_spread, sig_rel * scales_c * log_muls, 0.0)
        if float(rng.random()) < 0.20:
            u_lap = np.clip(rng.random((n_tot, p)) - 0.5, -0.4999, 0.4999)
            eps = -np.sign(u_lap) * np.log1p(-2.0 * np.abs(u_lap)) / math.sqrt(2.0)
        else:
            eps = rng.standard_normal((n_tot, p))
        X_full = np.where(fin_full, X_full + eps * sig_abs[None, :], X_full)

    perm_f = rng.permutation(p)
    X_full = X_full[:, perm_f]
    return X_full[:n_c], X_full[n_c:]


def _synth_teacher_target_np(Xc: np.ndarray, Xq: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    Zc, Zq = _std_ctx_f64(Xc, Xq)
    Z = np.vstack([Zc, Zq])
    n_tot, p = Z.shape
    if p == 0 or n_tot < 24:
        return None
    sub_k = min(p, int(rng.integers(2, min(16, p) + 1))) if p >= 2 else 1
    Zs = Z[:, rng.choice(p, size=sub_k, replace=False)]
    n_c = Xc.shape[0]
    ls = np.exp(rng.uniform(-0.8, 1.3, size=(1, sub_k)))
    W = rng.standard_normal((sub_k, 48)) / ls.T
    b = rng.uniform(0.0, 2.0 * math.pi, size=(1, 48))
    f_gp = (np.cos(Zs @ W + b) * math.sqrt(2.0 / 48)) @ rng.standard_normal(48)

    f_tree = np.zeros(n_tot, dtype=np.float64)
    n_trees = int(rng.integers(4, 9))
    axis_aligned = bool(rng.random() < 0.60)
    for _ in range(n_trees):
        depth = int(rng.integers(2, 6))
        leaf_idx = np.zeros(n_tot, dtype=np.int64)
        for _d in range(depth):
            if axis_aligned:
                feat_j = int(rng.integers(0, sub_k))
                proj = Zs[:, feat_j]
            else:
                proj = Zs @ rng.standard_normal(sub_k)
            thr = float(np.percentile(proj[:n_c], float(rng.uniform(15.0, 85.0))))
            leaf_idx = (leaf_idx << 1) | (proj > thr).astype(np.int64)
        f_tree += rng.standard_normal(1 << depth)[leaf_idx] / math.sqrt(n_trees)

    u = float(rng.random())
    f = f_gp if u < 0.30 else (f_tree if u < 0.65 else (0.55 * f_gp + 0.45 * f_tree))
    sd = float(np.std(f))
    if not math.isfinite(sd) or sd <= 1e-9:
        return None
    f = (f - float(np.mean(f))) / sd
    w_kind = int(rng.integers(0, 5))
    if w_kind == 1:
        f = np.sign(f) * (np.abs(f) ** float(rng.uniform(0.6, 1.8)))
    elif w_kind == 2:
        f = np.sinh(0.65 * f)
    elif w_kind == 3:
        f = f + 0.25 * (f ** 2)
    elif w_kind == 4:
        f = np.exp(0.45 * np.clip(f, -4.0, 4.0)) - 1.0

    noise_scale = float(math.exp(rng.uniform(math.log(0.01), math.log(0.24)))) * max(float(np.std(f)), 1e-6)
    if rng.random() < 0.25:
        chi = np.sqrt(np.maximum(np.mean(rng.standard_normal((n_tot, 4)) ** 2, axis=1), 0.05))
        eps = np.clip(rng.standard_normal(n_tot) / chi, -8.0, 8.0) * 0.7
    else:
        eps = rng.standard_normal(n_tot)
    if rng.random() < 0.30:
        het = 0.5 + np.abs(Zs[:, 0]) * 0.5
        eps = eps * np.clip(het, 0.25, 3.0)
    y = f + noise_scale * eps
    if float(np.std(y[:n_c])) <= 1e-9 or not np.all(np.isfinite(y)):
        return None
    return y[:n_c], y[n_c:]


def _select_cols(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = Xc.shape[1]
    if k >= p:
        return Xc, Xq
    if p <= WIDE_GPU or float(rng.random()) < 0.55:
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
    top_share = float(rng.uniform(0.35, 0.70))
    top = np.argsort(-scores)[: int(round(top_share * k))]
    rest = rng.choice(np.setdiff1d(np.arange(p), top), size=k - len(top), replace=False)
    idx = np.concatenate([top, rest])
    return Xc[:, idx], Xq[:, idx]


def _choose_width_spec(
    p: int,
    rng: np.random.Generator,
    feature_drop_every_batch: bool,
    drop_min_frac: float,
    drop_max_frac: float,
    synth_inj_every_batch: bool,
    inj_min_frac: float,
    inj_max_frac: float,
) -> tuple[int, int, int]:
    """Draws `(k_real, k_inj, F_b)` snapped to `WIDTH_BINS` while honoring wide drop & injection bounds."""
    if feature_drop_every_batch and p >= 2:
        d_lo = max(1, min(p - 1, int(math.floor(drop_min_frac * p))))
        d_hi = max(d_lo, min(p - 1, int(math.floor(drop_max_frac * p))))
        k_real_min = max(1, min(WIDE_GPU, p - d_hi))
        k_real_max = max(k_real_min, min(WIDE_GPU, p - d_lo))
    else:
        k_real_min = max(1, min(WIDE_GPU, p))
        k_real_max = k_real_min

    k_real_target = int(rng.integers(k_real_min, k_real_max + 1))
    f_inj_draw = float(rng.uniform(inj_min_frac, inj_max_frac)) if inj_max_frac > 0.0 else 0.0
    f_target = min(WIDE_GPU, max(1, int(round(k_real_target / max(0.20, 1.0 - f_inj_draw)))))

    candidates: list[tuple[int, int, int, int]] = []
    for fb in WIDTH_BINS:
        k_hi = int(math.floor(inj_max_frac * fb))
        k_lo = max(1, int(math.ceil(inj_min_frac * fb))) if (synth_inj_every_batch and k_hi >= 1) else 0
        k_lo = min(k_lo, k_hi)
        kr_lo = max(k_real_min, fb - k_hi)
        kr_hi = min(k_real_max, fb - k_lo)
        if kr_lo <= kr_hi:
            kr_pick = int(np.clip(k_real_target, kr_lo, kr_hi))
            ki_pick = fb - kr_pick
            candidates.append((abs(fb - f_target), kr_pick, ki_pick, fb))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        _, kr, ki, fb = candidates[0]
        return kr, ki, fb

    ki_fb = min(WIDE_GPU - k_real_target, max(0, int(round(f_inj_draw / max(0.20, 1.0 - f_inj_draw) * k_real_target))))
    if synth_inj_every_batch and inj_max_frac > 0.0 and ki_fb == 0 and k_real_target < WIDE_GPU and int(math.floor(inj_max_frac * (k_real_target + 1))) >= 1:
        ki_fb = 1
    return k_real_target, ki_fb, k_real_target + ki_fb


def sample_max_entropy_real_episode(
    bank: MmapF64Bank,
    t_idx: int,
    rng: np.random.Generator,
    valid_tcols_cache: dict[int, tuple[int, ...]],
    max_episode_rows: int = DEFAULT_MAX_EPISODE_ROWS,
    random_split_every_batch: bool = True,
    feature_drop_every_batch: bool = True,
    drop_min_frac: float = 0.05,
    drop_max_frac: float = 0.65,
    synth_inj_every_batch: bool = True,
    inj_min_frac: float = 0.05,
    inj_max_frac: float = 0.30,
    rand_target_prob: float = 0.45,
    synth_teacher_prob: float = 0.15,
    feature_noise_max: float = 0.05,
    force_canonical_target: bool = False,
    n_ctx_for_shape: dict[tuple[int, int], int] | None = None,
    preferred_rows: int | None = None,
):
    """Samples one high-entropy episode from `t_idx` across the full table (zero permanent row downsampling)."""
    tab_mmap = bank.tabs_f64[t_idx]
    r_raw, c_tot = tab_mmap.shape
    if r_raw < 28 or c_tot < 2:
        return None

    if t_idx not in valid_tcols_cache:
        valid_tcols_cache[t_idx] = _discover_valid_target_cols(tab_mmap, min_rows=28)
    valid_tcols = valid_tcols_cache[t_idx]

    # (3) Random target column choice where valid + canonical target + nonlinear synthetic teacher
    u_tm = 0.0 if force_canonical_target else float(rng.random())
    other_valid = [j for j in valid_tcols if j != (c_tot - 1)]
    blend_real_teacher = False
    if not force_canonical_target and u_tm < synth_teacher_prob:
        tmode = 2
        blend_real_teacher = bool(rng.random() < 0.30)
        tcol = int(rng.choice(valid_tcols)) if valid_tcols else (c_tot - 1)
    elif not force_canonical_target and other_valid and u_tm < (synth_teacher_prob + rand_target_prob):
        tmode = 1
        tcol = int(rng.choice(other_valid))
    else:
        tmode = 0
        tcol = c_tot - 1

    valid_rows: np.ndarray
    y_valid: np.ndarray | None = None
    if tmode in (0, 1) or blend_real_teacher:
        col_t = tab_mmap[:, tcol]
        valid_rows = np.flatnonzero(np.isfinite(col_t))
        if valid_rows.size < 28:
            return None
        y_valid = np.asarray(col_t[valid_rows], dtype=np.float64)
        if tmode == 0 and rng.random() < 0.40 and valid_rows.size >= 36:
            atom = _detect_atom_np(y_valid)
            if atom is not None:
                keep = np.abs(y_valid - atom) > 1e-7 * max(1.0, abs(atom))
                if int(np.sum(keep)) >= 32:
                    valid_rows = valid_rows[keep]
                    y_valid = y_valid[keep]
    else:
        valid_rows = np.arange(r_raw, dtype=np.int64)

    r_tot = int(valid_rows.size)
    if r_tot < 28:
        return None

    # (2) Varied episode row subsampling across full corpus (NO permanent downsample to 300 rows)
    r_cap = min(r_tot, int(max_episode_rows))
    if preferred_rows is not None:
        if r_cap >= int(preferred_rows):
            n_rows = int(preferred_rows)
        else:
            ok_fb = [n for n in FALLBACK_ROW_BINS if n <= r_cap]
            n_rows = ok_fb[-1] if ok_fb else r_cap
    else:
        ok_r = [n for n in ROW_BINS if n <= r_cap]
        if not ok_r:
            n_rows = r_cap
        elif len(ok_r) >= 3 and float(rng.random()) < 0.30:
            n_rows = int(rng.choice(ok_r[-2:]))
        else:
            n_rows = int(rng.choice(ok_r))

    feats = [j for j in range(c_tot) if j != tcol]
    if tmode == 2 and not blend_real_teacher and len(feats) > 1 and rng.random() < 0.50:
        feats = [j for j in feats if j != c_tot - 1] or feats
    p = len(feats)
    if p < 1:
        return None

    # (1) Wider random column drop + zero-leakage synthetic column injection
    k_real, k_inj, F = _choose_width_spec(
        p, rng,
        feature_drop_every_batch=feature_drop_every_batch,
        drop_min_frac=drop_min_frac,
        drop_max_frac=drop_max_frac,
        synth_inj_every_batch=synth_inj_every_batch,
        inj_min_frac=inj_min_frac,
        inj_max_frac=inj_max_frac,
    )

    if n_ctx_for_shape is not None and (n_rows, F) in n_ctx_for_shape:
        n_ctx = int(n_ctx_for_shape[(n_rows, F)])
        p_ctx = n_ctx / float(n_rows)
    else:
        p_ctx = float(rng.uniform(0.18, 0.95))
        n_ctx = max(16, min(n_rows - 4, int(round(p_ctx * n_rows))))
        if n_ctx_for_shape is not None:
            n_ctx_for_shape[(n_rows, F)] = n_ctx

    # High-entropy row subsampling over all r_tot rows in the full table
    u_sub = float(rng.random())
    if r_tot <= n_rows:
        sel_local = rng.permutation(r_tot)[:n_rows]
    elif u_sub < 0.50 or y_valid is None:
        sel_local = rng.choice(r_tot, size=n_rows, replace=False)
    elif u_sub < 0.85:
        y_ord = np.argsort(y_valid + 1e-12 * max(float(np.std(y_valid)), 1.0) * rng.standard_normal(r_tot), kind="mergesort")
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
        sel_local = rng.permutation(sel[:n_rows])
    else:
        # Covariate-density-tilted row subsample along a random feature column
        j_tilt = int(rng.choice(feats))
        col_tilt = np.asarray(tab_mmap[valid_rows, j_tilt], dtype=np.float64)
        fin_t = np.isfinite(col_tilt)
        if int(np.sum(fin_t)) >= n_rows:
            med_t = float(np.median(col_tilt[fin_t]))
            sd_t = max(float(np.std(col_tilt[fin_t])), 1e-9)
            z_t = np.clip((np.where(fin_t, col_tilt, med_t) - med_t) / sd_t, -4.0, 4.0)
            tilt_temp = float(rng.uniform(-0.65, 0.65))
            logits_w = tilt_temp * z_t
            logits_w -= float(np.max(logits_w))
            w_sub = np.exp(logits_w)
            w_sub /= float(np.sum(w_sub))
            sel_local = rng.choice(r_tot, size=n_rows, replace=False, p=w_sub)
        else:
            sel_local = rng.choice(r_tot, size=n_rows, replace=False)

    sub = np.asarray(tab_mmap[valid_rows[sel_local]], dtype=np.float64)
    X_all = sub[:, feats]
    y_all = sub[:, tcol].copy()

    # Fresh CSPRNG context/query split (50% pure uniform permutation, 50% quantile-stratified)
    if random_split_every_batch and float(rng.random()) < 0.50 and n_rows >= 28 and np.all(np.isfinite(y_all)):
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
    else:
        perm_r = rng.permutation(n_rows)
        X_all, y_all = X_all[perm_r], y_all[perm_r]

    Xc, Xq = X_all[:n_ctx], X_all[n_ctx:]
    yc, yq = y_all[:n_ctx].copy(), y_all[n_ctx:].copy()
    if (tmode < 2 or blend_real_teacher) and float(np.std(yc)) <= 1e-12:
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
    elif rng.random() < 0.40:
        sy = float(np.std(yc))
        e = rng.standard_normal(n_rows) / np.sqrt(np.maximum(np.mean(rng.standard_normal((n_rows, 3)) ** 2, axis=1), 1e-3)) / 1.732
        jit = sy * float(rng.uniform(0.01, 0.035)) * np.clip(e, -1.345, 1.345)
        yc, yq = yc + jit[:n_ctx], yq + jit[n_ctx:]

    if k_inj > 0:
        Ic, Iq = _inject_synth_cols_np(Xc, Xq, k_inj, rng)
        Xc, Xq = np.hstack([Xc, Ic]), np.hstack([Xq, Iq])

    # (4) Per-feature random scaling/affine, small context-scaled noise, and full random feature permutation
    Xc, Xq = _augment_features_zero_leakage(Xc, Xq, rng, noise_max_frac=feature_noise_max)

    # Target spaces: 42% 5-fold SVD robust-ridge residual, 33% Yeo-Johnson warp, 25% raw/random-affine target
    u_sp = float(rng.random())
    space = 0
    y_ep = np.concatenate([yc, yq])
    if u_sp < 0.42:
        oof, pq, used, _ = _robust_ridge_np(Xc, yc, Xq, seed=int(rng.integers(1, 1 << 30)))
        if used:
            y_ep = np.concatenate([yc - oof, yq - pq])
            space = 1
    elif u_sp < 0.75:
        yj_pair = _yeojohnson_pair_np(yc, yq)
        if yj_pair is not None and float(np.std(yj_pair[0])) > 1e-8:
            y_ep = np.concatenate([yj_pair[0], yj_pair[1]])
            space = 2
    else:
        y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        y_scale = float(math.exp(rng.uniform(-2.0, 2.0)))
        y_ep = y_sign * y_scale * y_ep

    X_ep = np.vstack([Xc, Xq])
    if X_ep.shape != (n_rows, F) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx])) <= 1e-12:
        return None
    return X_ep, y_ep, n_ctx, (int(t_idx), tmode, int(k_inj), int(F), int(n_rows), space, int(max(0, p - k_real)), int(p))


def _sample_full_synth_special(
    B: int,
    N: int,
    Fn: int,
    n_ctx: int,
    rng: np.random.Generator,
    feature_drop_every_batch: bool = True,
    drop_min_frac: float = 0.05,
    drop_max_frac: float = 0.65,
    synth_inj_every_batch: bool = True,
    inj_min_frac: float = 0.05,
    inj_max_frac: float = 0.30,
    feature_noise_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    mode = int(rng.integers(0, 3))
    xs = np.zeros((B, N, Fn), dtype=np.float64)
    ys = np.zeros((B, N), dtype=np.float64)
    Fn_gen = Fn + (max(1, int(round(drop_max_frac * Fn))) if feature_drop_every_batch else 0)
    for b in range(B):
        if mode == 0:
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
        elif mode == 1:
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
        else:
            r_lat = min(6, Fn_gen)
            z_lat = rng.standard_normal((N, r_lat))
            mix = rng.standard_normal((r_lat, Fn_gen)) / math.sqrt(r_lat)
            X_raw = z_lat @ mix + 0.45 * rng.standard_normal((N, Fn_gen))
            y_clean = np.zeros(N, dtype=np.float64)
            for _ in range(6):
                depth = int(rng.integers(2, 5))
                leaf = np.zeros(N, dtype=np.int64)
                for _d in range(depth):
                    cj = int(rng.integers(0, Fn_gen))
                    thr = float(np.percentile(X_raw[:, cj], float(rng.uniform(20.0, 80.0))))
                    leaf = (leaf << 1) | (X_raw[:, cj] > thr).astype(np.int64)
                y_clean += rng.standard_normal(1 << depth)[leaf] * 0.5
            miss_prob = float(rng.uniform(0.03, 0.18))
            mar_gate = 1.0 / (1.0 + np.exp(-X_raw[:, 0]))
            miss_mask = rng.random((N, Fn_gen)) < (miss_prob * mar_gate[:, None])
            miss_mask[:, 0] = False
            y_clean = y_clean + 0.35 * np.mean(miss_mask, axis=1)
            X_raw = np.where(miss_mask, np.nan, X_raw)
            y_b = y_clean + float(rng.uniform(0.02, 0.18)) * max(float(np.std(y_clean)), 1e-3) * rng.standard_normal(N)

        k_hi_sp = int(math.floor(inj_max_frac * Fn))
        k_lo_sp = max(1, int(math.ceil(inj_min_frac * Fn))) if (synth_inj_every_batch and k_hi_sp >= 1) else 0
        k_sp = int(rng.integers(min(k_lo_sp, k_hi_sp), k_hi_sp + 1)) if k_hi_sp >= 1 else 0
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
        Xc_aug, Xq_aug = _augment_features_zero_leakage(X_full[:n_ctx], X_full[n_ctx:], rng, noise_max_frac=feature_noise_max)
        xs[b] = np.vstack([Xc_aug, Xq_aug])
        ys[b] = y_b
    return xs, ys


def _pack_worker_item(
    xa: np.ndarray,
    ya: np.ndarray,
    n_ctx: int,
    infos,
    fps: list[bytes],
    rng: np.random.Generator,
    postnorm_affine_max: float = 0.22,
):
    zm_np, yz_np, ym_np, ys_np = ICLH450MPS.prenorm_cpu_np(xa, ya[:, :n_ctx], int(n_ctx))
    # Post-normalise_f64 per-feature affine jitter on non-missing cells so random feature scaling/shift
    # survives context IQR normalization directly into the 104M cell encoder!
    if postnorm_affine_max > 0.0 and float(rng.random()) < 0.85:
        B_p, _, Fn_p, _ = zm_np.shape
        g_af = np.exp(rng.uniform(-postnorm_affine_max, postnorm_affine_max, size=(B_p, 1, Fn_p))).astype(np.float32)
        d_af = rng.uniform(-0.8 * postnorm_affine_max, 0.8 * postnorm_affine_max, size=(B_p, 1, Fn_p)).astype(np.float32)
        z_val = zm_np[..., 0]
        not_miss = zm_np[..., 1] < 0.5
        zm_np[..., 0] = np.where(not_miss, np.clip(z_val * g_af + d_af, -float(CLIP_X), float(CLIP_X)), np.float32(0.0))
    yq_np = np.ascontiguousarray(ya[:, n_ctx:], dtype=np.float64)
    return (zm_np, yz_np, ym_np, ys_np, yq_np, int(n_ctx), infos, fps)


def _target_batch_size(N: int, Fn: int, bmax: int, tokens_cap: int, row_tokens_cap: int, attn_cap: int) -> int:
    return max(
        1,
        min(
            int(bmax),
            int(tokens_cap) // max(N * (Fn + 4), 1),
            int(row_tokens_cap) // max(N, 1),
            int(attn_cap) // max(N * N, 1),
        ),
    )


def _spawned_worker_loop(
    worker_id: int,
    run_nonce: bytes,
    mmap_dir_s: str,
    expect_sha: str,
    dataset_mix: str,
    max_episode_rows: int,
    tokens_cap: int,
    row_tokens_cap: int,
    attn_cap: int,
    bmax: int,
    cpu_hard: float,
    random_split_every_batch: bool,
    feature_drop_every_batch: bool,
    drop_min_frac: float,
    drop_max_frac: float,
    synth_inj_every_batch: bool,
    inj_min_frac: float,
    inj_max_frac: float,
    rand_target_prob: float,
    synth_teacher_prob: float,
    feature_noise_max: float,
    postnorm_affine_max: float,
    synth_prob: float,
    prior_seed_base: int,
    out_q,
    stop_ev,
) -> None:
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception:
        pass
    try:
        os.nice(12)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    sampling_mode = "uniform_dataset" if dataset_mix == "uniform" else "smart_stratified"
    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha, sampling_mode=sampling_mode)
    gov = HostCpuGovernor(hard_pct=cpu_hard, check_every_s=3.0)
    prior: Prior | None = None
    valid_tcols_cache: dict[int, tuple[int, ...]] = {}

    # Per-worker shape bin accumulator: preserves EXACT 1/3525 uniform-by-dataset exposure without row-count rejection!
    open_bins: dict[tuple[int, int], dict] = {}
    n_ctx_for_shape: dict[tuple[int, int], int] = {}
    ready_items: list = []
    allowed_row_bins = [n for n in ROW_BINS if n <= int(max_episode_rows)] or [min(32, int(max_episode_rows))]
    active_rows = allowed_row_bins[worker_id % len(allowed_row_bins)]
    seq = 0

    def _flush_bin(shape_key: tuple[int, int], rng_f: np.random.Generator) -> None:
        b_st = open_bins.pop(shape_key, None)
        n_ctx_for_shape.pop(shape_key, None)
        if not b_st or not b_st["xs"]:
            return
        xs_b, ys_b, infos_b, n_ctx_b = b_st["xs"], b_st["ys"], b_st["infos"], int(b_st["n_ctx"])
        fps_b = [dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_b, ys_b)]
        ready_items.append(
            _pack_worker_item(
                np.stack(xs_b), np.stack(ys_b), n_ctx_b, infos_b, fps_b, rng_f,
                postnorm_affine_max=postnorm_affine_max,
            )
        )

    while not stop_ev.is_set():
        while ready_items and not stop_ev.is_set():
            nxt = ready_items.pop(0)
            while not stop_ev.is_set():
                try:
                    out_q.put(nxt, timeout=1.0)
                    break
                except queue.Full:
                    gov.pace()
                    continue

        gov.pace()
        seq += 1
        b_seed = hashlib.blake2b(run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint32))

        if synth_prob > 0.0 and float(rng.random()) < synth_prob:
            if prior is None:
                prior = Prior(torch.device("cpu"), int(prior_seed_base) + 1009 * (worker_id + 1))
            r_py = std_random.Random(int(rng.integers(1, 1 << 62)))
            Fn = max(4, min(WIDE_GPU, draw_features(r_py)))
            N = int(rng.choice(SYNTH_N, p=SYNTH_N_P))
            N = min(N, int(max_episode_rows))
            B = _target_batch_size(N, Fn, min(int(bmax), 20), tokens_cap, row_tokens_cap, attn_cap)
            n_ctx = max(16, min(N - 4, int(round(float(rng.uniform(0.18, 0.95)) * N))))
            if float(rng.random()) < 0.55:
                Fn_gen = min(WIDE_GPU, Fn + (max(1, int(round(drop_max_frac * Fn))) if feature_drop_every_batch else 0))
                xt_c, yt_c, _, _ = INJ.sample_synth(prior, r_py, B, N, Fn_gen, 0.55, PROBS_V3S)
                xa_g, ya = xt_c.detach().cpu().double().numpy(), yt_c.detach().cpu().double().numpy()
                xa_list = []
                for b_i in range(B):
                    xb_raw = xa_g[b_i]
                    if feature_drop_every_batch and Fn_gen >= 3:
                        d_lo_s = max(1, min(Fn_gen - 2, int(math.floor(drop_min_frac * Fn_gen))))
                        d_hi_s = max(d_lo_s, min(Fn_gen - 2, int(math.floor(drop_max_frac * Fn_gen))))
                        n_ds = int(rng.integers(d_lo_s, d_hi_s + 1))
                        xb_sub = xb_raw[:, rng.choice(Fn_gen, size=Fn_gen - n_ds, replace=False)]
                    else:
                        xb_sub = xb_raw
                    k_hi_s = int(math.floor(inj_max_frac * Fn))
                    k_lo_s = max(1, int(math.ceil(inj_min_frac * Fn))) if (synth_inj_every_batch and k_hi_s >= 1) else 0
                    k_inj_s = int(rng.integers(min(k_lo_s, k_hi_s), k_hi_s + 1)) if k_hi_s >= 1 else 0
                    k_keep_s = max(1, Fn - k_inj_s)
                    if xb_sub.shape[1] > k_keep_s:
                        xb_sub = xb_sub[:, rng.choice(xb_sub.shape[1], size=k_keep_s, replace=False)]
                    elif xb_sub.shape[1] < k_keep_s:
                        xb_sub = np.hstack([xb_sub, rng.standard_normal((N, k_keep_s - xb_sub.shape[1]))])
                    if k_inj_s > 0:
                        Ic_s, Iq_s = _inject_synth_cols_np(xb_sub[:n_ctx], xb_sub[n_ctx:], k_inj_s, rng)
                        xb_sub = np.vstack([np.hstack([xb_sub[:n_ctx], Ic_s]), np.hstack([xb_sub[n_ctx:], Iq_s])])
                    Xc_a, Xq_a = _augment_features_zero_leakage(xb_sub[:n_ctx], xb_sub[n_ctx:], rng, noise_max_frac=feature_noise_max)
                    xa_list.append(np.vstack([Xc_a, Xq_a]))
                xa = np.stack(xa_list)
            else:
                xa, ya = _sample_full_synth_special(
                    B, N, Fn, n_ctx, rng,
                    feature_drop_every_batch=feature_drop_every_batch,
                    drop_min_frac=drop_min_frac,
                    drop_max_frac=drop_max_frac,
                    synth_inj_every_batch=synth_inj_every_batch,
                    inj_min_frac=inj_min_frac,
                    inj_max_frac=inj_max_frac,
                    feature_noise_max=feature_noise_max,
                )
            u_s_sp = float(rng.random())
            if u_s_sp < 0.40:
                seed_r = int(rng.integers(1, 1 << 30))
                for b_i in range(xa.shape[0]):
                    oof_b, pq_b, used_b, _ = _robust_ridge_np(xa[b_i, :n_ctx], ya[b_i, :n_ctx], xa[b_i, n_ctx:], seed=seed_r + 31 * b_i)
                    if used_b:
                        ya[b_i, :n_ctx] -= oof_b
                        ya[b_i, n_ctx:] -= pq_b
            elif u_s_sp < 0.70:
                for b_i in range(xa.shape[0]):
                    yj_b = _yeojohnson_pair_np(ya[b_i, :n_ctx], ya[b_i, n_ctx:])
                    if yj_b is not None and float(np.std(yj_b[0])) > 1e-8:
                        ya[b_i, :n_ctx] = yj_b[0]
                        ya[b_i, n_ctx:] = yj_b[1]
            fps = [dataset_fingerprint_128(xa[b_], ya[b_], n_ctx) for b_ in range(xa.shape[0])]
            ready_items.append(_pack_worker_item(xa, ya, int(n_ctx), None, fps, rng, postnorm_affine_max=postnorm_affine_max))
            continue

        # (5) Exact uniform-by-dataset draw over all 3,525 S3 tables (no row-count rejection!)
        if seq > 1 and (seq & 255) == 1:
            old_r = active_rows
            for k_old in [k_o for k_o, st_o in open_bins.items() if k_o[0] == old_r and len(st_o["xs"]) >= 2]:
                _flush_bin(k_old, rng)
            active_rows = int(rng.choice(allowed_row_bins))
        t_idx = bank.sample_table_idx(rng)
        got = None
        for retry_i in range(4):
            got = sample_max_entropy_real_episode(
                bank, t_idx, rng, valid_tcols_cache,
                max_episode_rows=max_episode_rows,
                random_split_every_batch=random_split_every_batch,
                feature_drop_every_batch=feature_drop_every_batch,
                drop_min_frac=drop_min_frac,
                drop_max_frac=drop_max_frac,
                synth_inj_every_batch=synth_inj_every_batch,
                inj_min_frac=inj_min_frac,
                inj_max_frac=inj_max_frac,
                rand_target_prob=rand_target_prob,
                synth_teacher_prob=synth_teacher_prob,
                feature_noise_max=feature_noise_max,
                force_canonical_target=(retry_i >= 2),
                n_ctx_for_shape=n_ctx_for_shape,
                preferred_rows=active_rows,
            )
            if got is not None:
                break
        if got is None:
            continue

        X_ep, y_ep, n_ctx_ep, info_ep = got
        sh_key = (int(X_ep.shape[0]), int(X_ep.shape[1]))
        b_st = open_bins.get(sh_key)
        if b_st is None or int(b_st["n_ctx"]) != int(n_ctx_ep):
            if b_st is not None:
                _flush_bin(sh_key, rng)
            b_tgt = _target_batch_size(sh_key[0], sh_key[1], min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap)
            b_st = {"xs": [], "ys": [], "infos": [], "n_ctx": int(n_ctx_ep), "b_target": b_tgt, "open_seq": seq}
            open_bins[sh_key] = b_st
            n_ctx_for_shape[sh_key] = int(n_ctx_ep)

        b_st["xs"].append(X_ep)
        b_st["ys"].append(y_ep)
        b_st["infos"].append(info_ep)
        if len(b_st["xs"]) >= int(b_st["b_target"]):
            _flush_bin(sh_key, rng)

        if (seq & 63) == 0 and open_bins:
            stale_keys = [k_s for k_s, st_s in open_bins.items() if (seq - int(st_s["open_seq"])) >= 384]
            for k_s in stale_keys:
                _flush_bin(k_s, rng)


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
        mean_ep = float(np.mean(self.table_eps)) if b.n_tables else 0.0
        cv_ep = (float(np.std(self.table_eps)) / mean_ep) if mean_ep > 1e-9 else None
        return {
            "per_cohort": per, "real_episodes": real_eps, "synthetic_datasets": self.synth_datasets,
            "real_cell_tokens": int(self.table_cells.sum()), "synthetic_cell_tokens": self.synth_cells,
            "dev91_share_of_real_episodes": (d91 / real_eps) if real_eps else None,
            "target_modes": dict(zip(("own", "other_column", "teacher"), self.tmode.tolist())),
            "target_spaces": dict(zip(("raw", "ridge_residual", "yeo_johnson"), self.space.tolist())),
            "injected_column_share": self.inj_cols / max(1, self.inj_cols + self.real_cols),
            "feature_drop_share_p_ge_2": (self.real_eps_dropped / self.real_eps_p_ge_2) if self.real_eps_p_ge_2 else None,
            "dropped_real_columns_total": self.dropped_real_cols,
            "table_episodes_cv": round(cv_ep, 4) if cv_ep is not None else None,
            "dataset_sampling_mode": getattr(b, "sampling_mode", "uniform_dataset"),
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


def _atomic_save(obj, path: Path, mode: int | None = None) -> None:
    tmp = path.with_suffix(".tmp")
    if path.exists():
        try:
            os.chmod(path, 0o644)
        except Exception:
            pass
    torch.save(obj, tmp)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), default="L")
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--mmap-bank", required=True)
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=96.0)
    ap.add_argument("--save-every-datasets", type=int, default=150_000)
    ap.add_argument("--keep-latest-only", type=int, choices=(0, 1), default=1)
    ap.add_argument("--last-every-s", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--swa-decay", type=float, default=0.9995)
    ap.add_argument("--synth-prob", type=float, default=0.20)
    ap.add_argument("--dataset-mix", choices=("uniform", "smart_stratified"), default="uniform")
    ap.add_argument("--max-episode-rows", type=int, default=DEFAULT_MAX_EPISODE_ROWS)
    ap.add_argument("--tokens", type=int, default=760_000)
    ap.add_argument("--row-tokens", type=int, default=15_360)
    ap.add_argument("--attn-cap", type=int, default=18_500_000)
    ap.add_argument("--bmax", type=int, default=40)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-row-blocks", type=int, choices=(0, 1), default=1)
    ap.add_argument("--cpu-hard-pct", type=float, default=82.0)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.80)
    ap.add_argument("--random-split-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-min-frac", type=float, default=0.05)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.65)
    ap.add_argument("--synth-inj-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--synth-inj-min-frac", type=float, default=0.05)
    ap.add_argument("--synth-inj-max-frac", type=float, default=0.30)
    ap.add_argument("--rand-target-prob", type=float, default=0.45)
    ap.add_argument("--synth-teacher-prob", type=float, default=0.15)
    ap.add_argument("--feature-noise-max", type=float, default=0.05)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=450_441)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if not (0.0 <= a.synth_inj_min_frac <= a.synth_inj_max_frac <= 0.50):
        raise SystemExit(f"invalid synth-inj range [{a.synth_inj_min_frac}, {a.synth_inj_max_frac}] (expected 0 <= min <= max <= 0.50)")
    if not (0.0 <= a.feature_drop_min_frac <= a.feature_drop_max_frac <= 0.85):
        raise SystemExit(f"invalid feature-drop range [{a.feature_drop_min_frac}, {a.feature_drop_max_frac}] (expected 0 <= min <= max <= 0.85)")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_p, last_p, latest_p = out_dir / "ckpt_h450_final.pt", out_dir / "ckpt_last.pt", out_dir / "ckpt_latest.pt"
    if final_p.exists():
        raise SystemExit(f"{final_p} exists (create-once)")
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    if sha256_file(Path(a.init)) != a.init_sha256:
        raise SystemExit("init checkpoint sha256 mismatch")

    sampling_mode = "uniform_dataset" if a.dataset_mix == "uniform" else "smart_stratified"
    t_load0 = time.time()
    bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, verify_data_sha=True, sampling_mode=sampling_mode)
    bank_load_s = round(time.time() - t_load0, 3)
    ledger = Ledger128(bank)
    run_nonce = hashlib.blake2b(secrets.token_bytes(32) + int(a.seed).to_bytes(8, "little"), digest_size=32).digest()

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

    model = ICLH450MPS(a.size, checkpoint_row_blocks=bool(a.checkpoint_row_blocks)).to(dev)
    ema = ICLH450MPS(a.size).to(dev).eval()
    ema_swa = ICLH450MPS(a.size).to(dev).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    for p in ema_swa.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)
    state = {
        "step": 0, "micro_batches": 0, "accum_in_group": 0, "real_batches": 0, "synth_batches": 0,
        "datasets": 0, "cell_tokens": 0, "oom_skips": 0, "nonfinite_skips": 0, "train_s": 0.0,
        "epochs_saved": 0, "last_epoch_datasets": 0, "resumes": 0, "switch_history": [],
    }
    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        if ck.get("bank_sha256") and ck["bank_sha256"] != bank.sha256:
            raise SystemExit(f"resume bank_sha256 mismatch: {ck.get('bank_sha256')} != {bank.sha256}")
        if ck.get("init_sha256") and ck["init_sha256"] != a.init_sha256:
            raise SystemExit(f"resume init_sha256 mismatch: {ck.get('init_sha256')} != {a.init_sha256}")
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
        ema_swa.load_state_dict(ck.get("ema_swa") or ck["ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        if "last_epoch_datasets" not in ck["state"]:
            state["last_epoch_datasets"] = int(state["datasets"])
        if "switch_history" not in state or not isinstance(state["switch_history"], list):
            state["switch_history"] = []
        state["resumes"] += 1
        ledger.load(ck["ledger"])
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        w_init = ck.get("ema") or ck["model"]
        model.load_state_dict(w_init, strict=True)
        ema.load_state_dict(w_init, strict=True)
        ema_swa.load_state_dict(w_init, strict=True)

    sampler_cfg = {
        "protocol": "fleet-r9-max-entropy-uniform-ds",
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "resume_step": int(state["step"]),
        "resume_datasets": int(state["datasets"]),
        "keep_latest_only": bool(a.keep_latest_only),
        "dataset_mix": a.dataset_mix,
        "max_episode_rows": int(a.max_episode_rows),
        "feature_drop_frac": [float(a.feature_drop_min_frac), float(a.feature_drop_max_frac)],
        "synth_inj_frac": [float(a.synth_inj_min_frac), float(a.synth_inj_max_frac)],
        "rand_target_prob": float(a.rand_target_prob),
        "synth_teacher_prob": float(a.synth_teacher_prob),
        "feature_noise_max": float(a.feature_noise_max),
        "postnorm_affine_max": float(a.postnorm_affine_max),
    }
    state["switch_history"].append(sampler_cfg)

    params_cnt = int(n_params(model))
    prior_seed_base = int(a.seed) + 17 + 1000 * int(state["resumes"])
    gov = HostCpuGovernor(hard_pct=a.cpu_hard_pct, check_every_s=5.0)

    ctx = mp.get_context("spawn")
    ep_q = ctx.Queue(maxsize=max(32, 8 * a.workers))
    stop_ev = ctx.Event()
    procs = [
        ctx.Process(
            target=_spawned_worker_loop,
            args=(
                w, run_nonce, str(a.mmap_bank), a.bank_sha256, a.dataset_mix, int(a.max_episode_rows),
                int(a.tokens), int(a.row_tokens), int(a.attn_cap), int(a.bmax), float(a.cpu_hard_pct),
                bool(a.random_split_every_batch), bool(a.feature_drop_every_batch),
                float(a.feature_drop_min_frac), float(a.feature_drop_max_frac),
                bool(a.synth_inj_every_batch), float(a.synth_inj_min_frac), float(a.synth_inj_max_frac),
                float(a.rand_target_prob), float(a.synth_teacher_prob),
                float(a.feature_noise_max), float(a.postnorm_affine_max),
                float(a.synth_prob), prior_seed_base,
                ep_q, stop_ev,
            ),
            daemon=True,
        )
        for w in range(a.workers)
    ]
    for p in procs:
        p.start()

    def save_latest_eval_ckpt() -> str:
        swa_cpu = {kk: vv.cpu() for kk, vv in ema_swa.state_dict().items()}
        _atomic_save(
            {"ema": swa_cpu, "model": swa_cpu, "state": dict(state), "size": a.size, "sampler_cfg": sampler_cfg},
            latest_p,
            mode=0o644,
        )
        return sha256_file(latest_p)

    def save_last() -> None:
        _atomic_save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                "ema_swa": {k: v.cpu() for k, v in ema_swa.state_dict().items()},
                "opt": opt.state_dict(), "state": state, "ledger": ledger.state(),
                "size": a.size, "bank_sha256": bank.sha256, "init_sha256": a.init_sha256, "argv": sys.argv,
            },
            last_p,
            mode=0o644,
        )

    def save_epoch(k: int) -> None:
        jp = out_dir / f"epoch_{k:02d}.json"
        if jp.exists():
            state["epochs_saved"] = max(int(state["epochs_saved"]), k)
            return
        if bool(a.keep_latest_only):
            # New policy: save ONLY rolling `ckpt_latest.pt` + metrics JSON (no per-epoch model `.pt` files)
            ck_sha = save_latest_eval_ckpt()
            ck_path_str = str(latest_p)
        else:
            ep_p = out_dir / f"ckpt_epoch_{k:02d}.pt"
            swa_cpu = {kk: vv.cpu() for kk, vv in ema_swa.state_dict().items()}
            _atomic_save({"ema": swa_cpu, "model": swa_cpu, "state": dict(state), "size": a.size}, ep_p, mode=0o444)
            ck_sha = sha256_file(ep_p)
            ck_path_str = str(ep_p)
        doc = {
            "schema": "jope.h450.epoch/v1", "epoch": k, "train_hours": state["train_s"] / 3600.0, "step": state["step"],
            "ckpt": ck_path_str, "ckpt_sha256": ck_sha, "keep_latest_only": bool(a.keep_latest_only),
            "protocol": "fleet-r9-max-entropy-uniform-ds", "sampler_cfg": sampler_cfg,
            "init_sha256": a.init_sha256, "bank_sha256": bank.sha256,
            "quarantined_in_corpus": True, "canonical_promotion_forbidden": True,
            "weights_source": "ema_swa", "swa_decay": float(a.swa_decay),
            "exposure": ledger.summary(),
        }
        jp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.chmod(jp, 0o444)
        state["epochs_saved"] = k

    # Ensure rolling `ckpt_latest.pt` exists immediately at start/resume boundary
    save_latest_eval_ckpt()

    budget_s = max(15.0, a.hours * 3600.0)
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)
    log_f.write(json.dumps({
        "start": time.strftime("%FT%TZ", time.gmtime()), "host": platform.node(), "device": str(dev),
        "resume": bool(a.resume), "params": params_cnt, "bank_tables": bank.n_tables,
        "bank_load_s": bank_load_s, "train_s": state["train_s"], "step": state["step"],
        "lr_base": a.lr, "swa_decay": a.swa_decay, "ckpt_row_blocks": bool(a.checkpoint_row_blocks),
        "sampler_cfg": sampler_cfg,
    }) + "\n")

    grad_accum = max(1, int(a.grad_accum))
    opt.zero_grad(set_to_none=True)
    t_seg = time.time()
    s_seg0 = state["train_s"]
    t_log = t_last = time.time()
    run_loss, run_n, run_ds = 0.0, 0, 0
    swa_decay_target = max(0.99, min(0.99995, float(a.swa_decay)))
    steps_since_start = 0

    try:
        while not stop_flag["stop"]:
            state["train_s"] = s_seg0 + (time.time() - t_seg)
            if state["train_s"] >= budget_s or (a.max_steps > 0 and state["step"] >= a.max_steps):
                break
            if state["datasets"] - int(state.get("last_epoch_datasets", 0)) >= max(1, a.save_every_datasets):
                state["last_epoch_datasets"] = int(state["datasets"])
                save_epoch(int(state["epochs_saved"]) + 1)

            frac = min(1.0, state["train_s"] / budget_s)
            lr_mult = (frac / 0.04) if frac < 0.04 else (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * (frac - 0.04) / 0.96)))
            cur_lr = a.lr * max(0.05, lr_mult)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            item_got = None
            t_wait0 = time.time()
            while not stop_flag["stop"]:
                try:
                    item_got = ep_q.get(timeout=2.0)
                    break
                except queue.Empty:
                    if time.time() - t_wait0 > 180.0:
                        raise TimeoutError("worker queue empty > 180s")
            if stop_flag["stop"] or item_got is None:
                break
            zm_np, yz_np, ym_np, ys_np, yq_np, n_ctx, infos, fps = item_got

            keep_idx = ledger.filter_unseen_indices(fps)
            if not keep_idx:
                continue
            if len(keep_idx) < zm_np.shape[0]:
                zm_np = zm_np[keep_idx]
                yz_np = yz_np[keep_idx]
                ym_np = ym_np[keep_idx]
                ys_np = ys_np[keep_idx]
                yq_np = yq_np[keep_idx]
                if infos is not None:
                    infos = [infos[i] for i in keep_idx]

            B_cur, N_cur, Fn_cur, _ = zm_np.shape
            if dev.type == "cuda":
                zm_t = torch.from_numpy(zm_np).pin_memory().to(dev, non_blocking=True)
                yz_t = torch.from_numpy(yz_np).pin_memory().to(dev, non_blocking=True)
                ym_t = torch.from_numpy(ym_np).pin_memory().to(dev, non_blocking=True)
                ys_t = torch.from_numpy(ys_np).pin_memory().to(dev, non_blocking=True)
                yq_t = torch.from_numpy(yq_np).pin_memory().to(dev, non_blocking=True)
            else:
                zm_t = torch.from_numpy(zm_np)
                yz_t = torch.from_numpy(yz_np)
                ym_t = torch.from_numpy(ym_np)
                ys_t = torch.from_numpy(ys_np)
                yq_t = torch.from_numpy(yq_np)

            # Independent random feature-slot permutation per episode [B_cur, 1, Fn_cur]
            slots = torch.rand(B_cur, FID_SLOTS, device=dev).argsort(dim=-1)[:, None, :Fn_cur]
            try:
                logits, ym, ys_ = model(
                    None, None, n_ctx, slots=slots,
                    amp=(dev.type in ("cuda", "mps")),
                    prenorm=(zm_t, yz_t, ym_t, ys_t),
                )
                loss = model.loss(logits, yq_t, ym, ys_).mean()
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
                state["cell_tokens"] += ledger.add_synth(B_cur, N_cur, Fn_cur)
                state["synth_batches"] += 1
            else:
                state["cell_tokens"] += ledger.add_real(infos)
                state["real_batches"] += 1
            state["micro_batches"] += 1
            state["accum_in_group"] += 1
            state["datasets"] += B_cur
            run_loss += float(loss.detach().cpu())
            run_n += 1
            run_ds += B_cur

            if state["accum_in_group"] >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                steps_since_start += 1
                state["accum_in_group"] = 0
                with torch.no_grad():
                    decay = min(0.995, (1.0 + state["step"]) / (10.0 + state["step"]))
                    swa_d = min(swa_decay_target, (1.0 + state["step"]) / (20.0 + state["step"]))
                    for pe, ps, pm in zip(ema.parameters(), ema_swa.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - decay)
                        ps.lerp_(pm, 1.0 - swa_d)
                if dev.type == "cuda" and a.cuda_vram_frac > 0.0 and steps_since_start == 3:
                    try:
                        free_b, tot_b = torch.cuda.mem_get_info(dev)
                        used_b = tot_b - free_b
                        tgt_b = int(a.cuda_vram_frac * tot_b)
                        if used_b < tgt_b - (64 << 20):
                            _pool_warm = torch.empty(tgt_b - used_b, dtype=torch.uint8, device=dev)
                            del _pool_warm
                    except Exception:
                        pass
                if steps_since_start == 10:
                    save_last()
                    save_latest_eval_ckpt()
                    t_last = time.time()

            now = time.time()
            if now - t_log >= 30.0 and run_n > 0:
                host_cpu_pct = gov.sample_pct()
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
                    "tmodes": s["target_modes"], "ds_mix": a.dataset_mix,
                    "host_cpu_pct": round(host_cpu_pct, 1), "oom": state["oom_skips"], "mem_gb": cuda_gb,
                }) + "\n")
                run_loss, run_n, run_ds, t_log = 0.0, 0, 0, now
            if now - t_last >= a.last_every_s:
                save_last()
                save_latest_eval_ckpt()
                t_last = now
    finally:
        stop_ev.set()
        try:
            ep_q.cancel_join_thread()
            ep_q.close()
        except Exception:
            pass
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=2.0)

    state["train_s"] = s_seg0 + (time.time() - t_seg)
    save_last()
    save_latest_eval_ckpt()
    if stop_flag["stop"]:
        log_f.write(json.dumps({"paused": time.strftime("%FT%TZ", time.gmtime()), "step": state["step"],
                                "datasets": state["datasets"], "train_h": round(state["train_s"] / 3600.0, 4)}) + "\n")
        log_f.close()
        return
    if state["datasets"] > int(state.get("last_epoch_datasets", 0)):
        state["last_epoch_datasets"] = int(state["datasets"])
        save_epoch(int(state["epochs_saved"]) + 1)
    swa_cpu = {kk: vv.cpu() for kk, vv in ema_swa.state_dict().items()}
    mod_cpu = {kk: vv.cpu() for kk, vv in model.state_dict().items()}
    _atomic_save({"model": mod_cpu, "ema": swa_cpu, "state": state, "size": a.size}, final_p, mode=0o444)
    manifest = {
        "schema": "jope.h450.ckpt_manifest/v4", "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True, "size": a.size, "params": params_cnt, "device": str(dev),
        "host": platform.node(), "seed": a.seed, "init_ckpt": str(a.init), "init_sha256": a.init_sha256,
        "bank": str(a.mmap_bank), "bank_sha256": bank.sha256, "bank_tables": bank.n_tables,
        "ckpt_final_path": str(final_p), "ckpt_final_sha256": sha256_file(final_p),
        "sampler_cfg": sampler_cfg, "state": state, "exposure": ledger.summary(),
    }
    man_p = out_dir / "manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(man_p, 0o444)
    log_f.close()
    print(json.dumps({k: manifest[k] for k in ("ckpt_final_sha256", "state")}, sort_keys=True))


if __name__ == "__main__":
    main()
