#!/usr/bin/env python3
"""Wave H455 `104M` (`ICL_L`) Maximum-Entropy S3-Source-Uniform + Shape-Binned Exact-GP Flywheel Trainer
(`train_104m_gp_flywheel_h455_maxent.py`).

Key H455 Upgrades over H454 (`canonical_promotion_forbidden = true`, `0` `CONFIRM`, `0` `FMGAP B/C`):
  1. Uniform Mixing Across All `2,389` S3/Corpus Source Families (`--dataset-mix s3_source_uniform`):
     - Eliminates the `36x` multi-cousin family skew and removes the `45%` (`20% d91 + 25% regret`) routing tilt
       (`95%` S3-source-uniform + `5%` gentle regret-tilt; teacher shards also inverse-sqrt family-balanced).
  2. Wider Per-Batch Random Feature Drop (`[0.00, 0.85]`) & 22-Family Zero-Leakage Feature Injection (`[0.00, 0.60]`):
     - `15%` full-feature clean view (`drop = 0.00`) + `85%` continuous `Uniform(0.05, 0.85)` random feature drop.
     - 22 context-only (`Xc, yc`) feature generators (`0` query leakage!), adding nested OOF `CW-1` discrete keys,
       `CW-2` protected ratios/log-modulus products, Random Fourier Features, LOO context-kNN summaries, and
       `20%` pure structured distractors (correlated Gaussian, Student-t3, row-permuted real cols, random steps).
     - Teacher-Clean / Student-Cluttered Exact-GP Distillation on `exact_kd` (`25%`) and `exact_resid` (`15%`).
  3. True Per-Batch Random Train/Test (`Context/Query`) Splits (`p_ctx in [0.08, 0.95]`) Across All 4 Streams:
     - 3-regime split mixture (`25% U(0.08,0.45)` + `35% U(0.45,0.78)` + `40% U(0.78,0.95)`), including dynamic
       disjoint `(Xc, Xq)` re-sampling on Exact-GP teacher shards (`0.0` query leakage).
  4. Cross-Shard Shape-Grid Batching (`B = 8..24` Distinct S3 Sources/Batch Instead of `B = 1`):
     - Quantizes `(N_bin, F_bin, n_ctx_bin)` across shards so `exact_kd` and `exact_resid` pack `8..24` distinct
       tables per GPU forward pass, eliminating `B=1` micro-batch GPU starvation.
  5. Endless Smooth Warm-Floor Cosine Valley Cycling (`--endless 1`, Never Stop Trainers at a Plateau):
     - Cycle 0 glides `lr -> lr_min` (`2.5e-6`), then continuously cycles `lr_min -> lr_cycle_peak -> lr_min` via a
       C1-continuous raised cosine (`0.5 * (1 - cos(2*pi*u))`), saving `ckpt_valley_NNN.pt` at every valley.
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
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if sys.path[:1] != [str(HERE)]:
    sys.path.insert(0, str(HERE))
REF_CODE = os.environ.get("JOPE_REF_CODE", "")
for _cand in (
    REF_CODE,
    "/home/ubuntu/jope-data/h450-fleet-agy/stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    str(Path.home() / "jope-data/h450-fleet-agy/fleet"),
    str(Path.home() / "jope-data/h450-fleet-agy/ref_code"),
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

from mmap_bank_h450 import SEALED_TOKENS, MmapF64Bank, sha256_file  # noqa: E402
import train_mps_104m_h450 as BASE  # noqa: E402
from icl_model import CLIP_X, FID_SLOTS  # noqa: E402

Prior = BASE.Prior
from train_104m_gp_flywheel_h452 import (  # noqa: E402
    ICLH450MPS,
    apply_monotone_warps_and_nans_zero_leakage,
    inject_synth_cols_18fam_zero_leakage,
    nonlinear_gp_oof_residual_np,
    sample_extreme_stratified_episode_h452,
)
from train_104m_gp_flywheel_h454 import (  # noqa: E402
    HWMGPWinnerStoreV3,
    build_gp_stage_cols_zero_leakage_v2,
)

N_GRID_H455 = (96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
F_GRID_H455 = (8, 12, 16, 24, 32, 48, 64)
TEACHER_N_GRID_H455 = (96, 128, 192, 256, 384, 512, 768)
TEACHER_F_GRID_H455 = (12, 16, 24, 32, 48, 64)


def draw_p_ctx_h455(rng: np.random.Generator, p_min: float = 0.08, p_max: float = 0.95) -> float:
    """3-regime per-batch train/test split mixture spanning `[0.08, 0.95]` (`40%` CV-dense `[0.78, 0.95]`)."""
    u = float(rng.random())
    if u < 0.25:
        lo, hi = p_min, min(0.45, p_max)
    elif u < 0.60:
        lo, hi = max(p_min, 0.45), min(0.78, p_max)
    else:
        lo, hi = max(p_min, 0.78), p_max
    if hi <= lo:
        return float(0.5 * (p_min + p_max))
    return float(rng.uniform(lo, hi))


def choose_drop_keep_h455(
    p: int,
    rng: np.random.Generator,
    drop_min_frac: float = 0.00,
    drop_max_frac: float = 0.85,
    full_view_prob: float = 0.15,
    min_keep: int = 2,
) -> int:
    """Wider `[0.00, 0.85]` feature drop: `15%` full-feature clean view + `85%` continuous subset projection."""
    if p <= min_keep:
        return p
    if drop_min_frac <= 1e-9 and float(rng.random()) < full_view_prob:
        return min(p, BASE.WIDE_GPU)
    lo = max(0.03, float(drop_min_frac))
    hi = max(lo, min(0.90, float(drop_max_frac)))
    df = float(rng.uniform(lo, hi))
    k_keep = int(round((1.0 - df) * p))
    return max(min(min_keep, p), min(p, BASE.WIDE_GPU, k_keep))


def inject_synth_cols_22fam_h455(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    k_want: int,
    rng: np.random.Generator,
    distractor_prob: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray]:
    """22-Family Zero-Leakage Feature Injector fitted STRICTLY on `(Xc, yc)` (`0.0` `Xq`/`yq` leakage!).

    Combines:
      - `45%`: 14+4 nonlinear/interaction/rank/Chebyshev/hinge/RFF families (`inject_synth_cols_18fam_zero_leakage`),
      - `33%`: 4 structured tabular builders (`CW-1` nested OOF discrete keys, `CW-2` protected ratios & log-modulus
               products, multi-scale Random Fourier Features, and LOO context-kNN distance/target summaries),
      - `22%`: 4 pure distractor generators independent of `y` (AR(1) Gaussian block, Student-t3 heavy-tail noise,
               row-permuted real feature copies, and random step indicators) so `104M` learns sharp clutter rejection.
    """
    n_c, p = Xc.shape
    n_q = Xq.shape[0]
    if k_want <= 0 or p <= 0 or n_c < 8:
        return np.zeros((n_c, 0), dtype=np.float64), np.zeros((n_q, 0), dtype=np.float64)

    Zc, Zq = BASE._std_ctx_f64(Xc, Xq)
    yc_f = np.asarray(yc, dtype=np.float64)
    ym_c = float(np.mean(yc_f)) if np.all(np.isfinite(yc_f)) else 0.0
    sy_c = max(float(np.std(yc_f)), 1e-9) if np.all(np.isfinite(yc_f)) else 1.0
    yc_z = (yc_f - ym_c) / sy_c if np.all(np.isfinite(yc_f)) else np.zeros(n_c, dtype=np.float64)

    n_dist = int(rng.binomial(k_want, min(0.50, max(0.0, float(distractor_prob)))))
    n_rem = k_want - n_dist
    n_struct = int(rng.binomial(n_rem, 0.45)) if n_rem > 0 else 0
    n_base = n_rem - n_struct

    cols_c: List[np.ndarray] = []
    cols_q: List[np.ndarray] = []

    if n_base > 0:
        Bc, Bq = inject_synth_cols_18fam_zero_leakage(Xc, Xq, n_base, rng)
        if Bc.shape[1] > 0:
            cols_c.append(Bc)
            cols_q.append(Bq)

    if n_struct > 0:
        Sc = np.zeros((n_c, n_struct), dtype=np.float64)
        Sq = np.zeros((n_q, n_struct), dtype=np.float64)
        for m in range(n_struct):
            sfam = int(rng.integers(0, 4))
            ja = int(rng.integers(0, p))
            jb = int(rng.integers(0, p))
            if p >= 2 and jb == ja:
                jb = (ja + 1) % p
            if sfam == 0 and n_c >= 20:
                # Family 14: Nested 5-fold OOF CW-1 key / quantile-bin target encoding (strictly on Xc, yc!)
                col_c = Zc[:, ja]
                col_q = Zq[:, ja]
                uv = np.unique(np.round(col_c, 5))
                if 2 <= uv.size <= max(16, n_c // 3):
                    kc_id = np.searchsorted(uv, np.round(col_c, 5))
                    kq_id = np.searchsorted(uv, np.round(col_q, 5))
                    in_sup = (kq_id < uv.size) & (np.abs(uv[np.minimum(kq_id, uv.size - 1)] - np.round(col_q, 5)) <= 1e-4)
                    n_bins = uv.size
                else:
                    n_bins = min(12, max(4, n_c // 10))
                    edges = np.percentile(col_c, np.linspace(10.0, 90.0, n_bins - 1))
                    kc_id = np.searchsorted(edges, col_c)
                    kq_id = np.searchsorted(edges, col_q)
                    in_sup = np.ones(n_q, dtype=bool)
                oof_te = np.zeros(n_c, dtype=np.float64)
                f_assign = rng.permutation(n_c) % 5
                prior_m = 5.0
                for fk in range(5):
                    va = f_assign == fk
                    tr = ~va
                    cnts = np.bincount(kc_id[tr], minlength=n_bins).astype(np.float64)
                    sums = np.bincount(kc_id[tr], weights=yc_z[tr], minlength=n_bins).astype(np.float64)
                    means = sums / np.maximum(cnts + prior_m, 1.0)
                    oof_te[va] = means[kc_id[va]]
                cnts_all = np.bincount(kc_id, minlength=n_bins).astype(np.float64)
                sums_all = np.bincount(kc_id, weights=yc_z, minlength=n_bins).astype(np.float64)
                means_all = sums_all / np.maximum(cnts_all + prior_m, 1.0)
                Sc[:, m] = oof_te
                Sq[:, m] = np.where(in_sup, means_all[np.minimum(kq_id, n_bins - 1)], 0.0)
            elif sfam == 1:
                # Family 15: CW-2 log-modulus product or protected difference-over-sum
                if float(rng.random()) < 0.5:
                    pr_c = Zc[:, ja] * Zc[:, jb]
                    pr_q = Zq[:, ja] * Zq[:, jb]
                    Sc[:, m] = np.sign(pr_c) * np.log1p(np.abs(pr_c))
                    Sq[:, m] = np.sign(pr_q) * np.log1p(np.abs(pr_q))
                else:
                    den_c = np.abs(Zc[:, ja]) + np.abs(Zc[:, jb]) + 0.5
                    den_q = np.abs(Zq[:, ja]) + np.abs(Zq[:, jb]) + 0.5
                    Sc[:, m] = (Zc[:, ja] - Zc[:, jb]) / den_c
                    Sq[:, m] = (Zq[:, ja] - Zq[:, jb]) / den_q
            elif sfam == 2:
                # Family 16: Multi-scale Random Fourier Feature projection
                sub_k = min(p, int(rng.integers(2, min(10, p) + 1))) if p >= 2 else 1
                sub = rng.choice(p, size=sub_k, replace=False)
                ls = float(math.exp(rng.uniform(-0.8, 1.2))) * math.sqrt(float(sub_k))
                w = rng.standard_normal(sub_k) / ls
                ph = float(rng.uniform(0.0, 2.0 * math.pi))
                Sc[:, m] = np.cos(Zc[:, sub] @ w + ph)
                Sq[:, m] = np.cos(Zq[:, sub] @ w + ph)
            else:
                # Family 17: Leave-One-Out Context k-NN target/distance summary (strictly on Zc, yc_z!)
                sub_k = min(p, 6)
                sub = rng.choice(p, size=sub_k, replace=False)
                # Cap reference anchors at 192 context rows so kNN stays O(N) fast
                if n_c > 192:
                    ref_idx = rng.choice(n_c, size=192, replace=False)
                else:
                    ref_idx = np.arange(n_c, dtype=np.int64)
                Z_ref = Zc[ref_idx][:, sub]
                y_ref = yc_z[ref_idx]
                dc2 = np.sum((Zc[:, None, sub] - Z_ref[None, :, :]) ** 2, axis=-1)
                # Mask exact self-matches on context rows (`0` self-label leakage!)
                for r_pos, orig_i in enumerate(ref_idx.tolist()):
                    dc2[orig_i, r_pos] = np.inf
                dq2 = np.sum((Zq[:, None, sub] - Z_ref[None, :, :]) ** 2, axis=-1) if n_q > 0 else np.zeros((0, len(ref_idx)))
                k_nn = min(5, max(1, len(ref_idx) - 1))
                nnc = np.argpartition(dc2, kth=k_nn - 1, axis=1)[:, :k_nn]
                wc = 1.0 / np.maximum(np.take_along_axis(dc2, nnc, axis=1) + 0.25, 1e-6)
                wc /= np.sum(wc, axis=1, keepdims=True)
                Sc[:, m] = np.sum(wc * y_ref[nnc], axis=1)
                if n_q > 0:
                    nnq = np.argpartition(dq2, kth=k_nn - 1, axis=1)[:, :k_nn]
                    wq_k = 1.0 / np.maximum(np.take_along_axis(dq2, nnq, axis=1) + 0.25, 1e-6)
                    wq_k /= np.sum(wq_k, axis=1, keepdims=True)
                    Sq[:, m] = np.sum(wq_k * y_ref[nnq], axis=1)
        cols_c.append(Sc)
        cols_q.append(Sq)

    if n_dist > 0:
        # Pure structured distractor columns independent of `y` (`0` query stat dependence: `rng` draws `c` FIRST, then `q`!)
        Dc = np.zeros((n_c, n_dist), dtype=np.float64)
        Dq = np.zeros((n_q, n_dist), dtype=np.float64)
        # Draw a dedicated distractor child seed from `(n_c, p, rng)` BEFORE touching `n_q` so `Dc` is 100% invariant to `n_q`!
        seed_c = int(rng.integers(0, 2**31 - 1))
        seed_q = int(rng.integers(0, 2**31 - 1))
        rng_dc = np.random.default_rng(seed_c)
        rng_dq = np.random.default_rng(seed_q)
        for m in range(n_dist):
            dfam = int(rng_dc.integers(0, 4))
            j_src = int(rng_dc.integers(0, p))
            if dfam == 0:
                # Correlated Gaussian distractor
                rho = float(rng_dc.uniform(-0.6, 0.6))
                Dc[:, m] = rho * Zc[rng_dc.permutation(n_c), j_src] + math.sqrt(max(0.04, 1.0 - rho * rho)) * rng_dc.standard_normal(n_c)
                if n_q > 0:
                    Dq[:, m] = rho * Zc[rng_dq.integers(0, n_c, size=n_q), j_src] + math.sqrt(max(0.04, 1.0 - rho * rho)) * rng_dq.standard_normal(n_q)
            elif dfam == 1:
                # Clipped Student-t3 heavy-tailed pure noise
                chi_c = np.maximum(np.mean(rng_dc.standard_normal((n_c, 3)) ** 2, axis=1), 0.05)
                Dc[:, m] = np.clip(rng_dc.standard_normal(n_c) / np.sqrt(chi_c), -12.0, 12.0)
                if n_q > 0:
                    chi_q = np.maximum(np.mean(rng_dq.standard_normal((n_q, 3)) ** 2, axis=1), 0.05)
                    Dq[:, m] = np.clip(rng_dq.standard_normal(n_q) / np.sqrt(chi_q), -12.0, 12.0)
            elif dfam == 2:
                # Row-permuted copy of a real context column (exact marginal distribution, 0 target mutual info!)
                Dc[:, m] = Zc[rng_dc.permutation(n_c), j_src]
                if n_q > 0:
                    Dq[:, m] = Zc[rng_dq.integers(0, n_c, size=n_q), j_src]
            else:
                # Random step indicator independent of y
                thr = float(rng_dc.uniform(-1.2, 1.2))
                Dc[:, m] = (rng_dc.standard_normal(n_c) > thr).astype(np.float64)
                if n_q > 0:
                    Dq[:, m] = (rng_dq.standard_normal(n_q) > thr).astype(np.float64)
        cols_c.append(Dc)
        cols_q.append(Dq)

    Mc = np.hstack(cols_c) if cols_c else np.zeros((n_c, 0), dtype=np.float64)
    Mq = np.hstack(cols_q) if cols_q else np.zeros((n_q, 0), dtype=np.float64)
    if Mc.shape[1] < k_want:
        Rc, Rq = inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want - Mc.shape[1], rng)
        Mc = np.hstack([Mc, Rc])
        Mq = np.hstack([Mq, Rq])
    return Mc[:, :k_want], Mq[:, :k_want]


def fit_cols_to_fbin_h455(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    f_bin: int,
    rng: np.random.Generator,
    pin_slot0: bool = False,
    distractor_prob: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Adjusts `(Xc, Xq)` width to exact `f_bin` (`0` query leakage) and randomly permutes columns."""
    p_cur = Xc.shape[1]
    k_inj = 0
    if p_cur > f_bin:
        if pin_slot0 and f_bin >= 2:
            s_c, s_q = BASE._select_cols(Xc[:, 1:], yc, Xq[:, 1:], f_bin - 1, rng)
            Xc = np.column_stack([Xc[:, 0], s_c])
            Xq = np.column_stack([Xq[:, 0], s_q])
        else:
            Xc, Xq = BASE._select_cols(Xc, yc, Xq, f_bin, rng)
    elif p_cur < f_bin:
        k_inj = f_bin - p_cur
        Ic, Iq = inject_synth_cols_22fam_h455(Xc, yc, Xq, k_inj, rng, distractor_prob=distractor_prob)
        Xc = np.hstack([Xc, Ic])
        Xq = np.hstack([Xq, Iq])
    perm = rng.permutation(f_bin)
    return Xc[:, perm], Xq[:, perm], k_inj


def _resample_rows_disjoint(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_want: int,
    rng: np.random.Generator,
    extra_arr: Optional[np.ndarray] = None,
    stratify_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Draws `n_want` rows from `(X_pool, y_pool)` without touching any other split pool (`0` cross-pool leakage!)."""
    n_src, p = X_pool.shape
    if n_want <= n_src:
        if stratify_y and n_want >= 12 and n_src >= 24 and np.all(np.isfinite(y_pool)):
            ord_y = np.argsort(y_pool + 1e-12 * rng.standard_normal(n_src), kind="mergesort")
            n_b = min(8, max(2, n_want // 4))
            parts = []
            for blk in np.array_split(ord_y, n_b):
                k_b = max(1, int(round(len(blk) * (n_want / float(n_src)))))
                parts.append(rng.permutation(blk)[:k_b])
            idx = np.concatenate(parts)
            if len(idx) < n_want:
                rem = np.setdiff1d(np.arange(n_src, dtype=np.int64), idx)
                idx = np.concatenate([idx, rng.permutation(rem)[: n_want - len(idx)]])
            idx = rng.permutation(idx[:n_want])
        else:
            idx = rng.choice(n_src, size=n_want, replace=False)
        Xo = X_pool[idx].copy()
        yo = y_pool[idx].copy()
        eo = extra_arr[idx].copy() if extra_arr is not None else None
        return Xo, yo, eo

    # Bootstrap pad strictly inside the same pool with tiny feature jitter (`0.015 * pool_std`) so rows are distinct
    base_idx = np.arange(n_src, dtype=np.int64)
    extra_idx = rng.integers(0, n_src, size=n_want - n_src)
    idx = rng.permutation(np.concatenate([base_idx, extra_idx]))
    Xo = X_pool[idx].copy()
    yo = y_pool[idx].copy()
    eo = extra_arr[idx].copy() if extra_arr is not None else None
    for j in range(p):
        fin = np.isfinite(X_pool[:, j])
        if int(np.sum(fin)) >= 6:
            vals = X_pool[fin, j]
            if np.unique(vals).size > 12:
                sd = max(float(np.std(vals)), 1e-9)
                f_mask = np.isfinite(Xo[:, j])
                Xo[f_mask, j] += 0.015 * sd * rng.standard_normal(int(np.sum(f_mask)))
    return Xo, yo, eo


class ExactGPTeacherShardStoreH455:
    """Hot-reloading store of verified Rust GP Teacher Shards with S3-family-balanced sampling & quarantine check."""

    def __init__(
        self,
        shards_dir: str,
        bank: MmapF64Bank,
        allow_d91: bool = True,
        min_regret: float = 0.010,
        gamma: float = 0.65,
        reload_every_s: float = 45.0,
    ) -> None:
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.bank = bank
        self.allow_d91 = bool(allow_d91)
        self.min_regret = float(min_regret)
        self.gamma = float(gamma)
        self.reload_every_s = float(reload_every_s)
        self._last_check = 0.0
        self._loaded_paths: set[str] = set()
        self.shards: List[dict] = []
        self.probs: Optional[np.ndarray] = None
        self.maybe_reload(force=True)

    def maybe_reload(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_check) < self.reload_every_s:
            return
        self._last_check = now
        if self.shards_dir is None or not self.shards_dir.is_dir():
            return
        added = 0
        try:
            for p in sorted(self.shards_dir.glob("shard_*.npz")):
                ps = str(p)
                if ps in self._loaded_paths:
                    continue
                try:
                    with np.load(p, allow_pickle=False) as z:
                        fam_s = str(z["family_id"]) if "family_id" in z.files else ""
                        if any(tok in fam_s.upper() for tok in SEALED_TOKENS):
                            raise SystemExit(f"QUARANTINE REFUSAL in teacher shard {p}: {fam_s}")
                        is_d91 = bool(z["is_dev91_table"])
                        if is_d91 and not self.allow_d91:
                            self._loaded_paths.add(ps)
                            continue
                        regret = float(z["regret_vs_base"])
                        if not math.isfinite(regret) or regret < self.min_regret:
                            self._loaded_paths.add(ps)
                            continue
                        t_idx = int(z["table_idx"])
                        if not (0 <= t_idx < self.bank.n_tables):
                            self._loaded_paths.add(ps)
                            continue
                        Xc = np.asarray(z["Xc"], dtype=np.float32)
                        yc = np.asarray(z["yc"], dtype=np.float64)
                        oof_c = np.asarray(z["oof_gp_c"], dtype=np.float64)
                        Xqr = np.asarray(z["Xq_real"], dtype=np.float32)
                        yqr = np.asarray(z["yq_real"], dtype=np.float64)
                        pqr = np.asarray(z["pred_gp_q_real"], dtype=np.float64)
                        Xcf = np.asarray(z["Xq_cf"], dtype=np.float32)
                        pcf = np.asarray(z["pred_gp_q_cf"], dtype=np.float64)
                        if len(yc) < 20 or len(yqr) < 6 or Xc.shape[1] < 1:
                            self._loaded_paths.add(ps)
                            continue
                        self.shards.append(
                            {
                                "Xc": Xc,
                                "yc": yc,
                                "oof_c": oof_c,
                                "Xqr": Xqr,
                                "yqr": yqr,
                                "pqr": pqr,
                                "Xcf": Xcf,
                                "pcf": pcf,
                                "regret": regret,
                                "table_idx": t_idx,
                                "family_idx": int(self.bank.family_idx[t_idx]),
                                "is_d91": is_d91,
                            }
                        )
                        self._loaded_paths.add(ps)
                        added += 1
                except SystemExit:
                    raise
                except (OSError, ValueError, KeyError, RuntimeError):
                    continue
            if added > 0 and self.shards:
                if len(self.shards) > 3200:
                    by_t: Dict[int, List[dict]] = {}
                    for sh in self.shards:
                        by_t.setdefault(int(sh["table_idx"]), []).append(sh)
                    trimmed: List[dict] = []
                    for lst in by_t.values():
                        lst.sort(key=lambda s: -float(s["regret"]))
                        trimmed.extend(lst[:3])
                    self.shards = trimmed
                fam_cnt: Dict[int, int] = {}
                for sh in self.shards:
                    fi = int(sh["family_idx"])
                    fam_cnt[fi] = fam_cnt.get(fi, 0) + 1
                w = np.array(
                    [
                        (min(max(float(s["regret"]), 0.01), 1.50) ** self.gamma)
                        / float(fam_cnt[int(s["family_idx"])]) ** 0.75
                        for s in self.shards
                    ],
                    dtype=np.float64,
                )
                self.probs = w / float(np.sum(w))
        except SystemExit:
            raise
        except (OSError, ValueError, KeyError, RuntimeError):
            pass

    def sample_shard(self, rng: np.random.Generator) -> Optional[dict]:
        if not self.shards or self.probs is None:
            return None
        return self.shards[int(rng.choice(len(self.shards), p=self.probs))]


def sample_binned_exact_kd_episode_h455(
    sh: dict,
    n_bin: int,
    f_bin: int,
    n_ctx_bin: int,
    rng: np.random.Generator,
    cf_weight_scale: float = 0.65,
    drop_max_frac: float = 0.30,
    feature_noise_max: float = 0.035,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """Teacher-Clean / Student-Cluttered Exact-GP KD episode quantized to `(n_bin, f_bin, n_ctx_bin)` (`0.0` leakage!)."""
    Xc_full = np.asarray(sh["Xc"], dtype=np.float64)
    yc_full = np.asarray(sh["yc"], dtype=np.float64)
    Xqr_full = np.asarray(sh["Xqr"], dtype=np.float64)
    yqr_full = np.asarray(sh["yqr"], dtype=np.float64)
    pqr_full = np.asarray(sh["pqr"], dtype=np.float64)
    Xcf_full = np.asarray(sh["Xcf"], dtype=np.float64)
    pcf_full = np.asarray(sh["pcf"], dtype=np.float64)
    regret = float(sh["regret"])
    t_idx = int(sh["table_idx"])
    p_raw = Xc_full.shape[1]
    n_q_bin = n_bin - n_ctx_bin
    if n_ctx_bin < 12 or n_q_bin < 4 or p_raw < 1:
        return None

    # Disjoint context draw strictly from `Xc_full`
    Xc, yc, _ = _resample_rows_disjoint(Xc_full, yc_full, n_ctx_bin, rng, stratify_y=True)

    # Disjoint query draw strictly from `(Xqr_full, Xcf_full)` (`0` rows from `Xc_full`!)
    alpha = float(np.clip(math.tanh(2.0 * regret), 0.15, 0.85))
    yqr_tgt_pool = (1.0 - alpha) * yqr_full + alpha * pqr_full
    if len(pcf_full) >= 4:
        n_qr_want = max(4, int(round(0.45 * n_q_bin)))
        n_cf_want = n_q_bin - n_qr_want
        Xqr, yqr_t, _ = _resample_rows_disjoint(Xqr_full, yqr_tgt_pool, n_qr_want, rng, stratify_y=False)
        Xcf, pcf_t, _ = _resample_rows_disjoint(Xcf_full, pcf_full, n_cf_want, rng, stratify_y=False)
        Xq_all = np.vstack([Xqr, Xcf])
        yq_all = np.concatenate([yqr_t, pcf_t])
        wq_all = np.concatenate(
            [
                np.ones(n_qr_want, dtype=np.float32),
                np.full(n_cf_want, float(cf_weight_scale * alpha), dtype=np.float32),
            ]
        )
        # Shuffle query order so real & counterfactual positions interleave
        q_perm = rng.permutation(n_q_bin)
        Xq_all, yq_all, wq_all = Xq_all[q_perm], yq_all[q_perm], wq_all[q_perm]
    else:
        Xq_all, yq_all, _ = _resample_rows_disjoint(Xqr_full, yqr_tgt_pool, n_q_bin, rng, stratify_y=False)
        wq_all = np.ones(n_q_bin, dtype=np.float32)

    # Student-Cluttered View: optional `[0.0, 0.30]` feature drop + 22-fam inject/distractor up to `f_bin` + warps
    k_keep = choose_drop_keep_h455(p_raw, rng, drop_min_frac=0.0, drop_max_frac=drop_max_frac, full_view_prob=0.35, min_keep=2)
    if k_keep < p_raw:
        Xc, Xq_all = BASE._select_cols(Xc, yc, Xq_all, k_keep, rng)
    Xc, Xq_all = apply_monotone_warps_and_nans_zero_leakage(Xc, Xq_all, rng, warp_prob=0.28, nan_prob=0.18)
    Xc, Xq_all, k_inj = fit_cols_to_fbin_h455(Xc, yc, Xq_all, f_bin, rng, pin_slot0=False, distractor_prob=0.25)
    Xc, Xq_all = BASE._augment_features_zero_leakage(Xc, Xq_all, rng, noise_max_frac=feature_noise_max)

    y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    y_scale = float(math.exp(rng.uniform(-1.8, 1.8)))
    y_ep = y_sign * y_scale * np.concatenate([yc, yq_all])
    X_ep = np.vstack([Xc, Xq_all])
    if X_ep.shape != (n_bin, f_bin) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_bin])) <= 1e-12:
        return None
    info = (t_idx, 2, int(k_inj), int(f_bin), int(n_bin), 0, 0, int(p_raw))
    return X_ep, y_ep, wq_all, info


def sample_binned_exact_resid_episode_h455(
    sh: dict,
    n_bin: int,
    f_bin: int,
    n_ctx_bin: int,
    rng: np.random.Generator,
    drop_max_frac: float = 0.30,
    feature_noise_max: float = 0.035,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """Exact Appended OOF GP-Residual episode quantized to `(n_bin, f_bin, n_ctx_bin)` (`0.0` query leakage!)."""
    Xc_full = np.asarray(sh["Xc"], dtype=np.float64)
    yc_full = np.asarray(sh["yc"], dtype=np.float64)
    oof_full = np.asarray(sh["oof_c"], dtype=np.float64)
    Xqr_full = np.asarray(sh["Xqr"], dtype=np.float64)
    yqr_full = np.asarray(sh["yqr"], dtype=np.float64)
    pqr_full = np.asarray(sh["pqr"], dtype=np.float64)
    t_idx = int(sh["table_idx"])
    p_raw = Xc_full.shape[1]
    n_q_bin = n_bin - n_ctx_bin
    if n_ctx_bin < 12 or n_q_bin < 4 or p_raw < 1 or f_bin < 4:
        return None

    rc_full = yc_full - oof_full
    rq_full = yqr_full - pqr_full
    if float(np.std(rc_full)) <= 1e-6 * max(float(np.std(yc_full)), 1.0):
        return None

    Xc, rc, oof_c = _resample_rows_disjoint(Xc_full, rc_full, n_ctx_bin, rng, extra_arr=oof_full, stratify_y=True)
    Xq, rq, pq = _resample_rows_disjoint(Xqr_full, rq_full, n_q_bin, rng, extra_arr=pqr_full, stratify_y=False)
    assert oof_c is not None and pq is not None

    k_keep = choose_drop_keep_h455(p_raw, rng, drop_min_frac=0.0, drop_max_frac=drop_max_frac, full_view_prob=0.35, min_keep=2)
    if k_keep < p_raw:
        Xc, Xq = BASE._select_cols(Xc, rc, Xq, k_keep, rng)
    Xc, Xq = apply_monotone_warps_and_nans_zero_leakage(Xc, Xq, rng, warp_prob=0.25, nan_prob=0.15)
    # Prepend GP anchor prediction at column 0 (`pin_slot0=True` prevents dropping it before permutation!)
    Xc_gp = np.column_stack([oof_c, Xc])
    Xq_gp = np.column_stack([pq, Xq])
    Xc_gp, Xq_gp, k_inj = fit_cols_to_fbin_h455(Xc_gp, rc, Xq_gp, f_bin, rng, pin_slot0=True, distractor_prob=0.22)
    Xc_gp, Xq_gp = BASE._augment_features_zero_leakage(Xc_gp, Xq_gp, rng, noise_max_frac=feature_noise_max)

    y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    y_scale = float(math.exp(rng.uniform(-1.8, 1.8)))
    y_ep = y_sign * y_scale * np.concatenate([rc, rq])
    X_ep = np.vstack([Xc_gp, Xq_gp])
    wq = np.ones(n_q_bin, dtype=np.float32)
    if X_ep.shape != (n_bin, f_bin) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_bin])) <= 1e-12:
        return None
    info = (t_idx, 0, int(k_inj), int(f_bin), int(n_bin), 0, 0, int(p_raw))
    return X_ep, y_ep, wq, info


def sample_binned_s3_maxent_episode_h455(
    bank: MmapF64Bank,
    t_idx: int,
    n_bin: int,
    f_bin: int,
    n_ctx_bin: int,
    rng: np.random.Generator,
    valid_tcols_cache: Dict[int, Tuple[int, ...]],
    win_t: Optional[dict],
    drop_min_frac: float = 0.00,
    drop_max_frac: float = 0.85,
    inj_min_frac: float = 0.00,
    inj_max_frac: float = 0.60,
    rand_target_prob: float = 0.38,
    synth_teacher_prob: float = 0.18,
    feature_noise_max: float = 0.06,
    gp_stage_prob: float = 0.45,
    gp_resid_prob: float = 0.20,
    distractor_prob: float = 0.22,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """Max-entropy S3 episode quantized to `(n_bin, f_bin, n_ctx_bin)` with `[0.00, 0.85]` drop & `[0.00, 0.60]` inject."""
    # Draw base raw episode with wider drop (`[0.00, 0.85]`) and fresh stratified/tilted crop
    eff_drop_min = 0.0 if (drop_min_frac <= 1e-9 and float(rng.random()) < 0.15) else max(0.04, drop_min_frac)
    eff_drop_max = 0.0 if eff_drop_min == 0.0 else max(eff_drop_min, drop_max_frac)
    eff_inj_min = 0.0 if (inj_min_frac <= 1e-9 and float(rng.random()) < 0.15) else max(0.05, inj_min_frac)
    eff_inj_max = 0.0 if eff_inj_min == 0.0 else max(eff_inj_min, min(0.60, inj_max_frac))

    got = sample_extreme_stratified_episode_h452(
        bank,
        t_idx,
        rng,
        valid_tcols_cache,
        winner=win_t,
        force_fractional_crop=(float(rng.random()) < 0.80),
        max_episode_rows=max(n_bin, 256),
        drop_min_frac=eff_drop_min,
        drop_max_frac=eff_drop_max,
        inj_min_frac=0.0,
        inj_max_frac=0.0,
        rand_target_prob=rand_target_prob,
        synth_teacher_prob=synth_teacher_prob,
        feature_noise_max=feature_noise_max,
        force_canonical_target=False,
        n_ctx_for_shape=None,
        preferred_rows=n_bin,
    )
    if got is None:
        return None

    X_raw, y_raw, n_ctx_raw, info_raw = got
    Xc_src, yc_src = X_raw[:n_ctx_raw], y_raw[:n_ctx_raw]
    Xq_src, yq_src = X_raw[n_ctx_raw:], y_raw[n_ctx_raw:]
    n_q_bin = n_bin - n_ctx_bin
    if len(yc_src) < 12 or len(yq_src) < 4 or n_q_bin < 4:
        return None

    # Re-align disjoint `(Xc_src, yc_src)` and `(Xq_src, yq_src)` to exact `(n_ctx_bin, n_q_bin)` (`0` cross-leakage!)
    Xc, yc, _ = _resample_rows_disjoint(Xc_src, yc_src, n_ctx_bin, rng, stratify_y=True)
    Xq, yq, _ = _resample_rows_disjoint(Xq_src, yq_src, n_q_bin, rng, stratify_y=False)

    # Target width after `[0.00, 0.60]` synthetic + GP-stage injection, quantized cleanly to `f_bin`
    p_cur = Xc.shape[1]
    inj_frac = float(rng.uniform(eff_inj_min, eff_inj_max)) if eff_inj_max > 0.0 else 0.0
    want_inj = int(round(inj_frac * max(p_cur, 4)))
    if want_inj > 0 and p_cur > f_bin - want_inj and f_bin >= 6:
        keep_real = max(2, f_bin - min(want_inj, f_bin // 2))
        if keep_real < p_cur:
            Xc, Xq = BASE._select_cols(Xc, yc, Xq, keep_real, rng)
            p_cur = Xc.shape[1]

    pin0 = False
    if win_t is not None and Xc.shape[1] >= 2 and Xc.shape[1] < f_bin and float(rng.random()) < gp_stage_prob:
        k_gp = max(1, min(f_bin - Xc.shape[1], 6))
        Gc, Gq = build_gp_stage_cols_zero_leakage_v2(Xc, yc, Xq, win_t["stages"], k_gp, rng)
        if Gc.shape[1] > 0 and np.all(np.isfinite(Gc)) and np.all(np.isfinite(Gq)):
            Xc = np.hstack([Xc, Gc])
            Xq = np.hstack([Xq, Gq])

    if Xc.shape[1] >= 2 and Xc.shape[1] < f_bin and float(rng.random()) < gp_resid_prob:
        head_b = win_t["head"] if win_t is not None else "oblivious"
        oof_gp, pq_gp, ok_gp = nonlinear_gp_oof_residual_np(Xc, yc, Xq, head_b, rng)
        if ok_gp:
            Xc = np.column_stack([oof_gp, Xc])
            Xq = np.column_stack([pq_gp, Xq])
            yc = yc - oof_gp
            yq = yq - pq_gp
            pin0 = True

    Xc, Xq, k_inj = fit_cols_to_fbin_h455(Xc, yc, Xq, f_bin, rng, pin_slot0=pin0, distractor_prob=distractor_prob)
    X_ep = np.vstack([Xc, Xq])
    y_ep = np.concatenate([yc, yq])
    wq = np.ones(n_q_bin, dtype=np.float32)
    if X_ep.shape != (n_bin, f_bin) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_bin])) <= 1e-12:
        return None
    tm_safe = int(info_raw[1]) if 0 <= int(info_raw[1]) <= 2 else 0
    sp_safe = int(info_raw[5]) if 0 <= int(info_raw[5]) <= 2 else 0
    info = (int(t_idx), tm_safe, int(k_inj), int(f_bin), int(n_bin), sp_safe, int(info_raw[6]), int(info_raw[7]))
    return X_ep, y_ep, wq, info


def compute_cyclic_valley_lr(
    train_s: float,
    start_s: float,
    lr_init: float,
    lr_cycle_peak: float,
    lr_min: float,
    cycle0_hours: float,
    cycle_hours: float,
) -> Tuple[float, int, float]:
    """C1-smooth cyclic warm-floor cosine schedule (`never stops at a plateau!`).

    Returns `(cur_lr, cycle_idx, cycle_u)`:
      - Cycle 0 (`dt < cycle0_s`): `lr_init -> lr_min` via `0.5 * (1 + cos(pi * u))`, landing smoothly at `lr_min`.
      - Cycle `k >= 1`: C1 raised cosine `lr_min + (lr_cycle_peak - lr_min) * 0.5 * (1 - cos(2 * pi * u))`
        starting at `lr_min` (`u=0`), peaking gently at `lr_cycle_peak` (`u=0.5`), and returning smoothly to
        `lr_min` at `u=1.0` without any discontinuous LR shock!
    """
    dt = max(0.0, float(train_s) - float(start_s))
    c0_s = max(60.0, float(cycle0_hours) * 3600.0)
    ck_s = max(60.0, float(cycle_hours) * 3600.0)
    if dt < c0_s:
        u0 = dt / c0_s
        lr = float(lr_min) + (float(lr_init) - float(lr_min)) * 0.5 * (1.0 + math.cos(math.pi * u0))
        return lr, 0, u0
    rem = dt - c0_s
    k_cyc = 1 + int(rem // ck_s)
    uk = (rem % ck_s) / ck_s
    lr = float(lr_min) + (float(lr_cycle_peak) - float(lr_min)) * 0.5 * (1.0 - math.cos(2.0 * math.pi * uk))
    return lr, k_cyc, uk


def _pack_worker_item_h455(
    xa: np.ndarray,
    ya: np.ndarray,
    wq: np.ndarray,
    n_ctx: int,
    infos,
    fps: List[bytes],
    rng: np.random.Generator,
    postnorm_affine_max: float = 0.26,
    batch_kind: str = "real_s3",
):
    zm_np, yz_np, ym_np, ys_np = ICLH450MPS.prenorm_cpu_np(xa, ya[:, :n_ctx], int(n_ctx))
    if postnorm_affine_max > 0.0 and float(rng.random()) < 0.85:
        B_p, _, Fn_p, _ = zm_np.shape
        g_af = np.exp(rng.uniform(-postnorm_affine_max, postnorm_affine_max, size=(B_p, 1, Fn_p))).astype(np.float32)
        d_af = rng.uniform(-0.8 * postnorm_affine_max, 0.8 * postnorm_affine_max, size=(B_p, 1, Fn_p)).astype(np.float32)
        z_val = zm_np[..., 0]
        not_miss = zm_np[..., 1] < 0.5
        zm_np[..., 0] = np.where(not_miss, np.clip(z_val * g_af + d_af, -float(CLIP_X), float(CLIP_X)), np.float32(0.0))
    yq_np = np.ascontiguousarray(ya[:, n_ctx:], dtype=np.float64)
    wq_np = np.ascontiguousarray(wq, dtype=np.float32)
    return (zm_np, yz_np, ym_np, ys_np, yq_np, wq_np, int(n_ctx), infos, fps, batch_kind)


def _flywheel_worker_loop_h455(
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
    drop_min_frac: float,
    drop_max_frac: float,
    inj_min_frac: float,
    inj_max_frac: float,
    distractor_prob: float,
    p_ctx_min: float,
    p_ctx_max: float,
    rand_target_prob: float,
    synth_teacher_prob: float,
    feature_noise_max: float,
    postnorm_affine_max: float,
    synth_prob: float,
    exact_kd_prob: float,
    exact_resid_prob: float,
    cf_weight_scale: float,
    teacher_shards_dir: str,
    allow_d91_teacher_shards: bool,
    hwm_json: str,
    gp_winners_jsonl: str,
    gp_stage_prob: float,
    gp_resid_prob: float,
    regret_tilt_prob: float,
    d91_boost_prob: float,
    prior_seed_base: int,
    out_q,
    stop_ev,
) -> None:
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.nice(12)
        torch.set_num_threads(1)
    except (OSError, ValueError, RuntimeError):
        pass

    sampling_mode = (
        "s3_source_uniform"
        if dataset_mix == "s3_source_uniform"
        else ("uniform_dataset" if dataset_mix == "uniform" else "smart_stratified")
    )
    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha, sampling_mode=sampling_mode)
    gov = BASE.HostCpuGovernor(hard_pct=cpu_hard, check_every_s=3.0)
    gp_store = HWMGPWinnerStoreV3(gp_winners_jsonl, hwm_json, bank, reload_every_s=45.0)
    shard_store = ExactGPTeacherShardStoreH455(
        teacher_shards_dir, bank, allow_d91=allow_d91_teacher_shards, min_regret=0.010, gamma=0.65, reload_every_s=45.0
    )
    prior: Optional[Prior] = None
    valid_tcols_cache: Dict[int, Tuple[int, ...]] = {}

    allowed_n_s3 = [n for n in N_GRID_H455 if n <= int(max_episode_rows)] or [96]
    allowed_n_t = [n for n in TEACHER_N_GRID_H455 if n <= int(max_episode_rows)] or [96]

    seq = 0
    while not stop_ev.is_set():
        gp_store.maybe_reload(force=False)
        shard_store.maybe_reload(force=False)
        gov.pace()
        seq += 1
        b_seed = hashlib.blake2b(
            run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32
        ).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint64))
        u_stream = float(rng.random())

        # Stream 1: Heavy Synthetic SCM / Multi-Index Prior (`synth_prob`, e.g. 10%)
        if synth_prob > 0.0 and u_stream < synth_prob:
            n_b = int(rng.choice(allowed_n_s3))
            f_b = int(rng.choice(F_GRID_H455))
            b_tgt = max(4, BASE._target_batch_size(n_b, f_b, int(bmax), tokens_cap, row_tokens_cap, attn_cap))
            p_c = draw_p_ctx_h455(rng, p_ctx_min, p_ctx_max)
            n_ctx_use = max(16, min(n_b - 4, int(round(p_c * n_b))))
            xa, ya = BASE._sample_full_synth_special(
                b_tgt,
                n_b,
                f_b,
                int(n_ctx_use),
                rng,
                feature_drop_every_batch=True,
                drop_min_frac=max(0.05, drop_min_frac),
                drop_max_frac=drop_max_frac,
                synth_inj_every_batch=True,
                inj_min_frac=max(0.05, inj_min_frac),
                inj_max_frac=inj_max_frac,
                feature_noise_max=feature_noise_max,
            )
            wq = np.ones((xa.shape[0], n_b - n_ctx_use), dtype=np.float32)
            fps = [BASE.dataset_fingerprint_128(xa[i], ya[i], int(n_ctx_use)) for i in range(xa.shape[0])]
            item = _pack_worker_item_h455(
                xa, ya, wq, int(n_ctx_use), None, fps, rng, postnorm_affine_max=postnorm_affine_max, batch_kind="synth"
            )
            while not stop_ev.is_set():
                try:
                    out_q.put(item, timeout=1.0)
                    break
                except queue.Full:
                    gov.pace()
            continue

        # Stream 2 (`exact_kd_prob`): Shape-Binned Multi-Source Exact-GP Teacher KD (`B = 8..24` distinct shards!)
        if len(shard_store.shards) >= 4 and u_stream < (synth_prob + exact_kd_prob):
            n_b = int(rng.choice(allowed_n_t))
            f_b = int(rng.choice(TEACHER_F_GRID_H455))
            p_c = draw_p_ctx_h455(rng, max(0.10, p_ctx_min), min(0.94, p_ctx_max))
            n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))
            b_tgt = max(6, BASE._target_batch_size(n_b, f_b, min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap))
            xs_l, ys_l, wqs_l, infos_l = [], [], [], []
            for _try in range(b_tgt * 3):
                if len(xs_l) >= b_tgt:
                    break
                sh_i = shard_store.sample_shard(rng)
                if sh_i is None:
                    break
                ep_kd = sample_binned_exact_kd_episode_h455(
                    sh_i,
                    n_b,
                    f_b,
                    n_ctx_b,
                    rng,
                    cf_weight_scale=cf_weight_scale,
                    drop_max_frac=0.30,
                    feature_noise_max=max(0.02, 0.55 * feature_noise_max),
                )
                if ep_kd is not None:
                    xs_l.append(ep_kd[0])
                    ys_l.append(ep_kd[1])
                    wqs_l.append(ep_kd[2])
                    infos_l.append(ep_kd[3])
            if len(xs_l) >= 2:
                fps_kd = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_l, ys_l)]
                item = _pack_worker_item_h455(
                    np.stack(xs_l),
                    np.stack(ys_l),
                    np.stack(wqs_l),
                    n_ctx_b,
                    infos_l,
                    fps_kd,
                    rng,
                    postnorm_affine_max=postnorm_affine_max,
                    batch_kind="exact_kd",
                )
                while not stop_ev.is_set():
                    try:
                        out_q.put(item, timeout=1.0)
                        break
                    except queue.Full:
                        gov.pace()
                continue

        # Stream 3 (`exact_resid_prob`): Shape-Binned Multi-Source Exact Appended OOF GP-Residual (`B = 8..24`!)
        if len(shard_store.shards) >= 4 and u_stream < (synth_prob + exact_kd_prob + exact_resid_prob):
            n_b = int(rng.choice(allowed_n_t))
            f_b = int(rng.choice(TEACHER_F_GRID_H455))
            p_c = draw_p_ctx_h455(rng, max(0.12, p_ctx_min), min(0.94, p_ctx_max))
            n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))
            b_tgt = max(6, BASE._target_batch_size(n_b, f_b, min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap))
            xs_l, ys_l, wqs_l, infos_l = [], [], [], []
            for _try in range(b_tgt * 3):
                if len(xs_l) >= b_tgt:
                    break
                sh_i = shard_store.sample_shard(rng)
                if sh_i is None:
                    break
                ep_r = sample_binned_exact_resid_episode_h455(
                    sh_i,
                    n_b,
                    f_b,
                    n_ctx_b,
                    rng,
                    drop_max_frac=0.30,
                    feature_noise_max=max(0.02, 0.60 * feature_noise_max),
                )
                if ep_r is not None:
                    xs_l.append(ep_r[0])
                    ys_l.append(ep_r[1])
                    wqs_l.append(ep_r[2])
                    infos_l.append(ep_r[3])
            if len(xs_l) >= 2:
                fps_r = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_l, ys_l)]
                item = _pack_worker_item_h455(
                    np.stack(xs_l),
                    np.stack(ys_l),
                    np.stack(wqs_l),
                    n_ctx_b,
                    infos_l,
                    fps_r,
                    rng,
                    postnorm_affine_max=postnorm_affine_max,
                    batch_kind="exact_resid",
                )
                while not stop_ev.is_set():
                    try:
                        out_q.put(item, timeout=1.0)
                        break
                    except queue.Full:
                        gov.pace()
                continue

        # Stream 4 (`~50%`): Shape-Binned Uniform-S3-Source Maximum-Entropy Real + GP-Stage/Residual Episodes
        n_b = int(rng.choice(allowed_n_s3))
        f_b = int(rng.choice(F_GRID_H455))
        p_c = draw_p_ctx_h455(rng, p_ctx_min, p_ctx_max)
        n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))
        b_tgt = max(4, BASE._target_batch_size(n_b, f_b, min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap))
        xs_l, ys_l, wqs_l, infos_l = [], [], [], []
        for _try in range(b_tgt * 3):
            if len(xs_l) >= b_tgt or stop_ev.is_set():
                break
            if dataset_mix == "s3_source_uniform" and float(rng.random()) >= (regret_tilt_prob + d91_boost_prob):
                t_idx = bank.sample_s3_uniform_table_idx(rng)
            else:
                t_idx, _ = gp_store.sample_table_idx(
                    bank, rng, regret_tilt_prob=regret_tilt_prob, d91_boost_prob=d91_boost_prob
                )
            win_t = gp_store.pick_winner_for_table(t_idx, rng)
            ep_s3 = sample_binned_s3_maxent_episode_h455(
                bank,
                t_idx,
                n_b,
                f_b,
                n_ctx_b,
                rng,
                valid_tcols_cache,
                win_t,
                drop_min_frac=drop_min_frac,
                drop_max_frac=drop_max_frac,
                inj_min_frac=inj_min_frac,
                inj_max_frac=inj_max_frac,
                rand_target_prob=rand_target_prob,
                synth_teacher_prob=synth_teacher_prob,
                feature_noise_max=feature_noise_max,
                gp_stage_prob=gp_stage_prob,
                gp_resid_prob=gp_resid_prob,
                distractor_prob=distractor_prob,
            )
            if ep_s3 is not None:
                xs_l.append(ep_s3[0])
                ys_l.append(ep_s3[1])
                wqs_l.append(ep_s3[2])
                infos_l.append(ep_s3[3])

        if len(xs_l) >= 2:
            fps_s3 = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_l, ys_l)]
            item = _pack_worker_item_h455(
                np.stack(xs_l),
                np.stack(ys_l),
                np.stack(wqs_l),
                n_ctx_b,
                infos_l,
                fps_s3,
                rng,
                postnorm_affine_max=postnorm_affine_max,
                batch_kind="real_s3",
            )
            while not stop_ev.is_set():
                try:
                    out_q.put(item, timeout=1.0)
                    break
                except queue.Full:
                    gov.pace()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), default="L")
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--mmap-bank", required=True)
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endless", type=int, choices=(0, 1), default=1, help="Never stop at a plateau or hour cap")
    ap.add_argument("--hours", type=float, default=960.0)
    ap.add_argument("--lr", type=float, default=1.6e-5)
    ap.add_argument("--lr-cycle-peak", type=float, default=7.5e-6)
    ap.add_argument("--lr-min", type=float, default=2.5e-6)
    ap.add_argument("--cycle0-hours", type=float, default=3.0)
    ap.add_argument("--cycle-hours", type=float, default=2.5)
    ap.add_argument("--reset-cycle-start", type=int, choices=(0, 1), default=0)
    ap.add_argument("--teacher-shards-dir", default="")
    ap.add_argument("--allow-d91-teacher-shards", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hwm-json", default="")
    ap.add_argument("--gp-winners-jsonl", default="")
    ap.add_argument("--exact-kd-prob", type=float, default=0.25)
    ap.add_argument("--exact-resid-prob", type=float, default=0.15)
    ap.add_argument("--cf-weight-scale", type=float, default=0.65)
    ap.add_argument("--gp-stage-prob", type=float, default=0.45)
    ap.add_argument("--gp-resid-prob", type=float, default=0.20)
    ap.add_argument("--regret-tilt-prob", type=float, default=0.05)
    ap.add_argument("--d91-random-split-boost", type=float, default=0.00)
    ap.add_argument("--save-every-datasets", type=int, default=50_000)
    ap.add_argument("--last-every-s", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--swa-decay", type=float, default=0.9993)
    ap.add_argument("--synth-prob", type=float, default=0.10)
    ap.add_argument("--dataset-mix", choices=("s3_source_uniform", "uniform", "smart_stratified"), default="s3_source_uniform")
    ap.add_argument("--max-episode-rows", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=380_000)
    ap.add_argument("--row-tokens", type=int, default=12_288)
    ap.add_argument("--attn-cap", type=int, default=9_500_000)
    ap.add_argument("--bmax", type=int, default=18)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-row-blocks", type=int, choices=(0, 1), default=1)
    ap.add_argument("--cpu-hard-pct", type=float, default=78.0)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.30)
    ap.add_argument("--feature-drop-min-frac", type=float, default=0.00)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.85)
    ap.add_argument("--synth-inj-min-frac", type=float, default=0.00)
    ap.add_argument("--synth-inj-max-frac", type=float, default=0.60)
    ap.add_argument("--distractor-prob", type=float, default=0.22)
    ap.add_argument("--p-ctx-min", type=float, default=0.08)
    ap.add_argument("--p-ctx-max", type=float, default=0.95)
    ap.add_argument("--rand-target-prob", type=float, default=0.38)
    ap.add_argument("--synth-teacher-prob", type=float, default=0.18)
    ap.add_argument("--feature-noise-max", type=float, default=0.06)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.26)
    ap.add_argument("--seed", type=int, default=455101)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_p = out_dir / "ckpt_last.pt"
    latest_p = out_dir / "ckpt_latest.pt"
    valley_latest_p = out_dir / "ckpt_valley_latest.pt"
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    if sha256_file(Path(a.init)) != a.init_sha256:
        raise SystemExit("init checkpoint sha256 mismatch")

    smode = (
        "s3_source_uniform"
        if a.dataset_mix == "s3_source_uniform"
        else ("uniform_dataset" if a.dataset_mix == "uniform" else "smart_stratified")
    )
    t_load0 = time.time()
    bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, verify_data_sha=True, sampling_mode=smode)
    bank_load_s = round(time.time() - t_load0, 3)
    ledger = BASE.Ledger128(bank)
    gov = BASE.HostCpuGovernor(hard_pct=float(a.cpu_hard_pct), check_every_s=5.0)

    if a.device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if 0.05 < float(a.cuda_vram_frac) <= 0.98:
            try:
                torch.cuda.set_per_process_memory_fraction(float(a.cuda_vram_frac))
            except (RuntimeError, ValueError):
                pass
    elif a.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    torch.manual_seed(a.seed)
    model = ICLH450MPS(a.size, checkpoint_row_blocks=bool(a.checkpoint_row_blocks)).to(dev)
    ema = ICLH450MPS(a.size).to(dev).eval()
    ema_swa = ICLH450MPS(a.size).to(dev).eval()
    ema_valley = ICLH450MPS(a.size).to(dev).eval()
    for m_aux in (ema, ema_swa, ema_valley):
        for p in m_aux.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)

    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
        ema_swa.load_state_dict(ck.get("ema_swa", ck["ema"]), strict=True)
        ema_valley.load_state_dict(ck.get("ema_valley", ck.get("ema_cooldown", ck.get("ema_swa", ck["ema"]))), strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        state = ck["state"]
        ledger.load(ck["ledger"])
        state.setdefault("exact_kd_batches", 0)
        state.setdefault("exact_resid_batches", 0)
        state.setdefault("exact_kd_datasets", 0)
        state.setdefault("exact_resid_datasets", 0)
        state.setdefault("valleys_saved", 0)
        if bool(a.reset_cycle_start) or "cycle_start_s" not in state:
            state["cycle_start_s"] = float(state["train_s"])
            ema_valley.load_state_dict(ema_swa.state_dict(), strict=True)
    else:
        ck_init = torch.load(a.init, map_location="cpu", weights_only=False)
        sd_init = (
            ck_init.get("ema_cooldown")
            or ck_init.get("ema_valley")
            or ck_init.get("ema_swa")
            or ck_init.get("ema")
            or ck_init["model"]
        )
        model.load_state_dict(sd_init, strict=True)
        ema.load_state_dict(sd_init, strict=True)
        ema_swa.load_state_dict(sd_init, strict=True)
        ema_valley.load_state_dict(sd_init, strict=True)
        state = {
            "step": 0,
            "micro_batches": 0,
            "accum_in_group": 0,
            "datasets": 0,
            "last_epoch_datasets": 0,
            "real_batches": 0,
            "synth_batches": 0,
            "exact_kd_batches": 0,
            "exact_resid_batches": 0,
            "exact_kd_datasets": 0,
            "exact_resid_datasets": 0,
            "cell_tokens": 0,
            "train_s": 0.0,
            "cycle_start_s": 0.0,
            "valleys_saved": 0,
            "epochs_saved": 0,
            "oom_skips": 0,
            "nonfinite_skips": 0,
        }

    sampler_cfg = {
        "protocol": "h455-maxent-s3uniform-binned-gp-flywheel",
        "canonical_promotion_forbidden": True,
        "confirm_opens": 0,
        "fmgap_bc_opens": 0,
        "dataset_mix": a.dataset_mix,
        "n_s3_families": bank.n_families,
        "n_tables": bank.n_tables,
        "endless": bool(a.endless),
        "exact_kd_prob": a.exact_kd_prob,
        "exact_resid_prob": a.exact_resid_prob,
        "feature_drop_range": [a.feature_drop_min_frac, a.feature_drop_max_frac],
        "synth_inj_range": [a.synth_inj_min_frac, a.synth_inj_max_frac],
        "distractor_prob": a.distractor_prob,
        "p_ctx_range": [a.p_ctx_min, a.p_ctx_max],
        "lr": a.lr,
        "lr_cycle_peak": a.lr_cycle_peak,
        "lr_min": a.lr_min,
        "cycle0_hours": a.cycle0_hours,
        "cycle_hours": a.cycle_hours,
    }

    stop_flag = {"stop": False}

    def _on_sig(_s, _f):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    run_nonce = os.urandom(32)
    prior_seed_base = int.from_bytes(os.urandom(4), "little")
    ctx_mp = mp.get_context("spawn") if sys.platform == "darwin" else mp.get_context("fork")
    ep_q = ctx_mp.Queue(maxsize=max(12, a.workers * 4))
    stop_ev = ctx_mp.Event()
    procs = [
        ctx_mp.Process(
            target=_flywheel_worker_loop_h455,
            args=(
                w,
                run_nonce,
                str(a.mmap_bank),
                str(a.bank_sha256),
                str(a.dataset_mix),
                int(a.max_episode_rows),
                int(a.tokens),
                int(a.row_tokens),
                int(a.attn_cap),
                int(a.bmax),
                float(a.cpu_hard_pct),
                float(a.feature_drop_min_frac),
                float(a.feature_drop_max_frac),
                float(a.synth_inj_min_frac),
                float(a.synth_inj_max_frac),
                float(a.distractor_prob),
                float(a.p_ctx_min),
                float(a.p_ctx_max),
                float(a.rand_target_prob),
                float(a.synth_teacher_prob),
                float(a.feature_noise_max),
                float(a.postnorm_affine_max),
                float(a.synth_prob),
                float(a.exact_kd_prob),
                float(a.exact_resid_prob),
                float(a.cf_weight_scale),
                str(a.teacher_shards_dir),
                bool(a.allow_d91_teacher_shards),
                str(a.hwm_json),
                str(a.gp_winners_jsonl),
                float(a.gp_stage_prob),
                float(a.gp_resid_prob),
                float(a.regret_tilt_prob),
                float(a.d91_random_split_boost),
                prior_seed_base,
                ep_q,
                stop_ev,
            ),
            daemon=True,
        )
        for w in range(a.workers)
    ]
    for p in procs:
        p.start()

    def _build_eval_payload() -> dict:
        ev_v = {kk: vv.cpu() for kk, vv in ema_valley.state_dict().items()}
        ev_s = {kk: vv.cpu() for kk, vv in ema_swa.state_dict().items()}
        soup = {kk: (0.5 * ev_v[kk].float() + 0.5 * ev_s[kk].float()).to(dtype=ev_v[kk].dtype) for kk in ev_v}
        return {
            "ema_valley": ev_v,
            "ema_cooldown": ev_v,
            "ema_swa": ev_s,
            "ema": soup,
            "model": soup,
            "state": dict(state),
            "size": a.size,
            "sampler_cfg": sampler_cfg,
        }

    def save_latest_eval_ckpt() -> str:
        payload = _build_eval_payload()
        BASE._atomic_save(payload, latest_p, mode=0o644)
        BASE._atomic_save(payload, valley_latest_p, mode=0o644)
        return sha256_file(latest_p)

    def save_valley_ckpt(cyc_idx: int) -> str:
        vp = out_dir / f"ckpt_valley_{cyc_idx:03d}.pt"
        if not vp.exists():
            BASE._atomic_save(_build_eval_payload(), vp, mode=0o644)
        return sha256_file(vp)

    def save_last() -> None:
        BASE._atomic_save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                "ema_swa": {k: v.cpu() for k, v in ema_swa.state_dict().items()},
                "ema_valley": {k: v.cpu() for k, v in ema_valley.state_dict().items()},
                "ema_cooldown": {k: v.cpu() for k, v in ema_valley.state_dict().items()},
                "opt": opt.state_dict(),
                "state": state,
                "ledger": ledger.state(),
                "size": a.size,
                "bank_sha256": bank.sha256,
                "init_sha256": a.init_sha256,
                "argv": sys.argv,
            },
            last_p,
            mode=0o644,
        )

    save_latest_eval_ckpt()
    budget_s = max(15.0, a.hours * 3600.0)
    params_cnt = sum(p.numel() for p in model.parameters())
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)
    log_f.write(
        json.dumps(
            {
                "start": time.strftime("%FT%TZ", time.gmtime()),
                "host": platform.node(),
                "device": str(dev),
                "resume": bool(a.resume),
                "params": params_cnt,
                "bank_tables": bank.n_tables,
                "bank_s3_families": bank.n_families,
                "bank_load_s": bank_load_s,
                "train_s": state["train_s"],
                "step": state["step"],
                "sampler_cfg": sampler_cfg,
            }
        )
        + "\n"
    )

    grad_accum = max(1, int(a.grad_accum))
    opt.zero_grad(set_to_none=True)
    t_seg = time.time()
    s_seg0 = float(state["train_s"])
    t_log = t_last = time.time()
    run_loss, run_n, run_ds = 0.0, 0, 0
    swa_decay_target = max(0.99, min(0.99995, float(a.swa_decay)))
    steps_since_start = 0
    last_cyc_recorded = int(state.get("valleys_saved", 0))

    try:
        while not stop_flag["stop"]:
            state["train_s"] = s_seg0 + (time.time() - t_seg)
            if a.max_steps > 0 and state["step"] >= a.max_steps:
                break
            if not bool(a.endless) and state["train_s"] >= budget_s:
                break

            cur_lr, cyc_idx, cyc_u = compute_cyclic_valley_lr(
                state["train_s"],
                float(state.get("cycle_start_s", 0.0)),
                float(a.lr),
                float(a.lr_cycle_peak),
                float(a.lr_min),
                float(a.cycle0_hours),
                float(a.cycle_hours),
            )
            for g in opt.param_groups:
                g["lr"] = cur_lr

            # Save a pinned valley checkpoint whenever a cycle finishes its valley (`cyc_idx > last_cyc_recorded`)
            if cyc_idx > last_cyc_recorded and steps_since_start >= 10:
                last_cyc_recorded = cyc_idx
                state["valleys_saved"] = cyc_idx
                v_sha = save_valley_ckpt(cyc_idx)
                save_latest_eval_ckpt()
                log_f.write(
                    json.dumps(
                        {
                            "event": "valley_saved",
                            "valley_idx": cyc_idx,
                            "step": state["step"],
                            "lr": round(cur_lr, 9),
                            "ckpt_sha256": v_sha,
                        }
                    )
                    + "\n"
                )

            if state["datasets"] - int(state.get("last_epoch_datasets", 0)) >= max(1, a.save_every_datasets):
                state["last_epoch_datasets"] = int(state["datasets"])
                k_ep = int(state["epochs_saved"]) + 1
                ck_sha = save_latest_eval_ckpt()
                jp = out_dir / f"epoch_{k_ep:03d}.json"
                if not jp.exists():
                    jp.write_text(
                        json.dumps(
                            {
                                "schema": "jope.h455.epoch/v1",
                                "epoch": k_ep,
                                "train_hours": state["train_s"] / 3600.0,
                                "step": state["step"],
                                "ckpt": str(latest_p),
                                "ckpt_sha256": ck_sha,
                                "sampler_cfg": sampler_cfg,
                                "exposure": ledger.summary(),
                            },
                            indent=1,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                state["epochs_saved"] = k_ep

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
            zm_np, yz_np, ym_np, ys_np, yq_np, wq_np, n_ctx, infos, fps, batch_kind = item_got

            keep_idx = ledger.filter_unseen_indices(fps)
            if not keep_idx:
                continue
            if len(keep_idx) < zm_np.shape[0]:
                zm_np, yz_np, ym_np, ys_np, yq_np, wq_np = (
                    zm_np[keep_idx],
                    yz_np[keep_idx],
                    ym_np[keep_idx],
                    ys_np[keep_idx],
                    yq_np[keep_idx],
                    wq_np[keep_idx],
                )
                if infos is not None:
                    infos = [infos[i] for i in keep_idx]

            B_cur, N_cur, Fn_cur, _ = zm_np.shape
            if dev.type == "cuda":
                zm_t = torch.from_numpy(zm_np).pin_memory().to(dev, non_blocking=True)
                yz_t = torch.from_numpy(yz_np).pin_memory().to(dev, non_blocking=True)
                ym_t = torch.from_numpy(ym_np).pin_memory().to(dev, non_blocking=True)
                ys_t = torch.from_numpy(ys_np).pin_memory().to(dev, non_blocking=True)
                yq_t = torch.from_numpy(yq_np).pin_memory().to(dev, non_blocking=True)
                wq_t = torch.from_numpy(wq_np).pin_memory().to(dev, non_blocking=True)
            elif dev.type == "mps":
                zm_t = torch.from_numpy(zm_np).to(dev)
                yz_t = torch.from_numpy(yz_np).to(dev)
                ym_t = torch.from_numpy(np.ascontiguousarray(ym_np, dtype=np.float32)).to(dev)
                ys_t = torch.from_numpy(np.ascontiguousarray(ys_np, dtype=np.float32)).to(dev)
                yq_t = torch.from_numpy(np.ascontiguousarray(yq_np, dtype=np.float32)).to(dev)
                wq_t = torch.from_numpy(wq_np).to(dev)
            else:
                zm_t, yz_t, ym_t, ys_t, yq_t = (
                    torch.from_numpy(zm_np),
                    torch.from_numpy(yz_np),
                    torch.from_numpy(ym_np),
                    torch.from_numpy(ys_np),
                    torch.from_numpy(yq_np),
                )
                wq_t = torch.from_numpy(wq_np).to(dev)

            slots = torch.rand(B_cur, FID_SLOTS, device=dev).argsort(dim=-1)[:, None, :Fn_cur]
            try:
                logits, ym, ys_ = model(
                    None,
                    None,
                    n_ctx,
                    slots=slots,
                    amp=(dev.type in ("cuda", "mps")),
                    prenorm=(zm_t, yz_t, ym_t, ys_t),
                )
                per_q_loss = model.loss(logits, yq_t, ym, ys_)
                loss = (per_q_loss * wq_t).sum() / wq_t.sum().clamp_min(1e-6)
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
                if batch_kind == "exact_kd":
                    state["exact_kd_batches"] += 1
                    state["exact_kd_datasets"] += B_cur
                elif batch_kind == "exact_resid":
                    state["exact_resid_batches"] += 1
                    state["exact_resid_datasets"] += B_cur
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
                    # Weight `ema_valley` 3x more strongly near the cool floor (`cur_lr <= 4.5e-6`)!
                    val_alpha = 0.0025 if cur_lr <= float(a.lr_min) * 1.8 else 0.0008
                    val_d = max(0.90, 1.0 - val_alpha) if steps_since_start > 20 else ((1.0 + steps_since_start) / (10.0 + steps_since_start))
                    for pe, ps, pv, pm in zip(
                        ema.parameters(), ema_swa.parameters(), ema_valley.parameters(), model.parameters()
                    ):
                        pe.lerp_(pm, 1.0 - decay)
                        ps.lerp_(pm, 1.0 - swa_d)
                        pv.lerp_(pm, 1.0 - val_d)
                if steps_since_start == 10:
                    save_last()
                    save_latest_eval_ckpt()
                    t_last = time.time()

            now = time.time()
            if now - t_log >= 30.0 and run_n > 0:
                host_cpu_pct = gov.sample_pct()
                s = ledger.summary()
                dt = max(1e-3, now - t_log)
                mps_gb = (
                    round(torch.mps.current_allocated_memory() / 2**30, 2)
                    if (dev.type == "mps" and hasattr(torch.mps, "current_allocated_memory"))
                    else 0.0
                )
                cuda_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else mps_gb
                log_f.write(
                    json.dumps(
                        {
                            "step": state["step"],
                            "train_h": round(state["train_s"] / 3600.0, 4),
                            "cycle": cyc_idx,
                            "cycle_u": round(cyc_u, 4),
                            "loss": round(run_loss / run_n, 5),
                            "lr": round(cur_lr, 9),
                            "datasets": state["datasets"],
                            "ds_per_s": round(run_ds / dt, 3),
                            "mb_per_s": round(run_n / dt, 3),
                            "real_batches": state["real_batches"],
                            "synth_batches": state["synth_batches"],
                            "exact_kd_batches": state["exact_kd_batches"],
                            "exact_resid_batches": state["exact_resid_batches"],
                            "exact_kd_datasets": state["exact_kd_datasets"],
                            "exact_resid_datasets": state["exact_resid_datasets"],
                            "kd_ds_per_batch": round(state["exact_kd_datasets"] / max(1, state["exact_kd_batches"]), 2),
                            "resid_ds_per_batch": round(state["exact_resid_datasets"] / max(1, state["exact_resid_batches"]), 2),
                            "real_eps": s["real_episodes"],
                            "d91_share": s["dev91_share_of_real_episodes"],
                            "unique_datasets": s["unique_datasets_seen"],
                            "dup_fed": s["duplicate_datasets_fed_to_model"],
                            "src_tables_seen": s["distinct_source_tables_seen"],
                            "host_cpu_pct": round(host_cpu_pct, 1),
                            "oom": state["oom_skips"],
                            "mem_gb": cuda_gb,
                            "protocol": sampler_cfg["protocol"],
                        }
                    )
                    + "\n"
                )
                log_f.flush()
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
        except (OSError, ValueError, AssertionError):
            pass
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=2.0)

    state["train_s"] = s_seg0 + (time.time() - t_seg)
    save_last()
    save_latest_eval_ckpt()
    log_f.write(
        json.dumps(
            {
                "paused": time.strftime("%FT%TZ", time.gmtime()),
                "step": state["step"],
                "datasets": state["datasets"],
                "exact_kd_datasets": state["exact_kd_datasets"],
                "exact_resid_datasets": state["exact_resid_datasets"],
                "train_s": round(state["train_s"], 2),
                "exposure": ledger.summary(),
            }
        )
        + "\n"
    )
    log_f.close()


if __name__ == "__main__":
    main()
