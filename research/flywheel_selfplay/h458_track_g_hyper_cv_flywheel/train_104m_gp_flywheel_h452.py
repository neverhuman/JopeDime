#!/usr/bin/env python3
"""H452 v2 `104M` (`ICL_L`) Extreme-Diversity Stratified-Crop + All-Data (`dev91` Random-Split) + Heavy-Synth
+ `GP-Winner -> 104M` Deep Learning Flywheel Trainer (`train_104m_gp_flywheel_h452.py`).

Implements Wave H452 v2 (`h452_extreme_diversity_stratified_flywheel_plan.md`):
  1. All 3,525 Tables in `mmap_v1` Active (`0` tables excluded; sealed cohorts blocked by `MmapF64Bank`):
     - `55%` pure `1/3525` uniform-by-dataset + `25%` GP-regret-tilted all-corpus + `20%` explicit `D91`
       (`D91-COMBINED-SHUFFLED` + `D91-S3-COUSIN`) random-split + fractional stratified-crop boost.
  2. Multi-Scale Dynamic Fractional Row Cropping (`17` bins `32..2048`, rotated every `32` episodes):
     - `75%` dynamic fractional sub-crop (`U(0.35, 0.92) * r_tot`) EVEN when `r_tot <= active_rows` (`100%` forced
       fractional/varied crop on boosted `D91` draws so small/medium tables yield infinite distinct sub-datasets).
     - `92%` 4-mode stratified row cropping: `32%` target-quantile (`K=6..24`), `28%` 2D `(Oblique X-Proj x y-Quantile)`,
       `20%` `Dirichlet(0.35..1.2)` sub-population regime crop, `12%` `2.2x` tail-amplified stratified, `8%` uniform.
  3. `100%` Fresh CSPRNG Context/Query Split Every Episode (`p_ctx ~ U(0.14, 0.95)`):
     - `60%` target-quantile stratified split (`CV-5/CV-7` aligned), `25%` uniform random, `15%` oblique covariate-drift split.
  4. Heavy Zero-Leakage Synthetic & GP-Winner Augmentation (100% context-fitted on `Xc, yc`, `0` query leakage):
     - Column drop `[5%, 75%]` + `18`-family synthetic column injection `[10%, 45%]` (Chebyshev `T_2/T_3`, safe ratios,
       3-way oblique hinges, symlog, RFF projections, RankGauss, prototype RBFs);
     - `30%` context-fitted monotone feature warps (`|z|^gamma`, `asinh`, RankGauss) + `25%` MCAR/MAR `NaN` masks (`2..14%`);
     - Per-feature sign flip + `exp(U(-3.5, 3.5))` scale + shift + `0.2..7%` Student-t/Laplace/Gaussian noise + `+-0.26`
       post-norm affine jitter + `100%` feature & `FID_SLOTS` permutation;
     - `30%` full-synthetic batches (`50%` GP-Winner AST distillation) + real-episode target mix (`35%` canonical,
       `40%` random valid feature-as-target, `25%` GP-AST/oblivious teacher with `40%` real-y blend);
     - `50%` Channel A GP-Winner stage transforms + `30%` Channel B 5-fold OOF nonlinear GP/tree residual conditioning.
  5. Zero file deletions (`0` `rm`/`unlink`/`rmtree`, atomic `.tmp -> os.replace` onto `ckpt_last.pt`,
     `ckpt_latest.pt`, `ckpt_cooldown_latest.pt`) and 128-bit `Ledger128` deduplication (`dup_fed == 0`).
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
    "/home/ubuntu/h450-stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    str(Path.home() / "jope-data/h450-fleet-agy/stage/fleet-r10/code"),
    str(Path.home() / "jope-data/h450-fleet-agy/fleet"),
    str(Path.home() / "jope-data/h450-fleet-agy/ref_code"),
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

import train_mps_104m_h450 as BASE  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_mps_h450 import ICLH450MPS, n_params  # noqa: E402
from icl_prior_v3s import Prior, draw_features  # noqa: E402
from mmap_bank_h450 import MmapF64Bank, sha256_file  # noqa: E402

ROW_BINS_V2 = (32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1408, 1728, 2048)

STAGE_TOKENS = (
    "pair_ratio",
    "pair_prod",
    "cheb",
    "symlog",
    "poly_interact",
    "hinge",
    "oblique_hinge",
    "spline",
    "rank",
    "bin_onehot",
)


def parse_dsl_stages_and_head(dsl: str) -> tuple[tuple[str, ...], str]:
    stages = tuple(tok for tok in STAGE_TOKENS if f"({tok}" in dsl)
    head = "ridge_gcv"
    for h in (
        "portfolio",
        "oblivious",
        "xtrees",
        "rforest",
        "histgbm2",
        "gpr",
        "krr",
        "ksmooth",
        "elasticnet",
        "ridge_gcv",
    ):
        if f"({h}" in dsl:
            head = h
            break
    return stages, head


class GPWinnerStore:
    """Hot-reloading store of verified `CV-7` GP winners + all-corpus / `D91` random-split table sampler."""

    def __init__(self, jsonl_path: str, bank: MmapF64Bank, reload_every_s: float = 30.0) -> None:
        self.path = Path(jsonl_path) if jsonl_path else None
        self.n_tables = int(bank.n_tables)
        self.d91_indices = np.flatnonzero(bank.is_d91).astype(np.int64)
        self.by_cid_idx = {str(cid): idx for idx, cid in enumerate(bank.cids)}
        self.reload_every_s = float(reload_every_s)
        self._last_check = 0.0
        self._last_mtime = -1.0
        self.by_table: dict[int, list[dict]] = {}
        self.all_winners: list[dict] = []
        self.table_probs: np.ndarray | None = None
        self.maybe_reload(force=True)

    def maybe_reload(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_check) < self.reload_every_s:
            return
        self._last_check = now
        if self.path is None or not self.path.is_file():
            return
        try:
            mt = self.path.stat().st_mtime
            if not force and mt <= self._last_mtime:
                return
            by_t: dict[int, list[dict]] = {}
            all_w: list[dict] = []
            regret_by_t = np.zeros(self.n_tables, dtype=np.float64)
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
                    dsl = str(rec.get("dsl", ""))
                    if not dsl:
                        continue
                    stages, head = parse_dsl_stages_and_head(dsl)
                    regret = max(0.0, float(rec.get("regret_vs_linear", rec.get("delta_u", 0.08))))
                    entry = {
                        "dsl": dsl,
                        "stages": stages or ("pair_ratio", "cheb", "symlog"),
                        "head": head,
                        "regret": regret,
                    }
                    all_w.append(entry)
                    t_idx = rec.get("table_idx")
                    if not isinstance(t_idx, int) or not (0 <= t_idx < self.n_tables):
                        cid_s = str(rec.get("cid", ""))
                        t_idx = self.by_cid_idx.get(cid_s, -1)
                    if isinstance(t_idx, int) and 0 <= t_idx < self.n_tables:
                        by_t.setdefault(t_idx, []).append(entry)
                        regret_by_t[t_idx] = max(regret_by_t[t_idx], min(regret, 1.8))
            if all_w:
                self.by_table = by_t
                self.all_winners = all_w
                w = 1.0 + 3.5 * regret_by_t
                self.table_probs = w / float(np.sum(w))
                self._last_mtime = mt
        except (OSError, ValueError, KeyError):
            pass

    def sample_table_idx(
        self,
        bank: MmapF64Bank,
        rng: np.random.Generator,
        regret_tilt_prob: float = 0.25,
        d91_boost_prob: float = 0.20,
    ) -> tuple[int, bool]:
        """Returns `(t_idx, is_d91_boosted)` across all 3,525 tables (`0` tables excluded)."""
        u = float(rng.random())
        if d91_boost_prob > 0.0 and self.d91_indices.size > 0 and u < d91_boost_prob:
            return int(self.d91_indices[int(rng.integers(0, self.d91_indices.size))]), True
        if self.table_probs is not None and u < (d91_boost_prob + regret_tilt_prob):
            return int(rng.choice(self.n_tables, p=self.table_probs)), False
        return bank.sample_table_idx(rng), False

    def pick_winner_for_table(self, t_idx: int, rng: np.random.Generator) -> dict | None:
        cands = self.by_table.get(int(t_idx))
        if cands and float(rng.random()) < 0.70:
            return cands[int(rng.integers(0, len(cands)))]
        if self.all_winners:
            return self.all_winners[int(rng.integers(0, len(self.all_winners)))]
        return None


def stratified_row_crop_h452(
    tab_mmap: np.ndarray,
    valid_rows: np.ndarray,
    y_valid: np.ndarray | None,
    feats: list[int],
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """4-mode stratified row cropper (`92%` stratified: y-quantile, 2D X-proj x y-quantile, Dirichlet regime, tail-boost)."""
    r_tot = int(valid_rows.size)
    if r_tot <= n_rows:
        return rng.permutation(r_tot)[:n_rows]
    u_mode = float(rng.random())
    if y_valid is None or u_mode >= 0.92:
        return rng.choice(r_tot, size=n_rows, replace=False)

    sd_y = max(float(np.std(y_valid)), 1.0)
    yv = y_valid + 1e-12 * sd_y * rng.standard_normal(r_tot)

    if u_mode < 0.32:
        # Mode S1 (32%): Fine target-quantile stratified crop (K = 6..24 strata)
        k_q = max(2, min(int(rng.integers(6, 25)), r_tot // 3))
        bins = np.array_split(np.argsort(yv, kind="mergesort"), k_q)
        weights = np.array([len(b) for b in bins], dtype=np.float64)
    elif u_mode < 0.60 and len(feats) >= 1:
        # Mode S2 (28%): 2D Joint (Random Oblique Covariate Projection x Target Quantile) Grid
        k_f = min(len(feats), int(rng.integers(1, 6)))
        sub_f = rng.choice(feats, size=k_f, replace=False)
        X_sub = np.asarray(tab_mmap[valid_rows][:, sub_f], dtype=np.float64)
        proj = np.zeros(r_tot, dtype=np.float64)
        w_dir = rng.standard_normal(k_f) / math.sqrt(float(k_f))
        for jj in range(k_f):
            col_j = X_sub[:, jj]
            fin_j = np.isfinite(col_j)
            if int(np.sum(fin_j)) >= 4:
                med_j = float(np.median(col_j[fin_j]))
                sd_j = max(float(np.std(col_j[fin_j])), 1e-9)
                proj += w_dir[jj] * np.clip((np.where(fin_j, col_j, med_j) - med_j) / sd_j, -6.0, 6.0)
        proj += 1e-12 * rng.standard_normal(r_tot)
        cx = int(rng.choice([3, 4, 6]))
        cy = int(rng.choice([3, 4]))
        qx = np.quantile(proj, np.linspace(0.0, 1.0, cx + 1)[1:-1])
        qy = np.quantile(yv, np.linspace(0.0, 1.0, cy + 1)[1:-1])
        cell = np.digitize(proj, qx) * cy + np.digitize(yv, qy)
        bins = [np.flatnonzero(cell == c_i) for c_i in range(cx * cy)]
        bins = [b for b in bins if len(b) > 0]
        weights = np.array([len(b) for b in bins], dtype=np.float64)
    elif u_mode < 0.80:
        # Mode S3 (20%): Dirichlet(0.35..1.20) Sub-Population Regime Stratified Crop
        k_reg = max(3, min(int(rng.integers(8, 13)), r_tot // 3))
        if feats and float(rng.random()) < 0.50:
            jf = int(rng.choice(feats))
            cf_col = np.asarray(tab_mmap[valid_rows, jf], dtype=np.float64)
            med_f = float(np.nanmedian(cf_col)) if np.any(np.isfinite(cf_col)) else 0.0
            key_v = np.where(np.isfinite(cf_col), cf_col, med_f) + 1e-12 * rng.standard_normal(r_tot)
        else:
            key_v = yv
        bins = np.array_split(np.argsort(key_v, kind="mergesort"), k_reg)
        alpha = float(rng.uniform(0.35, 1.20))
        weights = rng.dirichlet(np.full(len(bins), alpha))
    else:
        # Mode S4 (12%): Tail-Amplified Stratified Crop (2.2x extreme deciles)
        k_t = min(10, max(4, r_tot // 4))
        bins = np.array_split(np.argsort(yv, kind="mergesort"), k_t)
        weights = np.ones(len(bins), dtype=np.float64) * 0.85
        weights[0] = 2.2
        weights[-1] = 2.2
        if len(bins) >= 6:
            weights[1] = 1.2
            weights[-2] = 1.2

    if not bins:
        return rng.choice(r_tot, size=n_rows, replace=False)
    weights = weights / np.maximum(float(np.sum(weights)), 1e-12)
    picked: list[np.ndarray] = []
    need = n_rows
    for b_i, (b_arr, w_b) in enumerate(zip(bins, weights)):
        if len(b_arr) == 0 or need <= 0:
            continue
        take = need if b_i == len(bins) - 1 else min(len(b_arr), max(1, int(round(n_rows * float(w_b)))))
        take = min(take, len(b_arr), need)
        picked.append(rng.permutation(b_arr)[:take])
        need -= take
    sel = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    if len(sel) < n_rows:
        rest = np.setdiff1d(np.arange(r_tot, dtype=np.int64), sel)
        sel = np.concatenate([sel, rng.permutation(rest)[: n_rows - len(sel)]])
    return rng.permutation(sel[:n_rows])


def inject_synth_cols_18fam_zero_leakage(
    Xc: np.ndarray,
    Xq: np.ndarray,
    k_inj: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """18-family zero-leakage synthetic feature bank fitted STRICTLY on context rows `Xc` (`0` query leakage)."""
    p = Xc.shape[1]
    n_c, n_q = Xc.shape[0], Xq.shape[0]
    if p == 0 or k_inj <= 0:
        return np.zeros((n_c, 0), dtype=np.float64), np.zeros((n_q, 0), dtype=np.float64)

    k_base = int(rng.binomial(k_inj, 0.65))
    k_extra = k_inj - k_base
    cols_c: list[np.ndarray] = []
    cols_q: list[np.ndarray] = []
    if k_base > 0:
        Bc, Bq = BASE._inject_synth_cols_np(Xc, Xq, k_base, rng)
        cols_c.append(Bc)
        cols_q.append(Bq)
    if k_extra > 0:
        Zc, Zq = BASE._std_ctx_f64(Xc, Xq)
        extra_c = np.zeros((n_c, k_extra), dtype=np.float64)
        extra_q = np.zeros((n_q, k_extra), dtype=np.float64)
        for m in range(k_extra):
            fam = int(rng.integers(0, 4))
            ja = int(rng.integers(0, p))
            jb = int(rng.integers(0, p))
            if p >= 2 and jb == ja:
                jb = (ja + 1) % p
            if fam == 0:
                # Chebyshev T_2 or T_3 polynomial
                tc, tq = np.tanh(Zc[:, ja] / 3.0), np.tanh(Zq[:, ja] / 3.0)
                if rng.random() < 0.5:
                    extra_c[:, m], extra_q[:, m] = 2.0 * tc * tc - 1.0, 2.0 * tq * tq - 1.0
                else:
                    extra_c[:, m], extra_q[:, m] = 4.0 * (tc ** 3) - 3.0 * tc, 4.0 * (tq ** 3) - 3.0 * tq
            elif fam == 1:
                # Safe context-scaled pair ratio (fitted strictly on Zc[:, jb])
                sb = max(float(np.std(Zc[:, jb])), 0.5)
                den_c = np.where(Zc[:, jb] >= 0.0, 1.0, -1.0) * np.maximum(np.abs(Zc[:, jb]), 0.35 * sb)
                den_q = np.where(Zq[:, jb] >= 0.0, 1.0, -1.0) * np.maximum(np.abs(Zq[:, jb]), 0.35 * sb)
                extra_c[:, m] = np.clip(Zc[:, ja] / den_c, -22.0, 22.0)
                extra_q[:, m] = np.clip(Zq[:, ja] / den_q, -22.0, 22.0)
            elif fam == 2:
                # 3-way oblique ReLU hinge (knot strictly from context percentile!)
                sub_k = min(p, 3)
                sub = rng.choice(p, size=sub_k, replace=False)
                w = rng.standard_normal(sub_k) / math.sqrt(float(sub_k))
                pc, pq = Zc[:, sub] @ w, Zq[:, sub] @ w
                knot = float(np.percentile(pc, float(rng.uniform(20.0, 80.0))))
                extra_c[:, m] = np.maximum(0.0, pc - knot)
                extra_q[:, m] = np.maximum(0.0, pq - knot)
            else:
                # Context-fitted Random Fourier Feature kernel projection
                sub_k = min(p, int(rng.integers(2, min(8, p) + 1))) if p >= 2 else 1
                sub = rng.choice(p, size=sub_k, replace=False)
                ls = float(math.exp(rng.uniform(-0.4, 0.8))) * math.sqrt(float(sub_k))
                w = rng.standard_normal(sub_k) / ls
                ph = float(rng.uniform(0.0, 2.0 * math.pi))
                extra_c[:, m] = np.cos(Zc[:, sub] @ w + ph)
                extra_q[:, m] = np.cos(Zq[:, sub] @ w + ph)
        cols_c.append(extra_c)
        cols_q.append(extra_q)
    return np.hstack(cols_c), np.hstack(cols_q)


def apply_monotone_warps_and_nans_zero_leakage(
    Xc: np.ndarray,
    Xq: np.ndarray,
    rng: np.random.Generator,
    warp_prob: float = 0.30,
    nan_prob: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies context-fitted monotone feature warps + synthetic MCAR/MAR NaN masks strictly from `Xc` stats."""
    p = Xc.shape[1]
    n_c, n_q = Xc.shape[0], Xq.shape[0]
    if p == 0:
        return Xc, Xq
    Xc_o = Xc.astype(np.float64, copy=True)
    Xq_o = Xq.astype(np.float64, copy=True)

    if warp_prob > 0.0 and float(rng.random()) < warp_prob:
        n_w = max(1, int(round(float(rng.uniform(0.20, 0.50)) * p)))
        w_cols = rng.choice(p, size=min(p, n_w), replace=False)
        for j in w_cols:
            jj = int(j)
            fc = np.isfinite(Xc_o[:, jj])
            fq = np.isfinite(Xq_o[:, jj])
            if int(np.sum(fc)) < 8:
                continue
            vals = Xc_o[fc, jj]
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            sc = max((q75 - q25) / 1.349, float(np.std(vals)), 1e-9)
            zc = np.clip((Xc_o[fc, jj] - med) / sc, -25.0, 25.0)
            zq = np.clip((Xq_o[fq, jj] - med) / sc, -25.0, 25.0)
            w_kind = int(rng.integers(0, 3))
            if w_kind == 0:
                gam = float(rng.uniform(0.45, 1.85))
                Xc_o[fc, jj] = np.sign(zc) * (np.abs(zc) ** gam)
                Xq_o[fq, jj] = np.sign(zq) * (np.abs(zq) ** gam)
            elif w_kind == 1:
                c_s = float(math.exp(rng.uniform(-0.5, 1.2)))
                Xc_o[fc, jj] = np.arcsinh(zc / c_s)
                Xq_o[fq, jj] = np.arcsinh(zq / c_s)
            else:
                cs = np.sort(zc)
                rc = (np.searchsorted(cs, zc, side="left") + 0.5) / float(len(cs) + 1)
                rq = (np.searchsorted(cs, zq, side="left") + 0.5) / float(len(cs) + 1)
                Xc_o[fc, jj] = math.sqrt(2.0) * BASE._erfinv_np(np.clip(2.0 * rc - 1.0, -0.9995, 0.9995))
                Xq_o[fq, jj] = math.sqrt(2.0) * BASE._erfinv_np(np.clip(2.0 * rq - 1.0, -0.9995, 0.9995))

    if nan_prob > 0.0 and p >= 3 and n_c >= 24 and float(rng.random()) < nan_prob:
        miss_rate = float(rng.uniform(0.02, 0.14))
        # Leave anchor column j=0 intact so episodes never become completely empty
        cand_j = np.arange(1, p)
        n_mcols = max(1, int(round(float(rng.uniform(0.25, 0.65)) * len(cand_j))))
        m_cols = rng.choice(cand_j, size=min(len(cand_j), n_mcols), replace=False)
        if float(rng.random()) < 0.50 and np.any(np.isfinite(Xc_o[:, 0])):
            # Logistic MAR gate anchored strictly on context column 0 median/std!
            f0_c = Xc_o[:, 0]
            med0 = float(np.nanmedian(f0_c))
            sd0 = max(float(np.nanstd(f0_c)), 1e-9)
            g_c = 1.0 / (1.0 + np.exp(-np.clip((np.where(np.isfinite(f0_c), f0_c, med0) - med0) / sd0, -4.0, 4.0)))
            g_q = 1.0 / (1.0 + np.exp(-np.clip((np.where(np.isfinite(Xq_o[:, 0]), Xq_o[:, 0], med0) - med0) / sd0, -4.0, 4.0)))
            mc = rng.random((n_c, len(m_cols))) < (miss_rate * 1.5 * g_c[:, None])
            mq = rng.random((n_q, len(m_cols))) < (miss_rate * 1.5 * g_q[:, None])
        else:
            mc = rng.random((n_c, len(m_cols))) < miss_rate
            mq = rng.random((n_q, len(m_cols))) < miss_rate
        for idx_m, jj in enumerate(m_cols):
            # Ensure at least 8 finite context values remain in column `jj`
            if int(np.sum(np.isfinite(Xc_o[:, int(jj)]) & (~mc[:, idx_m]))) >= 8:
                Xc_o[mc[:, idx_m], int(jj)] = np.nan
                Xq_o[mq[:, idx_m], int(jj)] = np.nan

    return Xc_o, Xq_o


def build_gp_stage_cols_zero_leakage(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    stages: tuple[str, ...],
    k_want: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits GP-winner symbolic/stage transforms STRICTLY on `(Xc, yc)` and transforms `Xq` (`0` query leakage!)."""
    p = Xc.shape[1]
    n_c, n_q = Xc.shape[0], Xq.shape[0]
    if p == 0 or k_want <= 0 or n_c < 16:
        return np.zeros((n_c, 0), dtype=np.float64), np.zeros((n_q, 0), dtype=np.float64)

    Zc, Zq = BASE._std_ctx_f64(Xc, Xq)
    yc0 = yc - float(np.mean(yc))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    corrs = np.array(
        [
            abs(float(np.dot(Zc[:, j] - np.mean(Zc[:, j]), yc0)))
            / max(float(np.linalg.norm(Zc[:, j] - np.mean(Zc[:, j]))) * yn, 1e-12)
            for j in range(p)
        ],
        dtype=np.float64,
    )
    top_idx = np.argsort(-corrs)[: min(p, 8)]

    cand_c: list[np.ndarray] = []
    cand_q: list[np.ndarray] = []
    st_set = set(stages) if stages else {"pair_ratio", "pair_prod", "cheb", "symlog"}

    if ("pair_ratio" in st_set or "pair_prod" in st_set or "poly_interact" in st_set) and len(top_idx) >= 2:
        for i_a in range(min(4, len(top_idx))):
            for i_b in range(i_a + 1, min(5, len(top_idx))):
                ja, jb = int(top_idx[i_a]), int(top_idx[i_b])
                if "pair_ratio" in st_set:
                    s_b = max(float(np.std(Zc[:, jb])), 0.5)
                    den_c = np.where(Zc[:, jb] >= 0.0, 1.0, -1.0) * np.maximum(np.abs(Zc[:, jb]), 0.35 * s_b)
                    den_q = np.where(Zq[:, jb] >= 0.0, 1.0, -1.0) * np.maximum(np.abs(Zq[:, jb]), 0.35 * s_b)
                    cand_c.append(np.clip(Zc[:, ja] / den_c, -20.0, 20.0))
                    cand_q.append(np.clip(Zq[:, ja] / den_q, -20.0, 20.0))
                if "pair_prod" in st_set or "poly_interact" in st_set:
                    cand_c.append(np.clip(Zc[:, ja] * Zc[:, jb], -25.0, 25.0))
                    cand_q.append(np.clip(Zq[:, ja] * Zq[:, jb], -25.0, 25.0))

    if "cheb" in st_set or "poly_interact" in st_set or "spline" in st_set:
        for ja in top_idx[: min(4, len(top_idx))]:
            tc = np.tanh(Zc[:, int(ja)] / 3.0)
            tq = np.tanh(Zq[:, int(ja)] / 3.0)
            cand_c.append(2.0 * tc * tc - 1.0)
            cand_q.append(2.0 * tq * tq - 1.0)
            cand_c.append(4.0 * (tc ** 3) - 3.0 * tc)
            cand_q.append(4.0 * (tq ** 3) - 3.0 * tq)

    if "symlog" in st_set:
        c_val = float(math.exp(rng.uniform(-0.3, 0.9)))
        for ja in top_idx[: min(4, len(top_idx))]:
            cand_c.append(np.sign(Zc[:, int(ja)]) * np.log1p(np.abs(Zc[:, int(ja)]) / c_val))
            cand_q.append(np.sign(Zq[:, int(ja)]) * np.log1p(np.abs(Zq[:, int(ja)]) / c_val))

    if "hinge" in st_set or "oblique_hinge" in st_set or "spline" in st_set:
        for _ in range(min(4, max(2, k_want))):
            if "oblique_hinge" in st_set and len(top_idx) >= 2:
                sub = rng.choice(top_idx, size=min(3, len(top_idx)), replace=False)
                w = rng.standard_normal(len(sub)) / math.sqrt(len(sub))
                pc, pq = Zc[:, sub] @ w, Zq[:, sub] @ w
            else:
                j0 = int(top_idx[int(rng.integers(0, len(top_idx)))])
                pc, pq = Zc[:, j0], Zq[:, j0]
            knot = float(np.percentile(pc, float(rng.uniform(25.0, 75.0))))
            cand_c.append(np.maximum(0.0, pc - knot))
            cand_q.append(np.maximum(0.0, pq - knot))

    if "rank" in st_set or "bin_onehot" in st_set:
        for ja in top_idx[: min(3, len(top_idx))]:
            cs = np.sort(Zc[:, int(ja)])
            rc = (np.searchsorted(cs, Zc[:, int(ja)], side="left") + 0.5) / float(n_c + 1)
            rq = (np.searchsorted(cs, Zq[:, int(ja)], side="left") + 0.5) / float(n_c + 1)
            cand_c.append(2.0 * rc - 1.0)
            cand_q.append(2.0 * rq - 1.0)

    if not cand_c:
        return inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want, rng)

    scored = []
    for idx_c, cc in enumerate(cand_c):
        cc0 = cc - float(np.mean(cc))
        cn = float(np.linalg.norm(cc0))
        sc = abs(float(np.dot(cc0, yc0))) / (cn * yn) if cn > 1e-12 else 0.0
        scored.append((sc, idx_c))
    scored.sort(key=lambda t: -t[0])
    pick = [idx_c for _, idx_c in scored[:k_want]]
    Mc = np.column_stack([cand_c[i] for i in pick])
    Mq = np.column_stack([cand_q[i] for i in pick])
    if Mc.shape[1] < k_want:
        Rc, Rq = inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want - Mc.shape[1], rng)
        Mc = np.hstack([Mc, Rc])
        Mq = np.hstack([Mq, Rq])
    return Mc[:, :k_want], Mq[:, :k_want]


def nonlinear_gp_oof_residual_np(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    head: str,
    rng: np.random.Generator,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fast pure-NumPy 5-fold OOF nonlinear GP/RFF + oblivious-basis residual predictor fitted ONLY on `(Xc, yc)`."""
    n_c, p = Xc.shape
    n_q = Xq.shape[0]
    ym = float(np.mean(yc))
    sy = float(np.std(yc))
    if n_c < 24 or p == 0 or sy <= 1e-12:
        return np.full(n_c, ym), np.full(n_q, ym), False

    Zc, Zq = BASE._std_ctx_f64(Xc, Xq)
    d_rff = min(48, max(16, n_c // 3))
    ls = float(math.exp(rng.uniform(-0.3, 0.9))) * math.sqrt(max(p, 1))
    W_rff = rng.standard_normal((p, d_rff)) / ls
    b_rff = rng.uniform(0.0, 2.0 * math.pi, size=(1, d_rff))
    Phi_c = [Zc, np.cos(Zc @ W_rff + b_rff) * math.sqrt(2.0 / d_rff)]
    Phi_q = [Zq, np.cos(Zq @ W_rff + b_rff) * math.sqrt(2.0 / d_rff)]

    if head in ("oblivious", "xtrees", "rforest", "histgbm2", "portfolio"):
        ob_c = []
        ob_q = []
        for _ in range(4):
            j_s = rng.choice(p, size=3, replace=(p < 3))
            thrs = [float(np.percentile(Zc[:, int(j_s[d])], float(rng.uniform(25.0, 75.0)))) for d in range(3)]
            lc = (
                ((Zc[:, int(j_s[0])] > thrs[0]).astype(np.int64) << 2)
                | ((Zc[:, int(j_s[1])] > thrs[1]).astype(np.int64) << 1)
                | (Zc[:, int(j_s[2])] > thrs[2]).astype(np.int64)
            )
            lq = (
                ((Zq[:, int(j_s[0])] > thrs[0]).astype(np.int64) << 2)
                | ((Zq[:, int(j_s[1])] > thrs[1]).astype(np.int64) << 1)
                | (Zq[:, int(j_s[2])] > thrs[2]).astype(np.int64)
            )
            ob_c.append(np.eye(8, dtype=np.float64)[lc])
            ob_q.append(np.eye(8, dtype=np.float64)[lq])
        Phi_c.extend(ob_c)
        Phi_q.extend(ob_q)

    Bc = np.hstack(Phi_c)
    Bq = np.hstack(Phi_q)
    oof, pq, used, _ = BASE._robust_ridge_np(Bc, yc, Bq, seed=int(rng.integers(1, 1 << 30)), folds=folds)
    rmse = float(np.sqrt(np.mean((yc - oof) ** 2)))
    ok = bool(used and rmse < 0.96 * sy and float(np.std(yc - oof)) >= 1e-3 * sy)
    return oof, pq, ok


def distill_gp_winner_synth_episode(
    Xc: np.ndarray,
    Xq: np.ndarray,
    winner: dict | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Synthesizes `(yc, yq)` from a harvested GP-winner AST (`stages + head`) on covariates `(Xc, Xq)`."""
    Zc, Zq = BASE._std_ctx_f64(Xc, Xq)
    Z = np.vstack([Zc, Zq])
    n_tot, p = Z.shape
    n_c = Xc.shape[0]
    if p == 0 or n_tot < 24:
        return None

    stages = winner.get("stages", ("pair_ratio", "cheb", "symlog")) if winner else ("pair_ratio", "cheb", "symlog")
    head = str(winner.get("head", "oblivious")) if winner else "oblivious"
    feats = [Z[:, : min(p, 10)]]
    if "pair_ratio" in stages and p >= 2:
        ja, jb = int(rng.integers(0, p)), int(rng.integers(0, p))
        if ja == jb:
            jb = (ja + 1) % p
        den = np.where(Z[:, [jb]] >= 0.0, 1.0, -1.0) * np.maximum(np.abs(Z[:, [jb]]), 0.4)
        feats.append(np.clip(Z[:, [ja]] / den, -15.0, 15.0))
    if ("pair_prod" in stages or "poly_interact" in stages) and p >= 2:
        ja, jb = int(rng.integers(0, p)), int(rng.integers(0, p))
        feats.append(np.clip(Z[:, [ja]] * Z[:, [jb]], -20.0, 20.0))
    if "cheb" in stages or "spline" in stages:
        t = np.tanh(Z[:, [0]] / 2.5)
        feats.append(4.0 * (t ** 3) - 3.0 * t)
    if "symlog" in stages:
        feats.append(np.sign(Z[:, [0]]) * np.log1p(np.abs(Z[:, [0]]) / 1.5))
    Phi = np.hstack(feats)
    d = Phi.shape[1]

    if head in ("gpr", "krr", "ksmooth"):
        ls = np.exp(rng.uniform(-0.6, 1.0, size=(1, d)))
        W = rng.standard_normal((d, 48)) / ls.T
        b = rng.uniform(0.0, 2.0 * math.pi, size=(1, 48))
        f = (np.cos(Phi @ W + b) * math.sqrt(2.0 / 48)) @ rng.standard_normal(48)
    elif head in ("ridge_gcv", "elasticnet"):
        f = Phi @ (rng.standard_normal(d) / math.sqrt(max(d, 1)))
    else:
        f_tr = np.zeros(n_tot, dtype=np.float64)
        for _ in range(6):
            depth = int(rng.integers(2, 5))
            leaf = np.zeros(n_tot, dtype=np.int64)
            for _d in range(depth):
                cj = int(rng.integers(0, d))
                thr = float(np.percentile(Phi[:n_c, cj], float(rng.uniform(20.0, 80.0))))
                leaf = (leaf << 1) | (Phi[:, cj] > thr).astype(np.int64)
            f_tr += rng.standard_normal(1 << depth)[leaf] / math.sqrt(6.0)
        if head == "portfolio":
            f_lin = Phi @ (rng.standard_normal(d) / math.sqrt(max(d, 1)))
            f = 0.65 * f_tr + 0.35 * f_lin
        else:
            f = f_tr

    sd = float(np.std(f))
    if not math.isfinite(sd) or sd <= 1e-9:
        return None
    f = (f - float(np.mean(f))) / sd
    noise = float(math.exp(rng.uniform(math.log(0.012), math.log(0.22)))) * rng.standard_normal(n_tot)
    y = f + noise
    if float(np.std(y[:n_c])) <= 1e-9 or not np.all(np.isfinite(y)):
        return None
    return y[:n_c], y[n_c:]


def sample_extreme_stratified_episode_h452(
    bank: MmapF64Bank,
    t_idx: int,
    rng: np.random.Generator,
    valid_tcols_cache: dict[int, tuple[int, ...]],
    winner: dict | None = None,
    force_fractional_crop: bool = False,
    max_episode_rows: int = BASE.DEFAULT_MAX_EPISODE_ROWS,
    drop_min_frac: float = 0.05,
    drop_max_frac: float = 0.75,
    inj_min_frac: float = 0.10,
    inj_max_frac: float = 0.45,
    rand_target_prob: float = 0.40,
    synth_teacher_prob: float = 0.25,
    feature_noise_max: float = 0.07,
    force_canonical_target: bool = False,
    n_ctx_for_shape: dict[tuple[int, int], int] | None = None,
    preferred_rows: int | None = None,
):
    """Extreme-entropy episode sampler across ALL datasets (incl. `dev91` on 100% fresh random splits)."""
    tab_mmap = bank.tabs_f64[t_idx]
    r_raw, c_tot = tab_mmap.shape
    if r_raw < 28 or c_tot < 2:
        return None

    if t_idx not in valid_tcols_cache:
        valid_tcols_cache[t_idx] = BASE._discover_valid_target_cols(tab_mmap, min_rows=28)
    valid_tcols = valid_tcols_cache[t_idx]

    # Target selection: 35% canonical | 40% random valid feature-as-target | 25% GP-AST / nonlinear teacher
    u_tm = 1.0 if force_canonical_target else float(rng.random())
    other_valid = [j for j in valid_tcols if j != (c_tot - 1)]
    blend_real_teacher = False
    if not force_canonical_target and u_tm < synth_teacher_prob:
        tmode = 2
        blend_real_teacher = bool(rng.random() < 0.40)
        tcol = int(rng.choice(valid_tcols)) if valid_tcols else (c_tot - 1)
    elif not force_canonical_target and other_valid and u_tm < (synth_teacher_prob + rand_target_prob):
        tmode = 1
        tcol = int(rng.choice(other_valid))
    else:
        tmode = 0
        tcol = c_tot - 1

    y_valid: np.ndarray | None = None
    if tmode in (0, 1) or blend_real_teacher:
        col_t = tab_mmap[:, tcol]
        valid_rows = np.flatnonzero(np.isfinite(col_t))
        if valid_rows.size < 28:
            return None
        y_valid = np.asarray(col_t[valid_rows], dtype=np.float64)
        if tmode == 0 and rng.random() < 0.40 and valid_rows.size >= 36:
            atom = BASE._detect_atom_np(y_valid)
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

    # Dynamic multi-scale fractional row cropping (`75%` fractional crop even when `r_tot <= preferred_rows`,
    # `100%` on boosted `D91` draws so small/medium tables yield infinite distinct stratified sub-crops!)
    r_cap = min(r_tot, int(max_episode_rows))
    do_frac_crop = (force_fractional_crop or float(rng.random()) < 0.75) and r_tot >= 48
    if preferred_rows is not None and r_cap >= int(preferred_rows) and not (force_fractional_crop and r_tot <= int(preferred_rows)):
        if do_frac_crop and r_tot <= int(preferred_rows):
            r_frac_tgt = max(32, int(round(float(rng.uniform(0.35, 0.92)) * r_tot)))
            smaller = [b for b in ROW_BINS_V2 if b <= min(r_cap, r_frac_tgt)]
            n_rows = smaller[-1] if smaller else min(r_cap, r_frac_tgt)
        else:
            n_rows = int(preferred_rows)
    else:
        if do_frac_crop:
            r_frac_tgt = max(32, int(round(float(rng.uniform(0.35, 0.92)) * r_cap)))
            smaller = [b for b in ROW_BINS_V2 if b <= r_frac_tgt]
            n_rows = smaller[-1] if smaller else min(r_cap, r_frac_tgt)
        else:
            ok_b = [b for b in ROW_BINS_V2 if b <= r_cap]
            n_rows = ok_b[-1] if ok_b else r_cap

    feats = [j for j in range(c_tot) if j != tcol]
    if tmode == 2 and not blend_real_teacher and len(feats) > 1 and rng.random() < 0.50:
        feats = [j for j in feats if j != c_tot - 1] or feats
    p = len(feats)
    if p < 1:
        return None

    k_real, k_inj, F = BASE._choose_width_spec(
        p,
        rng,
        feature_drop_every_batch=True,
        drop_min_frac=drop_min_frac,
        drop_max_frac=drop_max_frac,
        synth_inj_every_batch=True,
        inj_min_frac=inj_min_frac,
        inj_max_frac=inj_max_frac,
    )

    if n_ctx_for_shape is not None and (n_rows, F) in n_ctx_for_shape:
        n_ctx = int(n_ctx_for_shape[(n_rows, F)])
        p_ctx = n_ctx / float(n_rows)
    else:
        p_ctx = float(rng.uniform(0.14, 0.95))
        n_ctx = max(16, min(n_rows - 4, int(round(p_ctx * n_rows))))
        if n_ctx_for_shape is not None:
            n_ctx_for_shape[(n_rows, F)] = n_ctx

    # 4-Mode Stratified Row Crop across the full table
    sel_local = stratified_row_crop_h452(tab_mmap, valid_rows, y_valid, feats, n_rows, rng)
    sub = np.asarray(tab_mmap[valid_rows[sel_local]], dtype=np.float64)
    X_all = sub[:, feats]
    y_all = sub[:, tcol].copy()

    # 100% Fresh CSPRNG Context/Query Split Every Episode:
    #   60% y-quantile stratified | 25% uniform random | 15% soft oblique covariate-drift split
    u_sp_mode = float(rng.random())
    if u_sp_mode < 0.60 and n_rows >= 28 and np.all(np.isfinite(y_all)):
        ord_s = np.argsort(y_all + 1e-12 * max(float(np.nanstd(y_all)), 1.0) * rng.standard_normal(n_rows), kind="mergesort")
        cp, qp = [], []
        for b_arr in np.array_split(ord_s, int(rng.integers(5, min(17, max(6, n_rows // 4))))):
            bp = rng.permutation(b_arr)
            k_c = int(round(p_ctx * len(bp)))
            cp.append(bp[:k_c])
            qp.append(bp[k_c:])
        c_idx, q_idx = rng.permutation(np.concatenate(cp)), rng.permutation(np.concatenate(qp))
        if len(c_idx) >= n_ctx:
            order = np.concatenate([c_idx[:n_ctx], c_idx[n_ctx:], q_idx])
        else:
            need_c = n_ctx - len(c_idx)
            order = np.concatenate([c_idx, q_idx[:need_c], q_idx[need_c:]])
        X_all, y_all = X_all[order], y_all[order]
    elif u_sp_mode >= 0.85 and X_all.shape[1] >= 1 and n_rows >= 28:
        j0 = int(rng.integers(0, X_all.shape[1]))
        col0 = X_all[:, j0]
        fin0 = np.isfinite(col0)
        if int(np.sum(fin0)) >= 8:
            med0 = float(np.median(col0[fin0]))
            sd0 = max(float(np.std(col0[fin0])), 1e-9)
            z0 = np.clip((np.where(fin0, col0, med0) - med0) / sd0, -4.0, 4.0)
            logits = float(rng.uniform(-0.65, 0.65)) * z0
            logits -= float(np.max(logits))
            w_c = np.exp(logits)
            w_c /= float(np.sum(w_c))
            c_idx = rng.choice(n_rows, size=n_ctx, replace=False, p=w_c)
            q_idx = rng.permutation(np.setdiff1d(np.arange(n_rows, dtype=np.int64), c_idx))
            order = np.concatenate([rng.permutation(c_idx), q_idx])
            X_all, y_all = X_all[order], y_all[order]
        else:
            perm_r = rng.permutation(n_rows)
            X_all, y_all = X_all[perm_r], y_all[perm_r]
    else:
        perm_r = rng.permutation(n_rows)
        X_all, y_all = X_all[perm_r], y_all[perm_r]

    Xc, Xq = X_all[:n_ctx], X_all[n_ctx:]
    yc, yq = y_all[:n_ctx].copy(), y_all[n_ctx:].copy()
    if (tmode < 2 or blend_real_teacher) and float(np.std(yc)) <= 1e-12:
        return None

    Xc, Xq = BASE._select_cols(Xc, yc if tmode < 2 else np.zeros(n_ctx), Xq, k_real, rng)
    if tmode == 2:
        # 55% GP-Winner AST teacher on real covariates, 45% RBF-GP + Oblivious-Tree teacher
        if winner is not None and float(rng.random()) < 0.55:
            t_pair = distill_gp_winner_synth_episode(Xc, Xq, winner, rng)
        else:
            t_pair = BASE._synth_teacher_target_np(Xc, Xq, rng)
        if t_pair is None:
            return None
        if blend_real_teacher:
            yc_s = (yc - float(np.mean(yc))) / max(float(np.std(yc)), 1e-9)
            yq_s = (yq - float(np.mean(yc))) / max(float(np.std(yc)), 1e-9)
            alpha = float(rng.uniform(0.25, 0.70))
            yc = (1.0 - alpha) * yc_s + alpha * t_pair[0]
            yq = (1.0 - alpha) * yq_s + alpha * t_pair[1]
        else:
            yc, yq = t_pair
    elif rng.random() < 0.45:
        sy = float(np.std(yc))
        e = rng.standard_normal(n_rows) / np.sqrt(np.maximum(np.mean(rng.standard_normal((n_rows, 3)) ** 2, axis=1), 1e-3)) / 1.732
        jit = sy * float(rng.uniform(0.008, 0.04)) * np.clip(e, -1.5, 1.5)
        yc, yq = yc + jit[:n_ctx], yq + jit[n_ctx:]

    # Zero-leakage monotone feature warps + MCAR/MAR NaN masks
    Xc, Xq = apply_monotone_warps_and_nans_zero_leakage(Xc, Xq, rng, warp_prob=0.30, nan_prob=0.25)

    # Heavy 18-family zero-leakage synthetic feature injection (`[10%, 45%]`)
    if k_inj > 0:
        Ic, Iq = inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_inj, rng)
        Xc, Xq = np.hstack([Xc, Ic]), np.hstack([Xq, Iq])

    # Wider raw log-uniform scale `exp(U(-3.5, 3.5))` + context-scaled noise `[0.2%, 7%]` + permutation
    p_aug = Xc.shape[1]
    if p_aug > 0:
        extra_mul = np.exp(rng.uniform(-0.5, 0.5, size=(1, p_aug)))
        Xc = np.where(np.isfinite(Xc), Xc * extra_mul, Xc)
        Xq = np.where(np.isfinite(Xq), Xq * extra_mul, Xq)
    Xc, Xq = BASE._augment_features_zero_leakage(Xc, Xq, rng, noise_max_frac=feature_noise_max)

    # Target spaces: 42% 5-fold SVD robust-ridge residual, 33% Yeo-Johnson warp, 25% signed random-scale raw
    u_sp = float(rng.random())
    space = 0
    y_ep = np.concatenate([yc, yq])
    if u_sp < 0.42:
        oof, pq, used, _ = BASE._robust_ridge_np(Xc, yc, Xq, seed=int(rng.integers(1, 1 << 30)))
        if used:
            y_ep = np.concatenate([yc - oof, yq - pq])
            space = 1
    elif u_sp < 0.75:
        yj_pair = BASE._yeojohnson_pair_np(yc, yq)
        if yj_pair is not None and float(np.std(yj_pair[0])) > 1e-8:
            y_ep = np.concatenate([yj_pair[0], yj_pair[1]])
            space = 2
    else:
        y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        y_scale = float(math.exp(rng.uniform(-2.2, 2.2)))
        y_ep = y_sign * y_scale * y_ep

    X_ep = np.vstack([Xc, Xq])
    if X_ep.shape != (n_rows, F) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx])) <= 1e-12:
        return None
    return X_ep, y_ep, n_ctx, (int(t_idx), tmode, int(k_inj), int(F), int(n_rows), space, int(max(0, p - k_real)), int(p))


def _flywheel_worker_loop(
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
    rand_target_prob: float,
    synth_teacher_prob: float,
    feature_noise_max: float,
    postnorm_affine_max: float,
    synth_prob: float,
    gp_winners_jsonl: str,
    gp_stage_prob: float,
    gp_resid_prob: float,
    gp_distill_prob: float,
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
    except (OSError, RuntimeError, ValueError):
        pass

    sampling_mode = "uniform_dataset" if dataset_mix == "uniform" else "smart_stratified"
    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha, sampling_mode=sampling_mode)
    gov = BASE.HostCpuGovernor(hard_pct=cpu_hard, check_every_s=3.0)
    gp_store = GPWinnerStore(gp_winners_jsonl, bank, reload_every_s=30.0)
    prior: Prior | None = None
    valid_tcols_cache: dict[int, tuple[int, ...]] = {}

    open_bins: dict[tuple[int, int], dict] = {}
    n_ctx_for_shape: dict[tuple[int, int], int] = {}
    ready_items: list = []
    allowed_row_bins = [n for n in ROW_BINS_V2 if n <= int(max_episode_rows)] or [min(32, int(max_episode_rows))]
    active_rows = allowed_row_bins[(worker_id * 3) % len(allowed_row_bins)]
    seq = 0
    real_seq = 0

    def _flush_bin(shape_key: tuple[int, int], rng_f: np.random.Generator) -> None:
        b_st = open_bins.pop(shape_key, None)
        n_ctx_for_shape.pop(shape_key, None)
        if not b_st or not b_st["xs"]:
            return
        xs_b, ys_b, infos_b, n_ctx_b = b_st["xs"], b_st["ys"], b_st["infos"], int(b_st["n_ctx"])
        fps_b = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_b, ys_b)]
        ready_items.append(
            BASE._pack_worker_item(
                np.stack(xs_b),
                np.stack(ys_b),
                n_ctx_b,
                infos_b,
                fps_b,
                rng_f,
                postnorm_affine_max=postnorm_affine_max,
            )
        )

    while not stop_ev.is_set():
        gp_store.maybe_reload(force=False)
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
        b_seed = hashlib.blake2b(
            run_nonce + worker_id.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32
        ).digest()
        rng = np.random.default_rng(np.frombuffer(b_seed, dtype=np.uint32))

        if synth_prob > 0.0 and float(rng.random()) < synth_prob:
            if prior is None:
                prior = Prior(torch.device("cpu"), int(prior_seed_base) + 1009 * (worker_id + 1))
            r_py = std_random.Random(int(rng.integers(1, 1 << 62)))
            Fn = max(4, min(BASE.WIDE_GPU, draw_features(r_py)))
            N = min(int(rng.choice(BASE.SYNTH_N, p=BASE.SYNTH_N_P)), int(max_episode_rows))
            B = BASE._target_batch_size(N, Fn, min(int(bmax), 20), tokens_cap, row_tokens_cap, attn_cap)
            n_ctx = max(16, min(N - 4, int(round(float(rng.uniform(0.14, 0.95)) * N))))
            xa, ya = BASE._sample_full_synth_special(
                B,
                N,
                Fn,
                n_ctx,
                rng,
                feature_drop_every_batch=True,
                drop_min_frac=drop_min_frac,
                drop_max_frac=drop_max_frac,
                synth_inj_every_batch=True,
                inj_min_frac=inj_min_frac,
                inj_max_frac=inj_max_frac,
                feature_noise_max=feature_noise_max,
            )
            # Channel C: GP-Winner AST synthetic distillation (`gp_distill_prob = 0.50` default)
            if gp_store.all_winners and float(rng.random()) < gp_distill_prob:
                for b_i in range(B):
                    win = gp_store.all_winners[int(rng.integers(0, len(gp_store.all_winners)))]
                    got_d = distill_gp_winner_synth_episode(xa[b_i, :n_ctx], xa[b_i, n_ctx:], win, rng)
                    if got_d is not None:
                        ya[b_i, :n_ctx], ya[b_i, n_ctx:] = got_d
            fps = [BASE.dataset_fingerprint_128(xa[b_], ya[b_], n_ctx) for b_ in range(xa.shape[0])]
            ready_items.append(
                BASE._pack_worker_item(xa, ya, int(n_ctx), None, fps, rng, postnorm_affine_max=postnorm_affine_max)
            )
            continue

        for _burst_i in range(16):
            if stop_ev.is_set():
                break
            real_seq += 1
            # Rotate `active_rows` every 32 episodes (`8x` faster multi-scale cycling!)
            if real_seq > 1 and (real_seq & 31) == 1:
                old_r = active_rows
                for k_old in [k_o for k_o, st_o in open_bins.items() if k_o[0] == old_r and len(st_o["xs"]) >= 2]:
                    _flush_bin(k_old, rng)
                active_rows = int(rng.choice(allowed_row_bins))

            t_idx, d91_boosted = gp_store.sample_table_idx(
                bank, rng, regret_tilt_prob=regret_tilt_prob, d91_boost_prob=d91_boost_prob
            )
            win_t = gp_store.pick_winner_for_table(t_idx, rng)
            got = None
            for retry_i in range(4):
                got = sample_extreme_stratified_episode_h452(
                    bank,
                    t_idx,
                    rng,
                    valid_tcols_cache,
                    winner=win_t,
                    force_fractional_crop=d91_boosted,
                    max_episode_rows=max_episode_rows,
                    drop_min_frac=drop_min_frac,
                    drop_max_frac=drop_max_frac,
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
            n_r, F_ep = X_ep.shape

            # Channel A (`50%`): Fitted GP-winner symbolic stage injection (strictly context-fitted on `(Xc, yc)`)
            if win_t is not None and F_ep >= 4 and float(rng.random()) < gp_stage_prob:
                k_gp = max(1, min(F_ep // 3, 8))
                Gc, Gq = build_gp_stage_cols_zero_leakage(
                    X_ep[:n_ctx_ep], y_ep[:n_ctx_ep], X_ep[n_ctx_ep:], win_t["stages"], k_gp, rng
                )
                if Gc.shape[1] == k_gp and np.all(np.isfinite(Gc)) and np.all(np.isfinite(Gq)):
                    rep_idx = rng.choice(F_ep, size=k_gp, replace=False)
                    X_ep = X_ep.copy()
                    X_ep[:n_ctx_ep, rep_idx] = Gc
                    X_ep[n_ctx_ep:, rep_idx] = Gq

            # Channel B (`30%`): 5-fold OOF nonlinear GP/tree residual conditioning (`icl_e8_gp_resid_cv7` aligned!)
            if F_ep >= 3 and float(rng.random()) < gp_resid_prob:
                head_b = win_t["head"] if win_t is not None else "oblivious"
                oof_gp, pq_gp, ok_gp = nonlinear_gp_oof_residual_np(
                    X_ep[:n_ctx_ep], y_ep[:n_ctx_ep], X_ep[n_ctx_ep:], head_b, rng
                )
                if ok_gp:
                    X_ep = X_ep.copy()
                    j_anc = int(rng.integers(0, F_ep))
                    X_ep[:n_ctx_ep, j_anc] = oof_gp
                    X_ep[n_ctx_ep:, j_anc] = pq_gp
                    y_ep = np.concatenate([y_ep[:n_ctx_ep] - oof_gp, y_ep[n_ctx_ep:] - pq_gp])

            if not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_ep])) <= 1e-12:
                continue

            sh_key = (int(X_ep.shape[0]), int(X_ep.shape[1]))
            b_st = open_bins.get(sh_key)
            if b_st is None or int(b_st["n_ctx"]) != int(n_ctx_ep):
                if b_st is not None:
                    _flush_bin(sh_key, rng)
                b_tgt = BASE._target_batch_size(sh_key[0], sh_key[1], min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap)
                b_st = {
                    "xs": [],
                    "ys": [],
                    "infos": [],
                    "n_ctx": int(n_ctx_ep),
                    "b_target": b_tgt,
                    "open_seq": real_seq,
                }
                open_bins[sh_key] = b_st
                n_ctx_for_shape[sh_key] = int(n_ctx_ep)

            b_st["xs"].append(X_ep)
            b_st["ys"].append(y_ep)
            b_st["infos"].append(info_ep)
            if len(b_st["xs"]) >= int(b_st["b_target"]):
                _flush_bin(sh_key, rng)

            if (real_seq & 31) == 0 and open_bins:
                stale_keys = [k_s for k_s, st_s in open_bins.items() if (real_seq - int(st_s["open_seq"])) >= 192]
                for k_s in stale_keys:
                    _flush_bin(k_s, rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("S", "M", "L"), default="L")
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--mmap-bank", required=True)
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=96.0)
    ap.add_argument("--lr-cooldown-hours", type=float, default=0.0)
    ap.add_argument("--lr-min", type=float, default=1.0e-6)
    ap.add_argument("--reset-cooldown-start", type=int, choices=(0, 1), default=0)
    ap.add_argument("--gp-winners-jsonl", default="")
    ap.add_argument("--gp-stage-prob", type=float, default=0.50)
    ap.add_argument("--gp-resid-prob", type=float, default=0.30)
    ap.add_argument("--gp-distill-prob", type=float, default=0.50)
    ap.add_argument("--regret-tilt-prob", type=float, default=0.25)
    ap.add_argument("--d91-random-split-boost", type=float, default=0.20)
    ap.add_argument("--save-every-datasets", type=int, default=100_000)
    ap.add_argument("--keep-latest-only", type=int, choices=(0, 1), default=1)
    ap.add_argument("--last-every-s", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--swa-decay", type=float, default=0.9995)
    ap.add_argument("--synth-prob", type=float, default=0.30)
    ap.add_argument("--dataset-mix", choices=("uniform", "smart_stratified"), default="uniform")
    ap.add_argument("--max-episode-rows", type=int, default=BASE.DEFAULT_MAX_EPISODE_ROWS)
    ap.add_argument("--tokens", type=int, default=760_000)
    ap.add_argument("--row-tokens", type=int, default=15_360)
    ap.add_argument("--attn-cap", type=int, default=18_500_000)
    ap.add_argument("--bmax", type=int, default=40)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-row-blocks", type=int, choices=(0, 1), default=1)
    ap.add_argument("--cpu-hard-pct", type=float, default=80.0)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.80)
    ap.add_argument("--random-split-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--feature-drop-min-frac", type=float, default=0.05)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.75)
    ap.add_argument("--synth-inj-every-batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--synth-inj-min-frac", type=float, default=0.10)
    ap.add_argument("--synth-inj-max-frac", type=float, default=0.45)
    ap.add_argument("--rand-target-prob", type=float, default=0.40)
    ap.add_argument("--synth-teacher-prob", type=float, default=0.25)
    ap.add_argument("--feature-noise-max", type=float, default=0.07)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.26)
    ap.add_argument("--seed", type=int, default=452_201)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_p = out_dir / "ckpt_last.pt"
    latest_p = out_dir / "ckpt_latest.pt"
    cd_latest_p = out_dir / "ckpt_cooldown_latest.pt"
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
    ledger = BASE.Ledger128(bank)
    run_nonce = hashlib.blake2b(secrets.token_bytes(32) + int(a.seed).to_bytes(8, "little"), digest_size=32).digest()

    stop_flag = {"stop": False}
    signal.signal(signal.SIGTERM, lambda s, f: stop_flag.__setitem__("stop", True))
    signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__("stop", True))

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
    ema_cooldown = ICLH450MPS(a.size).to(dev).eval()
    for m_ref in (ema, ema_swa, ema_cooldown):
        for p in m_ref.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-2)
    state = {
        "step": 0,
        "micro_batches": 0,
        "accum_in_group": 0,
        "real_batches": 0,
        "synth_batches": 0,
        "datasets": 0,
        "cell_tokens": 0,
        "oom_skips": 0,
        "nonfinite_skips": 0,
        "train_s": 0.0,
        "epochs_saved": 0,
        "last_epoch_datasets": 0,
        "resumes": 0,
        "switch_history": [],
    }

    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
        ema_swa.load_state_dict(ck.get("ema_swa") or ck["ema"], strict=True)
        ema_cooldown.load_state_dict(ck.get("ema_cooldown") or ck.get("ema_swa") or ck["ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        state.setdefault("last_epoch_datasets", int(state["datasets"]))
        if not isinstance(state.get("switch_history"), list):
            state["switch_history"] = []
        state["resumes"] += 1
        ledger.load(ck["ledger"])
    else:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        w_init = ck.get("ema") or ck["model"]
        for m_ref in (model, ema, ema_swa, ema_cooldown):
            m_ref.load_state_dict(w_init, strict=True)

    if a.lr_cooldown_hours > 0.0 and ("cooldown_start_s" not in state or bool(a.reset_cooldown_start)):
        state["cooldown_start_s"] = float(state["train_s"])
        state["cooldown_hours"] = float(a.lr_cooldown_hours)
        state["cooldown_lr_start"] = float(a.lr)
        state["cooldown_lr_min"] = float(a.lr_min)

    sampler_cfg = {
        "protocol": "h452-v2-extreme-stratified-flywheel-cd" if a.lr_cooldown_hours > 0 else "h452-v2-extreme-stratified-flywheel",
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "resume_step": int(state["step"]),
        "resume_datasets": int(state["datasets"]),
        "lr_base": float(a.lr),
        "lr_cooldown_hours": float(a.lr_cooldown_hours),
        "lr_min": float(a.lr_min),
        "gp_winners_jsonl": str(a.gp_winners_jsonl),
        "gp_stage_prob": float(a.gp_stage_prob),
        "gp_resid_prob": float(a.gp_resid_prob),
        "gp_distill_prob": float(a.gp_distill_prob),
        "regret_tilt_prob": float(a.regret_tilt_prob),
        "d91_random_split_boost": float(a.d91_random_split_boost),
        "drop_frac": [float(a.feature_drop_min_frac), float(a.feature_drop_max_frac)],
        "synth_inj_frac": [float(a.synth_inj_min_frac), float(a.synth_inj_max_frac)],
        "synth_prob": float(a.synth_prob),
        "synth_teacher_prob": float(a.synth_teacher_prob),
        "rand_target_prob": float(a.rand_target_prob),
    }
    state["switch_history"].append(sampler_cfg)

    params_cnt = int(n_params(model))
    prior_seed_base = int(a.seed) + 17 + 1000 * int(state["resumes"])
    gov = BASE.HostCpuGovernor(hard_pct=a.cpu_hard_pct, check_every_s=5.0)

    ctx = mp.get_context("spawn")
    ep_q = ctx.Queue(maxsize=max(32, 8 * a.workers))
    stop_ev = ctx.Event()
    procs = [
        ctx.Process(
            target=_flywheel_worker_loop,
            args=(
                w,
                run_nonce,
                str(a.mmap_bank),
                a.bank_sha256,
                a.dataset_mix,
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
                float(a.rand_target_prob),
                float(a.synth_teacher_prob),
                float(a.feature_noise_max),
                float(a.postnorm_affine_max),
                float(a.synth_prob),
                str(a.gp_winners_jsonl),
                float(a.gp_stage_prob),
                float(a.gp_resid_prob),
                float(a.gp_distill_prob),
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

    def save_latest_eval_ckpt() -> str:
        swa_cpu = {
            kk: vv.cpu()
            for kk, vv in (ema_cooldown if a.lr_cooldown_hours > 0 else ema_swa).state_dict().items()
        }
        BASE._atomic_save(
            {"ema": swa_cpu, "model": swa_cpu, "state": dict(state), "size": a.size, "sampler_cfg": sampler_cfg},
            latest_p,
            mode=0o644,
        )
        if a.lr_cooldown_hours > 0:
            BASE._atomic_save(
                {"ema": swa_cpu, "model": swa_cpu, "state": dict(state), "size": a.size, "sampler_cfg": sampler_cfg},
                cd_latest_p,
                mode=0o644,
            )
        return sha256_file(latest_p)

    def save_last() -> None:
        BASE._atomic_save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                "ema_swa": {k: v.cpu() for k, v in ema_swa.state_dict().items()},
                "ema_cooldown": {k: v.cpu() for k, v in ema_cooldown.state_dict().items()},
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
                "bank_load_s": bank_load_s,
                "train_s": state["train_s"],
                "step": state["step"],
                "lr_base": a.lr,
                "lr_cooldown_hours": a.lr_cooldown_hours,
                "lr_min": a.lr_min,
                "sampler_cfg": sampler_cfg,
            }
        )
        + "\n"
    )

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
                k_ep = int(state["epochs_saved"]) + 1
                ck_sha = save_latest_eval_ckpt()
                jp = out_dir / f"epoch_{k_ep:02d}.json"
                if not jp.exists():
                    jp.write_text(
                        json.dumps(
                            {
                                "schema": "jope.h452.epoch/v2",
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

            if a.lr_cooldown_hours > 0.0:
                cd_start = float(state.get("cooldown_start_s", s_seg0))
                cd_frac = min(1.0, max(0.0, (state["train_s"] - cd_start) / (a.lr_cooldown_hours * 3600.0)))
                cur_lr = float(a.lr_min) + (float(a.lr) - float(a.lr_min)) * 0.5 * (1.0 + math.cos(math.pi * cd_frac))
            else:
                frac = min(1.0, state["train_s"] / budget_s)
                lr_mult = (
                    (frac / 0.04)
                    if frac < 0.04
                    else (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * (frac - 0.04) / 0.96)))
                )
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
                zm_np, yz_np, ym_np, ys_np, yq_np = (
                    zm_np[keep_idx],
                    yz_np[keep_idx],
                    ym_np[keep_idx],
                    ys_np[keep_idx],
                    yq_np[keep_idx],
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
            else:
                zm_t, yz_t, ym_t, ys_t, yq_t = (
                    torch.from_numpy(zm_np),
                    torch.from_numpy(yz_np),
                    torch.from_numpy(ym_np),
                    torch.from_numpy(ys_np),
                    torch.from_numpy(yq_np),
                )

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
                    cd_d = min(0.9985, (1.0 + steps_since_start) / (10.0 + steps_since_start))
                    for pe, ps, pc, pm in zip(
                        ema.parameters(), ema_swa.parameters(), ema_cooldown.parameters(), model.parameters()
                    ):
                        pe.lerp_(pm, 1.0 - decay)
                        ps.lerp_(pm, 1.0 - swa_d)
                        pc.lerp_(pm, 1.0 - cd_d)
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
                            "loss": round(run_loss / run_n, 5),
                            "lr": round(cur_lr, 9),
                            "datasets": state["datasets"],
                            "ds_per_s": round(run_ds / dt, 3),
                            "mb_per_s": round(run_n / dt, 3),
                            "real_batches": state["real_batches"],
                            "synth_batches": state["synth_batches"],
                            "real_eps": s["real_episodes"],
                            "d91_share": s["dev91_share_of_real_episodes"],
                            "unique_datasets": s["unique_datasets_seen"],
                            "datasets_seen": s["datasets_seen"],
                            "dup_fed": s["duplicate_datasets_fed_to_model"],
                            "dup_filtered": s["duplicates_filtered_pre_forward"],
                            "src_tables_seen": s["distinct_source_tables_seen"],
                            "host_cpu_pct": round(host_cpu_pct, 1),
                            "oom": state["oom_skips"],
                            "mem_gb": cuda_gb,
                            "protocol": sampler_cfg["protocol"],
                        }
                    )
                    + "\n"
                )
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
                "lr": round(cur_lr, 9),
            }
        )
        + "\n"
    )
    log_f.close()


if __name__ == "__main__":
    main()
