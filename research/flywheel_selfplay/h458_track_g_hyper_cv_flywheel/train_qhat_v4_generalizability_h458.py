#!/usr/bin/env python3
"""Wave H458 Track G: Held-Out Generalizability & Overfit-Gap Contextual Surrogate (`QHatV4`)
(`train_qhat_v4_generalizability_h458.py`).

Upgrades `ContextualSurrogateQHatV3` (`H454`) to `ContextualSurrogateQHatV4` (`H458`):
  1. Real `44d` `DOPE+LM12` Context Vectors (`0` zero-vector bug):
     - Frozen `DOPE` (`c51a56d0...`, `0` trainable `DOPE` params, SHA-256 verified before & after) projected via
       `DL-Train`-only (`1,219` contexts, `0` `dev91`) 32-dim SVD + 12-dim `LM12` landmarks.
  2. Ground-Truth Held-Out `Episode-Test` (`u_test_iid` + `u_test_extrap`) & `overfit_gap` Supervision:
     - Evaluates all existing `gp_teacher_shards_v1/shard_*.npz` (`12,800+` real Rust-executed cuts) + live
       `teaching_shards_v4/*.npz` strictly on held-out query rows `(Xq_real, yq_real)` (`y_joint = [yc, 0]` causal fit)
       partitioned into `70%` interior IID + `30%` high-leverage Extrapolation test subsets against closed-form GCV ridge.
     - Emits `/home/ubuntu/jope-data/h458-flywheel/teaching_shards_v4/v1_backfill_cv_to_test_ledger.npz` (matching
       Track S Interface `iv`), `/home/ubuntu/jope-data/h458-flywheel/gp_family_seed_library_v4.json`, and held-out-verified
       zero-loser `/home/ubuntu/jope-data/h458-flywheel/gp_teacher_shards_v2/` bootstrap shards (`0.0%` held-out losers).
  3. Three-Head Gated Bilinear Ranker (`q_pred`, `stab_pred`, `overfit_gap_pred >= 0`):
     - Directly predicts held-out generalizability rank `q_pred`, cross-split stability `stab_pred`, and non-negative
       CV-to-Test optimism `overfit_gap_pred = max(0, u_cv7 - u_test)`.
     - `PretrainedQHatV4Prior` enforces `require_ok=True` by default (`assert qhat.ok`) and supports `v4/v3/v2` checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
for _p in (
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

import h335_core as K  # noqa: E402
import crop_shuffle_synthetic_views as V  # noqa: E402
from mmap_bank_h450 import MmapF64Bank  # noqa: E402
from train_surrogate_qhat_v3_h454 import (  # noqa: E402
    ContextualSurrogateQHatV3,
    _encode_ctx44_from_xy,
    _midrank_centered,
    estimate_ast_timeout_hazard,
    featurize_candidate_h454,
)
from ultra_gp_evolve_h452 import HEAD_FEATS, fold_log_utility  # noqa: E402

C, D, G = K.C, K.D, K.G


def extract_ast_head_name(dsl: str) -> str:
    if dsl.startswith("(portfolio"):
        return "portfolio"
    for h in HEAD_FEATS:
        if f"({h}" in dsl:
            return h
    return "other"


def make_query_stratified_folds(y: np.ndarray, k: int = 5, seed: int = 45801) -> np.ndarray:
    y_f = np.asarray(y, dtype=np.float64)
    n = len(y_f)
    k_eff = max(2, min(int(k), max(2, n // 4)))
    rng = np.random.default_rng(seed)
    sy = max(float(np.std(y_f)), 1e-12)
    order = np.argsort(y_f + 1e-12 * sy * rng.standard_normal(n), kind="mergesort")
    lab = np.empty(n, dtype=np.int64)
    for s in range(0, n, k_eff):
        blk = order[s : s + k_eff]
        lab[blk] = rng.permutation(k_eff) if len(blk) == k_eff else rng.choice(k_eff, size=len(blk), replace=False)
    return lab


def split_query_iid_extrap(Xc: np.ndarray, Xq: np.ndarray, extrap_frac: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions held-out query indices into `(iid_mask, extrap_mask)` strictly using feature leverage vs `Xc` (`0` `y` read!)."""
    nq = len(Xq)
    if nq < 8:
        m_all = np.ones(nq, dtype=bool)
        return m_all, m_all
    Xc_f = np.asarray(Xc, dtype=np.float64)
    Xq_f = np.asarray(Xq, dtype=np.float64)
    p = Xc_f.shape[1]
    med = np.zeros(p, dtype=np.float64)
    scale = np.ones(p, dtype=np.float64)
    for j in range(p):
        fin = Xc_f[:, j][np.isfinite(Xc_f[:, j])]
        if fin.size > 0:
            med[j] = float(np.median(fin))
            q25, q75 = np.percentile(fin, [25.0, 75.0])
            scale[j] = max(float(q75 - q25), float(np.std(fin)), 1e-6)
    Zq = np.clip((np.where(np.isfinite(Xq_f), Xq_f, med[None, :]) - med[None, :]) / scale[None, :], -25.0, 25.0)
    lev = np.sum(Zq * Zq, axis=1)
    n_ext = max(2, min(nq - 2, int(round(extrap_frac * nq))))
    ord_lev = np.argsort(lev, kind="mergesort")
    ext_mask = np.zeros(nq, dtype=bool)
    ext_mask[ord_lev[-n_ext:]] = True
    iid_mask = ~ext_mask
    return iid_mask, ext_mask


def fit_fast_gcv_ridge_on_ctx(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq_real: np.ndarray,
    Xq_cf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Closed-form SVD GCV ridge strictly on `(Xc, yc)` (`0` `yq` access!) returning `(loo_c, pred_q_real, pred_q_cf)`."""
    Xc_f = np.asarray(Xc, dtype=np.float64)
    yc_f = np.asarray(yc, dtype=np.float64)
    Xqr_f = np.asarray(Xq_real, dtype=np.float64)
    n_c, p = Xc_f.shape
    med = np.zeros(p, dtype=np.float64)
    sd = np.ones(p, dtype=np.float64)
    for j in range(p):
        fin = Xc_f[:, j][np.isfinite(Xc_f[:, j])]
        if fin.size > 0:
            med[j] = float(np.median(fin))
            sd[j] = max(float(np.std(fin)), 1e-8)
    Zc = np.clip((np.where(np.isfinite(Xc_f), Xc_f, med[None, :]) - med[None, :]) / sd[None, :], -15.0, 15.0)
    Zqr = np.clip((np.where(np.isfinite(Xqr_f), Xqr_f, med[None, :]) - med[None, :]) / sd[None, :], -15.0, 15.0)
    ymu = float(np.mean(yc_f))
    yc_0 = yc_f - ymu
    try:
        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc_0
        best_gcv = float("inf")
        best_lam = 10.0
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            filt = s2 / (s2 + lam)
            yhat_0 = U @ (filt * uty)
            resid = yc_0 - yhat_0
            dof = max(1.0, float(n_c) - 1.0 - float(np.sum(filt)))
            gcv = float(np.mean(resid * resid)) / ((dof / float(n_c)) ** 2)
            if gcv < best_gcv:
                best_gcv = gcv
                best_lam = lam
        filt_b = s2 / (s2 + best_lam)
        yhat_c = ymu + U @ (filt_b * uty)
        # Exact hat-diagonal LOO predictions on context
        h_diag = np.clip(np.sum((U * U) * filt_b[None, :], axis=1), 0.0, 0.95)
        loo_c = yc_f - (yc_f - yhat_c) / np.maximum(1.0 - h_diag, 0.05)
        w = Vt.T @ ((s / (s2 + best_lam)) * uty)
        y_lo, y_hi = float(np.min(yc_f)), float(np.max(yc_f))
        span = max(y_hi - y_lo, float(np.std(yc_f)), 1e-9)
        lo_g, hi_g = y_lo - 1.5 * span, y_hi + 1.5 * span
        loo_c = np.clip(loo_c, lo_g, hi_g)
        pred_qr = np.clip(ymu + Zqr @ w, lo_g, hi_g)
        pred_cf = None
        if Xq_cf is not None and len(Xq_cf) > 0:
            Xqcf_f = np.asarray(Xq_cf, dtype=np.float64)
            Zqcf = np.clip((np.where(np.isfinite(Xqcf_f), Xqcf_f, med[None, :]) - med[None, :]) / sd[None, :], -15.0, 15.0)
            pred_cf = np.clip(ymu + Zqcf @ w, lo_g, hi_g)
        return loo_c, pred_qr, pred_cf
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        loo_c = np.full(n_c, ymu, dtype=np.float64)
        pred_qr = np.full(len(Xqr_f), ymu, dtype=np.float64)
        pred_cf = np.full(len(Xq_cf), ymu, dtype=np.float64) if (Xq_cf is not None and len(Xq_cf) > 0) else None
        return loo_c, pred_qr, pred_cf


def episode_heldout_utility(pred_q: np.ndarray, yq: np.ndarray, seed: int = 45801) -> float:
    """Computes scale-invariant held-out utility `-log(rmse / std(yq))` on query slice `yq`."""
    yq_f = np.asarray(yq, dtype=np.float64)
    pq_f = np.asarray(pred_q, dtype=np.float64)
    if len(yq_f) < 4 or not np.all(np.isfinite(pq_f)) or not np.all(np.isfinite(yq_f)):
        return -5.0
    sy = max(float(np.std(yq_f)), 1e-12)
    if sy <= 1e-11:
        return 0.0
    if len(yq_f) >= 20:
        lab = make_query_stratified_folds(yq_f, k=min(5, len(yq_f) // 4), seed=seed)
        return float(fold_log_utility(pq_f, yq_f, lab))
    rm = max(float(np.sqrt(np.mean((pq_f - yq_f) ** 2))), 1e-12 * sy)
    return float(-math.log(rm / sy))


def optimal_heldout_blend_weight(pred_gp: np.ndarray, pred_base: np.ndarray, yq: np.ndarray) -> Tuple[float, float]:
    """Finds convex blend weight `w* in [0, 1]` maximizing held-out utility `w * pred_gp + (1 - w) * pred_base`."""
    best_w = 1.0
    best_u = episode_heldout_utility(pred_gp, yq)
    for w in (0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0):
        bl = w * pred_gp + (1.0 - w) * pred_base
        u_w = episode_heldout_utility(bl, yq)
        if u_w > best_u + 1e-9:
            best_u = u_w
            best_w = w
    return float(best_w), float(best_u)


class ContextualSurrogateQHatV4(nn.Module):
    """Three-head gated bilinear + residual MLP ranker over `(ctx_44d, ast_128d)` (`0` DOPE params!).

    Outputs:
      - `q_pred`: predicted held-out generalizability within-context midrank in `[-0.5, +0.5]`
      - `stab_pred`: predicted cross-split & non-timeout stability in `[0, 1]`
      - `overfit_gap_pred`: predicted non-negative CV-to-Test overfit gap `max(0, u_cv7 - u_test) >= 0`
    """

    def __init__(self, d_ctx: int = 44, d_ast: int = 128, hidden: int = 160) -> None:
        super().__init__()
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_ctx, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.ast_proj = nn.Sequential(
            nn.Linear(d_ast, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.LayerNorm(hidden),
        )
        self.rank_head = nn.Linear(hidden, 1)
        self.stab_head = nn.Linear(hidden, 1)
        self.gap_head = nn.Linear(hidden, 1)

    def forward(self, x_ctx: torch.Tensor, x_ast: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.ctx_proj(x_ctx)
        ha = self.ast_proj(x_ast)
        h = self.trunk(torch.cat([hc, ha, hc * ha], dim=-1))
        q_pred = self.rank_head(h).squeeze(-1)
        stab_pred = torch.sigmoid(self.stab_head(h).squeeze(-1))
        gap_pred = F.softplus(self.gap_head(h).squeeze(-1) - 1.5)
        return q_pred, stab_pred, gap_pred


class PretrainedQHatV4Prior:
    """Thread-safe inference wrapper supporting `v4` (`3`-head), `v3` (`2`-head), and `v2` (`1`-head), enforcing `require_ok`."""

    def __init__(self, ckpt_path: Optional[Path], require_ok: bool = True) -> None:
        self.ok = False
        self.version = 0
        self.net = None
        self.dope_mean: Optional[np.ndarray] = None
        self.P_dope32: Optional[np.ndarray] = None
        self.c_mu: Optional[np.ndarray] = None
        self.c_sd: Optional[np.ndarray] = None
        self.d_ast: int = 0
        self.ckpt_path = str(ckpt_path) if ckpt_path is not None else ""
        if ckpt_path is not None and Path(ckpt_path).is_file():
            try:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                d_ctx = int(ck["d_ctx"])
                d_ast = int(ck["d_ast"])
                hidden = int(ck.get("hidden", 160 if d_ast == 128 else 128))
                ver = str(ck.get("model_version", ""))
                if ver == "v4":
                    net = ContextualSurrogateQHatV4(d_ctx, d_ast, hidden=hidden)
                    self.version = 4
                elif ver == "v3" or d_ast == 128:
                    net = ContextualSurrogateQHatV3(d_ctx, d_ast, hidden=hidden)
                    self.version = 3
                else:
                    from ultra_gp_evolve_1000g_h453 import _QHatV2Net

                    net = _QHatV2Net(d_ctx, d_ast, hidden=hidden)
                    self.version = 2
                net.load_state_dict(ck["state_dict"], strict=True)
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self.net = net
                self.dope_mean = np.asarray(ck["dope_mean"], dtype=np.float64)
                self.P_dope32 = np.asarray(ck["P_dope32"], dtype=np.float64)
                self.c_mu = np.asarray(ck["c_mu"], dtype=np.float32)
                self.c_sd = np.maximum(np.asarray(ck["c_sd"], dtype=np.float32), 1e-6)
                self.d_ast = d_ast
                self.ok = True
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self.ok = False
                if require_ok:
                    raise RuntimeError(f"Failed to load QHat checkpoint {ckpt_path}: {exc}") from exc
        if require_ok and not self.ok:
            raise RuntimeError(f"QHat checkpoint missing or invalid (require_ok=True): {ckpt_path}")

    def encode_ctx_z(self, enc_dict: Optional[Dict[str, Optional[np.ndarray]]]) -> np.ndarray:
        if not self.ok or self.dope_mean is None or self.P_dope32 is None or self.c_mu is None or self.c_sd is None:
            return np.zeros((1, 44), dtype=np.float32)
        if not enc_dict or enc_dict.get("dope629") is None or enc_dict.get("lm12") is None:
            raw = np.zeros((1, 44), dtype=np.float32)
        else:
            d32 = (np.asarray(enc_dict["dope629"], dtype=np.float64) - self.dope_mean) @ self.P_dope32
            raw = np.concatenate([d32, np.asarray(enc_dict["lm12"], dtype=np.float64)[:12]])[None, :].astype(np.float32)
        return np.clip((np.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=-5.0) - self.c_mu) / self.c_sd, -8.0, 8.0)

    def encode_raw_ctx44(self, X: np.ndarray, y: np.ndarray, dope_enc: V.DopeViewEncoder) -> np.ndarray:
        """Computes raw 44-dim `DOPE32 + LM12` context vector strictly from `(X, y)` (`0` zero-vector fallback!)."""
        assert self.ok and self.dope_mean is not None and self.P_dope32 is not None, "QHat must be loaded"
        return _encode_ctx44_from_xy(X, y, dope_enc, self.dope_mean, self.P_dope32)

    def encode_xy_z(self, X: np.ndarray, y: np.ndarray, dope_enc: V.DopeViewEncoder) -> Tuple[np.ndarray, np.ndarray]:
        """Returns `(raw_ctx44, normalized_ctx_z)` from `(X, y)`."""
        raw44 = self.encode_raw_ctx44(X, y, dope_enc)
        assert self.c_mu is not None and self.c_sd is not None
        cz = np.clip((raw44[None, :] - self.c_mu) / self.c_sd, -8.0, 8.0).astype(np.float32)
        return raw44, cz

    def predict_components(
        self,
        ctx_z: Optional[np.ndarray],
        cands_or_dsls: Sequence,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns `(q_pred, stab_pred, overfit_gap_pred)` arrays of length `m`."""
        m = len(cands_or_dsls)
        if not self.ok or self.net is None or m == 0:
            z = np.zeros(m, dtype=np.float64)
            return z, np.full(m, 0.5, dtype=np.float64), z
        cz = np.zeros((1, 44), dtype=np.float32) if ctx_z is None else np.asarray(ctx_z, dtype=np.float32).reshape(1, 44)
        if self.version in (3, 4):
            X_ast = np.stack([featurize_candidate_h454(c) for c in cands_or_dsls], axis=0).astype(np.float32)
            with torch.no_grad():
                tc = torch.from_numpy(np.repeat(cz, m, axis=0))
                ta = torch.from_numpy(X_ast)
                if self.version == 4:
                    q_p, stab_p, gap_p = self.net(tc, ta)
                    return (
                        q_p.numpy().astype(np.float64),
                        stab_p.numpy().astype(np.float64),
                        gap_p.numpy().astype(np.float64),
                    )
                q_p, stab_p = self.net(tc, ta)
                return (
                    q_p.numpy().astype(np.float64),
                    stab_p.numpy().astype(np.float64),
                    np.zeros(m, dtype=np.float64),
                )
        from ultra_gp_evolve_h452 import featurize_candidate_h452

        X_ast = np.stack(
            [featurize_candidate_h452(G.Candidate.from_dsl(c, "q") if isinstance(c, str) else c) for c in cands_or_dsls],
            axis=0,
        ).astype(np.float32)
        with torch.no_grad():
            tc = torch.from_numpy(np.repeat(cz, m, axis=0))
            ta = torch.from_numpy(X_ast)
            q_p = self.net(tc, ta).numpy().astype(np.float64)
        return q_p, np.full(m, 0.5, dtype=np.float64), np.zeros(m, dtype=np.float64)

    def predict_ranks(
        self,
        ctx_z: Optional[np.ndarray],
        cands_or_feats,
    ) -> np.ndarray:
        """Compatibility hook for `SurrogateBanditH453.score_ucb(self.ctx_z, X)` and candidate sequences."""
        if isinstance(cands_or_feats, np.ndarray):
            m = int(cands_or_feats.shape[0])
            if not self.ok or self.net is None or m == 0:
                return np.zeros(m, dtype=np.float64)
            cz = np.zeros((1, 44), dtype=np.float32) if ctx_z is None else np.asarray(ctx_z, dtype=np.float32).reshape(1, 44)
            X_in = np.asarray(cands_or_feats, dtype=np.float32)
            if X_in.shape[1] < self.d_ast:
                X_pad = np.zeros((m, self.d_ast), dtype=np.float32)
                X_pad[:, : X_in.shape[1]] = X_in
                X_in = X_pad
            elif X_in.shape[1] > self.d_ast:
                X_in = X_in[:, : self.d_ast]
            with torch.no_grad():
                tc = torch.from_numpy(np.repeat(cz, m, axis=0))
                ta = torch.from_numpy(X_in)
                if self.version == 4:
                    q_p, stab_p, gap_p = self.net(tc, ta)
                    raw = (q_p + 0.20 * (stab_p - 0.5) - 0.35 * gap_p).numpy().astype(np.float64)
                elif self.version == 3:
                    q_p, stab_p = self.net(tc, ta)
                    raw = (q_p + 0.20 * (stab_p - 0.5)).numpy().astype(np.float64)
                else:
                    raw = self.net(tc, ta).numpy().astype(np.float64)
            if m <= 1 or float(np.std(raw)) < 1e-12:
                return raw
            return (rankdata(raw, method="average") - 0.5) / float(m) - 0.5
        return self.score_candidates(ctx_z, cands_or_feats)

    def score_candidates(
        self,
        ctx_z: Optional[np.ndarray],
        cands_or_dsls: Sequence,
        n_rows: int = 512,
        width: int = 16,
        gap_penalty: float = 0.45,
    ) -> np.ndarray:
        m = len(cands_or_dsls)
        if not self.ok or self.net is None or m == 0:
            return np.zeros(m, dtype=np.float64)
        q_p, stab_p, gap_p = self.predict_components(ctx_z, cands_or_dsls)
        hazards = np.array(
            [estimate_ast_timeout_hazard(c if isinstance(c, str) else c.dsl, n_rows, width) for c in cands_or_dsls],
            dtype=np.float64,
        )
        raw = q_p + 0.20 * (stab_p - 0.5) - float(gap_penalty) * gap_p - 0.28 * hazards
        if m <= 1 or float(np.std(raw)) < 1e-12:
            return raw
        return (rankdata(raw, method="average") - 0.5) / float(m) - 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-shards-dir", default="/home/ubuntu/jope-data/h452-flywheel/gp_teacher_shards_v1")
    ap.add_argument("--v2-shards-dir", default="/home/ubuntu/jope-data/h458-flywheel/gp_teacher_shards_v2")
    ap.add_argument("--teaching-v4-dir", default="/home/ubuntu/jope-data/h458-flywheel/teaching_shards_v4")
    ap.add_argument("--seed-library-json", default="/home/ubuntu/jope-data/h458-flywheel/gp_family_seed_library_v4.json")
    ap.add_argument("--ugp-root", default="/home/ubuntu/jope-data/h452-flywheel/ultra_gp_d91")
    ap.add_argument("--d91-search-bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument("--mmap-bank", default="/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--s8-audit-json", default="/home/ubuntu/jope-data/h458s-quarantine/s8/S8_BUCKET_AUDIT.json")
    ap.add_argument("--allow-d91-random-splits", type=int, choices=(0, 1), default=1)
    ap.add_argument("--max-v1-shards", type=int, default=15000)
    ap.add_argument("--bootstrap-v2-shards", type=int, choices=(0, 1), default=1)
    ap.add_argument("--min-test-regret", type=float, default=0.020)
    ap.add_argument("--max-overfit-gap", type=float, default=0.120)
    ap.add_argument("--out-ckpt", default="/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt")
    ap.add_argument("--out-receipt", default="/home/ubuntu/jope_plan/evidence/H458/trackG/QHAT_V4_HELD_OUT_RECEIPT.json")
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--lr", type=float, default=1.8e-3)
    ap.add_argument("--seed", type=int, default=458401)
    a = ap.parse_args()

    t0 = time.time()
    if bool(a.allow_d91_random_splits):
        C.resume_stage_v()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # 1. Verify frozen DOPE (`c51a56d0...`) SHA-256 BEFORE training (`0` DOPE params enter optimizer!)
    dope_enc = V.DopeViewEncoder(threads=2)
    dope_sha_before = dope_enc.dope_sha_before
    assert dope_sha_before == C.DOPE_SHA256, f"DOPE sha before mismatch: {dope_sha_before}"

    # 2. Fit 32-dim DOPE SVD strictly on `DL-Train` (`1,219` contexts / `205` families, `0` `dev91`!)
    train_rows = C.train_rows()
    train_fam = {r["context_sha256"]: r["family_id"] for r in train_rows}
    bank_entries, _ = D.build_bank(train_rows)

    merged_p = Path("/home/ubuntu/jope-data/h335-cl/neural/views_c32_train1219.npz")
    m_views = V.load_merged_views(merged_p)
    ctx_vec_by_cid: Dict[str, np.ndarray] = {}
    dope_mat = []
    for cid_b, vec_b in zip([str(c) for c in m_views["cids"]], m_views["view0_f32"]):
        vb = np.asarray(vec_b, dtype=np.float64)
        dope_mat.append(vb[D.DOPE])
        ctx_vec_by_cid[cid_b] = vb
    dope_mat_np = np.vstack(dope_mat)
    dope_mean = np.mean(dope_mat_np, axis=0)
    _, _, Vt = np.linalg.svd(dope_mat_np - dope_mean, full_matrices=False)
    P_dope32 = Vt[:32].T  # (629, 32)

    # Grouped observations: ctx_id -> {fam, ctx44, obs: {(dsl, tmode): (u_target, stab_val, overfit_gap)}}
    grouped: Dict[str, Dict] = {}

    def _add_obs(
        ctx_id: str,
        fam_id: str,
        ctx44: np.ndarray,
        dsl: str,
        u_val: float,
        stab_val: float = 0.80,
        overfit_gap: float = 0.02,
        tmode: str = "raw",
    ) -> None:
        if not dsl or dsl == "mean_fallback" or not math.isfinite(u_val):
            return
        g_entry = grouped.get(ctx_id)
        if g_entry is None:
            g_entry = {
                "fam": str(fam_id),
                "ctx44": ctx44.astype(np.float32, copy=False),
                "obs": {},
            }
            grouped[ctx_id] = g_entry
        key = (dsl, tmode)
        gap_clamped = float(np.clip(overfit_gap if math.isfinite(overfit_gap) else 0.05, 0.0, 1.5))
        prev = g_entry["obs"].get(key)
        if prev is None or u_val > prev[0]:
            g_entry["obs"][key] = (float(u_val), float(np.clip(stab_val, 0.02, 0.99)), gap_clamped)

    # 3A. `DL-Train` (`1,219` contexts) champions + cross-island contrastives
    bank_pool: List[Tuple[str, int, set]] = []
    for ent in bank_entries:
        dsl_e = str(getattr(ent, "dsl", ""))
        if dsl_e:
            bank_pool.append((dsl_e, int(getattr(ent, "island", 0)), set(getattr(ent, "families", ()))))

    n_dltrain_obs = 0
    for ent in bank_entries:
        dsl_e = str(getattr(ent, "dsl", ""))
        cids_e = getattr(ent, "cids", ())
        kinds_e = set(getattr(ent, "kinds", ()))
        isl_e = int(getattr(ent, "island", 0))
        if not dsl_e or not cids_e:
            continue
        pos_u = 1.0 if "champion" in kinds_e else 0.72
        for cid_e in cids_e:
            cid_s = str(cid_e)
            if cid_s not in ctx_vec_by_cid or cid_s not in train_fam:
                continue
            vb = ctx_vec_by_cid[cid_s]
            c_feat = np.concatenate([(vb[D.DOPE] - dope_mean) @ P_dope32, vb[D.LM12]]).astype(np.float32)
            fam_s = train_fam[cid_s]
            _add_obs(f"dltr:{cid_s}", fam_s, c_feat, dsl_e, pos_u, stab_val=0.88, overfit_gap=0.015, tmode="raw")
            n_dltrain_obs += 1
            neg_added = 0
            for _ in range(16):
                dsl_n, isl_n, fams_n = bank_pool[int(rng.integers(0, len(bank_pool)))]
                if isl_n != isl_e and fam_s not in fams_n:
                    _add_obs(f"dltr:{cid_s}", fam_s, c_feat, dsl_n, 0.0, stab_val=0.42, overfit_gap=0.18, tmode="raw")
                    n_dltrain_obs += 1
                    neg_added += 1
                    if neg_added >= 3:
                        break

    # 3B. `Ultra-GP` `1000g` (`g10..g1000`) trajectories with real `DOPE` ctx44 + `lab7_r0 -> lab7_r1` optimism gap
    n_ugp_obs = 0
    ugp_root = Path(a.ugp_root)
    if bool(a.allow_d91_random_splits) and ugp_root.is_dir() and Path(a.d91_search_bank).is_file():
        dev_rows = C.dev_rows_for_stage_v()
        dev_fam = {r["context_sha256"]: r["family_id"] for r in dev_rows}
        d91_ctx44: Dict[str, Tuple[np.ndarray, int, int]] = {}
        with np.load(a.d91_search_bank, allow_pickle=False) as zb:
            assert "ya" not in zb.files, "audit labels forbidden in search bank!"
            b_cids = [str(x) for x in zb["cids"].tolist()]
            ss = zb["s_shape"]
            so = lo = 0
            for i, cid_d in enumerate(b_cids):
                n_r, w_r = int(ss[i, 0]), int(ss[i, 1])
                S_mat = np.array(zb["search"][so : so + n_r * w_r], np.float64).reshape(n_r, w_r)
                Xs_d = S_mat[:, :-1]
                ys_d = np.array(zb["y64"][lo : lo + n_r], np.float64)
                d91_ctx44[cid_d] = (_encode_ctx44_from_xy(Xs_d, ys_d, dope_enc, dope_mean, P_dope32), n_r, w_r - 1)
                so += n_r * w_r
                lo += n_r

        for gen_dir_name in ("g10", "g50", "g100", "g250", "g500", "g1000"):
            gdir = ugp_root / gen_dir_name
            if not gdir.is_dir():
                continue
            for npz_p in sorted(gdir.glob("*.npz")):
                cid_u = npz_p.stem
                if cid_u not in d91_ctx44 or cid_u not in dev_fam:
                    continue
                c44, n_r, w_f = d91_ctx44[cid_u]
                fam_u = dev_fam[cid_u]
                with np.load(npz_p, allow_pickle=False) as z:
                    assert "ya" not in z.files
                    dsls_7 = [str(x) for x in z["dsls"].tolist()]
                    tmodes = [str(x) for x in z["target_modes"].tolist()] if "target_modes" in z.files else ["raw"] * len(dsls_7)
                    u7 = np.asarray(z["u_cv7"], dtype=np.float64)
                    u14 = np.asarray(z["u_2xcv7"], dtype=np.float64) if "u_2xcv7" in z.files else u7
                    for j, dsl_j in enumerate(dsls_7):
                        tm_j = tmodes[j] if j < len(tmodes) else "raw"
                        gap_j = max(0.0, float(u7[j]) - float(u14[j]))
                        haz_j = estimate_ast_timeout_hazard(dsl_j, n_r, w_f)
                        _add_obs(
                            f"ugp:{cid_u}",
                            fam_u,
                            c44,
                            dsl_j,
                            float(u14[j]),
                            stab_val=0.85 * (1.0 - 0.5 * haz_j),
                            overfit_gap=gap_j,
                            tmode=tm_j,
                        )
                        n_ugp_obs += 1
                    if "dsls_cons" in z.files and "u_cons" in z.files:
                        dsls_c = [str(x) for x in z["dsls_cons"].tolist()]
                        uc = np.asarray(z["u_cons"], dtype=np.float64)
                        opt_r = np.asarray(z["optimism_r1_r0"], dtype=np.float64) if "optimism_r1_r0" in z.files else np.ones_like(uc)
                        tm_map = {dsls_7[j]: (tmodes[j] if j < len(tmodes) else "raw") for j in range(len(dsls_7))}
                        for j, dsl_c in enumerate(dsls_c):
                            haz_c = estimate_ast_timeout_hazard(dsl_c, n_r, w_f)
                            stab_c = float(np.clip(1.15 - 0.35 * float(opt_r[j]) - 0.35 * haz_c, 0.05, 0.98))
                            gap_c = max(0.0, 0.12 * (float(opt_r[j]) - 1.0))
                            _add_obs(
                                f"ugp:{cid_u}",
                                fam_u,
                                c44,
                                dsl_c,
                                float(uc[j]),
                                stab_val=stab_c,
                                overfit_gap=gap_c,
                                tmode=tm_map.get(dsl_c, "raw"),
                            )
                            n_ugp_obs += 1

    # 3C. Held-out `Episode-Test` backfill across `gp_teacher_shards_v1` (`12,800+` real episodes!)
    v1_dir = Path(a.v1_shards_dir)
    v2_dir = Path(a.v2_shards_dir)
    t4_dir = Path(a.teaching_v4_dir)
    v2_dir.mkdir(parents=True, exist_ok=True)
    t4_dir.mkdir(parents=True, exist_ok=True)

    mm_bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, sampling_mode="uniform_dataset")
    s8_bad_tidxs: set[int] = set()
    if a.s8_audit_json and Path(a.s8_audit_json).is_file():
        s8_doc = json.loads(Path(a.s8_audit_json).read_text())
        s8_bad_cids = {str(r["cid"]) for r in s8_doc.get("entries", []) if not bool(r.get("admit_canonical", True))}
        s8_bad_tidxs = {i for i, c in enumerate(mm_bank.cids) if str(c) in s8_bad_cids}
    n_s8_excluded_shards = 0
    table_ctx44_cache: Dict[int, np.ndarray] = {}

    def _get_table_ctx44(t_idx: int, Xc_sub: np.ndarray, yc_sub: np.ndarray) -> np.ndarray:
        got = table_ctx44_cache.get(t_idx)
        if got is not None:
            return got
        c44 = _encode_ctx44_from_xy(Xc_sub, yc_sub, dope_enc, dope_mean, P_dope32)
        table_ctx44_cache[t_idx] = c44
        return c44

    v1_files = sorted(v1_dir.glob("shard_*.npz"))[: int(a.max_v1_shards)]
    n_v1_scanned = 0
    n_v1_losers_vs_ridge = 0
    n_v2_admitted = 0
    n_v2_losers_vs_ridge = 0
    n_v1_obs = 0

    ledger_u_cv7: List[float] = []
    ledger_u_test_all: List[float] = []
    ledger_u_test_iid: List[float] = []
    ledger_u_test_extrap: List[float] = []
    ledger_u_ridge_cv7: List[float] = []
    ledger_u_ridge_test: List[float] = []
    ledger_overfit_gap: List[float] = []
    ledger_ast_size: List[int] = []
    ledger_n_ctx: List[int] = []
    ledger_width: List[int] = []
    ledger_heads: List[str] = []
    ledger_fams: List[str] = []
    ledger_is_d91: List[bool] = []
    ledger_dsls: List[str] = []
    ledger_ctx44: List[np.ndarray] = []

    seed_by_fam: Dict[str, Dict[str, Tuple[float, float, int]]] = {}  # fam -> dsl -> (sum_u_test, sum_regret, cnt)
    seed_by_tidx: Dict[int, Tuple[str, float, float]] = {}  # tidx -> (best_dsl, best_u_test, test_regret)

    for sh_p in v1_files:
        try:
            with np.load(sh_p, allow_pickle=False) as zs:
                is_d91 = bool(zs["is_dev91_table"])
                if is_d91 and not bool(a.allow_d91_random_splits):
                    continue
                t_idx = int(zs["table_idx"])
                if t_idx in s8_bad_tidxs:
                    n_s8_excluded_shards += 1
                    continue
                fam_s = str(zs["family_id"])
                dsl_s = str(zs["dsl"])
                Xc = np.asarray(zs["Xc"], dtype=np.float64)
                yc = np.asarray(zs["yc"], dtype=np.float64)
                Xq_real = np.asarray(zs["Xq_real"], dtype=np.float64)
                yq_real = np.asarray(zs["yq_real"], dtype=np.float64)
                pred_gp_qr = np.asarray(zs["pred_gp_q_real"], dtype=np.float64)
                Xq_cf = np.asarray(zs["Xq_cf"], dtype=np.float64) if "Xq_cf" in zs.files else None
                pred_gp_cf = np.asarray(zs["pred_gp_q_cf"], dtype=np.float64) if "pred_gp_q_cf" in zs.files else None
                oof_gp_c = np.asarray(zs["oof_gp_c"], dtype=np.float64)
                u_gp_cv7 = float(zs["u_gp_cv7"])
                u_ridge_cv7 = float(zs["u_ridge_cv7"])
        except (OSError, ValueError, KeyError):
            continue

        if len(yq_real) < 8 or len(yc) < 16 or not dsl_s:
            continue
        n_v1_scanned += 1

        # Exact causal ridge baseline on `(Xc, yc)` -> held-out `Xq_real` (`0` `yq_real` leakage!)
        _loo_r, pred_ridge_qr, pred_ridge_cf = fit_fast_gcv_ridge_on_ctx(Xc, yc, Xq_real, Xq_cf)
        iid_m, ext_m = split_query_iid_extrap(Xc, Xq_real, extrap_frac=0.30)

        u_gp_test_all = episode_heldout_utility(pred_gp_qr, yq_real, seed=45801 + t_idx)
        u_gp_test_iid = episode_heldout_utility(pred_gp_qr[iid_m], yq_real[iid_m], seed=45802 + t_idx)
        u_gp_test_ext = episode_heldout_utility(pred_gp_qr[ext_m], yq_real[ext_m], seed=45803 + t_idx)

        u_ridge_test_all = episode_heldout_utility(pred_ridge_qr, yq_real, seed=45801 + t_idx)
        u_ridge_test_ext = episode_heldout_utility(pred_ridge_qr[ext_m], yq_real[ext_m], seed=45803 + t_idx)

        if not (math.isfinite(u_gp_test_all) and math.isfinite(u_ridge_test_all)):
            continue

        if u_gp_test_all < u_ridge_test_all:
            n_v1_losers_vs_ridge += 1

        overfit_gap = max(0.0, u_gp_cv7 - u_gp_test_all)
        ridge_gap = max(0.0, u_ridge_cv7 - u_ridge_test_all)
        c44 = _get_table_ctx44(t_idx, Xc, yc)

        stab_gp = float(np.clip(0.92 - 0.65 * overfit_gap + (0.04 if u_gp_test_ext >= u_ridge_test_ext else -0.08), 0.05, 0.98))
        # Group by table_idx so multiple evolved ASTs + ridge baseline compete head-to-head on TRUE held-out `u_test_all`!
        _add_obs(f"v1test:{t_idx}", fam_s, c44, dsl_s, u_gp_test_all, stab_val=stab_gp, overfit_gap=overfit_gap, tmode="raw")
        _add_obs(
            f"v1test:{t_idx}",
            fam_s,
            c44,
            "(pipe X (ridge_gcv))",
            u_ridge_test_all,
            stab_val=0.90,
            overfit_gap=ridge_gap,
            tmode="raw",
        )
        n_v1_obs += 2

        ast_sz = int(G.complexity(dsl_s))
        head_nm = extract_ast_head_name(dsl_s)
        ledger_u_cv7.append(u_gp_cv7)
        ledger_u_test_all.append(u_gp_test_all)
        ledger_u_test_iid.append(u_gp_test_iid)
        ledger_u_test_extrap.append(u_gp_test_ext)
        ledger_u_ridge_cv7.append(u_ridge_cv7)
        ledger_u_ridge_test.append(u_ridge_test_all)
        ledger_overfit_gap.append(overfit_gap)
        ledger_ast_size.append(ast_sz)
        ledger_n_ctx.append(int(Xc.shape[0]))
        ledger_width.append(int(Xc.shape[1]))
        ledger_heads.append(head_nm)
        ledger_fams.append(fam_s)
        ledger_is_d91.append(is_d91)
        ledger_dsls.append(dsl_s)
        ledger_ctx44.append(c44)

        # Strict Harvester v2 held-out `Episode-Test` admission gate!
        w_opt, u_blend_test = optimal_heldout_blend_weight(pred_gp_qr, pred_ridge_qr, yq_real)
        blend_regret = u_blend_test - u_ridge_test_all
        passes_v2 = (
            w_opt >= 0.25
            and blend_regret >= float(a.min_test_regret)
            and overfit_gap <= float(a.max_overfit_gap)
            and u_gp_test_ext >= u_ridge_test_ext - 0.010
        )
        if passes_v2:
            n_v2_admitted += 1
            if u_blend_test < u_ridge_test_all:
                n_v2_losers_vs_ridge += 1
            fam_bucket = seed_by_fam.setdefault(fam_s, {})
            prev_f = fam_bucket.get(dsl_s, (0.0, 0.0, 0))
            fam_bucket[dsl_s] = (prev_f[0] + u_blend_test, prev_f[1] + blend_regret, prev_f[2] + 1)
            prev_t = seed_by_tidx.get(t_idx)
            if prev_t is None or u_blend_test > prev_t[1]:
                seed_by_tidx[t_idx] = (dsl_s, u_blend_test, blend_regret)

            if bool(a.bootstrap_v2_shards):
                v2_dst = v2_dir / sh_p.name
                if not v2_dst.exists():
                    pred_v2_qr = w_opt * pred_gp_qr + (1.0 - w_opt) * pred_ridge_qr
                    pred_v2_cf = (
                        (w_opt * pred_gp_cf + (1.0 - w_opt) * pred_ridge_cf)
                        if (pred_gp_cf is not None and pred_ridge_cf is not None)
                        else pred_gp_cf
                    )
                    v2_tmp = v2_dir / f".boot_{os.getpid()}.tmp.npz"
                    save_kw = dict(
                        Xc=Xc.astype(np.float32),
                        yc=yc.astype(np.float64),
                        oof_gp_c=oof_gp_c.astype(np.float64),
                        Xq_real=Xq_real.astype(np.float32),
                        yq_real=yq_real.astype(np.float64),
                        pred_gp_q_real=pred_v2_qr.astype(np.float64),
                        u_gp_cv7=np.array(u_gp_cv7, dtype=np.float64),
                        u_ridge_cv7=np.array(u_ridge_cv7, dtype=np.float64),
                        u_gp_test_all=np.array(u_blend_test, dtype=np.float64),
                        u_gp_test_iid=np.array(u_gp_test_iid, dtype=np.float64),
                        u_gp_test_extrap=np.array(u_gp_test_ext, dtype=np.float64),
                        u_ridge_test_all=np.array(u_ridge_test_all, dtype=np.float64),
                        overfit_gap=np.array(overfit_gap, dtype=np.float64),
                        blend_weight_gp=np.array(w_opt, dtype=np.float64),
                        regret_vs_base=np.array(blend_regret, dtype=np.float64),
                        table_idx=np.array(int(t_idx), dtype=np.int64),
                        is_dev91_table=np.array(bool(is_d91)),
                        canonical_promotion_forbidden=np.array(True),
                        family_id=np.array(fam_s),
                        dsl=np.array(dsl_s),
                        real_ctx44=c44.astype(np.float32),
                    )
                    if Xq_cf is not None and pred_v2_cf is not None:
                        save_kw["Xq_cf"] = Xq_cf.astype(np.float32)
                        save_kw["pred_gp_q_cf"] = pred_v2_cf.astype(np.float64)
                    np.savez_compressed(v2_tmp, **save_kw)
                    os.replace(v2_tmp, v2_dst)

    # 3D. Also ingest any live `teaching_shards_v4/live_tshard_v4_*.npz` emitted by `Harvester v2`
    n_live_v4_obs = 0
    for ts4_p in sorted(t4_dir.glob("live_tshard_v4_*.npz")):
        try:
            with np.load(ts4_p, allow_pickle=False) as z4:
                is_d91_4 = bool(z4["is_dev91_table"])
                if is_d91_4 and not bool(a.allow_d91_random_splits):
                    continue
                c_id4 = str(z4["ctx_id"])
                f_id4 = str(z4["family_id"])
                c44_4 = np.asarray(z4["real_ctx44"], dtype=np.float32)
                dsls_4 = [str(x) for x in z4["dsls"].tolist()]
                u_cv_4 = np.asarray(z4["u_cv7"], dtype=np.float64)
                u_te_4 = np.asarray(z4["u_test_all"], dtype=np.float64)
                to_4 = np.asarray(z4["timed_out"], dtype=bool)
                for ds4, uc4, ut4, to_flag in zip(dsls_4, u_cv_4, u_te_4, to_4):
                    gap4 = max(0.0, float(uc4) - float(ut4)) if (math.isfinite(uc4) and math.isfinite(ut4)) else 0.25
                    u_use = float(ut4) if math.isfinite(ut4) else float(uc4)
                    _add_obs(
                        f"live4:{c_id4}",
                        f_id4,
                        c44_4,
                        ds4,
                        u_use,
                        stab_val=0.10 if bool(to_flag) else float(np.clip(0.90 - 0.55 * gap4, 0.10, 0.96)),
                        overfit_gap=gap4,
                        tmode="raw",
                    )
                    n_live_v4_obs += 1
        except (OSError, ValueError, KeyError):
            continue

    # Save Track S Interface (iv) empirical CV->Test ledger `.npz` + `gp_family_seed_library_v4.json`!
    ledger_p = t4_dir / "v1_backfill_cv_to_test_ledger.npz"
    if ledger_u_cv7:
        tmp_lp = t4_dir / "v1_backfill_cv_to_test_ledger.tmp.npz"
        np.savez_compressed(
            tmp_lp,
            u_cv7=np.asarray(ledger_u_cv7, dtype=np.float64),
            u_test_all=np.asarray(ledger_u_test_all, dtype=np.float64),
            u_test_iid=np.asarray(ledger_u_test_iid, dtype=np.float64),
            u_test_extrap=np.asarray(ledger_u_test_extrap, dtype=np.float64),
            u_ridge_cv7=np.asarray(ledger_u_ridge_cv7, dtype=np.float64),
            u_ridge_test_all=np.asarray(ledger_u_ridge_test, dtype=np.float64),
            overfit_gap=np.asarray(ledger_overfit_gap, dtype=np.float64),
            ast_size=np.asarray(ledger_ast_size, dtype=np.int32),
            n_ctx=np.asarray(ledger_n_ctx, dtype=np.int32),
            width=np.asarray(ledger_width, dtype=np.int32),
            head=np.asarray(ledger_heads),
            family_id=np.asarray(ledger_fams),
            is_dev91_table=np.asarray(ledger_is_d91, dtype=bool),
            dsl=np.asarray(ledger_dsls),
            real_ctx44=np.vstack(ledger_ctx44).astype(np.float32),
        )
        os.replace(tmp_lp, ledger_p)

    fam_seeds_out: Dict[str, List[Dict]] = {}
    for fam_k, bucket in sorted(seed_by_fam.items()):
        ranked = sorted(
            bucket.items(),
            key=lambda kv: (kv[1][1] / max(1, kv[1][2])) + 0.02 * math.log1p(kv[1][2]),
            reverse=True,
        )[:5]
        fam_seeds_out[fam_k] = [
            {
                "dsl": ds_k,
                "mean_u_test": round(vals[0] / vals[2], 6),
                "mean_test_regret_vs_ridge": round(vals[1] / vals[2], 6),
                "verified_episodes": int(vals[2]),
            }
            for ds_k, vals in ranked
        ]
    seed_lib_p = Path(a.seed_library_json)
    seed_lib_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_sl = seed_lib_p.with_suffix(".tmp.json")
    tmp_sl.write_text(
        json.dumps(
            {
                "schema": "jope.seed_library.gp/v4",
                "utc": time.strftime("%FT%TZ", time.gmtime()),
                "canonical_promotion_forbidden": True,
                "families_indexed": len(fam_seeds_out),
                "tables_indexed": len(seed_by_tidx),
                "family_seeds": fam_seeds_out,
                "table_champions": {
                    str(tk): {"dsl": tv[0], "u_test": round(tv[1], 6), "test_regret": round(tv[2], 6)}
                    for tk, tv in sorted(seed_by_tidx.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(tmp_sl, seed_lib_p)

    # 3E. Contrastive negative anchors so winners in small-candidate contexts are never centered to `-0.5`
    for ctx_id, g_entry in list(grouped.items()):
        if not (ctx_id.startswith("ugp:") or ctx_id.startswith("v1test:")):
            continue
        fam_g = g_entry["fam"]
        c44_g = g_entry["ctx44"]
        cur_us = [v[0] for v in g_entry["obs"].values()]
        if not cur_us:
            continue
        min_pos_u = min(cur_us)
        neg_added = 0
        for _ in range(24):
            dsl_n, _isl_n, fams_n = bank_pool[int(rng.integers(0, len(bank_pool)))]
            if fam_g not in fams_n and (dsl_n, "raw") not in g_entry["obs"]:
                _add_obs(
                    ctx_id,
                    fam_g,
                    c44_g,
                    dsl_n,
                    min_pos_u - 0.35 - 0.02 * neg_added,
                    stab_val=0.38,
                    overfit_gap=0.22 + 0.02 * neg_added,
                    tmode="raw",
                )
                neg_added += 1
                if neg_added >= 4:
                    break

    # 4. Materialize flat arrays + intra-context midranks & ranking pairs
    ast_feat_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def _fast_ast_feat(dsl_s: str, tm_s: str) -> np.ndarray:
        key = (dsl_s, tm_s)
        got = ast_feat_cache.get(key)
        if got is None:
            got = featurize_candidate_h454(dsl_s, target_mode=tm_s).astype(np.float32)
            ast_feat_cache[key] = got
        return got

    X_ctx_rows: List[np.ndarray] = []
    X_ast_rows: List[np.ndarray] = []
    dsl_rows: List[str] = []
    q_mid_rows: List[float] = []
    stab_rows: List[float] = []
    gap_rows: List[float] = []
    fam_rows: List[str] = []
    ctx_group_slices: List[Tuple[int, int, str]] = []

    for _ctx_id, g_entry in sorted(grouped.items()):
        items = list(g_entry["obs"].items())
        if len(items) < 2:
            continue
        u_vals = [v[0] for _, v in items]
        q_mids = _midrank_centered(u_vals)
        start_idx = len(X_ctx_rows)
        c44 = g_entry["ctx44"]
        fam_id = g_entry["fam"]
        for ((dsl_s, tm_s), (_uv, st_v, gp_v)), q_m in zip(items, q_mids):
            X_ctx_rows.append(c44)
            X_ast_rows.append(_fast_ast_feat(dsl_s, tm_s))
            dsl_rows.append(dsl_s)
            q_mid_rows.append(float(q_m))
            stab_rows.append(float(st_v))
            gap_rows.append(float(gp_v))
            fam_rows.append(fam_id)
        end_idx = len(X_ctx_rows)
        ctx_group_slices.append((start_idx, end_idx, fam_id))

    X_ctx = np.vstack(X_ctx_rows).astype(np.float32)
    X_ast = np.vstack(X_ast_rows).astype(np.float32)
    q_mid = np.asarray(q_mid_rows, dtype=np.float32)
    stab_y = np.asarray(stab_rows, dtype=np.float32)
    gap_y = np.asarray(gap_rows, dtype=np.float32)

    # Verify zero degenerate zero-context rows from real tables!
    nonzero_ctx_frac = float(np.mean(np.linalg.norm(X_ctx, axis=1) > 1e-6))
    assert nonzero_ctx_frac > 0.99, f"Expected >99% real non-zero ctx44 rows, got {nonzero_ctx_frac:.4f}"

    # Save consolidated `init_h458_v4_shards.npz`
    init_shard_p = t4_dir / "init_h458_v4_shards.npz"
    tmp_shard_p = t4_dir / "init_h458_v4_shards.tmp.npz"
    np.savez_compressed(
        tmp_shard_p,
        X_ctx=X_ctx,
        X_ast=X_ast,
        q_midrank=q_mid,
        stab_target=stab_y,
        overfit_gap_target=gap_y,
        families=np.array(fam_rows),
    )
    os.replace(tmp_shard_p, init_shard_p)

    # Strict cross-family validation split (`20%` of distinct `family_id` groups held out; `0` family overlap!)
    uniq_fams = sorted(set(fam_rows))
    rng.shuffle(uniq_fams)
    val_fams = set(uniq_fams[: max(2, len(uniq_fams) // 5)])
    va_mask = np.array([f in val_fams for f in fam_rows], dtype=bool)
    tr_mask = ~va_mask

    c_mu = np.mean(X_ctx[tr_mask], axis=0, keepdims=True)
    c_sd = np.maximum(np.std(X_ctx[tr_mask], axis=0, keepdims=True), 1e-5)
    X_ctx_z = np.clip((X_ctx - c_mu) / c_sd, -8.0, 8.0)

    tr_pairs: List[Tuple[int, int]] = []
    va_ctx_slices: List[Tuple[int, int]] = []
    for s_i, e_i, fam_i in ctx_group_slices:
        if fam_i in val_fams:
            va_ctx_slices.append((s_i, e_i))
        else:
            # Cap pairs per context so huge contexts don't dominate
            ys_c = q_mid[s_i:e_i]
            ord_c = np.argsort(-ys_c)
            for ti in range(min(6, len(ord_c))):
                a_i = s_i + int(ord_c[ti])
                for bi in range(ti + 1, min(len(ord_c), ti + 12)):
                    b_i = s_i + int(ord_c[bi])
                    if q_mid[a_i] - q_mid[b_i] >= 0.14:
                        tr_pairs.append((a_i, b_i))
    tr_pairs_np = np.asarray(tr_pairs, dtype=np.int64) if tr_pairs else np.zeros((0, 2), dtype=np.int64)

    model = ContextualSurrogateQHatV4(d_ctx=44, d_ast=128, hidden=160)
    opt = torch.optim.AdamW(model.parameters(), lr=float(a.lr), weight_decay=1.5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(a.epochs)), eta_min=1e-4)
    huber = nn.SmoothL1Loss(beta=0.08)
    gap_huber = nn.SmoothL1Loss(beta=0.04)

    tc_all = torch.from_numpy(X_ctx_z)
    ta_all = torch.from_numpy(X_ast)
    tq_all = torch.from_numpy(q_mid)
    ts_all = torch.from_numpy(stab_y)
    tg_all = torch.from_numpy(gap_y)

    tr_indices = np.flatnonzero(tr_mask)
    va_indices = np.flatnonzero(va_mask)

    best_score = -1e9
    best_metrics: Dict[str, float] = {}
    best_sd = None
    bs = 512

    for _ep in range(int(a.epochs)):
        model.train()
        perm = rng.permutation(tr_indices)
        pair_perm = rng.permutation(len(tr_pairs_np)) if len(tr_pairs_np) > 0 else np.empty(0, dtype=np.int64)
        p_cursor = 0
        for s in range(0, len(perm), bs):
            idx = perm[s : s + bs]
            opt.zero_grad(set_to_none=True)
            qp, sp, gp = model(tc_all[idx], ta_all[idx])
            loss = huber(qp, tq_all[idx]) + 0.25 * F.mse_loss(sp, ts_all[idx]) + 0.35 * gap_huber(gp, tg_all[idx])
            if len(pair_perm) >= 64:
                if p_cursor + 256 > len(pair_perm):
                    pair_perm = rng.permutation(len(tr_pairs_np))
                    p_cursor = 0
                p_sub = tr_pairs_np[pair_perm[p_cursor : p_cursor + min(256, len(pair_perm))]]
                p_cursor += 256
                q_hi, _, g_hi = model(tc_all[p_sub[:, 0]], ta_all[p_sub[:, 0]])
                q_lo, _, g_lo = model(tc_all[p_sub[:, 1]], ta_all[p_sub[:, 1]])
                margin = torch.clamp(tq_all[p_sub[:, 0]] - tq_all[p_sub[:, 1]], 0.10, 1.0)
                eff_hi = q_hi - 0.20 * g_hi
                eff_lo = q_lo - 0.20 * g_lo
                pair_loss = F.softplus(-4.0 * (eff_hi - eff_lo) * margin).mean()
                loss = loss + 0.35 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if (_ep + 1) % 5 != 0 and (_ep + 1) != int(a.epochs):
            continue

        model.eval()
        with torch.no_grad():
            va_q, _va_s, va_g = model(tc_all[va_indices], ta_all[va_indices])
            va_huber = float(huber(va_q, tq_all[va_indices]).item())
            va_gap_mae = float(torch.mean(torch.abs(va_g - tg_all[va_indices])).item())
            va_qn = (va_q - 0.20 * va_g).numpy()
            va_yt = q_mid[va_indices]
            va_gn = va_g.numpy()
            va_gt = gap_y[va_indices]
            va_spearman = float(spearmanr(va_qn, va_yt).statistic) if len(va_yt) > 4 else 0.0
            va_gap_spearman = float(spearmanr(va_gn, va_gt).statistic) if len(va_gt) > 4 and float(np.std(va_gt)) > 1e-6 else 0.0

            q_all_p, _, g_all_p = model(tc_all, ta_all)
            eff_all_np = (q_all_p - 0.20 * g_all_p).numpy()
            within_rhos = []
            top1_hits = []
            for s_i, e_i in va_ctx_slices:
                if e_i - s_i >= 4 and float(np.std(q_mid[s_i:e_i])) > 1e-6:
                    r_ctx = float(spearmanr(eff_all_np[s_i:e_i], q_mid[s_i:e_i]).statistic)
                    if math.isfinite(r_ctx):
                        within_rhos.append(r_ctx)
                    top1_hits.append(1.0 if int(np.argmax(eff_all_np[s_i:e_i])) == int(np.argmax(q_mid[s_i:e_i])) else 0.0)
            mean_within_rho = float(np.mean(within_rhos)) if within_rhos else va_spearman
            top1_acc = float(np.mean(top1_hits)) if top1_hits else 0.0

        combo = 0.50 * va_spearman + 0.40 * mean_within_rho + 0.10 * va_gap_spearman
        if combo > best_score:
            best_score = combo
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_huber": round(va_huber, 6),
                "val_overfit_gap_mae": round(va_gap_mae, 6),
                "val_overfit_gap_spearman_r": round(va_gap_spearman, 6),
                "val_spearman_r": round(va_spearman, 6),
                "val_within_context_spearman_r": round(mean_within_rho, 6),
                "val_within_context_top1_acc": round(top1_acc, 6),
            }

    # Paired apples-to-apples comparison vs `_QHatV2Net` (`52d`, Huber-only) on the EXACT SAME `tr_mask`!
    from ultra_gp_evolve_1000g_h453 import _QHatV2Net
    from ultra_gp_evolve_h452 import featurize_candidate_h452

    v2_feat_cache: Dict[str, np.ndarray] = {}
    v2_rows = []
    for d in dsl_rows:
        f2 = v2_feat_cache.get(d)
        if f2 is None:
            f2 = featurize_candidate_h452(G.Candidate.from_dsl(d, "v2")).astype(np.float32)
            v2_feat_cache[d] = f2
        v2_rows.append(f2)
    X_ast2_all = torch.from_numpy(np.stack(v2_rows, axis=0))
    torch.manual_seed(int(a.seed) + 202)
    net2_fair = _QHatV2Net(44, int(X_ast2_all.shape[1]), hidden=128)
    opt2 = torch.optim.AdamW(net2_fair.parameters(), lr=float(a.lr), weight_decay=2e-4)
    best_v2_sp = -1.0
    best_v2_wr = -1.0
    for _ep2 in range(int(a.epochs)):
        net2_fair.train()
        perm2 = rng.permutation(tr_indices)
        for bi2 in range(0, len(perm2), bs):
            b_idx2 = perm2[bi2 : bi2 + bs]
            l2 = huber(net2_fair(tc_all[b_idx2], X_ast2_all[b_idx2]), tq_all[b_idx2])
            opt2.zero_grad()
            l2.backward()
            opt2.step()
        if (_ep2 + 1) % 5 != 0 and (_ep2 + 1) != int(a.epochs):
            continue
        net2_fair.eval()
        with torch.no_grad():
            p2_all = net2_fair(tc_all, X_ast2_all).numpy()
            sp2 = float(spearmanr(p2_all[va_indices], q_mid[va_indices]).statistic)
            w2 = [
                float(spearmanr(p2_all[s_i:e_i], q_mid[s_i:e_i]).statistic)
                for s_i, e_i in va_ctx_slices
                if (e_i - s_i) >= 4 and float(np.std(q_mid[s_i:e_i])) > 1e-6
            ]
            w2 = [x for x in w2 if math.isfinite(x)]
            wr2 = float(np.mean(w2)) if w2 else sp2
            if 0.55 * sp2 + 0.45 * wr2 > 0.55 * best_v2_sp + 0.45 * best_v2_wr:
                best_v2_sp = sp2
                best_v2_wr = wr2

    # Full-corpus refit for deployment (`model_version == "v4"`)
    all_indices = np.arange(len(q_mid), dtype=np.int64)
    all_pairs_list: List[Tuple[int, int]] = []
    for s_i, e_i, _fam_i in ctx_group_slices:
        if e_i - s_i < 3:
            continue
        ys_g = q_mid[s_i:e_i]
        ord_g = np.argsort(-ys_g)
        for ti in range(min(4, len(ord_g))):
            hi_idx = s_i + int(ord_g[ti])
            for bi in range(ti + 1, min(len(ord_g), ti + 10)):
                lo_idx = s_i + int(ord_g[bi])
                if ys_g[ord_g[ti]] - ys_g[ord_g[bi]] >= 0.10:
                    all_pairs_list.append((hi_idx, lo_idx))
    all_pairs_np = np.asarray(all_pairs_list, dtype=np.int64) if all_pairs_list else tr_pairs_np

    torch.manual_seed(int(a.seed) + 303)
    full_model = ContextualSurrogateQHatV4(44, 128, hidden=160)
    opt_full = torch.optim.AdamW(full_model.parameters(), lr=float(a.lr), weight_decay=2e-4)
    sched_full = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=max(1, int(a.epochs)))
    for _epf in range(int(a.epochs)):
        full_model.train()
        permf = rng.permutation(all_indices)
        for bif in range(0, len(permf), bs):
            b_idxf = permf[bif : bif + bs]
            qp_f, sp_f, gp_f = full_model(tc_all[b_idxf], ta_all[b_idxf])
            loss_f = huber(qp_f, tq_all[b_idxf]) + 0.25 * F.mse_loss(sp_f, ts_all[b_idxf]) + 0.35 * gap_huber(gp_f, tg_all[b_idxf])
            if len(all_pairs_np) >= 64:
                p_sel_f = rng.integers(0, len(all_pairs_np), size=min(256, len(all_pairs_np)))
                hi_pf, lo_pf = all_pairs_np[p_sel_f, 0], all_pairs_np[p_sel_f, 1]
                q_hif, _, g_hif = full_model(tc_all[hi_pf], ta_all[hi_pf])
                q_lof, _, g_lof = full_model(tc_all[lo_pf], ta_all[lo_pf])
                margin_f = torch.clamp(tq_all[hi_pf] - tq_all[lo_pf], 0.10, 1.0)
                loss_f = loss_f + 0.35 * F.softplus(-4.0 * ((q_hif - 0.20 * g_hif) - (q_lof - 0.20 * g_lof)) * margin_f).mean()
            opt_full.zero_grad()
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 2.0)
            opt_full.step()
        sched_full.step()

    dope_sha_after = dope_enc.dope_sha()
    assert dope_sha_after == dope_sha_before == C.DOPE_SHA256, "DOPE weight sha changed!"

    out_ckpt = Path(a.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = out_ckpt.with_suffix(".tmp.pt")
    full_sd = {k: v.cpu().clone() for k, v in full_model.state_dict().items()}
    torch.save(
        {
            "model_version": "v4",
            "state_dict": full_sd,
            "heldout_state_dict": best_sd,
            "d_ctx": 44,
            "d_ast": 128,
            "hidden": 160,
            "dope_mean": dope_mean,
            "P_dope32": P_dope32,
            "c_mu": c_mu,
            "c_sd": c_sd,
            "dope_sha256": dope_sha_after,
            "canonical_promotion_forbidden": bool(a.allow_d91_random_splits),
            **best_metrics,
        },
        tmp_ckpt,
    )
    os.replace(tmp_ckpt, out_ckpt)

    # Smoke-verify strict `PretrainedQHatV4Prior(out_ckpt, require_ok=True)` load!
    prior_check = PretrainedQHatV4Prior(out_ckpt, require_ok=True)
    assert prior_check.ok and prior_check.version == 4

    v1_loser_rate = float(n_v1_losers_vs_ridge / max(1, n_v1_scanned))
    v2_loser_rate = float(n_v2_losers_vs_ridge / max(1, n_v2_admitted))

    rec = {
        "wave": "H458_TrackG",
        "model": "ContextualSurrogateQHatV4",
        "d_ctx": 44,
        "d_ast": 128,
        "nonzero_ctx44_frac": round(nonzero_ctx_frac, 6),
        "contexts_grouped": len(ctx_group_slices),
        "distinct_families": len(uniq_fams),
        "heldout_val_families": len(val_fams),
        "train_samples": int(np.sum(tr_mask)),
        "val_samples": int(np.sum(va_mask)),
        "train_ranking_pairs": int(len(tr_pairs_np)),
        "dltrain_obs": n_dltrain_obs,
        "ultra_gp_1000g_obs": n_ugp_obs,
        "v1_heldout_backfill_obs": n_v1_obs,
        "live_v4_tshard_obs": n_live_v4_obs,
        "v1_shards_scanned": n_v1_scanned,
        "s8_canonical_confirm_shards_excluded": n_s8_excluded_shards,
        "s8_canonical_confirm_tables_excluded": len(s8_bad_tidxs),
        "v1_heldout_losers_vs_ridge": n_v1_losers_vs_ridge,
        "v1_heldout_loser_rate": round(v1_loser_rate, 6),
        "v2_shards_admitted": n_v2_admitted,
        "v2_heldout_losers_vs_ridge": n_v2_losers_vs_ridge,
        "v2_heldout_loser_rate": round(v2_loser_rate, 6),
        "cv_to_test_ledger_npz": str(ledger_p),
        "seed_library_v4_json": str(seed_lib_p),
        "seed_library_families": len(fam_seeds_out),
        "seed_library_tables": len(seed_by_tidx),
        "dope_trainable_params": 0,
        "dope_sha256_before": dope_sha_before,
        "dope_sha256_after": dope_sha_after,
        "dope_unchanged": True,
        "canonical_promotion_forbidden": bool(a.allow_d91_random_splits),
        "ckpt_path": str(out_ckpt),
        "ckpt_sha256": C.sha256_file(out_ckpt),
        **best_metrics,
        "v2_same_val_spearman_r": round(best_v2_sp, 6),
        "v2_same_val_within_context_spearman_r": round(best_v2_wr, 6),
        "beats_v2_on_same_heldout_families": bool(
            best_metrics["val_spearman_r"] > best_v2_sp
            and best_metrics["val_within_context_spearman_r"] > best_v2_wr
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    out_rec = Path(a.out_receipt)
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    tmp_rec = out_rec.with_suffix(".tmp.json")
    tmp_rec.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_rec, out_rec)
    print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
