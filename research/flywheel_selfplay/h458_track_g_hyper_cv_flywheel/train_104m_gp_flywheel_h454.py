#!/usr/bin/env python3
"""Wave H454 `104M` (`ICL_L`) Closed-Loop Exact-GP-Teacher Distillation + Counterfactual Queries
+ Exact Appended `CV-7` OOF GP-Residual Co-Training Trainer (`train_104m_gp_flywheel_h454.py`).

Upgrades `train_104m_gp_flywheel_h452.py` with 100% `ckpt_last.pt` hot-resume compatibility:
  1. `ExactGPTeacherShardStore` (Hot-Reloads `gp_teacher_shards_v1/*.npz` Every `45s`):
     - Enforces strict positive-regret gating (`regret_vs_base > 0`) and priority-samples discovered Rust GP teachers
       by `clip(delta_u, 0.01, 1.50) ** 0.75` (never distilling on tables where linear/base already matches GP!).
     - Supports `--allow-d91-teacher-shards {0, 1}` for both quarantined full-corpus and `0% dev91` fair-OOS runs.
  2. Batch Mixture (`50% Diverse S3 + 25% Exact GP KD + Counterfactuals + 15% Exact GP-Resid + 10% Heavy Synth`):
     - `25%` Exact GP Teacher Distillation (`exact_kd`): Preserves 100% of signal columns, blends real query targets
       `(1 - alpha(du)) * yq_real + alpha(du) * pred_gp_q_real` (`alpha = clip(tanh(2 * du), 0.15, 0.85)`), and
       concatenates `2x` manifold counterfactual queries `(Xq_cf, pred_gp_q_cf)` (`weight = 0.65 * alpha(du)`) in the
       SAME `ICLH450MPS` forward pass (`3x` query gradient signal at `~1.08x` GPU cost; `0.0` query cross-talk!).
     - `15%` Exact Train/Test-Aligned GP-Residual Co-Training (`exact_resid`): Appends `[oof_gp_c; pred_gp_q_real]`
       as feature column `0` (never overwriting real columns!) and trains `104M` on exact residual `y - y_hat_gp`,
       matching test-time `cols_104m_e8_gpresid_h452.py` 1-to-1!
     - `50%` Extreme-Diversity Real & Stratified-Crop S3 Episodes across all `3,525` tables, with upgraded full-grammar
       `build_gp_stage_cols_zero_leakage_v2` (`svd_project`, `nystrom`, `blom`, `winsor_clip`, `robust_scale`,
       `pair_ratio`, `cheb`, `hinge`, `symlog`) **APPENDING** features instead of overwriting `rep_idx`!
     - `10%` Heavy Synthetic SCM / Symbolic / Multi-Index Prior Episodes.
  3. Zero file deletions (`0` `rm`/`unlink`/`rmtree`), zero `/tmp` usage, and 128-bit `Ledger128` deduplication.
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
from typing import Dict, List, Optional, Tuple

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
from icl_model import CLIP_X, FID_SLOTS  # noqa: E402
from icl_model_mps_h450 import ICLH450MPS, n_params  # noqa: E402
from icl_prior_v3s import Prior, draw_features  # noqa: E402
from mmap_bank_h450 import MmapF64Bank, sha256_file  # noqa: E402
from train_104m_gp_flywheel_h452 import (  # noqa: E402
    ROW_BINS_V2,
    inject_synth_cols_18fam_zero_leakage,
    nonlinear_gp_oof_residual_np,
    sample_extreme_stratified_episode_h452,
)

FULL_STAGE_TOKENS_H454 = (
    "pair_ratio",
    "pair_prod",
    "cheb",
    "symlog",
    "poly_interact",
    "hinge",
    "oblique_hinge",
    "spline",
    "rank",
    "blom",
    "winsor_clip",
    "robust_scale",
    "svd_project",
    "nystrom",
    "rff_cos",
    "bin_onehot",
)


def parse_dsl_stages_and_head_h454(dsl: str) -> Tuple[Tuple[str, ...], str]:
    stages = tuple(tok for tok in FULL_STAGE_TOKENS_H454 if f"({tok}" in dsl)
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


class HWMGPWinnerStoreV3:
    """Strict HWM-deduplicated GP winner store (`1` best champion per table; `0` stale generation pollution)."""

    def __init__(self, jsonl_path: str, hwm_json_path: str, bank: MmapF64Bank, reload_every_s: float = 45.0) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.hwm_path = Path(hwm_json_path) if hwm_json_path else None
        self.n_tables = int(bank.n_tables)
        self.d91_indices = np.flatnonzero(bank.is_d91).astype(np.int64)
        self.by_cid_idx = {str(cid): idx for idx, cid in enumerate(bank.cids)}
        self.reload_every_s = float(reload_every_s)
        self._last_check = 0.0
        self._last_sig = (-1.0, -1.0)
        self.by_table: Dict[int, dict] = {}
        self.all_winners: List[dict] = []
        self.table_probs: Optional[np.ndarray] = None
        self.maybe_reload(force=True)

    def maybe_reload(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_check) < self.reload_every_s:
            return
        self._last_check = now
        mt_j = self.jsonl_path.stat().st_mtime if (self.jsonl_path and self.jsonl_path.is_file()) else -1.0
        mt_h = self.hwm_path.stat().st_mtime if (self.hwm_path and self.hwm_path.is_file()) else -1.0
        if not force and (mt_j, mt_h) == self._last_sig:
            return

        by_t: Dict[int, dict] = {}
        regret_by_t = np.zeros(self.n_tables, dtype=np.float64)

        def _upsert(t_idx: int, dsl: str, u_sc: float, regret: float) -> None:
            if not dsl or regret <= 0.0:
                return
            stages, head = parse_dsl_stages_and_head_h454(dsl)
            entry = {
                "dsl": dsl,
                "stages": stages or ("pair_ratio", "cheb", "svd_project", "blom"),
                "head": head,
                "u_score": float(u_sc),
                "regret": float(regret),
            }
            if 0 <= t_idx < self.n_tables:
                prev = by_t.get(t_idx)
                if prev is None or float(u_sc) >= float(prev["u_score"]):
                    by_t[t_idx] = entry
                regret_by_t[t_idx] = max(regret_by_t[t_idx], min(float(regret), 1.5))

        try:
            if self.hwm_path and self.hwm_path.is_file():
                doc = json.loads(self.hwm_path.read_text(encoding="utf-8"))
                for k_s, rec in doc.get("champions", {}).items():
                    t_i = int(rec.get("table_idx", int(k_s)))
                    dsl_s = str(rec.get("best_raw_dsl") or rec.get("best_any_dsl") or "")
                    u_s = float(rec.get("best_raw_u", rec.get("best_any_u", 0.0)))
                    reg_s = max(0.0, float(rec.get("regret_est", 0.05)))
                    _upsert(t_i, dsl_s, u_s, reg_s)

            if self.jsonl_path and self.jsonl_path.is_file():
                with self.jsonl_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
                        dsl_s = str(rec.get("dsl", ""))
                        t_i = rec.get("table_idx")
                        if not isinstance(t_i, int) or not (0 <= t_i < self.n_tables):
                            t_i = self.by_cid_idx.get(str(rec.get("cid", "")), -1)
                        u_s = float(rec.get("u_2xcv7", rec.get("u_cv7", 0.0)))
                        reg_s = max(0.0, float(rec.get("regret_vs_linear", 0.0)))
                        _upsert(int(t_i), dsl_s, u_s, reg_s)

            if by_t:
                self.by_table = by_t
                self.all_winners = list(by_t.values())
                w = 1.0 + 3.5 * (np.clip(regret_by_t, 0.0, 1.5) ** 0.75)
                self.table_probs = w / float(np.sum(w))
                self._last_sig = (mt_j, mt_h)
        except (OSError, ValueError, KeyError):
            pass

    def sample_table_idx(
        self,
        bank: MmapF64Bank,
        rng: np.random.Generator,
        regret_tilt_prob: float = 0.25,
        d91_boost_prob: float = 0.20,
    ) -> Tuple[int, bool]:
        u = float(rng.random())
        if d91_boost_prob > 0.0 and self.d91_indices.size > 0 and u < d91_boost_prob:
            return int(self.d91_indices[int(rng.integers(0, self.d91_indices.size))]), True
        if self.table_probs is not None and u < (d91_boost_prob + regret_tilt_prob):
            return int(rng.choice(self.n_tables, p=self.table_probs)), False
        return bank.sample_table_idx(rng), False

    def pick_winner_for_table(self, t_idx: int, rng: np.random.Generator) -> Optional[dict]:
        cand = self.by_table.get(int(t_idx))
        if cand is not None and float(rng.random()) < 0.80:
            return cand
        if self.all_winners:
            return self.all_winners[int(rng.integers(0, len(self.all_winners)))]
        return None


class ExactGPTeacherShardStore:
    """Hot-reloading store of verified positive-regret Rust GP Teacher Shards (`gp_teacher_shards_v1/*.npz`)."""

    def __init__(
        self,
        shards_dir: str,
        allow_d91: bool = True,
        min_regret: float = 0.010,
        gamma: float = 0.75,
        reload_every_s: float = 45.0,
    ) -> None:
        self.shards_dir = Path(shards_dir) if shards_dir else None
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
                        is_d91 = bool(z["is_dev91_table"])
                        if is_d91 and not self.allow_d91:
                            self._loaded_paths.add(ps)
                            continue
                        regret = float(z["regret_vs_base"])
                        if not math.isfinite(regret) or regret < self.min_regret:
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
                        t_idx = int(z["table_idx"])
                        if len(yc) < 24 or len(yqr) < 6 or Xc.shape[1] < 1:
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
                                "is_d91": is_d91,
                            }
                        )
                        self._loaded_paths.add(ps)
                        added += 1
                except (OSError, ValueError, KeyError):
                    continue
            if added > 0 and self.shards:
                # Strict HWM supersession across cuts: keep at most 4 highest-regret cuts per `table_idx` in worker RAM
                if len(self.shards) > 2400:
                    by_t: Dict[int, List[dict]] = {}
                    for sh in self.shards:
                        by_t.setdefault(int(sh["table_idx"]), []).append(sh)
                    trimmed: List[dict] = []
                    for lst in by_t.values():
                        lst.sort(key=lambda s: -float(s["regret"]))
                        trimmed.extend(lst[:4])
                    self.shards = trimmed
                regs = np.array([min(max(float(s["regret"]), 0.01), 1.50) for s in self.shards], dtype=np.float64)
                w = regs ** self.gamma
                self.probs = w / float(np.sum(w))
        except (OSError, ValueError, KeyError):
            pass

    def sample_shard(self, rng: np.random.Generator) -> Optional[dict]:
        if not self.shards or self.probs is None:
            return None
        idx = int(rng.choice(len(self.shards), p=self.probs))
        return self.shards[idx]


def build_gp_stage_cols_zero_leakage_v2(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    stages: Tuple[str, ...],
    k_want: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full-grammar context-fitted GP stage bank (`svd_project`, `nystrom`, `blom`, `winsor_clip`, `robust_scale`,
    `pair_ratio`, `pair_prod`, `cheb`, `symlog`, `hinge`, `oblique_hinge`) with ZERO query leakage!"""
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
    cand_c: List[np.ndarray] = []
    cand_q: List[np.ndarray] = []
    st_set = set(stages) if stages else {"pair_ratio", "pair_prod", "cheb", "svd_project", "blom"}

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
            tc, tq = np.tanh(Zc[:, int(ja)] / 3.0), np.tanh(Zq[:, int(ja)] / 3.0)
            cand_c.append(2.0 * tc * tc - 1.0)
            cand_q.append(2.0 * tq * tq - 1.0)
            cand_c.append(4.0 * (tc ** 3) - 3.0 * tc)
            cand_q.append(4.0 * (tq ** 3) - 3.0 * tq)

    if "svd_project" in st_set and p >= 3:
        try:
            k_svd = min(3, p - 1, n_c - 1)
            if k_svd >= 1:
                Zc_center = Zc - np.mean(Zc, axis=0, keepdims=True)
                Zq_center = Zq - np.mean(Zc, axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Zc_center, full_matrices=False)
                V_k = Vt[:k_svd].T
                Sc = Zc_center @ V_k
                Sq = Zq_center @ V_k
                for ks in range(k_svd):
                    cand_c.append(Sc[:, ks])
                    cand_q.append(Sq[:, ks])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass

    if ("nystrom" in st_set or "rff_cos" in st_set) and p >= 2:
        try:
            m_lm = min(16, max(4, n_c // 4))
            lm_idx = rng.choice(n_c, size=m_lm, replace=False)
            L_mat = Zc[lm_idx]  # strictly context landmarks!
            gamma_ny = 1.0 / float(max(2 * p, 4))
            dc2 = np.sum((Zc[:, None, :] - L_mat[None, :, :]) ** 2, axis=-1)
            dq2 = np.sum((Zq[:, None, :] - L_mat[None, :, :]) ** 2, axis=-1)
            Kc = np.exp(-gamma_ny * np.clip(dc2, 0.0, 80.0))
            Kq = np.exp(-gamma_ny * np.clip(dq2, 0.0, 80.0))
            # Project onto top-2 context Nystrom directions correlated with yc
            w_ny = Kc.T @ yc0
            wn = float(np.linalg.norm(w_ny))
            if wn > 1e-12:
                cand_c.append((Kc @ w_ny) / wn)
                cand_q.append((Kq @ w_ny) / wn)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass

    if "blom" in st_set or "rank" in st_set or "bin_onehot" in st_set:
        for ja in top_idx[: min(3, len(top_idx))]:
            cs = np.sort(Zc[:, int(ja)])
            # Blom normal score `(rank - 3/8) / (n_c + 1/4)` fitted strictly against sorted `Zc`!
            rc = np.clip((np.searchsorted(cs, Zc[:, int(ja)], side="left") + 0.625) / float(n_c + 0.25), 1e-4, 1.0 - 1e-4)
            rq = np.clip((np.searchsorted(cs, Zq[:, int(ja)], side="left") + 0.625) / float(n_c + 0.25), 1e-4, 1.0 - 1e-4)
            cand_c.append(math.sqrt(2.0) * BASE._erfinv_np(2.0 * rc - 1.0))
            cand_q.append(math.sqrt(2.0) * BASE._erfinv_np(2.0 * rq - 1.0))

    if "winsor_clip" in st_set or "robust_scale" in st_set:
        for ja in top_idx[: min(3, len(top_idx))]:
            col_c = Xc[:, int(ja)]
            fin_c = col_c[np.isfinite(col_c)]
            if fin_c.size >= 8:
                q02, q25, q50, q75, q98 = np.percentile(fin_c, [2.0, 25.0, 50.0, 75.0, 98.0])
                iqr = max(float(q75 - q25), float(np.std(fin_c)) * 0.5, 1e-9)
                wc = (np.clip(np.where(np.isfinite(col_c), col_c, q50), q02, q98) - q50) / iqr
                col_q = Xq[:, int(ja)]
                wq = (np.clip(np.where(np.isfinite(col_q), col_q, q50), q02, q98) - q50) / iqr
                cand_c.append(wc)
                cand_q.append(wq)

    if "symlog" in st_set:
        c_val = float(math.exp(rng.uniform(-0.3, 0.9)))
        for ja in top_idx[: min(3, len(top_idx))]:
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

    if not cand_c:
        return inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want, rng)

    scored = []
    for idx_c, cc in enumerate(cand_c):
        if not np.all(np.isfinite(cc)) or not np.all(np.isfinite(cand_q[idx_c])):
            continue
        cc0 = cc - float(np.mean(cc))
        cn = float(np.linalg.norm(cc0))
        sc = abs(float(np.dot(cc0, yc0))) / (cn * yn) if cn > 1e-12 else 0.0
        scored.append((sc, idx_c))
    if not scored:
        return inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want, rng)
    scored.sort(key=lambda t: -t[0])
    pick = [idx_c for _, idx_c in scored[:k_want]]
    Mc = np.column_stack([cand_c[i] for i in pick])
    Mq = np.column_stack([cand_q[i] for i in pick])
    if Mc.shape[1] < k_want:
        Rc, Rq = inject_synth_cols_18fam_zero_leakage(Xc, Xq, k_want - Mc.shape[1], rng)
        Mc = np.hstack([Mc, Rc])
        Mq = np.hstack([Mq, Rq])
    return Mc[:, :k_want], Mq[:, :k_want]


def sample_exact_teacher_kd_episode(
    sh: dict,
    rng: np.random.Generator,
    cf_weight_scale: float = 0.65,
    feature_noise_max: float = 0.025,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple]]:
    """Builds a unique Exact GP Teacher KD + `2x` Counterfactual Query episode from shard `sh` (`100%` signal cols kept!)."""
    Xc_full = np.asarray(sh["Xc"], dtype=np.float64)
    yc_full = np.asarray(sh["yc"], dtype=np.float64)
    Xqr_full = np.asarray(sh["Xqr"], dtype=np.float64)
    yqr_full = np.asarray(sh["yqr"], dtype=np.float64)
    pqr_full = np.asarray(sh["pqr"], dtype=np.float64)
    Xcf_full = np.asarray(sh["Xcf"], dtype=np.float64)
    pcf_full = np.asarray(sh["pcf"], dtype=np.float64)
    regret = float(sh["regret"])
    t_idx = int(sh["table_idx"])

    n_c_tot, p_raw = Xc_full.shape
    if n_c_tot < 24 or len(yqr_full) < 6 or p_raw < 1:
        return None

    # Random `70%..100%` stratified context subsample + random query subsample so every draw is unique (`dup_fed == 0`)
    frac_c = float(rng.uniform(0.70, 1.0))
    n_c = max(24, int(round(frac_c * n_c_tot)))
    if n_c < n_c_tot:
        ord_c = np.argsort(yc_full + 1e-12 * rng.standard_normal(n_c_tot), kind="mergesort")
        c_sub = np.concatenate([rng.permutation(b)[: max(1, int(round(frac_c * len(b))))] for b in np.array_split(ord_c, 6)])
        if len(c_sub) < n_c:
            rem = np.setdiff1d(np.arange(n_c_tot), c_sub)
            c_sub = np.concatenate([c_sub, rng.permutation(rem)[: n_c - len(c_sub)]])
        c_sub = rng.permutation(c_sub[:n_c])
        Xc, yc = Xc_full[c_sub].copy(), yc_full[c_sub].copy()
    else:
        perm_c = rng.permutation(n_c_tot)
        Xc, yc = Xc_full[perm_c].copy(), yc_full[perm_c].copy()
        n_c = n_c_tot

    n_qr = min(len(yqr_full), max(8, int(round(float(rng.uniform(0.75, 1.0)) * len(yqr_full)))))
    qr_idx = rng.choice(len(yqr_full), size=n_qr, replace=False)
    Xqr, yqr, pqr = Xqr_full[qr_idx].copy(), yqr_full[qr_idx].copy(), pqr_full[qr_idx].copy()

    n_cf_want = min(len(pcf_full), 2 * n_qr)
    if n_cf_want >= 4:
        cf_idx = rng.choice(len(pcf_full), size=n_cf_want, replace=False)
        Xcf, pcf = Xcf_full[cf_idx].copy(), pcf_full[cf_idx].copy()
    else:
        Xcf = np.zeros((0, p_raw), dtype=np.float64)
        pcf = np.zeros(0, dtype=np.float64)
        n_cf_want = 0

    Xq_all = np.vstack([Xqr, Xcf]) if n_cf_want > 0 else Xqr

    # Keep 100% of signal columns (`<= 56` cols), optionally appending `0..6` context-fitted features + affine scale + perm
    if p_raw > 56:
        Xc, Xq_all = BASE._select_cols(Xc, yc, Xq_all, 56, rng)
    k_extra = int(rng.integers(0, min(7, max(1, BASE.WIDE_GPU - Xc.shape[1])))) if Xc.shape[1] < BASE.WIDE_GPU else 0
    if k_extra > 0:
        Ic, Iq = inject_synth_cols_18fam_zero_leakage(Xc, Xq_all, k_extra, rng)
        Xc, Xq_all = np.hstack([Xc, Ic]), np.hstack([Xq_all, Iq])

    Xc, Xq_all = BASE._augment_features_zero_leakage(Xc, Xq_all, rng, noise_max_frac=feature_noise_max)

    # Regret-gated teacher blend on real queries + noiseless exact teacher labels on counterfactual queries!
    alpha = float(np.clip(math.tanh(2.0 * regret), 0.15, 0.85))
    yq_real_tgt = (1.0 - alpha) * yqr + alpha * pqr
    if n_cf_want > 0:
        yq_all = np.concatenate([yq_real_tgt, pcf])
        wq_all = np.concatenate(
            [
                np.ones(n_qr, dtype=np.float32),
                np.full(n_cf_want, float(cf_weight_scale * alpha), dtype=np.float32),
            ]
        )
    else:
        yq_all = yq_real_tgt
        wq_all = np.ones(n_qr, dtype=np.float32)

    # Global equivariant target sign + log-uniform scale
    y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    y_scale = float(math.exp(rng.uniform(-1.8, 1.8)))
    y_ep = y_sign * y_scale * np.concatenate([yc, yq_all])
    X_ep = np.vstack([Xc, Xq_all])
    n_tot, F_ep = X_ep.shape

    if not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_c])) <= 1e-12:
        return None
    info = (t_idx, 2, int(k_extra), int(F_ep), int(n_tot), 0, 0, int(p_raw))
    return X_ep, y_ep, wq_all, int(n_c), info


def sample_exact_gp_resid_episode(
    sh: dict,
    rng: np.random.Generator,
    feature_noise_max: float = 0.03,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple]]:
    """Builds a 1-to-1 test-aligned Exact Appended OOF GP-Residual episode (`[y_hat_gp, X]` -> `y - y_hat_gp`)!"""
    Xc_full = np.asarray(sh["Xc"], dtype=np.float64)
    yc_full = np.asarray(sh["yc"], dtype=np.float64)
    oof_full = np.asarray(sh["oof_c"], dtype=np.float64)
    Xqr_full = np.asarray(sh["Xqr"], dtype=np.float64)
    yqr_full = np.asarray(sh["yqr"], dtype=np.float64)
    pqr_full = np.asarray(sh["pqr"], dtype=np.float64)
    t_idx = int(sh["table_idx"])

    n_c_tot, p_raw = Xc_full.shape
    if n_c_tot < 24 or len(yqr_full) < 6 or p_raw < 1:
        return None

    frac_c = float(rng.uniform(0.72, 1.0))
    n_c = max(24, int(round(frac_c * n_c_tot)))
    c_idx = rng.choice(n_c_tot, size=n_c, replace=False)
    n_qr = min(len(yqr_full), max(8, int(round(float(rng.uniform(0.75, 1.0)) * len(yqr_full)))))
    q_idx = rng.choice(len(yqr_full), size=n_qr, replace=False)

    Xc = Xc_full[c_idx].copy()
    yc = yc_full[c_idx].copy()
    oof_c = oof_full[c_idx].copy()
    Xq = Xqr_full[q_idx].copy()
    yq = yqr_full[q_idx].copy()
    pq = pqr_full[q_idx].copy()

    rc = yc - oof_c
    rq = yq - pq
    if float(np.std(rc)) <= 1e-6 * max(float(np.std(yc)), 1.0):
        return None

    if p_raw > 58:
        Xc, Xq = BASE._select_cols(Xc, rc, Xq, 58, rng)
    # APPEND `[oof_c; pq]` as a dedicated conditioning feature column (`0` real columns overwritten!)
    Xc_gp = np.column_stack([oof_c, Xc])
    Xq_gp = np.column_stack([pq, Xq])

    k_extra = int(rng.integers(0, min(5, max(1, BASE.WIDE_GPU - Xc_gp.shape[1])))) if Xc_gp.shape[1] < BASE.WIDE_GPU else 0
    if k_extra > 0:
        Ic, Iq = inject_synth_cols_18fam_zero_leakage(Xc_gp, Xq_gp, k_extra, rng)
        Xc_gp, Xq_gp = np.hstack([Xc_gp, Ic]), np.hstack([Xq_gp, Iq])

    Xc_gp, Xq_gp = BASE._augment_features_zero_leakage(Xc_gp, Xq_gp, rng, noise_max_frac=feature_noise_max)

    y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    y_scale = float(math.exp(rng.uniform(-1.8, 1.8)))
    y_ep = y_sign * y_scale * np.concatenate([rc, rq])
    X_ep = np.vstack([Xc_gp, Xq_gp])
    wq = np.ones(n_qr, dtype=np.float32)
    n_tot, F_ep = X_ep.shape

    if not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_c])) <= 1e-12:
        return None
    info = (t_idx, 0, int(k_extra), int(F_ep), int(n_tot), 0, 0, int(p_raw))
    return X_ep, y_ep, wq, int(n_c), info


def _pack_worker_item_h454(
    xa: np.ndarray,
    ya: np.ndarray,
    wq: np.ndarray,
    n_ctx: int,
    infos,
    fps: List[bytes],
    rng: np.random.Generator,
    postnorm_affine_max: float = 0.24,
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


def _flywheel_worker_loop_h454(
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
    except (OSError, RuntimeError, ValueError):
        pass

    sampling_mode = "uniform_dataset" if dataset_mix == "uniform" else "smart_stratified"
    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha, sampling_mode=sampling_mode)
    gov = BASE.HostCpuGovernor(hard_pct=cpu_hard, check_every_s=3.0)
    gp_store = HWMGPWinnerStoreV3(gp_winners_jsonl, hwm_json, bank, reload_every_s=45.0)
    shard_store = ExactGPTeacherShardStore(
        teacher_shards_dir, allow_d91=allow_d91_teacher_shards, min_regret=0.010, gamma=0.75, reload_every_s=45.0
    )
    prior: Optional[Prior] = None
    valid_tcols_cache: Dict[int, Tuple[int, ...]] = {}

    open_bins: Dict[Tuple[int, int], dict] = {}
    n_ctx_for_shape: Dict[Tuple[int, int], int] = {}
    ready_items: list = []
    allowed_row_bins = [n for n in ROW_BINS_V2 if n <= int(max_episode_rows)] or [min(32, int(max_episode_rows))]
    active_rows = allowed_row_bins[(worker_id * 3) % len(allowed_row_bins)]
    seq = 0
    real_seq = 0

    def _flush_bin(shape_key: Tuple[int, int], rng_f: np.random.Generator) -> None:
        b_st = open_bins.pop(shape_key, None)
        n_ctx_for_shape.pop(shape_key, None)
        if not b_st or not b_st["xs"]:
            return
        xs_b, ys_b, wqs_b, infos_b, n_ctx_b = b_st["xs"], b_st["ys"], b_st["wqs"], b_st["infos"], int(b_st["n_ctx"])
        fps_b = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_b, ys_b)]
        ready_items.append(
            _pack_worker_item_h454(
                np.stack(xs_b),
                np.stack(ys_b),
                np.stack(wqs_b),
                n_ctx_b,
                infos_b,
                fps_b,
                rng_f,
                postnorm_affine_max=postnorm_affine_max,
                batch_kind="real_s3",
            )
        )

    while not stop_ev.is_set():
        gp_store.maybe_reload(force=False)
        shard_store.maybe_reload(force=False)
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

        u_branch = float(rng.random())

        # Branch 1 (`10%`): Heavy Synthetic SCM / Multi-Index Prior Episodes
        if synth_prob > 0.0 and u_branch < synth_prob:
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
            wq = np.ones((xa.shape[0], N - n_ctx), dtype=np.float32)
            fps = [BASE.dataset_fingerprint_128(xa[b_], ya[b_], n_ctx) for b_ in range(xa.shape[0])]
            ready_items.append(
                _pack_worker_item_h454(
                    xa, ya, wq, int(n_ctx), None, fps, rng, postnorm_affine_max=postnorm_affine_max, batch_kind="synth"
                )
            )
            continue

        # Branch 2 (`25%`): Exact Rust GP Teacher KD + `2x` Counterfactual Queries (when shards available)
        if shard_store.shards and u_branch < (synth_prob + exact_kd_prob):
            # Pack a shape-matched mini-batch (`B = 1..8`) drawn from Exact Teacher Shards
            sh0 = shard_store.sample_shard(rng)
            if sh0 is not None:
                ep0 = sample_exact_teacher_kd_episode(sh0, rng, cf_weight_scale=cf_weight_scale)
                if ep0 is not None:
                    X0, y0, wq0, nc0, inf0 = ep0
                    N_t, Fn_t = X0.shape
                    B_tgt = BASE._target_batch_size(N_t, Fn_t, min(int(bmax), 8), tokens_cap, row_tokens_cap, attn_cap)
                    xs_k, ys_k, wqs_k, infs_k = [X0], [y0], [wq0], [inf0]
                    # Draw additional jittered/subsampled views of the same shard or shape-compatible shards
                    for _ in range(min(B_tgt - 1, 5)):
                        ep_m = sample_exact_teacher_kd_episode(sh0, rng, cf_weight_scale=cf_weight_scale)
                        if ep_m is not None and ep_m[0].shape == (N_t, Fn_t) and ep_m[3] == nc0:
                            xs_k.append(ep_m[0])
                            ys_k.append(ep_m[1])
                            wqs_k.append(ep_m[2])
                            infs_k.append(ep_m[4])
                    fps_k = [BASE.dataset_fingerprint_128(x_, y_, nc0) for x_, y_ in zip(xs_k, ys_k)]
                    ready_items.append(
                        _pack_worker_item_h454(
                            np.stack(xs_k),
                            np.stack(ys_k),
                            np.stack(wqs_k),
                            nc0,
                            infs_k,
                            fps_k,
                            rng,
                            postnorm_affine_max=0.14,
                            batch_kind="exact_kd",
                        )
                    )
                    continue

        # Branch 3 (`15%`): Exact Appended `CV-7` OOF GP-Residual Co-Training (`cols_104m_e8_gpresid_h452.py` 1-to-1!)
        if shard_store.shards and u_branch < (synth_prob + exact_kd_prob + exact_resid_prob):
            sh0 = shard_store.sample_shard(rng)
            if sh0 is not None:
                ep0 = sample_exact_gp_resid_episode(sh0, rng)
                if ep0 is not None:
                    X0, y0, wq0, nc0, inf0 = ep0
                    N_t, Fn_t = X0.shape
                    B_tgt = BASE._target_batch_size(N_t, Fn_t, min(int(bmax), 8), tokens_cap, row_tokens_cap, attn_cap)
                    xs_r, ys_r, wqs_r, infs_r = [X0], [y0], [wq0], [inf0]
                    for _ in range(min(B_tgt - 1, 5)):
                        ep_m = sample_exact_gp_resid_episode(sh0, rng)
                        if ep_m is not None and ep_m[0].shape == (N_t, Fn_t) and ep_m[3] == nc0:
                            xs_r.append(ep_m[0])
                            ys_r.append(ep_m[1])
                            wqs_r.append(ep_m[2])
                            infs_r.append(ep_m[4])
                    fps_r = [BASE.dataset_fingerprint_128(x_, y_, nc0) for x_, y_ in zip(xs_r, ys_r)]
                    ready_items.append(
                        _pack_worker_item_h454(
                            np.stack(xs_r),
                            np.stack(ys_r),
                            np.stack(wqs_r),
                            nc0,
                            infs_r,
                            fps_r,
                            rng,
                            postnorm_affine_max=0.14,
                            batch_kind="exact_resid",
                        )
                    )
                    continue

        # Branch 4 (`50%`): Extreme-Diversity Real & Stratified-Crop S3 Episodes (`0` real columns overwritten!)
        for _burst_i in range(12):
            if stop_ev.is_set():
                break
            real_seq += 1
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
                    n_ctx_for_shape=None,
                    preferred_rows=active_rows,
                )
                if got is not None:
                    break
            if got is None:
                continue

            X_ep, y_ep, n_ctx_ep, info_ep = got
            n_r, F_ep = X_ep.shape

            # Channel A v2: APPEND full-grammar GP stage features (`svd_project`, `nystrom`, `blom`, `winsor_clip`, ...)
            # NEVER overwriting real columns!
            if win_t is not None and F_ep >= 3 and F_ep < BASE.WIDE_GPU and float(rng.random()) < gp_stage_prob:
                k_gp = max(1, min(BASE.WIDE_GPU - F_ep, max(2, F_ep // 4), 8))
                Gc, Gq = build_gp_stage_cols_zero_leakage_v2(
                    X_ep[:n_ctx_ep], y_ep[:n_ctx_ep], X_ep[n_ctx_ep:], win_t["stages"], k_gp, rng
                )
                if Gc.shape[1] > 0 and np.all(np.isfinite(Gc)) and np.all(np.isfinite(Gq)):
                    X_ep = np.hstack([X_ep, np.vstack([Gc, Gq])])
                    F_ep = X_ep.shape[1]

            # Channel B fallback: APPEND OOF nonlinear residual column (`0` real columns overwritten!)
            if F_ep >= 3 and F_ep < BASE.WIDE_GPU and float(rng.random()) < gp_resid_prob:
                head_b = win_t["head"] if win_t is not None else "oblivious"
                oof_gp, pq_gp, ok_gp = nonlinear_gp_oof_residual_np(
                    X_ep[:n_ctx_ep], y_ep[:n_ctx_ep], X_ep[n_ctx_ep:], head_b, rng
                )
                if ok_gp:
                    X_ep = np.column_stack([np.concatenate([oof_gp, pq_gp]), X_ep])
                    F_ep = X_ep.shape[1]
                    y_ep = np.concatenate([y_ep[:n_ctx_ep] - oof_gp, y_ep[n_ctx_ep:] - pq_gp])

            if not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_ep])) <= 1e-12:
                continue

            # Update `info_ep` width `F` if columns were appended
            info_ep = (info_ep[0], info_ep[1], info_ep[2], int(F_ep), int(n_r), info_ep[5], info_ep[6], info_ep[7])
            sh_key = (int(X_ep.shape[0]), int(X_ep.shape[1]))
            b_st = open_bins.get(sh_key)
            if b_st is None or int(b_st["n_ctx"]) != int(n_ctx_ep):
                if b_st is not None:
                    _flush_bin(sh_key, rng)
                b_tgt = BASE._target_batch_size(sh_key[0], sh_key[1], min(int(bmax), 24), tokens_cap, row_tokens_cap, attn_cap)
                b_st = {
                    "xs": [],
                    "ys": [],
                    "wqs": [],
                    "infos": [],
                    "n_ctx": int(n_ctx_ep),
                    "b_target": b_tgt,
                    "open_seq": real_seq,
                }
                open_bins[sh_key] = b_st

            b_st["xs"].append(X_ep)
            b_st["ys"].append(y_ep)
            b_st["wqs"].append(np.ones(n_r - int(n_ctx_ep), dtype=np.float32))
            b_st["infos"].append(info_ep)
            if len(b_st["xs"]) >= int(b_st["b_target"]):
                _flush_bin(sh_key, rng)

            if (real_seq & 31) == 0 and open_bins:
                stale_keys = [k_s for k_s, st_s in open_bins.items() if (real_seq - int(st_s["open_seq"])) >= 160]
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
    ap.add_argument("--teacher-shards-dir", default="")
    ap.add_argument("--allow-d91-teacher-shards", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hwm-json", default="")
    ap.add_argument("--gp-winners-jsonl", default="")
    ap.add_argument("--exact-kd-prob", type=float, default=0.25)
    ap.add_argument("--exact-resid-prob", type=float, default=0.15)
    ap.add_argument("--cf-weight-scale", type=float, default=0.65)
    ap.add_argument("--gp-stage-prob", type=float, default=0.50)
    ap.add_argument("--gp-resid-prob", type=float, default=0.20)
    ap.add_argument("--regret-tilt-prob", type=float, default=0.25)
    ap.add_argument("--d91-random-split-boost", type=float, default=0.20)
    ap.add_argument("--save-every-datasets", type=int, default=100_000)
    ap.add_argument("--keep-latest-only", type=int, choices=(0, 1), default=1)
    ap.add_argument("--last-every-s", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.2e-5)
    ap.add_argument("--swa-decay", type=float, default=0.9995)
    ap.add_argument("--synth-prob", type=float, default=0.10)
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
    ap.add_argument("--rand-target-prob", type=float, default=0.35)
    ap.add_argument("--synth-teacher-prob", type=float, default=0.15)
    ap.add_argument("--feature-noise-max", type=float, default=0.06)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.24)
    ap.add_argument("--seed", type=int, default=454201)
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
        "exact_kd_batches": 0,
        "exact_resid_batches": 0,
        "exact_kd_datasets": 0,
        "exact_resid_datasets": 0,
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
        state.setdefault("exact_kd_batches", 0)
        state.setdefault("exact_resid_batches", 0)
        state.setdefault("exact_kd_datasets", 0)
        state.setdefault("exact_resid_datasets", 0)
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
        "protocol": "h454-exact-gp-teacher-flywheel",
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "resume_step": int(state["step"]),
        "resume_datasets": int(state["datasets"]),
        "lr_base": float(a.lr),
        "lr_cooldown_hours": float(a.lr_cooldown_hours),
        "lr_min": float(a.lr_min),
        "teacher_shards_dir": str(a.teacher_shards_dir),
        "allow_d91_teacher_shards": bool(a.allow_d91_teacher_shards),
        "canonical_promotion_forbidden": bool(a.allow_d91_teacher_shards or a.d91_random_split_boost > 0),
        "exact_kd_prob": float(a.exact_kd_prob),
        "exact_resid_prob": float(a.exact_resid_prob),
        "cf_weight_scale": float(a.cf_weight_scale),
        "synth_prob": float(a.synth_prob),
        "gp_stage_prob": float(a.gp_stage_prob),
        "gp_resid_prob": float(a.gp_resid_prob),
        "regret_tilt_prob": float(a.regret_tilt_prob),
        "d91_random_split_boost": float(a.d91_random_split_boost),
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
            target=_flywheel_worker_loop_h454,
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
                                "schema": "jope.h454.epoch/v1",
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
                            "exact_kd_batches": state["exact_kd_batches"],
                            "exact_resid_batches": state["exact_resid_batches"],
                            "exact_kd_datasets": state["exact_kd_datasets"],
                            "exact_resid_datasets": state["exact_resid_datasets"],
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
