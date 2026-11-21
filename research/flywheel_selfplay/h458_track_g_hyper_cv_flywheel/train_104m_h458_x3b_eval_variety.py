#!/usr/bin/env python3
"""Wave H458 `X3-B` Evaluation-Aligned Maximum-Variety `104M` (`ICL_L`) Flywheel Trainer
(`train_104m_h458_x3b_eval_variety.py`).

Resolves the paired H450 `r4` DiD pathology (`r4_L_B` gained `+0.0056 u` stack complementarity from variety
but lost `-0.0229 u` own-column skill due to `85%` feature drop and `60%` synthetic clutter) and eliminates the
`6.46%` `Episode-Test` loser GP teacher shards measured in `v1_backfill_cv_to_test_ledger.npz`:

  1. Strict `S8` + `B1` + `B7` Quarantine Mask across ALL 9 Cohorts (`build_s8_clean_eligibility_mask`):
     - Rejects empty family keys (`B1`) and masks every table in every cohort (including all `258` `DL-TRAIN`
       `views_canonical_confirm` tables, `B7`) whose family hashes to `APEX-split-v1` bucket `0 mod 5`.
     - Zero `CONFIRM`, `FMGAP B/C`, `DL-HOLDOUT`, `SEALED`, or `DL-VAL` opens (`canonical_promotion_forbidden = true`).
  2. Evaluation-Aligned Wide-Table `|corr|` Column Selection (`select_wide_cols_eval_aligned`, `B5`):
     - Context-only (`Xc, yc`) top-70% `|corr|` + 30% uniform random reducer (`0.0` query leakage), matching
       `cols_h450_gpu` evaluation preparation.
  3. 3-Stream Maximum-Variety Episode Mix with `100%` Random Feature-ID Slots (`cl_h458` K=5 unseen-stratum proof):
     - `55%` `eval_clean`: `0.0` feature drop (`100%` real columns up to `f_bin`), `0.0` synthetic clutter,
       context-fitted `yj`/`rg`/`raw` target modes (`50% / 25% / 25%`), 4-regime `p_ctx in [0.12, 0.94]` split.
     - `23%` `subspace_clean`: bounded moderate feature drop `[0.05, 0.35]` + light context-only `CW-1`/`CW-2`
       tabular interactions (`<= 0.20` frac, `0.05` distractor cap).
     - `22%` Zero-Loser Exact-GP Distillation (`14% exact_kd` + `8% exact_resid`) from `gp_teacher_shards_v2`
       (`0.0%` held-out `Episode-Test` losers, `S8`-filtered by `family_id`, clean student features `drop <= 0.15`).
  4. Hard `xbabe3` VRAM, Host RAM & Disk Guardrails (`--keep-latest-only 1`, `--valley-ring-slots 2`):
     - Atomically overwrites only `ckpt_last.pt`, `ckpt_eval_latest.pt`, and `ckpt_valley_slot{0..K-1}.pt`
       (`<= 4.0 GB` max checkpoint footprint, zero unbounded `ckpt_valley_NNN.pt` files).
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if sys.path[:1] != [str(HERE)]:
    sys.path.insert(0, str(HERE))
REF_CODE = os.environ.get("JOPE_REF_CODE", "")
for _cand in (
    REF_CODE,
    str(HERE.parent / "h455_fast_wave2_maxent"),
    "/home/ubuntu/jope-data/h450-fleet-agy/stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

from mmap_bank_h450 import SEALED_TOKENS, MmapF64Bank, sha256_file  # noqa: E402
import train_mps_104m_h450 as BASE  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from train_104m_gp_flywheel_h452 import (  # noqa: E402
    ICLH450MPS,
    apply_monotone_warps_and_nans_zero_leakage,
    sample_extreme_stratified_episode_h452,
)
from train_104m_gp_flywheel_h454 import (  # noqa: E402
    HWMGPWinnerStoreV3,
    build_gp_stage_cols_zero_leakage_v2,
)
from train_104m_gp_flywheel_h455_maxent import (  # noqa: E402
    F_GRID_H455,
    N_GRID_H455,
    TEACHER_F_GRID_H455,
    TEACHER_N_GRID_H455,
    _pack_worker_item_h455,
    _resample_rows_disjoint,
    compute_cyclic_valley_lr,
    fit_cols_to_fbin_h455,
    sample_binned_exact_kd_episode_h455,
    sample_binned_exact_resid_episode_h455,
)


S4_EXCLUDED_CIDS_BUILTIN = frozenset(
    {
        "1a2c07f148e714ffef6652e5d8c0d5b63d5ab021ca5cb5841d3c65b0ec2f7efc",
        "4e3c00388b4e6a4c",
    }
)
DEFAULT_EXCLUDE_CIDS_CANDIDATES = (
    "/home/ubuntu/jope-data/h458-x3b/banks/s8_b7_s4_excluded_cids.json",
    "/home/ubuntu/jope-data/h458-flywheel/banks/s8_b7_s4_excluded_cids.json",
)


def is_canonical_confirm_family(fk: str) -> bool:
    """Return True if `fk` is empty (`B1`), sealed, or canonical `APEX-split-v1` CONFIRM (`hexdigest[:8] mod 5 == 0`)."""
    fk_clean = str(fk).strip()
    if not fk_clean:
        return True
    if any(tok in fk_clean.upper() for tok in SEALED_TOKENS) or "4e3c00388b4e6a4c" in fk_clean.lower():
        return True
    bare = fk_clean
    for pfx in ("fresh:", "s3:"):
        if bare.startswith(pfx):
            bare = bare[len(pfx) :]
            break
    if not bare or "meta-album-md-6-mini" in bare.lower():
        return True
    if bare.startswith(("fam-", "d91:")):
        # Opaque DL-TRAIN group hashes (`fam-...`) and dev91 envelopes (`d91:...`) are checked via exact CID mask (`B7`).
        return False
    d_hex = hashlib.sha256(b"APEX-split-v1\x00" + bare.encode("utf-8")).hexdigest()
    return (int(d_hex[:8], 16) % 5) == 0


def load_excluded_cids_set(exclude_cids_json: str = "") -> set[str]:
    """Load exact `S8 + B7 + S4` excluded CIDs (`258` B7 on `r3`, `530` S8+B7+S4 on `v1`)."""
    out = set(S4_EXCLUDED_CIDS_BUILTIN)
    cands = [exclude_cids_json] if exclude_cids_json else list(DEFAULT_EXCLUDE_CIDS_CANDIDATES)
    for cp in cands:
        if cp and Path(cp).is_file():
            doc = json.loads(Path(cp).read_text())
            for k in ("all_excluded_cids", "b7_dl_train_canonical_confirm_cids", "s4_excluded_cids"):
                if isinstance(doc.get(k), list):
                    out.update(str(x).strip() for x in doc[k] if str(x).strip())
            break
    return out


def build_s8_clean_eligibility_mask(bank: MmapF64Bank, exclude_cids_json: str = "") -> Tuple[np.ndarray, int]:
    """Exclude empty family keys (`B1`), `S4`, and all `S8 + B7` canonical `APEX-CONFIRM` tables across all cohorts."""
    ex_cids = load_excluded_cids_set(exclude_cids_json)
    elig = np.ones(bank.n_tables, dtype=np.float64)
    n_masked = 0
    for idx, fam in enumerate(bank.families):
        cid_s = str(bank.cids[idx]).strip()
        coh = str(bank.cohorts[idx])
        if (
            cid_s in ex_cids
            or any(tok in coh.upper() for tok in SEALED_TOKENS)
            or any(tok in cid_s.upper() for tok in SEALED_TOKENS)
            or is_canonical_confirm_family(str(fam))
        ):
            elig[idx] = 0.0
            n_masked += 1
    return elig, n_masked


class S8CleanFamilySampler:
    """O(1) two-stage family-uniform sampler strictly over `S8`-clean eligible tables."""

    def __init__(self, bank: MmapF64Bank, elig_mask: np.ndarray):
        by_fam: Dict[str, List[int]] = {}
        tainted_fams: set[str] = set()
        for idx in range(bank.n_tables):
            fk = str(bank.families[idx]).strip()
            if elig_mask[idx] <= 0.0:
                if fk:
                    tainted_fams.add(fk)
                continue
            if not fk or is_canonical_confirm_family(fk):
                continue
            by_fam.setdefault(fk, []).append(int(idx))
        if not by_fam:
            raise SystemExit("S8CleanFamilySampler found 0 eligible tables")
        self.families: List[str] = sorted(by_fam.keys())
        self.fam_tables: List[Tuple[int, ...]] = [tuple(by_fam[f]) for f in self.families]
        self.fam_to_first_idx: Dict[str, int] = {f: int(by_fam[f][0]) for f in self.families}
        self.tainted_fams: frozenset[str] = frozenset(tainted_fams)
        self.n_clean_families = len(self.families)
        self.n_clean_tables = sum(len(t) for t in self.fam_tables)

    def sample_table_idx(self, rng: np.random.Generator) -> int:
        f_i = int(rng.integers(0, self.n_clean_families))
        cand = self.fam_tables[f_i]
        return int(cand[0]) if len(cand) == 1 else int(cand[int(rng.integers(0, len(cand)))])


def select_wide_cols_eval_aligned(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    max_cols: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Context-only `|corr|` top-70% + random 30% column reducer (`0.0` `Xq` leakage, matching `cols_h450_gpu`)."""
    p = int(Xc.shape[1])
    if p <= max_cols or max_cols <= 0:
        return Xc, Xq
    k_top = max(1, min(max_cols, int(round(0.70 * max_cols))))
    k_rnd = max(0, max_cols - k_top)
    yc_f = np.asarray(yc, dtype=np.float64)
    ym = float(np.nanmean(yc_f)) if np.any(np.isfinite(yc_f)) else 0.0
    yc_c = np.nan_to_num(yc_f - ym, nan=0.0)
    xc_m = np.nanmean(Xc, axis=0, keepdims=True)
    xc_m = np.where(np.isfinite(xc_m), xc_m, 0.0)
    xc_c = np.nan_to_num(Xc - xc_m, nan=0.0)
    denom = np.sqrt(np.sum(xc_c * xc_c, axis=0) * max(float(np.sum(yc_c * yc_c)), 1e-12))
    corr = np.abs(np.sum(xc_c * yc_c[:, None], axis=0)) / np.maximum(denom, 1e-12)
    # Deterministic tie-break by column index
    order = np.lexsort((np.arange(p, dtype=np.int64), -corr))
    top_idx = order[:k_top]
    rem = order[k_top:]
    if k_rnd > 0 and rem.size > 0:
        rnd_pick = rng.choice(rem, size=min(k_rnd, int(rem.size)), replace=False)
        keep = np.sort(np.concatenate([top_idx, rnd_pick]))
    else:
        keep = np.sort(top_idx)
    return Xc[:, keep], Xq[:, keep]


def _erfinv_approx(u: np.ndarray) -> np.ndarray:
    """Smooth rational Winitzki approximation to `erfinv(u)` on `(-0.9995, 0.9995)` without SciPy worker overhead."""
    x = np.clip(u, -0.9995, 0.9995)
    a = 0.147
    ln = np.log(np.maximum(1.0 - x * x, 1e-12))
    t1 = 2.0 / (math.pi * a) + 0.5 * ln
    rad = np.maximum(t1 * t1 - ln / a, 0.0)
    return np.sign(x) * np.sqrt(np.maximum(np.sqrt(rad) - t1, 0.0))


def _yeo_johnson_1d(z: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    zp = z[pos]
    zn = -z[~pos]
    if abs(lam) < 1e-6:
        out[pos] = np.log1p(zp)
    else:
        out[pos] = (np.power(1.0 + zp, lam) - 1.0) / lam
    if abs(2.0 - lam) < 1e-6:
        out[~pos] = -np.log1p(zn)
    else:
        out[~pos] = -(np.power(1.0 + zn, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def apply_eval_aligned_target_prep_zero_leakage(
    yc: np.ndarray,
    yq: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Context-only (`yc`) target prep matching evaluation modes (`0=raw_winsor`, `1=yj`, `2=rank_gauss`, `0.0` `yq` leak)."""
    yc_f = np.asarray(yc, dtype=np.float64)
    yq_f = np.asarray(yq, dtype=np.float64)
    mu_c = float(np.mean(yc_f))
    sd_c = max(float(np.std(yc_f)), 1e-8)
    zc = np.clip((yc_f - mu_c) / sd_c, -12.0, 12.0)
    zq = np.clip((yq_f - mu_c) / sd_c, -12.0, 12.0)

    u = float(rng.random())
    if u < 0.50:
        # Context-selected Yeo-Johnson (`lam` minimizing context skewness across fixed grid)
        best_lam = 1.0
        best_skew = 1e18
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            tc = _yeo_johnson_1d(zc, lam)
            s_tc = float(np.std(tc))
            if s_tc <= 1e-9 or not np.all(np.isfinite(tc)):
                continue
            sk = abs(float(np.mean(((tc - float(np.mean(tc))) / s_tc) ** 3)))
            if sk < best_skew:
                best_skew = sk
                best_lam = lam
        yc_t = _yeo_johnson_1d(zc, best_lam)
        yq_t = _yeo_johnson_1d(zq, best_lam)
        tmode = 1
    elif u < 0.75:
        # Context-fitted Rank-Gauss via piecewise-linear context ECDF (`0.0` query leakage)
        xs = np.sort(zc)
        n_c = len(xs)
        probs = (np.arange(n_c, dtype=np.float64) + 0.5) / float(n_c)
        uc = np.interp(zc, xs, probs, left=0.5 / n_c, right=1.0 - 0.5 / n_c)
        uq = np.interp(zq, xs, probs, left=0.5 / n_c, right=1.0 - 0.5 / n_c)
        yc_t = math.sqrt(2.0) * _erfinv_approx(2.0 * uc - 1.0)
        yq_t = math.sqrt(2.0) * _erfinv_approx(2.0 * uq - 1.0)
        tmode = 2
    else:
        lo_w, hi_w = np.percentile(zc, [0.5, 99.5])
        if hi_w <= lo_w + 1e-8:
            lo_w, hi_w = -8.0, 8.0
        yc_t = np.clip(zc, lo_w, hi_w)
        yq_t = np.clip(zq, lo_w, hi_w)
        tmode = 0

    m_out = float(np.mean(yc_t))
    s_out = max(float(np.std(yc_t)), 1e-8)
    yc_out = (yc_t - m_out) / s_out
    yq_out = np.clip((yq_t - m_out) / s_out, -12.0, 12.0)
    if not (np.all(np.isfinite(yc_out)) and np.all(np.isfinite(yq_out))):
        return zc, zq, 0
    return yc_out, yq_out, tmode


def draw_p_ctx_x3b(rng: np.random.Generator, p_min: float = 0.12, p_max: float = 0.94) -> float:
    """4-regime context fraction (`45%` CV-dense `[0.78, 0.94]` matching `CV-7`'s `6/7 = 0.857`)."""
    u = float(rng.random())
    if u < 0.15:
        lo, hi = p_min, min(0.40, p_max)
    elif u < 0.35:
        lo, hi = max(p_min, 0.40), min(0.62, p_max)
    elif u < 0.55:
        lo, hi = max(p_min, 0.62), min(0.78, p_max)
    else:
        lo, hi = max(p_min, 0.78), p_max
    if hi <= lo:
        return float(0.5 * (p_min + p_max))
    return float(rng.uniform(lo, hi))


class ZeroLoserShardStoreX3B:
    """Family-resolved loader for `gp_teacher_shards_v2` enforcing `S8` quarantine across bank versions."""

    def __init__(self, shards_dir: str | Path, fam_sampler: S8CleanFamilySampler, reload_every_s: float = 60.0):
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.fam_sampler = fam_sampler
        self.reload_every_s = float(reload_every_s)
        self._last_check = 0.0
        self._loaded_paths: set[str] = set()
        self.shards: List[dict] = []
        self.probs: Optional[np.ndarray] = None
        self.s8_rejected = 0
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
                self._loaded_paths.add(ps)
                try:
                    with np.load(p, allow_pickle=False) as z:
                        fam_s = str(z["family_id"]).strip() if "family_id" in z.files else ""
                        if is_canonical_confirm_family(fam_s) or fam_s in self.fam_sampler.tainted_fams:
                            self.s8_rejected += 1
                            continue
                        regret = float(z["regret_vs_base"])
                        if not math.isfinite(regret) or regret < 0.015:
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
                            continue
                        mapped_tidx = int(self.fam_sampler.fam_to_first_idx.get(fam_s, 0))
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
                                "table_idx": mapped_tidx,
                                "family_key": fam_s,
                            }
                        )
                        added += 1
                except (OSError, ValueError, KeyError, RuntimeError):
                    continue
            if added > 0 and self.shards:
                if len(self.shards) > 4000:
                    by_f: Dict[str, List[dict]] = {}
                    for sh in self.shards:
                        by_f.setdefault(str(sh["family_key"]), []).append(sh)
                    trimmed: List[dict] = []
                    for lst in by_f.values():
                        lst.sort(key=lambda s: -float(s["regret"]))
                        trimmed.extend(lst[:3])
                    self.shards = trimmed
                fam_cnt: Dict[str, int] = {}
                for sh in self.shards:
                    fk = str(sh["family_key"])
                    fam_cnt[fk] = fam_cnt.get(fk, 0) + 1
                w = np.array(
                    [
                        (min(max(float(s["regret"]), 0.015), 1.50) ** 0.65)
                        / float(fam_cnt[str(s["family_key"])]) ** 0.80
                        for s in self.shards
                    ],
                    dtype=np.float64,
                )
                self.probs = w / float(np.sum(w))
        except (OSError, ValueError, KeyError, RuntimeError):
            pass

    def sample_shard(self, rng: np.random.Generator) -> Optional[dict]:
        if not self.shards or self.probs is None:
            return None
        return self.shards[int(rng.choice(len(self.shards), p=self.probs))]


def sample_x3b_eval_or_subspace_episode(
    bank: MmapF64Bank,
    t_idx: int,
    n_bin: int,
    f_bin: int,
    n_ctx_bin: int,
    rng: np.random.Generator,
    valid_tcols_cache: Dict[int, Tuple[int, ...]],
    win_t: Optional[dict],
    eval_clean: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """Sample either an evaluation-aligned clean view (`eval_clean=True`) or bounded `<=0.35` subspace view."""
    got = sample_extreme_stratified_episode_h452(
        bank,
        t_idx,
        rng,
        valid_tcols_cache,
        winner=win_t,
        force_fractional_crop=(float(rng.random()) < 0.75),
        max_episode_rows=max(n_bin, 256),
        drop_min_frac=0.0 if eval_clean else 0.05,
        drop_max_frac=0.0 if eval_clean else 0.35,
        inj_min_frac=0.0,
        inj_max_frac=0.0,
        rand_target_prob=0.10 if eval_clean else 0.25,
        synth_teacher_prob=0.0 if eval_clean else 0.06,
        feature_noise_max=0.012 if eval_clean else 0.030,
        force_canonical_target=eval_clean and (float(rng.random()) < 0.85),
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

    Xc, yc, _ = _resample_rows_disjoint(Xc_src, yc_src, n_ctx_bin, rng, stratify_y=True)
    Xq, yq, _ = _resample_rows_disjoint(Xq_src, yq_src, n_q_bin, rng, stratify_y=False)
    yc, yq, tmode = apply_eval_aligned_target_prep_zero_leakage(yc, yq, rng)

    if Xc.shape[1] > f_bin:
        Xc, Xq = select_wide_cols_eval_aligned(Xc, yc, Xq, f_bin, rng)

    if not eval_clean and win_t is not None and 2 <= Xc.shape[1] < f_bin and float(rng.random()) < 0.35:
        k_gp = max(1, min(f_bin - Xc.shape[1], 4))
        Gc, Gq = build_gp_stage_cols_zero_leakage_v2(Xc, yc, Xq, win_t["stages"], k_gp, rng)
        if Gc.shape[1] > 0 and np.all(np.isfinite(Gc)) and np.all(np.isfinite(Gq)):
            Xc = np.hstack([Xc, Gc])
            Xq = np.hstack([Xq, Gq])

    if not eval_clean and float(rng.random()) < 0.30:
        Xc, Xq = apply_monotone_warps_and_nans_zero_leakage(Xc, Xq, rng, warp_prob=0.18, nan_prob=0.10)

    # Fit cleanly to `f_bin`: `0.0` pure distractors in `eval_clean`, `<= 0.05` distractor cap in `subspace_clean`
    Xc, Xq, k_inj = fit_cols_to_fbin_h455(
        Xc,
        yc,
        Xq,
        f_bin,
        rng,
        pin_slot0=False,
        distractor_prob=0.0 if eval_clean else 0.05,
    )

    y_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    y_scale = float(math.exp(rng.uniform(-1.2, 1.2)))
    y_ep = y_sign * y_scale * np.concatenate([yc, yq])
    X_ep = np.vstack([Xc, Xq])
    wq = np.ones(n_q_bin, dtype=np.float32)
    if X_ep.shape != (n_bin, f_bin) or not np.all(np.isfinite(y_ep)) or float(np.std(y_ep[:n_ctx_bin])) <= 1e-12:
        return None
    sp_safe = int(info_raw[5]) if 0 <= int(info_raw[5]) <= 2 else 0
    info = (int(t_idx), int(tmode), int(k_inj), int(f_bin), int(n_bin), sp_safe, int(info_raw[6]), int(info_raw[7]))
    return X_ep, y_ep, wq, info


def save_valley_ckpt_bounded(
    out_dir: Path,
    payload: dict,
    cyc_idx: int,
    keep_latest_only: bool,
    ring_slots: int = 2,
) -> Tuple[Path, str]:
    """Overwrite bounded valley ring slot (`ckpt_valley_slot{0..K-1}.pt`) when `keep_latest_only` is True."""
    if keep_latest_only:
        slot = int(cyc_idx) % max(1, int(ring_slots))
        vp = out_dir / f"ckpt_valley_slot{slot}.pt"
    else:
        vp = out_dir / f"ckpt_valley_{cyc_idx:03d}.pt"
    BASE._atomic_save(payload, vp, mode=0o644)
    return vp, sha256_file(vp)


def _x3b_worker_loop(
    worker_id: int,
    run_nonce: bytes,
    mmap_dir_s: str,
    expect_sha: str,
    max_episode_rows: int,
    tokens_cap: int,
    row_tokens_cap: int,
    attn_cap: int,
    bmax: int,
    cpu_hard: float,
    eval_clean_prob: float,
    subspace_prob: float,
    exact_kd_prob: float,
    exact_resid_prob: float,
    cf_weight_scale: float,
    p_ctx_min: float,
    p_ctx_max: float,
    postnorm_affine_max: float,
    teacher_shards_dir: str,
    hwm_json: str,
    gp_winners_jsonl: str,
    exclude_cids_json: str,
    out_q,
    stop_ev,
) -> None:
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.nice(12)
        torch.set_num_threads(1)
    except (OSError, ValueError, RuntimeError):
        pass

    bank = MmapF64Bank(Path(mmap_dir_s), expect_sha256=expect_sha, sampling_mode="s3_source_uniform")
    elig_mask, _ = build_s8_clean_eligibility_mask(bank, exclude_cids_json=exclude_cids_json)
    fam_sampler = S8CleanFamilySampler(bank, elig_mask)
    gov = BASE.HostCpuGovernor(hard_pct=cpu_hard, check_every_s=3.0)
    gp_store = HWMGPWinnerStoreV3(gp_winners_jsonl, hwm_json, bank, reload_every_s=60.0)
    shard_store = ZeroLoserShardStoreX3B(teacher_shards_dir, fam_sampler, reload_every_s=60.0)
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

        # Stream A (`exact_kd_prob`, default 14%): Zero-loser Exact-GP KD with clean student features (`drop <= 0.15`)
        if len(shard_store.shards) >= 4 and u_stream < exact_kd_prob:
            n_b = int(rng.choice(allowed_n_t))
            f_b = int(rng.choice(TEACHER_F_GRID_H455))
            p_c = draw_p_ctx_x3b(rng, p_ctx_min, p_ctx_max)
            n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))
            b_tgt = max(4, BASE._target_batch_size(n_b, f_b, min(int(bmax), 16), tokens_cap, row_tokens_cap, attn_cap))
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
                    drop_max_frac=0.15,
                    feature_noise_max=0.015,
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

        # Stream B (`exact_resid_prob`, default 8%): Zero-loser Appended OOF GP-Residual (`drop <= 0.15`)
        if len(shard_store.shards) >= 4 and u_stream < (exact_kd_prob + exact_resid_prob):
            n_b = int(rng.choice(allowed_n_t))
            f_b = int(rng.choice(TEACHER_F_GRID_H455))
            p_c = draw_p_ctx_x3b(rng, p_ctx_min, p_ctx_max)
            n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))
            b_tgt = max(4, BASE._target_batch_size(n_b, f_b, min(int(bmax), 16), tokens_cap, row_tokens_cap, attn_cap))
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
                    drop_max_frac=0.15,
                    feature_noise_max=0.015,
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

        # Stream C (`eval_clean`, default 55%) vs Stream D (`subspace_clean`, default 23%)
        rem_tot = max(1e-6, eval_clean_prob + subspace_prob)
        is_eval_clean = float(rng.random()) < (eval_clean_prob / rem_tot)
        n_b = int(rng.choice(allowed_n_s3))
        p_c = draw_p_ctx_x3b(rng, p_ctx_min, p_ctx_max)
        n_ctx_b = max(14, min(n_b - 6, int(round(p_c * n_b))))

        # Pick an initial S8-clean table so `f_b` matches natural width on `eval_clean`!
        first_t = fam_sampler.sample_table_idx(rng)
        p_first = max(4, min(64, int(bank.cols_arr[first_t])))
        if is_eval_clean:
            cand_f = [fb for fb in F_GRID_H455 if fb >= p_first]
            f_b = int(cand_f[0]) if cand_f else 64
        else:
            f_b = int(rng.choice(F_GRID_H455))

        b_tgt = max(3, BASE._target_batch_size(n_b, f_b, min(int(bmax), 16), tokens_cap, row_tokens_cap, attn_cap))
        xs_l, ys_l, wqs_l, infos_l = [], [], [], []
        for k_try in range(b_tgt * 3):
            if len(xs_l) >= b_tgt or stop_ev.is_set():
                break
            t_idx = first_t if k_try == 0 else fam_sampler.sample_table_idx(rng)
            win_t = gp_store.pick_winner_for_table(t_idx, rng)
            ep_x = sample_x3b_eval_or_subspace_episode(
                bank,
                t_idx,
                n_b,
                f_b,
                n_ctx_b,
                rng,
                valid_tcols_cache,
                win_t,
                eval_clean=is_eval_clean,
            )
            if ep_x is not None:
                xs_l.append(ep_x[0])
                ys_l.append(ep_x[1])
                wqs_l.append(ep_x[2])
                infos_l.append(ep_x[3])

        if len(xs_l) >= 2:
            fps_x = [BASE.dataset_fingerprint_128(x_, y_, n_ctx_b) for x_, y_ in zip(xs_l, ys_l)]
            item = _pack_worker_item_h455(
                np.stack(xs_l),
                np.stack(ys_l),
                np.stack(wqs_l),
                n_ctx_b,
                infos_l,
                fps_x,
                rng,
                postnorm_affine_max=(0.12 if is_eval_clean else postnorm_affine_max),
                batch_kind=("eval_clean" if is_eval_clean else "subspace_clean"),
            )
            while not stop_ev.is_set():
                try:
                    out_q.put(item, timeout=1.0)
                    break
                except queue.Full:
                    gov.pace()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="h458_x3b_L_eval_variety_v1")
    ap.add_argument("--size", choices=("S", "M", "L"), default="L")
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-sha256", required=True)
    ap.add_argument("--mmap-bank", required=True)
    ap.add_argument("--bank-sha256", default="9df954377259fc9b4cfbf1e878c801fe42f7df472d4b8f4edb95691fd423e3f4")
    ap.add_argument("--exclude-cids-json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endless", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hours", type=float, default=960.0)
    ap.add_argument("--lr", type=float, default=1.2e-5)
    ap.add_argument("--lr-cycle-peak", type=float, default=6.5e-6)
    ap.add_argument("--lr-min", type=float, default=2.0e-6)
    ap.add_argument("--cycle0-hours", type=float, default=2.0)
    ap.add_argument("--cycle-hours", type=float, default=2.0)
    ap.add_argument("--reset-cycle-start", type=int, choices=(0, 1), default=0)
    ap.add_argument("--teacher-shards-dir", default="")
    ap.add_argument("--hwm-json", default="")
    ap.add_argument("--gp-winners-jsonl", default="")
    ap.add_argument("--eval-clean-prob", type=float, default=0.55)
    ap.add_argument("--subspace-prob", type=float, default=0.23)
    ap.add_argument("--exact-kd-prob", type=float, default=0.14)
    ap.add_argument("--exact-resid-prob", type=float, default=0.08)
    ap.add_argument("--cf-weight-scale", type=float, default=0.55)
    ap.add_argument("--p-ctx-min", type=float, default=0.12)
    ap.add_argument("--p-ctx-max", type=float, default=0.94)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.18)
    ap.add_argument("--max-episode-rows", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=112_000)
    ap.add_argument("--row-tokens", type=int, default=6_144)
    ap.add_argument("--attn-cap", type=int, default=2_600_000)
    ap.add_argument("--bmax", type=int, default=10)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-row-blocks", type=int, choices=(0, 1), default=1)
    ap.add_argument("--cpu-hard-pct", type=float, default=82.0)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.58)
    ap.add_argument("--keep-latest-only", type=int, choices=(0, 1), default=1)
    ap.add_argument("--valley-ring-slots", type=int, default=2)
    ap.add_argument("--save-every-datasets", type=int, default=50_000)
    ap.add_argument("--last-every-s", type=float, default=240.0)
    ap.add_argument("--swa-decay", type=float, default=0.9993)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=458301)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_p = out_dir / "ckpt_last.pt"
    eval_latest_p = out_dir / "ckpt_eval_latest.pt"
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    if sha256_file(Path(a.init)) != a.init_sha256:
        raise SystemExit("init checkpoint sha256 mismatch")

    t_load0 = time.time()
    bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, verify_data_sha=True, sampling_mode="s3_source_uniform")
    elig_mask, n_s8_masked = build_s8_clean_eligibility_mask(bank, exclude_cids_json=str(a.exclude_cids_json))
    fam_sampler = S8CleanFamilySampler(bank, elig_mask)
    bank_load_s = round(time.time() - t_load0, 3)
    ledger = BASE.Ledger128(bank)
    gov = BASE.HostCpuGovernor(hard_pct=float(a.cpu_hard_pct), check_every_s=5.0)

    if a.device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if 0.05 < float(a.cuda_vram_frac) <= 0.72:
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
        ema_valley.load_state_dict(ck.get("ema_valley", ck["ema"]), strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        state = ck["state"]
        ledger.load(ck["ledger"])
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
            "eval_clean_batches": 0,
            "subspace_batches": 0,
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
            "oom_trims": 0,
            "nonfinite_skips": 0,
        }

    sampler_cfg = {
        "protocol": "h458-x3b-eval-aligned-max-variety-random-slots",
        "run_id": str(a.run_id),
        "canonical_promotion_forbidden": True,
        "confirm_opens": 0,
        "fmgap_part_bc_opens": 0,
        "slot_mode": "random_100pct",
        "bank_tables_total": bank.n_tables,
        "s8_b7_tables_masked": int(n_s8_masked),
        "s8_clean_tables_eligible": fam_sampler.n_clean_tables,
        "s8_clean_families_eligible": fam_sampler.n_clean_families,
        "keep_latest_only": bool(a.keep_latest_only),
        "valley_ring_slots": int(a.valley_ring_slots),
        "cuda_vram_frac": float(a.cuda_vram_frac),
        "stream_probs": {
            "eval_clean": float(a.eval_clean_prob),
            "subspace_clean": float(a.subspace_prob),
            "exact_kd_v2": float(a.exact_kd_prob),
            "exact_resid_v2": float(a.exact_resid_prob),
        },
        "lr": float(a.lr),
        "lr_cycle_peak": float(a.lr_cycle_peak),
        "lr_min": float(a.lr_min),
        "cycle0_hours": float(a.cycle0_hours),
        "cycle_hours": float(a.cycle_hours),
    }

    stop_flag = {"stop": False}

    def _on_sig(_s, _f):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    run_nonce = os.urandom(32)
    ctx_mp = mp.get_context("spawn") if sys.platform == "darwin" else mp.get_context("fork")
    ep_q = ctx_mp.Queue(maxsize=max(10, a.workers * 3))
    stop_ev = ctx_mp.Event()
    procs = [
        ctx_mp.Process(
            target=_x3b_worker_loop,
            args=(
                w,
                run_nonce,
                str(a.mmap_bank),
                str(a.bank_sha256),
                int(a.max_episode_rows),
                int(a.tokens),
                int(a.row_tokens),
                int(a.attn_cap),
                int(a.bmax),
                float(a.cpu_hard_pct),
                float(a.eval_clean_prob),
                float(a.subspace_prob),
                float(a.exact_kd_prob),
                float(a.exact_resid_prob),
                float(a.cf_weight_scale),
                float(a.p_ctx_min),
                float(a.p_ctx_max),
                float(a.postnorm_affine_max),
                str(a.teacher_shards_dir),
                str(a.hwm_json),
                str(a.gp_winners_jsonl),
                str(a.exclude_cids_json),
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
        BASE._atomic_save(payload, eval_latest_p, mode=0o644)
        return sha256_file(eval_latest_p)

    def save_last() -> None:
        BASE._atomic_save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                "ema_swa": {k: v.cpu() for k, v in ema_swa.state_dict().items()},
                "ema_valley": {k: v.cpu() for k, v in ema_valley.state_dict().items()},
                "opt": opt.state_dict(),
                "state": state,
                "ledger": ledger.state(),
                "size": a.size,
                "bank_sha256": bank.sha256,
                "init_sha256": a.init_sha256,
                "sampler_cfg": sampler_cfg,
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
                "event": "x3b_start",
                "run_id": str(a.run_id),
                "start": time.strftime("%FT%TZ", time.gmtime()),
                "host": platform.node(),
                "device": str(dev),
                "resume": bool(a.resume),
                "params": params_cnt,
                "bank_tables_total": bank.n_tables,
                "s8_clean_tables": fam_sampler.n_clean_tables,
                "s8_clean_families": fam_sampler.n_clean_families,
                "s8_masked_tables": int(n_s8_masked),
                "bank_load_s": bank_load_s,
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

            if cyc_idx > last_cyc_recorded and steps_since_start >= 10:
                last_cyc_recorded = cyc_idx
                state["valleys_saved"] = cyc_idx
                payload = _build_eval_payload()
                vp, v_sha = save_valley_ckpt_bounded(
                    out_dir,
                    payload,
                    cyc_idx,
                    keep_latest_only=bool(a.keep_latest_only),
                    ring_slots=int(a.valley_ring_slots),
                )
                BASE._atomic_save(payload, eval_latest_p, mode=0o644)
                log_f.write(
                    json.dumps(
                        {
                            "event": "valley_saved",
                            "valley_idx": cyc_idx,
                            "ckpt_path": str(vp),
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
                jp = out_dir / "epoch_latest.json" if bool(a.keep_latest_only) else (out_dir / f"epoch_{k_ep:03d}.json")
                jp.write_text(
                    json.dumps(
                        {
                            "schema": "jope.h458.x3b.epoch/v1",
                            "run_id": str(a.run_id),
                            "epoch": k_ep,
                            "train_hours": state["train_s"] / 3600.0,
                            "step": state["step"],
                            "ckpt": str(eval_latest_p),
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

            # GPU-side OOM trim-and-retry guard (`cl_h458` 22:09Z lesson): halve batch on OOM instead of crashing
            step_ok = False
            loss_val = 0.0
            while zm_np.shape[0] >= 1:
                B_cur, N_cur, Fn_cur, _ = zm_np.shape
                if dev.type == "cuda":
                    zm_t = torch.from_numpy(zm_np).pin_memory().to(dev, non_blocking=True)
                    yz_t = torch.from_numpy(yz_np).pin_memory().to(dev, non_blocking=True)
                    ym_t = torch.from_numpy(ym_np).pin_memory().to(dev, non_blocking=True)
                    ys_t = torch.from_numpy(ys_np).pin_memory().to(dev, non_blocking=True)
                    yq_t = torch.from_numpy(yq_np).pin_memory().to(dev, non_blocking=True)
                    wq_t = torch.from_numpy(wq_np).pin_memory().to(dev, non_blocking=True)
                else:
                    zm_t = torch.from_numpy(zm_np).to(dev)
                    yz_t = torch.from_numpy(yz_np).to(dev)
                    ym_t = torch.from_numpy(ym_np).to(dev)
                    ys_t = torch.from_numpy(ys_np).to(dev)
                    yq_t = torch.from_numpy(yq_np).to(dev)
                    wq_t = torch.from_numpy(wq_np).to(dev)

                # 100% random feature-ID slots (`cl_h458` 22:33Z K=5 unseen-stratum proof + cols_h450_gpu alignment)
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
                        break
                    (loss / float(grad_accum)).backward()
                    loss_val = float(loss.detach().cpu())
                    step_ok = True
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        if B_cur >= 2:
                            half_b = max(1, B_cur // 2)
                            zm_np, yz_np, ym_np, ys_np, yq_np, wq_np = (
                                zm_np[:half_b],
                                yz_np[:half_b],
                                ym_np[:half_b],
                                ys_np[:half_b],
                                yq_np[:half_b],
                                wq_np[:half_b],
                            )
                            if infos is not None:
                                infos = infos[:half_b]
                            state["oom_trims"] = int(state.get("oom_trims", 0)) + 1
                            continue
                        state["oom_skips"] += 1
                        opt.zero_grad(set_to_none=True)
                        state["accum_in_group"] = 0
                        break
                    raise

            if not step_ok:
                continue

            B_cur = int(zm_np.shape[0])
            if infos is not None:
                state["cell_tokens"] += ledger.add_real(infos)
            if batch_kind == "eval_clean":
                state["eval_clean_batches"] = int(state.get("eval_clean_batches", 0)) + 1
            elif batch_kind == "subspace_clean":
                state["subspace_batches"] = int(state.get("subspace_batches", 0)) + 1
            elif batch_kind == "exact_kd":
                state["exact_kd_batches"] += 1
                state["exact_kd_datasets"] += B_cur
            elif batch_kind == "exact_resid":
                state["exact_resid_batches"] += 1
                state["exact_resid_datasets"] += B_cur

            state["micro_batches"] += 1
            state["accum_in_group"] += 1
            state["datasets"] += B_cur
            run_loss += loss_val
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
                    val_alpha = 0.0025 if cur_lr <= float(a.lr_min) * 1.8 else 0.0008
                    val_d = (
                        max(0.90, 1.0 - val_alpha)
                        if steps_since_start > 20
                        else ((1.0 + steps_since_start) / (10.0 + steps_since_start))
                    )
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
            if now - t_log >= 25.0 and run_n > 0:
                host_cpu_pct = gov.sample_pct()
                s = ledger.summary()
                dt = max(1e-3, now - t_log)
                cuda_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0.0
                cuda_res_gb = round(torch.cuda.max_memory_reserved() / 2**30, 2) if dev.type == "cuda" else 0.0
                log_f.write(
                    json.dumps(
                        {
                            "run_id": str(a.run_id),
                            "step": state["step"],
                            "train_h": round(state["train_s"] / 3600.0, 4),
                            "cycle": cyc_idx,
                            "cycle_u": round(cyc_u, 4),
                            "loss": round(run_loss / run_n, 5),
                            "lr": round(cur_lr, 9),
                            "datasets": state["datasets"],
                            "ds_per_s": round(run_ds / dt, 3),
                            "mb_per_s": round(run_n / dt, 3),
                            "eval_clean_batches": state.get("eval_clean_batches", 0),
                            "subspace_batches": state.get("subspace_batches", 0),
                            "exact_kd_batches": state["exact_kd_batches"],
                            "exact_resid_batches": state["exact_resid_batches"],
                            "unique_datasets": s["unique_datasets_seen"],
                            "dup_fed": s["duplicate_datasets_fed_to_model"],
                            "src_tables_seen": s["distinct_source_tables_seen"],
                            "d91_share": s["dev91_share_of_real_episodes"],
                            "host_cpu_pct": round(host_cpu_pct, 1),
                            "oom": state["oom_skips"],
                            "oom_trims": state.get("oom_trims", 0),
                            "mem_alloc_gb": cuda_gb,
                            "mem_res_gb": cuda_res_gb,
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
                "run_id": str(a.run_id),
                "step": state["step"],
                "datasets": state["datasets"],
                "train_s": round(state["train_s"], 2),
                "exposure": ledger.summary(),
            }
        )
        + "\n"
    )
    log_f.close()


if __name__ == "__main__":
    main()
