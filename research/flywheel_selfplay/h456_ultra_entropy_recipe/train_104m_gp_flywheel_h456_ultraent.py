#!/usr/bin/env python3
"""Wave H456 Ultra-Entropy `104M` Neural-GP Flywheel Trainer (`train_104m_gp_flywheel_h456_ultraent.py`).

Advances over H455 (`train_104m_gp_flywheel_h455_maxent.py`):
  1. Exact `1 / 2,389` uniform S3 source mixing (`s3_source_exact_uniform` shuffled 2,389-family deck,
     `max/min` family mass == `1.000000x`, `regret_tilt_prob = 0.0`) + exact `1 / N_shard_families`
     hierarchical Exact-GP teacher shard sampling.
  2. Continuous multi-regime varied row sampling across `N_use in [32..2048]` (`60%` log-uniform +
     `40%` uniform, 4 fresh full-table row regimes, `p_ctx in [0.05..0.96]`, OOM-exact batch cap).
  3. Mandatory simultaneous feature removal (`drop in [0.05..0.90]`, `drop <= 0.75` on teacher KD/resid)
     + reserved `[10%..70%]` synthetic/distractor injection EVEN on wide tables (`p_raw >= F_use`) across
     28 zero-leakage context-only (`Xc, yc`) families (`F_use in [4..80]`).
  4. Unbiased column subsetting, 3-mode `FID_SLOTS` permutation, `50%` sign flips, `exp(U(-4, 4))`
     scale, signed-power `[0.35, 2.5]`, `K`-level quantization, Cauchy spikes, correlated noise,
     wider post-prenorm affine (`exp(+-0.45)`), and monotone target warps (`0.0` query leakage).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

HERE = Path(__file__).resolve().parent
for _cand in (
    str(HERE),
    os.environ.get("JOPE_REF_CODE", ""),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

from mmap_bank_h456 import MmapF64BankH456, SEALED_TOKENS, compute_file_sha256  # noqa: E402
from ultra_entropy_ops_h456 import (  # noqa: E402
    F_MAX_DEFAULT_H456,
    apply_permute_scale_noise_warps_h456,
    apply_postnorm_affine_h456,
    remove_and_inject_features_h456,
    sample_episode_shape_h456,
    sample_fid_slots_h456,
    sample_rows_multiregime_h456,
    warp_target_monotone_h456,
)


def _load_icl_backbone():
    from icl_model import FID_SLOTS  # noqa: WPS433
    from icl_model_h450 import ICLH450  # noqa: WPS433

    return FID_SLOTS, ICLH450


def raised_cosine_valley_lr(
    elapsed_h: float,
    lr_init_peak: float,
    lr_cycle_peak: float,
    lr_min: float,
    cycle0_h: float = 2.5,
    cycle_h: float = 2.5,
) -> tuple[float, int, float, bool]:
    """C1-continuous cyclic raised-cosine schedule returning `(lr, cycle_idx, u_phase, in_valley)`."""
    t_h = max(0.0, float(elapsed_h))
    c0 = max(float(cycle0_h), 1e-6)
    ck = max(float(cycle_h), 1e-6)
    if t_h < c0:
        u = t_h / c0
        lr = float(lr_min) + 0.5 * (float(lr_init_peak) - float(lr_min)) * (1.0 + math.cos(math.pi * u))
        return lr, 0, u, (u >= 0.80)
    rem = t_h - c0
    cyc = 1 + int(math.floor(rem / ck))
    u = (rem - float(cyc - 1) * ck) / ck
    u = min(max(u, 0.0), 1.0)
    lr = float(lr_min) + 0.5 * (float(lr_cycle_peak) - float(lr_min)) * (1.0 - math.cos(2.0 * math.pi * u))
    in_val = (u >= 0.88) or (u <= 0.08)
    return lr, cyc, u, in_val


@dataclass
class ShardEntryH456:
    path: Path
    cid: str
    family_id: str
    n_rows: int
    n_cols: int
    is_d91: bool


class ExactGPTeacherShardStoreH456:
    """Reloadable Exact-GP teacher shard store with exact `1 / N_shard_families` hierarchical draws."""

    def __init__(
        self,
        shards_dir: Path | None,
        allow_d91: bool = True,
        reload_every_s: float = 45.0,
        max_shards: int = 0,
    ) -> None:
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.allow_d91 = bool(allow_d91)
        self.reload_every_s = float(reload_every_s)
        self.max_shards = int(max_shards)
        self.last_load_t = 0.0
        self.known_names: set[str] = set()
        self.entries: list[ShardEntryH456] = []
        self.family_buckets: dict[str, list[int]] = {}
        self.family_keys: list[str] = []
        self.maybe_reload(force=True)

    def _inspect_one_shard(self, path_obj: Path) -> ShardEntryH456 | None:
        try:
            with np.load(path_obj, allow_pickle=False) as zf:
                files = set(zf.files)
                cid = str(zf["cid"].item()) if "cid" in files else path_obj.stem
                coh = str(zf["cohort"].item()) if "cohort" in files else ""
                fam = str(zf["family_id"].item()) if "family_id" in files else cid.split("__")[0]
                tag_u = f"{cid}|{coh}|{fam}".upper()
                if any(tok in tag_u for tok in SEALED_TOKENS):
                    return None
                is_d91 = bool(zf["is_dev91_table"].item()) if "is_dev91_table" in files else coh.upper().startswith("D91")
                if is_d91 and not self.allow_d91:
                    return None
                if {"Xc", "yc", "oof_gp_c", "Xq_real", "yq_real", "pred_gp_q_real"}.issubset(files):
                    c_sh = tuple(int(v) for v in zf["Xc"].shape)
                    q_sh = tuple(int(v) for v in zf["Xq_real"].shape)
                    if len(c_sh) != 2 or len(q_sh) != 2 or c_sh[1] != q_sh[1] or (c_sh[0] + q_sh[0]) < 32:
                        return None
                    return ShardEntryH456(path_obj, cid, fam, c_sh[0] + q_sh[0], c_sh[1], is_d91)
                if {"X_s", "y_s", "gp_oof_mu"}.issubset(files):
                    xs_sh = tuple(int(v) for v in zf["X_s"].shape)
                    if len(xs_sh) != 2 or xs_sh[0] < 32 or xs_sh[1] < 1:
                        return None
                    return ShardEntryH456(path_obj, cid, fam, xs_sh[0], xs_sh[1], is_d91)
                return None
        except (OSError, ValueError, KeyError):
            return None

    def maybe_reload(self, force: bool = False) -> int:
        now = time.monotonic()
        if not force and (now - self.last_load_t) < self.reload_every_s:
            return len(self.entries)
        self.last_load_t = now
        if self.shards_dir is None or not self.shards_dir.is_dir():
            return len(self.entries)
        added = 0
        for fp in sorted(self.shards_dir.glob("*.npz")):
            if self.max_shards > 0 and len(self.entries) >= self.max_shards:
                break
            if fp.name.startswith(".") or fp.name in self.known_names:
                continue
            self.known_names.add(fp.name)
            ent = self._inspect_one_shard(fp)
            if ent is None:
                continue
            idx = len(self.entries)
            self.entries.append(ent)
            self.family_buckets.setdefault(ent.family_id, []).append(idx)
            added += 1
        if added > 0:
            self.family_keys = sorted(self.family_buckets.keys())
        return len(self.entries)

    def family_uniform_mass_ratio(self) -> float:
        if not self.family_keys:
            return 1.0
        # Every family key has exact `1 / len(self.family_keys)` selection mass
        return 1.0

    def sample_shard_entry(self, rng: np.random.Generator, min_rows: int = 32) -> ShardEntryH456 | None:
        if not self.family_keys:
            return None
        elig_fams = [
            fk
            for fk in self.family_keys
            if any(self.entries[idx].n_rows >= int(min_rows) for idx in self.family_buckets[fk])
        ]
        if not elig_fams:
            return None
        fk = elig_fams[int(rng.integers(0, len(elig_fams)))]
        cand = [idx for idx in self.family_buckets[fk] if self.entries[idx].n_rows >= int(min_rows)]
        chosen_i = cand[int(rng.integers(0, len(cand)))]
        return self.entries[chosen_i]


def _load_shard_arrays(ent: ShardEntryH456) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        with np.load(ent.path, allow_pickle=False) as zf:
            files = set(zf.files)
            if {"Xc", "yc", "oof_gp_c", "Xq_real", "yq_real", "pred_gp_q_real"}.issubset(files):
                xs = np.vstack([np.asarray(zf["Xc"], dtype=np.float64), np.asarray(zf["Xq_real"], dtype=np.float64)])
                ys = np.concatenate([np.asarray(zf["yc"], dtype=np.float64), np.asarray(zf["yq_real"], dtype=np.float64)])
                mu = np.concatenate([np.asarray(zf["oof_gp_c"], dtype=np.float64), np.asarray(zf["pred_gp_q_real"], dtype=np.float64)])
                res_sd = float(np.std(ys - mu))
                sd = np.full_like(mu, max(res_sd, 1e-3))
                return xs, ys, mu, sd
            xs = np.asarray(zf["X_s"], dtype=np.float64)
            ys = np.asarray(zf["y_s"], dtype=np.float64)
            mu = np.asarray(zf["gp_oof_mu"], dtype=np.float64)
            sd = np.asarray(zf["gp_oof_std"], dtype=np.float64) if "gp_oof_std" in files else np.zeros_like(mu)
            if xs.shape[0] != ys.shape[0] or ys.shape[0] != mu.shape[0]:
                return None
            return xs, ys, mu, np.maximum(np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    except (OSError, ValueError, KeyError):
        return None


def build_teacher_microbatch_h456(
    shard_store: ExactGPTeacherShardStoreH456,
    mode: str,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
) -> dict | None:
    """Builds cross-shard shape-matched `exact_kd` or `exact_resid` microbatch with cluttered student views."""
    bsz, n_use, f_use, n_ctx = sample_episode_shape_h456(
        rng,
        max_rows=min(768, int(cfg.max_episode_rows)),
        max_cols=int(cfg.max_cols),
        tokens_cap=int(cfg.tokens),
        row_tokens_cap=int(cfg.row_tokens),
        attn_cap=int(cfg.attn_cap),
        bmax=int(cfg.bmax),
        p_ctx_min=float(cfg.p_ctx_min),
        p_ctx_max=float(cfg.p_ctx_max),
        is_teacher=True,
    )
    xs_list: list[np.ndarray] = []
    yc_list: list[np.ndarray] = []
    yq_true_list: list[np.ndarray] = []
    yq_mu_list: list[np.ndarray] = []
    yq_sd_list: list[np.ndarray] = []

    drop_hi = float(cfg.teacher_drop_max_frac) if mode == "exact_kd" else min(0.70, float(cfg.teacher_drop_max_frac))
    attempts = 0
    while len(xs_list) < bsz and attempts < bsz * 4:
        attempts += 1
        ent = shard_store.sample_shard_entry(rng, min_rows=max(32, n_use // 2))
        if ent is None:
            return None
        loaded = _load_shard_arrays(ent)
        if loaded is None:
            continue
        x_raw, y_raw, mu_raw, sd_raw = loaded
        # Pack `[X, mu, sd, y]` so row sampling keeps exact row alignment
        packed = np.column_stack([x_raw, mu_raw, sd_raw, y_raw])
        sub_pack, _ = sample_rows_multiregime_h456(packed, n_use, rng)
        p_orig = x_raw.shape[1]
        x_ep = sub_pack[:, :p_orig]
        mu_ep = np.nan_to_num(sub_pack[:, p_orig], nan=0.0, posinf=0.0, neginf=0.0)
        sd_ep = np.maximum(np.nan_to_num(sub_pack[:, p_orig + 1], nan=0.0, posinf=0.0, neginf=0.0), 1e-4)
        y_ep = np.nan_to_num(sub_pack[:, p_orig + 2], nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(y_ep[:n_ctx])) < 1e-9 or not np.all(np.isfinite(mu_ep)):
            continue

        y_ctx_ref = (y_ep[:n_ctx] - mu_ep[:n_ctx]) if mode == "exact_resid" else y_ep[:n_ctx]
        x_clut, _ = remove_and_inject_features_h456(
            x_ep,
            y_ctx_ref,
            n_ctx=n_ctx,
            f_target=f_use,
            rng=rng,
            drop_min_frac=float(cfg.feature_drop_min_frac),
            drop_max_frac=drop_hi,
            inj_min_frac=max(0.15, float(cfg.synth_inj_min_frac)),
            inj_max_frac=min(0.65, float(cfg.synth_inj_max_frac)),
            distractor_prob=float(cfg.distractor_prob),
            anchor_full_prob=0.15,
            zero_inj_prob=0.10,
        )
        x_warped = apply_permute_scale_noise_warps_h456(
            x_clut,
            n_ctx=n_ctx,
            rng=rng,
            noise_max=float(cfg.feature_noise_max),
            scale_log_max=float(cfg.scale_log_max),
        )
        if not np.all(np.isfinite(x_warped)):
            continue

        if mode == "exact_resid":
            res_ep = y_ep - mu_ep
            if float(np.std(res_ep[:n_ctx])) < 1e-9:
                continue
            yc_list.append(res_ep[:n_ctx])
            yq_true_list.append(res_ep[n_ctx:])
            yq_mu_list.append(np.zeros(n_use - n_ctx, dtype=np.float64))
            yq_sd_list.append(sd_ep[n_ctx:])
        else:
            yc_list.append(y_ep[:n_ctx])
            yq_true_list.append(y_ep[n_ctx:])
            yq_mu_list.append(mu_ep[n_ctx:])
            yq_sd_list.append(sd_ep[n_ctx:])
        xs_list.append(x_warped)

    if not xs_list:
        return None
    return {
        "stream": mode,
        "x": np.stack(xs_list, axis=0),
        "yc": np.stack(yc_list, axis=0),
        "yq": np.stack(yq_true_list, axis=0),
        "teacher_mu": np.stack(yq_mu_list, axis=0),
        "teacher_sd": np.stack(yq_sd_list, axis=0),
        "n_ctx": n_ctx,
        "seed": int(rng.integers(1, 1 << 30)),
    }


def build_synth_prior_microbatch_h456(rng: np.random.Generator, cfg: argparse.Namespace) -> dict:
    """Generates a structural causal / GP-RFF synthetic tabular prior microbatch with H456 clutter."""
    bsz, n_use, f_use, n_ctx = sample_episode_shape_h456(
        rng,
        max_rows=min(1024, int(cfg.max_episode_rows)),
        max_cols=int(cfg.max_cols),
        tokens_cap=int(cfg.tokens),
        row_tokens_cap=int(cfg.row_tokens),
        attn_cap=int(cfg.attn_cap),
        bmax=int(cfg.bmax),
        p_ctx_min=float(cfg.p_ctx_min),
        p_ctx_max=float(cfg.p_ctx_max),
        is_teacher=False,
    )
    xs_list: list[np.ndarray] = []
    yc_list: list[np.ndarray] = []
    yq_list: list[np.ndarray] = []
    for _ in range(bsz):
        p_lat = int(rng.integers(3, 18))
        cov_f = rng.standard_normal((p_lat, p_lat)) / math.sqrt(float(p_lat))
        z_lat = rng.standard_normal((n_use, p_lat)) @ cov_f
        # Random RFF + tree-threshold causal target
        n_rff = 24
        om = rng.normal(0.0, float(rng.uniform(0.4, 1.8)), size=(p_lat, n_rff))
        ph = rng.uniform(0.0, 2.0 * math.pi, size=(1, n_rff))
        phi = np.cos(z_lat @ om + ph)
        y_clean = phi @ rng.normal(0.0, 1.0 / math.sqrt(float(n_rff)), size=n_rff)
        y_clean += 0.35 * ((z_lat[:, 0] > 0.0).astype(np.float64) * z_lat[:, min(1, p_lat - 1)])
        y_raw = y_clean + float(rng.uniform(0.02, 0.35)) * rng.standard_normal(n_use)
        y_warped = warp_target_monotone_h456(y_raw, n_ctx=n_ctx, rng=rng, warp_prob=float(cfg.rand_target_prob))
        x_clut, _ = remove_and_inject_features_h456(
            z_lat,
            y_warped[:n_ctx],
            n_ctx=n_ctx,
            f_target=f_use,
            rng=rng,
            drop_min_frac=float(cfg.feature_drop_min_frac),
            drop_max_frac=float(cfg.feature_drop_max_frac),
            inj_min_frac=float(cfg.synth_inj_min_frac),
            inj_max_frac=float(cfg.synth_inj_max_frac),
            distractor_prob=float(cfg.distractor_prob),
        )
        x_out = apply_permute_scale_noise_warps_h456(
            x_clut,
            n_ctx=n_ctx,
            rng=rng,
            noise_max=float(cfg.feature_noise_max),
            scale_log_max=float(cfg.scale_log_max),
        )
        xs_list.append(x_out)
        yc_list.append(y_warped[:n_ctx])
        yq_list.append(y_warped[n_ctx:])

    return {
        "stream": "synth_prior",
        "x": np.stack(xs_list, axis=0),
        "yc": np.stack(yc_list, axis=0),
        "yq": np.stack(yq_list, axis=0),
        "teacher_mu": None,
        "teacher_sd": None,
        "n_ctx": n_ctx,
        "seed": int(rng.integers(1, 1 << 30)),
    }


def build_s3_ultraent_microbatch_h456(
    bank: MmapF64BankH456,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
) -> dict | None:
    """Builds a real-corpus `s3_source_exact_uniform` microbatch across all `2,389` S3 source families."""
    bsz, n_use, f_use, n_ctx = sample_episode_shape_h456(
        rng,
        max_rows=int(cfg.max_episode_rows),
        max_cols=int(cfg.max_cols),
        tokens_cap=int(cfg.tokens),
        row_tokens_cap=int(cfg.row_tokens),
        attn_cap=int(cfg.attn_cap),
        bmax=int(cfg.bmax),
        p_ctx_min=float(cfg.p_ctx_min),
        p_ctx_max=float(cfg.p_ctx_max),
        is_teacher=False,
    )
    xs_list: list[np.ndarray] = []
    yc_list: list[np.ndarray] = []
    yq_list: list[np.ndarray] = []
    regimes: list[int] = []

    attempts = 0
    while len(xs_list) < bsz and attempts < bsz * 5:
        attempts += 1
        ti = bank.sample_eligible_min_rows(rng, min_rows=max(32, n_use // 2))
        if ti is None:
            ti = bank.sample_table_idx(rng)
        tab = bank.tabs_f64[ti]
        sub_tab, reg_id = sample_rows_multiregime_h456(tab, n_use, rng)
        x_ep = sub_tab[:, :-1]
        y_ep = sub_tab[:, -1]
        fin_y = np.isfinite(y_ep)
        if not np.all(fin_y):
            if int(np.sum(fin_y[:n_ctx])) < 8:
                continue
            med_y = float(np.median(y_ep[:n_ctx][fin_y[:n_ctx]]))
            y_ep = np.where(fin_y, y_ep, med_y)
        if float(np.std(y_ep[:n_ctx])) < 1e-9:
            continue

        y_warped = warp_target_monotone_h456(y_ep, n_ctx=n_ctx, rng=rng, warp_prob=float(cfg.rand_target_prob))
        if float(np.std(y_warped[:n_ctx])) < 1e-9:
            continue

        x_clut, _ = remove_and_inject_features_h456(
            x_ep,
            y_warped[:n_ctx],
            n_ctx=n_ctx,
            f_target=f_use,
            rng=rng,
            drop_min_frac=float(cfg.feature_drop_min_frac),
            drop_max_frac=float(cfg.feature_drop_max_frac),
            inj_min_frac=float(cfg.synth_inj_min_frac),
            inj_max_frac=float(cfg.synth_inj_max_frac),
            distractor_prob=float(cfg.distractor_prob),
        )
        x_out = apply_permute_scale_noise_warps_h456(
            x_clut,
            n_ctx=n_ctx,
            rng=rng,
            noise_max=float(cfg.feature_noise_max),
            scale_log_max=float(cfg.scale_log_max),
        )
        if not np.all(np.isfinite(x_out)):
            continue
        xs_list.append(x_out)
        yc_list.append(y_warped[:n_ctx])
        yq_list.append(y_warped[n_ctx:])
        regimes.append(reg_id)

    if not xs_list:
        return None
    return {
        "stream": "s3_exact_uniform",
        "x": np.stack(xs_list, axis=0),
        "yc": np.stack(yc_list, axis=0),
        "yq": np.stack(yq_list, axis=0),
        "teacher_mu": None,
        "teacher_sd": None,
        "n_ctx": n_ctx,
        "regimes": regimes,
        "seed": int(rng.integers(1, 1 << 30)),
    }


def sample_any_microbatch_h456(
    bank: MmapF64BankH456,
    shard_store: ExactGPTeacherShardStoreH456,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
) -> dict:
    """Routes across `exact_kd`, `exact_resid`, `synth_prior`, and `s3_exact_uniform`."""
    u = float(rng.random())
    p_kd = float(cfg.exact_kd_prob)
    p_res = float(cfg.exact_resid_prob)
    p_syn = float(cfg.synth_prob)
    if shard_store.entries:
        if u < p_kd:
            mb = build_teacher_microbatch_h456(shard_store, "exact_kd", rng, cfg)
            if mb is not None:
                return mb
        elif u < p_kd + p_res:
            mb = build_teacher_microbatch_h456(shard_store, "exact_resid", rng, cfg)
            if mb is not None:
                return mb
    if u < p_kd + p_res + p_syn:
        return build_synth_prior_microbatch_h456(rng, cfg)
    mb_s3 = build_s3_ultraent_microbatch_h456(bank, rng, cfg)
    if mb_s3 is not None:
        return mb_s3
    return build_synth_prior_microbatch_h456(rng, cfg)


def install_postnorm_and_ckpt_patch_h456(
    model: nn.Module,
    affine_max: float = 0.45,
    checkpoint_rows: bool = False,
) -> None:
    """Installs H456 post-normalise channel affine jitter (`exp(+-0.45)`) + optional row-block checkpointing."""
    orig_norm = model.normalise_f64
    state_holder = {"seed": 0}
    model._h456_state = state_holder  # type: ignore[attr-defined]

    def _wrapped_normalise(x_t, y_ctx_t, n_ctx_i):
        zm, ym64, ys64 = orig_norm(x_t, y_ctx_t, n_ctx_i)
        if model.training and affine_max > 0.0:
            zm = apply_postnorm_affine_h456(zm, seed=int(state_holder["seed"]), affine_max=affine_max)
        return zm, ym64, ys64

    model.normalise_f64 = _wrapped_normalise  # type: ignore[method-assign]
    if checkpoint_rows and hasattr(model, "row_blocks"):
        for blk in model.row_blocks:
            orig_f = blk.forward

            def _make_wrapped(fn):
                def _wf(*args, **kwargs):
                    if model.training:
                        return grad_ckpt(fn, *args, use_reentrant=False, **kwargs)
                    return fn(*args, **kwargs)

                return _wf

            blk.forward = _make_wrapped(orig_f)


def compute_bar_distribution_loss_h456(
    model: nn.Module,
    mb: dict,
    fid_slots: int,
    dev: torch.device,
    kd_alpha: float = 0.55,
) -> torch.Tensor:
    """Computes `ICLH450` bar-distribution NLL + optional Gaussian-projected Exact-GP KD soft loss."""
    xt = torch.from_numpy(mb["x"]).to(device=dev, dtype=torch.float32)
    yc = torch.from_numpy(mb["yc"]).to(device=dev, dtype=torch.float32)
    yq = torch.from_numpy(mb["yq"]).to(device=dev, dtype=torch.float32)
    n_ctx = int(mb["n_ctx"])
    f_use = int(xt.shape[2])
    if hasattr(model, "_h456_state"):
        model._h456_state["seed"] = int(mb["seed"]) + 17

    slots = sample_fid_slots_h456(f_use, fid_slots, int(mb["seed"]), dev)
    logits, ym, ys = model(xt, yc, n_ctx, slots=slots, amp=(dev.type == "cuda"))
    nll_hard = model.loss(logits, yq, ym, ys).mean()

    if mb["stream"] == "exact_kd" and mb["teacher_mu"] is not None:
        centers = model.centers.to(device=dev, dtype=torch.float32)
        ys_safe = torch.clamp(ys.float(), min=1e-6)
        ym_f = ym.float()
        t_mu = torch.from_numpy(mb["teacher_mu"]).to(device=dev, dtype=torch.float32)
        t_sd = torch.from_numpy(mb["teacher_sd"]).to(device=dev, dtype=torch.float32)
        z_mu = torch.clamp((t_mu - ym_f) / ys_safe, float(centers[0].item()), float(centers[-1].item())).unsqueeze(-1)
        z_sd = torch.clamp(t_sd / ys_safe, min=0.06, max=2.50).unsqueeze(-1)
        q_logits = -0.5 * ((centers.view(1, 1, -1) - z_mu) / z_sd) ** 2
        q_prob = F.softmax(q_logits, dim=-1)
        log_bin_p = F.log_softmax(logits.float(), dim=-1)
        kd_ce = -(q_prob * log_bin_p).sum(dim=-1).mean()
        return (1.0 - float(kd_alpha)) * nll_hard + float(kd_alpha) * kd_ce

    return nll_hard


@torch.no_grad()
def _ema_update(ema_sd: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    d = float(decay)
    for k, v in model.state_dict().items():
        if k not in ema_sd:
            ema_sd[k] = v.detach().float().cpu().clone()
        elif v.is_floating_point():
            ema_sd[k].mul_(d).add_(v.detach().float().cpu(), alpha=1.0 - d)
        else:
            ema_sd[k].copy_(v.detach().cpu())


def save_checkpoint_atomic_h456(path_obj: Path, payload: dict) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.parent / f".{path_obj.name}.{os.getpid()}.tmp"
    torch.save(payload, tmp_p)
    os.replace(tmp_p, path_obj)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="L", choices=("S", "M", "L"))
    ap.add_argument("--init", default="")
    ap.add_argument("--init-sha256", default="")
    ap.add_argument("--random-init", type=int, default=0)
    ap.add_argument("--mmap-bank", default="/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
    ap.add_argument("--teacher-shards-dir", default="/home/ubuntu/jope-data/h452-flywheel/gp_teacher_shards_v1")
    ap.add_argument("--allow-d91-teacher-shards", type=int, default=1)
    ap.add_argument("--dataset-mix", default="s3_source_exact_uniform")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=96.0)
    ap.add_argument("--endless", type=int, default=0)
    ap.add_argument("--smoke-cpu-steps", type=int, default=0, help="Run N deterministic steps and exit (self-test)")
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--lr-cycle-peak", type=float, default=7.5e-6)
    ap.add_argument("--lr-min", type=float, default=2.5e-6)
    ap.add_argument("--cycle0-hours", type=float, default=2.5)
    ap.add_argument("--cycle-hours", type=float, default=2.5)
    ap.add_argument("--exact-kd-prob", type=float, default=0.28)
    ap.add_argument("--exact-resid-prob", type=float, default=0.16)
    ap.add_argument("--synth-prob", type=float, default=0.14)
    ap.add_argument("--feature-drop-min-frac", type=float, default=0.05)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.90)
    ap.add_argument("--teacher-drop-max-frac", type=float, default=0.75)
    ap.add_argument("--synth-inj-min-frac", type=float, default=0.10)
    ap.add_argument("--synth-inj-max-frac", type=float, default=0.70)
    ap.add_argument("--distractor-prob", type=float, default=0.35)
    ap.add_argument("--feature-noise-max", type=float, default=0.12)
    ap.add_argument("--scale-log-max", type=float, default=4.0)
    ap.add_argument("--postnorm-affine-max", type=float, default=0.45)
    ap.add_argument("--rand-target-prob", type=float, default=0.45)
    ap.add_argument("--p-ctx-min", type=float, default=0.05)
    ap.add_argument("--p-ctx-max", type=float, default=0.96)
    ap.add_argument("--max-episode-rows", type=int, default=2048)
    ap.add_argument("--max-cols", type=int, default=F_MAX_DEFAULT_H456)
    ap.add_argument("--tokens", type=int, default=96000)
    ap.add_argument("--row-tokens", type=int, default=6144)
    ap.add_argument("--attn-cap", type=int, default=2400000)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--checkpoint-row-blocks", type=int, default=1)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.0)
    ap.add_argument("--save-every-datasets", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=456101)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    return ap


def _init_model_and_emas(args: argparse.Namespace, dev: torch.device) -> tuple[nn.Module, int, dict[str, dict]]:
    fid_slots, icl_cls = _load_icl_backbone()
    model = icl_cls(args.size).to(dev)
    if not int(args.random_init):
        if not args.init:
            raise SystemExit("either --init or --random-init 1 is required")
        init_p = Path(args.init)
        if args.init_sha256:
            got_sha = compute_file_sha256(init_p)
            if got_sha != args.init_sha256:
                raise SystemExit(f"--init sha256 {got_sha} != expected {args.init_sha256}")
        ck = torch.load(init_p, map_location="cpu", weights_only=False)
        sd = ck.get("ema_cooldown") or ck.get("ema_valley") or ck.get("ema_swa") or ck.get("ema") or ck.get("model") or ck
        model.load_state_dict(sd, strict=True)
    install_postnorm_and_ckpt_patch_h456(
        model,
        affine_max=float(args.postnorm_affine_max),
        checkpoint_rows=bool(args.checkpoint_row_blocks),
    )
    base_cpu = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
    emas = {
        "ema": {k: v.clone() for k, v in base_cpu.items()},
        "ema_swa": {k: v.clone() for k, v in base_cpu.items()},
        "ema_cooldown": {k: v.clone() for k, v in base_cpu.items()},
        "ema_valley": {k: v.clone() for k, v in base_cpu.items()},
    }
    return model, fid_slots, emas


def run_training_loop_h456(args: argparse.Namespace) -> dict:
    dev = torch.device(args.device)
    if dev.type == "cuda" and float(args.cuda_vram_frac) > 0.0:
        torch.cuda.set_per_process_memory_fraction(min(0.95, max(0.05, float(args.cuda_vram_frac))))
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    bank = MmapF64BankH456(Path(args.mmap_bank), sampling_mode=args.dataset_mix)
    shard_store = ExactGPTeacherShardStoreH456(
        Path(args.teacher_shards_dir) if args.teacher_shards_dir else None,
        allow_d91=bool(args.allow_d91_teacher_shards),
    )
    model, fid_slots, emas = _init_model_and_emas(args, dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-2, betas=(0.9, 0.98))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": 0,
        "datasets": 0,
        "train_s": 0.0,
        "valleys_saved": 0,
        "oom_skips": 0,
        "nonfinite_skips": 0,
        "last_cycle_saved": -1,
    }
    if args.resume and (out_dir / "ckpt_last.pt").is_file():
        ck_r = torch.load(out_dir / "ckpt_last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck_r["model"], strict=True)
        for ek in emas:
            if ck_r.get(ek) is not None:
                emas[ek] = ck_r[ek]
        state.update(ck_r.get("state", {}))

    t_start = time.monotonic()
    accum = max(1, int(args.grad_accum))
    smoke_limit = int(args.smoke_cpu_steps)
    last_loss = 0.0
    last_gnorm = 0.0

    while True:
        elapsed_h = (float(state["train_s"]) + (time.monotonic() - t_start)) / 3600.0
        if smoke_limit <= 0 and not int(args.endless) and elapsed_h >= float(args.hours):
            break
        if smoke_limit > 0 and int(state["step"]) >= smoke_limit:
            break

        lr_now, cyc_idx, u_cyc, in_valley = raised_cosine_valley_lr(
            elapsed_h,
            args.lr,
            args.lr_cycle_peak,
            args.lr_min,
            args.cycle0_hours,
            args.cycle_hours,
        )
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        valid_mb = 0
        for _ in range(accum):
            shard_store.maybe_reload(force=False)
            mb = sample_any_microbatch_h456(bank, shard_store, rng, args)
            try:
                loss = compute_bar_distribution_loss_h456(model, mb, fid_slots, dev)
                if not torch.isfinite(loss):
                    state["nonfinite_skips"] = int(state["nonfinite_skips"]) + 1
                    continue
                (loss / float(accum)).backward()
                step_loss_sum += float(loss.detach().item())
                state["datasets"] = int(state["datasets"]) + int(mb["x"].shape[0])
                valid_mb += 1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and dev.type == "cuda":
                    state["oom_skips"] = int(state["oom_skips"]) + 1
                    torch.cuda.empty_cache()
                    continue
                raise

        if valid_mb == 0:
            continue
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gnorm):
            state["nonfinite_skips"] = int(state["nonfinite_skips"]) + 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        state["step"] = int(state["step"]) + 1
        last_loss = step_loss_sum / float(valid_mb)
        last_gnorm = gnorm

        _ema_update(emas["ema"], model, 0.999)
        _ema_update(emas["ema_swa"], model, 0.9995)
        if in_valley or smoke_limit > 0:
            _ema_update(emas["ema_cooldown"], model, 0.996)
            _ema_update(emas["ema_valley"], model, 0.996)

        reached_valley_end = (cyc_idx > int(state["last_cycle_saved"]) and u_cyc >= 0.98) or (
            smoke_limit > 0 and int(state["step"]) == smoke_limit
        )
        if reached_valley_end:
            state["valleys_saved"] = int(state["valleys_saved"]) + 1
            state["last_cycle_saved"] = cyc_idx
            payload = {
                "size": args.size,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **emas,
                "state": dict(state),
                "sampler_cfg": vars(args),
            }
            v_idx = int(state["valleys_saved"])
            save_checkpoint_atomic_h456(out_dir / f"ckpt_valley_{v_idx:03d}.pt", payload)
            save_checkpoint_atomic_h456(out_dir / "ckpt_last.pt", payload)

        if int(state["step"]) % 25 == 0 or smoke_limit > 0:
            print(
                json.dumps(
                    {
                        "step": state["step"],
                        "datasets": state["datasets"],
                        "lr": round(lr_now, 8),
                        "cycle": cyc_idx,
                        "cycle_u": round(u_cyc, 4),
                        "loss": round(last_loss, 5),
                        "gnorm": round(last_gnorm, 5),
                        "oom_skips": state["oom_skips"],
                        "nonfinite_skips": state["nonfinite_skips"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return {
        "steps": int(state["step"]),
        "datasets": int(state["datasets"]),
        "valleys_saved": int(state["valleys_saved"]),
        "last_loss": last_loss,
        "last_gnorm": last_gnorm,
        "oom_skips": int(state["oom_skips"]),
        "nonfinite_skips": int(state["nonfinite_skips"]),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_training_loop_h456(args)
    print(json.dumps({"event": "h456_finished", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
