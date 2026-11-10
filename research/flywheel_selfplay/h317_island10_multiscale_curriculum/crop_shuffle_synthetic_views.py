#!/usr/bin/env python3
"""H317 B=32 4-regime dataset views per TRAIN context + frozen-DOPE re-encoding into the H58 context-feature schema.

Views (plan §3.1; deterministic, seeds from `common_h317.stream_seed("H317-view", RECIPE_VERSION, cid, v)`):
  Regime 1 (0-7)   row crops: v0 = the canonical table (100% rows, original order); v1-v7 = shuffled row order + crops
                   stratified by (native fold x target decile) at 45/55/65/75/80/85/90% of rows.
  Regime 2 (8-15)  column subset crops keeping 55/55/65/65/75/75/85/85% of columns (2 seeds per level) in a random column
                   order; all rows. v15 additionally drops the single highest-leverage column (max |Pearson corr(x_j, y)|
                   on standardized columns; the plan says "highest-variance/highest-leverage" and raw variance is not
                   comparable across differently scaled columns).
  Regime 3 (16-23) appended synthetic columns, dW_v = max(4, min(20, ceil(f_v * W))) with the per-view fraction the plan
                   states (f = .35 .30 .40 .35 .35 .35 .35 .60; the plan's generic formula is the f=.40 instance):
                   16 Gaussian+Uniform distractors, 17 Student-t3+Cauchy, 18 noisy collinear duplicates
                   a*z_a+(1-a)*z_b+0.15e, 19 bilinear z_a*z_b and quadratic z_a^2-1, 20 monotone warps
                   sign(x)ln(1+|x|) and Phi^-1(rank), 21 quantile steps I[x>q25], I[x>q75], 22 random Fourier sin/cos,
                   23 all-7 cocktail (f=.60). Rows untouched.
  Regime 4 (24-31) compound stress: row crop 65-75% (stratified, shuffled) + column crop 75-85% (random order) + 30%
                   pure distractors (Gaussian/Uniform/t3/Cauchy) + full column shuffle + 1.5-3.0% Huberized target noise
                   (sd(y) * psi_1.345(t3/sqrt(3))). The 8 (row, col, noise) settings are a deterministic grid.
Every view keeps both native folds (row crops are fold-stratified) and returns the folds of its rows.

Subsets (nested, A4 ⊂ B16 ⊂ C32):
  A4  = (0, 5, 12, 23)  ~ H257: full100, sub80 row crop, column jitter, semi-synthetic cocktail.
  B16 = 4 per regime: (0, 1, 3, 5), (8, 10, 12, 14), (16, 18, 20, 23), (24, 26, 28, 30).
  C32 = 0..31.

DOPE encoding reproduces `research/flywheel_selfplay/h58/prepare_cache.py::extract_single_table_features`
(origin/v1/next): frozen Stage-1A landmarks (`FusedLandmarkWorkspace`, C++), `FusedJopeNetwork` from the pinned
`cpu_exact_fused.py` on CPU, `_run_dope_mlc_v3_and_lm48(use_landmarks=True)` then `_project_stage1b_tokens_batched_gpu`.
Output schema per view: tokens_10x32, mask_10 (constant), dope629, lm48_z, ctx_feat_584 (= zeros with [88:200] =
dope629[:112], [536:584] = lm48_z). The DOPE composite sha256 is checked before and after every encoding process.

CLI (GPU hosts, CPU only, one process per host, <= 4 numeric threads):
  encode   --shard k --nshards n     per-context npz to $H317_DATA/views/<cid>.npz (resumable)
  encode   --kind search --pilot-n 96   SEARCH-row view sets of the GP pilot contexts to $H317_DATA/views-search/<cid>.npz
  regress  view-0 regression gate against the staged H58 TRAIN-prefix extract
  merge    one merged float16 file neural/views_c32_train1219.npz (+ float32 view 0) for the trainers
"""

from __future__ import annotations

import os

# Numeric thread cap (coordinator amendment 4): <= 4 numeric threads per host in total; the encoder runs one process with
# 2 OpenMP/MKL/OpenBLAS threads, 2 torch intra-op threads and 2 landmark-engine threads (phases never overlap).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

RECIPE_VERSION = "h317-views-v1"
N_VIEWS = 32
VIEW_SUBSETS: Dict[str, Tuple[int, ...]] = {
    "A4": (0, 5, 12, 23),
    "B16": (0, 1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 23, 24, 26, 28, 30),
    "C32": tuple(range(32)),
}
VIEW_DIR = C.DATA_ROOT / "views"
VIEW_SEARCH_DIR = C.DATA_ROOT / "views-search"
NEURAL_ROOT = C.DATA_ROOT / "neural"
RECEIPTS = NEURAL_ROOT / "receipts"
MERGED_NAME = "views_c32_train1219.npz"
H58_TRAIN_DIR = NEURAL_ROOT / "h58train"
H58_TRAIN_CTX = "context_features_train1219.npz"
FEAT_KEYS = ("tokens_10x32", "dope629", "lm48_z")
VIEW_FEAT_DIM = (
    320 + 629 + 48
)  # flat per-view layout used by the neural ranker: [tokens_10x32 | dope629 | lm48_z]

FUSED_SOURCE_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
MICRO_PINS = {
    "assets/projections/frozen_stage1_stage2_inference_params_v3.pt": "136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d",
    "assets/models/dope_mlc_v3_multitask_best.pt": "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952",
    "assets/projections/program-feature-names.json": "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962",
    "python/rust_evaluator.py": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
    "python/model.py": "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a",
    "python/phase12_multimodal_model.py": "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a",
}

# ----------------------------------------------------------------------------------------------------------------------
# View recipes
# ----------------------------------------------------------------------------------------------------------------------
R1_ROW_FRACS = (1.0, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90)
R2_COL_FRACS = (0.55, 0.55, 0.65, 0.65, 0.75, 0.75, 0.85, 0.85)
R3_FAMILIES = (
    "gauss_uniform",
    "heavy_tail",
    "collinear",
    "interaction",
    "warp",
    "step",
    "fourier",
    "cocktail",
)
R3_FRACS = (0.35, 0.30, 0.40, 0.35, 0.35, 0.35, 0.35, 0.60)
COCKTAIL_ORDER = (
    "gauss_uniform",
    "heavy_tail",
    "collinear",
    "interaction",
    "warp",
    "step",
    "fourier",
)
R4_DISTRACTORS = ("gauss", "uniform", "t3", "cauchy")


def r4_setting(k: int) -> Tuple[float, float, float]:
    """(row_frac, col_frac, noise_level) for compound view 24+k: a de-correlated deterministic grid."""
    return (
        0.65 + 0.10 * k / 7.0,
        0.75 + 0.10 * ((3 * k) % 8) / 7.0,
        0.015 + 0.015 * ((5 * k) % 8) / 7.0,
    )


def view_rng(cid: str, v: int, kind: str = "full") -> np.random.Generator:
    """kind "full" (the whole TRAIN table; the original seed stream) or "search" (SEARCH rows only, own stream)."""
    if kind == "full":
        return np.random.default_rng(C.stream_seed("H317-view", RECIPE_VERSION, cid, v))
    return np.random.default_rng(
        C.stream_seed("H317-view", RECIPE_VERSION, cid, kind, v)
    )


def standardize(x: np.ndarray) -> np.ndarray:
    xf = np.where(np.isfinite(x), x, np.nan)
    with np.errstate(all="ignore"):
        mu = np.nanmean(xf, axis=0)
        sd = np.nanstd(xf, axis=0)
    mu = np.nan_to_num(mu)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return np.nan_to_num(np.clip((xf - mu) / sd, -8.0, 8.0))


def stratified_row_crop(
    y: np.ndarray, folds: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Rows kept per (native fold x target decile) stratum, returned in a shuffled order."""
    keep: List[np.ndarray] = []
    for f in np.unique(folds):
        idx = np.flatnonzero(folds == f)
        order = idx[np.argsort(y[idx], kind="stable")]
        for b in np.array_split(order, 10):
            if len(b):
                k = min(len(b), max(1, int(round(frac * len(b)))))
                keep.append(rng.choice(b, size=k, replace=False))
    return rng.permutation(np.concatenate(keep))


def leverage_column(z: np.ndarray, y: np.ndarray) -> int:
    yz = (y - y.mean()) / max(float(y.std()), 1e-12)
    corr = np.abs((z * yz[:, None]).mean(axis=0))
    return int(np.argmax(np.nan_to_num(corr)))


def column_crop(
    width: int, frac: float, rng: np.random.Generator, drop: int | None = None
) -> np.ndarray:
    cols = np.arange(width)
    if drop is not None and width > 1:
        cols = cols[cols != drop]
    k = min(len(cols), max(1, int(round(frac * width))))
    return rng.choice(cols, size=k, replace=False)  # random order = column permutation


def _two(width: int, rng: np.random.Generator) -> Tuple[int, int]:
    a = int(rng.integers(0, width))
    if width == 1:
        return a, a
    b = int(rng.integers(0, width - 1))
    return a, b + (b >= a)


def synth_column(
    family: str, i: int, x: np.ndarray, z: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One synthetic column of `family`; `i` alternates the family's two sub-kinds."""
    n, w = z.shape
    if family == "cocktail":
        return synth_column(COCKTAIL_ORDER[i % 7], i // 7, x, z, rng)
    if family in ("gauss_uniform", "gauss", "uniform"):
        if family == "gauss" or (family == "gauss_uniform" and i % 2 == 0):
            return rng.standard_normal(n)
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), n)
    if family in ("heavy_tail", "t3", "cauchy"):
        if family == "t3" or (family == "heavy_tail" and i % 2 == 0):
            return np.clip(rng.standard_t(3.0, n), -50.0, 50.0)
        return np.clip(rng.standard_cauchy(n), -100.0, 100.0)
    if family == "collinear":
        a, b = _two(w, rng)
        alpha = float(rng.uniform(0.0, 1.0))
        return alpha * z[:, a] + (1.0 - alpha) * z[:, b] + 0.15 * rng.standard_normal(n)
    if family == "interaction":
        a, b = _two(w, rng)
        if i % 2 == 0:
            p = z[:, a] * z[:, b]
            return np.clip((p - p.mean()) / max(float(p.std()), 1e-6), -10.0, 10.0)
        return z[:, a] ** 2 - 1.0
    if family == "warp":
        a = int(rng.integers(0, w))
        col = np.nan_to_num(np.where(np.isfinite(x[:, a]), x[:, a], 0.0))
        if i % 2 == 0:
            return np.sign(col) * np.log1p(np.abs(col))
        from scipy.special import ndtri
        from scipy.stats import rankdata

        return ndtri((rankdata(col) - 0.5) / n)
    if family == "step":
        a = int(rng.integers(0, w))
        q = np.quantile(z[:, a], 0.25 if i % 2 == 0 else 0.75)
        return (z[:, a] > q).astype(np.float64)
    if family == "fourier":
        a = int(rng.integers(0, w))
        omega = float(rng.uniform(0.5, 2.0))
        if i % 2 == 0:
            return np.sin(omega * z[:, a] + float(rng.uniform(0.0, 2.0 * math.pi)))
        return np.cos(omega * z[:, a])
    raise ValueError(f"unknown synthetic family {family}")


def n_synth(width: int, frac: float) -> int:
    return max(4, min(20, int(math.ceil(frac * width))))


def build_view(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    cid: str,
    v: int,
    kind: str = "full",
):
    """Return (x_v float64 [n_v, w_v], y_v float64 [n_v], folds_v uint8 [n_v], meta dict) for view v of context cid."""
    if not 0 <= v < N_VIEWS:
        raise ValueError(v)
    nt, width = x.shape
    rng = view_rng(cid, v, kind)
    meta = {
        "view": v,
        "regime": v // 8 + 1,
        "recipe": RECIPE_VERSION,
        "kind_of_table": kind,
    }
    if v == 0:
        meta.update(kind="canonical")
        return x, y, folds, meta
    if v < 8:
        rows = stratified_row_crop(y, folds, R1_ROW_FRACS[v], rng)
        meta.update(kind="row_crop", row_frac=R1_ROW_FRACS[v])
        return x[rows], y[rows], folds[rows], meta
    if v < 16:
        drop = leverage_column(standardize(x), y) if v == 15 else None
        cols = column_crop(width, R2_COL_FRACS[v - 8], rng, drop=drop)
        meta.update(
            kind="col_crop", col_frac=R2_COL_FRACS[v - 8], dropped_leverage_col=drop
        )
        return x[:, cols], y, folds, meta
    if v < 24:
        fam, frac = R3_FAMILIES[v - 16], R3_FRACS[v - 16]
        z = standardize(x)
        k = n_synth(width, frac)
        syn = np.column_stack([synth_column(fam, i, x, z, rng) for i in range(k)])
        meta.update(kind="synthetic", family=fam, n_synth=k)
        return np.concatenate([x, syn], axis=1), y, folds, meta
    row_frac, col_frac, noise = r4_setting(v - 24)
    rows = stratified_row_crop(y, folds, row_frac, rng)
    xr, yr, fr = x[rows], y[rows], folds[rows]
    cols = column_crop(width, col_frac, rng)
    xc = xr[:, cols]
    zc = standardize(xc)
    k = n_synth(len(cols), 0.30)
    syn = np.column_stack(
        [synth_column(R4_DISTRACTORS[i % 4], 0, xc, zc, rng) for i in range(k)]
    )
    xa = np.concatenate([xc, syn], axis=1)
    xa = xa[:, rng.permutation(xa.shape[1])]
    eps = np.clip(rng.standard_t(3.0, len(yr)) / math.sqrt(3.0), -1.345, 1.345)
    yn = yr + noise * float(np.std(yr)) * eps
    meta.update(
        kind="compound", row_frac=row_frac, col_frac=col_frac, noise=noise, n_synth=k
    )
    return xa, yn, fr, meta


def canonical_table_sha256(cid: str) -> str | None:
    """sha256 of the context's canonical train-only JTT1 table from the pinned selected.json (TRAIN or DL-Val row)."""
    sel = C.load_selected()
    for role in ("train", "dev"):
        for r in sel[role]:
            if r.get("context_sha256") == cid:
                return r.get("train_only_jtt1_sha256")
    return None


def build_32view_4regime_batch(
    cid: str, env: C.Jtt1, kind: str = "full", verify_canonical: bool = True
):
    """THE canonical view recipe (one recipe for both kinds).

    kind "full": views of the whole canonical table. kind "search": views of the SEARCH envelope only
    (`common_h317.search_and_audit_envelopes(cid, env)[0]`: the 70% non-AUDIT rows with their native 2-fold labels);
    no AUDIT row enters any search view. `env` must ALWAYS be the canonical full table (its bytes must hash to the
    selected.json table sha), so the AUDIT mask is applied exactly once, here; passing an already-cut SEARCH envelope
    (with either kind) raises. `verify_canonical=False` only for synthetic test tables."""
    if verify_canonical:
        want = canonical_table_sha256(cid)
        got = hashlib.sha256(env.to_bytes()).hexdigest()
        if want is None or got != want:
            raise ValueError(
                f"{cid[:12]}: env is not the canonical full table (sha {got[:12]} != {str(want)[:12]}); "
                "pass the full table, the AUDIT mask is applied once inside kind='search'"
            )
    if kind == "search":
        env = C.search_and_audit_envelopes(cid, env)[0]
    elif kind != "full":
        raise ValueError(kind)
    x = env.mat[:, :-1]
    y = env.mat[:, -1]
    return [build_view(x, y, env.folds, cid, v, kind) for v in range(N_VIEWS)]


def view_digest(xv: np.ndarray, yv: np.ndarray, fv: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (
        np.ascontiguousarray(xv, "<f8"),
        np.ascontiguousarray(yv, "<f8"),
        np.ascontiguousarray(fv, "u1"),
    ):
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def pack_view_feats(
    tokens_10x32: np.ndarray, dope629: np.ndarray, lm48_z: np.ndarray
) -> np.ndarray:
    """Flat per-view context vector [..., 997] = [tokens_10x32 (320) | dope629 (629) | lm48_z (48)]."""
    lead = dope629.shape[:-1]
    return np.concatenate(
        [
            np.asarray(tokens_10x32, np.float32).reshape(*lead, 320),
            np.asarray(dope629, np.float32),
            np.asarray(lm48_z, np.float32),
        ],
        axis=-1,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Frozen DOPE encoder (CPU, exact H58 path)
# ----------------------------------------------------------------------------------------------------------------------
def landmark_float32(features: np.ndarray) -> np.ndarray:
    """Copied from research/flywheel_selfplay/device_pool_cache.py::landmark_float32 (origin/v1/next)."""
    values = np.asarray(features, dtype=np.float64)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "h317_dope_freeze_guard", HERE / "dope_freeze_guard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class DopeViewEncoder:
    def __init__(self, threads: int = 4):
        import torch

        torch.set_num_threads(threads)
        for rel, want in MICRO_PINS.items():
            got = C.sha256_file(C.MICRO / rel)
            if got != want:
                raise RuntimeError(f"JopeDimeMicro pin mismatch {rel}: {got}")
        fused_path = C.RES01 / "cpu_exact_fused.py"
        if C.sha256_file(fused_path) != FUSED_SOURCE_SHA256:
            raise RuntimeError("cpu_exact_fused.py sha mismatch")
        sys.path.insert(0, str(C.MICRO / "python"))
        from hpc.landmark_workspace import FusedLandmarkWorkspace

        spec = importlib.util.spec_from_file_location(
            "h317_cpu_exact_fused", fused_path
        )
        fused = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused)
        self.torch = torch
        self.device = torch.device("cpu")
        self.net = fused.FusedJopeNetwork(
            micro_root=C.MICRO, device=self.device, adapter_variant="h40b"
        )
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.guard = _load_guard()
        self.guard.lock_dope_weights_and_eval_mode(self.net)
        self.dope_sha_before = self.dope_sha()
        if self.dope_sha_before != C.DOPE_SHA256:
            raise RuntimeError(f"DOPE sha {self.dope_sha_before} != {C.DOPE_SHA256}")
        self.ws = FusedLandmarkWorkspace(num_threads=threads)

    def dope_sha(self) -> str:
        return self.guard.snapshot_dope_weights(self.net).composite_sha256

    def encode(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Byte-for-byte the H58 `extract_single_table_features` path for one table."""
        torch = self.torch
        yd = np.asarray(y, dtype=np.float64)
        yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
        vec, cols, _ = self.ws.fit_transform_and_extract_bundle_high(
            landmark_float32(x), np.ascontiguousarray(yn)
        )
        colbuf = np.zeros((64, 17), np.float32)
        ncols = min(len(cols), 64)
        colbuf[:ncols, : min(cols.shape[1], 17)] = cols[:ncols, :17]
        with torch.no_grad():
            d629, lm48 = self.net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vec, device=self.device).reshape(1, 256),
                torch.as_tensor(colbuf, device=self.device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, mask, lm48_z = self.net._project_stage1b_tokens_batched_gpu(lm48, d629)
        return {
            "tokens_10x32": tok[0].cpu().numpy().astype(np.float32),
            "mask_10": mask[0].cpu().numpy().astype(bool),
            "dope629": d629[0].cpu().numpy().astype(np.float32),
            "lm48_z": lm48_z[0].cpu().numpy().astype(np.float32),
        }


def ctx_feat_584(dope629: np.ndarray, lm48_z: np.ndarray) -> np.ndarray:
    out = np.zeros(dope629.shape[:-1] + (584,), np.float32)
    out[..., 88:200] = dope629[..., :112]
    out[..., 536:584] = lm48_z
    return out


MASK_10_BASE = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)


def encode_context(
    enc: DopeViewEncoder, cid: str, row: dict, kind: str = "full"
) -> Dict[str, np.ndarray]:
    tpath = C.table_path(cid)
    if C.sha256_file(tpath) != row["train_only_jtt1_sha256"]:
        raise RuntimeError(f"table sha mismatch {cid}")
    env = C.read_jtt1(tpath)
    out = {k: [] for k in FEAT_KEYS}
    n_rows, n_cols, n_syn, digests = [], [], [], []
    views = build_32view_4regime_batch(cid, env, kind)
    if kind == "search" and len(views[0][1]) != env.nt - int(
        C.audit_mask(cid, env.y).sum()
    ):
        raise RuntimeError("SEARCH view 0 is not exactly the non-AUDIT rows")
    for xv, yv, fv, meta in views:
        f = enc.encode(xv, yv)
        if not np.array_equal(f["mask_10"], MASK_10_BASE):
            raise RuntimeError("mask_10 is not the constant H58 base mask")
        for k in FEAT_KEYS:
            out[k].append(f[k])
        n_rows.append(len(yv))
        n_cols.append(xv.shape[1])
        n_syn.append(int(meta.get("n_synth", 0)))
        digests.append(view_digest(xv, yv, fv))
    res = {k + "_f16": np.stack(v).astype(np.float16) for k, v in out.items()}
    res.update({k + "_v0_f32": out[k][0] for k in FEAT_KEYS})
    res.update(
        n_rows=np.array(n_rows, np.int32),
        n_cols=np.array(n_cols, np.int32),
        n_synth=np.array(n_syn, np.int32),
        view_digest=np.array(digests),
        recipe=np.array(RECIPE_VERSION),
        kind=np.array(kind),
        table_nt=np.array(env.nt, np.int32),
    )
    return res


def _save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _view_file_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return str(z["recipe"]) == RECIPE_VERSION and z["dope629_f16"].shape == (
                N_VIEWS,
                629,
            )
    except Exception:
        return False


def search_pilot_rows(n: int) -> List[dict]:
    """TRAIN rows of the GP pilot shard `common_h317.pilot_shard(rows, folds, n)` (the SEARCH-row view set)."""
    rows = C.train_rows()
    pilot = set(C.pilot_shard(rows, C.family_folds(rows), n=n))
    return [r for r in rows if r["context_sha256"] in pilot]


def cmd_encode(args) -> None:
    C.install_guard()
    if args.kind == "search":
        rows = search_pilot_rows(args.pilot_n)
        if args.view_dir == str(VIEW_DIR):
            args.view_dir = str(VIEW_SEARCH_DIR)
    else:
        rows = C.train_rows()
    shard = rows[args.shard :: args.nshards]
    if args.limit:
        shard = shard[: args.limit]
    t0 = time.time()
    enc = DopeViewEncoder(threads=args.threads)
    done = skipped = 0
    per_ctx = []
    for row in shard:
        cid = row["context_sha256"]
        path = Path(args.view_dir) / f"{cid}.npz"
        if _view_file_ok(path):
            skipped += 1
            continue
        _headroom()
        t1 = time.time()
        _save_npz(path, encode_context(enc, cid, row, args.kind))
        per_ctx.append(time.time() - t1)
        done += 1
        if done % 10 == 0:
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "done": done,
                        "skipped": skipped,
                        "of": len(shard),
                        "sec_per_ctx": round(float(np.mean(per_ctx[-10:])), 3),
                    }
                ),
                flush=True,
            )
    sha_after = enc.dope_sha()
    receipt = {
        "schema": "jope.h317.view-encode-shard/v1",
        "kind": args.kind,
        "pilot_n": args.pilot_n if args.kind == "search" else None,
        "host": os.uname().nodename,
        "shard": args.shard,
        "nshards": args.nshards,
        "contexts": len(shard),
        "encoded": done,
        "skipped_existing": skipped,
        "sec_per_ctx_mean": float(np.mean(per_ctx)) if per_ctx else None,
        "elapsed_s": round(time.time() - t0, 2),
        "threads": args.threads,
        "dope_sha256_before": enc.dope_sha_before,
        "dope_sha256_after": sha_after,
        "dope_unchanged": sha_after == enc.dope_sha_before == C.DOPE_SHA256,
        "recipe": RECIPE_VERSION,
        "guard": C.guard_report(),
    }
    C.write_json(
        RECEIPTS
        / f"encode_{args.kind}_{os.uname().nodename}_s{args.shard}of{args.nshards}.json",
        receipt,
    )
    print(json.dumps(receipt), flush=True)
    if not receipt["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


def _headroom() -> None:
    while True:
        with open("/proc/meminfo") as f:
            mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in f}
        if mem["MemAvailable"] >= 8 * 1024 * 1024:
            return
        print("MemAvailable < 8 GiB; waiting", flush=True)
        time.sleep(30)


def load_h58_train_ctx(path: Path | None = None) -> Dict[str, np.ndarray]:
    """The staged TRAIN-only H58 context-feature extract (made by stage_train_inputs.py via the H308 prefix loader)."""
    path = path or (H58_TRAIN_DIR / H58_TRAIN_CTX)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cmd_regress(args) -> None:
    C.install_guard()
    rows = C.train_rows()
    ref = load_h58_train_ctx()
    pos = {str(c): i for i, c in enumerate(ref["cids"])}
    keys = ("tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584")
    exact = {k: 0 for k in keys}
    maxabs = {k: 0.0 for k in keys}
    missing = []
    for row in rows:
        cid = row["context_sha256"]
        path = VIEW_DIR / f"{cid}.npz"
        if not _view_file_ok(path):
            missing.append(cid)
            continue
        with np.load(path, allow_pickle=False) as z:
            got = {k: z[k + "_v0_f32"] for k in FEAT_KEYS}
        got["mask_10"] = MASK_10_BASE
        got["ctx_feat_584"] = ctx_feat_584(got["dope629"], got["lm48_z"])
        i = pos[cid]
        for k in keys:
            a = np.asarray(got[k], np.float64)
            b = np.asarray(ref[k][i], np.float64)
            if np.array_equal(a, b):
                exact[k] += 1
            maxabs[k] = max(maxabs[k], float(np.max(np.abs(a - b))))
    n = len(rows) - len(missing)
    report = {
        "schema": "jope.h317.view0-regression/v1",
        "host": os.uname().nodename,
        "contexts_checked": n,
        "missing": len(missing),
        "exact_contexts": exact,
        "max_abs_diff": maxabs,
        "all_exact": n == len(rows) and all(v == n for v in exact.values()),
        "guard": C.guard_report(),
    }
    C.write_json(RECEIPTS / f"view0_regression_{os.uname().nodename}.json", report)
    print(json.dumps(report, indent=1))


def cmd_xcheck(args) -> None:
    """Cross-host determinism: compare every view array of the contexts present in `other_dir` with VIEW_DIR."""
    C.install_guard()
    other = Path(args.other_dir)
    keys = (
        [k + "_f16" for k in FEAT_KEYS]
        + [k + "_v0_f32" for k in FEAT_KEYS]
        + ["n_rows", "n_cols", "view_digest"]
    )
    n = 0
    exact = {k: 0 for k in keys}
    maxabs = {k: 0.0 for k in keys if not k.startswith("view_digest")}
    for p in sorted(other.glob("*.npz")):
        with (
            np.load(p, allow_pickle=False) as a,
            np.load(VIEW_DIR / p.name, allow_pickle=False) as b,
        ):
            n += 1
            for k in keys:
                exact[k] += int(np.array_equal(a[k], b[k]))
                if k in maxabs:
                    maxabs[k] = max(
                        maxabs[k],
                        float(
                            np.max(
                                np.abs(
                                    a[k].astype(np.float64) - b[k].astype(np.float64)
                                )
                            )
                        ),
                    )
    rep = {
        "schema": "jope.h317.view-xhost-check/v1",
        "host": os.uname().nodename,
        "contexts": n,
        "exact": exact,
        "max_abs_diff": maxabs,
        "all_exact": all(v == n for v in exact.values()),
        "guard": C.guard_report(),
    }
    C.write_json(RECEIPTS / f"view_xhost_check_{os.uname().nodename}.json", rep)
    print(json.dumps(rep, indent=1))


def cmd_merge(args) -> None:
    C.install_guard()
    rows = C.train_rows()
    cids = [r["context_sha256"] for r in rows]
    acc = {k: [] for k in FEAT_KEYS}
    acc0 = {k: [] for k in FEAT_KEYS}
    meta = {k: [] for k in ("n_rows", "n_cols", "n_synth")}
    for cid in cids:
        with np.load(VIEW_DIR / f"{cid}.npz", allow_pickle=False) as z:
            if str(z["recipe"]) != RECIPE_VERSION:
                raise RuntimeError(cid)
            for k in FEAT_KEYS:
                acc[k].append(z[k + "_f16"])
                acc0[k].append(z[k + "_v0_f32"])
            for k in meta:
                meta[k].append(z[k])
    arrays = {"cids": np.array(cids), "recipe": np.array(RECIPE_VERSION)}
    arrays.update({k + "_f16": np.stack(v) for k, v in acc.items()})
    arrays.update({k + "_v0_f32": np.stack(v) for k, v in acc0.items()})
    arrays.update({k: np.stack(v) for k, v in meta.items()})
    out = NEURAL_ROOT / MERGED_NAME
    _save_npz(out, arrays)
    rec = {
        "path": str(out),
        "sha256": C.sha256_file(out),
        "contexts": len(cids),
        "guard": C.guard_report(),
    }
    C.write_json(RECEIPTS / "views_merged.json", rec)
    print(json.dumps(rec))


def context_view_feats(
    cid: str,
    kind: str = "full",
    views: Sequence[int] | None = None,
    view_dir: Path | None = None,
) -> np.ndarray:
    """[len(views), 997] float32 screener input for one context, keyed (cid, kind): kind "full" = views of the whole
    TRAIN table ($H317_DATA/views), "search" = views of the SEARCH rows only ($H317_DATA/views-search). View 0 is the
    exact float32 encoding; views 1-31 are the stored float16 values."""
    d = (
        Path(view_dir)
        if view_dir
        else (VIEW_DIR if kind == "full" else VIEW_SEARCH_DIR)
    )
    with np.load(d / f"{cid}.npz", allow_pickle=False) as z:
        if str(z["recipe"]) != RECIPE_VERSION or (
            "kind" in z.files and str(z["kind"]) != kind
        ):
            raise RuntimeError(f"{cid}: view file is not ({kind}, {RECIPE_VERSION})")
        f = pack_view_feats(z["tokens_10x32_f16"], z["dope629_f16"], z["lm48_z_f16"])
        f[0] = pack_view_feats(
            z["tokens_10x32_v0_f32"], z["dope629_v0_f32"], z["lm48_z_v0_f32"]
        )
    idx = list(range(N_VIEWS)) if views is None else list(views)
    return f[idx]


def load_merged_views(path: Path | None = None) -> Dict[str, np.ndarray]:
    """Returns cids [N], feats [N, 32, 997] float16 (view 0 replaced by the exact float32 values when cast), meta."""
    path = path or (NEURAL_ROOT / MERGED_NAME)
    with np.load(path, allow_pickle=False) as z:
        if str(z["recipe"]) != RECIPE_VERSION:
            raise RuntimeError("views recipe mismatch")
        feats16 = pack_view_feats(
            z["tokens_10x32_f16"], z["dope629_f16"], z["lm48_z_f16"]
        ).astype(np.float16)
        v0 = pack_view_feats(
            z["tokens_10x32_v0_f32"], z["dope629_v0_f32"], z["lm48_z_v0_f32"]
        )
        return {
            "cids": z["cids"].astype(str),
            "feats16": feats16,
            "view0_f32": v0,
            "n_rows": z["n_rows"],
            "n_cols": z["n_cols"],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--nshards", type=int, default=1)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--kind", choices=("full", "search"), default="full")
    e.add_argument("--pilot-n", type=int, default=96)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--view-dir", default=str(VIEW_DIR))
    x = sub.add_parser("xcheck")
    x.add_argument("other_dir")
    sub.add_parser("regress")
    sub.add_parser("merge")
    args = ap.parse_args()
    {
        "encode": cmd_encode,
        "regress": cmd_regress,
        "merge": cmd_merge,
        "xcheck": cmd_xcheck,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
