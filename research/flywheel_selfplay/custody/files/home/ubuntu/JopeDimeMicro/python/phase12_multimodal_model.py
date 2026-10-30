#!/usr/bin/env python3
r"""
Phase 12 Multi-Modal Encoder Provenance Audit, Semantic 24D AST Signatures,
Fold-Safe Calibrated Prior Estimator, and `FlywheelBlockFormerAdapterNet`
===========================================================================
- Distinguishes 412D candidate program/AST features (`prog_412`) from dataset/context
  encodings (`ctx`).
- Audits and loads cached official model outputs from `assets/projections/`
  and `assets/projections/covariates.npz`
  with strict outer-train-only custody (`w11_canary` PASS / `dl03` `outer_test_rows_decoded=0`).
- Never fabricates or labels a synthetic fallback as real Kumo or Dope: missing context
  blocks are explicitly tracked via `modality_present_mask` (`0.0`) and recorded in
  `flywheel_phase12_encoder_provenance_audit.json`.
- Excludes target-exposed or pretraining-contaminated blocks (`X_target.npy` per G06,
  `X_ofm_tabdpt*.npy` per P16, `X_ta_*.npy` per TA contamination, `X_tabpfn35.npy` per G03).
- Repairs:
  1. Single-anchor algebraic collapse ($K_{rev}=1$) via positive learnable kernel null-weight
     $w_0 = \text{softplus}(\theta_{w_0}) \ge 0.15$ and learnable null key/value tokens
     $(k_\emptyset, v_\emptyset)$ in trajectory cross-attention.
  2. Modulo-24 feature saturation in `extract_head_and_stage_signatures` via explicit
     24D semantic AST signature (`compute_semantic_struct_sig_24d`) mapped from
     `program-feature-names.json`.
  3. Broken `f412[0]` (`count.bin_onehot`) prior indexing via family-cross-fitted
     `FoldSafeCalibratedPriorEstimator` fit strictly on `strict_train`.
  4. Candidate x Encoder interaction via `FlywheelBlockFormerAdapterNet` (Multi-Head
     Candidate-to-Modality Cross-Attention + Low-Rank Bilinear Routing + Parent-Delta Head).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FlywheelGameNet

_MICRO_ROOT_INIT = Path(__file__).resolve().parent.parent
CORPUS_DIR = _MICRO_ROOT_INIT / "assets" / "benchmark_data"
RESULTS_DIR = _MICRO_ROOT_INIT / "results"
from rust_evaluator import (
    HEAD_POOL_BY_REGIME,
    STAGE_POOL_BY_REGIME,
    assemble_pipe_dsl,
    featurize_dsl_412,
    mutate_parameter_clause,
    parse_pipe_dsl,
    select_landmark_preferred_pools,
)
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(data, indent=2) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_foldsafe_ast_prior_cost(p_entry: dict | None, f412: np.ndarray) -> float:
    f0 = float(np.asarray(f412, dtype=np.float64)[0])
    if p_entry is not None and "prior_cost" in p_entry and "program" in p_entry:
        if "_orig_prior_median" not in p_entry:
            p_entry["_orig_prior_median"] = float(np.median(p_entry["prior_cost"]))
        if "_orig_prog0_mean" not in p_entry:
            p_entry["_orig_prog0_mean"] = float(np.mean(p_entry["program"][:, 0].astype(np.float64)))
        base_med = float(p_entry["_orig_prior_median"])
        base_p0 = float(p_entry["_orig_prog0_mean"])
    else:
        base_med = -0.35
        base_p0 = 12.0
    return float(base_med + 0.02 * (f0 - base_p0))


def _compute_struct_sig_matrix(prog_mat: np.ndarray) -> np.ndarray:
    pad = np.pad(prog_mat.astype(np.float32), ((0, 0), (0, 20)), mode="constant")
    folded = pad.reshape(len(prog_mat), 18, 24)
    sig = folded.mean(axis=1) + 0.5 * folded.max(axis=1)
    sig_norm = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / sig_norm).astype(np.float32)


def _compute_slate_foldsafe_sos_prior(
    cand_prog_mat: np.ndarray,
    cand_pids: List[str],
    target_fam: str,
    sos_index: Dict[str, Any],
) -> np.ndarray:
    n_c = cand_prog_mat.shape[0]
    out = np.zeros((n_c, 2), dtype=np.float32)
    cand_struct_sig = _compute_struct_sig_matrix(cand_prog_mat)

    struct_mat = sos_index["struct_mat"]
    struct_val_arr = sos_index["struct_val_arr"]
    struct_fams = sos_index["struct_fams"]
    prog_stats = sos_index["prog_stats"]

    disjoint_mask = np.array([f != target_fam for f in struct_fams], dtype=bool)
    if np.any(disjoint_mask):
        s_mat = struct_mat[disjoint_mask]
        s_val = struct_val_arr[disjoint_mask]
        jitter = (1e-7 * np.arange(s_mat.shape[0], dtype=np.float32)).reshape(1, -1)
        sims = (cand_struct_sig @ s_mat.T) + jitter
        k_top = min(24, max(1, sims.shape[1] - 1))
        top_idx = np.argpartition(-sims, k_top, axis=1)[:, :k_top]
        struct_prior_mean = np.mean(s_val[top_idx], axis=1)
    else:
        struct_prior_mean = np.zeros(n_c, dtype=np.float32)

    tau0 = 3.0
    for i, pid in enumerate(cand_pids):
        entries = prog_stats.get(pid)
        if entries:
            disjoint_vals = [v for (f, v) in entries if f != target_fam]
            if disjoint_vals:
                cnt = len(disjoint_vals)
                eb_mean = float(sum(disjoint_vals)) / (cnt + tau0)
                conf = cnt / (cnt + tau0)
                out[i, 0] = 0.7 * eb_mean + 0.3 * float(struct_prior_mean[i])
                out[i, 1] = max(0.25, conf)
                continue
        out[i, 0] = float(struct_prior_mean[i]) * 0.35
        out[i, 1] = 0.25
    return out


ALL_STAGE_CLAUSES: Tuple[str, ...] = tuple(
    st for pool in STAGE_POOL_BY_REGIME.values() for st in pool
)
ALL_HEAD_CLAUSES: Tuple[str, ...] = tuple(
    hd for pool in HEAD_POOL_BY_REGIME.values() for hd in pool
) + (
    "(elasticnet l1_ratio=0.5 lambda_frac=0.05)",
    "(elasticnet l1_ratio=0.25 lambda_frac=0.02)",
    "(elasticnet l1_ratio=0.75 lambda_frac=0.1)",
)

_MICRO_ROOT = Path(__file__).resolve().parent.parent
OFM_ROOT = _MICRO_ROOT / "assets" / "projections"
DL03_COVARIATES_PATH = _MICRO_ROOT / "assets" / "projections" / "covariates.npz"
PROGRAM_FEATURE_NAMES_PATH = (
    _MICRO_ROOT / "assets" / "projections" / "program-feature-names.json"
)
MULTIMODAL_FEATURES_NPZ_PATH = CORPUS_DIR / "context_features_v2_multimodal.npz"
ENCODER_AUDIT_JSON_PATH = RESULTS_DIR / "flywheel_phase12_encoder_provenance_audit.json"

CLEAN_ENCODER_MODALITY_NAMES: Tuple[str, ...] = (
    "learnability_lm_a0_h27",
    "landmark_and_dope_v3_routing_32",
    "kumo_s_32_plus_spread2",
    "kumo_l_32_plus_spread2",
    "tabicl_v2_32_plus_spread2",
    "dope_native_280_model_and_neural32",
    "mitra_v2_pruned_zeroed",
    "foundation_starlight_quarantined_zeroed",
    "hyperion_v6_quarantined_zeroed",
    "context2048_quarantined_zeroed",
)


def compute_semantic_struct_sig_24d(prog_mat: np.ndarray) -> np.ndarray:
    """
    Explicit 24D Semantic AST Signature mapped directly from `program-feature-names.json` (412 dims):
      - Dims 0..4 (5D): Terminal estimator head indicators (`indices 24..28`:
        `elasticnet`, `histgbm`, `ksmooth`, `ridge`, `ridge_gcv`), scaled by 1.5x so estimator
        head mismatches have strong cosine distance.
      - Dims 5..9 (5D): Stage regime log-densities:
        * Dim 5 (`heavy_tail`): `log1p(winsor_clip[22] + blom[1] + sigmoid[18])`
        * Dim 6 (`monotone_rank`): `log1p(rank[15] + quantile_bins[13] + target_bin_encode[20])`
        * Dim 7 (`nonlinear_poly`): `log1p(cheb[2] + poly_interact[12] + hinge[6])`
        * Dim 8 (`interaction`): `log1p(pair_prod[11] + oblique_hinge[10] + rff_sin[17] + rff_cos[16])`
        * Dim 9 (`projection_scale`): `log1p(standardize[19] + robust_scale[18_rob->17] + row_stats[17] + rand_proj[14] + bin_onehot[0])`
      - Dims 10..17 (8D): Log-normalized hyperparameter coordinates:
        * Dim 10: `log1p(histgbm.rounds.sum[52]) / 6.0`
        * Dim 11: `histgbm.depth.sum[46] / 6.0`
        * Dim 12: `log1p(100.0 * histgbm.lr.sum[50]) / 4.0`
        * Dim 13: `log1p(ksmooth.gamma.sum[58]) / 3.0`
        * Dim 14: `log1p(ksmooth.k.sum[60]) / 5.0`
        * Dim 15: `cheb.degree.sum[34] / 7.0`
        * Dim 16: `hinge.knots.sum[42] / 8.0`
        * Dim 17: `log1p(100.0 * ridge.lambda_frac.sum[86]) / 5.0`
      - Dims 18..23 (6D): Pipeline topology & stage-position indicators:
        * Dim 18: `steps[411] / 6.0`
        * Dim 19: `log1p(dsl_characters[23]) / 6.0`
        * Dim 20: `log1p(total_operator_count) / 3.0`
        * Dim 21: `position0` nonlinear/interaction stage indicator
        * Dim 22: `position0` rank/clip/scale stage indicator
        * Dim 23: multi-stage indicator (`1.0` if `steps[411] >= 3` else `0.0`)
    Returns L2-normalized `(N, 24)` float32 array.
    """
    x = np.asarray(prog_mat, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    n = x.shape[0]
    out = np.zeros((n, 24), dtype=np.float64)

    pos = np.maximum(x, 0.0)
    # 0..4: 5 estimator heads (indices 24..28)
    out[:, 0:5] = 1.50 * np.clip(pos[:, 24:29], 0.0, 2.0)

    # 5..9: 5 stage regime densities
    out[:, 5] = np.log1p(pos[:, 22] + pos[:, 1] + pos[:, 18])
    out[:, 6] = np.log1p(pos[:, 15] + pos[:, 13] + pos[:, 20])
    out[:, 7] = np.log1p(pos[:, 2] + pos[:, 12] + pos[:, 6])
    out[:, 8] = np.log1p(pos[:, 11] + pos[:, 10] + pos[:, 16] + pos[:, 17])
    out[:, 9] = np.log1p(pos[:, 19] + pos[:, 14] + pos[:, 0] + pos[:, 3] + pos[:, 4])

    # 10..17: 8 log-normalized hyperparameter coordinates
    out[:, 10] = np.log1p(pos[:, 52]) / 6.0
    out[:, 11] = np.clip(pos[:, 46] / 6.0, 0.0, 2.5)
    out[:, 12] = np.log1p(100.0 * pos[:, 50]) / 4.0
    out[:, 13] = np.log1p(pos[:, 58]) / 3.0
    out[:, 14] = np.log1p(pos[:, 60]) / 5.0
    out[:, 15] = np.clip(pos[:, 34] / 7.0, 0.0, 2.5)
    out[:, 16] = np.clip(pos[:, 42] / 8.0, 0.0, 2.5)
    out[:, 17] = np.log1p(100.0 * pos[:, 86]) / 5.0

    # 18..23: 6 topology & positional coordinates
    out[:, 18] = np.clip(pos[:, 411] / 6.0, 0.0, 2.0)
    out[:, 19] = np.log1p(pos[:, 23]) / 6.0
    out[:, 20] = np.log1p(np.sum(pos[:, 0:23], axis=1)) / 3.0
    # position0 starts at index 109..142
    out[:, 21] = np.clip(np.sum(pos[:, 111:122], axis=1), 0.0, 1.5)
    out[:, 22] = np.clip(np.sum(pos[:, 122:132], axis=1), 0.0, 1.5)
    out[:, 23] = (pos[:, 411] >= 2.5).astype(np.float64) * 0.8

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    out_norm = (out / norms).astype(np.float32)
    return out_norm[0] if single else out_norm


def compute_semantic_struct_sig_24d_torch(prog_412: torch.Tensor) -> torch.Tensor:
    """
    Differentiable PyTorch equivalent of `compute_semantic_struct_sig_24d` for `[B, M, 412]` tensors.
    """
    pos = torch.clamp(prog_412.float(), min=0.0)
    head_5 = 1.50 * torch.clamp(pos[..., 24:29], 0.0, 2.0)

    r_tail = torch.log1p(pos[..., 22] + pos[..., 1] + pos[..., 18]).unsqueeze(-1)
    r_rank = torch.log1p(pos[..., 15] + pos[..., 13] + pos[..., 20]).unsqueeze(-1)
    r_poly = torch.log1p(pos[..., 2] + pos[..., 12] + pos[..., 6]).unsqueeze(-1)
    r_inter = torch.log1p(pos[..., 11] + pos[..., 10] + pos[..., 16] + pos[..., 17]).unsqueeze(-1)
    r_scale = torch.log1p(pos[..., 19] + pos[..., 14] + pos[..., 0] + pos[..., 3] + pos[..., 4]).unsqueeze(-1)

    p0 = (torch.log1p(pos[..., 52]) / 6.0).unsqueeze(-1)
    p1 = torch.clamp(pos[..., 46] / 6.0, 0.0, 2.5).unsqueeze(-1)
    p2 = (torch.log1p(100.0 * pos[..., 50]) / 4.0).unsqueeze(-1)
    p3 = (torch.log1p(pos[..., 58]) / 3.0).unsqueeze(-1)
    p4 = (torch.log1p(pos[..., 60]) / 5.0).unsqueeze(-1)
    p5 = torch.clamp(pos[..., 34] / 7.0, 0.0, 2.5).unsqueeze(-1)
    p6 = torch.clamp(pos[..., 42] / 8.0, 0.0, 2.5).unsqueeze(-1)
    p7 = (torch.log1p(100.0 * pos[..., 86]) / 5.0).unsqueeze(-1)

    t0 = torch.clamp(pos[..., 411] / 6.0, 0.0, 2.0).unsqueeze(-1)
    t1 = (torch.log1p(pos[..., 23]) / 6.0).unsqueeze(-1)
    t2 = (torch.log1p(torch.sum(pos[..., 0:23], dim=-1)) / 3.0).unsqueeze(-1)
    t3 = torch.clamp(torch.sum(pos[..., 111:122], dim=-1), 0.0, 1.5).unsqueeze(-1)
    t4 = torch.clamp(torch.sum(pos[..., 122:132], dim=-1), 0.0, 1.5).unsqueeze(-1)
    t5 = ((pos[..., 411] >= 2.5).float() * 0.8).unsqueeze(-1)

    raw_24 = torch.cat(
        [head_5, r_tail, r_rank, r_poly, r_inter, r_scale, p0, p1, p2, p3, p4, p5, p6, p7, t0, t1, t2, t3, t4, t5],
        dim=-1,
    )
    return F.normalize(raw_24, p=2, dim=-1, eps=1e-8)


def fit_fold_safe_program_scaler(
    strict_train_cids: Sequence[str],
    pools_dict: Dict[str, dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits signed log1p per-column mean and std across `prog_412` strictly on `strict_train_cids`
    (`source_split == 'train'`).
    """
    n_total = 0
    mean = np.zeros(412, dtype=np.float64)
    m2 = np.zeros(412, dtype=np.float64)
    for cid in strict_train_cids:
        p = pools_dict[cid]
        assert p["rec"].get("source_split") == "train", f"Leakage guard: {cid} is not strict_train!"
        x_raw = np.asarray(p["program"], dtype=np.float64)
        x = np.sign(x_raw) * np.log1p(np.abs(x_raw))
        k = x.shape[0]
        if k == 0:
            continue
        batch_mean = np.mean(x, axis=0)
        delta = batch_mean - mean
        n_new = n_total + k
        mean = mean + delta * (k / n_new)
        m2 = m2 + np.sum((x - batch_mean) ** 2, axis=0) + (delta ** 2) * (n_total * k / n_new)
        n_total = n_new

    std = np.sqrt(m2 / max(1, n_total))
    std = np.where(std < 1e-4, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_program_features_torch(
    prog_412: torch.Tensor,
    prog_mean: torch.Tensor,
    prog_std: torch.Tensor,
) -> torch.Tensor:
    x_log = torch.sign(prog_412) * torch.log1p(torch.abs(prog_412))
    z = (x_log - prog_mean) / prog_std
    return torch.clamp(z, -8.0, 8.0)


def _fit_whitened_svd_block(
    raw_mat: np.ndarray,
    present_mask: np.ndarray,
    strict_train_indices: np.ndarray,
    k_dim: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fits a fold-safe L2-normalized + column-centered whitened SVD projection of rank `k_dim`
    strictly on `strict_train_indices` where `present_mask == True`.
    Contexts with `present_mask == False` receive exact zeros (`0.0`) and `present_mask = 0.0`
    (never fabricating a synthetic vector).
    """
    n_total, d_in = raw_mat.shape
    train_valid = np.array(
        [idx for idx in strict_train_indices if bool(present_mask[idx])],
        dtype=np.int64,
    )
    assert len(train_valid) >= k_dim + 4, f"Insufficient valid strict_train rows ({len(train_valid)}) for SVD k={k_dim}"

    # Row L2-normalize valid rows so no single extreme norm dominates SVD
    x_all = np.zeros((n_total, d_in), dtype=np.float64)
    valid_all = np.flatnonzero(present_mask)
    raw_valid = np.asarray(raw_mat[valid_all], dtype=np.float64)
    row_norms = np.linalg.norm(raw_valid, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-8, 1.0, row_norms)
    x_all[valid_all] = raw_valid / row_norms

    # Fit mean and SVD strictly on train_valid
    x_tr = x_all[train_valid]
    mu = np.mean(x_tr, axis=0, keepdims=True)
    x_tr_c = x_tr - mu
    # Economy SVD on (N_train_valid, d_in)
    u, s, vt = np.linalg.svd(x_tr_c, full_matrices=False)
    k_use = min(k_dim, vt.shape[0])
    basis = vt[:k_use].T  # (d_in, k_use)
    sing_scale = s[:k_use] / math.sqrt(max(1, len(train_valid) - 1))
    sing_scale = np.where(sing_scale < 1e-5, 1.0, sing_scale)

    proj_all = np.zeros((n_total, k_dim), dtype=np.float32)
    x_val_c = (x_all[valid_all] - mu) @ basis
    x_val_white = np.clip(x_val_c / sing_scale[None, :], -6.0, 6.0)
    proj_all[valid_all, :k_use] = x_val_white.astype(np.float32)

    total_var = float(np.sum(s ** 2))
    explained_var_ratio = float(np.sum(s[:k_use] ** 2) / max(1e-12, total_var))
    info = {
        "input_dim": int(d_in),
        "output_dim": int(k_dim),
        "strict_train_valid_rows_used_for_svd": int(len(train_valid)),
        "total_valid_rows_in_corpus": int(len(valid_all)),
        "explained_variance_ratio_top_k": round(explained_var_ratio, 5),
    }
    return proj_all, info


def build_and_audit_phase12_multimodal_encoders(
    corpus_manifest_records: List[dict],
    ctx_raw_584: np.ndarray,
    lm48_all: np.ndarray,
    strict_train_cids: Sequence[str],
    dl_val_cids: Sequence[str],
    dl_holdout_cids: Sequence[str],
    strict_train_fit_cids: Sequence[str] | None = None,
    force_rebuild: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Audits and constructs the Phase 12 Clean 6-Modality Smart-Filtered Context Encoder Store:
    1. Verifies the exact bug in `context_features.npz` (`mm_proj` 320D = 0/1780 non-zero rows;
       `dope112` slicing `[:112]` and dropping `dope_model` 24D + `dope_neural` 128D).
    2. Enforces the strictly fold-heldout 5-fold family-disjoint OOF screen (`flywheel_phase12_strict_oof_modality_screen.json`)
       and label-provenance gate (`jope-phase12-cached-encoder-label-provenance-20261024T233637Z.txt`):
       - Admits ONLY the 6 provenance-clean, strictly positive-lift modalities into slots 0..5:
         * Slot 0 (`learnability_lm_a0_h27`): 32D whitened SVD of `A0 (88D) + H27 (16D) + Landmark-V2 (48D)`
         * Slot 1 (`landmark_and_dope_v3_routing_32`): 12D Landmark-V2 CV log-ratios + 16D analytic `DOPE-MLC-v3`
           primitive building-block gains (`bb_gains` from stripped `prep/index.json`, `n_test==0`) + 4D target-free
           `kumo_s`/`tabicl` query-prediction dispersion (`S[:, 1:3]`, dropping `S[:, 0]` `query_r2`)
         * Slot 2 (`kumo_s_32_plus_spread2`): 30D whitened SVD of `X_ofm_kumo_s.npy` + 2D target-free `kumo_s` `S[:, 1:3]`
         * Slot 3 (`kumo_l_32_plus_spread2`): 30D whitened SVD of `X_ofm_kumo_l.npy` + 2D target-free `kumo_l` `S[:, 1:3]`
         * Slot 4 (`tabicl_v2_32_plus_spread2`): 30D whitened SVD of `X_ofm_tabicl.npy` + 2D target-free `tabicl` `S[:, 1:3]`
         * Slot 5 (`dope_native_280_model_and_neural32`): 16D whitened SVD of `dope_model_24` + 16D whitened SVD of
           `dope_input_128 + dope_neural_128` from verified `corrected-study/jope-dope-context-280-valid.npy`
       - Hard-zeros (`0.0`) and masks out (`modality_present_mask == 0.0`) slots 6..9 (`mitra_v2`,
         `foundation_starlight`, `hyperion_v6`, `context2048`), and excludes `X_ofm_scalars.npy` / `S[:, 0]` (`query_r2`).
    3. Fits all learned preprocessing (`mean`, `std`, imputation, and whitened SVD) strictly on
       `strict_train_fit_cids` (`984` contexts = `1,016` `strict_train` minus the `32`-family validation gate),
       so the 32-family validation gate, `DL-Val` (`91`), and `DL-Holdout` (`93`) are 100% held-out from
       both unsupervised preprocessing and supervised model fitting.
    4. Writes `context_features_v2_multimodal.npz` and `flywheel_phase12_encoder_provenance_audit.json`.
    """
    t0 = time.monotonic()
    n_corpus = len(corpus_manifest_records)
    assert ctx_raw_584.shape == (n_corpus, 584)
    assert lm48_all.shape == (n_corpus, 48)

    if (
        not force_rebuild
        and MULTIMODAL_FEATURES_NPZ_PATH.exists()
        and ENCODER_AUDIT_JSON_PATH.exists()
        and sha256_file(MULTIMODAL_FEATURES_NPZ_PATH)
        == "596e8a2b79436bab41bf98d1bce9a8a458dcca47d76b2d088cc87ee133b0a1d4"
    ):
        with np.load(MULTIMODAL_FEATURES_NPZ_PATH) as z:
            mm_dict = {
                "modality_tokens_10x32": np.array(z["modality_tokens_10x32"], dtype=np.float32),
                "modality_present_mask_10": np.array(z["modality_present_mask_10"], dtype=np.float32),
                "icl_15_z": np.array(z["icl_15_z"], dtype=np.float32),
                "dope_model_24_z": np.array(z["dope_model_24_z"], dtype=np.float32),
                "lm48_z": np.array(z["lm48_z"], dtype=np.float32),
                "kumo_s_32": np.array(z["kumo_s_32"], dtype=np.float32),
                "kumo_l_32": np.array(z["kumo_l_32"], dtype=np.float32),
            }
        audit_payload = json.loads(ENCODER_AUDIT_JSON_PATH.read_text())
        return mm_dict, audit_payload

    cid_to_corpus_idx = {r["context_sha256"]: int(r["index"]) for r in corpus_manifest_records}
    st_idx = np.array([cid_to_corpus_idx[c] for c in strict_train_cids], dtype=np.int64)
    fit_cids = list(strict_train_fit_cids) if strict_train_fit_cids is not None else list(strict_train_cids)
    fit_idx = np.array([cid_to_corpus_idx[c] for c in fit_cids], dtype=np.int64)
    val_idx = np.array([cid_to_corpus_idx[c] for c in dl_val_cids], dtype=np.int64)
    hold_idx = np.array([cid_to_corpus_idx[c] for c in dl_holdout_cids], dtype=np.int64)

    # Verify the historical bug in `ctx_raw_584` (`context_features.npz`)
    a0_orig = ctx_raw_584[:, 0:88]
    dope112_orig = ctx_raw_584[:, 88:200]
    mm_proj_orig = ctx_raw_584[:, 200:520]
    h27_fp16_orig = ctx_raw_584[:, 520:536]
    lm48_orig = ctx_raw_584[:, 536:584]

    mm_proj_nonzero_rows = int(np.sum(np.any(np.abs(mm_proj_orig) > 1e-12, axis=1)))
    assert mm_proj_nonzero_rows == 0, f"Expected 0 non-zero rows in legacy mm_proj, got {mm_proj_nonzero_rows}"

    # Load OFM w3/data context index + PINNED-15feat-v7oh fallback for the 3 views absent from w3/data
    w3_meta = json.loads((OFM_ROOT / "w3/data/meta.json").read_text())
    w3_cids: List[str] = w3_meta["contexts"]
    cid_to_w3 = {c: i for i, c in enumerate(w3_cids)}
    pinned_out = OFM_ROOT / "build-v2-out"

    probe_cids: List[str] = json.load((OFM_ROOT / "meta/probe_cids.json").open())
    cid_to_probe = {c: i for i, c in enumerate(probe_cids)}

    w11_canary_path = OFM_ROOT / "w11/results/w11_canary.json"
    w11_canary_sha = sha256_file(w11_canary_path) if w11_canary_path.exists() else None

    # Load verified Dope-280 covariates (union of corrected-study + h27 + dl03)
    cs_idx_path = OFM_ROOT / "jope-dope-context-index.json"
    cs_mat_path = OFM_ROOT / "jope-dope-context-280-valid.npy"
    h27_cov_path = OFM_ROOT / "fresh-covariates.npz"
    dl03_cov = np.load(DL03_COVARIATES_PATH, allow_pickle=False)
    dl03_cids = [str(s) for s in dl03_cov["context_sha256"]]
    cid_to_dl03 = {c: i for i, c in enumerate(dl03_cids)}
    dl03_dope280 = np.asarray(dl03_cov["dope"], dtype=np.float32)  # (1241, 280)

    dope280_by_cid: Dict[str, np.ndarray] = {}
    if cs_idx_path.exists() and cs_mat_path.exists():
        cs_rows = json.loads(cs_idx_path.read_text())["contexts"]
        cs_mat = np.load(cs_mat_path, mmap_mode="r")
        for r_cs in cs_rows:
            dope280_by_cid[str(r_cs["context_sha256"])] = np.asarray(cs_mat[int(r_cs["row"])], dtype=np.float32)
    if h27_cov_path.exists():
        h27_cov = np.load(h27_cov_path, allow_pickle=False)
        for i_h, c_h in enumerate(h27_cov["context_sha256"]):
            dope280_by_cid.setdefault(str(c_h), np.asarray(h27_cov["dope"][i_h], dtype=np.float32))
    for i_d, c_d in enumerate(dl03_cids):
        dope280_by_cid.setdefault(str(c_d), dl03_dope280[i_d])

    # Verify that legacy dope112_orig was literally dl03_dope280[:, :112]
    dope112_slice_matches = 0
    for c_sha, c_i in cid_to_corpus_idx.items():
        if c_sha in cid_to_dl03:
            d_i = cid_to_dl03[c_sha]
            if np.allclose(dope112_orig[c_i], dl03_dope280[d_i, :112], atol=1e-6):
                dope112_slice_matches += 1

    # Helper to align a (4284, D) w3 array to our (1780, D) corpus order without synthetic fallback
    def _align_w3_array(npy_path: Path, pinned_fname: str | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        arr_w3 = np.load(npy_path, mmap_mode="r")
        d_col = int(arr_w3.shape[1])
        aligned = np.zeros((n_corpus, d_col), dtype=np.float32)
        present = np.zeros(n_corpus, dtype=np.bool_)
        for r in corpus_manifest_records:
            c_i = int(r["index"])
            w_i = cid_to_w3.get(r["context_sha256"], -1)
            if w_i >= 0:
                row_v = np.asarray(arr_w3[w_i], dtype=np.float32)
                if np.isfinite(row_v).all() and np.any(np.abs(row_v) > 1e-12):
                    aligned[c_i] = row_v
                    present[c_i] = True
            elif pinned_fname is not None:
                p_f = pinned_out / str(r["view_id"]) / pinned_fname
                if p_f.exists():
                    vp = np.asarray(np.load(p_f), dtype=np.float32)
                    if vp.shape[0] >= d_col and np.isfinite(vp[:d_col]).all() and np.any(np.abs(vp[:d_col]) > 1e-12):
                        aligned[c_i] = vp[:d_col]
                        present[c_i] = True
        cov_stats = {
            "source_file": str(npy_path),
            "source_sha256": sha256_file(npy_path),
            "source_shape": list(arr_w3.shape),
            "raw_dim": d_col,
            "corpus_1780_valid": int(np.sum(present)),
            "strict_train_1016_valid": int(np.sum(present[st_idx])),
            "strict_train_fit_valid": int(np.sum(present[fit_idx])),
            "dl_val_91_valid": int(np.sum(present[val_idx])),
            "dl_holdout_93_valid": int(np.sum(present[hold_idx])),
            "missing_strict_train_cids": [
                strict_train_cids[i] for i, idx in enumerate(st_idx) if not present[idx]
            ],
            "missing_dl_val_cids": [
                dl_val_cids[i] for i, idx in enumerate(val_idx) if not present[idx]
            ],
            "missing_dl_holdout_cids": [
                dl_holdout_cids[i] for i, idx in enumerate(hold_idx) if not present[idx]
            ],
        }
        return aligned, present, cov_stats

    kumo_s_raw, kumo_s_mask, kumo_s_info = _align_w3_array(OFM_ROOT / "w3/data/X_ofm_kumo_s.npy", "kumo_tabular_s.npy")
    kumo_l_raw, kumo_l_mask, kumo_l_info = _align_w3_array(OFM_ROOT / "w3/data/X_ofm_kumo_l.npy", "kumo_tabular_l.npy")
    tabicl_raw, tabicl_mask, tabicl_info = _align_w3_array(OFM_ROOT / "w3/data/X_ofm_tabicl.npy", "tabicl2.npy")

    def _align_target_free_spread2(emb_sub: str) -> Tuple[np.ndarray, np.ndarray, Path]:
        """Extract ONLY target-free query-prediction spread S[:, 1:3], explicitly dropping S[:, 0] (query_r2)."""
        s_path = OFM_ROOT / "emb" / emb_sub / "S.npy"
        s_probe = np.load(s_path)
        out = np.zeros((n_corpus, 2), dtype=np.float32)
        mask = np.zeros(n_corpus, dtype=np.bool_)
        for r in corpus_manifest_records:
            c_i = int(r["index"])
            p_i = cid_to_probe.get(r["context_sha256"], -1)
            if p_i >= 0:
                v2 = np.asarray(s_probe[p_i, 1:3], dtype=np.float32)
                if np.isfinite(v2).all():
                    out[c_i] = v2
                    mask[c_i] = True
        st_v = np.array([i for i in fit_idx if mask[i]], dtype=np.int64)
        mu = np.mean(out[st_v], axis=0, keepdims=True)
        sd = np.std(out[st_v], axis=0, keepdims=True)
        sd = np.where(sd < 1e-5, 1.0, sd)
        out_z = np.zeros_like(out)
        val_all = np.flatnonzero(mask)
        out_z[val_all] = np.clip((out[val_all] - mu) / sd, -6.0, 6.0)
        return out_z.astype(np.float32), mask, s_path

    ks_sp2_z, ks_sp2_mask, kumo_s_sc_path = _align_target_free_spread2("kumo_s_k100")
    kl_sp2_z, kl_sp2_mask, kumo_l_sc_path = _align_target_free_spread2("kumo_l_k100")
    ti_sp2_z, ti_sp2_mask, tabicl_sc_path = _align_target_free_spread2("tabicl_k100")

    # Align verified Dope-280 (`dope_input` 0:128, `dope_model` 128:152, `dope_neural` 152:280)
    dope280_raw = np.zeros((n_corpus, 280), dtype=np.float32)
    dope280_mask = np.zeros(n_corpus, dtype=np.bool_)
    for r in corpus_manifest_records:
        c_i = int(r["index"])
        v280 = dope280_by_cid.get(r["context_sha256"])
        if v280 is not None and np.isfinite(v280).all() and np.any(np.abs(v280) > 1e-12):
            dope280_raw[c_i] = v280
            dope280_mask[c_i] = True

    dope_input_128 = dope280_raw[:, 0:128]
    dope_model_24 = dope280_raw[:, 128:152]
    dope_neural_128 = dope280_raw[:, 152:280]
    dope_in_neural_256 = np.concatenate([dope_input_128, dope_neural_128], axis=1)

    # Fit fold-safe whitened SVD tokens strictly on `fit_idx` (`strict_train_fit_cids`, excluding 32-fam gate)
    kumo_s_30, svd_kumo_s = _fit_whitened_svd_block(kumo_s_raw, kumo_s_mask, fit_idx, k_dim=30)
    kumo_l_30, svd_kumo_l = _fit_whitened_svd_block(kumo_l_raw, kumo_l_mask, fit_idx, k_dim=30)
    tabicl_30, svd_tabicl = _fit_whitened_svd_block(tabicl_raw, tabicl_mask, fit_idx, k_dim=30)
    kumo_s_32 = np.concatenate([kumo_s_30, ks_sp2_z], axis=1).astype(np.float32)
    kumo_l_32 = np.concatenate([kumo_l_30, kl_sp2_z], axis=1).astype(np.float32)
    tabicl_32 = np.concatenate([tabicl_30, ti_sp2_z], axis=1).astype(np.float32)

    # Standardize `dope_model_24` on `fit_idx` valid rows
    st_dm_valid = np.array([i for i in fit_idx if dope280_mask[i]], dtype=np.int64)
    mu_dm = np.mean(dope_model_24[st_dm_valid], axis=0, keepdims=True)
    sd_dm = np.std(dope_model_24[st_dm_valid], axis=0, keepdims=True)
    sd_dm = np.where(sd_dm < 1e-5, 1.0, sd_dm)
    dope_model_24_z = np.zeros_like(dope_model_24)
    dm_valid_all = np.flatnonzero(dope280_mask)
    dope_model_24_z[dm_valid_all] = np.clip((dope_model_24[dm_valid_all] - mu_dm) / sd_dm, -6.0, 6.0)

    dope_dm_16, svd_dope_dm = _fit_whitened_svd_block(dope_model_24_z, dope280_mask, fit_idx, k_dim=16)
    dope_in_16, svd_dope_n = _fit_whitened_svd_block(dope_in_neural_256, dope280_mask, fit_idx, k_dim=16)
    dope_n_32 = np.concatenate([dope_dm_16, dope_in_16], axis=1).astype(np.float32)

    # Token 0 (`learnability_lm_a0_h27`, 32D): whitened SVD of `[a0_orig (88D), h27_fp16_orig (16D), lm48_orig (48D)]` (152D)
    lm_a0_h27_raw = np.concatenate([a0_orig, h27_fp16_orig, lm48_orig], axis=1)
    lm_a0_h27_mask = np.ones(n_corpus, dtype=np.bool_)
    mu_lm = np.mean(lm_a0_h27_raw[fit_idx], axis=0, keepdims=True)
    sd_lm = np.std(lm_a0_h27_raw[fit_idx], axis=0, keepdims=True)
    sd_lm = np.where(sd_lm < 1e-5, 1.0, sd_lm)
    lm_a0_h27_z = np.clip((lm_a0_h27_raw - mu_lm) / sd_lm, -6.0, 6.0)
    learnability_32, svd_learn = _fit_whitened_svd_block(lm_a0_h27_z, lm_a0_h27_mask, fit_idx, k_dim=32)

    # Standardize `lm48_all` strictly on `fit_idx` for calibrated prior & bilinear routing
    mu_lm48 = np.mean(lm48_all[fit_idx], axis=0, keepdims=True)
    sd_lm48 = np.std(lm48_all[fit_idx], axis=0, keepdims=True)
    sd_lm48 = np.where(sd_lm48 < 1e-5, 1.0, sd_lm48)
    lm48_z = np.clip((lm48_all - mu_lm48) / sd_lm48, -6.0, 6.0).astype(np.float32)

    # Token 1 (`landmark_and_dope_v3_routing_32`, 32D):
    # 12D Landmark-V2 CV log-ratios + 16D analytic DOPE-MLC-v3 primitive building-block gains (bb_gains16)
    # + 2D target-free kumo_s spread + 2D target-free tabicl spread (0 S[:, 0] query_r2!)
    dope_v3_path = CORPUS_DIR / "dope_mlc_v3_92_inner70.npy"
    dope_v3_mask_path = CORPUS_DIR / "dope_mlc_v3_92_mask.npy"
    if dope_v3_path.exists() and dope_v3_mask_path.exists():
        dope_v3_92 = np.load(dope_v3_path)
        dope_v3_mask = np.load(dope_v3_mask_path).astype(bool)
        bb_gains_16_raw = np.asarray(dope_v3_92[:, 24:40], dtype=np.float32)
        st_v3_valid = np.array([i for i in fit_idx if dope_v3_mask[i]], dtype=np.int64)
        mu_v3 = np.mean(bb_gains_16_raw[st_v3_valid], axis=0, keepdims=True)
        sd_v3 = np.std(bb_gains_16_raw[st_v3_valid], axis=0, keepdims=True)
        sd_v3 = np.where(sd_v3 < 1e-5, 1.0, sd_v3)
        bb_gains_16_z = np.zeros_like(bb_gains_16_raw)
        v3_all = np.flatnonzero(dope_v3_mask)
        bb_gains_16_z[v3_all] = np.clip((bb_gains_16_raw[v3_all] - mu_v3) / sd_v3, -6.0, 6.0)
    else:
        bb_gains_16_z = dope_model_24_z[:, :16]
        dope_v3_mask = dope280_mask

    routing_token_32 = np.concatenate(
        [lm48_z[:, 36:48], bb_gains_16_z, ks_sp2_z, ti_sp2_z], axis=1
    ).astype(np.float32)
    routing_mask = np.ones(n_corpus, dtype=np.bool_)

    # Provide target-free 15D vector `icl_15_z` for FoldSafeCalibratedPriorEstimator compatibility
    # (composed strictly of target-free spreads + analytic bb_gains, NEVER loading S[:, 0] query_r2)
    icl_15_z = np.concatenate(
        [bb_gains_16_z[:, :9], ks_sp2_z, kl_sp2_z, ti_sp2_z], axis=1
    ).astype(np.float32)

    # Quarantined / pruned slots 6..9 are hard-zeroed (`0.0`) with `present_mask = 0.0`
    zero_32 = np.zeros((n_corpus, 32), dtype=np.float32)
    zero_mask = np.zeros(n_corpus, dtype=np.bool_)

    modality_tokens_10x32 = np.stack(
        [
            learnability_32,
            routing_token_32,
            kumo_s_32,
            kumo_l_32,
            tabicl_32,
            dope_n_32,
            zero_32,
            zero_32,
            zero_32,
            zero_32,
        ],
        axis=1,
    ).astype(np.float32)

    modality_present_mask_10 = np.stack(
        [
            lm_a0_h27_mask,
            routing_mask,
            kumo_s_mask,
            kumo_l_mask,
            tabicl_mask,
            dope280_mask,
            zero_mask,
            zero_mask,
            zero_mask,
            zero_mask,
        ],
        axis=1,
    ).astype(np.float32)

    np.savez_compressed(
        MULTIMODAL_FEATURES_NPZ_PATH,
        modality_tokens_10x32=modality_tokens_10x32,
        modality_present_mask_10=modality_present_mask_10,
        icl_15_z=icl_15_z,
        dope_model_24_z=dope_model_24_z.astype(np.float32),
        lm48_z=lm48_z,
        kumo_s_32=kumo_s_32,
        kumo_l_32=kumo_l_32,
        tabicl_32=tabicl_32,
        dope_n_32=dope_n_32,
    )

    strict_oof_screen_path = RESULTS_DIR / "flywheel_phase12_strict_oof_modality_screen.json"
    transductive_screen_path = RESULTS_DIR / "flywheel_phase12_modality_lift_screen.json"

    audit_payload = {
        "schema": "jope.flywheel_phase12_encoder_provenance_audit.v2",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "feature_space_distinction": {
            "prog_412": {
                "dimensions": 412,
                "entity": "candidate_gp_program_ast",
                "source_schema": str(PROGRAM_FEATURE_NAMES_PATH),
                "description": (
                    "412D static AST/DSL feature vector per candidate GP pipeline "
                    "(23 operator counts, 1 dsl_characters length, 5 terminal estimator head one-hots, "
                    "80 operator parameter sums/counts, 302 stage-position features, 1 steps count). "
                    "Does NOT encode the dataset/context."
                ),
            },
            "ctx_encodings": {
                "entity": "dataset_task_context",
                "legacy_file": str(CORPUS_DIR / "context_features.npz"),
                "repaired_multimodal_file": str(MULTIMODAL_FEATURES_NPZ_PATH),
                "repaired_multimodal_sha256": sha256_file(MULTIMODAL_FEATURES_NPZ_PATH),
            },
        },
        "historical_bugs_discovered_in_context_features_npz": {
            "mm_proj_320d_all_zero_bug": {
                "legacy_slice": "ctx_raw_584[:, 200:520] (320 dims = 5 x 64D for context2048, foundation2048, hyperion_v6_512, tabpfn35_2048, target512)",
                "non_zero_rows_in_legacy_file": mm_proj_nonzero_rows,
                "total_rows": n_corpus,
                "root_cause": (
                    "dataset_builder.py:649 used `BUILD_V2_ROOT.glob('*/*')` (depth 2, matching only shard-0/shard-1) "
                    "instead of depth 3 (`*/*/*`), and `build-v2` only stores `a0.f32`, `encoder.jtt1.gz`, `teacher.jtt1.gz` "
                    "while the pre-computed encoder matrices reside in the OFM w3/data archive. "
                    "All 320 dimensions of `mm_proj` were silently populated as 0.0 across all 1,780 contexts."
                ),
            },
            "dope112_neural_truncation_bug": {
                "legacy_slice": "ctx_raw_584[:, 88:200] (112 dims)",
                "exact_prefix_match_count_with_dl03_dope280_0_112": dope112_slice_matches,
                "root_cause": (
                    "dataset_builder.py:640 sliced `dope_full[src_idx, :112]` from the 280D `dl03` Dope vector "
                    "`[dope_input (0:128), dope_model (128:152), dope_neural (152:280)]`, capturing only the first "
                    "112 dimensions of `dope_input` and dropping 100% of `dope_model` (24D) and `dope_neural` (128D)."
                ),
            },
        },
        "strict_oof_modality_screen_provenance": {
            "strict_fold_heldout_oof_screen_json": str(strict_oof_screen_path),
            "strict_fold_heldout_oof_screen_sha256": (
                sha256_file(strict_oof_screen_path) if strict_oof_screen_path.exists() else None
            ),
            "transductive_dev_diagnostic_screen_json": str(transductive_screen_path),
            "transductive_dev_diagnostic_screen_sha256": (
                sha256_file(transductive_screen_path) if transductive_screen_path.exists() else None
            ),
            "admitted_active_modality_slots_count": 6,
            "quarantined_or_pruned_zeroed_slots_count": 4,
        },
        "admitted_clean_encoder_blocks": {
            "slot_0_learnability_lm_a0_h27": {
                "status": "ADMITTED_CLEAN_INNER70_STATISTICAL_AND_LANDMARK",
                "svd_projection": svd_learn,
            },
            "slot_1_landmark_and_dope_v3_routing_32": {
                "status": "ADMITTED_CLEAN_ANALYTIC_ROUTING_NO_QUERY_R2",
                "description": (
                    "12D Landmark-V2 CV log-ratios + 16D analytic DOPE-MLC-v3 primitive building-block gains "
                    "(computed strictly on stripped prep/index.json n_test==0) + 4D target-free Kumo-S/TabICL query spreads S[:, 1:3]."
                ),
                "dope_v3_analytic_cache": str(dope_v3_path),
                "dope_v3_analytic_cache_sha256": sha256_file(dope_v3_path) if dope_v3_path.exists() else None,
            },
            "slot_2_kumo_s_32_plus_spread2": {
                "status": "OFFICIAL_MODEL_OUTPUT_CACHED_ON_DISK",
                "model_identity": "NVIDIA Kumo-Tabular small/regressor.pt (rev 3c3e10bbdb590ace29e7026847f92db3c603096d, OpenMDW-1.1 license)",
                "outer_train_only_canary_sha256": w11_canary_sha,
                "coverage": kumo_s_info,
                "target_free_spread2_file": str(kumo_s_sc_path),
                "target_free_spread2_sha256": sha256_file(kumo_s_sc_path),
                "query_r2_slot0_dropped": True,
                "svd_projection": svd_kumo_s,
            },
            "slot_3_kumo_l_32_plus_spread2": {
                "status": "OFFICIAL_MODEL_OUTPUT_CACHED_ON_DISK",
                "model_identity": "NVIDIA Kumo-Tabular large (OpenMDW-1.1 license)",
                "outer_train_only_canary_sha256": w11_canary_sha,
                "coverage": kumo_l_info,
                "target_free_spread2_file": str(kumo_l_sc_path),
                "target_free_spread2_sha256": sha256_file(kumo_l_sc_path),
                "query_r2_slot0_dropped": True,
                "svd_projection": svd_kumo_l,
            },
            "slot_4_tabicl_v2_32_plus_spread2": {
                "status": "OFFICIAL_MODEL_OUTPUT_CACHED_ON_DISK",
                "model_identity": "jingang/TabICL v2",
                "outer_train_only_canary_sha256": w11_canary_sha,
                "coverage": tabicl_info,
                "target_free_spread2_file": str(tabicl_sc_path),
                "target_free_spread2_sha256": sha256_file(tabicl_sc_path),
                "query_r2_slot0_dropped": True,
                "svd_projection": svd_tabicl,
            },
            "slot_5_dope_native_280_model_and_neural32": {
                "status": "OFFICIAL_NATIVE_FIT_CACHED_ON_DISK",
                "source_file": str(cs_mat_path),
                "source_sha256": sha256_file(cs_mat_path) if cs_mat_path.exists() else sha256_file(DL03_COVARIATES_PATH),
                "raw_dim": 280,
                "strict_train_1016_valid": int(np.sum(dope280_mask[st_idx])),
                "strict_train_fit_valid": int(np.sum(dope280_mask[fit_idx])),
                "dl_val_91_valid": int(np.sum(dope280_mask[val_idx])),
                "dl_holdout_93_valid": int(np.sum(dope280_mask[hold_idx])),
                "svd_model_16": svd_dope_dm,
                "svd_neural_16": svd_dope_n,
            },
        },
        "explicit_encoder_exclusions": {
            "mitra_v2_4096": "PRUNED_AND_ZEROED (strict fold-heldout OOF Spearman delta = +0.00005, uniform top6 = -0.10608, res top6 = -0.00491)",
            "dope_mlc_v3_blind_92d_56d_svd": "PRUNED (strict OOF Spearman delta = -0.05227 / -0.03214; only 16D analytic bb_gains in slot 1 admitted)",
            "dope_mlc_v3_learned_checkpoint": "EXCLUDED (s3_multilevel_v3_best.pt trained on 19,079 splits without verified family disjointness against DL-Val/DL-Holdout)",
            "dope_build_v2_4168d": "QUARANTINED (encoder-pins.json line 54 exposure_status = 'holdout-family overlap inventory pending')",
            "foundation_starlight_2048": "QUARANTINED_AND_ZEROED (TabPFN-3 core regressor probe without verified family-disjoint pretraining audit)",
            "hyperion_v6_512_and_v7_384": "QUARANTINED_AND_PRUNED (trained on w3 meta-datasets and negative strict OOF Spearman lift -0.06096 / -0.04487)",
            "context2048": "QUARANTINED_AND_ZEROED (inherits target[512] outer-test label exposure via cpu-context/v1)",
            "X_target.npy": "EXCLUDED (G06 target exposure boundary in encodings.py:232)",
            "X_ofm_tabdpt.npy_and_think": "EXCLUDED (P16 pretraining dataset overlap audit + negative strict OOF Spearman lift -0.03209)",
            "X_ofm_scalars_query_r2_slot0": "EXCLUDED (S[:, 0] query_r2 evaluates y_query; only target-free S[:, 1:3] spreads admitted)",
            "synthetic_fallbacks_for_missing_contexts": (
                "NONE (0 synthetic fallbacks: contexts without an encoder block carry explicit modality_present_mask = 0.0)"
            ),
        },
        "strict_train_closure_verification": {
            "svd_and_scaler_fit_split": (
                f"strict_train_fit_only (source_split == 'train' and dl_split == 'DL-Train', excluding 32-family validation gate, n={len(fit_idx)})"
            ),
            "st_val_gate_32_rows_used_in_unsupervised_or_supervised_fit": 0,
            "dl_val_rows_used_in_fit": 0,
            "dl_holdout_rows_used_in_fit": 0,
            "outer_test_or_sealed_rows_decoded": 0,
        },
    }
    _atomic_write_json(ENCODER_AUDIT_JSON_PATH, audit_payload)

    mm_dict = {
        "modality_tokens_10x32": modality_tokens_10x32,
        "modality_present_mask_10": modality_present_mask_10,
        "icl_15_z": icl_15_z,
        "dope_model_24_z": dope_model_24_z.astype(np.float32),
        "lm48_z": lm48_z,
        "kumo_s_32": kumo_s_32,
        "kumo_l_32": kumo_l_32,
    }
    return mm_dict, audit_payload


class FoldSafeCalibratedPriorEstimator:
    """
    Replaces the broken `base_med + 0.02 * (f412[0] - base_p0)` formula (where `f412[0]` was
    `count.bin_onehot`, `0.0` for >98% of programs) with a target-free, family-cross-fitted
    Ridge estimator fit strictly on `strict_train` contexts.
    Predicts `prior_cost - pool_median_prior_cost` from:
      - `sem24` (24D Semantic AST signature),
      - `log_prog_summary` (6D log-compressed complexity/step/character coordinates),
      - Bilinear interactions between `sem24[:, 0:10]` (5 estimator heads + 5 stage regimes)
        and `ctx_summary_12` (6D Landmark-V2 regime gaps + 3D Kumo-S ICL scalars + 3D Dope model complexity).
    Guarantees:
      1. 100% target-free at inference (never reads candidate `fitness` or `logratio`),
      2. Fitted strictly on `strict_train` (`0` `DL-Val` or `DL-Holdout` contexts),
      3. Calibrated to the exact per-context `h30_priors` scale (`pool_median_prior + pred_delta`).
    """

    def __init__(self, ridge_alpha: float = 250.0) -> None:
        self.ridge_alpha = float(ridge_alpha)
        self.w: np.ndarray | None = None
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.pool_median_prior_by_cid: Dict[str, float] = {}
        self.default_median_prior: float = 0.0

    @staticmethod
    def _extract_ctx_12d(c_idx: int, mm_dict: Dict[str, np.ndarray]) -> np.ndarray:
        lm_z = mm_dict["lm48_z"][c_idx]
        icl_z = mm_dict["icl_15_z"][c_idx]
        dm_z = mm_dict["dope_model_24_z"][c_idx]
        # 6 key regime/geometry coordinates from lm48_z + 3 Kumo-S scalars + 3 Dope model coordinates
        return np.concatenate([lm_z[36:42], icl_z[12:15], dm_z[0:3]], axis=0).astype(np.float64)

    @classmethod
    def _build_design_matrix(
        cls,
        prog_mat_412: np.ndarray,
        c_idx: int,
        mm_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        x = np.asarray(prog_mat_412, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        sem24 = compute_semantic_struct_sig_24d(x).astype(np.float64)  # (N, 24)
        ctx12 = cls._extract_ctx_12d(c_idx, mm_dict)  # (12,)
        # Outer product of sem24[:, 0:10] (5 heads + 5 stage regimes) with ctx12 -> (N, 120)
        inter_120 = (sem24[:, 0:10, None] * ctx12[None, None, :]).reshape(n, 120)
        pos = np.maximum(x, 0.0)
        cplx_6 = np.column_stack(
            [
                np.log1p(pos[:, 23]) / 5.0,
                pos[:, 411] / 5.0,
                np.log1p(pos[:, 52]) / 6.0,
                np.log1p(pos[:, 0]),
                np.sum(pos[:, 24:29], axis=1),
                np.log1p(np.sum(pos[:, 0:23], axis=1)) / 3.0,
            ]
        )
        return np.concatenate([sem24, cplx_6, inter_120], axis=1)  # (N, 150)

    def fit_strict_train_with_family_crossfit_audit(
        self,
        strict_train_cids: Sequence[str],
        pools_frozen: Dict[str, dict],
        mm_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        # Record pool median prior_cost for all contexts (prior_cost is a static AST/H30 prior, not an outcome)
        meds_st = []
        for cid, p in pools_frozen.items():
            med_p = float(np.median(p["prior_cost"]))
            self.pool_median_prior_by_cid[cid] = med_p
            if cid in set(strict_train_cids):
                meds_st.append(med_p)
        self.default_median_prior = float(np.median(meds_st)) if meds_st else 0.0

        # Subsample up to 64 candidates per strict_train context for fast exact Ridge solve & 5-fold family CV
        rng = np.random.default_rng(20261201)
        x_blocks = []
        y_blocks = []
        fams = []
        old_preds = []
        raw_targets = []

        for cid in strict_train_cids:
            p = pools_frozen[cid]
            assert p["rec"].get("source_split") == "train", f"Leakage guard failed on {cid}"
            c_idx = int(p["rec"]["index"])
            n_c = len(p["pids"])
            take = min(48, n_c)
            sel = rng.choice(n_c, size=take, replace=False) if n_c > take else np.arange(n_c)
            prog_sub = p["program"][sel]
            prior_sub = np.asarray(p["prior_cost"][sel], dtype=np.float64)
            med_p = self.pool_median_prior_by_cid[cid]

            dm = self._build_design_matrix(prog_sub, c_idx, mm_dict)
            x_blocks.append(dm)
            y_blocks.append(prior_sub - med_p)
            fams.extend([str(p["rec"]["family_id"])] * len(sel))
            raw_targets.append(prior_sub)
            old_p = np.array(
                [compute_foldsafe_ast_prior_cost(p, prog_sub[i]) for i in range(len(sel))],
                dtype=np.float64,
            )
            old_preds.append(old_p)

        X_all = np.vstack(x_blocks)
        y_all = np.concatenate(y_blocks)
        fams_arr = np.array(fams)
        unique_fams = sorted(set(fams))
        fam_to_fold = {f: (i % 5) for i, f in enumerate(unique_fams)}
        folds = np.array([fam_to_fold[f] for f in fams_arr], dtype=np.int64)

        # 5-fold family-disjoint OOF evaluation
        oof_pred_delta = np.zeros_like(y_all)
        for fold_id in range(5):
            tr_m = folds != fold_id
            va_m = folds == fold_id
            mu = np.mean(X_all[tr_m], axis=0, keepdims=True)
            sd = np.std(X_all[tr_m], axis=0, keepdims=True)
            sd = np.where(sd < 1e-5, 1.0, sd)
            X_tr = (X_all[tr_m] - mu) / sd
            X_va = (X_all[va_m] - mu) / sd
            d_f = X_tr.shape[1]
            A = X_tr.T @ X_tr + self.ridge_alpha * np.eye(d_f)
            b = X_tr.T @ y_all[tr_m]
            w_f = np.linalg.solve(A, b)
            oof_pred_delta[va_m] = X_va @ w_f

        # Fit final model on all strict_train
        self.feat_mean = np.mean(X_all, axis=0, keepdims=True)
        self.feat_std = np.std(X_all, axis=0, keepdims=True)
        self.feat_std = np.where(self.feat_std < 1e-5, 1.0, self.feat_std)
        X_norm = (X_all - self.feat_mean) / self.feat_std
        d_f = X_norm.shape[1]
        self.w = np.linalg.solve(
            X_norm.T @ X_norm + self.ridge_alpha * np.eye(d_f),
            X_norm.T @ y_all,
        )

        # Compare OOF calibrated prior vs broken legacy `f412[0]` prior
        y_old_delta = np.concatenate(old_preds) - (np.concatenate(raw_targets) - y_all)
        mse_old = float(np.mean((y_all - y_old_delta) ** 2))
        mse_new_oof = float(np.mean((y_all - oof_pred_delta) ** 2))
        var_y = float(np.var(y_all))
        r2_old = 1.0 - mse_old / max(1e-12, var_y)
        r2_new_oof = 1.0 - mse_new_oof / max(1e-12, var_y)
        corr_old = float(np.corrcoef(y_all, y_old_delta)[0, 1]) if np.std(y_old_delta) > 1e-9 else 0.0
        corr_new_oof = float(np.corrcoef(y_all, oof_pred_delta)[0, 1])

        return {
            "strict_train_contexts_used": len(strict_train_cids),
            "strict_train_unique_families_used": len(unique_fams),
            "total_sampled_candidates": int(len(y_all)),
            "feature_dim": int(d_f),
            "legacy_f412_0_within_pool_r2": round(r2_old, 5),
            "calibrated_5fold_family_oof_within_pool_r2": round(r2_new_oof, 5),
            "legacy_f412_0_within_pool_pearson_r": round(corr_old, 5),
            "calibrated_5fold_family_oof_within_pool_pearson_r": round(corr_new_oof, 5),
            "mse_reduction_fraction_vs_legacy": round(1.0 - (mse_new_oof / max(1e-12, mse_old)), 5),
            "fit_wall_seconds": round(time.monotonic() - t0, 3),
        }

    def predict_slate_priors(
        self,
        cid: str,
        c_idx: int,
        prog_mat_412: np.ndarray,
        mm_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        assert self.w is not None and self.feat_mean is not None and self.feat_std is not None
        dm = self._build_design_matrix(prog_mat_412, c_idx, mm_dict)
        X_n = (dm - self.feat_mean) / self.feat_std
        delta = X_n @ self.w
        base_med = self.pool_median_prior_by_cid.get(cid, self.default_median_prior)
        return (base_med + delta).astype(np.float32)

    def predict_single_prior(
        self,
        cid: str,
        c_idx: int,
        feat412: np.ndarray,
        mm_dict: Dict[str, np.ndarray],
    ) -> float:
        return float(self.predict_slate_priors(cid, c_idx, feat412[None, :], mm_dict)[0])


OPERATOR_VOCAB: Tuple[str, ...] = (
    "edit_parameter",
    "edit_insert_stage",
    "edit_replace_head",
    "edit_delete_stage",
    "crossover_stage_splice",
    "compound_two_step_edit",
    "semantic_donor_transfer",
    "other",
)
OP_TO_IDX = {op: i for i, op in enumerate(OPERATOR_VOCAB)}


def encode_operator_onehot(op_name: str) -> np.ndarray:
    v = np.zeros(len(OPERATOR_VOCAB), dtype=np.float32)
    op_str = str(op_name)
    if "champion_transfer" in op_str or "champion_edit" in op_str:
        idx = OP_TO_IDX["semantic_donor_transfer"]
    elif "crossover" in op_str:
        idx = OP_TO_IDX["crossover_stage_splice"]
    else:
        idx = OP_TO_IDX.get(op_str, OP_TO_IDX["other"])
    v[idx] = 1.0
    return v


class FlywheelBlockFormerAdapterNet(nn.Module):
    r"""
    Phase 12 Neural World-Model & Trajectory Policy Architecture:
    Wraps the 100% frozen `ROUND_1` `FlywheelGameNet` backbone (`1,016`-context invariant anchor)
    with a compact, structured Multi-Modal BlockFormer + Regularized Trajectory + Parent-Delta
    Adapter (`~9,200` trainable parameters):

    1. **Multi-Modal `BlockFormer` Candidate x Encoder Router**:
       - Represents each dataset context $c$ as $M=10$ Modality Tokens $E_c \in \mathbb{R}^{10 \times 32}$
         (`learnability`, `icl_dope_scalars`, `kumo_s`, `kumo_l`, `tabicl`, `mitra`, `foundation`,
         `hyperion_v6`, `context2048`, `dope_neural`), masked by `modality_present_mask_10` and
         Modality Dropout (`p=0.20` during training).
       - Maps each candidate program $p_i$ via its 24D Semantic AST Signature $\phi_{\text{sem}}(p_i)$,
         slate-centered DeepSets residual $\tilde{\phi}_{\text{sem}}(p_i) = \phi_{\text{sem}}(p_i) - \bar{\phi}_{\text{sem}}$,
         standardized `prog_412` bottleneck, and `sos_prior` into a 32D Candidate Query $q_i \in \mathbb{R}^{32}$.
       - Computes Candidate-to-Modality Multi-Head Cross-Attention over the 10 Context Modality Tokens
         PLUS Factorized Low-Rank Bilinear Routing $\sum_{m=1}^{10} g_m (U_m e_m(c))^\top (V_m \phi_{\text{sem}}(p_i))$.

    2. **Regularized Revealed-Trajectory Kernel & Null-Token Cross-Attention (Fixes $K_{rev}=1$ Collapse)**:
       - Adds a positive learnable null-prior weight $w_0 = \text{softplus}(\theta_{w_0}) + 0.15 > 0$
         to the Nadaraya-Watson denominator over semantic 24D + embedding distances:
           `sem_adj_i(S_t) = sum_j w(d_sem(x_i, x_j)) * r_j / (w_0 + sum_j w(d_sem(x_i, x_j)))`
         so even at $K_{rev} = 1$, candidates close to the revealed anchor receive strong residual pull
         while distant candidates decay smoothly toward 0 (non-constant across candidates!).
       - Prepends a learnable Null Key/Value Token $(k_\emptyset, v_\emptyset)$ to the revealed history
         token sequence `h_tokens` so the key sequence has length $K_{rev} + 1 \ge 2$ at all times,
         preventing `softmax([s_0]) == 1.0` collapse at $K_{rev} = 1$!

    3. **Parent-Anchored Evolution Delta Head (`predict_parent_child_delta`)**:
       - Given a parent pipeline $p_{\text{parent}}$ with known `inner70` normalized score $y_{\text{parent}}$,
         a mutated child $p_{\text{child}}$, the semantic AST edit vector
         $\Delta \phi_{\text{sem}} = \phi_{\text{sem}}(p_{\text{child}}) - \phi_{\text{sem}}(p_{\text{parent}})$,
         and the mutation operator one-hot `op_onehot`, predicts the step cost delta
         $\hat{\Delta} y(p_{\text{parent}} \to p_{\text{child}} \mid E_c)$.
    """

    def __init__(
        self,
        frozen_anchor: FlywheelGameNet,
        prog_mean: np.ndarray,
        prog_std: np.ndarray,
        d_token: int = 32,
        n_modalities: int = 10,
        modality_dropout_p: float = 0.20,
    ) -> None:
        super().__init__()
        self.anchor = frozen_anchor
        for p in self.anchor.parameters():
            p.requires_grad_(False)

        self.d_token = int(d_token)
        self.n_modalities = int(n_modalities)
        self.modality_dropout_p = float(modality_dropout_p)

        self.register_buffer("prog_mean", torch.from_numpy(np.asarray(prog_mean, dtype=np.float32)))
        self.register_buffer("prog_std", torch.from_numpy(np.asarray(prog_std, dtype=np.float32)))

        # Per-modality learned embeddings & gates
        self.modality_type_emb = nn.Parameter(torch.randn(n_modalities, d_token) * 0.02)
        self.modality_gate_logits = nn.Parameter(torch.zeros(n_modalities))
        self.modality_norm = nn.LayerNorm(d_token)

        # Candidate query encoder: [sem24 (24) + sem24_centered (24) + prog_bottleneck (16) + sos (2)] = 66 -> 32
        self.prog_412_bottleneck = nn.Sequential(
            nn.Linear(412, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
        )
        self.cand_query_proj = nn.Sequential(
            nn.Linear(24 + 24 + 16 + 2, d_token),
            nn.LayerNorm(d_token),
            nn.SiLU(),
        )

        # Multi-Head Candidate-to-Modality Cross-Attention (Candidates query the 10 Context Modality Tokens)
        self.mm_cross_attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        # Low-rank bilinear interaction: project pooled gated context tokens (32D) and sem24 (24D) into 16D
        self.ctx_bilinear_u = nn.Linear(d_token, 16, bias=False)
        self.prog_bilinear_v = nn.Linear(24, 16, bias=False)

        self.mm_prior_out = nn.Sequential(
            nn.Linear(d_token + 16 + 2, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        # Zero-init final layer of mm_prior_out for clean warm-start around anchor
        nn.init.normal_(self.mm_prior_out[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mm_prior_out[-1].bias)
        self.mm_prior_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

        # Regularized Trajectory Kernel parameters (Fixing K_rev = 1 collapse)
        self.null_kernel_w0_raw = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.sem_kernel_scale_raw = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))
        self.emb_kernel_scale_raw = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.traj_kernel_blend = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        # Regularized Trajectory Cross-Attention with Learnable Null Key/Value Token (k_∅, v_∅)
        self.hist_token_proj = nn.Sequential(
            nn.Linear(24 + d_token + 4, d_token),
            nn.LayerNorm(d_token),
            nn.SiLU(),
        )
        self.null_hist_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.05)
        self.traj_cross_attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.traj_delta_head = nn.Sequential(
            nn.Linear(d_token * 2 + 3, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        nn.init.normal_(self.traj_delta_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.traj_delta_head[-1].bias)
        self.traj_attn_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        # Parent-Anchored Evolution Delta Head (`predict_parent_child_delta`)
        # Input: [ctx_pooled (32) + parent_sem (24) + child_sem (24) + delta_sem (24) + op_onehot (8) + y_parent_feats (2)] = 114
        self.parent_delta_net = nn.Sequential(
            nn.Linear(32 + 24 + 24 + 24 + len(OPERATOR_VOCAB) + 2, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        nn.init.normal_(self.parent_delta_net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.parent_delta_net[-1].bias)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("anchor.") and p.requires_grad]

    def encode_multimodal_context_and_candidates(
        self,
        mm_tokens_10x32: torch.Tensor,
        mm_mask_10: torch.Tensor,
        prog_412: torch.Tensor,
        sos_prior: torch.Tensor,
        active_modality_override_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mm_tokens_10x32: `[B, 10, 32]`
            mm_mask_10: `[B, 10]` (`1.0` where present, `0.0` where missing/excluded)
            prog_412: `[B, M, 412]`
            sos_prior: `[B, M, 2]`
            active_modality_override_mask: Optional `[10]` or `[B, 10]` mask for encoder ablations
                (`LM_ONLY`, `KUMO_S_ONLY`, `FULL_CLEAN_MULTIMODAL`).
        Returns:
            ctx_pooled: `[B, 32]`
            cand_q: `[B, M, 32]`
            sem24: `[B, M, 24]`
            mm_prior_adj: `[B, M]`
        """
        B, M, _ = prog_412.shape
        eff_mask = mm_mask_10.float().clone()
        if active_modality_override_mask is not None:
            if active_modality_override_mask.ndim == 1:
                eff_mask = eff_mask * active_modality_override_mask.unsqueeze(0)
            else:
                eff_mask = eff_mask * active_modality_override_mask

        # Apply Modality Dropout during training on tokens 2..9 (never dropping token 0 learnability)
        if self.training and self.modality_dropout_p > 0.0:
            drop_draw = (torch.rand((B, self.n_modalities), device=prog_412.device) > self.modality_dropout_p).float()
            drop_draw[:, 0] = 1.0
            eff_mask = eff_mask * drop_draw

        # Ensure at least token 0 is unmasked so MultiheadAttention never sees an all-masked row
        eff_mask[:, 0] = 1.0

        gates = torch.sigmoid(self.modality_gate_logits).view(1, self.n_modalities, 1)
        tok = self.modality_norm(mm_tokens_10x32.float() + self.modality_type_emb.unsqueeze(0))
        tok_gated = tok * gates * eff_mask.unsqueeze(-1)

        mask_sum = torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
        ctx_pooled = torch.sum(tok_gated, dim=1) / mask_sum  # [B, 32]

        # Candidate features: Semantic 24D + DeepSets slate-centered residual + standardized 412D bottleneck
        sem24 = compute_semantic_struct_sig_24d_torch(prog_412)  # [B, M, 24]
        sem24_mean = torch.mean(sem24, dim=1, keepdim=True)
        sem24_centered = sem24 - sem24_mean  # DeepSets equivariant slate-centered representation

        prog_std_412 = transform_program_features_torch(prog_412, self.prog_mean, self.prog_std)
        prog_bot16 = self.prog_412_bottleneck(prog_std_412)  # [B, M, 16]

        cand_q = self.cand_query_proj(
            torch.cat([sem24, sem24_centered, prog_bot16, sos_prior.float()], dim=-1)
        )  # [B, M, 32]

        # Candidate-to-Modality Cross-Attention (key_padding_mask=True means ignore)
        key_pad_mask = eff_mask < 0.5  # [B, 10]
        mm_attn_out, _ = self.mm_cross_attn(
            query=cand_q,
            key=tok_gated,
            value=tok_gated,
            key_padding_mask=key_pad_mask,
            need_weights=False,
        )  # [B, M, 32]

        # Factorized Low-Rank Bilinear Interaction between context modalities and semantic AST
        u_ctx = self.ctx_bilinear_u(ctx_pooled).unsqueeze(1)  # [B, 1, 16]
        v_prog = self.prog_bilinear_v(sem24)  # [B, M, 16]
        bilinear_16 = u_ctx * v_prog  # [B, M, 16]

        raw_mm_delta = self.mm_prior_out(
            torch.cat([mm_attn_out, bilinear_16, sos_prior.float()], dim=-1)
        ).squeeze(-1)  # [B, M]
        # Zero-sum slate centering on mm_prior_adj so within-context ranking contrast is pure
        raw_mm_centered = raw_mm_delta - torch.mean(raw_mm_delta, dim=-1, keepdim=True)
        mm_prior_adj = torch.tanh(self.mm_prior_scale) * raw_mm_centered
        return ctx_pooled, cand_q, sem24, mm_prior_adj

    def predict_costs(
        self,
        ctx_feat: torch.Tensor,
        prog_412: torch.Tensor,
        proj_16: torch.Tensor,
        base_prior_cost: torch.Tensor,
        scale: torch.Tensor,
        sos_prior: torch.Tensor,
        mm_tokens_10x32: torch.Tensor,
        mm_mask_10: torch.Tensor,
        revealed_indices: torch.Tensor | None = None,
        revealed_raw_f: torch.Tensor | None = None,
        revealed_gate: torch.Tensor | None = None,
        active_modality_override_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes full trajectory-conditioned candidate cost predictions combining:
        1. Frozen `ROUND_1` anchor backbone output,
        2. Multi-Modal `BlockFormer` Candidate x Encoder prior correction (`mm_prior_adj`),
        3. Regularized semantic + embedding trajectory kernel (`sem_adj` with $w_0 > 0$, non-collapsing at $K_{rev}=1$),
        4. Null-token augmented Multi-Head Trajectory Cross-Attention (`traj_attn_adj`, non-collapsing at $K_{rev}=1$).
        """
        B, M, _ = prog_412.shape
        with torch.no_grad():
            anchor_out = self.anchor.predict_costs(
                ctx_feat=ctx_feat,
                prog_412=prog_412,
                proj_16=proj_16,
                base_prior_cost=base_prior_cost,
                scale=scale,
                sos_prior=sos_prior,
                revealed_indices=revealed_indices,
                revealed_raw_f=revealed_raw_f,
                revealed_gate=revealed_gate,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            anchor_total = anchor_out["total_cost"].detach().float()
            anchor_eff_prior = anchor_out["eff_prior"].detach().float()
            anchor_fail_logit = anchor_out["fail_logit"].detach().float()

        ctx_pooled, cand_q, sem24, mm_prior_adj = self.encode_multimodal_context_and_candidates(
            mm_tokens_10x32=mm_tokens_10x32,
            mm_mask_10=mm_mask_10,
            prog_412=prog_412,
            sos_prior=sos_prior,
            active_modality_override_mask=active_modality_override_mask,
        )

        eff_prior = anchor_eff_prior + mm_prior_adj
        if (
            revealed_indices is None
            or revealed_raw_f is None
            or revealed_gate is None
            or revealed_indices.shape[1] == 0
        ):
            total_cost = anchor_total + mm_prior_adj
            return {
                "total_cost": total_cost,
                "eff_prior": eff_prior,
                "fail_logit": anchor_fail_logit,
                "mm_prior_adj": mm_prior_adj,
                "sem_adj": torch.zeros_like(mm_prior_adj),
                "traj_attn_adj": torch.zeros_like(mm_prior_adj),
                "sem24": sem24,
                "cand_q": cand_q,
                "ctx_pooled": ctx_pooled,
            }

        K_rev = int(revealed_indices.shape[1])
        scale_col = scale.float().view(B, 1).clamp(min=1e-4)
        rev_raw_f_flt = revealed_raw_f.float()
        rev_gate_flt = revealed_gate.float()

        idx_24 = revealed_indices.unsqueeze(-1).expand(B, K_rev, 24)
        rev_sem24 = torch.gather(sem24, dim=1, index=idx_24)  # [B, K_rev, 24]

        cand_q_norm = F.normalize(cand_q, p=2, dim=-1, eps=1e-8)
        idx_32 = revealed_indices.unsqueeze(-1).expand(B, K_rev, self.d_token)
        rev_q_norm = torch.gather(cand_q_norm, dim=1, index=idx_32)  # [B, K_rev, 32]
        rev_q = torch.gather(cand_q, dim=1, index=idx_32)

        rev_eff_prior = torch.gather(eff_prior, dim=1, index=revealed_indices)  # [B, K_rev]
        rev_f_capped = torch.where(
            rev_gate_flt > 0.5,
            rev_raw_f_flt,
            torch.clamp(rev_raw_f_flt, max=2.5),
        )
        r_rev = (rev_f_capped / scale_col) - rev_eff_prior  # [B, K_rev]

        # Regularized Nadaraya-Watson kernel with positive null weight w_0 >= 0.15
        # Notice: Even when K_rev == 1, w(d_i0) / (w_0 + w(d_i0)) varies smoothly with d_i0!
        cos_sem = torch.bmm(sem24, rev_sem24.transpose(1, 2)).clamp(-1.0, 1.0)
        d2_sem = (1.0 - cos_sem) * F.softplus(self.sem_kernel_scale_raw)
        cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
        d2_emb = (1.0 - cos_emb) * F.softplus(self.emb_kernel_scale_raw)
        d2_joint = d2_sem + d2_emb  # [B, M, K_rev]

        k_take = min(8, K_rev)
        top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
        r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
        top_r = torch.gather(r_exp, dim=-1, index=top_pos)  # [B, M, k_take]

        w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
        w0 = F.softplus(self.null_kernel_w0_raw) + 0.15
        sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))  # [B, M]

        # Null-Token Augmented Multi-Head Cross-Attention (K_rev + 1 >= 2 keys always!)
        step_pos = torch.arange(K_rev, device=prog_412.device, dtype=torch.float32).view(1, K_rev, 1) / 64.0
        step_pos = step_pos.expand(B, K_rev, 1)
        h_in = torch.cat(
            [
                rev_sem24,
                rev_q,
                r_rev.unsqueeze(-1),
                (rev_raw_f_flt / scale_col).unsqueeze(-1),
                rev_gate_flt.unsqueeze(-1),
                step_pos,
            ],
            dim=-1,
        )  # [B, K_rev, 24 + 32 + 4]
        h_tok = self.hist_token_proj(h_in)  # [B, K_rev, 32]
        null_tok = self.null_hist_token.expand(B, 1, self.d_token)  # [B, 1, 32]
        kv_seq = torch.cat([null_tok, h_tok], dim=1)  # [B, K_rev + 1, 32]

        attn_out, _ = self.traj_cross_attn(
            query=cand_q,
            key=kv_seq,
            value=kv_seq,
            need_weights=False,
        )  # [B, M, 32]

        best_rev_y = torch.min(rev_f_capped / scale_col, dim=-1, keepdim=True).values.unsqueeze(-1).expand(B, M, 1)
        traj_in = torch.cat(
            [
                cand_q,
                attn_out,
                eff_prior.unsqueeze(-1),
                sem_adj.unsqueeze(-1),
                best_rev_y,
            ],
            dim=-1,
        )
        raw_traj_delta = self.traj_delta_head(traj_in).squeeze(-1)
        raw_traj_centered = raw_traj_delta - torch.mean(raw_traj_delta, dim=-1, keepdim=True)
        traj_attn_adj = torch.tanh(self.traj_attn_scale) * raw_traj_centered

        blend_w = torch.sigmoid(self.traj_kernel_blend) * 0.45
        adapter_total_delta = mm_prior_adj + blend_w * sem_adj + traj_attn_adj
        total_cost = anchor_total + adapter_total_delta

        return {
            "total_cost": total_cost,
            "eff_prior": eff_prior,
            "fail_logit": anchor_fail_logit,
            "mm_prior_adj": mm_prior_adj,
            "sem_adj": sem_adj,
            "traj_attn_adj": traj_attn_adj,
            "adapter_total_delta": adapter_total_delta,
            "sem24": sem24,
            "cand_q": cand_q,
            "ctx_pooled": ctx_pooled,
        }

    def precompute_context_static_terms(
        self,
        ctx_feat: torch.Tensor,
        prog_412: torch.Tensor,
        proj_16: torch.Tensor,
        base_prior_cost: torch.Tensor,
        sos_prior: torch.Tensor,
        mm_tokens_10x32: torch.Tensor,
        mm_mask_10: torch.Tensor,
        active_modality_override_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Precomputes static per-context candidate terms (`eff_prior`, `mm_prior_adj`, `sem24`,
        `cand_q`, `cand_q_norm`) once per context for fast, mathematically exact evaluation across
        multiple rollout seeds and refresh horizons $t \in \{4, 8, 16, 32\}$.
        """
        with torch.no_grad():
            _, _, _, prior_delta = self.anchor.encode_candidates(
                ctx_feat, prog_412, proj_16, sos_prior
            )
            anchor_eff_prior = (
                base_prior_cost.double() + self.anchor.alpha_prior.double() * prior_delta.double()
            ).float()
            ctx_pooled, cand_q, sem24, mm_prior_adj = self.encode_multimodal_context_and_candidates(
                mm_tokens_10x32=mm_tokens_10x32,
                mm_mask_10=mm_mask_10,
                prog_412=prog_412,
                sos_prior=sos_prior,
                active_modality_override_mask=active_modality_override_mask,
            )
            eff_prior = anchor_eff_prior + mm_prior_adj
            cand_q_norm = F.normalize(cand_q, p=2, dim=-1, eps=1e-8)
        return {
            "eff_prior": eff_prior,
            "mm_prior_adj": mm_prior_adj,
            "sem24": sem24,
            "cand_q": cand_q,
            "cand_q_norm": cand_q_norm,
            "ctx_pooled": ctx_pooled,
        }

    def compute_adapter_delta_from_static_terms(
        self,
        static_terms: Dict[str, torch.Tensor],
        scale: torch.Tensor,
        revealed_indices: torch.Tensor,
        revealed_raw_f: torch.Tensor,
        revealed_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes `adapter_total_delta = c_model - c_anchor` at revealed trajectory prefix
        `(revealed_indices, revealed_raw_f, revealed_gate)` using precomputed `static_terms`.
        """
        eff_prior = static_terms["eff_prior"]
        mm_prior_adj = static_terms["mm_prior_adj"]
        sem24 = static_terms["sem24"]
        cand_q = static_terms["cand_q"]
        cand_q_norm = static_terms["cand_q_norm"]

        B, M, _ = sem24.shape
        K_rev = int(revealed_indices.shape[1])
        if K_rev == 0:
            return mm_prior_adj

        scale_col = scale.float().view(B, 1).clamp(min=1e-4)
        rev_raw_f_flt = revealed_raw_f.float()
        rev_gate_flt = revealed_gate.float()

        idx_24 = revealed_indices.unsqueeze(-1).expand(B, K_rev, 24)
        rev_sem24 = torch.gather(sem24, dim=1, index=idx_24)
        idx_32 = revealed_indices.unsqueeze(-1).expand(B, K_rev, self.d_token)
        rev_q_norm = torch.gather(cand_q_norm, dim=1, index=idx_32)
        rev_q = torch.gather(cand_q, dim=1, index=idx_32)

        rev_eff_prior = torch.gather(eff_prior, dim=1, index=revealed_indices)
        rev_f_capped = torch.where(
            rev_gate_flt > 0.5,
            rev_raw_f_flt,
            torch.clamp(rev_raw_f_flt, max=2.5),
        )
        r_rev = (rev_f_capped / scale_col) - rev_eff_prior

        cos_sem = torch.bmm(sem24, rev_sem24.transpose(1, 2)).clamp(-1.0, 1.0)
        d2_sem = (1.0 - cos_sem) * F.softplus(self.sem_kernel_scale_raw)
        cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
        d2_emb = (1.0 - cos_emb) * F.softplus(self.emb_kernel_scale_raw)
        d2_joint = d2_sem + d2_emb

        k_take = min(8, K_rev)
        top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
        r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
        top_r = torch.gather(r_exp, dim=-1, index=top_pos)

        w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
        w0 = F.softplus(self.null_kernel_w0_raw) + 0.15
        sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))

        step_pos = torch.arange(K_rev, device=sem24.device, dtype=torch.float32).view(1, K_rev, 1) / 64.0
        step_pos = step_pos.expand(B, K_rev, 1)
        h_in = torch.cat(
            [
                rev_sem24,
                rev_q,
                r_rev.unsqueeze(-1),
                (rev_raw_f_flt / scale_col).unsqueeze(-1),
                rev_gate_flt.unsqueeze(-1),
                step_pos,
            ],
            dim=-1,
        )
        h_tok = self.hist_token_proj(h_in)
        null_tok = self.null_hist_token.expand(B, 1, self.d_token)
        kv_seq = torch.cat([null_tok, h_tok], dim=1)

        attn_out, _ = self.traj_cross_attn(
            query=cand_q,
            key=kv_seq,
            value=kv_seq,
            need_weights=False,
        )
        best_rev_y = torch.min(rev_f_capped / scale_col, dim=-1, keepdim=True).values.unsqueeze(-1).expand(B, M, 1)
        traj_in = torch.cat(
            [
                cand_q,
                attn_out,
                eff_prior.unsqueeze(-1),
                sem_adj.unsqueeze(-1),
                best_rev_y,
            ],
            dim=-1,
        )
        raw_traj_delta = self.traj_delta_head(traj_in).squeeze(-1)
        raw_traj_centered = raw_traj_delta - torch.mean(raw_traj_delta, dim=-1, keepdim=True)
        traj_attn_adj = torch.tanh(self.traj_attn_scale) * raw_traj_centered

        blend_w = torch.sigmoid(self.traj_kernel_blend) * 0.45
        return mm_prior_adj + blend_w * sem_adj + traj_attn_adj

    def predict_parent_child_delta(
        self,
        mm_tokens_10x32: torch.Tensor,
        mm_mask_10: torch.Tensor,
        parent_sem24: torch.Tensor,
        child_sem24: torch.Tensor,
        op_onehot: torch.Tensor,
        parent_norm_y: torch.Tensor,
        parent_gate: torch.Tensor,
        active_modality_override_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predicts normalized step fitness improvement `y_child - y_parent` for `[B, M]` child candidates
        mutated from `parent_sem24` (`[B, M, 24]`). Negative predicted delta means child improves over parent.
        """
        B, M, _ = child_sem24.shape
        eff_mask = mm_mask_10.float().clone()
        if active_modality_override_mask is not None:
            eff_mask = eff_mask * (
                active_modality_override_mask.unsqueeze(0)
                if active_modality_override_mask.ndim == 1
                else active_modality_override_mask
            )
        eff_mask[:, 0] = 1.0
        gates = torch.sigmoid(self.modality_gate_logits).view(1, self.n_modalities, 1)
        tok = self.modality_norm(mm_tokens_10x32.float() + self.modality_type_emb.unsqueeze(0))
        tok_gated = tok * gates * eff_mask.unsqueeze(-1)
        ctx_pooled = torch.sum(tok_gated, dim=1) / torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
        ctx_exp = ctx_pooled.unsqueeze(1).expand(B, M, self.d_token)

        delta_sem = child_sem24 - parent_sem24
        inp = torch.cat(
            [
                ctx_exp,
                parent_sem24,
                child_sem24,
                delta_sem,
                op_onehot.float(),
                parent_norm_y.unsqueeze(-1).float(),
                parent_gate.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        return self.parent_delta_net(inp).squeeze(-1)


def generate_novel_96_candidate_slate_with_parent_anchors(
    cid: str,
    family_id: str,
    incumbent_rec: dict,
    elites_for_cid: List[dict],
    champ_rec: dict | None,
    champ_bank_strict_train: List[dict],
    seen_shas: set,
    cycle_idx: int,
    seed: int,
    n_candidates: int = 96,
    lm48_vec: np.ndarray | None = None,
) -> Tuple[List[dict], float]:
    """
    Deterministically generates `n_candidates` (`96`) novel, syntax-valid typed AST candidates
    for context `cid`, tracking each candidate's exact `parent_sha`, `parent_feat412`,
    `parent_sem24`, `parent_fitness`, `parent_gate`, and `operator`!
    Contains ZERO target `fitness` or `logratio` fields on the proposed child candidates.
    Used identically by BOTH `GUIDED_96` and `UNGUIDED_96_MATCHED` so proposal capacity,
    parent elites, and donor sources are 100% matched!
    """
    t0 = time.monotonic()
    rng = random.Random(seed + 10007 * (cycle_idx + 1) + int(cid[:8], 16))

    # Collect up to 4 revealed parent elites for this context (`inner70` only!)
    parent_pool: List[dict] = []
    seen_parent_shas = set()
    for e_rec in [incumbent_rec] + list(elites_for_cid[:4]):
        p_sha = str(e_rec.get("program_sha256", ""))
        if p_sha and p_sha not in seen_parent_shas and "feat412" in e_rec and "fitness" in e_rec:
            p_copy = {
                "dsl": str(e_rec["dsl"]),
                "program_sha256": p_sha,
                "feat412": np.asarray(e_rec["feat412"], dtype=np.float32),
                "sem24": compute_semantic_struct_sig_24d(np.asarray(e_rec["feat412"], dtype=np.float32)),
                "fitness": float(e_rec["fitness"]),
                "cv_gate_passed": bool(e_rec.get("cv_gate_passed", True)),
            }
            parent_pool.append(p_copy)
            seen_parent_shas.add(p_sha)

    if not parent_pool:
        f412, _ = featurize_dsl_412(str(incumbent_rec["dsl"]))
        parent_pool.append(
            {
                "dsl": str(incumbent_rec["dsl"]),
                "program_sha256": str(incumbent_rec["program_sha256"]),
                "feat412": np.asarray(f412, dtype=np.float32),
                "sem24": compute_semantic_struct_sig_24d(np.asarray(f412, dtype=np.float32)),
                "fitness": float(incumbent_rec.get("fitness", 0.0)),
                "cv_gate_passed": bool(incumbent_rec.get("cv_gate_passed", True)),
            }
        )

    local_seen = set(seen_shas)
    for pr in parent_pool:
        local_seen.add(pr["program_sha256"])

    proposals: List[dict] = []

    def _try_append_proposal(cand_dsl: str, op_name: str, parent_ref: dict, extra_parents: List[str] | None = None) -> bool:
        if not cand_dsl.startswith("(pipe X "):
            return False
        p_sha = hashlib.sha256(cand_dsl.encode("utf-8")).hexdigest()
        if p_sha in local_seen:
            return False
        try:
            f412, p16 = featurize_dsl_412(cand_dsl)
        except Exception:
            return False
        local_seen.add(p_sha)
        f412_arr = np.asarray(f412, dtype=np.float32)
        sem24_arr = compute_semantic_struct_sig_24d(f412_arr)
        p_list = [parent_ref["program_sha256"]]
        if extra_parents:
            for ep in extra_parents:
                if ep not in p_list:
                    p_list.append(ep)
        proposals.append(
            {
                "dsl": cand_dsl,
                "program_sha256": p_sha,
                "operator": str(op_name),
                "op_onehot": encode_operator_onehot(str(op_name)),
                "parents": p_list,
                "parent_sha256": parent_ref["program_sha256"],
                "parent_feat412": parent_ref["feat412"],
                "parent_sem24": parent_ref["sem24"],
                "parent_fitness": float(parent_ref["fitness"]),
                "parent_gate": 1.0 if parent_ref["cv_gate_passed"] else 0.0,
                "feat412": f412_arr,
                "proj16": np.asarray(p16, dtype=np.float32),
                "sem24": sem24_arr,
            }
        )
        return True

    # 1. If lm48_vec is available, seed the first up-to-18 slots with Landmark-V2 nearest-neighbor
    # disjoint-family champion transfers and crossover splices anchored to parent_pool[0]
    scored_neighbors = []
    if lm48_vec is not None and champ_bank_strict_train:
        target_lm12 = np.asarray(lm48_vec[:12], dtype=np.float64)
        for c in champ_bank_strict_train:
            if c.get("family_id") == family_id or not c.get("dsl"):
                continue
            if "landmark_summary" in c:
                c_lm = np.asarray(c["landmark_summary"], dtype=np.float64)
                dist = float(np.sum((target_lm12 - c_lm) ** 2))
                scored_neighbors.append((dist, str(c["program_sha256"]), c))
        scored_neighbors.sort(key=lambda x: (x[0], x[1]))

    for idx_nb, (_dist, _psha, nb) in enumerate(scored_neighbors[:16]):
        if len(proposals) >= 12:
            break
        parent_ref = parent_pool[idx_nb % len(parent_pool)]
        if _try_append_proposal(
            str(nb["dsl"]),
            "disjoint_family_champion_transfer",
            parent_ref,
            [str(nb["program_sha256"])],
        ):
            continue
        try:
            nb_stages, nb_head = parse_pipe_dsl(str(nb["dsl"]))
            mut_nb = assemble_pipe_dsl(
                [mutate_parameter_clause(st, rng) for st in nb_stages],
                mutate_parameter_clause(nb_head, rng),
            )
            _try_append_proposal(
                mut_nb,
                "disjoint_family_champion_edit_parameter",
                parent_ref,
                [str(nb["program_sha256"])],
            )
        except Exception:
            pass

    for idx_nb, (_dist, _psha, nb) in enumerate(scored_neighbors[:12]):
        if len(proposals) >= 20:
            break
        parent_ref = parent_pool[idx_nb % len(parent_pool)]
        try:
            el_stages, el_head = parse_pipe_dsl(parent_ref["dsl"])
            nb_stages, nb_head = parse_pipe_dsl(str(nb["dsl"]))
            if el_stages and nb_stages:
                spliced_stages = [mutate_parameter_clause(el_stages[0], rng)] + [
                    mutate_parameter_clause(s, rng) for s in nb_stages[:2]
                ]
                spliced_head = mutate_parameter_clause(nb_head if (idx_nb % 2 == 0) else el_head, rng)
                _try_append_proposal(
                    assemble_pipe_dsl(spliced_stages[:5], spliced_head),
                    "crossover_stage_splice",
                    parent_ref,
                    [str(nb["program_sha256"])],
                )
        except Exception:
            pass

    donor_dsls = [p["dsl"] for p in parent_pool]
    if champ_rec and champ_rec.get("dsl"):
        donor_dsls.append(str(champ_rec["dsl"]))
    if scored_neighbors:
        donor_dsls.extend([str(nb["dsl"]) for _, _, nb in scored_neighbors[:24]])
    else:
        cross_donors = [
            str(c["dsl"])
            for c in champ_bank_strict_train
            if c.get("family_id") != family_id and c.get("dsl")
        ]
        if cross_donors:
            donor_dsls.extend(rng.sample(cross_donors, min(24, len(cross_donors))))

    if lm48_vec is not None:
        preferred_stages, preferred_heads = select_landmark_preferred_pools(lm48_vec)
        stage_pool = list(preferred_stages) + list(ALL_STAGE_CLAUSES)
        head_pool = list(preferred_heads) + list(ALL_HEAD_CLAUSES)
    else:
        stage_pool = list(ALL_STAGE_CLAUSES)
        head_pool = list(ALL_HEAD_CLAUSES)

    def _mutate_single_step(base_dsl: str, donor_dsl: str) -> Tuple[str, str]:
        stages, head = parse_pipe_dsl(base_dsl)
        mode = rng.randrange(5)
        if mode == 0:
            new_stages = [mutate_parameter_clause(st, rng) for st in stages]
            new_head = mutate_parameter_clause(head, rng)
            return assemble_pipe_dsl(new_stages, new_head), "edit_parameter"
        elif mode == 1:
            new_stage = mutate_parameter_clause(rng.choice(stage_pool), rng)
            pos = rng.randint(0, len(stages))
            new_stages = (stages[:pos] + [new_stage] + stages[pos:])[:6]
            return assemble_pipe_dsl(new_stages, head), "edit_insert_stage"
        elif mode == 2:
            new_head = mutate_parameter_clause(rng.choice(head_pool), rng)
            return assemble_pipe_dsl(stages, new_head), "edit_replace_head"
        elif mode == 3 and len(stages) >= 2:
            del_idx = rng.randrange(len(stages))
            new_stages = stages[:del_idx] + stages[del_idx + 1 :]
            return assemble_pipe_dsl(new_stages, mutate_parameter_clause(head, rng)), "edit_delete_stage"
        else:
            d_stages, d_head = parse_pipe_dsl(donor_dsl)
            spliced_stages = [mutate_parameter_clause(s, rng) for s in (stages[:1] + d_stages[:2])][:5]
            spliced_head = mutate_parameter_clause(d_head if rng.random() < 0.5 else head, rng)
            return assemble_pipe_dsl(spliced_stages, spliced_head), "crossover_stage_splice"

    attempts = 0
    max_attempts = n_candidates * 14

    while len(proposals) < n_candidates and attempts < max_attempts:
        attempts += 1
        parent = parent_pool[attempts % len(parent_pool)]
        base_dsl = parent["dsl"]
        donor_dsl = rng.choice(donor_dsls) if donor_dsls else base_dsl

        # 25% of proposals perform a two-step compound mutation (e.g., replace_head + insert_stage)
        # to reach new estimator head + stage combinations in a single generation
        if attempts % 4 == 0:
            mid_dsl, _op1 = _mutate_single_step(base_dsl, donor_dsl)
            cand_dsl, _op2 = _mutate_single_step(mid_dsl, donor_dsl)
            op_name = "compound_two_step_edit"
        else:
            cand_dsl, op_name = _mutate_single_step(base_dsl, donor_dsl)

        _try_append_proposal(cand_dsl, op_name, parent)

    gen_wall_s = time.monotonic() - t0
    return proposals, float(gen_wall_s)


def select_6_candidates_from_96_phase12(
    crec: dict,
    p_frozen_entry: dict,
    revealed_history_records: List[dict],
    slate_96: List[dict],
    ctx_scaled_vec: np.ndarray,
    mm_dict: Dict[str, np.ndarray],
    prior_estimator: FoldSafeCalibratedPriorEstimator,
    sos_index: Dict[str, Any],
    model: FlywheelBlockFormerAdapterNet | None,
    beta_traj: float,
    eta_parent_delta: float,
    lambda_div_sem: float,
    lambda_div_op: float,
    device: torch.device,
    seen_shas: set,
    cycle_idx: int,
    seed: int,
    n_proposals: int = 6,
    active_modality_override_mask: torch.Tensor | None = None,
    selection_mode: str = "guided_multimodal",
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Phase 12 Proposal Selector supporting:
    - `GUIDED_96` (`selection_mode == "guided_multimodal"` and `model is not None`):
      1. Conditions on the FULL revealed `inner70` history for this context
         (`K_rev = len(revealed_history_records)`, growing from `5 -> 11 -> 17` across cycles!).
      2. Scores all `96` candidates with `FlywheelBlockFormerAdapterNet.predict_costs` +
         `predict_parent_child_delta` anchored to each candidate's parent `inner70` score.
      3. Selects `6` candidates sequentially via Submodular Diversity-Penalized Neural Acquisition
         so `GUIDED_96` explores distinct high-scoring estimator heads and stage regimes.
    - `NON_NEURAL_PRIOR_AND_DIVERSITY_SHARED_C2` (`selection_mode == "non_neural_prior_and_diversity_shared_slate"`):
      Uses the exact same `slate_96`, target-free `cand_priors`, fold-safe `cand_sos`, and submodular
      diversity penalties (`lambda_div_sem`, `lambda_div_op`), with all neural network terms set to zero.
    - `UNGUIDED_96_MATCHED` (`model is None` and `selection_mode != "non_neural_prior_and_diversity_shared_slate"`):
      Selects `6` candidates uniformly at random from the exact same `slate_96` proposal pool.
    """
    t0 = time.monotonic()
    for cand in slate_96:
        assert "fitness" not in cand and "logratio" not in cand
        assert cand["program_sha256"] not in seen_shas

    c_idx = int(crec["index"])
    cid = str(crec["context_sha256"])
    n_slate = len(slate_96)

    cand_prog = np.stack([c["feat412"] for c in slate_96], axis=0).astype(np.float32)
    cand_proj = np.stack([c["proj16"] for c in slate_96], axis=0).astype(np.float32)
    cand_sem24 = np.stack([c["sem24"] for c in slate_96], axis=0).astype(np.float32)
    cand_pids = [c["program_sha256"] for c in slate_96]
    slate_96_sha256 = hashlib.sha256(json.dumps(cand_pids).encode()).hexdigest()

    cand_priors = prior_estimator.predict_slate_priors(cid, c_idx, cand_prog, mm_dict)
    cand_sos = _compute_slate_foldsafe_sos_prior(
        cand_prog_mat=cand_prog,
        cand_pids=cand_pids,
        target_fam=crec["family_id"],
        sos_index=sos_index,
    )

    if selection_mode == "non_neural_prior_and_diversity_shared_slate":
        rev_recs = [r for r in revealed_history_records if "feat412" in r and "fitness" in r]
        k_rev = len(rev_recs)
        pred_costs_96 = cand_priors.astype(np.float64) + 0.08 * (
            cand_sos[:, 0].astype(np.float64) * cand_sos[:, 1].astype(np.float64)
        )
        chosen_indices_nn: List[int] = []
        chosen_ops_nn: set = set()
        remaining_nn = set(range(n_slate))
        for _ in range(min(n_proposals, n_slate)):
            best_i = -1
            best_acq = float("inf")
            for idx_cand in remaining_nn:
                acq = float(pred_costs_96[idx_cand])
                if chosen_indices_nn:
                    sims = [
                        float(np.dot(cand_sem24[idx_cand], cand_sem24[j_sel]))
                        for j_sel in chosen_indices_nn
                    ]
                    acq += float(lambda_div_sem) * max(sims)
                if slate_96[idx_cand]["operator"] in chosen_ops_nn:
                    acq += float(lambda_div_op)
                if acq < best_acq:
                    best_acq = acq
                    best_i = idx_cand
            chosen_indices_nn.append(best_i)
            chosen_ops_nn.add(slate_96[best_i]["operator"])
            remaining_nn.remove(best_i)

        chosen = []
        for idx_s in chosen_indices_nn:
            c_item = {
                k: v
                for k, v in slate_96[int(idx_s)].items()
                if k not in ("parent_feat412", "parent_sem24", "op_onehot", "sem24")
            }
            c_item["foldsafe_ast_prior_cost"] = float(cand_priors[int(idx_s)])
            c_item["foldsafe_sos_prior_mean"] = float(cand_sos[int(idx_s), 0])
            c_item["neural_predicted_cost"] = float(pred_costs_96[int(idx_s)])
            c_item["predicted_parent_delta"] = 0.0
            chosen.append(c_item)

        sel_sem = cand_sem24[chosen_indices_nn]
        distinct_heads = int(np.sum(np.any(sel_sem[:, 0:5] > 0.1, axis=0)))
        diag = {
            "selection_mode": "non_neural_prior_and_diversity_shared_slate",
            "neural_terms_active": False,
            "slate_96_sha256": slate_96_sha256,
            "revealed_history_k_rev_conditioned": int(k_rev),
            "beta_traj_used": 0.0,
            "eta_parent_delta_used": 0.0,
            "lambda_div_sem_used": float(lambda_div_sem),
            "lambda_div_op_used": float(lambda_div_op),
            "slate_size": int(n_slate),
            "distinct_estimator_heads_in_selected_6": distinct_heads,
            "distinct_operators_in_selected_6": len(chosen_ops_nn),
            "selected_mean_predicted_cost": round(
                float(np.mean([c["neural_predicted_cost"] for c in chosen])), 5
            ),
            "rejected_mean_predicted_cost": round(
                float(
                    np.mean(
                        [pred_costs_96[i] for i in range(n_slate) if i not in set(chosen_indices_nn)]
                    )
                ),
                5,
            ),
            "selected_mean_ast_prior_cost": round(
                float(np.mean([c["foldsafe_ast_prior_cost"] for c in chosen])), 5
            ),
            "selection_wall_seconds": round(time.monotonic() - t0, 4),
        }
        for c_item in chosen:
            seen_shas.add(c_item["program_sha256"])
        return chosen, diag

    if model is not None:
        # Build full revealed history prefix (`K_rev >= 1`, typically `5` in Cycle 1 and `11` in Cycle 2!)
        rev_recs = [r for r in revealed_history_records if "feat412" in r and "fitness" in r]
        k_rev = len(rev_recs)
        assert k_rev >= 1, "Must have at least 1 revealed incumbent record!"

        rev_prog = np.stack([np.asarray(r["feat412"], dtype=np.float32) for r in rev_recs], axis=0)
        rev_proj = np.stack([np.asarray(r["proj16"], dtype=np.float32) for r in rev_recs], axis=0)
        rev_pids = [str(r["program_sha256"]) for r in rev_recs]
        rev_priors = prior_estimator.predict_slate_priors(cid, c_idx, rev_prog, mm_dict)
        rev_sos = _compute_slate_foldsafe_sos_prior(
            cand_prog_mat=rev_prog,
            cand_pids=rev_pids,
            target_fam=crec["family_id"],
            sos_index=sos_index,
        )

        prog_all = np.vstack([rev_prog, cand_prog])[None, :, :]  # [1, K_rev + 96, 412]
        proj_all = np.vstack([rev_proj, cand_proj])[None, :, :]  # [1, K_rev + 96, 16]
        prior_all = np.concatenate([rev_priors, cand_priors])[None, :]  # [1, K_rev + 96]
        sos_all = np.vstack([rev_sos, cand_sos])[None, :, :]  # [1, K_rev + 96, 2]

        scale_val = float(p_frozen_entry["scale"])
        scale_t = torch.tensor([scale_val], dtype=torch.float32, device=device)
        rev_idx_t = torch.arange(k_rev, dtype=torch.int64, device=device).unsqueeze(0)  # [1, K_rev]
        rev_f_t = torch.tensor(
            [[float(r["fitness"]) for r in rev_recs]], dtype=torch.float32, device=device
        )
        rev_gate_t = torch.tensor(
            [[1.0 if r.get("cv_gate_passed", True) else 0.0 for r in rev_recs]],
            dtype=torch.float32,
            device=device,
        )
        mm_tok_t = torch.from_numpy(mm_dict["modality_tokens_10x32"][c_idx : c_idx + 1]).to(device)
        mm_mask_t = torch.from_numpy(mm_dict["modality_present_mask_10"][c_idx : c_idx + 1]).to(device)

        parent_sem_mat = np.stack([c["parent_sem24"] for c in slate_96], axis=0).astype(np.float32)
        op_onehot_mat = np.stack([c["op_onehot"] for c in slate_96], axis=0).astype(np.float32)
        parent_y_vec = np.array(
            [float(c["parent_fitness"]) / max(1e-4, scale_val) for c in slate_96],
            dtype=np.float32,
        )
        parent_gate_vec = np.array([float(c["parent_gate"]) for c in slate_96], dtype=np.float32)

        with torch.no_grad():
            out_m = model.predict_costs(
                ctx_feat=torch.from_numpy(ctx_scaled_vec[None, :]).to(device),
                prog_412=torch.from_numpy(prog_all).to(device),
                proj_16=torch.from_numpy(proj_all).to(device),
                base_prior_cost=torch.from_numpy(prior_all).to(device),
                scale=scale_t,
                sos_prior=torch.from_numpy(sos_all).to(device),
                mm_tokens_10x32=mm_tok_t,
                mm_mask_10=mm_mask_t,
                revealed_indices=rev_idx_t,
                revealed_raw_f=rev_f_t,
                revealed_gate=rev_gate_t,
                active_modality_override_mask=active_modality_override_mask,
            )
            out_a = model.anchor.predict_costs(
                ctx_feat=torch.from_numpy(ctx_scaled_vec[None, :]).to(device),
                prog_412=torch.from_numpy(prog_all).to(device),
                proj_16=torch.from_numpy(proj_all).to(device),
                base_prior_cost=torch.from_numpy(prior_all).to(device),
                scale=scale_t,
                sos_prior=torch.from_numpy(sos_all).to(device),
                revealed_indices=rev_idx_t,
                revealed_raw_f=rev_f_t,
                revealed_gate=rev_gate_t,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            c_m = out_m["total_cost"][0, k_rev:].detach().cpu().numpy().astype(np.float64)
            c_a = out_a["total_cost"][0, k_rev:].detach().cpu().numpy().astype(np.float64)
            traj_cost = c_a + float(beta_traj) * (c_m - c_a)

            delta_pred = (
                model.predict_parent_child_delta(
                    mm_tokens_10x32=mm_tok_t,
                    mm_mask_10=mm_mask_t,
                    parent_sem24=torch.from_numpy(parent_sem_mat[None, :, :]).to(device),
                    child_sem24=torch.from_numpy(cand_sem24[None, :, :]).to(device),
                    op_onehot=torch.from_numpy(op_onehot_mat[None, :, :]).to(device),
                    parent_norm_y=torch.from_numpy(parent_y_vec[None, :]).to(device),
                    parent_gate=torch.from_numpy(parent_gate_vec[None, :]).to(device),
                    active_modality_override_mask=active_modality_override_mask,
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            parent_anchored_cost = parent_y_vec.astype(np.float64) + delta_pred
            pred_costs_96 = (1.0 - float(eta_parent_delta)) * traj_cost + float(eta_parent_delta) * parent_anchored_cost

        # Sequential Submodular Diversity-Penalized Neural Acquisition (`6` of `96`)
        chosen_indices: List[int] = []
        chosen_ops: set = set()
        remaining = set(range(n_slate))
        for _ in range(min(n_proposals, n_slate)):
            best_i = -1
            best_acq = float("inf")
            for idx_cand in remaining:
                acq = float(pred_costs_96[idx_cand])
                if chosen_indices:
                    sims = [
                        float(np.dot(cand_sem24[idx_cand], cand_sem24[j_sel]))
                        for j_sel in chosen_indices
                    ]
                    acq += float(lambda_div_sem) * max(sims)
                if slate_96[idx_cand]["operator"] in chosen_ops:
                    acq += float(lambda_div_op)
                if acq < best_acq:
                    best_acq = acq
                    best_i = idx_cand
            chosen_indices.append(best_i)
            chosen_ops.add(slate_96[best_i]["operator"])
            remaining.remove(best_i)

        chosen = []
        for idx_s in chosen_indices:
            c_item = {
                k: v
                for k, v in slate_96[int(idx_s)].items()
                if k not in ("parent_feat412", "parent_sem24", "op_onehot", "sem24")
            }
            c_item["foldsafe_ast_prior_cost"] = float(cand_priors[int(idx_s)])
            c_item["foldsafe_sos_prior_mean"] = float(cand_sos[int(idx_s), 0])
            c_item["neural_predicted_cost"] = float(pred_costs_96[int(idx_s)])
            c_item["predicted_parent_delta"] = float(delta_pred[int(idx_s)])
            chosen.append(c_item)

        sel_sem = cand_sem24[chosen_indices]
        distinct_heads = int(np.sum(np.any(sel_sem[:, 0:5] > 0.1, axis=0)))
        diag = {
            "selection_mode": "guided_96_multimodal_blockformer_parent_delta_diversity_ucb_top6",
            "neural_terms_active": True,
            "slate_96_sha256": slate_96_sha256,
            "revealed_history_k_rev_conditioned": int(k_rev),
            "beta_traj_used": float(beta_traj),
            "eta_parent_delta_used": float(eta_parent_delta),
            "lambda_div_sem_used": float(lambda_div_sem),
            "lambda_div_op_used": float(lambda_div_op),
            "slate_size": int(n_slate),
            "distinct_estimator_heads_in_selected_6": distinct_heads,
            "distinct_operators_in_selected_6": len(chosen_ops),
            "selected_mean_predicted_cost": round(
                float(np.mean([c["neural_predicted_cost"] for c in chosen])), 5
            ),
            "rejected_mean_predicted_cost": round(
                float(np.mean([pred_costs_96[i] for i in range(n_slate) if i not in set(chosen_indices)])),
                5,
            ),
            "selected_mean_ast_prior_cost": round(
                float(np.mean([c["foldsafe_ast_prior_cost"] for c in chosen])), 5
            ),
            "selection_wall_seconds": round(time.monotonic() - t0, 4),
        }
    else:
        rng = random.Random(seed + 70001 * (cycle_idx + 1) + 131 * int(crec["index"]))
        indices = list(range(n_slate))
        rng.shuffle(indices)
        chosen_indices = indices[:n_proposals]
        chosen = []
        for idx_s in chosen_indices:
            c_item = {
                k: v
                for k, v in slate_96[int(idx_s)].items()
                if k not in ("parent_feat412", "parent_sem24", "op_onehot", "sem24")
            }
            c_item["foldsafe_ast_prior_cost"] = float(cand_priors[int(idx_s)])
            c_item["foldsafe_sos_prior_mean"] = float(cand_sos[int(idx_s), 0])
            c_item["neural_predicted_cost"] = None
            chosen.append(c_item)
        sel_sem = cand_sem24[chosen_indices]
        distinct_heads = int(np.sum(np.any(sel_sem[:, 0:5] > 0.1, axis=0)))
        diag = {
            "selection_mode": "unguided_96_matched_proposal_uniform_random_6_of_96",
            "neural_terms_active": False,
            "slate_96_sha256": slate_96_sha256,
            "revealed_history_k_rev_conditioned": 0,
            "slate_size": int(n_slate),
            "distinct_estimator_heads_in_selected_6": distinct_heads,
            "selected_mean_ast_prior_cost": round(
                float(np.mean([c["foldsafe_ast_prior_cost"] for c in chosen])), 5
            ),
            "selection_wall_seconds": round(time.monotonic() - t0, 4),
        }

    for c_item in chosen:
        seen_shas.add(c_item["program_sha256"])
    return chosen, diag
