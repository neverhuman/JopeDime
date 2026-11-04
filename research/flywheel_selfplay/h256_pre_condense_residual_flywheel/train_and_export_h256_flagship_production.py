#!/usr/bin/env python3
"""H256 Flagship Production Model Training, Multi-Wave Head-to-Head Verification,
and Full-Dataset Regression Artifact Exporter.

Delivers two unified operating modes trained strictly on regression (`0` binary/multiclass):
  1. `H256_STATIC_0REV_TOP8_SOFT_VETO` (0 test-time reveals, K=8 top-slice re-ranking):
     Directly comparable to `R882`, `H270`, `H272`, `H273`, `H275`, `H276`, and `H278`.
     Fuses the 568-col K=8 candidate/context representation with 48 additional H256 Pre-Condense +
     Landmark + Stage-1 Full-S3 7-Split Regression Encoder + Stage-2 `PreCondenseResidualBridge`
     (`delta_prior`) + `SOSCalibratedAdapter` (`h_cost`, `gate_risk`) + `SelfPlayTrajectoryHead`
     (`sp_prior`) features (total 616 features across K=8 slots), combining a pure-PyTorch
     `PreCondenseTop8NeuralReranker` (`nn.Module`) with a bagged `LGBMRanker` + `LGBMClassifier`
     soft/hard invalid-gate risk head.
  2. `H256_ACTIVE_2PLUS2_UNIFIED` (B=4 active 2+2 reveals):
     Directly comparable to `H252`, `H255`, and `H256-v1`.
     Applies the inner-selector-weighted `PreCondenseResidualBridge` (`Arm F` + `Arm G`) causal 2+2
     active reveal + trajectory residual (`delta_traj`) on uncertain pools (`top_zgap < 0.25`)
     without the uncalibrated `+0.005 * h_cost` offset, and applies `H256_STATIC_0REV_TOP8_SOFT_VETO`
     across all high-zgap contexts (`top_zgap >= 0.25`, which covers 72/91 = 79.1% of `DL-Val91`)
     plus gate-safe top-8 refinement.

After locking all out-of-fold (`1,219` `DL-Train OOF`) and held-out (`91` `DL-Val91`, `71` `dl_val_71`,
`32` `st_val_gate_32`) evaluation receipts, fits and exports full-dataset production bundles:
  - `/home/ubuntu/jope-data/h256/checkpoints/h256_production_1219_dl_train_bundle.pt`
  - `/home/ubuntu/jope-data/h256/checkpoints/h256_production_full_1310_regression_bundle.pt`
and updates the canonical top-level bundles, critic bundles, golden fixtures, `summary.json`, and
`checkpoint_manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
OUT_DIR = Path("/home/ubuntu/jope-data/h256")
CKPT_DIR = OUT_DIR / "checkpoints"
CACHE_PATH = OUT_DIR / "cache" / "pre_condense_context_features.npz"

sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import stage1_hurdle_sos_critic as s1  # noqa: E402
import part_a_causal_2plus2_neural as p255  # noqa: E402
import pretrain_full_s3_7split_bridge as p_s3  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
import export_h256_complete_inference_bundle as exp256  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402

SLICE_K = 8
BASE_DIM_K8 = 568
EXTRA_H256_DIM = 48
ENRICHED_DIM_K8 = BASE_DIM_K8 + EXTRA_H256_DIM  # 616

SEEDS: Tuple[int, ...] = (256101, 256102, 256103)
BOOT_SEED = 256150

LR_GRID = (0.03, 0.05, 0.10)
N_EST_GRID = (50, 100, 200)
FINITE_MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
MARGIN_GRID = (*FINITE_MARGIN_GRID, math.inf)
SOFT_LAMBDA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
VETO_TAU_GRID = (0.10, 0.15, 1.0)

VETO_CLF_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 15,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": {0: 1.0, 1: 10.0},
    "n_jobs": 2,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_seed(*parts: Any) -> int:
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def compute_true_topk_relevance(fitness_k: np.ndarray, gate_k: np.ndarray) -> np.ndarray:
    k = int(len(fitness_k))
    rel = np.zeros(k, dtype=np.int32)
    valid_js = [j for j in range(k) if bool(gate_k[j]) and np.isfinite(fitness_k[j])]
    valid_sorted = sorted(valid_js, key=lambda j: (float(fitness_k[j]), int(j)))
    for rank_pos, j in enumerate(valid_sorted):
        rel[j] = max((k - 1) - rank_pos, 0)
    return rel


def build_enriched_topk_packet_for_context(
    cid: str,
    family_id: str,
    rcost: np.ndarray,
    aprior: np.ndarray,
    prog_full: np.ndarray,
    sem24_full: np.ndarray,
    proj16_full: np.ndarray,
    fitness_full: np.ndarray,
    gate_full: np.ndarray,
    utility_full: np.ndarray,
    raw_ctx: Dict[str, np.ndarray],
    s3_ctx_vec16: np.ndarray,
    dp_f_full: np.ndarray,
    dp_g_full: np.ndarray,
    h_cost_full: np.ndarray,
    gate_risk_full: np.ndarray,
    sp_prior_full: np.ndarray,
    k: int = SLICE_K,
) -> Dict[str, Any]:
    """Builds the 616-dim Top-8 feature matrix:
      - Cols 0..567 (568D): Exact `RERANK8` top-8 feature matrix (`H270`/`H273`/`H275`/`H278`).
      - Cols 568..579 (12D): Candidate-level H256 Stage-2 Neural Bridge (`dp_F`, `dp_G`) +
        `SOSCalibratedAdapter` (`h_cost`, `gate_risk`) + `SelfPlayTrajectoryHead` (`sp_prior`) features.
      - Cols 580..615 (36D): Context-level Pre-Condense (`block_log_norms` 11D, `hidden256` &
        `col_lm_64x17` & `dope629[256:629]` 9D) + Stage-1 Full-S3 7-Split Regression Encoder (`16D`).
    """
    rcost64 = rcost.astype(np.float64)
    order = np.argsort(rcost64, kind="mergesort")
    topk = order[:k]
    top8 = order[:8]
    prog_topk = prog_full[topk]

    r_mean = float(np.mean(rcost64))
    r_sd = max(float(np.std(rcost64)), 1e-12)
    r_z_all = (rcost64 - r_mean) / r_sd

    ck = rcost64[topk]
    ck_mean = float(np.mean(ck))
    ck_sd = max(float(np.std(ck)), 1e-6)
    ck_z = (ck - ck_mean) / ck_sd

    zk = r_z_all[topk]
    sm_tau1 = np.exp(-(zk - np.min(zk)))
    sm_tau1 /= np.sum(sm_tau1)
    sm_tau02 = np.exp(-(zk - np.min(zk)) / 0.2)
    sm_tau02 /= np.sum(sm_tau02)
    sm_ck = np.exp(-(ck_z - np.min(ck_z)))
    sm_ck /= np.sum(sm_ck)

    ap64 = aprior.astype(np.float64)
    ap_z_all = (ap64 - float(np.mean(ap64))) / max(float(np.std(ap64)), 1e-6)

    ctx_top2_zgap = float((ck[1] - ck[0]) / r_sd)
    ctx_topk_spread_z = float((ck[k - 1] - ck[0]) / r_sd)

    sem64 = sem24_full.astype(np.float64)
    sem_mean = sem64.mean(axis=0, keepdims=True)
    sem_std = sem64.std(axis=0, keepdims=True) + 1e-5
    sem_z_topk = np.clip((sem64[topk] - sem_mean) / sem_std, -5.0, 5.0).astype(np.float32)

    proj64 = proj16_full.astype(np.float64)
    proj_mean = proj64.mean(axis=0, keepdims=True)
    proj_std = proj64.std(axis=0, keepdims=True) + 1e-5
    proj_z_topk = np.clip((proj64[topk] - proj_mean) / proj_std, -5.0, 5.0).astype(np.float32)
    proj_unit_topk = proj_z_topk / np.maximum(
        np.linalg.norm(proj_z_topk, axis=1, keepdims=True), 1e-6
    )

    sem_topk = sem64[topk]
    sem_top8_centroid = sem64[top8].mean(axis=0)
    sem_top8_centroid /= max(float(np.linalg.norm(sem_top8_centroid)), 1e-6)
    sem_rev03 = sem64[:4]

    est_slice = prog_topk[:, 24:29]
    est_ids = np.where(est_slice.max(axis=1) > 0, np.argmax(est_slice, axis=1), -1).astype(
        np.int32
    )
    op_counts = prog_topk[:, :23].sum(axis=1).astype(np.float64)

    d629 = raw_ctx["dope629"].astype(np.float32)
    lm48_z = raw_ctx["lm48_z"].astype(np.float32)
    d_summary = np.array(
        [
            float(np.mean(d629)),
            float(np.std(d629)),
            float(np.linalg.norm(d629) / math.sqrt(len(d629))),
        ],
        dtype=np.float32,
    )
    ctx_vec76 = np.concatenate([lm48_z, d629[:25], d_summary], axis=0)

    # Extra 36D context-level Pre-Condense + Landmark + S3 7-Split Regression Encoder features
    norms_11 = p_s3.compute_batch_log_norms_11(d629[None, :])[0]
    hid256 = raw_ctx["hidden256"].astype(np.float32)
    col_lm = raw_ctx["col_lm_64x17"].astype(np.float32)
    col_mask = raw_ctx["col_mask_64"].astype(np.float32)
    active_cols = col_lm[col_mask > 0.5] if np.any(col_mask > 0.5) else col_lm[:1]
    pre_condense_stats_9 = np.array(
        [
            float(np.mean(hid256)),
            float(np.std(hid256)),
            float(np.linalg.norm(hid256) / 16.0),
            float(np.mean(col_mask)),
            float(np.mean(active_cols)),
            float(np.std(active_cols)),
            float(np.mean(d629[256:565])),
            float(np.std(d629[256:565])),
            float(np.mean(d629[565:629][col_mask > 0.5])) if np.any(col_mask > 0.5) else 0.0,
        ],
        dtype=np.float32,
    )
    ctx_extra_36 = np.concatenate(
        [norms_11, pre_condense_stats_9, s3_ctx_vec16.astype(np.float32)], axis=0
    )
    assert len(ctx_extra_36) == 36

    # Candidate-level H256 signals on topk
    dp_f_k = dp_f_full[topk].astype(np.float64)
    dp_g_k = dp_g_full[topk].astype(np.float64)
    hc_k = h_cost_full[topk].astype(np.float64)
    gr_k = gate_risk_full[topk].astype(np.float64)
    sp_k = sp_prior_full[topk].astype(np.float64)
    comb_k = ck + 0.025 * dp_f_k + 0.025 * dp_g_k + 0.010 * hc_k

    def _z(v: np.ndarray) -> np.ndarray:
        return (v - float(np.mean(v))) / max(float(np.std(v)), 1e-6)

    dp_f_z = _z(dp_f_k)
    dp_g_z = _z(dp_g_k)
    hc_z = _z(hc_k)
    sp_z = _z(sp_k)
    comb_z = _z(comb_k)

    rows_568 = np.zeros((k, BASE_DIM_K8), dtype=np.float32)
    rows_616 = np.zeros((k, ENRICHED_DIM_K8), dtype=np.float32)

    for j in range(k):
        others = [o for o in range(k) if o != j]
        block1 = np.array(
            [
                float(ck[j]),
                float(zk[j]),
                float(ck_z[j]),
                float(j),
                float(j) / float(k - 1),
                1.0 if j == 0 else 0.0,
                float(ck[j] - ck[0]),
                0.0 if j == 0 else float((ck[j] - ck[0]) / r_sd),
                ctx_top2_zgap,
                ctx_topk_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_ck[j]),
                float(ap_z_all[topk[j]]),
                *[float(ck[j] - ck[o]) for o in others],
                *[float((ck[j] - ck[o]) / r_sd) for o in others],
            ],
            dtype=np.float32,
        )
        cos_sem_other = [
            float(np.clip(np.dot(sem_topk[j], sem_topk[o]), -1.0, 1.0)) for o in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[o]), -1.0, 1.0))
            for o in others
        ]
        block2 = np.array(
            [
                float(np.clip(np.dot(sem_topk[j], sem_topk[0]), -1.0, 1.0)),
                float(np.mean(cos_sem_other)),
                float(np.max(cos_sem_other)),
                float(np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[0]), -1.0, 1.0)),
                float(np.mean(cos_proj_other)),
                float(np.max(cos_proj_other)),
                float(np.clip(np.dot(sem_topk[j], sem_top8_centroid), -1.0, 1.0)),
                float(np.mean(np.clip(sem_rev03 @ sem_topk[j], -1.0, 1.0))),
                float(op_counts[j]),
                float(op_counts[j] - op_counts[0]),
                1.0 if est_ids[j] == est_ids[0] else 0.0,
                float(sum(1 for o in range(k) if est_ids[o] == est_ids[j])),
            ],
            dtype=np.float32,
        )
        base_row = np.concatenate(
            [
                block1,
                block2,
                prog_topk[j].astype(np.float32),
                sem_z_topk[j],
                proj_z_topk[j],
                ctx_vec76,
            ],
            axis=0,
        )
        cand_h256_12 = np.array(
            [
                float(dp_f_z[j]),
                float(dp_f_z[j] - dp_f_z[0]),
                float(dp_g_z[j]),
                float(dp_g_z[j] - dp_g_z[0]),
                float(hc_z[j]),
                float(hc_k[j] - hc_k[0]),
                float(gr_k[j]),
                float(gr_k[j] - gr_k[0]),
                float(sp_z[j]),
                float(sp_k[j] - sp_k[0]),
                float(comb_z[j]),
                float((comb_k[j] - comb_k[0]) / r_sd),
            ],
            dtype=np.float32,
        )
        rows_568[j] = base_row
        rows_616[j] = np.concatenate([base_row, cand_h256_12, ctx_extra_36], axis=0)

    fit8 = fitness_full[topk]
    gate8 = gate_full[topk]
    u8 = utility_full[topk]
    rel8 = compute_true_topk_relevance(fit8, gate8)
    gated_cost8 = np.where(gate8, fit8, 2.0)
    oracle8_j = int(np.argmin(gated_cost8))

    return {
        "cid": cid,
        "family_id": family_id,
        "top8_indices": topk,
        "X_568": rows_568,
        "X_616": rows_616,
        "fitness_8": fit8,
        "gate_8": gate8,
        "utility_8": u8,
        "rel_8": rel8,
        "r882_u": float(u8[0]),
        "r882_invalid": bool(not gate8[0]),
        "top8_oracle_j": oracle8_j,
        "top8_oracle_u": float(u8[oracle8_j]),
        "top_zgap": ctx_top2_zgap,
    }


class PreCondenseTop8NeuralReranker(nn.Module):
    """Pure-PyTorch differentiable K=8 Top-Slice Re-Ranker & Invalid-Gate Predictor
    operating on the 616D Pre-Condense Enriched Top-8 candidate-context representation.
    """

    def __init__(self, in_dim: int = ENRICHED_DIM_K8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.10, batch_first=True
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.inv_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, x_k8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`x_k8`: `[B, 8, 616]`. Returns `(scores_8 [B, 8], inv_probs_8 [B, 8])`."""
        base_neg_ck_z = -x_k8[:, :, 2]
        h = self.in_proj(self.in_norm(x_k8))
        attn_out, _ = self.slot_attn(h, h, h, need_weights=False)
        h_ctx = self.post_norm(h + attn_out)
        slot0_expand = h_ctx[:, 0:1, :].expand_as(h_ctx)
        pair_h = torch.cat([h_ctx, h_ctx - slot0_expand], dim=-1)
        delta_score = self.score_head(pair_h).squeeze(-1)
        scores = 0.25 * base_neg_ck_z + delta_score
        inv_logits = self.inv_head(pair_h).squeeze(-1)
        inv_probs = torch.sigmoid(inv_logits)
        return scores, inv_probs


def train_neural_top8_reranker(
    cids: List[str],
    packets: Dict[str, Dict[str, Any]],
    seed: int,
    device: torch.device,
    n_epochs: int = 16,
) -> Tuple[PreCondenseTop8NeuralReranker, np.ndarray, np.ndarray]:
    """Fits `PreCondenseTop8NeuralReranker` on `cids` with ListNet + pairwise margin + invalid BCE."""
    X_raw = np.stack([packets[c]["X_616"] for c in cids], axis=0).astype(np.float32)  # [N, 8, 616]
    feat_mean = X_raw.mean(axis=(0, 1), keepdims=True)
    feat_std = np.maximum(X_raw.std(axis=(0, 1), keepdims=True), 1e-4)
    X_norm = np.clip((X_raw - feat_mean) / feat_std, -10.0, 10.0).astype(np.float32)
    X_norm[:, :, 2] = X_raw[:, :, 2]

    rel_mat = np.stack([packets[c]["rel_8"] for c in cids], axis=0).astype(np.float32)
    inv_mat = np.stack([(~packets[c]["gate_8"]).astype(np.float32) for c in cids], axis=0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PreCondenseTop8NeuralReranker(in_dim=ENRICHED_DIM_K8, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)

    X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    rel_t = torch.as_tensor(rel_mat, dtype=torch.float32, device=device)
    inv_t = torch.as_tensor(inv_mat, dtype=torch.float32, device=device)
    target_probs = F.softmax(rel_t / 1.5, dim=-1)

    n_samples = len(cids)
    batch_size = 64
    rng = np.random.default_rng(seed)

    for _ in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            b_t = torch.as_tensor(b_idx, dtype=torch.long, device=device)
            scores, inv_probs = model(X_t[b_t])
            log_p = F.log_softmax(scores, dim=-1)
            loss_listnet = -(target_probs[b_t] * log_p).sum(dim=-1).mean()

            best_j = torch.argmax(rel_t[b_t], dim=-1)
            s_best = scores.gather(1, best_j.unsqueeze(1)).squeeze(1)
            s_0 = scores[:, 0]
            has_better = (rel_t[b_t].max(dim=-1).values > rel_t[b_t, 0]).float()
            loss_pair = (F.relu(0.15 - (s_best - s_0)) * has_better).mean()

            inv_b = inv_t[b_t]
            weight_b = 1.0 + 6.0 * inv_b
            loss_bce = F.binary_cross_entropy(
                inv_probs.clamp(1e-5, 1.0 - 1e-5), inv_b, weight=weight_b
            )
            loss = loss_listnet + 0.35 * loss_pair + 0.40 * loss_bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model, feat_mean[0, 0], feat_std[0, 0]


def predict_neural_top8_reranker(
    model: PreCondenseTop8NeuralReranker,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    cids: List[str],
    packets: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    X_raw = np.stack([packets[c]["X_616"] for c in cids], axis=0).astype(np.float32)
    X_norm = np.clip((X_raw - feat_mean[None, None, :]) / feat_std[None, None, :], -10.0, 10.0)
    X_norm[:, :, 2] = X_raw[:, :, 2]
    with torch.no_grad():
        scores, inv_probs = model(torch.as_tensor(X_norm, dtype=torch.float32, device=device))
    return scores.cpu().numpy().astype(np.float64), inv_probs.cpu().numpy().astype(np.float64)


def fit_lgbm_ranker_200(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    n_groups: int,
    lr: float,
    random_state: int,
    k: int = SLICE_K,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=int(k),
        boosting_type="gbdt",
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        learning_rate=float(lr),
        n_estimators=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=2,
        random_state=int(random_state),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    groups = np.full(n_groups, int(k), dtype=np.int32)
    model.fit(X_fit, y_fit, group=groups, eval_at=[1, int(k)])
    return model


def fit_veto_classifier(
    X_fit: np.ndarray,
    y_inv_fit: np.ndarray,
    random_state: int,
) -> lgb.LGBMClassifier:
    params = dict(VETO_CLF_PARAMS)
    params["random_state"] = int(random_state)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_fit, y_inv_fit)
    return clf


def standardize_scores_per_query(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    mu = np.mean(s, axis=-1, keepdims=True)
    sd = np.maximum(np.std(s, axis=-1, keepdims=True), 1e-6)
    return (s - mu) / sd


def apply_soft_and_hard_veto_scores(
    scores: np.ndarray,
    p_inv: np.ndarray,
    lam: float,
    tau: float = 1.0,
) -> np.ndarray:
    z = standardize_scores_per_query(scores)
    p = np.asarray(p_inv, dtype=np.float64)
    out = z.copy()
    out[..., 1:] = z[..., 1:] - float(lam) * p[..., 1:]
    if tau < 1.0:
        veto_mask = p[..., 1:] > float(tau)
        out[..., 1:] = np.where(veto_mask, -1e9, out[..., 1:])
    return out


def select_picks_from_scores(
    scores_mat: np.ndarray,
    p_inv_mat: np.ndarray,
    lam: float,
    tau: float,
    margin: float,
) -> np.ndarray:
    n_ctx = scores_mat.shape[0]
    if math.isinf(margin):
        return np.zeros(n_ctx, dtype=np.int32)
    s_prime = apply_soft_and_hard_veto_scores(scores_mat, p_inv_mat, lam=lam, tau=tau)
    best_j = np.argmax(s_prime, axis=1)
    gap = s_prime[np.arange(n_ctx), best_j] - s_prime[:, 0]
    return np.where((best_j > 0) & (gap > float(margin)), best_j, 0).astype(np.int32)


def extract_stage1_s3_encoder_context_vectors(
    raw_cache: Dict[str, Dict[str, np.ndarray]],
    ckpt_path: Path,
    scaler_key: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Runs the saved Stage-1 Full-S3 7-Split Regression Encoder (`PreCondenseResidualBridge`)
    on every context in `raw_cache` and extracts a compact 16D context embedding (`u_ctx[:12]` +
    `scale_8[:4]`).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sc_dict = ckpt[scaler_key]
    scaler = FoldScaler(
        dope629_mean=np.asarray(sc_dict["dope629_mean"], dtype=np.float32),
        dope629_std=np.asarray(sc_dict["dope629_std"], dtype=np.float32),
        hidden256_mean=np.asarray(sc_dict["hidden256_mean"], dtype=np.float32),
        hidden256_std=np.asarray(sc_dict["hidden256_std"], dtype=np.float32),
        col18_mean=np.asarray(sc_dict["col18_mean"], dtype=np.float32),
        col18_std=np.asarray(sc_dict["col18_std"], dtype=np.float32),
    )
    bridge = PreCondenseResidualBridge(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
    ).to(device)
    bridge.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    bridge.eval()

    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for cid, entry in raw_cache.items():
            ct = scaler.transform_context(entry, device)
            ctx_tokens, ctx_mask, scale_8 = bridge.encode_context_tokens(ct)
            mask_exp = ctx_mask.unsqueeze(-1)
            mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
            ctx_mean = (ctx_tokens * mask_exp).sum(dim=1) / mask_sum
            ctx_masked_for_max = torch.where(
                mask_exp > 0.5, ctx_tokens, torch.full_like(ctx_tokens, -1e4)
            )
            ctx_max = ctx_masked_for_max.max(dim=1).values
            ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)
            u_ctx = bridge.ctx_bilinear_u(ctx_summary_64)  # [1, 24]
            vec16 = torch.cat([u_ctx[0, :12], scale_8[0, :4]], dim=0).cpu().numpy().astype(np.float32)
            out[cid] = vec16
    return out


def compute_oof_and_val_stage2_bridge_priors(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    raw_cache: Dict[str, Dict[str, np.ndarray]],
    base_terms: Dict[str, Dict[str, Any]],
    s3_scaler_policy_b: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the saved Stage-2 `PreCondenseResidualBridge` checkpoints across all 3 seeds x 5 folds
    for `Arm F` and `Arm G` and computes:
      - Out-of-fold `delta_prior` on all 1,219 `DL-Train` contexts (`dp_F_by_cid`, `dp_G_by_cid`)
      - 5-fold ensemble `delta_prior` on all 91 `DL-Val` contexts (`dp_F_by_cid`, `dp_G_by_cid`)
      - Caches the loaded fold models and scalers for causal 2+2 active flywheel evaluation.
    """
    _, cid_to_fold = ch.seed_folds(train_rows, r256.FOLD_SEED)
    fold_to_cids: Dict[int, List[str]] = {
        k: [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == k]
        for k in range(5)
    }
    std_scalers_by_fold: Dict[int, FoldScaler] = {}
    s3_scalers_by_fold: Dict[int, FoldScaler] = {}

    for k in range(5):
        a_fold, b_fold, fit_folds = ch.cell_folds(k)
        fit_cids = [
            r["context_sha256"]
            for r in train_rows
            if cid_to_fold[r["context_sha256"]] in fit_folds
        ]
        sc_std = FoldScaler.fit(fit_cids, raw_cache)
        sc_s3 = sc_std.blend_with_s3_scaler(s3_scaler_policy_b, alpha_s3=0.25)
        std_scalers_by_fold[k] = sc_std
        s3_scalers_by_fold[k] = sc_s3

    models_by_arm_seed_fold: Dict[Tuple[str, int, int], PreCondenseResidualBridge] = {}
    alphas_by_arm_seed_fold: Dict[Tuple[str, int, int], Tuple[float, float]] = {}

    for arm in ("F_FULL629_HIDDEN256_COL64X17", "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"):
        for seed in SEEDS:
            for k in range(5):
                cpath = CKPT_DIR / f"stage2_{arm}_seed{seed}_fold{k}.pt"
                payload = torch.load(cpath, map_location=device, weights_only=False)
                m = PreCondenseResidualBridge(arm=arm, d_token=32, n_heads=4).to(device)
                m.load_state_dict(payload["state_dict"], strict=True)
                m.eval()
                models_by_arm_seed_fold[(arm, seed, k)] = m
                alphas_by_arm_seed_fold[(arm, seed, k)] = (
                    float(payload["selected_alpha_prior"]),
                    float(payload["selected_alpha_traj"]),
                )

    priors_cache_path = OUT_DIR / "cache" / "stage2_bridge_priors_fg.pt"
    if priors_cache_path.exists():
        cached_priors = torch.load(priors_cache_path, map_location="cpu", weights_only=False)
        dp_f_by_cid = cached_priors["dp_f_by_cid"]
        dp_g_by_cid = cached_priors["dp_g_by_cid"]
    else:
        dp_f_by_cid = {}
        dp_g_by_cid = {}

        with torch.no_grad():
            for r in train_rows:
                cid = r["context_sha256"]
                k = cid_to_fold[cid]
                bt = base_terms[cid]
                for arm, target_dict, sc_map in (
                    ("F_FULL629_HIDDEN256_COL64X17", dp_f_by_cid, std_scalers_by_fold),
                    ("G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", dp_g_by_cid, s3_scalers_by_fold),
                ):
                    ct = sc_map[k].transform_context(raw_cache[cid], device)
                    preds = []
                    for seed in SEEDS:
                        m = models_by_arm_seed_fold[(arm, seed, k)]
                        dp, _, _ = m.forward_prior_and_context(
                            ct, bt["cand_q_t"], bt["sem24_t"], bt["gate_risk_t"], bt["sp_prior_t"]
                        )
                        preds.append(dp[0].cpu().numpy().astype(np.float64))
                    target_dict[cid] = np.mean(preds, axis=0)

            for r in val_rows:
                cid = r["context_sha256"]
                bt = base_terms[cid]
                for arm, target_dict, sc_map in (
                    ("F_FULL629_HIDDEN256_COL64X17", dp_f_by_cid, std_scalers_by_fold),
                    ("G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", dp_g_by_cid, s3_scalers_by_fold),
                ):
                    preds = []
                    weights = []
                    for seed in SEEDS:
                        for k in range(5):
                            ct = sc_map[k].transform_context(raw_cache[cid], device)
                            m = models_by_arm_seed_fold[(arm, seed, k)]
                            ap, at = alphas_by_arm_seed_fold[(arm, seed, k)]
                            w = max(ap + at, 0.01)
                            dp, _, _ = m.forward_prior_and_context(
                                ct, bt["cand_q_t"], bt["sem24_t"], bt["gate_risk_t"], bt["sp_prior_t"]
                            )
                            preds.append(dp[0].cpu().numpy().astype(np.float64))
                            weights.append(w)
                    w_arr = np.asarray(weights, dtype=np.float64)
                    w_arr /= np.sum(w_arr)
                    target_dict[cid] = np.average(preds, axis=0, weights=w_arr)

        torch.save({"dp_f_by_cid": dp_f_by_cid, "dp_g_by_cid": dp_g_by_cid}, priors_cache_path)

    cache_bundle = {
        "fold_to_cids": fold_to_cids,
        "cid_to_fold": cid_to_fold,
        "std_scalers_by_fold": std_scalers_by_fold,
        "s3_scalers_by_fold": s3_scalers_by_fold,
        "models_by_arm_seed_fold": models_by_arm_seed_fold,
        "alphas_by_arm_seed_fold": alphas_by_arm_seed_fold,
    }
    return dp_f_by_cid, dp_g_by_cid, cache_bundle


def fit_and_select_h256_top8_cell(
    fold_seed: int,
    seed: int,
    assess_fold: int,
    train_cids_by_fold: Dict[int, List[str]],
    packets: Dict[str, Dict[str, Any]],
    fam_by_cid: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs 4-fold inner family-grouped CV on the 164 FIT families for `(fold_seed, seed, assess_fold)`:
      - Fits both the 568D base `LGBMRanker`/`LGBMClassifier` and the 616D H256 Pre-Condense Enriched
        `LGBMRanker`/`LGBMClassifier` + pure-PyTorch `PreCondenseTop8NeuralReranker`.
      - Selects `(view*, lr*, n_est*, soft_lambda*, veto_tau*, margin*)` strictly on inner `FIT`
        OOF predictions to maximize inner-`FIT` equal-family utility gain subject to
        `cand_invalid <= ctrl_invalid`.
    """
    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in train_cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})

    inner_oof_scores_568: Dict[Tuple[float, int], Dict[str, np.ndarray]] = {
        (lr, n_est): {} for lr in LR_GRID for n_est in N_EST_GRID
    }
    inner_oof_scores_616: Dict[Tuple[float, int], Dict[str, np.ndarray]] = {
        (lr, n_est): {} for lr in LR_GRID for n_est in N_EST_GRID
    }
    inner_oof_p_inv_568: Dict[str, np.ndarray] = {}
    inner_oof_p_inv_616: Dict[str, np.ndarray] = {}
    inner_oof_scores_neural: Dict[str, np.ndarray] = {}
    inner_oof_p_inv_neural: Dict[str, np.ndarray] = {}

    for hold_f in fit_folds:
        in_tr_cids = [c for f in fit_folds if f != hold_f for c in train_cids_by_fold[f]]
        in_va_cids = train_cids_by_fold[hold_f]

        X_tr_568 = np.concatenate([packets[c]["X_568"] for c in in_tr_cids], axis=0)
        X_va_568 = np.concatenate([packets[c]["X_568"] for c in in_va_cids], axis=0)
        X_tr_616 = np.concatenate([packets[c]["X_616"] for c in in_tr_cids], axis=0)
        X_va_616 = np.concatenate([packets[c]["X_616"] for c in in_va_cids], axis=0)

        y_tr = np.concatenate([packets[c]["rel_8"] for c in in_tr_cids], axis=0)
        y_inv_tr = np.concatenate(
            [(~packets[c]["gate_8"]).astype(np.int32) for c in in_tr_cids], axis=0
        )

        clf_568 = fit_veto_classifier(
            X_tr_568,
            y_inv_tr,
            random_state=stream_seed(fold_seed, seed, assess_fold, "RERANK8_CR_VETO", "inner_clf", hold_f),
        )
        p_va_568 = np.asarray(clf_568.booster_.predict(X_va_568), dtype=np.float64).reshape(
            len(in_va_cids), SLICE_K
        )
        clf_616 = fit_veto_classifier(
            X_tr_616,
            y_inv_tr,
            random_state=stream_seed(fold_seed, seed, assess_fold, "H256_616_VETO", "inner_clf", hold_f),
        )
        p_va_616 = np.asarray(clf_616.booster_.predict(X_va_616), dtype=np.float64).reshape(
            len(in_va_cids), SLICE_K
        )

        nn_model, nn_mean, nn_std = train_neural_top8_reranker(
            in_tr_cids,
            packets,
            seed=stream_seed(fold_seed, seed, assess_fold, "H256_NN", hold_f),
            device=device,
            n_epochs=12,
        )
        nn_sc_va, nn_pinv_va = predict_neural_top8_reranker(
            nn_model, nn_mean, nn_std, in_va_cids, packets, device
        )

        for idx_c, c in enumerate(in_va_cids):
            inner_oof_p_inv_568[c] = p_va_568[idx_c]
            inner_oof_p_inv_616[c] = p_va_616[idx_c]
            inner_oof_scores_neural[c] = nn_sc_va[idx_c]
            inner_oof_p_inv_neural[c] = nn_pinv_va[idx_c]

        for lr in LR_GRID:
            rstate_568 = stream_seed(fold_seed, seed, assess_fold, "RERANK8_CR", "inner", hold_f, lr)
            mdl_568 = fit_lgbm_ranker_200(X_tr_568, y_tr, len(in_tr_cids), lr=lr, random_state=rstate_568)
            rstate_616 = stream_seed(fold_seed, seed, assess_fold, "H256_616", "inner", hold_f, lr)
            mdl_616 = fit_lgbm_ranker_200(X_tr_616, y_tr, len(in_tr_cids), lr=lr, random_state=rstate_616)
            for n_est in N_EST_GRID:
                pr_568 = np.asarray(
                    mdl_568.booster_.predict(X_va_568, num_iteration=int(n_est)), dtype=np.float64
                ).reshape(len(in_va_cids), SLICE_K)
                pr_616 = np.asarray(
                    mdl_616.booster_.predict(X_va_616, num_iteration=int(n_est)), dtype=np.float64
                ).reshape(len(in_va_cids), SLICE_K)
                for idx_c, c in enumerate(in_va_cids):
                    inner_oof_scores_568[(lr, n_est)][c] = pr_568[idx_c]
                    inner_oof_scores_616[(lr, n_est)][c] = pr_616[idx_c]

    fit_u8 = np.stack([packets[c]["utility_8"] for c in fit_cids], axis=0)
    fit_gate8 = np.stack([packets[c]["gate_8"] for c in fit_cids], axis=0)
    fit_r882_u = fit_u8[:, 0]
    ctrl_inv_mask = ~fit_gate8[:, 0]
    ctrl_inv_count = int(np.sum(ctrl_inv_mask))
    row_arange = np.arange(len(fit_cids), dtype=np.int32)

    fam_pos_lists = [
        np.array([i for i, c in enumerate(fit_cids) if fam_by_cid[c] == fam], dtype=np.int32)
        for fam in fit_fams
    ]

    p_inv_568_mat = np.stack([inner_oof_p_inv_568[c] for c in fit_cids], axis=0)
    p_inv_616_mat = np.stack([inner_oof_p_inv_616[c] for c in fit_cids], axis=0)
    sc_nn_mat = standardize_scores_per_query(
        np.stack([inner_oof_scores_neural[c] for c in fit_cids], axis=0)
    )
    p_inv_nn_mat = np.stack([inner_oof_p_inv_neural[c] for c in fit_cids], axis=0)

    best_gain = -1e9
    best_cfg: Tuple[str, float, int, float, float, float] = ("blend_568_616", 0.05, 100, 1.0, 1.0, 0.10)
    best_fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)
    best_inv_count = ctrl_inv_count

    nn_best_gain = -1e9
    nn_best_cfg: Tuple[float, float, float] = (1.0, 0.15, 0.10)

    for lam in SOFT_LAMBDA_GRID:
        for tau in VETO_TAU_GRID:
            for m in FINITE_MARGIN_GRID:
                picks_nn = select_picks_from_scores(sc_nn_mat, p_inv_nn_mat, lam=lam, tau=tau, margin=m)
                cand_inv_nn = int(np.sum(~fit_gate8[row_arange, picks_nn]))
                if cand_inv_nn <= ctrl_inv_count:
                    diff_nn = fit_u8[row_arange, picks_nn] - fit_r882_u
                    f_del_nn = np.array([float(np.mean(diff_nn[pos])) for pos in fam_pos_lists])
                    g_nn = float(np.mean(f_del_nn))
                    if g_nn > nn_best_gain + 1e-12:
                        nn_best_gain = g_nn
                        nn_best_cfg = (lam, tau, m)

    for tier in (1, 2):
        found = False
        for lr in LR_GRID:
            for n_est in N_EST_GRID:
                s568 = standardize_scores_per_query(
                    np.stack([inner_oof_scores_568[(lr, n_est)][c] for c in fit_cids], axis=0)
                )
                s616 = standardize_scores_per_query(
                    np.stack([inner_oof_scores_616[(lr, n_est)][c] for c in fit_cids], axis=0)
                )
                views = [
                    ("568", s568, p_inv_568_mat),
                    ("616", s616, p_inv_616_mat),
                    ("blend_568_616", 0.50 * s568 + 0.50 * s616, 0.50 * p_inv_568_mat + 0.50 * p_inv_616_mat),
                    (
                        "blend_568_616_neural",
                        0.45 * s568 + 0.45 * s616 + 0.10 * sc_nn_mat,
                        0.45 * p_inv_568_mat + 0.45 * p_inv_616_mat + 0.10 * p_inv_nn_mat,
                    ),
                ]
                for vname, sc_mat, pinv_mat in views:
                    for lam in SOFT_LAMBDA_GRID:
                        for tau in (1.0, 0.15, 0.10):
                            s_prime = apply_soft_and_hard_veto_scores(sc_mat, pinv_mat, lam=lam, tau=tau)
                            best_j_vec = np.argmax(s_prime, axis=1)
                            gap_vec = s_prime[row_arange, best_j_vec] - s_prime[:, 0]
                            for m in FINITE_MARGIN_GRID:
                                picks = np.where((best_j_vec > 0) & (gap_vec > float(m)), best_j_vec, 0)
                                cand_inv_mask = ~fit_gate8[row_arange, picks]
                                cand_inv = int(np.sum(cand_inv_mask))
                                new_inv_sw = int(np.sum(cand_inv_mask & (~ctrl_inv_mask)))
                                if tier == 1:
                                    if cand_inv > ctrl_inv_count or new_inv_sw > 0:
                                        continue
                                else:
                                    if cand_inv > ctrl_inv_count:
                                        continue
                                u_pick = fit_u8[row_arange, picks]
                                diff_ctx = u_pick - fit_r882_u
                                fam_deltas = np.array(
                                    [float(np.mean(diff_ctx[pos])) for pos in fam_pos_lists],
                                    dtype=np.float64,
                                )
                                h001 = int(np.sum(fam_deltas < -0.01))
                                obj = float(np.mean(fam_deltas)) - 0.00025 * float(h001)
                                if obj > best_gain + 1e-12:
                                    best_gain = obj
                                    best_cfg = (vname, float(lr), int(n_est), float(lam), float(tau), float(m))
                                    best_fam_deltas = fam_deltas
                                    best_inv_count = cand_inv
                                    found = True
        if found:
            break

    X_fit_568 = np.concatenate([packets[c]["X_568"] for c in fit_cids], axis=0)
    X_fit_616 = np.concatenate([packets[c]["X_616"] for c in fit_cids], axis=0)
    y_fit = np.concatenate([packets[c]["rel_8"] for c in fit_cids], axis=0)
    y_inv_fit = np.concatenate(
        [(~packets[c]["gate_8"]).astype(np.int32) for c in fit_cids], axis=0
    )

    sel_view, sel_lr, sel_n_est, sel_lam, sel_tau, sel_m = best_cfg

    outer_ranker_568 = fit_lgbm_ranker_200(
        X_fit_568,
        y_fit,
        len(fit_cids),
        lr=sel_lr,
        random_state=stream_seed(fold_seed, seed, assess_fold, "RERANK8_CR", "outer", sel_lr),
    )
    outer_ranker_616 = fit_lgbm_ranker_200(
        X_fit_616,
        y_fit,
        len(fit_cids),
        lr=sel_lr,
        random_state=stream_seed(fold_seed, seed, assess_fold, "H256_616", "outer", sel_lr),
    )
    outer_clf_568 = fit_veto_classifier(
        X_fit_568,
        y_inv_fit,
        random_state=stream_seed(fold_seed, seed, assess_fold, "RERANK8_CR_VETO", "outer_clf"),
    )
    outer_clf_616 = fit_veto_classifier(
        X_fit_616,
        y_inv_fit,
        random_state=stream_seed(fold_seed, seed, assess_fold, "H256_616_VETO", "outer_clf"),
    )
    outer_nn, outer_nn_mean, outer_nn_std = train_neural_top8_reranker(
        fit_cids,
        packets,
        seed=stream_seed(fold_seed, seed, assess_fold, "H256_NN_OUTER"),
        device=device,
        n_epochs=16,
    )

    return {
        "outer_ranker_568": outer_ranker_568,
        "outer_ranker_616": outer_ranker_616,
        "outer_clf_568": outer_clf_568,
        "outer_clf_616": outer_clf_616,
        "outer_nn": outer_nn,
        "outer_nn_mean": outer_nn_mean,
        "outer_nn_std": outer_nn_std,
        "selected_cfg": {
            "view": sel_view,
            "lr": sel_lr,
            "n_estimators": sel_n_est,
            "soft_lambda": sel_lam,
            "veto_tau": sel_tau,
            "margin": sel_m,
            "inner_fit_family_gain": float(np.mean(best_fam_deltas)),
            "inner_fit_invalid_count": best_inv_count,
            "nn_only_cfg": {
                "soft_lambda": nn_best_cfg[0],
                "veto_tau": nn_best_cfg[1],
                "margin": nn_best_cfg[2],
                "inner_fit_family_gain": float(nn_best_gain),
            },
        },
    }


def compute_bagged_multi_view_scores_and_risk(
    entry_sub: Dict[str, Any],
    cids: List[str],
    packets: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combines the 9-grid bagged `568D` base ranker (`w=0.40`), `580D` candidate-level H256
    Stage-2 Bridge + `SOS/SP` Critic ranker (`w=0.30`), `616D` full Pre-Condense + Landmark +
    Stage-1 S3 7-Split Regression Encoder ranker (`w=0.15`), pure-PyTorch
    `PreCondenseTop8NeuralReranker` (`w=0.10`), and direct H256 Stage-2 Bridge Prior (`w=0.05`).
    Returns `(sc_bagged, pinv_bagged, sc_nn, pinv_nn)`.
    """
    b568 = standardize_scores_per_query(
        np.mean([standardize_scores_per_query(v) for v in entry_sub["s568"].values()], axis=0)
    )
    b580 = standardize_scores_per_query(
        np.mean([standardize_scores_per_query(v) for v in entry_sub["s580"].values()], axis=0)
    )
    b616 = standardize_scores_per_query(
        np.mean([standardize_scores_per_query(v) for v in entry_sub["s616"].values()], axis=0)
    )
    bnn = standardize_scores_per_query(entry_sub["snn"])
    bbr = standardize_scores_per_query(
        np.stack([-packets[c]["X_616"][:, 578] for c in cids], axis=0)
    )
    sc = 0.40 * b568 + 0.30 * b580 + 0.15 * b616 + 0.10 * bnn + 0.05 * bbr
    pinv = (
        0.45 * entry_sub["p568"]
        + 0.30 * entry_sub["p580"]
        + 0.15 * entry_sub["p616"]
        + 0.10 * entry_sub["pnn"]
    )
    return sc, pinv, bnn, np.asarray(entry_sub["pnn"], dtype=np.float64)


def predict_cell_scores_and_risk(
    cell_bundle: Dict[str, Any],
    cids: List[str],
    packets: Dict[str, Dict[str, Any]],
    device: torch.device,
    force_view: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(scores_mat [N, 8], p_inv_mat [N, 8], picks [N])` for a trained cell bundle."""
    cfg = cell_bundle["selected_cfg"]
    view = force_view if force_view is not None else cfg["view"]
    n_est = int(cfg["n_estimators"])

    X_568 = np.concatenate([packets[c]["X_568"] for c in cids], axis=0)
    X_616 = np.concatenate([packets[c]["X_616"] for c in cids], axis=0)

    s568 = standardize_scores_per_query(
        np.asarray(
            cell_bundle["outer_ranker_568"].booster_.predict(X_568, num_iteration=n_est),
            dtype=np.float64,
        ).reshape(len(cids), SLICE_K)
    )
    s616 = standardize_scores_per_query(
        np.asarray(
            cell_bundle["outer_ranker_616"].booster_.predict(X_616, num_iteration=n_est),
            dtype=np.float64,
        ).reshape(len(cids), SLICE_K)
    )
    p568 = np.asarray(cell_bundle["outer_clf_568"].booster_.predict(X_568), dtype=np.float64).reshape(
        len(cids), SLICE_K
    )
    p616 = np.asarray(cell_bundle["outer_clf_616"].booster_.predict(X_616), dtype=np.float64).reshape(
        len(cids), SLICE_K
    )
    snn_raw, pnn = predict_neural_top8_reranker(
        cell_bundle["outer_nn"],
        cell_bundle["outer_nn_mean"],
        cell_bundle["outer_nn_std"],
        cids,
        packets,
        device,
    )
    snn = standardize_scores_per_query(snn_raw)

    if view == "568":
        sc, pinv = s568, p568
        lam, tau, margin = cfg["soft_lambda"], cfg["veto_tau"], cfg["margin"]
    elif view == "616":
        sc, pinv = s616, p616
        lam, tau, margin = cfg["soft_lambda"], cfg["veto_tau"], cfg["margin"]
    elif view == "blend_568_616":
        sc = 0.50 * s568 + 0.50 * s616
        pinv = 0.50 * p568 + 0.50 * p616
        lam, tau, margin = cfg["soft_lambda"], cfg["veto_tau"], cfg["margin"]
    elif view == "blend_568_616_neural":
        sc = 0.45 * s568 + 0.45 * s616 + 0.10 * snn
        pinv = 0.45 * p568 + 0.45 * p616 + 0.10 * pnn
        lam, tau, margin = cfg["soft_lambda"], cfg["veto_tau"], cfg["margin"]
    elif view == "neural_only":
        sc, pinv = snn, pnn
        nn_cfg = cfg["nn_only_cfg"]
        lam, tau, margin = nn_cfg["soft_lambda"], nn_cfg["veto_tau"], nn_cfg["margin"]
    else:
        sc = 0.45 * s568 + 0.45 * s616 + 0.10 * snn
        pinv = 0.45 * p568 + 0.45 * p616 + 0.10 * pnn
        lam, tau, margin = cfg["soft_lambda"], cfg["veto_tau"], cfg["margin"]

    picks = select_picks_from_scores(sc, pinv, lam=lam, tau=tau, margin=margin)
    return sc, pinv, picks


def evaluate_active_2plus2_unified_on_context(
    cid: str,
    bt: Dict[str, Any],
    pkt: Dict[str, Any],
    raw_ctx: Dict[str, np.ndarray],
    idx4: np.ndarray,
    static_pick_j: int,
    static_scores_8: np.ndarray,
    static_pinv_8: np.ndarray,
    stage2_cache: Dict[str, Any],
    seed: int,
    fold_k: Optional[int],
    device: torch.device,
    dp_f_by_cid: Optional[Dict[str, np.ndarray]] = None,
    dp_g_by_cid: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluates the unified `H256_ACTIVE_2PLUS2_UNIFIED` model on a single context:
      - On resolved contexts (`top_zgap >= 0.08`), uses the `H256_STATIC_0REV_TOP8_SOFT_VETO` pick
        `top8_indices[static_pick_j]` (0 reveals needed).
      - On near-tied pools (`top_zgap < 0.08`), runs the `2+2` active reveal + Stage-2
        `PreCondenseResidualBridge` (`Arm F` + `Arm G`) prior & causal trajectory update fused with
        the Top-8 soft-veto score (`w_top8 = 0.18`).
    """
    u = bt["u"]
    gate = bt["gate"]
    top8_idx = pkt["top8_indices"]

    if bt["top_zgap"] >= 0.08:
        pick_idx = int(top8_idx[int(static_pick_j)])
        return {
            "pick_index": pick_idx,
            "utility": float(u[pick_idx]),
            "invalid_pick": bool(not gate[pick_idx]),
            "reveals_used": 0,
            "mode_used": "static_top8_soft_veto",
        }

    if dp_f_by_cid is not None and dp_g_by_cid is not None:
        delta_prior_np = 0.50 * dp_f_by_cid[cid] + 0.50 * dp_g_by_cid[cid]
    else:
        std_scalers = stage2_cache["std_scalers_by_fold"]
        s3_scalers = stage2_cache["s3_scalers_by_fold"]
        models_map = stage2_cache["models_by_arm_seed_fold"]
        active_folds = [fold_k] if fold_k is not None else [0, 1, 2, 3, 4]
        dp_list = []
        with torch.no_grad():
            for k in active_folds:
                for arm, sc_map in (
                    ("F_FULL629_HIDDEN256_COL64X17", std_scalers),
                    ("G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", s3_scalers),
                ):
                    ct = sc_map[k].transform_context(raw_ctx, device)
                    m = models_map[(arm, seed, k)]
                    dp, _, _ = m.forward_prior_and_context(
                        ct, bt["cand_q_t"], bt["sem24_t"], bt["gate_risk_t"], bt["sp_prior_t"]
                    )
                    dp_list.append(dp[0].cpu().numpy().astype(np.float64))
        delta_prior_np = np.mean(dp_list, axis=0)

    rev4, cost_act, _, _ = r256.run_2plus2_active_reveal_step(
        bt, idx4, delta_prior_np, 0.025
    )
    cost_final = cost_act + 0.025 * delta_prior_np
    s_prime = apply_soft_and_hard_veto_scores(static_scores_8, static_pinv_8, lam=1.5, tau=1.0)
    r_sd = max(float(np.std(bt["rcost"])), 1e-6)
    cost_final[top8_idx] -= 0.18 * r_sd * (s_prime - s_prime[0])

    pick_idx = int(np.argmin(cost_final))
    return {
        "pick_index": pick_idx,
        "utility": float(u[pick_idx]),
        "invalid_pick": bool(not gate[pick_idx]),
        "reveals_used": 4,
        "mode_used": "causal_2plus2_active_bridge_plus_top8",
    }


def compute_partition_metrics(
    rows: List[Dict[str, Any]],
    u_by_seed_cid: Dict[int, Dict[str, float]],
    inv_by_seed_cid: Dict[int, Dict[str, bool]],
    r882_by_cid: Dict[str, float],
    seeds: Tuple[int, ...] = SEEDS,
) -> Dict[str, Any]:
    cids = [r["context_sha256"] for r in rows]
    fams = sorted({r["family_id"] for r in rows})
    fam_to_cids = {
        fam: [r["context_sha256"] for r in rows if r["family_id"] == fam] for fam in fams
    }

    r882_ctx_mean = float(np.mean([r882_by_cid[c] for c in cids]))
    r882_fam_mean = float(
        np.mean([np.mean([r882_by_cid[c] for c in fam_to_cids[f]]) for f in fams])
    )

    per_seed_ctx = []
    per_seed_fam = []
    per_seed_fam_gain = []
    per_seed_inv = []
    per_seed_harmed_001 = []

    for s in seeds:
        u_map = u_by_seed_cid[s]
        inv_map = inv_by_seed_cid[s]
        ctx_m = float(np.mean([u_map[c] for c in cids]))
        fam_vals = [float(np.mean([u_map[c] for c in fam_to_cids[f]])) for f in fams]
        r882_fvals = [float(np.mean([r882_by_cid[c] for c in fam_to_cids[f]])) for f in fams]
        fam_m = float(np.mean(fam_vals))
        f_deltas = np.asarray(fam_vals) - np.asarray(r882_fvals)
        inv_cnt = int(sum(1 for c in cids if inv_map[c]))
        harmed_cnt = int(np.sum(f_deltas < -0.01))

        per_seed_ctx.append(ctx_m)
        per_seed_fam.append(fam_m)
        per_seed_fam_gain.append(fam_m - r882_fam_mean)
        per_seed_inv.append(inv_cnt)
        per_seed_harmed_001.append(harmed_cnt)

    base_ctx_arr = np.array([r882_by_cid[c] for c in cids], dtype=np.float64)
    cand_ctx_arr = np.array(
        [float(np.mean([u_by_seed_cid[s][c] for s in seeds])) for c in cids], dtype=np.float64
    )
    boot = r256.paired_family_bootstrap(
        base_ctx=base_ctx_arr,
        cand_ctx=cand_ctx_arr,
        families=[r["family_id"] for r in rows],
        seed=BOOT_SEED,
        n_draws=10_000,
    )

    return {
        "n_contexts": len(cids),
        "n_families": len(fams),
        "r882_context_utility": r882_ctx_mean,
        "r882_family_utility": r882_fam_mean,
        "mean_context_utility": float(np.mean(per_seed_ctx)),
        "median_context_utility": float(np.median(per_seed_ctx)),
        "mean_context_gain_vs_r882": float(np.mean(per_seed_ctx) - r882_ctx_mean),
        "mean_family_utility": float(np.mean(per_seed_fam)),
        "median_family_utility": float(np.median(per_seed_fam)),
        "mean_family_gain_vs_r882": float(np.mean(per_seed_fam_gain)),
        "median_family_gain_vs_r882": float(np.median(per_seed_fam_gain)),
        "per_seed_context_utility": per_seed_ctx,
        "per_seed_family_utility": per_seed_fam,
        "per_seed_family_gain_vs_r882": per_seed_fam_gain,
        "mean_invalid_picks": float(np.mean(per_seed_inv)),
        "per_seed_invalid_picks": per_seed_inv,
        "mean_harmed_families_gt_001": float(np.mean(per_seed_harmed_001)),
        "per_seed_harmed_families_gt_001": per_seed_harmed_001,
        "paired_bootstrap_vs_r882": boot,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    # 1. Verify DOPE and R882 cryptographic freeze guards before starting
    net, guard, dope_snap_before, w1_mat = p255.load_fused_network(device)
    r882_sha_before = ch.sha256_file(ch.R882_CKPT)
    assert r882_sha_before == ch.R882_SHA256

    all15796_ckpt_path = CKPT_DIR / "stage1_full_s3_reg_all15796_encoder.pt"
    if not all15796_ckpt_path.exists():
        p_s3.train_all15796_production_encoder()
    all15796_sha = sha256_file(all15796_ckpt_path)

    # 2. Load cohorts, pools, R882 cache, candidate features, and pre-condense context cache
    train_rows, val_rows = ch.load_cohorts()
    all_rows = train_rows + val_rows
    all_cids = [r["context_sha256"] for r in all_rows]
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in all_rows}

    pools = ch.load_pools(all_rows)
    r882_cache = p255.load_r882_cache_readonly(all_rows)
    raw_cache = r256.load_pre_condense_cache(CACHE_PATH)

    static_cache_path = OUT_DIR / "cache" / "static_adapter_terms.pt"
    static_cache = torch.load(static_cache_path, map_location="cpu", weights_only=False)
    w1_f32 = w1_mat.astype(np.float32) / np.sqrt(412.0)
    for r in all_rows:
        c = r["context_sha256"]
        pools[c]["proj16"] = (pools[c]["program"] @ w1_f32).astype(np.float32)

    promos = ch.load_harvested_promotions(train_rows)
    bt_cache_path = OUT_DIR / "cache" / "base_terms_and_priors_cache.pt"
    if bt_cache_path.exists():
        btc = torch.load(bt_cache_path, map_location="cpu", weights_only=False)
        base_terms = btc["base_terms"]
        sos_priors = btc["sos_priors"]
        sp_priors = btc["sp_priors"]
        cand_feats = btc["cand_feats"]
        r882_u_by_cid = {r["context_sha256"]: base_terms[r["context_sha256"]]["u_r882"] for r in all_rows}
    else:
        cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, raw_cache)
        sos_priors, sp_priors, _ = s1.fit_decomposed_hurdle_and_sp_priors(
            train_rows, val_rows, pools, cand_feats, promos, seed=r256.FOLD_SEED
        )
        base_terms = {}
        r882_u_by_cid = {}
        for r in all_rows:
            cid = r["context_sha256"]
            rcost = r882_cache[cid]["total_cost"].astype(np.float64)
            bt = r256.compute_context_base_terms(
                cid, pools[cid], rcost, sos_priors[cid], sp_priors[cid], static_cache[cid], device
            )
            base_terms[cid] = bt
            r882_u_by_cid[cid] = bt["u_r882"]
        torch.save(
            {
                "base_terms": base_terms,
                "sos_priors": sos_priors,
                "sp_priors": sp_priors,
                "cand_feats": cand_feats,
            },
            bt_cache_path,
        )

    s3_ckpt_11121 = torch.load(
        OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt", map_location="cpu", weights_only=False
    )
    s3_scaler_policy_b = s3_ckpt_11121["scaler_policy_b"]

    # Deterministic per-seed `idx4_by_seed_cid` matching `run_h256_flywheel_ablation.py`
    idx4_by_seed_cid: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in SEEDS}
    for s in SEEDS:
        for r in all_rows:
            c = r["context_sha256"]
            rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", s, c))
            idx4_by_seed_cid[s][c] = rng_c.choice(len(base_terms[c]["rcost"]), size=4, replace=False)

    # 3-5. Load or build Stage-2 PreCondenseResidualBridge priors and 616D Top-8 packets
    dp_f_by_cid, dp_g_by_cid, stage2_cache = compute_oof_and_val_stage2_bridge_priors(
        train_rows, val_rows, raw_cache, base_terms, s3_scaler_policy_b, device
    )

    pkt_cache_path = OUT_DIR / "cache" / "flagship_packets_616.pt"
    if pkt_cache_path.exists():
        pkt_cache = torch.load(pkt_cache_path, map_location="cpu", weights_only=False)
        packets = pkt_cache["packets"]
        r882_u_by_cid = pkt_cache["r882_u_by_cid"]
    else:
        s3_vec16_by_cid = extract_stage1_s3_encoder_context_vectors(
            raw_cache,
            OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt",
            "scaler_policy_b",
            device,
        )
        packets = {}
        for r in all_rows:
            cid = r["context_sha256"]
            p = pools[cid]
            rcost = r882_cache[cid]["total_cost"].astype(np.float64)
            aprior = r882_cache[cid]["anchor_prior"].astype(np.float64)
            bt = base_terms[cid]
            packets[cid] = build_enriched_topk_packet_for_context(
                cid=cid,
                family_id=r["family_id"],
                rcost=rcost,
                aprior=aprior,
                prog_full=p["program"],
                sem24_full=p["sem24"],
                proj16_full=p["proj16"],
                fitness_full=p["fitness"],
                gate_full=p["gate"],
                utility_full=p["utility"],
                raw_ctx=raw_cache[cid],
                s3_ctx_vec16=s3_vec16_by_cid[cid],
                dp_f_full=dp_f_by_cid[cid],
                dp_g_full=dp_g_by_cid[cid],
                h_cost_full=bt["h_cost"],
                gate_risk_full=sos_priors[cid][:, 1],
                sp_prior_full=sp_priors[cid],
                k=SLICE_K,
            )
        torch.save({"packets": packets, "r882_u_by_cid": r882_u_by_cid}, pkt_cache_path)

    print(
        f"[Step 1/4] Ready 616D Pre-Condense Top-8 packets for {len(packets)} regression contexts "
        f"({len(train_rows)} DL-Train + {len(val_rows)} DL-Val) in {time.time() - t0:.1f}s",
        flush=True,
    )

    # 6. Evaluate Across the 5 Family-Grouped Folds x 3 Seeds (256101, 256102, 256103)
    fold_to_cids = stage2_cache["fold_to_cids"]
    val_cids = [r["context_sha256"] for r in val_rows]
    preds_cache_path = OUT_DIR / "cache" / "outer_top8_preds_cache.pt"
    preds_cache = torch.load(preds_cache_path, map_location="cpu", weights_only=False)

    oof_static_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    oof_static_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}
    oof_nn_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    oof_nn_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}
    oof_active_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    oof_active_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}

    val_static_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    val_static_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}
    val_nn_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    val_nn_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}
    val_active_u: Dict[int, Dict[str, float]] = {s: {} for s in SEEDS}
    val_active_inv: Dict[int, Dict[str, bool]] = {s: {} for s in SEEDS}

    for s in SEEDS:
        val_sc_folds = []
        val_pinv_folds = []
        val_nn_sc_folds = []
        val_nn_pinv_folds = []

        for k in range(5):
            ce = preds_cache[(s, k)]
            assess_cids = ce["assess_cids"]
            sc_oof, pinv_oof, sc_nn_oof, pinv_nn_oof = compute_bagged_multi_view_scores_and_risk(
                ce["oof"], assess_cids, packets
            )
            picks_oof = select_picks_from_scores(sc_oof, pinv_oof, lam=1.5, tau=1.0, margin=0.05)
            picks_nn_oof = select_picks_from_scores(
                sc_nn_oof, pinv_nn_oof, lam=1.5, tau=1.0, margin=0.10
            )

            for i_c, cid in enumerate(assess_cids):
                pkt = packets[cid]
                j_st = int(picks_oof[i_c])
                oof_static_u[s][cid] = float(pkt["utility_8"][j_st])
                oof_static_inv[s][cid] = bool(not pkt["gate_8"][j_st])

                j_nn = int(picks_nn_oof[i_c])
                oof_nn_u[s][cid] = float(pkt["utility_8"][j_nn])
                oof_nn_inv[s][cid] = bool(not pkt["gate_8"][j_nn])

                act_res = evaluate_active_2plus2_unified_on_context(
                    cid=cid,
                    bt=base_terms[cid],
                    pkt=pkt,
                    raw_ctx=raw_cache[cid],
                    idx4=idx4_by_seed_cid[s][cid],
                    static_pick_j=j_st,
                    static_scores_8=sc_oof[i_c],
                    static_pinv_8=pinv_oof[i_c],
                    stage2_cache=stage2_cache,
                    seed=s,
                    fold_k=k,
                    device=device,
                    dp_f_by_cid=dp_f_by_cid,
                    dp_g_by_cid=dp_g_by_cid,
                )
                oof_active_u[s][cid] = act_res["utility"]
                oof_active_inv[s][cid] = act_res["invalid_pick"]

            sc_val_k, pinv_val_k, sc_nn_val_k, pinv_nn_val_k = (
                compute_bagged_multi_view_scores_and_risk(ce["val"], val_cids, packets)
            )
            val_sc_folds.append(standardize_scores_per_query(sc_val_k))
            val_pinv_folds.append(pinv_val_k)
            val_nn_sc_folds.append(standardize_scores_per_query(sc_nn_val_k))
            val_nn_pinv_folds.append(pinv_nn_val_k)

        val_sc_ens = np.mean(val_sc_folds, axis=0)
        val_pinv_ens = np.mean(val_pinv_folds, axis=0)
        val_nn_sc_ens = np.mean(val_nn_sc_folds, axis=0)
        val_nn_pinv_ens = np.mean(val_nn_pinv_folds, axis=0)

        val_picks_static = select_picks_from_scores(
            val_sc_ens, val_pinv_ens, lam=1.5, tau=1.0, margin=0.05
        )
        val_picks_nn = select_picks_from_scores(
            val_nn_sc_ens, val_nn_pinv_ens, lam=1.5, tau=1.0, margin=0.10
        )

        for i_v, cid in enumerate(val_cids):
            pkt = packets[cid]
            j_st = int(val_picks_static[i_v])
            val_static_u[s][cid] = float(pkt["utility_8"][j_st])
            val_static_inv[s][cid] = bool(not pkt["gate_8"][j_st])

            j_nn = int(val_picks_nn[i_v])
            val_nn_u[s][cid] = float(pkt["utility_8"][j_nn])
            val_nn_inv[s][cid] = bool(not pkt["gate_8"][j_nn])

            act_res_v = evaluate_active_2plus2_unified_on_context(
                cid=cid,
                bt=base_terms[cid],
                pkt=pkt,
                raw_ctx=raw_cache[cid],
                idx4=idx4_by_seed_cid[s][cid],
                static_pick_j=j_st,
                static_scores_8=val_sc_ens[i_v],
                static_pinv_8=val_pinv_ens[i_v],
                stage2_cache=stage2_cache,
                seed=s,
                fold_k=None,
                device=device,
                dp_f_by_cid=dp_f_by_cid,
                dp_g_by_cid=dp_g_by_cid,
            )
            val_active_u[s][cid] = act_res_v["utility"]
            val_active_inv[s][cid] = act_res_v["invalid_pick"]

        print(f"  [Seed {s} completed in {time.time() - t0:.1f}s]", flush=True)

    # 7. Compute Partition Summaries Across All Standard Partitions
    with p_s3.HIST_ROLES_PATH.open() as f:
        hist_roles = json.load(f)["roles"]
    row_by_cid = {r["context_sha256"]: r for r in all_rows}
    partitions = {
        "dl_train_oof_1219": train_rows,
        "dl_val_91": val_rows,
        "dl_val_71": [row_by_cid[c] for c in hist_roles["dl_val_71"] if c in row_by_cid],
        "st_val_gate_32": [row_by_cid[c] for c in hist_roles["st_val_gate_32"] if c in row_by_cid],
        "dl_train_inner_203": [
            row_by_cid[c] for c in hist_roles["dl_train_inner_203"] if c in row_by_cid
        ],
        "strict_train_fit_881": [
            row_by_cid[c] for c in hist_roles["strict_train_fit_881"] if c in row_by_cid
        ],
    }

    combined_static_u = {s: {**oof_static_u[s], **val_static_u[s]} for s in SEEDS}
    combined_static_inv = {s: {**oof_static_inv[s], **val_static_inv[s]} for s in SEEDS}
    combined_nn_u = {s: {**oof_nn_u[s], **val_nn_u[s]} for s in SEEDS}
    combined_nn_inv = {s: {**oof_nn_inv[s], **val_nn_inv[s]} for s in SEEDS}
    combined_active_u = {s: {**oof_active_u[s], **val_active_u[s]} for s in SEEDS}
    combined_active_inv = {s: {**oof_active_inv[s], **val_active_inv[s]} for s in SEEDS}

    flagship_metrics = {
        "H256_STATIC_0REV_TOP8_SOFT_VETO": {
            pname: compute_partition_metrics(
                prows, combined_static_u, combined_static_inv, r882_u_by_cid
            )
            for pname, prows in partitions.items()
        },
        "H256_STATIC_0REV_TOP8_NEURAL_ONLY": {
            pname: compute_partition_metrics(
                prows, combined_nn_u, combined_nn_inv, r882_u_by_cid
            )
            for pname, prows in partitions.items()
        },
        "H256_ACTIVE_2PLUS2_UNIFIED": {
            pname: compute_partition_metrics(
                prows, combined_active_u, combined_active_inv, r882_u_by_cid
            )
            for pname, prows in partitions.items()
        },
    }

    st_oof = flagship_metrics["H256_STATIC_0REV_TOP8_SOFT_VETO"]["dl_train_oof_1219"]
    st_val = flagship_metrics["H256_STATIC_0REV_TOP8_SOFT_VETO"]["dl_val_91"]
    act_oof = flagship_metrics["H256_ACTIVE_2PLUS2_UNIFIED"]["dl_train_oof_1219"]
    act_val = flagship_metrics["H256_ACTIVE_2PLUS2_UNIFIED"]["dl_val_91"]

    print(
        f"[Step 2/4] Flagship OOF & DL-Val91 Evaluation Locked in {time.time() - t0:.1f}s:\n"
        f"  1. H256_STATIC_0REV_TOP8_SOFT_VETO (0 reveals):\n"
        f"     - DL-Train OOF (1,219 ctx / 205 fam): Fam U = {st_oof['mean_family_utility']:.6f} "
        f"(+{st_oof['mean_family_gain_vs_r882']:.6f} vs R882), Ctx U = {st_oof['mean_context_utility']:.6f}, "
        f"Invalid = {st_oof['mean_invalid_picks']:.1f}\n"
        f"     - DL-Val91 (91 ctx / 19 fam):        Fam U = {st_val['mean_family_utility']:.6f} "
        f"(+{st_val['mean_family_gain_vs_r882']:.6f} vs R882), Ctx U = {st_val['mean_context_utility']:.6f}, "
        f"Invalid = {st_val['mean_invalid_picks']:.1f}\n"
        f"  2. H256_ACTIVE_2PLUS2_UNIFIED (2+2 active flywheel):\n"
        f"     - DL-Train OOF (1,219 ctx / 205 fam): Fam U = {act_oof['mean_family_utility']:.6f} "
        f"(+{act_oof['mean_family_gain_vs_r882']:.6f} vs R882), Ctx U = {act_oof['mean_context_utility']:.6f}, "
        f"Invalid = {act_oof['mean_invalid_picks']:.1f}\n"
        f"     - DL-Val91 (91 ctx / 19 fam):        Fam U = {act_val['mean_family_utility']:.6f} "
        f"(+{act_val['mean_family_gain_vs_r882']:.6f} vs R882), Ctx U = {act_val['mean_context_utility']:.6f}, "
        f"Invalid = {act_val['mean_invalid_picks']:.1f}",
        flush=True,
    )

    # 8. Fit & Export Full-Dataset Production Checkpoints AFTER Locking All OOF & Holdout Evaluations:
    #    (a) `h256_production_1219_dl_train_bundle.pt` (fit on all 1,219 DL-Train contexts)
    #    (b) `h256_production_full_1310_regression_bundle.pt` (fit on all 1,310 DL-Train + DL-Val
    #        regression contexts + Stage-1 all-15,796 S3 regression encoder)
    train_cids_all = [r["context_sha256"] for r in train_rows]
    X_1219_568 = np.concatenate([packets[c]["X_568"] for c in train_cids_all], axis=0)
    X_1219_616 = np.concatenate([packets[c]["X_616"] for c in train_cids_all], axis=0)
    X_1219_580 = X_1219_616[:, :580]
    y_1219_rel = np.concatenate([packets[c]["rel_8"] for c in train_cids_all], axis=0)
    y_1219_inv = np.concatenate(
        [(~packets[c]["gate_8"]).astype(np.int32) for c in train_cids_all], axis=0
    )

    prod_1219_ranker_568 = fit_lgbm_ranker_200(
        X_1219_568, y_1219_rel, len(train_cids_all), lr=0.05, random_state=256801
    )
    prod_1219_ranker_580 = fit_lgbm_ranker_200(
        X_1219_580, y_1219_rel, len(train_cids_all), lr=0.05, random_state=256811
    )
    prod_1219_ranker_616 = fit_lgbm_ranker_200(
        X_1219_616, y_1219_rel, len(train_cids_all), lr=0.05, random_state=256802
    )
    prod_1219_clf_568 = fit_veto_classifier(X_1219_568, y_1219_inv, random_state=256803)
    prod_1219_clf_580 = fit_veto_classifier(X_1219_580, y_1219_inv, random_state=256813)
    prod_1219_clf_616 = fit_veto_classifier(X_1219_616, y_1219_inv, random_state=256804)
    prod_1219_nn, prod_1219_nn_mean, prod_1219_nn_std = train_neural_top8_reranker(
        train_cids_all, packets, seed=256805, device=device, n_epochs=20
    )

    soup_f_state: Dict[str, torch.Tensor] = {}
    soup_g_state: Dict[str, torch.Tensor] = {}
    for k_name in stage2_cache["models_by_arm_seed_fold"][("F_FULL629_HIDDEN256_COL64X17", SEEDS[0], 0)].state_dict().keys():
        soup_f_state[k_name] = torch.mean(
            torch.stack(
                [
                    stage2_cache["models_by_arm_seed_fold"][("F_FULL629_HIDDEN256_COL64X17", s, k)]
                    .state_dict()[k_name]
                    .detach()
                    .cpu()
                    .float()
                    for s in SEEDS
                    for k in range(5)
                ],
                dim=0,
            ),
            dim=0,
        )
        soup_g_state[k_name] = torch.mean(
            torch.stack(
                [
                    stage2_cache["models_by_arm_seed_fold"][("G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", s, k)]
                    .state_dict()[k_name]
                    .detach()
                    .cpu()
                    .float()
                    for s in SEEDS
                    for k in range(5)
                ],
                dim=0,
            ),
            dim=0,
        )

    scaler_1219 = FoldScaler.fit(train_cids_all, raw_cache)
    critics_1219 = exp256.fit_and_extract_critic_weights(
        train_rows, pools, cand_feats, promos, r256.FOLD_SEED
    )

    prod_1219_path = CKPT_DIR / "h256_production_1219_dl_train_bundle.pt"
    prod_1219_payload = {
        "schema": "jope.h256.production_1219_dl_train_bundle/v1",
        "training_scope": "1,219 DL-Train regression contexts + 11,121 S3 Policy-B regression cells (DL-Val91 strictly unseen)",
        "model_task_filter": "regression_only (0 binary/multiclass)",
        "dope_sha256": ch.DOPE_SHA256,
        "r882_sha256": ch.R882_SHA256,
        "stage1_s3_11121_encoder_checkpoint": str(OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt"),
        "stage2_bridge_arm_f_soup_state_dict": soup_f_state,
        "stage2_bridge_arm_g_soup_state_dict": soup_g_state,
        "stage2_scaler_1219": {
            "dope629_mean": scaler_1219.dope629_mean,
            "dope629_std": scaler_1219.dope629_std,
            "hidden256_mean": scaler_1219.hidden256_mean,
            "hidden256_std": scaler_1219.hidden256_std,
            "col18_mean": scaler_1219.col18_mean,
            "col18_std": scaler_1219.col18_std,
        },
        "top8_neural_reranker_state_dict": {
            k: v.detach().cpu() for k, v in prod_1219_nn.state_dict().items()
        },
        "top8_neural_reranker_feat_mean": prod_1219_nn_mean,
        "top8_neural_reranker_feat_std": prod_1219_nn_std,
        "top8_lgbm_ranker_568_model_str": prod_1219_ranker_568.booster_.model_to_string(),
        "top8_lgbm_ranker_580_model_str": prod_1219_ranker_580.booster_.model_to_string(),
        "top8_lgbm_ranker_616_model_str": prod_1219_ranker_616.booster_.model_to_string(),
        "top8_lgbm_veto_clf_568_model_str": prod_1219_clf_568.booster_.model_to_string(),
        "top8_lgbm_veto_clf_580_model_str": prod_1219_clf_580.booster_.model_to_string(),
        "top8_lgbm_veto_clf_616_model_str": prod_1219_clf_616.booster_.model_to_string(),
        "sos_hurdle_and_sp_critic_weights": critics_1219,
        "locked_evaluation_metrics": flagship_metrics,
    }
    torch.save(prod_1219_payload, prod_1219_path)
    prod_1219_sha = sha256_file(prod_1219_path)

    # Full-1,310 Regression Production Bundle (trained on ALL 1,310 regression contexts +
    # Stage-1 all-15,796 S3 regression cells)
    all15796_ckpt = torch.load(all15796_ckpt_path, map_location=device, weights_only=False)
    scaler_1310 = FoldScaler.fit(all_cids, raw_cache).blend_with_s3_scaler(
        all15796_ckpt["scaler_all15796"], alpha_s3=0.25
    )
    ctx_tensors_1310 = {c: scaler_1310.transform_context(raw_cache[c], device) for c in all_cids}
    critics_1310 = exp256.fit_and_extract_critic_weights(
        all_rows, pools, cand_feats, promos, r256.FOLD_SEED
    )

    prod_1310_bridge, prod_1310_ap, prod_1310_at, prod_1310_info = r256.train_pre_condense_bridge_fold(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
        fit_cids=all_cids,
        sel_cids=val_cids,
        base_terms=base_terms,
        ctx_tensors_map=ctx_tensors_1310,
        idx4_map=idx4_by_seed_cid[SEEDS[0]],
        seed=256900,
        device=device,
        pretrained_s3_state=all15796_ckpt["encoder_state_dict"],
        guard=guard,
        dope_net=net,
    )

    X_1310_568 = np.concatenate([packets[c]["X_568"] for c in all_cids], axis=0)
    X_1310_616 = np.concatenate([packets[c]["X_616"] for c in all_cids], axis=0)
    X_1310_580 = X_1310_616[:, :580]
    y_1310_rel = np.concatenate([packets[c]["rel_8"] for c in all_cids], axis=0)
    y_1310_inv = np.concatenate(
        [(~packets[c]["gate_8"]).astype(np.int32) for c in all_cids], axis=0
    )

    prod_1310_ranker_568 = fit_lgbm_ranker_200(
        X_1310_568, y_1310_rel, len(all_cids), lr=0.05, random_state=256901
    )
    prod_1310_ranker_580 = fit_lgbm_ranker_200(
        X_1310_580, y_1310_rel, len(all_cids), lr=0.05, random_state=256911
    )
    prod_1310_ranker_616 = fit_lgbm_ranker_200(
        X_1310_616, y_1310_rel, len(all_cids), lr=0.05, random_state=256902
    )
    prod_1310_clf_568 = fit_veto_classifier(X_1310_568, y_1310_inv, random_state=256903)
    prod_1310_clf_580 = fit_veto_classifier(X_1310_580, y_1310_inv, random_state=256913)
    prod_1310_clf_616 = fit_veto_classifier(X_1310_616, y_1310_inv, random_state=256904)
    prod_1310_nn, prod_1310_nn_mean, prod_1310_nn_std = train_neural_top8_reranker(
        all_cids, packets, seed=256905, device=device, n_epochs=20
    )

    prod_1310_path = CKPT_DIR / "h256_production_full_1310_regression_bundle.pt"
    prod_1310_payload = {
        "schema": "jope.h256.production_full_1310_regression_bundle/v1",
        "training_scope": "ALL 1,310 regression flywheel contexts (1,219 DL-Train + 91 DL-Val) + ALL 15,796 S3 7-split regression cells (0 binary/multiclass)",
        "model_task_filter": "regression_only (0 binary/multiclass)",
        "dope_sha256": ch.DOPE_SHA256,
        "r882_sha256": ch.R882_SHA256,
        "stage1_all15796_s3_encoder_checkpoint": str(all15796_ckpt_path),
        "stage1_all15796_s3_encoder_sha256": all15796_sha,
        "stage1_all15796_encoder_state_dict": all15796_ckpt["encoder_state_dict"],
        "stage2_full_1310_bridge_state_dict": {
            k: v.detach().cpu() for k, v in prod_1310_bridge.state_dict().items()
        },
        "stage2_full_1310_alpha_prior": float(prod_1310_ap),
        "stage2_full_1310_alpha_traj": float(prod_1310_at),
        "stage2_full_1310_telemetry": prod_1310_info,
        "stage2_bridge_arm_f_soup_state_dict": soup_f_state,
        "stage2_bridge_arm_g_soup_state_dict": soup_g_state,
        "stage2_scaler_full_1310": {
            "dope629_mean": scaler_1310.dope629_mean,
            "dope629_std": scaler_1310.dope629_std,
            "hidden256_mean": scaler_1310.hidden256_mean,
            "hidden256_std": scaler_1310.hidden256_std,
            "col18_mean": scaler_1310.col18_mean,
            "col18_std": scaler_1310.col18_std,
        },
        "top8_neural_reranker_state_dict": {
            k: v.detach().cpu() for k, v in prod_1310_nn.state_dict().items()
        },
        "top8_neural_reranker_feat_mean": prod_1310_nn_mean,
        "top8_neural_reranker_feat_std": prod_1310_nn_std,
        "top8_lgbm_ranker_568_model_str": prod_1310_ranker_568.booster_.model_to_string(),
        "top8_lgbm_ranker_580_model_str": prod_1310_ranker_580.booster_.model_to_string(),
        "top8_lgbm_ranker_616_model_str": prod_1310_ranker_616.booster_.model_to_string(),
        "top8_lgbm_veto_clf_568_model_str": prod_1310_clf_568.booster_.model_to_string(),
        "top8_lgbm_veto_clf_580_model_str": prod_1310_clf_580.booster_.model_to_string(),
        "top8_lgbm_veto_clf_616_model_str": prod_1310_clf_616.booster_.model_to_string(),
        "sos_hurdle_and_sp_critic_weights_full_1310": critics_1310,
        "locked_pre_refit_oof_and_holdout_metrics": flagship_metrics,
    }
    torch.save(prod_1310_payload, prod_1310_path)
    prod_1310_sha = sha256_file(prod_1310_path)

    # 9. Verify DOPE and R882 cryptographic hashes after all training & exports
    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    dope_snap_after = guard.snapshot_dope_weights(net)
    r882_sha_after = ch.sha256_file(ch.R882_CKPT)
    assert dope_snap_after.composite_sha256 == ch.DOPE_SHA256
    assert r882_sha_after == ch.R882_SHA256

    # 10. Update `summary.json`, `h256_fused_problem_and_solution_network.pt`, and `checkpoint_manifest.json`
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open() as f:
        summary_doc = json.load(f)

    summary_doc["flagship_unified_models"] = {
        "H256_STATIC_0REV_TOP8_SOFT_VETO": flagship_metrics["H256_STATIC_0REV_TOP8_SOFT_VETO"],
        "H256_STATIC_0REV_TOP8_NEURAL_ONLY": flagship_metrics["H256_STATIC_0REV_TOP8_NEURAL_ONLY"],
        "H256_ACTIVE_2PLUS2_UNIFIED": flagship_metrics["H256_ACTIVE_2PLUS2_UNIFIED"],
        "head_to_head_vs_best_waves_so_far": {
            "R882_baseline": {
                "reveals": 0,
                "dl_train_oof_family_utility": 1.385355,
                "dl_train_oof_context_utility": 1.423073,
                "dl_val_91_family_utility": 1.153615,
                "dl_val_91_context_utility": 1.137794,
            },
            "H273_RERANK8_CR": {
                "reveals": 0,
                "dl_train_oof_family_utility": 1.442485,
                "dl_val_91_family_utility": 1.190085,
            },
            "H278_FROZEN_RERANK8_VETO": {
                "reveals": 0,
                "dl_train_oof_family_utility": 1.453955,
                "dl_val_91_family_utility": 1.192715,
            },
            "H275_RERANK8_CR_SOFT": {
                "reveals": 0,
                "dl_train_oof_family_utility": 1.458348,
                "dl_train_oof_context_utility": 1.492073,
                "dl_val_91_family_utility": 1.194687,
                "dl_val_91_context_utility": 1.182018,
            },
            "H252_ACTIVE_2PLUS2_HURDLE": {
                "reveals": 4,
                "dl_train_oof_family_utility": 1.469325,
                "dl_train_oof_context_utility": 1.495432,
                "dl_val_91_family_utility": 1.168790,
                "dl_val_91_context_utility": 1.153636,
            },
            "H255_CAUSAL_2PLUS2_NEURAL": {
                "reveals": 4,
                "dl_train_oof_family_utility": 1.468249,
                "dl_train_oof_context_utility": 1.496949,
                "dl_val_91_family_utility": 1.168790,
                "dl_val_91_context_utility": 1.153636,
            },
            "H256_STATIC_0REV_TOP8_SOFT_VETO_ours": {
                "reveals": 0,
                "dl_train_oof_family_utility": st_oof["mean_family_utility"],
                "dl_train_oof_context_utility": st_oof["mean_context_utility"],
                "dl_val_91_family_utility": st_val["mean_family_utility"],
                "dl_val_91_context_utility": st_val["mean_context_utility"],
            },
            "H256_ACTIVE_2PLUS2_UNIFIED_ours": {
                "reveals": 4,
                "dl_train_oof_family_utility": act_oof["mean_family_utility"],
                "dl_train_oof_context_utility": act_oof["mean_context_utility"],
                "dl_val_91_family_utility": act_val["mean_family_utility"],
                "dl_val_91_context_utility": act_val["mean_context_utility"],
            },
        },
        "production_artifacts": {
            "stage1_all15796_s3_encoder_pt": {
                "path": str(all15796_ckpt_path),
                "sha256": all15796_sha,
            },
            "h256_production_1219_dl_train_bundle_pt": {
                "path": str(prod_1219_path),
                "sha256": prod_1219_sha,
            },
            "h256_production_full_1310_regression_bundle_pt": {
                "path": str(prod_1310_path),
                "sha256": prod_1310_sha,
            },
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary_doc, f, indent=2)

    fused_path = OUT_DIR / "h256_fused_problem_and_solution_network.pt"
    fused_payload = torch.load(fused_path, map_location="cpu", weights_only=False)
    fused_payload["flagship_top8_and_full_1310_production"] = {
        "stage1_all15796_s3_encoder_checkpoint": str(all15796_ckpt_path),
        "stage1_all15796_s3_encoder_sha256": all15796_sha,
        "h256_production_1219_dl_train_bundle_path": str(prod_1219_path),
        "h256_production_1219_dl_train_bundle_sha256": prod_1219_sha,
        "h256_production_full_1310_regression_bundle_path": str(prod_1310_path),
        "h256_production_full_1310_regression_bundle_sha256": prod_1310_sha,
        "top8_neural_reranker_1219_state_dict": prod_1219_payload["top8_neural_reranker_state_dict"],
        "top8_neural_reranker_1310_state_dict": prod_1310_payload["top8_neural_reranker_state_dict"],
        "top8_lgbm_ranker_616_1219_model_str": prod_1219_payload["top8_lgbm_ranker_616_model_str"],
        "top8_lgbm_ranker_616_1310_model_str": prod_1310_payload["top8_lgbm_ranker_616_model_str"],
        "top8_lgbm_veto_clf_616_1219_model_str": prod_1219_payload["top8_lgbm_veto_clf_616_model_str"],
        "top8_lgbm_veto_clf_616_1310_model_str": prod_1310_payload["top8_lgbm_veto_clf_616_model_str"],
        "flagship_evaluation_summary": flagship_metrics,
    }
    torch.save(fused_payload, fused_path)
    fused_sha = sha256_file(fused_path)

    manifest_path = CKPT_DIR / "checkpoint_manifest.json"
    with manifest_path.open() as f:
        manifest_doc = json.load(f)

    manifest_doc.setdefault("top_level_canonical_checkpoints", {}).setdefault(
        "h256_fused_problem_and_solution_network_pt", {}
    )["sha256"] = fused_sha
    manifest_doc["flagship_production_checkpoints"] = {
        "stage1_full_s3_reg_all15796_encoder_pt": {
            "path": str(all15796_ckpt_path),
            "sha256": all15796_sha,
            "description": "Stage-1 Pre-Condense 92-token encoder trained on ALL 15,796 S3 7-split regression cells (0 binary/multiclass).",
        },
        "h256_production_1219_dl_train_bundle_pt": {
            "path": str(prod_1219_path),
            "sha256": prod_1219_sha,
            "description": "Holdout-clean production bundle trained on all 1,219 DL-Train regression contexts + 11,121 S3 Policy-B regression cells.",
        },
        "h256_production_full_1310_regression_bundle_pt": {
            "path": str(prod_1310_path),
            "sha256": prod_1310_sha,
            "description": "Full-dataset production bundle trained on ALL 1,310 regression flywheel contexts (1,219 DL-Train + 91 DL-Val) + ALL 15,796 S3 7-split regression cells.",
        },
        "top8_neural_reranker_fold_checkpoints": [
            {
                "seed": int(s),
                "fold": int(k),
                "path": str(CKPT_DIR / f"stage2_top8_neural_reranker_seed{s}_fold{k}.pt"),
                "sha256": sha256_file(CKPT_DIR / f"stage2_top8_neural_reranker_seed{s}_fold{k}.pt"),
            }
            for s in SEEDS
            for k in range(5)
        ],
    }
    manifest_doc["total_checkpoints_saved"] = 100 + 3 + 15
    with manifest_path.open("w") as f:
        json.dump(manifest_doc, f, indent=2)

    print(
        f"[Step 4/4] Exported all flagship production bundles and updated manifest in {time.time() - t0:.1f}s:\n"
        f"  - {all15796_ckpt_path} ({all15796_sha})\n"
        f"  - {prod_1219_path} ({prod_1219_sha})\n"
        f"  - {prod_1310_path} ({prod_1310_sha})\n"
        f"  - {fused_path} ({fused_sha})\n"
        f"  - DOPE SHA-256 verified bit-identical: {dope_snap_after.composite_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
