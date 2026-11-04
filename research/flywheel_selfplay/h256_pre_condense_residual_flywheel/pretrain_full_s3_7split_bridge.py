#!/usr/bin/env python3
"""H256 Stage 1: Full-S3 7-Split Regression-Only Pretraining of the Pre-Condense Context Token
Encoder (`Full629` 20x32 + 11D log-norms, `Hidden256` 8x32, `Col64x18` 64x32) with Weight-Locked DOPE.

Strict Guardrails:
1. 100% Regression-Only (`model_task == "regression"`): zero binary or multiclass cells admitted.
2. Trains across all 7 S3 splits (`fit-iid-65-v1`, `official-train-full-v1`, `outer5-iid-fold0..4-v1`)
   on Policy B Gate-Disjoint (`11,121` regression cells across `1,589` datasets) and Policy C
   Maximal-Disjoint (`11,086` regression cells across `1,584` datasets).
3. Evaluates out-of-sample on `eval_gradient_disjoint_v2_panel` (`4,389` disjoint regression cells
   across `627` datasets x 7 splits) and `dev_val_ablation` (`286` disjoint regression cells across
   `41` datasets x 7 splits), comparing the Pre-Condense (`Full629 + Hidden256 + Col64x18`) encoder
   against the frozen Stage-1B `10 x 32` (`7` active tokens) bottleneck across all 7 splits.
4. Enforces `dope_freeze_guard.py` (`SHA-256: c51a56d0...`) before, during every epoch, and after.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import extract_pre_condense_cache as ex  # noqa: E402
from pre_condense_models import (  # noqa: E402
    BLOCK_SLICES,
    FoldScaler,
    PreCondenseResidualBridge,
)

S3_MIRROR_ROOT = Path("/mnt/jope-archive/dope-high-encodings-20261025/v2-fixed629")
CORPUS_MANIFEST_PATH = Path("/mnt/jope-archive/flywheel-20261023/corpus/corpus_manifest.json")
HIST_ROLES_PATH = Path("/home/ubuntu/jope-data/hwv-06/sources/historical-role-identities.json")
SPLIT_RECON_PATH = Path(
    "/mnt/jope-archive/flywheel-20261023/results/h40_h45_cluster/h40_fast/h40_split_role_reconciliation_receipt_v2.json"
)
OUT_DIR = Path("/home/ubuntu/jope-data/h256")
OUT_CKPT = OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt"
OUT_RECEIPT = OUT_DIR / "full_s3_reg_7split_receipt.json"

SEVEN_SPLITS = (
    "fit-iid-65-v1",
    "official-train-full-v1",
    "outer5-iid-fold0-v1",
    "outer5-iid-fold1-v1",
    "outer5-iid-fold2-v1",
    "outer5-iid-fold3-v1",
    "outer5-iid-fold4-v1",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy_b_and_c_excluded_datasets() -> Dict[str, Any]:
    with CORPUS_MANIFEST_PATH.open() as f:
        cm = json.load(f)
    cid_to_ds = {r["context_sha256"]: r["dataset_id"] for r in cm["records"]}

    with HIST_ROLES_PATH.open() as f:
        roles = json.load(f)["roles"]
    with SPLIT_RECON_PATH.open() as f:
        rec = json.load(f)

    st16_cids = list(rec["authorized_st16_inner70_allowlist_v2"].keys())
    val4_cids = list(rec["excluded_val4_fast_cids_v2"])
    stage2_20_cids = st16_cids + val4_cids
    quarantined_3_cids = list(rec["quarantined_3_cids_v2"])

    all_1298 = (
        set(roles["strict_train_fit_881"])
        | set(roles["dl_train_inner_203"])
        | set(stage2_20_cids)
        | set(roles["st_val_gate_32"])
        | set(roles["dl_val_71"])
        | set(roles["dl_holdout_91"])
    )
    non_outer = [
        r["context_sha256"]
        for r in cm["records"]
        if r["source_split"] == "train" and r["n_candidates"] > 0
    ]
    cross_fold_102_cids = [
        c for c in non_outer if c not in all_1298 and c not in quarantined_3_cids
    ]

    ds_stage2_20 = {cid_to_ds[c] for c in stage2_20_cids}
    ds_gate32 = {cid_to_ds[c] for c in roles["st_val_gate_32"]}
    ds_val71 = {cid_to_ds[c] for c in roles["dl_val_71"]}
    ds_holdout91 = {cid_to_ds[c] for c in roles["dl_holdout_91"]}
    ds_dl_val = {r["dataset_id"] for r in cm["records"] if r["dl_split"] == "DL-Val"}
    ds_dl_holdout = {r["dataset_id"] for r in cm["records"] if r["dl_split"] == "DL-Holdout"}

    ds_policy_b = ds_stage2_20 | ds_gate32 | ds_val71 | ds_holdout91 | ds_dl_val | ds_dl_holdout
    ds_policy_c = ds_policy_b | {
        cid_to_ds[c] for c in (cross_fold_102_cids + quarantined_3_cids)
    }

    return {
        "ds_policy_b": ds_policy_b,
        "ds_policy_c": ds_policy_c,
        "ds_stage2_20": ds_stage2_20,
        "ds_gate32": ds_gate32,
        "ds_val71": ds_val71,
        "ds_holdout91": ds_holdout91,
        "ds_dl_val": ds_dl_val,
        "ds_dl_holdout": ds_dl_holdout,
    }


def load_full_s3_regression_7split_bundles(
    excl: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    manifest_path = S3_MIRROR_ROOT / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    role_buckets: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for role in ("train_gradient", "eval_gradient_disjoint_v2_panel", "dev_val_ablation"):
        role_buckets[role] = {
            "dope629": [],
            "hidden256": [],
            "col_lm17": [],
            "col_mask64": [],
            "w_basis661": [],
            "dataset_id": [],
            "physical_table_id": [],
            "split": [],
            "landmark_report_origin": [],
            "precalc_plan_state": [],
        }

    non_regression_skipped = 0
    for b_info in manifest["partitioned_bundles"]:
        v3_role = b_info["v3_role"]
        if v3_role not in role_buckets:
            continue
        b_path = S3_MIRROR_ROOT / b_info["bundle_relpath"]
        with np.load(b_path, allow_pickle=False) as z:
            tasks = z["model_task"]
            reg_mask = tasks == "regression"
            non_regression_skipped += int(np.sum(~reg_mask))
            if not np.any(reg_mask):
                continue
            idx = np.where(reg_mask)[0]
            dope629 = z["neural_table_embedding_629_fp32"][idx].astype(np.float32)
            hidden256 = z["router_and_film_512_fp32"][idx, :256].astype(np.float32)
            col_lm24 = (
                z["slot_aligned_encoding_8192_fp16"][idx, 1141:2677]
                .astype(np.float32)
                .reshape(-1, 64, 24)
            )
            col_lm17 = np.ascontiguousarray(col_lm24[:, :, :17], dtype=np.float32)
            col_mask64 = z["col_mask64"][idx].astype(np.float32)
            w_basis661 = z["w_basis_slot_aligned_661_fp32"][idx].astype(np.float32)

            rb = role_buckets[v3_role]
            rb["dope629"].append(dope629)
            rb["hidden256"].append(hidden256)
            rb["col_lm17"].append(col_lm17)
            rb["col_mask64"].append(col_mask64)
            rb["w_basis661"].append(w_basis661)
            rb["dataset_id"].append(z["dataset_id"][idx])
            rb["physical_table_id"].append(z["physical_table_id"][idx])
            rb["split"].append(z["split"][idx])
            rb["landmark_report_origin"].append(z["landmark_report_origin"][idx])
            rb["precalc_plan_state"].append(z["precalc_plan_state"][idx])

    assembled: Dict[str, Dict[str, Any]] = {}
    for role, rb in role_buckets.items():
        assembled[role] = {k: np.concatenate(v, axis=0) for k, v in rb.items()}

    tg = assembled["train_gradient"]
    tg_ds = tg["dataset_id"]
    mask_b = np.array([d not in excl["ds_policy_b"] for d in tg_ds], dtype=bool)
    mask_c = np.array([d not in excl["ds_policy_c"] for d in tg_ds], dtype=bool)

    def subset_dict(src: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: v[mask] for k, v in src.items()}

    assembled["train_policy_b_11121"] = subset_dict(tg, mask_b)
    assembled["train_policy_c_11086"] = subset_dict(tg, mask_c)
    assembled["meta_counts"] = {
        "non_regression_cells_filtered_out": non_regression_skipped,
        "raw_regression_train_gradient_cells": int(len(tg_ds)),
        "policy_b_regression_train_cells": int(np.sum(mask_b)),
        "policy_b_regression_unique_datasets": int(len(set(tg_ds[mask_b].tolist()))),
        "policy_c_regression_train_cells": int(np.sum(mask_c)),
        "policy_c_regression_unique_datasets": int(len(set(tg_ds[mask_c].tolist()))),
        "eval_disjoint_v2_regression_cells": int(
            len(assembled["eval_gradient_disjoint_v2_panel"]["dataset_id"])
        ),
        "eval_disjoint_v2_regression_unique_datasets": int(
            len(set(assembled["eval_gradient_disjoint_v2_panel"]["dataset_id"].tolist()))
        ),
        "dev_val_ablation_regression_cells": int(len(assembled["dev_val_ablation"]["dataset_id"])),
        "dev_val_ablation_regression_unique_datasets": int(
            len(set(assembled["dev_val_ablation"]["dataset_id"].tolist()))
        ),
    }
    return assembled


def fit_s3_panel_scaler(panel: Dict[str, np.ndarray]) -> FoldScaler:
    dope_mat = panel["dope629"].astype(np.float64)
    hid_mat = panel["hidden256"].astype(np.float64)

    col_gates = panel["dope629"][:, 565:629, None]
    col18_all = np.concatenate([panel["col_lm17"], col_gates], axis=-1)
    mask_bool = panel["col_mask64"] > 0.5
    active_cols = col18_all[mask_bool].astype(np.float64)

    return FoldScaler(
        dope629_mean=dope_mat.mean(axis=0).astype(np.float32),
        dope629_std=np.maximum(dope_mat.std(axis=0), 1e-4).astype(np.float32),
        hidden256_mean=hid_mat.mean(axis=0).astype(np.float32),
        hidden256_std=np.maximum(hid_mat.std(axis=0), 1e-4).astype(np.float32),
        col18_mean=active_cols.mean(axis=0).astype(np.float32),
        col18_std=np.maximum(active_cols.std(axis=0), 1e-4).astype(np.float32),
    )


def compute_batch_log_norms_11(dope_z: np.ndarray) -> np.ndarray:
    def _rms(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.sqrt(np.mean(x * x, axis=1) + 1e-8))

    def _max(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.max(np.abs(x), axis=1))

    return np.stack(
        [
            _rms(dope_z[:, 0:256]),
            _max(dope_z[:, 0:256]),
            _rms(dope_z[:, 256:384]),
            _max(dope_z[:, 256:384]),
            _rms(dope_z[:, 384:512]),
            _max(dope_z[:, 384:512]),
            _rms(dope_z[:, 512:526]),
            _rms(dope_z[:, 526:549]),
            _rms(dope_z[:, 549:565]),
            _rms(dope_z[:, 565:629]),
            _max(dope_z[:, 565:629]),
        ],
        axis=1,
    ).astype(np.float32)


def pack_panel_tensors(
    panel: Dict[str, np.ndarray],
    scaler: FoldScaler,
    net,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    dope_raw = panel["dope629"].astype(np.float32)
    dope_z = np.clip(
        (dope_raw - scaler.dope629_mean[None, :]) / scaler.dope629_std[None, :],
        -15.0,
        15.0,
    ).astype(np.float32)
    hid_z = np.clip(
        (panel["hidden256"].astype(np.float32) - scaler.hidden256_mean[None, :])
        / scaler.hidden256_std[None, :],
        -15.0,
        15.0,
    ).astype(np.float32)

    col_mask = (panel["col_mask64"] > 0.5).astype(np.float32)
    col_gates = dope_raw[:, 565:629, None]
    col18_raw = np.concatenate([panel["col_lm17"], col_gates], axis=-1)
    col18_z = np.clip(
        (col18_raw - scaler.col18_mean[None, None, :]) / scaler.col18_std[None, None, :],
        -15.0,
        15.0,
    ).astype(np.float32)
    col18_z = col18_z * col_mask[:, :, None]
    norms_11 = compute_batch_log_norms_11(dope_z)

    with torch.no_grad():
        d_t = torch.as_tensor(dope_raw, dtype=torch.float32, device=device)
        v_t = torch.nan_to_num(d_t[:, :256], nan=0.0, posinf=0.0, neginf=0.0)
        v256_norm = torch.clamp(
            (v_t - net.step2_lm_mean_1x256) / net.step2_lm_std_1x256,
            -6.0,
            6.0,
        )
        lm48_t = v256_norm @ net.step2_svd_basis_256x48
        t_10x32, m_10, _ = net._project_stage1b_tokens_batched_gpu(lm48_t, d_t)

    return {
        "tokens_10x32": t_10x32.detach(),
        "mask_10": m_10.detach().float(),
        "dope629_z": torch.as_tensor(dope_z, dtype=torch.float32, device=device),
        "block_log_norms": torch.as_tensor(norms_11, dtype=torch.float32, device=device),
        "hidden256_z": torch.as_tensor(hid_z, dtype=torch.float32, device=device),
        "col18_z": torch.as_tensor(col18_z, dtype=torch.float32, device=device),
        "col_mask_64": torch.as_tensor(col_mask, dtype=torch.float32, device=device),
    }


def slice_batch_tensors(
    panel_tensors: Dict[str, torch.Tensor], b_t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {k: v[b_t] for k, v in panel_tensors.items()}


class S3PretrainBasisModel(nn.Module):
    """Wraps `PreCondenseResidualBridge(arm='G_FULL_S3_REG_7SPLIT_11121_PRETRAINED')` with:
      1. Cross-attention + global bilinear context summary + 661D regression `w_basis` head, and
      2. Masked DOPE629 semantic block reconstruction head.
    """

    def __init__(self, d_token: int = 32) -> None:
        super().__init__()
        self.bridge = PreCondenseResidualBridge(
            arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
            d_token=d_token,
        )
        self.query_tokens = nn.Parameter(torch.randn(1, 4, d_token) * 0.02)
        self.basis_head = nn.Sequential(
            nn.Linear(d_token * 4 + 64 + 24 + 8, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 661),
        )
        self.denoise_629_head = nn.Sequential(
            nn.Linear(d_token * 4 + 64 + 8, 256),
            nn.SiLU(),
            nn.Linear(256, 629),
        )

    def encode_summary(
        self, ctx_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = ctx_batch["dope629_z"].shape[0]
        ctx_tokens, ctx_mask, scale_8 = self.bridge.encode_context_tokens(ctx_batch)

        mask_exp = ctx_mask.unsqueeze(-1)
        mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
        ctx_mean = (ctx_tokens * mask_exp).sum(dim=1) / mask_sum
        ctx_masked_for_max = torch.where(
            mask_exp > 0.5, ctx_tokens, torch.full_like(ctx_tokens, -1e4)
        )
        ctx_max = ctx_masked_for_max.max(dim=1).values
        ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)  # [B, 64]

        q = self.query_tokens.expand(B, -1, -1)
        attn_q = self.bridge.ctx_cross_attn(q, ctx_tokens, valid_mask=ctx_mask).reshape(B, -1)
        u_ctx = self.bridge.ctx_bilinear_u(ctx_summary_64)  # [B, 24]

        full_summary = torch.cat([attn_q, ctx_summary_64, u_ctx, scale_8], dim=-1)
        denoise_in = torch.cat([attn_q, ctx_summary_64, scale_8], dim=-1)
        return full_summary, denoise_in

    def forward_basis(self, ctx_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        full_summary, _ = self.encode_summary(ctx_batch)
        return self.basis_head(full_summary)


class Frozen10TokBasisBaseline(nn.Module):
    """Matched baseline predicting regression `w_basis_661` strictly from the frozen Stage-1B
    `10 x 32` (`7` active tokens) representation.
    """

    def __init__(self, d_token: int = 32) -> None:
        super().__init__()
        self.bridge = PreCondenseResidualBridge(
            arm="B_CAPACITY_CTRL_10TOK",
            d_token=d_token,
        )
        self.query_tokens = nn.Parameter(torch.randn(1, 4, d_token) * 0.02)
        self.basis_head = nn.Sequential(
            nn.Linear(d_token * 4 + 64 + 24 + 8, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 661),
        )

    def forward(self, ctx_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = ctx_batch["tokens_10x32"].shape[0]
        ctx_tokens, ctx_mask, scale_8 = self.bridge.encode_context_tokens(ctx_batch)
        mask_exp = ctx_mask.unsqueeze(-1)
        mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
        ctx_mean = (ctx_tokens * mask_exp).sum(dim=1) / mask_sum
        ctx_masked_for_max = torch.where(
            mask_exp > 0.5, ctx_tokens, torch.full_like(ctx_tokens, -1e4)
        )
        ctx_max = ctx_masked_for_max.max(dim=1).values
        ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)

        q = self.query_tokens.expand(B, -1, -1)
        attn_q = self.bridge.ctx_cross_attn(q, ctx_tokens, valid_mask=ctx_mask).reshape(B, -1)
        u_ctx = self.bridge.ctx_bilinear_u(ctx_summary_64)
        return self.basis_head(torch.cat([attn_q, ctx_summary_64, u_ctx, scale_8], dim=-1))


def evaluate_basis_predictions(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    splits: np.ndarray,
) -> Dict[str, Any]:
    mse = float(np.mean((pred_z - target_z) ** 2))
    var = float(np.mean((target_z - np.mean(target_z, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - mse / max(var, 1e-12))

    p_norm = np.linalg.norm(pred_z, axis=1) + 1e-12
    t_norm = np.linalg.norm(target_z, axis=1) + 1e-12
    cos_sim = float(np.mean(np.sum(pred_z * target_z, axis=1) / (p_norm * t_norm)))

    per_split: Dict[str, Dict[str, float]] = {}
    for sp in SEVEN_SPLITS:
        m = splits == sp
        if not np.any(m):
            continue
        p_s = pred_z[m]
        t_s = target_z[m]
        mse_s = float(np.mean((p_s - t_s) ** 2))
        var_s = float(np.mean((t_s - np.mean(t_s, axis=0, keepdims=True)) ** 2))
        r2_s = float(1.0 - mse_s / max(var_s, 1e-12))
        pn = np.linalg.norm(p_s, axis=1) + 1e-12
        tn = np.linalg.norm(t_s, axis=1) + 1e-12
        cs = float(np.mean(np.sum(p_s * t_s, axis=1) / (pn * tn)))
        per_split[sp] = {
            "n_cells": int(np.sum(m)),
            "mse": mse_s,
            "r2_variance_explained": r2_s,
            "cosine_similarity": cs,
        }

    return {
        "n_cells": int(len(pred_z)),
        "mse": mse,
        "r2_variance_explained": r2,
        "cosine_similarity": cos_sim,
        "by_split": per_split,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. Load locked FusedJopeNetwork and verify DOPE freeze guard
    net, guard, dope_snap_before = ex.load_locked_fused_network(device)

    # 2. Load Policy B & Policy C dataset exclusion sets and full-S3 7-split regression bundles
    excl = load_policy_b_and_c_excluded_datasets()
    bundles = load_full_s3_regression_7split_bundles(excl)
    meta = bundles["meta_counts"]

    assert meta["raw_regression_train_gradient_cells"] == 11359, meta
    assert meta["policy_b_regression_train_cells"] == 11121, meta
    assert meta["policy_c_regression_train_cells"] == 11086, meta
    assert meta["eval_disjoint_v2_regression_cells"] == 4389, meta
    assert meta["dev_val_ablation_regression_cells"] == 286, meta

    train_b = bundles["train_policy_b_11121"]
    train_c = bundles["train_policy_c_11086"]
    eval_v2 = bundles["eval_gradient_disjoint_v2_panel"]
    dev_val = bundles["dev_val_ablation"]

    train_b_ds = set(train_b["dataset_id"].tolist())
    eval_phys = set(eval_v2["physical_table_id"].tolist()) | set(
        dev_val["physical_table_id"].tolist()
    )
    train_c_phys = set(train_c["physical_table_id"].tolist())

    overlap_checks = {
        "policy_b_intersect_stage2_20": len(train_b_ds & excl["ds_stage2_20"]),
        "policy_b_intersect_st_val_gate_32": len(train_b_ds & excl["ds_gate32"]),
        "policy_b_intersect_dl_val_71": len(train_b_ds & excl["ds_val71"]),
        "policy_b_intersect_dl_holdout_91": len(train_b_ds & excl["ds_holdout91"]),
        "policy_b_intersect_dl_val_91": len(train_b_ds & excl["ds_dl_val"]),
        "policy_b_intersect_dl_holdout_93": len(train_b_ds & excl["ds_dl_holdout"]),
        "policy_c_intersect_eval_physical_tables": len(train_c_phys & eval_phys),
    }
    if any(v != 0 for v in overlap_checks.values()):
        raise RuntimeError(f"Disjointness violation: {overlap_checks}")

    # 3. Compute Stage-1B bottleneck clipping & rank telemetry across all 11,121 train and 4,389 eval cells
    with torch.no_grad():
        d_tr = torch.as_tensor(train_b["dope629"], dtype=torch.float32, device=device)
        d_ev = torch.as_tensor(eval_v2["dope629"], dtype=torch.float32, device=device)
        s3_slot_telemetry = {
            slot: {
                "train_policy_b_11121": ex.compute_stage1b_slot_telemetry(net, d_tr, slot),
                "eval_disjoint_4389": ex.compute_stage1b_slot_telemetry(net, d_ev, slot),
            }
            for slot in ("svd_slot2_m2", "svd_slot4_d629_mid", "svd_slot5_d629_tail")
        }

    # 4. Fit FoldScaler on Policy B 11,121 regression cells and Policy C 11,086 regression cells
    scaler_b = fit_s3_panel_scaler(train_b)
    scaler_c = fit_s3_panel_scaler(train_c)

    # Standardize regression target `w_basis661` strictly on Policy B 11,121 train cells
    wb_mean = np.mean(train_b["w_basis661"], axis=0).astype(np.float32)
    wb_std = np.maximum(np.std(train_b["w_basis661"], axis=0), 1e-4).astype(np.float32)
    y_tr = np.clip((train_b["w_basis661"] - wb_mean) / wb_std, -8.0, 8.0).astype(np.float32)
    y_ev = np.clip((eval_v2["w_basis661"] - wb_mean) / wb_std, -8.0, 8.0).astype(np.float32)
    y_dv = np.clip((dev_val["w_basis661"] - wb_mean) / wb_std, -8.0, 8.0).astype(np.float32)

    # 5. Pack tensors for train_policy_b_11121, eval_disjoint_4389, and dev_val_286
    tr_tensors = pack_panel_tensors(train_b, scaler_b, net, device)
    ev_tensors = pack_panel_tensors(eval_v2, scaler_b, net, device)
    dv_tensors = pack_panel_tensors(dev_val, scaler_b, net, device)

    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)

    ds_to_indices: Dict[str, List[int]] = {}
    for idx_i, ds_id in enumerate(train_b["dataset_id"].tolist()):
        ds_to_indices.setdefault(ds_id, []).append(idx_i)
    multi_split_ds = [idxs for idxs in ds_to_indices.values() if len(idxs) >= 2]
    slice_ranges = list(BLOCK_SLICES.values())

    # 6. Train PreCondense S3PretrainBasisModel vs Frozen10TokBasisBaseline on Policy B 11,121 regression cells
    torch.manual_seed(256700)
    np.random.seed(256700)
    pre_model = S3PretrainBasisModel(d_token=32).to(device)
    base_10tok = Frozen10TokBasisBaseline(d_token=32).to(device)

    opt_pre = torch.optim.AdamW(pre_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    opt_10tok = torch.optim.AdamW(base_10tok.parameters(), lr=2.0e-3, weight_decay=1e-4)

    guard.assert_optimizer_excludes_dope(opt_pre, net)
    guard.assert_optimizer_excludes_dope(opt_10tok, net)

    n_train = len(y_tr)
    batch_size = 256
    n_epochs = 16
    rng = np.random.default_rng(256700)

    epoch_history: List[Dict[str, float]] = []
    for epoch in range(n_epochs):
        pre_model.train()
        base_10tok.train()
        perm = rng.permutation(n_train)
        ep_loss_pre = 0.0
        ep_loss_10 = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            b_idx = perm[start : start + batch_size]
            b_t = torch.as_tensor(b_idx, dtype=torch.long, device=device)
            batch_ctx = slice_batch_tensors(tr_tensors, b_t)
            y_b = y_tr_t[b_t]

            # (a) Pre-Condense model forward + masked block denoising + 7-split invariance
            opt_pre.zero_grad()
            full_sum, _ = pre_model.encode_summary(batch_ctx)
            pred_basis = pre_model.basis_head(full_sum)
            loss_basis = F.smooth_l1_loss(pred_basis, y_b, beta=0.5)

            # Randomly mask one semantic block of dope629_z and reconstruct full dope629_z from
            # the remaining blocks + hidden256 + col18_z
            s_blk, e_blk = slice_ranges[int(rng.integers(0, len(slice_ranges)))]
            masked_ctx = dict(batch_ctx)
            d_masked = batch_ctx["dope629_z"].clone()
            d_masked[:, s_blk:e_blk] = 0.0
            masked_ctx["dope629_z"] = d_masked
            _, denoise_in = pre_model.encode_summary(masked_ctx)
            rec_dope = pre_model.denoise_629_head(denoise_in)
            loss_denoise = F.mse_loss(
                rec_dope[:, s_blk:e_blk], batch_ctx["dope629_z"][:, s_blk:e_blk]
            )

            # 7-split invariance across splits of the same dataset
            ds_sample = rng.choice(
                len(multi_split_ds), size=min(32, len(multi_split_ds)), replace=False
            )
            idx_s1 = [multi_split_ds[i][0] for i in ds_sample]
            idx_s2 = [
                multi_split_ds[i][int(rng.integers(1, len(multi_split_ds[i])))]
                for i in ds_sample
            ]
            t_s1 = torch.as_tensor(idx_s1, dtype=torch.long, device=device)
            t_s2 = torch.as_tensor(idx_s2, dtype=torch.long, device=device)
            sum_s1, _ = pre_model.encode_summary(slice_batch_tensors(tr_tensors, t_s1))
            sum_s2, _ = pre_model.encode_summary(slice_batch_tensors(tr_tensors, t_s2))
            loss_split_inv = F.mse_loss(
                F.normalize(sum_s1, dim=-1), F.normalize(sum_s2, dim=-1)
            )

            loss_pre = loss_basis + 0.10 * loss_denoise + 0.10 * loss_split_inv
            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(pre_model.parameters(), 1.0)
            opt_pre.step()

            # (b) Frozen 10-token baseline forward
            opt_10tok.zero_grad()
            pred_10 = base_10tok(batch_ctx)
            loss_10 = F.smooth_l1_loss(pred_10, y_b, beta=0.5)
            loss_10.backward()
            torch.nn.utils.clip_grad_norm_(base_10tok.parameters(), 1.0)
            opt_10tok.step()

            ep_loss_pre += float(loss_pre.detach())
            ep_loss_10 += float(loss_10.detach())
            n_batches += 1

        guard.verify_dope_frozen_unchanged(net, dope_snap_before)
        epoch_history.append({
            "epoch": epoch + 1,
            "pre_condense_total_loss": ep_loss_pre / max(n_batches, 1),
            "frozen_10tok_basis_loss": ep_loss_10 / max(n_batches, 1),
        })

    # 7. Evaluate out-of-sample on `eval_gradient_disjoint_v2_panel` (4,389 regression cells)
    #    and `dev_val_ablation` (286 regression cells) across all 7 splits
    pre_model.eval()
    base_10tok.eval()
    with torch.no_grad():
        pred_tr_pre = pre_model.forward_basis(tr_tensors).cpu().numpy()
        pred_tr_10 = base_10tok(tr_tensors).cpu().numpy()
        pred_ev_pre = pre_model.forward_basis(ev_tensors).cpu().numpy()
        pred_ev_10 = base_10tok(ev_tensors).cpu().numpy()
        pred_dv_pre = pre_model.forward_basis(dv_tensors).cpu().numpy()
        pred_dv_10 = base_10tok(dv_tensors).cpu().numpy()

    eval_summary = {
        "train_policy_b_11121": {
            "pre_condense_92tok": evaluate_basis_predictions(pred_tr_pre, y_tr, train_b["split"]),
            "frozen_stage1b_10tok": evaluate_basis_predictions(pred_tr_10, y_tr, train_b["split"]),
        },
        "eval_gradient_disjoint_v2_panel_4389": {
            "pre_condense_92tok": evaluate_basis_predictions(pred_ev_pre, y_ev, eval_v2["split"]),
            "frozen_stage1b_10tok": evaluate_basis_predictions(pred_ev_10, y_ev, eval_v2["split"]),
        },
        "dev_val_ablation_286": {
            "pre_condense_92tok": evaluate_basis_predictions(pred_dv_pre, y_dv, dev_val["split"]),
            "frozen_stage1b_10tok": evaluate_basis_predictions(pred_dv_10, y_dv, dev_val["split"]),
        },
    }

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    dope_snap_after = guard.snapshot_dope_weights(net)

    encoder_prefixes = (
        "desc_proj.",
        "task_proj.",
        "dist_proj.",
        "heads_proj.",
        "colgate_proj.",
        "full629_norm.",
        "full629_type_emb",
        "full629_gate_logits",
        "hidden_proj.",
        "hidden_norm.",
        "hidden_type_emb",
        "hidden_gate_logits",
        "col_encoder.",
        "col_type_emb",
        "col_gate_logit",
        "ctx_cross_attn.",
        "ctx_bilinear_u.",
        "scale_side_proj.",
    )
    bridge_state = {
        k: v.detach().cpu()
        for k, v in pre_model.bridge.state_dict().items()
        if any(k.startswith(p) for p in encoder_prefixes)
    }

    ckpt_payload = {
        "schema": "jope.h256.full_s3_reg_7split_pretrained_encoder/v1",
        "model_task_filter": "regression_only",
        "dope_sha256": dope_snap_after.composite_sha256,
        "policy_b_regression_train_cells": meta["policy_b_regression_train_cells"],
        "policy_c_regression_train_cells": meta["policy_c_regression_train_cells"],
        "eval_disjoint_v2_regression_cells": meta["eval_disjoint_v2_regression_cells"],
        "dev_val_ablation_regression_cells": meta["dev_val_ablation_regression_cells"],
        "encoder_state_dict": bridge_state,
        "scaler_policy_b": {
            "dope629_mean": scaler_b.dope629_mean,
            "dope629_std": scaler_b.dope629_std,
            "hidden256_mean": scaler_b.hidden256_mean,
            "hidden256_std": scaler_b.hidden256_std,
            "col18_mean": scaler_b.col18_mean,
            "col18_std": scaler_b.col18_std,
        },
        "scaler_policy_c": {
            "dope629_mean": scaler_c.dope629_mean,
            "dope629_std": scaler_c.dope629_std,
            "hidden256_mean": scaler_c.hidden256_mean,
            "hidden256_std": scaler_c.hidden256_std,
            "col18_mean": scaler_c.col18_mean,
            "col18_std": scaler_c.col18_std,
        },
    }
    torch.save(ckpt_payload, OUT_CKPT)
    ckpt_sha256 = sha256_file(OUT_CKPT)

    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    full_stage1_ckpt_path = ckpt_dir / "stage1_full_s3_reg_7split_pre_condense_92tok_full_model.pt"
    full_stage1_payload = {
        "schema": "jope.h256.stage1_full_s3_reg_7split_full_model/v1",
        "stage": "stage1_full_s3_regression_7split_pretraining",
        "model_task_filter": "regression_only",
        "dope_sha256": dope_snap_after.composite_sha256,
        "pre_condense_92tok_full_state_dict": {
            k: v.detach().cpu() for k, v in pre_model.state_dict().items()
        },
        "bridge_state_dict": {
            k: v.detach().cpu() for k, v in pre_model.bridge.state_dict().items()
        },
        "encoder_state_dict": bridge_state,
        "frozen_stage1b_10tok_baseline_state_dict": {
            k: v.detach().cpu() for k, v in base_10tok.state_dict().items()
        },
        "w_basis661_mean": wb_mean,
        "w_basis661_std": wb_std,
        "scaler_policy_b": ckpt_payload["scaler_policy_b"],
        "scaler_policy_c": ckpt_payload["scaler_policy_c"],
        "out_of_sample_regression_basis_evaluation": eval_summary,
    }
    torch.save(full_stage1_payload, full_stage1_ckpt_path)
    full_stage1_ckpt_sha256 = sha256_file(full_stage1_ckpt_path)

    receipt = {
        "schema": "jope.h256.full_s3_reg_7split_receipt/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 3),
        "model_task_filter": "regression_only (0 binary/multiclass cells admitted)",
        "dope_freeze_verification": {
            "expected_sha256": ch.DOPE_SHA256,
            "before_sha256": dope_snap_before.composite_sha256,
            "after_sha256": dope_snap_after.composite_sha256,
            "verified_every_epoch": True,
            "optimizer_excludes_dope": True,
            "unchanged": dope_snap_before.composite_sha256
            == dope_snap_after.composite_sha256
            == ch.DOPE_SHA256,
        },
        "s3_regression_corpus_counts": meta,
        "disjointness_checks": overlap_checks,
        "stage1b_bottleneck_telemetry_full_s3_regression": s3_slot_telemetry,
        "pretraining_epochs": n_epochs,
        "epoch_history": epoch_history,
        "out_of_sample_regression_basis_evaluation": eval_summary,
        "pretrained_checkpoint_path": str(OUT_CKPT),
        "pretrained_checkpoint_sha256": ckpt_sha256,
        "full_stage1_checkpoint_path": str(full_stage1_ckpt_path),
        "full_stage1_checkpoint_sha256": full_stage1_ckpt_sha256,
    }
    with OUT_RECEIPT.open("w") as f:
        json.dump(receipt, f, indent=2)

    ev_pre = eval_summary["eval_gradient_disjoint_v2_panel_4389"]["pre_condense_92tok"]
    ev_10 = eval_summary["eval_gradient_disjoint_v2_panel_4389"]["frozen_stage1b_10tok"]
    dv_pre = eval_summary["dev_val_ablation_286"]["pre_condense_92tok"]
    dv_10 = eval_summary["dev_val_ablation_286"]["frozen_stage1b_10tok"]

    print(
        f"[H256 Stage-1 Full-S3 7-Split Regression Pretraining DONE in {time.time() - t0:.2f}s]\n"
        f"  Regression train_gradient: Policy B={meta['policy_b_regression_train_cells']} "
        f"({meta['policy_b_regression_unique_datasets']} datasets x 7 splits), "
        f"Policy C={meta['policy_c_regression_train_cells']} ({meta['policy_c_regression_unique_datasets']} datasets)\n"
        f"  Disjoint Eval (4,389 cells / 627 datasets x 7 splits): "
        f"Pre-Condense R2={ev_pre['r2_variance_explained']:.4f} (cos={ev_pre['cosine_similarity']:.4f}) vs "
        f"Frozen-10Tok R2={ev_10['r2_variance_explained']:.4f} (cos={ev_10['cosine_similarity']:.4f})\n"
        f"  Dev-Val Ablation (286 cells / 41 datasets x 7 splits): "
        f"Pre-Condense R2={dv_pre['r2_variance_explained']:.4f} (cos={dv_pre['cosine_similarity']:.4f}) vs "
        f"Frozen-10Tok R2={dv_10['r2_variance_explained']:.4f} (cos={dv_10['cosine_similarity']:.4f})\n"
        f"  DOPE SHA-256 unchanged: {dope_snap_after.composite_sha256}"
    )


def train_all15796_production_encoder() -> Dict[str, Any]:
    """Trains the production Stage-1 Pre-Condense 92-token encoder across ALL 15,796 S3 7-split
    regression cells (11,121 Policy-B train + 4,389 disjoint panel + 286 dev_val_ablation; 0 binary)
    while keeping `full_s3_reg_7split_pretrained_encoder.pt` (11,121 cells) untouched for OOF/holdout.
    """
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    net, guard, dope_snap_before = ex.load_locked_fused_network(device)
    excl = load_policy_b_and_c_excluded_datasets()
    bundles = load_full_s3_regression_7split_bundles(excl)

    train_b = bundles["train_policy_b_11121"]
    eval_v2 = bundles["eval_gradient_disjoint_v2_panel"]
    dev_val = bundles["dev_val_ablation"]

    all_panel: Dict[str, np.ndarray] = {}
    for k in train_b.keys():
        all_panel[k] = np.concatenate([train_b[k], eval_v2[k], dev_val[k]], axis=0)

    n_all = int(len(all_panel["dataset_id"]))
    assert n_all == 15796, f"Expected 15796 regression cells, got {n_all}"
    n_unique_ds = int(len(set(all_panel["dataset_id"].tolist())))

    scaler_all = fit_s3_panel_scaler(all_panel)
    wb_mean = np.mean(all_panel["w_basis661"], axis=0).astype(np.float32)
    wb_std = np.maximum(np.std(all_panel["w_basis661"], axis=0), 1e-4).astype(np.float32)
    y_all = np.clip((all_panel["w_basis661"] - wb_mean) / wb_std, -8.0, 8.0).astype(np.float32)

    all_tensors = pack_panel_tensors(all_panel, scaler_all, net, device)
    y_all_t = torch.as_tensor(y_all, dtype=torch.float32, device=device)

    ds_to_indices: Dict[str, List[int]] = {}
    for idx_i, ds_id in enumerate(all_panel["dataset_id"].tolist()):
        ds_to_indices.setdefault(ds_id, []).append(idx_i)
    multi_split_ds = [idxs for idxs in ds_to_indices.values() if len(idxs) >= 2]
    slice_ranges = list(BLOCK_SLICES.values())

    torch.manual_seed(256701)
    np.random.seed(256701)
    prod_model = S3PretrainBasisModel(d_token=32).to(device)
    opt_prod = torch.optim.AdamW(prod_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    guard.assert_optimizer_excludes_dope(opt_prod, net)

    batch_size = 256
    n_epochs = 16
    rng = np.random.default_rng(256701)
    epoch_history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        prod_model.train()
        perm = rng.permutation(n_all)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_all, batch_size):
            b_idx = perm[start : start + batch_size]
            b_t = torch.as_tensor(b_idx, dtype=torch.long, device=device)
            batch_ctx = slice_batch_tensors(all_tensors, b_t)
            y_b = y_all_t[b_t]

            opt_prod.zero_grad()
            full_sum, _ = prod_model.encode_summary(batch_ctx)
            pred_basis = prod_model.basis_head(full_sum)
            loss_basis = F.smooth_l1_loss(pred_basis, y_b, beta=0.5)

            s_blk, e_blk = slice_ranges[int(rng.integers(0, len(slice_ranges)))]
            masked_ctx = dict(batch_ctx)
            d_masked = batch_ctx["dope629_z"].clone()
            d_masked[:, s_blk:e_blk] = 0.0
            masked_ctx["dope629_z"] = d_masked
            _, denoise_in = prod_model.encode_summary(masked_ctx)
            rec_dope = prod_model.denoise_629_head(denoise_in)
            loss_denoise = F.mse_loss(
                rec_dope[:, s_blk:e_blk], batch_ctx["dope629_z"][:, s_blk:e_blk]
            )

            ds_sample = rng.choice(
                len(multi_split_ds), size=min(32, len(multi_split_ds)), replace=False
            )
            idx_s1 = [multi_split_ds[i][0] for i in ds_sample]
            idx_s2 = [
                multi_split_ds[i][int(rng.integers(1, len(multi_split_ds[i])))]
                for i in ds_sample
            ]
            t_s1 = torch.as_tensor(idx_s1, dtype=torch.long, device=device)
            t_s2 = torch.as_tensor(idx_s2, dtype=torch.long, device=device)
            sum_s1, _ = prod_model.encode_summary(slice_batch_tensors(all_tensors, t_s1))
            sum_s2, _ = prod_model.encode_summary(slice_batch_tensors(all_tensors, t_s2))
            loss_split_inv = F.mse_loss(
                F.normalize(sum_s1, dim=-1), F.normalize(sum_s2, dim=-1)
            )

            loss_total = loss_basis + 0.10 * loss_denoise + 0.10 * loss_split_inv
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(prod_model.parameters(), 1.0)
            opt_prod.step()

            ep_loss += float(loss_total.detach())
            n_batches += 1

        guard.verify_dope_frozen_unchanged(net, dope_snap_before)
        epoch_history.append({
            "epoch": epoch + 1,
            "pre_condense_total_loss": ep_loss / max(n_batches, 1),
        })

    prod_model.eval()
    with torch.no_grad():
        pred_all = prod_model.forward_basis(all_tensors).cpu().numpy()
    fit_eval = evaluate_basis_predictions(pred_all, y_all, all_panel["split"])

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    dope_snap_after = guard.snapshot_dope_weights(net)

    encoder_prefixes = (
        "desc_proj.",
        "task_proj.",
        "dist_proj.",
        "heads_proj.",
        "colgate_proj.",
        "full629_norm.",
        "full629_type_emb",
        "full629_gate_logits",
        "hidden_proj.",
        "hidden_norm.",
        "hidden_type_emb",
        "hidden_gate_logits",
        "col_encoder.",
        "col_type_emb",
        "col_gate_logit",
        "ctx_cross_attn.",
        "ctx_bilinear_u.",
        "scale_side_proj.",
    )
    bridge_state = {
        k: v.detach().cpu()
        for k, v in prod_model.bridge.state_dict().items()
        if any(k.startswith(p) for p in encoder_prefixes)
    }

    out_path = ckpt_dir / "stage1_full_s3_reg_all15796_encoder.pt"
    payload = {
        "schema": "jope.h256.stage1_full_s3_reg_all15796_encoder/v1",
        "stage": "stage1_full_s3_regression_7split_all15796_production",
        "model_task_filter": "regression_only (0 binary/multiclass cells admitted)",
        "dope_sha256": dope_snap_after.composite_sha256,
        "total_regression_cells": n_all,
        "unique_regression_datasets": n_unique_ds,
        "encoder_state_dict": bridge_state,
        "bridge_state_dict": {
            k: v.detach().cpu() for k, v in prod_model.bridge.state_dict().items()
        },
        "pre_condense_92tok_full_state_dict": {
            k: v.detach().cpu() for k, v in prod_model.state_dict().items()
        },
        "w_basis661_mean": wb_mean,
        "w_basis661_std": wb_std,
        "scaler_all15796": {
            "dope629_mean": scaler_all.dope629_mean,
            "dope629_std": scaler_all.dope629_std,
            "hidden256_mean": scaler_all.hidden256_mean,
            "hidden256_std": scaler_all.hidden256_std,
            "col18_mean": scaler_all.col18_mean,
            "col18_std": scaler_all.col18_std,
        },
        "epoch_history": epoch_history,
        "fit_evaluation": fit_eval,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    torch.save(payload, out_path)
    out_sha = sha256_file(out_path)
    print(
        f"[H256 Stage-1 All-15796 S3 Regression Production Encoder DONE in {time.time() - t0:.2f}s]\n"
        f"  Cells={n_all} across {n_unique_ds} regression datasets x 7 splits (0 binary)\n"
        f"  Fit R2={fit_eval['r2_variance_explained']:.4f}, cos={fit_eval['cosine_similarity']:.4f}, MSE={fit_eval['mse']:.4f}\n"
        f"  Saved: {out_path} (SHA-256: {out_sha})\n"
        f"  DOPE SHA-256 unchanged: {dope_snap_after.composite_sha256}"
    )
    return {
        "path": str(out_path),
        "sha256": out_sha,
        "total_regression_cells": n_all,
        "unique_regression_datasets": n_unique_ds,
        "fit_evaluation": fit_eval,
    }


if __name__ == "__main__":
    if "--all15796-only" in sys.argv:
        train_all15796_production_encoder()
    else:
        main()
        train_all15796_production_encoder()
