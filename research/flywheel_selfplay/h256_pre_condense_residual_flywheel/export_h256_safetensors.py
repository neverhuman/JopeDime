#!/usr/bin/env python3
"""H256 Deterministic `.safetensors` Exporter for 100% Rust Inference.

Converts all trained H256 flagship regression checkpoints (`.pt` / `.npz` / LightGBM strings)
into flat, canonical `.safetensors` bundles (`< 5 MiB` each) with zero pickle/Python/C++ runtime
dependencies so `crates/jope-service/src/ranker/h256.rs` can execute 100% Rust inference:

  1. `weights/h256_production_full_1310_regression_bundle.safetensors` (`~3.15 MiB`)
  2. `weights/h256_production_1219_dl_train_bundle.safetensors` (`~3.15 MiB`)
  3. `weights/stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors` (`~4.46 MiB`)
  4. `weights/stage1_full_s3_reg_all15796_encoder.safetensors` (`~0.75 MiB`)
  5. `weights/full_s3_reg_7split_pretrained_encoder.safetensors` (`~0.11 MiB`)
  6. `weights/sos_hurdle_and_sp_critic_models.safetensors` (`~0.04 MiB`)
  7. `weights/golden_inference_fixture_context0.safetensors` (`~2.90 MiB`)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import lightgbm as lgb
import numpy as np
import torch
from safetensors.numpy import load_file as load_safetensors_np
from safetensors.numpy import save_file as save_safetensors_np

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
DATA_DIR = Path("/home/ubuntu/jope-data/h256")
CKPT_DIR = DATA_DIR / "checkpoints"

sys.path.insert(0, str(HERE))
import pre_condense_models as pcm  # noqa: E402
import train_and_export_h256_flagship_production as prod256  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_lgbm_model_str_to_tensors(
    model_str: str,
    prefix: str,
    is_binary_clf: bool,
) -> Dict[str, np.ndarray]:
    """Parses a LightGBM `model_to_string()` dump into flat dense NumPy arrays for 100% Rust inference."""
    trees = []
    cur: Dict[str, Any] = {}
    for line in model_str.splitlines():
        if line.startswith("Tree="):
            if cur:
                trees.append(cur)
            cur = {"Tree": int(line.split("=")[1])}
        elif "=" in line and cur:
            k, v = line.split("=", 1)
            cur[k] = v.strip()
    if cur:
        trees.append(cur)

    n_trees = len(trees)
    max_leaves = max(int(t["num_leaves"]) for t in trees)
    max_internal = max(max_leaves - 1, 1)

    num_leaves = np.zeros(n_trees, dtype=np.int32)
    split_feature = np.zeros((n_trees, max_internal), dtype=np.int32)
    threshold = np.zeros((n_trees, max_internal), dtype=np.float64)
    decision_type = np.zeros((n_trees, max_internal), dtype=np.int32)
    left_child = np.zeros((n_trees, max_internal), dtype=np.int32)
    right_child = np.zeros((n_trees, max_internal), dtype=np.int32)
    leaf_value = np.zeros((n_trees, max_leaves), dtype=np.float64)

    for t_idx, t in enumerate(trees):
        nl = int(t["num_leaves"])
        num_leaves[t_idx] = nl
        ni = nl - 1
        if ni > 0:
            split_feature[t_idx, :ni] = [int(x) for x in t["split_feature"].split()]
            threshold[t_idx, :ni] = [float(x) for x in t["threshold"].split()]
            decision_type[t_idx, :ni] = [int(x) for x in t["decision_type"].split()]
            left_child[t_idx, :ni] = [int(x) for x in t["left_child"].split()]
            right_child[t_idx, :ni] = [int(x) for x in t["right_child"].split()]
        leaf_value[t_idx, :nl] = [float(x) for x in t["leaf_value"].split()]

    return {
        f"{prefix}.is_binary_clf": np.array([1 if is_binary_clf else 0], dtype=np.int32),
        f"{prefix}.num_leaves": num_leaves,
        f"{prefix}.split_feature": split_feature,
        f"{prefix}.threshold": threshold,
        f"{prefix}.decision_type": decision_type,
        f"{prefix}.left_child": left_child,
        f"{prefix}.right_child": right_child,
        f"{prefix}.leaf_value": leaf_value,
    }


def eval_flat_gbdt_tensors(
    X: np.ndarray,
    tensors: Dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    """Evaluates serialized flat GBDT tensors on `X [N, F]` and returns `[N]` float64 predictions."""
    is_bin = bool(int(tensors[f"{prefix}.is_binary_clf"][0]) != 0)
    num_leaves = tensors[f"{prefix}.num_leaves"]
    split_feature = tensors[f"{prefix}.split_feature"]
    threshold = tensors[f"{prefix}.threshold"]
    left_child = tensors[f"{prefix}.left_child"]
    right_child = tensors[f"{prefix}.right_child"]
    leaf_value = tensors[f"{prefix}.leaf_value"]

    out = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        row = X[i]
        acc = 0.0
        for t in range(len(num_leaves)):
            nl = int(num_leaves[t])
            if nl <= 1:
                acc += float(leaf_value[t, 0])
                continue
            node = 0
            while node >= 0:
                f_idx = int(split_feature[t, node])
                th = float(threshold[t, node])
                val = float(row[f_idx])
                if val <= th:
                    node = int(left_child[t, node])
                else:
                    node = int(right_child[t, node])
            acc += float(leaf_value[t, -1 - node])
        out[i] = 1.0 / (1.0 + np.exp(-acc)) if is_bin else acc
    return out


def add_state_dict_tensors(
    out: Dict[str, np.ndarray],
    prefix: str,
    state_dict: Dict[str, Any],
) -> None:
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        out[f"{prefix}.{k}"] = np.ascontiguousarray(arr)


def add_scaler_tensors(
    out: Dict[str, np.ndarray],
    prefix: str,
    scaler_dict: Dict[str, Any],
) -> None:
    for k in (
        "dope629_mean",
        "dope629_std",
        "hidden256_mean",
        "hidden256_std",
        "col18_mean",
        "col18_std",
    ):
        arr = np.asarray(scaler_dict[k], dtype=np.float32)
        out[f"{prefix}.{k}"] = np.ascontiguousarray(arr)


def add_critic_tensors(
    out: Dict[str, np.ndarray],
    prefix: str,
    critics: Dict[str, Any],
) -> None:
    for k in (
        "w_c_full",
        "w_g_full",
        "sp_dir_full",
        "w_c_oof_5fold",
        "w_g_oof_5fold",
        "sp_dir_oof_5fold",
    ):
        arr = np.asarray(critics[k], dtype=np.float64)
        out[f"{prefix}.{k}"] = np.ascontiguousarray(arr)


def add_production_gbdt_suite(
    out: Dict[str, np.ndarray],
    payload: Dict[str, Any],
) -> None:
    mapping = [
        ("gbdt.ranker_568", "top8_lgbm_ranker_568_model_str", False),
        ("gbdt.ranker_580", "top8_lgbm_ranker_580_model_str", False),
        ("gbdt.ranker_616", "top8_lgbm_ranker_616_model_str", False),
        ("gbdt.clf_568", "top8_lgbm_veto_clf_568_model_str", True),
        ("gbdt.clf_580", "top8_lgbm_veto_clf_580_model_str", True),
        ("gbdt.clf_616", "top8_lgbm_veto_clf_616_model_str", True),
    ]
    for prefix, key, is_bin in mapping:
        out.update(parse_lgbm_model_str_to_tensors(payload[key], prefix, is_bin))


def compute_production_top8_predictions_from_tensors(
    X_568: np.ndarray,
    X_616: np.ndarray,
    bundle_tensors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Computes single-context (`[8, 616]`) Top-8 neural + GBDT multi-view scores and veto risk."""
    X_580 = X_616[:, :580]

    # 1. Neural Top-8 Reranker
    nn_model = prod256.PreCondenseTop8NeuralReranker(in_dim=616, hidden_dim=128)
    nn_sd = {
        k[len("top8_nn.") :]: torch.as_tensor(v)
        for k, v in bundle_tensors.items()
        if k.startswith("top8_nn.") and k not in ("top8_nn.feat_mean", "top8_nn.feat_std")
    }
    nn_model.load_state_dict(nn_sd, strict=True)
    nn_model.eval()

    feat_mean = bundle_tensors["top8_nn.feat_mean"]
    feat_std = bundle_tensors["top8_nn.feat_std"]
    X_norm = np.clip((X_616[None, :, :] - feat_mean[None, None, :]) / feat_std[None, None, :], -10.0, 10.0).astype(
        np.float32
    )
    X_norm[:, :, 2] = X_616[None, :, 2]
    with torch.no_grad():
        nn_sc_t, nn_inv_t = nn_model(torch.as_tensor(X_norm, dtype=torch.float32))
    snn_raw = nn_sc_t[0].cpu().numpy().astype(np.float64)
    pnn = nn_inv_t[0].cpu().numpy().astype(np.float64)

    # 2. GBDT Rankers & Veto Classifiers
    r568_raw = eval_flat_gbdt_tensors(X_568, bundle_tensors, "gbdt.ranker_568")
    r580_raw = eval_flat_gbdt_tensors(X_580, bundle_tensors, "gbdt.ranker_580")
    r616_raw = eval_flat_gbdt_tensors(X_616, bundle_tensors, "gbdt.ranker_616")
    p568 = eval_flat_gbdt_tensors(X_568, bundle_tensors, "gbdt.clf_568")
    p580 = eval_flat_gbdt_tensors(X_580, bundle_tensors, "gbdt.clf_580")
    p616 = eval_flat_gbdt_tensors(X_616, bundle_tensors, "gbdt.clf_616")

    b568 = prod256.standardize_scores_per_query(r568_raw[None, :])[0]
    b580 = prod256.standardize_scores_per_query(r580_raw[None, :])[0]
    b616 = prod256.standardize_scores_per_query(r616_raw[None, :])[0]
    bnn = prod256.standardize_scores_per_query(snn_raw[None, :])[0]
    bbr = prod256.standardize_scores_per_query((-X_616[:, 578])[None, :])[0]

    sc = 0.40 * b568 + 0.30 * b580 + 0.15 * b616 + 0.10 * bnn + 0.05 * bbr
    pinv = 0.45 * p568 + 0.30 * p580 + 0.15 * p616 + 0.10 * pnn
    pick_j = int(
        prod256.select_picks_from_scores(sc[None, :], pinv[None, :], lam=1.5, tau=1.0, margin=0.05)[0]
    )
    details = {
        "nn_scores": snn_raw,
        "nn_inv_probs": pnn,
        "lgbm_ranker_568": r568_raw,
        "lgbm_ranker_580": r580_raw,
        "lgbm_ranker_616": r616_raw,
        "lgbm_clf_568": p568,
        "lgbm_clf_580": p580,
        "lgbm_clf_616": p616,
    }
    return sc, pinv, np.array([pick_j], dtype=np.int32), bnn, details


def main() -> None:
    p1310 = torch.load(
        WEIGHTS_DIR / "h256_production_full_1310_regression_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    p1219 = torch.load(
        WEIGHTS_DIR / "h256_production_1219_dl_train_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    s1_all15796 = torch.load(
        WEIGHTS_DIR / "stage1_full_s3_reg_all15796_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )
    s1_11121 = torch.load(
        WEIGHTS_DIR / "full_s3_reg_7split_pretrained_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )
    s2_bridge = torch.load(
        WEIGHTS_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt",
        map_location="cpu",
        weights_only=False,
    )
    critics_ckpt = torch.load(
        WEIGHTS_DIR / "sos_hurdle_and_sp_critic_models.pt",
        map_location="cpu",
        weights_only=False,
    )
    golden = torch.load(
        WEIGHTS_DIR / "golden_inference_fixture_context0.pt",
        map_location="cpu",
        weights_only=False,
    )

    # 1. Build `h256_production_full_1310_regression_bundle.safetensors` (384 tensors)
    t1310: Dict[str, np.ndarray] = {}
    add_state_dict_tensors(t1310, "s3_encoder", p1310["stage1_all15796_encoder_state_dict"])
    add_scaler_tensors(t1310, "s3_scaler", s1_all15796["scaler_all15796"])
    add_state_dict_tensors(t1310, "stage2_bridge", p1310["stage2_full_1310_bridge_state_dict"])
    add_state_dict_tensors(t1310, "stage2_arm_f_soup", p1310["stage2_bridge_arm_f_soup_state_dict"])
    add_state_dict_tensors(t1310, "stage2_arm_g_soup", p1310["stage2_bridge_arm_g_soup_state_dict"])
    add_scaler_tensors(t1310, "stage2_scaler", p1310["stage2_scaler_full_1310"])
    t1310["stage2_hyperparams"] = np.array(
        [
            float(p1310["stage2_full_1310_alpha_prior"]),
            float(p1310["stage2_full_1310_alpha_traj"]),
        ],
        dtype=np.float32,
    )
    add_critic_tensors(t1310, "critic", p1310["sos_hurdle_and_sp_critic_weights_full_1310"])
    add_state_dict_tensors(t1310, "top8_nn", p1310["top8_neural_reranker_state_dict"])
    t1310["top8_nn.feat_mean"] = np.ascontiguousarray(
        np.asarray(p1310["top8_neural_reranker_feat_mean"], dtype=np.float32)
    )
    t1310["top8_nn.feat_std"] = np.ascontiguousarray(
        np.asarray(p1310["top8_neural_reranker_feat_std"], dtype=np.float32)
    )
    add_production_gbdt_suite(t1310, p1310)

    out_1310_path = WEIGHTS_DIR / "h256_production_full_1310_regression_bundle.safetensors"
    save_safetensors_np(t1310, str(out_1310_path))

    # 2. Build `h256_production_1219_dl_train_bundle.safetensors` (exact same 384-tensor schema!)
    t1219: Dict[str, np.ndarray] = {}
    add_state_dict_tensors(t1219, "s3_encoder", s1_11121["encoder_state_dict"])
    add_scaler_tensors(t1219, "s3_scaler", s1_11121["scaler_policy_b"])
    add_state_dict_tensors(t1219, "stage2_bridge", s2_bridge["bridge_state_dict"])
    add_state_dict_tensors(t1219, "stage2_arm_f_soup", p1219["stage2_bridge_arm_f_soup_state_dict"])
    add_state_dict_tensors(t1219, "stage2_arm_g_soup", p1219["stage2_bridge_arm_g_soup_state_dict"])
    add_scaler_tensors(t1219, "stage2_scaler", s2_bridge["full_train_scaler"])
    t1219["stage2_hyperparams"] = np.array(
        [
            float(s2_bridge["ensemble_alpha_prior"]),
            float(s2_bridge["ensemble_alpha_traj"]),
        ],
        dtype=np.float32,
    )
    add_critic_tensors(t1219, "critic", p1219["sos_hurdle_and_sp_critic_weights"])
    add_state_dict_tensors(t1219, "top8_nn", p1219["top8_neural_reranker_state_dict"])
    t1219["top8_nn.feat_mean"] = np.ascontiguousarray(
        np.asarray(p1219["top8_neural_reranker_feat_mean"], dtype=np.float32)
    )
    t1219["top8_nn.feat_std"] = np.ascontiguousarray(
        np.asarray(p1219["top8_neural_reranker_feat_std"], dtype=np.float32)
    )
    add_production_gbdt_suite(t1219, p1219)

    assert set(t1219.keys()) == set(t1310.keys()), "1219 and 1310 bundles must have identical tensor keys"
    out_1219_path = WEIGHTS_DIR / "h256_production_1219_dl_train_bundle.safetensors"
    save_safetensors_np(t1219, str(out_1219_path))

    # 3. Build `stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors` (soup + 5-fold ensemble)
    ts2: Dict[str, np.ndarray] = {}
    add_state_dict_tensors(ts2, "soup_bridge", s2_bridge["bridge_state_dict"])
    add_scaler_tensors(ts2, "full_train_scaler", s2_bridge["full_train_scaler"])
    for k_fold in range(5):
        add_state_dict_tensors(ts2, f"fold{k_fold}_bridge", s2_bridge["fold_state_dicts"][k_fold])
        add_scaler_tensors(ts2, f"fold{k_fold}_scaler", s2_bridge["fold_scalers"][k_fold])
    ts2["ensemble_hyperparams"] = np.array(
        [
            float(s2_bridge["ensemble_alpha_prior"]),
            float(s2_bridge["ensemble_alpha_traj"]),
        ],
        dtype=np.float32,
    )
    ts2["fold_alpha_prior"] = np.ascontiguousarray(np.asarray(s2_bridge["fold_alpha_prior"], dtype=np.float32))
    ts2["fold_alpha_traj"] = np.ascontiguousarray(np.asarray(s2_bridge["fold_alpha_traj"], dtype=np.float32))
    add_critic_tensors(ts2, "critic", s2_bridge["sos_hurdle_and_sp_critic_models"])
    out_s2_path = WEIGHTS_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors"
    save_safetensors_np(ts2, str(out_s2_path))

    # 4. Build `stage1_full_s3_reg_all15796_encoder.safetensors` & `full_s3_reg_7split_pretrained_encoder.safetensors`
    ts1_all: Dict[str, np.ndarray] = {}
    add_state_dict_tensors(ts1_all, "encoder", s1_all15796["encoder_state_dict"])
    add_scaler_tensors(ts1_all, "scaler_all15796", s1_all15796["scaler_all15796"])
    out_s1_all_path = WEIGHTS_DIR / "stage1_full_s3_reg_all15796_encoder.safetensors"
    save_safetensors_np(ts1_all, str(out_s1_all_path))

    ts1_11121: Dict[str, np.ndarray] = {}
    add_state_dict_tensors(ts1_11121, "encoder", s1_11121["encoder_state_dict"])
    add_scaler_tensors(ts1_11121, "scaler_policy_b", s1_11121["scaler_policy_b"])
    out_s1_11121_path = WEIGHTS_DIR / "full_s3_reg_7split_pretrained_encoder.safetensors"
    save_safetensors_np(ts1_11121, str(out_s1_11121_path))

    # 5. Build `sos_hurdle_and_sp_critic_models.safetensors`
    tcrit: Dict[str, np.ndarray] = {}
    add_critic_tensors(tcrit, "critic", critics_ckpt)
    tcrit["w1_proj_412x16"] = np.ascontiguousarray(np.asarray(critics_ckpt["w1_proj_412x16"], dtype=np.float32))
    out_crit_path = WEIGHTS_DIR / "sos_hurdle_and_sp_critic_models.safetensors"
    save_safetensors_np(tcrit, str(out_crit_path))

    # 6. Build `golden_inference_fixture_context0.safetensors`
    tg: Dict[str, np.ndarray] = {}
    raw_in = golden["inputs"]["raw_problem_and_landmark_features"]
    for k, v in raw_in.items():
        arr = np.asarray(v)
        if arr.dtype == bool:
            arr = arr.astype(np.float32)
        tg[f"raw_ctx.{k}"] = np.ascontiguousarray(arr)

    std_in = golden["inputs"]["standardized_context_tensors"]
    for k, v in std_in.items():
        arr = np.asarray(v, dtype=np.float32)
        tg[f"std_ctx.{k}"] = np.ascontiguousarray(arr)

    pool_in = golden["inputs"]["candidate_pool_features"]
    for k in (
        "program_412",
        "proj16",
        "sem24",
        "cand_q_32",
        "eff_prior",
        "r882_total_cost",
        "r882_anchor_prior",
        "r882_anchor_total",
        "initial_probe_idx4",
        "candidate_fitness",
        "candidate_logratio",
        "candidate_gate",
        "candidate_utility",
    ):
        arr = np.asarray(pool_in[k])
        if arr.dtype == bool:
            arr = arr.astype(np.float32)
        tg[f"pool.{k}"] = np.ascontiguousarray(arr)

    for k in ("cand_feats_64", "sos_prior_2d", "sp_prior_1d", "h_cost"):
        arr = np.asarray(pool_in[k])
        tg[f"expected_critic.{k}"] = np.ascontiguousarray(arr)

    soup_out = golden["expected_outputs_single_soup_model"]
    for k in (
        "ctx_tokens_92x32",
        "ctx_mask_92",
        "scale_feat_8",
        "attn_ctx_32",
        "bilinear_24",
        "delta_prior_vec",
        "rev4_indices",
        "cost_act",
        "delta_traj_vec",
        "cost_final",
        "top4_indices",
    ):
        tg[f"expected_soup.{k}"] = np.ascontiguousarray(np.asarray(soup_out[k]))
    tg["expected_soup.pick1_index"] = np.array([int(soup_out["pick1_index"])], dtype=np.int64)
    tg["expected_soup.pick1_utility"] = np.array([float(soup_out["pick1_utility"])], dtype=np.float64)
    tg["expected_soup.hyperparams"] = np.array(
        [float(soup_out["alpha_prior"]), float(soup_out["alpha_traj"])], dtype=np.float64
    )

    ens_out = golden["expected_outputs_5fold_ensemble"]
    for k in (
        "delta_prior_vec",
        "rev4_indices",
        "cost_act",
        "delta_traj_vec",
        "cost_final",
        "top4_indices",
    ):
        tg[f"expected_ens.{k}"] = np.ascontiguousarray(np.asarray(ens_out[k]))
    tg["expected_ens.pick1_index"] = np.array([int(ens_out["pick1_index"])], dtype=np.int64)
    tg["expected_ens.pick1_utility"] = np.array([float(ens_out["pick1_utility"])], dtype=np.float64)
    tg["expected_ens.hyperparams"] = np.array(
        [float(ens_out["alpha_prior"]), float(ens_out["alpha_traj"])], dtype=np.float64
    )

    cid = golden["context_metadata"]["context_sha256"]
    fam_id = golden["context_metadata"]["family_id"]
    device = torch.device("cpu")

    s3_scaler = pcm.FoldScaler(
        dope629_mean=t1219["s3_scaler.dope629_mean"],
        dope629_std=t1219["s3_scaler.dope629_std"],
        hidden256_mean=t1219["s3_scaler.hidden256_mean"],
        hidden256_std=t1219["s3_scaler.hidden256_std"],
        col18_mean=t1219["s3_scaler.col18_mean"],
        col18_std=t1219["s3_scaler.col18_std"],
    )
    s3_bridge = pcm.PreCondenseResidualBridge(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
    )
    s3_bridge.load_state_dict(s1_11121["encoder_state_dict"], strict=False)
    s3_bridge.eval()
    with torch.no_grad():
        ct_s3 = s3_scaler.transform_context(raw_in, device)
        ctx_tokens_s3, ctx_mask_s3, scale_8_s3 = s3_bridge.encode_context_tokens(ct_s3)
        mask_exp = ctx_mask_s3.unsqueeze(-1)
        mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
        ctx_mean = (ctx_tokens_s3 * mask_exp).sum(dim=1) / mask_sum
        ctx_masked_for_max = torch.where(
            mask_exp > 0.5, ctx_tokens_s3, torch.full_like(ctx_tokens_s3, -1e4)
        )
        ctx_max = ctx_masked_for_max.max(dim=1).values
        ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)
        u_ctx = s3_bridge.ctx_bilinear_u(ctx_summary_64)
        s3_ctx_vec16 = (
            torch.cat([u_ctx[0, :12], scale_8_s3[0, :4]], dim=0).cpu().numpy().astype(np.float32)
        )

    bridge_f_soup = pcm.PreCondenseResidualBridge(arm="F_FULL629_HIDDEN256_COL64X17", d_token=32, n_heads=4)
    bridge_f_soup.load_state_dict(p1310["stage2_bridge_arm_f_soup_state_dict"], strict=True)
    bridge_f_soup.eval()

    bridge_g_soup = pcm.PreCondenseResidualBridge(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
    )
    bridge_g_soup.load_state_dict(p1310["stage2_bridge_arm_g_soup_state_dict"], strict=True)
    bridge_g_soup.eval()

    ct_val_t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in std_in.items()}
    cand_q_t = torch.as_tensor(pool_in["cand_q_32"], dtype=torch.float32).unsqueeze(0)
    sem24_t = torch.as_tensor(pool_in["sem24"], dtype=torch.float32).unsqueeze(0)
    gate_risk_t = torch.as_tensor(pool_in["sos_prior_2d"][:, 1], dtype=torch.float32).unsqueeze(0)
    sp_prior_t = torch.as_tensor(pool_in["sp_prior_1d"], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        dp_f_t, _, _ = bridge_f_soup.forward_prior_and_context(
            ct_val_t, cand_q_t, sem24_t, gate_risk_t, sp_prior_t
        )
        dp_g_t, _, _ = bridge_g_soup.forward_prior_and_context(
            ct_val_t, cand_q_t, sem24_t, gate_risk_t, sp_prior_t
        )
    dp_f_full = dp_f_t[0].cpu().numpy().astype(np.float64)
    dp_g_full = dp_g_t[0].cpu().numpy().astype(np.float64)

    pkt = prod256.build_enriched_topk_packet_for_context(
        cid=cid,
        family_id=fam_id,
        rcost=np.asarray(pool_in["r882_total_cost"], dtype=np.float64),
        aprior=np.asarray(pool_in["r882_anchor_prior"], dtype=np.float64),
        prog_full=np.asarray(pool_in["program_412"], dtype=np.float32),
        sem24_full=np.asarray(pool_in["sem24"], dtype=np.float32),
        proj16_full=np.asarray(pool_in["proj16"], dtype=np.float32),
        fitness_full=np.asarray(pool_in["candidate_fitness"], dtype=np.float64),
        gate_full=np.asarray(pool_in["candidate_gate"], dtype=bool),
        utility_full=np.asarray(pool_in["candidate_utility"], dtype=np.float64),
        raw_ctx=raw_in,
        s3_ctx_vec16=s3_ctx_vec16,
        dp_f_full=dp_f_full,
        dp_g_full=dp_g_full,
        h_cost_full=np.asarray(pool_in["h_cost"], dtype=np.float64),
        gate_risk_full=np.asarray(pool_in["sos_prior_2d"][:, 1], dtype=np.float64),
        sp_prior_full=np.asarray(pool_in["sp_prior_1d"], dtype=np.float64),
        k=8,
    )

    sc_1310, pinv_1310, pick_1310, _, det_1310 = compute_production_top8_predictions_from_tensors(
        pkt["X_568"], pkt["X_616"], t1310
    )
    sc_1219, pinv_1219, pick_1219, _, det_1219 = compute_production_top8_predictions_from_tensors(
        pkt["X_568"], pkt["X_616"], t1219
    )

    tg["expected_top8.top8_indices"] = np.ascontiguousarray(np.asarray(pkt["top8_indices"], dtype=np.int64))
    tg["expected_top8.s3_ctx_vec16"] = np.ascontiguousarray(s3_ctx_vec16)
    tg["expected_top8.dp_f_full"] = np.ascontiguousarray(dp_f_full)
    tg["expected_top8.dp_g_full"] = np.ascontiguousarray(dp_g_full)
    tg["expected_top8.X_568"] = np.ascontiguousarray(pkt["X_568"])
    tg["expected_top8.X_616"] = np.ascontiguousarray(pkt["X_616"])
    tg["expected_top8.p1310_nn_scores"] = np.ascontiguousarray(det_1310["nn_scores"])
    tg["expected_top8.p1310_nn_inv_probs"] = np.ascontiguousarray(det_1310["nn_inv_probs"])
    tg["expected_top8.p1310_lgbm_ranker_568"] = np.ascontiguousarray(det_1310["lgbm_ranker_568"])
    tg["expected_top8.p1310_lgbm_ranker_580"] = np.ascontiguousarray(det_1310["lgbm_ranker_580"])
    tg["expected_top8.p1310_lgbm_ranker_616"] = np.ascontiguousarray(det_1310["lgbm_ranker_616"])
    tg["expected_top8.p1310_lgbm_clf_568"] = np.ascontiguousarray(det_1310["lgbm_clf_568"])
    tg["expected_top8.p1310_lgbm_clf_580"] = np.ascontiguousarray(det_1310["lgbm_clf_580"])
    tg["expected_top8.p1310_lgbm_clf_616"] = np.ascontiguousarray(det_1310["lgbm_clf_616"])
    tg["expected_top8.p1310_bagged_scores"] = np.ascontiguousarray(sc_1310)
    tg["expected_top8.p1310_bagged_pinv"] = np.ascontiguousarray(pinv_1310)
    tg["expected_top8.p1310_static_pick_j"] = np.ascontiguousarray(pick_1310)
    tg["expected_top8.p1219_bagged_scores"] = np.ascontiguousarray(sc_1219)
    tg["expected_top8.p1219_bagged_pinv"] = np.ascontiguousarray(pinv_1219)
    tg["expected_top8.p1219_static_pick_j"] = np.ascontiguousarray(pick_1219)

    out_golden_path = WEIGHTS_DIR / "golden_inference_fixture_context0.safetensors"
    save_safetensors_np(tg, str(out_golden_path))

    all_exported = [
        (out_1310_path, t1310, "100% Rust .safetensors bundle for the full 1,310-context + 15,796-S3 regression production model."),
        (out_1219_path, t1219, "100% Rust .safetensors bundle for the holdout-clean 1,219 DL-Train + 11,121 Policy-B S3 regression model."),
        (out_s2_path, ts2, "100% Rust .safetensors bundle for the Stage-2 PreCondenseResidualBridge soup model + 5-fold ensemble + scalers + critics."),
        (out_s1_all_path, ts1_all, "100% Rust .safetensors bundle for the Stage-1 all-15,796 S3 7-split regression 92-token encoder."),
        (out_s1_11121_path, ts1_11121, "100% Rust .safetensors bundle for the Stage-1 Policy-B 11,121 S3 7-split regression 92-token encoder."),
        (out_crit_path, tcrit, "100% Rust .safetensors bundle for the fitted Decomposed Hurdle / SOS & Self-Play linear critics + w1_proj_412x16."),
        (out_golden_path, tg, "End-to-end golden inference fixture (.safetensors) for 100% Rust parity verification."),
    ]

    if CKPT_DIR.exists():
        for p, _, _ in all_exported:
            target = CKPT_DIR / p.name
            target.write_bytes(p.read_bytes())

    for p, _, _ in all_exported:
        assert p.stat().st_size <= 5 * 1024 * 1024, f"{p.name} exceeds 5 MiB cap: {p.stat().st_size}"
        loaded = load_safetensors_np(str(p))
        assert len(loaded) > 0
        for name, arr in loaded.items():
            assert np.all(np.isfinite(arr)), f"Non-finite values in {p.name}:{name}"

    manifest_path = WEIGHTS_DIR / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}

    manifest["rust_safetensors_inference_bundles"] = {
        p.name: {
            "path": f"research/flywheel_selfplay/h256_pre_condense_residual_flywheel/weights/{p.name}",
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "tensor_count": len(tdict),
            "description": desc,
        }
        for p, tdict, desc in all_exported
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if (CKPT_DIR / "checkpoint_manifest.json").exists():
        (CKPT_DIR / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for p, tdict, _ in all_exported:
        print(
            f"[OK] {p.name}: {p.stat().st_size:,} bytes ({p.stat().st_size / (1024*1024):.2f} MiB), "
            f"{len(tdict)} tensors, sha256={sha256_file(p)}"
        )


if __name__ == "__main__":
    main()
