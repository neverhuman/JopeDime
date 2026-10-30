#!/usr/bin/env python3
"""
Fused Single-Network GPU Inference & GP Fitter (`FusedJopeNetwork`)
===================================================================
Implements a single end-to-end `torch.nn.Module` (`FusedJopeNetwork`) executing:
  raw table on GPU -> optional GPU landmarks -> DOPE-MLC-v3 (629D) ->
  Stage 1B SVD tokens (10x32) -> GPU candidate slate & warm-start anchors ->
  BlockFormer (h40b / h45) + GameNet + QHat (93D) + batched sync-free SPQ-1 ->
  winner program selection -> batched GPU GP fit (weights w, b, predictions, R^2)

Key guarantees:
  1. Single `nn.Module`, single `forward()` call.
  2. Zero host round-trips (`0` D2H copies / `.item()` / `.cpu()` syncs) inside `forward()`.
  3. Supports `use_landmarks=True` and `use_landmarks=False` (optional landmarks).
  4. Supports multi-table batching (`B >= 1`) with masked variable-size tables `[B, N_max, P_max]`.
  5. Supports `fp32`, `fp16`, and `bf16` execution, `torch.cuda.CUDAGraph`, and `torch.compile`.
  6. Supports both `exact_ridge` (multi-transform + GCV grid GPU fitter) and
     `gpu_fast_fitter_approximate_track` (labeled approximate Ridge surrogate).
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MICRO_ROOT = Path("/home/ubuntu/JopeDimeMicro")
MICRO_PYTHON = MICRO_ROOT / "python"
if str(MICRO_PYTHON) not in sys.path:
    sys.path.insert(0, str(MICRO_PYTHON))

from model import FlywheelGameNet
from phase12_multimodal_model import (
    FlywheelBlockFormerAdapterNet,
    _compute_slate_foldsafe_sos_prior,
    compute_semantic_struct_sig_24d,
    compute_semantic_struct_sig_24d_torch,
    encode_operator_onehot,
)
from research.levels.multilevel_model import MultiLevelLandmarkModel
from rust_evaluator import get_w1_sign_matrix

EXPANDED_BANK_PATH = Path(__file__).resolve().parent / "expanded_gpu_bank_1766.pt"
_GLOBAL_BANK_DSLS: List[str] = []
_GLOBAL_DSL_TO_IDX: Dict[str, int] = {}


def _init_global_bank_registry() -> None:
    global _GLOBAL_BANK_DSLS, _GLOBAL_DSL_TO_IDX
    if _GLOBAL_BANK_DSLS:
        return
    if EXPANDED_BANK_PATH.exists():
        b_data = torch.load(EXPANDED_BANK_PATH, map_location="cpu", weights_only=False)
        all_dsls = b_data.get("all_registered_dsls") or b_data["bank_dsls"]
        _GLOBAL_BANK_DSLS = list(all_dsls)
        _GLOBAL_DSL_TO_IDX = {d: idx for idx, d in enumerate(_GLOBAL_BANK_DSLS)}


def register_dsls_in_global_bank(dsls: List[str], persist: bool = True) -> List[int]:
    """
    Registers candidate DSL strings in the shared global bank registry so that
    `top1_bank_idx` maps deterministically to `net.bank_dsls[idx]` across all
    `FusedJopeNetwork` instances and processes.
    """
    _init_global_bank_registry()
    added = False
    indices: List[int] = []
    for d in dsls:
        d_clean = str(d).strip()
        idx = _GLOBAL_DSL_TO_IDX.get(d_clean)
        if idx is None:
            idx = len(_GLOBAL_BANK_DSLS)
            _GLOBAL_BANK_DSLS.append(d_clean)
            _GLOBAL_DSL_TO_IDX[d_clean] = idx
            added = True
        indices.append(idx)
    if added and persist and EXPANDED_BANK_PATH.exists():
        try:
            b_data = torch.load(EXPANDED_BANK_PATH, map_location="cpu", weights_only=False)
            b_data["all_registered_dsls"] = list(_GLOBAL_BANK_DSLS)
            tmp_p = EXPANDED_BANK_PATH.with_suffix(".pt.tmp")
            torch.save(b_data, tmp_p)
            tmp_p.replace(EXPANDED_BANK_PATH)
        except Exception:
            pass
    return indices


STEP3_GATE_KEYS_14: Tuple[str, ...] = (
    "asinh_atan_w",
    "copula_retention",
    "het_scale_w",
    "hinge_w",
    "monotone_w",
    "neural_w",
    "noise_scale",
    "pair_w",
    "periodic_fract_w",
    "regime_sharpness",
    "rff_w",
    "ridge_alpha",
    "row_summary_w",
    "tree_w",
)


def parse_dsl_to_op_descriptor(dsl: str) -> List[float]:
    """
    Encodes a candidate DSL string into an 8-float GPU operator descriptor for
    branch-free batched GPU feature transformation and GP fitting:
      [0] winsor_q       (0.0 if absent, else clipping quantile in [0.01, 0.25])
      [1] winsor_append  (1.0 if append=true on winsor_clip else 0.0)
      [2] has_robust     (1.0 if robust_scale present else 0.0)
      [3] robust_append  (1.0 if robust_scale append=true else 0.0)
      [4] nonlin_type    (0=none, 1=sigmoid, 2=norm_row, 3=cheb, 4=hinge, 5=poly_interact, 6=pair_prod, 7=rank/blom)
      [5] nonlin_param   (degree / knots / topk / bins parameter)
      [6] nonlin_append  (1.0 if nonlinear op has append=true else 0.0)
      [7] head_type      (0=ridge, 1=ridge_gcv, 2=elasticnet, 3=histgbm, 4=ksmooth)
    """
    s = str(dsl).strip()
    winsor_q = 0.0
    winsor_app = 0.0
    if "winsor_clip" in s:
        winsor_q = 0.05
        for part in s.split("(winsor_clip")[1:]:
            seg = part.split(")")[0]
            if "append=true" in seg:
                winsor_app = 1.0
            if "q=" in seg:
                try:
                    q_str = seg.split("q=")[1].split()[0].rstrip(")")
                    winsor_q = float(q_str)
                except Exception:
                    winsor_q = 0.05
            break

    has_robust = 1.0 if "robust_scale" in s else 0.0
    robust_app = 0.0
    if has_robust > 0:
        seg = s.split("(robust_scale")[1].split(")")[0]
        if "append=true" in seg:
            robust_app = 1.0

    nonlin_type = 0.0
    nonlin_param = 0.0
    nonlin_app = 0.0
    if "(sigmoid" in s:
        nonlin_type = 1.0
        seg = s.split("(sigmoid")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
    elif "(norm_row" in s:
        nonlin_type = 2.0
        seg = s.split("(norm_row")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
    elif "(cheb" in s:
        nonlin_type = 3.0
        seg = s.split("(cheb")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
        nonlin_param = 3.0
        if "degree=" in seg:
            try:
                nonlin_param = float(seg.split("degree=")[1].split()[0].rstrip(")"))
            except Exception:
                nonlin_param = 3.0
    elif "(hinge" in s:
        nonlin_type = 4.0
        seg = s.split("(hinge")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
        nonlin_param = 5.0
        if "knots=" in seg:
            try:
                nonlin_param = float(seg.split("knots=")[1].split()[0].rstrip(")"))
            except Exception:
                nonlin_param = 5.0
    elif "(poly_interact" in s:
        nonlin_type = 5.0
        seg = s.split("(poly_interact")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
    elif "(pair_prod" in s:
        nonlin_type = 6.0
        seg = s.split("(pair_prod")[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0
        nonlin_param = 8.0
        if "topk=" in seg:
            try:
                nonlin_param = float(seg.split("topk=")[1].split()[0].rstrip(")"))
            except Exception:
                nonlin_param = 8.0
    elif "(rank" in s or "(blom" in s:
        nonlin_type = 7.0
        key = "(rank" if "(rank" in s else "(blom"
        seg = s.split(key)[1].split(")")[0]
        nonlin_app = 1.0 if "append=true" in seg else 0.0

    head_type = 0.0
    if "(ridge_gcv" in s:
        head_type = 1.0
    elif "(elasticnet" in s:
        head_type = 2.0
    elif "(histgbm" in s:
        head_type = 3.0
    elif "(ksmooth" in s:
        head_type = 4.0

    return [
        winsor_q,
        winsor_app,
        has_robust,
        robust_app,
        nonlin_type,
        nonlin_param,
        nonlin_app,
        head_type,
    ]


class Stage1ADistilledBridge(nn.Module):
    """
    Sync-free GPU bridge mapping batched GPU landmark / moment descriptors
    onto the canonical 256D `vec256` space expected by `dope_mlc_v3_multitask_best`
    and Stage 1B SVD projections.
    Can be initialized analytically (identity) and/or ridge-distilled on real
    training tables (`strict_train_fit_881` / S3 train corpus).
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("blend_alpha", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, raw_desc_256: torch.Tensor) -> torch.Tensor:
        projected = self.proj(raw_desc_256)
        return torch.lerp(raw_desc_256, projected, self.blend_alpha)


class FusedJopeNetwork(nn.Module):
    """
    Single Fused Deep Learning Network:
      raw table on GPU -> optional GPU landmarks -> DOPE encoder (629D) ->
      10x32 SVD tokens -> router/QHat/SPQ-1 -> winner program -> GPU GP fit.

    Zero host round-trips inside `forward()`.
    """

    def __init__(
        self,
        *,
        micro_root: Path = MICRO_ROOT,
        adapter_variant: str = "h40b",
        device: Optional[torch.device] = None,
        max_fit_basis_dim: int = 192,
    ) -> None:
        super().__init__()
        self.micro_root = Path(micro_root)
        self.models_dir = self.micro_root / "assets" / "models"
        self.proj_dir = self.micro_root / "assets" / "projections"
        self.adapter_variant = adapter_variant
        self.max_fit_basis_dim = int(max_fit_basis_dim)
        self.step3_gate_keys_14 = STEP3_GATE_KEYS_14
        dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1. Load frozen Stage 1 & Stage 2 parameters (scalers, SVD bases, QHat, priors, candidate bank)
        frozen_path = self.proj_dir / "frozen_stage1_stage2_inference_params_v3.pt"
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        sc = frozen["scalers"]
        sb = frozen["svd_bases"]

        self.register_buffer(
            "step2_lm_mean_1x256",
            torch.as_tensor(sc["step2_lm_mean_1x256"], dtype=torch.float32).reshape(1, 256),
        )
        self.register_buffer(
            "step2_lm_std_1x256",
            torch.as_tensor(sc["step2_lm_std_1x256"], dtype=torch.float32).reshape(1, 256).clamp_min(1e-6),
        )
        self.register_buffer(
            "step2_svd_basis_256x48",
            torch.as_tensor(sc["step2_svd_basis_256x48"], dtype=torch.float32).reshape(256, 48),
        )
        self.register_buffer(
            "lm80_mu",
            torch.as_tensor(sc["lm80_mu"], dtype=torch.float32).reshape(1, 80),
        )
        self.register_buffer(
            "lm80_sd",
            torch.as_tensor(sc["lm80_sd"], dtype=torch.float32).reshape(1, 80).clamp_min(1e-6),
        )
        self.register_buffer(
            "mu_lm48",
            torch.as_tensor(sc["mu_lm48"], dtype=torch.float32).reshape(1, 48),
        )
        self.register_buffer(
            "sd_lm48",
            torch.as_tensor(sc["sd_lm48"], dtype=torch.float32).reshape(1, 48).clamp_min(1e-6),
        )
        self.register_buffer(
            "mu_bb20",
            torch.as_tensor(sc["mu_bb20"], dtype=torch.float32).reshape(1, 20),
        )
        self.register_buffer(
            "sd_bb20",
            torch.as_tensor(sc["sd_bb20"], dtype=torch.float32).reshape(1, 20).clamp_min(1e-6),
        )
        self.register_buffer(
            "mu_dm24",
            torch.as_tensor(sc["mu_dm24"], dtype=torch.float32).reshape(1, 24),
        )
        self.register_buffer(
            "sd_dm24",
            torch.as_tensor(sc["sd_dm24"], dtype=torch.float32).reshape(1, 24).clamp_min(1e-6),
        )
        prog_mean_np = np.asarray(sc["prog_mean_412"], dtype=np.float32)
        prog_std_np = np.asarray(sc["prog_std_412"], dtype=np.float32)
        self.register_buffer(
            "prog_mean_412",
            torch.as_tensor(prog_mean_np, dtype=torch.float32).reshape(1, 1, 412),
        )
        self.register_buffer(
            "prog_std_412",
            torch.as_tensor(prog_std_np, dtype=torch.float32).reshape(1, 1, 412).clamp_min(1e-6),
        )

        # Register Stage 1B SVD slot buffers (`mu`, `sd_safe`, `active_mask`, `effective_proj_mat`)
        svd_slot_names = [
            "svd_slot0_m0",
            "svd_slot1_m1",
            "svd_slot2_m2",
            "svd_slot3_dm16",
            "svd_slot3_in16",
            "svd_slot4_d629_mid",
            "svd_slot5_d629_tail",
            "svd_slot6_d280_res",
            "svd_slot7_lm_fine",
            "svd_slot8_v3_full92",
            "svd_slot9_kron",
        ]
        self.svd_slot_names = svd_slot_names
        for slot_name in svd_slot_names:
            d_slot = sb[slot_name]
            self.register_buffer(
                f"{slot_name}_mu",
                torch.as_tensor(d_slot["mu"], dtype=torch.float32),
            )
            self.register_buffer(
                f"{slot_name}_sd_safe",
                torch.as_tensor(d_slot["sd_safe"], dtype=torch.float32).clamp_min(1e-6),
            )
            self.register_buffer(
                f"{slot_name}_active_mask",
                torch.as_tensor(d_slot["active_mask"], dtype=torch.float32),
            )
            self.register_buffer(
                f"{slot_name}_effective_proj_mat",
                torch.as_tensor(d_slot["effective_proj_mat"], dtype=torch.float32),
            )

        # Register QHat 93D linear head for `dope_self_contained_10tok`
        qm = frozen["qhat_models"]["dope_self_contained_10tok"]
        self.register_buffer(
            "qhat_mu_x_93",
            torch.as_tensor(qm["mu_x_93"], dtype=torch.float32).reshape(1, 1, 93),
        )
        self.register_buffer(
            "qhat_sd_x_93",
            torch.as_tensor(qm["sd_x_93"], dtype=torch.float32).reshape(1, 1, 93).clamp_min(1e-6),
        )
        self.register_buffer(
            "qhat_w_93",
            torch.as_tensor(qm["w_93"], dtype=torch.float32).reshape(93),
        )
        self.register_buffer(
            "qhat_b_0",
            torch.tensor(float(qm["b_0"]), dtype=torch.float32),
        )

        # Register 150D FoldSafeCalibratedPriorEstimator parameters (`dope_self_contained_10tok` and `fp_complete_4tok`)
        pe_10 = frozen["prior_estimators"]["dope_self_contained_10tok"]
        self.register_buffer(
            "prior10_feat_mean_150",
            torch.as_tensor(pe_10["feat_mean"], dtype=torch.float32).reshape(1, 1, 150),
        )
        self.register_buffer(
            "prior10_feat_std_150",
            torch.as_tensor(pe_10["feat_std"], dtype=torch.float32).reshape(1, 1, 150).clamp_min(1e-6),
        )
        self.register_buffer(
            "prior10_w_150",
            torch.as_tensor(pe_10["w"], dtype=torch.float32).reshape(150),
        )
        self.register_buffer(
            "prior10_default_median",
            torch.tensor(float(pe_10["default_median_prior"]), dtype=torch.float32),
        )

        pe_fp = frozen["prior_estimators"]["fp_complete_4tok"]
        self.register_buffer(
            "priorfp_feat_mean_150",
            torch.as_tensor(pe_fp["feat_mean"], dtype=torch.float32).reshape(1, 1, 150),
        )
        self.register_buffer(
            "priorfp_feat_std_150",
            torch.as_tensor(pe_fp["feat_std"], dtype=torch.float32).reshape(1, 1, 150).clamp_min(1e-6),
        )
        self.register_buffer(
            "priorfp_w_150",
            torch.as_tensor(pe_fp["w"], dtype=torch.float32).reshape(150),
        )
        self.register_buffer(
            "priorfp_default_median",
            torch.tensor(float(pe_fp["default_median_prior"]), dtype=torch.float32),
        )

        # Build GPU candidate bank (4 v3_bound + 881 champ_templates + 881 EditZero mutations = 1766 GPU bank candidates)
        _init_global_bank_registry()
        if EXPANDED_BANK_PATH.exists():
            exp_bank = torch.load(EXPANDED_BANK_PATH, map_location="cpu", weights_only=False)
            self.bank_dsls = _GLOBAL_BANK_DSLS
            self.bank_pids = list(exp_bank["bank_pids"])
            self.bank_ops = list(exp_bank["bank_ops"])
            self.register_buffer("bank_feat412", exp_bank["bank_feat412"].to(torch.float32))
            self.register_buffer("bank_proj16", exp_bank["bank_proj16"].to(torch.float32))
            self.register_buffer("bank_sem24", exp_bank["bank_sem24"].to(torch.float32))
            self.register_buffer("bank_lm48_z_881", exp_bank["bank_lm48_z_881"].to(torch.float32))
            self.register_buffer("bank_op_desc", exp_bank["bank_op_desc"].to(torch.float32))
            self.register_buffer("bank_op_onehot", exp_bank["bank_op_onehot"].to(torch.float32))
            self.register_buffer("bank_ns_mask_flt", exp_bank["bank_ns_mask_flt"].to(torch.float32))
            self.register_buffer("bank_sos_2", exp_bank["bank_sos_2"].to(torch.float32))
        else:
            v3_cands = frozen["v3_bound_admissible_candidates"]
            champ_881 = frozen["champ_templates_881"]
            self.bank_dsls = []
            self.bank_pids = []
            self.bank_ops = []
            bank_feat412_list: List[np.ndarray] = []
            bank_proj16_list: List[np.ndarray] = []
            bank_lm48_z_list: List[np.ndarray] = []
            bank_op_desc_list: List[List[float]] = []
            bank_op_onehot_list: List[np.ndarray] = []
            bank_ns_mask_list: List[bool] = []

            for c in v3_cands:
                dsl_s = str(c["dsl"])
                op_s = str(c.get("operator", "v3_bound_template"))
                pid_s = str(c.get("program_sha256") or hashlib.sha256(dsl_s.encode("utf-8")).hexdigest())
                self.bank_dsls.append(dsl_s)
                self.bank_pids.append(pid_s)
                self.bank_ops.append(op_s)
                bank_feat412_list.append(np.asarray(c["feat412"], dtype=np.float32))
                bank_proj16_list.append(np.asarray(c["proj16"], dtype=np.float32))
                bank_lm48_z_list.append(np.zeros(48, dtype=np.float32))
                bank_op_desc_list.append(parse_dsl_to_op_descriptor(dsl_s))
                bank_op_onehot_list.append(encode_operator_onehot(op_s).astype(np.float32))
                bank_ns_mask_list.append("crossover" in op_s.lower() or "champion" in op_s.lower())

            for c in champ_881:
                dsl_s = str(c["dsl"])
                op_s = "strict_train_881_knn_champion"
                pid_s = str(c.get("program_sha256") or hashlib.sha256(dsl_s.encode("utf-8")).hexdigest())
                self.bank_dsls.append(dsl_s)
                self.bank_pids.append(pid_s)
                self.bank_ops.append(op_s)
                bank_feat412_list.append(np.asarray(c["feat412"], dtype=np.float32))
                bank_proj16_list.append(np.asarray(c["proj16"], dtype=np.float32))
                lm_z = np.asarray(c["lm48_z"], dtype=np.float32)
                norm_lm = max(float(np.linalg.norm(lm_z)), 1e-6)
                bank_lm48_z_list.append(lm_z / norm_lm)
                bank_op_desc_list.append(parse_dsl_to_op_descriptor(dsl_s))
                bank_op_onehot_list.append(encode_operator_onehot(op_s).astype(np.float32))
                bank_ns_mask_list.append(True)

            bank_feat412_np = np.stack(bank_feat412_list, axis=0)
            bank_proj16_np = np.stack(bank_proj16_list, axis=0)
            bank_sem24_np = compute_semantic_struct_sig_24d(bank_feat412_np).astype(np.float32)
            bank_lm48_z_np = np.stack(bank_lm48_z_list, axis=0)
            bank_op_desc_np = np.asarray(bank_op_desc_list, dtype=np.float32)
            bank_op_onehot_np = np.stack(bank_op_onehot_list, axis=0)
            bank_ns_mask_np = np.asarray(bank_ns_mask_list, dtype=np.float32)
            bank_sos_np = _compute_slate_foldsafe_sos_prior(
                cand_prog_mat=bank_feat412_np,
                cand_pids=self.bank_pids,
                target_fam="__arbitrary_external_table__",
                sos_index=frozen["sos_index_base"],
            ).astype(np.float32)

            self.register_buffer("bank_feat412", torch.as_tensor(bank_feat412_np, dtype=torch.float32))
            self.register_buffer("bank_proj16", torch.as_tensor(bank_proj16_np, dtype=torch.float32))
            self.register_buffer("bank_sem24", torch.as_tensor(bank_sem24_np, dtype=torch.float32))
            self.register_buffer("bank_lm48_z_881", torch.as_tensor(bank_lm48_z_np[4:], dtype=torch.float32))
            self.register_buffer("bank_op_desc", torch.as_tensor(bank_op_desc_np, dtype=torch.float32))
            self.register_buffer("bank_op_onehot", torch.as_tensor(bank_op_onehot_np, dtype=torch.float32))
            self.register_buffer("bank_ns_mask_flt", torch.as_tensor(bank_ns_mask_np, dtype=torch.float32))
            self.register_buffer("bank_sos_2", torch.as_tensor(bank_sos_np, dtype=torch.float32))

        # Register GPU projection & SOS prior buffers for live AST candidates
        self.register_buffer(
            "w1_sign_matrix_412x16",
            torch.as_tensor(get_w1_sign_matrix(), dtype=torch.float32),
        )
        sos_base = frozen["sos_index_base"]
        s_mat_np = np.asarray(sos_base["struct_mat"], dtype=np.float32)
        s_val_np = np.asarray(sos_base["struct_val_arr"], dtype=np.float32)
        self.register_buffer("sos_struct_mat_256x24", torch.as_tensor(s_mat_np, dtype=torch.float32))
        self.register_buffer("sos_struct_val_256", torch.as_tensor(s_val_np, dtype=torch.float32))
        self.register_buffer(
            "sos_jitter_1x1x256",
            (1e-7 * torch.arange(s_mat_np.shape[0], dtype=torch.float32)).view(1, 1, -1),
        )

        # 2. Load DOPE-MLC-v3 (`MultiLevelLandmarkModel`)
        self.dope_model = MultiLevelLandmarkModel()
        dope_ckpt = torch.load(
            self.models_dir / "dope_mlc_v3_multitask_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        dope_sd = dope_ckpt["model_state_dict"] if "model_state_dict" in dope_ckpt else dope_ckpt
        self.dope_model.load_state_dict(dope_sd, strict=True)
        self.dope_model.eval()
        for p in self.dope_model.parameters():
            p.requires_grad_(False)

        # 3. Load Stage 2 `FlywheelGameNet` anchor
        self.gamenet_anchor = FlywheelGameNet(
            ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4
        )
        gn_ckpt = torch.load(
            self.models_dir / "flywheel_winner_pool_v3a_q25_train_only_430.pt",
            map_location="cpu",
            weights_only=False,
        )
        gn_sd = gn_ckpt["model_state_dict"] if "model_state_dict" in gn_ckpt else gn_ckpt
        self.gamenet_anchor.load_state_dict(gn_sd, strict=True)
        self.gamenet_anchor.eval()
        for p in self.gamenet_anchor.parameters():
            p.requires_grad_(False)

        # 4. Load Stage 2 `FlywheelBlockFormerAdapterNet` (`h40b` or `h45`)
        self.adapter_10tok = FlywheelBlockFormerAdapterNet(
            frozen_anchor=self.gamenet_anchor,
            prog_mean=prog_mean_np,
            prog_std=prog_std_np,
            d_token=32,
            n_modalities=10,
            modality_dropout_p=0.10,
        )
        if adapter_variant == "h45":
            ad_path = self.models_dir / "h45_full_s3_15151_gate_disjoint_all_dl_train_1100_10tok.pt"
        else:
            ad_path = self.models_dir / "h40b_round2_dope_self_contained_10tok.pt"
        ad_ckpt = torch.load(ad_path, map_location="cpu", weights_only=False)
        ad_sd = ad_ckpt["state_dict"] if "state_dict" in ad_ckpt else ad_ckpt.get("model_state_dict", ad_ckpt)
        self.adapter_10tok.load_state_dict(ad_sd, strict=True)
        self.adapter_10tok.eval()
        for p in self.adapter_10tok.parameters():
            p.requires_grad_(False)

        # 5. Stage 1A GPU Landmark Bridge & No-Landmark Distilled Bridge
        self.landmark_bridge = Stage1ADistilledBridge(256, 256)
        self.nolm_bridge = Stage1ADistilledBridge(256, 256)
        self.dope_to_lm48 = nn.Linear(629, 48, bias=True)
        nn.init.zeros_(self.dope_to_lm48.weight)
        nn.init.zeros_(self.dope_to_lm48.bias)
        self.register_buffer("nolm_use_distilled_lm48", torch.tensor(0.0, dtype=torch.float32))

        bridge_ckpt_path = Path(__file__).resolve().parent / "distilled_stage1a_bridge_real_s3.pt"
        if bridge_ckpt_path.exists():
            b_ckpt = torch.load(bridge_ckpt_path, map_location="cpu", weights_only=False)
            b_sd = b_ckpt.get("state_dict", b_ckpt)
            self.load_state_dict(b_sd, strict=False)

        # 6. Pre-registered grids for batched sync-free SPQ-1 and Stage 3 GPU GP Fitter
        self.register_buffer("tau_grid_257", torch.linspace(0.0, 6.0, 257, dtype=torch.float32))
        self.register_buffer("grid_idx_257", torch.arange(257, dtype=torch.long))
        self.register_buffer("frac_13", torch.linspace(0.0, 1.0, 13, dtype=torch.float32))
        self.register_buffer("grid_idx_13", torch.arange(13, dtype=torch.long))

        self.register_buffer(
            "gcv_lambdas",
            torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=torch.float32),
        )
        eye_b = torch.eye(self.max_fit_basis_dim + 1, dtype=torch.float32)
        eye_b[-1, -1] = 1e-6
        self.register_buffer("reg_eye_basis_max", eye_b)
        self.register_buffer(
            "anchor_eye_65",
            torch.eye(65, dtype=torch.float32),
        )
        self.register_buffer(
            "mask_10_base",
            torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0),
        )
        self.register_buffer(
            "anchor_alphas_4",
            torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float32).view(1, 4, 1, 1),
        )
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
            torch.backends.mha.set_fastpath_enabled(False)

        self.to(dev)
        self.eval()
        self._install_sync_free_mha_wrappers()

    def _install_sync_free_mha_wrappers(self) -> None:
        """
        Replaces `nn.MultiheadAttention.forward` on owned MHA instances with a pure
        `F.scaled_dot_product_attention` implementation using the exact same `in_proj_weight`,
        `in_proj_bias`, and `out_proj` parameters (`max_abs_diff ~ 1e-7`), eliminating
        PyTorch's internal `aten::item` CPU mask check inside `multi_head_attention_forward`.
        """
        import types

        def _sync_free_mha_forward(
            mha_self: nn.MultiheadAttention,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, Sq, D = query.shape
            Sk = key.shape[1]
            H = mha_self.num_heads
            Dh = D // H
            w_q, w_k, w_v = mha_self.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = mha_self.in_proj_bias.chunk(3, dim=0)
            Q = F.linear(query, w_q, b_q).view(B, Sq, H, Dh).transpose(1, 2)
            K = F.linear(key, w_k, b_k).view(B, Sk, H, Dh).transpose(1, 2)
            V = F.linear(value, w_v, b_v).view(B, Sk, H, Dh).transpose(1, 2)
            sdpa_mask = None
            if key_padding_mask is not None:
                sdpa_mask = torch.where(key_padding_mask.view(B, 1, 1, Sk), -1e4, 0.0).to(Q.dtype)
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, dropout_p=0.0)
            out = out.transpose(1, 2).reshape(B, Sq, D)
            return mha_self.out_proj(out), None

        for mod in self.modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = types.MethodType(_sync_free_mha_forward, mod)

        def _sync_free_encode_context_and_candidates(
            ad_self: Any,
            mm_tokens_10x32: torch.Tensor,
            mm_mask_10: torch.Tensor,
            prog_412: torch.Tensor,
            sos_prior: torch.Tensor,
            active_modality_override_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            from phase12_multimodal_model import transform_program_features_torch

            eff_mask = mm_mask_10.float()
            if active_modality_override_mask is not None:
                if active_modality_override_mask.ndim == 1:
                    eff_mask = eff_mask * active_modality_override_mask.unsqueeze(0)
                else:
                    eff_mask = eff_mask * active_modality_override_mask
            eff_mask = torch.cat([torch.ones_like(eff_mask[:, :1]), eff_mask[:, 1:]], dim=1)

            gates = torch.sigmoid(ad_self.modality_gate_logits).view(1, ad_self.n_modalities, 1)
            tok = ad_self.modality_norm(mm_tokens_10x32.float() + ad_self.modality_type_emb.unsqueeze(0))
            tok_gated = tok * gates * eff_mask.unsqueeze(-1)

            mask_sum = torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
            ctx_pooled = torch.sum(tok_gated, dim=1) / mask_sum

            sem24 = compute_semantic_struct_sig_24d_torch(prog_412)
            sem24_mean = torch.mean(sem24, dim=1, keepdim=True)
            sem24_centered = sem24 - sem24_mean

            prog_std_412 = transform_program_features_torch(prog_412, ad_self.prog_mean, ad_self.prog_std)
            prog_bot16 = ad_self.prog_412_bottleneck(prog_std_412)

            cand_q = ad_self.cand_query_proj(
                torch.cat([sem24, sem24_centered, prog_bot16, sos_prior.float()], dim=-1)
            )
            key_pad_mask = eff_mask < 0.5
            mm_attn_out, _ = ad_self.mm_cross_attn(
                query=cand_q,
                key=tok_gated,
                value=tok_gated,
                key_padding_mask=key_pad_mask,
                need_weights=False,
            )
            u_ctx = ad_self.ctx_bilinear_u(ctx_pooled).unsqueeze(1)
            v_prog = ad_self.prog_bilinear_v(sem24)
            bilinear_16 = u_ctx * v_prog

            raw_mm_delta = ad_self.mm_prior_out(
                torch.cat([mm_attn_out, bilinear_16, sos_prior.float()], dim=-1)
            ).squeeze(-1)
            raw_mm_centered = raw_mm_delta - torch.mean(raw_mm_delta, dim=-1, keepdim=True)
            mm_prior_adj = torch.tanh(ad_self.mm_prior_scale) * raw_mm_centered
            return ctx_pooled, cand_q, sem24, mm_prior_adj

        def _sync_free_predict_parent_child_delta(
            ad_self: Any,
            mm_tokens_10x32: torch.Tensor,
            mm_mask_10: torch.Tensor,
            parent_sem24: torch.Tensor,
            child_sem24: torch.Tensor,
            op_onehot: torch.Tensor,
            parent_norm_y: torch.Tensor,
            parent_gate: torch.Tensor,
            active_modality_override_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            B, M, _ = child_sem24.shape
            eff_mask = mm_mask_10.float()
            if active_modality_override_mask is not None:
                eff_mask = eff_mask * (
                    active_modality_override_mask.unsqueeze(0)
                    if active_modality_override_mask.ndim == 1
                    else active_modality_override_mask
                )
            eff_mask = torch.cat([torch.ones_like(eff_mask[:, :1]), eff_mask[:, 1:]], dim=1)
            gates = torch.sigmoid(ad_self.modality_gate_logits).view(1, ad_self.n_modalities, 1)
            tok = ad_self.modality_norm(mm_tokens_10x32.float() + ad_self.modality_type_emb.unsqueeze(0))
            tok_gated = tok * gates * eff_mask.unsqueeze(-1)
            ctx_pooled = torch.sum(tok_gated, dim=1) / torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
            ctx_exp = ctx_pooled.unsqueeze(1).expand(B, M, ad_self.d_token)

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
            return ad_self.parent_delta_net(inp).squeeze(-1)

        self.adapter_10tok.encode_multimodal_context_and_candidates = types.MethodType(
            _sync_free_encode_context_and_candidates, self.adapter_10tok
        )
        self.adapter_10tok.predict_parent_child_delta = types.MethodType(
            _sync_free_predict_parent_child_delta, self.adapter_10tok
        )

    @staticmethod
    def _cholesky_solve_sync_free(L: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solves `(L @ L^T) x = rhs` via two cuBLAS triangular solves (`solve_triangular`),
        avoiding PyTorch's `torch.cholesky_solve` MAGMA fallback (`magma_spotrs_batched`),
        which otherwise triggers a host-device sync and fails under `torch.cuda.CUDAGraph`.
        """
        y_tmp = torch.linalg.solve_triangular(L, rhs, upper=False)
        return torch.linalg.solve_triangular(L.transpose(-2, -1), y_tmp, upper=True)

    # =========================================================================
    # STAGE 1A: Batched Sync-Free GPU Landmarks & Fast Table Moments -> DOPE629
    # =========================================================================
    def _extract_stage1a_landmarks_batched_gpu(
        self,
        X_tr: torch.Tensor,
        y_tr_norm: torch.Tensor,
        tr_row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure tensorized batched GPU Stage 1A landmark extractor across `[B, N, P]`.
        Zero `.item()` calls, zero host syncs.
        """
        B, N, P = X_tr.shape
        row_m = tr_row_mask.unsqueeze(-1)  # [B, N, 1]
        col_m = col_mask.unsqueeze(1)  # [B, 1, P]
        valid_m = row_m * col_m  # [B, N, P]

        n_valid = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(2.0)  # [B, 1]
        p_valid = col_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        denom = (n_valid - 1.0).clamp_min(1.0)

        X_clean = torch.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0) * valid_m
        col_mean = X_clean.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        X_diff = (X_clean - col_mean) * valid_m
        col_var = (X_diff ** 2).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        col_std = torch.sqrt(col_var).clamp_min(1e-6)
        X_z = torch.clamp(X_diff / col_std, -8.0, 8.0) * valid_m

        y_c = y_tr_norm * tr_row_mask
        y_mean = y_c.sum(dim=1, keepdim=True) / n_valid
        y_diff = (y_c - y_mean) * tr_row_mask
        y_std = torch.sqrt((y_diff ** 2).sum(dim=1, keepdim=True) / denom).clamp_min(1e-6)
        y_z = (y_diff / y_std) * tr_row_mask

        y_skew = ((y_z ** 3) * tr_row_mask).sum(dim=1) / n_valid.squeeze(-1)
        y_kurt = ((y_z ** 4) * tr_row_mask).sum(dim=1) / n_valid.squeeze(-1) - 3.0
        col_skew = ((X_z ** 3) * valid_m).sum(dim=1) / n_valid
        col_kurt = ((X_z ** 4) * valid_m).sum(dim=1) / n_valid - 3.0

        lin_corr_raw = torch.einsum("bnp,bn->bp", X_z, y_z) / denom
        abs_corr = torch.abs(lin_corr_raw) * col_mask

        if P >= 64:
            top_vals, top_idx = torch.topk(abs_corr, k=64, dim=1)
            idx_exp = top_idx.unsqueeze(1).expand(B, N, 64)
            X_64 = torch.gather(X_z, dim=2, index=idx_exp)
            mask_64 = torch.gather(col_mask, dim=1, index=top_idx)
            skew_64 = torch.gather(col_skew, dim=1, index=top_idx)
            kurt_64 = torch.gather(col_kurt, dim=1, index=top_idx)
        else:
            pad_c = 64 - P
            top_vals_p, top_idx_p = torch.topk(abs_corr, k=P, dim=1)
            idx_exp = top_idx_p.unsqueeze(1).expand(B, N, P)
            X_p = torch.gather(X_z, dim=2, index=idx_exp)
            mask_p = torch.gather(col_mask, dim=1, index=top_idx_p)
            skew_p = torch.gather(col_skew, dim=1, index=top_idx_p)
            kurt_p = torch.gather(col_kurt, dim=1, index=top_idx_p)
            X_64 = F.pad(X_p, (0, pad_c))
            mask_64 = F.pad(mask_p, (0, pad_c))
            skew_64 = F.pad(skew_p, (0, pad_c))
            kurt_64 = F.pad(kurt_p, (0, pad_c))
            top_vals = F.pad(top_vals_p, (0, pad_c))

        valid_64 = row_m * mask_64.unsqueeze(1)

        probes = torch.stack(
            [
                torch.asinh(X_64),
                torch.atan(X_64),
                torch.erf(X_64),
                torch.sign(X_64) * (torch.abs(X_64) + 1e-8).pow(1.0 / 3.0),
                torch.sign(X_64) * torch.log1p(torch.abs(X_64)),
                torch.tanh(X_64),
                torch.sign(X_64) * (1.0 - torch.exp(-torch.abs(X_64))),
                torch.clamp(X_64, -1.5, 1.5),
                torch.tanh(5.0 * X_64),
                F.relu(X_64),
                torch.sin(X_64),
                torch.cos(X_64),
                X_64 ** 2,
                X_64,
            ],
            dim=-1,
        ) * valid_64.unsqueeze(-1)
        pr_mean = probes.sum(dim=1, keepdim=True) / n_valid.view(B, 1, 1, 1)
        pr_diff = (probes - pr_mean) * valid_64.unsqueeze(-1)
        pr_std = torch.sqrt(
            (pr_diff ** 2).sum(dim=1, keepdim=True) / denom.view(B, 1, 1, 1)
        ).clamp_min(1e-6)
        pr_z = pr_diff / pr_std

        probe_corrs = torch.abs(torch.einsum("bnck,bn->bck", pr_z, y_z) / denom.unsqueeze(-1)) * mask_64.unsqueeze(-1)

        skew_feat = torch.tanh(skew_64 / 4.0) * mask_64
        kurt_feat = torch.tanh(kurt_64 / 8.0) * mask_64
        col_lm = torch.cat(
            [
                probe_corrs,
                skew_feat.unsqueeze(-1),
                kurt_feat.unsqueeze(-1),
                mask_64.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, 64, 17]

        X_8 = X_64[:, :, :8]
        inter_prod = (X_8.unsqueeze(3) * X_8.unsqueeze(2)).reshape(B, N, 64) * row_m
        ip_mean = inter_prod.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        ip_diff = (inter_prod - ip_mean) * row_m
        ip_std = torch.sqrt((ip_diff ** 2).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)).clamp_min(1e-6)
        ip_corr = torch.abs(torch.einsum("bnk,bn->bk", ip_diff / ip_std, y_z) / denom)

        row_mean_x = X_64.sum(dim=2) / p_valid.clamp_min(1.0)
        row_l2_x = torch.sqrt((X_64 ** 2).sum(dim=2) / p_valid.clamp_min(1.0) + 1e-8)
        row_max_x = X_64.max(dim=2).values
        row_min_x = X_64.min(dim=2).values
        row_feats = torch.stack([row_mean_x, row_l2_x, row_max_x, row_min_x], dim=-1) * row_m
        rf_mean = row_feats.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        rf_diff = (row_feats - rf_mean) * row_m
        rf_std = torch.sqrt((rf_diff ** 2).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)).clamp_min(1e-6)
        rf_corr = torch.abs(torch.einsum("bnk,bn->bk", rf_diff / rf_std, y_z) / denom)

        log_n = torch.log1p(n_valid.squeeze(-1)) / 10.0
        log_p = torch.log1p(p_valid.squeeze(-1)) / 5.0
        aspect = p_valid.squeeze(-1) / n_valid.squeeze(-1)
        y_skew_c = torch.clamp(y_skew, -5.0, 5.0) / 5.0
        y_kurt_c = torch.clamp(y_kurt, -10.0, 10.0) / 10.0

        probe_mean_14 = probe_corrs.sum(dim=1) / p_valid.clamp_min(1.0)
        probe_max_14 = probe_corrs.max(dim=1).values
        probe_std_14 = torch.sqrt(
            ((probe_corrs - probe_mean_14.unsqueeze(1)) ** 2 * mask_64.unsqueeze(-1)).sum(dim=1)
            / p_valid.clamp_min(1.0)
            + 1e-8
        )

        nonlin_gain_64 = (probe_corrs[:, :, :13].max(dim=-1).values - probe_corrs[:, :, 13]) * mask_64
        meta_5 = torch.stack([log_n, log_p, aspect, y_skew_c, y_kurt_c], dim=-1)
        tail_13 = torch.cat([skew_feat[:, :7], kurt_feat[:, :6]], dim=-1)

        raw_vec256 = torch.cat(
            [
                meta_5,          # 5
                probe_mean_14,   # 14
                probe_max_14,    # 14
                probe_std_14,    # 14
                rf_corr,         # 4
                top_vals,        # 64
                nonlin_gain_64,  # 64
                ip_corr,         # 64
                tail_13,         # 13
            ],
            dim=-1,
        )  # [B, 256]
        raw_vec256 = torch.nan_to_num(raw_vec256, nan=0.0, posinf=0.0, neginf=0.0)
        vec256 = self.landmark_bridge(raw_vec256)
        return vec256, col_lm

    def _extract_stage1a_fast_moments_batched_gpu(
        self,
        X_tr: torch.Tensor,
        y_tr_norm: torch.Tensor,
        tr_row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ultra-fast no-landmark (`use_landmarks=False`) raw table moment & correlation sketch.
        Skips all 14 nonlinear column probes, pairwise interaction tensors, and sorting.
        Zero `.item()` calls, zero host syncs.
        """
        B, N, P = X_tr.shape
        row_m = tr_row_mask.unsqueeze(-1)
        col_m = col_mask.unsqueeze(1)
        valid_m = row_m * col_m

        n_valid = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(2.0)
        p_valid = col_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        denom = (n_valid - 1.0).clamp_min(1.0)

        X_clean = torch.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0) * valid_m
        col_mean = X_clean.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        X_diff = (X_clean - col_mean) * valid_m
        col_std = torch.sqrt((X_diff ** 2).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)).clamp_min(1e-6)
        X_z = torch.clamp(X_diff / col_std, -8.0, 8.0) * valid_m

        y_c = y_tr_norm * tr_row_mask
        y_mean = y_c.sum(dim=1, keepdim=True) / n_valid
        y_diff = (y_c - y_mean) * tr_row_mask
        y_std = torch.sqrt((y_diff ** 2).sum(dim=1, keepdim=True) / denom).clamp_min(1e-6)
        y_z = (y_diff / y_std) * tr_row_mask

        lin_corr = torch.abs(torch.einsum("bnp,bn->bp", X_z, y_z) / denom) * col_mask
        col_skew = ((X_z ** 3) * valid_m).sum(dim=1) / n_valid * col_mask
        col_kurt = (((X_z ** 4) * valid_m).sum(dim=1) / n_valid - 3.0) * col_mask

        if P >= 64:
            c64 = lin_corr[:, :64]
            s64 = torch.tanh(col_skew[:, :64] / 4.0)
            k64 = torch.tanh(col_kurt[:, :64] / 8.0)
            m64 = col_mask[:, :64]
        else:
            pad_c = 64 - P
            c64 = F.pad(lin_corr, (0, pad_c))
            s64 = F.pad(torch.tanh(col_skew / 4.0), (0, pad_c))
            k64 = F.pad(torch.tanh(col_kurt / 8.0), (0, pad_c))
            m64 = F.pad(col_mask, (0, pad_c))

        log_n = torch.log1p(n_valid.squeeze(-1)) / 10.0
        log_p = torch.log1p(p_valid.squeeze(-1)) / 5.0
        aspect = p_valid.squeeze(-1) / n_valid.squeeze(-1)
        c_mean = lin_corr.sum(dim=1) / p_valid.squeeze(-1)
        c_max = lin_corr.max(dim=1).values

        meta_64 = F.pad(torch.stack([log_n, log_p, aspect, c_mean, c_max], dim=-1), (0, 59))
        fast_256 = torch.cat([c64, s64, k64, meta_64], dim=-1)
        vec256 = self.nolm_bridge(fast_256)

        col_lm = torch.cat(
            [
                torch.zeros((B, 64, 13), device=X_tr.device, dtype=X_tr.dtype),
                torch.stack([c64, s64, k64, m64], dim=-1),
            ],
            dim=-1,
        )
        return vec256, col_lm

    def _run_dope_mlc_v3_and_lm48(
        self,
        vec256: torch.Tensor,
        col_lm: torch.Tensor,
        *,
        use_landmarks: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs `MultiLevelLandmarkModel` (`dope_mlc_v3_multitask_best`) and projects `lm48`
        using the exact canonical Stage 1A layout:
          `dope629 = [vec256(256), task_lat(128), dist_lat(128), gates_14(14), lm_pred(23), bb_pred(16), col_gates(64)]`
        Batched across `B` tables with zero host syncs.
        """
        B = vec256.shape[0]
        routed = self.dope_model.router(vec256, level="high")
        task_lat = routed["task_latent"]    # [B, 128]
        dist_lat = routed["dist_latent"]    # [B, 128]
        lm_pred = routed["landmark_pred"]   # [B, 23]
        bb_pred = routed["bb_pred"]         # [B, 16]
        gates_14 = torch.stack([routed["gates"][k] for k in self.step3_gate_keys_14], dim=-1)  # [B, 14]

        # Batched column gates across [B, 64, 128 + 17]
        t_rep = task_lat.unsqueeze(1).expand(B, 64, -1)
        c_pad = col_lm[:, :64, :17].to(dtype=t_rep.dtype)
        raw_g = self.dope_model.target_net.col_gate_head(torch.cat([t_rep, c_pad], dim=-1)).squeeze(-1)
        col_gates_64 = 1.0 + 0.25 * torch.tanh(raw_g)  # [B, 64]

        dope629 = torch.cat(
            [vec256, task_lat, dist_lat, gates_14, lm_pred, bb_pred, col_gates_64],
            dim=-1,
        )  # [B, 629]
        dope629 = torch.nan_to_num(dope629, nan=0.0, posinf=0.0, neginf=0.0)

        if use_landmarks:
            v256_norm = torch.clamp(
                (vec256 - self.step2_lm_mean_1x256) / self.step2_lm_std_1x256,
                -6.0,
                6.0,
            )
            lm48 = v256_norm @ self.step2_svd_basis_256x48  # [B, 48]
        else:
            lm48_distilled = self.dope_to_lm48(dope629.to(torch.float32))
            lm48 = lm48_distilled * self.nolm_use_distilled_lm48
        return dope629, lm48

    # =========================================================================
    # STAGE 1B: Exact Batched `compute_lm_smart_80d_gpu` & 10-Slot SVD Tokens
    # =========================================================================
    def _compute_lm_smart_80d_gpu(self, lm48_t: torch.Tensor) -> torch.Tensor:
        B = lm48_t.shape[0]
        raw_clean = torch.nan_to_num(lm48_t, nan=0.0, posinf=0.0, neginf=0.0)
        r16x3 = raw_clean.view(B, 16, 3)
        pm = r16x3[:, :, 0]
        ps = torch.abs(r16x3[:, :, 1])
        px = r16x3[:, :, 2]

        sp, _ = torch.sort(pm, dim=1)
        q25 = 0.25 * sp[:, 3:4] + 0.75 * sp[:, 4:5]
        med = 0.50 * sp[:, 7:8] + 0.50 * sp[:, 8:9]
        q75 = 0.75 * sp[:, 11:12] + 0.25 * sp[:, 12:13]
        iqr = torch.clamp(q75 - q25, min=0.05)

        rel_dom_16 = torch.clamp((pm - med) / iqr, -6.0, 6.0)
        tgap_16 = torch.clamp((px - pm) / (torch.abs(pm) + ps + 0.05), -6.0, 6.0)
        stab_8 = 0.5 * (tgap_16[:, 0::2] + tgap_16[:, 1::2])

        min_p = sp[:, 0:1]
        max_p = sp[:, -1:]
        best_gap = min_p - med
        top3_spread = sp[:, 2:3] - sp[:, 0:1]
        mean_std = torch.mean(ps, dim=1, keepdim=True)
        sum_8 = torch.cat([min_p, q25, med, q75, max_p, iqr, best_gap, top3_spread + 0.1 * mean_std], dim=1)

        lm80 = torch.cat([raw_clean, rel_dom_16, stab_8, sum_8], dim=1)
        return torch.clamp((lm80 - self.lm80_mu) / self.lm80_sd, -8.0, 8.0)

    def _svd_slot_project(self, X: torch.Tensor, slot_name: str) -> torch.Tensor:
        mu = getattr(self, f"{slot_name}_mu")
        sd_safe = getattr(self, f"{slot_name}_sd_safe")
        active_mask = getattr(self, f"{slot_name}_active_mask")
        proj_mat = getattr(self, f"{slot_name}_effective_proj_mat")
        X_clean = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_z = torch.clamp((X_clean - mu) / sd_safe, -8.0, 8.0) * active_mask
        out = X_z @ proj_mat
        return torch.clamp(out, -6.0, 6.0)

    def _project_stage1b_tokens_batched_gpu(
        self,
        lm48: torch.Tensor,
        dope629: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact batched Stage 1B token projection for `dope_self_contained_10tok`.
        Returns `(tokens_10x32 [B, 10, 32], mask_10 [B, 10], lm48_z [B, 48])`.
        """
        B = dope629.shape[0]
        device = dope629.device

        lm80_z = self._compute_lm_smart_80d_gpu(lm48)
        lm48_z = torch.clamp((lm48 - self.mu_lm48) / self.sd_lm48, -6.0, 6.0)

        a0 = torch.zeros((B, 88), device=device, dtype=torch.float32)
        h27 = torch.zeros((B, 16), device=device, dtype=torch.float32)
        bb20_z = torch.zeros((B, 20), device=device, dtype=torch.float32)
        tok3 = torch.zeros((B, 32), device=device, dtype=torch.float32)
        tok6 = torch.zeros((B, 32), device=device, dtype=torch.float32)
        tok8 = torch.zeros((B, 32), device=device, dtype=torch.float32)

        m0_in = torch.cat([a0, h27, lm80_z], dim=1)  # [B, 184]
        m1_in = torch.cat([lm48_z[:, 36:48], bb20_z, lm80_z[:, 48:64]], dim=1)  # [B, 48]

        tok0 = self._svd_slot_project(m0_in, "svd_slot0_m0")
        tok1 = self._svd_slot_project(m1_in, "svd_slot1_m1")
        tok2 = self._svd_slot_project(dope629, "svd_slot2_m2")
        tok4 = self._svd_slot_project(dope629, "svd_slot4_d629_mid")
        tok5 = self._svd_slot_project(dope629, "svd_slot5_d629_tail")
        tok7 = self._svd_slot_project(m0_in, "svd_slot7_lm_fine")
        kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(B, 64)
        tok9 = self._svd_slot_project(kron_64, "svd_slot9_kron")

        tokens_10x32 = torch.stack(
            [tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8, tok9],
            dim=1,
        )  # [B, 10, 32]
        mask_10 = self.mask_10_base.expand(B, 10).clone()
        return tokens_10x32, mask_10, lm48_z

    # =========================================================================
    # STAGE 2 PREP: Batched GPU Candidate Selection, Priors & Anchor Fits
    # =========================================================================
    def _predict_calibrated_prior_batched_gpu(
        self,
        cand_412: torch.Tensor,
        sem_24: torch.Tensor,
        lm48_z: torch.Tensor,
        *,
        track_prefix: str = "prior10",
    ) -> torch.Tensor:
        """
        Batched GPU evaluation of `FoldSafeCalibratedPriorEstimator` (`150D` Ridge) across `[B, 96]`.
        """
        B, M, _ = cand_412.shape
        device = cand_412.device
        ctx12 = torch.cat(
            [lm48_z[:, 36:42], torch.zeros((B, 6), device=device, dtype=torch.float32)],
            dim=-1,
        )  # [B, 12]
        inter_120 = (sem_24[:, :, :10].unsqueeze(-1) * ctx12.unsqueeze(1).unsqueeze(2)).reshape(B, M, 120)
        pos = F.relu(cand_412)
        cplx_6 = torch.stack(
            [
                torch.log1p(pos[:, :, 23]) / 5.0,
                pos[:, :, 411] / 5.0,
                torch.log1p(pos[:, :, 52]) / 6.0,
                torch.log1p(pos[:, :, 0]),
                pos[:, :, 24:29].sum(dim=-1),
                torch.log1p(pos[:, :, :23].sum(dim=-1)) / 3.0,
            ],
            dim=-1,
        )  # [B, M, 6]
        X_150 = torch.cat([sem_24, cplx_6, inter_120], dim=-1)  # [B, M, 150]
        f_mean = getattr(self, f"{track_prefix}_feat_mean_150")
        f_std = getattr(self, f"{track_prefix}_feat_std_150")
        w = getattr(self, f"{track_prefix}_w_150")
        med = getattr(self, f"{track_prefix}_default_median")
        z_150 = torch.clamp((X_150 - f_mean) / f_std, -8.0, 8.0)
        return med + torch.einsum("bcd,d->bc", z_150, w)

    def _compute_sos_prior_batched_gpu(
        self,
        cand_412: torch.Tensor,
        cand_sos_base_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched GPU evaluation of `_compute_slate_foldsafe_sos_prior` (`[B, 96, 2]`).
        Uses exact precomputed `bank_sos_2` for base bank templates (`0 <= idx < 885`)
        and structural cosine similarity top-24 prior for live AST mutations (`idx < 0`).
        Zero host syncs.
        """
        B, M, _ = cand_412.shape
        pad = F.pad(cand_412.to(torch.float32), (0, 20)).view(B, M, 18, 24)
        sig = F.normalize(pad.mean(dim=2) + 0.5 * pad.max(dim=2).values, p=2, dim=-1, eps=1e-6)
        sims = (sig @ self.sos_struct_mat_256x24.T) + self.sos_jitter_1x1x256
        top24_idx = torch.topk(sims, k=24, dim=-1).indices  # [B, M, 24]
        sp_mean = self.sos_struct_val_256[top24_idx].mean(dim=-1)  # [B, M]
        sos_mut = torch.stack([0.35 * sp_mean, torch.full_like(sp_mean, 0.25)], dim=-1)  # [B, M, 2]

        safe_idx = cand_sos_base_idx.clamp(min=0, max=884)
        sos_bank = torch.gather(
            self.bank_sos_2[:885].unsqueeze(0).expand(B, -1, -1),
            1,
            safe_idx.unsqueeze(-1).expand(B, M, 2),
        )
        use_bank = ((cand_sos_base_idx >= 0) & (cand_sos_base_idx < 885)).unsqueeze(-1)
        return torch.where(use_bank, sos_bank, sos_mut)

    def _select_slate_and_anchors_batched_gpu(
        self,
        X_tr: torch.Tensor,
        y_tr_norm: torch.Tensor,
        tr_row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        lm48_z: torch.Tensor,
        *,
        cand_bank_idx_override: Optional[torch.Tensor] = None,
        cand_prog_override: Optional[torch.Tensor] = None,
        cand_sos_base_idx_override: Optional[torch.Tensor] = None,
        cand_op_onehot_override: Optional[torch.Tensor] = None,
        cand_ns_mask_flt_override: Optional[torch.Tensor] = None,
        cand_op_desc_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        100% GPU candidate slate featurization (`proj16`, `sem24`, `sos_prior`) + calibrated priors +
        batched inner 80/20 warm-start anchor evaluation for the 4 revealed anchors.
        Zero host syncs.
        """
        B, N, P = X_tr.shape
        device = X_tr.device

        if (
            cand_bank_idx_override is not None
            and cand_prog_override is not None
            and cand_sos_base_idx_override is not None
            and cand_op_onehot_override is not None
            and cand_ns_mask_flt_override is not None
            and cand_op_desc_override is not None
        ):
            cand_bank_idx = cand_bank_idx_override
            cand_412 = cand_prog_override.to(torch.float32)
            cand_proj16 = (cand_412 @ self.w1_sign_matrix_412x16) / math.sqrt(412.0)
            sem_24 = compute_semantic_struct_sig_24d_torch(cand_412)
            cand_sos = self._compute_sos_prior_batched_gpu(cand_412, cand_sos_base_idx_override)
            cand_op_onehot = cand_op_onehot_override.to(torch.float32)
            cand_ns_mask_flt = cand_ns_mask_flt_override.to(torch.float32)
            cand_op_desc = cand_op_desc_override.to(torch.float32)
        else:
            # Select 96 candidates per table from the 1,766-candidate GPU bank:
            # 36 nearest strict_train_881 champions (ns_mask=1.0) + 4 v3_bound (ns_mask=0.0)
            # + 56 EditZero mutated candidates (ns_mask=0.0).
            q_norm = lm48_z / torch.clamp(torch.norm(lm48_z, dim=-1, keepdim=True), min=1e-6)  # [B, 48]
            sims = q_norm @ self.bank_lm48_z_881.T  # [B, 881]
            v3_idx = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0).expand(B, 4)
            if self.bank_feat412.shape[0] >= 1766:
                top56_idx = torch.topk(sims, k=56, dim=-1).indices  # [B, 56]
                cand_bank_idx = torch.cat(
                    [top56_idx[:, :36] + 4, v3_idx, top56_idx[:, :56] + 885],
                    dim=1,
                )  # [B, 96]
            else:
                top92_idx = torch.topk(sims, k=92, dim=-1).indices + 4
                cand_bank_idx = torch.cat([top92_idx[:, :32], v3_idx, top92_idx[:, 32:92]], dim=1)

            cand_412 = torch.gather(
                self.bank_feat412.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, 412),
            )  # [B, 96, 412]
            cand_proj16 = torch.gather(
                self.bank_proj16.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, 16),
            )  # [B, 96, 16]
            sem_24 = torch.gather(
                self.bank_sem24.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, 24),
            )  # [B, 96, 24]
            cand_sos = torch.gather(
                self.bank_sos_2.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, 2),
            )  # [B, 96, 2]
            op_oh_dim = self.bank_op_onehot.shape[1]
            cand_op_onehot = torch.gather(
                self.bank_op_onehot.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, op_oh_dim),
            )  # [B, 96, op_oh_dim]
            cand_ns_mask_flt = torch.gather(
                self.bank_ns_mask_flt.unsqueeze(0).expand(B, -1),
                1,
                cand_bank_idx,
            )  # [B, 96]
            cand_op_desc = torch.gather(
                self.bank_op_desc.unsqueeze(0).expand(B, -1, -1),
                1,
                cand_bank_idx.unsqueeze(-1).expand(B, 96, 8),
            )  # [B, 96, 8]

        priors_10tok = self._predict_calibrated_prior_batched_gpu(
            cand_412, sem_24, lm48_z, track_prefix="prior10"
        )  # [B, 96]
        priors_fp = self._predict_calibrated_prior_batched_gpu(
            cand_412, sem_24, lm48_z, track_prefix="priorfp"
        )  # [B, 96]

        # Base SPQ-1 prior distribution p0_base_96 from priors_fp
        fp_mean = priors_fp.mean(dim=-1, keepdim=True)
        fp_std = priors_fp.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        prior_logits = torch.clamp(-1.25 * (priors_fp - fp_mean) / fp_std, -3.0, 3.0)
        p0_base = F.softmax(prior_logits, dim=-1).clamp_min(1e-12)
        p0_base = p0_base / p0_base.sum(dim=-1, keepdim=True)  # [B, 96]

        # 2. Evaluate 4 warm-start anchors (cand_bank_idx[:, :4]) on inner 80/20 split of (X_tr, y_tr_norm) on GPU
        row_idx = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
        in_tr_m = tr_row_mask * ((row_idx % 5) < 4).to(torch.float32)  # [B, N]
        in_va_m = tr_row_mask * ((row_idx % 5) == 4).to(torch.float32)  # [B, N]

        P_anc = min(P, 64)
        X_anc = torch.nan_to_num(X_tr[:, :, :P_anc], nan=0.0, posinf=0.0, neginf=0.0)
        c_anc = col_mask[:, :P_anc].unsqueeze(1)
        n_itr = in_tr_m.sum(dim=1, keepdim=True).clamp_min(2.0).unsqueeze(-1)
        mu_anc = (X_anc * in_tr_m.unsqueeze(-1) * c_anc).sum(dim=1, keepdim=True) / n_itr
        sd_anc = torch.sqrt(
            (((X_anc - mu_anc) * in_tr_m.unsqueeze(-1) * c_anc) ** 2).sum(dim=1, keepdim=True) / n_itr + 1e-6
        )
        Z_anc = torch.clamp((X_anc - mu_anc) / sd_anc, -8.0, 8.0) * c_anc
        if P_anc < 64:
            Z_anc = F.pad(Z_anc, (0, 64 - P_anc))
        ones_col = torch.ones((B, N, 1), device=device, dtype=torch.float32)
        Phi_base = torch.cat([Z_anc, ones_col], dim=-1)  # [B, N, 65]
        Phi_tr = Phi_base * in_tr_m.unsqueeze(-1)
        y_itr = (y_tr_norm * in_tr_m).unsqueeze(-1)

        G = torch.bmm(Phi_tr.transpose(1, 2), Phi_tr)
        rhs = torch.bmm(Phi_tr.transpose(1, 2), y_itr)

        eye65 = self.anchor_eye_65.unsqueeze(0)
        G_4 = G.unsqueeze(1) + self.anchor_alphas_4 * eye65.unsqueeze(1)
        L_4, _ = torch.linalg.cholesky_ex(G_4)
        w_4 = self._cholesky_solve_sync_free(L_4, rhs.unsqueeze(1).expand(B, 4, 65, 1)).squeeze(-1)

        preds_va_4 = torch.einsum("bnp,bkp->bkn", Phi_base, w_4) * in_va_m.unsqueeze(1)
        y_va = (y_tr_norm * in_va_m).unsqueeze(1)
        n_iva = in_va_m.sum(dim=1, keepdim=True).clamp_min(1.0)
        y_va_mean = (y_va.squeeze(1).sum(dim=1, keepdim=True) / n_iva).unsqueeze(-1)
        tss_va = (((y_va - y_va_mean) * in_va_m.unsqueeze(1)) ** 2).sum(dim=-1).clamp_min(1e-6)
        sse_va = (((y_va - preds_va_4) * in_va_m.unsqueeze(1)) ** 2).sum(dim=-1)
        r2_va_4 = 1.0 - sse_va / tss_va  # [B, 4]

        # Match lines 2147-2150 of jope_micro_gpu_pipeline.py:
        rmse_ratio = torch.sqrt(torch.clamp(1.0 - torch.clamp(r2_va_4, -3.0, 0.9999), min=1e-4))
        rev_f_4 = torch.clamp(torch.log(rmse_ratio), -2.5, 2.5)  # [B, 4]
        rev_gate_4 = (r2_va_4 > 0.0).to(torch.float32)  # [B, 4]

        scale_t = torch.clamp(rev_f_4.std(dim=-1, unbiased=False), min=0.10)  # [B]
        inc_idx = torch.argmin(rev_f_4, dim=-1)  # [B]
        inc_u = torch.gather(rev_f_4, 1, inc_idx.unsqueeze(1)).squeeze(1)  # [B]
        inc_sem24 = torch.gather(
            sem_24[:, :4, :],
            1,
            inc_idx.view(B, 1, 1).expand(B, 1, 24),
        ).expand(B, 96, 24)  # [B, 96, 24]
        parent_y = (inc_u / scale_t).unsqueeze(1).expand(B, 96)  # [B, 96]
        parent_gate = torch.ones((B, 96), device=device, dtype=torch.float32)

        return {
            "cand_bank_idx": cand_bank_idx,
            "cand_prog": cand_412,
            "cand_proj": cand_proj16,
            "sem24_96": sem_24,
            "cand_sos": cand_sos,
            "priors_10tok": priors_10tok,
            "all_prog": torch.cat([cand_412[:, :4, :], cand_412], dim=1),
            "all_proj": torch.cat([cand_proj16[:, :4, :], cand_proj16], dim=1),
            "all_sos": torch.cat([cand_sos[:, :4, :], cand_sos], dim=1),
            "all_prior_10tok": torch.cat([priors_fp[:, :4], priors_10tok], dim=1),
            "scale_t": scale_t,
            "rev_idx": torch.arange(4, device=device, dtype=torch.long).unsqueeze(0).expand(B, 4),
            "rev_f": rev_f_4,
            "rev_gate": rev_gate_4,
            "parent_sem": inc_sem24,
            "child_sem": sem_24,
            "op_onehot": cand_op_onehot,
            "parent_y": parent_y,
            "parent_gate": parent_gate,
            "p0": p0_base,
            "ns_mask_flt": cand_ns_mask_flt,
            "lopo_96": torch.zeros((B, 96), device=device, dtype=torch.float32),
            "cand_op_desc": cand_op_desc,
        }

    # =========================================================================
    # STAGE 2: Exact Batched BlockFormer + GameNet + QHat + Sync-Free SPQ-1
    # =========================================================================
    def _solve_spq1_batched_sync_free_gpu(
        self,
        p0: torch.Tensor,
        ns_mask_flt: torch.Tensor,
        adv_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched `[B, 96]` version of `solve_spq1_sync_free_gpu` (`257 x 13` grid).
        Bit-exact to `captured_router.py:solve_spq1_sync_free_gpu` for `B=1`, zero host syncs.
        """
        B, M = p0.shape
        non_ns_flt = 1.0 - ns_mask_flt
        base_ns = torch.sum(p0 * ns_mask_flt, dim=-1, keepdim=True)  # [B, 1]
        non_ns_sum = torch.clamp(torch.sum(p0 * non_ns_flt, dim=-1, keepdim=True), min=1e-12)
        cond_adj = (base_ns > 0.25) & (torch.sum(non_ns_flt, dim=-1, keepdim=True) > 0.5)
        scale_ns = (0.25 * 0.85) / torch.clamp(base_ns, min=1e-12)
        scale_non_ns = (1.0 - 0.25 * 0.85) / non_ns_sum
        p_adj = p0 * (ns_mask_flt * scale_ns + non_ns_flt * scale_non_ns)
        p_adj = p_adj / torch.clamp(torch.sum(p_adj, dim=-1, keepdim=True), min=1e-12)
        p0_eff = torch.where(cond_adj, p_adj, p0)
        log_p0 = torch.log(torch.clamp(p0, min=1e-15))      # [B, 96]
        log_p0_eff = torch.log(torch.clamp(p0_eff, min=1e-15))  # [B, 96]

        # 257-point coarse tau grid: [B, 257, 96]
        tau_257 = self.tau_grid_257.view(1, 257, 1)
        logits_grid = log_p0_eff.unsqueeze(1) + tau_257 * adv_norm.unsqueeze(1)  # [B, 257, 96]
        q_grid = torch.clamp(F.softmax(logits_grid, dim=-1), min=1e-15)
        kl_grid = torch.clamp(
            torch.sum(q_grid * (torch.log(q_grid) - log_p0.unsqueeze(1)), dim=-1),
            min=0.0,
        )  # [B, 257]
        ns_grid = torch.sum(q_grid * ns_mask_flt.unsqueeze(1), dim=-1)  # [B, 257]

        adm_mask = (kl_grid <= (0.35 - 1e-5)) & (ns_grid <= (0.25 - 1e-5))  # [B, 257]
        adm_mask_hi = (kl_grid[:, -1] <= 0.35) & (ns_grid[:, -1] <= 0.25)  # [B]

        idx_257 = self.grid_idx_257.unsqueeze(0).expand(B, 257)
        best_idx = torch.max(torch.where(adm_mask, idx_257, torch.zeros_like(idx_257)), dim=-1).values  # [B]
        hi_idx = torch.clamp(best_idx + 1, max=256)

        tau_257_b = self.tau_grid_257.unsqueeze(0).expand(B, 257)
        lo_t = torch.gather(tau_257_b, 1, best_idx.unsqueeze(1))  # [B, 1]
        hi_t = torch.gather(tau_257_b, 1, hi_idx.unsqueeze(1))    # [B, 1]

        # 13-point fine sub-grid: [B, 13, 96]
        sub_taus = lo_t + self.frac_13.unsqueeze(0) * (hi_t - lo_t)  # [B, 13]
        logits_sub = log_p0_eff.unsqueeze(1) + sub_taus.unsqueeze(-1) * adv_norm.unsqueeze(1)  # [B, 13, 96]
        q_sub = torch.clamp(F.softmax(logits_sub, dim=-1), min=1e-15)
        kl_sub = torch.clamp(
            torch.sum(q_sub * (torch.log(q_sub) - log_p0.unsqueeze(1)), dim=-1),
            min=0.0,
        )  # [B, 13]
        ns_sub = torch.sum(q_sub * ns_mask_flt.unsqueeze(1), dim=-1)  # [B, 13]
        adm_sub = (kl_sub <= (0.35 - 1e-5)) & (ns_sub <= (0.25 - 1e-5))  # [B, 13]

        idx_13 = self.grid_idx_13.unsqueeze(0).expand(B, 13)
        pick_t = torch.max(torch.where(adm_sub, idx_13, torch.zeros_like(idx_13)), dim=-1).values  # [B]
        q_sub_pick = torch.gather(q_sub, 1, pick_t.view(B, 1, 1).expand(B, 1, M)).squeeze(1)  # [B, 96]
        tau_sub_pick = torch.gather(sub_taus, 1, pick_t.unsqueeze(1)).squeeze(1)  # [B]

        q_final = torch.where(adm_mask_hi.unsqueeze(-1), q_grid[:, -1, :], q_sub_pick)  # [B, 96]
        chosen_tau = torch.where(adm_mask_hi, self.tau_grid_257[-1], tau_sub_pick)      # [B]
        return q_final, chosen_tau

    def _run_stage2_router_qhat_spq1_batched_gpu(
        self,
        tokens_10x32: torch.Tensor,
        mask_10: torch.Tensor,
        st: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact batched Stage 2 execution across `B` tables simultaneously:
          1. `adapter_10tok.predict_costs(...)`
          2. `adapter_10tok.predict_parent_child_delta(...)`
          3. `QHat` (`93D` bilinear ridge head)
          4. Batched sync-free `SPQ-1` (`257 x 13` grid)
        Zero host syncs.
        """
        B = tokens_10x32.shape[0]
        device = tokens_10x32.device
        ctx_t = torch.zeros((B, 584), device=device, dtype=torch.float32)

        out_m = self.adapter_10tok.predict_costs(
            ctx_feat=ctx_t,
            prog_412=st["all_prog"],
            proj_16=st["all_proj"],
            base_prior_cost=st["all_prior_10tok"],
            scale=st["scale_t"],
            sos_prior=st["all_sos"],
            mm_tokens_10x32=tokens_10x32,
            mm_mask_10=mask_10,
            revealed_indices=st["rev_idx"],
            revealed_raw_f=st["rev_f"],
            revealed_gate=st["rev_gate"],
        )
        pred_delta = self.adapter_10tok.predict_parent_child_delta(
            mm_tokens_10x32=tokens_10x32,
            mm_mask_10=mask_10,
            parent_sem24=st["parent_sem"],
            child_sem24=st["child_sem"],
            op_onehot=st["op_onehot"],
            parent_norm_y=st["parent_y"],
            parent_gate=st["parent_gate"],
        ).float()  # [B, 96]

        cand_pred_cost = out_m["total_cost"].float()[:, 4:]  # [B, 96]
        cand_fail_logit = out_m["fail_logit"].float()[:, 4:]  # [B, 96]
        cand_fail_prob = torch.sigmoid(torch.clamp(cand_fail_logit, -20.0, 20.0))

        scale_col = st["scale_t"].unsqueeze(1)  # [B, 1]
        cand_sos = st["cand_sos"]  # [B, 96, 2]
        cand_prior = st["priors_10tok"]  # [B, 96]
        sos_bonus = 0.12 * scale_col * torch.clamp(cand_sos[:, :, 0], -2.0, 2.0)
        neural_cost = (
            0.65 * cand_prior
            + 0.35 * cand_pred_cost
            + 0.30 * scale_col * pred_delta
            + 0.30 * scale_col * cand_fail_prob
            + sos_bonus
        )  # [B, 96]

        cand_prog = st["cand_prog"]  # [B, 96, 412]
        cand_proj = st["cand_proj"]  # [B, 96, 16]
        sem24 = st["sem24_96"]       # [B, 96, 24]
        M = cand_prog.shape[1]

        c8 = tokens_10x32[:, :4, :16].reshape(B, 64)[:, :8].unsqueeze(1).expand(B, M, 8)  # [B, 96, 8]
        p8 = torch.cat([sem24[:, :, :5], cand_proj[:, :, :3]], dim=-1)  # [B, 96, 8]
        cross_64 = (c8.unsqueeze(3) * p8.unsqueeze(2)).reshape(B, M, 64)  # [B, 96, 64]
        pos = torch.clamp(cand_prog, min=0.0)
        summary_13 = torch.stack(
            [
                cand_prior,
                cand_sos[:, :, 0],
                cand_sos[:, :, 1],
                st["lopo_96"],
                sem24[:, :, 5],
                sem24[:, :, 6],
                sem24[:, :, 7],
                sem24[:, :, 8],
                torch.log1p(pos[:, :, 23]) / 5.0,
                pos[:, :, 411] / 5.0,
                torch.log1p(pos[:, :, 52]) / 6.0,
                torch.sum(pos[:, :, 24:29], dim=-1),
                torch.log1p(torch.sum(pos[:, :, 0:23], dim=-1)) / 3.0,
            ],
            dim=-1,
        )  # [B, 96, 13]
        X_93 = torch.cat([summary_13, c8, p8, cross_64], dim=-1)  # [B, 96, 93]
        X_93_n = (X_93 - self.qhat_mu_x_93) / self.qhat_sd_x_93
        qhat_cost = torch.einsum("bcd,d->bc", X_93_n, self.qhat_w_93) + self.qhat_b_0  # [B, 96]

        combined_cost = 0.70 * neural_cost + 0.20 * qhat_cost  # [B, 96]
        adv = -combined_cost
        # Note: torch.std(adv) in 1D uses unbiased=True (Bessel correction)
        adv_std = torch.clamp(torch.std(adv, dim=-1, keepdim=True, unbiased=True), min=1e-6)
        adv_norm = torch.clamp((adv - torch.mean(adv, dim=-1, keepdim=True)) / adv_std, -4.0, 4.0)

        q_final, chosen_tau = self._solve_spq1_batched_sync_free_gpu(
            st["p0"], st["ns_mask_flt"], adv_norm
        )
        top1_idx = torch.argmax(q_final, dim=-1)  # [B]
        return neural_cost, qhat_cost, combined_cost, q_final, top1_idx

    # =========================================================================
    # STAGE 3: Batched Sync-Free GPU GP Fitter (Exact Ridge + Fast Surrogate)
    # =========================================================================
    def _build_gp_basis_batched_gpu(
        self,
        X: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        tr_stats: Dict[str, torch.Tensor],
        win_op_desc: torch.Tensor,
        *,
        fitter_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the batched GP design matrix `Phi` `[B, N, K_max + 1]` on GPU for the
        winning program per table (`win_op_desc` `[B, 8]`), using branch-free tensor masks.
        Zero host syncs.
        """
        B, N, P = X.shape
        device = X.device
        K_max = self.max_fit_basis_dim
        P_cap = min(P, 64)

        row_m = row_mask.unsqueeze(-1)
        top64_col_idx = tr_stats.get("top64_col_idx")
        if P > 64 and top64_col_idx is not None:
            col_m_sub = torch.gather(col_mask, 1, top64_col_idx)
            col_m = col_m_sub.unsqueeze(1)
            valid_m = row_m * col_m
            X_raw_sub = torch.gather(X, 2, top64_col_idx.unsqueeze(1).expand(B, N, 64))
            X_c = torch.nan_to_num(X_raw_sub, nan=0.0, posinf=0.0, neginf=0.0) * valid_m
        else:
            col_m_sub = col_mask[:, :P_cap]
            col_m = col_m_sub.unsqueeze(1)
            valid_m = row_m * col_m
            X_c = torch.nan_to_num(X[:, :, :P_cap], nan=0.0, posinf=0.0, neginf=0.0) * valid_m

        mu_tr = tr_stats["mu_tr"]
        sd_tr = tr_stats["sd_tr"]
        med_tr = tr_stats["med_tr"]
        iqr_tr = tr_stats["iqr_tr"]
        q_lo = tr_stats["q_lo"]
        q_hi = tr_stats["q_hi"]
        top8_idx = tr_stats["top8_idx"]

        Z_std = torch.clamp((X_c - mu_tr) / sd_tr, -8.0, 8.0) * valid_m

        if fitter_mode == "gpu_fast_fitter_approximate_track":
            nonlin_type = win_op_desc[:, 4].view(B, 1, 1)
            is_cheb = (nonlin_type == 3.0).to(torch.float32)
            is_poly = ((nonlin_type == 5.0) | (nonlin_type == 6.0)).to(torch.float32)

            X_norm = torch.tanh(Z_std) * valid_m
            T2 = (2.0 * (X_norm ** 2) - 1.0) * valid_m
            T3 = (4.0 * (X_norm ** 3) - 3.0 * X_norm) * valid_m
            cheb_part = torch.cat([T2[:, :, :min(P_cap, 32)], T3[:, :, :min(P_cap, 32)]], dim=-1)
            if cheb_part.shape[2] < 64:
                cheb_part = F.pad(cheb_part, (0, 64 - cheb_part.shape[2]))

            idx8 = top8_idx.unsqueeze(1).expand(B, N, 8)
            Z_pad64 = F.pad(Z_std, (0, max(0, 64 - P_cap)))
            Z_8 = torch.gather(Z_pad64, 2, idx8)
            poly_part = (Z_8.unsqueeze(3) * Z_8.unsqueeze(2)).reshape(B, N, 64) * row_m

            extra_64 = is_cheb * cheb_part + is_poly * poly_part
            extra_mask = ((is_cheb + is_poly) > 0.0).to(torch.float32).squeeze(-1)
            col_m64 = F.pad(col_m_sub, (0, max(0, 64 - P_cap)))
            feat_128 = torch.cat([Z_pad64[:, :, :64], extra_64], dim=-1)
            mask_128 = torch.cat([col_m64[:, :64], extra_mask.expand(B, 64)], dim=-1)
            if K_max > 128:
                feat_k = F.pad(feat_128, (0, K_max - 128))
                mask_k = F.pad(mask_128, (0, K_max - 128))
            else:
                feat_k = feat_128[:, :, :K_max]
                mask_k = mask_128[:, :K_max]
            ones = row_mask.unsqueeze(-1)
            Phi = torch.cat([feat_k * row_m, ones], dim=-1)
            mask_full = torch.cat([mask_k, torch.ones((B, 1), device=device, dtype=torch.float32)], dim=-1)
            return Phi, mask_full

        has_winsor = (win_op_desc[:, 0] > 0.0).to(torch.float32).view(B, 1, 1)
        winsor_app = (win_op_desc[:, 1] > 0.5).to(torch.float32).view(B, 1, 1)
        has_robust = (win_op_desc[:, 2] > 0.5).to(torch.float32).view(B, 1, 1)
        robust_app = (win_op_desc[:, 3] > 0.5).to(torch.float32).view(B, 1, 1)
        nonlin_type = win_op_desc[:, 4].view(B, 1, 1)

        X_win = torch.clamp(X_c, q_lo, q_hi) * valid_m
        Z_win = torch.clamp((X_win - mu_tr) / sd_tr, -8.0, 8.0) * valid_m
        X_base = torch.lerp(Z_std, Z_win, has_winsor)
        Z_rob = torch.clamp(X_base / 1.3489795, -6.0, 6.0) * valid_m
        B0_raw = torch.lerp(X_base, Z_rob, has_robust)

        b0_mu = tr_stats["b0_mu"]
        b0_sd = tr_stats["b0_sd"]
        B0 = torch.clamp((B0_raw - b0_mu) / b0_sd, -8.0, 8.0) * valid_m

        B1 = (winsor_app * (Z_win - Z_std) + robust_app * torch.tanh(Z_std / 1.3489795)) * valid_m
        b1_active = ((winsor_app + robust_app) > 0.0).to(torch.float32).squeeze(-1)

        sig_feat = (torch.sigmoid(B0) - 0.5) * 2.0 * valid_m
        row_l2 = torch.sqrt((B0 ** 2).sum(dim=2, keepdim=True) + 1e-8)
        norm_row_feat = (B0 / row_l2) * valid_m
        X_t = torch.tanh(B0) * valid_m
        cheb_feat = (2.0 * (X_t ** 2) - 1.0 + 0.5 * (4.0 * (X_t ** 3) - 3.0 * X_t)) * valid_m
        hinge_feat = (F.relu(B0) - F.relu(-B0 - 0.5)) * valid_m

        B0_pad64 = F.pad(B0, (0, max(0, 64 - P_cap)))
        idx8 = top8_idx.unsqueeze(1).expand(B, N, 8)
        B0_8 = torch.gather(B0_pad64, 2, idx8)
        poly_64 = (B0_8.unsqueeze(3) * B0_8.unsqueeze(2)).reshape(B, N, 64) * row_m

        B1_pad64 = F.pad(B1, (0, max(0, 64 - P_cap)))
        sig_pad64 = F.pad(sig_feat, (0, max(0, 64 - P_cap)))
        nr_pad64 = F.pad(norm_row_feat, (0, max(0, 64 - P_cap)))
        ch_pad64 = F.pad(cheb_feat, (0, max(0, 64 - P_cap)))
        hg_pad64 = F.pad(hinge_feat, (0, max(0, 64 - P_cap)))

        B2_64 = (
            (nonlin_type == 1.0).to(torch.float32) * sig_pad64
            + (nonlin_type == 2.0).to(torch.float32) * nr_pad64
            + (nonlin_type == 3.0).to(torch.float32) * ch_pad64
            + (nonlin_type == 4.0).to(torch.float32) * hg_pad64
            + ((nonlin_type == 5.0) | (nonlin_type == 6.0)).to(torch.float32) * poly_64
            + (nonlin_type == 7.0).to(torch.float32) * torch.erf(B0_pad64)
        ) * row_m
        b2_active = (nonlin_type > 0.0).to(torch.float32).squeeze(-1)

        col_m64 = F.pad(col_m_sub, (0, max(0, 64 - P_cap)))
        poly_is_active = ((nonlin_type == 5.0) | (nonlin_type == 6.0)).to(torch.float32).squeeze(-1)
        m2_64 = torch.where(poly_is_active > 0.5, torch.ones_like(col_m64), col_m64) * b2_active

        feat_192 = torch.cat([B0_pad64[:, :, :64], B1_pad64[:, :, :64], B2_64[:, :, :64]], dim=-1)
        mask_192 = torch.cat([col_m64[:, :64], col_m64[:, :64] * b1_active, m2_64[:, :64]], dim=-1)

        feat_k = feat_192[:, :, :K_max]
        mask_k = mask_192[:, :K_max]
        Phi = torch.cat([feat_k * mask_k.unsqueeze(1) * row_m, row_mask.unsqueeze(-1)], dim=-1)
        mask_full = torch.cat([mask_k, torch.ones((B, 1), device=device, dtype=torch.float32)], dim=-1)
        return Phi, mask_full

    def _fit_gp_winner_batched_gpu(
        self,
        X_tr: torch.Tensor,
        y_tr_norm: torch.Tensor,
        X_ho: torch.Tensor,
        y_ho_norm: torch.Tensor,
        tr_row_mask: torch.Tensor,
        ho_row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        mu_y: torch.Tensor,
        sd_y: torch.Tensor,
        win_op_desc: torch.Tensor,
        *,
        fitter_mode: str,
    ) -> Dict[str, Any]:
        B, N_tr, P = X_tr.shape
        P_cap = min(P, 64)

        row_m = tr_row_mask.unsqueeze(-1)
        n_tr = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(2.0).unsqueeze(-1)
        y_c = y_tr_norm * tr_row_mask

        if P > 64:
            valid_all = row_m * col_mask.unsqueeze(1)
            X_all = torch.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0) * valid_all
            mu_all = X_all.sum(dim=1, keepdim=True) / n_tr
            sd_all = torch.sqrt(
                ((X_all - mu_all) ** 2 * valid_all).sum(dim=1, keepdim=True) / (n_tr - 1.0).clamp_min(1.0)
            ).clamp_min(1e-6)
            Z_all = torch.clamp((X_all - mu_all) / sd_all, -8.0, 8.0) * valid_all
            corr_all = torch.abs(torch.einsum("bnp,bn->bp", Z_all, y_c) / n_tr.squeeze(-1)) * col_mask - (1.0 - col_mask)
            top64_corr_idx = torch.topk(corr_all, k=64, dim=1).indices
            nat64_idx = torch.arange(64, device=X_tr.device, dtype=torch.long).unsqueeze(0).expand(B, 64)
            is_wide = col_mask.sum(dim=1, keepdim=True) > 64.5
            top64_col_idx = torch.where(is_wide, top64_corr_idx, nat64_idx)
            col_m_sub = torch.gather(col_mask, 1, top64_col_idx)
            col_m = col_m_sub.unsqueeze(1)
            valid_m = row_m * col_m
            X_c = torch.gather(X_all, 2, top64_col_idx.unsqueeze(1).expand(B, N_tr, 64)) * valid_m
        else:
            top64_col_idx = torch.arange(P_cap, device=X_tr.device, dtype=torch.long).unsqueeze(0).expand(B, P_cap)
            col_m_sub = col_mask[:, :P_cap]
            col_m = col_m_sub.unsqueeze(1)
            valid_m = row_m * col_m
            X_c = torch.nan_to_num(X_tr[:, :, :P_cap], nan=0.0, posinf=0.0, neginf=0.0) * valid_m

        mu_tr = X_c.sum(dim=1, keepdim=True) / n_tr
        sd_tr = torch.sqrt(
            ((X_c - mu_tr) ** 2 * valid_m).sum(dim=1, keepdim=True) / (n_tr - 1.0).clamp_min(1.0)
        ).clamp_min(1e-6)

        med_tr = mu_tr
        iqr_tr = (1.3489795 * sd_tr).clamp_min(1e-6)
        q_lo = mu_tr - 1.96 * sd_tr
        q_hi = mu_tr + 1.96 * sd_tr

        Z_std = torch.clamp((X_c - mu_tr) / sd_tr, -8.0, 8.0) * valid_m
        corr = torch.abs(torch.einsum("bnp,bn->bp", Z_std, y_c) / n_tr.squeeze(-1)) * col_m_sub
        corr_pad64 = F.pad(corr, (0, max(0, 64 - P_cap)))
        top8_idx = torch.topk(corr_pad64, k=8, dim=1).indices

        has_winsor = (win_op_desc[:, 0] > 0.0).to(torch.float32).view(B, 1, 1)
        has_robust = (win_op_desc[:, 2] > 0.5).to(torch.float32).view(B, 1, 1)
        X_win = torch.clamp(X_c, q_lo, q_hi) * valid_m
        Z_win = torch.clamp((X_win - mu_tr) / sd_tr, -8.0, 8.0) * valid_m
        X_base = torch.lerp(Z_std, Z_win, has_winsor)
        Z_rob = torch.clamp(X_base / 1.3489795, -6.0, 6.0) * valid_m
        B0_raw = torch.lerp(X_base, Z_rob, has_robust)
        b0_mu = B0_raw.sum(dim=1, keepdim=True) / n_tr
        b0_sd = torch.sqrt(
            ((B0_raw - b0_mu) ** 2 * valid_m).sum(dim=1, keepdim=True) / (n_tr - 1.0).clamp_min(1.0)
        ).clamp_min(1e-6)

        tr_stats = {
            "mu_tr": mu_tr,
            "sd_tr": sd_tr,
            "med_tr": med_tr,
            "iqr_tr": iqr_tr,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "top8_idx": top8_idx,
            "b0_mu": b0_mu,
            "b0_sd": b0_sd,
            "top64_col_idx": top64_col_idx,
        }

        Phi_tr, mask_full = self._build_gp_basis_batched_gpu(
            X_tr, tr_row_mask, col_mask, tr_stats, win_op_desc, fitter_mode=fitter_mode
        )
        Phi_ho, _ = self._build_gp_basis_batched_gpu(
            X_ho, ho_row_mask, col_mask, tr_stats, win_op_desc, fitter_mode=fitter_mode
        )

        K1 = Phi_tr.shape[2]
        G = torch.bmm(Phi_tr.transpose(1, 2), Phi_tr)
        rhs = torch.bmm(Phi_tr.transpose(1, 2), (y_tr_norm * tr_row_mask).unsqueeze(-1))
        inactive_diag = torch.diag_embed(1.0 - mask_full)
        reg_eye = self.reg_eye_basis_max[:K1, :K1].unsqueeze(0)
        n_tr_1d = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(4.0)

        if fitter_mode == "gpu_fast_fitter_approximate_track":
            scale_diag_fast = (n_tr_1d.view(B, 1, 1) / 256.0).clamp_min(1.0)
            G_reg = G + (1.0 * scale_diag_fast) * reg_eye + inactive_diag
            L, info_fast = torch.linalg.cholesky_ex(G_reg)
            bad_fast = (info_fast != 0).unsqueeze(-1).unsqueeze(-1)
            L_safe = torch.where(bad_fast, reg_eye.expand_as(L), L)
            theta = (
                torch.nan_to_num(
                    self._cholesky_solve_sync_free(L_safe, rhs).squeeze(-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                * mask_full
            )
        else:
            n_lam = self.gcv_lambdas.shape[0]
            lams = self.gcv_lambdas.view(1, n_lam, 1, 1)
            scale_diag = (n_tr_1d.view(B, 1, 1, 1) / 256.0).clamp_min(1.0)
            G_grid = G.unsqueeze(1) + (lams * scale_diag) * reg_eye.unsqueeze(1) + inactive_diag.unsqueeze(1)
            L_grid, info_grid = torch.linalg.cholesky_ex(G_grid)
            bad_chol = (info_grid != 0).unsqueeze(-1).unsqueeze(-1)
            L_safe = torch.where(bad_chol, reg_eye.unsqueeze(1).expand_as(L_grid), L_grid)
            rhs_grid = rhs.unsqueeze(1).expand(B, n_lam, K1, 1)
            theta_grid = (
                torch.nan_to_num(
                    self._cholesky_solve_sync_free(L_safe, rhs_grid).squeeze(-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                * mask_full.unsqueeze(1)
            )

            preds_tr_grid = torch.einsum("bnk,bmk->bmn", Phi_tr, theta_grid) * tr_row_mask.unsqueeze(1)
            res_sq = ((y_tr_norm.unsqueeze(1) - preds_tr_grid) * tr_row_mask.unsqueeze(1)) ** 2
            sse_grid = res_sq.sum(dim=-1)
            eff_df = mask_full[:, :-1].sum(dim=-1, keepdim=True) / (1.0 + self.gcv_lambdas.unsqueeze(0))
            gcv_denom = torch.clamp(1.0 - (eff_df + 1.0) / n_tr_1d, min=0.15) ** 2
            gcv_score = torch.where(info_grid != 0, torch.full_like(sse_grid, 1e12), sse_grid / gcv_denom)

            is_plain_ridge = win_op_desc[:, 7] == 0.0
            best_lam_idx = torch.argmin(gcv_score, dim=-1)
            best_lam_idx = torch.where(is_plain_ridge, torch.full_like(best_lam_idx, 3), best_lam_idx)
            idx_theta = best_lam_idx.view(B, 1, 1).expand(B, 1, K1)
            theta = torch.gather(theta_grid, 1, idx_theta).squeeze(1)

        w_fitted = theta[:, :-1]
        b_fitted = theta[:, -1:]

        pred_tr_norm = torch.einsum("bnk,bk->bn", Phi_tr, theta) * tr_row_mask
        pred_ho_norm = torch.einsum("bnk,bk->bn", Phi_ho, theta) * ho_row_mask

        train_preds = (pred_tr_norm * sd_y + mu_y) * tr_row_mask
        hold_preds = (pred_ho_norm * sd_y + mu_y) * ho_row_mask

        n_tr_v = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        y_tr_m = (y_tr_norm * tr_row_mask).sum(dim=1, keepdim=True) / n_tr_v
        tss_tr = (((y_tr_norm - y_tr_m) * tr_row_mask) ** 2).sum(dim=1).clamp_min(1e-8)
        sse_tr = (((y_tr_norm - pred_tr_norm) * tr_row_mask) ** 2).sum(dim=1)
        train_r2 = 1.0 - sse_tr / tss_tr

        n_ho_v = ho_row_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        y_ho_m = (y_ho_norm * ho_row_mask).sum(dim=1, keepdim=True) / n_ho_v
        tss_ho = (((y_ho_norm - y_ho_m) * ho_row_mask) ** 2).sum(dim=1).clamp_min(1e-8)
        sse_ho = (((y_ho_norm - pred_ho_norm) * ho_row_mask) ** 2).sum(dim=1)
        hold_r2 = 1.0 - sse_ho / tss_ho

        return {
            "w_fitted": w_fitted,
            "b_fitted": b_fitted,
            "train_preds_norm": pred_tr_norm,
            "hold_preds_norm": pred_ho_norm,
            "train_preds": train_preds,
            "hold_preds": hold_preds,
            "train_r2": train_r2,
            "hold_r2": hold_r2,
            "mu_y": mu_y,
            "sd_y": sd_y,
            "win_op_desc": win_op_desc,
            "mask_full": mask_full,
            "tr_stats": tr_stats,
        }

    # =========================================================================
    # SINGLE FUSED FORWARD PASS (`raw table on GPU -> ... -> GPU GP fit`)
    # =========================================================================
    def forward(
        self,
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        X_ho: torch.Tensor,
        y_ho: torch.Tensor,
        tr_row_mask: Optional[torch.Tensor] = None,
        ho_row_mask: Optional[torch.Tensor] = None,
        col_mask: Optional[torch.Tensor] = None,
        *,
        use_landmarks: bool = True,
        fitter_mode: str = "exact_ridge",
        amp_dtype: Optional[torch.dtype] = torch.float16,
        vec256_override: Optional[torch.Tensor] = None,
        col_lm_override: Optional[torch.Tensor] = None,
        staged_slate_override: Optional[Dict[str, torch.Tensor]] = None,
        lm_vec256: Optional[torch.Tensor] = None,
        lm_col_lm: Optional[torch.Tensor] = None,
        lm_cand_bank_idx: Optional[torch.Tensor] = None,
        lm_cand_prog: Optional[torch.Tensor] = None,
        lm_cand_sos_base_idx: Optional[torch.Tensor] = None,
        lm_cand_op_onehot: Optional[torch.Tensor] = None,
        lm_cand_ns_mask_flt: Optional[torch.Tensor] = None,
        lm_cand_op_desc: Optional[torch.Tensor] = None,
        nolm_cand_bank_idx: Optional[torch.Tensor] = None,
        nolm_cand_prog: Optional[torch.Tensor] = None,
        nolm_cand_sos_base_idx: Optional[torch.Tensor] = None,
        nolm_cand_op_onehot: Optional[torch.Tensor] = None,
        nolm_cand_ns_mask_flt: Optional[torch.Tensor] = None,
        nolm_cand_op_desc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Single fused forward call:
          raw table on GPU -> optional GPU landmarks -> DOPE encoder (629D) ->
          Stage 1B SVD tokens (10x32) -> router/QHat/SPQ-1 -> winner program -> GPU GP fit.

        Zero host round-trips inside `forward()`.
        """
        if X_tr.ndim == 2:
            X_tr = X_tr.unsqueeze(0)
            y_tr = y_tr.unsqueeze(0)
            X_ho = X_ho.unsqueeze(0)
            y_ho = y_ho.unsqueeze(0)

        B, N_tr, P = X_tr.shape
        N_ho = X_ho.shape[1]
        device = X_tr.device

        if tr_row_mask is None:
            tr_row_mask = torch.ones((B, N_tr), device=device, dtype=torch.float32)
        if ho_row_mask is None:
            ho_row_mask = torch.ones((B, N_ho), device=device, dtype=torch.float32)
        if col_mask is None:
            col_mask = torch.ones((B, P), device=device, dtype=torch.float32)

        # Normalize target y strictly from training rows
        n_tr_valid = tr_row_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        y_tr_clean = torch.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0) * tr_row_mask
        y_ho_clean = torch.nan_to_num(y_ho, nan=0.0, posinf=0.0, neginf=0.0) * ho_row_mask
        mu_y = y_tr_clean.sum(dim=1, keepdim=True) / n_tr_valid
        var_y = (((y_tr_clean - mu_y) ** 2) * tr_row_mask).sum(dim=1, keepdim=True) / n_tr_valid
        sd_y = torch.sqrt(var_y).clamp_min(1e-8)
        y_tr_norm = ((y_tr_clean - mu_y) / sd_y) * tr_row_mask
        y_ho_norm = ((y_ho_clean - mu_y) / sd_y) * ho_row_mask

        # Stage 1A: Optional GPU Landmarks + DOPE-MLC-v3 (629D) + lm48 (48D)
        if vec256_override is not None and col_lm_override is not None:
            vec256 = vec256_override
            col_lm = col_lm_override
        elif use_landmarks and lm_vec256 is not None and lm_col_lm is not None:
            vec256 = lm_vec256
            col_lm = lm_col_lm
        elif use_landmarks:
            vec256, col_lm = self._extract_stage1a_landmarks_batched_gpu(
                X_tr, y_tr_norm, tr_row_mask, col_mask
            )
        else:
            vec256, col_lm = self._extract_stage1a_fast_moments_batched_gpu(
                X_tr, y_tr_norm, tr_row_mask, col_mask
            )

        use_amp = (amp_dtype is not None) and (device.type == "cuda")
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            dope629_amp, lm48_amp = self._run_dope_mlc_v3_and_lm48(
                vec256, col_lm, use_landmarks=use_landmarks
            )
        dope629 = dope629_amp.to(torch.float32)
        lm48 = lm48_amp.to(torch.float32)

        # Stage 1B: 10x32 Spectrum-Weighted SVD Modality Tokens
        tokens_10x32, mask_10, lm48_z = self._project_stage1b_tokens_batched_gpu(lm48, dope629)

        # Candidate Slate & Warm-Start Anchors on GPU
        if staged_slate_override is not None:
            st = staged_slate_override
        elif use_landmarks and lm_cand_prog is not None:
            st = self._select_slate_and_anchors_batched_gpu(
                X_tr,
                y_tr_norm,
                tr_row_mask,
                col_mask,
                lm48_z,
                cand_bank_idx_override=lm_cand_bank_idx,
                cand_prog_override=lm_cand_prog,
                cand_sos_base_idx_override=lm_cand_sos_base_idx,
                cand_op_onehot_override=lm_cand_op_onehot,
                cand_ns_mask_flt_override=lm_cand_ns_mask_flt,
                cand_op_desc_override=lm_cand_op_desc,
            )
        elif (not use_landmarks) and nolm_cand_prog is not None:
            st = self._select_slate_and_anchors_batched_gpu(
                X_tr,
                y_tr_norm,
                tr_row_mask,
                col_mask,
                lm48_z,
                cand_bank_idx_override=nolm_cand_bank_idx,
                cand_prog_override=nolm_cand_prog,
                cand_sos_base_idx_override=nolm_cand_sos_base_idx,
                cand_op_onehot_override=nolm_cand_op_onehot,
                cand_ns_mask_flt_override=nolm_cand_ns_mask_flt,
                cand_op_desc_override=nolm_cand_op_desc,
            )
        elif lm_cand_prog is not None:
            st = self._select_slate_and_anchors_batched_gpu(
                X_tr,
                y_tr_norm,
                tr_row_mask,
                col_mask,
                lm48_z,
                cand_bank_idx_override=lm_cand_bank_idx,
                cand_prog_override=lm_cand_prog,
                cand_sos_base_idx_override=lm_cand_sos_base_idx,
                cand_op_onehot_override=lm_cand_op_onehot,
                cand_ns_mask_flt_override=lm_cand_ns_mask_flt,
                cand_op_desc_override=lm_cand_op_desc,
            )
        else:
            st = self._select_slate_and_anchors_batched_gpu(
                X_tr, y_tr_norm, tr_row_mask, col_mask, lm48_z
            )

        # Stage 2: Neural Router + QHat + Batched Sync-Free SPQ-1
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            neural_cost, qhat_cost, combined_cost, q_final, top1_idx = (
                self._run_stage2_router_qhat_spq1_batched_gpu(tokens_10x32, mask_10, st)
            )

        # Gather winning candidate bank index & operator descriptor per table [B]
        cand_bank_idx = st["cand_bank_idx"]
        top1_bank_idx = torch.gather(cand_bank_idx, 1, top1_idx.unsqueeze(1)).squeeze(1)  # [B]
        win_op_idx = top1_idx.view(B, 1, 1).expand(B, 1, 8)
        win_op_desc = torch.gather(st["cand_op_desc"], 1, win_op_idx).squeeze(1)  # [B, 8]

        # Stage 3: Batched GPU GP Fit on Winning Candidate
        fit_out = self._fit_gp_winner_batched_gpu(
            X_tr,
            y_tr_norm,
            X_ho,
            y_ho_norm,
            tr_row_mask,
            ho_row_mask,
            col_mask,
            mu_y,
            sd_y,
            win_op_desc,
            fitter_mode=fitter_mode,
        )

        return {
            "top1_idx": top1_idx,
            "top1_bank_idx": top1_bank_idx,
            "neural_cost": neural_cost,
            "qhat_cost": qhat_cost,
            "combined_cost": combined_cost,
            "q_final": q_final,
            "dope629": dope629,
            "lm48": lm48,
            "lm48_z": lm48_z,
            "vec256": vec256,
            "col_lm": col_lm,
            "tokens_10x32": tokens_10x32,
            "w_fitted": fit_out["w_fitted"],
            "b_fitted": fit_out["b_fitted"],
            "train_preds": fit_out["train_preds"],
            "hold_preds": fit_out["hold_preds"],
            "train_preds_norm": fit_out["train_preds_norm"],
            "hold_preds_norm": fit_out["hold_preds_norm"],
            "train_r2": fit_out["train_r2"],
            "hold_r2": fit_out["hold_r2"],
            "mu_y": fit_out["mu_y"],
            "sd_y": fit_out["sd_y"],
            "win_op_desc": fit_out["win_op_desc"],
            "mask_full": fit_out["mask_full"],
            "tr_stats": fit_out["tr_stats"],
        }


# =============================================================================
# LIVE STAGE 1A & 96-CANDIDATE SLATE STAGING HELPER
# =============================================================================
_SLATE_STAGE_LOCK = None
_SLATE_STAGE_STATE: Optional[Dict[str, Any]] = None
_DSL_FEAT412_CACHE: Dict[str, np.ndarray] = {}


def _get_slate_stage_state() -> Dict[str, Any]:
    global _SLATE_STAGE_LOCK, _SLATE_STAGE_STATE
    import threading
    if _SLATE_STAGE_LOCK is None:
        _SLATE_STAGE_LOCK = threading.Lock()
    with _SLATE_STAGE_LOCK:
        if _SLATE_STAGE_STATE is not None:
            return _SLATE_STAGE_STATE
        from hpc.landmark_workspace import FusedLandmarkWorkspace
        from rust_evaluator import dsl_features_dict, get_feat_names, propose_flywheel_candidates

        _init_global_bank_registry()
        frozen_path = MICRO_ROOT / "assets" / "projections" / "frozen_stage1_stage2_inference_params_v3.pt"
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        sc = frozen["scalers"]
        step2_mean = np.asarray(sc["step2_lm_mean_1x256"], dtype=np.float32).reshape(256)
        step2_std = np.maximum(np.asarray(sc["step2_lm_std_1x256"], dtype=np.float32).reshape(256), 1e-6)
        step2_basis = np.asarray(sc["step2_svd_basis_256x48"], dtype=np.float32).reshape(256, 48)
        mu_lm48 = np.asarray(sc["mu_lm48"], dtype=np.float32).reshape(48)
        sd_lm48 = np.maximum(np.asarray(sc["sd_lm48"], dtype=np.float32).reshape(48), 1e-6)

        v3_cands = frozen["v3_bound_admissible_candidates"]
        champ_881 = frozen["champ_templates_881"]
        champ_lm48_z = np.stack([np.asarray(c["lm48_z"], dtype=np.float32) for c in champ_881], axis=0)
        champ_t_norms = np.maximum(np.linalg.norm(champ_lm48_z, axis=1), 1e-6)
        champ_bank_lite = [
            {
                "dsl": t["dsl"],
                "program_sha256": t["program_sha256"],
                "family_id": t["family_id"],
                "dl_split": "DL-Train",
                "split": "train",
                "landmark_summary": np.asarray(t["lm48_z"][:12], dtype=np.float32),
            }
            for t in champ_881[:128]
        ]

        base_885_dsl_to_idx: Dict[str, int] = {}
        for i, c in enumerate(v3_cands):
            d_s = str(c["dsl"]).strip()
            base_885_dsl_to_idx[d_s] = i
            _DSL_FEAT412_CACHE[d_s] = np.asarray(c["feat412"], dtype=np.float32)
        for i, c in enumerate(champ_881):
            d_s = str(c["dsl"]).strip()
            if d_s not in base_885_dsl_to_idx:
                base_885_dsl_to_idx[d_s] = 4 + i
            _DSL_FEAT412_CACHE[d_s] = np.asarray(c["feat412"], dtype=np.float32)

        nolm_helper_net = FusedJopeNetwork(adapter_variant="h45", device=torch.device("cpu")).eval()

        _SLATE_STAGE_STATE = {
            "FusedLandmarkWorkspace": FusedLandmarkWorkspace,
            "dsl_features_dict": dsl_features_dict,
            "feat_names": get_feat_names(),
            "propose_flywheel_candidates": propose_flywheel_candidates,
            "step2_mean": step2_mean,
            "step2_std": step2_std,
            "step2_basis": step2_basis,
            "mu_lm48": mu_lm48,
            "sd_lm48": sd_lm48,
            "v3_cands": v3_cands,
            "champ_881": champ_881,
            "champ_lm48_z": champ_lm48_z,
            "champ_t_norms": champ_t_norms,
            "champ_bank_lite": champ_bank_lite,
            "base_885_dsl_to_idx": base_885_dsl_to_idx,
            "nolm_helper_net": nolm_helper_net,
            "tls": threading.local(),
        }
        return _SLATE_STAGE_STATE


def _featurize_dsl_412_fast(dsl_s: str, st: Dict[str, Any]) -> np.ndarray:
    cached = _DSL_FEAT412_CACHE.get(dsl_s)
    if cached is not None:
        return cached
    d = st["dsl_features_dict"](dsl_s)
    f412 = np.asarray([float(d.get(k, 0.0)) for k in st["feat_names"]], dtype=np.float32)
    _DSL_FEAT412_CACHE[dsl_s] = f412
    return f412


def _build_96_slate_arrays_from_lm48(
    lm48_1d: np.ndarray,
    lm48_z_1d: np.ndarray,
    tr_sha: str,
    st: Dict[str, Any],
) -> Dict[str, Any]:
    seed_int = int(tr_sha[:8], 16)
    q_norm = lm48_z_1d / max(float(np.linalg.norm(lm48_z_1d)), 1e-6)
    sims = (st["champ_lm48_z"] @ q_norm) / st["champ_t_norms"]
    top_t_indices = np.argsort(-sims)

    seen_dsls: set[str] = set()
    cands: List[Tuple[str, str, np.ndarray]] = []

    def _add_cand(dsl_str: str, op_name: str, f412_opt: Optional[np.ndarray] = None) -> bool:
        d_c = dsl_str.strip()
        if not d_c or d_c in seen_dsls:
            return False
        seen_dsls.add(d_c)
        f412 = np.asarray(f412_opt, dtype=np.float32) if f412_opt is not None else _featurize_dsl_412_fast(d_c, st)
        cands.append((d_c, op_name, f412))
        return True

    parent_progs: List[Dict[str, Any]] = []
    champ_881 = st["champ_881"]
    for idx_t in top_t_indices:
        tpl = champ_881[int(idx_t)]
        if _add_cand(str(tpl["dsl"]), "strict_train_881_knn_champion", tpl["feat412"]):
            if len(parent_progs) < 6:
                parent_progs.append({
                    "dsl": tpl["dsl"],
                    "program_sha256": tpl["program_sha256"],
                    "fitness": 0.0,
                })
        if len(cands) >= 36:
            break

    for vc in st["v3_cands"]:
        _add_cand(str(vc["dsl"]), str(vc["operator"]), vc["feat412"])

    mut_proposals = st["propose_flywheel_candidates"](
        parent_programs=parent_progs,
        landmark_v2_48=lm48_1d,
        champion_bank=st["champ_bank_lite"],
        target_family_id="__arbitrary_external_table__",
        n_proposals=64,
        seed=seed_int,
    )
    for mp in mut_proposals:
        if len(cands) >= 96:
            break
        _add_cand(str(mp["dsl"]), str(mp.get("operator", "EditZero_Mutate")))

    for idx_t in top_t_indices:
        if len(cands) >= 96:
            break
        tpl = champ_881[int(idx_t)]
        _add_cand(str(tpl["dsl"]), "strict_train_881_knn_champion", tpl["feat412"])

    while len(cands) < 96:
        cands.append(cands[len(cands) % max(1, len(cands))])
    cands_96 = cands[:96]

    dsls_96 = [c[0] for c in cands_96]
    ops_96 = [c[1] for c in cands_96]
    prog_96 = np.stack([c[2] for c in cands_96], axis=0).astype(np.float32)
    base_885_map = st["base_885_dsl_to_idx"]
    sos_base_idx_96 = np.asarray([base_885_map.get(d, -1) for d in dsls_96], dtype=np.int64)
    op_onehot_96 = np.stack([encode_operator_onehot(op).astype(np.float32) for op in ops_96], axis=0)
    ns_mask_96 = np.asarray(
        [1.0 if ("crossover" in op.lower() or "champion" in op.lower()) else 0.0 for op in ops_96],
        dtype=np.float32,
    )
    op_desc_96 = np.asarray([parse_dsl_to_op_descriptor(d) for d in dsls_96], dtype=np.float32)

    return {
        "dsls_96": dsls_96,
        "prog_96": prog_96,
        "sos_base_idx_96": sos_base_idx_96,
        "op_onehot_96": op_onehot_96,
        "ns_mask_96": ns_mask_96,
        "op_desc_96": op_desc_96,
    }


def stage_live_stage1a_and_slates(
    tables: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    use_landmarks: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes live Stage 1A descriptors and 96-candidate slates strictly from `(X_tr, y_tr)`
    and stages them as padded GPU/CPU tensors for `FusedJopeNetwork.forward()`.
    """
    from concurrent.futures import ThreadPoolExecutor

    st = _get_slate_stage_state()
    B = len(tables)
    do_lm = use_landmarks in (True, None)
    do_nolm = use_landmarks in (False, None)

    def _proc_one(idx_and_tbl: Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        _, (xtr, ytr, _, _) = idx_and_tbl
        xtr_c = np.ascontiguousarray(xtr, dtype=np.float32)
        ytr_c = np.ascontiguousarray(ytr, dtype=np.float32)
        tr_sha = hashlib.sha256(xtr_c.tobytes() + ytr_c.tobytes()).hexdigest()
        mu_y = float(np.mean(ytr_c.astype(np.float64)))
        sd_y = max(float(np.std(ytr_c.astype(np.float64))), 1e-8)
        ytr_n = ((ytr_c.astype(np.float64) - mu_y) / sd_y).astype(np.float32)

        res: Dict[str, Any] = {}
        if do_lm:
            tls = st["tls"]
            if not hasattr(tls, "ws"):
                tls.ws = st["FusedLandmarkWorkspace"]()
            vec256_np, col_lm_raw, _ = tls.ws.fit_transform_and_extract_bundle_high(xtr_c, ytr_n)
            p_use = max(1, min(int(xtr_c.shape[1]), 64, int(col_lm_raw.shape[0])))
            col_lm_pad = np.zeros((64, 17), dtype=np.float32)
            col_lm_pad[:p_use, : min(17, col_lm_raw.shape[1])] = col_lm_raw[
                :p_use, : min(17, col_lm_raw.shape[1])
            ]
            v_norm = np.clip((vec256_np - st["step2_mean"]) / st["step2_std"], -6.0, 6.0)
            lm48_lm = v_norm @ st["step2_basis"]
            lm48_z_lm = np.clip((lm48_lm - st["mu_lm48"]) / st["sd_lm48"], -6.0, 6.0)
            res["lm_vec256"] = vec256_np.astype(np.float32)
            res["lm_col_lm"] = col_lm_pad
            res["lm_slate"] = _build_96_slate_arrays_from_lm48(lm48_lm, lm48_z_lm, tr_sha, st)

        if do_nolm:
            nolm_net: FusedJopeNetwork = st["nolm_helper_net"]
            with torch.no_grad():
                xt_t = torch.from_numpy(xtr_c).unsqueeze(0)
                yt_t = torch.from_numpy(ytr_n).unsqueeze(0)
                rm_t = torch.ones((1, xtr_c.shape[0]), dtype=torch.float32)
                cm_t = torch.ones((1, xtr_c.shape[1]), dtype=torch.float32)
                v_nl, clm_nl = nolm_net._extract_stage1a_fast_moments_batched_gpu(xt_t, yt_t, rm_t, cm_t)
                _, lm48_nl_t = nolm_net._run_dope_mlc_v3_and_lm48(v_nl, clm_nl, use_landmarks=False)
                lm48_nl = lm48_nl_t[0].numpy()
                lm48_z_nl = np.clip((lm48_nl - st["mu_lm48"]) / st["sd_lm48"], -6.0, 6.0)
            res["nolm_slate"] = _build_96_slate_arrays_from_lm48(lm48_nl, lm48_z_nl, tr_sha, st)

        return res

    if B > 1:
        with ThreadPoolExecutor(max_workers=min(4, B)) as ex:
            items = list(ex.map(_proc_one, enumerate(tables)))
    else:
        items = [_proc_one((0, tables[0]))]

    # Register all candidate DSLs in the global bank in deterministic order and persist once
    all_new_dsls: List[str] = []
    for it in items:
        if do_lm:
            all_new_dsls.extend(it["lm_slate"]["dsls_96"])
        if do_nolm:
            all_new_dsls.extend(it["nolm_slate"]["dsls_96"])
    register_dsls_in_global_bank(all_new_dsls, persist=True)

    out_tensors: Dict[str, torch.Tensor] = {}
    if do_lm:
        lm_vec256_b = np.stack([it["lm_vec256"] for it in items], axis=0)
        lm_col_lm_b = np.stack([it["lm_col_lm"] for it in items], axis=0)
        lm_bank_idx_b = np.asarray(
            [[_GLOBAL_DSL_TO_IDX[d] for d in it["lm_slate"]["dsls_96"]] for it in items],
            dtype=np.int64,
        )
        lm_prog_b = np.stack([it["lm_slate"]["prog_96"] for it in items], axis=0)
        lm_sos_idx_b = np.stack([it["lm_slate"]["sos_base_idx_96"] for it in items], axis=0)
        lm_op_oh_b = np.stack([it["lm_slate"]["op_onehot_96"] for it in items], axis=0)
        lm_ns_m_b = np.stack([it["lm_slate"]["ns_mask_96"] for it in items], axis=0)
        lm_op_d_b = np.stack([it["lm_slate"]["op_desc_96"] for it in items], axis=0)

        out_tensors["lm_vec256"] = torch.from_numpy(lm_vec256_b).to(device, non_blocking=True)
        out_tensors["lm_col_lm"] = torch.from_numpy(lm_col_lm_b).to(device, non_blocking=True)
        out_tensors["lm_cand_bank_idx"] = torch.from_numpy(lm_bank_idx_b).to(device, non_blocking=True)
        out_tensors["lm_cand_prog"] = torch.from_numpy(lm_prog_b).to(device, non_blocking=True)
        out_tensors["lm_cand_sos_base_idx"] = torch.from_numpy(lm_sos_idx_b).to(device, non_blocking=True)
        out_tensors["lm_cand_op_onehot"] = torch.from_numpy(lm_op_oh_b).to(device, non_blocking=True)
        out_tensors["lm_cand_ns_mask_flt"] = torch.from_numpy(lm_ns_m_b).to(device, non_blocking=True)
        out_tensors["lm_cand_op_desc"] = torch.from_numpy(lm_op_d_b).to(device, non_blocking=True)

    if do_nolm:
        nl_bank_idx_b = np.asarray(
            [[_GLOBAL_DSL_TO_IDX[d] for d in it["nolm_slate"]["dsls_96"]] for it in items],
            dtype=np.int64,
        )
        nl_prog_b = np.stack([it["nolm_slate"]["prog_96"] for it in items], axis=0)
        nl_sos_idx_b = np.stack([it["nolm_slate"]["sos_base_idx_96"] for it in items], axis=0)
        nl_op_oh_b = np.stack([it["nolm_slate"]["op_onehot_96"] for it in items], axis=0)
        nl_ns_m_b = np.stack([it["nolm_slate"]["ns_mask_96"] for it in items], axis=0)
        nl_op_d_b = np.stack([it["nolm_slate"]["op_desc_96"] for it in items], axis=0)

        out_tensors["nolm_cand_bank_idx"] = torch.from_numpy(nl_bank_idx_b).to(device, non_blocking=True)
        out_tensors["nolm_cand_prog"] = torch.from_numpy(nl_prog_b).to(device, non_blocking=True)
        out_tensors["nolm_cand_sos_base_idx"] = torch.from_numpy(nl_sos_idx_b).to(device, non_blocking=True)
        out_tensors["nolm_cand_op_onehot"] = torch.from_numpy(nl_op_oh_b).to(device, non_blocking=True)
        out_tensors["nolm_cand_ns_mask_flt"] = torch.from_numpy(nl_ns_m_b).to(device, non_blocking=True)
        out_tensors["nolm_cand_op_desc"] = torch.from_numpy(nl_op_d_b).to(device, non_blocking=True)

    return out_tensors

