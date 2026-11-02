"""H204 Bigger Proposer: GPU Trainer & Research-Only DOPE Unfreeze Engine.

Implements:
  1. Research-only DOPE unfreeze gate (flag (a): `JOPE_RESEARCH_DOPE_UNFREEZE=1`
     + nonempty `JOPE_RESEARCH_RUN_ID`; refused in serving/flywheel entrypoints;
     default path remains frozen and byte-identical).
  2. Live differentiable `DOPE` (`net.dope_model.router`) -> Stage-1B SVD slot
     projection -> `tokens_10x32` bridge with 0.0 step-0 discrepancy against
     cached `r882` context tokens.
  3. `WiderBlockFormerProposer` at widths `128 -> 256 -> 512`, plus compute-matched
     controls `R882_CONT` (width 64, frozen DOPE) and `H180_RAND` (frozen random
     width-128 trunk + trained linear readout).
  4. Stage 0 exact `91/91` native `r882` parity check (`1.1377941898164525`)
     before any candidate forward.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jope.north_star.dope_freeze_guard import (
    lock_dope_weights_and_eval_mode,
)
from research.flywheel_selfplay.h204_bigger_proposer.analysis import (
    AUTHORITY_STRING,
    BOOT_SEED,
    EVALUATOR_CONTRACT_ID,
    FOLD_SEED,
    HOST,
    MAX_NUMERIC_THREADS,
    N_FOLDS,
    OWNER,
    SEEDS,
    STAGE1_ARMS,
    WAVE,
    assign_family_folds,
    evaluate_arm_gate,
    evaluate_h178_selector,
    first_argmin_cost,
    holm_bonferroni,
    paired_family_bootstrap,
    verify_r882_native_parity,
)
from research.flywheel_selfplay.h204_bigger_proposer.labeler import (
    DEFAULT_H50_INPUT_ROOT,
    DEFAULT_H58_CACHE_ROOT,
    DEFAULT_R882_CKPT,
    EvalContextBatch,
    LabeledContextBatch,
    LocalLRUCache,
    load_h204_stage1_datasets,
)

FORBIDDEN_UNFREEZE_ENTRYPOINTS: frozenset[str] = frozenset(
    {"serving", "flywheel", "production", "promotion"}
)


def enforce_numeric_thread_limits(max_threads: int = MAX_NUMERIC_THREADS) -> dict[str, Any]:
    """Enforce thread rule (b): <= 4 numeric threads; CUDA runtime/driver threads exempt."""
    cap = max(1, min(int(max_threads), MAX_NUMERIC_THREADS))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[var] = str(cap)
    try:
        torch.set_num_threads(cap)
    except RuntimeError:
        pass
    return {
        "numeric_thread_cap": cap,
        "torch_num_threads": int(torch.get_num_threads()),
        "env": {
            k: os.environ.get(k, "")
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
        "cuda_driver_helper_threads_exempt": True,
    }


def _hash_module_parameters(module: nn.Module) -> str:
    """Compute SHA-256 over all parameter tensors in `module`."""
    h = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        h.update(name.encode("utf-8"))
        arr = param.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def configure_dope_freeze_or_research_unfreeze(
    net: nn.Module,
    *,
    request_research_unfreeze: bool = False,
    entrypoint_mode: str = "research",
    lr_dope: float = 2e-5,
) -> dict[str, Any]:
    """Enforce default DOPE freeze or opt-in research DOPE unfreeze via flag (a)."""
    mode_norm = str(entrypoint_mode).strip().lower()
    env_flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE", "0").strip()
    env_run_id = os.environ.get("JOPE_RESEARCH_RUN_ID", "").strip()

    if mode_norm in FORBIDDEN_UNFREEZE_ENTRYPOINTS:
        if request_research_unfreeze or env_flag == "1":
            raise RuntimeError(
                f"DOPE research unfreeze (`JOPE_RESEARCH_DOPE_UNFREEZE=1`) is strictly "
                f"refused in `{mode_norm}` entrypoint."
            )

    lock_dope_weights_and_eval_mode(net)

    dope_mod = getattr(net, "dope_model", None)
    if dope_mod is None:
        raise RuntimeError("Network is missing `dope_model` attribute")

    before_sha256 = _hash_module_parameters(dope_mod)

    if not request_research_unfreeze:
        return {
            "research_dope_unfreeze": False,
            "entrypoint_mode": mode_norm,
            "env_flag": env_flag,
            "research_run_id": env_run_id,
            "dope_trainable_params": 0,
            "dope_before_sha256": before_sha256,
        }

    if env_flag != "1":
        raise RuntimeError(
            "Research DOPE unfreeze requested, but `JOPE_RESEARCH_DOPE_UNFREEZE` is not '1'."
        )
    if not env_run_id:
        raise RuntimeError(
            "Research DOPE unfreeze (`JOPE_RESEARCH_DOPE_UNFREEZE=1`) requires a nonempty "
            "`JOPE_RESEARCH_RUN_ID`."
        )

    anchor_mod = getattr(net, "gamenet_anchor", None)
    if anchor_mod is not None:
        anchor_mod.eval()
        for p in anchor_mod.parameters():
            p.requires_grad_(False)

    target_mod = getattr(dope_mod, "router", dope_mod)
    trainable_count = 0
    for p in target_mod.parameters():
        p.requires_grad_(True)
        trainable_count += int(p.numel())

    def _research_safe_fused_train(net_self: nn.Module, mode: bool = True) -> nn.Module:
        nn.Module.train(net_self, mode)
        for sub in net_self.dope_model.modules():
            sub.training = False
        if hasattr(net_self, "gamenet_anchor") and isinstance(net_self.gamenet_anchor, nn.Module):
            for sub in net_self.gamenet_anchor.modules():
                sub.training = False
            for p in net_self.gamenet_anchor.parameters():
                p.requires_grad_(False)
                p.grad = None
        return net_self

    net.train = types.MethodType(_research_safe_fused_train, net)

    if trainable_count <= 0:
        raise RuntimeError("Research DOPE unfreeze enabled 0 trainable parameters")

    return {
        "research_dope_unfreeze": True,
        "entrypoint_mode": mode_norm,
        "env_flag": env_flag,
        "research_run_id": env_run_id,
        "lr_dope": float(lr_dope),
        "dope_trainable_params": int(trainable_count),
        "dope_before_sha256": before_sha256,
    }


def compute_live_tokens_10x32(
    net: nn.Module,
    dope629: torch.Tensor,
    cached_tokens_10x32: torch.Tensor,
) -> torch.Tensor:
    """Differentiably recompute Stage-1B `tokens_10x32` from `net.dope_model.router(vec256)`.

    At step 0 (`r882` weights), this matches `cached_tokens_10x32` to 0.0 float32 diff
    on CPU and < 1e-6 on CUDA while allowing gradients to flow into `net.dope_model.router`.
    """
    b_sz = dope629.shape[0]
    vec256 = dope629[:, :256]
    gates_14 = dope629[:, 384:398]
    tail_64 = dope629[:, 565:629]

    routed = net.dope_model.router(vec256, level="high")
    d629_live = torch.cat(
        [
            vec256,
            routed["task_latent"],
            routed["dist_latent"],
            gates_14,
            routed["landmark_pred"],
            routed["bb_pred"],
            tail_64,
        ],
        dim=-1,
    )

    tok0 = cached_tokens_10x32[:, 0]
    tok2 = net._svd_slot_project(d629_live, "svd_slot2_m2")
    tok4 = net._svd_slot_project(d629_live, "svd_slot4_d629_mid")
    tok5 = net._svd_slot_project(d629_live, "svd_slot5_d629_tail")
    kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(b_sz, 64)
    tok9 = net._svd_slot_project(kron_64, "svd_slot9_kron")

    return torch.stack(
        [
            tok0,
            cached_tokens_10x32[:, 1],
            tok2,
            cached_tokens_10x32[:, 3],
            tok4,
            tok5,
            cached_tokens_10x32[:, 6],
            cached_tokens_10x32[:, 7],
            cached_tokens_10x32[:, 8],
            tok9,
        ],
        dim=1,
    )


class WiderBlockFormerProposer(nn.Module):
    """Wider BlockFormer / GameNet proposer supporting widths 64, 128, 256, 512.

    Combines:
      1. Token-to-candidate multi-head cross-attention at width `W`
      2. Low-rank bilinear context-candidate interaction router
      3. Regularized trajectory kernel over the 4 revealed anchors + context/anchor summaries
      4. Width-`W` residual MLP block with zero-initialized readout head anchored on `r882`
    """

    def __init__(
        self,
        width: int = 128,
        token_dim: int = 32,
        cand_dim: int = 42,
        ctx_sum_dim: int = 29,
        anc_sum_dim: int = 24,
        dropout: float = 0.10,
        freeze_trunk: bool = False,
    ) -> None:
        super().__init__()
        if width not in (64, 128, 256, 512):
            raise ValueError(f"Unsupported proposer width: {width} (expected 64, 128, 256, or 512)")
        self.width = int(width)
        self.freeze_trunk = bool(freeze_trunk)
        n_heads = max(4, self.width // 32)
        low_rank = max(16, self.width // 8)

        self.token_norm = nn.LayerNorm(token_dim)
        self.cand_norm = nn.LayerNorm(cand_dim)
        self.token_proj = nn.Linear(token_dim, self.width)
        self.cand_proj = nn.Linear(cand_dim, self.width)
        self.ctx_meta_proj = nn.Linear(ctx_sum_dim + anc_sum_dim, self.width)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.bilinear_ctx = nn.Linear(self.width, low_rank, bias=False)
        self.bilinear_cand = nn.Linear(self.width, low_rank, bias=False)
        self.bilinear_out = nn.Linear(low_rank, self.width)

        self.anchor_traj_proj = nn.Linear(cand_dim, self.width)
        self.traj_gate = nn.Linear(self.width * 2, self.width)

        self.ffn_norm = nn.LayerNorm(self.width)
        self.ffn = nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.width * 2, self.width),
        )
        self.out_norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        if self.freeze_trunk:
            for name, param in self.named_parameters():
                if not name.startswith("head.") and name != "residual_scale":
                    param.requires_grad_(False)

    def forward(
        self,
        tokens_10x32: torch.Tensor,
        context_summary: torch.Tensor,
        anchor_summary: torch.Tensor,
        cand_features: torch.Tensor,
        anchor_cand_features: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute candidate cost scores `(B, K)` anchored around `base_scores` `(B, K)`."""
        tok_h = self.token_proj(self.token_norm(tokens_10x32))
        cand_h = self.cand_proj(self.cand_norm(cand_features))

        meta = torch.cat([context_summary, anchor_summary], dim=-1)
        meta_h = self.ctx_meta_proj(meta).unsqueeze(1)

        attn_out, _ = self.cross_attn(query=cand_h, key=tok_h, value=tok_h, need_weights=False)
        pooled_tok = tok_h.mean(dim=1, keepdim=True) + meta_h
        bi_inter = self.bilinear_out(
            F.gelu(self.bilinear_ctx(pooled_tok) * self.bilinear_cand(cand_h))
        )

        anc_h = self.anchor_traj_proj(self.cand_norm(anchor_cand_features)).mean(
            dim=1, keepdim=True
        )
        traj_cat = torch.cat([cand_h, anc_h.expand_as(cand_h)], dim=-1)
        traj_h = torch.tanh(self.traj_gate(traj_cat))

        h = cand_h + attn_out + bi_inter + 0.5 * traj_h
        h = h + self.ffn(self.ffn_norm(h))
        delta = self.head(self.out_norm(h)).squeeze(-1)
        scale = torch.clamp(self.residual_scale, -0.50, 0.50)
        return base_scores + scale * delta


def load_cpu_exact_fused_network(
    h50_input_root: Path = DEFAULT_H50_INPUT_ROOT,
    r882_ckpt_path: Path = DEFAULT_R882_CKPT,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load the canonical `FusedJopeNetwork` with `round-882.pt` state_dict on `device`."""
    import research

    micro_py = "/home/ubuntu/JopeDimeMicro/python"
    if micro_py not in sys.path:
        sys.path.insert(0, micro_py)
    micro_research = "/home/ubuntu/JopeDimeMicro/python/research"
    if hasattr(research, "__path__") and micro_research not in research.__path__:
        research.__path__.append(micro_research)

    spec = importlib.util.spec_from_file_location(
        "h204_fused", h50_input_root / "cpu_exact_fused.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load cpu_exact_fused.py spec")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    ckpt = torch.load(r882_ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net = net.to(device)
    lock_dope_weights_and_eval_mode(net)
    net.eval()
    return net


@torch.no_grad()
def compute_r882_base_scores_batch(
    net: nn.Module,
    tokens_10x32: np.ndarray,
    mask_10: np.ndarray,
    full_prog_412: np.ndarray,
    full_proj_16: np.ndarray,
    full_fitness: np.ndarray,
    full_gates: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute exact `r882` candidate cost predictions `(N, 96)` via `net.adapter_10tok.predict_costs`."""
    net.eval()
    n_ctx, n_cand, _ = full_prog_412.shape
    out = np.zeros((n_ctx, n_cand), dtype=np.float64)
    for i in range(n_ctx):
        res = net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((1, 584), device=device, dtype=torch.float32),
            prog_412=torch.as_tensor(full_prog_412[i : i + 1], device=device, dtype=torch.float32),
            proj_16=torch.as_tensor(full_proj_16[i : i + 1], device=device, dtype=torch.float32),
            base_prior_cost=torch.zeros((1, n_cand), device=device, dtype=torch.float32),
            scale=torch.tensor([0.4208], device=device, dtype=torch.float32),
            sos_prior=torch.zeros((1, n_cand, 2), device=device, dtype=torch.float32),
            mm_tokens_10x32=torch.as_tensor(
                tokens_10x32[i : i + 1], device=device, dtype=torch.float32
            ),
            mm_mask_10=torch.as_tensor(mask_10[i : i + 1], device=device).bool(),
            revealed_indices=torch.arange(4, device=device)[None],
            revealed_raw_f=torch.as_tensor(
                full_fitness[i, :4], device=device, dtype=torch.float32
            )[None],
            revealed_gate=torch.as_tensor(
                full_gates[i, :4], device=device, dtype=torch.float32
            )[None],
        )
        out[i] = res["total_cost"][0].detach().cpu().numpy().astype(np.float64)
    return out


def verify_stage0_r882_parity_on_eval_batch(
    net: nn.Module,
    val_batch: EvalContextBatch,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run Stage 0 `91/91` native `r882` DL-Val parity check (`1.1377941898164525`) before any candidate forward."""
    # Run Stage-0 parity on CPU to guarantee bit-exact IEEE-754 parity with the CPU reference
    cpu_net = copy.deepcopy(net).to("cpu")
    cpu_net.eval()
    r882_val_scores = compute_r882_base_scores_batch(
        net=cpu_net,
        tokens_10x32=val_batch.tokens_10x32,
        mask_10=val_batch.mask_10,
        full_prog_412=val_batch.full_prog_412,
        full_proj_16=val_batch.full_proj_16,
        full_fitness=val_batch.full_fitness,
        full_gates=val_batch.full_gates,
        device=torch.device("cpu"),
    )
    picks = np.argmin(r882_val_scores, axis=1)
    picked_utilities = np.array(
        [val_batch.full_utilities[i, int(picks[i])] for i in range(len(picks))],
        dtype=np.float64,
    )
    parity = verify_r882_native_parity(picked_utilities)
    if not parity["passed"]:
        raise RuntimeError(f"Stage 0 native r882 parity check FAILED: {parity}")
    return parity, r882_val_scores


def compute_proposer_ranking_loss(
    pred_scores: torch.Tensor,
    target_costs: torch.Tensor,
    cand_gates: torch.Tensor,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Combined listwise soft-argmin cross-entropy + pairwise margin ranking + Huber loss."""
    effective_costs = target_costs + 0.25 * (1.0 - cand_gates)
    target_probs = F.softmax(-effective_costs / temperature, dim=-1)
    pred_log_probs = F.log_softmax(-pred_scores / temperature, dim=-1)
    listwise_kl = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

    diff_cost = effective_costs.unsqueeze(2) - effective_costs.unsqueeze(1)
    diff_pred = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
    pair_mask = (diff_cost < -1e-4).float()
    pair_loss = (
        F.relu(0.02 - (diff_pred * (-torch.sign(diff_cost)))) * pair_mask
    ).sum() / pair_mask.sum().clamp_min(1.0)

    pred_centered = pred_scores - pred_scores.mean(dim=-1, keepdim=True)
    cost_centered = effective_costs - effective_costs.mean(dim=-1, keepdim=True)
    reg_loss = F.smooth_l1_loss(pred_centered, cost_centered)

    return listwise_kl + 0.5 * pair_loss + 0.25 * reg_loss


def train_and_eval_single_fold(
    *,
    arm_name: str,
    width: int,
    unfreeze_dope: bool,
    freeze_trunk: bool,
    base_net: nn.Module,
    train_batch: LabeledContextBatch,
    train_sub_scores: np.ndarray,
    train_full_scores: np.ndarray,
    fold_train_mask: np.ndarray,
    fold_val_mask: np.ndarray,
    val_batch: EvalContextBatch,
    val_full_scores: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 30,
    batch_size: int = 32,
    lr_proposer: float = 8e-4,
    lr_dope: float = 1.5e-5,
) -> dict[str, Any]:
    """Train one fold model on `fold_train_mask` and evaluate on `fold_val_mask` and `DL-Val`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if unfreeze_dope:
        net = copy.deepcopy(base_net).to(device)
    else:
        net = base_net

    dope_cfg = configure_dope_freeze_or_research_unfreeze(
        net,
        request_research_unfreeze=unfreeze_dope,
        entrypoint_mode="research",
        lr_dope=lr_dope,
    )

    cand_dim = int(train_batch.cand_features.shape[-1])
    proposer = WiderBlockFormerProposer(
        width=width,
        cand_dim=cand_dim,
        ctx_sum_dim=48,
        anc_sum_dim=10,
        freeze_trunk=freeze_trunk,
    ).to(device)

    param_groups: list[dict[str, Any]] = [
        {
            "params": [p for p in proposer.parameters() if p.requires_grad],
            "lr": lr_proposer,
            "weight_decay": 1e-4,
        }
    ]
    if unfreeze_dope:
        dope_params = [p for p in net.dope_model.parameters() if p.requires_grad]
        param_groups.append(
            {
                "params": dope_params,
                "lr": lr_dope,
                "weight_decay": 1e-5,
            }
        )

    optimizer = torch.optim.AdamW(param_groups)
    tr_indices = np.where(fold_train_mask)[0]
    va_indices = np.where(fold_val_mask)[0]

    with torch.no_grad():
        sample_d629 = torch.as_tensor(
            train_batch.dope629[tr_indices[:16]], dtype=torch.float32, device=device
        )
        sample_tok = torch.as_tensor(
            train_batch.tokens_10x32[tr_indices[:16]], dtype=torch.float32, device=device
        )
        live_tok0 = compute_live_tokens_10x32(net, sample_d629, sample_tok)
        step0_token_max_diff = float(torch.max(torch.abs(live_tok0 - sample_tok)).item())

    rng = np.random.default_rng(seed)
    optimizer_steps = 0
    loss_history: list[float] = []

    proposer.train()
    for _epoch in range(epochs):
        perm = rng.permutation(tr_indices)
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            t_d629 = torch.as_tensor(
                train_batch.dope629[batch_idx], dtype=torch.float32, device=device
            )
            t_tok_cached = torch.as_tensor(
                train_batch.tokens_10x32[batch_idx], dtype=torch.float32, device=device
            )
            t_ctx = torch.as_tensor(
                train_batch.lm48_z[batch_idx], dtype=torch.float32, device=device
            )
            t_anc = torch.as_tensor(
                train_batch.mask_10[batch_idx], dtype=torch.float32, device=device
            )
            t_cand_feat = torch.as_tensor(
                train_batch.cand_features[batch_idx], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                train_batch.cand_features[batch_idx, :4, :], dtype=torch.float32, device=device
            )
            t_base_sub = torch.as_tensor(
                train_sub_scores[batch_idx], dtype=torch.float32, device=device
            )
            # Cost target = -utility (lower is better)
            t_costs = torch.as_tensor(
                -train_batch.cand_utilities[batch_idx], dtype=torch.float32, device=device
            )
            t_gates = torch.as_tensor(
                train_batch.cand_gates[batch_idx], dtype=torch.float32, device=device
            )

            t_tok = (
                compute_live_tokens_10x32(net, t_d629, t_tok_cached)
                if unfreeze_dope
                else t_tok_cached
            )
            pred_scores = proposer(
                tokens_10x32=t_tok,
                context_summary=t_ctx,
                anchor_summary=t_anc,
                cand_features=t_cand_feat,
                anchor_cand_features=t_anc_feat,
                base_scores=t_base_sub,
            )
            loss = compute_proposer_ranking_loss(pred_scores, t_costs, t_gates)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), max_norm=1.0)
            if unfreeze_dope:
                torch.nn.utils.clip_grad_norm_(net.dope_model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer_steps += 1
            loss_history.append(float(loss.detach().item()))

    proposer.eval()
    net.eval()
    dope_after_sha256 = _hash_module_parameters(net.dope_model)
    dope_weights_changed = bool(dope_after_sha256 != dope_cfg["dope_before_sha256"])
    if unfreeze_dope and not dope_weights_changed:
        raise RuntimeError(
            f"Arm {arm_name} requested research DOPE unfreeze but DOPE parameter hash did not change!"
        )
    if not unfreeze_dope and dope_weights_changed:
        raise RuntimeError(
            f"Arm {arm_name} has frozen DOPE but DOPE parameter hash changed unexpectedly!"
        )

    n_cand_full = train_batch.full_cand_features.shape[1]
    oof_scores = np.zeros((len(va_indices), n_cand_full), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(va_indices), batch_size):
            b_idx = va_indices[start : start + batch_size]
            t_d629 = torch.as_tensor(train_batch.dope629[b_idx], dtype=torch.float32, device=device)
            t_tok_c = torch.as_tensor(
                train_batch.tokens_10x32[b_idx], dtype=torch.float32, device=device
            )
            t_tok = compute_live_tokens_10x32(net, t_d629, t_tok_c) if unfreeze_dope else t_tok_c
            t_ctx = torch.as_tensor(
                train_batch.lm48_z[b_idx], dtype=torch.float32, device=device
            )
            t_anc = torch.as_tensor(
                train_batch.mask_10[b_idx], dtype=torch.float32, device=device
            )
            t_full_feat = torch.as_tensor(
                train_batch.full_cand_features[b_idx], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                train_batch.full_cand_features[b_idx, :4, :], dtype=torch.float32, device=device
            )
            t_base_full = torch.as_tensor(
                train_full_scores[b_idx], dtype=torch.float32, device=device
            )
            preds = proposer(
                tokens_10x32=t_tok,
                context_summary=t_ctx,
                anchor_summary=t_anc,
                cand_features=t_full_feat,
                anchor_cand_features=t_anc_feat,
                base_scores=t_base_full,
            )
            oof_scores[start : start + len(b_idx)] = preds.detach().cpu().numpy().astype(np.float64)

    val_scores = np.zeros((len(val_batch.context_ids), n_cand_full), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(val_batch.context_ids), batch_size):
            end = min(len(val_batch.context_ids), start + batch_size)
            t_d629 = torch.as_tensor(
                val_batch.dope629[start:end], dtype=torch.float32, device=device
            )
            t_tok_c = torch.as_tensor(
                val_batch.tokens_10x32[start:end], dtype=torch.float32, device=device
            )
            t_tok = compute_live_tokens_10x32(net, t_d629, t_tok_c) if unfreeze_dope else t_tok_c
            t_ctx = torch.as_tensor(
                val_batch.lm48_z[start:end], dtype=torch.float32, device=device
            )
            t_anc = torch.as_tensor(
                val_batch.mask_10[start:end], dtype=torch.float32, device=device
            )
            t_full_feat = torch.as_tensor(
                val_batch.full_cand_features[start:end], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                val_batch.full_cand_features[start:end, :4, :], dtype=torch.float32, device=device
            )
            t_base_full = torch.as_tensor(
                val_full_scores[start:end], dtype=torch.float32, device=device
            )
            preds = proposer(
                tokens_10x32=t_tok,
                context_summary=t_ctx,
                anchor_summary=t_anc,
                cand_features=t_full_feat,
                anchor_cand_features=t_anc_feat,
                base_scores=t_base_full,
            )
            val_scores[start:end] = preds.detach().cpu().numpy().astype(np.float64)

    return {
        "oof_indices": va_indices,
        "oof_scores": oof_scores,
        "val_scores": val_scores,
        "optimizer_steps": int(optimizer_steps),
        "initial_loss": float(loss_history[0]) if loss_history else 0.0,
        "final_loss": float(loss_history[-1]) if loss_history else 0.0,
        "step0_token_max_diff": step0_token_max_diff,
        "dope_before_sha256": dope_cfg["dope_before_sha256"],
        "dope_after_sha256": dope_after_sha256,
        "dope_weights_changed": dope_weights_changed,
        "dope_trainable_params": dope_cfg["dope_trainable_params"],
    }


def run_arm_across_seeds(
    *,
    arm_name: str,
    width: int,
    unfreeze_dope: bool,
    freeze_trunk: bool,
    base_net: nn.Module,
    train_batch: LabeledContextBatch,
    train_full_scores: np.ndarray,
    train_r882_utilities: np.ndarray,
    val_batch: EvalContextBatch,
    val_full_scores: np.ndarray,
    val_r882_utilities: np.ndarray,
    device: torch.device,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = 30,
    batch_size: int = 32,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run 5-fold cross-fitting across all preregistered seeds for one arm."""
    family_folds = assign_family_folds(train_batch.family_ids, n_folds=N_FOLDS, seed=FOLD_SEED)
    ctx_folds = np.array([family_folds[f] for f in train_batch.family_ids], dtype=np.int64)

    n_tr, k_cand = train_batch.cand_indices.shape
    train_sub_scores = np.zeros((n_tr, k_cand), dtype=np.float32)
    for i in range(n_tr):
        train_sub_scores[i] = train_full_scores[i, train_batch.cand_indices[i]].astype(np.float32)

    seed_records: list[dict[str, Any]] = []
    total_optimizer_steps = 0

    for seed in seeds:
        t0 = time.time()
        oof_pred_scores = np.zeros_like(train_full_scores)
        fold_val_score_list: list[np.ndarray] = []
        fold_diagnostics: list[dict[str, Any]] = []

        for fold_id in range(N_FOLDS):
            fold_val_mask = ctx_folds == fold_id
            fold_train_mask = ~fold_val_mask
            fold_seed = int(seed * 10 + fold_id)
            f_res = train_and_eval_single_fold(
                arm_name=arm_name,
                width=width,
                unfreeze_dope=unfreeze_dope,
                freeze_trunk=freeze_trunk,
                base_net=base_net,
                train_batch=train_batch,
                train_sub_scores=train_sub_scores,
                train_full_scores=train_full_scores,
                fold_train_mask=fold_train_mask,
                fold_val_mask=fold_val_mask,
                val_batch=val_batch,
                val_full_scores=val_full_scores,
                device=device,
                seed=fold_seed,
                epochs=epochs,
                batch_size=batch_size,
            )
            oof_pred_scores[f_res["oof_indices"]] = f_res["oof_scores"]
            fold_val_score_list.append(f_res["val_scores"])
            total_optimizer_steps += int(f_res["optimizer_steps"])
            fold_diagnostics.append(
                {
                    "fold_id": int(fold_id),
                    "optimizer_steps": f_res["optimizer_steps"],
                    "initial_loss": f_res["initial_loss"],
                    "final_loss": f_res["final_loss"],
                    "step0_token_max_diff": f_res["step0_token_max_diff"],
                    "dope_before_sha256": f_res["dope_before_sha256"],
                    "dope_after_sha256": f_res["dope_after_sha256"],
                    "dope_weights_changed": f_res["dope_weights_changed"],
                    "dope_trainable_params": f_res["dope_trainable_params"],
                }
            )
            if progress_callback is not None:
                progress_callback(
                    arm=arm_name,
                    seed=seed,
                    fold=fold_id,
                    total_optimizer_steps=total_optimizer_steps,
                )

        oof_cand_utilities = np.zeros(n_tr, dtype=np.float64)
        for i in range(n_tr):
            _, u_val = first_argmin_cost(oof_pred_scores[i], train_batch.full_utilities[i])
            oof_cand_utilities[i] = u_val

        selector = evaluate_h178_selector(
            r882_oof_costs=train_r882_utilities,
            cand_oof_costs=oof_cand_utilities,
            family_ids=train_batch.family_ids,
            seed=BOOT_SEED + seed,
            higher_is_better=True,
        )

        mean_val_scores = np.mean(np.stack(fold_val_score_list, axis=0), axis=0)
        n_val = len(val_batch.context_ids)
        raw_val_utilities = np.zeros(n_val, dtype=np.float64)
        raw_val_picks = np.zeros(n_val, dtype=np.int64)
        for i in range(n_val):
            p_idx, u_val = first_argmin_cost(mean_val_scores[i], val_batch.full_utilities[i])
            raw_val_picks[i] = p_idx
            raw_val_utilities[i] = u_val

        selected_val_utilities = (
            val_r882_utilities.copy() if selector["fallback_to_r882"] else raw_val_utilities
        )

        raw_dl_val_stats = paired_family_bootstrap(
            r882_costs=val_r882_utilities,
            candidate_costs=raw_val_utilities,
            family_ids=val_batch.family_ids,
            seed=BOOT_SEED,
            higher_is_better=True,
        )
        selected_dl_val_stats = paired_family_bootstrap(
            r882_costs=val_r882_utilities,
            candidate_costs=selected_val_utilities,
            family_ids=val_batch.family_ids,
            seed=BOOT_SEED,
            higher_is_better=True,
        )

        seed_records.append(
            {
                "seed": int(seed),
                "elapsed_sec": float(time.time() - t0),
                "selector": selector,
                "fallback_rate": 1.0 if selector["fallback_to_r882"] else 0.0,
                "dl_val": raw_dl_val_stats,
                "selected_dl_val": selected_dl_val_stats,
                "folds": fold_diagnostics,
            }
        )

    return {
        "arm": arm_name,
        "width": int(width),
        "unfreeze_dope": bool(unfreeze_dope),
        "freeze_trunk": bool(freeze_trunk),
        "total_optimizer_steps": int(total_optimizer_steps),
        "seeds": seed_records,
    }


def run_h204_stage1(
    *,
    run_dir: Path,
    device_str: str = "cuda:0",
    k_candidates: int = 16,
    epochs: int = 30,
    batch_size: int = 32,
    allow_stage2: bool = False,
) -> dict[str, Any]:
    """Execute H204 Stage 0 + Stage 1 (labeling + width 128 + R882_CONT + H180_RAND)."""
    t_start = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "cache"

    thread_info = enforce_numeric_thread_limits(MAX_NUMERIC_THREADS)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    run_id = os.environ.get("JOPE_RESEARCH_RUN_ID", "").strip() or f"h204-{run_dir.name}"
    os.environ["JOPE_RESEARCH_RUN_ID"] = run_id
    os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "1"

    cache = LocalLRUCache(cache_dir=cache_dir)
    base_net = load_cpu_exact_fused_network(
        DEFAULT_H50_INPUT_ROOT, DEFAULT_R882_CKPT, device=device
    )

    train_batch, val_batch, labeler_meta = load_h204_stage1_datasets(
        cache=cache,
        h50_input_root=DEFAULT_H50_INPUT_ROOT,
        h58_cache_root=DEFAULT_H58_CACHE_ROOT,
        k_candidates=k_candidates,
        seed=SEEDS[0],
    )

    # Stage 0: Verify 91/91 exact native r882 DL-Val parity BEFORE any candidate forward
    stage0_parity, val_full_scores = verify_stage0_r882_parity_on_eval_batch(
        base_net, val_batch, device=device
    )
    val_r882_picks = np.argmin(val_full_scores, axis=1)
    val_r882_utilities = np.array(
        [val_batch.full_utilities[i, int(val_r882_picks[i])] for i in range(len(val_r882_picks))],
        dtype=np.float64,
    )

    # Precompute frozen r882 anchor scores on DL-Train (1,219 contexts)
    train_full_scores = compute_r882_base_scores_batch(
        net=base_net,
        tokens_10x32=train_batch.tokens_10x32,
        mask_10=train_batch.mask_10,
        full_prog_412=train_batch.full_prog_412,
        full_proj_16=train_batch.full_proj_16,
        full_fitness=train_batch.full_fitness,
        full_gates=train_batch.full_gates,
        device=device,
    )
    train_r882_picks = np.argmin(train_full_scores, axis=1)
    train_r882_utilities = np.array(
        [
            train_batch.full_utilities[i, int(train_r882_picks[i])]
            for i in range(len(train_r882_picks))
        ],
        dtype=np.float64,
    )

    progress_path = run_dir / "progress.json"

    def _on_progress(arm: str, seed: int, fold: int, total_optimizer_steps: int) -> None:
        snap = {
            "wave": WAVE,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "current_arm": arm,
            "current_seed": int(seed),
            "current_fold": int(fold),
            "optimizer_steps_in_arm": int(total_optimizer_steps),
            "elapsed_sec": float(time.time() - t_start),
            "gpu_allocated_mib": (
                float(torch.cuda.memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
            "gpu_max_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
        }
        progress_path.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")

    arm_specs: list[tuple[str, int, bool, bool]] = [
        ("R882_CONT", 64, False, False),
        ("H180_RAND", 128, False, True),
        ("PROPOSER_W128", 128, True, False),
    ]

    arms_out: dict[str, Any] = {}
    for arm_name, width, unfreeze_dope, freeze_trunk in arm_specs:
        arms_out[arm_name] = run_arm_across_seeds(
            arm_name=arm_name,
            width=width,
            unfreeze_dope=unfreeze_dope,
            freeze_trunk=freeze_trunk,
            base_net=base_net,
            train_batch=train_batch,
            train_full_scores=train_full_scores,
            train_r882_utilities=train_r882_utilities,
            val_batch=val_batch,
            val_full_scores=val_full_scores,
            val_r882_utilities=val_r882_utilities,
            device=device,
            seeds=SEEDS,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=_on_progress,
        )

    control_gains = [
        float(np.mean([s["dl_val"]["mean_gain"] for s in arms_out[c]["seeds"]]))
        for c in ("R882_CONT", "H180_RAND")
    ]
    control_best_gain = float(max(control_gains))

    stage1_gate = evaluate_arm_gate(
        seed_results=arms_out["PROPOSER_W128"]["seeds"],
        control_best_gain=control_best_gain,
    )

    stage2_executed = False
    if stage1_gate["passed"] and allow_stage2:
        stage2_executed = True
        for arm_name, width in (("PROPOSER_W256", 256), ("PROPOSER_W512", 512)):
            arms_out[arm_name] = run_arm_across_seeds(
                arm_name=arm_name,
                width=width,
                unfreeze_dope=True,
                freeze_trunk=False,
                base_net=base_net,
                train_batch=train_batch,
                train_full_scores=train_full_scores,
                train_r882_utilities=train_r882_utilities,
                val_batch=val_batch,
                val_full_scores=val_full_scores,
                val_r882_utilities=val_r882_utilities,
                device=device,
                seeds=SEEDS,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=_on_progress,
            )

    raw_p_map = {
        arm: float(np.mean([s["dl_val"]["p_one_sided"] for s in arms_out[arm]["seeds"]]))
        for arm in arms_out
    }
    holm_p_map = holm_bonferroni(raw_p_map)

    summary = {
        "wave": WAVE,
        "owner": OWNER,
        "host": HOST,
        "authority": AUTHORITY_STRING,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "device": str(device),
        "gpu_name": gpu_name,
        "research_run_id": run_id,
        "research_dope_unfreeze": True,
        "thread_rule_b": thread_info,
        "stage0_native_r882_parity": stage0_parity,
        "labeler": labeler_meta,
        "stage1_arms": list(STAGE1_ARMS),
        "arms": arms_out,
        "raw_p_one_sided": raw_p_map,
        "holm_adjusted_p": holm_p_map,
        "control_best_gain": control_best_gain,
        "stage1_w128_gate": stage1_gate,
        "stage2_executed": stage2_executed,
        "decision": "ADVANCE_TO_W256" if stage1_gate["passed"] else "NO_ADVANCE",
        "elapsed_sec": float(time.time() - t_start),
    }

    out_file = run_dir / "stage1_results.json"
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
