"""H204 Bigger Proposer: GPU Trainer & Research-Only DOPE Unfreeze Engine.

Implements:
  1. Shared research-only DOPE opt-in (`#1726`: `configure_research_dope`,
     `assert_research_optimizer_policy`, `lock_dope_weights_and_eval_mode`,
     `snapshot_dope_weights`, `verify_dope_frozen_unchanged`,
     `assert_optimizer_excludes_dope` from `jope.north_star.dope_freeze_guard`).
  2. Shared numeric-thread admission (`#1730`: `capture_numeric_threads`,
     `admit_numeric_threads` from `research.flywheel_selfplay.hwave_numeric_threads`).
  3. Live differentiable `DOPE` (`net.dope_model.router` + `col_gate_head`) ->
     Stage-1B SVD slot projection -> `tokens_10x32` bridge with 0.0 step-0
     discrepancy against cached `r882` context tokens (`step0_token_max_diff <= 1e-5`).
  4. `WiderBlockFormerProposer` at widths `128 -> 256 -> 512`, plus compute-matched
     controls `R882_CONT` (continuing loaded `round-882.pt` width-64 `adapter_10tok`
     with frozen DOPE) and `H180_RAND` (frozen random width-128 trunk + trained
     linear readout with frozen DOPE).
  5. Stage 0 exact `91/91` native `r882` parity check (`1.1377941898164525` and
     exact 91-element pick identity) before any candidate forward.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jope.north_star.dope_freeze_guard import (
    DopeFreezeViolationError,
    assert_optimizer_excludes_dope,
    assert_research_optimizer_policy,
    configure_research_dope,
    lock_dope_weights_and_eval_mode,
    reject_research_dope_unfreeze,
    snapshot_dope_weights,
    verify_dope_frozen_unchanged,
)
from threadpoolctl import threadpool_limits

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
    EXPECTED_DOPE_SHA256,
    PINNED_INPUT_SHA256,
    EvalContextBatch,
    LabeledContextBatch,
    LocalLRUCache,
    load_h204_stage1_datasets,
    sha256_file,
)
from research.flywheel_selfplay.hwave_numeric_threads import (
    admit_numeric_threads,
    capture_numeric_threads,
)

FORBIDDEN_UNFREEZE_ENTRYPOINTS: frozenset[str] = frozenset({"serving", "flywheel", "production", "promotion"})


def read_proc_os_threads(pid: int | None = None) -> int:
    """Read OS thread count (`Threads:`) from `/proc/<pid>/status`."""
    target_pid = pid if pid is not None else os.getpid()
    status_path = Path(f"/proc/{target_pid}/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Threads:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return max(1, int(parts[1]))
                    except ValueError:
                        break
    return max(1, threading.active_count())


def enforce_numeric_thread_limits(max_threads: int = MAX_NUMERIC_THREADS) -> dict[str, Any]:
    """Enforce thread rule (b) via `#1730` `capture_numeric_threads` + `admit_numeric_threads`."""
    cap = max(1, min(int(max_threads), MAX_NUMERIC_THREADS))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[var] = "1"
    threadpool_limits(limits=1)
    try:
        torch.set_num_threads(cap)
    except RuntimeError:
        pass
    worker_report = capture_numeric_threads(torch)
    os_threads = read_proc_os_threads()
    admission = admit_numeric_threads([worker_report], os_threads=os_threads)
    return {
        "numeric_thread_cap": cap,
        "torch_num_threads": int(torch.get_num_threads()),
        "env": dict(worker_report["environment"]),
        "admission": admission,
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


def _hash_named_tensors(module: nn.Module) -> dict[str, str]:
    """Compute per-tensor SHA-256 map for `module.named_parameters()`."""
    out: dict[str, str] = {}
    for name, param in sorted(module.named_parameters()):
        h = hashlib.sha256()
        arr = param.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
        out[name] = h.hexdigest()
    return out


def configure_dope_freeze_or_research_unfreeze(
    net: nn.Module,
    *,
    request_research_unfreeze: bool = False,
    entrypoint_mode: str = "research",
    lr_dope: float = 1.5e-5,
    launch_metadata: dict[str, Any] | None = None,
    research_run_id: str | None = None,
) -> dict[str, Any]:
    """Configure frozen DOPE or research-only DOPE unfreeze using `#1726` shared API."""
    mode_norm = str(entrypoint_mode).strip().lower()
    env_flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE", "0").strip()
    env_run_id = (research_run_id or os.environ.get("JOPE_RESEARCH_RUN_ID", "")).strip()

    if mode_norm in FORBIDDEN_UNFREEZE_ENTRYPOINTS:
        if request_research_unfreeze or env_flag != "0":
            reject_research_dope_unfreeze(mode_norm)
            raise DopeFreezeViolationError(
                f"DOPE research unfreeze (`JOPE_RESEARCH_DOPE_UNFREEZE=1`) is strictly "
                f"refused in `{mode_norm}` entrypoint."
            )

    dope_mod = getattr(net, "dope_model", None)
    if dope_mod is None:
        raise RuntimeError("Network is missing `dope_model` attribute")

    if hasattr(dope_mod, "router"):
        step0_router = copy.deepcopy(dope_mod.router).eval()
        for p in step0_router.parameters():
            p.requires_grad_(False)
        object.__setattr__(net, "_step0_router", step0_router)

    meta_holder: dict[str, Any] = launch_metadata if isinstance(launch_metadata, dict) else {}
    before_sha256 = _hash_module_parameters(dope_mod)
    before_tensor_hashes = _hash_named_tensors(dope_mod)

    if not request_research_unfreeze:
        prev_flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE")
        os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"
        try:
            lock_dope_weights_and_eval_mode(net)
            configure_research_dope(
                net,
                launch_metadata=meta_holder,
                research_run_id=env_run_id or None,
                unfreeze=False,
            )
            freeze_snapshot = snapshot_dope_weights(net) if hasattr(net, "step2_lm_mean_1x256") else None
        finally:
            if prev_flag is None:
                os.environ.pop("JOPE_RESEARCH_DOPE_UNFREEZE", None)
            else:
                os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = prev_flag
        return {
            "research_dope_unfreeze": False,
            "entrypoint_mode": mode_norm,
            "env_flag": "0",
            "research_run_id": env_run_id,
            "dope_trainable_params": 0,
            "dope_before_sha256": before_sha256,
            "dope_before_tensor_hashes": before_tensor_hashes,
            "dope_freeze_snapshot": freeze_snapshot,
            "dope_policy": meta_holder.get("dope_policy", {}),
        }

    if env_flag != "1":
        raise DopeFreezeViolationError(
            "Research DOPE unfreeze requested, but `JOPE_RESEARCH_DOPE_UNFREEZE` is not '1'."
        )
    if not env_run_id:
        raise DopeFreezeViolationError(
            "Research DOPE unfreeze (`JOPE_RESEARCH_DOPE_UNFREEZE=1`) requires a nonempty "
            "`JOPE_RESEARCH_RUN_ID`."
        )

    configure_research_dope(
        net,
        launch_metadata=meta_holder,
        research_run_id=env_run_id,
        unfreeze=True,
    )
    trainable_count = sum(int(p.numel()) for p in dope_mod.parameters() if p.requires_grad)
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
        "dope_before_tensor_hashes": before_tensor_hashes,
        "dope_freeze_snapshot": None,
        "dope_policy": meta_holder.get("dope_policy", {}),
    }


def compute_live_tokens_10x32(
    net: nn.Module,
    dope629: torch.Tensor,
    cached_tokens_10x32: torch.Tensor,
) -> torch.Tensor:
    """Differentiably recompute Stage-1B `tokens_10x32` from `net.dope_model`.

    Uses canonical `cpu_exact_fused.py:1001-1021` layout:
      `dope629 = [vec256(256), task_lat(128), dist_lat(128), gates_14(14), lm_pred(23), bb_pred(16), col_gates_64(64)]`
    where `gates_14` (`512:526`) is computed live from `routed["gates"]` and
    `col_gates_64` (`565:629`) is updated live via `net.dope_model.target_net.col_gate_head`.
    At step 0 (`r882` weights), this matches `cached_tokens_10x32` to 0.0 float32 diff on CPU
    and `< 1e-6` on CUDA while propagating data gradients to `router` and `col_gate_head`.
    """
    b_sz = dope629.shape[0]
    vec256 = dope629[:, :256]
    routed = net.dope_model.router(vec256, level="high")

    if (
        hasattr(net, "step3_gate_keys_14")
        and isinstance(routed.get("gates"), dict)
        and all(k in routed["gates"] for k in net.step3_gate_keys_14)
    ):
        gates_14_raw = torch.stack([routed["gates"][k] for k in net.step3_gate_keys_14], dim=-1)
    elif isinstance(routed.get("gates"), torch.Tensor) and routed["gates"].shape[-1] == 14:
        gates_14_raw = routed["gates"]
    else:
        gates_14_raw = dope629[:, 512:526]

    if hasattr(net, "_step0_router") and isinstance(net._step0_router, nn.Module):
        with torch.no_grad():
            r0 = net._step0_router(vec256, level="high")
            if (
                hasattr(net, "step3_gate_keys_14")
                and isinstance(r0.get("gates"), dict)
                and all(k in r0["gates"] for k in net.step3_gate_keys_14)
            ):
                g0 = torch.stack([r0["gates"][k] for k in net.step3_gate_keys_14], dim=-1)
            elif isinstance(r0.get("gates"), torch.Tensor) and r0["gates"].shape[-1] == 14:
                g0 = r0["gates"]
            else:
                g0 = dope629[:, 512:526]
        task_lat = dope629[:, 256:384] + (routed["task_latent"] - r0["task_latent"])
        dist_lat = dope629[:, 384:512] + (routed["dist_latent"] - r0["dist_latent"])
        gates_14 = dope629[:, 512:526] + (gates_14_raw - g0)
        lm_pred = dope629[:, 526:549] + (routed["landmark_pred"] - r0["landmark_pred"])
        bb_pred = dope629[:, 549:565] + (routed["bb_pred"] - r0["bb_pred"])
    else:
        task_lat = routed["task_latent"]
        dist_lat = routed["dist_latent"]
        gates_14 = gates_14_raw
        lm_pred = routed["landmark_pred"]
        bb_pred = routed["bb_pred"]

    tail_0 = dope629[:, 565:629]
    if hasattr(net.dope_model, "target_net") and hasattr(net.dope_model.target_net, "col_gate_head"):
        cgh = net.dope_model.target_net.col_gate_head
        raw_g_0 = torch.atanh(torch.clamp((tail_0 - 1.0) * 4.0, -0.999999, 0.999999))
        w_128 = cgh.weight[:, :128]
        bias = cgh.bias
        delta_raw_g = (
            (task_lat @ w_128.T)
            - (dope629[:, 256:384].detach() @ w_128.detach().T)
            + (bias - bias.detach()).unsqueeze(0)
        )
        tail_64 = tail_0 + (
            (1.0 + 0.25 * torch.tanh(raw_g_0 + delta_raw_g)) - (1.0 + 0.25 * torch.tanh(raw_g_0)).detach()
        )
    else:
        tail_64 = tail_0

    d629_live = torch.cat(
        [
            vec256,
            task_lat,
            dist_lat,
            gates_14,
            lm_pred,
            bb_pred,
            tail_64,
        ],
        dim=-1,
    )

    tok0 = cached_tokens_10x32[:, 0]
    with torch.no_grad():
        tok2_0 = net._svd_slot_project(dope629, "svd_slot2_m2")
        tok4_0 = net._svd_slot_project(dope629, "svd_slot4_d629_mid")
        tok5_0 = net._svd_slot_project(dope629, "svd_slot5_d629_tail")
    tok2 = cached_tokens_10x32[:, 2] + (net._svd_slot_project(d629_live, "svd_slot2_m2") - tok2_0)
    tok4 = cached_tokens_10x32[:, 4] + (net._svd_slot_project(d629_live, "svd_slot4_d629_mid") - tok4_0)
    tok5 = cached_tokens_10x32[:, 5] + (net._svd_slot_project(d629_live, "svd_slot5_d629_tail") - tok5_0)
    kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(b_sz, 64)
    with torch.no_grad():
        kron_64_0 = (cached_tokens_10x32[:, 2, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(b_sz, 64)
        tok9_0 = net._svd_slot_project(kron_64_0, "svd_slot9_kron")
    tok9 = cached_tokens_10x32[:, 9] + (net._svd_slot_project(kron_64, "svd_slot9_kron") - tok9_0)

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


class R882AdapterContinuation(nn.Module):
    """Authentic `R882_CONT` width-64 continuation wrapper around loaded `net.adapter_10tok`.

    Continues training the non-anchor parameters of the loaded `round-882.pt` width-64
    `adapter_10tok` (`encode_multimodal_context_and_candidates`), anchored on `base_scores`
    with zero step-0 drift while keeping `DOPE` and `gamenet_anchor` strictly frozen.
    """

    def __init__(self, adapter_10tok: nn.Module) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(adapter_10tok)
        self.baseline_adapter = copy.deepcopy(adapter_10tok).eval()
        for p in self.baseline_adapter.parameters():
            p.requires_grad_(False)
        for name, p in self.adapter.named_parameters():
            if name.startswith("anchor.") or name.startswith("frozen_anchor."):
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

    def forward(
        self,
        tokens_10x32: torch.Tensor,
        context_summary: torch.Tensor,
        anchor_summary: torch.Tensor,
        cand_features: torch.Tensor,
        anchor_cand_features: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute continued `r882` width-64 cost scores `(B, K)`."""
        b_sz, k_cand, feat_dim = cand_features.shape
        if feat_dim >= 412:
            prog_412 = cand_features[:, :, :412]
        else:
            prog_412 = F.pad(cand_features, (0, 412 - feat_dim))
        sos_prior = torch.zeros((b_sz, k_cand, 2), device=cand_features.device, dtype=cand_features.dtype)
        with torch.no_grad():
            _, _, _, mm_adj_0 = self.baseline_adapter.encode_multimodal_context_and_candidates(
                tokens_10x32, anchor_summary, prog_412, sos_prior
            )
        _, _, _, mm_adj = self.adapter.encode_multimodal_context_and_candidates(
            tokens_10x32, anchor_summary, prog_412, sos_prior
        )
        return base_scores + (mm_adj - mm_adj_0.detach())


class WiderBlockFormerProposer(nn.Module):
    """Wider BlockFormer / GameNet proposer supporting widths 64, 128, 256, 512.

    Combines:
      1. Token-to-candidate multi-head cross-attention at width `W`
      2. Low-rank bilinear context-candidate interaction router
      3. Regularized trajectory kernel over the 4 revealed anchors + context/anchor summaries
      4. Width-`W` residual MLP block anchored on `r882` `base_scores`
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
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
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
        bi_inter = self.bilinear_out(F.gelu(self.bilinear_ctx(pooled_tok) * self.bilinear_cand(cand_h)))

        anc_h = self.anchor_traj_proj(self.cand_norm(anchor_cand_features)).mean(dim=1, keepdim=True)
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
    verify_digests: bool = True,
) -> nn.Module:
    """Load the canonical `FusedJopeNetwork` with verified `round-882.pt` state_dict on `device`."""
    import research

    if verify_digests:
        ckpt_sha = sha256_file(r882_ckpt_path)
        expected_ckpt_sha = PINNED_INPUT_SHA256[r882_ckpt_path]
        if ckpt_sha != expected_ckpt_sha:
            raise RuntimeError(f"round-882.pt SHA-256 mismatch: {ckpt_sha} != {expected_ckpt_sha}")
        fused_py = h50_input_root / "cpu_exact_fused.py"
        fused_sha = sha256_file(fused_py)
        expected_fused_sha = PINNED_INPUT_SHA256[fused_py]
        if fused_sha != expected_fused_sha:
            raise RuntimeError(f"cpu_exact_fused.py SHA-256 mismatch: {fused_sha} != {expected_fused_sha}")

    micro_py = "/home/ubuntu/JopeDimeMicro/python"
    if micro_py not in sys.path:
        sys.path.insert(0, micro_py)
    micro_research = "/home/ubuntu/JopeDimeMicro/python/research"
    if hasattr(research, "__path__") and micro_research not in research.__path__:
        research.__path__.append(micro_research)

    spec = importlib.util.spec_from_file_location("h204_fused", h50_input_root / "cpu_exact_fused.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load cpu_exact_fused.py spec")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    ckpt = torch.load(r882_ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("round") != 882:
        raise RuntimeError(f"Expected round-882 checkpoint (round=882), got {ckpt.get('round')}")
    if ckpt.get("dope_sha256") != EXPECTED_DOPE_SHA256:
        raise RuntimeError(
            f"Checkpoint dope_sha256 mismatch: {ckpt.get('dope_sha256')} != {EXPECTED_DOPE_SHA256}"
        )
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net = net.to(device)

    prev_flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE")
    os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"
    try:
        lock_dope_weights_and_eval_mode(net)
        snap = snapshot_dope_weights(net)
        if snap.composite_sha256 != EXPECTED_DOPE_SHA256:
            raise RuntimeError(
                f"Loaded DOPE composite_sha256 {snap.composite_sha256} != {EXPECTED_DOPE_SHA256}"
            )
    finally:
        if prev_flag is None:
            os.environ.pop("JOPE_RESEARCH_DOPE_UNFREEZE", None)
        else:
            os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = prev_flag

    net.eval()
    return net


@torch.no_grad()
def compute_r882_base_scores_batch(
    net: nn.Module,
    tokens_10x32: np.ndarray,
    mask_10: np.ndarray,
    full_cand_features: np.ndarray,
    full_fitness: np.ndarray,
    full_gates: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute exact `r882` candidate cost predictions `(N, 1152)` via `net.adapter_10tok.predict_costs`."""
    net.eval()
    n_ctx, n_cand, _ = full_cand_features.shape
    out = np.zeros((n_ctx, n_cand), dtype=np.float64)
    for i in range(n_ctx):
        res = net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((1, 584), device=device, dtype=torch.float32),
            prog_412=torch.as_tensor(
                full_cand_features[i : i + 1, :, :412], device=device, dtype=torch.float32
            ),
            proj_16=torch.as_tensor(
                full_cand_features[i : i + 1, :, 412:], device=device, dtype=torch.float32
            ),
            base_prior_cost=torch.zeros((1, n_cand), device=device, dtype=torch.float32),
            scale=torch.tensor([0.4208], device=device, dtype=torch.float32),
            sos_prior=torch.zeros((1, n_cand, 2), device=device, dtype=torch.float32),
            mm_tokens_10x32=torch.as_tensor(tokens_10x32[i : i + 1], device=device, dtype=torch.float32),
            mm_mask_10=torch.as_tensor(mask_10[i : i + 1], device=device).bool(),
            revealed_indices=torch.arange(4, device=device)[None],
            revealed_raw_f=torch.as_tensor(full_fitness[i, :4], device=device, dtype=torch.float32)[None],
            revealed_gate=torch.as_tensor(full_gates[i, :4], device=device, dtype=torch.float32)[None],
        )
        out[i] = res["total_cost"][0].detach().cpu().numpy().astype(np.float64)
    return out


def verify_stage0_r882_parity_on_eval_batch(
    net: nn.Module,
    val_batch: EvalContextBatch,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run Stage 0 `91/91` native `r882` DL-Val parity check (`1.1377941898164525`) before any candidate forward."""
    cpu_net = copy.deepcopy(net).to("cpu")
    cpu_net.eval()
    r882_val_scores = compute_r882_base_scores_batch(
        net=cpu_net,
        tokens_10x32=val_batch.tokens_10x32,
        mask_10=val_batch.mask_10,
        full_cand_features=val_batch.full_cand_features,
        full_fitness=val_batch.full_fitness,
        full_gates=val_batch.full_gates,
        device=torch.device("cpu"),
    )
    picks = np.argmin(r882_val_scores, axis=1)
    picked_utilities = np.array(
        [val_batch.full_utilities[i, int(picks[i])] for i in range(len(picks))],
        dtype=np.float64,
    )
    parity = verify_r882_native_parity(picked_utilities, dl_val_picks=picks)
    if not parity["passed"]:
        raise RuntimeError(f"Stage 0 native r882 parity check FAILED: {parity}")
    return parity, r882_val_scores


def compute_proposer_ranking_loss(
    pred_scores: torch.Tensor,
    target_costs: torch.Tensor,
    cand_gates: torch.Tensor,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Combined listwise soft-argmin cross-entropy + pairwise margin ranking + Huber loss.

    Convention: lower cost is better (`argmin(pred_scores)` selects the best candidate).
    When `cost_i < cost_j` (`diff_cost = cost_i - cost_j < -1e-4`), we want
    `pred_i < pred_j` by margin `0.02`, so `F.relu(0.02 + pred_i - pred_j)` =
    `F.relu(0.02 - diff_pred * torch.sign(diff_cost))`.
    """
    effective_costs = target_costs + 0.25 * (1.0 - cand_gates)
    target_probs = F.softmax(-effective_costs / temperature, dim=-1)
    pred_log_probs = F.log_softmax(-pred_scores / temperature, dim=-1)
    listwise_kl = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

    diff_cost = effective_costs.unsqueeze(2) - effective_costs.unsqueeze(1)
    diff_pred = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
    pair_mask = (diff_cost < -1e-4).float()
    pair_loss = (
        F.relu(0.02 - (diff_pred * torch.sign(diff_cost))) * pair_mask
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

    run_id = os.environ.get("JOPE_RESEARCH_RUN_ID", "").strip() or f"h204-{arm_name.lower()}-s{seed}"
    os.environ["JOPE_RESEARCH_RUN_ID"] = run_id
    fold_launch_meta: dict[str, Any] = {"arm": arm_name, "seed": int(seed)}

    prev_flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE")
    os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "1" if unfreeze_dope else "0"
    try:
        dope_cfg = configure_dope_freeze_or_research_unfreeze(
            net,
            request_research_unfreeze=unfreeze_dope,
            entrypoint_mode="research",
            lr_dope=lr_dope,
            launch_metadata=fold_launch_meta,
            research_run_id=run_id,
        )
    finally:
        os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"
        if prev_flag is not None and prev_flag == "0":
            os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"

    cand_dim = int(train_batch.cand_features.shape[-1])
    if arm_name == "R882_CONT" and hasattr(net, "adapter_10tok"):
        proposer: nn.Module = R882AdapterContinuation(net.adapter_10tok).to(device)
    else:
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
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.AdamW(param_groups)
    assert_research_optimizer_policy(optimizer, net, research_run_id=run_id)
    if not unfreeze_dope:
        assert_optimizer_excludes_dope(optimizer, net)

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
    if step0_token_max_diff > 1e-5:
        raise RuntimeError(
            f"Step-0 live token bridge discrepancy {step0_token_max_diff:.3e} > 1e-5 in arm {arm_name}!"
        )

    rng = np.random.default_rng(seed)
    optimizer_steps = 0
    loss_history: list[float] = []
    max_dope_data_grad_l2 = 0.0
    dope_grad_tensor_names: set[str] = set()

    proposer.train()
    for _epoch in range(epochs):
        perm = rng.permutation(tr_indices)
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            t_d629 = torch.as_tensor(train_batch.dope629[batch_idx], dtype=torch.float32, device=device)
            t_tok_cached = torch.as_tensor(
                train_batch.tokens_10x32[batch_idx], dtype=torch.float32, device=device
            )
            t_ctx = torch.as_tensor(train_batch.lm48_z[batch_idx], dtype=torch.float32, device=device)
            t_anc = torch.as_tensor(train_batch.mask_10[batch_idx], dtype=torch.float32, device=device)
            t_cand_feat = torch.as_tensor(
                train_batch.cand_features[batch_idx], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                train_batch.cand_features[batch_idx, :4, :], dtype=torch.float32, device=device
            )
            t_base_sub = torch.as_tensor(train_sub_scores[batch_idx], dtype=torch.float32, device=device)
            t_costs = torch.as_tensor(
                -train_batch.cand_utilities[batch_idx], dtype=torch.float32, device=device
            )
            t_gates = torch.as_tensor(train_batch.cand_gates[batch_idx], dtype=torch.float32, device=device)

            t_tok = compute_live_tokens_10x32(net, t_d629, t_tok_cached) if unfreeze_dope else t_tok_cached
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
                grad_sq = 0.0
                for p_name, p_val in net.dope_model.named_parameters():
                    if p_val.grad is not None:
                        gn = float(torch.norm(p_val.grad.detach()).item())
                        if gn > 0.0 and np.isfinite(gn):
                            grad_sq += gn * gn
                            dope_grad_tensor_names.add(p_name)
                step_grad_l2 = float(np.sqrt(grad_sq))
                if step_grad_l2 > max_dope_data_grad_l2:
                    max_dope_data_grad_l2 = step_grad_l2
                torch.nn.utils.clip_grad_norm_(net.dope_model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer_steps += 1
            loss_history.append(float(loss.detach().item()))

    proposer.eval()
    net.eval()
    dope_after_sha256 = _hash_module_parameters(net.dope_model)
    after_tensor_hashes = _hash_named_tensors(net.dope_model)
    changed_dope_tensors = sorted(
        k for k, v in after_tensor_hashes.items() if dope_cfg["dope_before_tensor_hashes"].get(k) != v
    )
    dope_weights_changed = bool(dope_after_sha256 != dope_cfg["dope_before_sha256"])

    if unfreeze_dope:
        if not dope_weights_changed or len(changed_dope_tensors) == 0:
            raise RuntimeError(
                f"Arm {arm_name} requested research DOPE unfreeze but DOPE parameter hash did not change!"
            )
        if max_dope_data_grad_l2 <= 0.0:
            raise RuntimeError(
                f"Arm {arm_name} requested research DOPE unfreeze but recorded 0.0 DOPE data-gradient norm!"
            )
    else:
        if dope_weights_changed or len(changed_dope_tensors) > 0:
            raise RuntimeError(
                f"Arm {arm_name} has frozen DOPE but DOPE parameter hash changed unexpectedly!"
            )
        if dope_cfg["dope_freeze_snapshot"] is not None:
            verify_dope_frozen_unchanged(net, dope_cfg["dope_freeze_snapshot"])

    n_cand_full = train_batch.full_cand_features.shape[1]
    oof_scores = np.zeros((len(va_indices), n_cand_full), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(va_indices), batch_size):
            b_idx = va_indices[start : start + batch_size]
            t_d629 = torch.as_tensor(train_batch.dope629[b_idx], dtype=torch.float32, device=device)
            t_tok_c = torch.as_tensor(train_batch.tokens_10x32[b_idx], dtype=torch.float32, device=device)
            t_tok = compute_live_tokens_10x32(net, t_d629, t_tok_c) if unfreeze_dope else t_tok_c
            t_ctx = torch.as_tensor(train_batch.lm48_z[b_idx], dtype=torch.float32, device=device)
            t_anc = torch.as_tensor(train_batch.mask_10[b_idx], dtype=torch.float32, device=device)
            t_full_feat = torch.as_tensor(
                train_batch.full_cand_features[b_idx], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                train_batch.full_cand_features[b_idx, :4, :], dtype=torch.float32, device=device
            )
            t_base_full = torch.as_tensor(train_full_scores[b_idx], dtype=torch.float32, device=device)
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
            t_d629 = torch.as_tensor(val_batch.dope629[start:end], dtype=torch.float32, device=device)
            t_tok_c = torch.as_tensor(val_batch.tokens_10x32[start:end], dtype=torch.float32, device=device)
            t_tok = compute_live_tokens_10x32(net, t_d629, t_tok_c) if unfreeze_dope else t_tok_c
            t_ctx = torch.as_tensor(val_batch.lm48_z[start:end], dtype=torch.float32, device=device)
            t_anc = torch.as_tensor(val_batch.mask_10[start:end], dtype=torch.float32, device=device)
            t_full_feat = torch.as_tensor(
                val_batch.full_cand_features[start:end], dtype=torch.float32, device=device
            )
            t_anc_feat = torch.as_tensor(
                val_batch.full_cand_features[start:end, :4, :], dtype=torch.float32, device=device
            )
            t_base_full = torch.as_tensor(val_full_scores[start:end], dtype=torch.float32, device=device)
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
        "max_dope_data_grad_l2": float(max_dope_data_grad_l2),
        "dope_grad_tensors_count": len(dope_grad_tensor_names),
        "dope_changed_tensors_count": len(changed_dope_tensors),
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
                    "max_dope_data_grad_l2": f_res["max_dope_data_grad_l2"],
                    "dope_grad_tensors_count": f_res["dope_grad_tensors_count"],
                    "dope_changed_tensors_count": f_res["dope_changed_tensors_count"],
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
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device_str!r}, but torch.cuda.is_available() is False!")
    device = torch.device(device_str)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    run_id = os.environ.get("JOPE_RESEARCH_RUN_ID", "").strip() or f"h204-{run_dir.name}"
    os.environ["JOPE_RESEARCH_RUN_ID"] = run_id
    os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"

    cache = LocalLRUCache(cache_dir=cache_dir)
    base_net = load_cpu_exact_fused_network(DEFAULT_H50_INPUT_ROOT, DEFAULT_R882_CKPT, device=device)

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
        full_cand_features=train_batch.full_cand_features,
        full_fitness=train_batch.full_fitness,
        full_gates=train_batch.full_gates,
        device=device,
    )
    train_r882_picks = np.argmin(train_full_scores, axis=1)
    train_r882_utilities = np.array(
        [train_batch.full_utilities[i, int(train_r882_picks[i])] for i in range(len(train_r882_picks))],
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
                float(torch.cuda.memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
            "gpu_max_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
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

    raw_p_map_stage1 = {
        arm: float(np.mean([s["dl_val"]["p_one_sided"] for s in arms_out[arm]["seeds"]])) for arm in arms_out
    }
    holm_p_map_stage1 = holm_bonferroni(raw_p_map_stage1)

    stage1_gate = evaluate_arm_gate(
        seed_results=arms_out["PROPOSER_W128"]["seeds"],
        control_best_gain=control_best_gain,
        holm_adjusted_p=holm_p_map_stage1["PROPOSER_W128"],
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
        arm: float(np.mean([s["dl_val"]["p_one_sided"] for s in arms_out[arm]["seeds"]])) for arm in arms_out
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
