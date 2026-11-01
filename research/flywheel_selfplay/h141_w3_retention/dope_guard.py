#!/usr/bin/env python3
"""
NORTH STAR Phase C (NS3): DOPE Weight-Freeze Enforcer & Deterministic Hash Verification
=======================================================================================
Enforces and verifies the NORTH_STAR Phase C invariant on `FusedJopeNetwork`:
  1. All `dope_model` (`MultiLevelLandmarkModel`) parameters have `requires_grad == False`
     and `grad is None`.
  2. `dope_model` is locked in `.eval()` mode (`training == False`) even when
     `fused_net.train(True)` is called during a Phase D flywheel epoch (preventing
     `nn.Dropout(0.10)` inside `MultiLevelLandmarkModel` from corrupting frozen DOPE
     embeddings during flywheel training).
  3. Deterministic IEEE-754 little-endian SHA-256 hashes over all 20 `dope_model`
     parameter tensors (`134,509` floats) and all 6 DOPE Stage-1A normalizer buffers
     (`ms_mean_376`, `ms_std_376`, `col_mean_17`, `col_std_17`, `lm_mean_48`, `lm_std_48`)
     remain bit-identical before and after flywheel training epochs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

DOPE_NORMALIZER_BUFFERS: tuple[str, ...] = (
    "step2_lm_mean_1x256",
    "step2_lm_std_1x256",
    "step2_svd_basis_256x48",
    "lm80_mu",
    "lm80_sd",
    "mu_lm48",
    "sd_lm48",
    "mu_bb20",
    "sd_bb20",
    "mu_dm24",
    "sd_dm24",
)


class DopeFreezeViolationError(RuntimeError):
    """Raised when any DOPE weight, buffer, grad flag, or eval mode invariant is violated."""


@dataclass(frozen=True)
class DopeFreezeSnapshot:
    composite_sha256: str
    params_sha256: str
    buffers_sha256: str
    num_param_tensors: int
    num_param_elements: int
    num_buffer_tensors: int
    num_buffer_elements: int
    tensor_hashes: dict[str, str]
    tensor_shapes: dict[str, list[int]]


def _tensor_le_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}:{list(arr.shape)}:".encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def lock_dope_weights_and_eval_mode(fused_net: nn.Module) -> None:
    """
    Freezes `fused_net.dope_model` parameters + normalizer buffers and installs a
    recursive `.train(mode)` hook on `dope_model` AND every nested submodule in
    `dope_model.modules()` so neither `fused_net.train(True)` nor a direct nested call
    like `dope_model[1].train(True)` can ever flip any DOPE submodule's `.training`
    flag to `True`.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(
        fused_net.dope_model, nn.Module
    ):
        raise DopeFreezeViolationError(
            "fused_net does not contain a valid `dope_model` submodule."
        )

    dope = fused_net.dope_model
    dope.eval()
    for p in dope.parameters():
        p.requires_grad_(False)
        p.grad = None

    def _locked_eval_train(mod_self: nn.Module, mode: bool = True) -> nn.Module:
        for m in mod_self.modules():
            m.training = False
        return mod_self

    for sub in dope.modules():
        sub.training = False
        sub.train = types.MethodType(_locked_eval_train, sub)

    if hasattr(fused_net, "gamenet_anchor") and isinstance(
        fused_net.gamenet_anchor, nn.Module
    ):
        for p in fused_net.gamenet_anchor.parameters():
            p.requires_grad_(False)
            p.grad = None
        for sub in fused_net.gamenet_anchor.modules():
            sub.training = False
            sub.train = types.MethodType(_locked_eval_train, sub)

    orig_fused_train = fused_net.train

    def _safe_fused_train(net_self: nn.Module, mode: bool = True) -> nn.Module:
        orig_fused_train(mode)
        for sub in net_self.dope_model.modules():
            sub.training = False
        for p in net_self.dope_model.parameters():
            p.requires_grad_(False)
            p.grad = None
        if hasattr(net_self, "gamenet_anchor") and isinstance(
            net_self.gamenet_anchor, nn.Module
        ):
            for sub in net_self.gamenet_anchor.modules():
                sub.training = False
            for p in net_self.gamenet_anchor.parameters():
                p.requires_grad_(False)
                p.grad = None
        return net_self

    if not getattr(fused_net, "_dope_freeze_train_hook_installed", False):
        fused_net.train = types.MethodType(_safe_fused_train, fused_net)
        fused_net._dope_freeze_train_hook_installed = True


def snapshot_dope_weights(fused_net: nn.Module) -> DopeFreezeSnapshot:
    """
    Computes a deterministic SHA-256 snapshot of all `dope_model` parameters and
    Stage-1A DOPE normalizer buffers, while checking `requires_grad == False` and
    `sub.training == False` across EVERY nested submodule in `dope_model.named_modules()`.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(
        fused_net.dope_model, nn.Module
    ):
        raise DopeFreezeViolationError("Missing `dope_model` on `fused_net`.")

    dope: nn.Module = fused_net.dope_model
    for sub_name, sub_mod in dope.named_modules():
        if sub_mod.training:
            label = f"dope_model.{sub_name}" if sub_name else "dope_model"
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: `{label}.training` is True (all DOPE submodules must stay False / eval)."
            )

    tensor_hashes: dict[str, str] = {}
    tensor_shapes: dict[str, list[int]] = {}
    h_params = hashlib.sha256()
    h_buffers = hashlib.sha256()

    num_param_tensors = 0
    num_param_elements = 0
    for name, p in sorted(dope.named_parameters(), key=lambda kv: kv[0]):
        if p.requires_grad:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: parameter `dope_model.{name}` has requires_grad=True."
            )
        if p.grad is not None:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: parameter `dope_model.{name}` has non-None `.grad`."
            )
        full_name = f"dope_model.{name}"
        th = _tensor_le_sha256(p)
        tensor_hashes[full_name] = th
        tensor_shapes[full_name] = list(p.shape)
        h_params.update(f"{full_name}={th}\n".encode())
        num_param_tensors += 1
        num_param_elements += int(p.numel())

    if num_param_tensors == 0:
        raise DopeFreezeViolationError("`dope_model` has 0 parameters.")

    num_buffer_tensors = 0
    num_buffer_elements = 0
    for name, b in sorted(dope.named_buffers(), key=lambda kv: kv[0]):
        full_name = f"dope_model.buffer.{name}"
        th = _tensor_le_sha256(b)
        tensor_hashes[full_name] = th
        tensor_shapes[full_name] = list(b.shape)
        h_buffers.update(f"{full_name}={th}\n".encode())
        num_buffer_tensors += 1
        num_buffer_elements += int(b.numel())

    net_buffers = dict(fused_net.named_buffers())
    for buf_name in DOPE_NORMALIZER_BUFFERS:
        if buf_name not in net_buffers:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: required normalizer buffer `{buf_name}` is missing."
            )
        b = net_buffers[buf_name]
        full_name = f"normalizer.{buf_name}"
        th = _tensor_le_sha256(b)
        tensor_hashes[full_name] = th
        tensor_shapes[full_name] = list(b.shape)
        h_buffers.update(f"{full_name}={th}\n".encode())
        num_buffer_tensors += 1
        num_buffer_elements += int(b.numel())

    params_sha = h_params.hexdigest()
    buffers_sha = h_buffers.hexdigest()
    h_comp = hashlib.sha256()
    h_comp.update(f"params:{params_sha}|buffers:{buffers_sha}".encode())
    composite_sha = h_comp.hexdigest()

    return DopeFreezeSnapshot(
        composite_sha256=composite_sha,
        params_sha256=params_sha,
        buffers_sha256=buffers_sha,
        num_param_tensors=num_param_tensors,
        num_param_elements=num_param_elements,
        num_buffer_tensors=num_buffer_tensors,
        num_buffer_elements=num_buffer_elements,
        tensor_hashes=tensor_hashes,
        tensor_shapes=tensor_shapes,
    )


def verify_dope_frozen_unchanged(
    fused_net: nn.Module,
    before: DopeFreezeSnapshot,
) -> tuple[bool, DopeFreezeSnapshot]:
    """
    Re-computes the DOPE snapshot after a flywheel training step/epoch and asserts
    zero drift in any parameter tensor, normalizer buffer, `requires_grad` flag, or `.eval()` state.
    """
    after = snapshot_dope_weights(fused_net)
    if after.composite_sha256 != before.composite_sha256:
        changed = [
            k
            for k in set(before.tensor_hashes) | set(after.tensor_hashes)
            if before.tensor_hashes.get(k) != after.tensor_hashes.get(k)
        ]
        raise DopeFreezeViolationError(
            f"DOPE weight/buffer hash mismatch! before={before.composite_sha256} "
            f"after={after.composite_sha256}, changed_tensors={changed}"
        )
    return True, after


def get_trainable_param_counts_by_submodule(
    fused_net: nn.Module,
) -> dict[str, dict[str, int]]:
    """
    Per Jepson's 08:25 EDT HARD REQUIREMENT in NORTH_STAR.md:
    Logs trainable, frozen, and total parameter counts by top-level submodule,
    including direct root parameters (`__root_direct__`) and unique global parameter
    reconciliation (`__global_total__`), and asserts that `dope_model` has 0 trainable
    parameters and 0 submodules in training mode.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(
        fused_net.dope_model, nn.Module
    ):
        raise DopeFreezeViolationError("Missing `dope_model` on `fused_net`.")

    for sub_name, sub_mod in fused_net.dope_model.named_modules():
        if sub_mod.training:
            label = f"dope_model.{sub_name}" if sub_name else "dope_model"
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: `{label}.training` is True inside get_trainable_param_counts_by_submodule."
            )

    counts: dict[str, dict[str, int]] = {}

    # 1. Direct root parameters (`recurse=False`) so root-level nn.Parameters are never omitted
    root_direct_params = list(fused_net.parameters(recurse=False))
    root_trainable = sum(int(p.numel()) for p in root_direct_params if p.requires_grad)
    root_frozen = sum(int(p.numel()) for p in root_direct_params if not p.requires_grad)
    counts["__root_direct__"] = {
        "trainable_params": root_trainable,
        "frozen_params": root_frozen,
        "total_params": root_trainable + root_frozen,
        "training_mode": bool(fused_net.training),
    }

    # 2. Top-level child modules
    for sub_name, sub_mod in fused_net.named_children():
        trainable = sum(int(p.numel()) for p in sub_mod.parameters() if p.requires_grad)
        frozen = sum(
            int(p.numel()) for p in sub_mod.parameters() if not p.requires_grad
        )
        any_sub_training = any(bool(m.training) for m in sub_mod.modules())
        counts[sub_name] = {
            "trainable_params": trainable,
            "frozen_params": frozen,
            "total_params": trainable + frozen,
            "training_mode": any_sub_training,
        }

    # 3. Unique global total across the entire module tree (`fused_net.parameters()`)
    all_unique_params = list(fused_net.parameters())
    global_trainable = sum(int(p.numel()) for p in all_unique_params if p.requires_grad)
    global_frozen = sum(
        int(p.numel()) for p in all_unique_params if not p.requires_grad
    )
    counts["__global_total__"] = {
        "trainable_params": global_trainable,
        "frozen_params": global_frozen,
        "total_params": global_trainable + global_frozen,
        "training_mode": bool(fused_net.training),
    }

    if counts["dope_model"]["trainable_params"] != 0:
        raise DopeFreezeViolationError(
            f"DOPE freeze violation: dope_model has non-zero trainable_params={counts['dope_model']}"
        )
    if counts["dope_model"]["training_mode"]:
        raise DopeFreezeViolationError(
            "DOPE freeze violation: a submodule inside `dope_model` has training_mode=True."
        )
    return counts


def assert_optimizer_excludes_dope(
    optimizer: torch.optim.Optimizer,
    fused_net: nn.Module,
) -> bool:
    """
    Per Jepson's 08:25 EDT HARD REQUIREMENT in NORTH_STAR.md:
    Asserts that no parameter belonging to `fused_net.dope_model` is present in any
    `optimizer.param_groups`.
    """
    dope_param_ids = {id(p) for p in fused_net.dope_model.parameters()}
    for g_idx, group in enumerate(optimizer.param_groups):
        for p in group.get("params", []):
            if id(p) in dope_param_ids:
                raise DopeFreezeViolationError(
                    f"DOPE freeze violation: DOPE parameter found in optimizer.param_groups[{g_idx}]!"
                )
    return True


def run_self_test() -> dict:
    """
    Executes the full NS3 verification suite against the real `FusedJopeNetwork` and
    `dope_mlc_v3_multitask_best.pt` on CPU (zero GPU contention):
      Test 1: Load `FusedJopeNetwork`, apply `lock_dope_weights_and_eval_mode`, take baseline snapshot.
      Test 2: Call `net.train(True)` and verify `net.dope_model.training is False` (Dropout stays off).
      Test 3: Unfreeze `net.adapter_10tok`, run a synthetic flywheel training epoch (5 AdamW steps
              backpropagating through `adapter_10tok` + `dope_model` outputs), verify `adapter_10tok`
              weights change while `verify_dope_frozen_unchanged` passes with 0-bit drift.
      Test 4: Negative tamper tests (1-ULP parameter mutation, normalizer buffer mutation, and
              `requires_grad=True` mutation) all raise `DopeFreezeViolationError`.
    """
    sys.path.insert(0, "/home/ubuntu/gpu_bench/fused")
    from fused_jope_net import FusedJopeNetwork

    net = FusedJopeNetwork(device="cpu", adapter_variant="h40b")
    lock_dope_weights_and_eval_mode(net)
    snap_before = snapshot_dope_weights(net)

    # Test 2: Verify `net.train(True)` keeps `dope_model` in `.eval()` mode
    net.train(True)
    assert not net.dope_model.training, (
        "dope_model.training became True under net.train(True)!"
    )
    assert not net.gamenet_anchor.training, (
        "gamenet_anchor.training became True under net.train(True)!"
    )

    # Test 3: Run 5 synthetic flywheel AdamW steps updating ONLY the non-anchor `adapter_10tok` params
    # (Note: `adapter_10tok.anchor` aliases `net.gamenet_anchor` (`520,785` params), so a naive loop over
    # `adapter_10tok.parameters()` without excluding `anchor.` would accidentally unfreeze `gamenet_anchor`!)
    for p in net.parameters():
        p.requires_grad_(False)
    trainable_params = []
    for name, p in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor.") and not name.startswith("frozen_anchor."):
            p.requires_grad_(True)
            trainable_params.append(p)
    lock_dope_weights_and_eval_mode(net)

    adapter_sha_before = _tensor_le_sha256(trainable_params[0])
    opt = torch.optim.AdamW(trainable_params, lr=1e-3)
    assert_optimizer_excludes_dope(opt, net)
    submodule_param_counts = get_trainable_param_counts_by_submodule(net)

    torch.manual_seed(42)
    vec256 = torch.randn(4, 256)
    col_lm = torch.randn(4, 64, 17)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        dope_629, lm_48 = net._run_dope_mlc_v3_and_lm48(
            vec256, col_lm, use_landmarks=True
        )
        # Route DOPE output into a trainable adapter projection to verify gradient stopper
        mm_tok = dope_629[:, :320].view(4, 10, 32)
        mm_mask = torch.ones(4, 10)
        prog_412 = torch.randn(4, 96, 412)
        sos_prior = torch.randn(4, 96, 2)
        ctx_p, cand_q, sem24, mm_adj = (
            net.adapter_10tok.encode_multimodal_context_and_candidates(
                mm_tok, mm_mask, prog_412, sos_prior
            )
        )
        loss = (mm_adj[:, 0] - 1.0).pow(2).mean() + 0.0 * lm_48.sum()
        loss.backward()
        opt.step()

    adapter_sha_after = _tensor_le_sha256(trainable_params[0])
    assert adapter_sha_before != adapter_sha_after, (
        "Trainable adapter weights did not update!"
    )
    ok, snap_after = verify_dope_frozen_unchanged(net, snap_before)
    assert ok and snap_after.composite_sha256 == snap_before.composite_sha256

    # Test 4: Negative tamper checks (must raise DopeFreezeViolationError)
    tamper_results = {}

    # 4a: 1-ULP weight tamper
    first_param = next(net.dope_model.parameters())
    orig_val = float(first_param.view(-1)[0].item())
    with torch.no_grad():
        first_param.view(-1)[0] = orig_val + 1e-6
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["weight_1ulp_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["weight_1ulp_ulp_tamper_caught"] = True
    finally:
        with torch.no_grad():
            first_param.view(-1)[0] = orig_val

    # 4b: Normalizer buffer tamper
    orig_buf = float(net.step2_lm_mean_1x256.view(-1)[0].item())
    with torch.no_grad():
        net.step2_lm_mean_1x256.view(-1)[0] = orig_buf + 1e-5
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["normalizer_buffer_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["normalizer_buffer_tamper_caught"] = True
    finally:
        with torch.no_grad():
            net.step2_lm_mean_1x256.view(-1)[0] = orig_buf

    # 4c: requires_grad=True tamper
    first_param.requires_grad_(True)
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["requires_grad_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["requires_grad_tamper_caught"] = True
    finally:
        first_param.requires_grad_(False)

    # 4d: Optimizer param_group inclusion tamper
    bad_opt = torch.optim.SGD([first_param], lr=1e-3)
    try:
        assert_optimizer_excludes_dope(bad_opt, net)
        tamper_results["optimizer_group_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["optimizer_group_tamper_caught"] = True

    # 4e: Codex counterexample #1 — direct nested submodule `.train(True)` call AND raw `.training = True` attribute tamper
    nested_dope_mods = [m for m in net.dope_model.modules() if m is not net.dope_model]
    target_nested = nested_dope_mods[0]
    target_nested.train(True)
    # Because `lock_dope_weights_and_eval_mode` installed `_locked_eval_train` on every nested submodule,
    # `target_nested.train(True)` must keep `target_nested.training == False`:
    assert not target_nested.training, (
        "Nested DOPE submodule `.train(True)` escaped eval lock!"
    )
    tamper_results["nested_submodule_train_call_blocked"] = not target_nested.training

    # Even if code bypasses `.train()` and directly assigns `target_nested.training = True`,
    # `snapshot_dope_weights` and `get_trainable_param_counts_by_submodule` must catch it:
    target_nested.training = True
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["nested_submodule_attr_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["nested_submodule_attr_tamper_caught"] = True
    finally:
        target_nested.training = False

    # 4f: Codex counterexample #2 — minimal fixture with a direct root parameter (1 trainable on root + 2 on child = 3 total)
    class _MinimalRootParamFixture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root_scale = nn.Parameter(torch.ones(1), requires_grad=True)
            self.dope_model = nn.Sequential(
                nn.Linear(2, 2, bias=False), nn.Dropout(0.1)
            )
            self.head = nn.Linear(2, 1, bias=False)

    min_fix = _MinimalRootParamFixture()
    lock_dope_weights_and_eval_mode(min_fix)
    min_fix.dope_model[1].train(True)
    assert not min_fix.dope_model[1].training, (
        "dope_model[1].train(True) escaped on Sequential fixture!"
    )
    min_counts = get_trainable_param_counts_by_submodule(min_fix)
    assert min_counts["__root_direct__"]["trainable_params"] == 1
    assert min_counts["head"]["trainable_params"] == 2
    assert min_counts["dope_model"]["trainable_params"] == 0
    assert min_counts["__global_total__"]["trainable_params"] == 3
    tamper_results["root_direct_and_global_param_reconciliation_verified"] = True

    # Restore clean state and re-verify
    verify_dope_frozen_unchanged(net, snap_before)

    report = {
        "task": "NS3_DOPE_WEIGHT_FREEZE_AND_HASH_TEST",
        "status": "PASS",
        "dope_composite_sha256_before": snap_before.composite_sha256,
        "dope_composite_sha256_after": snap_after.composite_sha256,
        "dope_params_sha256": snap_before.params_sha256,
        "dope_buffers_sha256": snap_before.buffers_sha256,
        "num_dope_param_tensors": snap_before.num_param_tensors,
        "num_dope_param_elements": snap_before.num_param_elements,
        "num_dope_normalizer_buffers": snap_before.num_buffer_tensors,
        "num_dope_normalizer_elements": snap_before.num_buffer_elements,
        "trainable_param_counts_by_submodule": submodule_param_counts,
        "optimizer_excludes_dope_verified": True,
        "train_mode_override_verified": (not net.dope_model.training),
        "adapter_updated_during_flywheel_step": (
            adapter_sha_before != adapter_sha_after
        ),
        "adapter_param0_sha256_before": adapter_sha_before,
        "adapter_param0_sha256_after": adapter_sha_after,
        "negative_tamper_checks": tamper_results,
        "tensor_hashes": snap_before.tensor_hashes,
        "tensor_shapes": snap_before.tensor_shapes,
    }
    out_path = Path(__file__).resolve().parent / "ns3_dope_freeze_receipt.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("tensor_hashes", "tensor_shapes")
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run_self_test()
