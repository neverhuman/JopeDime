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
import os
import re
import types

import torch
import torch.nn as nn

if __package__:
    from .dope_freeze_guard_types import (
        DOPE_NORMALIZER_BUFFERS,
        DopeFreezeSnapshot,
        DopeFreezeViolationError,
        _assert_finite_tensor,
        _storage_ptr,
        _tensor_le_sha256,
    )
else:  # Support direct execution from this package directory.
    from dope_freeze_guard_types import (
        DOPE_NORMALIZER_BUFFERS,
        DopeFreezeSnapshot,
        DopeFreezeViolationError,
        _assert_finite_tensor,
        _storage_ptr,
        _tensor_le_sha256,
    )


def reject_research_dope_unfreeze(entrypoint: str) -> None:
    """Serving/flywheel entrypoints must never consume the research environment."""
    flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE", "0")
    if flag != "0":
        raise DopeFreezeViolationError(
            f"{entrypoint} refuses JOPE_RESEARCH_DOPE_UNFREEZE={flag!r}; use a dedicated research runner."
        )


def configure_research_dope(
    fused_net: nn.Module,
    *,
    launch_metadata: dict,
    research_run_id: str | None = None,
    unfreeze: bool | None = None,
) -> None:
    """Explicit research-only optimizer opt-in; record the policy before training.

    Call on a separate research model after the native frozen-baseline gate.
    Keep DOPE in eval mode and its normalizer buffers/anchor frozen. This does
    not relax snapshot_dope_weights or assert_optimizer_excludes_dope, which
    remain strict serving/flywheel checks. Persist launch_metadata as launch.json.
    """
    flag = os.environ.get("JOPE_RESEARCH_DOPE_UNFREEZE", "0")
    if flag not in ("0", "1") or (unfreeze is not None and type(unfreeze) is not bool):
        raise DopeFreezeViolationError("research unfreeze requires a boolean or environment 0/1")
    if unfreeze is False and flag == "1":
        raise DopeFreezeViolationError("explicit frozen policy conflicts with research environment")
    enabled = flag == "1" if unfreeze is None else unfreeze
    run_id = research_run_id or os.environ.get("JOPE_RESEARCH_RUN_ID")
    if enabled and (
        not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id)
    ):
        raise DopeFreezeViolationError("DOPE unfreeze requires a valid research run id")
    if not isinstance(launch_metadata, dict):
        raise DopeFreezeViolationError("research policy requires launch metadata")
    dope = getattr(fused_net, "dope_model", None)
    anchor = getattr(fused_net, "gamenet_anchor", None)
    if enabled and isinstance(dope, nn.Module) and isinstance(anchor, nn.Module):
        dope_ids = {id(p) for p in dope.parameters()}
        anchor_ids = {id(p) for p in anchor.parameters()}
        dope_ptrs = {ptr for p in dope.parameters() if (ptr := _storage_ptr(p)) is not None}
        anchor_ptrs = {ptr for p in anchor.parameters() if (ptr := _storage_ptr(p)) is not None}
        if (dope_ids & anchor_ids) or (dope_ptrs & anchor_ptrs):
            raise DopeFreezeViolationError("research DOPE and frozen anchor share parameters")
    _lock_dope_weights_and_eval_mode(fused_net)
    launch_metadata["dope_policy"] = {
        "research_dope_unfreeze": enabled,
        "research_run_id": run_id if enabled else None,
        "entrypoint": "research",
        "dope_eval_mode": True,
        "serving_eligible": False,
    }
    if enabled:
        fused_net._research_dope_run_id = run_id
        for p in fused_net.dope_model.parameters():
            p.requires_grad_(True)


def assert_research_optimizer_policy(
    optimizer: torch.optim.Optimizer, fused_net: nn.Module, *, research_run_id: str
) -> bool:
    """Permit DOPE parameters only on the explicitly configured research model."""
    configured = getattr(fused_net, "_research_dope_run_id", None)
    if configured is None:
        return assert_optimizer_excludes_dope(optimizer, fused_net)
    if research_run_id != configured:
        raise DopeFreezeViolationError("optimizer research run id differs from model policy")
    if any(sub.training for sub in fused_net.dope_model.modules()):
        raise DopeFreezeViolationError("research DOPE must remain in eval mode")
    net_buffers = dict(fused_net.named_buffers())
    protected_buffers = list(fused_net.dope_model.buffers()) + [
        net_buffers[name] for name in DOPE_NORMALIZER_BUFFERS if name in net_buffers
    ]
    buf_ids = {id(b) for b in protected_buffers}
    buf_ptrs = {ptr for b in protected_buffers if (ptr := _storage_ptr(b)) is not None}
    if any(
        id(p) in buf_ids or ((ptr := _storage_ptr(p)) is not None and ptr in buf_ptrs)
        for g in optimizer.param_groups
        for p in g.get("params", [])
    ):
        raise DopeFreezeViolationError("research optimizer includes frozen DOPE normalizer buffer")
    anchor = getattr(fused_net, "gamenet_anchor", None)
    if isinstance(anchor, nn.Module):
        anchor_tensors = list(anchor.parameters()) + list(anchor.buffers())
        anchor_ids = {id(t) for t in anchor_tensors}
        anchor_ptrs = {ptr for t in anchor_tensors if (ptr := _storage_ptr(t)) is not None}
        if any(
            id(p) in anchor_ids or ((ptr := _storage_ptr(p)) is not None and ptr in anchor_ptrs)
            for g in optimizer.param_groups
            for p in g.get("params", [])
        ):
            raise DopeFreezeViolationError("research optimizer includes the frozen anchor")
    return True


def lock_dope_weights_and_eval_mode(fused_net: nn.Module) -> None:
    """Strict serving/flywheel freeze; refuse the research opt-in environment."""
    reject_research_dope_unfreeze("serving/flywheel freeze guard")
    _lock_dope_weights_and_eval_mode(fused_net)


def _lock_dope_weights_and_eval_mode(fused_net: nn.Module) -> None:
    """
    Freezes `fused_net.dope_model` parameters + normalizer buffers and installs a
    recursive `.train(mode)` hook on `dope_model` AND every nested submodule in
    `dope_model.modules()` so neither `fused_net.train(True)` nor a direct nested call
    like `dope_model[1].train(True)` can ever flip any DOPE submodule's `.training`
    flag to `True`.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(fused_net.dope_model, nn.Module):
        raise DopeFreezeViolationError("fused_net does not contain a valid `dope_model` submodule.")

    fused_net._research_dope_run_id = None
    dope = fused_net.dope_model
    dope.eval()
    for p in dope.parameters():
        p.requires_grad_(False)
        p.grad = None
    for b in dope.buffers():
        b.requires_grad_(False)
        b.grad = None
    net_buffers = dict(fused_net.named_buffers())
    for buf_name in DOPE_NORMALIZER_BUFFERS:
        if buf_name in net_buffers:
            b = net_buffers[buf_name]
            b.requires_grad_(False)
            b.grad = None

    def _locked_eval_train(mod_self: nn.Module, mode: bool = True) -> nn.Module:
        for m in mod_self.modules():
            m.training = False
        return mod_self

    for sub in dope.modules():
        sub.training = False
        sub.train = types.MethodType(_locked_eval_train, sub)

    if hasattr(fused_net, "gamenet_anchor") and isinstance(fused_net.gamenet_anchor, nn.Module):
        for p in fused_net.gamenet_anchor.parameters():
            p.requires_grad_(False)
            p.grad = None
        for b in fused_net.gamenet_anchor.buffers():
            b.requires_grad_(False)
            b.grad = None
        for sub in fused_net.gamenet_anchor.modules():
            sub.training = False
            sub.train = types.MethodType(_locked_eval_train, sub)

    orig_fused_train = fused_net.train

    def _safe_fused_train(net_self: nn.Module, mode: bool = True) -> nn.Module:
        orig_fused_train(mode)
        for sub in net_self.dope_model.modules():
            sub.training = False
        research = getattr(net_self, "_research_dope_run_id", None) is not None
        for p in net_self.dope_model.parameters():
            p.requires_grad_(research)
            if not research:
                p.grad = None
        for b in net_self.dope_model.buffers():
            b.requires_grad_(False)
            b.grad = None
        cur_buffers = dict(net_self.named_buffers())
        for buf_name in DOPE_NORMALIZER_BUFFERS:
            if buf_name in cur_buffers:
                b = cur_buffers[buf_name]
                b.requires_grad_(False)
                b.grad = None
        if hasattr(net_self, "gamenet_anchor") and isinstance(net_self.gamenet_anchor, nn.Module):
            for sub in net_self.gamenet_anchor.modules():
                sub.training = False
            for p in net_self.gamenet_anchor.parameters():
                p.requires_grad_(False)
                p.grad = None
            for b in net_self.gamenet_anchor.buffers():
                b.requires_grad_(False)
                b.grad = None
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
    if not hasattr(fused_net, "dope_model") or not isinstance(fused_net.dope_model, nn.Module):
        raise DopeFreezeViolationError("Missing `dope_model` on `fused_net`.")

    dope: nn.Module = fused_net.dope_model
    for sub_name, sub_mod in dope.named_modules():
        if sub_mod.training:
            label = f"dope_model.{sub_name}" if sub_name else "dope_model"
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: `{label}.training` is True (all DOPE submodules must stay False / eval)."
            )

    if hasattr(fused_net, "gamenet_anchor") and isinstance(fused_net.gamenet_anchor, nn.Module):
        for sub_name, sub_mod in fused_net.gamenet_anchor.named_modules():
            if sub_mod.training:
                label = f"gamenet_anchor.{sub_name}" if sub_name else "gamenet_anchor"
                raise DopeFreezeViolationError(
                    f"DOPE freeze violation: `{label}.training` is True (frozen anchor must stay False / eval)."
                )
        for name, p in sorted(fused_net.gamenet_anchor.named_parameters(), key=lambda kv: kv[0]):
            if p.requires_grad:
                raise DopeFreezeViolationError(
                    f"DOPE freeze violation: parameter `gamenet_anchor.{name}` has requires_grad=True."
                )
            if p.grad is not None:
                raise DopeFreezeViolationError(
                    f"DOPE freeze violation: parameter `gamenet_anchor.{name}` has non-None `.grad`."
                )
            _assert_finite_tensor(p, f"gamenet_anchor.{name}")

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
        _assert_finite_tensor(p, full_name)
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
        if b.requires_grad:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: buffer `{full_name}` has requires_grad=True."
            )
        if b.grad is not None:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: buffer `{full_name}` has non-None `.grad`."
            )
        _assert_finite_tensor(b, full_name)
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
        if b.requires_grad:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: normalizer buffer `{full_name}` has requires_grad=True."
            )
        if b.grad is not None:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: normalizer buffer `{full_name}` has non-None `.grad`."
            )
        _assert_finite_tensor(b, full_name)
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


def get_trainable_param_counts_by_submodule(fused_net: nn.Module) -> dict[str, dict[str, int]]:
    """
    Per Jepson's 08:25 EDT HARD REQUIREMENT in NORTH_STAR.md:
    Logs trainable, frozen, and total parameter counts by top-level submodule,
    including direct root parameters (`__root_direct__`) and unique global parameter
    reconciliation (`__global_total__`), and asserts that `dope_model` has 0 trainable
    parameters and 0 submodules in training mode.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(fused_net.dope_model, nn.Module):
        raise DopeFreezeViolationError("Missing `dope_model` on `fused_net`.")

    for sub_name, sub_mod in fused_net.dope_model.named_modules():
        if sub_mod.training:
            label = f"dope_model.{sub_name}" if sub_name else "dope_model"
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: `{label}.training` is True inside get_trainable_param_counts_by_submodule."
            )

    if hasattr(fused_net, "gamenet_anchor") and isinstance(fused_net.gamenet_anchor, nn.Module):
        for sub_name, sub_mod in fused_net.gamenet_anchor.named_modules():
            if sub_mod.training:
                label = f"gamenet_anchor.{sub_name}" if sub_name else "gamenet_anchor"
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
        frozen = sum(int(p.numel()) for p in sub_mod.parameters() if not p.requires_grad)
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
    global_frozen = sum(int(p.numel()) for p in all_unique_params if not p.requires_grad)
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
    if "gamenet_anchor" in counts:
        if counts["gamenet_anchor"]["trainable_params"] != 0:
            raise DopeFreezeViolationError(
                f"DOPE freeze violation: gamenet_anchor has non-zero trainable_params={counts['gamenet_anchor']}"
            )
        if counts["gamenet_anchor"]["training_mode"]:
            raise DopeFreezeViolationError(
                "DOPE freeze violation: a submodule inside `gamenet_anchor` has training_mode=True."
            )
    return counts


if __package__:
    from .dope_freeze_guard_optimizer import assert_optimizer_excludes_dope
else:  # Support direct execution from this package directory.
    from dope_freeze_guard_optimizer import assert_optimizer_excludes_dope


if __package__:
    from .dope_freeze_guard_self_test import run_self_test
else:  # Support direct execution from this package directory.
    from dope_freeze_guard_self_test import run_self_test


if __name__ == "__main__":
    run_self_test()
