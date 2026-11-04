"""Optimizer membership checks for protected DOPE and anchor tensors."""

import torch
import torch.nn as nn

if __package__:
    from .dope_freeze_guard_types import DOPE_NORMALIZER_BUFFERS, DopeFreezeViolationError, _storage_ptr
else:  # Support direct execution from this package directory.
    from dope_freeze_guard_types import DOPE_NORMALIZER_BUFFERS, DopeFreezeViolationError, _storage_ptr


def assert_optimizer_excludes_dope(
    optimizer: torch.optim.Optimizer,
    fused_net: nn.Module,
) -> bool:
    """
    Per Jepson's 08:25 EDT HARD REQUIREMENT in NORTH_STAR.md:
    Asserts that no parameter, tensor view/alias, normalizer buffer, or frozen anchor
    tensor belonging to `fused_net.dope_model` / `fused_net.gamenet_anchor` is present
    in any `optimizer.param_groups`.
    """
    if not hasattr(fused_net, "dope_model") or not isinstance(fused_net.dope_model, nn.Module):
        raise DopeFreezeViolationError("Missing `dope_model` on `fused_net`.")
    if not hasattr(optimizer, "param_groups") or not isinstance(optimizer.param_groups, list):
        raise DopeFreezeViolationError("Invalid optimizer: missing `param_groups`.")

    net_buffers = dict(fused_net.named_buffers())
    protected_tensors = (
        list(fused_net.dope_model.parameters())
        + list(fused_net.dope_model.buffers())
        + [net_buffers[name] for name in DOPE_NORMALIZER_BUFFERS if name in net_buffers]
    )
    anchor = getattr(fused_net, "gamenet_anchor", None)
    if isinstance(anchor, nn.Module):
        protected_tensors.extend(anchor.parameters())
        protected_tensors.extend(anchor.buffers())

    protected_ids = {id(t) for t in protected_tensors}
    protected_ptrs = {ptr for t in protected_tensors if (ptr := _storage_ptr(t)) is not None}

    for g_idx, group in enumerate(optimizer.param_groups):
        for p in group.get("params", []):
            p_ptr = _storage_ptr(p) if isinstance(p, torch.Tensor) else None
            if id(p) in protected_ids or (p_ptr is not None and p_ptr in protected_ptrs):
                raise DopeFreezeViolationError(
                    f"DOPE freeze violation: frozen DOPE/anchor tensor or storage alias found in optimizer.param_groups[{g_idx}]!"
                )
    return True
