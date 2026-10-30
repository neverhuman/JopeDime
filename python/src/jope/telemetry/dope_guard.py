"""Torch side of the DOPE freeze guard for fused JopeDime + frozen DOPE training (rows D1-D5).

Only the parameter walk is here: the host byte copies, `requires_grad`, optimizer membership
and `training` flags. Hashing (NS3's digest), comparison, the records and the fail-closed
decision are in Rust (`jope._telemetry.DopeGuard`), which raises `DopeFreezeViolationError`
after recording every broken invariant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from jope._telemetry import DOPE_NORMALIZER_BUFFERS, DOPE_SUBMODULE, DopeFreezeViolationError, DopeGuard


def _le_bytes(t: torch.Tensor) -> tuple[str, list[int], Any]:
    """Host copy in little-endian C order, tagged as NS3 does (bfloat16 via its int16 bits)."""
    t = t.detach().cpu().contiguous()
    arr, tag = (t.view(torch.int16).numpy(), "bfloat16") if t.dtype == torch.bfloat16 else (t.numpy(), None)
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return tag or arr.dtype.str, list(arr.shape), arr.reshape(-1).view("u1")


def _dope(fused_net: nn.Module) -> nn.Module:
    dope = getattr(fused_net, DOPE_SUBMODULE, None)
    if not isinstance(dope, nn.Module):
        raise DopeFreezeViolationError(f"fused model has no `{DOPE_SUBMODULE}` submodule")
    return dope


class DopeFreezeGuard:
    """Call ``run_start()`` once the optimizer exists, ``epoch_begin(e)`` / ``epoch_end(e)``
    around every epoch, ``checkpoint(step)`` before saving, and ``step(step)`` after each
    ``backward()``. ``sink`` is a ``Recorder.record_sink()``; without one the guard still enforces.
    """

    def __init__(
        self, fused_net: nn.Module, optimizer: Any, sink: Any = None, sample_every: int = 100
    ) -> None:
        self.net, self.optimizer = fused_net, optimizer
        self._guard = DopeGuard(sink, sample_every)

    def _walk(self) -> dict[str, Any]:
        dope = _dope(self.net)
        net_buffers = dict(self.net.named_buffers())
        dope_ids = {id(p) for p in dope.parameters()}
        in_opt = (
            {id(p) for g in self.optimizer.param_groups for p in g["params"]} if self.optimizer else set()
        )
        params = list(self.net.parameters())
        return {
            "params": list(dope.named_parameters()),
            "buffers": list(dope.named_buffers()),
            "normalizers": [(n, net_buffers[n]) for n in DOPE_NORMALIZER_BUFFERS if n in net_buffers],
            "to_bytes": _le_bytes,
            "named": [(n, int(p.numel()), bool(p.requires_grad)) for n, p in self.net.named_parameters()],
            "totals": (
                sum(int(p.numel()) for p in params if p.requires_grad),
                sum(int(p.numel()) for p in params if not p.requires_grad),
            ),
            "in_optimizer": len(dope_ids & in_opt),
            "grads_present": sum(1 for p in dope.parameters() if p.grad is not None),
            "training": [name for name, m in dope.named_modules() if m.training],
        }

    def _probe(self) -> tuple[int, list[str]]:
        dope = _dope(self.net)
        return sum(1 for p in dope.parameters() if p.grad is not None), [
            n for n, m in dope.named_modules() if m.training
        ]

    def run_start(self) -> dict[str, Any]:
        return self._guard.observe("run_start", 0, False, reset=True, **self._walk())

    def epoch_begin(self, epoch: int) -> dict[str, Any]:
        return self._guard.observe("epoch_begin", epoch, True, **self._walk())

    def epoch_end(self, epoch: int) -> dict[str, Any]:
        return self._guard.observe("epoch_end", epoch, True, **self._walk())

    def checkpoint(self, step: int) -> dict[str, Any]:
        return self._guard.observe("checkpoint", step, True, **self._walk())

    def step(self, step: int) -> None:
        self._guard.step(step, self._probe)


def lock_dope(fused_net: nn.Module) -> None:
    """Freeze DOPE: ``requires_grad=False`` and eval mode on every DOPE submodule, kept in eval
    when ``fused_net.train(True)`` is called later. The guard verifies; this is the setup."""
    dope = _dope(fused_net)
    for p in dope.parameters():
        p.requires_grad_(False)
        p.grad = None
    dope.eval()
    original_train = fused_net.train

    def train(mode: bool = True) -> nn.Module:
        original_train(mode)
        dope.eval()
        return fused_net

    fused_net.train = train  # type: ignore[method-assign]
