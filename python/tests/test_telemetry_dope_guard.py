"""DOPE freeze guard: NS3 digest compatibility and the fail-closed invariants (rows D1-D5)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from jope.telemetry import dope_guard as g

NS3 = Path("/home/ubuntu/jope_plan/ns/dope_freeze_guard.py")


class Fused(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.dope_model = nn.Sequential(nn.Linear(4, 3), nn.Sequential(nn.Dropout(0.1), nn.Linear(3, 2)))
        self.dope_model.register_buffer("stat", torch.arange(3, dtype=torch.float32))
        self.register_buffer("mu_lm48", torch.ones(2))
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dope_model(x))


def _load_ns3():
    spec = importlib.util.spec_from_file_location("ns3_dope_freeze_guard", NS3)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while the class body runs.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def frozen() -> tuple[Fused, torch.optim.Optimizer]:
    net = Fused()
    g.lock_dope(net)
    net.train(True)
    return net, torch.optim.AdamW(net.head.parameters(), lr=1e-2)


def test_golden_tensor_digest() -> None:
    # Reference value from NS3's `_tensor_le_sha256` on the same tensor.
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert g.tensor_le_sha256(t) == "e1f0996adf3c2088d4e9d60282254a15f7d717bf512b6dd52568656fb3e403d9"
    assert g.tensor_le_sha256(torch.zeros(2, dtype=torch.bfloat16)) != g.tensor_le_sha256(
        torch.zeros(2, dtype=torch.int16)
    )


# os.path.exists, not Path.exists: CI containers cannot traverse /home/ubuntu, and there
# Python 3.12's Path.exists raises PermissionError instead of returning False.
@pytest.mark.skipif(not os.path.exists(NS3), reason="NS3 reference guard is outside the product tree")
def test_digests_match_ns3_reference() -> None:
    ns3 = _load_ns3()
    net, _ = frozen()
    ours = g.snapshot(net)
    theirs = ns3.snapshot_dope_weights(net)
    assert ours.composite_sha256 == theirs.composite_sha256
    assert ours.params_sha256 == theirs.params_sha256
    assert ours.buffers_sha256 == theirs.buffers_sha256
    assert ours.tensor_hashes == theirs.tensor_hashes


def test_clean_head_only_epoch_passes_and_records() -> None:
    net, opt = frozen()
    records = []
    guard = g.DopeFreezeGuard(net, opt, emit=lambda kind, rec: records.append((kind, rec)), sample_every=1)
    start = guard.run_start()
    assert start["hashes_equal"] is None and start["trainable_jope_params"] == 3
    guard.epoch_begin(0)
    for step in range(3):
        loss = net(torch.randn(5, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        guard.step(step)
        opt.step()
    end = guard.epoch_end(0)
    assert end["hashes_equal"] is True and end["violations"] == []
    assert end["params_reconciled"] and end["dope_params_in_optimizer"] == 0
    assert all(kind == g.RECORD_KIND for kind, _ in records) and len(records) == 3


def test_nested_dropout_in_train_mode_fails() -> None:
    # NS3 spot-check counterexample: a nested submodule flipped while the root stays in eval.
    net, opt = frozen()
    guard = g.DopeFreezeGuard(net, opt)
    guard.run_start()
    net.dope_model[1].train(True)
    assert not net.dope_model.training
    with pytest.raises(g.DopeFreezeViolationError, match="train mode"):
        guard.epoch_end(0)


def test_root_parameters_are_counted_and_reconciled() -> None:
    # NS3 spot-check counterexample: a parameter directly on the root was missing from the
    # per-child counts.
    net, opt = frozen()
    net.scale = nn.Parameter(torch.ones(1))
    counts = g.param_counts(net)
    assert counts["by_submodule"][g.ROOT_GROUP]["trainable"] == 1
    assert counts["total_trainable"] == 4 and counts["reconciled"]


def test_dope_param_in_a_later_optimizer_group_fails() -> None:
    net, opt = frozen()
    opt.add_param_group({"params": [net.dope_model[0].weight], "lr": 0.0})
    with pytest.raises(g.DopeFreezeViolationError, match="optimizer groups"):
        g.DopeFreezeGuard(net, opt).run_start()


def test_weight_drift_names_the_changed_tensor() -> None:
    net, opt = frozen()
    records = []
    guard = g.DopeFreezeGuard(net, opt, emit=lambda k, r: records.append(r))
    guard.run_start()
    with torch.no_grad():
        net.dope_model[0].bias.add_(1e-7)
    with pytest.raises(g.DopeFreezeViolationError, match="dope_model.0.bias"):
        guard.epoch_end(0)
    assert records[-1]["changed_tensors"] == ["dope_model.0.bias"]


def test_requires_grad_and_leaked_grad_fail() -> None:
    net, opt = frozen()
    guard = g.DopeFreezeGuard(net, opt, sample_every=1)
    guard.run_start()
    net.dope_model[0].weight.grad = torch.zeros_like(net.dope_model[0].weight)
    with pytest.raises(g.DopeFreezeViolationError, match="hold a .grad"):
        guard.step(0)
    net.dope_model[0].weight.grad = None
    net.dope_model[0].weight.requires_grad_(True)
    with pytest.raises(g.DopeFreezeViolationError, match="requires_grad=True"):
        guard.epoch_end(0)


def test_no_trainable_jope_params_fails() -> None:
    net, _ = frozen()
    for p in net.head.parameters():
        p.requires_grad_(False)
    with pytest.raises(g.DopeFreezeViolationError, match="no trainable JopeDime"):
        g.DopeFreezeGuard(net, None).run_start()
