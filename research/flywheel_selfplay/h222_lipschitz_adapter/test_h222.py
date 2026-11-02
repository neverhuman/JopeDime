"""Data-free controls for actual regularizers, leakage boundaries and strict gates."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend

HERE = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(
        "h222_test_" + name, HERE / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


sci = module("science")
reg = module("regularize")
launcher = module("launcher")


def test_fresh_complete_family_folds():
    families = [f"f{i:03}" for i in range(205)]
    partitions = [sci.folds(families, s + 40) for s in sci.SEEDS]
    assert len({str(p) for p in partitions}) == 3
    for partition in partitions:
        assert sorted(x for fold in partition for x in fold) == families
        assert all(len(fold) == 41 for fold in partition)
        for held in partition:
            assert not set(held) & (set(families) - set(held))
    with pytest.raises(ValueError):
        sci.folds(families, 217141)


def test_spectral_all_trainable_linear_and_attention_once_per_update():
    model = nn.Module()
    model.anchor = nn.Linear(3, 4)
    model.anchor.requires_grad_(False)
    model.hidden = nn.Linear(3, 4)
    model.attn = nn.MultiheadAttention(4, 2, batch_first=True)
    anchor = model.anchor.weight.detach().clone()
    targets = reg.spectralize(model)
    assert {(n, f) for n, f, _, _ in targets} == {
        ("hidden", "weight"),
        ("attn", "in_proj_weight"),
        ("attn.out_proj", "weight"),
    }
    for _ in range(3):
        model.hidden(torch.ones(2, 3))
    assert all(n.steps == 0 for _, _, _, n in targets)
    for _ in range(2):
        reg.advance(targets)
    assert all(n.steps == 2 for _, _, _, n in targets)
    assert torch.equal(model.anchor.weight, anchor)
    objective = model.hidden(torch.randn(2, 3)).square().mean()
    objective.backward()
    assert model.hidden.parametrizations.weight.original.grad is not None


def test_actual_jacobian_penalty_supports_attention_double_backward():
    torch.manual_seed(7)
    attn = nn.MultiheadAttention(4, 2, batch_first=True)
    linear = nn.Linear(4, 8)
    tokens = torch.randn(2, 10, 4, requires_grad=True)
    mask = torch.ones(2, 10, dtype=torch.bool)
    with sdpa_kernel(SDPBackend.MATH):
        out = attn(tokens, tokens, tokens, need_weights=False)[0].mean(1)
        prediction = linear(out)
        penalty = reg.jacobian_penalty(prediction, tokens, mask, 222001)
        assert penalty > 0 and torch.isfinite(penalty)
        penalty.backward()
    assert linear.weight.grad is not None and linear.weight.grad.square().sum() > 0


def test_train_only_lambda_tie_selects_smaller():
    families = ["a", "a", "b"]
    lam, value = sci.choose_lambda({0.001: np.ones(3), 0.01: np.ones(3)}, families)
    assert lam == 0.001 and value[lam] == 1
    with pytest.raises(ValueError):
        sci.choose_lambda({0.001: np.ones(3)}, families)


def test_strict_complete_readout_rejects_ties_and_missing_seed():
    families = [f"f{i % 19}" for i in range(91)]
    base = np.ones((3, 91))
    panels = {a: base.copy() for a in ["R882", *sci.ARMS]}
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["SN"] += 0.1
    assert sci.readout(panels, families)["admitted"] == ["SN"]
    panels["SN"][0] -= 0.2
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["SN"] = panels["SN"][:2]
    with pytest.raises(ValueError):
        sci.readout(panels, families)


@pytest.mark.parametrize(
    "key,value",
    [
        ("load5", 61),
        ("free_disk_bytes", 299999999999),
        ("process_tree_rss_bytes", 4 * 2**30),
        ("host_cpu_percent", 86),
    ],
)
def test_resource_boundaries_refuse(key, value):
    observation = dict(
        load5=59,
        free_disk_bytes=301000000000,
        process_tree_rss_bytes=1000000,
        available_ram_bytes=20 * 2**30,
        host_cpu_percent=50,
        affinity=[1, 2, 3, 4],
        nice=19,
    )
    launcher.guard(observation, 1, 2)
    observation[key] = value
    with pytest.raises(RuntimeError):
        launcher.guard(observation, 1, 2)


def test_fit_rejects_validation_before_model_or_data_read():
    backend = module("backend")
    engine = backend.Engine.__new__(backend.Engine)
    engine.sci = sci
    called = []
    engine.model = lambda arm: called.append(arm)
    with pytest.raises(RuntimeError, match="TRAIN-only"):
        engine.fit(
            [dict(context_sha256="x", dl_split="DL-Val")],
            "CTRL",
            0.0,
            222101,
            "forbidden",
        )
    assert not called
