"""Unit tests for the NS2 auto-fuse harness (`tools/auto_fuse.py`, NS24).

CPU-only synthetic fixtures: tensor-state hashing (0-D tensors, byte identity), the DOPE freeze
summary and the released-checkpoint loader's validation. The GPU P2 evaluation is not exercised here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

TOOL = Path(__file__).resolve().parents[2] / "tools" / "auto_fuse.py"
# The tool points TMPDIR and the compiler caches at its lane target when imported; keep that out of
# the rest of the test session.
_TOOL_ENV = ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")


@pytest.fixture(scope="module")
def auto_fuse():
    saved = {key: os.environ.get(key) for key in _TOOL_ENV}
    spec = importlib.util.spec_from_file_location("auto_fuse_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def _kat_state() -> dict[str, torch.Tensor]:
    return {
        "w": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 4,
        "idx": torch.tensor([[1, -2], [3, 4]], dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "half": torch.tensor([0.5, -1.25], dtype=torch.bfloat16),
        "bytes": torch.tensor([0, 255, 7], dtype=torch.uint8),
    }


def _reference_sha(state: dict[str, torch.Tensor]) -> str:
    """The digest format written out independently of the tool's byte view."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        meta = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode()
        start = cpu.storage_offset() * cpu.element_size()
        raw = bytes(cpu.untyped_storage().tolist())[start : start + cpu.numel() * cpu.element_size()]
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


class _ToyFused(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dope_model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
        self.head = nn.Linear(2, 1)
        self.scale = nn.Parameter(torch.ones(4))


def test_tensor_state_sha_is_unchanged_for_tensors_main_already_hashed(auto_fuse) -> None:
    # Known answer from tools/auto_fuse.py before the 0-D fix (main e3dfb2c94).
    expected = "c90ac156ea142b572a9cbf7af7e7323cd10964b782396bbee6ce54655a2a8fde"
    assert auto_fuse.tensor_state_sha(_kat_state()) == expected
    assert _reference_sha(_kat_state()) == expected


def test_tensor_state_sha_hashes_zero_dim_non_byte_tensors(auto_fuse) -> None:
    state = {
        "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        "temperature": torch.tensor(0.25, dtype=torch.float64),
        "flag": torch.tensor(True),
    }
    digest = auto_fuse.tensor_state_sha(state)
    assert digest == _reference_sha(state)
    # The shape is part of the identity: a 0-D tensor and its one-element vector differ.
    widened = dict(state, num_batches_tracked=torch.tensor([7], dtype=torch.int64))
    assert auto_fuse.tensor_state_sha(widened) != digest


def test_tensor_state_sha_binds_names_dtypes_and_values_not_order_or_layout(auto_fuse) -> None:
    state = _kat_state()
    digest = auto_fuse.tensor_state_sha(state)
    assert auto_fuse.tensor_state_sha(dict(reversed(list(state.items())))) == digest
    transposed = dict(state, w=state["w"].t().contiguous().t())
    assert not transposed["w"].is_contiguous()
    assert auto_fuse.tensor_state_sha(transposed) == digest

    renamed = {("weight" if k == "w" else k): v for k, v in state.items()}
    same_bytes_other_dtype = dict(state, idx=state["idx"].view(torch.float64))
    changed_value = dict(state, bytes=torch.tensor([0, 255, 8], dtype=torch.uint8))
    for variant in (renamed, same_bytes_other_dtype, changed_value):
        assert auto_fuse.tensor_state_sha(variant) != digest


def test_tensor_state_sha_rejects_non_tensor_items(auto_fuse) -> None:
    with pytest.raises(TypeError, match="^non-tensor state item: step$"):
        auto_fuse.tensor_state_sha({"w": torch.zeros(2), "step": 3})


def test_freeze_dope_hashes_batchnorm_state_and_counts_trainables(auto_fuse) -> None:
    torch.manual_seed(0)
    model = _ToyFused()
    model.train()
    freeze = auto_fuse.freeze_dope(model)

    assert freeze["state_sha256"] == auto_fuse.tensor_state_sha(model.dope_model.state_dict())
    assert freeze["state_sha256"] == _reference_sha(model.dope_model.state_dict())
    assert freeze["parameter_tensors"] == 4
    assert freeze["parameter_elements"] == 3 * 2 + 2 + 2 + 2
    # running_mean, running_var and the 0-D num_batches_tracked
    assert freeze["buffer_tensors"] == 3
    assert freeze["buffer_elements"] == 2 + 2 + 1
    assert not any(m.training for m in model.dope_model.modules())
    assert auto_fuse.trainable_counts(model) == {
        "dope_model": 0,
        "head": 3,
        "<root-only>": 4,
        "<total>": 7,
    }


def test_dope_sha_requires_a_dope_submodule(auto_fuse) -> None:
    with pytest.raises(RuntimeError, match="^fused model is missing dope_model$"):
        auto_fuse.dope_sha(nn.Linear(2, 2))


def _checkpoint(tmp_path: Path, monkeypatch, auto_fuse, envelope, *, pin: bool = True) -> Path:
    path = tmp_path / "dope.pt"
    torch.save(envelope, path)
    if pin:
        monkeypatch.setattr(auto_fuse, "RELEASED_DOPE_SHA256", auto_fuse.file_sha(path))
    return path


def test_load_released_dope_refuses_an_unpinned_checkpoint(tmp_path, monkeypatch, auto_fuse) -> None:
    path = _checkpoint(
        tmp_path, monkeypatch, auto_fuse, {"model_state_dict": {"w": torch.ones(2)}}, pin=False
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"^refusing non-released DOPE checkpoint with SHA-256 {digest}$"):
        auto_fuse.load_released_dope(path)


def test_load_released_dope_returns_a_finite_pinned_state(tmp_path, monkeypatch, auto_fuse) -> None:
    state = dict(_kat_state(), num_batches_tracked=torch.tensor(3, dtype=torch.int64))
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, {"model_state_dict": state, "epoch": 4})
    loaded, digest = auto_fuse.load_released_dope(path)
    assert digest == auto_fuse.file_sha(path)
    assert auto_fuse.tensor_state_sha(loaded) == auto_fuse.tensor_state_sha(state)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.complex64])
def test_load_released_dope_refuses_non_finite_tensors(tmp_path, monkeypatch, auto_fuse, bad, dtype) -> None:
    poisoned = torch.tensor([1.0, bad, 2.0]).to(dtype)
    state = {"a": torch.ones(2), "z": poisoned, "m": torch.tensor(float("nan")), "idx": torch.arange(3)}
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, {"model_state_dict": state})
    with pytest.raises(ValueError, match="^released DOPE checkpoint has non-finite tensors: m, z$"):
        auto_fuse.load_released_dope(path)


@pytest.mark.parametrize(
    "envelope",
    [
        {"model_state_dict": {}},
        {"model_state_dict": {"w": torch.ones(2), "step": 3}},
        {"state_dict": {"w": torch.ones(2)}},
        [torch.ones(2)],
    ],
    ids=["empty", "non-tensor-item", "missing-key", "not-a-dict"],
)
def test_load_released_dope_refuses_malformed_envelopes(tmp_path, monkeypatch, auto_fuse, envelope) -> None:
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, envelope)
    with pytest.raises(ValueError, match="^released DOPE checkpoint has no tensor model_state_dict$"):
        auto_fuse.load_released_dope(path)
