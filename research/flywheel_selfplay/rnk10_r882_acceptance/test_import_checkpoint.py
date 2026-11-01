from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name("import_checkpoint.py")
SPEC = importlib.util.spec_from_file_location("rnk10_import_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_extract_state_accepts_only_pinned_round_and_dope_identity() -> None:
    state = {"adapter_10tok.prog_mean": torch.zeros(412, dtype=torch.float32)}
    payload = {"round": 882, "dope_sha256": module.EXPECTED_DOPE_SHA256, "state_dict": state}
    assert module.extract_state(payload) == state


def test_extract_state_refuses_wrong_round_or_dope() -> None:
    state = {"adapter_10tok.prog_mean": torch.zeros(412, dtype=torch.float32)}
    payload = {"round": 881, "dope_sha256": module.EXPECTED_DOPE_SHA256, "state_dict": state}
    with pytest.raises(ValueError, match="round"):
        module.extract_state(payload)
    payload["round"] = 882
    payload["dope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="DOPE"):
        module.extract_state(payload)


def test_extract_state_rejects_non_tensor_values() -> None:
    payload = {
        "round": 882,
        "dope_sha256": module.EXPECTED_DOPE_SHA256,
        "state_dict": {"adapter_10tok.bad": "not a tensor"},
    }
    with pytest.raises(ValueError, match="tensors only"):
        module.extract_state(payload)


def test_state_dtype_contract_allows_only_int64_grid_indices() -> None:
    state = {
        "adapter_10tok.prog_mean": torch.zeros(412, dtype=torch.float32),
        "grid_idx_257": torch.arange(257, dtype=torch.int64),
        "grid_idx_13": torch.arange(13, dtype=torch.int64),
    }
    module.validate_state_dtypes(state)
    state["grid_idx_13"] = torch.arange(13, dtype=torch.int32)
    with pytest.raises(ValueError, match="pinned int64"):
        module.validate_state_dtypes(state)
