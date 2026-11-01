from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).with_name("run_acceptance.py")
SPEC = importlib.util.spec_from_file_location("rnk10_run_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class _Predictor:
    def __init__(self, *, reject_multimodal: bool = False) -> None:
        self.reject_multimodal = reject_multimodal
        self.arguments: dict[str, torch.Tensor] | None = None

    def predict_costs(self, **kwargs):
        self.arguments = kwargs
        if self.reject_multimodal:
            assert "mm_tokens_10x32" not in kwargs
            assert "mm_mask_10" not in kwargs
        return {"total_cost": torch.arange(8, dtype=torch.float32).reshape(1, 8)}


class _Adapter:
    def __init__(self) -> None:
        self.anchor = _Predictor(reject_multimodal=True)

    def predict_costs(self, **kwargs):
        self.guided_arguments = kwargs
        return {"total_cost": torch.arange(8, dtype=torch.float32).reshape(1, 8)}


class _Net:
    def __init__(self) -> None:
        self.adapter_10tok = _Adapter()


def test_predict_pair_keeps_multimodal_inputs_only_on_guided_adapter() -> None:
    net = _Net()
    cid = "context"
    pool = {
        "rec": {"context_sha256": cid},
        "program": np.zeros((8, 412), dtype=np.float32),
        "proj16": np.zeros((8, 16), dtype=np.float64),
        "fitness": np.arange(8, dtype=np.float64),
        "gate": np.ones(8, dtype=bool),
    }

    guided, anchor = module.predict_pair(
        net,
        {cid: torch.zeros((10, 32), dtype=torch.float32)},
        {cid: torch.ones(10, dtype=torch.bool)},
        pool,
    )

    assert guided.tolist() == anchor.tolist() == list(map(float, range(8)))
    assert "mm_tokens_10x32" in net.adapter_10tok.guided_arguments
    assert "mm_mask_10" in net.adapter_10tok.guided_arguments
    assert "mm_tokens_10x32" not in net.adapter_10tok.anchor.arguments
    assert "mm_mask_10" not in net.adapter_10tok.anchor.arguments
