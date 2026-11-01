"""H185 adapter forward/gradient helpers and launch binding validator.

Reuses bit-exact static/dynamic adapter forward passes and None-as-zero 29,211-parameter gradient utilities
from `research.flywheel_selfplay.h177_pooled_selector.treatment`.
"""

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load_sibling_module(mod_name, rel_path):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    target = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_h177_treatment = _load_sibling_module(
    "h177_pooled_selector_treatment",
    "research/flywheel_selfplay/h177_pooled_selector/treatment.py",
)

PARAMETERS = _h177_treatment.PARAMETERS
TENSORS = _h177_treatment.TENSORS
adapter_forward = _h177_treatment.adapter_forward
adapter_forward_static = _h177_treatment.adapter_forward_static
adapter_l2_drift = _h177_treatment.adapter_l2_drift
flatten_grad_with_none_as_zero = _h177_treatment.flatten_grad_with_none_as_zero
grad_norms = _h177_treatment.grad_norms
precompute_static_slate = _h177_treatment.precompute_static_slate

__all__ = [
    "PARAMETERS",
    "TENSORS",
    "adapter_forward",
    "adapter_forward_static",
    "adapter_l2_drift",
    "flatten_grad_with_none_as_zero",
    "grad_norms",
    "precompute_static_slate",
    "validate_launch",
]


def validate_launch(
    launch,
    head,
    manifest_sha256,
    prereg_sha256,
    fold_sha256,
    h178_receipt_sha256,
    smoke=False,
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding mismatch")
    if smoke:
        return
    has_review = (type(launch.get("review_id")) is int and launch["review_id"] > 0) or (
        isinstance(launch.get("review_waiver"), str)
        and len(launch["review_waiver"].strip()) >= 20
    )
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or not has_review
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
        or launch.get("h178_receipt_sha256") != h178_receipt_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding mismatch")
