from __future__ import annotations

import torch

from objective import (
    AUXILIARY_WEIGHT,
    CAUSAL_MARGIN,
    landmark_causality_hinge,
    training_loss,
)
from train_candidate import independent_state_for_export
from safetensors.torch import load_file, save_file


def test_hinge_is_zero_when_landmark_margin_is_met():
    baseline = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    ablated = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    best = torch.tensor([4])
    worst = torch.tensor([5])

    loss = landmark_causality_hinge(baseline, ablated, best, worst)

    assert float(loss) == 0.0


def test_hinge_is_positive_below_margin_and_reaches_both_paths():
    baseline = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    ablated = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    best = torch.tensor([4])
    worst = torch.tensor([5])

    loss = landmark_causality_hinge(baseline, ablated, best, worst)
    loss.backward()

    assert abs(float(loss) - CAUSAL_MARGIN) < 1e-7
    assert baseline.grad is not None and torch.count_nonzero(baseline.grad) == 2
    assert ablated.grad is not None and torch.count_nonzero(ablated.grad) == 2


def test_source_regression_objective_is_retained_and_auxiliary_added_once():
    baseline = torch.tensor([[0.1, 0.2, 0.4, 0.5, 0.0, 0.2]], requires_grad=True)
    ablated = torch.zeros_like(baseline, requires_grad=True)
    target = torch.tensor([[0.0, 0.1, 0.3, 0.4, 0.1, 0.2]])
    best = torch.tensor([4])
    worst = torch.tensor([5])

    total, regression, causal = training_loss(baseline, ablated, target, best, worst)

    assert torch.isfinite(total)
    assert abs(float(total - regression - AUXILIARY_WEIGHT * causal)) < 1e-7
    total.backward()
    assert baseline.grad is not None and torch.isfinite(baseline.grad).all()
    assert ablated.grad is not None and torch.isfinite(ablated.grad).all()


def test_safetensors_export_copies_tied_state_into_independent_storage(tmp_path):
    shared = torch.arange(8, dtype=torch.float32)
    exported = independent_state_for_export({"anchor": shared, "alias": shared})
    assert exported["anchor"].data_ptr() != exported["alias"].data_ptr()
    path = tmp_path / "candidate.safetensors"
    save_file(exported, str(path))
    loaded = load_file(str(path), device="cpu")
    torch.testing.assert_close(loaded["anchor"], shared)
    torch.testing.assert_close(loaded["alias"], shared)
