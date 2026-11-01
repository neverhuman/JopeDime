from __future__ import annotations

import hashlib
import json

import torch
import pytest

from objective import (
    AUXILIARY_WEIGHT,
    CAUSAL_MARGIN,
    landmark_causality_hinge,
    training_loss,
)
from train_candidate import independent_state_for_export
from safetensors.torch import load_file, save_file
from evaluate_candidate import serving_selection
from train_candidate import authenticate_train_roster


def _roster_fixture(tmp_path, mutate=None):
    source = {
        "source_manifest_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
    }
    source_bytes = json.dumps(source, sort_keys=True).encode()
    (tmp_path / "source_manifest.json").write_bytes(source_bytes)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    rows = []
    for index in range(1219):
        rows.append(
            {
                "context_sha256": f"{index:064x}",
                "family_id": f"family-{index % 205}",
                "dl_split": "DL-Train",
                "task": "regression",
                "source_split": "train",
                "pool_sha256": "a" * 64,
                "train_only_jtt1_sha256": "b" * 64,
            }
        )
    if mutate:
        mutate(rows)
    packet = {
        "schema": "jope.rnk10-m5-train-packet/v1",
        "source_selected_sha256": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
        "source_manifest_sha256": source_sha,
        "train_contexts": 1219,
        "train_families": 205,
        "task": "regression",
        "split": "DL-Train",
        "rows": rows,
    }
    packet_bytes = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "selected_train.json").write_bytes(packet_bytes)
    return hashlib.sha256(packet_bytes).hexdigest(), source_sha


def test_authenticated_roster_accepts_exact_pinned_metadata_without_payloads(tmp_path):
    packet_sha, source_sha = _roster_fixture(tmp_path)
    manifest, rows = authenticate_train_roster(
        tmp_path, packet_sha256=packet_sha, source_manifest_sha256=source_sha
    )
    assert len(rows) == 1219
    assert manifest["split"] == "DL-Train"


def test_authenticated_roster_rejects_changed_context_before_payload_access(tmp_path):
    packet_sha, source_sha = _roster_fixture(
        tmp_path, lambda rows: rows[0].update(context_sha256="../pool")
    )
    with pytest.raises(RuntimeError, match="invalid or non-training"):
        authenticate_train_roster(
            tmp_path, packet_sha256=packet_sha, source_manifest_sha256=source_sha
        )


def test_authenticated_roster_rejects_changed_role_before_payload_access(tmp_path):
    packet_sha, source_sha = _roster_fixture(
        tmp_path, lambda rows: rows[0].update(dl_split="DL-Val")
    )
    with pytest.raises(RuntimeError, match="invalid or non-training"):
        authenticate_train_roster(
            tmp_path, packet_sha256=packet_sha, source_manifest_sha256=source_sha
        )


def test_authenticated_roster_rejects_changed_payload_pin_before_payload_access(
    tmp_path,
):
    packet_sha, source_sha = _roster_fixture(
        tmp_path, lambda rows: rows[0].update(pool_sha256="x" * 64)
    )
    with pytest.raises(RuntimeError, match="invalid or non-training"):
        authenticate_train_roster(
            tmp_path, packet_sha256=packet_sha, source_manifest_sha256=source_sha
        )


def test_m5_selection_uses_stable_cost_order_for_the_eight_proposals():
    costs = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    ).numpy()
    assert serving_selection(costs).tolist() == [7, 6, 5, 4]


def test_m5_selection_fails_closed_on_nonfinite_costs():
    costs = torch.zeros(12).numpy()
    costs[4] = float("nan")
    with pytest.raises(ValueError, match="twelve finite slate costs"):
        serving_selection(costs)


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
