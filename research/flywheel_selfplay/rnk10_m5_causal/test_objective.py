from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
from evaluate_candidate import (
    candidate_source_pins_match,
    family_bootstrap_elo_interval,
    m5,
    predict_fixture_costs,
    serving_selection,
)
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


class _FixtureAdapter:
    def __init__(self):
        self.payloads = []

    def predict_costs(self, **payload):
        self.payloads.append(payload)
        width = payload["prog_412"].shape[1]
        return {
            "total_cost": torch.arange(width, dtype=torch.float32).reshape(1, width)
        }


class _FixtureNet:
    def __init__(self):
        self.adapter_10tok = _FixtureAdapter()


@pytest.mark.parametrize("width", [12, 20])
def test_m5_candidate_runner_consumes_pinned_batched_fixture_shapes(width):
    # These dimensions are taken from the pinned golden-index fixture metadata.
    fields = {
        "ctx_feat": np.zeros((1, 584), dtype=np.float32),
        "prog_412": np.zeros((1, width, 412), dtype=np.float32),
        "proj_16": np.zeros((1, width, 16), dtype=np.float32),
        "base_prior_cost": np.zeros((1, width), dtype=np.float32),
        "scale": np.ones((1,), dtype=np.float32),
        "sos_prior": np.zeros((1, width, 2), dtype=np.float32),
        "mm_tokens_10x32": np.zeros((1, 10, 32), dtype=np.float32),
        "mm_mask_10": np.ones((1, 10), dtype=np.float32),
        "revealed_indices": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        "revealed_raw_f": np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        "revealed_gate": np.ones((1, 4), dtype=np.float32),
    }
    net = _FixtureNet()

    predict_fixture_costs(net, fields, fields["mm_tokens_10x32"])
    predict_fixture_costs(net, fields, fields["mm_tokens_10x32"], reverse_fitness=True)

    assert net.adapter_10tok.payloads[0]["prog_412"].shape == (1, width, 412)
    assert net.adapter_10tok.payloads[0]["revealed_raw_f"].shape == (1, 4)
    assert net.adapter_10tok.payloads[1]["revealed_raw_f"].tolist() == [
        [40.0, 30.0, 20.0, 10.0]
    ]


def test_m5_runner_completes_full_loop_with_pinned_fixture_metadata_shapes(
    tmp_path, monkeypatch
):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    train_input = tmp_path / "train"
    train_input.mkdir()
    input_dir = tmp_path / "model-input"
    input_dir.mkdir()
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"synthetic candidate marker")
    contexts = [f"{index:064x}" for index in range(50)]
    (train_input / "selected_train.json").write_text(
        json.dumps({"rows": [{"context_sha256": context} for context in contexts]})
    )
    arrays = {}
    files = {}
    for context in contexts:
        for width in (12, 20):
            name = f"slates/{context}-{width}.safetensors"
            path = fixtures / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic pinned-shape fixture")
            arrays[str(path)] = {
                "input.ctx_feat": np.zeros((1, 584), dtype=np.float32),
                "input.prog_412": np.zeros((1, width, 412), dtype=np.float32),
                "input.proj_16": np.zeros((1, width, 16), dtype=np.float32),
                "input.base_prior_cost": np.zeros((1, width), dtype=np.float32),
                "input.scale": np.ones((1,), dtype=np.float32),
                "input.sos_prior": np.zeros((1, width, 2), dtype=np.float32),
                "input.mm_tokens_10x32": np.zeros((1, 10, 32), dtype=np.float32),
                "input.mm_mask_10": np.ones((1, 10), dtype=np.float32),
                "input.revealed_indices": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                "input.revealed_raw_f": np.asarray(
                    [[10.0, 20.0, 30.0, 40.0]], dtype=np.float32
                ),
                "input.revealed_gate": np.ones((1, 4), dtype=np.float32),
            }
            files[name] = {"sha256": "synthetic-fixture-hash"}
    index_path = fixtures / "index.json"
    index_path.write_text(
        json.dumps(
            {"schema": "jope.golden-fixtures/v1", "contexts": 50, "files": files}
        )
    )
    from evaluate_candidate import digest as original_digest

    def fixture_digest(path):
        path = str(path)
        if path == str(index_path):
            return "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393"
        if path in arrays:
            return files[Path(path).relative_to(fixtures).as_posix()]["sha256"]
        return original_digest(Path(path))

    net = _FixtureNet()
    monkeypatch.setattr("evaluate_candidate.digest", fixture_digest)
    monkeypatch.setattr("evaluate_candidate.load_numpy", lambda path: arrays[path])
    monkeypatch.setattr("evaluate_candidate.load_net", lambda *_: net)
    monkeypatch.setattr("evaluate_candidate.wait_for_cpu_headroom", lambda: None)
    output = tmp_path / "m5.json"
    m5(
        SimpleNamespace(
            fixtures=fixtures,
            train_input=train_input,
            input=input_dir,
            candidate=candidate,
            output=output,
            receipt_sha256="receipt-hash",
            evidence_sha256="evidence-hash",
            runner_sha256="runner-hash",
        )
    )

    result = json.loads(output.read_text())
    assert result["contexts"] == 50
    assert result["slates"] == 100
    assert result["serving_slates"] == 50
    assert result["finite_cost_values"] == 1600
    assert len(net.adapter_10tok.payloads) == 300
    assert any(
        payload["revealed_raw_f"].tolist() == [[40.0, 30.0, 20.0, 10.0]]
        for payload in net.adapter_10tok.payloads
    )


def test_dev_elo_family_bootstrap_interval_is_pinned_and_deterministic():
    records = [
        SimpleNamespace(
            family_id=f"family-{family:02d}",
            scores_by_contender={"candidate": 1.0, "r882": 1.0},
        )
        for family in range(19)
    ]
    first = family_bootstrap_elo_interval(records)
    second = family_bootstrap_elo_interval(records)
    assert first == second
    assert first["draws"] == 10_000
    assert first["seed"] == 20261101
    assert first["family_count"] == 19
    assert first["interval95_candidate_elo"] == [1500.0, 1500.0]
    assert first["interval_is_descriptive_only"] is True


def test_dev_elo_bootstrap_matches_canonical_solver_at_extreme_boundary():
    records = [
        SimpleNamespace(
            family_id=f"family-{family:02d}",
            scores_by_contender={"candidate": 1.0, "r882": 0.0},
        )
        for family in range(19)
    ]
    interval = family_bootstrap_elo_interval(records)
    assert interval["interval95_candidate_elo"] == [6300.0, 6300.0]


def test_candidate_source_closure_binds_model_preprocessing_and_fused_hashes():
    source_pins = {
        "/pinned/python/model.py": "a" * 64,
        "/pinned/python/landmarks.py": "b" * 64,
        "/pinned/input/cpu_exact_fused.py": (
            "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
        ),
    }
    canonical = json.dumps(source_pins, sort_keys=True, separators=(",", ":")).encode()
    receipt = {"source_pins": source_pins}
    evidence = {
        "source_pins": source_pins,
        "source_pins_sha256": hashlib.sha256(canonical).hexdigest(),
        "cpu_fused_source_sha256": source_pins["/pinned/input/cpu_exact_fused.py"],
    }
    assert candidate_source_pins_match(receipt, evidence)
    evidence["source_pins"] = {**source_pins, "/pinned/python/model.py": "c" * 64}
    assert not candidate_source_pins_match(receipt, evidence)


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
