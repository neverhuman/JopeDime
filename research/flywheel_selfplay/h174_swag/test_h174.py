"""Data-free H174 SWAG, fit-scope, selector, and resource-control tests."""

import numpy as np
import pytest
import torch
import json
from pathlib import Path

import analysis as stat
from runner import locked_b_panel
from treatment import (
    RANK,
    SAMPLES,
    SNAPSHOT_STEPS,
    average_cost_vectors,
    fit_swag,
    sample_vectors,
    unflatten_state,
    validate_fit_only,
)


def synthetic_snapshots():
    out = {}
    for i, step in enumerate(SNAPSHOT_STEPS):
        out[step] = {
            "a": torch.tensor([float(i), float(i * i)], dtype=torch.float32),
            "b": torch.tensor([float(i % 3)], dtype=torch.float32),
        }
    return out


def test_swag_uses_exact_16_snapshots_diagonal_and_rank10():
    snapshots = synthetic_snapshots()
    fit = fit_swag(snapshots, ("a", "b"))
    matrix = torch.stack(
        [torch.cat([snapshots[s]["a"], snapshots[s]["b"]]) for s in SNAPSHOT_STEPS]
    ).double()
    assert fit["snapshot_count"] == 16 and fit["rank"] == RANK == 10
    assert torch.equal(fit["mean"], matrix.mean(0).float())
    expected_var = (
        (matrix.square().mean(0) - matrix.mean(0).square()).clamp_min(0).float()
    )
    assert torch.allclose(fit["variance"], expected_var)
    assert fit["deviations"].shape == (10, 3)
    with pytest.raises(ValueError):
        fit_swag({s: v for s, v in snapshots.items() if s != 10}, ("a", "b"))


def test_swag_samples_are_seeded_paired_by_covariance_scale_and_adapter_only():
    fit = fit_swag(synthetic_snapshots(), ("a", "b"))
    first = sample_vectors(fit, 174101)
    second = sample_vectors(fit, 174101)
    assert torch.equal(first[0.5], second[0.5])
    assert first[0.5].shape == (SAMPLES, 3)
    assert torch.allclose(
        first[1.0] - fit["mean"],
        (first[0.5] - fit["mean"]) * np.sqrt(2),
        atol=1e-5,
        rtol=1e-6,
    )
    restored = unflatten_state(first[0.5][0], synthetic_snapshots()[10], ("a", "b"))
    assert set(restored) == {"a", "b"} and restored["a"].shape == (2,)


def test_bma_averages_predicted_costs_before_context_argmin():
    costs = [[0.0, 4.0], [3.0, 0.0], [3.0, 0.0]] + [[3.0, 0.0]] * 13
    mean_cost = average_cost_vectors(costs)
    assert int(torch.argmin(mean_cost)) == 1
    assert int(torch.argmin(torch.tensor(costs[0]))) == 0


def test_bma_rejects_wrong_sample_count_or_nonfinite_costs():
    with pytest.raises(ValueError):
        average_cost_vectors([[0.0, 1.0]])
    bad = [[0.0, 1.0]] * SAMPLES
    bad[0][0] = float("nan")
    with pytest.raises(ValueError):
        average_cost_vectors(bad)


def test_locked_b_panel_retains_raw_costs_choices_and_utilities_by_context():
    rows = [dict(context_sha256=f"c{i:04}") for i in range(41)]
    raw = {
        row["context_sha256"]: np.arange(24, dtype=np.float32) + i
        for i, row in enumerate(rows)
    }
    panel = dict(
        context_ids=[row["context_sha256"] for row in rows],
        choices=[0] * len(rows),
        values=[-float(i) for i in range(len(rows))],
    )
    locked = locked_b_panel(rows, raw, panel)
    assert locked["context_ids"] == panel["context_ids"]
    assert locked["cost_vectors"][7] == raw["c0007"].tolist()
    assert locked["choices"] == panel["choices"]
    assert locked["utilities"] == panel["values"]
    with pytest.raises(ValueError, match="context identity"):
        locked_b_panel(rows, {"wrong": np.zeros(24)}, panel)


def test_trajectory_scope_rejects_selector_assessment_and_val_families():
    assert validate_fit_only({"f1", "f2"}, {"f1"}, {"b"}, {"a"}, {"v"})
    for trajectory in ({"b"}, {"a"}, {"v"}):
        with pytest.raises(ValueError):
            validate_fit_only({"f1", "f2"}, trajectory, {"b"}, {"a"}, {"v"})
    with pytest.raises(ValueError):
        validate_fit_only({"f1", "b"}, {"f1"}, {"b"}, {"a"}, {"v"})


def test_binary_candidate_selector_enforces_gain_lcb_and_tie_rejection():
    rows = [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(41)]
    panel = np.zeros((41, 2))
    panel[:, 1] = 0.005
    accepted = stat.choose_candidate(rows, panel, 3)
    assert accepted["accepted"] and accepted["selected_candidate"]
    panel[:, 1] = 0.001
    tied = stat.choose_candidate(rows, panel, 3)
    assert not tied["accepted"] and not tied["selected_candidate"]
    panel[:, 1] = 0.005
    panel[:21, 1] = -1
    panel[21:, 1] = 1
    uncertain = stat.choose_candidate(rows, panel, 3)
    assert uncertain["ci95"][0] <= 0 and not uncertain["accepted"]


def test_crossfit_is_123_41_41_and_batches_use_only_fit_families():
    rows = [dict(context_sha256=f"c{i:04}", family_id=f"f{i:03}") for i in range(205)]
    fit, select, assess = stat.partition(rows, 0)
    groups = [{r["family_id"] for r in part} for part in (fit, select, assess)]
    assert [len(g) for g in groups] == [123, 41, 41]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    orders = {r["context_sha256"]: np.arange(48) for r in fit}
    plans = stat.batch_plan(set(orders), orders, 174101)
    assert len(plans) == 40
    assert all(set(cids) <= set(orders) for _, cids, _ in plans)
    assert all(
        len(cids) == 8 and all(len(ix) == 24 for ix in indexes)
        for _, cids, indexes in plans
    )


def test_fixed_fold_manifest_matches_the_preregistered_seed():
    folder = Path(__file__).parent
    manifest = json.loads((folder / "fold_manifest.json").read_text())
    all_families = sorted(
        {family for group in manifest["families"] for family in group}
    )
    rows = [dict(family_id=family) for family in all_families]
    assert manifest["fold_seed"] == stat.FOLD_SEED == 174140
    assert stat.folds(rows) == manifest["families"]
