"""H197 data-free checks for folds, loss identity, and clipping algebra."""

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis
import design
import json_writer
import source_pins
import treatment

ROOT = Path(__file__).resolve().parents[3]


def test_local_source_pins_resolve_from_study_root(tmp_path):
    source = tmp_path / "PREREG.md"
    source.write_text("registered source\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_pins.verify(tmp_path, {"PREREG.md": digest}) is None
    with pytest.raises(RuntimeError, match="source pin mismatch"):
        source_pins.verify(tmp_path, {"PREREG.md": "0" * 64})
    with pytest.raises(RuntimeError, match="unsafe H197 source path"):
        source_pins.verify(tmp_path, {"../outside": digest})


def test_preflight_requires_exact_baseline_full_corpus_and_zero_updates():
    counters = {
        "backward_started": 0,
        "backward_completed": 0,
        "step_started": 0,
        "step_completed": 0,
        "dry_context_backward_calls": 0,
    }
    result = design.validate_preflight(True, 1310, 1310, counters)
    assert result["status"] == "PASS"
    assert result["optimizer_updates"] == 0
    with pytest.raises(ValueError, match="exact native DL-Val"):
        design.validate_preflight(False, 1310, 1310, counters)
    with pytest.raises(ValueError, match="full TRAIN/DL-Val corpus"):
        design.validate_preflight(True, 1309, 1310, counters)
    with pytest.raises(ValueError, match="zero optimizer updates"):
        design.validate_preflight(True, 1310, 1310, {**counters, "step_started": 1})


def test_independent_family_folds_are_closed_and_reproducible():
    ids = [f"f{i:03}" for i in range(205)]
    left = design.family_folds(ids, 197141)
    assert left == design.family_folds(ids, 197141)
    assert [len(x) for x in left] == [41] * 5
    right = design.family_folds(ids, 197142)
    assert any(set(a) != set(b) for a, b in zip(left, right, strict=True))
    doc = design.fold_manifest(ids)
    assert doc["fold_seeds"] == [197141, 197142, 197143]
    assert set(doc["pairwise_corresponding_fold_jaccard"]) == {
        "197101-197102",
        "197101-197103",
        "197102-197103",
    }


def test_fit_plan_has_fixed_size_and_excludes_holdout_roles():
    rows = [
        {"context_sha256": f"c{i}", "family_id": f"f{i // 5}"} for i in range(205 * 5)
    ]
    ids = [r["context_sha256"] for r in rows[:700]]
    orders = {cid: np.arange(30) for cid in ids}
    plan = design.batch_plan(ids, orders, 197101)
    assert len(plan) == 40
    assert design.validate_fit_plan(plan, ids, ["b"], ["a"])
    bad = list(plan)
    bad[0] = (bad[0][0], ["a"] + bad[0][1][1:], bad[0][2])
    with pytest.raises(ValueError, match="FIT-only"):
        design.validate_fit_plan(bad, ids, ["b"], ["a"])


def test_per_context_loss_mean_matches_pinned_h146_live_loss():
    path = ROOT / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h197_test_h146", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch.manual_seed(7)
    prediction = torch.randn(8, 24)
    fitness = torch.randn(8, 24)
    gate = torch.rand(8, 24) > 0.2
    assert torch.equal(
        treatment.per_context_losses(prediction, fitness, gate).mean(),
        runtime.loss(prediction, fitness, gate),
    )


def test_clip_then_average_and_within_batch_factor_shuffle():
    p = torch.nn.Parameter(torch.zeros(1))
    q = torch.nn.Parameter(torch.zeros(1))
    rows = [
        (torch.tensor([3.0]), torch.tensor([0.0])),
        (torch.tensor([0.0]), torch.tensor([4.0])),
    ]
    norms = np.asarray([3.0, 4.0])
    factors = treatment.clip_factors(norms, 3.0)
    assert factors.tolist() == [1.0, 0.75]
    aggregate = treatment.aggregate_context_gradients(rows, factors)
    assert aggregate[0].item() == pytest.approx(1.5)
    assert aggregate[1].item() == pytest.approx(1.5)
    shuffled, order = treatment.shuffle_factors(factors, 197150)
    assert sorted(shuffled.tolist()) == sorted(factors.tolist())
    assert sorted(order.tolist()) == [0, 1]
    assert (
        json.loads(
            json.dumps({"factor_permutation": treatment.permutation_to_json(order)})
        )["factor_permutation"]
        == order.tolist()
    )
    treatment.set_gradients([p, q], aggregate)
    assert [p.grad.item(), q.grad.item()] == pytest.approx([1.5, 1.5])


def test_h197_thresholds_are_exact_medians_and_reject_zero():
    c, g = treatment.dry_thresholds([1.0, 3.0, 5.0], [0.5, 2.0, 9.0])
    assert c == 3.0
    assert g == 2.0
    with pytest.raises(ValueError, match="zero clipping"):
        treatment.dry_thresholds([0.0, 0.0], [1.0])


def test_reporting_family_matrix_is_equal_family():
    values = np.asarray([1.0, 3.0, -1.0])
    families = ["a", "a", "b"]
    ordered, means = analysis.family_matrix(values, families)
    assert ordered == ["a", "b"]
    assert means.tolist() == [2.0, -1.0]


def test_a_scoring_requires_intact_fixed_step_b_lock(tmp_path):
    panel = tmp_path / "B-seed-panel.npz"
    panel.write_bytes(b"synthetic panel")
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    lock = tmp_path / "pooled_b_selection_lock.json"
    lock.write_text(
        json.dumps(
            {
                "wave": "H197",
                "candidate_step": 40,
                "policy": "fixed candidate step 40; no B-based step selection",
                "A_scoring_authorized_after_lock": True,
                "b_panel_files": {panel.name: digest},
            }
        )
    )
    assert (
        treatment.validate_b_lock(lock, [panel.name])
        == hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    panel.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        treatment.validate_b_lock(lock, [panel.name])


def test_no_ties_or_zero_effects_can_advance():
    rows = [
        {"family_id": f"fam-{i:03}", "context_sha256": f"ctx-{i:03}"}
        for i in range(205)
    ]
    panel = np.zeros((3, 205, 5), dtype=np.float64)
    report = analysis.summarize(rows, panel)
    assert report["verdict"] == "NO_ADVANCE"
    assert report["serving_promotion"] is False
    assert report["contrasts"]["PCGC40_vs_R882"]["holm_p"] == 1.0
    assert report["harm_rate"]["rate"] == 0.0
    assert report["pcgc_strictly_beats_shuffled_placebo"] is False


def test_every_arm_trace_and_cell_seed_record_round_trips_strict_json(tmp_path):
    arms = design.ALL_ARMS
    records = {}
    for arm_index, arm in enumerate(arms):
        clipped = arm in ("PCGC40", "PCGC_SHUF40")
        trace = []
        for step in range(1, 41):
            permutation = (
                np.arange(8, dtype=np.int64)[::-1]
                if arm == "PCGC_SHUF40"
                else np.arange(8)
            )
            trace.append(
                {
                    "step": step,
                    "stream": 197100 + step,
                    "loss": 0.5,
                    "raw_or_clipped_batch_gradient_norm": 1.25,
                    "learning_rate": 1e-4,
                    "per_context_gradient_norms": [0.5] * 8 if clipped else None,
                    "clip_factors": [0.8] * 8 if clipped else None,
                    "assigned_clip_factors": [0.8] * 8 if clipped else None,
                    "factor_permutation": treatment.permutation_to_json(permutation)
                    if clipped
                    else None,
                    "fit_context_ids": [f"ctx-{i:03}" for i in range(8)],
                    "candidate_indices": [list(range(24)) for _ in range(8)],
                }
            )
        records[f"trace-{arm}"] = trace

    fit_receipts = []
    calibration_cells = []
    for seed in design.SEEDS:
        for outer in range(5):
            calibration_cells.append(
                {
                    "seed": seed,
                    "outer_fold": outer,
                    "C": 0.75,
                    "GCLIP_threshold": 0.5,
                    "fit_families": 123,
                    "step0_context_gradient_norms": [0.5] * 8,
                    "dry_pcgc_averaged_batch_gradient_norms": [0.5] * 5,
                    "dry_fit_batches": 5,
                    "dry_context_backward_calls": 40,
                }
            )
            for arm in design.ARMS:
                fit_receipts.append(
                    {
                        "status": "FIT_COMPLETE",
                        "seed": seed,
                        "outer_fold": outer,
                        "arm": arm,
                        "updates": 40,
                        "fit_context_count": 738,
                        "fit_family_count": 123,
                        "C": 0.75,
                        "GCLIP_threshold": 0.5,
                        "batch_plan_sha256": "a" * 64,
                        "adapter_sha256": "b" * 64,
                        "state_file_sha256": "c" * 64,
                        "trace_sha256": "d" * 64,
                    }
                )
    records["fit_receipts"] = fit_receipts
    records["update-counters"] = {
        "backward_started": 64,
        "backward_completed": 64,
        "step_started": 40,
        "step_completed": 40,
        "dry_context_backward_calls": 40,
        "cell": "197101-fold0",
        "arm": "PCGC_SHUF40",
    }
    records["calibration"] = {
        "status": "PASS",
        "cells": calibration_cells,
        "C_definition": "median context norm",
        "GCLIP_threshold_definition": "median dry aggregate norm",
    }

    lock = {
        "wave": "H197",
        "source_commit": "e" * 40,
        "fold_manifest_sha256": "f" * 64,
        "policy": "fixed candidate step 40; no B-based step selection",
        "primary_arm": "PCGC40",
        "arms": list(arms),
        "pooled_b_families": 205,
        "pooled_b_contexts": 1219,
        "candidate_step": 40,
        "b_panel_files": {f"B-{i}.npz": "a" * 64 for i in range(15)},
        "fit_state_files": {
            f"{s}-{o}-{a}": "b" * 64
            for s in design.SEEDS
            for o in range(5)
            for a in design.ARMS
        },
        "A_scoring_authorized_after_lock": True,
        "B_outcomes_are_selection_diagnostics_only": True,
    }
    records["b-lock"] = lock
    records["b-lock-receipt"] = {
        "sha256": "c" * 64,
        "fsynced": True,
        "candidate_step": 40,
        "selection_rule": "no step selection",
    }
    records["launch-binding"] = {
        "source_commit": "e" * 40,
        "source_tree": "f" * 40,
        "manifest_sha256": "a" * 64,
        "launch_sha256": "b" * 64,
        "initial_adapter_sha256": "c" * 64,
        "dope_sha256": "d" * 64,
        "fold_manifest_sha256": "e" * 64,
    }
    records["smoke"] = {
        "status": "PASS",
        "optimizer_updates": 0,
        "trainable_parameters": 29211,
        "protected_reads": 0,
    }
    records["baseline"] = {
        "status": "PASS",
        "contexts": 91,
        "mean_utility": 1.1377941898164525,
        "utility_bit_exact": True,
        "cost_bytes_exact": True,
        "TRAIN_outcome_pools_opened": 0,
        "initial_adapter_sha256": "f" * 64,
    }
    records["catalog-join"] = {"status": "PASS", "datasets": 276, "synthetic": 0}
    records["audit"] = {
        "opened": ["/home/ubuntu/jope-data/fixture"],
        "refused": [],
        "protected_reads": 0,
    }

    rows = [
        {"family_id": f"fam-{i:03}", "context_sha256": f"ctx-{i:03}"}
        for i in range(205)
    ]
    panel = np.zeros((3, 205, len(arms)), dtype=np.float64)
    summary = analysis.summarize(rows, panel)
    records["A-metrics"] = summary
    records["per-seed-summaries"] = [
        {
            "seed": seed,
            "utility_family_equal": {
                arm: summary["utility_family_equal"][arm][i] for arm in arms
            },
            "primary_gain": summary["contrasts"]["PCGC40_vs_R882"]["per_seed"][i],
            "positive_primary": summary["contrasts"]["PCGC40_vs_R882"]["per_seed"][i]
            > 0,
        }
        for i, seed in enumerate(design.SEEDS)
    ]
    records["DLVal-descriptive"] = {
        "contexts": 91,
        "families": 19,
        "baseline_mean": 1.1377941898164525,
        "per_seed_mean_utility": {arm: [1.0, 1.1, 1.2] for arm in arms},
        "role": "descriptive only; no selection or advancement",
    }
    clip_stats = {
        arm: {
            "context_gradient_count": 8 * 15,
            "clipped_fraction": 0.25,
            "factor_quantiles": [0.2, 0.5, 0.8, 1.0, 1.0],
        }
        for arm in ("PCGC40", "PCGC_SHUF40")
    }
    records["clip-stats"] = clip_stats
    records["metrics"] = {
        **summary,
        "per_context_clip_stats": clip_stats,
        "baseline": {
            "native_91_91_bit_exact": True,
            "mean_utility": 1.1377941898164525,
        },
        "optimizer_updates": 2400,
        "audit": {"opened": [], "refused": [], "protected_reads": 0},
    }
    records["preflight"] = {
        "status": "PASS",
        "loaded_train_and_dl_val_pools": 1310,
        "optimizer_updates": 0,
        "trainer_started": False,
    }
    records["failure"] = {
        "status": "INCOMPLETE",
        "error": "synthetic failure",
        "counters": {"step_completed": 0, "arm": "PCGC_SHUF40"},
        "audit": {"opened": [], "refused": [], "protected_reads": 0},
        "elapsed_seconds": 0.5,
    }

    for name, value in records.items():
        path = tmp_path / f"{name}.json"
        json_writer.save(path, value)
        assert json.loads(path.read_text()) == value
