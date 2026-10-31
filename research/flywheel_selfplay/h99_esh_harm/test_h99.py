"""CPU-only contract tests for H99 preparation; no packet or checkpoint reads."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402


def test_fixed_paired_cells_and_h80_budget():
    plan = runner.execution_plan()
    assert plan["arms"] == ["NULL", "ESH-l2", "ESH-l4"]
    assert plan["lambda_by_arm"] == {"ESH-l2": 2.0, "ESH-l4": 4.0}
    assert plan["paired_seeds"] == [36, 37, 38]
    assert len(plan["candidate_cells"]) == 6
    assert {key: plan["null_cell"][key] for key in ("arm", "seed", "generations")} == {
        "arm": "NULL",
        "seed": 36,
        "generations": 16,
    }
    assert all(cell["generations"] == 16 for cell in plan["candidate_cells"])
    assert all(cell["scratch_pattern"].endswith("-attemptN/") for cell in plan["candidate_cells"])
    assert plan["null_cell"]["scratch_pattern"] == "scratch/NULL-seed36-attemptN/"
    assert plan["fold_seed"] != 4900
    assert plan["outcome_reads"] == plan["checkpoint_reads"] == 0
    assert plan["gpu_used"] is False and plan["gpu_ticket_created"] is False
    assert plan["timing_cells_run"] == plan["optimizer_updates"] == 0


def test_seed_stream_is_fixed_and_rejects_unregistered_cells():
    assert runner.seed_stream(36, 1) == 36 * 1_000_003 + 97_409 + 1_009
    assert runner.seed_stream(36, 1) == runner.seed_stream(36, 1)
    for seed, generation in ((35, 1), (39, 1), (36, 0), (36, 17)):
        try:
            runner.seed_stream(seed, generation)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unregistered stream seed={seed}, generation={generation}")


def test_cpu_dry_run_reuses_h80_math_without_data_or_training():
    result = runner.cpu_dry_run()
    assert result["status"] == "cpu_dry_run_pass"
    assert result["shared_h80_math"]["perturbations"] == 32
    assert result["shared_h80_math"]["antithetic_pairs"] == 16
    assert result["shared_h80_math"]["gradient_finite"] is True
    assert result["shared_h80_math"]["synthetic_lambda4_fitness"] < result["shared_h80_math"]["synthetic_lambda2_fitness"]
    assert result["outcome_reads"] == result["checkpoint_reads"] == 0
    assert result["optimizer_updates"] == result["timing_cells_run"] == 0
    assert not result["gpu_used"] and not result["gpu_ticket_created"]


def test_fresh_fold_is_family_grouped_and_train_only():
    class FixtureH48:
        @staticmethod
        def family_folds(rows, *, count, seed):
            assert count == 5 and seed == 9900
            families = sorted({row["family_id"] for row in rows})
            return [set(families[i::count]) for i in range(count)]

    rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"f{i}"}
        for i in range(10)
        for _context in range(2)
    ]
    assignment = runner.fresh_family_assignment(rows, h48=FixtureH48)
    assert assignment["seed"] == 9900
    assert len(assignment["fold_a_families"]) == 8
    assert len(assignment["fold_b_families"]) == 2
    assert not set(assignment["fold_a_families"]) & set(assignment["fold_b_families"])
    assert len(assignment["fold_assignment_sha256"]) == 64


def test_fresh_fold_rejects_protected_and_non_train_rows():
    for row in (
        {"dl_split": "DL-Val", "task": "regression", "family_id": "f0"},
        {"dl_split": "DL-Train", "task": "regression", "family_id": "f0", "has_meta_test": True},
        {"dl_split": "DL-Train", "task": "classification", "family_id": "f0"},
    ):
        try:
            runner.fresh_family_assignment([row], h48=object())
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted forbidden H99 fold row: {row}")
