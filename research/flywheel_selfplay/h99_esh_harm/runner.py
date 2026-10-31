#!/usr/bin/env python3
"""H99's data-free CPU preparation contract, reusing H80's tested ES primitives."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H80_RUNNER = ROOT / "research/flywheel_selfplay/h80_long_es/runner.py"
FOLD_SEED = 9900
SEEDS = (36, 37, 38)
NULL_SEED = 36
GENERATIONS = 16
POPULATION = 32
SIGMA = 0.001
ADAPTER_PARAMS = 29_211
LAMBDA_BY_ARM = {"ESH-l2": 2.0, "ESH-l4": 4.0}
ARMS = ("NULL", "ESH-l2", "ESH-l4")


def load_h80():
    """Load the merged H80 implementation without touching its inputs or run roots."""
    spec = importlib.util.spec_from_file_location("h99_shared_h80_runner", H80_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared H80 runner at {H80_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_h48():
    h80 = load_h80()
    spec = importlib.util.spec_from_file_location("h99_shared_h48_folds", h80.H48)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load H48 family-fold helper at {h80.H48}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fresh_family_assignment(train_rows: list[dict], *, h48=None) -> dict:
    """Partition only admitted DL-Train rows into fresh whole-family folds."""
    if not train_rows:
        raise ValueError("H99 requires a nonempty DL-Train roster")
    for row in train_rows:
        if row.get("dl_split") != "DL-Train" or row.get("task") != "regression":
            raise ValueError("H99 fold source must contain only regression DL-Train rows")
        if row.get("has_meta_test") is True or row.get("origin") not in (None, "real"):
            raise ValueError("H99 rejects meta-test or non-real rows before fold assignment")
        if not isinstance(row.get("family_id"), str) or not row["family_id"]:
            raise ValueError("H99 requires a family ID for every admitted train row")
    fold_helper = h48 or load_h48()
    folds = fold_helper.family_folds(train_rows, count=5, seed=FOLD_SEED)
    all_families = {row["family_id"] for row in train_rows}
    flattened = [family for fold in folds for family in fold]
    if len(folds) != 5 or any(not fold for fold in folds):
        raise RuntimeError("H99 requires five nonempty fresh family folds")
    if len(flattened) != len(set(flattened)) or set(flattened) != all_families:
        raise RuntimeError("H99 fold assignment must partition whole train families exactly once")
    assignment = {"seed": FOLD_SEED, "folds": [sorted(fold) for fold in folds]}
    canonical = json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    return {
        **assignment,
        "fold_a_families": sorted(set().union(*folds[:4])),
        "fold_b_families": sorted(folds[4]),
        "fold_assignment_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_split": "DL-Train",
        "sealed_meta_test_inputs_opened": False,
    }


def seed_stream(seed: int, generation: int) -> int:
    if seed not in SEEDS or generation < 1 or generation > GENERATIONS:
        raise ValueError("unregistered H99 paired seed or generation")
    # Keep H80's shared per-seed/per-generation stream formula; only seed IDs differ.
    return seed * 1_000_003 + 97_409 + generation * 1_009


def execution_plan() -> dict:
    candidate_cells = [
        {
            "arm": arm,
            "seed": seed,
            "generations": GENERATIONS,
            "scratch_pattern": f"scratch/{arm}-seed{seed}-attemptN/",
        }
        for seed in SEEDS
        for arm in LAMBDA_BY_ARM
    ]
    return {
        "schema": "jope.h99.cpu-preparation/v1",
        "status": "prepared_not_executed",
        "fold_seed": FOLD_SEED,
        "fold_source": "approved H50 DL-Train rows only; whole-family grouped; never H80 Fold B",
        "arms": list(ARMS),
        "lambda_by_arm": dict(LAMBDA_BY_ARM),
        "candidate_cells": candidate_cells,
        "null_cell": {
            "arm": "NULL",
            "seed": NULL_SEED,
            "generations": GENERATIONS,
            "scratch_pattern": f"scratch/NULL-seed{NULL_SEED}-attemptN/",
        },
        "paired_seeds": list(SEEDS),
        "sigma": SIGMA,
        "population": POPULATION,
        "antithetic_pairs": POPULATION // 2,
        "adapter_parameter_count": ADAPTER_PARAMS,
        "selection": "final mean-theta only",
        "fold_b_scored": False,
        "outcome_reads": 0,
        "checkpoint_reads": 0,
        "optimizer_updates": 0,
        "gpu_used": False,
        "gpu_ticket_created": False,
        "timing_cells_run": 0,
        "qualification_authorized": False,
    }


def cpu_dry_run() -> dict:
    """Exercise shared H80 math with synthetic in-memory values only."""
    h80 = load_h80()
    fixture_rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"fixture-family-{i}"}
        for i in range(10)
    ]

    class FixtureFoldHelper:
        @staticmethod
        def family_folds(rows, *, count, seed):
            # Use H48's production grouping algorithm on synthetic family IDs.
            return load_h48().family_folds(rows, count=count, seed=seed)

    fold_assignment = fresh_family_assignment(fixture_rows, h48=FixtureFoldHelper)
    center = torch.zeros(ADAPTER_PARAMS, dtype=torch.float32)
    noise_seed = seed_stream(NULL_SEED, 1)
    noises, candidates = h80.antithetic_population(center, SIGMA, noise_seed)
    ranks = h80.centered_ranks(np.arange(POPULATION, dtype=np.float64).tolist())
    gradient = h80.es_gradient(noises, ranks, SIGMA)

    baseline = {f"fixture-family-{i}": 0.5 for i in range(10)}
    synthetic = dict(baseline)
    synthetic["fixture-family-0"] -= 0.02
    low = h80.compute_fold_a_fitness(synthetic, baseline, "ESH", esh_lambda=2.0)
    high = h80.compute_fold_a_fitness(synthetic, baseline, "ESH", esh_lambda=4.0)
    if not high["fitness"] < low["fitness"]:
        raise RuntimeError("H99 lambda escalation did not increase the synthetic harm penalty")
    if len(noises) != POPULATION // 2 or len(candidates) != POPULATION:
        raise RuntimeError("H80 antithetic population contract changed")
    if not torch.isfinite(gradient).all():
        raise RuntimeError("shared H80 ES gradient is non-finite")

    return {
        **execution_plan(),
        "status": "cpu_dry_run_pass",
        "synthetic_fold_assignment_sha256": fold_assignment["fold_assignment_sha256"],
        "synthetic_fold_a_family_count": len(fold_assignment["fold_a_families"]),
        "synthetic_fold_b_family_count": len(fold_assignment["fold_b_families"]),
        "shared_h80_math": {
            "source": str(H80_RUNNER.relative_to(ROOT)),
            "perturbations": len(candidates),
            "antithetic_pairs": len(noises),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "synthetic_lambda2_fitness": low["fitness"],
            "synthetic_lambda4_fitness": high["fitness"],
        },
    }
