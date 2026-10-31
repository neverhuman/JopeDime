#!/usr/bin/env python3
"""H109 CPU-only preparation contract for family-balanced training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import stat

import numpy as np
import torch

ARMS = ("CONTROL", "FBAL", "FBAL+WCR")
SEEDS = (39, 40, 41)
FOLD_SEED = 10900
ADAPTER_PARAMS = 29_211
MICROBATCHES = 4
MICROBATCH_SIZE = 16
UPDATES = 40
WARMUP_STEPS = 2  # ceil(5% of 40 updates)
RESTART_PERIOD = UPDATES / 3.0
BASE_LR = 1e-4
R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MARGIN = 0.001
HARM_LIMIT = 0.10


def family_folds(
    rows: list[dict], *, count: int = 5, seed: int = FOLD_SEED
) -> list[list[str]]:
    """Group only eligible H50 DL-Train families into deterministic folds."""
    if count != 5 or seed != FOLD_SEED or not rows:
        raise ValueError("H109 requires the fixed five-fold DL-Train split")
    families = set()
    for row in rows:
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("has_meta_test") is True
            or row.get("origin") not in (None, "real")
            or not isinstance(row.get("family_id"), str)
            or not row["family_id"]
        ):
            raise ValueError("H109 fold input must contain real DL-Train families only")
        families.add(row["family_id"])
    if len(families) < count:
        raise ValueError("H109 requires at least five distinct DL-Train families")
    rng = np.random.default_rng(seed)
    ordered = sorted(families)
    permutation = rng.permutation(len(ordered))
    folds = [[] for _ in range(count)]
    for index, family_index in enumerate(permutation):
        folds[index % count].append(ordered[int(family_index)])
    return [sorted(fold) for fold in folds]


def fold_receipt(rows: list[dict]) -> dict:
    folds = family_folds(rows)
    all_families = {row["family_id"] for row in rows}
    flattened = [family for fold in folds for family in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != all_families:
        raise RuntimeError("H109 fold assignment is not an exact family partition")
    payload = {"seed": FOLD_SEED, "folds": folds}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "fold_a_families": sorted(set().union(*folds[:4])),
        "fold_b_families": folds[4],
        "fold_assignment_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_split": "H50 DL-Train only",
    }


def scratch_path(root: Path, arm: str, seed: int, attempt: int = 1) -> Path:
    if arm not in ARMS or seed not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError("invalid H109 cell/attempt")
    return Path(root) / "scratch" / f"{arm}-seed{seed}-attempt{attempt}"


def sample_microbatches(
    family_contexts: dict[str, list[str]], arm: str, seed: int
) -> list[list[str]]:
    """Build a fixed 40×4×16 replacement-sampling plan for the two samplers."""
    if arm not in ARMS or seed not in SEEDS or not family_contexts:
        raise ValueError("invalid H109 sampler inputs")
    if any(
        not family or not contexts or len(set(contexts)) != len(contexts)
        for family, contexts in family_contexts.items()
    ):
        raise ValueError(
            "H109 sampler requires unique contexts in every nonempty family"
        )
    families = sorted(family_contexts)
    contexts = [context for family in families for context in family_contexts[family]]
    rng = np.random.default_rng(seed * 1_000_003 + 109_001)
    plan = []
    for _ in range(UPDATES * MICROBATCHES):
        if arm == "CONTROL":
            selected = rng.choice(contexts, size=MICROBATCH_SIZE, replace=True)
        else:
            selected_families = rng.choice(families, size=MICROBATCH_SIZE, replace=True)
            selected = [
                rng.choice(family_contexts[family]) for family in selected_families
            ]
        plan.append([str(context) for context in selected])
    return plan


def require_fresh_scratch(root: Path, arm: str, seed: int, attempt: int = 1) -> Path:
    path = scratch_path(root, arm, seed, attempt)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"H109 cell scratch already exists or is unsafe: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_r882_copy(
    source: Path, cell_scratch: Path, *, expected_sha256: str = R882_SHA256
) -> Path:
    """Copy the read-only r882 source into an isolated, cell-unique scratch dir."""
    source, cell_scratch = Path(source), Path(cell_scratch)
    if source.is_symlink() or not source.is_file():
        raise ValueError("H109 r882 source must be a regular read-only file")
    if source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("H109 r882 source must remain read-only")
    if cell_scratch.is_symlink() or cell_scratch.exists():
        raise FileExistsError("H109 cell scratch must be new and non-symlink")
    cell_scratch.mkdir(parents=True)
    staged = cell_scratch / "r882_adapter_copy.pt"
    shutil.copy2(source, staged)
    if staged.samefile(source) or staged.resolve() == source.resolve():
        raise RuntimeError("H109 staged checkpoint aliases the source")
    digest = _sha256(staged)
    if digest != expected_sha256:
        raise RuntimeError("H109 staged r882 checkpoint hash mismatch")
    staged.chmod(0o444)
    return staged


def learning_rate(arm: str, update: int) -> float:
    if arm not in ARMS or type(update) is not int or not 0 <= update < UPDATES:
        raise ValueError("invalid H109 arm or optimizer update")
    if arm != "FBAL+WCR":
        return BASE_LR
    if update < WARMUP_STEPS:
        return BASE_LR * (update + 1) / WARMUP_STEPS
    phase = ((update - WARMUP_STEPS) % RESTART_PERIOD) / RESTART_PERIOD
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * phase))


def set_update_learning_rate(optimizer, arm: str, update: int) -> float:
    """Apply the registered per-update schedule to every optimizer parameter group."""
    rate = learning_rate(arm, update)
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate


def accumulate_step(params, optimizer, microbatch_losses):
    """Backpropagate four equally weighted microbatches into one optimizer step."""
    losses = tuple(microbatch_losses)
    if len(losses) != MICROBATCHES or not params:
        raise ValueError("H109 requires four microbatches and adapter parameters")
    optimizer.zero_grad(set_to_none=True)
    for loss in losses:
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("H109 microbatch loss must be finite scalar")
        (loss / MICROBATCHES).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(torch.stack([loss.detach() for loss in losses]).mean())


def execution_plan() -> dict:
    return {
        "schema": "jope.h109.cpu-preparation/v1",
        "status": "prepared_not_executed",
        "arms": list(ARMS),
        "paired_seeds": list(SEEDS),
        "baseline": "immutable r882 checkpoint; primary comparison is each candidate arm vs r882",
        "updates_per_cell": UPDATES,
        "microbatches_per_update": MICROBATCHES,
        "microbatch_size": MICROBATCH_SIZE,
        "effective_batch_size": MICROBATCHES * MICROBATCH_SIZE,
        "sampler": {
            "CONTROL": "uniform over DL-Train contexts",
            "FBAL": "uniform family then uniform context within family",
            "FBAL+WCR": "uniform family then uniform context within family",
        },
        "schedule": {
            "CONTROL": "flat 1e-4",
            "FBAL": "flat 1e-4",
            "FBAL+WCR": {
                "warmup_fraction": 0.05,
                "warmup_updates": WARMUP_STEPS,
                "cosine_restart_period_updates": RESTART_PERIOD,
            },
        },
        "admission": {
            "strict_margin": MARGIN,
            "ties_fail": True,
            "harm_fraction_max": HARM_LIMIT,
            "family_bootstrap": "equal-family paired bootstrap; median, two-sided CI95 and one-sided LCB95",
            "holm_correction": "three candidate-vs-r882 comparisons",
        },
        "adapter_parameter_count": ADAPTER_PARAMS,
        "frozen_dope_sha256": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
        "r882_checkpoint_sha256": R882_SHA256,
        "fold_source": "real H50 DL-Train only; five whole-family folds, first four train / fifth score",
        "dl_val_reads": 0,
        "sealed_holdout_meta_test_reads": 0,
        "outcome_reads": 0,
        "optimizer_updates": 0,
        "checkpoint_reads": 0,
        "gpu_used": False,
        "gpu_ticket_created": False,
        "qualification_authorized": False,
        "scratch_pattern": "scratch/{arm}-seed{seed}-attemptN/ (unique per cell and attempt)",
    }


def cpu_dry_run(scratch_root: Path | None = None) -> dict:
    """Exercise fold, accumulation, schedule, and unique scratch using synthetic data."""
    rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"fixture-{i}"}
        for i in range(10)
    ]
    folds = fold_receipt(rows)
    params = [torch.nn.Parameter(torch.tensor([1.0]))]
    optimizer = torch.optim.SGD(params, lr=0.01)
    loss = accumulate_step(
        params, optimizer, [params[0].square().sum() for _ in range(MICROBATCHES)]
    )
    scratch = [
        scratch_path(Path("/synthetic"), arm, seed) for arm in ARMS for seed in SEEDS
    ]
    if len(scratch) != len(set(scratch)) or not math.isfinite(loss):
        raise RuntimeError("H109 synthetic dry-run integrity check failed")
    if not (learning_rate("FBAL+WCR", 0) < learning_rate("FBAL+WCR", 1)):
        raise RuntimeError("H109 warmup is not increasing")
    if learning_rate("CONTROL", 0) != BASE_LR or learning_rate("FBAL", 0) != BASE_LR:
        raise RuntimeError("H109 matched flat controls changed")
    staging_check = "not_run"
    if scratch_root is not None:
        scratch_root = Path(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)
        # A tiny synthetic file proves copy isolation without reading a checkpoint.
        source = scratch_root / "synthetic-r882-placeholder"
        source.write_bytes(b"synthetic placeholder; not a checkpoint")
        source.chmod(0o444)
        expected = _sha256(source)
        staged = stage_r882_copy(
            source,
            require_fresh_scratch(scratch_root, "CONTROL", 39),
            expected_sha256=expected,
        )
        if _sha256(staged) != expected or staged.samefile(source):
            raise RuntimeError("H109 synthetic checkpoint staging failed")
        staging_check = "synthetic_copy_verified"
    return {
        **execution_plan(),
        "status": "cpu_dry_run_pass",
        "synthetic_fold_assignment_sha256": folds["fold_assignment_sha256"],
        "synthetic_fold_family_counts": [len(fold) for fold in folds["folds"]],
        "synthetic_accumulated_loss": loss,
        "synthetic_parameter_after_update": float(params[0].detach()),
        "unique_synthetic_scratch_cells": len(scratch),
        "synthetic_staging_check": staging_check,
        "schedule_probe": [learning_rate("FBAL+WCR", i) for i in (0, 1, 2, 15, 28, 39)],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            cpu_dry_run(args.scratch_root), indent=2, sort_keys=True, allow_nan=False
        )
    )
