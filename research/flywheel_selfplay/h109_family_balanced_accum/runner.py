#!/usr/bin/env python3
"""H109 CPU-only preparation contract for family-balanced training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from datetime import datetime, timezone

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
        "runner_sha256": _sha256(Path(__file__)),
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


READOUT_DRAWS = 10_000
READOUT_SEED = 20261031
READOUT_MARGIN = 0.001
READOUT_HARM = -0.01
DOPE_SHA256 = execution_plan()["frozen_dope_sha256"]


def prereg_sha256() -> str:
    """Hash the committed H109 protocol used to qualify cell receipts."""
    return _sha256(Path(__file__).with_name("PREREG.md"))


def _valid_sha(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_cell_receipt(
    cell: dict, *, synthetic: bool, emitted: bool = False
) -> dict:
    if not isinstance(cell, dict) or cell.get("schema") != "jope.h109.cell-outcome/v1":
        raise ValueError("H109 cell outcome must use the registered receipt schema")
    if (
        cell.get("wave") != "H109"
        or cell.get("arm") not in ARMS
        or cell.get("seed") not in SEEDS
        or type(cell.get("attempt")) is not int
        or cell["attempt"] < 1
        or cell.get("cell_status") != "COMPLETE"
        or cell.get("prereg_sha256") != prereg_sha256()
        or cell.get("fold_seed") != FOLD_SEED
        or cell.get("checkpoint_sha256") != R882_SHA256
        or cell.get("dope_before") != DOPE_SHA256
        or cell.get("dope_after") != DOPE_SHA256
        or cell.get("optimizer_updates") != UPDATES
        or cell.get("trained_parameter_count") != ADAPTER_PARAMS
        or cell.get("dl_val_reads") != 0
        or cell.get("sealed_holdout_meta_test_reads") != 0
    ):
        raise ValueError("H109 cell receipt identity, completion, or boundary mismatch")
    if synthetic:
        if cell.get("synthetic_fixture") is not True:
            raise ValueError("synthetic dry-run receipt must be explicitly marked")
        execution = cell.get("execution_provenance")
        if execution != {
            "gpu_used": False,
            "qualification_lock_held": False,
            "owner_authorized": False,
            "owner_authorization_ref": None,
            "qualification_lock_path": None,
            "fifo_ticket_id": None,
            "gpu_device": "synthetic",
        }:
            raise ValueError("synthetic receipt cannot claim GPU or FIFO execution")
    else:
        if cell.get("synthetic_fixture") is True:
            raise ValueError("synthetic fixture cannot be emitted as a real outcome")
        if (
            not _valid_sha(cell.get("source_commit"), 40)
            or not _valid_sha(cell.get("fold_assignment_sha256"), 64)
            or not _valid_sha(cell.get("runner_sha256"), 64)
        ):
            raise ValueError("H109 cell receipt source/fold/runner identity mismatch")
        execution = cell.get("execution_provenance")
        if (
            not isinstance(execution, dict)
            or execution.get("gpu_used") is not True
            or execution.get("qualification_lock_held") is not True
            or execution.get("owner_authorized") is not True
            or not isinstance(execution.get("owner_authorization_ref"), str)
            or not execution["owner_authorization_ref"].strip()
            or execution.get("qualification_lock_path")
            != str(Path.home() / "qualification.lock")
            or not isinstance(execution.get("fifo_ticket_id"), str)
            or not execution["fifo_ticket_id"].strip()
            or not isinstance(execution.get("gpu_device"), str)
            or not execution["gpu_device"].strip()
        ):
            raise ValueError(
                "real H109 receipt requires GPU, lock, owner, and FIFO evidence"
            )
    candidate_sha = cell.get("candidate_adapter_sha256")
    if not synthetic and not _valid_sha(candidate_sha, 64):
        raise ValueError("H109 candidate adapter SHA256 is required")
    if emitted:
        expected_receipt_schema = (
            "jope.h109.synthetic-cell-receipt/v1"
            if synthetic
            else "jope.h109.cell-receipt/v1"
        )
        if cell.get("receipt_schema_version") != expected_receipt_schema:
            raise ValueError("H109 emitted receipt schema/version is missing")
        if cell.get("synthetic_receipt") is not synthetic:
            raise ValueError("H109 receipt synthetic marker/schema mismatch")
    assessment = cell.get("assessment")
    reference = cell.get("r882_utility_by_family")
    utility = (
        assessment.get("utility_by_family") if isinstance(assessment, dict) else None
    )
    if not isinstance(utility, dict) or not isinstance(reference, dict):
        raise ValueError("H109 cell receipt requires per-family utility maps")
    families = sorted(utility)
    if len(families) < 5 or families != sorted(reference):
        raise ValueError("H109 cell receipt family roster is incomplete or unpaired")
    for mapping in (utility, reference):
        for family, value in mapping.items():
            if (
                not isinstance(family, str)
                or not family
                or isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError("H109 per-family utilities must be finite numbers")
    return cell


def _safe_output_directory(root: Path, *, synthetic: bool) -> Path:
    root = Path(root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("H109 receipt root must be an existing real directory")
    if synthetic:
        return root
    expected_parent = Path.home() / "jope-data" / "h109"
    if root.parent != expected_parent or not root.name.startswith("run-"):
        raise ValueError(
            "real H109 receipts must be written inside a run-* H109 data root"
        )
    for path in (expected_parent, expected_parent.parent, Path.home()):
        if path.is_symlink():
            raise ValueError("H109 receipt path cannot traverse symlinked directories")
    return root


def emit_outcome_receipt(
    cell: dict, output_root: Path, *, synthetic: bool = False
) -> dict:
    """Atomically emit one completed H109 cell receipt without replacing prior attempts."""
    _validate_cell_receipt(cell, synthetic=synthetic)
    if not synthetic and cell.get("runner_sha256") != _sha256(Path(__file__)):
        raise ValueError("H109 outcome runner SHA256 does not match this source")
    root = _safe_output_directory(Path(output_root), synthetic=synthetic)
    receipt_dir = root / "cell-receipts"
    if receipt_dir.is_symlink():
        raise ValueError("H109 cell receipt directory cannot be a symlink")
    receipt_dir.mkdir(mode=0o700, exist_ok=True)
    if not receipt_dir.is_dir():
        raise ValueError("H109 cell receipt path is not a directory")
    receipt = dict(cell)
    assessment = dict(cell["assessment"])
    assessment["utility_by_family"] = {
        family: float(value)
        for family, value in assessment["utility_by_family"].items()
    }
    receipt["assessment"] = assessment
    receipt["r882_utility_by_family"] = {
        family: float(value) for family, value in cell["r882_utility_by_family"].items()
    }
    receipt["emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["receipt_schema_version"] = (
        "jope.h109.synthetic-cell-receipt/v1"
        if synthetic
        else "jope.h109.cell-receipt/v1"
    )
    if synthetic:
        receipt["synthetic_receipt"] = True
    filename = (
        f"{cell['arm'].replace('+', 'plus')}-seed{cell['seed']}"
        f"-attempt{cell['attempt']}.json"
    )
    destination = receipt_dir / filename
    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=receipt_dir
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link atomically claims the final name and refuses to clobber it.
        os.link(temporary_path, destination)
        directory_fd = os.open(receipt_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        raise FileExistsError(
            f"H109 receipt already exists; preserve it: {destination}"
        )
    finally:
        temporary_path.unlink(missing_ok=True)
    raw = destination.read_bytes()
    return {
        "arm": cell["arm"],
        "seed": cell["seed"],
        "attempt": cell["attempt"],
        "path": str(destination),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "synthetic": synthetic,
    }


def read_outcome_receipts(
    output_root: Path,
    selected_attempts: dict[str, int],
    expected_sha256: dict[str, str],
    *,
    synthetic: bool = False,
) -> list[dict]:
    """Read exactly the selected 9 receipts and verify their externally recorded hashes."""
    expected_keys = {f"{arm}/seed{seed}" for arm in ARMS for seed in SEEDS}
    if set(selected_attempts) != expected_keys or set(expected_sha256) != expected_keys:
        raise RuntimeError(
            "INCOMPLETE: explicit attempt and receipt-hash maps required"
        )
    root = _safe_output_directory(Path(output_root), synthetic=synthetic)
    receipt_dir = root / "cell-receipts"
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise RuntimeError("H109 cell receipt directory is missing or unsafe")
    cells = []
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}/seed{seed}"
            attempt = selected_attempts[key]
            if type(attempt) is not int or attempt < 1:
                raise ValueError("H109 selected attempt must be a positive integer")
            digest = expected_sha256[key]
            if not _valid_sha(digest, 64):
                raise ValueError("H109 expected receipt SHA256 is malformed")
            filename = f"{arm.replace('+', 'plus')}-seed{seed}-attempt{attempt}.json"
            path = receipt_dir / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"H109 selected outcome receipt is missing/unsafe: {key}"
                )
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise RuntimeError(
                    f"H109 selected outcome receipt hash mismatch: {key}"
                )
            try:
                cell = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"H109 selected outcome receipt is invalid JSON: {key}"
                ) from exc
            if (
                cell.get("arm") != arm
                or cell.get("seed") != seed
                or cell.get("attempt") != attempt
            ):
                raise RuntimeError(f"H109 selected receipt identity mismatch: {key}")
            _validate_cell_receipt(cell, synthetic=synthetic, emitted=True)
            cells.append(cell)
    return cells


def _elo_rating(win_rate: float):
    if win_rate <= 0:
        return "-Infinity"
    if win_rate >= 1:
        return "Infinity"
    return float(1500 + 400 * math.log10(win_rate / (1 - win_rate)))


def _elo_readout(gain: np.ndarray, family_samples: np.ndarray) -> dict:
    scores = (gain > 0).astype(np.float64) + 0.5 * (gain == 0)
    per_seed_rates = scores.mean(axis=1)
    per_seed_boot = scores[:, family_samples].mean(axis=2)
    median_boot_rates = np.median(per_seed_boot, axis=0)
    return {
        "reference_elo": 1500,
        "per_seed": [
            {
                "seed": seed,
                "wins": int((gain[index] > 0).sum()),
                "ties": int((gain[index] == 0).sum()),
                "losses": int((gain[index] < 0).sum()),
                "elo": _elo_rating(float(per_seed_rates[index])),
            }
            for index, seed in enumerate(SEEDS)
        ],
        "median_elo": _elo_rating(float(np.median(per_seed_rates))),
        "median_ci95": [
            _elo_rating(float(rate))
            for rate in np.quantile(median_boot_rates, [0.025, 0.975])
        ],
        "median_lcb95": _elo_rating(float(np.quantile(median_boot_rates, 0.05))),
        "bootstrap_draws": int(family_samples.shape[0]),
    }


def summarize_readout(
    cells: list[dict], draws: int = READOUT_DRAWS, *, synthetic: bool = False
) -> dict:
    """Summarize complete H109 cell receipts; this function never reads files/data."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    indexed = {(cell.get("arm"), cell.get("seed")): cell for cell in cells}
    if (
        draws != READOUT_DRAWS
        or len(cells) != len(expected)
        or set(indexed) != expected
    ):
        raise RuntimeError(
            "INCOMPLETE: all nine unique H109 arm/seed receipts required"
        )
    prereg_hash = prereg_sha256()
    identity = None
    families = None
    references = {}
    utilities = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        seed_reference = None
        for arm in ARMS:
            cell = indexed[(arm, seed)]
            _validate_cell_receipt(cell, synthetic=synthetic, emitted=True)
            if (
                cell.get("schema") != "jope.h109.cell-outcome/v1"
                or cell.get("wave") != "H109"
                or cell.get("prereg_sha256") != prereg_hash
                or cell.get("fold_seed") != FOLD_SEED
                or cell.get("checkpoint_sha256") != R882_SHA256
                or cell.get("dope_before") != DOPE_SHA256
                or cell.get("dope_after") != DOPE_SHA256
                or cell.get("optimizer_updates") != UPDATES
                or cell.get("dl_val_reads") != 0
                or cell.get("sealed_holdout_meta_test_reads") != 0
            ):
                raise RuntimeError("INCOMPLETE: H109 identity/boundary mismatch")
            current_identity = (
                cell.get("source_commit"),
                cell.get("fold_assignment_sha256"),
                "synthetic" if synthetic else cell.get("runner_sha256"),
                "synthetic"
                if synthetic
                else cell["execution_provenance"]["fifo_ticket_id"],
                "synthetic"
                if synthetic
                else cell["execution_provenance"]["owner_authorization_ref"],
            )
            if not all(isinstance(value, str) and value for value in current_identity):
                raise RuntimeError("INCOMPLETE: H109 source/fold identity missing")
            if identity is not None and identity != current_identity:
                raise RuntimeError("INCOMPLETE: mixed H109 source/fold identities")
            identity = current_identity
            assessment = cell.get("assessment")
            reference = cell.get("r882_utility_by_family")
            values = (
                assessment.get("utility_by_family")
                if isinstance(assessment, dict)
                else None
            )
            if not isinstance(values, dict) or not isinstance(reference, dict):
                raise RuntimeError("INCOMPLETE: per-family utility maps required")
            current_families = sorted(values)
            if current_families != sorted(reference) or len(current_families) < 5:
                raise RuntimeError("INCOMPLETE: paired family roster mismatch")
            try:
                utility = np.asarray(
                    [values[f] for f in current_families], dtype=np.float64
                )
                base = np.asarray(
                    [reference[f] for f in current_families], dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "INCOMPLETE: numeric family utility required"
                ) from exc
            if not np.isfinite(utility).all() or not np.isfinite(base).all():
                raise RuntimeError("INCOMPLETE: non-finite family utility")
            if families is None:
                families = current_families
            elif families != current_families:
                raise RuntimeError("INCOMPLETE: inconsistent H109 family roster")
            if seed_reference is not None and not np.array_equal(seed_reference, base):
                raise RuntimeError("INCOMPLETE: r882 reference differs across arms")
            seed_reference = base
            utilities[arm].append(utility)
        references[seed] = seed_reference

    if any(
        not np.array_equal(references[SEEDS[0]], references[seed]) for seed in SEEDS[1:]
    ):
        raise RuntimeError("INCOMPLETE: r882 reference differs across paired seeds")

    if families is None or len(families) < 5:
        raise RuntimeError("INCOMPLETE: H109 family roster required")
    family_samples = np.random.default_rng(READOUT_SEED).integers(
        0, len(families), size=(READOUT_DRAWS, len(families))
    )
    gains = {}
    bootstrap_means = {}
    bootstrap_medians = {}
    prelim = {}
    for arm in ARMS:
        arm_utility = np.asarray(utilities[arm], dtype=np.float64)
        baseline = np.asarray([references[seed] for seed in SEEDS], dtype=np.float64)
        gain = arm_utility - baseline
        gains[arm] = gain
        family_gain = gain.mean(axis=0)
        boot_mean = family_gain[family_samples].mean(axis=1)
        boot_median = np.median(family_gain[family_samples], axis=1)
        bootstrap_means[arm] = boot_mean
        bootstrap_medians[arm] = boot_median
        per_seed_means = gain.mean(axis=1)
        harmed = family_gain < READOUT_HARM
        prelim[arm] = {
            "per_seed_mean_gain": per_seed_means.tolist(),
            "family_mean_gain": float(family_gain.mean()),
            "family_median_gain": float(np.median(family_gain)),
            "family_median_ci95": np.quantile(boot_median, [0.025, 0.975]).tolist(),
            "uncorrected_mean_lcb95": float(np.quantile(boot_mean, 0.05)),
            "harm_threshold": READOUT_HARM,
            "harm_fraction": float(harmed.mean()),
            "harmed_families": [
                f for f, bad in zip(families, harmed, strict=True) if bad
            ],
            "elo_vs_r882": _elo_readout(gain, family_samples),
        }

    tail = {
        arm: float(
            (1 + np.count_nonzero(bootstrap_means[arm] <= READOUT_MARGIN)) / (draws + 1)
        )
        for arm in ARMS
    }
    order = sorted(ARMS, key=lambda arm: (tail[arm], ARMS.index(arm)))
    holm, continue_gate, adjusted_so_far = {}, True, 0.0
    for rank, arm in enumerate(order):
        alpha = 0.05 / (len(ARMS) - rank)
        adjusted_so_far = max(adjusted_so_far, min(1.0, tail[arm] * (len(ARMS) - rank)))
        lcb = float(np.quantile(bootstrap_means[arm], alpha))
        continue_gate = bool(
            continue_gate and tail[arm] <= alpha and lcb > READOUT_MARGIN
        )
        holm[arm] = {
            "rank": rank + 1,
            "alpha": alpha,
            "raw_tail_area": tail[arm],
            "holm_adjusted_tail_area": adjusted_so_far,
            "holm_lcb95": lcb,
            "passes": continue_gate,
        }
    arms = {}
    for arm in ARMS:
        advance = bool(
            arm != "CONTROL"
            and holm[arm]["passes"]
            and prelim[arm]["family_median_gain"] >= 0
            and prelim[arm]["harm_fraction"] <= HARM_LIMIT
        )
        arms[arm] = {
            **prelim[arm],
            "holm": holm[arm],
            "advance_eligible": arm != "CONTROL",
            "admitted": advance,
        }
    return {
        "schema": "jope.h109.readout/v1",
        "complete": True,
        "protocol_sha256": prereg_hash,
        "fold_seed": FOLD_SEED,
        "source_commit": identity[0],
        "fold_assignment_sha256": identity[1],
        "runner_sha256": None if synthetic else identity[2],
        "fifo_ticket_id": None if synthetic else identity[3],
        "owner_authorization_ref": None if synthetic else identity[4],
        "cell_attempts": {
            f"{arm}/seed{seed}": indexed[(arm, seed)]["attempt"]
            for arm in ARMS
            for seed in SEEDS
        },
        "candidate_adapter_sha256": {
            f"{arm}/seed{seed}": indexed[(arm, seed)].get("candidate_adapter_sha256")
            for arm in ARMS
            for seed in SEEDS
        },
        "families": families,
        "bootstrap_draws": READOUT_DRAWS,
        "bootstrap_seed": READOUT_SEED,
        "holm_family": list(ARMS),
        "arms": arms,
        "input_files_opened_by_summarizer": 0,
    }


def readout_cpu_dry_run(receipt_root: Path) -> dict:
    """Exercise H109 readout statistics on synthetic arrays only."""
    families = [f"fixture-{i:02d}" for i in range(40)]
    reference = {family: 0.0 for family in families}
    delta = np.asarray([0.02] * 30 + [-0.005] * 10, dtype=np.float64)
    cells = []
    for arm in ARMS:
        for seed_index, seed in enumerate(SEEDS):
            values = delta + seed_index * 0.0001
            cells.append(
                {
                    "schema": "jope.h109.cell-outcome/v1",
                    "wave": "H109",
                    "arm": arm,
                    "seed": seed,
                    "attempt": 1,
                    "cell_status": "COMPLETE",
                    "trained_parameter_count": ADAPTER_PARAMS,
                    "synthetic_fixture": True,
                    "execution_provenance": {
                        "gpu_used": False,
                        "qualification_lock_held": False,
                        "owner_authorized": False,
                        "owner_authorization_ref": None,
                        "qualification_lock_path": None,
                        "fifo_ticket_id": None,
                        "gpu_device": "synthetic",
                    },
                    "prereg_sha256": prereg_sha256(),
                    "fold_seed": FOLD_SEED,
                    "source_commit": "synthetic-source",
                    "fold_assignment_sha256": "synthetic-fold",
                    "checkpoint_sha256": R882_SHA256,
                    "dope_before": DOPE_SHA256,
                    "dope_after": DOPE_SHA256,
                    "optimizer_updates": UPDATES,
                    "dl_val_reads": 0,
                    "sealed_holdout_meta_test_reads": 0,
                    "assessment": {
                        "utility_by_family": {
                            family: float(value)
                            for family, value in zip(families, values, strict=True)
                        }
                    },
                    "r882_utility_by_family": reference,
                }
            )
    emission_receipts = [
        emit_outcome_receipt(cell, receipt_root, synthetic=True) for cell in cells
    ]
    selected_attempts = {
        f"{receipt['arm']}/seed{receipt['seed']}": receipt["attempt"]
        for receipt in emission_receipts
    }
    expected_sha256 = {
        f"{receipt['arm']}/seed{receipt['seed']}": receipt["sha256"]
        for receipt in emission_receipts
    }
    emitted_cells = read_outcome_receipts(
        receipt_root, selected_attempts, expected_sha256, synthetic=True
    )
    metrics = summarize_readout(emitted_cells, synthetic=True)
    if (
        not metrics["complete"]
        or metrics["input_files_opened_by_summarizer"] != 0
        or metrics["arms"]["FBAL"]["family_mean_gain"] <= 0.001
        or not metrics["arms"]["FBAL"]["admitted"]
        or metrics["arms"]["CONTROL"]["admitted"]
    ):
        raise RuntimeError("H109 synthetic readout self-check failed")
    return {
        "schema": "jope.h109.readout-cpu-dry-run/v1",
        "status": "READOUT_DRY_RUN_PASS",
        "synthetic_families": len(families),
        "synthetic_cells": len(cells),
        "real_outcome_files_read": 0,
        "synthetic_receipts_read": len(emission_receipts),
        "synthetic_receipts_emitted": len(emission_receipts),
        "receipt_sha256": {
            Path(receipt["path"]).name: receipt["sha256"]
            for receipt in emission_receipts
        },
        "GPU_used": False,
        "qualification_lock_used": False,
        "FIFO_ticket_created": False,
        "synthetic_fb_admitted": metrics["arms"]["FBAL"]["admitted"],
        "synthetic_control_admitted": metrics["arms"]["CONTROL"]["admitted"],
        "bootstrap_draws": metrics["bootstrap_draws"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--readout-cpu-dry-run", action="store_true")
    parser.add_argument("--receipt-root", type=Path)
    args = parser.parse_args()
    if args.readout_cpu_dry_run:
        if args.receipt_root is None:
            parser.error("--readout-cpu-dry-run requires a fresh --receipt-root")
        args.receipt_root.mkdir(parents=True, exist_ok=False)
        print(
            json.dumps(
                readout_cpu_dry_run(args.receipt_root),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        raise SystemExit(0)
    if args.receipt_root is not None:
        parser.error("--receipt-root is only valid with --readout-cpu-dry-run")
    print(
        json.dumps(
            cpu_dry_run(args.scratch_root), indent=2, sort_keys=True, allow_nan=False
        )
    )
