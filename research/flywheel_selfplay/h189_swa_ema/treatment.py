"""H189 adapter-only SWA/EMA rules and data-free provenance guards."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat
from h182_source_treatment import adapter_forward, gradient_norm

__all__ = ["adapter_forward", "gradient_norm"]


def validate_launch(
    launch,
    head,
    manifest_sha256,
    prereg_sha256,
    fold_sha256,
    smoke=False,
    source_tree=None,
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
        or launch.get("owner_exception") != "/home/ubuntu/nudges/hwave_t0805_h189_go.md"
        or launch.get("owner_ci_exception") is not True
    ):
        raise RuntimeError("merged source, owner ruling, or prereg launch binding")
    review = launch.get("same_agent_review")
    if (
        not isinstance(review, dict)
        or review.get("conclusion") != "PASS"
        or review.get("reviewer_kind") != "same-agent substantive review"
        or not review.get("review_receipt_sha256")
        or not review.get("receipt_path")
        or review.get("reviewed_tree_sha256") != source_tree
    ):
        raise RuntimeError("exact-tree self-review receipt binding")
    receipt_path = Path(review["receipt_path"])
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != review["review_receipt_sha256"]:
        raise RuntimeError("self-review receipt hash mismatch")
    receipt = json.loads(receipt_bytes)
    if (
        receipt.get("conclusion") != "PASS"
        or receipt.get("reviewer_kind") != "same-agent substantive review"
        or receipt.get("reviewed_tree_sha256") != source_tree
    ):
        raise RuntimeError("self-review receipt does not bind this exact source tree")


EMA_DECAY = 0.95
UPDATES = 80


def adapter_drift(initial, final):
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return float(
        np.sqrt(
            sum(
                float(
                    np.square(
                        np.asarray(final[name], dtype=np.float64)
                        - np.asarray(initial[name], dtype=np.float64)
                    ).sum()
                )
                for name in initial
            )
        )
    )


def validate_partitions(fit_families, b_families, a_families):
    fit, b, a = set(fit_families), set(b_families), set(a_families)
    if len(fit) != 123 or len(b) != 41 or len(a) != 41:
        raise ValueError("H189 requires 123 FIT / 41 B / 41 A families")
    if fit & b or fit & a or b & a:
        raise ValueError("FIT, B, and A families must be disjoint")
    if len(fit | b | a) != 205:
        raise ValueError("fold does not close over all 205 TRAIN families")


def validate_snapshots(snapshots):
    if tuple(sorted(snapshots)) != stat.SNAPSHOT_STEPS:
        raise ValueError("SWA requires snapshots at exactly 40,45,...,80")
    keys = set(next(iter(snapshots.values())))
    if not keys or any(set(state) != keys for state in snapshots.values()):
        raise ValueError("adapter snapshot inventories differ")
    for state in snapshots.values():
        for value in state.values():
            arr = np.asarray(value)
            if arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError("snapshots must be finite adapter-only float32 values")


def uniform_swa(snapshots):
    """Uniform elementwise mean of adapter weights at steps 40..80 by fives."""
    validate_snapshots(snapshots)
    steps = stat.SNAPSHOT_STEPS
    out = {}
    for name in snapshots[steps[0]]:
        stacked = np.stack([np.asarray(snapshots[s][name]) for s in steps]).astype(
            np.float64
        )
        out[name] = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    return out


def state_array(value):
    """Return a copied float32 ndarray, including for scalar adapter tensors."""
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("adapter state values must be finite float32")
    return array.copy()


def validate_saved_cell(
    receipt, fold, seed, fit_context_ids, state_path, adapter_names
):
    """Validate and load a fully persisted FIT cell for guarded continuation."""
    if (
        receipt.get("fold") != fold
        or receipt.get("seed") != seed
        or receipt.get("updates") != UPDATES
        or receipt.get("fit_family_count") != 123
        or receipt.get("snapshot_steps") != list(stat.SNAPSHOT_STEPS)
        or receipt.get("ema_decay") != EMA_DECAY
        or sorted(receipt.get("fit_context_ids", [])) != sorted(fit_context_ids)
    ):
        raise RuntimeError("saved H189 FIT cell does not match the current plan")
    expected = {
        f"{arm}::{name}" for arm in ("CTRL80", "SWA", "EMA") for name in adapter_names
    }
    with np.load(state_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError("saved H189 adapter-state inventory mismatch")
        for name in expected:
            array = archive[name]
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise RuntimeError("saved H189 adapter state is not finite float32")
    return Path(state_path)


def ema_step(ema_state, current_state, decay=EMA_DECAY):
    if not 0.0 < decay < 1.0 or set(ema_state) != set(current_state):
        raise ValueError("EMA decay or tensor inventory")
    out = {}
    for name, current in current_state.items():
        e, c = np.asarray(ema_state[name]), np.asarray(current)
        if (
            e.shape != c.shape
            or e.dtype != np.float32
            or c.dtype != np.float32
            or not np.isfinite(e).all()
            or not np.isfinite(c).all()
        ):
            raise ValueError("EMA requires matching float32 adapter tensors")
        out[name] = (decay * e.astype(np.float64) + (1 - decay) * c).astype(np.float32)
    return out


def ema_over_trajectory(states, decay=EMA_DECAY):
    if len(states) != UPDATES or not 0.0 < decay < 1.0:
        raise ValueError("EMA requires exactly 80 post-update states")
    keys = set(states[0])
    if not keys or any(set(s) != keys for s in states):
        raise ValueError("EMA tensor inventories differ")
    ema = {
        name: np.asarray(value, dtype=np.float32).copy()
        for name, value in states[0].items()
    }
    if any(not np.isfinite(value).all() for value in ema.values()):
        raise ValueError("EMA initial state must be finite")
    for state in states[1:]:
        ema = ema_step(ema, state, decay)
    return ema


def validate_no_advance_before_b_lock(guard):
    try:
        guard.require_b_locked()
    except RuntimeError:
        return True
    raise AssertionError("A-stage entry unexpectedly allowed before B lock")


def verdict(b_decisions, a_results):
    if set(b_decisions) != set(stat.ARMS) or set(a_results) != set(stat.ARMS):
        raise ValueError("H189 verdict requires exactly SWA and EMA")
    per_arm = {}
    for arm in stat.ARMS:
        b_admitted = bool(b_decisions[arm]["admitted"])
        a_pass = bool(a_results[arm]["advance_a_gate"])
        per_arm[arm] = {
            "b_admitted": b_admitted,
            "a_gate_passed": a_pass,
            "advance": b_admitted and a_pass,
            "verdict": "ADVANCE_CANDIDATE" if b_admitted and a_pass else "NO_ADVANCE",
        }
    return per_arm
