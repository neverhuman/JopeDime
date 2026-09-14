"""Classify declared stages while retaining missing useful-work proof explicitly."""
from __future__ import annotations

import gpu_occupancy

KINDS = ("train", "propose", "monitor", "gp-eval", "pack", "staging",
         "collective-wait", "checkpoint", "failed", "held", "unexplained")
GPU_TASK_KINDS = ("train", "propose", "monitor")


def classify(propose_active, gp_eval_active, pack_active, held, kernel_mean=None,
             *, train_active=False, monitor_active=False):
    if held:
        kind = "held"
    elif train_active:
        kind = "train"
    elif propose_active:
        kind = "propose"
    elif monitor_active:
        kind = "monitor"
    elif gp_eval_active:
        kind = "gp-eval"
    elif pack_active:
        kind = "pack"
    else:
        kind = "unexplained"
    useful = kind in GPU_TASK_KINDS
    return dict(kind=kind, useful=useful, kernel_mean=kernel_mean,
                classification_basis="declared_stage_only", verified_useful=False)


def committed(outputs):
    """Caller paths and declarations are not original task completion evidence.

    A production verifier must join original funded reservations, observed task
    completion, committed artifact digests and fit-start reconciliation. This
    legacy reporting helper has none of those inputs and cannot certify them.
    """
    return False


def join(history, now, ranks, eligible, kind, outputs, kernel_mean=None):
    if kind not in KINDS:
        raise ValueError("Unknown GPU interval classification")
    useful = kind in GPU_TASK_KINDS and committed(outputs)
    # Occupancy helper still records rank-seconds; acceptance requires outputs.
    snapshot = gpu_occupancy.update(history, now, ranks if useful else None, eligible, kernel_mean, kind)
    snapshot["useful_kind"] = kind
    snapshot["declared_outputs"] = outputs
    snapshot["declared_ranks"] = ranks
    snapshot["verified_useful_gpu_seconds"] = None
    snapshot["committed_outputs"] = False
    snapshot["vram_residency_is_not_occupancy"] = True
    snapshot["acceptance"] = "inconclusive"
    snapshot["missing_evidence"] = [
        "original funded reservation and task identity",
        "independent task/rank interval observation",
        "committed output or optimizer increment verification",
        "fit-start reconciliation and predeclared hour"]
    return snapshot
