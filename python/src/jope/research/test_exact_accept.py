#!/usr/bin/env python3
"""Fail-closed acceptance gate for test-only fused policy checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
from pathlib import Path

from test_exact_common import atomic_json, sha256_file

# Conditional advantage = conditional reward head pairwise accuracy MINUS the
# detached genome-only ablation control. <= 0 means the head is not using the
# dataset embedding at all: it is a genome prior with noise on top. Read in this
# order; the trainer writes `val_conditional_advantage` per metrics row and the
# selected-checkpoint summary carries it forward under the other two names.
CONDITIONAL_ADVANTAGE_KEYS = (
    "val_conditional_advantage",
    "conditional_advantage",
    "selected_phase_b_conditional_advantage",
)
# Margins are in accuracy points, matching the macro_pairwise_lift convention.
CONDITIONAL_INTERNAL_MARGIN = 0.005
CONDITIONAL_MILESTONE_MARGIN = 0.02

# The two independent promotion branches. A generation that trades one against
# the other can never clear a single global gate -- which is the observed state
# (gen-1 ff95a336: binary +0.0421, regression -0.0161, tier "rejected", and the
# original anchor still accepted after every attempt). `task_verdicts` records
# the decision each branch reaches on its OWN evidence. It never widens the
# global verdict: `pass`, `promotion_tier` and `reasons` are computed exactly as
# before, and `task_verdicts` / `task_verdict_schema` / `tasks_passing` are
# purely additive keys.
#
# Keep in lockstep with `jope_selfplay_promotion.rs::task_verdicts` -- that Rust
# port is what the live controller runs; this is the reference implementation
# and the drift check compares the two on the same evidence.
TASK_BRANCHES = ("regression", "binary")
TASK_INTERNAL_LIFT_MARGIN = 0.002
TASK_MILESTONE_LIFT_MARGIN = 0.01
TASK_VERDICT_SCHEMA = "JopeSelfplayCheckpointGateTaskVerdict.v1"
# Aggregate reasons a per-task verdict supersedes when the evidence carries the
# task-scoped measurement. Absent that measurement the aggregate reason is
# inherited unchanged -- an unmeasured branch is fail-closed, never waived.
FAILURE_RATE_REASON = "candidate_failure_rate_regressed"
EQUAL_BUDGET_REASON = "candidate_did_not_win_equal_budget_proposal_gate"


def _task_number(node: dict, field: str, task: str):
    """node[field][task] as a finite float, or None."""
    block = node.get(field)
    if not isinstance(block, dict) or task not in block:
        return None
    value = block[task]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def task_verdicts(
    baseline: dict,
    candidate: dict,
    shared_reasons: list,
    task_own_reasons: dict,
    task_deltas: dict,
    shared_internal_extra: list,
    shared_milestone_extra: list,
) -> dict:
    """The global verdict, re-derived once per branch.

    A branch verdict is the global verdict with (a) the OTHER branch's pairwise
    regression dropped, (b) the macro lift replaced by this branch's own lift,
    and (c) any aggregate reason for which a task-scoped measurement exists
    replaced by that measurement's own reason. Everything else -- replay
    verification, conditionality, proposal validity, latency -- is a property of
    the checkpoint, not of a branch, and is inherited verbatim.
    """
    verdicts = {}
    for task in TASK_BRANCHES:
        delta = task_deltas.get(task)
        own = list(task_own_reasons.get(task) or [])

        superseded = set()
        task_specific = []
        failure_delta = None
        candidate_failure = _task_number(candidate, "failure_rate_by_task", task)
        baseline_failure = _task_number(baseline, "failure_rate_by_task", task)
        if candidate_failure is not None and baseline_failure is not None:
            superseded.add(FAILURE_RATE_REASON)
            failure_delta = candidate_failure - baseline_failure
            if failure_delta > 0.0025 + 1e-12:
                task_specific.append(f"{task}_failure_rate_regressed")
        equal_budget = None
        candidate_equal = _task_number(candidate, "equal_budget_result_by_task", task)
        baseline_equal = _task_number(baseline, "equal_budget_result_by_task", task)
        if candidate_equal is not None and baseline_equal is not None:
            superseded.add(EQUAL_BUDGET_REASON)
            equal_budget = {"candidate": candidate_equal, "baseline": baseline_equal}
            if candidate_equal <= baseline_equal + 1e-12:
                task_specific.append(f"{task}_did_not_win_equal_budget_proposal_gate")

        inherited = [reason for reason in shared_reasons if reason not in superseded]
        internal_reasons = own + task_specific + inherited
        milestone_reasons = list(internal_reasons)
        finite = delta is not None and math.isfinite(delta)
        if not (finite and delta >= TASK_INTERNAL_LIFT_MARGIN - 1e-12):
            internal_reasons.append(f"{task}_pairwise_lift_below_point_two")
        if not (finite and delta >= TASK_MILESTONE_LIFT_MARGIN - 1e-12):
            milestone_reasons.append(f"{task}_pairwise_lift_below_one_point")
        internal_reasons += shared_internal_extra
        milestone_reasons += shared_milestone_extra

        internal_pass = not internal_reasons
        milestone_pass = not milestone_reasons
        tier = "milestone" if milestone_pass else "internal" if internal_pass else "rejected"
        # True when nothing this branch controls is blocking it: every residual
        # reason is a property of the checkpoint as a whole. "binary was held
        # back only by task-agnostic evidence" is a different fact from "binary
        # lost on its own merits", and the operator needs both.
        branch_owned = sorted(
            set(own)
            | set(task_specific)
            | {r for r in internal_reasons if r.startswith(f"{task}_")}
        )
        verdicts[task] = {
            "pass": internal_pass,
            "internal_learning_pass": internal_pass,
            "milestone_pass": milestone_pass,
            "promotion_tier": tier,
            "reasons": [] if internal_pass else internal_reasons,
            "milestone_reasons": milestone_reasons,
            "pairwise_lift": delta if finite else None,
            "candidate_pairwise_accuracy": candidate.get(f"{task}_pairwise_accuracy"),
            "baseline_pairwise_accuracy": baseline.get(f"{task}_pairwise_accuracy"),
            "failure_rate_delta": failure_delta,
            "equal_budget_result": equal_budget,
            "blocked_only_by_shared_evidence": not internal_pass and not branch_owned,
            "blocking_reasons_owned_by_this_task": branch_owned,
        }
    return verdicts


def conditional_advantage(node: dict) -> float | None:
    """First present, finite, numeric conditional-advantage reading, else None.

    None is never treated as zero: an unmeasured checkpoint is fail-closed, the
    same way a missing replay verification is.
    """

    for key in CONDITIONAL_ADVANTAGE_KEYS:
        if key not in node:
            continue
        value = node[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def evaluate_gate(evidence: dict) -> dict:
    baseline = evidence["baseline"]
    candidate = evidence["candidate"]
    reasons = []
    macro_lift = float(candidate["macro_pairwise_accuracy"]) - float(
        baseline["macro_pairwise_accuracy"]
    )
    common_reasons = []
    task_deltas = {}
    # Branch bookkeeping for `task_verdicts`. These reasons still enter
    # `common_reasons` in the original order, so the global verdict is unchanged.
    task_own_reasons: dict[str, list[str]] = {}
    for task in TASK_BRANCHES:
        delta = float(candidate[f"{task}_pairwise_accuracy"]) - float(
            baseline[f"{task}_pairwise_accuracy"]
        )
        task_deltas[task] = delta
        if delta < -0.005 - 1e-12:
            reason = f"{task}_regressed_more_than_half_point"
            task_own_reasons.setdefault(task, []).append(reason)
            common_reasons.append(reason)
    if float(evidence["proposal_validity"]) < 0.99:
        common_reasons.append("proposal_validity_below_99_percent")
    if int(evidence.get("verified_checkpoint_origin_strict_wins") or 0) < 1:
        common_reasons.append("no_verified_checkpoint_origin_strict_test_improvement")
    candidate_equal_budget = float(
        evidence.get(
            "candidate_equal_budget_result",
            candidate.get("equal_budget_proposal_result", float("-inf")),
        )
    )
    baseline_equal_budget = float(
        evidence.get(
            "baseline_equal_budget_result",
            baseline.get("equal_budget_proposal_result", float("inf")),
        )
    )
    if candidate_equal_budget <= baseline_equal_budget + 1e-12:
        common_reasons.append("candidate_did_not_win_equal_budget_proposal_gate")
    candidate_failure = float(candidate.get("failure_rate", 1.0))
    baseline_failure = float(baseline.get("failure_rate", 0.0))
    failure_delta = candidate_failure - baseline_failure
    if failure_delta > 0.0025 + 1e-12:
        common_reasons.append("candidate_failure_rate_regressed")
    candidate_latency = float(candidate.get("p95_proposal_latency_ms", float("inf")))
    baseline_latency = float(baseline.get("p95_proposal_latency_ms", 0.0))
    latency_ratio = (
        float("inf") if baseline_latency <= 0.0 else candidate_latency / baseline_latency
    )
    latency_exception = evidence.get("latency_exception") or {}
    exception_valid = (
        latency_exception.get("approved") is True
        and float(latency_exception.get("accuracy_gain") or 0.0)
        > float(latency_exception.get("latency_penalty") or float("inf"))
        and str(latency_exception.get("rationale") or "").strip() != ""
    )
    if latency_ratio > 1.25 + 1e-12 and not exception_valid:
        common_reasons.append("candidate_p95_latency_exceeds_limit")
    replay = evidence.get("replay_verification") or {}
    replay_checks = (
        "predictions",
        "scores",
        "evaluator_identity",
        "dataset_hashes",
        "candidate_origin",
    )
    for check in replay_checks:
        if replay.get(check) is not True:
            common_reasons.append(f"replay_verification_failed:{check}")
    # Conditionality gate. A checkpoint whose conditional reward head cannot beat
    # its own genome-only ablation is not conditioned on the dataset, so it can
    # never be promoted — no macro lift buys past this, at either tier.
    candidate_conditional = conditional_advantage(candidate)
    baseline_conditional = conditional_advantage(baseline)
    conditional_lift = (
        None
        if candidate_conditional is None or baseline_conditional is None
        else candidate_conditional - baseline_conditional
    )
    if candidate_conditional is None:
        common_reasons.append("candidate_conditional_advantage_evidence_missing")
    if baseline_conditional is None:
        common_reasons.append("baseline_conditional_advantage_evidence_missing")
    if candidate_conditional is not None and candidate_conditional <= 1e-12:
        common_reasons.append("candidate_policy_is_not_conditional")
    internal_reasons = list(common_reasons)
    milestone_reasons = list(common_reasons)
    if macro_lift < 0.002 - 1e-12:
        internal_reasons.append("macro_pairwise_lift_below_point_two")
    if macro_lift < 0.01 - 1e-12:
        milestone_reasons.append("macro_pairwise_lift_below_one_point")
    if (
        candidate_conditional is not None
        and candidate_conditional < CONDITIONAL_INTERNAL_MARGIN - 1e-12
    ):
        internal_reasons.append("conditional_advantage_below_half_point")
    if conditional_lift is not None and conditional_lift < CONDITIONAL_INTERNAL_MARGIN - 1e-12:
        internal_reasons.append("conditional_advantage_lift_below_half_point")
    if (
        candidate_conditional is not None
        and candidate_conditional < CONDITIONAL_MILESTONE_MARGIN - 1e-12
    ):
        milestone_reasons.append("conditional_advantage_below_two_points")
    if conditional_lift is not None and conditional_lift < CONDITIONAL_MILESTONE_MARGIN - 1e-12:
        milestone_reasons.append("conditional_advantage_lift_below_two_points")
    internal_pass = not internal_reasons
    milestone_pass = not milestone_reasons
    reasons = [] if internal_pass else internal_reasons
    tier = "milestone" if milestone_pass else "internal" if internal_pass else "rejected"
    # Per-task verdicts. Derived from exactly the reasons the global verdict
    # already computed, minus the branch attribution: nothing here can make the
    # global gate laxer, and everything here is additive in the artifact.
    branch_owned_reasons = {r for reasons_ in task_own_reasons.values() for r in reasons_}
    shared_reasons = [r for r in common_reasons if r not in branch_owned_reasons]
    shared_internal_extra = []
    shared_milestone_extra = []
    if (
        candidate_conditional is not None
        and candidate_conditional < CONDITIONAL_INTERNAL_MARGIN - 1e-12
    ):
        shared_internal_extra.append("conditional_advantage_below_half_point")
    if conditional_lift is not None and conditional_lift < CONDITIONAL_INTERNAL_MARGIN - 1e-12:
        shared_internal_extra.append("conditional_advantage_lift_below_half_point")
    if (
        candidate_conditional is not None
        and candidate_conditional < CONDITIONAL_MILESTONE_MARGIN - 1e-12
    ):
        shared_milestone_extra.append("conditional_advantage_below_two_points")
    if conditional_lift is not None and conditional_lift < CONDITIONAL_MILESTONE_MARGIN - 1e-12:
        shared_milestone_extra.append("conditional_advantage_lift_below_two_points")
    verdicts = task_verdicts(
        baseline,
        candidate,
        shared_reasons,
        task_own_reasons,
        task_deltas,
        shared_internal_extra,
        shared_milestone_extra,
    )
    return {
        "schema": "JopeSelfplayCheckpointGate.v3",
        "pass": internal_pass,
        "internal_learning_pass": internal_pass,
        "milestone_pass": milestone_pass,
        "promotion_tier": tier,
        "reasons": reasons,
        "milestone_reasons": milestone_reasons,
        "macro_pairwise_lift": macro_lift,
        "candidate_conditional_advantage": candidate_conditional,
        "baseline_conditional_advantage": baseline_conditional,
        "conditional_advantage_lift": conditional_lift,
        "task_pairwise_deltas": task_deltas,
        "failure_rate_delta": failure_delta,
        "p95_latency_ratio": latency_ratio,
        "latency_exception_applied": latency_ratio > 1.25 and exception_valid,
        "model_lift_claim_publishable": False,
        "promotion_scope": "internal-selected-test-flywheel-only",
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "accepted_ancestor_sha256": baseline["checkpoint_sha256"],
        "candidate_equal_budget_result": candidate_equal_budget,
        "baseline_equal_budget_result": baseline_equal_budget,
        "task_verdicts": verdicts,
        "task_verdict_schema": TASK_VERDICT_SCHEMA,
        "tasks_passing": [task for task in TASK_BRANCHES if verdicts[task]["pass"]],
    }


def promote(
    *,
    evidence_path: Path,
    gate_path: Path,
    accepted_pointer: Path,
    immutable_root: Path,
    enforce_pointer_ancestor: bool = True,
) -> dict:
    evidence = json.loads(evidence_path.read_text())
    result = evaluate_gate(evidence)
    atomic_json(gate_path, result)
    if not result["pass"]:
        return result
    checkpoint = Path(evidence["candidate"]["checkpoint_path"])
    if sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("accepted checkpoint content hash mismatch")
    immutable_checkpoint = (
        immutable_root
        / "checkpoints"
        / f"{result['checkpoint_sha256']}.model.pt"
    )
    immutable_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not immutable_checkpoint.exists():
        temporary = immutable_checkpoint.with_name(
            f".{immutable_checkpoint.name}.tmp-{os.getpid()}"
        )
        shutil.copy2(checkpoint, temporary)
        if sha256_file(temporary) != result["checkpoint_sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("checkpoint changed while copying to immutable storage")
        os.replace(temporary, immutable_checkpoint)
    elif sha256_file(immutable_checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("immutable checkpoint path contains different bytes")

    evidence_hash = sha256_file(evidence_path)
    immutable_evidence = immutable_root / "gates" / f"{evidence_hash}.evidence.json"
    immutable_evidence.parent.mkdir(parents=True, exist_ok=True)
    if not immutable_evidence.exists():
        shutil.copy2(evidence_path, immutable_evidence)
    if sha256_file(immutable_evidence) != evidence_hash:
        raise RuntimeError("immutable evidence copy hash mismatch")
    gate_hash = sha256_file(gate_path)
    immutable_gate = immutable_root / "gates" / f"{gate_hash}.gate.json"
    if not immutable_gate.exists():
        shutil.copy2(gate_path, immutable_gate)
    if sha256_file(immutable_gate) != gate_hash:
        raise RuntimeError("immutable gate copy hash mismatch")

    lock_path = accepted_pointer.with_suffix(f"{accepted_pointer.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if accepted_pointer.is_file():
            current = json.loads(accepted_pointer.read_text())
            if current["checkpoint_sha256"] == result["checkpoint_sha256"]:
                return result
            if (
                enforce_pointer_ancestor
                and current["checkpoint_sha256"]
                != result["accepted_ancestor_sha256"]
            ):
                raise RuntimeError("accepted ancestor changed before pointer promotion")
        atomic_json(
            accepted_pointer,
            {
                "schema": "JopeSelfplayAcceptedCheckpoint.v2",
                "checkpoint": str(immutable_checkpoint),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "accepted_ancestor_sha256": result["accepted_ancestor_sha256"],
                "gate": str(immutable_gate),
                "gate_sha256": gate_hash,
                "evidence": str(immutable_evidence),
                "evidence_sha256": evidence_hash,
                "test_adaptive": True,
                "benchmark_valid": False,
                "selection_bias": "test-exact-reused",
                "promotion_scope": "internal-selected-test-flywheel-only",
            },
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accepted-pointer",
        type=Path,
        default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0/training/accepted.json"),
    )
    parser.add_argument(
        "--immutable-root",
        type=Path,
        default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0/training/immutable"),
    )
    args = parser.parse_args()
    result = promote(
        evidence_path=args.evidence,
        gate_path=args.output,
        accepted_pointer=args.accepted_pointer,
        immutable_root=args.immutable_root,
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
