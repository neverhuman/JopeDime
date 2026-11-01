"""Build a non-launchable RNK-10 packet shell from finalized H-wave metadata.

This module handles JSON metadata only. It never loads a checkpoint, dataset,
pool, outcome, S3 object, or evaluator. The result is a review artifact, not an
authorization to score outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
DOPE_C51 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_ID = "r882"


class PacketError(ValueError):
    """Input metadata is not eligible for a one-shot packet shell."""


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PacketError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PacketError(f"{name} must be nonempty text")
    return value


def _require_zero(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise PacketError(f"{name} must be exactly zero")


def validate_candidate(result: dict[str, Any]) -> dict[str, Any]:
    """Validate a publisher-normalized, final H155/H156/H157 result record."""
    if result.get("schema") != "jope.rnk10.hwave-candidate-result/v1":
        raise PacketError("unsupported H-wave candidate result schema")
    wave = result.get("wave")
    if wave not in {"H155", "H156", "H157"}:
        raise PacketError("wave must be H155, H156 or H157")
    if result.get("status") != "COMPLETE" or result.get("audit_status") != "PASS":
        raise PacketError(
            "candidate must have a complete result and passing independent audit"
        )
    if result.get("one_shot") is not True:
        raise PacketError(
            "candidate result must attest to its preregistered single attempt"
        )
    if result.get("retry_count") != 0:
        raise PacketError("candidate result must have zero retries")

    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise PacketError("protocol pins are required")
    if protocol.get("selection_split") != "DL-Train-inner":
        raise PacketError(
            "candidate selection must use the preregistered DL-Train inner split"
        )
    if protocol.get("selection_metric") != "family_grouped_elo_vs_r882":
        raise PacketError("selection must use family-grouped Elo versus r882")
    if protocol.get("speed_gate") not in (None, False):
        raise PacketError("speed gates are prohibited")
    if protocol.get("dope_sha256") != DOPE_C51:
        raise PacketError("DOPE identity differs from frozen c51a56d0")
    if protocol.get("r882_id") != R882_ID:
        raise PacketError("reference must be frozen r882")

    result_sha = _require_sha(result.get("result_sha256"), "result_sha256")
    audit_sha = _require_sha(result.get("audit_sha256"), "audit_sha256")
    protocol_sha = _require_sha(protocol.get("sha256"), "protocol.sha256")

    access = result.get("access_audit")
    if not isinstance(access, dict):
        raise PacketError("access audit is required")
    for label in (
        "sealed_reads",
        "holdout_reads",
        "outer_test_reads",
        "meta_test_reads",
    ):
        _require_zero(access.get(label), f"access_audit.{label}")

    selection = result.get("frozen_selection")
    if not isinstance(selection, dict) or selection.get("status") != "FROZEN_WINNER":
        raise PacketError(
            "candidate must be the independently audited frozen-rule winner"
        )
    if selection.get("rank") != 1:
        raise PacketError(
            "only rank-1 candidate from the frozen selection rule is eligible"
        )
    if selection.get("rule") != "highest_elo_then_wave_then_candidate_id":
        raise PacketError("frozen candidate tie-break rule is missing or changed")
    if (
        not isinstance(selection.get("candidate_count"), int)
        or isinstance(selection["candidate_count"], bool)
        or selection["candidate_count"] < 1
    ):
        raise PacketError("frozen selection candidate_count must be positive")
    if selection.get("audit_status") != "PASS":
        raise PacketError("frozen selection inventory must pass independent audit")
    selection_inventory_sha = _require_sha(
        selection.get("considered_results_sha256"),
        "frozen_selection.considered_results_sha256",
    )
    selection_rule_sha = _require_sha(
        selection.get("rule_sha256"), "frozen_selection.rule_sha256"
    )

    candidate = result.get("candidate")
    if not isinstance(candidate, dict):
        raise PacketError("candidate identity is required")
    candidate_id = _require_text(candidate.get("id"), "candidate.id")
    checkpoint_sha = _require_sha(
        candidate.get("checkpoint_sha256"), "candidate.checkpoint_sha256"
    )
    source_sha = _require_sha(candidate.get("source_sha256"), "candidate.source_sha256")
    if (
        not isinstance(candidate.get("optimizer_updates"), int)
        or isinstance(candidate["optimizer_updates"], bool)
        or candidate["optimizer_updates"] <= 0
    ):
        raise PacketError("candidate must attest to at least one optimizer update")
    if candidate.get("dope_sha256") != DOPE_C51:
        raise PacketError("candidate DOPE identity differs from frozen c51a56d0")

    reference_sha = _require_sha(
        protocol.get("r882_checkpoint_sha256"), "protocol.r882_checkpoint_sha256"
    )
    metrics = result.get("selection_metrics")
    if not isinstance(metrics, dict):
        raise PacketError("selection metrics are required")
    elo = metrics.get("family_grouped_elo_vs_r882")
    if (
        not isinstance(elo, (int, float))
        or isinstance(elo, bool)
        or not math.isfinite(elo)
        or elo <= 1500
    ):
        raise PacketError(
            "candidate must strictly beat r882 on the preregistered Elo metric"
        )
    if metrics.get("speed_used_for_selection") not in (None, False):
        raise PacketError("speed may not affect candidate selection")

    return {
        "wave": wave,
        "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha,
        "source_sha256": source_sha,
        "optimizer_updates": candidate["optimizer_updates"],
        "candidate_dope_sha256": candidate["dope_sha256"],
        "r882_checkpoint_sha256": reference_sha,
        "family_grouped_elo_vs_r882": float(elo),
        "result_sha256": result_sha,
        "audit_sha256": audit_sha,
        "protocol_sha256": protocol_sha,
        "access_audit": {
            label: 0
            for label in (
                "sealed_reads",
                "holdout_reads",
                "outer_test_reads",
                "meta_test_reads",
            )
        },
        "selection_inventory_sha256": selection_inventory_sha,
        "selection_rule_sha256": selection_rule_sha,
        "selection_candidate_count": selection["candidate_count"],
    }


def build_packet(
    result: dict[str, Any],
    *,
    evaluator_source_sha256: str,
    roster_sha256: str,
    evaluation_identity_sha256: str,
    max_unique_outcome_reads: int,
) -> dict[str, Any]:
    """Create a fixed-input packet shell; launch stays explicitly unauthorized."""
    candidate = validate_candidate(result)
    evaluator_sha = _require_sha(evaluator_source_sha256, "evaluator_source_sha256")
    roster_sha = _require_sha(roster_sha256, "roster_sha256")
    identity_sha = _require_sha(
        evaluation_identity_sha256, "evaluation_identity_sha256"
    )
    if (
        not isinstance(max_unique_outcome_reads, int)
        or isinstance(max_unique_outcome_reads, bool)
        or max_unique_outcome_reads <= 0
    ):
        raise PacketError("max_unique_outcome_reads must be a positive integer")

    packet = {
        "schema": "jope.rnk10.hwave-one-shot-packet/v1",
        "status": "READY_FOR_INDEPENDENT_REVIEW",
        "candidate_selection": {
            "rule": "highest eligible preregistered family-grouped Elo vs r882; lexical wave then candidate id breaks exact ties",
            "selection_split": "DL-Train-inner",
            "speed_gate": False,
            **candidate,
        },
        "evaluation": {
            "evaluator_source_sha256": evaluator_sha,
            "roster_sha256": roster_sha,
            "evaluation_identity_sha256": identity_sha,
            "max_unique_outcome_reads": max_unique_outcome_reads,
            "ledger_state": "UNUSED",
            "consumed_reads": 0,
            "retry_count": 0,
        },
        "protected_data": {
            "sealed": "EXCLUDED",
            "holdout": "EXCLUDED",
            "outer_test": "EXCLUDED",
            "meta_test": "EXCLUDED",
        },
        "launch_authorized": False,
        "required_before_launch": [
            "exact-head independent HWV review",
            "qualified pinned evaluator and frozen eligible roster",
            "owner-authorized single-use outcome ledger",
            "fresh resource and host-capacity check",
        ],
        "nonclaims": [
            "packet preparation is not outcome evaluation or qualification",
            "H-wave inner selection is not fresh acceptance evidence",
            "no production, serving, or default enablement is implied",
        ],
    }
    return packet


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PacketError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="final normalized H-wave result JSON")
    parser.add_argument("output", type=Path, help="destination packet JSON")
    parser.add_argument("--evaluator-source-sha256", required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--evaluation-identity-sha256", required=True)
    parser.add_argument("--max-unique-outcome-reads", required=True, type=int)
    args = parser.parse_args()

    packet = build_packet(
        _load_json(args.result),
        evaluator_source_sha256=args.evaluator_source_sha256,
        roster_sha256=args.roster_sha256,
        evaluation_identity_sha256=args.evaluation_identity_sha256,
        max_unique_outcome_reads=args.max_unique_outcome_reads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(packet, sort_keys=True, indent=2) + "\n").encode()
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "packet_sha256": hashlib.sha256(encoded).hexdigest(),
                "launch_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
