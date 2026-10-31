#!/usr/bin/env python3
"""Check pinned W5 identity relations; prepare a W9 receipt panel on refusal.

Offline metadata only. Does not fetch S3 references or read any numeric payload.
A relation PASS is conditional on independently admitted source/authority pins;
this checker never grants W5/W9 execution or authenticates a self-declaration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash64(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_ref(value):
    """Require immutable S3 identity, not an ETag or a local filename."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("s3_uri"), str)
        and re.fullmatch(r"s3://[^/\s]+/[^\s]+", value["s3_uri"]) is not None
        and isinstance(value.get("version_id"), str)
        and value["version_id"] not in ("", "null")
        and hash64(value.get("sha256"))
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_pinned(path, expected):
    raw = Path(path).read_bytes()
    if not hash64(expected) or digest(raw) != expected:
        raise ValueError("input byte SHA-256 mismatch")
    return json.loads(raw, object_pairs_hook=unique_object)


def check_w5(packet):
    if not isinstance(packet, dict):
        raise ValueError("packet must be an object")
    rows = packet.get("children")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty complete candidate-child census required")
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError("child rows must be objects")
    ids = [(r.get("context_sha256"), r.get("child_sha256")) for r in rows]
    if any(not hash64(v) for pair in ids for v in pair) or len(set(ids)) != len(ids):
        raise ValueError("duplicate or malformed child identity")
    family_values = [r.get("family_id") for r in rows]
    if any(not isinstance(f, str) or not f for f in family_values):
        raise ValueError("missing candidate family")
    families = set(family_values)
    sealed = packet.get("sealed_family_manifest")
    sealed_ok = (
        isinstance(sealed, dict)
        and sealed.get("scope") == "global_sealed_meta_test"
        and sealed.get("complete") is True
        and source_ref(sealed.get("source"))
        and source_ref(sealed.get("independent_authority_receipt"))
        and isinstance(sealed.get("family_ids"), list)
        and bool(sealed["family_ids"])
        and all(isinstance(x, str) and x for x in sealed["family_ids"])
        and len(set(sealed["family_ids"])) == len(sealed["family_ids"])
        and hash64(sealed.get("family_alias_registry_sha256"))
        and sealed["family_alias_registry_sha256"]
        == packet.get("family_alias_registry_sha256")
    )
    overlap = len(families & set(sealed["family_ids"])) if sealed_ok else None
    pools = packet.get("pool_membership", {})
    if not isinstance(pools, dict):
        raise ValueError("pool membership must be an object")
    lineage_missing = lineage_invalid = source_missing = source_invalid = 0
    for row in rows:
        parent = row.get("parent_sha256")
        pool = pools.get(row["context_sha256"])
        if parent is None or pool is None or not row.get("operator"):
            lineage_missing += 1
        elif not (
            isinstance(pool, dict)
            and isinstance(row.get("operator"), str)
            and hash64(parent)
            and parent != row["child_sha256"]
            and hash64(row.get("pool_sha256"))
            and row["pool_sha256"] == pool.get("pool_sha256")
            and row.get("pool_membership_receipt_sha256") == pool.get("receipt_sha256")
            and hash64(pool.get("receipt_sha256"))
            and isinstance(pool.get("program_sha256"), list)
            and all(hash64(v) for v in pool["program_sha256"])
            and parent in pool["program_sha256"]
            and row["child_sha256"] in pool["program_sha256"]
            and source_ref(row.get("lineage_source"))
        ):
            lineage_invalid += 1
        origin = row.get("original_row_source")
        if origin is None:
            source_missing += 1
        elif not (
            source_ref(origin)
            and isinstance(origin.get("source_row_id"), str)
            and bool(origin["source_row_id"])
            and origin.get("context_sha256") == row["context_sha256"]
            and origin.get("child_sha256") == row["child_sha256"]
            and hash64(origin.get("original_training_row_ledger_sha256"))
            and hash64(origin.get("verified_byte_receipt_sha256"))
        ):
            source_invalid += 1
    checks = {
        "global_sealed_family_exclusion": {
            "status": "PASS_CHECKED_RELATION" if sealed_ok and overlap == 0 else "FAIL",
            "overlap_count": overlap,
            "missing_artifact": None
            if sealed_ok
            else "Complete authoritative identity-only sealed-family manifest, independent authority receipt, and matching alias-registry pin.",
        },
        "parent_child_lineage": {
            "status": "PASS_CHECKED_RELATION"
            if not lineage_missing + lineage_invalid
            else "FAIL",
            "missing_rows": lineage_missing,
            "invalid_rows": lineage_invalid,
            "required_artifact": "Exact context/child/parent/operator ledger bound to the actual pool-membership byte receipt and immutable lineage source.",
        },
        "original_row_source": {
            "status": "PASS_CHECKED_RELATION"
            if not source_missing + source_invalid
            else "FAIL",
            "missing_rows": source_missing,
            "invalid_rows": source_invalid,
            "required_artifact": "Per-child original source-row locator, S3 key/version/byte SHA, verified byte receipt and original-training-row ledger.",
        },
    }
    passed = all(c["status"] == "PASS_CHECKED_RELATION" for c in checks.values())
    return {
        "schema": "jope.h101.w5-relations/v1",
        "verdict": "PASS_CHECKED_RELATIONS" if passed else "FAIL",
        "children_checked": len(rows),
        "contexts": len({r["context_sha256"] for r in rows}),
        "families": len(families),
        "checks": checks,
        "execution_admitted": False,
        "scope": "Pinned metadata relations only; independent authenticity/completeness, completion/evaluator/cut and H98 role-reconciliation gates remain required.",
        "next_action": "independent_admission_review"
        if passed
        else "W9_CPU_PREPARATION",
    }


def prepare_w9(protocol, protocol_sha256):
    """Expand H91's ten replay types into the exact H75 fold/seed receipt slots."""
    if (protocol.get("folds"), protocol.get("seeds")) != (5, [701, 702, 703]):
        raise ValueError("unexpected frozen fold/seed contract")
    if not hash64(protocol_sha256):
        raise ValueError("protocol pin required")
    budget = protocol["budgets"]
    if budget.get("reservation") != "existing_W9_envelope_not_additive":
        raise ValueError("W9 budget reservation must not reset")
    memories = ("A", "B", "C", "D", "uniform_fit_family")
    learners = ("neural", "strongest_W1_tree")
    slots = [
        {
            "slot_id": f"fold{fold}/seed{seed}/{memory}/{learner}",
            "held_fold": fold,
            "fit_folds": [i for i in range(5) if i != fold],
            "seed": seed,
            "memory": memory,
            "learner": learner,
            "receipt_sha256": None,
        }
        for fold in range(5)
        for seed in protocol["seeds"]
        for memory in memories
        for learner in learners
    ]
    return {
        "schema": "jope.h101.w9-cpu-receipt-plan/v1",
        "status": "CPU_PREPARATION_ONLY",
        "protocol_sha256": protocol_sha256,
        "slots": slots,
        "required_receipt_fields": [
            "slot_id",
            "fit_family_roster_sha256",
            "memory_source_key_version_sha256",
            "completed_input_manifest_sha256",
            "learner_recipe_sha256",
            "initialization_sha256",
            "exposure_receipt_sha256",
            "cost_ledger_sha256",
            "terminal_parameters_sha256",
            "endpoint_identity",
        ],
        "uniform_memory_source": "fit-family acquisition only; never terminal R held-context outcomes",
        "original_budget": budget,
        "actual_consumption": None,
        "remaining_authorized_budget": None,
        "strongest_W1_recipe_sha256": None,
        "completed_replay_receipts": 0,
        "pending_replay_receipts": len(slots),
        "protocol_amended": False,
        "execution_admitted": False,
        "metrics": {"Elo": None, "U4": None, "val_reg_inner": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--w9-protocol", type=Path, required=True)
    parser.add_argument("--w9-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_w5(load_pinned(args.input, args.sha256))
    protocol = load_pinned(args.w9_protocol, args.w9_protocol_sha256)
    plan = (
        prepare_w9(protocol, args.w9_protocol_sha256)
        if result["verdict"] == "FAIL"
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if result["verdict"] == "FAIL":
        (args.output / "w9_cpu_prep.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, sort_keys=True))
    return 1 if result["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
