#!/usr/bin/env python3
"""W5 admission preflight. No model, table, pool or outcome-body reader."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/flywheel-selfplay-20261023/"
PINS = {
    "round-01/persistence_receipt.json": (
        "75H4Wf31_uKWdyh39Ap9iY_yBvfW1KP5",
        "4cdc11dcf994b105614055237afda2847310190dadd289ba79f2f34951683128",
    ),
    "round-02/persistence_receipt.json": (
        "UyMRjyqasrO3DKM0viJVr7Cc5Si_iDOz",
        "563ecdad1f2d30288f47370a8c05dde93c9bb4dc6abd0806667f9246b4e0d72e",
    ),
    "round-03/persistence_receipt.json": (
        "5e06unUgEx6YBaC1bABOLmMujxBWfI7N",
        "ef5e1ace35b5ab298e259dbae4117493b8b7847703902a76fe916726550d3de0",
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_counter(client, key, version):
    """Fail before GET unless this exact key/version is explicitly admitted."""
    relative = key.removeprefix(PREFIX)
    if key != PREFIX + relative or relative not in PINS:
        raise ValueError("body not admitted")
    expected_version, expected_sha = PINS[relative]
    if version != expected_version:
        raise ValueError("version not admitted")
    response = client.get_object(Bucket=BUCKET, Key=key, VersionId=version)
    with response["Body"] as stream:
        body = stream.read(807)
    if response["VersionId"] != version or len(body) != 806 or sha(body) != expected_sha:
        raise ValueError("counter bytes differ from admission pin")
    return {
        "key": key, "version": response["VersionId"], "size": len(body),
        "sha256": sha(body), "body_utf8": body.decode(), "use": "retention_only",
    }


def row_errors(ledger):
    """Check metadata only; never reconstruct row IDs from a numeric table."""
    if ledger is None:
        return ["original_training_and_strat300_row_ledger_missing"]
    errors = []
    for field in ("context_sha256", "table_sha256", "cut_id", "family_alias_component"):
        if not ledger.get(field) or ledger.get("screen_" + field) != ledger[field]:
            errors.append("screen_identity_mismatch:" + field)
    original, screen = ledger.get("original_training_row_ids"), ledger.get("screen_row_ids")
    if not isinstance(original, list) or not isinstance(screen, list) or not original or not screen:
        errors.append("row_ids_missing")
    elif any(type(x) is not str or not x for x in original + screen):
        errors.append("row_ids_invalid")
    elif len(set(original)) != len(original) or len(set(screen)) != len(screen):
        errors.append("row_ids_duplicated")
    elif not set(screen).issubset(original):
        errors.append("screen_outside_target_original_training_rows")
    if ledger.get("source_context_sha256") != ledger.get("context_sha256"):
        errors.append("sibling_or_unknown_source_context")
    if ledger.get("role") != "DL-Train" or not ledger.get("current_role_alias_manifest_sha256"):
        errors.append("current_training_role_alias_proof_missing")
    return errors


def pair_errors(pair):
    """Exact joins and outcome provenance, not an admission authority or fitter."""
    if pair is None:
        return ["admitted_parent_operator_child_export_missing"]
    errors = []
    parent, child = pair.get("parent", {}), pair.get("child", {})
    if not parent.get("program_sha256") or not child.get("program_sha256"):
        errors.append("program_identity_missing")
    if (not pair.get("operator") or child.get("operator") != pair["operator"]
            or child.get("parent_ids") != [parent.get("program_sha256")]):
        errors.append("exact_parent_operator_join_missing")
    for field in ("context_sha256", "table_sha256", "cut_id", "evaluator", "metric", "gate", "vocabulary"):
        if not parent.get(field) or parent.get(field) != child.get(field):
            errors.append("pair_identity_mismatch:" + field)
    for name, row in (("parent", parent), ("child", child)):
        if row.get("status") != "completed" or row.get("evaluation_kind") != "full":
            errors.append(name + ":not_completed_full_evaluation")
        # A finite fitness=100/logratio=0 can be a failure sentinel.
        if row.get("outcome_origin") != "measured" or type(row.get("gate_passed")) is not bool:
            errors.append(name + ":measured_outcome_unproven")
        for field in ("source_version", "source_sha256", "collection_policy", "original_outcome_sha256"):
            if not row.get(field):
                errors.append(name + ":missing:" + field)
        if not row.get("outcome_sha256") or row.get("outcome_sha256") != row.get("original_outcome_sha256"):
            errors.append(name + ":unrounded_outcome_identity_unproven")
    return errors


def collect(profile):
    import boto3  # Collection only; offline verification and unit controls need stdlib.

    start = time.monotonic()
    client = boto3.Session(profile_name=profile).client("s3")
    result = {
        "schema": "jope.w5.metadata-inventory/v1", "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "profile": profile, "bucket": BUCKET, "prefix": PREFIX,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "verifier_sha256": sha(Path(__file__).read_bytes()), "objects": [], "counters": [],
        "listing_complete": False, "errors": [], "metadata_calls": 0,
        "outcome_table_pool_checkpoint_body_gets": 0,
    }
    try:
        token = None
        for _ in range(5):  # Bounded census; truncation is explicitly not completeness.
            args = {"Bucket": BUCKET, "Prefix": PREFIX, "MaxKeys": 1000}
            if token:
                args["ContinuationToken"] = token
            page = client.list_objects_v2(**args)
            result["metadata_calls"] += 1
            result["objects"].extend({
                "key": obj["Key"], "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            } for obj in page.get("Contents", []))
            if not page.get("IsTruncated"):
                result["listing_complete"] = True
                break
            token = page["NextContinuationToken"]
        for relative, (version, _) in PINS.items():
            result["counters"].append(read_counter(client, PREFIX + relative, version))
        # HEAD does not admit a body; retain actual versions of the mixed sources.
        for relative in [
            *(f"round-{i:02}/scores/retained_scores_round_{i:02}.jsonl" for i in (1, 2, 3)),
            "corpus_v2_train_winner_integrated/corpus_v2_manifest.json",
            "corpus_v3_wider_strict_train_256/corpus_v3_manifest.json",
        ]:
            head = client.head_object(Bucket=BUCKET, Key=PREFIX + relative)
            result["metadata_calls"] += 1
            result.setdefault("heads", []).append({
                "key": PREFIX + relative, "version": head.get("VersionId"),
                "size": head["ContentLength"], "body_admitted": False,
            })
    except Exception as exc:
        # Preserve partial evidence and fail; no latest-version or other-prefix fallback.
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return result


def verify(inventory):
    errors = list(inventory.get("errors", []))
    if inventory.get("bucket") != BUCKET or inventory.get("prefix") != PREFIX:
        errors.append("inventory_outside_approved_prefix")
    if not inventory.get("listing_complete"):
        errors.append("metadata_census_incomplete")
    counters = inventory.get("counters", [])
    if len(counters) != len(PINS) or {c.get("key") for c in counters} != {PREFIX + k for k in PINS}:
        errors.append("retention_counter_inventory_incomplete")
    for counter in counters:
        pin = PINS.get(counter.get("key", "").removeprefix(PREFIX))
        if not pin or (counter.get("version"), counter.get("sha256")) != pin or sha(
            counter.get("body_utf8", "").encode()) != pin[1]:
            errors.append("retention_counter_pin_mismatch")
    # There is currently NO independently admitted clean pair/row export. A key
    # listing, counter, census manifest or local filename cannot create admission.
    errors += pair_errors(None) + row_errors(None)
    return {
        "schema": "jope.w5.admission-verdict/v1", "verdict": "FAIL", "w5": "HELD",
        "reasons": errors, "retention_receipts_verified": len(counters) if not any(
            "counter" in e for e in errors) else None,
        "eligible_pair_count": None, "pair_coverage": None, "row_closure": "UNKNOWN",
        "holdout_exposure": "UNKNOWN; no closure assertion",
        "native_fits": 0, "gpu_seconds": 0, "w5_training_started": False,
        "scope": "Admission preflight only; counters are not outcomes or pair evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collect", action="store_true")
    source.add_argument("--inventory", type=Path)
    parser.add_argument("--profile", default="veox-jepson-prod")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect:
        inventory = collect(args.profile)
        raw = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
        (args.output / "inventory.json").write_bytes(raw)
    else:
        raw = args.inventory.read_bytes()
        inventory = json.loads(raw)
    report = verify(inventory)
    report["inventory_sha256"] = sha(raw)
    report["verifier_sha256"] = sha(Path(__file__).read_bytes())
    (args.output / "verdict.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 1  # Intentional refusal while the independent admission prerequisites are absent.


if __name__ == "__main__":
    raise SystemExit(main())
