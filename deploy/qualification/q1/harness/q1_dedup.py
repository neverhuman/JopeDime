#!/usr/bin/env python3
"""Build metadata-only Q1 drafts from the frozen owner-family selection (T1115)."""

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess

import q1
import q1_bindings
import q1_cut

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parents[2]
AMENDMENT = BASE / "amendments/t1115.json"
BUNDLE = "94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
TRAINING_CLOSURE = "8e786b2dcb153a5c1c4f43dbcb773b53df6f0dd7d325c94f8d7ec2f73352d3e8"
LEDGER = "/home/ubuntu/jope_plan/ns/qualification/q1_ledger"
MINIMUM = 5


def asset(path):
    resolved = (ROOT / path).resolve()
    q1.require(resolved.is_relative_to(BASE), "metadata asset outside Q1 source")
    return resolved


def bound(reference):
    path = asset(reference["path"])
    q1.require(q1.sha_file(path) == reference["sha256"], "metadata input hash mismatch")
    return q1.load(path)


def select(packet, membership):
    """One minimum-SHA dataset per existing owner family; never mint an identity."""
    q1.require(
        membership["family_key_codec"] == "literal-fam-prefix-removal/v1",
        "family codec",
    )
    families = membership["families"]
    rows = packet["official_roster"]
    q1.require(len(families) == 97 and len(rows) == 153, "frozen owner counts")
    by_id = {}
    seen_families = set()
    for family in families:
        fid = family["family_id"]
        q1.require(
            re.fullmatch(r"fam-[0-9a-f]{64}", fid) is not None, "frozen family ID"
        )
        q1.require(fid not in seen_families, "duplicate frozen family ID")
        seen_families.add(fid)
        q1.require(family["partition"] == "sealed_excluded", "owner sealed partition")
        q1.require(family["dataset_ids"], "empty frozen family")
        for dataset_id in family["dataset_ids"]:
            q1.require(dataset_id not in by_id, "ambiguous dataset family")
            by_id[dataset_id] = fid
    q1.require(len({r["dataset_id"] for r in rows}) == 153, "duplicate source dataset")
    q1.require(
        set(by_id) == {r["dataset_id"] for r in rows}, "owner membership inventory"
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(by_id[row["dataset_id"]], []).append(row)
    q1.require(len(grouped) == 97, "missing frozen family")
    selected = []
    excluded = []
    for family_id, variants in sorted(grouped.items()):
        variants = sorted(variants, key=lambda row: row["dataset_sha256"])
        keep = variants[0]
        selected.append(
            {
                "id": keep["dataset_id"],
                "task": "regression",
                "dataset_sha256": keep["dataset_sha256"],
                "family": family_id.removeprefix("fam-"),
                "family_authoritative": True,
                "rows": keep["rows"],
                "features": keep["features"],
                "cut_sha256": keep["cut_sha256"],
                "holdout_group": "sealed_meta_test",
            }
        )
        for row in variants[1:]:
            excluded.append(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_sha256": row["dataset_sha256"],
                    "cut_sha256": row["cut_sha256"],
                    "family_id": family_id,
                    "selected_dataset_id": keep["dataset_id"],
                    "reason": "same frozen owner family; selected minimum eligible dataset SHA-256",
                }
            )
    return sorted(selected, key=lambda row: row["dataset_sha256"]), excluded


def load_selection():
    amendment = q1.load(AMENDMENT)
    q1.require(
        amendment["required_distinct_families_per_subgroup"] == MINIMUM,
        "five-family floor required",
    )
    packet = bound(amendment["source_roster"])
    bound(amendment["identity_table"])
    membership = bound(amendment["membership"])
    declared = bound(amendment["selection"])
    bound(amendment["wide_census"])
    selected, excluded = select(packet, membership)
    q1.require(
        selected == declared["datasets"]
        and excluded == declared["excluded_family_variants"],
        "selection replay mismatch",
    )
    q1.require(
        declared["subgroup_minimum"] == MINIMUM, "selection cannot waive subgroup floor"
    )
    q1.require(
        declared["counts"]["subgroups"] == subgroup_counts(selected),
        "declared subgroup counts",
    )
    original = {r["dataset_id"]: r for r in packet["official_roster"]}
    for dataset in selected:
        definition = q1.load(asset(original[dataset["id"]]["cut_definition_path"]))
        q1.require(
            q1_cut.cut_sha256(definition) == dataset["cut_sha256"],
            "selected cut hash mismatch",
        )
    return selected


def subgroup_counts(datasets):
    return {
        group: sum(group in q1.memberships(d) for d in datasets) for group in q1.GROUPS
    }


def make_plan(datasets, training_families, source_commit, evaluator_sha256=None):
    q1.require(
        evaluator_sha256 is None or q1.sha(evaluator_sha256), "evaluator hash syntax"
    )
    return {
        "schema": "jope.q1-plan/v1",
        "mode": "qualification",
        "source_commit": source_commit,
        "preregistration_sha256": q1.sha_file(q1.PREREGISTRATION),
        "backend_sha256": q1.sha_file(BASE / "harness/production_backend.py"),
        "budget": 24,
        "seeds": [0, 1, 2],
        "datasets": copy.deepcopy(datasets),
        "pins": {
            "bundle_sha256": BUNDLE,
            "dope_sha256": DOPE,
            "evaluator_sha256": evaluator_sha256,
            **q1_bindings.expected_pins(),
            "training_family_manifest_sha256": TRAINING_CLOSURE,
        },
        "training_families": list(training_families),
        "timeout_seconds": 600,
        "ledger_root": LEDGER,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-closure", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--evaluator-sha256")
    args = parser.parse_args()
    q1.require(
        q1.sha_file(args.training_closure) == TRAINING_CLOSURE,
        "training closure bytes changed",
    )
    out = args.out_dir.resolve()
    q1.require(
        not out.is_relative_to(ROOT), "draft output must stay outside source checkout"
    )
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    selected = load_selection()
    closure = q1.load(args.training_closure)
    plan = make_plan(
        selected,
        closure["authoritative_bound_family_keys"],
        source,
        args.evaluator_sha256,
    )
    counts = subgroup_counts(selected)
    deficits = {group: MINIMUM - n for group, n in counts.items() if n < MINIMUM}
    try:
        q1.validate_plan(plan)
        validator = "PASS"
    except (ValueError, TypeError) as error:
        validator = str(error)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    readiness = {
        "schema": "jope.q1-dedup-draft-readiness/v1",
        "source_commit": source,
        "plan": str(path),
        "plan_sha256": q1.sha_file(path),
        "plan_digest": q1.digest(plan),
        "datasets": len(selected),
        "distinct_families": len({d["family"] for d in selected}),
        "subgroups": counts,
        "required_families_per_subgroup": MINIMUM,
        "family_deficits": deficits,
        "validator": validator,
        "metadata_only": True,
        "runtime_qualification": "NOT_VERIFIED_BY_METADATA_BUILDER",
        "signable": False,
        "Q1": "NOT_RUN",
        "signing_request_filed": False,
        "budget_reset": False,
        "payload_reads": 0,
    }
    (out / "READINESS.json").write_text(json.dumps(readiness, indent=2) + "\n")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
