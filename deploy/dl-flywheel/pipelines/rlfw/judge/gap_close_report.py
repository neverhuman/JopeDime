"""Build the gap-close evidence: cluster intervals, hierarchical power, sealed proposal.

Reads pinned local files and, when asked, an R9 context-cost stream on stdin.
Does not create or spend a seal and does not write a training closure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from rlfw.judge.cluster import cluster_bootstrap_ci, hierarchical_power, variance_components
from rlfw.judge.roster import proof_disjoint, proposal_document

PARENTS = Path(
    "/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/"
    "flywheel_native_full_20261009/inputs/parents.jsonl"
)
SPEC = Path(
    "/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/"
    "flywheel_native_full_20261009/spec.json"
)
ROLE = Path(
    "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/owner-relays/"
    "one-holdout-role-map-v1-20261007T231936Z.json"
)
VAL_IDS = Path(
    "/home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-OP/scratch-train/r7/result/contexts-full-val.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parents(path):
    context_parent = {}
    parent_partition = {}
    parent_contexts = defaultdict(list)
    dataset_hashes = set()
    view_ids = set()
    for line in Path(path).open():
        row = json.loads(line)
        lineage = row["parent_lineage_sha256"]
        parent_partition[lineage] = row["partition"]
        dataset_hashes.add(row["dataset_hash"])
        view_ids.add(row["view_id"])
        for context_id in row["retained_lineage_context_sha256"]:
            if context_id in context_parent:
                raise ValueError("overlapping retained lineage " + context_id)
            context_parent[context_id] = lineage
            parent_contexts[lineage].append(context_id)
    return {
        "context_parent": context_parent,
        "parent_partition": parent_partition,
        "parent_contexts": parent_contexts,
        "dataset_hashes": dataset_hashes,
        "view_ids": view_ids,
    }


def iter_costs(handle):
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("split") not in (None, "full_val"):
            continue
        yield row


def complete_series(handle):
    """Latest epoch where control, rloo, and pkpo_b8 each have 263 distinct ids."""
    buckets = defaultdict(dict)
    for row in iter_costs(handle):
        key = (row["arm"], int(row["epoch"]))
        buckets[key][row["context_id"]] = float(row["l6_cost"])
    epochs = sorted({epoch for (_arm, epoch) in buckets})
    chosen = None
    for epoch in epochs:
        arms = {}
        ok = True
        for arm in ("r23_control", "rloo", "pkpo_b8"):
            series = buckets.get((arm, epoch))
            if series is None or len(series) != 263:
                ok = False
                break
            arms[arm] = series
        if ok:
            chosen = epoch
            chosen_arms = arms
    if chosen is None:
        raise ValueError("no complete 263-id epoch")
    return chosen, chosen_arms


def paired_groups(arms, context_parent, arm):
    groups = defaultdict(list)
    for context_id, control in arms["r23_control"].items():
        parent = context_parent[context_id]
        groups[parent].append(control - arms[arm][context_id])
    ordered = [groups[key] for key in sorted(groups)]
    if sum(len(g) for g in ordered) != 263:
        raise ValueError("paired roster is not the 263 validation contexts")
    return ordered


def build_sealed(role):
    buckets = {
        "sealed_production": role["sealed"]["production"],
        "sealed_catalog": role["sealed"]["catalog"],
    }
    other = []
    for key in ("catalog_heldout_p3v1", "heldout_non_g3", "wide_shape_gate_k512"):
        other.extend(role["newly_training"][key])
    other.extend(role["invalid_data"])
    other.extend(role["g3"]["heldout_no_s1"])
    other_families = set()
    for row in other:
        family = (row.get("p3") or {}).get("family_id")
        if family:
            other_families.add(family)
    families = {}
    for source, rows in buckets.items():
        for row in rows:
            family = row["p3"]["family_id"]
            record = families.setdefault(
                family,
                {
                    "context_ids": [],
                    "view_ids": [],
                    "dataset_hashes": [],
                    "member_roles": ["sealed_final"] if family not in other_families else ["sealed_final", "other"],
                    "sources": [],
                },
            )
            if source not in record["sources"]:
                record["sources"].append(source)
            if row.get("context_sha256"):
                record["context_ids"].append(row["context_sha256"])
            if row.get("view_id"):
                record["view_ids"].append(row["view_id"])
            if row.get("dataset_hash"):
                record["dataset_hashes"].append(row["dataset_hash"])
    for record in families.values():
        for kind in ("context_ids", "view_ids", "dataset_hashes", "sources"):
            record[kind] = sorted(set(record[kind]))
    return families


def closure_sets(parents, spec, sealed_context_ids, sealed_view_ids):
    admitted_contexts = set()
    admitted_views = set()
    input_hashes = []
    for ref in spec["input_lists"]:
        path = Path(ref["path"])
        input_hashes.append({"path": ref["path"], "sha256": sha256_file(path), "bytes": path.stat().st_size})
        document = json.loads(path.read_text())
        for row in document.get("admitted", []):
            if row.get("context_sha256"):
                admitted_contexts.add(row["context_sha256"])
            if row.get("view_id"):
                admitted_views.add(row["view_id"])
    target_path = Path(spec["authority_pins"]["target"]["path"])
    target = json.loads(target_path.read_text())
    target_views = set(target["views"])
    train_families = {
        lineage for lineage, part in parents["parent_partition"].items() if part == "train"
    }
    confirmation_families = {
        lineage for lineage, part in parents["parent_partition"].items() if part == "validation"
    }
    closure = {
        "family_ids": train_families | confirmation_families,
        "context_ids": set(parents["context_parent"]) | admitted_contexts,
        "view_ids": set(parents["view_ids"]) | admitted_views | target_views,
        "dataset_hashes": set(parents["dataset_hashes"]),
    }
    return closure, {
        "input_lists": input_hashes,
        "target_sha256": sha256_file(target_path),
        "target_views": len(target_views),
        "admitted_contexts": len(admitted_contexts),
        "sealed_context_ids_in_closure": len(sealed_context_ids & closure["context_ids"]),
        "sealed_views_in_closure": len(sealed_view_ids & closure["view_ids"]),
    }


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main(roster_path, evidence_dir, r9_handle, r9_sha=None, r9_bytes=None):
    parents = load_parents(PARENTS)
    val_ids = json.loads(VAL_IDS.read_text())
    if len(val_ids) != 263 or any(cid not in parents["context_parent"] for cid in val_ids):
        raise SystemExit("validation roster is not the 263 retained contexts")
    val_lineages = {parents["context_parent"][cid] for cid in val_ids}
    if any(parents["parent_partition"][lin] != "validation" for lin in val_lineages):
        raise SystemExit("a validation context inherits a non-validation parent")
    role = json.loads(ROLE.read_text())
    spec = json.loads(SPEC.read_text())
    sealed = build_sealed(role)
    sealed_contexts = {cid for record in sealed.values() for cid in record["context_ids"]}
    sealed_views = {view for record in sealed.values() for view in record["view_ids"]}
    closure, closure_meta = closure_sets(parents, spec, sealed_contexts, sealed_views)
    result = proof_disjoint(sealed, closure)
    if not result["passed"]:
        raise SystemExit("sealed proposal overlaps the training closure: " + json.dumps(result["failures"][:5]))
    sizes = []
    for lineage in sorted(val_lineages):
        sizes.append(len(parents["parent_contexts"][lineage]))
    partition = {
        "meta_training": sorted(
            lineage for lineage, part in parents["parent_partition"].items() if part == "train"
        ),
        "development": [],
        "confirmation": sorted(val_lineages),
        "sealed_final": sorted(sealed),
    }
    sources = {
        "parents_jsonl_sha256": sha256_file(PARENTS),
        "spec_sha256": sha256_file(SPEC),
        "role_map_sha256": sha256_file(ROLE),
        "validation_roster_sha256": sha256_file(VAL_IDS),
        "closure": closure_meta,
        "development_note": (
            "The role map that reclassified dev and heldout to TRAINING. "
            "No separate development family list was measured."
        ),
        "confirmation_note": (
            "The 54 validation lineages are the full_val 263 roster. Scratch verdicts "
            "have already consulted them, so they are confirmation, not sealed_final."
        ),
        "family_id_namespace": (
            "Sealed family ids are p3.family_id. Closure family_ids are "
            "parent_lineage_sha256. Those namespaces are not a crosswalk. "
            "The disjointness proof is context_id, view_id, and dataset_hash "
            "against the pinned native snapshot closure only."
        ),
    }
    document = proposal_document(partition, result, sources)
    document["sealed_family_records"] = sealed
    document["confirmation_cluster_sizes"] = sizes
    document["family_clean"] = {
        "train_lineages": len(partition["meta_training"]),
        "validation_lineages": len(partition["confirmation"]),
        "validation_contexts": 263,
        "shared_train_validation_lineages": 0,
        "sealed_families": len(sealed),
    }
    write_json(roster_path, document)

    rounds = {
        "R7": Path("/home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-OP/scratch-train/r7/result/context-costs.jsonl"),
        "R8": Path("/home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-OP/scratch-train/r8/result/context-costs.jsonl"),
    }
    reports = []
    for name, path in rounds.items():
        with path.open() as handle:
            epoch, arms = complete_series(handle)
        reports.extend(arm_report(name, path, epoch, arms, parents["context_parent"], sizes))
    if r9_handle is not None:
        epoch, arms = complete_series(r9_handle)
        r9_path = getattr(r9_handle, "name", None)
        reports.extend(arm_report(
            "R9",
            Path(r9_path) if r9_path and r9_path not in ("<stdin>", "") else None,
            epoch,
            arms,
            parents["context_parent"],
            sizes,
            sha=r9_sha,
            nbytes=r9_bytes,
        ))
    evidence = {
        "schema": "jope.rlfw-gap-close-cluster/v1",
        "qualifying": False,
        "seal_spent": False,
        "n882_applies": False,
        "n882_reason": "owner round 0.053 is an SE_pair, and n=882 holds only when that value is a paired SD",
        "confirmation_cluster_sizes": sizes,
        "arms": reports,
    }
    write_json(Path(evidence_dir) / "cluster-verdicts.json", evidence)
    write_json(Path(evidence_dir) / "roster-proof.json", {"passed": result["passed"], "sources": sources, "failures": result["failures"]})
    print(json.dumps({"roster_families": len(sealed), "proof": result["passed"], "arms": [
        {k: arm[k] for k in ("round", "arm", "epoch", "mean", "context_low", "context_high", "cluster_low", "cluster_high", "power_at_0.005")}
        for arm in reports
    ]}, indent=2))


def arm_report(name, path, epoch, arms, context_parent, sizes, sha=None, nbytes=None):
    out = []
    for arm in ("rloo", "pkpo_b8"):
        groups = paired_groups(arms, context_parent, arm)
        interval = cluster_bootstrap_ci(groups, draws=2000, seed=20261011)
        parts = variance_components(groups)
        power = hierarchical_power(
            sizes,
            parts["sigma_within"],
            parts["sigma_between"],
            0.005,
            sims=200,
            draws=400,
            seed=20261011,
        )
        null = hierarchical_power(
            sizes,
            parts["sigma_within"],
            parts["sigma_between"],
            0.0,
            sims=200,
            draws=400,
            seed=20261012,
        )
        out.append({
            "round": name,
            "path": None if path is None else str(path),
            "sha256": sha256_file(path) if path is not None else sha,
            "bytes": path.stat().st_size if path is not None else nbytes,
            "arm": arm,
            "epoch": epoch,
            "n_contexts": interval["n_contexts"],
            "n_clusters": interval["n_clusters"],
            "mean": interval["mean"],
            "context_low": interval["context"]["low"],
            "context_high": interval["context"]["high"],
            "cluster_low": interval["cluster_bootstrap"]["low"],
            "cluster_high": interval["cluster_bootstrap"]["high"],
            "cluster_includes_0": interval["cluster_bootstrap"]["low"] <= 0.0 <= interval["cluster_bootstrap"]["high"],
            "sigma_within": parts["sigma_within"],
            "sigma_between": parts["sigma_between"],
            "icc": parts["icc"],
            "power_at_0.005": power["cluster_reject_rate"],
            "context_power_at_0.005": power["context_reject_rate"],
            "cluster_null_reject_rate": null["cluster_reject_rate"],
            "power_sims": power["sims"],
            "power_draws": power["draws"],
            "qualifying": False,
        })
    return out


if __name__ == "__main__":
    roster = sys.argv[1]
    evidence = sys.argv[2]
    r9_path = None
    r9_sha = None
    r9_bytes = None
    for arg in sys.argv:
        if arg.startswith("--r9-path="):
            r9_path = arg.split("=", 1)[1]
        if arg.startswith("--r9-sha="):
            r9_sha = arg.split("=", 1)[1]
        if arg.startswith("--r9-bytes="):
            r9_bytes = int(arg.split("=", 1)[1])
    handle = open(r9_path) if r9_path else (sys.stdin if "--r9-stdin" in sys.argv else None)
    try:
        main(roster, evidence, handle, r9_sha, r9_bytes)
    finally:
        if r9_path and handle is not None:
            handle.close()
