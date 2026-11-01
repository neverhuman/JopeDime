#!/usr/bin/env python3
"""H150 W5 provenance unblock: identity-only sealed-family manifest and the three W5 proofs (H98/H117 checks).

`build` writes an identity-only manifest of protected (sealed/holdout) families from existing identity and registry
metadata only: family ids, dataset hashes, view ids, context ids and catalog ids, with the SHA-256 of every source.
`check` reruns the three W5 proofs against the W5 candidate set (the pinned v3 corpus manifest):

1. global sealed absence: zero overlap with the manifest on every identity key, plus the P3 registry/holdout
   crosswalk; PASS requires an authority receipt for the manifest SHA-256;
2. parent -> child link: per-child lineage fields in the candidate manifest, or a lineage artifact in the source prefix;
3. original-row source: per-row source fields in the candidate manifest, or a row-ledger artifact in the source prefix.

No sealed rows, outcomes, feature bodies or pool arrays are opened. A proof is never weakened: missing evidence is
UNPROVED, never PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

V3_MANIFEST_SHA = "701affc61e14950bc5c890b6399a9d765f10e7f7f720a2ba8f1a16d93e1a7640"
PROPOSAL_SHA = "799d33eb305b64c0969aa0f4f2fc3e86d171fc58f1f295ded50944d561ab2efb"
OWNER_DEFINITION = {"production_contexts": 293, "catalog_views": 470}
PROTECTED_ASSIGNMENTS = ("sealed_excluded", "heldout")
PER_CHILD_LINEAGE = (
    "parent_program_sha256",
    "parent_dsl",
    "operator",
    "child_program_sha256",
)
PER_ROW_SOURCE = (
    "source_row_locator",
    "source_s3_key",
    "source_s3_version",
    "source_sha256",
    "original_row_ledger",
)
ARTIFACT_NAME = re.compile(
    r"lineage|parent|operator|child|row_ledger|source_row|strat300|original_row", re.I
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------------------ manifest


def from_proposal(proposal: dict) -> dict:
    """Owner t2300 roster: P3 sealed_excluded = 293 production contexts + 470 catalog views (RLFW enumeration)."""
    recs = proposal["sealed_family_records"]
    if any(r.get("member_roles") != ["sealed_final"] for r in recs.values()):
        raise ValueError("every proposal record must be sealed_final")
    contexts = {c for r in recs.values() for c in r.get("context_ids", [])}
    views = {v for r in recs.values() for v in r.get("view_ids", [])}
    counts = {
        "production_contexts": len(contexts),
        "catalog_views": len(views) - len(contexts),
    }
    if counts != OWNER_DEFINITION:
        raise ValueError(
            f"proposal does not enumerate the owner's 293 + 470 definition: {counts}"
        )
    return {
        "families": {f: sorted(r["sources"]) for f, r in recs.items()},
        "dataset_hashes": sorted(
            {d for r in recs.values() for d in r.get("dataset_hashes", [])}
        ),
        "view_ids": sorted(views),
        "context_ids": sorted(contexts),
        "counts": dict(counts, families=len(recs), views=len(views)),
    }


def build(
    proposal: dict, registry: dict, holdout: dict, catalog: dict, sources: dict
) -> dict:
    a = from_proposal(proposal)
    registry_sealed = sorted(
        f for f, v in registry["families"].items() if v.get("sealed")
    )
    holdout_protected = {
        k: sorted(f for f, x in holdout["assignment"].items() if x == k)
        for k in PROTECTED_ASSIGNMENTS
    }
    catalog_ids = sorted(catalog["inventory"]["sealed_name_matches_excluded"])
    families = dict(a["families"])
    for f in registry_sealed:
        families.setdefault(f, []).append("p3_registry_sealed")
    for k, fams in holdout_protected.items():
        for f in fams:
            families.setdefault(f, []).append(f"p3_holdout_{k}")
    manifest = {
        "schema": "jope.h150.sealed-family-identity-manifest/v1",
        "content": "identity only: family ids, dataset hashes, view ids, context ids, catalog ids; no rows, outcomes or features",
        "authority_definition": "owner t2300: the sealed meta-test group is all P3 sealed_excluded families (293 production + 470 catalog), by reference",
        "owner_definition_check": a["counts"],
        "families": {f: sorted(set(v)) for f, v in sorted(families.items())},
        "dataset_hashes": a["dataset_hashes"],
        "view_ids": a["view_ids"],
        "context_ids": a["context_ids"],
        "catalog_sealed_ids": catalog_ids,
        "counts": {
            "families_total": len(families),
            "owner_roster_families": a["counts"]["families"],
            "registry_sealed_families": len(registry_sealed),
            **{f"holdout_{k}_families": len(v) for k, v in holdout_protected.items()},
            "dataset_hashes": len(a["dataset_hashes"]),
            "view_ids": len(a["view_ids"]),
            "context_ids": len(a["context_ids"]),
            "catalog_sealed_ids": len(catalog_ids),
        },
        "sources": sources,
    }
    return manifest


# ------------------------------------------------------------------ proofs


def candidates(v3: dict) -> list[dict]:
    rows = {}
    for rev in v3["revisions"].values():
        for p in rev["materialized_pools"]:
            rows[p["context_sha256"]] = dict(
                context=p["context_sha256"],
                family=p["family_id"],
                dataset=p["dataset_id"],
                view=p["view_id"],
            )
    return sorted(rows.values(), key=lambda r: r["context"])


def proof1(
    cands: list[dict], manifest: dict, registry: dict, holdout: dict, receipt_ok: bool
) -> dict:
    fam = {r["family"] for r in cands}
    ds, vw, cx = ({r[k] for r in cands} for k in ("dataset", "view", "context"))
    prefixes = {v.split("__")[0] for v in vw}
    protected_ds = set(manifest["dataset_hashes"]) | set(manifest["catalog_sealed_ids"])
    overlap = {
        "families": sorted(fam & set(manifest["families"])),
        "dataset_hashes": sorted(ds & protected_ds),
        "view_ids": sorted(vw & set(manifest["view_ids"])),
        "context_ids": sorted(cx & set(manifest["context_ids"])),
        "view_dataset_prefixes": sorted(prefixes & protected_ds),
    }
    crosswalk = {
        "candidate_families_absent_from_registry": sorted(
            f for f in fam if f not in registry["families"]
        ),
        "candidate_families_registry_sealed": sorted(
            f for f in fam if registry["families"].get(f, {}).get("sealed")
        ),
        "candidate_families_holdout_protected": sorted(
            f for f in fam if holdout["assignment"].get(f) in PROTECTED_ASSIGNMENTS
        ),
    }
    clean = not any(overlap.values()) and not any(crosswalk.values())
    status = (
        "FAIL_OVERLAP"
        if not clean
        else ("PASS" if receipt_ok else "PASS_PENDING_AUTHORITY_RECEIPT")
    )
    return {
        "status": status,
        "overlap": overlap,
        "crosswalk": crosswalk,
        "authority_receipt": receipt_ok,
    }


def key_leaves(obj) -> set[str]:
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            out |= key_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= key_leaves(v)
    return out


def proof23(v3: dict, listing: list[list]) -> tuple[dict, dict]:
    leaves = key_leaves(v3)
    artifacts = sorted(
        k for k, *_ in listing if ARTIFACT_NAME.search(k.rsplit("/", 1)[-1])
    )
    lineage = sorted(f for f in PER_CHILD_LINEAGE if f in leaves)
    rows = sorted(f for f in PER_ROW_SOURCE if f in leaves)
    p2 = {
        "status": "UNPROVED"
        if not (lineage or artifacts)
        else "EVIDENCE_PRESENT_JOIN_UNCHECKED",
        "per_child_lineage_fields": lineage,
        "source_prefix_lineage_artifacts": artifacts,
        "source_prefix_objects": len(listing),
    }
    p3 = {
        "status": "UNPROVED"
        if not (rows or artifacts)
        else "EVIDENCE_PRESENT_BYTES_UNCHECKED",
        "per_row_source_fields": rows,
        "source_prefix_row_artifacts": artifacts,
        "source_prefix_objects": len(listing),
    }
    return p2, p3


def verdict(p1: dict, p2: dict, p3: dict) -> dict:
    passed = [p1["status"], p2["status"], p3["status"]] == ["PASS", "PASS", "PASS"]
    failing = [
        n
        for n, p in (("proof1", p1), ("proof2", p2), ("proof3", p3))
        if p["status"] != "PASS"
    ]
    return {
        "w5": "UNBLOCKED" if passed else "HELD",
        "all_three_pass": passed,
        "failing": failing,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for k in (
        "proposal",
        "registry",
        "alias-sources",
        "holdout",
        "catalog",
        "owner-ruling",
        "out",
    ):
        b.add_argument(f"--{k}", type=Path, required=True)
    c = sub.add_parser("check")
    for k in ("v3-manifest", "manifest", "registry", "holdout", "listing", "out"):
        c.add_argument(f"--{k}", type=Path, required=True)
    c.add_argument("--receipt", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        if sha(a.proposal) != PROPOSAL_SHA:
            raise SystemExit("proposal bytes differ from the pinned SHA-256")
        sources = {
            name: {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
            for name, p in (
                ("rlfw_sealed_roster_proposal", a.proposal),
                ("p3_family_registry", a.registry),
                ("p3_family_alias_registry", a.alias_sources),
                ("p3_holdout_assignment", a.holdout),
                ("s3_dataset_catalog", a.catalog),
                ("owner_ruling_t2300", a.owner_ruling),
            )
        }
        m = build(
            json.loads(a.proposal.read_text()),
            json.loads(a.registry.read_text()),
            json.loads(a.holdout.read_text()),
            json.loads(a.catalog.read_text()),
            sources,
        )
        m["sources"]["p3_holdout_assignment"]["s3_commitment"] = json.loads(
            a.holdout.read_text()
        )["commitment"]
        a.out.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"manifest_sha256": sha(a.out), "counts": m["counts"]}))
        return
    if sha(a.v3_manifest) != V3_MANIFEST_SHA:
        raise SystemExit("v3 manifest bytes differ from the H98-pinned object")
    manifest_sha = sha(a.manifest)
    receipt_ok = False
    if a.receipt and a.receipt.exists():
        r = json.loads(a.receipt.read_text())
        receipt_ok = (
            r.get("message_sha256") == manifest_sha and r.get("verify") == "PASS"
        )
    v3 = json.loads(a.v3_manifest.read_text())
    registry, holdout = (
        json.loads(a.registry.read_text()),
        json.loads(a.holdout.read_text()),
    )
    cands = candidates(v3)
    p1 = proof1(
        cands, json.loads(a.manifest.read_text()), registry, holdout, receipt_ok
    )
    p2, p3 = proof23(v3, json.loads(a.listing.read_text()))
    res = {
        "schema": "jope.h150.w5-proofs/v1",
        "inputs": {
            "v3_manifest_sha256": V3_MANIFEST_SHA,
            "manifest_sha256": manifest_sha,
            "listing_sha256": sha(a.listing),
            "receipt": str(a.receipt) if a.receipt else None,
        },
        "candidate_counts": {
            "contexts": len(cands),
            "families": len({r["family"] for r in cands}),
            "datasets": len({r["dataset"] for r in cands}),
            "views": len({r["view"] for r in cands}),
        },
        "proof1_global_sealed_absence": p1,
        "proof2_parent_child": p2,
        "proof3_original_row_source": p3,
        "verdict": verdict(p1, p2, p3),
        "payloads_read": {
            "sealed_rows": 0,
            "outcomes": 0,
            "feature_bodies": 0,
            "pool_arrays": 0,
        },
    }
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
