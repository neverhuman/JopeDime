"""H117 — W5 data-provenance re-check: identity-only overlap and lineage-field census (CPU, no payload reads).

H98 and H102 held W5 because three proofs are missing: (1) every W5 family absent from the global sealed meta-test
roster, (2) every child linked to an existing parent program in its pool, (3) every original row traced to a source
S3 object and SHA. H117 re-tests them against sources H98/H102 did not use, reading identities and key names only:

* the byte-pinned v3 corpus manifest (the W5 candidate set: 254 contexts / 123 families; the same object and SHA as H98);
* the repository's RLFW sealed-roster proposal (97 sealed-final families with dataset hashes, view ids and contexts);
* the S3 dataset catalog's sealed-matching catalog-id exclusion list.

It compares family, dataset, view, context and view-dataset-prefix identities, and lists every key path in the
manifest that could carry lineage or row-source evidence. It never decodes a pool array or an outcome value, never
reads the sealed bucket, and never opens report bodies that may serialize outer-test outcomes.

Statuses are deliberately conservative: a zero overlap against non-authoritative sources is UNPROVED_BOUNDED, not PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

V3_MANIFEST_SHA = "701affc61e14950bc5c890b6399a9d765f10e7f7f720a2ba8f1a16d93e1a7640"
LINEAGE_KEY = re.compile(
    r"parent|operator|lineage|ancest|source_row|row_source|s3_key|origin", re.I
)
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(manifest: dict) -> list[dict]:
    rows = {}
    for rev in manifest["revisions"].values():
        for p in rev["materialized_pools"]:
            rows[p["context_sha256"]] = dict(
                context=p["context_sha256"],
                family=p["family_id"],
                dataset=p["dataset_id"],
                view=p["view_id"],
            )
    return sorted(rows.values(), key=lambda r: r["context"])


def sealed_from_proposal(proposal: dict) -> dict:
    recs = proposal["sealed_family_records"]
    return dict(
        families=set(recs),
        datasets={d for r in recs.values() for d in r.get("dataset_hashes", [])},
        views={v for r in recs.values() for v in r.get("view_ids", [])},
        contexts={c for r in recs.values() for c in r.get("context_ids", [])},
    )


def sealed_catalog_ids(catalog_md: str) -> set[str]:
    m = re.search(r"Sealed-matching catalog ids excluded:([^.]*)\.", catalog_md)
    return set(re.findall(r"\b[0-9a-f]{16}\b", m.group(1))) if m else set()


def overlap(cands: list[dict], sealed: dict, catalog_ids: set[str]) -> dict:
    fam, ds, vw, cx = (
        {r[k] for r in cands} for k in ("family", "dataset", "view", "context")
    )
    prefixes = {v.split("__")[0] for v in vw}
    return dict(
        proposal=dict(
            families=sorted(fam & sealed["families"]),
            datasets=sorted(ds & sealed["datasets"]),
            views=sorted(vw & sealed["views"]),
            contexts=sorted(cx & sealed["contexts"]),
            view_dataset_prefixes=sorted(prefixes & sealed["datasets"]),
        ),
        catalog=dict(
            datasets=sorted(ds & catalog_ids),
            view_dataset_prefixes=sorted(prefixes & catalog_ids),
        ),
    )


def key_paths(obj, prefix="") -> set[str]:
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= key_paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            out |= key_paths(v, f"{prefix}[]")
    return out


def lineage_census(manifest: dict) -> dict:
    keys = key_paths(manifest)
    leaves = {k.rsplit(".", 1)[-1] for k in keys}
    return dict(
        lineage_like_keys=sorted(
            k for k in keys if LINEAGE_KEY.search(k.rsplit(".", 1)[-1])
        ),
        per_child_lineage_fields_present=sorted(
            f for f in PER_CHILD_LINEAGE if f in leaves
        ),
        per_row_source_fields_present=sorted(f for f in PER_ROW_SOURCE if f in leaves),
    )


def verdict(ov: dict, census: dict) -> dict:
    """This checker can refute (an overlap, a missing field) but never grant a W5 proof: none of its sources is an
    authoritative global roster or an admitted lineage/row ledger. W5 therefore stays HELD whatever it finds."""
    any_overlap = any(v for part in ov.values() for v in part.values())
    return dict(
        proof1_global_sealed_absence="FAIL_OVERLAP"
        if any_overlap
        else "UNPROVED_BOUNDED",
        proof2_parent_child="UNPROVED"
        if not census["per_child_lineage_fields_present"]
        else "FIELDS_PRESENT_JOIN_UNCHECKED",
        proof3_original_row_source="UNPROVED"
        if not census["per_row_source_fields_present"]
        else "FIELDS_PRESENT_BYTES_UNCHECKED",
        w5="HELD",
        note="A zero overlap against an unadopted proposal and a catalog exclusion list is bounded evidence, never a "
        "global PASS; proof 1 needs an authoritative identity-only sealed roster with an authority receipt.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-manifest", type=Path, required=True)
    ap.add_argument("--sealed-proposal", type=Path, required=True)
    ap.add_argument("--catalog-md", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if sha256(a.v3_manifest) != V3_MANIFEST_SHA:
        raise SystemExit("v3 manifest bytes differ from the H98-pinned object")
    manifest = json.loads(a.v3_manifest.read_text())
    proposal = json.loads(a.sealed_proposal.read_text())
    cands = candidates(manifest)
    sealed = sealed_from_proposal(proposal)
    cat = sealed_catalog_ids(a.catalog_md.read_text())
    ov, census = overlap(cands, sealed, cat), lineage_census(manifest)
    res = dict(
        schema="jope.h117.w5-identity/v1",
        inputs=dict(
            v3_manifest_sha256=V3_MANIFEST_SHA,
            sealed_proposal_sha256=sha256(a.sealed_proposal),
            sealed_proposal_adopted=bool(proposal.get("adopted")),
            catalog_md_sha256=sha256(a.catalog_md),
        ),
        candidate_counts=dict(
            contexts=len(cands),
            families=len({r["family"] for r in cands}),
            datasets=len({r["dataset"] for r in cands}),
            views=len({r["view"] for r in cands}),
        ),
        sealed_counts=dict(
            proposal_families=len(sealed["families"]),
            proposal_datasets=len(sealed["datasets"]),
            proposal_views=len(sealed["views"]),
            proposal_contexts=len(sealed["contexts"]),
            catalog_ids=len(cat),
        ),
        overlap=ov,
        lineage_census=census,
        verdict=verdict(ov, census),
        payloads_read=dict(
            pool_arrays=0, outcome_values=0, sealed_bucket_objects=0, report_bodies=0
        ),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(
        json.dumps(
            dict(
                verdict=res["verdict"],
                candidate_counts=res["candidate_counts"],
                sealed_counts=res["sealed_counts"],
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
