"""H150 synthetic controls (no sealed rows, outcomes or real identities)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w5_proofs as w  # noqa: E402


def proposal(contexts=293, catalog_views=470):
    views = [f"{i:016x}__r0__f0" for i in range(contexts + catalog_views)]
    return {
        "sealed_family_records": {
            "fam-s1": {
                "member_roles": ["sealed_final"],
                "sources": ["sealed_production"],
                "dataset_hashes": ["d" * 16],
                "view_ids": views[:contexts],
                "context_ids": [f"c{i}" for i in range(contexts)],
            },
            "fam-s2": {
                "member_roles": ["sealed_final"],
                "sources": ["sealed_catalog"],
                "dataset_hashes": ["e" * 16],
                "view_ids": views[contexts:],
                "context_ids": [],
            },
        }
    }


REGISTRY = {
    "families": {
        "fam-a": {"sealed": False},
        "fam-b": {"sealed": False},
        "fam-r": {"sealed": True},
    }
}
HOLDOUT = {
    "assignment": {
        "fam-a": "train_fit",
        "fam-b": "train_fit",
        "fam-h": "heldout",
        "fam-r": "sealed_excluded",
    }
}
CATALOG = {"inventory": {"sealed_name_matches_excluded": ["f" * 16]}}
V3 = {
    "revisions": {
        "a": {
            "materialized_pools": [
                dict(
                    context_sha256="ctxA",
                    family_id="fam-a",
                    dataset_id="a" * 16,
                    view_id="a" * 16 + "__r0__f0",
                    parent_pool_sha256="x",
                )
            ]
        },
        "b": {
            "materialized_pools": [
                dict(
                    context_sha256="ctxB",
                    family_id="fam-b",
                    dataset_id="b" * 16,
                    view_id="b" * 16 + "__r0__f1",
                )
            ]
        },
    }
}


def manifest():
    return w.build(proposal(), REGISTRY, HOLDOUT, CATALOG, {})


def test_owner_definition_is_enforced():
    m = manifest()
    assert m["owner_definition_check"]["production_contexts"] == 293
    assert m["owner_definition_check"]["catalog_views"] == 470
    with pytest.raises(ValueError, match="293 \\+ 470"):
        w.from_proposal(proposal(contexts=290))
    bad = proposal()
    bad["sealed_family_records"]["fam-s1"]["member_roles"] = ["development"]
    with pytest.raises(ValueError, match="sealed_final"):
        w.from_proposal(bad)


def test_manifest_is_a_conservative_union_and_identity_only():
    m = manifest()
    assert set(m["families"]) == {"fam-s1", "fam-s2", "fam-r", "fam-h"}
    assert "p3_holdout_heldout" in m["families"]["fam-h"]
    assert set(m) >= {
        "families",
        "dataset_hashes",
        "view_ids",
        "context_ids",
        "catalog_sealed_ids",
    }
    assert not {"rows", "outcomes", "features", "fitness"} & set(m)


def test_proof1_requires_zero_overlap_and_an_authority_receipt():
    cands = w.candidates(V3)
    assert (
        w.proof1(cands, manifest(), REGISTRY, HOLDOUT, False)["status"]
        == "PASS_PENDING_AUTHORITY_RECEIPT"
    )
    assert w.proof1(cands, manifest(), REGISTRY, HOLDOUT, True)["status"] == "PASS"


@pytest.mark.parametrize(
    "hit",
    [
        "family",
        "dataset",
        "view",
        "context",
        "prefix",
        "catalog",
        "registry",
        "holdout",
    ],
)
def test_any_identity_or_crosswalk_hit_fails_proof1(hit):
    m, reg, ho = (
        manifest(),
        {"families": dict(REGISTRY["families"])},
        {"assignment": dict(HOLDOUT["assignment"])},
    )
    if hit == "family":
        m["families"]["fam-b"] = ["x"]
    elif hit == "dataset":
        m["dataset_hashes"].append("b" * 16)
    elif hit == "view":
        m["view_ids"].append("a" * 16 + "__r0__f0")
    elif hit == "context":
        m["context_ids"].append("ctxB")
    elif hit == "prefix":
        m["dataset_hashes"].append("a" * 16)
    elif hit == "catalog":
        m["catalog_sealed_ids"].append("b" * 16)
    elif hit == "registry":
        reg["families"]["fam-a"] = {"sealed": True}
    else:
        ho["assignment"]["fam-b"] = "heldout"
    assert w.proof1(w.candidates(V3), m, reg, ho, True)["status"] == "FAIL_OVERLAP"


def test_proofs_2_and_3_stay_unproved_without_lineage_or_row_evidence():
    listing = [["experiments/x/results/round-01/summary.json", 10, "2026"]]
    p2, p3 = w.proof23(V3, listing)
    assert p2["status"] == p3["status"] == "UNPROVED"
    v = w.verdict({"status": "PASS"}, p2, p3)
    assert v == {"w5": "HELD", "all_three_pass": False, "failing": ["proof2", "proof3"]}


def test_lineage_evidence_is_never_promoted_to_pass():
    listing = [["experiments/x/child_lineage.jsonl", 10, "2026"]]
    p2, p3 = w.proof23(V3, listing)
    assert (
        p2["status"] == "EVIDENCE_PRESENT_JOIN_UNCHECKED"
        and p3["status"] == "EVIDENCE_PRESENT_BYTES_UNCHECKED"
    )
    assert w.verdict({"status": "PASS"}, p2, p3)["w5"] == "HELD"
