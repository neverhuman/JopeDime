"""H117 structural tests on synthetic fixtures (no data payloads)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_w5_identity as h  # noqa: E402

MANIFEST = {
    "revisions": {
        "a": {
            "materialized_pools": [
                dict(
                    context_sha256="c1",
                    family_id="fam-1",
                    dataset_id="aaaaaaaaaaaaaaaa",
                    view_id="aaaaaaaaaaaaaaaa__r0__f0",
                    parent_pool_sha256="x",
                )
            ]
        },
        "b": {
            "materialized_pools": [
                dict(
                    context_sha256="c1",
                    family_id="fam-1",
                    dataset_id="aaaaaaaaaaaaaaaa",
                    view_id="aaaaaaaaaaaaaaaa__r0__f0",
                    parent_pool_sha256="x",
                ),
                dict(
                    context_sha256="c2",
                    family_id="fam-2",
                    dataset_id="bbbbbbbbbbbbbbbb",
                    view_id="bbbbbbbbbbbbbbbb__r0__f1",
                    parent_pool_sha256="y",
                ),
            ]
        },
    }
}
PROPOSAL = {
    "adopted": False,
    "sealed_family_records": {
        "fam-9": dict(
            dataset_hashes=["9999999999999999"],
            view_ids=["9999999999999999__r0__f0"],
            context_ids=[],
        )
    },
}


def test_candidates_are_unique_contexts_across_revisions():
    c = h.candidates(MANIFEST)
    assert [r["context"] for r in c] == ["c1", "c2"] and {r["family"] for r in c} == {
        "fam-1",
        "fam-2",
    }


def test_zero_overlap_is_bounded_never_pass_and_w5_stays_held():
    ov = h.overlap(
        h.candidates(MANIFEST), h.sealed_from_proposal(PROPOSAL), {"cccccccccccccccc"}
    )
    v = h.verdict(ov, h.lineage_census(MANIFEST))
    assert v["proof1_global_sealed_absence"] == "UNPROVED_BOUNDED" and v["w5"] == "HELD"
    assert (
        v["proof2_parent_child"] == "UNPROVED"
        and v["proof3_original_row_source"] == "UNPROVED"
    )


@pytest.mark.parametrize("hit", ["family", "dataset", "view", "prefix", "catalog"])
def test_any_identity_overlap_fails_proof_one(hit):
    p = json.loads(json.dumps(PROPOSAL))
    cat = set()
    rec = p["sealed_family_records"]["fam-9"]
    if hit == "family":
        p["sealed_family_records"]["fam-2"] = dict(
            dataset_hashes=[], view_ids=[], context_ids=[]
        )
    elif hit == "dataset":
        rec["dataset_hashes"].append("bbbbbbbbbbbbbbbb")
    elif hit == "view":
        rec["view_ids"].append("aaaaaaaaaaaaaaaa__r0__f0")
    elif hit == "prefix":
        rec["dataset_hashes"].append("aaaaaaaaaaaaaaaa")
    else:
        cat = {"bbbbbbbbbbbbbbbb"}
    ov = h.overlap(h.candidates(MANIFEST), h.sealed_from_proposal(p), cat)
    assert (
        h.verdict(ov, h.lineage_census(MANIFEST))["proof1_global_sealed_absence"]
        == "FAIL_OVERLAP"
    )


def test_lineage_census_distinguishes_pool_level_parent_from_per_child_lineage():
    c = h.lineage_census(MANIFEST)
    assert any(k.endswith("parent_pool_sha256") for k in c["lineage_like_keys"])
    assert c["per_child_lineage_fields_present"] == []
    m = json.loads(json.dumps(MANIFEST))
    m["revisions"]["b"]["materialized_pools"][0]["operator"] = "mutate"
    assert h.lineage_census(m)["per_child_lineage_fields_present"] == ["operator"]


def test_catalog_ids_are_parsed_from_the_exclusion_sentence_only():
    md = "intro 0123456789abcdef\nSealed-matching catalog ids excluded: 0a8428ef25a69175, 24713c7f7bfa7807. Also fedcba9876543210"
    assert h.sealed_catalog_ids(md) == {"0a8428ef25a69175", "24713c7f7bfa7807"}


def test_manifest_hash_pin_is_enforced(tmp_path, monkeypatch):
    f = tmp_path / "m.json"
    f.write_text(json.dumps(MANIFEST))
    (tmp_path / "p.json").write_text(json.dumps(PROPOSAL))
    (tmp_path / "c.md").write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--v3-manifest",
            str(f),
            "--sealed-proposal",
            str(tmp_path / "p.json"),
            "--catalog-md",
            str(tmp_path / "c.md"),
            "--out",
            str(tmp_path / "o.json"),
        ],
    )
    with pytest.raises(SystemExit):
        h.main()
    monkeypatch.setattr(
        h, "V3_MANIFEST_SHA", hashlib.sha256(f.read_bytes()).hexdigest()
    )
    h.main()
    assert json.loads((tmp_path / "o.json").read_text())["verdict"]["w5"] == "HELD"
