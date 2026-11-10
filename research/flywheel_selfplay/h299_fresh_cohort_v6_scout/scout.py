"""Read-only catalog-v246 exhaustion proof; no dataset bodies or model calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from regress_v5 import load, sha

CATALOG_BUCKET = "veox-jopedime-datasets-use2"
CATALOG_KEY = "catalog/v246/catalog.jsonl"
CATALOG_VERSION = "eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC"
CATALOG_ETAG = "85058f0c455685c1159a343d52b60ebc"
CATALOG_SHA = "c5f5eb6db36d24daa9abc5c7808a49773a2b94c1765620d236e91a471fe2cb54"
COHORTS = {
    1: (
        "fresh_cohort_v1.json",
        "12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c",
    ),
    2: (
        "fresh_cohort_v2_H286.json",
        "0de1b3839cb3201d6c8a05aafec134e28e119ae6c233588b41d6169adb3e7b11",
    ),
    3: (
        "fresh_cohort_v3_H288.json",
        "e247983d0412f37ad28ba3553decee1f4f6a096bd461caefafca58fdc858eac3",
    ),
    4: (
        "fresh_cohort_v4_H292.json",
        "7f7ff09da91b5ab8338a43d62a553b0fefe2cda59e00a8d48e80e49c57266aba",
    ),
    5: (
        "fresh_cohort_v5_H294.json",
        "ea5364ad8d224719300a7932b0d8418709793405c1668dc581aac59302f7d0c9",
    ),
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def set_sha(values):
    return hashlib.sha256(canonical(sorted(set(values)))).hexdigest()


def split_family(key):
    digest = hashlib.sha256(b"H299-v6-split-v1\0" + key.encode()).digest()
    return "v6A" if digest[-1] % 2 == 0 else "v6B"


def scanned_partitions(cohorts):
    """Reconstruct full scanned spans, including failures, from frozen order and stop anchors."""
    v1 = cohorts[1]
    result = {}
    for half, first, middle, last, middle_v in [
        ("EVEN", "H283", "H286", "H294", 2),
        ("ODD", "H285", "H288", "H292", 3),
    ]:
        ordered = [r["family_key"] for r in v1["eligible"] if r["half"] == half]
        assigned = set(v1["assignment"][first]["families"])
        remaining = [f for f in ordered if f not in assigned]
        anchor = cohorts[middle_v]["assignment"][middle]["families"][-1]
        boundary = remaining.index(anchor) + 1
        result[first + "_assigned"] = assigned
        result[first + "_initial_scanned_prefix"] = set(
            ordered[: ordered.index(v1["assignment"][first]["families"][-1]) + 1]
        )
        result[middle + "_scanned_including_failures"] = set(remaining[:boundary])
        result[last + "_scanned_including_failures"] = set(remaining[boundary:])
        assert assigned.isdisjoint(result[middle + "_scanned_including_failures"])
        assert assigned | set(remaining) == set(ordered)
        assert set(remaining[:boundary]).isdisjoint(remaining[boundary:])
    return result


def fetch_catalog(out):
    import boto3

    session = boto3.Session(profile_name="veox-jepson-prod")
    identity = session.client("sts").get_caller_identity()
    assert identity["Account"] == "714044927654"
    client = session.client("s3", region_name="us-east-2")
    response = client.get_object(
        Bucket=CATALOG_BUCKET, Key=CATALOG_KEY, VersionId=CATALOG_VERSION
    )
    raw = response["Body"].read()
    response["Body"].close()
    assert response["VersionId"] == CATALOG_VERSION
    assert response["ETag"].strip('"') == CATALOG_ETAG
    assert hashlib.sha256(raw).hexdigest() == CATALOG_SHA
    (out / "catalog-v246.jsonl").write_bytes(raw)
    receipt = dict(
        uri=f"s3://{CATALOG_BUCKET}/{CATALOG_KEY}",
        version_id=CATALOG_VERSION,
        etag=CATALOG_ETAG,
        sha256=CATALOG_SHA,
        bytes=len(raw),
        exact_version=True,
        operation="GetObject",
        s3_write_operations=0,
        caller_arn=identity["Arn"],
        epoch=time.time(),
    )
    (out / "catalog-s3-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return [json.loads(line) for line in raw.splitlines()], receipt


def main(reference, out, source):
    assert json.loads((out / "regression.json").read_text())["status"] == "PASS"
    raw_catalog, remote = fetch_catalog(out)
    assert len(raw_catalog) == 4750
    assert all(d["origin"] == "real" for d in raw_catalog)
    fc = load(
        reference / "h283_rerank8_veto_fresh_cohort_gpu/fresh_cohort.py", "h299_v1"
    )
    assert (
        sha(reference / "h283_rerank8_veto_fresh_cohort_gpu/fresh_cohort.py")
        == "2b506d43123272b17f233c703682e6282344a32f5c9bb06c830833e2084b3df6"
    )
    receipts = Path("/home/ubuntu/jope_plan/ns/receipts")
    cohorts = {}
    for version, (name, digest) in COHORTS.items():
        assert sha(receipts / name) == digest
        cohorts[version] = json.loads((receipts / name).read_text())
    v1 = cohorts[1]
    catalog_path = fc.CATALOG
    assert sha(catalog_path) == v1["sources"]["catalog"]["sha256"]
    processed = json.loads(catalog_path.read_text())
    cat = processed["datasets"]
    by_id = {d["id"]: d for d in cat}
    sealed_ids = set(processed["inventory"]["sealed_name_matches_excluded"])
    assert sealed_ids == fc.SEALED_IDS
    assert {d["id"] for d in raw_catalog} == set(by_id) | sealed_ids
    assert set(by_id).isdisjoint(sealed_ids)
    for row in raw_catalog:
        if row["id"] in sealed_ids:
            continue  # metadata identifiers only; never fetch excluded manifests
        d = by_id[row["id"]]
        assert d["canonical_sha256"] == row["canonical_sha256"]
        assert d["task"] == row["task"]
        assert d.get("train_sha256") == row.get("files", {}).get("train", {}).get(
            "sha256"
        )
        assert d.get("family_key") == row.get("provenance", {}).get(
            "dedupe_fingerprint"
        )

    assert sha(fc.ADMITTED) == v1["sources"]["admitted_corpus"]["sha256"]
    assert sha(fc.PACKET) == v1["sources"]["packet"]["sha256"]
    s1, s1_hashes = fc.s1_entries()
    assert s1_hashes == v1["sources"]["s1_prod_input_lists"]
    # Exact original base exclusions/buildability metadata logic; no summary contents here.
    fc.main(str(out / "regenerated-v1.json"))
    assert sha(out / "regenerated-v1.json") == COHORTS[1][1]

    corpus_path = Path(
        "/mnt/jope-archive/flywheel-20261023/corpus/corpus_manifest.json"
    )
    assert (
        sha(corpus_path)
        == "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
    )
    corpus = json.loads(corpus_path.read_text())
    # Read roster metadata only; no outcome field is accessed or copied.
    roster = [
        {
            k: r[k]
            for k in [
                "context_sha256",
                "dataset_id",
                "family_id",
                "source_split",
                "dl_split",
                "n_candidates",
            ]
        }
        for r in corpus["records"]
    ]
    (out / "corpus-roster-projection.json").write_text(
        json.dumps(roster, indent=2) + "\n"
    )
    packet = json.loads(fc.PACKET.read_text())

    def families(ids):
        return {
            fc.fam_key(by_id[i]) for i in ids if i in by_id and fc.fam_key(by_id[i])
        }

    sets = scanned_partitions(cohorts)
    for version, cohort in cohorts.items():
        sets[f"cohort_v{version}_all_listed"] = {
            r["family_key"] for r in cohort["eligible"]
        }
    split_counts = {}
    for split, expected_count in [
        ("DL-Train", 1219),
        ("DL-Val", 91),
        ("DL-Holdout", 93),
    ]:
        selected_roster = [
            r for r in roster if r["n_candidates"] > 0 and r["dl_split"] == split
        ]
        assert len(selected_roster) == expected_count
        ids = {r["dataset_id"] for r in selected_roster}
        sets["flywheel_" + split] = families(ids)
        split_counts[split] = dict(
            contexts=len(selected_roster),
            coarse_families=len({r["family_id"] for r in selected_roster}),
            catalog_families=len(families(ids)),
        )
        if split != "DL-Holdout":
            key = "train" if split == "DL-Train" else "dev"
            assert {r["context_sha256"] for r in selected_roster} == {
                r["context_sha256"] for r in packet[key]
            }
    sets["flywheel_all_1780_metadata_contexts"] = families(
        r["dataset_id"] for r in roster
    )
    sets["H287_fit_TRAIN"] = families(r["dataset_id"] for r in packet["train"])
    sets["H291_fit_TRAIN"] = set(sets["H287_fit_TRAIN"])
    sets["H295_fit_TRAIN"] = set(sets["H287_fit_TRAIN"])
    coarse = {}
    for r in s1:
        coarse.setdefault(r["dataset_hash"], set()).add(r["_p3"].get("family_id"))
    for role in ["sealed_excluded", "heldout", "dev"]:
        ids = {r["dataset_hash"] for r in s1 if r["_p3"].get("assignment") == role}
        coarse_ids = {
            r["_p3"].get("family_id") for r in s1 if r["dataset_hash"] in ids
        } - {None}
        ids |= {i for i, fs in coarse.items() if fs & coarse_ids}
        sets["P3_" + role + "_coarse_closure"] = families(ids)
    sets["catalog_BeyondArena_or_sealed_text"] = {
        fc.fam_key(d)
        for d in cat
        if fc.fam_key(d)
        and any(
            t
            in json.dumps(
                [
                    d.get("name"),
                    d.get("source_url"),
                    d.get("source_file"),
                    d.get("collections"),
                ]
            ).lower()
            for t in ["sealed", "beyondarena"]
        )
    }
    # The sealed IDs were removed from the enriched catalog before family derivation.
    raw_sealed_families = {
        r.get("provenance", {}).get("dedupe_fingerprint")
        for r in raw_catalog
        if r["id"] in sealed_ids
    } - {None}
    sets["catalog_sealed_matching_families_known"] = raw_sealed_families
    eligible = {r["family_key"] for r in v1["eligible"]}
    partition_names = [
        "H283_assigned",
        "H286_scanned_including_failures",
        "H294_scanned_including_failures",
        "H285_assigned",
        "H288_scanned_including_failures",
        "H292_scanned_including_failures",
    ]
    partition = [sets[k] for k in partition_names]
    assert (
        sum(map(len, partition))
        == len(set().union(*partition))
        == len(eligible)
        == 2729
    )
    assert set().union(*partition) == eligible
    available = eligible - set().union(*sets.values())
    # This fixed-catalog proof refuses any unexplained remainder rather than manufacturing NO_COHORT.
    assert not available, (
        "Unexpected remainder needs unchanged H286 qualification before freezing"
    )
    coverage = []
    for d in raw_catalog:
        if d["task"] != "regression":
            reason = "not_regression"
        elif d["id"] in sealed_ids:
            reason = "catalog_sealed_matching_id"
        elif fc.fam_key(by_id[d["id"]]) not in eligible:
            reason = "unchanged_H283_H286_base_exclusion"
        else:
            reason = "prior_fresh_scanned_family_including_failures"
        coverage.append(dict(dataset_id=d["id"], reason=reason))
    counts = dict(Counter(r["reason"] for r in coverage))
    assert sum(counts.values()) == 4750
    (out / "catalog-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (out / "exclusion-sets.json").write_text(
        json.dumps({k: sorted(v) for k, v in sets.items()}, indent=2) + "\n"
    )
    proof = {
        k: dict(
            count=len(v), set_sha256=set_sha(v), intersection_count=len(v & available)
        )
        for k, v in sets.items()
    }
    proof["catalog_sealed_matching_ids"] = dict(
        count=len(sealed_ids),
        set_sha256=set_sha(sealed_ids),
        intersection_count=0,
        namespace="dataset_id",
    )
    manifest = dict(
        schema="jope.fresh-cohort/v6",
        wave="H299",
        verdict="NO_COHORT",
        catalog=remote,
        enriched_catalog_sha256=sha(catalog_path),
        qualification="Unchanged H286: admitted S1 TRAIN-side summary+teacher, >=2 accept lines, train_fit first then dataset/context SHA order.",
        split_rule='SHA256(UTF8("H299-v6-split-v1") || NUL || UTF8(family_key)); last byte parity 0=v6A,1=v6B',
        threshold_total_families=120,
        qualifying_families=0,
        candidates=[],
        v6A=[],
        v6B=[],
        exclusions=proof,
        exclusion_sets_sha256=sha(out / "exclusion-sets.json"),
        exclusion_family_sets={k: sorted(v) for k, v in sets.items()},
        catalog_sealed_matching_dataset_ids=sorted(sealed_ids),
        unknown_sealed_bucket_roster=dict(
            status="NOT_AVAILABLE",
            set_sha256=None,
            intersection_count=0,
            reason="Empty candidate set intersects every possible roster in zero; no complete sealed roster certification is claimed.",
        ),
        no_dataset_or_model_payload_reads=True,
        s3_write_operations=0,
        scoring_authorized=False,
    )
    (source / "cohort_v6_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    result = dict(
        status="COMPLETE",
        verdict="NO_COHORT",
        wave="H299",
        catalog_records=4750,
        catalog_coverage=counts,
        eligible_base_families=2729,
        prior_scanned_partition_counts={k: len(sets[k]) for k in partition_names},
        available_unseen_families=0,
        qualifying_families=0,
        v6A=0,
        v6B=0,
        regression=json.loads((out / "regression.json").read_text()),
        catalog=remote,
        manifest_sha256=sha(source / "cohort_v6_manifest.json"),
        coverage_sha256=sha(out / "catalog-coverage.json"),
        exclusion_proof=proof,
        flywheel_roster_counts=split_counts,
        metadata_sources=dict(
            corpus_manifest_sha256=sha(corpus_path),
            corpus_roster_projection_sha256=sha(out / "corpus-roster-projection.json"),
            original_v1_sha256=sha(out / "regenerated-v1.json"),
            s1_input_lists=s1_hashes,
        ),
        model_forwards=0,
        optimizer_updates=0,
        scout_accept_summary_reads=0,
        sealed_or_holdout_data_reads=0,
        s3_write_operations=0,
        H300_status="DRAFT_BLOCKED_NO_COHORT",
        limitations=[
            "Catalog v246 only; no claim about newer catalogs or newly ingested sources.",
            "Full authoritative sealed bucket roster unavailable; zero candidates make this a conservative exhaustion result, not a positive safety certification.",
            "H287/H291/H295 TRAIN fit exclusions follow their pinned source lineage; no fit outcomes were loaded.",
        ],
    )
    (source / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    assert (source / "metrics.json").stat().st_size < 50000
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "coverage": counts,
                "partition": result["prior_scanned_partition_counts"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ["reference", "out", "source"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.out, args.source)
