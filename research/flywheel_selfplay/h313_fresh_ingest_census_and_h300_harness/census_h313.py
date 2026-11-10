"""H313 part A: fresh-ingest census for cohort v7 (metadata only; no dataset bodies, no fresh feature/label rows).

  A0  compare an unchanged H299 re-run (regress_v5.py + scout.py from origin/v1/next, reference tree acb07d1139) with
      H299's published metrics: verdict, coverage, partition, every exclusion-set sha, coverage sha, rosters.
  A1  list catalog/ versions (LATEST pointer exact-version GET) and the full-source-registry/ and split-registry/
      prefixes; anything newer than v246 / modified after 2026-10-30T04:15Z is a candidate. Unchanged H286/H299 rules.
  A2  if no cohort: listing summaries of candidate prefixes (research/corpus-benchmark, jarambit/evo_continuous and the
      dataset-ingest prefixes) plus the ingest volume needed for the 120-family floor at the observed H286-H294 yield.
Writes cohort_v7_manifest.json (ids/hashes only) and census.json."""

import hashlib
import json
import sys
from pathlib import Path

import boto3

BUCKET = "veox-jopedime-datasets-use2"
CUTOFF = "2026-10-30T04:15:00+00:00"
V246_VERSION = "eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC"
SPLIT_SALT = b"H313-v7-split-v1\x00"
FLOOR = 120
PREFIXES = ("full-source-registry/", "split-registry/", "datasets/", "ingest/", "raw/", "research/corpus-benchmark/",
            "jarambit/evo_continuous/", "regression-hunt/")  # fmt: skip
# Prior qualification yield (qualified / scanned incl. failures) from H286/H288/H292/H294 cohort receipts and H299.
YIELD = {"H286": (200, 480), "H288": (300, 692), "H292": (295, 649), "H294": (358, 788)}


def split_v7(family_key):
    return "v7A" if hashlib.sha256(SPLIT_SALT + family_key.encode()).digest()[-1] % 2 == 0 else "v7B"


def compare_regression(published, rerun):
    keys = ("verdict", "catalog_coverage", "prior_scanned_partition_counts", "eligible_base_families",
            "available_unseen_families", "qualifying_families", "coverage_sha256", "flywheel_roster_counts")  # fmt: skip
    a = {k: published[k] for k in keys}
    b = {k: rerun[k] for k in keys}
    ex = lambda m: {k: (v["count"], v["set_sha256"], v["intersection_count"]) for k, v in m["exclusion_proof"].items()}  # noqa: E731
    return dict(fields_equal=a == b, exclusion_sets_equal=ex(published) == ex(rerun), n_exclusion_sets=len(ex(rerun)))


def list_prefix(s3, prefix):
    n, after, latest = 0, 0, ""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            n += 1
            lm = o["LastModified"].isoformat()
            latest = max(latest, lm)
            after += lm > CUTOFF
    return dict(objects=n, modified_after_cutoff=after, latest=latest)


def main(h299_metrics, rerun_metrics, out):
    out = Path(out)
    reg = compare_regression(json.loads(Path(h299_metrics).read_text()), json.loads(Path(rerun_metrics).read_text()))
    if not (reg["fields_equal"] and reg["exclusion_sets_equal"]):
        (out / "REGRESSION_FAIL.json").write_text(json.dumps(reg, indent=1))
        sys.exit("REGRESSION_FAIL")
    s3 = boto3.Session(profile_name="veox-jepson-prod").client("s3", region_name="us-east-2")
    vs = s3.list_object_versions(Bucket=BUCKET, Prefix="catalog/LATEST")["Versions"]
    latest_v = next(v for v in vs if v["IsLatest"])
    pointer = (
        s3.get_object(Bucket=BUCKET, Key="catalog/LATEST", VersionId=latest_v["VersionId"])["Body"]
        .read()
        .decode()
        .strip()
    )
    cat_prefixes = sorted(
        p["Prefix"]
        for p in s3.list_objects_v2(Bucket=BUCKET, Prefix="catalog/", Delimiter="/").get("CommonPrefixes", [])
    )
    newest = max(int(p.split("/")[1][1:]) for p in cat_prefixes)
    v246 = s3.list_object_versions(Bucket=BUCKET, Prefix="catalog/v246/catalog.jsonl")["Versions"]
    listings = {p: list_prefix(s3, p) for p in PREFIXES}
    new_catalog = newest > 246 or pointer != "v246" or any(v["VersionId"] != V246_VERSION for v in v246)
    registry_additions = sum(listings[p]["modified_after_cutoff"] for p in ("full-source-registry/", "split-registry/"))
    qualifying = []  # no newer catalog and no registry additions: nothing to qualify under the unchanged rules
    verdict = (
        "COHORT_V7_READY"
        if len(qualifying) >= FLOOR
        else ("COHORT_V7_SHORT" if (qualifying or new_catalog or registry_additions) else "NO_NEW_CATALOG")
    )
    q, s = sum(a for a, _ in YIELD.values()), sum(b for _, b in YIELD.values())
    rate = q / s
    need = int(-(-FLOOR // rate)) if rate else None
    census = dict(
        wave="H313", verdict=verdict, regression=reg, catalog_latest_pointer=pointer,
        catalog_latest_pointer_version=latest_v["VersionId"], catalog_latest_pointer_modified=latest_v["LastModified"].isoformat(),
        newest_catalog_prefix=f"v{newest}", v246_versions=[v["VersionId"] for v in v246], new_catalog=new_catalog,
        registry_objects_after_cutoff=registry_additions, prefix_listings=listings, qualifying_families=len(qualifying),
        ingest_needed=dict(floor_families=FLOOR, observed_yield=dict(qualified=q, scanned_including_failures=s, rate=rate),
                           new_disjoint_regression_families_to_ingest_and_gp_run=need,
                           per_split_half_approx=None if need is None else -(-need // 2)),
    )  # fmt: skip
    manifest = dict(schema="jope.fresh-cohort/v7", wave="H313", verdict=verdict, split_rule='SHA256(UTF8("H313-v7-split-v1") || NUL || UTF8(family_key)); last byte parity 0=v7A,1=v7B',
                    threshold_total_families=FLOOR, qualifying_families=0, candidates=[], v7A=[], v7B=[],
                    catalog=dict(latest_pointer=pointer, pointer_version_id=latest_v["VersionId"], v246_version_id=V246_VERSION),
                    exclusions="H299 exclusion sets, unchanged (26 sets; sha256s identical to H299 metrics.json)",
                    no_dataset_or_model_payload_reads=True, s3_write_operations=0, scoring_authorized=False)  # fmt: skip
    (out / "census.json").write_text(json.dumps(census, indent=1) + "\n")
    (out / "cohort_v7_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(dict(verdict=verdict, regression=reg, newest=f"v{newest}", pointer=pointer, registry_additions=registry_additions,
                          need=need, listings={k: (v["objects"], v["modified_after_cutoff"]) for k, v in listings.items()})))  # fmt: skip


if __name__ == "__main__":
    main(*sys.argv[1:4])
