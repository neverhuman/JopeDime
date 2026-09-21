#!/usr/bin/env python3
"""Bind existing raw-component views to a new registry-checked assembly cohort."""

import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--registry", required=True)
p.add_argument("--existing-index", required=True)
p.add_argument("--existing-jobs", action="append", required=True)
p.add_argument("--new-jobs", action="append", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((a.root / path).read_text())


def binding(path):
    return dict(path=path, sha256=sha(a.root / path))


registry = read(a.registry)
assert registry["qualification"] and not registry["independent_selection_eligible"]
admitted = {}
for family in registry["families"].values():
    for ds in family["datasets"]:
        assert ds["dataset"] not in admitted
        admitted[ds["dataset"]] = (family, ds)
old = read(a.existing_index)
oldrows = {(r["dataset"], r["cut"]): r for r in old["encodings"]}
assert len(oldrows) == len(old["encodings"])
jobs = []
seen = set()
prepared = {}
oldseen = set()
for path in a.existing_jobs + a.new_jobs:
    source = read(path)
    assert source["schema"] == "parallel.gp_jobs.v1" and source["qualification"]
    for job in source["jobs"]:
        pp = job["prepared"]
        if pp not in prepared:
            meta = read(pp + "/prepared.json")
            assert sha(a.root / pp / "prepared.json") == job["prepared_sha256"]
            family, ds = admitted[meta["dataset"]]
            assert not family["holds"] and ds["preparation_eligible"]
            assert meta["family"] == family["family"] and meta["meta_partition"] == family["partition"]
            if meta.get("family_registry_binding"):
                b = meta["family_registry_binding"]
                for k in ["family", "partition"]:
                    assert b[k] == family[k]
                for k in ["metadata_sha256", "entity_review_sha256"]:
                    assert b.get(k) == ds.get(k)
            prepared[pp] = meta
        meta = prepared[pp]
        assert sha(a.root / pp / "prepared.json") == job["prepared_sha256"]
        key = (meta["dataset"], job["cut"])
        assert key not in seen and any(c["index"] == job["cut"] for c in meta["cuts"])
        seen.add(key)
        if path in a.existing_jobs:
            row = oldrows[key]
            assert row["prepared"] == pp and row["prepared_sha256"] == job["prepared_sha256"]
            assert row["family"] == meta["family"] and row["partition"] == meta["meta_partition"]
            oldseen.add(key)
        jobs.append(job)
assert oldseen == set(oldrows), "Existing context membership must be preserved exactly"
out = dict(
    schema="parallel.gp_jobs.v1",
    qualification=True,
    jobs=jobs,
    purpose="assembly only: frozen raw vectors; refit transforms on registry meta_train families",
    registry=binding(a.registry),
    existing_index=binding(a.existing_index),
    source_manifests=[binding(x) for x in a.existing_jobs + a.new_jobs],
    existing_views=len(oldrows),
    new_views=len(jobs) - len(oldrows),
    copied_csv_bytes=0,
)
payload = (json.dumps(out, indent=2) + "\n").encode()
dest = a.root / a.output
if dest.exists():
    assert dest.read_bytes() == payload, "Existing cohort manifest changed"
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as f:
        f.write(payload)
print(
    json.dumps(
        dict(
            path=a.output,
            sha256=sha(dest),
            views=len(jobs),
            datasets=len({p["dataset"] for p in prepared.values()}),
            families=len({p["family"] for p in prepared.values()}),
            registry_sha256=sha(a.root / a.registry),
        )
    )
)
