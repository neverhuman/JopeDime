"""Resolve pinned lineage/schema metadata and fsync classification before any gain join."""

from pathlib import Path
import sys, os, json, hashlib, datetime, collections
import neighbors as n
from fetch_metadata import install_guard

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")
M = D / "metadata"


def sid(platform, ident):
    return (
        f"{str(platform).strip().casefold()}:{str(ident).strip()}"
        if n.valid(platform) and n.valid(ident)
        else None
    )


def extract(row, meta, receipts, source_registry, split_registry):
    platform = meta.get("source_platform") or row.get("provenance", {}).get("platform")
    prov = row.get("provenance", {})
    source_ids = {
        v
        for v in [
            sid(platform, meta.get("source_dataset_id")),
            sid(prov.get("platform"), prov.get("dataset_id")),
        ]
        if v
    }
    roots = set()
    root_fields = {}
    for key in [
        "source_family",
        "source_path",
        "source_file",
        "source_url",
        "parent_dataset_id",
        "source_table_id",
        "raw_source_id",
        "ingest_root",
        "lineage_root",
        "source_sha256",
        "raw_sha256",
    ]:
        value = meta.get(key)
        if n.valid(value):
            prefix = (
                f"{str(platform).casefold()}:"
                if n.valid(platform)
                and key not in ["source_url", "source_sha256", "raw_sha256"]
                else ""
            )
            token = prefix + key + ":" + str(value).strip()
            if key != "source_file":
                roots.add(token)
            root_fields[key] = value
    if n.valid(prov.get("url")):
        roots.add("source_url:" + str(prov["url"]).strip())
    fingerprints = {
        str(v).strip()
        for v in [prov.get("dedupe_fingerprint"), meta.get("dedupe_fingerprint")]
        if n.valid(v)
    }
    registry_matches = []
    for key, s in source_registry:
        source_id = sid(s.get("source_platform"), s.get("source_dataset_id"))
        if (
            s.get("dataset_hash_legacy") == row["id"]
            or source_id
            and source_id in source_ids
        ):
            registry_matches.append(key)
            if s.get("source", {}).get("sha256"):
                roots.add("raw_sha256:" + s["source"]["sha256"])
            if source_id:
                source_ids.add(source_id)
    # Use the ingested schema and its paired catalog row count. Raw schema/count retained separately.
    headers = meta.get("output_headers")
    columns = (
        n.names(headers)
        if isinstance(headers, list) and all(isinstance(x, str) for x in headers)
        else []
    )
    raw_headers = meta.get("original_headers")
    raw_columns = (
        n.names(raw_headers)
        if isinstance(raw_headers, list)
        and all(isinstance(x, str) for x in raw_headers)
        else []
    )
    nr = row.get("shape", {}).get("rows")
    nf = row.get("shape", {}).get("features")
    task = row.get("task")
    split_matches = [
        dict(
            key=key,
            dataset_canonical_sha256=s.get("dataset_canonical_sha256"),
            partitions={
                k: {f: v.get(f) for f in ["rows", "columns", "membership"]}
                for k, v in s.get("partitions", {}).items()
            },
            policy=s.get("policy"),
        )
        for key, s in split_registry
        if s.get("dataset_hash") == row["id"]
    ]
    for s in split_matches:
        assert s["dataset_canonical_sha256"] == row["canonical_sha256"]
    record = dict(
        dataset_id=row["id"],
        source_ids=sorted(source_ids),
        source_registry_entries=sorted(registry_matches),
        lineage_roots=sorted(roots),
        lineage_field_values=root_fields,
        dedupe_fingerprints=sorted(fingerprints),
        columns=columns,
        column_names_sha256=n.digest(columns) if columns else None,
        n_rows=nr,
        n_features=nf,
        task=task,
        raw_source_n_rows=meta.get("source_rows"),
        raw_column_names_sha256=n.digest(raw_columns) if raw_columns else None,
        raw_n_columns=len(raw_columns) if raw_columns else None,
        schema_origin="files.meta.output_headers paired with catalog.shape.rows/features",
        metadata_sha256=receipts["sha256"],
        metadata_version_id=receipts["version_id"],
        split_registry=split_matches,
    )
    record["missing_fields"] = [
        k
        for k, v in [
            ("source_ids", source_ids),
            ("lineage_roots", roots),
            ("dedupe_fingerprint", fingerprints),
            ("column_names", columns),
            ("n_rows", nr),
            ("n_features", nf),
            ("task", task),
        ]
        if not v
    ]
    record["conflicts"] = dict(
        source_ids_multiple=len(source_ids) > 1,
        fingerprints_multiple=len(fingerprints) > 1,
        feature_count_mismatch=bool(
            isinstance(meta.get("n_features"), int) and nf != meta["n_features"]
        ),
        header_count_mismatch=bool(
            columns and len(headers) != row.get("shape", {}).get("columns")
        ),
    )
    return record


def durable(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def main():
    assert (
        json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
    )
    install_guard()
    receipts = json.loads((M / "metadata_object_receipts.json").read_text())
    objects = {}
    for rec in receipts:
        path = D / rec["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == rec["sha256"]
        objects[rec["key"]] = json.loads(raw)
    bykey = {r["key"]: r for r in receipts}
    cat = json.loads((M / "selected_catalog.json").read_text())
    selected = json.loads((M / "selected_metadata.json").read_text())
    source_registry = [
        (r["key"], objects[r["key"]])
        for r in receipts
        if r["kind"] == "source_registry"
    ]
    split_registry = [
        (r["key"], objects[r["key"]]) for r in receipts if r["kind"] == "split_registry"
    ]
    records = {}
    for row in cat:
        key = "blobs/sha256/" + row["files"]["meta"]["sha256"]
        records[row["id"]] = extract(
            row, objects[key], bykey[key], source_registry, split_registry
        )
    families = {split: collections.defaultdict(set) for split in ["train", "dev"]}
    for split, rows in selected.items():
        for row in rows:
            families[split][row["family_id"]].add(row["dataset_id"])
    assert len(families["train"]) == 205 and len(families["dev"]) == 19
    train = [
        dict(records[ident], family_id=f)
        for f, ids in sorted(families["train"].items())
        for ident in sorted(ids)
    ]
    metadata = {
        split: {
            f: [dict(records[ident], family_id=f) for ident in sorted(ids)]
            for f, ids in sorted(fams.items())
        }
        for split, fams in families.items()
    }
    classifications = {}
    fullpairs = {}
    for f, dev in metadata["dev"].items():
        rec = n.classify(dev, train)
        fullpairs[f] = rec.pop("all_pair_receipts")
        classifications[f] = rec
    durable(M / "family_metadata_full.json", metadata)
    durable(M / "pair_comparisons.json", fullpairs)
    durable(M / "classification.json", classifications)
    compact = {
        split: {
            f: [{k: v for k, v in row.items() if k != "columns"} for row in rows]
            for f, rows in fs.items()
        }
        for split, fs in metadata.items()
    }
    (P / "source/family_metadata.json").write_text(json.dumps(compact, indent=2) + "\n")
    groups = {
        g: sorted(f for f, x in classifications.items() if x["status"] == g)
        for g in ["NEIGHBORED", "CLEAN", "UNKNOWN"]
    }
    rec = dict(
        status="PASS",
        classification_sha256=hashlib.sha256(
            (M / "classification.json").read_bytes()
        ).hexdigest(),
        metadata_sha256=hashlib.sha256(
            (M / "family_metadata_full.json").read_bytes()
        ).hexdigest(),
        pair_comparisons_sha256=hashlib.sha256(
            (M / "pair_comparisons.json").read_bytes()
        ).hexdigest(),
        groups={
            k: dict(count=len(v), ids_sha256=n.digest(v), family_ids=v)
            for k, v in groups.items()
        },
        dataset_records=len(records),
        selected_train_families=205,
        selected_dev_families=19,
        source_registry_entries_read=len(source_registry),
        split_registry_entries_read=len(split_registry),
        metadata_objects_read=len(receipts),
        missing_by_field=dict(
            collections.Counter(
                k for r in records.values() for k in r["missing_fields"]
            )
        ),
        source_registry_datasets_matched=sum(
            bool(r["source_registry_entries"]) for r in records.values()
        ),
        split_registry_datasets_matched=sum(
            bool(r["split_registry"]) for r in records.values()
        ),
        gain_files_opened=0,
        feature_or_label_rows_read=0,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    durable(P / "source/classification_receipt.json", rec)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
