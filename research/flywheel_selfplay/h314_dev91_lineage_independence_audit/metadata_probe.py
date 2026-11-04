"""Pin catalog and inventory registry metadata only, without reading outcomes."""

from pathlib import Path
import boto3, json, hashlib

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")


def main():
    assert (
        json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
    )
    s = boto3.Session(profile_name="veox-jepson-prod")
    assert s.client("sts").get_caller_identity()["Account"] == "714044927654"
    c = s.client("s3", region_name="us-east-2")
    b = "veox-jopedime-datasets-use2"
    out = D / "metadata"
    out.mkdir(exist_ok=True)
    key = "catalog/v246/catalog.jsonl"
    version = "eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC"
    r = c.get_object(Bucket=b, Key=key, VersionId=version)
    raw = r["Body"].read()
    r["Body"].close()
    assert r["VersionId"] == version
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "c5f5eb6db36d24daa9abc5c7808a49773a2b94c1765620d236e91a471fe2cb54"
    (out / "catalog.jsonl").write_bytes(raw)
    receipt = dict(
        bucket=b,
        key=key,
        version_id=version,
        sha256=digest,
        bytes=len(raw),
        path="metadata/catalog.jsonl",
        exact_version_get_verified=True,
    )
    (out / "catalog_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    inventory = {}
    for prefix in ["full-source-registry/", "split-registry/"]:
        inventory[prefix] = [
            dict(key=o["Key"], bytes=o["Size"], modified=o["LastModified"].isoformat())
            for pg in c.get_paginator("list_objects_v2").paginate(
                Bucket=b, Prefix=prefix
            )
            for o in pg.get("Contents", [])
        ]
    (out / "registry_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    selected = Path("/home/ubuntu/jope-data/h50/input/selected.json").read_bytes()
    assert (
        hashlib.sha256(selected).hexdigest()
        == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    )
    sel = json.loads(selected)
    safe = {
        split: [
            {
                k: r[k]
                for k in [
                    "context_sha256",
                    "dataset_id",
                    "view_id",
                    "family_id",
                    "task",
                ]
            }
            for r in sel[split]
        ]
        for split in ["train", "dev"]
    }
    (out / "selected_metadata.json").write_text(json.dumps(safe, indent=2) + "\n")
    ids = {r["dataset_id"] for split in safe.values() for r in split}
    cat = [json.loads(x) for x in raw.splitlines()]
    chosen = [x for x in cat if x["id"] in ids]
    (out / "selected_catalog.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                catalog_rows=len(cat),
                selected_ids=len(ids),
                matched_ids=len(chosen),
                first_selected_metadata=chosen[:1],
                registry_examples={p: rows[:4] for p, rows in inventory.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
