"""Allowlisted metadata JSON recovery; data blobs and raw registry CSVs are never fetched."""

from pathlib import Path
import sys, os, json, hashlib, boto3
from concurrent.futures import ThreadPoolExecutor

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")
M = D / "metadata"
BUCKET = "veox-jopedime-datasets-use2"


def allowed(spec, meta_hashes, selected_ids):
    key = spec["key"]
    if key.startswith("blobs/sha256/"):
        return key.split("/")[-1] in meta_hashes and spec["kind"] == "meta"
    if key.startswith("full-source-registry/v1/datasets/"):
        return key.endswith(".json") and spec["kind"] == "source_registry"
    if key.startswith("split-registry/v1/datasets/"):
        return (
            key.endswith(".json")
            and key.split("/")[3] in selected_ids
            and spec["kind"] == "split_registry"
        )
    return False


def install_guard():
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).lower()
            if any(
                x in path
                for x in [
                    "/references/",
                    "/h50/input/pools/",
                    "/h58/cache/",
                    "/sealed",
                    "/holdout",
                    "meta_test",
                    "meta-test",
                    "fresh_cohort",
                    "fresh-cohort",
                ]
            ) or path.endswith((".csv", ".parquet", ".pt")):
                raise RuntimeError("metadata-only read refused: " + path)

    sys.addaudithook(hook)


def main():
    assert (
        json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
    )
    assert (
        json.loads((P / "prereg-published.json").read_text())["head"]
        != json.loads((P / "initial-prereg-published.json").read_text())["head"]
    )
    install_guard()
    cat = json.loads((M / "selected_catalog.json").read_text())
    ids = {r["id"] for r in cat}
    meta = {r["files"]["meta"]["sha256"] for r in cat if r["files"].get("meta")}
    forbidden = {r["files"][part]["sha256"] for r in cat for part in ["train", "test"]}
    assert meta.isdisjoint(forbidden)
    inv = json.loads((M / "registry_inventory.json").read_text())
    specs = []
    for sha in sorted(meta):
        specs.append(dict(key="blobs/sha256/" + sha, kind="meta", expected_sha256=sha))
    for entry in inv["full-source-registry/"]:
        if entry["key"].startswith("full-source-registry/v1/datasets/") and entry[
            "key"
        ].endswith(".json"):
            specs.append(dict(key=entry["key"], kind="source_registry"))
    for entry in inv["split-registry/"]:
        parts = entry["key"].split("/")
        if (
            len(parts) > 4
            and parts[2] == "datasets"
            and parts[3] in ids
            and entry["key"].endswith(".json")
        ):
            specs.append(dict(key=entry["key"], kind="split_registry"))
    assert all(allowed(s, meta, ids) for s in specs)
    (M / "metadata_fetch_plan.json").write_text(
        json.dumps(
            dict(
                specs=specs,
                allowed_meta_hashes=sorted(meta),
                denied_data_hashes=sorted(forbidden),
            ),
            indent=2,
        )
        + "\n"
    )
    s = boto3.Session(profile_name="veox-jepson-prod")
    assert s.client("sts").get_caller_identity()["Account"] == "714044927654"
    client = s.client("s3", region_name="us-east-2")
    out = M / "objects"
    out.mkdir(exist_ok=True)

    def fetch(spec):
        key = spec["key"]
        assert allowed(spec, meta, ids)
        h = client.head_object(Bucket=BUCKET, Key=key)
        version = h["VersionId"]
        assert h["ContentLength"] < 4_000_000
        r = client.get_object(Bucket=BUCKET, Key=key, VersionId=version)
        raw = r["Body"].read()
        r["Body"].close()
        assert r["VersionId"] == version
        digest = hashlib.sha256(raw).hexdigest()
        assert not spec.get("expected_sha256") or spec["expected_sha256"] == digest
        obj = json.loads(raw)
        assert isinstance(obj, dict)
        path = out / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        path.write_bytes(raw)
        return dict(
            spec,
            bucket=BUCKET,
            version_id=version,
            sha256=digest,
            bytes=len(raw),
            path=str(path.relative_to(D)),
            exact_version_get_verified=True,
            top_level_keys=sorted(obj),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, specs))
    (M / "metadata_object_receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                objects=len(receipts),
                bytes=sum(r["bytes"] for r in receipts),
                by_kind={
                    k: sum(r["kind"] == k for r in receipts)
                    for k in ["meta", "source_registry", "split_registry"]
                },
                sample_keys={
                    k: next(
                        (r["top_level_keys"] for r in receipts if r["kind"] == k), None
                    )
                    for k in ["meta", "source_registry", "split_registry"]
                },
                data_blobs_fetched=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
