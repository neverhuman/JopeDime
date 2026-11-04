from pathlib import Path
import boto3, json, hashlib

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")
M = D / "metadata"
assert json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
c = boto3.Session(profile_name="veox-jepson-prod").client("s3", region_name="us-east-2")
bucket = "veox-jopedime-datasets-use2"
inv = json.loads((M / "registry_inventory.json").read_text())
selinv = json.loads((M / "selected_registry_inventory.json").read_text())
rows = json.loads((M / "selected_catalog.json").read_text())
keys = [
    inv["full-source-registry/"][0]["key"],
    selinv["split-registry/"][0]["key"],
    f"datasets/{rows[0]['id']}/manifest.json",
]
receipts = []
for i, key in enumerate(keys):
    h = c.head_object(Bucket=bucket, Key=key)
    v = h["VersionId"]
    r = c.get_object(Bucket=bucket, Key=key, VersionId=v)
    raw = r["Body"].read()
    r["Body"].close()
    assert r["VersionId"] == v
    path = M / f"schema_probe_{i}.json"
    path.write_bytes(raw)
    receipts.append(
        dict(
            key=key,
            version_id=v,
            sha256=hashlib.sha256(raw).hexdigest(),
            path=str(path.relative_to(D)),
            bytes=len(raw),
            exact_version_get_verified=True,
        )
    )
    # These are structured registry/manifest metadata; print shape, references and names, never dataset bodies.
    print(json.dumps(dict(key=key, metadata=json.loads(raw))), flush=True)
(M / "schema_probe_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
