"""Create-only S3 recovery before reclaiming this worker's completed inner models."""

from pathlib import Path
import hashlib, json, tarfile, io, boto3
from runtime import D, durable, sha

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/H327/astra-20261104T170100Z/"


def preserve_cell(tdir):
    files = sorted(p for p in tdir.rglob("*") if p.is_file())
    assert all(p.suffix in [".txt", ".json", ".jsonl", ".npz"] for p in files)
    archive = tdir / "models.tar.gz"
    members = {str(p.relative_to(tdir)): sha(p) for p in files}
    with tarfile.open(archive, "w:gz", compresslevel=1) as tf:
        for f in files:
            tf.add(f, arcname=str(f.relative_to(tdir)), recursive=False)
    session = boto3.Session(profile_name="veox-jepson-prod")
    assert session.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = session.client("s3", region_name="us-east-2")
    key = PREFIX + "models/" + "-".join(tdir.relative_to(D).parts) + ".tar.gz"
    digest = sha(archive)
    with archive.open("rb") as f:
        r = s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=f,
            ContentLength=archive.stat().st_size,
            IfNoneMatch="*",
            Metadata={"sha256": digest},
        )
    version = r["VersionId"]
    r = s3.get_object(Bucket=BUCKET, Key=key, VersionId=version)
    assert r["VersionId"] == version
    raw = r["Body"].read()
    r["Body"].close()
    assert hashlib.sha256(raw).hexdigest() == digest
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        assert sorted(tf.getnames()) == sorted(members)
        for name, h in members.items():
            assert hashlib.sha256(tf.extractfile(name).read()).hexdigest() == h
    receipt = dict(
        bucket=BUCKET,
        key=key,
        version_id=version,
        sha256=digest,
        bytes=archive.stat().st_size,
        exact_version_get_verified=True,
        members=members,
        members_verified=len(members),
        deleted_inner=[],
    )
    for f in files:
        if f.parent.name.startswith("inner_tau"):
            assert sha(f) == members[str(f.relative_to(tdir))]
            receipt["deleted_inner"].append(str(f.relative_to(tdir)))
            f.unlink()
    archive.unlink()
    durable(tdir / "model_archive.json", receipt)
    return {k: v for k, v in receipt.items() if k not in ["members", "deleted_inner"]}
