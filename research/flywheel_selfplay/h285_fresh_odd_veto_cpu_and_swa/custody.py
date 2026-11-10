"""Create-only S3 evidence with exact-version readback and payload SHA validation."""

from pathlib import Path
import hashlib
import io
import tarfile
import boto3
from dependencies import ROOT

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/H285/astra-t2305/"


def put_file(path, key):
    path = Path(path)
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    s3 = boto3.Session(profile_name="veox-jepson-prod").client(
        "s3", region_name="us-east-2"
    )
    key = PREFIX + key
    v = s3.put_object(
        Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": digest}
    )["VersionId"]
    z = s3.get_object(Bucket=BUCKET, Key=key, VersionId=v)
    restored = z["Body"].read()
    z["Body"].close()
    assert z["VersionId"] == v and hashlib.sha256(restored).hexdigest() == digest
    return dict(
        uri="s3://" + BUCKET + "/" + key,
        version_id=v,
        sha256=digest,
        bytes=len(body),
        exact_version_get_verified=True,
    )


def archive(files, path, key):
    members = []
    with tarfile.open(path, "w:gz") as tf:
        for f in sorted(files):
            raw = f.read_bytes()
            name = str(f.relative_to(ROOT))
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
            members.append(
                dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            )
    receipt = put_file(path, key)
    s3 = boto3.Session(profile_name="veox-jepson-prod").client("s3")
    z = s3.get_object(
        Bucket=BUCKET,
        Key=receipt["uri"].split("/", 3)[3],
        VersionId=receipt["version_id"],
    )
    with tarfile.open(fileobj=io.BytesIO(z["Body"].read()), mode="r:gz") as tf:
        assert sorted(tf.getnames()) == sorted(m["path"] for m in members)
        for m in members:
            assert (
                hashlib.sha256(tf.extractfile(m["path"]).read()).hexdigest()
                == m["sha256"]
            )
    receipt.update(members=members, members_verified=len(members))
    return receipt
