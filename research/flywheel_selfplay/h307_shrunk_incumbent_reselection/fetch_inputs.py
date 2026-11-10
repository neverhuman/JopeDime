"""Read only the exact approved S3 versions; extract only two allowlisted members."""

import hashlib
import json
import tarfile
from pathlib import Path
import boto3

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/H289/astra-t0005/inputs/"
INPUTS = [
    (
        "prepared-closure.tar.gz",
        "dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL",
        "66bc3154f4f2ec49bc9237163712826ec0bda82f32e3ba34c4881ba73b4bd071",
        "history.npz",
        "e6214cd5f3f3322a6ef22bfa7b3f7a797e106869c9195cdfe18d217b6ac7f446",
    ),
    (
        "recorded-snapshot.tar.gz",
        "PyUdeyhv8LqMyLfHB1ondkNF7dITELec",
        "f02363690b6988174afa30d209112501100527f3e8826141efcdc36d5f3dbdce",
        "adapter_admissions.jsonl",
        "7c01de3b03eeb993d67c84266649a732365afdeaa86c593295b73a9d05b0b868",
    ),
]


def fetch(scratch, output):
    scratch.mkdir(exist_ok=True, parents=True)
    session = boto3.Session(profile_name="veox-jepson-prod")
    assert session.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = session.client("s3", region_name="us-east-2")
    receipts = []
    for name, version, digest, member, member_sha in INPUTS:
        p = scratch / name
        r = s3.get_object(Bucket=BUCKET, Key=PREFIX + name, VersionId=version)
        assert r["VersionId"] == version and r["ContentLength"] < 200_000_000
        h = hashlib.sha256()
        with p.open("xb") as f:
            for block in r["Body"].iter_chunks(1024 * 1024):
                h.update(block)
                f.write(block)
        r["Body"].close()
        assert h.hexdigest() == digest
        with tarfile.open(p, "r:gz") as tf:
            found = [
                m for m in tf.getmembers() if Path(m.name).name == member and m.isfile()
            ]
            assert len(found) == 1 and found[0].size < 100_000_000
            raw = tf.extractfile(found[0]).read()
            assert hashlib.sha256(raw).hexdigest() == member_sha
            with (scratch / member).open("xb") as f:
                f.write(raw)
        receipts.append(
            dict(
                bucket=BUCKET,
                key=PREFIX + name,
                version_id=version,
                sha256=digest,
                bytes=p.stat().st_size,
                member=found[0].name,
                member_sha256=member_sha,
                member_bytes=len(raw),
                exact_version_get_verified=True,
            )
        )
    key = "experiments/rd_breakthrough_20261030/H304/metrics_full_h304.json"
    version = "J4bT7je3eRxxiOEKIonPSaRIS0ZLLsHY"
    r = s3.get_object(Bucket=BUCKET, Key=key, VersionId=version)
    raw = r["Body"].read()
    r["Body"].close()
    digest = "60c5f879437bcfe5310b8e6e58cf652f81cd0c5f084dc138468950d0596cd84a"
    assert r["VersionId"] == version and hashlib.sha256(raw).hexdigest() == digest
    (scratch / "h304_reference.json").write_bytes(raw)
    receipts.append(
        dict(
            bucket=BUCKET,
            key=key,
            version_id=version,
            sha256=digest,
            bytes=len(raw),
            exact_version_get_verified=True,
        )
    )
    output.write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    fetch(a.scratch, a.receipt)
