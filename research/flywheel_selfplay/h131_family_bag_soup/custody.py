"""Create-only H131 numerical evidence custody; source code remains in Git."""

from pathlib import Path
import datetime
import hashlib
import json
import subprocess

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/H131/astra-t1235/"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def put(path, key, custody):
    path, custody = Path(path), Path(custody)
    if path.suffix not in (".json", ".jsonl", ".pt", ".npz") or ".." in Path(key).parts:
        raise ValueError(
            "only numerical artifacts and JSON receipts may be uploaded; no code"
        )
    key = PREFIX + key
    digest = sha(path)
    args = ["aws", "--profile", "veox-jepson-prod", "--no-cli-pager", "s3api"]
    response = json.loads(
        subprocess.check_output(
            args
            + [
                "put-object",
                "--bucket",
                BUCKET,
                "--key",
                key,
                "--body",
                str(path),
                "--if-none-match",
                "*",
                "--metadata",
                json.dumps(dict(sha256=digest)),
            ]
        )
    )
    version = response["VersionId"]
    restored = custody / "readback" / key.removeprefix(PREFIX)
    restored.parent.mkdir(parents=True, exist_ok=True)
    if restored.exists():
        raise RuntimeError("readback collision")
    result = json.loads(
        subprocess.check_output(
            args
            + [
                "get-object",
                "--bucket",
                BUCKET,
                "--key",
                key,
                "--version-id",
                version,
                str(restored),
            ]
        )
    )
    if (
        result["VersionId"] != version
        or sha(restored) != digest
        or restored.stat().st_size != path.stat().st_size
    ):
        raise RuntimeError("S3 exact-version restoration mismatch")
    return dict(
        s3_uri=f"s3://{BUCKET}/{key}",
        version_id=version,
        sha256=digest,
        bytes=path.stat().st_size,
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
