"""Create-only H225 S3 archives with exact-version payload replay verification."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h225_multistep_regret/"
DATA = Path("/home/ubuntu/jope-data/h225")
EVIDENCE = Path("/home/ubuntu/jope_plan/evidence/H225/dsc01-t1405")
BUCKET = "veox-jopedime-learning-use2"
REMOTE = "experiments/rd_breakthrough_20261102/H225/dsc01-t1405/"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def publish(source, stage):
    import boto3

    directory = EVIDENCE / ("custody-" + stage)
    directory.mkdir(exist_ok=False)
    archive = directory / "payload.tar.gz"
    members = []
    with tarfile.open(archive, "w:gz") as tar:
        if stage == "prereg":
            paths = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", source, PREFIX],
                cwd=REPO,
                text=True,
            ).splitlines()
            manifest = json.loads(
                subprocess.check_output(
                    ["git", "show", source + ":" + PREFIX + "manifest.json"], cwd=REPO
                )
            )
            for path in paths + list(manifest["parent_modules"]):
                commit = (
                    source if path.startswith(PREFIX) else manifest["parent_commit"]
                )
                raw = subprocess.check_output(
                    ["git", "show", commit + ":" + path], cwd=REPO
                )
                item = tarfile.TarInfo(path)
                item.size = len(raw)
                item.mode = 0o600
                tar.addfile(item, io.BytesIO(raw))
                members.append(
                    dict(
                        path=path,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                        source_commit=commit,
                    )
                )
        else:
            for path in sorted(DATA.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    name = str(path.relative_to(DATA))
                    tar.add(path, arcname=name, recursive=False)
                    members.append(
                        dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
                    )
    archive_sha = digest(archive)
    # Verify every member's bytes before making an external custody claim.
    with tarfile.open(archive, "r:gz") as tar:
        for item in members:
            raw = tar.extractfile(item["path"]).read()
            if (
                len(raw) != item["bytes"]
                or hashlib.sha256(raw).hexdigest() != item["sha256"]
            ):
                raise RuntimeError("local custody member verification failed")
    s3 = boto3.Session(profile_name="veox-jepson-prod").client(
        "s3", region_name="us-east-2"
    )
    manifest = dict(
        wave="H225",
        stage=stage,
        source_commit=source,
        archive_sha256=archive_sha,
        archive_bytes=archive.stat().st_size,
        members=members,
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipts = []
    for path in (archive, manifest_path):
        key = (
            REMOTE
            + stage
            + "/"
            + source[:12]
            + "-"
            + archive_sha[:16]
            + "/"
            + path.name
        )
        with path.open("rb") as f:
            put = s3.put_object(Bucket=BUCKET, Key=key, Body=f, IfNoneMatch="*")
        version = put.get("VersionId")
        if not version:
            raise RuntimeError("versioned custody required")
        response = s3.get_object(Bucket=BUCKET, Key=key, VersionId=version)
        sha, size = hashlib.sha256(), 0
        with response["Body"] as body:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                sha.update(chunk)
                size += len(chunk)
        if sha.hexdigest() != digest(path) or size != path.stat().st_size:
            raise RuntimeError("exact-VersionId S3 replay differs")
        receipts.append(
            dict(
                uri="s3://" + BUCKET + "/" + key,
                version_id=version,
                sha256=sha.hexdigest(),
                bytes=size,
                replay_verified=True,
            )
        )
    receipt = dict(
        status="REMOTE_VERIFIED",
        wave="H225",
        source_commit=source,
        stage=stage,
        local_archive=str(archive),
        member_count=len(members),
        objects=receipts,
    )
    output = EVIDENCE / (stage.upper() + "-S3-RECEIPT.json")
    with output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--stage", choices=("prereg", "terminal"), required=True)
    args = ap.parse_args()
    if args.stage == "terminal" and not (DATA / "terminal-supervisor.json").is_file():
        raise RuntimeError("worker must be terminal before final archive")
    print(json.dumps(publish(args.source, args.stage)))


if __name__ == "__main__":
    main()
