"""Retrieve exact archived TRAIN references; extract only locks/refits/OOF arrays."""

from pathlib import Path
import hashlib, io, json, tarfile
import boto3

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h311-astra")
BUCKET = "veox-jopedime-learning-use2"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    session = boto3.Session(profile_name="veox-jepson-prod")
    assert session.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = session.client("s3", region_name="us-east-2")
    receipts = []
    refs = [
        (
            "h303",
            "experiments/rd_breakthrough_20261030/H303/h303_models_locks_and_outputs.tar.gz",
            "hshnLPUZuvpHewykQNTh..mWAGu9KGHo",
            "8957cf0e0b55c21231fd3d40d56ce5bdf27aa915dc97786bcbc00dbee782857d",
        ),
        (
            "h308",
            "experiments/rd_breakthrough_20261030/H308/h308_models_locks_and_outputs.tar.gz",
            "kA4PRbNsYKJTjAJIYLTbp0W58dX3qb2L",
            "1304eae8ad9cf55400c6d646cd0994b14bb23296e94a368d5e584f822e1f962f",
        ),
    ]
    for label, key, version, digest in refs:
        reply = s3.get_object(Bucket=BUCKET, Key=key, VersionId=version)
        assert reply["VersionId"] == version
        raw = reply["Body"].read()
        reply["Body"].close()
        assert sha(raw) == digest
        members = []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            names = tf.getnames()
            (P / (label + "-archive-inventory.json")).write_text(
                json.dumps(names, indent=1) + "\n"
            )
            for m in tf.getmembers():
                name = m.name.removeprefix("./")
                parts = Path(name).parts
                if not m.isfile():
                    continue
                wanted = (
                    name.endswith("/fit_lock.json")
                    or ("/refit/" in name and name.endswith(".txt"))
                    or name.endswith("/picks.npz")
                    or name.endswith("/metrics_full.json")
                )
                if not wanted:
                    continue
                assert not Path(name).is_absolute() and ".." not in parts
                b = tf.extractfile(m).read()
                out = D / "references" / label / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b)
                members.append(
                    dict(path=str(out.relative_to(D)), sha256=sha(b), bytes=len(b))
                )
        receipts.append(
            dict(
                bucket=BUCKET,
                key=key,
                version_id=version,
                sha256=digest,
                bytes=len(raw),
                members=members,
                exact_version_get_verified=True,
            )
        )
        del raw
        print(label, len(members), sum(m["bytes"] for m in members), flush=True)
    # Small, explicitly DL-Train H295 panel, used only by the separate regression process.
    manifest = json.loads(
        (P / "vendor/h295_label_free_validity_guard/s3_manifest.json").read_text()
    )
    for item in manifest["objects"]:
        if item["file"] != "outer_cv_panels.npz":
            continue
        reply = s3.get_object(
            Bucket=BUCKET, Key=item["s3_key"], VersionId=item["version_id"]
        )
        b = reply["Body"].read()
        reply["Body"].close()
        assert sha(b) == item["sha256"]
        out = D / "references/h295/outer_cv_panels.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b)
        receipts.append(
            dict(
                bucket=BUCKET,
                key=item["s3_key"],
                version_id=item["version_id"],
                sha256=sha(b),
                bytes=len(b),
                members=[
                    dict(path=str(out.relative_to(D)), sha256=sha(b), bytes=len(b))
                ],
                exact_version_get_verified=True,
            )
        )
    (P / "source/input_receipt.json").write_text(json.dumps(receipts, indent=1) + "\n")


if __name__ == "__main__":
    main()
