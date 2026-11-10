"""Exact-version input recovery, no feature decoding."""

from pathlib import Path
import boto3, hashlib, json, tarfile

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h312-astra")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    s = boto3.Session(profile_name="veox-jepson-prod")
    assert s.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = s.client("s3", region_name="us-east-2")
    bucket = "veox-jopedime-learning-use2"
    ref = D / "references"
    ref.mkdir(parents=True, exist_ok=True)
    objects = []
    members = []
    manifest = json.loads(
        (P / "vendor/h310_ensvote2_dev91_vs_r3175/s3_manifest.json").read_text()
    )
    objectspec = [
        dict(x, key="experiments/rd_breakthrough_20261030/H310/" + x["key"])
        for x in manifest["objects"]
    ]
    objectspec.append(
        dict(
            key="experiments/rd_breakthrough_20261030/H281/attempt-001/gpu-attempt-001-r3175-gpu-costs.npz",
            version_id="EgHvHGbSUy6RSXD60ZKiItTdX8HGMysg",
            sha256="d99363f62c36463e6fe1ca825a87332198e6a817021bc208c6379b2d2c330bab",
        )
    )
    for spec in objectspec:
        dest = ref / Path(spec["key"]).name
        assert not dest.exists()
        reply = s3.get_object(
            Bucket=bucket, Key=spec["key"], VersionId=spec["version_id"]
        )
        assert reply["VersionId"] == spec["version_id"]
        with dest.open("xb") as out:
            for chunk in reply["Body"].iter_chunks(1024 * 1024):
                out.write(chunk)
        reply["Body"].close()
        assert sha(dest) == spec["sha256"]
        objects.append(
            dict(spec, path=str(dest.relative_to(D)), bytes=dest.stat().st_size)
        )
        print("verified", dest.name, flush=True)
    archive = ref / "h310_locks_models_outputs.tar.gz"
    with tarfile.open(archive) as tf:
        inventory = [
            dict(name=m.name, bytes=m.size, type=repr(m.type)) for m in tf.getmembers()
        ]
        (P / "h310-archive-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        for m in tf.getmembers():
            # Only records and final locked models; no partial/inner/regression fits needed.
            if not m.isfile() or not (
                m.name.endswith("fit_lock.json")
                and "fold_4" in m.name
                or "/refit/" in m.name
                and "fold_4" in m.name
                or "/out/" in m.name
            ):
                continue
            rel = Path(m.name)
            assert not rel.is_absolute() and ".." not in rel.parts
            dest = ref / "archive" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = tf.extractfile(m).read()
            dest.write_bytes(raw)
            members.append(
                dict(
                    path=str(dest.relative_to(D)),
                    archive_member=m.name,
                    bytes=len(raw),
                    sha256=sha(dest),
                )
            )
    receipt = dict(status="PASS", objects=objects, members=members)
    (P / "source/input_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("members", len(members), flush=True)


if __name__ == "__main__":
    main()
