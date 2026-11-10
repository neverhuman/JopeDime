"""Recover only frozen BASE evidence from the exact-version H318 archive."""

from pathlib import Path
import json, hashlib, tarfile, datetime
import boto3

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h339-astra")
B = "veox-jopedime-learning-use2"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    D.mkdir()
    (D / "references").mkdir()
    pin = json.loads((P / "source/base_archive_pin.json").read_text())
    session = boto3.Session(profile_name="veox-jepson-prod")
    assert session.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = session.client("s3", region_name="us-east-2")
    key = pin["uri"].split(B + "/", 1)[1]
    reply = s3.get_object(Bucket=B, Key=key, VersionId=pin["version_id"])
    assert reply["VersionId"] == pin["version_id"]
    path = D / "references/h318_evidence.tar.gz"
    with path.open("xb") as f:
        for b in reply["Body"].iter_chunks(2**20):
            f.write(b)
    reply["Body"].close()
    assert sha(path) == pin["sha256"]
    members = []
    want = {m["path"]: m for m in pin["members"]}
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n = m.name
            use = (
                n.startswith("outputs/P")
                and (
                    "/refit/" in n
                    or n.endswith("/fit_lock.json")
                    or n.endswith("/outer_once.json")
                )
            ) or n in [
                "outputs/out/panel.npz",
                "source/score_receipt.json",
                "source/regression_receipt.json",
            ]
            if not use:
                continue
            assert ".." not in Path(n).parts and not Path(n).is_absolute()
            raw = tf.extractfile(m).read()
            h = hashlib.sha256(raw).hexdigest()
            assert h == want[n]["sha256"]
            rel = n.removeprefix("outputs/")
            out = D / "references" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            members.append(
                dict(
                    path=str(out.relative_to(D)),
                    archive_path=n,
                    sha256=h,
                    bytes=len(raw),
                    source_archive=key,
                )
            )
    assert len([x for x in members if x["path"].endswith("/fit_lock.json")]) == 36
    obj = dict(
        bucket=B,
        key=key,
        version_id=pin["version_id"],
        sha256=pin["sha256"],
        path=str(path.relative_to(D)),
        bytes=path.stat().st_size,
        exact_version_get_verified=True,
        local_archive_removed_after_verified_extraction=True,
    )
    path.unlink()
    (P / "source/input_receipt.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                objects=[obj],
                members=members,
                source_wave="H318",
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(status="PASS", members=len(members))))


if __name__ == "__main__":
    main()
