from pathlib import Path
import json, hashlib, tarfile, datetime, os
import boto3

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h318-astra")
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
    s = boto3.Session(profile_name="veox-jepson-prod")
    assert s.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = s.client("s3", region_name="us-east-2")
    objects = []
    members = []
    inventories = {}
    for wave, slug in [
        ("H303", "h303_ensvote2_fully_nested_hparams"),
        ("H308", "h308_ensvote2_trainonly_repartition"),
    ]:
        manifest = json.loads((P / "vendor" / slug / "s3_manifest.json").read_text())
        for rec in manifest["objects"]:
            if rec["key"] not in [
                "picks.npz",
                "metrics_full.json",
                wave.lower() + "_models_locks_and_outputs.tar.gz",
            ]:
                continue
            path = D / "references" / (wave + "_" + rec["key"])
            key = "experiments/rd_breakthrough_20261030/" + wave + "/" + rec["key"]
            reply = s3.get_object(Bucket=B, Key=key, VersionId=rec["version_id"])
            assert reply["VersionId"] == rec["version_id"]
            with path.open("xb") as f:
                for chunk in reply["Body"].iter_chunks(2**20):
                    f.write(chunk)
            reply["Body"].close()
            assert sha(path) == rec["sha256"]
            assert path.stat().st_size == rec["bytes"] if "bytes" in rec else True
            obj = dict(
                bucket=B,
                key=key,
                path=str(path.relative_to(D)),
                version_id=rec["version_id"],
                sha256=rec["sha256"],
                bytes=path.stat().st_size,
                exact_version_get_verified=True,
            )
            objects.append(obj)
            if path.name.endswith(".tar.gz"):
                inventory = []
                with tarfile.open(path, "r|gz") as tf:
                    for mem in tf:
                        if not mem.isfile():
                            continue
                        name = mem.name
                        inventory.append(dict(path=name, bytes=mem.size))
                        # Preserve frozen task locks and refits only; never decode historical models/arrays elsewhere.
                        parts = Path(name).parts
                        if "fold_0" not in parts and not any(
                            q in parts for q in ["fold_1", "fold_2", "fold_3"]
                        ):
                            continue
                        if not (name.endswith("/fit_lock.json") or "/refit/" in name):
                            continue
                        if wave == "H303" and not name.startswith("scratch/"):
                            continue
                        if wave == "H308" and not any(
                            "/" + q + "/" in "/" + name for q in ["P1", "P2"]
                        ):
                            continue
                        assert ".." not in parts and not Path(name).is_absolute()
                        raw = tf.extractfile(mem).read()
                        out = D / "references" / wave / name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(raw)
                        members.append(
                            dict(
                                path=str(out.relative_to(D)),
                                archive_path=name,
                                bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                source_archive=key,
                            )
                        )
                inventories[wave] = inventory
                # Original exact-version archive is its immutable recovery route; no unique source is discarded.
                assert sha(path) == rec["sha256"]
                path.unlink()
                obj["local_archive_removed_after_verified_extraction"] = True
        print(
            wave,
            "recovered",
            sum(x["source_archive"].split("/")[-2] == wave for x in members),
            "members",
            flush=True,
        )
    (P / "source/input_receipt.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                objects=objects,
                members=members,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )
    (P / "archive-input-inventory.json").write_text(
        json.dumps(inventories, indent=2) + "\n"
    )
    print(json.dumps(dict(status="PASS", objects=len(objects), members=len(members))))


if __name__ == "__main__":
    main()
