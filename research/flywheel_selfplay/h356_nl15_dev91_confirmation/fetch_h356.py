"""Exact-version recovery of H352 models and only the pinned dev91 input members."""

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
import boto3
from common import D, E, S, durable, load, prereg, rows, sha, utc


class DigestReader:
    def __init__(self, body):
        self.body = body
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, n=-1):
        b = self.body.read(n)
        self.digest.update(b)
        self.bytes += len(b)
        return b


def get_file(client, spec, dest=None):
    reply = client.get_object(
        Bucket=spec["bucket"], Key=spec["key"], VersionId=spec["version_id"]
    )
    assert reply["VersionId"] == spec["version_id"]
    reader = DigestReader(reply["Body"])
    out = Path(dest).open("xb") if dest is not None else None
    try:
        while True:
            b = reader.read(1 << 20)
            if not b:
                break
            if out is not None:
                out.write(b)
    finally:
        if out is not None:
            out.close()
        reply["Body"].close()
    assert reader.digest.hexdigest() == spec["sha256"]
    assert not spec.get("bytes") or reader.bytes == spec["bytes"]
    return dict(spec, bytes=reader.bytes, exact_version_get_sha256_verified=True)


class RemoteZip(io.RawIOBase):
    """Seekable view of one immutable version; ZipFile requests only selected members."""

    def __init__(self, client, spec):
        self.client, self.spec, self.pos = client, spec, 0
        self.ranges = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset
            if whence == 0
            else (self.pos if whence == 1 else self.spec["bytes"]) + offset
        )
        assert 0 <= self.pos <= self.spec["bytes"]
        return self.pos

    def read(self, size=-1):
        size = (
            self.spec["bytes"] - self.pos
            if size < 0
            else min(size, self.spec["bytes"] - self.pos)
        )
        if size == 0:
            return b""
        assert size <= 32_000_000, "Unexpected bulk member request"
        start = self.pos
        r = self.client.get_object(
            Bucket=self.spec["bucket"],
            Key=self.spec["key"],
            VersionId=self.spec["version_id"],
            Range=f"bytes={start}-{start + size - 1}",
        )
        assert r["VersionId"] == self.spec["version_id"]
        raw = r["Body"].read()
        r["Body"].close()
        assert len(raw) == size
        self.pos += size
        self.ranges.append(
            dict(lo=start, hi=self.pos, sha256=hashlib.sha256(raw).hexdigest())
        )
        return raw


def model_part(client, part):
    reply = client.get_object(
        Bucket=part["bucket"], Key=part["key"], VersionId=part["version_id"]
    )
    assert reply["VersionId"] == part["version_id"]
    reader = DigestReader(reply["Body"])
    seen = set()
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if member.name not in part["members"]:
                    continue
                pin = part["members"][member.name]
                assert (
                    member.isfile()
                    and member.size == pin["bytes"]
                    and member.name not in seen
                )
                rel = Path(member.name).relative_to("h352-jrb01")
                assert not rel.is_absolute() and ".." not in rel.parts
                assert rel.parts[0] in ("P0", "P1", "P2") and (
                    rel.name == "fit_lock.json" or rel.parent.name == "refit"
                )
                raw = tar.extractfile(member).read()
                assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
                dest = D / "models" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as out:
                    out.write(raw)
                seen.add(member.name)
        while reader.read(1 << 20):
            pass
        assert seen == set(part["members"])
        assert (
            reader.digest.hexdigest() == part["sha256"]
            and reader.bytes == part["bytes"]
        )
    finally:
        reply["Body"].close()
    return dict(
        key=part["key"],
        version_id=part["version_id"],
        sha256=part["sha256"],
        selected_members=len(seen),
        exact_version_get_sha256_verified=True,
        every_selected_member_verified=True,
    )


def snapshot(client, spec):
    path = Path(spec["retained_readonly_source"])
    assert not path.is_symlink() and sha(path) == spec["sha256"]
    with path.open("rb") as f:
        reply = client.put_object(
            Bucket=spec["bucket"],
            Key=spec["snapshot_key"],
            Body=f,
            ContentLength=spec["bytes"],
            IfNoneMatch="*",
            Metadata={
                "sha256": spec["sha256"],
                "wave": "H356",
                "provenance": "retained-hash-pinned-S3-derived-copy",
            },
        )
    version = reply.get("VersionId")
    assert version and version != "null"
    pin = dict(
        bucket=spec["bucket"],
        key=spec["snapshot_key"],
        version_id=version,
        sha256=spec["sha256"],
        bytes=spec["bytes"],
        provenance=spec["provenance"],
    )
    durable(D / ("SNAPSHOT-PUT-" + spec["name"] + ".json"), pin, exclusive=True)
    assert sha(path) == spec["sha256"]
    return get_file(client, pin, D / "input" / spec["name"])


def main():
    publication = prereg()
    durable(
        D / "FETCH-STARTED.json",
        dict(utc=utc(), prereg=publication["head"]),
        exclusive=True,
    )
    pins = load(S / "input_pins.json")
    inp = D / "input"
    inp.mkdir()
    session = boto3.Session(profile_name="veox-jepson-prod")
    assert session.client("sts").get_caller_identity()["Account"] == "714044927654"
    client = session.client("s3", region_name="us-east-2")
    parts = []
    for p in pins["h352_model_parts"]:
        parts.append(model_part(client, p))
        print(json.dumps(dict(stage="frozen_models", parts=len(parts))), flush=True)
    assert len(list((D / "models").glob("P*/fold_*/root_*/fit_lock.json"))) == 36
    assert len(list((D / "models").glob("P*/fold_*/root_*/refit/*.txt"))) == 1080
    # Hash the complete immutable archive without decoding unselected members.
    numeric = get_file(client, pins["numeric_archive"])
    remote = RemoteZip(client, pins["numeric_archive"])
    extracted = []
    wanted = {
        f"jope-data/h50/input/pools/{r['context_sha256']}.npz": (
            inp / "pools" / f"{r['context_sha256']}.npz",
            r["pool_sha256"],
        )
        for r in rows()
    }
    wanted.update(
        {
            "jope-data/h58/cache/context_features.npz": (
                inp / "context_features.npz",
                pins["context_features_sha256"],
            ),
            "jope-data/h50/input/selected.json": (
                inp / "selected.json",
                pins["selected_sha256"],
            ),
            "jope-data/h144/inputs/best_dev_checkpoint.json": (
                inp / "best_dev_checkpoint.json",
                pins["r882_reference_sha256"],
            ),
        }
    )
    with zipfile.ZipFile(remote) as z:
        assert set(wanted) <= set(z.namelist())
        for name, (dest, digest) in wanted.items():
            assert not name.endswith(".pt")
            raw = z.read(name)
            assert hashlib.sha256(raw).hexdigest() == digest, name
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as out:
                out.write(raw)
            extracted.append(dict(member=name, sha256=digest, bytes=len(raw)))
    direct = get_file(client, pins["r3175_costs"], inp / "r3175-costs.npz")
    snapshots = [snapshot(client, q) for q in pins["retained_input_snapshots"]]
    receipt = dict(
        status="PASS",
        utc=utc(),
        prereg=publication["head"],
        frozen_model_parts=parts,
        models=1080,
        fit_locks=36,
        numeric_archive=numeric,
        numeric_members=extracted,
        numeric_archive_ranges=remote.ranges,
        direct_reference=direct,
        retained_input_copies=snapshots,
        dev91_pool_members=91,
        numpy_payload_members_decoded=0,
        neural_checkpoint_members_extracted=0,
        model_fits=0,
        other_DLVal_holdout_sealed_payload_members_decoded=0,
    )
    durable(D / "FETCH-PASS.json", receipt)
    durable(E / "FETCH-PASS.json", receipt)
    print(
        json.dumps(
            dict(
                status="PASS",
                models=1080,
                locks=36,
                dev91_pools=91,
                decoded_numpy_members=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
