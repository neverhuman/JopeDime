"""Fixed H356 identities, source admission, and durable one-shot markers."""

from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

S = Path(__file__).resolve().parent
D = S.parent
E = Path("/home/ubuntu/jope_plan/evidence/H356/jrb01-20261105-t0935")
R = Path("/home/ubuntu/JopeDime")
STOP = datetime.datetime(2026, 11, 5, 12, 15, tzinfo=datetime.timezone.utc)
CLOSE = datetime.datetime(2026, 11, 5, 12, 30, tzinfo=datetime.timezone.utc)
PARTS = ("P0", "P1", "P2")
ROOTS = (30001, 30002, 30003)
DOPE = dict(
    composite_sha256_prefix="c51a56d0", loaded=False, scope="CPU selection layer only"
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def durable(path, obj, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path if exclusive else path.with_suffix(".new")
    with tmp.open("x" if exclusive else "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if not exclusive:
        os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prereg():
    p = load(E / "PUBLISHED-prereg.json")
    assert p["published"]
    subprocess.run(
        [
            "git",
            "-C",
            str(R),
            "merge-base",
            "--is-ancestor",
            p["head"],
            "origin/v1/next",
        ],
        check=True,
    )
    m = load(S / "manifest.json")
    for name, digest in m["source_sha256"].items():
        assert sha(S / name) == digest, name
    assert sha(S / "PREREG.md") == m["prereg_sha256"]
    assert datetime.datetime.now(datetime.timezone.utc) < STOP
    return p


def rows():
    result = load(S / "dev91_contexts.json")["contexts"]
    assert len(result) == 91 and len({r["family_id"] for r in result}) == 19
    assert len({r["context_sha256"] for r in result}) == 91
    return result


def fold(part, family):
    salt = {
        "P0": b"H302-outer-v1\0",
        "P1": b"H308-outer-v1\0",
        "P2": b"H308-outer-v2\0",
    }[part]
    return hashlib.sha256(salt + family.encode()).digest()[-1] % 4


def require_cid(cid, member=None, outcomes_locked=False):
    assert cid in {r["context_sha256"] for r in rows()}, "Context outside pinned dev91"
    if member is not None:
        assert member == "program" or (
            outcomes_locked and member in ("fitness", "gate")
        ), "Outcome before pick lock or forbidden pool member"


def install_read_guard():
    import sys

    def hook(event, args):
        if (
            event != "open"
            or not args
            or not isinstance(args[0], (str, bytes, os.PathLike))
        ):
            return
        path = os.fsdecode(args[0])
        lower = path.lower()
        assert not any(
            x in lower
            for x in (
                "/sealed",
                "/holdout",
                "meta_test",
                "meta-test",
                "fresh_cohort",
                "fresh-cohort",
                "dl-test",
                "dl_test",
            )
        )
        assert not lower.endswith((".pt", ".csv", ".parquet")), (
            "Neural checkpoint or raw dataset open forbidden"
        )
        if "/home/ubuntu/jope-data/" in path:
            assert Path(path).is_relative_to(D), (
                "Foreign data path forbidden in score worker"
            )

    sys.addaudithook(hook)
