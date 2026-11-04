"""Run read-only on xbabe1: freeze a complete journal prefix and copy recorded files."""

from pathlib import Path
import json
import tarfile
import io
import hashlib
import sys
import datetime
import os

p = Path("/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030")
jraw = (p / "adapter_admissions.jsonl").read_bytes()
end = jraw.rfind(b"\n") + 1
jraw = jraw[:end]
journal = [json.loads(x) for x in jraw.splitlines()]
latest = max(r["round"] for r in journal if (p / f"round-{r['round']}.json").exists())
wanted = list(range(3175, latest + 1))
paths = [p / f"round-{n}.json" for n in wanted]
missing = [f.name for f in paths if not f.exists()]
paths = [f for f in paths if f.exists()]
extras = [
    p / "best_dev_checkpoint.json",
    Path("/home/ubuntu/jope-data/flywheel-full-20261030/input/selected.json"),
]
size = sum(f.stat().st_size for f in paths + extras) + len(jraw)
assert size < 2 * 1024**3
inventory = dict(
    captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    host=os.uname().nodename,
    run=str(p),
    latest_complete_round=latest,
    journal_prefix_bytes=len(jraw),
    journal_prefix_sha256=hashlib.sha256(jraw).hexdigest(),
    requested_first=3175,
    requested_last=latest,
    missing_round_files=missing,
    files=[],
    read_only=True,
    no_forwards=True,
    no_remote_files_written=True,
)
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tar:
    for f in paths + extras:
        before = f.stat()
        raw = f.read_bytes()
        after = f.stat()
        assert (
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        ), "source changed during snapshot: " + str(f)
        name = "rounds/" + f.name if f in paths else f.name
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        inventory["files"].append(
            dict(
                path=name,
                source=str(f),
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                mtime_ns=before.st_mtime_ns,
            )
        )
    info = tarfile.TarInfo("adapter_admissions.jsonl")
    info.size = len(jraw)
    tar.addfile(info, io.BytesIO(jraw))
    inventory["files"].append(
        dict(
            path=info.name,
            source=str(p / info.name),
            sha256=hashlib.sha256(jraw).hexdigest(),
            bytes=len(jraw),
            prefix=True,
        )
    )
    raw = (json.dumps(inventory, indent=2) + "\n").encode()
    info = tarfile.TarInfo("capture-inventory.json")
    info.size = len(raw)
    tar.addfile(info, io.BytesIO(raw))
