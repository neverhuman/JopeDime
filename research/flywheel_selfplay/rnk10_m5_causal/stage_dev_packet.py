#!/usr/bin/env python3
"""Stream only the pinned DL-Val roster's pool and train-only table payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path, PurePosixPath

from stage_train_packet import CPU_FUSED_SHA256, SOURCE_MANIFEST_SHA256, sha256

SOURCE_ROOT = "flywheel-full-20261030/input"
SELECTED_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
DEV_ROSTER_SHA256 = "eedd7e98eb02e625f75ea71245a32d8765b75610f0b7f56689088a10b0a08193"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-host", default="xbabe1")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    roster_path = Path(__file__).with_name("dev_roster.json")
    if sha256(roster_path) != DEV_ROSTER_SHA256:
        raise SystemExit("committed metadata-only dev roster hash mismatch")
    roster = json.loads(roster_path.read_text())
    if (
        roster.get("schema") != "jope.rnk10-m5-dev-roster/v1"
        or roster.get("payloads_read") != 0
    ):
        raise SystemExit("committed metadata-only dev roster identity mismatch")
    selected_bytes = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.source_host,
            f"cat /home/ubuntu/jope-data/{SOURCE_ROOT}/selected.json",
        ]
    )
    if hashlib.sha256(selected_bytes).hexdigest() != SELECTED_SHA256:
        raise SystemExit("source selected roster hash mismatch")
    selected = json.loads(selected_bytes)
    dev_rows = selected.get("dev")
    if len(dev_rows) != 91:
        raise SystemExit("expected 91 DL-Val rows")
    if len({row.get("context_sha256") for row in dev_rows}) != 91:
        raise SystemExit("duplicate DL-Val context identity")
    pinned = roster["contexts"]
    if [row.get("context_sha256") for row in dev_rows] != [
        row["context_sha256"] for row in pinned
    ]:
        raise SystemExit("remote DL-Val context order differs from committed roster")
    for source, lock in zip(dev_rows, pinned, strict=True):
        if any(source.get(field) != lock.get(field) for field in lock):
            raise SystemExit(
                f"remote DL-Val row differs from committed pin: {lock['context_sha256']}"
            )

    members = [
        f"{SOURCE_ROOT}/source_manifest.json",
        f"{SOURCE_ROOT}/selected.json",
        f"{SOURCE_ROOT}/cpu_exact_fused.py",
    ]
    for row in dev_rows:
        cid = row["context_sha256"]
        if len(cid) != 64 or any(ch not in "0123456789abcdef" for ch in cid):
            raise SystemExit("unsafe context id in pinned roster")
        members.extend(
            [f"{SOURCE_ROOT}/pools/{cid}.npz", f"{SOURCE_ROOT}/tables/{cid}.jtt1"]
        )
    paths = [PurePosixPath(value) for value in members]
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise SystemExit("unsafe archive path")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        args.source_host,
        "tar",
        "-c",
        "-f",
        "-",
        "--directory=/home/ubuntu/jope-data",
        "--files-from=-",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert process.stdin is not None and process.stdout is not None
    writer_errors: list[BaseException] = []

    def send_paths() -> None:
        try:
            process.stdin.write(("\n".join(members) + "\n").encode())
            process.stdin.close()
        except BaseException as error:
            writer_errors.append(error)

    writer = threading.Thread(target=send_paths, daemon=True)
    writer.start()
    expected = set(members)
    seen: set[str] = set()
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            if member.name not in expected or not member.isfile():
                process.kill()
                raise SystemExit(f"unexpected dev packet member: {member.name}")
            relative = PurePosixPath(member.name)
            if relative.parts[:2] != PurePosixPath(SOURCE_ROOT).parts:
                process.kill()
                raise SystemExit("dev packet member escaped pinned source root")
            if member.name in seen:
                process.kill()
                raise SystemExit(f"duplicate dev packet member: {member.name}")
            seen.add(member.name)
            target = destination / Path(*relative.parts[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"missing staged file: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
    process.stdout.close()
    writer.join()
    if writer_errors or process.wait() != 0:
        raise SystemExit("failed to stream pinned DL-Val packet")
    if seen != expected:
        raise SystemExit(
            f"incomplete dev packet: {len(expected - seen)} required members absent"
        )
    if sha256(destination / "source_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise SystemExit("staged source-manifest hash mismatch")
    if sha256(destination / "cpu_exact_fused.py") != CPU_FUSED_SHA256:
        raise SystemExit("staged fused-model source hash mismatch")
    if sha256(destination / "selected.json") != SELECTED_SHA256:
        raise SystemExit("staged selected roster hash mismatch")
    print(
        json.dumps(
            {
                "contexts": 91,
                "payload_scope": "DL-Val rows only; 93 holdout and outer payloads omitted",
                "destination": str(destination),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
