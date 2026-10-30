"""Bound text receipts under the worker's round lock, without a worker restart.

Only summary.jsonl and direct custody/*.log files are eligible. Full original
bytes remain recoverable through independently checked local zstd and S3 copies.
This is custody maintenance, not telemetry computation or a model input path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path

from dev_incumbent import publish_checkpoint_archive

MIN_FREE_BYTES = 50 * 1024**3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_receipt(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def check_consumers(path, writer_pid=None):
    """Only the round-locked primary may retain an open O_APPEND writer."""
    result = subprocess.run(
        ["lsof", "-w", "-F", "pfa", "--", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise ValueError("consumer census unavailable")
    if result.returncode == 1 and result.stdout.strip():
        raise ValueError("inconsistent consumer census")
    pid, fd = None, None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("f"):
            fd = line[1:]
        elif line.startswith("a"):
            if line[1:] not in ("w", "u") or pid != writer_pid or not fd.isdecimal():
                raise ValueError("uncoordinated reader or writer; leave file unchanged")
            descriptor = Path(f"/proc/{pid}/fd/{fd}")
            info = dict(
                entry.split(":", 1)
                for entry in Path(f"/proc/{pid}/fdinfo/{fd}").read_text().splitlines()
                if ":" in entry
            )
            if not int(info["flags"].strip(), 8) & os.O_APPEND:
                raise ValueError("active writer lacks O_APPEND; safe restart required")
            a, b = descriptor.stat(), path.stat()
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ValueError("writer descriptor changed")


def rotate_file(path, archive_root, keep_rows, s3_prefix, writer_pid=None):
    """Caller holds the same EX lock as every primary round, including prints."""
    path, archive_root = Path(path), Path(archive_root)
    if keep_rows < 1 or path.is_symlink() or not path.is_file():
        raise ValueError("positive tail size and regular nonsymlink file required")
    original_stat = path.stat()
    tail, count, original_hash = deque(maxlen=keep_rows), 0, hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            tail.append(line)
            count += 1
            original_hash.update(line)
    if count <= keep_rows:
        return None
    check_consumers(path, writer_pid)
    existing = archive_root
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < MIN_FREE_BYTES + 2 * original_stat.st_size:
        raise ValueError("insufficient archive/readback headroom; nothing rotated")
    retained = b"".join(tail)  # Includes an unterminated final row, byte for byte.
    archive_root.mkdir(parents=True, exist_ok=True)
    name = f"text-{path.name}-{time.time_ns()}-{uuid.uuid4().hex}"
    archive = archive_root / (name + ".zst")
    with path.open("rb") as source:
        subprocess.run(
            ["zstd", "-T2", "-1", "-o", str(archive)],
            stdin=source,
            capture_output=True,
            timeout=120,
            check=True,
        )
    with archive.open("rb") as stream:
        os.fsync(stream.fileno())
    restored = hashlib.sha256()
    restored_bytes = 0
    with subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as decoder:
        for chunk in iter(lambda: decoder.stdout.read(1024 * 1024), b""):
            restored.update(chunk)
            restored_bytes += len(chunk)
        error = decoder.stderr.read()
        if decoder.wait() != 0:
            raise ValueError(f"zstd restore failed: {error.decode(errors='replace')}")
    expected = original_hash.hexdigest()
    if restored.hexdigest() != expected or restored_bytes != original_stat.st_size:
        raise ValueError("independent decompression differs; nothing rotated")
    manifest = {
        "source": str(path),
        "original_sha256": expected,
        "original_bytes": original_stat.st_size,
        "original_rows": count,
        "original_inode": original_stat.st_ino,
        "original_mode": stat.S_IMODE(original_stat.st_mode),
        "keep_rows": keep_rows,
        "tail_sha256": hashlib.sha256(retained).hexdigest(),
        "tail_bytes": len(retained),
        "archive_sha256": sha(archive),
        "writer_pid": writer_pid,
        "rotator_source_sha256": sha(Path(__file__)),
    }
    manifest_path = archive_root / (name + ".manifest.json")
    write_receipt(manifest_path, manifest)
    remote = publish_checkpoint_archive(archive, s3_prefix)
    remote_manifest = publish_checkpoint_archive(manifest_path, s3_prefix)
    # Network work must never turn an old snapshot into authority to truncate new data.
    check_consumers(path, writer_pid)
    now = path.stat()
    if (
        path.is_symlink()
        or (now.st_dev, now.st_ino, now.st_mtime_ns, now.st_size)
        != (
            original_stat.st_dev,
            original_stat.st_ino,
            original_stat.st_mtime_ns,
            original_stat.st_size,
        )
        or sha(path) != expected
        or sha(archive) != manifest["archive_sha256"]
    ):
        raise ValueError("source/archive changed; nothing rotated")
    receipt_path = archive_root / (name + ".receipt.json")
    receipt = {
        **manifest,
        "manifest": str(manifest_path),
        "archive": str(archive),
        "s3_archive": remote,
        "s3_manifest": remote_manifest,
        "status": "verified_before_rotation",
    }
    write_receipt(receipt_path, receipt)
    # Preserve the inode: existing O_APPEND stdout/stderr follow the shortened tail.
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r+b") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            now.st_dev,
            now.st_ino,
            now.st_size,
            now.st_mtime_ns,
        ):
            raise ValueError("source replaced at rotation boundary")
        stream.write(retained)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    if sha(path) != manifest["tail_sha256"]:
        raise ValueError("local tail differs; recover from verified archive")
    receipt.update(status="rotated", removed_rows=count - keep_rows)
    write_receipt(receipt_path, receipt)
    return receipt


def validate_locations(run, custody, archive_root):
    run, custody, archive_root = Path(run), Path(custody), Path(archive_root)
    if (
        run.is_symlink()
        or custody.is_symlink()
        or not run.is_dir()
        or not custody.is_dir()
    ):
        raise ValueError("real, existing source directories required")
    excluded = [run.resolve(), custody.resolve(), Path(__file__).resolve().parents[2]]
    excluded.extend(
        p for p in Path(__file__).resolve().parents if (p / ".git").is_dir()
    )
    if any(archive_root.resolve().is_relative_to(root) for root in excluded):
        raise ValueError("archives must be outside run, custody and checkout")
    return run, custody, archive_root


def rotate_logs(
    run,
    custody,
    archive_root,
    keep_rows=20,
    writer_pid=None,
    s3_prefix=None,
    max_files=4,
):
    """Caller holds the round lock; only documented allowlisted text can change."""
    if keep_rows < 1 or max_files < 1:
        raise ValueError("positive retention and batch bounds required")
    run, custody, archive_root = validate_locations(run, custody, archive_root)
    outcomes = []
    for path in [run / "summary.jsonl", *sorted(custody.glob("*.log"))]:
        if len(outcomes) >= max_files:
            break
        if not path.exists():
            continue
        try:
            result = rotate_file(path, archive_root, keep_rows, s3_prefix, writer_pid)
            if result:
                outcomes.append(result)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            outcomes.append(
                {"source": str(path), "status": "deferred", "error": str(error)}
            )
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--round-lock", type=Path, required=True)
    parser.add_argument("--writer-pid", type=int, required=True)
    parser.add_argument("--keep-rows", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-files-per-boundary", type=int, default=4)
    parser.add_argument(
        "--s3-prefix",
        default="s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/text-rotation/",
    )
    args = parser.parse_args()
    if (
        args.keep_rows < 1
        or args.interval_seconds <= 0
        or args.max_files_per_boundary < 1
    ):
        parser.error("positive retention and interval required")
    validate_locations(args.run, args.custody, args.archive_root)
    # Companion instances serialize separately before acquiring the existing round lock.
    args.archive_root.mkdir(parents=True, exist_ok=True)
    with (args.archive_root / "rotator.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_start = process_start(args.writer_pid)
        while True:
            with args.round_lock.open("a") as boundary:
                fcntl.flock(boundary, fcntl.LOCK_EX)
                if process_start(args.writer_pid) != expected_start:
                    raise ValueError("primary identity changed; companion exits")
                outcomes = rotate_logs(
                    args.run,
                    args.custody,
                    args.archive_root,
                    args.keep_rows,
                    args.writer_pid,
                    args.s3_prefix,
                    args.max_files_per_boundary,
                )
            if outcomes:
                print(
                    json.dumps({"writer_pid": args.writer_pid, "rotations": outcomes}),
                    flush=True,
                )
            if args.once:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
