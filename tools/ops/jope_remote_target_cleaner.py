#!/usr/bin/env python3
"""Remove only owner-approved shared Cargo target output when idle."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import socket
import stat
import sys
from pathlib import Path
from typing import Any

MIN_FREE_BYTES_BY_HOST = {"xbabe1": 120 * 1024**3, "xbabe3": 60 * 1024**3}
TARGET = Path("/home/ubuntu/remote_tests/target")
SCRATCH_ROOT = Path("/home/ubuntu/remote_tests")
AUDIT = Path("/var/log/jope-diskd-xbabe3/remote-target-cleaner.jsonl")
LOCK = Path("/run/lock/jope-remote-target-cleaner.lock")
DENIED_PARTS = {
    "checkpoint", "checkpoints", "dataset", "datasets", "evidence", "flywheel",
    "holdout", "jarambit_runs", "jope-evidence", "receipt", "receipts", "sealed",
}
DENIED_SUFFIXES = {
    ".csv", ".db", ".html", ".jsonl", ".md", ".npz", ".parquet", ".png",
    ".pt", ".pth", ".sqlite", ".svg", ".txt", ".yaml", ".yml",
}
ROOT_FILES = {"CACHEDIR.TAG", ".rustc_info.json"}
BUILD_DIRS = {".fingerprint", "build", "deps", "examples", "incremental", "tmp"}


def under(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def cargo_output_candidates(target: Path, owner_uid: int) -> list[tuple[Path, int, int]] | None:
    """Select only known Cargo output subtrees; retain target/tmp and unknown directories."""
    try:
        root_stat = target.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != owner_uid:
            return None
        root_device = root_stat.st_dev
        if target.resolve(strict=True) != target:
            return None
        if not any((target / name).is_file() for name in ROOT_FILES):
            return None
        candidates: list[tuple[Path, int, int]] = []
        for profile in ("debug", "release"):
            profile_root = target / profile
            try:
                profile_stat = profile_root.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(profile_stat.st_mode) or profile_stat.st_uid != owner_uid or profile_stat.st_dev != root_device:
                continue
            for name in sorted(BUILD_DIRS):
                candidate = profile_root / name
                try:
                    candidate_stat = candidate.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(candidate_stat.st_mode) or candidate_stat.st_uid != owner_uid or candidate_stat.st_dev != root_device:
                    continue
                inode_counts: dict[tuple[int, int], tuple[int, int]] = {}
                allocated = 0
                files_count = 0
                safe = True
                for directory, dirs, files in os.walk(candidate, topdown=True, followlinks=False):
                    parent = Path(directory)
                    if any(part.lower() in DENIED_PARTS for part in parent.relative_to(target).parts):
                        safe = False
                        break
                    parent_stat = parent.lstat()
                    if not stat.S_ISDIR(parent_stat.st_mode) or parent_stat.st_uid != owner_uid or parent_stat.st_dev != root_device:
                        safe = False
                        break
                    for child_name in dirs:
                        child = parent / child_name
                        child_stat = child.lstat()
                        if (
                            not stat.S_ISDIR(child_stat.st_mode)
                            or child_stat.st_uid != owner_uid
                            or child_stat.st_dev != root_device
                            or child_name.lower() in DENIED_PARTS
                        ):
                            safe = False
                            break
                    if not safe:
                        break
                    for child_name in files:
                        child = parent / child_name
                        item = child.lstat()
                        lowered = child_name.lower()
                        suffix = child.suffix.lower()
                        relative_parts = child.relative_to(target).parts
                        if (
                            not stat.S_ISREG(item.st_mode)
                            or item.st_uid != owner_uid
                            or item.st_dev != root_device
                            or any(part.lower() in DENIED_PARTS for part in relative_parts)
                            or any(term in lowered for term in ("checkpoint", "receipt", "evidence", "sealed", "holdout"))
                            or suffix in DENIED_SUFFIXES
                            or (suffix == ".json" and ".fingerprint" not in relative_parts)
                        ):
                            safe = False
                            break
                        key = (item.st_dev, item.st_ino)
                        found, expected = inode_counts.get(key, (0, item.st_nlink))
                        inode_counts[key] = (found + 1, expected)
                        allocated += int(getattr(item, "st_blocks", 0)) * 512 or item.st_size
                        files_count += 1
                    if not safe:
                        break
                if safe and all(found == expected for found, expected in inode_counts.values()):
                    candidates.append((candidate, allocated, files_count))
        return candidates
    except OSError:
        return None


def process_references(target: Path, proc_root: Path = Path("/proc")) -> bool | None:
    """Return busy/clear, or None when any process identity/handle scan is incomplete."""
    if not proc_root.is_dir():
        return None
    try:
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            try:
                status_text = (process / "status").read_text(encoding="utf-8")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                return None
            if not any(line.startswith("Uid:") for line in status_text.splitlines()):
                return None
            try:
                cwd = Path(os.readlink(process / "cwd"))
                environ = (process / "environ").read_bytes().split(b"\0")
                fd_list = list((process / "fd").iterdir())
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                return None
            except OSError:
                return None
            if under(target, cwd):
                return True
            for entry in environ:
                if not entry.startswith(b"CARGO_TARGET_DIR="):
                    continue
                raw = os.fsdecode(entry.split(b"=", 1)[1])
                cargo_target = Path(raw).expanduser()
                if not cargo_target.is_absolute():
                    continue
                if under(target, cargo_target) or under(cargo_target, target):
                    return True
            for fd in fd_list:
                try:
                    opened = Path(os.readlink(fd).removesuffix(" (deleted)"))
                    if opened.is_absolute() and under(target, opened):
                        return True
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except PermissionError:
                    return None
                except OSError:
                    return None
        return False
    except (OSError, RuntimeError):
        return None


def append_audit(event: dict[str, Any], audit: Path = AUDIT) -> None:
    audit.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    event = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        **event,
    }
    with audit.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(*, apply: bool, target: Path = TARGET, scratch_root: Path = SCRATCH_ROOT,
        audit: Path | None = None, lock: Path = LOCK, proc_root: Path = Path("/proc"),
        owner_uid: int = 1000, host: str | None = None) -> int:
    actual_host = host or socket.gethostname().split(".", 1)[0]
    minimum_free_bytes = MIN_FREE_BYTES_BY_HOST.get(actual_host)
    if minimum_free_bytes is None:
        print(f"SKIP: cleaner is not approved for {actual_host}")
        return 0
    if target != TARGET or scratch_root != SCRATCH_ROOT:
        print("SKIP: target path does not match the owner-approved shared Cargo target")
        return 0
    audit = audit or Path(f"/var/log/jope-diskd-{actual_host}/remote-target-cleaner.jsonl")
    try:
        item = target.lstat()
    except FileNotFoundError:
        print(f"OK: target is absent: {target}")
        return 0
    except OSError as exc:
        print(f"SKIP: cannot inspect target: {exc}")
        return 0
    if not stat.S_ISDIR(item.st_mode) or item.st_uid != owner_uid or target.resolve(strict=True) != target:
        print(f"SKIP: target is not an owned canonical directory: {target}")
        return 0
    free_bytes = shutil.disk_usage(scratch_root).free
    if free_bytes >= minimum_free_bytes:
        print(f"OK: free space {free_bytes} is at/above the {minimum_free_bytes}-byte threshold")
        return 0
    candidates = cargo_output_candidates(target, owner_uid)
    if candidates is None:
        print("SKIP: target root is not a canonical owned Cargo target")
        return 0
    if not candidates:
        print("OK: no verified Cargo output subtrees; retaining target root and non-Cargo contents")
        return 0
    try:
        lock.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with lock.open("a", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            references = process_references(target, proc_root)
            if references is not False:
                print("SKIP: a process reference, open handle, or process metadata is present/unknown")
                return 0
            if not apply:
                for path, allocated, files in candidates:
                    print(f"WOULD REMOVE {path} bytes={allocated} files={files} free_bytes={free_bytes}")
                return 0
            for candidate, allocated, files in candidates:
                refreshed = cargo_output_candidates(target, owner_uid)
                references = process_references(target, proc_root)
                current = next((entry for entry in refreshed or [] if entry[0] == candidate), None)
                if current is None or references is not False:
                    print(f"SKIP {candidate}: target or process state changed before removal")
                    continue
                _path, allocated, files = current
                intent = {
                    "event": "delete_intent",
                    "path": str(candidate),
                    "bytes_estimate": allocated,
                    "files": files,
                    "mode": "rebuildable-only",
                    "reason": f"disk_free_below_{minimum_free_bytes}_bytes_and_no_process_references",
                }
                append_audit(intent, audit)
                print(f"REMOVE {candidate} bytes_estimate={allocated} files={files}; audit={audit}", flush=True)
                try:
                    shutil.rmtree(candidate)
                except OSError as exc:
                    append_audit({**intent, "event": "delete_failed", "error": str(exc)}, audit)
                    raise
                append_audit({**intent, "event": "deleted"}, audit)
            return 0
    except BlockingIOError:
        print("SKIP: another remote target cleaner run is active")
        return 0
    except OSError as exc:
        print(f"ERROR: cleanup failed: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="remove only the verified idle Cargo target")
    args = parser.parse_args(argv)
    return run(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
