#!/usr/bin/env python3
"""Verify reviewed vendor custody pins without network access or source writes.

PROVENANCE.json is the Git-reviewed trust anchor; this does not authenticate an
unreviewed replacement of both provenance and source, or qualify crate behavior.
A dependency upgrade must update the reviewed inventory and expected identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tomllib

MAX_FILE_BYTES = 5 * 1024 * 1024
PREFIX = "vendor/security/"
PROVENANCE = "PROVENANCE.json"
EXPECTED_CRATES = frozenset(
    [("candle-core", "0.11.0"), ("crossterm", "0.28.1"), ("pulp", "0.22.3"),
     ("ratatui", "0.29.0"), ("tokenizers", "0.22.2"),
     ("tch", "0.26.0"), ("torch-sys", "0.26.0")]
    + [(name, version) for name in
       ("gemm", "gemm-c32", "gemm-c64", "gemm-common", "gemm-f16", "gemm-f32", "gemm-f64")
       for version in ("0.18.2", "0.19.0")]
)
HEX = re.compile(r"[0-9a-f]{64}\Z")
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


class CustodyError(ValueError):
    """A reviewed inventory or its on-disk bytes failed validation."""


def require(condition, message):
    if not condition:
        raise CustodyError(message)


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise CustodyError(f"nonfinite JSON constant: {value}")


def integer(value, label, maximum=None):
    require(type(value) is int and value >= 0, f"invalid {label}")
    require(maximum is None or value <= maximum, f"oversized {label}")
    return value


def digest(value, label):
    require(isinstance(value, str) and HEX.fullmatch(value), f"invalid {label}")
    return value


def relative(value):
    require(isinstance(value, str) and bool(value), "empty/non-string path")
    parts = value.split("/")
    require(not any(part in ("", ".", "..") for part in parts), f"unsafe path: {value}")
    require("\\" not in value and ":" not in value and
            all(ord(char) >= 32 and ord(char) != 127 for char in value), f"unsafe path: {value}")
    require(not PurePosixPath(value).is_absolute(), f"absolute path: {value}")
    return parts


def read_regular(parent_fd, name, label):
    # Anchor reads to directory descriptors and refuse link substitutions, FIFOs,
    # devices and changes while hashing. O_NONBLOCK prevents a raced FIFO hang.
    fd = os.open(name, FILE_FLAGS, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {label}")
        require(before.st_size <= MAX_FILE_BYTES, f"oversized file: {label}")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(65536, MAX_FILE_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            require(size <= MAX_FILE_BYTES, f"oversized file: {label}")
            chunks.append(chunk)
        after = os.fstat(fd)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(signature(before) == signature(after) and size == before.st_size,
                f"file changed while reading: {label}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def inventory(directory_fd):
    files, directories = {}, set()

    def visit(fd, prefix):
        with os.scandir(fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            path = prefix + name
            relative(path)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            require(not stat.S_ISLNK(info.st_mode), f"symlink forbidden: {path}")
            if stat.S_ISDIR(info.st_mode):
                directories.add(path)
                child = os.open(name, DIRECTORY_FLAGS, dir_fd=fd)
                try:
                    visit(child, path + "/")
                finally:
                    os.close(child)
            else:
                data = read_regular(fd, name, path)
                files[path] = (len(data), hashlib.sha256(data).hexdigest())
    visit(directory_fd, "")
    return files, directories


def verify_ratatui_lru(directory_fd, expected):
    # This source-reviewed manifest floor closes both cached lru soundness advisories.
    # Re-read through the existing descriptor custody rules and bind these bytes to
    # the inventory; coherent provenance edits cannot silently downgrade the floor.
    directory = "ratatui-0.29.0"
    fd = os.open(directory, DIRECTORY_FLAGS, dir_fd=directory_fd)
    try:
        raw = read_regular(fd, "Cargo.toml", directory + "/Cargo.toml")
    finally:
        os.close(fd)
    require((len(raw), hashlib.sha256(raw).hexdigest()) == expected[directory + "/Cargo.toml"],
            "ratatui manifest changed during dependency check")
    manifest = tomllib.loads(raw.decode("utf-8"))
    dependencies = manifest.get("dependencies")
    require(isinstance(dependencies, dict) and dependencies.get("lru") == {"version": "=0.18.4"},
            "ratatui lru must retain reviewed =0.18.4 advisory floor without overrides")


def verify(root):
    opened = []
    try:
        fd = os.open(os.fspath(root), DIRECTORY_FLAGS)
        opened.append(fd)
        for component in ("vendor", "security"):
            fd = os.open(component, DIRECTORY_FLAGS, dir_fd=fd)
            opened.append(fd)
        raw = read_regular(fd, PROVENANCE, PREFIX + PROVENANCE)
        data = json.loads(raw, object_pairs_hook=pairs_unique, parse_constant=reject_constant)
        require(isinstance(data, dict), "provenance must be an object")
        require(data.get("schema") == "jope.dependency-source-patch/v1", "unknown provenance schema")
        digest(data.get("input_lock_sha256"), "original lock checksum")
        crates = data.get("crates")
        require(isinstance(crates, list) and len(crates) == 21, "expected all 21 crates")
        crate_dirs, identities = {}, set()
        for crate in crates:
            require(isinstance(crate, dict), "invalid crate record")
            name, version, directory = (crate.get(key) for key in ("name", "version", "directory"))
            require(isinstance(name, str) and isinstance(version, str), "invalid crate identity")
            identity = (name, version)
            require(identity not in identities, "duplicate crate identity")
            identities.add(identity)
            require(relative(directory) == [directory] and directory == f"{name}-{version}",
                    "crate directory does not match identity")
            require(directory not in crate_dirs, "duplicate crate directory")
            crate_dirs[directory] = crate
            for key in ("archive_sha256", "manifest_upstream_sha256", "manifest_candidate_sha256"):
                digest(crate.get(key), key)
            integer(crate.get("retained_files"), "crate file count")
            integer(crate.get("retained_bytes"), "crate byte count")
        require(identities == EXPECTED_CRATES, "unexpected/missing crate identities")
        pins = data.get("retained_file_pins")
        require(isinstance(pins, list) and bool(pins), "missing retained file pins")
        expected, counts, sizes = {}, dict.fromkeys(crate_dirs, 0), dict.fromkeys(crate_dirs, 0)
        for pin in pins:
            require(isinstance(pin, dict), "invalid file pin")
            path = pin.get("path")
            parts = relative(path)
            require(len(parts) >= 4 and parts[:2] == ["vendor", "security"] and
                    parts[2] in crate_dirs, f"path outside declared crates: {path}")
            short = "/".join(parts[2:])
            require(short not in expected, f"duplicate pinned path: {path}")
            size = integer(pin.get("bytes"), "pinned file size", MAX_FILE_BYTES)
            upstream = digest(pin.get("upstream_sha256"), "upstream source checksum")
            candidate = digest(pin.get("candidate_sha256"), "candidate source checksum")
            require(type(pin.get("changed")) is bool and pin["changed"] == (upstream != candidate),
                    f"inconsistent changed flag: {path}")
            expected[short] = (size, candidate)
            counts[parts[2]] += 1
            sizes[parts[2]] += size
        for directory, crate in sorted(crate_dirs.items()):
            require(counts[directory] == crate["retained_files"] and counts[directory] > 0 and
                    sizes[directory] == crate["retained_bytes"], f"crate inventory mismatch: {directory}")
            manifest = directory + "/Cargo.toml"
            require(manifest in expected and expected[manifest][1] == crate["manifest_candidate_sha256"],
                    f"manifest candidate pin mismatch: {directory}")
            manifest_pin = next(pin for pin in pins if pin["path"] == PREFIX + manifest)
            require(manifest_pin["upstream_sha256"] == crate["manifest_upstream_sha256"],
                    f"manifest upstream pin mismatch: {directory}")
        total_bytes = sum(size for size, _ in expected.values())
        require(integer(data.get("files"), "global file count") == len(expected) and
                integer(data.get("bytes"), "global byte count") == total_bytes, "global inventory mismatch")
        actual, directories = inventory(fd)
        require(actual.pop(PROVENANCE, None) == (len(raw), hashlib.sha256(raw).hexdigest()),
                "provenance changed while checking")
        allowed_directories = set()
        for path in expected:
            parts = path.split("/")
            allowed_directories.update("/".join(parts[:i]) for i in range(1, len(parts)))
        require(not directories - allowed_directories,
                f"extra directories: {sorted(directories - allowed_directories)}")
        require(actual.keys() == expected.keys(),
                f"inventory paths differ; missing={sorted(expected.keys() - actual.keys())}; "
                f"extra={sorted(actual.keys() - expected.keys())}")
        for path in sorted(expected):
            require(actual[path] == expected[path], f"file size/hash mismatch: {path}")
        verify_ratatui_lru(fd, expected)
        return {"schema": "jope.vendor-custody/v1", "status": "pass", "crates": len(crates),
                "files": len(expected), "bytes": total_bytes,
                "provenance_sha256": hashlib.sha256(raw).hexdigest()}
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (default: current directory)")
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (CustodyError, OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        print(f"vendor custody FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
