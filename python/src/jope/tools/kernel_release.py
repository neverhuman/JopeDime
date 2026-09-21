#!/usr/bin/env python3
"""Content-addressed source releases with explicit dependency coverage.

The repository's vendor/feat-math is authoritative. Only the development mirror
is generated; experiment snapshots and published releases are never sync targets.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

KERNEL = Path("assets/source/jope-dime/vendor/feat-math")
HYPERION = Path("assets/source/hyperion-jain")
EXCLUDED = {"target", "build", "__pycache__", ".git"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_files(root):
    if not root.is_dir():
        raise ValueError(f"Missing source directory: {root}")
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in dirs + sorted(files):
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not immutable inputs: {path}")
        for name in sorted(files):
            yield Path(parent) / name


def runtime_files(root):
    paths = [root / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    for folder in ("src", "native", "tools", "tests", "examples", "systemd", "assets/grammar", str(KERNEL)):
        paths.extend(tree_files(root / folder))
    if (root / HYPERION).exists():
        paths.extend(tree_files(root / HYPERION))
    paths.extend(p for p in (root / "README.md", root / "rust-toolchain.toml") if p.exists())
    for path in sorted(set(paths)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or indirect runtime dependency: {path}")
        yield path


def manifest(root):
    return {
        str(p.relative_to(root)): {
            "sha256": digest(p),
            "bytes": p.stat().st_size,
            "executable": bool(p.stat().st_mode & 0o111),
        }
        for p in runtime_files(root)
    }


def identity(files):
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return "kernel-v1-" + hashlib.sha256(encoded).hexdigest()


def dependency_coverage(root):
    required = [HYPERION / "crates/feat-cli/src/bin/hyperion_vs.rs"]
    required.extend(
        HYPERION / "crates/feat-core/src" / name
        for name in ("progress.rs", "rng.rs", "hashing.rs", "hyperion.rs", "hyperion_v7/mod.rs")
    )
    missing = [str(p) for p in required if not (root / p).is_file()]
    return {
        "schema": "kernel.source-dependency-coverage.v1",
        "main_kernel_source": "included",
        "hyperion_host": {"source_status": "missing" if missing else "included", "missing_files": missing},
        "native_toolchains_and_libtorch": "require_build_qualification",
        "datasets_and_model_assets": "require_separate_verified_references",
    }


def verify_mirror(root):
    canonical = root.parent / "vendor/feat-math"
    mirror = root / KERNEL
    left = {str(p.relative_to(canonical)): digest(p) for p in tree_files(canonical)}
    right = {str(p.relative_to(mirror)): digest(p) for p in tree_files(mirror)}
    changed = [p for p in sorted(left.keys() | right.keys()) if left.get(p) != right.get(p)]
    if changed:
        raise ValueError("Kernel mirror differs: " + ", ".join(changed))
    return left


def sync_mirror(root):
    canonical = root.parent / "vendor/feat-math"
    mirror = root / KERNEL
    # Fixed routing prevents accidental writes to an archived experiment.
    if not (root.parent / "Cargo.toml").is_file() or root.name != "parallel":
        raise ValueError("Sync requires the repository development parallel directory")
    files = list(tree_files(canonical))
    existing = list(tree_files(mirror))
    names = {p.relative_to(canonical) for p in files}
    extras = [p for p in existing if p.relative_to(mirror) not in names]
    if extras:
        raise ValueError("Review extra mirror files before synchronization: " + str(extras))
    for source in files:
        destination = mirror / source.relative_to(canonical)
        if destination.exists() and digest(source) == digest(destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return verify_mirror(root)


def stage(root, releases):
    verify_mirror(root)
    files = manifest(root)
    release_id = identity(files)
    destination = releases / release_id
    if destination.exists():
        verify_release(destination)
        return destination
    releases.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases))
    try:
        source = temporary / "source"
        for name in files:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        if manifest(root) != files or manifest(source) != files:
            raise ValueError("Source drift during staging; refusing publication")
        (temporary / "release.json").write_text(
            json.dumps(
                {
                    "schema": "kernel.source-release.v1",
                    "release_id": release_id,
                    "files": files,
                    "qualification": "unqualified",
                    "dependency_coverage": dependency_coverage(source),
                },
                indent=2,
            )
            + "\n"
        )
        for path in source.rglob("*"):
            path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
        source.chmod(0o555)
        (temporary / "release.json").chmod(0o444)
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            shutil.rmtree(temporary)


def verify_release(release):
    receipt = json.loads((release / "release.json").read_text())
    files = manifest(release / "source")
    if (
        receipt["schema"] != "kernel.source-release.v1"
        or files != receipt["files"]
        or identity(files) != receipt["release_id"]
        or release.name != identity(files)
    ):
        raise ValueError("Release integrity failure")
    if receipt.get("dependency_coverage") != dependency_coverage(release / "source"):
        raise ValueError("Release dependency coverage mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-mirror")
    sub.add_parser("sync-mirror")
    create = sub.add_parser("stage")
    create.add_argument("--releases", type=Path, required=True)
    verify = sub.add_parser("verify-release")
    verify.add_argument("release", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "verify-mirror":
        print(f"{len(verify_mirror(root))} canonical kernel files verified")
    elif args.command == "sync-mirror":
        print(f"{len(sync_mirror(root))} canonical kernel files synchronized")
    elif args.command == "stage":
        print(stage(root, args.releases.resolve()))
    else:
        print(verify_release(args.release)["release_id"])


if __name__ == "__main__":
    main()
