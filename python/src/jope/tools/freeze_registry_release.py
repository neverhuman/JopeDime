"""Freeze a reviewable source closure; this never installs or activates a release."""

import argparse
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path

ROOTS = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "src",
    "tests",
    "scripts",
    "vendor",
    "manifests",
    "seeds",
    "systemd",
    "parallel/Cargo.toml",
    "parallel/Cargo.lock",
    "parallel/rust-toolchain.toml",
    "parallel/build.rs",
    "parallel/src",
    "parallel/tests",
    "parallel/tools",
    "parallel/examples",
    "parallel/native",
    "parallel/systemd",
    "parallel/assets/grammar",
    "parallel/assets/source/jope-dime/vendor",
    "parallel/assets/source/hyperion-jain",
)
OPTIONAL = (".cargo", "parallel/.cargo", "registry", "parallel/registry", "mcp/requirements.lock")
EXCLUDED = {".git", "__pycache__", ".pytest_cache", "target", ".DS_Store"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inventory(root, roots):
    result = {}
    for name in roots:
        base = root / name
        if not base.exists():
            raise RuntimeError("release dependency absent: " + name)
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts) or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise RuntimeError(
                    "release dependency requires explicit symlink resolution: " + str(relative)
                )
            if path.is_file():
                result[str(relative)] = dict(
                    sha256=sha(path),
                    bytes=path.stat().st_size,
                    executable=bool(path.stat().st_mode & stat.S_IXUSR),
                )
    return dict(sorted(result.items()))


def verify(source, manifest):
    expected = manifest["files"]
    actual = inventory(source, manifest["roots"])
    if actual != expected:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise RuntimeError("frozen source changed: " + ", ".join(changed[:20]))


def freeze(root, output, roots=ROOTS):
    root = root.resolve()
    roots = list(roots) + [name for name in OPTIONAL if (root / name).exists()]
    files = inventory(root, roots)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    for name, binding in files.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise RuntimeError("source changed during freeze: " + name)
        with destination.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        destination.chmod(0o555 if binding["executable"] else 0o444)
    if inventory(root, roots) != files:
        raise RuntimeError("source closure changed during freeze; incomplete freeze is unavailable")
    manifest = dict(
        schema="parallel.registry_source_freeze.v1",
        roots=roots,
        files=files,
        source_closure_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        deployed=False,
        production_namespaces_enabled=False,
        full_ci_passed=False,
        external_dependencies="Must be bound separately in target build and required CI receipts",
    )
    verify(source, manifest)
    # Stable tar headers make identical source closures reproducible.
    with tarfile.open(output / "source.tar", "w") as archive:
        for name in files:
            info = archive.gettarinfo(source / name, arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with (source / name).open("rb") as stream:
                archive.addfile(info, stream)
    manifest["archive_sha256"] = sha(output / "source.tar")
    with (output / "source-manifest.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.root, args.output)
    print(json.dumps({key: result[key] for key in ("source_closure_sha256", "archive_sha256", "deployed")}))
