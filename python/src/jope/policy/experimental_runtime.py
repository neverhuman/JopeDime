"""Prepare private source/package pins for the experimental CPU proposal worker.

Run this file with the physical interpreter and -I -B. This hashes installed bytes
and reads package metadata; it never loads a model, creates tensors or fits data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import sysconfig
from pathlib import Path


def encoded(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("runtime dependency must be a regular file")
    return {"path": str(path), "sha256": sha(path)}


def write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(path)


def prepare(source_root, site_packages, output):
    source_root, site_packages = (
        Path(source_root).resolve(strict=True),
        Path(site_packages).resolve(strict=True),
    )
    output = Path(output).absolute()
    if output.resolve().is_relative_to(Path("/tmp")) or output.exists():
        raise ValueError("runtime preparation requires a new durable output directory")
    if Path(sys.executable).absolute() != Path(sys.executable).resolve(strict=True) or not sys.flags.isolated:
        raise ValueError("invoke the physical interpreter with -I -B")
    if not (source_root / "jope/policy/experimental_proposal.py").is_file() or not site_packages.is_dir():
        raise ValueError("explicit proposal source and site-packages roots are required")
    stdlib = sysconfig.get_path("stdlib")
    sys.path[:] = [
        str(source_root),
        str(site_packages),
        stdlib,
        sysconfig.get_path("platstdlib"),
        os.path.join(stdlib, "lib-dynload"),
    ]
    # Match the pinned worker bootstrap. Imports initialize Python definitions,
    # never a policy/context model or optimizer.
    from jope.policy import training_campaign

    identity = training_campaign.runtime_identity()
    sources = [artifact(path) for path in sorted((source_root / "jope").rglob("*.py"))]
    packages, installed = {}, {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in packages:
            raise ValueError("ambiguous installed distribution")
        files = {}
        for item in sorted(distribution.files or [], key=str):
            path = Path(distribution.locate_file(item))
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            ref = artifact(path)
            files[str(item)] = ref
            prior = installed.setdefault(ref["path"], ref)
            if prior != ref:
                raise ValueError("shared package file differs")
        packages[name] = {"version": distribution.version, "files": files}
    if set(packages) != set(identity["packages"]):
        raise ValueError("runtime_identity must include every installed distribution")
    runtime = {
        "program": artifact(sys.executable),
        "source_root": str(source_root),
        "site_packages": str(site_packages),
        "source_files": sources,
        "package_files": [installed[path] for path in sorted(installed)],
        "runtime_sha256": hashlib.sha256(encoded(identity)).hexdigest(),
        "environment": {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    output.mkdir(parents=True, mode=0o700)
    identity_ref = write(output / "runtime-identity.json", identity)
    manifest_ref = write(
        output / "runtime-manifest.json",
        {
            "schema": "jope.experimental-python-runtime/v1",
            "runtime": runtime,
            "identity": identity_ref,
            "packages": packages,
        },
    )
    return write(output / "runtime.json", {"manifest": manifest_ref, **runtime})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    print(json.dumps(prepare(args.source_root, args.site_packages, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
