#!/usr/bin/env python3
"""Publish immutable CPU/neural executables from one verified source snapshot."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from kernel_release import manifest, stage, verify_release


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--neural-only", action="store_true")
    ap.add_argument("--release", type=Path, help="Verified immutable source release")
    args = ap.parse_args()
    root = args.root.resolve()
    assert args.version.isalnum(), "Version must be a simple identifier"
    receipt = root / f"runs/parity/runtime-{args.version}-build.json"
    assert not receipt.exists(), "Build receipt already exists"

    release = args.release.resolve() if args.release else stage(root, root / "releases")
    release_receipt = verify_release(release)
    source_root = release / "source"

    def sources():
        return manifest(source_root)

    before = sources()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.with_name(f"runtime-{args.version}-sources.json").write_text(json.dumps(before, indent=2) + "\n")
    env = dict(
        os.environ,
        LIBTORCH_USE_PYTORCH="1",
        CXX="g++",
        CARGO_BUILD_JOBS="4",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        PATH="/home/ubuntu/.cargo/bin:" + os.environ["PATH"],
    )
    builds = []
    targets = (
        [("neural", ["--features", "neural"])]
        if args.neural_only
        else [("cpu", []), ("neural", ["--features", "neural"])]
    )
    for kind, flags in targets:
        target = root / "build-output" / release_receipt["release_id"] / kind
        env["CARGO_TARGET_DIR"] = str(target)
        destination = root / f"bin/jope-parallel-{kind}-{args.version}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(), "Refusing to replace an executable"
        cmd = ["cargo", "build", "--release", "--locked", "--bin", "jope-parallel"] + flags
        if kind == "cpu":
            cmd += ["--example", "champions"]
        else:
            cmd += ["--example", "qualify_auxiliary"]
        begun = time.monotonic()
        with receipt.with_name(f"runtime-{args.version}-{kind}-build.log").open("x") as log:
            result = subprocess.run(cmd, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f"{kind} build failed; inspect log"
        assert before == sources(), "Source changed during build; refusing publication"
        with (target / "release/jope-parallel").open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        destination.chmod(0o755)
        builds.append(
            dict(
                kind=kind,
                path=str(destination.relative_to(root)),
                sha256=digest(destination),
                seconds=time.monotonic() - begun,
                command=cmd,
            )
        )
        print(json.dumps(builds[-1]), flush=True)
        if kind == "cpu":
            helper = root / f"bin/jope-parallel-champions-{args.version}"
            with (target / "release/examples/champions").open("rb") as src, helper.open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            helper.chmod(0o755)
            builds.append(
                dict(kind="champion_import", path=str(helper.relative_to(root)), sha256=digest(helper))
            )
        else:
            helper = root / f"bin/jope-parallel-qualify-auxiliary-{args.version}"
            with (target / "release/examples/qualify_auxiliary").open("rb") as src, helper.open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            helper.chmod(0o755)
            builds.append(
                dict(
                    kind="auxiliary_qualification", path=str(helper.relative_to(root)), sha256=digest(helper)
                )
            )
    assert before == sources(), "Final source fingerprint changed"
    receipt.write_text(
        json.dumps(
            dict(
                schema="parallel.immutable_runtime_build.v1",
                timestamp=int(time.time()),
                passed=True,
                release_id=release_receipt["release_id"],
                source_files=before,
                builds=builds,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
