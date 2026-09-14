#!/usr/bin/env python3
"""Build versioned experiment executables from an already isolated source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--kind", action="append", choices=("cpu", "neural"))
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    assert args.version.replace("-", "").isalnum()
    assert source.is_relative_to(root / "builds")
    output = root / "runs/parity" / (args.version + "-build")
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for folder in ("src", "tests", "examples", "native") for p in (source / folder).rglob("*") if p.is_file() and p.suffix in (".rs", ".cpp")]
    paths += [source / n for n in ("Cargo.toml", "Cargo.lock", "build.rs")]
    paths += [p for p in (source / "assets/source/jope-dime/vendor/feat-math").rglob("*") if p.is_file()]
    before = {str(p.relative_to(source)): digest(p) for p in paths}
    (output / "source-sha256.json").write_text(json.dumps(before, indent=2) + "\n")
    env = dict(os.environ, PATH="/home/ubuntu/.cargo/bin:" + os.environ["PATH"],
               LIBTORCH_USE_PYTORCH="1", CXX="g++", CARGO_BUILD_JOBS="2",
               CARGO_TARGET_DIR=str(root / "target"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    receipts = []
    for kind, flags in (("cpu", []), ("neural", ["--features", "neural"])):
        if args.kind and kind not in args.kind:
            continue
        destination = root / "bin" / f"jope-parallel-{kind}-{args.version}"
        assert not destination.exists(), "versioned executable already exists"
        command = ["nice", "-n", "10", "taskset", "-c", "60,123", "cargo", "build", "--release", "--locked", "--bin", "jope-parallel"] + flags
        start = time.monotonic()
        with (output / (kind + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f"{kind} build failed; inspect log"
        assert before == {str(p.relative_to(source)): digest(p) for p in paths}, "source mutated during build"
        with (root / "target/release/jope-parallel").open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        destination.chmod(0o755)
        receipt = dict(kind=kind, path=str(destination.relative_to(root)), sha256=digest(destination),
                       seconds=time.monotonic()-start, source_sha256=digest(output / "source-sha256.json"), command=command)
        receipts.append(receipt)
        (output / (kind + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    (output / "complete.json").write_text(json.dumps(dict(schema="parallel.experiment_build.v1", passed=True,
        source=str(source.relative_to(root)), builds=receipts, production_executables_replaced=False), indent=2)+"\n")


if __name__ == "__main__":
    main()
