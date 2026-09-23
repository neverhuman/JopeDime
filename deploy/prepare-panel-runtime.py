#!/usr/bin/env python3
"""Pin installed CLI and external adapter dependencies without fitting any model.

The immutable inventory is private preparation evidence. The controller checks
package bytes before/after an admitted run and checks adapter/request bytes at
every fit. Import failures remain explicit method slots in the frozen roster.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ENGINES = {"xgboost": "xgboost_adapter.py", "lightgbm": "lightgbm_adapter.py",
           "catboost": "catboost_adapter.py", "sklearn": "sklearn_rf_adapter.py"}
HOST_LOCK = Path("/run/lock/jope-exclusive-cpu.lock")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact(path):
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_relative_to("/tmp"):
        raise ValueError(f"requires durable regular artifact: {path}")
    return {"path": str(path), "sha256": digest(path)}


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def probe(script):
    # Setting v2 avoids the legacy adapter import path. Construction is not fit.
    sys.argv = [str(script), "--panel-v2"]
    spec = importlib.util.spec_from_file_location("jope_inventory_adapter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = module.version()
    for task in ("regression", "binary", "multiclass"):
        module.create_panel_model(task)
    package_owners = importlib.metadata.packages_distributions()
    owners = set()
    for name, imported in list(sys.modules.items()):
        filename = getattr(imported, "__file__", None)
        if filename and ("site-packages" in Path(filename).parts or "dist-packages" in Path(filename).parts):
            found = package_owners.get(name.split(".")[0], [])
            if not found:
                raise ValueError(f"imported package has no installed distribution: {name}")
            owners.update(found)
    files = set()
    distributions = []
    for name in sorted(owners):
        distribution = importlib.metadata.distribution(name)
        distributions.append({"name": name, "version": distribution.version})
        if distribution.files is None:
            raise ValueError(f"distribution has no file inventory: {name}")
        for entry in distribution.files:
            if entry.suffix in (".pyc", ".pyo"):
                continue
            path = Path(distribution.locate_file(entry)).resolve(strict=True)
            if path.is_file():
                files.add(path)
    return {"engine_version": version, "distributions": distributions,
            "package_files": [artifact(path) for path in sorted(files)],
            "fits": 0, "constructors_checked": ["regression", "binary", "multiclass"]}


def prepare(args):
    started = time.monotonic()
    # Provisioning owns the persistent inode. Never create/replace it here.
    with HOST_LOCK.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("exclusive CPU allocation is busy; retry preparation after it is released") from error
        binary = artifact(args.binary)
        build = json.loads(args.build_receipt.read_text())
        if (build.get("source_sha") != args.source_sha or
                build.get("binary_sha256") != binary["sha256"] or not build.get("ok", True)):
            raise ValueError("binary differs from pinned source/build receipt")
        if len(args.source_sha) != 40 or any(c not in "0123456789abcdef" for c in args.source_sha):
            raise ValueError("requires full lowercase source SHA")
        adapters = args.adapters.resolve(strict=True)
        python = artifact(args.python)
        site_packages = args.site_packages.resolve(strict=True)
        if not site_packages.is_dir():
            raise ValueError("site-packages directory absent")
        environment = {"PYTHONPATH": str(site_packages), "PYTHONNOUSERSITE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                       "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
        probe_environment = dict(environment, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                 MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
        runtimes = {"rust": {"program": binary, "argument_prefix": [args.native_command],
                            "inputs": [artifact(args.build_receipt)], "package_files": [],
                            "environment": {}, "version": f"jope source {args.source_sha}",
                            "unavailable": None}}
        receipts = {}
        for engine, filename in ENGINES.items():
            script = adapters / filename
            runtime = {"program": python, "argument_prefix": [str(script)],
                       "inputs": [artifact(script), artifact(adapters / "_panel.py")],
                       "package_files": [], "environment": environment,
                       "version": "unavailable", "unavailable": None}
            command = [python["path"], str(Path(__file__).resolve()), "--probe", str(script)]
            try:
                result = subprocess.run(command, env=probe_environment, stdin=subprocess.DEVNULL,
                                        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
                if result.returncode:
                    raise ValueError(f"probe exit {result.returncode}: {result.stderr[-4096:]}")
                receipt = json.loads(result.stdout)
                runtime["version"] = receipt["engine_version"]
                runtime["package_files"] = receipt["package_files"]
                receipts[engine] = receipt
            except (ValueError, subprocess.TimeoutExpired, OSError) as error:
                runtime["unavailable"] = str(error)
                receipts[engine] = {"unavailable": str(error), "fits": 0}
            runtimes[engine] = runtime
        immutable(args.output, runtimes)
        immutable(args.output.with_suffix(".receipt.json"), {
            "schema": "jope.panel-runtime-preparation/v1", "source_sha": args.source_sha,
            "inventory": artifact(args.output), "engines": receipts, "fits": 0,
            "elapsed_seconds": time.monotonic() - started, "execution_clock_started": False})
        return {"ok": True, "output": str(args.output), "fits": 0,
                "unavailable": {key: value["unavailable"] for key, value in runtimes.items()
                                if value["unavailable"] is not None}}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        print(json.dumps(probe(Path(sys.argv[2])), ensure_ascii=False, allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--native-command", choices=("benchmark", "panel"), default="benchmark")
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
