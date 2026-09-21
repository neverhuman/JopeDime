#!/usr/bin/env python3
"""Exercise preservation and failure behavior under an explicit custody directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def inventory(root):
    rows = {}
    for p in [root, *sorted(root.rglob("*"))]:
        s = p.lstat()
        rows[str(p.relative_to(root))] = {
            "mode": s.st_mode, "mtime_ns": s.st_mtime_ns,
            "content": os.readlink(p) if p.is_symlink() else
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--helper", type=Path,
                        default=Path(__file__).with_name("cargo-build-dirs.sh"))
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if scratch == Path("/tmp") or Path("/tmp") in scratch.parents:
        parser.error("fixtures must be outside /tmp")
    scratch.mkdir(parents=True, exist_ok=False)
    root = scratch / "fixtures"
    root.mkdir()
    for name in ["stale/.rustc_info.json", "fresh/.rustc_info.json",
                 "target/unknown", "generic/CACHEDIR.TAG", "stale/debug/receipt.bin",
                 "stale/release/proof", "stale/tmp/evidence", "stale/nested/.rustc_info.json",
                 "fingerprint/release/.fingerprint/marker", "fresh/debug/recent",
                 "odd\nname/.rustc_info.json"]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00preserve\xff\n")
    outside = scratch / "outside"
    outside.mkdir()
    (outside / ".rustc_info.json").write_text("{}")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    old = time.time() - 72 * 3600
    for p in [root, *root.rglob("*")]:
        os.utime(p, (old, old), follow_symlinks=False)
    os.utime(root / "fresh/debug/recent", None)
    before = inventory(root)
    helper = args.helper.resolve(strict=True)
    results = []

    def call(label, arguments, success, env=None):
        r = subprocess.run([str(helper), *map(str, arguments)],
                           capture_output=True, text=True, env=env, timeout=30)
        results.append({"test": label, "exit": r.returncode, "stdout": r.stdout,
                        "stderr": r.stderr})
        assert (r.returncode == 0) == success, (label, r.returncode, r.stderr)
        if not success:
            assert r.stderr and '"complete": true' not in r.stdout, label
        return r

    r = call("stale and marker selection", [root, "--stale", "48"], True)
    rows = json.loads(r.stdout)["candidates"]
    assert {Path(row["path"]).name for row in rows} == {"stale", "fingerprint", "odd\nname"}
    assert len(rows) == 3, "nested Cargo output must not be double-counted"
    for label, arguments in [
        ("delete rejected", [root, "--delete"]),
        ("delete before root validation", [scratch / "missing", "--delete"]),
        ("missing age", [root, "--stale"]),
        ("invalid age", [root, "--stale", "wat"]),
        ("negative age", [root, "--stale", "-1"]),
        ("missing root", [scratch / "missing"]),
        ("symlink root", [root / "linked"]),
    ]:
        call(label, arguments, False)
    locked = root / "inaccessible"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert os.geteuid() != 0, "run permission fixture as an unprivileged user"
        call("traversal failure", [root, "--stale", "48"], False)
    finally:
        locked.chmod(0o755)
    fakebin = scratch / "fakebin"
    fakebin.mkdir()
    du = fakebin / "du"
    du.write_text("#!/bin/sh\necho 'injected measurement failure' >&2\nexit 1\n")
    du.chmod(0o755)
    call("measurement failure", [root, "--stale", "48"], False,
         {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"]})
    # The permission fixture intentionally added one directory. Compare the original entries.
    after = inventory(root)
    for name in before:
        if name == ".":
            after[name]["mtime_ns"] = before[name]["mtime_ns"]
        assert before[name] == after[name], name
    reports = scratch / "reports"
    reports.mkdir()
    config = scratch / "invalid-config.json"
    config.write_text(json.dumps({"custody": str(reports), "cargo_roots": [],
                                  "stale_hours": 48, "measurement_roots": []}))
    wrapper = helper.with_name("jope-build-report.sh")
    r = subprocess.run([str(wrapper), "--config", str(config)], capture_output=True,
                       text=True, timeout=30)
    assert r.returncode != 0
    receipt = json.loads(next(reports.glob("*/receipt.json")).read_text())
    assert receipt["status"] == "failed" and receipt["exit_status"] == 1
    assert next(reports.glob("*/failure.txt")).read_text()
    (scratch / "before.json").write_text(json.dumps(before, indent=2))
    (scratch / "after.json").write_text(json.dumps(after, indent=2))
    (scratch / "results.json").write_text(json.dumps({"tests": results,
        "wrapper_failure": receipt, "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(helper.with_name("jope_report.py").read_bytes()).hexdigest(),
        "preservation": "all original contents, symlinks and modes unchanged"}, indent=2))
    print(f"PASS: {len(results)} scanner scenarios, failure receipt, full fixture preservation; {scratch}")


if __name__ == "__main__":
    main()
