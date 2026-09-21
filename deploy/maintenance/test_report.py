#!/usr/bin/env python3
"""Exercise preservation and failure behavior under an explicit custody directory."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


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
    if not __debug__:
        raise RuntimeError("test assertions require Python without -O")
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
                 "recent-nested/.rustc_info.json", "recent-nested/nested/.rustc_info.json",
                 "odd\nname/.rustc_info.json"]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00preserve\xff\n")
    outside = scratch / "outside"
    outside.mkdir()
    (outside / ".rustc_info.json").write_text("{}")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "marker-link").mkdir()
    (root / "marker-link/.rustc_info.json").symlink_to(outside / ".rustc_info.json")
    (root / "relsym/release").mkdir(parents=True)
    (root / "relsym/release/.fingerprint").symlink_to(outside, target_is_directory=True)
    locked = root / "inaccessible"
    locked.mkdir(mode=0o755)
    locked.chmod(0o755)
    old = time.time() - 72 * 3600
    for p in [root, *root.rglob("*")]:
        os.utime(p, (old, old), follow_symlinks=False)
    os.utime(root / "fresh/debug/recent", None)
    os.utime(root / "recent-nested/nested/.rustc_info.json", None)
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
    r = call("fingerprint cannot escape supplied root", [root / "fingerprint/release"], True)
    assert json.loads(r.stdout)["candidates"] == []
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
    locked.chmod(0)
    try:
        assert os.geteuid() != 0, "run permission fixture as an unprivileged user"
        call("traversal failure", [root, "--stale", "48"], False)
    finally:
        locked.chmod(0o755)
    spec = importlib.util.spec_from_file_location("jope_report", helper.with_name("jope_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for label, outcome in [
        ("measurement failure", subprocess.CalledProcessError(1, ["du"], stderr="failed")),
        ("partial numeric measurement fails", subprocess.CalledProcessError(
            1, ["du"], output="123\tpartial-root\n", stderr="permission denied")),
        ("invalid successful measurement", subprocess.CompletedProcess(["du"], 0, "garbage", "")),
    ]:
        out, err = io.StringIO(), io.StringIO()
        behavior = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
        with patch.object(module.subprocess, "run", **behavior), \
                patch.object(sys, "argv", [str(helper), "scan", str(root), "--stale", "48"]), \
                redirect_stdout(out), redirect_stderr(err):
            code = module.main()
        assert code == 1 and err.getvalue() and not out.getvalue(), label
        results.append({"test": label, "exit": code, "stderr": err.getvalue()})
    real_walk = module.walk

    def mtime_walk_failure(path):
        if Path(path) == root / "stale":
            raise PermissionError("injected descendant traversal failure after candidate discovery")
        return real_walk(path)

    with patch.object(module, "walk", side_effect=mtime_walk_failure):
        try:
            module.scan(root, 48)
        except PermissionError:
            results.append({"test": "descendant mtime traversal failure", "passed": True})
        else:
            raise AssertionError("candidate mtime traversal failure was suppressed")
    after = inventory(root)
    assert before == after, "fixture contents, membership, links, modes or mtimes changed"
    reports = scratch / "reports"
    reports.mkdir()
    config = scratch / "invalid-config.json"
    config.write_text(json.dumps({"cargo_roots": [],
                                  "stale_hours": 48, "measurement_roots": []}))
    wrapper = helper.with_name("jope-build-report.sh")
    r = subprocess.run([str(wrapper), "--config", str(config), "--custody", str(reports)], capture_output=True,
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
        "preservation": "exact unchanged file/directory membership, contents, symlinks, modes and mtimes; raw inventories"}, indent=2))
    print(f"PASS: {len(results)} scanner scenarios, failure receipt, full fixture preservation; {scratch}")


if __name__ == "__main__":
    main()
