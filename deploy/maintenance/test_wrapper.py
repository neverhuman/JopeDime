#!/usr/bin/env python3
"""Test report receipts; real installed-service runs separately cover live root scans."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


def main():
    if not __debug__:
        raise RuntimeError("test assertions require Python without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--core", type=Path, default=Path(__file__).with_name("jope_report.py"))
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if Path("/tmp") in [scratch, *scratch.parents]:
        parser.error("fixtures must be outside /tmp")
    scratch.mkdir(parents=True, exist_ok=False)
    core = args.core.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("jope_report", core)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    measured = scratch / "measured"
    measured.mkdir()
    (measured / "child").mkdir()
    (measured / "evidence.bin").write_bytes(b"\x00preserve\xff")
    config = {"cargo_roots": module.APPROVED_ROOTS, "stale_hours": 48,
              "measurement_roots": [{"group": "fixture", "path": str(measured)}]}
    cases = []

    def exercise(name, value, expected, *, measurement_failure=False, scan_failure=False):
        directory = scratch / name
        directory.mkdir()
        path = directory / "config.json"
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        real_run = module.subprocess.run
        invocations = []

        def measured_run(command, **kwargs):
            invocations.append(command)
            if command[0] == "/usr/bin/sudo":
                assert command[:3] == ["/usr/bin/sudo", "-n", "/usr/bin/du"]
                if measurement_failure:
                    raise subprocess.CalledProcessError(1, command, stderr="sudo denied (fixture)")
                return real_run(command[2:], **kwargs)
            return real_run(command, **kwargs)

        def scan(root, hours):
            assert root in module.APPROVED_ROOTS and hours == 48
            if scan_failure:
                raise PermissionError("fixture scan traversal denied")
            return {"root": root, "complete": True, "report_only": True, "candidates": []}

        with patch.object(module, "scan", side_effect=scan), \
                patch.object(module.subprocess, "run", side_effect=measured_run), \
                redirect_stdout(io.StringIO()):
            code = module.report(SimpleNamespace(config=path, custody=directory / "reports"))
        receipt_path = next((directory / "reports").glob("*/receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        assert receipt["checksums"][str(core)] == hashlib.sha256(core.read_bytes()).hexdigest()
        assert code == expected and receipt["exit_status"] == expected, name
        assert receipt["status"] == ("success" if expected == 0 else "failed"), name
        if expected:
            assert (receipt_path.parent / "failure.txt").read_text(), name
        else:
            assert receipt["measurements"][0]["allocated_bytes"] > 0
            assert len(receipt["scan_outputs"]) == 2
            assert (receipt_path.parent / "filesystem.txt").read_text()
            for scan_row in receipt["scan_outputs"]:
                assert json.loads((receipt_path.parent / scan_row["output"]).read_text())["complete"]
                assert (receipt_path.parent / scan_row["stderr"]).read_text() == ""
        cases.append({"name": name, "receipt": str(receipt_path), "commands": invocations})
        return receipt_path, receipt

    exercise("success", config, 0)
    for name, value in [("malformed", "{"), ("missing-key", {}), ("non-object", []),
                        ("wrong-rows", {**config, "measurement_roots": {}})]:
        exercise(name, value, 1)
    for name, second in [("duplicate", measured), ("nested", measured / "child")]:
        exercise(name, {**config, "measurement_roots": config["measurement_roots"] +
                       [{"group": "fixture", "path": str(second)}]}, 1)
    path, receipt = exercise("missing-measurement-root", {**config, "measurement_roots": [
        {"group": "fixture", "path": str(scratch / "absent")}]}, 1)
    assert len(receipt["scan_outputs"]) == 2
    for row in receipt["scan_outputs"]:
        assert json.loads((path.parent / row["output"]).read_text())["complete"]
    privileged = {**config, "measurement_roots": [{**config["measurement_roots"][0],
                                                   "privileged": True}]}
    exercise("privileged-command", privileged, 0)
    exercise("privilege-denied", privileged, 1, measurement_failure=True)
    path, receipt = exercise("scan-failure", config, 1, scan_failure=True)
    assert "traversal denied" in (path.parent / receipt["scan_outputs"][0]["stderr"]).read_text()

    # Send a real signal while a deliberately slow scan keeps the child inside report().
    directory = scratch / "signal"
    directory.mkdir()
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config))
    child = (
        "import importlib.util,sys,time; "
        "s=importlib.util.spec_from_file_location('report',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.scan=lambda *a: time.sleep(30); "
        "sys.argv=['report','report','--config',sys.argv[2],'--custody',sys.argv[3]]; "
        "sys.exit(m.main())"
    )
    process = subprocess.Popen([sys.executable, "-c", child, str(core), str(config_path),
                                str(directory / "reports")], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if list((directory / "reports").glob("*/scan-0.stderr")):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("signal fixture did not enter scan")
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=10)
        assert process.returncode == 1, (out, err)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    receipt_path = next((directory / "reports").glob("*/receipt.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed" and "signal 15" in receipt["error"]
    cases.append({"name": "real-SIGTERM", "receipt": str(receipt_path)})
    # Neither refused location is created or modified.
    for name, custody in [("tmp-custody-refused", Path("/tmp")),
                          ("working-tree-custody-refused", Path(module.APPROVED_ROOTS[0]))]:
        try:
            module.report(SimpleNamespace(config=config_path, custody=custody))
        except ValueError as error:
            assert "outside /tmp and Git working trees" in str(error)
        else:
            raise AssertionError(name)
        cases.append({"name": name, "refused_before_writes": True})
    assert (measured / "evidence.bin").read_bytes() == b"\x00preserve\xff"
    (scratch / "results.json").write_text(json.dumps({"cases": cases,
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "scope": "real receipt/disk/df paths; approved-root scanning isolated by mock, sudo success delegated to unprivileged du; installed service independently tests real roots/sudo"}, indent=2))
    print(f"PASS: {len(cases)} wrapper cases; {scratch}")


if __name__ == "__main__":
    main()
