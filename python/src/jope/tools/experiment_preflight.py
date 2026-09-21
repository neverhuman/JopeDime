"""Read-only production snapshot for experiment release/coexistence checks."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root):
    services = {}
    for name in ("gp", "campaign", "monitor", "reports", "teacher-telemetry"):
        unit = f"jope-parallel-{name}.service"
        result = subprocess.run(
            [
                "systemctl",
                "show",
                unit,
                "-p",
                "MainPID",
                "-p",
                "NRestarts",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "ExecStart",
                "-p",
                "FragmentPath",
                "-p",
                "ControlGroup",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        pid = fields.get("MainPID", "0")
        exe = Path(f"/proc/{pid}/exe")
        if exe.exists():
            fields.update(executable=str(exe.resolve()), executable_sha256=digest(exe))
        fragment = Path(fields.get("FragmentPath", ""))
        if fragment.is_file():
            fields["service_sha256"] = digest(fragment)
        services[unit] = fields
    receipts = {}
    for name in ("campaign-progress", "cycle-progress", "gp-service", "gpu-wait", "encoding-service"):
        path = root / "runs" / (name + ".json")
        try:
            value = json.loads(path.read_text())
            if "active_rows" in value:
                value["worker_executables"] = sorted(
                    {
                        str(Path(f"/proc/{r['pid']}/exe").resolve())
                        for r in value.pop("active_rows")
                        if Path(f"/proc/{r['pid']}/exe").exists()
                    }
                )
            receipts[name] = value
        except (OSError, ValueError) as error:
            receipts[name] = {"unavailable": str(error)}
    return {
        "schema": "parallel.experiment_preflight.v1",
        "timestamp": time.time(),
        "services": services,
        "receipts": receipts,
    }


def compare(before, after):
    failures = []
    for unit, old in before["services"].items():
        new = after["services"].get(unit, {})
        for key in ("MainPID", "NRestarts", "ActiveState", "SubState", "executable_sha256", "service_sha256"):
            if old.get(key) != new.get(key):
                failures.append(f"{unit}: {key} changed")
    return {
        "production_service_identity_unchanged": not failures,
        "failures": failures,
        "before_timestamp": before["timestamp"],
        "after_timestamp": after["timestamp"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), indent=2))
