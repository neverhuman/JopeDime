#!/usr/bin/env python3
"""Historical Python deployment helper retained for evidence/tests only.

Production must use ``target/release/jope_selfplay_bundle --activate`` and the
Rust-backed systemd units. This module is not a production entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from build_selfplay_bundle import build
from test_exact_common import ALLOWED_LIVE_STATES, atomic_json, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SOURCE = Path("/home/ubuntu/jope-dime")
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
UNITS = (
    "jope-selfplay-queue.service",
    "jope-selfplay-reserve.service",
    "jope-selfplay-ingestion.service",
    "jope-selfplay-controller.service",
    "jope-selfplay-watchdog.service",
    "jope-selfplay-freshness.service",
)


def run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, check=False, timeout=timeout
    )


def scheduler_jobs() -> dict[str, dict]:
    result = run([str(SCQ), "--json", "ls"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {
        row["id"]: row
        for row in json.loads(result.stdout).get("Jobs", {}).get("jobs", [])
        if row.get("id")
    }


def gpu_state() -> dict[str, dict]:
    result = run([str(SCQ), "--json", "nodes"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {
        row["node"]: row
        for row in json.loads(result.stdout).get("Nodes", {}).get("gpus", [])
        if row.get("node")
    }


def verify_bundle(manifest: dict) -> None:
    bundle_root = Path(manifest["files"][0]["source"]).parents[0]
    del bundle_root
    final = ROOT / "runtime-bundles" / manifest["bundle_sha256"]
    for row in manifest["files"]:
        if sha256_file(final / row["target"]) != row["sha256"]:
            raise RuntimeError(f"bundle verification failed: {row['target']}")
    scripts = [
        str(final / row["target"])
        for row in manifest["files"]
        if row["target"].startswith("runtime/scripts/") and row["target"].endswith(".py")
    ]
    compiled = run(["/usr/bin/python3", "-m", "py_compile", *scripts], timeout=300)
    if compiled.returncode:
        raise RuntimeError((compiled.stderr or compiled.stdout).strip())


def install_units(*, freshness_audit_only: bool) -> None:
    destination = Path.home() / ".config" / "systemd" / "user"
    destination.mkdir(parents=True, exist_ok=True)
    for unit in UNITS:
        source = SOURCE / "systemd" / unit
        target = destination / unit
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    dropin_dir = destination / "jope-selfplay-freshness.service.d"
    dropin = dropin_dir / "10-audit-only.conf"
    if freshness_audit_only:
        dropin_dir.mkdir(parents=True, exist_ok=True)
        temporary = dropin.with_name(f".{dropin.name}.tmp-{os.getpid()}")
        temporary.write_text(
            "[Service]\nEnvironment=JOPE_FRESHNESS_AUDIT_ONLY=1\n"
        )
        os.replace(temporary, dropin)
    else:
        dropin.unlink(missing_ok=True)


def activate(
    *,
    restart: bool = True,
    freshness_audit_only: bool = True,
) -> dict:
    from test_exact_evaluator import activate_protocol, active as active_evaluator

    before = scheduler_jobs()
    gpus_before = gpu_state()
    protected = {
        job_id: row
        for job_id, row in before.items()
        if row.get("state") in ALLOWED_LIVE_STATES
        and (
            row.get("node") in {"xbabe1", "xbabe3"}
            or "ratchet" in job_id.lower()
        )
    }
    manifest = build(ROOT / "runtime-bundles")
    verify_bundle(manifest)
    tests = run(
        [
            "/usr/bin/python3",
            "-m",
            "pytest",
            "-q",
            str(SOURCE / "tests" / "test_durable_selfplay.py"),
            str(SOURCE / "tests" / "test_selfplay_controller.py"),
            str(SOURCE / "tests" / "test_test_exact_campaign.py"),
            str(SOURCE / "tests" / "test_realtime_selfplay.py"),
        ],
        timeout=900,
    )
    if tests.returncode:
        raise RuntimeError((tests.stderr or tests.stdout).strip()[-8_000:])
    bundle_manifest = (
        ROOT
        / "runtime-bundles"
        / manifest["bundle_sha256"]
        / "bundle.json"
    )
    pointer = {
        "schema": "JopeSelfplayRuntimeBundlePointer.v1",
        "bundle": str(bundle_manifest),
        "bundle_manifest_sha256": sha256_file(bundle_manifest),
        "bundle_sha256": manifest["bundle_sha256"],
        "activated_at_ms": int(time.time() * 1000),
    }
    atomic_json(ROOT / "runtime-bundles" / "active.json", pointer)
    protocol_pointer = activate_protocol(ROOT)
    evaluator_pointer = active_evaluator(ROOT)
    if restart:
        install_units(freshness_audit_only=freshness_audit_only)
        reloaded = run(["systemctl", "--user", "daemon-reload"])
        if reloaded.returncode:
            raise RuntimeError((reloaded.stderr or reloaded.stdout).strip())
        run(
            [
                "systemctl",
                "--user",
                "disable",
                "--now",
                "jope-testonly-monitor.service",
            ]
        )
        enabled = run(["systemctl", "--user", "enable", *UNITS], timeout=300)
        if enabled.returncode:
            raise RuntimeError((enabled.stderr or enabled.stdout).strip())
        restarted = run(
            ["systemctl", "--user", "restart", *UNITS], timeout=300
        )
        if restarted.returncode:
            raise RuntimeError((restarted.stderr or restarted.stdout).strip())
        deadline = time.time() + 90
        while time.time() < deadline:
            reports = []
            for unit in UNITS:
                service = unit.removeprefix("jope-selfplay-").removesuffix(
                    ".service"
                )
                state = (
                    ROOT / "ops" / "freshness-status.json"
                    if service == "freshness"
                    else ROOT / "ops" / "services" / f"{service}.json"
                )
                if state.is_file():
                    reports.append(json.loads(state.read_text()))
            if len(reports) == len(UNITS) and {
                row.get("loaded_bundle_sha256") for row in reports
            } == {manifest["bundle_sha256"]}:
                break
            time.sleep(1)
        else:
            raise RuntimeError("services did not report the activated bundle hash")
    after = scheduler_jobs()
    gpus_after = gpu_state()
    missing = sorted(job_id for job_id in protected if job_id not in after)
    if missing:
        raise RuntimeError(f"protected workloads disappeared: {','.join(missing)}")
    if gpus_before.get("xbabe1", {}).get("status") != gpus_after.get(
        "xbabe1", {}
    ).get("status"):
        raise RuntimeError("xbabe1 external GPU workload state changed")
    result = {
        **pointer,
        "services_restarted": restart,
        "freshness_audit_only": freshness_audit_only,
        "protected_workloads_before": sorted(protected),
        "protected_workloads_missing": missing,
        "existing_jobs_cancelled": 0,
        "xbabe1_gpu_status_before": gpus_before.get("xbabe1", {}).get("status"),
        "xbabe1_gpu_status_after": gpus_after.get("xbabe1", {}).get("status"),
        "evaluation_protocol": protocol_pointer,
        "active_evaluator_sha256": evaluator_pointer["evaluator_sha256"],
    }
    atomic_json(ROOT / "ops" / "last-deployment.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument(
        "--freshness-active",
        action="store_true",
        help="remove the audit-only drop-in and allow automatic stale holds",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            activate(
                restart=not args.no_restart,
                freshness_audit_only=not args.freshness_active,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
