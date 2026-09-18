"""SCQ-only multi-node topology for the broad-shadow campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import canonical_json_bytes, content_sha256

CENTRAL_SCQ_WRAPPER = Path("ops/coordination/submit.sh")
SCQ_BINARY = Path("/home/ubuntu/super_cluster/scq")
PREFERRED_NODES = ("xbabe1", "xbabe2", "xbabe3")
_NODE = re.compile(r"^xbabe[1-9][0-9]*$")


def choose_trainer_nodes(
    health: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return every admitted idle node, preferring xbabe1 without waiting for it."""

    admitted = {
        str(row["node"])
        for row in health
        if row.get("healthy") is True
        and row.get("gpu_admitted") is True
        and row.get("externally_occupied") is False
        and row.get("disk_safe") is True
    }
    ordered = [node for node in PREFERRED_NODES if node in admitted]
    ordered.extend(sorted(admitted - set(ordered)))
    return ordered


def trainer_spec(
    *,
    node: str,
    release: Path,
    member: int,
    command: Sequence[str],
) -> str:
    if not _NODE.fullmatch(node):
        raise ValueError("invalid broad-shadow node")
    if member not in range(3):
        raise ValueError("ensemble member must be 0..2")
    if not command or any("\0" in value for value in command):
        raise ValueError("trainer command must be nonempty and NUL-free")
    executable = json.dumps(command[0])
    arguments = json.dumps(list(command[1:]), separators=(",", ":"))
    workdir = json.dumps(str(release))
    return f"""[job]
id = "broad-shadow-trainer-{member}-{node}-{release.name[:16]}"
project = "jope"
tier = "normal"
command = {executable}
args = {arguments}
workdir = {workdir}

[resource]
gpus_per_task = 1
cpus_reserved = 16
omp_threads = 1
ram_gb = 64
node = "{node}"
gang_size = 1
placement = "single_node"

[env]
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
JOPE_VRAM_LIMIT_GIB = "20"

[preempt]
preemptible = false
checkpoint = "none"
resume = "rerun"

[limits]
max_runtime_hours = 24
max_restarts = 8
"""


def evaluator_spec(
    *,
    node: str,
    release: Path,
    command: Sequence[str],
) -> str:
    if not _NODE.fullmatch(node):
        raise ValueError("invalid broad-shadow node")
    if not command or any("\0" in value for value in command):
        raise ValueError("evaluator command must be nonempty and NUL-free")
    executable = json.dumps(command[0])
    arguments = json.dumps(list(command[1:]), separators=(",", ":"))
    workdir = json.dumps(str(release))
    return f"""[job]
id = "broad-shadow-evaluator-{node}-{release.name[:16]}"
project = "jope"
tier = "normal"
command = {executable}
args = {arguments}
workdir = {workdir}

[resource]
gpus_per_task = 0
cpus_reserved = 96
omp_threads = 1
ram_gb = 48
node = "{node}"
gang_size = 1
placement = "single_node"

[env]
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"

[preempt]
preemptible = false
checkpoint = "none"
resume = "rerun"

[limits]
max_runtime_hours = 24
max_restarts = 8
"""


def preflight_spec(
    spec: str | Path,
    *,
    cwd: str | Path,
    minimum_disk_free_gib: int = 100,
) -> dict[str, Any]:
    """Run read-only doctor, disk, and spec checks before any submission."""

    cwd = Path(cwd).resolve()
    spec = Path(spec).resolve()
    if not spec.is_file():
        raise ValueError("SCQ spec is absent")
    stat = os.statvfs(cwd)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < minimum_disk_free_gib * 1024**3:
        raise RuntimeError(
            f"disk gate blocks pack expansion: {free_bytes / 1024**3:.1f} GiB free"
        )
    if not SCQ_BINARY.is_file():
        raise RuntimeError(f"SCQ binary is absent: {SCQ_BINARY}")
    results = {}
    for name, command in (
        ("doctor", [str(SCQ_BINARY), "doctor"]),
        ("check", [str(SCQ_BINARY), "check", str(spec)]),
    ):
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        results[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4_000:],
            "stderr": completed.stderr[-4_000:],
        }
        if completed.returncode:
            raise RuntimeError(f"scq {name} failed before submission")
    receipt = {
        "schema": "BroadScqPreflight.v2",
        "spec": str(spec),
        "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "disk_free_bytes": free_bytes,
        "minimum_disk_free_gib": minimum_disk_free_gib,
        "checks": results,
        "checked_at_ms": int(time.time() * 1000),
    }
    receipt["preflight_sha256"] = content_sha256(receipt)
    return receipt


def submit_via_central_wrapper(
    *,
    spec: str | Path,
    receipt_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Idempotently submit through the sole authorized SCQ wrapper."""

    root = Path(repository_root).resolve()
    wrapper = (root / CENTRAL_SCQ_WRAPPER).resolve()
    if not wrapper.is_file():
        raise ValueError("central SCQ wrapper is absent")
    spec = Path(spec).resolve()
    destination = Path(receipt_path)
    spec_sha256 = hashlib.sha256(spec.read_bytes()).hexdigest()
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != "BroadScqJobReceipt.v2"
            or existing.get("spec_sha256") != spec_sha256
        ):
            raise ValueError("existing SCQ receipt conflicts with requested spec")
        return existing
    completed = subprocess.run(
        [str(wrapper), str(spec)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"central SCQ submission failed: {completed.stderr[-1000:]}")
    receipt = {
        "schema": "BroadScqJobReceipt.v2",
        "spec": str(spec),
        "spec_sha256": spec_sha256,
        "wrapper": str(wrapper),
        "tier": "normal",
        "submitted_at_ms": int(time.time() * 1000),
        "stdout": completed.stdout[-4_000:],
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return receipt
