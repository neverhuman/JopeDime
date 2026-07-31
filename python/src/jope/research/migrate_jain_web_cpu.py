#!/usr/bin/env python3
"""Transactionally replace xbabe1 Jain containers with CPU-only clones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

CONTAINERS = (("jain", 8080), ("jain-gputest", 18_100))
STARTUP_TIMEOUT_SECONDS = 120


def command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, check=False, text=True)
    if check and result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result


def inspect(name: str) -> dict:
    return json.loads(command(["docker", "inspect", name]).stdout)[0]


def cpu_payload(value: dict) -> dict:
    config = dict(value["Config"])
    config["Image"] = value["Image"]
    config["Env"] = [
        item
        for item in config.get("Env") or []
        if not item.startswith(
            (
                "NVIDIA_VISIBLE_DEVICES=",
                "NVIDIA_DRIVER_CAPABILITIES=",
                "JAIN_DISABLE_GPU_EXEC=",
            )
        )
    ]
    config["Env"].extend(
        ("NVIDIA_VISIBLE_DEVICES=void", "JAIN_DISABLE_GPU_EXEC=1")
    )
    host = dict(value["HostConfig"])
    host["DeviceRequests"] = []
    host["Devices"] = [
        device
        for device in host.get("Devices") or []
        if "nvidia" not in str(device).lower()
    ]
    host["Runtime"] = "runc"
    config["HostConfig"] = host
    return config


def health(port: int, samples: int = 10) -> dict:
    timings = []
    body_hash = None
    for _ in range(samples):
        result = command(
            [
                "curl",
                "-fsS",
                "--max-time",
                "5",
                "-w",
                "\n%{http_code} %{time_total}",
                f"http://127.0.0.1:{port}/",
            ],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"port {port} root health failed: {result.stderr.strip()}")
        body, status = result.stdout.rsplit("\n", 1)
        code, elapsed = status.split()
        if code != "200":
            raise RuntimeError(f"port {port} root health returned HTTP {code}")
        digest = hashlib.sha256(body.encode()).hexdigest()
        if body_hash is None:
            body_hash = digest
        elif digest != body_hash:
            raise RuntimeError(f"port {port} health body changed within one sample window")
        timings.append(float(elapsed))
    return {
        "url": f"http://127.0.0.1:{port}/",
        "status": 200,
        "samples": len(timings),
        "median_seconds": statistics.median(timings),
        "p95_seconds": sorted(timings)[max(0, int(len(timings) * 0.95) - 1)],
        "body_sha256": body_hash,
    }


def create(name: str, payload_path: Path) -> None:
    version = command(
        ["docker", "version", "--format", "{{.Server.APIVersion}}"]
    ).stdout.strip()
    result = command(
        [
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--unix-socket",
            "/var/run/docker.sock",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            f"@{payload_path}",
            "-X",
            "POST",
            f"http://localhost/v{version}/containers/create?name={quote(name)}",
        ]
    )
    response = json.loads(result.stdout)
    if not response.get("Id"):
        raise RuntimeError(f"Docker did not create {name}: {response}")


def safe_record(value: dict) -> dict:
    return {
        "name": value["Name"].removeprefix("/"),
        "image": value["Config"]["Image"],
        "image_id": value["Image"],
        "created": value["Created"],
        "restart": value["HostConfig"]["RestartPolicy"],
        "network_mode": value["HostConfig"]["NetworkMode"],
        "device_requests": value["HostConfig"].get("DeviceRequests"),
        "mounts": value["Mounts"],
        "ports": value["NetworkSettings"]["Ports"],
    }


def gpu_processes() -> list[str]:
    result = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def write_receipt(path: Path, receipt: dict) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)


def migrate(output_root: Path, *, apply: bool) -> dict:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = output_root / stamp
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    records = []
    for name, port in CONTAINERS:
        value = inspect(name)
        full = root / f"{name}.inspect.json"
        full.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.chmod(full, 0o600)
        payload = root / f"{name}.cpu-create.json"
        payload.write_text(json.dumps(cpu_payload(value), sort_keys=True))
        os.chmod(payload, 0o600)
        records.append(
            {
                "container": safe_record(value),
                "inspect_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "baseline_health": health(port),
                "rollback_name": f"{name}-gpu-rollback-{stamp}",
                "rollback_commands": [
                    f"docker stop {name}",
                    f"docker rm {name}",
                    f"docker rename {name}-gpu-rollback-{stamp} {name}",
                    f"docker start {name}",
                ],
            }
        )
    receipt = {
        "schema": "JopeJainWebCpuMigration.v1",
        "created_at": stamp,
        "applied": False,
        "records": records,
        "gpu_processes_before": gpu_processes(),
    }
    if not apply:
        (root / "plan.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return {**receipt, "record_root": str(root)}

    migrated = []
    try:
        for record, (name, port) in zip(records, CONTAINERS):
            rollback = record["rollback_name"]
            command(["docker", "stop", name])
            command(["docker", "rename", name, rollback])
            try:
                create(name, root / f"{name}.cpu-create.json")
                command(["docker", "start", name])
                for _ in range(STARTUP_TIMEOUT_SECONDS):
                    try:
                        replacement = health(port)
                        break
                    except RuntimeError:
                        time.sleep(1)
                else:
                    raise RuntimeError(
                        f"{name} did not become healthy within "
                        f"{STARTUP_TIMEOUT_SECONDS} seconds"
                    )
                baseline = record["baseline_health"]
                latency_limit = max(0.020 + baseline["p95_seconds"], 1.5 * baseline["p95_seconds"])
                if replacement["p95_seconds"] > latency_limit:
                    raise RuntimeError(
                        f"{name} CPU p95 {replacement['p95_seconds']:.6f}s "
                        f"exceeds {latency_limit:.6f}s"
                    )
                if replacement["body_sha256"] != baseline["body_sha256"]:
                    raise RuntimeError(f"{name} root response content changed")
                record["replacement_health"] = replacement
                record["cpu_inspect"] = safe_record(inspect(name))
                if record["cpu_inspect"]["device_requests"]:
                    raise RuntimeError(f"{name} retained a GPU device request")
                migrated.append((name, rollback))
            except BaseException as exc:
                record["failure"] = f"{type(exc).__name__}: {exc}"
                logs = command(["docker", "logs", "--tail", "200", name], check=False)
                record["failure_logs_tail"] = (logs.stdout + logs.stderr)[-32_768:]
                command(["docker", "rm", "-f", name], check=False)
                command(["docker", "rename", rollback, name])
                command(["docker", "start", name])
                raise
        remaining = [
            row for row in gpu_processes() if "/usr/local/bin/jain-web" in row
        ]
        if remaining:
            raise RuntimeError(f"Jain GPU contexts remain after migration: {remaining}")
        receipt["applied"] = True
        receipt["gpu_processes_after"] = gpu_processes()
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        for name, rollback in reversed(migrated):
            command(["docker", "rm", "-f", name], check=False)
            command(["docker", "rename", rollback, name], check=False)
            command(["docker", "start", name], check=False)
        receipt["rolled_back"] = True
        receipt["gpu_processes_after_rollback"] = gpu_processes()
        write_receipt(root / "receipt.json", receipt)
        raise
    write_receipt(root / "receipt.json", receipt)
    return {**receipt, "record_root": str(root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0/ops/jain-cpu-migration"),
    )
    args = parser.parse_args()
    print(json.dumps(migrate(args.output_root, apply=args.apply), sort_keys=True))


if __name__ == "__main__":
    main()
