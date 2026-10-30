#!/usr/bin/env python3
"""Serialize the preregistered H52 cells with per-cell guards and lock custody."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path.home() / "jope-data/h52/runs"
STATUS = Path.home() / "jope_plan/ns/H52_STATUS.md"
LOCK = Path.home() / "jope_plan/qualification.lock"
CHECKPOINT = Path.home() / "jope-data/h51/ckpt/round-882.pt"
INPUT = Path.home() / "jope-data/h50/input"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SEEDS = (12, 13, 14)
RSS_LIMIT_KIB = 6 * 1024 * 1024
LOCK_CPU_FALLBACK_SECONDS = 20 * 60


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(message: str) -> None:
    line = f"{now()} — {message}\n"
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    print(line.rstrip(), flush=True)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_util() -> float:
    def sample() -> tuple[int, int]:
        fields = list(
            map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:])
        )
        return sum(fields), fields[3] + fields[4]

    first = sample()
    time.sleep(1)
    second = sample()
    total = second[0] - first[0]
    idle = second[1] - first[1]
    return 100.0 * (total - idle) / total


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable missing")


def gpu_free_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def preflight(device: str) -> dict:
    available = available_kib()
    cpu = cpu_util()
    result = {
        "device": device,
        "mem_available_kib": available,
        "cpu_utilization_percent": cpu,
        "swap_total_kib": 0,
        "swap_free_kib": 0,
    }
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("SwapTotal:"):
            result["swap_total_kib"] = int(line.split()[1])
        elif line.startswith("SwapFree:"):
            result["swap_free_kib"] = int(line.split()[1])
    failures = []
    if available < 20 * 1024 * 1024:
        failures.append("MemAvailable below 20 GiB")
    if cpu > 85:
        failures.append(f"CPU utilization {cpu:.1f}% exceeds 85%")
    if device == "cuda:0":
        free = gpu_free_mib()
        result["gpu_free_mib"] = free
        if free < 20 * 1024:
            failures.append(f"GPU free memory {free} MiB below 20 GiB")
    result["guard_failures"] = failures
    return result


def rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def configurations() -> list[dict]:
    cells = []
    controls = (
        ("k2-lr1e-4", 2, 1e-4),
        ("k16-lr1e-4", 16, 1e-4),
        ("k16-lr3e-4", 16, 3e-4),
    )
    for config, k, lr in controls:
        for seed in SEEDS:
            cells.append(
                {"arm": "CONTROL", "config": config, "seed": seed, "k": k, "lr": lr}
            )
    for tau in (0.1, 0.5):
        for seed in SEEDS:
            cells.append(
                {
                    "arm": "FOCUS",
                    "config": f"tau{tau}",
                    "seed": seed,
                    "k": 16,
                    "lr": 1e-4,
                    "tau": tau,
                    "control": "k16-lr1e-4",
                }
            )
    for seed in SEEDS:
        cells.append(
            {
                "arm": "GATE",
                "config": "default",
                "seed": seed,
                "k": 16,
                "lr": 1e-4,
                "tau": 0.1,
                "control": "k16-lr1e-4",
            }
        )
    for lam in (1.0, 10.0):
        for seed in SEEDS:
            cells.append(
                {
                    "arm": "ANCHOR",
                    "config": f"lambda{int(lam)}",
                    "seed": seed,
                    "k": 16,
                    "lr": 3e-4,
                    "lambda_anchor": lam,
                    "control": "k16-lr3e-4",
                }
            )
    for seed in SEEDS:
        cells.append(
            {
                "arm": "NULL",
                "config": "tau0.1",
                "seed": seed,
                "k": 16,
                "lr": 1e-4,
                "tau": 0.1,
                "control": "k16-lr1e-4",
            }
        )
    return cells


def acquire_gpu_lock() -> tuple[object | None, bool]:
    handle = LOCK.open("r+")
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle, False
        except BlockingIOError:
            if time.monotonic() - started >= LOCK_CPU_FALLBACK_SECONDS:
                handle.close()
                return None, True
            time.sleep(15)


def run_cell(cell: dict, repo: Path) -> bool:
    cell_dir = RUNS / f"{cell['arm']}-{cell['config']}" / f"seed{cell['seed']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    handle, cpu_fallback = acquire_gpu_lock()
    device = "cpu" if cpu_fallback else "cuda:0"
    guard = preflight(device)
    if guard["guard_failures"]:
        note(
            f"STOP before {cell['arm']} {cell['config']} seed{cell['seed']}: "
            f"guard failure {json.dumps(guard, sort_keys=True)}"
        )
        if handle:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return False
    if sha256(CHECKPOINT) != R882_SHA:
        note("STOP: r882 checkpoint SHA mismatch before cell")
        if handle:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return False
    paired_path = None
    if "control" in cell:
        paired_path = (
            RUNS
            / f"CONTROL-{cell['control']}"
            / f"seed{cell['seed']}"
            / (f"CONTROL-{cell['control']}-seed{cell['seed']}-family-gains.json")
        )
        if not paired_path.is_file():
            note(
                f"STOP: matched CONTROL missing for {cell['arm']} {cell['config']} seed{cell['seed']}"
            )
            if handle:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return False
    command = [
        sys.executable,
        str(repo / "research/flywheel_selfplay/h52/runner.py"),
        "--repo-root",
        str(repo),
        "--mode",
        "candidate",
        "--arm",
        cell["arm"],
        "--config",
        cell["config"],
        "--seed",
        str(cell["seed"]),
        "--device",
        device,
        "--k",
        str(cell["k"]),
        "--lr",
        str(cell["lr"]),
        "--out",
        str(cell_dir),
        "--input",
        str(INPUT),
        "--checkpoint",
        str(CHECKPOINT),
    ]
    if handle:
        command.append("--lock-held")
    if "tau" in cell:
        command.extend(("--tau", str(cell["tau"])))
    if "lambda_anchor" in cell:
        command.extend(("--lambda-anchor", str(cell["lambda_anchor"])))
    if paired_path:
        command.extend(("--paired-control-gains", str(paired_path)))
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        }
    )
    log_path = cell_dir / "run.log"
    note(
        f"START {cell['arm']} {cell['config']} seed{cell['seed']} {device}; "
        f"guard={json.dumps(guard, sort_keys=True)}; lock_cpu_fallback={cpu_fallback}"
    )
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(15),
        )
        rss_failed = False
        while process.poll() is None:
            current_rss = rss_kib(process.pid)
            if current_rss is not None and current_rss >= RSS_LIMIT_KIB:
                rss_failed = True
                process.terminate()
                break
            time.sleep(5)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if handle:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    if rss_failed:
        note(
            f"STOP after RSS exceeded 6 GiB in {cell['arm']} {cell['config']} seed{cell['seed']}; "
            f"cell incomplete, elapsed={elapsed:.1f}s; no later cells launched"
        )
        return False
    if return_code != 0:
        note(
            f"STOP after cell error {cell['arm']} {cell['config']} seed{cell['seed']} "
            f"exit={return_code}; elapsed={elapsed:.1f}s; log={log_path}"
        )
        return False
    summary_path = cell_dir / f"{cell['arm']}-{cell['config']}-seed{cell['seed']}.json"
    summary = json.loads(summary_path.read_text())
    if sha256(CHECKPOINT) != R882_SHA:
        note(
            f"STOP: r882 checkpoint SHA changed during {cell['arm']} {cell['config']} seed{cell['seed']}"
        )
        return False
    if (
        summary["dope_sha256_before"] != DOPE_SHA
        or summary["dope_sha256_after"] != DOPE_SHA
    ):
        note(f"STOP: DOPE identity mismatch after cell: {summary_path}")
        return False
    summary["preflight"] = guard
    summary["wall_seconds_coordinator"] = elapsed
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    note(
        f"DONE {cell['arm']} {cell['config']} seed{cell['seed']} device={device}; "
        f"wall={elapsed:.1f}s; paired_median={summary.get('paired_family_median_gain')}; "
        f"paired_LB95={summary.get('paired_family_bootstrap_lb95')}; accepted_provisional="
        f"{summary.get('accepted_provisional')}; validation_read=false"
    )
    if cell["arm"] == "FOCUS" and cell["config"] == "tau0.1" and cell["seed"] == 12:
        control = (
            RUNS / "CONTROL-k16-lr1e-4" / "seed12" / "CONTROL-k16-lr1e-4-seed12.json"
        )
        verify = subprocess.run(
            [
                sys.executable,
                str(repo / "research/flywheel_selfplay/h52/verify_slates.py"),
                str(control),
                str(summary_path),
            ],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if verify.returncode:
            note(f"STOP: matched slate identity check failed: {verify.stderr.strip()}")
            return False
        note(f"MATCHED SLATE PROOF {verify.stdout.strip()}")
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    if sha256(CHECKPOINT) != R882_SHA:
        raise SystemExit("offline r882 checkpoint SHA mismatch")
    RUNS.mkdir(parents=True, exist_ok=True)
    cells = configurations()
    note(
        f"H52 coordinator begins {len(cells)} preregistered cells; seeds={SEEDS}; base={R882_SHA}; "
        f"DOPE={DOPE_SHA}; repo_base=8d9f2fa2a05008b544e2aa8a925b67dc0e5a7055"
    )
    for index, cell in enumerate(cells, 1):
        note(
            f"CELL {index}/{len(cells)} queued: {cell['arm']} {cell['config']} seed{cell['seed']}"
        )
        if not run_cell(cell, repo):
            note(
                f"COORDINATOR STOPPED at {index}/{len(cells)}; completed evidence retained"
            )
            return
    note(
        "ALL 27 H52 training cells complete; run audit.py before any terminal validation read"
    )


if __name__ == "__main__":
    main()
