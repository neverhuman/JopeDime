#!/usr/bin/env python3
"""H93 FIFO attempts: retry only pre-scoring load voids, never reset the wall cap."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h93_cells", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
fair = r.load("h93_fifo", HERE.parent / "h61/run_wave.py")
CAP = 2700.0
MAX_RETRIES = 2
H80_CLOSEOUT = Path.home() / "jope_plan/ns/receipts/H80_closeout.json"


class HostLoadAbort(RuntimeError):
    def __init__(self, load):
        self.load = load
        super().__init__(f"host CPU above 85% load budget: {load:.2f}/102 cores")


def preflight():
    status = fair.safe_preflight()
    load = os.getloadavg()[0]
    if load > 85:
        raise HostLoadAbort(load)
    return dict(status, load1=load)


def cpu_gate(
    status,
    *,
    load_fn=os.getloadavg,
    sleep_fn=time.sleep,
    clock_fn=time.monotonic,
    max_wait=1800,
):
    started = clock_fn()
    consecutive = 0
    while True:
        load = float(load_fn()[0])
        consecutive = consecutive + 1 if math.isfinite(load) and load <= 85 else 0
        fair.note(status, f"CPU_GATE load1={load:.2f}/102; consecutive={consecutive}/2")
        if consecutive == 2:
            return clock_fn() - started
        if clock_fn() - started + 30 > max_wait:
            raise RuntimeError("CPU load gate did not stabilize within 30 minutes")
        sleep_fn(30)


def remaining(ledger):
    if ledger.get("in_progress"):
        raise RuntimeError("unaccounted launch: refusing budget reset/resume")
    value = CAP - sum(c["wall_seconds"] for c in ledger["attempts"])
    if value <= 10:
        raise RuntimeError("cumulative GPU wall exhausted (10s shutdown reserve)")
    return value


def retryable(record, phase, *, result_exists):
    return (
        record.get("error_kind") == "host_load"
        and record["attempt"] <= MAX_RETRIES
        and phase.get("outcome_started") is False
        and not result_exists
        and phase.get("attempt_token") == record["attempt_token"]
        and phase.get("arm") == record["arm"]
        and phase.get("seed") == record["seed"]
    )


def child_setup(parent_pid):
    os.nice(19)
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        os._exit(125)


def run_child(command, log_path, timeout, lease, record):
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    started = time.monotonic()
    parent_pid = os.getpid()
    with log_path.open("x") as log:
        proc = subprocess.Popen(
            command,
            cwd=r.ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lease.fileno(),),
            preexec_fn=lambda: child_setup(parent_pid),
        )
        record["child_pid"] = proc.pid
        try:
            while proc.poll() is None:
                rss = fair.process_rss_kib(proc.pid) or 0
                record["max_rss_kib"] = max(record.get("max_rss_kib", 0), rss)
                if rss >= fair.RSS_LIMIT_KIB:
                    raise RuntimeError("child RSS reached 6 GiB")
                if time.monotonic() - started >= timeout:
                    raise RuntimeError("remaining GPU wall expired")
                snapshot = preflight()
                record["max_load1"] = max(record.get("max_load1", 0), snapshot["load1"])
                record["max_host_cpu_percent"] = max(
                    record.get("max_host_cpu_percent", 0), snapshot["host_cpu_percent"]
                )
                record["min_mem_available_kib"] = min(
                    record.get("min_mem_available_kib", snapshot["mem_available_kib"]),
                    snapshot["mem_available_kib"],
                )
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f"cell exit {proc.returncode}; inspect {log_path}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def run_attempt(root, ledger, arm, seed, timing, attempt):
    timeout = remaining(ledger) - 10
    status = root / "H93_STATUS.log"
    gated = cpu_gate(status)
    waited_before = sum(a["wait_seconds"] for a in ledger["attempts"]) + gated
    if waited_before >= fair.MAX_LOCK_WAIT_SECONDS:
        raise RuntimeError("cumulative 3h resource/lock wait exhausted")
    lease, waited, free, snapshot = fair.acquire_lock(
        status,
        waited_before,
        after_release=True,
        wave_name="H93",
        preflight_fn=preflight,
    )
    started = time.monotonic()
    name = f"{'timing-' if timing else ''}{arm}-seed{seed}-attempt{attempt}"
    out = root / f"{name}.json"
    phase_path = out.with_suffix(".phase.json")
    token = f"{os.getpid()}-{time.time_ns()}-{name}"
    record = {
        "arm": arm,
        "seed": seed,
        "timing": timing,
        "attempt": attempt,
        "attempt_token": token,
        "wait_seconds": waited + gated,
        "cpu_gate_seconds": gated,
        "preflight": snapshot,
        "gpu_free_mib": free,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    error = None
    phase = {}
    try:
        ledger["in_progress"] = dict(record)
        r.atomic_json(root / "budget.json", ledger)
        r.atomic_json(
            phase_path,
            {
                "attempt_token": token,
                "arm": arm,
                "seed": seed,
                "stage": "launch",
                "outcome_started": False,
            },
        )
        command = [
            sys.executable,
            str(HERE / "runner.py"),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--phase",
            str(phase_path),
            "--attempt-token",
            token,
            "--attempt",
            str(attempt),
            "--lock-fd",
            str(lease.fileno()),
        ]
        if timing:
            command.append("--timing")
        record["command"] = command
        ledger["in_progress"]["command"] = command
        r.atomic_json(root / "budget.json", ledger)
        fair.note(status, f"RUN {name}; remaining_gpu_wall={timeout:.2f}s")
        run_child(
            command,
            root / f"{name}.log",
            max(0.1, timeout - (time.monotonic() - started)),
            lease,
            record,
        )
    except BaseException as exc:
        error = exc
        record["error"] = repr(exc)
        record["error_kind"] = (
            "host_load" if isinstance(exc, HostLoadAbort) else "fatal"
        )
        if isinstance(exc, HostLoadAbort):
            record["trigger_load1"] = exc.load
    finally:
        record["wall_seconds"] = time.monotonic() - started
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
    # Parse phase/receipts and persist accounting only after releasing the lease.
    if phase_path.exists():
        try:
            phase = json.loads(phase_path.read_text())
        except (ValueError, OSError):
            phase = {}
    record["outcome_started"] = phase.get("outcome_started")
    record["stage"] = phase.get("stage")
    ledger["attempts"].append(record)
    ledger["in_progress"] = None
    r.atomic_json(root / "budget.json", ledger)
    fair.note(
        status,
        f"RELEASE {name}; wall={record['wall_seconds']:.2f}s; error={record.get('error')}",
    )
    if error is not None:
        if retryable(record, phase, result_exists=out.exists()):
            return None, record
        raise error
    result = json.loads(out.read_text())
    if phase.get("outcome_started") is not True or phase.get("stage") != "complete":
        raise RuntimeError("successful child lacks a complete scoring boundary")
    record["receipt_sha256"] = r.h59.sha256(out)
    r.atomic_json(root / "budget.json", ledger)
    return result, record


def execute(root, ledger, arm, seed, timing=False, *, attempt_fn=run_attempt):
    for attempt in range(1, MAX_RETRIES + 2):
        result, record = attempt_fn(root, ledger, arm, seed, timing, attempt)
        if result is not None:
            return result, record
        fair.note(
            root / "H93_STATUS.log",
            f"INFRA_VOID pre-outcome load abort; retry at FIFO tail ({attempt}/{MAX_RETRIES} retries spent)",
        )
    raise RuntimeError("cell exhausted two pre-outcome host-load retries")


def validate_run_root(root):
    expected = Path.home() / "jope-data/h93/h93-wave"
    resolved = r.safe_path(root).resolve()
    if any(
        part.lower() in {"h76", "h76-wave", "h82", "h82-wave"}
        for part in resolved.parts
    ):
        raise RuntimeError("H76/H82 outcome custody is outside the H93 input closure")
    if resolved != expected.resolve():
        raise RuntimeError("H93 has one authorized run root; no alternate ledger/reset")


def validate_launch_barrier(
    closeout=H80_CLOSEOUT,
    queue_dir=fair.QUEUE_DIR,
    *,
    is_alive_fn=fair.fifo_lock.pid_is_alive,
):
    """Require the durable H80 closeout and a live H90 FIFO ticket before H93 can queue."""
    closeout = Path(closeout)
    queue_dir = Path(queue_dir)
    if closeout.is_symlink() or not closeout.is_file() or queue_dir.is_symlink():
        raise RuntimeError(
            "H93 requires a regular H80 closeout file and nonsymlink H90 queue directory"
        )
    try:
        record = json.loads(closeout.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("H93 blocked until the H80 closeout receipt exists") from exc
    if (
        record.get("schema") != "jope.h80.closeout/v1"
        or record.get("wave") != "H80"
        or record.get("complete") is not True
    ):
        raise RuntimeError("H93 blocked until H80 closeout is complete")
    for entry in queue_dir.iterdir() if queue_dir.exists() else ():
        if entry.is_symlink() or not entry.is_file():
            continue
        parsed = fair.fifo_lock.parse_ticket_name(entry.name)
        if parsed is None:
            continue
        epoch_ns, pid, wave = parsed
        if wave != "H90" or not is_alive_fn(pid):
            continue
        fields = dict(
            line.split("=", 1) for line in entry.read_text().splitlines() if "=" in line
        )
        if fields == {"epoch_ns": str(epoch_ns), "pid": str(pid), "wave": "H90"}:
            return {"h80_closeout": str(closeout), "h90_ticket": entry.name}
    raise RuntimeError("H93 blocked until H90 holds a live FIFO ticket")


def check_dependencies(cpu_pr):
    subprocess.run(["git", "fetch", "origin", "main"], cwd=r.ROOT, check=True)
    for pr in (832, 897, cpu_pr):
        info = json.loads(
            subprocess.check_output(
                [
                    "/home/ubuntu/.local/bin/gh",
                    "pr",
                    "view",
                    str(pr),
                    "--repo",
                    "neverhuman/JopeDime",
                    "--json",
                    "mergedAt,mergeCommit",
                ],
                text=True,
            )
        )
        if not info["mergedAt"]:
            raise RuntimeError(f"PR {pr} must merge before H93 FIFO launch")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", info["mergeCommit"]["oid"], "HEAD"],
            cwd=r.ROOT,
            check=True,
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(HERE)], cwd=r.ROOT, text=True
    ):
        raise RuntimeError("dirty H93 execution source")
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            merged = subprocess.check_output(
                ["git", "show", f"origin/main:{path.relative_to(r.ROOT)}"], cwd=r.ROOT
            )
            if merged != path.read_bytes():
                raise RuntimeError("H93 source is not the merged CPU protocol")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-pr", type=int)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path.home() / "jope-data/h93/h93-wave"
    )
    args = parser.parse_args()
    validate_run_root(args.root)
    if args.cpu_dry_run:
        print(
            json.dumps(
                {
                    "wave": "H93",
                    "arms": list(r.ARMS),
                    "seeds": list(r.SEEDS),
                    "updates_per_cell": 40,
                    "gpu_cell_wall_cap_seconds": CAP,
                    "reads_dev": False,
                    "created_fifo_ticket": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.cpu_pr is None:
        parser.error("--cpu-pr is required for execution")
    launch_barrier = validate_launch_barrier()
    check_dependencies(args.cpu_pr)
    args.root.mkdir(parents=True, exist_ok=True)
    status = args.root / "H93_STATUS.log"
    with (args.root / "coordinator.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root / "budget.json").exists():
            raise RuntimeError(
                "H93 already launched; do not reset its ledger or inspect H76/H82 outcomes"
            )
        ledger = {
            "wave": "H93",
            "cap_seconds": CAP,
            "attempts": [],
            "in_progress": None,
            "cpu_pr": args.cpu_pr,
            "launch_barrier": launch_barrier,
            "prereg_sha256": r.h59.sha256(HERE / "PREREG.md"),
        }
        r.atomic_json(args.root / "budget.json", ledger)
        try:
            timing, record = execute(args.root, ledger, "EB4", 30, True)
            charged = sum(a["wall_seconds"] for a in ledger["attempts"])
            prior_void = charged - record["wall_seconds"]
            projected = prior_void + record["wall_seconds"] * 13 * 1.25
            if projected > CAP:
                raise RuntimeError(f"timing projection {projected:.2f}s exceeds2700s")
            fair.note(
                status, f"TIMING_PASS projected={projected:.2f}s including all voids"
            )
            cells = []
            for seed in r.SEEDS:
                for arm in r.ARMS:
                    cell, record = execute(args.root, ledger, arm, seed)
                    cells.append(cell)
            metrics = r.summarize(cells)
            metrics.update(
                ledger=ledger,
                gpu_wall_seconds=sum(a["wall_seconds"] for a in ledger["attempts"]),
            )
            r.atomic_json(args.root / "metrics.json", metrics)
            fair.note(status, "COMPLETE all12 paired cells; metrics.json written")
        except BaseException as exc:
            r.atomic_json(
                args.root / "incomplete.json",
                {"complete": False, "reason": repr(exc), "ledger": ledger},
            )
            fair.note(status, f"STOP {exc}; missing outcomes unknown, no admission")
            raise


if __name__ == "__main__":
    main()
