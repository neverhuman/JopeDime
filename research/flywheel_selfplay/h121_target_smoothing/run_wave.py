#!/usr/bin/env python3
"""Owner-authorized H121 FIFO coordinator; CPU dry-run never enqueues or launches."""

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
ROOT = HERE.parents[2]
H97_PATH = HERE.parent / "h97_reg_ema" / "run_wave.py"
h97 = importlib.util.spec_from_file_location("h121_h97_coordinator", H97_PATH)
if h97 is None or h97.loader is None:
    raise ImportError("cannot load reviewed H97 FIFO machinery")
h97_module = importlib.util.module_from_spec(h97)
sys.modules[h97.name] = h97_module
h97.loader.exec_module(h97_module)
fair = h97_module.fair
local_runner = importlib.util.spec_from_file_location("h121_runner", HERE / "runner.py")
if local_runner is None or local_runner.loader is None:
    raise ImportError("cannot load H121 runner")
r = importlib.util.module_from_spec(local_runner)
sys.modules[local_runner.name] = r
local_runner.loader.exec_module(r)
CAP = 2700.0
MAX_RETRIES = 2
PREDECESSOR_PRS = (1011, 994, 997, 1017, 1007, 1048, 1049, 1059)


class HostLoadAbort(RuntimeError):
    def __init__(self, load):
        self.load = load
        super().__init__(f"host load above 85/102: {load:.2f}")


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
        if consecutive >= 2:
            return clock_fn() - started
        if clock_fn() - started + 30 > max_wait:
            raise RuntimeError("H121 CPU load gate did not stabilize")
        sleep_fn(30)


def active_wave_ticket(wave):
    q = fair.QUEUE_DIR
    if q.is_symlink() or not q.is_dir():
        raise RuntimeError("qualification queue must be a regular directory")
    for entry in q.iterdir():
        parsed = fair.fifo_lock.parse_ticket_name(entry.name)
        if (
            not parsed
            or parsed[2] != wave
            or not fair.fifo_lock.pid_is_alive(parsed[1])
        ):
            continue
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"unsafe {wave} queue ticket")
        fields = dict(
            line.split("=", 1) for line in entry.read_text().splitlines() if "=" in line
        )
        if fields != {"epoch_ns": str(parsed[0]), "pid": str(parsed[1]), "wave": wave}:
            raise RuntimeError(f"invalid {wave} queue ticket metadata")
        return entry.name
    return None


def closeout_records_complete(
    wave, budget, metrics, *, active_ticket=False, owner_marker=False
):
    if wave not in ("H99", "H100"):
        raise ValueError("H121 has only H99/H100 live execution predecessors")
    if active_ticket or owner_marker:
        return False
    attempts = budget.get("attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or budget.get("in_progress") is not None
    ):
        return False
    for item in attempts:
        if not isinstance(item, dict):
            return False
        if item.get("status") == "complete":
            continue
        safe_h100_void = (
            wave == "H100"
            and item.get("error_kind") == "host_load"
            and item.get("outcome_started") is False
            and not item.get("receipt_sha256")
        )
        if not safe_h100_void:
            return False
    if wave == "H99":
        return metrics.get("status") == "complete"
    return metrics.get("complete") is True


def predecessor_complete(wave):
    root = Path.home() / f"jope-data/{wave.lower()}/{wave.lower()}-wave"
    budget_path = root / "budget.json"
    if wave == "H99":
        metrics_path = root / "outcomes/admission.json"
    elif wave == "H100":
        metrics_path = root / "metrics.json"
    else:
        raise ValueError("H121 has only H99/H100 live execution predecessors")
    if root.is_symlink() or budget_path.is_symlink() or metrics_path.is_symlink():
        raise RuntimeError(f"unsafe {wave} completion record")
    active_ticket = bool(active_wave_ticket(wave))
    owner_marker = bool(list(fair.QUEUE_DIR.glob(f"{wave}-owner-*.json")))
    if not budget_path.is_file() or not metrics_path.is_file():
        return False
    try:
        budget = json.loads(budget_path.read_text())
        metrics = json.loads(metrics_path.read_text())
    except (OSError, ValueError):
        return False
    return closeout_records_complete(
        wave, budget, metrics, active_ticket=active_ticket, owner_marker=owner_marker
    )


def validate_authorization_payload(payload, source_sha):
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "jope.h121.gpu-authorization/v1"
    ):
        raise RuntimeError("H121 requires a valid owner GPU-authorization receipt")
    if payload.get("wave") != "H121" or payload.get("approved") is not True:
        raise RuntimeError("H121 GPU-authorization receipt is not approved")
    if not isinstance(payload.get("owner"), str) or not payload["owner"].strip():
        raise RuntimeError("H121 authorization receipt must identify its owner")
    if payload.get("source_commit") != source_sha:
        raise RuntimeError(
            "H121 authorization receipt does not match the exact source commit"
        )


def read_authorization_receipt(path, source_sha):
    receipt = Path(path)
    expected_parent = Path.home() / "jope_plan/ns/receipts"
    if receipt.resolve().parent != expected_parent.resolve():
        raise RuntimeError(
            "H121 authorization receipt must be stored in master receipt custody"
        )
    if receipt.is_symlink() or not receipt.is_file():
        raise RuntimeError("H121 owner authorization receipt must be a regular file")
    try:
        payload = json.loads(receipt.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "H121 owner authorization receipt is unreadable/invalid"
        ) from exc
    validate_authorization_payload(payload, source_sha)
    return {
        "path": str(receipt),
        "sha256": r.h59.sha256(receipt),
        "owner": payload["owner"],
    }


def require_scratch_budget(root, limit_bytes=2_000_000_000):
    scratch = Path(root) / "scratch"
    if scratch.is_symlink():
        raise RuntimeError("H121 scratch root may not be a symlink")
    if not scratch.exists():
        return 0
    total = 0
    for path in scratch.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("H121 scratch tree contains a symlink")
        if path.is_file():
            total += path.stat().st_size
            if total > limit_bytes:
                raise RuntimeError("H121 scratch exceeds the 2 GB wave limit")
    return total


def validate_launch_barrier():
    for wave in ("H99", "H100"):
        if not predecessor_complete(wave):
            raise RuntimeError(
                f"H121 blocked until {wave} has a complete closeout and released FIFO state"
            )
    return {
        "H99_complete": True,
        "H100_complete": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }


def check_dependencies(cpu_pr):
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
    main_sha = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head_sha != main_sha:
        raise RuntimeError(
            "H121 launch must run from the exact fetched origin/main checkout"
        )
    for pr in (*PREDECESSOR_PRS, cpu_pr):
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
        if not info.get("mergedAt") or not info.get("mergeCommit"):
            raise RuntimeError(f"PR #{pr} must be merged before H121 FIFO entry")
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                info["mergeCommit"]["oid"],
                "origin/main",
            ],
            cwd=ROOT,
            check=True,
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(HERE)], cwd=ROOT, text=True
    ):
        raise RuntimeError("dirty H121 execution source")
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            merged = subprocess.check_output(
                ["git", "show", f"origin/main:{path.relative_to(ROOT)}"], cwd=ROOT
            )
            if merged != path.read_bytes():
                raise RuntimeError(
                    f"H121 execution source differs from origin/main: {path.name}"
                )


def validate_root(root):
    expected = Path.home() / "jope-data/h121/h121-wave"
    resolved = Path(root).resolve()
    if Path(root).is_symlink() or resolved != expected.resolve():
        raise RuntimeError("H121 permits only ~/jope-data/h121/h121-wave as run root")
    if any(
        part.lower() in {"sealed", "holdout", "meta-test", "xbabe1"}
        for part in resolved.parts
    ):
        raise RuntimeError("forbidden H121 run-root path")


def remaining(ledger):
    if ledger.get("in_progress"):
        raise RuntimeError("unaccounted H121 launch; refuse resume/reset")
    left = CAP - sum(float(a["wall_seconds"]) for a in ledger["attempts"])
    if left <= 10:
        raise RuntimeError("H121 cumulative GPU wall exhausted")
    return left


def retryable(record, phase, result_exists):
    return (
        record.get("error_kind") == "host_load"
        and record["attempt"] <= MAX_RETRIES
        and phase.get("outcome_started") is False
        and not result_exists
        and phase.get("attempt_token") == record["attempt_token"]
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
            cwd=ROOT,
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
                    raise RuntimeError("H121 child RSS reached 6 GiB")
                if time.monotonic() - started >= timeout:
                    raise RuntimeError("H121 remaining GPU wall expired")
                snap = preflight()
                record["max_load1"] = max(record.get("max_load1", 0), snap["load1"])
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(
                    f"H121 cell exit {proc.returncode}; inspect {log_path}"
                )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def run_attempt(root, ledger, arm, seed, timing, attempt):
    require_scratch_budget(root)
    timeout = remaining(ledger) - 10
    status = root / "H121_STATUS.log"
    gated = cpu_gate(status)
    waited_before = sum(a["wait_seconds"] for a in ledger["attempts"]) + gated
    lease, waited, free, snap = fair.acquire_lock(
        status,
        waited_before,
        after_release=True,
        wave_name="H121",
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
        "preflight": snap,
        "gpu_free_mib": free,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    err = None
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
        err = exc
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
    if err is not None:
        if retryable(record, phase, out.exists()):
            return None, record
        raise err
    result = json.loads(out.read_text())
    if phase.get("outcome_started") is not True or phase.get("stage") != "complete":
        raise RuntimeError("H121 cell lacks complete scoring boundary")
    record["receipt_sha256"] = r.h59.sha256(out)
    r.atomic_json(root / "budget.json", ledger)
    return result, record


def execute(root, ledger, arm, seed, timing=False):
    for attempt in range(1, MAX_RETRIES + 2):
        result, record = run_attempt(root, ledger, arm, seed, timing, attempt)
        if result is not None:
            return result, record
        fair.note(
            root / "H121_STATUS.log",
            f"INFRA_VOID pre-scoring host-load abort; FIFO tail retry {attempt}/{MAX_RETRIES}",
        )
    raise RuntimeError("H121 cell exhausted host-load retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-pr", type=int)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--authorized-execution", action="store_true")
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / "jope-data/h121/h121-wave"
    )
    args = parser.parse_args()
    validate_root(args.root)
    if args.cpu_dry_run:
        print(
            json.dumps(
                {
                    "wave": "H121",
                    "arms": list(r.ARMS),
                    "seeds": list(r.SEEDS),
                    "updates_per_cell": 40,
                    "cells": 9,
                    "timing_cells": 1,
                    "gpu_cell_wall_cap_seconds": CAP,
                    "authorization_required": True,
                    "fifo_ticket_created": False,
                    "run_root_created": False,
                    "data_or_checkpoint_read": False,
                },
                sort_keys=True,
            )
        )
        return
    if (
        args.cpu_pr is None
        or not args.authorized_execution
        or args.authorization_receipt is None
    ):
        parser.error(
            "execution requires --cpu-pr, --authorized-execution and an owner authorization receipt"
        )
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("H121 coordinator must run at nice 10 or lower priority")
    validate_root(args.root)
    barrier = validate_launch_barrier()
    check_dependencies(args.cpu_pr)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    authorization = read_authorization_receipt(args.authorization_receipt, source_sha)
    args.root.parent.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=False, exist_ok=False)
    status = args.root / "H121_STATUS.log"
    with (args.root / "coordinator.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = args.root / "budget.json"
        ledger = {
            "wave": "H121",
            "cap_seconds": CAP,
            "attempts": [],
            "in_progress": None,
            "cpu_pr": args.cpu_pr,
            "prereg_sha256": r.h59.sha256(HERE / "PREREG.md"),
            "launch_barrier": barrier,
            "authorization_receipt": authorization,
        }
        r.atomic_json(budget, ledger)
        try:
            timing, rec = execute(args.root, ledger, "TS-0.3", 47, True)
            if sum(a["wall_seconds"] for a in ledger["attempts"]) * 1.25 * 10 > CAP:
                raise RuntimeError(
                    "timing projection exceeds 2700 second cell-wall cap"
                )
            cells = []
            for seed in r.SEEDS:
                for arm in r.ARMS:
                    cell, _ = execute(args.root, ledger, arm, seed, False)
                    cells.append(cell)
            metrics = r.summarize(cells)
            metrics["ledger"] = ledger
            metrics["gpu_wall_seconds"] = sum(
                a["wall_seconds"] for a in ledger["attempts"]
            )
            r.atomic_json(args.root / "metrics.json", metrics)
            fair.note(status, "COMPLETE all nine paired cells")
        except BaseException as exc:
            r.atomic_json(
                args.root / "incomplete.json",
                {"complete": False, "reason": repr(exc), "ledger": ledger},
            )
            fair.note(status, f"STOP {exc}; missing outcomes unknown, no admission")
            raise


if __name__ == "__main__":
    main()
