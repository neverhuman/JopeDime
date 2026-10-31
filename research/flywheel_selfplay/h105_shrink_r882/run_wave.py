#!/usr/bin/env python3
"""H105 FIFO attempts: retry only pre-scoring load voids, never reset the wall cap."""

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
spec = importlib.util.spec_from_file_location("h105_cells", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
fair = None  # Loaded only by an explicit future --execute invocation.
CAP = 2700.0
MAX_RETRIES = 2


PRECURSORS = {"H99": "h99_esh_harm", "H100": "h100_asam_confirm"}


class PredecessorPending(RuntimeError):
    pass


def native_h99_started(ledger):
    """H99's published ledger has no wave/command keys; a completed GPU cell proves start."""
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema") != "jope.h99.gpu-budget/v1"
        or ledger.get("cap_seconds") != 9000.0
    ):
        return False
    for cell in ledger.get("attempts", []):
        if not isinstance(cell, dict):
            continue
        wall = cell.get("wall_seconds")
        result = cell.get("result", {})
        artifact = result.get("artifact_sha256", "")
        summary = Path(result.get("summary", "/"))
        if (
            cell.get("status") == "complete"
            and type(wall) in (int, float)
            and math.isfinite(wall)
            and wall > 0
            and type(cell.get("gpu_free_mib")) in (int, float)
            and cell["gpu_free_mib"] > 0
            and cell.get("preflight", {}).get("wave") == "H99"
            and len(artifact) == 64
            and all(c in "0123456789abcdef" for c in artifact)
            and summary.is_relative_to(Path.home() / "jope-data/h99/h99-wave/outcomes")
        ):
            return True
    return False


def merged_metric(wave):
    path = f"research/flywheel_selfplay/{PRECURSORS[wave]}/metrics.json"
    result = subprocess.run(
        ["git", "show", f"origin/main:{path}"],
        cwd=r.ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PredecessorPending(f"{wave} merged terminal metrics absent")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise PredecessorPending(f"{wave} terminal metadata invalid") from exc


def started_budget(wave):
    path = Path.home() / f"jope-data/{wave.lower()}/{wave.lower()}-wave/budget.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise PredecessorPending(f"{wave} scheduling ledger is a symlink")
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise PredecessorPending(f"{wave} scheduling ledger unreadable") from exc


def require_predecessors(
    *,
    read_fn=merged_metric,
    budget_fn=started_budget,
    queue_dir=None,
    alive_fn=None,
):
    """AFTER is FIFO start order: do not impose another closeout approval gate."""
    alive_fn = alive_fn or fair.fifo_lock.pid_is_alive
    queue_dir = queue_dir or Path.home() / "jope_plan/qualification.queue"
    if queue_dir.is_symlink():
        raise PredecessorPending("FIFO queue directory is a symlink")
    tickets = {}
    for ticket in queue_dir.iterdir() if queue_dir.exists() else ():
        parsed = fair.fifo_lock.parse_ticket_name(ticket.name)
        if parsed is None or parsed[2] not in PRECURSORS or not alive_fn(parsed[1]):
            continue
        if ticket.is_symlink() or not ticket.is_file():
            raise PredecessorPending("precursor ticket is not a regular file")
        fields = dict(
            line.split("=", 1)
            for line in ticket.read_text().splitlines()
            if "=" in line
        )
        epoch, pid, wave = parsed
        if fields != {"epoch_ns": str(epoch), "pid": str(pid), "wave": wave}:
            raise PredecessorPending(f"{wave} FIFO ticket metadata mismatch")
        tickets[wave] = ticket.name
    records = {}
    for wave in PRECURSORS:
        if wave in tickets:
            records[wave] = {"older_live_ticket": tickets[wave]}
            continue
        ledger = budget_fn(wave)
        if wave == "H99" and native_h99_started(ledger):
            records[wave] = {
                "prior_gpu_start": True,
                "proof": "native-completed-GPU-cell",
            }
            continue
        if (
            isinstance(ledger, dict)
            and ledger.get("wave") == wave
            and type(ledger.get("cap_seconds")) in (int, float)
            and math.isfinite(ledger["cap_seconds"])
            and ledger["cap_seconds"] > 0
        ):
            attempts = ledger.get("attempts", [])
            pending = ledger.get("in_progress")
            launched = [
                a
                for a in attempts
                if isinstance(a, dict) and a.get("started_utc") and a.get("command")
            ]
            if launched or (
                isinstance(pending, dict)
                and pending.get("started_utc")
                and pending.get("command")
            ):
                records[wave] = {"prior_gpu_start": True}
                continue
        try:
            terminal = read_fn(wave)
        except PredecessorPending:
            terminal = {}
        if (
            terminal.get("complete") is True
            and str(terminal.get("schema", "")).startswith(f"jope.{wave.lower()}.")
            and "cpu" not in str(terminal.get("schema", "")).lower()
        ):
            records[wave] = {"merged_complete": True}
            continue
        raise PredecessorPending(
            f"{wave} has no older live ticket, prior GPU start or merged complete result"
        )
    return records


def wait_predecessors(root, *, sleep_fn=time.sleep, check_fn=require_predecessors):
    status = root / "H105_STATUS.log"
    while True:
        subprocess.run(["git", "fetch", "origin", "main"], cwd=r.ROOT, check=True)
        try:
            proof = check_fn()
            r.atomic_json(root / "precursor-order.json", proof)
            fair.note(status, "PRECURSOR_ORDER H99/H100 preceded H105; join FIFO tail")
            return proof
        except PredecessorPending as error:
            fair.note(
                status, f"ARMED outside FIFO: {error}; no H105 ticket/CUDA/ledger"
            )
            sleep_fn(30)


class HostLoadAbort(RuntimeError):
    def __init__(self, load):
        self.load = load
        super().__init__(f"host CPU above 85% load budget: {load:.2f}/102 cores")


def preflight():
    usage = __import__("shutil").disk_usage("/")
    if 100 * (usage.total - usage.free) / usage.total >= 94.5:
        raise RuntimeError("unsafe filesystem / before H105 launch: usage >=94.5%")
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
    for record in ledger["attempts"]:
        for field in ("wall_seconds", "wait_seconds"):
            value = record.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError("invalid or unknown cumulative budget charge")
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
    require_predecessors()
    timeout = remaining(ledger) - 10
    status = root / "H105_STATUS.log"
    gated = cpu_gate(status)
    waited_before = sum(a["wait_seconds"] for a in ledger["attempts"]) + gated
    if waited_before >= fair.MAX_LOCK_WAIT_SECONDS:
        raise RuntimeError("cumulative 3h resource/lock wait exhausted")
    require_predecessors()
    lease, waited, free, snapshot = fair.acquire_lock(
        status,
        waited_before,
        after_release=True,
        wave_name="H105",
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
            "--attempt",
            str(attempt),
            "--phase",
            str(phase_path),
            "--attempt-token",
            token,
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
            root / "H105_STATUS.log",
            f"INFRA_VOID pre-outcome load abort; retry at FIFO tail ({attempt}/{MAX_RETRIES} retries spent)",
        )
    raise RuntimeError("cell exhausted two pre-outcome host-load retries")


def validate_run_root(root):
    expected = (Path.home() / "jope-data").resolve() / "h105" / "h105-wave"
    resolved = r.safe_path(root).resolve()
    if root.is_symlink() or expected.parent.is_symlink() or resolved != expected:
        raise RuntimeError(
            "H105 has one authorized run root; no aliases, alternate ledger or reset"
        )


def check_dependencies(cpu_pr, source_commit):
    subprocess.run(["git", "fetch", "origin", "main"], cwd=r.ROOT, check=True)
    for pr in (832, 1017, cpu_pr):
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
            raise RuntimeError(f"PR {pr} must merge before FIFO launch")
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                info["mergeCommit"]["oid"],
                source_commit,
            ],
            cwd=r.ROOT,
            check=True,
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "origin/main"],
        cwd=r.ROOT,
        check=True,
    )
    return r.verify_execution_source(source_commit)


def main():
    global fair
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-dry-run", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument(
        "--execute",
        action="store_true",
        help="future owner-authorized merged-source execution only",
    )
    parser.add_argument("--cpu-pr", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--shared-source-root", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / "jope-data/h105/h105-wave"
    )
    args = parser.parse_args()
    if args.cpu_dry_run or args.plan:
        result = (
            r.cpu_dry_run()
            if args.cpu_dry_run
            else {
                "schema": "jope.h105.plan/v1",
                "protocol": r.protocol(),
                "outcome_cells": [
                    {"arm": a, "seed": s, "scratch": f"scratch/{a}-seed{s}-attemptN/"}
                    for s in r.SEEDS
                    for a in r.ARMS
                ],
                "timing": {"arm": "ANCHOR", "seed": 42, "assessment_read": False},
                "GPU_ticket_created": False,
                "execution_authorized_by_this_task": False,
            }
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    if args.cpu_pr is None:
        parser.error("--execute requires the merged CPU PR number")
    if args.source_commit is None:
        parser.error("--execute requires an explicit merged source commit")
    r.configure_source(args.source_commit, args.shared_source_root)
    validate_run_root(args.root)
    source_proof = check_dependencies(args.cpu_pr, args.source_commit)
    r.runtime()
    fair = r.load(
        "h105_fifo", r.SOURCE_ROOT / "research/flywheel_selfplay/h61/run_wave.py"
    )
    args.root.mkdir(parents=True, exist_ok=True)
    status = args.root / "H105_STATUS.log"
    with (args.root / "coordinator.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root / "budget.json").exists():
            raise RuntimeError(
                "H105 already launched; do not reset its ledger or inspect H76 outcomes"
            )
        closure = wait_predecessors(args.root)
        ledger = {
            "predecessor_order": closure,
            "wave": "H105",
            "cap_seconds": CAP,
            "attempts": [],
            "in_progress": None,
            "cpu_pr": args.cpu_pr,
            "execution_source_proof": source_proof,
            "prereg_sha256": r.h59.sha256(HERE / "PREREG.md"),
        }
        r.atomic_json(args.root / "budget.json", ledger)
        try:
            timing, record = execute(args.root, ledger, "ANCHOR", 42, True)
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
