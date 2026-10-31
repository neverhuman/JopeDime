#!/usr/bin/env python3
"""H122 six-cell coordinator and read-only, receipt-bound metrics collector.

The CPU preparation grant does not authorize --execute. No timing cell or retry.
"""

from __future__ import annotations

import argparse
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
import uuid

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h122_wave_cells", HERE / "runner.py")
r = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r
_spec.loader.exec_module(r)
CAP = 2700.0
WAIT_CAP = 10800.0
PREDECESSORS = ("H99", "H100", "H105", "H109", "H121")
CELLS = tuple((arm, seed) for seed in r.SEEDS for arm in r.ARMS)
fair = r.h61wave


def source_identity() -> dict:
    r.validate_source_pins()
    paths = [str(HERE.relative_to(r.ROOT)), *r.SOURCE_PINS]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=r.ROOT, text=True
    )
    if dirty:
        raise RuntimeError("H122 execution source must be clean")
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r.ROOT, text=True
        ).strip(),
        "protocol_sha256": r.protocol_sha256(),
        "prereg_sha256": r.sha256(HERE / "PREREG.md"),
        "runner_sha256": r.sha256(HERE / "runner.py"),
        "coordinator_sha256": r.sha256(HERE / "run_wave.py"),
    }


def check_predecessors(paths: dict[str, Path]) -> dict:
    """Read operator closeout attestations; never inspect predecessor outcome arrays."""
    if set(paths) != set(PREDECESSORS):
        raise RuntimeError("all five predecessor closeouts are required before FIFO")
    result = {}
    for wave in PREDECESSORS:
        path = r.safe_path(paths[wave])
        value = json.loads(path.read_text())
        if (
            value.get("wave") != wave
            or value.get("closed") is not True
            or value.get("qualification_lock_released") is not True
            or value.get("in_progress") is not None
        ):
            raise RuntimeError(f"{wave} is not closed with its lease released")
        result[wave] = {"path": str(path.resolve()), "sha256": r.sha256(path)}
    return result


def remaining(ledger: dict) -> float:
    if ledger.get("in_progress") is not None:
        raise RuntimeError("unknown reservation; budget cannot be reset or resumed")
    charged = [item["wall_seconds"] for item in ledger["attempts"]]
    if any(not math.isfinite(value) or value < 0 for value in charged):
        raise RuntimeError("invalid charged wall time")
    return CAP - sum(charged)


def cpu_gate(deadline: float) -> float:
    started = time.monotonic()
    ceiling = min(85.0, 0.85 * (os.cpu_count() or 1))
    consecutive = 0
    while time.monotonic() < deadline:
        load = os.getloadavg()[0]
        consecutive = consecutive + 1 if math.isfinite(load) and load <= ceiling else 0
        if consecutive == 2:
            return time.monotonic() - started
        if time.monotonic() + 30 >= deadline:
            break
        time.sleep(30)
    raise RuntimeError("cumulative resource wait exhausted before CPU admission")


def run_child(command, log_path, timeout, lease, record):
    """Supervise only our new child's session; no foreign PID is accepted."""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    started = time.monotonic()
    with log_path.open("x") as log:
        proc = subprocess.Popen(
            command,
            cwd=r.ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lease.fileno(),),
            start_new_session=True,
        )
        record["child_pid"] = proc.pid
        try:
            while proc.poll() is None:
                fair.safe_preflight()
                rss = fair.process_rss_kib(proc.pid) or 0
                record["max_rss_kib"] = max(record.get("max_rss_kib", 0), rss)
                if rss >= fair.RSS_LIMIT_KIB or time.monotonic() - started >= timeout:
                    raise RuntimeError("H122 child memory/wall limit reached")
                time.sleep(0.2)
            if proc.returncode:
                raise RuntimeError(
                    f"H122 child exit {proc.returncode}; inspect {log_path}"
                )
        finally:
            # The session was created here and contains only our worker/descendants.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def run_attempt(root: Path, ledger: dict, arm: str, seed: int, deadline: float) -> None:
    available = remaining(ledger)
    if available <= 10 or (arm, seed) != CELLS[len(ledger["attempts"])]:
        raise RuntimeError("H122 exhausted budget or noncanonical cell order")
    waited = cpu_gate(deadline)
    before = time.monotonic()
    # Account for every CPU gate, lock wait and bookkeeping interval across cells.
    elapsed_wait = WAIT_CAP - max(0.0, deadline - before)
    lease, lock_wait, _, snapshot = fair.acquire_lock(
        root / "H122_STATUS.log", elapsed_wait, wave_name="H122", after_release=True
    )
    started = time.monotonic()
    out = root / f"{arm}-seed{seed}-attempt1.json"
    phase_path = out.with_suffix(".phase.json")
    token = uuid.uuid4().hex
    record = {
        "arm": arm,
        "seed": seed,
        "attempt": 1,
        "attempt_token": token,
        "wait_seconds": waited + lock_wait,
        "preflight": snapshot,
    }
    error = None
    try:
        ledger["in_progress"] = dict(record)
        r.atomic_json(root / "budget.json", ledger)
        r.atomic_json(
            phase_path, {**record, "stage": "launch", "outcome_started": False}
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
            "1",
            "--lock-fd",
            str(lease.fileno()),
            "--parent-pid",
            str(os.getpid()),
        ]
        run_child(command, out.with_suffix(".log"), available - 10, lease, record)
        cell = json.loads(out.read_text())
        phase = json.loads(phase_path.read_text())
        r.validate_cell(cell)
        expected = {
            key: record[key] for key in ("arm", "seed", "attempt", "attempt_token")
        }
        if any(cell.get(k) != v or phase.get(k) != v for k, v in expected.items()) or (
            phase.get("stage") != "complete"
            or phase.get("outcome_started") is not True
            or cell["source_commit"] != ledger["identity"]["source_commit"]
        ):
            raise RuntimeError("H122 result/phase/source differs from reservation")
        record.update(receipt_sha256=r.sha256(out), phase_sha256=r.sha256(phase_path))
    except BaseException as exc:
        error = exc
        record["error"] = repr(exc)
    finally:
        record["wall_seconds"] = time.monotonic() - started
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
        ledger["attempts"].append(record)
        ledger["in_progress"] = None
        r.atomic_json(root / "budget.json", ledger)
    if error is not None:
        raise error
    if remaining(ledger) < 0:
        raise RuntimeError("H122 cumulative wall cap exceeded")


def collect(root: Path, identity: dict) -> dict:
    """Read only exact admitted attempt paths, then compute the complete-panel readout."""
    root = r.safe_path(root)
    ledger = json.loads((root / "budget.json").read_text())
    if (
        ledger.get("wave") != "H122"
        or ledger.get("identity") != identity
        or ledger.get("cap_seconds") != CAP
        or ledger.get("wait_cap_seconds") != WAIT_CAP
        or len(ledger.get("attempts", [])) != len(CELLS)
        or remaining(ledger) < 0
        or not math.isfinite(ledger.get("resource_wait_seconds", float("nan")))
        or not 0 <= ledger["resource_wait_seconds"] <= WAIT_CAP
    ):
        raise RuntimeError("INCOMPLETE: exact H122 identity/budget/panel required")
    predecessors = ledger.get("predecessors", {})
    if (
        set(predecessors) != set(PREDECESSORS)
        or check_predecessors(
            {wave: Path(record["path"]) for wave, record in predecessors.items()}
        )
        != predecessors
    ):
        raise RuntimeError("INCOMPLETE: predecessor closeout identity changed")
    cells = []
    for record, (arm, seed) in zip(ledger["attempts"], CELLS, strict=True):
        if (record.get("arm"), record.get("seed"), record.get("attempt")) != (
            arm,
            seed,
            1,
        ) or (
            record.get("error")
            or not record.get("receipt_sha256")
            or not record.get("phase_sha256")
        ):
            raise RuntimeError("INCOMPLETE: failed or unadmitted attempt")
        path = root / f"{arm}-seed{seed}-attempt1.json"
        phase_path = path.with_suffix(".phase.json")
        if (
            r.sha256(path) != record["receipt_sha256"]
            or r.sha256(phase_path) != record["phase_sha256"]
        ):
            raise RuntimeError("INCOMPLETE: receipt or phase bytes changed")
        cell, phase = json.loads(path.read_text()), json.loads(phase_path.read_text())
        if any(
            cell.get(k) != record.get(k) or phase.get(k) != record.get(k)
            for k in ("arm", "seed", "attempt", "attempt_token")
        ) or (
            phase.get("stage") != "complete"
            or phase.get("outcome_started") is not True
            or cell.get("source_commit") != identity["source_commit"]
        ):
            raise RuntimeError("INCOMPLETE: committed phase/reservation differs")
        cells.append(cell)
    return {
        **r.summarize(cells),
        "source_identity": identity,
        "gpu_wall_seconds": CAP - remaining(ledger),
        "resource_wait_seconds": ledger["resource_wait_seconds"],
    }


def execute(root: Path, predecessors: dict, identity: dict) -> dict:
    root = r.safe_path(root)
    # Predecessor gates precede run-root, ledger, FIFO and CUDA effects.
    closeouts = check_predecessors(predecessors)
    fair.safe_preflight()
    root.mkdir(parents=True, exist_ok=False)
    os.setpriority(os.PRIO_PROCESS, 0, max(19, os.getpriority(os.PRIO_PROCESS, 0)))
    with (root / "coordinator.lock").open("x") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = {
            "wave": "H122",
            "identity": identity,
            "cap_seconds": CAP,
            "wait_cap_seconds": WAIT_CAP,
            "predecessors": closeouts,
            "attempts": [],
            "in_progress": None,
            "resource_wait_seconds": 0.0,
        }
        r.atomic_json(root / "budget.json", ledger)
        deadline = time.monotonic() + WAIT_CAP
        try:
            for arm, seed in CELLS:
                before = time.monotonic()
                run_attempt(root, ledger, arm, seed, deadline)
                # GPU time is charged only to CAP, all other elapsed time to WAIT_CAP.
                deadline += ledger["attempts"][-1]["wall_seconds"]
                ledger["resource_wait_seconds"] += max(
                    0.0,
                    time.monotonic() - before - ledger["attempts"][-1]["wall_seconds"],
                )
                r.atomic_json(root / "budget.json", ledger)
            metrics = collect(root, identity)
            r.atomic_json(root / "metrics.json", metrics)
            return metrics
        except BaseException as exc:
            r.atomic_json(
                root / "incomplete.json", {"complete": False, "reason": repr(exc)}
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-dry-run", action="store_true")
    mode.add_argument("--collect", action="store_true")
    mode.add_argument(
        "--execute", action="store_true", help="requires separate owner launch grant"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--predecessor", action="append", default=[], metavar="WAVE=JSON"
    )
    args = parser.parse_args()
    if args.cpu_dry_run:
        print(json.dumps(r.cpu_dry_run(), sort_keys=True, allow_nan=False))
        return
    if args.root is None:
        parser.error("--root is required")
    identity = source_identity()
    if args.collect:
        print(json.dumps(collect(args.root, identity), sort_keys=True, allow_nan=False))
        return
    predecessors = {}
    for value in args.predecessor:
        wave, separator, path = value.partition("=")
        if not separator or wave in predecessors:
            parser.error("predecessor must be unique WAVE=JSON")
        predecessors[wave] = Path(path)
    execute(args.root, predecessors, identity)


if __name__ == "__main__":
    main()
