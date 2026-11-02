#!/usr/bin/env python3
"""Prepare Q1 intent metadata, or launch an admitted CPU panel through the shared FIFO."""

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import time

import q1
import q1_cut
from fifo_launch_prepare import (
    DOPE,
    ROOT,
    Paths,
    build_input_manifest,
    dry_run,
    load_amended_roster_packet,
    load_input_manifest,
    planning,
    prepare,
    write_new,
)

MAX_WAIT = 10800




def fifo_module(paths):
    spec = importlib.util.spec_from_file_location("q1_fifo", paths.fifo)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def limits(value, upper, name):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    q1.require(
        finite and 0.01 <= value <= upper,
        name + " must be finite, at least 0.01 seconds and within its bound",
    )














def source_identity(paths, source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(paths.root), *args])

    q1.require(
        git("rev-parse", "HEAD").decode().strip() == source, "signed source differs"
    )
    q1.require(
        not git("status", "--porcelain", "--untracked-files=all").strip(),
        "qualification requires a clean checkout",
    )
    git("merge-base", "--is-ancestor", source, "origin/main")
    for path in (Path(__file__), paths.driver, paths.prereg, paths.fifo):
        relative = path.relative_to(paths.root).as_posix()
        q1.require(
            not path.is_symlink()
            and path.read_bytes() == git("show", source + ":" + relative),
            "launcher input differs from the signed source: " + relative,
        )


def cpu_gate(load):
    q1.require(platform.node().split(".")[0] != "xbabe0", "xbabe0 is forbidden")
    q1.require(load() <= 85, "host load exceeds 85/102; no study dispatch")


def admit(args, paths, source_check=source_identity, load=lambda: os.getloadavg()[0]):
    cpu_gate(load)
    limits(args.wait_seconds, MAX_WAIT, "wait cap")
    files = {
        key: Path(getattr(args, key)).resolve(strict=True)
        for key in ("plan", "backend", "signature", "allowed_signers")
    }
    plan = q1.load(files["plan"])
    q1.validate_plan(plan, paths.prereg)
    q1.require(plan["mode"] == "qualification", "FIFO launcher refuses smoke plans")
    q1.require(plan["pins"]["dope_sha256"] == DOPE, "frozen DOPE differs")
    limits(
        args.wall_seconds,
        planning(len(plan["datasets"]))["serial_timeout_seconds"],
        "wall cap",
    )
    q1.require(
        q1.sha_file(files["backend"]) == plan["backend_sha256"], "backend hash differs"
    )
    q1.require(os.access(files["backend"], os.X_OK), "backend is not executable")
    q1.verify_owner(plan, files["signature"], files["allowed_signers"], paths.prereg)
    source_check(paths, plan.get("metadata_source_commit", plan["source_commit"]))
    ledger = Path(args.ledger)
    q1.require(
        ledger.is_absolute()
        and str(ledger) == plan["ledger_root"] == str(ledger.resolve()),
        "ledger must be the canonical owner-signed path",
    )
    for suffix in ("started", "finished"):
        marker = ledger / (plan["pins"]["bundle_sha256"] + "." + suffix + ".json")
        q1.require(
            not marker.exists() and not marker.is_symlink(),
            "candidate already consumed; no retry",
        )
    hashes = {key: q1.sha_file(path) for key, path in files.items()}
    hashes["resolved_paths"] = {key: str(path) for key, path in files.items()}
    hashes.update(
        {
            str(path): q1.sha_file(path)
            for path in (paths.driver, paths.prereg, paths.fifo, Path(__file__))
        }
    )
    return plan, files, hashes


def oldest_readonly(fifo, queue):
    live = []
    for path in queue.iterdir():
        parsed = fifo.parse_ticket_name(path.name)
        if (
            parsed
            and path.is_file()
            and not path.is_symlink()
            and fifo.pid_is_alive(parsed[1])
        ):
            live.append((*parsed, path))
    return min(live)[3] if live else None


@contextmanager
def lease(paths, fifo, wait_seconds, load, sleep=time.sleep, poll=30, backoff=45):
    q1.require(
        paths.queue.is_dir() and not paths.queue.is_symlink(),
        "existing FIFO directory required",
    )
    descriptor = os.open(paths.lock, os.O_RDONLY | os.O_NOFOLLOW)
    handle = os.fdopen(descriptor, "rb")
    ticket = None
    locked = False
    deadline = time.monotonic() + wait_seconds
    try:
        q1.require(
            stat.S_ISREG(os.fstat(descriptor).st_mode), "existing regular lock required"
        )
        ticket = fifo.enqueue_ticket(paths.queue, wave="Q1")
        # Give legacy waiters courtesy without searching their private run directories.
        sleep(min(backoff, max(0, deadline - time.monotonic())))
        while time.monotonic() < deadline:
            if load() <= 85 and oldest_readonly(fifo, paths.queue) == ticket:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    pass
                if locked:
                    same = os.fstat(descriptor)
                    current = paths.lock.lstat()
                    q1.require(
                        (same.st_dev, same.st_ino) == (current.st_dev, current.st_ino),
                        "qualification lock was replaced",
                    )
                    if load() <= 85 and oldest_readonly(fifo, paths.queue) == ticket:
                        yield handle, ticket
                        return
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    locked = False
            sleep(min(poll, max(0, deadline - time.monotonic())))
        raise TimeoutError("FIFO wait cap exhausted; study was not started")
    finally:
        if locked:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        fifo.remove_ticket(ticket)  # Only this invocation's numeric PID ticket.
        q1.require(
            ticket is None or not ticket.exists(), "own FIFO ticket cleanup failed"
        )


@contextmanager
def termination_handler():
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum, _frame):
        raise InterruptedError("launcher terminated; retain original study receipts")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_owned(command, log, environment, seconds, lock_fd, load, grace=5):
    limits(seconds, float("inf"), "remaining wall cap")
    cpu_gate(load)
    with log.open("xb") as stream:
        child = subprocess.Popen(
            ["nice", "-n", "10", *command],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
        deadline = time.monotonic() + seconds
        exited = False
        try:
            # Never poll()/reap before signalling the whole owned group.
            while (
                os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is None
            ):
                cpu_gate(load)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Q1 job wall cap exhausted; do not restart the candidate"
                    )
                time.sleep(0.05)
            exited = True
        finally:
            previous = {
                sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                # Backend invocations use their own process groups in this
                # session. Clean the whole private session before the lease
                # can unwind, while child.pid remains unreaped and pins the
                # session identity against reuse.
                q1.terminate_session(child.pid, grace if not exited else 0)
                child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        return child.returncode


def launch(
    args,
    paths=Paths(),
    source_check=source_identity,
    load=lambda: os.getloadavg()[0],
    lease_options=None,
    grace=5,
):
    q1.require(
        args.authorized_launch,
        "explicit --authorized-launch required; OWNER signature also required",
    )
    plan, files, frozen = admit(args, paths, source_check, load)
    folder = Path(args.run_dir)
    ledger = Path(args.ledger)
    q1.require(
        folder.is_absolute()
        and str(folder) == str(folder.resolve())
        and not folder.exists()
        and not folder.is_symlink()
        and not folder.is_relative_to(ledger)
        and not ledger.is_relative_to(folder),
        "new canonical run directory separate from the persistent ledger required",
    )
    fifo = fifo_module(paths)
    folder.mkdir(mode=0o700)
    receipt = {
        "schema": "jope.q1-launch/v1",
        "source_commit": plan["source_commit"],
        "inputs": frozen,
        "planning": planning(len(plan["datasets"])),
        "wait_cap_seconds": args.wait_seconds,
        "wall_cap_seconds": args.wall_seconds,
        "status": "not_started",
    }
    started = time.monotonic()
    try:
        with (
            termination_handler(),
            lease(paths, fifo, args.wait_seconds, load, **(lease_options or {})) as (
                handle,
                ticket,
            ),
        ):
            receipt.update(
                ticket=ticket.name,
                queue_wait_seconds=time.monotonic() - started,
                status="locked",
            )
            deadline = time.monotonic() + args.wall_seconds
            _, _, current = admit(args, paths, source_check, load)
            q1.require(current == frozen, "admission inputs changed while waiting")
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                PYTHONDONTWRITEBYTECODE="1",
                RAYON_NUM_THREADS="8",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                AWS_EC2_METADATA_DISABLED="true",
            )
            for key in (
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_PROFILE",
                "JOPE_AWS_PROFILE",
            ):
                environment.pop(key, None)
            common = [
                "--plan",
                str(files["plan"]),
                "--signature",
                str(files["signature"]),
                "--allowed-signers",
                str(files["allowed_signers"]),
            ]
            output = folder / "receipts.jsonl"
            command = [
                sys.executable,
                str(paths.driver),
                "run",
                *common,
                "--backend",
                str(files["backend"]),
                "--out",
                str(output),
                "--ledger",
                str(ledger),
            ]
            receipt["status"] = "driver_started_outcome_unknown"
            code = run_owned(
                command,
                folder / "run.log",
                environment,
                deadline - time.monotonic(),
                handle.fileno(),
                load,
                grace,
            )
            q1.require(
                code == 0,
                "Q1 run failed; retain original ledger/receipts, no automatic retry",
            )
            marker = ledger / (plan["pins"]["bundle_sha256"] + ".finished.json")
            finished = q1.load(marker)
            q1.require(
                finished.get("status") == "complete"
                and finished.get("plan_sha256") == q1.digest(plan)
                and finished.get("receipts_sha256") == q1.sha_file(output)
                and type(finished.get("failed_runs")) is int
                and finished["failed_runs"] == 0,
                "complete original ledger/receipt binding required",
            )
            source_check(
                paths, plan.get("metadata_source_commit", plan["source_commit"])
            )
            q1.require(
                all(
                    Path(getattr(args, key)).resolve() == path
                    and q1.sha_file(path) == frozen[key]
                    for key, path in files.items()
                ),
                "inputs changed before analysis",
            )
            for path in (paths.driver, paths.prereg, paths.fifo, Path(__file__)):
                q1.require(
                    q1.sha_file(path) == frozen[str(path)],
                    "source changed before analysis",
                )
            result = folder / "analysis.json"
            command = [
                sys.executable,
                str(paths.driver),
                "analyze",
                *common,
                "--receipts",
                str(output),
                "--out",
                str(result),
            ]
            code = run_owned(
                command,
                folder / "analyze.log",
                environment,
                deadline - time.monotonic(),
                handle.fileno(),
                load,
                grace,
            )
            q1.require(code == 0, "Q1 analysis failed; retain original evidence")
            analysis = q1.load(result)
            q1.require(
                analysis.get("schema") == "jope.q1-analysis/v1"
                and analysis.get("plan_sha256") == q1.digest(plan)
                and analysis.get("decision") in ("PASS", "FAIL"),
                "bound Q1 analysis required",
            )
            receipt.update(
                status="completed",
                q1_decision=analysis["decision"],
                analysis_sha256=q1.sha_file(result),
                default_changed=False,
            )
    except BaseException as error:
        receipt.update(
            last_phase=receipt["status"],
            status="incomplete_or_not_started",
            error=type(error).__name__ + ": " + str(error),
        )
        raise
    finally:
        receipt["total_wall_seconds"] = time.monotonic() - started
        write_new(folder / "launch-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--owner", default="sol_v1_obj02")
    dry = modes.add_parser("dry-run")
    dry.add_argument("--inputs-manifest", type=Path, default=Paths().input_manifest)
    dry.add_argument("--out", type=Path, required=True)
    run = modes.add_parser("launch")
    for key in ("plan", "backend", "signature", "allowed-signers", "ledger", "run-dir"):
        run.add_argument("--" + key, type=Path, required=True)
    for key in ("wait-seconds", "wall-seconds"):
        run.add_argument("--" + key, type=float, required=True)
    run.add_argument("--authorized-launch", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            result = prepare(Paths(), args.owner)
            write_new(args.out, result)
        elif args.mode == "dry-run":
            result = dry_run(args.inputs_manifest, args.out)
        else:
            result = launch(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        subprocess.SubprocessError,
    ) as error:
        print("Q1 FIFO refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
