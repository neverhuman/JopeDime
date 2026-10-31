"""JRB-01 CPU planner and bounded dispatcher for an explicitly pinned learner.

The learner contract is documented in README.md. No flywheel runner is imported.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    SCHEMA,
    admit_result,
    batches,
    file_sha,
    make_plan,
    safe_path,
    verify_catalog,
    verify_plan,
)

REAL_LOCK = Path.home() / "jope_plan/qualification.lock"
CAMPAIGNS = Path.home() / "jope-data/jrb-01/campaigns"


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextlib.contextmanager
def qualification_lock(path=REAL_LOCK):
    # Never unlink, replace, or acquire a surrogate GPU lock.
    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock.fileno()


def check_headroom():
    cpus = len(os.sched_getaffinity(0))
    mem = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if os.getloadavg()[0] > 0.85 * cpus or mem < 32 * 1024**2:
        raise RuntimeError(
            "CPU load/memory headroom unavailable; retry without preemption"
        )


def output_location(output):
    output = Path(output).resolve()
    if output.is_relative_to(Path("/tmp")) or any(
        (parent / ".git").exists() for parent in (output, *output.parents)
    ):
        raise ValueError("campaign output belongs outside working trees and /tmp")
    return output


def reserve_campaign(plan, output, ledger_root=CAMPAIGNS):
    """A different output directory cannot silently restart the same budget."""
    ledger = Path(ledger_root) / (plan["sha256"] + ".json")
    started = time.time()
    try:
        write_new(
            ledger,
            dict(
                plan_sha256=plan["sha256"],
                output=str(output),
                started_unix=started,
                deadline_unix=started + plan["config"]["wall_seconds"],
                wall_seconds=plan["config"]["wall_seconds"],
            ),
        )
    except FileExistsError as exc:
        raise ValueError(
            "campaign already reserved; no implicit retry or budget reset"
        ) from exc
    with ledger.open("rb") as stream:
        os.fsync(stream.fileno())
    directory = os.open(ledger.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return started + plan["config"]["wall_seconds"]


def request(plan, job, phase, previous=None):
    config = plan["config"]
    rows = job["train"] if phase == "train" else job["evaluation"]
    result = dict(
        schema=SCHEMA,
        phase=phase,
        plan_sha256=plan["sha256"],
        job_id=job["id"],
        arm=job["arm"],
        seed=job["seed"],
        fold=job["fold"],
        contexts=rows,
        config=config,
        learner_sha256=plan["learner_sha256"],
        initial_checkpoint_sha256=plan["initial_checkpoint_sha256"],
        dope_sha256=plan["dope_sha256"],
        evaluation_protocol=plan["evaluation_protocol"],
    )
    if phase == "train":
        result["batches"] = list(batches(job, config))
    else:
        result["training_receipt"] = previous
    return result


def worker_call(argv, env, lock_fd, log, timeout, output, limit):
    """Bound only this worker's process group; never signal another lane's job."""
    with subprocess.Popen(
        argv,
        cwd=output,
        env=env,
        pass_fds=(lock_fd,),
        start_new_session=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    ) as process:
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared campaign deadline exhausted")
                if (
                    sum(
                        path.stat().st_size
                        for path in output.rglob("*")
                        if path.is_file()
                    )
                    > limit
                ):
                    raise RuntimeError("shared campaign output budget exhausted")
                time.sleep(0.25)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, argv)
            if time.monotonic() > deadline:
                raise TimeoutError("shared campaign deadline exhausted")
        except BaseException:
            # Killing our own process group also releases any inherited lock FD.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > limit:
        raise RuntimeError("shared campaign output budget exhausted")


def execute(plan, packet, worker, checkpoint, output, store_module, store):
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    verify_plan(plan)
    output = output_location(output)
    if output.exists():
        raise ValueError(
            "new campaign directory required; no implicit restart/budget reset"
        )
    if (
        file_sha(worker) != plan["learner_sha256"]
        or file_sha(checkpoint) != plan["initial_checkpoint_sha256"]
    ):
        raise ValueError("learner/initial checkpoint drift")
    # Validate every permitted file before GPU acquisition; never load row content here.
    for row in plan["contexts"]:
        if file_sha(safe_path(packet, row["payload"])) != row["payload_sha256"]:
            raise ValueError("packet payload drift")
    check_headroom()
    with qualification_lock() as lock_fd:
        start = time.monotonic()
        absolute_deadline = reserve_campaign(plan, output)
        output.mkdir(parents=True)
        write_new(output / "plan.json", plan)
        for job in plan["jobs"]:
            train_receipt = None
            for phase in ("train", "evaluate"):
                check_headroom()
                remaining = min(
                    plan["config"]["wall_seconds"] - (time.monotonic() - start),
                    absolute_deadline - time.time(),
                )
                if remaining <= 0:
                    raise TimeoutError("shared campaign deadline exhausted")
                if (
                    sum(
                        path.stat().st_size
                        for path in output.rglob("*")
                        if path.is_file()
                    )
                    >= plan["config"]["output_bytes"]
                ):
                    raise RuntimeError("shared campaign output budget exhausted")
                prefix = output / f"{job['id']}-{phase}"
                req, response = (
                    prefix.with_suffix(".request.json"),
                    prefix.with_suffix(".response.json"),
                )
                write_new(req, request(plan, job, phase, train_receipt))
                env = dict(
                    os.environ,
                    JOPE_QUALIFICATION_LOCK_FD=str(lock_fd),
                    OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="1",
                )
                # An argv vector, never shell interpolation; one learner process at a time.
                with prefix.with_suffix(".log").open("xb") as log:
                    worker_call(
                        [
                            sys.executable,
                            str(Path(worker).resolve()),
                            "--request",
                            str(req),
                            "--response",
                            str(response),
                            "--packet",
                            str(Path(packet).resolve()),
                            "--initial-checkpoint",
                            str(Path(checkpoint).resolve()),
                        ],
                        env,
                        lock_fd,
                        log,
                        remaining,
                        output,
                        plan["config"]["output_bytes"],
                    )
                result = json.loads(response.read_text())
                if phase == "train":
                    train_receipt = result
                else:
                    admit_result(plan, job, result)
                    append_score_store(plan, job_rows(job, result), store_module, store)
        write_new(
            output / "execution.json",
            dict(wall_seconds=time.monotonic() - start, jobs=len(plan["jobs"])),
        )


def collect(plan, directory):
    """Admit the entire expected inventory before returning any scored observations."""
    verify_plan(plan)
    directory = Path(directory)
    expected = {f"{job['id']}-evaluate.response.json" for job in plan["jobs"]}
    actual = {path.name for path in directory.glob("*-evaluate.response.json")}
    if actual != expected:
        raise ValueError("missing/extra job results; no scientific verdict")
    rows = []
    initial_by_seed = {}
    for job in plan["jobs"]:
        receipt = admit_result(
            plan,
            job,
            json.loads((directory / f"{job['id']}-evaluate.response.json").read_text()),
        )
        previous = initial_by_seed.setdefault(job["seed"], receipt["parameters_before"])
        if previous != receipt["parameters_before"]:
            raise ValueError("paired arms did not start from identical parameters")
        rows.extend(job_rows(job, receipt))
    return rows


def job_rows(job, receipt):
    contexts = {row["id"]: row for row in job["evaluation"]}
    return [
        dict(
            score,
            arm=job["arm"],
            seed=job["seed"],
            fold=job["fold"],
            job_id=job["id"],
            task=contexts[score["id"]]["task"],
            family=contexts[score["id"]]["family"],
            dataset=contexts[score["id"]]["dataset"],
            role="cv"
            if contexts[score["id"]]["role"] == "train"
            else contexts[score["id"]]["role"],
        )
        for score in receipt["scores"]
    ]


def append_score_store(plan, rows, module, store):
    """Use JRB-02's owning API; never reimplement or rewrite its historical store."""
    spec = importlib.util.spec_from_file_location("jrb02_store", module)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    now = datetime.now(timezone.utc).isoformat()
    observations = []
    for row in rows:
        for metric, value, higher in (
            (row["metric"], row["value"], row["task"] == "binary"),
            (
                "log_rmse_ratio" if row["task"] == "regression" else "log_loss",
                row["curve"],
                False,
            ),
        ):
            observations.append(
                dict(
                    schema="jope.score-history/v1",
                    ts=now,
                    model=f"jrb01/{row['arm']}/seed{row['seed']}/fold{row['fold']}",
                    model_sha=row["model_sha256"],
                    task=row["task"],
                    dataset=row["dataset"],
                    family=row["family"],
                    split=f"jrb01-{row['role']}",
                    metric=metric,
                    value=value,
                    higher_is_better=higher,
                    source=f"jrb01:{plan['sha256']}:{row['job_id']}:{row['id']}:round{row['round']}",
                    round=row["round"],
                    host=socket.gethostname(),
                )
            )
    return implementation.append_rows(observations, Path(store))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--catalog", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("--packet", required=True, type=Path)
    run.add_argument("--worker", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--score-store-module", required=True, type=Path)
    run.add_argument("--score-store", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        manifest = json.loads(args.manifest.read_text())
        verify_catalog(manifest, args.catalog)
        write_new(args.out, make_plan(manifest))
    else:
        execute(
            json.loads(args.plan.read_text()),
            args.packet,
            args.worker,
            args.checkpoint,
            args.out,
            args.score_store_module,
            args.score_store,
        )


if __name__ == "__main__":
    main()
