"""Owner-authorized resume of missing JRB phases; frozen scientific code unchanged.

This controller replaces only the abandoned dispatch loop. It calls the frozen
maintained preflight, request, worker_call, admit_result and score-store APIs.
No monkeypatch, plan rekey, completed-unit replay, new campaign or budget reset.
The qualified learner's in-unit guards remain unchanged; only a proven configure-
phase headroom refusal (no execution/checkpoint/response) may retry after 120s.
"""

from __future__ import annotations

import datetime
import contextlib
import json
import math
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path("/home/ubuntu/JopeDime/.agent/worktrees/jrb-01-learner")
OUT = Path("/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t0415-lock")
OLD = OUT.parent / "sol-jrb01-t0240"
PLAN = OLD / "FROZEN-PLAN.json"
PACKET = Path("/home/ubuntu/jope-data/jrb-01/admitted-study-packet")
STUDY = Path("/home/ubuntu/jope-data/jrb-01/sol-jrb01-study1")
WORKER = ROOT / "research/joint_regbin/learner.py"
CHECKPOINT = Path("/home/ubuntu/jope-data/H141-t1815/r882_init.pt")
RULE = Path("/home/ubuntu/nudges/lanes_t0430_rulings.md")
RULE_SHA = "9f37306a25cafb0d213625d999c1d255bd8139425794ab2e5e82db15583da038"
PLAN_SHA = "96310d6916d79c411a0fc4c53d2824ac678140f95b22520d6fd9fc3b0936abb3"
WORKER_SHA = "558780bb5d91318864991ca398b9c63c4b31a12d1c8f3dd04ae26db7655b9918"
STORE_MODULE = Path("/home/ubuntu/jope_plan/scores/score_store.py")
STORE = Path("/home/ubuntu/jope_plan/scores/history.jsonl")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from research.joint_regbin import driver, learner, protocol  # noqa: E402
from research.flywheel_selfplay import qualification_lock_fifo as fifo  # noqa: E402

QUEUE = Path("/home/ubuntu/jope_plan/qualification.queue")


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(path.read_text())


def event(kind, **fields):
    record = dict(utc=now(), event=kind, **fields)
    raw = (json.dumps(record, allow_nan=False) + "\n").encode()
    fd = os.open(
        OUT / "resume-ledger.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
    )
    try:
        if os.write(fd, raw) != len(raw):
            raise OSError("short ledger append")
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(record), flush=True)


class LoadGate:
    """Five >85 observations spaced >=60s pause NEW dispatch; <70 resumes."""

    def __init__(self):
        self.last = None
        self.high = 0
        self.paused = False
        self.mutex = threading.Lock()

    def sample(self, value, tick):
        with self.mutex:
            if not math.isfinite(value) or not math.isfinite(tick):
                raise ValueError("finite load and monotonic sample time required")
            if self.last is not None and tick - self.last < 60:
                return None
            if self.last is not None and tick - self.last >= 120:
                self.high = 0
            self.last = tick
            self.high = self.high + 1 if value > 85 else 0
            if self.paused:
                if value < 70:
                    self.paused = False
            elif self.high >= 5:
                self.paused = True
            return dict(
                load1=value, consecutive_high=self.high, dispatch_paused=self.paused
            )

    def observe(self):
        sample = self.sample(os.getloadavg()[0], time.monotonic())
        if sample is not None:
            event("LOAD_SAMPLE", **sample)

    def blocked(self):
        with self.mutex:
            return self.paused


def proven_preexecution_refusal(log, response):
    """Never retry an unknown or started scientific computation."""
    if (
        response.exists()
        or (response.parent / (response.stem + "-checkpoints")).exists()
    ):
        return False
    text = log.read_text()
    return (
        "in configure\n" in text
        and "resource_check(device)" in text
        and (
            "CPU load exceeds 85% host budget" in text
            or "GPU memory exceeds 90% budget" in text
        )
        and not any(s in text for s in ("in execute\n", "in train\n", "in evaluate\n"))
    )


def usage():
    return sum(
        p.stat().st_size
        for root in (STUDY, OUT)
        for p in root.rglob("*")
        if p.is_file()
    )


def oldest_ticket():
    live = []
    for path in QUEUE.iterdir():
        if path.name.endswith(".done") or not path.is_file() or path.is_symlink():
            continue
        parsed = fifo.parse_ticket_name(path.name)
        if parsed and fifo.pid_is_alive(parsed[1]):
            live.append((*parsed, path))
    return min(live)[3] if live else None


@contextlib.contextmanager
def ordered_lock(deadline):
    ticket = fifo.enqueue_ticket(QUEUE, wave="JRB01T0430_96310d69")
    try:
        event(
            "FIFO_ENQUEUED",
            ticket=str(ticket),
            plan_sha256=PLAN_SHA,
            authority_sha256=RULE_SHA,
        )
        while time.time() < deadline:
            acquired = False
            if oldest_ticket() == ticket:
                try:
                    with driver.qualification_lock() as descriptor:
                        if oldest_ticket() != ticket:
                            continue
                        acquired = True
                        yield descriptor, ticket
                        return
                except BlockingIOError:
                    if acquired:
                        raise
            time.sleep(1)
        raise TimeoutError("original campaign deadline exhausted while queued")
    finally:
        # A killed PID is also treated as done by oldest_ticket/maintained PID
        # liveness. Never remove another owner's ticket. SIGTERM runs finally;
        # SIGKILL cannot execute Python cleanup and uses the dead-PID rule.
        done = ticket.with_name(ticket.name + ".done")
        ticket.rename(done)
        event("FIFO_DONE", ticket=str(done), plan_sha256=PLAN_SHA)


def main():
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    learner.torch.set_num_threads(1)
    assert protocol.file_sha(RULE) == RULE_SHA
    assert protocol.file_sha(WORKER) == WORKER_SHA
    learner.verify_sources()
    plan = read(PLAN)
    assert plan["sha256"] == PLAN_SHA and read(STUDY / "plan.json") == plan
    campaign_path = driver.CAMPAIGNS / (PLAN_SHA + ".json")
    campaign = read(campaign_path)
    assert campaign["plan_sha256"] == PLAN_SHA and campaign["output"] == str(STUDY)
    assert campaign["wall_seconds"] == plan["config"]["wall_seconds"] == 21600
    assert campaign["deadline_unix"] == campaign["started_unix"] + 21600
    deadline = campaign["deadline_unix"]
    monotonic_deadline = time.monotonic() + max(0, deadline - time.time())
    preflight = driver.preflight(plan, PACKET, WORKER, CHECKPOINT)
    initial_files = {
        str(p.relative_to(STUDY)): protocol.file_sha(p)
        for p in STUDY.rglob("*")
        if p.is_file()
    }
    finished = []
    for job in plan["jobs"]:
        response = STUDY / f"{job['id']}-evaluate.response.json"
        if response.exists():
            driver.admit_result(plan, job, read(response))
            train = read(STUDY / f"{job['id']}-train.response.json")
            evaluated = read(response)
            assert evaluated == dict(train, scores=evaluated["scores"])
            assert read(STUDY / f"{job['id']}-train.request.json") == driver.request(
                plan, job, "train"
            )
            assert read(STUDY / f"{job['id']}-evaluate.request.json") == driver.request(
                plan, job, "evaluate", train
            )
            for trace in train["round_receipts"]:
                assert (
                    protocol.file_sha(Path(trace["checkpoint_path"]))
                    == trace["checkpoint_sha256"]
                )
            finished.append(job["id"])
    assert finished == [plan["jobs"][0]["id"]], (
        "unexpected initial inventory; explicit audit required"
    )
    write(
        OUT / "FROZEN-PRIOR-INVENTORY.json",
        dict(
            files=initial_files,
            completed_jobs=finished,
            original_campaign=campaign,
            campaign_sha256=protocol.file_sha(campaign_path),
            frozen_plan_file_sha256=protocol.file_sha(PLAN),
            admission_sha256=protocol.file_sha(OUT.parent / "ADMISSION-PASS.json"),
            authority_sha256=RULE_SHA,
        ),
    )
    write(
        OUT / "CONTROLLER-CLAIM.json",
        dict(
            pid=os.getpid(),
            utc=now(),
            source_sha256=protocol.file_sha(Path(__file__)),
            source_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            authority=str(RULE),
            authority_sha256=RULE_SHA,
            plan_sha256=PLAN_SHA,
            preflight=preflight,
            original_deadline_unix=deadline,
            original_started_unix=campaign["started_unix"],
            budget_reset=False,
        ),
    )
    gate = LoadGate()
    stop = threading.Event()

    def check_budget():
        if time.time() >= deadline or time.monotonic() >= monotonic_deadline:
            raise TimeoutError("original campaign deadline exhausted; no reset")
        if shutil.disk_usage("/").free < 200_000_000_000:
            raise RuntimeError("xbabe2 free disk below 200 GB")
        if usage() >= plan["config"]["output_bytes"]:
            raise RuntimeError("original cumulative output cap exhausted")

    def monitor():
        while not stop.is_set():
            free = shutil.disk_usage("/").free
            if free < 200_000_000_000:
                event("DISK_STOP", free_bytes=free)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            gate.observe()
            stop.wait(1)

    def dispatch_ready():
        while True:
            check_budget()
            gate.observe()
            memory = next(
                int(s.split()[1])
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
                timeout=15,
            ).strip()
            if not gate.blocked() and memory >= 32 * 1024**2 and not gpu:
                return
            time.sleep(1)

    # The maintained FIFO ticket is the sole cross-owner handoff authority.
    with ordered_lock(deadline) as (lock_fd, ticket):
        event(
            "LOCK_ACQUIRED_JRB_FIRST",
            pid=os.getpid(),
            authority_sha256=RULE_SHA,
            original_deadline_unix=deadline,
            completed_frozen_jobs=finished,
            fifo_ticket=str(ticket),
        )
        write(
            OUT / "LOCK-ACQUIRED.json",
            dict(
                pid=os.getpid(),
                utc=now(),
                authority_sha256=RULE_SHA,
                deadline_unix=deadline,
            ),
        )
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            for job in plan["jobs"]:
                if job["id"] in finished:
                    event("FROZEN_JOB_SKIPPED", job_id=job["id"])
                    continue
                train_receipt = None
                for phase in ("train", "evaluate"):
                    dispatch_ready()
                    prefix = STUDY / f"{job['id']}-{phase}"
                    req = prefix.with_suffix(".request.json")
                    response = prefix.with_suffix(".response.json")
                    expected = driver.request(plan, job, phase, train_receipt)
                    assert not response.exists(), "never overwrite/replay a response"
                    if req.exists():
                        assert read(req) == expected
                    else:
                        driver.write_new(req, expected)
                    attempt = 0
                    while True:
                        dispatch_ready()
                        attempt += 1
                        log = (
                            OUT
                            / "unit-logs"
                            / f"{job['id']}-{phase}-attempt{attempt:03d}.log"
                        )
                        log.parent.mkdir(exist_ok=True)
                        env = dict(
                            os.environ,
                            JOPE_QUALIFICATION_LOCK_FD=str(lock_fd),
                            JOPE_JRB_DEVICE="cuda:0",
                            CUDA_VISIBLE_DEVICES="0",
                            CUBLAS_WORKSPACE_CONFIG=":4096:8",
                            OMP_NUM_THREADS="4",
                            MKL_NUM_THREADS="4",
                            OPENBLAS_NUM_THREADS="1",
                            PYTHONPATH=str(ROOT / "python/src") + ":" + str(ROOT),
                            CARGO_BUILD_JOBS="8",
                        )
                        for key in (
                            "GH_TOKEN",
                            "GITHUB_TOKEN",
                            "AWS_ACCESS_KEY_ID",
                            "AWS_SECRET_ACCESS_KEY",
                            "AWS_SESSION_TOKEN",
                        ):
                            env.pop(key, None)
                        argv = [
                            sys.executable,
                            str(WORKER),
                            "--request",
                            str(req),
                            "--response",
                            str(response),
                            "--packet",
                            str(PACKET),
                            "--initial-checkpoint",
                            str(CHECKPOINT),
                        ]
                        event(
                            "UNIT_STARTED",
                            job_id=job["id"],
                            phase=phase,
                            attempt=attempt,
                            request_sha256=protocol.file_sha(req),
                            learner_sha256=WORKER_SHA,
                        )
                        try:
                            check_budget()
                            with log.open("xb") as stream:
                                driver.worker_call(
                                    argv,
                                    env,
                                    lock_fd,
                                    stream,
                                    min(
                                        deadline - time.time(),
                                        monotonic_deadline - time.monotonic(),
                                    ),
                                    STUDY,
                                    plan["config"]["output_bytes"]
                                    - sum(
                                        p.stat().st_size
                                        for p in OUT.rglob("*")
                                        if p.is_file()
                                    ),
                                )
                        except subprocess.CalledProcessError:
                            if not proven_preexecution_refusal(log, response):
                                event(
                                    "UNIT_FAILURE_NOT_RETRIED",
                                    job_id=job["id"],
                                    phase=phase,
                                    attempt=attempt,
                                    log_sha256=protocol.file_sha(log),
                                )
                                raise
                            event(
                                "PRE_EXECUTION_HEADROOM_REFUSAL",
                                job_id=job["id"],
                                phase=phase,
                                attempt=attempt,
                                retry_seconds=120,
                                actual_optimizer_updates=0,
                                log_sha256=protocol.file_sha(log),
                            )
                            until = time.monotonic() + 120
                            while time.monotonic() < until:
                                check_budget()
                                time.sleep(min(1, until - time.monotonic()))
                            continue
                        break
                    check_budget()
                    result = read(response)
                    if phase == "train":
                        assert result["actual_budget"] == protocol.expected_counts(
                            plan["config"]
                        )
                        assert (
                            result["plan_sha256"] == PLAN_SHA
                            and result["job_id"] == job["id"]
                        )
                        train_receipt = result
                    else:
                        driver.admit_result(plan, job, result)
                        store_result = driver.append_score_store(
                            plan, driver.job_rows(job, result), STORE_MODULE, STORE
                        )
                        event(
                            "SCORE_STORE_APPENDED",
                            job_id=job["id"],
                            result=store_result,
                        )
                    event(
                        "UNIT_FINISHED",
                        job_id=job["id"],
                        phase=phase,
                        attempt=attempt,
                        response_sha256=protocol.file_sha(response),
                        actual_optimizer_steps=result["actual_budget"][
                            "optimizer_steps"
                        ]
                        if phase == "train"
                        else 0,
                    )
                finished.append(job["id"])
                event(
                    "JOB_FINISHED",
                    job_id=job["id"],
                    completed_jobs=len(finished),
                    expected_jobs=54,
                )
            driver.collect(plan, STUDY)  # full inventory/paired-source validation only
            for relative, sha in initial_files.items():
                assert protocol.file_sha(STUDY / relative) == sha, (
                    "prior artifact changed"
                )
            assert (
                protocol.file_sha(campaign_path)
                == read(OUT / "FROZEN-PRIOR-INVENTORY.json")["campaign_sha256"]
            )
            check_budget()
            driver.write_new(
                STUDY / "execution.json",
                dict(
                    wall_seconds=time.time() - campaign["started_unix"],
                    jobs=len(plan["jobs"]),
                    resume_authority_sha256=RULE_SHA,
                    budget_reset=False,
                ),
            )
            event(
                "ALL_JOBS_COMPLETE",
                jobs=54,
                optimizer_steps=5400,
                prior_artifacts_unchanged=True,
            )
        finally:
            stop.set()
            thread.join(timeout=2)
    argv = [
        sys.executable,
        "-B",
        "-m",
        "research.joint_regbin.report",
        str(PLAN),
        str(STUDY),
        "--out",
        "/home/ubuntu/jope-data/jrb-01/sol-jrb01-report1",
        "--score-store-module",
        str(STORE_MODULE),
        "--score-store",
        str(STORE),
    ]
    with (OUT / "report.log").open("xb") as log:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(
                1,
                min(deadline - time.time(), monotonic_deadline - time.monotonic()),
            ),
        )
    write(
        OUT / "REPORT-EXIT.json",
        dict(utc=now(), exit_code=result.returncode, argv=argv),
    )


if __name__ == "__main__":

    def interrupted(_signum, _frame):
        raise InterruptedError("controller terminated; preserve prior receipts")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as error:
        event(
            "CONTROLLER_STOPPED",
            error=type(error).__name__ + ":" + str(error),
            automatic_retry=False,
        )
        raise
