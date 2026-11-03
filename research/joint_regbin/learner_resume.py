"""Owner-authorized resume of missing JRB phases; frozen scientific code unchanged.

This controller replaces only the abandoned dispatch loop. It calls the frozen
maintained preflight, request, worker_call, admit_result and score-store APIs.
No monkeypatch, plan rekey, completed-unit replay, new campaign or budget reset.
The explicit, CPU-qualified resource-policy wrapper removes the base learner's
in-unit CPU-spike abort while preserving GPU memory safety. Only a configure-
phase headroom refusal (no execution/checkpoint/response) may retry after 120s.
"""

from __future__ import annotations

import datetime
import contextlib
import argparse
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
OUT = Path("/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1458")
PRIOR_OUT = OUT.parent / "sol-jrb01-t0415-lock"
OLD = OUT.parent / "sol-jrb01-t0240"
PLAN = OLD / "FROZEN-PLAN.json"
PACKET = Path("/home/ubuntu/jope-data/jrb-01/admitted-study-packet")
STUDY = Path("/home/ubuntu/jope-data/jrb-01/sol-jrb01-study1")
ATTEMPTS = STUDY / "recovery-t1458"
QUALIFIED = STUDY / "qualified-t1458"
BASELINE = OUT / "RECOVERY-BASELINE.json"
BASELINE_SHA = "2f5fd89053b604e9b3edeb3d06ac1c2f2ee0e6de8ec884b5dfffed22444566ce"
QUALIFIED_WORKER_SHA = (
    "cb6918ca5c0869541acf37375cb1d3ebd47f72a5875b08dbe7f3b1c9caf993f8"
)
WORKER = ROOT / "research/joint_regbin/learner.py"
CHECKPOINT = Path("/home/ubuntu/jope-data/H141-t1815/r882_init.pt")
RULE = Path("/home/ubuntu/nudges/lanes_t0430_rulings.md")
RULE_SHA = "9f37306a25cafb0d213625d999c1d255bd8139425794ab2e5e82db15583da038"
PLAN_SHA = "96310d6916d79c411a0fc4c53d2824ac678140f95b22520d6fd9fc3b0936abb3"
WORKER_SHA = "558780bb5d91318864991ca398b9c63c4b31a12d1c8f3dd04ae26db7655b9918"
RESOURCE_WORKER = ROOT / "research/joint_regbin/learner_resource.py"
RESOURCE_SHA = "3c6534979204b98450dac965935e9c019ad4f00f6a25958b0c92475b5298ca5c"
RESOURCE_QUALIFICATION = Path(
    "/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1450/RESOURCE-QUALIFICATION.json"
)
RESOURCE_QUALIFICATION_SHA = (
    "796f2a80a537ae06b61f52f41c3770743775c43954af56e18fdec1708571df32"
)
RESERVATION_SHA = "f59cc7e6e8ec2639c723a21db1f0225a0b304c099d63cf332190dcbe7b870242"
OWNER_EXTENSION_SHA = "48d45b3a0188d07fac037a2f30cfd216d9f8693006fc137654cea070ac7acd31"
STORE_MODULE = Path("/home/ubuntu/jope_plan/scores/score_store.py")
STORE = Path("/home/ubuntu/jope_plan/scores/history.jsonl")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from research.joint_regbin import (  # noqa: E402
    driver,
    learner_resource,
    learner_source,
    protocol,
)
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


def execution_charge(segments):
    """Charge every earlier attempt, including rejected work and drain time."""
    total = 0.0
    previous_end = None
    for start, end in segments:
        start = datetime.datetime.fromisoformat(start)
        end = datetime.datetime.fromisoformat(end)
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or end <= start
            or (previous_end is not None and start < previous_end)
        ):
            raise ValueError("invalid or overlapping prior execution interval")
        total += (end - start).total_seconds()
        previous_end = end
    return total


def verify_preserved(directory, files):
    """An audit cannot silently omit, mutate or substitute a prior artifact."""
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    if actual != set(files):
        raise ValueError("prior artifact inventory differs")
    for name, entry in files.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size != entry["bytes"]
            or protocol.file_sha(path) != entry["sha256"]
        ):
            raise ValueError("preserved artifact differs: " + name)


def require_worker_provenance(receipt, expected):
    updates = [u for trace in receipt["round_receipts"] for u in trace["updates"]]
    if not updates or any(u["worker_source_sha256"] != expected for u in updates):
        raise ValueError("actual worker source provenance differs")


def recovery_baseline(plan):
    if protocol.file_sha(BASELINE) != BASELINE_SHA:
        raise ValueError("recovery baseline hash differs")
    baseline = read(BASELINE)
    pinned = {}
    for name, pin in baseline["pins"].items():
        path = Path(pin["path"])
        if path.is_symlink() or protocol.file_sha(path) != pin["sha256"]:
            raise ValueError("recovery evidence differs: " + name)
        pinned[name] = read(path)
    review = pinned["source_review"]
    if (
        review["source_fix_verdict"] != "PASS"
        or review["qualification_sha256"] != RESOURCE_QUALIFICATION_SHA
        or review["actual_worker_fingerprint"] != QUALIFIED_WORKER_SHA
        or review["loader_sha256"] != protocol.file_sha(learner_source.__file__)
    ):
        raise ValueError("source review does not bind current recovery closure")
    files = pinned["preserved_inventory"]["files"]
    verify_preserved(STUDY, files)
    rejection = pinned["rejection"]
    rejected = set(baseline["rejected_job_ids"])
    finished = set(baseline["qualified_job_ids"])
    if (
        rejection["decision"] != "REJECT_SOURCE_CLOSURE_PRESERVE_AND_CHARGE"
        or {j["job_id"] for j in rejection["jobs"]} != rejected
        or rejected & finished
        or len(finished) != 1
        or len(rejected) != 2
    ):
        raise ValueError("rejected/qualified prior job inventory differs")
    spent = 0
    observed = set()
    for job in plan["jobs"]:
        response = STUDY / f"{job['id']}-evaluate.response.json"
        if not response.exists():
            continue
        job_id = job["id"]
        observed.add(job_id)
        train = read(STUDY / f"{job_id}-train.response.json")
        evaluated = read(response)
        if evaluated != dict(train, scores=evaluated["scores"]):
            raise ValueError("prior train/evaluation receipt pair differs")
        if train["actual_budget"] != protocol.expected_counts(plan["config"]):
            raise ValueError("prior physical update budget differs")
        spent += train["actual_budget"]["optimizer_steps"]
        for phase, previous in (("train", None), ("evaluate", train)):
            if read(STUDY / f"{job_id}-{phase}.request.json") != driver.request(
                plan, job, phase, previous
            ):
                raise ValueError("prior request differs from frozen job")
        if job_id in finished:
            require_worker_provenance(train, QUALIFIED_WORKER_SHA)
            driver.admit_result(plan, job, evaluated)
        elif job_id in rejected:
            require_worker_provenance(train, baseline["rejected_worker_source_sha256"])
        else:
            raise ValueError("unclassified prior job")
    if observed != finished | rejected or spent != 300:
        raise ValueError("prior physical charges or job inventory differ")
    if rejection["study_optimizer_updates_spent"] != spent:
        raise ValueError("rejection changed physical update charges")
    charge = execution_charge(
        [
            (pinned["first_start"]["utc"], pinned["first_end"]["utc"]),
            (pinned["second_start"]["utc"], pinned["second_end"]["utc"]),
        ]
    )
    return dict(
        files=files,
        qualified_job_ids=sorted(finished),
        rejected_job_ids=sorted(rejected),
        physical_updates_spent=spent,
        prior_execution_seconds=charge,
        remaining_qualified_jobs=len(plan["jobs"]) - len(finished),
        planned_qualified_updates=len(plan["jobs"]) * 100,
        planned_physical_updates=spent + (len(plan["jobs"]) - len(finished)) * 100,
    )


def verify_readmission(path, expected_sha):
    if path.is_symlink() or protocol.file_sha(path) != expected_sha:
        raise ValueError("TEL readmission receipt hash differs")
    receipt = read(path)
    binding = receipt.get("source_binding", {})
    accounting = receipt.get("recovery_accounting", {})
    if (
        receipt.get("decision") != "ADMISSION_PASS"
        or binding.get("controller_sha256") != protocol.file_sha(__file__)
        or binding.get("resource_qualification_sha256") != RESOURCE_QUALIFICATION_SHA
        or binding.get("source_loader_sha256")
        != protocol.file_sha(learner_source.__file__)
        or accounting.get("baseline_sha256") != BASELINE_SHA
        or accounting.get("physical_updates_spent") != 300
        or accounting.get("planned_physical_updates") != 5600
        or accounting.get("planned_qualified_updates") != 5400
    ):
        raise ValueError("TEL readmission lacks current resumer/accounting binding")
    return receipt


def publish_qualified(job_id, source):
    """Create an immutable collection view; rejected originals remain untouched."""
    receipt = read(source)
    if receipt.get("job_id") != job_id or receipt.get("plan_sha256") != PLAN_SHA:
        raise ValueError("qualified collection job/plan differs")
    require_worker_provenance(receipt, QUALIFIED_WORKER_SHA)
    target = QUALIFIED / f"{job_id}-evaluate.response.json"
    raw = source.read_bytes()
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if protocol.file_sha(target) != protocol.file_sha(source):
        raise ValueError("qualified collection copy differs")


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
        for root in (STUDY, OUT, PRIOR_OUT)
        for p in root.rglob("*")
        if p.is_file()
    )


def owner_deadline_extension():
    path = os.environ.get("JOPE_JRB_DEADLINE_EXTENSION")
    sha = os.environ.get("JOPE_JRB_DEADLINE_EXTENSION_SHA256")
    if path is None and sha is None:
        return None
    if (
        not path
        or sha != OWNER_EXTENSION_SHA
        or Path(path).is_symlink()
        or protocol.file_sha(Path(path)) != OWNER_EXTENSION_SHA
    ):
        raise ValueError("owner deadline extension hash differs")
    approval = read(Path(path))
    if (
        approval.get("kind") != "OWNER_DEADLINE_EXTENSION_APPROVAL"
        or approval.get("approval_ref") != "t188s1838"
        or approval.get("plan_sha256") != PLAN_SHA
        or approval.get("reservation_sha256") != RESERVATION_SHA
        or approval.get("computation", {}).get("window_seconds") != 22658
    ):
        raise ValueError("owner deadline extension scope differs")
    return approval


def amended_deadline(campaign, prior_execution_seconds, resumed_at):
    """T0430 permits bounded infrastructure extension; prior work stays charged."""
    original = campaign["deadline_unix"]
    if (
        campaign["plan_sha256"] != PLAN_SHA
        or campaign["wall_seconds"] != 21600
        or original != campaign["started_unix"] + 21600
        or not 0 < prior_execution_seconds < 21600
        or resumed_at < campaign["started_unix"] + prior_execution_seconds
    ):
        raise ValueError("original reservation/execution charge differs")
    extension = owner_deadline_extension()
    if extension is not None:
        latest = extension["new_deadline"]["resume_must_occur_before_utc"].split()[0]
        if resumed_at >= datetime.datetime.fromisoformat(latest).timestamp():
            raise ValueError(
                "owner deadline extension requires a fresh receipt after17:30"
            )
        return resumed_at + extension["computation"]["window_seconds"]
    return min(original + 7200, resumed_at + 21600 - prior_execution_seconds)


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
    global learner, WORKER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    resumed_at = time.time()
    run_started = time.monotonic()
    learner = learner_source.load()
    WORKER = Path(learner.__file__)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    learner.torch.set_num_threads(1)
    assert protocol.file_sha(RULE) == RULE_SHA
    assert protocol.file_sha(WORKER) == WORKER_SHA
    learner.verify_sources()
    assert protocol.file_sha(RESOURCE_WORKER) == RESOURCE_SHA
    learner_resource.qualification(RESOURCE_QUALIFICATION, RESOURCE_QUALIFICATION_SHA)
    plan = read(PLAN)
    assert plan["sha256"] == PLAN_SHA and read(STUDY / "plan.json") == plan
    campaign_path = driver.CAMPAIGNS / (PLAN_SHA + ".json")
    assert protocol.file_sha(campaign_path) == RESERVATION_SHA, (
        "original reservation changed"
    )
    campaign = read(campaign_path)
    assert campaign["plan_sha256"] == PLAN_SHA and campaign["output"] == str(STUDY)
    assert campaign["wall_seconds"] == plan["config"]["wall_seconds"] == 21600
    assert campaign["deadline_unix"] == campaign["started_unix"] + 21600
    audit = recovery_baseline(plan)
    prior_execution_seconds = audit["prior_execution_seconds"]
    deadline = amended_deadline(campaign, prior_execution_seconds, resumed_at)
    extension = owner_deadline_extension()
    extension_fields = (
        {}
        if extension is None
        else dict(
            owner_extension_sha256=OWNER_EXTENSION_SHA,
            owner_extension_receipt=os.environ["JOPE_JRB_DEADLINE_EXTENSION"],
            owner_extension_window_seconds=extension["computation"]["window_seconds"],
        )
    )
    monotonic_deadline = time.monotonic() + max(0, deadline - time.time())
    preflight = driver.preflight(plan, PACKET, WORKER, CHECKPOINT)
    if args.audit_only:
        print(
            json.dumps(
                dict(
                    status="RECOVERY_INPUTS_VERIFIED_NOT_LAUNCHED",
                    source_sha256=protocol.file_sha(__file__),
                    plan_sha256=PLAN_SHA,
                    qualification_sha256=RESOURCE_QUALIFICATION_SHA,
                    effective_deadline_unix=deadline,
                    accounting={k: v for k, v in audit.items() if k != "files"},
                    preflight=preflight,
                    **extension_fields,
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    readmission = Path(os.environ["JOPE_JRB_READMISSION"])
    readmission_sha = os.environ["JOPE_JRB_READMISSION_SHA256"]
    verify_readmission(readmission, readmission_sha)
    if ATTEMPTS.exists() or QUALIFIED.exists():
        raise ValueError(
            "recovery attempt/view already exists; explicit audit required"
        )
    initial_files = audit["files"]
    finished = list(audit["qualified_job_ids"])
    physical_updates = audit["physical_updates_spent"]
    write(
        OUT / "FROZEN-PRIOR-INVENTORY.json",
        dict(
            **audit,
            original_campaign=campaign,
            campaign_sha256=protocol.file_sha(campaign_path),
            baseline_sha256=BASELINE_SHA,
            frozen_plan_file_sha256=protocol.file_sha(PLAN),
            admission_sha256=protocol.file_sha(OUT.parent / "ADMISSION-PASS.json"),
            readmission_sha256=readmission_sha,
            authority_sha256=RULE_SHA,
            **extension_fields,
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
            qualified_source_commit=learner_source.COMMIT,
            source_loader_sha256=protocol.file_sha(learner_source.__file__),
            resource_wrapper_sha256=RESOURCE_SHA,
            resource_qualification_sha256=RESOURCE_QUALIFICATION_SHA,
            preflight=preflight,
            original_deadline_unix=campaign["deadline_unix"],
            effective_deadline_unix=deadline,
            deadline_amendment_seconds=deadline - campaign["deadline_unix"],
            maximum_deadline_extension_seconds=(
                7200 if extension is None else deadline - campaign["deadline_unix"]
            ),
            prior_execution_seconds=prior_execution_seconds,
            resumed_started_unix=resumed_at,
            physical_updates_spent=physical_updates,
            planned_physical_updates=audit["planned_physical_updates"],
            planned_qualified_updates=audit["planned_qualified_updates"],
            rejected_job_ids=audit["rejected_job_ids"],
            qualified_results=str(QUALIFIED),
            readmission_sha256=readmission_sha,
            recovery_baseline_sha256=BASELINE_SHA,
            maximum_total_execution_seconds=(
                21600
                if extension is None
                else prior_execution_seconds
                + extension["computation"]["window_seconds"]
            ),
            original_started_unix=campaign["started_unix"],
            budget_reset=False,
            **extension_fields,
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
            original_deadline_unix=campaign["deadline_unix"],
            effective_deadline_unix=deadline,
            completed_frozen_jobs=finished,
            fifo_ticket=str(ticket),
            **extension_fields,
        )
        write(
            OUT / "LOCK-ACQUIRED.json",
            dict(
                pid=os.getpid(),
                utc=now(),
                authority_sha256=RULE_SHA,
                deadline_unix=deadline,
                **extension_fields,
            ),
        )
        ATTEMPTS.mkdir()
        QUALIFIED.mkdir()
        for job_id in finished:
            publish_qualified(job_id, STUDY / f"{job_id}-evaluate.response.json")
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
                    prefix = ATTEMPTS / f"{job['id']}-{phase}"
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
                            JOPE_JRB_RESOURCE_QUALIFICATION=str(RESOURCE_QUALIFICATION),
                            JOPE_JRB_RESOURCE_QUALIFICATION_SHA256=RESOURCE_QUALIFICATION_SHA,
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
                            str(RESOURCE_WORKER),
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
                                        for root in (OUT, PRIOR_OUT)
                                        for p in root.rglob("*")
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
                    amendment = read(response.with_suffix(".resource-amendment.json"))
                    assert amendment == dict(
                        schema="jope.jrb01.resource-amendment-response/v1",
                        base_learner_sha256=WORKER_SHA,
                        wrapper_sha256=RESOURCE_SHA,
                        source_loader_sha256=protocol.file_sha(learner_source.__file__),
                        qualification_sha256=RESOURCE_QUALIFICATION_SHA,
                        authority_sha256=RULE_SHA,
                        request_sha256=protocol.file_sha(req),
                        response_sha256=protocol.file_sha(response),
                        plan_sha256=PLAN_SHA,
                        job_id=job["id"],
                        phase=phase,
                    ), "missing/invalid effective-resource provenance"
                    result = read(response)
                    require_worker_provenance(result, QUALIFIED_WORKER_SHA)
                    if phase == "train":
                        assert result["actual_budget"] == protocol.expected_counts(
                            plan["config"]
                        )
                        assert (
                            result["plan_sha256"] == PLAN_SHA
                            and result["job_id"] == job["id"]
                        )
                        train_receipt = result
                        physical_updates += result["actual_budget"]["optimizer_steps"]
                        if physical_updates > audit["planned_physical_updates"]:
                            raise ValueError(
                                "unexpected extra physical optimizer updates"
                            )
                    else:
                        driver.admit_result(plan, job, result)
                        publish_qualified(job["id"], response)
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
            driver.collect(
                plan, QUALIFIED
            )  # full inventory/paired-source validation only
            for relative, entry in initial_files.items():
                assert protocol.file_sha(STUDY / relative) == entry["sha256"], (
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
                    wall_seconds=prior_execution_seconds
                    + time.monotonic()
                    - run_started,
                    elapsed_since_reservation_seconds=time.time()
                    - campaign["started_unix"],
                    physical_optimizer_updates=physical_updates,
                    qualified_optimizer_updates=5400,
                    rejected_optimizer_updates=200,
                    qualified_results=str(QUALIFIED),
                    jobs=len(plan["jobs"]),
                    resume_authority_sha256=RULE_SHA,
                    budget_reset=False,
                    **extension_fields,
                ),
            )
            event(
                "ALL_JOBS_COMPLETE",
                jobs=54,
                optimizer_steps=5400,
                physical_optimizer_steps=physical_updates,
                rejected_optimizer_steps=200,
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
        str(QUALIFIED),
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
