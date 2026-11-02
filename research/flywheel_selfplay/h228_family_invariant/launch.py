"""H228 exact-PR-source, exclusive-GPU, single-attempt supervisor (no model imports)."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

PREFIX = "research/flywheel_selfplay/h228_family_invariant/"
GPU = "GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503"
DEFAULT_REPO = (
    Path("/home/ubuntu/JopeDime/.agent/worktrees/h205-argon")
    if Path("/home/ubuntu/JopeDime/.agent/worktrees/h205-argon").exists()
    else Path("/home/ubuntu/JopeDime/.agent/worktrees/h226-argon")
)
REPO = DEFAULT_REPO
CUSTODY = Path("/home/ubuntu/jope-rescue-20260921/h228-argon")
COMPUTE_DEADLINE_EPOCH = 1793662200  # 2026-11-02 23:30:00Z
LOAD5_PAUSE_THRESHOLD = 85.0
LOAD5_RESUME_THRESHOLD = 80.0
MIN_GPU_FREE_MIB = 6144
MIN_AVAILABLE_RAM = 8 * 2**30
MIN_FREE_DISK = 150_000_000_000


def blob(commit, path, repo=None):
    target_repo = REPO if repo is None else repo
    return subprocess.check_output(
        ["git", "-C", str(target_repo), "show", commit + ":" + path]
    )


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_vram_and_apps():
    raw_gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free_mib = 0
    for line in raw_gpu.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == GPU:
            free_mib = int(float(parts[1]))
            break
    raw_apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    app_mib = {}
    for line in raw_apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0] == GPU:
            try:
                app_mib[int(parts[1])] = int(float(parts[2]))
            except ValueError:
                app_mib[int(parts[1])] = 0
    return free_mib, app_mib


class SustainedLoad:
    """Separate monotonic clocks: a recovery resets only its own signal."""

    def __init__(self):
        self.since = {"load": None, "cpu": None}
        self.elapsed = {"load": 0.0, "cpu": 0.0}

    def overloaded(self, load1, load5, cpu, cores, now):
        for key, over in (("load", load1 > 0.85 * cores), ("cpu", cpu > 85)):
            if not over:
                self.since[key] = None
            elif self.since[key] is None:
                self.since[key] = now
            self.elapsed[key] = (
                0.0 if self.since[key] is None else now - self.since[key]
            )
        return load5 > 0.85 * cores or any(t >= 300 for t in self.elapsed.values())


LOAD_STATE = SustainedLoad()


def should_pause_worker(load5, paused):
    """Hysteresis load5 pause guard for own worker only: pause > 85.0, resume <= 80.0."""
    if not paused and load5 > LOAD5_PAUSE_THRESHOLD:
        return True
    if paused and load5 > LOAD5_RESUME_THRESHOLD:
        return True
    return False


def guard_violations(sample, *, paused=False, other_gpu_pids=None):
    violations = []
    if sample["available"] < MIN_AVAILABLE_RAM:
        violations.append(
            dict(field="available", value=sample["available"], limit=MIN_AVAILABLE_RAM)
        )
    if sample["free"] < MIN_FREE_DISK:
        violations.append(dict(field="free", value=sample["free"], limit=MIN_FREE_DISK))
    if sample.get("effective_gpu_free_mib", MIN_GPU_FREE_MIB) < MIN_GPU_FREE_MIB:
        violations.append(
            dict(
                field="gpu_free_mib",
                value=sample["effective_gpu_free_mib"],
                limit=MIN_GPU_FREE_MIB,
            )
        )
    if sample["effective_cores"] > 4:
        violations.append(
            dict(field="effective_cores", value=sample["effective_cores"], limit=4)
        )
    if sample.get("child_rss", 0) >= 6_000_000_000:
        violations.append(
            dict(field="child_rss", value=sample["child_rss"], limit=6_000_000_000)
        )
    if should_pause_worker(sample["load5"], paused):
        violations.append(
            dict(
                field="load5",
                value=sample["load5"],
                limit=LOAD5_RESUME_THRESHOLD if paused else LOAD5_PAUSE_THRESHOLD,
            )
        )
    if sample.get("overloaded"):
        violations.append(dict(field="overloaded", value=True, limit=False))
    if other_gpu_pids is not None:
        sample["other_gpu_pids"] = sorted(other_gpu_pids)
    sample["guard_trip"] = bool(violations)
    sample["violating_field"] = violations[0]["field"] if violations else None
    sample["violating_value"] = violations[0]["value"] if violations else None
    sample["violations"] = violations
    return violations


def guard(child=None, *, state=None, now=None, paused=False, pause_on_trip=False):
    disk = psutil.disk_usage("/home/ubuntu")
    host_cores = os.cpu_count() or 102
    sample = dict(
        load1=os.getloadavg()[0],
        load5=os.getloadavg()[1],
        effective_cores=len(os.sched_getaffinity(0)),
        host_cores=host_cores,
        cpu_percent=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        swap_free=psutil.swap_memory().free,
        free=disk.free,
        used_fraction=disk.used / (disk.used + disk.free),
    )
    state = LOAD_STATE if state is None else state
    overloaded = state.overloaded(
        sample["load1"],
        sample["load5"],
        sample["cpu_percent"],
        host_cores,
        time.monotonic() if now is None else now,
    )
    sample.update(
        load_over_seconds=state.elapsed["load"],
        cpu_over_seconds=state.elapsed["cpu"],
        overloaded=overloaded,
    )
    allowed = set()
    if child is not None:
        parent = psutil.Process(child.pid)
        tree = [parent]
        for p in parent.children(recursive=True):
            try:
                if p.is_running():
                    tree.append(p)
            except psutil.NoSuchProcess:
                pass
        allowed = {p.pid for p in tree}
        rss_total = 0
        for p in tree:
            try:
                if p.is_running():
                    rss_total += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        sample["child_rss"] = rss_total
    gpu_free_mib, app_mib = gpu_vram_and_apps()
    own_gpu_mib = sum(mib for pid, mib in app_mib.items() if pid in allowed)
    other_gpu = set(app_mib) - allowed
    sample["gpu_free_mib"] = gpu_free_mib
    sample["own_gpu_mib"] = own_gpu_mib
    sample["effective_gpu_free_mib"] = gpu_free_mib + own_gpu_mib
    guard_violations(sample, paused=paused, other_gpu_pids=other_gpu)
    if pause_on_trip:
        return sample
    if (
        sample["available"] < MIN_AVAILABLE_RAM
        or sample["free"] < MIN_FREE_DISK
        or sample["effective_gpu_free_mib"] < MIN_GPU_FREE_MIB
        or sample["effective_cores"] > 4
    ):
        raise RuntimeError(
            f"resource refusal ({sample['violating_field']}={sample['violating_value']}): {sample!r}"
        )
    if sample.get("child_rss", 0) >= 6_000_000_000:
        raise RuntimeError(
            f"child RSS limit ({sample['violating_field']}={sample['violating_value']})"
        )
    return sample


def admitted_pr(pr, commit):
    if (
        not str(pr.get("headRefName", "")).startswith("rd/h228-family-invariant")
        or pr.get("baseRefName") != "main"
        or pr.get("isDraft")
    ):
        return False
    return (
        pr.get("state") == "MERGED"
        and (pr.get("mergeCommit") or {}).get("oid") == commit
    )


@contextmanager
def fifo_lock():
    """Use the shared FIFO ticket order; retain own ticket and done marker."""
    queue = Path("/home/ubuntu/jope_plan/qualification.queue")
    queue.mkdir(parents=True, exist_ok=True)
    ticket = queue / f"{time.time_ns()}-{os.getpid()}-H228"
    ticket.write_text(f"pid={os.getpid()}\nwave=H228\n")
    try:
        with Path("/home/ubuntu/qualification.lock").open("a") as lock:
            while True:
                live = []
                for item in queue.iterdir():
                    parts = item.name.split("-", 2)
                    if (
                        len(parts) != 3
                        or not parts[0].isdigit()
                        or not parts[1].isdigit()
                        or item.name.endswith(".done")
                        or item.with_name(item.name + ".done").exists()
                    ):
                        continue
                    try:
                        process = psutil.Process(int(parts[1]))
                        if (
                            process.is_running()
                            and process.status() != psutil.STATUS_ZOMBIE
                        ):
                            live.append(item)
                    except psutil.NoSuchProcess:
                        pass
                live.sort(key=lambda item: (int(item.name.split("-", 1)[0]), item.name))
                if live and live[0] == ticket:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                if time.time() >= COMPUTE_DEADLINE_EPOCH:
                    raise RuntimeError("FIFO admission deadline; no study launched")
                time.sleep(2)
            yield str(ticket)
    finally:
        ticket.with_name(ticket.name + ".done").touch(exist_ok=False)


def load_pr_metadata(pr_number, pr_receipt=None, repo=REPO):
    if pr_receipt is not None:
        pr = json.loads(Path(pr_receipt).read_text())
        merge_oid = (pr.get("mergeCommit") or {}).get("oid", "")
        if not merge_oid:
            raise RuntimeError("PR receipt missing mergeCommit.oid")
        subprocess.check_call(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                merge_oid,
                "origin/main",
            ]
        )
        return pr
    return json.loads(
        subprocess.check_output(
            [
                "/home/ubuntu/.local/bin/gh-role",
                "writer",
                "pr",
                "view",
                str(pr_number),
                "--repo",
                "neverhuman/JopeDime",
                "--json",
                "state,mergeCommit,headRefOid,headRefName,baseRefName,isDraft,url",
            ],
            text=True,
        )
    )


def main(args):
    global REPO
    if getattr(args, "repo", None):
        REPO = Path(args.repo)
    os.nice(19)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    if os.uname().nodename != "xbabe1":
        raise RuntimeError("xbabe1 only")
    pr = load_pr_metadata(args.pr, getattr(args, "pr_receipt", None), REPO)
    if not admitted_pr(pr, args.commit):
        raise RuntimeError("requires exact merged H228 preregistration")
    pins_raw = blob(args.commit, PREFIX + "manifest.json", REPO)
    pins = json.loads(pins_raw)
    for name, expected in pins["source_sha256"].items():
        raw = blob(args.commit, PREFIX + name, REPO)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("merged source pin mismatch")
    if pins["gpu_uuid"] != GPU:
        raise RuntimeError("wrong registered GPU")
    actual = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if "0, " + GPU not in actual.splitlines():
        raise RuntimeError("device index/UUID mismatch")
    CUSTODY.mkdir(parents=True, exist_ok=True)
    with fifo_lock() as ticket:
        if time.time() >= COMPUTE_DEADLINE_EPOCH:
            raise RuntimeError("compute stop 23:30Z passed")
        psutil.cpu_percent()
        time.sleep(1)
        preflight = guard(paused=False, pause_on_trip=True)
        while preflight["guard_trip"]:
            with (CUSTODY / "preflight-guard.jsonl").open("a") as pf_log:
                pf_log.write(
                    json.dumps(
                        dict(
                            preflight,
                            at=time.time(),
                            paused_before_start=True,
                        )
                    )
                    + "\n"
                )
                pf_log.flush()
            if time.time() >= COMPUTE_DEADLINE_EPOCH:
                raise RuntimeError(
                    f"compute stop 23:30Z passed during preflight wait ({preflight['violating_field']}={preflight['violating_value']})"
                )
            time.sleep(2)
            preflight = guard(paused=True, pause_on_trip=True)
        claim = dict(
            wave="H228",
            fifo_ticket=ticket,
            authority="hw_t1640_argon_h228.md; hw_t1905_argon_h228_gpulost.md: pre-outcome xbabe1 RTX 4090 retarget authorized",
            attempt=1,
            source_commit=args.commit,
            pr=pr,
            manifest_sha256=hashlib.sha256(pins_raw).hexdigest(),
            gpu_uuid=GPU,
            preflight=preflight,
            start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=5400,
            deadline_epoch=min(time.time() + 5400, COMPUTE_DEADLINE_EPOCH),
        )
        write_new(CUSTODY / "attempt1-claim.json", claim)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            RAYON_NUM_THREADS="1",
            OMP_MAX_ACTIVE_LEVELS="1",
            OMP_NESTED="FALSE",
            PYTHONDONTWRITEBYTECODE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        bootstrap = "import subprocess,sys; p='research/flywheel_selfplay/h228_family_invariant/runner.py'; s=subprocess.check_output(['git','-C',sys.argv[2],'show',sys.argv[4]+':'+p]); exec(compile(s,p,'exec'),{'__name__':'__main__','__file__':p})"
        command = [
            sys.executable,
            "-c",
            bootstrap,
            "--repo",
            str(REPO),
            "--commit",
            args.commit,
            "--output",
            str(CUSTODY / "attempt1"),
        ]
        error, child = None, None
        paused = False
        pause_events = 0
        with (
            (CUSTODY / "attempt1.stdout").open("x") as output,
            (CUSTODY / "supervisor.jsonl").open("x") as monitor,
        ):
            try:
                child = subprocess.Popen(
                    command,
                    env=env,
                    cwd=CUSTODY,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_new(
                    CUSTODY / "attempt1-process.json",
                    dict(pid=child.pid, source_commit=args.commit, command=command),
                )
                while child.poll() is None:
                    try:
                        sample = guard(child, paused=paused, pause_on_trip=True)
                    except psutil.NoSuchProcess:
                        if child.poll() is not None:
                            break
                        raise
                    want_pause = bool(sample["guard_trip"])
                    if want_pause and not paused:
                        os.killpg(child.pid, signal.SIGSTOP)
                        paused = True
                        pause_events += 1
                    elif not want_pause and paused:
                        os.killpg(child.pid, signal.SIGCONT)
                        paused = False
                    sample["at"] = time.time()
                    sample["worker_paused"] = paused
                    sample["pause_events"] = pause_events
                    monitor.write(json.dumps(sample) + "\n")
                    monitor.flush()
                    if time.time() > claim["deadline_epoch"]:
                        raise RuntimeError("fixed attempt deadline")
                    time.sleep(2)
            except BaseException as exc:
                error = repr(exc)
                if child is not None and child.poll() is None:
                    if paused:
                        try:
                            os.killpg(child.pid, signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            finally:
                write_new(
                    CUSTODY / "attempt1-terminal.json",
                    dict(
                        exit_code=None if child is None else child.returncode,
                        error=error,
                        pause_events=pause_events,
                        source_commit=args.commit,
                        finished_epoch=time.time(),
                    ),
                )
        if error or child is None or child.returncode != 0:
            raise RuntimeError("attempt INCOMPLETE; inspect terminal receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--pr-receipt", default=None)
    main(parser.parse_args())
