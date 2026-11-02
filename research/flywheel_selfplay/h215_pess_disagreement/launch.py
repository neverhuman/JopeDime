"""H215 exact-PR-source, exclusive-GPU, single-attempt supervisor (no model imports)."""

import argparse
import fcntl
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

PREFIX = "research/flywheel_selfplay/h215_pess_disagreement/"
GPU = "GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b"
REPO = Path("/home/ubuntu/JopeDime")
CUSTODY = Path("/home/ubuntu/jope-rescue-20260921/h215-astra")


def blob(commit, path):
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", commit + ":" + path]
    )


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def process_gpu_pids():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(line.split(",")[1])
        for line in raw.splitlines()
        if line.split(",")[0].strip() == GPU
    }


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


def guard(child=None, *, state=None, now=None):
    disk = psutil.disk_usage("/home/ubuntu")
    sample = dict(
        load1=os.getloadavg()[0],
        load5=os.getloadavg()[1],
        effective_cores=len(os.sched_getaffinity(0)),
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
        sample["effective_cores"],
        time.monotonic() if now is None else now,
    )
    sample.update(
        load_over_seconds=state.elapsed["load"], cpu_over_seconds=state.elapsed["cpu"]
    )
    if (
        overloaded
        or (
            child is None
            and (
                sample["load1"] > 0.85 * sample["effective_cores"]
                or sample["cpu_percent"] > 85
            )
        )
        or sample["available"] < 16 * 2**30
        or sample["free"] < 40_000_000_000
        or sample["used_fraction"] > 0.938
    ):
        raise RuntimeError("resource refusal: " + repr(sample))
    allowed = set()
    if child is not None:
        parent = psutil.Process(child.pid)
        tree = [parent] + parent.children(recursive=True)
        allowed = {p.pid for p in tree}
        sample["child_rss"] = sum(p.memory_info().rss for p in tree if p.is_running())
        if sample["child_rss"] >= 6_000_000_000:
            raise RuntimeError("child RSS limit")
    if process_gpu_pids() - allowed:
        raise RuntimeError("other GPU consumer; no shared GPU")
    return sample


def admitted_pr(pr, commit):
    if (
        pr.get("headRefName") != "rd/h215-pess-disagreement"
        or pr.get("baseRefName") != "main"
        or pr.get("isDraft")
    ):
        return False
    if pr.get("state") == "OPEN":
        return pr.get("headRefOid") == commit
    return (
        pr.get("state") == "MERGED"
        and (pr.get("mergeCommit") or {}).get("oid") == commit
    )


@contextmanager
def fifo_lock():
    """Use the shared FIFO ticket order; retain own ticket and done marker."""
    queue = Path("/home/ubuntu/jope_plan/qualification.queue")
    queue.mkdir(parents=True, exist_ok=True)
    ticket = queue / f"{time.time_ns()}-{os.getpid()}-H215"
    ticket.write_text(f"pid={os.getpid()}\nwave=H215\n")
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
                if (
                    time.time() >= 1793615400
                ):  # 2026-11-02 10:30Z, full90min before12:00Z.
                    raise RuntimeError("FIFO admission deadline; no study launched")
                time.sleep(2)
            yield str(ticket)
    finally:
        ticket.with_name(ticket.name + ".done").touch(exist_ok=False)


def main(args):
    os.nice(19)
    if os.uname().nodename != "xbabe2":
        raise RuntimeError("xbabe2 only")
    pr = json.loads(
        subprocess.check_output(
            [
                "/home/ubuntu/.local/bin/gh-role",
                "writer",
                "pr",
                "view",
                str(args.pr),
                "--repo",
                "neverhuman/JopeDime",
                "--json",
                "state,mergeCommit,headRefOid,headRefName,baseRefName,isDraft,url",
            ],
            text=True,
        )
    )
    # hw_t0230 explicitly permits launch once this prereg PR is open.
    if not admitted_pr(pr, args.commit):
        raise RuntimeError("requires exact open or merged H215 preregistration")
    pins_raw = blob(args.commit, PREFIX + "manifest.json")
    pins = json.loads(pins_raw)
    for name, expected in pins["source_sha256"].items():
        raw = blob(args.commit, PREFIX + name)
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
        if time.time() >= 1793615400:
            raise RuntimeError("latest launch10:30Z passed")
        psutil.cpu_percent()
        time.sleep(1)
        preflight = guard()
        # This create-only claim survives failures and forbids an implicit retry.
        claim = dict(
            wave="H215",
            fifo_ticket=ticket,
            authority="hw_t0230_astra.md: exact open PR launch authorized",
            attempt=1,
            source_commit=args.commit,
            pr=pr,
            manifest_sha256=hashlib.sha256(pins_raw).hexdigest(),
            gpu_uuid=GPU,
            preflight=preflight,
            start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=5400,
            deadline_epoch=time.time() + 5400,
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
        bootstrap = "import subprocess,sys; p='research/flywheel_selfplay/h215_pess_disagreement/runner.py'; s=subprocess.check_output(['git','-C',sys.argv[2],'show',sys.argv[4]+':'+p]); exec(compile(s,p,'exec'),{'__name__':'__main__','__file__':p})"
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
                        sample = guard(child)
                    except psutil.NoSuchProcess:
                        if child.poll() is not None:
                            break
                        raise
                    sample["at"] = time.time()
                    monitor.write(json.dumps(sample) + "\n")
                    monitor.flush()
                    if time.time() > claim["deadline_epoch"]:
                        raise RuntimeError("fixed attempt deadline")
                    time.sleep(2)
            except BaseException as exc:
                error = repr(exc)
                if child is not None and child.poll() is None:
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
    main(parser.parse_args())
