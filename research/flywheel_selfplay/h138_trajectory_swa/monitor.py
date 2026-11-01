"""One attempt; CPU resource monitor outside numerical workers."""

from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import psutil


def worker_environment(base):
    return dict(
        base,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        RAYON_NUM_THREADS="1",
        JE_ARROW_MALLOC_CONF="background_thread:false",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
    )


HERE = Path(__file__).resolve().parent
ARROW_ALLOCATOR_CONF = "background_thread:false"
WALL_SECONDS = 2 * 60 * 60
RECOVERY_SECONDS = 60
POLL_SECONDS = 0.25


def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def reason(s):
    for bad, label in [
        (s["rss"] >= 6_000_000_000, "RSS>=6GB"),
        (s["threads"] > 4, "threads>4"),
        (s["free_bytes"] < 40_000_000_000, "free<40GB"),
        (s["scratch"] >= 2_000_000_000, "scratch>=2GB"),
        (s["load1"] > 0.85 * s["cpus"], "load>85%"),
        (s["available"] < 16 * 1024**3, "MemAvailable<16GiB"),
    ]:
        if bad:
            return label
    return None


def sample(root, proc=None):
    st = os.statvfs(root)
    rss = threads = 0
    if proc:
        for p in [proc] + proc.children(recursive=True):
            try:
                rss += p.memory_info().rss
                threads += p.num_threads()
            except psutil.NoSuchProcess:
                pass
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        disk_percent=100 * (1 - st.f_bavail / st.f_blocks),
        free_bytes=st.f_bavail * st.f_frsize,
        scratch=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        load1=os.getloadavg()[0],
        cpus=len(os.sched_getaffinity(0)),
        available=psutil.virtual_memory().available,
    )


def pressure_reason(s):
    if s["load1"] > 0.85 * s["cpus"]:
        return "load>85%"
    if s["available"] < 16 * 1024**3:
        return "MemAvailable<16GiB"
    return None


def hard_reason(s):
    bad = reason(s)
    return None if bad in ("load>85%", "MemAvailable<16GiB") else bad


class RecoveryGate:
    def __init__(self):
        self.paused = False
        self.stable_since = None

    def action(self, s, now):
        if not self.paused and pressure_reason(s):
            self.paused = True
            self.stable_since = None
            return "pause"
        if self.paused:
            if s["load1"] < 80 and s["available"] >= 16 * 1024**3:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= RECOVERY_SECONDS:
                    self.paused = False
                    self.stable_since = None
                    return "resume"
            else:
                self.stable_since = None
        return None


def terminate(child):
    if child is not None and child.poll() is None:
        # SIGTERM is queued while stopped; SIGCONT lets it terminate promptly.
        try:
            os.killpg(child.pid, signal.SIGTERM)
            os.killpg(child.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def watch(root, c, record, stage, deadline, child=None):
    """Wait before launch or supervise the same live child; never restart a fit."""
    gate, interval, peak = RecoveryGate(), None, 0
    proc = psutil.Process(child.pid) if child is not None else None
    try:
        while child is None or child.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError("two-hour wall budget exhausted")
            try:
                s = sample(c, proc)
            except psutil.NoSuchProcess:
                break
            peak = max(peak, s["rss"])
            with (c / "resources.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        dict(
                            stage=stage, waiting=child is None, paused=gate.paused, **s
                        )
                    )
                    + "\n"
                )
            bad = hard_reason(s)
            if bad:
                raise RuntimeError("guard: " + bad)
            action = gate.action(s, now)
            if action == "pause":
                if child is not None:
                    os.killpg(child.pid, signal.SIGSTOP)
                interval = dict(
                    stage=stage,
                    kind="worker_pause" if child else "pre_stage_wait",
                    pid=child.pid if child else None,
                    reason=pressure_reason(s),
                    start_utc=s["utc"],
                    start_monotonic=now,
                    end_utc=None,
                )
                record["pause_intervals"].append(interval)
                save(c / "monitor.json", record)
            elif action == "resume":
                if child is not None:
                    os.killpg(child.pid, signal.SIGCONT)
                interval.update(
                    end_utc=s["utc"],
                    duration_seconds=now - interval["start_monotonic"],
                    disposition="resumed" if child else "ready_after_wait",
                )
                interval = None
                save(c / "monitor.json", record)
            if child is None and not gate.paused:
                return 0
            time.sleep(POLL_SECONDS)
    except BaseException:
        terminate(child)
        raise
    finally:
        if interval is not None:
            interval.update(
                end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                duration_seconds=time.monotonic() - interval["start_monotonic"],
                disposition="terminal",
            )
            save(c / "monitor.json", record)
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    args = p.parse_args()
    c = args.custody.resolve()
    launch = json.loads((c / "launch.json").read_text())
    if not launch["merged_source_verified"] or not launch["fold_seeds_verified"]:
        raise RuntimeError("merge and fold-seed gates required")
    for name, digest in json.loads((HERE / "manifest.json").read_text())[
        "executor_sha256"
    ].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("executor changed")
    root = c / "run"
    root.mkdir()  # single attempt, never resume implicitly
    env = worker_environment(
        dict(
            os.environ,
            GIT_DIR=launch["git_dir"],
            JE_ARROW_MALLOC_CONF=ARROW_ALLOCATOR_CONF,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPYCACHEPREFIX=str(root / "bytecode"),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            NUMEXPR_MAX_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
    )
    record = dict(
        source=launch,
        host=socket.gethostname(),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_budget_seconds=WALL_SECONDS,
        pause_intervals=[],
        worker_environment={
            k: env[k]
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "JE_ARROW_MALLOC_CONF",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        },
        stages=[],
        status="RUNNING",
    )
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    child = None
    try:
        for stage in [
            "baseline",
            "train_R882",
            "train_SWA4_UNIFORM",
            "train_SWA2_LATE",
            "train_EMA_0.5",
            "primary",
            "secondary",
            "summary",
        ]:
            watch(root, c, record, stage, deadline)
            cmd = [
                "nice",
                "-n",
                "19",
                sys.executable,
                str(HERE / "runner.py"),
                "--custody",
                str(c),
                "--stage",
                stage,
            ]
            with (c / f"{stage}.log").open("w") as log:
                child = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                peak = watch(root, c, record, stage, deadline, child)
                code = child.wait()
                if time.monotonic() >= deadline:
                    raise RuntimeError("two-hour wall budget exhausted")
            record["stages"].append(
                dict(stage=stage, exit_status=code, peak_rss=peak, command=cmd)
            )
            save(c / "monitor.json", record)
            if code:
                raise RuntimeError("stage failed: " + stage)
        record["status"] = "COMPLETE"
    except BaseException as error:
        record["status"] = "INCOMPLETE"
        record["error"] = str(error)
        raise
    finally:
        terminate(child)
        record["wall_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        save(c / "monitor.json", record)


if __name__ == "__main__":
    main()
