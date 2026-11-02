"""Independent H225 supervisor: pause the owned worker on load, fixed deadline."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h225_multistep_regret/"
DATA = Path("/home/ubuntu/jope-data/h225")
DEADLINE = dt.datetime.fromisoformat("2026-11-02T19:00:00+00:00").timestamp()


def save(path, value, *, exclusive=False):
    path = Path(path)
    temp = path if exclusive else path.with_name(path.name + ".next")
    with temp.open("x" if exclusive else "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if not exclusive:
        os.replace(temp, path)


def observation(child=None):
    rss = psutil.Process().memory_info().rss
    processes = []
    if child is not None and child.poll() is None:
        try:
            processes = [
                psutil.Process(child.pid),
                *psutil.Process(child.pid).children(recursive=True),
            ]
        except psutil.NoSuchProcess:
            pass
    for proc in processes:
        try:
            rss += proc.memory_info().rss
            if len(proc.cpu_affinity()) > 4:
                raise RuntimeError("descendant affinity exceeds four cores")
        except psutil.NoSuchProcess:
            pass
    return dict(
        at=time.time(),
        load5=os.getloadavg()[1],
        available_ram=psutil.virtual_memory().available,
        free_disk=psutil.disk_usage("/").free,
        process_tree_rss=rss,
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        affinity=sorted(os.sched_getaffinity(0)),
    )


def pause_reasons(obs, now):
    if now >= DEADLINE:
        raise TimeoutError("fixed H225 19:00Z compute deadline")
    if obs["process_tree_rss"] >= 4 * 2**30:
        raise MemoryError("hard H225 process-tree RSS boundary")
    return [
        name
        for name, bad in (
            ("load5>60", obs["load5"] > 60),
            ("RAM<20GiB", obs["available_ram"] < 20 * 2**30),
            ("disk<300GB", obs["free_disk"] < 300_000_000_000),
        )
        if bad
    ]


def signal_owned(child, identity, sig):
    if child.poll() is not None:
        return
    try:
        proc = psutil.Process(child.pid)
        if (
            proc.create_time() != identity
            or os.getsid(child.pid) != child.pid
            or os.getpgid(child.pid) != child.pid
        ):
            raise RuntimeError("owned worker PID/session identity changed")
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        return


def stop_owned(child, identity):
    signal_owned(child, identity, signal.SIGCONT)
    signal_owned(child, identity, signal.SIGTERM)
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        signal_owned(child, identity, signal.SIGKILL)
        child.wait(timeout=3)


def run_phase(source, phase):
    root = DATA / ("smoke" if phase == "smoke" else "attempt-001")
    bootstrap = (
        "import subprocess,sys;src=sys.argv.pop(1);raw=subprocess.check_output(['git','show',src+':"
        + PREFIX
        + "runner.py'],cwd='/home/ubuntu/JopeDime');exec(compile(raw,'H225-immutable-runner.py','exec'),{'__name__':'__main__'})"
    )
    command = [
        sys.executable,
        "-B",
        "-c",
        bootstrap,
        source,
        "--source",
        source,
        "--phase",
        phase,
        "--root",
        str(root),
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AWS_", "GH_", "GITHUB_", "JOPE_AWS"))
        and k != "CARGO_TARGET_DIR"
    }
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(DATA / "tmp"),
    )
    with (DATA / (phase + ".log")).open("xb") as output:
        child = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        identity = psutil.Process(child.pid).create_time()
        save(
            DATA / (phase + "-launch.json"),
            dict(
                source_commit=source,
                command=command,
                pid=child.pid,
                create_time=identity,
                phase=phase,
                deadline="2026-11-02T19:00:00Z",
                observation=observation(child),
            ),
            exclusive=True,
        )
        paused = False
        peak = 0
        try:
            while child.poll() is None:
                obs = observation(child)
                peak = max(peak, obs["process_tree_rss"])
                reasons = pause_reasons(obs, time.time())
                if reasons and not paused:
                    signal_owned(child, identity, signal.SIGSTOP)
                    paused = True
                elif not reasons and paused:
                    signal_owned(child, identity, signal.SIGCONT)
                    paused = False
                save(
                    DATA / "supervisor-status.json",
                    dict(
                        phase=phase,
                        source_commit=source,
                        worker_pid=child.pid,
                        state="PAUSED" if paused else "RUNNING",
                        pause_reasons=reasons,
                        peak_process_tree_rss=peak,
                        observation=obs,
                        compute_stop="2026-11-02T19:00:00Z",
                        hard_closeout="2026-11-02T19:30:00Z",
                    ),
                )
                time.sleep(2)
            return dict(
                phase=phase,
                exit_code=child.returncode,
                peak_process_tree_rss=peak,
                metrics=str(root / "metrics.json"),
            )
        except BaseException:
            stop_owned(child, identity)
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2":
        raise RuntimeError("xbabe2 only")
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    # Admission waits without creating a scientific attempt and never resets the deadline.
    while pause_reasons(observation(), time.time()):
        time.sleep(2)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source, "origin/main"],
        cwd=REPO,
        check=True,
    )
    manifest = subprocess.check_output(
        ["git", "show", args.source + ":" + PREFIX + "manifest.json"], cwd=REPO
    )
    DATA.mkdir(exist_ok=True)
    (DATA / "tmp").mkdir(exist_ok=True)
    if (
        (DATA / "experiment-launch.json").exists()
        or (DATA / "smoke").exists()
        or (DATA / "attempt-001").exists()
    ):
        raise RuntimeError("one H225 smoke and one full attempt; no scientific retry")
    save(
        DATA / "experiment-launch.json",
        dict(
            source_commit=args.source,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            compute_deadline="2026-11-02T19:00:00Z",
            closeout_deadline="2026-11-02T19:30:00Z",
        ),
        exclusive=True,
    )
    result = dict(status="INCOMPLETE", source_commit=args.source, phases=[])
    try:
        for phase in ("smoke", "run"):
            stage = run_phase(args.source, phase)
            result["phases"].append(stage)
            if stage["exit_code"]:
                raise RuntimeError(phase + " exited nonzero; no retry")
        result["status"] = "COMPLETE"
    except BaseException as error:
        result["error"] = str(error)
    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(DATA / "terminal-supervisor.json", result, exclusive=True)
    save(DATA / "supervisor-status.json", dict(state=result["status"], **result))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
