"""H211 exact-PR-source, exclusive-GPU, single-attempt supervisor (no model imports)."""

import argparse
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

PREFIX = "research/flywheel_selfplay/h211_gpu_delta_calibration/"
GPU = "GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b"
REPO = Path("/home/ubuntu/JopeDime")
CUSTODY = Path("/home/ubuntu/jope-rescue-20260921/h211-astra")


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


def guard(child=None):
    disk = psutil.disk_usage("/home/ubuntu")
    sample = dict(
        load1=os.getloadavg()[0],
        effective_cores=len(os.sched_getaffinity(0)),
        cpu_percent=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        used_fraction=disk.used / (disk.used + disk.free),
    )
    if (
        sample["load1"] > 0.85 * sample["effective_cores"]
        or sample["cpu_percent"] > 85
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
        pr.get("headRefName") != "rd/h211-gpu-delta"
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
    # gbw_0040 explicitly permits launch once this prereg PR is open.
    if not admitted_pr(pr, args.commit):
        raise RuntimeError("requires exact open or merged H211 preregistration")
    pins_raw = blob(args.commit, PREFIX + "manifest.json")
    pins = json.loads(pins_raw)
    current_pins = dict(pins["source_sha256"])
    current_pins.update(pins.get("current_source_sha256", {}))
    for name, expected in current_pins.items():
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
    with Path("/home/ubuntu/qualification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        psutil.cpu_percent()
        time.sleep(1)
        preflight = guard()
        # This create-only claim survives failures and forbids an implicit retry.
        claim = dict(
            wave="H211",
            authority="gbw_0040_rd_astra.md: exact open PR launch authorized",
            attempt=1,
            source_commit=args.commit,
            pr=pr,
            manifest_sha256=hashlib.sha256(pins_raw).hexdigest(),
            gpu_uuid=GPU,
            preflight=preflight,
            start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=3600,
            deadline_epoch=time.time() + 3600,
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
        runner_path = REPO / "research/flywheel_selfplay/h211_gpu_delta_calibration/runner.py"
        command = [
            sys.executable,
            str(runner_path),
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
