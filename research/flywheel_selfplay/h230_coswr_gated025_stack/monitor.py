"""H230 single CPU attempt: kernel four-core tree, strict RSS and absolute deadline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil

REPO = Path("/home/ubuntu/JopeDime")
ROOT = Path("/home/ubuntu/jope-rescue-20260921/h230-astra")
PREFIX = "research/flywheel_selfplay/h230_coswr_gated025_stack/"
DEADLINE = 1793676600.0  # 2026-11-03 03:30 UTC; fixed even during pauses.


def resources(process=None, output=None):
    disk = psutil.disk_usage("/home/ubuntu")
    mem = psutil.virtual_memory()
    load = os.getloadavg()[1]
    tree = [] if process is None else [process] + process.children(recursive=True)
    rss = 0
    for child in tree:
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    sample = dict(
        epoch=time.time(),
        load5=load,
        free_bytes=disk.free,
        available_ram=mem.available,
        swap_free=psutil.swap_memory().free,
        tree_rss=rss,
    )
    if disk.free < 300_000_000_000 or mem.available < 20 * 2**30 or rss >= 4 * 2**30:
        raise RuntimeError("resource stop: " + repr(sample))
    if process is not None:
        allowed = set(os.sched_getaffinity(process.pid))
        if len(allowed) > 4:
            raise RuntimeError("four-core kernel affinity missing")
        for child in tree:
            try:
                threads = child.threads()
            except psutil.NoSuchProcess:
                continue
            for thread in threads:
                try:
                    affinity = set(os.sched_getaffinity(thread.id))
                except ProcessLookupError:
                    continue
                if not affinity <= allowed:
                    raise RuntimeError("thread escaped CPU affinity")
        sample["affinity"] = sorted(allowed)
    if output is not None and output.exists():
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        sample["output_bytes"] = size
        if size >= 2_000_000_000:
            raise RuntimeError("2GB own output cap")
    return sample


class LoadPause:
    """Hysteresis applies to load only; hard guards keep running while paused."""

    def __init__(self):
        self.paused = False
        self.next_check = 0.0

    def update(self, load5, now):
        if not self.paused and load5 > 60:
            self.paused = True
            self.next_check = now + 60
            return "pause"
        if self.paused and now >= self.next_check:
            self.next_check = now + 60
            if load5 <= 55:
                self.paused = False
                return "resume"
        return None


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main(args):
    if os.uname().nodename != "xbabe2":
        raise RuntimeError("xbabe2 CPU only")
    os.nice(19)
    cores = [2, 22, 67, 104]
    if not set(cores) <= set(os.sched_getaffinity(0)):
        raise RuntimeError("registered core IDs unavailable")
    os.sched_setaffinity(0, cores)
    if time.time() >= DEADLINE - 1800:
        raise RuntimeError("too late for single primary attempt")
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
                "state,mergeCommit,headRefName,baseRefName,isDraft,mergedAt",
            ]
        )
    )
    if (
        pr["state"] != "MERGED"
        or pr["mergeCommit"]["oid"] != args.commit
        or pr["headRefName"] != "rd/h230-coswr-gated025"
        or pr["baseRefName"] != "main"
        or pr["isDraft"]
    ):
        raise RuntimeError("exact merged prereg required")

    def blob(path):
        return subprocess.check_output(
            ["git", "-C", str(REPO), "show", args.commit + ":" + PREFIX + path]
        )

    if pr.get("mergedAt") is None:
        raise RuntimeError("merged timestamp required")
    from datetime import datetime

    if (
        datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00")).timestamp()
        > 1793656800
    ):
        raise RuntimeError("prereg merge missed22:00Z")
    pins = json.loads(blob("manifest.json"))
    for name, expected in pins["source_sha256"].items():
        if hashlib.sha256(blob(name)).hexdigest() != expected:
            raise RuntimeError("source pin mismatch")
    sample = resources(psutil.Process())
    claim = dict(
        wave="H230",
        attempt=1,
        source_commit=args.commit,
        pr=args.pr,
        authority="hw_t1945_astra_h224close_h230.md",
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        reboot_disposition="INTERRUPTED; no restart",
        deadline_epoch=DEADLINE,
        start_epoch=time.time(),
        affinity=cores,
        resource_limits=pins["resources"],
        admission=sample,
        retry_allowed=False,
    )
    write(ROOT / "attempt1-claim.json", claim)
    script = "import subprocess,sys; p='research/flywheel_selfplay/h230_coswr_gated025_stack/runner.py'; raw=subprocess.check_output(['git','-C','/home/ubuntu/JopeDime','show',sys.argv[2]+':'+p]);exec(compile(raw,p,'exec'),{'__name__':'__main__','__file__':p})"
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        OMP_THREAD_LIMIT="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NESTED="FALSE",
        OMP_MAX_ACTIVE_LEVELS="1",
    )
    child = None
    error = None
    load_state = LoadPause()
    with (
        (ROOT / "attempt1.stdout").open("x") as log,
        (ROOT / "supervisor.jsonl").open("x") as samples,
    ):
        try:
            child = subprocess.Popen(
                [
                    "python3",
                    "-c",
                    script,
                    "--commit",
                    args.commit,
                    "--root",
                    str(ROOT / "attempt1"),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(
                ROOT / "attempt1-process.json",
                dict(pid=child.pid, source_commit=args.commit),
            )
            while child.poll() is None:
                if time.time() >= DEADLINE:
                    raise RuntimeError("absolute compute stop03:30Z")
                try:
                    sample = resources(psutil.Process(), ROOT / "attempt1")
                except (psutil.NoSuchProcess, ProcessLookupError):
                    if child.poll() is not None:
                        break
                    raise
                action = load_state.update(sample["load5"], time.monotonic())
                if action is not None:
                    try:
                        os.kill(
                            child.pid,
                            signal.SIGSTOP if action == "pause" else signal.SIGCONT,
                        )
                    except ProcessLookupError:
                        if child.poll() is not None:
                            break
                        raise
                sample.update(load_paused=load_state.paused, load_action=action)
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                time.sleep(0.25)
            if time.time() >= DEADLINE:
                raise RuntimeError("terminal after deadline")
        except BaseException as exc:
            error = repr(exc)
            if child is not None and child.poll() is None:
                if load_state.paused:
                    os.kill(child.pid, signal.SIGCONT)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        finally:
            write(
                ROOT / "attempt1-terminal.json",
                dict(
                    exit_code=None if child is None else child.returncode,
                    error=error,
                    source_commit=args.commit,
                    finished_epoch=time.time(),
                ),
            )
    if error or child is None or child.returncode != 0:
        raise RuntimeError("single H230 attempt INCOMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--pr", type=int, required=True)
    main(parser.parse_args())
