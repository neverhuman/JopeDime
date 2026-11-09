#!/usr/bin/env python3
"""H335 resumable runner for the frontier AUDIT evaluator (``frontier_audit_h335``).

Runs every (cid, entrant) pair as its own subprocess of the pinned binary, under
RLIMIT_CPU (set in ``preexec_fn``) and a wall-clock limit, and appends one JSON line per
pair to an append-only JSONL file. Pairs already present in the file are skipped, so an
interrupted run resumes where it stopped.

The scheduler is single-threaded (``Popen`` + ``selectors`` + ``os.wait4``) so that
``preexec_fn`` is never used from a multi-threaded parent.

Record statuses:
  * ``completed`` / ``failed`` -- reported by the binary (failed => utility 0.0);
  * ``timeout``  -- CPU (RLIMIT_CPU -> SIGXCPU/SIGKILL) or wall limit exceeded, utility 0.0;
  * ``failed``   -- also used for crashes, unexpected exit codes or unparsable output.
An envelope the binary refuses (exit code 3: bad header, nq != 0, fold byte outside {0, 1})
gets no record; ``run`` raises ``EnvelopeRefused`` after the remaining pairs finish.

Stdlib only.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
import resource
import selectors
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

WRAPPER_VERSION = "h335-frontier-audit-wrapper-v1"
ENTRANTS = ("Opus5p5_Extra", "Gemini4_argon_high", "GPT6p1_Sol_Max", "GPT6_Astra_Max")
SEED = 42
EXIT_REFUSED = 3
CPU_HARD_GRACE_S = 5
STDOUT_CAP = 1 << 20
STDERR_TAIL = 4000
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RUST_TEST_THREADS",
)


class EnvelopeRefused(RuntimeError):
    """One or more envelopes were refused by the binary; ``summary`` holds the run summary."""

    def __init__(self, summary: dict):
        self.summary = summary
        super().__init__(f"envelopes refused: {summary['refused']}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_envelope_counts(path) -> tuple[int, int]:
    """(n_search, n_audit) from a JTT1 header + fold bytes (used for timeout records)."""
    with open(path, "rb") as f:
        head = f.read(17)
        if len(head) != 17 or head[:4] != b"JTT1":
            return (-1, -1)
        nt = struct.unpack_from("<I", head, 4)[0]
        folds = f.read(nt)
    return (folds.count(0), folds.count(1))


def _child_env() -> dict:
    env = dict(os.environ)
    for k in THREAD_ENV:
        env[k] = "1"
    env.pop("RUST_MIN_STACK", None)
    return env


def _make_preexec(cpu_limit_s: int):
    cpu_rlimit = (cpu_limit_s, cpu_limit_s + CPU_HARD_GRACE_S)
    setrlimit = resource.setrlimit
    rl_cpu = resource.RLIMIT_CPU
    rl_core = resource.RLIMIT_CORE

    def _preexec():
        setrlimit(rl_cpu, cpu_rlimit)
        setrlimit(rl_core, (0, 0))

    return _preexec


def _existing_pairs(out_jsonl: Path, binary_sha256: str) -> set:
    done = set()
    if not out_jsonl.exists():
        return done
    with open(out_jsonl, "rb") as f:
        for raw in f:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue  # partial trailing line from an interrupted write
            if not isinstance(rec, dict) or "cid" not in rec or "entrant" not in rec:
                continue
            have = rec.get("binary_sha256")
            if have is not None and have != binary_sha256:
                raise ValueError(
                    f"{out_jsonl} holds records from binary {have}, pinned {binary_sha256}"
                )
            done.add((rec["cid"], rec["entrant"]))
    return done


class _Job:
    __slots__ = (
        "cid", "entrant", "env_path", "env_sha", "proc", "t0", "out", "err",
        "open_fds", "wall_killed",
    )

    def __init__(self, cid, entrant, env_path, env_sha):
        self.cid = cid
        self.entrant = entrant
        self.env_path = env_path
        self.env_sha = env_sha
        self.proc = None
        self.t0 = 0.0
        self.out = bytearray()
        self.err = bytearray()
        self.open_fds = 0
        self.wall_killed = False


def _base_record(job: _Job, status: str, error: str | None) -> dict:
    n_search, n_audit = read_envelope_counts(job.env_path)
    return {
        "cid": job.cid,
        "entrant": job.entrant,
        "seed": SEED,
        "status": status,
        "n_search": n_search,
        "n_audit": n_audit,
        "rmse_audit": None,
        "base_rmse_audit": None,
        "ratio": None,
        "utility": 0.0,
        "wall_ms": None,
        "error": error,
        "audit_preds_sha256": None,
        "harness_version": None,
    }


def _stderr_tail(job: _Job) -> str:
    return job.err[-STDERR_TAIL:].decode("utf-8", "replace").strip()


def _classify(job: _Job, wait_status: int, cpu_s: float, wall_s: float,
              cpu_limit_s: int, wall_limit_s: float):
    """Return (record or None, refused_flag)."""
    code = os.waitstatus_to_exitcode(wait_status)
    sig = -code if code < 0 else None
    if job.wall_killed:
        return _base_record(job, "timeout", f"wall limit {wall_limit_s}s exceeded"), False
    if sig in (signal.SIGXCPU, signal.SIGKILL) and (
        sig == signal.SIGXCPU or cpu_s >= cpu_limit_s - 0.5
    ):
        return _base_record(job, "timeout", f"cpu limit {cpu_limit_s}s exceeded"), False
    if code == EXIT_REFUSED:
        return None, True
    if code != 0:
        what = f"signal {signal.Signals(sig).name}" if sig else f"exit code {code}"
        return _base_record(job, "failed", f"binary {what}: {_stderr_tail(job)}"), False
    lines = [ln for ln in bytes(job.out).splitlines() if ln.strip()]
    try:
        rec = json.loads(lines[-1])
        ok = (
            isinstance(rec, dict)
            and rec.get("cid") == job.cid
            and rec.get("entrant") == job.entrant
            and rec.get("status") in ("completed", "failed")
        )
    except (IndexError, ValueError):
        ok = False
    if not ok:
        return _base_record(job, "failed", f"unparsable binary output: {_stderr_tail(job)}"), False
    if rec["status"] != "completed" or not (
        isinstance(rec.get("utility"), (int, float)) and math.isfinite(rec["utility"])
    ):
        rec["status"] = "failed"
        rec["utility"] = 0.0
    return rec, False


def run(envs_dir, cids, out_jsonl, binary, cpu_limit_s=300, wall_limit_s=900, jobs=1, *,
        binary_sha256, entrants=ENTRANTS, poll_s=0.05) -> dict:
    """Evaluate every (cid, entrant) pair; append one JSON line per pair to ``out_jsonl``.

    ``binary_sha256`` is the pinned sha256 of ``binary``; a mismatch raises before any run.
    Returns a summary dict; raises ``EnvelopeRefused`` (carrying the summary) if the binary
    refused any envelope.
    """
    envs_dir = Path(envs_dir)
    out_jsonl = Path(out_jsonl)
    binary = Path(binary).resolve()
    cpu_limit_s = max(1, int(math.ceil(cpu_limit_s)))
    wall_limit_s = float(wall_limit_s)
    jobs = max(1, int(jobs))
    entrants = tuple(entrants)
    for e in entrants:
        if e not in ENTRANTS:
            raise ValueError(f"unknown entrant {e!r}")

    got = sha256_file(binary)
    if got != binary_sha256:
        raise ValueError(f"binary sha256 mismatch: {binary} is {got}, pinned {binary_sha256}")

    uniq = list(dict.fromkeys(str(c).strip() for c in cids if str(c).strip()))
    for c in uniq:
        if "/" in c or c in (".", ".."):
            raise ValueError(f"invalid cid {c!r}")
    env_paths = {c: envs_dir / f"{c}.jtt1" for c in uniq}
    missing = [c for c, p in env_paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing envelopes in {envs_dir}: {missing}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_jsonl, "ab")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RuntimeError(f"{out_jsonl} is locked by another runner")
    try:
        done = _existing_pairs(out_jsonl, binary_sha256)
        if out_jsonl.stat().st_size > 0:
            with open(out_jsonl, "rb") as rf:
                rf.seek(-1, os.SEEK_END)
                if rf.read(1) != b"\n":
                    fh.write(b"\n")
                    fh.flush()

        env_sha = {c: sha256_file(p) for c, p in env_paths.items()}
        pending = collections.deque(
            _Job(c, e, env_paths[c], env_sha[c])
            for c in uniq for e in entrants if (c, e) not in done
        )
        skipped = len(uniq) * len(entrants) - len(pending)
        summary = {
            "ran": 0, "skipped": skipped, "refused": [],
            "status_counts": collections.Counter(),
        }
        child_env = _child_env()
        preexec = _make_preexec(cpu_limit_s)
        sel = selectors.DefaultSelector()
        running: dict[int, _Job] = {}

        def _read(fileobj, job, buf_name):
            data = os.read(fileobj.fileno(), 65536)
            if not data:
                sel.unregister(fileobj)
                fileobj.close()
                job.open_fds -= 1
                return
            buf = getattr(job, buf_name)
            buf.extend(data)
            if buf_name == "err" and len(buf) > 4 * STDERR_TAIL:
                del buf[: len(buf) - STDERR_TAIL]
            elif buf_name == "out" and len(buf) > STDOUT_CAP:
                del buf[STDOUT_CAP:]

        def _drain(job):
            for fobj in (job.proc.stdout, job.proc.stderr):
                if fobj is not None and not fobj.closed:
                    while not fobj.closed:
                        _read(fobj, job, "out" if fobj is job.proc.stdout else "err")

        while pending or running:
            while pending and len(running) < jobs:
                job = pending.popleft()
                job.proc = subprocess.Popen(
                    [str(binary), str(job.env_path), job.entrant, "-"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=child_env, preexec_fn=preexec, close_fds=True,
                )
                job.t0 = time.monotonic()
                sel.register(job.proc.stdout, selectors.EVENT_READ, (job, "out"))
                sel.register(job.proc.stderr, selectors.EVENT_READ, (job, "err"))
                job.open_fds = 2
                running[job.proc.pid] = job

            for key, _ in sel.select(timeout=poll_s):
                job, buf_name = key.data
                _read(key.fileobj, job, buf_name)

            now = time.monotonic()
            for pid, job in list(running.items()):
                wpid, status, ru = os.wait4(pid, os.WNOHANG)
                if wpid == 0:
                    if not job.wall_killed and now - job.t0 > wall_limit_s:
                        os.kill(pid, signal.SIGKILL)
                        job.wall_killed = True
                    continue
                job.proc.returncode = os.waitstatus_to_exitcode(status)
                _drain(job)
                del running[pid]
                cpu_s = ru.ru_utime + ru.ru_stime
                wall_s = time.monotonic() - job.t0
                rec, refused = _classify(job, status, cpu_s, wall_s, cpu_limit_s, wall_limit_s)
                if refused:
                    summary["refused"].append(
                        {"cid": job.cid, "entrant": job.entrant, "stderr": _stderr_tail(job)}
                    )
                    continue
                rec.update({
                    "envelope_sha256": job.env_sha,
                    "binary_sha256": binary_sha256,
                    "exit_code": job.proc.returncode,
                    "cpu_s": round(cpu_s, 6),
                    "runner_wall_s": round(wall_s, 6),
                    "cpu_limit_s": cpu_limit_s,
                    "wall_limit_s": wall_limit_s,
                    "wrapper_version": WRAPPER_VERSION,
                })
                fh.write((json.dumps(rec, allow_nan=False) + "\n").encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
                summary["ran"] += 1
                summary["status_counts"][rec["status"]] += 1
        sel.close()
    finally:
        fh.close()

    summary["status_counts"] = dict(summary["status_counts"])
    if summary["refused"]:
        raise EnvelopeRefused(summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs-dir", required=True)
    ap.add_argument("--cids-file", required=True, help="one cid per line (envelope stem)")
    ap.add_argument("--out", required=True, help="append-only JSONL output")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--binary-sha256", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cpu-limit-s", type=float, default=300)
    ap.add_argument("--wall-limit-s", type=float, default=900)
    ap.add_argument("--entrants", default=",".join(ENTRANTS))
    a = ap.parse_args(argv)
    cids = Path(a.cids_file).read_text().split()
    try:
        summary = run(
            a.envs_dir, cids, a.out, a.binary, cpu_limit_s=a.cpu_limit_s,
            wall_limit_s=a.wall_limit_s, jobs=a.jobs, binary_sha256=a.binary_sha256,
            entrants=[e for e in a.entrants.split(",") if e],
        )
    except EnvelopeRefused as exc:
        print(json.dumps(exc.summary, sort_keys=True), file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
