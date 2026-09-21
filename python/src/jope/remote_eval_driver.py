"""Host-local parallel eval driver — runs ON a cluster node, driven over ONE ssh channel.

The Python orchestrator on the head node can only keep ~8 ssh lock-step lanes busy at once
(GIL + one reader thread per lane + ssh round-trip latency), so requesting 90 lanes gains
nothing. Instead the orchestrator ships a batch of rows to each host and runs THIS script
there: it reads JSONL rows (each carrying ``_idx``) from stdin and fans them across ``N``
persistent local bridge processes (fast local pipes, no ssh in the inner loop), streaming
one result line per row (echoing ``_idx``) to stdout. All parallelism happens on the node,
using its cores directly — no head-node GIL bottleneck.

Self-contained (stdlib only) so it runs on any node that has the bridge binary; invoked as
``python3 -m jope.remote_eval_driver --bin <path> --subcommand eval-regression --workers 90``.
"""

from __future__ import annotations

import argparse
import json
import queue
import select
import subprocess
import sys
import threading
import time


def _worker(
    bin_path: str, subcommand: str, jobs: queue.Queue, out_lock: threading.Lock, default_timeout: float | None
) -> None:
    """One persistent bridge process draining rows from the shared queue (lock-step).

    Timeouts use ``select`` on the bridge's stdout fd — no per-read thread — so N workers
    are N mostly-I/O-blocked threads driving N Rust bridge processes across N cores.
    """
    proc: subprocess.Popen | None = None

    def spawn() -> subprocess.Popen:
        return subprocess.Popen(
            [bin_path, subcommand],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def read_line(p: subprocess.Popen, timeout: float | None) -> str | None:
        if timeout is None:
            return p.stdout.readline() if p.stdout else None
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return "__TIMEOUT__"
            ready, _, _ = select.select([p.stdout], [], [], remaining)
            if ready:
                return p.stdout.readline()

    try:
        proc = spawn()
        while True:
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                break
            idx = row.get("_idx")
            timeout = row.get("_timeout_s", default_timeout)
            started = time.time()
            if proc is None or proc.poll() is not None:
                proc = spawn()
            try:
                proc.stdin.write(json.dumps(row) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                proc = spawn()
                try:
                    proc.stdin.write(json.dumps(row) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    _emit(
                        out_lock,
                        {"_idx": idx, "valid": False, "bridge_crash": True, "errors": ["write failed twice"]},
                    )
                    continue
            line = read_line(proc, timeout)
            if line == "__TIMEOUT__":
                proc.kill()
                proc = None
                _emit(
                    out_lock,
                    {
                        "_idx": idx,
                        "valid": False,
                        "timeout": True,
                        "errors": [f"eval timeout after {time.time() - started:.1f}s"],
                    },
                )
                continue
            if not line:
                proc.kill()
                proc = None
                _emit(out_lock, {"_idx": idx, "valid": False, "bridge_crash": True, "errors": ["eof"]})
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                result = {"valid": False, "errors": ["bad json from bridge"]}
            result["_idx"] = idx
            _emit(out_lock, result)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()


def _emit(lock: threading.Lock, obj: dict) -> None:
    line = json.dumps(obj)
    with lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _shard_worker(args_tuple) -> list[str]:
    """Run one persistent bridge over a shard of rows in a SEPARATE process.

    Process-level parallelism (not threads) is essential: N single-threaded bridge
    processes each in their own Python process have no shared GIL, so they truly use N
    cores. A thread pool serializes on the GIL for the per-row JSON/select work and
    collapses to ~1-way concurrency on fast evals.
    """
    bin_path, subcommand, rows, default_timeout = args_tuple
    out: list[str] = []

    def spawn():
        return subprocess.Popen(
            [bin_path, subcommand],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    proc = spawn()
    try:
        for row in rows:
            idx = row.get("_idx")
            timeout = row.get("_timeout_s", default_timeout)
            started = time.time()
            if proc.poll() is not None:
                proc = spawn()
            try:
                proc.stdin.write(json.dumps(row) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                proc = spawn()
                proc.stdin.write(json.dumps(row) + "\n")
                proc.stdin.flush()
            line = None
            if timeout is None:
                line = proc.stdout.readline()
            else:
                deadline = started + timeout
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        proc.kill()
                        proc = spawn()
                        line = json.dumps(
                            {
                                "_idx": idx,
                                "valid": False,
                                "timeout": True,
                                "errors": [f"eval timeout after {timeout:.1f}s"],
                            }
                        )
                        break
                    ready, _, _ = select.select([proc.stdout], [], [], remaining)
                    if ready:
                        line = proc.stdout.readline()
                        break
            if not line:
                out.append(json.dumps({"_idx": idx, "valid": False, "bridge_crash": True, "errors": ["eof"]}))
                proc = spawn()
                continue
            try:
                result = json.loads(line)
                result["_idx"] = idx
                out.append(json.dumps(result))
            except json.JSONDecodeError:
                out.append(line if line.startswith("{") else json.dumps({"_idx": idx, "valid": False}))
    finally:
        if proc.poll() is None:
            proc.kill()
    return out


def main(argv: list[str] | None = None) -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="bridge binary path on this host")
    ap.add_argument("--subcommand", default="eval-regression")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--default-timeout", type=float, default=None)
    args = ap.parse_args(argv)

    rows = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return 0

    workers = max(1, min(args.workers, len(rows)))
    # round-robin rows into `workers` shards so each process gets a balanced mix
    shards: list[list] = [[] for _ in range(workers)]
    for i, row in enumerate(rows):
        shards[i % workers].append(row)
    tasks = [(args.bin, args.subcommand, shard, args.default_timeout) for shard in shards if shard]

    with mp.Pool(processes=len(tasks)) as pool:
        for result_lines in pool.imap_unordered(_shard_worker, tasks):
            for rl in result_lines:
                sys.stdout.write(rl + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
