#!/usr/bin/env python3
"""Bounded optional GPR qualification; GPU phase holds the shared service lease."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--cases", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference", type=Path)
p.add_argument("--reference-only", action="store_true")
p.add_argument("--batch-size", type=int, default=4)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
lease = None
if not a.reference_only:
    assert a.reference is not None
    lease = (a.root / "runs/encoding-service.lock").open("a+")
    deadline = time.monotonic() + 600
    while True:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise RuntimeError("GPU lease wait exceeded 600 seconds")
            time.sleep(0.5)
begin = time.monotonic()
workers = []
try:
    for rank in range(8):
        cpu = rank * 2 if rank < 4 else 62 + (rank - 4) * 2
        command = [
            "taskset",
            "-c",
            f"{cpu},{cpu + 1}",
            str(a.binary),
            "--root",
            str(a.root),
            "--cases",
            str(a.cases),
            "--output",
            str(a.output / f"rank-{rank}.json"),
            "--rank",
            str(rank),
            "--world-size",
            "8",
            "--batch-size",
            str(a.batch_size),
        ]
        command += (
            ["--reference-only"]
            if a.reference_only
            else ["--device", f"cuda:{rank}", "--reference", str(a.reference)]
        )
        log = (a.output / f"rank-{rank}.log").open("w")
        env = dict(
            os.environ,
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            RAYON_NUM_THREADS="2",
        )
        workers.append((rank, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    for rank, worker, log in workers:
        code = worker.wait(timeout=max(1, 300 - (time.monotonic() - begin)))
        assert code == 0, f"rank {rank} failed ({code}); see log"
        log.close()
finally:
    for _, worker, log in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        log.close()
    if lease:
        lease.close()
reports = [json.loads((a.output / f"rank-{rank}.json").read_text()) for rank in range(8)]
assert len({r["cases_sha256"] for r in reports}) == 1
rows = [row for r in reports for row in r["rows"]]
summary = dict(
    schema="parallel.experimental_gpr_qualification_merged.v1",
    production_enabled=False,
    reward_admission=False,
    cases_sha256=reports[0]["cases_sha256"],
    executable_sha256=reports[0]["executable_sha256"],
    elapsed_seconds=time.monotonic() - begin,
    world_size=8,
    reference_only=a.reference_only,
    batch_size=a.batch_size,
    rows=rows,
    ranks=[
        dict(
            path=f"rank-{rank}.json",
            sha256=hashlib.sha256((a.output / f"rank-{rank}.json").read_bytes()).hexdigest(),
        )
        for rank in range(8)
    ],
)
(a.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(
    json.dumps(
        dict(
            output=str(a.output / "summary.json"),
            rows=len(rows),
            elapsed_seconds=summary["elapsed_seconds"],
            status_counts={
                s: sum(r.get("status", "reference_complete") == s for r in rows)
                for s in sorted({r.get("status", "reference_complete") for r in rows})
            },
        ),
        indent=2,
    )
)
