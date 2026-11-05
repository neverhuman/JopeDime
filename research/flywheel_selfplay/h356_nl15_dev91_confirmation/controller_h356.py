"""One admission, immutable artifact fetch, then one frozen-policy inference pass."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
import psutil
from common import D, E, S, STOP, durable, load, prereg, sha, utc
from resource_probe import probe, admission_pass
import guard_h356


def main():
    assert (
        os.uname().nodename.split(".")[0] == "xbabe2"
        and os.getpriority(os.PRIO_PROCESS, 0) == 19
    )
    publication = prereg()
    test = load(E / "CONTROLS-PASS.json")
    assert (
        test["failed"] == test["errors"] == test["skipped"] == 0
        and test["executed"] >= 15
    )
    for name, digest in test["source_sha256"].items():
        assert sha(S / name) == digest, name
    assert psutil.virtual_memory().available >= 20 * 2**30 and os.getloadavg()[0] <= 85
    durable(
        D / "ADMISSION-STARTED.json",
        dict(utc=utc(), prereg=publication["head"]),
        exclusive=True,
    )
    threading.Thread(target=guard_h356.monitor, daemon=True).start()
    cores, idle, raw, host = probe()
    (D / "capacity-interval-mpstat.json").write_text(raw)
    cap = dict(
        utc=utc(),
        eligible_cores=cores,
        both_smt_idle=idle,
        host_at_probe=host,
        capacity_probes=1,
        capacity_retries=0,
        pool="A0..14+sibs64..78",
        maximum_admitted=8,
        minimum_required=4,
        compute_worker_processes=1,
        numeric_threads=1,
        compute_stop=STOP.isoformat(),
    )
    durable(D / "capacity.json", cap)
    durable(E / "capacity.json", cap)
    if not admission_pass(cores, host["cpu_percent"]):
        durable(
            D / "INCOMPLETE_CPU_ADMISSION.json",
            dict(
                status="INCOMPLETE_CPU_ADMISSION",
                capacity=cap,
                score_calls=0,
                model_fits=0,
            ),
        )
        raise SystemExit(2)
    before = {
        p.name: sorted(os.sched_getaffinity(int(p.name)))
        for p in Path("/proc/self/task").iterdir()
    }
    for tid in before:
        os.sched_setaffinity(int(tid), set(cores))
    after = {
        p.name: sorted(os.sched_getaffinity(int(p.name)))
        for p in Path("/proc/self/task").iterdir()
    }
    assert all(set(v) == set(cores) for v in after.values())
    durable(
        D / "controller-affinity.json", dict(before=before, after=after, cores=cores)
    )
    durable(E / "ADMITTED.json", cap)
    for phase in ["fetch_h356.py", "infer_h356.py"]:
        subprocess.run([sys.executable, "-B", str(S / phase)], check=True)
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                utc=utc(),
                verdict=load(D / "metrics.json")["verdict"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
