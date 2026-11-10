"""Prepared BASE/native/prereg gates; ONE CPU admission; full36 or incomplete."""

from pathlib import Path
import json, os, subprocess, sys, threading, hashlib
import resource_guard as resource
import guard_h386 as guard
from resource_probe import probe, ASSIGNED, admission_pass
from forecast_h386 import estimate

HERE = Path(__file__).resolve().parent
P = HERE.parent
D = Path("/home/ubuntu/jope-data/h386-jrb01")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    assert (
        os.uname().nodename.split(".")[0] == "xbabe2"
        and os.getpriority(os.PRIO_PROCESS, 0) == 19
    )
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    assert (
        json.loads((P / "base-prepared.json").read_text())["status"] == "PASS"
        and json.loads((P / "nativecheck-exit.json").read_text())["EXIT_STATUS"] == "0"
    )
    assert (
        json.loads((D / "regression.json").read_text())["status"] == "PASS"
        and json.loads((P / "input_identity_validation.json").read_text())["status"]
        == "PASS"
    )
    assert json.loads((P / "base-native-config.json").read_text())["status"] == "PASS"
    manifest = json.loads((HERE / "manifest.json").read_text())
    for n, h in manifest["source_sha256"].items():
        assert hashlib.sha256((HERE / n).read_bytes()).hexdigest() == h
    assert (
        json.loads((D / "nl15-fetch-receipt.json").read_text())["status"]
        == "EXACT_VERSION_GET_AND_SHA_PASS"
    )
    assert json.loads((D / "nl15-regression.json").read_text())["status"] == "PASS"
    assert json.loads((D / "nl15-native-config.json").read_text())["status"] == "PASS"
    assert json.loads((D / "controls.json").read_text())["status"] == "PASS"
    assert json.loads((D / "reference-custody.json").read_text())["status"] == "PASS"
    assert not (P / "capacity-interval-mpstat.json").exists()
    total, idle = resource.ticks()
    state = resource.Window(__import__("time").monotonic(), total, idle)
    initial = resource.snapshot(state, "h386-run.service")
    save(P / "initial-resource.json", initial)
    if initial["reason"]:
        save(
            D / "STOP.json",
            dict(
                status="INCOMPLETE_RESOURCE_GUARD",
                stage="before_capacity_sample",
                **initial,
            ),
        )
        raise SystemExit(2)
    threading.Thread(target=guard.monitor, daemon=True).start()
    eligible, idle, raw, host = probe()
    (P / "capacity-interval-mpstat.json").write_text(raw)
    cap = dict(
        eligible_cores=eligible,
        both_smt_idle=idle,
        host_at_probe=host,
        host_cpu_maximum=70,
        sample_seconds=5,
        minimum_required=4,
        assigned_cores=list(ASSIGNED),
        maximum_taken=8,
        ordering="first at most8 passing pairs in the single union A+A2 interval",
        capacity_retries=0,
    )
    save(P / "capacity.json", cap)
    save(D / "pre_fit_resource.json", cap)
    if not admission_pass(eligible, host["cpu_percent"]):
        save(
            D / "INCOMPLETE_CPU_ADMISSION.json",
            dict(
                status="INCOMPLETE_CPU_ADMISSION",
                capacity=cap,
                reason="Need at least4 admitted cores and same-interval hostCPU<=70%",
                optimizer_fits=0,
                outer_scores=0,
            ),
        )
        raise SystemExit(2)
    # The controller and its resource-monitor thread inherit only admitted cores.
    before = {
        p.name: sorted(os.sched_getaffinity(int(p.name)))
        for p in Path("/proc/self/task").iterdir()
    }
    for tid in before:
        os.sched_setaffinity(int(tid), set(eligible))
    after = {
        p.name: sorted(os.sched_getaffinity(int(p.name)))
        for p in Path("/proc/self/task").iterdir()
    }
    assert all(v == eligible for v in after.values())
    save(
        P / "controller-affinity-binding.json",
        dict(before=before, after=after, eligible_cores=eligible, maximum_workers=8),
    )
    forecast = estimate(cap)
    forecast.update(capacity=cap)
    save(P / "runtime-forecast-at-launch.json", forecast)
    save(D / "runtime_forecast.json", forecast)
    print(json.dumps(dict(forecast=forecast)), flush=True)
    if not forecast["launch_full_36"]:
        save(D / "INCOMPLETE_RUNTIME_FORECAST.json", forecast)
        raise SystemExit(2)
    env = dict(os.environ, H386_WORKERS=str(len(eligible)))
    for phase in ["fit", "score"]:
        subprocess.run(
            [
                "taskset",
                "-c",
                ",".join(map(str, eligible)),
                sys.executable,
                str(HERE / "run_h386.py"),
                phase,
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
