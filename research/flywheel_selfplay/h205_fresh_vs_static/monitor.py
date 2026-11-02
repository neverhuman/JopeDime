"""H205 resource and thread-rule-(b) monitor on xbabe1 RTX 4090.

Enforces:
- Numeric threads <= 4 (OMP/MKL/OpenBLAS/NUMEXPR/RAYON/torch intra-op <= 4).
- CUDA runtime/driver helper threads exempt from the numeric cap without disabling resource monitoring.
- CPU <= 85%, sustained 1-min load <= 85% of online cores, nice == 19, RSS < 6 GB,
  MemAvailable >= 16 GiB, disk used <= 93.8% and free >= 40 GB.
- Never stops, signals, or renices foreign processes (including live flywheel PID 2297929 on xbabe1).
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
HOST_CORES = os.sysconf("SC_NPROCESSORS_ONLN")
LOAD_LIMIT = 0.85 * HOST_CORES
MAX_NUMERIC_THREADS = 4
LIVE_FLYWHEEL_PID = 2297929
CAMPAIGN_WALL_BUDGET_CAP_SECONDS = 25200.0
GPU_ATTEMPTS_1_2_ELAPSED_SECONDS = 1970.9372255061753
SEAM_HELPER_0752Z_ELAPSED_SECONDS = 50.10207135695964
PRIOR_ATTEMPTS_ELAPSED_SECONDS = 2021.039296863135
REMAINING_WALL_BUDGET_SECONDS = (
    CAMPAIGN_WALL_BUDGET_CAP_SECONDS - PRIOR_ATTEMPTS_ELAPSED_SECONDS
)
NUMERIC_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
CUDA_THREAD_PREFIXES = (
    "cuda",
    "nv",
    "pt_autograd",
    "cudnn",
    "cublas",
    "nccl",
)


def is_cuda_helper_thread(comm_name):
    """Return True iff the Linux task comm name belongs to a CUDA driver/runtime helper thread."""
    lower = str(comm_name).strip().lower()
    return any(lower.startswith(prefix) for prefix in CUDA_THREAD_PREFIXES)


def classify_thread_names(comm_names):
    """Split OS thread comm names into exempt CUDA helper threads vs numeric/compute threads."""
    names = [str(n).strip() for n in comm_names if str(n).strip()]
    cuda_helpers = [n for n in names if is_cuda_helper_thread(n)]
    numeric = [n for n in names if not is_cuda_helper_thread(n)]
    return dict(
        total_threads=len(names),
        numeric_threads=len(numeric),
        cuda_helper_threads=len(cuda_helpers),
        numeric_thread_names=numeric,
        cuda_helper_thread_names=cuda_helpers,
    )


def check_numeric_env_and_torch(env=None, torch_intra_op=None, torch_inter_op=None):
    """Verify OMP/MKL/OpenBLAS/NUMEXPR/RAYON and torch intra-op/inter-op thread settings are <= 4."""
    env_map = os.environ if env is None else env
    bad = []
    for k in NUMERIC_ENV_KEYS:
        val = env_map.get(k)
        if val is not None:
            try:
                n = int(val)
            except ValueError:
                bad.append(f"{k}={val!r}")
                continue
            if n < 1 or n > MAX_NUMERIC_THREADS:
                bad.append(f"{k}={n}>{MAX_NUMERIC_THREADS}")
    if torch_intra_op is not None and int(torch_intra_op) > MAX_NUMERIC_THREADS:
        bad.append(f"torch_intra_op={torch_intra_op}>{MAX_NUMERIC_THREADS}")
    if torch_inter_op is not None and int(torch_inter_op) > MAX_NUMERIC_THREADS:
        bad.append(f"torch_inter_op={torch_inter_op}>{MAX_NUMERIC_THREADS}")
    return bad


def compute_remaining_wall_budget_seconds(
    prior_elapsed_seconds=PRIOR_ATTEMPTS_ELAPSED_SECONDS,
    campaign_cap_seconds=CAMPAIGN_WALL_BUDGET_CAP_SECONDS,
    requested_remaining_seconds=None,
):
    """Compute remaining campaign wall budget and refuse launch if pre-consumed budget >= cap."""
    prior = float(prior_elapsed_seconds)
    cap = float(campaign_cap_seconds)
    if prior < PRIOR_ATTEMPTS_ELAPSED_SECONDS - 1e-9:
        raise RuntimeError(
            f"cannot reduce prior_elapsed_seconds below recorded ledger ({prior} < {PRIOR_ATTEMPTS_ELAPSED_SECONDS})"
        )
    if cap > CAMPAIGN_WALL_BUDGET_CAP_SECONDS + 1e-9:
        raise RuntimeError(
            f"cannot increase campaign_cap_seconds above {CAMPAIGN_WALL_BUDGET_CAP_SECONDS} ({cap})"
        )
    remaining = cap - prior
    if requested_remaining_seconds is not None:
        req = float(requested_remaining_seconds)
        if req > remaining + 1e-9:
            raise RuntimeError(
                f"requested_remaining_seconds ({req}) exceeds remaining campaign budget ({remaining})"
            )
        remaining = min(remaining, req)
    if remaining <= 0.0:
        raise RuntimeError(
            f"prelaunch cumulative budget refusal: prior_elapsed={prior}s, cap={cap}s, remaining={remaining}s <= 0"
        )
    return remaining


def compute_shared_process_deadline(
    process_start_monotonic,
    prior_elapsed_seconds=PRIOR_ATTEMPTS_ELAPSED_SECONDS,
    campaign_cap_seconds=CAMPAIGN_WALL_BUDGET_CAP_SECONDS,
    requested_remaining_seconds=None,
    env_hard_deadline_monotonic=None,
):
    """Bind supervisor, runner Stage 1/2/3, and JRB-01 wait loop to ONE process-start remaining deadline."""
    start_mono = float(process_start_monotonic)
    remaining = compute_remaining_wall_budget_seconds(
        prior_elapsed_seconds=prior_elapsed_seconds,
        campaign_cap_seconds=campaign_cap_seconds,
        requested_remaining_seconds=requested_remaining_seconds,
    )
    hard_deadline = start_mono + remaining
    if env_hard_deadline_monotonic is not None:
        hard_deadline = min(hard_deadline, float(env_hard_deadline_monotonic))
    eff_remaining = hard_deadline - start_mono
    if eff_remaining <= 0.0:
        raise RuntimeError(
            f"prelaunch cumulative budget refusal: non-positive effective remaining window ({eff_remaining}s)"
        )
    return dict(
        process_start_monotonic=start_mono,
        prior_elapsed_seconds=float(prior_elapsed_seconds),
        campaign_cap_seconds=float(campaign_cap_seconds),
        remaining_budget_seconds=float(eff_remaining),
        hard_deadline_monotonic=float(hard_deadline),
    )


def is_cumulative_wall_deadline_exceeded(
    elapsed_seconds,
    prior_elapsed_seconds=PRIOR_ATTEMPTS_ELAPSED_SECONDS,
    campaign_cap_seconds=CAMPAIGN_WALL_BUDGET_CAP_SECONDS,
    remaining_budget_seconds=REMAINING_WALL_BUDGET_SECONDS,
):
    """Return True iff elapsed_seconds exceeds remaining_budget_seconds or prior + elapsed > cap."""
    el = float(elapsed_seconds)
    return (
        el > 23178.960703136865
        or el > float(remaining_budget_seconds)
        or (float(prior_elapsed_seconds) + el) > float(campaign_cap_seconds)
    )


def enforce_shared_process_deadline(now_monotonic, deadline_spec, stage="execution"):
    """Raise RuntimeError immediately if now_monotonic exceeds the shared process-start deadline."""
    now_m = float(now_monotonic)
    elapsed = now_m - float(deadline_spec["process_start_monotonic"])
    prior = float(deadline_spec["prior_elapsed_seconds"])
    cap = float(deadline_spec["campaign_cap_seconds"])
    rem = float(deadline_spec["remaining_budget_seconds"])
    hard = float(deadline_spec["hard_deadline_monotonic"])
    if now_m > hard or is_cumulative_wall_deadline_exceeded(
        elapsed,
        prior_elapsed_seconds=prior,
        campaign_cap_seconds=cap,
        remaining_budget_seconds=rem,
    ):
        raise RuntimeError(
            f"cumulative 7h hard wall deadline exceeded during {stage}: "
            f"elapsed={elapsed:.3f}s, cumulative={prior + elapsed:.3f}s > {cap:.1f}s "
            f"(remaining_budget={rem:.3f}s)"
        )
    return dict(
        elapsed_seconds=elapsed,
        cumulative_elapsed_seconds=prior + elapsed,
        remaining_seconds=hard - now_m,
    )


def read_process_thread_names(pid):
    """Read /proc/<pid>/task/<tid>/comm for a process on Linux."""
    task_dir = Path(f"/proc/{pid}/task")
    if not task_dir.is_dir():
        return ["python3"]
    names = []
    for tid_dir in sorted(task_dir.iterdir()):
        comm_file = tid_dir / "comm"
        try:
            names.append(comm_file.read_text().strip())
        except OSError:
            continue
    return names or ["python3"]


class SustainedLoad:
    def __init__(self, limit=LOAD_LIMIT):
        self.limit = limit
        self.hits = []

    def check(self, load_value, now):
        if load_value <= self.limit:
            self.hits.clear()
            return False
        if not self.hits or now - self.hits[-1] >= 5.0:
            self.hits.append(now)
        return len(self.hits) >= 3


def violations(sample):
    """Evaluate resource and thread-rule-(b) guards on a monitor sample."""
    numeric_threads = sample.get("numeric_threads", sample.get("threads", 1))
    env_bad = sample.get("numeric_env_violations", [])
    return [
        reason
        for bad, reason in [
            (numeric_threads > MAX_NUMERIC_THREADS, "numeric threads>4"),
            (bool(env_bad), "numeric thread env/torch>4"),
            (sample["disk_used_fraction"] > 0.938, "disk used>93.8%"),
            (sample["free"] < 40_000_000_000, "disk free<40GB"),
            (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (sample["cpu"] > 85.0, "host CPU>85%"),
            (sample.get("nice", 19) != 19, "nice!=19"),
        ]
        if bad
    ]


def snapshot(child=None, env=None):
    disk = psutil.disk_usage("/home/ubuntu")
    sup = psutil.Process()
    worker_procs = []
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            worker_procs = [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            worker_procs = []
    all_procs = {p.pid: p for p in ([sup] + worker_procs)}.values()
    rss = 0
    total_threads = 0
    for p in all_procs:
        try:
            rss += p.memory_info().rss
            total_threads += p.num_threads()
        except psutil.NoSuchProcess:
            pass

    target_procs = worker_procs if worker_procs else [sup]
    comm_names = []
    for p in {p.pid: p for p in target_procs}.values():
        if p.pid == LIVE_FLYWHEEL_PID:
            raise RuntimeError("refusing to inspect or touch live flywheel PID 2297929")
        comm_names.extend(read_process_thread_names(p.pid))
    classified = classify_thread_names(comm_names)
    env_bad = check_numeric_env_and_torch(env=env)
    return dict(
        load=os.getloadavg()[0],
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        rss=rss,
        threads=total_threads,
        numeric_threads=classified["numeric_threads"],
        cuda_helper_threads=classified["cuda_helper_threads"],
        numeric_env_violations=env_bad,
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        live_flywheel_pid_untouched=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    if not args.smoke and host != "xbabe1":
        raise RuntimeError(f"H205 GPU run requires xbabe1 (got {host!r})")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.exists():
        raise RuntimeError("exclusive fresh output required")

    env = os.environ.copy()
    for k in NUMERIC_ENV_KEYS:
        env[k] = env.get(k, "2")
        if int(env[k]) > MAX_NUMERIC_THREADS:
            env[k] = str(MAX_NUMERIC_THREADS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if not args.smoke:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    psutil.cpu_percent()
    time.sleep(0.2 if args.smoke else 1.0)
    initial = snapshot(env=env)
    init_violations = violations(initial)
    if init_violations:
        raise RuntimeError("prelaunch resource refusal: " + repr(init_violations))

    load_guard = SustainedLoad()
    load_guard.check(initial["load"], time.monotonic())
    started = time.monotonic()
    launch_payload = (
        json.loads(args.launch.read_text()) if args.launch.is_file() else {}
    )
    deadline_spec = compute_shared_process_deadline(
        started,
        prior_elapsed_seconds=launch_payload.get(
            "prior_attempts_elapsed_seconds", PRIOR_ATTEMPTS_ELAPSED_SECONDS
        ),
        campaign_cap_seconds=launch_payload.get(
            "campaign_wall_budget_cap_seconds", CAMPAIGN_WALL_BUDGET_CAP_SECONDS
        ),
        requested_remaining_seconds=launch_payload.get("remaining_wall_budget_seconds"),
    )
    remaining_budget = deadline_spec["remaining_budget_seconds"]
    hard_deadline = deadline_spec["hard_deadline_monotonic"]
    env["H205_PROCESS_START_MONOTONIC"] = repr(started)
    env["H205_PRIOR_ELAPSED_SECONDS"] = repr(deadline_spec["prior_elapsed_seconds"])
    env["H205_REMAINING_BUDGET_SECONDS"] = repr(remaining_budget)
    env["H205_HARD_DEADLINE_MONOTONIC"] = repr(hard_deadline)

    claim_path = args.output.parent / "attempt.claim"
    if not args.smoke and not claim_path.exists():
        with claim_path.open("x") as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    utc=time.time(),
                    output=str(args.output.resolve()),
                    launch=str(args.launch.resolve()),
                    live_flywheel_pid_untouched=LIVE_FLYWHEEL_PID,
                    prior_attempts_elapsed_seconds=deadline_spec[
                        "prior_elapsed_seconds"
                    ],
                    remaining_wall_budget_seconds=remaining_budget,
                    campaign_wall_budget_cap_seconds=deadline_spec[
                        "campaign_cap_seconds"
                    ],
                ),
                f,
            )

    command = [
        sys.executable,
        "-B",
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        command.append("--smoke")

    reasons = []
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        resources.open("x") as samples,
    ):
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            sample = snapshot(child=child, env=env)
            sample["elapsed"] = time.monotonic() - started
            sample["cumulative_elapsed"] = (
                deadline_spec["prior_elapsed_seconds"] + sample["elapsed"]
            )
            sample["remaining_budget_seconds"] = remaining_budget - sample["elapsed"]
            sample["output_bytes"] = sum(
                p.stat().st_size for p in args.output.parent.rglob("*") if p.is_file()
            )
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            reasons = violations(sample)
            if load_guard.check(sample["load"], time.monotonic()):
                reasons.append("sustained load>85%: three readings five seconds apart")
            if sample["output_bytes"] >= 500_000_000:
                reasons.append("owned output root >=500MB")
            if sample["elapsed"] > 23178.960703136865:
                reasons.append(
                    "7h cumulative hard wall deadline (23178.960703136865s remaining)"
                )
            elif (
                sample["elapsed"] > remaining_budget or time.monotonic() > hard_deadline
            ):
                reasons.append("shared process-start remaining wall deadline exceeded")
            if reasons:
                if child.pid != LIVE_FLYWHEEL_PID:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.2 if args.smoke else 1.0)
        code = child.wait()

    elapsed_final = time.monotonic() - started
    result = dict(
        wave="H205",
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        exit_code=code,
        reasons=reasons,
        elapsed_seconds=elapsed_final,
        prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
        cumulative_elapsed_seconds=deadline_spec["prior_elapsed_seconds"]
        + elapsed_final,
        remaining_wall_budget_seconds_at_start=remaining_budget,
        campaign_wall_budget_cap_seconds=deadline_spec["campaign_cap_seconds"],
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        max_numeric_threads=MAX_NUMERIC_THREADS,
        cuda_runtime_threads_exempt=True,
        live_flywheel_pid_untouched=True,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
