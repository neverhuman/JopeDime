"""H205 resource and thread-rule-(b) monitor on xbabe1 RTX 4090.

Enforces:
- Numeric threads <= 4 (OMP/MKL/OpenBLAS/NUMEXPR/RAYON/torch intra-op <= 4).
- CUDA runtime/driver helper threads exempt from the numeric cap without disabling resource monitoring.
- CPU <= 85%, sustained 1-min load <= 85% of online cores, nice == 19, RSS < 6 GB,
  MemAvailable >= 16 GiB, disk used <= 93.8% and free >= 40 GB.
- Never stops, signals, or renices foreign processes (including live flywheel PID 2297929 on xbabe1).
"""

import time

PROCESS_START_MONOTONIC = time.monotonic()

import argparse  # noqa: E402 -- capture startup before imports
import fcntl  # noqa: E402 -- capture startup before imports
import hashlib  # noqa: E402 -- capture startup before imports
import json  # noqa: E402 -- capture startup before imports
import math  # noqa: E402 -- capture startup before imports
import os  # noqa: E402 -- capture startup before imports
from pathlib import Path  # noqa: E402 -- capture startup before imports
import signal  # noqa: E402 -- capture startup before imports
import socket  # noqa: E402 -- capture startup before imports
import subprocess  # noqa: E402 -- capture startup before imports
import sys  # noqa: E402 -- capture startup before imports
import tempfile  # noqa: E402 -- capture startup before imports


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
    if not math.isfinite(prior) or not math.isfinite(cap):
        raise RuntimeError("nonfinite cumulative campaign budget")
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
        if not math.isfinite(req) or req <= 0.0:
            raise RuntimeError(
                "prelaunch cumulative budget refusal: requested remaining budget must be finite and positive"
            )
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
    if not math.isfinite(start_mono) or start_mono < 0.0:
        raise RuntimeError("invalid process start clock")
    remaining = compute_remaining_wall_budget_seconds(
        prior_elapsed_seconds=prior_elapsed_seconds,
        campaign_cap_seconds=campaign_cap_seconds,
        requested_remaining_seconds=requested_remaining_seconds,
    )
    hard_deadline = start_mono + remaining
    if env_hard_deadline_monotonic is not None:
        inherited = float(env_hard_deadline_monotonic)
        if not math.isfinite(inherited):
            raise RuntimeError("nonfinite inherited hard deadline")
        hard_deadline = min(hard_deadline, inherited)
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
    if (
        not all(
            math.isfinite(value) for value in (now_m, elapsed, prior, cap, rem, hard)
        )
        or elapsed < 0.0
        or prior < PRIOR_ATTEMPTS_ELAPSED_SECONDS - 1e-9
        or cap <= 0.0
        or cap > CAMPAIGN_WALL_BUDGET_CAP_SECONDS + 1e-9
        or rem <= 0.0
        or rem > cap - prior + 1e-9
        or hard > float(deadline_spec["process_start_monotonic"]) + rem + 1e-9
    ):
        raise RuntimeError("invalid shared process deadline/budget identity")
    if now_m >= hard or is_cumulative_wall_deadline_exceeded(
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


def remaining_subprocess_timeout(now_monotonic, deadline_spec, timeout_cap_seconds):
    """Use the positive remaining shared window, retaining the smaller native/AWS cap."""
    if deadline_spec is None:
        raise RuntimeError("native/AWS subprocess requires the shared process deadline")
    cap = float(timeout_cap_seconds)
    if not math.isfinite(cap) or cap <= 0.0:
        raise RuntimeError("subprocess timeout cap must be finite and positive")
    state = enforce_shared_process_deadline(
        now_monotonic, deadline_spec, stage="before_subprocess"
    )
    return min(cap, state["remaining_seconds"])


_accounting_floor_seconds = 2021.039296863135
_accounting_limit_seconds = 25200.0
_accounting_cutoff = (
    "caller monotonic sample before this terminal ledger commit; excludes its "
    "serialization/fsync/replace/directory-fsync and subsequent process exit"
)


def _accounting_number(value, label):
    if type(value) not in (int, float):
        raise RuntimeError(f"invalid accounting {label}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RuntimeError(f"invalid accounting {label}") from exc
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"invalid accounting {label}")
    return number


def _accounting_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"invalid accounting {label}")
    return value


def _accounting_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("duplicate accounting JSON key")
        result[key] = value
    return result


def _accounting_load(raw, require_reconciled=True):
    try:
        state = json.loads(raw, object_pairs_hook=_accounting_object)
        json.dumps(state, allow_nan=False)
        if (
            state["schema"] != "h205-campaign-accounting/v1"
            or type(state["history_complete"]) is not bool
        ):
            raise RuntimeError("invalid accounting schema or history marker")
        if require_reconciled and state["history_complete"] is not True:
            raise RuntimeError("accounting history is not reconciled")
        if state["history_complete"] and state.get("unreconciled_tail") is not None:
            raise RuntimeError("accounting tail requires owner reconciliation")
        baseline = _accounting_number(state["baseline_elapsed_seconds"], "baseline")
        cap = _accounting_number(state["campaign_cap_seconds"], "cap")
        if (
            baseline < _accounting_floor_seconds
            or not 0 < cap <= _accounting_limit_seconds
        ):
            raise RuntimeError("accounting floor reduced or cap increased")
        if (
            not isinstance(state["source_evidence"], (list, dict))
            or not state["source_evidence"]
        ):
            raise RuntimeError("accounting source evidence required")
        completed = state["completed"]
        if not isinstance(completed, dict):
            raise RuntimeError("invalid accounting completed map")
        elapsed = []
        for key, record in completed.items():
            _accounting_text(key, "completed id")
            _accounting_text(record["status"], "completed status")
            elapsed.append(
                _accounting_number(record["elapsed_seconds"], "completed elapsed")
            )
        pending = state["pending"]
        if pending is not None:
            if not isinstance(pending, dict) or set(pending) != {
                "invocation_id",
                "process_start_monotonic",
            }:
                raise RuntimeError("unknown pending accounting state")
            _accounting_text(pending["invocation_id"], "pending id")
            _accounting_number(pending["process_start_monotonic"], "pending start")
            if pending["invocation_id"] in completed:
                raise RuntimeError("pending accounting id is already completed")
        return (
            state,
            _accounting_number(baseline + sum(elapsed), "cumulative elapsed"),
            cap,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("corrupt accounting ledger") from exc


def _accounting_save(path, state):
    payload = (
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".next", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _accounting_update(path, change, expected_sha256=None, require_reconciled=True):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("existing reconciled accounting ledger required")
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = path.read_bytes()
        if (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise RuntimeError("stale accounting ledger digest")
        state, prior, cap = _accounting_load(raw, require_reconciled=require_reconciled)
        changed, result = change(state, prior, cap)
        if changed:
            _accounting_save(path, state)
        return result


def begin_accounted_invocation(
    path, expected_sha256, invocation_id, process_start_monotonic
):
    _accounting_text(invocation_id, "invocation id")
    start = _accounting_number(process_start_monotonic, "process start")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise RuntimeError("expected accounting SHA256 required")

    def _accounting_change(state, prior, cap):
        if state["pending"] is not None:
            raise RuntimeError("pending invocation requires owner reconciliation")
        if invocation_id in state["completed"]:
            raise RuntimeError("accounting invocation id already completed")
        if prior >= cap:
            raise RuntimeError("cumulative campaign budget exhausted")
        state["pending"] = {
            "invocation_id": invocation_id,
            "process_start_monotonic": start,
        }
        return True, {
            "prior_elapsed_seconds": prior,
            "campaign_cap_seconds": cap,
            "invocation_id": invocation_id,
        }

    return _accounting_update(path, _accounting_change, expected_sha256)


def finish_accounted_invocation(path, invocation_id, now_monotonic, status):
    _accounting_text(invocation_id, "invocation id")
    now = _accounting_number(now_monotonic, "finish sample")
    _accounting_text(status, "finish status")

    def _accounting_change(state, prior, cap):
        if invocation_id in state["completed"]:
            return False, state["completed"][invocation_id]
        if state["history_complete"] is not True:
            raise RuntimeError("accounting history is not reconciled")
        pending = state["pending"]
        if pending is None or pending["invocation_id"] != invocation_id:
            raise RuntimeError("accounting pending invocation mismatch")
        elapsed = _accounting_number(
            now - pending["process_start_monotonic"], "invocation elapsed"
        )
        record = {
            "elapsed_seconds": elapsed,
            "status": status,
            "accounting_cutoff": _accounting_cutoff,
        }
        state["completed"][invocation_id], state["pending"] = record, None
        state["history_complete"] = False
        state["unreconciled_tail"] = {
            "invocation_id": invocation_id,
            "scope": "after accounted cutoff through supervisor exit",
            "seconds": None,
        }
        return True, record

    return _accounting_update(path, _accounting_change, require_reconciled=False)


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


def _supervise_accounted_worker(args, launch_payload, deadline_spec, invocation_id):
    global psutil
    import psutil

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
    started = deadline_spec["process_start_monotonic"]
    remaining_budget = deadline_spec["remaining_budget_seconds"]
    hard_deadline = deadline_spec["hard_deadline_monotonic"]
    env["H205_ACCOUNTING_INVOCATION_ID"] = invocation_id
    env["H205_ACCOUNTING_PARENT_PID"] = str(os.getpid())
    env["H205_ACCOUNTING_BOOT_ID"] = (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    )
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

    worker_launch = dict(
        launch_payload,
        prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
        remaining_wall_budget_seconds=remaining_budget,
        campaign_wall_budget_cap_seconds=deadline_spec["campaign_cap_seconds"],
        accounting_invocation_id=invocation_id,
    )
    worker_launch_path = args.output.with_suffix(".accounted-launch.json")
    with worker_launch_path.open("x") as stream:
        json.dump(worker_launch, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    command = [
        sys.executable,
        "-B",
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(worker_launch_path),
    ]
    if args.smoke:
        command.append("--smoke")
    if args.construct_single_regression_context:
        command.append("--construct-single-regression-context")

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
        try:
            while child.poll() is None:
                sample = snapshot(child=child, env=env)
                sample["elapsed"] = time.monotonic() - started
                sample["cumulative_elapsed"] = (
                    deadline_spec["prior_elapsed_seconds"] + sample["elapsed"]
                )
                sample["remaining_budget_seconds"] = (
                    remaining_budget - sample["elapsed"]
                )
                sample["output_bytes"] = sum(
                    p.stat().st_size
                    for p in args.output.parent.rglob("*")
                    if p.is_file()
                )
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                reasons = violations(sample)
                if load_guard.check(sample["load"], time.monotonic()):
                    reasons.append(
                        "sustained load>85%: three readings five seconds apart"
                    )
                if sample["output_bytes"] >= 500_000_000:
                    reasons.append("owned output root >=500MB")
                if sample["elapsed"] > 23178.960703136865:
                    reasons.append(
                        "7h cumulative hard wall deadline (23178.960703136865s remaining)"
                    )
                elif (
                    sample["elapsed"] > remaining_budget
                    or time.monotonic() > hard_deadline
                ):
                    reasons.append(
                        "shared process-start remaining wall deadline exceeded"
                    )
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
        except BaseException:
            if child.poll() is None and child.pid != LIVE_FLYWHEEL_PID:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise

    # A child can finish after expiry between the final poll and wait().
    try:
        enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="after_worker_exit_and_log_close"
        )
    except RuntimeError as exc:
        reasons.append(str(exc))
    worker_summary = None
    if args.construct_single_regression_context:
        summary_path = args.output / "single_regression_context_summary.json"
        if summary_path.is_file():
            worker_summary = json.loads(summary_path.read_text())
        ready = (
            isinstance(worker_summary, dict)
            and worker_summary.get("status") == "PENDING_FINAL_ACCOUNTING"
            and worker_summary.get("operation_succeeded") is True
            and worker_summary.get("seam_qualified") is False
            and worker_summary.get("accounting_invocation_id") == invocation_id
        )
        if not ready:
            reasons.append(
                "worker did not complete the same accounted helper operation"
            )
    elapsed_final = time.monotonic() - started
    result = dict(
        wave="H205",
        status="PENDING_FINAL_ACCOUNTING"
        if code == 0 and not reasons
        else "INCOMPLETE",
        operation_succeeded=code == 0 and not reasons,
        seam_qualified=False,
        accounting_invocation_id=invocation_id,
        worker_summary=worker_summary,
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
    return result


def run_accounted_observation(launch, process_start_monotonic, observe):
    """Charge the sole supervisor once; unknown terminal bookkeeping prevents a new window.

    `observe` includes preflight, worker imports/work/finalization, exit observation,
    owned log closure and the preliminary resource receipt. It never launches a
    second controller. A separate original-owner exit/cost reconciliation is
    required before this pending operation may qualify or another invocation starts.
    """
    ledger_path = launch.get("campaign_ledger_path")
    if not ledger_path:
        raise RuntimeError(
            "existing reconciled campaign_ledger_path required; no budget reset"
        )
    invocation_id = launch.get("accounting_invocation_id")
    admission = begin_accounted_invocation(
        ledger_path,
        launch.get("campaign_ledger_sha256"),
        invocation_id,
        process_start_monotonic,
    )
    terminal_status = "INCOMPLETE"
    result = None
    try:
        deadline_spec = compute_shared_process_deadline(
            process_start_monotonic,
            prior_elapsed_seconds=admission["prior_elapsed_seconds"],
            campaign_cap_seconds=admission["campaign_cap_seconds"],
            requested_remaining_seconds=launch.get("remaining_wall_budget_seconds"),
        )
        enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="after_accounting_admission"
        )
        result = observe(deadline_spec, invocation_id)
        enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="after_worker_finalization"
        )
        if result.get("operation_succeeded") is True and result.get("exit_code") == 0:
            terminal_status = "OPERATION_SUCCEEDED_PENDING_FINAL_RECONCILIATION"
    finally:
        accounting = finish_accounted_invocation(
            ledger_path, invocation_id, time.monotonic(), terminal_status
        )
    # This catches a ledger commit that itself crosses expiry. The charged cutoff
    # and unknown tail stay durable; it cannot turn into a qualified PASS.
    enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_accounting_commit"
    )
    return dict(
        result,
        status="PENDING_FINAL_ACCOUNTING"
        if terminal_status.startswith("OPERATION_SUCCEEDED")
        else "INCOMPLETE",
        seam_qualified=False,
        final_accounting=accounting,
        remaining_cost_tail_seconds=None,
        owner_reconciliation_required=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--construct-single-regression-context", action="store_true")
    args = ap.parse_args()
    try:
        if args.smoke and args.construct_single_regression_context:
            raise RuntimeError(
                "smoke and construction are separate accounted invocations"
            )
        launch = json.loads(args.launch.read_text())
        result = run_accounted_observation(
            launch,
            PROCESS_START_MONOTONIC,
            lambda deadline, identity: _supervise_accounted_worker(
                args, launch, deadline, identity
            ),
        )
    except BaseException as exc:
        # Bootstrap may fail before a ledger can be identified. Preserve a refusal
        # cost lower bound, never pretend it was charged or permit an automatic retry.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        refusal = dict(
            wave="H205",
            status="INCOMPLETE_ADMISSION_OR_INVOCATION",
            seam_qualified=False,
            captured_start_monotonic=PROCESS_START_MONOTONIC,
            observed_elapsed_lower_bound_seconds=time.monotonic()
            - PROCESS_START_MONOTONIC,
            terminal_tail_seconds=None,
            history_reconciliation_required=True,
            error_type=type(exc).__name__,
            budget_reset=False,
            existing_ledger_must_be_inspected_to_avoid_double_charge=True,
        )
        with args.output.with_suffix(".admission-refusal.json").open("x") as stream:
            json.dump(refusal, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        raise
    # This is final supervisor bookkeeping, explicitly outside the stored cutoff.
    # Its cost is UNKNOWN until the original owner reconciles the observed exit.
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["operation_succeeded"] else 1)


if __name__ == "__main__":
    main()
