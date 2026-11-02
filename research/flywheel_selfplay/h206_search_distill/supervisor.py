"""One admitted H206 controller, bounded child phases and cumulative accounting."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .contracts import ObjectRef, canonical, sha
from .executor import BudgetLedger

GPU_UUID = "GPU-2857566c-23c5-293a-df02-7c66a7571ac4"
NUMERIC_ENV = {
    k: "1"
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}
GPU_PHASES = {"propose", "train", "score"}
PHASES = GPU_PHASES | {"teacher", "utility"}


def atomic_record(path, document):
    """Evidence records are exclusive and fsynced, never shared mutable output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(canonical(document))
        out.flush()
        os.fsync(out.fileno())


def current_resources(storage, process=None):
    process = process or psutil.Process()
    disk = psutil.disk_usage(storage)
    rss = 0
    for p in (process, *process.children(recursive=True)):
        try:
            rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return {
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "available_bytes": psutil.virtual_memory().available,
        "tree_rss_bytes": rss,
        "disk_used_bytes": disk.used,
        "disk_available_bytes": disk.free,
        "effective_cores": len(os.sched_getaffinity(0)),
    }


def require_resources(snapshot, *, launching_gpu=False):
    values = (snapshot["cpu_percent"], snapshot["disk_used_bytes"], snapshot["disk_available_bytes"])
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise RuntimeError("invalid H206 resource observation")
    used, free = snapshot["disk_used_bytes"], snapshot["disk_available_bytes"]
    if (
        snapshot["nice"] != 19
        or not 0 <= snapshot["cpu_percent"] <= 85
        or snapshot["available_bytes"] < 16 * 2**30
        or snapshot["tree_rss_bytes"] >= 16 * 2**30
        or snapshot["effective_cores"] < 1
        or min(used, free) < 0
        or used + free <= 0
        or used / (used + free) >= 0.938
    ):
        raise RuntimeError("H206 running resource guard")
    if launching_gpu and (
        used / (used + free) > 0.88
        or free < 50_000_000_000
        or (used + 40_000_000_000) / (used + free) >= 0.938
    ):
        raise RuntimeError("H206 pre-cache GPU disk admission")


def actual_gpu():
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [tuple(part.strip() for part in line.split(",")) for line in gpu.stdout.splitlines()]
    target = [r for r in rows if r[0] == GPU_UUID]
    if len(target) != 1 or "RTX 3090" not in target[0][1]:
        raise RuntimeError("assigned xbabe3 RTX3090 identity unavailable")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    active = [line for line in apps.stdout.splitlines() if line.split(",")[0].strip() == GPU_UUID]
    return {
        "uuid": GPU_UUID,
        "utilization": int(target[0][2]),
        "memory_mib": int(target[0][3]),
        "compute_processes": active,
    }


def verify_merged_source(repo, source):
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != source["merged_head"]:
        raise ValueError("worker is not on the admitted merged head")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("scientific worker source is dirty")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", head, "origin/main"], check=True)
    if not source.get("files"):
        raise ValueError("merged source file closure required")
    for name, expected in source["files"].items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or sha(path.read_bytes()) != expected:
            raise ValueError("merged scientific source identity mismatch")
    return head


def verify_qualifications(admission, store):
    required = {
        "model_qualification": "h206-actual-model-qualified-v1",
        "native91_qualification": "h206-native91-qualified-v1",
        "input_qualification": "h206-approved-study-inputs-v1",
        "queue_authority": "h206-owner-queue-release-v1",
    }
    for key, schema in required.items():
        doc = json.loads(ObjectRef(**admission[key]).read(store))
        if doc.get("schema") != schema or doc.get("status") != "PASS" or doc.get("fixture") is not False:
            raise ValueError("genuine H206 qualification missing: " + key)
        if doc.get("source_files") != admission["source"]["files"]:
            raise ValueError("qualification source closure differs: " + key)
        if doc.get("study_sha256") != admission["study_sha256"]:
            raise ValueError("qualification study binding differs: " + key)
        if key == "input_qualification" and (
            doc.get("study_input_payload_sha256") != admission["study_input_payload_sha256"]
            or doc.get("fold_manifest_sha256") != admission["fold_manifest_sha256"]
        ):
            raise ValueError("qualified input/fold payload differs from actual study")
        if key in ("model_qualification", "native91_qualification") and (
            doc.get("bindings_sha256") != admission["bindings_sha256"]
        ):
            raise ValueError("qualified actual model bindings differ")
    return True


def process_cpu():
    import resource

    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


class ReservedPhaseBudget(BudgetLedger):
    """Child stage breakdown uses its parent's existing reservation, never a copy."""

    def __init__(self, path, token, stage):
        super().__init__(path)

        def verify(state):
            r = state["reservations"].get(token)
            if r is None or r.get("stage") != stage:
                raise ValueError("missing parent phase reservation")

        self.mutate(verify)  # Also checks original caps and reconciled usage.
        self.active = True

    def operation(self, *args, **kwargs):
        raise RuntimeError("phase child cannot create another cumulative reservation")


class PhaseSupervisor:
    def __init__(self, repo, output, cache, ledger, store, admission):
        self.repo, self.output, self.cache = (
            Path(repo).resolve(),
            Path(output).resolve(),
            Path(cache).resolve(),
        )
        if self.output.is_relative_to(self.repo) or self.output.is_relative_to("/tmp"):
            raise ValueError("supervisor evidence must be outside checkout and /tmp")
        owned = Path("/home/ubuntu/jope_plan/evidence/H206").resolve()
        if not self.output.is_relative_to(owned) or not self.cache.is_relative_to(owned):
            raise ValueError("H206-owned runtime storage namespace required")
        for name in ("native_cache", "native_scratch", "model_cache"):
            if not Path(admission[name]).resolve().is_relative_to(self.cache):
                raise ValueError("all runtime caches/scratch must share the40GB cache bound")
        if self.output.is_relative_to(self.cache) or self.cache.is_relative_to(self.output):
            raise ValueError("cache and evidence roots must be disjoint for exact accounting")
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget, self.store, self.admission = BudgetLedger(ledger), store, admission
        self.context = None
        self.active_child = None

    def validate(self):
        # No GPU probe/claim occurs before real source/input/budget/queue checks.
        verify_merged_source(self.repo, self.admission["source"])
        if self.admission["host"] != "xbabe3" or self.admission["gpu_uuid"] != GPU_UUID:
            raise ValueError("H206 assigned host/device mismatch")
        if self.admission["wave"] != "H206" or self.admission["attempt"] != 1:
            raise ValueError("original single H206 attempt required")
        if not isinstance(self.admission["deadline_epoch"], (int, float)) or (
            not math.isfinite(self.admission["deadline_epoch"])
            or time.time() >= self.admission["deadline_epoch"]
        ):
            raise RuntimeError("original campaign deadline absent/expired")
        if self.admission.get("sealed_access") is not False:
            raise ValueError("sealed confirmation is not part of this worker")
        verify_qualifications(self.admission, self.store)

        def accounting(state):
            if state.get("cpu_used_seconds") is None or state.get("execution_admitted") is not True:
                raise ValueError("canonical campaign accounting/admission still held")
            for kind in ("cpu", "gpu"):
                value = state[kind + "_used_seconds"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid actual cumulative usage")

        self.budget.mutate(accounting)
        require_resources(current_resources(self.cache))
        return True

    @contextlib.contextmanager
    def controller(self):
        before, began = process_cpu(), time.monotonic()
        self.validate()
        lock = (self.output / "controller.lock").open("a")
        overhead_token = "controller-" + str(os.getpid()) + "-" + str(time.time_ns())
        initial_entries = None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reserve = self.admission["controller_cpu_reserve_seconds"]
            if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
                raise ValueError("bounded original controller CPU reservation required")

            def claim(state):
                nonlocal initial_entries
                if state.get("scientific_attempt") is not None:
                    raise RuntimeError("original H206 attempt already claimed; no retry/reset")
                reserved = sum(r["cpu"] for r in state["reservations"].values())
                if state["cpu_used_seconds"] + reserved + reserve > state["cpu_cap_seconds"]:
                    raise RuntimeError("controller cumulative CPU budget exhausted")
                initial_entries = len(state["entries"])
                state["reservations"][overhead_token] = {
                    "cpu": reserve,
                    "gpu": 0,
                    "stage": "controller-overhead",
                }
                state["scientific_attempt"] = {
                    "attempt": 1,
                    "run_id": self.admission["run_id"],
                    "supervisor_pid": os.getpid(),
                    "admission_sha256": sha(canonical(self.admission)),
                    "deadline_epoch": self.admission["deadline_epoch"],
                    "terminal": False,
                }

            self.budget.mutate(claim)
            atomic_record(self.output / "attempt.json", self.admission)
            self.context = {"started_epoch": time.time(), "attempt": 1}
            prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

            def terminal_signal(signum, frame):
                raise RuntimeError("original controller terminal signal " + str(signum))

            for sig in prior_handlers:
                signal.signal(sig, terminal_signal)
            try:
                yield self
            finally:
                for sig, handler in prior_handlers.items():
                    signal.signal(sig, handler)
        finally:
            self.context = None
            if initial_entries is not None:
                total_cpu = process_cpu() - before

                def finish(state):
                    phase_cpu = sum(e["cpu_seconds"] for e in state["entries"][initial_entries:])
                    overhead_cpu = max(0.0, total_cpu - phase_cpu)
                    reservation = state["reservations"].pop(overhead_token)
                    state["cpu_used_seconds"] += overhead_cpu
                    state["exhausted"] = bool(
                        state.get("exhausted")
                        or overhead_cpu > reservation["cpu"]
                        or state["cpu_used_seconds"] > state["cpu_cap_seconds"]
                    )
                    state["entries"].append(
                        {
                            "stage": "controller-overhead",
                            "cpu_seconds": overhead_cpu,
                            "gpu_seconds": 0.0,
                            "wall_seconds": time.monotonic() - began,
                            "accounting": "CPU excludes all already charged phase CPU",
                        }
                    )
                    state["scientific_attempt"]["terminal"] = True

                self.budget.mutate(finish)
            lock.close()

    def _round_limit(self, round_index, phase, reserve):
        def check(state):
            prefix = f"round-{round_index}/"
            entries = [e for e in state["entries"] if e["stage"].startswith(prefix)]
            total = sum(e["gpu_seconds"] for e in entries)
            nonfit = sum(e["gpu_seconds"] for e in entries if not e["stage"].endswith("/train"))
            fits = total - nonfit
            if (
                total + reserve > 6 * 3600
                or (phase == "train" and fits + reserve > 5 * 3600)
                or (phase != "train" and nonfit + reserve > 3600)
            ):
                raise RuntimeError("original per-round GPU reservation exhausted")

        self.budget.mutate(check)

    @contextlib.contextmanager
    def _device(self, gpu):
        if not gpu:
            yield
            return
        path = Path(self.admission["gpu_lease_path"]).resolve()
        expected = Path("/home/ubuntu/jope_plan/gpu-leases/xbabe3") / (GPU_UUID + ".lock")
        if path != expected:
            raise ValueError("shared assigned GPU lease path required")
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            Path("/home/ubuntu/jope_plan/qualification.lock").open("a") as qualification,
            path.open("a") as lock,
        ):
            fcntl.flock(qualification, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_resources(current_resources(self.cache), launching_gpu=True)
            snapshot = actual_gpu()
            if snapshot["compute_processes"] or snapshot["utilization"] != 0 or snapshot["memory_mib"] > 16:
                raise RuntimeError("assigned GPU is occupied; no preemption")
            atomic_record(self.output / ("gpu-claim-" + str(time.time_ns()) + ".json"), snapshot)
            self._comms("CLAIM GPU", snapshot)
            try:
                yield
            finally:
                self._comms("RELEASE GPU", {"uuid": GPU_UUID, "epoch": time.time()})
                atomic_record(
                    self.output / ("gpu-release-" + str(time.time_ns()) + ".json"),
                    {"uuid": GPU_UUID, "epoch": time.time(), "child_alive": False},
                )

    def _comms(self, action, observation):
        path = Path(self.admission["comms_path"])
        with Path(str(path) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with path.open("a") as out:
                out.write(
                    "\n[H206 gpu_fused_redteam_codex] "
                    + action
                    + " "
                    + json.dumps(observation, sort_keys=True)
                    + " run="
                    + self.admission["run_id"]
                    + "\n"
                )
                out.flush()
                os.fsync(out.fileno())

    def phase(self, task, *, cpu_reserve, seconds_cap):
        if self.context is None:
            raise RuntimeError("one validated active controller required")
        phase, round_index = task["phase"], task["round"]
        if phase not in PHASES or round_index not in (1, 2, 3):
            raise ValueError("registered H206 phase/round required")
        if not 0 < seconds_cap or (phase == "train" and seconds_cap > 300):
            raise ValueError("fixed fit/phase deadline required")
        gpu = phase in GPU_PHASES
        if gpu:
            self._round_limit(round_index, phase, seconds_cap)
        phase_label = phase + ("-" + task["role"] if phase in ("score", "utility") else "")
        stage = f"round-{round_index}/{task['cell_id']}/{phase_label}"
        folder = self.output / stage
        folder.mkdir(parents=True, exist_ok=True)
        task = dict(
            task,
            admission=self.admission,
            ledger=str(self.budget.path),
            stage=stage,
            supervisor_pid=os.getpid(),
        )
        gpu_reserve = seconds_cap if gpu else 0
        with self.budget.operation(stage, cpu_reserve=cpu_reserve, gpu_reserve=gpu_reserve) as token:
            task["reservation"] = token
            atomic_record(folder / "task.json", task)
            with self._device(gpu):
                env = (
                    os.environ
                    | NUMERIC_ENV
                    | {"CUDA_VISIBLE_DEVICES": GPU_UUID if gpu else "", "PYTHONDONTWRITEBYTECODE": "1"}
                )
                env["PYTHONPATH"] = str(self.repo / "python/src") + ":" + str(self.repo)
                cmd = [
                    sys.executable,
                    "-B",
                    "-m",
                    "research.flywheel_selfplay.h206_search_distill.study",
                    "--worker",
                    str(folder / "task.json"),
                    "--task-sha256",
                    sha(canonical(task)),
                ]
                began, reason = time.monotonic(), None
                with (folder / "worker.log").open("xb") as log:
                    child = subprocess.Popen(
                        cmd,
                        cwd=self.repo,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.active_child = child.pid
                    try:
                        while child.poll() is None:
                            if time.monotonic() - began >= seconds_cap:
                                raise RuntimeError("fixed phase/fit cap reached")
                            if time.time() >= self.admission["deadline_epoch"]:
                                raise RuntimeError("original campaign deadline reached")
                            require_resources(current_resources(self.cache))
                            runtime_path = folder / "runtime.json"
                            if runtime_path.exists():
                                runtime = json.loads(runtime_path.read_text())
                                upper = runtime["numeric"]["numeric_threads_upper_bound"]
                                if runtime["pid"] != child.pid or not 0 < upper <= 4:
                                    raise RuntimeError("actual child numeric identity/bound")
                            if gpu:
                                visible = actual_gpu()
                                own = {child.pid}
                                try:
                                    own |= {p.pid for p in psutil.Process(child.pid).children(recursive=True)}
                                except psutil.NoSuchProcess:
                                    pass
                                if any(
                                    int(line.split(",")[1]) not in own
                                    for line in visible["compute_processes"]
                                ):
                                    raise RuntimeError("foreign GPU worker appeared; stop own phase")
                            cache_size = sum(
                                p.stat().st_size
                                for root in (self.cache, self.output)
                                for p in root.rglob("*")
                                if p.is_file()
                            )
                            if cache_size > 40_000_000_000:
                                raise RuntimeError("shared H206 cache cap")
                            time.sleep(0.5)
                    except BaseException as exc:
                        reason = repr(exc)
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                    finally:
                        child.wait()
                        self.active_child = None
                        atomic_record(
                            folder / "terminal.json",
                            {
                                "pid": child.pid,
                                "exit_code": child.returncode,
                                "reason": reason,
                                "wall_seconds": time.monotonic() - began,
                                "partial_evidence": "worker writes each update/proof before advancing",
                            },
                        )
                    if reason is not None or child.returncode != 0:
                        raise RuntimeError("H206 INCOMPLETE, no retry: " + str(reason or child.returncode))
                if not (folder / "runtime.json").is_file():
                    raise RuntimeError("phase never supplied actual numeric report")
                result = json.loads((folder / "result.json").read_text())
                ref = ObjectRef(**result["result"])
                ref.read(self.store)
                return ref


class GPUWorkerPermit:
    """Positive GPU path: actual source/qualification/parent reservation and device."""

    def __init__(self, task, repo, store):
        self.task, self.repo, self.store = task, Path(repo), store
        self.token = task["reservation"]
        self.device = None

    def validate(self):
        task = self.task
        if task["phase"] not in GPU_PHASES or os.getppid() != task["supervisor_pid"]:
            raise ValueError("worker is not the admitted GPU phase child")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise ValueError("GPU phase visibility differs from assigned device")
        verify_merged_source(self.repo, task["admission"]["source"])
        verify_qualifications(task["admission"], self.store)
        budget = BudgetLedger(task["ledger"])

        def verify(state):
            r = state["reservations"].get(self.token)
            attempt = state.get("scientific_attempt", {})
            if (
                r is None
                or r.get("stage") != task["stage"]
                or r["gpu"] <= 0
                or attempt.get("supervisor_pid") != task["supervisor_pid"]
                or attempt.get("admission_sha256") != sha(canonical(task["admission"]))
                or attempt.get("terminal") is not False
            ):
                raise ValueError("actual parent GPU reservation/attempt required")

        budget.mutate(verify)
        if time.time() >= task["admission"]["deadline_epoch"]:
            raise RuntimeError("original campaign deadline expired")
        # Only after merged source, real qualification and the actual cumulative
        # parent reservation exist may this child initialize any CUDA context.
        import torch

        if torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("actual assigned CUDA device unavailable")
        self.device = torch.device("cuda:0")
        return self
