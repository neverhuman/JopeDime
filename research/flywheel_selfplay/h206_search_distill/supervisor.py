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

from .contracts import ObjectRef, canonical, require_hash, sha
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


def scientific_source_files(repo):
    """Complete declared H206 executable/protocol/dependency closure, not a subset."""
    repo = Path(repo).resolve()
    folder = Path("research/flywheel_selfplay/h206_search_distill")
    manifest_path = repo / folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"contracts.py", "native.py", "student.py", "executor.py", "model.py", "study.py",
                "supervisor.py", "PREREG.md", "bindings.json", "native91_inputs.json"}
    if not required <= set(manifest["source_sha256"]) or not manifest.get("dependency_source_sha256"):
        raise ValueError("complete scientific source manifest required")
    files = {str(folder / name): value for name, value in manifest["source_sha256"].items()}
    files.update(manifest["dependency_source_sha256"])
    files[str(folder / "manifest.json")] = sha(manifest_path.read_bytes())
    for name, expected in files.items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or sha(path.read_bytes()) != expected:
            raise ValueError("declared scientific source/dependency bytes changed")
    return files


def verify_merged_source(repo, source):
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != source["merged_head"]:
        raise ValueError("worker is not on the admitted merged head")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("scientific worker source is dirty")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", head, "origin/main"], check=True)
    if source.get("files") != scientific_source_files(repo):
        raise ValueError("complete merged scientific source closure required; subsets refuse")
    for name, expected in source["files"].items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or sha(path.read_bytes()) != expected:
            raise ValueError("merged scientific source identity mismatch")
    return head


def runtime_qualification_plan():
    """Fixed non-scored protocol check using the existing four first-fold cells."""
    return {
        "schema": "h206-runtime-qualification-plan-v1",
        "seed": 206101,
        "initial_state_sha256": "dbe16fbce2060a0493b4129c5e5579414a874e1fde3d156ad6c2b44339313a2d",
        "fold": 0,
        "arms": {"CTRL": 1, "ACC4": 16, "ACC16": 64, "OC15": 256},
        "rounds": 3,
        "updates_per_round": 600,
        "batch_size": 32,
        "roles": ["FIT"],
        "scientific_endpoint": False,
        "checkpoints_eligible_for_study": False,
        "budget": "original cumulative H206 caps; no extra allowance",
    }


def execution_purpose(admission):
    purpose = admission.get("purpose", "study")
    if purpose not in ("study", "runtime_qualification"):
        raise ValueError("explicit existing H206 execution purpose required")
    return purpose


def validate_phase_purpose(task, admission):
    if execution_purpose(admission) == "runtime_qualification" and (
        task.get("phase") not in ("propose", "teacher", "train")
        or task.get("seed") != 206101
        or task.get("fold") != 0
        or runtime_qualification_plan()["arms"].get(task.get("arm")) != task.get("k")
    ):
        raise ValueError("qualification is fixed FIT-only first-fold four-arm protocol, never B/A")


def original_deadline_guard(admission, phase_deadline=None):
    deadlines = [admission["deadline_epoch"]]
    if phase_deadline is not None:
        deadlines.append(phase_deadline)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in deadlines):
        raise ValueError("finite original deadlines required")
    deadline = min(deadlines)
    monotonic_deadline = time.monotonic() + max(0.0, deadline - time.time())

    def check():
        if time.time() >= deadline or time.monotonic() >= monotonic_deadline:
            raise RuntimeError("original campaign/phase deadline expired")

    return check


def validate_runtime_proofs(document):
    plan = runtime_qualification_plan()
    if (
        document.get("fixture") is not False
        or document.get("qualification_plan") != plan
        or document.get("b_a_payload_reads") != 0
        or document.get("initial_state_sha256") != plan["initial_state_sha256"]
        or document.get("checkpoints_eligible_for_study") is not False
    ):
        raise ValueError("actual FIT-only registered runtime qualification required")
    proofs = document.get("round_proofs", [])
    expected = {(arm, round_index) for arm in plan["arms"] for round_index in (1, 2, 3)}
    seen = set()
    for item in proofs:
        key = (item["arm"], item["round"])
        if key not in expected or key in seen:
            raise ValueError("missing/duplicate runtime qualification cell round")
        seen.add(key)
        proof = item["proof"]
        require_hash(item["state_sha256"])
        if proof.get("state_sha256") != item["state_sha256"]:
            raise ValueError("actual terminal CUDA state/checkpoint identity missing")
        gradients = [proof["gradient_squared_sums"].get(k) for k in ("dope", "residual")]
        changed = proof["changed_parameters"]
        if (
            item["updates"] != 600 * item["round"]
            or proof["updates_this_round"] != 600
            or proof["examples_this_round"] != 600 * 32
            or item["durable_steps"] != 600
            or proof["forward_changed"] is not True
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in gradients)
            or not any("dope_model." in name for name in changed)
            or not any(name.startswith("residual.") for name in changed)
        ):
            raise ValueError("actual per-round CUDA update/gradient/state proof missing")
        ObjectRef(**item["round_object"])
    if seen != expected:
        raise ValueError("all twelve runtime qualification round proofs required")
    phases = document.get("phase_executions", [])
    phase_keys = set()
    for item in phases:
        key = (item["arm"], item["round"], item["phase"])
        if key in phase_keys or key[:2] not in expected or key[2] not in ("propose", "teacher", "train"):
            raise ValueError("runtime phase coverage/ordering mismatch")
        phase_keys.add(key)
        ObjectRef(**item["execution"])
    if len(phase_keys) != 36:
        raise ValueError("all thirty-six actual supervised phases required")
    return True


def verify_qualifications(admission, store):
    purpose = execution_purpose(admission)
    required = {
        "model_qualification": "h206-actual-model-qualified-v1",
        "native91_qualification": "h206-native91-qualified-v1",
        "input_qualification": "h206-approved-study-inputs-v1",
        "queue_authority": "h206-owner-queue-release-v1",
    }
    if purpose == "study":
        required["cuda_runtime_qualification"] = "h206-cuda-runtime-qualified-v1"
    for key, schema in required.items():
        if key not in admission:
            raise ValueError("genuine H206 qualification missing: " + key)
        doc = json.loads(ObjectRef(**admission[key]).read(store))
        if doc.get("schema") != schema or doc.get("status") != "PASS" or doc.get("fixture") is not False:
            raise ValueError("genuine H206 qualification missing: " + key)
        if doc.get("source_files") != admission["source"]["files"]:
            raise ValueError("qualification source closure differs: " + key)
        if doc.get("study_sha256") != admission["study_sha256"]:
            raise ValueError("qualification study binding differs: " + key)
        if key in ("input_qualification", "cuda_runtime_qualification") and (
            doc.get("study_input_payload_sha256") != admission["study_input_payload_sha256"]
            or doc.get("fold_manifest_sha256") != admission["fold_manifest_sha256"]
        ):
            raise ValueError("qualified input/fold payload differs from actual study")
        if key in ("model_qualification", "native91_qualification", "cuda_runtime_qualification") and (
            doc.get("bindings_sha256") != admission["bindings_sha256"]
        ):
            raise ValueError("qualified actual model bindings differ")
        if key == "queue_authority" and (
            doc.get("purpose") != purpose or doc.get("deadline_epoch") != admission["deadline_epoch"]
        ):
            raise ValueError("queue release does not authorize this purpose/original deadline")
        if key == "cuda_runtime_qualification":
            validate_runtime_proofs(doc)
            if doc.get("observed_exit_reconciled") is not True:
                raise ValueError("runtime invocation exit/tail accounting still pending")
            ObjectRef(**doc["observed_exit"])
            if doc.get("acceptance_publisher_costs_reconciled") is not True:
                raise ValueError("runtime acceptance publisher costs still pending")
            acceptance = ObjectRef(**doc["acceptance_candidate"])
            cost = json.loads(ObjectRef(**doc["acceptance_cost_receipt"]).read(store))
            if (
                cost.get("schema") != "h206-runtime-acceptance-accounting-v1"
                or cost.get("fixture") is not False
                or cost.get("status") != "PASS"
                or cost.get("acceptance_candidate") != acceptance.__dict__
                or cost.get("unresolved_tail") is not False
            ):
                raise ValueError("genuine runtime acceptance final-cost handback required")
    return True


def prepare_runtime_acceptance(pending, observed_exit, *, store, admission, budget, reservation_token):
    """Accounted proof restoration, still pending the publisher's final cost handback."""
    if not isinstance(budget, BudgetLedger) or not budget.active:
        raise ValueError("existing accounted runtime acceptance reservation required")
    ledger = budget.path

    def acceptance_reservation(state):
        r = state["reservations"].get(reservation_token, {})
        if (
            r.get("stage") != "runtime-qualification-acceptance"
            or r.get("cpu", 0) <= 0
            or r.get("gpu") != 0
            or set(state["reservations"]) != {reservation_token}
        ):
            raise ValueError("existing original-budget acceptance reservation required")
        return r["prior_ledger_sha256"]

    prior_ledger_sha = budget.mutate(acceptance_reservation)
    if execution_purpose(admission) != "runtime_qualification":
        raise ValueError("only runtime qualification can receive this observed-exit acceptance")
    doc = json.loads(pending.read(store))
    exit_doc = json.loads(observed_exit.read(store))
    if doc.get("schema") != "h206-cuda-runtime-pending-v1" or doc.get("status") != "PENDING_FINAL_ACCOUNTING":
        raise ValueError("pending actual runtime proof required")
    validate_runtime_proofs(doc)
    if (
        exit_doc.get("schema") != "h206-observed-qualification-exit-v1"
        or exit_doc.get("fixture") is not False
        or exit_doc.get("controller_exit_code") != 0
        or exit_doc.get("controller_alive") is not False
        or exit_doc.get("worker_alive") is not False
        or exit_doc.get("logs_closed") is not True
        or exit_doc.get("unresolved_tail") is not False
        or exit_doc.get("pending_proof") != pending.__dict__
        or exit_doc.get("admission_sha256") != sha(canonical(admission))
        or exit_doc.get("ledger_sha256") != prior_ledger_sha
    ):
        raise ValueError("genuine observed controller exit/cost reconciliation required")

    def settled(state):
        attempt = state.get("runtime_qualification_attempt", {})
        if (
            state.get("execution_admitted") is not True
            or state.get("exhausted")
            or set(state["reservations"]) != {reservation_token}
            or attempt.get("terminal") is not True
            or attempt.get("admission_sha256") != sha(canonical(admission))
            or exit_doc.get("controller_pid") != attempt.get("supervisor_pid")
        ):
            raise ValueError("canonical runtime qualification costs/exit remain held")
        entries = state["entries"][attempt["entry_start_index"] : attempt["entry_end_index"]]
        phases = [e for e in entries if e["stage"] != "controller-overhead"]
        expected = {
            f"round-{r}/seed-206101/fold-0/{a}/{p}"
            for a in runtime_qualification_plan()["arms"]
            for r in (1, 2, 3)
            for p in ("propose", "teacher", "train")
        }
        if (
            len(phases) != 36
            or {e["stage"] for e in phases} != expected
            or any(e.get("failed") or e.get("reservation_overrun") for e in phases)
        ):
            raise ValueError("all actual qualification phase costs/outcomes must be reconciled")

    BudgetLedger(ledger).mutate(settled)
    raw_rounds = {(p["arm"], p["round"]): p for p in doc["round_proofs"]}
    for phase in doc["phase_executions"]:
        raw = json.loads(ObjectRef(**phase["execution"]).read(store))
        gpu = phase["phase"] in GPU_PHASES
        if (
            raw.get("schema") != "h206-phase-execution-v1"
            or raw.get("fixture") is not False
            or raw.get("purpose") != "runtime_qualification"
            or raw.get("child_exit_code") != 0
            or raw.get("source_files") != admission["source"]["files"]
            or not 0 < raw["runtime"]["numeric"]["numeric_threads_upper_bound"] <= 4
            or raw["runtime"]["cuda_visible"] != (GPU_UUID if gpu else "")
            or (gpu and (raw.get("device") or {}).get("uuid") != GPU_UUID)
            or (not gpu and raw.get("device") is not None)
        ):
            raise ValueError("actual source-bound worker/numeric/device phase proof missing")
        if phase["phase"] == "propose" and phase["round"] == 1:
            proposal = json.loads(ObjectRef(**raw["result"]).read(store))
            if proposal.get("actor") != runtime_qualification_plan()["initial_state_sha256"]:
                raise ValueError("qualification model differs from pinned initial student")
        if phase["phase"] == "train":
            summary = raw_rounds[(phase["arm"], phase["round"])]
            if raw.get("result") != summary["round_object"]:
                raise ValueError("actual worker result differs from qualification round")
            full = json.loads(ObjectRef(**raw["result"]).read(store))
            if (
                full.get("proof") != summary["proof"]
                or len(full.get("steps", [])) != 600
                or full.get("state_sha256") != summary["state_sha256"]
                or full.get("updates") != summary["updates"]
            ):
                raise ValueError("full durable update/checkpoint proof differs from qualification summary")
    for key, value in (
        ("source_files", admission["source"]["files"]),
        ("study_sha256", admission["study_sha256"]),
        ("study_input_payload_sha256", admission["study_input_payload_sha256"]),
        ("fold_manifest_sha256", admission["fold_manifest_sha256"]),
        ("bindings_sha256", admission["bindings_sha256"]),
    ):
        if doc.get(key) != value:
            raise ValueError("runtime proof differs from original source/input/study")
    if psutil.pid_exists(exit_doc["controller_pid"]) or exit_doc["controller_pid"] == os.getpid():
        raise ValueError("qualification controller has not actually exited")
    original_deadline_guard(admission)()
    ref = store.put(
        "cuda-runtime-acceptance-candidate.json",
        canonical(
            {
                **doc,
                "schema": "h206-cuda-runtime-acceptance-candidate-v1",
                "status": "PENDING_ACCEPTANCE_PUBLISHER_COSTS",
                "acceptance_publisher_costs_reconciled": False,
                "acceptance_reservation": reservation_token,
                "observed_exit_reconciled": True,
                "observed_exit": observed_exit.__dict__,
            }
        ),
    )
    original_deadline_guard(admission)()
    return ref


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
        self.executions = []

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
        execution_purpose(self.admission)

        def accounting(state):
            if state.get("campaign_deadline_epoch") not in (None, self.admission["deadline_epoch"]):
                raise ValueError("original canonical deadline cannot be renewed")
            if state.get("cpu_used_seconds") is None or state.get("execution_admitted") is not True:
                raise ValueError("canonical campaign accounting/admission still held")
            for kind in ("cpu", "gpu"):
                value = state[kind + "_used_seconds"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid actual cumulative usage")

        self.budget.mutate(accounting)
        verify_qualifications(self.admission, self.store)
        require_resources(current_resources(self.cache))
        return True

    @contextlib.contextmanager
    def controller(self):
        before, began = process_cpu(), time.monotonic()
        self.validate()
        lock = (self.output / "controller.lock").open("a")
        overhead_token = "controller-" + str(os.getpid()) + "-" + str(time.time_ns())
        initial_entries = None
        purpose = execution_purpose(self.admission)
        attempt_key = "scientific_attempt" if purpose == "study" else "runtime_qualification_attempt"
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reserve = self.admission["controller_cpu_reserve_seconds"]
            if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
                raise ValueError("bounded original controller CPU reservation required")

            def claim(state):
                nonlocal initial_entries
                if state.get(attempt_key) is not None:
                    raise RuntimeError("original H206 attempt already claimed; no retry/reset")
                reserved = sum(r["cpu"] for r in state["reservations"].values())
                if state["cpu_used_seconds"] + reserved + reserve > state["cpu_cap_seconds"]:
                    raise RuntimeError("controller cumulative CPU budget exhausted")
                if state.get("campaign_deadline_epoch") not in (None, self.admission["deadline_epoch"]):
                    raise ValueError("original canonical deadline cannot be renewed")
                state.setdefault("campaign_deadline_epoch", self.admission["deadline_epoch"])
                initial_entries = len(state["entries"])
                state["reservations"][overhead_token] = {
                    "cpu": reserve,
                    "gpu": 0,
                    "stage": "controller-overhead",
                }
                state[attempt_key] = {
                    "purpose": purpose,
                    "entry_start_index": initial_entries,
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
                    state[attempt_key]["terminal"] = True
                    state[attempt_key]["entry_end_index"] = len(state["entries"])

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
            yield None
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
                yield snapshot
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
        validate_phase_purpose(task, self.admission)
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
            phase_deadline_epoch=min(self.admission["deadline_epoch"], time.time() + seconds_cap),
        )
        gpu_reserve = seconds_cap if gpu else 0
        with self.budget.operation(stage, cpu_reserve=cpu_reserve, gpu_reserve=gpu_reserve) as token:
            task["reservation"] = token
            atomic_record(folder / "task.json", task)
            with self._device(gpu) as device_observation:
                original_deadline_guard(self.admission, task["phase_deadline_epoch"])()
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
                runtime = json.loads((folder / "runtime.json").read_text())
                upper = runtime["numeric"]["numeric_threads_upper_bound"]
                if runtime["pid"] != child.pid or not 0 < upper <= 4:
                    raise RuntimeError("actual terminal child numeric identity/bound")
                original_deadline_guard(self.admission, task["phase_deadline_epoch"])()
                execution = self.store.put(
                    stage + "/execution.json",
                    canonical(
                        {
                            "schema": "h206-phase-execution-v1",
                            "fixture": False,
                            "purpose": execution_purpose(self.admission),
                            "task_sha256": sha(canonical(task)),
                            "runtime": runtime,
                            "child_exit_code": child.returncode,
                            "device": device_observation,
                            "result": ref.__dict__,
                            "source_files": self.admission["source"]["files"],
                        }
                    ),
                )
                self.executions.append(
                    {
                        "arm": task["arm"],
                        "round": round_index,
                        "phase": phase,
                        "execution": execution.__dict__,
                    }
                )
                original_deadline_guard(self.admission, task["phase_deadline_epoch"])()
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
        validate_phase_purpose(task, task["admission"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise ValueError("GPU phase visibility differs from assigned device")
        verify_merged_source(self.repo, task["admission"]["source"])
        budget = BudgetLedger(task["ledger"])

        def verify(state):
            r = state["reservations"].get(self.token)
            key = (
                "scientific_attempt"
                if execution_purpose(task["admission"]) == "study"
                else "runtime_qualification_attempt"
            )
            attempt = state.get(key, {})
            if (
                r is None
                or r.get("stage") != task["stage"]
                or r["gpu"] <= 0
                or attempt.get("supervisor_pid") != task["supervisor_pid"]
                or attempt.get("admission_sha256") != sha(canonical(task["admission"]))
                or attempt.get("terminal") is not False
                or state.get("campaign_deadline_epoch") != task["admission"]["deadline_epoch"]
            ):
                raise ValueError("actual parent GPU reservation/attempt required")

        budget.mutate(verify)
        verify_qualifications(task["admission"], self.store)
        if time.time() >= task["admission"]["deadline_epoch"]:
            raise RuntimeError("original campaign deadline expired")
        # Only after merged source, real qualification and the actual cumulative
        # parent reservation exist may this child initialize any CUDA context.
        import torch

        if torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("actual assigned CUDA device unavailable")
        self.device = torch.device("cuda:0")
        return self
