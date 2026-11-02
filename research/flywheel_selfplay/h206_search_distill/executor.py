"""Durable H206 round execution and one cumulative, resumable cost ledger."""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import canonical, digest
from .native import collect_teacher


class BudgetLedger:
    """Single coordinator ledger; workers/helpers must use its existing reservations.

    No constructor creates a new campaign or resets usage. Distributed submission
    requires routing reservations through this coordinator, not copying this file.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError("existing cumulative H206 ledger required")
        self.active = False

    def mutate(self, fn):
        with Path(str(self.path) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text())
            if (
                state.get("campaign_id") != "H206"
                or state["cpu_cap_seconds"] != 400 * 3600
                or state["gpu_cap_seconds"] != 18 * 3600
            ):
                raise ValueError("campaign caps changed")
            result = fn(state)
            temp = self.path.with_suffix(".next")
            with temp.open("w") as f:
                f.write(canonical(state).decode())
                f.flush()
                os.fsync(f.fileno())
            temp.replace(self.path)
            return result

    @contextlib.contextmanager
    def measure(self, stage):
        if not self.active:
            raise RuntimeError("stage requires active cumulative reservation")
        before = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        try:
            yield
        finally:
            after = resource.getrusage(resource.RUSAGE_SELF)
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = (
                after.ru_utime
                + after.ru_stime
                + children.ru_utime
                + children.ru_stime
                - before.ru_utime
                - before.ru_stime
                - child.ru_utime
                - child.ru_stime
            )
            self.mutate(
                lambda s: s["stages"].append(
                    {
                        "stage": stage,
                        "cpu_seconds": max(0.0, cpu),
                        "wall_seconds": time.monotonic() - start,
                        "accounting": "breakdown only; included once in round_total",
                    }
                )
            )

    @contextlib.contextmanager
    def operation(self, stage, *, cpu_reserve, gpu_reserve):
        if self.active or any(not math.isfinite(v) or v < 0 for v in (cpu_reserve, gpu_reserve)):
            raise ValueError("valid non-nested reservation required")
        token = str(time.time_ns())

        def reserve(s):
            if s.get("exhausted") or any(
                s[f"{kind}_used_seconds"] + sum(v[kind] for v in s["reservations"].values()) + requested
                > s[f"{kind}_cap_seconds"]
                for kind, requested in (("cpu", cpu_reserve), ("gpu", gpu_reserve))
            ):
                raise RuntimeError("cumulative campaign budget exhausted")
            s["reservations"][token] = {"cpu": cpu_reserve, "gpu": gpu_reserve}

        self.mutate(reserve)

        def cpu():
            a = resource.getrusage(resource.RUSAGE_SELF)
            b = resource.getrusage(resource.RUSAGE_CHILDREN)
            return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime

        started, before, failed = time.monotonic(), cpu(), True
        self.active = True
        try:
            yield
            failed = False
        finally:
            elapsed = time.monotonic() - started
            used_cpu = max(0.0, cpu() - before)
            used_gpu = elapsed if gpu_reserve else 0.0
            overrun = used_cpu > cpu_reserve or used_gpu > gpu_reserve

            def finish(s):
                s["reservations"].pop(token)
                s["cpu_used_seconds"] += used_cpu
                s["gpu_used_seconds"] += used_gpu
                s["exhausted"] = bool(s.get("exhausted") or overrun)
                s["entries"].append(
                    {
                        "stage": stage,
                        "cpu_seconds": used_cpu,
                        "gpu_seconds": used_gpu,
                        "wall_seconds": elapsed,
                        "failed": failed,
                        "reservation_overrun": overrun,
                    }
                )

            self.mutate(finish)
            self.active = False
            if overrun:
                raise RuntimeError("operation exceeded its cumulative reservation")


def evaluate_fixed(learner, rows, *, checkpoint_sha256, store, prefix, b_lock=None):
    if learner.state_hash() != checkpoint_sha256:
        raise ValueError("evaluation checkpoint differs from fixed terminal state")
    if any(row.context.role not in ("B", "A") for row in rows):
        raise ValueError("fixed evaluation requires held-out B/A roles")
    if any(row.context.role == "A" for row in rows):
        if b_lock is None:
            raise ValueError("A evaluation before durable B lock")
        lock = json.loads(b_lock.read(store))
        if lock.get("status") != "B_LOCKED" or checkpoint_sha256 not in lock.get("checkpoint_sha256", []):
            raise ValueError("A evaluation B lock/checkpoint mismatch")
    panel = {}
    for row in rows:
        row.verify()
        panel[row.context.context_id] = learner.score(row).tolist()
    # Flush before any subsequent wait, stop or selector operation.
    return store.put(
        prefix + "/score-panel.json",
        canonical(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "panel": panel,
                "roles": {r.context.context_id: r.context.role for r in rows},
            }
        ),
    )


def _run_round(
    *,
    round_index,
    learner,
    acquire,
    native,
    cache,
    store,
    prefix,
    budget,
    k,
    updates=600,
    evaluation_rows=(),
    prior_round=None,
):
    """Real actor -> native teacher -> exact-version labels -> optimizer -> scores.

    One seed/cell/arm. The full scientific coordinator/qualification remains gated.
    acquire is the approved generator/encoder transport, bound to this actor hash.
    """
    if round_index not in (1, 2, 3) or updates < 1 or updates > 600:
        raise ValueError("fixed three-round/update budget required")
    prior = {"seen_families": [], "seen_contexts": [], "seen_family_components": []}
    if round_index == 1 and learner.updates != 0:
        raise ValueError("round1 cannot reset a learner with prior updates")
    calls_before, failures_before, hits_before = native.calls, native.failures, cache.hits
    if round_index > 1:
        if prior_round is None:
            raise ValueError("durable prior round required")
        prior = json.loads(prior_round.read(store))
        if (
            prior["round"] != round_index - 1
            or prior["checkpoint_sha256"] != learner.state_hash()
            or prior["k"] != k
            or prior["optimizer_updates_total"] != learner.updates
        ):
            raise ValueError("round ancestry/state mismatch")
    actor_hash = learner.state_hash()
    with budget.measure("approved_context_acquisition"):
        loaded = list(acquire(round_index, actor_hash))
    if not loaded or len({r.context.context_id for r in loaded}) != len(loaded):
        raise ValueError("unique approved FIT contexts required")
    teachers, before_updates = [], learner.updates
    try:
        for i, row in enumerate(loaded):
            if row.context.role != "FIT" or row.context.actor_sha256 != actor_hash:
                raise ValueError("acquired context role/current actor mismatch")
            with budget.measure("actor_token_construction"):
                scores = learner.score(row)
            ref = collect_teacher(row, scores, k, native, cache, store, f"{prefix}/context-{i}", budget)
            teachers.append(ref)
        round_before = learner.round_snapshot(loaded[0])
        family_groups = {}
        for i, row in enumerate(loaded):
            family_groups.setdefault(row.family_component_sha256, []).append(i)
        family_names = sorted(family_groups)
        rng = np.random.default_rng(loaded[0].context.seed)
        batch_plan = [
            [int(rng.choice(family_groups[str(rng.choice(family_names))])) for _ in range(32)]
            for _ in range(updates)
        ]
        for indices in batch_plan:
            # All labels were durably persisted before the first optimizer update.
            with budget.measure("student_update"):
                learner.update_batch([loaded[i] for i in indices], [teachers[i] for i in indices], store)
        proof = learner.proof_since(round_before, loaded[0])
        checkpoint = io.BytesIO()
        torch.save(
            {
                "model": learner.model.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
                "updates": learner.updates,
                "examples": learner.examples,
                "scope_sha256": learner.scope_sha256,
                "gradient_squares": learner.gradient_squares,
            },
            checkpoint,
        )
        with budget.measure("checkpoint_custody"):
            checkpoint_ref = store.put(prefix + "/checkpoint.pt", checkpoint.getvalue())
        checkpoint_hash = learner.state_hash()
        panels = []
        if evaluation_rows and (round_index != 3 or learner.updates != 1800):
            raise ValueError("B scoring forbidden before fixed round3/update1800 endpoint")
        for role in ("B",):
            rows = [r for r in evaluation_rows if r.context.role == role]
            if rows:
                panels.append(
                    evaluate_fixed(
                        learner,
                        rows,
                        checkpoint_sha256=checkpoint_hash,
                        store=store,
                        prefix=prefix + "/" + role,
                    )
                )
        families = sorted({r.context.family_id for r in loaded})
        contexts = {r.context.context_id for r in loaded}
        previous_families, previous_contexts = set(prior["seen_families"]), set(prior["seen_contexts"])
        references = {f for r in loaded for f in r.reference_families}
        components = {r.family_component_sha256 for r in loaded}
        previous_components = set(prior["seen_family_components"])
        result = {
            "schema": "h206-round-v1",
            "round": round_index,
            "k": k,
            "actor_sha256": actor_hash,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint": checkpoint_ref.__dict__,
            "teachers": [r.__dict__ for r in teachers],
            "score_panels": [r.__dict__ for r in panels],
            "proof": proof,
            "optimizer_updates_this_round": learner.updates - before_updates,
            "optimizer_updates_total": learner.updates,
            "effective_batch_size": 32,
            "batch_plan_sha256": digest(batch_plan),
            "candidate_endpoint_eligible": round_index == 3 and learner.updates == 1800,
            "seen_families": sorted(previous_families | set(families)),
            "seen_contexts": sorted(previous_contexts | contexts),
            "seen_family_components": sorted(previous_components | components),
            "new_independent_families_vs_reference": sorted(components - references - previous_components),
            "new_contexts_to_campaign": sorted(contexts - previous_contexts),
            "repeated_contexts": sorted(contexts & previous_contexts),
            "new_native_calls": native.calls - calls_before,
            "new_native_failures": native.failures - failures_before,
            "cache_hits_this_round": cache.hits - hits_before,
            "families_observed": families,
            "contexts_observed": [r.context.context_id for r in loaded],
            "acquisition_claim": "observed identities only; novelty must be compared with prior/reference manifests",
            "cache": {k: getattr(cache, k) for k in ("hits", "misses", "hit_bytes", "miss_bytes")},
        }
        return store.put(prefix + "/round.json", canonical(result))
    except BaseException as exc:
        store.put(
            prefix + "/failure.json",
            canonical(
                {
                    "round": round_index,
                    "error": repr(exc),
                    "optimizer_updates": learner.updates - before_updates,
                    "verified_teachers": [r.__dict__ for r in teachers],
                    "failed_teacher": getattr(getattr(exc, "receipt", None), "__dict__", None),
                }
            ),
        )
        raise


def run_round(*, cpu_reserve_seconds, budget, learner, **kwargs):
    # CPU source qualification proceeds while the GPU queue remains closed.
    # The scientific CUDA phase controller/device reservation is a separate pin.
    if next(learner.model.parameters()).device.type != "cpu":
        raise RuntimeError("GPU phase controller not qualified; H206 queue closed")
    with budget.operation(
        "round_total_including_setup_waits_custody", cpu_reserve=cpu_reserve_seconds, gpu_reserve=0
    ):
        return _run_round(budget=budget, learner=learner, **kwargs)


def initialize_learner(
    factory, checkpoint_ref, store, *, budget, cpu_reserve_seconds, expected_checkpoint_sha256, **model_kwargs
):
    """Charge initialization, verify exact model bytes, call the real pinned factory."""
    from .student import Learner, SearchStudent

    if checkpoint_ref.sha256 != expected_checkpoint_sha256:
        raise ValueError("initial checkpoint identity differs from approved model pin")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU factory requires CUDA_VISIBLE_DEVICES empty before initialization")
    with budget.operation("model_setup_including_fetch", cpu_reserve=cpu_reserve_seconds, gpu_reserve=0):
        fused = factory(checkpoint_ref.read(store))
        if next(fused.parameters()).device.type != "cpu":
            raise RuntimeError("GPU initialization unavailable while queue closed")
        return Learner(SearchStudent(fused, **model_kwargs))


def restore_learner(learner, checkpoint_ref, store, *, expected_state_sha256, budget, cpu_reserve_seconds):
    """Verified state+optimizer resume; never creates or resets the cost ledger."""
    with budget.operation("checkpoint_resume", cpu_reserve=cpu_reserve_seconds, gpu_reserve=0):
        state = torch.load(io.BytesIO(checkpoint_ref.read(store)), map_location="cpu", weights_only=True)
        learner.model.load_state_dict(state["model"], strict=True)
        learner.optimizer.load_state_dict(state["optimizer"])
        learner.updates = state["updates"]
        learner.examples = state["examples"]
        learner.scope_sha256 = state["scope_sha256"]
        learner.gradient_squares = state["gradient_squares"]
        learner.model.assert_frozen()
        if learner.state_hash() != expected_state_sha256:
            raise ValueError("resumed checkpoint content identity mismatch")
        return learner
