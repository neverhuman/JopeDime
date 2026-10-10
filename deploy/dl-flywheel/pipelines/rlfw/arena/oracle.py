"""Wrap a pinned evaluator and disclose only after the outbox charge commits."""

from copy import deepcopy
import math
from pathlib import Path
import threading
import time

from .game_spec import digest, require_sha, R5_KPI_VERSION
from .kpi import SlotResult


class BlindOracle:
    def __init__(self, spec, evaluate, role_check, frozen_cache):
        self.spec, self.evaluate, self.role_check = spec, evaluate, role_check
        self.cache = deepcopy(frozen_cache)
        if digest(self.cache) != spec.cache_snapshot_sha256:
            raise ValueError("cache snapshot hash mismatch")
        self.own_results = {}
        self.locks = {}
        self.locks_guard = threading.Lock()

    def check_role(self, context):
        role = self.role_check(context.context_id)
        if role != {
            "role": "development",
            "role_map_sha256": self.spec.role_map_sha256,
        }:
            raise ValueError(
                "fresh role/exclusion authority changed; freeze and re-spec"
            )

    def execution_key(self, context, program):
        return digest(
            {
                "context_identity": context.identity_sha256,
                "program_sha256": program.program_sha256,
                "evaluator_sha256": self.spec.evaluator_sha256,
                "canonicalizer_sha256": self.spec.canonicalizer_sha256,
                "protocol": self.spec.protocol,
            }
        )

    def validate(self, result, context, program):
        result = dict(result)
        outcome = result.get("outcome_class")
        if outcome == "completed":
            outcome = "qualified_complete"
        if outcome not in (
            "qualified_complete",
            "program_failed",
            "timeout",
            "parity_unqualified",
            "parity_custody",
            "infra_unknown",
        ):
            raise ValueError("evaluator returned an unknown outcome taxonomy")
        if outcome == "parity_unqualified":
            if self.spec.kpi_version != R5_KPI_VERSION:
                raise ValueError("program parity requires an explicit R5 game version")
            if (
                result.get("qualified") is not False
                or result.get("parity_verified") is not False
            ):
                raise ValueError(
                    "an unqualified parity outcome cannot claim qualification"
                )
            error = result.get("parity_max_normalized_error")
            if error is not None and (
                not isinstance(error, (int, float))
                or not math.isfinite(error)
                or error <= 1e-9
            ):
                raise ValueError("a measured parity mismatch must exceed fixed 1e-9")
        elif outcome == "parity_custody" and self.spec.kpi_version == R5_KPI_VERSION:
            raise ValueError("R5 refuses legacy context-parity retry outcomes")
        result["outcome_class"] = outcome
        for name, value in (
            ("context_identity", context.identity_sha256),
            ("program_sha256", program.program_sha256),
            ("evaluator_sha256", self.spec.evaluator_sha256),
            ("canonicalizer_sha256", self.spec.canonicalizer_sha256),
            ("protocol", self.spec.protocol),
            ("role_map_sha256", self.spec.role_map_sha256),
        ):
            if result.get(name) != value:
                raise ValueError(f"evaluator/cache identity mismatch: {name}")
        if outcome == "qualified_complete":
            if (
                result.get("qualified") is not True
                or result.get("closure_verified") is not True
                or result.get("parity_verified") is not True
            ):
                raise ValueError("completion requires closure and native/Rust parity")
            error = result.get("parity_max_normalized_error")
            if (
                not isinstance(error, (float, int))
                or not math.isfinite(error)
                or not 0 <= error <= 1e-9
            ):
                raise ValueError("fixed normalized 1e-9 parity tolerance required")
            if type(result.get("q_grid")) is not int:
                raise ValueError("only measured integer micro-q enters scores")
            require_sha(result.get("receipt_sha256"), "execution receipt")
        elif (
            result.get("q_grid") is not None
            or result.get("q") is not None
            or result.get("q_opt") is not None
        ):
            raise ValueError("failures and unknowns cannot disclose numeric scores")
        for name in ("wall_seconds", "cpu_seconds"):
            if (
                not isinstance(result.get(name), (int, float))
                or not math.isfinite(result[name])
                or result[name] < 0
            ):
                raise ValueError("measured nonnegative evaluator costs required")
        return result

    def resolve(
        self,
        context,
        arm,
        seed,
        slot,
        candidate,
        *,
        duplicate,
        commit,
        attempt_store=None,
    ):
        """Scores are private inside this method until commit(slot) has succeeded."""
        key = self.execution_key(context, candidate.program)
        with self.locks_guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            return self._resolve(
                context,
                arm,
                seed,
                slot,
                candidate,
                duplicate=duplicate,
                commit=commit,
                attempt_store=attempt_store,
            )

    def _resolve(
        self, context, arm, seed, slot, candidate, *, duplicate, commit, attempt_store
    ):
        self.check_role(context)
        self.spec.verify_frozen()
        if (Path(self.spec.bench_root) / "state/STOP.json").exists():
            raise RuntimeError("campaign stop marker is present")
        if (
            self.spec.deadline_unix is not None
            and time.time() >= self.spec.deadline_unix
        ):
            raise RuntimeError("fixed campaign deadline expired")
        key = self.execution_key(context, candidate.program)
        request = {
            "context_id": context.context_id,
            "context_identity": context.identity_sha256,
            "lineage_id": context.lineage_id,
            "program": candidate.program.to_dict(),
            "logical_slot_id": digest(
                [self.spec.sha256, context.context_id, arm.name, seed, slot]
            ),
            "execution_key": key,
            "spec_sha256": self.spec.sha256,
            "evaluator_sha256": self.spec.evaluator_sha256,
            "canonicalizer_sha256": self.spec.canonicalizer_sha256,
            "protocol": self.spec.protocol,
            "role_map_sha256": self.spec.role_map_sha256,
        }
        source = self.cache.get(key, self.own_results.get(key))
        cached = source is not None
        attempt_key = (context.context_id, arm.name, seed, slot)
        history = attempt_store.attempt_history(attempt_key) if attempt_store else []
        terminal = (
            attempt_store.terminal_attempt(attempt_key)
            if attempt_store and self.spec.kpi_version == R5_KPI_VERSION
            else None
        )
        if terminal:
            if terminal["number"] != len(history):
                raise ValueError("terminal fate and attempt history differ")
            result = self.validate(terminal["result"], context, candidate.program)
            if result["outcome_class"] != "parity_unqualified":
                raise ValueError("terminal replay requires its resolved parity fate")
            # The original logical attempt keeps its actual cost and cache status.
            # The immutable program evidence retains the physical source costs.
            history = history[:-1] + [
                {
                    "result": {
                        **result,
                        "wall_seconds": terminal["wall_seconds"],
                        "cpu_seconds": terminal["cpu_seconds"],
                        "cost_unknown": terminal["unknown_cost"],
                    }
                }
            ]
            cached = terminal["cache_hit"]
        wall = sum(
            r["result"]["wall_seconds"] for r in history if r["result"] is not None
        )
        cpu = sum(
            r["result"]["cpu_seconds"] for r in history if r["result"] is not None
        )
        unknown_costs = sum(
            r["result"] is None or r["result"].get("cost_unknown", False)
            for r in history
        )
        result = (
            terminal["result"]
            if terminal
            else history[-1]["result"]
            if history and history[-1]["result"] is not None
            else {
                "outcome_class": "infra_unknown",
                "qualified": False,
                "q_grid": None,
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "cost_unknown": True,
                "reason": "reserved attempt outcome unavailable after interruption",
            }
        )
        attempt = len(history)
        for number in () if terminal else range(len(history) + 1, 4):
            if (Path(self.spec.bench_root) / "state/STOP.json").exists():
                raise RuntimeError("campaign stop marker is present")
            if (
                self.spec.deadline_unix is not None
                and time.time() >= self.spec.deadline_unix
            ):
                raise RuntimeError("fixed campaign deadline expired")
            self.check_role(context)
            self.spec.verify_frozen()
            attempt = (
                attempt_store.reserve_attempt(attempt_key) if attempt_store else number
            )
            result = self.validate(
                source if cached else self.evaluate({**request, "attempt": attempt}),
                context,
                candidate.program,
            )
            if not cached:
                wall += result["wall_seconds"]
                cpu += result["cpu_seconds"]
                unknown_costs += bool(result.get("cost_unknown", False))
            if result["outcome_class"] == "parity_unqualified":
                if attempt_store:
                    attempt_store.record_program_fate(
                        attempt_key, attempt, key, result, cached=cached
                    )
                self.own_results[key] = result
            if result["outcome_class"] not in ("parity_custody", "infra_unknown"):
                break
            # Unknown cache entries are never reusable outcomes.
            if attempt_store:
                attempt_store.record_unknown_attempt(attempt_key, attempt, result)
            cached, source = False, None
        self.check_role(context)
        outcome = result["outcome_class"]
        resolved = outcome not in ("parity_custody", "infra_unknown")
        row = SlotResult(
            context.context_id,
            context.lineage_id,
            arm.name,
            seed,
            slot,
            candidate.program.program_sha256,
            outcome if resolved else "censored",
            result.get("q_grid")
            if resolved and outcome == "qualified_complete"
            else None,
            resolved and outcome == "qualified_complete",
            cached,
            duplicate,
            attempt,
            wall,
            cpu,
            self.spec.sha256,
            result.get("receipt_sha256")
            if resolved and outcome == "qualified_complete"
            else None,
            None if resolved else outcome,
            unknown_costs,
        )
        # Persist the charge and the qualified result in one transaction before returning.
        commit(row, key, result)
        if resolved:
            self.own_results[key] = result
        return row
