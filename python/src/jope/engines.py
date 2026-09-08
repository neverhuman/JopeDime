"""EngineRouter: routes genomes to the correct engine bridge for validation and scoring.

Every genome is an ``(engine, genome)`` pair. The router owns one RustValidator per
engine (featmath -> rust/jope-validate, newmath -> rust/jope-newmath-validate), keeps
their evaluator versions for cache keys, provides per-genome wall-clock timeouts by
head cost class, and preserves input ordering when a batch mixes engines.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .validator import PROJECT_ROOT, RustValidator
from .prime_config import ssh_control_path

ENGINE_FEATMATH = "featmath"
ENGINE_NEWMATH = "newmath"
ENGINES = (ENGINE_FEATMATH, ENGINE_NEWMATH)
DEFAULT_ENGINE = ENGINE_FEATMATH

TASK_REGRESSION = "regression"
TASK_BINARY = "binary"
TASKS = (TASK_REGRESSION, TASK_BINARY)
DEFAULT_TASK = TASK_REGRESSION

# The bridge subcommand + cache-key evaluator string are keyed by (engine, task).
EVAL_SUBCOMMAND = {TASK_REGRESSION: "eval-regression", TASK_BINARY: "eval-binary"}
EVALUATOR_VERSIONS_BY_TASK = {
    (ENGINE_FEATMATH, TASK_REGRESSION): "jope-validate.eval-regression.v2",
    (ENGINE_NEWMATH, TASK_REGRESSION): "jope-newmath-validate.eval-regression.v1",
    (ENGINE_FEATMATH, TASK_BINARY): "jope-validate.eval-binary.v1",
    (ENGINE_NEWMATH, TASK_BINARY): "jope-newmath-validate.eval-binary.v1",
    ("prime", TASK_REGRESSION): "jope-prime-evald.compiled-plan.v1",
}
# Legacy flat alias (regression) kept for callers that index by engine only.
EVALUATOR_VERSIONS = {
    ENGINE_FEATMATH: EVALUATOR_VERSIONS_BY_TASK[(ENGINE_FEATMATH, TASK_REGRESSION)],
    ENGINE_NEWMATH: EVALUATOR_VERSIONS_BY_TASK[(ENGINE_NEWMATH, TASK_REGRESSION)],
}

FALLBACK_GENOMES = {
    ENGINE_FEATMATH: "(pipe X (robust_scale) (ridge))",
    ENGINE_NEWMATH: "(pipe X (robust_scale) (ridge alpha=1.0))",
}

# Wall-clock eval timeout per head cost class (seconds). "portfolio" covers the
# v35 multi-member OOF+NNLS head (~200 member-fits per eval; see
# docs/specs/portfolio_head.md §7.5).
TIMEOUT_BY_CLASS = {
    "cheap": 10.0,
    "medium": 30.0,
    "expensive": 90.0,
    "extreme": 120.0,
    "portfolio": 600.0,
}

HEAD_COST_CLASS = {
    ENGINE_FEATMATH: {
        "ridge": "cheap",
        "logistic": "cheap",
        "elasticnet": "cheap",
        # Jacobi eigensolve is O(p^3): ~1 ms at p=32 but ~10 s at the 512-col cap
        # after wide transform chains — "medium" (30 s) covers the tail.
        "ridge_gcv": "medium",
        "histgbm": "medium",
        "gbm": "medium",
        "ksmooth": "medium",
        "ensemble": "medium",
        "ensemble_ridge_gbm": "medium",
        "histgbm2": "expensive",
        "krr": "expensive",
        "stack": "expensive",
        "gpr": "extreme",
        "portfolio": "portfolio",
    },
    ENGINE_NEWMATH: {
        "ridge": "cheap",
        "logistic": "cheap",
        "elasticnet": "cheap",
        "huber_ridge": "cheap",
        "knn": "medium",
        "gbm": "medium",
        "random_forest": "medium",
        "ensemble_ridge_gbm": "medium",
    },
}

_HEAD_RE = re.compile(r"\(([a-z_0-9]+)[^()]*\)\s*\)\s*$")


def head_op_of(genome: str) -> str | None:
    """Best-effort last-stage op name from a genome string (for cost classing)."""
    match = _HEAD_RE.search(genome or "")
    return match.group(1) if match else None


@dataclass(frozen=True)
class EngineConfig:
    name: str
    manifest: Path
    binary_env: str
    release_binary: Path
    default_timeout_s: float = 30.0


DEFAULT_CONFIGS = {
    ENGINE_FEATMATH: EngineConfig(
        name=ENGINE_FEATMATH,
        manifest=PROJECT_ROOT / "rust" / "jope-validate" / "Cargo.toml",
        binary_env="JOPE_VALIDATE_BIN",
        release_binary=PROJECT_ROOT / "rust" / "jope-validate" / "target" / "release" / "jope-validate",
        default_timeout_s=30.0,
    ),
    ENGINE_NEWMATH: EngineConfig(
        name=ENGINE_NEWMATH,
        manifest=PROJECT_ROOT / "rust" / "jope-newmath-validate" / "Cargo.toml",
        binary_env="JOPE_NEWMATH_VALIDATE_BIN",
        release_binary=PROJECT_ROOT
        / "rust"
        / "jope-newmath-validate"
        / "target"
        / "release"
        / "jope-newmath-validate",
        default_timeout_s=15.0,
    ),
}


def eval_hosts_from_env() -> tuple[str, ...]:
    """Hosts for distributed evaluation: JOPE_EVAL_HOSTS=local,xbabe1,xbabe3."""
    raw = os.environ.get("JOPE_EVAL_HOSTS", "local")
    hosts = tuple(h.strip() for h in raw.split(",") if h.strip())
    return hosts or ("local",)


class EngineRouter:
    def __init__(
        self,
        configs: dict[str, EngineConfig] | None = None,
        *,
        validators: dict[str, RustValidator] | None = None,
        strict_handshake: bool = False,
        timeouts_enabled: bool = True,
        hosts: tuple[str, ...] | None = None,
        prime_search: Any | None = None,
    ) -> None:
        self.configs = dict(configs or DEFAULT_CONFIGS)
        if "prime" in self.configs or "prime" in (validators or {}):
            raise ValueError(
                "'prime' is reserved for the persistent native evaluator and "
                "cannot be configured as a legacy bridge"
            )
        self._validators: dict[tuple[str, str], RustValidator] = {}
        for engine, validator in (validators or {}).items():
            self._validators[(engine, "local")] = validator
        self.timeouts_enabled = timeouts_enabled
        self.hosts = tuple(hosts) if hosts else eval_hosts_from_env()
        self.prime_search = prime_search
        if strict_handshake:
            for engine in self.configs:
                self.handshake(engine)

    def engines(self) -> tuple[str, ...]:
        engines = list(self.configs)
        if self.prime_search is not None:
            engines.append("prime")
        return tuple(engines)

    def validator(self, engine: str, host: str = "local") -> RustValidator:
        engine = self._check_engine(engine)
        if engine == "prime":
            raise ValueError("Prime has no legacy validator; use the native evaluator")
        key = (engine, host)
        if key not in self._validators:
            cfg = self.configs[engine]
            self._validators[key] = RustValidator(
                manifest=cfg.manifest,
                engine=engine,
                binary_env=cfg.binary_env,
                release_binary=cfg.release_binary,
                ssh_host=None if host == "local" else host,
            )
        return self._validators[key]

    def evaluator_version(self, engine: str, task: str = DEFAULT_TASK) -> str:
        engine = self._check_engine(engine)
        if engine == "prime":
            if task != TASK_REGRESSION:
                raise ValueError("Prime native evaluation is regression-only")
            return "battle-evald.native.protocol-v4"
        return EVALUATOR_VERSIONS_BY_TASK[(engine, task)]

    def fallback(self, engine: str) -> str:
        engine = self._check_engine(engine)
        return "(pipe X (ridge))" if engine == "prime" else FALLBACK_GENOMES[engine]

    def timeout_for(self, engine: str, genome: str) -> float:
        engine = self._check_engine(engine)
        if engine == "prime":
            return float(getattr(self.prime_search, "request_timeout", None) or 300.0)
        head = head_op_of(genome)
        cost = HEAD_COST_CLASS.get(engine, {}).get(head or "", None)
        if cost is None:
            return self.configs[engine].default_timeout_s
        return TIMEOUT_BY_CLASS[cost]

    def handshake(self, engine: str) -> dict:
        """Query the bridge's grammar and sanity-check it. Raises on drift."""
        if engine == "prime":
            self._check_engine(engine)
            return {
                "engine": "prime",
                "evaluator_version": self.evaluator_version("prime"),
                "protocol_version": 4,
                "capabilities": self.prime_search.client.capabilities,
            }
        info = self.validator(engine).grammar()
        if info.get("engine") != engine:
            raise RuntimeError(f"bridge for {engine!r} reports engine {info.get('engine')!r}")
        if info.get("evaluator_version") != EVALUATOR_VERSIONS[engine]:
            raise RuntimeError(
                f"{engine} evaluator_version drift: bridge={info.get('evaluator_version')!r} "
                f"python={EVALUATOR_VERSIONS[engine]!r}"
            )
        return info

    def validate_many(self, items: Sequence[tuple[str, str] | dict]) -> list[dict]:
        """Validate (engine, genome) pairs (or row dicts with an ``engine`` key).

        Batches per engine, restores input order, stamps ``engine`` on outputs.
        """
        normalized: list[tuple[str, str | dict]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append((str(item.get("engine") or DEFAULT_ENGINE), item))
            else:
                engine, genome = item
                normalized.append((str(engine), genome))
        grouped: dict[str, list[int]] = {}
        for idx, (engine, _) in enumerate(normalized):
            grouped.setdefault(engine, []).append(idx)
        out: list[dict | None] = [None] * len(normalized)
        for engine, indices in grouped.items():
            if engine == "prime":
                self._check_engine(engine)
                raise ValueError(
                    "Prime validation is shape-aware and available only through "
                    "compiled native evaluation"
                )
            results = self.validator(engine).validate_many([normalized[i][1] for i in indices])
            for i, result in zip(indices, results, strict=True):
                result["engine"] = engine
                out[i] = result
        return [row for row in out if row is not None]

    def emit(self, items: Sequence[tuple[str, str] | dict], *, host: str = "local") -> list[dict]:
        """Transpile (engine, genome) pairs to standalone source via each engine's bridge.

        Same group-by-engine / restore-order contract as ``validate_many``. Rows the bridge
        can't emit come back ``supported: False`` (the caller falls back to a preview).
        """
        normalized: list[tuple[str, str | dict]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append((str(item.get("engine") or DEFAULT_ENGINE), item))
            else:
                engine, genome = item
                normalized.append((str(engine), genome))
        grouped: dict[str, list[int]] = {}
        for idx, (engine, _) in enumerate(normalized):
            grouped.setdefault(engine, []).append(idx)
        out: list[dict | None] = [None] * len(normalized)
        for engine, indices in grouped.items():
            if engine == "prime":
                self._check_engine(engine)
                raise ValueError(
                    "Prime user emission requires exact promotion through "
                    "PrimeNativeSearch.emit_user_python"
                )
            results = self.validator(engine, host).emit_many([normalized[i][1] for i in indices])
            for i, result in zip(indices, results, strict=True):
                result["engine"] = engine
                out[i] = result
        return [row for row in out if row is not None]

    def emit_one(self, engine: str, genome: str, *, host: str = "local") -> dict:
        return self.emit([(engine, genome)], host=host)[0]

    def eval_one(self, row: dict, *, timeout_s: float | None = None) -> dict:
        for _, result in self.iter_eval_regression_many([dict(row, **({} if timeout_s is None else {"_timeout_s": timeout_s}))]):
            return result
        raise RuntimeError("eval_one produced no result")

    def iter_eval_via_host_drivers(
        self,
        rows: Sequence[dict],
        *,
        workers_per_host: int = 96,
        hosts: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[int, dict]]:
        """High-throughput eval: ship each host its shard over ONE channel and evaluate it
        with the bridge's rayon ``eval-batch`` — the whole shard runs in ONE Rust process
        across up to ``workers_per_host`` cores (RAYON_NUM_THREADS, capped at 100).

        This is the FAST path and keeps the hot loop entirely in Rust: Python only ships the
        batch and collects results. It replaces both the per-lane ssh lock-step (capped at
        ~8-way by the head-node GIL + ssh latency) and any per-eval subprocess overhead.
        """
        import shlex

        hosts = tuple(hosts) if hosts else self.hosts
        if any(str(row.get("engine") or DEFAULT_ENGINE) == "prime" for row in rows):
            raise ValueError(
                "Prime rows must use the persistent native evaluator, not host drivers"
            )
        threads_per_host = max(1, min(int(workers_per_host), 100))  # user cap: never > 100 cores
        # group by (engine, task) so each host-call uses the right binary + subcommand
        grouped: dict[tuple[str, str], list[int]] = {}
        for idx, row in enumerate(rows):
            engine = self._check_engine(str(row.get("engine") or DEFAULT_ENGINE))
            task = str(row.get("task") or DEFAULT_TASK)
            grouped.setdefault((engine, task), []).append(idx)

        results: "queue.Queue[tuple[int, dict] | Exception | None]" = queue.Queue()
        driver = str(PROJECT_ROOT / "src" / "jope" / "remote_eval_driver.py")

        def run_host(engine: str, task: str, host: str, indices: list[int]) -> None:
            try:
                cfg = self.configs[engine]
                bin_path = str(cfg.release_binary)
                # SEPARATE processes on the node (not in-process rayon threads): the eval
                # functions in newmath-core hold non-thread-safe global state and panic
                # under in-process parallelism, so process isolation is the safe fast path.
                subcommand = EVAL_SUBCOMMAND[task]
                payload = "".join(json.dumps({**rows[i], "_idx": i}) + "\n" for i in indices)
                # compute-saturation guard: attenuate workers on an oversaturated REMOTE host
                # (never `local` — that's the v33 primary node). Fail-open on probe error.
                from .cluster_monitor import eval_worker_multiplier
                mult = eval_worker_multiplier(host)
                host_threads = max(1, round(threads_per_host * mult))
                if mult < 1.0:
                    print(f"[eval-guard] {host} saturated → workers {threads_per_host}→{host_threads}", flush=True)
                base = ["python3", driver, "--bin", bin_path, "--subcommand", subcommand,
                        "--workers", str(host_threads)]
                if self.timeouts_enabled:
                    # per-genome cost classes (e.g. portfolio=600s) must survive the batch
                    # path: use the max class across this batch, not the engine default.
                    batch_timeout = max(
                        (self.timeout_for(engine, str(rows[i].get("genome") or "")) for i in indices),
                        default=self.configs[engine].default_timeout_s,
                    )
                    base += ["--default-timeout", str(batch_timeout)]
                if host not in ("local", ""):
                    remote = " ".join(shlex.quote(c) for c in base)
                    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
                           "-o", f"ControlPath={ssh_control_path(host)}", "-o", "ControlPersist=600", host, remote]
                else:
                    cmd = base
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True)
                out, _ = proc.communicate(payload)
                for line in out.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = r.pop("_idx", None)
                    if idx is not None:
                        r.setdefault("engine", engine)
                        results.put((int(idx), r))
            except Exception as exc:
                results.put(exc)
            finally:
                results.put(None)

        host_jobs: list[tuple[str, str, str, list[int]]] = []
        for (engine, task), indices in grouped.items():
            live = [h for h in hosts]
            buckets: list[list[int]] = [[] for _ in live]
            for pos, idx in enumerate(indices):
                buckets[pos % len(live)].append(idx)
            for host, bucket in zip(live, buckets):
                if bucket:
                    host_jobs.append((engine, task, host, bucket))

        threads = [threading.Thread(target=run_host, args=job, daemon=True) for job in host_jobs]
        for t in threads:
            t.start()
        finished = 0
        while finished < len(threads):
            item = results.get()
            if item is None:
                finished += 1
                continue
            if isinstance(item, Exception):
                raise item
            yield item

    def iter_eval_regression_distributed(
        self,
        rows: Sequence[dict],
        *,
        workers_per_host: int = 16,
        hosts: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[int, dict]]:
        """Fan evaluation out across cluster hosts (all CPUs).

        Rows are sharded round-robin per engine into (host x worker) lanes; each lane
        runs ONE persistent bridge process (over ssh for remote hosts) in lock-step
        with per-row timeouts, so the ssh/process cost amortizes over the whole run.
        Yields (original_index, result) as lanes complete rows.
        """
        hosts = tuple(hosts) if hosts else self.hosts
        grouped: dict[tuple[str, str], list[int]] = {}
        for idx, row in enumerate(rows):
            engine = self._check_engine(str(row.get("engine") or DEFAULT_ENGINE))
            task = str(row.get("task") or DEFAULT_TASK)
            grouped.setdefault((engine, task), []).append(idx)

        lanes: list[tuple[str, str, str, list[int]]] = []  # (engine, task, host, row indices)
        for (engine, task), indices in grouped.items():
            lane_count = max(1, len(hosts) * workers_per_host)
            lane_count = min(lane_count, len(indices)) or 1
            buckets: list[list[int]] = [[] for _ in range(lane_count)]
            for pos, idx in enumerate(indices):
                buckets[pos % lane_count].append(idx)
            for lane_idx, bucket in enumerate(buckets):
                if bucket:
                    lanes.append((engine, task, hosts[lane_idx % len(hosts)], bucket))

        results: queue.Queue[tuple[int, dict] | Exception | None] = queue.Queue()

        def run_lane(engine: str, task: str, host: str, indices: list[int]) -> None:
            try:
                validator = self.validator(engine, host)
                lane_rows = [dict(rows[i]) for i in indices]

                def timeout_for(row: dict) -> float | None:
                    if not self.timeouts_enabled:
                        return None
                    genome = str(row.get("genome") or row.get("canonical_genome") or "")
                    return self.timeout_for(engine, genome)

                for local_idx, out in enumerate(
                    validator.iter_eval_many(
                        lane_rows, subcommand=EVAL_SUBCOMMAND[task], timeout_for=timeout_for
                    )
                ):
                    results.put((indices[local_idx], out))
            except Exception as exc:  # surface lane failures instead of hanging
                results.put(exc)
            finally:
                results.put(None)

        threads = [
            threading.Thread(target=run_lane, args=lane, daemon=True) for lane in lanes
        ]
        for thread in threads:
            thread.start()
        finished = 0
        while finished < len(threads):
            item = results.get()
            if item is None:
                finished += 1
                continue
            if isinstance(item, Exception):
                raise item
            yield item

    def iter_eval_regression_many(self, rows: Sequence[dict]) -> Iterator[tuple[int, dict]]:
        """Evaluate rows (each carrying ``engine``) with per-row timeouts.

        Yields ``(original_index, result)`` as results complete, one engine stream at
        a time (engines are processed sequentially; order within an engine is stable).
        """
        grouped: dict[tuple[str, str], list[int]] = {}
        for idx, row in enumerate(rows):
            engine = self._check_engine(str(row.get("engine") or DEFAULT_ENGINE))
            task = str(row.get("task") or DEFAULT_TASK)
            grouped.setdefault((engine, task), []).append(idx)
        for (engine, task), indices in grouped.items():
            engine_rows = [dict(rows[i]) for i in indices]
            if engine == "prime":
                if task != TASK_REGRESSION or self.prime_search is None:
                    raise ValueError("Prime native evaluation is regression-only")
                for local_idx, result in self.prime_search.evaluate_rows(engine_rows):
                    yield indices[local_idx], result
                continue
            validator = self.validator(engine)
            explicit = [row.pop("_timeout_s", None) for row in engine_rows]

            def timeout_for(row: dict, _explicit=explicit, _rows=engine_rows, _engine=engine) -> float | None:
                if not self.timeouts_enabled:
                    return None
                idx = next((i for i, r in enumerate(_rows) if r is row), None)
                if idx is not None and _explicit[idx] is not None:
                    return float(_explicit[idx])
                genome = str(row.get("genome") or row.get("canonical_genome") or "")
                return self.timeout_for(_engine, genome)

            for local_idx, result in enumerate(
                validator.iter_eval_many(engine_rows, subcommand=EVAL_SUBCOMMAND[task], timeout_for=timeout_for)
            ):
                yield indices[local_idx], result

    def _check_engine(self, engine: str) -> str:
        if engine == "prime":
            if self.prime_search is None:
                raise ValueError("Prime rows require a configured native evaluator")
            return engine
        if engine not in self.configs:
            raise ValueError(f"unknown engine {engine!r}; known: {sorted(self.engines())}")
        return engine


_DEFAULT_ROUTER: EngineRouter | None = None


def default_router() -> EngineRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = EngineRouter()
    return _DEFAULT_ROUTER
