from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .engines import DEFAULT_ENGINE, EVALUATOR_VERSIONS, EngineRouter
from .validator import RustValidator

# Legacy alias: the featmath evaluator version (kept for old imports/tests).
EVALUATOR_VERSION = EVALUATOR_VERSIONS[DEFAULT_ENGINE]
GENOME_KEYS = ("genome", "gene", "dsl", "dsl_expr", "canonical_genome")


def eval_regression_with_ledger(
    rows: Iterable[dict[str, Any]],
    *,
    ledger: str | Path | None,
    validator: RustValidator | None = None,
    router: EngineRouter | None = None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    return [
        out
        for _, out in iter_eval_regression_with_ledger(
            rows, ledger=ledger, validator=validator, router=router, workers=workers
        )
    ]


def iter_eval_regression_with_ledger(
    rows: Iterable[dict[str, Any]],
    *,
    ledger: str | Path | None,
    validator: RustValidator | None = None,
    router: EngineRouter | None = None,
    workers: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Ledger-cached evaluation, engine-aware.

    Pass ``router`` for multi-engine rows (each carrying an ``engine`` key). The legacy
    ``validator`` kwarg keeps single-engine (featmath) callers working unchanged.
    """
    rows = list(rows)
    if any(row.get("engine") == "prime" for row in rows) and (
        router is None or router.prime_search is None
    ):
        raise ValueError("Prime rows require a configured native evaluator")
    if router is None and validator is None:
        validator = RustValidator()
    worker_count = _resolve_worker_count(workers)
    if ledger is None:
        if router is not None:
            yield from _iter_eval(rows, validator=validator, router=router, workers=worker_count)
            return
        if worker_count > 1:
            yield from _iter_eval_parallel(rows, validator=validator, router=router, workers=worker_count)
            return
        yield from _iter_eval(rows, validator=validator, router=router)
        return

    ledger_path = Path(ledger)
    cached = _read_ledger(ledger_path)
    keys = _candidate_cache_keys(rows, router if router is not None else validator)
    misses: list[dict[str, Any]] = []
    miss_indices: list[int] = []
    cache_hits: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = keys.get(idx)
        hit = cached.get(key) if key is not None else None
        if hit is not None:
            out = copy.deepcopy(hit)
            out["cache_hit"] = True
            cache_hits[idx] = out
        else:
            misses.append(row)
            miss_indices.append(idx)

    if worker_count > 1:
        for idx, hit in cache_hits.items():
            yield idx, hit
        if router is not None:
            miss_iterable = _iter_eval(misses, validator=None, router=router, workers=worker_count)
        else:
            miss_iterable = _iter_eval_parallel(misses, validator=validator, router=router, workers=worker_count)
        for local_idx, out in miss_iterable:
            idx = miss_indices[local_idx]
            _append_ledger(ledger_path, [out])
            yield idx, out
        return

    hit_indices = set(cache_hits)
    miss_results = _iter_eval(misses, validator=validator, router=router, workers=workers)
    buffered: dict[int, dict[str, Any]] = {}
    next_local = 0
    local_of = {idx: local for local, idx in enumerate(miss_indices)}
    for idx in range(len(rows)):
        if idx in hit_indices:
            yield idx, cache_hits[idx]
            continue
        want_local = local_of[idx]
        while want_local not in buffered:
            local_idx, out = next(miss_results)
            buffered[local_idx] = out
        out = buffered.pop(want_local)
        _append_ledger(ledger_path, [out])
        yield idx, out
        next_local += 1


def _iter_eval(
    rows: list[dict[str, Any]],
    *,
    validator: RustValidator | None,
    router: EngineRouter | None,
    workers: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not rows:
        return
    if router is not None:
        prime_indices = [
            index for index, row in enumerate(rows) if row.get("engine") == "prime"
        ]
        if prime_indices:
            if router.prime_search is None:
                raise ValueError("Prime rows require a configured native evaluator")
            prime_rows = [rows[index] for index in prime_indices]
            for local_index, result in router.prime_search.evaluate_rows(prime_rows):
                yield prime_indices[local_index], result
            legacy_indices = [
                index for index, row in enumerate(rows) if row.get("engine") != "prime"
            ]
            if not legacy_indices:
                return
            legacy_rows = [rows[index] for index in legacy_indices]
            if len(router.hosts) > 1 or (workers or 0) > 1:
                for local_index, result in router.iter_eval_via_host_drivers(
                    legacy_rows, workers_per_host=max(1, workers or 90)
                ):
                    yield legacy_indices[local_index], result
            else:
                for local_index, result in router.iter_eval_regression_many(legacy_rows):
                    yield legacy_indices[local_index], result
            return
        # Cluster fan-out: ship each host its shard over ONE channel and parallelize on the
        # node via separate bridge PROCESSES (host-local driver) — ~5-6x the old ssh
        # lock-step lanes, which the head-node GIL + ssh latency capped at ~8-way.
        if len(router.hosts) > 1 or (workers or 0) > 1:
            yield from router.iter_eval_via_host_drivers(
                rows, workers_per_host=max(1, workers or 90)
            )
            return
        yield from router.iter_eval_regression_many(rows)
        return
    assert validator is not None
    for idx, out in enumerate(validator.iter_eval_regression_many(rows)):
        yield idx, out


def _iter_eval_parallel(
    rows: list[dict[str, Any]],
    *,
    validator: RustValidator | None,
    router: EngineRouter | None,
    workers: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not rows:
        return
    max_workers = max(1, min(workers, len(rows)))

    def _eval_one(idx: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if router is not None:
            return idx, router.eval_one(row)
        assert validator is not None
        return idx, validator.eval_regression_many([row])[0]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, idx, row): idx for idx, row in enumerate(rows)}
        for future in as_completed(futures):
            yield future.result()


def _resolve_worker_count(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    raw = os.environ.get("JOPE_EVAL_WORKERS")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _candidate_cache_keys(
    rows: list[dict[str, Any]],
    validator: RustValidator | EngineRouter,
) -> dict[int, str]:
    """Python-side ledger lookup keys, byte-identical to the bridges' ``cache_key``.

    Engine disambiguation is total: each engine's ``evaluator_version`` string is part
    of the hashed text, and canonicalization runs through that engine's validator. A
    genome that happens to be valid in both grammars therefore gets distinct keys per
    engine (no cross-engine cache poisoning).
    """
    genomes: list[str | dict] = []
    indices: list[int] = []
    engines: list[str] = []
    tasks: list[str] = []
    train_hashes: dict[int, str] = {}
    test_hashes: dict[int, str] = {}
    for idx, row in enumerate(rows):
        genome = _extract_genome(row)
        train_csv = row.get("train_csv")
        if genome is None or not isinstance(train_csv, str):
            continue
        engine = str(row.get("engine") or _validator_engine(validator))
        # Prime's persistent daemon owns parsing, dataset identity, and fold
        # caching. Skip before any file I/O: the Python ledger intentionally
        # has no Prime key and must not hash a shared CSV once per candidate.
        if engine == "prime":
            continue
        try:
            train_hashes[idx] = hashlib.sha256(Path(train_csv).read_bytes()).hexdigest()
        except OSError:
            continue
        test_csv = row.get("test_csv")
        if isinstance(test_csv, str):
            try:
                test_hashes[idx] = hashlib.sha256(Path(test_csv).read_bytes()).hexdigest()
            except OSError:
                test_hashes[idx] = ""
        else:
            test_hashes[idx] = ""
        genomes.append(genome)
        indices.append(idx)
        engines.append(engine)
        tasks.append(str(row.get("task") or "regression"))

    keys: dict[int, str] = {}
    if not genomes:
        return keys
    if isinstance(validator, EngineRouter):
        metas = validator.validate_many(list(zip(engines, [g if isinstance(g, str) else str(g) for g in genomes])))
        # regression rows keep byte-identical keys (task defaults to regression); binary
        # rows land in disjoint key space via the eval-binary evaluator_version string.
        versions = [validator.evaluator_version(engine, task) for engine, task in zip(engines, tasks)]
    else:
        metas = validator.validate_many(genomes)
        version = EVALUATOR_VERSIONS.get(validator.engine, EVALUATOR_VERSION)
        versions = [version] * len(genomes)
    for idx, meta, version in zip(indices, metas, versions, strict=True):
        if not meta.get("valid"):
            continue
        canonical = meta.get("canonical_genome")
        if not isinstance(canonical, str):
            continue
        row = rows[idx]
        dataset_hash = str(row.get("dataset_hash") or row.get("dataset_id") or "unknown")
        folds = max(2, int(row.get("folds") or 3))
        text = (
            f"{dataset_hash}\n{canonical}\n{folds}\n{train_hashes[idx]}\n"
            f"{test_hashes.get(idx, '')}\n{version}"
        )
        keys[idx] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return keys


def _validator_engine(validator: RustValidator | EngineRouter) -> str:
    if isinstance(validator, EngineRouter):
        return DEFAULT_ENGINE
    return getattr(validator, "engine", DEFAULT_ENGINE)


def migrate_ledger_v3(path: str | Path) -> dict[str, Any]:
    """Annotate legacy ledger rows with an ``engine`` inferred from evaluator_version.

    Keys are untouched (they already embed the per-engine evaluator_version). Writes a
    ``.bak`` alongside; drops rows without a cache_key.
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"path": str(ledger_path), "rows": 0, "migrated": 0, "dropped": 0, "engines": {}}
    rows: list[dict[str, Any]] = []
    dropped = 0
    migrated = 0
    engines: dict[str, int] = {}
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not row.get("cache_key"):
                dropped += 1
                continue
            if not row.get("engine"):
                version = str(row.get("evaluator_version") or "")
                if version.startswith("jope-newmath-validate."):
                    row["engine"] = "newmath"
                else:
                    row["engine"] = "featmath"
                migrated += 1
            engines[row["engine"]] = engines.get(row["engine"], 0) + 1
            rows.append(row)
    backup = ledger_path.with_suffix(ledger_path.suffix + ".bak")
    ledger_path.replace(backup)
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "path": str(ledger_path),
        "backup": str(backup),
        "rows": len(rows),
        "migrated": migrated,
        "dropped": dropped,
        "engines": engines,
    }


def _read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("cache_key")
            if isinstance(key, str):
                out[key] = row
    return out


def _append_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row.get("cache_key"):
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _extract_genome(row: dict[str, Any]) -> str | None:
    for key in GENOME_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None
