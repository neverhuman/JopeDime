"""Native Prime search control plane.

Python only batches data and genes over the evaluator protocol. Candidate
semantics execute inside the persistent Rust/CUDA service; portable source is
never compiled or imported by this module. Only an exact-promoted winner can be
exported, and that public export is deliberately restricted to Python.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .grammars.prime import entry_version, registry_hash
from .prime_config import PrimeConfig, load_config
from .prime_evaluator_client import (
    PROTOCOL_VERSION,
    EvaluatorTimeoutError,
    EvaluatorTransportError,
    PrimeEvaluatorClient,
    UnknownDatasetError,
)


class NativeSearchError(RuntimeError):
    """The native evaluator could not honor the requested search contract."""


@dataclass
class _RegisteredDataset:
    signature: tuple[int, int]
    dataset_id: str
    rows: int
    columns: int
    folds: dict[tuple[int, int, str | None], str] = field(default_factory=dict)


def _parse_cell(cell: str) -> float:
    text = cell.strip()
    if not text or "_" in text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _looks_numeric(cell: str) -> bool:
    text = cell.strip()
    if not text:
        return True
    if "_" in text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _read_labeled_csv(path: Path) -> tuple[list[float], list[float], int, int]:
    """Mirror the owned Rust bridge's simple numeric CSV ingestion contract."""
    records = [
        line.split(",")
        for line in path.read_text(encoding="utf-8").splitlines()
        if any(cell.strip() for cell in line.split(","))
    ]
    if records and not all(_looks_numeric(cell) for cell in records[0]):
        records.pop(0)
    if not records:
        raise ValueError(f"{path}: CSV is empty")
    if any(len(row) < 2 for row in records):
        raise ValueError(f"{path}: labeled rows need a feature and target")
    columns = max(len(row) - 1 for row in records)
    if columns <= 0:
        raise ValueError(f"{path}: no feature columns")
    features: list[float] = []
    target: list[float] = []
    for row in records:
        values = [_parse_cell(cell) for cell in row]
        target.append(values[-1])
        feature_row = values[:-1]
        feature_row.extend([0.0] * (columns - len(feature_row)))
        features.extend(feature_row[:columns])
    return features, target, len(records), columns


def _assignments_key(assignments: Sequence[int] | None) -> str | None:
    if assignments is None:
        return None
    return json.dumps(list(assignments), separators=(",", ":"))


def _successful_payload(result: Mapping[str, Any]) -> dict[str, Any] | None:
    outcome = result.get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("status") != "success":
        return None
    payload = outcome.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else None


_NATIVE_RANKING_OBJECTIVE = "robust_oof_r2_v1"


def _native_ranking(response: Mapping[str, Any], *, context: str) -> list[tuple[str, float]]:
    if response.get("cancelled") is not False:
        raise NativeSearchError(f"{context} was cancelled")
    ranking = response.get("ranking")
    if not isinstance(ranking, Mapping):
        raise NativeSearchError(f"{context} omitted its native ranking")
    if ranking.get("objective") != _NATIVE_RANKING_OBJECTIVE:
        raise NativeSearchError(f"{context} returned an unknown ranking objective")
    candidates = ranking.get("candidates")
    if not isinstance(candidates, list):
        raise NativeSearchError(f"{context} returned malformed ranking candidates")
    output: list[tuple[str, float]] = []
    seen: set[str] = set()
    previous: float | None = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise NativeSearchError(f"{context} returned a malformed ranking entry")
        candidate_id = candidate.get("candidate_id")
        score = candidate.get("robust_score")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise NativeSearchError(f"{context} returned an invalid ranked candidate id")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise NativeSearchError(f"{context} returned a non-numeric native score")
        score = float(score)
        if not math.isfinite(score):
            raise NativeSearchError(f"{context} returned a non-finite native score")
        if previous is not None and score > previous:
            raise NativeSearchError(f"{context} ranking is not descending")
        seen.add(candidate_id)
        output.append((candidate_id, score))
        previous = score
    return output


class PrimeNativeSearch:
    """Batch genes through the persistent compiled evaluator.

    ``backend="auto"`` means CUDA, then parallel fast Rust CPU, then exact Rust
    according to the daemon's recorded fallback policy. ``require_cuda=True``
    turns any CUDA-eligible fallback into a hard campaign failure while still
    permitting an explicitly CPU-only baseline in a mixed request.
    """

    def __init__(
        self,
        client: PrimeEvaluatorClient,
        *,
        backend: str | None = None,
        require_cuda: bool = False,
        request_timeout: float | None = None,
    ) -> None:
        backend = backend or "auto"
        if backend not in {"auto", "fast_cuda", "fast_cpu", "exact"}:
            raise ValueError(f"unsupported Prime search backend {backend!r}")
        self.client = client
        self.backend = backend
        self.require_cuda = bool(require_cuda)
        self.request_timeout = request_timeout
        self._datasets: dict[Path, _RegisteredDataset] = {}
        capabilities = client.capabilities
        fast_cuda_backend = client.fast_cuda_backend
        if self._cuda_available(capabilities) != (fast_cuda_backend is not None):
            raise NativeSearchError(
                "evaluator FastCuda capability and backend identity disagree"
            )
        self._fast_cuda_backend = (
            json.loads(json.dumps(fast_cuda_backend))
            if fast_cuda_backend is not None
            else None
        )
        maximum = capabilities.get("max_batch_size")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise NativeSearchError("evaluator advertised an invalid maximum batch size")
        self.max_batch_size = maximum
        if self.require_cuda and not self._cuda_available(capabilities):
            raise NativeSearchError("strict CUDA search requested but FastCuda is unavailable")

    @classmethod
    def connect(
        cls,
        socket_path: str | Path | None = None,
        *,
        config: PrimeConfig | None = None,
        backend: str | None = None,
        require_cuda: bool = False,
        request_timeout: float = 300.0,
    ) -> PrimeNativeSearch:
        config = config or load_config()
        backend = backend or config.discovery_backend
        path = Path(socket_path) if socket_path is not None else config.cache / "battle-evald.sock"
        client = PrimeEvaluatorClient(
            path,
            prime_root=config.root,
            expected_registry_hash=registry_hash(),
            expected_registry_version=entry_version(),
            request_timeout=request_timeout,
        )
        return cls(
            client,
            backend=backend,
            require_cuda=require_cuda,
            request_timeout=request_timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> PrimeNativeSearch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _cuda_available(capabilities: Mapping[str, Any]) -> bool:
        for backend in capabilities.get("backends", []):
            if (
                isinstance(backend, Mapping)
                and backend.get("selection") == "fast_cuda"
                and backend.get("available") is True
            ):
                return True
        return False

    @staticmethod
    def _provenance_integer(
        provenance: Mapping[str, Any],
        field: str,
        *,
        bits: int,
        positive: bool,
    ) -> int:
        value = provenance.get(field)
        minimum = 1 if positive else 0
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not minimum <= value < (1 << bits)
        ):
            qualifier = "positive " if positive else "nonnegative "
            raise NativeSearchError(
                f"native provenance {field} must be a {qualifier}u{bits} integer"
            )
        return value

    def _validate_result_provenance(
        self,
        provenance: Mapping[str, Any],
        *,
        expected_request_size: int,
    ) -> tuple[int, int, int, int]:
        if provenance.get("protocol_version") != PROTOCOL_VERSION:
            raise NativeSearchError(
                "successful native result did not attest protocol v4 provenance"
            )
        tier = provenance.get("numerical_tier")
        backend_value = provenance.get("backend")
        if not isinstance(backend_value, Mapping):
            raise NativeSearchError("successful native result omitted its backend identity")
        backend = dict(backend_value)
        if tier == "fast_cuda":
            if self._fast_cuda_backend is None:
                raise NativeSearchError(
                    "FastCuda result has no handshaken CUDA backend identity"
                )
            try:
                backend_wire = json.dumps(
                    backend, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                expected_wire = json.dumps(
                    self._fast_cuda_backend,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as error:
                raise NativeSearchError(
                    "FastCuda result returned a malformed backend identity"
                ) from error
            if backend_wire != expected_wire:
                raise NativeSearchError(
                    "FastCuda result backend differs from the handshaken identity"
                )
        elif backend.get("kind") == "cuda" or "execution_identity" in backend:
            raise NativeSearchError(
                f"{tier!r} result must not claim a CUDA execution identity"
            )
        request_elapsed = self._provenance_integer(
            provenance, "request_elapsed_micros", bits=64, positive=False
        )
        request_size = self._provenance_integer(
            provenance, "request_batch_size", bits=32, positive=True
        )
        backend_elapsed = self._provenance_integer(
            provenance, "batch_elapsed_micros", bits=64, positive=False
        )
        backend_size = self._provenance_integer(
            provenance, "execution_batch_size", bits=32, positive=True
        )
        if request_size != expected_request_size:
            raise NativeSearchError(
                "native provenance request_batch_size differs from the request"
            )
        if backend_size > request_size:
            raise NativeSearchError(
                "native backend cohort exceeds the whole request size"
            )
        if backend_elapsed > request_elapsed:
            raise NativeSearchError(
                "native backend cohort elapsed time exceeds the whole request"
            )
        return request_elapsed, request_size, backend_elapsed, backend_size

    def _register_dataset(self, train_csv: Path, *, force: bool = False) -> _RegisteredDataset:
        path = train_csv.expanduser().resolve(strict=True)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._datasets.get(path)
        if not force and cached is not None and cached.signature == signature:
            return cached
        features, target, rows, columns = _read_labeled_csv(path)
        response = self.client.register_dataset(
            task="regression",
            rows=rows,
            columns=columns,
            features=features,
            target=target,
            claimed_content_hash=None,
            schema=None,
            timeout=self.request_timeout,
        )
        dataset_id = response.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise NativeSearchError("evaluator returned no dataset identity")
        registered = _RegisteredDataset(signature, dataset_id, rows, columns)
        self._datasets[path] = registered
        return registered

    def _register_fold(
        self,
        train_csv: Path,
        *,
        folds: int,
        seed: int,
        assignments: Sequence[int] | None,
        force: bool = False,
    ) -> tuple[_RegisteredDataset, str]:
        registered = self._register_dataset(train_csv, force=force)
        key = (folds, seed, _assignments_key(assignments))
        if not force and key in registered.folds:
            return registered, registered.folds[key]
        response = self.client.register_fold_plan(
            dataset_id=registered.dataset_id,
            k=folds,
            assignments=assignments,
            seed=seed,
            timeout=self.request_timeout,
        )
        fold_id = response.get("fold_plan_id")
        if not isinstance(fold_id, str) or not fold_id:
            raise NativeSearchError("evaluator returned no fold-plan identity")
        registered.folds[key] = fold_id
        return registered, fold_id

    def _register_fold_retry(
        self,
        train_csv: Path,
        *,
        folds: int,
        seed: int,
        assignments: Sequence[int] | None,
    ) -> tuple[_RegisteredDataset, str]:
        try:
            return self._register_fold(
                train_csv,
                folds=folds,
                seed=seed,
                assignments=assignments,
            )
        except UnknownDatasetError:
            return self._register_fold(
                train_csv,
                folds=folds,
                seed=seed,
                assignments=assignments,
                force=True,
            )

    def _recover_evaluator(self, error: EvaluatorTransportError) -> None:
        if isinstance(error, EvaluatorTimeoutError):
            raise error
        try:
            self.client.reconnect()
        except Exception as reconnect_error:
            raise NativeSearchError(
                f"native evaluator reconnect failed: {reconnect_error}"
            ) from reconnect_error
        self._datasets.clear()

    def evaluate_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        suite_hash: str | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Evaluate Prime rows without invoking any portable artifact."""
        return self._evaluate_rows(
            rows,
            seed=seed,
            suite_hash=suite_hash,
            batch_size=batch_size,
            native_ranked=False,
        )

    def evaluate_ranked_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        suite_hash: str | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return one server-ranked cohort without Python score merging."""
        return self._evaluate_rows(
            rows,
            seed=seed,
            suite_hash=suite_hash,
            batch_size=None,
            native_ranked=True,
        )

    def _evaluate_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        suite_hash: str | None,
        batch_size: int | None,
        native_ranked: bool,
    ) -> list[tuple[int, dict[str, Any]]]:
        limit = self.max_batch_size if batch_size is None else min(batch_size, self.max_batch_size)
        if limit <= 0:
            raise ValueError("batch_size must be positive")
        grouped: dict[tuple[Path, int, int, str | None], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(rows):
            if str(row.get("engine") or "prime") != "prime":
                raise ValueError("PrimeNativeSearch accepts only engine='prime' rows")
            train_csv = row.get("train_csv")
            genome = row.get("genome") or row.get("canonical_genome")
            if not isinstance(train_csv, str) or not isinstance(genome, str) or not genome:
                raise ValueError("each Prime row needs train_csv and genome strings")
            folds = int(row.get("folds", 5))
            row_seed = int(row.get("fold_seed", seed))
            assignments = row.get("fold_assignments")
            if assignments is not None and not isinstance(assignments, Sequence):
                raise ValueError("fold_assignments must be a sequence")
            key = (
                Path(train_csv).expanduser().resolve(strict=True),
                folds,
                row_seed,
                _assignments_key(assignments),
            )
            grouped[key].append((index, row))

        if native_ranked:
            if len(grouped) > 1:
                raise ValueError("a native-ranked cohort must use one dataset and fold plan")
            if len(rows) > self.max_batch_size:
                raise NativeSearchError(
                    "native ranking cannot merge chunk-local scores; submit a cohort "
                    "within the evaluator maximum batch size"
                )

        output: list[tuple[int, dict[str, Any]]] = []
        for (train_csv, folds, row_seed, assignments_json), members in grouped.items():
            assignments = json.loads(assignments_json) if assignments_json is not None else None
            try:
                registered, fold_id = self._register_fold_retry(
                    train_csv, folds=folds, seed=row_seed, assignments=assignments
                )
            except EvaluatorTransportError as error:
                self._recover_evaluator(error)
                registered, fold_id = self._register_fold_retry(
                    train_csv, folds=folds, seed=row_seed, assignments=assignments
                )
            chunks = [members] if native_ranked else [
                members[offset : offset + limit]
                for offset in range(0, len(members), limit)
            ]
            for chunk in chunks:
                try:
                    response = self._evaluate_chunk(
                        chunk,
                        registered.dataset_id,
                        fold_id,
                        seed=seed,
                        suite_hash=suite_hash,
                    )
                except UnknownDatasetError:
                    registered, fold_id = self._register_fold(
                        train_csv,
                        folds=folds,
                        seed=row_seed,
                        assignments=assignments,
                        force=True,
                    )
                    response = self._evaluate_chunk(
                        chunk,
                        registered.dataset_id,
                        fold_id,
                        seed=seed,
                        suite_hash=suite_hash,
                    )
                except EvaluatorTransportError as error:
                    self._recover_evaluator(error)
                    registered, fold_id = self._register_fold_retry(
                        train_csv,
                        folds=folds,
                        seed=row_seed,
                        assignments=assignments,
                    )
                    response = self._evaluate_chunk(
                        chunk,
                        registered.dataset_id,
                        fold_id,
                        seed=seed,
                        suite_hash=suite_hash,
                    )
                bridged = self._bridge_results(chunk, response)
                if native_ranked:
                    output.extend(self._native_order(chunk, bridged, response))
                else:
                    output.extend(bridged)
        if not native_ranked:
            output.sort(key=lambda item: item[0])
        return output

    def _evaluate_chunk(
        self,
        chunk: Sequence[tuple[int, Mapping[str, Any]]],
        dataset_id: str,
        fold_id: str,
        *,
        seed: int,
        suite_hash: str | None,
    ) -> dict[str, Any]:
        candidates = [
            {
                "candidate_id": str(index),
                "genome": str(row.get("genome") or row.get("canonical_genome")),
                "claimed_plan_hash": row.get("semantic_plan_hash"),
            }
            for index, row in chunk
        ]
        return self.client.evaluate_batch(
            job_id=f"prime-search-{uuid.uuid4().hex}",
            dataset_id=dataset_id,
            fold_plan_id=fold_id,
            candidates=candidates,
            policy={
                "backend": self.backend,
                "seed": seed,
                "exact_gate": "not_requested",
                "compiled_only": True,
                "suite_hash": suite_hash,
                "return_fold_scores": True,
            },
            timeout=self.request_timeout,
        )

    def _bridge_results(
        self,
        chunk: Sequence[tuple[int, Mapping[str, Any]]],
        response: Mapping[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        ranking = dict(_native_ranking(response, context="native evaluation"))
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise NativeSearchError("native evaluation returned malformed results")
        by_id: dict[str, Mapping[str, Any]] = {}
        for result in raw_results:
            if not isinstance(result, Mapping):
                raise NativeSearchError("native evaluation returned a malformed result")
            candidate_id = result.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in by_id
            ):
                raise NativeSearchError("native evaluation returned duplicate candidate ids")
            by_id[candidate_id] = result
        expected_ids = {str(index) for index, _ in chunk}
        if set(by_id) != expected_ids:
            raise NativeSearchError("native evaluation result ids differ from the request")
        if not set(ranking).issubset(expected_ids):
            raise NativeSearchError("native ranking referenced an unknown candidate")
        output: list[tuple[int, dict[str, Any]]] = []
        request_facts: tuple[int, int] | None = None
        for index, row in chunk:
            result = by_id.get(str(index), {})
            payload = _successful_payload(result)
            if payload is None:
                outcome = result.get("outcome") if isinstance(result, Mapping) else None
                output.append(
                    (
                        index,
                        {
                            "engine": "prime",
                            "dataset_hash": row.get("dataset_hash"),
                            "canonical_genome": row.get("genome") or row.get("canonical_genome"),
                            "valid": False,
                            "cache_hit": False,
                            "native_execution": True,
                            "cv": None,
                            "errors": [f"native evaluator outcome: {outcome!r}"],
                        },
                    )
                )
                continue
            provenance = payload.get("provenance")
            metrics = payload.get("metrics")
            if not isinstance(provenance, Mapping) or not isinstance(metrics, Mapping):
                raise NativeSearchError("successful result omitted metrics or provenance")
            if provenance.get("compiled_plan_execution") is not True:
                raise NativeSearchError(
                    "compiled-only native evaluation did not execute its CompiledPlan binding"
                )
            timing_facts = self._validate_result_provenance(
                provenance, expected_request_size=len(chunk)
            )
            result_request_facts = timing_facts[:2]
            if request_facts is None:
                request_facts = result_request_facts
            elif result_request_facts != request_facts:
                raise NativeSearchError(
                    "successful results disagree on whole-request provenance"
                )
            cuda_eligible = provenance.get("cuda_eligible")
            if not isinstance(cuda_eligible, bool):
                raise NativeSearchError("native evaluation omitted CUDA eligibility")
            if (
                self.require_cuda
                and cuda_eligible
                and provenance.get("numerical_tier") != "fast_cuda"
            ):
                raise NativeSearchError(
                    "strict CUDA search received an eligible "
                    f"{provenance.get('numerical_tier')!r}: {provenance.get('fallback')!r}"
                )
            values = metrics.get("values")
            if not isinstance(values, Mapping):
                raise NativeSearchError("successful result omitted metric values")
            native_score = ranking.get(str(index))
            metric_score = metrics.get("robust_score")
            if (
                native_score is None
                or isinstance(metric_score, bool)
                or not isinstance(metric_score, (int, float))
                or not math.isfinite(float(metric_score))
                or float(metric_score) != native_score
            ):
                raise NativeSearchError(
                    "native ranking score does not match the candidate metrics"
                )
            elapsed = provenance.get("elapsed_micros")
            evaluator_version = provenance.get("evaluator_version")
            if not isinstance(evaluator_version, str) or not evaluator_version:
                raise NativeSearchError("native evaluation omitted evaluator version")
            output.append(
                (
                    index,
                    {
                        "engine": "prime",
                        "dataset_hash": row.get("dataset_hash"),
                        "canonical_genome": payload.get("canonical_genome"),
                        "semantic_plan_hash": provenance.get("compiled_plan_hash"),
                        "valid": True,
                        "cache_hit": False,
                        "native_execution": True,
                        "evaluator_version": f"prime-native/{evaluator_version}",
                        "native_robust_score": native_score,
                        "cv": {
                            "oof_r2": values.get("oof_r2", metrics.get("primary_value")),
                            "fold_r2": list(metrics.get("fold_values") or []),
                        },
                        "timings": {
                            "total_ms": float(elapsed) / 1000.0
                            if isinstance(elapsed, (int, float))
                            else None,
                            "native_request_total_ms": timing_facts[0] / 1000.0,
                            "native_request_size": timing_facts[1],
                            "native_backend_batch_total_ms": timing_facts[2] / 1000.0,
                            "native_backend_batch_size": timing_facts[3],
                        },
                        "native_provenance": dict(provenance),
                    },
                )
            )
        return output

    @staticmethod
    def _native_order(
        chunk: Sequence[tuple[int, Mapping[str, Any]]],
        bridged: Sequence[tuple[int, dict[str, Any]]],
        response: Mapping[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        ranked = _native_ranking(response, context="native evaluation")
        expected = {str(index): index for index, _ in chunk}
        by_index = dict(bridged)
        ordered: list[tuple[int, dict[str, Any]]] = []
        seen: set[int] = set()
        previous_rank: tuple[float, int] | None = None
        for candidate_id, score in ranked:
            index = expected.get(candidate_id)
            if index is None or index in seen:
                raise NativeSearchError("native ranking referenced an unknown candidate")
            result = by_index.get(index)
            if result is None or not result.get("valid"):
                raise NativeSearchError("native ranking referenced an unsuccessful candidate")
            if previous_rank is not None and score == previous_rank[0] and index < previous_rank[1]:
                raise NativeSearchError("native ranking changed request order for an exact tie")
            result["native_robust_score"] = score
            ordered.append((index, result))
            seen.add(index)
            previous_rank = (score, index)
        valid_unranked = [
            index
            for index, result in bridged
            if result.get("valid") and index not in seen
        ]
        if valid_unranked:
            raise NativeSearchError(
                f"native ranking omitted successful candidates {valid_unranked!r}"
            )
        ordered.extend((index, result) for index, result in bridged if index not in seen)
        return ordered

    def promote_exact(
        self,
        row: Mapping[str, Any],
        *,
        seed: int = 0,
        suite_hash: str | None = None,
    ) -> dict[str, Any]:
        """Recompute one finalist in authoritative native exact f64."""
        ranked = self.promote_exact_rows([row], seed=seed, suite_hash=suite_hash)
        if len(ranked) != 1:
            raise NativeSearchError("single exact promotion returned the wrong result count")
        return ranked[0][1]

    def promote_exact_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        suite_hash: str | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Recompute one cohort and consume the server's exact native order."""
        if not rows:
            raise ValueError("exact promotion needs at least one candidate")
        if len(rows) > self.max_batch_size:
            raise NativeSearchError("exact promotion cohort exceeds evaluator batch capacity")
        normalized: list[tuple[Path, int, int, str | None, Mapping[str, Any]]] = []
        for row in rows:
            train_csv = row.get("train_csv")
            genome = row.get("genome") or row.get("canonical_genome")
            if not isinstance(train_csv, str) or not isinstance(genome, str) or not genome:
                raise ValueError("exact promotion needs train_csv and genome strings")
            folds = int(row.get("folds", 5))
            fold_seed = int(row.get("fold_seed", seed))
            assignments = row.get("fold_assignments")
            if assignments is not None and not isinstance(assignments, Sequence):
                raise ValueError("fold_assignments must be a sequence")
            normalized.append(
                (
                    Path(train_csv).expanduser().resolve(strict=True),
                    folds,
                    fold_seed,
                    _assignments_key(assignments),
                    row,
                )
            )
        group_keys = {item[:4] for item in normalized}
        if len(group_keys) != 1:
            raise ValueError("one exact promotion cohort must share dataset and folds")
        train_csv, folds, fold_seed, assignments_json = normalized[0][:4]
        assignments = json.loads(assignments_json) if assignments_json is not None else None
        registered, fold_id = self._register_fold_retry(
            train_csv,
            folds=folds,
            seed=fold_seed,
            assignments=assignments,
        )
        candidates = [
            {
                "candidate_id": str(index),
                "genome": str(row.get("genome") or row.get("canonical_genome")),
                "claimed_plan_hash": row.get("semantic_plan_hash"),
            }
            for index, (_, _, _, _, row) in enumerate(normalized)
        ]
        promotion_job_id = f"prime-exact-{uuid.uuid4().hex}"

        def submit() -> dict[str, Any]:
            return self.client.promote_exact(
                job_id=promotion_job_id,
                source_job_id=None,
                dataset_id=registered.dataset_id,
                fold_plan_id=fold_id,
                candidates=candidates,
                seed=seed,
                compiled_only=True,
                suite_hash=suite_hash,
                return_fold_scores=True,
                timeout=self.request_timeout,
            )

        try:
            response = submit()
        except UnknownDatasetError:
            registered, fold_id = self._register_fold(
                train_csv,
                folds=folds,
                seed=fold_seed,
                assignments=assignments,
                force=True,
            )
            response = submit()

        ranking = _native_ranking(response, context="exact promotion")
        raw_results = response.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(rows):
            raise NativeSearchError("exact promotion returned an invalid result set")
        by_id: dict[str, Mapping[str, Any]] = {}
        for result in raw_results:
            if not isinstance(result, Mapping):
                raise NativeSearchError("exact promotion returned a malformed result")
            candidate_id = result.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or candidate_id in by_id
                or candidate_id not in {str(index) for index in range(len(rows))}
            ):
                raise NativeSearchError("exact promotion returned an invalid candidate id")
            by_id[candidate_id] = result
        if len(ranking) != len(rows):
            raise NativeSearchError("exact promotion did not rank every finalist")

        output: list[tuple[int, dict[str, Any]]] = []
        seen: set[str] = set()
        previous_rank: tuple[float, int] | None = None
        request_facts: tuple[int, int] | None = None
        for candidate_id, native_score in ranking:
            if candidate_id in seen or candidate_id not in by_id:
                raise NativeSearchError("exact promotion ranking referenced an invalid candidate")
            result = by_id[candidate_id]
            payload = _successful_payload(result)
            if payload is None:
                raise NativeSearchError(f"exact promotion failed: {result!r}")
            provenance = payload.get("provenance")
            metrics = payload.get("metrics")
            if not isinstance(provenance, Mapping):
                raise NativeSearchError("exact promotion omitted provenance")
            if not isinstance(metrics, Mapping):
                raise NativeSearchError("exact promotion omitted metrics")
            metric_score = metrics.get("robust_score")
            if (
                isinstance(metric_score, bool)
                or not isinstance(metric_score, (int, float))
                or not math.isfinite(float(metric_score))
                or float(metric_score) != native_score
            ):
                raise NativeSearchError(
                    "exact native ranking score does not match candidate metrics"
                )
            if provenance.get("compiled_plan_execution") is not True:
                raise NativeSearchError("exact promotion bypassed the CompiledPlan binding")
            timing_facts = self._validate_result_provenance(
                provenance, expected_request_size=len(rows)
            )
            result_request_facts = timing_facts[:2]
            if request_facts is None:
                request_facts = result_request_facts
            elif result_request_facts != request_facts:
                raise NativeSearchError(
                    "exact results disagree on whole-request provenance"
                )
            if provenance.get("numerical_tier") != "exact_f64":
                raise NativeSearchError("exact promotion did not execute exact f64")
            if provenance.get("exact_gate") != "passed":
                raise NativeSearchError("exact promotion did not pass its exact gate")
            index = int(candidate_id)
            if (
                previous_rank is not None
                and native_score == previous_rank[0]
                and index < previous_rank[1]
            ):
                raise NativeSearchError(
                    "exact native ranking changed request order for an exact tie"
                )
            payload = dict(payload)
            payload["_native_robust_score"] = native_score
            payload["_promotion_job_id"] = promotion_job_id
            payload["_promotion_candidate_id"] = candidate_id
            payload["_promotion_dataset_id"] = registered.dataset_id
            payload["_promotion_fold_plan_id"] = fold_id
            payload["_promotion_fold_seed"] = fold_seed
            output.append((index, payload))
            seen.add(candidate_id)
            previous_rank = (native_score, index)
        return output

    def emit_user_python(
        self,
        row: Mapping[str, Any],
        exact_result: Mapping[str, Any],
        *,
        include_test_vectors: bool = True,
    ) -> dict[str, Any]:
        """Emit the sole user-facing artifact after exact native promotion."""
        provenance = exact_result.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("exact_result has no provenance")
        if (
            provenance.get("numerical_tier") != "exact_f64"
            or provenance.get("exact_gate") != "passed"
            or provenance.get("compiled_plan_execution") is not True
        ):
            raise ValueError("winner must pass exact native promotion before emission")
        train_csv = row.get("train_csv")
        genome = row.get("genome") or row.get("canonical_genome")
        if not isinstance(train_csv, str) or not isinstance(genome, str):
            raise ValueError("winner needs train_csv and genome")
        folds = int(row.get("folds", 5))
        promoted_fold_seed = exact_result.get("_promotion_fold_seed")
        if not isinstance(promoted_fold_seed, int) or isinstance(promoted_fold_seed, bool):
            raise ValueError("exact_result has no promoted fold seed")
        fold_seed = int(row.get("fold_seed", promoted_fold_seed))
        if fold_seed != promoted_fold_seed:
            raise ValueError("winner fold seed differs from its promotion receipt")
        assignments = row.get("fold_assignments")
        registered, fold_id = self._register_fold_retry(
            Path(train_csv), folds=folds, seed=fold_seed, assignments=assignments
        )
        plan_hash = provenance.get("compiled_plan_hash")
        if not isinstance(plan_hash, str):
            raise ValueError("exact_result has no compiled-plan hash")
        promotion_job_id = exact_result.get("_promotion_job_id")
        if not isinstance(promotion_job_id, str) or not promotion_job_id:
            raise ValueError("exact_result has no server promotion receipt")
        candidate_id = exact_result.get("_promotion_candidate_id")
        promoted_dataset_id = exact_result.get("_promotion_dataset_id")
        promoted_fold_id = exact_result.get("_promotion_fold_plan_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("exact_result has no promoted candidate identity")
        if (
            not isinstance(promoted_dataset_id, str)
            or not isinstance(promoted_fold_id, str)
            or registered.dataset_id != promoted_dataset_id
            or fold_id != promoted_fold_id
        ):
            raise NativeSearchError("winner registration differs from its promotion receipt")

        def compile_once() -> dict[str, Any]:
            return self.client.compile_artifact(
                job_id=f"prime-python-{uuid.uuid4().hex}",
                candidate={
                    "candidate_id": candidate_id,
                    "genome": genome,
                    "claimed_plan_hash": plan_hash,
                },
                source_promotion_job_id=promotion_job_id,
                artifact_kinds=["portable_python"],
                dataset_id=registered.dataset_id,
                fold_plan_id=fold_id,
                include_test_vectors=include_test_vectors,
                timeout=self.request_timeout,
            )

        try:
            response = compile_once()
        except UnknownDatasetError:
            registered, fold_id = self._register_fold(
                Path(train_csv),
                folds=folds,
                seed=fold_seed,
                assignments=assignments,
                force=True,
            )
            if registered.dataset_id != promoted_dataset_id or fold_id != promoted_fold_id:
                raise NativeSearchError(
                    "re-registered winner differs from its promotion receipt"
                )
            response = compile_once()
        if (
            response.get("candidate_id") != candidate_id
            or response.get("plan_hash") != plan_hash
        ):
            raise NativeSearchError("Python emission response identity does not match winner")
        bundles = response.get("bundles")
        if not isinstance(bundles, list) or len(bundles) != 1:
            raise NativeSearchError("Python emission returned an invalid bundle set")
        bundle = bundles[0]
        if not isinstance(bundle, Mapping) or bundle.get("kind") != "portable_python":
            raise NativeSearchError("winner export was not portable Python")
        return dict(bundle)


__all__ = ["NativeSearchError", "PrimeNativeSearch"]
