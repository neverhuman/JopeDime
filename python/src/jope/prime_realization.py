"""Portable transform conformance and full-model realization evidence.

``RealizationLabel`` remains the CountSketch transform-conformance record.
``ModelRealizationLabel`` is separate: it runs a complete emitted regression
program, including fitting, on an explicit train-side holdout and fold plan.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .prime_artifacts import CompileArtifact, lint_portable_source
from .prime_config import PrimeConfig, load_config
from .prime_ops import fit_transform

REALIZATION_SCHEMA = "jope-prime-transform-conformance-v2"
MODEL_ARTIFACT_SCHEMA = "jope-prime-model-artifact-source-v2"
MODEL_REALIZATION_SCHEMA = "jope-prime-model-realization-v2"
TRAIN_SIDE_SPLITS = frozenset({"rest", "dev_a", "dev_b", "dev_c"})
PORTABLE_MODEL_TARGETS = frozenset({"portable_python", "portable_rust"})


@dataclass(frozen=True)
class ModelArtifactSource:
    """Versioned, integrity-addressed input for a standalone model program."""

    schema: str
    artifact_kind: str
    artifact_hash: str
    manifest_sha256: str
    test_vectors_sha256: str
    source: str
    source_sha256: str
    semantic_plan_hash: str
    plan_version: int | str
    registry_version: int | str
    registry_hash: str
    emitter_version: str
    canonical_genome: str | None
    numerical_tier: str
    input_columns: int
    seeds: dict[str, Any]
    operator_versions: dict[str, Any]
    backend_provenance: dict[str, Any]

    @classmethod
    def create(
        cls,
        source: str,
        *,
        artifact_kind: str,
        manifest_sha256: str,
        test_vectors_sha256: str,
        semantic_plan_hash: str,
        plan_version: int | str,
        registry_version: int | str,
        registry_hash: str,
        emitter_version: str,
        canonical_genome: str | None,
        input_columns: int,
        seeds: dict[str, Any],
        operator_versions: dict[str, Any],
        backend_provenance: dict[str, Any],
        artifact_hash: str | None = None,
        numerical_tier: str = "exact_f64",
    ) -> ModelArtifactSource:
        source_sha256 = _hash(source)
        identity = {
            "schema": MODEL_ARTIFACT_SCHEMA,
            "artifact_kind": artifact_kind,
            "manifest_sha256": manifest_sha256,
            "test_vectors_sha256": test_vectors_sha256,
            "source_sha256": source_sha256,
            "semantic_plan_hash": semantic_plan_hash,
            "plan_version": plan_version,
            "registry_version": registry_version,
            "registry_hash": registry_hash,
            "emitter_version": emitter_version,
            "canonical_genome": canonical_genome,
            "numerical_tier": numerical_tier,
            "input_columns": input_columns,
            "seeds": seeds,
            "operator_versions": operator_versions,
        }
        return cls(
            schema=MODEL_ARTIFACT_SCHEMA,
            artifact_kind=artifact_kind,
            artifact_hash=artifact_hash or _hash(identity),
            manifest_sha256=manifest_sha256,
            test_vectors_sha256=test_vectors_sha256,
            source=source,
            source_sha256=source_sha256,
            semantic_plan_hash=semantic_plan_hash,
            plan_version=plan_version,
            registry_version=registry_version,
            registry_hash=registry_hash,
            emitter_version=emitter_version,
            canonical_genome=canonical_genome,
            numerical_tier=numerical_tier,
            input_columns=input_columns,
            seeds=dict(seeds),
            operator_versions=dict(operator_versions),
            backend_provenance=dict(backend_provenance),
        )

    @classmethod
    def from_evaluator_bundle(
        cls,
        source: str,
        manifest: str,
        test_vectors: bytes,
        *,
        backend_provenance: dict[str, Any],
        manifest_sha256: str,
    ) -> ModelArtifactSource:
        """Verify and import one ``battle-portable-artifact-v3`` bundle."""
        computed_manifest_hash = _hash(manifest)
        if not _is_sha256(manifest_sha256) or manifest_sha256 != computed_manifest_hash:
            raise ValueError("evaluator bundle manifest hash mismatch")
        try:
            payload = json.loads(manifest)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluator artifact manifest: {error}") from error
        if not isinstance(payload, dict) or payload.get("schema") != "battle-portable-artifact-v3":
            raise ValueError("unsupported evaluator artifact manifest schema")
        artifact_kind = str(payload.get("artifact_kind") or "")
        source_name = {
            "portable_python": "model.py",
            "portable_rust": "model.rs",
        }.get(artifact_kind)
        if source_name is None:
            raise ValueError("evaluator bundle is not a portable Python or Rust artifact")
        checksums = payload.get("checksums")
        if not isinstance(checksums, dict) or checksums.get(source_name) != _hash(source):
            raise ValueError("evaluator bundle source checksum mismatch")
        expected_artifact_hash = _hash(
            {
                "schema": "battle-portable-artifact-identity-v1",
                "artifact_kind": artifact_kind,
                "compiled_plan_hash": payload.get("compiled_plan_hash"),
                "checksums": checksums,
            }
        )
        if payload.get("artifact_hash") != expected_artifact_hash:
            raise ValueError("evaluator bundle artifact identity mismatch")
        vectors_hash = hashlib.sha256(test_vectors).hexdigest()
        vectors_contract = payload.get("deterministic_test_vectors")
        if (
            not isinstance(vectors_contract, dict)
            or vectors_contract.get("included") is not True
            or vectors_contract.get("sha256") != vectors_hash
            or checksums.get("test_vectors.json") != vectors_hash
        ):
            raise ValueError("evaluator bundle test-vector checksum mismatch")
        try:
            vectors_payload = json.loads(test_vectors)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluator test vectors: {error}") from error
        if (
            not isinstance(vectors_payload, dict)
            or vectors_payload.get("schema") != "battle-artifact-test-vectors-v1"
            or vectors_payload.get("numerical_tier") != "exact_f64"
        ):
            raise ValueError("evaluator test vectors do not declare exact-f64 semantics")
        numerical_modes = payload.get("numerical_modes")
        if not isinstance(numerical_modes, list) or "exact_f64" not in numerical_modes:
            raise ValueError("evaluator artifact does not declare exact-f64 semantics")
        registry = payload.get("operator_registry")
        input_contract = payload.get("input_contract")
        seeds = payload.get("seeds")
        if not isinstance(registry, dict) or not isinstance(
            registry.get("operator_versions"), dict
        ):
            raise ValueError("evaluator artifact has no operator registry identity")
        if not isinstance(input_contract, dict):
            raise ValueError("evaluator artifact has no input contract")
        if not isinstance(seeds, dict):
            raise ValueError("evaluator artifact has no deterministic seed map")
        input_columns = int(input_contract.get("columns") or 0)
        train_rows = vectors_payload.get("train_rows")
        test_features = vectors_payload.get("test_features")
        expected_predictions = vectors_payload.get("expected_predictions")
        if (
            input_columns <= 0
            or not isinstance(train_rows, list)
            or not train_rows
            or any(
                not isinstance(row, list) or len(row) != input_columns + 1
                for row in train_rows
            )
            or not isinstance(test_features, list)
            or not test_features
            or any(
                not isinstance(row, list) or len(row) != input_columns
                for row in test_features
            )
            or not isinstance(expected_predictions, list)
            or len(expected_predictions) != len(test_features)
            or vectors_payload.get("features") != test_features
            or vectors_payload.get("predictions") != expected_predictions
        ):
            raise ValueError("evaluator test vectors are not self-contained and replayable")
        artifact = cls.create(
            source,
            artifact_kind=artifact_kind,
            artifact_hash=str(payload.get("artifact_hash") or ""),
            manifest_sha256=computed_manifest_hash,
            test_vectors_sha256=vectors_hash,
            semantic_plan_hash=str(payload.get("compiled_plan_hash") or ""),
            plan_version=payload.get("plan_version", ""),
            registry_version=registry.get("version", ""),
            registry_hash=str(registry.get("hash") or ""),
            emitter_version=str(payload.get("emitter_version") or ""),
            canonical_genome=(
                str(payload["canonical_genome"])
                if payload.get("canonical_genome") is not None
                else None
            ),
            input_columns=input_columns,
            seeds=seeds,
            operator_versions=registry["operator_versions"],
            backend_provenance=backend_provenance,
            numerical_tier="exact_f64",
        )
        _validate_model_artifact_scope(artifact)
        return artifact


@dataclass(frozen=True)
class ModelRealizationLabel:
    """Observed standalone-model quality from an allowed train-side split."""

    schema: str
    artifact_schema: str
    dataset_id: str
    dataset_hash: str
    data_sha256: str
    split: str
    engine: str
    task: str
    target: str
    artifact_hash: str
    manifest_sha256: str
    test_vectors_sha256: str
    source_sha256: str
    semantic_plan_hash: str
    plan_version: int | str
    registry_version: int | str
    registry_hash: str
    emitter_version: str
    canonical_genome: str | None
    numerical_tier: str
    input_columns: int
    seeds: dict[str, Any]
    fold_plan_hash: str
    target_hash: str
    operator_versions: dict[str, Any]
    backend_provenance: dict[str, Any]
    holdout_predictions: list[float] | None
    holdout_predictions_sha256: str | None
    realized_score: float | None
    holdout_r2: float | None
    holdout_rmse: float | None
    holdout_mae: float | None
    cv_profile: list[dict[str, Any]]
    oof_predictions_sha256: str | None
    oof_r2: float | None
    exact_prediction_maximum_absolute_error: float | None
    exact_prediction_rmse: float | None
    promotion_eligible: bool
    execution_runtime_ms: float
    compile_runtime_ms: float | None
    total_runtime_ms: float
    peak_memory_bytes: int | None
    failure: str | None
    label_hash: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealizationLabel:
    """Backward-compatible type name for a transform-conformance record."""
    schema: str
    dataset_id: str
    split: str
    target: str
    operator: str
    parameters: dict[str, Any]
    artifact_hash: str
    manifest_payload_sha256: str
    source_sha256: str
    reference_outputs: list[list[float]]
    artifact_outputs: list[list[float]] | None
    parity: bool
    maximum_absolute_error: float | None
    runtime_ms: float | None
    compile_runtime_ms: float | None
    peak_memory_bytes: int | None
    backend_provenance: dict[str, Any]
    failure: str | None
    label_hash: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"shape": list(array.shape), "dtype": "float64"}).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def prime_owned_path(path: str | Path, config: PrimeConfig | None = None) -> Path:
    """Resolve a write target and reject every path outside ``prime.root``."""
    config = config or load_config()
    root = config.root.resolve(strict=False)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"realization output must resolve below Prime root {root}")
    return candidate


def append_realization_label(
    path: str | Path,
    label: RealizationLabel,
    *,
    config: PrimeConfig | None = None,
) -> Path:
    """Append one label only after enforcing the Prime-owned path boundary."""
    destination = prime_owned_path(path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(label.to_json()) + "\n")
    return destination


def append_model_realization_label(
    path: str | Path,
    label: ModelRealizationLabel,
    *,
    config: PrimeConfig | None = None,
) -> Path:
    """Append a full-model label only to a Prime-owned JSONL file."""
    destination = prime_owned_path(path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(label.to_json()) + "\n")
    return destination


def _sealed_dataset_hashes(config: PrimeConfig) -> set[str]:
    """Load the frozen validation pool; realization must fail closed without it."""
    payload = json.loads(config.validation_dataset_manifest.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("sealed validation manifest has no datasets list")
    hashes = {
        str(item["dataset_hash"])
        for item in datasets
        if isinstance(item, dict) and item.get("dataset_hash")
    }
    if not hashes:
        raise ValueError("sealed validation manifest contains no dataset hashes")
    return hashes


def _verify_artifact(artifact: CompileArtifact, target: str) -> tuple[str, dict[str, Any]]:
    manifest = artifact.manifest
    manifest_base = {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_hash", "manifest_payload_sha256"}
    }
    payload_hash = _hash(manifest_base)
    if payload_hash != manifest.get("manifest_payload_sha256"):
        raise ValueError("artifact manifest payload hash mismatch")
    vectors_hash = _hash(artifact.test_vectors)
    if vectors_hash != manifest.get("test_vectors", {}).get("sha256"):
        raise ValueError("artifact test-vector hash mismatch")
    sources = {
        "portable_python": artifact.portable_python,
        "portable_rust": artifact.portable_rust,
    }
    if target not in sources:
        raise ValueError("target must be 'portable_python' or 'portable_rust'")
    source = sources[target]
    source_hash = _hash(source)
    if source_hash != manifest.get("source_sha256", {}).get(target):
        raise ValueError(f"{target} source hash mismatch")
    language = "python" if target == "portable_python" else "rust"
    violations = lint_portable_source(source, language)
    if violations:
        raise ValueError(f"{target} source failed independence lint: {violations}")
    expected_artifact_hash = _hash(
        {
            "manifest_payload_sha256": payload_hash,
            "source_sha256": manifest["source_sha256"],
            "test_vectors_sha256": vectors_hash,
        }
    )
    if expected_artifact_hash != manifest.get("artifact_hash"):
        raise ValueError("artifact identity hash mismatch")
    return source, manifest


def _python_predictions(
    source: str,
    matrix: np.ndarray,
    *,
    artifact_hash: str,
    config: PrimeConfig,
    timeout_s: float,
) -> tuple[np.ndarray, float, int]:
    artifact_cache = prime_owned_path(
        config.cache / "realizations" / artifact_hash, config
    )
    artifact_cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-python-", dir=artifact_cache))
    source_path = work / "count_sketch.py"
    input_path = work / "input.csv"
    output_path = work / "output.csv"
    source_path.write_text(source, encoding="utf-8")
    np.savetxt(input_path, matrix, delimiter=",", fmt="%.17g")
    environment = dict(os.environ)
    environment["TMPDIR"] = str(work)
    runtime_ms, peak, _ = _run_process_with_peak_memory(
        [sys.executable, str(source_path), str(input_path), str(output_path)],
        env=environment,
        timeout_s=timeout_s,
    )
    predictions = np.loadtxt(output_path, delimiter=",", ndmin=2)
    return np.asarray(predictions, dtype=np.float64), runtime_ms, peak


def _run_process_with_peak_memory(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_s: float,
    cwd: Path | None = None,
    description: str = "portable Rust",
) -> tuple[float, int, str]:
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )
    monitor = psutil.Process(process.pid)
    peak = 0
    deadline = time.monotonic() + timeout_s
    while process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            raise TimeoutError(f"{description} execution exceeded {timeout_s:g}s")
        try:
            peak = max(peak, int(monitor.memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.001)
    _, stderr = process.communicate()
    runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if process.returncode != 0:
        raise RuntimeError(
            f"{description} exited {process.returncode}: {stderr.strip()}"
        )
    return runtime_ms, peak, stderr


def _rust_predictions(
    source: str,
    matrix: np.ndarray,
    *,
    artifact_hash: str,
    config: PrimeConfig,
    timeout_s: float,
) -> tuple[np.ndarray, float, float, int]:
    rustc = shutil.which("rustc")
    if rustc is None:
        raise RuntimeError("rustc is unavailable")
    artifact_cache = prime_owned_path(config.cache / "realizations" / artifact_hash, config)
    artifact_cache.mkdir(parents=True, exist_ok=True)
    # Every invocation owns its source, binary, and I/O files. Reusing fixed names
    # under the artifact hash races when one artifact is realized concurrently.
    work = Path(tempfile.mkdtemp(prefix="run-", dir=artifact_cache))
    source_path = work / "count_sketch.rs"
    binary_path = work / "count_sketch"
    input_path = work / "input.csv"
    output_path = work / "output.csv"
    source_path.write_text(source, encoding="utf-8")
    np.savetxt(input_path, matrix, delimiter=",", fmt="%.17g")
    environment = dict(os.environ)
    environment["TMPDIR"] = str(work)
    compile_started = time.perf_counter_ns()
    compiled = subprocess.run(
        [rustc, "-O", str(source_path), "-o", str(binary_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout_s,
    )
    compile_runtime_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
    if compiled.returncode != 0:
        raise RuntimeError(f"rustc failed: {compiled.stderr.strip()}")
    runtime_ms, peak, _ = _run_process_with_peak_memory(
        [str(binary_path), str(input_path), str(output_path)],
        env=environment,
        timeout_s=timeout_s,
    )
    predictions = np.loadtxt(output_path, delimiter=",", ndmin=2)
    return np.asarray(predictions, dtype=np.float64), runtime_ms, compile_runtime_ms, peak


def realize_artifact(
    artifact: CompileArtifact,
    rows: Any,
    *,
    dataset_id: str,
    split: str,
    target: str = "portable_python",
    output_path: str | Path | None = None,
    config: PrimeConfig | None = None,
    timeout_s: float = 30.0,
) -> RealizationLabel:
    """Execute one portable transform and record exact-reference conformance evidence."""
    if split not in TRAIN_SIDE_SPLITS:
        raise ValueError(f"realization split must be one of {sorted(TRAIN_SIDE_SPLITS)}")
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    config = config or load_config()
    if dataset_id in _sealed_dataset_hashes(config):
        raise ValueError(f"dataset {dataset_id!r} belongs to the sealed validation pool")
    destination = prime_owned_path(output_path, config) if output_path is not None else None
    manifest = artifact.manifest
    operator = str(manifest.get("operator", {}).get("name") or "unknown")
    parameters = dict(manifest.get("parameters") or {})
    input_columns = int(manifest.get("input_contract", {}).get("columns") or 0)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != input_columns:
        raise ValueError("realization rows do not satisfy the artifact input contract")
    matrix = np.ascontiguousarray(matrix)
    reference, _ = fit_transform(operator, matrix, parameters)
    predictions: np.ndarray | None = None
    runtime_ms: float | None = None
    compile_runtime_ms: float | None = None
    peak_memory_bytes: int | None = None
    failure: str | None = None
    source_hash = str(manifest.get("source_sha256", {}).get(target) or "")
    provenance: dict[str, Any] = {
        "target": target,
        "numerical_tier": manifest.get("numerical_tier"),
        "operator_versions": {
            operator: manifest.get("operator", {}).get("semantic_version")
        },
        "implementation_version": manifest.get("operator", {}).get(
            "implementation_version"
        ),
    }
    try:
        source, _ = _verify_artifact(artifact, target)
        if target == "portable_python":
            predictions, runtime_ms, peak_memory_bytes = _python_predictions(
                source,
                matrix,
                artifact_hash=artifact.artifact_hash,
                config=config,
                timeout_s=timeout_s,
            )
            provenance["executor"] = "python_subprocess"
        elif target == "portable_rust":
            predictions, runtime_ms, compile_runtime_ms, peak_memory_bytes = _rust_predictions(
                source,
                matrix,
                artifact_hash=artifact.artifact_hash,
                config=config,
                timeout_s=timeout_s,
            )
            provenance["executor"] = "rustc"
        else:
            raise ValueError("target must be 'portable_python' or 'portable_rust'")
    except Exception as error:  # exact failures are labels, not silent drops
        failure = f"{type(error).__name__}: {error}"

    parity = False
    maximum_error: float | None = None
    if predictions is not None:
        if predictions.shape == reference.shape:
            maximum_error = float(
                np.max(np.abs(predictions - reference)) if reference.size else 0.0
            )
            parity = bool(np.array_equal(predictions, reference))
            if not parity and failure is None:
                failure = f"output parity mismatch (maximum absolute error {maximum_error:.17g})"
        elif failure is None:
            failure = (
                f"output shape mismatch: artifact={predictions.shape}, reference={reference.shape}"
            )

    record = {
        "schema": REALIZATION_SCHEMA,
        "dataset_id": dataset_id,
        "split": split,
        "target": target,
        "operator": operator,
        "parameters": parameters,
        "artifact_hash": str(manifest.get("artifact_hash") or ""),
        "manifest_payload_sha256": str(manifest.get("manifest_payload_sha256") or ""),
        "source_sha256": source_hash,
        "reference_outputs": reference.tolist(),
        "artifact_outputs": predictions.tolist() if predictions is not None else None,
        "parity": parity,
        "maximum_absolute_error": maximum_error,
        "runtime_ms": runtime_ms,
        "compile_runtime_ms": compile_runtime_ms,
        "peak_memory_bytes": peak_memory_bytes,
        "backend_provenance": provenance,
        "failure": failure,
    }
    label = RealizationLabel(**record, label_hash=_hash(record))
    if destination is not None:
        append_realization_label(destination, label, config=config)
    return label


def _validate_model_artifact_scope(artifact: ModelArtifactSource) -> None:
    if artifact.schema != MODEL_ARTIFACT_SCHEMA:
        raise ValueError(f"model artifact schema must be {MODEL_ARTIFACT_SCHEMA!r}")
    if artifact.artifact_kind not in PORTABLE_MODEL_TARGETS:
        raise ValueError("only portable Python or Rust artifacts can produce labels")
    if artifact.numerical_tier != "exact_f64":
        raise ValueError("accelerated-only results cannot produce realization labels")
    if artifact.backend_provenance.get("accelerated_only") is True:
        raise ValueError("accelerated-only results cannot produce realization labels")
    if not _is_sha256(artifact.artifact_hash):
        raise ValueError("model artifact_hash must be a lowercase SHA-256 digest")
    if not _is_sha256(artifact.manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    if not _is_sha256(artifact.test_vectors_sha256):
        raise ValueError("test_vectors_sha256 must be a lowercase SHA-256 digest")
    if not _is_sha256(artifact.semantic_plan_hash):
        raise ValueError("semantic_plan_hash must be a lowercase SHA-256 digest")
    if not _is_sha256(artifact.registry_hash):
        raise ValueError("registry_hash must be a lowercase SHA-256 digest")
    if artifact.registry_version in {"", None} or not artifact.emitter_version:
        raise ValueError("model artifact must record registry and emitter versions")
    if not 0 < artifact.input_columns <= 512:
        raise ValueError("model artifact input_columns must be in 1..=512")
    if not isinstance(artifact.seeds, dict):
        raise ValueError("model artifact seeds must be a mapping")
    if not artifact.operator_versions:
        raise ValueError("model artifact must record operator_versions")
    _canonical_json(
        {
            "plan_version": artifact.plan_version,
            "registry_version": artifact.registry_version,
            "input_columns": artifact.input_columns,
            "seeds": artifact.seeds,
            "operator_versions": artifact.operator_versions,
            "backend_provenance": artifact.backend_provenance,
        }
    )


def _validated_model_arrays(
    train_rows: Any,
    test_features: Any,
    test_targets: Any,
    exact_predictions: Any | None,
    input_columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    train = np.ascontiguousarray(np.asarray(train_rows, dtype=np.float64))
    features = np.ascontiguousarray(np.asarray(test_features, dtype=np.float64))
    targets = np.ascontiguousarray(np.asarray(test_targets, dtype=np.float64).reshape(-1))
    if train.ndim != 2 or train.shape[0] < 2 or train.shape[1] < 2:
        raise ValueError("train_rows must have at least two rows, one feature, and a target")
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("test_features must be a non-empty two-dimensional matrix")
    if features.shape[1] != train.shape[1] - 1:
        raise ValueError("test feature count does not match train_rows")
    if features.shape[1] != input_columns:
        raise ValueError("model inputs do not satisfy the artifact input contract")
    if targets.shape != (features.shape[0],):
        raise ValueError("test_targets must contain one value per test row")
    if not np.all(np.isfinite(train[:, -1])) or not np.all(np.isfinite(targets)):
        raise ValueError("training and holdout targets must be finite")
    exact: np.ndarray | None = None
    if exact_predictions is not None:
        exact = np.ascontiguousarray(
            np.asarray(exact_predictions, dtype=np.float64).reshape(-1)
        )
        if exact.shape != targets.shape or not np.all(np.isfinite(exact)):
            raise ValueError("exact_predictions must be finite and match test_targets")
    return train, features, targets, exact


def _validated_folds(
    folds: Sequence[Sequence[int]], row_count: int
) -> list[list[int]]:
    if isinstance(folds, (str, bytes)) or len(folds) < 2:
        raise ValueError("folds must contain at least two explicit validation folds")
    normalized: list[list[int]] = []
    observed: list[int] = []
    for fold_number, fold in enumerate(folds):
        if isinstance(fold, (str, bytes)) or not fold:
            raise ValueError(f"fold {fold_number} must contain validation rows")
        indices: list[int] = []
        for index in fold:
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                raise ValueError("fold indices must be integers")
            resolved = int(index)
            if not 0 <= resolved < row_count:
                raise ValueError(f"fold index {resolved} is outside train_rows")
            indices.append(resolved)
        if len(indices) != len(set(indices)):
            raise ValueError(f"fold {fold_number} contains duplicate indices")
        if len(indices) == row_count:
            raise ValueError(f"fold {fold_number} leaves no training rows")
        normalized.append(indices)
        observed.extend(indices)
    if sorted(observed) != list(range(row_count)):
        raise ValueError("explicit folds must cover every training row exactly once")
    return normalized


def _regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predictions - targets
        squared_error = float(np.dot(residual, residual))
        centered = targets - float(np.mean(targets))
        total_variance = float(np.dot(centered, centered))
    if total_variance == 0.0:
        r2 = 1.0 if squared_error == 0.0 else 0.0
    else:
        r2 = 1.0 - squared_error / total_variance
    metrics = {
        "r2": float(r2),
        "rmse": float(np.sqrt(squared_error / targets.size)),
        "mae": float(np.mean(np.abs(residual))),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("standalone program produced non-finite regression metrics")
    return metrics


def _read_predictions(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError("standalone program did not create predictions.csv")
    try:
        predictions = np.asarray(
            np.loadtxt(path, delimiter=",", dtype=np.float64), dtype=np.float64
        ).reshape(-1)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid predictions.csv: {error}") from error
    if predictions.shape != (expected_rows,):
        raise RuntimeError(
            f"prediction count mismatch: expected {expected_rows}, got {predictions.size}"
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("standalone program emitted non-finite predictions")
    return np.ascontiguousarray(predictions)


def _execute_model_program(
    command: list[str],
    *,
    work: Path,
    name: str,
    train_rows: np.ndarray,
    test_features: np.ndarray,
    environment: dict[str, str],
    timeout_s: float,
    description: str,
) -> tuple[np.ndarray, float, int]:
    run_dir = work / name
    run_dir.mkdir()
    train_path = run_dir / "train.csv"
    test_path = run_dir / "test_features.csv"
    predictions_path = run_dir / "predictions.csv"
    np.savetxt(train_path, train_rows, delimiter=",", fmt="%.17g")
    np.savetxt(test_path, test_features, delimiter=",", fmt="%.17g")
    runtime_ms, peak_memory, _ = _run_process_with_peak_memory(
        command + [str(train_path), str(test_path), str(predictions_path)],
        env=environment,
        timeout_s=timeout_s,
        cwd=run_dir,
        description=description,
    )
    return _read_predictions(predictions_path, test_features.shape[0]), runtime_ms, peak_memory


def realize_model_artifact(
    artifact: ModelArtifactSource,
    train_rows: Any,
    test_features: Any,
    test_targets: Any,
    *,
    dataset_id: str,
    split: str,
    folds: Sequence[Sequence[int]],
    dataset_hash: str | None = None,
    exact_predictions: Any | None = None,
    exact_tolerance: float = 0.0,
    output_path: str | Path | None = None,
    config: PrimeConfig | None = None,
    timeout_s: float = 30.0,
) -> ModelRealizationLabel:
    """Run a portable fitted model and record holdout plus explicit-fold CV quality."""
    _validate_model_artifact_scope(artifact)
    if split not in TRAIN_SIDE_SPLITS:
        raise ValueError(f"realization split must be one of {sorted(TRAIN_SIDE_SPLITS)}")
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    if not np.isfinite(exact_tolerance) or exact_tolerance < 0.0:
        raise ValueError("exact_tolerance must be finite and non-negative")
    config = config or load_config()
    destination = prime_owned_path(output_path, config) if output_path is not None else None
    train, holdout_x, holdout_y, exact = _validated_model_arrays(
        train_rows,
        test_features,
        test_targets,
        exact_predictions,
        artifact.input_columns,
    )
    normalized_folds = _validated_folds(folds, train.shape[0])
    data_sha256 = _hash(
        {
            "train_rows": _array_hash(train),
            "test_features": _array_hash(holdout_x),
            "test_targets": _array_hash(holdout_y),
        }
    )
    resolved_dataset_hash = str(dataset_hash or data_sha256)
    sealed = _sealed_dataset_hashes(config)
    if {dataset_id, resolved_dataset_hash, data_sha256} & sealed:
        raise ValueError(f"dataset {dataset_id!r} belongs to the sealed validation pool")

    fold_plan_hash = _hash(normalized_folds)
    target_hash = _array_hash(np.concatenate((train[:, -1], holdout_y)))
    started = time.perf_counter_ns()
    holdout_predictions: np.ndarray | None = None
    holdout_metrics: dict[str, float] | None = None
    cv_profile: list[dict[str, Any]] = []
    oof_predictions: np.ndarray | None = None
    exact_maximum_error: float | None = None
    exact_rmse: float | None = None
    execution_runtime_ms = 0.0
    compile_runtime_ms: float | None = None
    peak_memory_bytes: int | None = None
    failure: str | None = None
    executor = "python_subprocess"

    try:
        if _hash(artifact.source) != artifact.source_sha256:
            raise ValueError(f"{artifact.artifact_kind} source hash mismatch")
        language = "python" if artifact.artifact_kind == "portable_python" else "rust"
        violations = lint_portable_source(artifact.source, language)
        if violations:
            raise ValueError(
                f"{artifact.artifact_kind} source failed independence lint: {violations}"
            )

        artifact_cache = prime_owned_path(
            config.cache / "model_realizations" / artifact.artifact_hash, config
        )
        artifact_cache.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=artifact_cache))
        environment = dict(os.environ)
        environment.update(
            {
                "TMPDIR": str(work),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
            }
        )
        if artifact.artifact_kind == "portable_python":
            source_path = work / "model.py"
            source_path.write_text(artifact.source, encoding="utf-8")
            command = [sys.executable, str(source_path)]
            compile_runtime_ms = 0.0
            description = "portable Python model"
        else:
            rustc = shutil.which("rustc")
            if rustc is None:
                raise RuntimeError("rustc is unavailable")
            source_path = work / "model.rs"
            binary_path = work / "model"
            source_path.write_text(artifact.source, encoding="utf-8")
            compile_runtime_ms, compile_peak, _ = _run_process_with_peak_memory(
                [rustc, "-O", str(source_path), "-o", str(binary_path)],
                env=environment,
                timeout_s=timeout_s,
                cwd=work,
                description="rustc compilation",
            )
            peak_memory_bytes = compile_peak
            command = [str(binary_path)]
            executor = "rustc_subprocess"
            description = "portable Rust model"

        holdout_predictions, runtime_ms, peak = _execute_model_program(
            command,
            work=work,
            name="holdout",
            train_rows=train,
            test_features=holdout_x,
            environment=environment,
            timeout_s=timeout_s,
            description=description,
        )
        execution_runtime_ms += runtime_ms
        peak_memory_bytes = max(peak_memory_bytes or 0, peak)
        holdout_metrics = _regression_metrics(holdout_y, holdout_predictions)
        if exact is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                exact_error = holdout_predictions - exact
                exact_maximum_error = float(np.max(np.abs(exact_error)))
                exact_rmse = float(np.sqrt(np.mean(np.square(exact_error))))
            if not np.isfinite(exact_maximum_error) or not np.isfinite(exact_rmse):
                raise RuntimeError("exact-prediction comparison produced non-finite error")

        oof_predictions = np.empty(train.shape[0], dtype=np.float64)
        all_indices = np.arange(train.shape[0])
        for fold_number, validation_indices in enumerate(normalized_folds):
            validation = np.asarray(validation_indices, dtype=np.int64)
            training = np.setdiff1d(all_indices, validation, assume_unique=True)
            predictions, runtime_ms, peak = _execute_model_program(
                command,
                work=work,
                name=f"fold-{fold_number:03d}",
                train_rows=train[training],
                test_features=train[validation, :-1],
                environment=environment,
                timeout_s=timeout_s,
                description=description,
            )
            execution_runtime_ms += runtime_ms
            peak_memory_bytes = max(peak_memory_bytes or 0, peak)
            oof_predictions[validation] = predictions
            metrics = _regression_metrics(train[validation, -1], predictions)
            cv_profile.append(
                {
                    "fold": fold_number,
                    "training_rows": int(training.size),
                    "validation_rows": int(validation.size),
                    "validation_indices_sha256": _hash(validation_indices),
                    "predictions_sha256": _array_hash(predictions),
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "runtime_ms": runtime_ms,
                }
            )
    except Exception as error:  # standalone failures are adversarial labels
        failure = f"{type(error).__name__}: {error}"

    oof_metrics = (
        _regression_metrics(train[:, -1], oof_predictions)
        if oof_predictions is not None and len(cv_profile) == len(normalized_folds)
        else None
    )
    if (
        failure is None
        and exact_maximum_error is not None
        and exact_maximum_error > exact_tolerance
    ):
        failure = (
            "exact-reference mismatch: maximum absolute error "
            f"{exact_maximum_error:.17g} exceeds {exact_tolerance:.17g}"
        )
    total_runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    backend_provenance = {
        "artifact": dict(artifact.backend_provenance),
        "realization_executor": executor,
        "numerical_tier": artifact.numerical_tier,
    }
    record = {
        "schema": MODEL_REALIZATION_SCHEMA,
        "artifact_schema": artifact.schema,
        "dataset_id": dataset_id,
        "dataset_hash": resolved_dataset_hash,
        "data_sha256": data_sha256,
        "split": split,
        "engine": "prime",
        "task": "regression",
        "target": artifact.artifact_kind,
        "artifact_hash": artifact.artifact_hash,
        "manifest_sha256": artifact.manifest_sha256,
        "test_vectors_sha256": artifact.test_vectors_sha256,
        "source_sha256": artifact.source_sha256,
        "semantic_plan_hash": artifact.semantic_plan_hash,
        "plan_version": artifact.plan_version,
        "registry_version": artifact.registry_version,
        "registry_hash": artifact.registry_hash,
        "emitter_version": artifact.emitter_version,
        "canonical_genome": artifact.canonical_genome,
        "numerical_tier": artifact.numerical_tier,
        "input_columns": artifact.input_columns,
        "seeds": dict(artifact.seeds),
        "fold_plan_hash": fold_plan_hash,
        "target_hash": target_hash,
        "operator_versions": dict(artifact.operator_versions),
        "backend_provenance": backend_provenance,
        "holdout_predictions": (
            holdout_predictions.tolist() if holdout_predictions is not None else None
        ),
        "holdout_predictions_sha256": (
            _array_hash(holdout_predictions) if holdout_predictions is not None else None
        ),
        # This is the selector-facing label. A partial run may retain useful
        # holdout diagnostics, but it must never become promotion evidence.
        "realized_score": (
            holdout_metrics["r2"] if holdout_metrics and failure is None else None
        ),
        "holdout_r2": holdout_metrics["r2"] if holdout_metrics else None,
        "holdout_rmse": holdout_metrics["rmse"] if holdout_metrics else None,
        "holdout_mae": holdout_metrics["mae"] if holdout_metrics else None,
        "cv_profile": cv_profile,
        "oof_predictions_sha256": (
            _array_hash(oof_predictions) if oof_metrics is not None else None
        ),
        "oof_r2": oof_metrics["r2"] if oof_metrics else None,
        "exact_prediction_maximum_absolute_error": exact_maximum_error,
        "exact_prediction_rmse": exact_rmse,
        "promotion_eligible": (
            failure is None
            and oof_metrics is not None
            and exact_maximum_error is not None
            and exact_maximum_error <= exact_tolerance
        ),
        "execution_runtime_ms": execution_runtime_ms,
        "compile_runtime_ms": compile_runtime_ms,
        "total_runtime_ms": total_runtime_ms,
        "peak_memory_bytes": peak_memory_bytes,
        "failure": failure,
    }
    label = ModelRealizationLabel(**record, label_hash=_hash(record))
    if destination is not None:
        append_model_realization_label(destination, label, config=config)
    return label
