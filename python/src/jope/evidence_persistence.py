"""Crash-safe, nonblocking persistence for representative Prime evidence.

Search producers submit one immutable task sample to ``AsyncEvidenceWriter``.
The background thread writes a real Parquet/Zstd shard, verifies its content
hash and row count, then commits a content-hashed TaskEvidenceManifest.v1.
``COMPLETE.json`` is written last and is the only task-completion marker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

CANDIDATE_OBSERVATION_V1 = "CandidateObservation.v1"
HONEST_EVALUATION_V2 = "HonestEvaluation.v2"
TASK_EVIDENCE_MANIFEST_V1 = "TaskEvidenceManifest.v1"
ARENA_PROPOSAL_DECISION_V1 = "ArenaProposalDecision.v1"
TASK_EVIDENCE_INTENT_V1 = "TaskEvidenceIntent.v1"
TASK_EVIDENCE_RETRY_V1 = "TaskEvidenceRetry.v1"
EVIDENCE_RECORD_BINDING_V1 = "EvidenceRecordBinding.v1"
EVIDENCE_PERSISTENCE_BATCH_V1 = "EvidencePersistenceBatch.v1"
GP_EVIDENCE_PERSISTENCE_BATCH_V1 = "GpEvidencePersistenceBatch.v1"
MAX_TASK_OBSERVATIONS = 1_024
EVIDENCE_STRATA = {
    "top_near_cutoff",
    "uncertainty_disagreement",
    "quartile_random",
    "structural_novelty_parentage",
    "invalid_timeout_failure",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FAMILY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class EvidenceSubmission:
    accepted: bool
    task_id: str
    reason: str | None = None


@dataclass(frozen=True)
class EvidenceWriteResult:
    task_id: str
    status: str
    manifest_path: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class EvidenceRecordResult:
    task_id: str
    record_schema: str
    content_sha256: str
    record_path: Path
    binding_path: Path


@dataclass(frozen=True)
class _TaskWrite:
    task_id: str
    release_id: str
    observations: tuple[dict[str, Any], ...]
    telemetry_totals: dict[str, float]


@dataclass(frozen=True)
class _IncompleteWrite:
    task_id: str
    release_id: str
    reason: str


def _retry_spool_path(root: Path, task_id: str) -> Path:
    return _task_directory(root, task_id) / "RETRY.json"


def _retry_payload(task: _TaskWrite) -> dict[str, Any]:
    content = {
        "schema": TASK_EVIDENCE_RETRY_V1,
        "task_id": task.task_id,
        "release_id": task.release_id,
        "observations": list(task.observations),
        "telemetry_totals": task.telemetry_totals,
    }
    return {
        **content,
        "content_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def _commit_retry_spool(root: Path, task: _TaskWrite) -> Path:
    path = _retry_spool_path(root, task.task_id)
    encoded = _canonical_json(_retry_payload(task))
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("task evidence retry spool has an immutable collision")
    else:
        _atomic_write(path, encoded)
    return path


def _read_retry_spool(path: Path) -> _TaskWrite:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "task_id",
        "release_id",
        "observations",
        "telemetry_totals",
        "content_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("task evidence retry spool has invalid fields")
    content = {key: payload[key] for key in expected - {"content_sha256"}}
    if (
        payload["schema"] != TASK_EVIDENCE_RETRY_V1
        or not isinstance(payload["task_id"], str)
        or not payload["task_id"].strip()
        or not _SHA256.fullmatch(str(payload["release_id"]))
        or not isinstance(payload["observations"], list)
        or not isinstance(payload["telemetry_totals"], dict)
        or not _SHA256.fullmatch(str(payload["content_sha256"]))
        or hashlib.sha256(_canonical_json(content)).hexdigest()
        != payload["content_sha256"]
    ):
        raise ValueError("task evidence retry spool is invalid or corrupted")
    observations = _snapshot_observations(payload["observations"])
    telemetry = json.loads(_canonical_json(payload["telemetry_totals"]))
    task = _TaskWrite(
        payload["task_id"], payload["release_id"], observations, telemetry
    )
    if _retry_payload(task) != payload:
        raise ValueError("task evidence retry spool is not canonical")
    return task


def _remove_retry_spool(root: Path, task: _TaskWrite) -> None:
    path = _retry_spool_path(root, task.task_id)
    if not path.exists():
        return
    if _read_retry_spool(path) != task:
        raise ValueError("task evidence retry spool differs before acknowledgement")
    path.unlink()
    _fsync_directory(path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _task_directory(root: Path, task_id: str) -> Path:
    key = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return root / "tasks" / key


@contextmanager
def _task_commit_lock(root: Path, task_id: str) -> Iterator[None]:
    """Serialize one task's authoritative commit across writer processes."""
    task_dir = _task_directory(root, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(task_dir / ".commit.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate observation requires nonempty {name}")
    return value


def _require_optional_finite(value: Any, name: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(float(value))
    ):
        raise ValueError(f"candidate observation {name} must be finite when present")


def _validate_observation(row: dict[str, Any], task_id: str, release_id: str) -> None:
    required_fields = {
        "schema",
        "task_id",
        "task",
        "dataset",
        "source_family",
        "source_family_manifest_sha256",
        "engine",
        "candidate_id",
        "raw_genome",
        "semantic_id",
        "canonical_genome",
        "generation",
        "origin",
        "complexity",
        "search_plan",
        "selection_stratum",
        "selection_probability",
        "registry_identity",
        "release_id",
    }
    optional_fields = {
        "parents",
        "mutation",
        "proxy_score",
        "rung_scores",
        "dispersion_score",
        "acquisition_score",
        "failure",
    }
    missing = sorted(required_fields - set(row))
    unknown = sorted(set(row) - required_fields - optional_fields)
    if missing or unknown:
        raise ValueError(
            f"candidate observation fields differ from CandidateObservation.v1; "
            f"missing={missing}, unknown={unknown}"
        )
    if row["schema"] != CANDIDATE_OBSERVATION_V1:
        raise ValueError("unsupported candidate observation schema")
    if row["task_id"] != task_id or row["release_id"] != release_id:
        raise ValueError("candidate observation differs from task/release identity")
    for name in (
        "task_id",
        "task",
        "dataset",
        "engine",
        "candidate_id",
        "raw_genome",
        "origin",
        "search_plan",
    ):
        _require_nonempty_string(row[name], name)
    if not isinstance(row["source_family"], str) or not _SOURCE_FAMILY.fullmatch(
        row["source_family"]
    ):
        raise ValueError("candidate observation source_family must be a lowercase slug")
    if not _SHA256.fullmatch(str(row["source_family_manifest_sha256"])):
        raise ValueError(
            "candidate observation source_family_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    semantic_missing = (
        not isinstance(row["semantic_id"], str) or not row["semantic_id"].strip()
    )
    canonical_missing = (
        not isinstance(row["canonical_genome"], str)
        or not row["canonical_genome"].strip()
    )
    if semantic_missing != canonical_missing:
        raise ValueError(
            "candidate canonical and semantic identities must be present together"
        )
    generation = row["generation"]
    complexity = row["complexity"]
    if type(generation) is not int or not 0 <= generation <= (1 << 64) - 1:
        raise ValueError(
            "candidate observation generation must be an unsigned 64-bit integer"
        )
    if type(complexity) is not int or not 0 <= complexity <= (1 << 32) - 1:
        raise ValueError(
            "candidate observation complexity must be an unsigned 32-bit integer"
        )
    parents = row.get("parents", [])
    if not isinstance(parents, list):
        raise ValueError("candidate observation parents must be a list")
    validated_parents = [
        _require_nonempty_string(parent, "parent") for parent in parents
    ]
    if len(set(validated_parents)) != len(validated_parents):
        raise ValueError("candidate observation parents must be unique")
    mutation = row.get("mutation")
    if mutation is not None:
        _require_nonempty_string(mutation, "mutation")
    for name in ("proxy_score", "dispersion_score", "acquisition_score"):
        _require_optional_finite(row.get(name), name)
    rung_scores = row.get("rung_scores", {})
    if not isinstance(rung_scores, dict):
        raise ValueError("candidate observation rung_scores must be an object")
    for name, score in rung_scores.items():
        _require_nonempty_string(name, "rung name")
        if type(score) not in (int, float) or not math.isfinite(float(score)):
            raise ValueError("candidate observation rung score must be finite")
    registry = row["registry_identity"]
    if not isinstance(registry, dict) or set(registry) != {
        "registry_hash",
        "vocabulary_hash",
        "release_hash",
    }:
        raise ValueError("candidate observation has invalid registry identity")
    for name, digest in registry.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(
                f"candidate observation {name} must be a lowercase SHA-256 digest"
            )
    if registry["release_hash"] != release_id:
        raise ValueError("candidate observation release differs from registry identity")
    failure = row.get("failure")
    if failure is not None:
        if not isinstance(failure, dict) or set(failure) != {"code", "message"}:
            raise ValueError("candidate observation failure has invalid fields")
        _require_nonempty_string(failure["code"], "failure code")
        _require_nonempty_string(failure["message"], "failure message")
    if semantic_missing:
        if (
            row["selection_stratum"] != "invalid_timeout_failure"
            or failure is None
            or complexity != 0
            or row.get("proxy_score") is not None
            or rung_scores
            or row.get("dispersion_score") is not None
            or row.get("acquisition_score") is not None
        ):
            raise ValueError(
                "pre-canonical candidate failures require empty identities, zero complexity, no scores, and the failure stratum"
            )


def _validate_honest_evaluation(row: dict[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "task",
        "dataset",
        "source_family",
        "source_family_manifest_sha256",
        "engine",
        "candidate_id",
        "semantic_id",
        "canonical_genome",
        "evaluator_version",
        "fold_plan",
        "folds",
        "fold_scores",
        "oof_score",
        "robust_score",
        "runtime_seconds",
        "selection_stratum",
        "selection_probability",
        "evidence_tier",
        "registry_identity",
        "release_id",
        "failure",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("honest evaluation has unknown or missing fields")
    if row["schema"] != HONEST_EVALUATION_V2:
        raise ValueError("unsupported honest evaluation schema")
    for name in (
        "task_id",
        "task",
        "dataset",
        "engine",
        "candidate_id",
        "semantic_id",
        "canonical_genome",
        "evaluator_version",
        "fold_plan",
        "evidence_tier",
    ):
        _require_nonempty_string(row[name], name)
    if not isinstance(row["source_family"], str) or not _SOURCE_FAMILY.fullmatch(
        row["source_family"]
    ):
        raise ValueError("honest evaluation source_family must be a lowercase slug")
    if not _SHA256.fullmatch(str(row["source_family_manifest_sha256"])):
        raise ValueError("honest evaluation has invalid source-family authority hash")
    if str(row["evidence_tier"]).lower() == "proxy":
        raise ValueError("proxy evidence cannot be represented as HonestEvaluation.v2")
    folds = row["folds"]
    scores = row["fold_scores"]
    if (
        type(folds) is not int
        or not 2 <= folds <= (1 << 16) - 1
        or not isinstance(scores, list)
        or len(scores) != folds
        or any(
            score is not None
            and (type(score) not in (int, float) or not math.isfinite(float(score)))
            for score in scores
        )
    ):
        raise ValueError("honest evaluation must account for every declared fold")
    runtime = row["runtime_seconds"]
    if (
        type(runtime) not in (int, float)
        or not math.isfinite(float(runtime))
        or runtime < 0
    ):
        raise ValueError("honest evaluation runtime must be finite and nonnegative")
    failure = row["failure"]
    success = (
        failure is None
        and row["oof_score"] is not None
        and row["robust_score"] is not None
        and all(score is not None for score in scores)
    )
    if success:
        _require_optional_finite(row["oof_score"], "oof_score")
        _require_optional_finite(row["robust_score"], "robust_score")
    elif (
        row["oof_score"] is None
        and row["robust_score"] is None
        and isinstance(failure, dict)
        and set(failure) == {"code", "message"}
        and all(score is None for score in scores)
    ):
        _require_nonempty_string(failure["code"], "failure code")
        _require_nonempty_string(failure["message"], "failure message")
    else:
        raise ValueError(
            "honest evaluation requires complete scores or one classified failure with no partial fold scores"
        )
    probability = row["selection_probability"]
    if (
        row["selection_stratum"] not in EVIDENCE_STRATA
        or type(probability) not in (int, float)
        or not math.isfinite(float(probability))
        or not 0.0 < float(probability) <= 1.0
    ):
        raise ValueError("honest evaluation has invalid selection evidence")
    registry = row["registry_identity"]
    if not isinstance(registry, dict) or set(registry) != {
        "registry_hash",
        "vocabulary_hash",
        "release_hash",
    }:
        raise ValueError("honest evaluation has invalid registry identity")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in registry.values()
    ):
        raise ValueError("honest evaluation registry identity has invalid hashes")
    if not isinstance(row["release_id"], str) or not _SHA256.fullmatch(
        row["release_id"]
    ):
        raise ValueError("honest evaluation release_id is invalid")
    if registry["release_hash"] != row["release_id"]:
        raise ValueError("honest evaluation release differs from registry identity")


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_arena_proposal_decision(row: dict[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "champion_checkpoint",
        "challenger_checkpoint",
        "dataset",
        "dataset_sha256",
        "suite",
        "proposal_seed",
        "mmr_lambda",
        "proposals",
        "registry_identity",
        "release_id",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("arena proposal decision has unknown or missing fields")
    if row["schema"] != ARENA_PROPOSAL_DECISION_V1:
        raise ValueError("unsupported arena proposal decision schema")
    _require_nonempty_string(row["task_id"], "arena proposal task_id")
    _require_nonempty_string(row["dataset"], "arena proposal dataset")
    for name in ("champion_checkpoint", "challenger_checkpoint", "dataset_sha256"):
        if not isinstance(row[name], str) or not _SHA256.fullmatch(row[name]):
            raise ValueError(f"arena proposal decision {name} is invalid")
    if row["champion_checkpoint"] == row["challenger_checkpoint"]:
        raise ValueError("arena proposal decision players must differ")
    if row["suite"] not in {"stage_a", "stage_b", "double_holdout"}:
        raise ValueError("arena proposal decision has unsupported suite")
    if (
        type(row["proposal_seed"]) is not int
        or not 0 <= row["proposal_seed"] <= (1 << 64) - 1
    ):
        raise ValueError("arena proposal decision proposal_seed is invalid")
    if type(row["mmr_lambda"]) not in (int, float) or float(row["mmr_lambda"]) not in {
        0.0,
        0.1,
        0.25,
        0.5,
    }:
        raise ValueError(
            "arena proposal decision MMR lambda is outside the locked grid"
        )
    registry = row["registry_identity"]
    if not isinstance(registry, dict) or set(registry) != {
        "registry_hash",
        "vocabulary_hash",
        "release_hash",
    }:
        raise ValueError("arena proposal decision has invalid registry identity")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in registry.values()
    ):
        raise ValueError("arena proposal decision registry identity has invalid hashes")
    if (
        not isinstance(row["release_id"], str)
        or not _SHA256.fullmatch(row["release_id"])
        or row["release_id"] != registry["release_hash"]
    ):
        raise ValueError(
            "arena proposal decision release differs from registry identity"
        )

    proposals = row["proposals"]
    if not isinstance(proposals, list) or len(proposals) != 160:
        raise ValueError("arena proposal decision must preserve all 160 raw proposals")
    proposal_fields = {
        "candidate_id",
        "engine",
        "raw_genome",
        "canonical_genome",
        "semantic_id",
        "raw_predictions",
        "expected_score",
        "uncertainty",
        "safety_probability",
        "structural_cluster",
        "rank",
        "selection_reason",
        "cv_result",
        "failure_reason",
    }
    candidate_ids: set[str] = set()
    ranks: set[int] = set()
    selected_semantics: set[str] = set()
    selected_clusters: dict[str, int] = {}
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != proposal_fields:
            raise ValueError("arena proposal prediction has unknown or missing fields")
        for name in (
            "candidate_id",
            "engine",
            "raw_genome",
            "structural_cluster",
            "selection_reason",
        ):
            _require_nonempty_string(proposal[name], f"proposal {name}")
        if proposal["candidate_id"] in candidate_ids:
            raise ValueError("arena proposal candidate IDs must be unique")
        candidate_ids.add(proposal["candidate_id"])
        predictions = proposal["raw_predictions"]
        if (
            not isinstance(predictions, list)
            or not predictions
            or any(not _is_finite_number(value) for value in predictions)
            or not _is_finite_number(proposal["expected_score"])
            or not _is_finite_number(proposal["uncertainty"])
            or float(proposal["uncertainty"]) < 0.0
            or not _is_finite_number(proposal["safety_probability"])
            or not 0.0 <= float(proposal["safety_probability"]) <= 1.0
        ):
            raise ValueError("arena proposal prediction signals are invalid")
        canonical = proposal["canonical_genome"]
        semantic = proposal["semantic_id"]
        failure = proposal["failure_reason"]
        if canonical is None and semantic is None and failure is not None:
            pass
        elif (
            isinstance(canonical, str)
            and canonical.strip()
            and isinstance(semantic, str)
            and semantic.strip()
        ):
            pass
        else:
            raise ValueError(
                "proposal canonical and semantic identities must be present together"
            )
        if failure is not None:
            _require_nonempty_string(failure, "proposal failure_reason")
        rank = proposal["rank"]
        if rank is not None and (
            type(rank) is not int or not 0 <= rank < 10 or rank in ranks
        ):
            raise ValueError("arena proposal selected rank is invalid or duplicate")
        cv_result = proposal["cv_result"]
        if cv_result is not None:
            _validate_honest_evaluation(cv_result)
            if (
                cv_result["candidate_id"] != proposal["candidate_id"]
                or cv_result["engine"] != proposal["engine"]
                or cv_result["semantic_id"] != semantic
                or cv_result["canonical_genome"] != canonical
            ):
                raise ValueError(
                    "proposal CV result differs from its raw proposal identity"
                )
        if rank is not None and ((cv_result is not None) == (failure is not None)):
            raise ValueError(
                "selected proposal requires exactly one CV result or failure reason"
            )
        if rank is None and cv_result is not None:
            raise ValueError(
                "unselected proposal cannot carry an authoritative CV result"
            )
        if rank is not None:
            ranks.add(rank)
            if not isinstance(semantic, str) or not semantic.strip():
                raise ValueError("selected arena proposal has no semantic identity")
            if semantic in selected_semantics:
                raise ValueError("arena proposal selected semantics must be unique")
            selected_semantics.add(semantic)
            cluster = proposal["structural_cluster"]
            selected_clusters[cluster] = selected_clusters.get(cluster, 0) + 1
            if selected_clusters[cluster] > 2:
                raise ValueError(
                    "arena proposal selects more than two candidates from one cluster"
                )
    if ranks != set(range(10)):
        raise ValueError(
            "arena proposal decision must select exactly ranks 0 through 9"
        )


def _validate_task(
    task_id: str,
    release_id: str,
    observations: Iterable[dict[str, Any]],
    telemetry_totals: dict[str, float],
) -> list[dict[str, Any]]:
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    if not _SHA256.fullmatch(release_id):
        raise ValueError("release_id must be a lowercase SHA-256 digest")
    rows = list(observations)
    if not rows or len(rows) > MAX_TASK_OBSERVATIONS:
        raise ValueError("task evidence requires between 1 and 1024 observations")
    candidate_ids: set[str] = set()
    semantic_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate observation must be an object")
        _validate_observation(row, task_id, release_id)
        candidate_id = row["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError("candidate observations must have unique candidate IDs")
        candidate_ids.add(candidate_id)
        semantic_id = row["semantic_id"]
        if semantic_id and semantic_id in semantic_ids:
            raise ValueError("candidate observations must have unique semantic IDs")
        if semantic_id:
            semantic_ids.add(semantic_id)
        stratum = row.get("selection_stratum")
        probability = row.get("selection_probability")
        if stratum not in EVIDENCE_STRATA:
            raise ValueError("candidate observation has unsupported selection stratum")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(float(probability))
            or not 0.0 < float(probability) <= 1.0
        ):
            raise ValueError("candidate observation has invalid selection probability")
        _canonical_json(row)
    for name, value in telemetry_totals.items():
        if not name.strip() or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("telemetry totals must be named, finite, and nonnegative")
    return sorted(
        rows, key=lambda row: (row["candidate_id"], row.get("semantic_id", ""))
    )


def _observation_set_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _intent_payload(
    task_id: str,
    release_id: str,
    rows: list[dict[str, Any]],
    telemetry_totals: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema": TASK_EVIDENCE_INTENT_V1,
        "task_id": task_id,
        "release_id": release_id,
        "observation_set_sha256": _observation_set_sha256(rows),
        "row_count": len(rows),
        "telemetry_totals": {
            name: float(value) for name, value in sorted(telemetry_totals.items())
        },
    }


def _commit_intent(
    root: Path,
    task_id: str,
    release_id: str,
    rows: list[dict[str, Any]],
    telemetry_totals: dict[str, float],
) -> Path:
    path = _task_directory(root, task_id) / "INTENT.json"
    encoded = _canonical_json(
        _intent_payload(task_id, release_id, rows, telemetry_totals)
    )
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(
                "task evidence intent already exists with different content"
            )
    else:
        _atomic_write(path, encoded)
    return path


def _read_intent(
    root: Path, task_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    path = _task_directory(root, task_id) / "INTENT.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "task_id",
            "release_id",
            "observation_set_sha256",
            "row_count",
            "telemetry_totals",
        }
        or payload["schema"] != TASK_EVIDENCE_INTENT_V1
        or payload["task_id"] != task_id
        or payload["row_count"] != len(rows)
        or payload["observation_set_sha256"] != _observation_set_sha256(rows)
        or not _SHA256.fullmatch(str(payload["release_id"]))
        or not isinstance(payload["telemetry_totals"], dict)
    ):
        raise ValueError("task evidence intent is invalid or differs from its shard")
    telemetry: dict[str, float] = {}
    for name, value in payload["telemetry_totals"].items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("task evidence intent has invalid telemetry")
        telemetry[name] = float(value)
    payload["telemetry_totals"] = telemetry
    return payload


def _parquet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema": row["schema"],
            "task_id": row["task_id"],
            "candidate_id": row["candidate_id"],
            "semantic_id": str(row.get("semantic_id") or ""),
            "selection_stratum": row["selection_stratum"],
            "selection_probability": float(row["selection_probability"]),
            "observation_json": _canonical_json(row).decode("utf-8").rstrip("\n"),
        }
        for row in rows
    ]


def _write_shard(task_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, str]:
    staging = task_dir / "staging"
    shards = task_dir / "shards"
    staging.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=staging, prefix="candidate-observations.", suffix=".parquet.tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(_parquet_rows(rows))
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digest = _sha256_file(temporary)
        final = shards / f"{digest}.candidate-observations.parquet"
        if final.exists():
            if _sha256_file(final) != digest:
                raise ValueError(f"immutable evidence shard collision at {final}")
        else:
            os.replace(temporary, final)
            _fsync_directory(shards)
        metadata = pq.ParquetFile(final).metadata
        if metadata.num_rows != len(rows):
            raise ValueError("evidence Parquet row count changed during commit")
        codecs = {
            str(metadata.row_group(group).column(column).compression).upper()
            for group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
        }
        if codecs != {"ZSTD"}:
            raise ValueError(
                f"evidence Parquet is not uniformly Zstd-compressed: {codecs}"
            )
        return final, digest
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_for(
    root: Path,
    task_id: str,
    release_id: str,
    rows: list[dict[str, Any]],
    telemetry_totals: dict[str, float],
    shard: Path | None,
    shard_hash: str | None,
    *,
    completion_status: str = "complete",
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    if (shard is None) != (shard_hash is None):
        raise ValueError("evidence shard path and hash must be present together")
    counts = {stratum: 0 for stratum in sorted(EVIDENCE_STRATA)}
    for row in rows:
        counts[row["selection_stratum"]] += 1
    return {
        "schema": TASK_EVIDENCE_MANIFEST_V1,
        "task_id": task_id,
        "release_id": release_id,
        "row_count": len(rows),
        "row_counts_by_stratum": counts,
        "telemetry_totals": {
            name: float(value) for name, value in sorted(telemetry_totals.items())
        },
        "shards": (
            [
                {
                    "location": str(shard.relative_to(root)),
                    "sha256": shard_hash,
                    "row_count": len(rows),
                    "format": "parquet",
                    "compression": "zstd",
                }
            ]
            if shard is not None and shard_hash is not None
            else []
        ),
        "completion_status": completion_status,
        "incomplete_reason": incomplete_reason,
    }


def _safe_shard_path(root: Path, task_dir: Path, location: Any) -> Path:
    if not isinstance(location, str) or not location.strip():
        raise ValueError("task evidence shard location is empty")
    relative = Path(location)
    if "\\" in location or any(part in {"", ".", ".."} for part in location.split("/")):
        raise ValueError("task evidence shard location is not a safe relative path")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("task evidence shard location is not a safe relative path")
    resolved_root = root.resolve()
    resolved_task_shards = (task_dir / "shards").resolve()
    resolved = (root / relative).resolve()
    if (
        not resolved.is_relative_to(resolved_root)
        or resolved.parent != resolved_task_shards
    ):
        raise ValueError(
            "task evidence shard location escapes its task shard directory"
        )
    return resolved


def _validate_committed_manifest(
    root: Path, task_id: str, task_dir: Path, payload: Any
) -> None:
    expected_fields = {
        "schema",
        "task_id",
        "release_id",
        "row_count",
        "row_counts_by_stratum",
        "telemetry_totals",
        "shards",
        "completion_status",
        "incomplete_reason",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("task evidence manifest has unknown or missing fields")
    if (
        payload["schema"] != TASK_EVIDENCE_MANIFEST_V1
        or payload["task_id"] != task_id
        or not _SHA256.fullmatch(str(payload["release_id"]))
        or payload["completion_status"] != "complete"
        or payload["incomplete_reason"] is not None
        or type(payload["row_count"]) is not int
        or not 1 <= payload["row_count"] <= MAX_TASK_OBSERVATIONS
    ):
        raise ValueError("task evidence manifest has invalid complete-task authority")
    counts = payload["row_counts_by_stratum"]
    if (
        not isinstance(counts, dict)
        or set(counts) != EVIDENCE_STRATA
        or any(type(value) is not int or value < 0 for value in counts.values())
        or sum(counts.values()) != payload["row_count"]
    ):
        raise ValueError("task evidence manifest has invalid stratum row counts")
    telemetry = payload["telemetry_totals"]
    if not isinstance(telemetry, dict):
        raise ValueError("task evidence manifest telemetry must be an object")
    for name, value in telemetry.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("task evidence manifest telemetry is invalid")
    descriptors = payload["shards"]
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("complete task evidence requires at least one shard")
    locations: set[str] = set()
    hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    descriptor_rows = 0
    expected_columns = {
        "schema",
        "task_id",
        "candidate_id",
        "semantic_id",
        "selection_stratum",
        "selection_probability",
        "observation_json",
    }
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "location",
            "sha256",
            "row_count",
            "format",
            "compression",
        }:
            raise ValueError("task evidence shard descriptor has invalid fields")
        location = descriptor["location"]
        digest = descriptor["sha256"]
        row_count = descriptor["row_count"]
        if (
            location in locations
            or digest in hashes
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or type(row_count) is not int
            or row_count <= 0
            or str(descriptor["format"]).lower() != "parquet"
            or str(descriptor["compression"]).lower() != "zstd"
        ):
            raise ValueError("task evidence shard descriptor is invalid or duplicate")
        locations.add(location)
        hashes.add(digest)
        path = _safe_shard_path(root, task_dir, location)
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError("task evidence manifest references an invalid shard")
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        if (
            metadata.num_rows != row_count
            or set(parquet.schema.names) != expected_columns
        ):
            raise ValueError("task evidence Parquet schema or row count differs")
        codecs = {
            str(metadata.row_group(group).column(column).compression).upper()
            for group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
        }
        if codecs != {"ZSTD"}:
            raise ValueError("task evidence Parquet is not uniformly Zstd-compressed")
        table = pq.read_table(path, columns=["observation_json"])
        rows.extend(json.loads(value.as_py()) for value in table.column(0))
        descriptor_rows += row_count
    if descriptor_rows != payload["row_count"]:
        raise ValueError("task evidence manifest row count differs from its shards")
    validated = _validate_task(
        task_id, payload["release_id"], rows, payload["telemetry_totals"]
    )
    actual_counts = {stratum: 0 for stratum in EVIDENCE_STRATA}
    for row in validated:
        actual_counts[row["selection_stratum"]] += 1
    if any(
        actual_counts[stratum] != counts.get(stratum, 0) for stratum in EVIDENCE_STRATA
    ):
        raise ValueError(
            "task evidence manifest stratum counts differ from its shard rows"
        )


def _validate_incomplete_manifest(
    root: Path, task_id: str, task_dir: Path, payload: Any
) -> None:
    expected_fields = {
        "schema",
        "task_id",
        "release_id",
        "row_count",
        "row_counts_by_stratum",
        "telemetry_totals",
        "shards",
        "completion_status",
        "incomplete_reason",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("incomplete task evidence manifest has invalid fields")
    if (
        payload["schema"] != TASK_EVIDENCE_MANIFEST_V1
        or payload["task_id"] != task_id
        or not _SHA256.fullmatch(str(payload["release_id"]))
        or payload["completion_status"] != "incomplete"
        or not isinstance(payload["incomplete_reason"], str)
        or not payload["incomplete_reason"].strip()
        or type(payload["row_count"]) is not int
        or not 0 <= payload["row_count"] <= MAX_TASK_OBSERVATIONS
    ):
        raise ValueError("incomplete task evidence manifest has invalid authority")
    counts = payload["row_counts_by_stratum"]
    if (
        not isinstance(counts, dict)
        or set(counts) != EVIDENCE_STRATA
        or any(type(value) is not int or value < 0 for value in counts.values())
        or sum(counts.values()) != payload["row_count"]
    ):
        raise ValueError("incomplete task evidence manifest has invalid stratum counts")
    telemetry = payload["telemetry_totals"]
    if not isinstance(telemetry, dict) or any(
        not isinstance(name, str)
        or not name.strip()
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for name, value in telemetry.items()
    ):
        raise ValueError("incomplete task evidence manifest has invalid telemetry")
    descriptors = payload["shards"]
    if not isinstance(descriptors, list) or len(descriptors) != (
        1 if payload["row_count"] else 0
    ):
        raise ValueError("incomplete task evidence shard count differs from its rows")
    if not descriptors:
        return
    descriptor = descriptors[0]
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "location",
        "sha256",
        "row_count",
        "format",
        "compression",
    }:
        raise ValueError("incomplete task evidence shard descriptor is invalid")
    digest = descriptor["sha256"]
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or descriptor["row_count"] != payload["row_count"]
        or str(descriptor["format"]).lower() != "parquet"
        or str(descriptor["compression"]).lower() != "zstd"
    ):
        raise ValueError("incomplete task evidence shard identity is invalid")
    path = _safe_shard_path(root, task_dir, descriptor["location"])
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError("incomplete task evidence shard bytes are invalid")
    parquet = pq.ParquetFile(path)
    expected_columns = {
        "schema",
        "task_id",
        "candidate_id",
        "semantic_id",
        "selection_stratum",
        "selection_probability",
        "observation_json",
    }
    if (
        parquet.metadata.num_rows != payload["row_count"]
        or set(parquet.schema.names) != expected_columns
    ):
        raise ValueError("incomplete task evidence Parquet schema or rows differ")
    codecs = {
        str(parquet.metadata.row_group(group).column(column).compression).upper()
        for group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    if codecs != {"ZSTD"}:
        raise ValueError("incomplete task evidence Parquet is not Zstd-compressed")
    table = pq.read_table(path, columns=["observation_json"])
    rows = [json.loads(value.as_py()) for value in table.column(0)]
    validated = _validate_task(
        task_id, payload["release_id"], rows, payload["telemetry_totals"]
    )
    actual_counts = {stratum: 0 for stratum in EVIDENCE_STRATA}
    for row in validated:
        actual_counts[row["selection_stratum"]] += 1
    if actual_counts != counts:
        raise ValueError("incomplete task evidence counts differ from shard rows")


def _read_committed(root: Path, task_id: str) -> tuple[dict[str, Any], Path] | None:
    task_dir = _task_directory(root, task_id)
    marker = task_dir / "COMPLETE.json"
    if not marker.is_file():
        return None
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(completion, dict) or set(completion) != {
        "schema",
        "task_id",
        "manifest_sha256",
    }:
        raise ValueError("task evidence completion marker has invalid fields")
    if (
        completion["schema"] != "TaskEvidenceCompletion.v1"
        or completion["task_id"] != task_id
    ):
        raise ValueError("task evidence completion marker has invalid task authority")
    manifest_hash = completion["manifest_sha256"]
    if not _SHA256.fullmatch(str(manifest_hash or "")):
        raise ValueError("task evidence completion marker has invalid manifest hash")
    manifest = task_dir / "manifests" / f"{manifest_hash}.task-evidence-manifest.json"
    if not manifest.is_file() or _sha256_file(manifest) != manifest_hash:
        raise ValueError(
            "task evidence completion marker references an invalid manifest"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _validate_committed_manifest(root, task_id, task_dir, payload)
    return payload, manifest


def _committed_observation_index(root: Path, task_id: str) -> dict[str, dict[str, Any]]:
    committed = _read_committed(root, task_id)
    if committed is None:
        raise ValueError("candidate task evidence is not committed")
    manifest, _ = committed
    rows: dict[str, dict[str, Any]] = {}
    for descriptor in manifest["shards"]:
        table = pq.read_table(
            root / descriptor["location"], columns=["observation_json"]
        )
        for value in table.column(0):
            row = json.loads(value.as_py())
            candidate_id = row["candidate_id"]
            if candidate_id in rows:
                raise ValueError(
                    "committed candidate task contains duplicate join keys"
                )
            rows[candidate_id] = row
    return rows


def _validate_honest_candidate_join(
    evaluation: dict[str, Any], candidate: dict[str, Any]
) -> None:
    if "proxy" in evaluation["evidence_tier"].casefold():
        raise ValueError("proxy-derived tiers cannot enter honest evaluation authority")
    joined_fields = (
        "task_id",
        "task",
        "dataset",
        "source_family",
        "source_family_manifest_sha256",
        "engine",
        "candidate_id",
        "semantic_id",
        "canonical_genome",
        "selection_stratum",
        "selection_probability",
        "registry_identity",
        "release_id",
    )
    if any(evaluation[name] != candidate[name] for name in joined_fields):
        raise ValueError(
            "honest evaluation join keys differ from its committed candidate observation"
        )


def _honest_join_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "HonestEvaluationJoin.v1",
        "task_id": row["task_id"],
        "candidate_id": row["candidate_id"],
        "evaluator_version": row["evaluator_version"],
        "fold_plan": row["fold_plan"],
        "evidence_tier": row["evidence_tier"],
    }


def _arena_join_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ArenaProposalDecisionJoin.v1",
        "task_id": row["task_id"],
        "champion_checkpoint": row["champion_checkpoint"],
        "challenger_checkpoint": row["challenger_checkpoint"],
        "dataset": row["dataset"],
        "dataset_sha256": row["dataset_sha256"],
        "suite": row["suite"],
        "proposal_seed": row["proposal_seed"],
    }


def _record_directory(
    root: Path, task_id: str, kind: str, join_key: dict[str, Any]
) -> tuple[Path, str]:
    join_sha256 = hashlib.sha256(_canonical_json(join_key)).hexdigest()
    return _task_directory(root, task_id) / "records" / kind / join_sha256, join_sha256


def _read_bound_record(
    root: Path,
    *,
    task_id: str,
    kind: str,
    join_key: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], EvidenceRecordResult]:
    directory, join_sha256 = _record_directory(root, task_id, kind, join_key)
    binding_path = directory / "COMMITTED.json"
    if not binding_path.is_file():
        raise FileNotFoundError("evidence record binding is not committed")
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    expected_fields = {
        "schema",
        "task_id",
        "record_schema",
        "join_key",
        "join_sha256",
        "content_sha256",
        "record_file",
    }
    if not isinstance(binding, dict) or set(binding) != expected_fields:
        raise ValueError("evidence record binding has unknown or missing fields")
    content_sha256 = binding["content_sha256"]
    expected_file = f"{content_sha256}.record.json"
    if (
        binding["schema"] != EVIDENCE_RECORD_BINDING_V1
        or binding["task_id"] != task_id
        or binding["join_key"] != join_key
        or binding["join_sha256"] != join_sha256
        or not isinstance(content_sha256, str)
        or not _SHA256.fullmatch(content_sha256)
        or binding["record_file"] != expected_file
        or binding_bytes != _canonical_json(binding)
    ):
        raise ValueError("evidence record binding has invalid join authority")
    record_path = directory / expected_file
    records = sorted(directory.glob("*.record.json"))
    if records != [record_path]:
        raise ValueError("evidence record binding has ambiguous append-only content")
    if not record_path.is_file() or _sha256_file(record_path) != content_sha256:
        raise ValueError("evidence record binding references invalid content")
    encoded = record_path.read_bytes()
    row = json.loads(encoded)
    if not isinstance(row, dict) or encoded != _canonical_json(row):
        raise ValueError("evidence record content is not canonical")
    validator(row)
    if binding["record_schema"] != row.get("schema"):
        raise ValueError("evidence record binding schema differs from its content")
    return row, EvidenceRecordResult(
        task_id=task_id,
        record_schema=binding["record_schema"],
        content_sha256=content_sha256,
        record_path=record_path,
        binding_path=binding_path,
    )


def _commit_bound_record(
    root: Path,
    *,
    task_id: str,
    kind: str,
    join_key: dict[str, Any],
    row: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
) -> EvidenceRecordResult:
    validator(row)
    encoded = _canonical_json(row)
    content_sha256 = hashlib.sha256(encoded).hexdigest()
    directory, join_sha256 = _record_directory(root, task_id, kind, join_key)
    binding_path = directory / "COMMITTED.json"
    if binding_path.exists():
        stored, result = _read_bound_record(
            root,
            task_id=task_id,
            kind=kind,
            join_key=join_key,
            validator=validator,
        )
        if _canonical_json(stored) != encoded:
            raise ValueError(
                "evidence record join key is already committed with different content"
            )
        return result

    record_path = directory / f"{content_sha256}.record.json"
    existing_records = sorted(directory.glob("*.record.json"))
    if existing_records and existing_records != [record_path]:
        raise ValueError("evidence record join key has a conflicting pre-commit append")
    if record_path.exists():
        if record_path.read_bytes() != encoded:
            raise ValueError("immutable evidence record content collision")
    else:
        _atomic_write(record_path, encoded)
    binding = {
        "schema": EVIDENCE_RECORD_BINDING_V1,
        "task_id": task_id,
        "record_schema": row["schema"],
        "join_key": join_key,
        "join_sha256": join_sha256,
        "content_sha256": content_sha256,
        "record_file": record_path.name,
    }
    _atomic_write(binding_path, _canonical_json(binding))
    stored, result = _read_bound_record(
        root,
        task_id=task_id,
        kind=kind,
        join_key=join_key,
        validator=validator,
    )
    if _canonical_json(stored) != encoded:
        raise ValueError("evidence record changed during commit readback")
    return result


def commit_honest_evaluation(
    root: str | Path, evaluation: dict[str, Any]
) -> EvidenceRecordResult:
    """Append one candidate-joined honest evaluation and verify marker-last readback."""
    root = Path(root)
    row = json.loads(_canonical_json(evaluation))
    _validate_honest_evaluation(row)
    with _task_commit_lock(root, row["task_id"]):
        candidates = _committed_observation_index(root, row["task_id"])
        candidate = candidates.get(row["candidate_id"])
        if candidate is None:
            raise ValueError(
                "honest evaluation candidate is absent from committed task evidence"
            )
        _validate_honest_candidate_join(row, candidate)
        return _commit_bound_record(
            root,
            task_id=row["task_id"],
            kind="honest-evaluations",
            join_key=_honest_join_key(row),
            row=row,
            validator=_validate_honest_evaluation,
        )


def read_honest_evaluation(
    root: str | Path,
    *,
    task_id: str,
    candidate_id: str,
    evaluator_version: str,
    fold_plan: str,
    evidence_tier: str,
) -> dict[str, Any]:
    """Read one immutable honest row and re-prove its candidate join."""
    key = {
        "schema": "HonestEvaluationJoin.v1",
        "task_id": _require_nonempty_string(task_id, "task_id"),
        "candidate_id": _require_nonempty_string(candidate_id, "candidate_id"),
        "evaluator_version": _require_nonempty_string(
            evaluator_version, "evaluator_version"
        ),
        "fold_plan": _require_nonempty_string(fold_plan, "fold_plan"),
        "evidence_tier": _require_nonempty_string(evidence_tier, "evidence_tier"),
    }
    root = Path(root)
    row, _ = _read_bound_record(
        root,
        task_id=task_id,
        kind="honest-evaluations",
        join_key=key,
        validator=_validate_honest_evaluation,
    )
    candidates = _committed_observation_index(root, task_id)
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise ValueError("honest evaluation candidate observation is missing")
    _validate_honest_candidate_join(row, candidate)
    return row


def _validate_arena_joins(root: Path, decision: dict[str, Any]) -> None:
    candidates = _committed_observation_index(root, decision["task_id"])
    for proposal in decision["proposals"]:
        if proposal["rank"] is None:
            continue
        candidate = candidates.get(proposal["candidate_id"])
        if candidate is None:
            raise ValueError(
                "selected arena proposal is absent from committed task evidence"
            )
        if (
            candidate["task_id"] != decision["task_id"]
            or candidate["dataset"] != decision["dataset"]
            or candidate["engine"] != proposal["engine"]
            or candidate["raw_genome"] != proposal["raw_genome"]
            or candidate["canonical_genome"] != proposal["canonical_genome"]
            or candidate["semantic_id"] != proposal["semantic_id"]
            or candidate["registry_identity"] != decision["registry_identity"]
            or candidate["release_id"] != decision["release_id"]
        ):
            raise ValueError(
                "selected arena proposal join keys differ from its candidate observation"
            )
        evaluation = proposal["cv_result"]
        if evaluation is None:
            continue
        if (
            evaluation["task_id"] != decision["task_id"]
            or evaluation["dataset"] != decision["dataset"]
            or evaluation["registry_identity"] != decision["registry_identity"]
            or evaluation["release_id"] != decision["release_id"]
        ):
            raise ValueError("arena CV result differs from decision authority")
        _validate_honest_candidate_join(evaluation, candidate)
        persisted, _ = _read_bound_record(
            root,
            task_id=evaluation["task_id"],
            kind="honest-evaluations",
            join_key=_honest_join_key(evaluation),
            validator=_validate_honest_evaluation,
        )
        if _canonical_json(persisted) != _canonical_json(evaluation):
            raise ValueError("arena CV result differs from persisted honest evaluation")


def commit_arena_proposal_decision(
    root: str | Path, decision: dict[str, Any]
) -> EvidenceRecordResult:
    """Append one strict 160-to-10 arena decision after all selected joins verify."""
    root = Path(root)
    row = json.loads(_canonical_json(decision))
    _validate_arena_proposal_decision(row)
    with _task_commit_lock(root, row["task_id"]):
        _validate_arena_joins(root, row)
        return _commit_bound_record(
            root,
            task_id=row["task_id"],
            kind="arena-proposal-decisions",
            join_key=_arena_join_key(row),
            row=row,
            validator=_validate_arena_proposal_decision,
        )


def read_arena_proposal_decision(
    root: str | Path,
    *,
    task_id: str,
    champion_checkpoint: str,
    challenger_checkpoint: str,
    dataset: str,
    dataset_sha256: str,
    suite: str,
    proposal_seed: int,
) -> dict[str, Any]:
    """Read one immutable arena decision and re-prove selected candidate/CV joins."""
    key = {
        "schema": "ArenaProposalDecisionJoin.v1",
        "task_id": task_id,
        "champion_checkpoint": champion_checkpoint,
        "challenger_checkpoint": challenger_checkpoint,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "suite": suite,
        "proposal_seed": proposal_seed,
    }
    root = Path(root)
    row, _ = _read_bound_record(
        root,
        task_id=task_id,
        kind="arena-proposal-decisions",
        join_key=key,
        validator=_validate_arena_proposal_decision,
    )
    _validate_arena_joins(root, row)
    return row


def _validate_evidence_persistence_batch(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "schema",
        "intent_sha256",
        "phase",
        "task_id",
        "release_id",
        "observations",
        "honest_evaluations",
        "arena_decision",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("evidence persistence batch has unknown or missing fields")
    if payload["schema"] != EVIDENCE_PERSISTENCE_BATCH_V1:
        raise ValueError("unsupported evidence persistence batch schema")
    if not isinstance(payload["intent_sha256"], str) or not _SHA256.fullmatch(
        payload["intent_sha256"]
    ):
        raise ValueError("evidence persistence batch intent hash is invalid")
    phase = payload["phase"]
    if phase not in {"candidates", "honest", "decision"}:
        raise ValueError("evidence persistence batch phase is invalid")
    task_id = _require_nonempty_string(payload["task_id"], "batch task_id")
    release_id = payload["release_id"]
    if not isinstance(release_id, str) or not _SHA256.fullmatch(release_id):
        raise ValueError("evidence persistence batch release_id is invalid")
    if not isinstance(payload["observations"], list):
        raise ValueError("evidence persistence batch observations must be a list")
    observations = _validate_task(
        task_id,
        release_id,
        payload["observations"],
        {},
    )
    if len(observations) != 10:
        raise ValueError("arena evidence persistence requires exactly ten observations")
    observations_by_candidate = {row["candidate_id"]: row for row in observations}

    evaluations = payload["honest_evaluations"]
    if not isinstance(evaluations, list):
        raise ValueError("evidence persistence batch honest rows must be a list")
    expected_honest = {"candidates": 0, "honest": 1, "decision": 10}[phase]
    if len(evaluations) != expected_honest:
        raise ValueError(
            f"{phase} evidence persistence requires {expected_honest} honest rows"
        )
    evaluation_ids: set[str] = set()
    for evaluation in evaluations:
        _validate_honest_evaluation(evaluation)
        candidate = observations_by_candidate.get(evaluation["candidate_id"])
        if candidate is None:
            raise ValueError("honest batch row has no selected candidate observation")
        _validate_honest_candidate_join(evaluation, candidate)
        if evaluation["candidate_id"] in evaluation_ids:
            raise ValueError("evidence persistence batch repeats an honest candidate")
        evaluation_ids.add(evaluation["candidate_id"])

    decision = payload["arena_decision"]
    if phase == "decision":
        if not isinstance(decision, dict):
            raise ValueError("decision evidence persistence requires an arena decision")
        _validate_arena_proposal_decision(decision)
        if decision["task_id"] != task_id or decision["release_id"] != release_id:
            raise ValueError("arena decision differs from its persistence batch")
        evaluation_by_candidate = {row["candidate_id"]: row for row in evaluations}
        selected = [
            proposal
            for proposal in decision["proposals"]
            if proposal["rank"] is not None
        ]
        if {proposal["candidate_id"] for proposal in selected} != set(
            evaluation_by_candidate
        ):
            raise ValueError(
                "arena decision selected set differs from honest batch rows"
            )
        for proposal in selected:
            if _canonical_json(proposal["cv_result"]) != _canonical_json(
                evaluation_by_candidate[proposal["candidate_id"]]
            ):
                raise ValueError(
                    "arena decision CV bytes differ from its honest batch row"
                )
    elif decision is not None:
        raise ValueError(
            "only decision evidence persistence may carry an arena decision"
        )
    return observations, evaluations


def _validate_gp_evidence_persistence_batch(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    required = {
        "schema",
        "intent_sha256",
        "task_id",
        "release_id",
        "observations",
        "telemetry_totals",
        "completion_status",
        "incomplete_reason",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("GP evidence persistence batch has unknown or missing fields")
    if payload["schema"] != GP_EVIDENCE_PERSISTENCE_BATCH_V1:
        raise ValueError("unsupported GP evidence persistence batch schema")
    if not isinstance(payload["intent_sha256"], str) or not _SHA256.fullmatch(
        payload["intent_sha256"]
    ):
        raise ValueError("GP evidence persistence intent hash is invalid")
    task_id = _require_nonempty_string(payload["task_id"], "GP batch task_id")
    release_id = payload["release_id"]
    if not isinstance(release_id, str) or not _SHA256.fullmatch(release_id):
        raise ValueError("GP evidence persistence release_id is invalid")
    status = payload["completion_status"]
    reason = payload["incomplete_reason"]
    if (status == "complete" and reason is None) or (
        status == "incomplete" and isinstance(reason, str) and reason.strip()
    ):
        pass
    else:
        raise ValueError("GP evidence completion status differs from its reason")
    observations = payload["observations"]
    if not isinstance(observations, list):
        raise ValueError("GP evidence observations must be a list")
    telemetry_payload = payload["telemetry_totals"]
    if not isinstance(telemetry_payload, dict):
        raise ValueError("GP evidence telemetry must be an object")
    telemetry: dict[str, float] = {}
    for name, value in telemetry_payload.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("GP evidence telemetry must be named and nonnegative")
        telemetry[name] = float(value)
    if observations:
        rows = _validate_task(task_id, release_id, observations, telemetry)
    else:
        rows = []
    if status == "complete" and not rows:
        raise ValueError("complete GP evidence requires at least one observation")
    return rows, telemetry


def commit_task_evidence(
    root: str | Path,
    *,
    task_id: str,
    release_id: str,
    observations: Iterable[dict[str, Any]],
    telemetry_totals: dict[str, float] | None = None,
) -> EvidenceWriteResult:
    """Durably commit one complete immutable task sample."""
    root = Path(root)
    telemetry = dict(telemetry_totals or {})
    rows = _validate_task(task_id, release_id, observations, telemetry)
    with _task_commit_lock(root, task_id):
        existing = _read_committed(root, task_id)
        if existing is not None:
            manifest, manifest_path = existing
            stored_rows = []
            for descriptor in manifest["shards"]:
                table = pq.read_table(
                    root / descriptor["location"], columns=["observation_json"]
                )
                stored_rows.extend(
                    json.loads(value.as_py()) for value in table.column(0)
                )
            if (
                sorted(
                    stored_rows,
                    key=lambda row: (
                        row["candidate_id"],
                        row.get("semantic_id", ""),
                    ),
                )
                != rows
                or manifest["release_id"] != release_id
                or manifest["telemetry_totals"]
                != {name: float(value) for name, value in sorted(telemetry.items())}
            ):
                raise ValueError(
                    "task evidence is already committed with different content"
                )
            return EvidenceWriteResult(task_id, "complete", manifest_path)

        task_dir = _task_directory(root, task_id)
        _commit_intent(root, task_id, release_id, rows, telemetry)
        shard, shard_hash = _write_shard(task_dir, rows)
        manifest = _manifest_for(
            root, task_id, release_id, rows, telemetry, shard, shard_hash
        )
        manifest_bytes = _canonical_json(manifest)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = (
            task_dir / "manifests" / f"{manifest_hash}.task-evidence-manifest.json"
        )
        if manifest_path.exists():
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError(
                    f"immutable evidence manifest collision at {manifest_path}"
                )
        else:
            _atomic_write(manifest_path, manifest_bytes)
        _atomic_write(
            task_dir / "COMPLETE.json",
            _canonical_json(
                {
                    "schema": "TaskEvidenceCompletion.v1",
                    "task_id": task_id,
                    "manifest_sha256": manifest_hash,
                }
            ),
        )
        _read_committed(root, task_id)
        return EvidenceWriteResult(task_id, "complete", manifest_path)


def commit_incomplete_task_evidence(
    root: str | Path,
    *,
    task_id: str,
    release_id: str,
    observations: Iterable[dict[str, Any]],
    telemetry_totals: dict[str, float] | None,
    reason: str,
) -> EvidenceWriteResult:
    """Persist recoverable rows without creating task-completion authority."""
    root = Path(root)
    _require_nonempty_string(task_id, "task_id")
    if not _SHA256.fullmatch(release_id):
        raise ValueError("release_id must be a lowercase SHA-256 digest")
    _require_nonempty_string(reason, "incomplete reason")
    telemetry_payload = dict(telemetry_totals or {})
    telemetry: dict[str, float] = {}
    for name, value in telemetry_payload.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("telemetry totals must be named, finite, and nonnegative")
        telemetry[name] = float(value)
    snapshot = _snapshot_observations(observations)
    rows = _validate_task(task_id, release_id, snapshot, telemetry) if snapshot else []
    with _task_commit_lock(root, task_id):
        existing = _read_committed(root, task_id)
        if existing is not None:
            manifest, manifest_path = existing
            if manifest["release_id"] != release_id:
                raise ValueError(
                    "complete task evidence has a different release identity"
                )
            return EvidenceWriteResult(task_id, "complete", manifest_path)
        task_dir = _task_directory(root, task_id)
        shard: Path | None = None
        shard_hash: str | None = None
        if rows:
            shard, shard_hash = _write_shard(task_dir, rows)
        manifest = _manifest_for(
            root,
            task_id,
            release_id,
            rows,
            telemetry,
            shard,
            shard_hash,
            completion_status="incomplete",
            incomplete_reason=reason,
        )
        manifest_bytes = _canonical_json(manifest)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = (
            task_dir / "manifests" / f"{manifest_hash}.task-evidence-manifest.json"
        )
        if manifest_path.exists():
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("immutable incomplete evidence manifest collision")
        else:
            _atomic_write(manifest_path, manifest_bytes)
        if _sha256_file(manifest_path) != manifest_hash:
            raise ValueError("incomplete evidence manifest changed during commit")
        _validate_incomplete_manifest(root, task_id, task_dir, manifest)
        if (task_dir / "COMPLETE.json").exists():
            raise ValueError(
                "incomplete evidence unexpectedly created a completion marker"
            )
        return EvidenceWriteResult(task_id, "incomplete", manifest_path, reason)


def commit_evidence_persistence_batch(
    root: str | Path, batch: dict[str, Any]
) -> dict[str, Any]:
    """Commit one Rust-produced arena phase in dependency order.

    Candidate observations are committed first on every phase so a replayed
    honest or decision intent repairs a missed earlier intent idempotently.
    The decision is committed only after all ten separately bound honest rows.
    """
    row = json.loads(_canonical_json(batch))
    observations, evaluations = _validate_evidence_persistence_batch(row)
    root = Path(root)
    manifest = commit_task_evidence(
        root,
        task_id=row["task_id"],
        release_id=row["release_id"],
        observations=observations,
    )
    honest_receipts = [
        commit_honest_evaluation(root, evaluation) for evaluation in evaluations
    ]
    arena_receipt = (
        commit_arena_proposal_decision(root, row["arena_decision"])
        if row["arena_decision"] is not None
        else None
    )
    return {
        "schema": "EvidencePersistenceBatchReceipt.v1",
        "intent_sha256": row["intent_sha256"],
        "phase": row["phase"],
        "task_id": row["task_id"],
        "status": "complete",
        "manifest_path": str(manifest.manifest_path),
        "honest_content_sha256": [
            receipt.content_sha256 for receipt in honest_receipts
        ],
        "arena_content_sha256": (
            arena_receipt.content_sha256 if arena_receipt is not None else None
        ),
    }


def commit_gp_evidence_persistence_batch(
    root: str | Path, batch: dict[str, Any]
) -> dict[str, Any]:
    """Commit proxy-only GP rows, leaving incomplete sources marker-free."""
    row = json.loads(_canonical_json(batch))
    observations, telemetry = _validate_gp_evidence_persistence_batch(row)
    root = Path(root)
    if row["completion_status"] == "complete":
        result = commit_task_evidence(
            root,
            task_id=row["task_id"],
            release_id=row["release_id"],
            observations=observations,
            telemetry_totals=telemetry,
        )
    else:
        result = commit_incomplete_task_evidence(
            root,
            task_id=row["task_id"],
            release_id=row["release_id"],
            observations=observations,
            telemetry_totals=telemetry,
            reason=row["incomplete_reason"],
        )
    return {
        "schema": "GpEvidencePersistenceBatchReceipt.v1",
        "intent_sha256": row["intent_sha256"],
        "task_id": row["task_id"],
        "status": (
            "complete" if result.status == "complete" else "evidence_incomplete"
        ),
        "manifest_path": str(result.manifest_path),
        "row_count": len(observations),
        "reason": result.reason,
    }


def _commit_incomplete_manifest(
    root: Path, task_id: str, release_id: str, reason: str
) -> Path | None:
    if not task_id.strip() or not _SHA256.fullmatch(release_id) or not reason.strip():
        return None
    manifest = {
        "schema": TASK_EVIDENCE_MANIFEST_V1,
        "task_id": task_id,
        "release_id": release_id,
        "row_count": 0,
        "row_counts_by_stratum": {stratum: 0 for stratum in sorted(EVIDENCE_STRATA)},
        "telemetry_totals": {},
        "shards": [],
        "completion_status": "incomplete",
        "incomplete_reason": reason,
    }
    encoded = _canonical_json(manifest)
    digest = hashlib.sha256(encoded).hexdigest()
    path = (
        _task_directory(root, task_id)
        / "manifests"
        / f"{digest}.task-evidence-manifest.json"
    )
    try:
        if path.exists():
            if path.read_bytes() != encoded:
                return None
        else:
            _atomic_write(path, encoded)
        return path
    except OSError:
        return None


def _snapshot_observations(
    observations: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    snapshot: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if index >= MAX_TASK_OBSERVATIONS:
            raise ValueError("task evidence exceeds the 1024-observation limit")
        if not isinstance(observation, dict):
            raise ValueError("candidate observation must be an object")
        snapshot.append(json.loads(_canonical_json(observation)))
    return tuple(snapshot)


class AsyncEvidenceWriter:
    """Bounded best-effort queue; enqueue and write failures never fail search."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_pending_tasks: int = 32,
        max_results: int | None = None,
    ):
        if max_pending_tasks <= 0 or (max_results is not None and max_results <= 0):
            raise ValueError("evidence writer queue/result bounds must be positive")
        self.root = Path(root)
        self._queue: queue.Queue[_TaskWrite | None] = queue.Queue(max_pending_tasks)
        self._incomplete_queue: queue.Queue[_IncompleteWrite | None] = queue.Queue(
            max_pending_tasks
        )
        self._max_results = max_results or max_pending_tasks * 4
        self._results: OrderedDict[str, EvidenceWriteResult] = OrderedDict()
        self._condition = threading.Condition()
        self._state_lock = threading.Lock()
        self._pending: set[str] = set()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="prime-evidence-writer", daemon=True
        )
        self._incomplete_thread = threading.Thread(
            target=self._run_incomplete,
            name="prime-evidence-incomplete-writer",
            daemon=True,
        )
        self._thread.start()
        self._incomplete_thread.start()

    def submit(
        self,
        *,
        task_id: str,
        release_id: str,
        observations: Iterable[dict[str, Any]],
        telemetry_totals: dict[str, float] | None = None,
    ) -> EvidenceSubmission:
        try:
            frozen_observations = _snapshot_observations(observations)
            frozen_telemetry = json.loads(_canonical_json(telemetry_totals or {}))
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            immediate = False
            with self._state_lock:
                if not self._closed and task_id not in self._pending:
                    self._pending.add(task_id)
                    try:
                        self._incomplete_queue.put_nowait(
                            _IncompleteWrite(task_id, release_id, reason)
                        )
                    except queue.Full:
                        immediate = True
            if immediate:
                self._record_result(
                    EvidenceWriteResult(
                        task_id,
                        "incomplete",
                        _commit_incomplete_manifest(
                            self.root, task_id, release_id, reason
                        ),
                        reason,
                    )
                )
            return EvidenceSubmission(False, task_id, reason)
        task = _TaskWrite(task_id, release_id, frozen_observations, frozen_telemetry)
        with self._state_lock:
            if self._closed:
                return EvidenceSubmission(False, task_id, "evidence writer is closed")
            if task_id in self._pending:
                return EvidenceSubmission(
                    False, task_id, "evidence task is already pending"
                )
        try:
            _commit_retry_spool(self.root, task)
        except Exception as error:
            reason = f"durable retry spool failed: {type(error).__name__}: {error}"
            self._record_result(
                EvidenceWriteResult(
                    task_id,
                    "incomplete",
                    _commit_incomplete_manifest(self.root, task_id, release_id, reason),
                    reason,
                )
            )
            return EvidenceSubmission(False, task_id, reason)
        immediate: EvidenceWriteResult | None = None
        accepted = False
        rejection_reason: str | None = None
        with self._state_lock:
            if self._closed:
                reason = "evidence writer is closed after durable retry commit"
                immediate = EvidenceWriteResult(task_id, "incomplete", None, reason)
            if task_id in self._pending:
                return EvidenceSubmission(
                    False, task_id, "evidence task is already pending"
                )
            if immediate is None:
                self._pending.add(task_id)
                try:
                    self._queue.put_nowait(task)
                    accepted = True
                except queue.Full:
                    reason = "evidence queue is full; durable retry is pending"
                    rejection_reason = reason
                    try:
                        self._incomplete_queue.put_nowait(
                            _IncompleteWrite(task_id, release_id, reason)
                        )
                    except queue.Full:
                        immediate = EvidenceWriteResult(
                            task_id, "incomplete", None, reason
                        )
        if immediate is not None:
            if immediate.manifest_path is None:
                immediate = EvidenceWriteResult(
                    task_id,
                    "incomplete",
                    _commit_incomplete_manifest(
                        self.root,
                        task_id,
                        release_id,
                        immediate.reason or "evidence write incomplete",
                    ),
                    immediate.reason,
                )
            self._record_result(immediate)
            return EvidenceSubmission(False, task_id, immediate.reason)
        return EvidenceSubmission(accepted, task_id, rejection_reason)

    def wait(
        self, task_id: str, timeout: float | None = None
    ) -> EvidenceWriteResult | None:
        with self._condition:
            self._condition.wait_for(lambda: task_id in self._results, timeout=timeout)
            return self._results.get(task_id)

    def acknowledge(self, task_id: str) -> EvidenceWriteResult | None:
        """Remove and return one completed result from the bounded result cache."""
        with self._condition:
            return self._results.pop(task_id, None)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
            self._incomplete_queue.put(None)
        self._thread.join()
        self._incomplete_thread.join()

    def _record_result(self, result: EvidenceWriteResult) -> None:
        with self._state_lock:
            self._pending.discard(result.task_id)
        with self._condition:
            self._results[result.task_id] = result
            self._results.move_to_end(result.task_id)
            while len(self._results) > self._max_results:
                self._results.popitem(last=False)
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            try:
                result = commit_task_evidence(
                    self.root,
                    task_id=task.task_id,
                    release_id=task.release_id,
                    observations=task.observations,
                    telemetry_totals=task.telemetry_totals,
                )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                result = EvidenceWriteResult(
                    task.task_id,
                    "incomplete",
                    _commit_incomplete_manifest(
                        self.root, task.task_id, task.release_id, reason
                    ),
                    reason,
                )
            if result.status == "complete":
                try:
                    _remove_retry_spool(self.root, task)
                except Exception as error:
                    result = EvidenceWriteResult(
                        task.task_id,
                        "complete",
                        result.manifest_path,
                        f"retry acknowledgement failed: {type(error).__name__}: {error}",
                    )
            self._record_result(result)
            self._queue.task_done()

    def _run_incomplete(self) -> None:
        while True:
            task = self._incomplete_queue.get()
            if task is None:
                self._incomplete_queue.task_done()
                return
            manifest_path = _commit_incomplete_manifest(
                self.root, task.task_id, task.release_id, task.reason
            )
            result = EvidenceWriteResult(
                task.task_id, "incomplete", manifest_path, task.reason
            )
            self._record_result(result)
            self._incomplete_queue.task_done()

    def __enter__(self) -> AsyncEvidenceWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def recover_orphan_shards(root: str | Path) -> list[EvidenceWriteResult]:
    """Replay durable retries and finish shards left before manifest commit."""
    root = Path(root)
    recovered: list[EvidenceWriteResult] = []
    for retry_path in sorted((root / "tasks").glob("*/RETRY.json")):
        try:
            task = _read_retry_spool(retry_path)
            if (
                retry_path.parent.resolve()
                != _task_directory(root, task.task_id).resolve()
            ):
                raise ValueError(
                    "task evidence retry spool directory is not task-bound"
                )
            committed = _read_committed(root, task.task_id)
            if committed is None:
                recovered.append(
                    commit_task_evidence(
                        root,
                        task_id=task.task_id,
                        release_id=task.release_id,
                        observations=task.observations,
                        telemetry_totals=task.telemetry_totals,
                    )
                )
            _remove_retry_spool(root, task)
        except Exception:
            # A corrupt or temporarily unwritable retry is isolated and left in
            # place for operator inspection or a later recovery pass.
            continue
    for task_dir in sorted((root / "tasks").glob("*")):
        try:
            marker = task_dir / "COMPLETE.json"
            if marker.exists():
                completion = json.loads(marker.read_text(encoding="utf-8"))
                committed_task_id = completion.get("task_id")
                if not isinstance(committed_task_id, str):
                    raise ValueError("completion marker has no task identity")
                _read_committed(root, committed_task_id)
                continue
            for temporary in (task_dir / "staging").glob("*.tmp"):
                temporary.unlink(missing_ok=True)
            shards = sorted(
                (task_dir / "shards").glob("*.candidate-observations.parquet")
            )
            if len(shards) != 1:
                continue
            shard = shards[0]
            expected_hash = shard.name.split(".", 1)[0]
            if (
                not _SHA256.fullmatch(expected_hash)
                or _sha256_file(shard) != expected_hash
            ):
                continue
            table = pq.read_table(shard, columns=["observation_json"])
            rows = [json.loads(value.as_py()) for value in table.column(0)]
            if not rows:
                continue
            task_id = rows[0].get("task_id")
            release_id = rows[0].get("release_id")
            intent = _read_intent(root, task_id, rows)
            if intent is not None:
                release_id = intent["release_id"]
                telemetry_totals = intent["telemetry_totals"]
            else:
                telemetry_totals = {}
            recovered.append(
                commit_task_evidence(
                    root,
                    task_id=task_id,
                    release_id=release_id,
                    observations=rows,
                    telemetry_totals=telemetry_totals,
                )
            )
        except Exception:
            # Recovery is best-effort and task-isolated. One corrupt shard or
            # marker must never prevent valid tasks from being recovered.
            continue
    return recovered


def _commit_batch_command(root: Path, request_path: Path) -> int:
    payload: Any = None
    try:
        encoded = request_path.read_bytes()
        request_hash = request_path.name.split(".", 1)[0]
        if (
            not _SHA256.fullmatch(request_hash)
            or hashlib.sha256(encoded).hexdigest() != request_hash
        ):
            raise ValueError(
                "evidence capture request filename is not content-addressed"
            )
        payload = json.loads(encoded)
        receipt = commit_evidence_persistence_batch(root, payload)
        sys.stdout.buffer.write(_canonical_json(receipt) + b"\n")
        return 0
    except Exception as error:  # noqa: BLE001 - process boundary reports a classified retry
        reason = f"{type(error).__name__}: {error}"
        if isinstance(payload, dict):
            task_id = payload.get("task_id")
            release_id = payload.get("release_id")
            if isinstance(task_id, str) and isinstance(release_id, str):
                _commit_incomplete_manifest(root, task_id, release_id, reason)
        sys.stderr.buffer.write(
            _canonical_json(
                {
                    "schema": "EvidencePersistenceBatchReceipt.v1",
                    "status": "evidence_incomplete",
                    "reason": reason,
                }
            )
            + b"\n"
        )
        return 1


def _commit_gp_batch_command(root: Path, request_path: Path) -> int:
    payload: Any = None
    try:
        encoded = request_path.read_bytes()
        request_hash = request_path.name.split(".", 1)[0]
        if (
            not _SHA256.fullmatch(request_hash)
            or hashlib.sha256(encoded).hexdigest() != request_hash
        ):
            raise ValueError(
                "GP evidence capture request filename is not content-addressed"
            )
        payload = json.loads(encoded)
        receipt = commit_gp_evidence_persistence_batch(root, payload)
        receipt_bytes = _canonical_json(receipt) + b"\n"
        if receipt["status"] == "complete":
            sys.stdout.buffer.write(receipt_bytes)
            return 0
        sys.stderr.buffer.write(receipt_bytes)
        return 1
    except Exception as error:  # noqa: BLE001 - process boundary classifies retry
        reason = f"{type(error).__name__}: {error}"
        if isinstance(payload, dict):
            task_id = payload.get("task_id")
            release_id = payload.get("release_id")
            if isinstance(task_id, str) and isinstance(release_id, str):
                _commit_incomplete_manifest(root, task_id, release_id, reason)
        sys.stderr.buffer.write(
            _canonical_json(
                {
                    "schema": "GpEvidencePersistenceBatchReceipt.v1",
                    "status": "evidence_incomplete",
                    "reason": reason,
                }
            )
            + b"\n"
        )
        return 1


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prime evidence persistence bridge")
    subcommands = parser.add_subparsers(dest="command", required=True)
    commit = subcommands.add_parser(
        "commit-arena-capture",
        help="commit one content-addressed Rust arena evidence batch",
    )
    commit.add_argument("--root", type=Path, required=True)
    commit.add_argument("--request", type=Path, required=True)
    commit_gp = subcommands.add_parser(
        "commit-gp-capture",
        help="commit one content-addressed Rust GP evidence batch",
    )
    commit_gp.add_argument("--root", type=Path, required=True)
    commit_gp.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "commit-arena-capture":
        return _commit_batch_command(args.root, args.request)
    if args.command == "commit-gp-capture":
        return _commit_gp_batch_command(args.root, args.request)
    parser.error("unsupported evidence persistence command")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
