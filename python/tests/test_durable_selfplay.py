from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import test_exact_backlog as backlog_module  # noqa: E402
import test_exact_service as service_module  # noqa: E402
from build_selfplay_bundle import sources as bundle_sources  # noqa: E402
from test_exact_accept import evaluate_gate  # noqa: E402
from test_exact_common import (  # noqa: E402
    EXPECTED_METRIC,
    dynamic_queue_targets,
    evaluation_protocol_sha256,
    sha256_text,
    validate_catalog_counts,
    validate_catalog_manifest,
)
from test_exact_controller import default_handlers  # noqa: E402
from test_exact_evaluator import (  # noqa: E402
    active,
    candidate,
    load_pointer,
    promote_candidate,
)
from test_exact_ingest import local_receipt_dir  # noqa: E402
from test_exact_pair_registry import PairRegistry  # noqa: E402
from test_exact_sealed_audit import SealedAuditRegistry  # noqa: E402
from test_exact_service import loaded_bundle, record_starting  # noqa: E402
from test_exact_status import service_is_stale  # noqa: E402
from test_exact_wave import wave_seed  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_fixture(tmp_path: Path) -> Path:
    contexts = [b"a" * 8192, b"b" * 8192]
    data = tmp_path / "contexts.bin"
    data.write_bytes(b"".join(contexts))
    entries = []
    for index, task in enumerate(("regression", "binary")):
        entries.append(
            {
                "catalog_index": index,
                "task": task,
                "dataset_hash": f"d-{index}",
                "metric": "r2" if task == "regression" else "auc",
                "source_family": f"family-{index}",
                "split": "train" if index == 0 else "validation",
                "context_sha256": hashlib.sha256(contexts[index]).hexdigest(),
                "train_sha256": str(index + 1) * 64,
                "test_sha256": str(index + 3) * 64,
            }
        )
    manifest = {
        "schema": "JopeTestOnlyContextCatalog.v1",
        "immutable": True,
        "data_sha256": digest(data),
        "record_stride_bytes": 8192,
        "row_count": 2,
        "regression_count": 1,
        "binary_count": 1,
        "source_family_disjoint": True,
        "entries": entries,
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(manifest))
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    return pointer


def protocol(task="regression", version="v1"):
    return evaluation_protocol_sha256(
        task=task,
        metric="r2" if task == "regression" else "auc",
        score_tier="test_exact",
        train_sha256="1" * 64,
        test_sha256="2" * 64,
        version=version,
    )


def gate_evidence(lift: float) -> dict:
    return {
        "baseline": {
            "checkpoint_sha256": "a" * 64,
            "macro_pairwise_accuracy": 0.60,
            "regression_pairwise_accuracy": 0.60,
            "binary_pairwise_accuracy": 0.60,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.02,
        },
        "candidate": {
            "checkpoint_sha256": "b" * 64,
            "macro_pairwise_accuracy": 0.60 + lift,
            "regression_pairwise_accuracy": 0.60 + lift,
            "binary_pairwise_accuracy": 0.60 + lift,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.06,
        },
        "proposal_validity": 0.99,
        "verified_checkpoint_origin_strict_wins": 1,
        "candidate_equal_budget_result": 0.51,
        "baseline_equal_budget_result": 0.49,
        "replay_verification": {
            key: True
            for key in (
                "predictions",
                "scores",
                "evaluator_identity",
                "dataset_hashes",
                "candidate_origin",
            )
        },
    }


PARITY_FIXTURE = (
    Path(__file__).parents[2]
    / "crates"
    / "jope-campaigns"
    / "tests"
    / "fixtures"
    / "conditionality_gate_parity.json"
)


def candidate_row(index: int, source: str) -> dict:
    genome = f"(pipe X (cheb degree={index + 2}) (ridge))"
    return {
        "canonical_genome": genome,
        "semantic_sha256": sha256_text(genome),
        "source": source,
    }


def _reexport_durable_selfplay_groups():
    import _durable_selfplay_promotion as promotion
    import _durable_selfplay_registry as registry
    import _durable_selfplay_runtime as runtime

    globals().update(
        {
            name: value
            for module in (promotion, registry, runtime)
            for name, value in vars(module).items()
            if name.startswith("test_")
        }
    )


_reexport_durable_selfplay_groups()


__all__ = [
    "EXPECTED_METRIC",
    "PairRegistry",
    "Path",
    "SealedAuditRegistry",
    "ThreadPoolExecutor",
    "active",
    "backlog_module",
    "bundle_sources",
    "candidate",
    "default_handlers",
    "dynamic_queue_targets",
    "evaluate_gate",
    "evaluation_protocol_sha256",
    "hashlib",
    "json",
    "load_pointer",
    "loaded_bundle",
    "local_receipt_dir",
    "os",
    "promote_candidate",
    "pytest",
    "record_starting",
    "service_is_stale",
    "service_module",
    "sha256_text",
    "subprocess",
    "validate_catalog_counts",
    "validate_catalog_manifest",
    "wave_seed",
]
