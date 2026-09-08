"""Realized-label runner — the conversion linchpin (Track A3 / C1).

Turns emitted standalone artifacts for champion genomes into Prime's first
*promotable, realized-scored* corpus. It wires together primitives that already
exist but were never connected end-to-end (`prime_factory.py` stops at emit):

    champion genome
      → an exact-f64 portable artifact  (ModelArtifactSource)
      → realize_model_artifact(holdout + explicit k-fold CV)  → ModelRealizationLabel
      → append_model_realization_label(prime/runs/realizations/…)
      → plan-hash aggregate (jope-prime-realization-aggregate-v1, prime/artifacts/…)

The aggregate is keyed `(dataset_hash, engine="prime", semantic_plan_hash)` and
carries the fields `selector_calibration.load_selector_training_rows` consumes,
so it drops straight into the correlation report (C2), the realized-target
reranker + per-K Elo (C3), and the operator certification gate (C4).

This module owns the **seam-independent core** — deterministic holdout/fold
construction, aggregation, and the Prime-owned output boundary — all unit-tested
without a daemon. The artifact-emission seam (`ArtifactSource` protocol) is
pluggable: the daemon path (`PrimeNativeSearch.emit_user_python` →
`ModelArtifactSource.from_evaluator_bundle`) for compiled-plan-eligible genomes,
or an in-process/legacy `emit` for the broader champion set — wired per the
seam Codex confirms, without changing anything below.

Legacy inputs are read-only; all outputs resolve below ``prime.root``. Sealed
validation datasets are refused (inside ``realize_model_artifact``) and the
resolved dataset set must pass ``dev_pools.assert_dev_hashes``.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data.active_v3 import robust_score
from .prime_config import PrimeConfig, load_config
from .prime_realization import (
    ModelArtifactSource,
    ModelRealizationLabel,
    append_model_realization_label,
    prime_owned_path,
    realize_model_artifact,
)

AGGREGATE_SCHEMA = "jope-prime-realization-aggregate-v1"
DEFAULT_FOLDS = 8
DEFAULT_HOLDOUT_FRACTION = 0.15


def _per_dataset_seed(dataset_hash: str, split: str) -> int:
    """Deterministic per-(dataset, split) seed — stable across reruns."""
    digest = hashlib.sha256(f"{dataset_hash}|{split}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_holdout_and_folds(
    x: Any,
    y: Any,
    *,
    seed: int,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    folds: int = DEFAULT_FOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Deterministic holdout split + explicit k-fold partition over the train rows.

    Returns ``(train_rows, test_features, test_targets, fold_index_lists)`` in the
    exact shape ``realize_model_artifact`` requires: ``train_rows = [X | y]`` (y in
    the last column), the holdout as separate feature/target arrays, and ``folds``
    as validation-index lists partitioning ``0..n_train-1`` exactly once.
    """
    features = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    targets = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("x must be 2-D with one target per row")
    n = features.shape[0]
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")

    order = np.random.Generator(np.random.PCG64(int(seed) & ((1 << 64) - 1))).permutation(n)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    n_holdout = min(n_holdout, n - 2)  # keep at least 2 training rows
    if n_holdout < 1:
        raise ValueError("dataset too small to carve a holdout with >= 2 training rows")
    holdout_idx = order[:n_holdout]
    train_idx = order[n_holdout:]
    n_train = train_idx.shape[0]

    train_x = features[train_idx]
    train_y = targets[train_idx]
    train_rows = np.column_stack((train_x, train_y))
    test_features = features[holdout_idx]
    test_targets = targets[holdout_idx]

    k = max(2, min(int(folds), n_train))
    # Contiguous chunks of a fresh permutation → a clean partition of 0..n_train-1.
    fold_perm = np.random.Generator(
        np.random.PCG64((int(seed) ^ 0x5DEECE66D) & ((1 << 64) - 1))
    ).permutation(n_train)
    fold_index_lists = [sorted(int(i) for i in chunk) for chunk in np.array_split(fold_perm, k)]
    fold_index_lists = [chunk for chunk in fold_index_lists if chunk]
    return train_rows, test_features, test_targets, fold_index_lists


def per_dataset_seed(dataset_hash: str, split: str) -> int:
    """Public alias for the deterministic per-(dataset, split) seed."""
    return _per_dataset_seed(dataset_hash, split)


def emitted_reference_predictions(
    source: str,
    train_rows: Any,
    holdout_x: Any,
    *,
    artifact_kind: str = "portable_python",
    config: PrimeConfig | None = None,
    timeout_s: float = 30.0,
) -> np.ndarray:
    """Independently execute an emitted standalone artifact to capture golden holdout
    predictions.

    The execution is byte-format identical to ``realize_model_artifact`` (same
    ``%.17g`` CSV encoding, same locked-down environment, same
    ``train.csv test.csv predictions.csv`` argument contract). For a deterministic
    portable program the returned vector is therefore the exact-f64 reference: passing
    it back as ``exact_predictions`` yields
    ``exact_prediction_maximum_absolute_error == 0.0``, certifying that the artifact
    reproduces its own predictions bit-for-bit across independent process executions.
    """
    if artifact_kind != "portable_python":
        raise ValueError("reference execution currently supports portable_python only")
    config = config or load_config()
    train = np.ascontiguousarray(np.asarray(train_rows, dtype=np.float64))
    features = np.ascontiguousarray(np.asarray(holdout_x, dtype=np.float64))
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("holdout_x must be a non-empty two-dimensional matrix")
    cache = prime_owned_path(config.cache / "reference_runs", config)
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ref-", dir=cache))
    source_path = work / "model.py"
    train_path = work / "train.csv"
    test_path = work / "test_features.csv"
    predictions_path = work / "predictions.csv"
    source_path.write_text(source, encoding="utf-8")
    np.savetxt(train_path, train, delimiter=",", fmt="%.17g")
    np.savetxt(test_path, features, delimiter=",", fmt="%.17g")
    environment = dict(os.environ)
    environment.update({"TMPDIR": str(work), "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""})
    result = subprocess.run(
        [sys.executable, str(source_path), str(train_path), str(test_path), str(predictions_path)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(work),
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(f"reference execution failed: {result.stderr.strip()[:400]}")
    if not predictions_path.is_file():
        raise RuntimeError("reference execution did not create predictions.csv")
    predictions = np.asarray(
        np.loadtxt(predictions_path, delimiter=",", dtype=np.float64), dtype=np.float64
    ).reshape(-1)
    if predictions.shape != (features.shape[0],) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("reference execution produced an invalid prediction vector")
    return np.ascontiguousarray(predictions)


def realize_champion(
    artifact: ModelArtifactSource,
    x: Any,
    y: Any,
    *,
    dataset_id: str,
    dataset_hash: str,
    split: str,
    config: PrimeConfig | None = None,
    output_path: str | Path | None = None,
    exact_predictions: Any | None = None,
    exact_reference: bool = False,
    folds: int = DEFAULT_FOLDS,
    timeout_s: float = 30.0,
) -> ModelRealizationLabel:
    """Realize one champion artifact on one dataset/split, appending the label.

    When ``exact_reference`` is set and no ``exact_predictions`` are supplied, an
    independent golden reference execution of the artifact is captured on this split's
    holdout and threaded through as the exact-f64 reference — yielding a fidelity check
    (``exact_prediction_maximum_absolute_error``) alongside the realized quality.
    """
    config = config or load_config()
    seed = _per_dataset_seed(dataset_hash, split)
    train_rows, test_features, test_targets, fold_lists = build_holdout_and_folds(
        x, y, seed=seed, folds=folds
    )
    if exact_predictions is None and exact_reference:
        exact_predictions = emitted_reference_predictions(
            artifact.source,
            train_rows,
            test_features,
            artifact_kind=artifact.artifact_kind,
            config=config,
            timeout_s=timeout_s,
        )
    return realize_model_artifact(
        artifact,
        train_rows,
        test_features,
        test_targets,
        dataset_id=dataset_id,
        split=split,
        folds=fold_lists,
        dataset_hash=dataset_hash,
        exact_predictions=exact_predictions,
        output_path=output_path,
        config=config,
        timeout_s=timeout_s,
    )


def _label_field(label: Any, name: str) -> Any:
    if isinstance(label, ModelRealizationLabel):
        return getattr(label, name)
    if isinstance(label, dict):
        return label.get(name)
    return None


def aggregate_realization_labels(labels: Sequence[Any]) -> list[dict[str, Any]]:
    """Group realized labels by ``(dataset_hash, semantic_plan_hash)`` → aggregate rows.

    Emits one selector-consumable row per plan: ``reward`` is
    ``robust_score(mean_realized_oof, per-fold r2)``, ``p_bad`` is the failure
    probability across evidence, and ``canonical_genome`` is preserved (required
    non-empty by ``selector_calibration``).
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    for label in labels:
        dataset_hash = str(_label_field(label, "dataset_hash") or "")
        semantic_plan_hash = str(_label_field(label, "semantic_plan_hash") or "")
        if not dataset_hash or not semantic_plan_hash:
            continue
        groups.setdefault((dataset_hash, semantic_plan_hash), []).append(label)

    rows: list[dict[str, Any]] = []
    for (dataset_hash, semantic_plan_hash), members in sorted(groups.items()):
        n_evidence = len(members)
        failures = [m for m in members if _label_field(m, "failure") is not None]
        successes = [m for m in members if _label_field(m, "failure") is None]
        realized = [
            float(_label_field(m, "realized_score"))
            for m in successes
            if isinstance(_label_field(m, "realized_score"), (int, float))
        ]
        oof = [
            float(_label_field(m, "oof_r2"))
            for m in successes
            if isinstance(_label_field(m, "oof_r2"), (int, float))
        ]
        mean_realized = statistics.fmean(realized) if realized else None
        realized_std = statistics.pstdev(realized) if len(realized) > 1 else 0.0
        reward = robust_score(mean_realized, oof if oof else None)
        canonical = ""
        for m in members:
            cg = _label_field(m, "canonical_genome")
            if isinstance(cg, str) and cg:
                canonical = cg
                break
        # promotion_eligible aggregates conjunctively: every piece of evidence must qualify.
        promo_flags = [bool(_label_field(m, "promotion_eligible")) for m in members]
        rows.append(
            {
                "schema": AGGREGATE_SCHEMA,
                "dataset_hash": dataset_hash,
                "engine": "prime",
                "semantic_plan_hash": semantic_plan_hash,
                "canonical_genome": canonical,
                "reward": reward,
                "reward_std": realized_std,
                "mean_realized": mean_realized,
                "p_bad": (len(failures) / n_evidence) if n_evidence else None,
                "n_evidence": n_evidence,
                "n_failed": len(failures),
                "promotion_eligible": bool(promo_flags) and all(promo_flags),
            }
        )
    return rows


def append_realization_aggregate(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    *,
    config: PrimeConfig | None = None,
) -> Path:
    """Append aggregate rows to a Prime-owned JSONL file (path boundary enforced)."""
    config = config or load_config()
    destination = prime_owned_path(path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    return destination


__all__ = [
    "AGGREGATE_SCHEMA",
    "build_holdout_and_folds",
    "per_dataset_seed",
    "emitted_reference_predictions",
    "realize_champion",
    "aggregate_realization_labels",
    "append_realization_aggregate",
]
