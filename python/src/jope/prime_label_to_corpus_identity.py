"""Identity normalization and grammar resolution for Prime training labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .schema import TrainingRow

AGGREGATE_SCHEMA = "jope-prime-realization-aggregate-v1"
REALIZATION_SCHEMA = "jope-prime-model-realization-v2"
LEAGUE_LABEL_SCHEMA = "LeagueLabel.v1"

# Grammar engines to probe, in priority order. NOTE: this is the model grammar key,
# not the realization "engine" field (which names the Prime fork, e.g. "prime").
GRAMMAR_ENGINE_PRIORITY = ("featmath", "newmath", "prime")

# v3.2 dual dataset embedding = concat[Hyperion signature 512, foundation-expert 1024].
DUAL_EMBEDDING_DIM = 1536
TARGET_SUMMARY_DIM = 512

DEFAULT_PROVENANCE = "prime_realized"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _league_label_evidence(label: dict[str, Any]) -> dict[str, Any]:
    """Validate comparison authority and retain only trainer-safe provenance."""
    folds = label.get("folds")
    if type(folds) is not int or folds < 2:
        raise ValueError("LeagueLabel.v1 folds must be an integer of at least two")

    strings: dict[str, str] = {}
    for field in ("fold_plan", "evidence_tier", "evaluator_version"):
        value = label.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"LeagueLabel.v1 {field} must be a nonempty string")
        strings[field] = value.strip()

    runtime = label.get("runtime_seconds")
    if type(runtime) not in (int, float) or not math.isfinite(float(runtime)) or runtime < 0:
        raise ValueError("LeagueLabel.v1 runtime_seconds must be finite and nonnegative")

    raw_provenance = label.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError("LeagueLabel.v1 provenance must be an object")
    provenance: dict[str, Any] = {}
    if "gp_task_id" in raw_provenance:
        value = raw_provenance["gp_task_id"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("LeagueLabel.v1 provenance.gp_task_id must be nonempty")
        provenance["gp_task_id"] = value.strip()
    if "oof_r2" in raw_provenance:
        value = raw_provenance["oof_r2"]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("LeagueLabel.v1 provenance.oof_r2 must be finite")
        provenance["oof_r2"] = float(value)
    if "fold_r2" in raw_provenance:
        values = raw_provenance["fold_r2"]
        if not isinstance(values, list) or len(values) != folds:
            raise ValueError("LeagueLabel.v1 provenance.fold_r2 must cover every fold")
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
            raise ValueError("LeagueLabel.v1 provenance.fold_r2 must contain finite scores")
        provenance["fold_r2"] = [float(value) for value in values]
    if "robust_formula" in raw_provenance:
        value = raw_provenance["robust_formula"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("LeagueLabel.v1 provenance.robust_formula must be nonempty")
        provenance["robust_formula"] = value.strip()

    return {
        "folds": folds,
        **strings,
        "runtime_seconds": float(runtime),
        "provenance": provenance,
    }


def _reward_pair_identity(
    *,
    dataset: str,
    task: str,
    folds: int,
    fold_plan: str,
    evaluator_version: str,
) -> str:
    payload = {
        "dataset": dataset,
        "evaluator_version": evaluator_version,
        "fold_plan": fold_plan,
        "folds": folds,
        "schema": "RewardPairIdentity.v1",
        "task": task,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"reward-pair-v1:{hashlib.sha256(encoded).hexdigest()}"


# --------------------------------------------------------------------------- #
# Genome -> grammar engine resolution                                         #
# --------------------------------------------------------------------------- #
class GenomeEngineResolver:
    """Resolve the grammar engine that can tokenize a genome (cached per genome).

    Uses the Prime action vocabulary (covers featmath+newmath+prime) and the real
    ``encode_row_artifacts`` path so a resolvable genome is guaranteed encodable by
    ``train_v3`` under the returned engine.
    """

    def __init__(self, priority: tuple[str, ...] = GRAMMAR_ENGINE_PRIORITY) -> None:
        from .policy.tokens import build_prime_action_vocab

        self._vocab = build_prime_action_vocab()
        self._priority = tuple(e for e in priority if e in self._vocab.engines)
        self._cache: dict[tuple[str, str], str | None] = {}

    @property
    def vocab(self):  # exposed for callers that want the probing vocab
        return self._vocab

    def resolve(
        self,
        genome: str,
        declared_engine: str | None = None,
        task_type: str = "regression",
    ) -> str | None:
        if not genome:
            return None
        cache_key = (task_type, genome)
        if cache_key in self._cache:
            return self._cache[cache_key]
        from .policy.automaton import IllegalToken
        from .policy.model_v3 import encode_row_artifacts

        order = self._priority
        if declared_engine in self._vocab.engines:  # honor a real grammar hint first
            order = (declared_engine, *(e for e in self._priority if e != declared_engine))
        resolved: str | None = None
        for engine in order:
            probe = TrainingRow(
                dataset_id="_probe",
                source="_probe",
                embedding=[],
                genome=genome,
                canonical_genome=genome,
                engine=engine,
                metadata={"task_type": task_type},
            )
            try:
                encode_row_artifacts(probe, self._vocab)
            except (IllegalToken, ValueError):
                continue
            resolved = engine
            break
        self._cache[cache_key] = resolved
        return resolved
