"""Adapter: realized-label / aggregate rows -> policy_v3 ``TrainingRow`` corpus.

This is the data seam of the self-improvement loop's brain. The loop realizes
champions on held-out data and writes labels; this module turns those labels into
the exact corpus that ``jope.policy.train_v3.train_policy_v3_from_jsonl`` consumes,
so the REAL transformer proposer can train on the loop's realized outcomes.

Supported input schemas (dispatched on the ``schema`` field, with fallbacks):

* ``jope-prime-realization-aggregate-v1`` (the primary target) — one row per
  (dataset_hash, semantic_plan_hash) group, carrying ``canonical_genome``,
  ``reward`` (== ``mean_realized``), ``p_bad``, ``reward_std``, ``n_evidence``.
* ``jope-prime-model-realization-v2`` — a single realized champion; ``realized_score``
  is the reward, plus ``oof_r2`` and ``failure``.
* ``LeagueLabel.v1`` (see ``rust/jope-league/src/contracts.rs``) — the league's
  wire label; ``robust_score`` is the reward, ``failure`` marks a negative.

The two load-bearing mappings:

* ``canonical_genome`` -> genome tokens. IMPORTANT: the realization rows are tagged
  ``engine="prime"`` meaning the *Prime fork*, NOT the policy grammar engine. Their
  genomes are ordinary ``featmath`` (and occasionally ``newmath``) pipelines. So we
  resolve the true *grammar* engine by probing which grammar can encode the genome,
  because ``train_v3`` tokenizes with ``row.engine`` as a grammar key.
* ``reward``/``mean_realized`` -> the ranking target. ``train_v3`` recomputes the
  within-dataset listwise labels (``oof_delta_to_incumbent``, ``rank_pct``) from
  ``_score_label`` = ``metadata.score_robust`` | ``metadata.oof_r2_raw`` | ``row.score``,
  grouped by ``metadata.reward_pair_identity`` when present, otherwise the base dataset.
  League labels derive that identity from dataset/task/fold plan/evaluator authority;
  ``dataset_id`` remains ``dataset_hash`` for conditioning, splits, and routing.

Dataset conditioning (the model's ``embedding`` / ``target_embedding``): produced the
same way ``self_play``/``gp_factory`` build it — ``propose.embedding_from_vs`` /
``target_summary_from_vs`` over the per-dataset ``.vs`` signature files named in a
split manifest. When those signatures are unavailable the adapter falls back to a
deterministic per-dataset placeholder of the requested dimensionality and flags the
row (``metadata.embedding_synthetic``); a production checkpoint requires the real
signatures (documented on ``PlaceholderEmbedder``).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import TrainingRow
from .split_integrity import (
    DEVELOPMENT_SPLIT,
    TEMPORAL_AUDIT_SPLIT,
    assignment_for_dataset,
    is_training_authority_label,
    validate_split_manifest,
)

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
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in values
        ):
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


# --------------------------------------------------------------------------- #
# Dataset conditioning embedders                                              #
# --------------------------------------------------------------------------- #
class SignatureEmbedder:
    """Real dataset conditioning from ``.vs`` signature files (self_play's path).

    ``manifest`` maps ``dataset_hash -> {"vs_path", "train_target_vs", ...}`` (the
    ``split_manifest.json`` produced by the corpus builder). ``standardization`` is the
    matching ``standardization.json`` (carries the signature dim + foundation projector
    spec so ``embedding_from_vs`` yields the 1536-d dual embedding).
    """

    def __init__(
        self,
        manifest: str | Path,
        standardization: str | Path | None = None,
        *,
        signature_root: str | Path | None = None,
        require_hashes: bool = False,
    ) -> None:
        import json

        self._manifest_path = Path(manifest).resolve()
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._datasets: dict[str, dict] = data.get("datasets", data)
        self._signature_root = Path(signature_root).resolve() if signature_root else None
        self._require_hashes = require_hashes
        if standardization is not None:
            from .encoders.projection import resolve_foundation_paths

            standardization_path = Path(standardization).resolve()
            self._std = resolve_foundation_paths(
                json.loads(standardization_path.read_text(encoding="utf-8")),
                standardization_path.parent,
            )
        else:
            self._std = None
        self._cache: dict[str, tuple[list[float] | None, list[float] | None]] = {}

    def _path(self, dataset_hash: str, entry: dict, field: str) -> Path | None:
        raw = entry.get(field)
        if not raw:
            return None
        source = Path(str(raw))
        if self._signature_root is not None:
            candidate = (self._signature_root / dataset_hash / source.name).resolve()
            try:
                candidate.relative_to(self._signature_root)
            except ValueError as error:
                raise ValueError(f"staged signature path escapes root: {candidate}") from error
        elif source.is_absolute():
            candidate = source
        else:
            candidate = (self._manifest_path.parent / source).resolve()
        if not candidate.is_file():
            raise ValueError(f"staged signature is missing: {candidate}")
        expected = entry.get(f"{field}_sha256")
        if field == "vs_path":
            expected = expected or entry.get("vs_sha256")
        elif field == "train_target_vs":
            expected = expected or entry.get("train_target_sha256")
        if self._require_hashes and not expected:
            raise ValueError(f"{field} for dataset {dataset_hash} has no staged SHA-256")
        if expected:
            digest = _sha256_file(candidate)
            if digest != expected:
                raise ValueError(
                    f"staged signature hash mismatch for {dataset_hash}/{candidate.name}"
                )
        return candidate

    def covers(self, dataset_hash: str) -> bool:
        return dataset_hash in self._datasets

    def __call__(self, dataset_hash: str) -> tuple[list[float] | None, list[float] | None]:
        if dataset_hash in self._cache:
            return self._cache[dataset_hash]
        entry = self._datasets.get(dataset_hash)
        if entry is None or not entry.get("vs_path"):
            out: tuple[list[float] | None, list[float] | None] = (None, None)
        else:
            from .propose import embedding_from_vs, target_summary_from_vs

            vs_path = self._path(dataset_hash, entry, "vs_path")
            target_path = self._path(dataset_hash, entry, "train_target_vs")
            emb = embedding_from_vs(vs_path, self._std)
            tgt = target_summary_from_vs(target_path)
            out = (emb, tgt)
        self._cache[dataset_hash] = out
        return out


class PlaceholderEmbedder:
    """Deterministic per-dataset placeholder embedding (no real signatures available).

    REQUIREMENT for a production checkpoint: replace with ``SignatureEmbedder`` over the
    real per-dataset ``.vs`` Hyperion/foundation signatures. The placeholder is stable
    per ``dataset_hash`` (so the per-dataset listwise grouping and conditioning stay
    self-consistent) but carries no real dataset information — it proves the data path,
    not model quality.
    """

    def __init__(self, dim: int = DUAL_EMBEDDING_DIM, target_dim: int = TARGET_SUMMARY_DIM) -> None:
        self.dim = dim
        self.target_dim = target_dim

    def __call__(self, dataset_hash: str) -> tuple[list[float], list[float]]:
        seed = int(hashlib.sha256(dataset_hash.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        emb = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        tgt = [rng.gauss(0.0, 1.0) for _ in range(self.target_dim)]
        return emb, tgt


# --------------------------------------------------------------------------- #
# Row parsing (schema -> canonical fields)                                    #
# --------------------------------------------------------------------------- #
def _parse_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize any supported label schema into canonical adapter fields."""
    committed = row
    if isinstance(row.get("label"), dict):
        row = row["label"]
    schema = str(row.get("schema") or "")
    if schema == LEAGUE_LABEL_SCHEMA:
        if str(row.get("split") or "train").lower() != "train":
            return None
        genome = row.get("canonical_genome") or row.get("semantic_genome")
        dataset = row.get("dataset")
        if not genome or not dataset:
            return None
        evidence = _league_label_evidence(row)
        safe_provenance = evidence["provenance"]
        failure = row.get("failure")
        task_type = str(row.get("task") or "regression")
        return {
            "dataset_hash": str(dataset),
            "canonical_genome": str(genome),
            "engine": row.get("engine"),
            "reward": _finite(row.get("robust_score")),
            "oof_r2": _finite(safe_provenance.get("oof_r2")),
            "fold_r2": safe_provenance.get("fold_r2"),
            "folds": evidence["folds"],
            "fold_plan": evidence["fold_plan"],
            "evaluator_version": evidence["evaluator_version"],
            "evidence_tier": evidence["evidence_tier"],
            "runtime_seconds": evidence["runtime_seconds"],
            "gp_task_id": safe_provenance.get("gp_task_id"),
            "robust_formula": safe_provenance.get("robust_formula"),
            "reward_pair_identity": _reward_pair_identity(
                dataset=str(dataset),
                task=task_type,
                folds=evidence["folds"],
                fold_plan=evidence["fold_plan"],
                evaluator_version=evidence["evaluator_version"],
            ),
            "p_bad": 1.0 if failure else 0.0,
            "failed": bool(failure),
            "task_type": task_type,
            "semantic_plan_hash": row.get("semantic_genome"),
            "reward_std": None,
            "n_evidence": None,
            "promotion_eligible": row.get("failure") is None,
            "label_watermark": committed.get("watermark") or committed.get("_watermark"),
            "committed_at_ms": committed.get("committed_at_ms") or committed.get("_committed_at_ms"),
            "replay_hot_until_ms": committed.get("_replay_hot_until_ms"),
        }
    # aggregate-v1 and realization-v2 share dataset_hash + canonical_genome
    genome = row.get("canonical_genome")
    dataset = row.get("dataset_hash") or row.get("dataset_id")
    if not genome or not dataset:
        return None
    reward = _finite(row.get("reward"))
    if reward is None:
        reward = _finite(row.get("mean_realized"))
    if reward is None:
        reward = _finite(row.get("realized_score"))
    oof = _finite(row.get("oof_r2"))
    failed = bool(row.get("failure")) or (row.get("n_failed") or 0) > 0 and reward is None
    return {
        "dataset_hash": str(dataset),
        "canonical_genome": str(genome),
        "engine": row.get("engine"),
        "reward": reward,
        "oof_r2": oof if oof is not None else reward,
        "p_bad": _finite(row.get("p_bad")),
        "failed": bool(failed),
        "task_type": str(row.get("task") or "regression"),
        "semantic_plan_hash": row.get("semantic_plan_hash"),
        "reward_std": _finite(row.get("reward_std")),
        "n_evidence": row.get("n_evidence"),
        "fold_r2": None,
        "folds": None,
        "fold_plan": None,
        "evaluator_version": None,
        "evidence_tier": None,
        "runtime_seconds": None,
        "gp_task_id": None,
        "robust_formula": None,
        "reward_pair_identity": None,
        "promotion_eligible": bool(row.get("promotion_eligible", True)),
        "label_watermark": committed.get("watermark") or committed.get("_watermark"),
        "committed_at_ms": committed.get("committed_at_ms") or committed.get("_committed_at_ms"),
        "replay_hot_until_ms": committed.get("_replay_hot_until_ms"),
    }


def label_to_training_row(
    row: dict[str, Any],
    *,
    embedder: Any,
    resolver: GenomeEngineResolver,
    split: str = "train",
    provenance: str = DEFAULT_PROVENANCE,
) -> tuple[TrainingRow | None, str]:
    """Convert one label row into a ``TrainingRow``. Returns (row_or_None, reason)."""
    if not is_training_authority_label(row):
        return None, "report_only_evidence"
    parsed = _parse_row(row)
    if parsed is None:
        return None, "missing_identity"
    genome = parsed["canonical_genome"]
    engine = resolver.resolve(genome, parsed.get("engine"))
    if engine is None:
        return None, "unencodable_genome"

    reward = parsed["reward"]
    if reward is None and not parsed["failed"]:
        return None, "no_finite_reward"

    emb, tgt = embedder(parsed["dataset_hash"])
    synthetic = emb is None
    if synthetic:  # signature embedder missed this dataset -> deterministic placeholder
        emb, tgt = PlaceholderEmbedder(DUAL_EMBEDDING_DIM, TARGET_SUMMARY_DIM)(parsed["dataset_hash"])

    metadata: dict[str, Any] = {
        "provenance": provenance,
        "task_type": parsed["task_type"],
        "split": split,
        "dataset_hash": parsed["dataset_hash"],
        "semantic_plan_hash": parsed.get("semantic_plan_hash"),
        "p_bad": parsed.get("p_bad"),
        "reward_std": parsed.get("reward_std"),
        "n_evidence": parsed.get("n_evidence"),
        "promotion_eligible": parsed.get("promotion_eligible"),
        "realization_engine": parsed.get("engine"),
        "embedding_synthetic": synthetic,
        "label_watermark": parsed.get("label_watermark"),
        "committed_at_ms": parsed.get("committed_at_ms"),
        "replay_hot_until_ms": parsed.get("replay_hot_until_ms"),
        "fold_r2": parsed.get("fold_r2"),
        "folds": parsed.get("folds"),
        "fold_plan": parsed.get("fold_plan"),
        "evaluator_version": parsed.get("evaluator_version"),
        "evidence_tier": parsed.get("evidence_tier"),
        "gp_task_id": parsed.get("gp_task_id"),
        "robust_formula": parsed.get("robust_formula"),
        "reward_pair_identity": parsed.get("reward_pair_identity"),
    }
    if reward is not None:
        # feed all three keys train_v3._score_label reads, so the within-dataset
        # listwise/reward selector labels are recomputed from the realized reward.
        metadata["score_robust"] = reward
        metadata["oof_r2_raw"] = parsed["oof_r2"] if parsed["oof_r2"] is not None else reward
    if parsed["failed"]:
        # a realized failure is a hard negative for the p_bad / stability head.
        metadata["eval_error"] = True

    tr = TrainingRow(
        dataset_id=parsed["dataset_hash"],
        source=provenance,
        embedding=emb,
        target_embedding=tgt,
        genome=genome,
        canonical_genome=genome,
        engine=engine,
        valid=not parsed["failed"],
        score=reward,
        cost_ms=(
            parsed["runtime_seconds"] * 1_000.0
            if parsed.get("runtime_seconds") is not None
            else None
        ),
        metadata=metadata,
    )
    return tr, "ok"


def build_corpus(
    rows: Iterable[dict[str, Any]],
    *,
    embedder: Any | None = None,
    resolver: GenomeEngineResolver | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    provenance: str = DEFAULT_PROVENANCE,
    validation_datasets: Iterable[str] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> tuple[list[TrainingRow], dict[str, Any]]:
    """Adapt an iterable of label rows into a policy_v3 ``TrainingRow`` corpus.

    Whole datasets are held out for the validation split (``train_v3`` requires one and
    the listwise loss groups by dataset). Returns ``(training_rows, stats)``.
    """
    rows = list(rows)
    resolver = resolver or GenomeEngineResolver()
    embedder = embedder or PlaceholderEmbedder()

    # Assign validation by DATASET so a dataset never straddles train/val.  A
    # continuous trainer passes the initially selected identities back on every
    # epoch: newly committed TRAIN labels can then enter replay without moving
    # the fixed internal validation boundary.
    def dataset_id(row: dict[str, Any]) -> str:
        label = row.get("label") if isinstance(row.get("label"), dict) else row
        return str(
            label.get("dataset_hash")
            or label.get("dataset_id")
            or label.get("dataset")
            or ""
        )

    if split_manifest is not None:
        validate_split_manifest(split_manifest)
        if split_manifest["status"] != "complete":
            raise ValueError("cannot build a corpus from an incomplete split manifest")
    hashes = sorted(
        {dataset_id(r) for r in rows if is_training_authority_label(r)} - {""}
    )
    if split_manifest is not None:
        val_datasets = {
            dataset
            for dataset, record in split_manifest["assignments"].items()
            if record["split"] == DEVELOPMENT_SPLIT
        }
    elif validation_datasets is None:
        shuffled = hashes[:]
        random.Random(seed).shuffle(shuffled)
        n_val = (
            max(1, round(val_fraction * len(shuffled)))
            if (val_fraction > 0 and len(shuffled) > 1)
            else 0
        )
        val_datasets = set(shuffled[:n_val])
    else:
        val_datasets = {str(value) for value in validation_datasets}

    out: list[TrainingRow] = []
    reasons: dict[str, int] = {}
    engines: dict[str, int] = {}
    synthetic = 0
    audit_datasets: set[str] = set()
    for r in rows:
        dh = dataset_id(r)
        if not is_training_authority_label(r):
            reasons["report_only_evidence"] = reasons.get("report_only_evidence", 0) + 1
            continue
        assignment = (
            assignment_for_dataset(split_manifest, dh)
            if split_manifest is not None
            else None
        )
        if assignment is not None and assignment["split"] == TEMPORAL_AUDIT_SPLIT:
            audit_datasets.add(dh)
            reasons["report_only_temporal_audit"] = (
                reasons.get("report_only_temporal_audit", 0) + 1
            )
            continue
        split = (
            assignment["split"]
            if assignment is not None
            else "validation"
            if dh in val_datasets
            else "train"
        )
        tr, reason = label_to_training_row(
            r, embedder=embedder, resolver=resolver, split=split, provenance=provenance
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if tr is None:
            continue
        engines[tr.engine] = engines.get(tr.engine, 0) + 1
        if (tr.metadata or {}).get("embedding_synthetic"):
            synthetic += 1
        if assignment is not None:
            tr.metadata.update(
                {
                    "source_family": assignment["source_family"],
                    "split_assignment": assignment["split"],
                    "split_manifest_schema": split_manifest["schema"],
                    "split_manifest_sha256": split_manifest["manifest_sha256"],
                }
            )
        out.append(tr)

    train_datasets = {str(r.dataset_id) for r in out if (r.metadata or {}).get("split") == "train"}
    val_seen = {
        str(r.dataset_id)
        for r in out
        if (r.metadata or {}).get("split") in {"validation", DEVELOPMENT_SPLIT}
    }
    stats = {
        "input_rows": len(rows),
        "training_rows": len(out),
        "skipped": {k: v for k, v in reasons.items() if k != "ok"},
        "engines": engines,
        "datasets": len(train_datasets | val_seen),
        "train_datasets": len(train_datasets),
        "validation_datasets": len(val_seen),
        "validation_dataset_ids": sorted(val_datasets),
        "temporal_audit_datasets": len(audit_datasets),
        "temporal_audit_dataset_ids": sorted(audit_datasets),
        "split_manifest_sha256": (
            split_manifest["manifest_sha256"] if split_manifest is not None else None
        ),
        "embedding_dim": len(out[0].embedding) if out else 0,
        "target_summary_dim": len(out[0].target_embedding or []) if out else 0,
        "synthetic_embedding_rows": synthetic,
    }
    return out, stats


def write_corpus(rows: list[TrainingRow], out_path: str | Path) -> Path:
    """Write ``TrainingRow`` corpus as JSONL (the format ``train_v3`` reads)."""
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json()) + "\n")
    return out_path
