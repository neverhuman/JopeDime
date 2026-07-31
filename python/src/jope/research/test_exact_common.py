"""Shared invariants for the internal, test-adaptive self-play campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Per-task authority metric. Keep in lockstep with the Rust side
# (`jope_selfplay_promotion.rs::expected_catalog_metric` /
# `task::TaskKind::primary_metric`): "negative_mase" is the existing wire
# value corpus_eval pins for timeseries receipts (R1) — never "neg_mase".
EXPECTED_METRIC = {
    "regression": "r2",
    "binary": "auc",
    "timeseries": "negative_mase",
    "multiclass": "macro_f1",
}
# Tasks that predate the four-task widening. Their catalog declarations stay
# mandatory; catalogs written before the new tasks existed simply omit the
# new tasks, which is a declared count of zero.
LEGACY_TASKS = ("regression", "binary")
SELFPLAY_SELECTION_BIAS = "test-exact-reused"
SLOT_COUNTS = {"local_elite": 32, "dl_proposal": 64, "common_winner": 96, "offspring": 64}
CATALOG_STRIDE_BYTES = 2_048 * 4
EVALUATOR_PROTOCOL_VERSION = "test-exact-v1"
ALLOWED_LIVE_STATES = {
    "queued",
    "starting",
    "dispatching",
    "running",
    "preempting",
    "held",
}
USEFUL_QUEUE_STATES = {"queued", "starting"}
NON_INTERRUPTIBLE_STATES = {"dispatching", "running", "preempting"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OP = re.compile(r"\(([a-zA-Z0-9_+-]+)(?:\s|\))")
_STRICT_WAVE = re.compile(r"^strict-wave-\d{6}$")


def prime_source_root(source: Path | None = None) -> Path:
    """Resolve the Prime source pinned beside an immutable runtime bundle."""

    origin = (source or Path(__file__)).resolve()
    for parent in origin.parents:
        candidate = parent / "jope-prime" / "src"
        if candidate.is_dir():
            return candidate
    return Path("/home/ubuntu/jope-prime/src")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    )
    os.replace(temporary, path)


def receipt_authority_kind(row: Mapping[str, object]) -> str | None:
    """Classify internally usable receipt metadata without rewriting history.

    ``legacy-superseded`` exists only so immutable, already-ingested receipts
    remain reproducible. New manifests must use the corrected authority.
    """

    task = row.get("task")
    common = (
        task in EXPECTED_METRIC
        and row.get("tier", row.get("score_tier")) == "test_exact"
        and row.get("metric") == EXPECTED_METRIC[str(task)]
        and row.get("valid") is True
        and isinstance(row.get("score"), (int, float))
        and math.isfinite(float(row["score"]))
        and all(
            isinstance(row.get(field), str) and bool(_SHA256.fullmatch(str(row[field])))
            for field in (
                "frozen_seed_sha256",
                "model_checkpoint_sha256",
                "hall_of_fame_sha256",
            )
        )
    )
    if not common:
        return None
    if (
        row.get("test_adaptive") is True
        and row.get("benchmark_valid") is False
        and row.get("selection_bias") == SELFPLAY_SELECTION_BIAS
        and isinstance(row.get("generation"), str)
        and bool(_STRICT_WAVE.fullmatch(str(row["generation"])))
    ):
        return "corrected"
    if (
        row.get("test_adaptive") is False
        and row.get("benchmark_valid") is True
        and row.get("generation") == "zero"
        and row.get("selection_bias") in (None, "")
    ):
        return "legacy-superseded"
    return None


def manifest_authority_kind(
    authority: Mapping[str, object], *, task: str
) -> str | None:
    if task not in EXPECTED_METRIC:
        return None
    common = (
        authority.get("score_tier") == "test_exact"
        and authority.get("metric") == EXPECTED_METRIC[task]
        and all(
            isinstance(authority.get(field), str)
            and bool(_SHA256.fullmatch(str(authority[field])))
            for field in (
                "frozen_seed_sha256",
                "model_checkpoint_sha256",
                "hall_of_fame_sha256",
            )
        )
    )
    if not common:
        return None
    if (
        authority.get("test_adaptive") is True
        and authority.get("benchmark_valid") is False
        and authority.get("selection_bias") == SELFPLAY_SELECTION_BIAS
        and isinstance(authority.get("generation"), str)
        and bool(_STRICT_WAVE.fullmatch(str(authority["generation"])))
    ):
        return "corrected"
    if (
        authority.get("test_adaptive") is False
        and authority.get("benchmark_valid") is True
        and authority.get("selection_bias") in (None, "")
        and authority.get("generation") == "zero"
    ):
        return "legacy-superseded"
    return None


def is_strict_receipt(
    row: Mapping[str, object], *, allow_legacy: bool = False
) -> bool:
    kind = receipt_authority_kind(row)
    return kind == "corrected" or (allow_legacy and kind == "legacy-superseded")


def selfplay_authority(
    *,
    task: str,
    wave: str,
    frozen_seed_sha256: str,
    model_checkpoint_sha256: str,
    hall_of_fame_sha256: str,
) -> dict[str, object]:
    if task not in EXPECTED_METRIC:
        raise ValueError(f"unsupported task {task!r}")
    if not _STRICT_WAVE.fullmatch(wave):
        raise ValueError("self-play generation must be an actual strict-wave identifier")
    hashes = {
        "frozen_seed_sha256": frozen_seed_sha256,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "hall_of_fame_sha256": hall_of_fame_sha256,
    }
    if any(not _SHA256.fullmatch(value) for value in hashes.values()):
        raise ValueError("self-play authority hashes must be lowercase SHA-256 values")
    return {
        "score_tier": "test_exact",
        "metric": EXPECTED_METRIC[task],
        "test_adaptive": True,
        "benchmark_valid": False,
        "selection_bias": SELFPLAY_SELECTION_BIAS,
        "generation": wave,
        **hashes,
    }


def _declared_catalog_counts(manifest: Mapping[str, object]) -> dict[str, int]:
    declared = manifest.get("task_counts") or manifest.get("declared_task_counts")
    if isinstance(declared, Mapping):
        # The legacy tasks must always be declared (-1 fails closed in
        # `validate_catalog_counts`); missing new tasks read as zero.
        return {
            task: int(declared.get(task, -1 if task in LEGACY_TASKS else 0))
            for task in EXPECTED_METRIC
        }
    return {
        "regression": int(manifest.get("regression_count", -1)),
        "binary": int(manifest.get("binary_count", -1)),
    }


def validate_catalog_counts(
    entries: Iterable[Mapping[str, object]],
    declared_counts: Mapping[str, object],
) -> dict[str, int]:
    """Validate identities against counts declared by an immutable manifest.

    Counts deliberately have no code-side default. The catalog is the authority,
    so changing a catalog cannot be accidentally blocked by a stale literal.
    Missing NEW-task declarations mean zero datasets for that task; the legacy
    regression/binary declarations stay mandatory, and unknown tasks are still
    rejected. Keep in lockstep with
    `jope_selfplay_promotion.rs::validate_catalog_counts`.
    """

    declared = {str(task): int(count) for task, count in declared_counts.items()}
    if any(value < 0 for value in declared.values()):
        raise ValueError("catalog manifest does not declare every per-task count")
    if not set(declared) <= set(EXPECTED_METRIC) or not set(LEGACY_TASKS) <= set(
        declared
    ):
        raise ValueError("catalog manifest declares unsupported task counts")
    expected = {task: 0 for task in EXPECTED_METRIC}
    expected.update(declared)
    counts = {task: 0 for task in EXPECTED_METRIC}
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        task = str(entry.get("task") or "")
        dataset = str(entry.get("dataset_hash") or "")
        if task not in counts or not dataset:
            raise ValueError("catalog contains an unsupported or anonymous dataset")
        identity = (task, dataset)
        if identity in seen:
            raise ValueError(f"catalog repeats {task}:{dataset}")
        seen.add(identity)
        counts[task] += 1
    if counts != expected:
        raise ValueError(
            f"catalog count mismatch: expected {expected}, found {counts}"
        )
    return counts


def validate_catalog_manifest(
    pointer: Mapping[str, object] | str | Path,
    *,
    verify_contexts: bool = True,
) -> dict[str, object]:
    """Verify the immutable catalog pointer, manifest, contexts, and split topology."""

    pointer_path: Path | None = None
    if isinstance(pointer, (str, Path)):
        pointer_path = Path(pointer)
        value = json.loads(pointer_path.read_text())
    else:
        value = dict(pointer)
    for field in ("catalog", "catalog_sha256"):
        if not value.get(field):
            raise ValueError(f"catalog pointer is missing {field}")
    catalog_path = Path(str(value["catalog"]))
    if sha256_file(catalog_path) != value["catalog_sha256"]:
        raise ValueError("catalog pointer hash mismatch")
    manifest = json.loads(catalog_path.read_text())
    if manifest.get("schema") == "JopeExpandedCatalog.v2":
        for field in ("data", "data_sha256"):
            if not value.get(field):
                raise ValueError(f"expanded catalog pointer is missing {field}")
        data_path = Path(str(value["data"]))
        if (
            not data_path.is_file()
            or sha256_file(data_path) != value["data_sha256"]
            or manifest.get("data_sha256") != value["data_sha256"]
        ):
            raise ValueError("expanded catalog context pointer hash mismatch")
        if manifest.get("immutable") is not True or manifest.get("append_only") is not True:
            raise ValueError("expanded catalog is not immutable and append-only")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("expanded catalog entries are absent")
        counts = validate_catalog_counts(entries, _declared_catalog_counts(manifest))
        if int(manifest.get("row_count", -1)) != len(entries):
            raise ValueError("expanded catalog declared row count mismatch")
        stride = int(manifest.get("record_stride_bytes", -1))
        if stride != CATALOG_STRIDE_BYTES:
            raise ValueError("expanded catalog context stride mismatch")
        if data_path.stat().st_size != len(entries) * stride:
            raise ValueError("expanded catalog context blob row count mismatch")
        indexes = [int(entry.get("catalog_index", -1)) for entry in entries]
        if sorted(indexes) != list(range(len(entries))):
            raise ValueError("expanded catalog indexes are not unique and contiguous")
        split_by_family: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            task = str(entry["task"])
            if entry.get("metric") != EXPECTED_METRIC[task]:
                raise ValueError(f"expanded catalog metric mismatch for {task}")
            family = str(entry.get("source_family") or "")
            split = str(entry.get("split") or "")
            if not family or split not in {"train", "validation"}:
                raise ValueError("expanded catalog family/split identity is invalid")
            split_by_family[family].add(split)
            for field in (
                "train_sha256",
                "test_sha256",
                "cache_manifest_sha256",
                "context_sha256",
            ):
                if not _SHA256.fullmatch(str(entry.get(field) or "")):
                    raise ValueError(f"expanded catalog entry has invalid {field}")
            encodings = entry.get("encoding_hashes")
            if not isinstance(encodings, Mapping) or not encodings:
                raise ValueError("expanded catalog encoding hashes are absent")
            if any(not _SHA256.fullmatch(str(digest)) for digest in encodings.values()):
                raise ValueError("expanded catalog encoding hash is invalid")
            if not _SHA256.fullmatch(
                str(entry.get("encoder_protocol_sha256") or "")
            ):
                raise ValueError("expanded catalog encoder protocol hash is invalid")
            if verify_contexts:
                cache = Path(str(entry.get("cache") or ""))
                cache_manifest = cache / "cache-manifest.v1.json"
                if (
                    not cache_manifest.is_file()
                    or sha256_file(cache_manifest) != entry["cache_manifest_sha256"]
                ):
                    raise ValueError("expanded catalog cache manifest hash mismatch")
                for name, digest in encodings.items():
                    path = cache / str(name)
                    if not path.is_file() or sha256_file(path) != digest:
                        raise ValueError("expanded catalog encoding content drift")
        overlapping = [
            family for family, splits in split_by_family.items() if len(splits) != 1
        ]
        if overlapping:
            raise ValueError("expanded catalog source families cross splits")
        if manifest.get("source_family_disjoint") is not True:
            raise ValueError("expanded catalog does not declare family-disjoint splits")
        if verify_contexts:
            with data_path.open("rb") as handle:
                for entry in sorted(
                    entries, key=lambda row: int(row["catalog_index"])
                ):
                    context = handle.read(stride)
                    if sha256_bytes(context) != entry["context_sha256"]:
                        raise ValueError(
                            "expanded catalog context hash mismatch: "
                            f"{entry['task']}:{entry['dataset_hash']}"
                        )
                if handle.read(1):
                    raise ValueError(
                        "expanded catalog context blob contains trailing bytes"
                    )
        return {
            "pointer": str(pointer_path) if pointer_path else None,
            "pointer_sha256": sha256_file(pointer_path) if pointer_path else None,
            "catalog": str(catalog_path),
            "catalog_sha256": value["catalog_sha256"],
            "data": str(data_path),
            "data_sha256": value["data_sha256"],
            "row_count": len(entries),
            "task_counts": counts,
            "source_family_disjoint": True,
            "catalog_version": 2,
        }
    for field in ("data", "data_sha256"):
        if not value.get(field):
            raise ValueError(f"catalog pointer is missing {field}")
    data_path = Path(str(value["data"]))
    if sha256_file(data_path) != value["data_sha256"]:
        raise ValueError("catalog data pointer hash mismatch")
    if manifest.get("immutable") is not True:
        raise ValueError("catalog manifest is not declared immutable")
    if manifest.get("data_sha256") != value["data_sha256"]:
        raise ValueError("catalog manifest and pointer disagree on context data")
    declared = _declared_catalog_counts(manifest)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("catalog manifest entries are absent")
    counts = validate_catalog_counts(entries, declared)
    if int(manifest.get("row_count", -1)) != len(entries):
        raise ValueError("catalog declared row count mismatch")
    stride = int(manifest.get("record_stride_bytes", -1))
    if stride != CATALOG_STRIDE_BYTES:
        raise ValueError("catalog context stride mismatch")
    if data_path.stat().st_size != len(entries) * stride:
        raise ValueError("catalog context blob row count mismatch")
    indexes = [int(entry.get("catalog_index", -1)) for entry in entries]
    if sorted(indexes) != list(range(len(entries))):
        raise ValueError("catalog indexes are not unique and contiguous")
    if manifest.get("source_family_disjoint") is not True:
        raise ValueError("catalog does not declare family-disjoint splits")
    split_by_family: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        task = str(entry["task"])
        if entry.get("metric") != EXPECTED_METRIC[task]:
            raise ValueError(f"catalog metric mismatch for {task}")
        family = str(entry.get("source_family") or "")
        split = str(entry.get("split") or "")
        if not family or split not in {"train", "validation"}:
            raise ValueError("catalog contains an invalid family/split identity")
        split_by_family[family].add(split)
        for field in ("context_sha256", "train_sha256", "test_sha256"):
            if not _SHA256.fullmatch(str(entry.get(field) or "")):
                raise ValueError(f"catalog entry has invalid {field}")
    overlapping = sorted(
        family for family, splits in split_by_family.items() if len(splits) != 1
    )
    if overlapping:
        raise ValueError(
            f"catalog source families cross splits: {','.join(overlapping[:3])}"
        )
    if verify_contexts:
        with data_path.open("rb") as handle:
            for entry in sorted(entries, key=lambda row: int(row["catalog_index"])):
                context = handle.read(stride)
                if sha256_bytes(context) != entry["context_sha256"]:
                    raise ValueError(
                        f"catalog context hash mismatch: "
                        f"{entry['task']}:{entry['dataset_hash']}"
                    )
            if handle.read(1):
                raise ValueError("catalog context blob contains trailing bytes")
    return {
        "pointer": str(pointer_path) if pointer_path else None,
        "pointer_sha256": sha256_file(pointer_path) if pointer_path else None,
        "catalog": str(catalog_path),
        "catalog_sha256": value["catalog_sha256"],
        "data": str(data_path),
        "data_sha256": value["data_sha256"],
        "row_count": len(entries),
        "task_counts": counts,
        "source_family_disjoint": True,
    }


def evaluation_protocol_sha256(
    *,
    task: str,
    metric: str,
    score_tier: str,
    train_sha256: str,
    test_sha256: str,
    version: str = EVALUATOR_PROTOCOL_VERSION,
) -> str:
    if task not in EXPECTED_METRIC or metric != EXPECTED_METRIC[task]:
        raise ValueError("evaluation protocol task/metric mismatch")
    if score_tier != "test_exact":
        raise ValueError("evaluation protocol score tier must be test_exact")
    if not version.strip():
        raise ValueError("evaluation protocol version is empty")
    for digest in (train_sha256, test_sha256):
        if not _SHA256.fullmatch(digest):
            raise ValueError("evaluation protocol content hash is invalid")
    material = {
        "metric": metric,
        "score_tier": score_tier,
        "task": task,
        "test_sha256": test_sha256,
        "train_sha256": train_sha256,
        "version": version,
    }
    return sha256_text(
        json.dumps(material, sort_keys=True, separators=(",", ":"))
    )


def authoritative_observations(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """One latest, deterministic, corrected observation per task/dataset/genome."""
    chosen: dict[tuple[str, str, str], dict] = {}
    for raw in rows:
        row = dict(raw)
        if not is_strict_receipt(row):
            continue
        key = (
            str(row["task"]),
            str(row["dataset_hash"]),
            str(row["semantic_sha256"]),
        )
        rank = (
            int(row.get("created_at_ms") or 0),
            str(row.get("receipt_sha256") or row.get("receipt_hash") or ""),
        )
        current = chosen.get(key)
        if current is None or rank > current["_authority_rank"]:
            row["_authority_rank"] = rank
            chosen[key] = row
    result = []
    for key in sorted(chosen):
        row = chosen[key]
        row.pop("_authority_rank", None)
        result.append(row)
    return result


def family_disjoint_split(
    dataset_families: Mapping[str, str],
    *,
    validation_fraction: float = 0.1,
    seed: int = 20260726,
) -> dict[str, str]:
    """Stable whole-family assignment with an approximate validation fraction."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    families: dict[str, list[str]] = defaultdict(list)
    for dataset, family in dataset_families.items():
        if not dataset or not family:
            raise ValueError("dataset and family must be nonempty")
        families[family].append(dataset)
    ordered = sorted(
        families,
        key=lambda family: hashlib.sha256(f"{seed}|{family}".encode()).digest(),
    )
    target = min(
        len(dataset_families) - 1,
        max(1, round(len(dataset_families) * validation_fraction)),
    )
    validation: set[str] = set()
    remaining = target
    for family in ordered:
        members = sorted(families[family])
        if remaining <= 0:
            break
        if len(members) <= remaining or not validation:
            validation.update(members)
            remaining -= len(members)
    if not validation or len(validation) == len(dataset_families):
        raise ValueError("family topology cannot produce nonempty disjoint splits")
    return {
        dataset: ("validation" if dataset in validation else "train")
        for dataset in sorted(dataset_families)
    }


def structural_key(genome: str) -> tuple[str, ...]:
    ops = _OP.findall(genome)
    if ops and ops[0] == "pipe":
        ops = ops[1:]
    return tuple(ops)


def rank_common_winners(
    observations: Iterable[tuple[str, str, float]],
) -> list[dict]:
    """Cross-dataset median/lower-quartile percentile, wins, then diversity."""
    by_dataset: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset, semantic, score in observations:
        current = by_dataset[dataset].get(semantic)
        if current is None or score > current:
            by_dataset[dataset][semantic] = float(score)
    percentiles: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    for values in by_dataset.values():
        ranked = sorted(values.items(), key=lambda item: (item[1], item[0]))
        denominator = max(1, len(ranked) - 1)
        best = max(score for _, score in ranked)
        for index, (semantic, score) in enumerate(ranked):
            percentiles[semantic].append(index / denominator)
            if score == best:
                wins[semantic] += 1
    ranked_rows = []
    for semantic, values in percentiles.items():
        values.sort()
        count = len(values)
        median = (
            values[count // 2]
            if count % 2
            else (values[count // 2 - 1] + values[count // 2]) / 2
        )
        lower_quartile = values[int(0.25 * (count - 1))]
        ranked_rows.append(
            {
                "semantic_sha256": semantic,
                "median_percentile": median,
                "lower_quartile_percentile": lower_quartile,
                "win_count": wins[semantic],
                "dataset_count": count,
            }
        )
    return sorted(
        ranked_rows,
        key=lambda row: (
            -row["median_percentile"],
            -row["lower_quartile_percentile"],
            -row["win_count"],
            -row["dataset_count"],
            row["semantic_sha256"],
        ),
    )


def deterministic_deduplicate(candidates: Sequence[Mapping[str, object]]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for raw in candidates:
        row = dict(raw)
        genome = str(row["canonical_genome"])
        semantic = sha256_text(genome)
        if semantic in seen:
            continue
        row["semantic_sha256"] = semantic
        seen.add(semantic)
        result.append(row)
    return result


def compose_seed(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    fills: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Build the four fixed slot groups, filling only after primary selection."""
    result: list[dict] = []
    seen: set[str] = set()
    fill_index = 0
    for group, required in SLOT_COUNTS.items():
        accepted = 0
        for raw in groups.get(group, ()):
            row = dict(raw)
            semantic = sha256_text(str(row["canonical_genome"]))
            if semantic in seen:
                continue
            row.update(
                semantic_sha256=semantic,
                slot_group=group,
                slot_index=accepted,
            )
            result.append(row)
            seen.add(semantic)
            accepted += 1
            if accepted == required:
                break
        while accepted < required:
            while fill_index < len(fills):
                row = dict(fills[fill_index])
                fill_index += 1
                semantic = sha256_text(str(row["canonical_genome"]))
                if semantic not in seen:
                    break
            else:
                raise ValueError(f"cannot fill {group}: only {len(result)} unique candidates")
            row.update(
                semantic_sha256=semantic,
                slot_group=group,
                slot_index=accepted,
                fills_shortage_for=group,
            )
            result.append(row)
            seen.add(semantic)
            accepted += 1
    if len(result) != 256 or len(seen) != 256:
        raise AssertionError("generation-zero seed must have exactly 256 unique genomes")
    return result


def interleave_tasks(
    regression: Sequence[dict], binary: Sequence[dict]
) -> list[dict]:
    result: list[dict] = []
    for index in range(max(len(regression), len(binary))):
        if index < len(regression):
            result.append(regression[index])
        if index < len(binary):
            result.append(binary[index])
    return result


def dynamic_queue_targets(
    jobs_per_minute: Sequence[float],
    effective_cluster_slots: int = 0,
) -> tuple[int, int, float]:
    """Return the 15-minute target and 30-minute hard cap.

    Only launchable queued/starting jobs consume these horizons.  Running,
    dispatching, and held work is reported separately by the caller.
    """

    if effective_cluster_slots < 0:
        raise ValueError("effective cluster slots cannot be negative")
    values = sorted(max(0.0, float(value)) for value in jobs_per_minute)
    if values:
        p95 = values[max(0, math.ceil(0.95 * len(values)) - 1)]
    else:
        p95 = 0.0
    target = min(
        2_048,
        max(
            256,
            2 * effective_cluster_slots,
            math.ceil(p95 * 15.0),
        ),
    )
    hard_cap = min(
        4_096,
        max(
            512,
            target,
            math.ceil(p95 * 30.0),
        ),
    )
    return target, hard_cap, p95


def queue_refill_needed(
    queued_jobs: int, *, target: int = 256, hard_cap: int = 512
) -> int:
    if target < 0 or hard_cap < target:
        raise ValueError("queue hard cap must be at least the queue target")
    if queued_jobs >= target:
        return 0
    return min(target - queued_jobs, hard_cap - queued_jobs)
