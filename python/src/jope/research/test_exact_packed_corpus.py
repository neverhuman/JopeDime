#!/usr/bin/env python3
"""Content-addressed mmap corpus for internal test-exact policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from test_exact_common import atomic_json, prime_source_root, sha256_file

try:
    import orjson

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - stdlib is the portable fallback
    _json_loads = json.loads

PACK_SCHEMA = "PackedCorpus.v3"
LEGACY_PACK_SCHEMA = "PackedCorpus.v2"
PACK_LAYOUT = "content-addressed-label-shards-plus-mmap-columns.v1"
LABEL_SHARD_ROWS = 65_536
TASK_CODE = {"regression": 0, "binary": 1}
SPLIT_CODE = {"train": 0, "validation": 1}


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _json_loads(line)


def _default_tokenizer(row: dict) -> list[int]:
    prime = prime_source_root()
    if str(prime) not in sys.path:
        sys.path.insert(0, str(prime))
    from jope.policy.model_v3 import encode_row_artifacts
    from jope.policy.tokens import build_prime_action_vocab
    from jope.schema import TrainingRow

    vocab = getattr(_default_tokenizer, "_vocab", None)
    if vocab is None:
        vocab = build_prime_action_vocab()
        _default_tokenizer._vocab = vocab
    tokens, _ = encode_row_artifacts(TrainingRow.from_json(row), vocab)
    return tokens


def _open_array(path: Path, dtype: str, count: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count,))


def _packed_records(manifest: dict, root: Path) -> Iterator[dict]:
    if manifest["schema"] == LEGACY_PACK_SCHEMA:
        yield from _jsonl(root / "observations.jsonl")
        return
    for shard in manifest["label_shards"]:
        yield from _jsonl(root / shard["path"])


class _LabelShardWriter:
    def __init__(self, temporary: Path, shared_root: Path):
        self.temporary = temporary
        self.shared_root = shared_root
        self.shared_root.mkdir(parents=True, exist_ok=True)
        self.handle = None
        self.building = None
        self.rows = 0
        self.first_row = 0
        self.shards: list[dict] = []

    def write(self, record: dict, ordinal: int) -> None:
        if self.handle is None:
            self.first_row = ordinal + 1
            self.building = self.temporary / f".label-shard-{ordinal:012d}.tmp"
            self.handle = self.building.open("wb")
        self.handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        self.rows += 1
        if self.rows == LABEL_SHARD_ROWS:
            self._finish(ordinal + 1)

    def close(self, row_count: int) -> list[dict]:
        if self.handle is not None:
            self._finish(row_count)
        return self.shards

    def _finish(self, last_row: int) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        digest = sha256_file(self.building)
        shared = self.shared_root / f"{digest}.jsonl"
        reused = shared.is_file()
        if reused:
            if sha256_file(shared) != digest:
                raise ValueError("shared label shard content drift")
            self.building.unlink()
        else:
            os.replace(self.building, shared)
        name = f"label-shard-{digest}.jsonl"
        os.link(shared, self.temporary / name)
        self.shards.append(
            {
                "first_receipt": self.first_row,
                "last_receipt": last_row,
                "rows": self.rows,
                "path": name,
                "sha256": digest,
                "reused": reused,
                "reused_from_corpus_sha256": None,
            }
        )
        self.handle = None
        self.building = None
        self.rows = 0


def _write_causal_pair_index(path: Path, members: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for member in members:
        grouped.setdefault(member["pair_id"], {}).setdefault(
            member["lane"], []
        ).append(member)
    complete = []
    exclusions: Counter[str] = Counter()
    for pair_id, lanes in sorted(grouped.items()):
        if set(lanes) != {"native_dl_direct", "matched_gp_comparator"} or any(
            len(rows) != 1 for rows in lanes.values()
        ):
            exclusions["partial_or_ambiguous_selected_pair"] += 1
            continue
        direct = lanes["native_dl_direct"][0]
        comparator = lanes["matched_gp_comparator"][0]
        identity_fields = (
            "task",
            "dataset_hash",
            "accepted_parent_checkpoint_sha256",
            "wave",
            "pair_index",
        )
        if (
            any(direct[field] != comparator[field] for field in identity_fields)
            or direct["split"] != "train"
            or comparator["split"] != "train"
            or direct["ordinal"] == comparator["ordinal"]
        ):
            exclusions["identity_or_holdout_mismatch"] += 1
            continue
        complete.append(
            {
                "pair_id": pair_id,
                "task": direct["task"],
                "dataset_hash": direct["dataset_hash"],
                "accepted_parent_checkpoint_sha256": direct[
                    "accepted_parent_checkpoint_sha256"
                ],
                "wave": direct["wave"],
                "pair_index": direct["pair_index"],
                "direct_ordinal": direct["ordinal"],
                "comparator_ordinal": comparator["ordinal"],
                "direct_semantic_sha256": direct["semantic_sha256"],
                "comparator_semantic_sha256": comparator["semantic_sha256"],
                "direct_score": direct["score"],
                "comparator_score": comparator["score"],
                "score_gap": direct["score"] - comparator["score"],
            }
        )
    latest_wave = Counter()
    for pair in complete:
        latest_wave[pair["task"]] = max(
            latest_wave[pair["task"]], int(pair["wave"])
        )
    task_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as output:
        for pair in complete:
            pair["hard"] = abs(float(pair["score_gap"])) <= 0.05 + 1e-12
            pair["novel"] = int(pair["wave"]) >= latest_wave[pair["task"]] - 2
            output.write(json.dumps(pair, sort_keys=True) + "\n")
            task_counts[pair["task"]] += 1
    return {
        "complete_pairs": len(complete),
        "task_counts": dict(task_counts),
        "exclusions": dict(exclusions),
        "fixed_anchor_holdout": "validation rows are excluded from the pair index",
    }


def _write_recipe_e_evidence_index(path: Path, members: list[dict]) -> dict:
    """Write the small, pinned sidecar used to admit descendant-credit training."""

    task_counts: dict[str, Counter[str]] = {}
    descendant_receipts: dict[str, set[str]] = {}
    with path.open("w", encoding="utf-8") as output:
        for member in members:
            task = member["task"]
            lane = member["lane"]
            task_counts.setdefault(task, Counter())[lane] += 1
            if lane == "completed_descendant_trajectories":
                descendant_receipts.setdefault(task, set()).add(
                    member["descendant_credit_receipt_sha256"]
                )
            output.write(json.dumps(member, sort_keys=True) + "\n")
    return {
        "rows": len(members),
        "task_counts": {
            task: {
                **dict(counts),
                "completed_descendant_receipts": len(
                    descendant_receipts.get(task, set())
                ),
            }
            for task, counts in sorted(task_counts.items())
        },
        "censored_descendants_excluded": True,
    }


def materialize(
    manifest_path: Path,
    output_root: Path,
    *,
    tokenizer: Callable[[dict], list[int]] = _default_tokenizer,
    catalog_pointer_path: Path = Path(
        "/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json"
    ),
    catalog_binding_path: Path | None = None,
) -> dict:
    source = json.loads(manifest_path.read_text())
    train_path = Path(source["train"])
    validation_path = Path(source["validation"])
    if catalog_binding_path is not None:
        catalog_binding = json.loads(catalog_binding_path.read_text())
        if catalog_binding.get("schema") != "CatalogBinding.v1":
            raise ValueError("unsupported immutable catalog binding")
        catalog_binding_sha256 = sha256_file(catalog_binding_path)
        if catalog_binding_sha256 != source.get(
            "catalog_binding_sha256", catalog_binding_sha256
        ):
            raise ValueError("training corpus catalog binding hash mismatch")
        catalog_pointer = {
            "catalog": catalog_binding["catalog_path"],
            "catalog_sha256": catalog_binding["catalog_sha256"],
            "data": catalog_binding["data_path"],
            "data_sha256": catalog_binding["data_sha256"],
        }
        for path_key, hash_key in (
            ("catalog", "catalog_sha256"),
            ("data", "data_sha256"),
        ):
            if sha256_file(Path(catalog_pointer[path_key])) != catalog_pointer[hash_key]:
                raise ValueError(f"catalog binding {path_key} hash mismatch")
    else:
        catalog_binding = None
        catalog_pointer = json.loads(catalog_pointer_path.read_text())
        catalog_binding_sha256 = hashlib.sha256(
            json.dumps(
                catalog_pointer, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
    if source["catalog_sha256"] != catalog_pointer["catalog_sha256"]:
        raise ValueError("training corpus and packed catalog identities differ")
    train_sha256 = sha256_file(train_path)
    validation_sha256 = sha256_file(validation_path)
    if train_sha256 != source["train_sha256"]:
        raise ValueError("training JSONL changed after corpus freeze")
    if validation_sha256 != source["validation_sha256"]:
        raise ValueError("validation JSONL changed after corpus freeze")
    catalog_index = {
        (entry["task"], entry["dataset_hash"]): int(entry["catalog_index"])
        for entry in catalog["entries"]
    }
    binding = hashlib.sha256(
        json.dumps(
            {
                "train_sha256": train_sha256,
                "validation_sha256": validation_sha256,
                "catalog_sha256": source["catalog_sha256"],
                "catalog_binding_sha256": catalog_binding_sha256,
                "schema": PACK_SCHEMA,
                "layout": PACK_LAYOUT,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    final = output_root / binding
    manifest_final = final / "manifest.json"
    if manifest_final.is_file():
        value = json.loads(manifest_final.read_text())
        verify(value, final)
        return value

    row_count = sum(
        1
        for path in (train_path, validation_path)
        for _ in _jsonl(path)
    )
    temporary = output_root / f".{binding}.tmp-{os.getpid()}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        arrays = {
            "dataset_indices": _open_array(
                temporary / "dataset_indices.npy", "<u4", row_count
            ),
            "tasks": _open_array(temporary / "tasks.npy", "u1", row_count),
            "splits": _open_array(temporary / "splits.npy", "u1", row_count),
            "scores": _open_array(temporary / "scores.npy", "<f4", row_count),
            "runtime_ms": _open_array(
                temporary / "runtime_ms.npy", "<f4", row_count
            ),
            "genome_offsets": _open_array(
                temporary / "genome_offsets.npy", "<u8", row_count + 1
            ),
            "token_offsets": _open_array(
                temporary / "token_offsets.npy", "<u8", row_count + 1
            ),
        }
        arrays["genome_offsets"][0] = 0
        arrays["token_offsets"][0] = 0
        genome_cursor = token_cursor = 0
        task_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        causal_members = []
        recipe_e_members = []
        shard_writer = _LabelShardWriter(
            temporary, output_root / "_label_shards_v3"
        )
        with (
            (temporary / "genomes.bin").open("wb") as genomes,
            (temporary / "token_ids.u16.bin").open("wb") as token_output,
            (temporary / "selected_row_index.sha256.bin").open("wb") as selected_index,
        ):
            rows = (
                (row, split)
                for path, split in (
                    (train_path, "train"),
                    (validation_path, "validation"),
                )
                for row in _jsonl(path)
            )
            for ordinal, (row, split) in enumerate(rows):
                metadata = row.get("metadata") or {}
                task = str(metadata.get("task_type") or "")
                dataset = str(metadata.get("dataset_hash") or row.get("dataset_id") or "")
                key = (task, dataset)
                if key not in catalog_index:
                    raise ValueError(f"packed row is absent from the catalog: {task}:{dataset}")
                if metadata.get("test_adaptive") is not True:
                    raise ValueError("packed row does not disclose test adaptation")
                if metadata.get("benchmark_valid") is not False:
                    raise ValueError("packed row incorrectly claims benchmark validity")
                genome = str(row.get("canonical_genome") or row.get("genome") or "")
                genome_bytes = genome.encode()
                raw_tokens = tokenizer(row)
                if any(
                    not isinstance(token, (int, np.integer))
                    or int(token) < 0
                    or int(token) > 65_535
                    for token in raw_tokens
                ):
                    raise ValueError("policy token is outside the packed uint16 vocabulary")
                tokens = np.asarray(raw_tokens, dtype="<u2")
                if not len(tokens):
                    raise ValueError("packed row has an empty policy token sequence")
                genomes.write(genome_bytes)
                tokens.tofile(token_output)
                genome_cursor += len(genome_bytes)
                token_cursor += len(tokens)
                arrays["dataset_indices"][ordinal] = catalog_index[key]
                arrays["tasks"][ordinal] = TASK_CODE[task]
                arrays["splits"][ordinal] = SPLIT_CODE[split]
                arrays["scores"][ordinal] = float(row["score"])
                arrays["runtime_ms"][ordinal] = float(row.get("cost_ms") or 0.0)
                arrays["genome_offsets"][ordinal + 1] = genome_cursor
                arrays["token_offsets"][ordinal + 1] = token_cursor
                record = {
                    "row": ordinal,
                    "task": task,
                    "dataset_hash": dataset,
                    "engine": row.get("engine"),
                    "semantic_sha256": metadata.get("semantic_plan_hash"),
                    "source": row.get("source"),
                    "receipt_sha256": metadata.get("receipt_sha256"),
                    "model_checkpoint_sha256": metadata.get(
                        "model_checkpoint_sha256"
                    ),
                    "generation": metadata.get("generation"),
                    "metric": row.get("metric"),
                    "valid": row.get("valid", True),
                    "metadata": metadata,
                }
                shard_writer.write(record, ordinal)
                receipt_identity = str(metadata.get("receipt_sha256") or "")
                if len(receipt_identity) == 64 and all(
                    character in "0123456789abcdef"
                    for character in receipt_identity
                ):
                    selected_index.write(bytes.fromhex(receipt_identity))
                else:
                    selected_index.write(
                        hashlib.sha256(
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).digest()
                    )
                if metadata.get("causal_pair_id"):
                    causal_members.append(
                        {
                            "pair_id": str(metadata["causal_pair_id"]),
                            "lane": str(metadata.get("causal_lane") or ""),
                            "ordinal": ordinal,
                            "split": split,
                            "task": task,
                            "dataset_hash": dataset,
                            "accepted_parent_checkpoint_sha256": str(
                                metadata.get(
                                    "causal_accepted_parent_checkpoint_sha256"
                                )
                                or ""
                            ),
                            "wave": int(metadata.get("causal_wave") or 0),
                            "pair_index": int(
                                metadata.get("causal_pair_index") or 0
                            ),
                            "semantic_sha256": str(
                                metadata.get("semantic_plan_hash") or ""
                            ),
                            "score": float(row["score"]),
                        }
                    )
                descendant_receipt = str(
                    metadata.get("descendant_credit_receipt_sha256") or ""
                )
                if descendant_receipt and not metadata.get(
                    "descendant_censored", False
                ):
                    if len(descendant_receipt) != 64 or any(
                        character not in "0123456789abcdef"
                        for character in descendant_receipt
                    ):
                        raise ValueError(
                            "recipe E descendant receipt has an invalid identity"
                        )
                    if metadata.get("descendant_horizon") not in {1, 2, 4, 8}:
                        raise ValueError(
                            "recipe E descendant row has an invalid horizon"
                        )
                    recipe_e_members.append(
                        {
                            "ordinal": ordinal,
                            "task": task,
                            "lane": "completed_descendant_trajectories",
                            "descendant_credit_receipt_sha256": descendant_receipt,
                            "descendant_horizon": int(
                                metadata["descendant_horizon"]
                            ),
                        }
                    )
                elif any(
                    bool(metadata.get(name))
                    for name in (
                        "validity_hard_negative",
                        "instability_hard_negative",
                        "adversarial_hard_case",
                        "candidate_invalid",
                    )
                ):
                    recipe_e_members.append(
                        {
                            "ordinal": ordinal,
                            "task": task,
                            "lane": "validity_instability_hard_negatives",
                        }
                    )
                task_counts[task] += 1
                split_counts[split] += 1
        label_shards = shard_writer.close(row_count)
        causal_pair_stats = _write_causal_pair_index(
            temporary / "causal_pairs.jsonl", causal_members
        )
        recipe_e_evidence_stats = _write_recipe_e_evidence_index(
            temporary / "recipe_e_evidence.jsonl", recipe_e_members
        )
        for array in arrays.values():
            array.flush()
        del arrays

        files = {}
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json":
                continue
            files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        packed = {
            "schema": PACK_SCHEMA,
            "layout": PACK_LAYOUT,
            "watermark": source["source_label_watermark"],
            "binding_sha256": binding,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "train_sha256": source["train_sha256"],
            "validation_sha256": source["validation_sha256"],
            "source_label_watermark": source["source_label_watermark"],
            "catalog_sha256": source["catalog_sha256"],
            "catalog_binding": (
                str(catalog_binding_path) if catalog_binding_path is not None else None
            ),
            "catalog_binding_sha256": catalog_binding_sha256,
            "catalog_data": catalog_pointer["data"],
            "catalog_data_sha256": catalog_pointer["data_sha256"],
            "row_count": row_count,
            "task_counts": dict(task_counts),
            "split_counts": dict(split_counts),
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "selected_row_index": {
                "path": "selected_row_index.sha256.bin",
                "sha256": files["selected_row_index.sha256.bin"]["sha256"],
                "dtype": "sha256",
                "rows": row_count,
            },
            "selected_row_index_dtype": "sha256",
            "causal_pair_index": "causal_pairs.jsonl",
            "exact_pair_sidecar": {
                "path": "causal_pairs.jsonl",
                "sha256": files["causal_pairs.jsonl"]["sha256"],
                "dtype": "jsonl",
                "rows": causal_pair_stats["complete_pairs"],
            },
            "causal_pair_count": causal_pair_stats["complete_pairs"],
            "causal_pair_stats": causal_pair_stats,
            "recipe_e_evidence_index": "recipe_e_evidence.jsonl",
            "recipe_e_evidence_sidecar": {
                "path": "recipe_e_evidence.jsonl",
                "sha256": files["recipe_e_evidence.jsonl"]["sha256"],
                "dtype": "jsonl",
                "rows": recipe_e_evidence_stats["rows"],
            },
            "recipe_e_evidence_stats": recipe_e_evidence_stats,
            "task_columns": {
                "task_code": "tasks.npy",
                "split_code": "splits.npy",
            },
            "task_column": {
                "path": "tasks.npy",
                "sha256": files["tasks.npy"]["sha256"],
                "dtype": "u1",
                "rows": row_count,
            },
            "split_column": {
                "path": "splits.npy",
                "sha256": files["splits.npy"]["sha256"],
                "dtype": "u1",
                "rows": row_count,
            },
            "loader": {
                "deterministic_workers": 4,
                "pinned_memory_prefetch": True,
            },
            "label_shards": label_shards,
            "created_at_ms": int(time.time() * 1_000),
            "files": files,
        }
        atomic_json(temporary / "manifest.json", packed)
        os.replace(temporary, final)
        verify(packed, final)
        return packed
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify(
    manifest: dict,
    root: Path,
    *,
    source_manifest_path: Path | None = None,
    source_paths: dict[str, Path] | None = None,
) -> None:
    if manifest.get("schema") not in {PACK_SCHEMA, LEGACY_PACK_SCHEMA}:
        raise ValueError("unsupported packed corpus schema")
    if manifest.get("test_adaptive") is not True or manifest.get("benchmark_valid") is not False:
        raise ValueError("packed corpus provenance labels are invalid")
    binding_path = manifest.get("catalog_binding")
    if binding_path is not None:
        binding_path = Path(binding_path)
        if (
            not binding_path.is_file()
            or sha256_file(binding_path) != manifest["catalog_binding_sha256"]
        ):
            raise ValueError("packed corpus catalog binding hash mismatch")
    source_manifest = source_manifest_path or Path(manifest["source_manifest"])
    if (
        not source_manifest.is_file()
        or sha256_file(source_manifest) != manifest["source_manifest_sha256"]
    ):
        raise ValueError("packed corpus source manifest hash mismatch")
    source = json.loads(source_manifest.read_text())
    for key in ("train", "validation"):
        expected = manifest[f"{key}_sha256"]
        if source.get(f"{key}_sha256") != expected:
            raise ValueError(f"packed corpus source {key} binding mismatch")
        # A packed corpus is a complete immutable training input. Requiring its
        # 9+ GiB source JSONL files here made every packed trainer stage and
        # re-hash those files before Python could start, defeating the packed
        # boundary and exceeding the scheduler's dispatch window. Callers may
        # still provide source_paths for an explicit full-source audit.
        if source_paths is not None:
            path = source_paths[key]
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"packed corpus source {key} hash mismatch")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"packed corpus file hash mismatch: {name}")
    row_count = int(manifest["row_count"])
    for name in ("dataset_indices", "tasks", "splits", "scores", "runtime_ms"):
        if len(np.load(root / f"{name}.npy", mmap_mode="r")) != row_count:
            raise ValueError(f"packed corpus row count mismatch: {name}")
    for name in ("genome_offsets", "token_offsets"):
        offsets = np.load(root / f"{name}.npy", mmap_mode="r")
        if len(offsets) != row_count + 1 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"packed corpus offsets are invalid: {name}")
    token_offsets = np.load(root / "token_offsets.npy", mmap_mode="r")
    if (root / "token_ids.u16.bin").stat().st_size != int(token_offsets[-1]) * 2:
        raise ValueError("packed corpus token byte length is invalid")
    if manifest["schema"] == LEGACY_PACK_SCHEMA:
        with (root / manifest["reversible_index"]).open(encoding="utf-8") as handle:
            indexed = sum(1 for line in handle if line.strip())
        if indexed != row_count:
            raise ValueError("packed corpus reversible-index row count mismatch")
    else:
        selected = root / manifest["selected_row_index"]["path"]
        if selected.name == "observations.jsonl" or selected.stat().st_size != row_count * 32:
            raise ValueError("packed selected-row digest index is invalid")
        shard_rows = 0
        previous = 0
        for shard in manifest["label_shards"]:
            if (
                int(shard["first_receipt"]) <= previous
                or int(shard["last_receipt"]) < int(shard["first_receipt"])
                or sha256_file(root / shard["path"]) != shard["sha256"]
            ):
                raise ValueError("packed content-addressed label shard is invalid")
            rows = sum(1 for _ in _jsonl(root / shard["path"]))
            if rows != int(shard["rows"]):
                raise ValueError("packed label-shard row count mismatch")
            shard_rows += rows
            previous = int(shard["last_receipt"])
        if shard_rows != row_count:
            raise ValueError("packed label-shard total differs from selected rows")
    causal_index = manifest.get("causal_pair_index")
    if causal_index is not None:
        causal_rows = 0
        seen_pairs = set()
        with (root / causal_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json_loads(line)
                pair_id = str(row.get("pair_id") or "")
                direct = int(row.get("direct_ordinal", -1))
                comparator = int(row.get("comparator_ordinal", -1))
                if (
                    len(pair_id) != 64
                    or any(character not in "0123456789abcdef" for character in pair_id)
                    or pair_id in seen_pairs
                    or not 0 <= direct < row_count
                    or not 0 <= comparator < row_count
                    or direct == comparator
                ):
                    raise ValueError("packed causal pair index is invalid")
                seen_pairs.add(pair_id)
                causal_rows += 1
        if causal_rows != int(manifest.get("causal_pair_count") or 0):
            raise ValueError("packed causal pair count mismatch")
    recipe_e_index = manifest.get("recipe_e_evidence_index")
    if recipe_e_index is not None:
        recipe_e_rows = 0
        with (root / recipe_e_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json_loads(line)
                lane = row.get("lane")
                ordinal = int(row.get("ordinal", -1))
                if (
                    row.get("task") not in TASK_CODE
                    or lane
                    not in {
                        "completed_descendant_trajectories",
                        "validity_instability_hard_negatives",
                    }
                    or not 0 <= ordinal < row_count
                ):
                    raise ValueError("packed recipe E evidence index is invalid")
                if lane == "completed_descendant_trajectories":
                    receipt = str(
                        row.get("descendant_credit_receipt_sha256") or ""
                    )
                    if (
                        len(receipt) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in receipt
                        )
                        or row.get("descendant_horizon") not in {1, 2, 4, 8}
                    ):
                        raise ValueError(
                            "packed recipe E descendant evidence is invalid"
                        )
                recipe_e_rows += 1
        expected_rows = (
            manifest.get("recipe_e_evidence_sidecar") or {}
        ).get("rows")
        if recipe_e_rows != int(expected_rows or 0):
            raise ValueError("packed recipe E evidence count mismatch")


class Float32View(Sequence[float]):
    """List-compatible view over one mmap context row without owning its floats."""

    def __init__(self, values: np.ndarray):
        self._values = values

    def __bool__(self) -> bool:
        return len(self._values) > 0

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return (float(value) for value in self._values)

    def __getitem__(self, item):
        value = self._values[item]
        if isinstance(item, slice):
            return [float(number) for number in value]
        return float(value)


def load_training_rows(
    manifest_path: Path,
    *,
    task: str | None = None,
    source_manifest_path: Path | None = None,
    source_train: Path | None = None,
    source_validation: Path | None = None,
    catalog_data: Path | None = None,
):
    """Return TrainingRows sharing mmap contexts plus their packed token cache."""

    if task not in {None, "binary", "regression"}:
        raise ValueError("packed corpus task must be binary or regression")
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    source_paths = None
    if source_train is not None or source_validation is not None:
        if source_train is None or source_validation is None:
            raise ValueError("both packed source row paths must be provided")
        source_paths = {"train": source_train, "validation": source_validation}
    verify(
        manifest,
        root,
        source_manifest_path=source_manifest_path,
        source_paths=source_paths,
    )
    prime = prime_source_root()
    if str(prime) not in sys.path:
        sys.path.insert(0, str(prime))
    from jope.policy.automaton import replay_masks
    from jope.policy.tokens import build_prime_action_vocab
    from jope.schema import TrainingRow

    catalog_path = catalog_data or Path(manifest["catalog_data"])
    if sha256_file(catalog_path) != manifest["catalog_data_sha256"]:
        raise ValueError("packed corpus catalog data hash mismatch")
    catalog_rows = catalog_path.stat().st_size // (2_048 * 4)
    catalog = np.memmap(
        catalog_path,
        dtype="<f4",
        mode="r",
        shape=(catalog_rows, 2_048),
    )
    dataset_indices = np.load(root / "dataset_indices.npy", mmap_mode="r")
    scores = np.load(root / "scores.npy", mmap_mode="r")
    runtime_ms = np.load(root / "runtime_ms.npy", mmap_mode="r")
    genome_offsets = np.load(root / "genome_offsets.npy", mmap_mode="r")
    token_offsets = np.load(root / "token_offsets.npy", mmap_mode="r")
    genomes = np.memmap(root / "genomes.bin", dtype="u1", mode="r")
    token_ids = np.memmap(root / "token_ids.u16.bin", dtype="<u2", mode="r")
    vocab = build_prime_action_vocab()
    rows = []
    rows_by_ordinal = {}
    artifact_cache = {}
    for ordinal, record in enumerate(_packed_records(manifest, root)):
        if ordinal and ordinal % 25_000 == 0:
            print(
                "packed-loader "
                f"rows={ordinal}/{manifest['row_count']} "
                f"percent={100.0 * ordinal / int(manifest['row_count']):.1f}",
                flush=True,
            )
        if task is not None and record.get("task") != task:
            continue
        context = catalog[int(dataset_indices[ordinal])]
        genome_start, genome_end = map(
            int, genome_offsets[ordinal : ordinal + 2]
        )
        token_start, token_end = map(int, token_offsets[ordinal : ordinal + 2])
        genome = bytes(genomes[genome_start:genome_end]).decode()
        row = object.__new__(TrainingRow)
        row.dataset_id = record["dataset_hash"]
        row.source = record["source"]
        row.embedding = Float32View(context[:1_536])
        row.target_embedding = Float32View(context[1_536:])
        row.genome = genome
        row.canonical_genome = genome
        row.valid = bool(record["valid"])
        row.metric = record["metric"]
        row.score = float(scores[ordinal])
        row.rank = None
        row.elo = None
        row.cost_ms = float(runtime_ms[ordinal])
        row.family_bitmap = {}
        row.complexity = None
        row.runtime_class = None
        row.feedback_origin = None
        row.engine = record["engine"]
        row.metadata = record["metadata"]
        rows.append(row)
        rows_by_ordinal[ordinal] = row
        tokens = [int(value) for value in token_ids[token_start:token_end]]
        artifact_cache[id(row)] = (tokens, replay_masks(tokens, vocab))
    causal_index = manifest.get("causal_pair_index")
    if causal_index is not None:
        with (root / causal_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                pair = _json_loads(line)
                if task is not None and pair.get("task") != task:
                    continue
                direct = rows_by_ordinal.get(int(pair["direct_ordinal"]))
                comparator = rows_by_ordinal.get(int(pair["comparator_ordinal"]))
                if direct is None or comparator is None:
                    raise ValueError("packed causal pair row is absent from task view")
                common = {
                    "causal_pair_id": pair["pair_id"],
                    "causal_pair_index": int(pair["pair_index"]),
                    "causal_accepted_parent_checkpoint_sha256": pair[
                        "accepted_parent_checkpoint_sha256"
                    ],
                    "causal_wave": int(pair["wave"]),
                    "causal_score_gap": float(pair["score_gap"]),
                    "causal_pair_hard": bool(pair["hard"]),
                    "causal_pair_novel": bool(pair["novel"]),
                }
                direct.metadata.update(
                    {**common, "causal_lane": "native_dl_direct"}
                )
                comparator.metadata.update(
                    {**common, "causal_lane": "matched_gp_comparator"}
                )
    expected_rows = (
        manifest["row_count"]
        if task is None
        else int((manifest.get("task_counts") or {}).get(task) or 0)
    )
    if len(rows) != expected_rows:
        raise ValueError("packed observation index row count mismatch")
    return rows, artifact_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--catalog-pointer",
        type=Path,
        default=Path(
            "/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json"
        ),
    )
    parser.add_argument("--catalog-binding", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.manifest,
                args.output_root,
                catalog_pointer_path=args.catalog_pointer,
                catalog_binding_path=args.catalog_binding,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
