from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

TOKENIZER_VERSION = "dime-score-tokenizer-v2-global-genome-split"
PAD = 0
BOS = 1
EOS = 2
SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield row


def action_token(action: dict[str, Any]) -> str:
    kind = str(action.get("kind") or "")
    operation = str(action.get("operation") or "")
    parameter = str(action.get("parameter") or "")
    if "numeric_value" in action:
        value = "<NUM>"
    else:
        value = str(action.get("categorical_value") or "")
    if not kind:
        raise ValueError("gene action lacks kind")
    return "\x1f".join((kind, operation, parameter, value))


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("kind") or "")


def _validate_source(corpus: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = corpus / "manifest.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "jope-dime.score-corpus.v1":
        raise ValueError("unsupported Dime score corpus manifest")
    examples = corpus / str(manifest.get("examples_file"))
    gene_schema_path = corpus / str(manifest.get("gene_schema_file"))
    if sha256_file(examples) != manifest.get("examples_sha256"):
        raise ValueError("Dime examples SHA-256 mismatch")
    if sha256_file(gene_schema_path) != manifest.get("gene_schema_sha256"):
        raise ValueError("Dime gene schema SHA-256 mismatch")
    gene_schema = json.loads(gene_schema_path.read_text(encoding="utf-8"))
    if gene_schema.get("schema") != "jope-dime.gene-schema.v1":
        raise ValueError("unsupported Dime gene schema")
    return manifest, gene_schema, examples


def _deterministic_row_splits(
    identities: Sequence[str],
    semantic_hashes: Sequence[str],
) -> np.ndarray:
    if len(identities) != len(semantic_hashes):
        raise ValueError("dataset and semantic identity counts differ")
    unique_semantics = sorted(
        set(semantic_hashes),
        key=lambda semantic: hashlib.sha256(semantic.encode()).digest(),
    )
    count = len(unique_semantics)
    train_end = int(math.floor(0.8 * count))
    validation_end = train_end + int(math.floor(0.1 * count))
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        validation_end = max(train_end + 1, min(validation_end, count - 1))
    semantic_split = {
        semantic: 0 if ordinal < train_end else 1 if ordinal < validation_end else 2
        for ordinal, semantic in enumerate(unique_semantics)
    }
    return np.asarray(
        [semantic_split[semantic] for semantic in semantic_hashes],
        dtype=np.uint8,
    )


def _dataset_holdout_splits(dataset_identities: Sequence[str]) -> np.ndarray:
    order = sorted(
        range(len(dataset_identities)),
        key=lambda index: hashlib.sha256(dataset_identities[index].encode()).digest(),
    )
    count = len(order)
    train_end = int(math.floor(0.8 * count))
    validation_end = train_end + int(math.floor(0.1 * count))
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        validation_end = max(train_end + 1, min(validation_end, count - 1))
    split = np.empty(count, dtype=np.uint8)
    for ordinal, index in enumerate(order):
        split[index] = 0 if ordinal < train_end else 1 if ordinal < validation_end else 2
    return split


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_prepared(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.v1.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "jope-prime.dime-tokenized-corpus.v1":
        raise ValueError("unsupported tokenized corpus manifest")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"tokenized corpus SHA-256 mismatch: {name}")
    return manifest


def prepare_corpus(corpus_root: str | Path) -> Path:
    corpus = Path(corpus_root).resolve()
    source, gene_schema, examples_path = _validate_source(corpus)
    binding = hashlib.sha256(
        (
            TOKENIZER_VERSION
            + "\0"
            + source["examples_sha256"]
            + "\0"
            + source["gene_schema_sha256"]
        ).encode()
    ).hexdigest()
    final = corpus / "tokenized" / binding
    if final.exists():
        _verify_prepared(final)
        return final

    tokens = {"PAD", "BOS", "EOS"}
    action_types = {"special"}
    identities: list[str] = []
    semantics: list[str] = []
    lengths: list[int] = []
    percentile: list[float] = []
    validity: list[float] = []
    rmse: list[float] = []
    for row in _iter_jsonl(examples_path):
        cohort = row.get("cohort")
        dataset = row.get("dataset")
        semantic = row.get("semantic_sha256")
        actions = row.get("genome", {}).get("actions")
        if not all(isinstance(value, str) and value for value in (cohort, dataset, semantic)):
            raise ValueError("score example lacks cohort, dataset, or semantic identity")
        if not isinstance(actions, list):
            raise ValueError(f"score example {semantic} has no structured actions")
        if hashlib.sha256(str(row.get("canonical_genome")).encode()).hexdigest() != semantic:
            raise ValueError(f"score example {semantic} canonical identity mismatch")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(f"score example {semantic} has malformed action")
            tokens.add(action_token(action))
            action_types.add(_action_type(action))
            numeric = action.get("numeric_value")
            if numeric is not None and (not isinstance(numeric, (int, float)) or not math.isfinite(numeric)):
                raise ValueError(f"score example {semantic} has non-finite numeric action")
        identities.append(f"{cohort}:{dataset}")
        semantics.append(semantic)
        lengths.append(len(actions) + 2)
        score = row.get("percentile")
        score_value = float(score) if score is not None else math.nan
        if score is not None and (
            not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0
        ):
            raise ValueError(f"score example {semantic} has invalid percentile")
        percentile.append(score_value)
        success = float(row.get("success_rate"))
        if not math.isfinite(success) or not 0.0 <= success <= 1.0:
            raise ValueError(f"score example {semantic} has invalid success rate")
        validity.append(success)
        score_rmse = row.get("rmse")
        rmse_value = float(score_rmse) if score_rmse is not None else math.nan
        if score_rmse is not None and (
            not math.isfinite(rmse_value) or rmse_value < 0.0
        ):
            raise ValueError(f"score example {semantic} has invalid RMSE")
        if (score is None) != (score_rmse is None):
            raise ValueError(
                f"score example {semantic} has inconsistent percentile and RMSE"
            )
        rmse.append(rmse_value)
    if not identities:
        raise ValueError("Dime score corpus has no examples")
    if len(identities) != source.get("unique_examples"):
        raise ValueError("Dime score corpus example count differs from manifest")

    token_list = ["PAD", "BOS", "EOS"] + sorted(tokens - {"PAD", "BOS", "EOS"})
    token_to_id = {token: index for index, token in enumerate(token_list)}
    type_list = ["special"] + sorted(action_types - {"special"})
    type_to_id = {name: index for index, name in enumerate(type_list)}
    dataset_list = sorted(set(identities))
    dataset_to_id = {identity: index for index, identity in enumerate(dataset_list)}
    row_split = _deterministic_row_splits(identities, semantics)
    heldout_dataset_split = _dataset_holdout_splits(dataset_list)

    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".build-{binding}-{os.getpid()}"
    if temporary.exists():
        raise ValueError(f"stale tokenized corpus build exists: {temporary}")
    temporary.mkdir()
    total_tokens = sum(lengths)
    token_ids = np.lib.format.open_memmap(
        temporary / "token_ids.npy", mode="w+", dtype=np.int32, shape=(total_tokens,)
    )
    numeric_values = np.lib.format.open_memmap(
        temporary / "numeric_values.npy", mode="w+", dtype=np.float64, shape=(total_tokens,)
    )
    numeric_mask = np.lib.format.open_memmap(
        temporary / "numeric_mask.npy", mode="w+", dtype=np.uint8, shape=(total_tokens,)
    )
    action_type_ids = np.lib.format.open_memmap(
        temporary / "action_type_ids.npy", mode="w+", dtype=np.uint8, shape=(total_tokens,)
    )
    offsets = np.lib.format.open_memmap(
        temporary / "offsets.npy", mode="w+", dtype=np.int64, shape=(len(identities) + 1,)
    )
    dataset_indices = np.lib.format.open_memmap(
        temporary / "dataset_indices.npy", mode="w+", dtype=np.int32, shape=(len(identities),)
    )
    percentiles = np.lib.format.open_memmap(
        temporary / "percentiles.npy", mode="w+", dtype=np.float32, shape=(len(identities),)
    )
    validities = np.lib.format.open_memmap(
        temporary / "validities.npy", mode="w+", dtype=np.float32, shape=(len(identities),)
    )
    rmses = np.lib.format.open_memmap(
        temporary / "rmse.npy", mode="w+", dtype=np.float64, shape=(len(identities),)
    )
    row_splits = np.lib.format.open_memmap(
        temporary / "splits.npy", mode="w+", dtype=np.uint8, shape=(len(identities),)
    )
    dataset_splits = np.lib.format.open_memmap(
        temporary / "dataset_splits.npy", mode="w+", dtype=np.uint8, shape=(len(identities),)
    )
    numeric_values.fill(0.0)
    numeric_mask.fill(0)
    offsets[0] = 0
    numeric_samples: dict[int, list[float]] = defaultdict(list)
    cursor = 0
    for row_index, row in enumerate(_iter_jsonl(examples_path)):
        actions = row["genome"]["actions"]
        sequence: list[tuple[int, float, int, int]] = [(BOS, 0.0, 0, type_to_id["special"])]
        for action in actions:
            token_id = token_to_id[action_token(action)]
            type_id = type_to_id[_action_type(action)]
            if "numeric_value" in action:
                value = float(action["numeric_value"])
                sequence.append((token_id, value, 1, type_id))
                if row_split[row_index] == 0:
                    numeric_samples[token_id].append(math.copysign(math.log1p(abs(value)), value))
            else:
                sequence.append((token_id, 0.0, 0, type_id))
        sequence.append((EOS, 0.0, 0, type_to_id["special"]))
        if len(sequence) != lengths[row_index]:
            raise AssertionError("structured sequence length changed between corpus passes")
        end = cursor + len(sequence)
        token_ids[cursor:end] = [item[0] for item in sequence]
        numeric_values[cursor:end] = [item[1] for item in sequence]
        numeric_mask[cursor:end] = [item[2] for item in sequence]
        action_type_ids[cursor:end] = [item[3] for item in sequence]
        offsets[row_index + 1] = end
        dataset_id = dataset_to_id[identities[row_index]]
        dataset_indices[row_index] = dataset_id
        percentiles[row_index] = percentile[row_index]
        validities[row_index] = validity[row_index]
        rmses[row_index] = rmse[row_index]
        row_splits[row_index] = row_split[row_index]
        dataset_splits[row_index] = heldout_dataset_split[dataset_id]
        cursor = end
    if cursor != total_tokens:
        raise AssertionError("tokenized corpus silently dropped or added actions")

    numeric_stats: dict[str, dict[str, float | int]] = {}
    for token_id, values in sorted(numeric_samples.items()):
        array = np.asarray(values, dtype=np.float64)
        scale = float(array.std())
        numeric_stats[str(token_id)] = {
            "count": len(values),
            "mean": float(array.mean()),
            "scale": scale if scale >= 1e-8 else 1.0,
        }
    arrays = (
        token_ids,
        numeric_values,
        numeric_mask,
        action_type_ids,
        offsets,
        dataset_indices,
        percentiles,
        validities,
        rmses,
        row_splits,
        dataset_splits,
    )
    for array in arrays:
        array.flush()
    del arrays
    for name in (
        "token_ids.npy",
        "numeric_values.npy",
        "numeric_mask.npy",
        "action_type_ids.npy",
        "offsets.npy",
        "dataset_indices.npy",
        "percentiles.npy",
        "validities.npy",
        "rmse.npy",
        "splits.npy",
        "dataset_splits.npy",
    ):
        _fsync_file(temporary / name)

    index = {
        "schema": "jope-prime.dime-token-index.v1",
        "tokenizer_version": TOKENIZER_VERSION,
        "tokens": token_list,
        "action_types": type_list,
        "datasets": dataset_list,
        "semantic_sha256": semantics,
        "numeric_stats": numeric_stats,
        "split_names": list(SPLIT_NAMES),
        "split_counts": {
            name: int(np.sum(row_split == index)) for index, name in enumerate(SPLIT_NAMES)
        },
        "row_split_contract": {
            "method": "sha256-global-semantic-order",
            "partition_unit": "semantic_sha256",
            "fractions": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
            "semantic_overlap": False,
            "unique_genome_counts": {
                name: len(
                    {
                        semantics[row]
                        for row in np.flatnonzero(row_split == split_index)
                    }
                )
                for split_index, name in enumerate(SPLIT_NAMES)
            },
            "dataset_counts": {
                name: len(
                    {
                        identities[row]
                        for row in np.flatnonzero(row_split == split_index)
                    }
                )
                for split_index, name in enumerate(SPLIT_NAMES)
            },
        },
        "dataset_holdout_counts": {
            name: int(np.sum(heldout_dataset_split == index))
            for index, name in enumerate(SPLIT_NAMES)
        },
        "gene_schema_contract_sha256": gene_schema.get("contract_sha256"),
    }
    _write_json(temporary / "index.v1.json", index)
    file_names = sorted(path.name for path in temporary.iterdir() if path.is_file())
    file_hashes = {name: sha256_file(temporary / name) for name in file_names}
    manifest = {
        "schema": "jope-prime.dime-tokenized-corpus.v1",
        "binding_sha256": binding,
        "tokenizer_version": TOKENIZER_VERSION,
        "source_archive_sha256": source["archive_sha256"],
        "source_examples_sha256": source["examples_sha256"],
        "source_gene_schema_sha256": source["gene_schema_sha256"],
        "examples": len(identities),
        "tokens": total_tokens,
        "vocabulary": len(token_list),
        "datasets": len(dataset_list),
        "max_sequence_length": max(lengths),
        "files": file_hashes,
    }
    _write_json(temporary / "manifest.v1.json", manifest)
    try:
        temporary.rename(final)
    except FileExistsError:
        shutil.rmtree(temporary)
    _verify_prepared(final)
    return final


class TokenizedCorpus:
    def __init__(self, root: str | Path, *, verify: bool = True) -> None:
        self.root = Path(root)
        self.manifest = (
            _verify_prepared(self.root)
            if verify
            else json.loads((self.root / "manifest.v1.json").read_text(encoding="utf-8"))
        )
        self.index = json.loads((self.root / "index.v1.json").read_text(encoding="utf-8"))
        self.token_ids = np.load(self.root / "token_ids.npy", mmap_mode="r")
        self.numeric_values = np.load(self.root / "numeric_values.npy", mmap_mode="r")
        self.numeric_mask = np.load(self.root / "numeric_mask.npy", mmap_mode="r")
        self.action_type_ids = np.load(self.root / "action_type_ids.npy", mmap_mode="r")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.dataset_indices = np.load(self.root / "dataset_indices.npy", mmap_mode="r")
        self.percentiles = np.load(self.root / "percentiles.npy", mmap_mode="r")
        self.validities = np.load(self.root / "validities.npy", mmap_mode="r")
        self.rmse = np.load(self.root / "rmse.npy", mmap_mode="r")
        self.splits = np.load(self.root / "splits.npy", mmap_mode="r")
        self.dataset_splits = np.load(self.root / "dataset_splits.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def sequence(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return (
            self.token_ids[start:end],
            self.numeric_values[start:end],
            self.numeric_mask[start:end],
            self.action_type_ids[start:end],
        )


class DatasetUniformBatchSampler:
    """Choose datasets uniformly, then unique semantic rows uniformly within each."""

    def __init__(
        self,
        corpus: TokenizedCorpus,
        *,
        split: int = 0,
        datasets_per_batch: int = 16,
        genomes_per_dataset: int = 16,
        seed: int = 0,
        steps: int | None = None,
        heldout_mode: bool = False,
    ) -> None:
        grouped: dict[int, list[int]] = defaultdict(list)
        for row, dataset in enumerate(corpus.dataset_indices):
            selected = (
                int(corpus.dataset_splits[row]) == split
                and int(corpus.splits[row]) == (0 if split == 0 else 2)
                if heldout_mode
                else int(corpus.splits[row]) == split
            )
            if selected:
                grouped[int(dataset)].append(row)
        self.rows = {dataset: np.asarray(rows, dtype=np.int64) for dataset, rows in grouped.items()}
        self.datasets = np.asarray(sorted(self.rows), dtype=np.int64)
        if not len(self.datasets):
            raise ValueError("dataset-uniform sampler has no eligible datasets")
        self.datasets_per_batch = min(datasets_per_batch, len(self.datasets))
        self.genomes_per_dataset = genomes_per_dataset
        self.seed = seed
        self.epoch = 0
        eligible = sum(len(rows) for rows in self.rows.values())
        nominal = max(1, self.datasets_per_batch * genomes_per_dataset)
        self.steps = steps or math.ceil(eligible / nominal)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            datasets = rng.choice(
                self.datasets, size=self.datasets_per_batch, replace=False
            )
            batch: list[int] = []
            for dataset in datasets:
                rows = self.rows[int(dataset)]
                count = min(self.genomes_per_dataset, len(rows))
                batch.extend(rng.choice(rows, size=count, replace=False).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
