"""Content-addressed mmap corpus and exact-once epoch scheduler."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import constrained_semantic_genome_splits
from .contracts import (
    BROAD_CORPUS_SCHEMA,
    DATASET_EMBEDDING_DIMENSIONS,
    TARGET_SUMMARY_DIMENSIONS,
    build_hashed,
    content_sha256,
    normalize_score_within_dataset,
    validate_broad_catalog,
    validate_broad_corpus_snapshot,
    validate_broad_receipt,
    validate_probe_slate,
)

PACK_FILES = {
    "token_ids": ("token_ids.npy", "uint16"),
    "offsets": ("offsets.npy", "uint64"),
    "dataset_embeddings": ("dataset_embeddings.npy", "float32"),
    "target_summaries": ("target_summaries.npy", "float32"),
    "anchor_features": ("anchor_features.npy", "float32"),
    "anchor_masks": ("anchor_masks.npy", "uint8"),
    "scores": ("scores.npy", "float32"),
    "validity": ("validity.npy", "uint8"),
    "runtime": ("runtime.npy", "float32"),
    "metadata": ("metadata.npy", "uint32"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_vector(root: Path, reference: Mapping[str, Any]) -> np.ndarray:
    path = root / str(reference["path"])
    count = int(reference["dimensions"])
    offset = int(reference["offset_bytes"])
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(count * 4)
    if len(payload) != count * 4:
        raise ValueError(f"encoding reference exceeds {path}")
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"encoding reference SHA-256 mismatch: {path}")
    values = np.frombuffer(payload, dtype="<f4").copy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"encoding reference contains nonfinite values: {path}")
    return values


def _anchor_signature(
    receipts: Sequence[Mapping[str, Any]],
    anchors: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in receipts:
        semantic = str(receipt["semantic_sha256"])
        if semantic not in anchors:
            continue
        key = (str(receipt["dataset_hash"]), semantic)
        prior = selected.get(key)
        if prior is None or (receipt["fidelity"] == "exact" and prior["fidelity"] != "exact"):
            selected[key] = receipt
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    datasets = sorted({dataset for dataset, _ in selected})
    for dataset in datasets:
        features = np.zeros((4, 3), dtype=np.float32)
        mask = np.zeros(4, dtype=np.uint8)
        observed_scores = [
            float(selected[(dataset, semantic)]["robust_score"])
            for semantic in anchors
            if (dataset, semantic) in selected and selected[(dataset, semantic)]["valid"]
        ]
        normalized = normalize_score_within_dataset(observed_scores)
        score_iter = iter(normalized)
        for index, semantic in enumerate(anchors):
            receipt = selected.get((dataset, semantic))
            if receipt is None:
                continue
            mask[index] = 1
            valid = bool(receipt["valid"])
            features[index, 0] = next(score_iter) if valid else 0.0
            runtime = receipt["runtime_seconds"]
            features[index, 1] = math.log1p(float(runtime)) if runtime is not None else 0.0
            features[index, 2] = float(valid)
        output[dataset] = features, mask
    return output


def materialize_broad_corpus(
    *,
    catalog: Mapping[str, Any],
    probe_slate: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    tokenized_genomes: Mapping[str, Sequence[int]],
    vocabulary_tokens: Sequence[str],
    vocabulary_sha256: str,
    catalog_root: str | Path,
    output_root: str | Path,
    created_at_ms: int,
    production: bool = False,
) -> Path:
    """Parse JSON once, then publish a fully mmap-backed immutable pack."""

    validate_broad_catalog(catalog, production=production)
    validate_probe_slate(probe_slate, production=production)
    rows = [dict(receipt) for receipt in receipts]
    for receipt in rows:
        validate_broad_receipt(receipt)
        if receipt["vocabulary_sha256"] != vocabulary_sha256:
            raise ValueError("receipt vocabulary differs from corpus vocabulary")
    if not rows:
        raise ValueError("broad corpus cannot be empty")
    if (
        not vocabulary_tokens
        or any(not isinstance(token, str) or not token for token in vocabulary_tokens)
        or len(vocabulary_tokens) > 65_536
    ):
        raise ValueError("broad vocabulary must contain 1..65536 nonempty tokens")
    rows.sort(
        key=lambda row: (
            row["dataset_hash"],
            row["fidelity"],
            row["semantic_sha256"],
            row["fold_plan_sha256"],
        )
    )
    catalog_by_dataset = {row["dataset_hash"]: row for row in catalog["datasets"]}
    for receipt in rows:
        catalog_row = catalog_by_dataset.get(receipt["dataset_hash"])
        if catalog_row is None:
            raise ValueError("receipt dataset is absent from BroadDatasetCatalog.v1")
        if catalog_row["train_sha256"] != receipt["train_sha256"]:
            raise ValueError("receipt train hash differs from catalog authority")
        tokens = tokenized_genomes.get(receipt["semantic_sha256"])
        if (
            tokens is None
            or not tokens
            or any(type(token) is not int or not 0 <= token <= 65_535 for token in tokens)
            or max(tokens) >= len(vocabulary_tokens)
        ):
            raise ValueError("broad corpus requires ordered uint16 tokens for every genome")

    binding = content_sha256(
        {
            "catalog_sha256": catalog["catalog_sha256"],
            "probe_slate_sha256": probe_slate["slate_sha256"],
            "vocabulary_sha256": vocabulary_sha256,
            "receipt_sha256": [row["receipt_sha256"] for row in rows],
        }
    )
    parent = Path(output_root).resolve() / "broad_corpora"
    final = parent / binding
    if final.exists():
        BroadCorpus(final)
        return final
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".build-{binding}-{os.getpid()}"
    temporary.mkdir()
    try:
        dataset_names = sorted({str(row["dataset_hash"]) for row in rows})
        dataset_index = {name: index for index, name in enumerate(dataset_names)}
        anchor_semantics = [str(anchor["semantic_sha256"]) for anchor in probe_slate["anchors"]]
        signatures = _anchor_signature(rows, anchor_semantics)
        total_tokens = sum(len(tokenized_genomes[row["semantic_sha256"]]) for row in rows)
        arrays = {
            "token_ids": np.lib.format.open_memmap(
                temporary / PACK_FILES["token_ids"][0],
                mode="w+",
                dtype="<u2",
                shape=(total_tokens,),
            ),
            "offsets": np.lib.format.open_memmap(
                temporary / PACK_FILES["offsets"][0],
                mode="w+",
                dtype="<u8",
                shape=(len(rows) + 1,),
            ),
            "dataset_embeddings": np.lib.format.open_memmap(
                temporary / PACK_FILES["dataset_embeddings"][0],
                mode="w+",
                dtype="<f4",
                shape=(len(rows), DATASET_EMBEDDING_DIMENSIONS),
            ),
            "target_summaries": np.lib.format.open_memmap(
                temporary / PACK_FILES["target_summaries"][0],
                mode="w+",
                dtype="<f4",
                shape=(len(rows), TARGET_SUMMARY_DIMENSIONS),
            ),
            "anchor_features": np.lib.format.open_memmap(
                temporary / PACK_FILES["anchor_features"][0],
                mode="w+",
                dtype="<f4",
                shape=(len(rows), 4, 3),
            ),
            "anchor_masks": np.lib.format.open_memmap(
                temporary / PACK_FILES["anchor_masks"][0],
                mode="w+",
                dtype="u1",
                shape=(len(rows), 4),
            ),
            "scores": np.lib.format.open_memmap(
                temporary / PACK_FILES["scores"][0],
                mode="w+",
                dtype="<f4",
                shape=(len(rows),),
            ),
            "validity": np.lib.format.open_memmap(
                temporary / PACK_FILES["validity"][0],
                mode="w+",
                dtype="u1",
                shape=(len(rows),),
            ),
            "runtime": np.lib.format.open_memmap(
                temporary / PACK_FILES["runtime"][0],
                mode="w+",
                dtype="<f4",
                shape=(len(rows),),
            ),
            # dataset index, domain, fidelity, dataset split, sequence length,
            # anchor owner (4 = none), global semantic-genome split.
            "metadata": np.lib.format.open_memmap(
                temporary / PACK_FILES["metadata"][0],
                mode="w+",
                dtype="<u4",
                shape=(len(rows), 7),
            ),
        }
        arrays["offsets"][0] = 0
        arrays["scores"].fill(np.nan)
        arrays["runtime"].fill(np.nan)
        vector_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_scores: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        cursor = 0
        domain_counts: Counter[str] = Counter()
        fidelity_counts: Counter[str] = Counter()
        split_codes = {
            "train": 0,
            "validation": 1,
            "test": 2,
            "cold_development": 3,
            "prospective_gate": 4,
            "double_holdout": 5,
        }
        semantic_splits = constrained_semantic_genome_splits(
            rows,
            seed=int(probe_slate["seed"]),
        )
        semantic_split_codes = {"train": 0, "validation": 1, "test": 2}
        for index, receipt in enumerate(rows):
            catalog_row = catalog_by_dataset[receipt["dataset_hash"]]
            dataset = str(receipt["dataset_hash"])
            if dataset not in vector_cache:
                vector_cache[dataset] = (
                    _read_vector(Path(catalog_root), catalog_row["dataset_encoding"]),
                    _read_vector(Path(catalog_root), catalog_row["target_encoding"]),
                )
            dataset_vector, target_vector = vector_cache[dataset]
            tokens = np.asarray(tokenized_genomes[receipt["semantic_sha256"]], dtype=np.uint16)
            end = cursor + len(tokens)
            arrays["token_ids"][cursor:end] = tokens
            arrays["offsets"][index + 1] = end
            arrays["dataset_embeddings"][index] = dataset_vector
            arrays["target_summaries"][index] = target_vector
            feature, observed = signatures.get(
                dataset,
                (
                    np.zeros((4, 3), dtype=np.float32),
                    np.zeros(4, dtype=np.uint8),
                ),
            )
            feature = feature.copy()
            observed = observed.copy()
            anchor_owner = (
                anchor_semantics.index(receipt["semantic_sha256"])
                if receipt["semantic_sha256"] in anchor_semantics
                else 4
            )
            if anchor_owner < 4:
                feature[anchor_owner] = 0
                observed[anchor_owner] = 0
            arrays["anchor_features"][index] = feature
            arrays["anchor_masks"][index] = observed
            arrays["validity"][index] = int(receipt["valid"])
            if receipt["runtime_seconds"] is not None:
                arrays["runtime"][index] = math.log1p(float(receipt["runtime_seconds"]))
            if receipt["valid"]:
                raw_scores[(dataset, receipt["fidelity"])].append((index, float(receipt["robust_score"])))
            domain = (
                "online"
                if str(receipt["selection_channel"]).startswith("online")
                else str(catalog_row["domain"])
            )
            domain_code = {"real": 0, "synthetic": 1, "online": 2}[domain]
            fidelity_code = {"screening": 0, "exact": 1}[receipt["fidelity"]]
            arrays["metadata"][index] = (
                dataset_index[dataset],
                domain_code,
                fidelity_code,
                split_codes[catalog_row["split"]],
                len(tokens),
                anchor_owner,
                semantic_split_codes[semantic_splits[receipt["semantic_sha256"]]],
            )
            domain_counts[domain] += 1
            fidelity_counts[receipt["fidelity"]] += 1
            cursor = end
        if cursor != total_tokens:
            raise AssertionError("broad pack token count changed")
        for values in raw_scores.values():
            normalized = normalize_score_within_dataset([value for _, value in values])
            for (index, _), score in zip(values, normalized, strict=True):
                arrays["scores"][index] = score
        for array in arrays.values():
            array.flush()
        del arrays

        file_contracts: dict[str, dict[str, Any]] = {}
        for logical, (filename, dtype) in PACK_FILES.items():
            path = temporary / filename
            array = np.load(path, mmap_mode="r")
            file_contracts[logical] = {
                "path": filename,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "dtype": dtype,
                "shape": list(array.shape),
            }
        snapshot = build_hashed(
            BROAD_CORPUS_SCHEMA,
            "snapshot_sha256",
            catalog_sha256=catalog["catalog_sha256"],
            probe_slate_sha256=probe_slate["slate_sha256"],
            vocabulary_sha256=vocabulary_sha256,
            created_at_ms=created_at_ms,
            row_count=len(rows),
            valid_row_count=sum(bool(row["valid"]) for row in rows),
            dataset_count=len(dataset_names),
            domain_counts={name: domain_counts[name] for name in ("real", "synthetic", "online")},
            fidelity_counts={name: fidelity_counts[name] for name in ("screening", "exact")},
            files=file_contracts,
        )
        validate_broad_corpus_snapshot(snapshot, production=production)
        _write_json_new(temporary / "manifest.v1.json", snapshot)
        _write_json_new(
            temporary / "index.v1.json",
            {
                "schema": "BroadCorpusIndex.v1",
                "binding_sha256": binding,
                "datasets": dataset_names,
                "vocabulary_tokens": list(vocabulary_tokens),
                "semantic_sha256": [row["semantic_sha256"] for row in rows],
                "receipt_sha256": [row["receipt_sha256"] for row in rows],
            },
        )
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        try:
            temporary.rename(final)
        except FileExistsError:
            shutil.rmtree(temporary)
        BroadCorpus(final)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class BroadCorpus:
    """Hot-path reader: initialization parses manifests; row access is mmap-only."""

    def __init__(self, root: str | Path, *, verify: bool = True) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.v1.json").read_text(encoding="utf-8"))
        validate_broad_corpus_snapshot(self.manifest)
        self.index = json.loads((self.root / "index.v1.json").read_text(encoding="utf-8"))
        if verify:
            for contract in self.manifest["files"].values():
                path = self.root / contract["path"]
                if sha256_file(path) != contract["sha256"]:
                    raise ValueError(f"broad corpus SHA-256 mismatch: {path}")
        for logical, (filename, _) in PACK_FILES.items():
            setattr(self, logical, np.load(self.root / filename, mmap_mode="r"))

    def __len__(self) -> int:
        return int(self.manifest["row_count"])

    def sequence(self, index: int) -> np.ndarray:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.token_ids[start:end]


class ExactOnceBatchSampler:
    """Round-robin shuffled dataset queues with randomized length buckets."""

    def __init__(
        self,
        corpus: BroadCorpus,
        *,
        batch_size: int,
        seed: int,
        eligible_rows: Sequence[int] | None = None,
        bucket_buffer: int = 2048,
        domain_mix: Mapping[str, float] | None = None,
    ) -> None:
        if batch_size < 1 or bucket_buffer < batch_size:
            raise ValueError("invalid exact-once batch or bucket-buffer size")
        selected = (
            np.arange(len(corpus), dtype=np.int64)
            if eligible_rows is None
            else np.asarray(eligible_rows, dtype=np.int64)
        )
        if len(selected) != len(set(selected.tolist())):
            raise ValueError("eligible rows contain duplicates")
        if np.any(selected < 0) or np.any(selected >= len(corpus)):
            raise ValueError("eligible row is outside corpus")
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row in selected.tolist():
            grouped[(int(corpus.metadata[row, 1]), int(corpus.metadata[row, 0]))].append(row)
        codes = {"real": 0, "synthetic": 1, "online": 2}
        mix = dict(domain_mix or {})
        if mix and (
            set(mix) - set(codes)
            or any(
                not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0
                for weight in mix.values()
            )
            or sum(float(weight) for weight in mix.values()) <= 0
        ):
            raise ValueError("domain_mix must contain nonnegative real/synthetic/online weights")
        self.corpus = corpus
        self.rows = grouped
        self.domain_mix = {codes[name]: float(weight) for name, weight in mix.items() if weight > 0}
        self.eligible = selected
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_buffer = bucket_buffer
        self.epoch = 0
        self.last_report: dict[str, Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.eligible) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        shuffle_seed = self.seed + self.epoch
        rng = np.random.default_rng(shuffle_seed)
        groups = list(self.rows)
        rng.shuffle(groups)
        queues: dict[tuple[int, int], deque[int]] = {}
        for group in groups:
            values = np.asarray(self.rows[group], dtype=np.int64)
            rng.shuffle(values)
            queues[group] = deque(values.tolist())
        active_by_domain: dict[int, deque[tuple[int, int]]] = defaultdict(deque)
        for group in groups:
            active_by_domain[group[0]].append(group)
        weights = self.domain_mix or {domain: 1.0 for domain in sorted(active_by_domain)}
        # Smooth weighted round-robin domains; each domain internally
        # round-robins shuffled dataset queues.
        credit = {domain: 0.0 for domain in set(active_by_domain) | set(weights)}
        stream: list[int] = []
        while any(active_by_domain.values()):
            available = [domain for domain, active in active_by_domain.items() if active]
            total_weight = sum(weights.get(domain, 0.0) for domain in available)
            if total_weight <= 0:
                total_weight = float(len(available))
                local_weights = {domain: 1.0 for domain in available}
            else:
                local_weights = {domain: weights.get(domain, 0.0) for domain in available}
            for domain in available:
                credit[domain] += local_weights[domain]
            domain = max(available, key=lambda value: (credit[value], -value))
            credit[domain] -= total_weight
            group = active_by_domain[domain].popleft()
            stream.append(queues[group].popleft())
            if queues[group]:
                active_by_domain[domain].append(group)

        batches: list[list[int]] = []
        for start in range(0, len(stream), self.bucket_buffer):
            buffer = stream[start : start + self.bucket_buffer]
            buffer.sort(key=lambda row: int(self.corpus.metadata[row, 4]))
            local = [
                buffer[index : index + self.batch_size] for index in range(0, len(buffer), self.batch_size)
            ]
            rng.shuffle(local)
            for batch in local:
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        flattened = [row for batch in batches for row in batch]
        unique = len(set(flattened))
        duplicates = len(flattened) - unique
        self.last_report = {
            "schema": "BroadEpochCoverage.v1",
            "epoch": self.epoch,
            "shuffle_seed": shuffle_seed,
            "eligible_rows": len(self.eligible),
            "visited_rows": len(flattened),
            "unique_rows": unique,
            "duplicate_rows": duplicates,
            "coverage": unique / max(len(self.eligible), 1),
            "domain_rows": {
                name: sum(int(self.corpus.metadata[row, 1]) == code for row in flattened)
                for name, code in (
                    ("real", 0),
                    ("synthetic", 1),
                    ("online", 2),
                )
            },
            "target_domain_mix": {
                name: self.domain_mix.get(code, 0.0)
                for name, code in (
                    ("real", 0),
                    ("synthetic", 1),
                    ("online", 2),
                )
            },
        }
        if (
            len(flattened) != len(self.eligible)
            or unique != len(self.eligible)
            or duplicates
            or set(flattened) != set(self.eligible.tolist())
        ):
            raise AssertionError("exact-once epoch coverage invariant failed")
        yield from batches
