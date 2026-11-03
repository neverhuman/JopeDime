from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .corpus import (
    PAD,
    DatasetUniformBatchSampler,
    TokenizedCorpus,
    sha256_file,
)

if TYPE_CHECKING:
    from .train import TrainConfig

AUTOTUNE_WORKERS = (0, 2, 4, 8)
AUTOTUNE_BATCH_SIZES = (256, 512, 1024, 2048)
MINIMUM_VRAM_HEADROOM_MIB = 4096.0
AUTOTUNE_WARMUP_STEPS = 2
AUTOTUNE_TIMED_STEPS = 4


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _update_suite_progress(
    config: TrainConfig,
    *,
    checkpoint_state: str,
    phase: str | None = None,
    quality_gate: str | None = None,
) -> None:
    if config.suite_dir is None:
        return
    config.suite_dir.mkdir(parents=True, exist_ok=True)
    active = os.path.relpath(config.run_dir, config.suite_dir)
    value: dict[str, Any] = {
        "schema": "jope-prime.dime-suite-progress.v1",
        "active_phase": phase or config.suite_phase or "training",
        "active_run": active,
        "checkpoint_state": checkpoint_state,
        "updated_unix_seconds": time.time(),
    }
    if quality_gate is not None:
        value["quality_gate"] = quality_gate
    _atomic_json(config.suite_dir / "suite-progress.v1.json", value)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_embedding_matrix(root: str | Path) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.v1.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "jope-prime.dime-embeddings.v1"
        or manifest.get("version") != "hybrid-v3.2"
        or manifest.get("shape") != [55, 1536]
    ):
        raise ValueError("unsupported Dime embedding manifest")
    path = root / str(manifest.get("embeddings_file"))
    if sha256_file(path) != manifest.get("embeddings_sha256"):
        raise ValueError("Dime embedding matrix SHA-256 mismatch")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (55, 1536) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Dime embedding matrix is malformed")
    identities = manifest.get("identities")
    if not isinstance(identities, list) or len(identities) != 55 or len(set(identities)) != 55:
        raise ValueError("Dime embedding identities are malformed")
    return matrix, {identity: index for index, identity in enumerate(identities)}, manifest


def _numeric_normalization(
    corpus: TokenizedCorpus, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    size = len(corpus.index["tokens"])
    mean = torch.zeros(size, dtype=torch.float32, device=device)
    scale = torch.ones(size, dtype=torch.float32, device=device)
    for token_id, stats in corpus.index.get("numeric_stats", {}).items():
        mean[int(token_id)] = float(stats["mean"])
        scale[int(token_id)] = float(stats["scale"])
    return mean, scale


def _collate_cpu(
    corpus: TokenizedCorpus,
    indexes: Iterable[int],
    embedding_matrix: np.ndarray,
    embedding_lookup: np.ndarray,
) -> dict[str, torch.Tensor]:
    indexes = list(indexes)
    sequences = [corpus.sequence(index) for index in indexes]
    lengths = torch.tensor([len(item[0]) for item in sequences], dtype=torch.long)
    width = int(lengths.max().item())
    batch = len(indexes)
    tokens = torch.full((batch, width), PAD, dtype=torch.long)
    numeric = torch.zeros((batch, width), dtype=torch.float64)
    numeric_mask = torch.zeros((batch, width), dtype=torch.float32)
    action_types = torch.zeros((batch, width), dtype=torch.long)
    for row, (token_ids, numeric_values, mask, type_ids) in enumerate(sequences):
        length = len(token_ids)
        tokens[row, :length] = torch.from_numpy(np.asarray(token_ids, dtype=np.int64).copy())
        numeric[row, :length] = torch.from_numpy(np.asarray(numeric_values, dtype=np.float64).copy())
        numeric_mask[row, :length] = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy())
        action_types[row, :length] = torch.from_numpy(np.asarray(type_ids, dtype=np.int64).copy())
    dataset_ids = np.asarray(corpus.dataset_indices[indexes], dtype=np.int64)
    embedding_rows = embedding_lookup[dataset_ids]
    if np.any(embedding_rows < 0):
        raise ValueError("corpus references a dataset absent from the embedding manifest")
    embeddings = torch.from_numpy(np.asarray(embedding_matrix[embedding_rows], dtype=np.float32).copy())
    return {
        "embedding": embeddings,
        "token_ids": tokens,
        "numeric_values": numeric.to(dtype=torch.float32),
        "numeric_mask": numeric_mask,
        "action_type_ids": action_types,
        "lengths": lengths,
        "percentiles": torch.from_numpy(np.asarray(corpus.percentiles[indexes], dtype=np.float32).copy()),
        "validities": torch.from_numpy(np.asarray(corpus.validities[indexes], dtype=np.float32).copy()),
        "dataset_indices": torch.from_numpy(dataset_ids.copy()),
        "row_indices": torch.tensor(indexes, dtype=torch.long),
    }


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in batch.items()}


def _collate(
    corpus: TokenizedCorpus,
    indexes: Iterable[int],
    embedding_matrix: np.ndarray,
    embedding_lookup: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return _move_batch(
        _collate_cpu(corpus, indexes, embedding_matrix, embedding_lookup),
        device,
        non_blocking=device.type == "cuda",
    )


class _CorpusRowIndexes(Dataset[int]):
    """DataLoader index target; the batch sampler owns all stochastic choices."""

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> int:
        return index


class _MmapCollator:
    """Open immutable mmap files lazily in each worker and build a CPU batch."""

    def __init__(
        self,
        corpus_root: Path,
        embeddings_path: Path,
        embedding_lookup: np.ndarray,
    ) -> None:
        self.corpus_root = corpus_root
        self.embeddings_path = embeddings_path
        self.embedding_lookup = embedding_lookup
        self._corpus: TokenizedCorpus | None = None
        self._embeddings: np.ndarray | None = None

    def __call__(self, indexes: list[int]) -> dict[str, torch.Tensor]:
        if self._corpus is None:
            # Revalidate the corpus when each worker opens its own lazy file handles.
            self._corpus = TokenizedCorpus(self.corpus_root, verify=True)
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r", allow_pickle=False)
        assert self._embeddings is not None
        return _collate_cpu(
            self._corpus,
            indexes,
            self._embeddings,
            self.embedding_lookup,
        )


def _loader_worker_init(_: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _make_data_loader(
    corpus: TokenizedCorpus,
    embeddings_path: Path,
    embedding_lookup: np.ndarray,
    sampler: DatasetUniformBatchSampler,
    *,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    seed: int = 0,
) -> DataLoader:
    if workers < 0 or workers > 8:
        raise ValueError("loader workers must be between 0 and 8")
    if prefetch_factor < 1:
        raise ValueError("prefetch factor must be positive")
    options: dict[str, Any] = {
        "dataset": _CorpusRowIndexes(len(corpus)),
        "batch_sampler": sampler,
        "collate_fn": _MmapCollator(
            corpus.root,
            embeddings_path,
            embedding_lookup,
        ),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "worker_init_fn": _loader_worker_init if workers > 0 else None,
        # Worker base seeds must not consume the model/dropout RNG stream.
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["multiprocessing_context"] = "spawn"
    return DataLoader(**options)


def _queue_depth(iterator: Any) -> int | None:
    queue = getattr(iterator, "_data_queue", None)
    if queue is None:
        return None
    try:
        return int(queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return None


class _BatchPrefetcher(Iterator[tuple[dict[str, torch.Tensor], dict[str, float | int | bool | None]]]):
    """Overlap pinned-memory H2D copies with model work on a dedicated stream."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.iterator = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.next_batch: dict[str, torch.Tensor] | None = None
        self.next_timing: dict[str, Any] | None = None
        self._preload()

    def __iter__(self) -> _BatchPrefetcher:
        return self

    def _preload(self) -> None:
        depth = _queue_depth(self.iterator)
        started = time.perf_counter()
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            self.next_timing = None
            return
        loader_wait_ms = (time.perf_counter() - started) * 1000.0
        timing: dict[str, Any] = {
            "loader_wait_ms": loader_wait_ms,
            "queue_depth": depth,
            "starved": bool(loader_wait_ms >= 2.0 and (depth is None or depth == 0)),
        }
        if self.stream is None:
            self.next_batch = cpu_batch
            timing["h2d_ms"] = 0.0
        else:
            started_event = torch.cuda.Event(enable_timing=True)
            finished_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                started_event.record(self.stream)
                self.next_batch = _move_batch(
                    cpu_batch,
                    self.device,
                    non_blocking=True,
                )
                finished_event.record(self.stream)
            timing["h2d_events"] = (started_event, finished_event)
        self.next_timing = timing

    def __next__(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | int | bool | None]]:
        if self.next_batch is None or self.next_timing is None:
            raise StopIteration
        batch, timing = self.next_batch, self.next_timing
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            for tensor in batch.values():
                tensor.record_stream(torch.cuda.current_stream(self.device))
            started, finished = timing.pop("h2d_events")
            finished.synchronize()
            timing["h2d_ms"] = float(started.elapsed_time(finished))
        self._preload()
        return batch, timing


def _shutdown_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _embedding_lookup(corpus: TokenizedCorpus, identities: dict[str, int]) -> np.ndarray:
    lookup = np.full(len(corpus.index["datasets"]), -1, dtype=np.int64)
    for index, identity in enumerate(corpus.index["datasets"]):
        if identity in identities:
            lookup[index] = identities[identity]
    missing = [corpus.index["datasets"][index] for index, value in enumerate(lookup) if value < 0]
    if missing:
        raise ValueError(f"score corpus datasets missing embeddings: {missing}")
    return lookup


def _validity_positive_weight(corpus: TokenizedCorpus, heldout: bool) -> float:
    mask = np.asarray(corpus.splits) == 0
    if heldout:
        mask &= np.asarray(corpus.dataset_splits) == 0
    labels = np.asarray(corpus.validities[mask], dtype=np.float64)
    positive = float(labels.sum())
    negative = float((1.0 - labels).sum())
    return max(0.05, min(20.0, negative / max(positive, 1e-8)))


def _available_loader_workers() -> tuple[int, ...]:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    maximum = min(8, max(0, cpus - 4))
    return tuple(value for value in AUTOTUNE_WORKERS if value <= maximum)
