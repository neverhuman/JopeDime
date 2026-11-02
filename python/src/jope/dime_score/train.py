from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .corpus import (
    PAD,
    DatasetUniformBatchSampler,
    TokenizedCorpus,
    prepare_corpus,
    sha256_file,
)
from .model import DimeScoreModel, partial_warm_start, score_losses


@dataclass
class TrainConfig:
    corpus_root: Path
    embeddings_root: Path
    run_dir: Path
    seed: int = 17
    epochs: int = 30
    patience: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    datasets_per_batch: int = 16
    genomes_per_dataset: int = 16
    device: str = "cuda"
    resume: bool = False
    warm_start: Path | None = None
    genome_only: bool = False
    heldout_diagnostic: bool = False
    compile_model: bool = True
    autotune: bool = True
    loader_workers: int | str = "auto"
    prefetch_factor: int = 4
    live_hz: float = 10.0
    hardware_hz: float = 1.0
    suite_dir: Path | None = None
    suite_phase: str | None = None
    run_kind: str = "fit"
    stop_after_epoch: int | None = None
    checkpoint_every: int = 0


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


def select_autotune_candidate(
    records: list[dict[str, Any]],
    *,
    minimum_headroom_mib: float = MINIMUM_VRAM_HEADROOM_MIB,
) -> dict[str, Any]:
    eligible = [
        row
        for row in records
        if row.get("status") == "ok"
        and isinstance(row.get("samples_per_second"), (int, float))
        and math.isfinite(float(row["samples_per_second"]))
        and float(row.get("free_vram_mib", 0.0)) >= minimum_headroom_mib
    ]
    if not eligible:
        raise RuntimeError("no autotuned configuration retained the required 4 GiB VRAM headroom")
    return max(
        eligible,
        key=lambda row: (
            float(row["samples_per_second"]),
            -int(row.get("loader_workers", 0)),
            -int(row.get("actual_examples", 0)),
        ),
    )


def _timed_training_step(
    forward: Any,
    model: DimeScoreModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    config: TrainConfig,
    validity_positive_weight: float,
) -> tuple[dict[str, torch.Tensor], float, float, float]:
    optimizer.zero_grad(set_to_none=True)
    cuda = batch["token_ids"].device.type == "cuda"
    compute_started = torch.cuda.Event(enable_timing=True) if cuda else time.perf_counter()
    compute_finished = torch.cuda.Event(enable_timing=True) if cuda else None
    if cuda:
        compute_started.record()
    with torch.autocast(
        device_type=batch["token_ids"].device.type,
        dtype=torch.bfloat16,
        enabled=cuda,
    ):
        output = forward(
            batch["embedding"],
            batch["token_ids"],
            batch["numeric_values"],
            batch["numeric_mask"],
            batch["action_type_ids"],
            batch["lengths"],
            numeric_mean,
            numeric_scale,
            genome_only=config.genome_only,
        )
        losses = score_losses(
            output,
            batch["percentiles"],
            batch["validities"],
            batch["dataset_indices"],
            validity_positive_weight=validity_positive_weight,
        )
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("Dime score training produced a non-finite loss")
    losses["total"].backward()
    if cuda:
        assert isinstance(compute_finished, torch.cuda.Event)
        compute_finished.record()
        optimizer_started: torch.cuda.Event | float = torch.cuda.Event(enable_timing=True)
        optimizer_finished: torch.cuda.Event | None = torch.cuda.Event(enable_timing=True)
        optimizer_started.record()
    else:
        optimizer_started = time.perf_counter()
        optimizer_finished = None
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if cuda:
        assert isinstance(optimizer_finished, torch.cuda.Event)
        optimizer_finished.record()
        optimizer_finished.synchronize()
        compute_ms = float(compute_started.elapsed_time(compute_finished))
        optimizer_ms = float(optimizer_started.elapsed_time(optimizer_finished))
    else:
        compute_ms = (float(optimizer_started) - float(compute_started)) * 1000.0
        optimizer_ms = (time.perf_counter() - float(optimizer_started)) * 1000.0
    return losses, float(gradient_norm), compute_ms, optimizer_ms


def _benchmark_configuration(
    model: DimeScoreModel,
    corpus: TokenizedCorpus,
    embeddings_path: Path,
    embedding_lookup: np.ndarray,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    device: torch.device,
    config: TrainConfig,
    *,
    workers: int,
    requested_examples: int,
    forward: Any,
    stage: str,
    mode: str,
) -> dict[str, Any]:
    from jope.telemetry import gpu_snapshot

    eligible_datasets = len(
        DatasetUniformBatchSampler(
            corpus,
            datasets_per_batch=config.datasets_per_batch,
            genomes_per_dataset=1,
            seed=config.seed,
            steps=1,
            heldout_mode=config.heldout_diagnostic,
        ).datasets
    )
    datasets = min(config.datasets_per_batch, eligible_datasets)
    genomes = max(1, requested_examples // datasets)
    sampler = DatasetUniformBatchSampler(
        corpus,
        datasets_per_batch=datasets,
        genomes_per_dataset=genomes,
        seed=config.seed,
        steps=AUTOTUNE_WARMUP_STEPS + AUTOTUNE_TIMED_STEPS,
        heldout_mode=config.heldout_diagnostic,
    )
    loader: DataLoader | None = None
    base = {
        "stage": stage,
        "mode": mode,
        "requested_examples": requested_examples,
        "datasets": datasets,
        "genomes_per_dataset": genomes,
        "loader_workers": workers,
        "warmup_steps": AUTOTUNE_WARMUP_STEPS,
        "timed_steps": AUTOTUNE_TIMED_STEPS,
    }
    try:
        loader = _make_data_loader(
            corpus,
            embeddings_path,
            embedding_lookup,
            sampler,
            workers=workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=True,
        )
        prefetcher = _BatchPrefetcher(loader, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples = 0
        elapsed = 0.0
        loader_wait = h2d = compute = optimizer_ms = 0.0
        starvations = 0
        queue_depths: list[int] = []
        actual_examples = 0
        for step in range(AUTOTUNE_WARMUP_STEPS + AUTOTUNE_TIMED_STEPS):
            started = time.perf_counter()
            batch, loading = next(prefetcher)
            _, _, compute_step, optimizer_step = _timed_training_step(
                forward,
                model,
                optimizer,
                batch,
                numeric_mean,
                numeric_scale,
                config,
                1.0,
            )
            if step < AUTOTUNE_WARMUP_STEPS:
                continue
            step_elapsed = time.perf_counter() - started
            actual_examples = len(batch["row_indices"])
            samples += actual_examples
            elapsed += step_elapsed
            loader_wait += float(loading["loader_wait_ms"])
            h2d += float(loading["h2d_ms"])
            compute += compute_step
            optimizer_ms += optimizer_step
            starvations += int(bool(loading["starved"]))
            if loading["queue_depth"] is not None:
                queue_depths.append(int(loading["queue_depth"]))
        free_bytes, _ = torch.cuda.mem_get_info()
        free_mib = free_bytes / (1024 * 1024)
        record = {
            **base,
            "status": ("ok" if free_mib >= MINIMUM_VRAM_HEADROOM_MIB else "insufficient_headroom"),
            "actual_examples": actual_examples,
            "samples_per_second": samples / max(elapsed, 1e-9),
            "loader_wait_ms": loader_wait / AUTOTUNE_TIMED_STEPS,
            "loader_wait_pct": 100.0 * loader_wait / max(elapsed * 1000.0, 1e-9),
            "h2d_ms": h2d / AUTOTUNE_TIMED_STEPS,
            "compute_ms": compute / AUTOTUNE_TIMED_STEPS,
            "optimizer_ms": optimizer_ms / AUTOTUNE_TIMED_STEPS,
            "queue_starvations": starvations,
            "mean_queue_depth": (float(np.mean(queue_depths)) if queue_depths else None),
            "free_vram_mib": free_mib,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
            **gpu_snapshot(MINIMUM_VRAM_HEADROOM_MIB),
        }
        return record
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {**base, "status": "out_of_memory"}
    except Exception as error:
        return {
            **base,
            "status": "compiler_failed" if mode == "compiled" else "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        model.zero_grad(set_to_none=True)
        if loader is not None:
            _shutdown_loader(loader)


def _autotune_training(
    model: DimeScoreModel,
    corpus: TokenizedCorpus,
    embeddings_path: Path,
    embedding_lookup: np.ndarray,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    device: torch.device,
    config: TrainConfig,
) -> tuple[int, int, Any, dict[str, Any]]:
    if device.type != "cuda" or not config.autotune:
        workers = 0 if config.loader_workers == "auto" else int(config.loader_workers)
        forward = torch.compile(model) if config.compile_model and device.type == "cuda" else model
        return config.genomes_per_dataset, workers, forward, {}

    if config.loader_workers == "auto":
        worker_candidates = _available_loader_workers()
    else:
        workers = int(config.loader_workers)
        if workers < 0 or workers > 8:
            raise ValueError("loader workers must be between 0 and 8")
        worker_candidates = (workers,)
    if not worker_candidates:
        worker_candidates = (0,)

    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    records: list[dict[str, Any]] = []
    try:
        for workers in worker_candidates:
            model.load_state_dict(state)
            records.append(
                _benchmark_configuration(
                    model,
                    corpus,
                    embeddings_path,
                    embedding_lookup,
                    numeric_mean,
                    numeric_scale,
                    device,
                    config,
                    workers=workers,
                    requested_examples=AUTOTUNE_BATCH_SIZES[0],
                    forward=model,
                    stage="loader_workers",
                    mode="eager",
                )
            )
        selected_worker = int(select_autotune_candidate(records)["loader_workers"])
        for requested_examples in AUTOTUNE_BATCH_SIZES[1:]:
            model.load_state_dict(state)
            records.append(
                _benchmark_configuration(
                    model,
                    corpus,
                    embeddings_path,
                    embedding_lookup,
                    numeric_mean,
                    numeric_scale,
                    device,
                    config,
                    workers=selected_worker,
                    requested_examples=requested_examples,
                    forward=model,
                    stage="batch_size",
                    mode="eager",
                )
            )
        eager_selected = select_autotune_candidate(records)
        selected = eager_selected
        forward: Any = model
        compile_decision: dict[str, Any] = {
            "requested": config.compile_model,
            "selected": "eager",
            "reason": "compilation disabled",
        }
        if config.compile_model:
            model.load_state_dict(state)
            try:
                compiled = torch.compile(model)
                compiled_record = _benchmark_configuration(
                    model,
                    corpus,
                    embeddings_path,
                    embedding_lookup,
                    numeric_mean,
                    numeric_scale,
                    device,
                    config,
                    workers=int(eager_selected["loader_workers"]),
                    requested_examples=int(eager_selected["requested_examples"]),
                    forward=compiled,
                    stage="compiler",
                    mode="compiled",
                )
            except Exception as error:
                compiled_record = {
                    "stage": "compiler",
                    "mode": "compiled",
                    "status": "compiler_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                compiled = None
            records.append(compiled_record)
            if (
                compiled is not None
                and compiled_record.get("status") == "ok"
                and float(compiled_record["samples_per_second"]) > float(eager_selected["samples_per_second"])
            ):
                selected = compiled_record
                forward = compiled
                compile_decision = {
                    "requested": True,
                    "selected": "compiled",
                    "reason": "higher measured samples_per_second",
                }
            else:
                compile_decision = {
                    "requested": True,
                    "selected": "eager",
                    "reason": (
                        "compilation failed"
                        if compiled_record.get("status") == "compiler_failed"
                        else "compiled throughput did not exceed eager"
                    ),
                }
        for row in records:
            row["selected"] = row is selected
        decision = {
            "schema": "jope-prime.dime-batch-autotune.v2",
            "selection_rule": "maximum measured samples_per_second",
            "minimum_vram_headroom_mib": MINIMUM_VRAM_HEADROOM_MIB,
            "worker_candidates": list(worker_candidates),
            "batch_size_candidates": list(AUTOTUNE_BATCH_SIZES),
            "warmup_steps": AUTOTUNE_WARMUP_STEPS,
            "timed_steps": AUTOTUNE_TIMED_STEPS,
            "candidates": records,
            "compiler": compile_decision,
            "selected": {
                "loader_workers": int(selected["loader_workers"]),
                "requested_examples": int(selected["requested_examples"]),
                "actual_examples": int(selected["actual_examples"]),
                "datasets_per_batch": int(selected["datasets"]),
                "genomes_per_dataset": int(selected["genomes_per_dataset"]),
                "samples_per_second": float(selected["samples_per_second"]),
                "free_vram_mib": float(selected["free_vram_mib"]),
                "peak_vram_mib": float(selected["peak_vram_mib"]),
                "mode": str(selected["mode"]),
            },
        }
        return (
            int(selected["genomes_per_dataset"]),
            int(selected["loader_workers"]),
            forward,
            decision,
        )
    finally:
        model.load_state_dict(state)
        torch.random.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 2:
        return math.nan
    left, right = _rankdata(predictions), _rankdata(targets)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    count_positive, count_negative = int(positive.sum()), int((~positive).sum())
    if not count_positive or not count_negative:
        return math.nan
    ranks = _rankdata(scores) + 1.0
    return float(
        (ranks[positive].sum() - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    if not positive.any():
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered = positive[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def quality_metrics(
    scores: np.ndarray,
    validities: np.ndarray,
    targets: np.ndarray,
    validity_targets: np.ndarray,
    datasets: np.ndarray,
    train_constant: float,
) -> dict[str, float]:
    valid = np.isfinite(targets)
    per_dataset_spearman: list[float] = []
    pair_correct = pair_total = 0
    top_precision: list[float] = []
    top_regret: list[float] = []
    for dataset in np.unique(datasets):
        indexes = np.flatnonzero((datasets == dataset) & valid)
        if len(indexes) < 2:
            continue
        per_dataset_spearman.append(_spearman(scores[indexes], targets[indexes]))
        left, right = np.triu_indices(len(indexes), 1)
        target_delta = targets[indexes[left]] - targets[indexes[right]]
        keep = target_delta != 0
        prediction_delta = scores[indexes[left]] - scores[indexes[right]]
        pair_correct += int(np.sum(np.sign(prediction_delta[keep]) == np.sign(target_delta[keep])))
        pair_total += int(keep.sum())
        count = max(1, math.ceil(0.1 * len(indexes)))
        predicted_top = indexes[np.argsort(-scores[indexes], kind="mergesort")[:count]]
        actual_top = set(indexes[np.argsort(-targets[indexes], kind="mergesort")[:count]].tolist())
        top_precision.append(sum(int(index in actual_top) for index in predicted_top) / count)
        top_regret.append(float(targets[indexes].max() - targets[predicted_top].max()))
    mae = float(np.mean(np.abs(scores[valid] - targets[valid]))) if valid.any() else math.nan
    constant_mae = float(np.mean(np.abs(train_constant - targets[valid]))) if valid.any() else math.nan
    bins = np.minimum((scores[valid] * 10).astype(int), 9) if valid.any() else np.array([])
    calibration = []
    for bin_index in range(10):
        selected = bins == bin_index
        if np.any(selected):
            calibration.append(abs(float(scores[valid][selected].mean() - targets[valid][selected].mean())))
    labels = (validity_targets >= 1.0 - 1e-12).astype(np.float64)
    return {
        "mean_per_dataset_spearman": float(np.nanmean(per_dataset_spearman))
        if per_dataset_spearman
        else math.nan,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else math.nan,
        "mae": mae,
        "constant_percentile_mae": constant_mae,
        "calibration_mae": float(np.mean(calibration)) if calibration else math.nan,
        "top_decile_precision": float(np.mean(top_precision)) if top_precision else math.nan,
        "top_decile_regret": float(np.mean(top_regret)) if top_regret else math.nan,
        "validity_auroc": _binary_auc(labels, validities),
        "validity_pr_auc": _average_precision(labels, validities),
    }


@torch.no_grad()
def evaluate(
    model: DimeScoreModel,
    corpus: TokenizedCorpus,
    embedding_matrix: np.ndarray,
    embedding_lookup: np.ndarray,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    device: torch.device,
    *,
    split: int,
    genome_only: bool,
    heldout: bool,
    train_constant: float,
    batch_size: int = 512,
) -> dict[str, Any]:
    if heldout:
        indexes = np.flatnonzero(
            (np.asarray(corpus.dataset_splits) == split) & (np.asarray(corpus.splits) == 2)
        )
    else:
        indexes = np.flatnonzero(np.asarray(corpus.splits) == split)
    model.eval()
    scores: list[np.ndarray] = []
    score_logits: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    validity_logits: list[np.ndarray] = []
    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size].tolist()
        batch = _collate(corpus, batch_indexes, embedding_matrix, embedding_lookup, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["embedding"],
                batch["token_ids"],
                batch["numeric_values"],
                batch["numeric_mask"],
                batch["action_type_ids"],
                batch["lengths"],
                numeric_mean,
                numeric_scale,
                genome_only=genome_only,
            )
        scores.append(output["score"].float().cpu().numpy())
        score_logits.append(output["score_logit"].float().cpu().numpy())
        validities.append(output["validity"].float().cpu().numpy())
        validity_logits.append(output["validity_logit"].float().cpu().numpy())
    if not scores:
        raise ValueError("validation split has no examples")
    score_values = np.concatenate(scores)
    score_logit_values = np.concatenate(score_logits)
    validity_values = np.concatenate(validities)
    validity_logit_values = np.concatenate(validity_logits)
    targets = np.asarray(corpus.percentiles[indexes], dtype=np.float64)
    validity_targets = np.asarray(corpus.validities[indexes], dtype=np.float64)
    datasets = np.asarray(corpus.dataset_indices[indexes], dtype=np.int64)
    metrics: dict[str, Any] = quality_metrics(
        score_values,
        validity_values,
        targets,
        validity_targets,
        datasets,
        train_constant,
    )
    losses = score_losses(
        {
            "score": torch.from_numpy(score_values),
            "score_logit": torch.from_numpy(score_logit_values),
            "validity_logit": torch.from_numpy(validity_logit_values),
        },
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(validity_targets.astype(np.float32)),
        torch.from_numpy(datasets),
        validity_positive_weight=_validity_positive_weight(corpus, heldout),
    )
    metrics["losses"] = {name: float(value) for name, value in losses.items()}
    metrics["examples"] = int(len(indexes))
    metrics["split_name"] = "dataset-held-out test" if heldout else ("train", "validation", "test")[split]
    return metrics


def _checkpoint(
    model: DimeScoreModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    corpus: TokenizedCorpus,
    embedding_manifest: dict[str, Any],
    config: TrainConfig,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    warm_start_report: dict[str, Any] | None,
    *,
    genomes_per_dataset: int,
    loader_workers: int,
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "model_version": "dime-score-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "tokens": corpus.index["tokens"],
        "action_types": corpus.index["action_types"],
        "numeric_stats": corpus.index["numeric_stats"],
        "grammar_contract_sha256": corpus.index["gene_schema_contract_sha256"],
        "tokenized_corpus_binding_sha256": corpus.manifest["binding_sha256"],
        "embedding_sha256": embedding_manifest["embeddings_sha256"],
        "embedding_frozen": embedding_manifest["frozen"],
        "architecture": {
            "dataset_dim": model.dataset_dim,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "ffn_dim": model.ffn_dim,
            "max_sequence_length": model.max_sequence_length,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "warm_start_report": warm_start_report,
        "data_loading": {
            "genomes_per_dataset": genomes_per_dataset,
            "loader_workers": loader_workers,
            "prefetch_factor": config.prefetch_factor,
            "execution_mode": execution_mode,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _restore_rng(checkpoint: dict[str, Any]) -> None:
    rng = checkpoint.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.random.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def warm_start_preflight(config: TrainConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "jope-prime.dime-warm-start-preflight.v1",
        "checkpoint": str(config.warm_start) if config.warm_start else None,
        "eligible": False,
        "reason": "no warm-start checkpoint supplied",
    }
    if config.warm_start is not None:
        tokenized_root = prepare_corpus(config.corpus_root)
        corpus = TokenizedCorpus(tokenized_root)
        model = DimeScoreModel(
            vocabulary_size=len(corpus.index["tokens"]),
            action_type_count=len(corpus.index["action_types"]),
            max_sequence_length=int(corpus.manifest["max_sequence_length"]),
        )
        warm = torch.load(config.warm_start, map_location="cpu", weights_only=False)
        mapping = partial_warm_start(model, warm, corpus.index["tokens"])
        mapped = int(mapping.get("loaded_tensor_count", 0))
        report.update(
            {
                "mapping": mapping,
                "eligible": mapped > 0,
                "reason": (
                    f"{mapped} exact-shape tensors mapped" if mapped else "zero exact-shape tensors mapped"
                ),
            }
        )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(config.run_dir / "warm-start-preflight.v1.json", report)
    return report


def _train_dime_score(config: TrainConfig) -> dict[str, Any]:
    if config.prefetch_factor < 1:
        raise ValueError("prefetch factor must be positive")
    if config.live_hz <= 0 or config.hardware_hz <= 0:
        raise ValueError("telemetry frequencies must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenized_root = prepare_corpus(config.corpus_root)
    corpus = TokenizedCorpus(tokenized_root)
    embedding_matrix, identities, embedding_manifest = load_embedding_matrix(config.embeddings_root)
    embedding_lookup = _embedding_lookup(corpus, identities)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": "jope-prime.dime-score-run.v1",
        "corpus_archive_sha256": corpus.manifest["source_archive_sha256"],
        "tokenized_corpus_binding_sha256": corpus.manifest["binding_sha256"],
        "embedding_sha256": embedding_manifest["embeddings_sha256"],
        "embedding_frozen": embedding_manifest["frozen"],
        "grammar_contract_sha256": corpus.index["gene_schema_contract_sha256"],
        "initialization": "warm" if config.warm_start else "scratch",
        "seed": config.seed,
        "genome_only": config.genome_only,
        "heldout_diagnostic": config.heldout_diagnostic,
        "run_kind": config.run_kind,
        "row_split": corpus.index["row_split_contract"],
        "batch_randomization": {
            "strategy": "dataset-uniform-without-replacement",
            "epoch_reseeded": True,
            "seed": config.seed,
            "batch_order_shuffled": True,
        },
        "binary_packing": {
            "format": "numpy-mmap-flat-offsets-v1",
            "tokens": int(corpus.manifest["tokens"]),
            "max_sequence_length": int(corpus.manifest["max_sequence_length"]),
        },
    }
    identity_path = config.run_dir / "run.v1.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise ValueError("run directory identity differs from requested training run")
    else:
        _atomic_json(identity_path, run_identity)

    model = DimeScoreModel(
        vocabulary_size=len(corpus.index["tokens"]),
        action_type_count=len(corpus.index["action_types"]),
        max_sequence_length=int(corpus.manifest["max_sequence_length"]),
    ).to(device)
    numeric_mean, numeric_scale = _numeric_normalization(corpus, device)
    warm_start_report = None
    last_path = config.run_dir / "last.pt"
    start_epoch = 0
    best_metric = -math.inf
    best_epoch = -1
    checkpoint = None
    if config.resume:
        if not last_path.is_file():
            raise ValueError(f"resume requested but checkpoint is absent: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("tokenized_corpus_binding_sha256") != corpus.manifest["binding_sha256"]:
            raise ValueError("resume checkpoint corpus binding mismatch")
        if checkpoint.get("embedding_sha256") != embedding_manifest["embeddings_sha256"]:
            raise ValueError("resume checkpoint embedding binding mismatch")
        model.load_state_dict(checkpoint["model"])
        warm_start_report = checkpoint.get("warm_start_report")
    elif config.warm_start:
        warm = torch.load(config.warm_start, map_location="cpu", weights_only=False)
        warm_start_report = partial_warm_start(model, warm, corpus.index["tokens"])

    embeddings_path = config.embeddings_root / str(embedding_manifest["embeddings_file"])
    if checkpoint is not None:
        loading = checkpoint.get("data_loading") or {}
        genomes_per_dataset = int(loading.get("genomes_per_dataset", config.genomes_per_dataset))
        loader_workers = int(
            loading.get(
                "loader_workers",
                0 if config.loader_workers == "auto" else config.loader_workers,
            )
        )
        execution_mode = str(loading.get("execution_mode", "eager"))
        compiled_forward: Any = (
            torch.compile(model)
            if execution_mode == "compiled" and config.compile_model and device.type == "cuda"
            else model
        )
    else:
        _update_suite_progress(config, checkpoint_state="autotuning")
        _atomic_json(
            config.run_dir / "progress.v1.json",
            {
                "schema": "jope-prime.dime-score-progress.v1",
                "checkpoint_state": "autotuning",
            },
        )
        (
            genomes_per_dataset,
            loader_workers,
            compiled_forward,
            autotune_decision,
        ) = _autotune_training(
            model,
            corpus,
            embeddings_path,
            embedding_lookup,
            numeric_mean,
            numeric_scale,
            device,
            config,
        )
        execution_mode = (
            str(autotune_decision["selected"]["mode"])
            if autotune_decision
            else ("compiled" if compiled_forward is not model and device.type == "cuda" else "eager")
        )
        if autotune_decision:
            _atomic_json(
                config.run_dir / "batch-autotune.v2.json",
                autotune_decision,
            )

    steps_per_epoch = DatasetUniformBatchSampler(
        corpus,
        split=0,
        datasets_per_batch=config.datasets_per_batch,
        genomes_per_dataset=genomes_per_dataset,
        seed=config.seed,
        heldout_mode=config.heldout_diagnostic,
    ).steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    total_steps = max(1, config.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        _restore_rng(checkpoint)

    train_mask = np.asarray(corpus.splits) == 0
    if config.heldout_diagnostic:
        train_mask &= np.asarray(corpus.dataset_splits) == 0
    train_targets = np.asarray(corpus.percentiles[train_mask], dtype=np.float64)
    train_constant = float(np.nanmean(train_targets))
    positive_weight = _validity_positive_weight(corpus, config.heldout_diagnostic)
    metrics_path = config.run_dir / "metrics.jsonl"
    progress_path = config.run_dir / "progress.v1.json"
    stopped_early = False
    interrupted = False
    sampler = DatasetUniformBatchSampler(
        corpus,
        split=0,
        datasets_per_batch=config.datasets_per_batch,
        genomes_per_dataset=genomes_per_dataset,
        seed=config.seed,
        heldout_mode=config.heldout_diagnostic,
    )
    loader = _make_data_loader(
        corpus,
        embeddings_path,
        embedding_lookup,
        sampler,
        workers=loader_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=device.type == "cuda",
        seed=config.seed,
    )
    from jope.telemetry import HardwareMonitor, LiveMetricsWriter

    hardware = HardwareMonitor(config.hardware_hz, MINIMUM_VRAM_HEADROOM_MIB)
    sampler.set_epoch(start_epoch)
    prefetcher = _BatchPrefetcher(loader, device)
    hardware.start()
    live = LiveMetricsWriter(config.run_dir / "live-metrics.jsonl", config.live_hz)
    _update_suite_progress(config, checkpoint_state="training")
    global_step = start_epoch * len(sampler)
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            if epoch != start_epoch:
                sampler.set_epoch(epoch)
                prefetcher = _BatchPrefetcher(loader, device)
            epoch_started = time.monotonic()
            samples = 0
            totals = {
                "total": 0.0,
                "huber": 0.0,
                "ranking": 0.0,
                "validity_bce": 0.0,
            }
            timing_totals = {
                "loader_wait_ms": 0.0,
                "h2d_ms": 0.0,
                "compute_ms": 0.0,
                "optimizer_ms": 0.0,
            }
            starvations = 0
            last_progress = 0.0
            for step in range(len(sampler)):
                step_started = time.monotonic()
                batch, loading = next(prefetcher)
                losses, gradient_norm, compute_ms, optimizer_ms = _timed_training_step(
                    compiled_forward,
                    model,
                    optimizer,
                    batch,
                    numeric_mean,
                    numeric_scale,
                    config,
                    positive_weight,
                )
                scheduler.step()
                batch_samples = len(batch["row_indices"])
                samples += batch_samples
                global_step += 1
                loss_values = {name: float(losses[name].detach()) for name in totals}
                for name, value in loss_values.items():
                    totals[name] += value
                step_timing = {
                    "loader_wait_ms": float(loading["loader_wait_ms"]),
                    "h2d_ms": float(loading["h2d_ms"]),
                    "compute_ms": compute_ms,
                    "optimizer_ms": optimizer_ms,
                }
                for name, value in step_timing.items():
                    timing_totals[name] += value
                starvations += int(bool(loading["starved"]))
                now = time.monotonic()
                elapsed = max(now - epoch_started, 1e-9)
                eta_seconds = (
                    (config.epochs - epoch - (step + 1) / len(sampler))
                    * elapsed
                    / max((step + 1) / len(sampler), 1e-9)
                )
                live.add(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "steps": len(sampler),
                        "global_step": global_step,
                        "batch_samples": batch_samples,
                        "batch_shape": [
                            sampler.datasets_per_batch,
                            genomes_per_dataset,
                        ],
                        "total_loss": loss_values["total"],
                        "huber_loss": loss_values["huber"],
                        "bradley_terry_loss": loss_values["ranking"],
                        "validity_loss": loss_values["validity_bce"],
                        **step_timing,
                        "gradient_norm": gradient_norm,
                        "queue_depth": loading["queue_depth"],
                        "starved": loading["starved"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "eta_seconds": eta_seconds,
                        "step_wall_ms": (now - step_started) * 1000.0,
                    }
                )
                if now - last_progress >= 1.0 / config.hardware_hz:
                    progress = {
                        "schema": "jope-prime.dime-score-progress.v1",
                        "epoch": epoch,
                        "epochs": config.epochs,
                        "batch": step + 1,
                        "batches": len(sampler),
                        "samples": samples,
                        "samples_per_second": samples / elapsed,
                        "loader_wait_pct": 100.0
                        * timing_totals["loader_wait_ms"]
                        / max(elapsed * 1000.0, 1e-9),
                        "h2d_ms": timing_totals["h2d_ms"] / (step + 1),
                        "compute_ms": timing_totals["compute_ms"] / (step + 1),
                        "optimizer_ms": timing_totals["optimizer_ms"] / (step + 1),
                        "queue_depth": loading["queue_depth"],
                        "queue_starvations": starvations,
                        "batch_shape": [
                            sampler.datasets_per_batch,
                            genomes_per_dataset,
                        ],
                        "loader_workers": loader_workers,
                        "prefetch_factor": config.prefetch_factor,
                        "execution_mode": execution_mode,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": gradient_norm,
                        "checkpoint_state": "training",
                        "eta_seconds": eta_seconds,
                        **hardware.snapshot(),
                    }
                    _atomic_json(progress_path, progress)
                    last_progress = now
            elapsed = max(time.monotonic() - epoch_started, 1e-6)
            live.flush()
            _atomic_json(
                progress_path,
                {
                    "schema": "jope-prime.dime-score-progress.v1",
                    "epoch": epoch,
                    "epochs": config.epochs,
                    "batch": len(sampler),
                    "batches": len(sampler),
                    "samples": samples,
                    "batch_shape": [
                        sampler.datasets_per_batch,
                        genomes_per_dataset,
                    ],
                    "checkpoint_state": "validating",
                    **hardware.snapshot(),
                },
            )
            validation_split = 2 if config.heldout_diagnostic else 1
            validation = evaluate(
                model,
                corpus,
                embedding_matrix,
                embedding_lookup,
                numeric_mean,
                numeric_scale,
                device,
                split=validation_split,
                genome_only=config.genome_only,
                heldout=config.heldout_diagnostic,
                train_constant=train_constant,
            )
            epoch_metrics = {
                "schema": "jope-prime.dime-score-metrics.v1",
                "epoch": epoch,
                "epochs": config.epochs,
                "train_losses": {name: total / len(sampler) for name, total in totals.items()},
                "validation": validation,
                "samples_per_second": samples / elapsed,
                "samples": samples,
                "epoch_seconds": elapsed,
                "timing": {name: value / len(sampler) for name, value in timing_totals.items()},
                "queue_starvations": starvations,
                "batch_shape": [
                    sampler.datasets_per_batch,
                    genomes_per_dataset,
                ],
                "loader_workers": loader_workers,
                "execution_mode": execution_mode,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "initialization": run_identity["initialization"],
                "genome_only": config.genome_only,
                "heldout_diagnostic": config.heldout_diagnostic,
                **hardware.snapshot(),
            }
            _append_jsonl(metrics_path, epoch_metrics)
            metric = float(validation["mean_per_dataset_spearman"])
            improved = math.isfinite(metric) and metric > best_metric + 1e-8
            if improved:
                best_metric = metric
                best_epoch = epoch
            saved = _checkpoint(
                model,
                optimizer,
                scheduler,
                corpus,
                embedding_manifest,
                config,
                epoch,
                best_metric,
                best_epoch,
                warm_start_report,
                genomes_per_dataset=genomes_per_dataset,
                loader_workers=loader_workers,
                execution_mode=execution_mode,
            )
            _atomic_torch_save(last_path, saved)
            if improved:
                _atomic_torch_save(config.run_dir / "best.pt", saved)
            if config.checkpoint_every > 0 and (
                (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.epochs
            ):
                _atomic_torch_save(
                    config.run_dir / f"epoch-{epoch + 1:04d}.pt",
                    saved,
                )
            if config.stop_after_epoch is not None and epoch >= config.stop_after_epoch:
                interrupted = True
                break
            if epoch - best_epoch >= config.patience:
                stopped_early = True
                break
    finally:
        live.close()
        hardware.stop()
        _shutdown_loader(loader)

    if interrupted:
        summary = {
            "schema": "jope-prime.dime-score-summary.v1",
            "run_dir": str(config.run_dir),
            "state": "interrupted",
            "checkpoint": str(last_path),
            "epoch": epoch,
        }
        _atomic_json(config.run_dir / "summary.v1.json", summary)
        _atomic_json(
            progress_path,
            {
                "schema": "jope-prime.dime-score-progress.v1",
                "checkpoint_state": "interrupted",
                "epoch": epoch,
            },
        )
        _update_suite_progress(config, checkpoint_state="interrupted")
        return summary

    best_path = config.run_dir / "best.pt"
    if not best_path.exists():
        best_path = last_path
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation_split = 2 if config.heldout_diagnostic else 1
    final_metrics = evaluate(
        model,
        corpus,
        embedding_matrix,
        embedding_lookup,
        numeric_mean,
        numeric_scale,
        device,
        split=validation_split,
        genome_only=config.genome_only,
        heldout=config.heldout_diagnostic,
        train_constant=train_constant,
    )
    test_metrics = (
        None
        if config.heldout_diagnostic
        else evaluate(
            model,
            corpus,
            embedding_matrix,
            embedding_lookup,
            numeric_mean,
            numeric_scale,
            device,
            split=2,
            genome_only=config.genome_only,
            heldout=False,
            train_constant=train_constant,
        )
    )
    gate_metrics = test_metrics or final_metrics
    beats_constant = bool(
        math.isfinite(gate_metrics["mae"]) and gate_metrics["mae"] < gate_metrics["constant_percentile_mae"]
    )
    summary = {
        "schema": "jope-prime.dime-score-summary.v1",
        "run_dir": str(config.run_dir),
        "best_epoch": int(best.get("best_epoch", -1)),
        "best_mean_per_dataset_spearman": float(best.get("best_metric", math.nan)),
        "validation": final_metrics,
        "test": test_metrics,
        "selection_split": final_metrics["split_name"],
        "quality_gate_split": gate_metrics["split_name"],
        "beats_constant_percentile_baseline": beats_constant,
        "dataset_conditioning_beats_genome_only": None if config.genome_only else "requires matched ablation",
        "quality_gate": "publishable" if beats_constant else "diagnostic",
        "stopped_early": stopped_early,
        "checkpoint": str(best_path),
        "warm_start_report": warm_start_report,
        "run_kind": config.run_kind,
    }
    if beats_constant:
        publication = dict(best)
        publication["publication"] = summary
        _atomic_torch_save(config.run_dir / "published.pt", publication)
    _atomic_json(config.run_dir / "summary.v1.json", summary)
    _atomic_json(
        progress_path,
        {
            "schema": "jope-prime.dime-score-progress.v1",
            "checkpoint_state": "complete",
            "quality_gate": summary["quality_gate"],
            "best_epoch": summary["best_epoch"],
            "validation": final_metrics,
            "test": test_metrics,
            **hardware.snapshot(),
        },
    )
    _update_suite_progress(
        config,
        checkpoint_state="complete",
        quality_gate=summary["quality_gate"],
    )
    return summary


def train_dime_score(config: TrainConfig) -> dict[str, Any]:
    try:
        return _train_dime_score(config)
    except Exception as error:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "jope-prime.dime-score-error.v1",
            "time_unix_seconds": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _append_jsonl(config.run_dir / "errors.jsonl", failure)
        _atomic_json(
            config.run_dir / "progress.v1.json",
            {
                "schema": "jope-prime.dime-score-progress.v1",
                "checkpoint_state": "failed",
                **failure,
            },
        )
        _update_suite_progress(config, checkpoint_state="failed")
        raise


def run_initialization_pilots(
    base: TrainConfig,
    *,
    seeds: tuple[int, ...] = (17, 29, 43),
    pilot_epochs: int = 5,
) -> dict[str, Any]:
    preflight = warm_start_preflight(base)
    initializations = ["scratch"]
    if preflight["eligible"]:
        initializations.append("warm")
    results: dict[str, list[dict[str, Any]]] = {initialization: [] for initialization in initializations}
    for initialization in initializations:
        for seed in seeds:
            run = base.run_dir / "pilots" / initialization / f"seed-{seed}"
            config = TrainConfig(
                **{
                    **asdict(base),
                    "run_dir": run,
                    "seed": seed,
                    "epochs": pilot_epochs,
                    "resume": (run / "last.pt").is_file(),
                    "warm_start": base.warm_start if initialization == "warm" else None,
                    "autotune": False,
                    "compile_model": False,
                    "suite_dir": base.run_dir,
                    "suite_phase": f"pilot_{initialization}_seed_{seed}",
                    "run_kind": "diagnostic_pilot",
                    "stop_after_epoch": None,
                }
            )
            results[initialization].append(train_dime_score(config))
    comparison: dict[str, dict[str, float]] = {}
    for initialization, summaries in results.items():
        spearman = sorted(float(summary["validation"]["mean_per_dataset_spearman"]) for summary in summaries)
        pairwise = sorted(float(summary["validation"]["pairwise_accuracy"]) for summary in summaries)
        middle = len(spearman) // 2
        comparison[initialization] = {
            "median_mean_per_dataset_spearman": spearman[middle],
            "median_pairwise_accuracy": pairwise[middle],
        }
    selected = max(
        comparison,
        key=lambda name: (
            comparison[name]["median_mean_per_dataset_spearman"],
            comparison[name]["median_pairwise_accuracy"],
        ),
    )
    output = {
        "schema": "jope-prime.dime-initialization-pilots.v1",
        "seeds": list(seeds),
        "eligibility": {
            "scratch": {"eligible": True, "reason": "always eligible"},
            "warm": preflight,
        },
        "comparison": comparison,
        "selected_initialization": selected,
        "runs": results,
    }
    base.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(base.run_dir / "initialization-pilots.v1.json", output)
    return output
