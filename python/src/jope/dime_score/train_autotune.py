from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .corpus import (
    DatasetUniformBatchSampler,
    TokenizedCorpus,
)
from .model import DimeScoreModel, score_losses
from .train_data import (
    AUTOTUNE_BATCH_SIZES,
    AUTOTUNE_TIMED_STEPS,
    AUTOTUNE_WARMUP_STEPS,
    MINIMUM_VRAM_HEADROOM_MIB,
    _available_loader_workers,
    _BatchPrefetcher,
    _make_data_loader,
    _shutdown_loader,
)

if TYPE_CHECKING:
    from .train import TrainConfig


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
    except (Exception, ArithmeticError) as error:
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
            except (Exception, ArithmeticError) as error:
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
