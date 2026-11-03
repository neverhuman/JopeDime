from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .corpus import (
    DatasetUniformBatchSampler,
    TokenizedCorpus,
    prepare_corpus,
)
from .model import DimeScoreModel, partial_warm_start


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


def _train_dime_score(config: TrainConfig) -> dict[str, Any]:
    from .train_autotune import _autotune_training, _timed_training_step
    from .train_checkpoint import _checkpoint, _restore_rng
    from .train_data import (
        MINIMUM_VRAM_HEADROOM_MIB,
        _append_jsonl,
        _atomic_json,
        _atomic_torch_save,
        _BatchPrefetcher,
        _embedding_lookup,
        _make_data_loader,
        _numeric_normalization,
        _shutdown_loader,
        _update_suite_progress,
        _validity_positive_weight,
        load_embedding_matrix,
    )
    from .train_quality import evaluate

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
    from .train_data import _append_jsonl, _atomic_json, _update_suite_progress

    try:
        return _train_dime_score(config)
    except (Exception, ArithmeticError) as error:
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
    from .train_checkpoint import warm_start_preflight
    from .train_data import _atomic_json

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


_TRAIN_SPLIT_EXPORTS = {
    "PAD": "corpus",
    "sha256_file": "corpus",
    "score_losses": "model",
    "AUTOTUNE_BATCH_SIZES": "train_data",
    "AUTOTUNE_TIMED_STEPS": "train_data",
    "AUTOTUNE_WARMUP_STEPS": "train_data",
    "AUTOTUNE_WORKERS": "train_data",
    "MINIMUM_VRAM_HEADROOM_MIB": "train_data",
    "_BatchPrefetcher": "train_data",
    "_CorpusRowIndexes": "train_data",
    "_MmapCollator": "train_data",
    "_append_jsonl": "train_data",
    "_atomic_json": "train_data",
    "_atomic_torch_save": "train_data",
    "_available_loader_workers": "train_data",
    "_collate": "train_data",
    "_collate_cpu": "train_data",
    "_embedding_lookup": "train_data",
    "_loader_worker_init": "train_data",
    "_make_data_loader": "train_data",
    "_move_batch": "train_data",
    "_numeric_normalization": "train_data",
    "_queue_depth": "train_data",
    "_shutdown_loader": "train_data",
    "_update_suite_progress": "train_data",
    "_validity_positive_weight": "train_data",
    "load_embedding_matrix": "train_data",
    "_autotune_training": "train_autotune",
    "_benchmark_configuration": "train_autotune",
    "_timed_training_step": "train_autotune",
    "select_autotune_candidate": "train_autotune",
    "_average_precision": "train_quality",
    "_binary_auc": "train_quality",
    "_rankdata": "train_quality",
    "_spearman": "train_quality",
    "evaluate": "train_quality",
    "quality_metrics": "train_quality",
    "_checkpoint": "train_checkpoint",
    "_restore_rng": "train_checkpoint",
    "warm_start_preflight": "train_checkpoint",
}


def __getattr__(name: str) -> Any:
    module_name = _TRAIN_SPLIT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __package__)
    return getattr(module, name)
