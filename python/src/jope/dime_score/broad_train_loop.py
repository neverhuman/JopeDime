"""Training loop and ensemble inference for the broad structural critic."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from jope.broad_shadow.corpus import BroadCorpus, ExactOnceBatchSampler

from .broad_train import (
    BroadTrainConfig,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _autotune_broad_batch,
    _checkpoint,
    _collate,
    _configure_vram_cap,
    _forward,
    _metric_gap,
    _metric_set,
    _retain_latest_three,
    broad_quality_metrics,
    evaluate_broad,
)
from .model import DimeBroadCritic, broad_critic_losses, broad_partial_warm_start
from .train import MINIMUM_VRAM_HEADROOM_MIB


def train_broad_critic(config: BroadTrainConfig) -> dict[str, Any]:
    if (
        config.epochs < 1
        or config.minimum_early_stop_epoch < 100
        or config.patience < 30
        or not 0 < config.maximum_loader_wait_percent <= 100
    ):
        raise ValueError(
            "broad training requires >=100 warmup epochs, patience >=30, and a loader wait gate in (0, 100]"
        )
    corpus = BroadCorpus(config.corpus_root)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = config.run_dir / "telemetry"
    telemetry_dir.mkdir(exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this release environment; use an "
            "admitted SCQ GPU with a driver-compatible PyTorch build"
        )
    vram_cap = _configure_vram_cap(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    vocabulary_size = int(np.max(corpus.token_ids)) + 1
    max_sequence_length = int(np.max(corpus.metadata[:, 4]))
    model = DimeBroadCritic(
        vocabulary_size,
        1,
        max_sequence_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    ).to(device)
    warm_start_report = None
    if config.warm_start is not None and not config.resume:
        source = torch.load(config.warm_start, map_location="cpu", weights_only=False)
        tokens = list(corpus.index["vocabulary_tokens"])
        if len(tokens) != vocabulary_size:
            raise ValueError("broad corpus vocabulary rows differ from token IDs")
        warm_start_report = broad_partial_warm_start(model, source, tokens)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_rows = np.flatnonzero(
        (np.asarray(corpus.metadata[:, 3]) == 0) & (np.asarray(corpus.metadata[:, 6]) == 0)
    )
    validation_rows = np.flatnonzero(
        (np.asarray(corpus.metadata[:, 3]) == 0) & (np.asarray(corpus.metadata[:, 6]) == 1)
    )
    cold_rows = np.flatnonzero(np.asarray(corpus.metadata[:, 3]) == 3)
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("broad corpus lacks train/validation semantic rows")
    has_online = bool(np.any(np.asarray(corpus.metadata[train_rows, 1]) == 2))
    sampler = ExactOnceBatchSampler(
        corpus,
        batch_size=config.batch_size,
        seed=config.seed,
        eligible_rows=train_rows,
        bucket_buffer=config.bucket_buffer,
        domain_mix=(
            {"real": 0.50, "synthetic": 0.40, "online": 0.10}
            if has_online
            else {"real": 0.60, "synthetic": 0.40}
        ),
    )
    numeric_mean = torch.zeros(vocabulary_size, device=device)
    numeric_scale = torch.ones(vocabulary_size, device=device)
    effective_batch_size, autotune = _autotune_broad_batch(
        model=model,
        corpus=corpus,
        train_rows=train_rows,
        config=config,
        device=device,
        numeric_mean=numeric_mean,
        numeric_scale=numeric_scale,
    )
    sampler.batch_size = effective_batch_size
    if autotune is not None:
        _atomic_json(config.run_dir / "batch-autotune.v1.json", autotune)
    start_epoch = 0
    best_epoch = -1
    best_ndcg = -math.inf
    best_regret = math.inf
    last_path = config.run_dir / "last.pt"
    if config.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["corpus_snapshot_sha256"] != corpus.manifest["snapshot_sha256"]:
            raise ValueError("broad resume corpus identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_ndcg = float(checkpoint["best_ndcg_at_10"])
        best_regret = float(checkpoint["best_regret_at_1"])

    _atomic_json(
        config.run_dir / "run.v1.json",
        {
            "schema": "DimeBroadCriticRun.v1",
            "corpus_snapshot_sha256": corpus.manifest["snapshot_sha256"],
            "seed": config.seed,
            "genome_only": config.genome_only,
            "warm_start": warm_start_report,
            "vram_cap": vram_cap,
            "batch_autotune": autotune,
            "effective_batch_size": effective_batch_size,
            "loss_weights": {
                "ranking": 0.60,
                "score": 0.20,
                "validity": 0.10,
                "runtime": 0.10,
            },
            "early_stopping": {
                "minimum_epoch": config.minimum_early_stop_epoch,
                "patience": config.patience,
                "metrics": ["ndcg_at_10", "regret_at_1"],
            },
        },
    )
    history_path = telemetry_dir / "epoch-history.v2.jsonl"
    stopped_early = False
    from jope.telemetry import HardwareMonitor, RatchetLive

    hardware = HardwareMonitor(1.0, MINIMUM_VRAM_HEADROOM_MIB)
    hardware.start()
    live = RatchetLive(telemetry_dir / "live.v2.json", hardware, "broad_shadow")
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_started = time.monotonic()
        totals: dict[str, float] = {}
        examples = tokens_seen = 0
        compute_seconds = 0.0
        loading_seconds = 0.0
        train_predictions: list[np.ndarray] = []
        train_validity_predictions: list[np.ndarray] = []
        train_targets: list[np.ndarray] = []
        train_validity_targets: list[np.ndarray] = []
        train_datasets: list[np.ndarray] = []
        train_fidelities: list[np.ndarray] = []
        iterator_started = time.perf_counter()
        for batch_rows in sampler:
            loaded_at = time.perf_counter()
            loading_seconds += loaded_at - iterator_started
            batch = _collate(corpus, batch_rows, device)
            compute_started = time.perf_counter()
            outputs = _forward(
                model,
                batch,
                numeric_mean,
                numeric_scale,
                zero_shot=False,
                genome_only=config.genome_only,
            )
            losses = broad_critic_losses(
                outputs,
                batch["scores"],
                batch["validity"],
                batch["log_runtime"],
                batch["dataset_indices"],
                batch["fidelity_ids"],
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            compute_seconds += time.perf_counter() - compute_started
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            train_predictions.append(outputs["score"].detach().cpu().numpy())
            train_validity_predictions.append(outputs["validity"].detach().cpu().numpy())
            train_targets.append(batch["scores"].detach().cpu().numpy())
            train_validity_targets.append(batch["validity"].detach().cpu().numpy())
            train_datasets.append(batch["dataset_indices"].detach().cpu().numpy())
            train_fidelities.append(batch["fidelity_ids"].detach().cpu().numpy())
            examples += len(batch_rows)
            tokens_seen += int(batch["lengths"].sum())
            live.tick(best_epoch >= 0)
            iterator_started = time.perf_counter()
        elapsed = max(time.monotonic() - epoch_started, 1e-9)
        coverage = sampler.last_report
        if coverage is None or coverage["coverage"] != 1.0:
            raise AssertionError("broad epoch completed without full coverage")
        train_quality = broad_quality_metrics(
            np.concatenate(train_predictions),
            np.concatenate(train_validity_predictions),
            np.concatenate(train_targets),
            np.concatenate(train_validity_targets),
            np.concatenate(train_datasets),
            np.concatenate(train_fidelities),
        )
        validation = evaluate_broad(
            model,
            corpus,
            validation_rows,
            batch_size=effective_batch_size,
            device=device,
            numeric_mean=numeric_mean,
            numeric_scale=numeric_scale,
            zero_shot=False,
            genome_only=config.genome_only,
        )
        cold = {}
        if len(cold_rows):
            cold = {
                "zero_shot": evaluate_broad(
                    model,
                    corpus,
                    cold_rows,
                    batch_size=effective_batch_size,
                    device=device,
                    numeric_mean=numeric_mean,
                    numeric_scale=numeric_scale,
                    zero_shot=True,
                    genome_only=config.genome_only,
                ),
                "four_anchor": evaluate_broad(
                    model,
                    corpus,
                    cold_rows,
                    batch_size=effective_batch_size,
                    device=device,
                    numeric_mean=numeric_mean,
                    numeric_scale=numeric_scale,
                    zero_shot=False,
                    genome_only=config.genome_only,
                ),
            }
        ndcg = float(validation["ndcg_at_10"])
        regret = float(validation["regret_at_1"])
        improved = math.isfinite(ndcg) and (
            ndcg > best_ndcg + 1e-8 or (abs(ndcg - best_ndcg) <= 1e-8 and regret < best_regret - 1e-8)
        )
        if improved:
            best_ndcg, best_regret, best_epoch = ndcg, regret, epoch
        train_set = _metric_set(train_quality)
        validation_set = _metric_set(validation)
        cold_zero_set = _metric_set(cold.get("zero_shot", {}))
        cold_anchor_set = _metric_set(cold.get("four_anchor", {}))
        epoch_row = {
            "schema": "RatchetEpochHistory.v2",
            "epoch": epoch,
            "train": train_set,
            "validation": validation_set,
            "cold_real_zero_shot": cold_zero_set,
            "cold_real_four_anchor": cold_anchor_set,
            "train_validation_gap": _metric_gap(train_set, validation_set),
            "best_checkpoint": improved,
            "coverage": coverage["coverage"],
            "unique_rows": coverage["unique_rows"],
            "duplicate_rows": coverage["duplicate_rows"],
            "examples_per_second": examples / elapsed,
            "tokens_per_second": tokens_seen / elapsed,
            "shuffle_seed": coverage["shuffle_seed"],
        }
        _append_jsonl(history_path, epoch_row)
        saved = _checkpoint(
            model,
            optimizer,
            config,
            corpus,
            epoch=epoch,
            best_epoch=best_epoch,
            best_ndcg=best_ndcg,
            best_regret=best_regret,
        )
        _atomic_torch_save(last_path, saved)
        resume_path = config.run_dir / f"resume-epoch-{epoch + 1:04d}.pt"
        _atomic_torch_save(resume_path, saved)
        _retain_latest_three(config.run_dir)
        if (epoch + 1) % 10 == 0:
            _atomic_torch_save(config.run_dir / f"epoch-{epoch + 1:04d}.pt", saved)
        if improved:
            _atomic_torch_save(config.run_dir / "best.pt", saved)
            _atomic_torch_save(config.run_dir / f"best-epoch-{epoch + 1:04d}.pt", saved)
        loader_wait_percent = 100.0 * loading_seconds / max(loading_seconds + compute_seconds, 1e-9)
        _atomic_json(
            config.run_dir / "progress.v2.json",
            {
                "schema": "DimeBroadCriticProgress.v2",
                "updated_at_ms": int(time.time() * 1000),
                "epoch": epoch,
                "epochs": config.epochs,
                "coverage": coverage,
                "examples_per_second": examples / elapsed,
                "tokens_per_second": tokens_seen / elapsed,
                "loader_wait_percent": loader_wait_percent,
                "loader_acceptance_percent": (config.maximum_loader_wait_percent),
                "best_epoch": best_epoch,
                "best_ndcg_at_10": best_ndcg,
                "best_regret_at_1": best_regret,
            },
        )
        if device.type == "cuda" and loader_wait_percent > config.maximum_loader_wait_percent:
            raise RuntimeError("broad data loader exceeded the 10% steady-state step-time gate")
        if epoch + 1 >= config.minimum_early_stop_epoch and epoch - best_epoch >= config.patience:
            stopped_early = True
            break
    hardware.stop()
    summary = {
        "schema": "DimeBroadCriticSummary.v1",
        "state": "complete",
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch,
        "best_ndcg_at_10": best_ndcg,
        "best_regret_at_1": best_regret,
        "stopped_early": stopped_early,
        "checkpoint": str(config.run_dir / "best.pt"),
    }
    _atomic_json(config.run_dir / "summary.v1.json", summary)
    return summary


@torch.no_grad()
def ensemble_predict(
    models: list[DimeBroadCritic],
    batch: dict[str, torch.Tensor],
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    *,
    zero_shot: bool = False,
) -> dict[str, torch.Tensor]:
    if len(models) < 2:
        raise ValueError("epistemic ensemble requires at least two independent members")
    outputs = [
        _forward(
            model.eval(),
            batch,
            numeric_mean,
            numeric_scale,
            zero_shot=zero_shot,
            genome_only=False,
        )
        for model in models
    ]
    scores = torch.stack([output["score"] for output in outputs])
    validities = torch.stack([output["validity"] for output in outputs])
    runtimes = torch.stack([output["log_runtime"] for output in outputs])
    return {
        "score_mean": scores.mean(0),
        "epistemic_uncertainty": scores.std(0, unbiased=False),
        "validity_mean": validities.mean(0),
        "log_runtime_mean": runtimes.mean(0),
        "members": torch.as_tensor(len(models), device=scores.device),
    }
