from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .io import read_jsonl
from .model import (
    JopeActiveModelV2,
    JopeOneShotModel,
    active_v2_validation_metrics,
    rows_to_active_v2_batch,
    rows_to_batch,
)
from .schema import TrainingRow

from .train_runtime import (
    ACTIVE_V2_LOSS_COMPONENTS,
    _DistributedContext,
    _active_v2_loss_profile_weights,
    _active_v2_loss_summary,
    _aggregate_loss_totals,
    _checkpoint_metric_better,
    _checkpoint_metric_value,
    _configure_cuda_fast_math,
    _cuda_metrics,
    _dataset_grouped_batches,
    _destroy_distributed,
    _distributed_barrier,
    _epoch_loss_profile,
    _flatten_metrics,
    _gather_rank_status,
    _init_distributed_if_needed,
    _load_active_v2_vocab,
    _nvidia_smi_metrics,
    _open_step_telemetry,
    _reset_cuda_peak,
    _resolve_device,
    _unwrap_model,
    _wrap_distributed,
)
def train_one_epoch(
    rows: list[TrainingRow],
    *,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[JopeOneShotModel, float]:
    model = JopeOneShotModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    return _train_epochs(model, opt, rows, epochs=1, batch_size=batch_size, device=device)[0:2]


def _train_epochs(
    model: JopeOneShotModel,
    opt: torch.optim.Optimizer,
    rows: list[TrainingRow],
    *,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int = 0,
    metrics_jsonl: str | Path | None = None,
) -> tuple[JopeOneShotModel, float, list[float]]:
    history: list[float] = []
    rng = random.Random(seed)
    train_rows = rows[:]
    total = 0.0
    metrics_path = Path(metrics_jsonl) if metrics_jsonl is not None else None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, max(1, epochs) + 1):
        rng.shuffle(train_rows)
        epoch_total = 0.0
        steps = 0
        model.train()
        for start in range(0, len(train_rows), batch_size):
            batch_rows = train_rows[start : start + batch_size]
            batch = rows_to_batch(batch_rows, device=device)
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_total += float(loss.detach().cpu())
            steps += 1
        total = epoch_total / max(1, steps)
        history.append(total)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "epochs": max(1, epochs),
                            "loss": total,
                            "rows": len(rows),
                            "batch_size": batch_size,
                            "device": device,
                            "elapsed_sec": time.time() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return model, total, history


def load_rows(path: str | Path) -> list[TrainingRow]:
    return [TrainingRow.from_json(row) for row in read_jsonl(path)]


def train_from_jsonl(
    corpus: str | Path,
    checkpoint: str | Path | None = None,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    metrics_jsonl: str | Path | None = None,
) -> dict:
    rows = load_rows(corpus)
    if not rows:
        raise ValueError("corpus contains no rows")
    model = JopeOneShotModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model, loss, history = _train_epochs(
        model,
        opt,
        rows,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        metrics_jsonl=metrics_jsonl,
    )
    if checkpoint is not None:
        out = Path(checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "loss": loss, "loss_history": history, "epochs": epochs}, out
        )
    return {
        "rows": len(rows),
        "epochs": epochs,
        "loss": loss,
        "loss_history": history,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def train_active_v2_from_jsonl(
    corpus: str | Path,
    features: str | Path | None = None,
    checkpoint: str | Path | None = None,
    *,
    epochs: int = 1,
    batch_size: int = 512,
    lr: float = 5e-4,
    device: str = "cpu",
    metrics_jsonl: str | Path | None = None,
    seed: int = 0,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    loss_profile: str = "two_phase",
    checkpoint_metric: str = "val_pairwise_win_accuracy",
) -> dict:
    rows = load_rows(corpus)
    from .split_integrity import partition_training_rows

    input_rows = len(rows)
    split_partition = partition_training_rows(rows)
    train_rows = split_partition["train"]
    val_rows = split_partition["development"]
    model_row_ids = {id(row) for row in (*train_rows, *val_rows)}
    rows = [row for row in rows if id(row) in model_row_ids]
    if not train_rows:
        raise ValueError("active_v2 corpus contains no train-split rows")
    vocab = _load_active_v2_vocab(Path(corpus), rows)
    resolved_device = _resolve_device(device)
    distributed, resolved_device = _init_distributed_if_needed(resolved_device)
    cuda_fast_math = _configure_cuda_fast_math(resolved_device)
    model = JopeActiveModelV2(vocab, hidden_dim=hidden_dim, dropout=dropout).to(resolved_device)
    if distributed.enabled:
        model = _wrap_distributed(model, resolved_device, distributed)
    base_model = _unwrap_model(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    history: list[float] = []
    metric_history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric_value: float | None = None
    best_epoch: int | None = None
    metrics_path = Path(metrics_jsonl) if metrics_jsonl is not None else None
    if metrics_path is not None and distributed.is_rank0:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tel = _open_step_telemetry(
        device=resolved_device,
        rank=distributed.rank,
        metadata={
            "trainer": "python.active_v2",
            "device": resolved_device,
            "batch_size": batch_size,
            "epochs": max(1, epochs),
            "seed": seed,
            "world_size": distributed.world_size,
            "data_identity": "unavailable_without_extra_input_read",
        },
    )
    telemetry_step = 0
    for epoch in range(1, max(1, epochs) + 1):
        epoch_started = time.time()
        epoch_loss_profile = _epoch_loss_profile(loss_profile, epoch=epoch, epochs=max(1, epochs))
        profile_weights = _active_v2_loss_profile_weights(epoch_loss_profile)
        _reset_cuda_peak(resolved_device)
        if distributed.enabled:
            sampler = DistributedSampler(
                train_rows,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            sampler.set_epoch(epoch)
            epoch_train_rows = [train_rows[index] for index in sampler]
        else:
            rng.shuffle(train_rows)
            epoch_train_rows = train_rows
        model.train()
        total = 0.0
        steps = 0
        component_totals: dict[str, float] = {}
        for batch_rows in _dataset_grouped_batches(epoch_train_rows, batch_size, rng):
            if tel is not None:
                tel.begin()
            batch = rows_to_active_v2_batch(batch_rows, base_model.vocab, device=resolved_device)
            if tel is not None:
                tel.mark("step.span.data")
            out = model(batch.embedding, batch.target_summary, batch.genome_features)
            components = base_model.loss_components_from_output(out, batch)
            loss = base_model.weighted_loss(components, profile_weights=profile_weights)
            if tel is not None:
                tel.mark("step.span.fwd")
            opt.zero_grad()
            loss.backward()
            if tel is not None:
                tel.mark("step.span.bwd")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if tel is not None:
                tel.mark("step.span.opt")
            loss_host = float(loss.detach().cpu())
            total += loss_host
            if tel is not None:
                tel.put("train.loss", loss_host)
            for name, value in components.items():
                value_host = float(value.detach().cpu())
                component_totals[name] = component_totals.get(name, 0.0) + value_host
                if tel is not None:
                    tel.put(f"train.loss.component/{name}", value_host)
            if tel is not None:
                # The existing loss and component read-backs: the host waits here for the whole
                # step's queued GPU work, so this span, not `opt`, carries the GPU drain.
                tel.mark("step.span.readback")
                tel.put("optim.lr.group", float(opt.param_groups[0]["lr"]))
                tel.end()
                telemetry_step += 1
                tel.commit(telemetry_step)
            steps += 1
        train_loss, train_loss_components, global_steps = _aggregate_loss_totals(
            total,
            steps,
            component_totals,
            resolved_device,
            distributed,
        )
        history.append(train_loss)
        validation_rows = val_rows if val_rows else train_rows[: min(len(train_rows), batch_size)]
        val_loss: float | None = None
        val_loss_components: dict[str, float] = {}
        metrics: dict[str, float] = {}
        if distributed.is_rank0:
            val_loss, val_loss_components = _active_v2_loss_summary(
                base_model,
                validation_rows,
                base_model.vocab,
                batch_size=batch_size,
                device=resolved_device,
                profile_weights=profile_weights,
            )
            metrics = active_v2_validation_metrics(
                base_model,
                validation_rows,
                base_model.vocab,
                batch_size=batch_size,
                device=resolved_device,
            )
        _distributed_barrier(distributed)
        rank_status = _gather_rank_status(resolved_device, distributed)
        epoch_sec = time.time() - epoch_started
        row = {
            "epoch": epoch,
            "epochs": max(1, epochs),
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "validation_loss_rows": len(validation_rows),
            "batch_size": batch_size,
            "lr": lr,
            "device": resolved_device,
            "requested_device": device,
            "cuda_fast_math": cuda_fast_math,
            "distributed": distributed.enabled,
            "distributed_backend": distributed.backend,
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "local_rank": distributed.local_rank,
            "hostname": socket.gethostname(),
            "global_train_steps": global_steps,
            "local_train_steps": steps,
            "local_train_rows": len(epoch_train_rows),
            "features": str(features) if features else None,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "loss_profile": loss_profile,
            "epoch_loss_profile": epoch_loss_profile,
            "loss_profile_weights": profile_weights,
            "checkpoint_metric": checkpoint_metric,
            "epoch_sec": epoch_sec,
            "elapsed_sec": time.time() - started,
            "examples_per_sec": len(train_rows) / max(epoch_sec, 1e-9),
            "train_loss_components": train_loss_components,
            "val_loss_components": val_loss_components,
            "loss_weights": {
                name: float(torch.exp(-param.detach()).cpu())
                for name, param in base_model.loss_log_vars.items()
            },
            "rank_status": rank_status,
            **_flatten_metrics("train_loss", train_loss_components),
            **_flatten_metrics("val_loss", val_loss_components),
            **_cuda_metrics(resolved_device),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        current_metric = _checkpoint_metric_value(row, checkpoint_metric)
        if (
            distributed.is_rank0
            and current_metric is not None
            and _checkpoint_metric_better(
                checkpoint_metric,
                current_metric,
                best_metric_value,
            )
        ):
            best_metric_value = current_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
        row["checkpoint_metric_value"] = current_metric
        row["best_checkpoint_metric_value"] = best_metric_value
        row["best_checkpoint_epoch"] = best_epoch
        if distributed.is_rank0:
            metric_history.append(row)
        if metrics_path is not None and distributed.is_rank0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if tel is not None:
            tel.flush()

    if checkpoint is not None and distributed.is_rank0:
        out = Path(checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        model_state = (
            best_state
            if best_state is not None
            else {key: value.detach().cpu() for key, value in base_model.state_dict().items()}
        )
        torch.save(
            {
                "model_version": "active_v2",
                "model": model_state,
                "vocab": base_model.vocab,
                "loss": history[-1] if history else None,
                "loss_history": history,
                "metrics_history": metric_history,
                "epochs": epochs,
                "dataset_dim": base_model.dataset_dim,
                "target_dim": base_model.target_dim,
                "hidden_dim": base_model.hidden_dim,
                "dropout": base_model.dropout,
                "loss_profile": loss_profile,
                "checkpoint_metric": checkpoint_metric,
                "best_checkpoint_metric_value": best_metric_value,
                "best_checkpoint_epoch": best_epoch,
            },
            out,
        )
    _distributed_barrier(distributed)
    _destroy_distributed(distributed)
    telemetry_manifest = tel.close() if tel is not None else None
    return {
        "rows": len(rows),
        "input_rows": input_rows,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "report_only_rows": len(split_partition["report_only"]),
        "report_only_split_counts": split_partition["report_only_counts"],
        "split_compatibility": split_partition["compatibility"],
        "split_manifest_sha256": split_partition["split_manifest_sha256"],
        "epochs": epochs,
        "loss": history[-1] if history else None,
        "loss_history": history,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "features": str(features) if features else None,
        "device": resolved_device,
        "requested_device": device,
        "distributed": distributed.enabled,
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "loss_profile": loss_profile,
        "checkpoint_metric": checkpoint_metric,
        "best_checkpoint_metric_value": best_metric_value,
        "best_checkpoint_epoch": best_epoch,
        "telemetry": telemetry_manifest,
    }
