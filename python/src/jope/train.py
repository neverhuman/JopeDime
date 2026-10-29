from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
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
    normalize_active_v2_vocab,
    rows_to_active_v2_batch,
    rows_to_batch,
)
from .schema import TrainingRow

ACTIVE_V2_LOSS_COMPONENTS = [
    "complexity_huber",
    "family_bce",
    "head_ce",
    "listwise_top_ce",
    "param_ce",
    "rank_huber",
    "ranknet",
    "reward_bt",
    "reward_listwise",
    "runtime_huber",
    "slot_append_bce",
    "slot_op_ce",
    "top_candidate_bce",
    "transform_count_ce",
    "value_huber",
]


@dataclass(frozen=True)
class _DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


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
    step_telemetry = _open_step_telemetry(
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
            telemetry_timer = step_telemetry[2] if step_telemetry is not None else None
            if telemetry_timer is not None:
                telemetry_timer.begin()
            batch = rows_to_active_v2_batch(batch_rows, base_model.vocab, device=resolved_device)
            if telemetry_timer is not None:
                telemetry_timer.mark(step_telemetry[3]["step.span.data"])
            out = model(batch.embedding, batch.target_summary, batch.genome_features)
            components = base_model.loss_components_from_output(out, batch)
            loss = base_model.weighted_loss(components, profile_weights=profile_weights)
            if telemetry_timer is not None:
                telemetry_timer.mark(step_telemetry[3]["step.span.fwd"])
            opt.zero_grad()
            loss.backward()
            if telemetry_timer is not None:
                telemetry_timer.mark(step_telemetry[3]["step.span.bwd"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_host = float(loss.detach().cpu())
            total += loss_host
            if telemetry_timer is not None:
                batch_obj, _, _, column = step_telemetry
                batch_obj.put(column["train.loss"], loss_host)
            for name, value in components.items():
                value_host = float(value.detach().cpu())
                component_totals[name] = component_totals.get(name, 0.0) + value_host
                if telemetry_timer is not None:
                    batch_obj.put(step_telemetry[4][name], value_host)
            if telemetry_timer is not None:
                telemetry_timer.mark(step_telemetry[3]["step.span.opt"])
                batch_obj.put(column["optim.lr.group"], float(opt.param_groups[0]["lr"]))
                telemetry_timer.end(column["step.wall"], column["step.unattributed"])
                telemetry_step += 1
                batch_obj.commit(telemetry_step)
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
        if step_telemetry is not None:
            step_telemetry[0].flush()

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
    telemetry_manifest = step_telemetry[1].close() if step_telemetry is not None else None
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


def _open_step_telemetry(*, device: str, rank: int, metadata: dict[str, Any]):
    """Initialize the optional PyO3 recorder only when the caller explicitly opts in."""
    run_dir = os.environ.get("JOPE_TELEMETRY_RUN_DIR")
    if not run_dir or rank != 0:
        return None
    from .telemetry import GpSession, Recorder, metric_registry
    from .telemetry.train_hooks import StepBatch, StepTimer

    if GpSession is None or metric_registry is None:
        raise RuntimeError("jope telemetry PyO3 extension is unavailable")
    component_names = tuple(sorted(ACTIVE_V2_LOSS_COMPONENTS))
    columns = (
        "train.loss",
        *(f"train.loss.component/{name}" for name in component_names),
        "optim.lr.group",
        "step.span.data",
        "step.span.fwd",
        "step.span.bwd",
        "step.span.opt",
        "step.wall",
        "step.unattributed",
    )
    registry = {name: (metric_id, grouped) for metric_id, name, _, _, grouped in metric_registry()}
    layout: list[tuple[int, int]] = []
    column_ids: dict[str, int] = {}
    component_columns: dict[str, int] = {}
    for col, name in enumerate(columns):
        if name.startswith("train.loss.component/"):
            component_name = name.split("/", 1)[1]
            base, group = "train.loss.component", component_names.index(component_name)
            component_columns[component_name] = col
        elif name == "optim.lr.group":
            base, group = name, 0
        else:
            base, group = name, 0
        metric_id, grouped = registry[base]
        if group and not grouped:
            raise RuntimeError(f"telemetry metric {base} is not declared grouped")
        layout.append((metric_id, group))
        column_ids[name] = col
    metric_groups = {
        str(registry["train.loss.component"][0]): {
            str(group): name for group, name in enumerate(component_names)
        },
        str(registry["optim.lr.group"][0]): {"0": "optimizer_group_0"},
    }
    recorder = Recorder(
        run_dir,
        ["python-trainer"],
        {**metadata, "device": device, "metric_groups": metric_groups},
    )
    recorder.set_metric_layout(0, layout)
    batch = StepBatch(columns, lambda steps, values, rows: recorder.submit(0, steps, values, rows))
    return batch, recorder, StepTimer(batch), column_ids, component_columns


def _dataset_grouped_batches(
    rows: list[TrainingRow],
    batch_size: int,
    rng: random.Random,
) -> list[list[TrainingRow]]:
    """Pack rows into batches that keep comparable candidates together.

    Within-dataset ranking losses (Bradley-Terry / listwise) need multiple genomes from
    the same evidence authority in one batch to form pairs; uniform shuffling almost
    never delivers that. ``dataset_id`` remains the conditioning identity, while an
    optional ``reward_pair_identity`` prevents incompatible CV plans from pairing.
    """
    groups: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        identity = (row.metadata or {}).get("reward_pair_identity") or row.dataset_id
        groups[str(identity)].append(row)
    order = list(groups.keys())
    rng.shuffle(order)
    batches: list[list[TrainingRow]] = []
    current: list[TrainingRow] = []
    for dataset_id in order:
        group = groups[dataset_id]
        if len(group) > batch_size:
            for start in range(0, len(group), batch_size):
                batches.append(group[start : start + batch_size])
            continue
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def _init_distributed_if_needed(device: str) -> tuple[_DistributedContext, str]:
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    if world_size <= 1:
        return _DistributedContext(enabled=False), device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return _DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    ), device


def _wrap_distributed(
    model: JopeActiveModelV2,
    device: str,
    distributed: _DistributedContext,
) -> DistributedDataParallel:
    if device.startswith("cuda"):
        return DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            find_unused_parameters=True,
        )
    return DistributedDataParallel(model, find_unused_parameters=True)


def _unwrap_model(model: Any) -> JopeActiveModelV2:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _aggregate_loss_totals(
    total: float,
    steps: int,
    component_totals: dict[str, float],
    device: str,
    distributed: _DistributedContext,
) -> tuple[float, dict[str, float], int]:
    names = ACTIVE_V2_LOSS_COMPONENTS
    if distributed.enabled:
        values = [float(total), float(steps)] + [float(component_totals.get(name, 0.0)) for name in names]
        tensor = torch.tensor(
            values, dtype=torch.float64, device=device if device.startswith("cuda") else "cpu"
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total = float(tensor[0].item())
        steps = int(tensor[1].item())
        component_totals = {name: float(tensor[index + 2].item()) for index, name in enumerate(names)}
    denom = max(1, steps)
    return (
        total / denom,
        {name: value / denom for name, value in sorted(component_totals.items()) if value != 0.0},
        denom,
    )


def _distributed_barrier(distributed: _DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.barrier()


def _destroy_distributed(distributed: _DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _gather_rank_status(device: str, distributed: _DistributedContext) -> list[dict[str, Any]]:
    status: dict[str, Any] = {
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "local_rank": distributed.local_rank,
        "hostname": socket.gethostname(),
        "device": device,
        **_cuda_metrics(device),
    }
    nvidia = _nvidia_smi_metrics()
    if nvidia:
        status.update(nvidia)
    if not distributed.enabled:
        return [status]
    gathered: list[dict[str, Any] | None] = [None for _ in range(distributed.world_size)]
    dist.all_gather_object(gathered, status)
    return [row for row in gathered if isinstance(row, dict)] if distributed.is_rank0 else []


def _active_v2_loss_summary(
    model: JopeActiveModelV2,
    rows: list[TrainingRow],
    vocab: dict,
    *,
    batch_size: int,
    device: str,
    profile_weights: dict[str, float] | None = None,
) -> tuple[float | None, dict[str, float]]:
    if not rows:
        return None, {}
    model.eval()
    total = 0.0
    steps = 0
    component_totals: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = rows_to_active_v2_batch(batch_rows, vocab, device=device)
            components = model.loss_components(batch)
            loss = model.weighted_loss(components, profile_weights=profile_weights)
            total += float(loss.detach().cpu())
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
    return total / max(1, steps), {
        name: value / max(1, steps) for name, value in sorted(component_totals.items())
    }


def _flatten_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _nvidia_smi_metrics() -> dict[str, float]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        used, total, util = [float(part.strip()) for part in proc.stdout.splitlines()[0].split(",")[:3]]
    except (TypeError, ValueError):
        return {}
    return {
        "nvidia_smi_memory_used_mb": used,
        "nvidia_smi_memory_total_mb": total,
        "nvidia_smi_gpu_utilization_pct": util,
    }


def _reset_cuda_peak(device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    except RuntimeError:
        return


def _cuda_metrics(device: str) -> dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    try:
        cuda_device = torch.device(device)
        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated(cuda_device) / (1024 * 1024),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved(cuda_device) / (1024 * 1024),
            "gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated(cuda_device) / (1024 * 1024),
        }
    except RuntimeError:
        return {}


def _configure_cuda_fast_math(device: str) -> bool:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return False
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def _epoch_loss_profile(loss_profile: str, *, epoch: int, epochs: int) -> str:
    profile = str(loss_profile or "two_phase")
    if profile != "two_phase":
        return profile
    switch_epoch = max(1, int(round(max(1, epochs) * 0.3)))
    return "decoder_pretrain" if epoch <= switch_epoch else "reward_focus"


def _active_v2_loss_profile_weights(profile: str) -> dict[str, float]:
    if profile == "legacy":
        return {name: 1.0 for name in ACTIVE_V2_LOSS_COMPONENTS}
    base = {name: 1.0 for name in ACTIVE_V2_LOSS_COMPONENTS}
    # Parameter CE has thousands of sparse labels and can dominate the raw sum
    # without improving proposal quality. Keep it as a weak syntax prior.
    base["param_ce"] = 0.05
    if profile in ("balanced", "quality"):
        base["ranknet"] = 1.5
        base["reward_bt"] = 2.0
        base["reward_listwise"] = 1.5
        base["listwise_top_ce"] = 1.2
        return base
    if profile == "decoder_pretrain":
        return {
            "transform_count_ce": 1.0,
            "slot_op_ce": 1.0,
            "slot_append_bce": 0.5,
            "head_ce": 1.0,
            "param_ce": 0.05,
            "family_bce": 0.5,
            "complexity_huber": 0.2,
            "runtime_huber": 0.2,
            "value_huber": 0.0,
            "rank_huber": 0.0,
            "top_candidate_bce": 0.0,
            "ranknet": 0.0,
            "listwise_top_ce": 0.0,
            "reward_bt": 0.0,
            "reward_listwise": 0.0,
        }
    if profile == "reward_focus":
        return {
            "transform_count_ce": 0.15,
            "slot_op_ce": 0.15,
            "slot_append_bce": 0.05,
            "head_ce": 0.25,
            "param_ce": 0.01,
            "family_bce": 0.15,
            "complexity_huber": 0.1,
            "runtime_huber": 0.1,
            "value_huber": 1.0,
            "rank_huber": 1.0,
            "top_candidate_bce": 1.0,
            "ranknet": 2.0,
            "listwise_top_ce": 1.5,
            "reward_bt": 3.0,
            "reward_listwise": 2.0,
        }
    raise ValueError(f"unknown active_v2 loss profile: {profile}")


def _checkpoint_metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None and not metric.startswith("val_"):
        value = row.get(f"val_{metric}")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _checkpoint_metric_better(metric: str, current: float, best: float | None) -> bool:
    if best is None:
        return True
    lower_is_better = any(token in metric for token in ("loss", "regret", "mae", "rmse"))
    return current < best if lower_is_better else current > best


def _load_active_v2_vocab(corpus: Path, rows: list[TrainingRow]) -> dict:
    candidates = [corpus.parent / "active_vocab.json", corpus.parent / "vocab.json"]
    for path in candidates:
        if path.exists():
            try:
                return normalize_active_v2_vocab(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    from .data.active import build_active_vocab

    return normalize_active_v2_vocab(build_active_vocab(row.canonical_genome or row.genome for row in rows))


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device
