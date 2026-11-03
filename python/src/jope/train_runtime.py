from __future__ import annotations

import json
import os
import random
import socket
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import (
    JopeActiveModelV2,
    normalize_active_v2_vocab,
    rows_to_active_v2_batch,
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


def _open_step_telemetry(*, device: str, rank: int, metadata: dict[str, Any]):
    """Open the Rust step recorder only when the caller explicitly opts in (rank 0)."""
    run_dir = os.environ.get("JOPE_TELEMETRY_RUN_DIR")
    if not run_dir or rank != 0:
        return None
    from .telemetry import open_python_trainer

    return open_python_trainer(run_dir, sorted(ACTIVE_V2_LOSS_COMPONENTS), {**metadata, "device": device})


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
