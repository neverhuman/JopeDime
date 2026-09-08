"""Minimal, model-agnostic DDP helpers for policy_v3 cluster training.

torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; we init NCCL (GPU) or gloo (CPU), pin the local
GPU, wrap the model, aggregate scalars across ranks, and gate side effects to rank 0. Mirrors
the working v2 infra in ``jope/train.py`` but without the v2-model coupling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class DistContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"


def init_distributed(requested_device: str) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        # single process: honor the requested device (cuda→cpu if unavailable)
        dev = requested_device if (not requested_device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        return DistContext(enabled=False, device=dev)
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def wrap_ddp(model, ctx: DistContext):
    """Broadcast rank-0 weights so all ranks start identical, then return the model UNWRAPPED.

    We deliberately do NOT use DistributedDataParallel: this model has a tied embedding/output
    weight (reused twice per forward) and a data-dependent loss graph (reward/runtime/stability
    terms appear only when a batch carries the right labels). Both break DDP's autograd-hook
    reducer ("mark ready only once") and rule out static_graph. Instead we sync gradients
    manually via ``all_reduce_gradients`` after each backward — robust to any graph shape.
    """
    if ctx.enabled:
        broadcast_parameters(model, ctx)
    return model


def broadcast_parameters(model, ctx: DistContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for b in model.buffers():
            dist.broadcast(b.data, src=0)


def all_reduce_gradients(model, ctx: DistContext) -> None:
    """Average .grad across ranks (params unused this step contribute a zero grad so every
    rank all-reduces the same tensor set — safe with the data-dependent graph)."""
    if not ctx.enabled:
        return
    import torch.distributed as dist

    world = float(ctx.world_size)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def is_main(ctx: DistContext) -> bool:
    return (not ctx.enabled) or ctx.rank == 0


def shard_for_rank(items: list, ctx: DistContext) -> list:
    """Give each rank a disjoint slice (stride by world_size) — preserves item grouping."""
    if not ctx.enabled:
        return items
    return items[ctx.rank :: ctx.world_size]


def aggregate_scalars(values: dict[str, float], ctx: DistContext) -> dict[str, float]:
    """Sum a dict of scalars across all ranks (for loss/step totals). No-op if single-proc."""
    if not ctx.enabled:
        return values
    import torch.distributed as dist

    keys = sorted(values)
    t = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: float(v) for k, v in zip(keys, t.tolist())}


def all_ranks_true(value: bool, ctx: DistContext) -> bool:
    if not ctx.enabled:
        return bool(value)
    import torch.distributed as dist

    t = torch.tensor([1.0 if value else 0.0], dtype=torch.float32, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(float(t.item()) >= 1.0)


def barrier(ctx: DistContext) -> None:
    if ctx.enabled:
        import torch.distributed as dist

        dist.barrier()


def shutdown_distributed(ctx: DistContext) -> None:
    if ctx.enabled:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def configure_cuda_fast_math() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
