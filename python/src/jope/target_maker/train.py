from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .data import (
    PROFILE_DIM,
    TASK_BINARY,
    TASK_REGRESSION,
    TargetCacheDataset,
    collate_target_batch,
    load_manifest,
)
from .model import ModelConfig, TargetEncoder


@dataclass
class TrainConfig:
    cache_dir: Path
    run_dir: Path
    epochs: int = 1
    final_all_data_epochs: int = 0
    batch_size: int = 256
    window_size: int = 96
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 4
    seed: int = 17
    val_fraction: float = 0.05
    steps_per_epoch: int | None = None
    length_multiplier: int = 4
    amp_dtype: str = "bf16"
    save_every: int = 1
    hidden_dim: int = 128
    token_dim: int = 128
    attention_layers: int = 2
    attention_heads: int = 4
    rich_stats: bool = False
    temperature: float = 0.10
    instance_loss_weight: float = 1.0
    group_loss_weight: float = 0.5
    profile_loss_weight: float = 0.2
    task_loss_weight: float = 0.05
    collapse_loss_weight: float = 0.05


def ddp_info() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, world_size, rank, local_rank


def setup_distributed() -> tuple[torch.device, bool, int, int]:
    is_ddp, world_size, rank, local_rank = ddp_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if is_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return device, is_ddp, rank, world_size


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def move_view(view: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in view.items()}


def encode_view(model: nn.Module, view: Mapping[str, torch.Tensor], batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    task = batch["task"].to(device, non_blocking=True)
    length = batch["length"].to(device, non_blocking=True)
    moved = move_view(view, device)
    return model(
        moved["values"],
        task,
        length,
        moved["effective_count"],
        mask=moved["mask"],
        replacement_fraction=moved["replacement_fraction"],
        return_aux=True,
    )


def supervised_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    embeddings = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    labels = torch.cat([labels, labels], dim=0).to(embeddings.device)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    denom = exp_logits.sum(dim=1).clamp_min(1.0e-8)
    log_prob = logits - torch.log(denom).unsqueeze(1)
    positives = positive_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / positives
    return loss.mean()


def instance_matching_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def variance_regularization(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_tensor(0.0)
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1.0e-4)
    variance = F.relu(0.05 - std).mean()
    cov = centered.T @ centered / max(1, embedding.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    rank_proxy = off_diag.square().mean()
    return variance + 0.05 * rank_proxy


def total_loss(
    out1: Mapping[str, torch.Tensor],
    out2: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    device: torch.device,
    cfg: TrainConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg is None:
        cfg = TrainConfig(cache_dir=Path("."), run_dir=Path("."))
    profile = batch["profile"].to(device, non_blocking=True)
    task = batch["task"].to(device, non_blocking=True)
    labels = batch["group_label"].to(device, non_blocking=True)
    instance = instance_matching_loss(out1["embedding"], out2["embedding"], temperature=cfg.temperature)
    contrastive = supervised_contrastive_loss(out1["embedding"], out2["embedding"], labels, temperature=cfg.temperature)
    profile_loss = 0.5 * (F.mse_loss(out1["profile"], profile) + F.mse_loss(out2["profile"], profile))
    task_loss = 0.5 * (F.cross_entropy(out1["task_logits"], task) + F.cross_entropy(out2["task_logits"], task))
    collapse = variance_regularization(torch.cat([out1["embedding"], out2["embedding"]], dim=0))
    total = (
        cfg.instance_loss_weight * instance
        + cfg.group_loss_weight * contrastive
        + cfg.profile_loss_weight * profile_loss
        + cfg.task_loss_weight * task_loss
        + cfg.collapse_loss_weight * collapse
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "instance": float(instance.detach().cpu()),
        "contrastive": float(contrastive.detach().cpu()),
        "profile": float(profile_loss.detach().cpu()),
        "task": float(task_loss.detach().cpu()),
        "collapse": float(collapse.detach().cpu()),
    }


@torch.no_grad()
def retrieval_metrics(z1: torch.Tensor, z2: torch.Tensor, group_labels: torch.Tensor) -> dict[str, float]:
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    sims = z1 @ z2.T
    order = torch.argsort(sims, dim=1, descending=True)
    own = torch.arange(z1.shape[0], device=z1.device)
    top1 = (order[:, 0] == own).float().mean().item()
    top10 = (order[:, : min(10, z1.shape[0])] == own[:, None]).any(dim=1).float().mean().item()
    group = group_labels.to(z1.device)
    group_top1 = group[order[:, 0]].eq(group).float().mean().item()
    return {"retrieval@1": top1, "retrieval@10": top10, "group_retrieval@1": group_top1}


@torch.no_grad()
def embedding_rank_health(embedding: torch.Tensor) -> float:
    if embedding.shape[0] < 2:
        return 0.0
    centered = embedding.float() - embedding.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    return float((singular > singular.max().clamp_min(1.0e-8) * 0.01).float().sum().cpu())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 32, cfg: TrainConfig | None = None) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    embeddings: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        out1 = encode_view(model, batch["view1"], batch, device)
        out2 = encode_view(model, batch["view2"], batch, device)
        loss, parts = total_loss(out1, out2, batch, device, cfg)
        metrics = retrieval_metrics(out1["embedding"], out2["embedding"], batch["group_label"])
        profile = batch["profile"].to(device, non_blocking=True)
        mae = F.l1_loss(out1["profile"], profile).item()
        task = batch["task"].to(device, non_blocking=True)
        binary_mask = task == TASK_BINARY
        regression_mask = task == TASK_REGRESSION
        binary_rate_mae = (
            F.l1_loss(out1["profile"][binary_mask, 2], profile[binary_mask, 2]).item() if binary_mask.any() else 0.0
        )
        regression_quantile_mae = (
            F.l1_loss(out1["profile"][regression_mask, 8:19], profile[regression_mask, 8:19]).item()
            if regression_mask.any()
            else 0.0
        )
        row = {**parts, **metrics, "profile_mae": mae, "binary_rate_mae": binary_rate_mae, "regression_quantile_mae": regression_quantile_mae}
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        embeddings.append(out1["embedding"].detach().cpu())
        count += 1
    if count == 0:
        return {}
    averaged = {key: value / count for key, value in totals.items()}
    averaged["embedding_rank"] = embedding_rank_health(torch.cat(embeddings, dim=0))
    return averaged


def make_loader(
    records,
    cfg: TrainConfig,
    *,
    train: bool,
    is_ddp: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = TargetCacheDataset(
        records,
        window_size=cfg.window_size,
        seed=cfg.seed + (0 if train else 10_000),
        two_views=True,
        length_multiplier=cfg.length_multiplier if train else 1,
        augment_lengths=(32, 64, cfg.window_size) if train else (cfg.window_size,),
        randomize=train,
    )
    sampler = None
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train, seed=cfg.seed)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_target_batch,
        persistent_workers=cfg.num_workers > 0,
        drop_last=train,
    )
    return loader, sampler if isinstance(sampler, DistributedSampler) else None


def append_metrics(run_dir: Path, row: Mapping[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def save_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    metrics: Mapping[str, float],
    name: str = "checkpoint.pt",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    path = run_dir / name
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "model_config": model_to_save.config.to_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": dict(metrics),
            "train_config": {**cfg.__dict__, "cache_dir": str(cfg.cache_dir), "run_dir": str(cfg.run_dir)},
        },
        path,
    )
    return path


def autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def train(cfg: TrainConfig) -> Path | None:
    device, is_ddp, rank, world_size = setup_distributed()
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    try:
        train_records = load_manifest(cfg.cache_dir, split="train", val_fraction=cfg.val_fraction, seed=cfg.seed)
        val_records = load_manifest(cfg.cache_dir, split="val", val_fraction=cfg.val_fraction, seed=cfg.seed)
        if not train_records:
            raise ValueError("No training records found. Run prepare-cache first.")
        if not val_records:
            val_records = train_records[: min(len(train_records), max(1, cfg.batch_size))]
        train_loader, train_sampler = make_loader(train_records, cfg, train=True, is_ddp=is_ddp, rank=rank, world_size=world_size)
        val_loader, _ = make_loader(val_records, cfg, train=False, is_ddp=False, rank=0, world_size=1)
        model = TargetEncoder(
            ModelConfig(
                hidden_dim=cfg.hidden_dim,
                token_dim=cfg.token_dim,
                attention_layers=cfg.attention_layers,
                attention_heads=cfg.attention_heads,
                rich_stats=cfg.rich_stats,
            )
        ).to(device)
        if is_ddp:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_metric = -1.0
        best_path: Path | None = None
        global_step = 0
        for epoch in range(cfg.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            start = time.time()
            seen = 0
            for step, batch in enumerate(train_loader):
                if cfg.steps_per_epoch is not None and step >= cfg.steps_per_epoch:
                    break
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, cfg.amp_dtype):
                    out1 = encode_view(model, batch["view1"], batch, device)
                    out2 = encode_view(model, batch["view2"], batch, device)
                    loss, parts = total_loss(out1, out2, batch, device, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_size = int(batch["task"].shape[0])
                seen += batch_size
                global_step += 1
                if is_main_process() and (global_step == 1 or global_step % 20 == 0):
                    elapsed = max(1.0e-6, time.time() - start)
                    append_metrics(
                        cfg.run_dir,
                        {
                            **parts,
                            "epoch": epoch,
                            "step": global_step,
                            "samples_per_sec": seen * world_size / elapsed,
                            "datasets_per_sec": seen * world_size / elapsed,
                            "phase": "train",
                            "checkpoint": str(cfg.run_dir / "checkpoint.pt"),
                            "time": time.time(),
                        },
                    )
            if is_ddp:
                torch.distributed.barrier()
            if is_main_process():
                metrics = evaluate(model.module if isinstance(model, DistributedDataParallel) else model, val_loader, device, cfg=cfg)
                metric = metrics.get("retrieval@1", 0.0) + metrics.get("group_retrieval@1", 0.0)
                row = {**metrics, "epoch": epoch, "step": global_step, "phase": "val", "time": time.time()}
                append_metrics(cfg.run_dir, row)
                checkpoint = save_checkpoint(cfg.run_dir, model, optimizer, cfg, epoch, metrics)
                if metric >= best_metric:
                    best_metric = metric
                    best_path = save_checkpoint(cfg.run_dir, model, optimizer, cfg, epoch, metrics, name="best.pt")
                append_metrics(cfg.run_dir, {**row, "checkpoint": str(checkpoint), "best_checkpoint": str(best_path)})
        if cfg.final_all_data_epochs > 0:
            all_records = load_manifest(cfg.cache_dir, split="all", seed=cfg.seed)
            all_loader, all_sampler = make_loader(all_records, cfg, train=True, is_ddp=is_ddp, rank=rank, world_size=world_size)
            for epoch in range(cfg.final_all_data_epochs):
                model.train()
                if all_sampler is not None:
                    all_sampler.set_epoch(cfg.epochs + epoch)
                for step, batch in enumerate(all_loader):
                    if cfg.steps_per_epoch is not None and step >= cfg.steps_per_epoch:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, cfg.amp_dtype):
                        out1 = encode_view(model, batch["view1"], batch, device)
                        out2 = encode_view(model, batch["view2"], batch, device)
                        loss, _ = total_loss(out1, out2, batch, device, cfg)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    global_step += 1
            if is_ddp:
                torch.distributed.barrier()
            if is_main_process():
                final_path = save_checkpoint(cfg.run_dir, model, optimizer, cfg, cfg.epochs + cfg.final_all_data_epochs, {}, name="final.pt")
                append_metrics(
                    cfg.run_dir,
                    {
                        "epoch": cfg.epochs + cfg.final_all_data_epochs,
                        "step": global_step,
                        "phase": "final",
                        "checkpoint": str(final_path),
                        "final_checkpoint": str(final_path),
                        "best_checkpoint": str(best_path) if best_path is not None else None,
                        "time": time.time(),
                    },
                )
                if best_path is None:
                    best_path = final_path
        return best_path if is_main_process() else None
    finally:
        cleanup_distributed()
