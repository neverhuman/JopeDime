"""Broad structural-critic training over the mmap-only corpus."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jope.broad_shadow.corpus import BroadCorpus
from jope.broad_shadow.corpus import ExactOnceBatchSampler as ExactOnceBatchSampler

from .model import DimeBroadCritic, broad_critic_losses


def __getattr__(name: str) -> Any:
    modules = {
        "ensemble_predict": "broad_train_loop",
        "train_broad_critic": "broad_train_loop",
        "broad_partial_warm_start": "model",
        "MINIMUM_VRAM_HEADROOM_MIB": "train",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __package__), name)


@dataclass
class BroadTrainConfig:
    corpus_root: Path
    run_dir: Path
    seed: int = 17
    epochs: int = 300
    minimum_early_stop_epoch: int = 100
    patience: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 512
    bucket_buffer: int = 4096
    device: str = "cuda"
    autotune_batch_size: bool = True
    minimum_free_vram_mib: int = 4096
    maximum_loader_wait_percent: float = 10.0
    resume: bool = False
    warm_start: Path | None = None
    genome_only: bool = False
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _collate(
    corpus: BroadCorpus,
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    lengths = np.asarray([len(corpus.sequence(row)) for row in rows], dtype=np.int64)
    width = int(lengths.max())
    tokens = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        sequence = corpus.sequence(row)
        tokens[index, : len(sequence)] = sequence
    row_array = np.asarray(rows, dtype=np.int64)

    def tensor(value: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = torch.from_numpy(np.array(value, copy=True))
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result.to(device, non_blocking=device.type == "cuda")

    metadata = np.asarray(corpus.metadata[row_array])
    return {
        "row_indices": tensor(row_array),
        "dataset_embedding": tensor(np.asarray(corpus.dataset_embeddings[row_array]), dtype=torch.float32),
        "target_summary": tensor(np.asarray(corpus.target_summaries[row_array]), dtype=torch.float32),
        "anchor_features": tensor(np.asarray(corpus.anchor_features[row_array]), dtype=torch.float32),
        "anchor_mask": tensor(np.asarray(corpus.anchor_masks[row_array]), dtype=torch.float32),
        "fidelity_ids": tensor(metadata[:, 2], dtype=torch.long),
        "token_ids": tensor(tokens, dtype=torch.long),
        "numeric_values": tensor(np.zeros_like(tokens, dtype=np.float32), dtype=torch.float32),
        "numeric_mask": tensor(np.zeros_like(tokens, dtype=np.float32), dtype=torch.float32),
        "action_type_ids": tensor(np.zeros_like(tokens, dtype=np.int64), dtype=torch.long),
        "lengths": tensor(lengths, dtype=torch.long),
        "anchor_owner": tensor(metadata[:, 5], dtype=torch.long),
        "dataset_indices": tensor(metadata[:, 0], dtype=torch.long),
        "scores": tensor(np.asarray(corpus.scores[row_array]), dtype=torch.float32),
        "validity": tensor(np.asarray(corpus.validity[row_array]), dtype=torch.float32),
        "log_runtime": tensor(np.asarray(corpus.runtime[row_array]), dtype=torch.float32),
    }


def _forward(
    model: DimeBroadCritic,
    batch: dict[str, torch.Tensor],
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    *,
    zero_shot: bool,
    genome_only: bool,
) -> dict[str, torch.Tensor]:
    return model(
        batch["dataset_embedding"],
        batch["target_summary"],
        batch["anchor_features"],
        batch["anchor_mask"],
        batch["fidelity_ids"],
        batch["token_ids"],
        batch["numeric_values"],
        batch["numeric_mask"],
        batch["action_type_ids"],
        batch["lengths"],
        numeric_mean,
        numeric_scale,
        anchor_owner=batch["anchor_owner"],
        zero_shot=zero_shot,
        genome_only=genome_only,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return math.nan
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def broad_quality_metrics(
    predictions: np.ndarray,
    validity_predictions: np.ndarray,
    targets: np.ndarray,
    validity_targets: np.ndarray,
    datasets: np.ndarray,
    fidelities: np.ndarray,
) -> dict[str, float]:
    spearman: list[float] = []
    ndcg: list[float] = []
    regrets: list[float] = []
    normalized_mae: list[float] = []
    pair_correct = pair_total = 0
    top_valid = []
    keys = sorted(set(zip(datasets.tolist(), fidelities.tolist(), strict=True)))
    for dataset, fidelity in keys:
        indexes = np.flatnonzero((datasets == dataset) & (fidelities == fidelity) & np.isfinite(targets))
        if not len(indexes):
            continue
        actual = targets[indexes]
        predicted = predictions[indexes]
        if len(indexes) >= 2:
            spearman.append(_spearman(predicted, actual))
            left, right = np.triu_indices(len(indexes), 1)
            delta = actual[left] - actual[right]
            keep = delta != 0
            pair_correct += int(
                np.sum(np.sign(predicted[left][keep] - predicted[right][keep]) == np.sign(delta[keep]))
            )
            pair_total += int(keep.sum())
        order = np.argsort(-predicted, kind="mergesort")[:10]
        ideal = np.argsort(-actual, kind="mergesort")[:10]
        relevance = np.exp2(actual - actual.min()) - 1.0
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
        dcg = float(np.sum(relevance[order] * discounts))
        idcg = float(np.sum(relevance[ideal] * discounts))
        ndcg.append(dcg / idcg if idcg > 1e-12 else 1.0)
        chosen = indexes[order[0]]
        span = float(actual.max() - actual.min())
        regrets.append(0.0 if span <= 1e-12 else float((actual.max() - targets[chosen]) / span))
        normalized_mae.append(float(np.mean(np.abs(predicted - actual)) / max(span, 1e-6)))
        top_valid.append(float(validity_targets[chosen]))
    return {
        "pairwise_accuracy": pair_correct / pair_total if pair_total else math.nan,
        "mean_per_dataset_spearman": (float(np.nanmean(spearman)) if spearman else math.nan),
        "ndcg_at_10": float(np.mean(ndcg)) if ndcg else math.nan,
        "normalized_mae": (float(np.mean(normalized_mae)) if normalized_mae else math.nan),
        "regret_at_1": float(np.mean(regrets)) if regrets else math.nan,
        "validity_accuracy": float(np.mean((validity_predictions >= 0.5) == (validity_targets >= 0.5))),
        "top1_validity": float(np.mean(top_valid)) if top_valid else math.nan,
    }


def _metric_set(metrics: dict[str, Any]) -> dict[str, float | None]:
    def finite(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

    return {
        "pairwise_accuracy": finite("pairwise_accuracy"),
        "spearman": finite("mean_per_dataset_spearman"),
        "ndcg_at_10": finite("ndcg_at_10"),
        "normalized_mae": finite("normalized_mae"),
        "regret_at_1": finite("regret_at_1"),
    }


def _metric_gap(
    train: dict[str, float | None],
    validation: dict[str, float | None],
) -> dict[str, float | None]:
    return {
        name: (
            float(train[name] - validation[name])
            if train[name] is not None and validation[name] is not None
            else None
        )
        for name in train
    }


@torch.no_grad()
def evaluate_broad(
    model: DimeBroadCritic,
    corpus: BroadCorpus,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    zero_shot: bool,
    genome_only: bool,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    validity_predictions: list[np.ndarray] = []
    losses: dict[str, float] = {}
    steps = 0
    for start in range(0, len(rows), batch_size):
        batch = _collate(corpus, rows[start : start + batch_size].tolist(), device)
        outputs = _forward(
            model,
            batch,
            numeric_mean,
            numeric_scale,
            zero_shot=zero_shot,
            genome_only=genome_only,
        )
        batch_losses = broad_critic_losses(
            outputs,
            batch["scores"],
            batch["validity"],
            batch["log_runtime"],
            batch["dataset_indices"],
            batch["fidelity_ids"],
        )
        for name, value in batch_losses.items():
            losses[name] = losses.get(name, 0.0) + float(value)
        predictions.append(outputs["score"].cpu().numpy())
        validity_predictions.append(outputs["validity"].cpu().numpy())
        steps += 1
    prediction = np.concatenate(predictions)
    validity_prediction = np.concatenate(validity_predictions)
    metadata = np.asarray(corpus.metadata[rows])
    metrics = broad_quality_metrics(
        prediction,
        validity_prediction,
        np.asarray(corpus.scores[rows], dtype=np.float64),
        np.asarray(corpus.validity[rows], dtype=np.float64),
        metadata[:, 0],
        metadata[:, 2],
    )
    metrics["losses"] = {name: value / steps for name, value in losses.items()}
    metrics["examples"] = len(rows)
    metrics["mode"] = "genome_only" if genome_only else "zero_shot" if zero_shot else "four_anchor"
    return metrics


def _checkpoint(
    model: DimeBroadCritic,
    optimizer: torch.optim.Optimizer,
    config: BroadTrainConfig,
    corpus: BroadCorpus,
    *,
    epoch: int,
    best_epoch: int,
    best_ndcg: float,
    best_regret: float,
) -> dict[str, Any]:
    return {
        "schema": "DimeBroadCriticCheckpoint.v1",
        "model_version": "dime-broad-critic-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_ndcg_at_10": best_ndcg,
        "best_regret_at_1": best_regret,
        "corpus_snapshot_sha256": corpus.manifest["snapshot_sha256"],
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        },
    }


def _retain_latest_three(run_dir: Path) -> None:
    paths = sorted(run_dir.glob("resume-epoch-*.pt"))
    for path in paths[:-3]:
        path.unlink()


def _configure_vram_cap(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    configured_gib = float(os.environ.get("JOPE_VRAM_LIMIT_GIB", "20"))
    if configured_gib <= 0:
        raise ValueError("JOPE_VRAM_LIMIT_GIB must be positive")
    cap_bytes = min(total_bytes, int(configured_gib * 1024**3))
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total_bytes, device)
    return {
        "configured_vram_limit_gib": configured_gib,
        "device_total_vram_mib": total_bytes / 1024**2,
        "allocator_cap_mib": cap_bytes / 1024**2,
    }


def _autotune_broad_batch(
    *,
    model: DimeBroadCritic,
    corpus: BroadCorpus,
    train_rows: np.ndarray,
    config: BroadTrainConfig,
    device: torch.device,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
) -> tuple[int, dict[str, Any] | None]:
    if device.type != "cuda" or not config.autotune_batch_size:
        return config.batch_size, None
    if config.minimum_free_vram_mib < 4096:
        raise ValueError("broad CUDA training must retain at least 4 GiB free VRAM")

    lengths = np.asarray(corpus.metadata[train_rows, 4], dtype=np.int64)
    longest_first = train_rows[np.argsort(-lengths, kind="stable")]
    candidates = sorted(
        {
            min(config.batch_size, candidate)
            for candidate in (32, 64, 128, 256, 512, 1024, config.batch_size)
            if candidate <= config.batch_size
        }
    )
    records: list[dict[str, Any]] = []
    model.train()
    for candidate in candidates:
        rows = longest_first[: min(candidate, len(longest_first))].tolist()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            batch = _collate(corpus, rows, device)
            outputs = _forward(
                model,
                batch,
                numeric_mean,
                numeric_scale,
                zero_shot=False,
                genome_only=config.genome_only,
            )
            loss = broad_critic_losses(
                outputs,
                batch["scores"],
                batch["validity"],
                batch["log_runtime"],
                batch["dataset_indices"],
                batch["fidelity_ids"],
            )["total"]
            loss.backward()
            torch.cuda.synchronize(device)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            elapsed = max(time.perf_counter() - started, 1e-9)
            free_mib = free_bytes / 1024**2
            records.append(
                {
                    "batch_size": candidate,
                    "examples": len(rows),
                    "examples_per_second": len(rows) / elapsed,
                    "peak_vram_mib": (torch.cuda.max_memory_allocated(device) / 1024**2),
                    "free_vram_mib": free_mib,
                    "accepted": free_mib >= config.minimum_free_vram_mib,
                }
            )
        except torch.OutOfMemoryError:
            records.append(
                {
                    "batch_size": candidate,
                    "examples": len(rows),
                    "accepted": False,
                    "failure": "cuda_out_of_memory",
                }
            )
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    accepted = [row for row in records if row["accepted"]]
    if not accepted:
        raise RuntimeError("no broad batch size retained the required 4 GiB free VRAM")
    selected = max(
        accepted,
        key=lambda row: (
            float(row["examples_per_second"]),
            int(row["batch_size"]),
        ),
    )
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    return int(selected["batch_size"]), {
        "schema": "DimeBroadBatchAutotune.v1",
        "minimum_free_vram_mib": config.minimum_free_vram_mib,
        "candidates": records,
        "selected": selected,
    }
