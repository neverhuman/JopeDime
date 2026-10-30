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

from jope.broad_shadow.corpus import BroadCorpus, ExactOnceBatchSampler

from .model import DimeBroadCritic, broad_critic_losses, broad_partial_warm_start
from .train import MINIMUM_VRAM_HEADROOM_MIB


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
