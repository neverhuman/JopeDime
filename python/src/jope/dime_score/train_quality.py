from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .corpus import (
    PAD,
    DatasetUniformBatchSampler,
    TokenizedCorpus,
    prepare_corpus,
    sha256_file,
)
from .model import DimeScoreModel, partial_warm_start, score_losses

from .train_data import _collate, _validity_positive_weight


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 2:
        return math.nan
    left, right = _rankdata(predictions), _rankdata(targets)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    count_positive, count_negative = int(positive.sum()), int((~positive).sum())
    if not count_positive or not count_negative:
        return math.nan
    ranks = _rankdata(scores) + 1.0
    return float(
        (ranks[positive].sum() - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    if not positive.any():
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered = positive[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def quality_metrics(
    scores: np.ndarray,
    validities: np.ndarray,
    targets: np.ndarray,
    validity_targets: np.ndarray,
    datasets: np.ndarray,
    train_constant: float,
) -> dict[str, float]:
    valid = np.isfinite(targets)
    per_dataset_spearman: list[float] = []
    pair_correct = pair_total = 0
    top_precision: list[float] = []
    top_regret: list[float] = []
    for dataset in np.unique(datasets):
        indexes = np.flatnonzero((datasets == dataset) & valid)
        if len(indexes) < 2:
            continue
        per_dataset_spearman.append(_spearman(scores[indexes], targets[indexes]))
        left, right = np.triu_indices(len(indexes), 1)
        target_delta = targets[indexes[left]] - targets[indexes[right]]
        keep = target_delta != 0
        prediction_delta = scores[indexes[left]] - scores[indexes[right]]
        pair_correct += int(np.sum(np.sign(prediction_delta[keep]) == np.sign(target_delta[keep])))
        pair_total += int(keep.sum())
        count = max(1, math.ceil(0.1 * len(indexes)))
        predicted_top = indexes[np.argsort(-scores[indexes], kind="mergesort")[:count]]
        actual_top = set(indexes[np.argsort(-targets[indexes], kind="mergesort")[:count]].tolist())
        top_precision.append(sum(int(index in actual_top) for index in predicted_top) / count)
        top_regret.append(float(targets[indexes].max() - targets[predicted_top].max()))
    mae = float(np.mean(np.abs(scores[valid] - targets[valid]))) if valid.any() else math.nan
    constant_mae = float(np.mean(np.abs(train_constant - targets[valid]))) if valid.any() else math.nan
    bins = np.minimum((scores[valid] * 10).astype(int), 9) if valid.any() else np.array([])
    calibration = []
    for bin_index in range(10):
        selected = bins == bin_index
        if np.any(selected):
            calibration.append(abs(float(scores[valid][selected].mean() - targets[valid][selected].mean())))
    labels = (validity_targets >= 1.0 - 1e-12).astype(np.float64)
    return {
        "mean_per_dataset_spearman": float(np.nanmean(per_dataset_spearman))
        if per_dataset_spearman
        else math.nan,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else math.nan,
        "mae": mae,
        "constant_percentile_mae": constant_mae,
        "calibration_mae": float(np.mean(calibration)) if calibration else math.nan,
        "top_decile_precision": float(np.mean(top_precision)) if top_precision else math.nan,
        "top_decile_regret": float(np.mean(top_regret)) if top_regret else math.nan,
        "validity_auroc": _binary_auc(labels, validities),
        "validity_pr_auc": _average_precision(labels, validities),
    }


@torch.no_grad()
def evaluate(
    model: DimeScoreModel,
    corpus: TokenizedCorpus,
    embedding_matrix: np.ndarray,
    embedding_lookup: np.ndarray,
    numeric_mean: torch.Tensor,
    numeric_scale: torch.Tensor,
    device: torch.device,
    *,
    split: int,
    genome_only: bool,
    heldout: bool,
    train_constant: float,
    batch_size: int = 512,
) -> dict[str, Any]:
    if heldout:
        indexes = np.flatnonzero(
            (np.asarray(corpus.dataset_splits) == split) & (np.asarray(corpus.splits) == 2)
        )
    else:
        indexes = np.flatnonzero(np.asarray(corpus.splits) == split)
    model.eval()
    scores: list[np.ndarray] = []
    score_logits: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    validity_logits: list[np.ndarray] = []
    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size].tolist()
        batch = _collate(corpus, batch_indexes, embedding_matrix, embedding_lookup, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["embedding"],
                batch["token_ids"],
                batch["numeric_values"],
                batch["numeric_mask"],
                batch["action_type_ids"],
                batch["lengths"],
                numeric_mean,
                numeric_scale,
                genome_only=genome_only,
            )
        scores.append(output["score"].float().cpu().numpy())
        score_logits.append(output["score_logit"].float().cpu().numpy())
        validities.append(output["validity"].float().cpu().numpy())
        validity_logits.append(output["validity_logit"].float().cpu().numpy())
    if not scores:
        raise ValueError("validation split has no examples")
    score_values = np.concatenate(scores)
    score_logit_values = np.concatenate(score_logits)
    validity_values = np.concatenate(validities)
    validity_logit_values = np.concatenate(validity_logits)
    targets = np.asarray(corpus.percentiles[indexes], dtype=np.float64)
    validity_targets = np.asarray(corpus.validities[indexes], dtype=np.float64)
    datasets = np.asarray(corpus.dataset_indices[indexes], dtype=np.int64)
    metrics: dict[str, Any] = quality_metrics(
        score_values,
        validity_values,
        targets,
        validity_targets,
        datasets,
        train_constant,
    )
    losses = score_losses(
        {
            "score": torch.from_numpy(score_values),
            "score_logit": torch.from_numpy(score_logit_values),
            "validity_logit": torch.from_numpy(validity_logit_values),
        },
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(validity_targets.astype(np.float32)),
        torch.from_numpy(datasets),
        validity_positive_weight=_validity_positive_weight(corpus, heldout),
    )
    metrics["losses"] = {name: float(value) for name, value in losses.items()}
    metrics["examples"] = int(len(indexes))
    metrics["split_name"] = "dataset-held-out test" if heldout else ("train", "validation", "test")[split]
    return metrics


