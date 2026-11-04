from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .corpus import (
    TokenizedCorpus,
    prepare_corpus,
)
from .model import DimeScoreModel, partial_warm_start
from .train_data import _atomic_json

if TYPE_CHECKING:
    from .train import TrainConfig


def _checkpoint(
    model: DimeScoreModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    corpus: TokenizedCorpus,
    embedding_manifest: dict[str, Any],
    config: TrainConfig,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    warm_start_report: dict[str, Any] | None,
    *,
    genomes_per_dataset: int,
    loader_workers: int,
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "model_version": "dime-score-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "tokens": corpus.index["tokens"],
        "action_types": corpus.index["action_types"],
        "numeric_stats": corpus.index["numeric_stats"],
        "grammar_contract_sha256": corpus.index["gene_schema_contract_sha256"],
        "tokenized_corpus_binding_sha256": corpus.manifest["binding_sha256"],
        "embedding_sha256": embedding_manifest["embeddings_sha256"],
        "embedding_frozen": embedding_manifest["frozen"],
        "architecture": {
            "dataset_dim": model.dataset_dim,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "ffn_dim": model.ffn_dim,
            "max_sequence_length": model.max_sequence_length,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "warm_start_report": warm_start_report,
        "data_loading": {
            "genomes_per_dataset": genomes_per_dataset,
            "loader_workers": loader_workers,
            "prefetch_factor": config.prefetch_factor,
            "execution_mode": execution_mode,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _restore_rng(checkpoint: dict[str, Any]) -> None:
    rng = checkpoint.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.random.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def warm_start_preflight(config: TrainConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "jope-prime.dime-warm-start-preflight.v1",
        "checkpoint": str(config.warm_start) if config.warm_start else None,
        "eligible": False,
        "reason": "no warm-start checkpoint supplied",
    }
    if config.warm_start is not None:
        tokenized_root = prepare_corpus(config.corpus_root)
        corpus = TokenizedCorpus(tokenized_root)
        model = DimeScoreModel(
            vocabulary_size=len(corpus.index["tokens"]),
            action_type_count=len(corpus.index["action_types"]),
            max_sequence_length=int(corpus.manifest["max_sequence_length"]),
        )
        warm = torch.load(config.warm_start, map_location="cpu", weights_only=False)
        mapping = partial_warm_start(model, warm, corpus.index["tokens"])
        mapped = int(mapping.get("loaded_tensor_count", 0))
        report.update(
            {
                "mapping": mapping,
                "eligible": mapped > 0,
                "reason": (
                    f"{mapped} exact-shape tensors mapped" if mapped else "zero exact-shape tensors mapped"
                ),
            }
        )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(config.run_dir / "warm-start-preflight.v1.json", report)
    return report
