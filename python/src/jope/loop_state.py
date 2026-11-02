"""Loop configuration and persisted run state helpers."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engines import ENGINES


@dataclass
class LoopConfig:
    art_dir: Path
    checkpoint: Path
    run_dir: Path
    ledger: Path
    engines: tuple[str, ...] = ENGINES
    datasets_per_iter: int = 32
    samples_per_dataset_engine: int = 8
    cv_top_m: int = 2
    cv_uncertain_m: int = 1
    max_cv_per_engine: int = 64  # per iteration
    folds: int = 3
    temperature_schedule: tuple[float, ...] = (0.6, 0.8, 1.0)
    refit_reward_every: int = 1
    refit_policy_every: int = 0  # 0 = policy frozen (reward-only loop)
    refit_epochs: int = 2
    refit_lr: float = 1e-4
    refit_batch_size: int = 128
    max_iterations: int | None = None
    max_hours: float | None = None
    patience_iters: int = 8
    min_delta: float = 0.002
    eval_workers: int | None = None
    seed: int = 0
    device: str = "cpu"
    gene_feed_path: Path | None = None  # when set, stream per-gene events for the live TUI


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "iteration": 0,
        "roster_cursor": 0,
        "best_window": [],
        "policy_refits_halted": False,
        "spearman_bad_streak": 0,
    }


def _patience_exhausted(state: dict[str, Any], cfg: LoopConfig) -> bool:
    window = state.get("best_window") or []
    if len(window) < cfg.patience_iters + 1:
        return False
    recent = window[-cfg.patience_iters :]
    baseline = window[-(cfg.patience_iters + 1)]
    return max(recent) < baseline + cfg.min_delta
