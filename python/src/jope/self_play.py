"""Self-play generation loop: each new model must beat ALL former model attempts.

Per generation g:
  TRAIN    2-node DDP (xbabe2 rank0 local + xbabe3 rank1 ssh) on corpus ∪ self_rows_g*
  PROPOSE  new checkpoint picks per train dataset: 1 deployed top-1 + E exploration samples
  EVAL     cluster CPU eval of all picks (ledger-cached)
  APPEND   rows provenance="self_g<g>"; per-dataset self-best bar rises monotonically
  GATE     beat_rate vs generation g-1 (ship iff ≥ threshold), new-bests, mean rank-pct
  BENCH    elo-bench top-1 on the held-out validation ladder

State/telemetry (TUI-tailable, atomic):
  <run_dir>/self_play_state.json     current gen + phase + progress
  <run_dir>/self_play_metrics.jsonl  one row per generation (the learning curve)
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .data.active_v3 import robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .io import read_jsonl
from .prime_config import load_config
from .propose import embedding_from_vs, target_summary_from_vs
from .schema import TrainingRow

PROJ = load_config().root
NCCL_ENV = (
    "CUDA_VISIBLE_DEVICES=0 NCCL_SOCKET_IFNAME=enp37s0f1 GLOO_SOCKET_IFNAME=enp37s0f1 "
    "NCCL_IB_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src"
)
MASTER_ADDR = "192.168.68.87"  # xbabe2 (this host)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


from .self_play_selection import (
    _candidate_pool_id,
    _candidate_pool_record,
    _choose_deployed_candidate,
    _finite_float,
    _proposal_failure_exceeded,
    _resolve_ship_decision,
    deployment_elo,
)
from .self_play_training import SelfPlayTrainingMixin
from .self_play_evaluation import SelfPlayEvaluationMixin


class SelfPlay(SelfPlayTrainingMixin, SelfPlayEvaluationMixin):
    """Keep the public orchestrator while grouping its implementation by phase."""


def run_self_play(art_dir: str, run_dir: str, generations: int, **kw) -> None:
    SelfPlay(art_dir, run_dir, **kw).run(generations)
