"""Host and run configuration for the GP factory."""

from __future__ import annotations

import os

from .prime_config import load_config

# Fresh policy-model proposals to seed the GP with (idea: warm-start gen-0 with what the
# current model thinks is good, not just historical champions). These live in the self-play
# run's per-generation files; we scan the most recent few for the slice's datasets.
SELF_ROWS_GLOB = "self_rows_*_g*.jsonl"
SELF_ROWS_GLOBS = ("self_rows_g*.jsonl", SELF_ROWS_GLOB)
MODEL_SEED_RECENT = 3

GP_HOST = "xbabe2"
# Secondary worker host: xbabe3's spare cores (3 workers x 24 threads = 72, nice-5;
# it also serves DDP rank1 + eval bursts, which win contention at normal priority).
GP_HOST2 = "xbabe3"
GP_HOST2_WORKERS = 5
_PRIME_CONFIG = load_config()


def _gp_host_map(default_workers: int) -> dict[str, int]:
    """host→worker-count map; override via JOPE_GP_HOSTS="xbabe1:4,xbabe3:4,xbabe2:2"."""
    spec = os.environ.get("JOPE_GP_HOSTS", "").strip()
    if not spec:
        return {GP_HOST: default_workers, GP_HOST2: GP_HOST2_WORKERS}
    out: dict[str, int] = {}
    for part in spec.split(","):
        host, _, n = part.strip().partition(":")
        if host:
            out[host] = max(0, int(n or default_workers))
    return out or {GP_HOST: default_workers}


JAIN_BIN = str(_PRIME_CONFIG.jain_binary)
GPU_CACHE = str(_PRIME_CONFIG.gpu_cache)
REG_ROOT = str(_PRIME_CONFIG.regression_dataset_root)
GP_WORK_ROOT = _PRIME_CONFIG.cache / "gp_datasets"
MODEL_NAME = "jope_gp_v1"

# Game mode: per-dataset model bars + model seeds come from the NEWEST self-play candidate
# pool file (every proposal the model made last generation, measured or not) — not the
# self_rows scan, which only carries the evaluated top picks.
CANDIDATE_POOLS_GLOB = str(
    _PRIME_CONFIG.legacy_root / "runs/self_play_v33/candidate_pools/candidate_pools_g*.jsonl"
)
