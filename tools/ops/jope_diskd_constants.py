from __future__ import annotations

from pathlib import Path

GIB = 1024**3
SAFE_CACHE_PATHS = {
    "uv-cache": Path(".cache/uv"),
    "pip-cache": Path(".cache/pip"),
}
SUPPORTED_LOCAL_HOSTS = frozenset({"xbabe1", "xbabe2", "xbabe3"})
CACHE_METADATA_NAMES = {"cachedir.tag", ".gitignore", ".lock", "lock"}
REBUILDABLE_WORKTREE_IGNORED_DIRS = {
    "__pycache__", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".tox", ".venv", "node_modules", "venv",
}
REBUILDABLE_WORKTREE_IGNORED_FILES = {"AGENT_CHAT.lock", "AGENT_CHAT.md.lock"}
LOG_LIMIT_BYTES = 256 * 1024**2
LOG_RETENTION_SECONDS = 14 * 24 * 3600
TARGET_RETENTION_SECONDS = 24 * 3600
WORKTREE_IDLE_SECONDS = 2 * 3600
OFFLOAD_BUCKET = "veox-jopedime-learning-use2"
OFFLOAD_PREFIX = "offload"
TARGET_EVIDENCE_PARTS = {
    "checkpoint", "checkpoints", "evidence", "health", "receipt", "receipts",
    "report", "reports", "score", "scores", "sealed", "holdout", "flywheel",
}
TARGET_EVIDENCE_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".hdf5", ".joblib", ".npy", ".onnx", ".pkl",
    ".pickle", ".pt", ".pth", ".safetensors", ".tfrecord", ".tfrecords",
}
CARGO_INCREMENTAL_BINARY_CACHE_NAMES = {
    "dep-graph.bin", "query-cache.bin", "work-products.bin",
}
PROTECTED_PARTS = {
    "admiralty",
    "checkpoints",
    "custody",
    "datasets",
    "evidence",
    "guestd",
    "flywheel",
    "holdout",
    "jeryu",
    "jope-data",
    "jope-rescue-20260921",
    "jope_plan",
    "jope-plan",
    "receipts",
    "sealed",
    "veox",
}
PROTECTED_PREFIXES = {
    "admiralty",
    "checkpoint",
    "datasets",
    "evidence",
    "flywheel",
    "holdout",
    "jeryu",
    "jope-data",
    "jope-plan",
    "jope-rescue-20260921",
    "jope_plan",
    "receipt",
    "sealed",
    "veox",
    "guestd",
}
