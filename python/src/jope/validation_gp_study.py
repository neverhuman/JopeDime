"""Validation-only Jain GP study runner.

This module is intentionally separate from ``gp_factory``. It runs a fixed
search protocol over a small validation set, keeps search stages test-final-only,
then performs exact final refits after robust-CV selection is frozen.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data.active_v3 import robust_score as robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger as iter_eval_regression_with_ledger
from .prime_config import load_config
from .prime_staging import stage_read_only_dataset as stage_read_only_dataset
from .propose import embedding_from_vs as embedding_from_vs
from .propose import target_summary_from_vs as target_summary_from_vs

DEFAULT_DATASETS = (
    "ab0239b9ef0ba28e",
    "ee501c454aa21268",
    "dd89601cec1fc2a5",
    "d28cb31a17cb67d6",
    "dd16ac872ffb8673",
    "7a445e1b87747a58",
    "14bc0936c4d80456",
    "877665868fca8cce",
    "305ee367881623f4",
)
_PRIME_CONFIG = load_config()
DEFAULT_ART_DIR = _PRIME_CONFIG.artifacts
DEFAULT_ROOT = _PRIME_CONFIG.regression_dataset_root
DEFAULT_CHECKPOINT = _PRIME_CONFIG.warm_start_checkpoint
DEFAULT_JAIN_BIN = _PRIME_CONFIG.jain_binary
DEFAULT_RUN_DIR = _PRIME_CONFIG.runs / "validation_gp_study"
DEFAULT_HOSTS = ("xbabe1", "xbabe2", "xbabe3")
HISTORICAL_MODELS = ("jain_regressor_300", "jain_regressor_fast30", "jope_gp_v1")

_MODEL_LOCK = threading.Lock()
_TLS = threading.local()


@dataclass
class StudyConfig:
    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    art_dir: str = str(DEFAULT_ART_DIR)
    root: str = str(DEFAULT_ROOT)
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    jain_bin: str = str(DEFAULT_JAIN_BIN)
    run_dir: str = str(DEFAULT_RUN_DIR)
    study_tag: str = ""
    hosts: list[str] = field(default_factory=lambda: list(DEFAULT_HOSTS))
    workers_per_host: int = 3
    threads_per_worker: int = 24
    proposal_count: int = 32
    proposal_keep: int = 8
    seed_cap: int = 64
    warm_secs: int = 60
    main_secs: int = 300
    pop: int = 128
    gens: int = 5000
    top_k: int = 10
    search_folds: int = 5
    search_rows: int = 768
    robust_folds: int = 8
    robust_workers: int = 8
    gpu_kernel_configs: int = 0
    seed: int = 20260710
    dry_run: bool = False


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, dataset: str | None, stage: str, event: str, **payload: Any) -> None:
        ts = time.time()
        row = {
            "ts": ts,
            "iso": datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds"),
            "dataset": dataset,
            "stage": stage,
            "event": event,
            **payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def local_router() -> EngineRouter:
    router = getattr(_TLS, "router", None)
    if router is None:
        router = EngineRouter(hosts=("local",))
        _TLS.router = router
    return router


def extract_report_genomes(report: dict[str, Any]) -> list[str]:
    genomes: list[str] = []
    for genome in [report.get("genome"), report.get("gene")]:
        if isinstance(genome, str):
            genomes.append(genome)
    for finalist in report.get("finalists") or []:
        if isinstance(finalist, dict):
            genome = finalist.get("genome") or finalist.get("gene")
            if isinstance(genome, str):
                genomes.append(genome)
    return dedupe_strings([g for g in genomes if g])


def canonicalize_featmath(genome: str) -> str | None:
    meta = local_router().validate_many([("featmath", genome)])[0]
    if not meta.get("valid"):
        return None
    canonical = meta.get("canonical_genome")
    return canonical if isinstance(canonical, str) else None


def dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def proposal_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    score = row.get("selection_score")
    key = row.get("selection_key")
    reward = row.get("reward")
    return (
        float(score) if finite(score) else float("-inf"),
        float(key) if finite(key) else float("-inf"),
        float(reward) if finite(reward) else float("-inf"),
    )


def robust_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -(float(row["robust_score"]) if finite(row.get("robust_score")) else float("-inf")),
        -(float(row["lcb95"]) if finite(row.get("lcb95")) else float("-inf")),
        -(float(row["min_fold_r2"]) if finite(row.get("min_fold_r2")) else float("-inf")),
        float(row["cost_ms"]) if finite(row.get("cost_ms")) else float("inf"),
    )


def lcb95(values: list[float]) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = float(np.std(vals)) if len(vals) >= 2 else 0.0
    return mean - 1.96 * std / math.sqrt(len(vals))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_targets(path: Path) -> list[float]:
    out: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(line.split(",")[-1]))
            except ValueError as exc:
                if line_no == 1:
                    continue
                raise exc
    return out


def read_prediction_file(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def regression_metrics(y: list[float], pred: list[float]) -> dict[str, float]:
    n = min(len(y), len(pred))
    if n == 0 or len(y) != len(pred):
        raise ValueError(f"target/prediction length mismatch: {len(y)} vs {len(pred)}")
    mse = sum((y[i] - pred[i]) ** 2 for i in range(n)) / n
    rmse = math.sqrt(mse)
    mean_y = sum(y) / n
    ss_tot = sum((value - mean_y) ** 2 for value in y)
    r2 = 1.0 - (mse * n) / ss_tot if ss_tot > 0 else 0.0
    mae = sum(abs(y[i] - pred[i]) for i in range(n)) / n
    return {"rmse": rmse, "r2": r2, "mae": mae, "n": n}


def beat_flag(candidate_rmse: Any, incumbent_rmse: Any) -> bool | None:
    if not finite(candidate_rmse) or not finite(incumbent_rmse):
        return None
    return float(candidate_rmse) < float(incumbent_rmse)


def delta_abs(candidate_rmse: Any, incumbent_rmse: Any) -> float | None:
    if not finite(candidate_rmse) or not finite(incumbent_rmse):
        return None
    return float(incumbent_rmse) - float(candidate_rmse)


def delta_pct(candidate_rmse: Any, incumbent_rmse: Any) -> float | None:
    if not finite(candidate_rmse) or not finite(incumbent_rmse) or float(incumbent_rmse) == 0.0:
        return None
    return 100.0 * (float(incumbent_rmse) - float(candidate_rmse)) / float(incumbent_rmse)


def assert_search_report_honesty(report: dict[str, Any], dataset: str, stage: str) -> None:
    if "metrics" in report:
        raise RuntimeError(f"{dataset} {stage}: search report contains top-level metrics")
    if "predictions_path" in report:
        raise RuntimeError(f"{dataset} {stage}: search report contains predictions_path")
    for finalist in report.get("finalists") or []:
        if isinstance(finalist, dict) and "full_fit" in finalist:
            raise RuntimeError(f"{dataset} {stage}: search finalist contains full_fit holdout diagnostics")


def is_local_host(host: str) -> bool:
    normalized = host.strip().lower()
    local_names = {"", "local", "localhost", "127.0.0.1", socket.gethostname().lower(), "xbabe2"}
    return normalized in local_names


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def shell_join(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in cmd)


def _load_study_mixins() -> tuple[type, type, type]:
    from .validation_gp_study_execution import _StudyExecution
    from .validation_gp_study_reporting import _StudyReporting
    from .validation_gp_study_selection import _StudySelection

    return _StudyExecution, _StudySelection, _StudyReporting


_StudyExecution, _StudySelection, _StudyReporting = _load_study_mixins()
del _load_study_mixins


class ValidationGpStudy(_StudyExecution, _StudySelection, _StudyReporting):
    def __init__(self, config: StudyConfig) -> None:
        self.cfg = config
        if not self.cfg.study_tag:
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self.cfg.study_tag = f"validation_gp_{stamp}"
        self.prime_config = load_config()
        self.source_root = Path(self.cfg.root)
        self.root = self.prime_config.cache / "validation_gp_datasets" / self.cfg.study_tag
        self.art_dir = Path(self.cfg.art_dir)
        self.out_dir = Path(self.cfg.run_dir) / self.cfg.study_tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = EventLog(self.out_dir / "event_log.jsonl")
        self.seed_pool_path = self.out_dir / "seed_pool.jsonl"
        self.proposal_rows_path = self.out_dir / "proposal_rows.jsonl"
        self.robust_path = self.out_dir / "robust_cv_top10.jsonl"
        self.per_dataset_path = self.out_dir / "per_dataset_summary.jsonl"
        self.compare_path = self.out_dir / "incumbent_compare.csv"
        self.manifest = json.loads((self.art_dir / "split_manifest.json").read_text())["datasets"]
        self.standardization = json.loads((self.art_dir / "standardization.json").read_text())
        self._model = None
        self._jsonl_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        self._write_config()
        self._validate_inputs()
        if self.cfg.dry_run:
            summary = {"status": "dry_run", "datasets": self.cfg.datasets, "out_dir": str(self.out_dir)}
            self._write_summary(summary, [])
            return summary

        slots = self._host_slots()
        self.event_log.emit(None, "study", "start", datasets=self.cfg.datasets, slots=slots)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = {}
            for idx, dataset in enumerate(self.cfg.datasets):
                host, slot = slots[idx % len(slots)]
                futures[pool.submit(self._run_dataset, dataset, host, slot)] = dataset
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    results.append(future.result())
                except (Exception, ArithmeticError) as exc:  # noqa: BLE001
                    row = {"dataset": dataset, "status": "error", "error": repr(exc)}
                    self._append_jsonl(self.per_dataset_path, row)
                    self.event_log.emit(dataset, "pipeline", "error", error=repr(exc))
                    results.append(row)

        results.sort(key=lambda r: self.cfg.datasets.index(str(r.get("dataset"))))
        self._write_incumbent_compare(results)
        summary = self._summary(results)
        self._write_summary(summary, results)
        self.event_log.emit(None, "study", "done", summary=summary)
        return summary
