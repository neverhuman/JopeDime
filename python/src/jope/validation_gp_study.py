"""Validation-only Jain GP study runner.

This module is intentionally separate from ``gp_factory``. It runs a fixed
search protocol over a small validation set, keeps search stages test-final-only,
then performs exact final refits after robust-CV selection is frozen.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data.active_v3 import robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .prime_config import load_config
from .prime_staging import stage_read_only_dataset
from .propose import embedding_from_vs, target_summary_from_vs

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


class ValidationGpStudy:
    def __init__(self, config: StudyConfig) -> None:
        self.cfg = config
        if not self.cfg.study_tag:
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self.cfg.study_tag = f"validation_gp_{stamp}"
        self.prime_config = load_config()
        self.source_root = Path(self.cfg.root)
        self.root = (
            self.prime_config.cache / "validation_gp_datasets" / self.cfg.study_tag
        )
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
                except Exception as exc:  # noqa: BLE001
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

    def _run_dataset(self, dataset: str, host: str, slot: int) -> dict[str, Any]:
        t0 = time.time()
        ds_dir = self.out_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        stage_read_only_dataset(
            self.source_root / dataset,
            config=self.prime_config,
            destination=self.root / dataset,
        )
        entry = self._dataset_entry(dataset)
        self.event_log.emit(dataset, "pipeline", "start", host=host, slot=slot)

        seeds, proposal_rows = self._seed_pool(dataset, entry)
        warm_model = self._model_name(dataset, "warm60")
        warm_report = self._run_jain_stage(
            dataset,
            host,
            "warm60",
            warm_model,
            self.cfg.warm_secs,
            "v1",
            seeds,
            ds_dir,
            calib=False,
        )
        warm_genomes = extract_report_genomes(warm_report)
        main_seeds = dedupe_strings(seeds + warm_genomes)[: self.cfg.seed_cap]
        main_model = self._model_name(dataset, "main300")
        main_report = self._run_jain_stage(
            dataset,
            host,
            "main300",
            main_model,
            self.cfg.main_secs,
            "v2",
            main_seeds,
            ds_dir,
            calib=True,
        )
        candidates = extract_report_genomes(main_report)[: self.cfg.top_k]
        robust_rows = self._robust_rerank(dataset, entry, candidates)
        if not robust_rows:
            raise RuntimeError(f"{dataset}: no robust-CV candidates survived")
        single = robust_rows[0]
        members = robust_rows[: min(3, len(robust_rows))]
        downgrade = None if len(members) == 3 else f"top3_downgraded_to_{len(members)}"
        final = self._final_exact_fits(dataset, entry, single, members)
        incumbent = self._incumbent(dataset)
        summary = {
            "dataset": dataset,
            "status": "completed",
            "host": host,
            "slot": slot,
            "single_primary": {
                "genome": single["genome"],
                "model_name": final["single"]["model_name"],
                "rmse": final["single"]["metrics"].get("rmse"),
                "r2": final["single"]["metrics"].get("r2"),
                "robust_rank": 1,
                "robust_score": single.get("robust_score"),
            },
            "ensemble_secondary": {
                "members": [m["genome"] for m in members],
                "member_count": len(members),
                "downgrade": downgrade,
                "rmse": final["ensemble"]["metrics"].get("rmse"),
                "r2": final["ensemble"]["metrics"].get("r2"),
                "predictions_path": final["ensemble"]["predictions_path"],
            },
            "incumbent": incumbent,
            "beat_single": beat_flag(final["single"]["metrics"].get("rmse"), incumbent.get("rmse")),
            "beat_ensemble": beat_flag(final["ensemble"]["metrics"].get("rmse"), incumbent.get("rmse")),
            "proposal_count": len(proposal_rows),
            "seed_count": len(seeds),
            "warm_model": warm_model,
            "main_model": main_model,
            "elapsed_secs": round(time.time() - t0, 3),
        }
        (ds_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self._append_jsonl(self.per_dataset_path, summary)
        self.event_log.emit(dataset, "pipeline", "done", elapsed_secs=summary["elapsed_secs"])
        return summary

    def _seed_pool(self, dataset: str, entry: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        proposals = self._fresh_proposals(dataset, entry)
        top_proposals = proposals[: self.cfg.proposal_keep]
        seeds: list[str] = []
        seed_rows: list[dict[str, Any]] = []

        def add(genome: str, source: str, **meta: Any) -> None:
            canonical = canonicalize_featmath(genome)
            if canonical is None or canonical in seeds or len(seeds) >= self.cfg.seed_cap:
                return
            seeds.append(canonical)
            row = {"dataset": dataset, "source": source, "genome": canonical, **meta}
            seed_rows.append(row)
            self._append_jsonl(self.seed_pool_path, row)

        for row in top_proposals:
            add(
                str(row["genome"]),
                "jope_gen31_proposal",
                selection_score=row.get("selection_score"),
                selection_key=row.get("selection_key"),
                reward=row.get("reward"),
                rank=row.get("rank"),
            )
        for row in self._historical_seed_rows(dataset):
            add(str(row["genome"]), str(row["source"]), rank=row.get("rank"), report=row.get("report"))

        self.event_log.emit(
            dataset,
            "seeds",
            "done",
            proposal_kept=len(top_proposals),
            seed_count=len(seeds),
            historical_count=sum(1 for row in seed_rows if row["source"] != "jope_gen31_proposal"),
        )
        return seeds, proposals

    def _fresh_proposals(self, dataset: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        import torch

        from .policy.decode import propose_v3
        from .policy.model_v3 import load_policy_v3

        if self._model is None:
            with _MODEL_LOCK:
                if self._model is None:
                    state = torch.load(self.cfg.checkpoint, map_location="cpu", weights_only=False)
                    self._model = load_policy_v3(state)
        router = local_router()
        emb = embedding_from_vs(entry["vs_path"], self.standardization)
        target = target_summary_from_vs(entry.get("train_target_vs"))
        with _MODEL_LOCK:
            selected, all_candidates = propose_v3(
                emb,
                target_summary=target,
                model=self._model,
                router=router,
                n=self.cfg.proposal_count,
                engines=("featmath",),
                task="regression",
                samples_per_engine=max(self.cfg.proposal_count, 32),
                seed=self.cfg.seed ^ int(dataset[:8], 16),
                temperature=0.7,
                return_all_candidates=True,
                device="cpu",
            )
        del selected
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, row in enumerate(all_candidates):
            if row.get("engine") != "featmath":
                continue
            genome = row.get("canonical_genome") or row.get("genome")
            if not isinstance(genome, str):
                continue
            canonical = canonicalize_featmath(genome)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            out = {
                "dataset": dataset,
                "rank": len(rows),
                "raw_rank": rank,
                "genome": canonical,
                "selection_score": row.get("selection_score"),
                "selection_key": row.get("selection_key"),
                "reward": row.get("reward"),
                "reward_std": row.get("reward_std"),
                "p_bad": row.get("p_bad"),
                "head_type": row.get("head_type"),
            }
            rows.append(out)
            if len(rows) >= self.cfg.proposal_count:
                break
        rows.sort(key=proposal_sort_key, reverse=True)
        for rank, row in enumerate(rows):
            row["rank"] = rank
            self._append_jsonl(self.proposal_rows_path, row)
        self.event_log.emit(dataset, "proposals", "done", generated=len(rows), kept=self.cfg.proposal_keep)
        return rows

    def _historical_seed_rows(self, dataset: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for model in HISTORICAL_MODELS:
            path = self.source_root / dataset / "reports" / f"{model}.json"
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for rank, genome in enumerate(extract_report_genomes(report)):
                out.append({"source": model, "rank": rank, "genome": genome, "report": str(path)})
        return out

    def _run_jain_stage(
        self,
        dataset: str,
        host: str,
        stage: str,
        model_name: str,
        budget_secs: int,
        fitness: str,
        seeds: list[str],
        ds_dir: Path,
        *,
        calib: bool,
    ) -> dict[str, Any]:
        stage_dir = self.out_dir / stage / dataset
        stage_dir.mkdir(parents=True, exist_ok=True)
        seeds_path = stage_dir / "seeds.txt"
        seeds_path.write_text("\n".join(seeds) + ("\n" if seeds else ""), encoding="utf-8")
        status_path = stage_dir / "status.jsonl"
        log_path = stage_dir / "jain.log"
        report_path = self.root / dataset / "reports" / f"{model_name}.json"
        report_path.unlink(missing_ok=True)
        cmd = self._jain_cmd(dataset, model_name, budget_secs, fitness, seeds_path, status_path, calib=calib)
        self.event_log.emit(
            dataset,
            stage,
            "start",
            host=host,
            model_name=model_name,
            budget_secs=budget_secs,
            fitness=fitness,
            seed_count=len(seeds),
        )
        if is_local_host(host):
            rc = run_logged(cmd, log_path)
        else:
            rc = self._run_remote_jain(host, dataset, stage, model_name, seeds_path, log_path, status_path, cmd)
        if rc != 0:
            raise RuntimeError(f"{stage} Jain failed for {dataset} on {host}; rc={rc}; log={log_path}")
        if not report_path.exists():
            raise RuntimeError(f"{stage} report missing for {dataset}: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pulled = stage_dir / "report.json"
        pulled.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        assert_search_report_honesty(report, dataset, stage)
        self.event_log.emit(
            dataset,
            stage,
            "done",
            model_name=model_name,
            cv_oof=(report.get("cv") or {}).get("oof_r2"),
            finalists=len(report.get("finalists") or []),
            report=str(pulled),
        )
        return report

    def _jain_cmd(
        self,
        dataset: str,
        model_name: str,
        budget_secs: int,
        fitness: str,
        seeds_path: Path,
        status_path: Path,
        *,
        calib: bool,
    ) -> list[str]:
        cmd = [
            str(self.cfg.jain_bin),
            "jain-regressor",
            "--root",
            str(self.root),
            "--dataset",
            dataset,
            "--time-budget-secs",
            str(budget_secs),
            "--pop",
            str(self.cfg.pop),
            "--gens",
            str(self.cfg.gens),
            "--top-k",
            str(self.cfg.top_k),
            "--folds",
            str(self.cfg.search_folds),
            "--search-rows",
            str(self.cfg.search_rows),
            "--fitness",
            fitness,
            "--gpu-kernel-configs",
            str(self.cfg.gpu_kernel_configs),
            "--gpu-cache-dir",
            str(_PRIME_CONFIG.gpu_cache),
            "--threads",
            str(self.cfg.threads_per_worker),
            "--seed",
            str(self.cfg.seed ^ int(dataset[:8], 16) ^ budget_secs),
            "--seed-genomes-file",
            str(seeds_path),
            "--model-name",
            model_name,
            "--overwrite",
            "--no-predictions",
            "--skip-holdout-metrics",
            "--status-jsonl",
            str(status_path),
        ]
        if calib:
            cmd.append("--calib")
        return cmd

    def _run_remote_jain(
        self,
        host: str,
        dataset: str,
        stage: str,
        model_name: str,
        seeds_path: Path,
        log_path: Path,
        status_path: Path,
        local_cmd: list[str],
    ) -> int:
        remote_dir = (
            f"{_PRIME_CONFIG.remote_scratch_root}/validation_gp/"
            f"{self.cfg.study_tag}/{dataset}/{stage}"
        )
        remote_seeds = f"{remote_dir}/seeds.txt"
        remote_status = f"{remote_dir}/status.jsonl"
        remote_cmd = list(local_cmd)
        remote_cmd[remote_cmd.index(str(seeds_path))] = remote_seeds
        remote_cmd[remote_cmd.index(str(status_path))] = remote_status
        report = self.root / dataset / "reports" / f"{model_name}.json"
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                f"mkdir -p {remote_dir} {self.root}/{dataset}/reports {self.root}/{dataset}/outputs",
            ],
            check=True,
        )
        subprocess.run(
            [
                "rsync",
                "-a",
                "--exclude",
                "reports/",
                "--exclude",
                "outputs/",
                f"{self.root / dataset}/",
                f"{host}:{self.root}/{dataset}/",
            ],
            check=True,
        )
        subprocess.run(["rsync", "-a", str(seeds_path), f"{host}:{remote_seeds}"], check=True)
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    host,
                    f"rm -f {report}; nice -n 5 {shell_join(remote_cmd)}",
                ],
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        subprocess.run(["rsync", "-a", f"{host}:{remote_status}", str(status_path)], check=False)
        subprocess.run(["rsync", "-a", f"{host}:{report}", str(report.parent) + "/"], check=False)
        return proc.returncode

    def _robust_rerank(self, dataset: str, entry: dict[str, Any], candidates: list[str]) -> list[dict[str, Any]]:
        rows = [
            {
                "engine": "featmath",
                "dataset_hash": dataset,
                "train_csv": entry["train_csv"],
                "genome": genome,
                "folds": self.cfg.robust_folds,
            }
            for genome in candidates
        ]
        ledger = self.out_dir / "robust_cv_ledger.jsonl"
        results: dict[int, dict[str, Any]] = {}
        for idx, res in iter_eval_regression_with_ledger(
            rows,
            ledger=str(ledger),
            router=local_router(),
            workers=self.cfg.robust_workers,
        ):
            results[idx] = res
        out: list[dict[str, Any]] = []
        for idx, genome in enumerate(candidates):
            res = results.get(idx) or {}
            cv = res.get("cv") if isinstance(res.get("cv"), dict) else {}
            folds = [float(v) for v in (cv.get("fold_r2") or []) if finite(v)]
            oof = cv.get("oof_r2")
            score = robust_score(float(oof), folds) if finite(oof) else None
            row = {
                "dataset": dataset,
                "rank_input": idx,
                "genome": genome,
                "oof_r2": oof,
                "fold_r2": folds,
                "robust_score": score,
                "lcb95": lcb95(folds),
                "min_fold_r2": min(folds) if folds else None,
                "cost_ms": (res.get("timings") or {}).get("total_ms"),
                "cache_hit": bool(res.get("cache_hit")),
                "errors": res.get("errors") or [],
            }
            out.append(row)
            self._append_jsonl(self.robust_path, row)
        ranked = [row for row in out if finite(row.get("robust_score"))]
        ranked.sort(key=robust_sort_key)
        for rank, row in enumerate(ranked, 1):
            row["robust_rank"] = rank
        self.event_log.emit(dataset, "robust_cv", "done", candidates=len(candidates), ranked=len(ranked))
        return ranked

    def _final_exact_fits(
        self,
        dataset: str,
        entry: dict[str, Any],
        single: dict[str, Any],
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        final_dir = self.out_dir / "final_exact_fits" / dataset
        pred_dir = self.out_dir / "predictions" / dataset
        final_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        fits: list[dict[str, Any]] = []
        single_name = self._model_name(dataset, "single_primary")
        single_fit = self._fit_genome(dataset, single["genome"], single_name, final_dir, pred_dir)
        for idx, row in enumerate(members, 1):
            name = self._model_name(dataset, f"ensemble_m{idx}")
            fits.append(self._fit_genome(dataset, row["genome"], name, final_dir, pred_dir))
        ensemble_path, ensemble_metrics = self._ensemble_predictions(dataset, entry, fits, pred_dir)
        return {
            "single": single_fit,
            "members": fits,
            "ensemble": {
                "predictions_path": str(ensemble_path),
                "metrics": ensemble_metrics,
                "member_count": len(fits),
            },
        }

    def _fit_genome(
        self,
        dataset: str,
        genome: str,
        model_name: str,
        final_dir: Path,
        pred_dir: Path,
    ) -> dict[str, Any]:
        cmd = [
            str(self.cfg.jain_bin),
            "fit-genome",
            "--root",
            str(self.root),
            "--dataset",
            dataset,
            "--genome",
            genome,
            "--model-name",
            model_name,
            "--overwrite",
        ]
        log_path = final_dir / f"{model_name}.log"
        rc = run_logged(cmd, log_path)
        if rc != 0:
            raise RuntimeError(f"fit-genome failed for {dataset} {model_name}; rc={rc}; log={log_path}")
        report_src = self.root / dataset / "reports" / f"{model_name}.json"
        pred_src = self.root / dataset / "predictions" / f"{model_name}.csv"
        if not report_src.exists() or not pred_src.exists():
            raise RuntimeError(f"fit-genome missing report or predictions for {dataset} {model_name}")
        report_dst = final_dir / f"{model_name}.json"
        pred_dst = pred_dir / f"{model_name}.csv"
        shutil.copy2(report_src, report_dst)
        shutil.copy2(pred_src, pred_dst)
        report = json.loads(report_dst.read_text(encoding="utf-8"))
        return {
            "model_name": model_name,
            "genome": genome,
            "report_path": str(report_dst),
            "predictions_path": str(pred_dst),
            "metrics": report.get("metrics") or {},
        }

    def _ensemble_predictions(
        self,
        dataset: str,
        entry: dict[str, Any],
        fits: list[dict[str, Any]],
        pred_dir: Path,
    ) -> tuple[Path, dict[str, float]]:
        if not fits:
            raise RuntimeError(f"{dataset}: no ensemble members")
        preds = [read_prediction_file(Path(fit["predictions_path"])) for fit in fits]
        n = len(preds[0])
        for fit, values in zip(fits, preds, strict=True):
            if len(values) != n:
                raise RuntimeError(
                    f"{dataset}: prediction row-count mismatch for {fit['model_name']}: {len(values)} vs {n}"
                )
        mean_preds = [sum(values[i] for values in preds) / len(preds) for i in range(n)]
        y = read_targets(Path(entry["test_csv"]))
        if len(y) != n:
            raise RuntimeError(f"{dataset}: ensemble predictions {n} rows but test target has {len(y)} rows")
        path = pred_dir / f"{self._model_name(dataset, 'ensemble_secondary')}.csv"
        path.write_text("\n".join(f"{p:.17g}" for p in mean_preds) + "\n", encoding="utf-8")
        return path, regression_metrics(y, mean_preds)

    def _incumbent(self, dataset: str) -> dict[str, Any]:
        best: dict[str, Any] = {"model": None, "report": None, "rmse": None, "r2": None}
        reports = sorted((self.source_root / dataset / "reports").glob("*.json"))
        for path in reports:
            if path.stem.startswith(self.cfg.study_tag):
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
            rmse = metrics.get("rmse")
            if not finite(rmse):
                continue
            if best["rmse"] is None or float(rmse) < float(best["rmse"]):
                best = {
                    "model": path.stem,
                    "report": str(path),
                    "rmse": float(rmse),
                    "r2": metrics.get("r2"),
                }
        return best

    def _dataset_entry(self, dataset: str) -> dict[str, Any]:
        entry = self.manifest.get(dataset)
        if not isinstance(entry, dict):
            raise ValueError(f"{dataset} missing from split_manifest")
        for key in ("train_csv", "test_csv", "vs_path"):
            if not entry.get(key):
                raise ValueError(f"{dataset} missing {key} in split_manifest")
            if not Path(entry[key]).exists():
                raise ValueError(f"{dataset} {key} does not exist: {entry[key]}")
        return entry

    def _model_name(self, dataset: str, suffix: str) -> str:
        return f"{self.cfg.study_tag}__{dataset}__{suffix}"

    def _host_slots(self) -> list[tuple[str, int]]:
        slots: list[tuple[str, int]] = []
        for host in self.cfg.hosts:
            for slot in range(self.cfg.workers_per_host):
                slots.append((host, slot))
        return slots

    def _validate_inputs(self) -> None:
        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(self.cfg.checkpoint)
        if not Path(self.cfg.jain_bin).exists():
            raise FileNotFoundError(self.cfg.jain_bin)
        for dataset in self.cfg.datasets:
            self._dataset_entry(dataset)

    def _write_config(self) -> None:
        (self.out_dir / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_incumbent_compare(self, results: list[dict[str, Any]]) -> None:
        with self.compare_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "dataset",
                    "incumbent_model",
                    "incumbent_report",
                    "incumbent_rmse",
                    "single_rmse",
                    "ensemble_rmse",
                    "single_beat",
                    "ensemble_beat",
                    "single_delta_abs",
                    "ensemble_delta_abs",
                    "single_delta_pct",
                    "ensemble_delta_pct",
                ],
            )
            writer.writeheader()
            for row in results:
                inc = row.get("incumbent") or {}
                single_rmse = ((row.get("single_primary") or {}).get("rmse"))
                ensemble_rmse = ((row.get("ensemble_secondary") or {}).get("rmse"))
                inc_rmse = inc.get("rmse")
                writer.writerow(
                    {
                        "dataset": row.get("dataset"),
                        "incumbent_model": inc.get("model"),
                        "incumbent_report": inc.get("report"),
                        "incumbent_rmse": inc_rmse,
                        "single_rmse": single_rmse,
                        "ensemble_rmse": ensemble_rmse,
                        "single_beat": beat_flag(single_rmse, inc_rmse),
                        "ensemble_beat": beat_flag(ensemble_rmse, inc_rmse),
                        "single_delta_abs": delta_abs(single_rmse, inc_rmse),
                        "ensemble_delta_abs": delta_abs(ensemble_rmse, inc_rmse),
                        "single_delta_pct": delta_pct(single_rmse, inc_rmse),
                        "ensemble_delta_pct": delta_pct(ensemble_rmse, inc_rmse),
                    }
                )

    def _summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [r for r in results if r.get("status") == "completed"]
        single_wins = sum(1 for r in completed if r.get("beat_single") is True)
        ensemble_wins = sum(1 for r in completed if r.get("beat_ensemble") is True)
        return {
            "study_tag": self.cfg.study_tag,
            "out_dir": str(self.out_dir),
            "datasets": len(results),
            "completed": len(completed),
            "errors": len(results) - len(completed),
            "single_primary_wins": single_wins,
            "ensemble_secondary_wins": ensemble_wins,
            "did_single_win_any": single_wins > 0,
            "did_ensemble_win_any": ensemble_wins > 0,
            "incumbent_compare_csv": str(self.compare_path),
        }

    def _write_summary(self, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
        (self.out_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "datasets": results}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        readme = [
            f"# {self.cfg.study_tag}",
            "",
            "Validation GP study with search-only Jain stages and final exact-genome scoring.",
            "",
            f"- datasets: {summary.get('datasets')}",
            f"- completed: {summary.get('completed')}",
            f"- single_primary wins: {summary.get('single_primary_wins')}",
            f"- ensemble_secondary wins: {summary.get('ensemble_secondary_wins')}",
            "",
            "Primary choice is `single_primary`; `ensemble_secondary` is reported as the fixed secondary arm.",
        ]
        (self.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_lock:
            with path.open("a", encoding="utf-8") as fh:
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
