from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .validation_gp_study import (
    _MODEL_LOCK,
    _PRIME_CONFIG,
    HISTORICAL_MODELS,
    assert_search_report_honesty,
    beat_flag,
    canonicalize_featmath,
    dedupe_strings,
    embedding_from_vs,
    extract_report_genomes,
    is_local_host,
    local_router,
    proposal_sort_key,
    run_logged,
    shell_join,
    stage_read_only_dataset,
    target_summary_from_vs,
)


class _StudyExecution:
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
            rc = self._run_remote_jain(
                host, dataset, stage, model_name, seeds_path, log_path, status_path, cmd
            )
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
            f"{_PRIME_CONFIG.remote_scratch_root}/validation_gp/{self.cfg.study_tag}/{dataset}/{stage}"
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
