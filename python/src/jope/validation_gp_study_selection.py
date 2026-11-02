from __future__ import annotations

from .validation_gp_study import *
from .validation_gp_study import _MODEL_LOCK, _PRIME_CONFIG


class _StudySelection:
    def _robust_rerank(
        self, dataset: str, entry: dict[str, Any], candidates: list[str]
    ) -> list[dict[str, Any]]:
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
