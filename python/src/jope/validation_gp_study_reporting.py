from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .validation_gp_study import (
    beat_flag,
    delta_abs,
    delta_pct,
)


class _StudyReporting:
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
                single_rmse = (row.get("single_primary") or {}).get("rmse")
                ensemble_rmse = (row.get("ensemble_secondary") or {}).get("rmse")
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
