"""Validation recorder (TELEMETRY_SPEC rows V1-V7).

A trainer's validation pass calls ``add`` once per evaluated dataset, using the predictions and
metrics it has already computed, then calls ``finish`` once. The recorder aggregates
**within a family first, then equally across families** (``ns/S3_SPLITS.md``), and reports
family and dataset counts beside every aggregate.

* There is no frozen family roster yet, so every record carries
  ``availability: no_frozen_family_roster`` until a roster digest is supplied. Without family
  ids, family weighting is reported as unavailable, not imitated by a per-dataset mean.
* Binary metrics (AUC, logloss, Brier, accuracy, ECE) are secondary diagnostics. Progress is
  reported as Elo against methods (row F2).
* V7: only the partition label already present on the result is read. A sealed label fails
  closed with :class:`SealedPartitionError`; a missing label is counted, never assumed safe.
  The sealed roster and sealed data are never opened.

Pass opaque or hashed dataset and family ids; nothing here needs names.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

RECORD_KIND = "jope.telemetry.validation/v1"
SEALED_LABELS = ("sealed", "meta_test", "meta-test", "sealed_meta_test")
MAX_DATASETS = 4096
TASKS = ("regression", "binary_classification")
METRICS = {
    "regression": ("loss", "r2"),
    "binary_classification": ("loss", "auc", "logloss", "brier", "acc", "ece"),
}

Emit = Callable[[str, dict[str, Any]], Any]


class SealedPartitionError(RuntimeError):
    """An evaluated result was labeled as the sealed partition (row V7, alarm A8)."""


def check_partition_label(label: str | None) -> str:
    """``not_sealed``, ``sealed`` or ``label_missing``, from the label alone (same rule as Rust)."""
    if label is None or not label.strip():
        return "label_missing"
    return "sealed" if label.strip().lower() in SEALED_LABELS else "not_sealed"


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _quantiles(vals: list[float]) -> dict[str, float | None]:
    v = sorted(vals)

    def q(p: float) -> float | None:
        if not v:
            return None
        h = (len(v) - 1) * p
        lo = int(h)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (h - lo) * (v[hi] - v[lo])

    return {
        "p10": q(0.1),
        "p50": q(0.5),
        "p90": q(0.9),
        "min": v[0] if v else None,
        "max": v[-1] if v else None,
    }


class ValidationRecorder:
    def __init__(
        self,
        emit: Emit | None = None,
        family_roster_sha256: str | None = None,
        fold_map_sha256: str | None = None,
        max_datasets: int = MAX_DATASETS,
    ) -> None:
        self.emit = emit
        self.roster = family_roster_sha256
        self.fold_map = fold_map_sha256
        self.max_datasets = int(max_datasets)
        self._reset()

    def _reset(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._dropped = 0
        self._sealed_touch = 0
        self._label_missing = 0

    def add(
        self,
        *,
        dataset: str,
        task: str,
        partition: str | None,
        family: str | None = None,
        fold: int | None = None,
        prediction_mean: float | None = None,
        prediction_std: float | None = None,
        selected_program_sha256: str | None = None,
        selected_in_bank: bool | None = None,
        **metrics: float | None,
    ) -> None:
        verdict = check_partition_label(partition)
        if verdict == "sealed":
            self._sealed_touch += 1
            raise SealedPartitionError(f"evaluated dataset {dataset} carries sealed label {partition!r}")
        if verdict == "label_missing":
            self._label_missing += 1
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        unknown = set(metrics) - set(METRICS[task])
        if unknown:
            raise ValueError(f"metrics {sorted(unknown)} don't apply to {task}")
        if len(self._rows) >= self.max_datasets:
            self._dropped += 1
            return
        self._rows.append(
            {
                "dataset": dataset,
                "family": family,
                "task": task,
                "fold": fold,
                "metrics": {k: _finite(v) for k, v in metrics.items()},
                "prediction_mean": _finite(prediction_mean),
                "prediction_std": _finite(prediction_std),
                "selected": selected_program_sha256,
                "in_bank": selected_in_bank,
            }
        )

    def _task_summary(self, task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_datasets": len(rows)}
        has_families = all(r["family"] is not None for r in rows)
        families: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            families.setdefault(r["family"] or "", []).append(r)
        out["n_families"] = len(families) if has_families else None
        for m in METRICS[task]:
            per_dataset = [r["metrics"][m] for r in rows if r["metrics"].get(m) is not None]
            entry: dict[str, Any] = {"n": len(per_dataset)}
            if not per_dataset:
                out[m] = entry
                continue
            entry["dataset_mean"] = sum(per_dataset) / len(per_dataset)
            if has_families:
                fam_means = {}
                for f, rs in families.items():
                    vals = [r["metrics"][m] for r in rs if r["metrics"].get(m) is not None]
                    if vals:
                        fam_means[f] = sum(vals) / len(vals)
                entry["family_weighted_mean"] = sum(fam_means.values()) / len(fam_means)
                entry["family_quantiles"] = _quantiles(list(fam_means.values()))
                lower_is_better = m in ("loss", "logloss", "brier", "ece")
                worst = (max if lower_is_better else min)(fam_means, key=fam_means.get)
                entry["worst_family"] = {"family": worst, "value": fam_means[worst]}
                if m == "r2":
                    entry["n_negative_families"] = sum(1 for v in fam_means.values() if v < 0.0)
            else:
                entry["family_weighted_mean"] = None
                entry["availability"] = "no_family_ids"
            out[m] = entry
        return out

    def finish(self, at: int) -> dict[str, Any]:
        rows = self._rows
        rec: dict[str, Any] = {
            "at": int(at),
            "family_roster_sha256": self.roster,
            "fold_map_sha256": self.fold_map,
            "availability": None if self.roster else "no_frozen_family_roster",
            "n_datasets": len(rows),
            "dropped_datasets": self._dropped,
            "sealed_touch": self._sealed_touch,
            "sealed_label_missing": self._label_missing,
            "secondary_metrics": ["auc", "logloss", "brier", "acc", "ece"],
            "tasks": {t: self._task_summary(t, [r for r in rows if r["task"] == t]) for t in TASKS},
        }
        means = [r["prediction_mean"] for r in rows if r["prediction_mean"] is not None]
        stds = [r["prediction_std"] for r in rows if r["prediction_std"] is not None]
        if len(means) >= 2:
            mu = sum(means) / len(means)
            spread = math.sqrt(sum((x - mu) ** 2 for x in means) / (len(means) - 1))
        else:
            spread = None
        rec["prediction_spread"] = {
            "across_dataset_std_of_means": spread,
            "within_dataset_std": _quantiles(stds),
        }
        sel = [r["selected"] for r in rows if r["selected"] is not None]
        if sel:
            counts: dict[str, int] = {}
            for s in sel:
                counts[s] = counts.get(s, 0) + 1
            n = len(sel)
            entropy = -sum(c / n * math.log(c / n) for c in counts.values())
            in_bank = [r["in_bank"] for r in rows if r["selected"] is not None and r["in_bank"] is not None]
            rec["selection"] = {
                "n": n,
                "unique_frac": len(counts) / n,
                "top1_share": max(counts.values()) / n,
                "entropy_norm": entropy / math.log(n) if n > 1 else None,
                "bank_frac": sum(in_bank) / len(in_bank) if in_bank else None,
            }
        else:
            rec["selection"] = {"n": 0, "availability": "no_selected_programs"}
        self._reset()
        if self.emit is not None:
            self.emit(RECORD_KIND, rec)
        return rec
