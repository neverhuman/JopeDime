from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data.active import parse_pipeline_actions
from .engines import EngineRouter
from .eval_ledger import EVALUATOR_VERSION, iter_eval_regression_with_ledger
from .io import read_jsonl
from .model import JopeActiveModelV2
from .propose import (
    embedding_from_vs,
    load_any_model,
    load_standardization,
    propose_active_v2,
    target_summary_from_vs,
)
from .schema import TrainingRow
from .selector_calibration import load_selector_calibration
from .status import artifact_freshness
from .validator import RustValidator


def evaluate_active_v2(
    *,
    art_dir: str | Path,
    checkpoint: str | Path,
    split: str = "validation",
    datasets: str = "all",
    proposals_per_dataset: int = 3,
    ledger: str | Path | None = None,
    out_dir: str | Path,
    folds: int = 3,
    seed: int = 0,
    temperature: float = 0.0,
    eval_workers: int | None = None,
    engines: tuple[str, ...] = ("featmath",),
    router: EngineRouter | None = None,
    selector_calibration: str | Path | None = None,
) -> dict[str, Any]:
    art = Path(art_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_any_model(checkpoint)
    from .policy.model_v3 import JopePolicyModelV3

    is_v3 = isinstance(model, JopePolicyModelV3)
    if not is_v3 and not isinstance(model, JopeActiveModelV2):
        raise ValueError("evaluate-active requires an active_v2 or policy_v3 checkpoint")
    calibration = load_selector_calibration(selector_calibration) if selector_calibration else None
    router = router or EngineRouter()
    standardization = load_standardization(art / "standardization.json")
    split_manifest = _read_json(art / "split_manifest.json")
    rows = [TrainingRow.from_json(row) for row in read_jsonl(art / "active_rows.jsonl")]
    historical = _historical_by_dataset(rows)
    selected = _select_datasets(split_manifest, split=split, datasets=datasets)
    proposals_path = out / "proposals.jsonl"
    evaluations_path = out / "evaluation.jsonl"
    candidate_audit_path = out / "candidate_audit.jsonl"
    proposal_rows: list[dict[str, Any]] = []
    candidate_audit_rows: list[dict[str, Any]] = []
    eval_inputs: list[dict[str, Any]] = []
    eval_meta: list[dict[str, Any]] = []
    validator = RustValidator()

    for dataset_idx, (dataset_hash, entry) in enumerate(selected.items()):
        vs_path = entry.get("vs_path")
        train_csv = entry.get("train_csv")
        if not vs_path or not train_csv:
            continue
        incumbent = _incumbent_for(dataset_hash, entry, historical)
        embedding = embedding_from_vs(vs_path, standardization)
        target_summary = target_summary_from_vs(entry.get("train_target_vs"))
        dataset_audit: list[dict[str, Any]] = []
        if is_v3:
            from .policy.decode import propose_v3

            proposed = propose_v3(
                embedding,
                target_summary=target_summary,
                model=model,
                router=router,
                n=max(1, proposals_per_dataset) * 4,
                seed=seed + dataset_idx * 1009,
                temperature=temperature if temperature > 0 else 0.7,
                engines=engines,
                historical=_historical_pairs_for(dataset_hash, historical),
                candidate_audit=dataset_audit,
                selector_calibration=calibration,
            )
        else:
            proposed = propose_active_v2(
                embedding,
                target_summary=target_summary,
                model=model,
                validator=validator,
                # Over-request reward-ranked candidates so novelty filtering still has
                # high-reward options after historical (non-novel) genomes are removed.
                n=max(1, proposals_per_dataset) * 4,
                seed=seed + dataset_idx * 1009,
                temperature=temperature,
                beam=True,
                historical_genomes=_historical_genomes_for(dataset_hash, historical),
                candidate_audit=dataset_audit,
                selector_calibration=calibration,
            )
        for audit_row in dataset_audit:
            candidate_audit_rows.append(
                {
                    **audit_row,
                    "dataset_hash": dataset_hash,
                    "split": entry.get("split", split),
                    "incumbent_oof_r2": incumbent,
                }
            )
        seen = set(historical.get(dataset_hash, {}).get("canonical", set()))
        novel_for_dataset = 0
        eval_skip_reason = "missing_incumbent" if incumbent is None else None
        for proposal in proposed:
            canonical = proposal.get("canonical_genome")
            engine = str(proposal.get("engine") or "featmath")
            novel = bool(
                proposal.get("valid") and isinstance(canonical, str) and (engine, canonical) not in seen
            )
            proposal_row = {
                **proposal,
                "dataset_hash": dataset_hash,
                "split": entry.get("split", split),
                "incumbent_oof_r2": incumbent,
                "train_csv": train_csv,
                "test_csv": entry.get("test_csv"),
                "train_target_vs": entry.get("train_target_vs"),
                "novel": novel,
            }
            if eval_skip_reason is not None:
                proposal_row["evaluation_skip_reason"] = eval_skip_reason
            proposal_rows.append(proposal_row)
            if eval_skip_reason is not None or not novel or novel_for_dataset >= proposals_per_dataset:
                continue
            seen.add((engine, str(canonical)))
            novel_for_dataset += 1
            eval_inputs.append(
                {
                    "engine": engine,
                    "dataset_hash": dataset_hash,
                    "train_csv": train_csv,
                    "test_csv": entry.get("test_csv"),
                    "genome": canonical,
                    "folds": folds,
                }
            )
            eval_meta.append(proposal_row)

    _write_jsonl(proposals_path, proposal_rows)
    _write_jsonl(candidate_audit_path, candidate_audit_rows)
    evaluation_rows: list[dict[str, Any]] = []
    _write_jsonl(evaluations_path, [])
    summary = _summary(proposal_rows, evaluation_rows, selected)
    summary.update(
        _summary_metadata(
            art,
            checkpoint,
            split,
            datasets,
            len(selected),
            proposals_per_dataset,
            eval_workers,
            ledger,
            proposals_path,
            evaluations_path,
            candidate_audit_path,
            selector_calibration,
        )
    )
    summary.update(_evaluation_progress(total=len(eval_inputs), done=0, in_progress=bool(eval_inputs)))
    _write_eval_summary(out, summary)

    for eval_idx, result in iter_eval_regression_with_ledger(
        eval_inputs,
        ledger=ledger,
        router=router,
        workers=eval_workers,
    ):
        meta = eval_meta[eval_idx]
        cv = result.get("cv") if isinstance(result.get("cv"), dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
        incumbent = meta.get("incumbent_oof_r2")
        oof_r2 = cv.get("oof_r2")
        delta = (
            float(oof_r2) - float(incumbent)
            if isinstance(oof_r2, (int, float)) and isinstance(incumbent, (int, float))
            else None
        )
        # Holdout: model was trained on the whole train.csv and scored on test.csv (last col = target).
        test_r2 = test_metrics.get("r2") if isinstance(test_metrics.get("r2"), (int, float)) else None
        evaluation_row = {
            **result,
            "dataset_hash": meta["dataset_hash"],
            "split": meta["split"],
            "proposal_index": meta.get("proposal_index"),
            "decoder": meta.get("decoder"),
            "evaluator_version": result.get("evaluator_version") or EVALUATOR_VERSION,
            "reward": meta.get("reward"),
            "selection_score": meta.get("selection_score"),
            "selection_policy": meta.get("selection_policy"),
            "selection_slot": meta.get("selection_slot"),
            "calibrated_expected_delta": meta.get("calibrated_expected_delta"),
            "calibrated_win_rate": meta.get("calibrated_win_rate"),
            "calibrated_tail_risk": meta.get("calibrated_tail_risk"),
            "selection_score_components": meta.get("selection_score_components"),
            "reward_std": meta.get("reward_std"),
            "reward_percentile_in_engine": meta.get("reward_percentile_in_engine"),
            "p_bad": meta.get("p_bad"),
            "predicted_oof_delta": meta.get("predicted_oof_delta"),
            "uncertainty": meta.get("uncertainty"),
            "head_type": meta.get("head_type"),
            "incumbent_oof_r2": incumbent,
            "oof_delta_vs_incumbent": delta,
            "test_csv": meta.get("test_csv") or result.get("test_csv"),
            "test_r2": test_r2,
            "train_target_vs": meta.get("train_target_vs"),
        }
        evaluation_rows.append(evaluation_row)
        _append_jsonl(evaluations_path, evaluation_row)
        summary = _summary(proposal_rows, evaluation_rows, selected)
        summary.update(
            _summary_metadata(
                art,
                checkpoint,
                split,
                datasets,
                len(selected),
                proposals_per_dataset,
                eval_workers,
                ledger,
                proposals_path,
                evaluations_path,
                candidate_audit_path,
                selector_calibration,
            )
        )
        summary.update(
            _evaluation_progress(
                total=len(eval_inputs),
                done=len(evaluation_rows),
                in_progress=len(evaluation_rows) < len(eval_inputs),
            )
        )
        _write_eval_summary(out, summary)
    summary = _summary(proposal_rows, evaluation_rows, selected)
    summary.update(
        _summary_metadata(
            art,
            checkpoint,
            split,
            datasets,
            len(selected),
            proposals_per_dataset,
            eval_workers,
            ledger,
            proposals_path,
            evaluations_path,
            candidate_audit_path,
            selector_calibration,
        )
    )
    summary.update(_evaluation_progress(total=len(eval_inputs), done=len(evaluation_rows), in_progress=False))
    _write_eval_summary(out, summary)
    return summary


def _historical_by_dataset(rows: list[TrainingRow]) -> dict[str, dict[str, Any]]:
    # canonical keys are (engine, canonical) — a genome string valid in both grammars
    # is a different candidate per engine (distinct evaluators).
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = out.setdefault(row.dataset_id, {"canonical": set(), "best": None, "scored": []})
        engine = getattr(row, "engine", "featmath")
        if row.canonical_genome:
            entry["canonical"].add((engine, row.canonical_genome))
        if row.score is not None:
            entry["best"] = (
                float(row.score) if entry["best"] is None else max(float(entry["best"]), float(row.score))
            )
            if row.canonical_genome:
                entry["scored"].append((float(row.score), engine, row.canonical_genome))
    for entry in out.values():
        entry["scored"] = [
            (engine, genome)
            for _, engine, genome in sorted(entry.get("scored") or [], key=lambda item: item[0], reverse=True)
        ]
    return out


def _historical_genomes_for(
    dataset_hash: str, historical: dict[str, dict[str, Any]], *, limit: int = 8
) -> list[str]:
    # featmath-only genome strings for the legacy v2 mutation path.
    entry = historical.get(dataset_hash) or {}
    pairs = entry.get("scored") if isinstance(entry.get("scored"), list) else []
    return [genome for engine, genome in pairs if engine == "featmath"][:limit]


def _historical_pairs_for(
    dataset_hash: str, historical: dict[str, dict[str, Any]], *, limit: int = 16
) -> list[tuple[str, str]]:
    entry = historical.get(dataset_hash) or {}
    pairs = entry.get("scored") if isinstance(entry.get("scored"), list) else []
    return list(pairs[:limit])


def _select_datasets(
    split_manifest: dict[str, Any], *, split: str, datasets: str
) -> dict[str, dict[str, Any]]:
    all_datasets = split_manifest.get("datasets") if isinstance(split_manifest.get("datasets"), dict) else {}
    selected = {
        dataset_hash: dict(entry)
        for dataset_hash, entry in sorted(all_datasets.items())
        if split == "all" or entry.get("split") == split
    }
    if datasets == "all":
        return selected
    requested = {item.strip() for item in datasets.split(",") if item.strip()}
    return {dataset_hash: entry for dataset_hash, entry in selected.items() if dataset_hash in requested}


def _incumbent_for(
    dataset_hash: str, entry: dict[str, Any], historical: dict[str, dict[str, Any]]
) -> float | None:
    value = entry.get("best_known_oof_r2")
    if isinstance(value, (int, float)):
        return float(value)
    value = historical.get(dataset_hash, {}).get("best")
    return float(value) if isinstance(value, (int, float)) else None


def _summary(
    proposal_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in proposal_rows if row.get("valid")]
    novel = [row for row in proposal_rows if row.get("novel")]
    invalid_reasons = Counter(
        str(error) for row in proposal_rows if not row.get("valid") for error in (row.get("errors") or [])
    )
    deltas = [
        float(row["oof_delta_vs_incumbent"])
        for row in evaluation_rows
        if isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and math.isfinite(float(row["oof_delta_vs_incumbent"]))
    ]
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in evaluation_rows:
        delta = row.get("oof_delta_vs_incumbent")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            by_dataset[str(row.get("dataset_hash"))].append(float(delta))
    best_of_n = [max(values) for values in by_dataset.values() if values]
    test_r2s = [
        float(row["test_r2"])
        for row in evaluation_rows
        if isinstance(row.get("test_r2"), (int, float)) and math.isfinite(float(row["test_r2"]))
    ]
    heads = Counter(
        str(row.get("head_type") or _head_from_genome(row.get("canonical_genome") or "")) for row in valid
    )
    eval_skip_reasons = Counter(
        str(row.get("evaluation_skip_reason")) for row in proposal_rows if row.get("evaluation_skip_reason")
    )
    eval_skipped_datasets = {
        str(row.get("dataset_hash"))
        for row in proposal_rows
        if row.get("evaluation_skip_reason") and row.get("dataset_hash") is not None
    }
    gp_evals = [
        row
        for row in evaluation_rows
        if (row.get("head_type") == "gpr" or "(gpr" in str(row.get("canonical_genome") or ""))
    ]
    reward_pairs = [
        (float(row["reward"]), float(row["oof_delta_vs_incumbent"]))
        for row in evaluation_rows
        if isinstance(row.get("reward"), (int, float))
        and isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and math.isfinite(float(row["reward"]))
        and math.isfinite(float(row["oof_delta_vs_incumbent"]))
    ]
    evaluator_versions = Counter(str(row.get("evaluator_version") or "unknown") for row in evaluation_rows)
    return {
        "proposal_count": len(proposal_rows),
        "proposal_valid_count": len(valid),
        "proposal_valid_rate": len(valid) / max(1, len(proposal_rows)),
        "proposal_novel_count": len(novel),
        "proposal_novel_rate": len(novel) / max(1, len(proposal_rows)),
        "invalid_reasons": dict(invalid_reasons.most_common(8)),
        "evaluated_count": len(evaluation_rows),
        "cache_hit_count": sum(1 for row in evaluation_rows if row.get("cache_hit")),
        "cache_hit_rate": sum(1 for row in evaluation_rows if row.get("cache_hit"))
        / max(1, len(evaluation_rows)),
        "novel_eval_count": sum(1 for row in evaluation_rows if not row.get("cache_hit")),
        "eval_skipped_dataset_count": len(eval_skipped_datasets),
        "eval_skip_reasons": dict(eval_skip_reasons.most_common()),
        "mean_delta_vs_incumbent": _mean(deltas),
        "mean_delta_vs_incumbent_clipped": _mean([max(d, -1.0) for d in deltas]),
        "median_delta_vs_incumbent": _median(deltas),
        "win_rate_vs_incumbent": sum(1 for delta in deltas if delta > 0.0) / max(1, len(deltas)),
        "best_of_n_mean_delta": _mean(best_of_n),
        "best_of_n_median_delta": _median(best_of_n),
        "holdout_test_eval_count": len(test_r2s),
        "mean_test_r2": _mean(test_r2s),
        "median_test_r2": _median(test_r2s),
        "head_distribution": dict(heads),
        "head_diversity": len([head for head, count in heads.items() if count > 0 and head != "None"]),
        "gp_proposal_count": sum(
            1
            for row in valid
            if row.get("head_type") == "gpr" or "(gpr" in str(row.get("canonical_genome") or "")
        ),
        "gp_eval_count": len(gp_evals),
        "gp_fold_r2": [
            ((row.get("cv") or {}).get("fold_r2")) for row in gp_evals if isinstance(row.get("cv"), dict)
        ],
        "reward_actual_correlation": _correlation(
            [pair[0] for pair in reward_pairs], [pair[1] for pair in reward_pairs]
        ),
        "head_outcomes": _head_outcomes(evaluation_rows),
        "worst_reward_outliers": _worst_reward_outliers(evaluation_rows),
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_versions": dict(evaluator_versions),
        "selected_dataset_count": len(selected),
        "per_engine": _per_engine_summary(proposal_rows, evaluation_rows),
    }


def _per_engine_summary(
    proposal_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in proposal_rows:
        engine = str(row.get("engine") or "featmath")
        entry = out.setdefault(
            engine,
            {"proposals": 0, "valid": 0, "novel": 0, "evaluated": 0, "wins": 0, "deltas": []},
        )
        entry["proposals"] += 1
        entry["valid"] += int(bool(row.get("valid")))
        entry["novel"] += int(bool(row.get("novel")))
    for row in evaluation_rows:
        engine = str(row.get("engine") or "featmath")
        entry = out.setdefault(
            engine,
            {"proposals": 0, "valid": 0, "novel": 0, "evaluated": 0, "wins": 0, "deltas": []},
        )
        entry["evaluated"] += 1
        delta = row.get("oof_delta_vs_incumbent")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            entry["deltas"].append(float(delta))
            entry["wins"] += int(float(delta) > 0.0)
    for entry in out.values():
        deltas = entry.pop("deltas")
        entry["valid_rate"] = entry["valid"] / max(1, entry["proposals"])
        entry["win_rate_vs_incumbent"] = entry["wins"] / max(1, entry["evaluated"])
        entry["median_delta_vs_incumbent"] = _median(deltas)
    return out


def _head_from_genome(genome: str) -> str | None:
    try:
        return str(parse_pipeline_actions(genome)["head"]["op"])
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _summary_metadata(
    art: Path,
    checkpoint: str | Path,
    split: str,
    datasets: str,
    datasets_selected: int,
    proposals_per_dataset: int,
    eval_workers: int | None,
    ledger: str | Path | None,
    proposals_path: Path,
    evaluations_path: Path,
    candidate_audit_path: Path,
    selector_calibration: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "art_dir": str(art),
        "checkpoint": str(checkpoint),
        "split": split,
        "datasets_requested": datasets,
        "datasets_selected": datasets_selected,
        "proposals_per_dataset": proposals_per_dataset,
        "eval_workers": eval_workers,
        "ledger": str(ledger) if ledger else None,
        "proposals_jsonl": str(proposals_path),
        "evaluation_jsonl": str(evaluations_path),
        "candidate_audit_jsonl": str(candidate_audit_path),
        "selector_calibration": str(selector_calibration) if selector_calibration else None,
        "evaluator_version": EVALUATOR_VERSION,
        "artifact_freshness": artifact_freshness(art_dir=art, run_dir=evaluations_path.parent),
    }


def _evaluation_progress(*, total: int, done: int, in_progress: bool) -> dict[str, Any]:
    return {
        "eval_requested_count": total,
        "eval_completed_count": done,
        "eval_remaining_count": max(0, total - done),
        "evaluation_in_progress": in_progress,
    }


def _write_eval_summary(out: Path, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _head_outcomes(evaluation_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_head: dict[str, list[float]] = defaultdict(list)
    for row in evaluation_rows:
        delta = row.get("oof_delta_vs_incumbent")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            continue
        head = str(row.get("head_type") or _head_from_genome(row.get("canonical_genome") or "") or "unknown")
        by_head[head].append(float(delta))
    return {
        head: {
            "count": len(values),
            "wins": sum(1 for value in values if value > 0.0),
            "win_rate": sum(1 for value in values if value > 0.0) / max(1, len(values)),
            "median_delta": _median(values),
        }
        for head, values in sorted(by_head.items())
    }


def _worst_reward_outliers(evaluation_rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        row
        for row in evaluation_rows
        if isinstance(row.get("reward"), (int, float))
        and isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and float(row["oof_delta_vs_incumbent"]) < 0.0
    ]
    rows.sort(
        key=lambda row: (float(row.get("reward") or 0.0), -float(row.get("oof_delta_vs_incumbent") or 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        out.append(
            {
                "dataset_hash": row.get("dataset_hash"),
                "head_type": row.get("head_type") or _head_from_genome(row.get("canonical_genome") or ""),
                "reward": row.get("reward"),
                "oof_delta_vs_incumbent": row.get("oof_delta_vs_incumbent"),
                "canonical_genome": row.get("canonical_genome"),
            }
        )
    return out
