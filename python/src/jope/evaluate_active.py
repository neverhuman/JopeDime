from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._evaluate_active_summary import (
    _correlation as _correlation,
)
from ._evaluate_active_summary import (
    _head_from_genome as _head_from_genome,
)
from ._evaluate_active_summary import (
    _head_outcomes as _head_outcomes,
)
from ._evaluate_active_summary import (
    _mean as _mean,
)
from ._evaluate_active_summary import (
    _median as _median,
)
from ._evaluate_active_summary import (
    _per_engine_summary as _per_engine_summary,
)
from ._evaluate_active_summary import (
    _summary,
)
from ._evaluate_active_summary import (
    _worst_reward_outliers as _worst_reward_outliers,
)
from .data.active import parse_pipeline_actions as parse_pipeline_actions
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
