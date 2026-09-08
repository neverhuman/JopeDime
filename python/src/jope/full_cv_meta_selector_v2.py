from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold

from .data.active_v3 import robust_score
from .data.vs import dataset_signature_from_vs
from .elo_bench import ladder_elo, select_bench_datasets
from .prime_config import load_config
from .propose import target_summary_from_vs


REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIME_CONFIG = load_config()
DEFAULT_RUN_DIR = _PRIME_CONFIG.legacy_root / "runs" / "self_play_v33"
DEFAULT_ART_DIR = (
    _PRIME_CONFIG.legacy_root / "data" / "processed" / "active_v3_dual_20260706"
)
DEFAULT_BENCH_DIR = _PRIME_CONFIG.validation_benchmark_root
DEFAULT_OUT_DIR = _PRIME_CONFIG.artifacts / "full_cv_meta_selector_v2"
DEFAULT_V1_DIR = DEFAULT_BENCH_DIR / "full_cv_meta_selector_v1"
REGRESSION_ROOT = _PRIME_CONFIG.regression_dataset_root
EXPECTED_FULL_CV_GEN1_30 = 464_884
NGRAM_BUCKETS = 512
TOP_NDCG = 10
MODEL_THREADS = 8
PAIRWISE_TOP_K = 8
PAIRWISE_HARD_TOP_K = 12
LEAKY_PREFIXES = ("metrics.test", "test_", "label_", "oracle_", "regret", "rank_test")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?")
OP_RE = re.compile(r"\(([A-Za-z_][A-Za-z0-9_]*)")
PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?)")

FAMILY_TOKENS: dict[str, tuple[str, ...]] = {
    "basis": ("poly", "cheb", "spline", "fourier", "radial", "sigmoid"),
    "chaos": ("logistic", "tent", "henon", "chaos"),
    "ensemble": ("ensemble", "bag", "blend"),
    "geometry": ("quasicrystal", "voronoi", "delaunay", "geometry"),
    "interaction": ("interact", "pair", "prod", "cross"),
    "kernel": ("krr", "kernel", "rbf", "nystrom"),
    "linear": ("ridge", "lasso", "elasticnet", "linear", "huber"),
    "neighbors": ("knn", "neighbor"),
    "projection": ("proj", "pca", "ica", "random_proj", "rand_proj"),
    "rank": ("rank", "quantile", "bins"),
    "reduction": ("svd", "pca", "reduce"),
    "robust": ("huber", "winsor", "robust", "clip"),
    "rowwise": ("row_", "rowwise"),
    "scaling": ("scale", "standardize", "normalize"),
    "spectral": ("spectral", "fft", "wave", "eigen"),
    "symbolic": ("derive", "expr", "symbolic", "target_bin"),
    "target_aware": ("target", "encode"),
    "topology": ("persistence", "topology"),
    "tree": ("gbm", "forest", "tree", "histgbm", "xgb"),
    "wavelet": ("wavelet", "haar"),
}

TARGET_AWARE_OPS = {"target_encode", "target_bin_encode", "target_mean_encode"}
RANKISH_OPS = {"rank", "quantile_bins", "quantile", "bins", "winsor"}
STOCHASTIC_OPS = {"rand_proj", "random_proj", "rff_cos", "rff_sin", "bag"}
NONLINEAR_REPEAT_OPS = {"sigmoid", "rank", "winsor", "target_bin_encode"}
FEATURE_FAMILIES = {
    "base": ("valid", "cache_hit", "folds", "timing_", "cv_oof_", "cv_mean_", "pool_", "head_type=", "engine="),
    "cv_reliability": ("cvrel_", "cv_fold_", "cv_sorted_", "cv_lcb_", "cv_boot_", "cv_rank_", "cv_margin_", "cv_prob_"),
    "pool_context": ("ctx_", "consensus_", "delta_vs_", "rank_", "pct_", "pooldist_"),
    "genome_structure": ("genome_", "gstruct_", "gparam_", "op=", "family=", "ngram_"),
    "dataset_stats": ("ds_", "dsstat_", "shift_"),
    "anchor": ("anchor_",),
}

_DATASET_FEATURE_CACHE: dict[str, dict[str, Any]] = {}
_DATASET_NUMERIC_CACHE: dict[str, dict[str, Any]] = {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    art_dir = Path(args.art_dir)
    bench_dir = Path(args.bench_dir)
    v1_dir = Path(args.v1_dir)

    log(f"output: {out_dir}")
    validation_sets = select_bench_datasets(art_dir, limit=args.limit_validation_datasets)
    validation_hashes = validation_sets["validation"]
    double_holdout_hashes = validation_sets["double_holdout"]
    validation_hash_set = set(validation_hashes)
    log(f"validation datasets: {len(validation_hashes)}; double-holdout: {len(double_holdout_hashes)}")

    split_meta = load_split_manifest(art_dir)
    source_audit = build_source_audit(
        run_dir=run_dir,
        bench_dir=bench_dir,
        art_dir=art_dir,
        split_meta=split_meta,
        validation_hashes=validation_hash_set,
    )
    pool_lookup, pool_audit = load_candidate_pool_lookup(args.candidate_pool, run_dir, bench_dir)
    source_audit["candidate_pool_join"] = pool_audit

    train_paths = [run_dir / "ledger.jsonl", run_dir / "miner_ledger.jsonl"]
    if args.extra_train_ledger:
        train_paths.extend(Path(p) for p in args.extra_train_ledger)
    validation_path = Path(args.validation_ledger)

    train_df = load_eval_rows(
        train_paths,
        split_meta=split_meta,
        pool_lookup=pool_lookup,
        validation_hashes=validation_hash_set,
        role="train",
        max_datasets=args.max_train_datasets,
    )
    val_df = load_eval_rows(
        [validation_path],
        split_meta=split_meta,
        pool_lookup=pool_lookup,
        validation_hashes=validation_hash_set,
        role="validation",
        max_datasets=args.limit_validation_datasets,
    )
    if train_df.empty:
        raise RuntimeError("no train rows with heldout test labels were loaded")
    if val_df.empty:
        raise RuntimeError("no validation rows with heldout test labels were loaded")
    assert_no_validation_leakage(train_df, validation_hash_set)
    log(f"loaded train rows={len(train_df):,} datasets={train_df['dataset_hash'].nunique():,}")
    log(f"loaded validation rows={len(val_df):,} datasets={val_df['dataset_hash'].nunique():,}")

    train_df = add_group_labels(train_df)
    val_df = add_group_labels(val_df)
    train_df = add_within_dataset_context_features(train_df)
    val_df = add_within_dataset_context_features(val_df)
    train_df["label_cv_red_flag"] = compute_cv_red_flag_label(train_df)

    feature_dicts_train = [row_to_features(row) for row in train_df.to_dict("records")]
    feature_dicts_val = [row_to_features(row) for row in val_df.to_dict("records")]
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    x_train = vectorizer.fit_transform(feature_dicts_train).tocsr()
    x_val = vectorizer.transform(feature_dicts_val).tocsr()
    feature_names = list(vectorizer.feature_names_)
    assert_no_feature_leakage(feature_names)
    feature_schema = build_feature_schema(feature_names)
    write_json(out_dir / "feature_schema.json", feature_schema)
    log(f"feature matrix: train={x_train.shape} validation={x_val.shape}")

    groups = train_df["dataset_hash"].astype(str).to_numpy()
    group_order = np.unique(groups)
    cv_folds = min(max(2, args.cv_folds), len(group_order))
    splitter = GroupKFold(n_splits=cv_folds)

    component_oof, component_models, component_compare = cross_validate_components(
        train_df=train_df,
        x_train=x_train,
        groups=groups,
        splitter=splitter,
        feature_names=feature_names,
        random_state=args.seed,
    )
    for key, values in component_oof.items():
        train_df[key] = values

    blend_result = choose_blend_weights(train_df)
    train_df["score_meta_selector_v2_cvfit"] = score_with_blend(train_df, blend_result["weights"])
    model_compare = build_model_compare(train_df, component_compare, blend_result)
    pd.DataFrame(model_compare).to_csv(out_dir / "model_compare.csv", index=False)

    ablations = run_feature_ablations(
        train_df=train_df,
        x_train=x_train,
        groups=groups,
        splitter=splitter,
        feature_names=feature_names,
        random_state=args.seed,
    )
    pd.DataFrame(ablations).to_csv(out_dir / "feature_ablation.csv", index=False)

    final_models = fit_final_components(
        train_df=train_df,
        val_df=val_df,
        x_train=x_train,
        x_val=x_val,
        feature_names=feature_names,
        random_state=args.seed,
    )
    val_df["score_stage1_reg"] = final_models["stage1_reg"].predict(x_val)
    val_df["score_stage1_rank"] = final_models["stage1_rank"]
    val_df["score_risk_gate"] = final_models["risk_gate"]
    val_df["score_pairwise"] = predict_pairwise_anchor_scores(
        df=val_df,
        x=x_val,
        model=final_models["pairwise_model"],
    )
    val_df["score_meta_selector_v2"] = score_with_blend(val_df, blend_result["weights"])

    train_df["score_meta_selector_v2_final_fit"] = score_with_blend(train_df, blend_result["weights"])

    model_artifact = {
        "schema_version": "full_cv_meta_selector_v2",
        "created_at_unix": time.time(),
        "feature_names": feature_names,
        "feature_schema": feature_schema,
        "blend_weights": blend_result["weights"],
        "score_direction": "higher_is_better",
        "vectorizer": vectorizer,
        "stage1_reg_model": final_models["stage1_reg"],
        "risk_gate_model": final_models["risk_gate_model"],
        "pairwise_model": final_models["pairwise_model"],
        "stage1_rank_model": final_models["stage1_rank_model"],
        "pairwise_feature_source": "sparse difference over candidate features",
        "train_rows": int(len(train_df)),
        "train_datasets": int(train_df["dataset_hash"].nunique()),
        "validation_rows_scored": int(len(val_df)),
        "validation_datasets_scored": int(val_df["dataset_hash"].nunique()),
        "source_run_dir": str(run_dir),
        "validation_ledger": str(validation_path),
    }
    joblib.dump(model_artifact, out_dir / "model_artifact.joblib")
    write_feature_importance(out_dir / "feature_importance.csv", final_models["stage1_reg_model"], feature_names)
    write_shap_stub(out_dir / "shap_summary.csv", feature_names)

    choices_by_mode = build_validation_choices(
        val_df=val_df,
        validation_hashes=validation_hashes,
        current_per_ds_path=Path(args.current_per_ds),
        v1_choices_path=v1_dir / "choices_validation.jsonl",
    )
    write_choices(out_dir / "choices_validation.jsonl", choices_by_mode["meta_selector_v2"])

    elo_report, metrics_rows, per_dataset_by_mode = build_elo_report(
        choices_by_mode=choices_by_mode,
        validation_hashes=validation_hashes,
        double_holdout_hashes=double_holdout_hashes,
        current_per_ds_path=Path(args.current_per_ds),
        current_metrics_path=Path(args.current_metrics),
        model_compare=model_compare,
        blend_result=blend_result,
        source_audit=source_audit,
        jable_path=v1_dir / "jable_regressor_augmented_ladder.json",
    )
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
    write_json(out_dir / "elo_report.json", elo_report)
    write_json(out_dir / "per_dataset_by_mode.json", per_dataset_by_mode)
    write_json(out_dir / "hard_misses.json", build_hard_misses(val_df, choices_by_mode))
    write_json(out_dir / "cv_failure_report.json", build_cv_failure_report(train_df))
    write_json(out_dir / "oracle_gap_report.json", build_oracle_gap_report(per_dataset_by_mode))
    write_readme(out_dir / "README.md", elo_report=elo_report, blend_result=blend_result, ablations=ablations)

    source_audit["runtime_sec"] = round(time.time() - t0, 3)
    source_audit["feature_matrix"] = {
        "train_rows": int(x_train.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "feature_family_counts": {k: len(v["features"]) for k, v in feature_schema["families"].items()},
        "leakage_prefixes_blocked": list(LEAKY_PREFIXES),
    }
    write_json(out_dir / "source_audit.json", source_audit)
    log("done")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-generation offline full-CV meta-selector study.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--art-dir", default=str(DEFAULT_ART_DIR))
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    parser.add_argument("--validation-ledger", default=str(DEFAULT_RUN_DIR / "elo_ledger.jsonl"))
    parser.add_argument("--current-per-ds", default=str(DEFAULT_BENCH_DIR / "per_ds.json"))
    parser.add_argument("--current-metrics", default=str(DEFAULT_BENCH_DIR / "metrics.csv"))
    parser.add_argument("--candidate-pool", action="append", default=[])
    parser.add_argument("--extra-train-ledger", action="append", default=[])
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-datasets", type=int)
    parser.add_argument("--limit-validation-datasets", type=int)
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[full-cv-meta-v2] {message}", flush=True)


def load_split_manifest(art_dir: Path) -> dict[str, dict[str, Any]]:
    data = json.loads((art_dir / "split_manifest.json").read_text(encoding="utf-8"))
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"unexpected split_manifest datasets schema in {art_dir}")
    return {str(k): dict(v) for k, v in datasets.items()}


def build_source_audit(
    *,
    run_dir: Path,
    bench_dir: Path,
    art_dir: Path,
    split_meta: dict[str, dict[str, Any]],
    validation_hashes: set[str],
) -> dict[str, Any]:
    metrics_candidates = [
        run_dir / "self_play_metrics.jsonl",
        run_dir / "takeover_backups/20260709T205018Z-governance-repair/self_play_metrics.before.jsonl",
        run_dir / "takeover_backups/20260709T194834Z/self_play_metrics.before.jsonl",
        run_dir / "backups/gen36plus_before_v34_restart_20260709T233751Z/self_play_metrics.before.jsonl",
    ]
    metrics_reconciliations = []
    for path in metrics_candidates:
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        gen30 = [r for r in rows if int(r.get("generation") or 0) <= 30]
        proposal = sum(int(r.get("rows_appended") or 0) for r in gen30)
        miner = sum(int(r.get("miner_mutants_raced") or 0) for r in gen30)
        metrics_reconciliations.append(
            {
                "path": str(path),
                "rows": len(rows),
                "last_generation": max((int(r.get("generation") or 0) for r in rows), default=None),
                "gen1_30_proposal_evals": proposal,
                "gen1_30_gp_mutant_evals": miner,
                "gen1_30_total": proposal + miner,
                "matches_published_464884": proposal + miner == EXPECTED_FULL_CV_GEN1_30,
            }
        )

    ledgers = {
        "self_play_v33_train_proposals": run_dir / "ledger.jsonl",
        "self_play_v33_gp_mutants": run_dir / "miner_ledger.jsonl",
        "self_play_v33_validation_elo": run_dir / "elo_ledger.jsonl",
        "current_validation_k": bench_dir / "ledger.jsonl",
    }
    ledger_counts = {
        name: count_eval_ledger(path, validation_hashes)
        for name, path in ledgers.items()
        if path.exists()
    }
    recoverable_train = (
        ledger_counts.get("self_play_v33_train_proposals", {}).get("test_labeled_rows", 0)
        + ledger_counts.get("self_play_v33_gp_mutants", {}).get("test_labeled_rows", 0)
    )
    best_reconciled = max((r["gen1_30_total"] for r in metrics_reconciliations), default=0)
    return {
        "expected_full_cv_gen1_30": {
            "proposal_evals": 304_570,
            "gp_mutant_evals": 160_314,
            "total": EXPECTED_FULL_CV_GEN1_30,
        },
        "self_play_metrics_reconciliation": metrics_reconciliations,
        "ledger_counts": ledger_counts,
        "recoverable_train_test_labeled_rows": recoverable_train,
        "recoverable_gap_vs_best_reconciled": max(0, best_reconciled - recoverable_train),
        "validation_hash_count": len(validation_hashes),
        "split_manifest": {
            "path": str(art_dir / "split_manifest.json"),
            "dataset_count": len(split_meta),
            "train_count": sum(1 for v in split_meta.values() if v.get("split") == "train"),
            "validation_count": sum(1 for v in split_meta.values() if v.get("split") == "validation"),
        },
        "notes": [
            "Training uses only recoverable row-level ledgers with finite metrics.test.rmse labels.",
            "Validation hashes are filtered out of model fitting.",
            "This v2 study adds pairwise and risk-gate stages but keeps validation test labels for reporting only.",
        ],
    }


def count_eval_ledger(path: Path, validation_hashes: set[str]) -> dict[str, Any]:
    rows = 0
    valid_rows = 0
    test_labeled_rows = 0
    validation_rows = 0
    datasets: set[str] = set()
    for row in iter_jsonl(path):
        rows += 1
        h = str(row.get("dataset_hash") or "")
        if h:
            datasets.add(h)
        if h in validation_hashes:
            validation_rows += 1
        if bool(row.get("valid")):
            valid_rows += 1
        if finite_float(nested_get(row, ("metrics", "test", "rmse"))) is not None:
            test_labeled_rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "datasets": len(datasets),
        "valid_rows": valid_rows,
        "test_labeled_rows": test_labeled_rows,
        "rows_on_validation_hashes": validation_rows,
    }


def load_candidate_pool_lookup(
    explicit_paths: list[str],
    run_dir: Path,
    bench_dir: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    paths = [Path(p) for p in explicit_paths]
    paths.extend(sorted((run_dir / "candidate_pools").glob("candidate_pools*.jsonl")))
    paths.append(bench_dir / "candidate_pools/candidate_pools_elo.jsonl")
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    path_stats = []
    for path in unique_paths(paths):
        if not path.exists():
            continue
        rows = 0
        joined = 0
        for row in iter_jsonl(path):
            rows += 1
            h = str(row.get("dataset_hash") or "")
            engine = str(row.get("engine") or "")
            genome = str(row.get("canonical_genome") or row.get("genome") or "")
            if not h or not engine or not genome:
                continue
            key = (h, engine, genome)
            if key not in lookup:
                lookup[key] = row
                joined += 1
        path_stats.append({"path": str(path), "rows": rows, "new_keys": joined})
    return lookup, {"paths": path_stats, "unique_keys": len(lookup)}


def load_eval_rows(
    paths: list[Path],
    *,
    split_meta: dict[str, dict[str, Any]],
    pool_lookup: dict[tuple[str, str, str], dict[str, Any]],
    validation_hashes: set[str],
    role: str,
    max_datasets: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_allow: set[str] | None = None
    if max_datasets is not None:
        if role == "validation":
            dataset_allow = set(sorted(validation_hashes)[:max_datasets])
        else:
            train_hashes = sorted(h for h, meta in split_meta.items() if meta.get("split") != "validation")
            dataset_allow = set(train_hashes[:max_datasets])

    global_ordinal = 0
    for source_path in paths:
        for row in iter_jsonl(source_path):
            global_ordinal += 1
            h = str(row.get("dataset_hash") or "")
            if role == "train" and h in validation_hashes:
                continue
            if role == "validation" and h not in validation_hashes:
                continue
            if dataset_allow is not None and h not in dataset_allow:
                continue
            record = record_from_eval_row(
                row,
                source_path=source_path,
                source_ordinal=global_ordinal,
                split_meta=split_meta,
                pool_lookup=pool_lookup,
            )
            if record is not None:
                rows.append(record)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["dataset_hash", "source_path", "source_ordinal"], kind="stable").reset_index(drop=True)
    df["candidate_ordinal"] = df.groupby("dataset_hash").cumcount()
    return df


def record_from_eval_row(
    row: dict[str, Any],
    *,
    source_path: Path,
    source_ordinal: int,
    split_meta: dict[str, dict[str, Any]],
    pool_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    test_rmse = finite_float(nested_get(row, ("metrics", "test", "rmse")))
    if test_rmse is None:
        return None
    h = str(row.get("dataset_hash") or "")
    engine = str(row.get("engine") or "")
    genome = str(row.get("canonical_genome") or row.get("genome") or "")
    if not h or not engine or not genome:
        return None
    cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
    train_metrics = nested_get(row, ("metrics", "train"))
    if not isinstance(train_metrics, dict):
        train_metrics = {}
    fold_r2 = [float(v) for v in cv.get("fold_r2", []) if finite_float(v) is not None]
    oof_r2 = finite_float(cv.get("oof_r2"))
    mean_r2 = finite_float(cv.get("mean_r2"))
    score_robust = robust_score(oof_r2, fold_r2)
    pool = pool_lookup.get((h, engine, genome), {})
    meta = split_meta.get(h, {})
    dataset_features = dataset_feature_summary(h, meta)
    genome_features = parse_genome_features(genome, pool.get("family_bitmap") if isinstance(pool, dict) else None)
    fold_stats = summarize_folds(fold_r2)
    rec: dict[str, Any] = {
        "dataset_hash": h,
        "engine": engine,
        "canonical_genome": genome,
        "source_path": str(source_path),
        "source_ordinal": int(source_ordinal),
        "valid": bool(row.get("valid")),
        "cache_hit": bool(row.get("cache_hit")),
        "folds": int(row.get("folds") or len(fold_r2) or 0),
        "evaluator_version": str(row.get("evaluator_version") or ""),
        "head_type": str(row.get("head_type") or pool.get("head_type") or genome_features.get("head_type") or ""),
        "stage_count": finite_float(row.get("stage_count")),
        "transform_count": finite_float(row.get("transform_count")),
        "timing_total_ms": finite_float(nested_get(row, ("timings", "total_ms"))),
        "cv_oof_r2": oof_r2,
        "cv_mean_r2": mean_r2,
        "cv_score_robust": score_robust,
        "train_rmse": finite_float(train_metrics.get("rmse")),
        "train_r2": finite_float(train_metrics.get("r2")),
        "train_mae": finite_float(train_metrics.get("mae")),
        "label_test_rmse": test_rmse,
        "label_test_r2": finite_float(nested_get(row, ("metrics", "test", "r2"))),
        "label_test_mae": finite_float(nested_get(row, ("metrics", "test", "mae"))),
        "pool_reward_mean": finite_float(pool.get("reward_mean", pool.get("reward")) if isinstance(pool, dict) else None),
        "pool_reward_std": finite_float(pool.get("reward_std") if isinstance(pool, dict) else None),
        "pool_reward_lcb": finite_float(pool.get("reward_lcb") if isinstance(pool, dict) else None),
        "pool_reward_percentile": finite_float(pool.get("reward_percentile_in_engine") if isinstance(pool, dict) else None),
        "pool_p_bad": finite_float(pool.get("p_bad") if isinstance(pool, dict) else None),
        "pool_selection_key": finite_float(pool.get("selection_key", pool.get("final_selection_key")) if isinstance(pool, dict) else None),
        "pool_selection_score": finite_float(pool.get("selection_score") if isinstance(pool, dict) else None),
        "pool_selected_for_eval_rank": finite_float(pool.get("selected_for_eval_rank") if isinstance(pool, dict) else None),
        "pool_decoder": str(pool.get("decoder") or "") if isinstance(pool, dict) else "",
        "pool_generation": finite_float(pool.get("generation") if isinstance(pool, dict) else None),
        "pool_candidate_risk": str(pool.get("candidate_risk_reason") or "") if isinstance(pool, dict) else "",
        "pool_cost_ms": finite_float(pool.get("cost_ms") if isinstance(pool, dict) else None),
        "pool_joined": bool(pool),
        **fold_stats,
        **genome_features,
        **dataset_features,
    }
    rec["train_cv_r2_gap"] = safe_sub(rec.get("train_r2"), rec.get("cv_oof_r2"))
    rec["train_cv_mean_gap"] = safe_sub(rec.get("train_r2"), rec.get("cv_mean_r2"))
    rec["train_rmse_over_cv_proxy"] = safe_div(rec.get("train_rmse"), cv_rmse_proxy(rec.get("cv_oof_r2")))
    rec["train_mae_rmse_ratio"] = safe_div(rec.get("train_mae"), rec.get("train_rmse"))
    rec["train_test_rmse_gap_label_only"] = safe_sub(rec.get("label_test_rmse"), rec.get("train_rmse"))
    rec["anchor_jable_available"] = 1.0 if dataset_features.get("anchor_jable_available") else 0.0
    return rec


def cv_rmse_proxy(oof_r2: Any) -> float | None:
    r2 = finite_float(oof_r2)
    if r2 is None:
        return None
    return float(math.sqrt(max(0.0, 1.0 - r2)))


def dataset_feature_summary(dataset_hash: str, meta: dict[str, Any]) -> dict[str, Any]:
    if dataset_hash in _DATASET_FEATURE_CACHE:
        return _DATASET_FEATURE_CACHE[dataset_hash]
    d = REGRESSION_ROOT / dataset_hash
    train_csv = Path(meta.get("train_csv") or d / "train.csv")
    test_csv = Path(meta.get("test_csv") or d / "test.csv")
    vs_path = Path(meta.get("vs_path") or d / "hyperion3train128dim.vs")
    target_vs = Path(meta.get("train_target_vs") or d / "train_target.vs")
    rec: dict[str, Any] = {
        "ds_split_manifest_candidate_rows": finite_float(meta.get("candidate_rows")),
        "ds_split_manifest_best_oof": finite_float(meta.get("best_known_oof_r2")),
    }
    train_shape = csv_shape(train_csv)
    test_shape = csv_shape(test_csv)
    rec.update(
        {
            "ds_train_rows": train_shape[0],
            "ds_train_cols": train_shape[1],
            "ds_test_rows": test_shape[0],
            "ds_test_cols": test_shape[1],
            "ds_train_test_row_ratio": safe_div(train_shape[0], test_shape[0]),
        }
    )
    rec.update(dataset_numeric_stats(train_csv, test_csv))
    try:
        sig = np.asarray(dataset_signature_from_vs(vs_path), dtype=np.float32).reshape(-1)
        rec.update(prefix_stats("ds_sig", sig))
        for i, value in enumerate(sig[:32]):
            rec[f"ds_sig_{i:02d}"] = float(value)
    except Exception as exc:  # noqa: BLE001
        rec["ds_sig_error"] = type(exc).__name__
    try:
        tgt = np.asarray(target_summary_from_vs(target_vs), dtype=np.float32).reshape(-1)
        rec.update(prefix_stats("ds_target", tgt))
        for i, value in enumerate(tgt[:32]):
            rec[f"ds_target_{i:02d}"] = float(value)
    except Exception as exc:  # noqa: BLE001
        rec["ds_target_error"] = type(exc).__name__
    rec["anchor_jable_available"] = True
    _DATASET_FEATURE_CACHE[dataset_hash] = rec
    return rec


def dataset_numeric_stats(train_csv: Path, test_csv: Path) -> dict[str, Any]:
    cache_key = str(train_csv)
    if cache_key in _DATASET_NUMERIC_CACHE:
        return _DATASET_NUMERIC_CACHE[cache_key]
    rec: dict[str, Any] = {}
    if not train_csv.exists():
        _DATASET_NUMERIC_CACHE[cache_key] = rec
        return rec
    try:
        train_df = pd.read_csv(train_csv, header=None)
        test_df = pd.read_csv(test_csv, header=None) if test_csv.exists() else None
    except Exception as exc:  # noqa: BLE001
        rec["dsstat_error"] = type(exc).__name__
        _DATASET_NUMERIC_CACHE[cache_key] = rec
        return rec

    train_num = train_df.apply(pd.to_numeric, errors="coerce")
    num_cols = int(train_num.shape[1])
    rec["dsstat_numeric_col_ratio"] = 1.0 if num_cols else 0.0
    rec["dsstat_missing_mean"] = float(train_num.isna().mean().mean()) if num_cols else None
    rec["dsstat_missing_max"] = float(train_num.isna().mean().max()) if num_cols else None
    rec["dsstat_duplicate_row_ratio"] = safe_div(int(train_df.duplicated().sum()), len(train_df))
    nunique = train_df.nunique(dropna=False)
    rec["dsstat_constant_cols"] = float((nunique <= 1).sum())
    rec["dsstat_near_constant_cols"] = float((train_num.std(numeric_only=True, ddof=0).fillna(0.0) <= 1e-6).sum())

    means = train_num.mean(numeric_only=True).to_numpy(np.float64, copy=False)
    stds = train_num.std(numeric_only=True, ddof=0).to_numpy(np.float64, copy=False)
    skews = train_num.skew(numeric_only=True).to_numpy(np.float64, copy=False)
    kurts = train_num.kurt(numeric_only=True).to_numpy(np.float64, copy=False)
    rec.update(prefix_stats("dsstat_col_mean", means))
    rec.update(prefix_stats("dsstat_col_std", stds))
    rec.update(prefix_stats("dsstat_col_skew", skews))
    rec.update(prefix_stats("dsstat_col_kurt", kurts))

    corr_cols = min(24, train_num.shape[1])
    if corr_cols >= 2:
        corr = train_num.iloc[:, :corr_cols].corr().to_numpy(np.float64, copy=False)
        tri = corr[np.triu_indices_from(corr, k=1)]
        rec["dsstat_abs_corr_mean"] = float(np.nanmean(np.abs(tri))) if tri.size else None
        rec["dsstat_abs_corr_p90"] = float(np.nanpercentile(np.abs(tri), 90)) if tri.size else None
    if test_df is not None:
        test_num = test_df.apply(pd.to_numeric, errors="coerce")
        common = min(train_num.shape[1], test_num.shape[1], 32)
        if common > 0:
            train_mean = train_num.iloc[:, :common].mean(numeric_only=True).to_numpy(np.float64, copy=False)
            test_mean = test_num.iloc[:, :common].mean(numeric_only=True).to_numpy(np.float64, copy=False)
            train_std = train_num.iloc[:, :common].std(numeric_only=True, ddof=0).replace(0.0, np.nan).to_numpy(np.float64, copy=False)
            test_std = test_num.iloc[:, :common].std(numeric_only=True, ddof=0).to_numpy(np.float64, copy=False)
            shift = np.abs(train_mean - test_mean) / np.where(np.isfinite(train_std), train_std, 1.0)
            rec["shift_mean_abs_z_mean"] = float(np.nanmean(shift))
            rec["shift_mean_abs_z_p90"] = float(np.nanpercentile(shift, 90))
            std_shift = np.abs(train_std - test_std) / np.where(np.abs(train_std) > 1e-12, np.abs(train_std), 1.0)
            rec["shift_std_ratio_mean"] = float(np.nanmean(std_shift))
    _DATASET_NUMERIC_CACHE[cache_key] = rec
    return rec


def csv_shape(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    rows = 0
    cols: int | None = None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if cols is None:
                cols = len(row)
            rows += 1
    return rows, cols


def prefix_stats(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite.size == 0:
        return {f"{prefix}_{name}": None for name in ("mean", "std", "min", "max", "p10", "p90")}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_p10": float(np.percentile(finite, 10)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
    }


def summarize_folds(values: list[float]) -> dict[str, float | None]:
    rec: dict[str, float | None] = {}
    if not values:
        for key in (
            "cv_fold_min",
            "cv_fold_max",
            "cv_fold_std",
            "cv_fold_range",
            "cvrel_median",
            "cvrel_iqr",
            "cvrel_skew",
            "cvrel_kurt",
            "cvrel_se_mean",
            "cvrel_lcb95",
            "cvrel_bootstrap_top1_proxy",
            "cvrel_best_minus_median",
            "cvrel_worst_minus_median",
            "cvrel_cov",
            "cvrel_loo_min",
            "cvrel_loo_max",
        ):
            rec[key] = None
        return rec
    arr = np.asarray(values, dtype=np.float64)
    sorted_arr = np.sort(arr)
    median = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    se_mean = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    for i, value in enumerate(arr[:8]):
        rec[f"cv_fold_{i}"] = float(value)
    for i, value in enumerate(sorted_arr[:8]):
        rec[f"cv_sorted_{i}"] = float(value)
    loo_means = []
    for i in range(len(arr)):
        keep = np.delete(arr, i)
        loo_means.append(float(np.mean(keep)) if keep.size else float(arr[i]))
    rec.update(
        {
            "cv_fold_min": float(np.min(arr)),
            "cv_fold_max": float(np.max(arr)),
            "cv_fold_std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "cv_fold_range": float(np.max(arr) - np.min(arr)),
            "cvrel_median": median,
            "cvrel_iqr": q75 - q25,
            "cvrel_skew": float(pd.Series(arr).skew()) if len(arr) >= 3 else 0.0,
            "cvrel_kurt": float(pd.Series(arr).kurt()) if len(arr) >= 4 else 0.0,
            "cvrel_se_mean": se_mean,
            "cvrel_lcb95": float(np.mean(arr) - 1.96 * se_mean),
            "cvrel_bootstrap_top1_proxy": float(np.mean(arr) - 0.5 * np.std(arr)),
            "cvrel_best_minus_median": float(np.max(arr) - median),
            "cvrel_worst_minus_median": float(np.min(arr) - median),
            "cvrel_cov": safe_div(np.std(arr), abs(np.mean(arr)) + 1e-6),
            "cvrel_loo_min": float(np.min(loo_means)),
            "cvrel_loo_max": float(np.max(loo_means)),
        }
    )
    return rec


def parse_genome_features(genome: str, family_bitmap: Any = None) -> dict[str, Any]:
    text = genome or ""
    lower = text.lower()
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    ops = [op.lower() for op in OP_RE.findall(text) if op.lower() not in {"pipe", "x"}]
    token_counts = Counter(tokens)
    op_counts = Counter(ops)
    params = [(k.lower(), float(v)) for k, v in PARAM_RE.findall(text)]
    stages = parse_pipeline_stages(text)
    rec: dict[str, Any] = {
        "genome_len": len(text),
        "genome_token_count": len(tokens),
        "genome_unique_token_count": len(token_counts),
        "genome_op_count": len(ops),
        "genome_unique_op_count": len(op_counts),
        "genome_append_true_count": lower.count("append=true"),
        "genome_append_false_count": lower.count("append=false"),
        "genome_numeric_literal_count": sum(1 for tok in tokens if tok[0].isdigit() or tok[0] in "+-"),
        "head_type": ops[-1] if ops else "",
    }
    denom = rec["genome_append_true_count"] + rec["genome_append_false_count"]
    rec["genome_append_true_ratio"] = safe_div(rec["genome_append_true_count"], denom)
    for op, count in op_counts.most_common(24):
        rec[f"op={op}"] = float(count)

    bitmap = family_bitmap if isinstance(family_bitmap, dict) else infer_family_bitmap(lower)
    for family in sorted(FAMILY_TOKENS):
        rec[f"family={family}"] = 1.0 if bool(bitmap.get(family)) else 0.0
    for bucket, count in hashed_ngrams(tokens, buckets=NGRAM_BUCKETS).items():
        rec[f"ngram_{bucket:03d}"] = float(count)

    rec.update(structured_genome_features(stages, params, rec["head_type"], lower))
    return rec


def parse_pipeline_stages(genome: str) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for match in re.finditer(r"\(([A-Za-z_][A-Za-z0-9_]*)([^()]*)\)", genome):
        op = match.group(1).lower()
        if op in {"pipe", "x"}:
            continue
        body = match.group(2)
        params = {k.lower(): float(v) for k, v in PARAM_RE.findall(body)}
        append = "append=true" in body.lower()
        stages.append({"op": op, "params": params, "append": append})
    return stages


def structured_genome_features(
    stages: list[dict[str, Any]],
    params: list[tuple[str, float]],
    head_type: str,
    lower_genome: str,
) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    transforms = stages[:-1] if stages else []
    head = stages[-1] if stages else None
    rec["gstruct_stage_count"] = float(len(stages))
    rec["gstruct_transform_count"] = float(len(transforms))
    rec["gstruct_head_known"] = 1.0 if head is not None else 0.0
    rec["gstruct_append_expansion_est"] = float(sum(1 for s in transforms if s.get("append")))
    rec["gstruct_target_aware_count"] = float(sum(1 for s in stages if s["op"] in TARGET_AWARE_OPS))
    rec["gstruct_rankish_count"] = float(sum(1 for s in stages if s["op"] in RANKISH_OPS))
    rec["gstruct_stochastic_count"] = float(sum(1 for s in stages if s["op"] in STOCHASTIC_OPS))
    rec["gstruct_repeated_nonlinear_count"] = float(sum(1 for s in stages if s["op"] in NONLINEAR_REPEAT_OPS))
    rec["gstruct_first_transform"] = transforms[0]["op"] if transforms else ""
    rec["gstruct_last_transform"] = transforms[-1]["op"] if transforms else ""
    rec["gstruct_last_before_head"] = transforms[-1]["op"] if transforms else ""
    rec["gstruct_has_repeated_op"] = 1.0 if len({s["op"] for s in stages}) < len(stages) else 0.0
    rec["gstruct_redflag_target_bin"] = 1.0 if "target_bin_encode" in lower_genome else 0.0
    rec["gstruct_redflag_repeated_sigmoid"] = 1.0 if lower_genome.count("sigmoid") >= 2 else 0.0
    rec["gstruct_redflag_repeated_rank"] = 1.0 if lower_genome.count("rank") >= 2 else 0.0
    rec["gstruct_est_feature_explosion"] = float(
        sum(1.0 + len(s.get("params", {})) for s in transforms if s["op"] in {"cheb", "poly_interact", "bin_onehot", "quantile_bins"})
    )
    if head_type:
        rec[f"gstruct_head={head_type}"] = 1.0
    for i, stage in enumerate(stages[:8]):
        rec[f"gstruct_stage_{i}_op={stage['op']}"] = 1.0
        rec[f"gstruct_stage_{i}_append"] = 1.0 if stage.get("append") else 0.0
    for a, b in zip(stages, stages[1:], strict=False):
        rec[f"gstruct_pair={a['op']}->{b['op']}"] = rec.get(f"gstruct_pair={a['op']}->{b['op']}", 0.0) + 1.0
    for a, b, c in zip(stages, stages[1:], stages[2:], strict=False):
        key = f"gstruct_triple={a['op']}->{b['op']}->{c['op']}"
        rec[key] = rec.get(key, 0.0) + 1.0
    for key, value in params:
        binned = bucket_numeric_param(key, value)
        rec[f"gparam_{key}_value"] = value
        rec[f"gparam_{key}_bin={binned}"] = 1.0
        if head_type:
            rec[f"gparam_head={head_type}:{key}"] = value
    return rec


def bucket_numeric_param(name: str, value: float) -> str:
    name = name.lower()
    if name in {"k", "topk", "bins", "m", "steps", "rounds"}:
        if value <= 4:
            return "tiny"
        if value <= 16:
            return "small"
        if value <= 64:
            return "medium"
        if value <= 256:
            return "large"
        return "xlarge"
    if name in {"degree", "depth"}:
        if value <= 2:
            return "shallow"
        if value <= 5:
            return "mid"
        return "deep"
    if name in {"lr", "alpha", "gamma", "q"}:
        if value < 0.03:
            return "low"
        if value < 0.15:
            return "mid"
        return "high"
    return "other"


def infer_family_bitmap(lower_genome: str) -> dict[str, bool]:
    out = {}
    for family, needles in FAMILY_TOKENS.items():
        out[family] = any(needle in lower_genome for needle in needles)
    return out


def hashed_ngrams(tokens: list[str], *, buckets: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for n in (1, 2, 3):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest, "little") % buckets
            counts[bucket] += 1
    return counts


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_dataset_oracle_rmse"] = df.groupby("dataset_hash")["label_test_rmse"].transform("min")
    df["label_regret"] = df["label_test_rmse"] - df["label_dataset_oracle_rmse"]
    df["label_test_rank"] = df.groupby("dataset_hash")["label_test_rmse"].rank(method="min", ascending=True)
    group_size = df.groupby("dataset_hash")["label_test_rmse"].transform("count").clip(lower=1)
    df["label_test_percentile"] = np.where(group_size > 1, (df["label_test_rank"] - 1.0) / (group_size - 1.0), 0.0)
    df["label_oracle_top1"] = (df["label_test_rank"] <= 1.0).astype(np.float32)
    df["label_oracle_top3"] = (df["label_test_rank"] <= 3.0).astype(np.float32)
    df["label_beats_robust"] = 0.0
    robust_best = df.loc[df.groupby("dataset_hash")["cv_score_robust"].idxmax()][["dataset_hash", "label_test_rmse"]]
    robust_best = robust_best.rename(columns={"label_test_rmse": "robust_pick_rmse"})
    df = df.merge(robust_best, on="dataset_hash", how="left")
    df["label_beats_robust"] = (df["label_test_rmse"] <= df["robust_pick_rmse"] + 1e-12).astype(np.float32)
    df["label_relevance"] = np.rint((1.0 - df["label_test_percentile"]) * 30.0).astype(np.float32)
    return df


def add_within_dataset_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    metric_specs = (
        ("cv_oof_r2", False),
        ("cv_mean_r2", False),
        ("cv_score_robust", False),
        ("timing_total_ms", True),
        ("pool_selection_key", False),
        ("pool_reward_mean", False),
        ("pool_reward_lcb", False),
        ("train_cv_r2_gap", True),
    )
    for col, asc in metric_specs:
        rank_col = f"ctx_rank_{col}"
        pct_col = f"ctx_pct_{col}"
        df[rank_col] = df.groupby("dataset_hash")[col].rank(method="average", ascending=asc, na_option="bottom")
        size = df.groupby("dataset_hash")[col].transform("count").replace(0, np.nan)
        df[pct_col] = df[rank_col] / size

    out_frames = []
    for _, group in df.groupby("dataset_hash", sort=False):
        work = group.copy()
        size = len(work)
        work["ctx_candidate_count"] = float(size)
        work["ctx_head_diversity"] = float(work["head_type"].fillna("").nunique())
        work["ctx_engine_diversity"] = float(work["engine"].fillna("").nunique())
        work["ctx_duplicate_genome_count"] = float(size - work["canonical_genome"].nunique())
        work["ctx_target_aware_frac"] = float((work.get("family=target_aware", 0.0) > 0).mean()) if "family=target_aware" in work.columns else 0.0
        work["ctx_high_gap_frac"] = float((work["train_cv_r2_gap"].fillna(0.0) > 0.05).mean())
        robust_scores = work["cv_score_robust"].astype(float)
        work["ctx_cv_entropy"] = shannon_entropy(robust_scores.to_numpy(float))
        work["ctx_topk_robust_spread"] = float(robust_scores.nlargest(min(5, size)).max() - robust_scores.nlargest(min(5, size)).min()) if size else 0.0
        work["ctx_robust_z"] = groupwise_zscore(work["cv_score_robust"])
        work["ctx_oof_z"] = groupwise_zscore(work["cv_oof_r2"])
        work["ctx_mean_z"] = groupwise_zscore(work["cv_mean_r2"])
        work["ctx_policy_z"] = groupwise_zscore(work["pool_selection_key"])
        work["consensus_borda"] = (
            -(work["ctx_rank_cv_oof_r2"] + work["ctx_rank_cv_mean_r2"] + work["ctx_rank_cv_score_robust"] + work["ctx_rank_pool_selection_key"].fillna(size))
        )
        work["consensus_rrf"] = (
            reciprocal_rank(work["ctx_rank_cv_oof_r2"])
            + reciprocal_rank(work["ctx_rank_cv_mean_r2"])
            + reciprocal_rank(work["ctx_rank_cv_score_robust"])
            + reciprocal_rank(work["ctx_rank_pool_selection_key"])
        )
        work["consensus_disagreement_policy_cv"] = np.abs(work["ctx_pct_pool_selection_key"].fillna(1.0) - work["ctx_pct_cv_score_robust"])

        robust_top = work.sort_values("cv_score_robust", ascending=False, kind="stable").iloc[0]
        policy_top = work.sort_values("pool_selection_key", ascending=False, kind="stable").iloc[0] if work["pool_selection_key"].notna().any() else robust_top
        oof_top = work.sort_values("cv_oof_r2", ascending=False, kind="stable").iloc[0]
        mean_top = work.sort_values("cv_mean_r2", ascending=False, kind="stable").iloc[0]
        fast_top = work.sort_values(["timing_total_ms", "cv_score_robust"], ascending=[True, False], kind="stable").iloc[0]
        for idx, row in work.iterrows():
            head_type = str(row.get("head_type") or "")
            same_head = work[work["head_type"] == head_type] if head_type else work.iloc[:0]
            same_head_top = same_head.sort_values("cv_score_robust", ascending=False, kind="stable").iloc[0] if not same_head.empty else robust_top
            updates = comparative_context_row(row, robust_top, policy_top, oof_top, mean_top, fast_top, same_head_top)
            for key, value in updates.items():
                work.at[idx, key] = value
        out_frames.append(work)
    out = pd.concat(out_frames, ignore_index=True)
    out["cvrel_gap_robust_mean"] = out["cv_score_robust"] - out["cv_mean_r2"]
    out["cvrel_gap_oof_mean"] = out["cv_oof_r2"] - out["cv_mean_r2"]
    out["cvrel_gap_robust_oof"] = out["cv_score_robust"] - out["cv_oof_r2"]
    return out


def comparative_context_row(
    row: pd.Series,
    robust_top: pd.Series,
    policy_top: pd.Series,
    oof_top: pd.Series,
    mean_top: pd.Series,
    fast_top: pd.Series,
    same_head_top: pd.Series,
) -> dict[str, float]:
    return {
        "delta_vs_top_robust_cv": safe_sub(row.get("cv_score_robust"), robust_top.get("cv_score_robust")) or 0.0,
        "delta_vs_top_robust_oof": safe_sub(row.get("cv_oof_r2"), robust_top.get("cv_oof_r2")) or 0.0,
        "delta_vs_top_policy_selection": safe_sub(row.get("pool_selection_key"), policy_top.get("pool_selection_key")) or 0.0,
        "delta_vs_top_policy_cv": safe_sub(row.get("cv_score_robust"), policy_top.get("cv_score_robust")) or 0.0,
        "delta_vs_fast_good_cv": safe_sub(row.get("cv_score_robust"), fast_top.get("cv_score_robust")) or 0.0,
        "delta_vs_same_head_cv": safe_sub(row.get("cv_score_robust"), same_head_top.get("cv_score_robust")) or 0.0,
        "ctx_margin_to_rank2": rank_margin_from_top(row, robust_top, "cv_score_robust"),
        "ctx_margin_oof_to_best": safe_sub(row.get("cv_oof_r2"), oof_top.get("cv_oof_r2")) or 0.0,
        "ctx_margin_mean_to_best": safe_sub(row.get("cv_mean_r2"), mean_top.get("cv_mean_r2")) or 0.0,
        "ctx_is_robust_top": 1.0 if row.name == robust_top.name else 0.0,
        "ctx_is_policy_top": 1.0 if row.name == policy_top.name else 0.0,
        "ctx_is_same_head_top": 1.0 if row.name == same_head_top.name else 0.0,
    }


def rank_margin_from_top(row: pd.Series, best_row: pd.Series, col: str) -> float:
    return safe_sub(row.get(col), best_row.get(col)) or 0.0


def groupwise_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mu = values.mean()
    sigma = values.std(ddof=0)
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 1e-12:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((values - mu) / sigma).astype(np.float32)


def reciprocal_rank(series: pd.Series) -> pd.Series:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(series.max(skipna=True) if series.notna().any() else 999.0)
    return 1.0 / (50.0 + values)


def shannon_entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    shifted = arr - np.min(arr)
    probs = shifted + 1e-6
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def compute_cv_red_flag_label(df: pd.DataFrame) -> np.ndarray:
    high_cv = df["ctx_pct_cv_score_robust"].fillna(1.0) <= 0.2
    poor_test = df["label_test_percentile"].fillna(0.0) >= 0.5
    big_gap = df["label_regret"].fillna(0.0) > df.groupby("dataset_hash")["label_regret"].transform("median").fillna(0.0)
    return (high_cv & poor_test & big_gap).astype(np.float32).to_numpy()


def row_to_features(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key, value in row.items():
        if is_feature_excluded(key):
            continue
        if isinstance(value, bool):
            features[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                features[key] = float(value)
        elif isinstance(value, str):
            if value:
                features[f"{key}={value[:120]}"] = 1.0
    return features


def is_feature_excluded(key: str) -> bool:
    excluded_exact = {
        "canonical_genome",
        "dataset_hash",
        "source_path",
        "source_ordinal",
        "candidate_ordinal",
        "train_test_rmse_gap_label_only",
        "robust_pick_rmse",
    }
    if key in excluded_exact:
        return True
    return key.startswith("label_") or key.startswith("score_meta_selector")


def assert_no_feature_leakage(feature_names: list[str]) -> None:
    bad = [name for name in feature_names if name.startswith(LEAKY_PREFIXES)]
    if bad:
        raise AssertionError(f"feature leakage columns detected: {bad[:10]}")


def assert_no_validation_leakage(train_df: pd.DataFrame, validation_hashes: set[str]) -> None:
    overlap = sorted(set(train_df["dataset_hash"].astype(str)) & validation_hashes)
    if overlap:
        raise AssertionError(f"validation hashes leaked into train rows: {overlap[:10]}")


def build_feature_schema(feature_names: list[str]) -> dict[str, Any]:
    families = {}
    for family, prefixes in FEATURE_FAMILIES.items():
        matched = [name for name in feature_names if name.startswith(prefixes)]
        families[family] = {"prefixes": list(prefixes), "features": matched}
    assigned = set(itertools.chain.from_iterable(v["features"] for v in families.values()))
    families["other"] = {"prefixes": [], "features": sorted(set(feature_names) - assigned)}
    return {"schema_version": "full_cv_meta_selector_v2.feature_schema", "families": families}


def cross_validate_components(
    *,
    train_df: pd.DataFrame,
    x_train: sparse.csr_matrix,
    groups: np.ndarray,
    splitter: GroupKFold,
    feature_names: list[str],
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    n = len(train_df)
    y_rmse = train_df["label_test_rmse"].to_numpy(np.float32)
    y_rel = train_df["label_relevance"].to_numpy(np.float32)
    y_risk = train_df["label_cv_red_flag"].to_numpy(np.float32)
    stage1_reg = np.full(n, np.nan, dtype=np.float32)
    stage1_rank = np.full(n, np.nan, dtype=np.float32)
    risk_pred = np.full(n, np.nan, dtype=np.float32)
    pairwise_pred = np.full(n, np.nan, dtype=np.float32)
    component_compare: list[dict[str, Any]] = []

    last_models: dict[str, Any] = {}
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_rmse, groups), start=1):
        log(f"components fold {fold}/{splitter.n_splits}")
        reg_model = train_stage1_regressor(x_train[tr_idx], y_rmse[tr_idx], seed=random_state + fold)
        stage1_reg[va_idx] = -np.asarray(reg_model.predict(x_train[va_idx]), dtype=np.float32)
        rank_model = train_stage1_ranker(x_train, y_rel, groups, tr_idx, seed=random_state + 100 + fold)
        stage1_rank[va_idx] = predict_stage1_ranker(rank_model, x_train[va_idx])
        risk_model = train_risk_gate(x_train[tr_idx], y_risk[tr_idx], seed=random_state + 200 + fold)
        risk_pred[va_idx] = predict_binary_model(risk_model, x_train[va_idx])
        pair_model = train_pairwise_anchor_model(train_df.iloc[tr_idx].reset_index(drop=True), x_train[tr_idx], seed=random_state + 300 + fold)
        pairwise_pred[va_idx] = predict_pairwise_anchor_scores(
            df=train_df.iloc[va_idx].reset_index(drop=True),
            x=x_train[va_idx],
            model=pair_model,
        )
        last_models = {
            "stage1_reg": reg_model,
            "stage1_rank": rank_model,
            "risk_gate": risk_model,
            "pairwise": pair_model,
        }

    work = train_df.copy()
    work["score_stage1_reg"] = stage1_reg
    work["score_stage1_rank"] = stage1_rank
    work["score_risk_gate"] = risk_pred
    work["score_pairwise"] = pairwise_pred
    for name in ("score_stage1_reg", "score_stage1_rank", "score_pairwise"):
        metric = evaluate_scores(work, work[name].to_numpy(np.float32))
        metric.update({"model": name, "kind": "component", "status": "ok"})
        component_compare.append(metric)

    return (
        {
            "score_stage1_reg": stage1_reg,
            "score_stage1_rank": stage1_rank,
            "score_risk_gate": risk_pred,
            "score_pairwise": pairwise_pred,
        },
        last_models,
        component_compare,
    )


def train_stage1_regressor(x: sparse.csr_matrix, y_rmse: np.ndarray, *, seed: int) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        random_state=seed,
        n_jobs=MODEL_THREADS,
        verbosity=-1,
    )
    model.fit(x, y_rmse)
    return model


def train_stage1_ranker(
    x: sparse.csr_matrix,
    y_rel: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
) -> Any:
    import lightgbm as lgb

    tr_sorted = sort_indices_by_group(indices, groups)
    group_sizes = group_sizes_for_indices(tr_sorted, groups)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        random_state=seed,
        n_jobs=MODEL_THREADS,
        verbosity=-1,
    )
    model.fit(x[tr_sorted], y_rel[tr_sorted], group=group_sizes)
    return model


def predict_stage1_ranker(model: Any, x: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=np.float32).reshape(-1)


def train_risk_gate(x: sparse.csr_matrix, y: np.ndarray, *, seed: int) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=24,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=30,
        random_state=seed,
        n_jobs=MODEL_THREADS,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_binary_model(model: Any, x: sparse.csr_matrix) -> np.ndarray:
    prob = model.predict_proba(x)
    if isinstance(prob, list):
        prob = np.asarray(prob)
    return np.asarray(prob[:, 1], dtype=np.float32).reshape(-1)


def train_pairwise_anchor_model(df: pd.DataFrame, x: sparse.csr_matrix, *, seed: int) -> Any:
    examples_x = []
    examples_y = []
    for _, group in df.groupby("dataset_hash", sort=False):
        anchors = anchor_rows_for_group(group)
        hard = group.sort_values("cv_score_robust", ascending=False, kind="stable").head(PAIRWISE_HARD_TOP_K)
        for _, cand in hard.iterrows():
            for anchor in anchors:
                if int(cand.name) == int(anchor.name):
                    continue
                left = sparse_difference_row(x, int(cand.name), int(anchor.name))
                winner = 1 if float(cand["label_test_rmse"]) < float(anchor["label_test_rmse"]) else 0
                examples_x.append(left)
                examples_y.append(winner)
                examples_x.append(-left)
                examples_y.append(1 - winner)
    if not examples_x:
        return None
    xx = sparse.vstack(examples_x).tocsr()
    yy = np.asarray(examples_y, dtype=np.int32)
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(xx, yy)
    return model


def anchor_rows_for_group(group: pd.DataFrame) -> list[pd.Series]:
    anchors = []
    anchors.append(group.sort_values("cv_score_robust", ascending=False, kind="stable").iloc[0])
    anchors.append(group.sort_values("cv_oof_r2", ascending=False, kind="stable").iloc[0])
    anchors.append(group.sort_values("cv_mean_r2", ascending=False, kind="stable").iloc[0])
    if group["pool_selection_key"].notna().any():
        anchors.append(group.sort_values("pool_selection_key", ascending=False, kind="stable").iloc[0])
    anchors.append(group.sort_values(["timing_total_ms", "cv_score_robust"], ascending=[True, False], kind="stable").iloc[0])
    unique = []
    seen = set()
    for row in anchors:
        idx = int(row.name)
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(row)
    return unique[:PAIRWISE_TOP_K]


def sparse_difference_row(x: sparse.csr_matrix, left_idx: int, right_idx: int) -> sparse.csr_matrix:
    return x[left_idx] - x[right_idx]


def predict_pairwise_anchor_scores(df: pd.DataFrame, x: sparse.csr_matrix, model: Any) -> np.ndarray:
    if model is None:
        return np.zeros(len(df), dtype=np.float32)
    scores = np.zeros(len(df), dtype=np.float32)
    for _, group in df.groupby("dataset_hash", sort=False):
        anchors = anchor_rows_for_group(group)
        anchor_positions = [int(pos) for pos in group.index if any(int(pos) == int(anchor.name) for anchor in anchors)]
        if not anchor_positions:
            continue
        pos_map = {int(idx): i for i, idx in enumerate(group.index.tolist())}
        for idx in group.index:
            values = []
            left_pos = pos_map[int(idx)]
            for anchor_idx in anchor_positions:
                if int(idx) == anchor_idx:
                    continue
                right_pos = pos_map[anchor_idx]
                diff = x[group.index[left_pos]] - x[group.index[right_pos]]
                prob = predict_binary_model(model, diff)
                values.append(float(prob[0]))
            scores[group.index.get_loc(idx) if isinstance(group.index, pd.RangeIndex) else group.index.tolist().index(idx)] = float(np.mean(values)) if values else 0.5
    if len(df.index) and not isinstance(df.index, pd.RangeIndex):
        aligned = np.zeros(len(df), dtype=np.float32)
        for i, idx in enumerate(df.index.tolist()):
            aligned[i] = scores[i]
        return aligned
    return scores


def choose_blend_weights(train_df: pd.DataFrame) -> dict[str, Any]:
    candidate_weights = []
    for robust_w in (0.4, 0.6):
        for stage1_w in (0.4, 0.6):
            for pairwise_w in (0.0, 0.2):
                for gate_w in (0.0, 0.2):
                    for risk_w in (0.1, 0.2):
                        candidate_weights.append(
                            {
                                "robust": robust_w,
                                "stage1": stage1_w,
                                "pairwise": pairwise_w,
                                "gate": gate_w,
                                "risk": risk_w,
                                "redflag": 0.1,
                            }
                        )
    components = compute_blend_components(train_df)
    scored = []
    baseline_robust = evaluate_scores(train_df, train_df["cv_score_robust"].to_numpy(np.float32))
    for weights in candidate_weights:
        score = blend_from_components(components, weights)
        metrics = evaluate_scores(train_df, score)
        picks = pick_rows(train_df.assign(_score=score), "_score")
        robust_picks = pick_rows(train_df.assign(_score=train_df["cv_score_robust"]), "_score")
        beat_robust = mean_or_none(
            [
                1.0 if float(picks[h]["label_test_rmse"]) <= float(robust_picks[h]["label_test_rmse"]) + 1e-12 else 0.0
                for h in picks
                if h in robust_picks
            ]
        )
        metrics["win_rate_vs_robust"] = beat_robust
        metrics["improvement_vs_robust_regret"] = (
            None
            if metrics["mean_regret_at_1"] is None or baseline_robust["mean_regret_at_1"] is None
            else float(baseline_robust["mean_regret_at_1"]) - float(metrics["mean_regret_at_1"])
        )
        scored.append((weights, metrics))
    scored.sort(
        key=lambda item: (
            float(item[1]["mean_regret_at_1"]) if item[1]["mean_regret_at_1"] is not None else float("inf"),
            -float(item[1]["top1_oracle_capture"] or 0.0),
            -float(item[1]["top3_oracle_capture"] or 0.0),
            -float(item[1]["win_rate_vs_robust"] or 0.0),
        )
    )
    best_weights, best_metrics = scored[0]
    return {"weights": best_weights, "train_cv_metrics": best_metrics, "search_size": len(scored)}


def score_with_blend(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return blend_from_components(compute_blend_components(df), weights)


def compute_blend_components(df: pd.DataFrame) -> dict[str, np.ndarray]:
    robust = grouped_component(df, "cv_score_robust")
    stage1 = grouped_component(df, ["score_stage1_reg", "score_stage1_rank"])
    pairwise = grouped_component(df, "score_pairwise")
    risk = (
        df["score_risk_gate"].fillna(0.0).to_numpy(np.float32)
        if "score_risk_gate" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    redflag = rule_redflag_penalty(df)
    gate = pairwise * robust_top_risk(df)
    return {
        "robust": robust,
        "stage1": stage1,
        "pairwise": pairwise,
        "risk": risk,
        "redflag": redflag,
        "gate": gate,
    }


def blend_from_components(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    robust = components["robust"]
    stage1 = components["stage1"]
    pairwise = components["pairwise"]
    gate = components["gate"]
    risk = components["risk"]
    redflag = components["redflag"]
    return (
        weights["robust"] * robust
        + weights["stage1"] * stage1
        + weights["pairwise"] * pairwise
        + weights["gate"] * gate
        - weights["risk"] * risk
        - weights["redflag"] * redflag
    ).astype(np.float32)


def grouped_component(df: pd.DataFrame, cols: str | list[str]) -> np.ndarray:
    if isinstance(cols, str):
        values = df[cols].fillna(0.0).astype(float)
    else:
        values = df[cols].fillna(0.0).astype(float).mean(axis=1)
    out = np.zeros(len(df), dtype=np.float32)
    for _, group in df.assign(_value=values).groupby("dataset_hash", sort=False):
        z = groupwise_zscore(group["_value"])
        out[group.index.to_numpy()] = z.to_numpy(np.float32)
    return out


def rule_redflag_penalty(df: pd.DataFrame) -> np.ndarray:
    penalty = np.zeros(len(df), dtype=np.float32)
    for i, row in enumerate(df.to_dict("records")):
        value = 0.0
        if finite_float(row.get("gstruct_redflag_target_bin")):
            value += 0.35 * float(row.get("gstruct_redflag_target_bin") or 0.0)
        if finite_float(row.get("gstruct_redflag_repeated_sigmoid")):
            value += 0.25 * float(row.get("gstruct_redflag_repeated_sigmoid") or 0.0)
        if finite_float(row.get("gstruct_redflag_repeated_rank")):
            value += 0.2 * float(row.get("gstruct_redflag_repeated_rank") or 0.0)
        if finite_float(row.get("gstruct_est_feature_explosion")) and finite_float(row.get("ds_train_rows")):
            if float(row["gstruct_est_feature_explosion"]) >= 4.0 and float(row["ds_train_rows"]) <= 200:
                value += 0.3
        penalty[i] = value
    return penalty


def robust_top_risk(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float32)
    if "score_risk_gate" not in df.columns:
        return out
    for _, group in df.groupby("dataset_hash", sort=False):
        top = group.sort_values("cv_score_robust", ascending=False, kind="stable").iloc[0]
        risk = float(top.get("score_risk_gate") or 0.0)
        out[group.index.to_numpy()] = risk
    return out


def build_model_compare(train_df: pd.DataFrame, component_compare: list[dict[str, Any]], blend_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    baseline_specs = {
        "cv_mean_full": train_df["cv_mean_r2"].to_numpy(np.float32),
        "oof_cv_full": train_df["cv_oof_r2"].to_numpy(np.float32),
        "robust_cv_full": train_df["cv_score_robust"].to_numpy(np.float32),
    }
    for name, scores in baseline_specs.items():
        metric = evaluate_scores(train_df, scores)
        metric.update({"model": name, "kind": "baseline", "status": "ok"})
        rows.append(metric)
    rows.extend(component_compare)
    blend_metric = evaluate_scores(train_df, train_df["score_meta_selector_v2_cvfit"].to_numpy(np.float32))
    blend_metric.update(
        {
            "model": "meta_selector_v2_blend",
            "kind": "blended",
            "status": "ok",
            "weight_search_size": blend_result["search_size"],
            "weights": json.dumps(blend_result["weights"], sort_keys=True),
            "win_rate_vs_robust": blend_result["train_cv_metrics"].get("win_rate_vs_robust"),
        }
    )
    rows.append(blend_metric)
    return rows


def run_feature_ablations(
    *,
    train_df: pd.DataFrame,
    x_train: sparse.csr_matrix,
    groups: np.ndarray,
    splitter: GroupKFold,
    feature_names: list[str],
    random_state: int,
) -> list[dict[str, Any]]:
    family_masks = feature_family_masks(feature_names)
    ablation_specs = [
        ("base", ["base"]),
        ("base_plus_cv_reliability", ["base", "cv_reliability"]),
        ("base_plus_pool_context", ["base", "pool_context"]),
        ("base_plus_genome_structure", ["base", "genome_structure"]),
        ("base_plus_dataset_stats", ["base", "dataset_stats"]),
        ("full", ["base", "cv_reliability", "pool_context", "genome_structure", "dataset_stats", "anchor", "other"]),
    ]
    rows = []
    y_rmse = train_df["label_test_rmse"].to_numpy(np.float32)
    for name, families in ablation_specs:
        cols = np.zeros(len(feature_names), dtype=bool)
        for family in families:
            cols |= family_masks.get(family, np.zeros(len(feature_names), dtype=bool))
        if not cols.any():
            continue
        pred = np.full(len(train_df), np.nan, dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_rmse, groups), start=1):
            model = train_stage1_regressor(x_train[tr_idx][:, cols], y_rmse[tr_idx], seed=random_state + 500 + fold)
            pred[va_idx] = -np.asarray(model.predict(x_train[va_idx][:, cols]), dtype=np.float32)
        metrics = evaluate_scores(train_df, pred)
        metrics.update({"ablation": name, "families": ",".join(families), "feature_count": int(cols.sum())})
        rows.append(metrics)
    return rows


def feature_family_masks(feature_names: list[str]) -> dict[str, np.ndarray]:
    masks = {}
    for family, prefixes in FEATURE_FAMILIES.items():
        masks[family] = np.asarray([name.startswith(prefixes) for name in feature_names], dtype=bool)
    assigned = np.zeros(len(feature_names), dtype=bool)
    for mask in masks.values():
        assigned |= mask
    masks["other"] = ~assigned
    return masks


def fit_final_components(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    feature_names: list[str],
    random_state: int,
) -> dict[str, Any]:
    y_rmse = train_df["label_test_rmse"].to_numpy(np.float32)
    y_rel = train_df["label_relevance"].to_numpy(np.float32)
    y_risk = train_df["label_cv_red_flag"].to_numpy(np.float32)
    groups = train_df["dataset_hash"].astype(str).to_numpy()
    idx = np.arange(len(train_df))
    stage1_reg_model = train_stage1_regressor(x_train, y_rmse, seed=random_state)
    stage1_reg = -np.asarray(stage1_reg_model.predict(x_train), dtype=np.float32)
    stage1_rank_model = train_stage1_ranker(x_train, y_rel, groups, idx, seed=random_state + 100)
    stage1_rank = predict_stage1_ranker(stage1_rank_model, x_train)
    risk_gate_model = train_risk_gate(x_train, y_risk, seed=random_state + 200)
    risk_gate = predict_binary_model(risk_gate_model, x_train)
    pairwise_model = train_pairwise_anchor_model(train_df.reset_index(drop=True), x_train, seed=random_state + 300)
    train_df["score_stage1_reg"] = stage1_reg
    train_df["score_stage1_rank"] = stage1_rank
    train_df["score_risk_gate"] = risk_gate
    train_df["score_pairwise"] = predict_pairwise_anchor_scores(train_df.reset_index(drop=True), x_train, pairwise_model)
    return {
        "stage1_reg": stage1_reg_model,
        "stage1_rank": predict_stage1_ranker(stage1_rank_model, x_val),
        "stage1_rank_model": stage1_rank_model,
        "risk_gate": predict_binary_model(risk_gate_model, x_val),
        "risk_gate_model": risk_gate_model,
        "pairwise_model": pairwise_model,
        "stage1_reg_model": stage1_reg_model,
    }


def evaluate_scores(df: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    work = df[["dataset_hash", "label_test_rmse", "label_regret", "label_relevance", "label_test_rank"]].copy()
    work["score"] = scores
    picks = []
    ndcgs = []
    correlations = []
    top3_hits = []
    for _, group in work.groupby("dataset_hash", sort=False):
        finite = group[np.isfinite(group["score"].to_numpy(dtype=float))]
        if finite.empty:
            continue
        selected = finite.sort_values("score", ascending=False, kind="stable").iloc[0]
        picks.append(selected)
        ranked = finite.sort_values("score", ascending=False, kind="stable")
        ndcgs.append(ndcg_at_k(ranked["label_relevance"].to_numpy(float), k=TOP_NDCG))
        top3_hits.append(1.0 if float(selected["label_test_rank"]) <= 3.0 else 0.0)
        if len(finite) >= 2:
            pred = finite["score"].to_numpy(float)
            actual = -finite["label_test_rmse"].to_numpy(float)
            if np.std(pred) > 1e-12 and np.std(actual) > 1e-12:
                correlations.append(float(np.corrcoef(pred, actual)[0, 1]))
    if not picks:
        return empty_metrics()
    pick_df = pd.DataFrame(picks)
    return {
        "dataset_count": int(pick_df["dataset_hash"].nunique()),
        "mean_regret_at_1": float(pick_df["label_regret"].mean()),
        "median_regret_at_1": float(pick_df["label_regret"].median()),
        "tail_regret_p90": float(np.percentile(pick_df["label_regret"].to_numpy(float), 90)),
        "top1_oracle_capture": float((pick_df["label_regret"] <= 1e-12).mean()),
        "top3_oracle_capture": float(np.mean(top3_hits)),
        "ndcg_at_10": mean_or_none(ndcgs),
        "score_test_correlation_mean": mean_or_none(correlations),
    }


def ndcg_at_k(relevance_sorted_by_pred: np.ndarray, *, k: int) -> float | None:
    rel = np.asarray(relevance_sorted_by_pred[:k], dtype=float)
    if rel.size == 0:
        return None
    ideal = np.sort(np.asarray(relevance_sorted_by_pred, dtype=float))[::-1][:k]
    dcg = float(sum((2.0**gain - 1.0) / math.log2(i + 2) for i, gain in enumerate(rel)))
    idcg = float(sum((2.0**gain - 1.0) / math.log2(i + 2) for i, gain in enumerate(ideal)))
    return dcg / idcg if idcg > 0 else None


def empty_metrics() -> dict[str, Any]:
    return {
        "dataset_count": 0,
        "mean_regret_at_1": None,
        "median_regret_at_1": None,
        "tail_regret_p90": None,
        "top1_oracle_capture": None,
        "top3_oracle_capture": None,
        "ndcg_at_10": None,
        "score_test_correlation_mean": None,
    }


def build_validation_choices(
    *,
    val_df: pd.DataFrame,
    validation_hashes: list[str],
    current_per_ds_path: Path,
    v1_choices_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    choices: dict[str, dict[str, dict[str, Any]]] = {
        "meta_selector_v2": choose_by_score(val_df, "score_meta_selector_v2", higher=True),
        "cv_mean_full": choose_by_score(val_df, "cv_mean_r2", higher=True),
        "oof_cv_full": choose_by_score(val_df, "cv_oof_r2", higher=True),
        "robust_cv_full": choose_by_score(val_df, "cv_score_robust", higher=True),
        "oracle_full": choose_by_score(val_df, "label_test_rmse", higher=False),
    }
    if v1_choices_path.exists():
        choices["meta_selector_v1"] = load_choices_jsonl(v1_choices_path)
    if val_df["pool_selection_key"].notna().sum() >= len(validation_hashes) // 2:
        choices["policy_selection_key_full"] = choose_by_score(val_df, "pool_selection_key", higher=True)
    if current_per_ds_path.exists():
        current = json.loads(current_per_ds_path.read_text(encoding="utf-8"))
        choices["current_top1"] = choose_from_current_per_ds(current, "top1")
        choices["current_cv_select"] = choose_from_current_per_ds(current, "cv_select")
        choices["current_best_k"] = choose_from_current_per_ds(current, "best_k")
    return choices


def choose_by_score(df: pd.DataFrame, score_col: str, *, higher: bool) -> dict[str, dict[str, Any]]:
    out = {}
    for h, group in df.groupby("dataset_hash", sort=False):
        scored = group[group[score_col].notna()]
        if scored.empty:
            continue
        selected = scored.sort_values(score_col, ascending=not higher, kind="stable").iloc[0]
        out[str(h)] = compact_choice_row(selected, score_col=score_col)
    return out


def load_choices_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in iter_jsonl(path):
        h = str(row.get("dataset_hash") or "")
        if h:
            out[h] = dict(row)
    return out


def choose_from_current_per_ds(current: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for h, rows in current.items():
        ok = [r for r in rows if finite_float(r.get("test_rmse")) is not None]
        if not ok:
            continue
        if mode == "top1":
            pick = min(ok, key=lambda r: int(r.get("rank") or 0))
        elif mode == "cv_select":
            withcv = [r for r in ok if finite_float(r.get("score_robust")) is not None or finite_float(r.get("oof_r2")) is not None]
            pick = max(withcv, key=lambda r: probe_score(r)) if withcv else min(ok, key=lambda r: int(r.get("rank") or 0))
        elif mode == "best_k":
            pick = min(ok, key=lambda r: float(r["test_rmse"]))
        else:
            raise ValueError(mode)
        out[str(h)] = {
            "dataset_hash": str(h),
            "engine": pick.get("engine"),
            "canonical_genome": pick.get("genome") or pick.get("canonical_genome"),
            "rank": 0,
            "source_rank": pick.get("rank"),
            "test_rmse": finite_float(pick.get("test_rmse")),
            "test_r2": finite_float(pick.get("test_r2")),
            "oof_r2": finite_float(pick.get("oof_r2")),
            "score_robust": finite_float(pick.get("score_robust")),
            "selection_score": pick.get("selection_score"),
            "mode_source": "current_per_ds",
        }
    return out


def probe_score(row: dict[str, Any]) -> float:
    robust = finite_float(row.get("score_robust"))
    if robust is not None:
        return robust
    oof = finite_float(row.get("oof_r2"))
    return oof if oof is not None else float("-inf")


def compact_choice_row(row: pd.Series, *, score_col: str) -> dict[str, Any]:
    return {
        "dataset_hash": str(row["dataset_hash"]),
        "engine": row.get("engine"),
        "canonical_genome": row.get("canonical_genome"),
        "rank": 0,
        "candidate_ordinal": int(row.get("candidate_ordinal", 0)),
        "test_rmse": finite_float(row.get("label_test_rmse")),
        "test_r2": finite_float(row.get("label_test_r2")),
        "oof_r2": finite_float(row.get("cv_oof_r2")),
        "mean_r2": finite_float(row.get("cv_mean_r2")),
        "score_robust": finite_float(row.get("cv_score_robust")),
        "selector_score": finite_float(row.get(score_col)),
        "score_column": score_col,
        "source_path": row.get("source_path"),
    }


def build_elo_report(
    *,
    choices_by_mode: dict[str, dict[str, dict[str, Any]]],
    validation_hashes: list[str],
    double_holdout_hashes: list[str],
    current_per_ds_path: Path,
    current_metrics_path: Path,
    model_compare: list[dict[str, Any]],
    blend_result: dict[str, Any],
    source_audit: dict[str, Any],
    jable_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metrics_rows = []
    per_dataset_by_mode = {}
    modes_report = {}
    for mode, choices in choices_by_mode.items():
        per_ds = choices_to_per_ds(choices)
        res = ladder_elo(per_ds, validation_hashes, "top1")
        res_dh = ladder_elo(per_ds, double_holdout_hashes, "top1") if double_holdout_hashes else None
        row = {
            "mode": mode,
            "elo": res["elo"],
            "field_rank": res["field_rank"],
            "field_size": res["field_size"],
            "scored": res["n_scored"],
            "battles_won": res["battles_won"],
            "top3_finishes": res["battles_top3"],
            "double_holdout_elo": None if res_dh is None else res_dh["elo"],
            "double_holdout_rank": None if res_dh is None else res_dh["field_rank"],
        }
        metrics_rows.append(row)
        modes_report[mode] = {**row, "top_incumbents": res.get("top_incumbents")}
        per_dataset_by_mode[mode] = res["per_dataset"]

    reproduction = reproduce_current_metrics(current_per_ds_path, current_metrics_path, validation_hashes, double_holdout_hashes)
    jable = load_jable_summary(jable_path)
    report = {
        "schema_version": "full_cv_meta_selector_v2",
        "validation_dataset_count": len(validation_hashes),
        "double_holdout_dataset_count": len(double_holdout_hashes),
        "modes": modes_report,
        "model_compare": model_compare,
        "blend": blend_result,
        "current_baseline_reproduction": reproduction,
        "jable_anchor": jable,
        "source_audit_summary": {
            "recoverable_train_test_labeled_rows": source_audit.get("recoverable_train_test_labeled_rows"),
            "recoverable_gap_vs_best_reconciled": source_audit.get("recoverable_gap_vs_best_reconciled"),
            "expected_full_cv_total": EXPECTED_FULL_CV_GEN1_30,
        },
        "notes": [
            "meta_selector_v2 is trained only on non-validation self_play_v33 train-run rows.",
            "oracle_full uses validation test labels to select and is a non-deployable upper bound.",
            "meta_selector_v1 is replayed from the published v1 validation choices artifact.",
        ],
    }
    return report, metrics_rows, per_dataset_by_mode


def choices_to_per_ds(choices: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for h, choice in choices.items():
        out[h] = [
            {
                "rank": 0,
                "engine": choice.get("engine"),
                "genome": choice.get("canonical_genome"),
                "oof_r2": choice.get("oof_r2"),
                "score_robust": choice.get("score_robust"),
                "test_rmse": choice.get("test_rmse"),
                "test_r2": choice.get("test_r2"),
                "valid": True,
            }
        ]
    return out


def reproduce_current_metrics(
    per_ds_path: Path,
    metrics_path: Path,
    validation_hashes: list[str],
    double_holdout_hashes: list[str],
) -> dict[str, Any]:
    if not per_ds_path.exists():
        return {"status": "missing_per_ds", "path": str(per_ds_path)}
    per_ds = json.loads(per_ds_path.read_text(encoding="utf-8"))
    published = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                published[row["mode"]] = row
    modes = {}
    for mode in ("top1", "cv_select", "best_k"):
        res = ladder_elo(per_ds, validation_hashes, mode)
        dh = ladder_elo(per_ds, double_holdout_hashes, mode) if double_holdout_hashes else None
        pub = published.get(mode, {})
        modes[mode] = {
            "computed_elo": res["elo"],
            "computed_rank": res["field_rank"],
            "computed_scored": res["n_scored"],
            "computed_double_holdout_elo": None if dh is None else dh["elo"],
            "computed_double_holdout_rank": None if dh is None else dh["field_rank"],
            "published_elo": finite_float(pub.get("elo")),
            "published_rank": finite_float(pub.get("field_rank")),
            "published_double_holdout_elo": finite_float(pub.get("double_holdout_elo")),
            "published_double_holdout_rank": finite_float(pub.get("double_holdout_rank")),
        }
        modes[mode]["elo_matches_published"] = (
            modes[mode]["published_elo"] is not None
            and abs(float(modes[mode]["published_elo"]) - float(res["elo"])) <= 0.11
        )
    return {"status": "ok", "per_ds_path": str(per_ds_path), "metrics_path": str(metrics_path), "modes": modes}


def load_jable_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "path": str(path),
        "validation_elo": nested_get(data, ("jable_as_incumbent", "validation", "elo")),
        "double_holdout_elo": nested_get(data, ("jable_as_incumbent", "double_holdout", "elo")),
        "coverage": data.get("jable_report_coverage"),
    }


def build_hard_misses(val_df: pd.DataFrame, choices_by_mode: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    robust = choices_by_mode.get("robust_cv_full", {})
    v2 = choices_by_mode.get("meta_selector_v2", {})
    misses = []
    for h, pick in v2.items():
        robust_pick = robust.get(h)
        if robust_pick is None:
            continue
        regret = finite_float(pick.get("test_rmse"))
        robust_regret = finite_float(robust_pick.get("test_rmse"))
        if regret is None or robust_regret is None:
            continue
        if regret > robust_regret + 1e-12:
            misses.append(
                {
                    "dataset_hash": h,
                    "v2_test_rmse": regret,
                    "robust_test_rmse": robust_regret,
                    "delta_vs_robust": regret - robust_regret,
                    "v2_engine": pick.get("engine"),
                    "robust_engine": robust_pick.get("engine"),
                }
            )
    misses.sort(key=lambda row: row["delta_vs_robust"], reverse=True)
    return {"count": len(misses), "worst_50": misses[:50]}


def build_cv_failure_report(train_df: pd.DataFrame) -> dict[str, Any]:
    robust_picks = pick_rows(train_df.assign(_score=train_df["cv_score_robust"]), "_score")
    rows = []
    for h, row in robust_picks.items():
        rows.append(
            {
                "dataset_hash": h,
                "test_rmse": float(row["label_test_rmse"]),
                "regret": float(row["label_regret"]),
                "cv_score_robust": float(row["cv_score_robust"]),
                "cv_oof_r2": finite_float(row.get("cv_oof_r2")),
                "head_type": row.get("head_type"),
                "target_bin": finite_float(row.get("gstruct_redflag_target_bin")),
                "repeat_rank": finite_float(row.get("gstruct_redflag_repeated_rank")),
                "feature_explosion": finite_float(row.get("gstruct_est_feature_explosion")),
                "ds_train_rows": finite_float(row.get("ds_train_rows")),
            }
        )
    rows.sort(key=lambda r: r["regret"], reverse=True)
    return {
        "schema_version": "full_cv_meta_selector_v2.cv_failure_report",
        "dataset_count": len(rows),
        "worst_100_robust_failures": rows[:100],
        "median_robust_regret": float(np.median([r["regret"] for r in rows])) if rows else None,
    }


def build_oracle_gap_report(per_dataset_by_mode: dict[str, Any]) -> dict[str, Any]:
    oracle = per_dataset_metric_map(per_dataset_by_mode.get("oracle_full"))
    robust = per_dataset_metric_map(per_dataset_by_mode.get("robust_cv_full"))
    v2 = per_dataset_metric_map(per_dataset_by_mode.get("meta_selector_v2"))
    gaps = []
    for h, oracle_row in oracle.items():
        oracle_rmse = finite_float(oracle_row.get("rmse", oracle_row.get("test_rmse")))
        robust_row = robust.get(h)
        v2_row = v2.get(h)
        robust_rmse = finite_float(robust_row.get("rmse", robust_row.get("test_rmse"))) if robust_row else None
        v2_rmse = finite_float(v2_row.get("rmse", v2_row.get("test_rmse"))) if v2_row else None
        if oracle_rmse is None or robust_rmse is None or v2_rmse is None:
            continue
        gaps.append(
            {
                "dataset_hash": h,
                "oracle_rmse": oracle_rmse,
                "robust_gap": robust_rmse - oracle_rmse,
                "v2_gap": v2_rmse - oracle_rmse,
                "v2_minus_robust_gap": (v2_rmse - oracle_rmse) - (robust_rmse - oracle_rmse),
            }
        )
    gaps.sort(key=lambda row: row["robust_gap"], reverse=True)
    return {
        "schema_version": "full_cv_meta_selector_v2.oracle_gap_report",
        "dataset_count": len(gaps),
        "median_robust_gap": float(np.median([r["robust_gap"] for r in gaps])) if gaps else None,
        "median_v2_gap": float(np.median([r["v2_gap"] for r in gaps])) if gaps else None,
        "worst_100_robust_gaps": gaps[:100],
    }


def per_dataset_metric_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(k): dict(v[0] if isinstance(v, list) and v else v) for k, v in value.items()}
    if isinstance(value, list):
        out = {}
        for row in value:
            if isinstance(row, dict) and row.get("hash"):
                out[str(row["hash"])] = dict(row)
        return out
    return {}


def pick_rows(df: pd.DataFrame, score_col: str) -> dict[str, pd.Series]:
    out = {}
    for h, group in df.groupby("dataset_hash", sort=False):
        finite = group[group[score_col].notna()]
        if finite.empty:
            continue
        out[str(h)] = finite.sort_values(score_col, ascending=False, kind="stable").iloc[0]
    return out


def sort_indices_by_group(indices: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(indices.tolist(), key=lambda i: (str(groups[i]), int(i))), dtype=np.int64)


def group_sizes_for_indices(indices: np.ndarray, groups: np.ndarray) -> list[int]:
    sizes: list[int] = []
    prev: str | None = None
    count = 0
    for idx in indices:
        group = str(groups[idx])
        if prev is None:
            prev = group
            count = 1
        elif group == prev:
            count += 1
        else:
            sizes.append(count)
            prev = group
            count = 1
    if prev is not None:
        sizes.append(count)
    return sizes


def write_feature_importance(path: Path, model: Any, feature_names: list[str]) -> None:
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    rows = []
    if values is not None:
        arr = np.asarray(values, dtype=float).reshape(-1)
        for name, value in sorted(zip(feature_names, arr, strict=False), key=lambda item: -abs(item[1]))[:500]:
            rows.append({"feature": name, "importance": float(value)})
    pd.DataFrame(rows, columns=["feature", "importance"]).to_csv(path, index=False)


def write_shap_stub(path: Path, feature_names: list[str]) -> None:
    pd.DataFrame([], columns=["feature", "shap_abs_mean"]).to_csv(path, index=False)


def write_choices(path: Path, choices: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for h in sorted(choices):
            fh.write(json.dumps(choices[h], sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, *, elo_report: dict[str, Any], blend_result: dict[str, Any], ablations: list[dict[str, Any]]) -> None:
    modes = elo_report.get("modes", {})
    v2 = modes.get("meta_selector_v2", {})
    robust = modes.get("robust_cv_full", {})
    oracle = modes.get("oracle_full", {})
    lines = [
        "# Full CV Meta Selector V2",
        "",
        "This artifact is an honest offline selector study: fitting uses only non-validation self-play rows with heldout test labels; validation test labels are used only for final reporting.",
        "",
        "## Headline Metrics",
        "",
        f"- `meta_selector_v2` validation Elo: `{v2.get('elo')}`",
        f"- `robust_cv_full` validation Elo: `{robust.get('elo')}`",
        f"- `oracle_full` validation Elo: `{oracle.get('elo')}`",
        f"- `meta_selector_v2` double-holdout Elo: `{v2.get('double_holdout_elo')}`",
        "",
        "## Protocol",
        "",
        "- Stage 1: LightGBM regressor and LambdaRank scorer over enriched deployable features.",
        "- Stage 2: Sparse pairwise anchor classifier trained on within-dataset hard comparisons.",
        "- Stage 3: LightGBM risk gate for CV-optimistic red flags.",
        "- Stage 4: Grid-searched blend over robust CV, stage-1 score, pairwise score, gate override, and rule penalties.",
        "",
        "## Blend Weights",
        "",
        f"```json\n{json.dumps(blend_result.get('weights', {}), indent=2, sort_keys=True)}\n```",
        "",
        "## Feature Ablations",
        "",
    ]
    for row in ablations:
        lines.append(
            f"- `{row['ablation']}`: regret@1=`{row.get('mean_regret_at_1')}`, top1=`{row.get('top1_oracle_capture')}`, top3=`{row.get('top3_oracle_capture')}`"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- External anchors are reported only when a compatible artifact already exists on disk.",
            "- `oracle_full` remains diagnostic only and is never used as a deployable selector.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def nested_get(row: Any, path: tuple[str, ...]) -> Any:
    cur = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_sub(a: Any, b: Any) -> float | None:
    fa = finite_float(a)
    fb = finite_float(b)
    if fa is None or fb is None:
        return None
    return fa - fb


def safe_div(a: Any, b: Any) -> float | None:
    fa = finite_float(a)
    fb = finite_float(b)
    if fa is None or fb is None or abs(fb) < 1e-12:
        return None
    return fa / fb


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else None


if __name__ == "__main__":
    raise SystemExit(main())
