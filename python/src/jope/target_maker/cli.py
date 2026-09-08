from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import TASK_BINARY, TASK_REGRESSION, default_cache_workers, default_roots, load_manifest, prepare_cache
from .encode import encode_file
from .train import TrainConfig, train
from .tui import watch


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="target-maker", description="Train and run a target-only encoder.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-cache", help="Cache final-column targets from complete quant datasets.")
    roots = default_roots()
    prep.add_argument("--cache-dir", type=_path, required=True)
    prep.add_argument("--binary-root", type=_path, default=roots[TASK_BINARY])
    prep.add_argument("--regression-root", type=_path, default=roots[TASK_REGRESSION])
    prep.add_argument("--seed", type=int, default=17)
    prep.add_argument("--limit-per-task", type=int, default=None)
    prep.add_argument("--overwrite", action="store_true")
    prep.add_argument("--workers", type=int, default=default_cache_workers())

    inspect = sub.add_parser("inspect-cache", help="Print manifest counts.")
    inspect.add_argument("--cache-dir", type=_path, required=True)

    tr = sub.add_parser("train", help="Train with optional torchrun/DDP.")
    tr.add_argument("--cache-dir", type=_path, required=True)
    tr.add_argument("--run-dir", type=_path, required=True)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--final-all-data-epochs", type=int, default=0)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--window-size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=3.0e-4)
    tr.add_argument("--weight-decay", type=float, default=1.0e-4)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--val-fraction", type=float, default=0.05)
    tr.add_argument("--steps-per-epoch", type=int, default=None)
    tr.add_argument("--length-multiplier", type=int, default=4)
    tr.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    tr.add_argument("--hidden-dim", type=int, default=128)
    tr.add_argument("--token-dim", type=int, default=128)
    tr.add_argument("--attention-layers", type=int, default=2)
    tr.add_argument("--attention-heads", type=int, default=4)
    tr.add_argument("--rich-stats", action="store_true")
    tr.add_argument("--temperature", type=float, default=0.10)
    tr.add_argument("--instance-loss-weight", type=float, default=1.0)
    tr.add_argument("--group-loss-weight", type=float, default=0.5)
    tr.add_argument("--profile-loss-weight", type=float, default=0.2)
    tr.add_argument("--task-loss-weight", type=float, default=0.05)
    tr.add_argument("--collapse-loss-weight", type=float, default=0.05)

    baseline = sub.add_parser("classical-baseline", help="Train non-neural summary-stat same-dataset pair baselines.")
    baseline.add_argument("--cache-dir", type=_path, required=True)
    baseline.add_argument("--run-dir", type=_path, required=True)
    baseline.add_argument("--window-size", type=int, default=96)
    baseline.add_argument("--views-per-record", type=int, default=4)
    baseline.add_argument("--limit-per-task", type=int, default=5000)
    baseline.add_argument("--train-pairs", type=int, default=50_000)
    baseline.add_argument("--val-pairs", type=int, default=10_000)
    baseline.add_argument("--test-pairs", type=int, default=10_000)
    baseline.add_argument("--retrieval-queries", type=int, default=1000)
    baseline.add_argument("--retrieval-candidates", type=int, default=256)
    baseline.add_argument("--seed", type=int, default=17)
    baseline.add_argument("--train-fraction", type=float, default=0.70)
    baseline.add_argument("--val-fraction", type=float, default=0.15)
    baseline.add_argument("--positive-fraction", type=float, default=0.50)
    baseline.add_argument("--same-task-negative-fraction", type=float, default=0.80)
    baseline.add_argument("--include-cache-profile", action="store_true")
    baseline.add_argument("--models", default="length_only,dummy,sgd_logistic,hist_gbdt,extra_trees,ensemble")
    baseline.add_argument("--n-jobs", type=int, default=-1)

    ev = sub.add_parser("eval-representations", help="Evaluate neural and classical representations on one shared split.")
    ev.add_argument("--cache-dir", type=_path, required=True)
    ev.add_argument("--run-dir", type=_path, required=True)
    ev.add_argument("--neural-checkpoint", "--checkpoint", dest="neural_checkpoints", type=_path, action="append", default=[])
    ev.add_argument("--classical-model-dir", type=_path, default=None)
    ev.add_argument("--classical-models", default="")
    ev.add_argument("--fused-bundle", dest="fused_bundles", type=_path, action="append", default=[])
    ev.add_argument("--window-size", type=int, default=96)
    ev.add_argument("--views-per-record", type=int, default=4)
    ev.add_argument("--limit-per-task", type=int, default=5000)
    ev.add_argument("--train-pairs", type=int, default=50_000)
    ev.add_argument("--val-pairs", type=int, default=10_000)
    ev.add_argument("--test-pairs", type=int, default=10_000)
    ev.add_argument("--retrieval-queries", type=int, default=1000)
    ev.add_argument("--retrieval-candidates", type=int, default=256)
    ev.add_argument("--seed", type=int, default=17)
    ev.add_argument("--train-fraction", type=float, default=0.70)
    ev.add_argument("--val-fraction", type=float, default=0.15)
    ev.add_argument("--positive-fraction", type=float, default=0.50)
    ev.add_argument("--same-task-negative-fraction", type=float, default=0.80)
    ev.add_argument("--include-cache-profile", action="store_true")
    ev.add_argument("--n-jobs", type=int, default=-1)
    ev.add_argument("--device", default=None)
    ev.add_argument("--batch-size", type=int, default=512)

    enc = sub.add_parser("encode", help="Encode a target CSV, dataset directory, or .npy file.")
    enc.add_argument("--checkpoint", type=_path, required=True)
    enc.add_argument("--input", type=_path, required=True)
    enc.add_argument("--task", choices=["binary", "regression"], required=True)
    enc.add_argument("--output", type=_path, default=None)
    enc.add_argument("--window-size", type=int, default=96)
    enc.add_argument("--stride", type=int, default=None)
    enc.add_argument("--max-windows", type=int, default=None)
    enc.add_argument("--seed", type=int, default=17)
    enc.add_argument("--device", default=None)

    fusion = sub.add_parser("train-fusion", help="Train and publish a fused target encoder/scorer bundle.")
    fusion.add_argument("--cache-dir", type=_path, required=True)
    fusion.add_argument("--run-dir", type=_path, default=None)
    fusion.add_argument("--best-checkpoint", type=_path, default=None)
    fusion.add_argument("--final-checkpoint", type=_path, default=None)
    fusion.add_argument("--window-size", type=int, default=96)
    fusion.add_argument("--views-per-record", type=int, default=4)
    fusion.add_argument("--signature-views", type=int, default=8)
    fusion.add_argument("--limit-per-task", type=int, default=None)
    fusion.add_argument("--train-pairs", type=int, default=250_000)
    fusion.add_argument("--val-pairs", type=int, default=50_000)
    fusion.add_argument("--test-pairs", type=int, default=50_000)
    fusion.add_argument("--retrieval-queries", type=int, default=5000)
    fusion.add_argument("--retrieval-candidates", type=int, default=512)
    fusion.add_argument("--seed", type=int, default=17)
    fusion.add_argument("--train-fraction", type=float, default=0.70)
    fusion.add_argument("--val-fraction", type=float, default=0.15)
    fusion.add_argument("--positive-fraction", type=float, default=0.50)
    fusion.add_argument("--same-task-negative-fraction", type=float, default=0.80)
    fusion.add_argument("--include-cache-profile", action="store_true")
    fusion.add_argument("--models", default="sgd_logistic,hist_gbdt,extra_trees,ensemble")
    fusion.add_argument("--vector-dim", type=int, default=512)
    fusion.add_argument("--classical-component-dim", type=int, default=384)
    fusion.add_argument("--n-jobs", type=int, default=-1)
    fusion.add_argument("--device", default=None)
    fusion.add_argument("--batch-size", type=int, default=512)

    fused_enc = sub.add_parser("encode-fused", help="Encode a target with a fused bundle.")
    fused_enc.add_argument("--bundle", type=_path, required=True)
    fused_enc.add_argument("--input", type=_path, required=True)
    fused_enc.add_argument("--task", choices=["binary", "regression"], required=True)
    fused_enc.add_argument("--output", type=_path, default=None)
    fused_enc.add_argument("--seed", type=int, default=17)
    fused_enc.add_argument("--device", default=None)

    fused_score = sub.add_parser("score-fused", help="Score whether two targets come from the same dataset with a fused bundle.")
    fused_score.add_argument("--bundle", type=_path, required=True)
    fused_score.add_argument("--left-input", type=_path, required=True)
    fused_score.add_argument("--left-task", choices=["binary", "regression"], required=True)
    fused_score.add_argument("--right-input", type=_path, required=True)
    fused_score.add_argument("--right-task", choices=["binary", "regression"], required=True)
    fused_score.add_argument("--seed", type=int, default=17)
    fused_score.add_argument("--device", default=None)

    tui = sub.add_parser("tui", help="Watch a training run with Rich.")
    tui.add_argument("--run-dir", type=_path, required=True)
    tui.add_argument("--refresh-seconds", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-cache":
        records = prepare_cache(
            args.cache_dir,
            {TASK_BINARY: args.binary_root, TASK_REGRESSION: args.regression_root},
            seed=args.seed,
            limit_per_task=args.limit_per_task,
            overwrite=args.overwrite,
            workers=args.workers,
        )
        counts = {"binary": 0, "regression": 0}
        for record in records:
            counts["binary" if record.task == TASK_BINARY else "regression"] += 1
        print(json.dumps({"cache_dir": str(args.cache_dir), "records": len(records), **counts}, sort_keys=True))
        return 0
    if args.command == "inspect-cache":
        records = load_manifest(args.cache_dir)
        counts = {"binary": 0, "regression": 0}
        hidden = 0
        for record in records:
            counts["binary" if record.task == TASK_BINARY else "regression"] += 1
            hidden += int(record.source_dir.name.startswith("."))
        print(json.dumps({"records": len(records), "hidden": hidden, **counts}, sort_keys=True))
        return 0
    if args.command == "train":
        checkpoint = train(
            TrainConfig(
                cache_dir=args.cache_dir,
                run_dir=args.run_dir,
                epochs=args.epochs,
                final_all_data_epochs=args.final_all_data_epochs,
                batch_size=args.batch_size,
                window_size=args.window_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                seed=args.seed,
                val_fraction=args.val_fraction,
                steps_per_epoch=args.steps_per_epoch,
                length_multiplier=args.length_multiplier,
                amp_dtype=args.amp_dtype,
                hidden_dim=args.hidden_dim,
                token_dim=args.token_dim,
                attention_layers=args.attention_layers,
                attention_heads=args.attention_heads,
                rich_stats=args.rich_stats,
                temperature=args.temperature,
                instance_loss_weight=args.instance_loss_weight,
                group_loss_weight=args.group_loss_weight,
                profile_loss_weight=args.profile_loss_weight,
                task_loss_weight=args.task_loss_weight,
                collapse_loss_weight=args.collapse_loss_weight,
            )
        )
        if checkpoint is not None:
            print(checkpoint)
        return 0
    if args.command == "classical-baseline":
        from .classical import ClassicalBaselineConfig, run_classical_baseline

        report = run_classical_baseline(
            ClassicalBaselineConfig(
                cache_dir=args.cache_dir,
                run_dir=args.run_dir,
                window_size=args.window_size,
                views_per_record=args.views_per_record,
                limit_per_task=args.limit_per_task,
                train_pairs=args.train_pairs,
                val_pairs=args.val_pairs,
                test_pairs=args.test_pairs,
                retrieval_queries=args.retrieval_queries,
                retrieval_candidates=args.retrieval_candidates,
                seed=args.seed,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                positive_fraction=args.positive_fraction,
                same_task_negative_fraction=args.same_task_negative_fraction,
                include_cache_profile=args.include_cache_profile,
                models=tuple(part.strip() for part in args.models.split(",") if part.strip()),
                n_jobs=args.n_jobs,
            )
        )
        models = report["models"]
        best_name = None
        best_auc = -1.0
        if isinstance(models, dict):
            for model_name, model_report in models.items():
                if not isinstance(model_report, dict):
                    continue
                test = model_report.get("test", {})
                auc = float(test.get("roc_auc", -1.0)) if isinstance(test, dict) else -1.0
                if auc > best_auc:
                    best_auc = auc
                    best_name = str(model_name)
        print(
            json.dumps(
                {
                    "report": str(args.run_dir / "classical_baseline_report.json"),
                    "window_feature_dim": report["window_feature_dim"],
                    "pair_feature_dim": report["pair_feature_dim"],
                    "best_model": best_name,
                    "best_test_roc_auc": best_auc,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "eval-representations":
        from .evaluation import EvalRepresentationsConfig, eval_representations

        report = eval_representations(
            EvalRepresentationsConfig(
                cache_dir=args.cache_dir,
                run_dir=args.run_dir,
                neural_checkpoints=tuple(args.neural_checkpoints),
                classical_model_dir=args.classical_model_dir,
                classical_models=tuple(part.strip() for part in args.classical_models.split(",") if part.strip()),
                fused_bundles=tuple(args.fused_bundles),
                window_size=args.window_size,
                views_per_record=args.views_per_record,
                limit_per_task=args.limit_per_task,
                train_pairs=args.train_pairs,
                val_pairs=args.val_pairs,
                test_pairs=args.test_pairs,
                retrieval_queries=args.retrieval_queries,
                retrieval_candidates=args.retrieval_candidates,
                seed=args.seed,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                positive_fraction=args.positive_fraction,
                same_task_negative_fraction=args.same_task_negative_fraction,
                include_cache_profile=args.include_cache_profile,
                n_jobs=args.n_jobs,
                device=args.device,
                batch_size=args.batch_size,
            )
        )
        print(
            json.dumps(
                {
                    "report": str(args.run_dir / "eval_representations_report.json"),
                    "neural": sorted(report["neural"]) if isinstance(report.get("neural"), dict) else [],
                    "classical_trained": sorted(report["classical_trained"])
                    if isinstance(report.get("classical_trained"), dict)
                    else [],
                    "classical_loaded": sorted(report["classical_loaded"])
                    if isinstance(report.get("classical_loaded"), dict)
                    else [],
                    "fused": sorted(report["fused"]) if isinstance(report.get("fused"), dict) else [],
                    "length_only_test_roc_auc": report["length_only"]["test"]["roc_auc"],
                    "retrieval_queries": report["retrieval_queries"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "encode":
        embedding = encode_file(
            args.checkpoint,
            args.input,
            args.task,
            output_path=args.output,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=args.max_windows,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps({"shape": list(embedding.shape), "dtype": str(embedding.dtype)}))
        return 0
    if args.command == "train-fusion":
        from .fused import DEFAULT_BEST_CHECKPOINT, DEFAULT_FINAL_CHECKPOINT, FusedTrainingConfig, default_fusion_run_dir, train_fusion

        run_dir = args.run_dir or default_fusion_run_dir()
        report = train_fusion(
            FusedTrainingConfig(
                cache_dir=args.cache_dir,
                run_dir=run_dir,
                best_checkpoint=args.best_checkpoint or DEFAULT_BEST_CHECKPOINT,
                final_checkpoint=args.final_checkpoint or DEFAULT_FINAL_CHECKPOINT,
                window_size=args.window_size,
                views_per_record=args.views_per_record,
                signature_views=args.signature_views,
                limit_per_task=args.limit_per_task,
                train_pairs=args.train_pairs,
                val_pairs=args.val_pairs,
                test_pairs=args.test_pairs,
                retrieval_queries=args.retrieval_queries,
                retrieval_candidates=args.retrieval_candidates,
                seed=args.seed,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                positive_fraction=args.positive_fraction,
                same_task_negative_fraction=args.same_task_negative_fraction,
                include_cache_profile=args.include_cache_profile,
                models=tuple(part.strip() for part in args.models.split(",") if part.strip()),
                vector_dim=args.vector_dim,
                classical_component_dim=args.classical_component_dim,
                n_jobs=args.n_jobs,
                device=args.device,
                batch_size=args.batch_size,
            )
        )
        selection = report.get("best_report", {}).get("selection", {}) if isinstance(report.get("best_report"), dict) else {}
        print(
            json.dumps(
                {
                    "report": str(run_dir / "fused_training_report.json"),
                    "best_bundle": report["best_bundle"],
                    "final_bundle": report["final_bundle"],
                    "inference_bundle": report["inference_bundle"],
                    "fusion_wins": selection.get("fusion_wins") if isinstance(selection, dict) else None,
                    "selected_default": selection.get("selected_default") if isinstance(selection, dict) else None,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "encode-fused":
        from .fused import encode_fused_file

        embedding = encode_fused_file(
            args.bundle,
            args.input,
            args.task,
            output_path=args.output,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps({"shape": list(embedding.shape), "dtype": str(embedding.dtype)}))
        return 0
    if args.command == "score-fused":
        from .fused import score_fused_files

        score = score_fused_files(
            args.bundle,
            args.left_input,
            args.left_task,
            args.right_input,
            args.right_task,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps({"score": score}, sort_keys=True))
        return 0
    if args.command == "tui":
        watch(args.run_dir, refresh_seconds=args.refresh_seconds)
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
