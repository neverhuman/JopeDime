from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from .cost import estimate_costs
from .data.active import ACTIVE_TRAIN_SIDECAR, build_active_corpus, build_active_v2_corpus
from .data.build import build_corpus
from .eval_ledger import iter_eval_regression_with_ledger
from .evaluate_active import evaluate_active_v2
from .feedback import build_active_feedback_corpus, ingest_feedback
from .grammar import corrupt_genome, sample_genome
from .io import write_jsonl
from .model import JopeActiveModelV2
from .prime_config import load_config
from .propose import (
    embedding_from_vs,
    load_any_model,
    load_embedding,
    load_standardization,
    propose_active_v2,
    propose_one,
    target_summary_from_vs,
)
from .selector_calibration import (
    DEFAULT_EVALUATION_JSONL,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_PROPOSALS_JSONL,
    benchmark_selector,
    build_selector_calibration,
    write_selector_calibration,
)
from .selector_reranker import replay_selector_reranker, train_selector_reranker, write_selector_reranker
from .status import artifact_freshness
from .train import train_active_v2_from_jsonl, train_from_jsonl
from .tui import run_tui
from .validator import RustValidator


def _dime_loader_workers(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "loader workers must be auto or an integer from 0 to 8"
        ) from error
    if workers < 0 or workers > 8:
        raise argparse.ArgumentTypeError(
            "loader workers must be auto or an integer from 0 to 8"
        )
    return workers


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "ratchet":
        _exec_rust_jope(raw)
    if raw and raw[0] == "broad-shadow":
        from .broad_shadow.cli import main as broad_shadow_main

        return broad_shadow_main(raw[1:])

    parser = argparse.ArgumentParser(prog="jope")
    prime_config = load_config()
    active_v1_dir = prime_config.artifacts / "active_v1"
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("genomes", nargs="*")

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--n", type=int, default=1)
    p_sample.add_argument("--invalid", type=int, default=0)
    p_sample.add_argument("--seed", type=int, default=0)

    p_data = sub.add_parser("data")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_build = data_sub.add_parser("build")
    p_build.add_argument("--quant-root")
    p_build.add_argument("--jmath-jsonl", action="append", default=[])
    p_build.add_argument("--generated", type=int, default=0)
    p_build.add_argument("--invalid", type=int, default=0)
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--no-validate", action="store_true")
    p_build_active = data_sub.add_parser("build-active")
    p_build_active.add_argument("--quant-root", default=str(prime_config.regression_dataset_root))
    p_build_active.add_argument("--split-json")
    p_build_active.add_argument("--out")
    p_build_active.add_argument("--out-dir", default=str(active_v1_dir))
    p_build_active.add_argument("--limit-datasets", type=int)
    p_build_active.add_argument("--validation-fraction", type=float, default=0.1)
    p_build_active.add_argument("--seed", type=int, default=0)
    p_build_active.add_argument("--no-validate", action="store_true")
    p_build_active.add_argument("--target-encoder", choices=["none", "summary64", "neural64", "train_target_vs"], default="none")
    p_build_active.add_argument("--target-model")
    p_build_active_v2 = data_sub.add_parser("build-active-v2")
    p_build_active_v2.add_argument("--quant-root", default=str(prime_config.regression_dataset_root))
    p_build_active_v2.add_argument("--split-json")
    p_build_active_v2.add_argument("--out")
    p_build_active_v2.add_argument("--out-dir", required=True)
    p_build_active_v2.add_argument("--limit-datasets", type=int)
    p_build_active_v2.add_argument("--validation-fraction", type=float, default=0.1)
    p_build_active_v2.add_argument("--seed", type=int, default=0)
    p_build_active_v2.add_argument("--no-validate", action="store_true")
    p_build_active_v2.add_argument(
        "--target-encoder",
        choices=["none", "summary64", "neural64", "train_target_vs"],
        default="train_target_vs",
    )
    p_build_active_v2.add_argument("--target-model")

    p_build_active_v3 = data_sub.add_parser("build-active-v3")
    p_build_active_v3.add_argument("--quant-root", default=str(prime_config.regression_dataset_root))
    p_build_active_v3.add_argument("--split-json")
    p_build_active_v3.add_argument("--out")
    p_build_active_v3.add_argument("--out-dir", required=True)
    p_build_active_v3.add_argument("--limit-datasets", type=int)
    p_build_active_v3.add_argument("--validation-fraction", type=float, default=0.1)
    p_build_active_v3.add_argument("--seed", type=int, default=0)
    p_build_active_v3.add_argument("--engines", default="featmath,newmath")
    p_build_active_v3.add_argument("--seed-genomes-per-engine", type=int, default=24)
    p_build_active_v3.add_argument("--expr-fraction", type=float, default=0.2)
    p_build_active_v3.add_argument(
        "--max-seed-cost-class", choices=["cheap", "medium", "expensive", "extreme"], default="medium"
    )
    p_build_active_v3.add_argument("--skip-seed-eval", action="store_true")
    p_build_active_v3.add_argument("--ledger")
    p_build_active_v3.add_argument("--eval-workers", type=int)
    p_build_active_v3.add_argument("--folds", type=int, default=3)
    p_build_active_v3.add_argument(
        "--target-encoder",
        choices=["none", "summary64", "neural64", "train_target_vs"],
        default="train_target_vs",
    )
    p_cache_dime = data_sub.add_parser("cache-dime-embeddings")
    p_cache_dime.add_argument("--output-root", default=str(prime_config.cache))
    p_cache_dime.add_argument("--split-manifest")
    p_cache_dime.add_argument("--standardization")
    p_cache_dime.add_argument(
        "--tabarena-root",
        default="/home/ubuntu/tabArenaLite/data/tabarena-lite-v0.1/datasets/regression",
    )
    p_cache_dime.add_argument("--device", default="cuda")

    p_train = sub.add_parser("train")
    p_train.add_argument("--corpus", required=True)
    p_train.add_argument("--checkpoint")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--metrics-jsonl")

    p_train_active = sub.add_parser("train-active")
    p_train_active.add_argument("--corpus", default=str(active_v1_dir / "active_rows.jsonl"))
    p_train_active.add_argument(
        "--checkpoint", default=str(prime_config.checkpoints / "active_v1-smoke.pt")
    )
    p_train_active.add_argument("--epochs", type=int, default=1)
    p_train_active.add_argument("--batch-size", type=int, default=32)
    p_train_active.add_argument("--lr", type=float, default=1e-3)
    p_train_active.add_argument("--device", default="cpu")
    p_train_active.add_argument("--metrics-jsonl")
    p_train_active_v2 = sub.add_parser("train-active-v2")
    p_train_active_v2.add_argument("--corpus", required=True)
    p_train_active_v2.add_argument("--features")
    p_train_active_v2.add_argument("--checkpoint", required=True)
    p_train_active_v2.add_argument("--epochs", type=int, default=60)
    p_train_active_v2.add_argument("--batch-size", type=int, default=512)
    p_train_active_v2.add_argument("--lr", type=float, default=5e-4)
    p_train_active_v2.add_argument("--device", default="cpu")
    p_train_active_v2.add_argument("--metrics-jsonl")
    p_train_active_v2.add_argument("--seed", type=int, default=0)
    p_train_active_v2.add_argument("--hidden-dim", type=int, default=256)
    p_train_active_v2.add_argument("--dropout", type=float, default=0.1)
    p_train_active_v2.add_argument(
        "--loss-profile",
        choices=["two_phase", "quality", "balanced", "decoder_pretrain", "reward_focus", "legacy"],
        default="two_phase",
    )
    p_train_active_v2.add_argument("--checkpoint-metric", default="val_pairwise_win_accuracy")

    p_train_policy = sub.add_parser("train-policy")
    p_train_policy.add_argument("--corpus", required=True)
    p_train_policy.add_argument("--checkpoint", required=True)
    p_train_policy.add_argument("--epochs", type=int, default=60)
    p_train_policy.add_argument("--batch-size", type=int, default=256)
    p_train_policy.add_argument("--lr", type=float, default=5e-4)
    p_train_policy.add_argument("--device", default="cpu")
    p_train_policy.add_argument("--metrics-jsonl")
    p_train_policy.add_argument("--seed", type=int, default=0)
    p_train_policy.add_argument("--phase-a-fraction", type=float, default=0.4)
    p_train_policy.add_argument("--checkpoint-metric", default="val_pairwise_win_accuracy")
    p_train_policy.add_argument("--allow-train-val-fallback", action="store_true")
    p_train_policy.add_argument("--greedy-eval-datasets", type=int, default=16)
    # Model scaling (bigger model = higher GPU utilization) + fast-pipeline knobs.
    p_train_policy.add_argument("--d-model", type=int, default=256)
    p_train_policy.add_argument("--n-layers", type=int, default=2)
    p_train_policy.add_argument("--ffn-dim", type=int, default=1024)
    p_train_policy.add_argument("--n-heads", type=int, default=4)
    p_train_policy.add_argument("--amp", action="store_true", help="mixed-precision autocast (CUDA)")
    p_train_policy.add_argument("--progress-jsonl", help="atomic per-batch progress file for the TUI")
    p_train_policy.add_argument("--extra-rows", nargs="*", default=[],
                                help="extra row files merged into the corpus (self-play generations)")
    p_train_policy.add_argument("--init-from", help="warm-start weights from a prior checkpoint (shape-filtered)")

    p_train_dime = sub.add_parser("train-dime-score")
    p_train_dime.add_argument("--corpus", required=True)
    p_train_dime.add_argument("--embeddings", required=True)
    p_train_dime.add_argument("--run-dir", required=True)
    p_train_dime.add_argument("--seed", type=int, default=17)
    p_train_dime.add_argument("--epochs", type=int, default=30)
    p_train_dime.add_argument("--patience", type=int, default=6)
    p_train_dime.add_argument("--lr", type=float, default=3e-4)
    p_train_dime.add_argument("--weight-decay", type=float, default=0.01)
    p_train_dime.add_argument("--datasets-per-batch", type=int, default=16)
    p_train_dime.add_argument("--genomes-per-dataset", type=int, default=16)
    p_train_dime.add_argument("--device", default="cuda")
    p_train_dime.add_argument("--resume", action="store_true")
    p_train_dime.add_argument("--warm-start")
    p_train_dime.add_argument("--genome-only", action="store_true")
    p_train_dime.add_argument("--heldout-diagnostic", action="store_true")
    p_train_dime.add_argument("--no-compile", action="store_true")
    p_train_dime.add_argument("--no-autotune", action="store_true")
    p_train_dime.add_argument(
        "--loader-workers",
        type=_dime_loader_workers,
        default="auto",
    )
    p_train_dime.add_argument("--prefetch-factor", type=int, default=4)
    p_train_dime.add_argument("--live-hz", type=float, default=10.0)
    p_train_dime.add_argument("--hardware-hz", type=float, default=1.0)
    p_train_dime.add_argument("--pilot-suite", action="store_true")
    p_train_dime.add_argument("--pilot-epochs", type=int, default=5)

    p_tui_dime = sub.add_parser("tui-dime-score")
    p_tui_dime.add_argument("--run-dir", required=True)
    p_tui_dime.add_argument("--refresh", type=float, default=0.1)
    p_tui_dime.add_argument("--once", action="store_true")

    p_propose = sub.add_parser("propose")
    p_propose.add_argument("--embedding", required=True)
    p_propose.add_argument("--checkpoint")
    p_propose.add_argument("--n", type=int, default=1)
    p_propose.add_argument("--seed", type=int, default=0)

    p_propose_dataset = sub.add_parser("propose-dataset")
    p_propose_dataset.add_argument("--dataset-dir")
    p_propose_dataset.add_argument("--vs")
    p_propose_dataset.add_argument(
        "--standardization", default=str(active_v1_dir / "standardization.json")
    )
    p_propose_dataset.add_argument("--checkpoint")
    p_propose_dataset.add_argument("--n", type=int, default=1)
    p_propose_dataset.add_argument("--seed", type=int, default=0)
    p_propose_dataset.add_argument("--temperature", type=float, default=0.0)
    p_propose_dataset.add_argument("--no-beam", action="store_true")
    p_propose_dataset.add_argument("--selector-calibration")

    p_eval = sub.add_parser("eval-regression")
    p_eval.add_argument("--dataset-hash")
    p_eval.add_argument("--train-csv")
    p_eval.add_argument("--test-csv")
    p_eval.add_argument("--genome")
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--ledger", default=str(prime_config.cache / "eval_ledger.jsonl"))
    p_eval.add_argument("--no-ledger", action="store_true")

    p_eval_active = sub.add_parser("evaluate-active")
    p_eval_active.add_argument("--art-dir", required=True)
    p_eval_active.add_argument("--checkpoint", required=True)
    p_eval_active.add_argument("--split", default="validation")
    p_eval_active.add_argument("--datasets", default="all")
    p_eval_active.add_argument("--proposals-per-dataset", type=int, default=3)
    p_eval_active.add_argument("--ledger", required=True)
    p_eval_active.add_argument("--out-dir", required=True)
    p_eval_active.add_argument("--folds", type=int, default=3)
    p_eval_active.add_argument("--seed", type=int, default=0)
    p_eval_active.add_argument("--temperature", type=float, default=0.0)
    p_eval_active.add_argument("--eval-workers", type=int)
    p_eval_active.add_argument("--engines", default="featmath")
    p_eval_active.add_argument("--selector-calibration")

    p_tui = sub.add_parser("tui")
    p_tui.add_argument("--run-dir", required=True)
    p_tui.add_argument("--art-dir")
    p_tui.add_argument("--refresh", type=float, default=1.0)
    p_tui.add_argument("--once", action="store_true")
    p_tui.add_argument("--no-gpu", action="store_true")
    p_tui.add_argument(
        "--screen",
        choices=["auto", "dashboard", "stream"],
        default="auto",
        help="auto = stream if a gene feed exists, else the classic dashboard",
    )
    p_tui.add_argument("--train-metrics", help="path to a training metrics.jsonl for the live loss panel")
    p_tui.add_argument(
        "--cluster-hosts",
        default="local,xbabe1,xbabe3",
        help="comma-separated hosts for the live GPU/core panel ('none' to disable)",
    )

    p_status = sub.add_parser("status")
    p_status.add_argument("--art-dir")
    p_status.add_argument("--run-dir")

    p_selector = sub.add_parser("selector")
    selector_sub = p_selector.add_subparsers(dest="selector_command", required=True)
    p_selector_calibrate = selector_sub.add_parser("calibrate")
    p_selector_calibrate.add_argument("--evaluation-jsonl", default=str(DEFAULT_EVALUATION_JSONL))
    p_selector_calibrate.add_argument("--proposals-jsonl", default=str(DEFAULT_PROPOSALS_JSONL))
    p_selector_calibrate.add_argument("--extra-training-jsonl", action="append")
    p_selector_calibrate.add_argument("--out", default=str(DEFAULT_OUTPUT_JSON))
    p_selector_calibrate.add_argument("--split", default="train")
    p_selector_calibrate.add_argument("--folds", type=int, default=3)
    p_selector_calibrate.add_argument("--bin-count", type=int, default=5)
    p_selector_calibrate.add_argument("--min-head-rows", type=int, default=12)
    p_selector_calibrate.add_argument("--min-pattern-rows", type=int, default=8)
    p_selector_benchmark = selector_sub.add_parser("benchmark")
    p_selector_benchmark.add_argument("--evaluation-jsonl", default=str(DEFAULT_EVALUATION_JSONL))
    p_selector_benchmark.add_argument("--proposals-jsonl", default=str(DEFAULT_PROPOSALS_JSONL))
    p_selector_benchmark.add_argument("--extra-training-jsonl", action="append")
    p_selector_benchmark.add_argument("--calibration")
    p_selector_benchmark.add_argument("--split", default="train")
    p_selector_benchmark.add_argument("--folds", type=int, default=3)
    p_selector_train_reranker = selector_sub.add_parser("train-reranker")
    p_selector_train_reranker.add_argument("--candidate-pool-jsonl", action="append", required=True)
    p_selector_train_reranker.add_argument("--out", required=True)
    p_selector_train_reranker.add_argument("--l2", type=float, default=1e-3)
    p_selector_replay_reranker = selector_sub.add_parser("replay-reranker")
    p_selector_replay_reranker.add_argument("--candidate-pool-jsonl", action="append", required=True)
    p_selector_replay_reranker.add_argument("--reranker", required=True)
    p_selector_replay_reranker.add_argument("--k", type=int, default=6)

    p_cost = sub.add_parser("cost")
    cost_sub = p_cost.add_subparsers(dest="cost_command", required=True)
    p_estimate = cost_sub.add_parser("estimate")
    p_estimate.add_argument("--quant-root")
    p_estimate.add_argument("--gpu-hourly-usd", type=float)
    p_estimate.add_argument("--cpu-hourly-usd", type=float)

    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--art-dir", required=True)
    p_loop.add_argument("--checkpoint", required=True)
    p_loop.add_argument("--run-dir", required=True)
    p_loop.add_argument("--ledger", required=True)
    p_loop.add_argument("--engines", default="featmath,newmath")
    p_loop.add_argument("--datasets-per-iter", type=int, default=32)
    p_loop.add_argument("--samples-per-dataset-engine", type=int, default=8)
    p_loop.add_argument("--cv-top-m", type=int, default=2)
    p_loop.add_argument("--cv-uncertain-m", type=int, default=1)
    p_loop.add_argument("--max-cv-per-engine", type=int, default=64)
    p_loop.add_argument("--folds", type=int, default=3)
    p_loop.add_argument("--refit-reward-every", type=int, default=1)
    p_loop.add_argument("--refit-policy-every", type=int, default=0)
    p_loop.add_argument("--max-iterations", type=int)
    p_loop.add_argument("--max-hours", type=float)
    p_loop.add_argument("--eval-workers", type=int)
    p_loop.add_argument("--seed", type=int, default=0)
    p_loop.add_argument("--device", default="cpu")
    p_loop.add_argument("--gene-feed", help="write a live gene_events.jsonl feed for the TUI")

    p_fused = sub.add_parser("data-build-fused", help="build a unified regression+binary corpus")
    p_fused.add_argument("--base-corpus-dir", required=True, help="existing regression v3 corpus dir")
    p_fused.add_argument("--binary-root", default=str(prime_config.binary_dataset_root))
    p_fused.add_argument("--out-dir", required=True)
    p_fused.add_argument("--max-binary-datasets", type=int, default=200)
    p_fused.add_argument("--seed-genomes-per-engine", type=int, default=8)
    p_fused.add_argument("--folds", type=int, default=3)
    p_fused.add_argument("--ledger")
    p_fused.add_argument("--eval-workers", type=int)
    p_fused.add_argument("--seed", type=int, default=0)

    p_stream = sub.add_parser("stream", help="continuous gene generation with a live feed")
    p_stream.add_argument("--art-dir", required=True)
    p_stream.add_argument("--checkpoint", required=True)
    p_stream.add_argument("--run-dir", required=True)
    p_stream.add_argument("--ledger", required=True)
    p_stream.add_argument("--engines", default="featmath,newmath")
    p_stream.add_argument("--sample-batch", type=int, default=48)
    p_stream.add_argument("--datasets-per-cycle", type=int, default=4)
    p_stream.add_argument("--cv-top-m", type=int, default=3)
    p_stream.add_argument("--cv-uncertain-m", type=int, default=1)
    p_stream.add_argument("--folds", type=int, default=3)
    p_stream.add_argument("--temperature", type=float, default=0.9)
    p_stream.add_argument("--eval-workers", type=int)
    p_stream.add_argument("--refit-interval-s", type=float, default=30.0)
    p_stream.add_argument("--refit-min-new-rows", type=int, default=64)
    p_stream.add_argument("--max-genes", type=int)
    p_stream.add_argument("--max-seconds", type=float)
    p_stream.add_argument("--seed", type=int, default=0)
    p_stream.add_argument("--device", default="cpu")

    p_sig = sub.add_parser("signatures", help="generate/verify Hyperion .vs sidecars for datasets")
    sig_sub = p_sig.add_subparsers(dest="signatures_command", required=True)
    p_sig_make = sig_sub.add_parser("make")
    p_sig_make.add_argument("--dataset-dir", help="single dataset dir")
    p_sig_make.add_argument("--quant-root", help="root of dataset dirs (batch)")
    p_sig_make.add_argument("--limit", type=int)
    p_sig_make.add_argument("--force", action="store_true")
    p_sig_make.add_argument("--host-index", type=int, default=0)
    p_sig_make.add_argument("--host-count", type=int, default=1)
    p_sig_make.add_argument("--report-jsonl")
    p_sig_verify = sig_sub.add_parser("verify")
    p_sig_verify.add_argument("--quant-root", required=True)
    p_sig_verify.add_argument("--deep", action="store_true")
    p_sig_verify.add_argument("--limit", type=int)

    p_elo = sub.add_parser("elo-bench", help="place the policy on the 97-algorithm regression Elo ladder")
    p_elo.add_argument("--checkpoint", required=True)
    p_elo.add_argument("--art-dir", required=True, help="processed corpus dir (standardization + split_manifest)")
    p_elo.add_argument("--k", type=int, default=16, help="candidates proposed per dataset")
    p_elo.add_argument("--folds", type=int, default=3)
    p_elo.add_argument("--seed", type=int, default=17)
    p_elo.add_argument("--temperature", type=float, default=0.7)
    p_elo.add_argument("--limit", type=int, help="cap dataset count (smoke)")
    p_elo.add_argument("--subset", choices=("validation", "double_holdout"), default="validation")
    p_elo.add_argument("--out-dir", default=str(prime_config.artifacts / "elo_bench"))
    p_elo.add_argument("--ledger")
    p_elo.add_argument("--workers", type=int, default=90)
    p_elo.add_argument("--selector-calibration")
    p_elo.add_argument("--selector-reranker")
    p_elo.add_argument("--samples-per-engine", type=int)
    p_elo.add_argument("--candidate-pool-dir")
    p_elo.add_argument("--proposal-failure-threshold", type=float, default=0.01)

    p_blast = sub.add_parser("blast", help="firehose TUI: genes→code on N Rust cores, tokens/s metered")
    p_blast.add_argument("--follow", help="gene_events.jsonl to tail (default: Prime self-play run)")
    p_blast.add_argument("--replay", action="store_true", help="cycle the corpus genome pool at --rate")
    p_blast.add_argument("--rate", type=int, default=94396)
    p_blast.add_argument("--workers", type=int, default=48)
    p_blast.add_argument("--fps", type=int, default=100)
    p_blast.add_argument("--headless", action="store_true")
    p_blast.add_argument("--seconds", type=int, default=10)

    p_sp = sub.add_parser("self-play", help="self-play generation loop: each model must beat all former selves")
    p_sp.add_argument("--art-dir", required=True)
    p_sp.add_argument("--run-dir", default=str(prime_config.runs / "self_play"))
    p_sp.add_argument("--generations", type=int, default=4)
    p_sp.add_argument("--epochs", type=int, default=14)
    p_sp.add_argument("--batch-size", type=int, default=512)
    p_sp.add_argument("--d-model", type=int, default=256)
    p_sp.add_argument("--n-layers", type=int, default=2)
    p_sp.add_argument("--ffn-dim", type=int, default=1024)
    p_sp.add_argument("--explore", type=int, default=3, help="exploration samples per dataset beyond top-1")
    p_sp.add_argument("--dataset-limit", type=int, help="cap train datasets (smoke)")
    p_sp.add_argument("--eval-workers", type=int, default=90)
    p_sp.add_argument("--single-node", action="store_true", help="train on the local GPU only (no xbabe3)")
    p_sp.add_argument("--train-hosts", help="comma list, e.g. local,xbabe3 (overrides --single-node)")
    p_sp.add_argument("--miner-slice", type=int, default=400, help="datasets champion-mined per generation (0=off)")
    p_sp.add_argument("--elo-every", type=int, default=1, help="run elo-bench every N generations")
    p_sp.add_argument("--lr", type=float, default=5e-4, help="LR for fresh-init generations")
    p_sp.add_argument("--warm-lr", type=float, default=1.5e-4,
                      help="refinement LR for warm-started generations (0 disables)")
    p_sp.add_argument("--selector-calibration")
    p_sp.add_argument("--selector-reranker")
    p_sp.add_argument("--samples-per-engine", type=int)
    p_sp.add_argument("--candidate-pool-dir")
    p_sp.add_argument("--proposal-failure-threshold", type=float, default=0.01)

    p_gpf = sub.add_parser("gp-factory", help="run the real jain GP per dataset on xbabe1, harvest champions")
    p_gpf.add_argument("--art-dir", required=True)
    p_gpf.add_argument("--run-dir", default=str(prime_config.runs / "gp_factory"))
    p_gpf.add_argument("--passes", type=int, default=1)
    p_gpf.add_argument("--max-batches", type=int)
    p_gpf.add_argument("--slice-size", type=int, default=48)
    p_gpf.add_argument("--workers", type=int, default=4)
    p_gpf.add_argument("--threads-per-worker", type=int, default=24)
    p_gpf.add_argument("--gens", type=int, default=5000)
    p_gpf.add_argument("--pop", type=int, default=128)
    p_gpf.add_argument("--time-budget-secs", type=int, default=240)
    p_gpf.add_argument("--gpu-kernel-configs", type=int, default=0)
    p_gpf.add_argument("--game", action="store_true",
                       help="game mode: iterate-to-beat max(champion bar, model bar) per dataset")
    p_gpf.add_argument("--game-rounds", type=int, default=3)
    p_gpf.add_argument("--fitness", default="v2", choices=("v1", "v2"),
                       help="jain fitness variant; v2 lite-race is default (better proxy<->realized correlation)")
    p_gpf.add_argument("--fresh-proposals",
                       help="policy ckpt for fresh CPU featmath proposals in game mode (slow)")
    p_gpf.add_argument("--candidate-pools-glob",
                       help="override the candidate_pools_g*.jsonl glob for model bars/seeds")

    p_study_gp = sub.add_parser(
        "study-validation-gp",
        help="run the 9-dataset validation-only Jain GP study",
    )
    p_study_gp.add_argument("--dataset", action="append", dest="datasets",
                            help="dataset hash to run; repeatable (default: fixed 9)")
    p_study_gp.add_argument("--art-dir", default=str(prime_config.artifacts))
    p_study_gp.add_argument("--root", default=str(prime_config.regression_dataset_root))
    p_study_gp.add_argument("--checkpoint", default=str(prime_config.warm_start_checkpoint))
    p_study_gp.add_argument("--jain-bin", default=str(prime_config.jain_binary))
    p_study_gp.add_argument(
        "--run-dir", default=str(prime_config.runs / "validation_gp_study")
    )
    p_study_gp.add_argument("--study-tag")
    p_study_gp.add_argument("--hosts", default="xbabe1,xbabe2,xbabe3")
    p_study_gp.add_argument("--workers-per-host", type=int, default=3)
    p_study_gp.add_argument("--threads-per-worker", type=int, default=24)
    p_study_gp.add_argument("--warm-secs", type=int, default=60)
    p_study_gp.add_argument("--main-secs", type=int, default=300)
    p_study_gp.add_argument("--pop", type=int, default=128)
    p_study_gp.add_argument("--gens", type=int, default=5000)
    p_study_gp.add_argument("--top-k", type=int, default=10)
    p_study_gp.add_argument("--search-folds", type=int, default=5)
    p_study_gp.add_argument("--search-rows", type=int, default=768)
    p_study_gp.add_argument("--robust-folds", type=int, default=8)
    p_study_gp.add_argument("--robust-workers", type=int, default=8)
    p_study_gp.add_argument("--seed", type=int, default=20260710)
    p_study_gp.add_argument("--dry-run", action="store_true")

    p_mine = sub.add_parser("mine-champions", help="CPU mutation-race against per-dataset champions")
    p_mine.add_argument("--art-dir", required=True)
    p_mine.add_argument("--generation", type=int, required=True)
    p_mine.add_argument("--slice-file", required=True, help="JSON list of dataset hashes to mine")
    p_mine.add_argument("--out", required=True)
    p_mine.add_argument("--workers", type=int, default=90)
    p_mine.add_argument(
        "--ledger", default=str(prime_config.runs / "self_play" / "miner_ledger.jsonl")
    )
    p_mine.add_argument("--per-dataset", type=int, default=14)
    p_mine.add_argument("--seed", type=int, default=0)
    p_mine.add_argument("--progress-file", help="atomic miner progress json for the TUI")
    p_mine.add_argument("--gene-feed", help="shared gene_events.jsonl to emit scored/champion events")

    p_prep = sub.add_parser("prep-v31-labels", help="stamp rank_pct/task_type/global-RWR onto an active corpus")
    p_prep.add_argument("--corpus", required=True)
    p_prep.add_argument("--dry-run", action="store_true")

    p_migrate = sub.add_parser("migrate-vocab-v4", help="migrate a frozen active_vocab.json to schema v4")
    p_migrate.add_argument("--vocab", required=True)
    p_migrate.add_argument("--dry-run", action="store_true")

    p_gp = sub.add_parser("gp", help="seed jope genes into the jmath GP engine / harvest winners")
    gp_sub = p_gp.add_subparsers(dest="gp_command", required=True)
    p_gp_export = gp_sub.add_parser("export")
    p_gp_export.add_argument("--source", action="append", required=True, help="jope JSONL (loop_rows/evaluation); repeatable")
    p_gp_export.add_argument("--out-dir", required=True)
    p_gp_export.add_argument("--top", type=int, default=200)
    p_gp_export.add_argument("--min-oof", type=float, default=0.0)
    p_gp_export.add_argument("--engine", default="newmath")
    p_gp_export.add_argument("--task-family", default="regression")
    p_gp_export.add_argument("--state-dir", help="GP state dir (READ-ONLY, for novelty dedupe)")
    p_gp_harvest = gp_sub.add_parser("harvest")
    p_gp_harvest.add_argument("--winners-dir", default=str(prime_config.newmath_winners_dir))
    p_gp_harvest.add_argument("--out", required=True)

    p_ledger = sub.add_parser("ledger")
    ledger_sub = p_ledger.add_subparsers(dest="ledger_command", required=True)
    p_ledger_migrate = ledger_sub.add_parser("migrate")
    p_ledger_migrate.add_argument("--ledger", required=True)

    p_feedback = sub.add_parser("feedback")
    feedback_sub = p_feedback.add_subparsers(dest="feedback_command", required=True)
    p_ingest = feedback_sub.add_parser("ingest")
    p_ingest.add_argument("--report", required=True)
    p_ingest.add_argument("--ledger", default=str(prime_config.runs / "feedback.jsonl"))
    p_feedback_active = feedback_sub.add_parser("build-active")
    p_feedback_active.add_argument("--art-dir", required=True)
    p_feedback_active.add_argument("--evaluation-jsonl", required=True)
    p_feedback_active.add_argument("--out", required=True)
    p_feedback_active.add_argument("--base-corpus")
    p_feedback_active.add_argument("--allow-non-train", action="store_true")

    args = parser.parse_args(raw)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "sample":
        return _cmd_sample(args)
    if args.command == "data" and args.data_command == "build":
        rows = build_corpus(
            quant_root=args.quant_root,
            jmath_jsonl=args.jmath_jsonl,
            generated=args.generated,
            invalid=args.invalid,
            seed=args.seed,
            validate=not args.no_validate,
        )
        count = write_jsonl(args.out, rows)
        print(json.dumps({"rows": count, "out": args.out}, sort_keys=True))
        return 0
    if args.command == "data" and args.data_command == "build-active":
        build = build_active_corpus(
            quant_root=args.quant_root,
            split_json=args.split_json,
            out_dir=args.out_dir,
            out_jsonl=args.out,
            limit_datasets=args.limit_datasets,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            validate=not args.no_validate,
            target_encoder=args.target_encoder,
            target_model=args.target_model,
        )
        print(
            json.dumps(
                {
                    "rows": len(build.rows),
                    "out": args.out,
                    "out_dir": args.out_dir,
                    "manifest": build.manifest,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "data" and args.data_command == "build-active-v2":
        build = build_active_v2_corpus(
            quant_root=args.quant_root,
            split_json=args.split_json,
            out_dir=args.out_dir,
            out_jsonl=args.out,
            limit_datasets=args.limit_datasets,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            validate=not args.no_validate,
            target_encoder=args.target_encoder,
            target_model=args.target_model,
        )
        print(
            json.dumps(
                {
                    "rows": len(build.rows),
                    "out": args.out,
                    "out_dir": args.out_dir,
                    "manifest": build.manifest,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "data" and args.data_command == "build-active-v3":
        from .data.active_v3 import build_active_v3_corpus

        build = build_active_v3_corpus(
            quant_root=args.quant_root,
            split_json=args.split_json,
            out_dir=args.out_dir,
            out_jsonl=args.out,
            limit_datasets=args.limit_datasets,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            engines=tuple(e.strip() for e in args.engines.split(",") if e.strip()),
            seed_genomes_per_engine=args.seed_genomes_per_engine,
            expr_fraction=args.expr_fraction,
            max_seed_cost_class=args.max_seed_cost_class,
            skip_seed_eval=args.skip_seed_eval,
            ledger=args.ledger,
            eval_workers=args.eval_workers,
            folds=args.folds,
            target_encoder=args.target_encoder,
        )
        print(
            json.dumps(
                {
                    "rows": len(build.rows),
                    "rows_per_engine": build.manifest.get("rows_per_engine"),
                    "out_dir": args.out_dir,
                    "seed_generation": build.manifest.get("seed_generation"),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "data" and args.data_command == "cache-dime-embeddings":
        from .dime_score.embeddings import cache_dime_embeddings

        print(
            json.dumps(
                cache_dime_embeddings(
                    output_root=args.output_root,
                    split_manifest=args.split_manifest,
                    standardization_path=args.standardization,
                    tabarena_root=args.tabarena_root,
                    device=args.device,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "train-dime-score":
        from .dime_score.train import (
            TrainConfig,
            run_initialization_pilots,
            train_dime_score,
        )

        config = TrainConfig(
            corpus_root=Path(args.corpus),
            embeddings_root=Path(args.embeddings),
            run_dir=Path(args.run_dir),
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            datasets_per_batch=args.datasets_per_batch,
            genomes_per_dataset=args.genomes_per_dataset,
            device=args.device,
            resume=args.resume,
            warm_start=Path(args.warm_start) if args.warm_start else None,
            genome_only=args.genome_only,
            heldout_diagnostic=args.heldout_diagnostic,
            compile_model=not args.no_compile,
            autotune=not args.no_autotune,
            loader_workers=args.loader_workers,
            prefetch_factor=args.prefetch_factor,
            live_hz=args.live_hz,
            hardware_hz=args.hardware_hz,
        )
        result = (
            run_initialization_pilots(config, pilot_epochs=args.pilot_epochs)
            if args.pilot_suite
            else train_dime_score(config)
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "tui-dime-score":
        from .dime_score.tui import run_dime_score_tui

        run_dime_score_tui(args.run_dir, refresh=args.refresh, once=args.once)
        return 0
    if args.command == "train-policy":
        from .engines import EngineRouter
        from .policy.train_v3 import train_policy_v3_from_jsonl

        print(
            json.dumps(
                train_policy_v3_from_jsonl(
                    args.corpus,
                    args.checkpoint,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    metrics_jsonl=args.metrics_jsonl,
                    seed=args.seed,
                    phase_a_fraction=args.phase_a_fraction,
                    checkpoint_metric=args.checkpoint_metric,
                    allow_train_val_fallback=args.allow_train_val_fallback,
                    greedy_eval_datasets=args.greedy_eval_datasets,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    ffn_dim=args.ffn_dim,
                    n_heads=args.n_heads,
                    amp=args.amp,
                    progress_jsonl=args.progress_jsonl,
                    extra_rows=args.extra_rows,
                    init_from=args.init_from,
                    router=EngineRouter(),
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "train":
        print(
            json.dumps(
                train_from_jsonl(
                    args.corpus,
                    args.checkpoint,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    metrics_jsonl=args.metrics_jsonl,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "train-active":
        print(
            json.dumps(
                train_from_jsonl(
                    args.corpus,
                    args.checkpoint,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    metrics_jsonl=args.metrics_jsonl,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "train-active-v2":
        print(
            json.dumps(
                train_active_v2_from_jsonl(
                    args.corpus,
                    features=args.features,
                    checkpoint=args.checkpoint,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    metrics_jsonl=args.metrics_jsonl,
                    seed=args.seed,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    loss_profile=args.loss_profile,
                    checkpoint_metric=args.checkpoint_metric,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "propose":
        embedding = load_embedding(args.embedding)
        model = load_any_model(args.checkpoint) if args.checkpoint else None
        if isinstance(model, JopeActiveModelV2):
            for row in propose_active_v2(embedding, model=model, n=args.n, seed=args.seed):
                print(json.dumps(row, sort_keys=True))
        else:
            for i in range(args.n):
                print(json.dumps(propose_one(embedding, model=model, seed=args.seed + i), sort_keys=True))
        return 0
    if args.command == "propose-dataset":
        vs_path = _resolve_vs_path(args)
        standardization = load_standardization(args.standardization) if args.standardization else None
        embedding = embedding_from_vs(vs_path, standardization)
        model = load_any_model(args.checkpoint) if args.checkpoint else None
        if isinstance(model, JopeActiveModelV2):
            target_summary = target_summary_from_vs(_resolve_target_vs_path(args, vs_path))
            for row in propose_active_v2(
                embedding,
                target_summary=target_summary,
                model=model,
                n=args.n,
                seed=args.seed,
                temperature=args.temperature,
                beam=not args.no_beam,
                selector_calibration=args.selector_calibration,
            ):
                row["vs_path"] = str(vs_path)
                print(json.dumps(row, sort_keys=True))
        else:
            for i in range(args.n):
                row = propose_one(embedding, model=model, seed=args.seed + i)
                row["vs_path"] = str(vs_path)
                print(json.dumps(row, sort_keys=True))
        return 0
    if args.command == "eval-regression":
        rows = _eval_rows_from_args_or_stdin(args)
        ledger = None if args.no_ledger else args.ledger
        for _, row in iter_eval_regression_with_ledger(rows, ledger=ledger, validator=RustValidator()):
            print(json.dumps(row, sort_keys=True))
        return 0
    if args.command == "evaluate-active":
        print(
            json.dumps(
                evaluate_active_v2(
                    art_dir=args.art_dir,
                    checkpoint=args.checkpoint,
                    split=args.split,
                    datasets=args.datasets,
                    proposals_per_dataset=args.proposals_per_dataset,
                    ledger=args.ledger,
                    out_dir=args.out_dir,
                    folds=args.folds,
                    seed=args.seed,
                    temperature=args.temperature,
                    eval_workers=args.eval_workers,
                    engines=tuple(e.strip() for e in args.engines.split(",") if e.strip()),
                    selector_calibration=args.selector_calibration,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "tui":
        feed = Path(args.run_dir) / "gene_events.jsonl.meta.json"
        use_stream = args.screen == "stream" or (args.screen == "auto" and feed.exists())
        if use_stream:
            from .tui_stream import run_stream_tui

            hosts = None
            if args.cluster_hosts and args.cluster_hosts.lower() != "none":
                hosts = tuple(h.strip() for h in args.cluster_hosts.split(",") if h.strip())
            # Stream screen defaults to 15fps (~66ms) so batch-level training progress lands
            # sub-100ms; the progress file is rewritten every ~50ms. User --refresh overrides.
            stream_refresh = args.refresh if args.refresh and args.refresh > 1.0 else 15.0
            run_stream_tui(
                run_dir=args.run_dir,
                refresh=stream_refresh,
                once=args.once,
                train_metrics=args.train_metrics,
                cluster_hosts=hosts,
            )
            return 0
        if not args.art_dir:
            parser.error("tui dashboard screen requires --art-dir")
        run_tui(
            run_dir=args.run_dir,
            art_dir=args.art_dir,
            refresh=args.refresh,
            once=args.once,
            no_gpu=args.no_gpu,
        )
        return 0
    if args.command == "status":
        print(json.dumps(artifact_freshness(art_dir=args.art_dir, run_dir=args.run_dir), sort_keys=True))
        return 0
    if args.command == "selector" and args.selector_command == "calibrate":
        calibration = build_selector_calibration(
            evaluation_jsonl=args.evaluation_jsonl,
            proposals_jsonl=args.proposals_jsonl,
            extra_training_jsonl=args.extra_training_jsonl,
            split=args.split,
            folds=args.folds,
            bin_count=args.bin_count,
            min_head_rows=args.min_head_rows,
            min_pattern_rows=args.min_pattern_rows,
        )
        write_selector_calibration(calibration, args.out)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "schema_version": calibration["schema_version"],
                    "selected_policy": calibration["selected_policy"],
                    "row_count": calibration["source"]["row_count"],
                    "dataset_count": calibration["source"]["dataset_count"],
                    "cv": calibration["cv"]["policies"].get(calibration["selected_policy"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "selector" and args.selector_command == "benchmark":
        print(
            json.dumps(
                benchmark_selector(
                    evaluation_jsonl=args.evaluation_jsonl,
                    proposals_jsonl=args.proposals_jsonl,
                    extra_training_jsonl=args.extra_training_jsonl,
                    calibration=args.calibration,
                    split=args.split,
                    folds=args.folds,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "selector" and args.selector_command == "train-reranker":
        artifact = train_selector_reranker(args.candidate_pool_jsonl, l2=args.l2)
        write_selector_reranker(artifact, args.out)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "schema_version": artifact["schema_version"],
                    "row_count": artifact["fit"]["row_count"],
                    "dataset_count": artifact["fit"]["dataset_count"],
                    "replay": artifact.get("replay"),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "selector" and args.selector_command == "replay-reranker":
        reranker = json.loads(Path(args.reranker).read_text(encoding="utf-8"))
        print(
            json.dumps(
                replay_selector_reranker(args.candidate_pool_jsonl, reranker, k=args.k),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "loop":
        from .loop import LoopConfig, run_loop

        result = run_loop(
            LoopConfig(
                art_dir=Path(args.art_dir),
                checkpoint=Path(args.checkpoint),
                run_dir=Path(args.run_dir),
                ledger=Path(args.ledger),
                engines=tuple(e.strip() for e in args.engines.split(",") if e.strip()),
                datasets_per_iter=args.datasets_per_iter,
                samples_per_dataset_engine=args.samples_per_dataset_engine,
                cv_top_m=args.cv_top_m,
                cv_uncertain_m=args.cv_uncertain_m,
                max_cv_per_engine=args.max_cv_per_engine,
                folds=args.folds,
                refit_reward_every=args.refit_reward_every,
                refit_policy_every=args.refit_policy_every,
                max_iterations=args.max_iterations,
                max_hours=args.max_hours,
                eval_workers=args.eval_workers,
                seed=args.seed,
                device=args.device,
                gene_feed_path=Path(args.gene_feed) if args.gene_feed else None,
            )
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "data-build-fused":
        from .data.active_fused import build_fused_corpus
        from .engines import EngineRouter

        result = build_fused_corpus(
            base_corpus_dir=args.base_corpus_dir,
            binary_root=args.binary_root,
            out_dir=args.out_dir,
            max_binary_datasets=args.max_binary_datasets,
            seed_genomes_per_engine=args.seed_genomes_per_engine,
            folds=args.folds,
            ledger=args.ledger,
            eval_workers=args.eval_workers,
            seed=args.seed,
            router=EngineRouter(),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "stream":
        from .stream import StreamConfig, run_stream

        result = run_stream(
            StreamConfig(
                art_dir=Path(args.art_dir),
                checkpoint=Path(args.checkpoint),
                run_dir=Path(args.run_dir),
                ledger=Path(args.ledger),
                engines=tuple(e.strip() for e in args.engines.split(",") if e.strip()),
                sample_batch=args.sample_batch,
                datasets_per_cycle=args.datasets_per_cycle,
                cv_top_m=args.cv_top_m,
                cv_uncertain_m=args.cv_uncertain_m,
                folds=args.folds,
                temperature=args.temperature,
                eval_workers=args.eval_workers,
                refit_interval_s=args.refit_interval_s,
                refit_min_new_rows=args.refit_min_new_rows,
                max_genes=args.max_genes,
                max_seconds=args.max_seconds,
                seed=args.seed,
                device=args.device,
            )
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "signatures" and args.signatures_command == "make":
        from .signatures import discover_datasets, make_signatures, make_signatures_many

        if args.dataset_dir:
            report = make_signatures(Path(args.dataset_dir), force=args.force)
            print(json.dumps(report.__dict__, sort_keys=True))
            return 0
        dirs = discover_datasets(Path(args.quant_root))
        if args.limit:
            dirs = dirs[: args.limit]
        counts: dict[str, int] = {}
        for report in make_signatures_many(
            dirs, host_index=args.host_index, host_count=args.host_count,
            force=args.force, report_jsonl=Path(args.report_jsonl) if args.report_jsonl else None,
        ):
            counts[report.status] = counts.get(report.status, 0) + 1
        print(json.dumps({"datasets": len(dirs), "by_status": counts}, sort_keys=True))
        return 0
    if args.command == "signatures" and args.signatures_command == "verify":
        from .signatures import discover_datasets, verify_sidecars

        dirs = discover_datasets(Path(args.quant_root))
        if args.limit:
            dirs = dirs[: args.limit]
        corrupt = missing = complete = 0
        bad: list[str] = []
        for d in dirs:
            v = verify_sidecars(d, deep=args.deep)
            if v["complete"]:
                complete += 1
            else:
                if v["corrupt"]:
                    corrupt += 1
                if v["missing"]:
                    missing += 1
                if len(bad) < 20:
                    bad.append(f"{d.name}: missing={v['missing']} corrupt={v['corrupt']}")
        print(json.dumps({"datasets": len(dirs), "complete": complete, "corrupt": corrupt,
                          "missing_only": missing - corrupt if missing > corrupt else missing,
                          "examples": bad}, sort_keys=True))
        return 0
    if args.command == "blast":
        import os as _os
        import subprocess as _sp

        root = Path(__file__).resolve().parents[2]
        binary = root / "rust/jope-blast/target/release/jope-blast"
        if not binary.exists():
            _sp.run(["cargo", "build", "--release", "--manifest-path",
                     str(root / "rust/jope-blast/Cargo.toml")], check=True)
        pool = prime_config.artifacts / "blast_pool.tsv"
        cmd = [str(binary), "--workers", str(args.workers), "--fps", str(args.fps),
               "--rate", str(args.rate)]
        if args.replay:
            if not pool.exists():
                raise SystemExit(f"pool missing: {pool} (build it from the corpus first)")
            cmd += ["--replay", str(pool)]
        else:
            cmd += [
                "--follow",
                args.follow or str(prime_config.runs / "self_play_v32/gene_events.jsonl"),
            ]
        if args.headless:
            cmd += ["--headless", "--seconds", str(args.seconds)]
        _os.execv(str(binary), cmd)

    if args.command == "self-play":
        from .self_play import SelfPlay

        if args.train_hosts:
            hosts = tuple(h.strip() for h in args.train_hosts.split(",") if h.strip())
        else:
            hosts = ("local",) if args.single_node else ("local", "xbabe3")
        SelfPlay(
            args.art_dir,
            args.run_dir,
            train_hosts=hosts,
            eval_workers=args.eval_workers,
            explore=args.explore,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dataset_limit=args.dataset_limit,
            miner_slice=args.miner_slice,
            elo_every=args.elo_every,
            lr=args.lr,
            warm_lr=args.warm_lr or None,
            selector_calibration=args.selector_calibration,
            selector_reranker=args.selector_reranker,
            samples_per_engine=args.samples_per_engine,
            candidate_pool_dir=args.candidate_pool_dir,
            proposal_failure_threshold=args.proposal_failure_threshold,
        ).run(args.generations)
        return 0
    if args.command == "gp-factory":
        from .gp_factory import GpFactory

        GpFactory(
            args.art_dir,
            args.run_dir,
            slice_size=args.slice_size,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            gens=args.gens,
            pop=args.pop,
            time_budget_secs=args.time_budget_secs,
            gpu_kernel_configs=args.gpu_kernel_configs,
            game_mode=args.game,
            game_rounds=args.game_rounds,
            fitness=args.fitness,
            fresh_proposals_ckpt=args.fresh_proposals,
            candidate_pools_glob=args.candidate_pools_glob,
        ).run(passes=args.passes, max_batches=args.max_batches)
        return 0
    if args.command == "study-validation-gp":
        from .validation_gp_study import DEFAULT_DATASETS, StudyConfig, ValidationGpStudy

        cfg = StudyConfig(
            datasets=args.datasets or list(DEFAULT_DATASETS),
            art_dir=args.art_dir,
            root=args.root,
            checkpoint=args.checkpoint,
            jain_bin=args.jain_bin,
            run_dir=args.run_dir,
            study_tag=args.study_tag or "",
            hosts=[h.strip() for h in args.hosts.split(",") if h.strip()],
            workers_per_host=args.workers_per_host,
            threads_per_worker=args.threads_per_worker,
            warm_secs=args.warm_secs,
            main_secs=args.main_secs,
            pop=args.pop,
            gens=args.gens,
            top_k=args.top_k,
            search_folds=args.search_folds,
            search_rows=args.search_rows,
            robust_folds=args.robust_folds,
            robust_workers=args.robust_workers,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        summary = ValidationGpStudy(cfg).run()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "mine-champions":
        from pathlib import Path as _P

        from .champion_miner import build_champions, mine_slice

        art = _P(args.art_dir)
        manifest = json.loads((art / "split_manifest.json").read_text())["datasets"]
        wanted = set(json.loads(_P(args.slice_file).read_text()))
        datasets = {h: e for h, e in manifest.items() if h in wanted}
        champions = build_champions(art)
        mine_slice(
            art, args.out, generation=args.generation, datasets=datasets,
            champions=champions, per_dataset=args.per_dataset, workers=args.workers,
            seed=args.seed, ledger=args.ledger,
            progress_path=args.progress_file, gene_feed_path=args.gene_feed,
        )
        return 0
    if args.command == "prep-v31-labels":
        from .data.prep_v31_labels import prep_v31_labels

        print(json.dumps(prep_v31_labels(args.corpus, write=not args.dry_run), indent=2))
        return 0
    if args.command == "migrate-vocab-v4":
        from .data.migrate_vocab_v4 import migrate_vocab_v4

        result = migrate_vocab_v4(args.vocab, write=not args.dry_run)
        print(json.dumps(result["report"], indent=2, sort_keys=True))
        return 0
    if args.command == "elo-bench":
        from .elo_bench import run_elo_bench

        run_elo_bench(
            args.checkpoint,
            args.art_dir,
            k=args.k,
            folds=args.folds,
            seed=args.seed,
            temperature=args.temperature,
            limit=args.limit,
            subset=args.subset,
            out_dir=args.out_dir,
            ledger=args.ledger,
            workers=args.workers,
            selector_calibration=args.selector_calibration,
            selector_reranker=args.selector_reranker,
            samples_per_engine=args.samples_per_engine,
            candidate_pool_dir=args.candidate_pool_dir,
            proposal_failure_threshold=args.proposal_failure_threshold,
        )
        return 0
    if args.command == "gp" and args.gp_command == "export":
        from .gp_interop import gp_export

        result = gp_export(
            sources=[Path(s) for s in args.source],
            out_dir=Path(args.out_dir),
            top=args.top,
            min_oof=args.min_oof,
            engine=args.engine,
            task_family=args.task_family,
            state_dir=Path(args.state_dir) if args.state_dir else None,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "gp" and args.gp_command == "harvest":
        from .gp_interop import gp_harvest

        print(json.dumps(gp_harvest(winners_dir=Path(args.winners_dir), out=Path(args.out)), sort_keys=True))
        return 0
    if args.command == "ledger" and args.ledger_command == "migrate":
        from .eval_ledger import migrate_ledger_v3

        print(json.dumps(migrate_ledger_v3(args.ledger), sort_keys=True))
        return 0
    if args.command == "cost" and args.cost_command == "estimate":
        print(
            json.dumps(
                estimate_costs(
                    quant_root=args.quant_root,
                    gpu_hourly_usd=args.gpu_hourly_usd,
                    cpu_hourly_usd=args.cpu_hourly_usd,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "feedback" and args.feedback_command == "ingest":
        print(json.dumps(ingest_feedback(args.report, args.ledger), sort_keys=True))
        return 0
    if args.command == "feedback" and args.feedback_command == "build-active":
        print(
            json.dumps(
                build_active_feedback_corpus(
                    art_dir=args.art_dir,
                    evaluation_jsonl=args.evaluation_jsonl,
                    out=args.out,
                    base_corpus=args.base_corpus,
                    allow_non_train=args.allow_non_train,
                ),
                sort_keys=True,
            )
        )
        return 0
    parser.error("unhandled command")
    return 2


def _exec_rust_jope(argv: list[str]) -> None:
    configured = os.environ.get("JOPE_RUST_BIN")
    binary = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[2] / "rust/target/release/jope"
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(
            f"Rust jope binary is missing or not executable: {binary}. "
            "Build it with: cargo build --release --manifest-path rust/Cargo.toml "
            "-p jope-league"
        )
    os.execv(str(binary), [str(binary), *argv])
    raise RuntimeError("os.execv returned unexpectedly")


def _cmd_validate(args: argparse.Namespace) -> int:
    validator = RustValidator()
    if args.genomes:
        outputs = validator.validate_many(args.genomes)
    else:
        rows = [json.loads(line) for line in sys.stdin if line.strip()]
        outputs = validator.validate_many(rows)
    for row in outputs:
        print(json.dumps(row, sort_keys=True))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    for _ in range(args.n):
        print(json.dumps({"genome": sample_genome(rng), "valid": True}, sort_keys=True))
    for _ in range(args.invalid):
        print(json.dumps({"genome": corrupt_genome(sample_genome(rng), rng), "valid": False}, sort_keys=True))
    return 0


def _resolve_vs_path(args: argparse.Namespace) -> Path:
    if args.vs:
        return Path(args.vs)
    if args.dataset_dir:
        return Path(args.dataset_dir) / ACTIVE_TRAIN_SIDECAR
    raise SystemExit("propose-dataset requires --vs or --dataset-dir")


def _resolve_target_vs_path(args: argparse.Namespace, vs_path: Path) -> Path | None:
    if args.dataset_dir:
        return Path(args.dataset_dir) / "train_target.vs"
    return vs_path.parent / "train_target.vs"


def _eval_rows_from_args_or_stdin(args: argparse.Namespace) -> list[dict]:
    if args.genome or args.train_csv:
        missing = [name for name in ("dataset_hash", "train_csv", "genome") if getattr(args, name) is None]
        if missing:
            raise SystemExit("eval-regression missing required args: " + ", ".join(missing))
        return [
            {
                "dataset_hash": args.dataset_hash,
                "train_csv": args.train_csv,
                "test_csv": args.test_csv,
                "genome": args.genome,
                "folds": args.folds,
            }
        ]
    return [json.loads(line) for line in sys.stdin if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
