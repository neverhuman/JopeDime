"""Operator CLI for local broad-shadow artifact construction and gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from jope.dime_score.broad_train import BroadTrainConfig, train_broad_critic

from .catalog import build_broad_catalog
from .contracts import (
    validate_broad_catalog,
    validate_broad_corpus_snapshot,
    validate_broad_receipt,
    validate_probe_slate,
)
from .corpus import BroadCorpus, materialize_broad_corpus
from .experiment import broad_critic_experiment
from .gates import evaluate_offline_gate, evaluate_prospective_gate
from .probe import build_probe_slate
from .receipts import ReceiptStore


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jope broad-shadow")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("kind", choices=("catalog", "slate", "receipt", "corpus"))
    validate.add_argument("path", type=Path)
    validate.add_argument("--production", action="store_true")

    catalog = sub.add_parser("freeze-catalog")
    catalog.add_argument("--rows", required=True, type=Path)
    catalog.add_argument("--out", required=True, type=Path)
    catalog.add_argument("--created-at-ms", required=True, type=int)
    catalog.add_argument("--foundation-cache-count", type=int, default=27_875)
    catalog.add_argument("--production", action="store_true")

    slate = sub.add_parser("freeze-probe-slate")
    slate.add_argument("--spec", required=True, type=Path)
    slate.add_argument("--out", required=True, type=Path)
    slate.add_argument("--production", action="store_true")

    ingest = sub.add_parser("ingest-receipts")
    ingest.add_argument("--jsonl", required=True, type=Path)
    ingest.add_argument("--store", required=True, type=Path)

    verify = sub.add_parser("verify-receipts")
    verify.add_argument("--store", required=True, type=Path)

    materialize = sub.add_parser("materialize-corpus")
    materialize.add_argument("--catalog", required=True, type=Path)
    materialize.add_argument("--probe-slate", required=True, type=Path)
    materialize.add_argument("--receipt-store", required=True, type=Path)
    materialize.add_argument("--tokens", required=True, type=Path)
    materialize.add_argument("--vocabulary", required=True, type=Path)
    materialize.add_argument("--vocabulary-sha256", required=True)
    materialize.add_argument("--catalog-root", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)
    materialize.add_argument("--created-at-ms", required=True, type=int)
    materialize.add_argument("--production", action="store_true")

    gate = sub.add_parser("gate")
    gate.add_argument("kind", choices=("offline", "prospective"))
    gate.add_argument("--rows", required=True, type=Path)
    gate.add_argument("--out", required=True, type=Path)
    gate.add_argument("--seed", type=int, default=17)
    gate.add_argument("--diagnostic", action="store_true")

    experiment = sub.add_parser("experiment-plan")
    experiment.add_argument("--out", required=True, type=Path)
    experiment.add_argument("--seeds", default="17,29,43")

    train = sub.add_parser("train")
    train.add_argument("--corpus", required=True, type=Path)
    train.add_argument("--run-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--device", default="cuda")
    train.add_argument("--no-autotune", action="store_true")
    train.add_argument("--minimum-free-vram-mib", type=int, default=4096)
    train.add_argument("--maximum-loader-wait-percent", type=float, default=10.0)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--warm-start", type=Path)
    train.add_argument("--genome-only", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "validate":
        value = _json(args.path)
        if args.kind == "catalog":
            validate_broad_catalog(value, production=args.production)
        elif args.kind == "slate":
            validate_probe_slate(value, production=args.production)
        elif args.kind == "receipt":
            validate_broad_receipt(value)
        else:
            validate_broad_corpus_snapshot(value, production=args.production)
            BroadCorpus(args.path.parent)
        print("valid")
        return 0
    if args.command == "freeze-catalog":
        rows = _json(args.rows)
        if not isinstance(rows, list):
            raise ValueError("catalog rows input must be a JSON list")
        value = build_broad_catalog(
            rows,
            created_at_ms=args.created_at_ms,
            foundation_cache_count=args.foundation_cache_count,
            production=args.production,
        )
        _write(args.out, value)
        print(args.out)
        return 0
    if args.command == "freeze-probe-slate":
        spec = _json(args.spec)
        value = build_probe_slate(
            catalog_sha256=spec["catalog_sha256"],
            anchors=spec["anchors"],
            pilot_datasets=spec["pilot_datasets"],
            candidate_pools=spec["candidate_pools"],
            seed=int(spec["seed"]),
            created_at_ms=int(spec["created_at_ms"]),
            production=args.production,
        )
        _write(args.out, value)
        print(args.out)
        return 0
    if args.command == "ingest-receipts":
        store = ReceiptStore(args.store)
        rows = (
            json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        print(json.dumps(store.import_all(rows), sort_keys=True))
        return 0
    if args.command == "verify-receipts":
        print(json.dumps(ReceiptStore(args.store).verify(), sort_keys=True))
        return 0
    if args.command == "materialize-corpus":
        tokenized = _json(args.tokens)
        vocabulary = _json(args.vocabulary)
        final = materialize_broad_corpus(
            catalog=_json(args.catalog),
            probe_slate=_json(args.probe_slate),
            receipts=list(ReceiptStore(args.receipt_store)),
            tokenized_genomes=tokenized,
            vocabulary_tokens=vocabulary,
            vocabulary_sha256=args.vocabulary_sha256,
            catalog_root=args.catalog_root,
            output_root=args.output_root,
            created_at_ms=args.created_at_ms,
            production=args.production,
        )
        print(final)
        return 0
    if args.command == "gate":
        rows = _json(args.rows)
        result = (
            evaluate_offline_gate(rows, seed=args.seed, production=not args.diagnostic)
            if args.kind == "offline"
            else evaluate_prospective_gate(rows, seed=args.seed, production=not args.diagnostic)
        )
        _write(args.out, result)
        print(args.out)
        return int(not result["passed"])
    if args.command == "experiment-plan":
        seeds = [int(value) for value in args.seeds.split(",") if value]
        _write(args.out, broad_critic_experiment(paired_seeds=seeds))
        print(args.out)
        return 0
    if args.command == "train":
        summary = train_broad_critic(
            BroadTrainConfig(
                corpus_root=args.corpus,
                run_dir=args.run_dir,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                device=args.device,
                autotune_batch_size=not args.no_autotune,
                minimum_free_vram_mib=args.minimum_free_vram_mib,
                maximum_loader_wait_percent=args.maximum_loader_wait_percent,
                resume=args.resume,
                warm_start=args.warm_start,
                genome_only=args.genome_only,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise AssertionError("unhandled broad-shadow command")
