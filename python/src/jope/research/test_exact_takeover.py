#!/usr/bin/env python3
"""Freeze a consistent, content-addressed takeover snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from test_exact_common import atomic_json, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SOURCE = Path("/home/ubuntu/jope-dime")
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")


def sqlite_backup(source: Path, destination: Path) -> dict:
    source_db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_db = sqlite3.connect(destination)
    try:
        source_db.backup(destination_db, pages=4_096, sleep=0.01)
        destination_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        integrity = destination_db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite takeover backup failed integrity check: {integrity}")
        return {
            "page_count": int(
                destination_db.execute("PRAGMA page_count").fetchone()[0]
            ),
            "freelist_count": int(
                destination_db.execute("PRAGMA freelist_count").fetchone()[0]
            ),
        }
    finally:
        destination_db.close()
        source_db.close()


def scheduler_state() -> dict:
    state = {}
    for command in ("ls", "nodes"):
        result = subprocess.run(
            [str(SCQ), "--json", command],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        state[command] = (
            json.loads(result.stdout)
            if result.returncode == 0
            else {
                "available": False,
                "returncode": result.returncode,
                "error": (result.stderr or result.stdout).strip()[:2_000],
            }
        )
    return state


def source_inventory() -> list[dict]:
    paths = [
        SOURCE / "Cargo.toml",
        SOURCE / "Cargo.lock",
        SOURCE / "src/bin/corpus_eval.rs",
        SOURCE / "target/release/corpus_eval",
        SOURCE / "manifests/test-exact-provenance-correction.v1.json",
        *sorted((SOURCE / "scripts").glob("test_exact_*.py")),
    ]
    trainer = Path("/home/ubuntu/jope-prime/src/jope/policy/train_v3.py")
    if trainer.is_file():
        paths.append(trainer)
    paths.extend(
        sorted(Path("/home/ubuntu/jope-prime/src/jope").rglob("*.py"))
    )
    if Path("/home/ubuntu/smart_cluster/etc/projects.toml").is_file():
        paths.append(Path("/home/ubuntu/smart_cluster/etc/projects.toml"))
    paths = sorted(set(paths))
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]


def freeze(output_root: Path) -> dict:
    corpus = ROOT / "corpus" / "strict-test-exact.sqlite"
    catalog_pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
    catalog_path = Path(catalog_pointer["catalog"])
    catalog = json.loads(catalog_path.read_text())
    task_counts = {
        task: sum(entry["task"] == task for entry in catalog["entries"])
        for task in ("regression", "binary")
    }
    accepted_pointer = ROOT / "training" / "accepted.json"
    if accepted_pointer.is_file():
        accepted = json.loads(accepted_pointer.read_text())
    else:
        from test_exact_train import accepted_checkpoint

        checkpoint, digest = accepted_checkpoint()
        accepted = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest}

    temporary = output_root / f".takeover-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        db_path = temporary / "strict-test-exact.sqlite"
        db_state = sqlite_backup(corpus, db_path)
        state_root = temporary / "state"
        state_root.mkdir()
        state_files = [
            ROOT / "catalog/current.json",
            catalog_path,
            Path(catalog_pointer["data"]),
            ROOT / "hof/current.json",
            ROOT / "indexes/champions.json",
            ROOT / "indexes/hall-of-fame.json",
            ROOT / "indexes/strict-wins.json",
            ROOT / "ops/backlog.sqlite",
            ROOT / "ops/status.json",
            ROOT / "ops/node-canaries.json",
            ROOT / "ops/direct-node-canaries.json",
            ROOT / "training/current-corpus.json",
            accepted_pointer,
        ]
        copied = []
        for source in state_files:
            if not source.is_file():
                continue
            destination = state_root / source.name
            if destination.exists():
                destination = state_root / f"{sha256_file(source)[:12]}-{source.name}"
            if source.suffix == ".sqlite":
                sqlite_backup(source, destination)
            else:
                shutil.copy2(source, destination)
            copied.append(
                {
                    "source": str(source),
                    "snapshot": str(destination.relative_to(temporary)),
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        scheduler = scheduler_state()
        atomic_json(temporary / "scheduler.json", scheduler)
        family_material = [
            (entry["task"], entry["dataset_hash"], entry["source_family"], entry["split"])
            for entry in catalog["entries"]
        ]
        family_hash = hashlib.sha256(
            json.dumps(family_material, separators=(",", ":")).encode()
        ).hexdigest()
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as database:
            watermark = int(
                database.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            )
        checkpoints = []
        checkpoint_paths = [Path(accepted["checkpoint"])]
        for result in sorted((ROOT / "training").glob("**/train-result.json")):
            value = json.loads(result.read_text())
            if value.get("checkpoint_path"):
                checkpoint_paths.append(Path(value["checkpoint_path"]))
        for checkpoint in sorted(set(checkpoint_paths)):
            if checkpoint.is_file():
                checkpoints.append(
                    {
                        "path": str(checkpoint),
                        "sha256": sha256_file(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                    }
                )
        receipt = {
            "schema": "JopeSelfplayTakeoverReceipt.v1",
            "created_at_ms": int(time.time() * 1_000),
            "corpus_watermark": watermark,
            "corpus_snapshot": "strict-test-exact.sqlite",
            "corpus_snapshot_sha256": sha256_file(db_path),
            "corpus_sqlite_state": db_state,
            "catalog_sha256": catalog_pointer["catalog_sha256"],
            "catalog_data_sha256": catalog_pointer["data_sha256"],
            "catalog_counts": task_counts,
            "catalog_count_gate_passed": task_counts
            == {
                task: int(count)
                for task, count in (
                    catalog.get("task_counts")
                    or {
                        "regression": catalog["regression_count"],
                        "binary": catalog["binary_count"],
                    }
                ).items()
            },
            "evaluator_sha256": sha256_file(SOURCE / "target/release/corpus_eval"),
            "accepted_ancestor_sha256": accepted["checkpoint_sha256"],
            "model_checkpoints": checkpoints,
            "source_family_split_sha256": family_hash,
            "scheduler_epoch": hashlib.sha256(
                json.dumps(scheduler, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "provenance_correction": str(
                SOURCE / "manifests/test-exact-provenance-correction.v1.json"
            ),
            "source_inventory": source_inventory(),
            "state_inventory": copied,
        }
        identity_material = {key: value for key, value in receipt.items() if key != "created_at_ms"}
        identity = hashlib.sha256(
            json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipt["takeover_sha256"] = identity
        atomic_json(temporary / "takeover.json", receipt)
        final = output_root / identity
        if final.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final)
        return {**receipt, "snapshot": str(final)}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "takeover")
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
