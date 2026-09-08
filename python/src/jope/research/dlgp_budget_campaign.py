#!/usr/bin/env python3
"""Preserve and verify evidence for the budget-first DLGP campaign.

The round-2 promoter is deliberately read-only with respect to its source.  It
copies the complete replay surface into an immutable, content-addressed bundle
and verifies the copied bytes with both hashlib and the system sha256sum.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "JopeDlgpRound2Evidence.v1"
FINALES = ("cv7", "t3_boot", "t4_rcv1se", "t2_uni5", "t2_greedy")
SEARCHES = ("c1000", "t5")
SEEDS = (101, 202)
EXPECTED_RESULTS = 88
EXPECTED_SCORE_CELLS = 440
TOP_LEVEL_FILES = (
    "r2-verdict.json",
    "r2-score-cache.json",
    "regression42-baselines.json",
    "regression42-shapes.json",
    "regression42-strata.json",
    "prior-global-lodo.json",
    "prior-neighbor.json",
    "dl-proposal-manifest.json",
    "analyze_screen2.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def recompute_verdict(results: list[dict[str, Any]], scores: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    identities: set[tuple[str, str, int]] = set()
    datasets: set[str] = set()
    for result in results:
        if result.get("schema") != "JopeStrategyScreenResult.v1":
            raise ValueError("round-2 result schema drift")
        if result.get("test_csv_read") is not False:
            raise ValueError("round-2 search result does not attest test_csv_read=false")
        identity = (result["dataset_hash"], result["strategy"], result["master_seed"])
        if identity in identities:
            raise ValueError(f"duplicate round-2 result identity: {identity}")
        identities.add(identity)
        datasets.add(result["dataset_hash"])

    expected_identities = {
        (dataset, search, seed)
        for dataset in datasets
        for search in SEARCHES
        for seed in SEEDS
    }
    if identities != expected_identities:
        raise ValueError("round-2 result identities do not form the complete paired grid")

    verdict: dict[str, Any] = {}
    for search in SEARCHES:
        for finale in FINALES:
            robust_wins = 0
            any_wins = 0
            log_ratios: list[float] = []
            for dataset in sorted(datasets):
                target = baselines[dataset]["target_rmse"]
                values = [scores.get(f"{dataset}|{finale}|{search}|{seed}") for seed in SEEDS]
                finite_values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
                if len(finite_values) < 2 or target <= 0:
                    continue
                wins = [value < target for value in finite_values]
                robust_wins += int(all(wins))
                any_wins += int(any(wins))
                log_ratios.append(sum(math.log(value / target) for value in finite_values if value > 0) / len(finite_values))
            verdict[f"{search}/{finale}"] = {
                "robust_wins": robust_wins,
                "any_wins": any_wins,
                "med_lr": round(median(log_ratios), 4) if log_ratios else None,
            }
    return verdict


def parse_job(job_path: Path, config_path: Path) -> tuple[str, dict[str, str]]:
    with job_path.open("rb") as handle:
        job = tomllib.load(handle)
    inputs = {entry["name"]: entry for entry in job.get("input", [])}
    if set(inputs) != {"screenbin", "config", "train"}:
        raise ValueError(f"unexpected SCQ inputs in {job_path}")
    if inputs["config"]["sha256"] != sha256_file(config_path):
        raise ValueError(f"config digest drift in {job_path}")
    return inputs["screenbin"]["sha256"], {
        "source": inputs["train"]["source"],
        "sha256": inputs["train"]["sha256"],
    }


def inventory(root: Path, relative_files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "bytes": (root / path).stat().st_size,
            "sha256": sha256_file(root / path),
        }
        for path in sorted(relative_files)
    ]


def verify_with_sha256sum(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        output = subprocess.run(
            ["/usr/bin/sha256sum", str(root / entry["path"])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if output.split()[0] != entry["sha256"]:
            raise ValueError(f"sha256sum disagrees for {entry['path']}")


def validate_retention_date(value: str) -> None:
    retention = dt.date.fromisoformat(value)
    if retention <= dt.datetime.now(dt.UTC).date():
        raise ValueError("source retention date must be in the future")


def collect_source(source: Path) -> tuple[list[Path], list[dict[str, Any]], str]:
    relative_files = [Path(name) for name in TOP_LEVEL_FILES]
    result_paths = sorted(source.glob("screen-results2/*/result.json"))
    config_paths = sorted(source.glob("screen-jobs2/*/config.json"))
    job_paths = sorted(source.glob("screen-jobs2/*/job.toml"))
    if not (len(result_paths) == len(config_paths) == len(job_paths) == EXPECTED_RESULTS):
        raise ValueError("round-2 source must contain exactly 88 results, configs, and job specs")

    train_by_dataset: dict[str, dict[str, str]] = {}
    binary_digests: set[str] = set()
    for result_path, config_path, job_path in zip(result_paths, config_paths, job_paths, strict=True):
        result_job = result_path.parent.name
        if config_path.parent.name != result_job or job_path.parent.name != result_job:
            raise ValueError("round-2 result/config/job directories are not identical")
        config = read_json(config_path)
        result = read_json(result_path)
        expected = (config["dataset_hash"], config["strategy"], config["master_seed"])
        actual = (result["dataset_hash"], result["strategy"], result["master_seed"])
        if expected != actual:
            raise ValueError(f"result/config identity mismatch in {result_job}")
        binary_digest, train = parse_job(job_path, config_path)
        binary_digests.add(binary_digest)
        prior = train_by_dataset.setdefault(config["dataset_hash"], train)
        if prior != train:
            raise ValueError(f"training binding drift for {config['dataset_hash']}")
        relative_files.extend(
            [
                result_path.relative_to(source),
                config_path.relative_to(source),
                job_path.relative_to(source),
            ]
        )
    if len(train_by_dataset) != 22 or len(binary_digests) != 1:
        raise ValueError("round-2 source must bind 22 training tables and one binary digest")
    train_manifest = [
        {"dataset_hash": dataset, **binding}
        for dataset, binding in sorted(train_by_dataset.items())
    ]
    return relative_files, train_manifest, binary_digests.pop()


def verify_round2_payload(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("round-2 evidence manifest schema drift")
    entries = manifest.get("files", [])
    if not entries or entries != sorted(entries, key=lambda entry: entry["path"]):
        raise ValueError("round-2 evidence inventory is empty or unsorted")
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"round-2 evidence byte drift: {entry['path']}")
    if canonical_sha256(entries) != manifest["payload_sha256"]:
        raise ValueError("round-2 evidence payload identity drift")
    verify_with_sha256sum(root, entries)

    results = [read_json(path) for path in sorted(root.glob("screen-results2/*/result.json"))]
    if len(results) != EXPECTED_RESULTS:
        raise ValueError("round-2 evidence no longer contains 88 results")
    scores = read_json(root / "r2-score-cache.json")
    if len(scores) != EXPECTED_SCORE_CELLS:
        raise ValueError("round-2 evidence no longer contains 440 cached scores")
    verdict = recompute_verdict(results, scores, read_json(root / "regression42-baselines.json"))
    if verdict != read_json(root / "r2-verdict.json") or verdict != manifest["recomputed_verdict"]:
        raise ValueError("round-2 verdict does not replay identically")


def promote(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    validate_retention_date(args.retention_until)
    relative_files, train_manifest, binary_digest = collect_source(source)
    for path in relative_files:
        if not (source / path).is_file():
            raise ValueError(f"required round-2 evidence is missing: {path}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=".round2-stage-", dir=output_root))
    try:
        assert temporary is not None
        for relative in sorted(relative_files):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        source_copy = temporary / "source" / args.strategy_source.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.strategy_source.resolve(), source_copy)
        relative_files.append(source_copy.relative_to(temporary))

        train_manifest_path = temporary / "train-manifest.json"
        train_manifest_path.write_text(json.dumps({"schema": "JopeDlgpRound2TrainManifest.v1", "datasets": train_manifest}, indent=2) + "\n")
        relative_files.append(train_manifest_path.relative_to(temporary))

        entries = inventory(temporary, relative_files)
        payload_sha256 = canonical_sha256(entries)
        destination = output_root / payload_sha256
        results = [read_json(path) for path in sorted(temporary.glob("screen-results2/*/result.json"))]
        scores = read_json(temporary / "r2-score-cache.json")
        verdict = recompute_verdict(results, scores, read_json(temporary / "regression42-baselines.json"))
        if len(scores) != EXPECTED_SCORE_CELLS or verdict != read_json(temporary / "r2-verdict.json"):
            raise ValueError("round-2 source verdict does not replay identically")
        verify_with_sha256sum(temporary, entries)
        manifest = {
            "schema": SCHEMA,
            "payload_sha256": payload_sha256,
            "source_root": str(source),
            "source_retention_until": args.retention_until,
            "source_originals_deleted": False,
            "source_revision": args.source_revision,
            "strategy_source_sha256": sha256_file(source_copy),
            "binary_sha256": binary_digest,
            "result_count": len(results),
            "score_cell_count": len(scores),
            "test_csv_read": False,
            "hash_verifications": ["python_hashlib_sha256", "coreutils_sha256sum"],
            "recomputed_verdict": verdict,
            "files": entries,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_round2_payload(temporary, manifest)
        if destination.exists():
            verify_round2_payload(destination, read_json(destination / "manifest.json"))
        else:
            os.replace(temporary, destination)
            temporary = None
        print(destination)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def verify(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    manifest = read_json(bundle / "manifest.json")
    verify_round2_payload(bundle, manifest)
    print(manifest["payload_sha256"])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    promote_parser = commands.add_parser("promote-round2")
    promote_parser.add_argument("--source", type=Path, required=True)
    promote_parser.add_argument("--output-root", type=Path, required=True)
    promote_parser.add_argument("--retention-until", required=True)
    promote_parser.add_argument("--source-revision", required=True)
    promote_parser.add_argument("--strategy-source", type=Path, required=True)
    promote_parser.set_defaults(handler=promote)
    verify_parser = commands.add_parser("verify-round2")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.set_defaults(handler=verify)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
