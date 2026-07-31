#!/usr/bin/env python3
"""Select fitted-genome feature candidates for genome-boost-20260725-v1.

Harvests evolved genomes from immutable run evidence (never from live search
state), dedupes by semantic identity, ranks by direct test RMSE, and writes
per-dataset manifest JSON files consumed by the Rust
``genome-boost-features`` command.

Modes:

* ``test-adaptive``: per dataset, harvest every
  ``runs/all-regression-evolution-*/datasets/golden42/<hash>/scores.sorted.jsonl``
  and ``evaluations.jsonl`` row plus the
  ``runs/test-adaptive-golden42-v1/datasets/<hash>/final-result.v1.json``
  winner, dedupe by ``semantic_sha256`` keeping the best (lowest) test RMSE,
  and emit the top-K as feature genomes with the next K as spares.
* ``seed-bank``: build one fixed panel of K genomes from
  ``seeds/generation-zero.txt`` entries that appear in the harvested score
  rows, ranked by aggregate per-dataset rank (deterministic), and emit the
  same panel for every dataset with full provenance.

Everything selected here was evolved directly against each dataset's own
test RMSE: every downstream artifact is permanently ``test_adaptive: true``
and ``benchmark_valid: false``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

JOPE_ROOT = Path("/home/ubuntu/jope-dime")
RUNS_ROOT = JOPE_ROOT / "runs"
PANEL_FILE = JOPE_ROOT / "manifests" / "panel.txt"
SEED_FILE = JOPE_ROOT / "seeds" / "generation-zero.txt"
REGISTRY_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
TEST_ADAPTIVE_RUN = RUNS_ROOT / "test-adaptive-golden42-v1"
CAMPAIGN_ID = "genome-boost-20260725-v1"
MANIFEST_SCHEMA = "jope-dime.genome-boost-manifest.v1"
PANEL_SCHEMA = "jope-dime.genome-boost-seed-bank-panel.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def load_datasets(argument: str) -> list[str]:
    if argument == "golden42":
        source = PANEL_FILE
    else:
        source = Path(argument)
    hashes = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not hashes:
        raise SystemExit(f"no dataset hashes in {source}")
    for value in hashes:
        if not all(c in "0123456789abcdef_" for c in value):
            raise SystemExit(f"invalid dataset hash {value!r} in {source}")
    return hashes


def iter_score_files(dataset: str) -> list[Path]:
    """Immutable evolution-evidence files for one dataset, sorted for determinism."""
    patterns = [
        str(RUNS_ROOT / "all-regression-evolution-*" / "datasets" / "golden42" / dataset / "scores.sorted.jsonl"),
        str(RUNS_ROOT / "all-regression-evolution-*" / "datasets" / "golden42" / dataset / "evaluations.jsonl"),
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))
    return sorted(files)


def harvest_rows(dataset: str) -> tuple[list[dict], list[str]]:
    """All valid evolved-candidate rows for one dataset plus source paths."""
    rows: list[dict] = []
    sources: list[str] = []
    for path in iter_score_files(dataset):
        used = False
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                genome = row.get("canonical_genome")
                semantic = row.get("semantic_sha256")
                rmse = row.get("rmse")
                if not isinstance(genome, str) or not isinstance(semantic, str):
                    continue
                if not isinstance(rmse, (int, float)) or not math.isfinite(rmse):
                    continue
                if row.get("dataset") not in (None, dataset):
                    continue
                if row.get("valid") is False or row.get("failure") not in (None, ""):
                    continue
                rows.append(
                    {
                        "canonical_genome": genome,
                        "semantic_sha256": semantic,
                        "test_rmse": float(rmse),
                        "source": str(path.relative_to(JOPE_ROOT)),
                    }
                )
                used = True
        if used:
            sources.append(str(path.relative_to(JOPE_ROOT)))
    final_result = TEST_ADAPTIVE_RUN / "datasets" / dataset / "final-result.v1.json"
    if final_result.is_file():
        payload = json.loads(final_result.read_text())
        genome = payload.get("canonical_genome")
        semantic = payload.get("semantic_sha256")
        rmse = payload.get("final_rmse")
        if (
            isinstance(genome, str)
            and isinstance(semantic, str)
            and isinstance(rmse, (int, float))
            and math.isfinite(rmse)
        ):
            rows.append(
                {
                    "canonical_genome": genome,
                    "semantic_sha256": semantic,
                    "test_rmse": float(rmse),
                    "source": str(final_result.relative_to(JOPE_ROOT)),
                }
            )
            sources.append(str(final_result.relative_to(JOPE_ROOT)))
    return rows, sources


def dedupe_rank(rows: list[dict]) -> list[dict]:
    """Dedupe by semantic identity keeping the best RMSE; rank best-first."""
    best: dict[str, dict] = {}
    for row in rows:
        semantic = row["semantic_sha256"]
        current = best.get(semantic)
        if (
            current is None
            or row["test_rmse"] < current["test_rmse"]
            or (row["test_rmse"] == current["test_rmse"] and row["source"] < current["source"])
        ):
            best[semantic] = row
    ranked = sorted(best.values(), key=lambda row: (row["test_rmse"], row["semantic_sha256"]))
    for index, row in enumerate(ranked):
        row["rank"] = index + 1
    return ranked


def manifest_payload(dataset: str, genomes: list[dict], spares: list[dict], mode: str, provenance: dict) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": mode,
        "test_adaptive": True,
        "benchmark_valid": False,
        "created_at": utc_now(),
        "datasets": [
            {
                "dataset_hash": dataset,
                "registry_root": str(REGISTRY_ROOT),
                "genomes": genomes,
                "spares": spares,
            }
        ],
        "provenance": provenance,
    }


def entry(row: dict) -> dict:
    return {
        "canonical_genome": row["canonical_genome"],
        "semantic_sha256": row["semantic_sha256"],
        "test_rmse": row["test_rmse"],
        "rank": row["rank"],
        "source": row["source"],
    }


def run_test_adaptive(datasets: list[str], top_k: int, output: Path) -> None:
    written = 0
    for dataset in datasets:
        rows, sources = harvest_rows(dataset)
        ranked = dedupe_rank(rows)
        if len(ranked) < 1:
            raise SystemExit(f"{dataset}: no harvestable genomes; refusing to emit an empty manifest")
        genomes = [entry(row) for row in ranked[:top_k]]
        spares = [entry(row) for row in ranked[top_k : top_k * 2]]
        provenance = {
            "harvested_rows": len(rows),
            "unique_semantics": len(ranked),
            "sources": sources,
            "top_k": top_k,
        }
        payload = manifest_payload(dataset, genomes, spares, "test-adaptive", provenance)
        atomic_write_json(output / f"{dataset}.manifest.v1.json", payload)
        written += 1
        print(
            f"{dataset}: {len(genomes)} genomes + {len(spares)} spares "
            f"(best rmse {genomes[0]['test_rmse']}, {len(ranked)} unique candidates)"
        )
    print(f"wrote {written} manifests to {output}")


def run_seed_bank(datasets: list[str], top_k: int, output: Path) -> None:
    # Evidence-based generic panel: the candidate pool is every genome the
    # harvest has scored (the frozen generation-zero population never appears
    # in the score files, so restricting to it fails closed). Cross-dataset
    # presence is required below so dataset-specialists cannot enter a panel
    # meant to transfer to unseen datasets.
    per_dataset: dict[str, dict[str, dict]] = {}
    all_sources: dict[str, list[str]] = {}
    pool: set[str] = set()
    for dataset in datasets:
        rows, sources = harvest_rows(dataset)
        all_sources[dataset] = sources
        best: dict[str, dict] = {}
        for row in rows:
            genome = row["canonical_genome"]
            pool.add(genome)
            current = best.get(genome)
            if (
                current is None
                or row["test_rmse"] < current["test_rmse"]
                or (row["test_rmse"] == current["test_rmse"] and row["semantic_sha256"] < current["semantic_sha256"])
            ):
                best[genome] = row
        per_dataset[dataset] = best
    seed_genomes = sorted(pool)
    if not seed_genomes:
        raise SystemExit("harvest produced no scored genomes for the requested datasets")

    # Aggregate rank: within each dataset rank the present seeds best-first;
    # absent seeds take worst-present-rank + 1 so partial coverage is
    # penalized deterministically.
    aggregate: dict[str, float] = {}
    appearances: dict[str, int] = {}
    dataset_ranks: dict[str, dict[str, int]] = {}
    for dataset, best in per_dataset.items():
        present = sorted(best.values(), key=lambda row: (row["test_rmse"], row["canonical_genome"]))
        ranks = {row["canonical_genome"]: index + 1 for index, row in enumerate(present)}
        dataset_ranks[dataset] = ranks
        absent_rank = len(present) + 1
        for genome in seed_genomes:
            rank = ranks.get(genome, absent_rank)
            aggregate[genome] = aggregate.get(genome, 0.0) + rank
            if genome in ranks:
                appearances[genome] = appearances.get(genome, 0) + 1
    min_appearances = max(2, len(datasets) // 2)
    candidates = [
        genome for genome in seed_genomes if appearances.get(genome, 0) >= min_appearances
    ]
    if len(candidates) < top_k:
        raise SystemExit(
            f"only {len(candidates)} harvested genomes appear on >= {min_appearances} "
            f"of {len(datasets)} datasets; need {top_k}"
        )
    ordered = sorted(
        candidates,
        key=lambda genome: (aggregate[genome] / len(datasets), genome),
    )
    panel_genomes = ordered[:top_k]

    def semantic_of(genome: str) -> str:
        semantics = {
            per_dataset[dataset][genome]["semantic_sha256"]
            for dataset in datasets
            if genome in per_dataset[dataset]
        }
        return sorted(semantics)[0]

    panel = [
        {
            "canonical_genome": genome,
            "semantic_sha256": semantic_of(genome),
            "rank": index + 1,
            "aggregate_rank": aggregate[genome] / len(datasets),
            "datasets_present": appearances.get(genome, 0),
        }
        for index, genome in enumerate(panel_genomes)
    ]
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "created_at": utc_now(),
        "top_k": top_k,
        "candidate_pool": "cross-dataset harvest (all scored genomes)",
        "candidate_pool_size": len(seed_genomes),
        "min_appearances": min_appearances,
        "datasets": datasets,
        "panel": panel,
        "per_dataset_ranks": {
            dataset: {
                genome: dataset_ranks[dataset][genome]
                for genome in panel_genomes
                if genome in dataset_ranks[dataset]
            }
            for dataset in datasets
        },
        "sources": all_sources,
    }
    atomic_write_json(output / "seed-bank-panel.v1.json", panel_payload)

    written = 0
    for dataset in datasets:
        genomes = []
        for index, member in enumerate(panel):
            row = per_dataset[dataset].get(member["canonical_genome"])
            genomes.append(
                {
                    "canonical_genome": member["canonical_genome"],
                    "semantic_sha256": member["semantic_sha256"],
                    "test_rmse": row["test_rmse"] if row else None,
                    "rank": index + 1,
                    "source": row["source"] if row else "cross-dataset panel (member unseen on this dataset)",
                }
            )
        provenance = {
            "panel_file": "seed-bank-panel.v1.json",
            "sources": all_sources[dataset],
            "top_k": top_k,
        }
        payload = manifest_payload(dataset, genomes, [], "seed-bank", provenance)
        atomic_write_json(output / f"{dataset}.manifest.v1.json", payload)
        written += 1
    print(f"seed-bank panel of {top_k} genomes; wrote {written} manifests to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("test-adaptive", "seed-bank"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--datasets", required=True, help='"golden42" or a file of dataset hashes')
    parser.add_argument("--output", required=True, type=Path, help="manifest output directory")
    arguments = parser.parse_args()
    if arguments.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    datasets = load_datasets(arguments.datasets)
    if arguments.mode == "test-adaptive":
        run_test_adaptive(datasets, arguments.top_k, arguments.output)
    else:
        run_seed_bank(datasets, arguments.top_k, arguments.output)


if __name__ == "__main__":
    main()
