#!/usr/bin/env python3
"""Generate three immutable canary proposal lanes inside one pinned GPU job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def accepted_checkpoint_sha256(intent: dict) -> str:
    value = intent.get("accepted_parent_checkpoint_sha256") or intent.get(
        "accepted_checkpoint_sha256"
    )
    if not value:
        raise RuntimeError("canary intent lacks accepted checkpoint hash")
    return str(value)


def run_generator(
    generator: Path,
    pointer: Path,
    catalog: Path,
    data: Path,
    checkpoint: Path,
    cohort: Path,
    output: Path,
    seed: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--catalog-pointer",
            str(pointer),
            "--catalog",
            str(catalog),
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--cohort",
            str(cohort),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--seed",
            str(seed),
            "--proposals-per-dataset",
            "128",
            "--batch-contexts",
            "64",
            "--allow-deficient-floor",
            "0",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--catalog-pointer", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intent = json.loads(args.intent.read_text())
    pointer = json.loads(args.catalog_pointer.read_text())
    catalog = json.loads(args.catalog.read_text())
    if sha256_file(args.catalog) != pointer["catalog_sha256"]:
        raise RuntimeError("canary catalog hash drift")
    if sha256_file(args.data) != pointer["data_sha256"]:
        raise RuntimeError("canary catalog data hash drift")
    candidate_sha = sha256_file(args.candidate)
    accepted_sha = sha256_file(args.accepted)
    if candidate_sha != intent["candidate_checkpoint_sha256"]:
        raise RuntimeError("canary candidate checkpoint hash drift")
    if accepted_sha != accepted_checkpoint_sha256(intent):
        raise RuntimeError("canary accepted checkpoint hash drift")

    requested = {
        (intent["task"], row["context_identity"])
        for row in intent["contexts"]
    }
    available = {
        (row["task"], row["dataset_hash"]): row
        for row in catalog["entries"]
    }
    if not requested.issubset(available):
        raise RuntimeError("canary context is absent from the pinned catalog")
    ranked_padding = sorted(
        (identity for identity in available if identity not in requested),
        key=lambda identity: hashlib.sha256(
            (
                intent["candidate_checkpoint_sha256"]
                + "\0"
                + str(intent["run_index"])
                + "\0"
                + identity[0]
                + "\0"
                + identity[1]
            ).encode()
        ).digest(),
    )
    identities = [*sorted(requested), *ranked_padding[: 128 - len(requested)]]
    if len(set(identities)) != 128:
        raise RuntimeError("catalog cannot pad the canary proposal cohort to 128")
    entries = [
        {
            "task": task,
            "dataset_hash": dataset,
            "micro_lane": "canary_padding",
        }
        for task, dataset in identities
    ]
    work = args.output.parent / "proposal-work"
    work.mkdir(parents=True, exist_ok=True)
    candidate_cohort = work / "candidate-cohort.json"
    accepted_cohort = work / "accepted-cohort.json"
    common = {
        "schema": "JopeMicroCohort.v2",
        "catalog_sha256": pointer["catalog_sha256"],
        "immutable": True,
        "entries": entries,
    }
    atomic_json(
        candidate_cohort,
        {**common, "checkpoint_sha256": candidate_sha, "wave": "canary-candidate"},
    )
    atomic_json(
        accepted_cohort,
        {**common, "checkpoint_sha256": accepted_sha, "wave": "canary-accepted"},
    )
    candidate_rows = work / "candidate.jsonl"
    accepted_rows = work / "accepted.jsonl"
    run_generator(
        args.generator,
        args.catalog_pointer,
        args.catalog,
        args.data,
        args.candidate,
        candidate_cohort,
        candidate_rows,
        int(intent["seed"]),
    )
    run_generator(
        args.generator,
        args.catalog_pointer,
        args.catalog,
        args.data,
        args.accepted,
        accepted_cohort,
        accepted_rows,
        int(intent["seed"]) + 1,
    )

    def selected(path: Path) -> dict[tuple[str, str], list[dict]]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        seen: dict[tuple[str, str], set[str]] = defaultdict(set)
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["task"], row["dataset_hash"])
                if key not in requested or row["semantic_sha256"] in seen[key]:
                    continue
                if len(grouped[key]) == 8:
                    continue
                grouped[key].append(
                    {
                        "canonical_genome": row["canonical_genome"],
                        "semantic_sha256": row["semantic_sha256"],
                        "proposal_score": row.get("proposal_score"),
                    }
                )
                seen[key].add(row["semantic_sha256"])
        return grouped

    candidate = selected(candidate_rows)
    accepted = selected(accepted_rows)
    bank = json.loads(args.seed_bank.read_text())
    gp = []
    gp_seen = set()
    for row in bank["tasks"][intent["task"]]:
        if row.get("dl_origin", False) or row["semantic_sha256"] in gp_seen:
            continue
        gp.append(
            {
                "canonical_genome": row["canonical_genome"],
                "semantic_sha256": row["semantic_sha256"],
            }
        )
        gp_seen.add(row["semantic_sha256"])
        if len(gp) == 8:
            break
    if len(gp) != 8:
        raise RuntimeError("canary cannot fill eight zero-DL-ancestry GP proposals")
    contexts = []
    for row in intent["contexts"]:
        key = (intent["task"], row["context_identity"])
        contexts.append(
            {
                **row,
                "candidate": candidate.get(key, []),
                "accepted": accepted.get(key, []),
                "gp": gp,
            }
        )
    output = {
        "schema": "CanaryProposalSet.v2",
        "intent_sha256": sha256_file(args.intent),
        "candidate_checkpoint_sha256": candidate_sha,
        "accepted_parent_checkpoint_sha256": accepted_sha,
        "catalog_sha256": pointer["catalog_sha256"],
        "contexts": contexts,
    }
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
