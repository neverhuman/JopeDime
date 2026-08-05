#!/usr/bin/env python3
"""Generate three immutable canary proposal lanes inside one pinned GPU job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch


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
    selection_policy: Path,
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
            "--selection-policy",
            str(selection_policy),
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
    parser.add_argument("--selection-policy", type=Path, required=True)
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

    requested_order = [
        (intent["task"], row["context_identity"])
        for row in intent["contexts"]
    ]
    requested = set(requested_order)
    if len(requested) != len(requested_order):
        raise RuntimeError("canary intent repeats a context")
    available = {
        (row["task"], row["dataset_hash"]): row
        for row in catalog["entries"]
    }
    if not requested.issubset(available):
        raise RuntimeError("canary context is absent from the pinned catalog")
    work = args.output.parent / "proposal-work"
    work.mkdir(parents=True, exist_ok=True)
    common = {
        "schema": "JopeMicroCohort.v2",
        "catalog_sha256": pointer["catalog_sha256"],
        "immutable": True,
    }
    candidate_outputs = []
    accepted_outputs = []
    for chunk_index, start in enumerate(range(0, len(requested_order), 128)):
        requested_chunk = requested_order[start : start + 128]
        ranked_padding = sorted(
            (identity for identity in available if identity not in requested),
            key=lambda identity: hashlib.sha256(
                (
                    intent["candidate_checkpoint_sha256"]
                    + "\0"
                    + str(intent["run_index"])
                    + "\0"
                    + str(chunk_index)
                    + "\0"
                    + identity[0]
                    + "\0"
                    + identity[1]
                ).encode()
            ).digest(),
        )
        identities = [*requested_chunk, *ranked_padding[: 128 - len(requested_chunk)]]
        if len(set(identities)) != 128:
            raise RuntimeError("catalog cannot pad a canary proposal cohort to 128")
        entries = [
            {
                "task": task,
                "dataset_hash": dataset,
                "micro_lane": "canary_padding",
            }
            for task, dataset in identities
        ]
        candidate_cohort = work / f"candidate-cohort-{chunk_index:02}.json"
        accepted_cohort = work / f"accepted-cohort-{chunk_index:02}.json"
        atomic_json(
            candidate_cohort,
            {
                **common,
                "checkpoint_sha256": candidate_sha,
                "wave": f"canary-candidate-{chunk_index:02}",
                "entries": entries,
            },
        )
        atomic_json(
            accepted_cohort,
            {
                **common,
                "checkpoint_sha256": accepted_sha,
                "wave": f"canary-accepted-{chunk_index:02}",
                "entries": entries,
            },
        )
        candidate_rows = work / f"candidate-{chunk_index:02}.jsonl"
        accepted_rows = work / f"accepted-{chunk_index:02}.jsonl"
        run_generator(
            args.generator,
            args.catalog_pointer,
            args.catalog,
            args.data,
            args.candidate,
            candidate_cohort,
            candidate_rows,
            args.selection_policy,
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
            args.selection_policy,
            # Common random numbers: checkpoint identity is the treatment.
            # A seed+1 accepted arm confounds lift with decoder draw luck.
            int(intent["seed"]),
        )
        candidate_outputs.append(candidate_rows)
        accepted_outputs.append(accepted_rows)

    def selected(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        seen: dict[tuple[str, str], set[str]] = defaultdict(set)
        for path in paths:
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

    candidate = selected(candidate_outputs)
    accepted = selected(accepted_outputs)
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

    from jope.grammars import grammar_for
    from jope.pipeline_text import parse_pipeline_actions
    from jope.policy.model_v3 import pad_sequences
    from jope.policy.tokens import encode_actions
    from jope.propose import load_any_model

    context_blob = args.data.read_bytes()
    stride_floats = 1536 + 512
    stride_bytes = stride_floats * 4

    def union_for(key: tuple[str, str]) -> list[dict]:
        union = []
        seen = set()
        for row in [*candidate.get(key, []), *accepted.get(key, []), *gp]:
            if row["semantic_sha256"] in seen:
                continue
            seen.add(row["semantic_sha256"])
            union.append(row)
        if not union:
            raise RuntimeError(f"canary common proposal union is empty for {key}")
        return union

    def conditional_advantages(checkpoint: Path) -> dict[tuple[str, str], float]:
        model = load_any_model(checkpoint, map_location="cpu").to("cuda")
        model.eval()
        if not hasattr(model, "reward_ablation"):
            raise RuntimeError("promotion checkpoint lacks the genome-only reward ablation head")
        output = {}
        with torch.no_grad():
            for key in requested_order:
                entry = available[key]
                offset = int(entry["catalog_index"]) * stride_bytes
                values = struct.unpack(
                    f"<{stride_floats}f",
                    context_blob[offset : offset + stride_bytes],
                )
                embedding = torch.tensor(values[:1536], dtype=torch.float32)
                target = torch.tensor(values[1536:], dtype=torch.float32)
                union = union_for(key)
                sequences = [
                    encode_actions(
                        "featmath",
                        parse_pipeline_actions(
                            row["canonical_genome"], grammar_for("featmath")
                        ),
                        model.vocab,
                        task=key[0],
                    )
                    for row in union
                ]
                conditional, _ = model.score_token_sequences(
                    embedding, target, sequences, device="cuda", task=key[0]
                )
                token_ids, pad = pad_sequences(sequences, device="cuda")
                ablation = model.reward_ablation(token_ids, pad)
                advantage = float((conditional - ablation).mean().cpu())
                if not (-float("inf") < advantage < float("inf")):
                    raise RuntimeError("promotion conditional advantage is non-finite")
                output[key] = advantage
        del model
        torch.cuda.empty_cache()
        return output

    promotion_evaluation = intent.get("promotion_evaluation_v3") is True
    candidate_advantage = (
        conditional_advantages(args.candidate) if promotion_evaluation else {}
    )
    accepted_advantage = (
        conditional_advantages(args.accepted) if promotion_evaluation else {}
    )
    contexts = []
    for row in intent["contexts"]:
        key = (intent["task"], row["context_identity"])
        common_union = union_for(key)
        context = {
            **row,
            "candidate": candidate.get(key, []),
            "accepted": accepted.get(key, []),
            "gp": gp,
        }
        if promotion_evaluation:
            context.update(
                {
                    "common_union_semantic_sha256": [
                        proposal["semantic_sha256"] for proposal in common_union
                    ],
                    "conditional_advantage": {
                        "candidate_checkpoint": candidate_advantage[key],
                        "accepted_checkpoint": accepted_advantage[key],
                    },
                }
            )
        contexts.append(context)
    output = {
        "schema": "CanaryProposalSet.v2",
        "intent_sha256": sha256_file(args.intent),
        "candidate_checkpoint_sha256": candidate_sha,
        "accepted_parent_checkpoint_sha256": accepted_sha,
        "catalog_sha256": pointer["catalog_sha256"],
        "proposal_selection_policy_sha256": sha256_file(args.selection_policy),
        "contexts": contexts,
    }
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
