#!/usr/bin/env python3
"""Current-checkpoint DL proposal export and in-budget GP injection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from test_exact_common import sha256_file, sha256_text

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")


def export_gp_seeds(
    proposal_path: Path,
    *,
    task: str,
    dataset_hash: str,
    checkpoint_sha256: str,
) -> dict:
    rows = []
    with proposal_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task") != task or row.get("dataset_hash") != dataset_hash:
                continue
            if row.get("model_checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError("GP seed proposal checkpoint provenance drift")
            if sha256_text(row["canonical_genome"]) != row["semantic_sha256"]:
                raise RuntimeError("GP seed semantic identity drift")
            rows.append(row)
    rows.sort(key=lambda row: int(row["slot"]))
    if (
        len(rows) != 64
        or [int(row["slot"]) for row in rows] != list(range(64))
        or len({row["semantic_sha256"] for row in rows}) != 64
    ):
        raise RuntimeError("GP seed export requires all 64 unique proposal slots")
    return {
        "schema": "JopeGpSeedExport.v1",
        "task": task,
        "dataset_hash": dataset_hash,
        "model_checkpoint_sha256": checkpoint_sha256,
        "proposal_set_sha256": sha256_file(proposal_path),
        "population_budget_delta": 0,
        "generation_zero_consumption": "replace_existing_seed_slots",
        "proposals": rows,
    }


def inject_promoted_proposals(
    population: Iterable[Mapping[str, object]],
    proposals: Iterable[Mapping[str, object]],
    *,
    checkpoint_sha256: str,
    elite_count: int,
    count: int = 8,
) -> tuple[list[dict], dict]:
    """Replace the lowest non-elites; never increase population size."""

    members = [dict(row) for row in population]
    if not 0 <= elite_count <= len(members):
        raise ValueError("invalid GP elite count")
    existing = {str(row["semantic_sha256"]) for row in members}
    candidates = []
    for raw in proposals:
        row = dict(raw)
        if (
            row.get("model_checkpoint_sha256") != checkpoint_sha256
            or sha256_text(str(row["canonical_genome"])) != row.get("semantic_sha256")
            or row["semantic_sha256"] in existing
        ):
            continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -float(
                row["score"]
                if row.get("score") is not None
                else row["proposal_score"]
                if row.get("proposal_score") is not None
                else float("-inf")
            ),
            int(row.get("slot") or 0),
            str(row["semantic_sha256"]),
        )
    )
    selected = candidates[: min(count, len(members) - elite_count)]
    replaceable = sorted(
        range(elite_count, len(members)),
        key=lambda index: (
            float(
                members[index]["score"]
                if members[index].get("score") is not None
                else float("-inf")
            ),
            -index,
        ),
    )[: len(selected)]
    replaced = []
    for index, proposal in zip(replaceable, selected, strict=True):
        prior = members[index]
        members[index] = {
            **proposal,
            "dl_origin": True,
            "injected_checkpoint_sha256": checkpoint_sha256,
            "ancestry": {
                "operation": "checkpoint_proposal_injection",
                "parents": [proposal["semantic_sha256"]],
            },
        }
        replaced.append(
            {
                "slot": index,
                "removed_semantic_sha256": prior["semantic_sha256"],
                "injected_semantic_sha256": proposal["semantic_sha256"],
            }
        )
    return members, {
        "schema": "JopeGpProposalInjection.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "population_before": len(members),
        "population_after": len(members),
        "elite_count": elite_count,
        "requested": count,
        "injected": len(selected),
        "replacements": replaced,
    }


def lineage_contribution(
    lineage: Iterable[Mapping[str, object]],
    final_champion_semantic_sha256: str,
) -> dict:
    rows = [dict(row) for row in lineage]
    dl_origins = {
        str(row["semantic_sha256"])
        for row in rows
        if row.get("dl_origin") is True
    }
    descendants = {
        str(row["semantic_sha256"])
        for row in rows
        if set(row.get("parents") or []) & dl_origins
    }
    strict_wins = sum(
        row.get("strict_win") is True
        and (
            row.get("semantic_sha256") in dl_origins
            or row.get("semantic_sha256") in descendants
        )
        for row in rows
    )
    return {
        "dl_origin_parents": len(dl_origins),
        "dl_origin_descendants": len(descendants),
        "dl_origin_strict_wins": strict_wins,
        "final_champion_has_dl_contribution": (
            final_champion_semantic_sha256 in dl_origins
            or final_champion_semantic_sha256 in descendants
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--task", choices=("regression", "binary"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = export_gp_seeds(
        args.proposals,
        task=args.task,
        dataset_hash=args.dataset,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        temporary.write_text(text)
        os.replace(temporary, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "output_sha256": sha256_file(args.output),
                    "proposals": 64,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
