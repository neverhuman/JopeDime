#!/usr/bin/env python3
"""Build the next immutable strict-test wave from current policy-era parents."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from test_exact_campaign import HEAD_PREFIXES, dataset_order, top_level_stages
from test_exact_common import (
    atomic_json,
    compose_seed,
    evaluation_protocol_sha256,
    prime_source_root,
    selfplay_authority,
    sha256_file,
    sha256_text,
    validate_catalog_manifest,
)
from test_exact_pair_registry import PairRegistry
from test_exact_freshness import epoch_for_checkpoint

from exact_wave_selection import (
    ROOT,
    CATALOG_POINTER,
    SEALED_COHORT,
    MICRO_COHORT_SCHEMA,
    CORPUS,
    BACKLOG,
    PAIR_LEDGER,
    CHAMPIONS,
    CURRENT_HOF,
    ACCEPTED,
    PROPOSALS,
    BASE_CHECKPOINT,
    SHARD_SCHEMA,
    WAVE,
    TRANSFORMS,
    MICRO_LANE_SHAPE,
    LEGACY_COHORT_TASKS,
    checkpoint,
    next_wave_id,
    task_valid,
    normalized_body,
    mutate,
    load_known_pairs,
    load_parents,
    load_checkpoint_proposals,
    select_micro_cohort,
)
from exact_wave_planning import (
    RUST_WAVE_PLANNER,
    PROPOSALS_POINTER,
    micro_cohort_lane_sizes,
    _validate_pending_cohort,
    _rust_plan_micro_cohort,
    plan_micro_cohort,
)

def wave_seed(
    *,
    task: str,
    dataset: str,
    wave_id: str,
    local: list[dict],
    dl: list[dict],
    common: list[dict],
) -> list[dict]:
    if len(dl) < 64:
        raise RuntimeError(f"{task}:{dataset} has only {len(dl)} checkpoint proposals")
    parents = [*local, *dl, *common]
    for row in parents:
        row.setdefault("semantic_sha256", sha256_text(row["canonical_genome"]))
    offspring = []
    for attempt in range(20_000):
        child = mutate(
            parents,
            task=task,
            dataset_hash=dataset,
            wave_id=wave_id,
            attempt=attempt,
        )
        if child is not None and task_valid(child["canonical_genome"], task):
            offspring.append(child)
        if len(offspring) >= 512:
            break
    fills = [*offspring[64:], *dl[64:], *common[96:], *local[32:]]
    return compose_seed(
        {
            "local_elite": local,
            "dl_proposal": dl,
            "common_winner": common,
            "offspring": offspring,
        },
        fills,
    )


def build(
    candidates_per_dataset: int = 256,
    max_shard: int = 100,
    *,
    proposals_path: Path = PROPOSALS,
    cohort_path: Path | None = None,
) -> dict:
    if not 1 <= max_shard <= 100:
        raise RuntimeError("max shard size must be in 1..100")
    if candidates_per_dataset != 256:
        raise RuntimeError("strict waves require exactly 256 candidates per dataset")
    if not CORPUS.is_file() or not CURRENT_HOF.is_file() or not CHAMPIONS.is_file():
        raise RuntimeError("strict corpus and champion indexes must exist")
    wave_id = next_wave_id()
    wave_final = ROOT / "waves" / wave_id
    manifests_final = ROOT / "manifests" / wave_id
    if manifests_final.exists():
        raise RuntimeError(f"immutable wave already exists: {wave_id}")
    # -------------------------------------------------------------------
    # Claim the identity ATOMICALLY, up front, instead of only checking it.
    #
    # Two lanes allocate wave ids independently: this builder and the Rust
    # micro-wave supply lane. Checking `wave_final.exists()` here and renaming
    # onto it ~35 minutes later is a TOCTOU window the width of the whole build,
    # and it closed for real on 2026-07-28: `seed_next_strict_wave` attempt 899
    # ran for 36 minutes and died with
    #   OSError: [Errno 39] Directory not empty:
    #     '.../waves/.strict-wave-000172.tmp-3799533' -> '.../waves/strict-wave-000172'
    # because the supply lane published its own strict-wave-000172 (schema
    # JopeRealtimeSelfplayWave.v1, 1,408 manifests, a DIFFERENT cohort) at
    # 23:43:00 while this build was in its pair-registry phase.
    #
    # `mkdir(exist_ok=False)` is a single atomic syscall, and the final
    # `os.replace` below still works because rename(2) may replace an EMPTY
    # directory. The other lane is diverted because both Rust allocators key on
    # the directory NAME only (`jope_selfplay_wave.rs::next_wave_id` :1039-1066
    # via `parse_wave_id`, `jope_selfplay_supply.rs` via `valid_wave_id`).
    #
    # An orphaned claim (this process SIGKILLed mid-build, which happens here --
    # `waves/.strict-wave-000005.tmp-867990` has been orphaned since 2026-07-26)
    # leaves an empty wave directory. That is INERT to every consumer: both
    # `jope_selfplay_supply.rs::validate_ready_wave` (:686-688) and
    # `jope_selfplay_queue.rs` (:1800-1802) skip a wave directory with no
    # `summary.json`. It costs one wave id out of 999,999.
    # -------------------------------------------------------------------
    wave_final.parent.mkdir(parents=True, exist_ok=True)
    try:
        wave_final.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"immutable wave already exists: {wave_id}") from error
    wave_temporary = wave_final.with_name(f".{wave_id}.tmp-{os.getpid()}")
    manifests_temporary = manifests_final.with_name(
        f".{wave_id}.tmp-{os.getpid()}"
    )
    wave_temporary.mkdir(parents=True, exist_ok=False)
    manifests_temporary.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CURRENT_HOF, wave_temporary / "hall-of-fame.json")
    hof_hash = sha256_file(wave_temporary / "hall-of-fame.json")
    _, checkpoint_hash = checkpoint()
    proposals = load_checkpoint_proposals(proposals_path, checkpoint_hash)
    authority = validate_catalog_manifest(CATALOG_POINTER, verify_contexts=False)
    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog = json.loads(Path(pointer["catalog"]).read_text())
    if authority["row_count"] != len(catalog["entries"]):
        raise RuntimeError("catalog authority row count drift")
    known = load_known_pairs()
    local_parents, global_parents = load_parents()
    hof_pointer = json.loads((ROOT / "hof" / "current.json").read_text())
    seed_hof = json.loads(Path(hof_pointer["hall_of_fame"]).read_text())
    counts: dict[str, dict[str, int]] = {"regression": {}, "binary": {}}
    for (task, dataset), semantics in known.items():
        counts[task][dataset] = len(semantics)
    if cohort_path is not None:
        planned = json.loads(cohort_path.read_text())
        if (
            planned.get("schema") != MICRO_COHORT_SCHEMA
            or planned.get("wave") != wave_id
            or planned.get("checkpoint_sha256") != checkpoint_hash
            or planned.get("catalog_sha256") != pointer["catalog_sha256"]
            or len(planned.get("entries") or []) != 128
        ):
            raise RuntimeError("planned micro-cohort provenance drift")
        planned_rows = {
            (row["task"], row["dataset_hash"]): row
            for row in planned["entries"]
        }
        cohort = [
            {**row, **planned_rows[(row["task"], row["dataset_hash"])]}
            for row in catalog["entries"]
            if (row["task"], row["dataset_hash"]) in planned_rows
        ]
    else:
        cohort = select_micro_cohort(catalog["entries"], {
            (task, dataset): count
            for task, task_counts in counts.items()
            for dataset, count in task_counts.items()
        }, wave_id=wave_id)
    cohort_ids = {
        (row["task"], row["dataset_hash"]) for row in cohort
    }
    ordered = dataset_order(
        [
            row
            for row in catalog["entries"]
            if (row["task"], row["dataset_hash"]) in cohort_ids
        ],
        counts,
    )
    lanes = {
        (row["task"], row["dataset_hash"]): row["micro_lane"]
        for row in cohort
    }
    sys.path.insert(0, str(prime_source_root()))
    from jope.prime_label_to_corpus import GenomeEngineResolver

    resolver = GenomeEngineResolver()
    registry = PairRegistry()
    manifest_count = pair_count = deficient = 0
    reused_count = 0
    for ordinal, entry in enumerate(ordered):
        task, dataset = entry["task"], entry["dataset_hash"]
        local = [
            row
            for row in local_parents.get((task, dataset), [])
            if task_valid(row["canonical_genome"], task)
            and resolver.resolve(row["canonical_genome"], task_type=task) is not None
        ]
        common = [
            {
                **row,
                "source": "task_common_winner",
                "semantic_sha256": row["semantic_sha256"],
            }
            for row in seed_hof["common_winners"][task]
            if task_valid(row["canonical_genome"], task)
            and resolver.resolve(row["canonical_genome"], task_type=task) is not None
        ]
        if not local:
            local = [
                row
                for row in global_parents[task]
                if task_valid(row["canonical_genome"], task)
            ][:32]
        if not local:
            raise RuntimeError(f"{task}:{dataset} has no task-valid policy parent")
        selected = wave_seed(
            task=task,
            dataset=dataset,
            wave_id=wave_id,
            local=local,
            dl=[dict(row) for row in proposals.get((task, dataset), [])],
            common=common,
        )
        frozen_hash = sha256_text(
            json.dumps(
                {
                    "wave": wave_id,
                    "task": task,
                    "dataset_hash": dataset,
                    "semantics": [row["semantic_sha256"] for row in selected],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for row in selected:
            row.update(
                frozen_seed_sha256=frozen_hash,
                model_checkpoint_sha256=checkpoint_hash,
                hall_of_fame_sha256=hof_hash,
            )
        protocol = evaluation_protocol_sha256(
            task=task,
            metric="r2" if task == "regression" else "auc",
            score_tier="test_exact",
            train_sha256=entry["train_sha256"],
            test_sha256=entry["test_sha256"],
        )
        generated_at_ms = int(time.time() * 1000)
        owner = f"wave:{wave_id}"
        fresh = []
        reservation_outcomes = registry.reserve_batch(
            [
                {
                    "task": task,
                    "dataset_hash": dataset,
                    "semantic_sha256": row["semantic_sha256"],
                    "protocol_sha256": protocol,
                    "canonical_genome": row["canonical_genome"],
                }
                for row in selected
            ],
            owner=owner,
        )
        for row, created in zip(selected, reservation_outcomes, strict=True):
            if created:
                fresh.append(row)
            else:
                existing = registry.get(
                    task=task,
                    dataset_hash=dataset,
                    semantic_sha256=row["semantic_sha256"],
                    protocol_sha256=protocol,
                )
                if existing is not None and existing["reservation_owner"] == owner:
                    fresh.append(row)
                else:
                    reused_count += 1
        if not fresh:
            deficient += 1
        dataset_bindings = []
        for offset in range(0, len(fresh), max_shard):
            candidates = fresh[offset : offset + max_shard]
            manifest = {
                "schema": SHARD_SCHEMA,
                "strict_wave": wave_id,
                "micro_lane": lanes[(task, dataset)],
                "task": task,
                "dataset_hash": dataset,
                "checkpoint_epoch": epoch_for_checkpoint(checkpoint_hash),
                "checkpoint_sha256": checkpoint_hash,
                "catalog_sha256": pointer["catalog_sha256"],
                "protocol_sha256": protocol,
                "generated_at_ms": generated_at_ms,
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task=task,
                    wave=wave_id,
                    frozen_seed_sha256=frozen_hash,
                    model_checkpoint_sha256=checkpoint_hash,
                    hall_of_fame_sha256=hof_hash,
                ),
                "candidates": candidates,
            }
            name = (
                f"{manifest_count:06d}-{task[0]}-{dataset}-"
                f"{offset // max_shard:02d}.json"
            )
            atomic_json(manifests_temporary / name, manifest)
            manifest_path = manifests_temporary / name
            manifest_hash = sha256_file(manifest_path)
            dataset_bindings.append(
                {
                    "manifest_path": str(manifests_final / name),
                    "manifest_sha256": manifest_hash,
                    "pairs": [
                        {
                            "task": task,
                            "dataset_hash": dataset,
                            "semantic_sha256": row["semantic_sha256"],
                            "protocol_sha256": protocol,
                        }
                        for row in candidates
                    ],
                }
            )
            manifest_count += 1
            pair_count += len(candidates)
        registry.bind_manifests(
            owner=owner,
            manifests=dataset_bindings,
        )
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(ordered):
            print(
                json.dumps(
                    {
                        "wave": wave_id,
                        "datasets_complete": ordinal + 1,
                        "datasets_total": len(ordered),
                        "fresh_pairs": pair_count,
                        "manifests": manifest_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "schema": "JopeTestOnlyStrictWave.v1",
        "wave_kind": "deterministic_128_dataset_micro_wave",
        "wave": wave_id,
        "checkpoint_epoch": epoch_for_checkpoint(checkpoint_hash),
        "catalog_sha256": pointer["catalog_sha256"],
        "hall_of_fame_sha256": hof_hash,
        "model_checkpoint_sha256": checkpoint_hash,
        "datasets": len(ordered),
        "task_datasets": {
            task: sum(row["task"] == task for row in ordered)
            for task in sorted({row["task"] for row in ordered})
        },
        "micro_cohort": [
            {
                "task": row["task"],
                "dataset_hash": row["dataset_hash"],
                "lane": row["micro_lane"],
            }
            for row in cohort
        ],
        "candidates_per_dataset": candidates_per_dataset,
        "fresh_pairs": pair_count,
        "known_protocol_pairs_reused": reused_count,
        "seed_pairs": len(ordered) * 256,
        "manifests": manifest_count,
        "deficient_datasets": deficient,
        "max_candidates_per_manifest": max_shard,
    }
    atomic_json(wave_temporary / "summary.json", summary)
    wave_final.parent.mkdir(parents=True, exist_ok=True)
    manifests_final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(wave_temporary, wave_final)
    os.replace(manifests_temporary, manifests_final)
    registry.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-per-dataset", type=int, default=256)
    parser.add_argument("--max-shard", type=int, default=100)
    parser.add_argument("--proposals", type=Path, default=PROPOSALS)
    parser.add_argument("--cohort", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.candidates_per_dataset,
                args.max_shard,
                proposals_path=args.proposals,
                cohort_path=args.cohort,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
