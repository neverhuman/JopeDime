#!/usr/bin/env python3
"""Generate proposals for every frozen dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import torch

if __package__:
    from .test_exact_dl_generate_args import parse_args
else:
    from test_exact_dl_generate_args import parse_args

if __package__:
    from .test_exact_dl_generate_support import (
        DEFAULT_CATALOG,
        DEFAULT_CHECKPOINT,
        DEFAULT_OUTPUT,
        PROPOSAL_SELECTION_POLICY_SCHEMA,
        STRIDE_FLOATS,
        decode_batch,
        deterministic_normal,
        filter_exact_dataset_manifest,
        load_proposal_selection_policy,
        rank_scored_proposals,
        score_context_candidates,
        sha256_file,
    )
else:
    from test_exact_dl_generate_support import (
        DEFAULT_CATALOG,
        DEFAULT_CHECKPOINT,
        DEFAULT_OUTPUT,
        PROPOSAL_SELECTION_POLICY_SCHEMA,
        STRIDE_FLOATS,
        decode_batch,
        deterministic_normal,
        filter_exact_dataset_manifest,
        load_proposal_selection_policy,
        rank_scored_proposals,
        score_context_candidates,
        sha256_file,
    )

def main() -> None:
    args = parse_args()
    from jope.policy.decode import sample_sequences_batched
    from jope.propose import load_any_model
    from jope.selector_calibration import load_selector_calibration

    pointer = json.loads(args.catalog_pointer.read_text())
    catalog_path = args.catalog or Path(pointer["catalog"])
    data_path = args.data or Path(pointer["data"])
    manifest = json.loads(catalog_path.read_text())
    blob = data_path.read_bytes()
    if sha256_file(data_path) != pointer["data_sha256"]:
        raise RuntimeError("context blob does not match its frozen pointer")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.checkpoint.stem.split(".")[0]:
        raise RuntimeError("accepted checkpoint filename/hash mismatch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation")
    model = load_any_model(args.checkpoint, map_location="cpu").to(args.device)
    model.eval()
    selector_calibration = load_selector_calibration(args.selector_calibration)
    selection_policy, selection_policy_sha256 = load_proposal_selection_policy(
        args.selection_policy
    )
    if selector_calibration is not None:
        raise RuntimeError(
            "learned selector calibration is disabled by JopeProposalSelectionPolicy.v1"
        )
    embedding_sd = None
    embedding_sd_hash = None
    if args.embedding_sd is not None:
        sd_document = json.loads(args.embedding_sd.read_text())
        if sd_document.get("schema") != "TrainOnlyEmbeddingScalingStatistic.v1":
            raise RuntimeError("unsupported embedding scaling-statistic schema")
        embedding_sd = sd_document.get("population_sd")
        if (
            not isinstance(embedding_sd, list)
            or len(embedding_sd) != 1536
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in embedding_sd
            )
        ):
            raise RuntimeError("embedding scaling statistic is invalid")
        embedding_sd_hash = sha256_file(args.embedding_sd)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    # Accounting maps carry ALL FOUR tasks even while sampling only iterates
    # regression/binary. The canary path feeds this script entries derived from
    # the LIVE catalog, and the live catalog now contains multiclass and
    # timeseries rows -- with two-key maps, the first such entry KeyError'd at
    # the histogram update and permanently wedged the canary executor (job id =
    # intent_hash + implementation_sha, so the crash-looping job never rotated).
    ALL_TASKS = ("regression", "binary", "multiclass", "timeseries")
    counts = {task: 0 for task in ALL_TASKS}
    validity = {"decoded": 0, "kept": 0}
    # Native (model-sampled) unique count for every context that fell short of
    # the budget, keyed "<task>:<dataset_hash>". Always recorded now -- it is the
    # measurement that exposes head/mode collapse, so it must not be gated behind
    # the tolerance flag.
    deficient_contexts: dict[str, int] = {}
    # Terminal-head histogram over MODEL-SAMPLED rows only (backfilled rows are
    # copies of another context's model-sampled rows, so counting them would
    # dilute exactly the signal this measures). A task whose entire histogram is
    # one head has mode-collapsed; jope_selfplay_alarms reads this field.
    native_head_counts: dict[str, dict[str, int]] = {task: {} for task in ALL_TASKS}
    native_unique_total = {task: 0 for task in ALL_TASKS}
    context_totals = {task: 0 for task in ALL_TASKS}
    with temporary.open("w", encoding="utf-8") as handle:
        entries = [
            entry
            for entry in manifest["entries"]
            if (args.split is None or entry.get("split") == args.split)
            and (args.task is None or entry["task"] == args.task)
        ]
        if args.task is not None and not entries:
            raise RuntimeError(f"catalog has no {args.task} entries to propose for")
        cohort_hash = None
        dataset_manifest_hash = None
        if args.dataset_manifest is not None:
            entries = filter_exact_dataset_manifest(
                entries,
                args.dataset_manifest,
                catalog_sha256=pointer["catalog_sha256"],
            )
            dataset_manifest_hash = sha256_file(args.dataset_manifest)
        if args.cohort is not None:
            cohort = json.loads(args.cohort.read_text())
            if (
                cohort.get("checkpoint_sha256") != checkpoint_hash
                or cohort.get("catalog_sha256") != pointer["catalog_sha256"]
            ):
                raise RuntimeError("proposal micro-cohort provenance drift")
            identities = {
                (row["task"], row["dataset_hash"])
                for row in cohort.get("entries") or []
            }
            # The wave contract is tasks.len() * 64 (micro_wave_size in
            # selfplay_data.rs): 128, 192, or 256 datasets. A cohort without a
            # declared task list predates the field and is two-task by
            # definition -- the same default CohortPlan::tasks deserializes to.
            cohort_tasks = cohort.get("tasks") or ["regression", "binary"]
            expected = 64 * len(cohort_tasks)
            if len(identities) != expected:
                raise RuntimeError(
                    f"proposal micro-cohort must contain {expected} datasets "
                    f"for tasks {cohort_tasks}, found {len(identities)}"
                )
            entries = [
                entry
                for entry in entries
                if (entry["task"], entry["dataset_hash"]) in identities
            ]
            if len(entries) != expected:
                raise RuntimeError("proposal micro-cohort is absent from the catalog")
            cohort_hash = sha256_file(args.cohort)
        for start in range(0, len(entries), args.batch_contexts):
            batch = entries[start : start + args.batch_contexts]
            embeddings, targets = [], []
            for entry in batch:
                offset = int(entry["catalog_index"]) * STRIDE_FLOATS * 4
                values = struct.unpack(
                    f"<{STRIDE_FLOATS}f",
                    blob[offset : offset + STRIDE_FLOATS * 4],
                )
                embeddings.append(values[:1536])
                targets.append(values[1536:])
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            if embedding_sd is not None:
                jittered = []
                for entry, embedding in zip(batch, embeddings):
                    jittered.append(
                        [
                            float(value)
                            + args.embedding_jitter_scale
                            * float(embedding_sd[coordinate])
                            * deterministic_normal(
                                args.perturbation_seed,
                                entry["dataset_hash"],
                                coordinate,
                            )
                            for coordinate, value in enumerate(embedding)
                        ]
                    )
                embedding_tensor = torch.tensor(jittered, dtype=torch.float32)
            by_context: list[list[dict]] = [[] for _ in batch]
            # Iterate the tasks PRESENT, not a hard-coded pair: a multiclass or
            # timeseries context must be sampled with its own task conditioning,
            # never silently skipped (skipping left the context empty and the
            # wave builder counted it as a proposal deficiency).
            for task in sorted({entry["task"] for entry in batch}):
                indices = [index for index, entry in enumerate(batch) if entry["task"] == task]
                if not indices:
                    continue
                emb = embedding_tensor[indices]
                tgt = target_tensor[indices]
                greedy = sample_sequences_batched(
                    model,
                    emb,
                    tgt,
                    n=1,
                    engine="featmath",
                    task=task,
                    temperature=0.0,
                    seed=args.seed + start,
                    device=args.device,
                )
                samples = sample_sequences_batched(
                    model,
                    emb,
                    tgt,
                    n=max(96, args.proposals_per_dataset * 4),
                    engine="featmath",
                    task=task,
                    temperature=0.7,
                    top_p=0.9,
                    seed=args.seed + start,
                    device=args.device,
                )
                for relative, batch_index in enumerate(indices):
                    rows = decode_batch(
                        model, greedy[relative], "greedy", args.seed + start,
                        task, validity,
                    )
                    rows.extend(
                        decode_batch(
                            model, samples[relative], "nucleus", args.seed + start,
                            task, validity,
                        )
                    )
                    by_context[batch_index].extend(rows)
            for refill_round in range(1, 2):
                deficient = [
                    index
                    for index, rows in enumerate(by_context)
                    if len({row["semantic_sha256"] for row in rows})
                    < args.proposals_per_dataset
                ]
                if not deficient:
                    break
                for task in sorted({batch[index]["task"] for index in deficient}):
                    indices = [
                        index
                        for index in deficient
                        if batch[index]["task"] == task
                    ]
                    if not indices:
                        continue
                    extra = sample_sequences_batched(
                        model,
                        embedding_tensor[indices],
                        target_tensor[indices],
                        n=max(1, args.proposals_per_dataset * 4),
                        engine="featmath",
                        task=task,
                        temperature=min(1.35, 0.7 + refill_round * 0.08),
                        top_p=min(0.995, 0.9 + refill_round * 0.015),
                        seed=args.seed + start + refill_round * 1_000_003,
                        device=args.device,
                    )
                    for relative, batch_index in enumerate(indices):
                        by_context[batch_index].extend(
                            decode_batch(
                                model,
                                extra[relative],
                                "nucleus_shortage_fill",
                                args.seed + start + refill_round * 1_000_003,
                                task,
                                validity,
                            )
                        )
            for batch_index, (entry, proposals) in enumerate(zip(batch, by_context)):
                unique = {}
                for proposal in proposals:
                    unique.setdefault(proposal["semantic_sha256"], proposal)
                scored = score_context_candidates(
                    model,
                    embedding_tensor[batch_index],
                    target_tensor[batch_index],
                    list(unique.values()),
                    task=entry["task"],
                    selection_policy=selection_policy,
                    device=args.device,
                    selector_calibration=selector_calibration,
                )
                selected = rank_scored_proposals(
                    scored, args.proposals_per_dataset
                )
                # A thin context returns its unused allocation to GP. Borrowing
                # a proposal from another context would destroy the causal
                # attribution this generator exists to measure.
                if len(selected) < args.proposals_per_dataset:
                    deficient_contexts[
                        f"{entry['task']}:{entry['dataset_hash']}"
                    ] = len(selected)
                offset = int(entry["catalog_index"]) * STRIDE_FLOATS * 4
                context_sha256 = hashlib.sha256(
                    blob[offset : offset + STRIDE_FLOATS * 4]
                ).hexdigest()
                target_summary_bytes = blob[
                    offset + 1536 * 4 : offset + STRIDE_FLOATS * 4
                ]
                proposal_context_bytes = struct.pack(
                    f"<{STRIDE_FLOATS}f",
                    *embedding_tensor[batch_index].tolist(),
                    *targets[batch_index],
                )
                if proposal_context_bytes[1536 * 4 :] != target_summary_bytes:
                    raise RuntimeError("target-summary bytes changed during proposal generation")
                if (
                    args.proposal_origin == "exact_context"
                    and proposal_context_bytes
                    != blob[offset : offset + STRIDE_FLOATS * 4]
                ):
                    raise RuntimeError("exact proposal context changed during generation")
                for slot, proposal in enumerate(selected):
                    head_op = proposal.pop("_head_op")
                    proposal.pop("_actions")
                    redteam_fields = {}
                    if args.redteam_metadata:
                        redteam_fields = {
                            "exact_context_sha256": context_sha256,
                            "proposal_context_sha256": hashlib.sha256(
                                proposal_context_bytes
                            ).hexdigest(),
                            "target_summary_sha256": hashlib.sha256(
                                target_summary_bytes
                            ).hexdigest(),
                            "target_summary_unchanged": True,
                            "proposal_origin": args.proposal_origin,
                            "invocation_seed": args.seed,
                            "perturbation_seed": args.perturbation_seed,
                            "embedding_jitter_scale": args.embedding_jitter_scale,
                            "embedding_scaling_statistic_sha256": embedding_sd_hash,
                        }
                    row = {
                        "schema": "DlProposalSetRow.v2",
                        "task": entry["task"],
                        "dataset_hash": entry["dataset_hash"],
                        "context_sha256": context_sha256,
                        "catalog_sha256": pointer["catalog_sha256"],
                        "cohort_sha256": cohort_hash,
                        "slot": slot,
                        "model_checkpoint_sha256": checkpoint_hash,
                        "origin": "dl_direct",
                        "native_origin": True,
                        "retargeted": False,
                        "reward_rank": slot,
                        "head_type": head_op,
                        **redteam_fields,
                        **proposal,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    histogram = native_head_counts[entry["task"]]
                    histogram[head_op] = histogram.get(head_op, 0) + 1
                counts[entry["task"]] += len(selected)
                context_totals[entry["task"]] += 1
                native_unique_total[entry["task"]] += len(selected)
            handle.flush()
            print(
                json.dumps(
                    {
                        "datasets_complete": min(start + len(batch), len(entries)),
                        "datasets_total": len(entries),
                        "proposals": sum(counts.values()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        handle.flush()
    os.replace(temporary, args.output)
    # A thin context is visible and survivable: it emits fewer direct rows and
    # the wave builder transfers the unused slots to GP.
    deficiency_by_task = {}
    for task in sorted(context_totals):
        total = context_totals[task]
        if not total:
            continue
        short = {
            key: value
            for key, value in deficient_contexts.items()
            if key.startswith(f"{task}:")
        }
        deficiency_by_task[task] = {
            "contexts": total,
            "deficient_contexts": len(short),
            "deficient_fraction": len(short) / total,
            "min_native_unique": (
                min(short.values()) if short else args.proposals_per_dataset
            ),
            "mean_native_unique": native_unique_total[task] / total,
            "backfilled_rows": 0,
            "backfilled_fraction": 0.0,
            "unfilled_contexts": len(short),
        }
    summary = {
        "schema": "DlProposalSet.v2",
        "catalog_sha256": pointer["catalog_sha256"],
        "model_checkpoint_sha256": checkpoint_hash,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "counts": counts,
        "deficient_contexts": deficient_contexts,
        "deficiency_by_task": deficiency_by_task,
        "backfilled_contexts": {},
        "backfilled_rows": {"regression": 0, "binary": 0},
        "cross_context_backfill_rows": 0,
        "zero_cross_context_backfill": True,
        "unfilled_contexts": deficient_contexts,
        "native_head_counts_by_task": native_head_counts,
        "proposal_validity": (
            validity["kept"] / validity["decoded"] if validity["decoded"] else 0.0
        ),
        # The blended figure hides a task collapse whenever one task is a small
        # share of the catalog: a checkpoint that has lost its classifier head
        # can still blend above the promotion gate's 0.99 if binary contexts are
        # rare. Report the per-task numbers so the gate (and the seeding audit)
        # can look at the weakest task rather than the average.
        "proposal_validity_by_task": {
            task: (
                validity[f"{task}:kept"] / validity[f"{task}:decoded"]
                if validity.get(f"{task}:decoded")
                else None
            )
            for task in ("regression", "binary")
            if validity.get(f"{task}:decoded")
        },
        "proposal_validity_counts": dict(validity),
        "datasets": len(entries),
        "proposals_per_dataset": args.proposals_per_dataset,
        "split": args.split,
        # None = both branches (the historical whole-catalog set). A named task
        # means this file is a per-branch set generated from that branch's own
        # accepted checkpoint; the installer keys the published name on it.
        "task": args.task,
        "cohort_sha256": cohort_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "seed": args.seed,
        "batch_contexts": args.batch_contexts,
        "selection_policy": PROPOSAL_SELECTION_POLICY_SCHEMA,
        "selection_policy_sha256": selection_policy_sha256,
        "native_engine": "featmath",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "p95_proposal_latency_ms": (
            (time.perf_counter() - started) * 1000.0 / max(1, len(entries))
        ),
    }
    if args.redteam_metadata:
        summary.update(
            {
                "proposal_origin": args.proposal_origin,
                "perturbation_seed": args.perturbation_seed,
                "embedding_jitter_scale": args.embedding_jitter_scale,
                "embedding_scaling_statistic_sha256": embedding_sd_hash,
                "target_summaries_unchanged": True,
            }
        )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
