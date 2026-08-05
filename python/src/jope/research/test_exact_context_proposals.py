#!/usr/bin/env python3
"""Generate and cache proposal reservoirs for an explicit set of contexts.

The cache identity deliberately excludes the catalog pointer. A catalog append
can therefore reuse every artifact whose context bytes, accepted checkpoint,
seed, and decoding policy are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import torch

from test_exact_common import sha256_file, sha256_text
from test_exact_dl_generate import (
    STRIDE_FLOATS,
    decode_batch,
    head_valid_for_task,
    load_proposal_selection_policy,
    rank_scored_proposals,
    score_context_candidates,
)

SCHEMA = "ContextProposalArtifact.v3"
REQUEST_SCHEMA = "ContextProposalRequest.v4"
SUPPORTED_REQUEST_SCHEMAS = {
    "ContextProposalRequest.v3",
    REQUEST_SCHEMA,
}
RUN_SCHEMA = "ContextProposalRun.v3"
SLOT_LOG_SCHEMA = "ContextProposalSlotLog.v1"
RESERVOIRS = (64, 128, 256)
SLOTS = 8
REJECTION_REASONS = (
    "invalid",
    "duplicate",
    "unavailable",
    "compilation",
    "ancestry",
    "diversity",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a lowercase sha256 digest")
    return value


def calibration_qualifies(value: dict | None) -> bool:
    if not value:
        return False
    correlation = value.get("realized_error_correlation_lower_bound_95")
    coverage = value.get("interval_coverage")
    target = value.get("target_coverage")
    return (
        isinstance(correlation, (int, float))
        and math.isfinite(float(correlation))
        and float(correlation) >= 0.0
        and isinstance(coverage, (int, float))
        and math.isfinite(float(coverage))
        and isinstance(target, (int, float))
        and math.isfinite(float(target))
        and abs(float(coverage) - float(target)) <= 0.05
    )


def decoding_policy(
    calibration_evidence: dict | None,
    calibration_evidence_sha256: str | None,
    selector_calibration_sha256: str | None,
    proposal_selection_policy: dict,
    proposal_selection_policy_sha256: str,
) -> dict:
    learned_risk_enabled = (
        proposal_selection_policy["learned_risk"]["enabled"]
        and calibration_qualifies(calibration_evidence)
    )
    return {
        "schema": "ContextDecodingPolicy.v4",
        "reservoirs": list(RESERVOIRS),
        "slots_per_context": SLOTS,
        "engine": "featmath",
        "temperature": 0.7,
        "top_p": 0.9,
        "learned_uncertainty_and_risk_enabled": learned_risk_enabled,
        "uncertainty_fallback": (
            None if learned_risk_enabled else "empirical_reward_mad"
        ),
        "calibration_evidence_sha256": calibration_evidence_sha256,
        "selector_calibration_sha256": (
            selector_calibration_sha256 if learned_risk_enabled else None
        ),
        "selection": "reward-stability-risk-head-diversity-v4",
        "proposal_selection_policy_sha256": proposal_selection_policy_sha256,
        "validity_repair": "dataset-local-deterministic-featmath-grammar-v1",
        "validity_repair_only_for_missing_slots": True,
    }


def context_seed(root_seed: int, context_sha256: str) -> int:
    payload = f"{root_seed}:{context_sha256}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def cache_path(cache_root: Path, key: dict) -> Path:
    return (
        cache_root
        / key["task"]
        / key["accepted_checkpoint_sha256"]
        / key["context_sha256"]
        / str(key["seed"])
        / key["decoding_policy_sha256"]
        / f"{key['reservoir_size']}.v3.json"
    )


def slot_log_path(cache_root: Path, key: dict) -> Path:
    return (
        cache_root
        / "_slot_logs"
        / key["task"]
        / key["accepted_checkpoint_sha256"]
        / key["context_sha256"]
        / str(key["seed"])
        / key["decoding_policy_sha256"]
        / f"{key['reservoir_size']}.requested.json"
    )


def atomic_create_json(path: Path, value: dict) -> bool:
    """Create an immutable JSON object, returning false if it already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)


def write_slot_log(cache_root: Path, key: dict) -> Path:
    path = slot_log_path(cache_root, key)
    value = {
        "schema": SLOT_LOG_SCHEMA,
        "key": key,
        "logged_at_ms": time.time_ns() // 1_000_000,
        "slots": [
            {"slot": slot, "status": "pending_decode"} for slot in range(SLOTS)
        ],
    }
    atomic_create_json(path, value)
    return path


def empty_rejections() -> dict[str, int]:
    return {reason: 0 for reason in REJECTION_REASONS}


def validate_artifact(value: dict, expected_key: dict | None = None) -> dict:
    if value.get("schema") != SCHEMA:
        raise RuntimeError("unsupported context proposal artifact")
    key = value.get("key")
    if not isinstance(key, dict) or (expected_key is not None and key != expected_key):
        raise RuntimeError("context proposal cache key drift")
    if key.get("reservoir_size") not in RESERVOIRS:
        raise RuntimeError("invalid context proposal reservoir")
    rejections = value.get("rejection_counts")
    if not isinstance(rejections, dict) or set(rejections) != set(REJECTION_REASONS):
        raise RuntimeError("context proposal rejection accounting is incomplete")
    slots = value.get("slots")
    proposals = value.get("proposals")
    if (
        not isinstance(slots, list)
        or len(slots) != SLOTS
        or not isinstance(proposals, list)
    ):
        raise RuntimeError("context proposal artifact must contain eight slots")
    ready = []
    for index, slot in enumerate(slots):
        if slot.get("slot") != index:
            raise RuntimeError("context proposal slots are not dense")
        status = slot.get("status")
        if status == "ready":
            ready.append(slot)
            require_sha256(slot.get("candidate_sha256"), "slot candidate")
        elif status == "pending_retry":
            if key["reservoir_size"] >= 256 or slot.get("candidate_sha256") is not None:
                raise RuntimeError("invalid pending proposal slot")
        elif status == "permanent_failure":
            if key["reservoir_size"] != 256 or slot.get("candidate_sha256") is not None:
                raise RuntimeError("proposal slot failed permanently before reservoir 256")
        else:
            raise RuntimeError("invalid proposal slot status")
    if len(ready) != len(proposals):
        raise RuntimeError("ready slots and cached proposals differ")
    for index, (slot, proposal) in enumerate(zip(ready, proposals)):
        if (
            proposal.get("slot") != index
            or proposal.get("candidate_sha256") != slot.get("candidate_sha256")
        ):
            raise RuntimeError("cached proposal payload does not match slot")
    return value


def load_artifact(path: Path, key: dict) -> dict:
    return validate_artifact(json.loads(path.read_text()), key)


def empirical_dispersion_rank(proposals: list[dict]) -> list[dict]:
    if not proposals:
        return proposals
    rewards = sorted(float(row["reward_mean"]) for row in proposals)
    middle = len(rewards) // 2
    median = (
        rewards[middle]
        if len(rewards) % 2
        else (rewards[middle - 1] + rewards[middle]) / 2.0
    )
    deviations = sorted(abs(value - median) for value in rewards)
    middle = len(deviations) // 2
    mad = (
        deviations[middle]
        if len(deviations) % 2
        else (deviations[middle - 1] + deviations[middle]) / 2.0
    )
    uncertainty = 1.4826 * mad
    for row in proposals:
        reward = float(row["reward_mean"])
        row["reward_uncertainty"] = uncertainty
        row["reward_lcb"] = reward - 0.5 * uncertainty
        # Until calibration opens, neither learned uncertainty nor learned
        # instability/risk affects ordering.
        row["selection_score"] = row["reward_lcb"]
        row["proposal_score"] = row["selection_score"]
    return proposals


def proposal_artifact(
    key: dict,
    decoded_candidates: int,
    raw_rows: list[dict],
    selected: list[dict],
) -> dict:
    unique_count = len({row["semantic_sha256"] for row in raw_rows})
    rejections = empty_rejections()
    rejections["invalid"] = max(0, decoded_candidates - len(raw_rows))
    rejections["duplicate"] = max(0, len(raw_rows) - unique_count)
    rejections["diversity"] = max(0, unique_count - len(selected))
    missing = max(0, SLOTS - len(selected))
    rejections["unavailable"] = missing
    final = key["reservoir_size"] == RESERVOIRS[-1]
    slots = []
    proposals = []
    for slot in range(SLOTS):
        if slot < len(selected):
            row = selected[slot]
            candidate = require_sha256(
                row["semantic_sha256"], "proposal semantic identity"
            )
            slots.append(
                {
                    "slot": slot,
                    "status": "ready",
                    "candidate_sha256": candidate,
                    "rejections": {},
                }
            )
            proposals.append(
                {
                    "slot": slot,
                    "candidate_sha256": candidate,
                    "canonical_genome": row["canonical_genome"],
                    "decoder": row["decoder"],
                    "decoder_seed": int(row["decoder_seed"]),
                    "tokens": [int(token) for token in row["tokens"]],
                    "head_type": row["_head_op"],
                    "reward_mean": float(row["reward_mean"]),
                    "reward_uncertainty": float(row["reward_uncertainty"]),
                    "reward_lcb": float(row["reward_lcb"]),
                    "instability_probability": float(
                        row["instability_probability"]
                    ),
                    "selection_score": float(row["selection_score"]),
                    "candidate_risk_reason": row.get("candidate_risk_reason"),
                    "source": row.get("source", "accepted_dl_checkpoint"),
                }
            )
        else:
            slots.append(
                {
                    "slot": slot,
                    "status": "permanent_failure" if final else "pending_retry",
                    "candidate_sha256": None,
                    "rejections": dict(rejections),
                }
            )
    artifact = {
        "schema": SCHEMA,
        "key": key,
        "generated_at_ms": time.time_ns() // 1_000_000,
        "decoded_candidates": decoded_candidates,
        "rejection_counts": rejections,
        "slots": slots,
        "proposals": proposals,
    }
    return validate_artifact(artifact, key)


def context_values(data_handle, catalog_index: int) -> tuple[bytes, tuple[float, ...]]:
    offset = catalog_index * STRIDE_FLOATS * 4
    data_handle.seek(offset)
    context = data_handle.read(STRIDE_FLOATS * 4)
    if len(context) != STRIDE_FLOATS * 4:
        raise RuntimeError("context blob ended before requested catalog index")
    return context, struct.unpack(f"<{STRIDE_FLOATS}f", context)


def request_entries(request: dict, catalog: dict, data_path: Path) -> list[dict]:
    if request.get("schema") not in SUPPORTED_REQUEST_SCHEMAS:
        raise RuntimeError("unsupported context proposal request")
    task = request.get("task")
    # All four task families plus the "mixed" cohort request. This gate crashed
    # wave 546 seventeen times: the Rust planner produced a three-task cohort
    # while this validator still named two tasks, and the whole cluster idled
    # behind the failed proposer. The task list must track the wave contract.
    if task not in ("binary", "regression", "multiclass", "timeseries", "mixed"):
        raise RuntimeError("context proposal request has an invalid task")
    requested = request.get("requested_contexts", request.get("contexts"))
    if not isinstance(requested, list) or not requested:
        raise RuntimeError("context proposal request is empty")
    if len(
        {
            (
                row.get("dataset_hash"),
                row.get("context_sha256"),
            )
            for row in requested
        }
    ) != len(requested):
        raise RuntimeError("context proposal request contains duplicate contexts")
    by_identity = {
        (row["task"], row["dataset_hash"], row.get("context_sha256")): row
        for row in catalog.get("entries", [])
    }
    selected = []
    with data_path.open("rb") as data_handle:
        for requested_row in requested:
            row_task = requested_row.get("task", task)
            if row_task not in ("binary", "regression", "multiclass", "timeseries"):
                raise RuntimeError("requested context has an invalid task")
            if task != "mixed" and row_task != task:
                raise RuntimeError("requested context crosses the request task")
            context_sha256 = require_sha256(
                requested_row.get("context_sha256"), "requested context"
            )
            identity = (
                row_task,
                requested_row.get("dataset_hash"),
                context_sha256,
            )
            entry = by_identity.get(identity)
            if entry is None:
                raise RuntimeError(f"requested context is absent from catalog: {identity}")
            context, _ = context_values(data_handle, int(entry["catalog_index"]))
            if hashlib.sha256(context).hexdigest() != context_sha256:
                raise RuntimeError("requested context bytes do not match their identity")
            selected.append(entry)
    return selected


def generate_reservoir(
    model,
    embedding: torch.Tensor,
    target: torch.Tensor,
    task: str,
    seed: int,
    reservoir: int,
    device: str,
    proposal_selection_policy: dict,
    learned_risk_enabled: bool,
    selector_calibration,
) -> tuple[list[dict], list[dict]]:
    from jope.policy.decode import sample_sequences_batched

    validity: dict[str, int] = {"decoded": 0, "kept": 0}
    greedy = sample_sequences_batched(
        model,
        embedding.reshape(1, -1),
        target.reshape(1, -1),
        n=1,
        engine="featmath",
        task=task,
        temperature=0.0,
        seed=seed,
        device=device,
    )[0]
    samples = sample_sequences_batched(
        model,
        embedding.reshape(1, -1),
        target.reshape(1, -1),
        n=reservoir - 1,
        engine="featmath",
        task=task,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
        device=device,
    )[0]
    raw = decode_batch(model, greedy, "greedy", seed, task, validity)
    raw.extend(decode_batch(model, samples, "nucleus", seed, task, validity))
    unique = {}
    for row in raw:
        unique.setdefault(row["semantic_sha256"], row)
    scored = score_context_candidates(
        model,
        embedding,
        target,
        list(unique.values()),
        task=task,
        selection_policy=proposal_selection_policy,
        device=device,
        selector_calibration=(
            selector_calibration if learned_risk_enabled else None
        ),
    )
    if not learned_risk_enabled:
        empirical_dispersion_rank(scored)
    selected = rank_scored_proposals(scored, SLOTS)
    if len(selected) < SLOTS:
        repairs = grammar_validity_repairs(
            model,
            task=task,
            seed=seed ^ (reservoir << 32),
            requested=max(32, (SLOTS - len(selected)) * 8),
        )
        scored_repairs = score_context_candidates(
            model,
            embedding,
            target,
            repairs,
            task=task,
            selection_policy=proposal_selection_policy,
            device=device,
            selector_calibration=(
                selector_calibration if learned_risk_enabled else None
            ),
        )
        if not learned_risk_enabled:
            empirical_dispersion_rank(scored_repairs)
        used = {row["semantic_sha256"] for row in selected}
        for repair in rank_scored_proposals(scored_repairs, len(scored_repairs)):
            if repair["semantic_sha256"] in used:
                continue
            selected.append(repair)
            used.add(repair["semantic_sha256"])
            if len(selected) == SLOTS:
                break
        raw.extend(repairs)
    return raw, selected


def grammar_validity_repairs(
    model,
    *,
    task: str,
    seed: int,
    requested: int,
) -> list[dict]:
    """Create deterministic, task-valid candidates only for missing model slots."""

    from jope.grammars import grammar_for
    from jope.grammars.sampler import sample_genome_from_spec
    from jope.pipeline_text import format_pipeline_actions, parse_pipeline_actions
    from jope.policy.tokens import encode_actions

    spec = grammar_for("featmath")
    rng = random.Random(seed)
    rows = []
    semantics = set()
    for _ in range(max(256, requested * 64)):
        genome = sample_genome_from_spec(
            spec,
            rng,
            max_transforms=6,
            expr_fraction=0.1,
            max_cost_class="expensive",
            task=task,
        )
        try:
            actions = parse_pipeline_actions(genome, spec)
            if not head_valid_for_task(actions["head"]["op"], task):
                continue
            canonical = format_pipeline_actions(actions)
            tokens = encode_actions(
                "featmath",
                actions,
                model.vocab,
                task=task,
            )
        except (KeyError, ValueError):
            continue
        semantic = sha256_text(canonical)
        if semantic in semantics:
            continue
        semantics.add(semantic)
        rows.append(
            {
                "task": task,
                "canonical_genome": canonical,
                "semantic_sha256": semantic,
                "source": "grammar_validity_repair",
                "decoder": "grammar_validity_repair",
                "decoder_seed": seed,
                "tokens": tokens,
                "_head_op": actions["head"]["op"],
                "_actions": actions,
            }
        )
        if len(rows) == requested:
            break
    if len(rows) < requested:
        raise RuntimeError(
            f"validity repair produced {len(rows)} of {requested} task-valid candidates"
        )
    return rows


def aggregate_row(
    proposal: dict,
    *,
    task: str,
    dataset_hash: str,
    context_sha256: str,
    catalog_sha256: str,
    checkpoint_sha256: str,
) -> dict:
    source = proposal.get("source", "accepted_dl_checkpoint")
    native_origin = source == "accepted_dl_checkpoint"
    return {
        "schema": "DlProposalSetRow.v2",
        "task": task,
        "dataset_hash": dataset_hash,
        "context_sha256": context_sha256,
        "catalog_sha256": catalog_sha256,
        "cohort_sha256": None,
        "slot": proposal["slot"],
        "model_checkpoint_sha256": checkpoint_sha256,
        "origin": "dl_direct" if native_origin else "grammar_validity_repair",
        "native_origin": native_origin,
        "retargeted": False,
        "reward_rank": proposal["slot"],
        "head_type": proposal["head_type"],
        "canonical_genome": proposal["canonical_genome"],
        "semantic_sha256": proposal["candidate_sha256"],
        "source": source,
        "decoder": proposal["decoder"],
        "decoder_seed": proposal["decoder_seed"],
        "tokens": proposal["tokens"],
        "reward_mean": proposal["reward_mean"],
        "reward_uncertainty": proposal["reward_uncertainty"],
        "reward_lcb": proposal["reward_lcb"],
        "instability_probability": proposal["instability_probability"],
        "candidate_risk_reason": proposal.get("candidate_risk_reason"),
        "selection_score": proposal["selection_score"],
        "proposal_score": proposal["selection_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--catalog-pointer", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path)
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument("--selection-policy", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from jope.propose import load_any_model
    from jope.selector_calibration import load_selector_calibration

    started = time.perf_counter()
    request = json.loads(args.request.read_text())
    pointer = json.loads(args.catalog_pointer.read_text())
    catalog_path = args.catalog or Path(pointer["catalog"])
    data_path = args.data or Path(pointer["data"])
    if sha256_file(catalog_path) != pointer["catalog_sha256"]:
        raise RuntimeError("catalog does not match its pointer")
    if sha256_file(data_path) != pointer["data_sha256"]:
        raise RuntimeError("context blob does not match its pointer")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        require_sha256(
            request.get("accepted_checkpoint_sha256"), "accepted checkpoint"
        )
        != checkpoint_sha256
    ):
        raise RuntimeError("request is not pinned to the supplied checkpoint")
    calibration_evidence = (
        json.loads(args.calibration_evidence.read_text())
        if args.calibration_evidence
        else None
    )
    calibration_evidence_sha256 = (
        sha256_file(args.calibration_evidence) if args.calibration_evidence else None
    )
    selector_sha256 = (
        sha256_file(args.selector_calibration)
        if args.selector_calibration
        else None
    )
    proposal_selection_policy, proposal_selection_policy_sha256 = (
        load_proposal_selection_policy(args.selection_policy)
    )
    policy = decoding_policy(
        calibration_evidence,
        calibration_evidence_sha256,
        selector_sha256,
        proposal_selection_policy,
        proposal_selection_policy_sha256,
    )
    policy_sha256 = sha256_json(policy)
    learned_risk_enabled = policy["learned_uncertainty_and_risk_enabled"]
    selector_calibration = (
        load_selector_calibration(args.selector_calibration)
        if learned_risk_enabled and args.selector_calibration
        else None
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for context proposal generation")
    model = load_any_model(args.checkpoint, map_location="cpu").to(args.device)
    model.eval()
    catalog = json.loads(catalog_path.read_text())
    entries = request_entries(request, catalog, data_path)
    request_task = request["task"]
    root_seed = int(request.get("seed", 0))
    cache_hits = 0
    cache_misses = 0
    escalations = 0
    final_artifacts = []
    with data_path.open("rb") as data_handle:
        for entry in entries:
            context, values = context_values(data_handle, int(entry["catalog_index"]))
            context_sha256 = hashlib.sha256(context).hexdigest()
            seed = context_seed(root_seed, context_sha256)
            task = entry["task"]
            embedding = torch.tensor(values[:1536], dtype=torch.float32)
            target = torch.tensor(values[1536:], dtype=torch.float32)
            final_artifact = None
            for reservoir in RESERVOIRS:
                key = {
                    "task": task,
                    "accepted_checkpoint_sha256": checkpoint_sha256,
                    "context_sha256": context_sha256,
                    "seed": seed,
                    "decoding_policy_sha256": policy_sha256,
                    "reservoir_size": reservoir,
                }
                write_slot_log(args.cache_root, key)
                path = cache_path(args.cache_root, key)
                if path.is_file():
                    artifact = load_artifact(path, key)
                    cache_hits += 1
                else:
                    raw, selected = generate_reservoir(
                        model,
                        embedding,
                        target,
                        task,
                        seed,
                        reservoir,
                        args.device,
                        proposal_selection_policy,
                        learned_risk_enabled,
                        selector_calibration,
                    )
                    candidate = proposal_artifact(key, reservoir, raw, selected)
                    created = atomic_create_json(path, candidate)
                    artifact = candidate if created else load_artifact(path, key)
                    cache_misses += int(created)
                    cache_hits += int(not created)
                final_artifact = artifact
                ready = sum(
                    slot["status"] == "ready" for slot in artifact["slots"]
                )
                if ready == SLOTS or reservoir == RESERVOIRS[-1]:
                    break
                escalations += 1
            assert final_artifact is not None
            final_artifacts.append((entry, final_artifact))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    permanent_missing = 0
    ready_by_task = {t: 0 for t in ("binary", "regression", "multiclass", "timeseries")}
    requested_by_task = {t: 0 for t in ("binary", "regression", "multiclass", "timeseries")}
    with temporary.open("w", encoding="utf-8") as handle:
        for entry, artifact in final_artifacts:
            task = entry["task"]
            requested_by_task[task] += SLOTS
            ready_by_task[task] += len(artifact["proposals"])
            permanent_missing += sum(
                slot["status"] == "permanent_failure"
                for slot in artifact["slots"]
            )
            for proposal in artifact["proposals"]:
                row = aggregate_row(
                    proposal,
                    task=task,
                    dataset_hash=entry["dataset_hash"],
                    context_sha256=artifact["key"]["context_sha256"],
                    catalog_sha256=pointer["catalog_sha256"],
                    checkpoint_sha256=checkpoint_sha256,
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "schema": RUN_SCHEMA,
        "task": None if request_task == "mixed" else request_task,
        "model_checkpoint_sha256": checkpoint_sha256,
        "accepted_checkpoint_sha256": checkpoint_sha256,
        "request_sha256": sha256_file(args.request),
        "catalog_sha256": pointer["catalog_sha256"],
        "decoding_policy": policy,
        "decoding_policy_sha256": policy_sha256,
        "proposal_selection_policy_sha256": proposal_selection_policy_sha256,
        "contexts": len(entries),
        "datasets": len(entries),
        "requested_slots": len(entries) * SLOTS,
        "ready_slots": sum(
            len(artifact["proposals"]) for _, artifact in final_artifacts
        ),
        "permanent_missing_slots": permanent_missing,
        "counts": ready_by_task,
        "proposal_validity": sum(ready_by_task.values())
        / max(1, sum(requested_by_task.values())),
        "proposal_validity_by_task": {
            task: ready_by_task[task] / requested
            for task, requested in requested_by_task.items()
            if requested
        },
        "cohort_sha256": request.get("cohort_sha256"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "reservoir_escalations": escalations,
        "cross_context_backfill_rows": 0,
        "zero_cross_context_backfill": True,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "elapsed_ms": elapsed_ms,
        "mean_context_to_ready_ms": elapsed_ms / max(1, len(entries)),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
