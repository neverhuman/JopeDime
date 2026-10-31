#!/usr/bin/env python3
"""Descriptive W4 cached-pool random/hash baseline shard; never a gate runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np

PACKET_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
PACKET_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
ROSTER_SHA = "823b0240d98c6ef11704c3fdfe853afdaca7074e13e84a9a2ddf8b82004c8a7b"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
TASK_METADATA_SHA = "510fb1a0536766246564002b6d8186c260426f88a0f1ae9d16661b53dcda13f0"
SEEDS = (701, 702, 703)
POLICIES = ("random4", "hash_ordered4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def select_ids(policy: str, candidate_ids: list[str], context_sha: str, seed: int) -> list[str]:
    """Select using identities only; outcomes are not an input to this function."""
    if len(candidate_ids) < 4 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate panel must contain at least four unique identities")
    if policy == "random4":
        token = hashlib.sha256(f"random4:{seed}:{context_sha}".encode()).hexdigest()
        rng = random.Random(int(token[:16], 16))
        return rng.sample(candidate_ids, 4)
    if policy == "hash_ordered4":
        return sorted(
            candidate_ids,
            key=lambda item: hashlib.sha256(f"hash4:{seed}:{context_sha}:{item}".encode()).hexdigest(),
        )[:4]
    raise ValueError(f"unknown registered policy: {policy}")


def utility(fitness: float, gate: bool) -> float:
    if not math.isfinite(fitness):
        raise ValueError("non-finite selected fitness")
    return -fitness if gate else -2.0


def self_test() -> None:
    ids = [f"candidate-{index}" for index in range(20)]
    for policy in POLICIES:
        selected = select_ids(policy, ids, "context", 701)
        assert len(selected) == len(set(selected)) == 4
        assert selected == select_ids(policy, ids, "context", 701)
    assert set(select_ids("random4", ids, "context", 701)) <= set(ids)
    assert set(select_ids("hash_ordered4", ids, "context", 701)) <= set(ids)
    assert utility(0.5, True) == -0.5
    assert utility(0.5, False) == -2.0
    try:
        select_ids("hash_ordered4", ids[:3], "context", 701)
    except AssertionError:
        pass
    except ValueError:
        pass
    else:
        raise AssertionError("short panel was not refused")


def verify_roster(input_root: Path, roster_path: Path) -> tuple[dict, list[dict]]:
    packet_manifest = input_root / "source_manifest.json"
    selected = input_root / "selected.json"
    metadata = input_root / "task_metadata.json"
    expected = {
        packet_manifest: PACKET_FILE_SHA,
        selected: SELECTED_SHA,
        metadata: TASK_METADATA_SHA,
        roster_path: ROSTER_SHA,
    }
    for path, expected_sha in expected.items():
        if sha256(path) != expected_sha:
            raise ValueError(f"pinned input hash mismatch: {path}")

    packet = json.loads(packet_manifest.read_text())
    roster = json.loads(roster_path.read_text())
    if packet.get("source_manifest_sha256") != PACKET_ID:
        raise ValueError("H50 packet identity mismatch")
    if packet.get("outer_rows_delivered") != 0 or packet.get("excluded_payloads_opened") != 0:
        raise ValueError("packet records prohibited rows or payload access")
    contexts = roster.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 50:
        raise ValueError("expected the frozen 50-context H67 acquisition roster")
    if roster.get("declared_train_contexts") != 1219 or roster.get("declared_train_families") != 205:
        raise ValueError("roster declared corpus counts differ from the admitted packet")
    if roster.get("resolved_family_groups") != 205:
        raise ValueError("unexpected family/table alias resolution count")
    if roster.get("source_manifest_sha256") != PACKET_ID or roster.get("outcomes_in_manifest") is not False:
        raise ValueError("roster does not match the admitted training packet")
    folds = defaultdict(int)
    identities: set[str] = set()
    for row in contexts:
        if row.get("dl_split") != "DL-Train" or row.get("task") != "regression":
            raise ValueError("roster contains a non-training or non-regression row")
        if row.get("fold") not in range(5) or not row.get("family_group"):
            raise ValueError("roster has an invalid fold/family identity")
        if row["context_sha256"] in identities:
            raise ValueError("duplicate context in frozen roster")
        identities.add(row["context_sha256"])
        folds[row["fold"]] += 1
    if sorted(folds.items()) != [(fold, 10) for fold in range(5)]:
        raise ValueError("roster must contain ten fit-only contexts per family fold")
    return roster, contexts


def run(input_root: Path, roster_path: Path, out_dir: Path, s3_prefix: str) -> dict:
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    cpu_started = time.process_time()
    roster, contexts = verify_roster(input_root, roster_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    trace_path = out_dir / "selection_trace.jsonl"
    by_key: dict[tuple[str, int, str], float] = {}
    family_by_context = {row["context_sha256"]: row["family_group"] for row in contexts}
    context_fold = {row["context_sha256"]: row["fold"] for row in contexts}
    pool_hashes = []

    with trace_path.open("x", encoding="utf-8") as trace:
        for row in contexts:
            context = row["context_sha256"]
            pool_path = input_root / "pools" / f"{context}.npz"
            pool_sha = row["pool_sha256"]
            if sha256(pool_path) != pool_sha:
                raise ValueError(f"training pool hash mismatch for {context}")
            with np.load(pool_path, allow_pickle=False) as pool:
                ids = [item.decode() if isinstance(item, bytes) else str(item) for item in pool["program_sha256"].tolist()]
                if len(ids) != len(set(ids)) or len(ids) < 4:
                    raise ValueError(f"invalid candidate identities for {context}")

            # Freeze every selection before opening the cached outcome arrays.
            choices = [
                (seed, policy, select_ids(policy, ids, context, seed))
                for seed in SEEDS
                for policy in POLICIES
            ]
            with np.load(pool_path, allow_pickle=False) as pool:
                fitness = np.asarray(pool["fitness"], dtype=np.float64)
                gates = np.asarray(pool["gate"], dtype=np.bool_)
            if len(fitness) != len(ids) or len(gates) != len(ids):
                raise ValueError(f"outcome arrays do not align for {context}")
            index = {program_id: offset for offset, program_id in enumerate(ids)}
            for seed, policy, chosen in choices:
                # Outcomes are consulted only after each identity-only selection is frozen.
                scores = [utility(float(fitness[index[item]]), bool(gates[index[item]])) for item in chosen]
                retained = max(scores)
                by_key[(context, seed, policy)] = retained
                probability = 4.0 / len(ids) if policy == "random4" else 1.0
                trace.write(json.dumps({
                    "context_sha256": context,
                    "family_group": row["family_group"],
                    "fold": row["fold"],
                    "seed": seed,
                    "policy": policy,
                    "candidate_count": len(ids),
                    "selected_program_ids": chosen,
                    "selected_candidate_inclusion_probability": probability,
                    "retained_best_utility": retained,
                }, sort_keys=True, allow_nan=False) + "\n")
            if sha256(pool_path) != pool_sha:
                raise ValueError(f"training pool changed during read for {context}")
            pool_hashes.append({"context_sha256": context, "sha256": pool_sha, "s3_key": None})

    family_deltas: dict[int, dict[str, list[float]]] = {seed: defaultdict(list) for seed in SEEDS}
    for row in contexts:
        context = row["context_sha256"]
        family = row["family_group"]
        for seed in SEEDS:
            random_value = by_key[(context, seed, "random4")]
            hash_value = by_key[(context, seed, "hash_ordered4")]
            family_deltas[seed][family].append(hash_value - random_value)

    seed_metrics = {}
    for seed in SEEDS:
        arm_means = {
            policy: mean(by_key[(row["context_sha256"], seed, policy)] for row in contexts)
            for policy in POLICIES
        }
        per_family = [mean(values) for values in family_deltas[seed].values()]
        seed_metrics[str(seed)] = {
            "contexts": len(contexts),
            "families": len(per_family),
            "mean_retained_best_utility": arm_means,
            "hash_ordered4_minus_random4": {
                "family_mean": mean(per_family),
                "family_median": median(per_family),
                "families_losing_more_than_0_01_fraction": sum(value < -0.01 for value in per_family) / len(per_family),
            },
            "fold_context_counts": {str(fold): sum(context_fold[row["context_sha256"]] == fold for row in contexts) for fold in range(5)},
        }

    finished = datetime.now(timezone.utc)
    metrics = {
        "schema": "jopedime.w4-cached-baseline-shard-metrics.v1",
        "status": "DESCRIPTIVE_CACHED_POOL_SCREEN_ONLY",
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "context_count": len(contexts),
        "family_groups": len(set(family_by_context.values())),
        "fold_count": 5,
        "cached_replay_reveals": len(contexts) * len(SEEDS) * len(POLICIES) * 4,
        "native_fits": 0,
        "seed_metrics": seed_metrics,
        "Elo": None,
        "val_reg_inner": None,
        "U4": None,
        "advance_gate_evaluated": False,
        "scientific_verdict": None,
    }
    inputs = [
        {"local_path": str(input_root / "source_manifest.json"), "s3_key": None, "sha256": PACKET_FILE_SHA},
        {"local_path": str(input_root / "selected.json"), "s3_key": None, "sha256": SELECTED_SHA},
        {"local_path": str(input_root / "task_metadata.json"), "s3_key": None, "sha256": TASK_METADATA_SHA},
        {"local_path": str(roster_path), "s3_key": None, "sha256": ROSTER_SHA},
        *pool_hashes,
    ]
    manifest = {
        "schema": "jopedime.w4-cached-baseline-shard-run.v1",
        "run_id": out_dir.name,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_file_sha256": sha256(Path(__file__)),
        "s3_output_prefix": s3_prefix.rstrip("/"),
        "s3_outputs": [
            f"{s3_prefix.rstrip('/')}/metrics.json",
            f"{s3_prefix.rstrip('/')}/manifest.json",
            f"{s3_prefix.rstrip('/')}/selection_trace.jsonl",
        ],
        "input_packet_id": PACKET_ID,
        "inputs": inputs,
        "seeds": list(SEEDS),
        "fold_seed": roster["fold_seed"],
        "folds": list(range(5)),
        "host": socket.gethostname(),
        "cpu_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.monotonic() - clock,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "metrics": {"Elo": None, "val_reg_inner": None, "U4": None, "descriptive_metrics_file": "metrics.json"},
        "verdict": "NO_ADVANCEMENT_CLAIM; FIT_ONLY_RETROSPECTIVE_CACHED_POOL_DIAGNOSTIC",
        "limitations": [
            "H67 fit-only acquisition contexts are development inputs, not held-family evaluation data.",
            "Only registered random4 and hash_ordered4 arms are included; no r882, program-prior, adaptive, or prospective arm.",
            "Cached retrospective selections are not zero-cost deployment evaluations; native_fits is zero.",
            "The admitted H50 offline manifest does not publish per-pool S3 object keys; no new S3 input was read.",
            "No Elo, val_reg_inner, U4 primary, bootstrap gate, or scientific verdict was computed.",
        ],
    }
    (out_dir / "metrics.json").write_bytes(json_bytes(metrics))
    (out_dir / "manifest.json").write_bytes(json_bytes(manifest))
    return {"manifest": str(out_dir / "manifest.json"), "metrics": str(out_dir / "metrics.json"), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--roster", type=Path, default=Path("research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--s3-prefix")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("H75_SELF_TEST_PASS")
        return
    if args.out is None or args.s3_prefix is None:
        parser.error("--out and --s3-prefix are required for a data-backed run")
    print(json.dumps(run(args.input, args.roster, args.out, args.s3_prefix), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
