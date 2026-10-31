#!/usr/bin/env python3
"""Build the W4 training-only family roster and run a bounded protocol smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

SOURCE_MANIFEST_SHA = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
SOURCE_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
EXPECTED_TRAIN = 1219
EXPECTED_FAMILIES = 205


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def admitted_train_rows(input_dir: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    source_path = input_dir / "source_manifest.json"
    if sha256(source_path) != SOURCE_FILE_SHA:
        raise ValueError("source manifest file SHA256 differs from the admitted H50 packet")
    source = json.loads(source_path.read_text())
    if source.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA:
        raise ValueError("source manifest identity mismatch")
    if (source.get("excluded_dl_holdout_count"), source.get("excluded_payloads_opened"), source.get("outer_rows_delivered")) != (93, 0, 0):
        raise ValueError("holdout/outer-row closure does not match the admitted packet")

    selected = json.loads((input_dir / "selected.json").read_text())
    rows = selected.get("train")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TRAIN:
        raise ValueError("expected exactly 1,219 allowlisted DL-Train contexts")
    if len({row.get("context_sha256") for row in rows}) != EXPECTED_TRAIN:
        raise ValueError("duplicate or missing context identity")
    if len({row.get("family_id") for row in rows}) != EXPECTED_FAMILIES:
        raise ValueError("expected 205 declared DL-Train families")
    if any(row.get("dl_split") != "DL-Train" or row.get("task") != "regression" or row.get("source_split") not in {"train", "inner"} for row in rows):
        raise ValueError("selected roster contains a non-training or non-regression row")

    metadata = json.loads((input_dir / "task_metadata.json").read_text())
    for row in rows:
        item = metadata.get(row["dataset_id"])
        if not item or not item.get("canonical_sha256"):
            raise ValueError(f"missing canonical physical-table identity for {row['dataset_id']}")
    return source, rows, metadata


def resolved_groups(rows: list[dict], metadata: dict[str, dict]) -> dict[str, str]:
    """Merge declared families that share a physical canonical table identity."""
    aliases = UnionFind()
    for row in rows:
        family = "family:" + row["family_id"]
        table = "table:" + metadata[row["dataset_id"]]["canonical_sha256"]
        aliases.union(family, table)
    return {row["context_sha256"]: aliases.find("family:" + row["family_id"]) for row in rows}


def make_manifest(rows: list[dict], metadata: dict[str, dict], source: dict, selected_path: Path, metadata_path: Path) -> dict:
    groups = resolved_groups(rows, metadata)
    families = sorted(set(groups.values()))
    if len(families) > EXPECTED_FAMILIES:
        raise ValueError("alias resolution increased family count")
    rng = random.Random(4900)
    shuffled = list(families)
    rng.shuffle(shuffled)
    fold_groups = [set(shuffled[i::5]) for i in range(5)]
    if any(not fold for fold in fold_groups):
        raise ValueError("empty family fold")
    fold_by_group = {
        family_group: fold
        for fold, held_groups in enumerate(fold_groups)
        for family_group in held_groups
    }

    # Freeze ten deterministic, training-only contexts in each held-family fold.
    contexts = []
    for fold, held_groups in enumerate(fold_groups):
        candidates = [row for row in rows if groups[row["context_sha256"]] in held_groups]
        candidates.sort(key=lambda row: hashlib.sha256(f"4900:{fold}:{row['context_sha256']}".encode()).hexdigest())
        if len(candidates) < 10:
            raise ValueError(f"fold {fold} has fewer than ten acquisition contexts")
        contexts.extend({
            "context_sha256": row["context_sha256"],
            "dataset_id": row["dataset_id"],
            "view_id": row["view_id"],
            "family_group": groups[row["context_sha256"]],
            "fold": fold,
            "pool_sha256": row["pool_sha256"],
            "train_only_jtt1_sha256": row["train_only_jtt1_sha256"],
            "dl_split": row["dl_split"],
            "task": row["task"],
        } for row in candidates[:10])
    if len(contexts) != 50 or len({row["context_sha256"] for row in contexts}) != 50:
        raise ValueError("fit-only roster must contain 50 distinct contexts")
    return {
        "schema": "jopedime.w4-active-reveal-training-roster.v1",
        "preparation_only": True,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_manifest_file_sha256": SOURCE_FILE_SHA,
        "selected_train_file_sha256": sha256(selected_path),
        "task_metadata_file_sha256": sha256(metadata_path),
        "fold_seed": 4900,
        "fold_count": 5,
        "declared_train_contexts": EXPECTED_TRAIN,
        "declared_train_families": EXPECTED_FAMILIES,
        "resolved_family_groups": len(families),
        "family_folds": [
            {"fold": fold, "family_groups": sorted(held_groups)}
            for fold, held_groups in enumerate(fold_groups)
        ],
        "context_fold_assignments": [
            {
                "context_sha256": row["context_sha256"],
                "family_group": groups[row["context_sha256"]],
                "fold": fold_by_group[groups[row["context_sha256"]]],
            }
            for row in sorted(rows, key=lambda item: item["context_sha256"])
        ],
        "seed_rows": [701, 702, 703],
        "contexts_per_fold": 10,
        "contexts": contexts,
        "excluded": {"dl_val": True, "outer_test": True, "sealed": True, "holdout_payloads_opened": 0},
        "outcomes_in_manifest": False,
        "scientific_verdict": None,
    }


def two_plus_two(
    candidate_ids: list[str],
    reveal: Callable[[str], float],
    choose_first: Callable[[list[str]], list[str]],
    choose_second: Callable[[list[str], dict[str, float]], list[str]],
) -> tuple[list[str], dict[str, float]]:
    """Select two, reveal those only, then select two new candidates."""
    if len(candidate_ids) < 4 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("need at least four unique candidates")
    first = choose_first(list(candidate_ids))
    if len(first) != 2 or len(set(first)) != 2 or not set(first) <= set(candidate_ids):
        raise ValueError("first acquisition must select two distinct candidates")
    observed = {candidate: reveal(candidate) for candidate in first}
    remaining = [candidate for candidate in candidate_ids if candidate not in observed]
    second = choose_second(remaining, dict(observed))
    if len(second) != 2 or len(set(second)) != 2 or not set(second) <= set(remaining):
        raise ValueError("second acquisition must select two distinct unrevealed candidates")
    observed.update({candidate: reveal(candidate) for candidate in second})
    return first + second, observed


def smoke_one_real_train_pool(input_dir: Path, record: dict) -> dict:
    pool_path = input_dir / "pools" / f"{record['context_sha256']}.npz"
    if sha256(pool_path) != record["pool_sha256"]:
        raise ValueError("selected real-S3-derived training pool hash mismatch")
    with np.load(pool_path, allow_pickle=False) as pool:
        ids = [item.decode() if isinstance(item, bytes) else str(item) for item in pool["program_sha256"].tolist()]
        fitness = pool["fitness"].astype(np.float64)
        gate = pool["gate"].astype(bool)
    if len(ids) < 4 or len(set(ids)) != len(ids) or len(fitness) != len(ids) or len(gate) != len(ids):
        raise ValueError("candidate identity/outcome arrays are malformed")
    id_to_index = {candidate: i for i, candidate in enumerate(ids)}
    revealed_calls: list[str] = []

    def reveal(candidate: str) -> float:
        if candidate not in id_to_index:
            raise ValueError("attempted to reveal an unknown candidate")
        revealed_calls.append(candidate)
        # Outcomes are indexed only at the explicit reveal boundary.
        index = id_to_index[candidate]
        return float(-fitness[index] if gate[index] else -2.0)

    seed = 701
    choose_first = lambda remaining: sorted(remaining, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())[:2]

    def choose_second(remaining: list[str], history: dict[str, float]) -> list[str]:
        # Deterministic protocol stand-in: history affects the candidate ordering,
        # but no unrevealed outcome is passed to this callback.
        history_tag = hashlib.sha256(json.dumps(sorted(history.items())).encode()).hexdigest()
        return sorted(remaining, key=lambda item: hashlib.sha256(f"{history_tag}:{item}".encode()).hexdigest())[:2]

    chosen, observed = two_plus_two(ids, reveal, choose_first, choose_second)
    if len(chosen) != 4 or len(set(chosen)) != 4 or revealed_calls != chosen or set(observed) != set(chosen):
        raise ValueError("two-plus-two smoke did not reveal exactly four distinct choices")
    return {
        "status": "PASS_PROTOCOL_ONLY",
        "context_sha256": record["context_sha256"],
        "pool_sha256": record["pool_sha256"],
        "candidate_count": len(ids),
        "fit_count": len(chosen),
        "revealed_program_ids": chosen,
        "unrevealed_outcomes_used_for_selection": 0,
        "pool_outcome_arrays_loaded_for_smoke": True,
        "utility_comparison": None,
        "scientific_verdict": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("family_fold_manifest.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    cpu_started = time.process_time()
    source, rows, metadata = admitted_train_rows(args.input)
    manifest = make_manifest(rows, metadata, source, args.input / "selected.json", args.input / "task_metadata.json")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {"roster": "PASS", "contexts": len(manifest["contexts"]), "resolved_family_groups": manifest["resolved_family_groups"]}
    if args.smoke:
        result["protocol_smoke"] = smoke_one_real_train_pool(args.input, manifest["contexts"][0])
        runner_path = Path(__file__)
        receipt = {
            "schema": "jopedime.w4-active-reveal-protocol-smoke.v1",
            "status": result["protocol_smoke"]["status"],
            "preparation_only": True,
            "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": "nice -n 15 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 research/flywheel_selfplay/h67_w4_active_reveal_2plus2/prepare.py --input ~/jope-data/h50/input --manifest research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json --smoke",
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "process_cpu_seconds": time.process_time() - cpu_started,
            "cpu_thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "input_source_manifest_file_sha256": sha256(args.input / "source_manifest.json"),
            "input_selected_train_file_sha256": sha256(args.input / "selected.json"),
            "input_task_metadata_file_sha256": sha256(args.input / "task_metadata.json"),
            "runner_sha256": sha256(runner_path),
            "family_fold_manifest_sha256": sha256(args.manifest),
            "smoke": result["protocol_smoke"],
            "utility_comparison": None,
            "scientific_verdict": None,
        }
        receipt_path = args.manifest.with_name("smoke_receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        result["smoke_receipt"] = str(receipt_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
