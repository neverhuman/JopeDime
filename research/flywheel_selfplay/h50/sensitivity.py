#!/usr/bin/env python3
"""Train-fold-only L2 sensitivity probe for the frozen r785 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
H48 = HERE.parent / "h48"
sys.path.insert(0, str(H48))
import run as h48  # noqa: E402

RADII = (0.01, 0.03, 0.1, 0.3, 1.0)
DIRECTIONS_PER_RADIUS = 64
SEED_BASE = 50000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resource_guard(device: torch.device) -> dict:
    available = psutil.virtual_memory().available
    cpu = psutil.cpu_percent(interval=1)
    free, _ = torch.cuda.mem_get_info(device)
    sample = {
        "available_ram_bytes": available,
        "cpu_percent": cpu,
        "cuda_free_bytes": free,
    }
    if available < 20 * 1024**3:
        raise RuntimeError(f"available RAM below 20 GiB: {sample}")
    if cpu > 85:
        raise RuntimeError(f"host CPU above 85%: {sample}")
    if free < 20 * 1024**3:
        raise RuntimeError(f"cuda:0 free memory below 20 GiB: {sample}")
    return sample


def flat_direction(seed: int, parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    count = sum(parameter.numel() for parameter in parameters)
    direction = torch.randn(count, generator=generator, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    result = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        result.append(direction[offset : offset + size].reshape(parameter.shape).to(parameter))
        offset += size
    if offset != count:
        raise RuntimeError("direction did not cover the complete adapter")
    return result


def family_fold_rows(selected: dict, train_ids: list[str], family_by_cid: dict[str, str]):
    folds = h48.family_folds(selected["train"], count=5, seed=4800)
    assignments = []
    seen: set[str] = set()
    for index, held_families in enumerate(folds):
        held = [cid for cid in train_ids if family_by_cid[cid] in held_families]
        overlap = seen & held_families
        if overlap:
            raise RuntimeError(f"family appears in multiple folds: {sorted(overlap)}")
        seen.update(held_families)
        assignments.append((index, sorted(held_families), held))
    if seen != set(family_by_cid.values()):
        raise RuntimeError("inner family folds do not cover DL-Train")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h50/ckpt/round-785.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h50/runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("H50-S requires cuda:0; CPU fallback is prohibited")
    device = torch.device("cuda:0")
    resource_guard(device)
    if sha(args.checkpoint) != h48.CHECKPOINT_SHA:
        raise RuntimeError("r785 checkpoint hash mismatch")
    manifest = json.loads((args.input / "source_manifest.json").read_text())
    manifest_file_sha = sha(args.input / "source_manifest.json")
    if manifest_file_sha != "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4":
        raise RuntimeError("H50 input manifest file hash mismatch")
    if manifest.get("excluded_dl_holdout_count") != 93 or manifest.get("excluded_payloads_opened") != 0 or manifest.get("outer_rows_delivered") != 0:
        raise RuntimeError("input manifest holdout/outer audit failed")

    selected, pools, train_ids, train_family, net, trainable, tokens, masks, guard = h48.build_model(
        args.input, args.checkpoint, device
    )
    if sum(parameter.numel() for parameter in trainable) != 29211:
        raise RuntimeError("unexpected adapter parameter count")
    if h48.h46.checkpoint_guard(guard, net) != h48.DOPE_SHA:
        raise RuntimeError("DOPE identity changed")
    predict = h48.make_predictor(net, pools, tokens, masks, device)
    assignments = family_fold_rows(selected, train_ids, train_family)
    baseline_weights = [parameter.detach().clone() for parameter in trainable]
    baseline_by_family: dict[str, float] = {}
    baseline_folds = []
    for fold_index, held_families, held in assignments:
        baseline = h48.evaluate_utility(held, pools, predict)
        baseline_by_family.update(baseline["per_family"])
        baseline_folds.append({
            "fold": fold_index,
            "heldout_families": held_families,
            "heldout_contexts": len(held),
            "baseline_family_utility": baseline["mean"],
        })
    if len(baseline_by_family) != 205:
        raise RuntimeError("baseline does not cover exactly 205 DL-Train families")

    progress_path = args.output / "directions.jsonl"
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            row = json.loads(line)
            completed[(row["radius_index"], row["direction_index"])] = row
    started = time.monotonic()
    radius_results = []
    for radius_index, radius in enumerate(RADII):
        rows = []
        for direction_index in range(DIRECTIONS_PER_RADIUS):
            prior = completed.get((radius_index, direction_index))
            if prior is not None:
                rows.append(prior)
                continue
            resource = resource_guard(device)
            direction = flat_direction(SEED_BASE + radius_index * DIRECTIONS_PER_RADIUS + direction_index, trainable)
            with torch.no_grad():
                for parameter, baseline, vector in zip(trainable, baseline_weights, direction, strict=True):
                    parameter.copy_(baseline + radius * vector)
            h48.h46.checkpoint_guard(guard, net, expected=h48.DOPE_SHA)
            fold_index = direction_index % len(assignments)
            _, held_families, held = assignments[fold_index]
            candidate_by_family = h48.evaluate_utility(held, pools, predict)["per_family"]
            if set(candidate_by_family) != set(held_families):
                raise RuntimeError("candidate did not score exactly its assigned held-out families")
            family_delta = {
                family: candidate_by_family[family] - baseline_by_family[family]
                for family in baseline_by_family
                if family in candidate_by_family
            }
            utility_delta = float(np.mean(list(family_delta.values())))
            moved_families = sum(delta != 0.0 for delta in family_delta.values())
            row = {
                "radius": radius,
                "radius_index": radius_index,
                "direction_index": direction_index,
                "seed": SEED_BASE + radius_index * DIRECTIONS_PER_RADIUS + direction_index,
                "heldout_fold": fold_index,
                "heldout_family_count": len(held_families),
                "l2_norm": radius,
                "utility_delta": utility_delta,
                "exactly_zero_utility_delta": utility_delta == 0.0,
                "moved_families": moved_families,
                "resource_guard": resource,
            }
            rows.append(row)
            with progress_path.open("a") as progress:
                progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({"radius": radius, "direction": direction_index, "delta": utility_delta, "moved_families": moved_families}), flush=True)
            with torch.no_grad():
                for parameter, baseline in zip(trainable, baseline_weights, strict=True):
                    parameter.copy_(baseline)

        zeros = sum(row["exactly_zero_utility_delta"] for row in rows)
        result = {
            "radius": radius,
            "directions": DIRECTIONS_PER_RADIUS,
            "exactly_zero_count": zeros,
            "exactly_zero_fraction": zeros / DIRECTIONS_PER_RADIUS,
            "smallest_nonzero_abs_utility_delta": min(
                (abs(row["utility_delta"]) for row in rows if row["utility_delta"] != 0.0),
                default=None,
            ),
            "directions_with_any_family_movement": sum(row["moved_families"] > 0 for row in rows),
            "utility_delta_min": min(row["utility_delta"] for row in rows),
            "utility_delta_median": float(np.median([row["utility_delta"] for row in rows])),
            "utility_delta_max": max(row["utility_delta"] for row in rows),
            "fold_assignments": {
                str(fold_index): {
                    "directions": sum(row["heldout_fold"] == fold_index for row in rows),
                    "exactly_zero_count": sum(row["heldout_fold"] == fold_index and row["exactly_zero_utility_delta"] for row in rows),
                }
                for fold_index in range(len(assignments))
            },
        }
        radius_results.append(result)
        write_json(args.output / "sensitivity-progress.json", {
            "schema": "jope.h50.sensitivity/v1",
            "completed_radii": radius_results,
            "validation_scored": False,
            "validation_payloads_opened": False,
        })
        print(json.dumps({"radius_summary": result}), flush=True)

    zero_everywhere = all(row["exactly_zero_fraction"] > 0.90 for row in radius_results)
    summary = {
        "schema": "jope.h50.sensitivity/v1",
        "arm": "H50-S",
        "device": "cuda:0",
        "radii": list(RADII),
        "directions_per_radius": DIRECTIONS_PER_RADIUS,
        "seed_base": SEED_BASE,
        "trainable_params": 29211,
        "inner_folds": baseline_folds,
        "baseline_family_utility": float(np.mean(list(baseline_by_family.values()))),
        "direction_fold_assignment": "direction_index modulo five; each candidate is evaluated once on one whole-family held-out DL-Train fold",
        "radius_results": radius_results,
        "smallest_radius_with_any_movement": next(
            (row["radius"] for row in radius_results if row["directions_with_any_family_movement"] > 0), None
        ),
        "zero_fraction_above_90_percent_at_every_radius": zero_everywhere,
        "decision": "adapter not the bottleneck; stop search arms" if zero_everywhere else "continue preregistered H50 search arms",
        "dope_sha256": h48.DOPE_SHA,
        "checkpoint_sha256": sha(args.checkpoint),
        "input_source_manifest_file_sha256": manifest_file_sha,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "train_contexts": 1219,
        "train_families": 205,
        "excluded_dl_holdout_count": 93,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "outer_or_holdout_payloads_opened": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output / "sensitivity.json", summary)
    print(json.dumps({"decision": summary["decision"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
