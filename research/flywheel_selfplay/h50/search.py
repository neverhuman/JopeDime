#!/usr/bin/env python3
"""H50 ES3, diagonal CMA-ES, and EMA-of-ES train-fold search arms."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
H48_DIR = HERE.parent / "h48"
sys.path.insert(0, str(H48_DIR))
import run as h48  # noqa: E402

POPULATION = 32
SIGMA0 = 1e-3
STRICT_MARGIN = 0.001
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 50823
WALL_BUDGET = 282.0
FOLD_SEED = 4800


def resource_guard(device: torch.device) -> dict:
    available = psutil.virtual_memory().available
    cpu = psutil.cpu_percent(interval=1)
    free, _ = torch.cuda.mem_get_info(device)
    status = {"available_ram_bytes": available, "cpu_percent": cpu, "cuda_free_bytes": free}
    if available < 20 * 1024**3 or cpu > 85 or free < 20 * 1024**3:
        raise RuntimeError(f"H50 preseed guard failed: {status}")
    return status


def flatten_snapshot(snapshot: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().float().cpu().reshape(-1) for value in snapshot.values()])


def unflatten_snapshot(vector: torch.Tensor, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    offset = 0
    for name, value in template.items():
        size = value.numel()
        result[name] = vector[offset : offset + size].reshape(value.shape).to(value)
        offset += size
    if offset != vector.numel():
        raise RuntimeError("flat adapter vector length mismatch")
    return result


def make_folds(selected: dict, train_ids: list[str], family_by_cid: dict[str, str]):
    folds = h48.family_folds(selected["train"], count=5, seed=FOLD_SEED)
    heldouts = [[cid for cid in train_ids if family_by_cid[cid] in fold] for fold in folds]
    return heldouts


def family_deltas(net, candidate, heldouts, pools, predict, guard, baseline_by_family):
    h48.load_adapter_snapshot(net, candidate)
    h48.h46.checkpoint_guard(guard, net, expected=h48.DOPE_SHA)
    candidate_by_family = {}
    for held in heldouts:
        scored = h48.evaluate_utility(held, pools, predict)
        candidate_by_family.update(scored["per_family"])
    if set(candidate_by_family) != set(baseline_by_family):
        raise RuntimeError("candidate family set differs from fixed inner-CV baseline")
    return {family: candidate_by_family[family] - baseline_by_family[family] for family in baseline_by_family}


def bootstrap_median_ci(deltas: dict[str, float], sample_indexes: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray([deltas[family] for family in sorted(deltas)], dtype=np.float64)
    medians = np.median(values[sample_indexes], axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def score_candidate(net, candidate, heldouts, pools, predict, guard, baseline_by_family, samples):
    deltas = family_deltas(net, candidate, heldouts, pools, predict, guard, baseline_by_family)
    mean_delta = float(np.mean(list(deltas.values())))
    median_delta = float(np.median(list(deltas.values())))
    ci = bootstrap_median_ci(deltas, samples) if median_delta > STRICT_MARGIN else (median_delta, float("-inf"), median_delta)
    return {
        "candidate": candidate,
        "family_deltas": deltas,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "bootstrap_median_ci": list(ci),
        "strictly_admitted": ci[1] > STRICT_MARGIN,
    }


def centered_ranks(scores: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = (np.arange(len(scores), dtype=np.float64) - (len(scores) - 1) / 2) / len(scores)
    return ranks


def choose_admitted(best, candidate):
    if candidate["strictly_admitted"] and (
        best is None or candidate["mean_delta"] > best["mean_delta"]
    ):
        return candidate
    return best


def es_generation(center, sigma, seed, generation):
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + generation * 1009)
    noises = [torch.randn(center.shape, generator=generator, dtype=center.dtype) for _ in range(POPULATION // 2)]
    candidates = [center + sign * sigma * noise for noise in noises for sign in (1.0, -1.0)]
    return noises, candidates


def update_es(center, sigma, noises, scores):
    ranks = centered_ranks(scores)
    gradient = torch.zeros_like(center)
    for pair_index, noise in enumerate(noises):
        gradient.add_(noise, alpha=float((ranks[2 * pair_index] - ranks[2 * pair_index + 1]) / POPULATION))
    norm = torch.linalg.vector_norm(gradient).clamp_min(1e-12)
    center = center + (0.05 * sigma) * gradient / norm
    return center, ranks


def run_es(args, net, center, heldouts, pools, predict, guard, baseline_by_family, samples, ema=False):
    started = time.monotonic()
    current = center.clone()
    ema_center = center.clone()
    best = None
    generations = []
    generation = 0
    while generation == 0 or time.monotonic() - started + 150 < WALL_BUDGET:
        generation += 1
        noises, candidates = es_generation(current, SIGMA0, args.seed, generation)
        scored = []
        for index, vector in enumerate(candidates):
            resource_guard(torch.device("cuda:0"))
            result = score_candidate(
                net, unflatten_snapshot(vector, center_template), heldouts, pools, predict,
                guard, baseline_by_family, samples,
            )
            scored.append(result)
            if not ema:
                best = choose_admitted(best, result)
        current, ranks = update_es(current, SIGMA0, noises, [row["mean_delta"] for row in scored])
        if ema:
            ema_center = 0.9 * ema_center + 0.1 * current
            ema_result = score_candidate(
                net, unflatten_snapshot(ema_center, center_template), heldouts, pools,
                predict, guard, baseline_by_family, samples,
            )
            best = choose_admitted(best, ema_result)
        generations.append({
            "generation": generation,
            "population": POPULATION,
            "sigma": SIGMA0,
            "mean_delta_min": min(row["mean_delta"] for row in scored),
            "mean_delta_median": float(np.median([row["mean_delta"] for row in scored])),
            "mean_delta_max": max(row["mean_delta"] for row in scored),
            "strictly_admitted_count": sum(row["strictly_admitted"] for row in scored),
            "rank_shaping": ranks.tolist(),
            "ema_candidate": ({
                "mean_delta": ema_result["mean_delta"],
                "median_delta": ema_result["median_delta"],
                "bootstrap_median_ci": ema_result["bootstrap_median_ci"],
                "strictly_admitted": ema_result["strictly_admitted"],
            } if ema else None),
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"arm": args.arm, "seed": args.seed, "generation": generation, "best_admitted_mean_delta": None if best is None else best["mean_delta"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    return best, generations, time.monotonic() - started


def run_cma(args, net, center, heldouts, pools, predict, guard, baseline_by_family, samples):
    started = time.monotonic()
    current = center.clone()
    dimension = current.numel()
    covariance = torch.ones_like(current)
    p_sigma = torch.zeros_like(current)
    p_c = torch.zeros_like(current)
    sigma = SIGMA0
    mu = POPULATION // 2
    weights = torch.log(torch.tensor(mu + 0.5, dtype=torch.float64)) - torch.log(torch.arange(1, mu + 1, dtype=torch.float64))
    weights /= weights.sum()
    mueff = 1.0 / float(torch.sum(weights**2))
    cc = (4 + mueff / dimension) / (dimension + 4 + 2 * mueff / dimension)
    cs = (mueff + 2) / (dimension + mueff + 5)
    c1 = 2 / ((dimension + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dimension + 2) ** 2 + mueff))
    damping = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dimension + 1)) - 1) + cs
    chi_n = math.sqrt(dimension) * (1 - 1 / (4 * dimension) + 1 / (21 * dimension * dimension))
    generator = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + 88003)
    best = None
    generations = []
    generation = 0
    while generation == 0 or time.monotonic() - started + 150 < WALL_BUDGET:
        generation += 1
        noises = [torch.randn(current.shape, generator=generator, dtype=current.dtype) for _ in range(POPULATION)]
        y_rows = [covariance.sqrt() * noise for noise in noises]
        scored = []
        for noise, step in zip(noises, y_rows, strict=True):
            resource_guard(torch.device("cuda:0"))
            result = score_candidate(
                net, unflatten_snapshot(current + sigma * step, center_template), heldouts,
                pools, predict, guard, baseline_by_family, samples,
            )
            scored.append(result)
            best = choose_admitted(best, result)
        order = np.argsort([-row["mean_delta"] for row in scored], kind="stable")[:mu]
        selected_z = [noises[int(i)].double() for i in order]
        w = weights.to(dtype=torch.float32)
        z_mean = sum((selected_z[i].float() * w[i] for i in range(mu)), torch.zeros_like(current))
        old_sigma = sigma
        old_sqrt_covariance = covariance.sqrt()
        weighted_y = old_sqrt_covariance * z_mean
        invsqrt_step = z_mean / covariance.sqrt().clamp_min(1e-12)
        p_sigma = (1 - cs) * p_sigma + math.sqrt(cs * (2 - cs) * mueff) * invsqrt_step
        path_norm = float(torch.linalg.vector_norm(p_sigma))
        hsig = path_norm / math.sqrt(1 - (1 - cs) ** (2 * generation)) / chi_n < (1.4 + 2 / (dimension + 1))
        weighted_y2 = sum((w[i] * y_rows[int(order[i])] ** 2 for i in range(mu)), torch.zeros_like(current))
        p_c = (1 - cc) * p_c + (math.sqrt(cc * (2 - cc) * mueff) * weighted_y if hsig else 0)
        covariance = (1 - c1 - cmu) * covariance + c1 * (p_c**2 + (0 if hsig else cc * (2 - cc) * covariance)) + cmu * weighted_y2
        covariance.clamp_(min=1e-12, max=1e12)
        sigma *= math.exp((cs / damping) * (path_norm / chi_n - 1))
        sigma = min(max(sigma, 1e-6), 0.1)
        current = current + old_sigma * weighted_y
        generations.append({
            "generation": generation,
            "population": POPULATION,
            "sigma": sigma,
            "mean_delta_min": min(row["mean_delta"] for row in scored),
            "mean_delta_median": float(np.median([row["mean_delta"] for row in scored])),
            "mean_delta_max": max(row["mean_delta"] for row in scored),
            "strictly_admitted_count": sum(row["strictly_admitted"] for row in scored),
            "covariance_diagonal_min": float(covariance.min()),
            "covariance_diagonal_median": float(covariance.median()),
            "covariance_diagonal_max": float(covariance.max()),
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"arm": args.arm, "seed": args.seed, "generation": generation, "best_admitted_mean_delta": None if best is None else best["mean_delta"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    return best, generations, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("ES3", "CMA", "ESEMA"), required=True)
    parser.add_argument("--seed", type=int, choices=(6, 7, 8), required=True)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h50/ckpt/round-785.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h50/runs")
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if not args.lock_held:
        raise RuntimeError("search must be launched by the lock-owning H50 coordinator")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("H50 search is CUDA-only")
    torch.cuda.set_per_process_memory_fraction(0.90)
    resource_guard(device)
    if h48.digest(args.checkpoint) != h48.CHECKPOINT_SHA:
        raise RuntimeError("H50 r785 checkpoint hash mismatch")
    source_manifest = json.loads((args.input / "source_manifest.json").read_text())
    selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = h48.build_model(args.input, args.checkpoint, device)
    if h48.h46.checkpoint_guard(guard, net) != h48.DOPE_SHA or sum(p.numel() for p in trainable) != 29211:
        raise RuntimeError("H50 frozen-DOPE or trainable-adapter inventory mismatch")
    predict = h48.make_predictor(net, pools, tokens, masks, device)
    heldouts = make_folds(selected, train_ids, family_by_cid)
    baseline_by_family = {}
    for held in heldouts:
        baseline_by_family.update(h48.evaluate_utility(held, pools, predict)["per_family"])
    if len(baseline_by_family) != 205:
        raise RuntimeError("r785 baseline does not cover 205 inner-CV families")

    template = h48.parameter_snapshot(net)
    center = flatten_snapshot(template)
    global center_template
    center_template = template
    baseline_score = float(np.mean(list(baseline_by_family.values())))
    samples = np.random.default_rng(BOOTSTRAP_SEED + args.seed).integers(
        0, len(baseline_by_family), size=(BOOTSTRAP_DRAWS, len(baseline_by_family)), dtype=np.uint16
    )
    if args.arm == "CMA":
        best, generations, elapsed = run_cma(args, net, center, heldouts, pools, predict, guard, baseline_by_family, samples)
    else:
        best, generations, elapsed = run_es(args, net, center, heldouts, pools, predict, guard, baseline_by_family, samples, ema=args.arm == "ESEMA")

    output = args.output / f"H50-{args.arm}" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    selected_candidate = {
        "arm": args.arm,
        "seed": args.seed,
        "accepted": best is not None,
        "selection_rule": "family-bootstrap 95% lower bound of median held-out family-mean utility delta > 0.001",
        "checkpoint": None,
    }
    if best is not None:
        snapshot = best["candidate"]
        h48.load_adapter_snapshot(net, snapshot)
        h48.h46.checkpoint_guard(guard, net, expected=h48.DOPE_SHA)
        h48.save_adapter_snapshot(output / "selected-adapter.pt", snapshot, h48.digest(args.checkpoint))
        selected_candidate.update({
            "checkpoint": "selected-adapter.pt",
            "selected_mean_family_delta": best["mean_delta"],
            "selected_median_family_delta": best["median_delta"],
            "selected_bootstrap_median_ci": best["bootstrap_median_ci"],
            "selected_family_deltas": best["family_deltas"],
        })
    h48.atomic_json(output / "selected-candidate.json", selected_candidate)
    result = {
        "schema": "jope.h50.train-arm/v1",
        "arm": args.arm,
        "seed": args.seed,
        "device": "cuda:0",
        "population": POPULATION,
        "wall_budget_seconds": WALL_BUDGET,
        "wall_seconds": elapsed,
        "strict_inner_margin": STRICT_MARGIN,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "inner_cv_baseline_family_utility": baseline_score,
        "selected_candidate_accepted": best is not None,
        "selected_mean_family_delta": None if best is None else best["mean_delta"],
        "selected_median_family_delta": None if best is None else best["median_delta"],
        "selected_bootstrap_median_ci": None if best is None else best["bootstrap_median_ci"],
        "generations": generations,
        "controls": {
            "r785_checkpoint_sha256": h48.digest(args.checkpoint),
            "r882_control_file": str(Path.home() / "jope-data/h50/reference/r882-control.json"),
            "r882_control_kind": "historical exact-tie negative control, unchanged; not rescored",
            "r882_observed_gain": 0.0,
        },
        "trainable_params": 29211,
        "dope_sha256": h48.DOPE_SHA,
        "checkpoint_sha256": h48.digest(args.checkpoint),
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "input_source_manifest_file_sha256": h48.digest(args.input / "source_manifest.json"),
        "excluded_dl_holdout_count": 93,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "validation_scored": False,
        "validation_payloads_opened": False,
    }
    h48.atomic_json(output / "train-result.json", result)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "accepted": best is not None, "selected_delta": None if best is None else best["mean_delta"], "wall_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
