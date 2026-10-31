#!/usr/bin/env python3
"""Leakage-guarded H57 antithetic direct-utility ES cells."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
H55 = ROOT / "research/flywheel_selfplay/h55/runner.py"
H48 = ROOT / "research/flywheel_selfplay/h48/run.py"
H49 = ROOT / "research/flywheel_selfplay/h49/score.py"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
FOLD_SEED = 4900
TRAINABLE_PARAMS = 29_211
SEEDS = (15, 16, 17)
SIGMAS = {"sigma001": 0.001, "sigma003": 0.003}
NULL_ARM = "null_sigma001"
POPULATION = 32
PAIRS = POPULATION // 2
DEFAULT_GENERATIONS = 8
ES_LEARNING_RATE = 0.01
INPUT = Path.home() / "jope-data/h50/input"
CHECKPOINT = Path.home() / "jope-data/h51/ckpt/round-882.pt"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def centered_ranks(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (POPULATION,) or not np.isfinite(values).all():
        raise ValueError("H57 requires 32 finite direct-utility scores")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(POPULATION, dtype=np.float64)
    ranks[order] = (np.arange(POPULATION, dtype=np.float64) - (POPULATION - 1) / 2) / POPULATION
    return ranks


def stream_seed(seed: int, sigma_name: str, generation: int, null_arm: bool) -> int:
    if seed not in SEEDS or sigma_name not in SIGMAS or generation < 1:
        raise ValueError("unregistered H57 random stream")
    sigma_code = 1 if sigma_name == "sigma001" else 3
    # The null shares the perturbation stream with sigma001/seed15; only
    # its rank-to-pair assignment is independently permuted.
    return seed * 1_000_003 + sigma_code * 97_409 + generation * 1_009


def antithetic_population(center: torch.Tensor, sigma: float, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if center.ndim != 1 or center.numel() != TRAINABLE_PARAMS or sigma <= 0:
        raise ValueError("H57 center must be the flat 29,211-parameter adapter")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noises = [torch.randn(center.shape, generator=generator, dtype=center.dtype) for _ in range(PAIRS)]
    candidates = [center + sign * sigma * noise for noise in noises for sign in (1.0, -1.0)]
    return noises, candidates


def es_gradient(noises: list[torch.Tensor], ranks: np.ndarray, sigma: float) -> torch.Tensor:
    if len(noises) != PAIRS or ranks.shape != (POPULATION,) or sigma <= 0:
        raise ValueError("invalid H57 antithetic gradient inputs")
    gradient = torch.zeros_like(noises[0])
    for pair, noise in enumerate(noises):
        coefficient = float(ranks[2 * pair] - ranks[2 * pair + 1]) / (2 * POPULATION * sigma)
        gradient.add_(noise, alpha=coefficient)
    return gradient


def adapter_vector(state: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    if tuple(sorted(state)) != names:
        raise RuntimeError("r882 adapter tensor names differ from frozen template")
    return torch.cat([state[name].detach().cpu().float().reshape(-1) for name in names])


def vector_state(vector: torch.Tensor, template: dict[str, torch.Tensor], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    state = {}
    offset = 0
    for name in names:
        source = template[name]
        count = source.numel()
        state[name] = vector[offset:offset + count].reshape(source.shape).to(dtype=source.dtype)
        offset += count
    if offset != vector.numel():
        raise RuntimeError("H57 adapter vector length mismatch")
    return state


def make_context(device: torch.device, input_dir: Path, checkpoint: Path) -> tuple[dict, object]:
    if sha256(checkpoint) != R882_SHA:
        raise RuntimeError("H57 r882 checkpoint SHA mismatch")
    h55 = import_module("h57_h55", H55)
    ctx = h55.prepare_training(input_dir, checkpoint, device)
    ctx["input_dir"] = input_dir
    if sha256(input_dir / "source_manifest.json") != MANIFEST_SHA:
        raise RuntimeError("H57 source manifest SHA mismatch")
    if sum(p.numel() for p in ctx["trainable"]) != TRAINABLE_PARAMS:
        raise RuntimeError("H57 adapter parameter count mismatch")
    ctx["net"].eval()
    dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if dope != DOPE_SHA:
        raise RuntimeError("H57 DOPE guard mismatch")
    return ctx, h55


def score_train_fitness(ctx: dict, h55, vector: torch.Tensor,
                        template: dict[str, torch.Tensor], names: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    h55.set_adapter(ctx["net"], vector_state(vector, template, names))
    ctx["net"].eval()
    ids = sorted(ctx["pools_a"])
    result = ctx["h48"].evaluate_utility(ids, ctx["pools_a"], ctx["predictor_a"])
    if len(result["per_family"]) != 164:
        raise RuntimeError("H57 fold-A family count changed")
    return float(result["mean"]), result["per_family"]


def run_generation(ctx: dict, h55, theta: torch.Tensor, optimizer: torch.optim.Optimizer,
                   template: dict[str, torch.Tensor], names: tuple[str, ...], *, seed: int,
                   sigma_name: str, generation: int, null_arm: bool) -> tuple[dict, float]:
    sigma = SIGMAS[sigma_name]
    before_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if before_dope != DOPE_SHA:
        raise RuntimeError("DOPE mismatch before H57 generation")
    started = time.monotonic()
    noises, candidates = antithetic_population(
        theta.detach().cpu(), sigma,
        stream_seed(seed, sigma_name, generation, null_arm),
    )
    scores = []
    for candidate in candidates:
        score, _ = score_train_fitness(ctx, h55, candidate, template, names)
        scores.append(score)
    ranks = centered_ranks(scores)
    if null_arm:
        rng = np.random.default_rng(stream_seed(seed, sigma_name, generation, True) + 17)
        ranks = ranks[rng.permutation(POPULATION)]
    gradient = es_gradient(noises, ranks, sigma)
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    old_theta = theta.detach().clone()
    # Adam maximizes direct utility by descending the negative ES gradient.
    optimizer.zero_grad(set_to_none=True)
    theta.grad = -gradient.to(theta)
    optimizer.step()
    update_l2 = float(torch.linalg.vector_norm(theta.detach() - old_theta))
    after_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if after_dope != DOPE_SHA:
        raise RuntimeError("DOPE changed during H57 generation")
    row = {
        "generation": generation, "population": POPULATION, "antithetic_pairs": PAIRS,
        "sigma": sigma, "fitness_mean": float(np.mean(scores)),
        "fitness_median": float(np.median(scores)), "fitness_min": float(np.min(scores)),
        "fitness_max": float(np.max(scores)), "es_gradient_l2": gradient_norm,
        "mean_theta_update_l2": update_l2, "rank_fitness_permuted": null_arm,
        "dope_sha256_before": before_dope, "dope_sha256_after": after_dope,
        "generation_wall_seconds": time.monotonic() - started,
    }
    return row, time.monotonic() - started


def source_hashes(input_dir: Path) -> dict[str, str]:
    paths = [H55, H48, H49, ROOT / "research/flywheel_selfplay/h46/run.py",
             ROOT / "research/flywheel_selfplay/h57/runner.py",
             input_dir / "cpu_exact_fused.py"]
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
            for path in paths}


def run_train(args) -> dict:
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H57 timing/outcome generations require cuda:0 and qualification.lock")
    if args.arm not in SIGMAS or args.seed not in SEEDS:
        raise RuntimeError("unregistered H57 candidate arm or seed")
    if not 1 <= args.generations <= 12:
        raise RuntimeError("H57 generation horizon outside preregistered 1..12 range")
    if args.null_arm and (args.arm, args.seed, args.mode) != ("sigma001", 15, "outcome"):
        raise RuntimeError("only the sigma001 seed15 outcome may be the preregistered null")
    if args.mode == "timing" and (args.arm, args.seed, args.generations) != ("sigma001", 15, 1):
        raise RuntimeError("only sigma001 seed15 one-generation timing is preregistered")
    if args.mode == "outcome" and args.generations not in (8, 12):
        raise RuntimeError("outcome horizon must be fixed at eight or amended to twelve")
    torch.set_num_threads(1)
    ctx, h55 = make_context(torch.device(args.device), args.input, args.checkpoint)
    template = h55.adapter_state(ctx["net"])
    names = tuple(sorted(template))
    theta0 = adapter_vector(template, names)
    if theta0.numel() != TRAINABLE_PARAMS:
        raise RuntimeError("H57 flat r882 adapter length mismatch")
    theta = torch.nn.Parameter(theta0.clone())
    optimizer = torch.optim.Adam([theta], lr=ES_LEARNING_RATE)
    dope_before = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    baseline_a, _ = score_train_fitness(ctx, h55, theta0, template, names)
    generations = []
    started = time.monotonic()
    for generation in range(1, args.generations + 1):
        row, _ = run_generation(ctx, h55, theta, optimizer, template, names,
                                seed=args.seed, sigma_name=args.arm,
                                generation=generation, null_arm=args.null_arm)
        generations.append(row)
        print(json.dumps({"arm": args.arm, "seed": args.seed, **row}, sort_keys=True), flush=True)
    final_state = vector_state(theta.detach().cpu(), template, names)
    adapter_sha = h55.adapter_digest(final_state)
    wall = time.monotonic() - started
    dope_after = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    payload = {
        "schema": "jope.h57.cell/v1", "status": "complete", "mode": args.mode,
        "arm": args.arm, "cell_id": NULL_ARM if args.null_arm else args.arm,
        "seed": args.seed, "null_arm": args.null_arm,
        "generations_completed": len(generations), "evaluated_perturbations": len(generations) * POPULATION,
        "population": POPULATION, "sigma": SIGMAS[args.arm], "adam_learning_rate": ES_LEARNING_RATE,
        "selection": "final ES mean theta after the fixed last generation; no best-generation or sampled-candidate selection",
        "fold_seed": FOLD_SEED, "fold_assignment": ctx["fold_assignment"],
        "fold_assignment_sha256": ctx["fold_sha"], "fold_a_family_count": len(ctx["fold_a_families"]),
        "fold_b_scored": False, "validation_payloads_opened": False,
        "baseline_fold_a_family_utility": baseline_a,
        "generations": generations, "generation_wall_seconds": [r["generation_wall_seconds"] for r in generations],
        "wall_seconds": wall, "adapter_sha256": adapter_sha,
        "r882_checkpoint_sha256": R882_SHA, "source_manifest_sha256": MANIFEST_SHA,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes(args.input), "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after, "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
        "update_norms": [r["mean_theta_update_l2"] for r in generations],
        "gradient_norms": [r["es_gradient_l2"] for r in generations],
    }
    if dope_before != DOPE_SHA or dope_after != DOPE_SHA:
        raise RuntimeError("H57 DOPE checksum changed across cell")
    if args.mode == "outcome":
        h55.check_disk_budget()
        ctx["h48"].save_adapter_snapshot(args.out / "candidate_adapter.pt", final_state, R882_SHA)
        payload["candidate_adapter_artifact"] = str(args.out / "candidate_adapter.pt")
        payload["candidate_adapter_artifact_sha256"] = sha256(args.out / "candidate_adapter.pt")
    atomic_json(args.out / ("timing.json" if args.mode == "timing" else "summary.json"), payload)
    return payload


def score_fold_b(args) -> dict:
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H57 fold-B admission scoring requires cuda:0 and qualification.lock")
    h55 = import_module("h57_h55_score", H55)
    h48 = import_module("h57_h48_score", H48)
    h48.CHECKPOINT_SHA = R882_SHA
    selected, train_rows, _ = h55.input_metadata(args.input, h48)
    folds = h55.family_folds(train_rows, h48)
    fold_b_families = folds[4]
    rows_b = [row for row in train_rows if row["family_id"] in fold_b_families]
    fold_a_families = set().union(*folds[:4])
    rows_a_count = sum(row["family_id"] in fold_a_families for row in train_rows)
    if not rows_b or rows_a_count + len(rows_b) != len(train_rows):
        raise RuntimeError("H57 fold-A/B assignment is empty or incomplete")
    fold_assignment = {"seed": FOLD_SEED, "folds": [sorted(f) for f in folds],
                       "fold_a_families": sorted(fold_a_families),
                       "fold_b_families": sorted(fold_b_families),
                       "train_contexts_a": rows_a_count, "train_contexts_b": len(rows_b)}
    fold_sha = hashlib.sha256(json.dumps(fold_assignment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    pools, tables = h55.load_rows(args.input, rows_b, h48)
    net, trainable, guard, _ = h55.build_model(args.input, args.checkpoint, torch.device(args.device), h48)
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    ids = [row["context_sha256"] for row in rows_b]
    dope_before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    net.eval()
    baseline = h48.evaluate_utility(ids, pools, predict)
    candidates = []
    for path in sorted(args.run_root.glob("outcomes/cells/*/seed*/candidate_adapter.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (artifact.get("base_checkpoint_sha256") != R882_SHA or
                artifact.get("dope_sha256") != DOPE_SHA or artifact.get("trainable_params") != TRAINABLE_PARAMS):
            raise RuntimeError(f"H57 candidate artifact pin mismatch: {path}")
        h55.set_adapter(net, artifact["adapter_trainable_state"])
        net.eval()
        result = h48.evaluate_utility(ids, pools, predict)
        if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
            raise RuntimeError("DOPE changed during H57 fold-B candidate scoring")
        summary_path = path.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        candidates.append({"arm": summary["arm"], "seed": summary["seed"],
                           "adapter_sha256": sha256(path), "family_utility": result["per_family"],
                           "context_utility": result["per_context"],
                           "family_gain_vs_r882": {family: result["per_family"][family] - baseline["per_family"][family]
                                                   for family in baseline["per_family"]},
                           "fold_b_scoring_passes": 1})
    dope_after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    result = {"schema": "jope.h57.fold-b/v1", "status": "complete", "fold_seed": FOLD_SEED,
              "fold_assignment": fold_assignment, "fold_assignment_sha256": fold_sha,
              "r882_checkpoint_sha256": R882_SHA, "source_manifest_sha256": MANIFEST_SHA,
              "dope_sha256_before": dope_before, "dope_sha256_after": dope_after,
              "baseline_family_utility": baseline["per_family"],
              "baseline_context_utility": baseline["per_context"], "candidates": candidates,
              "candidate_count": len(candidates), "fold_b_read_count": len(candidates),
              "validation_payloads_opened": False, "holdout_payloads_opened": 0,
              "outer_rows_delivered": 0,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes(args.input)}
    atomic_json(args.run_root / "outcomes/fold_b.json", result)
    return result


def run_terminal_val(args) -> dict:
    admission = json.loads((args.run_root / "outcomes/admission.json").read_text())
    if admission.get("arms", {}).get(args.arm, {}).get("admitted") is not True:
        raise RuntimeError("H57 terminal DL-Val is forbidden for a nonadmitted sigma arm")
    if args.seed not in SEEDS or not args.lock_held or args.device != "cuda:0":
        raise RuntimeError("H57 terminal DL-Val requires a registered seed and qualification.lock")
    if args.out.exists():
        raise RuntimeError("refusing duplicate H57 terminal DL-Val receipt")
    h55 = import_module("h57_h55_val", H55)
    h48 = import_module("h57_h48_val", H48)
    h48.CHECKPOINT_SHA = R882_SHA
    h49 = import_module("h57_h49_val", H49)
    net, _, guard, _ = h55.build_model(args.input, args.checkpoint, torch.device(args.device), h48)
    rows, pools, tables = h49.load_validation_packet(args.input)
    if len(rows) != 91 or len({row["family_id"] for row in rows}) != 19:
        raise RuntimeError("H57 terminal DL-Val roster mismatch")
    candidate = args.run_root / f"outcomes/cells/{args.arm}/seed{args.seed}/candidate_adapter.pt"
    artifact = torch.load(candidate, map_location="cpu", weights_only=True)
    if (artifact.get("base_checkpoint_sha256") != R882_SHA or
            artifact.get("dope_sha256") != DOPE_SHA or artifact.get("trainable_params") != TRAINABLE_PARAMS):
        raise RuntimeError("H57 terminal adapter artifact pin mismatch")
    before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    h55.set_adapter(net, artifact["adapter_trainable_state"])
    net.eval()
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
    predictor = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    result = h48.evaluate_utility([row["context_sha256"] for row in rows], pools, predictor)
    after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    if before != DOPE_SHA or after != DOPE_SHA:
        raise RuntimeError("H57 terminal DL-Val DOPE checksum mismatch")
    payload = {"schema": "jope.h57.terminal-val/v1", "arm": args.arm, "seed": args.seed,
               "terminal_dl_val_score_count": 1, "validation_payloads_opened": True,
               "family_utility": result["mean"], "context_utility": result["context_mean"],
               "family_delta_vs_r882_harness": result["mean"] - 1.1536152769912449,
               "family_utilities": result["per_family"], "context_utilities": result["per_context"],
               "candidate_adapter_sha256": sha256(candidate), "dope_sha256_before": before,
               "dope_sha256_after": after, "holdout_payloads_opened": 0,
               "outer_rows_delivered": 0, "source_manifest_sha256": MANIFEST_SHA,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    atomic_json(args.out, payload)
    return payload


def run_dry(args) -> dict:
    torch.set_num_threads(1)
    ctx, h55 = make_context(torch.device("cpu"), args.input, args.checkpoint)
    template = h55.adapter_state(ctx["net"])
    names = tuple(sorted(template))
    center = adapter_vector(template, names)
    noises, candidates = antithetic_population(center, SIGMAS["sigma001"], stream_seed(15, "sigma001", 1, False))
    ranks = centered_ranks(list(range(POPULATION)))
    gradient = es_gradient(noises, ranks, SIGMAS["sigma001"])
    result = {"schema": "jope.h57.dry-run/v1", "status": "pass", "population": len(candidates),
              "antithetic_pairs": len(noises), "adapter_parameter_count": center.numel(),
              "fold_seed": FOLD_SEED, "fold_assignment_sha256": ctx["fold_sha"],
              "fold_b_scored": False, "validation_payloads_opened": False,
              "optimizer_updates": 0, "synthetic_gradient_l2": float(torch.linalg.vector_norm(gradient)),
              "dope_sha256": ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)}
    atomic_json(args.out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "timing", "outcome", "fold-b", "terminal-val"), required=True)
    parser.add_argument("--arm", choices=tuple(SIGMAS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--null-arm", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=Path.home() / "jope-data/h57/h57-wave")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "dry-run":
        result = run_dry(args)
    elif args.mode == "fold-b":
        result = score_fold_b(args)
    elif args.mode == "terminal-val":
        if args.arm is None or args.seed is None:
            raise RuntimeError("H57 terminal DL-Val requires arm and seed")
        result = run_terminal_val(args)
    else:
        if args.arm is None or args.seed is None:
            raise RuntimeError("H57 timing/outcome needs a registered arm and seed")
        if args.out.exists():
            raise RuntimeError(f"refusing to overwrite H57 receipt {args.out}")
        args.out.mkdir(parents=True, exist_ok=False)
        result = run_train(args)
    print(json.dumps({"mode": args.mode, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
