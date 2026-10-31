#!/usr/bin/env python3
"""Leakage-guarded H77 long direct-utility ES cells (ES vs ESH with family-harm penalty)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
H57_RUNNER = ROOT / "research/flywheel_selfplay/h57/runner.py"
H55 = ROOT / "research/flywheel_selfplay/h55/runner.py"
H48 = ROOT / "research/flywheel_selfplay/h48/run.py"
H49 = ROOT / "research/flywheel_selfplay/h49/score.py"

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
FUSED_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"

FOLD_SEED = 4900
TRAINABLE_PARAMS = 29_211
SEEDS = (18, 19, 20)
SIGMA = 0.001
ARMS = ("ES", "ESH")
NULL_ARM = "null_s001"
POPULATION = 64
PAIRS = POPULATION // 2
DEFAULT_GENERATIONS = 24
ES_LEARNING_RATE = 0.01
ESH_LAMBDA = 1.0
HARM_THRESHOLD = -0.01
MAX_DISK_USED_PERCENT = 95.0

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


h57_base = import_module("h77_h57_base", H57_RUNNER)
sha256 = h57_base.sha256
adapter_vector = h57_base.adapter_vector
vector_state = h57_base.vector_state


def check_disk_abort(statvfs_fn=os.statvfs) -> dict[str, float | int]:
    """Abort immediately if xbabe2 (/) disk usage exceeds 95% or free space < 1 GiB."""
    stats = statvfs_fn("/")
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    used_percent = 100.0 * (stats.f_blocks - stats.f_bavail) / stats.f_blocks if stats.f_blocks else 100.0
    if available < 1024**3:
        raise RuntimeError(f"xbabe2 disk available below 1 GiB ({available} bytes)")
    if used_percent > MAX_DISK_USED_PERCENT or used_percent >= MAX_DISK_USED_PERCENT:
        raise RuntimeError(f"xbabe2 disk usage at or above 95% abort threshold ({used_percent:.2f}%)")
    return {"total_bytes": total, "available_bytes": available, "used_percent": used_percent}


def atomic_json(path: Path, value: dict, *, statvfs_fn=os.statvfs) -> None:
    check_disk_abort(statvfs_fn=statvfs_fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(path: Path) -> Path:
    """Reject sealed/holdout/meta-test/outer or live xbabe1 flywheel paths before opening."""
    path = Path(path)
    forbidden = ("sealed", "holdout", "meta-test", "meta_test", "outer", "flywheel-full-20261030")
    for candidate in (path.absolute(), path.resolve()):
        for part in candidate.parts:
            low = part.lower()
            if any(token in low for token in forbidden):
                raise RuntimeError(f"H77 rejected protected or live-loop path: {path}")
    return path


def validate_roster_rows(input_dir: Path, rows: list[dict], *, allow_val: bool = False) -> None:
    allowed_splits = ("DL-Train", "DL-Val") if allow_val else ("DL-Train",)
    for row in rows:
        cid = str(row.get("context_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise RuntimeError(f"H77 invalid context_sha256: {cid!r}")
        if row.get("task") != "regression" or row.get("dl_split") not in allowed_splits:
            raise RuntimeError(f"H77 unapproved roster row split={row.get('dl_split')!r} task={row.get('task')!r}")
        if row.get("has_meta_test") is True or row.get("origin") not in (None, "real"):
            raise RuntimeError(f"H77 sealed or non-real row rejected: {cid}")
        for directory, suffix in (("pools", ".npz"), ("tables", ".jtt1")):
            safe_path(input_dir / directory / f"{cid}{suffix}")


def stage_r882_copy(source_checkpoint: Path, scratch_dir: Path) -> Path:
    """Stage an isolated copy of round-882.pt and verify SHA256 before and after copying."""
    safe_path(source_checkpoint)
    safe_path(scratch_dir)
    if sha256(source_checkpoint) != R882_SHA:
        raise RuntimeError("H77 source r882 checkpoint SHA mismatch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    staged = scratch_dir / f"r882-copy-{os.getpid()}.pt"
    shutil.copy2(source_checkpoint, staged)
    if sha256(staged) != R882_SHA:
        staged.unlink(missing_ok=True)
        raise RuntimeError("H77 staged r882 copy SHA mismatch")
    return staged


def validate_trainable_inventory(named_parameters) -> tuple[str, ...]:
    """Ensure only the 29,211-parameter adapter_10tok tensors are trainable."""
    trainable_names = []
    total = 0
    for name, param in named_parameters:
        if param.requires_grad:
            if not name.startswith("adapter_10tok.") or "anchor" in name:
                raise RuntimeError(f"H77 non-adapter or anchor parameter marked trainable: {name}")
            trainable_names.append(name)
            total += int(param.numel())
    if total != TRAINABLE_PARAMS or not trainable_names:
        raise RuntimeError(f"H77 trainable adapter parameter count mismatch: {total} != {TRAINABLE_PARAMS}")
    return tuple(sorted(trainable_names))


def centered_ranks(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (POPULATION,) or not np.isfinite(values).all():
        raise ValueError(f"H77 requires {POPULATION} finite direct-utility scores")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(POPULATION, dtype=np.float64)
    ranks[order] = (np.arange(POPULATION, dtype=np.float64) - (POPULATION - 1) / 2.0) / POPULATION
    return ranks


def stream_seed(seed: int, generation: int, *, arm: str = "ES", null_arm: bool = False) -> int:
    if seed not in SEEDS or arm not in ARMS or generation < 1:
        raise ValueError("unregistered H77 random stream")
    if null_arm and (arm, seed) != ("ES", 18):
        raise ValueError("only ES seed 18 has a preregistered permuted-fitness NULL stream")
    # ES, ESH, and NULL share the exact perturbation stream for a given (seed, generation).
    return seed * 1_000_003 + 1 * 97_409 + generation * 1_009


def antithetic_population(
    center: torch.Tensor, sigma: float, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if center.ndim != 1 or center.numel() != TRAINABLE_PARAMS or abs(sigma - SIGMA) > 1e-12:
        raise ValueError("H77 center must be the flat 29,211-parameter adapter with sigma=0.001")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noises = [torch.randn(center.shape, generator=generator, dtype=center.dtype) for _ in range(PAIRS)]
    candidates = [center + sign * sigma * noise for noise in noises for sign in (1.0, -1.0)]
    return noises, candidates


def es_gradient(noises: list[torch.Tensor], ranks: np.ndarray, sigma: float = SIGMA) -> torch.Tensor:
    if len(noises) != PAIRS or ranks.shape != (POPULATION,) or abs(sigma - SIGMA) > 1e-12:
        raise ValueError("invalid H77 antithetic gradient inputs")
    gradient = torch.zeros_like(noises[0])
    for pair, noise in enumerate(noises):
        coefficient = float(ranks[2 * pair] - ranks[2 * pair + 1]) / (2.0 * POPULATION * sigma)
        gradient.add_(noise, alpha=coefficient)
    return gradient


def compute_fold_a_fitness(
    per_family: dict[str, float],
    baseline_per_family: dict[str, float],
    arm: str,
    *,
    esh_lambda: float = ESH_LAMBDA,
) -> dict[str, float]:
    """
    Compute family-balanced fold-A fitness for ES or ESH:
      - ES:  mean_f(u_f)
      - ESH: mean_f(u_f) - lambda * mean_f(|min(0, gain_f + 0.01)|)
             == mean_f(u_f) + lambda * mean_f(min(0, gain_f + 0.01)), lambda = 1.0 fixed a priori
    where gain_f = u_f(candidate) - u_f(r882) and each family in Fold A has equal weight.
    When gain_f < -0.01, min(0, gain_f + 0.01) < 0, so ESH strictly penalizes harmed families (ESH <= ES).
    """
    if arm not in ARMS:
        raise ValueError(f"unregistered H77 arm: {arm!r}")
    if not per_family or set(per_family) != set(baseline_per_family):
        raise ValueError("H77 fold-A family keys mismatch between candidate and r882 baseline")
    families = sorted(per_family)
    u_vals = np.asarray([float(per_family[f]) for f in families], dtype=np.float64)
    b_vals = np.asarray([float(baseline_per_family[f]) for f in families], dtype=np.float64)
    if not (np.isfinite(u_vals).all() and np.isfinite(b_vals).all()):
        raise ValueError("H77 fold-A family utilities must be finite")
    gains = u_vals - b_vals
    mean_u = float(np.mean(u_vals))
    harm_clip = np.minimum(0.0, gains - HARM_THRESHOLD)  # min(0, gain_f + 0.01) <= 0
    harm_term = float(np.mean(harm_clip))
    harm_penalty = float(-esh_lambda * harm_term)  # >= 0 magnitude subtracted from mean_u
    harm_fraction = float(np.mean(gains < HARM_THRESHOLD))
    fitness = mean_u if arm == "ES" else float(mean_u - harm_penalty)
    return {
        "fitness": fitness,
        "family_mean_utility": mean_u,
        "harm_term_min0_gain_plus_001": harm_term,
        "harm_penalty_subtracted": 0.0 if arm == "ES" else harm_penalty,
        "fold_a_harm_fraction": harm_fraction,
    }


def make_context(device: torch.device, input_dir: Path, staged_checkpoint: Path) -> tuple[dict, object]:
    safe_path(input_dir)
    safe_path(staged_checkpoint)
    for name, expected_sha in (
        ("source_manifest.json", MANIFEST_SHA),
        ("selected.json", SELECTED_SHA),
        ("cpu_exact_fused.py", FUSED_SHA),
    ):
        fpath = safe_path(input_dir / name)
        if sha256(fpath) != expected_sha:
            raise RuntimeError(f"H77 input file SHA mismatch for {name}")
    if sha256(staged_checkpoint) != R882_SHA:
        raise RuntimeError("H77 r882 checkpoint SHA mismatch")
    h55 = import_module("h77_h55", H55)
    ctx = h55.prepare_training(input_dir, staged_checkpoint, device)
    ctx["input_dir"] = input_dir
    validate_trainable_inventory(ctx["net"].named_parameters())
    if sum(p.numel() for p in ctx["trainable"]) != TRAINABLE_PARAMS:
        raise RuntimeError("H77 adapter parameter count mismatch")
    ctx["net"].eval()
    dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if dope != DOPE_SHA:
        raise RuntimeError("H77 DOPE guard mismatch")
    return ctx, h55


def score_train_fitness(
    ctx: dict,
    h55,
    vector: torch.Tensor,
    template: dict[str, torch.Tensor],
    names: tuple[str, ...],
    baseline_per_family: dict[str, float] | None = None,
    arm: str = "ES",
) -> tuple[float, dict[str, float], dict[str, float]]:
    h55.set_adapter(ctx["net"], vector_state(vector, template, names))
    ctx["net"].eval()
    ids = sorted(ctx["pools_a"])
    result = ctx["h48"].evaluate_utility(ids, ctx["pools_a"], ctx["predictor_a"])
    per_family = result["per_family"]
    if len(per_family) != 164:
        raise RuntimeError("H77 fold-A family count changed")
    ref_family = per_family if baseline_per_family is None else baseline_per_family
    fit_info = compute_fold_a_fitness(per_family, ref_family, arm=arm)
    return fit_info["fitness"], per_family, fit_info


def run_generation(
    ctx: dict,
    h55,
    theta: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    template: dict[str, torch.Tensor],
    names: tuple[str, ...],
    baseline_per_family: dict[str, float],
    *,
    seed: int,
    arm: str,
    generation: int,
    null_arm: bool,
) -> tuple[dict, float]:
    check_disk_abort()
    before_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if before_dope != DOPE_SHA:
        raise RuntimeError("DOPE mismatch before H77 generation")
    started = time.monotonic()
    gen_seed = stream_seed(seed, generation, arm=arm, null_arm=null_arm)
    noises, candidates = antithetic_population(theta.detach().cpu(), SIGMA, gen_seed)
    scores: list[float] = []
    raw_means: list[float] = []
    harm_terms: list[float] = []
    for candidate in candidates:
        score, _, fit_info = score_train_fitness(
            ctx, h55, candidate, template, names, baseline_per_family=baseline_per_family, arm=arm
        )
        scores.append(score)
        raw_means.append(fit_info["family_mean_utility"])
        harm_terms.append(fit_info["harm_term_min0_gain_plus_001"])
    ranks = centered_ranks(scores)
    if null_arm:
        rng = np.random.default_rng(gen_seed + 17)
        ranks = ranks[rng.permutation(POPULATION)]
    gradient = es_gradient(noises, ranks, SIGMA)
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    old_theta = theta.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    theta.grad = -gradient.to(theta)
    optimizer.step()
    update_l2 = float(torch.linalg.vector_norm(theta.detach() - old_theta))
    after_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if after_dope != DOPE_SHA:
        raise RuntimeError("DOPE changed during H77 generation")
    check_disk_abort()
    elapsed = time.monotonic() - started
    row = {
        "generation": generation,
        "arm": arm,
        "population": POPULATION,
        "antithetic_pairs": PAIRS,
        "sigma": SIGMA,
        "esh_lambda": ESH_LAMBDA if arm == "ESH" else 0.0,
        "fitness_mean": float(np.mean(scores)),
        "fitness_median": float(np.median(scores)),
        "fitness_min": float(np.min(scores)),
        "fitness_max": float(np.max(scores)),
        "raw_family_utility_mean": float(np.mean(raw_means)),
        "harm_term_mean": float(np.mean(harm_terms)),
        "es_gradient_l2": gradient_norm,
        "mean_theta_update_l2": update_l2,
        "rank_fitness_permuted": null_arm,
        "dope_sha256_before": before_dope,
        "dope_sha256_after": after_dope,
        "generation_wall_seconds": elapsed,
    }
    return row, elapsed


def source_hashes(input_dir: Path) -> dict[str, str]:
    paths = [
        H55,
        H48,
        H49,
        H57_RUNNER,
        ROOT / "research/flywheel_selfplay/h46/run.py",
        HERE / "PREREG.md",
        HERE / "runner.py",
        input_dir / "cpu_exact_fused.py",
    ]
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
        for path in paths
    }


def project_wave_wall_from_timing(
    measured_generation_seconds: float,
    *,
    cap_seconds: float = 9000.0,
    outcome_cells: int = 7,
    generations_per_cell: int = DEFAULT_GENERATIONS,
    overhead_generations_equiv: float = 2.0,
) -> dict[str, float | int | bool]:
    """Project total GPU cell wall for 7 x 24-generation cells + Fold-B/terminal overhead against <=2.5h cap."""
    if measured_generation_seconds <= 0 or not np.isfinite(measured_generation_seconds):
        raise ValueError("measured_generation_seconds must be positive and finite")
    total_generations = outcome_cells * generations_per_cell
    projected_seconds = float(measured_generation_seconds * (total_generations + overhead_generations_equiv))
    fits = bool(projected_seconds <= cap_seconds)
    return {
        "measured_generation_seconds": float(measured_generation_seconds),
        "outcome_cells": outcome_cells,
        "generations_per_cell": generations_per_cell,
        "total_outcome_generations": total_generations,
        "overhead_generations_equiv": float(overhead_generations_equiv),
        "projected_total_seconds": projected_seconds,
        "cap_seconds": float(cap_seconds),
        "fits_within_cap": fits,
    }


def run_train(args) -> dict:
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H77 timing/outcome generations require cuda:0 and qualification.lock")
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise RuntimeError("unregistered H77 candidate arm or seed")
    if args.null_arm and (args.arm, args.seed, args.mode) != ("ES", 18, "outcome"):
        raise RuntimeError("only the ES seed18 outcome may be the preregistered permuted-fitness null")
    if args.mode == "timing" and (args.arm, args.seed, args.generations, args.null_arm) != ("ES", 18, 1, False):
        raise RuntimeError("only ES seed18 one-generation timing is preregistered")
    if args.mode == "outcome" and args.generations != DEFAULT_GENERATIONS:
        raise RuntimeError(f"H77 outcome horizon must be fixed at {DEFAULT_GENERATIONS} generations")
    check_disk_abort()
    torch.set_num_threads(1)
    scratch_dir = args.out / "_scratch"
    staged_ckpt = stage_r882_copy(args.checkpoint, scratch_dir)
    try:
        ctx, h55 = make_context(torch.device(args.device), args.input, staged_ckpt)
        template = h55.adapter_state(ctx["net"])
        names = tuple(sorted(template))
        theta0 = adapter_vector(template, names)
        if theta0.numel() != TRAINABLE_PARAMS:
            raise RuntimeError("H77 flat r882 adapter length mismatch")
        theta = torch.nn.Parameter(theta0.clone())
        optimizer = torch.optim.Adam([theta], lr=ES_LEARNING_RATE)
        dope_before = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
        baseline_a, baseline_per_family, _ = score_train_fitness(
            ctx, h55, theta0, template, names, baseline_per_family=None, arm="ES"
        )
        generations = []
        started = time.monotonic()
        for generation in range(1, args.generations + 1):
            row, _ = run_generation(
                ctx,
                h55,
                theta,
                optimizer,
                template,
                names,
                baseline_per_family,
                seed=args.seed,
                arm=args.arm,
                generation=generation,
                null_arm=args.null_arm,
            )
            generations.append(row)
            print(json.dumps({"arm": args.arm, "seed": args.seed, **row}, sort_keys=True), flush=True)
        # Final mean theta only — no best-generation pick
        final_state = vector_state(theta.detach().cpu(), template, names)
        adapter_sha = h55.adapter_digest(final_state)
        wall = time.monotonic() - started
        dope_after = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
        if dope_before != DOPE_SHA or dope_after != DOPE_SHA:
            raise RuntimeError("H77 DOPE checksum changed across cell")
        payload = {
            "schema": "jope.h77.cell/v1",
            "status": "complete",
            "mode": args.mode,
            "arm": args.arm,
            "cell_id": NULL_ARM if args.null_arm else args.arm,
            "seed": args.seed,
            "null_arm": args.null_arm,
            "generations_completed": len(generations),
            "evaluated_perturbations": len(generations) * POPULATION,
            "population": POPULATION,
            "antithetic_pairs": PAIRS,
            "sigma": SIGMA,
            "esh_lambda": ESH_LAMBDA if args.arm == "ESH" else 0.0,
            "adam_learning_rate": ES_LEARNING_RATE,
            "selection": "final ES mean theta after generation 24 only; no best-generation or sampled-candidate pick",
            "fold_seed": FOLD_SEED,
            "fold_assignment": ctx["fold_assignment"],
            "fold_assignment_sha256": ctx["fold_sha"],
            "fold_a_family_count": len(ctx["fold_a_families"]),
            "fold_b_scored": False,
            "validation_payloads_opened": False,
            "baseline_fold_a_family_utility": baseline_a,
            "generations": generations,
            "generation_wall_seconds": [r["generation_wall_seconds"] for r in generations],
            "wall_seconds": wall,
            "adapter_sha256": adapter_sha,
            "r882_checkpoint_sha256": R882_SHA,
            "source_manifest_sha256": MANIFEST_SHA,
            "selected_sha256": SELECTED_SHA,
            "prereg_sha256": sha256(HERE / "PREREG.md"),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes(args.input),
            "dope_sha256_before": dope_before,
            "dope_sha256_after": dope_after,
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "update_norms": [r["mean_theta_update_l2"] for r in generations],
            "gradient_norms": [r["es_gradient_l2"] for r in generations],
        }
        if args.mode == "timing":
            payload["timing_projection"] = project_wave_wall_from_timing(
                generations[0]["generation_wall_seconds"]
            )
        if args.mode == "outcome":
            check_disk_abort()
            ctx["h48"].save_adapter_snapshot(args.out / "candidate_adapter.pt", final_state, R882_SHA)
            payload["candidate_adapter_artifact"] = str(args.out / "candidate_adapter.pt")
            payload["candidate_adapter_artifact_sha256"] = sha256(args.out / "candidate_adapter.pt")
        atomic_json(args.out / ("timing.json" if args.mode == "timing" else "summary.json"), payload)
        return payload
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def score_fold_b(args) -> dict:
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H77 fold-B admission scoring requires cuda:0 and qualification.lock")
    check_disk_abort()
    safe_path(args.input)
    safe_path(args.checkpoint)
    scratch_dir = args.out / "_scratch"
    staged_ckpt = stage_r882_copy(args.checkpoint, scratch_dir)
    try:
        h55 = import_module("h77_h55_score", H55)
        h48 = import_module("h77_h48_score", H48)
        h48.CHECKPOINT_SHA = R882_SHA
        selected, train_rows, _ = h55.input_metadata(args.input, h48)
        validate_roster_rows(args.input, train_rows, allow_val=False)
        folds = h55.family_folds(train_rows, h48)
        fold_b_families = folds[4]
        rows_b = [row for row in train_rows if row["family_id"] in fold_b_families]
        fold_a_families = set().union(*folds[:4])
        rows_a_count = sum(row["family_id"] in fold_a_families for row in train_rows)
        if not rows_b or rows_a_count + len(rows_b) != len(train_rows):
            raise RuntimeError("H77 fold-A/B assignment is empty or incomplete")
        fold_assignment = {
            "seed": FOLD_SEED,
            "folds": [sorted(f) for f in folds],
            "fold_a_families": sorted(fold_a_families),
            "fold_b_families": sorted(fold_b_families),
            "train_contexts_a": rows_a_count,
            "train_contexts_b": len(rows_b),
        }
        fold_sha = hashlib.sha256(
            json.dumps(fold_assignment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        pools, tables = h55.load_rows(args.input, rows_b, h48)
        net, _, guard, _ = h55.build_model(args.input, staged_ckpt, torch.device(args.device), h48)
        validate_trainable_inventory(net.named_parameters())
        tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
        predict = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
        ids = [row["context_sha256"] for row in rows_b]
        dope_before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        net.eval()
        baseline = h48.evaluate_utility(ids, pools, predict)
        candidates = []
        for path in sorted(args.run_root.glob("outcomes/cells/*/seed*/candidate_adapter.pt")):
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if (
                artifact.get("base_checkpoint_sha256") != R882_SHA
                or artifact.get("dope_sha256") != DOPE_SHA
                or artifact.get("trainable_params") != TRAINABLE_PARAMS
            ):
                raise RuntimeError(f"H77 candidate artifact pin mismatch: {path}")
            h55.set_adapter(net, artifact["adapter_trainable_state"])
            net.eval()
            result = h48.evaluate_utility(ids, pools, predict)
            if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
                raise RuntimeError("DOPE changed during H77 fold-B candidate scoring")
            summary_path = path.parent / "summary.json"
            summary = json.loads(summary_path.read_text())
            candidates.append(
                {
                    "arm": summary["arm"],
                    "seed": summary["seed"],
                    "adapter_sha256": sha256(path),
                    "family_utility": result["per_family"],
                    "context_utility": result["per_context"],
                    "family_gain_vs_r882": {
                        family: result["per_family"][family] - baseline["per_family"][family]
                        for family in baseline["per_family"]
                    },
                    "fold_b_scoring_passes": 1,
                }
            )
        null_diagnostic = None
        null_path = args.run_root / "outcomes/null/seed18/candidate_adapter.pt"
        if null_path.exists():
            null_art = torch.load(null_path, map_location="cpu", weights_only=True)
            h55.set_adapter(net, null_art["adapter_trainable_state"])
            net.eval()
            null_res = h48.evaluate_utility(ids, pools, predict)
            if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
                raise RuntimeError("DOPE changed during H77 fold-B null diagnostic scoring")
            null_diagnostic = {
                "cell_id": NULL_ARM,
                "seed": 18,
                "adapter_sha256": sha256(null_path),
                "family_utility": null_res["per_family"],
                "context_utility": null_res["per_context"],
                "family_gain_vs_r882": {
                    family: null_res["per_family"][family] - baseline["per_family"][family]
                    for family in baseline["per_family"]
                },
                "diagnostic_only": True,
                "not_admissible": True,
            }
        dope_after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        result = {
            "schema": "jope.h77.fold-b/v1",
            "status": "complete",
            "fold_seed": FOLD_SEED,
            "fold_assignment": fold_assignment,
            "fold_assignment_sha256": fold_sha,
            "r882_checkpoint_sha256": R882_SHA,
            "source_manifest_sha256": MANIFEST_SHA,
            "dope_sha256_before": dope_before,
            "dope_sha256_after": dope_after,
            "baseline_family_utility": baseline["per_family"],
            "baseline_context_utility": baseline["per_context"],
            "candidates": candidates,
            "candidate_count": len(candidates),
            "fold_b_read_count": len(candidates),
            "null_diagnostic": null_diagnostic,
            "validation_payloads_opened": False,
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes(args.input),
        }
        atomic_json(args.run_root / "outcomes/fold_b.json", result)
        return result
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_terminal_val(args) -> dict:
    admission = json.loads((args.run_root / "outcomes/admission.json").read_text())
    if admission.get("arms", {}).get(args.arm, {}).get("admitted") is not True:
        raise RuntimeError("H77 terminal DL-Val is forbidden for a nonadmitted arm")
    if args.arm not in ARMS or args.seed not in SEEDS or not args.lock_held or args.device != "cuda:0":
        raise RuntimeError("H77 terminal DL-Val requires a registered arm, seed, cuda:0, and qualification.lock")
    if args.out.exists():
        raise RuntimeError("refusing duplicate H77 terminal DL-Val receipt")
    check_disk_abort()
    scratch_dir = args.out.parent / f"_scratch_{args.arm}_{args.seed}"
    staged_ckpt = stage_r882_copy(args.checkpoint, scratch_dir)
    try:
        h55 = import_module("h77_h55_val", H55)
        h48 = import_module("h77_h48_val", H48)
        h48.CHECKPOINT_SHA = R882_SHA
        h49 = import_module("h77_h49_val", H49)
        net, _, guard, _ = h55.build_model(args.input, staged_ckpt, torch.device(args.device), h48)
        rows, pools, tables = h49.load_validation_packet(args.input)
        validate_roster_rows(args.input, rows, allow_val=True)
        if len(rows) != 91 or len({row["family_id"] for row in rows}) != 19:
            raise RuntimeError("H77 terminal DL-Val roster mismatch")
        candidate = args.run_root / f"outcomes/cells/{args.arm}/seed{args.seed}/candidate_adapter.pt"
        artifact = torch.load(candidate, map_location="cpu", weights_only=True)
        if (
            artifact.get("base_checkpoint_sha256") != R882_SHA
            or artifact.get("dope_sha256") != DOPE_SHA
            or artifact.get("trainable_params") != TRAINABLE_PARAMS
        ):
            raise RuntimeError("H77 terminal adapter artifact pin mismatch")
        before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        h55.set_adapter(net, artifact["adapter_trainable_state"])
        net.eval()
        tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
        predictor = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
        result = h48.evaluate_utility([row["context_sha256"] for row in rows], pools, predictor)
        after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        if before != DOPE_SHA or after != DOPE_SHA:
            raise RuntimeError("H77 terminal DL-Val DOPE checksum mismatch")
        payload = {
            "schema": "jope.h77.terminal-val/v1",
            "arm": args.arm,
            "seed": args.seed,
            "terminal_dl_val_score_count": 1,
            "validation_payloads_opened": True,
            "family_utility": result["mean"],
            "context_utility": result["context_mean"],
            "family_delta_vs_r882_harness": result["mean"] - 1.1536152769912449,
            "family_utilities": result["per_family"],
            "context_utilities": result["per_context"],
            "candidate_adapter_sha256": sha256(candidate),
            "dope_sha256_before": before,
            "dope_sha256_after": after,
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "source_manifest_sha256": MANIFEST_SHA,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        atomic_json(args.out, payload)
        return payload
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_dry(args) -> dict:
    torch.set_num_threads(1)
    scratch_dir = args.out.parent / "_scratch_dry"
    staged_ckpt = stage_r882_copy(args.checkpoint, scratch_dir)
    try:
        ctx, h55 = make_context(torch.device("cpu"), args.input, staged_ckpt)
        template = h55.adapter_state(ctx["net"])
        names = tuple(sorted(template))
        center = adapter_vector(template, names)
        noises, candidates = antithetic_population(center, SIGMA, stream_seed(18, 1, arm="ES", null_arm=False))
        ranks = centered_ranks(list(range(POPULATION)))
        gradient = es_gradient(noises, ranks, SIGMA)
        result = {
            "schema": "jope.h77.dry-run/v1",
            "status": "pass",
            "population": len(candidates),
            "antithetic_pairs": len(noises),
            "sigma": SIGMA,
            "esh_lambda": ESH_LAMBDA,
            "adapter_parameter_count": center.numel(),
            "fold_seed": FOLD_SEED,
            "fold_assignment_sha256": ctx["fold_sha"],
            "fold_b_scored": False,
            "validation_payloads_opened": False,
            "optimizer_updates": 0,
            "synthetic_gradient_l2": float(torch.linalg.vector_norm(gradient)),
            "dope_sha256": ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA),
        }
        atomic_json(args.out, result)
        return result
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "timing", "outcome", "fold-b", "terminal-val"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--null-arm", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=Path.home() / "jope-data/h77/h77-wave")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "dry-run":
        result = run_dry(args)
    elif args.mode == "fold-b":
        result = score_fold_b(args)
    elif args.mode == "terminal-val":
        if args.arm is None or args.seed is None:
            raise RuntimeError("H77 terminal DL-Val requires arm and seed")
        result = run_terminal_val(args)
    else:
        if args.arm is None or args.seed is None:
            raise RuntimeError("H77 timing/outcome needs a registered arm and seed")
        if args.out.exists():
            raise RuntimeError(f"refusing to overwrite H77 receipt {args.out}")
        args.out.mkdir(parents=True, exist_ok=False)
        result = run_train(args)
    print(json.dumps({"mode": args.mode, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
