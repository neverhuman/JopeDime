#!/usr/bin/env python3
"""Train H49 arms using inner family CV only; DL-Val is reserved for score.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H48_DIR = HERE.parent / "h48"
SPEC = importlib.util.spec_from_file_location("h49_h48_run", H48_DIR / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the reviewed H48 utility/fold helpers")
h48 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h48
SPEC.loader.exec_module(h48)

DEVICE = "cuda:0"
FOLD_SEED = 4900
ROUNDS = 16
UPDATES_PER_ROUND = 2
SWA_CYCLE_UPDATES = ROUNDS * UPDATES_PER_ROUND
SOURCE_MANIFEST_SHA = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"


def _sam_update(args, net, optimizer, trainable, pools, family_by_cid, predict, batch, indexes):
    """One persistent ASAM update with rho=.05 and H46-0 Huber+KL objective."""
    optimizer.zero_grad(set_to_none=True)
    first = h48.training_loss(
        "0", predict(batch, indexes), pools, batch, indexes, family_by_cid
    )
    first.backward()
    adaptive = {
        id(parameter): parameter.detach().abs().clamp_min(1e-3)
        for parameter in trainable
    }
    norm = torch.sqrt(sum(
        torch.sum((adaptive[id(parameter)] * parameter.grad.detach()) ** 2)
        for parameter in trainable if parameter.grad is not None
    )).clamp_min(1e-12)
    perturbations = {}
    with torch.no_grad():
        for parameter in trainable:
            if parameter.grad is None:
                continue
            e_w = (adaptive[id(parameter)] ** 2) * parameter.grad * (0.05 / norm)
            parameter.add_(e_w)
            perturbations[id(parameter)] = e_w
    optimizer.zero_grad(set_to_none=True)
    second = h48.training_loss("0", predict(batch, indexes), pools, batch, indexes, family_by_cid)
    second.backward()
    with torch.no_grad():
        for parameter in trainable:
            if id(parameter) in perturbations:
                parameter.sub_(perturbations[id(parameter)])
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    optimizer.step()
    return float(second.detach().cpu()), grad_norm


def _prepare(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(DEVICE)
    if not torch.cuda.is_available():
        raise RuntimeError("H49 requires cuda:0; CPU fallback is prohibited")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.90)
    if h48.base.psutil.virtual_memory().available < 20 * 1024**3:
        raise RuntimeError("available RAM below 20 GiB before H49 seed")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 20 * 1024**3:
        raise RuntimeError("cuda:0 free memory below 20 GiB before H49 seed")
    if h48.base.psutil.cpu_percent(interval=1) > 85:
        raise RuntimeError("host CPU exceeds 85% before H49 seed")
    selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = h48.build_model(
        args.input, args.checkpoint, device
    )
    args.tokens = tokens
    args.masks = masks
    args.device = device
    args.output.mkdir(parents=True, exist_ok=True)
    return selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard


def train_sam_swa(args) -> dict:
    """Train SAM or SWA with five fixed whole-family folds and a single CV selection per fold."""
    selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = _prepare(args)
    predict = h48.make_predictor(net, pools, tokens, masks, args.device)
    checkpoint_state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    fold_sets = h48.family_folds(
        [pools[cid]["rec"] for cid in train_ids], count=5, seed=FOLD_SEED
    )
    folds = []
    whole_start = time.monotonic()
    for fold_i, held_families in enumerate(fold_sets):
        held = [cid for cid in train_ids if family_by_cid[cid] in held_families]
        fit = [cid for cid in train_ids if family_by_cid[cid] not in held_families]
        net.load_state_dict(checkpoint_state["state_dict"], strict=True)
        h48.h46.checkpoint_guard(h48.base.guard, net, expected=h48.DOPE_SHA)
        base_eval = h48.evaluate_utility(held, pools, predict)
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
        h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        if args.arm == "SWA":
            for group in optimizer.param_groups:
                group["lr"] = 2e-5
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=2e-5, max_lr=1e-4, step_size_up=16,
                cycle_momentum=False,
            )
            avg = h48.parameter_snapshot(net)
            for name in avg:
                avg[name].zero_()
            avg_count = 0
        incumbent = h48.parameter_snapshot(net)
        best_weights = incumbent
        best_eval = base_eval
        best_round = 0
        trajectory = []
        t0 = time.monotonic()
        last_loss = 0.0
        last_norm = 0.0
        for round_i in range(ROUNDS):
            for update_i in range(UPDATES_PER_ROUND):
                net.train()
                batch_seed = args.seed * 1_000_003 + fold_i * 10_007 + round_i * 31 + update_i
                batch = h48.family_balanced_context_batch(fit, family_by_cid, 256, batch_seed)
                indexes = [h48.base.training_indices(pools[cid], 192) for cid in batch]
                if args.arm == "SAM":
                    loss_value, last_norm = _sam_update(
                        args, net, optimizer, trainable, pools, family_by_cid,
                        predict, batch, indexes,
                    )
                    last_loss = loss_value
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss = h48.training_loss("0", predict(batch, indexes), pools, batch, indexes, family_by_cid)
                    loss.backward()
                    last_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                    optimizer.step()
                    scheduler.step()
                    last_loss = float(loss.detach().cpu())
                    with torch.no_grad():
                        for name, parameter in net.named_parameters():
                            if parameter.requires_grad:
                                avg[name].add_(parameter.detach())
                    avg_count += 1
            net.eval()
            h48.h46.checkpoint_guard(h48.base.guard, net, expected=h48.DOPE_SHA)
            if args.arm == "SWA":
                if round_i != ROUNDS - 1 or avg_count != SWA_CYCLE_UPDATES:
                    continue
                candidate_weights = {name: value / avg_count for name, value in avg.items()}
                h48.load_adapter_snapshot(net, candidate_weights)
                candidate_eval = h48.evaluate_utility(held, pools, predict)
                best_round = "full-cycle"
            else:
                candidate_eval = h48.evaluate_utility(held, pools, predict)
                candidate_weights = h48.parameter_snapshot(net)
                new_best = candidate_eval["mean"] > best_eval["mean"]
                if new_best:
                    best_eval = candidate_eval
                    best_weights = candidate_weights
                    best_round = round_i + 1
                trajectory.append({
                    "round": round_i + 1,
                    "loss": last_loss,
                    "grad_norm": last_norm,
                    "inner_cv_family_utility": candidate_eval["mean"],
                    "inner_cv_context_utility": candidate_eval["context_mean"],
                    "new_inner_cv_best": new_best,
                    "elapsed_seconds": time.monotonic() - t0,
                })
        if args.arm == "SWA":
            best_eval = candidate_eval
            best_weights = candidate_weights
            trajectory.append({
                "round": ROUNDS,
                "updates": SWA_CYCLE_UPDATES,
                "schedule": "one full triangular cyclic-LR cycle; average all 32 post-update adapter states",
                "inner_cv_family_utility": best_eval["mean"],
                "inner_cv_context_utility": best_eval["context_mean"],
                "elapsed_seconds": time.monotonic() - t0,
            })
        h48.save_adapter_snapshot(args.output / f"fold{fold_i}-selected.pt", best_weights, h48.digest(args.checkpoint))
        folds.append({
            "fold": fold_i,
            "fit_families": len({family_by_cid[cid] for cid in fit}),
            "fit_contexts": len(fit),
            "heldout_families": sorted(held_families),
            "heldout_contexts": len(held),
            "baseline_family_utility": base_eval["mean"],
            "baseline_per_family_utility": base_eval["per_family"],
            "selected_family_utility": best_eval["mean"],
            "selected_context_utility": best_eval["context_mean"],
            "selected_per_family_utility": best_eval["per_family"],
            "selected_per_context_utility": best_eval["per_context"],
            "selected_candidate_delta": best_eval["mean"] - base_eval["mean"],
            "trajectory": trajectory,
            "selected_round": best_round,
            "wall_seconds": time.monotonic() - t0,
        })
        print(json.dumps({"arm": args.arm, "seed": args.seed, "fold": fold_i, "delta": folds[-1]["selected_candidate_delta"]}), flush=True)
    all_held_families = [f for fold in folds for f in fold["heldout_families"]]
    assert len(all_held_families) == 205 and len(set(all_held_families)) == 205
    baseline_by_family = {k: v for fold in folds for k, v in fold["baseline_per_family_utility"].items()}
    selected_by_family = {k: v for fold in folds for k, v in fold["selected_per_family_utility"].items()}
    cv_base = float(np.mean(list(baseline_by_family.values())))
    cv_selected = float(np.mean(list(selected_by_family.values())))
    result = {
        "schema": "jope.h49.train-arm/v1",
        "arm": args.arm,
        "seed": args.seed,
        "device": DEVICE,
        "fold_seed": FOLD_SEED,
        "selection_metric": "mean held-out whole-family actual utility; train contexts only",
        "objective": "H46-0 Huber+KL; H48-PF was worse than H48-0 on terminal family median",
        "sam_rho": 0.05 if args.arm == "SAM" else None,
        "swa_cycle_updates": SWA_CYCLE_UPDATES if args.arm == "SWA" else None,
        "folds": folds,
        "inner_cv_baseline_family_utility": cv_base,
        "inner_cv_selected_family_utility": cv_selected,
        "inner_cv_selected_delta": cv_selected - cv_base,
        "dope_sha256": h48.DOPE_SHA,
        "checkpoint_sha256": h48.digest(args.checkpoint),
        "trainable_params": h48.TRAINABLE_PARAMS,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "excluded_dl_holdout_count": 93,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "wall_seconds": time.monotonic() - whole_start,
    }
    h48.atomic_json(args.output / "train-result.json", result)
    chosen = max(folds, key=lambda row: row["selected_candidate_delta"])
    h48.atomic_json(args.output / "selected-candidate.json", {
        "arm": args.arm,
        "seed": args.seed,
        "fold": chosen["fold"],
        "selected_family_utility": chosen["selected_family_utility"],
        "selected_delta": chosen["selected_candidate_delta"],
        "checkpoint": f"fold{chosen['fold']}-selected.pt",
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("0", "SAM", "SWA", "ES2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h49/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h49/ckpt/round-785.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h49/runs")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--budget-seconds", type=float, default=141.0)
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if args.device != DEVICE:
        raise RuntimeError("H49 is preregistered for cuda:0 only")
    if h48.digest(args.checkpoint) != h48.CHECKPOINT_SHA:
        raise RuntimeError("H49 checkpoint copy SHA mismatch")
    seed_map = {"0": 0, "SAM": 0, "SWA": 0, "ES2": 3}
    if args.seed not in range(seed_map[args.arm], seed_map[args.arm] + 3):
        raise RuntimeError(f"seed outside preregistered range for {args.arm}")
    if not args.lock_held:
        raise RuntimeError("H49 run_wave.sh must own qualification.lock before launching a seed")
    seed_dir = args.output / f"H49-{args.arm}" / f"seed{args.seed}"
    if (seed_dir / "train-result.json").exists() and (seed_dir / "selected-candidate.json").exists():
        print(f"skip completed {args.arm} seed{args.seed}")
        return
    args.output = seed_dir
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.arm in ("0", "ES2"):
        h48_args = argparse.Namespace(
            input=args.input, checkpoint=args.checkpoint, output=seed_dir, seed=args.seed,
            fold_seed=FOLD_SEED, budget_seconds=args.budget_seconds,
            device=DEVICE, lock_held=args.lock_held,
            arm="0" if args.arm == "0" else "ES",
        )
        result = h48.run_training_arm(h48_args)
        result["h49_arm"] = args.arm
        h48.atomic_json(seed_dir / "train-result.json", result)
    else:
        result = train_sam_swa(args)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "wall_seconds": result.get("wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
