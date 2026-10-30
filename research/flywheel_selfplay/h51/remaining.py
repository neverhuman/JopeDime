#!/usr/bin/env python3
"""H51 DATA, combined CLIP+SMOOTH, and matched-compute NULL arms."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run as h51  # noqa: E402

import steps as shared  # noqa: E402
h48 = shared.h48
h48.CHECKPOINT_SHA = shared.R882_SHA

SEEDS = (9, 10, 11)
K = 16
LR = 1e-4
NULL_N = 16
CLIP = 0.5
SMOOTH_TAU = 0.1


def nested_subsets(input_dir: Path, output: Path) -> dict:
    selected = json.loads((input_dir / "selected.json").read_text())
    rows = selected["train"]
    families = sorted({r["family_id"] for r in rows})
    folds = h48.family_folds(rows, count=5, seed=h51.FOLD_SEED)
    plan = {"schema": "jope.h51.data-subsets/v1", "fold_seed": h51.FOLD_SEED, "seeds": {}}
    for seed in SEEDS:
        plan["seeds"][str(seed)] = {}
        for fold_id, held in enumerate(folds):
            fit_families = sorted(set(families) - held)
            rng = np.random.default_rng(51_000_000 + seed * 1009 + fold_id)
            order = list(np.asarray(fit_families, dtype=object)[rng.permutation(len(fit_families))])
            subsets = {}
            for pct in (25, 50, 100):
                count = len(fit_families) if pct == 100 else max(1, int(round(len(fit_families) * pct / 100)))
                subsets[str(pct)] = sorted(map(str, order[:count]))
            if not set(subsets["25"]).issubset(subsets["50"]) or not set(subsets["50"]).issubset(subsets["100"]):
                raise RuntimeError("DATA subsets are not nested")
            plan["seeds"][str(seed)][str(fold_id)] = {"heldout_families": sorted(held), "fit_families": subsets}
    h51.atomic_json(output / "DATA" / "subset-plan.json", plan)
    return plan


def clipped_smooth_loss(prediction, pools, cids, indexes, family_by_cid):
    device = prediction.device
    target = torch.as_tensor(
        np.stack([pools[cid]["fitness"][ix] for cid, ix in zip(cids, indexes, strict=True)]),
        dtype=torch.float32, device=device,
    )
    gates = torch.as_tensor(
        np.stack([pools[cid]["gate"][ix] for cid, ix in zip(cids, indexes, strict=True)]),
        dtype=torch.bool, device=device,
    )
    # Clip passing-program objective deltas around the per-context median;
    # retain the +2 cost (utility floor -2) for every gate failure.
    centers = target.masked_fill(~gates, float("nan")).nanmedian(dim=1, keepdim=True).values
    centers = torch.nan_to_num(centers, nan=0.0)
    clipped = torch.where(gates, centers + (target - centers).clamp(-CLIP, CLIP), torch.full_like(target, 2.0))
    weights = torch.softmax(-clipped / SMOOTH_TAU, dim=1)
    huber = F.smooth_l1_loss(prediction, clipped, reduction="none").mean(dim=1)
    smooth_rank = -(weights * F.log_softmax(-prediction / SMOOTH_TAU, dim=1)).sum(dim=1)
    fam_weights = torch.as_tensor(
        h48.family_balanced_weights([family_by_cid[cid] for cid in cids]),
        dtype=torch.float32, device=device,
    )
    return torch.sum(fam_weights * (huber + 0.1 * smooth_rank))


def family_interval(gains: dict[str, float], seed: int):
    return h51.family_median_interval(gains, seed)


def load_steps_gains(output: Path, seed: int) -> dict[str, float]:
    path = output / "STEPS" / "k16-lr1e-04" / f"seed{seed}" / "inner-family-gains.json"
    data = json.loads(path.read_text())
    gains = data["family_gains"]
    if len(gains) != 205:
        raise RuntimeError(f"incomplete paired STEPS control: {path}")
    return {str(k): float(v) for k, v in gains.items()}


def run_trained(args, subset_plan: dict) -> dict:
    if args.arm not in ("DATA25", "DATA50", "DATA100", "CLIPSMOOTH"):
        raise RuntimeError(f"invalid trained H51 arm {args.arm}")
    if args.seed not in SEEDS or args.device not in ("cuda:0", "cpu"):
        raise RuntimeError("invalid H51 seed/device")
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("GPU arm requires qualification.lock")
    out = args.output / args.arm / f"seed{args.seed}"
    summary_path = out / "candidate-summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    state = shared.verify_inputs(args.input, args.checkpoint)
    torch.set_num_threads(4 if args.device == "cuda:0" else 8)
    device = torch.device(args.device)
    selected, pools, train_ids, fams, net, trainable, tokens, masks, guard = h48.build_model(
        args.input, args.checkpoint, device
    )
    shared.check_rss()
    predict = h48.make_predictor(net, pools, tokens, masks, device)
    folds = h48.family_folds(selected["train"], count=5, seed=h51.FOLD_SEED)
    gains: dict[str, float] = {}
    fold_rows = []
    started = time.monotonic()
    for fold_id, held_families in enumerate(folds):
        held = [cid for cid in train_ids if fams[cid] in held_families]
        all_fit = [cid for cid in train_ids if fams[cid] not in held_families]
        fit_families = set(fams[cid] for cid in all_fit)
        if args.arm.startswith("DATA"):
            pct = int(args.arm.removeprefix("DATA"))
            chosen = set(subset_plan["seeds"][str(args.seed)][str(fold_id)]["fit_families"][str(pct)])
            fit = [cid for cid in all_fit if fams[cid] in chosen]
            if not chosen.issubset(fit_families):
                raise RuntimeError("DATA subset escaped fold training families")
        else:
            fit = all_fit
        net.load_state_dict(state["state_dict"], strict=True)
        base_weights = h48.parameter_snapshot(net)
        h48.h46.checkpoint_guard(guard, net, expected=shared.DOPE_SHA)
        baseline = h48.evaluate_utility(held, pools, predict)
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
        h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = LR
        update_l2 = []
        for step in range(K):
            net.train()
            batch_seed = args.seed * 1_000_003 + fold_id * 10_007 + K * 101 + step
            batch = h48.family_balanced_context_batch(fit, fams, 256, batch_seed)
            indexes = [h48.base.training_indices(pools[cid], 192) for cid in batch]
            before = h48.parameter_snapshot(net)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(batch, indexes)
            if args.arm == "CLIPSMOOTH":
                loss = clipped_smooth_loss(prediction, pools, batch, indexes, fams)
            else:
                loss = h48.training_loss("0", prediction, pools, batch, indexes, fams)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            after = h48.parameter_snapshot(net)
            update_l2.append(shared.l2_between(after, before))
            h48.h46.checkpoint_guard(guard, net, expected=shared.DOPE_SHA)
            shared.check_rss()
        net.eval()
        candidate = h48.evaluate_utility(held, pools, predict)
        weights = h48.parameter_snapshot(net)
        h48.save_adapter_snapshot(out / f"fold{fold_id}-selected.pt", weights, shared.R882_SHA)
        family_delta = {fam: float(candidate["per_family"][fam] - baseline["per_family"][fam]) for fam in baseline["per_family"]}
        for fam, delta in family_delta.items():
            if fam in gains:
                raise RuntimeError(f"duplicate heldout family {fam}")
            gains[fam] = delta
        fold_rows.append({
            "fold": fold_id, "n_families": len(held_families), "fit_families": len(fit_families),
            "used_fit_families": len(set(fams[cid] for cid in fit)),
            "family_mean_delta_vs_r882": float(np.mean(list(family_delta.values()))),
            "per_family_gain_vs_r882": family_delta,
            "update_l2_by_step": update_l2,
            "total_update_l2_vs_r882": shared.l2_between(weights, base_weights),
        })
    if len(gains) != 205:
        raise RuntimeError(f"expected 205 families, got {len(gains)}")
    control_gains = load_steps_gains(args.output, args.seed)
    if set(control_gains) != set(gains):
        raise RuntimeError("paired family identities differ")
    paired = {fam: gains[fam] - control_gains[fam] for fam in gains}
    med, lb, ub = family_interval(paired, args.seed * 100 + sum(map(ord, args.arm)))
    base_med, base_lb, base_ub = family_interval(gains, args.seed * 100 + 17)
    chosen_fold = max(fold_rows, key=lambda row: row["family_mean_delta_vs_r882"])["fold"]
    adapter_sha, r882_adapter_sha, update_l2_total = h51.adapter_weight_identity(
        out / f"fold{chosen_fold}-selected.pt", args.checkpoint
    )
    summary = {
        "schema": "jope.h51.candidate-summary/v2",
        "candidate_id": f"H51-{args.arm}-k16-lr1e-4-seed{args.seed}",
        "arm": args.arm, "seed": args.seed, "device": args.device, "k": K, "learning_rate": LR,
        "adapter_sha256": adapter_sha, "r882_adapter_sha256": r882_adapter_sha,
        "adapter_update_l2_vs_r882": update_l2_total, "adapter_param_count": h48.TRAINABLE_PARAMS,
        "family_median_gain_vs_r882": base_med, "family_bootstrap_lb95_vs_r882": base_lb,
        "family_bootstrap_ub95_vs_r882": base_ub, "inner_cv_mean_gain_vs_r882": float(np.mean(list(gains.values()))),
        "paired_control": "STEPS-k16-lr1e-4-same-seed-fold-batch-order",
        "paired_family_median_gain": med, "paired_family_bootstrap_lb95": lb,
        "paired_family_bootstrap_ub95": ub, "paired_inner_cv_mean_gain": float(np.mean(list(paired.values()))),
        "accepted_paired": med > h51.STRICT_MARGIN and lb > h51.STRICT_MARGIN,
        "accepted_vs_r882_continuity": base_med > h51.STRICT_MARGIN and base_lb > h51.STRICT_MARGIN,
        "n_families": len(gains), "selected_fold": chosen_fold,
        "folds": [{k: v for k, v in row.items() if k != "per_family_gain_vs_r882"} for row in fold_rows],
        "validation_scored": False, "validation_payloads_opened": False,
        "dope_sha256_before": shared.DOPE_SHA, "dope_sha256_after": shared.DOPE_SHA,
        "checkpoint_sha256": shared.R882_SHA, "family_gains_file": "inner-family-gains.json",
        "wall_seconds": time.monotonic() - started,
    }
    h51.atomic_json(out / "inner-family-gains.json", {
        "schema": "jope.h51.paired-family-gains/v1", "candidate_id": summary["candidate_id"],
        "family_gains_vs_r882": gains, "paired_family_gains_vs_steps": paired,
    })
    h51.atomic_json(summary_path, summary)
    print(json.dumps({k: summary[k] for k in (
        "candidate_id", "family_median_gain_vs_r882", "paired_family_median_gain",
        "paired_family_bootstrap_lb95", "accepted_paired", "adapter_update_l2_vs_r882", "wall_seconds",
    )}, allow_nan=False), flush=True)
    return summary


def run_null(args) -> dict:
    if args.seed not in SEEDS or args.device not in ("cuda:0", "cpu"):
        raise RuntimeError("invalid NULL seed/device")
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("GPU NULL requires qualification.lock")
    out = args.output / "NULL" / f"seed{args.seed}"
    summary_path = out / "candidate-summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    state = shared.verify_inputs(args.input, args.checkpoint)
    torch.set_num_threads(4 if args.device == "cuda:0" else 8)
    device = torch.device(args.device)
    selected, pools, train_ids, fams, net, trainable, tokens, masks, guard = h48.build_model(args.input, args.checkpoint, device)
    shared.check_rss()
    predict = h48.make_predictor(net, pools, tokens, masks, device)
    folds = h48.family_folds(selected["train"], count=5, seed=h51.FOLD_SEED)
    control_gains = load_steps_gains(args.output, args.seed)
    all_gains = [dict() for _ in range(NULL_N)]
    direction_sets = []
    started = time.monotonic()
    generator = torch.Generator(device="cpu").manual_seed(51_000 + args.seed)
    base_state = h48.parameter_snapshot(net)
    for candidate_id in range(NULL_N):
        delta = {name: torch.randn(value.shape, generator=generator, dtype=value.dtype) for name, value in base_state.items()}
        norm = torch.sqrt(sum(torch.sum(value.float() ** 2) for value in delta.values()))
        direction = {name: value.to(device) * (0.30 / float(norm)) for name, value in delta.items()}
        direction_sets.append(direction)
        for fold_id, held_families in enumerate(folds):
            held = [cid for cid in train_ids if fams[cid] in held_families]
            net.load_state_dict(state["state_dict"], strict=True)
            baseline = h48.evaluate_utility(held, pools, predict)
            with torch.no_grad():
                for name, parameter in net.named_parameters():
                    if parameter.requires_grad:
                        parameter.add_(direction[name])
            h48.h46.checkpoint_guard(guard, net, expected=shared.DOPE_SHA)
            candidate = h48.evaluate_utility(held, pools, predict)
            for family, utility in candidate["per_family"].items():
                all_gains[candidate_id][family] = float(utility - baseline["per_family"][family])
    best = None
    rows = []
    for idx, gains in enumerate(all_gains):
        paired = {fam: gains[fam] - control_gains[fam] for fam in gains}
        med, lb, ub = family_interval(paired, args.seed * 1000 + idx)
        base_med, base_lb, base_ub = family_interval(gains, args.seed * 1000 + 100 + idx)
        row = {"candidate": idx, "family_median_gain_vs_r882": base_med, "family_bootstrap_lb95_vs_r882": base_lb,
               "family_bootstrap_ub95_vs_r882": base_ub, "inner_mean_gain_vs_r882": float(np.mean(list(gains.values()))),
               "paired_family_median_gain": med, "paired_family_bootstrap_lb95": lb,
               "paired_family_bootstrap_ub95": ub}
        rows.append(row)
        if best is None or med > best["median"]:
            best = {"id": idx, "median": med, "gains": gains, "paired": paired, "lb": lb, "ub": ub,
                    "base_median": base_med, "base_lb": base_lb, "base_ub": base_ub}
    chosen_id = best["id"]
    med, lb, ub = best["median"], best["lb"], best["ub"]
    for fold_id, held_families in enumerate(folds):
        held = [cid for cid in train_ids if fams[cid] in held_families]
        net.load_state_dict(state["state_dict"], strict=True)
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(direction_sets[chosen_id][name])
        h48.save_adapter_snapshot(out / f"fold{fold_id}-selected.pt", h48.parameter_snapshot(net), shared.R882_SHA)
    adapter_sha, r882_adapter_sha, update_l2 = h51.adapter_weight_identity(
        out / f"fold{max(range(len(folds)), key=lambda i: len(folds[i]))}-selected.pt", args.checkpoint
    )
    summary = {
        "schema": "jope.h51.candidate-summary/v2", "candidate_id": f"H51-NULL-N{NULL_N}-seed{args.seed}",
        "arm": "NULL", "seed": args.seed, "device": args.device, "n_random_directions": NULL_N,
        "direction_l2": 0.30, "selected_direction": chosen_id,
        "adapter_sha256": adapter_sha, "r882_adapter_sha256": r882_adapter_sha,
        "adapter_update_l2_vs_r882": update_l2, "adapter_param_count": h48.TRAINABLE_PARAMS,
        "selection_rule": "maximum paired family-median gain vs matched STEPS among 16 directions",
        "candidate_summaries": rows, "family_median_gain_vs_r882": best["base_median"],
        "family_bootstrap_lb95_vs_r882": best["base_lb"], "family_bootstrap_ub95_vs_r882": best["base_ub"],
        "inner_cv_mean_gain_vs_r882": float(np.mean(list(best["gains"].values()))),
        "paired_control": "STEPS-k16-lr1e-4-same-seed-fold (batch order not applicable to random directions)",
        "paired_family_median_gain": med, "paired_family_bootstrap_lb95": lb,
        "paired_family_bootstrap_ub95": ub,
        "accepted_paired": med > h51.STRICT_MARGIN and lb > h51.STRICT_MARGIN,
        "accepted_vs_r882_continuity": best["base_median"] > h51.STRICT_MARGIN and best["base_lb"] > h51.STRICT_MARGIN,
        "n_families": 205, "validation_scored": False, "validation_payloads_opened": False,
        "checkpoint_sha256": shared.R882_SHA, "dope_sha256_before": shared.DOPE_SHA,
        "dope_sha256_after": shared.DOPE_SHA, "wall_seconds": time.monotonic() - started,
    }
    h51.atomic_json(out / "inner-family-gains.json", {
        "schema": "jope.h51.paired-family-gains/v1", "candidate_id": summary["candidate_id"],
        "family_gains_vs_r882": best["gains"], "paired_family_gains_vs_steps": best["paired"],
    })
    h51.atomic_json(summary_path, summary)
    print(json.dumps({k: summary[k] for k in ("candidate_id", "paired_family_median_gain", "paired_family_bootstrap_lb95", "accepted_paired", "wall_seconds")}, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("DATA25", "DATA50", "DATA100", "CLIPSMOOTH", "NULL"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h51/ckpt/round-882.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h51/runs")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if args.arm == "NULL":
        run_null(args)
    else:
        plan_path = args.output / "DATA" / "subset-plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.is_file() else nested_subsets(args.input, args.output)
        run_trained(args, plan)


if __name__ == "__main__":
    main()
