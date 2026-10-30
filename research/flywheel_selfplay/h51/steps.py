#!/usr/bin/env python3
"""H51 persistent-AdamW step-budget factorial, evaluated on DL-Train folds only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run as h51  # noqa: E402

H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
SPEC = importlib.util.spec_from_file_location("h51_steps_h48", H48_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the pinned H48 offline runner")
h48 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h48
SPEC.loader.exec_module(h48)

R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
MANIFEST_SHA = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
MAX_RSS = 6 * 1024**3

# H48 is reused for its model/fold helpers; bind its checkpoint guard to this
# preregistered offline source rather than its original r785 default.
h48.CHECKPOINT_SHA = R882_SHA


def verify_inputs(input_dir: Path, checkpoint: Path) -> dict:
    if h51.sha256(checkpoint) != R882_SHA:
        raise RuntimeError("refusing STEPS: offline r882 checkpoint SHA mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE_SHA:
        raise RuntimeError("refusing STEPS: r882 or DOPE metadata mismatch")
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_SHA
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("H51 source-manifest scope check failed")
    return state


def l2_between(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return float(torch.sqrt(sum(torch.sum((left[k].float() - right[k].float()) ** 2) for k in left)))


def check_rss() -> int:
    import psutil

    rss = psutil.Process().memory_info().rss
    if rss > MAX_RSS:
        raise RuntimeError(f"H51 RSS guard exceeded: {rss} bytes > {MAX_RSS}")
    return rss


def run_steps(args: argparse.Namespace) -> dict:
    if args.seed not in (9, 10, 11):
        raise RuntimeError("H51 STEPS seeds must be 9, 10, or 11")
    if args.k not in (2, 16, 64) or args.lr not in (1e-4, 3e-4):
        raise RuntimeError("unregistered H51 STEPS factorial setting")
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("GPU STEPS requires qualification.lock")
    if args.device not in ("cuda:0", "cpu"):
        raise RuntimeError("device must be cuda:0 or cpu")
    seed_dir = args.output / "STEPS" / f"k{args.k}-lr{args.lr:.0e}" / f"seed{args.seed}"
    summary_path = seed_dir / "candidate-summary.json"
    if summary_path.is_file():
        print(f"skip completed H51-STEPS k={args.k} lr={args.lr:.0e} seed={args.seed}", flush=True)
        return json.loads(summary_path.read_text())
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_state = verify_inputs(args.input, args.checkpoint)
    before_dope = DOPE_SHA
    torch.set_num_threads(8 if args.device == "cpu" else 4)
    selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = h48.build_model(
        args.input, args.checkpoint, torch.device(args.device)
    )
    rss_after_load = check_rss()
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    folds = h48.family_folds(selected["train"], count=5, seed=h51.FOLD_SEED)
    if h51.sha256(args.checkpoint) != R882_SHA:
        raise RuntimeError("r882 checkpoint changed after model load")
    if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
        raise RuntimeError("DOPE guard failed before STEPS")

    fold_rows = []
    family_gains: dict[str, float] = {}
    arm_started = time.monotonic()
    for fold_index, held_families in enumerate(folds):
        held = [cid for cid in train_ids if family_by_cid[cid] in held_families]
        fit = [cid for cid in train_ids if family_by_cid[cid] not in held_families]
        net.load_state_dict(checkpoint_state["state_dict"], strict=True)
        base_adapter = h48.parameter_snapshot(net)
        h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        net.eval()
        baseline = h48.evaluate_utility(held, pools, predict)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        update_l2 = []
        for update_index in range(args.k):
            net.train()
            batch_seed = args.seed * 1_000_003 + fold_index * 10_007 + args.k * 101 + update_index
            batch = h48.family_balanced_context_batch(fit, family_by_cid, 256, batch_seed)
            indexes = [h48.base.training_indices(pools[cid], 192) for cid in batch]
            before = h48.parameter_snapshot(net)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(batch, indexes)
            loss = h48.training_loss("0", prediction, pools, batch, indexes, family_by_cid)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            after = h48.parameter_snapshot(net)
            update_l2.append(l2_between(after, before))
            h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
            if check_rss() > MAX_RSS:
                raise RuntimeError("unreachable RSS guard check")
        net.eval()
        candidate = h48.evaluate_utility(held, pools, predict)
        h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        candidate_weights = h48.parameter_snapshot(net)
        h48.save_adapter_snapshot(
            seed_dir / f"fold{fold_index}-selected.pt",
            candidate_weights,
            R882_SHA,
        )
        for family, value in candidate["per_family"].items():
            if family in family_gains:
                raise RuntimeError(f"duplicate held-out family: {family}")
            family_gains[family] = float(value) - float(baseline["per_family"][family])
        fold_rows.append({
            "fold": fold_index,
            "heldout_families": sorted(held_families),
            "n_families": len(held_families),
            "fit_families": len({family_by_cid[cid] for cid in fit}),
            "baseline_family_utility": baseline["mean"],
            "candidate_family_utility": candidate["mean"],
            "family_delta": candidate["mean"] - baseline["mean"],
            "per_family_gain": {
                family: float(candidate["per_family"][family] - baseline["per_family"][family])
                for family in baseline["per_family"]
            },
            "update_l2_by_step": update_l2,
            "total_update_l2_vs_r882": l2_between(candidate_weights, base_adapter),
            "last_grad_norm": grad_norm,
            "rss_bytes_after_model_load": rss_after_load,
        })
        print(json.dumps({
            "arm": "STEPS", "seed": args.seed, "k": args.k, "lr": args.lr,
            "fold": fold_index, "family_delta": fold_rows[-1]["family_delta"],
            "max_update_l2": max(update_l2),
        }), flush=True)

    if len(family_gains) != 205:
        raise RuntimeError(f"incomplete family CV: {len(family_gains)} families")
    chosen_fold = max(fold_rows, key=lambda row: row["family_delta"])["fold"]
    chosen_path = seed_dir / f"fold{chosen_fold}-selected.pt"
    adapter_sha, r882_adapter_sha, total_l2 = h51.adapter_weight_identity(chosen_path, args.checkpoint)
    median, lower, upper = h51.family_median_interval(
        family_gains, args.seed * 10 + args.k + int(args.lr * 1e7)
    )
    accepted = median > h51.STRICT_MARGIN and lower > h51.STRICT_MARGIN
    if h51.sha256(args.checkpoint) != R882_SHA:
        raise RuntimeError("r882 checkpoint identity changed during STEPS")
    if before_dope != DOPE_SHA:
        raise RuntimeError("DOPE identity changed during STEPS")
    summary = {
        "schema": "jope.h51.candidate-summary/v1",
        "candidate_id": f"H51-STEPS-k{args.k}-lr{args.lr:.0e}-seed{args.seed}",
        "arm": "STEPS",
        "seed": args.seed,
        "device": args.device,
        "k": args.k,
        "learning_rate": args.lr,
        "adapter_sha256": adapter_sha,
        "adapter_file_sha256": h51.sha256(chosen_path),
        "r882_adapter_sha256": r882_adapter_sha,
        "adapter_update_l2_vs_r882": total_l2,
        "adapter_param_count": h48.TRAINABLE_PARAMS,
        "family_median_gain": median,
        "family_bootstrap_lb95": lower,
        "family_bootstrap_ub95": upper,
        "n_families": len(family_gains),
        "inner_cv_mean_gain": float(np.mean(list(family_gains.values()))),
        "bootstrap_draws": h51.BOOTSTRAP_DRAWS,
        "strict_margin": h51.STRICT_MARGIN,
        "accepted": accepted,
        "selected_fold": chosen_fold,
        "candidate_checkpoint": chosen_path.name,
        "update_l2_by_fold": {str(row["fold"]): row["update_l2_by_step"] for row in fold_rows},
        "folds": [{key: value for key, value in row.items() if key != "per_family_gain"} for row in fold_rows],
        "validation_scored": False,
        "validation_payloads_opened": False,
        "outer_rows_delivered": 0,
        "dope_sha256_before": before_dope,
        "dope_sha256_after": DOPE_SHA,
        "checkpoint_sha256": R882_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "fold_seed": h51.FOLD_SEED,
        "rss_limit_bytes": MAX_RSS,
        "wall_seconds": time.monotonic() - arm_started,
    }
    h51.atomic_json(summary_path, summary)
    h51.atomic_json(seed_dir / "inner-family-gains.json", {
        "schema": "jope.h51.family-gains/v1",
        "candidate_id": summary["candidate_id"],
        "family_gains": family_gains,
    })
    print(json.dumps({key: summary[key] for key in (
        "candidate_id", "adapter_sha256", "adapter_update_l2_vs_r882",
        "family_median_gain", "family_bootstrap_lb95", "family_bootstrap_ub95", "accepted",
    )}, allow_nan=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h51/ckpt/round-882.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h51/runs")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    run_steps(args)


if __name__ == "__main__":
    main()
