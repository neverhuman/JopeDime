#!/usr/bin/env python3
"""One H53 learn-from-zero training cell on the approved offline packet."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
H49_SCORE_PATH = ROOT / "research/flywheel_selfplay/h49/score.py"
INPUT_SHA = "99a53569fcaf59f9306e9e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
TRAINABLE_PARAMS = 29_211
TARGET_UTILITY = 1.1377942


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_sha(net) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(net.adapter_10tok.named_parameters()):
        if not name.startswith("anchor."):
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def reset_adapter(net, scale: float, seed: int):
    old = net.adapter_10tok
    cls = type(old)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fresh = cls(
        frozen_anchor=old.anchor,
        prog_mean=old.prog_mean.detach().cpu().numpy(),
        prog_std=old.prog_std.detach().cpu().numpy(),
        d_token=32,
        n_modalities=10,
        modality_dropout_p=0.10,
    ).to(next(old.parameters()).device)
    for method_name in (
        "encode_multimodal_context_and_candidates",
        "predict_parent_child_delta",
    ):
        bound = getattr(old, method_name)
        setattr(fresh, method_name, MethodType(bound.__func__, fresh))
    net.adapter_10tok = fresh
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
            if scale != 1.0:
                with torch.no_grad():
                    parameter.mul_(scale)
    trainable = [p for p in net.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != TRAINABLE_PARAMS:
        raise RuntimeError("fresh adapter trainable parameter count mismatch")
    if any(name.startswith("anchor.") and p.requires_grad for name, p in net.adapter_10tok.named_parameters()):
        raise RuntimeError("frozen anchor became trainable")
    return trainable


def bootstrap(gains: dict[str, float], seed: int) -> dict:
    values = np.asarray(list(gains.values()), dtype=np.float64)
    if values.size != 19:
        raise RuntimeError(f"expected 19 DL-Val families, got {values.size}")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(10_000, values.size))
    means = values[indexes].mean(axis=1)
    medians = np.median(values[indexes], axis=1)
    return {
        "family_count": int(values.size),
        "mean_gain": float(values.mean()),
        "median_gain": float(np.median(values)),
        "mean_lcb_one_sided_95": float(np.quantile(means, 0.05)),
        "median_lcb_two_sided_95": float(np.quantile(medians, 0.025)),
        "median_ucb_two_sided_95": float(np.quantile(medians, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(12, 13, 14), required=True)
    parser.add_argument("--scale", type=float, choices=(1.0, 0.1), required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h51/ckpt/round-882.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if args.rounds != 3:
        raise RuntimeError("first H53 pilot is preregistered for exactly three rounds")
    if args.device == "cuda:0" and torch.cuda.is_available() and torch.cuda.current_device() != 0:
        torch.cuda.set_device(0)
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("GPU H53 cell requires qualification.lock")
    if sha256(args.input / "source_manifest.json") != INPUT_SHA:
        raise RuntimeError("H53 input manifest file identity mismatch")
    if sha256(args.checkpoint) != R882_SHA:
        raise RuntimeError("r882 comparator checkpoint identity mismatch")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    h48 = import_module("h53_h48", H48_PATH)
    h48.CHECKPOINT_SHA = R882_SHA
    h49_score = import_module("h53_h49_score", H49_SCORE_PATH)
    selected, train_pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = h48.build_model(
        args.input, args.checkpoint, device
    )
    if len(train_ids) != 1219 or len({family_by_cid[cid] for cid in train_ids}) != 205:
        raise RuntimeError("H53 DL-Train roster mismatch")
    selected_val, val_pools, val_tables = h49_score.load_validation_packet(args.input)
    if selected_val != selected["dev"]:
        raise RuntimeError("H53 DL-Val roster differs between offline loaders")
    val_tokens, val_masks = h48.h46.make_tokens(net, val_tables, device)
    val_ids = [row["context_sha256"] for row in selected_val]
    val_families = {row["context_sha256"]: row["family_id"] for row in selected_val}
    baseline_predict = h48.make_predictor(net, val_pools, val_tokens, val_masks, device)
    baseline = h48.evaluate_utility(val_ids, val_pools, baseline_predict)
    baseline_by_context = baseline["per_context"]
    baseline_by_family = baseline["per_family"]
    baseline_adapter_digest = adapter_sha(net)

    # Reset every trainable adapter tensor after loading the frozen r882 comparator.
    # The comparator weights are scored, then the full trainable adapter module is discarded.
    trainable = reset_adapter(net, args.scale, args.seed)
    initial_sha = adapter_sha(net)
    frozen_dope_before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    if frozen_dope_before != DOPE_SHA:
        raise RuntimeError("DOPE source checksum mismatch before training")
    predict = h48.make_predictor(net, train_pools, tokens, masks, device)
    val_predict = h48.make_predictor(net, val_pools, val_tokens, val_masks, device)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    trajectory = []
    started = time.monotonic()
    net.eval()
    initial_val = h48.evaluate_utility(val_ids, val_pools, val_predict)
    trajectory.append({"round": 0, "family_utility": initial_val["mean"], "context_utility": initial_val["context_mean"], "per_family_utility": initial_val["per_family"], "per_context_utility": initial_val["per_context"], "loss": None})
    for round_index in range(1, args.rounds + 1):
        net.train()
        batch_seed = args.seed * 1_000_003 + round_index * 31
        batch = h48.family_balanced_context_batch(train_ids, family_by_cid, 128, batch_seed)
        indexes = [h48.base.training_indices(train_pools[cid], 128) for cid in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = h48.training_loss("0", predict(batch, indexes), train_pools, batch, indexes, family_by_cid)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        dope = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        if dope != DOPE_SHA:
            raise RuntimeError("DOPE identity changed during H53 training")
        net.eval()
        val = h48.evaluate_utility(val_ids, val_pools, val_predict)
        trajectory.append({
            "round": round_index,
            "family_utility": val["mean"],
            "context_utility": val["context_mean"],
            "per_family_utility": val["per_family"],
            "per_context_utility": val["per_context"],
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "elapsed_seconds": time.monotonic() - started,
            "diagnostic_only": True,
        })
    candidate = trajectory[-1]["per_family_utility"]
    gains = {family: float(candidate[family]) - float(baseline_by_family[family]) for family in baseline_by_family}
    stats = bootstrap(gains, 53000 + args.seed + int(args.scale * 10))
    # New ruling is prospective and uses a one-sided mean LCB plus no-harm median.
    admitted_mean_rule = stats["mean_lcb_one_sided_95"] > 0.001 and stats["median_gain"] >= 0.0
    old_median_rule = stats["median_gain"] > 0.001 and stats["median_lcb_two_sided_95"] > 0.001
    # Elo is computed on the same diagnostic DL-Val families, with r882 anchored at 1500.
    elo_path = ROOT / "python/src/jope/north_star/elo_tracker.py"
    elo = import_module("h53_elo_tracker", elo_path)
    records = [elo.DatasetEvalRecord(
        dataset_id=cid,
        family_id=val_families[cid],
        task_type="regression",
        scores_by_contender={"r882": float(baseline_by_context[cid]), "H53": float(trajectory[-1]["per_context_utility"][cid])},
    ) for cid in val_ids]
    elo_result = elo.compute_family_grouped_bradley_terry_elo(records, anchor_contender="r882", anchor_elo=1500.0)
    frozen_dope_after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    if frozen_dope_after != DOPE_SHA:
        raise RuntimeError("DOPE identity mismatch after H53 cell")
    if sha256(args.input / "source_manifest.json") != INPUT_SHA or sha256(args.checkpoint) != R882_SHA:
        raise RuntimeError("pinned input identity changed during H53 cell")
    payload = {
        "schema": "jope.h53.pilot-cell/v1",
        "seed": args.seed,
        "initialization_scale": args.scale,
        "rounds": args.rounds,
        "device": str(device),
        "train_contexts": len(train_ids),
        "train_families": 205,
        "validation_contexts": len(val_ids),
        "validation_families": 19,
        "manifest_identity": MANIFEST_ID,
        "manifest_file_sha256": INPUT_SHA,
        "dope_sha256_before": frozen_dope_before,
        "dope_sha256_after": frozen_dope_after,
        "r882_checkpoint_sha256_comparator_only": R882_SHA,
        "r882_adapter_sha256_scored_then_discarded": baseline_adapter_digest,
        "fresh_adapter_sha256_before_training": initial_sha,
        "fresh_adapter_scale": args.scale,
        "r882_adapter_transfer_to_candidate": False,
        "family_gain_vs_r882_diagnostic": gains,
        "family_bootstrap": stats,
        "t537_admit_mean_lcb_and_no_harm": bool(admitted_mean_rule),
        "old_median_lb_rule": bool(old_median_rule),
        "elo_vs_r882_diagnostic": elo_result,
        "r882_baseline_utility": baseline["mean"],
        "round_to_target_1_1377942": next((row["round"] for row in trajectory if row["family_utility"] >= TARGET_UTILITY), None),
        "best_utility_in_pilot": max(row["family_utility"] for row in trajectory),
        "trajectory": trajectory,
        "validation_diagnostic_only": True,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "wall_seconds": time.monotonic() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out / "summary.json.new"
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out / "summary.json")
    print(json.dumps({k: v for k, v in payload.items() if k != "trajectory" and k != "family_gain_vs_r882_diagnostic"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
