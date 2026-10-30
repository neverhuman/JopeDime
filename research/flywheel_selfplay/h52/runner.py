#!/usr/bin/env python3
"""H52 source draft. Copy into the H52 repo lane only after H51 closes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
MANIFEST_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
FOLD_SEED = 4900
SEEDS = (12, 13, 14)
H51_CONTROL_SEEDS = (9, 10, 11)
DRAWS = 10_000
MARGIN = 0.001


def import_h48(repo: Path):
    path = repo / "research/flywheel_selfplay/h48/run.py"
    spec = importlib.util.spec_from_file_location("h52_h48", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load H48 helper at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CHECKPOINT_SHA = R882_SHA
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def load_h51_model(h48, input_dir: Path, checkpoint: Path, device: torch.device):
    if sha256(checkpoint) != R882_SHA:
        raise RuntimeError("r882 checkpoint SHA mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE_SHA:
        raise RuntimeError("r882 round/DOPE metadata mismatch")
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if sha256(input_dir / "source_manifest.json") != MANIFEST_FILE_SHA:
        raise RuntimeError("input manifest file SHA mismatch")
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_ID
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("manifest scope/identity mismatch")
    selected, pools, train_ids, fams, net, trainable, tokens, masks, guard = (
        h48.build_model(input_dir, checkpoint, device)
    )
    if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
        raise RuntimeError("DOPE guard failed")
    if sum(parameter.numel() for parameter in trainable) != 29_211:
        raise RuntimeError("trainable adapter parameter count mismatch")
    return state, selected, pools, train_ids, fams, net, trainable, tokens, masks, guard


def predict_all(h48, cids, pools, predict):
    """Predict the full program pool for each context; cids must be DL-Train."""
    grouped: dict[int, list[str]] = {}
    for cid in cids:
        grouped.setdefault(len(pools[cid]["pids"]), []).append(cid)
    result = {}
    for size, contexts in sorted(grouped.items()):
        for start in range(0, len(contexts), 8):
            batch = contexts[start : start + 8]
            indexes = [np.arange(size) for _ in batch]
            with torch.no_grad():
                values = predict(batch, indexes).detach().cpu().numpy()
            for cid, row in zip(batch, values, strict=True):
                result[cid] = int(np.argmin(row))
    return result


def run_step0(args, h48):
    state, selected, pools, train_ids, fams, net, trainable, tokens, masks, guard = (
        load_h51_model(h48, args.input, args.checkpoint, torch.device("cpu"))
    )
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cpu"))
    fold_sets = h48.family_folds(selected["train"], count=5, seed=FOLD_SEED)
    baseline_choices = {}
    fail_gaps, pass_gaps = [], []
    fail_count = pass_count = no_pass_count = 0
    for held_families in fold_sets:
        held = [cid for cid in train_ids if fams[cid] in held_families]
        choices = predict_all(h48, held, pools, predict)
        baseline_choices.update(choices)
        for cid, choice in choices.items():
            pool = pools[cid]
            gate = pool["gate"]
            fitness = pool["fitness"]
            passed = np.flatnonzero(gate)
            if passed.size == 0:
                no_pass_count += 1
                continue
            oracle_cost = float(np.min(fitness[passed]))
            if not bool(gate[choice]):
                fail_count += 1
                fail_gaps.append(2.0 - oracle_cost)
            else:
                pass_count += 1
                pass_gaps.append(float(fitness[choice]) - oracle_cost)
    family_flips = {}
    for seed in H51_CONTROL_SEEDS:
        any_flip = set()
        for fold_index, held_families in enumerate(fold_sets):
            path = args.control_root / f"seed{seed}" / f"fold{fold_index}-selected.pt"
            if not path.is_file():
                raise RuntimeError(f"missing H51 CONTROL checkpoint: {path}")
            net.load_state_dict(state["state_dict"], strict=True)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            h48.load_adapter_snapshot(net, payload["adapter_trainable_state"])
            h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
            held = [cid for cid in train_ids if fams[cid] in held_families]
            changed = predict_all(h48, held, pools, predict)
            for cid, idx in changed.items():
                if pools[cid]["pids"][idx] != pools[cid]["pids"][baseline_choices[cid]]:
                    any_flip.add(fams[cid])
        family_flips[str(seed)] = {
            "families_with_any_flip": len(any_flip),
            "fraction": len(any_flip) / 205,
        }
    total = len(train_ids)
    summary = {
        "schema": "jope.h52.step0-diagnostic/v1",
        "checkpoint_sha256": R882_SHA,
        "dope_sha256": DOPE_SHA,
        "manifest_identity": MANIFEST_ID,
        "fold_seed": FOLD_SEED,
        "n_contexts": total,
        "argmin_gate_failed_count": fail_count,
        "argmin_gate_failed_fraction": fail_count / total,
        "argmin_passing_count": pass_count,
        "no_passing_program_count": no_pass_count,
        "mean_oracle_gap_gate_failures": float(np.mean(fail_gaps))
        if fail_gaps
        else None,
        "mean_oracle_gap_suboptimal_passing": float(np.mean(pass_gaps))
        if pass_gaps
        else None,
        "families_with_any_argmin_flip_under_h51_control": family_flips,
        "validation_scored": False,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }
    atomic_json(args.output / "step0-diagnostic.json", summary)
    print(json.dumps(summary, allow_nan=False))


def context_costs(pools, cid, indexes, device):
    p = pools[cid]
    ix = np.asarray(indexes, dtype=np.int64)
    target = torch.as_tensor(
        p["fitness"][ix].copy(), dtype=torch.float32, device=device
    )
    gate = torch.as_tensor(p["gate"][ix].copy(), dtype=torch.bool, device=device)
    cost = torch.where(gate, target, torch.full_like(target, 2.0))
    return cost, gate


def objective_loss(
    h48,
    arm,
    prediction,
    pools,
    batch,
    indexes,
    family_by_cid,
    tau,
    anchor_delta,
    lam,
    seed,
):
    device = prediction.device
    costs, gates = zip(
        *(
            context_costs(pools, cid, ix, device)
            for cid, ix in zip(batch, indexes, strict=True)
        ),
        strict=True,
    )
    costs = torch.stack(costs)
    gates = torch.stack(gates)
    if arm == "CONTROL":
        return h48.training_loss("0", prediction, pools, batch, indexes, family_by_cid)
    if arm == "ANCHOR":
        loss = h48.training_loss("0", prediction, pools, batch, indexes, family_by_cid)
        penalty = sum(torch.sum(value.float() ** 2) for value in anchor_delta.values())
        return loss + lam * (0.5 / 29_211) * penalty
    if arm == "NULL":
        rng = np.random.default_rng(seed)
        shuffled = costs.clone()
        by_family: dict[str, list[int]] = {}
        for i, cid in enumerate(batch):
            by_family.setdefault(family_by_cid[cid], []).append(i)
        for rows in by_family.values():
            if len(rows) > 1:
                perm = rng.permutation(rows)
                shuffled[rows] = costs[perm]
        costs = shuffled
    median = costs.median(dim=1, keepdim=True).values
    mad = (costs - median).abs().median(dim=1, keepdim=True).values.clamp_min(0.1)
    target = ((costs - median) / mad).clamp(-3, 3)
    pred = ((prediction - median) / mad).clamp(-3, 3)
    weights = torch.softmax(-costs / tau, dim=1)
    per_program = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    loss = (weights * per_program).sum(dim=1).mean()
    if arm == "GATE":
        margins = []
        for row in range(len(batch)):
            passed = torch.nonzero(gates[row], as_tuple=False).flatten()
            failed = torch.nonzero(~gates[row], as_tuple=False).flatten()
            if passed.numel() and failed.numel():
                best = passed[torch.argmin(costs[row, passed])]
                margins.extend(
                    F.relu(0.5 + prediction[row, best] - prediction[row, failed])
                )
        if margins:
            loss = loss + torch.stack(margins).mean()
    return loss


def run_candidate(args, h48):
    torch.set_num_threads(8 if args.device == "cpu" else 4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state, selected, pools, train_ids, fams, net, trainable, tokens, masks, guard = (
        load_h51_model(h48, args.input, args.checkpoint, torch.device(args.device))
    )
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    folds = h48.family_folds(selected["train"], count=5, seed=FOLD_SEED)
    before_dope = DOPE_SHA
    gains = {}
    fold_reports = []
    for fold_id, held_families in enumerate(folds):
        held = [cid for cid in train_ids if fams[cid] in held_families]
        fit = [cid for cid in train_ids if fams[cid] not in held_families]
        net.load_state_dict(state["state_dict"], strict=True)
        baseline = h48.evaluate_utility(held, pools, predict)
        base_adapter = h48.parameter_snapshot(net)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        steps = args.k
        slate_hashes = []
        step_drifts = []
        for step in range(steps):
            if args.arm in ("FOCUS", "GATE", "NULL"):
                if step < 2:
                    lr_now = args.lr * (step + 1) / 2
                else:
                    lr_now = (
                        args.lr
                        * 0.5
                        * (1 + math.cos(math.pi * (step - 2) / max(1, steps - 3)))
                    )
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
            batch_seed = args.seed * 1_000_003 + fold_id * 10_007 + step * 31
            batch = h48.family_balanced_context_batch(fit, fams, 256, batch_seed)
            indexes = [h48.base.training_indices(pools[cid], 192) for cid in batch]
            slate_digest = hashlib.sha256()
            for cid, ix in zip(batch, indexes, strict=True):
                slate_digest.update(cid.encode())
                slate_digest.update(b"\0")
                slate_digest.update(np.asarray(ix, dtype="<i8").tobytes())
            slate_hashes.append(slate_digest.hexdigest())
            net.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(batch, indexes)
            anchor_delta = {
                name: parameter - base_adapter[name]
                for name, parameter in net.named_parameters()
                if parameter.requires_grad
            }
            loss = objective_loss(
                h48,
                args.arm,
                prediction,
                pools,
                batch,
                indexes,
                fams,
                args.tau,
                anchor_delta,
                args.lambda_anchor,
                batch_seed,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
            with torch.no_grad():
                drift = torch.sqrt(
                    sum(
                        torch.sum((parameter - base_adapter[name]) ** 2)
                        for name, parameter in net.named_parameters()
                        if parameter.requires_grad
                    )
                )
                step_drifts.append(float(drift))
        net.eval()
        candidate = h48.evaluate_utility(held, pools, predict)
        snapshot = h48.parameter_snapshot(net)
        h48.save_adapter_snapshot(
            args.out / f"{args.arm}-{args.config}-seed{args.seed}-fold{fold_id}.pt",
            snapshot,
            R882_SHA,
        )
        for family, value in candidate["per_family"].items():
            gains[family] = float(value) - float(baseline["per_family"][family])
        fold_reports.append(
            {
                "fold": fold_id,
                "family_delta": candidate["mean"] - baseline["mean"],
                "total_update_l2": float(
                    torch.sqrt(
                        sum(
                            torch.sum((snapshot[k] - base_adapter[k]) ** 2)
                            for k in snapshot
                        )
                    )
                ),
                "training_slate_sha256_by_step": slate_hashes,
                "adapter_drift_l2_by_step": step_drifts if args.arm == "ANCHOR" else [],
            }
        )
    values = np.asarray(list(gains.values()), dtype=np.float64)
    rng = np.random.default_rng(52_000 + args.seed)
    indices = rng.integers(0, len(values), size=(DRAWS, len(values)))
    boot = np.median(values[indices], axis=1)
    median = float(np.median(values))
    lb = float(np.quantile(boot, 0.025))
    ub = float(np.quantile(boot, 0.975))
    paired_gains = None
    paired_stats = None
    if args.paired_control_gains is not None:
        control_record = json.loads(args.paired_control_gains.read_text())
        control = control_record.get(
            "family_gains", control_record.get("family_gains_vs_r882", {})
        )
        if len(control) != len(gains) or set(control) != set(gains):
            raise RuntimeError("paired CONTROL family inventory mismatch")
        paired_gains = {
            family: float(gains[family]) - float(control[family]) for family in gains
        }
        paired_values = np.asarray(list(paired_gains.values()), dtype=np.float64)
        paired_rng = np.random.default_rng(52_100 + args.seed)
        paired_indexes = paired_rng.integers(
            0, len(paired_values), size=(DRAWS, len(paired_values))
        )
        paired_boot = np.median(paired_values[paired_indexes], axis=1)
        paired_stats = (
            float(np.median(paired_values)),
            float(np.quantile(paired_boot, 0.025)),
            float(np.quantile(paired_boot, 0.975)),
        )
        positive = int(np.count_nonzero(paired_values > 0))
        negative = int(np.count_nonzero(paired_values < 0))
        nonzero = positive + negative
        sign_n = nonzero
        sign_k = min(positive, negative)
        sign_tail = (
            sum(math.comb(sign_n, i) for i in range(sign_k + 1)) / (2**sign_n)
            if sign_n
            else 1.0
        )
        sign_p = min(1.0, 2.0 * sign_tail)
        mean_boot = np.mean(paired_values[paired_indexes], axis=1)
        paired_diagnostics = {
            "paired_family_count": len(paired_values),
            "paired_nonzero_family_count": nonzero,
            "paired_nonzero_family_fraction": nonzero / len(paired_values),
            "paired_positive_family_count": positive,
            "paired_negative_family_count": negative,
            "paired_zero_family_count": len(paired_values) - nonzero,
            "paired_sign_test_two_sided_p": sign_p,
            "paired_mean_bootstrap_lb95_diagnostic_only": float(
                np.quantile(mean_boot, 0.025)
            ),
            "paired_mean_bootstrap_ub95_diagnostic_only": float(
                np.quantile(mean_boot, 0.975)
            ),
        }
    else:
        paired_diagnostics = None
    if args.arm != "CONTROL" and paired_stats is None:
        raise RuntimeError(
            "every H52 objective arm requires a matched CONTROL gain vector"
        )
    adapter_shas = []
    for fold_id in range(5):
        payload = torch.load(
            args.out / f"{args.arm}-{args.config}-seed{args.seed}-fold{fold_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        digest = hashlib.sha256()
        for name, value in sorted(payload["adapter_trainable_state"].items()):
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
        adapter_shas.append(digest.hexdigest())
    summary = {
        "schema": "jope.h52.candidate-summary/v2",
        "arm": args.arm,
        "config": args.config,
        "seed": args.seed,
        "device": args.device,
        "adapter_sha256_by_fold": adapter_shas,
        "family_median_gain": median,
        "family_bootstrap_lb95": lb,
        "family_bootstrap_ub95": ub,
        "n_families": len(gains),
        "inner_cv_mean_gain": float(np.mean(values)),
        "paired_control_config": args.paired_control_gains.name
        if args.paired_control_gains
        else None,
        "paired_family_median_gain": paired_stats[0] if paired_stats else None,
        "paired_family_bootstrap_lb95": paired_stats[1] if paired_stats else None,
        "paired_family_bootstrap_ub95": paired_stats[2] if paired_stats else None,
        "paired_inner_cv_mean_gain": float(np.mean(list(paired_gains.values())))
        if paired_gains
        else None,
        "accepted_provisional": bool(
            paired_stats and paired_stats[0] > MARGIN and paired_stats[1] > MARGIN
        ),
        "paired_diagnostics_diagnostic_only": paired_diagnostics,
        "training_index_rng_seed": args.seed,
        "folds": fold_reports,
        "dope_sha256_before": before_dope,
        "dope_sha256_after": DOPE_SHA,
        "checkpoint_sha256": R882_SHA,
        "input_manifest_identity": MANIFEST_ID,
        "input_manifest_file_sha256": MANIFEST_FILE_SHA,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "terminal_dl_val_read": False,
    }
    atomic_json(args.out / f"{args.arm}-{args.config}-seed{args.seed}.json", summary)
    atomic_json(
        args.out / f"{args.arm}-{args.config}-seed{args.seed}-family-gains.json",
        {
            "family_gains_vs_r882": gains,
            "paired_family_gains_vs_control": paired_gains,
        },
    )
    print(json.dumps(summary, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "jope-data/h51/ckpt/round-882.pt",
    )
    p.add_argument("--output", type=Path, default=Path.home() / "jope-data/h52/runs")
    p.add_argument("--out", type=Path, default=Path.home() / "jope-data/h52/runs")
    p.add_argument("--mode", choices=("step0", "candidate"), required=True)
    p.add_argument("--arm", choices=("CONTROL", "ANCHOR", "FOCUS", "GATE", "NULL"))
    p.add_argument("--config", default="default")
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lambda-anchor", dest="lambda_anchor", type=float, default=1.0)
    p.add_argument("--lock-held", action="store_true")
    p.add_argument(
        "--control-root", type=Path, default=Path.home() / "jope-data/h51/runs/CONTROL"
    )
    p.add_argument("--paired-control-gains", type=Path)
    args = p.parse_args()
    h48 = import_h48(args.repo_root)
    if args.mode == "step0":
        run_step0(args, h48)
    else:
        if args.device == "cuda:0" and not args.lock_held:
            raise RuntimeError("GPU arm requires qualification.lock")
        run_candidate(args, h48)


if __name__ == "__main__":
    main()
