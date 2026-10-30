#!/usr/bin/env python3
"""H48 inner-family utility alignment diagnostic and optimizer experiments.

The first supported arm is G: estimate the H46 surrogate gradient on a
family-balanced DL-Train minibatch and score its negative direction only on
the held-out DL-Train families. DL-Val is never read by this program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay/h46"))
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
import run as h46  # noqa: E402
import run_fused_closed_loop as base  # noqa: E402
from losses import family_balanced_weights  # noqa: E402

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
CHECKPOINT_SHA = "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a"
TRAINABLE_PARAMS = 29211
STEP_SIZES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def family_folds(train_rows: list[dict], count: int = 5, seed: int = 4800) -> list[set[str]]:
    """Return deterministic, disjoint folds containing whole train families."""
    families = sorted({row["family_id"] for row in train_rows})
    if count < 2 or count > len(families):
        raise ValueError("fold count must be between 2 and the family count")
    rng = np.random.default_rng(seed)
    order = list(np.asarray(families, dtype=object)[rng.permutation(len(families))])
    return [set(map(str, chunk)) for chunk in np.array_split(order, count)]


def family_balanced_context_batch(
    cids: list[str], family_by_cid: dict[str, str], size: int, seed: int
) -> list[str]:
    """Sample contexts with equal probability mass per represented family."""
    by_family: dict[str, list[str]] = {}
    for cid in cids:
        by_family.setdefault(family_by_cid[cid], []).append(cid)
    if not by_family or size <= 0:
        raise ValueError("a nonempty context set and positive size are required")
    rng = np.random.default_rng(seed)
    names = sorted(by_family)
    return [
        str(rng.choice(by_family[names[int(i)]]))
        for i in rng.integers(0, len(names), size=size)
    ]


def load_train_packet(input_dir: Path):
    """Load only DL-Train payloads; DL-Val and outer payloads stay unopened."""
    selected = json.loads((input_dir / "selected.json").read_text())
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if (
        manifest.get("source_manifest_sha256")
        != "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("H48 red-team source-manifest audit failed")
    train_rows = selected.get("train", [])
    dev_rows = selected.get("dev", [])
    if len(train_rows) != 1219 or len(dev_rows) != 91:
        raise RuntimeError("H48 cohort inventory mismatch")
    if len({row["context_sha256"] for row in train_rows}) != 1219:
        raise RuntimeError("duplicate DL-Train context identity")
    if len({row["family_id"] for row in train_rows}) != 205:
        raise RuntimeError("DL-Train family inventory mismatch")
    if len({row["family_id"] for row in dev_rows}) != 19:
        raise RuntimeError("DL-Val family inventory mismatch")
    if {row["family_id"] for row in train_rows} & {row["family_id"] for row in dev_rows}:
        raise RuntimeError("DL-Train/DL-Val family overlap")

    pools = {}
    tables = {}
    for row in train_rows:
        if row.get("dl_split") != "DL-Train" or row.get("source_split") not in ("train", "inner"):
            raise RuntimeError("outside approved DL-Train input closure")
        if row.get("task") != "regression":
            raise RuntimeError("H48 is restricted to regression contexts")
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if base.sha(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"pool hash mismatch for {cid}")
        if base.sha(table_path) != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"train-only table hash mismatch for {cid}")
        with np.load(pool_path, allow_pickle=False) as source:
            programs = source["program"].astype(np.float32)
            pids = [value.decode() if isinstance(value, bytes) else str(value) for value in source["program_sha256"].tolist()]
            pools[cid] = {
                "rec": row,
                "pids": pids,
                "program": programs,
                "proj16": programs.astype(np.float64) @ base.evaluator.get_w1_sign_matrix() / np.sqrt(412),
                "struct_sig": base.compute_semantic_struct_sig_24d(programs),
                "fitness": source["fitness"].astype(np.float64),
                "logratio": source["logratio"].astype(np.float64),
                "gate": source["gate"].astype(bool),
                "prior_cost": np.zeros(len(programs), dtype=np.float64),
                "scale": 0.4208,
            }
        raw = table_path.read_bytes()
        nt, nq, width = base.struct.unpack_from("<III", raw, 4)
        if nq or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError("requires train-only two-fold table input")
        values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        tables[cid] = (values[:, :-1], values[:, -1])
    return selected, pools, tables


def build_model(input_dir: Path, checkpoint: Path, device: torch.device):
    selected, pools, tables = load_train_packet(input_dir)
    train_ids = [row["context_sha256"] for row in selected["train"]]
    train_family = {row["context_sha256"]: row["family_id"] for row in selected["train"]}
    if len(set(train_ids)) != 1219 or len(train_family) != 1219:
        raise RuntimeError("duplicate DL-Train context identity")
    spec = importlib.util.spec_from_file_location(
        "h48_fused", input_dir / "cpu_exact_fused.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load approved fused model module")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    guard = base.guard
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state["state_dict"], strict=True)
    dope_hash = h46.checkpoint_guard(guard, net)
    if dope_hash != DOPE_SHA:
        raise RuntimeError(f"DOPE identity mismatch: {dope_hash}")
    actual_checkpoint_sha = digest(checkpoint)
    if actual_checkpoint_sha != CHECKPOINT_SHA:
        raise RuntimeError(f"checkpoint identity mismatch: {actual_checkpoint_sha}")
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != TRAINABLE_PARAMS:
        raise RuntimeError("unexpected trainable adapter inventory")
    tokens, masks = h46.make_tokens(net, tables, device)
    return selected, pools, train_ids, train_family, net, trainable, tokens, masks, guard


def make_predictor(net, pools, tokens, masks, device):
    def predict(cids: list[str], indexes: list[np.ndarray]) -> torch.Tensor:
        progs = np.stack([pools[cid]["program"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        proj = np.stack([pools[cid]["proj16"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        batch, slate = progs.shape[:2]
        histf = torch.as_tensor(
            np.stack([pools[cid]["fitness"][ix[: min(4, slate)]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32, device=device,
        )
        histg = torch.as_tensor(
            np.stack([pools[cid]["gate"][ix[: min(4, slate)]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32, device=device,
        )
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((batch, 584), device=device),
            prog_412=torch.as_tensor(progs, device=device),
            proj_16=torch.as_tensor(proj, device=device),
            base_prior_cost=torch.zeros((batch, slate), device=device),
            scale=torch.full((batch,), 0.4208, device=device),
            sos_prior=torch.zeros((batch, slate, 2), device=device),
            mm_tokens_10x32=torch.stack([tokens[cid] for cid in cids]),
            mm_mask_10=torch.stack([masks[cid] for cid in cids]),
            revealed_indices=torch.arange(min(4, slate), device=device).reshape(1, -1).expand(batch, -1),
            revealed_raw_f=histf,
            revealed_gate=histg,
        )["total_cost"].float()

    return predict


def evaluate_utility(cids, pools, predict) -> dict:
    """Evaluate actual argmin utility for the requested DL-Train contexts."""
    decisions = []
    by_size: dict[int, list[str]] = {}
    for cid in cids:
        by_size.setdefault(len(pools[cid]["pids"]), []).append(cid)
    for pool_size, contexts in sorted(by_size.items()):
        for start in range(0, len(contexts), 8):
            batch = contexts[start : start + 8]
            indexes = [np.arange(pool_size) for _ in batch]
            with torch.no_grad():
                values = predict(batch, indexes).cpu().numpy()
            for row, cid in zip(values, batch, strict=True):
                choice = int(np.argmin(row))
                fitness = float(pools[cid]["fitness"][choice])
                passed = bool(pools[cid]["gate"][choice])
                decisions.append((cid, -fitness if passed else -2.0))
    per_family: dict[str, list[float]] = {}
    for cid, value in decisions:
        per_family.setdefault(pools[cid]["rec"]["family_id"], []).append(value)
    family_means = {family: float(np.mean(values)) for family, values in per_family.items()}
    vals = [value for _, value in decisions]
    return {
        "mean": float(np.mean(list(family_means.values()))),
        "context_mean": float(np.mean(vals)),
        "per_context": {cid: val for cid, val in decisions},
        "per_family": family_means,
    }


def parameter_snapshot(net) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }


def load_adapter_snapshot(net, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name])


def training_loss(arm, prediction, pools, cids, indexes, family_by_cid):
    target_np = np.stack(
        [pools[cid]["fitness"][ix] for cid, ix in zip(cids, indexes, strict=True)]
    )
    gate_np = np.stack(
        [pools[cid]["gate"][ix] for cid, ix in zip(cids, indexes, strict=True)]
    )
    target = torch.as_tensor(target_np, dtype=torch.float32, device=prediction.device)
    gates = torch.as_tensor(gate_np, dtype=torch.bool, device=prediction.device)
    target = torch.where(gates, target, torch.full_like(target, 2.0))
    weights = torch.as_tensor(
        family_balanced_weights([family_by_cid[cid] for cid in cids]),
        dtype=torch.float32,
        device=prediction.device,
    )
    if arm == "PF":
        median = target.median(dim=1, keepdim=True).values
        mad = (target - median).abs().median(dim=1, keepdim=True).values.clamp_min(0.1)
        normalized_prediction = ((prediction - median) / mad).clamp(-3, 3)
        normalized_target = ((target - median) / mad).clamp(-3, 3)
        row_loss = F.smooth_l1_loss(
            normalized_prediction, normalized_target, reduction="none", beta=1.0
        ).mean(dim=1)
    else:
        row_huber = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
        row_kl = F.kl_div(
            F.log_softmax(-prediction / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(dim=1)
        row_loss = row_huber + 0.1 * row_kl
    return torch.sum(weights * row_loss) / len(cids)


def select_fold_arm(args, net, pools, train_ids, family_by_cid, predict, trainable, guard, checkpoint_state):
    arm_started = time.monotonic()
    fold_assignments = family_folds(
        [pools[cid]["rec"] for cid in train_ids], count=5, seed=args.fold_seed
    )
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    for fold_index, held_families in enumerate(fold_assignments):
        held = [cid for cid in train_ids if family_by_cid[cid] in held_families]
        fit = [cid for cid in train_ids if family_by_cid[cid] not in held_families]
        net.load_state_dict(checkpoint_state["state_dict"], strict=True)
        h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        baseline = evaluate_utility(held, pools, predict)
        optimizer = torch.optim.AdamW(
            trainable, lr=1e-4, weight_decay=1e-4
        )
        h46.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        incumbent_weights = parameter_snapshot(net)
        incumbent_optimizer = copy.deepcopy(optimizer.state_dict())
        incumbent_metric = baseline["mean"]
        best_weights = incumbent_weights
        best_metric = incumbent_metric
        best_eval = baseline
        best_round = 0
        ema = parameter_snapshot(net)
        trajectory = []
        started = time.monotonic()

        for round_index in range(16):
            for update_index in range(2):
                net.train()
                seed = args.seed * 1_000_003 + fold_index * 10_007 + round_index * 31 + update_index
                batch = family_balanced_context_batch(fit, family_by_cid, 256, seed)
                indexes = [base.training_indices(pools[cid], 192) for cid in batch]
                optimizer.zero_grad(set_to_none=True)
                prediction = predict(batch, indexes)
                loss = training_loss(args.arm, prediction, pools, batch, indexes, family_by_cid)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                optimizer.step()
                if args.arm == "PF":
                    with torch.no_grad():
                        for name, parameter in net.named_parameters():
                            if parameter.requires_grad:
                                ema[name].mul_(0.98).add_(parameter.detach(), alpha=0.02)

            raw_weights = parameter_snapshot(net)
            net.eval()
            if args.arm == "PF":
                load_adapter_snapshot(net, ema)
            candidate = evaluate_utility(held, pools, predict)
            h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
            candidate_weights = parameter_snapshot(net)
            if args.arm == "PF":
                load_adapter_snapshot(net, raw_weights)
            improved_best = candidate["mean"] > best_metric
            trajectory.append({
                "round": round_index + 1,
                "loss": float(loss.detach().cpu()),
                "grad_norm": norm,
                "inner_cv_family_utility": candidate["mean"],
                "inner_cv_context_utility": candidate["context_mean"],
                "accepted": args.arm == "0" and candidate["mean"] > incumbent_metric,
                "new_inner_cv_best": improved_best,
                "elapsed_seconds": time.monotonic() - started,
            })

            if candidate["mean"] > best_metric:
                best_metric = candidate["mean"]
                best_weights = candidate_weights
                best_eval = candidate
                best_round = round_index + 1

            if args.arm == "0":
                if candidate["mean"] > incumbent_metric:
                    incumbent_metric = candidate["mean"]
                    incumbent_weights = candidate_weights
                    incumbent_optimizer = copy.deepcopy(optimizer.state_dict())
                else:
                    load_adapter_snapshot(net, incumbent_weights)
                    optimizer.load_state_dict(incumbent_optimizer)
            # PF intentionally retains raw weights, AdamW moments, and EMA state
            # after every rejected/nonbest checkpoint.

        h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
        h46.save_adapter_state(
            args.output / f"fold{fold_index}-selected.pt",
            _AdapterView(net, best_weights),
            digest(args.checkpoint),
            DOPE_SHA,
        )
        summaries.append({
            "fold": fold_index,
            "fit_families": len({family_by_cid[cid] for cid in fit}),
            "fit_contexts": len(fit),
            "heldout_families": sorted(held_families),
            "heldout_contexts": len(held),
            "baseline_family_utility": baseline["mean"],
            "baseline_per_family_utility": baseline["per_family"],
            "selected_family_utility": best_eval["mean"],
            "selected_context_utility": best_eval["context_mean"],
            "selected_per_family_utility": best_eval["per_family"],
            "selected_per_context_utility": best_eval["per_context"],
            "selected_candidate_delta": best_eval["mean"] - baseline["mean"],
            "trajectory": trajectory,
            "wall_seconds": time.monotonic() - started,
            "selected_round": best_round,
        })
        print(json.dumps({"arm": args.arm, "seed": args.seed, "fold": fold_index, "selected_utility": best_metric, "delta": best_metric - baseline["mean"]}), flush=True)

    all_families = [family for row in summaries for family in row["heldout_families"]]
    if len(all_families) != len(set(all_families)) or len(all_families) != 205:
        raise RuntimeError("inner CV family partition is incomplete or overlapping")
    pooled_per_family = {
        family: value
        for row in summaries
        for family, value in row["selected_per_family_utility"].items()
    }
    baseline_by_family = {
        family: value
        for row in summaries
        for family, value in row["baseline_per_family_utility"].items()
    }
    cv_baseline = float(np.mean(list(baseline_by_family.values())))
    cv_selected = float(np.mean(list(pooled_per_family.values())))
    result = {
        "schema": "jope.h48.train-arm/v1",
        "arm": args.arm,
        "seed": args.seed,
        "device": str(args.device),
        "fold_seed": args.fold_seed,
        "selection_metric": "mean of held-out family utility within each fixed inner family fold",
        "folds": summaries,
        "inner_cv_baseline_family_utility": cv_baseline,
        "inner_cv_selected_family_utility": cv_selected,
        "inner_cv_selected_delta": cv_selected - cv_baseline,
        "dope_sha256": DOPE_SHA,
        "checkpoint_sha256": digest(args.checkpoint),
        "trainable_params": TRAINABLE_PARAMS,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "source_manifest_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
        "excluded_dl_holdout_count": 93,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "wall_seconds": time.monotonic() - arm_started,
    }
    atomic_json(args.output / "train-result.json", result)
    # Select the fold model with greatest family-held-out utility for one final
    # terminal DL-Val score after every arm and seed has finished training.
    chosen = max(summaries, key=lambda row: row["selected_candidate_delta"])
    selection = {
        "arm": args.arm,
        "seed": args.seed,
        "fold": chosen["fold"],
        "selected_family_utility": chosen["selected_family_utility"],
        "selected_delta": chosen["selected_candidate_delta"],
        "checkpoint": f"fold{chosen['fold']}-selected.pt",
    }
    atomic_json(args.output / "selected-candidate.json", selection)
    return result


def run_es_arm(args, net, pools, train_ids, family_by_cid, predict, trainable, guard):
    net.eval()
    folds = family_folds(
        [pools[cid]["rec"] for cid in train_ids], count=5, seed=args.fold_seed
    )
    heldouts = [
        [cid for cid in train_ids if family_by_cid[cid] in fold] for fold in folds
    ]
    center = parameter_snapshot(net)
    baseline_rows = [evaluate_utility(cids, pools, predict) for cids in heldouts]
    baseline_per_family = {
        family: value
        for row in baseline_rows
        for family, value in row["per_family"].items()
    }
    baseline_score = float(np.mean(list(baseline_per_family.values())))
    sigma = 1e-3
    population = 16
    update_scale = 0.05 * sigma
    budget = max(1.0, float(args.budget_seconds))
    started = time.monotonic()
    generations = []
    best_score = baseline_score
    best_candidate = center
    best_per_family = baseline_per_family
    generation = 0
    while generation == 0 or time.monotonic() - started < budget:
        generation += 1
        generation_center = center
        noise_rows = []
        candidate_scores = []
        candidate_families = []
        for member in range(population // 2):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed * 1_000_003 + generation * 1009 + member
            )
            noise = {
                name: torch.randn(value.shape, generator=generator, dtype=value.dtype)
                for name, value in center.items()
            }
            noise_rows.append(noise)
            for sign in (1.0, -1.0):
                candidate = {
                    name: value + sign * sigma * noise[name].to(value.device)
                    for name, value in center.items()
                }
                load_adapter_snapshot(net, candidate)
                h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
                fold_rows = [evaluate_utility(cids, pools, predict) for cids in heldouts]
                per_family = {
                    family: value
                    for row in fold_rows
                    for family, value in row["per_family"].items()
                }
                candidate_scores.append(float(np.mean(list(per_family.values()))))
                candidate_families.append(per_family)
        ranks = np.empty(population, dtype=np.float64)
        order = np.argsort(candidate_scores, kind="stable")
        for rank, index in enumerate(order):
            ranks[index] = (rank - (population - 1) / 2) / population
        gradient = {name: torch.zeros_like(value) for name, value in center.items()}
        for pair_index, noise in enumerate(noise_rows):
            shaped = ranks[2 * pair_index] - ranks[2 * pair_index + 1]
            for name in gradient:
                gradient[name].add_(noise[name].to(gradient[name].device), alpha=float(shaped / population))
        grad_norm = torch.sqrt(sum(torch.sum(value**2) for value in gradient.values()))
        with torch.no_grad():
            center = {
                name: value + update_scale * gradient[name] / grad_norm.clamp_min(1e-12)
                for name, value in center.items()
            }
        if max(candidate_scores) > best_score:
            best_index = int(np.argmax(candidate_scores))
            best_score = candidate_scores[best_index]
            best_candidate = {
                name: value + (1.0 if best_index % 2 == 0 else -1.0) * sigma * noise_rows[best_index // 2][name].to(value.device)
                for name, value in generation_center.items()
            }
            best_per_family = candidate_families[best_index]
        generations.append({
            "generation": generation,
            "population": population,
            "sigma": sigma,
            "rank_shaped_utilities": ranks.tolist(),
            "candidate_cv_family_utility": candidate_scores,
            "candidate_cv_deltas": [score - baseline_score for score in candidate_scores],
            "update_l2_norm": float((update_scale * grad_norm).detach().cpu()),
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"arm": "ES", "seed": args.seed, "generation": generation, "best_cv_utility": best_score, "elapsed": time.monotonic() - started}), flush=True)
        if time.monotonic() - started >= budget:
            break

    load_adapter_snapshot(net, best_candidate)
    h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    args.output.mkdir(parents=True, exist_ok=True)
    h46.save_adapter_state(
        args.output / "es-selected.pt",
        _AdapterView(net, best_candidate),
        digest(args.checkpoint),
        DOPE_SHA,
    )
    result = {
        "schema": "jope.h48.train-arm/v1",
        "arm": "ES",
        "seed": args.seed,
        "device": str(args.device),
        "fold_seed": args.fold_seed,
        "inner_cv_baseline_family_utility": baseline_score,
        "inner_cv_selected_family_utility": float(np.mean(list(best_per_family.values()))),
        "inner_cv_selected_delta": float(np.mean(list(best_per_family.values()))) - baseline_score,
        "selected_per_family_utility": best_per_family,
        "population": population,
        "sigma": sigma,
        "rank_shaping": "centered population ranks",
        "update_scale": update_scale,
        "wall_budget_seconds": budget,
        "generations": generations,
        "wall_seconds": time.monotonic() - started,
        "dope_sha256": DOPE_SHA,
        "checkpoint_sha256": digest(args.checkpoint),
        "trainable_params": TRAINABLE_PARAMS,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "source_manifest_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
        "excluded_dl_holdout_count": 93,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }
    atomic_json(args.output / "train-result.json", result)
    atomic_json(args.output / "selected-candidate.json", {
        "arm": "ES", "seed": args.seed, "fold": "all",
        "selected_family_utility": result["inner_cv_selected_family_utility"],
        "selected_delta": result["inner_cv_selected_delta"],
        "checkpoint": "es-selected.pt",
    })
    return result


def run_training_arm(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.90)
    lock_file = None
    if not args.lock_held:
        lock_file = open(Path.home() / "jope_plan/qualification.lock", "a")
        while True:
            if base.psutil.virtual_memory().available < 20 * 1024**3:
                lock_file.close()
                raise RuntimeError("available RAM below 20 GiB")
            if device.type == "cpu" and base.psutil.cpu_percent(interval=1) > 85:
                print("host CPU above 85%; waiting before CPU arm", flush=True)
                time.sleep(10)
                continue
            try:
                base.fcntl.flock(lock_file, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
            except BlockingIOError:
                print("qualification.lock busy; waiting", flush=True)
                time.sleep(15)
                continue
            if device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if free_bytes < 20 * 1024**3:
                    base.fcntl.flock(lock_file, base.fcntl.LOCK_UN)
                    time.sleep(30)
                    continue
            break
    else:
        if base.psutil.virtual_memory().available < 20 * 1024**3:
            raise RuntimeError("available RAM below 20 GiB")
        if device.type == "cpu" and base.psutil.cpu_percent(interval=1) > 85:
            raise RuntimeError("CPU fallback host load exceeds 85%")
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < 20 * 1024**3:
                raise RuntimeError("CUDA free memory below 20 GiB")
    try:
        selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = build_model(
            args.input, args.checkpoint, device
        )
        predict = make_predictor(net, pools, tokens, masks, device)
        checkpoint_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if args.arm == "ES":
            return run_es_arm(
                args, net, pools, train_ids, family_by_cid, predict, trainable, guard
            )
        return select_fold_arm(
            args, net, pools, train_ids, family_by_cid, predict, trainable, guard, checkpoint_state
        )
    finally:
        if lock_file is not None:
            base.fcntl.flock(lock_file, base.fcntl.LOCK_UN)
            lock_file.close()


class _AdapterView:
    """Expose a chosen parameter snapshot to H46's guarded adapter saver."""

    def __init__(self, net, snapshot):
        self._net = net
        self._snapshot = snapshot

    def named_parameters(self):
        for name, parameter in self._net.named_parameters():
            if parameter.requires_grad:
                yield name, self._snapshot[name]


def surrogate_gradient(cids, family_by_cid, pools, predict, trainable, seed: int):
    sampled = family_balanced_context_batch(cids, family_by_cid, 256, seed)
    indexes = [base.training_indices(pools[cid], 192) for cid in sampled]
    prediction = predict(sampled, indexes)
    target_np = np.stack([pools[cid]["fitness"][ix] for cid, ix in zip(sampled, indexes, strict=True)])
    gate_np = np.stack([pools[cid]["gate"][ix] for cid, ix in zip(sampled, indexes, strict=True)])
    target = torch.as_tensor(target_np, dtype=torch.float32, device=prediction.device)
    gates = torch.as_tensor(gate_np, dtype=torch.bool, device=prediction.device)
    target = torch.where(gates, target, torch.full_like(target, 2.0))
    row_huber = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    row_kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(dim=1)
    weights = torch.as_tensor(
        family_balanced_weights([family_by_cid[cid] for cid in sampled]),
        dtype=torch.float32,
        device=prediction.device,
    )
    loss = torch.sum(weights * (row_huber + 0.1 * row_kl)) / len(sampled)
    gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
    gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(trainable, gradients, strict=True)
    )
    norm = torch.sqrt(sum(torch.sum(gradient**2) for gradient in gradients))
    norm_value = float(norm.detach().cpu())
    if not np.isfinite(norm_value) or norm_value <= 0:
        raise RuntimeError("surrogate gradient is non-finite or zero")
    direction = [gradient / norm for gradient in gradients]
    return float(loss.detach().cpu()), norm_value, direction


def run_gradient_diagnostic(args) -> dict:
    if args.device != "cuda:0":
        raise RuntimeError("H48-G must use the approved CUDA device for comparisons")
    while base.psutil.cpu_percent(interval=1) > 85:
        print("host CPU above 85%; waiting before H48-G", flush=True)
        time.sleep(10)
    lock = open(Path.home() / "jope_plan/qualification.lock", "a")
    base.fcntl.flock(lock, base.fcntl.LOCK_EX)
    try:
        if base.psutil.virtual_memory().available < 20 * 1024**3:
            raise RuntimeError("available RAM below 20 GiB")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free_bytes, _ = torch.cuda.mem_get_info(args.device)
        if free_bytes < 20 * 1024**3:
            raise RuntimeError("CUDA free memory below 20 GiB")
        torch.set_num_threads(4)
        random.seed(4800)
        np.random.seed(4800)
        torch.manual_seed(4800)
        device = torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(0.90)
        selected, pools, train_ids, family_by_cid, net, trainable, tokens, masks, guard = build_model(
            args.input, args.checkpoint, device
        )
        predict = make_predictor(net, pools, tokens, masks, device)
        folds = family_folds(selected["train"], count=5, seed=args.fold_seed)
        rows = []
        started = time.monotonic()
        for fold_index, held_families in enumerate(folds):
            held = [cid for cid in train_ids if family_by_cid[cid] in held_families]
            fit = [cid for cid in train_ids if family_by_cid[cid] not in held_families]
            if set(fit) & set(held) or set(fit) | set(held) != set(train_ids):
                raise RuntimeError("inner fold context partition is invalid")
            fold_loss, grad_norm, direction = surrogate_gradient(
                fit, family_by_cid, pools, predict, trainable, seed=4800 + fold_index
            )
            base_weights = [parameter.detach().clone() for parameter in trainable]
            baseline = evaluate_utility(held, pools, predict)
            step_rows = []
            for step_size in STEP_SIZES:
                with torch.no_grad():
                    for parameter, original, vector in zip(trainable, base_weights, direction, strict=True):
                        parameter.copy_(original - step_size * vector)
                h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
                scored = evaluate_utility(held, pools, predict)
                step_rows.append({
                    "step_size": step_size,
                    "heldout_utility": scored["mean"],
                    "delta_vs_fold_baseline": scored["mean"] - baseline["mean"],
                })
                with torch.no_grad():
                    for parameter, original in zip(trainable, base_weights, strict=True):
                        parameter.copy_(original)
            rows.append({
                "fold": fold_index,
                "heldout_families": sorted(held_families),
                "fit_contexts": len(fit),
                "heldout_contexts": len(held),
                "surrogate_loss": fold_loss,
                "gradient_l2_norm": grad_norm,
                "baseline_heldout_utility": baseline["mean"],
                "tested_steps": step_rows,
            })
            print(json.dumps({"arm": "G", "fold": fold_index, "baseline": baseline["mean"], "step_results": step_rows}), flush=True)
        aggregates = []
        for step_index, step_size in enumerate(STEP_SIZES):
            fold_deltas = [row["tested_steps"][step_index]["delta_vs_fold_baseline"] for row in rows]
            aggregates.append({
                "step_size": step_size,
                "mean_heldout_utility_delta": float(np.mean(fold_deltas)),
                "fold_deltas": fold_deltas,
                "improved_folds": sum(delta > 0 for delta in fold_deltas),
            })
        result = {
            "schema": "jope.h48.gradient-diagnostic/v1",
            "arm": "H48-G",
            "device": str(device),
            "fold_seed": args.fold_seed,
            "fold_assignment": [sorted(fold) for fold in folds],
            "steps_are_unit_l2_negative_gradient_distance": True,
            "step_sizes": list(STEP_SIZES),
            "folds": rows,
            "aggregates": aggregates,
            "any_tested_step_improves_aggregate_heldout_utility": any(
                row["mean_heldout_utility_delta"] > 0 for row in aggregates
            ),
            "any_fold_has_any_positive_tested_delta": any(
                row["improved_folds"] > 0 for row in aggregates
            ),
            "conclusion_scope": "If no tested step improves, this only establishes that the tested normalized surrogate-descent direction did not improve these inner held-out train folds; it is not a global impossibility claim.",
            "train_contexts": len(train_ids),
            "train_families": len({family_by_cid[cid] for cid in train_ids}),
            "validation_scored": False,
            "validation_payloads_opened": False,
            "dope_sha256": DOPE_SHA,
            "checkpoint_sha256": digest(args.checkpoint),
            "source_manifest_sha256": json.loads(
                (args.input / "source_manifest.json").read_text()
            )["source_manifest_sha256"],
            "excluded_dl_holdout_count": 93,
            "excluded_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "trainable_params": TRAINABLE_PARAMS,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.output / "gradient-diagnostic.json", result)
        return result
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h48/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h48/ckpt/round-785.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h48/runs")
    parser.add_argument("--arm", choices=("G", "0", "PF", "ES"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=4800)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if digest(args.checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("approved checkpoint SHA mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.arm == "G":
        result = run_gradient_diagnostic(args)
    else:
        args.output = args.output / f"H48-{args.arm}" / f"seed{args.seed}"
        result = run_training_arm(args)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "result": str(args.output), "elapsed": result.get("wall_seconds", result.get("elapsed_seconds"))}), flush=True)


if __name__ == "__main__":
    main()
