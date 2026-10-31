#!/usr/bin/env python3
"""H100 fixed ASAM confirmation; CPU dry run never opens corpus payloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("CONTROL", "ASAM05", "ASAM10", "DRIFT_MATCH")
SEEDS = (39, 40, 41)
RHO = {"ASAM05": 0.5, "ASAM10": 1.0}
MARGIN = 0.001546
LR_SCALE = 0.8465252628604802
DRIFT_LR = 8.465252628604803e-5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load("h100_h82", HERE.parent / "h82_sam_steps/runner.py")
science, h59, h55 = legacy.science, legacy.h59, legacy.h55
batch_plan, atomic_json = legacy.batch_plan, legacy.atomic_json
safe_path, validate_paths = legacy.safe_path, legacy.validate_paths
assert_lease, mark_phase = legacy.assert_lease, legacy.mark_phase
restore_adapter = legacy.restore_adapter
SELECTED_SHA, FUSED_SHA = legacy.SELECTED_SHA, legacy.FUSED_SHA


def learning_rate(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return DRIFT_LR if arm == "DRIFT_MATCH" else 1e-4


def adapter_l2(state, initial):
    if state.keys() != initial.keys() or not state:
        raise RuntimeError("adapter distance inventory mismatch")
    squared_distances = []
    for name, value in state.items():
        if value.shape != initial[name].shape:
            raise RuntimeError("adapter distance shape mismatch")
        delta = value.detach().cpu().double() - initial[name].detach().cpu().double()
        if not torch.isfinite(delta).all():
            raise RuntimeError("nonfinite adapter distance")
        squared_distances.append(float(delta.square().sum()))
    return math.sqrt(sum(squared_distances))


def stage_copy(source, scratch, *, expected_sha=None):
    expected_sha = expected_sha or h59.R882_SHA
    source, scratch = Path(source), Path(scratch)
    if h59.sha256(source) != expected_sha:
        raise RuntimeError("source checkpoint hash mismatch")
    for parent in (scratch, scratch.parent):
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise RuntimeError("scratch must be a real directory")
    staged = scratch / "r882_adapter_copy.pt"
    if staged.is_symlink() or staged.resolve() == source.resolve():
        raise RuntimeError("scratch cannot alias source checkpoint")
    if staged.exists():
        if staged.samefile(source):
            raise RuntimeError("scratch hardlinks source checkpoint")
        if (
            not stat.S_ISREG(staged.lstat().st_mode)
            or h59.sha256(staged) != expected_sha
        ):
            raise RuntimeError("unknown staged checkpoint")
    else:
        import shutil

        scratch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged)
    staged.chmod(0o444)  # Only the isolated copy; never the canonical source.
    if h59.sha256(staged) != expected_sha:
        raise RuntimeError("staged checkpoint hash mismatch")
    return staged


def optimizer_step(named, optimizer, closure, arm):
    if arm in ("CONTROL", "DRIFT_MATCH"):
        return legacy.optimizer_step(named, optimizer, closure, "CONTROL")
    if arm not in RHO:
        raise ValueError(arm)
    named = [(n, p) for n, p in named if p.requires_grad]
    params = [p for _, p in named]
    optimizer.zero_grad(set_to_none=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if any(p.is_cuda for p in params) else None
    )
    loss = closure()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    active = [(n, p) for n, p in named if p.grad is not None]
    scales = [
        p.detach().abs() + 0.01 if "weight" in n else torch.ones_like(p)
        for n, p in active
    ]
    norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(t * p.grad)
                for (_, p), t in zip(active, scales, strict=True)
            ]
        )
    )
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite perturbation norm")
    originals = []
    try:
        with torch.no_grad():
            for (_, p), scale in zip(active, scales, strict=True):
                originals.append((p, p.detach().clone()))
                p.add_(RHO[arm] * scale.square() * p.grad / (norm + 1e-16))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        loss = closure()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite perturbed loss")
        loss.backward()
    finally:
        with torch.no_grad():
            for p, original in originals:
                p.copy_(original)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def top4_indices(scores):
    result = {}
    for cid, values in scores.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 8 or not np.isfinite(values).all():
            raise RuntimeError("invalid unrevealed top-four packet")
        result[cid] = (np.argsort(values[4:], kind="stable")[:4] + 4).tolist()
    return result


def overlap_readout(left, right, families):
    if not left or left.keys() != right.keys() or left.keys() != families.keys():
        raise RuntimeError("top-four roster mismatch")
    grouped = {}
    for cid in sorted(left):
        if any(
            len(v) != 4 or len(set(v)) != 4 or min(v) < 4
            for v in (left[cid], right[cid])
        ):
            raise RuntimeError("top-four includes revealed/duplicate indices")
        grouped.setdefault(families[cid], []).append(
            len(set(left[cid]) & set(right[cid])) / 4
        )
    means = {f: float(np.mean(v)) for f, v in grouped.items()}
    return {"mean": float(np.mean(list(means.values()))), "per_family": means}


def top4_overlap(scores, reference, pools):
    left, right = top4_indices(scores), top4_indices(reference)
    families = {c: str(pools[c]["rec"]["family_id"]) for c in pools}
    return {
        **overlap_readout(left, right, families),
        "candidate": left,
        "r882": right,
        "families": families,
    }


def holm(tail_areas):
    ordered = sorted(tail_areas, key=lambda a: (tail_areas[a], a))
    adjusted, running, passing = {}, 0.0, True
    for rank, arm in enumerate(ordered):
        p = tail_areas[arm]
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise RuntimeError("invalid Holm tail area")
        running = max(running, (len(ordered) - rank) * p)
        passing = passing and p < 0.05 / (len(ordered) - rank)
        adjusted[arm] = {
            "tail_area": p,
            "adjusted": min(1.0, running),
            "passes": passing,
        }
    return adjusted


def dry_run(input_dir):
    safe_path(input_dir)
    selected, train, dev = h55.input_metadata(input_dir, None)
    del selected
    if (
        h59.sha256(input_dir / "selected.json") != SELECTED_SHA
        or h59.sha256(input_dir / "cpu_exact_fused.py") != FUSED_SHA
    ):
        raise RuntimeError("dry-run input identity mismatch")
    calibration = json.loads((HERE / "calibration.json").read_text())
    if calibration["lr_scale"] != LR_SCALE or calibration["drift_match_lr"] != DRIFT_LR:
        raise RuntimeError("frozen drift calibration mismatch")
    return {
        "schema": "jope.h100.cpu-dry-run/v1",
        "status": "CPU prereg",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "planned_outcome_cells": 12,
        "train_contexts": len(train),
        "train_families": len({r["family_id"] for r in train}),
        "dev_metadata_contexts": len(dev),
        "dev_metadata_families": len({r["family_id"] for r in dev}),
        "drift_match_lr": DRIFT_LR,
        "dev_payloads_opened": 0,
        "train_payloads_opened": 0,
        "optimizer_updates": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_ticket_created": False,
        "prereg_sha256": h59.sha256(HERE / "PREREG.md"),
    }


def rating(p):
    delta = science.elo_delta(float(p))
    return delta if isinstance(delta, str) else 1500 + delta


def elo_summary(gains, samples):
    wins = (gains > 0).astype(float) + 0.5 * (gains == 0)
    p = wins.mean(axis=1)
    boot = wins[:, samples].mean(axis=2)
    median_boot = np.median(boot, axis=0)
    return {
        "per_seed": [
            {
                "seed": seed,
                "win_rate": float(prob),
                "elo": rating(prob),
                "elo_ci95": [rating(x) for x in np.quantile(boot[i], [0.025, 0.975])],
            }
            for i, (seed, prob) in enumerate(zip(SEEDS, p, strict=True))
        ],
        "median_elo": rating(np.median(p)),
        "median_elo_ci95": [
            rating(x) for x in np.quantile(median_boot, [0.025, 0.975])
        ],
        "pooled_elo": rating(wins.mean()),
        "pooled_elo_ci95": [
            rating(x) for x in np.quantile(boot.mean(axis=0), [0.025, 0.975])
        ],
    }


def summarize(cells, draws=10000):
    indexed = {(c["arm"], c["seed"]): c for c in cells}
    if len(cells) != 12 or set(indexed) != {(a, seed) for a in ARMS for seed in SEEDS}:
        raise RuntimeError("all twelve unique H100 cells required")
    first = indexed["CONTROL", SEEDS[0]]
    families = sorted(first["dev"]["per_family"])
    if len(families) != 19:
        raise RuntimeError("nineteen dev families required")
    identities = set()
    for cell in cells:
        arm = cell["arm"]
        if (
            cell["wave"] != "H100"
            or cell["timing_only"]
            or cell["trainable_count"] != 29211
            or cell["updates"] != 40
            or cell["backward_passes"] != (80 if arm in RHO else 40)
            or cell["dope_before"] != h59.DOPE_SHA
            or cell["dope_after"] != h59.DOPE_SHA
            or cell["checkpoint_sha256"] != h59.R882_SHA
            or cell["manifest_sha256"] != h59.MANIFEST_SHA
            or cell["selected_sha256"] != SELECTED_SHA
            or cell["calibration_sha256"] != h59.sha256(HERE / "calibration.json")
            or cell["learning_rate"] != learning_rate(arm)
            or cell["rho"] != RHO.get(arm)
            or cell["initial_adapter_sha256"] == cell["final_adapter_sha256"]
            or cell["holdout_payloads_opened"] != 0
            or cell["outer_rows_delivered"] != 0
            or not math.isfinite(cell["adapter_l2"])
            or cell["adapter_l2"] <= 0
        ):
            raise RuntimeError("H100 cell integrity mismatch")
        if (
            len(cell["train"]["per_context"]) != 1219
            or len(cell["train"]["per_family"]) != 205
            or len(cell["dev"]["per_context"]) != 91
            or sorted(cell["dev"]["per_family"]) != families
            or sorted(cell["r882_dev"]["per_family"]) != families
            or set(cell["train"]["per_family"]) & set(families)
            or cell["r882_dev"] != indexed["CONTROL", cell["seed"]]["r882_dev"]
        ):
            raise RuntimeError("H100 roster/reference mismatch")
        for packet in ("train", "dev", "r882_dev"):
            if not math.isfinite(cell[packet]["mean"]) or (
                not np.isfinite(list(cell[packet]["per_family"].values())).all()
                or not np.isfinite(list(cell[packet]["per_context"].values())).all()
            ):
                raise RuntimeError("nonfinite H100 packet")
        for packet in ("train", "dev"):
            if (
                cell[packet]["per_context"].keys()
                != first[packet]["per_context"].keys()
            ):
                raise RuntimeError("context roster mismatch")
        top = cell["top4"]
        if top["candidate"].keys() != cell["dev"]["per_context"].keys():
            raise RuntimeError("top-four roster mismatch")
        checked = overlap_readout(top["candidate"], top["r882"], top["families"])
        if checked["mean"] != top["mean"] or checked["per_family"] != top["per_family"]:
            raise RuntimeError("top-four readout mismatch")
        identities.add(
            json.dumps(
                [
                    cell["source_commit"],
                    cell["prereg_sha256"],
                    cell["initial_adapter_sha256"],
                    cell["folds"],
                    cell["source_hashes"],
                ],
                sort_keys=True,
            )
        )
    if len(identities) != 1:
        raise RuntimeError("source/prereg/initial-adapter/fold mismatch")

    def matrix(arm, packet):
        return np.asarray(
            [
                [indexed[arm, seed][packet]["per_family"][f] for f in families]
                for seed in SEEDS
            ]
        )

    control, reference = matrix("CONTROL", "dev"), matrix("CONTROL", "r882_dev")
    samples = np.random.default_rng(100000).integers(0, 19, size=(draws, 19))
    result = {
        "schema": "jope.h100.metrics/v1",
        "complete": True,
        "seeds": list(SEEDS),
        "reference_elo": 1500,
        "bootstrap_draws": draws,
        "arms": {},
    }
    tail_areas = {}
    for arm in ARMS:
        utilities = matrix(arm, "dev")
        gains = utilities - control
        boot = np.median(gains[:, samples].mean(axis=2), axis=0)
        point, lcb = (
            float(np.median(gains.mean(axis=1))),
            float(np.quantile(boot, 0.05)),
        )
        delta = utilities - reference
        harmed = delta.mean(axis=0) < -0.01
        harm = float(harmed.mean())
        rows = [indexed[arm, seed] for seed in SEEDS]
        a = {
            "per_seed_val_reg_gain": gains.mean(axis=1).tolist(),
            "median_gain": point,
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": lcb,
            "raw_gate": lcb > MARGIN and point >= 0,
            "harm_fraction": harm,
            "harmed_families": [
                f for f, bad in zip(families, harmed, strict=True) if bad
            ],
            "per_seed_harm_fraction": (delta < -0.01).mean(axis=1).tolist(),
            "harm_gate": harm <= 0.10,
            "elo_vs_control": elo_summary(gains, samples),
            "elo_vs_r882": elo_summary(delta, samples),
            "mean_adapter_l2": float(np.mean([c["adapter_l2"] for c in rows])),
            "mean_top4_overlap_vs_r882": float(
                np.mean([c["top4"]["mean"] for c in rows])
            ),
            "seeds": [
                {
                    "seed": c["seed"],
                    "train_reg": c["train"]["mean"],
                    "val_reg": c["dev"]["mean"],
                    "gap": c["train"]["mean"] - c["dev"]["mean"],
                    "adapter_l2": c["adapter_l2"],
                    "top4_overlap_vs_r882": c["top4"]["mean"],
                }
                for c in rows
            ],
        }
        if arm != "CONTROL":
            tail_areas[arm] = float(
                (1 + np.count_nonzero(boot <= MARGIN)) / (draws + 1)
            )
        result["arms"][arm] = a
    result["holm"] = holm(tail_areas)
    for arm, a in result["arms"].items():
        a["admitted"] = bool(
            arm != "CONTROL"
            and a["raw_gate"]
            and a["harm_gate"]
            and result["holm"][arm]["passes"]
        )
    difference = matrix("ASAM05", "dev") - matrix("DRIFT_MATCH", "dev")
    equiv_ci = np.quantile(
        np.median(difference[:, samples].mean(axis=2), axis=0), [0.05, 0.95]
    ).tolist()
    ratio = (
        result["arms"]["DRIFT_MATCH"]["mean_adapter_l2"]
        / result["arms"]["ASAM05"]["mean_adapter_l2"]
    )
    matched = 0.90 <= ratio <= 1.10
    equivalent = equiv_ci[0] > -MARGIN and equiv_ci[1] < MARGIN
    cross = [
        overlap_readout(
            indexed["ASAM05", seed]["top4"]["candidate"],
            indexed["DRIFT_MATCH", seed]["top4"]["candidate"],
            indexed["CONTROL", seed]["top4"]["families"],
        )["mean"]
        for seed in SEEDS
    ]
    result["drift_interpretation"] = {
        "actual_l2_ratio_drift_over_asam05": ratio,
        "l2_match": matched,
        "paired_asam05_minus_drift_ci90": equiv_ci,
        "equivalence": equivalent,
        "per_seed_top4_overlap_asam05_vs_drift": cross,
        "reading": "consistent with drift control; no established additional flatness benefit"
        if matched and equivalent
        else "drift-control explanation unresolved",
    }
    return result


def run_cell(args):
    assert_lease(args.lock_fd)
    if args.seed not in SEEDS or args.arm not in ARMS:
        raise RuntimeError("unregistered H100 arm/seed")
    mark_phase(args, "loading")
    torch.set_num_threads(4)
    safe_path(args.input)
    safe_path(args.checkpoint)
    for name in ("source_manifest.json", "selected.json", "cpu_exact_fused.py"):
        safe_path(args.input / name)
    if h59.sha256(args.input / "selected.json") != SELECTED_SHA:
        raise RuntimeError("selected roster SHA mismatch")
    if h59.sha256(args.input / "cpu_exact_fused.py") != FUSED_SHA:
        raise RuntimeError("fused model source SHA mismatch")
    h48 = load("h100_h48", h59.H48_PATH)
    selected, train_rows, dev_rows = h55.input_metadata(args.input, h48)
    validate_paths(args.input, train_rows + dev_rows)
    folds = h55.family_folds(train_rows, h48)
    staged = stage_copy(args.checkpoint, args.out.parent / "scratch" / args.out.stem)
    pools, tables = h55.load_rows(args.input, train_rows, h48)
    net, params, guard, checkpoint = h55.build_model(
        args.input, staged, torch.device("cuda:0"), h48
    )
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device("cuda:0"))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cuda:0"))
    ctx = {"net": net, "h48": h48, "guard": guard}
    names = h59.verify_cell_integrity(ctx)
    initial_state = h55.adapter_state(net)
    initial = h55.adapter_digest(initial_state)
    sampler = load("h100_h53b", h55.H53B_PATH)
    training = load("h100_sampling_reference", h55.TRAINING_PATH)
    orders = {cid: np.argsort(pool["fitness"]) for cid, pool in pools.items()}
    state = np.random.get_state()
    np.random.seed(61353)
    probe = sorted(pools)[:16]
    reference = [training.training_indices(pools[c], 192) for c in probe]
    np.random.seed(61353)
    cached = [sampler.cached_training_indices(pools[c], orders[c], 192) for c in probe]
    np.random.set_state(state)
    if any(not np.array_equal(a, b) for a, b in zip(reference, cached, strict=True)):
        raise RuntimeError("cached sampler differs from reference")
    plan = [
        b
        for b in batch_plan(pools, sampler.cached_training_indices, args.seed)
        if b.step_index < 2
    ]
    optimizer = torch.optim.AdamW(params, lr=learning_rate(args.arm), weight_decay=1e-4)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    mark_phase(args, "training")
    losses = []
    for batch in plan:
        # Pair stochastic model operations independently of arm compute count.
        torch.manual_seed(batch.seed)

        def closure():
            prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
            return h59.loss_by_arm(
                "CONTROL",
                prediction,
                h59._batch_targets(pools, batch, prediction.device),
            )

        losses.append(
            optimizer_step(net.named_parameters(), optimizer, closure, args.arm)[0]
        )
    h59.verify_cell_integrity(ctx)
    candidate_state = h55.adapter_state(net)
    final = h55.adapter_digest(candidate_state)
    if initial == final:
        raise RuntimeError("no adapter parameter changed")
    if not args.timing:
        h48.save_adapter_snapshot(
            args.out.with_suffix(".adapter.pt"), h55.adapter_state(net), h59.R882_SHA
        )
    drift = adapter_l2(candidate_state, initial_state)
    for p in params:
        p.requires_grad_(False)
    net.eval()
    h61 = load("h100_h61", HERE.parent / "h61/runner.py")

    def evaluate(packet, predictor):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError("nonfinite terminal predictions")
        return h61.evaluate_pool_predictions(packet, packet, scores), scores

    mark_phase(args, "scoring", outcome_started=True)
    train, _ = evaluate(pools, predict)
    top4 = None
    r882_dev = None
    dev = None
    if not args.timing:
        if h59.sha256(args.input / "selected.json") != SELECTED_SHA:
            raise RuntimeError("selected roster changed before dev read")
        h49 = load("h100_h49", h59.H49_SCORE_PATH)
        rows, dev_pools, dev_tables = h49.load_validation_packet(args.input)
        if [r["context_sha256"] for r in rows] != [
            r["context_sha256"] for r in dev_rows
        ]:
            raise RuntimeError("dev roster changed")
        dt, dm = h48.h46.make_tokens(net, dev_tables, torch.device("cuda:0"))
        dp = h48.make_predictor(net, dev_pools, dt, dm, torch.device("cuda:0"))
        dev, scores = evaluate(dev_pools, dp)
        restore_adapter(net, initial_state)
        try:
            r882_dev, reference_scores = evaluate(dev_pools, dp)
            top4 = top4_overlap(scores, reference_scores, dev_pools)
        finally:
            restore_adapter(net, candidate_state)
    for p in params:
        p.requires_grad_(True)
    h59.verify_cell_integrity(ctx)
    assert_lease(args.lock_fd)
    if h55.adapter_digest(h55.adapter_state(net)) != final:
        raise RuntimeError("candidate weights not restored after reference scoring")
    result = {
        "wave": "H100",
        "arm": args.arm,
        "seed": args.seed,
        "timing_only": args.timing,
        "updates": len(plan),
        "backward_passes": len(plan) * (2 if args.arm in RHO else 1),
        "adapter_l2": drift,
        "learning_rate": learning_rate(args.arm),
        "rho": RHO.get(args.arm),
        "calibration_sha256": h59.sha256(HERE / "calibration.json"),
        "trainable_count": sum(p.numel() for p in params),
        "trainable_names": list(names),
        "dope_before": h59.DOPE_SHA,
        "dope_after": h59.DOPE_SHA,
        "checkpoint_sha256": h59.R882_SHA,
        "manifest_sha256": h59.MANIFEST_SHA,
        "selected_sha256": h59.sha256(args.input / "selected.json"),
        "prereg_sha256": h59.sha256(HERE / "PREREG.md"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": h55.source_hashes(args.input),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
        "folds": [sorted(f) for f in folds],
        "initial_adapter_sha256": initial,
        "final_adapter_sha256": final,
        "final_loss": losses[-1],
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }
    if not args.timing:
        result.update(train=train, dev=dev, top4=top4)
        if r882_dev is not None:
            result["r882_dev"] = r882_dev
    atomic_json(args.out, result)
    mark_phase(args, "complete", outcome_started=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, default="CONTROL")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=39)
    parser.add_argument("--input", type=Path, default=h59.DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=h59.DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--attempt-token")
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("repeated H100 output")
    if args.dry_run:
        atomic_json(args.out, dry_run(args.input))
    else:
        if args.phase != args.out.with_suffix(".phase.json") or args.lock_fd is None:
            raise RuntimeError("H100 cell requires phase and inherited FIFO lease")
        run_cell(args)


if __name__ == "__main__":
    main()
