#!/usr/bin/env python3
"""H82 scientific cell: unchanged H76 updates, fresh seeds, durable scoring boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("CONTROL", "K8", "SAM", "ASAM")
SEEDS = (24, 25, 26)
MARGIN = 0.001546


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


science = load("h82_h76_science", HERE.parent / "h76_sam_steps/runner.py")
h59, h55 = science.h59, science.h55
batch_plan = science.batch_plan
optimizer_step = science.optimizer_step
safe_path = science.safe_path
validate_paths = science.validate_paths
assert_lease = science.assert_lease
SELECTED_SHA, FUSED_SHA = science.SELECTED_SHA, science.FUSED_SHA


def atomic_json(path, value):
    h59.check_writer_disk_budget()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mark_phase(args, stage, *, outcome_started=False):
    previous = json.loads(args.phase.read_text())
    if previous.get("attempt_token") != args.attempt_token:
        raise RuntimeError("attempt phase token mismatch")
    if previous.get("outcome_started") and not outcome_started:
        raise RuntimeError("refusing to clear the scoring boundary")
    atomic_json(
        args.phase, dict(previous, stage=stage, outcome_started=outcome_started)
    )


def restore_adapter(net, state):
    """Restore only the verified adapter inventory, even while inference is frozen."""
    named = dict(net.named_parameters())
    if sum(t.numel() for t in state.values()) != 29211 or any(
        not n.startswith("adapter_10tok.") or ".anchor." in n or n.endswith(".anchor")
        for n in state
    ):
        raise RuntimeError("invalid reference adapter inventory")
    with torch.no_grad():
        for name, tensor in state.items():
            if name not in named or named[name].shape != tensor.shape:
                raise RuntimeError("reference adapter shape mismatch")
            named[name].copy_(tensor.to(named[name].device))


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
    if len(cells) != 12 or set(indexed) != {(a, s) for a in ARMS for s in SEEDS}:
        raise RuntimeError("all twelve unique H82 paired cells required")
    identities = set()
    families = sorted(indexed["CONTROL", 24]["dev"]["per_family"])
    if len(families) != 19:
        raise RuntimeError("nineteen dev families required")
    for cell in cells:
        if (
            cell["wave"] != "H82"
            or cell["timing_only"]
            or cell["trainable_count"] != 29211
            or cell["updates"] != (160 if cell["arm"] == "K8" else 40)
            or cell["dope_before"] != h59.DOPE_SHA
            or cell["dope_after"] != h59.DOPE_SHA
            or cell["checkpoint_sha256"] != h59.R882_SHA
            or cell["manifest_sha256"] != h59.MANIFEST_SHA
            or cell["selected_sha256"] != SELECTED_SHA
            or cell["initial_adapter_sha256"] == cell["final_adapter_sha256"]
        ):
            raise RuntimeError("H82 cell integrity mismatch")
        if sorted(cell["dev"]["per_family"]) != families:
            raise RuntimeError("dev family roster mismatch")
        identities.add(
            (
                cell["source_commit"],
                cell["prereg_sha256"],
                str(cell["folds"]),
                cell["initial_adapter_sha256"],
            )
        )
    if len(identities) != 1:
        raise RuntimeError("source/prereg/fold/initial adapter mismatch")
    reference = np.asarray(
        [
            [indexed["CONTROL", s]["r882_dev"]["per_family"][f] for f in families]
            for s in SEEDS
        ]
    )
    controls = np.asarray(
        [
            [indexed["CONTROL", s]["dev"]["per_family"][f] for f in families]
            for s in SEEDS
        ]
    )
    if not np.isfinite(reference).all() or not np.isfinite(controls).all():
        raise RuntimeError("nonfinite reference scores")
    samples = np.random.default_rng(76000).integers(0, 19, size=(draws, 19))
    result = {
        "schema": "jope.h82.metrics/v1",
        "complete": True,
        "seeds": list(SEEDS),
        "reference_elo": 1500,
        "nominal_comparisons": 3,
        "arms": {},
    }
    for arm in ARMS:
        utilities = np.asarray(
            [[indexed[arm, s]["dev"]["per_family"][f] for f in families] for s in SEEDS]
        )
        gains = utilities - controls
        if not np.isfinite(gains).all():
            raise RuntimeError("nonfinite arm outcome")
        per_seed = gains.mean(axis=1)
        point = float(np.median(per_seed))
        boot = np.median(gains[:, samples].mean(axis=2), axis=0)
        lcb = float(np.quantile(boot, 0.05))
        result["arms"][arm] = {
            "per_seed_val_reg_gain": per_seed.tolist(),
            "median_gain": point,
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": lcb,
            "bonferroni_lcb": float(np.quantile(boot, 0.05 / 3)),
            "admitted": arm != "CONTROL" and lcb > MARGIN and point >= 0,
            "elo_vs_control": elo_summary(gains, samples),
            "elo_vs_r882": elo_summary(utilities - reference, samples),
            "seeds": [
                {
                    "seed": s,
                    "train_reg": indexed[arm, s]["train"]["mean"],
                    "val_reg": indexed[arm, s]["dev"]["mean"],
                    "gap": indexed[arm, s]["train"]["mean"]
                    - indexed[arm, s]["dev"]["mean"],
                }
                for s in SEEDS
            ],
        }
    return result


def run_cell(args):
    assert_lease(args.lock_fd)
    if args.seed not in SEEDS or args.arm not in ARMS:
        raise RuntimeError("unregistered H82 arm/seed")
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
    h48 = load("h82_h48", h59.H48_PATH)
    selected, train_rows, dev_rows = h55.input_metadata(args.input, h48)
    validate_paths(args.input, train_rows + dev_rows)
    folds = h55.family_folds(train_rows, h48)
    staged = h59.stage_r882_copy(args.checkpoint, args.out.parent / "scratch")
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
    sampler = load("h82_h53b", h55.H53B_PATH)
    training = load("h82_sampling_reference", h55.TRAINING_PATH)
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
        if args.arm == "K8" or b.step_index < 2
    ]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
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
    for p in params:
        p.requires_grad_(False)
    net.eval()
    h61 = load("h82_h61", HERE.parent / "h61/runner.py")

    def evaluate(packet, predictor):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError("nonfinite terminal predictions")
        return h61.evaluate_pool_predictions(packet, packet, scores)

    mark_phase(args, "scoring", outcome_started=True)
    train = evaluate(pools, predict)
    r882_dev = None
    dev = None
    if not args.timing:
        if h59.sha256(args.input / "selected.json") != SELECTED_SHA:
            raise RuntimeError("selected roster changed before dev read")
        h49 = load("h82_h49", h59.H49_SCORE_PATH)
        rows, dev_pools, dev_tables = h49.load_validation_packet(args.input)
        if [r["context_sha256"] for r in rows] != [
            r["context_sha256"] for r in dev_rows
        ]:
            raise RuntimeError("dev roster changed")
        dt, dm = h48.h46.make_tokens(net, dev_tables, torch.device("cuda:0"))
        dp = h48.make_predictor(net, dev_pools, dt, dm, torch.device("cuda:0"))
        dev = evaluate(dev_pools, dp)
        if args.arm == "CONTROL":
            restore_adapter(net, initial_state)
            try:
                r882_dev = evaluate(dev_pools, dp)
            finally:
                restore_adapter(net, candidate_state)
    for p in params:
        p.requires_grad_(True)
    h59.verify_cell_integrity(ctx)
    assert_lease(args.lock_fd)
    if h55.adapter_digest(h55.adapter_state(net)) != final:
        raise RuntimeError("candidate weights not restored after reference scoring")
    result = {
        "wave": "H82",
        "arm": args.arm,
        "seed": args.seed,
        "timing_only": args.timing,
        "updates": len(plan),
        "backward_passes": len(plan) * (2 if args.arm in ("SAM", "ASAM") else 1),
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
        result.update(train=train, dev=dev)
        if r882_dev is not None:
            result["r882_dev"] = r882_dev
    atomic_json(args.out, result)
    mark_phase(args, "complete", outcome_started=True)
    staged.unlink()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--input", type=Path, default=h59.DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=h59.DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--attempt-token", required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or args.phase != args.out.with_suffix(".phase.json"):
        raise RuntimeError("repeated cell or unexpected phase path")
    run_cell(args)


if __name__ == "__main__":
    main()
