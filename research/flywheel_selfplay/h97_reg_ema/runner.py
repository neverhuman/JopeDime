#!/usr/bin/env python3
"""H97 candidate-generation cell for weight decay, hidden dropout, and EMA."""

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
ARMS = ("CONTROL", "WD", "DROP", "EMA")
SEEDS = (33, 34, 35)
MARGIN = 0.001546
LCB_TIE_TOL = 1e-12
ROUNDS = 20
UPDATES_PER_ROUND = 2
BATCH_SIZE = 16
EMA_DECAY = 0.9
DROP_P = 0.1


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


science = load("h97_h76_science", HERE.parent / "h76_sam_steps/runner.py")
h59, h55 = science.h59, science.h55
batch_plan = science.batch_plan
optimizer_step = science.optimizer_step
safe_path = science.safe_path
validate_paths = science.validate_paths
assert_lease = science.assert_lease
SELECTED_SHA, FUSED_SHA = science.SELECTED_SHA, science.FUSED_SHA


def make_update_plan(batches, arm):
    """Use the first two H76 streams for two matched updates in each round."""
    if arm not in ARMS:
        raise ValueError(f"unregistered H97 arm: {arm}")
    plan = []
    for round_index in range(1, ROUNDS + 1):
        round_batches = [b for b in batches if b.round_index == round_index]
        if len(round_batches) != 8:
            raise RuntimeError("H97 requires the unchanged eight-stream H76 round plan")
        plan.extend((batch,) for batch in round_batches[:UPDATES_PER_ROUND])
    if len(plan) != ROUNDS * UPDATES_PER_ROUND:
        raise RuntimeError("H97 fixed 40-update plan changed")
    return plan


class HiddenActivationDropout:
    """Apply train-only dropout to adapter SiLU hidden activations, not the frozen anchor."""

    def __init__(self, adapter, probability=DROP_P):
        if probability != DROP_P:
            raise ValueError("H97 hidden dropout is fixed at p=0.1")
        self.probability = probability
        self.enabled = True
        self.handles = [
            module.register_forward_hook(self._drop)
            for name, module in adapter.named_modules()
            if not name.startswith("anchor.")
            if isinstance(module, torch.nn.SiLU)
        ]
        if not self.handles:
            raise RuntimeError("H97 DROP requires adapter hidden SiLU activations")

    def _drop(self, _module, _inputs, output):
        if not self.enabled:
            return output
        return torch.nn.functional.dropout(output, p=self.probability, training=True)

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def ema_update(shadow, net, decay=EMA_DECAY):
    if decay != EMA_DECAY:
        raise ValueError("H97 EMA decay is fixed at 0.9")
    parameters = {
        name: parameter
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(shadow):
        raise RuntimeError("EMA adapter parameter inventory changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if shadow[name].shape != parameter.shape:
                raise RuntimeError("EMA adapter parameter shape changed")
            shadow[name].mul_(decay).add_(parameter.detach().cpu(), alpha=1 - decay)


def attempt_scratch_path(root, cell, attempt):
    if not isinstance(cell, str) or not cell or "/" in cell or ".." in cell:
        raise ValueError("invalid H97 cell name")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("H97 attempt must be a positive integer")
    path = Path(root) / "scratch" / f"{cell}-attempt{attempt}"
    if path.exists():
        raise RuntimeError("H97 attempt scratch already exists")
    return path


def top4_indices(packet, predictions_by_cid):
    """Stable top four among unrevealed candidates 4..P-1 for each dev context."""
    result = {}
    for cid in sorted(packet):
        values = np.asarray(predictions_by_cid[cid], dtype=np.float64)
        if values.ndim != 1 or len(values) <= 7 or not np.isfinite(values).all():
            raise RuntimeError(
                "H97 top-four diagnostic requires finite unrevealed scores"
            )
        order = np.argsort(values[4:], kind="stable")[:4] + 4
        result[cid] = [int(index) for index in order]
    return result


def top4_overlap(candidate, reference):
    if set(candidate) != set(reference) or not candidate:
        raise RuntimeError("H97 top-four diagnostic context roster mismatch")
    scores = []
    for cid, chosen in candidate.items():
        base = reference[cid]
        if (
            len(chosen) != 4
            or len(set(chosen)) != 4
            or len(base) != 4
            or len(set(base)) != 4
        ):
            raise RuntimeError("H97 top-four diagnostic requires four unique indices")
        scores.append(len(set(chosen) & set(base)) / 4.0)
    return float(np.mean(scores))


def pseudoreplication(contexts, families, scalars):
    """Match H90's exact seed-vector grouping, including identical scalar utilities."""

    def groups(values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] != len(SEEDS) or not np.isfinite(values).all():
            raise ValueError("three complete finite seed vectors required")
        result = []
        for index, seed in enumerate(SEEDS):
            for group in result:
                if np.array_equal(values[index], values[SEEDS.index(group[0])]):
                    group.append(seed)
                    break
            else:
                result.append([seed])
        return result

    result = {
        "context_utility_groups": groups(contexts),
        "family_utility_groups": groups(families),
        "scalar_utility_groups": groups(scalars),
    }
    result["identical_utility_flag"] = any(
        len(group) > 1 for grouping in result.values() for group in grouping
    )
    result["independent_seed_replicates_claimed"] = 0
    result["interval_unit"] = "19 paired dev families; complete seed vectors retained"
    return result


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
        raise RuntimeError("all twelve unique H97 paired cells required")
    identities = set()
    families = sorted(indexed["CONTROL", SEEDS[0]]["dev"]["per_family"])
    if len(families) != 19:
        raise RuntimeError("nineteen dev families required")
    for cell in cells:
        if (
            cell["wave"] != "H97"
            or cell["timing_only"]
            or cell["trainable_count"] != 29211
            or cell["updates"] != 40
            or cell["dope_before"] != h59.DOPE_SHA
            or cell["dope_after"] != h59.DOPE_SHA
            or cell["checkpoint_sha256"] != h59.R882_SHA
            or cell["manifest_sha256"] != h59.MANIFEST_SHA
            or cell["selected_sha256"] != SELECTED_SHA
            or cell["initial_adapter_sha256"] == cell["final_adapter_sha256"]
        ):
            raise RuntimeError("H97 cell integrity mismatch")
        if sorted(cell["dev"]["per_family"]) != families:
            raise RuntimeError("dev family roster mismatch")
        if (
            not np.isfinite(float(cell["adapter_l2_from_r882"]))
            or cell["adapter_l2_from_r882"] < 0
        ):
            raise RuntimeError("H97 adapter-distance diagnostic is invalid")
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
    samples = np.random.default_rng(95000).integers(0, 19, size=(draws, 19))
    result = {
        "schema": "jope.h97.metrics/v1",
        "complete": True,
        "seeds": list(SEEDS),
        "reference_elo": 1500,
        "nominal_comparisons": 3,
        "arms": {},
    }
    for arm in ARMS:
        contexts = sorted(indexed["CONTROL", SEEDS[0]]["dev"]["per_context"])
        if len(contexts) != 91 or any(
            sorted(indexed[arm, seed]["dev"]["per_context"]) != contexts
            or sorted(indexed[arm, seed]["dev_top4_indices"]) != contexts
            or sorted(indexed["CONTROL", seed]["r882_top4_indices"]) != contexts
            for seed in SEEDS
        ):
            raise RuntimeError("dev context/top4 roster mismatch")
        utilities = np.asarray(
            [[indexed[arm, s]["dev"]["per_family"][f] for f in families] for s in SEEDS]
        )
        context_utilities = np.asarray(
            [
                [
                    indexed[arm, seed]["dev"]["per_context"][context]
                    for context in contexts
                ]
                for seed in SEEDS
            ]
        )
        gains = utilities - controls
        if not np.isfinite(gains).all():
            raise RuntimeError("nonfinite arm outcome")
        per_seed = gains.mean(axis=1)
        point = float(np.median(per_seed))
        boot = np.median(gains[:, samples].mean(axis=2), axis=0)
        lcb = float(np.quantile(boot, 0.05))
        pseudo_replicate = pseudoreplication(
            context_utilities,
            utilities,
            utilities.mean(axis=1),
        )
        overlaps = [
            top4_overlap(
                indexed[arm, seed]["dev_top4_indices"],
                indexed["CONTROL", seed]["r882_top4_indices"],
            )
            for seed in SEEDS
        ]
        distances = [
            float(indexed[arm, seed]["adapter_l2_from_r882"]) for seed in SEEDS
        ]
        result["arms"][arm] = {
            "per_seed_val_reg_gain": per_seed.tolist(),
            "median_gain": point,
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": lcb,
            "bonferroni_lcb": float(np.quantile(boot, 0.05 / 3)),
            "admitted": (
                arm != "CONTROL" and lcb > MARGIN + LCB_TIE_TOL and point >= 0
            ),
            "seed_pseudoreplication": pseudo_replicate,
            "elo_vs_control": elo_summary(gains, samples),
            "elo_vs_r882": elo_summary(utilities - reference, samples),
            "top4_overlap_with_r882_per_seed": dict(zip(SEEDS, overlaps, strict=True)),
            "top4_overlap_with_r882_median": float(np.median(overlaps)),
            "adapter_l2_from_r882_per_seed": dict(zip(SEEDS, distances, strict=True)),
            "adapter_l2_from_r882_median": float(np.median(distances)),
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
        raise RuntimeError("unregistered H97 arm/seed")
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
    h48 = load("h97_h48", h59.H48_PATH)
    selected, train_rows, dev_rows = h55.input_metadata(args.input, h48)
    validate_paths(args.input, train_rows + dev_rows)
    folds = h55.family_folds(train_rows, h48)
    cell = f"{args.arm}-seed{args.seed}"
    scratch = attempt_scratch_path(args.out.parent, cell, args.attempt)
    staged = h59.stage_r882_copy(args.checkpoint, scratch)
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
    sampler = load("h97_h53b", h55.H53B_PATH)
    training = load("h97_sampling_reference", h55.TRAINING_PATH)
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
    plan = make_update_plan(
        batch_plan(pools, sampler.cached_training_indices, args.seed), args.arm
    )
    learning_rate = 1e-4
    weight_decay = 1e-2 if args.arm == "WD" else 1e-4
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    mark_phase(args, "training")
    losses = []
    dropout = HiddenActivationDropout(net.adapter_10tok) if args.arm == "DROP" else None
    ema_state = (
        {name: tensor.clone() for name, tensor in initial_state.items()}
        if args.arm == "EMA"
        else None
    )
    for microbatches in plan:

        def loss_for(batch):
            # Pair stochastic model operations by the unchanged H76 stream.
            torch.manual_seed(batch.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch.seed)
            prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
            return h59.loss_by_arm(
                "CONTROL",
                prediction,
                h59._batch_targets(pools, batch, prediction.device),
            )

        losses.append(
            optimizer_step(
                net.named_parameters(),
                optimizer,
                lambda: loss_for(microbatches[0]),
                "CONTROL",
            )[0]
        )
        if ema_state is not None:
            ema_update(ema_state, net)
    if dropout is not None:
        dropout.close()
    h59.verify_cell_integrity(ctx)
    if ema_state is not None:
        restore_adapter(net, ema_state)
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
    h61 = load("h97_h61", HERE.parent / "h61/runner.py")

    def evaluate(packet, predictor, *, keep_top4=False):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError("nonfinite terminal predictions")
        report = h61.evaluate_pool_predictions(packet, packet, scores)
        return (report, top4_indices(packet, scores)) if keep_top4 else report

    mark_phase(args, "scoring", outcome_started=True)
    train = None if args.timing else evaluate(pools, predict)
    r882_dev = None
    dev = None
    dev_top4 = None
    r882_top4 = None
    adapter_l2 = float(
        torch.sqrt(
            sum(
                torch.sum(
                    (candidate_state[name] - initial_state[name]).to(torch.float64) ** 2
                )
                for name in initial_state
            )
        )
        .detach()
        .cpu()
    )
    if not args.timing:
        if h59.sha256(args.input / "selected.json") != SELECTED_SHA:
            raise RuntimeError("selected roster changed before dev read")
        h49 = load("h97_h49", h59.H49_SCORE_PATH)
        rows, dev_pools, dev_tables = h49.load_validation_packet(args.input)
        if [r["context_sha256"] for r in rows] != [
            r["context_sha256"] for r in dev_rows
        ]:
            raise RuntimeError("dev roster changed")
        dt, dm = h48.h46.make_tokens(net, dev_tables, torch.device("cuda:0"))
        dp = h48.make_predictor(net, dev_pools, dt, dm, torch.device("cuda:0"))
        dev, dev_top4 = evaluate(dev_pools, dp, keep_top4=True)
        if args.arm == "CONTROL":
            restore_adapter(net, initial_state)
            try:
                r882_dev, r882_top4 = evaluate(dev_pools, dp, keep_top4=True)
            finally:
                restore_adapter(net, candidate_state)
    for p in params:
        p.requires_grad_(True)
    h59.verify_cell_integrity(ctx)
    assert_lease(args.lock_fd)
    if h55.adapter_digest(h55.adapter_state(net)) != final:
        raise RuntimeError("candidate weights not restored after reference scoring")
    result = {
        "wave": "H97",
        "arm": args.arm,
        "seed": args.seed,
        "timing_only": args.timing,
        "updates": len(plan),
        "backward_passes": sum(len(update) for update in plan),
        "effective_batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_activation_dropout_p": DROP_P if args.arm == "DROP" else 0.0,
        "ema_decay": EMA_DECAY if args.arm == "EMA" else None,
        "ema_updates": len(plan) if args.arm == "EMA" else 0,
        "staged_checkpoint_path": str(staged),
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
        "source_hashes": dict(
            h55.source_hashes(args.input),
            h97_runner=h59.sha256(HERE / "runner.py"),
            h97_coordinator=h59.sha256(HERE / "run_wave.py"),
            h97_prereg=h59.sha256(HERE / "PREREG.md"),
        ),
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
        "adapter_l2_from_r882": adapter_l2,
        "final_loss": losses[-1],
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }
    if not args.timing:
        result.update(train=train, dev=dev, dev_top4_indices=dev_top4)
        if r882_dev is not None:
            result["r882_dev"] = r882_dev
            result["r882_top4_indices"] = r882_top4
    else:
        result["train_summary_discarded"] = True
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
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if (
        args.attempt < 1
        or args.out.exists()
        or args.phase != args.out.with_suffix(".phase.json")
    ):
        raise RuntimeError("repeated cell or unexpected phase path")
    run_cell(args)


if __name__ == "__main__":
    main()
