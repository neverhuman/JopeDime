"""One guarded H197 CPU attempt: fixed-step adapter continuation with PCGC."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H189_DIR = REPO / "research/flywheel_selfplay/h189_swa_ema"
sys.path.insert(0, str(H189_DIR))
import h182_base as base  # noqa: E402


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load H197 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d = load_local_module("h197_design", HERE / "design.py")
_previous_design = sys.modules.get("design")
sys.modules["design"] = d
try:
    stat = load_local_module("h197_analysis", HERE / "analysis.py")
finally:
    if _previous_design is None:
        del sys.modules["design"]
    else:
        sys.modules["design"] = _previous_design
tx = load_local_module("h197_treatment", HERE / "treatment.py")
source_pins = load_local_module("h197_source_pins", HERE / "source_pins.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    base.save(path, value)


def fsync_dir(path):
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_launch(launch, pins, head, smoke=False):
    if launch.get("manifest_sha256") != sha(HERE / "manifest.json"):
        raise RuntimeError("launch manifest binding mismatch")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or launch.get("source_tree")
        != subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=HERE, text=True
        ).strip()
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != sha(HERE / "PREREG.md")
        or launch.get("fold_manifest_sha256") != sha(HERE / "fold_manifest.json")
        or len(str(launch.get("review_receipt_sha256", ""))) != 64
        or launch.get("science_attempt") != 1
    ):
        raise RuntimeError("H197 launch must bind merged exact source and one attempt")


def load_parent_rows(pins):
    path = Path(pins["catalog_copy"])
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[str(row["id"])] = row
    return rows


def save_state(path, state):
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: torch.from_numpy(archive[name].copy()) for name in archive.files}


def save_panel(path, rows, panel, families):
    cids = [row["context_sha256"] for row in rows]
    costs, offsets = [], [0]
    for cid in cids:
        value = np.asarray(panel[cid]["costs"], dtype=np.float32)
        costs.append(value)
        offsets.append(offsets[-1] + len(value))
    with Path(path).open("xb") as f:
        np.savez_compressed(
            f,
            context_ids=np.asarray(cids),
            family_ids=np.asarray([families[c] for c in cids]),
            utilities=np.asarray(
                [panel[cid]["utility"] for cid in cids], dtype=np.float64
            ),
            choices=np.asarray([panel[cid]["choice"] for cid in cids], dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            costs=np.concatenate(costs).astype(np.float32),
        )
        f.flush()
        os.fsync(f.fileno())


def evaluate_cell(
    engine, rt, net, names, initial, state_path, arm, rows, output, tag, native=False
):
    if arm == "R882":
        base.restore(net, names, initial)
    else:
        base.restore(net, names, load_state(state_path))
    values, raw = engine.evaluate(rows, native=native, prove_identity=native)
    panel = {}
    for row in rows:
        cid = row["context_sha256"]
        panel[cid] = {
            "utility": float(values[cid]),
            "choice": int(np.argmin(raw[cid])),
            "costs": np.asarray(raw[cid], dtype=np.float32),
        }
    save_panel(
        output / f"{tag}-{arm}.npz",
        rows,
        panel,
        {r["context_sha256"]: r["family_id"] for r in rows},
    )
    return values, sha(output / f"{tag}-{arm}.npz")


def forward_losses(engine, rt, stream, cids, indexes):
    kwargs, tokens, masks, fitness, gate = engine.batch(cids, indexes)
    anchor = engine.anchored(kwargs)
    torch.manual_seed(stream)
    prediction = engine.forward(kwargs, tokens, masks, anchor)
    losses = tx.per_context_losses(prediction, fitness, gate)
    live = rt.loss(prediction, fitness, gate)
    if not torch.equal(losses.mean(), live):
        raise RuntimeError(
            "per-context loss does not reproduce the pinned live objective"
        )
    if not torch.isfinite(losses).all():
        raise RuntimeError("non-finite per-context live loss")
    return losses


def calibrate(engine, rt, params, plan, counter):
    """Use five deterministic FIT-only dry batches; never step the optimizer."""
    rows, step0_norms = [], None
    for index, (stream, cids, indexes) in enumerate(plan[:5]):
        losses = forward_losses(engine, rt, stream, cids, indexes)
        gradients, norms = tx.context_gradients(losses, params)
        if index == 0:
            step0_norms = norms.copy()
            C = float(np.median(step0_norms))
            if not np.isfinite(C) or C <= 0:
                raise RuntimeError(
                    "step-0 FIT-only median context norm is not positive"
                )
        factors = tx.clip_factors(norms, C)
        aggregate = tx.aggregate_context_gradients(gradients, factors)
        rows.append(tx.gradient_norm(aggregate))
        counter["dry_context_backward_calls"] += len(cids)
    C, global_C = tx.dry_thresholds(step0_norms, rows)
    return (
        C,
        global_C,
        {
            "step0_context_gradient_norms": step0_norms.tolist(),
            "dry_pcgc_averaged_batch_gradient_norms": rows,
            "dry_fit_batches": 5,
            "dry_context_backward_calls": counter["dry_context_backward_calls"],
        },
    )


def train_arm(
    engine,
    rt,
    net,
    names,
    initial,
    fit_rows,
    plan,
    arm,
    C,
    global_C,
    seed,
    outer,
    output,
    counter,
):
    base.restore(net, names, initial)
    params = [p for p in net.parameters() if p.requires_grad]
    if len(params) != 47 or sum(p.numel() for p in params) != 29211:
        raise RuntimeError("only the 47-tensor/29,211-parameter adapter may train")
    optimizer = torch.optim.AdamW(
        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    fit_ids = {r["context_sha256"] for r in fit_rows}
    trace = []
    for step, (stream, cids, indexes) in enumerate(plan, 1):
        if not set(cids) <= fit_ids:
            raise RuntimeError("non-FIT context in optimizer plan")
        optimizer.zero_grad(set_to_none=True)
        losses = forward_losses(engine, rt, stream, cids, indexes)
        actual_factors = assigned_factors = None
        raw_norm = None
        if arm in ("PCGC40", "PCGC_SHUF40"):
            counter["backward_started"] += len(cids)
            save(output / "update_counter.json", counter)
            gradients, norms = tx.context_gradients(losses, params)
            counter["backward_completed"] += len(cids)
            save(output / "update_counter.json", counter)
            actual_factors = tx.clip_factors(norms, C)
            if arm == "PCGC_SHUF40":
                assigned_factors, permutation = tx.shuffle_factors(
                    actual_factors,
                    d.stream_seed(seed, outer, step, "H197-factor-shuffle-v1"),
                )
            else:
                assigned_factors = actual_factors.copy()
                permutation = list(range(len(cids)))
            aggregate = tx.aggregate_context_gradients(gradients, assigned_factors)
            tx.set_gradients(params, aggregate)
            raw_norm = tx.gradient_norm(aggregate)
        else:
            counter["backward_started"] += 1
            save(output / "update_counter.json", counter)
            losses.mean().backward()
            counter["backward_completed"] += 1
            raw_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0 if arm == "CTRL40" else global_C,
                    error_if_nonfinite=True,
                )
            )
        counter["step_started"] += 1
        save(output / "update_counter.json", counter)
        optimizer.step()
        counter["step_completed"] += 1
        save(output / "update_counter.json", counter)
        rt.check_frozen(net, engine.guard, names, engine.frozen)
        engine.check()
        row = {
            "step": step,
            "stream": stream,
            "loss": float(losses.mean().detach()),
            "raw_or_clipped_batch_gradient_norm": float(raw_norm),
            "learning_rate": 1e-4,
            "per_context_gradient_norms": None
            if actual_factors is None
            else norms.tolist(),
            "clip_factors": None if actual_factors is None else actual_factors.tolist(),
            "assigned_clip_factors": None
            if assigned_factors is None
            else assigned_factors.tolist(),
            "factor_permutation": tx.permutation_to_json(permutation)
            if actual_factors is not None
            else None,
            "fit_context_ids": cids,
            "candidate_indices": [
                np.asarray(ix, dtype=np.int64).tolist() for ix in indexes
            ],
        }
        trace.append(row)
    state = base.adapter_state(net, names)
    if rt.tensor_hash(state) == rt.tensor_hash(initial):
        raise RuntimeError("H197 arm produced an unchanged adapter")
    return state, trace


def fill_panel(target, positions, seed_index, rows, values):
    for row in rows:
        cid = row["context_sha256"]
        if cid in positions and np.isfinite(target[seed_index, positions[cid]]):
            raise RuntimeError("duplicate pooled assessment context")
        target[seed_index, positions[cid]] = values[cid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.preflight:
        raise RuntimeError("H197 smoke and preflight modes are mutually exclusive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("H197 is CPU-only on xbabe2 at nice 19")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    if (
        pins.get("wave") != "H197"
        or pins.get("arms") != ["R882", *d.ARMS]
        or pins.get("seeds") != list(d.SEEDS)
        or pins.get("fold_seeds") != list(d.FOLD_SEEDS)
        or pins.get("boot_seed") != d.BOOT_SEED
        or pins.get("updates_per_cell") != d.UPDATES
        or pins.get("max_total_updates") != 2400
        or pins.get("max_trajectories") != 60
        or pins.get("dry_fit_batches") != 5
    ):
        raise RuntimeError("H197 frozen bounds mismatch")
    shared_pins = dict(pins)
    shared_pins["source_sha256"] = {}
    base.verify(shared_pins)
    source_pins.verify(HERE, pins["source_sha256"])
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    launch = json.loads(args.launch.read_text())
    validate_launch(launch, pins, head, smoke=args.smoke)
    rt = base.runtime()
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("H197 DOPE composite pin mismatch")
    provenance = json.loads((HERE / "provenance.json").read_text())
    train, val = base.cohorts(rt, provenance)
    selected_sha = sha(rt.INPUT / "selected.json")
    if selected_sha != pins["fold_source_selected_sha256"]:
        raise RuntimeError("selected TRAIN metadata source hash mismatch")
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    folds_by_seed = {}
    for seed in d.SEEDS:
        entry = fold_doc["replicates"][str(seed)]
        if entry["fold_seed"] != d.FOLD_SEEDS[d.SEEDS.index(seed)]:
            raise RuntimeError("H197 fold seed mismatch")
        folds_by_seed[seed] = entry["families"]
    expected = d.fold_manifest([r["family_id"] for r in train], selected_sha)
    if fold_doc != expected:
        raise RuntimeError("committed H197 independent fold manifest differs")

    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    counters = {
        "backward_started": 0,
        "backward_completed": 0,
        "step_started": 0,
        "step_completed": 0,
        "dry_context_backward_calls": 0,
    }

    def interrupted(signum, _frame):
        raise RuntimeError("H197 supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        initial = base.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        if (
            len(names) != 47
            or len(params) != 47
            or sum(p.numel() for p in params) != 29211
        ):
            raise RuntimeError("adapter trainable inventory mismatch")
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("H197 requires the live eval/dropout-off mode")
        rt.check_frozen(net, guard, names, frozen)
        base.save(
            output / "launch_binding.json",
            {
                "source_commit": head,
                "source_tree": subprocess.check_output(
                    ["git", "rev-parse", "HEAD^{tree}"], cwd=HERE, text=True
                ).strip(),
                "manifest_sha256": sha(HERE / "manifest.json"),
                "launch_sha256": sha(args.launch),
                "initial_adapter_sha256": initial_hash,
                "dope_sha256": rt.DOPE,
                "fold_manifest_sha256": sha(HERE / "fold_manifest.json"),
            },
        )
        if args.smoke:
            base.save(
                output / "smoke.json",
                {
                    "status": "PASS",
                    "optimizer_updates": 0,
                    "trainable_parameters": 29211,
                    "protected_reads": 0,
                },
            )
            return

        engine = base.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, baseline_costs = engine.evaluate(
            val, native=True, prove_identity=True
        )
        baseline_rows = json.loads(Path(pins["reference"]).read_text())["utility"][
            "per_context"
        ]
        baseline_vector = np.asarray(
            [baseline[r["context_sha256"]] for r in val], dtype=np.float64
        )
        if (
            not np.array_equal(
                baseline_vector, np.asarray(baseline_rows, dtype=np.float64)
            )
            or float(baseline_vector.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native r882 DL-Val 91/91 baseline is not exact")
        with np.load(
            pins["baseline_cost_reference"], allow_pickle=False
        ) as reference_costs:
            if any(
                baseline_costs[c].dtype != np.float32
                or baseline_costs[c].tobytes() != reference_costs[c].tobytes()
                for c in baseline_costs
            ):
                raise RuntimeError("native r882 baseline cost bytes mismatch")
        base.save(
            output / "baseline.json",
            {
                "status": "PASS",
                "contexts": 91,
                "mean_utility": 1.1377941898164525,
                "utility_bit_exact": True,
                "cost_bytes_exact": True,
                "TRAIN_outcome_pools_opened": 0,
                "initial_adapter_sha256": initial_hash,
            },
        )
        print(
            "BASELINE PASS: native r882 91/91; TRAIN outcome pools opened=0", flush=True
        )
        catalog = load_parent_rows(pins)
        dataset_ids = {r["dataset_id"] for r in train + val}
        if not dataset_ids <= catalog.keys() or any(
            catalog[k].get("origin") != "real" for k in dataset_ids
        ):
            raise RuntimeError("real S3 catalog provenance check failed")
        base.save(
            output / "catalog_join.json",
            {"status": "PASS", "datasets": len(dataset_ids), "synthetic": 0},
        )

        if args.preflight:
            engine.load(train)
            preflight = d.validate_preflight(
                baseline_exact=True,
                loaded_pool_count=len(engine.pools),
                expected_pool_count=len(train) + len(val),
                counters=counters,
            )
            preflight.update(
                {
                    "source_commit": head,
                    "source_tree": subprocess.check_output(
                        ["git", "rev-parse", "HEAD^{tree}"], cwd=HERE, text=True
                    ).strip(),
                    "manifest_sha256": sha(HERE / "manifest.json"),
                    "source_pin_count": len(pins["source_sha256"]),
                    "input_pin_count": len(pins["inputs"]),
                    "asset_pin_count": len(pins["assets"]),
                    "train_contexts": len(train),
                    "train_families": len({r["family_id"] for r in train}),
                    "dl_val_contexts": len(val),
                    "trainer_started": False,
                }
            )
            base.save(output / "preflight.json", preflight)
            print(
                "H197 PREFLIGHT PASS: full pinned corpus loaded; zero updates",
                flush=True,
            )
            return

        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        panel_shape = (3, len(train))
        B = {arm: np.full(panel_shape, np.nan) for arm in d.ALL_ARMS}
        A = {arm: np.full(panel_shape, np.nan) for arm in d.ALL_ARMS}
        states, calibration = {}, []

        # FIT only: all four arm states are trained before any B outcome pool is loaded.
        for seed_index, seed in enumerate(d.SEEDS):
            folds = folds_by_seed[seed]
            for outer in range(5):
                fit_rows, b_rows, a_rows = d.partition(train, outer, folds)
                fit_ids = [r["context_sha256"] for r in fit_rows]
                b_ids = [r["context_sha256"] for r in b_rows]
                a_ids = [r["context_sha256"] for r in a_rows]
                engine.load(fit_rows)
                orders = {
                    cid: np.argsort(engine.pools[cid]["fitness"]) for cid in fit_ids
                }
                plan_seed = d.stream_seed(seed, outer, "H197-common-batch-plan-v1")
                plan = d.batch_plan(fit_ids, orders, plan_seed)
                d.validate_fit_plan(plan, fit_ids, b_ids, a_ids)
                base.restore(net, names, initial)
                C, global_C, dry = calibrate(engine, rt, params, plan, counters)
                calibration.append(
                    {
                        "seed": seed,
                        "outer_fold": outer,
                        "C": C,
                        "GCLIP_threshold": global_C,
                        "fit_families": len({r["family_id"] for r in fit_rows}),
                        **dry,
                    }
                )
                cell_dir = output / f"seed{seed}-fold{outer}"
                cell_dir.mkdir()
                plan_sha = hashlib.sha256(
                    json.dumps(
                        [
                            [st, ids, [np.asarray(ix).tolist() for ix in ixrow]]
                            for st, ids, ixrow in plan
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                for arm in d.ARMS:
                    counter_path = cell_dir / f"{arm}-counter.json"
                    counters.update({"cell": f"{seed}-fold{outer}", "arm": arm})
                    save(counter_path, counters)
                    state, trace = train_arm(
                        engine,
                        rt,
                        net,
                        names,
                        initial,
                        fit_rows,
                        plan,
                        arm,
                        C,
                        global_C,
                        seed,
                        outer,
                        cell_dir,
                        counters,
                    )
                    state_path = cell_dir / f"{arm}-adapter.npz"
                    save_state(state_path, state)
                    save(cell_dir / f"{arm}-trace.json", trace)
                    receipt = {
                        "status": "FIT_COMPLETE",
                        "seed": seed,
                        "outer_fold": outer,
                        "arm": arm,
                        "updates": d.UPDATES,
                        "fit_context_count": len(fit_rows),
                        "fit_family_count": 123,
                        "C": C,
                        "GCLIP_threshold": global_C,
                        "batch_plan_sha256": plan_sha,
                        "adapter_sha256": rt.tensor_hash(state),
                        "state_file_sha256": sha(state_path),
                        "trace_sha256": sha(cell_dir / f"{arm}-trace.json"),
                    }
                    save(cell_dir / f"{arm}-receipt.json", receipt)
                    states[(seed, outer, arm)] = state_path
                    print("FIT COMPLETE", seed, outer, arm, flush=True)
        if sum(1 for _ in states) != 60:
            raise RuntimeError("H197 requires exactly 60 completed trajectories")
        if counters["step_completed"] != 2400 or counters["step_started"] != 2400:
            raise RuntimeError("H197 optimizer update total mismatch")
        save(
            output / "calibration.json",
            {
                "status": "PASS",
                "cells": calibration,
                "C_definition": "median eight context gradient norms from the first fixed FIT batch at initial r882 adapter",
                "GCLIP_threshold_definition": "median norm of PCGC-clipped mean gradient across the first five fixed FIT batches, no updates",
            },
        )

        # Pooled B diagnostics, then immutable/fsynced fixed-step lock, then A.
        for seed_index, seed in enumerate(d.SEEDS):
            folds = folds_by_seed[seed]
            for outer in range(5):
                _, b_rows, _ = d.partition(train, outer, folds)
                engine.load(b_rows)
                values, _ = evaluate_cell(
                    engine,
                    rt,
                    net,
                    names,
                    initial,
                    None,
                    "R882",
                    b_rows,
                    output,
                    f"B-seed{seed}-fold{outer}",
                    native=True,
                )
                fill_panel(B["R882"], positions, seed_index, b_rows, values)
                for arm in d.ARMS:
                    values, _ = evaluate_cell(
                        engine,
                        rt,
                        net,
                        names,
                        initial,
                        states[(seed, outer, arm)],
                        arm,
                        b_rows,
                        output,
                        f"B-seed{seed}-fold{outer}",
                    )
                    fill_panel(B[arm], positions, seed_index, b_rows, values)
        if any(not np.isfinite(v).all() for v in B.values()):
            raise RuntimeError("pooled B panel incomplete")
        b_files = sorted(output.glob("B-*.npz"))
        lock = {
            "wave": "H197",
            "source_commit": head,
            "fold_manifest_sha256": sha(HERE / "fold_manifest.json"),
            "policy": "fixed candidate step 40; no B-based step selection",
            "primary_arm": "PCGC40",
            "arms": list(d.ALL_ARMS),
            "pooled_b_families": 205,
            "pooled_b_contexts": 1219,
            "candidate_step": 40,
            "b_panel_files": {p.name: sha(p) for p in b_files},
            "fit_state_files": {
                f"{seed}-{outer}-{arm}": sha(path)
                for (seed, outer, arm), path in states.items()
            },
            "A_scoring_authorized_after_lock": True,
            "B_outcomes_are_selection_diagnostics_only": True,
        }
        lock_path = output / "pooled_b_selection_lock.json"
        save(lock_path, lock)
        fsync_dir(output)
        locked_sha = tx.validate_b_lock(lock_path, [p.name for p in b_files])
        base.save(
            output / "pooled_b_lock_receipt.json",
            {
                "sha256": locked_sha,
                "fsynced": True,
                "candidate_step": 40,
                "selection_rule": "no step selection",
            },
        )
        fsync_dir(output)

        if tx.validate_b_lock(lock_path, [p.name for p in b_files]) != locked_sha:
            raise RuntimeError("pooled B lock changed before A scoring")

        for seed_index, seed in enumerate(d.SEEDS):
            folds = folds_by_seed[seed]
            for outer in range(5):
                _, _, a_rows = d.partition(train, outer, folds)
                engine.load(a_rows)
                values, _ = evaluate_cell(
                    engine,
                    rt,
                    net,
                    names,
                    initial,
                    None,
                    "R882",
                    a_rows,
                    output,
                    f"A-seed{seed}-fold{outer}",
                    native=True,
                )
                fill_panel(A["R882"], positions, seed_index, a_rows, values)
                for arm in d.ARMS:
                    values, _ = evaluate_cell(
                        engine,
                        rt,
                        net,
                        names,
                        initial,
                        states[(seed, outer, arm)],
                        arm,
                        a_rows,
                        output,
                        f"A-seed{seed}-fold{outer}",
                    )
                    fill_panel(A[arm], positions, seed_index, a_rows, values)
        if any(not np.isfinite(v).all() for v in A.values()):
            raise RuntimeError("pooled A panel incomplete")
        ordered_panel = np.stack([A[arm] for arm in d.ALL_ARMS], axis=2)
        A_rows = train
        metrics = stat.summarize(A_rows, ordered_panel)
        save(output / "A_metrics.json", metrics)

        # DL-Val is descriptive only; no selector or ruling reads this summary.
        val_panel = np.zeros((3, len(val), len(d.ALL_ARMS)), dtype=np.float64)
        val_initial, _ = engine.evaluate(val, native=True, prove_identity=True)
        for si in range(3):
            val_panel[si, :, 0] = [val_initial[r["context_sha256"]] for r in val]
            for outer in range(5):
                for ai, arm in enumerate(d.ARMS, 1):
                    values, _ = evaluate_cell(
                        engine,
                        rt,
                        net,
                        names,
                        initial,
                        states[(d.SEEDS[si], outer, arm)],
                        arm,
                        val,
                        output,
                        f"Val-seed{d.SEEDS[si]}-fold{outer}",
                    )
                    val_panel[si, :, ai] += (
                        np.asarray([values[r["context_sha256"]] for r in val]) / 5.0
                    )
        save(
            output / "DLVal_descriptive.json",
            {
                "contexts": 91,
                "families": 19,
                "baseline_mean": 1.1377941898164525,
                "per_seed_mean_utility": {
                    arm: val_panel[:, :, i].mean(axis=1).tolist()
                    for i, arm in enumerate(d.ALL_ARMS)
                },
                "role": "descriptive only; no selection or advancement",
            },
        )
        clip_records = []
        for (_, _, arm), state_path in states.items():
            if arm not in ("PCGC40", "PCGC_SHUF40"):
                continue
            trace_path = state_path.with_name(arm + "-trace.json")
            for step in json.loads(trace_path.read_text()):
                factors = step["assigned_clip_factors"]
                clip_records.extend((arm, float(v)) for v in factors)
        clip_metrics = {}
        for arm in ("PCGC40", "PCGC_SHUF40"):
            values = np.asarray(
                [v for a, v in clip_records if a == arm], dtype=np.float64
            )
            clip_metrics[arm] = {
                "context_gradient_count": int(len(values)),
                "clipped_fraction": float(np.mean(values < 1.0)),
                "factor_quantiles": np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            }
        metrics["per_context_clip_stats"] = clip_metrics
        metrics["baseline"] = {
            "native_91_91_bit_exact": True,
            "mean_utility": 1.1377941898164525,
        }
        metrics["optimizer_updates"] = counters["step_completed"]
        audit_receipt = audit.receipt()
        save(output / "audit.json", audit_receipt)
        if audit_receipt["refused"] or audit_receipt["protected_reads"] != 0:
            raise RuntimeError("H197 audit refusal/protected read")
        metrics["audit"] = audit_receipt
        save(output / "metrics.json", metrics)
        print("H197 COMPLETE", metrics["verdict"], "updates=2400", flush=True)
    except BaseException as exc:
        save(
            output / "failure.json",
            {
                "status": "INCOMPLETE",
                "error": repr(exc),
                "counters": counters,
                "audit": audit.receipt(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
