"""H189 guarded SWA/EMA continuation experiment; H182's source-bound engine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import analysis as stat
import h182_base as base
import treatment as tx

HERE = Path(__file__).resolve().parent


def _sync(path):
    with Path(path).open("rb") as f:
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 only")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    if (
        pins.get("wave") != "H189"
        or pins.get("arms") != ["R882", "CTRL80", "SWA80", "EMA80"]
        or pins.get("seeds") != [str(seed) for seed in stat.SEEDS]
        or pins.get("fold_seed") != stat.FOLD_SEED
        or pins.get("boot_seed") != stat.BOOT_SEED
        or pins.get("max_total_updates") != 1200
        or pins.get("updates_per_cell") != tx.UPDATES
        or pins.get("swa_snapshot_steps") != list(stat.SNAPSHOT_STEPS)
        or pins.get("ema_decay") != tx.EMA_DECAY
    ):
        raise RuntimeError("H189 design bounds mismatch")
    base.verify(pins)
    h178_path = (
        base.REPO
        / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
    )
    h178_receipt = json.loads(h178_path.read_text())
    h177_analysis = (
        base.REPO / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
    )
    if (
        h178_receipt.get("wave") != "H178"
        or h178_receipt.get("status") != "COMPLETE"
        or h178_receipt.get("null_and_harmful_admission_le_005") is not True
        or h178_receipt.get("max_null_or_harmful_admission_rate", 1.0) > 0.05
        or h178_receipt.get("replayed_four_arm_holm_m") != 4
        or h178_receipt.get("h177_analysis_sha256") != base.sha(h177_analysis)
    ):
        raise RuntimeError("pinned H178/H177 provenance receipt mismatch")
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=HERE, text=True
    ).strip()
    tx.validate_launch(
        launch,
        head,
        base.sha(HERE / "manifest.json"),
        base.sha(HERE / "PREREG.md"),
        base.sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
        source_tree=source_tree,
    )
    rt = base.runtime()
    provenance = json.loads((HERE / "provenance.json").read_text())
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE composite pin mismatch")
    train, val = base.cohorts(rt, provenance)
    fold_record = json.loads((HERE / "fold_manifest.json").read_text())
    folds = fold_record["families"]
    if fold_record.get("seed") != stat.FOLD_SEED or folds != stat.family_folds(
        [r["family_id"] for r in train]
    ):
        raise RuntimeError("H189 family fold manifest identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started, updates = time.monotonic(), 0
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        if len(names) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("H189 trainable adapter inventory")
        initial = base.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(module["p"] for module in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("H189 frozen eval/dropout state mismatch")
        rt.check_frozen(net, guard, names, frozen)
        if rt.tensor_hash(base.adapter_state(net, names)) != initial_hash:
            raise RuntimeError("r882 adapter drift at launch")
        if args.smoke:
            base.save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    initial_adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                ),
            )
            return

        reference = json.loads(Path(pins["reference"]).read_text())
        engine = base.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, _ = engine.evaluate(val, native=True, prove_identity=True)
        baseline_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        if (
            not np.array_equal(
                baseline_values, np.asarray(reference["utility"]["per_context"])
            )
            or float(np.mean(baseline_values)) != 1.1377941898164525
        ):
            raise RuntimeError("exact native 91-context DL-Val baseline failed")
        base.save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                families=19,
                mean=1.1377941898164525,
                bit_exact=True,
                initial_adapter_sha256=initial_hash,
            ),
        )
        print("BASELINE PASS: exact native r882 91/91", flush=True)
        if rt.tensor_hash(base.adapter_state(net, names)) != initial_hash:
            raise RuntimeError("r882 adapter changed during baseline gate")
        base.save(
            output / "launch_binding.json",
            {
                **launch,
                "source_tree": source_tree,
                "manifest_sha256": base.sha(HERE / "manifest.json"),
                "launch_sha256": base.sha(args.launch),
            },
        )

        context_position = {r["context_sha256"]: i for i, r in enumerate(train)}
        b_base = np.full((3, len(train)), np.nan, dtype=np.float64)
        b_panel = {arm: np.full_like(b_base, np.nan) for arm in stat.ARMS}
        a_base = np.full_like(b_base, np.nan)
        a_panel = {arm: np.full_like(b_base, np.nan) for arm in stat.ARMS}
        a_ctrl = np.full_like(b_base, np.nan)
        states = {}
        cell_receipts = []

        # Train each cell on FIT. Do not open B or A outcome pools in this phase.
        for outer in range(5):
            fit_fams, b_fams, a_fams = stat.partition(
                {r["family_id"] for r in train}, outer, folds
            )
            fit_rows = [r for r in train if r["family_id"] in fit_fams]
            b_rows = [r for r in train if r["family_id"] in b_fams]
            a_rows = [r for r in train if r["family_id"] in a_fams]
            tx.validate_partitions(fit_fams, b_fams, a_fams)
            engine.load(fit_rows)
            fit_ids = [r["context_sha256"] for r in fit_rows]
            b_ids = [r["context_sha256"] for r in b_rows]
            a_ids = [r["context_sha256"] for r in a_rows]
            orders = {cid: np.argsort(engine.pools[cid]["fitness"]) for cid in fit_ids}
            for seed in stat.SEEDS:
                batch_seed = stat.stream_seed(
                    seed, outer, "H189-H176-continuation-batches-v1"
                )
                plan = stat.batch_plan(fit_ids, orders, batch_seed, updates=tx.UPDATES)
                stat.validate_fit_plan(plan, fit_ids, b_ids + a_ids)
                snapshots, ema_state = {}, None

                def capture(step, state):
                    nonlocal ema_state, snapshots
                    current = {
                        name: tensor.detach().cpu().float().numpy().copy()
                        for name, tensor in state.items()
                    }
                    ema_state = (
                        current
                        if ema_state is None
                        else tx.ema_step(ema_state, current, tx.EMA_DECAY)
                    )
                    if step in stat.SNAPSHOT_STEPS:
                        snapshots[step] = {
                            name: tensor.detach().cpu().float().clone()
                            for name, tensor in state.items()
                        }

                tag = f"fold{outer}-seed{seed}"
                fit = base.run_cell(
                    net,
                    rt,
                    engine,
                    names,
                    frozen,
                    guard,
                    initial,
                    fit_rows,
                    batch_seed,
                    tx.UPDATES,
                    output,
                    tag,
                    plan=plan,
                    excluded_context_ids=b_ids + a_ids,
                    snapshot_callback=capture,
                )
                updates += tx.UPDATES
                if tuple(sorted(snapshots)) != stat.SNAPSHOT_STEPS or ema_state is None:
                    raise RuntimeError("H189 snapshot/EMA trajectory incomplete")
                sw = tx.uniform_swa(
                    {
                        step: {name: tensor.numpy() for name, tensor in snap.items()}
                        for step, snap in snapshots.items()
                    }
                )
                candidates = {
                    "CTRL80": fit["state"],
                    "SWA": {
                        name: torch.from_numpy(value.copy())
                        for name, value in sw.items()
                    },
                    "EMA": {
                        name: torch.from_numpy(value.copy())
                        for name, value in ema_state.items()
                    },
                }
                if (
                    rt.tensor_hash(candidates["SWA"]) == initial_hash
                    or rt.tensor_hash(candidates["EMA"]) == initial_hash
                ):
                    raise RuntimeError("H189 candidate state equals initial r882 state")
                state_path = output / f"candidate-states-{tag}.npz"
                np.savez_compressed(
                    state_path,
                    **{
                        f"{arm}::{name}": tensor.numpy()
                        for arm, candidate in candidates.items()
                        for name, tensor in candidate.items()
                    },
                )
                _sync(state_path)
                states[(outer, seed)] = state_path
                cell_receipts.append(
                    dict(
                        fold=outer,
                        seed=seed,
                        fit_family_count=len(fit_fams),
                        updates=tx.UPDATES,
                        fit_context_ids=sorted(fit_ids),
                        batch_plan_sha256=hashlib.sha256(
                            json.dumps(
                                [
                                    [s, c, [idx.tolist() for idx in ix]]
                                    for s, c, ix in plan
                                ],
                                sort_keys=True,
                            ).encode()
                        ).hexdigest(),
                        snapshot_steps=list(stat.SNAPSHOT_STEPS),
                        ema_decay=tx.EMA_DECAY,
                        trajectory_adapter_sha256=rt.tensor_hash(fit["state"]),
                        swa_adapter_sha256=rt.tensor_hash(candidates["SWA"]),
                        ema_adapter_sha256=rt.tensor_hash(candidates["EMA"]),
                    )
                )
                base.save(output / f"{tag}.json", cell_receipts[-1])
                print("FIT COMPLETE", tag, flush=True)
                del snapshots, ema_state, candidates, fit, plan

        if updates != pins["max_total_updates"]:
            raise RuntimeError(
                "H189 optimizer update count differs from preregistration"
            )

        # Stage 1 is pooled B across all folds and seeds. Lock it before any A read.
        stage = stat.StageOrderGuard()
        stage.mark_b_read()
        for outer in range(5):
            _, b_fams, _ = stat.partition({r["family_id"] for r in train}, outer, folds)
            rows = [r for r in train if r["family_id"] in b_fams]
            engine.load(rows)
            for si, seed in enumerate(stat.SEEDS):
                base.restore(net, names, initial)
                ref, ref_raw = engine.evaluate(rows, native=True)
                for row in rows:
                    b_base[si, context_position[row["context_sha256"]]] = ref[
                        row["context_sha256"]
                    ]
                base.save(
                    output / f"B-costs-fold{outer}-seed{seed}-R882.json",
                    {
                        row["context_sha256"]: {
                            "costs": ref_raw[row["context_sha256"]].tolist(),
                            "score_dtype": str(ref_raw[row["context_sha256"]].dtype),
                            "selected_choice": int(
                                np.argmin(ref_raw[row["context_sha256"]])
                            ),
                            "utility": ref[row["context_sha256"]],
                        }
                        for row in rows
                    },
                )
                archive = np.load(states[(outer, seed)], allow_pickle=False)
                for arm in stat.ARMS:
                    candidate = {
                        name: torch.from_numpy(archive[f"{arm}::{name}"].copy())
                        for name in names
                    }
                    base.restore(net, names, candidate)
                    values, raw = engine.evaluate(rows)
                    for row in rows:
                        b_panel[arm][si, context_position[row["context_sha256"]]] = (
                            values[row["context_sha256"]]
                        )
                    raw_path = output / f"B-costs-fold{outer}-seed{seed}-{arm}.json"
                    base.save(
                        raw_path,
                        {
                            row["context_sha256"]: {
                                "costs": raw[row["context_sha256"]].tolist(),
                                "score_dtype": str(raw[row["context_sha256"]].dtype),
                                "selected_choice": int(
                                    np.argmin(raw[row["context_sha256"]])
                                ),
                                "utility": values[row["context_sha256"]],
                            }
                            for row in rows
                        },
                    )
                base.restore(net, names, initial)
        if not np.isfinite(b_base).all() or any(
            not np.isfinite(v).all() for v in b_panel.values()
        ):
            raise RuntimeError("pooled B panel incomplete")
        decisions = stat.pooled_b_select(train, b_base, b_panel)
        b_npz = output / "pooled_B_panels.npz"
        np.savez_compressed(b_npz, r882=b_base, **b_panel)
        _sync(b_npz)
        lock = output / "pooled_b_selection_lock.json"
        base.save(
            lock,
            dict(
                wave="H189",
                pooled_b_families=205,
                boot_seed=stat.BOOT_SEED,
                decisions=decisions,
                b_panels_file=b_npz.name,
                b_panels_sha256=base.sha(b_npz),
                raw_b_cost_files={
                    p.name: {"sha256": base.sha(p), "bytes": p.stat().st_size}
                    for p in sorted(output.glob("B-costs-*.json"))
                },
                cell_receipts=cell_receipts,
                h178_receipt=h178_receipt,
                h178_use="provenance only; not represented as calibration of H189 m=2",
            ),
        )
        dfd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        stage.mark_b_locked(lock)
        stage.require_b_locked()
        print(
            "POOLED B LOCKED",
            {arm: decisions[arm]["admitted"] for arm in stat.ARMS},
            flush=True,
        )

        # Stage 2: only now open A outcomes.
        for outer in range(5):
            _, _, a_fams = stat.partition({r["family_id"] for r in train}, outer, folds)
            rows = [r for r in train if r["family_id"] in a_fams]
            engine.load(rows)
            for si, seed in enumerate(stat.SEEDS):
                base.restore(net, names, initial)
                ref, _ = engine.evaluate(rows, native=True)
                for row in rows:
                    a_base[si, context_position[row["context_sha256"]]] = ref[
                        row["context_sha256"]
                    ]
                archive = np.load(states[(outer, seed)], allow_pickle=False)
                ctrl = {
                    name: torch.from_numpy(archive[f"CTRL80::{name}"].copy())
                    for name in names
                }
                base.restore(net, names, ctrl)
                ctrl_values, _ = engine.evaluate(rows)
                for row in rows:
                    a_ctrl[si, context_position[row["context_sha256"]]] = ctrl_values[
                        row["context_sha256"]
                    ]
                for arm in stat.ARMS:
                    candidate = {
                        name: torch.from_numpy(archive[f"{arm}::{name}"].copy())
                        for name in names
                    }
                    base.restore(net, names, candidate)
                    values, _ = engine.evaluate(rows)
                    for row in rows:
                        a_panel[arm][si, context_position[row["context_sha256"]]] = (
                            values[row["context_sha256"]]
                        )
                base.restore(net, names, initial)
        if not np.isfinite(a_base).all() or any(
            not np.isfinite(v).all() for v in a_panel.values()
        ):
            raise RuntimeError("pooled A panel incomplete")
        a_results = stat.pooled_a_confirm(train, a_base, a_panel)
        ctrl_result = stat.pooled_a_confirm(train, a_base, {"CTRL80": a_ctrl})["CTRL80"]
        ctrl_result.pop("advance_a_gate")
        ctrl_result["role"] = "descriptive only; excluded from selection and Holm m=2"
        verdicts = tx.verdict(decisions, a_results)
        selected_panel = {
            arm: np.where(decisions[arm]["admitted"], a_panel[arm], a_base)
            for arm in stat.ARMS
        }
        metrics = dict(
            wave="H189",
            design=dict(
                seeds=list(stat.SEEDS),
                fold_seed=stat.FOLD_SEED,
                boot_seed=stat.BOOT_SEED,
                outer_folds=5,
                train_families=205,
                fit_families_per_fold=123,
                b_families_per_fold=41,
                a_families_per_fold=41,
                updates_per_cell=tx.UPDATES,
                trajectories=len(stat.SEEDS) * 5,
                optimizer_updates=updates,
                swa_snapshot_steps=list(stat.SNAPSHOT_STEPS),
                ema_decay=tx.EMA_DECAY,
                holm_m=2,
                pooled_selector_mde80_heuristic=0.0258,
            ),
            exact_baseline=dict(contexts=91, mean=1.1377941898164525, bit_exact=True),
            pooled_b=decisions,
            pooled_a=a_results,
            ctrl80_descriptive=ctrl_result,
            candidate_elo=stat.family_elo_summary(train, a_base, a_panel),
            selected_policy_elo=stat.family_elo_summary(train, a_base, selected_panel),
            utility={
                "R882": stat.utility_summary(train, a_base),
                "CTRL80": stat.utility_summary(train, a_ctrl),
                "SWA80": stat.utility_summary(train, a_panel["SWA"]),
                "EMA80": stat.utility_summary(train, a_panel["EMA"]),
            },
            selected_policy_a={
                arm: stat.pooled_a_confirm(
                    train,
                    a_base,
                    {
                        arm: selected_panel[arm],
                        "EMA" if arm == "SWA" else "SWA": a_base,
                    },
                )[arm]
                for arm in stat.ARMS
            },
            verdict=verdicts,
            metrics_scope="Elo/utility only; no AUC or speed",
            diagnostic_only=True,
            live_flywheel_write=False,
            dope_sha256=rt.DOPE,
        )
        base.save(output / "metrics.json", metrics)
        a_npz = output / "pooled_A_panels.npz"
        np.savez_compressed(a_npz, r882=a_base, CTRL80=a_ctrl, **a_panel)
        _sync(a_npz)
        base.save(
            output / "audit.json",
            dict(
                status="PASS",
                opened=audit.receipt()["opened"],
                refused=audit.receipt()["refused"],
                protected_reads=audit.receipt()["protected_reads"],
                optimizer_updates=updates,
                h178_receipt_holm_m=4,
                h189_holm_m=2,
            ),
        )
        print("COMPLETE", verdicts, flush=True)
    except BaseException as exc:
        base.save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
