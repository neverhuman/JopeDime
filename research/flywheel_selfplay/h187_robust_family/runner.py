"""H187 single guarded H180-recipe run: FIT-only training, pooled-B lock, then A."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
import design as d
import objectives as obj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(m):
    for kind in ["inputs", "assets", "parent_sources", "source_sha256"]:
        for name, digest in m[kind].items():
            path = (
                ROOT / name
                if kind == "parent_sources"
                else HERE / name
                if kind == "source_sha256"
                else Path(name)
            )
            if sha(path) != digest:
                raise RuntimeError("pin mismatch " + str(path))
    for name, version in m["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch " + name)


def parent():
    path = ROOT / "research/flywheel_selfplay/h180_adapter_ablation/runner.py"
    sys.path.insert(0, str(path.parent))
    return d.sibling("h187_h180", str(path.relative_to(ROOT)))


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
        os.uname().nodename != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 only")
    manifest = json.loads((HERE / "manifest.json").read_text())
    verify(manifest)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if not args.smoke and not (
        launch["source_commit"] == head
        and launch["merged_source_verified"]
        and launch["manifest_sha256"] == sha(HERE / "manifest.json")
        and launch["review_receipt"]
    ):
        raise RuntimeError("launch/review/source binding")
    out = args.output.resolve()
    out.mkdir()
    progress = dict(
        optimizer_calls_started=0,
        optimizer_calls_completed=0,
        backward_calls_started=0,
        backward_calls_completed=0,
        phase="PREFLIGHT",
    )
    if args.smoke:
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                model_loads=0,
                scored_contexts=0,
                optimizer_updates=0,
                pins_verified=True,
            ),
        )
        return

    def interrupted(signum, frame):
        raise InterruptedError("terminal signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        p = parent()
        rt = p.runtime()
        train, val = p.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
        if (
            d.folds(train)
            != json.loads((HERE / "fold_manifest.json").read_text())["families"]
        ):
            raise RuntimeError("fold pin")
        allowed = (
            set(manifest["inputs"])
            | set(manifest["assets"])
            | {str(args.launch.resolve())}
            | {
                str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
                for r in train + val
            }
        )
        audit = rt.Audit(allowed, out)
        sys.addaudithook(audit.check)
        net, guard, names, frozen, fused = rt.build(manifest["checkpoint"])
        initial = p.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        whole_before = rt.tensor_hash(net.state_dict())
        dope_before = guard.snapshot_dope_weights(net).composite_sha256
        params = [x for x in net.parameters() if x.requires_grad]
        if (
            len(names) != 47
            or len(params) != 47
            or sum(x.numel() for x in params) != 29211
        ):
            raise RuntimeError("only29211 adapter parameters")
        engine = p.Engine(net, rt, names, frozen, guard, fused, train + val)

        def check():
            disk = psutil.disk_usage("/home/ubuntu")
            proc = psutil.Process()
            if (
                os.getloadavg()[0] > 0.85 * os.sysconf("SC_NPROCESSORS_ONLN")
                or psutil.virtual_memory().available < 16 * 2**30
                or disk.free < 40e9
                or disk.used / (disk.used + disk.free) >= 0.938
                or proc.memory_info().rss >= 6e9
                or proc.num_threads() > 3
            ):
                raise RuntimeError("terminal resource bound")
            engine.check()
            if {name for name, x in net.named_parameters() if x.requires_grad} != set(
                names
            ):
                raise RuntimeError("trainability drift")

        check()
        engine.load(val)
        baseline, raw = engine.evaluate(val, native=True, prove_identity=True)
        actual = np.array(
            [baseline[r["context_sha256"]] for r in val], dtype=np.float64
        )
        ref = np.array(
            json.loads(Path(manifest["reference"]).read_text())["utility"][
                "per_context"
            ],
            dtype=np.float64,
        )
        if actual.tobytes() != ref.tobytes() or actual.mean() != 1.1377941898164525:
            raise RuntimeError("exact native91 utility gate")
        with np.load(manifest["baseline_cost_reference"], allow_pickle=False) as z:
            if not all(raw[c].tobytes() == z[c].tobytes() for c in raw):
                raise RuntimeError("exact native91 cost gate")
        save(
            out / "baseline.json",
            dict(
                status="PASS",
                native_contexts=91,
                mean=float(actual.mean()),
                bit_exact=True,
                candidate_cost_bytes_exact=True,
                initial_adapter_sha256=initial_hash,
                DOPE_before=dope_before,
                whole_model_sha256=whole_before,
                optimizer_updates=0,
                TRAIN_outcome_pools_opened=0,
            ),
        )
        print(
            "BASELINE PASS native91 exact utilities/costs; frozen DOPE; only29211 adapter trainable",
            flush=True,
        )
        # TRAIN outcome access begins only after baseline; all cyclic folds stay role-local.
        catalog = {
            r["id"]: r
            for r in map(
                json.loads, Path(manifest["catalog_copy"]).read_text().splitlines()
            )
        }
        ids = {r["dataset_id"] for r in train + val}
        if len(ids) != 276 or not all(catalog[k]["origin"] == "real" for k in ids):
            raise RuntimeError("real S3 source")
        save(
            out / "provenance.json",
            dict(
                status="PASS",
                zero_synthetic_datasets=True,
                real_S3_dataset_ids=276,
                TRAIN_contexts=1219,
                TRAIN_families=205,
                DLVal_contexts=91,
                DLVal_families=19,
            ),
        )
        engine.load(train)
        r882, rcost = engine.evaluate(train, native=True)
        np.savez_compressed(out / "r882-costs.npz", **rcost)
        del rcost
        ids = [r["context_sha256"] for r in train]
        index = {c: i for i, c in enumerate(ids)}
        base = np.array([r882[c] for c in ids])
        B = np.full((3, len(train), 4), np.nan)
        A = B.copy()
        V = np.full((3, 5, len(val), 4), np.nan)
        B[:, :, 0] = base
        A[:, :, 0] = base
        V[:, :, :, 0] = actual
        records = []
        execution = None
        stage = None
        initial_hashes = []
        fit_proofs = []

        def persist_scores(tag, rows, values, costs, phase):
            np.savez_compressed(out / f"{tag}-{phase}-scores.npz", **costs)
            return dict(
                context_ids=[r["context_sha256"] for r in rows],
                values=[values[r["context_sha256"]] for r in rows],
                choices=[int(np.argmin(costs[r["context_sha256"]])) for r in rows],
            )

        def train_until(arm, fit, plan, start, end, optimizer, trace, times, tag):
            progress.update(phase="FIT", cell=tag, arm=arm)
            for step in range(start, end + 1):
                stream, cids, indices = plan[step - 1]
                d.require_fit(cids, fit)
                tick = time.monotonic()
                kwargs, tokens, masks, fitness, gate = engine.batch(cids, indices)
                anchor = engine.anchored(kwargs)
                torch.manual_seed(stream)
                optimizer.zero_grad(set_to_none=True)
                pred = engine.forward(kwargs, tokens, masks, anchor)
                if arm == d.ARMS[1]:
                    loss, extra = obj.normalized_loss(pred, anchor[0], fitness, gate)
                else:
                    loss = rt.loss(pred, fitness, gate)
                    extra = None
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                progress["backward_calls_started"] += 1
                save(out / "progress.json", progress)
                loss.backward()
                progress["backward_calls_completed"] += 1
                norm = float(
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                )
                progress["optimizer_calls_started"] += 1
                save(out / "progress.json", progress)
                optimizer.step()
                progress["optimizer_calls_completed"] += 1
                save(out / "progress.json", progress)
                check()
                times.append(time.monotonic() - tick)
                rec = dict(
                    step=step,
                    stream=stream,
                    cids=cids,
                    indexes=[a.tolist() for a in indices],
                    loss=float(loss.detach()),
                    raw_grad_norm=norm,
                    lr=1e-4,
                )
                if extra:
                    influence = extra["influence"].detach()
                    if influence.abs().max() > 2:
                        raise RuntimeError("RNORM influence bound")
                    rec.update(
                        scale_min=float(extra["anchor_loss_scale"].min()),
                        scale_floor_contexts=int(
                            (extra["anchor_loss_scale"] == obj.SCALE_FLOOR).sum()
                        ),
                        z_max=float(extra["standardized_loss"].detach().max()),
                        max_abs_influence=float(influence.abs().max()),
                        clipped_contexts=int(
                            (extra["standardized_loss"].detach().abs() > 2).sum()
                        ),
                    )
                trace.append(rec)

        for outer in range(5):
            fit, select, assess = d.partition(train, outer)
            fitc = {r["context_sha256"] for r in fit}
            bc = {r["context_sha256"] for r in select}
            ac = {r["context_sha256"] for r in assess}
            if (
                fitc & bc
                or fitc & ac
                or bc & ac
                or [len({r["family_id"] for r in x}) for x in (fit, select, assess)]
                != [123, 41, 41]
            ):
                raise RuntimeError("FIT/B/A leakage")
            orders = {
                r["context_sha256"]: np.argsort(
                    engine.pools[r["context_sha256"]]["fitness"]
                )
                for r in fit
            }
            fit_proofs.append(
                dict(
                    outer=outer,
                    FIT=sorted(fitc),
                    B=sorted(bc),
                    A=sorted(ac),
                    family_probabilities=d.family_probabilities(
                        sorted(fit, key=lambda r: r["context_sha256"])
                    ).tolist(),
                )
            )
            for si, seed in enumerate(d.SEEDS):
                tag = f"F{outer}-S{seed}"
                init = p.random_adapter_state(
                    net, names, d.h177.stream_seed(seed, outer, "fresh-native-init-v1")
                )
                paired = obj.paired_initial_states(init, d.ARMS)
                h0 = rt.tensor_hash(init)
                initial_hashes.append(h0)
                torch.save(init, out / (tag + "-init.pt"))
                batch_seed = d.h177.stream_seed(
                    seed, outer, "fit-123", "h156-batches-v1"
                )
                common = p.stat.batch_plan(
                    list(orders), orders, batch_seed, updates=160
                )
                balanced = d.balanced_plan(fit, orders, batch_seed, 160)
                for j, arm in enumerate(d.ARMS, 1):
                    p.restore(net, names, paired[arm])
                    assert rt.tensor_hash(p.adapter_state(net, names)) == h0
                    optimizer = torch.optim.AdamW(
                        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
                    )
                    plan = balanced if arm == d.ARMS[2] else common
                    trace = []
                    times = []
                    begin = 1
                    if execution is None:
                        if (outer, seed, arm) != (0, d.SEEDS[0], d.ARMS[0]):
                            raise RuntimeError("pilot order")
                        train_until(arm, fit, plan, 1, 80, optimizer, trace, times, tag)
                        execution = d.execution_plan(float(np.median(times[1:])))
                        save(
                            out / "pilot_budget.json",
                            dict(
                                **execution,
                                seconds_per_update=float(np.median(times[1:])),
                                policy="80-update prefix of first FIT-only control cell; no B/A outcomes; fixed25pct overhead+1200s reserve; all arms/folds/seeds common80or160",
                            ),
                        )
                        stage = d.StageOrderGuard(
                            expected_steps={
                                slot: execution["updates_per_cell"]
                                for slot in d.h177.ARMS
                            }
                        )
                        begin = 81
                        print("PILOT LOCKED", execution, flush=True)
                    steps = execution["updates_per_cell"]
                    train_until(
                        arm, fit, plan, begin, steps, optimizer, trace, times, tag
                    )
                    final = p.adapter_state(net, names)
                    hf = rt.tensor_hash(final)
                    if hf == h0:
                        raise RuntimeError("unchanged trained adapter")
                    torch.save(final, out / f"{tag}-{arm}.pt")
                    stage.mark_b_read()
                    progress["phase"] = "B"
                    values, costs = engine.evaluate(select)
                    panel = persist_scores(tag + "-" + arm, select, values, costs, "B")
                    for cid, y in values.items():
                        if np.isfinite(B[si, index[cid], j]):
                            raise RuntimeError("duplicate B context")
                        B[si, index[cid], j] = y
                    record = dict(
                        tag=tag,
                        outer=outer,
                        seed=seed,
                        arm=arm,
                        updates=steps,
                        init_hash=h0,
                        final_hash=hf,
                        fit_ids=sorted(fitc),
                        B=panel,
                        trace=trace,
                        batch_plan_sha256=hashlib.sha256(
                            json.dumps(
                                [
                                    [s, c, [x.tolist() for x in ix]]
                                    for s, c, ix in plan[:steps]
                                ],
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                    )
                    save(out / f"{tag}-{arm}.json", record)
                    records.append(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ["trace", "B", "fit_ids"]
                        }
                    )
                    print(
                        "B CELL COMPLETE",
                        tag,
                        arm,
                        "updates",
                        progress["optimizer_calls_completed"],
                        flush=True,
                    )
                    del optimizer, final, costs
        if (
            len(set(initial_hashes)) != 15
            or len(records) != 45
            or not np.isfinite(B).all()
        ):
            raise RuntimeError("incomplete B grid or duplicate init seeds")
        for outer in range(5):
            for seed in d.SEEDS:
                paired_records = [
                    r for r in records if r["outer"] == outer and r["seed"] == seed
                ]
                if (
                    len(paired_records) != 3
                    or len({r["init_hash"] for r in paired_records}) != 1
                    or paired_records[0]["batch_plan_sha256"]
                    != paired_records[1]["batch_plan_sha256"]
                ):
                    raise RuntimeError(
                        "paired initial-state/control-RNORM plan mismatch"
                    )
        decisions, native = d.select(
            train, B[:, :, 0], {arm: B[:, :, j] for j, arm in enumerate(d.ARMS, 1)}
        )
        for arm in d.ARMS:
            decisions[arm]["actual_prespecified_updates"] = execution[
                "updates_per_cell"
            ]
        lock = out / "pooled_b_selection_lock.json"
        save(
            lock,
            dict(
                wave="H177",
                study_wave="H187",
                pooled_b_families=205,
                pooled_b_contexts=1219,
                decisions=native,
                H187_m3_decisions=decisions,
                actual_candidate_updates=execution["updates_per_cell"],
                native_slot_step_metadata=40,
                H178_receipt_sha256=sha(
                    ROOT
                    / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
                ),
                locked_before_A=True,
            ),
        )
        stage.mark_b_locked(lock)
        lock_hash = sha(lock)
        save(out / "fit_role_proofs.json", fit_proofs)
        np.savez_compressed(out / "B-panels.npz", values=B, context_ids=np.array(ids))
        print("POOLED B LOCKED", flush=True)
        for outer in range(5):
            _, _, assess = d.partition(train, outer)
            for si, seed in enumerate(d.SEEDS):
                for j, arm in enumerate(d.ARMS, 1):
                    stage.require_b_locked()
                    if sha(lock) != lock_hash:
                        raise RuntimeError("B lock mutation")
                    tag = f"F{outer}-S{seed}-{arm}"
                    state = torch.load(
                        out / (tag + ".pt"), map_location="cpu", weights_only=True
                    )
                    p.restore(net, names, state)
                    progress.update(phase="A", cell=tag)
                    values, costs = engine.evaluate(assess)
                    apanel = persist_scores(tag, assess, values, costs, "A")
                    for cid, y in values.items():
                        if np.isfinite(A[si, index[cid], j]):
                            raise RuntimeError("duplicate A context")
                        A[si, index[cid], j] = y
                    stage.require_b_locked()
                    values, costs = engine.evaluate(val)
                    vpanel = persist_scores(tag, val, values, costs, "Val")
                    V[si, outer, :, j] = [values[r["context_sha256"]] for r in val]
                    save(
                        out / (tag + "-confirmation.json"),
                        dict(A=apanel, DLVal=vpanel, B_lock_sha256=lock_hash),
                    )
                    check()
                    print("A CONFIRMATION COMPLETE", tag, flush=True)
        if not np.isfinite(A).all() or not np.isfinite(V).all():
            raise RuntimeError("incomplete confirmation")
        progress["phase"] = "SUMMARY"
        save(out / "progress.json", progress)
        np.savez_compressed(
            out / "panels.npz",
            B=B,
            A=A,
            DLVal=V,
            context_ids=np.array(ids),
            Val_context_ids=np.array([r["context_sha256"] for r in val]),
        )
        summaryA = d.summarize(train, A)
        summaryB = d.summarize(train, B)
        summaryV = d.summarize(val * 5, V.reshape(3, 5 * len(val), 4))
        decision = d.verdict(decisions, summaryA)
        receipt = json.loads(
            (
                ROOT
                / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
            ).read_text()
        )
        cal = d.H177_DECISION_RULE.resolve_verdict(
            any(c["admitted"] for c in decisions.values()),
            receipt,
            expected_analysis_sha256=manifest["parent_sources"][
                "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
            ],
        )
        if any(c["admitted"] for c in decisions.values()) and not cal["lift_qualified"]:
            raise RuntimeError("H178 receipt does not qualify inherited machinery")
        p.restore(net, names, initial)
        check()
        whole_after = rt.tensor_hash(net.state_dict())
        dope_after = guard.snapshot_dope_weights(net).composite_sha256
        if (
            whole_before != whole_after
            or dope_before != dope_after
            or audit.refused
            or progress["optimizer_calls_started"]
            != progress["optimizer_calls_completed"]
            or progress["optimizer_calls_completed"]
            != execution["total_optimizer_updates"]
        ):
            raise RuntimeError("state/update/audit closure")
        verify(manifest)
        summaryV.update(
            unique_contexts=91,
            fold_models=5,
            context_model_pairs_per_seed=455,
            scope="DL-Val descriptive only; random-uniform fold model, NOT score-averaged ensemble; exact-2 counts out of455 model/context pairs",
        )
        metrics = dict(
            wave="H187",
            status="COMPLETE",
            **decision,
            source_commit=head,
            manifest_sha256=sha(HERE / "manifest.json"),
            seeds=d.SEEDS,
            fold_seed=d.FOLD_SEED,
            boot_seed=d.BOOT_SEED,
            execution=execution,
            optimizer_updates=progress["optimizer_calls_completed"],
            progress=progress,
            baseline_mean=1.1377941898164525,
            DOPE_before=dope_before,
            DOPE_after=dope_after,
            frozen_sha256=frozen,
            restored_whole_model_before=whole_before,
            restored_whole_model_after=whole_after,
            adapter_trainable_parameters=29211,
            initial_hashes=initial_hashes,
            cell_records=records,
            B_decisions=decisions,
            pooled_B=summaryB,
            pooled_A=summaryA,
            DLVal=summaryV,
            H178_gate=cal,
            B_lock_sha256=lock_hash,
            audit=audit.receipt(),
            limitations=[
                "Same205TRAIN families rotate through FIT/B/A in different folds; pooled A and pooled B are correlated, not independent new-family replication.",
                "H177 imported three-treatment branch with R882 dummyCTRL; H178 receipt calibrates four-arm mechanism, not a separate empirical H187 m3 calibration.",
                "RNORM Huber influence is bounded in normalized LOSS units, not a bound on total parameter gradient; normalizer floor1e-3.",
                "FBAL draws contexts WITH replacement to make per-draw family probability exactly equal; minibatch duplicates allowed.",
                "No live rule/model adoption; ADVANCE only nominates a separately reviewed future candidate.",
            ],
        )
        save(out / "metrics.json", metrics)
        print(
            "COMPLETE",
            decision["verdict"],
            progress["optimizer_calls_completed"],
            flush=True,
        )
    except BaseException as error:
        save(
            out / "failure.json",
            dict(status="INCOMPLETE", error=repr(error), **progress),
        )
        raise


if __name__ == "__main__":
    main()
