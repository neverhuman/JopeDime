"""H263 single offline run: paired round streams, FIT-only rules, final-only DL-Val."""

import argparse
import ctypes
import gc
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
import engine as eng
from treatment import (
    ControlAdamW,
    adapter_drift,
    gradient_norm,
    l2sp_penalty,
    retract,
    head4_loss,
)

HERE = Path(__file__).resolve().parent
PREFIX = "research/flywheel_selfplay/h263_head4_rerank/"


def durable_npz(path, arrays):
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    return eng.sha(path)


def sync_directory(out):
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def release():
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def assert_final_lock(out):
    lock = out / "final-lock.json"
    receipt = json.loads(lock.read_text())
    if receipt["completed_cells"] != 15 or receipt["optimizer_updates"] != 2400:
        raise RuntimeError("final lock incomplete")
    for name, digest in receipt["files"].items():
        if eng.sha(out / name) != digest:
            raise RuntimeError("final checkpoint mutated")
    return eng.sha(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or sorted(os.sched_getaffinity(0)) != [48, 50, 51, 52]
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("host/resource contract")
    out = args.output.resolve()
    out.mkdir()
    eng.PAUSE = out.with_suffix(".pause.json")
    pins = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    head = os.environ["H263_SOURCE"]
    for key, value in [
        ("source_commit", head),
        ("reviewed_source", head),
        ("review_status", "PASS"),
        ("publication_branch", "v1/next"),
        ("published_source_verified", True),
        ("manifest_sha256", eng.sha(HERE / "manifest.json")),
        ("prereg_sha256", eng.sha(HERE / "PREREG.md")),
        ("fold_manifest_sha256", eng.sha(HERE / "fold_manifest.json")),
    ]:
        if launch.get(key) != value:
            raise RuntimeError("launch binding: " + key)
    if launch.get("resume_number", 0) != 0:
        raise RuntimeError("single run only")
    for f in HERE.iterdir():
        if f.is_file() and f.read_bytes() != subprocess.check_output(
            ["git", "show", head + ":" + PREFIX + f.name], cwd=eng.REPO
        ):
            raise RuntimeError("published source mismatch: " + f.name)
    eng.verify(pins)
    rt = eng.runtime(pins)
    train, val = eng.cohorts(rt)
    fold = json.loads((HERE / "fold_manifest.json").read_text())
    if fold["families_by_seed"] != {
        str(seed): stat.folds(train, seed) for seed in stat.SEEDS
    }:
        raise RuntimeError("fold identity")
    provenance = json.loads((HERE / "provenance.json").read_text())
    if (
        provenance["status"] != "PASS"
        or not provenance["zero_synthetic"]
        or sorted({r["dataset_id"] for r in train + val}) != provenance["dataset_ids"]
    ):
        raise RuntimeError("real S3 closure")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    allowed |= {str(f.resolve()) for f in HERE.iterdir() if f.is_file()}
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    updates = 0
    cells = []
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]

        def snapshot():
            return {n: net.state_dict()[n].detach().clone() for n in names}

        def restore(state):
            with torch.no_grad():
                for n, param in net.named_parameters():
                    if n in state:
                        param.copy_(state[n])
            engine.check()

        initial = snapshot()
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params) != 29211
            or len(names) != 47
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("adapter inventory/mode")
        engine = eng.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, base_costs = engine.evaluate(val, native=True, prove_identity=True)
        ref = json.loads(Path(pins["reference"]).read_text())
        contract = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            ref["evaluation_contract"]["contexts"] != contract
            or hashlib.sha256(
                json.dumps(ref["evaluation_contract"], sort_keys=True).encode()
            ).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation identity")
        if (
            not np.array_equal(baseline["values"], ref["utility"]["per_context"])
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("Native91 exact mismatch")
        durable_npz(out / "baseline_costs.npz", base_costs)

        def top4(rows, costs):
            return [
                stat.top4_pick(
                    costs[r["context_sha256"]],
                    engine.pools[r["context_sha256"]]["fitness"],
                    engine.pools[r["context_sha256"]]["gate"],
                )
                for r in rows
            ]

        base_top = top4(val, base_costs)
        if abs(float(np.mean([x["utility"] for x in base_top])) - 1.240360) > 5e-7:
            raise RuntimeError("H258 TOP4_PICK baseline mismatch")
        eng.save(
            out / "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=1.1377941898164525,
                top4=base_top,
                top4_context_mean=float(np.mean([x["utility"] for x in base_top])),
            ),
        )
        print("BASELINE PASS exact91 and TOP4_PICK", flush=True)
        base_heads = {
            cid: np.argsort(cost, kind="stable")[:4] for cid, cost in base_costs.items()
        }
        engine.load(train)
        trainbase, traincost = engine.evaluate(train, native=True, prove_identity=True)
        heads = {
            cid: np.argsort(cost, kind="stable")[:4] for cid, cost in traincost.items()
        }
        durable_npz(out / "train-baseline-costs.npz", traincost)
        eng.save(out / "train-baseline.json", trainbase)
        eng.save(
            out / "frozen-heads.json",
            dict(
                train={k: v.tolist() for k, v in heads.items()},
                val={k: v.tolist() for k, v in base_heads.items()},
                before_optimizer_updates=0,
            ),
        )
        del traincost
        initial_params = [p.detach().clone() for p in params]

        def update(arm, optimizer, stream, cids, indexes, fitids):
            eng.cooperate()
            if not set(cids) <= fitids:
                raise RuntimeError("FIT-only update")
            torch.manual_seed(stream)
            optimizer.zero_grad(set_to_none=True)
            kwargs, tok, mask, fitness, gate = engine.batch(cids, indexes)
            pred = engine.forward(kwargs, tok, mask, engine.anchored(kwargs))
            control = rt.loss(pred, fitness, gate)
            control.backward()
            head_total = 0.0
            if arm in ("HEAD4", "HEAD4_NOTR"):
                # Exact native full-pool candidate closure and first-four support;
                # one context graph at a time bounds memory. No head labels as conditioning.
                for cid in cids:
                    eng.cooperate()
                    kw, tk, mk, ff, gg = engine.batch(
                        [cid], [np.arange(len(engine.pools[cid]["fitness"]))]
                    )
                    anchor = tuple(t.unsqueeze(0) for t in engine.cached[cid])
                    pr = engine.forward(kw, tk, mk, anchor)[0]
                    ix = heads[cid]
                    loss = head4_loss(
                        pr[ix],
                        torch.as_tensor(
                            engine.pools[cid]["fitness"][ix], dtype=torch.float64
                        ),
                        gg[0, ix],
                    )
                    (0.5 * loss / len(cids)).backward()
                    head_total += float(loss.detach()) / len(cids)
                    del kw, tk, mk, ff, gg, pr, loss
            anchor_value = 0.0
            if arm in ("L2SP", "HEAD4"):
                penalty = l2sp_penalty(params, initial_params)
                penalty.backward()
                anchor_value = float(penalty.detach())
            norm = float(
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            )
            clipped = gradient_norm(params)
            optimizer.step()
            if arm in ("L2SP", "HEAD4"):
                retract(params, initial_params)
            engine.check()
            record = dict(
                control_loss=float(control.detach()),
                head4_loss=head_total,
                anchor_penalty=anchor_value,
                loss=float(control.detach()) + 0.5 * head_total + anchor_value,
                raw_grad_l2=norm,
                clipped_grad_l2=clipped,
                stream=stream,
                cids=cids,
            )
            if not np.isfinite(
                list(record[k] for k in ["loss", "raw_grad_l2", "clipped_grad_l2"])
            ).all():
                raise RuntimeError("nonfinite update")
            return record

        fit0, _, _ = stat.partition(train, 0, stat.SEEDS[0])
        fitids0 = {r["context_sha256"] for r in fit0}
        orders0 = {c: np.argsort(engine.pools[c]["fitness"]) for c in fitids0}
        stream, cids, indexes = stat.batch_plan(fitids0, orders0, stat.SEEDS[0])[0]
        smoke = []
        for arm in stat.ARMS:
            restore(initial)
            optimizer = ControlAdamW(params)
            trace = update(arm, optimizer, stream, cids, indexes, fitids0)
            changed = rt.tensor_hash(snapshot()) != initial_hash
            if not changed:
                raise RuntimeError("smoke unchanged")
            smoke.append(dict(arm=arm, parameters_changed=True, **trace))
        restore(initial)
        if rt.tensor_hash(snapshot()) != initial_hash:
            raise RuntimeError("smoke reset")
        eng.save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=4,
                arms=smoke,
                restored=True,
                treatment_assessment_reads=0,
            ),
        )
        print("SMOKE PASS four updates; reset", flush=True)
        del optimizer
        lock_files = {}
        for outer in range(5):
            for seed in stat.SEEDS:
                fit, select, assess = stat.partition(train, outer, seed)
                fitids = {r["context_sha256"] for r in fit}
                forbidden = {r["context_sha256"] for r in select + assess + val}
                if fitids & forbidden:
                    raise RuntimeError("family leakage")
                orders = {c: np.argsort(engine.pools[c]["fitness"]) for c in fitids}
                plan = stat.batch_plan(fitids, orders, seed)
                cellname = f"A{outer}-{seed}"
                restore(initial)
                selbase, selcost = engine.evaluate(select)
                selbase["costs_file"] = cellname + "-SELECT-step0-costs.npz"
                selbase["costs_sha256"] = durable_npz(
                    out / selbase["costs_file"], selcost
                )
                del selcost
                decisions = {}
                traces = {}
                panels = {}
                drifts = {}
                snap_hashes = {}
                for arm in stat.ARMS:
                    restore(initial)
                    optimizer = ControlAdamW(params)
                    if sorted(
                        id(p) for g in optimizer.param_groups for p in g["params"]
                    ) != sorted(id(p) for p in params):
                        raise RuntimeError("optimizer coverage")
                    traces[arm] = []
                    panels[arm] = [selbase]
                    snap_hashes[arm] = {"0": initial_hash}
                    for step, (stream, cids, indexes) in enumerate(plan, 1):
                        trace = update(arm, optimizer, stream, cids, indexes, fitids)
                        updates += 1
                        traces[arm].append(dict(step=step, **trace))
                        if step in stat.STEPS:
                            state = snapshot()
                            name = f"{cellname}-{arm}-step{step}.npz"
                            lock_files[name] = durable_npz(
                                out / name, {n: t.numpy() for n, t in state.items()}
                            )
                            snap_hashes[arm][str(step)] = rt.tensor_hash(state)
                            panel, costs = engine.evaluate(select)
                            panel["costs_file"] = (
                                f"{cellname}-{arm}-SELECT-step{step}-costs.npz"
                            )
                            panel["costs_sha256"] = durable_npz(
                                out / panel["costs_file"], costs
                            )
                            panels[arm].append(panel)
                            del costs, state
                    drifts[arm] = adapter_drift(initial, snapshot())
                    if snap_hashes[arm]["40"] == initial_hash:
                        raise RuntimeError("no optimizer change")
                    decisions[arm] = stat.choose(
                        select,
                        np.array([x["values"] for x in panels[arm]]).T,
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, arm),
                    )
                    del optimizer
                cell = dict(
                    outer_fold=outer,
                    seed=seed,
                    decisions=decisions,
                    selector_panels=panels,
                    traces=traces,
                    step40_drift=drifts,
                    adapter_sha256=snap_hashes,
                    fit_ids=sorted(fitids),
                    select_ids=[r["context_sha256"] for r in select],
                    assess_ids=[r["context_sha256"] for r in assess],
                    batch_plan_sha256=stat.plan_digest(plan),
                    completed_epoch=time.time(),
                )
                name = cellname + "-selection.json"
                eng.save(out / name, cell)
                lock_files[name] = eng.sha(out / name)
                sync_directory(out)
                cells.append(cell)
                eng.save(
                    out / "progress.json",
                    dict(
                        phase="TRAIN",
                        complete_cells=len(cells),
                        optimizer_updates=updates,
                        last_cell=cellname,
                    ),
                )
                print("TRAIN CELL COMPLETE", cellname, "updates", updates, flush=True)
                restore(initial)
                del plan, panels, traces
                release()
        if updates != 2400 or len(cells) != 15:
            raise RuntimeError("incomplete global selection")
        eng.save(
            out / "final-lock.json",
            dict(
                completed_cells=15,
                optimizer_updates=updates,
                files=lock_files,
                locked_epoch=time.time(),
                trained_DLVal_reads=0,
            ),
        )
        sync_directory(out)
        lock_sha = assert_final_lock(out)
        eng.save(
            out / "assessment-start.json",
            dict(epoch=time.time(), global_lock_sha256=lock_sha),
        )
        print("GLOBAL SELECTION LOCK PASS; assessment begins", flush=True)
        panel = np.empty((3, 5, 91, 5))
        last = np.empty_like(panel)
        top = np.empty_like(panel)
        headacc = np.empty_like(panel)
        membership = np.empty_like(panel)
        primary = np.empty((3, 1219, 5))
        last_oof = np.empty_like(primary)
        pos = {r["context_sha256"]: i for i, r in enumerate(train)}
        base_train = {
            c: v
            for c, v in zip(trainbase["context_ids"], trainbase["values"], strict=True)
        }

        def diagnostics(costs):
            records = []
            for row in val:
                cid = row["context_sha256"]
                cost = costs[cid]
                h = base_heads[cid]
                pool = engine.pools[cid]
                true = np.where(pool["gate"][h], pool["fitness"][h], 2.0)
                # For head accuracy, invalid ranks below every valid candidate.
                valid = pool["gate"][h]
                best = np.min(true[valid]) if valid.any() else 2.0
                chosen = int(np.argmin(cost[h]))
                correct = bool(
                    (valid[chosen] and true[chosen] <= best + 1e-4)
                    if valid.any()
                    else True
                )
                records.append(
                    dict(
                        head_choice=int(h[chosen]),
                        correct=correct,
                        all_invalid_head=not bool(valid.any()),
                        full_pick_in_frozen_head=bool(np.argmin(cost) in h),
                        top4=stat.top4_pick(cost, pool["fitness"], pool["gate"]),
                    )
                )
            return records

        basediag = diagnostics(base_costs)
        for cell in cells:
            outer, seed = cell["outer_fold"], cell["seed"]
            si = stat.SEEDS.index(seed)
            cellname = f"A{outer}-{seed}"
            _, _, assess = stat.partition(train, outer, seed)
            indices = [pos[r["context_sha256"]] for r in assess]
            if eng.sha(out / "final-lock.json") != lock_sha:
                raise RuntimeError("lock mutation")
            panel[si, outer, :, 0] = last[si, outer, :, 0] = baseline["values"]
            primary[si, indices, 0] = last_oof[si, indices, 0] = [
                base_train[r["context_sha256"]] for r in assess
            ]
            top[si, outer, :, 0] = [x["top4"]["utility"] for x in basediag]
            headacc[si, outer, :, 0] = [x["correct"] for x in basediag]
            membership[si, outer, :, 0] = [
                x["full_pick_in_frozen_head"] for x in basediag
            ]
            assessment = {}
            for ai, arm in enumerate(stat.ARMS, 1):
                selected = cell["decisions"][arm]["selected_step"]
                assessment[arm] = {}
                for step in sorted({selected, 40}):
                    if step == 0:
                        v = baseline
                        a = dict(
                            values=[base_train[r["context_sha256"]] for r in assess]
                        )
                        diag = basediag
                        drift = 0.0
                        item = dict(
                            val=v,
                            outer=a,
                            diagnostics=diag,
                            drift_l2=0.0,
                            baseline_identity=True,
                        )
                    else:
                        path = out / f"{cellname}-{arm}-step{step}.npz"
                        if eng.sha(path) != lock_files[path.name]:
                            raise RuntimeError("snapshot mutation")
                        with np.load(path, allow_pickle=False) as z:
                            state = {n: torch.from_numpy(z[n].copy()) for n in names}
                        restore(state)
                        drift = adapter_drift(initial, state)
                        del state
                        a, ac = engine.evaluate(assess)
                        v, vc = engine.evaluate(val)
                        diag = diagnostics(vc)
                        name = f"{cellname}-{arm}-ASSESS-step{step}-costs.npz"
                        digest = durable_npz(
                            out / name,
                            {
                                **{"outer_" + k: w for k, w in ac.items()},
                                **{"val_" + k: w for k, w in vc.items()},
                            },
                        )
                        item = dict(
                            val=v,
                            outer=a,
                            diagnostics=diag,
                            drift_l2=drift,
                            costs_file=name,
                            costs_sha256=digest,
                        )
                        del ac, vc
                    assessment[arm][str(step)] = item
                    if step == selected:
                        panel[si, outer, :, ai] = v["values"]
                        primary[si, indices, ai] = a["values"]
                        top[si, outer, :, ai] = [x["top4"]["utility"] for x in diag]
                        headacc[si, outer, :, ai] = [x["correct"] for x in diag]
                        membership[si, outer, :, ai] = [
                            x["full_pick_in_frozen_head"] for x in diag
                        ]
                    if step == 40:
                        last[si, outer, :, ai] = v["values"]
                        last_oof[si, indices, ai] = a["values"]
                restore(initial)
                release()
            eng.save(
                out / (cellname + "-assessment.json"),
                dict(
                    assessment=assessment,
                    global_lock_sha256=lock_sha,
                    completed_epoch=time.time(),
                ),
            )
            eng.save(
                out / "progress.json",
                dict(
                    phase="ASSESS",
                    complete_cells=15,
                    optimizer_updates=updates,
                    last_cell=cellname,
                ),
            )
            print("ASSESS CELL COMPLETE", cellname, flush=True)
        durable_npz(
            out / "panels.npz",
            dict(
                selected=panel,
                step40=last,
                oof=primary,
                step40_oof=last_oof,
                top4=top,
                head_accuracy=headacc,
                full_pick_in_head=membership,
            ),
        )
        final = stat.summarize(val, panel.mean(axis=1))
        metrics = dict(
            wave="H263",
            status="COMPLETE",
            source_commit=head,
            optimizer_updates=updates,
            smoke_optimizer_updates=4,
            complete_cells=15,
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            baseline_mean=1.1377941898164525,
            primary_dlval=final,
            diagnostic_step40_val=stat.summarize(val, last.mean(axis=1)),
            secondary_oof=stat.summarize(train, primary),
            diagnostic_step40_oof=stat.summarize(train, last_oof),
            top4_descriptive=stat.summarize(val, top.mean(axis=1)),
            all_invalid_head_count=sum(x["all_invalid_head"] for x in basediag),
            head_accuracy_context_by_seed=headacc.mean(axis=(1, 2)).tolist(),
            head_accuracy_family_by_seed=np.stack(
                [
                    stat.family_matrix(v, [r["family_id"] for r in val])[1].mean(axis=0)
                    for v in headacc.mean(axis=1)
                ]
            ).tolist(),
            full_pick_in_head_by_seed=membership.mean(axis=(1, 2)).tolist(),
            selections=[c["decisions"] for c in cells],
            step40_drift={a: [c["step40_drift"][a] for c in cells] for a in stat.ARMS},
            invalid_minus2_counts_by_seed_and_fold=(panel == -2.0).sum(axis=2).tolist(),
            cliff_context=dict(
                index=27,
                context_sha256=val[27]["context_sha256"],
                selected_by_seed=panel.mean(axis=1)[:, 27, :].tolist(),
                step40_by_seed=last.mean(axis=1)[:, 27, :].tolist(),
            ),
            dope_before=rt.DOPE,
            dope_after=guard.snapshot_dope_weights(net).composite_sha256,
            initial_adapter_sha256=initial_hash,
            global_lock_sha256=lock_sha,
            serving_promotion=False,
            scope="Five independently selected cell utilities averaged per seed; not a single deployable checkpoint or prediction ensemble. Native fixed4 support conditioning retained; no extra reveal/reselection for primary.",
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        if metrics["dope_after"] != rt.DOPE:
            raise RuntimeError("DOPE drift")
        metrics["primary_verdict"] = (
            "ADVANCE"
            if final["contrasts"]["HEAD4_minus_r882"]["advance"]
            else "NO_ADVANCE"
        )
        metrics["secondary_verdict"] = (
            "ADVANCE"
            if final["contrasts"]["HEAD4_NOTR_minus_r882"]["advance"]
            else "NO_ADVANCE"
        )
        metrics["verdict"] = (
            "ADVANCE"
            if "ADVANCE" in (metrics["primary_verdict"], metrics["secondary_verdict"])
            else "NO_ADVANCE"
        )
        eng.save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        eng.save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
