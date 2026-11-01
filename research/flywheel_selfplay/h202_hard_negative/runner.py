"""H202 single guarded CPU attempt: fixed 40-update arms, B lock, pooled A."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from monitor import admit_host

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path("/home/ubuntu/jope-data/h202")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def native_modules():
    old_path = sys.path[:]
    saved = {
        k: sys.modules.pop(k, None)
        for k in ["analysis", "treatment", "h182_source_treatment"]
    }
    try:
        sys.path.insert(0, str(HERE.parent / "h189_swa_ema"))
        base = load_module(
            "h202_H197_native_base", HERE.parent / "h189_swa_ema/h182_base.py"
        )
    finally:
        sys.path[:] = old_path
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return base


def evaluate(engine, rows, native=False, prove=False, progress=None):
    """Batch1 matches the live utility routine's shape; H197 Engine is unchanged."""
    values, costs = {}, {}
    for index, row in enumerate(rows):
        v, c = engine.evaluate([row], native=native, prove_identity=prove)
        values.update(v)
        costs.update(c)
        if progress and ((index + 1) % 50 == 0 or index + 1 == len(rows)):
            progress(index + 1, len(rows))
    return values, costs


def main():
    host = admit_host()
    if host != "xbabe1" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("the sole H202 science attempt runs on xbabe1 at nice19")
    global np, torch, d, objective, artifacts, stat
    import numpy as np
    import torch
    import design as d
    import objectives as objective
    import artifacts
    import analysis as stat

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--launch", required=True, type=Path)
    args = ap.parse_args()
    if args.output != DATA / "attempt-001" or args.output.exists():
        raise RuntimeError("one fresh H202 attempt-001 only")
    pins = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if (
        not launch["merged_source_verified"]
        or launch["source_commit"] != head
        or launch["manifest_sha256"] != sha(HERE / "manifest.json")
    ):
        raise RuntimeError("reviewed merged source/manifest binding mismatch")
    for group in ["inputs", "assets", "parent_sources", "source_sha256"]:
        for name, expected in pins[group].items():
            path = (
                HERE / name
                if group == "source_sha256"
                else REPO / name
                if group == "parent_sources"
                else Path(name)
            )
            if sha(path) != expected:
                raise RuntimeError("pin mismatch: " + str(path))
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)
    args.output.mkdir()
    output = args.output
    counter = dict(
        optimizer_started=0,
        optimizer_completed=0,
        backward_started=0,
        backward_completed=0,
        completed_cells=0,
        A_cells=0,
        phase="BASELINE",
    )
    d.save_json(output / "counter.json", counter)
    base = native_modules()
    rt = base.runtime()
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, val = packet["train"], packet["dev"]
    train_ids = [r["context_sha256"] for r in train]
    val_ids = [r["context_sha256"] for r in val]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
    ):
        raise RuntimeError("fixed approved roster dimensions")
    if len(set(train_ids + val_ids)) != 1310:
        raise RuntimeError("duplicate approved context IDs")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("TRAIN/Val family overlap")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["source_split"] not in ("train", "inner")
            or r.get("has_meta_test", False)
            for r in rows
        ):
            raise RuntimeError("unapproved input role")
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    groups = {
        seed: d.folds([r["family_id"] for r in train], fseed)
        for seed, fseed in zip(d.SEEDS, d.FOLD_SEEDS, strict=True)
    }
    if fold_doc != {
        str(seed): dict(fold_seed=fseed, families=groups[seed])
        for seed, fseed in zip(d.SEEDS, d.FOLD_SEEDS, strict=True)
    }:
        raise RuntimeError("fresh committed folds differ")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch)}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()

    def interrupted(sig, _):
        raise RuntimeError("owned supervisor signal " + str(sig))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        if (
            objective.adapter_mask(net) != names
            or len(names) != 47
            or sum(p.numel() for p in net.parameters() if p.requires_grad) != 29211
        ):
            raise RuntimeError("only the 47-tensor/29211-param adapter may train")
        objective.assert_grad_mask(net, names)
        rt.check_frozen(net, guard, names, frozen)
        initial = base.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        whole_before = rt.tensor_hash(net.state_dict())
        engine = base.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline_val, reference = evaluate(engine, val, native=True, prove=True)
        vector = np.asarray([baseline_val[c] for c in val_ids])
        expected = json.loads(Path(pins["reference"]).read_text())["utility"][
            "per_context"
        ]
        if (
            not np.array_equal(vector, np.asarray(expected))
            or float(vector.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native R88291 utility regression not bit-exact")
        with np.load(pins["baseline_cost_reference"], allow_pickle=False) as z:
            cost_gold = all(
                reference[c].dtype == z[c].dtype
                and reference[c].tobytes() == z[c].tobytes()
                for c in val_ids
            )
        d.save_json(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                families=19,
                batch_size=1,
                mean=float(vector.mean()),
                utility_bit_exact=True,
                native_cached_costs_bit_exact=True,
                H170_batch8_cost_bytes_equal_diagnostic=cost_gold,
                TRAIN_outcome_pools_opened=0,
                optimizer_updates=0,
            ),
            exclusive=True,
        )
        artifacts.save_state(output / "initial-adapter.npz", initial)
        d.save_json(
            output / "parameter-inventory.json",
            dict(
                names=list(names),
                count=29211,
                initial_sha256=initial_hash,
                nonadapter_sha256=frozen,
            ),
            exclusive=True,
        )
        print(
            "BASELINE PASS native91 utility exact; batch1; zero TRAIN pools/updates",
            flush=True,
        )
        catalog = {
            r["id"]: r
            for r in map(
                json.loads, Path(pins["catalog_copy"]).read_text().splitlines()
            )
        }
        if any(catalog[r["dataset_id"]]["origin"] != "real" for r in train + val):
            raise RuntimeError("real S3 join failed")
        engine.load(train)
        counter["phase"] = "R882_TRAIN_REFERENCE"
        d.save_json(output / "counter.json", counter)

        def reference_progress(done, total):
            counter["reference_contexts"] = done
            d.save_json(output / "counter.json", counter)
            print("R882 REFERENCE", done, "/", total, flush=True)

        baseline_train, train_costs = evaluate(
            engine, train, progress=reference_progress
        )
        reference.update(train_costs)
        all_ids = val_ids + train_ids
        artifacts.save_npz(
            output / "r882-reference-costs.npz",
            context_ids=np.asarray(all_ids),
            offsets=np.r_[0, np.cumsum([len(reference[c]) for c in all_ids])].astype(
                np.int64
            ),
            costs=np.concatenate([reference[c] for c in all_ids]).astype(np.float32),
        )
        del train_costs
        positions = {c: j for j, c in enumerate(train_ids)}
        base_train = np.asarray([baseline_train[c] for c in train_ids])
        B = np.full((3, len(train), 4, 5), np.nan, dtype=np.float64)
        B[:, :, 0, :] = base_train[None, :, None]
        B[:, :, :, 0] = base_train[None, :, None]
        records = []
        state_files = {}
        B_files = {}
        arm_stats = {
            a: dict(contexts=0, pairs=0, zero_pair_contexts=0, nonzero_gradient_steps=0)
            for a in d.ARMS
        }
        for si, seed in enumerate(d.SEEDS):
            for outer in range(5):
                fit, brows, arows = d.partition(train, groups[seed], outer)
                fit_ids = {r["context_sha256"] for r in fit}
                plan = d.plan(fit, seed, outer)
                plan_sha = hashlib.sha256(
                    json.dumps(plan, separators=(",", ":")).encode()
                ).hexdigest()
                for ai, arm in enumerate(d.ARMS, 1):
                    tag = f"S{seed}-F{outer}-{arm}"
                    base.restore(net, names, initial)
                    if rt.tensor_hash(base.adapter_state(net, names)) != initial_hash:
                        raise RuntimeError(
                            "cell did not restore the identical r882 adapter"
                        )
                    rt.check_frozen(net, guard, names, frozen)
                    objective.assert_grad_mask(net, names)
                    params = [p for p in net.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(
                        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
                    )
                    trace = []
                    training_cids = []
                    training_costs = []
                    before_dope = guard.snapshot_dope_weights(net).composite_sha256
                    counter.update(phase="FIT_AND_B", cell=tag, step=0)
                    d.save_json(output / "counter.json", counter)
                    with torch.autograd.set_multithreading_enabled(False):
                        for item in plan:
                            step = item["step"]
                            optimizer.zero_grad(set_to_none=True)
                            micro = []
                            if not set(item["context_ids"]) <= fit_ids:
                                raise RuntimeError("non-FIT training context")
                            for slot, cid in enumerate(item["context_ids"]):
                                n = len(engine.pools[cid]["fitness"])
                                kwargs, tokens, masks, fitness, gate = engine.batch(
                                    [cid], [np.arange(n)]
                                )
                                anchor = tuple(t[None] for t in engine.cached[cid])
                                prediction = engine.forward(
                                    kwargs, tokens, masks, anchor
                                )[0]
                                costs = prediction.detach().numpy().copy()
                                if not np.isfinite(costs).all():
                                    raise RuntimeError("nonfinite training costs")
                                u = np.where(
                                    engine.pools[cid]["gate"],
                                    -engine.pools[cid]["fitness"],
                                    -2.0,
                                )
                                shuffle_seed = d.stream(
                                    seed,
                                    outer,
                                    cid,
                                    "H202-within-context-label-shuffle-v1",
                                )
                                labels = (
                                    objective.shuffled_utilities(u, shuffle_seed)
                                    if arm == "HNEG_SHUF"
                                    else u
                                )
                                pair_loss, detail = objective.hard_negative(
                                    prediction, labels
                                )
                                loss = (
                                    rt.loss(prediction[None], fitness, gate)
                                    if arm == "CTRL40"
                                    else pair_loss
                                )
                                if not torch.isfinite(loss):
                                    raise RuntimeError("nonfinite loss")
                                training_cids.append(cid)
                                training_costs.append(costs)
                                micro.append(
                                    dict(
                                        cid=cid,
                                        loss=float(loss.detach()),
                                        shuffle_seed=shuffle_seed
                                        if arm == "HNEG_SHUF"
                                        else None,
                                        **detail,
                                    )
                                )
                                arm_stats[arm]["contexts"] += 1
                                arm_stats[arm]["pairs"] += detail["pair_count"]
                                arm_stats[arm]["zero_pair_contexts"] += (
                                    detail["pair_count"] == 0
                                )
                                counter["backward_started"] += 1
                                d.save_json(output / "counter.json", counter)
                                (loss / 4).backward()
                                counter["backward_completed"] += 1
                                objective.assert_grad_mask(net, names)
                                del (
                                    prediction,
                                    pair_loss,
                                    loss,
                                    kwargs,
                                    tokens,
                                    masks,
                                    fitness,
                                    gate,
                                    anchor,
                                )
                            norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    params, 1.0, error_if_nonfinite=True
                                )
                            )
                            arm_stats[arm]["nonzero_gradient_steps"] += norm > 0
                            counter["optimizer_started"] += 1
                            counter["step"] = step
                            d.save_json(output / "counter.json", counter)
                            optimizer.step()
                            counter["optimizer_completed"] += 1
                            d.save_json(output / "counter.json", counter)
                            engine.check()
                            objective.assert_grad_mask(net, names)
                            trace.append(
                                dict(
                                    step=step,
                                    stream=item["stream"],
                                    context_ids=item["context_ids"],
                                    family_ids=item["family_ids"],
                                    raw_gradient_norm=norm,
                                    micro=micro,
                                )
                            )
                            if step in d.STEPS[1:]:
                                state = base.adapter_state(net, names)
                                path = output / (tag + f"-T{step}-state.npz")
                                artifacts.save_state(path, state)
                                state_files[path.name] = sha(path)
                                values, cost = evaluate(engine, brows)
                                for r in brows:
                                    B[
                                        si,
                                        positions[r["context_sha256"]],
                                        ai,
                                        d.STEPS.index(step),
                                    ] = values[r["context_sha256"]]
                                cids = [r["context_sha256"] for r in brows]
                                bp = output / (tag + f"-T{step}-B.npz")
                                artifacts.save_scores(
                                    bp,
                                    cids,
                                    [cost[c] for c in cids],
                                    reference,
                                    [values[c] for c in cids],
                                    [int(cost[c].argmin()) for c in cids],
                                )
                                B_files[bp.name] = sha(bp)
                                print(
                                    "B SNAPSHOT",
                                    tag,
                                    step,
                                    "updates",
                                    counter["optimizer_completed"],
                                    flush=True,
                                )
                    artifacts.save_scores(
                        output / (tag + "-training-costs.npz"),
                        training_cids,
                        training_costs,
                        reference,
                    )
                    d.save_json(output / (tag + "-trace.json"), trace, exclusive=True)
                    final = base.adapter_state(net, names)
                    after_dope = guard.snapshot_dope_weights(net).composite_sha256
                    if before_dope != after_dope or before_dope != rt.DOPE:
                        raise RuntimeError("cell DOPE changed")
                    record = dict(
                        cell=tag,
                        seed=seed,
                        outer=outer,
                        arm=arm,
                        updates=40,
                        fit_ids=sorted(fit_ids),
                        B_ids=[r["context_sha256"] for r in brows],
                        A_ids=[r["context_sha256"] for r in arows],
                        plan_sha256=plan_sha,
                        initial_adapter_sha256=initial_hash,
                        final_adapter_sha256=rt.tensor_hash(final),
                        DOPE_before=before_dope,
                        DOPE_after=after_dope,
                    )
                    d.save_json(output / (tag + "-cell.json"), record, exclusive=True)
                    records.append(record)
                    counter["completed_cells"] += 1
                    d.save_json(output / "counter.json", counter)
                    del optimizer, params, training_costs, training_cids, trace
        if (
            counter["optimizer_started"] != 1800
            or counter["optimizer_completed"] != 1800
            or counter["backward_started"] != 7200
            or counter["backward_completed"] != 7200
            or counter["completed_cells"] != 45
        ):
            raise RuntimeError("optimizer/backward/cell closure mismatch")
        selection = d.choose_steps(train, B)
        artifacts.save_npz(
            output / "B-panels.npz", panel=B, context_ids=np.asarray(train_ids)
        )
        lock = dict(
            wave="H202",
            source_commit=head,
            fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
            locked_before_A=True,
            **selection,
            B_files=B_files,
            state_files=state_files,
            B_panel_sha256=sha(output / "B-panels.npz"),
        )
        lock_path = output / "B-selection-lock.json"
        lock_sha = d.write_lock(lock_path, lock)
        counter["phase"] = "LOCKED_A_CONFIRMATION"
        d.save_json(output / "counter.json", counter)
        print("B LOCKED", selection["selected_steps"], flush=True)
        A = np.full((3, len(train), 4), np.nan)
        A[:, :, 0] = base_train
        V = np.full((3, 5, 91, 4), np.nan)
        V[:, :, :, 0] = vector
        for si, seed in enumerate(d.SEEDS):
            for outer in range(5):
                _, _, arows = d.partition(train, groups[seed], outer)
                for ai, arm in enumerate(d.ARMS, 1):
                    selected = d.authorize_A(lock_path, lock_sha)[arm]
                    tag = f"S{seed}-F{outer}-{arm}"
                    before_dope = guard.snapshot_dope_weights(net).composite_sha256
                    if selected:
                        state_path = output / (tag + f"-T{selected}-state.npz")
                        if sha(state_path) != lock["state_files"][state_path.name]:
                            raise RuntimeError("selected adapter changed after B lock")
                    state = (
                        initial if selected == 0 else artifacts.load_state(state_path)
                    )
                    base.restore(net, names, state)
                    engine.check()
                    for phase, rows in [("A", arows), ("Val", val)]:
                        if selected == 0:
                            values = {
                                r["context_sha256"]: (
                                    baseline_train if phase == "A" else baseline_val
                                )[r["context_sha256"]]
                                for r in rows
                            }
                            costs = {
                                r["context_sha256"]: reference[r["context_sha256"]]
                                for r in rows
                            }
                        else:
                            values, costs = evaluate(engine, rows)
                        cids = [r["context_sha256"] for r in rows]
                        artifacts.save_scores(
                            output / (tag + "-" + phase + ".npz"),
                            cids,
                            [costs[c] for c in cids],
                            reference,
                            [values[c] for c in cids],
                            [int(costs[c].argmin()) for c in cids],
                        )
                        if phase == "A":
                            for c in cids:
                                A[si, positions[c], ai] = values[c]
                        else:
                            V[si, outer, :, ai] = [values[c] for c in cids]
                    after_dope = guard.snapshot_dope_weights(net).composite_sha256
                    if before_dope != after_dope or after_dope != rt.DOPE:
                        raise RuntimeError("confirmation DOPE changed")
                    d.save_json(
                        output / (tag + "-confirmation.json"),
                        dict(
                            selected_step=selected,
                            lock_sha256=lock_sha,
                            DOPE_before=before_dope,
                            DOPE_after=after_dope,
                        ),
                        exclusive=True,
                    )
                    counter["A_cells"] += 1
                    d.save_json(output / "counter.json", counter)
                    print("A CONFIRMED", tag, selected, flush=True)
        if (
            not np.isfinite(A).all()
            or not np.isfinite(V).all()
            or counter["A_cells"] != 45
        ):
            raise RuntimeError("incomplete A/Val")
        artifacts.save_npz(
            output / "A-Val-panels.npz",
            A=A,
            Val=V,
            context_ids=np.asarray(train_ids),
            val_context_ids=np.asarray(val_ids),
        )
        counter["phase"] = "SUMMARY"
        d.save_json(output / "counter.json", counter)
        primary = stat.summarize(train, A)
        descriptive = stat.summarize(val * 5, V.reshape(3, 455, 4))
        descriptive["verdict"] = "DESCRIPTIVE_ONLY"
        descriptive["unique_contexts"] = 91
        descriptive["model_context_pairs_per_seed"] = 455
        base.restore(net, names, initial)
        engine.check()
        objective.assert_grad_mask(net, names)
        if rt.tensor_hash(net.state_dict()) != whole_before:
            raise RuntimeError("restored whole-state mismatch")
        receipt = audit.receipt()
        d.save_json(output / "audit.json", receipt, exclusive=True)
        if receipt["refused"] or receipt["protected_reads"]:
            raise RuntimeError("protected input refusal")
        metrics = dict(
            status="COMPLETE",
            wave="H202",
            verdict=primary["verdict"],
            source_commit=head,
            host=host,
            optimizer_updates=counter["optimizer_completed"],
            backward_calls=counter["backward_completed"],
            counters=counter,
            adapter_parameters=29211,
            adapter_parameter_names=list(names),
            baseline=json.loads((output / "baseline.json").read_text()),
            DOPE_before=rt.DOPE,
            DOPE_after=guard.snapshot_dope_weights(net).composite_sha256,
            nonadapter_sha256=frozen,
            restored_whole_model_sha256=whole_before,
            initial_adapter_sha256=initial_hash,
            selection=selection,
            B_lock_sha256=lock_sha,
            A=primary,
            DLVal=descriptive,
            training_telemetry=arm_stats,
            audit=dict(
                opened_count=len(receipt["opened"]),
                refused=receipt["refused"],
                protected_reads=0,
            ),
            seeds=list(d.SEEDS),
            fold_seeds=list(d.FOLD_SEEDS),
            bootstrap_seed=d.BOOT_SEED,
            native_protocol="full-pool first argmin, CPU batch1, unchanged H197 Engine/forward; frozen anchors cached",
            cost_encoding="lossless XOR-f32 against immutable r882 costs",
            serving_promotion=False,
            limitations="A/B reuse the same205families across cyclic roles; DLVal repeatedly selected and descriptive; current B-picked common step may be0 despite all1800training calls",
        )
        d.save_json(output / "metrics.json", metrics, exclusive=True)
        print(
            "H202 COMPLETE", metrics["verdict"], "updates1800/backward7200", flush=True
        )
    except BaseException as exc:
        d.save_json(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(exc),
                counter=counter,
                audit=audit.receipt(),
                elapsed=time.monotonic() - started,
            ),
            exclusive=True,
        )
        raise


if __name__ == "__main__":
    main()
