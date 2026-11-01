"""One H194 CPU attempt: FIT-only margin, fixed40 continuation, pooled B then A."""

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
import torch
import design as d
import objective as obj
from parents import runner_parent

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P = runner_parent()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            target = REPO / path if kind == "parent_sources" else Path(path)
            if sha(target) != digest:
                raise RuntimeError("pin mismatch: " + str(target))
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def plan_hash(plan):
    data = [[s, ids, [x.tolist() for x in ix]] for s, ids, ix in plan]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def persist_scores(output, tag, rows, raw, engine):
    ids = [r["context_sha256"] for r in rows]
    arrays = [np.asarray(raw[c], dtype=np.float32) for c in ids]
    counts = np.array([len(x) for x in arrays], dtype=np.int32)
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
        raise RuntimeError("invalid score vector")
    choices = np.array([int(np.argmin(x)) for x in arrays], dtype=np.int32)
    utilities = np.array(
        [
            (-engine.pools[c]["fitness"][j] if engine.pools[c]["gate"][j] else -2.0)
            for c, j in zip(ids, choices, strict=True)
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        output / (tag + ".npz"),
        context_ids=np.array(ids),
        family_ids=np.array([r["family_id"] for r in rows]),
        candidate_counts=counts,
        offsets=np.r_[0, np.cumsum(counts, dtype=np.int64)],
        scores=np.concatenate(arrays),
        choices=choices,
        utilities=utilities,
    )
    return dict(zip(ids, utilities.tolist(), strict=True)), dict(
        zip(ids, choices.tolist(), strict=True)
    )


def train_cell(
    engine, initial, names, fit, excluded, plan, arm, margin, seed, outer, out, counter
):
    P.restore(engine.net, names, initial)
    params = [p for p in engine.net.parameters() if p.requires_grad]
    if len(params) != 47 or sum(p.numel() for p in params) != 29211:
        raise RuntimeError("adapter inventory")
    optimizer = torch.optim.AdamW(
        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    P.stat.validate_fit_plan(plan, [r["context_sha256"] for r in fit], excluded, ())
    traces = []
    for step, (stream, ids, indexes) in enumerate(plan, 1):
        d.require_fit(ids, fit)
        kwargs, tokens, masks, fitness, gate = engine.batch(ids, indexes)
        anchor = engine.anchored(kwargs)
        torch.manual_seed(stream)
        optimizer.zero_grad(set_to_none=True)
        pred = engine.forward(kwargs, tokens, masks, anchor)
        pseudo_seed = d.h177.stream_seed(seed, outer, step, "pseudo-invalid-v1")
        loss, info = obj.loss(
            pred,
            fitness,
            gate,
            engine.rt.loss,
            arm,
            margin,
            np.random.default_rng(pseudo_seed),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite objective")
        counter["backward_started"] += 1
        save(out / "update_counter.json", counter)
        loss.backward()
        counter["backward_completed"] += 1
        norm = float(
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        )
        counter["step_started"] += 1
        save(out / "update_counter.json", counter)
        optimizer.step()
        counter["step_completed"] += 1
        save(out / "update_counter.json", counter)
        engine.check()
        traces.append(
            dict(
                step=step,
                stream=stream,
                pseudo_seed=pseudo_seed,
                fit_context_ids=ids,
                candidate_indices=[x.tolist() for x in indexes],
                loss=float(loss.detach()),
                raw_grad_norm=norm,
                **info,
            )
        )
    state = P.adapter_state(engine.net, names)
    if engine.rt.tensor_hash(state) == engine.rt.tensor_hash(initial):
        raise RuntimeError("unchanged trajectory")
    return state, traces


def complete_panel(rows, bank):
    ids = [r["context_sha256"] for r in rows]
    if any(set(bank[s][a]) != set(ids) for s in d.SEEDS for a in d.ALL_ARMS):
        raise RuntimeError("incomplete or duplicate pooled panel")
    return np.array(
        [[[bank[s][a][c] for a in d.ALL_ARMS] for c in ids] for s in d.SEEDS]
    )


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
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    if pins["max_total_updates"] != 1800 or pins["arms"] != list(d.ALL_ARMS):
        raise RuntimeError("fixed H194 design identity")
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    P.validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = P.runtime()
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE pin mismatch")
    train, val = P.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
    expected_folds = {
        str(s): dict(fold_seed=f, families=d.folds(train, s))
        for s, f in zip(d.SEEDS, d.FOLD_SEEDS, strict=True)
    }
    if (
        expected_folds
        != json.loads((HERE / "fold_manifest.json").read_text())["replicates"]
    ):
        raise RuntimeError("independent fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    counter = dict(
        backward_started=0, backward_completed=0, step_started=0, step_completed=0
    )

    def interrupted(signum, _frame):
        raise RuntimeError("supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        if (
            len(names) != 47
            or len(params) != 47
            or sum(p.numel() for p in params) != 29211
        ):
            raise RuntimeError("47 adapter tensors/29211 parameters only")
        initial = P.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        whole_hash = rt.tensor_hash(net.state_dict())
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("dropout/mode mismatch")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    scored_contexts=0,
                    optimizer_updates=0,
                    dope_sha256=rt.DOPE,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                ),
            )
            return
        engine = P.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, raw = engine.evaluate(val, native=True, prove_identity=True)
        values = np.array(
            [baseline[r["context_sha256"]] for r in val], dtype=np.float64
        )
        expected = np.asarray(
            json.loads(Path(pins["reference"]).read_text())["utility"]["per_context"],
            dtype=np.float64,
        )
        if (
            values.tobytes() != expected.tobytes()
            or float(values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native91 utility bytes")
        with np.load(pins["baseline_cost_reference"], allow_pickle=False) as ref:
            if any(
                raw[c].dtype != np.float32 or raw[c].tobytes() != ref[c].tobytes()
                for c in raw
            ):
                raise RuntimeError("native91 cost bytes")
        persist_scores(out, "baseline_val", val, raw, engine)
        save(
            out / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                bit_exact=True,
                candidate_cost_bytes_exact=True,
                TRAIN_outcome_pools_opened=0,
                mean=float(values.mean()),
                initial_adapter_sha256=initial_hash,
                dope_sha256=rt.DOPE,
                frozen_sha256=frozen,
            ),
        )
        print(
            "BASELINE PASS91: native utility/cost bytes; TRAIN pools0; updates0",
            flush=True,
        )
        catalog = {}
        for line in Path(pins["catalog_copy"]).read_text().splitlines():
            row = json.loads(line)
            catalog[str(row["id"])] = row
        ids = {r["dataset_id"] for r in train + val}
        if not ids <= catalog.keys() or any(
            catalog[c].get("origin") != "real" for c in ids
        ):
            raise RuntimeError("real S3 catalog join")
        save(
            out / "catalog_join.json",
            dict(status="PASS", datasets=len(ids), synthetic=0),
        )
        torch.save(initial, out / "initial_adapter.pt")
        B = {s: {a: {} for a in d.ALL_ARMS} for s in d.SEEDS}
        A = {s: {a: {} for a in d.ALL_ARMS} for s in d.SEEDS}
        cells = []
        stage = d.StageOrderGuard()
        for seed in d.SEEDS:
            for outer in range(5):
                fit, b, a = d.partition(train, outer, seed)
                excluded = [r["context_sha256"] for r in a + b]
                engine.load(fit + b)
                P.restore(net, names, initial)
                _, fit_costs = engine.evaluate(fit, native=True, prove_identity=True)
                fit_ids = [r["context_sha256"] for r in fit]
                gates = {c: engine.pools[c]["gate"] for c in fit_ids}
                margin, gaps = obj.fit_margin(fit_costs, gates, fit_ids)
                celltag = f"seed{seed}-fold{outer}"
                margin_path = out / ("margin-" + celltag + ".json")
                save(
                    margin_path,
                    dict(
                        margin=margin,
                        definition="median absolute gap of two cheapest valid r882 full-pool costs",
                        fit_ids=fit_ids,
                        gaps=gaps,
                        locked_before_training=True,
                    ),
                )
                margin_hash = sha(margin_path)
                persist_scores(out, "margin-costs-" + celltag, fit, fit_costs, engine)
                orders = {c: np.argsort(engine.pools[c]["fitness"]) for c in fit_ids}
                batch_seed = d.h177.stream_seed(
                    seed, outer, "fit-123", "h156-batches-v1"
                )
                plan = P.stat.batch_plan(fit_ids, orders, batch_seed, updates=40)
                P.stat.validate_fit_plan(plan, fit_ids, excluded, ())
                b_ref, raw = engine.evaluate(b, native=True, prove_identity=True)
                if set(B[seed]["R882"]) & set(b_ref):
                    raise RuntimeError("duplicate B assignment")
                B[seed]["R882"].update(b_ref)
                persist_scores(out, "B-R882-" + celltag, b, raw, engine)
                for arm in d.ARMS:
                    if sha(margin_path) != margin_hash:
                        raise RuntimeError("margin changed")
                    state, traces = train_cell(
                        engine,
                        initial,
                        names,
                        fit,
                        excluded,
                        plan,
                        arm,
                        margin,
                        seed,
                        outer,
                        out,
                        counter,
                    )
                    tag = celltag + "-" + arm
                    torch.save(state, out / ("state-" + tag + ".pt"))
                    stage.mark_b_read()
                    util, raw = engine.evaluate(b)
                    B[seed][arm].update(util)
                    persist_scores(out, "B-" + tag, b, raw, engine)
                    record = dict(
                        seed=seed,
                        outer=outer,
                        arm=arm,
                        updates=40,
                        fit_families=sorted({r["family_id"] for r in fit}),
                        B_families=sorted({r["family_id"] for r in b}),
                        A_families=sorted({r["family_id"] for r in a}),
                        margin=margin,
                        margin_lock_sha256=margin_hash,
                        initial_adapter_sha256=initial_hash,
                        final_adapter_sha256=rt.tensor_hash(state),
                        batch_plan_sha256=plan_hash(plan),
                        trace=traces,
                    )
                    save(out / ("cell-" + tag + ".json"), record)
                    cells.append(record)
                    print(
                        f"B COMPLETE {tag} updates={counter['step_completed']}",
                        flush=True,
                    )
                P.restore(net, names, initial)
        bp = complete_panel(train, B)
        np.savez_compressed(
            out / "pooled_B.npz",
            panel=bp,
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            arms=np.array(d.ALL_ARMS),
            seeds=np.array(d.SEEDS),
        )
        decision, native = d.select(
            train, bp[:, :, 0], {arm: bp[:, :, j] for j, arm in enumerate(d.ARMS, 1)}
        )
        lock = out / "pooled_b_selection_lock.json"
        save(
            lock,
            dict(
                wave="H177",
                study_wave="H194",
                pooled_b_families=205,
                decisions=native,
                H194_IMH_m1=decision,
                locked_before_A=True,
                B_panel_sha256=sha(out / "pooled_B.npz"),
                candidate_step=40,
                actual_updates=counter["step_completed"],
            ),
        )
        stage.mark_b_locked(lock)
        lock_hash = sha(lock)
        print("B LOCK FSYNCED; candidate A scoring now permitted", flush=True)
        V = np.empty((3, 5, len(val), 4), dtype=np.float64)
        V[:, :, :, 0] = values
        for si, seed in enumerate(d.SEEDS):
            for outer in range(5):
                _, _, a = d.partition(train, outer, seed)
                stage.require_b_locked()
                if sha(lock) != lock_hash:
                    raise RuntimeError("B lock mutation")
                engine.load(a)
                P.restore(net, names, initial)
                util, raw = engine.evaluate(a, native=True, prove_identity=True)
                A[seed]["R882"].update(util)
                persist_scores(out, f"A-R882-seed{seed}-fold{outer}", a, raw, engine)
                for ai, arm in enumerate(d.ARMS, 1):
                    stage.require_b_locked()
                    if sha(lock) != lock_hash:
                        raise RuntimeError("B lock mutation")
                    tag = f"seed{seed}-fold{outer}-{arm}"
                    state = torch.load(
                        out / ("state-" + tag + ".pt"),
                        map_location="cpu",
                        weights_only=True,
                    )
                    P.restore(net, names, state)
                    util, raw = engine.evaluate(a)
                    A[seed][arm].update(util)
                    persist_scores(out, "A-" + tag, a, raw, engine)
                    util, raw = engine.evaluate(val)
                    V[si, outer, :, ai] = [util[r["context_sha256"]] for r in val]
                    persist_scores(out, "Val-" + tag, val, raw, engine)
                print(f"A/VAL COMPLETE seed{seed}-fold{outer}", flush=True)
        apanel = complete_panel(train, A)
        np.savez_compressed(
            out / "pooled_A.npz",
            panel=apanel,
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            arms=np.array(d.ALL_ARMS),
            seeds=np.array(d.SEEDS),
        )
        np.savez_compressed(
            out / "DLVal.npz",
            panel=V,
            context_ids=np.array([r["context_sha256"] for r in val]),
            family_ids=np.array([r["family_id"] for r in val]),
            arms=np.array(d.ALL_ARMS),
            seeds=np.array(d.SEEDS),
        )
        selected = apanel.copy()
        if not decision["admitted"]:
            selected[:, :, 2] = selected[:, :, 0]
        selected_A = d.summarize(train, selected)
        result = dict(
            wave="H194",
            **d.verdict(decision, selected_A),
            pooled_B=decision,
            pooled_A=selected_A,
            unconditional_A=d.summarize(train, apanel),
            DLVal=d.summarize(val, V.mean(axis=1)),
            DLVal_scope="Descriptive expected uniform choice among five fold models; utility averaging, no score ensemble; never selected on Val",
            optimizer_updates=counter["step_completed"],
            trajectories=len(cells),
            outer_folds=5,
            seeds=list(d.SEEDS),
            fold_seeds=list(d.FOLD_SEEDS),
            bootstrap_seed=d.BOOT_SEED,
            B_lock_sha256=lock_hash,
            dope_sha256=rt.DOPE,
            inference="unchanged active r882 kernel; no gate veto or oracle masking",
            historical_TRAIN_exposure=True,
            multiplicity="single owner-prescribed IMH test; no claim of H178 m4 empirical calibration",
        )
        P.restore(net, names, initial)
        engine.check()
        dope_after = guard.snapshot_dope_weights(net).composite_sha256
        if rt.tensor_hash(net.state_dict()) != whole_hash or dope_after != rt.DOPE:
            raise RuntimeError("restored whole model/DOPE differs")
        if len(cells) != 45 or set(counter.values()) != {1800}:
            raise RuntimeError("fixed complete45/1800 grid required")
        save(out / "metrics.json", result)
        save(
            out / "audit.json",
            dict(
                status="PASS",
                **counter,
                audit=audit.receipt(),
                dope_before=rt.DOPE,
                dope_after=dope_after,
                whole_model_restored=True,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        print("COMPLETE " + result["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(exc),
                counter=counter,
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
