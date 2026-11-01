"""Single guarded H183 attempt; fixed FIT-only teachers, targets and paired student."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import analysis as stat
from distill import (
    require_fit,
    student_objective,
    teacher_distribution,
    validate_initial_hashes,
)
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


def fit_cache(engine, plan, fit_ids, excluded_ids):
    batches = []
    for _, ids, indexes in plan:
        require_fit(ids, fit_ids, excluded_ids)
        kwargs, tokens, masks, fitness, gate = engine.batch(ids, indexes)
        batches.append((kwargs, tokens, masks, fitness, gate, engine.anchored(kwargs)))
    return batches


def make_targets(net, rt, engine, names, states, batches, plan, fit_ids, excluded_ids):
    if len(states) != 10 or len(batches) != 80:
        raise RuntimeError("ten teachers and fixed80-step student slate plan required")
    costs = np.empty((10, 80, 8, 24), dtype=np.float32)
    with torch.no_grad():
        for mi, state in enumerate(states):
            P.restore(net, names, state)
            for bi, (kwargs, tokens, masks, _, _, anchor) in enumerate(batches):
                require_fit(plan[bi][1], fit_ids, excluded_ids)
                costs[mi, bi] = engine.forward(kwargs, tokens, masks, anchor).numpy()
            engine.check()
    mean, prob = teacher_distribution(costs)
    return costs, mean, prob


def train_student(
    net, rt, engine, names, initial, batches, plan, target, fit_ids, excluded_ids
):
    P.restore(net, names, initial)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    snapshots, traces = {}, []
    for j, ((stream, ids, _), batch) in enumerate(zip(plan, batches, strict=True), 1):
        require_fit(ids, fit_ids, excluded_ids)
        kwargs, tokens, masks, fitness, gate, anchor = batch
        torch.manual_seed(stream)
        optimizer.zero_grad(set_to_none=True)
        pred = engine.forward(kwargs, tokens, masks, anchor)
        loss, live, kl = student_objective(pred, fitness, gate, rt.loss, target[j - 1])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite student loss")
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        )
        optimizer.step()
        engine.check()
        traces.append(
            dict(
                step=j,
                stream=stream,
                fit_context_ids=ids,
                loss=float(loss.detach()),
                live_loss=float(live.detach()),
                teacher_KL=float(kl.detach()),
                raw_grad_norm=norm,
            )
        )
        if j in (40, 80):
            snapshots[j] = P.adapter_state(net, names)
    return snapshots, traces


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
    verify(pins)
    if pins["max_total_updates"] != 7800 or pins["arms"] != list(stat.ARMS):
        raise RuntimeError("fixed H183 arm/update budget identity")
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
    folds = stat.family_folds([r["family_id"] for r in train])
    if folds != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fresh fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    counter = {"updates": 0}
    original_step = torch.optim.AdamW.step
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial = P.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        params = [p for p in net.parameters() if p.requires_grad]
        if (
            len(names) != 47
            or len(params) != 47
            or sum(p.numel() for p in params) != 29211
        ):
            raise RuntimeError("only47 adapter tensors/29211 parameters may train")
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("effective dropout/frozen mode mismatch")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    scored_contexts=0,
                    optimizer_updates=0,
                    trainable_parameters=29211,
                    adapter_tensors=47,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                ),
            )
            return
        engine = P.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, baseline_costs = engine.evaluate(
            val, native=True, prove_identity=True
        )
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
            raise RuntimeError("exact91 native r882 baseline bytes/mean")
        with np.load(pins["baseline_cost_reference"], allow_pickle=False) as ref:
            if any(
                baseline_costs[c].dtype != np.float32
                or baseline_costs[c].tobytes() != ref[c].tobytes()
                for c in baseline_costs
            ):
                raise RuntimeError("native candidate-cost reference bytes")
        np.savez_compressed(output / "baseline_costs.npz", **baseline_costs)
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                mean=float(values.mean()),
                bit_exact=True,
                candidate_cost_bytes_exact=True,
                TRAIN_outcome_pools_opened=0,
                dope_sha256=rt.DOPE,
                frozen_sha256=frozen,
                initial_adapter_sha256=initial_hash,
            ),
        )
        print("BASELINE PASS:91/91 utility and native cost bytes;0updates", flush=True)
        fresh = {}
        init_hashes = {}
        all_seeds = [s for seed in stat.SEEDS for s in stat.teacher_seeds(seed)] + list(
            stat.STUDENT_SEEDS
        )
        for seed in all_seeds:
            fresh[seed] = P.random_adapter_state(net, names, seed)
            init_hashes[seed] = rt.tensor_hash(fresh[seed])
            engine.check()
        if len(set(init_hashes.values())) != 33:
            raise RuntimeError("33distinct constructor initialization hashes required")
        save(output / "initial_hashes.json", init_hashes)
        torch.save(fresh, output / "initial_adapter_states.pt")
        allowed_params = {id(p) for p in params}

        def counted_step(optimizer, *a, **kw):
            ps = [p for group in optimizer.param_groups for p in group["params"]]
            if len(ps) != 47 or {id(p) for p in ps} != allowed_params:
                raise RuntimeError("optimizer escaped private adapter")
            if counter["updates"] >= pins["max_total_updates"]:
                raise RuntimeError("optimizer update budget exhausted")
            value = original_step(optimizer, *a, **kw)
            counter["updates"] += 1
            return value

        torch.optim.AdamW.step = counted_step
        policies = {s: {a: {} for a in stat.ARMS} for s in stat.SEEDS}
        agreements = {s: {a: {} for a in stat.PRIMARY_FAMILY} for s in stat.SEEDS}
        cell_records = []
        fold_count = None
        for outer in range(5):
            if fold_count is not None and outer >= fold_count:
                break
            fit_f, b_f, a_f = stat.partition(
                [r["family_id"] for r in train], outer, folds
            )
            fit = [r for r in train if r["family_id"] in fit_f]
            assess = [r for r in train if r["family_id"] in a_f]
            fit_ids = [r["context_sha256"] for r in fit]
            excluded = [
                r["context_sha256"] for r in train if r["family_id"] in b_f | a_f
            ]
            engine.load(fit)
            orders = {c: np.argsort(engine.pools[c]["fitness"]) for c in fit_ids}
            for seed in stat.SEEDS:
                tag = f"A{outer}-S{seed}"
                cell_start = time.monotonic()
                before = counter["updates"]
                teachers = stat.teacher_seeds(seed)
                ss = stat.student_seed(seed)
                validate_initial_hashes(
                    [init_hashes[x] for x in teachers], init_hashes[ss]
                )
                finals = []
                records = []
                for mi, ts in enumerate(teachers):
                    batch_seed = stat.stream_seed(
                        seed, outer, f"TEACHER{mi + 1}", "h183-fit-v1"
                    )
                    plan = stat.batch_plan(fit_ids, orders, batch_seed, updates=40)
                    stat.validate_fit_plan(plan, fit_ids, excluded, ())
                    c = P.run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        fresh[ts],
                        fit,
                        batch_seed,
                        40,
                        output,
                        tag,
                        plan=plan,
                        excluded_context_ids=excluded,
                    )
                    finals.append(c["state"])
                    rec = dict(
                        role="TEACHER",
                        member=mi + 1,
                        init_seed=ts,
                        initial_hash=init_hashes[ts],
                        final_hash=rt.tensor_hash(c["state"]),
                        fit_plan_sha256=plan_hash(plan),
                        trace=c["traces"],
                    )
                    records.append(rec)
                if len({x["final_hash"] for x in records}) != 10:
                    raise RuntimeError("trained teacher states unexpectedly identical")
                batch_seed = stat.stream_seed(seed, outer, "STUDENT", "h183-fit-v1")
                plan = stat.batch_plan(fit_ids, orders, batch_seed, updates=80)
                stat.validate_fit_plan(plan, fit_ids, excluded, ())
                batches = fit_cache(engine, plan, fit_ids, excluded)
                costs, mean, target = make_targets(
                    net, rt, engine, names, finals, batches, plan, fit_ids, excluded
                )
                np.savez_compressed(
                    output / (tag + "-FIT-targets.npz"),
                    teacher_costs=costs,
                    mean_cost=mean,
                    probability=target.numpy(),
                    context_ids=np.array([ids for _, ids, _ in plan]),
                    candidate_indices=np.array(
                        [[x for x in indexes] for _, _, indexes in plan]
                    ),
                )
                save(
                    output / (tag + "-FIT-target-lock.json"),
                    dict(
                        FIT_context_ids=fit_ids,
                        excluded_A_B_ids=excluded,
                        plan_sha256=plan_hash(plan),
                        teacher_final_hashes=[rt.tensor_hash(x) for x in finals],
                        probability_sha256=hashlib.sha256(
                            target.numpy().tobytes()
                        ).hexdigest(),
                    ),
                )
                control = P.run_cell(
                    net,
                    rt,
                    engine,
                    names,
                    frozen,
                    guard,
                    fresh[ss],
                    fit,
                    batch_seed,
                    40,
                    output,
                    tag,
                    plan=plan[:40],
                    excluded_context_ids=excluded,
                )
                snapshots, traces = train_student(
                    net,
                    rt,
                    engine,
                    names,
                    fresh[ss],
                    batches,
                    plan,
                    target,
                    fit_ids,
                    excluded,
                )
                if counter["updates"] - before != 520:
                    raise RuntimeError(
                        "cell must contain exactly520 actual optimizer updates"
                    )
                if fold_count is None:
                    execution = stat.execution_plan(time.monotonic() - cell_start)
                    save(
                        output / "pilot_budget.json",
                        dict(
                            **execution,
                            FIT_only=True,
                            pilot_updates=520,
                            budget_selected_before_A_scores=True,
                            fallback="first3outerfolds ofsame5foldpartition;123assessedfamilies",
                        ),
                    )
                    if not execution["within_cap"]:
                        raise RuntimeError(
                            "FIT-only timing pilot exceeds four-hour cap"
                        )
                    fold_count = execution["outer_folds"]
                save(
                    output / (tag + "-FIT-training.json"),
                    dict(
                        outer_fold=outer,
                        seed=seed,
                        fit_families=sorted(fit_f),
                        A_families=sorted(a_f),
                        B_reserved_unused=sorted(b_f),
                        optimizer_updates=520,
                        teachers=records,
                        FRESH1_trace=control["traces"],
                        DIST10_80_trace=traces,
                        shared_student_init_hash=init_hashes[ss],
                        student_plan_sha256=plan_hash(plan),
                        DIST10_step40_hash=rt.tensor_hash(snapshots[40]),
                        DIST10_step80_hash=rt.tensor_hash(snapshots[80]),
                        FRESH1_final_hash=rt.tensor_hash(control["state"]),
                    ),
                )
                torch.save(
                    dict(
                        teachers=finals,
                        FRESH1=control["state"],
                        DIST10=snapshots[40],
                        DIST10_80=snapshots[80],
                    ),
                    output / (tag + "-adapter-states.pt"),
                )
                # All targets/training and the timing-only plan are fixed before any model sees A.
                engine.load(assess)
                P.restore(net, names, initial)
                _, raw = engine.evaluate(assess, native=True, prove_identity=True)
                base_u, _ = persist_scores(output, tag + "-R882-A", assess, raw, engine)
                policies[seed]["R882"].update(base_u)
                random_scores, trained_scores = [], []
                for mi, ts in enumerate(teachers):
                    for label, state, collection in [
                        ("RANDOM", fresh[ts], random_scores),
                        ("TRAINED", finals[mi], trained_scores),
                    ]:
                        P.restore(net, names, state)
                        _, raw = engine.evaluate(assess)
                        persist_scores(
                            output, tag + f"-{label}{mi + 1}-A", assess, raw, engine
                        )
                        collection.append(raw)
                teacher_choices = None
                for arm, collection in [
                    ("RENS10", random_scores),
                    ("ENS10", trained_scores),
                ]:
                    averaged = {
                        row["context_sha256"]: np.mean(
                            np.stack([v[row["context_sha256"]] for v in collection]),
                            axis=0,
                            dtype=np.float32,
                        )
                        for row in assess
                    }
                    u, choices = persist_scores(
                        output, tag + "-" + arm + "-A", assess, averaged, engine
                    )
                    policies[seed][arm].update(u)
                    if arm == "ENS10":
                        teacher_choices = choices
                for arm, state in [
                    ("FRESH1", control["state"]),
                    ("DIST10", snapshots[40]),
                    ("DIST10_80", snapshots[80]),
                ]:
                    P.restore(net, names, state)
                    _, raw = engine.evaluate(assess)
                    u, choices = persist_scores(
                        output, tag + "-" + arm + "-A", assess, raw, engine
                    )
                    policies[seed][arm].update(u)
                    if arm in stat.PRIMARY_FAMILY:
                        agreements[seed][arm].update(
                            {
                                cid: choices[cid] == teacher_choices[cid]
                                for cid in choices
                            }
                        )
                cell_records.append(
                    dict(
                        outer=outer,
                        seed=seed,
                        actual_updates=520,
                        teacher_initial_hashes=[init_hashes[t] for t in teachers],
                        teacher_final_hashes=[rt.tensor_hash(s) for s in finals],
                    )
                )
                del (
                    batches,
                    costs,
                    mean,
                    target,
                    random_scores,
                    trained_scores,
                    finals,
                    snapshots,
                    control,
                )
                P.restore(net, names, initial)
                engine.check()
                save(
                    output / "progress.json",
                    dict(
                        completed_cells=len(cell_records),
                        optimizer_updates=counter["updates"],
                        outer_folds=fold_count,
                    ),
                )
                print(f"CELL COMPLETE {tag} updates={counter['updates']}", flush=True)
        if (
            fold_count is None
            or len(cell_records) != fold_count * 3
            or counter["updates"] != fold_count * 3 * 520
        ):
            raise RuntimeError("incomplete fixed cell grid")
        ids = sorted(policies[stat.SEEDS[0]]["R882"])
        family_of = {r["context_sha256"]: r["family_id"] for r in train}
        fam_ids = [family_of[c] for c in ids]
        panel = np.array(
            [
                [[policies[s][arm][c] for arm in stat.ARMS] for c in ids]
                for s in stat.SEEDS
            ]
        )
        agree = {
            arm: np.array([[agreements[s][arm][c] for c in ids] for s in stat.SEEDS])
            for arm in stat.PRIMARY_FAMILY
        }
        summary = stat.summarize(panel, fam_ids, agreement=agree)
        summary.update(
            wave="H183",
            status="COMPLETE",
            source_commit=head,
            manifest_sha256=sha(HERE / "manifest.json"),
            outer_folds=fold_count,
            actual_optimizer_updates=counter["updates"],
            nominal_arm_exposures=fold_count * 3 * 560,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            initial_adapter_sha256=initial_hash,
            baseline_mean=1.1377941898164525,
            historical_r882_TRAIN_exposure=True,
            raw_A_candidate_score_and_choice_persistence=True,
            student40_is_prefix_of_student80=True,
            unused_selector_B=True,
        )
        np.savez_compressed(
            output / "OOF-panels.npz",
            context_ids=np.array(ids),
            family_ids=np.array(fam_ids),
            seed_ids=np.array(stat.SEEDS),
            arm_names=np.array(stat.ARMS),
            utility=panel,
            **{a + "_teacher_agreement": v for a, v in agree.items()},
        )
        save(output / "metrics.json", summary)
        save(
            output / "audit.json",
            dict(
                status="PASS",
                **audit.receipt(),
                optimizer_updates=counter["updates"],
                dope_sha256=rt.DOPE,
                frozen_sha256=frozen,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        print("COMPLETE " + summary["verdict"], flush=True)
    except BaseException as exc:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(exc),
                optimizer_updates=counter["updates"],
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise
    finally:
        torch.optim.AdamW.step = original_step


if __name__ == "__main__":
    main()
