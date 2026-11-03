"""H262 single offline run: paired round streams, FIT-only rules, final-only DL-Val."""

import argparse
import copy
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
from treatment import ControlAdamW, adapter_drift, gradient_norm

HERE = Path(__file__).resolve().parent
PREFIX = "research/flywheel_selfplay/h262_strict_acceptance_replay/"


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
    if receipt["completed_rounds"] != 60 or receipt["optimizer_updates"] != 2400:
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
    head = os.environ["H262_SOURCE"]
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
    fit, accept = stat.partition(train)
    fold = json.loads((HERE / "fold_manifest.json").read_text())
    for role, rows in [("FIT_train", fit), ("FIT_accept", accept)]:
        if fold[role + "_contexts"] != [r["context_sha256"] for r in rows]:
            raise RuntimeError("fold context identity")
    if {r["family_id"] for r in fit} & {r["family_id"] for r in accept + val}:
        raise RuntimeError("family leak")
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
    rounds = []
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
        del base_costs
        engine.load(train)
        fitids = {r["context_sha256"] for r in fit}
        forbidden = {r["context_sha256"] for r in accept + val}
        orders = {cid: np.argsort(engine.pools[cid]["fitness"]) for cid in fitids}

        def update(opt, stream, cids, indexes):
            eng.cooperate()
            if not set(cids) <= fitids or set(cids) & forbidden:
                raise RuntimeError("training boundary")
            torch.manual_seed(stream)
            opt.zero_grad(set_to_none=True)
            kw, tok, mask, fitness, gate = engine.batch(cids, indexes)
            anchor = engine.anchored(kw)
            pred = engine.forward(kw, tok, mask, anchor)
            loss = rt.loss(pred, fitness, gate)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            raw = float(
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            )
            clipped = gradient_norm(params)
            opt.step()
            engine.check()
            return dict(
                loss=float(loss.detach()), raw_grad_l2=raw, clipped_grad_l2=clipped
            )

        smoke = []
        for arm in ["CURRENT", "STRICT_LCB"]:
            restore(initial)
            opt = ControlAdamW(params)
            trace = update(
                opt,
                *stat.batch_plan(fitids, orders, stat.stream_seed(stat.SEEDS[0], 1))[0],
            )
            changed = rt.tensor_hash(snapshot()) != initial_hash
            if not changed:
                raise RuntimeError("smoke unchanged")
            smoke.append(dict(arm=arm, parameters_changed=changed, **trace))
        restore(initial)
        del opt
        release()
        if rt.tensor_hash(snapshot()) != initial_hash:
            raise RuntimeError("smoke restore")
        eng.save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=2,
                restored=True,
                trained_assessment_reads=0,
                arms=smoke,
            ),
        )
        print("SMOKE PASS 2 updates; restored r882", flush=True)
        accept_base, accept_costs = engine.evaluate(accept)
        durable_npz(out / "accept-baseline-costs.npz", accept_costs)
        del accept_costs
        eng.save(out / "accept-baseline.json", accept_base)
        afams = [r["family_id"] for r in accept]
        final_files = {}
        finals = {}
        accept_counts = {}
        plan_hashes = {}
        for seed in stat.SEEDS:
            for arm in ["CURRENT", "STRICT_LCB"]:
                restore(initial)
                opt = ControlAdamW(params)
                incumbent = snapshot()
                inc_opt = copy.deepcopy(opt.state_dict())
                inc_panel = accept_base
                accepted_count = 0
                for round_idx in range(1, 11):
                    name = f"{seed}-{arm}-r{round_idx:02d}"
                    plan = stat.batch_plan(
                        fitids, orders, stat.stream_seed(seed, round_idx)
                    )
                    digest = stat.plan_digest(plan)
                    key = f"{seed}-{round_idx}"
                    if key in plan_hashes and plan_hashes[key] != digest:
                        raise RuntimeError("unpaired random stream")
                    plan_hashes[key] = digest
                    before_hash = rt.tensor_hash(incumbent)
                    before_drift = adapter_drift(initial, incumbent)
                    traces = []
                    for step, item in enumerate(plan, 1):
                        trace = update(opt, *item)
                        updates += 1
                        traces.append(dict(step=step, stream=item[0], **trace))
                    candidate = snapshot()
                    candidate_hash = rt.tensor_hash(candidate)
                    if candidate_hash == before_hash:
                        raise RuntimeError("no candidate parameter update")
                    candidate_file = out / (name + "-candidate.npz")
                    candidate_sha = durable_npz(
                        candidate_file, {n: t.numpy() for n, t in candidate.items()}
                    )
                    panel, costs = engine.evaluate(accept)
                    costfile = out / (name + "-accept-costs.npz")
                    costsha = durable_npz(costfile, costs)
                    del costs
                    if arm == "CURRENT":
                        decision = stat.current_accept(
                            panel["values"], inc_panel["values"]
                        )
                    else:
                        decision = stat.strict_accept(
                            panel["values"],
                            inc_panel["values"],
                            afams,
                            stat.stream_seed(stat.BOOT_SEED, seed, round_idx, "accept"),
                        )
                    old_values = inc_panel["values"]
                    if decision["accepted"]:
                        incumbent = candidate
                        inc_opt = copy.deepcopy(opt.state_dict())
                        inc_panel = panel
                        accepted_count += 1
                    else:
                        restore(incumbent)
                        opt.load_state_dict(copy.deepcopy(inc_opt))
                    if rt.tensor_hash(snapshot()) != rt.tensor_hash(incumbent):
                        raise RuntimeError("incumbent restore hash")
                    record = dict(
                        seed=seed,
                        arm=arm,
                        round=round_idx,
                        decision=decision,
                        before_hash=before_hash,
                        candidate_hash=candidate_hash,
                        incumbent_hash=rt.tensor_hash(incumbent),
                        candidate_file=candidate_file.name,
                        candidate_file_sha256=candidate_sha,
                        accept_costs_file=costfile.name,
                        accept_costs_sha256=costsha,
                        candidate_panel=panel,
                        previous_values=old_values,
                        incumbent_values=inc_panel["values"],
                        batch_plan_sha256=digest,
                        trace=traces,
                        optimizer_state_steps=sorted(
                            {int(v["step"]) for v in opt.state.values() if "step" in v}
                        ),
                        candidate_drift=adapter_drift(initial, candidate),
                        before_drift=before_drift,
                        incumbent_drift=adapter_drift(initial, incumbent),
                    )
                    eng.save(out / (name + ".json"), record)
                    rounds.append(
                        dict(
                            seed=seed,
                            arm=arm,
                            round=round_idx,
                            accepted=decision["accepted"],
                            receipt=name + ".json",
                        )
                    )
                    eng.save(
                        out / "progress.json",
                        dict(
                            complete_rounds=len(rounds),
                            optimizer_updates=updates,
                            last_round=name,
                            round_receipt_sha256=eng.sha(out / (name + ".json")),
                        ),
                    )
                    print("ROUND COMPLETE", name, "updates", updates, flush=True)
                    del candidate, panel, traces, record, plan
                    release()
                key = f"{seed}-{arm}"
                file = out / (key + "-final.npz")
                final_files[file.name] = durable_npz(
                    file, {n: t.numpy() for n, t in incumbent.items()}
                )
                finals[key] = dict(
                    file=file.name,
                    adapter_sha256=rt.tensor_hash(incumbent),
                    drift=adapter_drift(initial, incumbent),
                )
                accept_counts[key] = accepted_count
                del opt, inc_opt, incumbent
                release()
        if updates != 2400 or len(rounds) != 60:
            raise RuntimeError("incomplete trajectory")
        eng.save(
            out / "final-lock.json",
            dict(
                completed_rounds=60,
                optimizer_updates=updates,
                files=final_files,
                finals=finals,
                accept_counts=accept_counts,
                batch_plan_sha256=plan_hashes,
                rounds=rounds,
            ),
        )
        sync_directory(out)
        lock_sha = assert_final_lock(out)
        print("ALL FINAL CHECKPOINTS LOCKED", flush=True)
        values = np.empty((3, 91, 3))
        top_values = np.empty_like(values)
        final_panels = {}
        read_count = 0
        values[:, :, 0] = baseline["values"]
        top_values[:, :, 0] = [x["utility"] for x in base_top]
        for si, seed in enumerate(stat.SEEDS):
            for ai, arm in enumerate(["CURRENT", "STRICT_LCB"], 1):
                if assert_final_lock(out) != lock_sha:
                    raise RuntimeError("final lock changed")
                key = f"{seed}-{arm}"
                with np.load(out / finals[key]["file"], allow_pickle=False) as z:
                    state = {n: torch.from_numpy(z[n].copy()) for n in names}
                restore(state)
                del state
                panel, costs = engine.evaluate(val)
                read_count += 1
                top = top4(val, costs)
                costfile = out / (key + "-final-val-costs.npz")
                digest = durable_npz(costfile, costs)
                del costs
                final_panels[key] = dict(
                    single_pick=panel,
                    top4=top,
                    costs_file=costfile.name,
                    costs_sha256=digest,
                    adapter=finals[key],
                    accept_count=accept_counts[key],
                )
                values[si, :, ai] = panel["values"]
                top_values[si, :, ai] = [x["utility"] for x in top]
                release()
        durable_npz(out / "panels.npz", dict(single_pick=values, top4=top_values))
        eng.save(out / "final-panels.json", final_panels)
        summary = stat.summarize(val, values)
        topsummary = stat.summarize(val, top_values)
        primary = summary["contrasts"]["STRICT_LCB_minus_r882"]
        secondary = summary["contrasts"]["STRICT_LCB_minus_CURRENT"]
        harm = {}
        for ai, arm in enumerate(["CURRENT", "STRICT_LCB"], 1):
            eligible = [
                si
                for si, seed in enumerate(stat.SEEDS)
                if accept_counts[f"{seed}-{arm}"] > 0
            ]
            worse = [
                si
                for si in eligible
                if stat.family_matrix(
                    values[si, :, ai] - values[si, :, 0], [r["family_id"] for r in val]
                )[1].mean()
                < 0
            ]
            harm[arm] = dict(
                trajectories_with_accepts=len(eligible),
                final_family_worse_count=len(worse),
                accept_then_final_DLVal_worse_rate=len(worse) / len(eligible)
                if eligible
                else None,
                per_round_DLVal_worse_rate=None,
                limitation="No intermediate DL-Val scoring permitted; trajectory-level final harm only, not per-accept harm.",
            )
        after = guard.snapshot_dope_weights(net).composite_sha256
        if after != rt.DOPE or read_count != 6:
            raise RuntimeError("frozen/final scoring count")
        metrics = dict(
            wave="H262",
            status="COMPLETE",
            verdict="ADVANCE" if primary["advance"] else "NO_ADVANCE",
            finding="checkpoint lift"
            if primary["advance"]
            else "acceptance-rule finding"
            if secondary["advance"]
            else "no qualifying acceptance-rule or checkpoint lift",
            primary_verdict="ADVANCE" if primary["advance"] else "NO_ADVANCE",
            secondary_verdict="ADVANCE" if secondary["advance"] else "NO_ADVANCE",
            source_commit=head,
            manifest_sha256=eng.sha(HERE / "manifest.json"),
            seeds=list(stat.SEEDS),
            arms=list(stat.ARMS),
            optimizer_updates=updates,
            smoke_updates=2,
            complete_rounds=60,
            baseline_mean=1.1377941898164525,
            baseline_top4_mean=float(top_values[0, :, 0].mean()),
            primary_dlval=summary,
            top4_dlval=topsummary,
            top4_role="descriptive test-time four-reveal search, not checkpoint lift",
            accept_counts=accept_counts,
            finals=finals,
            accept_then_worse=harm,
            trained_DLVal_evaluations=read_count,
            intermediate_DLVal_evaluations=0,
            final_lock_sha256=lock_sha,
            dope_before=rt.DOPE,
            dope_after=after,
            initial_adapter_sha256=initial_hash,
            trainable_parameters=29211,
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        eng.save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], metrics["finding"], flush=True)
    except BaseException as exc:
        eng.save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                complete_rounds=len(rounds),
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
