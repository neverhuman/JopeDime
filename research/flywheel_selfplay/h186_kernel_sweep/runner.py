"""One zero-update H186 attempt; native baseline, pooled B lock, then A and DL-Val."""

import argparse
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
import analysis as stat
from parents import ROOT, engine_parent, h184_treatment, h177
from treatment import adapter_components, constants, scores

HERE = Path(__file__).resolve().parent
parent = engine_parent()
sha, save = parent.sha, parent.save


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources", "source_sha256"):
        for path, digest in pins[kind].items():
            p = (
                ROOT / path
                if kind == "parent_sources"
                else HERE / path
                if kind == "source_sha256"
                else Path(path)
            )
            if sha(p) != digest:
                raise RuntimeError(f"{kind} pin mismatch: {p}")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version: " + name)


def forbidden_training(*args, **kwargs):
    raise RuntimeError("H186 forbids optimizer construction and backward")


class Engine(parent.Engine):
    def set_policy_guard(self, guard, initial_hash):
        self.stage_guard = guard
        self.initial_hash = initial_hash
        self.selection_hash = None

    def evaluate_many(self, rows, stage, output, tag):
        if stage not in ("B", "A", "DL-Val"):
            raise ValueError("invalid scoring stage")
        with stat.bindings():
            if stage == "B":
                if self.stage_guard.b_locked_path is not None:
                    raise RuntimeError("B scoring after lock")
                self.stage_guard.mark_b_read()
            else:
                self.stage_guard.require_b_locked()
                if (
                    stat.lock_hash(self.stage_guard.b_locked_path)
                    != self.selection_hash
                ):
                    raise RuntimeError("selection lock mutation")
        groups, utilities, choices, raw, invalid, proof = {}, {}, {}, {}, {}, 0
        for row in rows:
            cid = row["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.inference_mode():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    kwargs, tokens, masks, _, _ = self.batch(
                        batch, [np.arange(k) for _ in batch]
                    )
                    if any(cid not in self.cached for cid in batch):
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                    anchor = tuple(
                        torch.stack([self.cached[c][j] for c in batch])
                        for j in range(2)
                    )
                    static = h184_treatment.static_terms(
                        self.net,
                        kwargs["prog_412"],
                        kwargs["sos_prior"],
                        tokens,
                        masks,
                        anchor[1],
                    )
                    args = (
                        self.net.adapter_10tok,
                        static,
                        kwargs["scale"],
                        kwargs["revealed_indices"],
                        kwargs["revealed_raw_f"],
                        kwargs["revealed_gate"],
                    )
                    native_parts = h184_treatment.adapter_components(*args)
                    base_parts = adapter_components(*args)
                    if any(
                        not torch.equal(native_parts[key], base_parts[key])
                        for key in native_parts
                    ):
                        raise RuntimeError("H184/R882 component parity")
                    native = (
                        self.net.adapter_10tok.compute_adapter_delta_from_static_terms(
                            static_terms=static,
                            scale=kwargs["scale"],
                            revealed_indices=kwargs["revealed_indices"],
                            revealed_raw_f=kwargs["revealed_raw_f"],
                            revealed_gate=kwargs["revealed_gate"],
                        )
                    )
                    h184_treatment.assert_component_sum(base_parts, native)
                    base = scores(anchor[0], base_parts)
                    cost = [base.numpy().copy()]
                    for arm in stat.ARMS:
                        parts = adapter_components(*args, **constants(arm))
                        predicted = scores(anchor[0], parts)
                        if arm.startswith("TOPK") and not torch.equal(predicted, base):
                            raise RuntimeError(
                                "four reveals must make TOPK4/16 exact no-ops"
                            )
                        cost.append(predicted.numpy().copy())
                    stacked = np.stack(cost, axis=1)
                    if not np.isfinite(stacked).all():
                        raise RuntimeError("nonfinite candidate scores")
                    for i, cid in enumerate(batch):
                        pick = np.argmin(stacked[i], axis=1)
                        pool = self.pools[cid]
                        utilities[cid] = np.where(
                            pool["gate"][pick], -pool["fitness"][pick], -2.0
                        )
                        choices[cid] = pick
                        invalid[cid] = ~pool["gate"][pick]
                        raw[cid] = stacked[i]
                    proof += len(batch)
        self.check()
        if (
            self.rt.tensor_hash(parent.adapter_state(self.net, self.names))
            != self.initial_hash
        ):
            raise RuntimeError("adapter weights changed in zero-update experiment")
        np.savez_compressed(output / (tag + "-scores.npz"), **raw)
        ordered = [r["context_sha256"] for r in rows]
        np.savez_compressed(
            output / (tag + "-choices.npz"),
            context_ids=np.asarray(ordered),
            choice=np.stack([choices[c] for c in ordered]),
            utility=np.stack([utilities[c] for c in ordered]),
            invalid=np.stack([invalid[c] for c in ordered]),
        )
        save(
            output / (tag + ".json"),
            dict(
                stage=stage,
                contexts=ordered,
                families=sorted({r["family_id"] for r in rows}),
                component_parity_contexts=proof,
                adapter_sha256=self.initial_hash,
                optimizer_updates=0,
                selection_lock_sha256=self.selection_hash if stage != "B" else None,
            ),
        )
        print("SCORED", tag, len(rows), flush=True)
        return utilities, invalid


def fold_panel(engine, train, stage, output):
    values = np.empty((3, len(train), 8))
    invalid = np.empty_like(values, dtype=bool)
    position = {r["context_sha256"]: i for i, r in enumerate(train)}
    for si, seed in enumerate(stat.SEEDS):
        seen = set()
        for outer in range(5):
            with stat.bindings(seed):
                fit, select, assess = h177.partition(train, outer)
            rows = select if stage == "B" else assess
            result, bad = engine.evaluate_many(
                rows, stage, output, f"{stage}-s{seed}-f{outer}"
            )
            if seen.intersection(result):
                raise RuntimeError("duplicate stage contexts")
            seen.update(result)
            for cid in result:
                values[si, position[cid]] = result[cid]
                invalid[si, position[cid]] = bad[cid]
        if seen != set(position):
            raise RuntimeError("incomplete stage coverage")
    return values, invalid


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
        raise RuntimeError("CPU xbabe2 nice19 required")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    h184_treatment.validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = parent.runtime()
    train, val = parent.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
    if (
        stat.fold_manifest(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["reshuffles"]
    ):
        raise RuntimeError("fresh fold manifest mismatch")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()

    def interrupted(signum, frame):
        raise RuntimeError(f"signal {signum}; no retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, dope_guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial = parent.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        torch.optim.Optimizer.__init__ = forbidden_training
        torch.autograd.backward = forbidden_training
        rt.check_frozen(net, dope_guard, names, frozen)
        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    contexts_scored=0,
                    optimizer_updates=0,
                    backward_calls=0,
                    initial_adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                ),
            )
            return
        with stat.bindings():
            stage_guard = stat.StageOrderGuard(dict.fromkeys(stat.ARMS, 1))
        engine = Engine(net, rt, names, frozen, dope_guard, fused, train + val)
        engine.set_policy_guard(stage_guard, initial_hash)
        engine.load(val)
        baseline, baseline_raw = engine.evaluate(val, native=True, prove_identity=True)
        baseline_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        reference = json.loads(Path(pins["reference"]).read_text())
        if (
            not np.array_equal(baseline_values, reference["utility"]["per_context"])
            or float(baseline_values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91-context r882 baseline failed")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                mean=float(baseline_values.mean()),
                bit_exact=True,
                optimizer_updates=0,
                dope_sha256=rt.DOPE,
            ),
        )
        np.savez_compressed(output / "baseline-scores.npz", **baseline_raw)
        print("BASELINE PASS 91/91 bit-exact", flush=True)
        engine.load(train)
        b, bad_b = fold_panel(engine, train, "B", output)
        np.savez_compressed(output / "B-panel.npz", utility=b, invalid=bad_b)
        decisions = stat.select(train, b)
        lock = output / "pooled_b_selection_lock.json"
        save(
            lock,
            dict(
                wave="H177",
                study="H186",
                schema_note="Imported H177 guard schema; candidate step=1 is a version ID, actual updates=0",
                pooled_b_families=205,
                pooled_b_contexts=1219,
                decisions=decisions,
                candidate_constants={arm: constants(arm) for arm in stat.ALL_ARMS},
                initial_adapter_sha256=initial_hash,
                b_panel_sha256=sha(output / "B-panel.npz"),
                locked_before_a_and_treatment_val=True,
            ),
        )
        with stat.bindings():
            stage_guard.mark_b_locked(lock)
        engine.selection_hash = sha(lock)
        a, bad_a = fold_panel(engine, train, "A", output)
        vals, invalids = [], []
        for seed in stat.SEEDS:
            result, bad = engine.evaluate_many(val, "DL-Val", output, f"VAL-s{seed}")
            vals.append(np.stack([result[r["context_sha256"]] for r in val]))
            invalids.append(np.stack([bad[r["context_sha256"]] for r in val]))
        v, bad_v = np.stack(vals), np.stack(invalids)
        np.savez_compressed(
            output / "final-panels.npz",
            B=b,
            A=a,
            VAL=v,
            invalid_B=bad_b,
            invalid_A=bad_a,
            invalid_VAL=bad_v,
            train_context_ids=np.asarray([r["context_sha256"] for r in train]),
            val_context_ids=np.asarray([r["context_sha256"] for r in val]),
            arms=np.asarray(stat.ALL_ARMS),
            seeds=np.asarray(stat.SEEDS),
        )
        b_summary, a_summary, v_summary = (
            stat.summarize(rows, panel)
            for rows, panel in [(train, b), (train, a), (val, v)]
        )
        calibration_path = (
            ROOT
            / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
        )
        decision = stat.verdict(
            decisions,
            a_summary,
            json.loads(calibration_path.read_text()),
            sha(Path(h177.__file__)),
        )
        dose_columns = [1, 0, 2, 3]
        dose_gains = [
            0.0 if i == 0 else a_summary["contrasts"][stat.ALL_ARMS[i]]["median_gain"]
            for i in dose_columns
        ]
        engine.check()
        if rt.tensor_hash(parent.adapter_state(net, names)) != initial_hash:
            raise RuntimeError("final adapter drift")
        metrics = dict(
            wave="H186",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            optimizer_updates=0,
            backward_calls=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            adapter_sha256_before=initial_hash,
            adapter_sha256_after=initial_hash,
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            pooled_B=decisions,
            B_readout=b_summary,
            A_readout=a_summary,
            DL_Val_descriptive=v_summary,
            selection_lock_sha256=sha(lock),
            dose_response=dict(
                multipliers=[0.5, 1, 1.5, 2],
                A_median_gain=dose_gains,
                nondecreasing=bool(np.all(np.diff(dose_gains) >= 0)),
                nonincreasing=bool(np.all(np.diff(dose_gains) <= 0)),
            ),
            invalid_counts_by_seed=dict(
                B=bad_b.sum(axis=1).tolist(),
                A=bad_a.sum(axis=1).tolist(),
                VAL=bad_v.sum(axis=1).tolist(),
            ),
            identical_seed_panels=dict(
                B=bool(np.array_equal(b[0], b[1]) and np.array_equal(b[1], b[2])),
                A=bool(np.array_equal(a[0], a[1]) and np.array_equal(a[1], a[2])),
            ),
            pooled_A_equals_B=bool(np.array_equal(a, b)),
            topk_structural_noop=True,
            limitations=[
                "Zero fitted models; seed variation is only fold reshuffling.",
                "Pooled A is a correlated reuse of the same 205 families, not independent confirmation.",
                "H178 calibrated original m=4; H186 uses the same code with stricter m=7 and has no separate m=7 empirical calibration.",
                "Historically exposed r882 TRAIN and repeatedly exposed DL-Val; no sealed qualification.",
            ],
            serving_promotion=False,
        )
        if len(json.dumps(metrics, indent=2).encode()) >= 100000:
            raise RuntimeError("metrics >=100KB")
        save(output / "metrics.json", metrics)
        save(
            output / "audit.json",
            dict(
                status="PASS",
                **audit.receipt(),
                optimizer_updates=0,
                backward_calls=0,
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as error:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(error),
                optimizer_updates=0,
                backward_calls=0,
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
