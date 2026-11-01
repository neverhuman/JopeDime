"""Single zero-update attempt: native91, fifteen B choices, durable lock, A then Val."""

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
from parents import ROOT, h177, scoring_parent
from treatment import GRID, GRID_ARMS, constants

HERE = Path(__file__).resolve().parent
h186 = scoring_parent(stat, constants)
parent = h186.parent
sha, save = h186.sha, h186.save
Engine = h186.Engine


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
            raise RuntimeError("package version mismatch: " + name)


def forbidden_training(*args, **kwargs):
    raise RuntimeError("H191 forbids optimizer construction and backward")


def stage_panel(engine, train, stage, output, cells=None):
    values = np.empty((3, len(train), 6 if stage == "B" else 5))
    invalid = np.empty_like(values, dtype=bool)
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    selected = {} if stage == "B" else cells
    for si, seed in enumerate(stat.SEEDS):
        seen = set()
        for outer in range(5):
            fit, b, a = stat.partition(train, seed, outer)

            def family(rows):
                return {r["family_id"] for r in rows}

            if tuple(map(lambda rows: len(family(rows)), (fit, b, a))) != (123, 41, 41):
                raise RuntimeError("unexpected family partition sizes")
            if family(fit) & (family(b) | family(a)) or family(b) & family(a):
                raise RuntimeError("family overlap within a cell")
            rows = b if stage == "B" else a
            key, tag = f"s{seed}-f{outer}", f"{stage}-s{seed}-f{outer}"
            result, bad = engine.evaluate_many(rows, stage, output, tag)
            ids = [r["context_sha256"] for r in rows]
            if set(result) != set(ids) or seen.intersection(ids):
                raise RuntimeError("incomplete or duplicate cell")
            seen.update(ids)
            raw = np.stack([result[cid] for cid in ids])
            raw_bad = np.stack([bad[cid] for cid in ids])
            if stage == "B":
                selected[key] = dict(
                    **stat.choose(rows, raw),
                    seed=seed,
                    outer_fold=outer,
                    b_families=sorted(family(b)),
                    b_choices_sha256=sha(output / (tag + "-choices.npz")),
                )
                use, use_bad = raw, raw_bad
            else:
                use = stat.project(raw, cells[key]["grid_index"])
                use_bad = stat.project(raw_bad, cells[key]["grid_index"])
            for i, cid in enumerate(ids):
                values[si, positions[cid]], invalid[si, positions[cid]] = (
                    use[i],
                    use_bad[i],
                )
        if seen != set(positions):
            raise RuntimeError("each stage must cover every context exactly once")
    return values, invalid, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
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
    h186.h184_treatment.validate_launch(
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
        raise RuntimeError(f"signal {signum}; no scientific retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, dope_guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial_hash = rt.tensor_hash(parent.adapter_state(net, names))
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
        guard = stat.SelectionGuard(sha(HERE / "fold_manifest.json"), initial_hash)
        engine = Engine(net, rt, names, frozen, dope_guard, fused, train + val)
        engine.set_policy_guard(guard, initial_hash)
        engine.load(val)
        baseline, raw_baseline = engine.evaluate(val, native=True, prove_identity=True)
        baseline_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        reference = json.loads(Path(pins["reference"]).read_text())
        if (
            not np.array_equal(baseline_values, reference["utility"]["per_context"])
            or float(baseline_values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("exact native91 r882 baseline failed")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                mean=float(baseline_values.mean()),
                bit_exact=True,
                optimizer_updates=0,
            ),
        )
        np.savez_compressed(output / "baseline-scores.npz", **raw_baseline)
        print("BASELINE PASS 91/91 bit-exact", flush=True)
        engine.load(train)
        b, bad_b, cells = stage_panel(engine, train, "B", output)
        np.savez_compressed(output / "B-panel.npz", utility=b, invalid=bad_b)
        val_indices = {
            str(seed): stat.val_choice(
                [cells[f"s{seed}-f{k}"]["grid_index"] for k in range(5)]
            )
            for seed in stat.SEEDS
        }
        lock = output / "kx_selection_lock.json"
        save(
            lock,
            dict(
                wave="H191",
                cells=cells,
                val_grid_index=val_indices,
                grid=GRID,
                fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
                adapter_sha256=initial_hash,
                b_panel_sha256=sha(output / "B-panel.npz"),
                locked_before_a_and_treatment_val=True,
            ),
        )
        guard.mark_b_locked(lock)
        engine.selection_hash = sha(lock)
        a, bad_a, _ = stage_panel(engine, train, "A", output, cells)
        vals, val_bad = [], []
        for seed in stat.SEEDS:
            result, bad = engine.evaluate_many(val, "DL-Val", output, f"VAL-s{seed}")
            vals.append(
                stat.project(
                    np.stack([result[r["context_sha256"]] for r in val]),
                    val_indices[str(seed)],
                )
            )
            val_bad.append(
                stat.project(
                    np.stack([bad[r["context_sha256"]] for r in val]),
                    val_indices[str(seed)],
                )
            )
        v, bad_v = np.stack(vals), np.stack(val_bad)
        np.savez_compressed(
            output / "final-panels.npz",
            B_grid=b,
            A=a,
            VAL=v,
            invalid_B_grid=bad_b,
            invalid_A=bad_a,
            invalid_VAL=bad_v,
            train_context_ids=np.asarray([r["context_sha256"] for r in train]),
            val_context_ids=np.asarray([r["context_sha256"] for r in val]),
            grid_arms=np.asarray(GRID_ARMS),
            readout_arms=np.asarray(stat.READOUT_ARMS),
            seeds=np.asarray(stat.SEEDS),
        )
        a_summary, v_summary = stat.summarize(train, a), stat.summarize(val, v)
        admission = stat.select_a(train, a)
        receipt = (
            ROOT
            / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
        )
        decision = stat.verdict(
            admission,
            a_summary,
            json.loads(receipt.read_text()),
            sha(Path(h177.__file__)),
        )
        engine.check()
        if rt.tensor_hash(parent.adapter_state(net, names)) != initial_hash:
            raise RuntimeError("final adapter drift")
        metrics = dict(
            wave="H191",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            optimizer_updates=0,
            backward_calls=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            adapter_sha256_before=initial_hash,
            adapter_sha256_after=initial_hash,
            seeds=stat.SEEDS,
            fold_seeds=stat.FOLD_SEEDS,
            grid=GRID,
            readout_arms=stat.READOUT_ARMS,
            pooled_A_admission=admission,
            A_readout=a_summary,
            DL_Val_descriptive=v_summary,
            kx_selection_by_cell={
                key: cell["multiplier"] for key, cell in cells.items()
            },
            selection_counts={
                str(kx): sum(cell["multiplier"] == kx for cell in cells.values())
                for kx in GRID
            },
            val_multiplier_by_seed={
                key: GRID[index] for key, index in val_indices.items()
            },
            selection_lock_sha256=sha(lock),
            invalid_counts_by_seed=dict(
                A=bad_a.sum(axis=1).tolist(), VAL=bad_v.sum(axis=1).tolist()
            ),
            fixed_grid_seed_panels_identical=bool(
                np.array_equal(b[0], b[1]) and np.array_equal(b[1], b[2])
            ),
            kxsel_A_seed_panels_identical=bool(
                np.array_equal(a[0, :, 1], a[1, :, 1])
                and np.array_equal(a[1, :, 1], a[2, :, 1])
            ),
            limitations=[
                "Seed variation is family reshuffling, not independent fitted models; positive seed counts are not independent replication.",
                "A_k is excluded from its own B_k choice; families recur in other folds and were already exposed in H186/r882 development.",
                "Bootstrap conditions on the selected fold policies; it does not refit B selection inside bootstrap draws.",
                "H178 empirically calibrated original m4, not nested H191 m1; receipt gate is not a new empirical FWER guarantee.",
                "DL-Val and fixed2.5/3/4 readouts are descriptive only; no alternate-primary rescue or sealed qualification.",
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
