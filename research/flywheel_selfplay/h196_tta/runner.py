"""One frozen CPU attempt: native91, FIT/B, fsynced lock, A, descriptive Val."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import boto3
import numpy as np
import torch
import analysis as stat
from engine import Engine
from parents import ROOT, h177, h188, h192
from s3_arrays import upload
from treatment import averaged_pick, fit_scale

HERE = Path(__file__).resolve().parent
sha, save = h192.sha, h192.save


def forbidden_training(*args, **kwargs):
    raise RuntimeError("H196 forbids optimizer construction/backward")


def save_panel(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources", "source_sha256"):
        for name, digest in pins[kind].items():
            path = (
                ROOT / name
                if kind == "parent_sources"
                else HERE / name
                if kind == "source_sha256"
                else Path(name)
            )
            if sha(path) != digest:
                raise RuntimeError(f"{kind} pin mismatch: {path}")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def utility(pool, pick):
    return float(-pool["fitness"][pick] if pool["gate"][pick] else -2.0)


def panel(rows, engine, native, noise, perm=None, reused_perm=None):
    result = []
    for row in rows:
        cid = row["context_sha256"]
        pool = engine.pools[cid]
        noise_pick, _ = averaged_pick(noise[cid])
        single_pick = int(np.argmin(noise[cid][0]))
        pv = native[cid]
        if perm is not None:
            pv = utility(pool, averaged_pick(perm[cid])[0])
        if reused_perm is not None:
            pv = reused_perm[cid]
        result.append(
            [native[cid], utility(pool, noise_pick), pv, utility(pool, single_pick)]
        )
    return np.asarray(result, np.float64)


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
        raise RuntimeError("xbabe2/nice19/CPU required")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    h188.validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = h192.runtime()
    train, val = h192.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
    folds = stat.fold_manifest(train)
    if folds != json.loads((HERE / "fold_manifest.json").read_text())["reshuffles"]:
        raise RuntimeError("fold manifest mismatch")
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(args.launch.resolve())}
        | {
            str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
            for r in train + val
        }
    )
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()

    def interrupted(signum, frame):
        raise RuntimeError(f"signal {signum}; no scientific retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    remote = {}
    try:
        net, dope_guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial_hash = rt.tensor_hash(h192.adapter_state(net, names))
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
                    adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                ),
            )
            return
        client = boto3.client("s3", region_name="us-east-2")

        def store(key, arrays):
            receipt = upload(client, key + ".npz", arrays)
            remote[key] = receipt
            save(output / "remote-arrays.json", remote)
            return receipt

        engine = Engine(net, rt, names, frozen, dope_guard, fused, train + val)
        engine.initial_hash = initial_hash
        engine.check()
        engine.load(val)
        baseline, raw_val = engine.evaluate(val, native=True, prove_identity=True)
        reference = json.loads(Path(pins["reference"]).read_text())
        base_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        if (
            not np.array_equal(base_values, reference["utility"]["per_context"])
            or float(base_values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native91 baseline mismatch")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                bit_exact=True,
                mean=float(base_values.mean()),
                per_context=base_values.tolist(),
            ),
        )
        store("native/VAL", raw_val)
        print("BASELINE PASS 91/91 bit-exact", flush=True)
        engine.load(train)
        native, raw_train = engine.evaluate(train, native=True, prove_identity=True)
        store("native/TRAIN", raw_train)
        engine.native_costs = {**raw_train, **raw_val}
        native.update(baseline)
        probes = {}
        for seed in stat.SEEDS:
            fit, _, _ = stat.partition(train, seed, 0)
            probes[str(seed)], arrays = engine.probe(fit, seed)
            probes[str(seed)]["receipt"] = store(f"FIT-probe/s{seed}", arrays)
            print("FIT PROBE", seed, "skip", probes[str(seed)]["skip"], flush=True)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        bpanel = np.full((3, len(train), 4), np.nan)
        cells = {}
        for si, seed in enumerate(stat.SEEDS):
            for outer in range(5):
                fit, b, a = stat.partition(train, seed, outer)
                key = f"s{seed}-f{outer}"
                scale = fit_scale(
                    [r["context_sha256"] for r in fit], engine.tokens, engine.masks
                )
                noise = engine.noise(b, scale, seed, outer)
                noise_receipt = store("B-noise/" + key, noise)
                perm = None if probes[str(seed)]["skip"] else engine.permuted(b, seed)
                perm_receipt = None if perm is None else store("B-perm/" + key, perm)
                values = panel(b, engine, native, noise, perm)
                for row, v in zip(b, values, strict=True):
                    bpanel[si, positions[row["context_sha256"]]] = v
                cells[key] = dict(
                    seed=seed,
                    outer=outer,
                    scale=scale.tolist(),
                    fit_families=sorted({r["family_id"] for r in fit}),
                    b_families=sorted({r["family_id"] for r in b}),
                    a_families=sorted({r["family_id"] for r in a}),
                    b_family_means=h177.family_matrix(
                        values, [r["family_id"] for r in b]
                    )[1]
                    .mean(axis=0)
                    .tolist(),
                    noise_receipt=noise_receipt,
                    perm_receipt=perm_receipt,
                )
                del noise, perm
                print("B COMPLETE", key, flush=True)
        if not np.isfinite(bpanel).all():
            raise RuntimeError("incomplete B coverage")
        save_panel(output / "B-panel.npz", utility=bpanel)
        val_scales = {
            str(s): np.median(
                [cells[f"s{s}-f{k}"]["scale"] for k in range(5)], axis=0
            ).tolist()
            for s in stat.SEEDS
        }
        lockpath = output / "b_lock.json"
        save(
            lockpath,
            dict(
                wave="H196",
                sigma=0.05,
                cells=cells,
                probes=probes,
                val_scales=val_scales,
                locked_before_a_and_treatment_val=True,
                b_panel_sha256=sha(output / "B-panel.npz"),
                fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
                adapter_sha256=initial_hash,
                manifest_sha256=sha(HERE / "manifest.json"),
                native_receipts={
                    k: v for k, v in remote.items() if k.startswith("native/")
                },
            ),
        )
        guard = stat.StageGuard(sha(HERE / "fold_manifest.json"), initial_hash)
        guard.lock(lockpath)
        print("LOCK PASS all15 B cells", flush=True)
        apanel = np.full_like(bpanel, np.nan)
        for si, seed in enumerate(stat.SEEDS):
            reused_perm = {cid: bpanel[si, i, 2] for cid, i in positions.items()}
            for outer in range(5):
                locked = guard.require()
                _, _, a = stat.partition(train, seed, outer)
                key = f"s{seed}-f{outer}"
                noise = engine.noise(a, locked["cells"][key]["scale"], seed, outer)
                store("A-noise/" + key, noise)
                values = panel(a, engine, native, noise, reused_perm=reused_perm)
                for row, v in zip(a, values, strict=True):
                    apanel[si, positions[row["context_sha256"]]] = v
                del noise
                print("A COMPLETE", key, flush=True)
        vpanel = []
        for seed in stat.SEEDS:
            locked = guard.require()
            noise = engine.noise(val, locked["val_scales"][str(seed)], seed, "VAL")
            store(f"VAL-noise/s{seed}", noise)
            perm = None if probes[str(seed)]["skip"] else engine.permuted(val, seed)
            if perm is not None:
                store(f"VAL-perm/s{seed}", perm)
            vpanel.append(panel(val, engine, native, noise, perm))
            del noise, perm
        vpanel = np.asarray(vpanel)
        guard.require()
        engine.check()
        save_panel(
            output / "final-panels.npz",
            B=bpanel,
            A=apanel,
            VAL=vpanel,
            train_context_ids=np.asarray(list(positions)),
            val_context_ids=np.asarray([r["context_sha256"] for r in val]),
            arms=np.asarray(stat.ARMS),
        )
        a_summary, v_summary = (
            stat.summarize(train, apanel),
            stat.summarize(val, vpanel),
        )
        decision = stat.verdict(a_summary)
        metrics = dict(
            wave="H196",
            status="COMPLETE",
            **decision,
            optimizer_updates=0,
            backward_calls=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            adapter_sha256_before=initial_hash,
            adapter_sha256_after=rt.tensor_hash(h192.adapter_state(net, names)),
            seeds=stat.SEEDS,
            fold_seeds=stat.FOLD_SEEDS,
            arms=stat.ARMS,
            sigma=0.05,
            pooled_A=a_summary,
            DL_Val_descriptive=v_summary,
            pairwise_fold_jaccards=stat.jaccards(folds),
            permutation_probes={
                s: {k: v for k, v in p.items() if k != "receipt"}
                for s, p in probes.items()
            },
            b_lock_sha256=sha(lockpath),
            remote_objects=len(remote),
            limitations=[
                "Historically exposed TRAIN/Val; no sealed data. Cross-fold FIT/B/A roles overlap.",
                "Bootstrap conditions on fixed scales, fold allocations, augmentation draws and FIT probe decisions; three reshuffles are not independent replications.",
                "PERM8 skip is conditional on 64 predeclared FIT bit comparisons, not proof of global invariance.",
                "Harm counts any strictly negative seed-mean family delta; supplementary loss below -.01 is also reported.",
                "DL-Val is descriptive; no live adoption.",
            ],
        )
        if len(json.dumps(metrics).encode()) >= 100000:
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
        print("COMPLETE", decision["verdict"], flush=True)
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
                remote_arrays=remote,
            ),
        )
        raise


if __name__ == "__main__":
    main()
