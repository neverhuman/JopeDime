"""H181 baseline-first frozen model audit, then read-only TRAIN-bank statistics."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import psutil
import torch
import science as s
import replay
import posthoc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(manifest):
    for kind in ["inputs", "assets", "parent_sources", "source_sha256"]:
        for name, digest in manifest[kind].items():
            path = (
                ROOT / name
                if kind == "parent_sources"
                else HERE / name
                if kind == "source_sha256"
                else Path(name)
            )
            if sha(path) != digest:
                raise RuntimeError("pin mismatch " + str(path))
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch " + name)


def baseline_gate(values, reference):
    actual = np.asarray(values, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    if (
        actual.shape != (91,)
        or expected.shape != (91,)
        or actual.tobytes() != expected.tobytes()
        or float(actual.mean()) != 1.1377941898164525
    ):
        raise RuntimeError("exact native91 baseline refusal")


def parents():
    path = ROOT / "research/flywheel_selfplay/h170_onecycle/runner.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("h181_parent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module.runtime()


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
        raise RuntimeError("CPU xbabe2/nice19 only")
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
        and launch["reviewed_head"]
        and launch["review_receipt"]
    ):
        raise RuntimeError("launch/source/review binding")
    out = args.output.resolve()
    out.mkdir()
    if args.smoke:
        replay.save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                model_loads=0,
                scored_contexts=0,
                source_pins_verified=True,
            ),
        )
        return

    def forbidden(*args, **kwargs):
        raise RuntimeError("H181 forbids every optimizer construction/backward")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.set_grad_enabled(False)
    parent, rt = parents()
    train, val = parent.cohorts(rt)
    families = sorted({r["family_id"] for r in train})
    folds = json.loads((HERE / "fold_manifest.json").read_text())
    if [[families[j] for j in g] for g in s.groups()] != folds["families"]:
        raise RuntimeError("fold pin")
    allowed = (
        set(manifest["inputs"]) | set(manifest["assets"]) | {str(args.launch.resolve())}
    )
    allowed |= {str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in val}
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    net, guard, names, frozen, fused = rt.build(manifest["checkpoint"])
    initial = rt.tensor_hash(net.state_dict())
    for param in net.parameters():
        param.requires_grad_(False)
    if initial != manifest["all_model_state_sha256"]:
        raise RuntimeError("state identity")

    def check():
        d = psutil.disk_usage("/home/ubuntu")
        proc = psutil.Process()
        if (
            os.getloadavg()[0] > 0.85 * os.sysconf("SC_NPROCESSORS_ONLN")
            or psutil.virtual_memory().available < 16 * 2**30
            or d.free < 40_000_000_000
            or d.used / (d.used + d.free) >= 0.938
            or proc.num_threads() > 4
            or proc.memory_info().rss >= 6_000_000_000
        ):
            raise RuntimeError("H181 terminal resource bound")
        if torch.cuda.is_initialized() or any(
            p.requires_grad for p in net.parameters()
        ):
            raise RuntimeError("device/trainability bound")
        rt.check_frozen(net, guard, names, frozen)

    check()
    engine = parent.Engine(net, rt, names, frozen, guard, fused, train + val)
    engine.load(val)
    baseline, vcost = engine.evaluate(val, native=True, prove_identity=True)
    ref = json.loads(Path(manifest["reference"]).read_text())
    baseline_gate(baseline["values"], ref["utility"]["per_context"])
    with np.load(manifest["baseline_cost_reference"], allow_pickle=False) as gold:
        if not all(vcost[c].tobytes() == gold[c].tobytes() for c in vcost):
            raise RuntimeError("native cost bit identity")
    replay.save(
        out / "baseline.json",
        dict(
            status="PASS",
            native_contexts=91,
            mean=1.1377941898164525,
            native_costs_bit_exact=True,
            all_model_state_sha256=initial,
            DOPE_sha256=rt.DOPE,
            optimizer_updates=0,
            TRAIN_outcome_pools_opened=0,
        ),
    )
    print(
        "BASELINE PASS native91 mean1.1377941898164525 exact utilities/cost bits; DOPE frozen; all trainable0",
        flush=True,
    )
    # Only now decode TRAIN counterfactual outcomes and closed-wave panels.
    raw = Path(manifest["catalog_copy"]).read_bytes()
    provenance = json.loads((HERE / "provenance.json").read_text())
    if hashlib.sha256(raw).hexdigest() != provenance["catalog_sha256"]:
        raise RuntimeError("catalog pin")
    catalog = {r["id"]: r for r in map(json.loads, raw.splitlines())}
    ids = {r["dataset_id"] for r in train + val}
    if not all(catalog[k]["origin"] == "real" for k in ids) or len(ids) != 276:
        raise RuntimeError("real S3 provenance")
    replay.save(
        out / "provenance.json",
        dict(
            status="PASS",
            real_dataset_ids=len(ids),
            zero_synthetic_datasets=True,
            H175_bank_bytes_pinned=True,
            catalog_version=provenance["catalog_version_id"],
            all_training_outcomes_from_real_S3=True,
            policy_simulations_not_synthetic_datasets=True,
        ),
    )
    bank = Path(manifest["bank"])
    costs = {}
    utilities = {}
    with np.load(bank / "counterfactual_inputs.npz", allow_pickle=False) as z:
        if len(z.files) != 2620:
            raise RuntimeError("bank closure")
        for row in train:
            cid = row["context_sha256"]
            costs[cid] = z[cid + "_cost"].copy()
            utilities[cid] = z[cid + "_utility"].copy()
            if (
                len(costs[cid]) != row["n_candidates"]
                or len(utilities[cid]) != row["n_candidates"]
            ):
                raise RuntimeError("candidate count")
            costs[cid].setflags(write=False)
            utilities[cid].setflags(write=False)
        if not all(vcost[c].tobytes() == z[c + "_cost"].tobytes() for c in vcost):
            raise RuntimeError("H175 native baseline costs mismatch")
    del engine, vcost
    metrics = replay.run(train, costs, utilities, bank, out, check)
    X = np.stack(
        [s.covariates(train, costs, utilities, "full", seed) for seed in s.BANK_SEEDS]
    )
    metrics["posthoc"] = posthoc.run(train, X, out)
    check()
    final = rt.tensor_hash(net.state_dict())
    if initial != final or audit.refused:
        raise RuntimeError("frozen state/audit")
    verify(manifest)
    metrics.update(
        source_commit=head,
        manifest_sha256=sha(HERE / "manifest.json"),
        baseline_mean=1.1377941898164525,
        model_trainable_parameters=0,
        model_state_before_sha256=initial,
        model_state_after_sha256=final,
        dope_sha256=rt.DOPE,
        frozen_sha256=frozen,
        audit=audit.receipt(),
        provenance="real S3 H175 TRAIN bank; new181 activation/selector streams; source175 MC panels retained, not new independent datasets",
    )
    replay.save(out / "metrics.json", metrics)
    print("COMPLETE " + metrics["verdict"], flush=True)


if __name__ == "__main__":
    main()
