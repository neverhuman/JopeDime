"""Single guarded H175 read-only audit. Optimizers/backward are forbidden."""

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
import protocol as p
import power

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(manifest):
    for kind in ["inputs", "assets", "parent_sources"]:
        for path, digest in manifest[kind].items():
            target = ROOT / path if kind == "parent_sources" else Path(path)
            if sha(target) != digest:
                raise RuntimeError("Pin mismatch: " + str(target))
    for path, digest in manifest["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("Source mismatch: " + path)
    for package, version in manifest["packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Package mismatch")


def baseline_gate(values, reference):
    actual = np.asarray(values, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    if (
        actual.shape != (91,)
        or expected.shape != (91,)
        or actual.tobytes() != expected.tobytes()
        or float(actual.mean()) != 1.1377941898164525
    ):
        raise RuntimeError("91-context exact native baseline gate")


def parents():
    path = ROOT / "research/flywheel_selfplay/h170_onecycle/runner.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("h175_native_parent", path)
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
        and launch["reviewed_head"]
        and launch["review_id"]
    ):
        raise RuntimeError("Launch/source/review binding")
    out = args.output.resolve()
    out.mkdir()
    if args.smoke:
        power.save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                scored_contexts=0,
                model_loads=0,
                source_pins_verified=True,
            ),
        )
        return

    def forbidden(*args, **kwargs):
        raise RuntimeError("H175 forbids optimizer construction and backward")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.set_grad_enabled(False)
    parent, rt = parents()
    train, val = parent.cohorts(rt)
    if (
        p.folds(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["families"]
    ):
        raise RuntimeError("fold pin mismatch")
    hist = json.loads((HERE / "histories.json").read_text())
    allowed = (
        set(manifest["inputs"]) | set(manifest["assets"]) | {str(args.launch.resolve())}
    )
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    net, guard, names, frozen, fused = rt.build(manifest["checkpoint"])
    initial = rt.tensor_hash(net.state_dict())
    for param in net.parameters():
        param.requires_grad_(False)
    if any(x.requires_grad for x in net.parameters()):
        raise RuntimeError("All H175 model tensors must be frozen")

    def check():
        d = psutil.disk_usage("/home/ubuntu")
        proc = psutil.Process()
        if (
            os.getloadavg()[0] > 0.85 * os.sysconf("SC_NPROCESSORS_ONLN")
            or psutil.virtual_memory().available < 16 * 2**30
            or d.free < 40_000_000_000
            or d.used / (d.used + d.free) >= 0.95
            or proc.num_threads() > 4
            or proc.memory_info().rss >= 6_000_000_000
        ):
            raise RuntimeError("H175 resource bound")
        if torch.cuda.is_initialized() or any(
            x.requires_grad for x in net.parameters()
        ):
            raise RuntimeError("Device/trainability bound")
        rt.check_frozen(net, guard, names, frozen)

    check()
    engine = parent.Engine(net, rt, names, frozen, guard, fused, train + val)
    engine.load(val)
    baseline, vcost = engine.evaluate(val, native=True, prove_identity=True)
    ref = json.loads(Path(manifest["reference"]).read_text())
    baseline_gate(baseline["values"], ref["utility"]["per_context"])
    # Independent, previously restored native per-candidate bit reference.
    with np.load(manifest["baseline_cost_reference"], allow_pickle=False) as gold:
        assert all(vcost[c].tobytes() == gold[c].tobytes() for c in vcost)
    power.save(
        out / "baseline.json",
        dict(
            status="PASS",
            native_contexts=91,
            mean=1.1377941898164525,
            cached_native_costs_bit_exact=True,
            restored_reference_costs_bit_exact=True,
            all_model_state_sha256=initial,
            DOPE_sha256=rt.DOPE,
            optimizer_updates=0,
            TRAIN_outcome_pools_opened=0,
        ),
    )
    print(
        "BASELINE PASS native91 utilities and cost bits; zero optimizer updates",
        flush=True,
    )
    # Outcome decoding/statistics on TRAIN and historical metrics start only here.
    provenance = json.loads((HERE / "provenance.json").read_text())
    raw = Path(manifest["catalog_copy"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["catalog_sha256"]:
        raise RuntimeError("catalog SHA mismatch")
    catalog = {r["id"]: r for r in map(json.loads, raw.splitlines())}
    ids = {r["dataset_id"] for r in train + val}
    if not all(catalog[x].get("origin") == "real" for x in ids):
        raise RuntimeError("non-real S3 dataset")
    (out / "real-catalog.jsonl").write_bytes(raw)
    power.save(
        out / "provenance.json",
        dict(
            status="PASS",
            actual_preflight_exact_version_catalog_GET=True,
            real_dataset_ids=len(ids),
            zero_synthetic_datasets=True,
            counterfactual_policy_injections=True,
        ),
    )
    engine.load(train)
    _, tcost = engine.evaluate(train, native=False)
    cost = dict(vcost, **tcost)
    payload = {}
    for row in train + val:
        cid = row["context_sha256"]
        pool = engine.pools[cid]
        utility = np.where(pool["gate"], -pool["fitness"], -2.0)
        payload[cid] = (cost[cid], utility)
    np.savez_compressed(
        out / "counterfactual_inputs.npz",
        **{
            cid + "_" + name: values[i]
            for cid, values in payload.items()
            for i, name in enumerate(["cost", "utility"])
        },
    )
    del engine, tcost, vcost, cost
    historical = power.history(hist, ROOT)
    power.save(out / "historical_inner_B.json", historical)
    headroom, curves = power.run(train, val, payload, out, check)
    available = [
        r["max_unconstrained_inner_B_gain"]
        for r in historical
        if r.get("verdict_threshold_eligible", False)
    ]
    typical = float(np.median(available)) if available else None
    md = headroom["full"]["MDE80"]["context"]["STRICT"]
    bound = (
        md["gain"] if md["status"] == "GRID_DETECTED" else md["gain_lower_tested_bound"]
    )
    gap = headroom["full"]["TRAIN"]["equal_family_headroom"]
    if gap["ci95"][1] <= 0.001 + p.EPS:
        verdict = "CEILING_LIKELY"
    elif len(available) >= 3 and bound > typical + p.EPS:
        verdict = "SELECTOR_UNDERPOWERED"
    else:
        verdict = "MIXED"
    check()
    final = rt.tensor_hash(net.state_dict())
    assert initial == final and not audit.refused
    metrics = dict(
        wave="H175",
        status="COMPLETE",
        verdict=verdict,
        seeds=p.SEEDS,
        fold_seed=p.FOLD_SEED,
        boot_seed=p.BOOT_SEED,
        optimizer_updates=0,
        model_trainable_parameters=0,
        model_state_before_sha256=initial,
        model_state_after_sha256=final,
        dope_sha256=rt.DOPE,
        frozen_sha256=frozen,
        baseline_mean=1.1377941898164525,
        headroom=headroom,
        curves=curves,
        historical=historical,
        typical_study_max_inner_B_gain=typical,
        available_historical_studies=len(available),
        selector_decisions_per_rule=168000,
        source_commit=head,
        manifest_sha256=sha(HERE / "manifest.json"),
        audit=audit.receipt(),
        serving_promotion=False,
        limitations=[
            "Oracle injections use outcomes pointwise and are not deployable learned policies.",
            "Nonnegative oracle gains give optimistic power; they omit the mixed-sign per-family errors of real training.",
            "One counterfactual candidate differs from best-of-four snapshot selection; no empirical calibrated familywise-error claim.",
            "10k family CIs condition on fixed folds/decisions and 200 Monte Carlo draws; additional five-fold block intervals expose fold dependence.",
            "H160-H174 availability is fixed at prereg; missing live/unreplayable studies excluded, not zero-filled.",
            "Synthetic policies, no synthetic datasets; frozen historical r882 and reused TRAIN/Val cannot establish external generalization.",
        ],
    )
    power.save(out / "metrics.json", metrics)
    print("COMPLETE", verdict, "optimizer_updates=0", flush=True)


if __name__ == "__main__":
    main()
