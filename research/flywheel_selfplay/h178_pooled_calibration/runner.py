"""Single guarded H178 read-only calibration audit. Optimizers/backward are forbidden."""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calibration as cal  # noqa: E402


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
            raise RuntimeError("Package mismatch: " + package)


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
    spec = importlib.util.spec_from_file_location("h178_native_parent", path)
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
        cal.save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                scored_contexts=0,
                model_loads=0,
                source_pins_verified=True,
                imported_h177_rule_identity=bool(
                    cal.pooled_b_select is cal.h177_analysis.pooled_b_select
                ),
            ),
        )
        return

    def forbidden(*args, **kwargs):
        raise RuntimeError("H178 forbids optimizer construction and backward")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.set_grad_enabled(False)
    parent, rt = parents()
    if rt.DOPE != "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814":
        raise RuntimeError("DOPE sha256 mismatch before run")
    train, val = parent.cohorts(rt)
    if (
        cal.folds(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["families"]
    ):
        raise RuntimeError("fold pin mismatch")
    allowed = (
        set(manifest["inputs"]) | set(manifest["assets"]) | {str(args.launch.resolve())}
    )
    allowed |= {str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in val}
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    net, guard, names, frozen, fused = rt.build(manifest["checkpoint"])
    initial = rt.tensor_hash(net.state_dict())
    if initial != "6eb0e998896bd87553bbea815be06b534d5f3f1ae2a63cbf7ff4ec22e89c4f8a":
        raise RuntimeError("Initial model state SHA-256 mismatch")
    for param in net.parameters():
        param.requires_grad_(False)
    if any(x.requires_grad for x in net.parameters()):
        raise RuntimeError("All H178 model tensors must be frozen")

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
            raise RuntimeError("H178 resource bound")
        if torch.cuda.is_initialized() or any(
            x.requires_grad for x in net.parameters()
        ):
            raise RuntimeError("Device/trainability bound")
        rt.check_frozen(net, guard, names, frozen)

    check()
    # 1. Exact 91-context r882 baseline PASS first BEFORE any TRAIN outcome pools or cached counterfactual inputs are opened.
    engine = parent.Engine(net, rt, names, frozen, guard, fused, train + val)
    engine.load(val)
    baseline, vcost = engine.evaluate(val, native=True, prove_identity=True)
    ref = json.loads(Path(manifest["reference"]).read_text())
    baseline_gate(baseline["values"], ref["utility"]["per_context"])
    with np.load(manifest["baseline_cost_reference"], allow_pickle=False) as gold:
        assert all(vcost[c].tobytes() == gold[c].tobytes() for c in vcost)
    cal.save(
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
        "BASELINE PASS native91 utilities and cost bits; zero optimizer updates; TRAIN_outcome_pools_opened=0",
        flush=True,
    )

    # 2. Verify real S3 catalog provenance before opening H175's cached counterfactual_inputs.npz.
    provenance = json.loads((HERE / "provenance.json").read_text())
    raw = Path(manifest["catalog_copy"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["catalog_sha256"]:
        raise RuntimeError("catalog SHA mismatch")
    catalog = {r["id"]: r for r in map(json.loads, raw.splitlines())}
    ids = {r["dataset_id"] for r in train + val}
    if not all(catalog[x].get("origin") == "real" for x in ids):
        raise RuntimeError("non-real S3 dataset")
    (out / "real-catalog.jsonl").write_bytes(raw)
    cal.save(
        out / "provenance.json",
        dict(
            status="PASS",
            actual_preflight_exact_version_catalog_GET=True,
            real_dataset_ids=len(ids),
            zero_synthetic_datasets=True,
            counterfactual_policy_injections=True,
            reused_h175_counterfactual_inputs_sha256=sha(
                manifest["counterfactual_inputs"]
            ),
        ),
    )

    # 3. Load H175's cached counterfactual_inputs.npz and verify native DL-Val cost bit-identity.
    payload = {}
    with np.load(manifest["counterfactual_inputs"], allow_pickle=False) as cf:
        for row in train + val:
            cid = row["context_sha256"]
            payload[cid] = (cf[cid + "_cost"], cf[cid + "_utility"])
    if not all(payload[c][0].tobytes() == vcost[c].tobytes() for c in vcost):
        raise RuntimeError("DL-Val cached cost bit mismatch vs live engine")
    del engine, vcost

    h175_sim = json.loads(Path(manifest["h175_simulation_metrics"]).read_text())
    headroom, curves, verdict_info = cal.run_calibration(
        train, val, payload, out, check, h175_ref=h175_sim
    )
    check()
    final = rt.tensor_hash(net.state_dict())
    if (
        initial != final
        or rt.DOPE != "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
        or audit.refused
    ):
        raise RuntimeError("Post-run model state, DOPE, or audit integrity failure")

    metrics = dict(
        wave="H178",
        title="Empirical Calibration of the H177 Pooled Cross-Fit Selector on the H175 Counterfactual Bank",
        status="COMPLETE",
        null_and_harmful_admission_le_005=verdict_info[
            "null_and_harmful_admission_le_005"
        ],
        max_null_or_harmful_admission_rate=verdict_info[
            "max_null_or_harmful_admission_rate"
        ],
        replayed_four_arm_holm_m=verdict_info["replayed_four_arm_holm_m"],
        h177_analysis_sha256=verdict_info["h177_analysis_sha256"],
        calibration_verdict=verdict_info["calibration_verdict"],
        power_verdict=verdict_info["power_verdict"],
        verdict=verdict_info["verdict"],
        verdict_summary=verdict_info,
        imported_h177_rule_identity=bool(
            cal.pooled_b_select is cal.h177_analysis.pooled_b_select
        ),
        seeds=list(cal.SEEDS),
        slate_seeds=list(cal.SLATE_SEEDS),
        fold_seed=cal.FOLD_SEED,
        boot_seed=cal.BOOT_SEED,
        optimizer_updates=0,
        model_trainable_parameters=0,
        model_state_before_sha256=initial,
        model_state_after_sha256=final,
        dope_sha256=rt.DOPE,
        frozen_sha256=frozen,
        baseline_mean=1.1377941898164525,
        headroom=headroom,
        curves=curves,
        selector_cohorts_per_pooled_3seed_rule=2800,
        selector_arm_decisions_per_pooled_3seed_rule=11200,
        selector_decisions_per_single_seed_diagnostic=33600,
        selector_decisions_per_single_fold_rule=168000,
        source_commit=head,
        manifest_sha256=sha(HERE / "manifest.json"),
        audit=audit.receipt(),
        serving_promotion=False,
        limitations=[
            "Counterfactual oracle injections use outcomes pointwise within B_outer and are not deployable learned policies.",
            "Nonnegative oracle gains give optimistic power; they omit the mixed-sign per-family errors of real training.",
            "Primary H177_POOLED consumes the exact returned decisions of h177_analysis.pooled_b_select (m=4 step-down Holm across CTRL, ACC4, ACC16, OC15 on 3-seed within-family averaged (3, 205) panels across 50 disjoint 4-arm cohorts / 200 marginal arm decisions per cell); cloned-4-arm tie and single-seed cloned sensitivities are reported as explicit diagnostics.",
            "For the pointwise H175 policy bank, pooled A and pooled B contain the same 205 families and 1,219 contexts across rotated fold roles (correlated cross-fold candidate readout, not an independent new-family confirmation sample), and any policy with 3-seed finite-TRAIN gain <= 0 already fails the deterministic > 0.001 + 1e-12 mean gate.",
            "Testing exact zero and random-nonoracle finite-bank policies does not certify external-population effects just below the +0.001 margin (e.g. an external normal population with mu=0.00099, family SD=0.001, N=205 crosses the observed +0.001 gate ~44.3% of the time).",
            "Synthetic policies on real S3 TRAIN contexts; zero synthetic datasets; frozen historical r882 and reused TRAIN/DL-Val cannot establish external generalization.",
        ],
    )
    cal.save(out / "h178_receipt.json", verdict_info["h178_receipt"])
    cal.save(out / "metrics.json", metrics)
    print(
        "COMPLETE",
        verdict_info["verdict"],
        "optimizer_updates=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
