#!/usr/bin/env python3
"""Real DL-Train forward/backward smoke; no optimizer, assessment, queue or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h105_cpu_science", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--shared-source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise RuntimeError("unique CPU receipt required")
    if os.getloadavg()[0] > 85:
        raise RuntimeError("CPU load exceeds85")
    r.configure_source(args.source_commit, args.shared_source_root)
    proof = r.verify_execution_source(args.source_commit)
    torch.set_num_threads(4)

    def forbidden(*a, **k):
        raise RuntimeError("CPU validation forbids CUDA and optimizer construction")

    torch.cuda._lazy_init = forbidden
    torch.cuda.init = forbidden
    torch.optim.AdamW = forbidden
    input_dir = Path.home() / "jope-data/h50/input"
    for name, h in [
        ("source_manifest.json", r.MANIFEST_SHA),
        ("selected.json", r.SELECTED_SHA),
        ("cpu_exact_fused.py", r.FUSED_SHA),
    ]:
        if r.sha256(input_dir / name) != h:
            raise RuntimeError("H50 input hash mismatch")
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if (
        manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("H50 closure mismatch")
    selected = json.loads((input_dir / "selected.json").read_text())
    rows = selected["train"]
    split = r.family_assignment(rows)
    training = [x for x in rows if x["family_id"] in split["train_families"]]
    assessment = [x for x in rows if x["family_id"] in split["assessment_families"]]
    chosen = sorted(training, key=lambda x: x["context_sha256"])[:16]
    if (
        len(rows) != 1219
        or len(split["train_families"]) != 164
        or len(split["assessment_families"]) != 41
    ):
        raise RuntimeError("full metadata partition required")
    allowed = {x["context_sha256"] for x in chosen}
    opened = set()
    refused = []

    def audit(event, argv):
        if event != "open" or not isinstance(argv[0], (str, bytes)):
            return
        p = Path(argv[0]).resolve()
        parts = str(p).lower()
        if any(
            s in parts
            for s in (
                "admiralty",
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "flywheel-full-20261030",
            )
        ):
            refused.append(str(p))
            raise RuntimeError("protected/live path refused")
        if p.parent in [input_dir / "pools", input_dir / "tables"]:
            if p.stem not in allowed:
                refused.append(str(p))
                raise RuntimeError("non-smoke/assessment payload refused")
            opened.add(p.stem)

    sys.addaudithook(audit)
    shared = r.runtime()
    shared.validate_paths(input_dir, chosen)
    h48 = r.load("h105_cpu_h48", r.h59.H48_PATH)
    checkpoint = Path.home() / "jope-data/h51/ckpt/round-882.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = r.h59.stage_r882_copy(checkpoint, args.out.parent / "cpu-smoke-scratch")
    pools, tables = r.h55.load_rows(input_dir, chosen, h48)
    net, params, guard, _ = r.h55.build_model(
        input_dir, staged, torch.device("cpu"), h48
    )
    before = guard.snapshot_dope_weights(net).composite_sha256
    initial = r.h55.adapter_state(net)
    initial_digest = r.h55.adapter_digest(initial)
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device("cpu"))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cpu"))
    sampler = r.load("h105_cpu_sampler", r.h55.H53B_PATH)
    plan = shared.make_update_plan(
        shared.batch_plan(pools, sampler.cached_training_indices, 42), "CONTROL"
    )
    if len(plan) != 40:
        raise RuntimeError("recipe update-plan length mismatch")
    batch = plan[0][0]
    torch.manual_seed(batch.seed)
    prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
    loss = r.h59.loss_by_arm(
        "CONTROL", prediction, r.h59._batch_targets(pools, batch, prediction.device)
    )
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    active = [g for g in grads if g is not None]
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in active))
    if (
        not torch.isfinite(loss)
        or not active
        or not all(torch.isfinite(g).all() for g in active)
        or norm <= 0
    ):
        raise RuntimeError("finite positive real-train gradient required")
    base = {n: t.double() for n, t in initial.items()}
    named = [n for n, p in net.named_parameters() if p.requires_grad]
    fine = {
        n: base[n] - (1e-4 / max(norm, 1.0)) * g.double()
        if g is not None
        else base[n].clone()
        for n, g in zip(named, grads, strict=True)
    }
    distance = math.sqrt(sum(float((fine[n] - base[n]).square().sum()) for n in base))
    for alpha in (0.0, 0.25, 0.5, 1.0):
        candidate = r.interpolate(base, fine, alpha)
        actual = math.sqrt(
            sum(float((candidate[n] - base[n]).square().sum()) for n in base)
        )
        if not math.isclose(actual, alpha * distance, rel_tol=1e-8, abs_tol=1e-12):
            raise RuntimeError("shrinkage distance mismatch")
    penalty = float(r.anchor_penalty(fine, base))
    if not math.isclose(
        penalty, 0.5 * r.ANCHOR_LAMBDA * distance**2, rel_tol=1e-10, abs_tol=1e-14
    ):
        raise RuntimeError("anchor raw-L2 mismatch")
    after = guard.snapshot_dope_weights(net).composite_sha256
    if (
        before != after
        or before != r.DOPE_SHA
        or r.h55.adapter_digest(r.h55.adapter_state(net)) != initial_digest
    ):
        raise RuntimeError("CPU smoke mutated weights")
    if torch.cuda.is_initialized() or refused or opened != allowed:
        raise RuntimeError("CPU payload/device closure mismatch")
    if r.verify_execution_source(args.source_commit) != proof:
        raise RuntimeError("CPU source changed")
    receipt = {
        "schema": "jope.h105.real-train-cpu-validation/v1",
        "status": "PASS",
        "source_proof": proof,
        "protocol_sha256": r.protocol_hash(),
        "train_families": 164,
        "assessment_families": 41,
        "train_contexts": len(training),
        "assessment_contexts": len(assessment),
        "smoke_training_contexts": 16,
        "smoke_contexts_sha256": hashlib.sha256(
            "".join(sorted(allowed)).encode()
        ).hexdigest(),
        "payload_contexts_opened": len(opened),
        "assessment_payloads_opened": 0,
        "DL_Val_payloads_opened": 0,
        "outer_or_protected_payloads_opened": 0,
        "training_loss": float(loss),
        "gradient_l2": norm,
        "trainable_parameters": sum(p.numel() for p in params),
        "adapter_digest_before_after": initial_digest,
        "DOPE_before": before,
        "DOPE_after": after,
        "optimizer_updates": 0,
        "planned_updates_per_cell": 40,
        "synthetic_gradient_direction_math_only": True,
        "CUDA_initialized": False,
        "GPU_ticket_created": False,
        "Elo": None,
        "admission_assessed": False,
        "read_audit_refusals": refused,
    }
    r.atomic_json(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
