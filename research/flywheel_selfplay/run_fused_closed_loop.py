#!/usr/bin/env python3
"""Owner-directed train-only restart; CPU-exact landmarks, frozen DOPE, GPU head.

Inputs are a custody packet containing selected.json, pools/, tables/, winners.json
and a pinned CPU-exact FusedJopeNetwork source. No Phase-2 kernels are imported.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Import the CPU-chain dependencies explicitly, never a Phase-2 worktree.
MICRO = Path("/home/ubuntu/JopeDimeMicro")
sys.path.insert(0, str(MICRO / "python"))
import rust_evaluator as evaluator  # noqa: E402 -- project path is added above.
from hpc.landmark_workspace import FusedLandmarkWorkspace  # noqa: E402
from phase12_multimodal_model import compute_semantic_struct_sig_24d  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "freeze_guard", HERE.parents[1] / "python/src/jope/north_star/dope_freeze_guard.py"
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)
sys.path.insert(0, str(HERE))
from automatic_promotion import (  # noqa: E402
    promote,
    replay,
    training_cv_envelope,
    training_indices,
    write_tree_shard,
)
from dev_incumbent import (  # noqa: E402
    CandidateNovelty,
    DevIncumbent,
    RejectedCandidateError,
    add_admission_arguments,
    batch_seed,
    gp_selection_indices,
    project_score_margins,
    proposal_plan,
    publish_checkpoint_archive,
    retain_checkpoint_payloads,
    validate_admission_options,
    warm_restart_adapter,
)
from device_pool_cache import (  # noqa: E402
    DevicePoolCache,
    landmark_float32,
    release_gpu_round,
    wait_for_gpu_headroom,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def headroom():
    while psutil.cpu_percent(interval=0.25) > 80:
        time.sleep(0.5)


def load_packet(root):
    selected = json.loads((root / "selected.json").read_text())
    tf = {r["family_id"] for r in selected["train"]}
    df = {r["family_id"] for r in selected["dev"]}
    if not tf or not df or tf & df:
        raise ValueError("family overlap or empty partition")
    pools = {}
    tables = {}
    full = selected.get("protocol") == "full-DL-Train1219-DL-Val91/v1"
    if full and (len(selected["train"]) != 1219 or len(selected["dev"]) != 91):
        raise ValueError("full cohort inventory mismatch")
    for role in ("train", "dev"):
        for rec in selected[role]:
            if rec["dl_split"] != (
                "DL-Val" if full and role == "dev" else "DL-Train"
            ) or rec["source_split"] not in (
                "train",
                "inner",
            ):
                raise ValueError("outside training closure")
            if full and rec.get("task") != "regression":
                raise ValueError(
                    "this GP evaluator is regression-only; binary requires task-correct pools and admission"
                )
            cid = rec["context_sha256"]
            pp = root / "pools" / f"{cid}.npz"
            if sha(pp) != rec["pool_sha256"]:
                raise ValueError("pool hash mismatch")
            with np.load(pp, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                pids = [
                    v.decode() if isinstance(v, bytes) else str(v)
                    for v in z["program_sha256"].tolist()
                ]
                pools[cid] = {
                    "rec": rec,
                    "pids": pids,
                    "program": prog,
                    "proj16": prog.astype(np.float64)
                    @ evaluator.get_w1_sign_matrix()
                    / np.sqrt(412),
                    "struct_sig": compute_semantic_struct_sig_24d(prog),
                    "fitness": z["fitness"].astype(np.float64),
                    "logratio": z["logratio"].astype(np.float64),
                    "gate": z["gate"].astype(bool),
                    "prior_cost": np.zeros(len(prog), dtype=np.float64),
                    "scale": 0.4208,
                }
            path = root / "tables" / f"{cid}.jtt1"
            if sha(path) != rec["train_only_jtt1_sha256"]:
                raise ValueError("training input hash mismatch")
            raw = path.read_bytes()
            nt, nq, width = struct.unpack_from("<III", raw, 4)
            if nq or raw[:4] != b"JTT1" or raw[16] != 2:
                raise ValueError("requires train-only 2-fold input")
            a = (
                np.frombuffer(raw, dtype="<f8", offset=17 + nt)
                .reshape(nt, width + 1)
                .copy()
            )
            tables[cid] = (a[:, :-1], a[:, -1])
    return selected, pools, tables


def main():
    guard.reject_research_dope_unfreeze("fused closed-loop flywheel")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--slate-size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--campaign-prior-output", type=Path, action="append", default=[])
    ap.add_argument("--best-checkpoint", type=Path)
    add_admission_arguments(ap)
    ap.add_argument("--checkpoint-keep-last", type=int, default=20)
    ap.add_argument(
        "--retention-s3-prefix",
        default="s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/retention/",
    )
    ap.add_argument(
        "--checkpoint-archive-dir",
        type=Path,
        default=Path.home() / "jope-data/archive/flywheel-retention",
    )
    ap.add_argument(
        "--retain-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="additional checkpoint pin, relative to output or absolute",
    )
    ap.add_argument(
        "--phase3-gate-bundles",
        type=Path,
        help="directory containing round-NN.json keyed-S3 qualification bundles",
    )
    args = ap.parse_args()
    try:
        args.adapter_min_gain = validate_admission_options(
            args.strict_adapter_admission, args.adapter_min_gain
        )
    except ValueError as error:
        ap.error(str(error))
    if (
        min(
            args.rounds,
            args.steps,
            args.batch_size,
            args.slate_size,
            args.checkpoint_keep_last,
        )
        <= 0
    ):
        raise ValueError("positive bounds required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(Path.home() / "jope_plan/qualification.lock", "a")  # noqa: SIM115 -- held for process lifetime
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_for_gpu_headroom(torch.device("cuda"))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    selected, pools, tables = load_packet(args.input)
    full = selected.get("protocol") == "full-DL-Train1219-DL-Val91/v1"
    reporter = None
    if full:
        from jope._telemetry import FullFlywheelReporter

        if args.steps > 2 or args.batch_size > 8192 or args.batch_size < 1219:
            raise ValueError(
                "full cohort must retain max2 steps and 1219..8192 batch bound"
            )
        reporter = FullFlywheelReporter(
            args.input / "selected.json",
            args.output,
            Path.home() / "jope_plan/scores/history.jsonl",
        )
    train = [r["context_sha256"] for r in selected["train"]]
    dev = [r["context_sha256"] for r in selected["dev"]]
    journal = args.output / "promotions.jsonl"
    replay({c: pools[c] for c in train}, journal)
    winners = json.loads((args.input / "winners.json").read_text())[
        "all_90_qualified_novel_winner_records"
    ]
    if len(winners) != 90:
        raise ValueError("historical inventory must contain 90")
    historical = []
    for row in winners:
        cid = row["context_sha256"]
        if cid not in train:
            raise ValueError("historical candidate outside train partition")
        feat, proj = evaluator.featurize_dsl_412(row["dsl"])
        cand = {
            "context_sha256": cid,
            "dsl": row["dsl"],
            "program_sha256": row["program_sha256"],
            "status": "completed",
            "cv_gate_passed": row["verified_full_jtt1_cv_gate_passed"],
            "fitness": row["verified_full_jtt1_fitness"],
            "logratio": row["verified_full_jtt1_logratio"],
            "feat412": feat,
            "proj16": proj,
            "evaluation_scope": "full_train_cv",
            "input_identity": {
                k: row[k]
                for k in (
                    "teacher_jtt1_sha256",
                    "evaluator_binary_sha256",
                    "source_scores_jsonl_sha256",
                )
            },
        }
        ev = promote(pools[cid], cand, journal, 0, historical_admission=True)
        if ev:
            historical.append(ev)
    write_json(
        args.output / "historical_injection.json",
        {
            "expected": 90,
            "admitted": sum(
                1
                for line in journal.read_text().splitlines()
                if json.loads(line)["round"] == 0
            ),
            "new_admissions_this_start": len(historical),
            "strict_promotions": sum(
                json.loads(line)["kind"] == "promotion"
                for line in journal.read_text().splitlines()
                if json.loads(line)["round"] == 0
            ),
            "note": "ordered ratchet against current incumbent; six beat frozen original best in later audit",
            "sealed_meta_test_read": False,
            "outer_rows_in_packet": 0,
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "admitted": len(historical),
                    "promotions": sum(e["kind"] == "promotion" for e in historical),
                }
            )
        )
        return
    # Use pinned CPU-exact source, which imports no GPU Stage-1A/slate extension.
    source = args.input / "cpu_exact_fused.py"
    if (
        sha(source)
        != "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
    ):
        raise ValueError("CPU-exact fused source drift")
    fs = importlib.util.spec_from_file_location("cpu_exact_fused", source)
    fused = importlib.util.module_from_spec(fs)
    fs.loader.exec_module(fused)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    torch.cuda.set_per_process_memory_fraction(0.90)
    headroom()
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for name, p in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            p.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    before = guard.snapshot_dope_weights(net)
    if (
        before.composite_sha256
        != "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
    ):
        raise RuntimeError("frozen DOPE source differs from campaign pin")
    counts = guard.get_trainable_param_counts_by_submodule(net)
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    guard.assert_optimizer_excludes_dope(opt, net)
    start_round = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state.get("admission") == "rejected":
            raise ValueError("cannot resume a rejected adapter proposal")
        net.load_state_dict(state["state_dict"], strict=True)
        opt.load_state_dict(state["optimizer"])
        guard.verify_dope_frozen_unchanged(net, before)
        start_round = int(state["round"]) + 1
    token = {}
    mask = {}
    landmark = {}
    ws = FusedLandmarkWorkspace(num_threads=4)
    tprep = time.monotonic()
    for prep_idx, (cid, (x, y)) in enumerate(tables.items()):
        # Four worker threads remain fixed; check host headroom in bounded groups
        # instead of imposing a 250ms sampling sleep on every tiny context.
        if not full or prep_idx % 16 == 0:
            headroom()
        yn = (
            (y.astype(np.float64) - np.mean(y.astype(np.float64)))
            / max(float(np.std(y.astype(np.float64))), 1e-8)
        ).astype(np.float32)
        v, col, _ = ws.fit_transform_and_extract_bundle_high(
            landmark_float32(x), np.ascontiguousarray(yn)
        )
        cl = np.zeros((64, 17), np.float32)
        n = min(len(col), 64)
        cl[:n, : min(col.shape[1], 17)] = col[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(v, device=device).reshape(1, 256),
                torch.as_tensor(cl, device=device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, msk, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        token[cid] = tok[0].detach()
        mask[cid] = msk[0].detach()
        landmark[cid] = lm[0].cpu().numpy()
    prep_wall = time.monotonic() - tprep
    evaluator.EVAL_SCRATCH_ROOT = args.output / "rust_eval"
    if sha(evaluator.WORKER_BIN) != evaluator.WORKER_SHA256:
        raise ValueError("evaluator drift")
    bank = [
        r
        for r in json.loads((args.input / "champions.json").read_text())
        if r["context_sha256"] in train
    ]

    for parent in bank:
        if "feat412" not in parent:
            feat, proj = evaluator.featurize_dsl_412(parent["dsl"])
            parent.update(
                feat412=feat.tolist(),
                proj16=proj.tolist(),
                landmark_summary=landmark[parent["context_sha256"]][:12].tolist(),
            )

    # Recovered promotions also become mutation parents after a process restart.
    known_bank = {(r["context_sha256"], r["program_sha256"]) for r in bank}
    for line in journal.read_text().splitlines():
        ev = json.loads(line)
        key = (ev["context_sha256"], ev["program_sha256"])
        if key not in known_bank:
            bank.append(
                {
                    "context_sha256": key[0],
                    "family_id": ev["family_id"],
                    "dl_split": "DL-Train",
                    "dsl": ev["dsl"],
                    "program_sha256": key[1],
                    "feat412": ev["feat412"],
                    "proj16": ev["proj16"],
                    "landmark_summary": landmark[key[0]][:12].tolist(),
                }
            )
            known_bank.add(key)

    device_pool_cache = DevicePoolCache(device)

    def predict(cids, selections, packed=None):
        b = len(cids)
        m = len(selections[0])
        if packed is None:
            prog = np.stack([pools[c]["program"][s] for c, s in zip(cids, selections)])
            proj = np.stack([pools[c]["proj16"][s] for c, s in zip(cids, selections)])
            prog_tensor = torch.as_tensor(prog, device=device)
            proj_tensor = torch.as_tensor(proj, device=device)
            history_f = torch.as_tensor(
                np.stack(
                    [
                        pools[c]["fitness"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
            history_gate = torch.as_tensor(
                np.stack(
                    [
                        pools[c]["gate"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            prog_tensor = packed["program"]
            proj_tensor = packed["proj16"]
            history_f = packed["fitness"][:, : min(4, m)].float()
            history_gate = packed["gate"][:, : min(4, m)].float()
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584), device=device),
            prog_412=prog_tensor,
            proj_16=proj_tensor,
            base_prior_cost=torch.zeros((b, m), device=device),
            scale=torch.full((b,), 0.4208, device=device),
            sos_prior=torch.zeros((b, m, 2), device=device),
            mm_tokens_10x32=torch.stack([token[c] for c in cids]),
            mm_mask_10=torch.stack([mask[c] for c in cids]),
            revealed_indices=torch.arange(min(4, m), device=device)
            .reshape(1, -1)
            .expand(b, -1),
            revealed_raw_f=history_f,
            revealed_gate=history_gate,
        )["total_cost"].float()

    def utility(cids=None, role="dev"):
        cids = dev if cids is None else cids
        net.eval()
        values, fitness, gates, choices = [], [], [], []
        with torch.no_grad():
            for c in cids:
                p = pools[c]
                pred = predict([c], [np.arange(len(p["pids"]))])[0].cpu().numpy()
                i = int(np.argmin(pred))
                fitness.append(float(p["fitness"][i]))
                gates.append(bool(p["gate"][i]))
                choices.append(p["pids"][i])
                if not full:
                    values.append(-fitness[-1] if gates[-1] else -2.0)
        if reporter:
            return reporter.utility(cids, fitness, gates, choices, role)
        return {
            "mean": float(np.mean(values)),
            "per_context": values,
            "selected_programs": choices,
            "scope": "family-disjoint development subset of DL-Train; not sealed generalization",
        }

    # Exact parity on real, repeated-context slates before enabling the fast path.
    parity_cids = train[:4] + train[:1]
    parity_rng = np.random.default_rng(args.seed)
    parity_selections = [
        parity_rng.integers(0, len(pools[c]["pids"]), size=args.slate_size)
        for c in parity_cids
    ]
    parity_pack = device_pool_cache.pack(pools, parity_cids, parity_selections)
    parity_fields = {}
    for field in device_pool_cache.FIELDS:
        expected = torch.as_tensor(
            np.stack(
                [pools[c][field][idx] for c, idx in zip(parity_cids, parity_selections)]
            ),
            device=device,
        )
        parity_fields[field] = bool(torch.equal(expected, parity_pack[field]))
    net.eval()
    with torch.no_grad():
        legacy_cost = predict(parity_cids, parity_selections)
        cached_cost = predict(parity_cids, parity_selections, parity_pack)
    parity = {
        "fields_bit_identical": parity_fields,
        "predicted_costs_bit_identical": bool(torch.equal(legacy_cost, cached_cost)),
        "max_abs_cost_delta": float((legacy_cost - cached_cost).abs().max()),
        "contexts": parity_cids,
    }
    if not all(parity_fields.values()) or not parity["predicted_costs_bit_identical"]:
        raise RuntimeError("device pool cache changed legacy inputs/predicted costs")
    write_json(
        args.output / f"device_pool_cache_parity_round{start_round:02d}.json", parity
    )
    del parity_pack, expected, legacy_cost, cached_cost

    resumed_stats = utility()
    eval_contract = {
        "protocol": "fixed-dev-cost-argmin-negative-fitness/v1",
        "selected_packet_sha256": sha(args.input / "selected.json"),
        "dope_sha256": before.composite_sha256,
        "source_sha256": sha(source),
        "model_source_sha256": sha(MICRO / "python/model.py"),
        "adapter_source_sha256": sha(MICRO / "python/phase12_multimodal_model.py"),
        "contexts": [
            {
                "cid": c,
                "pool_sha256": pools[c]["rec"]["pool_sha256"],
                "training_table_sha256": pools[c]["rec"]["train_only_jtt1_sha256"],
            }
            for c in dev
        ],
    }
    evaluation_identity = hashlib.sha256(
        json.dumps(eval_contract, sort_keys=True).encode()
    ).hexdigest()
    historical_best = None
    if full and args.best_checkpoint:
        raise ValueError("old-cohort best cannot enter full-cohort admission")
    if args.best_checkpoint:
        best_path = args.best_checkpoint
        historical_best = json.loads(best_path.with_suffix(".json").read_text())
    else:
        history = [
            json.loads(path.read_text()) for path in args.output.glob("round-*.json")
        ]
        eligible = [
            row
            for row in history
            if (args.output / f"round-{row['round']:02d}.pt").exists()
        ]
        # Most recent tied best permits continued training without lowering dev utility.
        historical_best = (
            max(eligible, key=lambda row: (row["dev_utility"]["mean"], row["round"]))
            if eligible
            else None
        )
        best_path = (
            args.output / f"round-{historical_best['round']:02d}.pt"
            if historical_best
            else args.resume
        )
    pointer_path = args.output / "best_dev_checkpoint.json"
    if pointer_path.exists():
        prior_best = json.loads(pointer_path.read_text())
        if prior_best["evaluation_identity"] != evaluation_identity:
            raise ValueError("persisted best-dev evaluation identity changed")
        prior_path = Path(prior_best["best_checkpoint"])
        if sha(prior_path) != prior_best["checkpoint_sha256"]:
            raise ValueError("persisted best-dev checkpoint hash mismatch")
        # This identity/hash-validated pointer is the committed admission state.
        # Scientific round receipts are observations; older writers may differ
        # by a JSON parsing ULP and must never supersede the committed pointer.
        historical_best = {
            "round": prior_best["best_round"],
            "dev_utility": prior_best["utility"],
        }
        best_path = prior_path
    if historical_best:
        best_state = torch.load(best_path, map_location=device, weights_only=True)
        net.load_state_dict(best_state["state_dict"], strict=True)
        opt.load_state_dict(best_state["optimizer"])
        guard.verify_dope_frozen_unchanged(net, before)
        initial = utility()
        if initial != historical_best["dev_utility"]:
            raise ValueError(
                "historical best does not replay on the current fixed dev evaluator"
            )
        best_round = int(historical_best["round"])
    else:
        initial = resumed_stats
        best_round = start_round - 1
    incumbent = DevIncumbent(
        net,
        opt,
        initial,
        evaluation_identity,
        str(best_path or "initial_model"),
        best_round,
        strict_admission=args.strict_adapter_admission,
        min_gain=args.adapter_min_gain,
    )
    best_metadata = {
        "evaluation_identity": evaluation_identity,
        "evaluation_contract": eval_contract,
        "best_checkpoint": incumbent.best_checkpoint,
        "best_round": incumbent.best_round,
        "best_utility": incumbent.best_utility,
        "utility": incumbent.best_stats,
        "checkpoint_sha256": sha(best_path) if best_path else None,
    }
    if incumbent.strict_admission:
        best_metadata["adapter_admission_policy"] = "strict-margin/v1"
        best_metadata["adapter_min_gain"] = incumbent.min_gain
    write_json(args.output / "best_dev_checkpoint.json", best_metadata)
    if resumed_stats["mean"] < initial["mean"]:
        with (args.output / "adapter_admissions.jsonl").open("a") as admission_log:
            admission_log.write(
                json.dumps(
                    {
                        "action": "startup_restore_best",
                        "round": start_round,
                        "resumed_utility": resumed_stats["mean"],
                        "best_utility": initial["mean"],
                        "best_checkpoint": str(best_path),
                        "evaluation_identity": evaluation_identity,
                        "timestamp": time.time(),
                    }
                )
                + "\n"
            )
    if full:
        margin_policy = {"version": "disabled-DL-Val-never-gradient/v1", "anchors": []}
        constraint_functions = []
    else:
        # Preserve only uncertain incumbent rankings, using prediction margins, not
        # held-out candidate utilities as gradient targets. Raw-argmin evaluation stays unchanged.
        constraint_path = args.output / "adapter_margin_constraints.json"
        if constraint_path.exists():
            margin_policy = json.loads(constraint_path.read_text())
            if margin_policy["evaluation_identity"] != evaluation_identity:
                raise ValueError("margin constraint evaluation identity changed")
            if (
                sha(margin_policy["anchor_checkpoint"])
                != margin_policy["anchor_checkpoint_sha256"]
            ):
                raise ValueError("margin anchor checkpoint changed")
        elif historical_best and best_state.get("margin_projection"):
            margin_policy = best_state["margin_projection"]
            if margin_policy["evaluation_identity"] != evaluation_identity:
                raise ValueError("checkpoint margin constraint identity changed")
            write_json(constraint_path, margin_policy)
        else:
            anchors = []
            with torch.no_grad():
                for c, pid in zip(dev, initial["selected_programs"]):
                    costs = predict([c], [np.arange(len(pools[c]["pids"]))])[0]
                    ref = pools[c]["pids"].index(pid)
                    others = torch.cat([costs[:ref], costs[ref + 1 :]])
                    gap = float(others.min() - costs[ref])
                    if 0 <= gap < 1e-3:
                        anchors.append(
                            {
                                "context_sha256": c,
                                "reference_program": pid,
                                "initial_margin": gap,
                                "minimum_margin": 0.5 * gap,
                            }
                        )
            margin_policy = {
                "version": "projected-low-margin-ranking/v1",
                "evaluation_identity": evaluation_identity,
                "anchor_checkpoint": str(best_path),
                "anchor_checkpoint_sha256": sha(best_path),
                "activation_margin": 1e-3,
                "retained_fraction": 0.5,
                "max_passes_per_step": 2,
                "anchors": anchors,
                "scope": "development-feature prediction-stability tuning; no candidate utility gradient targets; not independent validation",
            }
            write_json(constraint_path, margin_policy)
        if margin_policy["version"] != "projected-low-margin-ranking/v1":
            raise ValueError("unknown margin constraint policy")
        if (
            sha(margin_policy["anchor_checkpoint"])
            != margin_policy["anchor_checkpoint_sha256"]
        ):
            raise ValueError("margin anchor checkpoint changed")
        constraint_functions = []
        for anchor in margin_policy["anchors"]:
            cid = anchor["context_sha256"]
            if cid not in dev:
                raise ValueError(
                    "margin anchor is outside the unchanged development cohort"
                )
            ref = pools[cid]["pids"].index(anchor["reference_program"])

            def margin_fn(cid=cid, ref=ref):
                values = predict([cid], [np.arange(len(pools[cid]["pids"]))])[0]
                return torch.cat([values[:ref], values[ref + 1 :]]).min() - values[ref]

            constraint_functions.append((cid, margin_fn, anchor["minimum_margin"]))

    manifest = {
        "node": socket.gethostname(),
        "source_sha": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "command": sys.argv,
        "seed": args.seed,
        "train_contexts": train,
        "dev_contexts": dev,
        "dope_sha256": before.composite_sha256,
        "parameter_counts": counts,
        "path": "cpu_exact_landmarks+frozen_dope+GPU_adapter",
        "phase2_kernel_used": False,
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "task_counts": {
            "train_regression": len(train),
            "val_regression": len(dev),
            "train_binary": 0,
            "val_binary": 0,
        }
        if full
        else None,
        "landmark_preprocessing": "float64 raw tables; only out-of-f32-range feature columns scaled by finite max before frozen landmark standardization; raw evaluator bytes and labels unchanged",
        "protocol_transition": "fresh DL-Val91 admission baseline; prior DL-Train8 development score is incomparable"
        if full
        else None,
        "campaign_prior_outputs": [str(p) for p in args.campaign_prior_output],
        "prep_wall_seconds": prep_wall,
        "initial_dev": initial,
        "resumed_dev": resumed_stats,
        "best_dev": best_metadata,
        "margin_projection": margin_policy,
        "input_hashes": {
            p.name: sha(p)
            for p in (
                args.input / "selected.json",
                args.input / "winners.json",
                args.input / "champions.json",
                args.input / "source_manifest.json",
                source,
            )
        },
        "external_sources": {
            str(p): sha(p)
            for p in [
                Path(evaluator.__file__),
                MICRO / "python/model.py",
                MICRO / "python/phase12_multimodal_model.py",
                MICRO / "assets/models/dope_mlc_v3_multitask_best.pt",
            ]
        },
        "budget": {
            "rounds": args.rounds,
            "steps_per_round": args.steps,
            "batch_size": args.batch_size,
            "slate_size": args.slate_size,
            "cpu_cap_pct": 85,
            "gpu_memory_fraction_cap": 0.90,
            "continuous": args.continuous,
            "max_run_bytes": 10 * 1024**3,
            "min_free_bytes": 50 * 1024**3,
        },
    }
    manifest["executed_source_hashes"] = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            HERE / "automatic_promotion.py",
            HERE / "dev_incumbent.py",
            HERE / "device_pool_cache.py",
        ]
    }
    if full:
        native_path = Path(sys.modules["jope._telemetry"].__file__)
        manifest["native_telemetry_extension"] = {
            "path": str(native_path),
            "sha256": sha(native_path),
        }
    preregistration = args.output / "gp_exploration_preregistration.json"
    manifest["gp_exploration_preregistration"] = (
        {"path": str(preregistration), "sha256": sha(preregistration)}
        if preregistration.exists()
        else None
    )
    manifest["resume_checkpoint"] = (
        {"path": str(args.resume), "sha256": sha(args.resume)} if args.resume else None
    )
    write_json(args.output / f"run_manifest_round{start_round:02d}.json", manifest)
    round_indices = (
        itertools.count(start_round)
        if args.continuous
        else range(start_round, start_round + args.rounds)
    )
    proposal_history = [
        json.loads(path.read_text()) for path in args.output.glob("round-*.json")
    ]
    admission_path = args.output / "adapter_admissions.jsonl"
    admission_history = (
        [json.loads(line) for line in admission_path.read_text().splitlines()]
        if admission_path.exists()
        else []
    )
    novelty = CandidateNovelty(
        evaluation_identity, [*proposal_history, *admission_history]
    )
    release_gpu_round(lock)  # startup qualification is complete
    for rnd in round_indices:
        if reporter:
            reporter.begin()
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (args.output / f"round-{rnd:02d}.json").exists():
            raise ValueError(
                "completed round exists; use a new run with explicit checkpoint resume"
            )
        retention = None
        used_bytes = sum(
            p.stat().st_size
            for root in [args.output, *args.campaign_prior_output]
            for p in root.rglob("*")
            if p.is_file()
        )
        if rnd == start_round:
            retention_started = time.monotonic()
            protected = [
                p.resolve()
                for p in [args.resume, args.best_checkpoint]
                if p is not None
            ]
            for name in [
                incumbent.best_checkpoint,
                margin_policy.get("anchor_checkpoint"),
            ]:
                if name and name != "initial_model":
                    protected.append(Path(name).resolve())
            protected.extend(
                p if p.is_absolute() else args.output.resolve() / p
                for p in args.retain_checkpoint
            )
            try:
                retention = retain_checkpoint_payloads(
                    args.output,
                    args.checkpoint_archive_dir,
                    protected,
                    keep_last=args.checkpoint_keep_last,
                    guarded_roots=args.campaign_prior_output,
                    s3_prefix=args.retention_s3_prefix,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
                tarfile.TarError,
            ) as error:
                # An unsafe/unavailable retention route never authorizes deletion.
                # The original cap still stops the worker if reclamation was needed.
                retention = {"pruned_files": None, "deferred": str(error)}
            retention["wall_seconds"] = time.monotonic() - retention_started
            print(
                json.dumps({"checkpoint_retention": retention, "before_round": rnd}),
                flush=True,
            )
            used_bytes = sum(
                p.stat().st_size
                for root in [args.output, *args.campaign_prior_output]
                for p in root.rglob("*")
                if p.is_file()
            )
        if (
            used_bytes > 10 * 1024**3
            or shutil.disk_usage(args.output).free < 50 * 1024**3
        ):
            raise RuntimeError(
                "storage guard: retain receipts; stop before exceeding 10 GiB or 50 GiB free reserve"
            )
        plan = proposal_plan(args.seed, rnd, proposal_history, args.steps)
        if full:
            plan["steps"] = args.steps
            plan["lr_factor"] = 1.0
            plan["reason"] = "full_cohort_seeded_adapter_warm_restart"
        if constraint_functions:
            plan["lr_factor"] = 1.0
            plan["reason"] = "projected_score_margin_step_not_further_lr_decay"
            plan["margin_projection"] = True
        random.seed(plan["round_seed"])
        np.random.seed(plan["round_seed"] % (2**32))
        torch.manual_seed(plan["round_seed"])
        for group in opt.param_groups:
            group["lr"] = max(plan["minimum_lr"], group["lr"] * plan["lr_factor"])
        rotation = plan["training_rotation"] % len(train)
        round_train = train[rotation:] + train[:rotation]
        # Return unused cached segments between rounds to limit fragmentation.
        wait_for_gpu_headroom(device)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        phase = {
            k: 0.0
            for k in (
                "headroom_wait",
                "slate_assembly",
                "forward_and_targets",
                "backward_and_step",
                "dev_gate",
                "gp",
            )
        }
        grad_norms = []
        margin_projections = []
        batch_profile = {
            "device_pack_seconds": 0.0,
            "model_forward_target_seconds": 0.0,
            "cpu_index_bytes": 0,
            "avoided_cpu_feature_slate_bytes": 0,
        }
        transfers_before = device_pool_cache.transfers
        projection_failed = False
        executed_steps = 0
        training_batches = []
        optimizer_step_before = max(
            (int(v["step"]) for v in opt.state.values() if "step" in v), default=0
        )
        effective_lr = [g["lr"] for g in opt.param_groups]
        net.train()
        snap = {
            n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad
        }
        if full:
            plan["warm_restart"] = warm_restart_adapter(
                net, opt, plan["round_seed"], rnd
            )
            effective_lr = [g["lr"] for g in opt.param_groups]
        for step in range(plan["steps"]):
            phase_start = time.monotonic()
            headroom()
            phase["headroom_wait"] += time.monotonic() - phase_start
            phase_start = time.monotonic()
            if reporter:
                reporter.begin_pack()
            step_seed = batch_seed(args.seed, rnd, step)
            cids = (
                reporter.training_order(step_seed, args.batch_size, False)
                if reporter
                else random.Random(step_seed).choices(round_train, k=args.batch_size)
            )
            training_batches.append(
                {
                    "step": step,
                    "seed": step_seed,
                    "contexts": len(cids),
                    "order_sha256": hashlib.sha256(
                        json.dumps(cids).encode()
                    ).hexdigest(),
                }
            )
            selections = []
            for c in cids:
                selections.append(training_indices(pools[c], args.slate_size))
            phase["slate_assembly"] += time.monotonic() - phase_start
            phase_start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            pack_start = time.monotonic()
            packed = device_pool_cache.pack(pools, cids, selections)
            torch.cuda.synchronize()
            if reporter:
                reporter.packed(cids, packed["host_payload_bytes"])
            batch_profile["device_pack_seconds"] += time.monotonic() - pack_start
            batch_profile["cpu_index_bytes"] += packed["cpu_index_bytes"]
            batch_profile["avoided_cpu_feature_slate_bytes"] += packed[
                "avoided_cpu_feature_slate_bytes"
            ]
            if full:
                device_pool_cache.release_resident_pools()
                torch.cuda.empty_cache()
            model_start = time.monotonic()
            pred = predict(cids, selections, packed)
            target = packed["fitness"].float()
            gates = packed["gate"]
            target = torch.where(gates, target, torch.full_like(target, 2.0))
            loss = F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, dim=-1),
                F.softmax(-target / 0.2, dim=-1),
                reduction="batchmean",
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            torch.cuda.synchronize()
            phase["forward_and_targets"] += time.monotonic() - phase_start
            batch_profile["model_forward_target_seconds"] += (
                time.monotonic() - model_start
            )
            phase_start = time.monotonic()
            loss.backward()
            grad_norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
                )
            )
            opt.step()
            executed_steps += 1
            if constraint_functions:
                net.eval()
                projected = project_score_margins(
                    [p for p in net.parameters() if p.requires_grad],
                    constraint_functions,
                    max_passes=margin_policy["max_passes_per_step"],
                )
                margin_projections.append(projected)
                if not projected["satisfied"]:
                    projection_failed = True
                    incumbent.restore(net, opt)
                net.train()
            losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            phase["backward_and_step"] += time.monotonic() - phase_start
            del packed  # release the large slate before the next gather/GP phase
            if projection_failed:
                break
        # The final autograd result keeps references to large input/signature
        # tensors even after backward. Drop them before the next round guard.
        del pred, target, gates, loss
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_wall = time.monotonic() - start
        guard.verify_dope_frozen_unchanged(net, before)
        changed = sum(
            not torch.equal(snap[n], p) for n, p in net.named_parameters() if n in snap
        )
        if not changed and not projection_failed:
            raise RuntimeError("no parameter update")
        update_l2 = float(
            torch.sqrt(
                sum(
                    torch.sum((p.detach() - snap[n]) ** 2)
                    for n, p in net.named_parameters()
                    if n in snap
                )
            )
        )
        optimizer_step_candidate = max(
            (int(v["step"]) for v in opt.state.values() if "step" in v), default=0
        )
        phase_start = time.monotonic()
        candidate_hash = hashlib.sha256()
        for name, parameter in sorted(net.named_parameters()):
            if parameter.requires_grad:
                candidate_hash.update(name.encode())
                candidate_hash.update(
                    parameter.detach().cpu().contiguous().numpy().tobytes()
                )
        candidate_parameter_sha256 = candidate_hash.hexdigest()
        try:
            candidate_dev, candidate_train = novelty.evaluate(
                candidate_parameter_sha256,
                lambda: (utility(), utility(train, "train") if full else None),
            )
        except RejectedCandidateError as error:
            incumbent.restore(net, opt)
            guard.verify_dope_frozen_unchanged(net, before)
            write_json(
                args.output / f"candidate_refusal_round-{rnd}.json",
                {
                    "round": rnd,
                    "candidate_parameter_sha256": candidate_parameter_sha256,
                    "evaluation_identity": evaluation_identity,
                    "training_batches": training_batches,
                    "reason": str(error),
                    "candidate_evaluated": False,
                    "completed_round": False,
                    "incumbent_restored": True,
                },
            )
            raise
        if reporter:
            reporter.scores(
                dev,
                candidate_dev["per_context"],
                "negative_train_cv_fitness_gate_penalty",
                True,
                "flywheel-adapter-candidate",
                candidate_parameter_sha256,
                rnd,
            )
            reporter.scores(
                train,
                candidate_train["per_context"],
                "negative_train_cv_fitness_gate_penalty",
                True,
                "flywheel-adapter-candidate",
                candidate_parameter_sha256,
                rnd,
            )
        candidate_rejected = (
            not incumbent.accepts(candidate_dev, evaluation_identity)
            if incumbent.strict_admission
            else candidate_dev["mean"] < incumbent.best_utility
        )
        if candidate_rejected:
            rejected = args.output / "rejected" / f"round-{rnd:02d}.pt"
            rejected.parent.mkdir(exist_ok=True)
            rejected_tmp = rejected.with_suffix(".pt.new")
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "round": rnd,
                    "admission": "rejected",
                    "dev_utility": candidate_dev,
                    "dope_sha256": before.composite_sha256,
                    **(
                        {
                            "adapter_admission_policy": "strict-margin/v1",
                            "adapter_min_gain": incumbent.min_gain,
                        }
                        if incumbent.strict_admission
                        else {}
                    ),
                },
                rejected_tmp,
            )
            rejected_tmp.replace(rejected)
        if projection_failed:
            adapter_admission = {
                "round": rnd,
                "evaluation_identity": evaluation_identity,
                "candidate_utility": candidate_dev["mean"],
                "previous_best_utility": incumbent.best_utility,
                "accepted": False,
                "action": "projection_failed_keep_incumbent_and_continue_GP",
                "best_utility": incumbent.best_utility,
                "best_round": incumbent.best_round,
                "best_checkpoint": incumbent.best_checkpoint,
            }
            if incumbent.strict_admission:
                adapter_admission["admission_policy"] = "strict-margin/v1"
                adapter_admission["min_gain"] = incumbent.min_gain
        else:
            adapter_admission = incumbent.decide(
                net,
                opt,
                candidate_dev,
                evaluation_identity,
                str(args.output / f"round-{rnd:02d}.pt"),
                rnd,
            )
        guard.verify_dope_frozen_unchanged(net, before)
        dev_stats = utility()
        if dev_stats != incumbent.best_stats:
            raise RuntimeError("incumbent restore did not reproduce best-dev selection")
        adapter_admission["timestamp"] = time.time()
        adapter_admission["candidate_parameter_sha256"] = candidate_parameter_sha256
        with (args.output / "adapter_admissions.jsonl").open("a") as admission_log:
            admission_log.write(json.dumps(adapter_admission) + "\n")
            admission_log.flush()
            os.fsync(admission_log.fileno())
        novelty.record(adapter_admission)
        phase["dev_gate"] = time.monotonic() - phase_start
        gp_start = time.monotonic()
        seen_before_round = {pid for c in train for pid in pools[c]["pids"]}
        gp_proposal_changes = []
        generated = []
        tree_records = []
        events = []
        passed = 0
        failed = 0
        token_count = 0
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        evaluator.EVAL_SCRATCH_ROOT = args.output / f"rust_eval_round{rnd:02d}"
        gp_contexts = (
            reporter.training_order(plan["round_seed"], 16, True) if reporter else train
        )
        for i, c in enumerate(gp_contexts):
            headroom()
            p = pools[c]
            parents = [r for r in bank if r["context_sha256"] == c]
            parent_offset = (rnd + i) % len(parents) if parents else 0
            parents = parents[parent_offset:] + parents[:parent_offset]
            proposals = evaluator.propose_flywheel_candidates(
                parents,
                landmark[c],
                bank,
                p["rec"]["family_id"],
                n_proposals=plan["gp_generated_per_context"],
                seed=plan["round_seed"] + i,
            )
            generated_count = len(proposals)
            known_pids = set(p["pids"])
            for proposal in proposals:
                if proposal["program_sha256"] in known_pids:
                    previous_index = p["pids"].index(proposal["program_sha256"])
                    tree_records.append(
                        {
                            "dsl": proposal["dsl"],
                            "program_sha256": proposal["program_sha256"],
                            "context_sha256": c,
                            "round": rnd,
                            "status": "pool_duplicate",
                            "fitness": float(p["fitness"][previous_index]),
                            "cv_gate_passed": bool(p["gate"][previous_index]),
                            "score_source": "existing_train_pool",
                            "pool_accepted": False,
                        }
                    )
            proposals = [
                candidate
                for candidate in proposals
                if candidate["program_sha256"] not in known_pids
            ]
            # Fused head ranks actual proposed ASTs before exact Rust CV scoring.
            if proposals:
                f = np.stack(
                    [evaluator.featurize_dsl_412(x["dsl"])[0] for x in proposals]
                )
                pr = np.stack(
                    [evaluator.featurize_dsl_412(x["dsl"])[1] for x in proposals]
                )
                temporary = dict(p)
                temporary["program"] = np.concatenate([p["program"][:4], f])
                temporary["proj16"] = np.concatenate([p["proj16"][:4], pr])
                pools[c] = temporary
                with torch.no_grad():
                    rank = (
                        predict([c], [np.arange(4 + len(proposals))])[0, 4:]
                        .cpu()
                        .numpy()
                    )
                pools[c] = p
                selected_indices = gp_selection_indices(
                    rank, plan["round_seed"] + i, plan["gp_exploration"]
                )
                chosen = [proposals[j] for j in selected_indices]
            else:
                chosen = []
                rank = []
            gp_proposal_changes.append(
                {
                    "context_sha256": c,
                    "parent_rotation": parent_offset,
                    "generated": generated_count,
                    "novel_after_pool_dedup": len(proposals),
                    "exploration": plan["gp_exploration"],
                    "selected_programs": [v["program_sha256"] for v in chosen],
                }
            )
            gz = args.output / "train_inputs" / f"{c}.jtt1.gz"
            gz.parent.mkdir(exist_ok=True)
            gz.write_bytes(
                gzip.compress(
                    training_cv_envelope(
                        (args.input / "tables" / f"{c}.jtt1").read_bytes()
                    ),
                    mtime=0,
                )
            )
            chosen_ids = {v["program_sha256"] for v in chosen}
            for proposal, predicted_cost in zip(proposals, rank):
                if proposal["program_sha256"] not in chosen_ids:
                    tree_records.append(
                        {
                            "dsl": proposal["dsl"],
                            "program_sha256": proposal["program_sha256"],
                            "context_sha256": c,
                            "round": rnd,
                            "status": "not_selected",
                            "predicted_cost": float(predicted_cost),
                            "pool_accepted": False,
                        }
                    )
            results = evaluator.evaluate_gp_batch_rust(
                str(gz), chosen, run_outer_test=False, max_workers=2, timeout_s=25
            )
            for r in results:
                generated.append(r["program_sha256"])
                passed += int(r["status"] == "completed" and r["cv_gate_passed"])
                failed += int(r["status"] != "completed")
                token_count += len(tokenizer.encode(r["dsl"]))
                with (args.output / f"round-{rnd:02d}-scores.jsonl").open(
                    "a"
                ) as score_log:
                    score_log.write(
                        json.dumps(
                            {
                                k: v
                                for k, v in r.items()
                                if k not in ("feat412", "proj16")
                            }
                        )
                        + "\n"
                    )
                if reporter and r["status"] == "completed":
                    reporter.scores(
                        [c],
                        [float(r["fitness"])],
                        "train_cv_fitness",
                        False,
                        "gp-program",
                        r["program_sha256"],
                        rnd,
                    )
                r["context_sha256"] = c
                r["evaluation_scope"] = "full_train_cv"
                r["input_identity"] = {
                    "train_only_jtt1_sha256": p["rec"]["train_only_jtt1_sha256"],
                    "evaluator_binary_sha256": evaluator.WORKER_SHA256,
                    "protocol": "jope.s1-two-fold/v1",
                }
                ev = promote(p, r, journal, rnd)
                tree_records.append({**r, "round": rnd, "pool_accepted": bool(ev)})
                if ev:
                    events.append(ev)
                    bank.append(
                        {
                            "context_sha256": c,
                            "family_id": p["rec"]["family_id"],
                            "dl_split": "DL-Train",
                            "dsl": r["dsl"],
                            "program_sha256": r["program_sha256"],
                            "feat412": r["feat412"].tolist(),
                            "proj16": r["proj16"].tolist(),
                            "landmark_summary": landmark[c][:12].tolist(),
                        }
                    )
        phase["gp"] = time.monotonic() - gp_start
        guard.verify_dope_frozen_unchanged(net, before)
        checkpoint_tmp = args.output / f"round-{rnd:02d}.pt.new"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "optimizer": opt.state_dict(),
                "round": rnd,
                "dope_sha256": before.composite_sha256,
                "admission": "incumbent",
                "best_dev_round": incumbent.best_round,
                "evaluation_identity": evaluation_identity,
                "margin_projection": margin_policy,
                **(
                    {
                        "adapter_admission_policy": "strict-margin/v1",
                        "adapter_min_gain": incumbent.min_gain,
                    }
                    if incumbent.strict_admission
                    else {}
                ),
            },
            checkpoint_tmp,
        )
        with checkpoint_tmp.open("rb") as checkpoint_stream:
            os.fsync(checkpoint_stream.fileno())
        checkpoint_tmp.replace(args.output / f"round-{rnd:02d}.pt")
        best_metadata = {
            "evaluation_identity": evaluation_identity,
            "evaluation_contract": eval_contract,
            "best_checkpoint": incumbent.best_checkpoint,
            "best_round": incumbent.best_round,
            "best_utility": incumbent.best_utility,
            "utility": incumbent.best_stats,
            "checkpoint_sha256": sha(incumbent.best_checkpoint),
        }
        if incumbent.strict_admission:
            best_metadata["adapter_admission_policy"] = "strict-margin/v1"
            best_metadata["adapter_min_gain"] = incumbent.min_gain
        write_json(args.output / "best_dev_checkpoint.json", best_metadata)
        wall = time.monotonic() - start
        row = {
            "round": rnd,
            "node": socket.gethostname(),
            "path": manifest["path"],
            "optimizer_steps": executed_steps,
            "margin_projection_policy": margin_policy["version"],
            "margin_projections": margin_projections,
            "projection_failed": projection_failed,
            "proposal_plan": plan,
            "candidate_parameter_sha256": candidate_parameter_sha256,
            "checkpoint_retention": retention,
            "training_batch_rng": "round-step/v1",
            "training_batches": training_batches,
            "changed_tensors": changed,
            "trainable_params": counts["__global_total__"]["trainable_params"],
            "parameter_counts": counts,
            "dope_checksum_match": True,
            "dope_sha256": before.composite_sha256,
            "training_context_presentations": executed_steps * args.batch_size,
            "training_contexts_per_second": plan["steps"]
            * args.batch_size
            / train_wall,
            "round_contexts_per_second": len(train) / wall,
            "gp_dsl_tokens_per_second": None if full else token_count / wall,
            "tokens_count": token_count,
            "tokens_scope": "evaluated GP DSL tokens (cl100k_base) per full round wall second; not emitted target-language code",
            "dev_utility": dev_stats,
            "candidate_dev_utility": candidate_dev,
            "candidate_train_utility": candidate_train,
            "adapter_admission": adapter_admission,
            "evaluation_identity": evaluation_identity,
            "phase_wall_seconds": phase,
            "batch_profile": {
                **batch_profile,
                "pool_transfers_this_round": device_pool_cache.transfers
                - transfers_before,
            },
            "device_pool_cache_parity": parity,
            "learning_rate": effective_lr,
            "optimizer_step_before": optimizer_step_before,
            "optimizer_step_candidate": optimizer_step_candidate,
            "parameter_update_l2": update_l2,
            "gradient_norms_before_clip": grad_norms,
            "initial_dev_utility": initial,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "GP": {
                "proposal_changes": gp_proposal_changes,
                "evaluated": len(generated),
                "passing": passed,
                "failed": failed,
                "unique": len(set(generated)),
                "unique_fraction": len(set(generated)) / max(len(generated), 1),
                "novelty": len(set(generated) - seen_before_round)
                / max(len(set(generated)), 1),
                "collapse": bool(generated and len(set(generated)) == 1),
            },
            "promotions": len(events),
            "train_wall_seconds": train_wall,
            "wall_seconds": wall,
            "old_live_expansion_baseline_seconds": [183.26, 133.06, 211.24],
            "baseline_comparability": "different scope: old GP-only 16 contexts, now training + GP; no speedup claim",
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if args.phase3_gate_bundles:
            from phase3_round_runner import qualify_phase3_bundle

            bundle_path = args.phase3_gate_bundles / f"round-{rnd:02d}.json"
            try:
                phase3 = qualify_phase3_bundle(json.loads(bundle_path.read_text()))
            except Exception as exc:  # noqa: BLE001 -- invalid bundles remain unqualified
                phase3 = {
                    "schema": "jope.phase3.round_qualification.v1",
                    "scope": "keyed_subset",
                    "round": rnd,
                    "qualified": False,
                    "qualified_rounds_increment": 0,
                    "gate_error": f"{type(exc).__name__}: {exc}",
                }
            row["phase3_qualification"] = phase3
            write_json(args.output / f"round-{rnd:02d}.phase3.json", phase3)
        if reporter:
            row = reporter.finish(row)
        else:
            write_json(args.output / f"round-{rnd:02d}.json", row)
        try:
            shard = write_tree_shard(
                tree_records,
                args.output / "tree-shards" / f"trees-round-{rnd}.jsonl.zst",
            )
            shard["s3"] = publish_checkpoint_archive(
                Path(shard["path"]), args.retention_s3_prefix + "trees/"
            )
            protected = [
                p.resolve()
                for p in [args.resume, args.best_checkpoint]
                if p is not None
            ]
            protected += [
                Path(incumbent.best_checkpoint).resolve(),
                args.output.resolve() / f"round-{rnd:02d}.pt",
            ]
            protected += [
                p if p.is_absolute() else args.output.resolve() / p
                for p in args.retain_checkpoint
            ]
            if margin_policy.get("anchor_checkpoint"):
                protected.append(Path(margin_policy["anchor_checkpoint"]).resolve())
            cleanup = retain_checkpoint_payloads(
                args.output,
                args.checkpoint_archive_dir,
                protected,
                keep_last=args.checkpoint_keep_last,
                guarded_roots=args.campaign_prior_output,
                s3_prefix=args.retention_s3_prefix,
                scored_round=rnd,
            )
            cleanup["trees"] = shard
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            tarfile.TarError,
        ) as error:
            cleanup = {"pruned_files": None, "deferred": str(error)}
        cleanup["round"] = rnd
        cleanup["run_bytes_after"] = sum(
            p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        )
        cleanup["combined_bytes_after"] = sum(
            p.stat().st_size
            for root in [args.output, *args.campaign_prior_output]
            for p in root.rglob("*")
            if p.is_file()
        )
        write_json(args.output / f"retention-round-{rnd}.json", cleanup)
        print(json.dumps({"after_scored_round": rnd, "retention": cleanup}), flush=True)
        proposal_history.append(row)
        remaining = (
            "continuous"
            if args.continuous
            else str(start_round + args.rounds - 1 - rnd)
        )
        round_target = (
            "continuous" if args.continuous else str(start_round + args.rounds - 1)
        )
        eta = (
            f"next round approximately {wall:.0f}s"
            if args.continuous
            else f"remaining approximately {(start_round + args.rounds - 1 - rnd) * wall:.0f}s"
        )
        if not full:
            status = Path.home() / "jope_plan/ns/FLYWHEEL_GP_FUSED_STATUS.md"
            status.write_text(
                "# Fused flywheel status\n\n"
                + f"Node {row['node']}; completed round {rnd}/{round_target}.\nPath: {row['path']}.\nDOPE {before.composite_sha256}: unchanged.\nTrainable parameters: {row['trainable_params']}; changed tensors {changed}.\nTraining contexts/s {row['training_contexts_per_second']:.2f}; GP DSL tok/s {row['gp_dsl_tokens_per_second']:.2f} (not target-language code).\nDev utility {dev_stats['mean']:.6f}; candidate dev {candidate_dev['mean']:.6f}; adapter action {adapter_admission['action']}; best round {incumbent.best_round}. GP passing {passed}/{len(generated)}; promotions {len(events)}.\nWall {wall:.2f}s; training {train_wall:.2f}s; device packing {batch_profile['device_pack_seconds']:.2f}s; model/targets {batch_profile['model_forward_target_seconds']:.2f}s. Old GP-only baseline 133–211s, scope differs.\nReceipts: {args.output}.\nProposal seed {plan['round_seed']}; retry level {plan['retry_level']}; actual updates {plan['steps']}; GP exploration {plan['gp_exploration']}. Effective LR {effective_lr}; optimizer steps {optimizer_step_before} -> {optimizer_step_candidate}; update L2 {update_l2:.6g}. Remaining rounds: {remaining}; ETA: {eta}.\n"
            )
            findings = Path.home() / "jope_plan/ns/FLYWHEEL_GP_FUSED_FINDINGS.md"
            if findings.exists():
                with status.open("a") as status_stream:
                    status_stream.write("\n" + findings.read_text())
        if generated and failed == len(generated):
            raise RuntimeError(
                "all GP evaluations failed; round retained as unqualified, stopping continuation"
            )
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "round",
                        "optimizer_steps",
                        "changed_tensors",
                        "dope_checksum_match",
                        "promotions",
                        "wall_seconds",
                        "training_contexts_per_second",
                    )
                }
            ),
            flush=True,
        )
        release_gpu_round(lock)
        time.sleep(1)  # give waiting xbabe1 jobs an inter-round handoff window


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status = Path.home() / "jope_plan/ns/FLYWHEEL_GP_FUSED_STATUS.md"
        prior = status.read_text() if status.exists() else "# Fused flywheel status\n"
        status.write_text(
            prior
            + f"\nWorker stopped: {type(error).__name__}: {error}\nResume only from the last complete checkpoint; incomplete round outcome is unknown.\n"
        )
        raise
