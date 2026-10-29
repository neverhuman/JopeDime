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
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Import the CPU-chain dependencies explicitly, never a Phase-2 worktree.
MICRO = Path("/home/ubuntu/JopeDimeMicro")
sys.path.insert(0, str(MICRO / "python"))
import rust_evaluator as evaluator
from hpc.landmark_workspace import FusedLandmarkWorkspace
from phase12_multimodal_model import compute_semantic_struct_sig_24d

spec = importlib.util.spec_from_file_location(
    "freeze_guard", HERE.parents[1] / "python/src/jope/north_star/dope_freeze_guard.py"
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)
sys.path.insert(0, str(HERE))
from automatic_promotion import promote, replay, training_cv_envelope, training_indices
from dev_incumbent import DevIncumbent, gp_selection_indices, proposal_plan


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
    for role in ("train", "dev"):
        for rec in selected[role]:
            if rec["dl_split"] != "DL-Train" or rec["source_split"] not in (
                "train",
                "inner",
            ):
                raise ValueError("outside training closure")
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
            tables[cid] = (a[:, :-1].astype(np.float32), a[:, -1].astype(np.float32))
    return selected, pools, tables


def main():
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
    ap.add_argument("--best-checkpoint", type=Path)
    args = ap.parse_args()
    if min(args.rounds, args.steps, args.batch_size, args.slate_size) <= 0:
        raise ValueError("positive bounds required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(Path.home() / "jope_plan/qualification.lock", "a")  # noqa: SIM115 -- held for process lifetime
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    selected, pools, tables = load_packet(args.input)
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
    ws = FusedLandmarkWorkspace()
    tprep = time.monotonic()
    for cid, (x, y) in tables.items():
        headroom()
        yn = (
            (y.astype(np.float64) - np.mean(y.astype(np.float64)))
            / max(float(np.std(y.astype(np.float64))), 1e-8)
        ).astype(np.float32)
        v, col, _ = ws.fit_transform_and_extract_bundle_high(
            np.ascontiguousarray(x), np.ascontiguousarray(yn)
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

    def predict(cids, selections):
        b = len(cids)
        m = len(selections[0])
        prog = np.stack([pools[c]["program"][s] for c, s in zip(cids, selections)])
        proj = np.stack([pools[c]["proj16"][s] for c, s in zip(cids, selections)])
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584), device=device),
            prog_412=torch.as_tensor(prog, device=device),
            proj_16=torch.as_tensor(proj, device=device),
            base_prior_cost=torch.zeros((b, m), device=device),
            scale=torch.full((b,), 0.4208, device=device),
            sos_prior=torch.zeros((b, m, 2), device=device),
            mm_tokens_10x32=torch.stack([token[c] for c in cids]),
            mm_mask_10=torch.stack([mask[c] for c in cids]),
            revealed_indices=torch.arange(min(4, m), device=device)
            .reshape(1, -1)
            .expand(b, -1),
            revealed_raw_f=torch.as_tensor(
                np.stack(
                    [
                        pools[c]["fitness"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            ),
            revealed_gate=torch.as_tensor(
                np.stack(
                    [
                        pools[c]["gate"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            ),
        )["total_cost"].float()

    def utility():
        net.eval()
        values = []
        choices = []
        with torch.no_grad():
            for c in dev:
                p = pools[c]
                s = np.arange(len(p["pids"]))
                pred = predict([c], [s])[0].cpu().numpy()
                i = int(np.argmin(pred))
                values.append(-float(p["fitness"][i]) if p["gate"][i] else -2.0)
                choices.append(p["pids"][i])
        return {
            "mean": float(np.mean(values)),
            "per_context": values,
            "selected_programs": choices,
            "scope": "family-disjoint development subset of DL-Train; not sealed generalization",
        }

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
        if (
            historical_best is None
            or prior_best["best_utility"] >= historical_best["dev_utility"]["mean"]
        ):
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
        "prep_wall_seconds": prep_wall,
        "initial_dev": initial,
        "resumed_dev": resumed_stats,
        "best_dev": best_metadata,
        "input_hashes": {
            p.name: sha(p)
            for p in (args.input / "selected.json", args.input / "winners.json", source)
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
        str(p): sha(p) for p in [Path(__file__), HERE / "automatic_promotion.py"]
    }
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
    for rnd in round_indices:
        used_bytes = sum(
            p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        )
        if (
            used_bytes > 10 * 1024**3
            or shutil.disk_usage(args.output).free < 50 * 1024**3
        ):
            raise RuntimeError(
                "storage guard: retain receipts; stop before exceeding 10 GiB or 50 GiB free reserve"
            )
        if (args.output / f"round-{rnd:02d}.json").exists():
            raise ValueError(
                "completed round exists; use a new run with explicit checkpoint resume"
            )
        plan = proposal_plan(args.seed, rnd, proposal_history, args.steps)
        random.seed(plan["round_seed"])
        np.random.seed(plan["round_seed"] % (2**32))
        torch.manual_seed(plan["round_seed"])
        for group in opt.param_groups:
            group["lr"] = max(plan["minimum_lr"], group["lr"] * plan["lr_factor"])
        rotation = plan["training_rotation"] % len(train)
        round_train = train[rotation:] + train[:rotation]
        # Return unused cached segments between rounds to limit fragmentation.
        torch.cuda.empty_cache()
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
        optimizer_step_before = max(
            (int(v["step"]) for v in opt.state.values() if "step" in v), default=0
        )
        effective_lr = [g["lr"] for g in opt.param_groups]
        net.train()
        snap = {
            n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad
        }
        for step in range(plan["steps"]):
            phase_start = time.monotonic()
            headroom()
            phase["headroom_wait"] += time.monotonic() - phase_start
            phase_start = time.monotonic()
            cids = random.choices(round_train, k=args.batch_size)
            selections = []
            for c in cids:
                selections.append(training_indices(pools[c], args.slate_size))
            phase["slate_assembly"] += time.monotonic() - phase_start
            phase_start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            pred = predict(cids, selections)
            target = torch.as_tensor(
                np.stack([pools[c]["fitness"][s] for c, s in zip(cids, selections)]),
                dtype=torch.float32,
                device=device,
            )
            gates = torch.as_tensor(
                np.stack([pools[c]["gate"][s] for c, s in zip(cids, selections)]),
                device=device,
            )
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
            phase_start = time.monotonic()
            loss.backward()
            grad_norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
                )
            )
            opt.step()
            losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            phase["backward_and_step"] += time.monotonic() - phase_start
        torch.cuda.synchronize()
        train_wall = time.monotonic() - start
        guard.verify_dope_frozen_unchanged(net, before)
        changed = sum(
            not torch.equal(snap[n], p) for n, p in net.named_parameters() if n in snap
        )
        if not changed:
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
        candidate_dev = utility()
        candidate_hash = hashlib.sha256()
        for name, parameter in sorted(net.named_parameters()):
            if parameter.requires_grad:
                candidate_hash.update(name.encode())
                candidate_hash.update(
                    parameter.detach().cpu().contiguous().numpy().tobytes()
                )
        candidate_parameter_sha256 = candidate_hash.hexdigest()
        if candidate_dev["mean"] < incumbent.best_utility:
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
                },
                rejected_tmp,
            )
            rejected_tmp.replace(rejected)
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
        with (args.output / "adapter_admissions.jsonl").open("a") as admission_log:
            admission_log.write(json.dumps(adapter_admission) + "\n")
            admission_log.flush()
            os.fsync(admission_log.fileno())
        phase["dev_gate"] = time.monotonic() - phase_start
        gp_start = time.monotonic()
        seen_before_round = {pid for c in train for pid in pools[c]["pids"]}
        gp_proposal_changes = []
        generated = []
        events = []
        passed = 0
        failed = 0
        token_count = 0
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        evaluator.EVAL_SCRATCH_ROOT = args.output / f"rust_eval_round{rnd:02d}"
        for i, c in enumerate(train):
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
                r["context_sha256"] = c
                r["evaluation_scope"] = "full_train_cv"
                r["input_identity"] = {
                    "train_only_jtt1_sha256": p["rec"]["train_only_jtt1_sha256"],
                    "evaluator_binary_sha256": evaluator.WORKER_SHA256,
                    "protocol": "jope.s1-two-fold/v1",
                }
                ev = promote(p, r, journal, rnd)
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
            },
            checkpoint_tmp,
        )
        with checkpoint_tmp.open("rb") as checkpoint_stream:
            os.fsync(checkpoint_stream.fileno())
        checkpoint_tmp.replace(args.output / f"round-{rnd:02d}.pt")
        write_json(
            args.output / "best_dev_checkpoint.json",
            {
                "evaluation_identity": evaluation_identity,
                "evaluation_contract": eval_contract,
                "best_checkpoint": incumbent.best_checkpoint,
                "best_round": incumbent.best_round,
                "best_utility": incumbent.best_utility,
                "utility": incumbent.best_stats,
                "checkpoint_sha256": sha(incumbent.best_checkpoint),
            },
        )
        wall = time.monotonic() - start
        row = {
            "round": rnd,
            "node": socket.gethostname(),
            "path": manifest["path"],
            "optimizer_steps": plan["steps"],
            "proposal_plan": plan,
            "candidate_parameter_sha256": candidate_parameter_sha256,
            "changed_tensors": changed,
            "trainable_params": counts["__global_total__"]["trainable_params"],
            "parameter_counts": counts,
            "dope_checksum_match": True,
            "dope_sha256": before.composite_sha256,
            "training_context_presentations": plan["steps"] * args.batch_size,
            "training_contexts_per_second": plan["steps"]
            * args.batch_size
            / train_wall,
            "round_contexts_per_second": len(train) / wall,
            "tokens_per_second": token_count / wall,
            "tokens_count": token_count,
            "tokens_scope": "evaluated GP DSL tokens (cl100k_base) per full round wall second; not emitted target-language code",
            "dev_utility": dev_stats,
            "candidate_dev_utility": candidate_dev,
            "adapter_admission": adapter_admission,
            "evaluation_identity": evaluation_identity,
            "phase_wall_seconds": phase,
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
        write_json(args.output / f"round-{rnd:02d}.json", row)
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
        status = Path.home() / "jope_plan/ns/FLYWHEEL_GP_FUSED_STATUS.md"
        status.write_text(
            "# Fused flywheel status\n\n"
            + f"Node {row['node']}; completed round {rnd}/{round_target}.\nPath: {row['path']}.\nDOPE {before.composite_sha256}: unchanged.\nTrainable parameters: {row['trainable_params']}; changed tensors {changed}.\nTraining contexts/s {row['training_contexts_per_second']:.2f}; GP DSL tok/s {row['tokens_per_second']:.2f} (not target-language code).\nDev utility {dev_stats['mean']:.6f}; candidate dev {candidate_dev['mean']:.6f}; adapter action {adapter_admission['action']}; best round {incumbent.best_round}. GP passing {passed}/{len(generated)}; promotions {len(events)}.\nWall {wall:.2f}s; old GP-only baseline 133–211s, scope differs.\nReceipts: {args.output}.\nProposal seed {plan['round_seed']}; retry level {plan['retry_level']}; actual updates {plan['steps']}; GP exploration {plan['gp_exploration']}. Effective LR {effective_lr}; optimizer steps {optimizer_step_before} -> {optimizer_step_candidate}; update L2 {update_l2:.6g}. Remaining rounds: {remaining}; ETA: {eta}.\n"
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
