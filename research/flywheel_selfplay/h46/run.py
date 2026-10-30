#!/usr/bin/env python3
"""Offline H46 gradient-smoothing experiment on the approved checkpoint copy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
import run_fused_closed_loop as base  # noqa: E402
from losses import family_balanced_weights  # noqa: E402

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
BASE_UTILITY = 1.1375952947484103


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_adapter_state(path, net, base_sha, dope_sha):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }
    if sum(tensor.numel() for tensor in adapter.values()) != 29211:
        raise RuntimeError("refusing to save unexpected adapter tensor inventory")
    torch.save(
        {
            "base_checkpoint_sha256": base_sha,
            "dope_sha256": dope_sha,
            "trainable_params": 29211,
            "adapter_trainable_state": adapter,
        },
        temporary,
    )
    temporary.replace(path)


def process_matches(proc, fragment):
    try:
        command = " ".join(proc.info.get("cmdline") or [])
    except (base.psutil.NoSuchProcess, base.psutil.AccessDenied):
        return False
    return fragment in command


def coordinate_d_runs(input_dir, runs_root, checkpoint, device):
    script = Path(__file__).resolve()
    last_report = 0
    lock_busy_started = None
    while True:
        active_wave = False
        active_runner = False
        for proc in base.psutil.process_iter(["pid", "cmdline"]):
            if proc.info["pid"] == os.getpid():
                continue
            if process_matches(proc, "research/flywheel_selfplay/h46/run_wave.sh"):
                active_wave = True
            if process_matches(proc, "research/flywheel_selfplay/h46/run.py"):
                active_runner = True
        if not active_wave and not active_runner:
            break
        if active_wave and not active_runner:
            probe = subprocess.run(
                [
                    "flock",
                    "-n",
                    str(Path.home() / "jope_plan/qualification.lock"),
                    "-c",
                    "true",
                ],
                check=False,
            )
            if probe.returncode != 0:
                lock_busy_started = lock_busy_started or time.monotonic()
                if time.monotonic() - lock_busy_started >= 1200:
                    print(
                        "qualification.lock stayed busy for 20 minutes between runs; switching all comparisons to CPU",
                        flush=True,
                    )
                    for proc in base.psutil.process_iter(["pid", "cmdline"]):
                        if proc.info["pid"] != os.getpid() and process_matches(
                            proc, "research/flywheel_selfplay/h46/run_wave.sh"
                        ):
                            proc.send_signal(signal.SIGTERM)
                    cpu_env = os.environ.copy()
                    cpu_env["H46_DEVICE"] = "cpu"
                    cpu_env["H46_RUN_ROOT"] = str(
                        Path.home() / "jope-data/h46/runs/cpu"
                    )
                    return subprocess.run(
                        [str(script.parent / "run_wave.sh")],
                        cwd=ROOT,
                        env=cpu_env,
                        check=False,
                    ).returncode
            else:
                lock_busy_started = None
        else:
            lock_busy_started = None
        if time.monotonic() - last_report >= 60:
            print(
                "H46-D coordinator waiting for the serialized 0/A/B/C wave to finish",
                flush=True,
            )
            last_report = time.monotonic()
        time.sleep(2)

    for seed in range(3):
        output = runs_root / "H46-D" / f"seed{seed}"
        command = [
            "nice",
            "-n",
            "10",
            sys.executable,
            str(script),
            "--input",
            str(input_dir),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--arm",
            "D",
            "--seed",
            str(seed),
            "--device",
            device,
            "--own-lock",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode == 75:
            cpu_env = os.environ.copy()
            cpu_env["H46_DEVICE"] = "cpu"
            cpu_env["H46_RUN_ROOT"] = str(Path.home() / "jope-data/h46/runs/cpu")
            cpu_result = subprocess.run(
                [str(script.parent / "run_wave.sh")],
                cwd=ROOT,
                env=cpu_env,
                check=False,
            )
            return cpu_result.returncode
        if result.returncode != 0:
            return result.returncode
    return 0


def coordinate_c0_replay(input_dir, checkpoint, device):
    script = Path(__file__).resolve()
    while True:
        active_wave = False
        active_runner = False
        for proc in base.psutil.process_iter(["pid", "cmdline"]):
            if proc.info["pid"] == os.getpid():
                continue
            if process_matches(proc, "research/flywheel_selfplay/h46/run_wave.sh"):
                active_wave = True
            if process_matches(proc, "research/flywheel_selfplay/h46/run.py"):
                active_runner = True
        if not active_wave and not active_runner:
            break
        time.sleep(2)

    output = Path.home() / "jope-data/h46/runs/H46-C/replay_seed0"
    command = [
        "nice",
        "-n",
        "10",
        sys.executable,
        str(script),
        "--input",
        str(input_dir),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--arm",
        "C",
        "--seed",
        "0",
        "--device",
        device,
        "--own-lock",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode == 75:
        cpu_env = os.environ.copy()
        cpu_env["H46_DEVICE"] = "cpu"
        cpu_env["H46_RUN_ROOT"] = str(Path.home() / "jope-data/h46/runs/cpu")
        return subprocess.run(
            [str(script.parent / "run_wave.sh")], cwd=ROOT, env=cpu_env, check=False
        ).returncode
    if result.returncode != 0:
        return result.returncode

    original = json.loads(
        (Path.home() / "jope-data/h46/runs/H46-C/seed0/result.json").read_text()
    )
    replay = json.loads((output / "result.json").read_text())
    exact = all(
        left["val"]["per_context"] == right["val"]["per_context"]
        for left, right in zip(
            original["evaluations"], replay["evaluations"], strict=True
        )
    )
    print(f"C seed0 checkpoint replay exact={exact}", flush=True)
    return 0 if exact else 1


def checkpoint_guard(guard, net, expected=DOPE_SHA):
    snap = guard.snapshot_dope_weights(net)
    if snap.composite_sha256 != expected:
        raise RuntimeError(f"DOPE guard failed: {snap.composite_sha256}")
    counts = guard.get_trainable_param_counts_by_submodule(net)
    if (
        counts.get("__global_total__", {}).get("trainable_params") != 29211
        or counts.get("adapter_10tok", {}).get("trainable_params") != 29211
    ):
        raise RuntimeError(f"trainable parameter guard failed: {counts}")
    return snap.composite_sha256


def make_tokens(net, tables, device):
    tokens, masks = {}, {}
    ws = base.FusedLandmarkWorkspace(num_threads=4)
    for i, (cid, (x, y)) in enumerate(tables.items(), 1):
        if i % 16 == 0:
            base.headroom()
        yd = y.astype(np.float64)
        yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
        vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
            base.landmark_float32(x), np.ascontiguousarray(yn)
        )
        colbuf = np.zeros((64, 17), np.float32)
        n = min(len(cols), 64)
        colbuf[:n, : min(cols.shape[1], 17)] = cols[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vec, device=device).reshape(1, 256),
                torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, mask, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        tokens[cid], masks[cid] = tok[0].detach(), mask[0].detach()
    return tokens, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path.home() / "jope-data/h46/input")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "jope-data/h46/ckpt/round-785.pt",
    )
    ap.add_argument("--output", type=Path, default=Path.home() / "jope-data/h46/runs")
    ap.add_argument("--arm", choices=["0", "A", "B", "C", "D"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1219)
    ap.add_argument("--coordinator", action="store_true")
    ap.add_argument("--replay-c0-coordinator", action="store_true")
    ap.add_argument("--own-lock", action="store_true")
    args = ap.parse_args()
    if args.coordinator:
        raise SystemExit(
            coordinate_d_runs(
                args.input,
                Path.home() / "jope-data/h46/runs",
                args.checkpoint,
                args.device,
            )
        )
    if args.replay_c0_coordinator:
        raise SystemExit(
            coordinate_c0_replay(args.input, args.checkpoint, args.device)
        )
    if args.arm is None or args.seed is None:
        raise ValueError("--arm and --seed are required unless running as coordinator")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.90)
    lock_file = None
    if args.own_lock:
        lock_started = time.monotonic()
        while True:
            lock_file = open(Path.home() / "jope_plan/qualification.lock", "a")
            try:
                base.fcntl.flock(lock_file, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
            except BlockingIOError:
                lock_file.close()
                if time.monotonic() - lock_started >= 1200:
                    print(
                        "qualification.lock busy for 20 minutes; CPU fallback required",
                        flush=True,
                    )
                    raise SystemExit(75)
                time.sleep(30)
                continue
            if base.psutil.virtual_memory().available < 20 * 1024**3:
                raise RuntimeError("available RAM below 20 GiB")
            if device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if free_bytes < 20 * 1024**3:
                    base.fcntl.flock(lock_file, base.fcntl.LOCK_UN)
                    lock_file.close()
                    lock_file = None
                    time.sleep(30)
                    continue
            break
    selected, pools, tables = base.load_packet(args.input)
    if len(selected["train"]) != 1219 or len(selected["dev"]) != 91:
        raise RuntimeError("H46 cohort inventory mismatch")
    train = [r["context_sha256"] for r in selected["train"]]
    dev = [r["context_sha256"] for r in selected["dev"]]
    train_family = {r["context_sha256"]: r["family_id"] for r in selected["train"]}
    spec = importlib.util.spec_from_file_location(
        "h46_fused", args.input / "cpu_exact_fused.py"
    )
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    guard = base.guard
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for name, p in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            p.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state["state_dict"], strict=True)
    before = checkpoint_guard(guard, net)
    checkpoint_sha = digest(args.checkpoint)
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    guard.assert_optimizer_excludes_dope(opt, net)
    opt.load_state_dict(state["optimizer"])
    tokens, masks = make_tokens(net, tables, device)

    def predict(cids, indexes):
        progs = np.stack([pools[c]["program"][ix] for c, ix in zip(cids, indexes)])
        proj = np.stack([pools[c]["proj16"][ix] for c, ix in zip(cids, indexes)])
        m = progs.shape[1]
        b = len(cids)
        histf = torch.as_tensor(
            np.stack(
                [pools[c]["fitness"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)]
            ),
            dtype=torch.float32,
            device=device,
        )
        histg = torch.as_tensor(
            np.stack(
                [pools[c]["gate"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)]
            ),
            dtype=torch.float32,
            device=device,
        )
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584), device=device),
            prog_412=torch.as_tensor(progs, device=device),
            proj_16=torch.as_tensor(proj, device=device),
            base_prior_cost=torch.zeros((b, m), device=device),
            scale=torch.full((b,), 0.4208, device=device),
            sos_prior=torch.zeros((b, m, 2), device=device),
            mm_tokens_10x32=torch.stack([tokens[c] for c in cids]),
            mm_mask_10=torch.stack([masks[c] for c in cids]),
            revealed_indices=torch.arange(min(4, m), device=device)
            .reshape(1, -1)
            .expand(b, -1),
            revealed_raw_f=histf,
            revealed_gate=histg,
        )["total_cost"].float()

    @torch.no_grad()
    def evaluate(cids):
        net.eval()
        by_pool_size = {}
        for cid in cids:
            by_pool_size.setdefault(len(pools[cid]["pids"]), []).append(cid)
        decisions = {}
        for pool_size, contexts in by_pool_size.items():
            for start_idx in range(0, len(contexts), 8):
                batch = contexts[start_idx : start_idx + 8]
                indexes = [np.arange(pool_size) for _ in batch]
                predictions = predict(batch, indexes).cpu().numpy()
                for row, cid in zip(predictions, batch):
                    choice = int(np.argmin(row))
                    decisions[cid] = (
                        float(pools[cid]["fitness"][choice]),
                        bool(pools[cid]["gate"][choice]),
                        pools[cid]["pids"][choice],
                    )
        fs = [decisions[cid][0] for cid in cids]
        gs = [decisions[cid][1] for cid in cids]
        chosen = [decisions[cid][2] for cid in cids]
        values = [-f if g else -2.0 for f, g in zip(fs, gs)]
        family_by_cid = {
            r["context_sha256"]: r["family_id"]
            for role in ("train", "dev")
            for r in selected[role]
        }
        fam = {}
        for c, v in zip(cids, values):
            fam.setdefault(family_by_cid[c], []).append(v)
        return {
            "mean": float(np.mean(values)),
            "per_context": values,
            "per_family": {k: float(np.mean(v)) for k, v in fam.items()},
            "selected_programs": chosen,
        }

    baseline = evaluate(dev)
    checkpoint_guard(guard, net)
    steps = 32 if args.arm == "0" else 64
    eval_every = 2 if args.arm == "0" else 8
    trainable = [p for p in net.parameters() if p.requires_grad]
    ema = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
    incumbent = {
        n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad
    }
    incumbent_opt = copy.deepcopy(opt.state_dict())
    incumbent_utility = BASE_UTILITY
    incumbent_result = baseline
    hist = []
    start = time.monotonic()
    for step in range(steps):
        net.train()
        rng = np.random.default_rng(args.seed * 1009 + step)
        cids = list(train)
        rng.shuffle(cids)
        cids = cids[: args.batch_size]
        indexes = [base.training_indices(pools[c], 192) for c in cids]
        opt.zero_grad(set_to_none=True)
        pred = predict(cids, indexes)
        target = torch.as_tensor(
            np.stack([pools[c]["fitness"][ix] for c, ix in zip(cids, indexes)]),
            dtype=torch.float32,
            device=device,
        )
        gates = torch.as_tensor(
            np.stack([pools[c]["gate"][ix] for c, ix in zip(cids, indexes)]),
            dtype=torch.bool,
            device=device,
        )
        target = torch.where(gates, target, torch.full_like(target, 2.0))
        if args.arm == "C":
            med = target.median(dim=1, keepdim=True).values
            mad = (target - med).abs().median(dim=1, keepdim=True).values.clamp_min(0.1)
            pn = ((pred - med) / mad).clamp(-3, 3)
            tn = ((target - med) / mad).clamp(-3, 3)
            loss = F.smooth_l1_loss(pn, tn, beta=1.0)
        elif args.arm == "D":
            weights = torch.as_tensor(
                family_balanced_weights([train_family[cid] for cid in cids]),
                dtype=torch.float32,
                device=device,
            )
            row_huber = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
            row_kl = F.kl_div(
                F.log_softmax(-pred / 0.2, dim=-1),
                F.softmax(-target / 0.2, dim=-1),
                reduction="none",
            ).sum(dim=1)
            loss = torch.sum(weights * (row_huber + 0.1 * row_kl)) / len(cids)
        else:
            loss = F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, dim=-1),
                F.softmax(-target / 0.2, dim=-1),
                reduction="batchmean",
            )
        if args.arm == "B":
            lr = (
                3e-4 * (step + 1) / 4
                if step < 4
                else 3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + np.cos(np.pi * (step - 3) / 61))
            )
            for group in opt.param_groups:
                group["lr"] = float(lr)
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, 0.5 if args.arm == "B" else 1.0)
        )
        opt.step()
        with torch.no_grad():
            for n, p in net.named_parameters():
                if p.requires_grad:
                    ema[n].mul_(0.98).add_(p.detach(), alpha=0.02)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            checkpoint_guard(guard, net)
            latest = {
                n: p.detach().clone()
                for n, p in net.named_parameters()
                if p.requires_grad
            }
            if args.arm != "0":
                with torch.no_grad():
                    for n, p in net.named_parameters():
                        if p.requires_grad:
                            p.copy_(ema[n])
            result = evaluate(dev)
            checkpoint_guard(guard, net)
            save_adapter_state(
                args.output / f"adapter-step{step + 1:04d}.pt",
                net,
                checkpoint_sha,
                before,
            )
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": norm,
                "val": result,
                "delta": result["mean"] - BASE_UTILITY,
                "dope_sha256": before,
            }
            hist.append(row)
            atomic(args.output / "evaluations.json", hist)
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if p.requires_grad:
                        p.copy_(latest[n])
            if args.arm == "0":
                if result["mean"] > incumbent_utility:
                    incumbent_utility = result["mean"]
                    incumbent_result = result
                    incumbent = {
                        n: p.detach().clone()
                        for n, p in net.named_parameters()
                        if p.requires_grad
                    }
                    incumbent_opt = copy.deepcopy(opt.state_dict())
                else:
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if p.requires_grad:
                                p.copy_(incumbent[n])
                    opt.load_state_dict(incumbent_opt)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "seed": args.seed,
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "elapsed": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    if args.arm == "0":
        final_val = incumbent_result
    else:
        with torch.no_grad():
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.copy_(ema[n])
        checkpoint_guard(guard, net)
        final_val = evaluate(dev)
    final_train = evaluate(train)
    checkpoint_guard(guard, net)
    save_adapter_state(args.output / "adapter-final.pt", net, checkpoint_sha, before)
    output = {
        "arm": args.arm,
        "seed": args.seed,
        "device": str(device),
        "baseline": baseline,
        "baseline_utility": BASE_UTILITY,
        "candidate": {
            "val": final_val,
            "train": final_train,
            "train_val_gap": final_train["mean"] - final_val["mean"],
        },
        "steps": steps,
        "evaluations": hist,
        "wall_seconds": time.monotonic() - start,
        "dope_sha256": before,
        "checkpoint_sha256": checkpoint_sha,
        "trainable_params": 29211,
    }
    atomic(args.output / "result.json", output)


if __name__ == "__main__":
    main()
