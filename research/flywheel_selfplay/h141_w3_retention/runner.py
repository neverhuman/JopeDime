"""H141 bounded retained-TRAIN W3 experiment. CUDA requires inherited flock fd9."""

from __future__ import annotations
import argparse
from collections import defaultdict
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from torch import nn
import w3_losses
import dope_guard

HERE = Path(__file__).resolve().parent
R882 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
ARMS = ("a", "b", "c", "d", "d_scalar", "d_shuffled")


def SHA(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_net(device):
    source = Path("/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py")
    assert (
        SHA(source)
        == "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
    )
    spec = importlib.util.spec_from_file_location("h141_fused", source)
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    checkpoint_path = Path("/home/ubuntu/jope-data/H141-t1815/r882_init.pt")
    assert SHA(checkpoint_path) == R882
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    dope_guard.lock_dope_weights_and_eval_mode(net)
    assert dope_guard.snapshot_dope_weights(net).composite_sha256 == DOPE
    return net.to(device)


class Panel:
    def __init__(self, root, net, device):
        self.device = device
        receipt = json.loads((root / "feature_receipt.json").read_text())
        path = root / "admitted_features.npz"
        assert SHA(path) == receipt["features_sha256"]
        with np.load(path, allow_pickle=False) as src:
            self.cpu = {key: src[key] for key in src.files}
        self.cids = list(map(str, self.cpu["cids"]))
        self.families = list(map(str, self.cpu["families"]))
        self.offsets = self.cpu["offsets"]
        # Restore the original hash-ordered history, independent of metric values.
        permutation = np.concatenate(
            [
                np.argsort(self.cpu["pids"][lo:hi], kind="stable") + lo
                for lo, hi in zip(self.offsets[:-1], self.offsets[1:])
            ]
        )
        for key in ("program", "proj16", "fitness", "logratio", "gate", "pids"):
            self.cpu[key] = self.cpu[key][permutation]
        self.t = {
            key: torch.as_tensor(self.cpu[key], device=device)
            for key in ("program", "proj16", "tokens_10x32", "mask_10", "ctx_feat_584")
        }
        self.prior = torch.empty(len(self.cpu["program"]), device=device)
        self.sos = torch.empty((len(self.prior), 2), device=device)
        sys.path.insert(0, "/home/ubuntu/JopeDimeMicro/python")
        from phase12_multimodal_model import compute_semantic_struct_sig_24d

        with torch.no_grad():
            for i, (lo, hi) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
                if hi == lo:
                    continue
                programs = self.t["program"][lo:hi].unsqueeze(0)
                semantic = torch.as_tensor(
                    compute_semantic_struct_sig_24d(self.cpu["program"][lo:hi]),
                    device=device,
                ).unsqueeze(0)
                lm = torch.as_tensor(self.cpu["lm48_z"][i : i + 1], device=device)
                self.prior[lo:hi] = net._predict_calibrated_prior_batched_gpu(
                    programs, semantic, lm
                )[0]
                idx = torch.full((1, hi - lo), -1, dtype=torch.long, device=device)
                self.sos[lo:hi] = net._compute_sos_prior_batched_gpu(programs, idx)[0]
        assert (
            SHA(root / "fold_manifest.json")
            == "cabcfc6f182ef3aeb77a10890d81de0c655f60b5bb3db2a56b70c44a58567067"
        )
        fold_manifest = json.loads((root / "fold_manifest.json").read_text())
        held = {
            cid: fold["fold_index"]
            for fold in fold_manifest["folds"]
            for cid in fold["held_contexts"]
        }
        self.fold = np.asarray([held[cid] for cid in self.cids])

    def n(self, index):
        return int(self.offsets[index + 1] - self.offsets[index])

    def batch(self, indices, local_rows):
        global_rows = np.stack(
            [self.offsets[i] + rows for i, rows in zip(indices, local_rows)]
        )
        take = torch.as_tensor(global_rows, device=self.device)
        hist = global_rows[:, :4]
        scale = np.asarray(
            [max(float(np.std(self.cpu["fitness"][r])), 0.1) for r in hist],
            dtype=np.float32,
        )
        return dict(
            ctx_feat=self.t["ctx_feat_584"][indices],
            prog_412=self.t["program"][take],
            proj_16=self.t["proj16"][take],
            base_prior_cost=self.prior[take],
            scale=torch.as_tensor(scale, device=self.device),
            sos_prior=self.sos[take],
            mm_tokens_10x32=self.t["tokens_10x32"][indices],
            mm_mask_10=self.t["mask_10"][indices],
            revealed_indices=torch.arange(4, device=self.device)[None].expand(
                len(indices), -1
            ),
            revealed_raw_f=torch.as_tensor(
                self.cpu["fitness"][hist], dtype=torch.float32, device=self.device
            ),
            revealed_gate=torch.as_tensor(
                self.cpu["gate"][hist], dtype=torch.float32, device=self.device
            ),
        )

    def targets(self, indices, rows):
        result = []
        for i, local in zip(indices, rows):
            slate = []
            for j, k in enumerate(self.offsets[i] + local):
                passed = bool(self.cpu["gate"][k])
                slate.append(
                    {
                        "status": "completed_pass" if passed else "completed_gate_fail",
                        "fitness": float(self.cpu["fitness"][k]) if passed else None,
                        "logratio": float(self.cpu["logratio"][k]),
                        "gate": passed,
                        "revealed": j < 4,
                    }
                )
            result.append(slate)
        return result


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(35, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, output, batch, scalar=False):
        base = output["total_cost"]
        x = torch.cat(
            [
                output["cand_q"],
                base[..., None],
                batch["base_prior_cost"][..., None],
                batch["scale"][:, None, None].expand(-1, base.shape[1], 1),
            ],
            dim=-1,
        )
        delta = self.layers(x)
        if scalar:
            return base[..., None] + delta
        return torch.stack(
            [base / 0.9 + delta[..., 0], delta[..., 1], delta[..., 2]], dim=-1
        )


def components(net, panel, indices, rows, heads=None, arm="a"):
    batch = panel.batch(indices, rows)
    out = net.adapter_10tok.predict_costs(**batch)
    auxiliary = None if heads is None else heads(out, batch, arm == "d_scalar")
    return out["total_cost"], auxiliary


def trainable(net):
    result = []
    for name, parameter in net.adapter_10tok.named_parameters():
        active = not name.startswith(("anchor.", "parent_delta_net."))
        parameter.requires_grad_(active)
        if active:
            result.append(parameter)
    assert sum(p.numel() for p in result) == 23546
    return result


def batches_by_size(panel, indices):
    groups = defaultdict(list)
    for index in indices:
        if panel.n(index) >= 4:
            groups[panel.n(index)].append(index)
    for n, group in sorted(groups.items()):
        for start in range(0, len(group), 32):
            batch = group[start : start + 32]
            yield batch, [np.arange(n) for _ in batch]


def raw_utility(panel, index):
    lo, hi = panel.offsets[index : index + 2]
    return np.where(panel.cpu["gate"][lo:hi], -panel.cpu["fitness"][lo:hi], -2.0)


def calibration(net, panel, heads, fit_indices, arm):
    mean_n = len(fit_indices)
    family_counts = defaultdict(int)
    for i in fit_indices:
        family_counts[panel.families[i]] += 1
    features, targets, weights = [], [], []
    with torch.no_grad():
        for indices, rows in batches_by_size(panel, fit_indices):
            _, auxiliary = components(net, panel, indices, rows, heads, arm)
            for i, a in zip(indices, auxiliary.cpu().double().numpy()):
                a = a[4:]
                if not len(a):
                    continue
                features.append(a)
                targets.append(raw_utility(panel, i)[4:])
                weights.append(
                    np.full(len(a), 1 / (family_counts[panel.families[i]] * len(a)))
                )
    x, y, w = np.concatenate(features), np.concatenate(targets), np.concatenate(weights)
    mu = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x - mu) ** 2, axis=0, weights=w)).clip(1e-6)
    design = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    # Fixed ridge=1, equal family/context weights. No held-fold target enters.
    penalty = np.diag([0.0, 1.0, 1.0, 1.0])
    beta = np.linalg.solve(
        design.T @ (w[:, None] * design) + penalty, design.T @ (w * y)
    )
    return {
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "beta": beta.tolist(),
        "fit_contexts": mean_n,
        "fit_rows": len(y),
        "ridge": 1.0,
    }


def score(net, panel, indices, seed, heads=None, arm="r882", calibrator=None):
    results = {}
    with torch.no_grad():
        for batch, rows in batches_by_size(panel, indices):
            prediction, auxiliary = components(net, panel, batch, rows, heads, arm)
            costs = prediction.cpu().double().numpy()
            if calibrator is not None:
                a = auxiliary.cpu().double().numpy()
                x = (a - np.asarray(calibrator["mean"])) / np.asarray(calibrator["std"])
                beta = np.asarray(calibrator["beta"])
                costs = -(beta[0] + x @ beta[1:])
            for i, cost in zip(batch, costs):
                if arm == "ridge150":
                    lo, hi = panel.offsets[i : i + 2]
                    cost = panel.prior[lo:hi].cpu().double().numpy()
                elif arm == "zero":
                    cost = np.zeros_like(cost)
                utility = raw_utility(panel, i)
                if arm == "best_revealed":
                    choice = int(np.argmax(utility[:4]))
                else:
                    choice = int(np.argsort(cost, kind="stable")[0])
                new = np.argsort(cost[4:], kind="stable")[:4] + 4
                if arm == "random4" and len(cost) >= 8:
                    rng = np.random.default_rng(seed + int(panel.cids[i][:8], 16))
                    new = rng.choice(np.arange(4, len(cost)), 4, replace=False)
                    choice = int(new[0])
                best0 = float(np.max(utility[:4]))
                u4 = (
                    float(np.max(utility[np.r_[np.arange(4), new]]))
                    if len(new) == 4
                    else None
                )
                if arm == "best_revealed":
                    u4 = None
                best_seen = int(
                    np.r_[np.arange(4), new][
                        np.argmax(utility[np.r_[np.arange(4), new]])
                    ]
                )
                results[panel.cids[i]] = {
                    "context_sha256": panel.cids[i],
                    "family_id": panel.families[i],
                    "seed": seed,
                    "arm": arm,
                    "choice_local_row": choice,
                    "choice_program_sha256": str(
                        panel.cpu["pids"][panel.offsets[i] + choice]
                    ),
                    "val_reg_inner": float(utility[choice]),
                    "gate_failure": not bool(
                        panel.cpu["gate"][panel.offsets[i] + choice]
                    ),
                    "U0": best0,
                    "U4": u4,
                    "U4_gate_failure": (
                        not bool(panel.cpu["gate"][panel.offsets[i] + best_seen])
                    )
                    if u4 is not None
                    else None,
                    "new_programs": [
                        str(panel.cpu["pids"][panel.offsets[i] + j]) for j in new
                    ],
                    "unrevealed_choice_utility": float(
                        utility[4 + int(np.argmin(cost[4:]))]
                    )
                    if len(cost) > 4
                    else None,
                }
    return results


def one_cell(net, initial, panel, arm, fold, seed, destination, deadline):
    torch.manual_seed(seed)
    net.load_state_dict(initial, strict=True)
    parameters = trainable(net)
    heads = Heads().to(panel.device) if arm.startswith("d") else None
    if heads is not None:
        parameters += list(heads.parameters())
    fit = [
        i for i in range(len(panel.cids)) if panel.fold[i] != fold and panel.n(i) > 4
    ]
    held = [i for i in range(len(panel.cids)) if panel.fold[i] == fold]
    by_family = defaultdict(list)
    for i in fit:
        by_family[panel.families[i]].append(i)
    names = sorted(by_family)
    rng = np.random.default_rng(seed * 1000003 + fold * 97)
    shuffle_rng = np.random.default_rng(seed * 1000003 + fold * 97 + 77)
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=3e-5)
    start_weights = [p.detach().clone() for p in parameters]
    ema = [p.detach().clone() for p in parameters]
    losses = []
    exposure = hashlib.sha256()
    initial_guard = dope_guard.snapshot_dope_weights(net)
    assert initial_guard.composite_sha256 == DOPE
    net.train()
    attempt_path = destination / f"{arm}-fold{fold}-seed{seed}.attempt.json"
    attempt = {
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "actual_optimizer_updates": 0,
        "status": "STARTED",
        "initial_r882_sha256": R882,
    }
    write(attempt_path, attempt)
    for step in range(40):
        if time.monotonic() >= deadline:
            raise TimeoutError("H141 aggregate GPU lock-wall ceiling reached")
        if os.getloadavg()[0] > 0.85 * (os.cpu_count() or 1):
            raise RuntimeError("host CPU headroom above 85 percent")
        indices = [
            int(rng.choice(by_family[names[int(rng.integers(len(names)))]]))
            for _ in range(32)
        ]
        rows = [
            np.r_[
                np.arange(4),
                rng.choice(np.arange(4, panel.n(i)), 188, replace=panel.n(i) < 192),
            ]
            for i in indices
        ]
        exposure.update(np.asarray(indices, dtype="<i8").tobytes())
        exposure.update(np.asarray(rows, dtype="<i8").tobytes())
        targets = panel.targets(indices, rows)
        torch.manual_seed(seed * 1000 + step)
        optimizer.zero_grad(set_to_none=True)
        pred, aux = components(net, panel, indices, rows, heads, arm)
        if arm in ("a", "b", "c"):
            loss = w3_losses.w3_loss(
                arm, torch.zeros_like(pred), pred, targets, enabled=True
            )["loss"]
        elif arm == "d_scalar":
            loss = (
                sum(
                    w3_losses.w3_loss(
                        "a", torch.zeros_like(pred), aux[..., j], targets, enabled=True
                    )["loss"]
                    for j in range(3)
                )
                / 3
            )
        else:
            if arm == "d_shuffled":
                targets = [
                    slate[:4]
                    + [slate[j] for j in shuffle_rng.permutation(np.arange(4, 192))]
                    for slate in targets
                ]
            loss = w3_losses.w3_loss(
                "d",
                torch.zeros_like(pred),
                pred,
                targets,
                enabled=True,
                heads={
                    key: aux[..., j]
                    for j, key in enumerate(("quality", "residual", "gate"))
                },
            )["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 2.0, error_if_nonfinite=True)
        assert all(p.grad is None for p in net.dope_model.parameters())
        optimizer.step()
        scheduler.step()
        attempt["actual_optimizer_updates"] = step + 1
        write(attempt_path, attempt)
        with torch.no_grad():
            for average, parameter in zip(ema, parameters):
                average.mul_(0.9).add_(parameter, alpha=0.1)
        losses.append(float(loss.detach()))
    with torch.no_grad():
        for parameter, average in zip(parameters, ema):
            parameter.copy_(average)
    net.eval()
    final_guard = dope_guard.snapshot_dope_weights(net)
    assert final_guard.composite_sha256 == DOPE
    assert final_guard.tensor_hashes == initial_guard.tensor_hashes
    changed = sum(not torch.equal(a, b) for a, b in zip(start_weights, parameters))
    assert changed > 0
    calibrator = calibration(net, panel, heads, fit, arm) if heads is not None else None
    result = score(net, panel, held, seed, heads, arm, calibrator)
    name = f"{arm}-fold{fold}-seed{seed}"
    model_path = destination / (name + ".pt")
    torch.save(
        {
            "adapter": {
                k: v.detach().cpu()
                for k, v in net.adapter_10tok.state_dict().items()
                if not k.startswith("anchor.")
            },
            "heads": None
            if heads is None
            else {k: v.detach().cpu() for k, v in heads.state_dict().items()},
            "calibrator": calibrator,
            "r882_initial_sha256": R882,
            "dope_sha256": DOPE,
        },
        model_path,
    )
    receipt = {
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "updates": 40,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "changed_parameter_tensors": changed,
        "dope_sha256": DOPE,
        "all_dope_tensors_bit_identical": True,
        "exposure_sha256": exposure.hexdigest(),
        "losses": losses,
        "model_sha256": SHA(model_path),
        "calibrator": calibrator,
        "held_contexts_expected": len(held),
        "held_contexts_scored": len(result),
        "results": result,
    }
    write(destination / (name + ".json"), receipt)
    attempt["status"] = "COMPLETE"
    write(attempt_path, attempt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    lock = Path("/home/ubuntu/jope_plan/qualification.lock")
    assert os.fstat(9).st_ino == lock.stat().st_ino
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not torch.cuda.is_initialized()
    root = args.root
    launch = json.loads((root / "launch_admission.json").read_text())
    assert launch["admitted"] is True and launch["prereg_merged_commit"]
    for file, expected in launch["code_sha256"].items():
        assert SHA(HERE / file) == expected
    for file, expected in launch["inputs_sha256"].items():
        assert SHA(root / file) == expected
    start = time.monotonic()
    deadline = start + 7100
    destination = root / "run"
    destination.mkdir(exist_ok=False)
    status = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "RUNNING",
        "gpu_lock_wall_seconds": 0,
        "optimizer_updates": 0,
        "native_fits": 0,
        "cells_completed": 0,
    }
    write(destination / "budget.json", status)
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("external ceiling")),
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        # This is charged CUDA work, after the merged prereg and exclusive lock.
        parity = []
        synthetic = [
            [
                {
                    "status": "completed_pass",
                    "fitness": -1 + j * 0.03,
                    "logratio": -0.5 + j * 0.01,
                    "gate": True,
                    "revealed": j < 4,
                }
                for j in range(24)
            ]
        ]
        for arm in "abcd":
            cpu = torch.linspace(-1.0, 1.0, 24, dtype=torch.float64)[
                None
            ].requires_grad_()
            gpu = cpu.detach().to(device).requires_grad_()
            hc = {k: cpu.clone() for k in ("quality", "residual", "gate")}
            hg = {k: gpu.clone() for k in ("quality", "residual", "gate")}
            left = w3_losses.w3_loss(
                arm,
                torch.zeros_like(cpu),
                cpu,
                synthetic,
                enabled=True,
                heads=hc if arm == "d" else None,
            )["loss"]
            right = w3_losses.w3_loss(
                arm,
                torch.zeros_like(gpu),
                gpu,
                synthetic,
                enabled=True,
                heads=hg if arm == "d" else None,
            )["loss"]
            left.backward()
            right.backward()
            assert torch.allclose(left, right.cpu(), rtol=1e-10, atol=1e-10)
            assert torch.allclose(cpu.grad, gpu.grad.cpu(), rtol=1e-9, atol=1e-9)
            parity.append({"arm": arm, "loss_and_gradient_CPU_CUDA_parity": True})
        write(destination / "cuda_loss_parity.json", parity)
        net = load_net(device)
        initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
        panel = Panel(root, net, device)
        baseline = {}
        net.eval()
        for seed in (701, 702, 703):
            for arm in ("r882", "ridge150", "zero", "best_revealed", "random4"):
                baseline[f"{arm}-{seed}"] = score(
                    net, panel, list(range(len(panel.cids))), seed, arm=arm
                )
        write(destination / "baselines.json", baseline)
        for fold in range(5):
            for seed in (701, 702, 703):
                paired_exposure = None
                for arm in ARMS:
                    receipt = one_cell(
                        net, initial, panel, arm, fold, seed, destination, deadline
                    )
                    if paired_exposure is None:
                        paired_exposure = receipt["exposure_sha256"]
                    assert receipt["exposure_sha256"] == paired_exposure
                    status["cells_completed"] += 1
                    status["optimizer_updates"] += receipt["updates"]
                    status["gpu_lock_wall_seconds"] = time.monotonic() - start
                    write(destination / "budget.json", status)
                    print(
                        json.dumps({k: v for k, v in status.items() if k != "utc"}),
                        flush=True,
                    )
        status["status"] = "COMPLETE"
    except BaseException as exc:
        status["status"] = "FAILED_OR_INCOMPLETE"
        status["error"] = repr(exc)
        raise
    finally:
        status["gpu_lock_wall_seconds"] = time.monotonic() - start
        status["optimizer_updates"] = sum(
            json.loads(p.read_text())["actual_optimizer_updates"]
            for p in destination.glob("*.attempt.json")
        )
        write(destination / "budget.json", status)


if __name__ == "__main__":
    main()
