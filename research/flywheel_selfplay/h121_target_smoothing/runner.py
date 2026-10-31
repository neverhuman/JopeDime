#!/usr/bin/env python3
"""H121 CPU-only preregistration and synthetic validation helpers."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def _load_h59():
    import importlib.util
    import sys

    p = Path(__file__).resolve().parent.parent / "h97_reg_ema" / "runner.py"
    spec = importlib.util.spec_from_file_location("h121_runner_contract", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.h59


h59 = _load_h59()


def atomic_json(path: Path, value: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


ARMS = ("CONTROL", "TS-0.1", "TS-0.3")
SEEDS = (45, 46, 47)
FOLD_SEED = 12100
UPDATES = 40
EPSILON = {"CONTROL": 0.0, "TS-0.1": 0.1, "TS-0.3": 0.3}
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"


def smooth_targets(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Blend each row's candidate utilities toward its own slate mean."""
    if targets.ndim != 2 or targets.shape[1] < 2 or not torch.isfinite(targets).all():
        raise ValueError("targets must be a finite [contexts,candidates] tensor")
    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be finite and in [0,1)")
    return (1 - epsilon) * targets + epsilon * targets.mean(dim=1, keepdim=True)


def family_folds(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty fold roster")
    fams = set()
    for r in rows:
        if (
            r.get("dl_split") != "DL-Train"
            or r.get("task") != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in (None, "train", "inner")
            or r.get("origin") not in (None, "real")
            or not isinstance(r.get("family_id"), str)
            or not r["family_id"]
        ):
            raise ValueError(
                "fold roster must be real DL-Train regression families only"
            )
        fams.add(r["family_id"])
    if len(fams) < 5:
        raise ValueError("five or more families required")
    ordered = sorted(fams)
    rng = np.random.default_rng(FOLD_SEED)
    perm = rng.permutation(len(ordered))
    folds = [[] for _ in range(5)]
    for i, ix in enumerate(perm):
        folds[i % 5].append(ordered[int(ix)])
    folds = [sorted(x) for x in folds]
    payload = {"seed": FOLD_SEED, "folds": folds}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **payload,
        "train_families": sorted(set().union(*folds[:4])),
        "score_families": folds[4],
        "assignment_sha256": digest,
    }


def scratch_path(root: Path, arm: str, seed: int, attempt: int) -> Path:
    if arm not in ARMS or seed not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError("invalid arm/seed/attempt")
    return Path(root) / "scratch" / f"{arm}-seed{seed}-attempt{attempt}"


def mean_regret(
    predicted_utility: torch.Tensor, true_utility: torch.Tensor
) -> torch.Tensor:
    """Mean regret across every candidate: per row max(true utility)-predicted candidate utility."""
    if predicted_utility.shape != true_utility.shape or predicted_utility.ndim != 2:
        raise ValueError(
            "prediction and truth must be matching context-by-candidate matrices"
        )
    if (
        not torch.isfinite(predicted_utility).all()
        or not torch.isfinite(true_utility).all()
    ):
        raise ValueError("utilities must be finite")
    oracle = true_utility.max(dim=1, keepdim=True).values
    return (oracle - predicted_utility).mean(dim=1)


def plan() -> dict:
    return {
        "schema": "jope.h121.cpu-preparation/v1",
        "status": "prepared_not_executed",
        "arms": list(ARMS),
        "epsilon": EPSILON,
        "formula": "y'=(1-epsilon)*y+epsilon*mean_ctx(y)",
        "control": "H82 recipe unchanged",
        "paired_seeds": list(SEEDS),
        "updates_per_cell": UPDATES,
        "fold": {
            "source": "real H50 DL-Train only",
            "whole_family_folds": 5,
            "train": "first four folds",
            "score": "fifth fold",
            "seed": FOLD_SEED,
        },
        "primary": "paired family delta vs same-cell r882 on official argmin utility; median over three seeds",
        "admission": {
            "median_delta_strictly_positive": True,
            "family_bootstrap_lcb95_strictly_above": 0.001,
            "harm_definition": "delta < -0.01",
            "harm_fraction_max": 0.10,
            "holm": "two treatment-vs-r882 tests; ties are not wins",
        },
        "co_readout": "mean regret over all candidates, per H114(a), descriptive beside official utility",
        "elo": "family-grouped Bradley-Terry vs r882=1500 with 95% CI",
        "auc": False,
        "dl_val": "91 contexts descriptive only",
        "frozen_dope_sha256": DOPE_SHA256,
        "adapter_only": True,
        "checkpoint": "fresh per-cell copy of ~/jope-data/h51/ckpt/round-882.pt",
        "scratch": "scratch/{arm}-seed{seed}-attemptN/ unique per attempt",
        "executor": "H97 guarded training cell with H82 loss and H61 scoring; H50 DL-Train only",
        "coordinator": "H97/H61 FIFO lease, H99/H100 closeout barrier, explicit --authorized-execution",
        "cpu_prep": {
            "synthetic_only": True,
            "optimizer_updates": 0,
            "checkpoint_reads": 0,
            "outcome_reads": 0,
            "gpu_used": False,
            "gpu_ticket_created": False,
        },
    }


def cpu_dry_run() -> dict:
    fixture = torch.tensor([[0.0, 1.0, 3.0], [5.0, 5.0, -1.0]])
    sm = {str(e): smooth_targets(fixture, e).tolist() for e in (0.0, 0.1, 0.3)}
    rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"synthetic-{i}"}
        for i in range(25)
    ]
    fold = family_folds(rows)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.01)
    start = p.detach().clone()
    opt.zero_grad()
    (p.square()).backward()
    opt.step()
    assert not torch.equal(start, p.detach())
    assert len(fold["score_families"]) == 5 and not set(fold["train_families"]) & set(
        fold["score_families"]
    )
    from execution import build_runtime

    h97, state, _loss, _atomic = build_runtime()
    assert h97.ARMS == ARMS and h97.SEEDS == SEEDS
    assert state["folds"] is None and state["score_rows"] is None
    return {
        "plan": plan(),
        "synthetic_smoothing": sm,
        "synthetic_fold_sha256": fold["assignment_sha256"],
        "synthetic_adapter_update_changed_parameter": True,
        "runtime_adapter_loaded": True,
        "runtime_arms": list(h97.ARMS),
        "runtime_seeds": list(h97.SEEDS),
        "real_checkpoint_or_data_accessed": False,
        "gpu_used": False,
        "gpu_ticket_created": False,
    }


def run_cell(args):
    """Execute one guarded H121 cell through the H97 CUDA/runtime path."""
    from execution import run_cell as run_h121_cell

    return run_h121_cell(args)


def summarize(cells, draws=10000):
    """Compute the registered family-bootstrap, Holm, Elo, and regret readouts."""
    from execution import summarize as summarize_h121

    return summarize_h121(cells, draws=draws)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "jope-data/h51/ckpt/round-882.pt",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--attempt-token")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.cpu_dry_run:
        result = cpu_dry_run()
        blob = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(blob)
        else:
            print(blob, end="")
        return
    if (
        args.arm is None
        or args.seed is None
        or args.out is None
        or args.phase is None
        or args.attempt_token is None
        or args.lock_fd is None
    ):
        parser.error(
            "execution requires arm, seed, out, phase, attempt-token and inherited lock-fd"
        )
    if (
        args.attempt < 1
        or args.out.exists()
        or args.phase != args.out.with_suffix(".phase.json")
    ):
        raise RuntimeError("repeated cell or unexpected phase path")
    result = run_cell(args)
    print(
        json.dumps(
            {
                "wave": "H121",
                "arm": result["arm"],
                "seed": result["seed"],
                "updates": result["updates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
