"""Validate the EWMA standard error with a circular block bootstrap.

Reads delta_raw from a train_metrics.jsonl, recomputes the debiased EWMA SE,
and compares against the bootstrap distribution of the EWMA estimate over
block-resampled delta series (block=25, 1000 resamples). Inflation factor
= bootstrap SD / analytic SE; if > 1.5, multiply displayed bands by it.

Usage: python3 -m python.elo_finetune.check_se_bootstrap metrics.jsonl \
           [--half-life 50] [--block 25] [--resamples 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .elo_meter import EwmaState


def ewma_estimate(deltas: np.ndarray, half_life: float) -> tuple[float, float]:
    state = EwmaState(half_life)
    m_hat, se = 0.0, 0.0
    for x in deltas:
        m_hat, se = state.update(float(x))
    return m_hat, se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--half-life", type=float, default=50.0)
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    deltas = []
    with open(args.metrics) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "train" and "delta_raw" in rec:
                deltas.append(rec["delta_raw"])
    deltas = np.array(deltas)
    if len(deltas) < 4 * args.block:
        raise SystemExit(f"only {len(deltas)} train steps; need >= {4 * args.block}")

    m_hat, analytic_se = ewma_estimate(deltas, args.half_life)

    rng = np.random.default_rng(args.seed)
    n = len(deltas)
    n_blocks = int(np.ceil(n / args.block))
    estimates = np.empty(args.resamples)
    for r in range(args.resamples):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(args.block)[None, :]).ravel() % n
        resampled = deltas[idx[:n]]
        estimates[r], _ = ewma_estimate(resampled, args.half_life)

    boot_sd = float(estimates.std(ddof=1))
    inflation = boot_sd / max(analytic_se, 1e-12)
    lag1 = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if n > 2 else 0.0
    print(
        json.dumps(
            {
                "steps": n,
                "half_life": args.half_life,
                "ewma_debiased": m_hat,
                "analytic_se": analytic_se,
                "bootstrap_sd": boot_sd,
                "inflation_factor": round(inflation, 3),
                "lag1_autocorr": round(lag1, 4),
                "action": "multiply bands by inflation_factor" if inflation > 1.5 else "bands ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
