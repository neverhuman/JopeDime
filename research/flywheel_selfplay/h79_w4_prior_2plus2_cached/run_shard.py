#!/usr/bin/env python3
"""H79 W4 cached-pool screen: nonadaptive prior-top4 versus a two-plus-two reveal policy.

Descriptive retrospective screening on the frozen H67 fit-only roster; never a gate runner.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.flywheel_selfplay.h75_closed_loop_evolution.scoring.score_panel import (  # noqa: E402
    ELO_SHA256,
    load_reference,
)
from research.flywheel_selfplay.h75_w4_cached_baseline_shard import run_shard as h75_w4  # noqa: E402

ELO_SOURCE = REPO_ROOT / "python/src/jope/north_star/elo_tracker.py"
H75_W4_METRICS = (
    REPO_ROOT / "research/flywheel_selfplay/h75_w4_cached_baseline_shard/metrics.json"
)
W1_PRIORS_SHA = "5213faa8d0bcc906e1d3c3226d6104e3846cf11c9da9c222298e9f0dc2fb46d9"
W1_PRIORS_S3 = (
    "s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/"
    "w1_cpu_prep_20261031T0312Z/w1_train_only_priors.npz"
)
W1_FOLDS_SHA = "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1"
SEEDS = h75_w4.SEEDS
ARMS = ("random4", "prior10_top4", "gbm_top4", "gbm_2plus2", "gbm_2plus2_swap")
ANCHOR = "gbm_top4"
PRIMARY = ("gbm_2plus2", "gbm_top4")
SECONDARY = (
    ("gbm_top4", "random4"),
    ("gbm_top4", "prior10_top4"),
    ("gbm_2plus2", "gbm_2plus2_swap"),
)
NOISE_RATIO = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261031
TIE = 1e-4
HARM = 0.01


class Oracle:
    """Holds one context's cached outcomes; selectors see only what they reveal, four at most."""

    def __init__(self, fitness: np.ndarray, gate: np.ndarray, budget: int = 4):
        self._fitness = fitness
        self._gate = gate
        self._budget = budget
        self.revealed: list[int] = []

    def reveal(self, indices: list[int]) -> list[float]:
        if len(set(indices)) != len(indices) or set(indices) & set(self.revealed):
            raise ValueError("a candidate may be revealed only once")
        if len(self.revealed) + len(indices) > self._budget:
            raise ValueError("reveal budget exceeded")
        self.revealed.extend(indices)
        return [
            h75_w4.utility(float(self._fitness[i]), bool(self._gate[i]))
            for i in indices
        ]


def ranked(
    score: np.ndarray, ids: list[str], exclude: set[int] = frozenset()
) -> list[int]:
    """Indices by descending score; program SHA breaks ties, so ranking is seed-free."""
    return sorted(
        (i for i in range(len(ids)) if i not in exclude),
        key=lambda i: (-float(score[i]), ids[i]),
    )


def lengthscale(x: np.ndarray) -> float:
    """Median pairwise Euclidean distance over the whole pool: outcome-blind."""
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    value = float(np.median(np.sqrt(d2[np.triu_indices(len(x), 1)])))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "degenerate program embedding: median pairwise distance is not positive"
        )
    return value


def posterior_mean(
    mu: np.ndarray, x: np.ndarray, shown: list[int], residual: list[float], scale: float
) -> np.ndarray:
    """Prior mean plus an RBF Gaussian-process correction from the revealed residuals."""

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (
            np.sum(a * a, axis=1)[:, None]
            + np.sum(b * b, axis=1)[None, :]
            - 2.0 * (a @ b.T)
        )
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * scale * scale))

    xs = x[shown]
    alpha = np.linalg.solve(
        kernel(xs, xs) + NOISE_RATIO * np.eye(len(shown)), np.asarray(residual)
    )
    return mu + kernel(x, xs) @ alpha


def top4(mu: np.ndarray, ids: list[str], oracle: Oracle) -> list[int]:
    picks = ranked(mu, ids)[:4]
    oracle.reveal(picks)
    return picks


def two_plus_two(
    mu: np.ndarray,
    x: np.ndarray,
    ids: list[str],
    oracle: Oracle,
    scale: float,
    swap: bool = False,
) -> list[int]:
    first = ranked(mu, ids)[:2]
    residual = [u - float(mu[i]) for i, u in zip(first, oracle.reveal(first))]
    if swap:
        residual.reverse()
    second = ranked(posterior_mean(mu, x, first, residual, scale), ids, set(first))[:2]
    oracle.reveal(second)
    return first + second


def select(
    arm: str, seed: int, context: str, ids: list[str], prior: dict, oracle: Oracle
) -> list[int]:
    if arm == "random4":
        index = {item: i for i, item in enumerate(ids)}
        picks = [
            index[item] for item in h75_w4.select_ids("random4", ids, context, seed)
        ]
        oracle.reveal(picks)
        return picks
    if arm == "prior10_top4":
        return top4(-prior["prior10_cost"], ids, oracle)
    if arm == "gbm_top4":
        return top4(prior["mu"], ids, oracle)
    if arm in ("gbm_2plus2", "gbm_2plus2_swap"):
        return two_plus_two(
            prior["mu"],
            prior["proj16"],
            ids,
            oracle,
            prior["scale"],
            swap=arm.endswith("swap"),
        )
    raise ValueError(f"unknown arm: {arm}")


def check_family_exclusion(
    fold_manifest: dict, family_by_context: dict[str, str], contexts: list[dict]
) -> dict:
    """Each acquisition family must sit wholly inside one W1 fold, so its out-of-fold model never saw it."""
    w1_fold = {
        c: f["fold_index"] for f in fold_manifest["folds"] for c in f["held_contexts"]
    }
    if set(w1_fold) != set(family_by_context):
        raise ValueError("W1 fold manifest and H67 roster cover different contexts")
    folds_by_family = defaultdict(set)
    for context, family in family_by_context.items():
        folds_by_family[family].add(w1_fold[context])
    acquisition_families = {
        family_by_context[row["context_sha256"]] for row in contexts
    }
    split = sorted(f for f in acquisition_families if len(folds_by_family[f]) != 1)
    if split:
        raise ValueError(f"acquisition family spans W1 folds: {split[:3]}")
    return {
        "acquisition_families_single_w1_fold": len(acquisition_families),
        "all_families_single_w1_fold": sum(
            len(v) == 1 for v in folds_by_family.values()
        ),
        "families": len(folds_by_family),
        # H67 and W1 number their seed-4900 folds differently; exclusion is checked per family, not by label.
        "acquisition_contexts_whose_h67_fold_label_differs_from_w1": sum(
            w1_fold[row["context_sha256"]] != row["fold"] for row in contexts
        ),
    }


def family_deltas(
    ctx_means: dict[str, dict[str, float]],
    family_by_context: dict[str, str],
    a: str,
    b: str,
) -> dict[str, float]:
    by_family = defaultdict(list)
    for context, values in ctx_means.items():
        by_family[family_by_context[context]].append(values[a] - values[b])
    return {family: mean(values) for family, values in sorted(by_family.items())}


def contrast(deltas: dict[str, float], draws: np.ndarray) -> dict:
    values = np.asarray(list(deltas.values()))
    boot = values[draws].mean(axis=1)
    return {
        "families": len(values),
        "family_mean": float(values.mean()),
        "family_median": float(np.median(values)),
        "families_losing_more_than_0_01_fraction": float(np.mean(values < -HARM)),
        "family_win_tie_loss": [
            int(np.sum(values > TIE)),
            int(np.sum(np.abs(values) <= TIE)),
            int(np.sum(values < -TIE)),
        ],
        "bootstrap_p2_5": float(np.percentile(boot, 2.5)),
        "bootstrap_p5": float(np.percentile(boot, 5.0)),
        "bootstrap_p97_5": float(np.percentile(boot, 97.5)),
    }


def summarize(
    ctx_means: dict[str, dict[str, float]], family_by_context: dict[str, str], reference
) -> dict:
    families = sorted(set(family_by_context[c] for c in ctx_means))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # One shared index matrix pairs every contrast on the same resampled families.
    draws = rng.integers(0, len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    by_family = defaultdict(lambda: defaultdict(list))
    for context, values in ctx_means.items():
        for arm, value in values.items():
            by_family[family_by_context[context]][arm].append(value)
    records = [
        reference.DatasetEvalRecord(c, family_by_context[c], "regression", v)
        for c, v in sorted(ctx_means.items())
    ]
    return {
        "context_mean_retained_best_utility": {
            arm: mean(v[arm] for v in ctx_means.values()) for arm in ARMS
        },
        "family_mean_retained_best_utility": {
            arm: mean(mean(by_family[f][arm]) for f in families) for arm in ARMS
        },
        "primary": {
            "contrast": f"{PRIMARY[0]} - {PRIMARY[1]}",
            **contrast(family_deltas(ctx_means, family_by_context, *PRIMARY), draws),
        },
        "secondary": {
            f"{a} - {b}": contrast(
                family_deltas(ctx_means, family_by_context, a, b), draws
            )
            for a, b in SECONDARY
        },
        "family_elo": reference.compute_family_grouped_bradley_terry_elo(
            records,
            anchor_contender=ANCHOR,
            anchor_elo=1500.0,
            tie_margin=TIE,
            prior=0.01,
            n_iters=400,
        ),
    }


def load_w1_priors(path: Path, contexts: list[str]) -> dict[str, dict]:
    if h75_w4.sha256(path) != W1_PRIORS_SHA:
        raise ValueError("W1 prior artifact hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        cids = [str(c) for c in z["cids"]]
        offsets = z["offsets"]
        position = {c: i for i, c in enumerate(cids)}
        out = {}
        for context in contexts:
            lo, hi = (
                int(offsets[position[context]]),
                int(offsets[position[context] + 1]),
            )
            out[context] = {
                "mu": -z["oof_cost_prog412_ctx"][lo:hi].astype(np.float64),
                "prior10_cost": z["prior10_cost"][lo:hi].astype(np.float64),
                "proj16": z["proj16"][lo:hi].astype(np.float64),
            }
    return out


def run(
    input_root: Path,
    roster_path: Path,
    priors_path: Path,
    folds_path: Path,
    out_dir: Path,
    s3_prefix: str,
) -> dict:
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    cpu_started = time.process_time()
    roster, contexts = h75_w4.verify_roster(input_root, roster_path)
    if h75_w4.sha256(folds_path) != W1_FOLDS_SHA:
        raise ValueError("W1 fold manifest hash mismatch")
    fold_manifest = json.loads(folds_path.read_text())
    family_by_context = {
        r["context_sha256"]: r["family_group"]
        for r in roster["context_fold_assignments"]
    }
    acquisition = [row["context_sha256"] for row in contexts]
    exclusion = check_family_exclusion(fold_manifest, family_by_context, contexts)
    priors = load_w1_priors(priors_path, acquisition)
    reference = load_reference(ELO_SOURCE)

    out_dir.mkdir(parents=True, exist_ok=False)
    by_key: dict[tuple[str, int, str], float] = {}
    selection_cpu = defaultdict(float)
    pool_hashes = []
    with (out_dir / "selection_trace.jsonl").open("x", encoding="utf-8") as trace:
        for row in contexts:
            context = row["context_sha256"]
            pool_path = input_root / "pools" / f"{context}.npz"
            if h75_w4.sha256(pool_path) != row["pool_sha256"]:
                raise ValueError(f"training pool hash mismatch for {context}")
            with np.load(pool_path, allow_pickle=False) as pool:
                ids = [item.decode() for item in pool["program_sha256"].tolist()]
                fitness = np.asarray(pool["fitness"], dtype=np.float64)
                gate = np.asarray(pool["gate"], dtype=np.bool_)
            prior = priors[context]
            if (
                not len(ids)
                == len(set(ids))
                == len(fitness)
                == len(gate)
                == len(prior["mu"])
                == len(prior["proj16"])
            ):
                raise ValueError(f"pool and W1 prior slots do not align for {context}")
            prior["scale"] = lengthscale(prior["proj16"])
            for seed in SEEDS:
                for arm in ARMS:
                    oracle = Oracle(fitness, gate)
                    tick = time.process_time()
                    picks = select(arm, seed, context, ids, prior, oracle)
                    selection_cpu[arm] += time.process_time() - tick
                    if picks != oracle.revealed or len(picks) != 4:
                        raise ValueError("selection and reveal log disagree")
                    scores = [
                        h75_w4.utility(float(fitness[i]), bool(gate[i])) for i in picks
                    ]
                    by_key[(context, seed, arm)] = max(scores)
                    trace.write(
                        json.dumps(
                            {
                                "context_sha256": context,
                                "family_group": row["family_group"],
                                "fold": row["fold"],
                                "seed": seed,
                                "arm": arm,
                                "candidate_count": len(ids),
                                "reveal_order_program_ids": [ids[i] for i in picks],
                                "revealed_utilities": scores,
                                "retained_best_utility": max(scores),
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
            pool_hashes.append(
                {
                    "context_sha256": context,
                    "sha256": row["pool_sha256"],
                    "s3_key": None,
                }
            )

    published = json.loads(H75_W4_METRICS.read_text())["seed_metrics"]
    reproduction = {
        str(seed): mean(by_key[(c, seed, "random4")] for c in acquisition)
        - published[str(seed)]["mean_retained_best_utility"]["random4"]
        for seed in SEEDS
    }
    if any(value != 0.0 for value in reproduction.values()):
        raise ValueError(
            f"random4 does not reproduce the published H75 W4 shard: {reproduction}"
        )
    ctx_means = {
        c: {arm: mean(by_key[(c, s, arm)] for s in SEEDS) for arm in ARMS}
        for c in acquisition
    }
    metrics = {
        "schema": "jopedime.h79-w4-prior-2plus2-cached-metrics.v1",
        "status": "DESCRIPTIVE_CACHED_POOL_SCREEN_ONLY",
        "arms": list(ARMS),
        "anchor": ANCHOR,
        "seeds": list(SEEDS),
        "context_count": len(acquisition),
        "family_groups": len({family_by_context[c] for c in acquisition}),
        "cached_replay_reveals": len(by_key) * 4,
        "native_fits": 0,
        "gpu_hours": 0,
        "random4_reproduction_delta_vs_h75_w4": reproduction,
        "family_exclusion": exclusion,
        "selection_cpu_seconds": dict(selection_cpu),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "family",
            "paired_shared_indices": True,
        },
        **summarize(ctx_means, family_by_context, reference),
        "val_reg_inner": None,
        "advance_gate_evaluated": False,
        "scientific_verdict": None,
    }
    finished = datetime.now(timezone.utc)
    prefix = s3_prefix.rstrip("/")
    manifest = {
        "schema": "jopedime.h79-w4-prior-2plus2-cached-run.v1",
        "run_id": out_dir.name,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "source_file_sha256": h75_w4.sha256(Path(__file__)),
        "s3_output_prefix": prefix,
        "s3_outputs": [
            f"{prefix}/{name}"
            for name in ("metrics.json", "manifest.json", "selection_trace.jsonl")
        ],
        "input_packet_id": h75_w4.PACKET_ID,
        "inputs": [
            {
                "local_path": str(input_root / "source_manifest.json"),
                "s3_key": None,
                "sha256": h75_w4.PACKET_FILE_SHA,
            },
            {
                "local_path": str(roster_path),
                "s3_key": None,
                "sha256": h75_w4.ROSTER_SHA,
            },
            {
                "local_path": str(priors_path),
                "s3_key": W1_PRIORS_S3,
                "sha256": W1_PRIORS_SHA,
            },
            {"local_path": str(folds_path), "s3_key": None, "sha256": W1_FOLDS_SHA},
            {"local_path": str(ELO_SOURCE), "s3_key": None, "sha256": ELO_SHA256},
            *pool_hashes,
        ],
        "seeds": list(SEEDS),
        "fold_seed": roster["fold_seed"],
        "host": socket.gethostname(),
        "cpu_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.monotonic() - clock,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "metrics": {
            "Elo": metrics["family_elo"]["ratings"],
            "val_reg_inner": None,
            "U4": metrics["family_mean_retained_best_utility"],
            "descriptive_metrics_file": "metrics.json",
        },
        "verdict": "NO_ADVANCEMENT_CLAIM; FIT_ONLY_RETROSPECTIVE_CACHED_POOL_SCREEN",
    }
    (out_dir / "metrics.json").write_bytes(h75_w4.json_bytes(metrics))
    (out_dir / "manifest.json").write_bytes(h75_w4.json_bytes(manifest))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT
        / "research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json",
    )
    parser.add_argument(
        "--priors",
        type=Path,
        default=Path.home() / "jope-data/w1/cache/w1_train_only_priors.npz",
    )
    parser.add_argument(
        "--folds", type=Path, default=Path.home() / "jope-data/w1/fold_manifest.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    args = parser.parse_args()
    metrics = run(
        args.input, args.roster, args.priors, args.folds, args.out, args.s3_prefix
    )
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("primary", "family_elo", "family_mean_retained_best_utility")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
