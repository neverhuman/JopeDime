#!/usr/bin/env python3
"""H51 offline arm runner; terminal DL-Val scoring is intentionally separate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
SPEC = importlib.util.spec_from_file_location("h51_h48", H48_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the pinned H48 offline runner")
h48 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h48
SPEC.loader.exec_module(h48)

MANIFEST_SHA = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
CHECKPOINT_SHA = "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SEEDS = (9, 10, 11)
FOLD_SEED = 4900
BOOTSTRAP_DRAWS = 10_000
STRICT_MARGIN = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def family_median_interval(gains: dict[str, float], seed: int) -> tuple[float, float, float]:
    families = sorted(gains)
    if len(families) != 205:
        raise RuntimeError(f"expected 205 inner-CV families, got {len(families)}")
    values = np.asarray([gains[family] for family in families], dtype=np.float64)
    rng = np.random.default_rng(510_000 + seed)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    boot_medians = np.median(values[indexes], axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(boot_medians, 0.025)),
        float(np.quantile(boot_medians, 0.975)),
    )


def run_control(args: argparse.Namespace) -> dict:
    if args.seed not in SEEDS:
        raise RuntimeError(f"H51 seeds are exactly {SEEDS}")
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("run_wave.sh must hold qualification.lock before an arm")
    if args.device not in ("cuda:0", "cpu"):
        raise RuntimeError("device must be cuda:0 or cpu")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("r785 checkpoint SHA mismatch")
    manifest = json.loads((args.input / "source_manifest.json").read_text())
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_SHA
        or manifest.get("dl_split_counts") != {"DL-Train": 1219, "DL-Val": 91}
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("H51 source manifest failed its pinned-scope checks")

    seed_dir = args.output / "CONTROL" / f"seed{args.seed}"
    if (seed_dir / "candidate-summary.json").exists():
        print(f"skip completed H51-CONTROL seed{args.seed}", flush=True)
        return json.loads((seed_dir / "candidate-summary.json").read_text())
    seed_dir.mkdir(parents=True, exist_ok=True)
    before_dope = h48.DOPE_SHA
    torch.set_num_threads(8 if args.device == "cpu" else 4)
    h48_args = argparse.Namespace(
        input=args.input,
        checkpoint=args.checkpoint,
        output=seed_dir,
        seed=args.seed,
        fold_seed=FOLD_SEED,
        budget_seconds=300.0,
        device=torch.device(args.device),
        lock_held=True,
        arm="0",
    )
    result = h48.run_training_arm(h48_args)
    if result.get("validation_scored") or result.get("validation_payloads_opened"):
        raise RuntimeError("inner-CV runner unexpectedly opened DL-Val")
    if result.get("dope_sha256") != DOPE_SHA or before_dope != DOPE_SHA:
        raise RuntimeError("DOPE identity changed during H51-CONTROL")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("r785 checkpoint changed during H51-CONTROL")

    family_gains: dict[str, float] = {}
    folds = result.get("folds", [])
    if len(folds) != 5:
        raise RuntimeError(f"expected five family folds, got {len(folds)}")
    for fold in folds:
        baseline = fold["baseline_per_family_utility"]
        selected = fold["selected_per_family_utility"]
        if set(baseline) != set(selected):
            raise RuntimeError("fold family roster differs between baseline and candidate")
        for family, value in selected.items():
            if family in family_gains:
                raise RuntimeError(f"family appears in more than one held-out fold: {family}")
            family_gains[family] = float(value) - float(baseline[family])
    median, lower, upper = family_median_interval(family_gains, args.seed)
    selection = json.loads((seed_dir / "selected-candidate.json").read_text())
    adapter_path = seed_dir / selection["checkpoint"]
    adapter_sha = sha256(adapter_path)
    accepted = median > STRICT_MARGIN and lower > STRICT_MARGIN
    summary = {
        "schema": "jope.h51.candidate-summary/v1",
        "candidate_id": f"H51-CONTROL-seed{args.seed}",
        "arm": "CONTROL",
        "seed": args.seed,
        "device": args.device,
        "adapter_sha256": adapter_sha,
        "family_median_gain": median,
        "family_bootstrap_lb95": lower,
        "family_bootstrap_ub95": upper,
        "n_families": len(family_gains),
        "inner_cv_mean_gain": float(result["inner_cv_selected_delta"]),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": 510_000 + args.seed,
        "strict_margin": STRICT_MARGIN,
        "accepted": accepted,
        "validation_scored": False,
        "validation_payloads_opened": False,
        "outer_rows_delivered": 0,
        "dope_sha256_before": before_dope,
        "dope_sha256_after": result["dope_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "fold_seed": FOLD_SEED,
        "candidate_checkpoint": selection["checkpoint"],
        "wall_seconds": result.get("wall_seconds"),
    }
    atomic_json(seed_dir / "candidate-summary.json", summary)
    atomic_json(seed_dir / "inner-family-gains.json", {
        "schema": "jope.h51.family-gains/v1",
        "candidate_id": summary["candidate_id"],
        "family_gains": family_gains,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("CONTROL",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h50/ckpt/round-785.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "jope-data/h51/runs")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    summary = run_control(args)
    print(json.dumps({key: summary[key] for key in (
        "arm", "seed", "adapter_sha256", "family_median_gain",
        "family_bootstrap_lb95", "family_bootstrap_ub95", "accepted",
    )}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
