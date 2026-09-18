"""Trajectory backtest: prove (or refute) a candidate fitness BEFORE any run.

We own a labeled overfit trajectory (base → peak ≈ step 850-1000 → overfit
1828). A trustworthy fitness, computed on FRESH episodes of NEVER-TRAINED
tasks, must (a) put its argmax at the known peak, (b) decline strictly after,
(c) correlate with the known held-out Elo trajectory (Spearman ρ ≥ 0.8).
The NEGATIVE CONTROL — the old fixed-seed fitness (seed-A winrate vs frozen
archive competitors on trained datasets) — must keep rising past the peak,
proving the backtest discriminates.

Usage (on xbabe1, where the checkpoints live):
  python3 -m python.elo_finetune.backtest_metric --device cuda:0 \
      --tasks 400 --seed 20260703
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ckpt import CheckpointManager
from .elo_meter import outcomes
from .episodes import list_synth_tasks, prepare_episodes
from .eval_protocol import ProtocolRunner
from .fitness_v2 import evaluate
from .model_io import CONFIG_NAME, default_weights_dir, load_regressor
from .pack import DEFAULT_PACK_DIR, FablePack, bucket_schedule, collate
from .paths import import_regressor_train

# Held-out ground truth: test-split spliced proxy Elo at each checkpoint step
# (from the run's archived metrics; base = meter level at init).
GROUND_TRUTH = {"base": 1312.4, "1000": 1317.5, "1400": 1302.0,
                "1500": 1300.6, "1828": 1295.0}
CKPT_STEPS = {"base": 0, "1000": 1000, "1400": 1400, "1500": 1500, "1828": 1828}


def available_tags(ckpt_dir: Path) -> list[str]:
    tags = ["base"]
    for tag in CKPT_STEPS:
        if tag == "base":
            continue
        name = "best.ckpt" if tag == "1000" else f"step_{tag}.ckpt"
        if (ckpt_dir / name).is_file():
            tags.append(tag)
    return tags


def build_checkpoint(tag: str, ckpt_dir: Path, device):
    if tag == "base":
        return load_regressor(device=device)
    train_mod = import_regressor_train()
    cfg = json.loads((default_weights_dir() / CONFIG_NAME).read_text())
    model = train_mod.get_architecture(train_mod.TabPFNV3Config.from_dict(cfg))
    name = "best.ckpt" if tag == "1000" else f"step_{tag}.ckpt"
    payload = CheckpointManager.load(ckpt_dir / name, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)  # RAW weights
    return model.to(device).eval()


def old_fitness(runner: ProtocolRunner, pack: FablePack, n_datasets: int) -> float:
    """The discredited metric: seed-A fixed-subset winrate vs frozen archive
    competitors on TRAINED datasets. Negative control: must rise past the peak."""
    plans = bucket_schedule(pack, "train", run_seed=1, epoch=0, micro_size=16)
    sel, covered = [], set()
    for p in plans:
        sel.append(p)
        covered.update(p.dataset_indices)
        if len(covered) >= n_datasets:
            break
    wrs = []
    with torch.no_grad():
        for p in sel:
            batch = collate(pack, p, seed_a=1001)
            utils = runner.tier1(batch, with_grad=False).utilities_f64()
            for i, u in zip(p.dataset_indices, utils):
                _, cu = pack.competitor_utilities(int(i))
                wrs.append(float(outcomes(float(u), cu).mean()))
    return float(np.mean(wrs))


def spearman(xs: list[float], ys: list[float]) -> float:
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", type=int, default=400)
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path.home() / "fable_god_run" / "ckpt")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--old-fitness-datasets", type=int, default=240)
    parser.add_argument("--real-frac", type=float, default=0.4,
                        help="fraction of episodes drawn as real column-as-target "
                             "(Tier C, pre-registered >=30% in the design brief)")
    parser.add_argument("--narrow", action="store_true",
                        help="F_narrow variant: TRUE-target episodes from real "
                             "TRAIN-split tasks, fresh subsets/warps (the "
                             "specialization signal; expected to keep rising)")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "fable_god_run" / "eval_full" / "backtest_v2.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    t0 = time.time()
    from .episodes import list_real_tasks

    pack = FablePack(args.pack_dir / "fable_real_v1.pack", preload=True)
    if args.narrow:
        train_ids = [pack.dataset_ids[i] for i in pack.split_indices("train")]
        rng = np.random.default_rng(args.seed)
        take = min(args.tasks, len(train_ids))
        chosen = sorted(rng.choice(train_ids, size=take, replace=False).tolist())
        print(f"drawing {take} TRUE-target episodes from real TRAIN tasks (narrow) ...")
        episodes = prepare_episodes(chosen, args.per_task, args.seed, args.workers)
    else:
        n_synth = int(args.tasks * (1.0 - args.real_frac))
        n_real = args.tasks - n_synth
        print(f"drawing episodes: {n_synth} synth + {n_real} real column-as-target ...")
        episodes = prepare_episodes(
            list_synth_tasks(limit=n_synth, seed=args.seed),
            args.per_task, args.seed, args.workers)
        if n_real:
            episodes += prepare_episodes(
                list_real_tasks(limit=n_real, seed=args.seed),
                args.per_task, args.seed + 1, args.workers, col_as_target=True)
    buckets = {b: sum(1 for e in episodes if e.bucket == b)
               for b in ("easy", "medium", "hard", "noisy")}
    print(f"  {len(episodes)} episodes ready in {time.time()-t0:.0f}s, buckets={buckets}")

    tags_present = available_tags(args.ckpt_dir)
    print(f"checkpoints present: {tags_present}")
    base_state_cpu = None
    results, base_margins = {}, None
    for tag in tags_present:
        t0 = time.time()
        model = build_checkpoint(tag, args.ckpt_dir, device)
        if tag == "base":
            base_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            sp_sq = 0.0
        else:
            sp_sq = 0.0
            n = 0
            with torch.no_grad():
                for k, p in model.state_dict().items():
                    d = p.cpu() - base_state_cpu[k]
                    sp_sq += float((d.double() ** 2).sum())
                    n += p.numel()
            sp_sq /= max(n, 1)
        runner = ProtocolRunner(model, device)
        report = evaluate(runner, episodes, theta_sp_sq=sp_sq)
        control = old_fitness(runner, pack, args.old_fitness_datasets)

        paired = None
        if tag == "base":
            base_margins = report.per_episode_margin
        elif base_margins and len(report.per_episode_margin) == len(base_margins):
            deltas = np.array(report.per_episode_margin) - np.array(base_margins)
            paired = {"mean_delta_margin": float(deltas.mean()),
                      "se": float(deltas.std(ddof=1) / np.sqrt(len(deltas)))}
        results[tag] = {
            "fitness": report.fitness, "loss_task": report.loss_task,
            "mean_margin": report.mean_margin, "mean_winrate_vs_zoo": report.mean_winrate,
            "sp_sq": sp_sq, "n_episodes": report.n_episodes,
            "per_bucket_loss": report.per_bucket_loss,
            "old_fitness_seedA_winrate": control,
            "paired_vs_base": paired,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  {tag}: fitness={report.fitness:+.4f} margin={report.mean_margin:+.4f} "
              f"wr_zoo={report.mean_winrate:.3f} old_fitness={control:.4f} "
              f"({results[tag]['seconds']}s)")
        del model, runner
        torch.cuda.empty_cache()

    tags = tags_present
    fit = [results[t]["fitness"] for t in tags]
    old = [results[t]["old_fitness_seedA_winrate"] for t in tags]
    truth = [GROUND_TRUTH[t] for t in tags]
    verdict = {
        "argmax_tag": tags[int(np.argmax(fit))],
        "argmax_expected": "1000 (nearest saved point to the ~850 peak)",
        "declines_after_peak": bool(all(
            fit[i] > fit[i + 1] for i in range(tags.index("1000"), len(tags) - 1))),
        "spearman_fitness_vs_truth": spearman(fit, truth),
        "spearman_oldfitness_vs_truth": spearman(old, truth),
        "old_fitness_keeps_rising": bool(all(old[i] < old[i + 1] for i in range(len(old) - 1))),
    }
    verdict["PASS"] = bool(
        verdict["argmax_tag"] == "1000"
        and verdict["declines_after_peak"]
        and verdict["spearman_fitness_vs_truth"] >= 0.8
        and verdict["spearman_oldfitness_vs_truth"] < 0.5
    )
    payload = {"config": vars(args) | {"ckpt_dir": str(args.ckpt_dir),
                                       "pack_dir": str(args.pack_dir),
                                       "out": str(args.out)},
               "ground_truth_test_spliced_elo": GROUND_TRUTH,
               "episode_buckets": buckets,
               "results": results, "verdict": verdict}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
