"""Elo-objective fine-tuning of the deployed tabpfn_v3_regressor weights.

Single-GPU by default; becomes 3-node DDP when torchrun sets RANK/WORLD_SIZE.
Rank 0 owns the meters, metrics JSONL, evals, and checkpoints.

Measurement design (docs/elo_training.md): per-step REAL spliced-replay Elo
on the train split, paired ΔElo vs the frozen-base proxy (u_base) smoothed by
debiased EWMAs, a per-step test-split probe meter, and periodic tier-2
(6-view, full rows) evals on fixed val/test subsets replayed exactly by the
Rust fable-elo binary. Promotion/rollback gate on the val tier-2 Elo.

Usage (single GPU):
  python3 -m python.elo_finetune.train_elo --device cuda:0 --out-root ~/fable_god_run
Usage (3-node): see scripts/launch_3node.sh (torchrun sets the env).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .check_meter_parity import fable_elo_bin, write_extra
from .ckpt import CheckpointManager, restore_rng
from .elo_meter import EwmaState, SplicedEloMeter
from .eval_protocol import ProtocolRunner
from .losses import EloLoss, KlAnchor, combined_loss
from .metrics import JsonlWriter, new_run_id, write_anchors
from .model_io import load_regressor
from .pack import DEFAULT_PACK_DIR, OUR_ALGORITHM, FablePack, bucket_schedule, collate
from .schedule import PromotionGate, StagePlan, group_multiplier, group_of


@dataclass
class RunConfig:
    pack_dir: str = str(DEFAULT_PACK_DIR)
    out_root: str = str(Path.home() / "fable_god_run")
    device: str = "cuda:0"
    run_seed: int = 20260702
    seed_a: int = 1001
    micro_size: int = 24
    accum: int = 6
    # cell_budget (rows*B*F, the feature-stage memory) forces the heavy
    # high-row/high-feature shapes to B=1 (~10GB peak, was ~21GB at B=2 which
    # OOM'd once DDP buckets + resumed optimizer were added); row_budget
    # (rows*B, the ICL-stage memory) stays at the proven single-GPU-safe value
    # so low-feature shapes still batch. In DDP an OOM cannot be skipped (it
    # desyncs all-reduce), so these are sized with headroom on the 24GB cards.
    cell_budget: int = 245_000
    row_budget: int = 3500
    total_steps: int = 20000
    stage0_steps: int = 300
    stage1_steps: int = 1200
    tau0: float = 0.5
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_every: int = 500
    eval_subset: int = 128
    eval_subset_seed: int = 17
    test_probe_plans: int = 1
    ckpt_minutes: float = 15.0
    ewma_fast: float = 50.0
    ewma_slow: float = 400.0
    max_epochs: int = 10000
    resume: str = ""


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    for field, default in asdict(RunConfig()).items():
        flag = "--" + field.replace("_", "-")
        parser.add_argument(flag, type=type(default), default=None)
    args = parser.parse_args()
    cfg = RunConfig()
    if args.config:
        for k, v in json.loads(args.config.read_text()).items():
            setattr(cfg, k, v)
    for field in asdict(cfg):
        override = getattr(args, field)
        if override is not None:
            setattr(cfg, field, override)
    return cfg


def build_optimizer(model, plan: StagePlan, cfg: RunConfig):
    groups: dict[tuple[str, bool], list] = {}
    for name, param in model.named_parameters():
        group = group_of(name)
        decay = param.ndim > 1 and "norm" not in name.lower()
        groups.setdefault((group, decay), []).append(param)
    param_groups = [
        {
            "params": params,
            "lr": plan.head_lr_at(1) * group_multiplier(group),
            "weight_decay": cfg.weight_decay if decay else 0.0,
            "group_name": group,
        }
        for (group, decay), params in sorted(groups.items())
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(param_groups, fused=fused)


def set_stage(model, plan: StagePlan, stage: str) -> None:
    trainable = plan.trainable_groups(stage)
    for name, param in model.named_parameters():
        param.requires_grad = group_of(name) in trainable


def set_lrs(optimizer, plan: StagePlan, step: int) -> float:
    head_lr = plan.head_lr_at(step)
    for group in optimizer.param_groups:
        group["lr"] = head_lr * group_multiplier(group["group_name"])
    return head_lr


def ema_update(ema_model, model, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def eval_subsets(pack: FablePack, cfg: RunConfig) -> dict[str, np.ndarray]:
    """Fixed, seeded, test_rows-stratified subsets — identical across the run."""
    subsets = {}
    te_off = pack.section("test_row_offsets")
    for split in ("val", "test"):
        idx = pack.split_indices(split)
        rows = np.diff(te_off)[idx]
        order = idx[np.argsort(rows, kind="stable")]
        take = min(cfg.eval_subset, len(order))
        rng = np.random.default_rng([cfg.eval_subset_seed, take])
        strata = np.array_split(order, take)
        subsets[split] = np.sort(np.array([s[rng.integers(len(s))] for s in strata]))
    return subsets


class ExactEloEval:
    """Tier-2 on a fixed subset → fable-elo exact replay → (elo, rank)."""

    def __init__(self, pack: FablePack, pack_dir: Path, eval_dir: Path, subsets: dict[str, np.ndarray]):
        self.pack = pack
        self.pack_dir = pack_dir
        self.eval_dir = eval_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.subset_files = {}
        for split, indices in subsets.items():
            path = eval_dir / f"subset_{split}.txt"
            path.write_text("".join(pack.dataset_ids[i] + "\n" for i in indices))
            self.subset_files[split] = path
        self.subsets = subsets

    def replay(self, split: str, utilities: dict[int, float], step: int) -> dict:
        extra = self.eval_dir / f"step_{step}_{split}_ours.jsonl"
        write_extra(extra, self.pack, utilities)
        out = self.eval_dir / f"step_{step}_{split}_elo.json"
        subprocess.run(
            [
                str(fable_elo_bin()),
                "elo-json",
                "--cache",
                str(self.pack_dir / "utilities.jsonl"),
                "--extra",
                str(extra),
                "--datasets",
                str(self.subset_files[split]),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        ratings = json.loads(out.read_text())["ratings"]
        ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
        rank = 1 + [name for name, _ in ordered].index(OUR_ALGORITHM)
        return {"elo": ratings[OUR_ALGORITHM], "rank": rank, "ratings": ratings, "path": str(out)}


def tier2_utilities(
    runner: ProtocolRunner, pack: FablePack, indices: np.ndarray, rank: int, world: int
) -> dict[int, float]:
    shard = indices[rank::world]
    local = {int(i): -runner.tier2_rmse(pack.dataset(int(i))) for i in shard}
    if world == 1:
        return local
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local)
    merged: dict[int, float] = {}
    for part in gathered:
        merged.update(part)
    return merged


def main() -> None:
    cfg = parse_args()
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        cfg.device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
    device = torch.device(cfg.device)
    is_main = rank == 0

    out_root = Path(cfg.out_root).expanduser()
    pack_dir = Path(cfg.pack_dir).expanduser()
    pack = FablePack(pack_dir / "fable_real_v1.pack", preload=True)
    proxy = np.load(pack_dir / "fable_base_proxy_v1.npz")
    u_base = proxy["u_base"]
    if int(proxy["seed_a"]) != cfg.seed_a:
        raise ValueError(f"u_base seed_a {int(proxy['seed_a'])} != cfg {cfg.seed_a}")

    model = load_regressor(device=device).train()
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    plan = StagePlan(
        stage0_steps=cfg.stage0_steps, stage1_steps=cfg.stage1_steps, total_steps=cfg.total_steps
    )
    optimizer = build_optimizer(model, plan, cfg)
    runner = ProtocolRunner(model, device)
    runner_ema = ProtocolRunner(ema_model, device)
    borders = model.regression_borders.float().to(device)
    elo_loss = EloLoss(pack, device, tau0=cfg.tau0)
    kl_anchor = None
    try:
        kl_anchor = KlAnchor(pack_dir, pack, cfg.seed_a, device)
    except (FileNotFoundError, ValueError) as err:
        if is_main:
            print(f"KL anchor disabled: {err}")

    # No DDP wrapper. The model has params that are conditionally unused on
    # some batch shapes (and staged unlock freezes most), which makes DDP's
    # reducer fragile: find_unused=False errors on unused params, find_unused=
    # True issues mismatched collectives across ranks (NCCL "invalid usage").
    # Instead we accumulate grads locally over the micro-batches, then do ONE
    # explicit flattened bf16 all-reduce of the trainable grads per step — a
    # single collective, robust to frozen/conditionally-unused params, and a
    # good fit for the 1GbE link.
    runner.model = model

    def allreduce_grads():
        if not ddp:
            return
        # Reduce over ALL trainable params (identical set on every rank, since
        # set_stage is deterministic per step), zero-filling params whose grad
        # is None on this rank (conditionally unused here but maybe used on
        # another rank). This keeps the flattened layout identical across ranks.
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch._utils._flatten_dense_tensors(grads).to(torch.bfloat16)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat = flat.float() / world
        for p, synced in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
            if p.grad is None:
                p.grad = synced.clone()
            else:
                p.grad.copy_(synced)

    run_id = new_run_id()
    start_step, start_epoch, plan_cursor = 0, 0, 0
    gate = PromotionGate()
    ewma_fast = EwmaState(cfg.ewma_fast)
    ewma_slow = EwmaState(cfg.ewma_slow)
    ewma_test = EwmaState(cfg.ewma_fast)
    grad_ewma = EwmaState(200.0)

    if cfg.resume:
        payload = CheckpointManager.load(Path(cfg.resume).expanduser(), map_location="cpu")
        model.load_state_dict(payload["model"])
        ema_model.load_state_dict(payload["ema_model"])
        optimizer.load_state_dict(payload["optimizer"])
        gate.load_state_dict(payload["gate"])
        ewma_fast.load_state_dict(payload["ewma_fast"])
        ewma_slow.load_state_dict(payload["ewma_slow"])
        ewma_test.load_state_dict(payload["ewma_test"])
        grad_ewma.load_state_dict(payload["grad_ewma"])
        start_step = payload["step"]
        start_epoch = payload["epoch"]
        plan_cursor = payload["plan_cursor"]
        # Keep the FRESH run_id (line above) rather than the checkpoint's: the
        # metrics files are append-mode and seq restarts at 0 each launch, so
        # reusing the old run_id makes a live TUI drop every new line as a
        # "duplicate seq" (its dedup high-water mark for that run_id is already
        # past). A fresh run_id renders as a new run segment with a clean break.
        if is_main:
            restore_rng(payload["rng"])

    train_meter = test_meter = None
    writer_train = writer_eval = None
    exact_eval = None
    test_plans: list = []
    test_plan_ptr = 0
    if is_main:
        train_idx = pack.split_indices("train").tolist()
        test_idx = pack.split_indices("test").tolist()
        train_meter = SplicedEloMeter(pack, "train", {i: float(u_base[i]) for i in train_idx})
        test_meter = SplicedEloMeter(pack, "test", {i: float(u_base[i]) for i in test_idx})
        if cfg.resume:
            train_meter.load_state_dict(payload["train_meter"])
            test_meter.load_state_dict(payload["test_meter"])
        metrics_dir = out_root / "metrics"
        writer_train = JsonlWriter(metrics_dir / "train_metrics.jsonl", run_id)
        writer_eval = JsonlWriter(metrics_dir / "eval_metrics.jsonl", run_id)
        test_plans = bucket_schedule(pack, "test", run_seed=cfg.seed_a, epoch=0, micro_size=cfg.micro_size)
        test_plan_ptr = start_step % max(1, len(test_plans))
    subsets = eval_subsets(pack, cfg)
    if is_main:
        exact_eval = ExactEloEval(pack, pack_dir, out_root / "eval", subsets)
    ckpts = CheckpointManager(out_root / "ckpt")

    def checkpoint_payload(step: int, epoch: int, cursor: int) -> dict:
        return {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gate": gate.state_dict(),
            "ewma_fast": ewma_fast.state_dict(),
            "ewma_slow": ewma_slow.state_dict(),
            "ewma_test": ewma_test.state_dict(),
            "grad_ewma": grad_ewma.state_dict(),
            "train_meter": train_meter.state_dict() if train_meter else None,
            "test_meter": test_meter.state_dict() if test_meter else None,
            "step": step,
            "epoch": epoch,
            "plan_cursor": cursor,
            "run_id": run_id,
            "config": asdict(cfg),
        }

    def run_eval(step: int, first: bool, epoch_now: int, cursor_now: int) -> None:
        results = {}
        for split in ("val", "test"):
            utils = tier2_utilities(runner_ema, pack, subsets[split], rank, world)
            if is_main:
                results[split] = exact_eval.replay(split, utils, step)
        if not is_main:
            return
        val, test = results["val"], results["test"]
        verdict = gate.observe(val["elo"], train_meter.our_rating)
        if first:
            base_elo = {"train": train_meter.our_rating, "val": val["elo"], "test": test["elo"]}
            base_rank = {"train": 0, "val": val["rank"], "test": test["rank"]}
            leaderboard = [
                {"algorithm": a, "elo": e} for a, e in sorted(val["ratings"].items(), key=lambda kv: -kv[1])
            ]
            write_anchors(
                out_root / "metrics" / "anchors.json",
                run_id,
                base_elo,
                base_rank,
                leaderboard,
                {"train": 1629, "val": 340, "test": 353},
            )
            anchors_path = out_root / "metrics" / "anchors.json"
            writer_eval.event(step, f"anchors written: {anchors_path}")
        base = json.loads((out_root / "metrics" / "anchors.json").read_text())["base_elo"]
        for split, res in results.items():
            writer_eval.write(
                "eval",
                step,
                {
                    "split": split,
                    "elo_exact": res["elo"],
                    "delta_vs_base": res["elo"] - base[split],
                    "elo_rank": res["rank"],
                    "n_datasets": len(subsets[split]),
                    "replay_path": res["path"],
                    "promoted": bool(verdict["promote"]) and split == "val",
                    "rollback": bool(verdict["rollback"]) and split == "val",
                    "ema": True,
                },
            )
        if verdict["promote"]:
            path = ckpts.save(checkpoint_payload(step, epoch_now, cursor_now), step, best=True)
            writer_eval.write("ckpt", step, {"path": str(path), "sha256": None, "best": True})
            writer_eval.event(step, f"PROMOTED at val elo ewma {verdict['ewma']:.2f}")
        if verdict["rollback"]:
            (out_root / "control").mkdir(exist_ok=True)
            (out_root / "control" / "rollback.request").write_text(str(step))
            writer_eval.event(step, "ROLLBACK requested", level="error")

    step = start_step
    epoch = start_epoch
    cursor = plan_cursor
    last_ckpt_time = time.time()
    current_stage = ""

    if step == 0:
        set_stage(model, plan, "stage0")
        current_stage = "stage0"
        run_eval(0, first=True, epoch_now=epoch, cursor_now=cursor)
    else:
        current_stage = plan.stage(step)
        set_stage(model, plan, current_stage)

    stop = False
    while not stop and epoch < cfg.max_epochs:
        plans = bucket_schedule(
            pack,
            "train",
            cfg.run_seed,
            epoch,
            cfg.micro_size,
            cell_budget=cfg.cell_budget,
            row_budget=cfg.row_budget,
        )
        sync_size = world * cfg.accum
        n_groups = len(plans) // sync_size
        for g in range(cursor, n_groups):
            step += 1
            if step > cfg.total_steps:
                stop = True
                break
            stage = plan.stage(step)
            if stage != current_stage:
                set_stage(model, plan, stage)
                current_stage = stage
                if is_main:
                    writer_train.event(step, f"stage transition -> {stage}")
            head_lr = set_lrs(optimizer, plan, step)
            lam_nll, lam_kl = plan.lambdas(step)
            if kl_anchor is None:
                lam_kl = 0.0

            group_plans = plans[g * sync_size : (g + 1) * sync_size]
            my_plans = group_plans[rank * cfg.accum : (rank + 1) * cfg.accum]
            step_start = time.time()
            batch_utils: dict[int, float] = {}
            agg = {
                "loss": 0.0,
                "loss_elo": 0.0,
                "loss_nll": 0.0,
                "loss_kl": 0.0,
                "entropy_bar": 0.0,
                "winrate_mean": 0.0,
            }
            rows_done = 0
            for micro in my_plans:
                # Grads accumulate locally across micro-batches; one explicit
                # all-reduce follows the loop. In ddp mode an OOM must abort
                # (a skipped micro would desync the collective); budgets are
                # sized so it doesn't happen.
                try:
                    batch = collate(pack, micro, cfg.seed_a)
                    out = runner.tier1(batch, with_grad=True)
                    loss, lm = combined_loss(
                        elo_loss, out, micro.dataset_indices, borders, lam_nll, lam_kl, kl_anchor
                    )
                    loss = loss / len(my_plans)
                    loss.backward()
                except torch.OutOfMemoryError:
                    if ddp:
                        raise  # skipping a backward would desync all-reduce
                    out = loss = None  # noqa: F841 - drop refs before empty_cache
                    torch.cuda.empty_cache()
                    if is_main:
                        writer_train.event(
                            step,
                            f"OOM: skipped micro ctx={micro.ctx} qry={micro.qry} "
                            f"F={micro.n_features} B={len(micro.dataset_indices)}",
                            level="warn",
                        )
                    continue
                for i, u in zip(micro.dataset_indices, out.utilities_f64()):
                    batch_utils[int(i)] = float(u)
                for key in agg:
                    if key == "loss":
                        agg[key] += float(loss.detach()) * len(my_plans)
                    else:
                        agg[key] += lm[key]
                rows_done += batch.x.shape[0] * batch.x.shape[1]
            for key in agg:
                agg[key] /= max(1, len(my_plans))

            # sync grads across ranks (one flattened bf16 all-reduce), then all
            # ranks clip identically off the global gradient.
            allreduce_grads()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            g_hat, _ = grad_ewma.update(min(grad_norm, 1e6))
            skip = not math.isfinite(grad_norm) or (grad_ewma.t > 50 and grad_norm > 10.0 * g_hat)
            if skip:
                optimizer.zero_grad(set_to_none=True)
                if is_main:
                    writer_train.event(step, f"skip step: grad_norm={grad_norm:.2f}", level="warn")
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema_update(ema_model, model, cfg.ema_decay)

            if ddp:
                gathered: list = [None] * world
                dist.all_gather_object(gathered, batch_utils)
                if is_main:
                    for part in gathered:
                        batch_utils.update(part)

            if is_main:
                meter_out = train_meter.update(batch_utils)
                delta = meter_out["delta_raw"]
                g_rate = train_meter.exchange_rate()
                m_fast, se_fast = ewma_fast.update(delta)
                m_slow, se_slow = ewma_slow.update(delta)

                probe = test_plans[test_plan_ptr % len(test_plans)]
                test_plan_ptr += 1
                tbatch = collate(pack, probe, cfg.seed_a)
                tout = runner.tier1(tbatch, with_grad=False)
                tutils = {int(i): float(u) for i, u in zip(probe.dataset_indices, tout.utilities_f64())}
                tmeter_out = test_meter.update(tutils)
                tg = test_meter.exchange_rate()
                tm_fast, tse_fast = ewma_test.update(tmeter_out["delta_raw"])

                step_time = time.time() - step_start
                writer_train.write(
                    "train",
                    step,
                    {
                        "phase": stage,
                        "lr": head_lr,
                        "loss": agg["loss"],
                        "loss_elo": agg["loss_elo"],
                        "loss_nll": agg["loss_nll"],
                        "loss_kl": agg["loss_kl"],
                        "grad_norm": grad_norm,
                        "n_datasets": len(batch_utils),
                        "elo_train_spliced": meter_out["elo_spliced"],
                        "delta_raw": delta,
                        "delta_elo_raw": meter_out["delta_elo_raw"],
                        "delta_elo_fast": g_rate * m_fast,
                        "delta_elo_fast_band": 1.96 * g_rate * se_fast,
                        "delta_elo_slow": g_rate * m_slow,
                        "delta_elo_slow_band": 1.96 * g_rate * se_slow,
                        "elo_test_spliced": tmeter_out["elo_spliced"],
                        "delta_elo_test_fast": tg * tm_fast,
                        "delta_elo_test_fast_band": 1.96 * tg * tse_fast,
                        "utility_mean": float(np.mean(list(batch_utils.values()))),
                        "winrate_mean": agg["winrate_mean"],
                        "entropy_bar": agg["entropy_bar"],
                        "rows_per_s": rows_done / max(step_time, 1e-6),
                        "sync_ms": None,
                        "per_rank_ms": None,
                    },
                )

            if step % cfg.eval_every == 0:
                run_eval(step, first=False, epoch_now=epoch, cursor_now=g + 1)
                last_ckpt_time = time.time()
                if is_main:
                    path = ckpts.save(checkpoint_payload(step, epoch, g + 1), step)
                    writer_eval.write("ckpt", step, {"path": str(path), "sha256": None, "best": False})
            elif is_main and time.time() - last_ckpt_time > cfg.ckpt_minutes * 60:
                ckpts.save(checkpoint_payload(step, epoch, g + 1), step)
                last_ckpt_time = time.time()
        cursor = 0
        epoch += 1

    if is_main:
        ckpts.save(checkpoint_payload(step, epoch, 0), step)
        writer_train.event(step, "training complete")
        writer_train.close()
        writer_eval.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
