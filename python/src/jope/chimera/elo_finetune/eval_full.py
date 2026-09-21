"""Full-sweep held-out evaluation: authoritative tier-2 (6-view ensemble,
full rows, exact Rust replay) on the COMPLETE val (340) + test (353) real
datasets, sharded across nodes. Use it to pick/confirm the deployment
checkpoint beyond the noisy 128-subset gate.

--ckpt base           -> frozen deployed weights (the baseline)
--ckpt PATH --which ema|model  -> that checkpoint's EMA (default) or raw weights

torchrun --nnodes=N --nproc-per-node=1 --node-rank=R --rdzv-... \
  -m python.elo_finetune.eval_full --ckpt ~/fable_god_run/ckpt/best.ckpt --tag best
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .check_meter_parity import fable_elo_bin, write_extra
from .ckpt import CheckpointManager
from .eval_protocol import ProtocolRunner
from .model_io import CONFIG_NAME, default_weights_dir, load_regressor
from .pack import DEFAULT_PACK_DIR, OUR_ALGORITHM, FablePack
from .paths import import_regressor_train


def build_model(ckpt: str, which: str, device):
    if ckpt == "base":
        return load_regressor(device=device), 0
    train_mod = import_regressor_train()
    cfg = json.loads((default_weights_dir() / CONFIG_NAME).read_text())
    model = train_mod.get_architecture(train_mod.TabPFNV3Config.from_dict(cfg))
    payload = CheckpointManager.load(Path(ckpt).expanduser(), map_location="cpu")
    model.load_state_dict(payload[which], strict=True)
    return model.to(device).eval(), int(payload.get("step", 0))


def sweep_split(runner, pack, split, rank, world):
    idx = pack.split_indices(split)
    shard = idx[rank::world]
    local = {}
    for i in shard.tolist():
        local[int(i)] = -runner.tier2_rmse(pack.dataset(int(i)))
    if world == 1:
        return local
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    merged = {}
    for part in gathered:
        merged.update(part)
    return merged


def replay_full(pack, pack_dir, out_dir, tag, split, utils, step):
    """Splice our full-sweep utilities into all datasets of `split` and run
    the exact Rust replay over the full split."""
    idx = pack.split_indices(split)
    datasets = out_dir / f"{tag}_{split}_datasets.txt"
    datasets.write_text("".join(pack.dataset_ids[int(i)] + "\n" for i in idx))
    extra = out_dir / f"{tag}_{split}_ours.jsonl"
    write_extra(extra, pack, utils)
    out = out_dir / f"{tag}_{split}_elo.json"
    subprocess.run(
        [
            str(fable_elo_bin()),
            "elo-json",
            "--cache",
            str(pack_dir / "utilities.jsonl"),
            "--extra",
            str(extra),
            "--datasets",
            str(datasets),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    ratings = json.loads(out.read_text())["ratings"]
    ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
    rank_pos = 1 + [n for n, _ in ordered].index(OUR_ALGORITHM)
    return {
        "elo": ratings[OUR_ALGORITHM],
        "rank": rank_pos,
        "n": len(idx),
        "above": [n for n, _ in ordered[max(0, rank_pos - 3) : rank_pos - 1]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--which", default="ema_model", choices=["ema_model", "model"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "fable_god_run" / "eval_full")
    args = parser.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    pack = FablePack(args.pack_dir / "fable_real_v1.pack", preload=True)
    model, step = build_model(args.ckpt, args.which, device)
    runner = ProtocolRunner(model, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = {"tag": args.tag, "ckpt": args.ckpt, "which": args.which, "step": step}
    for split in ("val", "test"):
        utils = sweep_split(runner, pack, split, rank, world)
        if is_main:
            result[split] = replay_full(pack, args.pack_dir, args.out_dir, args.tag, split, utils, step)
    if is_main:
        result["seconds"] = round(time.time() - started, 1)
        (args.out_dir / f"{args.tag}_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
