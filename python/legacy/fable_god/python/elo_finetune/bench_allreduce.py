"""All-reduce microbenchmark for the 3-node 1GbE fleet.

Launch under the same torchrun config as training. Go/no-go: median < 2.5s
for the 58.28M-element bf16 tensor (the model's gradient payload).

  torchrun --nnodes=3 --nproc-per-node=1 --node-rank=$R \
    --rdzv-backend=c10d --rdzv-endpoint=192.168.68.86:29400 --rdzv-id=bench \
    -m python.elo_finetune.bench_allreduce
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    torch.cuda.set_device(device)
    payload = torch.randn(58_274_944, dtype=torch.bfloat16, device=device)

    for _ in range(3):
        dist.all_reduce(payload)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = time.perf_counter()
        dist.all_reduce(payload)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times.sort()
    median = times[len(times) // 2]
    if rank == 0:
        bus_gb = 2 * (dist.get_world_size() - 1) / dist.get_world_size()
        bus_gb *= payload.numel() * payload.element_size() / 1e9
        print(json.dumps({
            "world": dist.get_world_size(),
            "payload_mb": round(payload.numel() * payload.element_size() / 1e6, 1),
            "median_s": round(median, 3),
            "min_s": round(times[0], 3),
            "max_s": round(times[-1], 3),
            "effective_gbps": round(bus_gb / median * 8, 2),
            "go": median < 2.5,
        }))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
