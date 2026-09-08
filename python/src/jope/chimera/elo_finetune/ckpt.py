"""Atomic checkpointing (tabby_cluster pattern: tmp + fsync + os.replace).

Contents: fp32 model, EMA model, optimizer, meter/EWMA/gate states, RNG,
step/epoch/plan cursor, config, pack sha. Policy: last.ckpt overwritten
atomically, step_N keep-last-3, best.ckpt on promotion.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, keep_step_ckpts: int = 3):
        self.dir = ckpt_dir
        self.keep = keep_step_ckpts
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict, step: int, best: bool = False) -> Path:
        payload = dict(payload)
        payload["rng"] = rng_states()
        save_atomic(payload, self.dir / "last.ckpt")
        step_path = self.dir / f"step_{step}.ckpt"
        save_atomic(payload, step_path)
        if best:
            save_atomic(payload, self.dir / "best.ckpt")
        self._prune()
        return step_path

    def _prune(self) -> None:
        step_ckpts = sorted(
            self.dir.glob("step_*.ckpt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for old in step_ckpts[: -self.keep]:
            old.unlink(missing_ok=True)

    @staticmethod
    def load(path: Path, map_location="cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)
