"""Episode sources for evolution — every node can produce identical episodes
from seeds alone (CRN across the fleet).

- Real (F_narrow): TRUE-target episodes from train-split tasks in the local
  fable_real_v1.pack (bit-exact CSV parity, present on every node).
- Synth realtime (guard + grad-arm fuel): gen_synth_regression.build_one
  fresh in-memory tasks with difficulty targeting.
- Col-as-target real (guard component): decontaminated real-marginal tasks.

A worker prepares episodes with a multiprocessing Pool (zoo re-fit per
episode) and keeps a prefetch queue so the GPU never waits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .episodes import Episode, column_as_target, difficulty_bucket, draw_episode, score_zoo, zoo_r2_best
from .pack import FablePack


def _find_generator() -> Path:
    for candidate in (
        Path.home() / "fable_god_run" / "code" / "gen_synth_regression.py",
        Path("/home/ubuntu/remote_super/tabby/gen_synth_regression.py"),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("gen_synth_regression.py not found")


def _gen_module():
    path = _find_generator()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    import gen_synth_regression

    return gen_synth_regression


class RealTaskSource:
    """Pooled X/y for a pack task — mirrors episodes.load_task exactly."""

    def __init__(self, pack_dir: Path, split: str = "train"):
        self.pack = FablePack(pack_dir / "fable_real_v1.pack", preload=False)
        self.task_ids = [self.pack.dataset_ids[i]
                         for i in self.pack.split_indices(split)]
        self._index = {t: i for t, i in zip(
            self.task_ids, self.pack.split_indices(split).tolist())}

    def load(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        view = self.pack.dataset(self._index[task])
        X = np.concatenate([view.Xtr, view.Xte], axis=0)
        y = np.concatenate([view.ytr, view.yte], axis=0)
        return np.ascontiguousarray(X), np.ascontiguousarray(y)


# ---- worker-side builders (run inside a multiprocessing Pool) --------------

_POOL_STATE: dict = {}


def _pool_init(pack_dir: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _POOL_STATE["real"] = RealTaskSource(Path(pack_dir), "train")
    _POOL_STATE["val"] = None  # constructed lazily for the val gate


def _finish(ep: Episode | None) -> Episode | None:
    if ep is None:
        return None
    ep.zoo_rmse = score_zoo(ep)
    if sum(np.isfinite(v) for v in ep.zoo_rmse.values()) < 3:
        return None
    ep.bucket = difficulty_bucket(zoo_r2_best(ep))
    return ep


def build_real(spec: tuple) -> Episode | None:
    """spec = ("real", task_id, seed) — TRUE-target train-task episode."""
    _, task, seed = spec
    try:
        X, y = _POOL_STATE["real"].load(task)
        if X.shape[0] < 96:
            return None
        return _finish(draw_episode(task, seed, X, y))
    except Exception:
        return None


def build_val(spec: tuple) -> Episode | None:
    """spec = ("val", task_id, seed) — Thresholdout gate episodes only."""
    _, task, seed = spec
    try:
        if _POOL_STATE["val"] is None:
            _POOL_STATE["val"] = RealTaskSource(
                Path(os.environ["FABLE_PACK_DIR"]), "val")
        X, y = _POOL_STATE["val"].load(task)
        if X.shape[0] < 96:
            return None
        return _finish(draw_episode(task, seed, X, y))
    except Exception:
        return None


def build_synth(spec: tuple) -> Episode | None:
    """spec = ("synth", gen_seed, ep_seed, profile) — realtime generation."""
    _, gen_seed, ep_seed, profile = spec
    try:
        gen = _gen_module()
        task = gen.build_one(int(gen_seed), profile=profile, max_features=192)
        if task is None:
            return None
        X = np.concatenate([np.asarray(task["Xtr"], np.float32),
                            np.asarray(task["Xte"], np.float32)], axis=0)
        y = np.concatenate([np.asarray(task["ytr"], np.float32),
                            np.asarray(task["yte"], np.float32)])
        if X.shape[0] < 96:
            return None
        return _finish(draw_episode(f"synthrt_{gen_seed:016x}", ep_seed, X, y))
    except Exception:
        return None


def build_coltarget(spec: tuple) -> Episode | None:
    """spec = ("coltar", task_id, seed) — real marginals, new label function."""
    _, task, seed = spec
    try:
        X, y = _POOL_STATE["real"].load(task)
        converted = column_as_target(X, y, seed)
        if converted is None:
            return None
        X2, y2 = converted
        if X2.shape[0] < 96 or X2.shape[1] < 2:
            return None
        return _finish(draw_episode(task + "_ct", seed, X2, y2))
    except Exception:
        return None


_BUILDERS = {"real": build_real, "val": build_val,
             "synth": build_synth, "coltar": build_coltarget}


def build_episode(spec: tuple) -> Episode | None:
    return _BUILDERS[spec[0]](spec)


def episode_specs_narrow(task_ids: list[str], gen: int, n: int,
                         shard: tuple[int, int]) -> list[tuple]:
    """Deterministic CRN spec list for a generation's F_narrow episodes;
    shard = (offset, world) partitions the SAME global list per node."""
    rng = np.random.default_rng([gen, 5001])
    tasks = rng.choice(task_ids, size=n, replace=n > len(task_ids))
    specs = [("real", str(t), int(gen) * 1_000_003 + k)
             for k, t in enumerate(tasks)]
    return specs[shard[0]::shard[1]]


def episode_specs_guard(task_ids: list[str], gen: int, n: int,
                        shard: tuple[int, int]) -> list[tuple]:
    """60% realtime synth + 40% col-as-target real."""
    rng = np.random.default_rng([gen, 7001])
    specs: list[tuple] = []
    n_synth = int(0.6 * n)
    for k in range(n_synth):
        specs.append(("synth", int(rng.integers(2**62)), gen * 900_001 + k,
                      "balanced"))
    tasks = rng.choice(task_ids, size=n - n_synth, replace=False)
    for k, t in enumerate(tasks):
        specs.append(("coltar", str(t), gen * 800_009 + k))
    return specs[shard[0]::shard[1]]


class EpisodeProducer:
    """Multiprocessing producer with an in-order result list (CRN pairing
    requires stable episode identity/order across candidates and nodes)."""

    def __init__(self, pack_dir: Path, workers: int = 48):
        from multiprocessing import Pool

        self.pool = Pool(workers, initializer=_pool_init,
                         initargs=(str(pack_dir),))

    def prepare(self, specs: list[tuple]) -> list[Episode]:
        episodes = self.pool.map(build_episode, specs, chunksize=2)
        return [e for e in episodes if e is not None]

    def close(self) -> None:
        self.pool.terminate()
