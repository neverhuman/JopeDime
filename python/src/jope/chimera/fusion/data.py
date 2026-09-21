"""Quant dataset loader for fusion training.

Format (verified): ~/remote_super/quant/{binary,regression,...}/<hash>/ with
meta.json + headerless train.csv/test.csv (values min-max scaled to [0,1], target
= last column). Splits are predefined. `binary_central_C/` has no meta.json.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import pandas as pd

QUANT_ROOT = os.environ.get("FUSION_QUANT_ROOT", os.path.expanduser("~/remote_super/quant"))


@dataclass
class QuantDataset:
    hash: str
    task: str  # "binary" | "regression" | "multiclass"
    Xtr: np.ndarray  # (Ntr, F) float32
    ytr: np.ndarray  # (Ntr,)  int (clf) or float32 (reg)
    Xte: np.ndarray
    yte: np.ndarray
    n_features: int
    n_classes: int  # 0 for regression
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return self.Xtr.shape[0]


def load_dataset(d: str) -> QuantDataset:
    d = d.rstrip("/")
    h = os.path.basename(d)
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if not isinstance(meta, dict):  # some meta.json parse to a bare string
        meta = {}
    # Robust: target is ALWAYS the final column; features = everything else.
    tr = pd.read_csv(f"{d}/train.csv", header=None)
    te = pd.read_csv(f"{d}/test.csv", header=None)
    Xtr = tr.iloc[:, :-1].to_numpy("float32")
    Xte = te.iloc[:, :-1].to_numpy("float32")
    ytr = tr.iloc[:, -1].to_numpy()
    yte = te.iloc[:, -1].to_numpy()
    task = meta.get("task_type") or ("regression" if not np.array_equal(ytr, ytr.astype(int)) else "binary")
    ts = meta.get("target_scaling", {})
    if not isinstance(ts, dict):
        ts = {}
    n_classes = int(ts.get("n_classes", 0)) if ts.get("mode") == "classification" else 0
    if task == "regression":
        ytr = ytr.astype("float32")
        yte = yte.astype("float32")
    else:
        ytr = ytr.astype("int64")
        yte = yte.astype("int64")
        n_classes = n_classes or int(max(ytr.max(), yte.max()) + 1)
    return QuantDataset(h, task, Xtr, ytr, Xte, yte, Xtr.shape[1], n_classes, meta)


def list_datasets(
    category: str, limit: int | None = None, shuffle_seed: int | None = None, root: str = QUANT_ROOT
) -> list[str]:
    dirs = sorted(glob(os.path.join(root, category, "*", "")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "train.csv"))]
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(dirs)
    return dirs[:limit] if limit else dirs


# ---- gate features: dataset descriptor (generalizable — never dataset identity) ----
GATE_META_DIM = 8


def gate_meta_vector(ds) -> np.ndarray:
    """Generalizable dataset descriptor for the reliability gate. NO identity.
    Works for QuantDataset and PackedTask (no .meta)."""
    m = getattr(ds, "meta", {}) or {}
    if not isinstance(m, dict):  # some meta.json parse to a bare string
        m = {}
    if ds.task != "regression":
        yall = np.concatenate([ds.ytr, ds.yte])
        imbalance = float(min((yall == c).mean() for c in np.unique(yall)))
    else:
        imbalance = 0.0
    if ds.task == "regression":
        y = np.concatenate([ds.ytr, ds.yte]).astype(np.float64)
        skew = float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-9))
        spread = float(np.subtract(*np.percentile(y, [90, 10])))
    else:
        skew = spread = 0.0
    return np.array(
        [
            math.log1p(ds.n_train),
            math.log1p(ds.Xte.shape[0]),
            math.log1p(ds.n_features),
            ds.n_classes / 10.0,
            imbalance,
            1.0 if ds.task == "regression" else 0.0,
            np.tanh(skew),
            np.tanh(spread * 2 - 1),
        ],
        dtype=np.float32,
    )


# ---- packed shards (fast path): group same-shape tasks for BATCHED inference ----
import glob as _glob
from collections import defaultdict as _dd


class PackedTask:
    __slots__ = ("hash", "Xtr", "ytr", "Xte", "yte", "task", "n_features", "n_classes")

    def __init__(self, h, Xtr, ytr, Xte, yte, task):
        self.hash, self.Xtr, self.ytr, self.Xte, self.yte, self.task = h, Xtr, ytr, Xte, yte, task
        self.n_features = Xtr.shape[1]
        self.n_classes = 0 if task == "regression" else int(len(np.unique(np.concatenate([ytr, yte]))))

    n_train = property(lambda s: s.Xtr.shape[0])


def iter_packed_shards(base, split="train", root=None):
    root = root or QUANT_ROOT
    yield from sorted(_glob.glob(os.path.join(root, base, split, "*.npz")))


def load_packed_shard(path):
    """Return list[PackedTask] from one shard (all same feature-count)."""
    z = np.load(path, allow_pickle=True)
    to, so = z["train_offsets"], z["test_offsets"]
    ids = z["task_ids"] if "task_ids" in z else [f"t{i}" for i in range(len(to) - 1)]
    reg = np.issubdtype(z["train_y"].dtype, np.floating)
    task = "regression" if reg else "binary"
    out = []
    for i in range(len(to) - 1):
        Xtr = z["train_x"][to[i] : to[i + 1]].astype("float32")
        Xte = z["test_x"][so[i] : so[i + 1]].astype("float32")
        ytr = z["train_y"][to[i] : to[i + 1]]
        yte = z["test_y"][so[i] : so[i + 1]]
        h = str(ids[i]) if i < len(ids) else f"{os.path.basename(path)}#{i}"
        out.append(PackedTask(h, Xtr, ytr, Xte, yte, task))
    return out


def group_by_shape(tasks, ctx_buckets=(512, 256, 128), n_test=128):
    """Standardize each task to a bucketed context size + fixed test count, then
    group by (ctx, n_features) so MANY same-shape tasks batch in one forward.
    Truncation keeps the biggest context bucket that fits (experts are robust to
    context size; consistency across the batch is what enables the speedup)."""
    g = _dd(list)
    for t in tasks:
        cb = next((c for c in sorted(ctx_buckets, reverse=True) if t.Xtr.shape[0] >= c), None)
        if cb is None or t.Xte.shape[0] < n_test:
            continue
        t.Xtr = t.Xtr[:cb]
        t.ytr = t.ytr[:cb]
        t.Xte = t.Xte[:n_test]
        t.yte = t.yte[:n_test]
        g[(cb, t.n_features)].append(t)
    return list(g.values())


# ---- column-as-target self-supervised augmentation (near-infinite real tasks) ----
def column_as_target_tasks(task, k=6, seed=0, min_minority=0.1):
    """From a real table, generate up to k binary tasks by picking feature columns
    as the target (binarize continuous via TRAIN median). Remaining columns are the
    features. Validated: the archive's ~74k binary tasks are built this way. Gives
    the fusion far more diverse per-dataset routing scenarios -> routing generalizes."""
    Xtr, Xte = task.Xtr, task.Xte
    F = Xtr.shape[1]
    if F < 3:
        return
    rng = np.random.default_rng([seed, hash(task.hash) % (2**31)])
    cols = rng.permutation(F)
    made = 0
    for c in cols:
        if made >= k:
            break
        thr = float(np.median(Xtr[:, c]))
        ytr = (Xtr[:, c] > thr).astype(np.int64)
        yte = (Xte[:, c] > thr).astype(np.int64)
        rate = ytr.mean()
        if rate < min_minority or rate > 1 - min_minority:  # degenerate split
            continue
        keep = [i for i in range(F) if i != c]
        yield PackedTask(f"{task.hash}#col{c}", Xtr[:, keep].copy(), ytr, Xte[:, keep].copy(), yte, "binary")
        made += 1
