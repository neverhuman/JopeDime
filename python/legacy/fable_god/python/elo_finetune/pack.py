"""fable_pack_v1: single memory-mapped binary holding every real dataset's
X/y plus the frozen competitor utility CSR.

Layout: magic "FABLPACK" | u32 version | u32 header_len | header JSON |
64-byte-aligned sections. All integers little-endian. X/y are f32 (bit-exact
vs float32 CSV parse); utilities are f64 (the exact replay ties at 1e-12).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _default_pack_dir() -> Path:
    override = os.environ.get("FABLE_PACK_DIR")
    if override:
        return Path(override)
    for candidate in (
        Path("/home/ubuntu/fable_god/cache/pack"),
        Path.home() / "fable_god_run" / "pack",
    ):
        if candidate.is_dir():
            return candidate
    return Path("/home/ubuntu/fable_god/cache/pack")


DEFAULT_PACK_DIR = _default_pack_dir()

MAGIC = b"FABLPACK"
VERSION = 1
ALIGN = 64
SPLIT_NAMES = ("train", "val", "test")

OUR_ALGORITHM = "tabpfn_v3_regressor"
FEATURE_CAP = 256
CTX_BUCKETS = (64, 128, 256, 512, 700)
QRY_BUCKETS = (32, 64, 128, 256)

SECTION_ORDER = [
    ("dataset_ids", np.uint8),
    ("n_features", np.int32),
    ("split", np.uint8),
    ("seed", np.int64),
    ("train_row_offsets", np.int64),
    ("test_row_offsets", np.int64),
    ("x_train_offsets", np.int64),
    ("x_test_offsets", np.int64),
    ("X_train", np.float32),
    ("y_train", np.float32),
    ("X_test", np.float32),
    ("y_test", np.float32),
    ("util_offsets", np.int64),
    ("util_algo_ids", np.int32),
    ("util_values", np.float64),
]


def _pad(offset: int) -> int:
    return (offset + ALIGN - 1) // ALIGN * ALIGN


def write_pack(path: Path, header_meta: dict, arrays: dict[str, np.ndarray]) -> None:
    sections = {}
    offset = 0
    for name, dtype in SECTION_ORDER:
        arr = np.ascontiguousarray(arrays[name], dtype=dtype)
        arrays[name] = arr
        offset = _pad(offset)
        sections[name] = {
            "offset": offset,
            "nbytes": arr.nbytes,
            "dtype": np.dtype(dtype).name,
            "shape": list(arr.shape),
        }
        offset += arr.nbytes

    header = dict(header_meta)
    header["schema"] = "fable_pack_v1"
    header["sections"] = sections
    header_bytes = json.dumps(header, sort_keys=True).encode()
    preamble = MAGIC + np.uint32(VERSION).tobytes() + np.uint32(len(header_bytes)).tobytes()
    body_start = _pad(len(preamble) + len(header_bytes))

    with open(path, "wb") as f:
        f.write(preamble)
        f.write(header_bytes)
        f.write(b"\0" * (body_start - len(preamble) - len(header_bytes)))
        pos = 0
        for name, _ in SECTION_ORDER:
            meta = sections[name]
            f.write(b"\0" * (meta["offset"] - pos))
            f.write(arrays[name].tobytes())
            pos = meta["offset"] + meta["nbytes"]


@dataclass
class DatasetView:
    dataset_id: str
    index: int
    split: int
    seed: int
    n_features: int
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    util_algo_ids: np.ndarray
    util_values: np.ndarray

    @property
    def train_rows(self) -> int:
        return self.ytr.shape[0]

    @property
    def test_rows(self) -> int:
        return self.yte.shape[0]


class FablePack:
    def __init__(self, path: str | Path, preload: bool = False):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            if f.read(8) != MAGIC:
                raise ValueError(f"{self.path}: bad magic")
            version = int(np.frombuffer(f.read(4), np.uint32)[0])
            if version != VERSION:
                raise ValueError(f"{self.path}: version {version} != {VERSION}")
            header_len = int(np.frombuffer(f.read(4), np.uint32)[0])
            self.header = json.loads(f.read(header_len))

        body_start = _pad(16 + header_len)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=body_start)
        if preload:
            int(np.add.reduce(self._mm[:: 1 << 20], dtype=np.int64))

        self._sec = {}
        for name, meta in self.header["sections"].items():
            raw = self._mm[meta["offset"] : meta["offset"] + meta["nbytes"]]
            self._sec[name] = raw.view(np.dtype(meta["dtype"])).reshape(meta["shape"])

        self.n_datasets = int(self.header["n_datasets"])
        self.algos: list[str] = self.header["algos"]
        self.algo_id = {name: i for i, name in enumerate(self.algos)}
        ids = self._sec["dataset_ids"]
        self.dataset_ids = [bytes(row).decode() for row in ids]
        self.dataset_index = {h: i for i, h in enumerate(self.dataset_ids)}

    def __len__(self) -> int:
        return self.n_datasets

    def section(self, name: str) -> np.ndarray:
        return self._sec[name]

    def dataset(self, i: int) -> DatasetView:
        s = self._sec
        tr0, tr1 = int(s["train_row_offsets"][i]), int(s["train_row_offsets"][i + 1])
        te0, te1 = int(s["test_row_offsets"][i]), int(s["test_row_offsets"][i + 1])
        xtr0, xtr1 = int(s["x_train_offsets"][i]), int(s["x_train_offsets"][i + 1])
        xte0, xte1 = int(s["x_test_offsets"][i]), int(s["x_test_offsets"][i + 1])
        u0, u1 = int(s["util_offsets"][i]), int(s["util_offsets"][i + 1])
        n_feat = int(s["n_features"][i])
        return DatasetView(
            dataset_id=self.dataset_ids[i],
            index=i,
            split=int(s["split"][i]),
            seed=int(s["seed"][i]),
            n_features=n_feat,
            Xtr=s["X_train"][xtr0:xtr1].reshape(tr1 - tr0, n_feat),
            ytr=s["y_train"][tr0:tr1],
            Xte=s["X_test"][xte0:xte1].reshape(te1 - te0, n_feat),
            yte=s["y_test"][te0:te1],
            util_algo_ids=s["util_algo_ids"][u0:u1],
            util_values=s["util_values"][u0:u1],
        )

    def split_indices(self, split: str) -> np.ndarray:
        code = SPLIT_NAMES.index(split)
        return np.nonzero(self._sec["split"] == code)[0]

    def competitor_utilities(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Frozen utilities on dataset i with our own record removed."""
        view = self.dataset(i)
        ours = self.algo_id.get(OUR_ALGORITHM, -1)
        keep = view.util_algo_ids != ours
        return view.util_algo_ids[keep], view.util_values[keep]


def _hash_int(dataset_id: str) -> int:
    return int(dataset_id, 16)


def quantize(value: int, buckets: tuple[int, ...]) -> int:
    """Largest bucket <= value, or value itself when below the smallest."""
    chosen = None
    for b in buckets:
        if b <= value:
            chosen = b
    return chosen if chosen is not None else value


def protocol(train_rows: int, test_rows: int, n_features: int) -> tuple[int, int, int]:
    """Per-dataset deterministic (ctx, qry, F_eff) — a function of the dataset
    only, never of batch composition, so pairing with u_base stays exact."""
    return (
        quantize(train_rows, CTX_BUCKETS),
        quantize(test_rows, QRY_BUCKETS),
        min(n_features, FEATURE_CAP),
    )


def seeded_rows(dataset_id: str, seed_a: int, n_total: int, n_take: int, tag: int) -> np.ndarray:
    if n_take >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng([seed_a, _hash_int(dataset_id) & (2**63 - 1), tag])
    return np.sort(rng.choice(n_total, size=n_take, replace=False))


def seeded_cols(dataset_id: str, seed_a: int, n_features: int, cap: int) -> np.ndarray:
    if n_features <= cap:
        return np.arange(n_features)
    rng = np.random.default_rng([seed_a, _hash_int(dataset_id) & (2**63 - 1), 3])
    return np.sort(rng.choice(n_features, size=cap, replace=False))


@dataclass
class MicroPlan:
    dataset_indices: list[int]
    ctx: int
    qry: int
    n_features: int


def bucket_schedule(
    pack: FablePack,
    split: str,
    run_seed: int,
    epoch: int,
    micro_size: int,
    cell_budget: int = 1_600_000,
    row_budget: int = 1 << 30,
) -> list[MicroPlan]:
    """Deterministic pure function of its arguments: identical on every rank.

    cell_budget caps rows*B*F per micro-batch (the per-column ISAB stage);
    row_budget caps rows*B (the 24-layer ICL stack). With autograd both
    stages hold activations, so training needs far tighter caps than
    no-grad passes (OOM'd at 1.6M cells on 24GB with grad).
    """
    n_feat = pack.section("n_features")
    tr_off = pack.section("train_row_offsets")
    te_off = pack.section("test_row_offsets")

    groups: dict[tuple[int, int, int], list[int]] = {}
    for i in pack.split_indices(split).tolist():
        train_rows = int(tr_off[i + 1] - tr_off[i])
        test_rows = int(te_off[i + 1] - te_off[i])
        ctx, qry, f_eff = protocol(train_rows, test_rows, int(n_feat[i]))
        groups.setdefault((f_eff, ctx, qry), []).append(i)

    rng = np.random.default_rng([run_seed, epoch])
    plans: list[MicroPlan] = []
    for key in sorted(groups):
        f_eff, ctx, qry = key
        rows = ctx + qry
        size = max(
            1,
            min(
                micro_size,
                cell_budget // max(1, rows * f_eff),
                row_budget // max(1, rows),
            ),
        )
        members = groups[key]
        order = rng.permutation(len(members))
        for start in range(0, len(members), size):
            chunk = [members[j] for j in order[start : start + size]]
            plans.append(MicroPlan(chunk, ctx, qry, f_eff))
    plan_order = rng.permutation(len(plans))
    return [plans[j] for j in plan_order]


@dataclass
class Batch:
    x: np.ndarray  # (ctx+qry, B, F) f32
    y_ctx: np.ndarray  # (ctx, B) f32
    y_qry: np.ndarray  # (qry, B) f32
    dataset_indices: list[int]
    ctx: int
    qry: int


def collate(pack: FablePack, plan: MicroPlan, seed_a: int) -> Batch:
    B = len(plan.dataset_indices)
    x = np.empty((plan.ctx + plan.qry, B, plan.n_features), dtype=np.float32)
    y_ctx = np.empty((plan.ctx, B), dtype=np.float32)
    y_qry = np.empty((plan.qry, B), dtype=np.float32)
    for b, i in enumerate(plan.dataset_indices):
        view = pack.dataset(i)
        rows_c = seeded_rows(view.dataset_id, seed_a, view.train_rows, plan.ctx, tag=1)
        rows_q = seeded_rows(view.dataset_id, seed_a, view.test_rows, plan.qry, tag=2)
        cols = seeded_cols(view.dataset_id, seed_a, view.n_features, plan.n_features)
        x[: plan.ctx, b, :] = view.Xtr[np.ix_(rows_c, cols)]
        x[plan.ctx :, b, :] = view.Xte[np.ix_(rows_q, cols)]
        y_ctx[:, b] = view.ytr[rows_c]
        y_qry[:, b] = view.yte[rows_q]
    return Batch(x, y_ctx, y_qry, list(plan.dataset_indices), plan.ctx, plan.qry)
