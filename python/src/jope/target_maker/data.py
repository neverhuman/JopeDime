from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

TASK_BINARY = 0
TASK_REGRESSION = 1
TASK_NAMES = {TASK_BINARY: "binary", TASK_REGRESSION: "regression"}
TASK_IDS = {name: task for task, name in TASK_NAMES.items()}

PROFILE_QUANTILES = np.asarray([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0], dtype=np.float32)
PROFILE_HIST_BINS = 16
PROFILE_DIM = 8 + len(PROFILE_QUANTILES) + PROFILE_HIST_BINS


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    task: int
    source_dir: Path
    cache_path: Path
    length: int
    group_key: str
    positives: int | None = None

    @classmethod
    def from_json(cls, row: Mapping[str, object], manifest_dir: Path) -> "DatasetRecord":
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.is_absolute():
            cache_path = manifest_dir / cache_path
        return cls(
            dataset_id=str(row["id"]),
            task=TASK_IDS[str(row["task"])],
            source_dir=Path(str(row["source_dir"])),
            cache_path=cache_path,
            length=int(row["length"]),
            group_key=str(row["group_key"]),
            positives=None if row.get("positives") is None else int(row["positives"]),
        )

    def to_json(self, manifest_dir: Path) -> dict[str, object]:
        try:
            cache_path = self.cache_path.relative_to(manifest_dir)
        except ValueError:
            cache_path = self.cache_path
        return {
            "id": self.dataset_id,
            "task": TASK_NAMES[self.task],
            "source_dir": str(self.source_dir),
            "cache_path": str(cache_path),
            "length": self.length,
            "group_key": self.group_key,
            "positives": self.positives,
        }


def stable_hash(text: str | bytes, *, digest_size: int = 16) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def stable_seed(*parts: object) -> int:
    digest = stable_hash("|".join(map(str, parts)), digest_size=8)
    return int(digest, 16) & 0x7FFF_FFFF_FFFF_FFFF


def default_roots() -> dict[int, Path]:
    base = Path.home() / "quant"
    return {
        TASK_BINARY: base / "binary",
        TASK_REGRESSION: base / "regression",
    }


def iter_complete_dataset_dirs(root: Path) -> Iterator[Path]:
    """Yield complete dataset directories, including hidden .incoming-* directories."""
    if not root.exists():
        return
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / "train.csv").is_file() and (child / "test.csv").is_file():
            yield child


def scan_datasets(roots: Mapping[int, Path] | None = None, limit_per_task: int | None = None) -> list[tuple[int, Path]]:
    roots = roots or default_roots()
    found: list[tuple[int, Path]] = []
    for task in (TASK_BINARY, TASK_REGRESSION):
        count = 0
        for dataset_dir in iter_complete_dataset_dirs(Path(roots[task]).expanduser()):
            found.append((task, dataset_dir))
            count += 1
            if limit_per_task is not None and count >= limit_per_task:
                break
    return found


def read_final_column_csv(path: Path) -> np.ndarray:
    """Read only the last CSV column as float32, skipping feature columns."""
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            last = stripped.rsplit(",", 1)[-1].strip()
            try:
                value = float(last)
            except ValueError as exc:
                if line_number == 1:
                    continue
                raise ValueError(f"Could not parse final column in {path}:{line_number}: {last!r}") from exc
            if math.isfinite(value):
                values.append(value)
    return np.asarray(values, dtype=np.float32)


def read_dataset_target(dataset_dir: Path) -> np.ndarray:
    train = read_final_column_csv(Path(dataset_dir) / "train.csv")
    test = read_final_column_csv(Path(dataset_dir) / "test.csv")
    if train.size + test.size == 0:
        raise ValueError(f"No target values found in {dataset_dir}")
    return np.concatenate([train, test]).astype(np.float32, copy=False)


def deterministic_shuffle(values: np.ndarray, dataset_key: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("shuffle", seed, dataset_key))
    order = rng.permutation(len(values))
    return values[order]


def pack_bits_uint32(bits: np.ndarray) -> np.ndarray:
    bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bit_array.size == 0:
        return np.zeros(0, dtype=np.uint32)
    words = np.zeros((bit_array.size + 31) // 32, dtype=np.uint32)
    indexes = np.nonzero(bit_array)[0]
    if indexes.size:
        np.bitwise_or.at(words, indexes // 32, (np.uint32(1) << (indexes % 32).astype(np.uint32)))
    return words


def unpack_bits_uint32(words: np.ndarray, length: int) -> np.ndarray:
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    if length == 0:
        return np.zeros(0, dtype=np.uint8)
    indexes = np.arange(length, dtype=np.uint32)
    return ((packed[indexes // 32] >> (indexes % 32)) & np.uint32(1)).astype(np.uint8)


def bits_at_indices(words: np.ndarray, indices: np.ndarray) -> np.ndarray:
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    idx = np.asarray(indices, dtype=np.uint32)
    return ((packed[idx // 32] >> (idx % 32)) & np.uint32(1)).astype(np.float32)


def popcount_uint32(words: np.ndarray, length: int) -> int:
    if length == 0:
        return 0
    bits = unpack_bits_uint32(words, length)
    return int(bits.sum())


def compute_profile(values: np.ndarray, task: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float32)
    clipped = np.clip(arr, 0.0, 1.0)
    mean = float(np.mean(clipped))
    std = float(np.std(clipped))
    centered = clipped - mean
    skew = float(np.mean(centered**3) / ((std + 1e-6) ** 3))
    kurt = float(np.mean(centered**4) / ((std + 1e-6) ** 4))
    q = np.quantile(clipped, PROFILE_QUANTILES).astype(np.float32)
    hist = np.histogram(clipped, bins=PROFILE_HIST_BINS, range=(0.0, 1.0))[0].astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    entropy = float(-(hist * np.log(hist + 1e-8)).sum() / math.log(PROFILE_HIST_BINS))
    prefix = np.asarray(
        [
            float(task),
            math.log1p(float(arr.size)) / 16.0,
            mean,
            std,
            float(np.min(clipped)),
            float(np.max(clipped)),
            skew,
            kurt,
        ],
        dtype=np.float32,
    )
    tail = np.asarray([entropy], dtype=np.float32)
    return np.concatenate([prefix, q, hist[:-1], tail]).astype(np.float32, copy=False)


def profile_group_key(values: np.ndarray, task: int) -> str:
    profile = compute_profile(values, task)
    rounded = np.round(profile, 3 if task == TASK_REGRESSION else 4)
    return stable_hash(rounded.tobytes(), digest_size=12)


def _atomic_save_npz(path: Path, **arrays: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    np.savez_compressed(tmp, **arrays)
    generated = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    generated.replace(path)


def default_cache_workers() -> int:
    return max(1, min(96, (os.cpu_count() or 1) - 8))


def _prepare_cache_record(
    item: tuple[int, Path, Path, int, bool],
) -> DatasetRecord:
    task, dataset_dir, target_dir, seed, overwrite = item
    dataset_dir = Path(dataset_dir)
    target_dir = Path(target_dir)
    dataset_key = f"{TASK_NAMES[task]}:{dataset_dir.resolve()}"
    dataset_id = stable_hash(dataset_key, digest_size=16)
    cache_path = target_dir / f"{dataset_id}.npz"
    if cache_path.exists() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as data:
            length = int(data["length"])
            group_key = str(data["group_key"].item())
            positives = int(data["positives"]) if "positives" in data.files else None
    else:
        values = deterministic_shuffle(read_dataset_target(dataset_dir), dataset_key, seed)
        if task == TASK_BINARY:
            bits = (values >= 0.5).astype(np.uint8, copy=False)
            positives = int(bits.sum())
            profile_values = bits.astype(np.float32, copy=False)
            group_key = profile_group_key(profile_values, task)
            _atomic_save_npz(
                cache_path,
                task=np.asarray(task, dtype=np.int64),
                length=np.asarray(bits.size, dtype=np.int64),
                bits=pack_bits_uint32(bits),
                positives=np.asarray(positives, dtype=np.int64),
                profile=compute_profile(profile_values, task),
                group_key=np.asarray(group_key),
            )
            length = int(bits.size)
        else:
            values = values.astype(np.float32, copy=False)
            group_key = profile_group_key(values, task)
            _atomic_save_npz(
                cache_path,
                task=np.asarray(task, dtype=np.int64),
                length=np.asarray(values.size, dtype=np.int64),
                values=values,
                profile=compute_profile(values, task),
                group_key=np.asarray(group_key),
            )
            positives = None
            length = int(values.size)
    return DatasetRecord(
        dataset_id=dataset_id,
        task=task,
        source_dir=dataset_dir,
        cache_path=cache_path,
        length=length,
        group_key=group_key,
        positives=positives,
    )


def prepare_cache(
    cache_dir: Path,
    roots: Mapping[int, Path] | None = None,
    *,
    seed: int = 17,
    limit_per_task: int | None = None,
    overwrite: bool = False,
    workers: int | None = None,
) -> list[DatasetRecord]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    target_dir = cache_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.jsonl"
    workers = default_cache_workers() if workers is None else max(1, int(workers))
    jobs = [(task, Path(dataset_dir), target_dir, seed, overwrite) for task, dataset_dir in scan_datasets(roots, limit_per_task=limit_per_task)]
    if workers == 1 or len(jobs) <= 1:
        records = [_prepare_cache_record(job) for job in jobs]
    else:
        chunksize = max(1, len(jobs) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_prepare_cache_record, jobs, chunksize=chunksize))
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(cache_dir), sort_keys=True) + "\n")
    return records


def load_manifest(cache_dir: Path, split: str | None = None, val_fraction: float = 0.05, seed: int = 17) -> list[DatasetRecord]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.jsonl"
    records: list[DatasetRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(DatasetRecord.from_json(json.loads(line), cache_dir))
    if split is None or split == "all":
        return records
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    train_records: list[DatasetRecord] = []
    val_records: list[DatasetRecord] = []
    for record in records:
        bucket = stable_seed("split", seed, record.dataset_id) / float(0x7FFF_FFFF_FFFF_FFFF)
        (val_records if bucket < val_fraction else train_records).append(record)
    return val_records if split == "val" else train_records


def sample_indices(length: int, window_size: int, rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
    if length <= 0:
        raise ValueError("Cannot sample from an empty target")
    if length >= window_size:
        indexes = rng.choice(length, size=window_size, replace=False)
        return indexes.astype(np.int64), window_size, 0.0
    indexes = rng.choice(length, size=window_size, replace=True)
    return indexes.astype(np.int64), length, 1.0 - (float(length) / float(window_size))


class TargetCacheDataset(Dataset):
    def __init__(
        self,
        records: Sequence[DatasetRecord],
        *,
        window_size: int = 96,
        seed: int = 17,
        two_views: bool = True,
        length_multiplier: int = 1,
        augment_lengths: Sequence[int] = (32, 64, 96),
        randomize: bool = False,
    ) -> None:
        self.records = list(records)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.two_views = bool(two_views)
        self.length_multiplier = max(1, int(length_multiplier))
        self.augment_lengths = tuple(int(x) for x in augment_lengths if int(x) > 0)
        self.randomize = bool(randomize)
        self._random_rng_key: tuple[int, int, int] | None = None
        self._random_rng: np.random.Generator | None = None
        if not self.records:
            raise ValueError("TargetCacheDataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records) * self.length_multiplier

    def _sample_one(self, record: DatasetRecord, npz: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, object]:
        requested = int(rng.choice(self.augment_lengths)) if self.augment_lengths else self.window_size
        requested = min(max(1, requested), self.window_size)
        idx, effective_count, replacement_fraction = sample_indices(record.length, requested, rng)
        values = np.zeros(self.window_size, dtype=np.float32)
        mask = np.zeros(self.window_size, dtype=np.float32)
        if record.task == TASK_BINARY:
            sampled = bits_at_indices(npz["bits"], idx)
        else:
            sampled = np.asarray(npz["values"], dtype=np.float32)[idx]
        values[:requested] = sampled.astype(np.float32, copy=False)
        mask[:requested] = 1.0
        return {
            "values": values,
            "mask": mask,
            "effective_count": effective_count,
            "replacement_fraction": replacement_fraction,
            "requested_length": requested,
        }

    def _worker_rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        worker_seed = self.seed if worker_info is None else int(worker_info.seed)
        rank = int(os.environ.get("RANK", "0"))
        key = (rank, worker_id, worker_seed)
        if self._random_rng is None or self._random_rng_key != key:
            self._random_rng_key = key
            self._random_rng = np.random.default_rng(stable_seed("sample-random", self.seed, rank, worker_id, worker_seed))
        return self._random_rng

    def _sample_rng(self, record: DatasetRecord, index: int) -> np.random.Generator:
        if self.randomize:
            return self._worker_rng()
        return np.random.default_rng(stable_seed("sample", self.seed, record.dataset_id, index))

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index % len(self.records)]
        rng = self._sample_rng(record, index)
        with np.load(record.cache_path, allow_pickle=False) as npz:
            first = self._sample_one(record, npz, rng)
            second = self._sample_one(record, npz, rng) if self.two_views else first
            profile = np.asarray(npz["profile"], dtype=np.float32)
        return {
            "dataset_id": record.dataset_id,
            "group_key": record.group_key,
            "task": record.task,
            "length": record.length,
            "profile": profile,
            "view1": first,
            "view2": second,
        }


def collate_target_batch(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    group_keys = [str(item["group_key"]) for item in items]
    dataset_ids = [str(item["dataset_id"]) for item in items]
    unique_groups = {key: i for i, key in enumerate(sorted(set(group_keys)))}

    def stack_view(name: str) -> dict[str, torch.Tensor]:
        view = [item[name] for item in items]
        return {
            "values": torch.from_numpy(np.stack([v["values"] for v in view])).float(),
            "mask": torch.from_numpy(np.stack([v["mask"] for v in view])).float(),
            "effective_count": torch.tensor([v["effective_count"] for v in view], dtype=torch.float32),
            "replacement_fraction": torch.tensor([v["replacement_fraction"] for v in view], dtype=torch.float32),
            "requested_length": torch.tensor([v["requested_length"] for v in view], dtype=torch.float32),
        }

    return {
        "dataset_id": dataset_ids,
        "group_key": group_keys,
        "group_label": torch.tensor([unique_groups[key] for key in group_keys], dtype=torch.long),
        "task": torch.tensor([item["task"] for item in items], dtype=torch.long),
        "length": torch.tensor([item["length"] for item in items], dtype=torch.float32),
        "profile": torch.from_numpy(np.stack([item["profile"] for item in items])).float(),
        "view1": stack_view("view1"),
        "view2": stack_view("view2"),
    }


def read_target_input(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.is_dir():
        return read_dataset_target(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32, copy=False).reshape(-1)
    return read_final_column_csv(path)
