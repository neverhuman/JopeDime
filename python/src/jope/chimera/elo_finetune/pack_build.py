"""Build fable_real_v1.pack from the raw regression archive.

Inputs: archive dataset dirs (train.csv/test.csv/meta.json), the fable-elo
utilities cache (single shared report parser), and the canonical v2 split
assignment. The header carries no timestamp so two builds of the same inputs
are byte-identical (verified by verify_pack.py --rebuild-check).

Usage: python3 -m python.elo_finetune.pack_build [--out DIR] [--jobs N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .pack import SPLIT_NAMES, write_pack

ARCHIVE_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
DEFAULT_OUT = Path("/home/ubuntu/fable_god/cache/pack")
REAL_DIR_RE = re.compile(r"^[0-9a-f]{16}$")
PACKER_VERSION = "1.0.0"


def read_csv_f32(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, header=None, dtype=np.float32)
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float32))


def load_dataset(args: tuple[int, str]) -> dict:
    idx, dataset_id = args
    root = ARCHIVE_ROOT / dataset_id
    meta = json.loads((root / "meta.json").read_text())
    counts = meta["counts"]
    train = read_csv_f32(root / "train.csv")
    test = read_csv_f32(root / "test.csv")

    n_features = int(counts["n_features"])
    for name, arr, rows_key in (("train", train, "train_rows"), ("test", test, "test_rows")):
        expected = (int(counts[rows_key]), n_features + 1)
        if arr.shape != expected:
            raise ValueError(f"{dataset_id}/{name}.csv shape {arr.shape} != meta {expected}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{dataset_id}/{name}.csv contains non-finite values")

    return {
        "idx": idx,
        "dataset_id": dataset_id,
        "n_features": n_features,
        "seed": int(meta.get("sampling", {}).get("seed", -1)),
        "Xtr": train[:, :-1],
        "ytr": train[:, -1],
        "Xte": test[:, :-1],
        "yte": test[:, -1],
    }


def load_utilities(cache_path: Path, real_ids: set[str]) -> dict[str, dict[str, float]]:
    per_dataset: dict[str, dict[str, float]] = {h: {} for h in real_ids}
    with open(cache_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            dataset = rec["dataset_hash"]
            if dataset not in per_dataset:
                continue
            utility = rec.get("utility")
            if rec.get("status") != "completed" or utility is None:
                continue
            if not np.isfinite(utility):
                continue
            algo = rec["algorithm"]
            prev = per_dataset[dataset].get(algo)
            if prev is None or utility > prev:
                per_dataset[dataset][algo] = float(utility)
    return per_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(out_dir: Path, jobs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_map = json.loads((out_dir / "real_split_v2.json").read_text())
    utilities_path = out_dir / "utilities.jsonl"

    on_disk = sorted(
        d.name for d in ARCHIVE_ROOT.iterdir() if d.is_dir() and REAL_DIR_RE.match(d.name)
    )
    dataset_ids = [h for h in on_disk if h in split_map]
    dropped = [h for h in on_disk if h not in split_map]

    per_dataset_utils = load_utilities(utilities_path, set(dataset_ids))
    algos = sorted({a for utils in per_dataset_utils.values() for a in utils})
    algo_id = {name: i for i, name in enumerate(algos)}

    with Pool(jobs) as pool:
        loaded = pool.map(load_dataset, list(enumerate(dataset_ids)), chunksize=8)
    loaded.sort(key=lambda r: r["idx"])

    D = len(loaded)
    arrays: dict[str, np.ndarray] = {
        "dataset_ids": np.frombuffer(
            "".join(r["dataset_id"] for r in loaded).encode(), np.uint8
        ).reshape(D, 16),
        "n_features": np.array([r["n_features"] for r in loaded], np.int32),
        "split": np.array(
            [SPLIT_NAMES.index(split_map[r["dataset_id"]]) for r in loaded], np.uint8
        ),
        "seed": np.array([r["seed"] for r in loaded], np.int64),
    }

    def csr(counts: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])

    arrays["train_row_offsets"] = csr([r["ytr"].shape[0] for r in loaded])
    arrays["test_row_offsets"] = csr([r["yte"].shape[0] for r in loaded])
    arrays["x_train_offsets"] = csr([r["Xtr"].size for r in loaded])
    arrays["x_test_offsets"] = csr([r["Xte"].size for r in loaded])
    arrays["X_train"] = np.concatenate([r["Xtr"].ravel() for r in loaded])
    arrays["y_train"] = np.concatenate([r["ytr"] for r in loaded])
    arrays["X_test"] = np.concatenate([r["Xte"].ravel() for r in loaded])
    arrays["y_test"] = np.concatenate([r["yte"] for r in loaded])

    util_ids, util_vals, util_counts = [], [], []
    for r in loaded:
        utils = per_dataset_utils[r["dataset_id"]]
        pairs = sorted((algo_id[a], v) for a, v in utils.items())
        util_counts.append(len(pairs))
        util_ids.extend(p[0] for p in pairs)
        util_vals.extend(p[1] for p in pairs)
    arrays["util_offsets"] = csr(util_counts)
    arrays["util_algo_ids"] = np.array(util_ids, np.int32)
    arrays["util_values"] = np.array(util_vals, np.float64)

    header = {
        "source_root": str(ARCHIVE_ROOT),
        "packer_version": PACKER_VERSION,
        "n_datasets": D,
        "n_algos": len(algos),
        "algos": algos,
    }
    pack_path = out_dir / "fable_real_v1.pack"
    write_pack(pack_path, header, arrays)

    pack_sha = sha256_file(pack_path)
    (out_dir / "fable_real_v1.pack.sha256").write_text(
        f"{pack_sha}  fable_real_v1.pack\n"
    )
    split_counts = {
        name: int((arrays["split"] == code).sum()) for code, name in enumerate(SPLIT_NAMES)
    }
    manifest = {
        "schema": "fable_pack_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pack_sha256": pack_sha,
        "utilities_jsonl_sha256": sha256_file(utilities_path),
        "n_datasets": D,
        "split_counts": split_counts,
        "n_algos": len(algos),
        "utility_nnz": int(arrays["util_values"].size),
        "total_train_rows": int(arrays["train_row_offsets"][-1]),
        "total_test_rows": int(arrays["test_row_offsets"][-1]),
        "pack_bytes": pack_path.stat().st_size,
        "dropped_datasets": dropped,
    }
    (out_dir / "fable_real_v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    manifest = build(args.out, args.jobs)
    print(json.dumps({k: v for k, v in manifest.items() if k != "dropped_datasets"}, indent=2))
    if manifest["dropped_datasets"]:
        print(f"dropped {len(manifest['dropped_datasets'])} datasets not in v2 split map")


if __name__ == "__main__":
    main()
