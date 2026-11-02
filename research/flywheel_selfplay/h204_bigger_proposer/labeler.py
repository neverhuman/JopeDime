"""H204 Bigger Proposer: Bounded-Cache S3 Streamer & Exact Rust CV Cost Labeler.

Labels nonsealed D4 `TRAIN` regression contexts with exact Rust two-fold CV costs
(`8-16` candidates per context, `run_outer_test=False`), respecting:
  - `nice 19`
  - `<= 2` labeler workers (`xbabe2` load guard)
  - `<= 4` numeric threads (`OMP`/`MKL`/`OpenBLAS`/`NUMEXPR`/`RAYON`/PyTorch intra-op)
  - `<= 20 GB` local LRU cache on `xbabe2`
  - Disk guard: `used_ratio <= 0.938` and `free_bytes >= 40 GiB`
  - Sealed confirmation families (`DL-Val` 39 families / 91 contexts + meta-test) strictly excluded
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.flywheel_selfplay.h204_bigger_proposer.analysis import (
    DL_TRAIN_EXPECTED_CONTEXTS,
    DL_TRAIN_EXPECTED_FAMILIES,
    DL_VAL_EXPECTED_CONTEXTS,
    MAX_DISK_USED_RATIO,
    MAX_LABELER_WORKERS,
    MAX_LOCAL_CACHE_BYTES,
    MIN_DISK_FREE_BYTES,
    S3_DATASETS_BUCKET,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_H50_INPUT_ROOT = Path("/home/ubuntu/jope-data/h50/input")
DEFAULT_H58_CACHE_ROOT = Path("/home/ubuntu/jope-data/h58/cache")
DEFAULT_R882_CKPT = Path("/home/ubuntu/jope-data/h51/ckpt/round-882.pt")
DEFAULT_JRB01_MANIFEST = Path("/home/ubuntu/jope-data/jrb01/admitted_train_datasets.json")

EXPECTED_DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"

PINNED_INPUT_SHA256: dict[Path, str] = {
    DEFAULT_H50_INPUT_ROOT
    / "source_manifest.json": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
    DEFAULT_H50_INPUT_ROOT
    / "selected.json": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    DEFAULT_H50_INPUT_ROOT
    / "cpu_exact_fused.py": "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab",
    DEFAULT_H58_CACHE_ROOT
    / "cache_manifest.json": "66b1df1374242580d482b7d6f9b72b6fe8f65830f90f658ca662a698aa229897",
    DEFAULT_H58_CACHE_ROOT
    / "context_features.npz": "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf",
    DEFAULT_H58_CACHE_ROOT
    / "pool_priors.npz": "6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe",
    DEFAULT_R882_CKPT: "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
}

PINNED_REPO_SHA256: dict[str, str] = {
    "research/flywheel_selfplay/rust_evaluator.py": (
        "7b0cb0ecb98c7836b16a781f632a9fc74683c2ab9facee820790ceebc759cb45"
    ),
    "python/src/jope/north_star/dope_freeze_guard.py": (
        "908547a91fcb6841868aed68e1c0811befbdc2cf182053125fdf4004549fd2b1"
    ),
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_input_closure_digests(
    *,
    repo_root: Path = REPO_ROOT,
    verify_external_inputs: bool = True,
) -> dict[str, str]:
    """Verify SHA-256 digests of all pinned repository helpers and external inputs."""
    verified: dict[str, str] = {}
    for rel_path, expected_sha in PINNED_REPO_SHA256.items():
        full_p = repo_root / rel_path
        actual_sha = sha256_file(full_p)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Pinned repo file SHA-256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )
        verified[rel_path] = actual_sha

    if verify_external_inputs:
        for ext_path, expected_sha in PINNED_INPUT_SHA256.items():
            actual_sha = sha256_file(ext_path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Pinned external input SHA-256 mismatch for {ext_path}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            verified[str(ext_path)] = actual_sha
    return verified


def check_disk_budget(
    path: Path,
    max_used_ratio: float = MAX_DISK_USED_RATIO,
    min_free_bytes: int = MIN_DISK_FREE_BYTES,
) -> dict[str, Any]:
    """Verify disk usage on `xbabe2` stays below 93.8% and >= 40 GiB free."""
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    denom = usage.total
    used_ratio = float(usage.used / denom) if denom > 0 else 1.0
    free_bytes = int(usage.free)
    ok = bool(used_ratio <= max_used_ratio and free_bytes >= min_free_bytes)
    return {
        "ok": ok,
        "path": str(target),
        "used_bytes": int(usage.used),
        "free_bytes": free_bytes,
        "total_bytes": int(usage.total),
        "used_ratio": used_ratio,
        "max_used_ratio": max_used_ratio,
        "min_free_bytes": min_free_bytes,
    }


class LocalLRUCache:
    """Bounded local cache enforcing `max_bytes <= 20 GiB` with LRU eviction."""

    def __init__(self, cache_dir: Path, max_bytes: int = MAX_LOCAL_CACHE_BYTES) -> None:
        if max_bytes <= 0 or max_bytes > MAX_LOCAL_CACHE_BYTES:
            raise ValueError(f"max_bytes must be in (0, {MAX_LOCAL_CACHE_BYTES}], got {max_bytes}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.evicted_files = 0
        self.evicted_bytes = 0

    def current_bytes(self) -> int:
        """Return total bytes currently stored under `cache_dir`."""
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total

    def evict_to_fit(self, incoming_bytes: int = 0) -> dict[str, int]:
        """Evict oldest-accessed files until `current_bytes() + incoming_bytes <= max_bytes`."""
        if incoming_bytes > self.max_bytes:
            raise ValueError(f"Single payload ({incoming_bytes} B) exceeds cache cap ({self.max_bytes} B)")
        files: list[tuple[float, int, Path]] = []
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((float(st.st_mtime), int(st.st_size), p))
                total += int(st.st_size)

        files.sort(key=lambda item: item[0])
        evicted_count = 0
        evicted_b = 0
        for _, size, fpath in files:
            if total + incoming_bytes <= self.max_bytes:
                break
            try:
                fpath.unlink()
                total -= size
                evicted_count += 1
                evicted_b += size
                self.evicted_files += 1
                self.evicted_bytes += size
            except OSError:
                continue

        return {
            "remaining_bytes": total,
            "evicted_files": evicted_count,
            "evicted_bytes": evicted_b,
            "total_evicted_files": self.evicted_files,
            "total_evicted_bytes": self.evicted_bytes,
        }

    def put_bytes(self, key: str, payload: bytes) -> Path:
        """Write `payload` into bounded cache after enforcing disk & LRU caps."""
        disk = check_disk_budget(self.cache_dir)
        if not disk["ok"]:
            raise RuntimeError(
                f"Disk budget exceeded before cache write: used_ratio={disk['used_ratio']:.5f}, "
                f"free_bytes={disk['free_bytes']}"
            )
        self.evict_to_fit(len(payload))
        safe_rel = Path(key.lstrip("/"))
        if ".." in safe_rel.parts:
            raise ValueError(f"Refusing traversal key: {key}")
        dest = self.cache_dir / safe_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        now = time.time()
        os.utime(dest, (now, now))
        return dest

    def get_bytes(self, key: str) -> bytes | None:
        """Read cached bytes and refresh LRU mtime if present."""
        safe_rel = Path(key.lstrip("/"))
        if ".." in safe_rel.parts:
            raise ValueError(f"Refusing traversal key: {key}")
        dest = self.cache_dir / safe_rel
        if not dest.is_file():
            return None
        now = time.time()
        try:
            os.utime(dest, (now, now))
        except OSError:
            pass
        return dest.read_bytes()


@dataclass(frozen=True)
class LabeledContextBatch:
    """Labeled training context batch with 8-16 exact Rust CV candidate labels per context."""

    context_ids: list[str]
    family_ids: list[str]
    dope629: np.ndarray  # (N, 629) float32
    tokens_10x32: np.ndarray  # (N, 10, 32) float32
    mask_10: np.ndarray  # (N, 10) float32
    lm48_z: np.ndarray  # (N, 48) float32
    cand_indices: np.ndarray  # (N, K) int64 in [0, 1152)
    cand_features: np.ndarray  # (N, K, 428) float32 (program 412 + proj16 16)
    cand_utilities: np.ndarray  # (N, K) float32 (np.where(gate, -fitness, -2.0))
    cand_gates: np.ndarray  # (N, K) float32
    full_cand_features: np.ndarray  # (N, 1152, 428) float32
    full_fitness: np.ndarray  # (N, 1152) float64
    full_utilities: np.ndarray  # (N, 1152) float64 (higher is better: -fitness when gate else -2.0)
    full_gates: np.ndarray  # (N, 1152) float32


@dataclass(frozen=True)
class EvalContextBatch:
    """Evaluation context batch (DL-Val 91 contexts) with strictly no labeler training access."""

    context_ids: list[str]
    family_ids: list[str]
    dope629: np.ndarray  # (N, 629) float32
    tokens_10x32: np.ndarray  # (N, 10, 32) float32
    mask_10: np.ndarray  # (N, 10) float32
    lm48_z: np.ndarray  # (N, 48) float32
    full_cand_features: np.ndarray  # (N, 1152, 428) float32
    full_fitness: np.ndarray  # (N, 1152) float64
    full_utilities: np.ndarray  # (N, 1152) float64
    full_gates: np.ndarray  # (N, 1152) float32


def select_candidate_subset_indices(
    n_total: int = 1152,
    k_candidates: int = 16,
    seed: int = 204101,
    context_idx: int = 0,
) -> np.ndarray:
    """Deterministically select `k_candidates` (8..16) indices per context.

    Always includes the 4 revealed anchors (indices 0, 1, 2, 3) plus `k_candidates - 4`
    stratified candidate indices across the non-anchor slots `4..n_total-1`.
    """
    if not (8 <= k_candidates <= 16):
        raise ValueError(f"k_candidates must be in [8, 16], got {k_candidates}")
    if n_total < k_candidates:
        raise ValueError(f"n_total ({n_total}) < k_candidates ({k_candidates})")

    anchors = np.array([0, 1, 2, 3], dtype=np.int64)
    n_extra = k_candidates - 4
    rng = np.random.default_rng(seed + 1009 * int(context_idx))
    non_anchors = np.arange(4, n_total, dtype=np.int64)
    bins = np.array_split(non_anchors, n_extra)
    picked = [int(rng.choice(b)) for b in bins]
    return np.concatenate([anchors, np.array(picked, dtype=np.int64)], axis=0)


def check_jrb01_admission_status(
    jrb01_manifest_path: Path = DEFAULT_JRB01_MANIFEST,
) -> dict[str, Any]:
    """Check whether `luna_v1_tel` has posted a verified JRB-01 D4 TRAIN admission manifest."""
    if not jrb01_manifest_path.is_file():
        return {
            "jrb01_ready": False,
            "mode": "admitted_205_family_train_pool",
            "manifest_path": str(jrb01_manifest_path),
            "admitted_dataset_count": DL_TRAIN_EXPECTED_FAMILIES,
            "s3_bucket": S3_DATASETS_BUCKET,
        }
    try:
        payload = json.loads(jrb01_manifest_path.read_text(encoding="utf-8"))
        admitted_ids = payload.get("admitted_train_dataset_ids", [])
        sealed_overlap = payload.get("sealed_family_overlap_count", 0)
        if isinstance(admitted_ids, list) and len(admitted_ids) > 0 and int(sealed_overlap) == 0:
            return {
                "jrb01_ready": True,
                "mode": "jrb01_d4_train_stream",
                "manifest_path": str(jrb01_manifest_path),
                "admitted_dataset_count": len(admitted_ids),
                "s3_bucket": S3_DATASETS_BUCKET,
            }
    except Exception as exc:
        return {
            "jrb01_ready": False,
            "mode": "admitted_205_family_train_pool",
            "manifest_path": str(jrb01_manifest_path),
            "error": str(exc),
            "admitted_dataset_count": DL_TRAIN_EXPECTED_FAMILIES,
            "s3_bucket": S3_DATASETS_BUCKET,
        }
    return {
        "jrb01_ready": False,
        "mode": "admitted_205_family_train_pool",
        "manifest_path": str(jrb01_manifest_path),
        "admitted_dataset_count": DL_TRAIN_EXPECTED_FAMILIES,
        "s3_bucket": S3_DATASETS_BUCKET,
    }


def evaluate_context_candidates_rust_cv(
    raw_jtt1_bytes: bytes,
    programs: list[str] | list[dict[str, Any]],
    cache: LocalLRUCache,
    cache_key: str | None = None,
    max_workers: int = 1,
    timeout_s: float = 25.0,
) -> list[dict[str, Any]]:
    """Evaluate `8-16` GP candidate programs via exact two-fold Rust CV (`run_outer_test=False`).

    Uses the exact `research.flywheel_selfplay.rust_evaluator.evaluate_gp_batch_rust`
    signature: `(teacher_jtt1_gz_path, candidates, run_outer_test, max_workers, timeout_s)`.
    Content-binds the cache key to `sha256(jtt1_gz_bytes) + sha256(candidates_json)`.
    """
    if not (1 <= max_workers <= MAX_LABELER_WORKERS):
        raise ValueError(f"max_workers must be in [1, {MAX_LABELER_WORKERS}], got {max_workers}")
    if not (8 <= len(programs) <= 16):
        raise ValueError(f"Expected 8..16 candidate programs per context, got {len(programs)}")

    candidates: list[dict[str, Any]] = []
    for item in programs:
        if isinstance(item, str):
            candidates.append({"dsl": item, "operator": "edit_zero", "parents": []})
        elif isinstance(item, dict) and "dsl" in item:
            candidates.append(
                {
                    "dsl": str(item["dsl"]),
                    "operator": str(item.get("operator", "edit_zero")),
                    "parents": list(item.get("parents", [])),
                }
            )
        else:
            raise ValueError(f"Invalid candidate specification: {item!r}")

    jtt1_gz = raw_jtt1_bytes if raw_jtt1_bytes[:2] == b"\x1f\x8b" else gzip.compress(raw_jtt1_bytes, mtime=0)
    content_hash = sha256_bytes(jtt1_gz + json.dumps(candidates, sort_keys=True).encode("utf-8"))
    bound_key = f"{cache_key}_{content_hash[:16]}.json" if cache_key else f"cv_{content_hash}.json"

    cached = cache.get_bytes(bound_key)
    if cached is not None:
        return list(json.loads(cached.decode("utf-8")))

    from research.flywheel_selfplay.rust_evaluator import evaluate_gp_batch_rust

    jtt1_rel_key = f"jtt1_{sha256_bytes(jtt1_gz)[:16]}.jtt1.gz"
    jtt1_gz_path = cache.put_bytes(jtt1_rel_key, jtt1_gz)

    res = evaluate_gp_batch_rust(
        teacher_jtt1_gz_path=str(jtt1_gz_path),
        candidates=candidates,
        run_outer_test=False,
        max_workers=max_workers,
        timeout_s=timeout_s,
    )
    serialized = json.dumps(res, sort_keys=True).encode("utf-8")
    cache.put_bytes(bound_key, serialized)
    return list(res)


def load_h204_stage1_datasets(
    *,
    cache: LocalLRUCache,
    h50_input_root: Path = DEFAULT_H50_INPUT_ROOT,
    h58_cache_root: Path = DEFAULT_H58_CACHE_ROOT,
    k_candidates: int = 16,
    seed: int = 204101,
    verify_digests: bool = True,
) -> tuple[LabeledContextBatch, EvalContextBatch, dict[str, Any]]:
    """Load and verify the admitted 205-family DL-Train pool (`8-16` CV candidates/ctx) and 91 DL-Val pool.

    Verifies pinned SHA-256 digests of the input closure (`source_manifest.json`, `selected.json`,
    `cpu_exact_fused.py`, `context_features.npz`, `pool_priors.npz`, `round-882.pt`, and every
    train-only `.jtt1` table and `.npz` pool), enforces zero outer-test rows (`n_te == 0`), and
    enforces strict train/val family disjointness.
    """
    if not (8 <= k_candidates <= 16):
        raise ValueError(f"k_candidates must be in [8, 16], got {k_candidates}")

    verified_digests = verify_input_closure_digests(verify_external_inputs=verify_digests)

    source_manifest = json.loads((h50_input_root / "source_manifest.json").read_text(encoding="utf-8"))
    if (
        source_manifest.get("excluded_payloads_opened") != 0
        or source_manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("Input closure violated in source_manifest.json")

    selected = json.loads((h50_input_root / "selected.json").read_text(encoding="utf-8"))
    train_rows = selected["train"]
    dev_rows = selected["dev"]

    train_fams = {str(r["family_id"]) for r in train_rows}
    val_fams = {str(r["family_id"]) for r in dev_rows}
    overlap = train_fams & val_fams
    if overlap:
        raise RuntimeError(f"Train/Val family leakage detected: {sorted(overlap)}")
    if len(train_rows) != DL_TRAIN_EXPECTED_CONTEXTS or len(train_fams) != DL_TRAIN_EXPECTED_FAMILIES:
        raise RuntimeError(
            f"Unexpected DL-Train size: {len(train_rows)} contexts, {len(train_fams)} families"
        )
    if len(dev_rows) != DL_VAL_EXPECTED_CONTEXTS:
        raise RuntimeError(f"Unexpected DL-Val size: {len(dev_rows)} contexts")

    cf_npz = np.load(h58_cache_root / "context_features.npz", allow_pickle=False)
    pp_npz = np.load(h58_cache_root / "pool_priors.npz", allow_pickle=False)

    cid_to_idx = {str(c): i for i, c in enumerate(cf_npz["cids"])}
    all_dope629 = np.asarray(cf_npz["dope629"], dtype=np.float32)
    all_tokens = np.asarray(cf_npz["tokens_10x32"], dtype=np.float32)
    all_mask10 = np.asarray(cf_npz["mask_10"], dtype=np.float32)
    all_lm48z = np.asarray(cf_npz["lm48_z"], dtype=np.float32)
    offsets = np.asarray(pp_npz["offsets"], dtype=np.int64)
    proj16_all = np.asarray(pp_npz["proj16"], dtype=np.float32)

    pools_dir = h50_input_root / "pools"
    tables_dir = h50_input_root / "tables"

    n_tr = len(train_rows)
    tr_cids = [str(r["context_sha256"]) for r in train_rows]
    tr_fids = [str(r["family_id"]) for r in train_rows]
    tr_cf_indices = [cid_to_idx[cid] for cid in tr_cids]

    tr_dope629 = all_dope629[tr_cf_indices]
    tr_tokens = all_tokens[tr_cf_indices]
    tr_mask10 = all_mask10[tr_cf_indices]
    tr_lm48z = all_lm48z[tr_cf_indices]

    tr_cand_indices = np.zeros((n_tr, k_candidates), dtype=np.int64)
    tr_cand_features = np.zeros((n_tr, k_candidates, 428), dtype=np.float32)
    tr_cand_utilities = np.zeros((n_tr, k_candidates), dtype=np.float32)
    tr_cand_gates = np.zeros((n_tr, k_candidates), dtype=np.float32)

    tr_full_features = np.zeros((n_tr, 1152, 428), dtype=np.float32)
    tr_full_fitness = np.zeros((n_tr, 1152), dtype=np.float64)
    tr_full_utilities = np.zeros((n_tr, 1152), dtype=np.float64)
    tr_full_gates = np.zeros((n_tr, 1152), dtype=np.float32)

    for local_i, (row, cid, cf_idx) in enumerate(zip(train_rows, tr_cids, tr_cf_indices, strict=True)):
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("source_split") not in ("train", "inner")
            or row.get("task") != "regression"
            or row.get("has_meta_test") is True
        ):
            raise RuntimeError(f"Forbidden or non-regression train row: {cid}")

        if verify_digests:
            table_bytes = (tables_dir / f"{cid}.jtt1").read_bytes()
            if (
                sha256_bytes(table_bytes) != row["train_only_jtt1_sha256"]
                or table_bytes[:4] != b"JTT1"
                or struct.unpack_from("<I", table_bytes, 8)[0] != 0
                or table_bytes[16] != 2
            ):
                raise RuntimeError(f"Train-only JTT1 table identity/header mismatch for {cid}")

        pool_path = pools_dir / f"{cid}.npz"
        if verify_digests and sha256_file(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"Pool SHA-256 mismatch for {cid}")

        p_npz = np.load(pool_path, allow_pickle=False)
        prog = np.asarray(p_npz["program"], dtype=np.float32)
        fitness = np.asarray(p_npz["fitness"], dtype=np.float64)
        gate_bool = np.asarray(p_npz["gate"], dtype=bool)
        gates = gate_bool.astype(np.float32)
        utilities = np.where(gate_bool, -fitness, -2.0).astype(np.float64)

        lo, hi = int(offsets[cf_idx]), int(offsets[cf_idx + 1])
        proj = proj16_all[lo:hi]
        feats_428 = np.concatenate([prog, proj], axis=1).astype(np.float32)

        sub_idx = select_candidate_subset_indices(
            n_total=feats_428.shape[0],
            k_candidates=k_candidates,
            seed=seed,
            context_idx=local_i,
        )
        tr_cand_indices[local_i] = sub_idx
        tr_cand_features[local_i] = feats_428[sub_idx]
        tr_cand_utilities[local_i] = utilities[sub_idx].astype(np.float32)
        tr_cand_gates[local_i] = gates[sub_idx]

        m = int(feats_428.shape[0])
        tr_full_features[local_i, :m] = feats_428
        tr_full_fitness[local_i, :m] = fitness
        tr_full_utilities[local_i, :m] = utilities
        tr_full_gates[local_i, :m] = gates
        if m < 1152:
            tr_full_features[local_i, m:] = feats_428[-1:]
            tr_full_fitness[local_i, m:] = 2.0
            tr_full_utilities[local_i, m:] = -2.0
            tr_full_gates[local_i, m:] = 0.0

    label_digest = sha256_bytes(tr_cand_indices.tobytes() + tr_cand_utilities.tobytes())
    cache_receipt = {
        "seed": int(seed),
        "k_candidates": int(k_candidates),
        "n_train_contexts": int(n_tr),
        "n_train_families": len(train_fams),
        "label_sha256": label_digest,
    }
    cache.put_bytes(
        f"h204_labels_seed{seed}_k{k_candidates}.json",
        json.dumps(cache_receipt, sort_keys=True).encode("utf-8"),
    )

    n_val = len(dev_rows)
    val_cids = [str(r["context_sha256"]) for r in dev_rows]
    val_fids = [str(r["family_id"]) for r in dev_rows]
    val_cf_indices = [cid_to_idx[cid] for cid in val_cids]

    val_dope629 = all_dope629[val_cf_indices]
    val_tokens = all_tokens[val_cf_indices]
    val_mask10 = all_mask10[val_cf_indices]
    val_lm48z = all_lm48z[val_cf_indices]

    val_full_features = np.zeros((n_val, 1152, 428), dtype=np.float32)
    val_full_fitness = np.zeros((n_val, 1152), dtype=np.float64)
    val_full_utilities = np.zeros((n_val, 1152), dtype=np.float64)
    val_full_gates = np.zeros((n_val, 1152), dtype=np.float32)

    for local_i, (row, cid, cf_idx) in enumerate(zip(dev_rows, val_cids, val_cf_indices, strict=True)):
        pool_path = pools_dir / f"{cid}.npz"
        if verify_digests and sha256_file(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"Dev pool SHA-256 mismatch for {cid}")

        p_npz = np.load(pool_path, allow_pickle=False)
        prog = np.asarray(p_npz["program"], dtype=np.float32)
        fitness = np.asarray(p_npz["fitness"], dtype=np.float64)
        gate_bool = np.asarray(p_npz["gate"], dtype=bool)
        gates = gate_bool.astype(np.float32)
        utilities = np.where(gate_bool, -fitness, -2.0).astype(np.float64)

        lo, hi = int(offsets[cf_idx]), int(offsets[cf_idx + 1])
        proj = proj16_all[lo:hi]
        feats_428 = np.concatenate([prog, proj], axis=1).astype(np.float32)

        m = int(feats_428.shape[0])
        val_full_features[local_i, :m] = feats_428
        val_full_fitness[local_i, :m] = fitness
        val_full_utilities[local_i, :m] = utilities
        val_full_gates[local_i, :m] = gates
        if m < 1152:
            val_full_features[local_i, m:] = feats_428[-1:]
            val_full_fitness[local_i, m:] = 2.0
            val_full_utilities[local_i, m:] = -2.0
            val_full_gates[local_i, m:] = 0.0

    jrb_status = check_jrb01_admission_status()
    labeler_meta = {
        "jrb01_admission": jrb_status,
        "label_source": "pinned_205_family_two_fold_rust_cv_pools",
        "verified_input_digests": verified_digests,
        "k_candidates_per_context": int(k_candidates),
        "n_train_contexts": int(n_tr),
        "n_train_families": len(train_fams),
        "n_val_contexts": int(n_val),
        "n_val_families": len(val_fams),
        "sealed_overlap_count": len(overlap),
        "label_sha256": label_digest,
        "cache_current_bytes": cache.current_bytes(),
        "cache_max_bytes": cache.max_bytes,
    }

    return (
        LabeledContextBatch(
            context_ids=tr_cids,
            family_ids=tr_fids,
            dope629=tr_dope629,
            tokens_10x32=tr_tokens,
            mask_10=tr_mask10,
            lm48_z=tr_lm48z,
            cand_indices=tr_cand_indices,
            cand_features=tr_cand_features,
            cand_utilities=tr_cand_utilities,
            cand_gates=tr_cand_gates,
            full_cand_features=tr_full_features,
            full_fitness=tr_full_fitness,
            full_utilities=tr_full_utilities,
            full_gates=tr_full_gates,
        ),
        EvalContextBatch(
            context_ids=val_cids,
            family_ids=val_fids,
            dope629=val_dope629,
            tokens_10x32=val_tokens,
            mask_10=val_mask10,
            lm48_z=val_lm48z,
            full_cand_features=val_full_features,
            full_fitness=val_full_fitness,
            full_utilities=val_full_utilities,
            full_gates=val_full_gates,
        ),
        labeler_meta,
    )
