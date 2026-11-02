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

from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import shutil
import time
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

DEFAULT_H50_INPUT_ROOT = Path("/home/ubuntu/jope-data/h50/input")
DEFAULT_H58_CONTEXT_CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")
DEFAULT_JRB01_MANIFEST = Path("/home/ubuntu/jope-data/jrb01/admitted_train_datasets.json")


def check_disk_budget(
    path: Path,
    max_used_ratio: float = MAX_DISK_USED_RATIO,
    min_free_bytes: int = MIN_DISK_FREE_BYTES,
) -> dict[str, Any]:
    """Verify disk usage on `xbabe2` stays below 93.8% and >= 40 GiB free."""
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    denom = usage.used + usage.free
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
            raise ValueError(
                f"Single payload ({incoming_bytes} B) exceeds cache cap ({self.max_bytes} B)"
            )
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
    dope629: np.ndarray           # (N, 629) float32
    tokens_10x32: np.ndarray      # (N, 10, 32) float32
    context_summary: np.ndarray   # (N, 29) float32
    anchor_Summary: np.ndarray    # (N, 24) float32
    cand_indices: np.ndarray      # (N, K) int64 in [0, 1152)
    cand_features: np.ndarray     # (N, K, 42) float32
    cand_costs: np.ndarray        # (N, K) float32 (-fitness from two-fold Rust CV)
    cand_gates: np.ndarray        # (N, K) float32
    full_cand_features: np.ndarray  # (N, 1152, 42) float32 for OOF / argmin evaluation
    full_cand_costs: np.ndarray   # (N, 1152) float32 for OOF / argmin evaluation
    full_cand_gates: np.ndarray   # (N, 1152) float32
    r882_picks: np.ndarray        # (N,) int64
    r882_costs: np.ndarray        # (N,) float64


@dataclass(frozen=True)
class EvalContextBatch:
    """Evaluation context batch (DL-Val 91 contexts) with strictly no labeler training access."""

    context_ids: list[str]
    family_ids: list[str]
    dope629: np.ndarray           # (N, 629) float32
    tokens_10x32: np.ndarray      # (N, 10, 32) float32
    context_summary: np.ndarray   # (N, 29) float32
    anchor_Summary: np.ndarray    # (N, 24) float32
    full_cand_features: np.ndarray  # (N, 1152, 42) float32
    full_cand_costs: np.ndarray   # (N, 1152) float64
    full_cand_gates: np.ndarray   # (N, 1152) float32
    r882_picks: np.ndarray        # (N,) int64
    r882_costs: np.ndarray        # (N,) float64


def select_candidate_subset_indices(
    n_total: int = 1152,
    k_candidates: int = 16,
    seed: int = 204101,
    context_idx: int = 0,
) -> np.ndarray:
    """Deterministically select `k_candidates` (8..16) indices per context.

    Always includes the 4 revealed anchors (indices 0, 1, 2, 3) plus `k_candidates - 4`
    stratified candidate indices across the 1,148 non-anchor slots.
    """
    if not (8 <= k_candidates <= 16):
        raise ValueError(f"k_candidates must be in [8, 16], got {k_candidates}")
    if n_total < k_candidates:
        raise ValueError(f"n_total ({n_total}) < k_candidates ({k_candidates})")

    anchors = np.array([0, 1, 2, 3], dtype=np.int64)
    n_extra = k_candidates - 4
    rng = np.random.default_rng(seed + 1009 * int(context_idx))
    non_anchors = np.arange(4, n_total, dtype=np.int64)
    # Stratified draw across `n_extra` contiguous bins of the 1,148 candidate slots
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
    programs: list[str],
    cache: LocalLRUCache,
    cache_key: str,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Evaluate `8-16` GP candidate programs via exact two-fold Rust CV (`run_outer_test=False`)."""
    if not (1 <= max_workers <= MAX_LABELER_WORKERS):
        raise ValueError(f"max_workers must be in [1, {MAX_LABELER_WORKERS}], got {max_workers}")
    if not (8 <= len(programs) <= 16):
        raise ValueError(f"Expected 8..16 candidate programs per context, got {len(programs)}")

    cached = cache.get_bytes(cache_key)
    if cached is not None:
        return json.loads(cached.decode("utf-8"))

    from research.flywheel_selfplay.rust_evaluator import evaluate_gp_batch_rust

    # Compress raw JTT1 if not already gzipped
    jtt1_gz = (
        raw_jtt1_bytes
        if raw_jtt1_bytes[:2] == b"\x1f\x8b"
        else gzip.compress(raw_jtt1_bytes, mtime=0)
    )
    res = evaluate_gp_batch_rust(
        jtt1_gz_bytes=jtt1_gz,
        programs=programs,
        timeout_sec=30.0,
        max_workers=max_workers,
        run_outer_test=False,
    )
    serialized = json.dumps(res, sort_keys=True).encode("utf-8")
    cache.put_bytes(cache_key, serialized)
    return res


def load_h204_stage1_datasets(
    *,
    cache: LocalLRUCache,
    h50_input_root: Path = DEFAULT_H50_INPUT_ROOT,
    h58_context_cache: Path = DEFAULT_H58_CONTEXT_CACHE,
    k_candidates: int = 16,
    seed: int = 204101,
) -> tuple[LabeledContextBatch, EvalContextBatch, dict[str, Any]]:
    """Load and label the admitted 205-family DL-Train pool (`8-16` CV candidates/ctx) and 91 DL-Val pool.

    Enforces strict train/val family disjointness (zero overlap between DL-Train 205 families
    and DL-Val 39 families) and caches the labeled candidate subset in `LocalLRUCache` (`<= 20 GB`).
    """
    if not (8 <= k_candidates <= 16):
        raise ValueError(f"k_candidates must be in [8, 16], got {k_candidates}")

    ctx_npz = np.load(h58_context_cache, allow_pickle=False)
    all_cids = [str(x) for x in ctx_npz["cids"]]
    all_fids = [str(x) for x in ctx_npz["fids"]]
    all_splits = [str(x) for x in ctx_npz["splits"]]
    all_dope629 = np.asarray(ctx_npz["dope629"], dtype=np.float32)
    all_tokens = np.asarray(ctx_npz["tokens_10x32"], dtype=np.float32)
    all_ctx_sum = np.asarray(ctx_npz["context_summary"], dtype=np.float32)
    all_anc_sum = np.asarray(ctx_npz["anchor_summary"], dtype=np.float32)

    train_idx = [i for i, s in enumerate(all_splits) if s == "train"]
    val_idx = [i for i, s in enumerate(all_splits) if s == "val"]

    train_fams = {all_fids[i] for i in train_idx}
    val_fams = {all_fids[i] for i in val_idx}
    overlap = train_fams & val_fams
    if overlap:
        raise RuntimeError(f"Train/Val family leakage detected: {sorted(overlap)}")
    if len(train_idx) != DL_TRAIN_EXPECTED_CONTEXTS or len(train_fams) != DL_TRAIN_EXPECTED_FAMILIES:
        raise RuntimeError(
            f"Unexpected DL-Train size: {len(train_idx)} contexts, {len(train_fams)} families"
        )
    if len(val_idx) != DL_VAL_EXPECTED_CONTEXTS:
        raise RuntimeError(f"Unexpected DL-Val size: {len(val_idx)} contexts")

    pools_dir = h50_input_root / "pools"

    # Build Training Labeled Batch
    n_tr = len(train_idx)
    tr_cids = [all_cids[i] for i in train_idx]
    tr_fids = [all_fids[i] for i in train_idx]
    tr_dope629 = all_dope629[train_idx]
    tr_tokens = all_tokens[train_idx]
    tr_ctx_sum = all_ctx_sum[train_idx]
    tr_anc_sum = all_anc_sum[train_idx]

    tr_cand_indices = np.zeros((n_tr, k_candidates), dtype=np.int64)
    tr_cand_features = np.zeros((n_tr, k_candidates, 42), dtype=np.float32)
    tr_cand_costs = np.zeros((n_tr, k_candidates), dtype=np.float32)
    tr_cand_gates = np.zeros((n_tr, k_candidates), dtype=np.float32)
    tr_full_features = np.zeros((n_tr, 1152, 42), dtype=np.float32)
    tr_full_costs = np.zeros((n_tr, 1152), dtype=np.float32)
    tr_full_gates = np.zeros((n_tr, 1152), dtype=np.float32)
    tr_r882_picks = np.zeros(n_tr, dtype=np.int64)
    tr_r882_costs = np.zeros(n_tr, dtype=np.float64)

    for local_i, cid in enumerate(tr_cids):
        p_npz = np.load(pools_dir / f"{cid}.npz", allow_pickle=False)
        feats = np.asarray(p_npz["cand_features"], dtype=np.float32)
        fitness = np.asarray(p_npz["fitness"], dtype=np.float64)
        gates = np.asarray(p_npz["gate"], dtype=np.float32)
        costs = (-fitness).astype(np.float32)
        r882_pick = int(np.asarray(p_npz["r882_pick"]).item())

        sub_idx = select_candidate_subset_indices(
            n_total=feats.shape[0],
            k_candidates=k_candidates,
            seed=seed,
            context_idx=local_i,
        )
        tr_cand_indices[local_i] = sub_idx
        tr_cand_features[local_i] = feats[sub_idx]
        tr_cand_costs[local_i] = costs[sub_idx]
        tr_cand_gates[local_i] = gates[sub_idx]
        tr_full_features[local_i] = feats
        tr_full_costs[local_i] = costs
        tr_full_gates[local_i] = gates
        tr_r882_picks[local_i] = r882_pick
        tr_r882_costs[local_i] = float(-fitness[r882_pick])

    # Record compact labeler receipt in bounded cache
    label_digest = sha256_bytes(tr_cand_indices.tobytes() + tr_cand_costs.tobytes())
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

    # Build DL-Val Evaluation Batch (strictly isolated from training labels)
    n_val = len(val_idx)
    val_cids = [all_cids[i] for i in val_idx]
    val_fids = [all_fids[i] for i in val_idx]
    val_dope629 = all_dope629[val_idx]
    val_tokens = all_tokens[val_idx]
    val_ctx_sum = all_ctx_sum[val_idx]
    val_anc_sum = all_anc_sum[val_idx]

    val_full_features = np.zeros((n_val, 1152, 42), dtype=np.float32)
    val_full_costs = np.zeros((n_val, 1152), dtype=np.float64)
    val_full_gates = np.zeros((n_val, 1152), dtype=np.float32)
    val_r882_picks = np.zeros(n_val, dtype=np.int64)
    val_r882_costs = np.zeros(n_val, dtype=np.float64)

    for local_i, cid in enumerate(val_cids):
        p_npz = np.load(pools_dir / f"{cid}.npz", allow_pickle=False)
        feats = np.asarray(p_npz["cand_features"], dtype=np.float32)
        fitness = np.asarray(p_npz["fitness"], dtype=np.float64)
        gates = np.asarray(p_npz["gate"], dtype=np.float32)
        r882_pick = int(np.asarray(p_npz["r882_pick"]).item())

        val_full_features[local_i] = feats
        val_full_costs[local_i] = -fitness
        val_full_gates[local_i] = gates
        val_r882_picks[local_i] = r882_pick
        val_r882_costs[local_i] = float(-fitness[r882_pick])

    jrb_status = check_jrb01_admission_status()
    labeler_meta = {
        "jrb01_admission": jrb_status,
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
            context_summary=tr_ctx_sum,
            anchor_Summary=tr_anc_sum,
            cand_indices=tr_cand_indices,
            cand_features=tr_cand_features,
            cand_costs=tr_cand_costs,
            cand_gates=tr_cand_gates,
            full_cand_features=tr_full_features,
            full_cand_costs=tr_full_costs,
            full_cand_gates=tr_full_gates,
            r882_picks=tr_r882_picks,
            r882_costs=tr_r882_costs,
        ),
        EvalContextBatch(
            context_ids=val_cids,
            family_ids=val_fids,
            dope629=val_dope629,
            tokens_10x32=val_tokens,
            context_summary=val_ctx_sum,
            anchor_Summary=val_anc_sum,
            full_cand_features=val_full_features,
            full_cand_costs=val_full_costs,
            full_cand_gates=val_full_gates,
            r882_picks=val_r882_picks,
            r882_costs=val_r882_costs,
        ),
        labeler_meta,
    )
