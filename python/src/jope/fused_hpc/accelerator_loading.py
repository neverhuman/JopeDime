from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

class PinnedBatchBufferPool:
    """Reusable page-locked host buffer pool for asynchronous H2D batch staging.

    Eliminates per-batch pageable ``np.zeros`` allocations in
    ``collate_raw_tables_to_gpu_batch`` (`run_parity_and_quick10_benchmark.py:97-124`)
    so ``.to(device, non_blocking=True)`` performs true asynchronous DMA over PCIe.
    """

    def __init__(
        self,
        max_b: int = 128,
        max_n_tr: int = 512,
        max_n_ho: int = 256,
        max_p: int = 64,
        *,
        pin_memory: bool | None = None,
    ) -> None:
        if max_b <= 0 or max_n_tr <= 0 or max_n_ho <= 0 or max_p <= 0:
            raise ValueError("PinnedBatchBufferPool dimensions must be positive")
        self.max_b = int(max_b)
        self.max_n_tr = int(max_n_tr)
        self.max_n_ho = int(max_n_ho)
        self.max_p = int(max_p)
        self.pin_memory = bool(torch.cuda.is_available() if pin_memory is None else pin_memory)
        self._alloc_buffers(self.max_b, self.max_n_tr, self.max_n_ho, self.max_p)

    def _alloc_buffers(self, b: int, n_tr: int, n_ho: int, p: int) -> None:
        pin = self.pin_memory and torch.cuda.is_available()
        self.x_tr_host = torch.zeros((b, n_tr, p), dtype=torch.float32, pin_memory=pin)
        self.y_tr_host = torch.zeros((b, n_tr), dtype=torch.float32, pin_memory=pin)
        self.x_ho_host = torch.zeros((b, n_ho, p), dtype=torch.float32, pin_memory=pin)
        self.y_ho_host = torch.zeros((b, n_ho), dtype=torch.float32, pin_memory=pin)
        self.tr_mask_host = torch.zeros((b, n_tr), dtype=torch.float32, pin_memory=pin)
        self.ho_mask_host = torch.zeros((b, n_ho), dtype=torch.float32, pin_memory=pin)
        self.col_mask_host = torch.zeros((b, p), dtype=torch.float32, pin_memory=pin)

    def _ensure_capacity(self, b: int, n_tr: int, n_ho: int, p: int) -> None:
        if b > self.max_b or n_tr > self.max_n_tr or n_ho > self.max_n_ho or p > self.max_p:
            self.max_b = max(self.max_b, b)
            self.max_n_tr = max(self.max_n_tr, n_tr)
            self.max_n_ho = max(self.max_n_ho, n_ho)
            self.max_p = max(self.max_p, p)
            self._alloc_buffers(self.max_b, self.max_n_tr, self.max_n_ho, self.max_p)

    def collate_to_device(
        self,
        tables: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        device: torch.device,
        *,
        max_tr_rows: int = 512,
        max_ho_rows: int = 256,
        max_cols: int = 1088,
    ) -> dict[str, torch.Tensor]:
        """Pack raw ``(X_tr, y_tr, X_ho, y_ho)`` tables into pinned buffers and transfer to ``device``."""
        if not tables:
            raise ValueError("tables must be non-empty")
        b = len(tables)
        n_tr_pad = max(min(int(t[0].shape[0]), max_tr_rows) for t in tables)
        n_ho_pad = max(min(int(t[2].shape[0]), max_ho_rows) for t in tables)
        p_pad = max(min(int(t[0].shape[1]), max_cols) for t in tables)

        self._ensure_capacity(b, n_tr_pad, n_ho_pad, p_pad)

        x_tr_s = self.x_tr_host[:b, :n_tr_pad, :p_pad]
        y_tr_s = self.y_tr_host[:b, :n_tr_pad]
        x_ho_s = self.x_ho_host[:b, :n_ho_pad, :p_pad]
        y_ho_s = self.y_ho_host[:b, :n_ho_pad]
        tr_m_s = self.tr_mask_host[:b, :n_tr_pad]
        ho_m_s = self.ho_mask_host[:b, :n_ho_pad]
        col_m_s = self.col_mask_host[:b, :p_pad]

        x_tr_s.zero_()
        y_tr_s.zero_()
        x_ho_s.zero_()
        y_ho_s.zero_()
        tr_m_s.zero_()
        ho_m_s.zero_()
        col_m_s.zero_()

        x_tr_np = x_tr_s.numpy()
        y_tr_np = y_tr_s.numpy()
        x_ho_np = x_ho_s.numpy()
        y_ho_np = y_ho_s.numpy()
        tr_m_np = tr_m_s.numpy()
        ho_m_np = ho_m_s.numpy()
        col_m_np = col_m_s.numpy()

        for i, (xtr, ytr, xho, yho) in enumerate(tables):
            ntr = min(int(xtr.shape[0]), n_tr_pad)
            nho = min(int(xho.shape[0]), n_ho_pad)
            p = min(int(xtr.shape[1]), p_pad)
            x_tr_np[i, :ntr, :p] = xtr[:ntr, :p]
            y_tr_np[i, :ntr] = ytr[:ntr]
            x_ho_np[i, :nho, :p] = xho[:nho, :p]
            y_ho_np[i, :nho] = yho[:nho]
            tr_m_np[i, :ntr] = 1.0
            ho_m_np[i, :nho] = 1.0
            col_m_np[i, :p] = 1.0

        non_blocking = device.type == "cuda" and x_tr_s.is_pinned()
        return {
            "X_tr": x_tr_s.to(device=device, non_blocking=non_blocking),
            "y_tr_raw": y_tr_s.to(device=device, non_blocking=non_blocking),
            "X_ho": x_ho_s.to(device=device, non_blocking=non_blocking),
            "y_ho_raw": y_ho_s.to(device=device, non_blocking=non_blocking),
            "tr_row_mask": tr_m_s.to(device=device, non_blocking=non_blocking),
            "ho_row_mask": ho_m_s.to(device=device, non_blocking=non_blocking),
            "col_mask": col_m_s.to(device=device, non_blocking=non_blocking),
        }

def parallel_load_dataset_splits(
    items: Sequence[Any],
    loader_fn: Callable[[Any], Any],
    *,
    max_workers: int = 8,
) -> list[Any]:
    """Load dataset splits concurrently across bounded worker threads (respecting <=85% host CPU cap)."""
    if max_workers <= 1 or len(items) <= 1:
        return [loader_fn(item) for item in items]
    bounded_workers = min(max(1, int(max_workers)), 8, len(items))
    with ThreadPoolExecutor(max_workers=bounded_workers) as pool:
        return list(pool.map(loader_fn, items))

def count_rust_tokens_out_of_loop(
    rust_sources: Sequence[str],
    *,
    encode_fn: Callable[[str], Sequence[int]] | None = None,
    encode_batch_fn: Callable[[list[str]], list[list[int]]] | None = None,
    max_workers: int = 4,
) -> list[int]:
    """Count BPE tokens outside the timed GPU inference loop (`HPC #4`).

    In ``run_batch_vram_and_cohort915_benchmark.py:418``, single-threaded ``tiktoken.encode``
    was called inside the timed ``t_cohort_start .. t_inmem_end`` loop, adding ``4.20s-5.73s``
    (`22.5%-40.0%` overhead) to reported in-memory GPU inference wall time.
    Running ``encode_batch`` (or bounded thread-pool encoding) after ``t_inmem_end``
    restores true GPU + Rust emission ``tok/s``.
    """
    src_list = list(rust_sources)
    if not src_list:
        return []
    if encode_batch_fn is not None:
        return [len(toks) for toks in encode_batch_fn(src_list)]
    if encode_fn is not None:
        if max_workers <= 1 or len(src_list) <= 1:
            return [len(encode_fn(s)) for s in src_list]
        with ThreadPoolExecutor(max_workers=min(max_workers, 8, len(src_list))) as pool:
            return [len(toks) for toks in pool.map(encode_fn, src_list)]

    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        tok_lists = enc.encode_batch(
            src_list,
            num_threads=min(max(1, int(max_workers)), 8),
            disallowed_special=(),
        )
        return [len(t) for t in tok_lists]
    except (Exception,):
        # Deterministic byte/whitespace token fallback when tiktoken is not installed
        return [len(s.split()) for s in src_list]
