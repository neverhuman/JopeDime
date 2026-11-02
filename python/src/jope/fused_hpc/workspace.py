"""Persistent winner workspace types for Phase 2 fused HPC kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from jope.fused_hpc.block256_slate_and_workspace import (
    IBUF_HIST_EXTRA_INTS,
    IBUF_STRIDE_PER_ROW,
    RBUF_STRIDE_PER_ROW,
)

class Phase2HeadSliceSpec:
    """Per-winner buffer layout specification for a Phase 2 head family."""

    family: str
    rbuf_floats_per_row: int
    ibuf_ints_per_row: int
    ibuf_const_ints: int


PHASE2_HEAD_SLICE_SPECS: dict[str, Phase2HeadSliceSpec] = {
    "ridge_gcv": Phase2HeadSliceSpec("ridge_gcv", 260, 0, 0),
    "pls1_nipals": Phase2HeadSliceSpec("pls1_nipals", 260, 0, 0),
    "knn_3_7_15": Phase2HeadSliceSpec("knn_3_7_15", 260, 4, 0),
    "ksmooth": Phase2HeadSliceSpec("ksmooth", 260, 4, 0),
    "histgbm": Phase2HeadSliceSpec("histgbm", 260, 268, IBUF_HIST_EXTRA_INTS),
}
class Phase2WinnerWorkspace:
    """Phase 2 HPC #6: Persistent non-aliasing workspace pool for Phase 2 head evaluators.

    Pre-allocates reusable ``rbuf_all`` (``W_max x (260 * R_max)`` ``float64``) and
    ``ibuf_all`` (``W_max x (268 * R_max + 2048)`` ``int32``) buffers so that ``<<<W_n, 256>>>``
    Phase 2 head kernels execute with zero per-batch ``cudaMalloc`` / ``cudaFree`` or full-workspace
    ``cudaMemsetAsync`` overhead and without buffer overruns on ``knn_3_7_15``, ``ksmooth``, or
    ``histgbm``.
    """

    def __init__(
        self,
        max_winners: int = 96,
        max_rows: int = 4096,
        device: str | torch.device = "cpu",
    ) -> None:
        if max_winners <= 0 or max_rows <= 0:
            raise ValueError("max_winners and max_rows must be positive")
        self.max_winners = int(max_winners)
        self.max_rows = int(max_rows)
        self.device = torch.device(device)
        self.rbuf_stride = RBUF_STRIDE_PER_ROW * self.max_rows
        self.ibuf_stride = IBUF_STRIDE_PER_ROW * self.max_rows + IBUF_HIST_EXTRA_INTS
        self.rbuf_all = torch.empty(
            (self.max_winners, self.rbuf_stride),
            dtype=torch.float64,
            device=self.device,
        )
        self.ibuf_all = torch.empty(
            (self.max_winners, self.ibuf_stride),
            dtype=torch.int32,
            device=self.device,
        )
        self.acquire_count = 0
        self.realloc_count = 0

    def ensure_capacity(
        self,
        n_winners: int,
        n_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return views for ``(n_winners, rbuf_stride(n_rows))`` without reallocating when in bounds."""
        if n_winners <= 0 or n_rows <= 0:
            raise ValueError("n_winners and n_rows must be positive")
        if n_winners > self.max_winners or n_rows > self.max_rows:
            self.max_winners = max(self.max_winners, int(n_winners))
            self.max_rows = max(self.max_rows, int(n_rows))
            self.rbuf_stride = RBUF_STRIDE_PER_ROW * self.max_rows
            self.ibuf_stride = IBUF_STRIDE_PER_ROW * self.max_rows + IBUF_HIST_EXTRA_INTS
            self.rbuf_all = torch.empty(
                (self.max_winners, self.rbuf_stride),
                dtype=torch.float64,
                device=self.device,
            )
            self.ibuf_all = torch.empty(
                (self.max_winners, self.ibuf_stride),
                dtype=torch.int32,
                device=self.device,
            )
            self.realloc_count += 1
        self.acquire_count += 1
        req_rbuf = RBUF_STRIDE_PER_ROW * int(n_rows)
        req_ibuf = IBUF_STRIDE_PER_ROW * int(n_rows) + IBUF_HIST_EXTRA_INTS
        return (
            self.rbuf_all[:n_winners, :req_rbuf],
            self.ibuf_all[:n_winners, :req_ibuf],
        )

    def verify_non_aliasing(self, n_winners: int, n_rows: int) -> dict[str, Any]:
        """Verify that every Phase 2 head family's buffer offsets fit within the per-winner stride."""
        rbuf_view, ibuf_view = self.ensure_capacity(n_winners, n_rows)
        r_req = RBUF_STRIDE_PER_ROW * int(n_rows)
        i_req = IBUF_STRIDE_PER_ROW * int(n_rows) + IBUF_HIST_EXTRA_INTS

        # Explicit head-family offset checks matching phase2_universe_kernels.cu [D1]/[D3]:
        # - ridge_gcv / pls1_nipals / knn_3_7_15 / ksmooth: X_tr (256*R), resid/pred (4*R) -> 260*R
        # - histgbm: X_tr (256*R), resid/g/h/pred (4*R) -> 260*R floats;
        #   q_tr (256*R) + node_id (8*R) + valid_idx (4*R) + tree tables (2048) -> 268*R + 2048 ints
        family_footprints = {
            "ridge_gcv": {"rbuf_end": 260 * n_rows, "ibuf_end": 0},
            "pls1_nipals": {"rbuf_end": 260 * n_rows, "ibuf_end": 0},
            "knn_3_7_15": {"rbuf_end": 260 * n_rows, "ibuf_end": 4 * n_rows},
            "ksmooth": {"rbuf_end": 260 * n_rows, "ibuf_end": 4 * n_rows},
            "histgbm": {
                "rbuf_end": 260 * n_rows,
                "ibuf_end": 268 * n_rows + IBUF_HIST_EXTRA_INTS,
            },
        }
        for fam, bounds in family_footprints.items():
            if bounds["rbuf_end"] > rbuf_view.shape[1]:
                raise AssertionError(f"{fam} rbuf overflow: {bounds['rbuf_end']} > {rbuf_view.shape[1]}")
            if bounds["ibuf_end"] > ibuf_view.shape[1]:
                raise AssertionError(f"{fam} ibuf overflow: {bounds['ibuf_end']} > {ibuf_view.shape[1]}")
        return {
            "n_winners": int(n_winners),
            "n_rows": int(n_rows),
            "rbuf_stride": r_req,
            "ibuf_stride": i_req,
            "families_ok": True,
            "family_footprints": family_footprints,
        }
