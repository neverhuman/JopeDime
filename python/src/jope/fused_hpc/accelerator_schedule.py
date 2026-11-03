from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .accelerator import ALLOWED_GPU_NODES, DEFAULT_NODE_THROUGHPUT_WEIGHTS
from .accelerator_sort_corr import sort_valid_rows_radix_f64, sort_valid_values_radix_f64


@dataclass(frozen=True)
class ShapeBucketPlan:
    """Summary and batch schedule for 2D ``(P_tier, N_tr)`` shape bucketing."""

    ordered_indices: tuple[int, ...]
    batch_ranges: tuple[tuple[int, int], ...]
    batch_tier_labels: tuple[str, ...]
    padded_cells_1d: int
    padded_cells_2d: int
    padding_reduction_ratio: float


def _compute_schedule_padded_cells(
    rows_and_cols: Sequence[tuple[int, int]],
    ordered_indices: Sequence[int],
    batch_ranges: Sequence[tuple[int, int]],
    *,
    max_n_tr_cap: int = 512,
    max_p_cap: int = 1088,
) -> int:
    total_cells = 0
    for start, end in batch_ranges:
        if end <= start:
            continue
        batch_idx = ordered_indices[start:end]
        b_len = len(batch_idx)
        n_pad = max(min(int(rows_and_cols[i][0]), max_n_tr_cap) for i in batch_idx)
        p_pad = max(min(int(rows_and_cols[i][1]), max_p_cap) for i in batch_idx)
        total_cells += b_len * n_pad * p_pad
    return total_cells


def bucket_tables_2d_by_p_tier_and_ntr(
    rows_and_cols: Sequence[tuple[int, int]],
    *,
    p_tiers: tuple[int, int] = (16, 64),
    batch_sizes_by_tier: tuple[int, int, int] = (64, 32, 16),
    baseline_batch_size: int = 32,
    max_n_tr_cap: int = 512,
    max_p_cap: int = 1088,
) -> ShapeBucketPlan:
    """Partition tables into 2D ``(P_tier, N_tr_capped, P)`` buckets to eliminate column-padding waste.

    In ``run_batch_vram_and_cohort915_benchmark.py:286``, sorting by uncapped ``(N_tr, P)``
    places wide tables (`P = 376..1087`) into the same 32-table batch as narrow tables
    (`P = 4..12`), inflating padded ``B * N_pad * P_pad`` volume by ``~2.5x-3.6x``.
    Partitioning first into three column-width tiers (`P <= 16`, `17 <= P <= 64`, `P > 64`)
    and sorting by ``(min(N_tr, max_n_tr_cap), P)`` within each tier (and by ``(P, min(N_tr, max_n_tr_cap))``
    for the wide ``P > 64`` tier) isolates wide tables and cuts padded tensor volume by ``~3x``.
    """
    n_items = len(rows_and_cols)
    if n_items == 0:
        return ShapeBucketPlan(
            ordered_indices=(),
            batch_ranges=(),
            batch_tier_labels=(),
            padded_cells_1d=0,
            padded_cells_2d=0,
            padding_reduction_ratio=1.0,
        )

    p_t0, p_t1 = int(p_tiers[0]), int(p_tiers[1])
    b_s0, b_s1, b_s2 = (
        max(1, int(batch_sizes_by_tier[0])),
        max(1, int(batch_sizes_by_tier[1])),
        max(1, int(batch_sizes_by_tier[2])),
    )

    # Baseline 1D sort by uncapped (N_tr, P) with fixed baseline_batch_size=32
    order_1d = sorted(
        range(n_items),
        key=lambda i: (int(rows_and_cols[i][0]), int(rows_and_cols[i][1]), i),
    )
    ranges_1d = [
        (s, min(s + baseline_batch_size, n_items)) for s in range(0, n_items, max(1, baseline_batch_size))
    ]
    padded_cells_1d = _compute_schedule_padded_cells(
        rows_and_cols,
        order_1d,
        ranges_1d,
        max_n_tr_cap=max_n_tr_cap,
        max_p_cap=max_p_cap,
    )

    # 2D (P_tier, N_tr_capped, P) bucketing
    tier_buckets: list[list[int]] = [[], [], []]
    tier_names = (f"P<={p_t0}", f"{p_t0 + 1}<=P<={p_t1}", f"P>{p_t1}")
    tier_batch_sizes = (b_s0, b_s1, b_s2)

    for idx, (_n_r, p_c) in enumerate(rows_and_cols):
        if p_c <= p_t0:
            tier_buckets[0].append(idx)
        elif p_c <= p_t1:
            tier_buckets[1].append(idx)
        else:
            tier_buckets[2].append(idx)

    ordered_2d: list[int] = []
    batch_ranges_2d: list[tuple[int, int]] = []
    batch_labels_2d: list[str] = []

    for t_idx, bucket in enumerate(tier_buckets):
        if not bucket:
            continue
        if t_idx == 2:
            bucket.sort(
                key=lambda i: (
                    int(rows_and_cols[i][1]),
                    min(int(rows_and_cols[i][0]), max_n_tr_cap),
                    i,
                )
            )
        else:
            bucket.sort(
                key=lambda i: (
                    min(int(rows_and_cols[i][0]), max_n_tr_cap),
                    int(rows_and_cols[i][1]),
                    i,
                )
            )
        b_size = tier_batch_sizes[t_idx]
        base_offset = len(ordered_2d)
        ordered_2d.extend(bucket)
        for local_s in range(0, len(bucket), b_size):
            local_e = min(local_s + b_size, len(bucket))
            batch_ranges_2d.append((base_offset + local_s, base_offset + local_e))
            batch_labels_2d.append(tier_names[t_idx])

    padded_cells_2d = _compute_schedule_padded_cells(
        rows_and_cols,
        ordered_2d,
        batch_ranges_2d,
        max_n_tr_cap=max_n_tr_cap,
        max_p_cap=max_p_cap,
    )
    ratio = float(padded_cells_1d) / float(max(1, padded_cells_2d))

    return ShapeBucketPlan(
        ordered_indices=tuple(ordered_2d),
        batch_ranges=tuple(batch_ranges_2d),
        batch_tier_labels=tuple(batch_labels_2d),
        padded_cells_1d=int(padded_cells_1d),
        padded_cells_2d=int(padded_cells_2d),
        padding_reduction_ratio=ratio,
    )


class FusedStageCudaGraphRunner:
    """Captures and replays ``CUDAGraph``s for fixed-shape Stage 1A-neural and Stage 2 sub-networks.

    In ``FusedJopeNetwork``, Stage 1A's ``_run_dope_mlc_v3_and_lm48(ms_376, col_lm_64x17)``
    and Stage 2's ``_run_stage2_router_qhat_spq1_batched_gpu(...)`` operate on tensors whose
    feature dimensions (`376`, `64x17`, `96x412`, `4x24`) are independent of table shape
    ``(N_tr, P)``. Capturing per-batch-size ``CUDAGraph``s eliminates Python/driver launch
    overhead while automatically falling back to eager execution on CPU or uncaptured shapes.
    """

    def __init__(
        self,
        stage1a_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        *,
        enable_cuda_graphs: bool = True,
    ) -> None:
        self.stage1a_fn = stage1a_fn
        self.enable_cuda_graphs = bool(enable_cuda_graphs)
        self._captured_stage1a: dict[
            tuple[int, torch.device, torch.dtype],
            tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def run_stage1a_neural(
        self,
        ms_376: torch.Tensor,
        col_lm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute ``_run_dope_mlc_v3_and_lm48`` via captured ``CUDAGraph`` on CUDA or eager fallback."""
        if not self.enable_cuda_graphs or not ms_376.is_cuda or not col_lm.is_cuda or torch.is_grad_enabled():
            with torch.no_grad():
                return self.stage1a_fn(ms_376, col_lm)

        bsz = int(ms_376.shape[0])
        key = (bsz, ms_376.device, ms_376.dtype)
        entry = self._captured_stage1a.get(key)
        if entry is None:
            static_ms = torch.empty_like(ms_376).copy_(ms_376)
            static_col = torch.empty_like(col_lm).copy_(col_lm)
            # Warmup before capture
            with torch.no_grad():
                _ = self.stage1a_fn(static_ms, static_col)
            torch.cuda.synchronize(ms_376.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                static_dope629, static_lm48 = self.stage1a_fn(static_ms, static_col)
            entry = (graph, static_ms, static_col, static_dope629, static_lm48)
            self._captured_stage1a[key] = entry

        graph, static_ms, static_col, static_dope629, static_lm48 = entry
        static_ms.copy_(ms_376)
        static_col.copy_(col_lm)
        graph.replay()
        return static_dope629.clone(), static_lm48.clone()


def plan_multigpu_shards(
    items: Sequence[dict[str, Any]],
    *,
    node_weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically partition datasets across ``xbabe1``, ``xbabe2``, and ``xbabe3`` (`HPC #5`).

    Strictly enforces the ``NORTH_STAR.md`` / ``COMPUTE.md`` hard rule: **NEVER ``xbabe0``**.
    Balances estimated table compute cost ``max(1, rows) * max(1, features)`` across
    active GPU nodes proportionally to ``node_weights`` (defaulting to
    ``{"xbabe1": 0.45, "xbabe2": 0.275, "xbabe3": 0.275}``).
    """
    weights = dict(DEFAULT_NODE_THROUGHPUT_WEIGHTS if node_weights is None else node_weights)
    if not weights:
        raise ValueError("node_weights must be non-empty")
    for node_name in weights:
        if node_name == "xbabe0" or node_name not in ALLOWED_GPU_NODES:
            raise ValueError(
                f"Forbidden or unknown GPU node {node_name!r}; allowed nodes are {ALLOWED_GPU_NODES}"
            )
    if any(w <= 0.0 or not math.isfinite(w) for w in weights.values()):
        raise ValueError("All node_weights must be positive finite numbers")

    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}
    shards: dict[str, list[dict[str, Any]]] = {k: [] for k in weights}
    accum_cost: dict[str, float] = {k: 0.0 for k in weights}

    # Sort descending by estimated work (rows * features) with deterministic dataset_id tie-break
    def _item_cost(it: dict[str, Any]) -> float:
        r = max(1, int(it.get("rows", 1)))
        c = max(1, int(it.get("features", 1)))
        return float(r * c)

    sorted_items = sorted(
        items,
        key=lambda it: (-_item_cost(it), str(it.get("dataset_id", ""))),
    )

    for it in sorted_items:
        cost = _item_cost(it)
        # Assign to the node with the smallest normalized load (accum_cost[node] / weight[node])
        best_node = min(
            weights.keys(),
            key=lambda k: (accum_cost[k] / norm_weights[k], k),
        )
        shards[best_node].append(it)
        accum_cost[best_node] += cost

    return shards


def _snapshot_dope_state_sha256(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 over ``model.dope_model.state_dict()``."""
    dope = getattr(model, "dope_model", None)
    if not isinstance(dope, nn.Module):
        raise ValueError("Model does not contain an nn.Module 'dope_model' submodule")
    h = hashlib.sha256()
    for name, tensor in sorted(dope.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        header = f"{name}:{cpu.dtype}:{tuple(cpu.shape)}:".encode()
        h.update(header)
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def patch_fused_network_hpc(
    fused_net: nn.Module,
    *,
    gpu_stage1a_module: Any | None = None,
    enable_cuda_graphs: bool = True,
    force_stage2_fp32: bool = True,
) -> dict[str, Any]:
    """Attach Phase E HPC optimizations to ``fused_net`` while strictly locking DOPE weights.

    1. Verifies and enforces the ``NORTH_STAR.md`` hard requirement on ``fused_net.dope_model``:
       all parameters have ``requires_grad=False``, ``grad=None``, and all nested DOPE modules
       are locked in ``.eval()`` mode with identical pre/post SHA-256 state checksums.
    2. Wraps ``_run_dope_mlc_v3_and_lm48`` in ``FusedStageCudaGraphRunner`` when enabled.
    3. Disables the zero-benefit ``fp16`` autocast in Stage 2 (``force_stage2_fp32=True``)
       that caused 6 winner mismatches in ``FUSION_REDTEAM.md`` ``[A2]``.
    4. Optionally patches ``gpu_stage1a_module._sort_valid_rows_f64`` and
       ``gpu_stage1a_module._sort_valid_values_f64`` to use coalesced CUB radix sort (`HPC #2`).
    """
    dope = getattr(fused_net, "dope_model", None)
    if not isinstance(dope, nn.Module):
        raise ValueError("fused_net is missing required 'dope_model' submodule")

    sha_before = _snapshot_dope_state_sha256(fused_net)
    for param in dope.parameters():
        param.requires_grad_(False)
        param.grad = None
    dope.eval()

    # Lock train() on every nested DOPE module so fused_net.train(True) never enables DOPE Dropout
    def _locked_eval_train(self_mod: nn.Module, mode: bool = True) -> nn.Module:
        nn.Module.train(self_mod, False)
        return self_mod

    for submod in dope.modules():
        submod.training = False
        submod.train = _locked_eval_train.__get__(submod, type(submod))  # type: ignore[method-assign]

    if hasattr(fused_net, "_run_dope_mlc_v3_and_lm48"):
        orig_stage1a_fn = fused_net._run_dope_mlc_v3_and_lm48
        graph_runner = FusedStageCudaGraphRunner(
            orig_stage1a_fn,
            enable_cuda_graphs=enable_cuda_graphs,
        )
        fused_net._hpc_stage1a_graph_runner = graph_runner  # type: ignore[attr-defined]
        fused_net._run_dope_mlc_v3_and_lm48 = graph_runner.run_stage1a_neural  # type: ignore[method-assign]

    if force_stage2_fp32 and hasattr(fused_net, "forward"):
        orig_forward = fused_net.forward

        def _forward_fp32_stage2(*args: Any, **kwargs: Any) -> Any:
            kwargs["amp_dtype"] = torch.float32
            return orig_forward(*args, **kwargs)

        fused_net.forward = _forward_fp32_stage2  # type: ignore[method-assign]

    if gpu_stage1a_module is not None:
        gpu_stage1a_module._sort_valid_values_f64 = sort_valid_values_radix_f64
        gpu_stage1a_module._sort_valid_rows_f64 = sort_valid_rows_radix_f64

    sha_after = _snapshot_dope_state_sha256(fused_net)
    if sha_after != sha_before:
        raise RuntimeError(f"DOPE state SHA-256 mutated during HPC patch: {sha_before} != {sha_after}")

    trainable_by_child = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in fused_net.named_children()
    }
    if trainable_by_child.get("dope_model", 0) != 0:
        raise RuntimeError("DOPE parameters must have 0 trainable elements")

    return {
        "dope_state_sha256_before": sha_before,
        "dope_state_sha256_after": sha_after,
        "dope_freeze_verified": sha_before == sha_after and trainable_by_child.get("dope_model", 0) == 0,
        "trainable_param_counts_by_submodule": trainable_by_child,
        "cuda_graphs_enabled": bool(enable_cuda_graphs),
        "force_stage2_fp32": bool(force_stage2_fp32),
        "stage1a_radix_sort_patched": gpu_stage1a_module is not None,
    }
