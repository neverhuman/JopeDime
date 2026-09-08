"""Sparse-perturbation machinery for swarm evolution.

- Block table: every perturbable tensor partitioned into row-aligned slices
  of ~16,384 params (32 output-rows for 512-wide tensors); tensors smaller
  than one slice are a single block. `regression_borders` and
  `column_aggregator.rope.freqs` are excluded (decode/positional geometry).
- Candidates are seeds: (gen, cand_id, mask_seed, sigma_seed, value_seed)
  regenerate the identical delta on every node (values depend only on the
  champion weights, which are checksum-verified identical).
- Apply/restore is bit-exact: save the masked originals (~7MB), scatter the
  delta in-place, evaluate, write the buffer back.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

EXCLUDE = ("regression_borders", "column_aggregator.rope.freqs")
TARGET_BLOCK = 16_384
SPARSITY = 0.03

# per-module-group sigma multipliers (adapted by the 1/5th rule)
MODULE_GROUPS = ("head", "icl_16_23", "icl_8_15", "icl_0_7", "feature")


def module_group(tensor_name: str) -> str:
    root = tensor_name.split(".")[0]
    if root in ("output_projection", "output_norm", "icl_y_encoder", "col_y_encoder"):
        return "head"
    if root == "icl_blocks":
        layer = int(tensor_name.split(".")[1])
        return "icl_16_23" if layer >= 16 else ("icl_8_15" if layer >= 8 else "icl_0_7")
    return "feature"  # x_embed, feature_distribution_embedder, column_aggregator


@dataclass
class Block:
    id: int
    tensor: str
    start: int  # flat offset within tensor
    end: int
    group: str


class BlockTable:
    """Deterministic partition of the perturbable parameter space."""

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self.blocks: list[Block] = []
        self.tensor_names: list[str] = []
        bid = 0
        for name in sorted(state_dict.keys()):
            if any(name.startswith(e) or name == e for e in EXCLUDE):
                continue
            t = state_dict[name]
            numel = t.numel()
            if numel == 0:
                continue
            self.tensor_names.append(name)
            group = module_group(name)
            if t.ndim >= 2 and numel > TARGET_BLOCK:
                row_numel = numel // t.shape[0]
                rows_per_block = max(1, TARGET_BLOCK // max(row_numel, 1))
                step = rows_per_block * row_numel
                for start in range(0, numel, step):
                    self.blocks.append(
                        Block(bid, name, start, min(start + step, numel), group))
                    bid += 1
            else:
                self.blocks.append(Block(bid, name, 0, numel, group))
                bid += 1
        self.n_blocks = bid
        self.numels = np.array([b.end - b.start for b in self.blocks], dtype=np.int64)
        self.total_params = int(self.numels.sum())
        self.group_of_block = np.array(
            [MODULE_GROUPS.index(b.group) for b in self.blocks], dtype=np.int32)

    def layout_json(self) -> dict:
        """Tensor → block-range table for the TUI's evo_layout.json."""
        tensors = []
        by_tensor: dict[str, list[Block]] = {}
        for b in self.blocks:
            by_tensor.setdefault(b.tensor, []).append(b)
        offset = 0
        for name in sorted(by_tensor):
            blocks = by_tensor[name]
            numel = sum(b.end - b.start for b in blocks)
            tensors.append({
                "name": name,
                "offset": offset,
                "numel": numel,
                "block0": blocks[0].id,
                "nblocks": len(blocks),
            })
            offset += numel
        return {"n_params": self.total_params, "n_blocks": self.n_blocks,
                "tensors": tensors}


@dataclass
class CandidateSpec:
    gen: int
    cand_id: int
    mask_seed: int
    sigma_seed: int
    value_seed: int
    sign: int = +1  # antithetic pairs: ±1 on the same seeds

    def key(self) -> list[int]:
        return [self.gen, self.cand_id if self.sign > 0 else -self.cand_id]


def sample_mask(table: BlockTable, p: np.ndarray, mask_seed: int,
                target_params: int | None = None) -> np.ndarray:
    """Blocks drawn without replacement ∝ p until Σ params ≥ 3% of the net."""
    target = target_params or int(round(SPARSITY * table.total_params))
    rng = np.random.default_rng([mask_seed, 11])
    # full weighted permutation: robust even when p concentrates on tiny blocks
    order = rng.choice(table.n_blocks, size=table.n_blocks, replace=False, p=p)
    chosen, total = [], 0
    for b in order:
        chosen.append(int(b))
        total += int(table.numels[b])
        if total >= target:
            break
    return np.array(sorted(chosen), dtype=np.int64)


class DeltaEngine:
    """Regenerates, applies, and reverts sparse deltas on a live model."""

    def __init__(self, model: torch.nn.Module, table: BlockTable):
        self.table = table
        self.params = dict(model.named_parameters())
        self.buffers = dict(model.named_buffers())
        self._flat: dict[str, torch.Tensor] = {}
        for name in table.tensor_names:
            t = self.params.get(name, self.buffers.get(name))
            assert t is not None, name
            self._flat[name] = t.data.view(-1)
        self.refresh_stats()

    def refresh_stats(self) -> None:
        """Per-tensor RMS floors — refresh after every champion promotion."""
        self.rms = {
            name: float(flat.float().pow(2).mean().sqrt().clamp_min(1e-8))
            for name, flat in self._flat.items()
        }

    def build_delta(self, spec: CandidateSpec, mask_blocks: np.ndarray,
                    sigma_groups: np.ndarray) -> list[tuple[Block, torch.Tensor]]:
        """δ_i = σ_c · g_m · max(|w_i|, 0.3·RMS(tensor)) · z_i, deterministic
        in (spec, champion weights)."""
        srng = np.random.default_rng([spec.sigma_seed, 23])
        sigma_c = float(np.clip(0.02 * np.exp(0.4 * srng.standard_normal()),
                                0.003, 0.03))
        out = []
        for k, bid in enumerate(mask_blocks.tolist()):
            block = self.table.blocks[bid]
            w = self._flat[block.tensor][block.start : block.end]
            vrng = np.random.default_rng([spec.value_seed, bid, 37])
            z = torch.from_numpy(
                vrng.standard_normal(block.end - block.start).astype(np.float32)
            ).to(w.device)
            g_m = float(sigma_groups[MODULE_GROUPS.index(block.group)])
            scale = torch.clamp(w.abs(), min=0.3 * self.rms[block.tensor])
            out.append((block, spec.sign * sigma_c * g_m * scale * z))
        return out

    def apply(self, delta: list[tuple[Block, torch.Tensor]]) -> list[torch.Tensor]:
        saved = []
        with torch.no_grad():
            for block, d in delta:
                seg = self._flat[block.tensor][block.start : block.end]
                saved.append(seg.clone())
                seg.add_(d)
        return saved

    def restore(self, delta: list[tuple[Block, torch.Tensor]],
                saved: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for (block, _), orig in zip(delta, saved):
                self._flat[block.tensor][block.start : block.end].copy_(orig)

    def apply_sparse_file(self, npz_path: str, sign: float = 1.0) -> None:
        """PERMANENTLY apply a stored sparse delta (champion promotions —
        never reverted; indices are unique per tensor so this is
        deterministic and order-independent across nodes)."""
        data = np.load(npz_path)
        with torch.no_grad():
            for name in data["tensors"]:
                flat = self._flat[str(name)]
                idx = torch.from_numpy(data[f"{name}::idx"]).to(flat.device)
                val = torch.from_numpy(data[f"{name}::val"]).to(flat.device)
                flat.index_add_(0, idx, sign * val)

    def apply_sparse_file_saved(self, npz_path: str) -> list:
        """Apply a file delta REVERSIBLY (grad-arm/recomb/elite candidates):
        saves the touched originals so restore is bit-exact."""
        data = np.load(npz_path)
        saved = []
        with torch.no_grad():
            for name in data["tensors"]:
                flat = self._flat[str(name)]
                idx = torch.from_numpy(data[f"{name}::idx"]).to(flat.device)
                val = torch.from_numpy(data[f"{name}::val"]).to(flat.device)
                saved.append((str(name), idx, flat[idx].clone()))
                flat.index_add_(0, idx, val)
        return saved

    def restore_sparse_saved(self, saved: list) -> None:
        with torch.no_grad():
            for name, idx, orig in saved:
                self._flat[name][idx] = orig

    def checksum(self, sample_tensors: int = 4) -> str:
        h = hashlib.sha256()
        names = self.table.tensor_names[:: max(1, len(self.table.tensor_names)
                                               // sample_tensors)][:sample_tensors]
        for name in names:
            h.update(self._flat[name][:1024].float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def full_checksum(self) -> str:
        h = hashlib.sha256()
        for name in self.table.tensor_names:
            h.update(self._flat[name].float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]


def save_sparse_delta(path: str, deltas: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """deltas: tensor_name -> (flat idx int64, values f32)."""
    payload: dict = {"tensors": np.array(sorted(deltas.keys()))}
    for name, (idx, val) in deltas.items():
        payload[f"{name}::idx"] = idx.astype(np.int64)
        payload[f"{name}::val"] = val.astype(np.float32)
    np.savez_compressed(path, **payload)


def delta_to_sparse(delta: list[tuple[Block, torch.Tensor]]
                    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert an in-memory block delta to the sparse-file representation."""
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for block, d in delta:
        idx = np.arange(block.start, block.end, dtype=np.int64)
        grouped.setdefault(block.tensor, []).append((idx, d.cpu().numpy()))
    return {
        name: (np.concatenate([i for i, _ in parts]),
               np.concatenate([v for _, v in parts]))
        for name, parts in grouped.items()
    }
