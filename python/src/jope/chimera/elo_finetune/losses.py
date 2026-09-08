"""Loss terms: soft win mass (the differentiable Elo surrogate) + anchors.

Soft win mass: L_elo = −Σ_d w_d Σ_j σ((û_d − u_dj)/τ_d) / Σ_d w_d·|C_d|.
The converged 60-round replay rating is strictly monotone in total win mass,
so this is the first-order surrogate of the rating itself. τ_d is a MAD-based
per-dataset temperature (robust to broken-competitor outliers); w_d damps
tiny-query datasets (min(1, test_rows/8)).

Anchors: bar-distribution NLL (mean-reduced, tabby_cluster:1591 semantics)
keeps the bucket distribution a calibrated density; KL(p_θ ‖ p_base) against
the frozen model's cached logits is the trust region for full unlock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .eval_protocol import Tier1Output
from .pack import FablePack

TAU_MIN = 1e-4
TAU_MAX = 1.0
TINY_QRY_DAMP = 8.0


class EloLoss:
    def __init__(self, pack: FablePack, device: torch.device, tau0: float = 0.5):
        self.device = device
        self.tau0 = tau0
        te_off = pack.section("test_row_offsets")
        self._comp: dict[int, np.ndarray] = {}
        self._tau: dict[int, float] = {}
        self._weight: dict[int, float] = {}
        for i in range(len(pack)):
            _, utils = pack.competitor_utilities(i)
            # Broken competitors can report astronomically bad rmse; clip into
            # f32 range so they stay finite (they are certain soft wins with
            # zero gradient either way).
            self._comp[i] = np.clip(utils, -1e30, 1e30).astype(np.float32)
            med = np.median(utils)
            mad = float(np.median(np.abs(utils - med)))
            self._tau[i] = float(np.clip(tau0 * 1.4826 * mad, TAU_MIN, TAU_MAX))
            test_rows = float(te_off[i + 1] - te_off[i])
            self._weight[i] = min(1.0, test_rows / TINY_QRY_DAMP)

    def soft_win_mass(
        self, u_soft: torch.Tensor, dataset_indices: list[int]
    ) -> tuple[torch.Tensor, dict]:
        B = len(dataset_indices)
        c_max = max(self._comp[i].shape[0] for i in dataset_indices)
        comp = np.full((B, c_max), np.nan, dtype=np.float32)
        for b, i in enumerate(dataset_indices):
            comp[b, : self._comp[i].shape[0]] = self._comp[i]
        comp_t = torch.from_numpy(comp).to(self.device)
        mask = torch.isfinite(comp_t)
        comp_t = torch.nan_to_num(comp_t)
        tau = torch.tensor(
            [self._tau[i] for i in dataset_indices], device=self.device
        ).unsqueeze(1)
        weight = torch.tensor(
            [self._weight[i] for i in dataset_indices], device=self.device
        )

        z = (u_soft.unsqueeze(1) - comp_t) / tau
        soft_wins = torch.sigmoid(z) * mask
        per_dataset_wins = soft_wins.sum(dim=1)
        pair_counts = mask.sum(dim=1).float()
        loss = -(weight * per_dataset_wins).sum() / (weight * pair_counts).sum()

        with torch.no_grad():
            hard = (z > 0).float() + 0.5 * (z == 0).float()
            winrate = ((hard * mask).sum(dim=1) / pair_counts).mean()
        return loss, {"winrate_mean": float(winrate)}


def bar_nll_mean(logits: torch.Tensor, y_qry_z: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Mean-reduced bar-distribution NLL (tabby_cluster regression_nll_sum)."""
    widths = borders[1:] - borders[:-1]
    n_buckets = widths.numel()
    idx = (torch.bucketize(y_qry_z, borders) - 1).clamp(0, n_buckets - 1)
    logp = torch.log_softmax(logits, dim=-1)
    logdens = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1) - torch.log(widths[idx])
    return -logdens.mean()


def bar_entropy_mean(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean())


class KlAnchor:
    """KL(p_θ ‖ p_base) on seed-A query rows, from the fp16 logit cache."""

    def __init__(self, pack_dir: Path, pack: FablePack, seed_a: int, device: torch.device):
        import json

        meta = json.loads((pack_dir / "kl_base_meta_v1.json").read_text())
        if meta["seed_a"] != seed_a:
            raise ValueError(f"KL cache seed_a {meta['seed_a']} != run seed_a {seed_a}")
        self.n_buckets = meta["n_buckets"]
        self.offsets = np.load(pack_dir / "kl_base_offsets_v1.npy")
        self.flat = np.memmap(pack_dir / "kl_base_logits_v1.bin", dtype=np.float16, mode="r")
        self.device = device

    def has(self, i: int) -> bool:
        return self.offsets[i + 1] > self.offsets[i]

    def kl(self, logits: torch.Tensor, dataset_indices: list[int]) -> torch.Tensor:
        qry = logits.shape[0]
        base = np.empty((qry, len(dataset_indices), self.n_buckets), dtype=np.float16)
        for b, i in enumerate(dataset_indices):
            if not self.has(i):
                raise KeyError(f"dataset {i} missing from KL cache")
            arr = self.flat[self.offsets[i] : self.offsets[i + 1]]
            base[:, b, :] = arr.reshape(qry, self.n_buckets)
        base_t = torch.from_numpy(base).to(self.device).float()
        logp_cur = torch.log_softmax(logits.float(), dim=-1)
        logp_base = torch.log_softmax(base_t, dim=-1)
        return (logp_cur.exp() * (logp_cur - logp_base)).sum(dim=-1).mean()


def combined_loss(
    elo_loss: EloLoss,
    out: Tier1Output,
    dataset_indices: list[int],
    borders: torch.Tensor,
    lam_nll: float,
    lam_kl: float,
    kl_anchor: KlAnchor | None,
) -> tuple[torch.Tensor, dict]:
    l_elo, stats = elo_loss.soft_win_mass(out.u_soft, dataset_indices)
    l_nll = bar_nll_mean(out.logits, out.y_qry_z, borders)
    loss = l_elo + lam_nll * l_nll
    l_kl = None
    if lam_kl > 0.0 and kl_anchor is not None:
        l_kl = kl_anchor.kl(out.logits, dataset_indices)
        loss = loss + lam_kl * l_kl
    metrics = {
        "loss_elo": float(l_elo.detach()),
        "loss_nll": float(l_nll.detach()),
        "loss_kl": float(l_kl.detach()) if l_kl is not None else 0.0,
        "entropy_bar": bar_entropy_mean(out.logits.detach()),
        **stats,
    }
    return loss, metrics
