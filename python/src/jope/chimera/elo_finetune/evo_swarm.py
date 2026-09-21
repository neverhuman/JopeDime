"""The location swarm: WHERE in the network evolution concentrates.

Probability p over blocks with velocity (momentum) — PBIL/EDA with rank-based
occupancy credit (scale-free, so big-magnitude modules can't dominate by
sensitivity alone), a probability cap, a uniform exploration floor, and an
entropy guard against collapse. Also owns per-module-group sigma multipliers
(1/5th rule) and the per-block signed credit EWMA the TUI renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .evo_blocks import MODULE_GROUPS


@dataclass
class SwarmConfig:
    eta: float = 0.12  # PBIL learning rate
    beta: float = 0.7  # velocity momentum
    gamma: float = 0.06  # velocity contribution
    rho: float = 0.10  # uniform exploration floor
    rho_max: float = 0.30
    p_cap_mult: float = 50.0  # max p_b = cap_mult / B
    top_frac: float = 0.25
    credit_ewma: float = 0.9  # per-block signed credit smoothing (TUI)
    fifth_lo: float = 0.15  # 1/5th rule bounds on success rate
    fifth_hi: float = 0.25
    fifth_step: float = 1.15
    g_clip: tuple = (0.5, 1.5)


class Swarm:
    def __init__(self, n_blocks: int, cfg: SwarmConfig | None = None):
        self.cfg = cfg or SwarmConfig()
        self.n = n_blocks
        self.p = np.full(n_blocks, 1.0 / n_blocks)
        self.v = np.zeros(n_blocks)
        self.credit = np.zeros(n_blocks)  # signed EWMA (TUI diagnostic)
        self.credit_count = np.zeros(n_blocks)
        self.sigma_groups = np.ones(len(MODULE_GROUPS))
        self.success_window: list[int] = []
        self.rho = self.cfg.rho

    def entropy_bits(self) -> float:
        nz = self.p[self.p > 0]
        return float(-(nz * np.log2(nz)).sum())

    def update(self, units: list[dict]) -> None:
        """units: one per distinct mask this generation, each
        {mask_blocks: np.ndarray, delta: float (paired mean margin Δ)},
        ranked by delta descending by the caller or here."""
        cfg = self.cfg
        units = sorted(units, key=lambda u: -u["delta"])
        n_top = max(1, math.ceil(len(units) * cfg.top_frac))
        freq = np.zeros(self.n)
        for rank, unit in enumerate(units[:n_top], start=1):
            w = max(math.log(len(units) / 2 + 1) - math.log(rank), 0.0)
            freq[unit["mask_blocks"]] += w
        if freq.sum() > 0:
            freq /= freq.sum()
        else:
            freq = np.full(self.n, 1.0 / self.n)

        self.v = cfg.beta * self.v + (1 - cfg.beta) * (freq - self.p)
        p = (1 - cfg.eta) * self.p + cfg.eta * freq + cfg.gamma * self.v
        p = np.clip(p, 0.0, cfg.p_cap_mult / self.n)
        p = (1 - self.rho) * p / max(p.sum(), 1e-12) + self.rho / self.n
        self.p = p / p.sum()

        if self.entropy_bits() < 0.5 * math.log2(self.n):
            self.rho = min(cfg.rho_max, 2 * self.rho)
        else:
            self.rho = max(cfg.rho, 0.95 * self.rho)

        # signed per-block credit EWMA (rendering only, never sampling)
        for unit in units:
            blocks = unit["mask_blocks"]
            self.credit[blocks] = (
                cfg.credit_ewma * self.credit[blocks] + (1 - cfg.credit_ewma) * unit["delta"]
            )
            self.credit_count[blocks] += 1

    def fifth_rule(self, successes: int, attempts: int) -> None:
        """Per-generation success bookkeeping; adapts all groups uniformly in
        v1 (per-group attribution needs more data than 20 candidates give)."""
        self.success_window.append((successes, attempts))
        self.success_window = self.success_window[-10:]
        s = sum(a for a, _ in self.success_window)
        t = sum(b for _, b in self.success_window)
        if t < 20:
            return
        rate = s / t
        if rate > self.cfg.fifth_hi:
            self.sigma_groups = np.clip(self.sigma_groups * self.cfg.fifth_step, *self.cfg.g_clip)
        elif rate < self.cfg.fifth_lo:
            self.sigma_groups = np.clip(self.sigma_groups / self.cfg.fifth_step, *self.cfg.g_clip)

    def success_rate(self) -> float:
        t = sum(b for _, b in self.success_window)
        return (sum(a for a, _ in self.success_window) / t) if t else 0.0

    def save(self, path: str) -> None:
        np.savez(
            path,
            p=self.p,
            v=self.v,
            credit=self.credit,
            credit_count=self.credit_count,
            sigma_groups=self.sigma_groups,
            rho=self.rho,
            success_window=np.array(self.success_window or [(0, 0)]),
        )

    @classmethod
    def load(cls, path: str, n_blocks: int) -> Swarm:
        s = cls(n_blocks)
        data = np.load(path)
        s.p, s.v = data["p"], data["v"]
        s.credit, s.credit_count = data["credit"], data["credit_count"]
        s.sigma_groups = data["sigma_groups"]
        s.rho = float(data["rho"])
        s.success_window = [tuple(x) for x in data["success_window"].tolist()]
        return s
