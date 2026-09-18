"""Per-batch REAL Elo measurement and the trust statistics.

- SplicedEloMeter: persistent utility table over one split; competitor-vs-
  competitor pair stats are constants, our aggregated pair stats update
  incrementally; the exact 60-round replay runs every batch (<5ms numpy).
  Must match fable-elo's compute_elo bit-for-bit in semantics (outcome ties
  at |Δ|<=1e-12, synchronous rounds, delta averaged by match count).
- Paired ΔElo: δ_t = mean_d(winrate(û_d) − winrate(u_base[d])) cancels
  dataset-difficulty noise exactly in expectation; converted to Elo units by
  the exchange rate g = 400/(ln10 · weighted mean E_j(1−E_j)).
- DebiasedEwma: Adam-style bias-corrected mean/variance with N_eff-adjusted
  standard error, at half-lives 50 (fast) and 400 (slow) batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pack import OUR_ALGORITHM, FablePack

TIE_EPS = 1e-12
ELO_ROUNDS = 60
ELO_K = 64.0
ELO_SCALE = 400.0
ELO_INIT = 1000.0


def outcomes(u_ours: float, u_comp: np.ndarray) -> np.ndarray:
    """Hard outcome of ours-vs-each-competitor: 1 win, 0 loss, 0.5 tie."""
    diff = u_ours - u_comp
    out = np.where(diff > 0.0, 1.0, 0.0)
    out[np.abs(diff) <= TIE_EPS] = 0.5
    return out


@dataclass
class EwmaState:
    half_life: float
    m: float = 0.0
    v: float = 0.0
    t: int = 0

    @property
    def beta(self) -> float:
        return 2.0 ** (-1.0 / self.half_life)

    def update(self, x: float) -> tuple[float, float]:
        """Returns (debiased mean, standard error)."""
        b = self.beta
        self.t += 1
        debias = 1.0 - b**self.t
        self.m = b * self.m + (1.0 - b) * x
        m_hat = self.m / debias
        self.v = b * self.v + (1.0 - b) * (x - m_hat) ** 2
        v_hat = self.v / debias
        n_eff = debias * (1.0 + b) / (1.0 - b)
        return m_hat, math.sqrt(max(v_hat, 0.0) / max(n_eff, 1.0))

    def state_dict(self) -> dict:
        return {"half_life": self.half_life, "m": self.m, "v": self.v, "t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.half_life = state["half_life"]
        self.m, self.v, self.t = state["m"], state["v"], state["t"]


class SplicedEloMeter:
    """Exact replay over one split with our utilities spliced in live.

    Our live utility replaces the frozen `tabpfn_v3_regressor` record on every
    split dataset; all other algorithms stay frozen. Pair counts never change
    (presence is fixed), only outcome sums do — so cc-pairs are constants and
    our ~271 aggregated pair stats update incrementally per batch.
    """

    def __init__(self, pack: FablePack, split: str, u_init: dict[int, float]):
        self.pack = pack
        self.split = split
        self.ours = pack.algo_id[OUR_ALGORITHM]
        self.dataset_indices = pack.split_indices(split).tolist()
        missing = [i for i in self.dataset_indices if i not in u_init]
        if missing:
            raise ValueError(f"u_init missing {len(missing)} datasets (e.g. {missing[:3]})")

        self.comp_ids: dict[int, np.ndarray] = {}
        self.comp_utils: dict[int, np.ndarray] = {}
        self.u_table: dict[int, float] = {}
        self.w_base: dict[int, float] = {}

        A = len(pack.algos)
        cc_out = np.zeros((A, A), dtype=np.float64)
        cc_cnt = np.zeros((A, A), dtype=np.int64)
        self.our_out = np.zeros(A, dtype=np.float64)
        self.our_cnt = np.zeros(A, dtype=np.int64)

        for i in self.dataset_indices:
            ids, utils = pack.competitor_utilities(i)
            self.comp_ids[i] = ids.astype(np.int64)
            self.comp_utils[i] = utils.astype(np.float64)
            self.u_table[i] = float(u_init[i])
            self.w_base[i] = float(outcomes(self.u_table[i], self.comp_utils[i]).mean())

            diff = utils[:, None] - utils[None, :]
            out = np.where(diff > 0.0, 1.0, 0.0)
            out[np.abs(diff) <= TIE_EPS] = 0.5
            idx = ids.astype(np.int64)
            upper = np.triu_indices(len(idx), k=1)
            rows, cols = idx[upper[0]], idx[upper[1]]
            swap = rows > cols
            lo = np.where(swap, cols, rows)
            hi = np.where(swap, rows, cols)
            val = np.where(swap, 1.0 - out[upper], out[upper])
            np.add.at(cc_out, (lo, hi), val)
            np.add.at(cc_cnt, (lo, hi), 1)

            self.our_out[idx] += outcomes(self.u_table[i], self.comp_utils[i])
            self.our_cnt[idx] += 1

        lo, hi = np.nonzero(cc_cnt)
        self.cc_left = lo
        self.cc_right = hi
        self.cc_out = cc_out[lo, hi]
        self.cc_cnt = cc_cnt[lo, hi].astype(np.float64)

        self.active = np.nonzero(self.our_cnt)[0]
        match_count = np.zeros(A, dtype=np.float64)
        np.add.at(match_count, self.cc_left, self.cc_cnt)
        np.add.at(match_count, self.cc_right, self.cc_cnt)
        match_count += self.our_cnt
        match_count[self.ours] = self.our_cnt.sum()
        self.match_count = match_count

        self._ratings = self.replay()

    def update(self, batch: dict[int, float]) -> dict:
        """Splice new utilities for the batch's datasets, replay, and return
        the level + the paired innovation. Values must be float64."""
        delta_sum, n = 0.0, 0
        for i, u_new in batch.items():
            comp = self.comp_utils[i]
            ids = self.comp_ids[i]
            o_new = outcomes(float(u_new), comp)
            o_old = outcomes(self.u_table[i], comp)
            np.add.at(self.our_out, ids, o_new - o_old)
            self.u_table[i] = float(u_new)
            delta_sum += o_new.mean() - self.w_base[i]
            n += 1
        self._ratings = self.replay()
        delta = delta_sum / max(n, 1)
        return {
            "elo_spliced": self.our_rating,
            "delta_raw": delta,
            "delta_elo_raw": self.exchange_rate() * delta,
        }

    def replay(self) -> np.ndarray:
        ratings = np.full(len(self.pack.algos), ELO_INIT, dtype=np.float64)
        our_active_out = self.our_out[self.active]
        our_active_cnt = self.our_cnt[self.active].astype(np.float64)
        safe_count = np.maximum(self.match_count, 1.0)
        for _ in range(ELO_ROUNDS):
            deltas = np.zeros_like(ratings)
            e_cc = 1.0 / (
                1.0 + 10.0 ** ((ratings[self.cc_right] - ratings[self.cc_left]) / ELO_SCALE)
            )
            np.add.at(deltas, self.cc_left, self.cc_out - self.cc_cnt * e_cc)
            np.add.at(
                deltas,
                self.cc_right,
                (self.cc_cnt - self.cc_out) - self.cc_cnt * (1.0 - e_cc),
            )
            e_ours = 1.0 / (
                1.0 + 10.0 ** ((ratings[self.active] - ratings[self.ours]) / ELO_SCALE)
            )
            np.add.at(
                deltas,
                self.active,
                (our_active_cnt - our_active_out) - our_active_cnt * (1.0 - e_ours),
            )
            deltas[self.ours] += float(
                np.sum(our_active_out - our_active_cnt * e_ours)
            )
            ratings += ELO_K * deltas / safe_count
        return ratings

    @property
    def our_rating(self) -> float:
        return float(self._ratings[self.ours])

    def ratings_by_name(self) -> dict[str, float]:
        active = set(self.active.tolist()) | {self.ours}
        return {
            self.pack.algos[a]: float(self._ratings[a])
            for a in range(len(self.pack.algos))
            if a in active or self.match_count[a] > 0
        }

    def exchange_rate(self) -> float:
        """dElo/d(mean winrate) at the current ratings (loss-design §0.4)."""
        e = 1.0 / (
            1.0 + 10.0 ** ((self._ratings[self.active] - self.our_rating) / ELO_SCALE)
        )
        weights = self.our_cnt[self.active].astype(np.float64)
        mean_ee = float(np.sum(weights * e * (1.0 - e)) / np.sum(weights))
        return ELO_SCALE / (math.log(10.0) * max(mean_ee, 1e-6))

    def state_dict(self) -> dict:
        return {
            "split": self.split,
            "u_table": {str(i): u for i, u in self.u_table.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        assert state["split"] == self.split
        updates = {int(i): u for i, u in state["u_table"].items()}
        self.update(updates)
