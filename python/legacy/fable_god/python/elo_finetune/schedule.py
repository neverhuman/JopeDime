"""Staged unlock, layer-wise LR decay, λ schedules, and the promotion gate.

Stages (steps are global optimizer steps):
  stage0 [1, s0]      head only (output_projection, y-encoders, output_norm), head LR 1e-5
  stage1 (s0, s1]     + top 8 ICL blocks, head LR 8e-6, LLRD 0.9/block
  stage2 (s1, total]  full stack (trunk ≈6.4e-7, feature stack ≈4.7e-7)

Safety migrates from tiny LR to anchors + gating: the deployed weights came
from a deliberately non-destructive 1e-9 lineage, so every stage re-warms LR
over 100 steps and the feature stack gets ~15x less LR than the head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

HEAD_MODULES = ("output_projection", "output_norm", "icl_y_encoder", "col_y_encoder")
FEATURE_MODULES = ("column_aggregator", "feature_distribution_embedder", "x_embed")
N_ICL = 24
LLRD = 0.9
REWARM_STEPS = 100
MIN_LR = 1e-7


def group_of(param_name: str) -> str:
    root = param_name.split(".")[0]
    if root in HEAD_MODULES:
        return "head"
    if root == "icl_blocks":
        return f"icl_{int(param_name.split('.')[1])}"
    if root in FEATURE_MODULES:
        return "feature"
    raise ValueError(f"unmapped parameter {param_name}")


def group_multiplier(group: str) -> float:
    if group == "head":
        return 1.0
    if group.startswith("icl_"):
        block = int(group.split("_")[1])
        return LLRD ** (N_ICL - block)
    if group == "feature":
        return LLRD ** (N_ICL + 3)
    raise ValueError(group)


@dataclass
class StagePlan:
    stage0_steps: int = 300
    stage1_steps: int = 1200
    total_steps: int = 20000
    head_lr_stage0: float = 1e-5
    head_lr: float = 8e-6
    lam_nll_hi: float = 0.1
    lam_nll_lo: float = 0.02
    lam_kl_hi: float = 0.5
    lam_kl_lo: float = 0.05

    def stage(self, step: int) -> str:
        if step <= self.stage0_steps:
            return "stage0"
        if step <= self.stage1_steps:
            return "stage1"
        return "stage2"

    def stage_start(self, stage: str) -> int:
        return {"stage0": 1, "stage1": self.stage0_steps + 1,
                "stage2": self.stage1_steps + 1}[stage]

    def trainable_groups(self, stage: str) -> set[str]:
        groups = {"head"}
        if stage in ("stage1", "stage2"):
            groups |= {f"icl_{b}" for b in range(N_ICL - 8, N_ICL)}
        if stage == "stage2":
            groups |= {f"icl_{b}" for b in range(N_ICL)} | {"feature"}
        return groups

    def head_lr_at(self, step: int) -> float:
        stage = self.stage(step)
        base = self.head_lr_stage0 if stage == "stage0" else self.head_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, self.total_steps) / self.total_steps))
        lr = MIN_LR + (base - MIN_LR) * cosine
        into_stage = step - self.stage_start(stage)
        if into_stage < REWARM_STEPS:
            lr *= 0.1 + 0.9 * (into_stage / REWARM_STEPS)
        return lr

    def lambdas(self, step: int) -> tuple[float, float]:
        stage = self.stage(step)
        if stage == "stage0":
            return self.lam_nll_hi, 0.0
        s2 = self.stage_start("stage2")
        frac = min(1.0, max(0.0, (step - s2) / max(1, self.total_steps - s2)))
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        lam_nll = self.lam_nll_lo + (self.lam_nll_hi - self.lam_nll_lo) * cos
        lam_kl = self.lam_kl_lo + (self.lam_kl_hi - self.lam_kl_lo) * cos
        return lam_nll, lam_kl


@dataclass
class PromotionGate:
    """Eval-level gate on the val split's exact tier-2 Elo.

    Promote iff the debiased eval-EWMA beats the best promoted value by
    promote_margin AND its lower confidence edge clears the best AND the
    train-split spliced Elo hasn't collapsed. Rollback after
    rollback_patience consecutive evals below best − rollback_margin.
    """

    half_life: float = 3.0
    promote_margin: float = 3.0
    rollback_margin: float = 10.0
    rollback_patience: int = 3
    train_drop_limit: float = 5.0

    best: float = float("-inf")
    m: float = 0.0
    v: float = 0.0
    t: int = 0
    below_count: int = 0
    baseline_train_elo: float = field(default=float("nan"))

    def observe(self, val_elo: float, train_spliced_elo: float) -> dict:
        b = 2.0 ** (-1.0 / self.half_life)
        self.t += 1
        debias = 1.0 - b**self.t
        self.m = b * self.m + (1.0 - b) * val_elo
        m_hat = self.m / debias
        self.v = b * self.v + (1.0 - b) * (val_elo - m_hat) ** 2
        v_hat = self.v / debias
        n_eff = debias * (1.0 + b) / (1.0 - b)
        se = math.sqrt(max(v_hat, 0.0) / max(n_eff, 1.0))

        if self.best == float("-inf"):
            self.best = m_hat
            self.baseline_train_elo = train_spliced_elo
            return {"ewma": m_hat, "se": se, "promote": False, "rollback": False,
                    "baseline_set": True}

        train_ok = (
            math.isnan(self.baseline_train_elo)
            or train_spliced_elo >= self.baseline_train_elo - self.train_drop_limit
        )
        promote = (
            m_hat >= self.best + self.promote_margin
            and (m_hat - 1.96 * se) > self.best
            and train_ok
        )
        if promote:
            self.best = m_hat
            self.below_count = 0
        elif m_hat < self.best - self.rollback_margin:
            self.below_count += 1
        else:
            self.below_count = 0
        rollback = self.below_count >= self.rollback_patience
        return {"ewma": m_hat, "se": se, "promote": promote, "rollback": rollback,
                "baseline_set": False}

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("best", "m", "v", "t", "below_count", "baseline_train_elo")}

    def load_state_dict(self, state: dict) -> None:
        for k, value in state.items():
            setattr(self, k, value)
