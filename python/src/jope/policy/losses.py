"""policy_v3 losses: legal-set label-smoothed sequence NLL, RWR weights, and the
Kendall–Gal combination with the inactive-log-var anchor fix.

The reward ranking losses (Bradley–Terry, ListNet) are reused from jope.model — they
are pure functions of (predictions, labels, dataset_ids).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import nn

# Loss components combined by Kendall–Gal weighting. Single source of truth shared by
# the model, the trainer, and metric aggregation (fixes the v2 dual-list hazard).
# v3.1 adds "ablation_bt": the genome-only control head's BT loss, trained on a DETACHED
# genome embedding (cannot shape the trunk) — its validation pairwise-win is the
# anti-collapse yardstick the conditional heads must beat.
POLICY_V3_LOSS_COMPONENTS = [
    "seq_nll",
    "reward_bt",
    "reward_listwise",
    "runtime_huber",
    "stability_bce",
    "ablation_bt",
    "routing_ce",
]

# Reported-but-never-optimized quantities. These ride along in the components dict so the
# trainer can aggregate them exactly like a loss, but kg_combine SKIPS them entirely (no
# term in the sum, no log-var) and loss_log_vars is NOT keyed on them — adding a name here
# therefore cannot change the checkpoint tensor count.
#
# Why they exist (measured on run c-r000024, 24 epochs):
#   * reward_listwise is ListNet cross-entropy, bounded below by the entropy of its own
#     softmax target. Train reward_listwise fell 2.4850 -> 2.2600 while the target entropy
#     it cannot go below sat at ~2.35, i.e. ~57% of the headline train loss was an
#     irreducible constant and the "improvement" was partly the group-size mix moving.
#     reward_listwise_entropy lets the trainer report CE - H, which is 0 at a perfect fit.
#   * stability_bce is a pos_weighted BCE whose best CONSTANT predictor is not ln 2. On the
#     live validation split the regression positive rate is 0.1373, so at pos_weight 6 the
#     best constant scores 1.1959 — and measured val stability_bce was 1.3079 at epoch 1
#     rising to 1.6479 at epoch 24, i.e. worse than a constant from the first epoch.
#     stability_bce_baseline makes that gap a directly reportable number.
POLICY_V3_DIAGNOSTIC_COMPONENTS = [
    "reward_listwise_entropy",
    "stability_bce_baseline",
]

# Curriculum profiles: component -> scale. Zero-scale terms are excluded from the sum
# but their log-vars are anchored back to 0 (see kg_combine).
# The reward branch trains in BOTH phases (a frozen-random phase-B start collapsed the
# reward to a constant — exact ln(2) BT plateau); phase A/B differ in RWR weighting and
# the z-encoder learning rate, not in which branches learn.
# reward_listwise is BACK ON in v3.1: targets are now per-dataset RANK PERCENTILES
# (uniform [0,1] spread by construction), not tiny raw deltas — sharpened at
# LISTWISE_TARGET_TAU they separate decisively, killing the old near-uniform collapse.
#
# routing_ce is DEMOTED 0.5 -> ROUTING_CE_WEIGHT (0.05). Measured on run c-r000024:
# train routing_ce 1.8562 -> 0.0058 while val routing_ce bottomed at 1.5384 (epoch 11) and
# then ROSE monotonically to 1.7608 by epoch 24 — a 304x train/val ratio on a 14-way
# classifier over ~2.5k datasets. That is dataset-identity memorisation, not routing
# structure, and until v3.4 it was injected into the single encoder BOTH objectives shared.
# It keeps a small non-zero weight rather than 0 because the original probe (raw z carries
# 0.41 routing accuracy, the unsupervised encoder output only 0.28) says the signal is real;
# 0.05 keeps the pressure without letting memorisation dominate the shared surface. In v3.4
# the routing heads read the REWARD context encoder, so this term can no longer reshape the
# sequence path at all.
ROUTING_CE_WEIGHT = 0.05
PHASE_A_PROFILE = {
    "seq_nll": 1.0,
    "runtime_huber": 0.2,
    "reward_bt": 1.0,
    "reward_listwise": 0.5,
    "stability_bce": 0.5,
    "ablation_bt": 1.0,
    "routing_ce": ROUTING_CE_WEIGHT,
}
PHASE_B_PROFILE = {
    "seq_nll": 1.0,
    "runtime_huber": 0.2,
    "reward_bt": 1.0,
    "reward_listwise": 0.5,
    "stability_bce": 0.5,
    "ablation_bt": 1.0,
    "routing_ce": ROUTING_CE_WEIGHT,
}
# Components excluded from Kendall-Gal reweighting (fixed scale from the profile).
# K&G's +0.5*s bonus actively SUPPRESSES slow-learning terms: reward_bt sits near its
# ln(2) plateau early, so K&G down-weights it, the gradient weakens further, and weight
# decay drives the reward to a constant. The ranking objective is the point of this
# model — it gets a fixed weight, not a learned excuse to vanish.
KG_EXCLUDED = (
    "reward_bt",
    "reward_listwise",
    "ablation_bt",
    "routing_ce",
    # Stability is a calibrated class-imbalanced probability head. Letting K&G
    # attenuate its BCE makes Brier/log-loss incomparable across checkpoints.
    "stability_bce",
)
# ListNet target sharpening: softmax(centered_delta / TAU). Within-dataset deltas are
# tiny (median ~0.01), so an unsharpened target is near-uniform and its optimum is a
# CONSTANT reward — the collapse driver. TAU rescales deltas into a separating range.
LISTWISE_TARGET_TAU = 0.25


def masked_sequence_nll(
    logits: torch.Tensor,  # (B, T-1, V) — predictions for positions 1..T-1
    targets: torch.Tensor,  # (B, T-1) token ids
    legal: torch.Tensor,  # (B, T-1, V) bool
    pad_mask: torch.Tensor,  # (B, T-1) bool, True = real target
    *,
    label_smoothing: float = 0.05,
    row_weights: torch.Tensor | None = None,  # (B,)
) -> torch.Tensor:
    """Per-token CE renormalized over the legal set, label-smoothed within it.

    Illegal logits are masked to -inf BEFORE log-softmax, so the model never spends
    capacity suppressing illegal tokens and forced tokens contribute exactly 0 NLL.
    """
    masked_logits = logits.masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(masked_logits, dim=-1)
    # NLL of the target token
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    if label_smoothing > 0:
        legal_count = legal.sum(dim=-1).clamp_min(1)
        smooth = -(logp.masked_fill(~legal, 0.0).sum(dim=-1)) / legal_count
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    nll = torch.where(pad_mask, nll, torch.zeros_like(nll))
    per_row = nll.sum(dim=-1) / pad_mask.sum(dim=-1).clamp_min(1)
    if row_weights is not None:
        weights = row_weights.clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-8)
        return (per_row * weights).sum() / denom
    return per_row.mean()


def kg_combine(
    components: Mapping[str, torch.Tensor],
    log_vars: nn.ParameterDict,
    profile: Mapping[str, float] | None = None,
    *,
    anchor: float = 1e-3,
) -> torch.Tensor:
    """Kendall–Gal task-uncertainty combination with the inactive-log-var anchor.

    L = sum_{scale>0} [exp(-s) * scale * L_i + 0.5*s]  +  sum_{scale==0} anchor * s^2

    The quadratic anchor pulls inactive log-vars back toward 0 so a term re-enters the
    objective at weight ~1 (never the +0.5*s constant-gradient freefall, never the v2
    silent-drop freeze). Clamp log-vars to [-3, 3] after each optimizer step.
    """
    terms: list[torch.Tensor] = []
    profile = profile or {}
    active = set(components)
    for name, value in components.items():
        # Diagnostics ride in the components dict for reporting only. They carry no
        # log-var, contribute no term, and must never reach the backward pass.
        if name in POLICY_V3_DIAGNOSTIC_COMPONENTS:
            continue
        scale = float(profile.get(name, 1.0))
        log_var = log_vars.get(name)
        if name in KG_EXCLUDED:
            if scale > 0:
                terms.append(value * scale)
            if log_var is not None:
                terms.append(anchor * log_var**2)
            continue
        if log_var is None:
            if scale > 0:
                terms.append(value * scale)
            continue
        if scale > 0:
            terms.append(torch.exp(-log_var) * (scale * value) + 0.5 * log_var)
        else:
            terms.append(anchor * log_var**2)
    # anchor log-vars whose component is absent from this batch entirely
    for name, log_var in log_vars.items():
        if name not in active:
            terms.append(anchor * log_var**2)
    if not terms:
        raise ValueError("no loss terms")
    return torch.stack([t if t.dim() == 0 else t.sum() for t in terms]).sum()


def clamp_log_vars(log_vars: nn.ParameterDict, lo: float = -3.0, hi: float = 3.0) -> None:
    with torch.no_grad():
        for param in log_vars.values():
            param.clamp_(lo, hi)


def bootstrap_mask(dataset_id: str, engine: str, canonical: str, member: int, *, keep_pct: int = 80) -> bool:
    """Deterministic bootstrap membership — stable across epochs, ranks, and sessions."""
    digest = hashlib.sha256(f"{dataset_id}|{engine}|{canonical}|{member}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < keep_pct


def self_play_bt_loss(
    reward: torch.Tensor,  # (N,) conditional reward scores
    value: torch.Tensor,  # (N,) true scores (robust); ordering source
    dataset_ids: list[str],
    *,
    rank_pct: torch.Tensor | None = None,  # (N,) in [0,1]; top-heavy boost source
    is_self: torch.Tensor | None = None,  # (N,) bool; former-generation picks
    self_margin: float = 0.5,
    top_boost: float = 1.0,  # extra weight when a pair touches the dataset's top decile
    self_boost: float = 2.0,  # extra weight when a pair touches a former-self pick
    max_pairs: int = 4096,
) -> torch.Tensor | None:
    """Margin-shifted, top-heavy, self-aware Bradley–Terry over within-dataset pairs.

    Within a dataset z is constant, so pair differences cancel any per-dataset constant —
    the loss is centered by construction (safe with the conditional reward). Three
    self-play mechanics fold into ONE pair loss:
      • pair weight ∝ true score gap (decisive orderings matter more)
      • ×(1+top_boost) when either member is in the dataset's top decile (top-of-list is
        what deployment uses)
      • ×self_boost AND a logistic MARGIN when the LOSER is a former-self pick: the new
        model must beat its ancestors by ``self_margin``, not merely tie them —
        loss = softplus(r_loser − r_winner + m·[loser is former self]).
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    vals = value.detach().cpu().tolist()
    for idx, d in enumerate(dataset_ids):
        grouped[d].append(idx)
    pct = rank_pct.detach().cpu().tolist() if rank_pct is not None else None
    selfs = is_self.detach().cpu().tolist() if is_self is not None else None
    winners: list[int] = []
    losers: list[int] = []
    weights: list[float] = []
    margins: list[float] = []
    for indices in grouped.values():
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                va, vb = vals[a], vals[b]
                if not (math.isfinite(va) and math.isfinite(vb)) or va == vb:
                    continue
                w, l = (a, b) if va > vb else (b, a)
                weight = abs(va - vb)
                if pct is not None and max(pct[a], pct[b]) >= 0.9:
                    weight *= 1.0 + top_boost
                loser_is_self = bool(selfs[l]) if selfs is not None else False
                if selfs is not None and (selfs[a] or selfs[b]):
                    weight *= self_boost
                winners.append(w)
                losers.append(l)
                weights.append(weight)
                margins.append(self_margin if loser_is_self else 0.0)
    if not winners:
        return None
    win_t = torch.tensor(winners, device=reward.device)
    lose_t = torch.tensor(losers, device=reward.device)
    # AMP makes reward outputs float16. Normalize the finite score gaps before
    # materializing them so legitimate large negative scores cannot overflow the
    # pair weights; keep the BT arithmetic in float32 while preserving gradients
    # through the cast reward values.
    weight_scale = math.fsum(weights) / len(weights)
    normalized_weights = [weight / weight_scale for weight in weights]
    weight_t = torch.tensor(normalized_weights, dtype=torch.float32, device=reward.device)
    margin_t = torch.tensor(margins, dtype=torch.float32, device=reward.device)
    if win_t.numel() > max_pairs:
        sel = torch.linspace(0, win_t.numel() - 1, max_pairs, device=reward.device).round().long()
        win_t, lose_t, weight_t, margin_t = win_t[sel], lose_t[sel], weight_t[sel], margin_t[sel]
    weight_t = weight_t / weight_t.mean().clamp_min(1e-6)
    pair_loss = torch.nn.functional.softplus(reward[lose_t].float() - reward[win_t].float() + margin_t)
    return (pair_loss * weight_t).mean()


def rank_pct_listwise_loss(
    reward: torch.Tensor,  # (N,)
    rank_pct: torch.Tensor,  # (N,) in [0,1] per-dataset percentiles
    dataset_ids: list[str],
    *,
    tau: float = 0.15,
    min_group: int = 3,
) -> torch.Tensor | None:
    """ListNet CE with per-dataset softmax targets over RANK PERCENTILES.

    Percentiles span [0,1] within every dataset by construction, so at tau=0.15 the
    target puts ~decisive mass on the top of the list — unlike raw deltas (median ~0.01)
    whose near-uniform targets made constant rewards optimal (the old collapse).
    """
    pair = rank_pct_listwise_terms(reward, rank_pct, dataset_ids, tau=tau, min_group=min_group)
    return None if pair is None else pair[0]


def rank_pct_listwise_terms(
    reward: torch.Tensor,  # (N,)
    rank_pct: torch.Tensor,  # (N,) in [0,1] per-dataset percentiles
    dataset_ids: list[str],
    *,
    tau: float = 0.15,
    min_group: int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(cross_entropy, target_entropy)`` for the ListNet objective above.

    ListNet CE is bounded below by the entropy of its OWN softmax target, and that entropy
    is a pure function of the group-size distribution and the percentile spacing — nothing
    the model can influence. Reporting CE alone therefore reports a large irreducible
    constant: measured train reward_listwise ran 2.4850 -> 2.2600 over 24 epochs against a
    target entropy of ~2.35, so the headline number was majority floor.

    Both tensors are averaged over the SAME groups with the SAME weighting, so
    ``cross_entropy - target_entropy`` is exactly the mean per-group KL(target || pred):
    zero at a perfect fit, always >= 0, and comparable across corpora.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    pcts = rank_pct.detach()
    for idx, d in enumerate(dataset_ids):
        grouped[d].append(idx)
    terms: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for indices in grouped.values():
        idx_t = torch.tensor(indices, device=reward.device)
        p = pcts[idx_t]
        if idx_t.numel() < min_group or not torch.isfinite(p).all():
            continue
        target = torch.softmax(p / tau, dim=0)
        logp = torch.log_softmax(reward[idx_t], dim=0)
        terms.append(-(target * logp).sum())
        # H(target) in nats. target is a softmax so it is strictly positive; the clamp
        # only guards float16 underflow on very peaked groups.
        entropies.append(-(target * target.clamp_min(1e-12).log()).sum())
    if not terms:
        return None
    return torch.stack(terms).mean(), torch.stack(entropies).mean().detach()


# Stationary class weight for the instability head.
#
# The v3.3 code recomputed pos_weight = negatives/positives PER BATCH and PER TASK. That is
# a moving objective: the quantity being minimised changes with every draw, the reported
# number is not comparable between two batches let alone two checkpoints, and on a task
# whose positives are rare a single-positive batch produced pos_weight ~255.
#
# Measured on the live validation split (9,152 rows,
# snapshot-001805437-full_consolidation-eq32v32c128):
#   regression: 7,776 labelled rows, positive (oof_r2_raw < 0) rate 0.13735 -> (1-p)/p = 6.28
#   binary:     1,376 labelled rows, positive rate EXACTLY 0.0
# so 6.0 is the stationary weight, and the binary task supplies no positives at all — see
# ``stability_bce_term`` for why that task is now skipped instead of averaged in.
STABILITY_POS_WEIGHT = 6.0


def stability_bce_term(
    logits: torch.Tensor,  # (N,) raw instability logits
    labels: torch.Tensor,  # (N,) float {0,1}
    *,
    pos_weight: float = STABILITY_POS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(weighted_bce, best_constant_predictor_bce)`` or None when the term is degenerate.

    Returns None unless BOTH classes are present. A pos_weighted BCE over an all-negative
    label set is not a classifier objective at all — its only minimiser is logit -> -inf,
    it contributes no discriminative gradient, and averaging it against a real term (as
    v3.3 did across regression/binary) drags the reported component toward the degenerate
    task. On the live corpus the binary task has zero positives in 1,376 labelled rows, so
    exactly half of the reported stability_bce was this artefact.

    The second tensor is the loss of the best CONSTANT predictor under the same weight and
    the batch's own base rate p:
        q* = w p / (w p + 1 - p),   L* = -(w p ln q* + (1-p) ln(1-q*))
    Subtracting it turns "is this head better than predicting the base rate?" into the sign
    of a single number. Measured val stability_bce was 1.3079 (epoch 1) -> 1.6479 (epoch 24)
    against a constant-predictor floor of 1.1959 at p = 0.13735, w = 6.28 — worse than
    constant from the first epoch and 38% worse by the last.
    """
    labels = labels.float()
    positives = labels.sum()
    if positives <= 0 or positives >= labels.numel():
        return None
    weight = torch.as_tensor(float(pos_weight), device=logits.device, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    with torch.no_grad():
        p = (positives / labels.numel()).float()
        w = weight
        q = (w * p / (w * p + 1.0 - p)).clamp(1e-6, 1.0 - 1e-6)
        baseline = -(w * p * q.log() + (1.0 - p) * (1.0 - q).log())
    return loss, baseline


def rwr_weights(
    dataset_ids: list[str],
    robust_scores: list[float | None],
    provenance: list[str],
    *,
    tau_floor: float = 0.05,
    tau_scale: float = 0.5,
    source_multiplier: Mapping[str, float] | None = None,
) -> list[float]:
    """Expert-iteration reward-weighted-regression weights.

    Per dataset group (cross-engine): A_i = y_i - mean(y); tau = max(tau_scale*std, floor);
    w = |G| * softmax(A/tau) (mean 1 per group), times a provenance multiplier. Rows
    without a score get weight equal to the multiplier (pure imitation weight).
    Pure function of labels: computed once at corpus load, never per batch.
    """
    # gp_report rows are curated winners — the primary imitation signal until loop rows
    # exist. Random seeds are ranking fodder, not imitation targets (RF5: the old
    # {gp:0.25, seed:0.5} handed 63% of imitation weight to random junk).
    multipliers = {
        "loop": 1.0,
        "loop_old": 0.5,
        "gp_report": 1.0,
        "historical_import": 0.25,
        "seed": 0.25,
        # self-play generations ("self_g3") and miner champions ("champion_g3"):
        # champions are the rising imitation targets — weighted ABOVE everything else.
        "self": 1.0,
        "champion": 1.5,
    }
    if source_multiplier:
        multipliers.update(source_multiplier)

    def _mult(src: str) -> float:
        if src in multipliers:
            return multipliers[src]
        base = src.split("_g")[0]  # self_g3 -> self, champion_g2 -> champion
        return multipliers.get(base, 0.25)

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, (dataset_id, score) in enumerate(zip(dataset_ids, robust_scores)):
        if score is not None and math.isfinite(score):
            grouped[dataset_id].append(idx)
    weights = [_mult(src) for src in provenance]
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        values = [float(robust_scores[i]) for i in indices]  # type: ignore[arg-type]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        tau = max(tau_scale * std, tau_floor)
        exps = [math.exp(min((v - mean) / tau, 30.0)) for v in values]
        total = sum(exps)
        for i, e in zip(indices, exps):
            weights[i] = weights[i] * (len(indices) * e / total)
    return weights
