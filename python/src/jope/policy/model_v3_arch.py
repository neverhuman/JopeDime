"""JopePolicyModelV3: autoregressive grammar-masked policy + reward ensemble.

Architecture (per the approved plan):
- dataset context: the v2 encoder MLP (dataset 512 + target 512 -> 256), injected as a
  prefix embedding CTX = W_ctx(z) at position 0 of a 2-layer causal pre-LN Transformer
  (d=256, 4 heads, FFN 1024, tied input/output embeddings, learned positions).
- reward: a separate 1-layer bidirectional token encoder (shared embedding TABLE only,
  not the trunk — the reward must not drift when EI/GRPO moves the policy) with
  attention pooling -> g; K=4 reward heads on cat(z, g); mean = estimate, std =
  epistemic uncertainty; runtime head on cat(z, g) predicts log1p(cost_ms).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..model import _reward_listwise_loss
from .automaton import replay_masks
from .losses import (
    LISTWISE_TARGET_TAU,
    PHASE_A_PROFILE,
    PHASE_B_PROFILE,
    POLICY_V3_LOSS_COMPONENTS,
    STABILITY_POS_WEIGHT,
    bootstrap_mask,
    kg_combine,
    masked_sequence_nll,
    rank_pct_listwise_terms,
    rwr_weights,
    self_play_bt_loss,
    stability_bce_term,
)
from .tokens import PAD, ActionVocab, encode_actions
from .model_v3_arch_methods import PolicyV3ModelMethods

MAX_SEQ_LEN = 128
MODEL_VERSION = "policy_v3.4-split-context"
LEGACY_MODEL_VERSION = "policy_v3.3-fused-heads"

# ---- v3.4 input conditioning constants -------------------------------------------------
#
# Measured on the live context catalog
# ($R/catalog/afa412d4….f32.bin, 19,951 rows, dataset_dimensions=1536, target_dimensions=512,
#  record_stride_bytes=8192) and independently on the catalog the frozen packed corpus binds
# ($R/catalog/3875bc2f….f32.bin, 8,485 rows). Both agree exactly:
#
#   dataset block  per-element RMS  1.228194 (current) / 1.244325 (packed); per-row L2 47.07
#   target  block  per-element RMS  0.044194 on BOTH; per-row L2 exactly 1.0000000
#   target dims that are all-zero across every row: 376, and they are the CONTIGUOUS TAIL
#   [136, 511]. Verified at source too: /home/ubuntu/remote_super/quant/regression/
#   0014b9dad5725a1e/train_target.vs is 512 floats, L2 = 1.0000001, with exactly 136 non-zero
#   entries and max non-zero index 135. selfplay_expansion.rs:990 target_summary() zero-pads
#   the sidecar to TARGET_DIM, and validate_sidecar (:2110) fixes the sidecar at 1x512.
#
# Consequence in v3.3: cat([dataset, target]) -> Linear(2048, 512) with NO per-block scaling,
# so the target block carried 1.0 / (47.07^2 + 1.0) = 0.045% of the input variance, and the
# 376 dead columns of encoder.0.weight (376 x 512 = 192,512 parameters) received exactly zero
# gradient forever.
#
# v3.4 fixes both: the target block is sliced to its live width and each block is rescaled to
# unit per-element RMS. The rescale is a pure reparameterisation — the migration divides the
# matching weight columns by the same constant, so the function is preserved EXACTLY — but it
# equalises the two blocks' effective learning rate, because an Adam step of fixed size on a
# weight column moves the pre-activation by (step x input magnitude).
DATASET_BLOCK_RMS = 1.228194
TARGET_BLOCK_RMS = 0.085749  # = 1 / sqrt(136): unit-L2 row spread over its live width
DEFAULT_TARGET_KEEP_DIM = 136
# Number of forward passes on which the dropped target tail is verified to be zero. The tail
# is structural, but this system is explicitly designed to grow encodings forever, so a wider
# target sidecar must fail loudly rather than be silently truncated.
TARGET_TAIL_CHECK_BATCHES = 64

# Four-task program (D4): declared lane aliases for the NEURAL heads only.
# timeseries rides the regression reward/stability/routing heads (reduction-v2
# tables are regression problems); multiclass rides the binary heads. Genome-level
# multiclass is NOT aliased — it gets its own ovr_ridge engine head (M1).
# Aliasing is declared here and recorded in checkpoint metadata
# (PrimeTaskHeads.v2) — never silent: unknown tasks raise in _task_lane_mask.
TASK_LANE = {
    "regression": "regression",
    "binary": "binary",
    "timeseries": "regression",
    "multiclass": "binary",
}


def _task_lane_mask(task_types: Sequence[str] | None, rows: int, device: torch.device) -> torch.Tensor:
    """Boolean binary-lane mask (True = binary-lane heads) for one batch.

    Tasks map through TASK_LANE; any task outside it raises — the reward_stats
    incident precedent: a missing/typo'd task must never silently route to the
    regression heads.
    """
    task_types = list(task_types or ["regression"] * rows)
    if len(task_types) != rows:
        raise ValueError("task types must contain exactly one value per row")
    unknown = sorted({task for task in task_types if task not in TASK_LANE})
    if unknown:
        raise ValueError(f"unknown task type(s) {unknown}; known tasks: {sorted(TASK_LANE)}")
    return torch.tensor(
        [TASK_LANE[task] == "binary" for task in task_types],
        dtype=torch.bool,
        device=device,
    )


@dataclass
class PolicyV3Batch:
    embedding: torch.Tensor  # (B, 512)
    target_summary: torch.Tensor  # (B, 512)
    dataset_ids: list[str]
    reward_pair_ids: list[str]
    engines: list[str]
    task_types: list[str]
    token_ids: torch.Tensor  # (B, T) int64, starts with BOS, PAD-padded
    pad_mask: torch.Tensor  # (B, T) bool, True = real token
    legal: torch.Tensor  # (B, T-1, V) bool, legality for predicting token t (t>=1)
    robust_delta: torch.Tensor  # (B,) float, nan when unscored
    rwr_weight: torch.Tensor  # (B,) float
    runtime: torch.Tensor  # (B,) float, nan when unknown
    bootstrap: torch.Tensor  # (B, K) bool
    bad_label: torch.Tensor | None = None  # (B,) float {0,1}, nan unknown (RF2 labels)
    canonicals: list[str] = field(default_factory=list)
    # v3.1 self-play / ranking labels (nan/False when absent → losses skip)
    rank_pct: torch.Tensor | None = None  # (B,) float in [0,1], per-dataset rank percentile
    is_self: torch.Tensor | None = None  # (B,) bool, row came from a former model generation
    former_best: torch.Tensor | None = (
        None  # (B,) float, best former-self score for the row's dataset (nan none)
    )
    routing_label: torch.Tensor | None = None  # (B,) int64, winner-head-family class (-1 unknown)
    seq_weight: torch.Tensor | None = (
        None  # (B,) float, provenance imitation weight (V7: anti self-confirmation)
    )


class JopePolicyModelV3(PolicyV3ModelMethods, nn.Module):
    def __init__(
        self,
        vocab: ActionVocab,
        *,
        dataset_dim: int = 512,
        target_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        reward_heads: int = 4,
        embedding_rows: int | None = None,
        routing_classes: int = 0,
        fused_task_heads: bool = False,
        target_keep_dim: int | None = None,
        split_reward_context: bool = True,
        input_block_norm: bool = True,
        dataset_block_rms: float = DATASET_BLOCK_RMS,
        target_block_rms: float = TARGET_BLOCK_RMS,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        # Experimental full-table lineage; absent for historical grammar snapshots.
        self.frozen_action_vocab: dict[str, Any] | None = None
        self.dataset_dim = dataset_dim
        self.target_dim = target_dim
        # v3.4: live width of the target block. None keeps all 512 dims (the v3.3 shape), so
        # every historical checkpoint rebuilds byte-identically; production passes the width
        # train_v3 measures from the corpus itself.
        if target_keep_dim is None:
            target_keep_dim = target_dim
        if not 1 <= int(target_keep_dim) <= target_dim:
            raise ValueError(f"target_keep_dim must be in [1, {target_dim}], got {target_keep_dim!r}")
        self.target_keep_dim = int(target_keep_dim)
        self.split_reward_context = bool(split_reward_context)
        self.input_block_norm = bool(input_block_norm)
        self.dataset_block_rms = float(dataset_block_rms)
        self.target_block_rms = float(target_block_rms)
        if min(self.dataset_block_rms, self.target_block_rms) <= 0.0:
            raise ValueError("block RMS constants must be positive")
        self._dataset_block_scale = 1.0 / self.dataset_block_rms if self.input_block_norm else 1.0
        self._target_block_scale = 1.0 / self.target_block_rms if self.input_block_norm else 1.0
        self._target_tail_checks_remaining = TARGET_TAIL_CHECK_BATCHES
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.reward_k = reward_heads
        self.embedding_rows = embedding_rows or (vocab.size + 256)
        # Training-only regularizers are configured explicitly by the trainer.
        # Zero preserves inference and the historical training path exactly.
        self.training_embedding_jitter = 0.0
        self.training_token_dropout = 0.0
        # v3.2 routing supervision: force the encoder to PRESERVE which-algorithm-
        # wins signal (probes: raw z carries 0.41 routing acc; the unsupervised
        # encoder output carried 0.28 — below baseline; seq/reward never demand it).
        self.routing_classes = routing_classes
        self.fused_task_heads = fused_task_heads
        self.routing_head = nn.Linear(d_model, routing_classes) if routing_classes > 0 else None
        self.binary_routing_head = (
            nn.Linear(d_model, routing_classes) if fused_task_heads and routing_classes > 0 else None
        )

        def _context_encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dataset_dim + self.target_keep_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        # v3.4 UN-SHARING. In v3.3 this single 1,313,792-parameter encoder was the ONLY
        # surface the two objectives shared: 63.9% of the parameters (the 6-layer decoder,
        # token_emb, pos_emb, ctx_proj) served seq_nll exclusively and the reward branch had
        # its own embedding table, so every gradient conflict between "predict the next
        # token" and "rank genomes / classify the winning family" was forced through z.
        # Measured symptom: routing_ce train 0.0078 vs val 1.3865 (a 178x ratio on a 14-way
        # classifier over 2,571 datasets) — the shared z was encoding dataset IDENTITY.
        #
        # `encoder` now serves the sequence path alone; `reward_encoder` serves every
        # reward-side consumer (z_gamma, z_proj, the reward heads, runtime, stability,
        # routing). They start as exact clones of each other on a migrated warm start, so
        # generation 0 of v3.4 is numerically identical to its v3.3 parent, and diverge
        # thereafter because they finally receive different gradients.
        self.encoder = _context_encoder()
        self.reward_encoder = _context_encoder() if self.split_reward_context else None
        self.ctx_proj = nn.Linear(d_model, d_model)

        self.token_emb = nn.Embedding(self.embedding_rows, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN + 1, d_model)  # +1 for the CTX prefix
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        # tied output projection (embedding table transpose)

        # reward branch
        reward_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        del reward_layer  # the transformer reward encoder collapsed; see genome_embedding
        # Independent embedding table: the reward branch must not chase representations
        # that seq-NLL is reshaping (drift was a collapse driver).
        self.reward_token_emb = nn.Embedding(self.embedding_rows, d_model, padding_idx=PAD)
        self.reward_norm = nn.LayerNorm(d_model)
        # v3.1 CONDITIONAL reward: score(dataset, genome), not score(genome).
        # The earlier collapse came from giving the head a PURE-z pathway (a per-dataset
        # constant that dominates when the pairwise signal is weak). The fix is
        # interaction-ONLY conditioning: every z term is multiplied against or differenced
        # with g, so a constant-in-dataset shortcut cannot exist, while within-dataset
        # ranking finally sees dataset↔genome compatibility.
        #   phi = [g, g ⊙ γ(z), |g − proj(z)|]  (3·d_model)
        self.z_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.z_proj = nn.Linear(d_model, d_model)
        self.reward_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(d_model, 1),
                )
                for _ in range(reward_heads)
            ]
        )
        self.binary_reward_heads = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(3 * d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(0.05),
                        nn.Linear(d_model, 1),
                    )
                    for _ in range(reward_heads)
                ]
            )
            if fused_task_heads
            else None
        )
        # Frozen-purpose ABLATION head (genome-only): trained with the same BT loss, used
        # only as a validation control — the conditional heads must beat it on pairwise
        # win-rate or conditioning is not adding information (anti-collapse guard).
        self.reward_head_ablation = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.05), nn.Linear(d_model, 1)
        )
        self.runtime_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # RF2: blowup/instability predictor — now dataset-conditioned (a transform safe at
        # n=10k/p=20 blows up at n=200/p=2000). Same interaction-only feature map.
        self.stability_head = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.binary_stability_head = (
            nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
            if fused_task_heads
            else None
        )
        self.loss_log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in POLICY_V3_LOSS_COMPONENTS}
        )

    # ---- context ------------------------------------------------------------
    def context_input(self, embedding: torch.Tensor, target_summary: torch.Tensor | None) -> torch.Tensor:
        """Per-block-normalised, live-width encoder input. Shared by both encoders."""
        if target_summary is None:
            target_summary = torch.zeros(
                (embedding.shape[0], self.target_dim), dtype=embedding.dtype, device=embedding.device
            )
        target_summary = target_summary.float()
        if self.target_keep_dim < target_summary.shape[-1]:
            if self._target_tail_checks_remaining > 0:
                self._target_tail_checks_remaining -= 1
                tail = target_summary[..., self.target_keep_dim :]
                if tail.numel() and bool(tail.abs().amax() > 0):
                    raise ValueError(
                        "target summary carries signal in dims dropped by "
                        f"target_keep_dim={self.target_keep_dim}: the target encoding widened "
                        "and this checkpoint would silently truncate it"
                    )
            target_summary = target_summary[..., : self.target_keep_dim]
        dataset = embedding.float() * self._dataset_block_scale
        target = target_summary * self._target_block_scale
        return torch.cat([dataset, target], dim=-1)

    def context(self, embedding: torch.Tensor, target_summary: torch.Tensor | None) -> torch.Tensor:
        """SEQUENCE-path dataset context. Consumers: ctx_proj -> the causal decoder."""
        return self.encoder(self.context_input(embedding, target_summary))

    def reward_context(self, embedding: torch.Tensor, target_summary: torch.Tensor | None) -> torch.Tensor:
        """REWARD-path dataset context.

        Consumers: z_gamma, z_proj (hence every reward head), runtime_head, the stability
        heads and the routing heads. On a v3.3 checkpoint (split_reward_context=False) this
        falls back to the single shared encoder, so historical checkpoints keep their exact
        behaviour.
        """
        encoder = self.reward_encoder if self.reward_encoder is not None else self.encoder
        return encoder(self.context_input(embedding, target_summary))

    # ---- policy trunk ---------------------------------------------------------
    def _full_logits(self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Causal logits over all h positions 0..T (h = [CTX, tok_0..tok_{T-1}]).

        h index j >= 1 holds token j-1 and predicts token j; h[L] (L = prefix length)
        predicts the NEXT token after the prefix. Returns (B, T+1, V).
        """
        B, T = token_ids.shape
        ctx = self.ctx_proj(z).unsqueeze(1)  # (B, 1, d)
        tok = self.token_emb(token_ids)  # (B, T, d)
        h = torch.cat([ctx, tok], dim=1)  # (B, T+1, d)
        positions = torch.arange(T + 1, device=token_ids.device)
        h = h + self.pos_emb(positions).unsqueeze(0)
        causal = torch.triu(torch.ones(T + 1, T + 1, device=token_ids.device, dtype=torch.bool), diagonal=1)
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device), ~pad_mask], dim=1)
        h = self.decoder(h, mask=causal, src_key_padding_mask=key_pad)
        h = self.out_norm(h)
        logits = h @ self.token_emb.weight.t()  # (B, T+1, rows)
        return logits[..., : self.vocab.size]

    def decoder_logits(
        self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forcing logits for predicting tokens 1..T-1. Returns (B, T-1, V)."""
        T = token_ids.shape[1]
        return self._full_logits(z, token_ids, pad_mask)[:, 1:T, :]

    def next_token_logits(
        self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Next-token logits after each row's prefix. Returns (B, V)."""
        full = self._full_logits(z, token_ids, pad_mask)
        lengths = pad_mask.sum(dim=1)  # prefix length L per row; h[L] predicts token L
        return full[torch.arange(token_ids.shape[0], device=token_ids.device), lengths]

    # ---- reward branch --------------------------------------------------------
    def genome_embedding(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled independent token embeddings (learned bag-of-actions).

        This is the learned generalization of the v2 handcrafted featurizer (op counts /
        head one-hot), which provably supported pairwise_win 0.571. A from-scratch
        transformer encoder under a weak pairwise signal repeatedly collapsed; mean
        pooling makes the genome->feature map trivially non-constant and well-behaved.
        """
        emb = self.reward_token_emb(token_ids)
        mask = pad_mask.unsqueeze(-1).float()
        if self.training and self.training_token_dropout > 0.0:
            mask = mask * (torch.rand_like(mask) >= self.training_token_dropout)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.reward_norm(pooled)

    def _zg_features(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Interaction-only dataset↔genome features: no pure-z pathway (see __init__)."""
        return torch.cat([g, g * self.z_gamma(z), torch.abs(g - self.z_proj(z))], dim=-1)

    def reward_all(
        self,
        z: torch.Tensor,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        task_types: Sequence[str] | None = None,
    ) -> torch.Tensor:
        g = self.genome_embedding(token_ids, pad_mask)
        phi = self._zg_features(z, g)
        regression = torch.stack([head(phi).squeeze(-1) for head in self.reward_heads], dim=-1)
        binary = (
            torch.stack([head(phi).squeeze(-1) for head in self.binary_reward_heads], dim=-1)
            if self.binary_reward_heads is not None
            else regression
        )
        mask = _task_lane_mask(task_types, z.shape[0], z.device).unsqueeze(-1)
        return torch.where(mask, binary, regression)

    def reward_ablation(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Genome-only control score — the conditional heads must beat this on val."""
        g = self.genome_embedding(token_ids, pad_mask)
        return self.reward_head_ablation(g).squeeze(-1)

    def reward_stats(
        self, z, token_ids, pad_mask, task_types: Sequence[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self.reward_all(z, token_ids, pad_mask, task_types)
        return rewards.mean(dim=-1), rewards.std(dim=-1)

    def runtime_pred(self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        g = self.genome_embedding(token_ids, pad_mask)
        return F.softplus(self.runtime_head(torch.cat([z, g], dim=-1)).squeeze(-1))

    def instability_logit(
        self,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        z: torch.Tensor | None = None,
        task_types: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Logit of P(bad outcome) for a genome — dataset-conditioned when z is given
        (a transform safe on wide-short data can blow up on tall-narrow and vice versa)."""
        g = self.genome_embedding(token_ids, pad_mask)
        if z is None:
            z = torch.zeros_like(g)
        phi = self._zg_features(z, g)
        regression = self.stability_head(phi).squeeze(-1)
        binary = (
            self.binary_stability_head(phi).squeeze(-1)
            if self.binary_stability_head is not None
            else regression
        )
        mask = _task_lane_mask(task_types, z.shape[0], z.device)
        return torch.where(mask, binary, regression)

    def score_token_sequences(
        self,
        embedding: torch.Tensor,  # (512,) or (1, 512) — one dataset
        target_summary: torch.Tensor | None,
        sequences: list[list[int]],
        device: str | torch.device = "cpu",
        task: str = "regression",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reward (mean, std) for K token sequences against one dataset context."""
        from .model_v3_batch import pad_sequences

        emb = embedding.float().reshape(1, -1).to(device)
        tgt = None if target_summary is None else target_summary.float().reshape(1, -1).to(device)
        z = self.reward_context(emb, tgt)  # v3.4: reward heads read the reward encoder
        token_ids, pad = pad_sequences(sequences, device=device)
        zk = z.expand(token_ids.shape[0], -1)
        return self.reward_stats(zk, token_ids, pad, [task] * token_ids.shape[0])

    # ---- losses ---------------------------------------------------------------
    def forward(self, batch: PolicyV3Batch, *, phase: str = "B") -> dict[str, torch.Tensor]:
        # Route through forward so DistributedDataParallel registers its grad-sync hooks;
        # single-process training calls this identically.
        return self.loss_components(batch, phase=phase)



    def set_phase(self, phase: str) -> None:
        """Curriculum switch. The reward branch trains in BOTH phases (freezing it in
        phase A made phase B start from random init and collapse to constant rewards);
        phases differ in RWR weighting and z-encoder LR, handled by the trainer."""
        del phase

    # ---- persistence ------------------------------------------------------------
