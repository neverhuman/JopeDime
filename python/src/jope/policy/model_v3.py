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


def _task_lane_mask(
    task_types: Sequence[str] | None, rows: int, device: torch.device
) -> torch.Tensor:
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
        raise ValueError(
            f"unknown task type(s) {unknown}; known tasks: {sorted(TASK_LANE)}"
        )
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
    former_best: torch.Tensor | None = None  # (B,) float, best former-self score for the row's dataset (nan none)
    routing_label: torch.Tensor | None = None  # (B,) int64, winner-head-family class (-1 unknown)
    seq_weight: torch.Tensor | None = None  # (B,) float, provenance imitation weight (V7: anti self-confirmation)


class JopePolicyModelV3(nn.Module):
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
        self.dataset_dim = dataset_dim
        self.target_dim = target_dim
        # v3.4: live width of the target block. None keeps all 512 dims (the v3.3 shape), so
        # every historical checkpoint rebuilds byte-identically; production passes the width
        # train_v3 measures from the corpus itself.
        if target_keep_dim is None:
            target_keep_dim = target_dim
        if not 1 <= int(target_keep_dim) <= target_dim:
            raise ValueError(
                f"target_keep_dim must be in [1, {target_dim}], got {target_keep_dim!r}"
            )
        self.target_keep_dim = int(target_keep_dim)
        self.split_reward_context = bool(split_reward_context)
        self.input_block_norm = bool(input_block_norm)
        self.dataset_block_rms = float(dataset_block_rms)
        self.target_block_rms = float(target_block_rms)
        if min(self.dataset_block_rms, self.target_block_rms) <= 0.0:
            raise ValueError("block RMS constants must be positive")
        self._dataset_block_scale = (
            1.0 / self.dataset_block_rms if self.input_block_norm else 1.0
        )
        self._target_block_scale = (
            1.0 / self.target_block_rms if self.input_block_norm else 1.0
        )
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
            nn.Linear(d_model, routing_classes)
            if fused_task_heads and routing_classes > 0
            else None
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
        self.binary_reward_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(d_model, 1),
                )
                for _ in range(reward_heads)
            ]
        ) if fused_task_heads else None
        # Frozen-purpose ABLATION head (genome-only): trained with the same BT loss, used
        # only as a validation control — the conditional heads must beat it on pairwise
        # win-rate or conditioning is not adding information (anti-collapse guard).
        self.reward_head_ablation = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.05), nn.Linear(d_model, 1)
        )
        self.runtime_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        # RF2: blowup/instability predictor — now dataset-conditioned (a transform safe at
        # n=10k/p=20 blows up at n=200/p=2000). Same interaction-only feature map.
        self.stability_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.binary_stability_head = (
            nn.Sequential(
                nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
            if fused_task_heads
            else None
        )
        self.loss_log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in POLICY_V3_LOSS_COMPONENTS}
        )

    # ---- context ------------------------------------------------------------
    def context_input(
        self, embedding: torch.Tensor, target_summary: torch.Tensor | None
    ) -> torch.Tensor:
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

    def reward_context(
        self, embedding: torch.Tensor, target_summary: torch.Tensor | None
    ) -> torch.Tensor:
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
        key_pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device), ~pad_mask], dim=1
        )
        h = self.decoder(h, mask=causal, src_key_padding_mask=key_pad)
        h = self.out_norm(h)
        logits = h @ self.token_emb.weight.t()  # (B, T+1, rows)
        return logits[..., : self.vocab.size]

    def decoder_logits(self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing logits for predicting tokens 1..T-1. Returns (B, T-1, V)."""
        T = token_ids.shape[1]
        return self._full_logits(z, token_ids, pad_mask)[:, 1:T, :]

    def next_token_logits(self, z: torch.Tensor, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
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
            mask = mask * (
                torch.rand_like(mask) >= self.training_token_dropout
            )
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
        regression = torch.stack(
            [head(phi).squeeze(-1) for head in self.reward_heads], dim=-1
        )
        binary = (
            torch.stack(
                [head(phi).squeeze(-1) for head in self.binary_reward_heads], dim=-1
            )
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

    def loss_components(self, batch: PolicyV3Batch, *, phase: str = "B") -> dict[str, torch.Tensor]:
        embedding = batch.embedding
        if (
            self.training
            and self.training_embedding_jitter > 0.0
            and embedding.shape[0] > 1
        ):
            scale = embedding.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            embedding = embedding + (
                self.training_embedding_jitter
                * scale
                * torch.randn_like(embedding)
            )
        z = self.context(embedding, batch.target_summary)
        # v3.4: every reward-side consumer reads its own encoder. On a v3.3 checkpoint
        # reward_context() returns the same encoder, so behaviour is unchanged.
        z_reward = self.reward_context(embedding, batch.target_summary)
        components: dict[str, torch.Tensor] = {}

        if self.routing_head is not None and batch.routing_label is not None:
            binary_mask = _task_lane_mask(
                batch.task_types, z_reward.shape[0], z_reward.device
            )
            routing_terms: list[torch.Tensor] = []
            for task_mask, head in (
                (~binary_mask, self.routing_head),
                (binary_mask, self.binary_routing_head),
            ):
                mask = task_mask & (batch.routing_label >= 0)
                if head is not None and mask.sum() >= 2:
                    routing_terms.append(
                        F.cross_entropy(head(z_reward[mask]), batch.routing_label[mask])
                    )
            if routing_terms:
                components["routing_ce"] = torch.stack(routing_terms).mean()

        logits = self.decoder_logits(z, batch.token_ids, batch.pad_mask)
        targets = batch.token_ids[:, 1:]
        target_pad = batch.pad_mask[:, 1:]
        row_weights = batch.rwr_weight if phase == "B" else None
        # V7: provenance imitation weights — winners imitate at full weight, generic
        # self rows barely, unscored non-corpus rows not at all (anti self-confirmation).
        if batch.seq_weight is not None:
            row_weights = batch.seq_weight if row_weights is None else row_weights * batch.seq_weight
        components["seq_nll"] = masked_sequence_nll(
            logits, targets, batch.legal, target_pad, row_weights=row_weights
        )

        finite = torch.isfinite(batch.robust_delta)
        if finite.sum() >= 2:
            rewards = self.reward_all(
                z_reward, batch.token_ids, batch.pad_mask, batch.task_types
            )  # (B, K)
            keep_all = finite
            task_reward_pair_ids = [
                f"{task}:{identity}"
                for task, identity in zip(batch.task_types, batch.reward_pair_ids)
            ]
            ids_all = [d for d, m in zip(task_reward_pair_ids, keep_all.tolist()) if m]
            rank_pct = batch.rank_pct[keep_all] if batch.rank_pct is not None else None
            is_self = batch.is_self[keep_all] if batch.is_self is not None else None
            bt_terms: list[torch.Tensor] = []
            lw_terms: list[torch.Tensor] = []
            lw_entropies: list[torch.Tensor] = []
            for k in range(self.reward_k):
                keep = batch.bootstrap[:, k] & finite
                if keep.sum() < 2:
                    continue
                ids = [d for d, m in zip(task_reward_pair_ids, keep.tolist()) if m]
                # v3.1: margin-shifted top-heavy self-aware BT (centered by construction)
                bt = self_play_bt_loss(
                    rewards[keep, k],
                    batch.robust_delta[keep],
                    ids,
                    rank_pct=batch.rank_pct[keep] if batch.rank_pct is not None else None,
                    is_self=batch.is_self[keep] if batch.is_self is not None else None,
                )
                if bt is not None:
                    bt_terms.append(bt)
                if batch.rank_pct is not None:
                    pair = rank_pct_listwise_terms(
                        rewards[keep, k], batch.rank_pct[keep], ids
                    )
                    lw = None if pair is None else pair[0]
                    if pair is not None:
                        lw_entropies.append(pair[1])
                else:
                    lw = _reward_listwise_loss(
                        rewards[keep, k], batch.robust_delta[keep] / LISTWISE_TARGET_TAU, ids
                    )
                if lw is not None:
                    lw_terms.append(lw)
            if bt_terms:
                components["reward_bt"] = torch.stack(bt_terms).mean()
            if lw_terms:
                components["reward_listwise"] = torch.stack(lw_terms).mean()
            if lw_entropies:
                # Diagnostic only — kg_combine skips it and loss_log_vars is not keyed on
                # it, so this cannot change the checkpoint tensor count.
                components["reward_listwise_entropy"] = torch.stack(lw_entropies).mean()
            # Genome-only ablation control: identical BT objective on a DETACHED genome
            # embedding — trains only the ablation head; its val pairwise-win is the bar
            # the conditional heads must beat (anti-collapse guard).
            if keep_all.sum() >= 2:
                g_detached = self.genome_embedding(batch.token_ids, batch.pad_mask).detach()
                abl = self.reward_head_ablation(g_detached[keep_all]).squeeze(-1)
                abl_bt = self_play_bt_loss(abl, batch.robust_delta[keep_all], ids_all,
                                           rank_pct=rank_pct, is_self=is_self)
                if abl_bt is not None:
                    components["ablation_bt"] = abl_bt

        runtime_finite = torch.isfinite(batch.runtime)
        if runtime_finite.any():
            pred = self.runtime_pred(z_reward, batch.token_ids, batch.pad_mask)
            components["runtime_huber"] = F.huber_loss(
                torch.log1p(pred[runtime_finite]), torch.log1p(batch.runtime[runtime_finite].float())
            )
        if batch.bad_label is not None:
            bad_finite = torch.isfinite(batch.bad_label)
            if bad_finite.any():
                stability_logits = self.instability_logit(
                    batch.token_ids, batch.pad_mask, z_reward, batch.task_types
                )
                binary_mask = _task_lane_mask(
                    batch.task_types, z_reward.shape[0], z_reward.device
                )
                stability_terms: list[torch.Tensor] = []
                stability_baselines: list[torch.Tensor] = []
                for task_mask in (~binary_mask, binary_mask):
                    mask = bad_finite & task_mask
                    if not mask.any():
                        continue
                    # v3.4: stationary pos_weight, and single-class subsets are SKIPPED.
                    # v3.3 recomputed pos_weight = negatives/positives per batch per task
                    # (a moving objective, uncomparable between batches) and still emitted a
                    # term for the binary task, whose measured positive rate on the live
                    # validation split is exactly 0 in 1,376 labelled rows — half of the
                    # reported stability_bce was an all-negative BCE whose only minimiser is
                    # logit -> -inf.
                    pair = stability_bce_term(
                        stability_logits[mask],
                        batch.bad_label[mask],
                        pos_weight=STABILITY_POS_WEIGHT,
                    )
                    if pair is None:
                        continue
                    stability_terms.append(pair[0])
                    stability_baselines.append(pair[1])
                if stability_terms:
                    components["stability_bce"] = torch.stack(stability_terms).mean()
                    components["stability_bce_baseline"] = torch.stack(
                        stability_baselines
                    ).mean()
        return components

    def loss(self, batch: PolicyV3Batch, *, phase: str = "B") -> torch.Tensor:
        profile = PHASE_A_PROFILE if phase == "A" else PHASE_B_PROFILE
        return kg_combine(self.loss_components(batch, phase=phase), self.loss_log_vars, profile)

    def set_phase(self, phase: str) -> None:
        """Curriculum switch. The reward branch trains in BOTH phases (freezing it in
        phase A made phase B start from random init and collapse to constant rewards);
        phases differ in RWR weighting and z-encoder LR, handled by the trainer."""
        del phase

    # ---- persistence ------------------------------------------------------------
    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "model": self.state_dict(),
            "action_vocab": {"tokens": self.vocab.tokens, "version": self.vocab.version},
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "dataset_dim": self.dataset_dim,
            "target_dim": self.target_dim,
            "reward_ensemble_k": self.reward_k,
            "embedding_rows": self.embedding_rows,
            "routing_classes": self.routing_classes,
            "fused_task_heads": self.fused_task_heads,
            # v3.4 architecture record. load_policy_v3 defaults every one of these to its
            # v3.3 value when absent, so a historical checkpoint rebuilds byte-identically.
            "target_keep_dim": self.target_keep_dim,
            "split_reward_context": self.split_reward_context,
            "input_block_norm": self.input_block_norm,
            "dataset_block_rms": self.dataset_block_rms,
            "target_block_rms": self.target_block_rms,
            "context_schema": {
                "schema": "PrimeSplitContext.v1",
                "sequence_encoder": "encoder",
                "reward_encoder": "reward_encoder" if self.split_reward_context else "encoder",
                "reward_consumers": [
                    "z_gamma",
                    "z_proj",
                    "reward_heads",
                    "binary_reward_heads",
                    "runtime_head",
                    "stability_head",
                    "binary_stability_head",
                    "routing_head",
                    "binary_routing_head",
                ],
            },
            "task_head_schema": {
                # v2 (four-task program): still exactly two physical head lanes
                # (regression/binary); timeseries and multiclass are declared
                # ALIASES onto them (TASK_LANE, D4) — recorded here so a loader
                # can see the routing without reading code. v1 checkpoints
                # (schema "PrimeTaskHeads.v1", 2 tasks, identity lanes) load
                # unchanged: the physical architecture is identical.
                "schema": "PrimeTaskHeads.v2",
                "tasks": (
                    ["regression", "binary", "timeseries", "multiclass"]
                    if self.fused_task_heads
                    else ["regression"]
                ),
                "lanes": (
                    dict(TASK_LANE)
                    if self.fused_task_heads
                    else {"regression": "regression"}
                ),
                "aliased_tasks": (
                    ["timeseries", "multiclass"] if self.fused_task_heads else []
                ),
                "reward_ensemble_per_task": self.reward_k,
                "separate_stability": self.fused_task_heads,
                "separate_routing": self.fused_task_heads and self.routing_classes > 0,
            },
            **extra,
        }


def pad_sequences(
    sequences: Sequence[Sequence[int]], device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in sequences)
    token_ids = torch.full((len(sequences), T), PAD, dtype=torch.long, device=device)
    pad_mask = torch.zeros((len(sequences), T), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        token_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        pad_mask[i, : len(seq)] = True
    return token_ids, pad_mask


def encode_row_artifacts(row, vocab: ActionVocab):
    """Encode one row to (token seq, per-position legal-id lists). Pure fn of the row —
    cache it once and reuse across epochs (this is the dominant per-epoch CPU cost)."""
    actions = (row.metadata or {}).get("actions")
    if not isinstance(actions, dict):
        from ..grammars import grammar_for
        from ..pipeline_text import parse_pipeline_actions

        actions = parse_pipeline_actions(row.canonical_genome or row.genome, grammar_for(row.engine))
    task = str((row.metadata or {}).get("task_type") or actions.get("task") or "regression")
    seq = encode_actions(row.engine, actions, vocab, task=task)
    masks = replay_masks(seq, vocab)
    return seq, masks


def row_artifact_key(row) -> str:
    """Stable identity for the pure genome->token/mask preprocessing result."""
    import hashlib
    import json

    metadata = row.metadata or {}
    payload = [
        row.engine,
        str(metadata.get("task_type") or "regression"),
        row.canonical_genome or row.genome,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def precompute_row_artifacts(
    rows: list,
    vocab: ActionVocab,
    persistent_cache: dict[str, tuple[list[int], list[list[int]]]] | None = None,
) -> dict[int, tuple[list[int], list[list[int]]]]:
    """Encode every row ONCE (token ids + legal masks), keyed by id(row). Rows that fall
    outside the policy action space are omitted (skip-not-crash, counted)."""
    from .automaton import IllegalToken

    cache: dict[int, tuple[list[int], list[list[int]]]] = {}
    for row in rows:
        key = row_artifact_key(row)
        if persistent_cache is not None and key in persistent_cache:
            cache[id(row)] = persistent_cache[key]
            continue
        try:
            artifact = encode_row_artifacts(row, vocab)
            cache[id(row)] = artifact
            if persistent_cache is not None:
                persistent_cache[key] = artifact
        except (IllegalToken, ValueError):
            vocab.counters["rows_skipped_encode"] = vocab.counters.get("rows_skipped_encode", 0) + 1
    return cache


def rows_to_policy_v3_batch(
    rows: list,
    vocab: ActionVocab,
    *,
    device: str | torch.device = "cpu",
    reward_k: int = 4,
    precomputed_rwr: dict[int, float] | None = None,
    artifact_cache: dict[int, tuple[list[int], list[list[int]]]] | None = None,
) -> PolicyV3Batch:
    """Collate TrainingRows into a policy_v3 batch (tokens, masks, labels).

    ``artifact_cache`` (from ``precompute_row_artifacts``) skips per-epoch re-encoding —
    the biggest CPU cost that otherwise starves the GPU.
    """
    from .automaton import IllegalToken

    sequences: list[list[int]] = []
    masks_per_row: list[list[list[int]]] = []
    canonicals: list[str] = []
    kept_rows: list = []
    for row in rows:
        cached = artifact_cache.get(id(row)) if artifact_cache is not None else None
        if cached is not None:
            seq, masks = cached
        else:
            try:
                seq, masks = encode_row_artifacts(row, vocab)
            except (IllegalToken, ValueError):
                vocab.counters["rows_skipped_encode"] = vocab.counters.get("rows_skipped_encode", 0) + 1
                continue
        sequences.append(seq)
        masks_per_row.append(masks)
        canonicals.append(row.canonical_genome or row.genome)
        kept_rows.append(row)
    if not sequences:
        raise ValueError("no encodable rows in batch")
    rows = kept_rows
    task_types = [
        str((row.metadata or {}).get("task_type") or "regression") for row in rows
    ]
    if any(task not in TASK_LANE for task in task_types):
        raise ValueError(
            "policy rows must use a known task_type "
            f"({'/'.join(TASK_LANE)}); unknown values are never coerced"
        )
    token_ids, pad_mask = pad_sequences(sequences, device=device)
    B, T = token_ids.shape

    # Vectorized legal-mask scatter: gather all (batch, time, token) triples, one index op,
    # built on CPU then moved — replaces the per-position Python double loop.
    bi: list[int] = []
    ti: list[int] = []
    vi: list[int] = []
    for i, masks in enumerate(masks_per_row):
        for t, ids in enumerate(masks):
            n = len(ids)
            if n:
                bi.extend([i] * n)
                ti.extend([t] * n)
                vi.extend(ids)
    legal = torch.zeros((B, T - 1, vocab.size), dtype=torch.bool)
    if vi:
        legal[torch.tensor(bi), torch.tensor(ti), torch.tensor(vi)] = True
    legal = legal.to(device, non_blocking=True)

    def _finite(value) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if math.isfinite(out) else float("nan")

    robust_delta = torch.tensor(
        [_finite((row.metadata or {}).get("oof_delta_to_incumbent")) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    runtime = torch.tensor(
        [_finite(row.cost_ms) for row in rows], dtype=torch.float32, device=device
    )

    def _bad(row, task: str) -> float:
        """1.0 when a candidate is worse than its task's trivial baseline.

        Every branch must be explicit about which metric it is reading, because the
        threshold that means "worse than baseline" is different for each. Previously
        only binary was special-cased and everything else fell through to the r2 test
        ``value < 0.0`` against ``row.score``:

          * timeseries scores are ``negative_mase`` (= -MASE), which is negative for
            essentially every candidate, so the whole task was labelled bad;
          * multiclass scores are ``macro_f1`` in [0, 1], which is never negative, so
            the whole task was labelled good.

        Either way the model got a constant label and could learn nothing from the
        task while the pipeline looked healthy.
        """
        meta = row.metadata or {}
        if meta.get("eval_error"):
            return 1.0
        if task == "binary":
            value = _finite(meta.get("oof_auc", meta.get("score_robust", row.score)))
            return float("nan") if math.isnan(value) else 0.0
        if task == "multiclass":
            # macro_f1 in [0, 1]. Like AUC it has no negative region, so a finite
            # score is not evidence of failure on its own.
            value = _finite(meta.get("oof_macro_f1", row.score))
            return float("nan") if math.isnan(value) else 0.0
        if task == "timeseries":
            # negative_mase = -MASE, so parity with the seasonal-naive baseline is
            # -1.0 and anything below it is genuinely worse than doing nothing.
            value = _finite(meta.get("oof_negative_mase", row.score))
            if math.isnan(value):
                return float("nan")
            return 1.0 if value < -1.0 else 0.0
        raw = meta.get("oof_r2_raw", row.score)
        value = _finite(raw)
        if math.isnan(value):
            return float("nan")
        return 1.0 if value < 0.0 else 0.0

    bad_label = torch.tensor(
        [_bad(row, task) for row, task in zip(rows, task_types)],
        dtype=torch.float32,
        device=device,
    )
    reward_pair_ids = [
        str((row.metadata or {}).get("reward_pair_identity") or row.dataset_id)
        for row in rows
    ]
    if precomputed_rwr is not None:
        weight_list = [precomputed_rwr.get(id(row), 1.0) for row in rows]
    elif any((row.metadata or {}).get("rwr_weight_global") is not None for row in rows):
        # global RWR weights were precomputed corpus-wide (prep_v31) but never used —
        # batch-local recomputation is noisier; prefer the stamped weights.
        weight_list = [float((row.metadata or {}).get("rwr_weight_global") or 1.0) for row in rows]
    else:
        weight_list = rwr_weights(
            [
                f"{task}:{identity}"
                for task, identity in zip(task_types, reward_pair_ids)
            ],
            [
                None if not math.isfinite(_finite((row.metadata or {}).get("score_robust", row.score)))
                else _finite((row.metadata or {}).get("score_robust", row.score))
                for row in rows
            ],
            [str((row.metadata or {}).get("provenance") or "gp_report") for row in rows],
        )
    rwr_weight = torch.tensor(weight_list, dtype=torch.float32, device=device)
    bootstrap = torch.tensor(
        [
            [
                bootstrap_mask(
                    f"{task}:{identity}", row.engine, canonical, k
                )
                for k in range(reward_k)
            ]
            for row, canonical, task, identity in zip(
                rows, canonicals, task_types, reward_pair_ids
            )
        ],
        dtype=torch.bool,
        device=device,
    )
    # v3.1 self-play labels (absent → None; losses fall back to legacy targets)
    rank_pct_vals = [_finite((row.metadata or {}).get("rank_pct")) for row in rows]
    has_rank = any(math.isfinite(v) for v in rank_pct_vals)
    is_self_vals = [
        str((row.metadata or {}).get("provenance", "")).startswith("self")
        or str((row.metadata or {}).get("provenance", "")) == "self"
        for row in rows
    ]
    former_best_vals = [_finite((row.metadata or {}).get("former_best_score")) for row in rows]

    return PolicyV3Batch(
        embedding=torch.tensor([row.embedding for row in rows], dtype=torch.float32, device=device),
        target_summary=torch.tensor(
            [
                (row.target_embedding or [0.0] * 512)[:512] + [0.0] * max(0, 512 - len(row.target_embedding or []))
                for row in rows
            ],
            dtype=torch.float32,
            device=device,
        ),
        dataset_ids=[str(row.dataset_id) for row in rows],
        reward_pair_ids=reward_pair_ids,
        engines=[row.engine for row in rows],
        task_types=task_types,
        token_ids=token_ids,
        pad_mask=pad_mask,
        legal=legal,
        robust_delta=robust_delta,
        rwr_weight=rwr_weight,
        runtime=runtime,
        bad_label=bad_label,
        bootstrap=bootstrap,
        canonicals=canonicals,
        rank_pct=(
            torch.tensor(rank_pct_vals, dtype=torch.float32, device=device) if has_rank else None
        ),
        is_self=(
            torch.tensor(is_self_vals, dtype=torch.bool, device=device)
            if any(is_self_vals)
            else None
        ),
        former_best=(
            torch.tensor(former_best_vals, dtype=torch.float32, device=device)
            if any(math.isfinite(v) for v in former_best_vals)
            else None
        ),
        routing_label=torch.tensor(
            [int((row.metadata or {}).get("routing_class", -1)) for row in rows],
            dtype=torch.int64,
            device=device,
        ),
        seq_weight=torch.tensor(
            [float((row.metadata or {}).get("seq_weight", 1.0)) for row in rows],
            dtype=torch.float32,
            device=device,
        ),
    )


def _validate_action_vocab_compatibility(
    saved_tokens: list[str] | None,
    current_tokens: list[str],
    embedding_rows: int,
) -> None:
    """Accept only an exact append-only extension that fits the frozen table."""

    if not saved_tokens or saved_tokens == current_tokens:
        return
    if current_tokens[: len(saved_tokens)] != saved_tokens:
        raise ValueError(
            "action vocab drift: checkpoint tokens are not an exact current prefix"
        )
    if embedding_rows < len(current_tokens):
        raise ValueError(
            "action vocab extension exceeds the checkpoint embedding table"
        )


def load_policy_v3(state: dict[str, Any]) -> JopePolicyModelV3:
    """Rebuild a policy_v3 model from a checkpoint payload."""
    from ..grammars.spec import GrammarSpec
    from .tokens import build_action_vocab, build_prime_action_vocab

    specs_json = state.get("grammar_specs")
    if specs_json:
        specs = [GrammarSpec.from_json(s) for s in specs_json]
        vocab = (
            build_prime_action_vocab()
            if any(spec.engine == "prime" for spec in specs)
            else build_action_vocab(specs)
        )
    else:
        raise ValueError("policy_v3 checkpoint missing grammar_specs")
    saved_tokens = (state.get("action_vocab") or {}).get("tokens")
    _validate_action_vocab_compatibility(
        saved_tokens,
        vocab.tokens,
        int(state.get("embedding_rows") or 0),
    )
    # Task-head schema: v1 (2 tasks, identity lanes) and v2 (4 tasks, TASK_LANE
    # aliases) rebuild the SAME physical architecture — lanes are runtime routing,
    # not tensors — so both load; anything else is from a future writer and must
    # not be silently reinterpreted.
    task_head_schema = state.get("task_head_schema") or {}
    schema_name = str(task_head_schema.get("schema") or "PrimeTaskHeads.v1")
    if schema_name not in {"PrimeTaskHeads.v1", "PrimeTaskHeads.v2"}:
        raise ValueError(
            f"unsupported task_head_schema {schema_name!r}; "
            "this build understands PrimeTaskHeads.v1 (identity lanes) and "
            "PrimeTaskHeads.v2 (timeseries/multiclass lane aliases)"
        )
    model = JopePolicyModelV3(
        vocab,
        dataset_dim=int(state.get("dataset_dim", 512)),
        target_dim=int(state.get("target_dim", 512)),
        d_model=int(state.get("d_model", 256)),
        n_layers=int(state.get("n_layers", 2)),
        n_heads=int(state.get("n_heads", 4)),
        ffn_dim=int(state.get("ffn_dim", 1024)),
        reward_heads=int(state.get("reward_ensemble_k", 4)),
        embedding_rows=int(state.get("embedding_rows") or 0) or None,
        routing_classes=int(state.get("routing_classes", 0)),
        fused_task_heads=bool(state.get("fused_task_heads", False)),
        # v3.4 keys default to the v3.3 architecture so historical checkpoints rebuild
        # exactly as they were written and load 152/152 tensors.
        target_keep_dim=state.get("target_keep_dim"),
        split_reward_context=bool(state.get("split_reward_context", False)),
        input_block_norm=bool(state.get("input_block_norm", False)),
        dataset_block_rms=float(state.get("dataset_block_rms", DATASET_BLOCK_RMS)),
        target_block_rms=float(state.get("target_block_rms", TARGET_BLOCK_RMS)),
    )
    # strict=False: checkpoints predating the stability head still load (its params
    # stay at init and retrain on the next fit). v3.1 changed reward/stability head
    # SHAPES (conditional 3d inputs) — shape-mismatched tensors from older checkpoints
    # are skipped (those heads re-init and retrain) instead of crashing the load.
    current = model.state_dict()
    filtered = {
        k: v for k, v in state["model"].items()
        if k in current and current[k].shape == v.shape
    }
    # A silently-skipped CONTEXT tensor is the "lobotomy" failure class: the network still
    # loads, PyTorch reports nothing, and the dataset conditioning is random for the rest of
    # the run. Heads may re-init; the two context encoders may not.
    context_missing = sorted(
        key
        for key in current
        if key.split(".")[0] in {"encoder", "reward_encoder", "ctx_proj"}
        and key not in filtered
    )
    if context_missing:
        raise ValueError(
            "policy_v3 checkpoint cannot supply the context encoder(s) at the rebuilt "
            f"shape; would re-randomise {context_missing}. Migrate the checkpoint instead "
            "of loading it (see train_v3.migrate_context_state)."
        )
    model.load_state_dict(filtered, strict=False)
    model._grammar_specs = specs  # for checkpoint round-trips (loop refits)
    model.eval()
    return model
