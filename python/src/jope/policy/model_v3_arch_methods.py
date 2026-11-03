"""Training and checkpoint methods for the v3 policy architecture."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from ..model import _reward_listwise_loss

if TYPE_CHECKING:
    from .model_v3_arch import PolicyV3Batch

from .losses import (
    LISTWISE_TARGET_TAU,
    PHASE_A_PROFILE,
    PHASE_B_PROFILE,
    STABILITY_POS_WEIGHT,
    kg_combine,
    masked_sequence_nll,
    rank_pct_listwise_terms,
    self_play_bt_loss,
    stability_bce_term,
)


class PolicyV3ModelMethods:
    def loss_components(self, batch: PolicyV3Batch, *, phase: str = "B") -> dict[str, torch.Tensor]:
        from .model_v3_arch import _task_lane_mask

        embedding = batch.embedding
        if self.training and self.training_embedding_jitter > 0.0 and embedding.shape[0] > 1:
            scale = embedding.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            embedding = embedding + (self.training_embedding_jitter * scale * torch.randn_like(embedding))
        z = self.context(embedding, batch.target_summary)
        # v3.4: every reward-side consumer reads its own encoder. On a v3.3 checkpoint
        # reward_context() returns the same encoder, so behaviour is unchanged.
        z_reward = self.reward_context(embedding, batch.target_summary)
        components: dict[str, torch.Tensor] = {}

        if self.routing_head is not None and batch.routing_label is not None:
            binary_mask = _task_lane_mask(batch.task_types, z_reward.shape[0], z_reward.device)
            routing_terms: list[torch.Tensor] = []
            for task_mask, head in (
                (~binary_mask, self.routing_head),
                (binary_mask, self.binary_routing_head),
            ):
                mask = task_mask & (batch.routing_label >= 0)
                if head is not None and mask.sum() >= 2:
                    routing_terms.append(F.cross_entropy(head(z_reward[mask]), batch.routing_label[mask]))
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
            rewards = self.reward_all(z_reward, batch.token_ids, batch.pad_mask, batch.task_types)  # (B, K)
            keep_all = finite
            task_reward_pair_ids = [
                f"{task}:{identity}" for task, identity in zip(batch.task_types, batch.reward_pair_ids)
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
                    pair = rank_pct_listwise_terms(rewards[keep, k], batch.rank_pct[keep], ids)
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
                abl_bt = self_play_bt_loss(
                    abl, batch.robust_delta[keep_all], ids_all, rank_pct=rank_pct, is_self=is_self
                )
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
                binary_mask = _task_lane_mask(batch.task_types, z_reward.shape[0], z_reward.device)
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
                    components["stability_bce_baseline"] = torch.stack(stability_baselines).mean()
        return components

    def loss(self, batch: PolicyV3Batch, *, phase: str = "B") -> torch.Tensor:
        profile = PHASE_A_PROFILE if phase == "A" else PHASE_B_PROFILE
        return kg_combine(self.loss_components(batch, phase=phase), self.loss_log_vars, profile)

    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        from .model_v3_arch import MODEL_VERSION, TASK_LANE

        return {
            "model_version": MODEL_VERSION,
            "model": self.state_dict(),
            "action_vocab": {"tokens": self.vocab.tokens, "version": self.vocab.version},
            **(
                {"frozen_action_vocab": copy.deepcopy(self.frozen_action_vocab)}
                if self.frozen_action_vocab is not None
                else {}
            ),
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
                "lanes": (dict(TASK_LANE) if self.fused_task_heads else {"regression": "regression"}),
                "aliased_tasks": (["timeseries", "multiclass"] if self.fused_task_heads else []),
                "reward_ensemble_per_task": self.reward_k,
                "separate_stability": self.fused_task_heads,
                "separate_routing": self.fused_task_heads and self.routing_classes > 0,
            },
            **extra,
        }
