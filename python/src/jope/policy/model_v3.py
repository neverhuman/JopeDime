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


from .model_v3_arch import (
    DATASET_BLOCK_RMS,
    DEFAULT_TARGET_KEEP_DIM,
    LEGACY_MODEL_VERSION,
    MAX_SEQ_LEN,
    MODEL_VERSION,
    TARGET_BLOCK_RMS,
    TARGET_TAIL_CHECK_BATCHES,
    TASK_LANE,
    JopePolicyModelV3,
    PolicyV3Batch,
    _task_lane_mask,
)
from .model_v3_batch import (
    encode_row_artifacts,
    pad_sequences,
    precompute_row_artifacts,
    row_artifact_key,
    rows_to_policy_v3_batch,
)
from .model_v3_checkpoint import _validate_action_vocab_compatibility, load_policy_v3
