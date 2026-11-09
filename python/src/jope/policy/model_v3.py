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

from ..model import _reward_listwise_loss as _reward_listwise_loss
from .automaton import replay_masks as replay_masks
from .losses import (
    LISTWISE_TARGET_TAU as LISTWISE_TARGET_TAU,
)
from .losses import (
    PHASE_A_PROFILE as PHASE_A_PROFILE,
)
from .losses import (
    PHASE_B_PROFILE as PHASE_B_PROFILE,
)
from .losses import (
    POLICY_V3_LOSS_COMPONENTS as POLICY_V3_LOSS_COMPONENTS,
)
from .losses import (
    STABILITY_POS_WEIGHT as STABILITY_POS_WEIGHT,
)
from .losses import (
    bootstrap_mask as bootstrap_mask,
)
from .losses import (
    kg_combine as kg_combine,
)
from .losses import (
    masked_sequence_nll as masked_sequence_nll,
)
from .losses import (
    rank_pct_listwise_terms as rank_pct_listwise_terms,
)
from .losses import (
    rwr_weights as rwr_weights,
)
from .losses import (
    self_play_bt_loss as self_play_bt_loss,
)
from .losses import (
    stability_bce_term as stability_bce_term,
)
from .model_v3_arch import (
    DATASET_BLOCK_RMS as DATASET_BLOCK_RMS,
)
from .model_v3_arch import (
    DEFAULT_TARGET_KEEP_DIM as DEFAULT_TARGET_KEEP_DIM,
)
from .model_v3_arch import (
    LEGACY_MODEL_VERSION as LEGACY_MODEL_VERSION,
)
from .model_v3_arch import (
    MAX_SEQ_LEN as MAX_SEQ_LEN,
)
from .model_v3_arch import (
    MODEL_VERSION as MODEL_VERSION,
)
from .model_v3_arch import (
    TARGET_BLOCK_RMS as TARGET_BLOCK_RMS,
)
from .model_v3_arch import (
    TARGET_TAIL_CHECK_BATCHES as TARGET_TAIL_CHECK_BATCHES,
)
from .model_v3_arch import (
    TASK_LANE as TASK_LANE,
)
from .model_v3_arch import (
    JopePolicyModelV3 as JopePolicyModelV3,
)
from .model_v3_arch import (
    PolicyV3Batch as PolicyV3Batch,
)
from .model_v3_arch import (
    _task_lane_mask as _task_lane_mask,
)
from .model_v3_batch import (
    encode_row_artifacts as encode_row_artifacts,
)
from .model_v3_batch import (
    pad_sequences as pad_sequences,
)
from .model_v3_batch import (
    precompute_row_artifacts as precompute_row_artifacts,
)
from .model_v3_batch import (
    row_artifact_key as row_artifact_key,
)
from .model_v3_batch import (
    rows_to_policy_v3_batch as rows_to_policy_v3_batch,
)
from .model_v3_checkpoint import (
    _validate_action_vocab_compatibility as _validate_action_vocab_compatibility,
)
from .model_v3_checkpoint import (
    load_policy_v3 as load_policy_v3,
)
from .tokens import PAD as PAD
from .tokens import ActionVocab as ActionVocab
from .tokens import encode_actions as encode_actions
