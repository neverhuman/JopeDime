from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data.active import (
    ACTIVE_MAX_TRANSFORMS,
    TARGET_SUMMARY_DIM,
    format_param_token,
    format_pipeline_actions,
    parse_param_token,
    parse_pipeline_actions,
)
from .grammar import HEADS, TRANSFORMS
from .schema import EMBEDDING_DIM
from .model import FAMILY_LIST, HEAD_LIST, NONE_TOKEN, TRANSFORM_LIST


from .active_v2_batch import (
    active_v2_labels_to_genome,
    active_v2_validation_metrics,
    decode_active_v2_output,
    rows_to_active_v2_batch,
)
from .active_v2_losses import (
    _balanced_cross_entropy,
    _decode_params,
    _finite_or_nan,
    _group_by_dataset,
    _labels_to_params,
    _listwise_top_loss,
    _mae,
    _pairwise_accuracy,
    _pearson,
    _ranknet_loss,
    _reward_bradley_terry_loss,
    _reward_listwise_loss,
    _row_actions,
    _row_target_summary,
    _select_index,
    _top1_regret,
)
from .active_v2_model import JopeActiveModelV2
from .active_v2_vocab import (
    ActiveV2Batch,
    _GENOME_SCALAR_FEATURES,
    _normalize_param_vocab,
    _param_key,
    _safe_key,
    active_v2_param_specs,
    genome_feature_dim,
    genome_features_from_actions,
    normalize_active_v2_vocab,
)
