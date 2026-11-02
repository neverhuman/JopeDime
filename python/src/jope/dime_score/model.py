from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model_config import DATASET_DIM, D_MODEL, FFN_DIM, N_HEADS, N_LAYERS
from .model_broad import (
    DimeBroadCritic,
    _broad_groups,
    _dataset_balanced_mean,
    broad_critic_losses,
    broad_partial_warm_start,
)
from .model_legacy import DimeScoreModel, bradley_terry_loss, partial_warm_start, score_losses
