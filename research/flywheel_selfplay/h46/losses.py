"""Loss-weight helpers for the offline H46 training arms."""

from collections import Counter

import numpy as np


def family_balanced_weights(family_ids):
    """Return mean-one weights so every represented family has equal total weight."""
    if not family_ids:
        raise ValueError("at least one family is required")
    counts = Counter(family_ids)
    weights = np.asarray(
        [1.0 / counts[family] for family in family_ids], dtype=np.float64
    )
    return weights * (len(family_ids) / weights.sum())
