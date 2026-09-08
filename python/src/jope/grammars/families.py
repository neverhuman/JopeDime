"""Unified 21-key family taxonomy shared by both engine bridges.

Mirrored as ``FAMILY_KEYS`` consts in rust/jope-validate and rust/jope-newmath-validate;
the grammar handshake test asserts all three stay identical.
"""

from __future__ import annotations

FAMILY_LIST_V3 = [
    "scaling",
    "rank",
    "basis",
    "random_features",
    "interaction",
    "target_aware",
    "rowwise",
    "spectral",
    "kernel",
    "tree",
    "linear",
    "ensemble",
    "symbolic",
    "wavelet",
    "chaos",
    "topology",
    "projection",
    "geometry",
    "reduction",
    "neighbors",
    "robust",
]
