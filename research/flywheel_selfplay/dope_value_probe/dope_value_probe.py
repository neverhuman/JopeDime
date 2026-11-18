"""Unscored layout contract for a future DOPE value probe.

The scored probe is NOT_RUN. This module does not load weights, import a
ranker, or read sealed or val91 rows. Product crates stay unchanged.
"""
from __future__ import annotations

DOPE629_DIM = 629
DOPE512_PREFIX = 512
LEARNED_TAIL = DOPE629_DIM - 256
POOLED_COL_LM = 34  # mean and max of 17 column landmarks

# (name, start, end) of the released high-629 vector.
DOPE629_SLICES: tuple[tuple[str, int, int], ...] = (
    ("vec256", 0, 256),
    ("task_latent", 256, 384),
    ("dist_latent", 384, 512),
    ("gates", 512, 526),
    ("landmark_pred", 526, 549),
    ("bb_pred", 549, 565),
    ("col_gates", 565, 629),
)

# Fine landmarks masked at level high. Indices match the 23-name landmark list.
FINE_LANDMARKS: tuple[tuple[str, int], ...] = (
    ("gbdt", 17),
    ("knn1", 18),
    ("knn16", 19),
    ("net", 20),
    ("gbdt_quarter", 22),
)

# Fixed feature widths. R_packet is the global token dope.packet-global/v0 only.
ARM_WIDTHS: dict[str, int] = {
    "R_none": 0,
    "R_shape": 3,
    "R_desc": 256 + POOLED_COL_LM,
    "R_learned": LEARNED_TAIL,
    "R_dope": DOPE629_DIM + POOLED_COL_LM,
    "R_hidden": 256,
    "R_tok": 320,
    "R_randrouter": DOPE629_DIM + POOLED_COL_LM,
    "R_shuffled": DOPE629_DIM + POOLED_COL_LM,
    "R_noise": DOPE629_DIM + POOLED_COL_LM,
    "R_desc_ultra": 256 + POOLED_COL_LM,
    "R_dope_ultra": DOPE629_DIM + POOLED_COL_LM,
    "R_packet": 24,
}

# Outcome ceiling. Not a feature vector.
CEILING_ARMS: tuple[str, ...] = ("profile_oracle",)

# (id, arm, reference). C4 is recorded twice, once per placebo.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C1", "R_dope", "R_desc"),
    ("C2", "R_dope", "R_randrouter"),
    ("C3", "R_desc", "R_shape"),
    ("C4", "R_shuffled", "R_none"),
    ("C4", "R_noise", "R_none"),
    ("C5", "R_dope", "R_tok"),
)

PLACEBO_CONTRASTS: tuple[str, ...] = ("C4",)

# Cited from crates/jope-encoders. This module does not change that crate.
DOPE512_RECIPE = "dope-mlcv3-high-f32-descriptor256-task128-dist128/v1"
DOPE_WEIGHTS_SHA256 = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c"

NOT_RUN: tuple[str, ...] = (
    "scored_probe",
    "holdout_read",
    "sealed_rows",
    "DVP-S1",
    "DVP-S2",
    "U-S2",
    "P-S1",
)


def split_dope629(values: object) -> dict[str, tuple[float, ...]]:
    """Split one 629-vector into the released blocks."""
    if isinstance(values, (str, bytes)):
        raise ValueError("dope629 requires length 629")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError("dope629 requires length 629") from None
    if len(items) != DOPE629_DIM:
        raise ValueError("dope629 requires length 629")
    try:
        numbers = tuple(float(item) for item in items)
    except (TypeError, ValueError):
        raise ValueError("dope629 requires numeric coordinates") from None
    return {name: numbers[start:end] for name, start, end in DOPE629_SLICES}


def paired_family_mean_delta(left, right, families) -> float:
    """Unweighted mean of per-family mean paired deltas.

    ``left``, ``right`` and ``families`` must be the same nonempty length.
    """
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)) or isinstance(families, (str, bytes)):
        raise ValueError("paired family delta requires equal nonempty lengths")
    try:
        pairs = list(zip(left, right, families, strict=True))
    except ValueError:
        raise ValueError("paired family delta requires equal nonempty lengths") from None
    except TypeError:
        raise ValueError("paired family delta requires equal nonempty lengths") from None
    if not pairs:
        raise ValueError("paired family delta requires equal nonempty lengths")
    buckets: dict[object, list[float]] = {}
    for a, b, family in pairs:
        try:
            delta = float(a) - float(b)
        except (TypeError, ValueError):
            raise ValueError("paired family delta requires numbers") from None
        buckets.setdefault(family, []).append(delta)
    family_means = [sum(deltas) / len(deltas) for deltas in buckets.values()]
    return sum(family_means) / len(family_means)


def placebo_interval_includes_zero(low: float, high: float) -> bool:
    """C4 placebos must include 0. This does not score a probe."""
    return float(low) <= 0.0 <= float(high)
