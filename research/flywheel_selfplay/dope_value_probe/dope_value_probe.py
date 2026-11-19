"""Layout, contrast, and pure-Python statistical contract for the DOPE value probe.

This contract module stays strictly off the product path (zero imports beyond
``__future__``). Scored execution lives in ``../h470_dope_value_probe/``;
sealed and holdout rows remain unread. Product crates stay unchanged.
"""
from __future__ import annotations

DOPE629_DIM = 629
DOPE512_PREFIX = 512
LEARNED_TAIL = DOPE629_DIM - 256
POOLED_COL_LM = 34  # mean and max of 17 column landmarks
PACKET_GLOBAL_V1_DIM = 28
PACKET_V1_DIM = 196
PACKET_GLOBAL_SCHEMA = "dope.packet-global/v1"
PACKET_FEATURES_SCHEMA = "dope.packet-features/v1"
ORACLE_BASKET_K = 64
R_ORACLE64_WIDTH = ORACLE_BASKET_K * 2  # 64 centered fitness values + 64 observed/gate masks

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

# (name, start, end) of the scale-invariant, extrema-free 196D R_packet_v1 vector.
PACKET_V1_SLICES: tuple[tuple[str, int, int], ...] = (
    ("gates14", 0, 14),
    ("task_residual_globals14", 14, 28),
    ("copula_regime_basis16", 28, 44),
    ("col_pool_sym96", 44, 140),
    ("col_pool_activity32", 140, 172),
    ("tree_pair_latent24", 172, 196),
)

# Fine landmarks masked at level high. Indices match the 23-name landmark list.
FINE_LANDMARKS: tuple[tuple[str, int], ...] = (
    ("gbdt", 17),
    ("knn1", 18),
    ("knn16", 19),
    ("net", 20),
    ("gbdt_quarter", 22),
)

# Fixed feature widths. R_packet is the legacy global token dope.packet-global/v0 (24);
# R_packet_v1 is the full 196D scale-invariant, extrema-free pooled packet encoding.
ARM_WIDTHS: dict[str, int] = {
    "R_none": 0,
    "R_shape": 3,
    "lm48_z": 48,
    "R_lm80": 80,
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
    "R_packet_global_v1": PACKET_GLOBAL_V1_DIM,
    "R_packet_v1": PACKET_V1_DIM,
    "R_packet_notarget": PACKET_V1_DIM,
    "R_dope_plus_packet_v1": DOPE629_DIM + POOLED_COL_LM + PACKET_V1_DIM,
    "B03_svd96": 96,
}

# Outcome ceilings (evaluated on split-pool B or basket-excluded slates) and zero-retraining upgrades.
CEILING_ARMS: tuple[str, ...] = ("profile_oracle", "R_oracle64")
UPGRADE_ARMS: tuple[str, ...] = ("WP3_pred_reveal", "B03_svd96", "B03_ultra_packet")

# (id, arm, reference). Contrasts C0-C8 covering ceiling, router, descriptor,
# placebos, Stage-1B SVD condensation, Ultra unmasking, Packet v1, and WP3/B03 upgrades.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C0", "profile_oracle", "R_none"),
    ("C0", "R_oracle64", "R_dope"),
    ("C1", "R_dope", "R_desc"),
    ("C2", "R_dope", "R_randrouter"),
    ("C3", "R_desc", "R_shape"),
    ("C3", "R_lm80", "lm48_z"),
    ("C4", "R_shuffled", "R_none"),
    ("C4", "R_noise", "R_none"),
    ("C5", "R_dope", "R_tok"),
    ("C6", "R_dope_ultra", "R_dope"),
    ("C7", "R_dope_plus_packet_v1", "R_dope"),
    ("C7", "R_packet_v1", "R_packet_notarget"),
    ("C8", "WP3_pred_reveal", "incumbent_0rev"),
    ("C8", "B03_svd96", "pairnf_z40"),
)

PLACEBO_CONTRASTS: tuple[str, ...] = ("C4",)

# Cited from crates/jope-encoders and dope_freeze_guard. This module does not change product crates.
DOPE512_RECIPE = "dope-mlcv3-high-f32-descriptor256-task128-dist128/v1"
DOPE_WEIGHTS_SHA256 = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c"
DOPE_FUSED_COMPOSITE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"

SCORED_WAVE = "H470"
NOT_RUN: tuple[str, ...] = (
    "holdout_read",
    "sealed_rows",
)

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _is_finite_number(val: object) -> float:
    if isinstance(val, bool):
        raise ValueError("boolean values are not valid numeric measurements")
    try:
        num = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("numeric measurement required") from None
    if num != num or num == float("inf") or num == float("-inf"):
        raise ValueError("finite numeric measurement required")
    return num


def _split_sliced_vector(
    values: object,
    expected_dim: int,
    slices: tuple[tuple[str, int, int], ...],
    label: str,
) -> dict[str, tuple[float, ...]]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} requires length {expected_dim}")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(f"{label} requires length {expected_dim}") from None
    if len(items) != expected_dim:
        raise ValueError(f"{label} requires length {expected_dim}")
    try:
        numbers = tuple(_is_finite_number(item) for item in items)
    except ValueError:
        raise ValueError(f"{label} requires finite numeric coordinates") from None
    return {name: numbers[start:end] for name, start, end in slices}


def split_dope629(values: object) -> dict[str, tuple[float, ...]]:
    """Split one 629-vector into the released blocks."""
    return _split_sliced_vector(values, DOPE629_DIM, DOPE629_SLICES, "dope629")


def split_packet_v1(values: object) -> dict[str, tuple[float, ...]]:
    """Split one 196-vector R_packet_v1 into its six scale-invariant blocks."""
    return _split_sliced_vector(values, PACKET_V1_DIM, PACKET_V1_SLICES, "R_packet_v1")


def _collect_family_deltas(left, right, families) -> tuple[list[tuple[object, list[float]]], int]:
    if (
        isinstance(left, (str, bytes))
        or isinstance(right, (str, bytes))
        or isinstance(families, (str, bytes))
    ):
        raise ValueError("paired family delta requires equal nonempty lengths")
    try:
        pairs = list(zip(left, right, families, strict=True))
    except (ValueError, TypeError):
        raise ValueError("paired family delta requires equal nonempty lengths") from None
    if not pairs:
        raise ValueError("paired family delta requires equal nonempty lengths")
    buckets: dict[object, list[float]] = {}
    for a, b, family in pairs:
        try:
            delta = _is_finite_number(a) - _is_finite_number(b)
        except ValueError:
            raise ValueError("paired family delta requires numbers") from None
        buckets.setdefault(family, []).append(delta)
    ordered = sorted(buckets.items(), key=lambda kv: (type(kv[0]).__qualname__, str(kv[0])))
    return ordered, len(pairs)


def paired_family_mean_delta(left, right, families) -> float:
    """Unweighted mean of per-family mean paired deltas.

    ``left``, ``right`` and ``families`` must be the same nonempty length.
    """
    ordered, _ = _collect_family_deltas(left, right, families)
    family_means = [sum(deltas) / len(deltas) for _, deltas in ordered]
    return sum(family_means) / len(family_means)


def _splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return state, z


def _rand_below(state: int, upper: int) -> tuple[int, int]:
    """Unbiased integer draw in [0, upper) from 64-bit SplitMix64."""
    threshold = ((1 << 64) - upper) % upper
    while True:
        state, z = _splitmix64_next(state)
        if z >= threshold:
            return state, z % upper


def _quantile_sorted(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def paired_family_bootstrap_ci(
    left,
    right,
    families,
    *,
    n_boot: int = 10000,
    seed: int = 20261118,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Deterministic pure-Python paired family bootstrap over per-family mean deltas.

    Resamples families with replacement ``n_boot`` times using a platform-invariant
    64-bit SplitMix64 PRNG so results are bit-reproducible without external libraries.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    alpha_f = _is_finite_number(alpha)
    if not (0.0 < alpha_f < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")

    ordered, n_contexts = _collect_family_deltas(left, right, families)
    family_means = [sum(deltas) / len(deltas) for _, deltas in ordered]
    n_fams = len(family_means)
    mean_delta = sum(family_means) / n_fams

    fam_wins = sum(1 for d in family_means if d > 0.0)
    fam_ties = sum(1 for d in family_means if d == 0.0)
    fam_losses = sum(1 for d in family_means if d < 0.0)

    state = seed & _MASK64
    boot_means: list[float] = []
    inv_fams = 1.0 / n_fams
    for _ in range(n_boot):
        acc = 0.0
        for _ in range(n_fams):
            state, idx = _rand_below(state, n_fams)
            acc += family_means[idx]
        boot_means.append(acc * inv_fams)

    boot_means.sort()
    ci_low = _quantile_sorted(boot_means, 0.5 * alpha_f)
    ci_high = _quantile_sorted(boot_means, 1.0 - 0.5 * alpha_f)

    le_zero = sum(1 for v in boot_means if v <= 0.0)
    ge_zero = sum(1 for v in boot_means if v >= 0.0)
    p_one = min(le_zero + 1, ge_zero + 1) / (n_boot + 1)
    p_two = min(1.0, 2.0 * p_one)

    return {
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_two,
        "n_families": n_fams,
        "n_contexts": n_contexts,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha_f,
        "family_wins": fam_wins,
        "family_ties": fam_ties,
        "family_losses": fam_losses,
        "family_harm_rate": fam_losses * inv_fams,
    }


def holm_bonferroni(p_values):
    """Step-down Holm-Bonferroni multiple-testing correction in pure Python.

    Accepts either a sequence of p-values (returning a ``tuple[float, ...]``) or a
    ``dict`` mapping contrast names to p-values (returning a ``dict`` with identical keys).
    """
    if isinstance(p_values, (str, bytes)):
        raise ValueError("p_values must be a sequence or mapping of probabilities")
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        adjusted = holm_bonferroni([p_values[k] for k in keys])
        return dict(zip(keys, adjusted, strict=True))
    try:
        raw_items = list(p_values)
    except TypeError:
        raise ValueError("p_values must be a sequence or mapping of probabilities") from None

    probs: list[float] = []
    for item in raw_items:
        p = _is_finite_number(item)
        if p < 0.0 or p > 1.0:
            raise ValueError("p_values must lie in [0.0, 1.0]")
        probs.append(p)

    m = len(probs)
    if m == 0:
        return ()

    order = sorted(range(m), key=lambda idx: (probs[idx], idx))
    out = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * probs[idx]
        if scaled > 1.0:
            scaled = 1.0
        if scaled < running_max:
            scaled = running_max
        else:
            running_max = scaled
        out[idx] = scaled
    return tuple(out)


def placebo_interval_includes_zero(low: float, high: float) -> bool:
    """C4 placebos must include 0."""
    lo = _is_finite_number(low)
    hi = _is_finite_number(high)
    if lo > hi:
        raise ValueError("interval requires low <= high")
    return lo <= 0.0 <= hi
